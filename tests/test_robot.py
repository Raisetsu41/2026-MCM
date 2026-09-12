import json
import math
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

import robot.main as robot_main
from robot.q3_agent import Q3Agent, polygon_clear_sites, q3_scan_sites
from robot.client import ApiClient, ApiError
from robot.mock_server import Jammer, MockArena, MockServer
from robot.q4_agent import Q4Agent, bearing_clear_sites, q4_scan_sites

class RobotTest(unittest.TestCase):
  def test_retry_reuses_request_without_double_time(self) -> None:
    arena = MockArena(
      [Jammer(1, 100.0, 0.0, 1000.0)], drop_first_action=True)
    with MockServer(arena) as server:
      client = ApiClient(
        server.url, arena.robot_id, max_retry=2, backoff_s=0.001)
      client.enter()
      body = client.measure(0.0, 0.0, 1)
      client.exit()
    self.assertEqual(body["measure_result"], "direction")
    self.assertAlmostEqual(float(body["virtual_time_s"]), 5.0)
    self.assertEqual(arena.measure_accepted_count, 1)

  def test_same_location_has_fixed_error(self) -> None:
    arena = MockArena([Jammer(1, 100.0, 50.0, 1000.0)])
    with MockServer(arena) as server:
      client = ApiClient(server.url, arena.robot_id)
      client.enter()
      first = client.measure(0.0, 0.0, 1)
      second = client.measure(0.0, 0.0, 1)
      client.exit()
    self.assertEqual(first["svd_deg"], second["svd_deg"])
    self.assertAlmostEqual(float(second["virtual_time_s"]), 10.0)

  def test_deadline_guard_stops_new_action(self) -> None:
    client = ApiClient("http://127.0.0.1:1", "local-mock")
    client.deadline = time.monotonic() + 0.01
    with self.assertRaises(ApiError):
      client.measure(0.0, 0.0, 1)

  def test_client_has_one_authoritative_robot_state(self) -> None:
    client = ApiClient("http://127.0.0.1:1", "local-mock")
    self.assertFalse(hasattr(client, "pos"))
    self.assertFalse(hasattr(client, "current_channel"))

  def test_seven_site_cover_margin(self) -> None:
    sites = q3_scan_sites()
    worst = math.sqrt(
      1800.0 ** 2 + 1200.0 ** 2
      - 2.0 * 1800.0 * 1200.0 * math.cos(math.pi / 6.0))
    self.assertEqual(len(sites), 7)
    self.assertAlmostEqual(worst, 968.9015717, places=6)
    self.assertLess(worst, 1000.0)

  def test_near_is_a_certified_clear(self) -> None:
    arena = MockArena([Jammer(1, 3.0, 0.0, 1000.0)])
    with MockServer(arena) as server:
      result = Q3Agent(ApiClient(server.url, arena.robot_id)).run()
    self.assertTrue(result.complete)
    self.assertEqual(result.cleared, 1)
    self.assertEqual(arena.clear_failure_count, 0)

  def test_q3_batch_beats_immediate_mean_without_weakening(self) -> None:
    immediate_time = []
    batch_time = []
    for seed in range(20260911, 20260916):
      for schedule, values in (
          ("immediate", immediate_time), ("batch", batch_time)):
        arena = MockArena.random_q3(seed, count=10)
        with MockServer(arena) as server:
          result = Q3Agent(
            ApiClient(server.url, arena.robot_id), schedule=schedule).run()
        self.assertTrue(result.complete)
        self.assertEqual(result.cleared, 10)
        self.assertEqual(arena.clear_failure_count, 0)
        self.assertTrue(all(jammer.cleared for jammer in arena.jammers))
        values.append(result.virtual_time_s)
    self.assertLess(float(np.mean(batch_time)), float(np.mean(immediate_time)))

  def test_sixteen_source_upper_bound_shortcut(self) -> None:
    arena = MockArena.random_q3(20260918, count=16)
    with MockServer(arena) as server:
      result = Q3Agent(ApiClient(server.url, arena.robot_id)).run()
    self.assertTrue(result.complete)
    self.assertEqual(result.cleared, 16)
    self.assertEqual(result.absent_certified, 4)
    self.assertLess(arena.measure_accepted_count, 7 * 20 + 16 * 4)

  def test_polygon_fallback_covers_bounding_box(self) -> None:
    polygon = np.array([
      [-10.0, -20.0], [91.0, -20.0], [91.0, 55.0], [-10.0, 55.0],
    ])
    path = polygon_clear_sites(polygon)
    xx, yy = np.meshgrid(
      np.linspace(-10.0, 91.0, 31), np.linspace(-20.0, 55.0, 31))
    source = np.column_stack((xx.ravel(), yy.ravel()))
    nearest = np.min(
      np.linalg.norm(source[:, None, :] - path[None, :, :], axis=2), axis=1)
    self.assertLessEqual(float(np.max(nearest)), 20.0 + 1e-9)

  def test_q4_grid_has_direction_certificate(self) -> None:
    sites = q4_scan_sites()
    self.assertEqual(len(sites), 31)
    rng = np.random.default_rng(20260911)
    for _ in range(5000):
      radius = 1800.0 * math.sqrt(float(rng.random()))
      angle = rng.uniform(0.0, 2.0 * math.pi)
      source = radius * np.array([math.cos(angle), math.sin(angle)])
      heading = rng.uniform(0.0, 2.0 * math.pi)
      direction = np.array([math.cos(heading), math.sin(heading)])
      delta = sites - source
      visible = ((np.linalg.norm(delta, axis=1) <= 1000.0 + 1e-9)
                 & (delta @ direction >= -1e-12))
      self.assertTrue(bool(np.any(visible)))

  def test_q4_bearing_fallback_covers_continuum_grid(self) -> None:
    site = np.array([100.0, -200.0])
    path = bearing_clear_sites(site, 37.0)
    rho = np.linspace(5.0, 1500.0, 301)
    angle = np.radians(37.0 + np.linspace(-1.0, 1.0, 41))
    worst = 0.0
    for value in angle:
      source = site + np.column_stack((rho * np.cos(value), rho * np.sin(value)))
      nearest = np.min(
        np.linalg.norm(source[:, None, :] - path[None, :, :], axis=2), axis=1)
      worst = max(worst, float(np.max(nearest)))
    self.assertLessEqual(worst, 19.9)

  def test_q4_random_mixed_cases_clear_every_source(self) -> None:
    for seed in (41, 42):
      arena = MockArena.random_q4(seed, count=10, directional_rate=0.5)
      with MockServer(arena) as server:
        result = Q4Agent(ApiClient(server.url, arena.robot_id)).run()
      self.assertTrue(result.complete)
      self.assertEqual(result.cleared, 10)
      self.assertTrue(all(jammer.cleared for jammer in arena.jammers))

  def test_command_log_redacts_identity(self) -> None:
    old_path = robot_main.log_path
    with tempfile.TemporaryDirectory() as temp_dir:
      try:
        path = Path(temp_dir) / "actions.jsonl"
        robot_main.log_path = path
        robot_main.log_call(
          "/measure", {"robot_id": "sensitive-value"}, 200,
          {"accepted": True, "robot_id": "sensitive-value"}, 1)
        text = path.read_text(encoding="utf-8")
      finally:
        robot_main.log_path = old_path
    self.assertNotIn("sensitive-value", text)
    row = json.loads(text)
    self.assertEqual(row["request"]["robot_id"], "<redacted>")

if __name__ == "__main__":
  unittest.main(verbosity=2)
