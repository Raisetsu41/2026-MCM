# 通信幂等性和 Q3 端到端闭环测试.
import tempfile
import unittest
from pathlib import Path

import json
import math

import numpy as np

from robot.agent import (
  Q3Agent,
  fallback_clear_sites,
  polygon_clear_sites,
  q3_scan_sites,
  q4_scan_sites,
)
from robot.client import ApiClient
from robot.mock_server import Jammer, MockArena, MockServer


class RobotTest(unittest.TestCase):
  def test_retry_reuses_request_without_double_time(self) -> None:
    arena = MockArena(
      [Jammer(1, 100.0, 0.0, 1000.0)], drop_first_action=True)
    lines: list[str] = []
    with MockServer(arena) as server:
      client = ApiClient(
        server.url, arena.robot_id, max_retry=2, backoff_s=0.001)
      client._write = lambda item: lines.append(json.dumps(item))  # type: ignore[method-assign]
      client.enter()
      body = client.measure(0.0, 0.0, 1)
      self.assertEqual(body["measure_result"], "direction")
      # 重试必须复用同一 request_id, 因此虚拟时间不得被重复推进.
      self.assertAlmostEqual(float(body["virtual_time_s"]), 5.0)
      ids = {json.loads(line)["payload"]["request_id"]
             for line in lines if "payload" in json.loads(line)}
      self.assertEqual(len(ids), 2)  # 仅 enter-* 与 measure-* 两个动作
      client.exit()

  def test_same_location_has_fixed_error(self) -> None:
    arena = MockArena([Jammer(1, 100.0, 50.0, 1000.0)])
    with MockServer(arena) as server:
      client = ApiClient(server.url, arena.robot_id)
      client.enter()
      first = client.measure(0.0, 0.0, 1)
      second = client.measure(0.0, 0.0, 1)
      self.assertEqual(first["svd_deg"], second["svd_deg"])
      self.assertAlmostEqual(float(second["virtual_time_s"]), 10.0)
      client.exit()

  def test_seven_site_cover_margin(self) -> None:
    sites = q3_scan_sites()
    self.assertEqual(len(sites), 7)
    worst = (1800.0 ** 2 + 1200.0 ** 2
             - 2.0 * 1800.0 * 1200.0 * (3.0 ** 0.5 / 2.0)) ** 0.5
    self.assertAlmostEqual(worst, 968.9015717, places=6)
    self.assertLess(worst, 1000.0)

  def test_q3_random_case_clears_every_source(self) -> None:
    arena = MockArena.random_q3(20260911, count=10)
    with MockServer(arena) as server:
      result = Q3Agent(ApiClient(server.url, arena.robot_id)).run()
    self.assertTrue(result.complete)
    self.assertEqual(result.cleared, 10)
    self.assertEqual(result.absent_certified, 10)
    self.assertTrue(all(jammer.cleared for jammer in arena.jammers))

  def test_q4_grid_has_direction_certificate(self) -> None:
    sites = q4_scan_sites()
    rng = np.random.default_rng(20260911)
    for _ in range(1000):
      radius = 1800.0 * math.sqrt(float(rng.random()))
      angle = rng.uniform(0.0, 2.0 * math.pi)
      source = radius * np.array([math.cos(angle), math.sin(angle)])
      head = rng.uniform(0.0, 2.0 * math.pi)
      direction = np.array([math.cos(head), math.sin(head)])
      delta = sites - source
      visible = (np.linalg.norm(delta, axis=1) <= 1000.0) & (delta @ direction >= -1e-10)
      self.assertTrue(bool(np.any(visible)))

  def test_q4_fallback_covers_first_bearing_sector(self) -> None:
    site = np.array([100.0, -200.0])
    path = fallback_clear_sites(site, 37.0)
    rho = np.linspace(5.0, 1500.0, 101)
    angle = np.radians(37.0 + np.linspace(-1.0, 1.0, 21))
    for a in angle:
      src = site + np.column_stack((rho * np.cos(a), rho * np.sin(a)))
      near = np.min(np.linalg.norm(src[:, None, :] - path[None, :, :], axis=2), axis=1)
      self.assertLessEqual(float(np.max(near)), 20.0 + 1e-9)

  def test_polygon_fallback_has_finite_cover(self) -> None:
    poly = np.array([[-10.0, -20.0], [91.0, -20.0], [91.0, 55.0], [-10.0, 55.0]])
    path = polygon_clear_sites(poly)
    xx, yy = np.meshgrid(np.linspace(-10.0, 91.0, 31), np.linspace(-20.0, 55.0, 31))
    src = np.column_stack((xx.ravel(), yy.ravel()))
    near = np.min(np.linalg.norm(src[:, None, :] - path[None, :, :], axis=2), axis=1)
    self.assertLessEqual(float(np.max(near)), 20.0 + 1e-9)

  def test_agent_does_not_enter_twice(self) -> None:
    """调用方已进入时, run(enter_body=...) 不得再次 /enter (否则本局被拒)."""
    arena = MockArena([Jammer(1, 100.0, 0.0, 1000.0)])
    with MockServer(arena) as server:
      client = ApiClient(server.url, arena.robot_id)
      body = client.enter_when_open(wait_s=5.0)
      agent = Q3Agent(client)
      result = agent.run(enter_body=body)
    self.assertFalse(agent.entered_here)
    self.assertEqual(arena.enter_calls, 1)
    self.assertTrue(result.complete)
    self.assertEqual(result.cleared, 1)

  def test_duplicate_enter_is_rejected(self) -> None:
    """真实模拟器会拒绝重复 /enter, Mock 必须复现这一行为."""
    arena = MockArena([Jammer(1, 100.0, 0.0, 1000.0)])
    with MockServer(arena) as server:
      client = ApiClient(server.url, arena.robot_id)
      client.enter_when_open(wait_s=5.0)
      with self.assertRaises(Exception):
        client.enter()
      self.assertGreaterEqual(arena.enter_calls, 2)
      client.exit()

  def test_measurement_cap_uses_clear_fallback(self) -> None:
    arena = MockArena([Jammer(1, 850.0, 430.0, 1000.0)])
    with MockServer(arena) as server:
      result = Q3Agent(ApiClient(server.url, arena.robot_id), max_obs=1).run()
    self.assertTrue(result.complete)
    self.assertEqual(result.cleared, 1)


if __name__ == "__main__":
  unittest.main(verbosity=2)
