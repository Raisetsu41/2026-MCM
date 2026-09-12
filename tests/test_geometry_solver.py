import math
import unittest

import numpy as np

from geometry_solver import (
  bearing_halfplanes,
  bearing_jacobian,
  continuous_two_site_error_bound,
  detected_candidate_mask,
  error_propagation,
  gdop,
  halfplane_intersection,
  localization_diameter_bounds,
  localization_polygon,
  minimum_enclosing_circle,
  pareto_mask,
  pareto_second_sites_cone,
  polygon_diameter,
  robust_candidate_mask,
  sector_max_distance,
  second_site_metrics_sources,
  source_cone_samples,
  two_site_gdop,
)

class GeometryTest(unittest.TestCase):
  def test_calipers_matches_brute_force_across_scales(self) -> None:
    rng = np.random.default_rng(20260911)
    for scale in (1e-3, 1.0, 2e6):
      for n in range(3, 35):
        points = rng.normal(size=(n, 2)) * scale
        diameter, pair = polygon_diameter(points)
        brute = max(np.linalg.norm(a - b) for a in points for b in points)
        self.assertTrue(math.isclose(diameter, brute, rel_tol=1e-11,
                                     abs_tol=1e-14 * scale))
        self.assertTrue(math.isclose(
          float(np.linalg.norm(pair[0] - pair[1])), diameter,
          rel_tol=1e-11, abs_tol=1e-14 * scale))

  def test_calipers_small_scale_regression(self) -> None:
    points = np.array([
      [0.000126, 0.000254], [0.000528, 0.000043],
      [0.000013, 0.000461], [0.000453, 0.000504],
    ])
    diameter, _ = polygon_diameter(points)
    brute = max(np.linalg.norm(a - b) for a in points for b in points)
    self.assertAlmostEqual(diameter, brute, places=15)

  def test_halfplane_square_and_translation(self) -> None:
    bound = np.array([[-3, -3], [3, -3], [3, 3], [-3, 3]], dtype=float)
    hps = np.array([[1, 0, 1], [-1, 0, 1], [0, 1, 1], [0, -1, 1]], dtype=float)
    base = halfplane_intersection(hps, bound)
    self.assertEqual(len(base), 4)
    offset = np.array([2e6, -2e6])
    moved_hps = hps.copy()
    moved_hps[:, 2] += hps[:, :2] @ offset
    moved = halfplane_intersection(moved_hps, bound + offset)
    np.testing.assert_allclose(moved - offset, base, atol=2e-9)

  def test_bearing_inputs_reject_nonfinite(self) -> None:
    with self.assertRaises(ValueError):
      bearing_halfplanes((0.0, 0.0), math.nan)
    with self.assertRaises(ValueError):
      localization_polygon([[0.0, 0.0]], [math.inf])
    with self.assertRaises(ValueError):
      halfplane_intersection([[math.nan, 0.0, 1.0]], [[0.0, 0.0]])
    with self.assertRaises(ValueError):
      source_cone_samples((0.0, 0.0), 0.0, target_rad=math.nan)
    with self.assertRaises(ValueError):
      robust_candidate_mask(
        [[0.0, 0.0]], (0.0, 0.0), 0.0, measure_s=math.nan)

  def test_single_bearing_is_bounded_by_target_disk(self) -> None:
    poly, _ = localization_polygon([[0.0, 0.0]], [0.0], n_bound=180)
    self.assertGreater(len(poly), 2)
    diameter, _ = polygon_diameter(poly)
    self.assertGreater(diameter, 0.0)
    self.assertLessEqual(diameter, 3601.0)

  def test_parallel_bearings_do_not_silently_become_zero(self) -> None:
    poly, _ = localization_polygon([[0.0, -10.0], [0.0, 10.0]], [0.0, 0.0])
    diameter, _ = polygon_diameter(poly)
    self.assertGreater(diameter, 1000.0)

  def test_localization_diameter_bracket(self) -> None:
    sites = np.array([[-2000.0, 0.0], [0.0, -2000.0]])
    target = np.array([1200.0, 1200.0])
    angle = np.degrees(np.arctan2(
      target[1] - sites[:, 1], target[0] - sites[:, 0])) % 360.0
    result = localization_diameter_bounds(sites, angle, n_bound=180)
    self.assertLessEqual(float(result["lower_m"]), float(result["upper_m"]))

  def test_minimum_circle_translation_and_scale(self) -> None:
    side = 1e-3
    points = np.array([
      [0.0, 0.0], [side, 0.0],
      [side / 2.0, math.sqrt(3.0) * side / 2.0],
    ])
    center, radius = minimum_enclosing_circle(points)
    self.assertAlmostEqual(radius, side / math.sqrt(3.0), places=14)
    offset = np.array([2e6, -2e6])
    moved_center, moved_radius = minimum_enclosing_circle(points + offset)
    np.testing.assert_allclose(moved_center, center + offset, atol=3e-10)
    self.assertTrue(math.isclose(moved_radius, radius, rel_tol=3e-7))

  def test_jacobian_translation_does_not_use_global_scale(self) -> None:
    for shift in (0.0, 2e6):
      target = np.array([shift + 1e-3, shift])
      sites = np.array([[shift, shift], [shift + 1e-3, shift - 1e-3]])
      jacobian = bearing_jacobian(target, sites)
      self.assertTrue(np.all(np.isfinite(jacobian)))

  def test_jacobian_rejects_overlap(self) -> None:
    with self.assertRaises(ValueError):
      bearing_jacobian((0.0, 0.0), [(0.0, 0.0)])

  def test_gdop_closed_form_and_degeneracy(self) -> None:
    target = np.array([0.0, 0.0])
    sites = np.array([[-3.0, 0.0], [0.0, -4.0]])
    self.assertAlmostEqual(gdop(target, sites), 5.0, places=12)
    self.assertAlmostEqual(two_site_gdop(target, sites[0], sites[1]), 5.0,
                           places=12)
    self.assertTrue(math.isinf(two_site_gdop(target, (-3.0, 0.0), (4.0, 0.0))))
    self.assertTrue(math.isinf(gdop(target, [(-3.0, 0.0), (4.0, 0.0)])))

  def test_error_propagation_covariance(self) -> None:
    result = error_propagation((0.0, 0.0), [(-3.0, 0.0), (0.0, -4.0)])
    sigma = math.radians(1.0 / math.sqrt(3.0))
    self.assertAlmostEqual(float(result["rms"]), 5.0 * sigma, places=12)
    np.testing.assert_allclose(result["gain"] @ result["h"], np.eye(2), atol=1e-12)
    with self.assertRaises(ValueError):
      error_propagation((0.0, 0.0), [(-3.0, 0.0), (4.0, 0.0)])

  def test_sector_max_distance_matches_dense_search(self) -> None:
    candidates = np.array([[1103.949, -285.169], [700.0, 500.0]])
    exact = sector_max_distance(candidates, (0.0, 0.0), 25.0)
    rho = np.array([5.0, 1500.0])
    angle = np.radians(np.linspace(24.0, 26.0, 20001))
    source = np.vstack([
      np.column_stack((r * np.cos(angle), r * np.sin(angle))) for r in rho
    ])
    brute = np.max(
      np.linalg.norm(candidates[:, None, :] - source[None, :, :], axis=2),
      axis=1)
    np.testing.assert_allclose(exact, brute, atol=2e-8)

  def test_cone_sampling_keeps_disk_tangent_center_ray(self) -> None:
    samples = source_cone_samples(
      (3300.0, 0.0), 180.0, ranges=(1499.9, 1500.0),
      target_rad=1800.0, n_range=2, n_angle=2)
    self.assertGreater(len(samples), 0)
    self.assertTrue(np.any(np.linalg.norm(samples - [1800.0, 0.0], axis=1) < 1e-8))

  def test_conditional_reception_is_larger_than_strict_core(self) -> None:
    candidate = np.array([[0.0, 0.0]])
    strict = robust_candidate_mask(candidate, (0.0, 0.0), 0.0)
    conditional = detected_candidate_mask(candidate, (0.0, 0.0), 0.0)
    self.assertFalse(bool(strict[0]))
    self.assertTrue(bool(conditional[0]))

  def test_analytic_guarantee_catches_sampling_gap(self) -> None:
    phase = math.radians(-179.95)
    candidate = 500.0001 * np.array([[math.cos(phase), math.sin(phase)]])
    result = pareto_second_sites_cone(
      candidate, (0.0, 0.0), 0.0, err_deg=1.0, recv_rad=2000.0,
      n_range=2, n_angle=2, conditional_reception=False)
    self.assertAlmostEqual(float(result["coverage"][0]), 1.0, places=12)
    self.assertFalse(bool(result["guaranteed"][0]))
    self.assertFalse(bool(result["pareto"][0]))

  def test_wrapper_forwards_error_angle(self) -> None:
    candidate = np.array([[500.0, 500.0]])
    sources = source_cone_samples(
      (0.0, 0.0), 0.0, err_deg=2.0, n_range=3, n_angle=3)
    direct = second_site_metrics_sources(
      candidate, (0.0, 0.0), sources, err_deg=2.0)
    wrapped = pareto_second_sites_cone(
      candidate, (0.0, 0.0), 0.0, err_deg=2.0,
      n_range=3, n_angle=3)
    self.assertAlmostEqual(
      float(wrapped["sampled_error"][0]), float(direct["sampled_error"][0]),
      places=12)

  def test_continuous_error_detects_unsampled_collinearity(self) -> None:
    angle = math.radians(0.05)
    candidate = 500.0 * np.array([math.cos(angle), math.sin(angle)])
    certified = continuous_two_site_error_bound(
      (0.0, 0.0), candidate, 0.0, err_deg=1.0)
    self.assertTrue(math.isinf(float(certified["upper_m"])))

  def test_continuous_error_encloses_dense_search(self) -> None:
    site2 = np.array([993.7385850681706, -143.46994300796996])
    certified = continuous_two_site_error_bound(
      (0.0, 0.0), site2, 25.0, abs_tol_m=1e-3)
    self.assertTrue(bool(certified["certified"]))
    self.assertLessEqual(
      float(certified["upper_m"]) - float(certified["lower_m"]), 1e-3)
    angle = np.radians(np.linspace(24.0, 26.0, 20001))
    sampled = []
    for source_range in (5.0, 1500.0):
      source = source_range * np.column_stack((np.cos(angle), np.sin(angle)))
      result = second_site_metrics_sources(
        site2[None, :], (0.0, 0.0), source, err_deg=1.0)
      sampled.append(float(result["sampled_error"][0]))
    dense = max(sampled)
    self.assertLessEqual(dense, float(certified["upper_m"]))
    self.assertGreaterEqual(dense, float(certified["lower_m"]) - 1e-3)

  def test_pareto_handles_infinite_and_mixed_units(self) -> None:
    values = np.array([
      [1e-9, 400.0], [2e-9, 300.0], [3e-9, 500.0], [math.inf, 1.0],
    ])
    np.testing.assert_array_equal(pareto_mask(values), [True, True, False, False])
    np.testing.assert_array_equal(
      pareto_mask(np.full((2, 2), math.inf)), [False, False])

if __name__ == "__main__":
  unittest.main(verbosity=2)
