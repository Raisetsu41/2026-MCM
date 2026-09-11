# 几何算法独立单元测试.
# 解析解, 暴力解和有限差分交叉核验.
import math
import unittest

import numpy as np

from geometry_solver import (
  bearing_halfplanes,
  bearing_jacobian,
  candidate_region_mask,
  convex_hull,
  diameter_circle_coverage,
  error_propagation,
  gdop,
  halfplane_intersection,
  intersection_angle,
  localization_diameter_bounds,
  localization_polygon,
  minimum_enclosing_circle,
  pareto_mask,
  polygon_diameter,
  robust_candidate_mask,
  sector_max_distance,
  second_site_metrics_sources,
  source_cone_samples,
  two_site_gdop,
)


class GeometryTest(unittest.TestCase):
  def test_convex_hull_and_square_diameter(self) -> None:
    p = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5], [1, 1]])
    h = convex_hull(p)
    d, pair = polygon_diameter(p)
    self.assertEqual(len(h), 4)
    self.assertAlmostEqual(d, math.sqrt(2.0), places=12)
    self.assertAlmostEqual(float(np.linalg.norm(pair[0] - pair[1])), d, places=12)

  def test_calipers_matches_brute_force(self) -> None:
    rng = np.random.default_rng(20260910)
    for n in range(3, 50):
      for _ in range(10):
        p = rng.normal(size=(n, 2))
        d, _ = polygon_diameter(p)
        brute = max(np.linalg.norm(a - b) for a in p for b in p)
        self.assertAlmostEqual(d, brute, places=10)

  def test_halfplane_square(self) -> None:
    hp = np.array([[1, 0, 1], [-1, 0, 1], [0, 1, 1], [0, -1, 1]], dtype=float)
    b = np.array([[-3, -3], [3, -3], [3, 3], [-3, 3]], dtype=float)
    p = halfplane_intersection(hp, b)
    self.assertEqual(len(p), 4)
    self.assertTrue(np.all(hp[:, :2] @ p.T <= hp[:, 2, None] + 1e-10))
    self.assertAlmostEqual(polygon_diameter(p)[0], math.sqrt(8.0), places=12)

  def test_bearing_cone_orientation(self) -> None:
    hp = bearing_halfplanes((0, 0), 0.0, 1.0)
    self.assertTrue(np.all(hp[:, :2] @ np.array([100.0, 0.0]) <= hp[:, 2] + 1e-12))
    self.assertTrue(np.any(hp[:, :2] @ np.array([-100.0, 0.0]) > hp[:, 2] + 1e-12))

  def test_jung_counterexample(self) -> None:
    d = 100.0
    p = np.array([[0.0, 0.0], [d, 0.0], [d / 2.0, math.sqrt(3.0) * d / 2.0]])
    ok, _, rad, far = diameter_circle_coverage(p, p[:2])
    self.assertFalse(ok)
    self.assertAlmostEqual(rad, d / 2.0, places=12)
    self.assertAlmostEqual(far, math.sqrt(3.0) * d / 2.0, places=12)

  def test_minimum_circle_equilateral(self) -> None:
    d = 100.0
    p = np.array([[0.0, 0.0], [d, 0.0], [d / 2.0, math.sqrt(3.0) * d / 2.0]])
    cen, rad = minimum_enclosing_circle(p)
    np.testing.assert_allclose(cen, [d / 2.0, d / (2.0 * math.sqrt(3.0))], atol=1e-10)
    self.assertAlmostEqual(rad, d / math.sqrt(3.0), places=10)
    off = np.array([2e6, -2e6])
    moved_cen, moved_rad = minimum_enclosing_circle(p + off)
    np.testing.assert_allclose(moved_cen, cen + off, atol=1e-8)
    self.assertAlmostEqual(moved_rad, rad, places=8)

  def test_empty_geometry_is_not_zero_diameter(self) -> None:
    with self.assertRaises(ValueError):
      polygon_diameter([])
    with self.assertRaises(ValueError):
      diameter_circle_coverage([], [[0, 0], [1, 0]])

  def test_localization_translation_invariance(self) -> None:
    g = np.array([500.0, 300.0])
    s = np.array([[-300.0, -100.0], [900.0, -500.0], [-100.0, 1000.0]])
    ang = np.degrees(np.arctan2(g[1] - s[:, 1], g[0] - s[:, 0])) % 360.0
    base, _ = localization_polygon(s, ang)
    d0, _ = polygon_diameter(base)
    for shift in (1e3, 1e5, 1e6, 2e6):
      off = np.array([shift, -shift])
      moved, _ = localization_polygon(s + off, ang, center=off)
      d1, _ = polygon_diameter(moved)
      self.assertAlmostEqual(d1, d0, places=7)

  def test_circle_discretization_brackets_diameter(self) -> None:
    s = np.array([[-2000.0, 0.0], [0.0, -2000.0]])
    g = np.array([1200.0, 1200.0])
    ang = np.degrees(np.arctan2(g[1] - s[:, 1], g[0] - s[:, 0])) % 360.0
    res = localization_diameter_bounds(s, ang, n_bound=180)
    self.assertLessEqual(float(res["lower_m"]), float(res["upper_m"]))
    self.assertGreater(len(res["outer_poly"]), 0)

  def test_jacobian_finite_difference(self) -> None:
    g = np.array([31.0, -17.0])
    s = np.array([[-8.0, 4.0], [40.0, 12.0], [5.0, -30.0]])
    h = bearing_jacobian(g, s)
    dh = 1e-5
    num = np.empty_like(h)
    for j in range(2):
      e = np.zeros(2)
      e[j] = dh
      ap = np.arctan2((g + e - s)[:, 1], (g + e - s)[:, 0])
      am = np.arctan2((g - e - s)[:, 1], (g - e - s)[:, 0])
      num[:, j] = np.angle(np.exp(1j * (ap - am))) / (2.0 * dh)
    np.testing.assert_allclose(h, num, rtol=1e-8, atol=1e-9)

  def test_gdop_orthogonal_345(self) -> None:
    g = np.array([0.0, 0.0])
    s = np.array([[-3.0, 0.0], [0.0, -4.0]])
    self.assertAlmostEqual(gdop(g, s), 5.0, places=12)
    self.assertAlmostEqual(two_site_gdop(g, s[0], s[1]), 5.0, places=12)
    self.assertAlmostEqual(intersection_angle(g, s[0], s[1]), 90.0, places=12)

  def test_two_site_formula_matches_matrix(self) -> None:
    rng = np.random.default_rng(41)
    g = np.array([2.0, -3.0])
    for _ in range(100):
      s = rng.normal(size=(2, 2)) * 20.0
      if abs(np.linalg.det(s - g)) < 1e-4:
        continue
      a = gdop(g, s)
      b = two_site_gdop(g, s[0], s[1])
      self.assertTrue(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-9))

  def test_error_propagation_covariance(self) -> None:
    g = np.array([0.0, 0.0])
    s = np.array([[-3.0, 0.0], [0.0, -4.0]])
    res = error_propagation(g, s)
    sig = math.radians(1.0 / math.sqrt(3.0))
    self.assertAlmostEqual(float(res["rms"]), 5.0 * sig, places=12)
    np.testing.assert_allclose(res["gain"] @ res["h"], np.eye(2), atol=1e-12)
    np.testing.assert_allclose(res["cov"], res["cov"].T, atol=1e-12)

  def test_deterministic_error_box_bound(self) -> None:
    res = second_site_metrics_sources(
      [[0.0, -4.0]], [-3.0, 0.0], [[0.0, 0.0]], quantile=1.0)
    self.assertAlmostEqual(float(res["error_bound"][0]), 5.0 * math.radians(1.0), places=12)

  def test_singular_gdop(self) -> None:
    g = np.array([0.0, 0.0])
    s = np.array([[-3.0, 0.0], [4.0, 0.0]])
    self.assertTrue(math.isinf(gdop(g, s)))

  def test_far_orthogonal_gdop_is_finite(self) -> None:
    g = np.array([0.0, 0.0])
    s = np.array([[-1e6, 0.0], [0.0, -1e6]])
    self.assertAlmostEqual(gdop(g, s), math.sqrt(2.0) * 1e6, places=5)

  def test_candidate_boundary(self) -> None:
    rho = 900.0
    b = 400.0
    dev = b * math.tan(math.radians(20.0))
    p = np.array([[rho + dev, b]])
    ok = candidate_region_mask(p, (0, 0), 0.0, (rho, rho), 20.0,
                               recv_rad=1000.0, time_limit_s=300.0)
    self.assertTrue(bool(ok[0]))

  def test_sector_max_distance_matches_brute_force(self) -> None:
    p = np.array([[1103.949, -285.169], [700.0, 500.0]])
    exact = sector_max_distance(p, (0, 0), 25.0, (5.0, 1500.0), 1.0)
    rho = np.linspace(5.0, 1500.0, 2001)
    ang = np.radians(np.linspace(24.0, 26.0, 1001))
    brute = []
    for q in p:
      far = 0.0
      for a in ang:
        src = np.column_stack((rho * np.cos(a), rho * np.sin(a)))
        far = max(far, float(np.max(np.linalg.norm(src - q, axis=1))))
      brute.append(far)
    np.testing.assert_allclose(exact, brute, rtol=0, atol=1e-8)

  def test_old_q2_point_fails_full_cone_guarantee(self) -> None:
    p = np.array([[1103.949, -285.169]])
    ok = robust_candidate_mask(p, (0, 0), 25.0, time_limit_s=260.0)
    self.assertFalse(bool(ok[0]))
    self.assertGreater(float(sector_max_distance(p, (0, 0), 25.0)[0]), 1100.0)

  def test_cone_samples_obey_disk_and_angle(self) -> None:
    p = source_cone_samples((0, 0), 359.5, n_range=11, n_angle=9)
    self.assertTrue(np.all(np.linalg.norm(p, axis=1) <= 1800.0 + 1e-10))
    angle = np.degrees(np.arctan2(p[:, 1], p[:, 0])) % 360.0
    dev = np.abs((angle - 359.5 + 180.0) % 360.0 - 180.0)
    self.assertLessEqual(float(dev.max()), 1.0 + 1e-10)

  def test_pareto_filter(self) -> None:
    x = np.array([[1, 4], [2, 3], [3, 2], [4, 1], [3, 4], [2, 3]], dtype=float)
    ok = pareto_mask(x)
    np.testing.assert_array_equal(ok, [True, True, True, True, False, True])


if __name__ == "__main__":
  unittest.main(verbosity=2)
