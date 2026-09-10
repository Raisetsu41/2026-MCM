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
  pareto_mask,
  polygon_diameter,
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

  def test_singular_gdop(self) -> None:
    g = np.array([0.0, 0.0])
    s = np.array([[-3.0, 0.0], [4.0, 0.0]])
    self.assertTrue(math.isinf(gdop(g, s)))

  def test_candidate_boundary(self) -> None:
    rho = 900.0
    b = 400.0
    dev = b * math.tan(math.radians(20.0))
    p = np.array([[rho + dev, b]])
    ok = candidate_region_mask(p, (0, 0), 0.0, (rho, rho), 20.0,
                               recv_rad=1000.0, time_limit_s=300.0)
    self.assertTrue(bool(ok[0]))

  def test_pareto_filter(self) -> None:
    x = np.array([[1, 4], [2, 3], [3, 2], [4, 1], [3, 4], [2, 3]], dtype=float)
    ok = pareto_mask(x)
    np.testing.assert_array_equal(ok, [True, True, True, True, False, True])


if __name__ == "__main__":
  unittest.main(verbosity=2)
