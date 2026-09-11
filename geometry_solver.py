# 测向交会几何库: 半平面交、旋转卡壳、最小覆盖圆, 还有 Q2 用的 GDOP 和候选点筛选.
# 坐标单位一律是米, 角度一律是度.
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


Arr = np.ndarray
eps = 1e-10


def _pts(x: Sequence[Sequence[float]] | Arr) -> Arr:
  p = np.asarray(x, dtype=float)
  if p.size == 0:
    return np.empty((0, 2), dtype=float)
  if p.ndim != 2 or p.shape[1] != 2:
    raise ValueError("points must have shape (n, 2)")
  if not np.all(np.isfinite(p)):
    raise ValueError("points must be finite")
  return p


def cross(a: Sequence[float] | Arr, b: Sequence[float] | Arr) -> float:
  x = np.asarray(a, dtype=float)
  y = np.asarray(b, dtype=float)
  return float(x[0] * y[1] - x[1] * y[0])


def convex_hull(x: Sequence[Sequence[float]] | Arr, tol: float = eps) -> Arr:
  # Andrew 单调链, 先按 x 再按 y 排序, 正反各扫一遍
  p = _pts(x)
  if len(p) <= 1:
    return p.copy()
  p = np.unique(p, axis=0)
  p = p[np.lexsort((p[:, 1], p[:, 0]))]

  span = max(float(np.ptp(p[:, 0])), float(np.ptp(p[:, 1])), 1.0)
  area_tol = tol * span * span

  def build(a: Arr) -> list[Arr]:
    s: list[Arr] = []
    for q in a:
      # <= 而不是 <, 共线点直接丢掉, 免得后面卡壳卡在退化边上
      while len(s) >= 2 and cross(s[-1] - s[-2], q - s[-1]) <= area_tol:
        s.pop()
      s.append(q)
    return s

  if len(p) <= 2:
    return p.copy()
  lo = build(p)
  hi = build(p[::-1])
  return np.asarray(lo[:-1] + hi[:-1], dtype=float)


def regular_bound(rad: float = 1800.0, n: int = 720,
                  center: Sequence[float] = (0.0, 0.0),
                  outer: bool = True) -> Arr:
  # outer=True 是圆的外接正 n 边形(结果偏大, 当上界用), False 是内接的(当下界用)
  if rad <= 0 or n < 3:
    raise ValueError("rad must be positive and n must be at least 3")
  c = np.asarray(center, dtype=float)
  r = rad / math.cos(math.pi / n) if outer else rad
  off = math.pi / n if outer else 0.0
  ang = off + np.arange(n) * 2.0 * math.pi / n
  return c + r * np.column_stack((np.cos(ang), np.sin(ang)))


def bearing_halfplanes(site: Sequence[float], bearing_deg: float,
                       err_deg: float = 1.0) -> Arr:
  if not 0 <= err_deg < 90:
    raise ValueError("err_deg must be in [0, 90)")
  s = np.asarray(site, dtype=float)
  if s.shape != (2,) or not np.all(np.isfinite(s)):
    raise ValueError("site must be a finite point")
  lo = math.radians(bearing_deg - err_deg)
  hi = math.radians(bearing_deg + err_deg)
  dl = np.array([math.cos(lo), math.sin(lo)])
  dh = np.array([math.cos(hi), math.sin(hi)])
  h1 = np.array([dl[1], -dl[0], dl[1] * s[0] - dl[0] * s[1]])
  h2 = np.array([-dh[1], dh[0], -dh[1] * s[0] + dh[0] * s[1]])
  return np.vstack((h1, h2))


def _clean_poly(x: Arr, tol: float) -> Arr:
  if len(x) == 0:
    return np.empty((0, 2), dtype=float)
  span = max(float(np.ptp(x[:, 0])), float(np.ptp(x[:, 1])), 1.0)
  cut = tol * span
  out = [x[0]]
  for p in x[1:]:
    if np.linalg.norm(p - out[-1]) > cut:
      out.append(p)
  if len(out) > 1 and np.linalg.norm(out[0] - out[-1]) <= cut:
    out.pop()
  return np.asarray(out, dtype=float)


def clip_halfplane(poly: Sequence[Sequence[float]] | Arr,
                   hp: Sequence[float] | Arr, tol: float = eps) -> Arr:
  p = _pts(poly)
  h = np.asarray(hp, dtype=float)
  norm = float(np.linalg.norm(h[:2])) if h.shape == (3,) else 0.0
  if h.shape != (3,) or norm <= np.finfo(float).tiny:
    raise ValueError("half-plane must be [a, b, c] with nonzero normal")
  if len(p) == 0:
    return p
  # 大坐标直接相减会掉精度, 所以先平移到多边形自己的平均位置再算残差
  org = p.mean(axis=0)
  nrm = h[:2] / norm
  cut = float((h[2] - h[:2] @ org) / norm)
  q = p - org
  span = max(float(np.ptp(q[:, 0])), float(np.ptp(q[:, 1])), 1.0)
  # 容差按多边形跨度缩放, 不能用固定的绝对量, 否则平移一下结果就变了
  cut_tol = tol * span
  out: list[Arr] = []
  for i, a in enumerate(q):
    b = q[(i + 1) % len(q)]
    fa = float(nrm @ a - cut)
    fb = float(nrm @ b - cut)
    ina, inb = fa <= cut_tol, fb <= cut_tol
    if ina:
      out.append(a)
    if ina != inb:
      # 跨越边界, 补一个交点; 两边残差几乎相等时说明是平行边, 跳过
      den = fa - fb
      if abs(den) > np.finfo(float).eps * span:
        out.append(a + fa / den * (b - a))
  if not out:
    return np.empty((0, 2), dtype=float)
  return _clean_poly(np.asarray(out, dtype=float) + org, tol)


def halfplane_intersection(hps: Sequence[Sequence[float]] | Arr,
                           bound: Sequence[Sequence[float]] | Arr,
                           tol: float = eps) -> Arr:
  h = np.asarray(hps, dtype=float)
  if h.size == 0:
    return _pts(bound).copy()
  if h.ndim != 2 or h.shape[1] != 3:
    raise ValueError("hps must have shape (n, 3)")
  p = _pts(bound).copy()
  for row in h:
    p = clip_halfplane(p, row, tol)
    if len(p) == 0:
      break
  return p


def localization_polygon(sites: Sequence[Sequence[float]] | Arr,
                         bearings_deg: Sequence[float] | Arr,
                         err_deg: float = 1.0, rad: float = 1800.0,
                         n_bound: int = 720,
                         center: Sequence[float] = (0.0, 0.0),
                         outer: bool = True) -> tuple[Arr, Arr]:
  s = _pts(sites)
  ang = np.asarray(bearings_deg, dtype=float)
  if ang.shape != (len(s),):
    raise ValueError("one bearing is required for each site")
  if len(s) == 0:
    raise ValueError("at least one bearing is required")
  c = np.asarray(center, dtype=float)
  if c.shape != (2,) or not np.all(np.isfinite(c)):
    raise ValueError("center must be a finite point")
  local_sites = s - c
  local_hps = np.vstack([
    bearing_halfplanes(p, a, err_deg) for p, a in zip(local_sites, ang)
  ])
  local_poly = halfplane_intersection(
    local_hps, regular_bound(rad, n_bound, outer=outer))
  global_hps = np.vstack([
    bearing_halfplanes(p, a, err_deg) for p, a in zip(s, ang)
  ])
  return local_poly + c, global_hps


def localization_diameter_bounds(
    sites: Sequence[Sequence[float]] | Arr,
    bearings_deg: Sequence[float] | Arr,
    err_deg: float = 1.0, rad: float = 1800.0,
    n_bound: int = 720,
    center: Sequence[float] = (0.0, 0.0),
) -> dict[str, Arr | float]:
  outer_poly, hps = localization_polygon(
    sites, bearings_deg, err_deg, rad, n_bound, center, outer=True)
  if len(outer_poly) == 0:
    raise ValueError("bearing constraints have empty intersection")
  inner_poly, _ = localization_polygon(
    sites, bearings_deg, err_deg, rad, n_bound, center, outer=False)
  upper, upper_pair = polygon_diameter(outer_poly)
  if len(inner_poly):
    lower, lower_pair = polygon_diameter(inner_poly)
  else:
    lower = 0.0
    lower_pair = np.empty((0, 2), dtype=float)
  return {
    "inner_poly": inner_poly,
    "outer_poly": outer_poly,
    "halfplanes": hps,
    "lower_m": lower,
    "upper_m": upper,
    "lower_pair": lower_pair,
    "upper_pair": upper_pair,
  }


def polygon_diameter(x: Sequence[Sequence[float]] | Arr,
                     tol: float = eps) -> tuple[float, Arr]:
  p = convex_hull(x, tol)
  n = len(p)
  if n == 0:
    raise ValueError("diameter is undefined for an empty set")
  if n == 1:
    return 0.0, np.vstack((p[0], p[0]))
  if n == 2:
    return float(np.linalg.norm(p[1] - p[0])), p.copy()
  j = 1
  best = -1.0
  pair = np.vstack((p[0], p[1]))

  def upd(a: int, b: int) -> None:
    nonlocal best, pair
    val = float((p[a] - p[b]) @ (p[a] - p[b]))
    if val > best + tol:
      best = val
      pair = np.vstack((p[a], p[b]))

  # 旋转卡壳: j 只会沿凸包单调往前走, 所以整个循环 j 总共绕一圈, 是线性的
  for i in range(n):
    ni = (i + 1) % n
    while True:
      nj = (j + 1) % n
      cur = abs(cross(p[ni] - p[i], p[j] - p[i]))
      nxt = abs(cross(p[ni] - p[i], p[nj] - p[i]))
      if nxt > cur + tol:
        j = nj            # 三角形面积还在变大, 对踵点继续往前挪
      else:
        break
    upd(i, j)
    upd(ni, j)
    # 两条边平行时会出现两个对踵点, 都得试
    nj = (j + 1) % n
    cur = abs(cross(p[ni] - p[i], p[j] - p[i]))
    nxt = abs(cross(p[ni] - p[i], p[nj] - p[i]))
    if abs(nxt - cur) <= tol:
      upd(i, nj)
      upd(ni, nj)
  return math.sqrt(max(best, 0.0)), pair


def diameter_circle_coverage(x: Sequence[Sequence[float]] | Arr,
                             pair: Sequence[Sequence[float]] | Arr,
                             tol: float = eps) -> tuple[bool, Arr, float, float]:
  p = _pts(x)
  if len(p) == 0:
    raise ValueError("coverage is undefined for an empty set")
  q = _pts(pair)
  if len(q) != 2:
    raise ValueError("pair must contain two points")
  c = q.mean(axis=0)
  r = float(np.linalg.norm(q[1] - q[0])) / 2.0
  far = float(np.max(np.linalg.norm(p - c, axis=1)))
  return far <= r + tol, c, r, far


def _circle3(a: Arr, b: Arr, c: Arr, tol: float) -> tuple[Arr, float] | None:
  ab = b - a
  ac = c - a
  d = 2.0 * cross(ab, ac)
  span = max(float(np.linalg.norm(ab)), float(np.linalg.norm(ac)), 1.0)
  if abs(d) <= tol * span * span:
    return None
  ab2 = float(ab @ ab)
  ac2 = float(ac @ ac)
  off = np.array([
    (ab2 * ac[1] - ac2 * ab[1]) / d,
    (ab[0] * ac2 - ac[0] * ab2) / d,
  ])
  cen = a + off
  return cen, float(np.linalg.norm(cen - a))


def minimum_enclosing_circle(
    x: Sequence[Sequence[float]] | Arr,
    tol: float = eps,
) -> tuple[Arr, float]:
  p = _pts(x)
  if len(p) == 0:
    raise ValueError("minimum circle is undefined for an empty set")
  order = np.random.default_rng(0).permutation(len(p))
  q = p[order]
  cen = q[0].copy()
  rad = 0.0
  for i in range(len(q)):
    if np.linalg.norm(q[i] - cen) <= rad + tol * max(rad, 1.0):
      continue
    cen = q[i].copy()
    rad = 0.0
    for j in range(i):
      if np.linalg.norm(q[j] - cen) <= rad + tol * max(rad, 1.0):
        continue
      cen = (q[i] + q[j]) / 2.0
      rad = float(np.linalg.norm(q[i] - q[j])) / 2.0
      for k in range(j):
        if np.linalg.norm(q[k] - cen) <= rad + tol * max(rad, 1.0):
          continue
        cir = _circle3(q[i], q[j], q[k], tol)
        if cir is None:
          tri = np.vstack((q[i], q[j], q[k]))
          _, pair = polygon_diameter(tri)
          cen = pair.mean(axis=0)
          rad = float(np.linalg.norm(pair[1] - pair[0])) / 2.0
        else:
          cen, rad = cir
  return cen, rad


def bearing_jacobian(target: Sequence[float],
                     sites: Sequence[Sequence[float]] | Arr) -> Arr:
  # atan2 对位置的偏导, 第 i 行就是第 i 个测站那两列
  g = np.asarray(target, dtype=float)
  s = _pts(sites)
  d = g - s
  r2 = np.sum(d * d, axis=1)
  # 判重合的阈值跟着坐标量级走. 之前写死 1e-10, 尺度小的算例会误报
  scale = max(float(np.linalg.norm(g)), float(np.max(np.linalg.norm(s, axis=1))), 1.0)
  zero = 32.0 * np.finfo(float).eps * scale
  if np.any(r2 <= zero * zero):
    raise ValueError("target and site must be distinct")
  return np.column_stack((-d[:, 1] / r2, d[:, 0] / r2))


def gdop(target: Sequence[float], sites: Sequence[Sequence[float]] | Arr,
         tol: float = 1e-12) -> float:
  h = bearing_jacobian(target, sites)
  info = h.T @ h
  val = np.linalg.eigvalsh(info)
  if val[-1] <= 0.0 or val[0] <= tol * val[-1]:
    return math.inf
  return math.sqrt(float(np.trace(np.linalg.inv(info))))


def error_propagation(target: Sequence[float],
                      sites: Sequence[Sequence[float]] | Arr,
                      sig_deg: float | Sequence[float] = 1.0 / math.sqrt(3.0),
                      tol: float = 1e-12) -> dict[str, Arr | float]:
  # 加权最小二乘: 测角残差 e 到位置增量 dG 的映射 K = (H'WH)^-1 H'W, 维数 2×m
  h = bearing_jacobian(target, sites)
  sig = np.asarray(sig_deg, dtype=float)
  if sig.ndim == 0:
    sig = np.full(len(h), float(sig))
  if sig.shape != (len(h),) or np.any(sig <= 0):
    raise ValueError("sig_deg must be positive for each site")
  sig = np.radians(sig)
  r = np.diag(sig * sig)
  w = np.diag(1.0 / (sig * sig))
  info = h.T @ w @ h
  val = np.linalg.eigvalsh(info)
  # 信息矩阵奇异说明两条视线共线, 这种情况直接报错, 不能硬算逆
  if val[-1] <= 0.0 or val[0] <= tol * val[-1]:
    raise ValueError("bearing geometry is singular")
  cov = np.linalg.inv(info)
  gain = cov @ h.T @ w
  return {
    "h": h,
    "r": r,
    "gain": gain,
    "cov": cov,
    "gdop": gdop(target, sites, tol),
    "rms": math.sqrt(float(np.trace(cov))),
  }


def intersection_angle(target: Sequence[float], site1: Sequence[float],
                       site2: Sequence[float]) -> float:
  g = np.asarray(target, dtype=float)
  a = np.asarray(site1, dtype=float) - g
  b = np.asarray(site2, dtype=float) - g
  den = float(np.linalg.norm(a) * np.linalg.norm(b))
  if den <= eps:
    raise ValueError("target and site must be distinct")
  val = min(1.0, abs(float(a @ b)) / den)
  return math.degrees(math.acos(val))


def two_site_gdop(target: Sequence[float], site1: Sequence[float],
                  site2: Sequence[float]) -> float:
  g = np.asarray(target, dtype=float)
  a = np.asarray(site1, dtype=float) - g
  b = np.asarray(site2, dtype=float) - g
  r1, r2 = float(np.linalg.norm(a)), float(np.linalg.norm(b))
  if min(r1, r2) <= eps:
    raise ValueError("target and site must be distinct")
  sn = abs(cross(a, b)) / (r1 * r2)
  if sn <= eps:
    return math.inf
  return math.sqrt(r1 * r1 + r2 * r2) / sn


def second_site_time(site1: Sequence[float], site2: Sequence[float],
                     speed: float = 5.0, switched: bool = False,
                     measure_s: float = 5.0, switch_s: float = 1.0) -> float:
  if speed <= 0:
    raise ValueError("speed must be positive")
  d = np.asarray(site2, dtype=float) - np.asarray(site1, dtype=float)
  return float(np.linalg.norm(d)) / speed + measure_s + switch_s * switched


def nominal_range_bounds(site: Sequence[float], bearing_deg: float,
                         target_rad: float = 1800.0,
                         recv_max: float = 1500.0,
                         near: float = 5.0) -> tuple[float, float]:
  s = np.asarray(site, dtype=float)
  ang = math.radians(bearing_deg)
  u = np.array([math.cos(ang), math.sin(ang)])
  su = float(s @ u)
  det = su * su + target_rad * target_rad - float(s @ s)
  if det < 0:
    raise ValueError("bearing ray misses target disk")
  lo = max(near, -su - math.sqrt(det), 0.0)
  hi = min(recv_max, -su + math.sqrt(det))
  if hi <= lo:
    raise ValueError("no feasible nominal target range")
  return lo, hi


def candidate_region_mask(cand: Sequence[Sequence[float]] | Arr,
                          site1: Sequence[float], bearing_deg: float,
                          ranges: tuple[float, float], angle_tol_deg: float,
                          recv_rad: float = 1000.0,
                          time_limit_s: float = math.inf,
                          switched: bool = False, near: float = 5.0,
                          target_rad: float | None = None,
                          speed: float = 5.0,
                          measure_s: float = 5.0,
                          switch_s: float = 1.0) -> Arr:
  p = _pts(cand)
  s = np.asarray(site1, dtype=float)
  lo, hi = ranges
  if not 0 <= lo <= hi or not 0 < angle_tol_deg < 90:
    raise ValueError("invalid ranges or angle tolerance")
  ang = math.radians(bearing_deg)
  u = np.array([math.cos(ang), math.sin(ang)])
  v = np.array([-math.sin(ang), math.cos(ang)])
  d = p - s
  a, b = d @ u, d @ v
  dev = np.maximum(np.abs(a - lo), np.abs(a - hi))
  ok = dev <= np.abs(b) * math.tan(math.radians(angle_tol_deg)) + eps
  far2 = np.maximum((a - lo) ** 2 + b * b, (a - hi) ** 2 + b * b)
  ok &= far2 <= recv_rad * recv_rad + eps
  q = np.clip(a, lo, hi)
  near2 = (a - q) ** 2 + b * b
  ok &= near2 > near * near + eps
  if speed <= 0 or measure_s < 0 or switch_s < 0:
    raise ValueError("time parameters must be nonnegative and speed positive")
  t = (np.linalg.norm(d, axis=1) / speed + measure_s
       + switch_s * float(switched))
  ok &= t <= time_limit_s + eps
  if target_rad is not None:
    ok &= np.linalg.norm(p, axis=1) <= target_rad + eps
  return ok


def second_site_metrics(cand: Sequence[Sequence[float]] | Arr,
                        site1: Sequence[float], bearing_deg: float,
                        ranges: Sequence[float] | Arr,
                        recv_rad: float = 1000.0, near: float = 5.0,
                        quantile: float = 1.0,
                        switched: bool = False) -> dict[str, Arr]:
  p = _pts(cand)
  s = np.asarray(site1, dtype=float)
  rho = np.asarray(ranges, dtype=float)
  if rho.ndim != 1 or len(rho) == 0 or np.any(rho <= 0):
    raise ValueError("ranges must be a positive vector")
  if not 0 <= quantile <= 1:
    raise ValueError("quantile must be in [0, 1]")
  ang = math.radians(bearing_deg)
  u = np.array([math.cos(ang), math.sin(ang)])
  g = s + rho[:, None] * u
  d = p[:, None, :] - g[None, :, :]
  r2 = np.linalg.norm(d, axis=2)
  cs = np.abs(np.sum((-u)[None, None, :] * d, axis=2)) / np.maximum(r2, eps)
  cs = np.clip(cs, 0.0, 1.0)
  sn = np.sqrt(np.maximum(1.0 - cs * cs, 0.0))
  gp = np.sqrt(rho[None, :] ** 2 + r2 * r2) / np.maximum(sn, eps)
  valid = (r2 <= recv_rad + eps) & (r2 > near + eps)
  cov = valid.mean(axis=1)
  acc = np.quantile(cs, quantile, axis=1)
  gq = np.quantile(gp, quantile, axis=1)
  tm = np.asarray([second_site_time(s, q, switched=switched) for q in p])
  return {"angle_cost": acc, "gdop": gq, "time_s": tm, "coverage": cov}


def source_cone_samples(
    site: Sequence[float], bearing_deg: float,
    ranges: tuple[float, float] = (5.0, 1500.0),
    err_deg: float = 1.0, target_rad: float = 1800.0,
    n_range: int = 41, n_angle: int = 21,
) -> Arr:
  # 首次测到方向后, 可能的目标位置就是这个扇形: 距离 5~1500 米, 角度 ±1°, 再和目标圆求交
  s = np.asarray(site, dtype=float)
  lo, hi = ranges
  if s.shape != (2,) or not np.all(np.isfinite(s)):
    raise ValueError("site must be a finite point")
  if not 0 <= lo < hi or not 0 <= err_deg < 90:
    raise ValueError("invalid cone ranges or angle error")
  if target_rad <= 0 or n_range < 2 or n_angle < 2:
    raise ValueError("invalid disk radius or sample counts")
  rho = np.linspace(lo, hi, n_range)
  ang = np.radians(bearing_deg + np.linspace(-err_deg, err_deg, n_angle))
  rr, aa = np.meshgrid(rho, ang)
  p = s + np.column_stack((
    (rr * np.cos(aa)).ravel(),
    (rr * np.sin(aa)).ravel(),
  ))
  return p[np.linalg.norm(p, axis=1) <= target_rad + eps]


def sector_max_distance(
    cand: Sequence[Sequence[float]] | Arr,
    site: Sequence[float], bearing_deg: float,
    ranges: tuple[float, float] = (5.0, 1500.0),
    err_deg: float = 1.0,
) -> Arr:
  # 候选点到整个扇形的最远距离. 距离平方对 rho 是凸的, 最大值只可能在 rho 的两个端点取到
  p = _pts(cand)
  s = np.asarray(site, dtype=float)
  lo, hi = ranges
  if not 0 <= lo <= hi or not 0 <= err_deg < 90:
    raise ValueError("invalid cone ranges or angle error")
  ang = math.radians(bearing_deg)
  u = np.array([math.cos(ang), math.sin(ang)])
  v = np.array([-math.sin(ang), math.cos(ang)])
  d = p - s
  a, b = d @ u, d @ v
  err = math.radians(err_deg)
  # 两个角度端点里取更近的那个; 另外如果正后方落在 ±err 内, 那一点更近
  end_min = np.minimum(
    a * math.cos(err) + b * math.sin(err),
    a * math.cos(err) - b * math.sin(err),
  )
  phase = np.arctan2(b, a)
  opp = (phase + math.pi + math.pi) % (2.0 * math.pi) - math.pi
  dot_min = np.where(np.abs(opp) <= err + eps, -np.hypot(a, b), end_min)
  q_lo = np.sum(d * d, axis=1) + lo * lo - 2.0 * lo * dot_min
  q_hi = np.sum(d * d, axis=1) + hi * hi - 2.0 * hi * dot_min
  return np.sqrt(np.maximum(np.maximum(q_lo, q_hi), 0.0))


def robust_candidate_mask(
    cand: Sequence[Sequence[float]] | Arr,
    site1: Sequence[float], bearing_deg: float,
    ranges: tuple[float, float] = (5.0, 1500.0),
    err_deg: float = 1.0, recv_rad: float = 1000.0,
    time_limit_s: float = math.inf, speed: float = 5.0,
    measure_s: float = 5.0, switched: bool = False,
    switch_s: float = 1.0,
) -> Arr:
  p = _pts(cand)
  s = np.asarray(site1, dtype=float)
  if recv_rad <= 0 or speed <= 0 or measure_s < 0 or switch_s < 0:
    raise ValueError("invalid reception or time parameters")
  far = sector_max_distance(p, s, bearing_deg, ranges, err_deg)
  t = (np.linalg.norm(p - s, axis=1) / speed + measure_s
       + switch_s * float(switched))
  return (far <= recv_rad + eps) & (t <= time_limit_s + eps)


def second_site_metrics_sources(
    cand: Sequence[Sequence[float]] | Arr,
    site1: Sequence[float], sources: Sequence[Sequence[float]] | Arr,
    sig_deg: float = 1.0 / math.sqrt(3.0),
    err_deg: float = 1.0,
    recv_rad: float = 1000.0, quantile: float = 1.0,
    speed: float = 5.0, measure_s: float = 5.0,
    switched: bool = False, switch_s: float = 1.0,
) -> dict[str, Arr]:
  p = _pts(cand)
  g = _pts(sources)
  s = np.asarray(site1, dtype=float)
  if len(g) == 0 or not 0 <= quantile <= 1:
    raise ValueError("sources must be nonempty and quantile valid")
  first = s[None, :] - g
  r1 = np.linalg.norm(first, axis=1)
  second = p[:, None, :] - g[None, :, :]
  r2 = np.linalg.norm(second, axis=2)
  dot = np.sum(first[None, :, :] * second, axis=2)
  den = r1[None, :] * r2
  cos_val = np.abs(dot) / np.maximum(den, np.finfo(float).tiny)
  cos_val = np.clip(cos_val, 0.0, 1.0)
  sin_val = np.sqrt(np.maximum(1.0 - cos_val * cos_val, 0.0))
  raw = np.sqrt(r1[None, :] ** 2 + r2 * r2) / np.maximum(sin_val, eps)
  raw[(r1[None, :] <= eps) | (r2 <= eps)] = math.inf
  h1x = first[:, 1] / np.maximum(r1 * r1, np.finfo(float).tiny)
  h1y = -first[:, 0] / np.maximum(r1 * r1, np.finfo(float).tiny)
  h2x = second[:, :, 1] / np.maximum(r2 * r2, np.finfo(float).tiny)
  h2y = -second[:, :, 0] / np.maximum(r2 * r2, np.finfo(float).tiny)
  det = h1x[None, :] * h2y - h1y[None, :] * h2x
  safe_det = np.where(np.abs(det) > np.finfo(float).tiny, det, math.nan)
  k1x = h2y / safe_det
  k1y = -h2x / safe_det
  k2x = -h1y[None, :] / safe_det
  k2y = h1x[None, :] / safe_det
  plus = np.hypot(k1x + k2x, k1y + k2y)
  minus = np.hypot(k1x - k2x, k1y - k2y)
  if sig_deg <= 0 or err_deg <= 0:
    raise ValueError("angle scales must be positive")
  bound = math.radians(err_deg) * np.maximum(plus, minus)
  bound[~np.isfinite(bound)] = math.inf
  if quantile == 1.0:
    ang_cost = np.max(cos_val, axis=1)
    gdop_val = np.max(raw, axis=1)
    error_bound = np.max(bound, axis=1)
  elif quantile == 0.0:
    ang_cost = np.min(cos_val, axis=1)
    gdop_val = np.min(raw, axis=1)
    error_bound = np.min(bound, axis=1)
  else:
    ang_cost = np.quantile(cos_val, quantile, axis=1)
    gdop_val = np.quantile(raw, quantile, axis=1)
    error_bound = np.quantile(bound, quantile, axis=1)
  rms = math.radians(sig_deg) * gdop_val
  coverage = np.mean(r2 <= recv_rad + eps, axis=1)
  time_s = (np.linalg.norm(p - s, axis=1) / speed + measure_s
            + switch_s * float(switched))
  return {
    "angle_cost": ang_cost,
    "gdop": gdop_val,
    "rms": rms,
    "error_bound": error_bound,
    "time_s": time_s,
    "coverage": coverage,
  }


def pareto_mask(vals: Sequence[Sequence[float]] | Arr,
                tol: float = eps) -> Arr:
  # 两目标非支配筛选. 含 inf 的点直接出局, 剩下的两两比: 只要有人两项都不差且有一项更好, 就被支配
  x = np.asarray(vals, dtype=float)
  if x.ndim != 2 or len(x) == 0:
    raise ValueError("vals must have shape (n, m) with n positive")
  ok = np.all(np.isfinite(x), axis=1)
  ids = np.flatnonzero(ok)
  keep = np.zeros(len(x), dtype=bool)
  for i in ids:
    dom = np.all(x[ids] <= x[i] + tol, axis=1)
    strict = np.any(x[ids] < x[i] - tol, axis=1)
    if not np.any(dom & strict):
      keep[i] = True
  return keep


def pareto_second_sites(cand: Sequence[Sequence[float]] | Arr,
                        site1: Sequence[float], bearing_deg: float,
                        ranges: Sequence[float] | Arr,
                        recv_rad: float = 1000.0, quantile: float = 0.9,
                        min_coverage: float = 1.0,
                        switched: bool = False) -> dict[str, Arr]:
  res = second_site_metrics(cand, site1, bearing_deg, ranges, recv_rad,
                            quantile=quantile, switched=switched)
  vals = np.column_stack((res["angle_cost"], res["time_s"]))
  feas = res["coverage"] + eps >= min_coverage
  vals[~feas] = math.inf
  res["pareto"] = pareto_mask(vals)
  return res


def pareto_second_sites_cone(
    cand: Sequence[Sequence[float]] | Arr,
    site1: Sequence[float], bearing_deg: float,
    ranges: tuple[float, float] = (5.0, 1500.0),
    err_deg: float = 1.0, recv_rad: float = 1000.0,
    quantile: float = 1.0, min_coverage: float = 1.0,
    n_range: int = 41, n_angle: int = 21,
    switched: bool = False,
) -> dict[str, Arr]:
  # Q2 主模型: 精度用确定性最坏误差界, 时间用移动+测向, 对完整先验求 Pareto
  src = source_cone_samples(
    site1, bearing_deg, ranges, err_deg, n_range=n_range, n_angle=n_angle)
  res = second_site_metrics_sources(
    cand, site1, src, recv_rad=recv_rad, quantile=quantile,
    switched=switched)
  vals = np.column_stack((res["error_bound"], res["time_s"]))
  feasible = res["coverage"] + eps >= min_coverage
  safe_vals = vals.copy()
  safe_vals[~feasible] = math.inf
  res["pareto"] = pareto_mask(safe_vals)
  return res
