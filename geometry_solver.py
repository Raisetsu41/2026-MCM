# 测向交会几何算法库.
# 实现半平面交, 旋转卡壳, GDOP 和候选点筛选.
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
  p = _pts(x)
  if len(p) <= 1:
    return p.copy()
  p = np.unique(p, axis=0)
  p = p[np.lexsort((p[:, 1], p[:, 0]))]

  def build(a: Arr) -> list[Arr]:
    s: list[Arr] = []
    for q in a:
      while len(s) >= 2 and cross(s[-1] - s[-2], q - s[-1]) <= tol:
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
  out = [x[0]]
  for p in x[1:]:
    if np.linalg.norm(p - out[-1]) > tol:
      out.append(p)
  if len(out) > 1 and np.linalg.norm(out[0] - out[-1]) <= tol:
    out.pop()
  return np.asarray(out, dtype=float)


def clip_halfplane(poly: Sequence[Sequence[float]] | Arr,
                   hp: Sequence[float] | Arr, tol: float = eps) -> Arr:
  p = _pts(poly)
  h = np.asarray(hp, dtype=float)
  if h.shape != (3,) or np.linalg.norm(h[:2]) <= tol:
    raise ValueError("half-plane must be [a, b, c] with nonzero normal")
  if len(p) == 0:
    return p
  out: list[Arr] = []
  for i, a in enumerate(p):
    b = p[(i + 1) % len(p)]
    fa = float(h[:2] @ a - h[2])
    fb = float(h[:2] @ b - h[2])
    ina, inb = fa <= tol, fb <= tol
    if ina:
      out.append(a)
    if ina != inb:
      den = fa - fb
      if abs(den) > tol:
        out.append(a + fa / den * (b - a))
  return _clean_poly(np.asarray(out, dtype=float), tol)


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
                         n_bound: int = 720) -> tuple[Arr, Arr]:
  s = _pts(sites)
  ang = np.asarray(bearings_deg, dtype=float)
  if ang.shape != (len(s),):
    raise ValueError("one bearing is required for each site")
  h = np.vstack([bearing_halfplanes(p, a, err_deg) for p, a in zip(s, ang)])
  p = halfplane_intersection(h, regular_bound(rad, n_bound))
  return p, h


def polygon_diameter(x: Sequence[Sequence[float]] | Arr,
                     tol: float = eps) -> tuple[float, Arr]:
  p = convex_hull(x, tol)
  n = len(p)
  if n == 0:
    return 0.0, np.empty((0, 2), dtype=float)
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

  for i in range(n):
    ni = (i + 1) % n
    while True:
      nj = (j + 1) % n
      cur = abs(cross(p[ni] - p[i], p[j] - p[i]))
      nxt = abs(cross(p[ni] - p[i], p[nj] - p[i]))
      if nxt > cur + tol:
        j = nj
      else:
        break
    upd(i, j)
    upd(ni, j)
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
  q = _pts(pair)
  if len(q) != 2:
    raise ValueError("pair must contain two points")
  c = q.mean(axis=0)
  r = float(np.linalg.norm(q[1] - q[0])) / 2.0
  far = float(np.max(np.linalg.norm(p - c, axis=1))) if len(p) else 0.0
  return far <= r + tol, c, r, far


def bearing_jacobian(target: Sequence[float],
                     sites: Sequence[Sequence[float]] | Arr) -> Arr:
  g = np.asarray(target, dtype=float)
  s = _pts(sites)
  d = g - s
  r2 = np.sum(d * d, axis=1)
  if np.any(r2 <= eps):
    raise ValueError("target and site must be distinct")
  return np.column_stack((-d[:, 1] / r2, d[:, 0] / r2))


def gdop(target: Sequence[float], sites: Sequence[Sequence[float]] | Arr,
         tol: float = 1e-12) -> float:
  h = bearing_jacobian(target, sites)
  info = h.T @ h
  val = np.linalg.eigvalsh(info)
  if val[0] <= tol * max(val[-1], 1.0):
    return math.inf
  return math.sqrt(float(np.trace(np.linalg.inv(info))))


def error_propagation(target: Sequence[float],
                      sites: Sequence[Sequence[float]] | Arr,
                      sig_deg: float | Sequence[float] = 1.0 / math.sqrt(3.0),
                      tol: float = 1e-12) -> dict[str, Arr | float]:
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
  if val[0] <= tol * max(val[-1], 1.0):
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
                          target_rad: float | None = None) -> Arr:
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
  t = np.linalg.norm(d, axis=1) / 5.0 + 5.0 + float(switched)
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


def pareto_mask(vals: Sequence[Sequence[float]] | Arr,
                tol: float = eps) -> Arr:
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
