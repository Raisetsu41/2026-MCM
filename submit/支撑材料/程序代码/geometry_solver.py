from __future__ import annotations

import math
import heapq
from collections.abc import Sequence

import numpy as np

Arr = np.ndarray
eps = 1e-10
rank_rel_tol = 1e-12
sin_angle_tol = 1e-12

def _relative_tol(value: float, name: str = "tol") -> float:
  if not math.isfinite(value) or value < 0.0:
    raise ValueError(f"{name} must be finite and nonnegative")
  return float(value)

def _point(x: Sequence[float] | Arr, name: str) -> Arr:
  p = np.asarray(x, dtype=float)
  if p.shape != (2,) or not np.all(np.isfinite(p)):
    raise ValueError(f"{name} must be a finite point")
  return p

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
  tol = _relative_tol(tol)
  if len(p) <= 1:
    return p.copy()
  p = np.unique(p, axis=0)
  p = p[np.lexsort((p[:, 1], p[:, 0]))]

  span = max(float(np.ptp(p[:, 0])), float(np.ptp(p[:, 1])),
             np.finfo(float).tiny)
  area_tol = tol * span * span

  def build(a: Arr) -> list[Arr]:
    s: list[Arr] = []
    for q in a:
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
  if (not math.isfinite(rad) or rad <= 0 or not isinstance(n, int)
      or isinstance(n, bool) or n < 3):
    raise ValueError("rad must be positive and n must be at least 3")
  c = _point(center, "center")
  r = rad / math.cos(math.pi / n) if outer else rad
  off = math.pi / n if outer else 0.0
  ang = off + np.arange(n) * 2.0 * math.pi / n
  return c + r * np.column_stack((np.cos(ang), np.sin(ang)))

def bearing_halfplanes(site: Sequence[float], bearing_deg: float,
                       err_deg: float = 1.0) -> Arr:
  if (not math.isfinite(bearing_deg) or not math.isfinite(err_deg)
      or not 0 <= err_deg < 90):
    raise ValueError("err_deg must be in [0, 90)")
  s = _point(site, "site")
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
  span = max(float(np.ptp(x[:, 0])), float(np.ptp(x[:, 1])),
             np.finfo(float).tiny)
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
  tol = _relative_tol(tol)
  h = np.asarray(hp, dtype=float)
  norm = float(np.linalg.norm(h[:2])) if h.shape == (3,) else 0.0
  if (h.shape != (3,) or not np.all(np.isfinite(h))
      or norm <= np.finfo(float).tiny):
    raise ValueError("half-plane must be [a, b, c] with nonzero normal")
  if len(p) == 0:
    return p
  org = p.mean(axis=0)
  nrm = h[:2] / norm
  cut = float((h[2] - h[:2] @ org) / norm)
  q = p - org
  span = max(float(np.ptp(q[:, 0])), float(np.ptp(q[:, 1])),
             np.finfo(float).tiny)
  roundoff = 32.0 * np.finfo(float).eps * max(
    float(np.linalg.norm(org)), abs(cut), span)
  cut_tol = tol * span + roundoff
  out: list[Arr] = []
  for i, a in enumerate(q):
    b = q[(i + 1) % len(q)]
    fa = float(nrm @ a - cut)
    fb = float(nrm @ b - cut)
    ina, inb = fa <= cut_tol, fb <= cut_tol
    if ina:
      out.append(a)
    if ina != inb:
      den = fa - fb
      den_tol = 32.0 * np.finfo(float).eps * max(
        abs(fa), abs(fb), span)
      if abs(den) > den_tol:
        out.append(a + fa / den * (b - a))
  if not out:
    return np.empty((0, 2), dtype=float)
  return _clean_poly(np.asarray(out, dtype=float) + org, tol)

def halfplane_intersection(hps: Sequence[Sequence[float]] | Arr,
                           bound: Sequence[Sequence[float]] | Arr,
                           tol: float = eps) -> Arr:
  h = np.asarray(hps, dtype=float)
  tol = _relative_tol(tol)
  if h.size == 0:
    return _pts(bound).copy()
  if h.ndim != 2 or h.shape[1] != 3 or not np.all(np.isfinite(h)):
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
  if ang.shape != (len(s),) or not np.all(np.isfinite(ang)):
    raise ValueError("one bearing is required for each site")
  if len(s) == 0:
    raise ValueError("at least one bearing is required")
  c = _point(center, "center")
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
    if lower > upper:
      slack = eps * max(lower, upper, np.finfo(float).tiny)
      if lower - upper > slack:
        raise RuntimeError("inner diameter exceeds outer diameter")
      lower = upper
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
  raw = _pts(x)
  tol = _relative_tol(tol)
  if len(raw) == 0:
    raise ValueError("diameter is undefined for an empty set")
  org = raw.mean(axis=0)
  p = convex_hull(raw - org, tol)
  n = len(p)
  if n == 1:
    return 0.0, np.vstack((p[0], p[0])) + org
  if n == 2:
    return float(np.linalg.norm(p[1] - p[0])), p.copy() + org
  span = max(float(np.ptp(p[:, 0])), float(np.ptp(p[:, 1])),
             np.finfo(float).tiny)
  area_tol = tol * span * span
  dist2_tol = tol * span * span
  j = 1
  best = -1.0
  pair = np.vstack((p[0], p[1]))

  def upd(a: int, b: int) -> None:
    nonlocal best, pair
    val = float((p[a] - p[b]) @ (p[a] - p[b]))
    if val > best + dist2_tol:
      best = val
      pair = np.vstack((p[a], p[b]))

  for i in range(n):
    ni = (i + 1) % n
    while True:
      nj = (j + 1) % n
      cur = abs(cross(p[ni] - p[i], p[j] - p[i]))
      nxt = abs(cross(p[ni] - p[i], p[nj] - p[i]))
      if nxt > cur + area_tol:
        j = nj
      else:
        break
    upd(i, j)
    upd(ni, j)
    nj = (j + 1) % n
    cur = abs(cross(p[ni] - p[i], p[j] - p[i]))
    nxt = abs(cross(p[ni] - p[i], p[nj] - p[i]))
    if abs(nxt - cur) <= area_tol:
      upd(i, nj)
      upd(ni, nj)
  return math.sqrt(max(best, 0.0)), pair + org

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
  span = max(float(np.linalg.norm(ab)), float(np.linalg.norm(ac)),
             np.finfo(float).tiny)
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
  tol = _relative_tol(tol)
  if len(p) == 0:
    raise ValueError("minimum circle is undefined for an empty set")
  org = p.mean(axis=0)
  span = max(float(np.ptp(p[:, 0])), float(np.ptp(p[:, 1])),
             np.finfo(float).tiny)
  cut = tol * span
  order = np.random.default_rng(0).permutation(len(p))
  q = p[order] - org
  cen = q[0].copy()
  rad = 0.0
  for i in range(len(q)):
    if np.linalg.norm(q[i] - cen) <= rad + cut:
      continue
    cen = q[i].copy()
    rad = 0.0
    for j in range(i):
      if np.linalg.norm(q[j] - cen) <= rad + cut:
        continue
      cen = (q[i] + q[j]) / 2.0
      rad = float(np.linalg.norm(q[i] - q[j])) / 2.0
      for k in range(j):
        if np.linalg.norm(q[k] - cen) <= rad + cut:
          continue
        cir = _circle3(q[i], q[j], q[k], tol)
        if cir is None:
          tri = np.vstack((q[i], q[j], q[k]))
          _, pair = polygon_diameter(tri)
          cen = pair.mean(axis=0)
          rad = float(np.linalg.norm(pair[1] - pair[0])) / 2.0
        else:
          cen, rad = cir
  return cen + org, rad

def bearing_jacobian(target: Sequence[float],
                     sites: Sequence[Sequence[float]] | Arr) -> Arr:
  g = _point(target, "target")
  s = _pts(sites)
  if len(s) == 0:
    raise ValueError("at least one site is required")
  d = g - s
  r2 = np.sum(d * d, axis=1)
  if np.any(r2 <= np.finfo(float).tiny):
    raise ValueError("target and site must be distinct")
  return np.column_stack((-d[:, 1] / r2, d[:, 0] / r2))

def gdop(target: Sequence[float], sites: Sequence[Sequence[float]] | Arr,
         tol: float = rank_rel_tol) -> float:
  tol = _relative_tol(tol)
  h = bearing_jacobian(target, sites)
  info = h.T @ h
  val = np.linalg.eigvalsh(info)
  if val[-1] <= 0.0 or val[0] <= tol * val[-1]:
    return math.inf
  cov = np.linalg.solve(info, np.eye(2))
  return math.sqrt(float(np.trace(cov)))

def error_propagation(target: Sequence[float],
                      sites: Sequence[Sequence[float]] | Arr,
                      sig_deg: float | Sequence[float] = 1.0 / math.sqrt(3.0),
                      tol: float = rank_rel_tol) -> dict[str, Arr | float]:
  tol = _relative_tol(tol)
  h = bearing_jacobian(target, sites)
  sig = np.asarray(sig_deg, dtype=float)
  if sig.ndim == 0:
    sig = np.full(len(h), float(sig))
  if (sig.shape != (len(h),) or not np.all(np.isfinite(sig))
      or np.any(sig <= 0)):
    raise ValueError("sig_deg must be positive for each site")
  sig = np.radians(sig)
  r = np.diag(sig * sig)
  w = np.diag(1.0 / (sig * sig))
  info = h.T @ w @ h
  val = np.linalg.eigvalsh(info)
  if val[-1] <= 0.0 or val[0] <= tol * val[-1]:
    raise ValueError("bearing geometry is singular")
  cov = np.linalg.solve(info, np.eye(2))
  gain = np.linalg.solve(info, h.T @ w)
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
  g = _point(target, "target")
  a = _point(site1, "site1") - g
  b = _point(site2, "site2") - g
  den = float(np.linalg.norm(a) * np.linalg.norm(b))
  if den <= np.finfo(float).tiny:
    raise ValueError("target and site must be distinct")
  val = min(1.0, abs(float(a @ b)) / den)
  return math.degrees(math.acos(val))

def two_site_gdop(target: Sequence[float], site1: Sequence[float],
                  site2: Sequence[float],
                  angle_tol: float = sin_angle_tol) -> float:
  angle_tol = _relative_tol(angle_tol, "angle_tol")
  g = _point(target, "target")
  a = _point(site1, "site1") - g
  b = _point(site2, "site2") - g
  r1, r2 = float(np.linalg.norm(a)), float(np.linalg.norm(b))
  if min(r1, r2) <= np.finfo(float).tiny:
    raise ValueError("target and site must be distinct")
  sn = abs(cross(a, b)) / (r1 * r2)
  if sn <= angle_tol:
    return math.inf
  return math.sqrt(r1 * r1 + r2 * r2) / sn

def second_site_time(site1: Sequence[float], site2: Sequence[float],
                     speed: float = 5.0, switched: bool = False,
                     measure_s: float = 5.0, switch_s: float = 1.0) -> float:
  if (not math.isfinite(speed) or speed <= 0
      or not math.isfinite(measure_s) or measure_s < 0
      or not math.isfinite(switch_s) or switch_s < 0):
    raise ValueError("time parameters must be nonnegative and speed positive")
  d = _point(site2, "site2") - _point(site1, "site1")
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
  s = _point(site, "site")
  lo, hi = ranges
  if (not math.isfinite(bearing_deg) or not math.isfinite(lo)
      or not math.isfinite(hi) or not math.isfinite(err_deg)
      or not 0 <= lo < hi or not 0 <= err_deg < 90):
    raise ValueError("invalid cone ranges or angle error")
  if (not math.isfinite(target_rad) or target_rad <= 0
      or not isinstance(n_range, int) or isinstance(n_range, bool)
      or not isinstance(n_angle, int) or isinstance(n_angle, bool)
      or n_range < 2 or n_angle < 2):
    raise ValueError("invalid disk radius or sample counts")
  offsets = np.unique(np.append(
    np.linspace(-err_deg, err_deg, n_angle), 0.0))
  angles = np.radians(bearing_deg + offsets)
  samples = []
  for angle in angles:
    direction = np.array([math.cos(angle), math.sin(angle)])
    projection = float(s @ direction)
    discriminant = projection * projection + target_rad * target_rad - float(s @ s)
    if discriminant < 0.0:
      continue
    root = math.sqrt(max(discriminant, 0.0))
    feasible_lo = max(lo, -projection - root)
    feasible_hi = min(hi, -projection + root)
    if feasible_lo > feasible_hi + eps:
      continue
    rho = np.linspace(feasible_lo, feasible_hi, n_range)
    samples.append(s + rho[:, None] * direction)
  if not samples:
    return np.empty((0, 2), dtype=float)
  return np.vstack(samples)

def sector_max_distance(
    cand: Sequence[Sequence[float]] | Arr,
    site: Sequence[float], bearing_deg: float,
    ranges: tuple[float, float] = (5.0, 1500.0),
    err_deg: float = 1.0,
) -> Arr:
  p = _pts(cand)
  s = _point(site, "site")
  lo, hi = ranges
  if (not math.isfinite(bearing_deg) or not math.isfinite(lo)
      or not math.isfinite(hi) or not math.isfinite(err_deg)
      or not 0 <= lo <= hi or not 0 <= err_deg < 90):
    raise ValueError("invalid cone ranges or angle error")
  ang = math.radians(bearing_deg)
  u = np.array([math.cos(ang), math.sin(ang)])
  v = np.array([-math.sin(ang), math.cos(ang)])
  d = p - s
  a, b = d @ u, d @ v
  err = math.radians(err_deg)
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
  s = _point(site1, "site1")
  if (not math.isfinite(recv_rad) or recv_rad <= 0
      or not math.isfinite(speed) or speed <= 0
      or not math.isfinite(measure_s) or measure_s < 0
      or not math.isfinite(switch_s) or switch_s < 0
      or math.isnan(time_limit_s)):
    raise ValueError("invalid reception or time parameters")
  far = sector_max_distance(p, s, bearing_deg, ranges, err_deg)
  t = (np.linalg.norm(p - s, axis=1) / speed + measure_s
       + switch_s * float(switched))
  return (far <= recv_rad + eps) & (t <= time_limit_s + eps)

def detected_candidate_mask(
    cand: Sequence[Sequence[float]] | Arr,
    site1: Sequence[float], bearing_deg: float,
    ranges: tuple[float, float] = (5.0, 1500.0),
    err_deg: float = 1.0, recv_min: float = 1000.0,
    time_limit_s: float = math.inf, speed: float = 5.0,
    measure_s: float = 5.0, switched: bool = False,
    switch_s: float = 1.0,
) -> Arr:
  p = _pts(cand)
  s = _point(site1, "site1")
  lo, hi = ranges
  if (not math.isfinite(recv_min) or recv_min <= 0
      or not math.isfinite(speed) or speed <= 0
      or not math.isfinite(measure_s) or measure_s < 0
      or not math.isfinite(switch_s) or switch_s < 0
      or math.isnan(time_limit_s)):
    raise ValueError("invalid reception or time parameters")
  if lo <= recv_min:
    far = sector_max_distance(
      p, s, bearing_deg, (lo, min(hi, recv_min)), err_deg)
    reception = far <= recv_min + eps
  else:
    far = sector_max_distance(p, s, bearing_deg, (lo, lo), err_deg)
    reception = far <= lo + eps
  time_s = (np.linalg.norm(p - s, axis=1) / speed + measure_s
            + switch_s * float(switched))
  return reception & (time_s <= time_limit_s + eps)

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
  s = _point(site1, "site1")
  if len(g) == 0 or not 0 <= quantile <= 1:
    raise ValueError("sources must be nonempty and quantile valid")
  if (not math.isfinite(sig_deg) or not math.isfinite(err_deg)
      or sig_deg <= 0 or err_deg <= 0):
    raise ValueError("angle scales must be finite and positive")
  if (not math.isfinite(recv_rad) or recv_rad <= 0
      or not math.isfinite(speed) or speed <= 0
      or not math.isfinite(measure_s) or measure_s < 0
      or not math.isfinite(switch_s) or switch_s < 0):
    raise ValueError("invalid reception or time parameters")
  first = s[None, :] - g
  r1 = np.linalg.norm(first, axis=1)
  second = p[:, None, :] - g[None, :, :]
  r2 = np.linalg.norm(second, axis=2)
  dot = np.sum(first[None, :, :] * second, axis=2)
  den = r1[None, :] * r2
  cos_val = np.abs(dot) / np.maximum(den, np.finfo(float).tiny)
  cos_val = np.clip(cos_val, 0.0, 1.0)
  cross_val = (first[None, :, 0] * second[:, :, 1]
               - first[None, :, 1] * second[:, :, 0])
  sin_val = np.abs(cross_val) / np.maximum(den, np.finfo(float).tiny)
  sin_val = np.clip(sin_val, 0.0, 1.0)
  unsafe = ((r1[None, :] <= np.finfo(float).tiny)
            | (r2 <= np.finfo(float).tiny)
            | (sin_val <= sin_angle_tol))
  raw = np.sqrt(r1[None, :] ** 2 + r2 * r2) / np.where(
    unsafe, math.nan, sin_val)
  raw[unsafe] = math.inf
  h1x = first[:, 1] / np.maximum(r1 * r1, np.finfo(float).tiny)
  h1y = -first[:, 0] / np.maximum(r1 * r1, np.finfo(float).tiny)
  h2x = second[:, :, 1] / np.maximum(r2 * r2, np.finfo(float).tiny)
  h2y = -second[:, :, 0] / np.maximum(r2 * r2, np.finfo(float).tiny)
  det = h1x[None, :] * h2y - h1y[None, :] * h2x
  safe_det = np.where(unsafe, math.nan, det)
  k1x = h2y / safe_det
  k1y = -h2x / safe_det
  k2x = -h1y[None, :] / safe_det
  k2y = h1x[None, :] / safe_det
  plus = np.hypot(k1x + k2x, k1y + k2y)
  minus = np.hypot(k1x - k2x, k1y - k2y)
  bound = math.radians(err_deg) * np.maximum(plus, minus)
  bound[~np.isfinite(bound)] = math.inf
  if quantile == 1.0:
    ang_cost = np.max(cos_val, axis=1)
    gdop_val = np.max(raw, axis=1)
    sampled_error = np.max(bound, axis=1)
  elif quantile == 0.0:
    ang_cost = np.min(cos_val, axis=1)
    gdop_val = np.min(raw, axis=1)
    sampled_error = np.min(bound, axis=1)
  else:
    ang_cost = np.quantile(cos_val, quantile, axis=1)
    gdop_val = np.quantile(raw, quantile, axis=1)
    sampled_error = np.quantile(bound, quantile, axis=1)
  rms = math.radians(sig_deg) * gdop_val
  coverage = np.mean(r2 <= recv_rad + eps, axis=1)
  detection_coverage = np.mean(
    r2 <= np.maximum(recv_rad, r1)[None, :] + eps, axis=1)
  time_s = (np.linalg.norm(p - s, axis=1) / speed + measure_s
            + switch_s * float(switched))
  return {
    "angle_cost": ang_cost,
    "gdop": gdop_val,
    "rms": rms,
    "sampled_error": sampled_error,
    "time_s": time_s,
    "coverage": coverage,
    "detection_coverage": detection_coverage,
  }

def _cos_interval(lo: float, hi: float) -> tuple[float, float]:
  values = [math.cos(lo), math.cos(hi)]
  first = math.ceil(lo / math.pi)
  last = math.floor(hi / math.pi)
  for index in range(first, last + 1):
    values.append(math.cos(index * math.pi))
  return min(values), max(values)

def _contains_sin_zero(lo: float, hi: float) -> bool:
  return math.ceil(lo / math.pi) <= math.floor(hi / math.pi)

def continuous_two_site_error_bound(
    site1: Sequence[float] | Arr, site2: Sequence[float] | Arr,
    bearing_deg: float, ranges: tuple[float, float] = (5.0, 1500.0),
    err_deg: float = 1.0, abs_tol_m: float = 1e-3,
    max_intervals: int = 100_000,
) -> dict[str, float | int | bool]:
  first_site = _point(site1, "site1")
  second_site = _point(site2, "site2")
  lo_range, hi_range = ranges
  if (not math.isfinite(bearing_deg) or not math.isfinite(err_deg)
      or not math.isfinite(lo_range) or not math.isfinite(hi_range)
      or not 0 <= err_deg < 90 or not 0 < lo_range <= hi_range):
    raise ValueError("invalid bearing sector")
  if (not math.isfinite(abs_tol_m) or abs_tol_m <= 0
      or max_intervals < 1):
    raise ValueError("invalid interval controls")
  delta = second_site - first_site
  baseline = float(np.linalg.norm(delta))
  if baseline <= np.finfo(float).tiny:
    return {"lower_m": math.inf, "upper_m": math.inf,
            "interval_count": 0, "certified": True}
  alpha = math.atan2(float(delta[1]), float(delta[0]))
  center = math.radians(bearing_deg)
  err = math.radians(err_deg)
  beta_lo = alpha - (center + err)
  beta_hi = alpha - (center - err)
  if _contains_sin_zero(beta_lo, beta_hi):
    return {"lower_m": math.inf, "upper_m": math.inf,
            "interval_count": 0, "certified": True}
  error_rad = err

  def exact(beta: float, source_range: float) -> float:
    cosine = math.cos(beta)
    sine = abs(math.sin(beta))
    if sine == 0.0:
      return math.inf
    range2 = (source_range * source_range + baseline * baseline
              - 2.0 * source_range * baseline * cosine)
    numerator = (range2
                 + source_range * abs(source_range - baseline * cosine))
    return error_rad * math.hypot(
      source_range, numerator / (baseline * sine))

  def point_value(beta: float) -> float:
    return max(exact(beta, lo_range), exact(beta, hi_range))

  def upper(box_lo: float, box_hi: float) -> float:
    cos_lo, cos_hi = _cos_interval(box_lo, box_hi)
    cos_lo = math.nextafter(cos_lo, -math.inf)
    cos_hi = math.nextafter(cos_hi, math.inf)
    sine_min = min(abs(math.sin(box_lo)), abs(math.sin(box_hi)))
    sine_min = max(math.nextafter(sine_min, 0.0), np.finfo(float).tiny)
    values = []
    for source_range in (lo_range, hi_range):
      range2_upper = (source_range * source_range + baseline * baseline
                      - 2.0 * source_range * baseline * cos_lo)
      abs_upper = max(
        abs(source_range - baseline * cos_lo),
        abs(source_range - baseline * cos_hi),
      )
      numerator_upper = range2_upper + source_range * abs_upper
      value = error_rad * math.hypot(
        source_range, numerator_upper / (baseline * sine_min))
      values.append(math.nextafter(value, math.inf))
    return max(values)

  midpoint = (beta_lo + beta_hi) / 2.0
  lower = max(point_value(beta_lo), point_value(midpoint),
              point_value(beta_hi))
  heap = [(-upper(beta_lo, beta_hi), beta_lo, beta_hi)]
  interval_count = 1
  while heap and -heap[0][0] > lower + abs_tol_m:
    if interval_count >= max_intervals:
      break
    _, box_lo, box_hi = heapq.heappop(heap)
    middle = (box_lo + box_hi) / 2.0
    lower = max(lower, point_value(middle))
    for left, right in ((box_lo, middle), (middle, box_hi)):
      heapq.heappush(heap, (-upper(left, right), left, right))
    interval_count += 1
  certified_upper = max(lower, -heap[0][0] if heap else lower)
  return {
    "lower_m": lower,
    "upper_m": certified_upper,
    "interval_count": interval_count,
    "certified": certified_upper <= lower + abs_tol_m,
  }

def pareto_mask(vals: Sequence[Sequence[float]] | Arr,
                 tol: float = eps) -> Arr:
  x = np.asarray(vals, dtype=float)
  tol = _relative_tol(tol)
  if x.ndim != 2 or len(x) == 0:
    raise ValueError("vals must have shape (n, m) with n positive")
  ok = np.all(np.isfinite(x), axis=1)
  ids = np.flatnonzero(ok)
  keep = np.zeros(len(x), dtype=bool)
  if len(ids) == 0:
    return keep
  scale = np.maximum(np.max(np.abs(x[ids]), axis=0),
                     np.finfo(float).tiny)
  cut = tol * scale
  for i in ids:
    dom = np.all(x[ids] <= x[i] + cut, axis=1)
    strict = np.any(x[ids] < x[i] - cut, axis=1)
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
    switched: bool = False, conditional_reception: bool = True,
) -> dict[str, Arr]:
  src = source_cone_samples(
    site1, bearing_deg, ranges, err_deg, n_range=n_range, n_angle=n_angle)
  res = second_site_metrics_sources(
    cand, site1, src, err_deg=err_deg, recv_rad=recv_rad,
    quantile=quantile,
    switched=switched)
  vals = np.column_stack((res["sampled_error"], res["time_s"]))
  if conditional_reception:
    guaranteed = detected_candidate_mask(
      cand, site1, bearing_deg, ranges, err_deg, recv_rad,
      switched=switched)
    sampled_coverage = res["detection_coverage"]
  else:
    guaranteed = robust_candidate_mask(
      cand, site1, bearing_deg, ranges, err_deg, recv_rad,
      switched=switched)
    sampled_coverage = res["coverage"]
  feasible = guaranteed & (sampled_coverage + eps >= min_coverage)
  safe_vals = vals.copy()
  safe_vals[~feasible] = math.inf
  res["guaranteed"] = guaranteed
  res["pareto"] = pareto_mask(safe_vals)
  return res
