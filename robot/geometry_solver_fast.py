from __future__ import annotations

import math

import numpy as np

from geometry_solver import (bearing_halfplanes, convex_hull,
                             minimum_enclosing_circle, regular_bound)

Arr = np.ndarray
guard = 1e-6

def cut(poly: Arr, normal: Arr, limit: float) -> Arr:
  if len(poly) == 0:
    return poly.copy()
  normal = np.asarray(normal, dtype=float)
  size = float(np.linalg.norm(normal))
  if not math.isfinite(size) or size <= 0:
    raise ValueError("invalid clipping normal")
  n = normal / size
  origin = poly[0]
  local = poly - origin
  pad = max(guard, 64 * np.finfo(float).eps * (
    float(np.max(np.abs(poly))) + abs(limit / size)))
  bound = limit / size - float(n @ origin) + pad
  residual = local @ n - bound
  if np.all(residual <= 0):
    return poly.copy()
  if np.all(residual > 0):
    return np.empty((0, 2))
  out = []
  for i, a in enumerate(local):
    j = (i + 1) % len(local)
    fa, fb = float(residual[i]), float(residual[j])
    if fa <= 0:
      out.append(a + origin)
    if (fa <= 0) != (fb <= 0):
      t = min(1.0, max(0.0, fa / (fa - fb)))
      out.append(a + t * (local[j] - a) + origin)
  return np.asarray(out, dtype=float).reshape(-1, 2)

def cut_bearing(poly: Arr, site: Arr, angle: float, err: float) -> Arr:
  out = poly
  for hp in bearing_halfplanes(site, angle, err):
    out = cut(out, hp[:2], float(hp[2]))
  return out

def cut_disk(poly: Arr, center: Arr, radius: float) -> Arr:
  out = poly
  for angle in np.arange(128) * (2 * math.pi / 128):
    n = np.array([math.cos(angle), math.sin(angle)])
    out = cut(out, n, float(n @ center) + radius)
    if len(out) == 0:
      break
  return out

def enclosing(poly: Arr) -> tuple[Arr, float]:
  if len(poly) == 0 or not np.all(np.isfinite(poly)):
    raise ValueError("empty or nonfinite localization region")
  center, _ = minimum_enclosing_circle(poly)
  radius = float(np.max(np.linalg.norm(poly - center, axis=1))) + guard
  if not np.all(np.isfinite(center)) or not math.isfinite(radius):
    raise ValueError("nonfinite enclosing circle")
  return center, radius

def forecast(poly: Arr, center: Arr, radius: float,
             site: Arr, err: float) -> float:
  delta = center - site
  if np.linalg.norm(delta) <= max(5.0, radius * 0.15):
    return max(2.5, radius * 0.55)
  angle = math.degrees(math.atan2(delta[1], delta[0]))
  pred = cut_bearing(poly, site, angle, err)
  if len(pred) == 0:
    return radius
  _, value = enclosing(pred)
  return min(radius, value)

def path_length(points: Arr, start: Arr) -> float:
  if len(points) == 0:
    return 0.0
  p = np.vstack((start, points))
  return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())

def route_order(points: Arr, start: Arr) -> list[int]:
  if len(points) == 0:
    return []
  left = set(range(len(points)))
  order = []
  pos = start
  while left:
    i = min(left, key=lambda k: (float(np.linalg.norm(points[k] - pos)), k))
    order.append(i)
    left.remove(i)
    pos = points[i]
  for _ in range(12):
    changed = False
    for i in range(len(order) - 1):
      a = start if i == 0 else points[order[i - 1]]
      for j in range(i + 1, len(order)):
        b, c = points[order[i]], points[order[j]]
        before = float(np.linalg.norm(a - b))
        after = float(np.linalg.norm(a - c))
        if j + 1 < len(order):
          d = points[order[j + 1]]
          before += float(np.linalg.norm(c - d))
          after += float(np.linalg.norm(b - d))
        if after + 1e-6 < before:
          order[i:j + 1] = reversed(order[i:j + 1])
          changed = True
    if not changed:
      break
  return order

def insertion(center: Arr, start: Arr, remaining: Arr) -> tuple[int, float]:
  if len(remaining) == 0:
    return 0, float(np.linalg.norm(center - start))
  p = np.vstack((start, remaining))
  costs = [float(np.linalg.norm(center - p[i])
                 + np.linalg.norm(center - p[i + 1])
                 - np.linalg.norm(p[i + 1] - p[i]))
           for i in range(len(p) - 1)]
  costs.append(float(np.linalg.norm(center - p[-1])))
  i = min(range(len(costs)), key=lambda j: (costs[j], j))
  return i, costs[i]

def cover_cells(poly: Arr, radius: float, angle: float) -> list[Arr]:
  u = np.array([math.cos(angle), math.sin(angle)])
  v = np.array([-u[1], u[0]])
  stack = [poly.copy()]
  leaves = []
  while stack:
    p = stack.pop()
    _, r = enclosing(p)
    if r <= radius:
      leaves.append(p)
      continue
    spans = [float(np.ptp(p @ axis)) for axis in (u, v)]
    axis = u if spans[0] >= spans[1] else v
    proj = p @ axis
    mid = (float(proj.min()) + float(proj.max())) / 2
    left = cut(p, axis, mid)
    right = cut(p, -axis, -mid)
    if len(left) == 0 or len(right) == 0 or max(spans) < 1e-4:
      raise ValueError("cover subdivision failed to progress")
    stack.extend((right, left))
  return leaves

def definitely_disjoint(first: Arr, second: Arr) -> bool:
  if len(first) == 0 or len(second) == 0:
    return True
  axes = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
  for p in (first, second):
    for edge in np.roll(p, -1, axis=0) - p:
      size = float(np.linalg.norm(edge))
      if size > guard:
        axes.append(np.array([-edge[1], edge[0]]) / size)
  for n in axes:
    a, b = first @ n, second @ n
    if float(a.max()) < float(b.min()) - guard:
      return True
    if float(b.max()) < float(a.min()) - guard:
      return True
  return False

def inside_hull(hull: Arr, points: Arr) -> bool:
  if len(hull) < 3:
    return False
  edges = np.roll(hull, -1, axis=0) - hull
  lengths = np.linalg.norm(edges, axis=1)
  if np.any(lengths <= 1e-9):
    return False
  offset = points[:, None, :] - hull[None, :, :]
  signed = (edges[None, :, 0] * offset[:, :, 1]
            - edges[None, :, 1] * offset[:, :, 0]) / lengths[None, :]
  return bool(np.all(signed >= 1e-6))

def direction_certificate(sites: Arr) -> tuple[bool, int]:
  boundary = regular_bound(1800.0, 72, outer=True)
  stack = [(np.vstack((np.zeros(2), boundary[i], boundary[(i + 1) % 72])), 0)
           for i in range(72)]
  count = 0
  while stack:
    cell, depth = stack.pop()
    count += 1
    distances = np.linalg.norm(cell[:, None, :] - sites[None, :, :], axis=2)
    eligible = sites[np.max(distances, axis=0) <= 994.99]
    if len(eligible) >= 3:
      hull = convex_hull(eligible, tol=0.0)
      if inside_hull(hull, cell):
        continue
    if depth >= 22 or count >= 60000:
      return False, count
    lengths = np.linalg.norm(np.roll(cell, -1, axis=0) - cell, axis=1)
    i = int(np.argmax(lengths))
    j, k = (i + 1) % 3, (i + 2) % 3
    middle = (cell[i] + cell[j]) / 2.0
    stack.append((np.vstack((cell[i], middle, cell[k])), depth + 1))
    stack.append((np.vstack((middle, cell[j], cell[k])), depth + 1))
  return True, count

def compact_q4_sites() -> Arr:
  inner_r, outer_r = 975.0, 1850.0
  a = np.arange(7) * (2 * math.pi / 7)
  b = np.arange(14) * (2 * math.pi / 14)
  inner = inner_r * np.column_stack((np.cos(a), np.sin(a)))
  outer = outer_r * np.column_stack((np.cos(b), np.sin(b)))
  return np.vstack((np.zeros(2), inner, outer[np.r_[12, 13, np.arange(12)]]))

def covered_by_negative_disks(poly: Arr, negatives: Arr) -> bool:
  stack = [(poly, 0)]
  count = 0
  while stack:
    cell, depth = stack.pop()
    count += 1
    far = np.max(np.linalg.norm(cell[:, None, :] - negatives[None, :, :], axis=2),
                 axis=0)
    if np.any(far <= 999.99):
      continue
    if depth >= 9 or count >= 80:
      return False
    center = cell.mean(axis=0)
    if np.min(np.linalg.norm(negatives - center, axis=1)) >= 999.99:
      return False
    horizontal = np.ptp(cell[:, 0]) >= np.ptp(cell[:, 1])
    axis = np.array([1.0, 0.0]) if horizontal else np.array([0.0, 1.0])
    projection = cell @ axis
    middle = (float(projection.min()) + float(projection.max())) / 2.0
    for normal, limit in ((axis, middle), (-axis, -middle)):
      part = cut(cell, normal, limit)
      if len(part):
        stack.append((part, depth + 1))
  return True

def unknown_negative_implied(site: Arr, negatives: list[Arr]) -> bool:
  if not negatives:
    return False
  old = np.asarray(negatives)
  if np.min(np.linalg.norm(old - site, axis=1)) <= 1e-7:
    return True
  if len(old) < 2:
    return False
  poly = cut_disk(regular_bound(1800.0, 72, outer=True), site, 1500.0)
  for previous in old:
    delta = previous - site
    limit = float(delta @ site + 0.5 * (delta @ delta))
    poly = cut(poly, delta, limit)
    if len(poly) == 0:
      return True
  return covered_by_negative_disks(poly, old)

def route_cost(order: list[int], distances: Arr) -> float:
  previous = len(distances) - 1
  total = 0.0
  for i in order:
    total += float(distances[previous, i])
    previous = i
  return total

def exact_open(distances: Arr) -> list[int]:
  n = len(distances) - 1
  dp = np.full((1 << n, n), math.inf)
  parent = np.full((1 << n, n), -1, dtype=np.int16)
  for i in range(n):
    dp[1 << i, i] = distances[n, i]
  for mask in range(1, 1 << n):
    for last in range(n):
      if not mask & (1 << last) or not math.isfinite(float(dp[mask, last])):
        continue
      for nxt in range(n):
        if mask & (1 << nxt):
          continue
        new_mask = mask | (1 << nxt)
        value = float(dp[mask, last] + distances[last, nxt])
        if value < dp[new_mask, nxt]:
          dp[new_mask, nxt] = value
          parent[new_mask, nxt] = last
  mask = (1 << n) - 1
  last = int(np.argmin(dp[mask]))
  order = []
  while last >= 0:
    order.append(last)
    previous = int(parent[mask, last])
    mask ^= 1 << last
    last = previous
  return order[::-1]

def improve_open(order: list[int], distances: Arr) -> list[int]:
  out = order.copy()
  n = len(out)
  start = n
  for _ in range(8):
    changed = False
    for i in range(n - 1):
      a = start if i == 0 else out[i - 1]
      for j in range(i + 1, n):
        b, c = out[i], out[j]
        before = float(distances[a, b])
        after = float(distances[a, c])
        if j + 1 < n:
          d = out[j + 1]
          before += float(distances[c, d])
          after += float(distances[b, d])
        if after + 1e-6 < before:
          out[i:j + 1] = out[i:j + 1][::-1]
          changed = True
    best_cost = route_cost(out, distances)
    best = None
    for i in range(n):
      reduced = out[:i] + out[i + 1:]
      for j in range(n):
        candidate = reduced[:j] + [out[i]] + reduced[j:]
        value = route_cost(candidate, distances)
        if value + 1e-6 < best_cost:
          best_cost, best = value, candidate
    if best is not None:
      out = best
      changed = True
    if not changed:
      break
  return out

def joint_route(points: Arr, start: Arr, station_count: int) -> list[int]:
  n = len(points)
  if n == 0:
    return []
  p = np.vstack((points, start))
  distances = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
  if n <= 10:
    return exact_open(distances)
  incumbent = list(range(station_count))
  for i in range(station_count, n):
    candidates = [incumbent[:j] + [i] + incumbent[j:]
                  for j in range(len(incumbent) + 1)]
    incumbent = min(candidates, key=lambda order: route_cost(order, distances))
  seeds = [incumbent]
  for first in np.argsort(distances[n, :n])[:3]:
    order = [int(first)]
    left = set(range(n)) - {int(first)}
    while left:
      i = min(left, key=lambda j: (float(distances[order[-1], j]), j))
      order.append(i)
      left.remove(i)
    seeds.append(order)
  best = incumbent
  best_cost = route_cost(best, distances)
  for seed in seeds:
    order = improve_open(seed, distances)
    value = route_cost(order, distances)
    if value + 1e-6 < best_cost:
      best, best_cost = order, value
  if sorted(best) != list(range(n)):
    raise ValueError("route lost a required node")
  return best

def bearing_radius_bound(poly: Arr, site: Arr, err: float = 1.01,
                         step: float = 2.0) -> float:
  if not 0 < step <= 10 or not 0 < err < 10:
    raise ValueError("invalid bearing enclosure parameters")
  if len(poly) == 0 or not np.all(np.isfinite(poly)) or not np.all(np.isfinite(site)):
    raise ValueError("invalid bearing enclosure geometry")
  delta = poly - site
  angles = np.sort(np.mod(np.arctan2(delta[:, 1], delta[:, 0]), 2 * math.pi))
  gaps = np.diff(np.r_[angles, angles[0] + 2 * math.pi])
  k = int(np.argmax(gaps))
  span = 2 * math.pi - float(gaps[k])
  if span < math.pi - 1e-8 and np.min(np.linalg.norm(delta, axis=1)) > 1e-7:
    lo = math.degrees(float(angles[(k + 1) % len(angles)])) - err - 1e-7
    hi = lo + math.degrees(span) + 2 * err + 2e-7
  else:
    lo, hi = 0.0, 360.0
  count = max(1, math.ceil((hi - lo) / step))
  width = (hi - lo) / count
  worst = 0.0
  for i in range(count):
    angle = lo + (i + 0.5) * width
    outer = cut_bearing(poly, site, angle, err + width / 2 + 1e-7)
    if len(outer):
      worst = max(worst, enclosing(outer)[1])
  return worst + 1e-3

def visible_kernel(poly: Arr, positives: Arr) -> Arr:
  if len(positives) < 2 or len(poly) == 0:
    return np.empty((0, 2))
  hulls = [convex_hull(np.vstack((positives, vertex)), tol=0.0) for vertex in poly]
  if any(len(hull) < 3 for hull in hulls):
    return np.empty((0, 2))
  kernel = hulls[0].copy()
  for hull in hulls:
    for a, b in zip(hull, np.roll(hull, -1, axis=0)):
      edge = b - a
      length = float(np.linalg.norm(edge))
      if length <= 1e-8:
        return np.empty((0, 2))
      normal = np.array([edge[1], -edge[0]]) / length
      kernel = cut(kernel, normal, float(normal @ a) - 1e-4)
      if len(kernel) == 0:
        return kernel
  for hull in hulls:
    for a, b in zip(hull, np.roll(hull, -1, axis=0)):
      edge = b - a
      cross = edge[0] * (kernel[:, 1] - a[1]) - edge[1] * (kernel[:, 0] - a[0])
      if np.min(cross) < 1e-5 * np.linalg.norm(edge):
        return np.empty((0, 2))
  return kernel

def packet_sites(center: Arr, radius: float, start: Arr) -> Arr:
  if not 0 <= radius <= 300.0:
    raise ValueError("packet radius exceeds its receiving certificate")
  size = 1.2 * radius + 8.0
  phase = math.atan2(start[1] - center[1], start[0] - center[0])
  angles = phase + np.arange(12) * math.pi / 6.0
  return center + size * np.column_stack((np.cos(angles), np.sin(angles)))

def packet_localization_bound(radius: float, err: float = 1.01) -> float:
  size = 1.2 * radius + 8.0
  h, w = size * math.cos(math.pi / 12), size * math.sin(math.pi / 12)
  low = 2 * math.atan2(w, h + radius)
  high = 2 * math.atan2(w, h - radius)
  error = math.radians(err)
  gap = min(low - 2 * error, math.pi - high - 2 * error)
  if gap <= 0:
    return math.inf
  factor = math.sin(error) / math.sin(gap / 2)
  if factor >= 1:
    return math.inf
  return 2 * factor * (size + radius) / (1 - factor) + 1e-3
