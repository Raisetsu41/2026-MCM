# v2: 连续域方向覆盖认证, 阴性证据推理, 固定节点集上的开放路径优化.
from __future__ import annotations

import math

import numpy as np

from geometry_solver import convex_hull, regular_bound
from robot.fast_geometry import Arr, cut, cut_disk


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
  """认证包含目标圆的外接多边形, 未解决的单元一律拒绝发证书."""
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
        # 距离的凸性保证整个三角单元距 eligible 中每个点都 < 995 m.
        # 单元顶点都在该凸包内部, 所以整个单元满足方向完备条件.
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
  # I_6 与 O_12 同向, 接续距离为 875 m. 两环都不用闭环返回.
  return np.vstack((np.zeros(2), inner, outer[np.r_[12, 13, np.arange(12)]]))


def covered_by_negative_disks(poly: Arr, negatives: Arr) -> bool:
  """只能证明覆盖才返回 True, 预算耗尽等同无法推理."""
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
    # 中心未被覆盖意味着这个外包围尚不能完成认证, 正常测量即可.
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
  """Q3: 假定本站能接收, 与既有真实阴性观测联立后检验是否矛盾."""
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
    # 接收于 site 且不接收于 previous, 必须 d(site,G) < d(previous,G).
    poly = cut(poly, delta, limit)
    if len(poly) == 0:
      return True
  # 真实阴性点的 1000 m 圆内没有该源, 这里只使用连续域覆盖证明.
  return covered_by_negative_disks(poly, old)


def route_cost(order: list[int], distances: Arr) -> float:
  previous = len(distances) - 1
  total = 0.0
  for i in order:
    total += float(distances[previous, i])
    previous = i
  return total


def exact_open(distances: Arr) -> list[int]:
  """小节点集的 Held-Karp, 固定起点, 自由终点."""
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
  """同一冻结节点集上不劣于固定扫描顺序加最便宜插入, 不宣称在线最优."""
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
