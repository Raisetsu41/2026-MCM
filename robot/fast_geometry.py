# 快速版的保守几何和路径辅助, 所有定位多边形都保持外包围.
from __future__ import annotations

import math

import numpy as np

from geometry_solver import bearing_halfplanes, minimum_enclosing_circle


Arr = np.ndarray
guard = 1e-6


def cut(poly: Arr, normal: Arr, limit: float) -> Arr:
  """裁剪 n.x <= limit, 向外放宽后再求交, 避免边界分类不一致."""
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
  """用圆的支撑半平面裁剪, 绝不用内接多边形缩小真实可行集."""
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
  # 复算全部顶点, 不把 MEC 内部容差直接带入清除证书.
  radius = float(np.max(np.linalg.norm(poly - center, axis=1))) + guard
  if not np.all(np.isfinite(center)) or not math.isfinite(radius):
    raise ValueError("nonfinite enclosing circle")
  return center, radius


def forecast(poly: Arr, center: Arr, radius: float,
             site: Arr, err: float) -> float:
  """名义测量仅给调度评分, 不能用来更新证据或认证清除."""
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
  """开放路径的最近邻加 2-opt, 首点也允许改变."""
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
  """沿首测方向二分外包围, 每个叶子都有半径证书."""
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
  """只在存在严格分离轴时丢弃覆盖单元, 退化线段宁可保留."""
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
