# v3: 全测向结果包围, 保持可见性的测站核, 小半径方向完备测站包.
from __future__ import annotations

import math

import numpy as np

from geometry_solver import convex_hull
from robot.fast_geometry import Arr, cut, cut_bearing, enclosing


def bearing_radius_bound(poly: Arr, site: Arr, err: float = 1.01,
                         step: float = 2.0) -> float:
  """将所有可能返回角分箱, 每箱用扩大后的锥外包围, 不以名义角代替上界."""
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
  # 原裁剪器有向外的距离保护, 窄锥尖端的位移比宽锥稍大, 额外留 1 mm.
  return worst + 1e-3


def visible_kernel(poly: Arr, positives: Arr) -> Arr:
  """交集 K = intersect_G conv(positives union {G}), 只保留严格内部."""
  # 支撑函数 min_G max(h_positive, u.G) 的极小值在 poly 顶点取得.
  # X 在这个核内时, X-G 是已接收向量和零向量的凸组合:
  # 既不越过未知定向半平面, 也不超过该源的实际未知接收半径.
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
  # cut 本身向外放宽, 最后再逐边验证内缩量, 不把浮点容差当成可见证书.
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
  # size*cos(30 deg) > radius: 任意朝向的可见弧宽严格大于两个站间角.
  # 所有站到任意可行源至多 2.2*radius+8 <= 668 m.
  return center + size * np.column_stack((np.cos(angles), np.sin(angles)))


def packet_localization_bound(radius: float, err: float = 1.01) -> float:
  """目标在半径 radius 单元内时, 十二点包结束后的全局位置误差上界."""
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
  # 可见相邻两站的测向矩阵满足 ||H^-1||_(inf->2) <= 1/sin(gap/2).
  # 任意可行点的位置误差 E <= factor*(2*(size+radius)+E), 移项得到此界.
  return 2 * factor * (size + radius) / (1 - factor) + 1e-3
