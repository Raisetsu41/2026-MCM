from __future__ import annotations

import math

import numpy as np

from robot.q3_agent import Arr, ChannelTrack, Q3Agent
from robot.client import ApiError

def ordered_route(points: Arr, start: Arr) -> Arr:
  p = np.asarray(points, dtype=float)
  if p.ndim != 2 or p.shape[1] != 2 or len(p) == 0:
    raise ValueError("points must have shape (n, 2) with n positive")
  current = np.asarray(start, dtype=float)
  unused = list(range(len(p)))
  order: list[int] = []
  while unused:
    pick = min(unused, key=lambda i: (float(np.linalg.norm(p[i] - current)), i))
    order.append(pick)
    unused.remove(pick)
    current = p[pick]
  route = p[order].copy()
  improved = True
  while improved:
    improved = False
    for i in range(1, len(route) - 1):
      for j in range(i + 1, len(route)):
        before = float(np.linalg.norm(route[i - 1] - route[i]))
        after = float(np.linalg.norm(route[i - 1] - route[j]))
        if j + 1 < len(route):
          before += float(np.linalg.norm(route[j] - route[j + 1]))
          after += float(np.linalg.norm(route[i] - route[j + 1]))
        if after + 1e-9 < before:
          route[i:j + 1] = route[i:j + 1][::-1]
          improved = True
  return route

def q4_scan_sites(spacing: float = 990.0,
                  target_rad: float = 1800.0) -> Arr:
  if spacing <= 0 or spacing >= 1000.0 or target_rad <= 0:
    raise ValueError("invalid Q4 lattice parameters")
  e1 = np.array([spacing, 0.0])
  e2 = np.array([spacing / 2.0, math.sqrt(3.0) * spacing / 2.0])
  cutoff = target_rad + spacing
  limit = int(math.ceil(cutoff / spacing)) * 2 + 2
  points = []
  for i in range(-limit, limit + 1):
    for j in range(-limit, limit + 1):
      point = i * e1 + j * e2
      if np.linalg.norm(point) <= cutoff + 1e-9:
        points.append(point)
  return ordered_route(np.asarray(points), np.zeros(2))

def _distance_to_sector(points: Arr, lo: float, hi: float,
                        err_rad: float) -> Arr:
  x = points[:, 0]
  y = points[:, 1]
  radius = np.hypot(x, y)
  angle = np.arctan2(y, x)
  inside = np.abs(angle) <= err_rad
  radial = np.abs(radius - np.clip(radius, lo, hi))
  abs_y = np.abs(y)
  projection = x * math.cos(err_rad) + abs_y * math.sin(err_rad)
  edge_r = np.clip(projection, lo, hi)
  edge = np.hypot(
    x - edge_r * math.cos(err_rad),
    abs_y - edge_r * math.sin(err_rad),
  )
  return np.where(inside, radial, edge)

def bearing_clear_sites(
    site: Arr, bearing_deg: float, err_deg: float = 1.01,
    ranges: tuple[float, float] = (5.0, 1500.0),
    cover_rad: float = 20.0, start: Arr | None = None,
) -> Arr:
  origin = np.asarray(site, dtype=float)
  if origin.shape != (2,) or not np.all(np.isfinite(origin)):
    raise ValueError("site must be a finite point")
  lo, hi = ranges
  if not 0 <= lo < hi or not 0 <= err_deg < 90 or cover_rad <= 0:
    raise ValueError("invalid bearing sector parameters")
  spacing = cover_rad * math.sqrt(3.0) * 0.995
  e1 = np.array([spacing, 0.0])
  e2 = np.array([spacing / 2.0, math.sqrt(3.0) * spacing / 2.0])
  height = float(e2[1])
  limit_i = int(math.ceil((hi + cover_rad) / spacing)) + 4
  limit_j = int(math.ceil((hi * math.sin(math.radians(err_deg))
                           + cover_rad) / height)) + 4
  local = []
  for i in range(-limit_i, limit_i + 1):
    for j in range(-limit_j, limit_j + 1):
      point = i * e1 + j * e2
      local.append(point)
  local_points = np.asarray(local)
  keep = _distance_to_sector(
    local_points, lo, hi, math.radians(err_deg)) <= cover_rad + 1e-9
  local_points = local_points[keep]
  angle = math.radians(bearing_deg)
  rotation = np.array([
    [math.cos(angle), -math.sin(angle)],
    [math.sin(angle), math.cos(angle)],
  ])
  global_points = origin + local_points @ rotation.T
  route_start = origin if start is None else np.asarray(start, dtype=float)
  return ordered_route(global_points, route_start)

class Q4Agent(Q3Agent):

  def _scan_sites(self) -> Arr:
    return q4_scan_sites()

  def _lost_signal(self, track: ChannelTrack) -> None:
    first = track.obs[0]
    track.status = "clear_fallback"
    for site in bearing_clear_sites(
        first.site, first.bearing_deg, self.err_deg, start=self.pos):
      if self._clear(site, track, certified=False):
        return
    track.status = "geometry_fault"
    raise ApiError(f"directional fallback exhausted on channel {track.channel}")
