# Q4: 双环 25 点方向完备扫描, 多站交会和局部覆盖兜底.
from __future__ import annotations

import math

import numpy as np

from robot.agent_fast import FastTrack, Q3FastAgent
from robot.client import ApiClient, ApiError
from robot.fast_geometry import (
  Arr, cover_cells, definitely_disjoint, enclosing, path_length, route_order,
)
from robot.q4_agent import q4_scan_sites


def q4_ring_sites() -> Arr:
  """36 个小三角形覆盖目标圆域, 每条边小于 1000 m."""
  inner_r, outer_r = 980.0, 1870.0
  angle = np.arange(12) * math.pi / 6.0
  inner = inner_r * np.column_stack((np.cos(angle), np.sin(angle)))
  outer = outer_r * np.column_stack((
    np.cos(angle + math.pi / 12.0), np.sin(angle + math.pi / 12.0)))
  cross_edge = math.sqrt(inner_r**2 + outer_r**2
                         - 2 * inner_r * outer_r * math.cos(math.pi / 12.0))
  max_edge = max(inner_r, 2 * inner_r * math.sin(math.pi / 12.0),
                 2 * outer_r * math.sin(math.pi / 12.0), cross_edge)
  if max_edge >= 999.0 or outer_r * math.cos(math.pi / 12.0) <= 1800.0:
    raise ValueError("direction-complete ring certificate failed")
  # 记内环为 I_i, 外环为 O_i. 网格三角形分别为:
  # (0,I_i,I_{i+1}), (I_i,O_i,I_{i+1}), (O_i,O_{i+1},I_{i+1}).
  # 三角形任意一点到其三个顶点都不超过最长边. 圆域位于网格内部,
  # 因此任意源的 1000 m 邻域内测站的凸包包含该源, 与发射朝向无关.
  # 源在网格边或顶点上时取相邻三角形并集, 仍有严格朝前的测站.
  # 双环各走 11 条边, 从 I_11 接 O_11, 不增加闭环返回边.
  return np.vstack((np.zeros(2), inner, outer[np.r_[11, np.arange(11)]]))


class Q4FastAgent(Q3FastAgent):
  directional = True

  def __init__(self, client: ApiClient, err_deg: float = 1.01,
               opportunistic: bool = True, inline: bool = True,
               scan: str = "rings") -> None:
    if scan not in {"rings", "lattice"}:
      raise ValueError("Q4 scan must be rings or lattice")
    self.scan = scan
    super().__init__(client, err_deg, opportunistic, inline)

  def _scan_sites(self) -> Arr:
    return q4_ring_sites() if self.scan == "rings" else q4_scan_sites()

  def _coarse_packet(self, track: FastTrack) -> None:
    """沿当前多边形的长轴铺两排测站, 信号丢失后仍能从别的方位接近."""
    if track.poly is None:
      raise ApiError("missing directional region")
    origin = track.obs[0].site
    angle = math.radians(track.obs[0].bearing_deg)
    u = np.array([math.cos(angle), math.sin(angle)])
    v = np.array([-u[1], u[0]])
    local = track.poly - origin
    xs, ys = local @ u, local @ v
    lo, hi = float(xs.min()) - 60.0, float(xs.max()) + 60.0
    bottom, top = float(ys.min()) - 160.0, float(ys.max()) + 160.0
    # 首次 1.01 度锥在 1500 m 内的横宽不到 53 m. 即便放宽到 2 度,
    # 这里的矩形单元对角线仍远小于 1000 m, 提供连续域发现保证.
    nx = max(1, int(math.ceil((hi - lo) / 450.0)))
    ny = max(1, int(math.ceil((top - bottom) / 450.0)))
    sites = np.array([origin + x * u + y * v
                      for x in np.linspace(lo, hi, nx + 1)
                      for y in np.linspace(bottom, top, ny + 1)])
    for i in route_order(sites, self.pos):
      if self._ready_clear(track) or track.radius <= 180.0:
        return
      self._sense(track, sites[i])
      self._harvest(sites[i], excluded=track.channel)

  def _localize(self, track: FastTrack) -> None:
    for _ in range(3):
      if self._ready_clear(track):
        return
      site = self._choose_probe(track)
      if site is None:
        break
      old = track.radius
      body = self._sense(track, site)
      self._harvest(site, excluded=track.channel)
      if track.status == "cleared":
        return
      if body.get("measure_result") == "no_signal" or track.radius > 0.9 * old:
        break
    if self._ready_clear(track):
      return
    if track.radius > 300.0:
      self._coarse_packet(track)
    for _ in range(8):
      if self._ready_clear(track):
        return
      if track.radius > 300.0:
        break
      if self._prefer_cover(track):
        break
      self._close_packet(track)
    self._finish_cover(track)

  def _prefer_cover(self, track: FastTrack) -> bool:
    if track.poly is None or track.center is None or track.radius > 150.0:
      return False
    angle = math.radians(track.obs[0].bearing_deg)
    cells = cover_cells(track.poly, 19.5, angle)
    centers = np.vstack([enclosing(cell)[0] for cell in cells])
    order = route_order(centers, self.pos)
    cover_time = path_length(centers[order], self.pos) / 5.0
    cover_time += 3.0 * (len(cells) - 1) + 5.0
    ring_r = 2.0 * track.radius + 40.0
    approach = abs(float(np.linalg.norm(self.pos - track.center)) - ring_r)
    # 覆盖时耗取全遍历上界, 测站包只作调度估算, 不拿估算发证书.
    packet_time = (approach + 2.0 * ring_r) / 5.0 + 15.0
    return cover_time <= packet_time

  def _finish_cover(self, track: FastTrack) -> None:
    if self._ready_clear(track):
      return
    if track.poly is None:
      raise ApiError("missing directional fallback region")
    angle = math.radians(track.obs[0].bearing_deg)
    # 覆盖所有测向和距离约束的交集, 不重新覆盖第一次的完整扇形.
    cells = cover_cells(track.poly, 19.5, angle)
    centers = np.vstack([enclosing(cell)[0] for cell in cells])
    for i in route_order(centers, self.pos):
      cell = cells[i]
      if definitely_disjoint(cell, track.poly):
        continue
      # 只有整个单元都被已失败的清除圆覆盖时才跳过, 不删除部分重叠单元.
      if any(np.max(np.linalg.norm(cell - old, axis=1)) <= 19.999
             for old in track.failures):
        continue
      if any(np.linalg.norm(centers[i] - old) <= 1e-7 for old in track.failures):
        continue
      if self._clear(track, centers[i], certified=False):
        return
    track.status = "geometry_fault"
    raise ApiError(f"directional coverage exhausted on channel {track.channel}")
