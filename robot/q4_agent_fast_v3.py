# Q4: 全局空清除预算, 大覆盖区域改用有误差上界的局部测站包.
from __future__ import annotations

import math

import numpy as np

from robot.agent_fast import FastTrack
from robot.agent_fast_v3 import LocalV3
from robot.client import ApiError
from robot.fast_geometry import Arr, cover_cells, definitely_disjoint, enclosing
from robot.fast_geometry_v3 import packet_localization_bound, packet_sites
from robot.q4_agent_fast import Q4FastAgent
from robot.q4_agent_fast_v2 import Q4FastAgentV2


class Q4FastAgentV3(LocalV3, Q4FastAgentV2):
  def __init__(self, *args, empty_limit: int = 5, **kwargs) -> None:
    if not 0 <= empty_limit <= 5:
      raise ValueError("empty_limit must be in 0..5")
    super().__init__(*args, **kwargs)
    self.empty_limit = empty_limit
    self.empty_calls = 0

  def _clear(self, track: FastTrack, site: Arr, near: bool = False,
             certified: bool = True) -> bool:
    if track.status == "cleared":
      return True
    if not certified and self.empty_calls >= self.empty_limit:
      if track.poly is None or not track.obs:
        raise ApiError("missing last-cell certificate")
      angle = math.radians(track.obs[0].bearing_deg)
      cells = cover_cells(track.poly, 19.5, angle)
      disks = track.failures + [site]
      if not all(any(np.max(np.linalg.norm(cell - p, axis=1)) <= 19.999
                     for p in disks) for cell in cells):
        raise ApiError("clear would exceed the mission-wide budget")
      # 全区域被失败圆与本站圆覆盖, 真源不在失败圆内, 所以本站必定成功.
    ok = super()._clear(track, site, near, certified)
    if not ok:
      self.empty_calls += 1
      if self.empty_calls > self.empty_limit:
        track.status = "geometry_fault"
        raise ApiError("last-cell clear contradicted its coverage certificate")
    return ok

  def _cover_fits(self, track: FastTrack) -> bool:
    if track.poly is None or not track.obs:
      return False
    angle = math.radians(track.obs[0].bearing_deg)
    count = len(cover_cells(track.poly, 19.5, angle))
    # 遍历 k 个覆盖单元, 最坏 k-1 次失败. 预算按整局计算, 不按频道重置.
    return count - 1 <= self.empty_limit - self.empty_calls

  def _prefer_cover(self, track: FastTrack) -> bool:
    return self._cover_fits(track) and super()._prefer_cover(track)

  def _coarse_packet(self, track: FastTrack) -> None:
    before, self.phase = self.phase, "coarse_packet"
    self.branches["coarse_packet"] += 1
    try:
      super()._coarse_packet(track)
    finally:
      self.phase = before

  def _finish_cover(self, track: FastTrack) -> None:
    if self._ready_clear(track):
      return
    if self._cover_fits(track):
      before, self.phase = self.phase, "budgeted_clear_cover"
      self.branches["budgeted_clear_cover"] += 1
      try:
        Q4FastAgent._finish_cover(self, track)
      finally:
        self.phase = before
      return
    if track.poly is None:
      raise ApiError("missing directional packet-cover region")
    before, self.phase = self.phase, "directional_packet_cover"
    self.branches["directional_packet_cover"] += 1
    try:
      angle = math.radians(track.obs[0].bearing_deg)
      radius = 19.5
      while packet_localization_bound(radius, self.err_deg) > 19.0:
        radius /= 2
        if radius < 0.01:
          raise ApiError("no finite packet localization certificate")
      cells = cover_cells(track.poly, radius, angle)
      centers = np.vstack([enclosing(cell)[0] for cell in cells])
      left = set(range(len(cells)))
      while left:
        if self._ready_clear(track):
          return
        i = min(left, key=lambda j: (float(np.linalg.norm(centers[j] - self.pos)), j))
        left.remove(i)
        if definitely_disjoint(cells[i], track.poly):
          continue
        sites = packet_sites(centers[i], radius, self.pos)
        for site in sites:
          self._sense(track, site)
          if self._ready_clear(track):
            return
      track.status = "geometry_fault"
      raise ApiError("certified directional packet coverage exhausted")
    finally:
      self.phase = before
