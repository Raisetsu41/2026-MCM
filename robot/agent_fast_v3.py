# 保留 v2 路由与全部证书, 仅替换局部测站决策并记录实际分支.
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from robot.agent_fast import FastTrack
from robot.agent_fast_v2 import Q3FastAgentV2
from robot.fast_geometry import Arr
from robot.fast_geometry_v3 import bearing_radius_bound, packet_sites, visible_kernel


class LocalV3:
  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.phase = "startup"
    self.branches: Counter[str] = Counter()
    self.last_bound: float | None = None

  def _certify(self) -> None:
    found = sum(bool(t.obs) or t.status == "cleared" for t in self.tracks.values())
    if found == 16 and any(t.status == "unknown" for t in self.tracks.values()):
      self.branches["found_16_stop"] += 1
    super()._certify()

  def _scan_station(self, index: int) -> None:
    before, self.phase = self.phase, "scan"
    self.branches["scan_station"] += 1
    try:
      super()._scan_station(index)
    finally:
      self.phase = before

  def _localize(self, track: FastTrack) -> None:
    before, self.phase = self.phase, "localize"
    self.branches["localize"] += 1
    try:
      super()._localize(track)
    finally:
      self.phase = before

  def _harvest(self, site: Arr, excluded: int = 0) -> None:
    before, self.phase = self.phase, "harvest"
    try:
      super()._harvest(site, excluded)
    finally:
      self.phase = before

  def _choose_probe(self, track: FastTrack) -> Arr | None:
    base = super()._choose_probe(track)
    self.last_bound = None
    if track.center is None or track.poly is None or track.radius > 450.0:
      return base
    center, radius = track.center, track.radius
    base_distance = math.inf if base is None else float(
      np.linalg.norm(base - self.pos) + max(0.0, np.linalg.norm(base - center) - 20.0))
    candidates: list[Arr] = []
    if self.directional:
      positives = np.asarray([obs.site for obs in track.obs])
      kernel = visible_kernel(track.poly, positives)
      if len(kernel):
        candidates.extend(kernel)
        candidates.append(kernel.mean(axis=0))
        # 核的凸组合仍在核内, 沿真实可见方向接近可行区域.
        close = kernel[int(np.argmin(np.linalg.norm(kernel - center, axis=1)))]
        candidates.append(0.8 * close + 0.2 * kernel.mean(axis=0))
    else:
      if base is not None:
        candidates.append(base)
      phase = math.atan2(self.pos[1] - center[1], self.pos[0] - center[0])
      angles = phase + np.arange(8) * math.pi / 4.0
      size = max(25.0, 1.2 * radius + 8.0)
      candidates.extend(center + size * np.column_stack((np.cos(angles), np.sin(angles))))
      candidates.append(center.copy())
    choices = []
    for site in candidates:
      if self._cached(track, site) is not None:
        continue
      if not self.directional and not self._guaranteed_signal(track, site):
        continue
      if self._forecast(track, site) >= 0.85 * radius:
        continue
      travel = float(np.linalg.norm(site - self.pos))
      onward = max(0.0, float(np.linalg.norm(site - center)) - 20.0)
      # 冻结同一个名义中心比较, 不用更长的进出路程换取更漂亮的预测.
      if travel + onward > base_distance + 1e-6:
        continue
      nominal = (travel + onward + 2 * self._forecast(track, site)) / 5.0
      choices.append((nominal, travel, onward, site))
    best, value = None, math.inf
    for _, travel, onward, site in sorted(choices, key=lambda row: row[0])[:4]:
      bound = bearing_radius_bound(track.poly, site, self.err_deg)
      if bound > 19.5 and bound >= 0.75 * radius:
        continue
      score = (travel + onward + 2 * bound) / 5.0
      score += 6 * max(0.0, math.log2(max(bound, 20.0) / 20.0))
      if score < value:
        best, value, self.last_bound = site.copy(), score, bound
    if best is not None:
      self.branches["bounded_probe"] += 1
      return best
    return base

  def _close_packet(self, track: FastTrack) -> None:
    if track.center is None or track.radius > 300.0:
      return
    before, self.phase = self.phase, "small_packet"
    self.branches["small_packet"] += 1
    old = track.radius
    sites = packet_sites(track.center.copy(), old, self.pos)
    try:
      # 从最近角开始, 相邻弦依次遍历, 路长有闭式上界.
      for site in sites:
        if self._ready_clear(track):
          return
        self._sense(track, site)
        if track.status == "cleared" or track.radius < 0.55 * old:
          return
    finally:
      self.phase = before

  def _finish_cover(self, track: FastTrack) -> None:
    before, self.phase = self.phase, "near_cover"
    self.branches["near_cover"] += 1
    try:
      super()._finish_cover(track)
    finally:
      self.phase = before


class Q3FastAgentV3(LocalV3, Q3FastAgentV2):
  """Q3 不引入试探清除, v2 的发现和不存在证书原样保留."""
