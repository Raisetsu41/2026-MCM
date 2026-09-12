# Q3 v2: 全部剩余任务联合选路, 合并站内检测, 仅凭严格阴性推理跳过检测.
from __future__ import annotations

import numpy as np

from robot.agent import MissionResult
from robot.agent_fast import FastTrack, Q3FastAgent
from robot.client import ApiClient, ApiError
from robot.fast_geometry import Arr
from robot.fast_geometry_v2 import joint_route, unknown_negative_implied


class Q3FastAgentV2(Q3FastAgent):
  def __init__(self, client: ApiClient, err_deg: float = 1.01,
               opportunistic: bool = True, inline: bool = True) -> None:
    super().__init__(client, err_deg, opportunistic, inline)
    self.todo = set(range(len(self.sites)))
    self.future = list(range(len(self.sites)))
    self.inference_cache: dict[tuple[int, int, int], bool] = {}
    self.inferred_negative_count = 0
    self.next_channel = 0

  def _inferred_at(self, track: FastTrack, index: int) -> bool:
    if self.directional:
      return False
    key = (track.channel, len(track.negatives), index)
    if key not in self.inference_cache:
      self.inference_cache[key] = unknown_negative_implied(
        self.sites[index], track.negatives)
    return self.inference_cache[key]

  def _retire_stations(self) -> None:
    self._certify()
    unknown = [t for t in self.tracks.values() if t.status == "unknown"]
    if not unknown:
      self.todo.clear()
      return
    for index in sorted(self.todo):
      bit = 1 << index
      for track in unknown:
        if not track.station_mask & bit and self._inferred_at(track, index):
          # 推理单独计数, 不伪造响应, 不追加虚构测量到 history/negatives.
          track.station_mask |= bit
          self.inferred_negative_count += 1
      if all(t.station_mask & bit for t in unknown):
        self.todo.remove(index)
    self._certify()

  def _known_negative(self, track: FastTrack, site: Arr) -> bool:
    if track.center is None or track.poly is None:
      return False
    if np.linalg.norm(site - track.center) - track.radius > 1500.01:
      return True
    if self.directional:
      return False
    for previous in track.negatives:
      delta = previous - site
      value = float(delta @ delta) - 2.0 * ((track.poly - site) @ delta)
      if float(np.max(value)) < -1e-5:
        return True
    return False

  def _opportunity_jobs(self, site: Arr, excluded: int = 0) -> list[FastTrack]:
    if not self.opportunistic:
      return []
    jobs = []
    future = [self.sites[i] for i in self.future
              if i in self.todo and np.linalg.norm(self.sites[i] - site) > 1e-7]
    for track in self.tracks.values():
      if (track.channel == excluded or track.center is None or track.poly is None
          or track.status in {"unknown", "cleared", "absent_certified"}
          or track.radius <= 19.999 or track.opportunities >= 5
          or self._cached(track, site) is not None or self._known_negative(track, site)):
        continue
      pred = self._forecast(track, site)
      saving = 2.0 * max(0.0, track.radius - pred) / 5.0
      cost = 5.0 + float(track.channel != self.current_channel) + float(excluded != 0)
      if pred > 19.5 and saving < 1.5 * cost:
        continue
      if pred > 19.5 and pred >= 0.68 * track.radius:
        continue
      # 单频道收尾时减少跳出去再跳回来的弱收益测量.
      if excluded and pred > 19.5 and pred >= 0.35 * track.radius:
        continue
      defer = False
      for later in future[:3]:
        if self._cached(track, later) is not None or self._known_negative(track, later):
          continue
        if (np.linalg.norm(later - track.center) + 100.0
            < np.linalg.norm(site - track.center)
            and self._forecast(track, later) <= pred + 1.0):
          defer = True
          break
      if not defer:
        jobs.append(track)
    return jobs

  def _order_jobs(self, jobs: list[FastTrack]) -> list[FastTrack]:
    def key(track: FastTrack) -> tuple[int, int]:
      if track.channel == self.current_channel:
        return 0, track.channel
      if track.channel == self.next_channel:
        return 2, track.channel
      return 1, track.channel
    return sorted(jobs, key=key)

  def _clear_here(self, track: FastTrack, site: Arr) -> None:
    if track.status != "cleared" and track.poly is not None:
      if np.max(np.linalg.norm(track.poly - site, axis=1)) <= 19.999:
        self._clear(track, site)

  def _harvest(self, site: Arr, excluded: int = 0) -> None:
    for track in self._order_jobs(self._opportunity_jobs(site, excluded)):
      track.opportunities += 1
      self._sense(track, site)
      self._clear_here(track, site)

  def _scan_station(self, index: int) -> None:
    site = self.sites[index]
    bit = 1 << index
    mandatory = [t for t in self.tracks.values()
                 if t.status == "unknown" and not t.station_mask & bit]
    optional = self._opportunity_jobs(site)
    required_channels = {t.channel for t in mandatory}
    # 把 unknown 和已发现频道合在一次站内排程中, 当前频道优先, 不重复测量.
    for track in self._order_jobs(mandatory + optional):
      if track.status in {"cleared", "absent_certified"}:
        continue
      if track.channel not in required_channels:
        track.opportunities += 1
      body = self._sense(track, site)
      if track.channel in required_channels and body.get("measure_result") == "no_signal":
        track.station_mask |= bit
      self._clear_here(track, site)
      self._certify()

  def _plan(self) -> tuple[str, int] | None:
    self._retire_stations()
    indices = [i for i in self.future if i in self.todo]
    indices.extend(i for i in sorted(self.todo) if i not in indices)
    nodes: list[tuple[str, int]] = [("scan", i) for i in indices]
    points = [self.sites[i] for i in indices]
    for track in self.tracks.values():
      if (track.center is None
          or track.status in {"unknown", "cleared", "absent_certified"}):
        continue
      if self.todo and (not self.inline or track.radius > 300.0):
        continue
      # 有必经站明显更接近且能预测达到清除精度时, 先等那一条真实观测.
      if self.todo and track.radius > 19.999 and any(
          self._cached(track, self.sites[i]) is None
          and not self._known_negative(track, self.sites[i])
          and self._forecast(track, self.sites[i]) <= 19.5
          and np.linalg.norm(self.sites[i] - track.center) + 100.0
          < np.linalg.norm(self.pos - track.center) for i in indices[:3]):
        continue
      nodes.append(("service", track.channel))
      points.append(track.center)
    if not nodes:
      return None
    order = joint_route(np.asarray(points), self.pos, len(indices))
    self.future = [nodes[i][1] for i in order if nodes[i][0] == "scan"]
    self.next_channel = next((nodes[i][1] for i in order if nodes[i][0] == "service"), 0)
    return nodes[order[0]]

  def run(self, enter_body: dict[str, object] | None = None) -> MissionResult:
    if self._entered:
      raise ApiError("agent instances cannot enter a second mission")
    if enter_body is None:
      enter_body = self.client.enter()
    self._entered = True
    self.virtual_s = float(enter_body["virtual_time_s"])
    try:
      while True:
        action = self._plan()
        if action is None:
          break
        kind, index = action
        if kind == "scan":
          self.todo.remove(index)
          self._scan_station(index)
        else:
          self._localize(self.tracks[index])
      self._certify()
      if any(t.status not in {"cleared", "absent_certified"}
             for t in self.tracks.values()):
        raise ApiError("mission has unresolved channel certificates")
      cleared = sum(t.status == "cleared" for t in self.tracks.values())
      if not 10 <= cleared <= 16:
        raise ApiError("cleared count violates the stated source-count bounds")
      self._exit()
      return MissionResult(cleared, 20 - cleared, self.measures,
                           self.clear_calls, self.virtual_s, True)
    except BaseException:
      try:
        self._exit()
      except Exception:
        pass
      raise
