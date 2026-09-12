# Q3: 扫描站复用方位, 阴性测量约束, 清除任务插入同一条开放路径.
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from geometry_solver import regular_bound
from robot.agent import MissionResult, Observation, q3_scan_sites, safe_lateral_site
from robot.client import ApiClient, ApiError
from robot.fast_geometry import (
  Arr, cover_cells, cut, cut_bearing, cut_disk, definitely_disjoint,
  enclosing, forecast, guard, insertion, route_order,
)


@dataclass
class FastTrack:
  channel: int
  status: str = "unknown"
  obs: list[Observation] = field(default_factory=list)
  negatives: list[Arr] = field(default_factory=list)
  history: list[tuple[Arr, dict[str, object]]] = field(default_factory=list)
  failures: list[Arr] = field(default_factory=list)
  poly: Arr | None = None
  center: Arr | None = None
  radius: float = math.inf
  station_mask: int = 0
  opportunities: int = 0


class Q3FastAgent:
  directional = False

  def __init__(self, client: ApiClient, err_deg: float = 1.01,
               opportunistic: bool = True, inline: bool = True) -> None:
    if not math.isfinite(err_deg) or not 1.01 <= err_deg <= 2.0:
      raise ValueError("err_deg must be in [1.01, 2.0]")
    self.client = client
    self.err_deg = err_deg
    self.opportunistic = opportunistic
    self.inline = inline
    self.tracks = {i: FastTrack(i) for i in range(1, 21)}
    self.pos = np.zeros(2)
    self.current_channel = 1
    self.measures = 0
    self.clear_calls = 0
    self.virtual_s = 0.0
    self._entered = False
    self._exited = False
    self.sites = self._scan_sites()
    self.full_mask = (1 << len(self.sites)) - 1

  def _scan_sites(self) -> Arr:
    return q3_scan_sites()

  @staticmethod
  def _cached(track: FastTrack, site: Arr) -> dict[str, object] | None:
    for old, body in track.history:
      if np.linalg.norm(site - old) <= 1e-7:
        return body
    return None

  def _refresh(self, track: FastTrack) -> None:
    if track.poly is None or len(track.poly) == 0:
      track.status = "geometry_fault"
      raise ApiError(f"empty localization region on channel {track.channel}")
    track.center, track.radius = enclosing(track.poly)
    track.status = "clear_ready" if track.radius <= 19.999 else "localizing"

  def _range_order(self, poly: Arr, positive: Arr, negative: Arr) -> Arr:
    # 同一接收半径下, d(G,positive) < d(G,negative). Q4 不能使用此式.
    delta = negative - positive
    if np.linalg.norm(delta) <= 1e-7:
      raise ApiError("conflicting measurements at the same position")
    limit = float(delta @ positive + 0.5 * (delta @ delta))
    return cut(poly, delta, limit)

  def _accept_direction(self, track: FastTrack, site: Arr,
                        bearing: float) -> None:
    if not math.isfinite(bearing):
      raise ApiError("nonfinite bearing")
    if track.poly is None:
      track.poly = regular_bound(1800.0, 360, outer=True)
    track.poly = cut_bearing(track.poly, site, bearing, self.err_deg)
    track.poly = cut_disk(track.poly, site, 1500.0)
    if not self.directional:
      for negative in track.negatives:
        track.poly = self._range_order(track.poly, site, negative)
    track.obs.append(Observation(site.copy(), bearing))
    self._refresh(track)

  def _accept_negative(self, track: FastTrack, site: Arr) -> None:
    track.negatives.append(site.copy())
    if track.poly is not None and not self.directional:
      for obs in track.obs:
        track.poly = self._range_order(track.poly, obs.site, site)
      self._refresh(track)

  def _sense(self, track: FastTrack, site: Arr) -> dict[str, object]:
    cached = self._cached(track, site)
    if cached is not None:
      return cached
    body = self.client.measure(float(site[0]), float(site[1]), track.channel)
    self.pos = site.copy()
    self.current_channel = track.channel
    self.measures += 1
    self.virtual_s = float(body["virtual_time_s"])
    track.history.append((site.copy(), dict(body)))
    result = body.get("measure_result")
    if result == "direction":
      self._accept_direction(track, site, float(body["svd_deg"]))
    elif result == "no_signal":
      self._accept_negative(track, site)
    elif result == "near":
      self._clear(track, site, near=True)
    else:
      raise ApiError(f"unknown measurement result on channel {track.channel}")
    return body

  def _clear(self, track: FastTrack, site: Arr, near: bool = False,
             certified: bool = True) -> bool:
    if track.status == "cleared":
      return True
    if certified and not near:
      if track.poly is None:
        raise ApiError("clear requested without geometric evidence")
      far = float(np.max(np.linalg.norm(track.poly - site, axis=1)))
      if not math.isfinite(far) or far > 19.999:
        raise ApiError("clear position failed the vertex certificate")
    if not certified and not self.directional:
      raise ApiError("uncertified clear is disabled in Q3")
    body = self.client.clear(float(site[0]), float(site[1]), track.channel)
    self.pos = site.copy()
    # /clear 不执行测向频道切换, current_channel 仍是最后测向频道.
    self.clear_calls += 1
    self.virtual_s = float(body["virtual_time_s"])
    result = body.get("clear_result")
    if result == "success":
      track.status = "cleared"
      return True
    if result != "no_target_in_range":
      raise ApiError(f"unknown clear result on channel {track.channel}")
    if certified:
      track.status = "geometry_fault"
      raise ApiError(f"certified clear failed on channel {track.channel}")
    track.failures.append(site.copy())
    return False

  def _clear_position(self, track: FastTrack) -> Arr:
    if track.center is None:
      raise ApiError("missing localization center")
    delta = self.pos - track.center
    distance = float(np.linalg.norm(delta))
    slack = max(0.0, 19.998 - track.radius)
    if distance <= slack:
      return self.pos.copy()
    if distance == 0:
      return track.center.copy()
    return track.center + delta * (slack / distance)

  def _ready_clear(self, track: FastTrack) -> bool:
    if track.status == "cleared":
      return True
    if track.radius <= 19.999:
      return self._clear(track, self._clear_position(track))
    return False

  def _guaranteed_signal(self, track: FastTrack, site: Arr) -> bool:
    if self.directional or track.poly is None:
      return False
    if np.max(np.linalg.norm(track.poly - site, axis=1)) < 999.99:
      return True
    # 对整个外包围都比某个已接收点更近, 则未知接收半径也足够.
    for obs in track.obs:
      delta = site - obs.site
      value = (float(delta @ delta)
               - 2.0 * ((track.poly - obs.site) @ delta))
      if float(value.max()) < -guard:
        return True
    return False

  def _forecast(self, track: FastTrack, site: Arr) -> float:
    if track.poly is None or track.center is None:
      return math.inf
    return forecast(track.poly, track.center, track.radius, site, self.err_deg)

  def _harvest(self, site: Arr, excluded: int = 0) -> None:
    if not self.opportunistic:
      return
    pending = []
    for track in self.tracks.values():
      if (track.channel == excluded or track.status in {"unknown", "cleared",
          "absent_certified"} or track.center is None or track.poly is None
          or track.opportunities >= 6 or self._cached(track, site) is not None):
        continue
      if track.radius <= 19.999:
        continue
      if np.linalg.norm(site - track.center) > 1500.0 + track.radius:
        continue
      pred = self._forecast(track, site)
      if pred <= 19.5 or pred < 0.68 * track.radius:
        pending.append(track)
    pending.sort(key=lambda t: (t.channel != self.current_channel, t.channel))
    for track in pending:
      track.opportunities += 1
      self._sense(track, site)
      if track.status != "cleared" and track.poly is not None:
        # 只在当前点本身覆盖整个定位区域时原地清除, 不打断站内批测.
        if np.max(np.linalg.norm(track.poly - site, axis=1)) <= 19.999:
          self._clear(track, site)

  def _certify(self) -> None:
    found = sum(bool(t.obs) or t.status == "cleared"
                for t in self.tracks.values())
    if found > 16:
      raise ApiError("observations exceed the source-count upper bound")
    for track in self.tracks.values():
      if track.status == "unknown":
        if found == 16 or track.station_mask == self.full_mask:
          track.status = "absent_certified"

  def _scan_station(self, index: int) -> None:
    site = self.sites[index]
    unknown = [t for t in self.tracks.values() if t.status == "unknown"]
    unknown.sort(key=lambda t: (t.channel != self.current_channel, t.channel))
    for track in unknown:
      if track.status != "unknown":
        continue
      body = self._sense(track, site)
      if body.get("measure_result") == "no_signal":
        track.station_mask |= 1 << index
      self._certify()
    if np.linalg.norm(self.pos - site) <= 1e-7:
      self._harvest(site)

  def _choose_probe(self, track: FastTrack) -> Arr | None:
    if track.center is None or track.poly is None:
      return None
    center, radius = track.center, track.radius
    first = track.obs[0]
    theta = math.radians(first.bearing_deg)
    u = np.array([math.cos(theta), math.sin(theta)])
    v = np.array([-u[1], u[0]])
    candidates = [center.copy(), self.pos.copy()]
    for scale in (0.6, 1.5):
      distance = max(30.0, min(240.0, radius * scale))
      for axis in (u, v):
        candidates.extend((center + distance * axis, center - distance * axis))
    if len(track.obs) == 1:
      for side in (-1.0, 1.0):
        candidates.append(safe_lateral_site(first.site, first.bearing_deg, side))
    best = None
    best_score = math.inf
    for site in candidates:
      if self._cached(track, site) is not None:
        continue
      if not self.directional and not self._guaranteed_signal(track, site):
        continue
      if np.max(np.linalg.norm(track.poly - site, axis=1)) > 1499.99:
        continue
      pred = self._forecast(track, site)
      travel = float(np.linalg.norm(site - self.pos))
      onward = max(0.0, float(np.linalg.norm(site - center)) - 20.0)
      score = (travel + onward + 2.0 * pred) / 5.0
      score += 6.0 * max(0.0, math.log2(max(pred, 20.0) / 20.0))
      if self.directional:
        # 沿过去可见方向靠近的点通常更好, 这里只改变排序, 不删假设.
        old = first.site - center
        new = site - center
        if float(old @ new) < 0:
          score += 12.0
      if score < best_score:
        best_score, best = score, site.copy()
    return best

  def _localize(self, track: FastTrack) -> None:
    stagnant = 0
    for _ in range(10):
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
      if body.get("measure_result") == "no_signal":
        break
      if track.radius > 0.995 * old:
        stagnant += 1
      else:
        stagnant = 0
      if stagnant >= 2:
        break
    for _ in range(8):
      if self._ready_clear(track):
        return
      if track.radius > 300.0:
        break
      self._close_packet(track)
    self._finish_cover(track)

  def _close_packet(self, track: FastTrack) -> None:
    """小区域外的六点测站包, 方向未知时也覆盖多个观测方位."""
    if track.center is None or track.radius > 300.0:
      return
    center, old = track.center.copy(), track.radius
    radius = 2.0 * old + 40.0
    phase = math.atan2(self.pos[1] - center[1], self.pos[0] - center[0])
    angles = phase + np.arange(6) * math.pi / 3.0
    sites = center + radius * np.column_stack((np.cos(angles), np.sin(angles)))
    # 全部测站到区域内任意点至多 3*old+40 <= 940 m.
    for i in route_order(sites, self.pos):
      if self._ready_clear(track):
        return
      self._sense(track, sites[i])
      if track.status == "cleared" or track.radius < 0.55 * old:
        return

  def _finish_cover(self, track: FastTrack) -> None:
    # 极端退化时只测覆盖单元, 命中 5 m near 才清除, 保持 Q3 零试探清除.
    if self._ready_clear(track):
      return
    if track.poly is None:
      raise ApiError("missing fallback region")
    angle = math.radians(track.obs[0].bearing_deg)
    cells = cover_cells(track.poly, 4.5, angle)
    centers = np.vstack([enclosing(cell)[0] for cell in cells])
    # 小单元可能较多, 避免为兜底本身做立方级路径优化.
    left = set(range(len(cells)))
    while left:
      if self._ready_clear(track):
        return
      i = min(left, key=lambda j: (float(np.linalg.norm(centers[j] - self.pos)), j))
      left.remove(i)
      if definitely_disjoint(cells[i], track.poly):
        continue
      self._sense(track, centers[i])
    if not self._ready_clear(track):
      track.status = "geometry_fault"
      raise ApiError(f"near-coverage exhausted on channel {track.channel}")

  def _insert_services(self, remaining: Arr) -> None:
    if not self.inline or len(remaining) == 0:
      return
    while True:
      choices = []
      for track in self.tracks.values():
        if (track.status in {"unknown", "cleared", "absent_certified"}
            or track.center is None or track.radius > 180.0):
          continue
        index, cost = insertion(track.center, self.pos, remaining)
        if index != 0:
          continue
        # 若后续必经站就能达到清除精度, 暂缓这次主动靠近.
        if track.radius > 19.999 and any(
            self._cached(track, site) is None
            and self._forecast(track, site) <= 19.5 for site in remaining[:3]):
          continue
        choices.append((cost + track.radius, track.channel, track))
      if not choices:
        return
      _, _, track = min(choices, key=lambda x: (x[0], x[1]))
      self._localize(track)

  def _finish_pending(self) -> None:
    while True:
      pending = [t for t in self.tracks.values()
                 if t.status not in {"cleared", "absent_certified", "unknown"}]
      if not pending:
        return
      if any(t.center is None for t in pending):
        raise ApiError("pending channel has no localization region")
      points = np.vstack([t.center for t in pending])
      order = route_order(points, self.pos)
      self._localize(pending[order[0]])

  def _exit(self) -> None:
    if self._entered and not self._exited:
      body = self.client.exit()
      self._exited = True
      self.virtual_s = float(body["virtual_time_s"])

  def run(self, enter_body: dict[str, object] | None = None) -> MissionResult:
    if self._entered:
      raise ApiError("agent instances cannot enter a second mission")
    if enter_body is None:
      enter_body = self.client.enter()
    self._entered = True
    self.virtual_s = float(enter_body["virtual_time_s"])
    try:
      for index in range(len(self.sites)):
        self._certify()
        if not any(t.status == "unknown" for t in self.tracks.values()):
          break
        self._insert_services(self.sites[index:])
        self._scan_station(index)
      self._certify()
      self._finish_pending()
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
