from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from geometry_solver import localization_polygon, minimum_enclosing_circle
from robot.client import ApiClient, ApiError

Arr = np.ndarray

@dataclass
class Observation:
  site: Arr
  bearing_deg: float

@dataclass
class ChannelTrack:
  channel: int
  status: str = "unknown"
  obs: list[Observation] = field(default_factory=list)
  radii: list[float] = field(default_factory=list)
  clear_site: Arr | None = None

@dataclass
class MissionResult:
  cleared: int
  absent_certified: int
  measures: int
  clear_calls: int
  virtual_time_s: float
  complete: bool

def q3_scan_sites(radius: float = 1200.0) -> Arr:
  angle = np.arange(6) * math.pi / 3.0
  ring = radius * np.column_stack((np.cos(angle), np.sin(angle)))
  return np.vstack((np.zeros(2), ring))

def polygon_clear_sites(poly: Arr, cover_rad: float = 20.0):
  p = np.asarray(poly, dtype=float)
  if p.ndim != 2 or p.shape[1] != 2 or len(p) == 0 or cover_rad <= 0:
    raise ValueError("invalid fallback polygon")
  lo = p.min(axis=0)
  hi = p.max(axis=0)
  step = cover_rad * math.sqrt(2.0)
  nx = max(1, int(math.ceil((hi[0] - lo[0]) / step)))
  ny = max(1, int(math.ceil((hi[1] - lo[1]) / step)))
  xs = np.linspace(lo[0], hi[0], nx + 1)
  ys = np.linspace(lo[1], hi[1], ny + 1)
  points = []
  for i, x in enumerate(xs):
    row = ys if i % 2 == 0 else ys[::-1]
    points.extend(np.array([x, y]) for y in row)
  return np.asarray(points)

def safe_lateral_site(site: Arr, bearing_deg: float, side: float = 1.0) -> Arr:
  angle = math.radians(bearing_deg)
  u = np.array([math.cos(angle), math.sin(angle)])
  v = np.array([-math.sin(angle), math.cos(angle)])
  return site + 400.0 * u + side * 300.0 * v

class Q3Agent:
  def __init__(self, client: ApiClient, err_deg: float = 1.01,
               max_obs: int = 8, schedule: str = "batch") -> None:
    if schedule not in {"batch", "immediate"}:
      raise ValueError("schedule must be batch or immediate")
    self.client = client
    self.err_deg = err_deg
    self.max_obs = max_obs
    self.schedule = schedule
    self.tracks = {channel: ChannelTrack(channel) for channel in range(1, 21)}
    self.measures = 0
    self.clear_calls = 0
    self.virtual_s = 0.0
    self.pos = np.zeros(2)
    self.current_channel = 1
    self.cleared_mask = 0
    self.absent_mask = 0
    self.entered_here = False

  def _measure(self, site: Arr, channel: int) -> dict[str, object]:
    body = self.client.measure(float(site[0]), float(site[1]), channel)
    self.measures += 1
    self.virtual_s = float(body["virtual_time_s"])
    self.pos = site.copy()
    self.current_channel = channel
    return body

  def _clear(self, site: Arr, track: ChannelTrack,
             certified: bool = True) -> bool:
    body = self.client.clear(float(site[0]), float(site[1]), track.channel)
    self.clear_calls += 1
    self.virtual_s = float(body["virtual_time_s"])
    self.pos = site.copy()
    if body.get("clear_result") != "success":
      if not certified:
        return False
      track.status = "geometry_fault"
      raise ApiError(f"certified clear failed on channel {track.channel}")
    track.status = "cleared"
    track.clear_site = site.copy()
    self.cleared_mask |= 1 << (track.channel - 1)
    return True

  def _record(self, track: ChannelTrack, site: Arr,
              body: dict[str, object]) -> None:
    result = body.get("measure_result")
    if result == "near":
      track.status = "clear_ready"
      self._clear(site, track)
      return
    if result != "direction":
      raise ApiError(f"expected positive detection on channel {track.channel}")
    bearing = float(body["svd_deg"])
    if not math.isfinite(bearing):
      raise ApiError(f"invalid bearing on channel {track.channel}")
    track.obs.append(Observation(site.copy(), bearing))
    track.status = "positively_detected" if len(track.obs) == 1 else "localizing"

  def _region(self, track: ChannelTrack) -> tuple[Arr, Arr, float]:
    sites = np.vstack([obs.site for obs in track.obs])
    bearings = np.array([obs.bearing_deg for obs in track.obs])
    poly, _ = localization_polygon(
      sites, bearings, err_deg=self.err_deg, n_bound=1440, outer=True)
    if len(poly) == 0:
      track.status = "geometry_fault"
      raise ApiError(f"empty bearing intersection on channel {track.channel}")
    center, radius = minimum_enclosing_circle(poly)
    return poly, center, radius

  def _next_site(self, track: ChannelTrack, center: Arr, radius: float) -> Arr:
    first = track.obs[0]
    if len(track.obs) == 1:
      return safe_lateral_site(first.site, first.bearing_deg, 1.0)
    if radius <= 900.0:
      if not self._seen(track, center):
        return center
      step = min(40.0, max((1000.0 - radius) / 4.0, 0.01))
      for index in range(16):
        angle = math.radians(first.bearing_deg + 137.5 * index)
        candidate = center + step * np.array([math.cos(angle), math.sin(angle)])
        if not self._seen(track, candidate):
          return candidate
    if len(track.obs) == 2:
      return safe_lateral_site(first.site, first.bearing_deg, -1.0)
    track.status = "geometry_fault"
    raise ApiError(f"localization radius remains unsafe on channel {track.channel}")

  @staticmethod
  def _seen(track: ChannelTrack, site: Arr):
    return any(np.linalg.norm(obs.site - site) <= 1e-7 for obs in track.obs)

  def _lost_signal(self, track: ChannelTrack) -> None:
    track.status = "geometry_fault"
    raise ApiError(f"guaranteed follow-up lost signal on channel {track.channel}")

  def _finish_channel(self, track: ChannelTrack) -> None:
    last_poly: Arr | None = None
    for _ in range(self.max_obs):
      if track.status == "cleared":
        return
      if len(track.obs) == 1:
        center = track.obs[0].site
        radius = math.inf
      else:
        poly, center, radius = self._region(track)
        last_poly = poly
        if radius <= 20.0:
          track.status = "clear_ready"
          self._clear(center, track)
          return
        track.radii.append(radius)
      site = self._next_site(track, center, radius)
      if self._seen(track, site):
        track.status = "geometry_fault"
        raise ApiError(f"repeated localization site on channel {track.channel}")
      body = self._measure(site, track.channel)
      result = body.get("measure_result")
      if result == "no_signal":
        self._lost_signal(track)
        return
      self._record(track, site, body)
    track.status = "geometry_fault"
    detail = "after intersection" if last_poly is not None else "before intersection"
    raise ApiError(
      f"measurement cap reached {detail} on channel {track.channel}")

  def _scan_sites(self) -> Arr:
    return q3_scan_sites()

  def _known_count(self) -> int:
    return sum(track.status != "unknown" for track in self.tracks.values())

  def _channel_order(self) -> list[int]:
    unknown = [
      channel for channel, track in self.tracks.items()
      if track.status == "unknown"
    ]
    return sorted(unknown, key=lambda channel: (channel - self.current_channel) % 20)

  def _discover(self, immediate: bool) -> None:
    stop = False
    for site in self._scan_sites():
      for channel in self._channel_order():
        track = self.tracks[channel]
        body = self._measure(site, channel)
        if body.get("measure_result") == "no_signal":
          continue
        self._record(track, site, body)
        if immediate and track.status != "cleared":
          self._finish_channel(track)
        if self._known_count() >= 16:
          stop = True
          break
      if stop:
        break

  def _finish_pending(self) -> None:
    while True:
      pending = [
        track for track in self.tracks.values()
        if track.status in {"positively_detected", "localizing"}
      ]
      if not pending:
        return
      def travel(track: ChannelTrack) -> float:
        first = track.obs[0]
        site = safe_lateral_site(first.site, first.bearing_deg)
        return float(np.linalg.norm(site - self.pos))

      track = min(pending, key=lambda item: (travel(item), item.channel))
      self._finish_channel(track)

  def _certify_unknown(self) -> None:
    for track in self.tracks.values():
      if track.status == "unknown":
        track.status = "absent_certified"
        self.absent_mask |= 1 << (track.channel - 1)

  def run(self, enter_body: dict[str, object] | None = None) -> MissionResult:
    if enter_body is None:
      enter_body = self.client.enter()
      self.entered_here = True
    self.virtual_s = float(enter_body["virtual_time_s"])
    try:
      self._discover(immediate=self.schedule == "immediate")
      if self.schedule == "batch":
        self._finish_pending()
      self._certify_unknown()
      complete = self.cleared_mask | self.absent_mask == (1 << 20) - 1
      if not complete:
        raise ApiError("mission ended without a channel certificate")
      out = self.client.exit()
      self.virtual_s = float(out["virtual_time_s"])
    except Exception:
      try:
        self.client.exit()
      except Exception:
        pass
      raise
    cleared = sum(track.status == "cleared" for track in self.tracks.values())
    absent = sum(track.status == "absent_certified" for track in self.tracks.values())
    return MissionResult(
      cleared, absent, self.measures, self.clear_calls,
      self.virtual_s, complete)
