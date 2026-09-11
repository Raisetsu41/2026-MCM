# Q3 全向干扰源的确定性发现与集合定位闭环.
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


def q4_scan_sites(step: float = 990.0, target_rad: float = 1800.0) -> Arr:
  if step <= 0 or target_rad <= 0 or step > 1000.0:
    raise ValueError("invalid triangular grid parameters")
  e1 = np.array([step, 0.0])
  e2 = np.array([step / 2.0, math.sqrt(3.0) * step / 2.0])
  lim = int(math.ceil((target_rad + step) / (step * math.sqrt(3.0) / 2.0))) + 2
  points = []
  for i in range(-2 * lim, 2 * lim + 1):
    for j in range(-lim, lim + 1):
      point = i * e1 + j * e2
      if np.linalg.norm(point) <= target_rad + step + 1e-9:
        points.append(point)
  return np.asarray(points)


def fallback_clear_sites(
    site: Arr, bearing_deg: float, err_deg: float = 1.01,
    near_m: float = 5.0, far_m: float = 1500.0,
    cover_rad: float = 20.0,
) -> Arr:
  if not 0 <= err_deg < 90 or not 0 <= near_m < far_m or cover_rad <= 0:
    raise ValueError("invalid fallback parameters")
  angle = math.radians(bearing_deg)
  u = np.array([math.cos(angle), math.sin(angle)])
  v = np.array([-math.sin(angle), math.cos(angle)])
  err = math.radians(err_deg)
  lo_a = near_m * math.cos(err)
  hi_a = far_m
  hi_b = far_m * math.sin(err)
  step = cover_rad * math.sqrt(2.0)
  n_a = max(1, int(math.ceil((hi_a - lo_a) / step)))
  n_b = max(1, int(math.ceil(2.0 * hi_b / step)))
  axis_a = np.linspace(lo_a, hi_a, n_a + 1)
  axis_b = np.linspace(-hi_b, hi_b, n_b + 1)
  points = []
  for i, a in enumerate(axis_a):
    row = axis_b if i % 2 == 0 else axis_b[::-1]
    points.extend(site + a * u + b * v for b in row)
  return np.asarray(points)


def polygon_clear_sites(poly: Arr, cover_rad: float = 20.0) -> Arr:
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
               max_obs: int = 8) -> None:
    self.client = client
    self.err_deg = err_deg
    self.max_obs = max_obs
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
    track.obs.append(Observation(site.copy(), float(body["svd_deg"])))
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
      return center
    if len(track.obs) == 2:
      return safe_lateral_site(first.site, first.bearing_deg, -1.0)
    track.status = "geometry_fault"
    raise ApiError(f"localization radius remains unsafe on channel {track.channel}")

  @staticmethod
  def _seen(track: ChannelTrack, site: Arr) -> bool:
    return any(np.linalg.norm(obs.site - site) <= 1e-7 for obs in track.obs)

  def _fallback(self, track: ChannelTrack, poly: Arr) -> None:
    track.status = "clear_fallback"
    for site in polygon_clear_sites(poly):
      if self._clear(site, track, certified=False):
        return
    track.status = "geometry_fault"
    raise ApiError(f"fallback grid exhausted on channel {track.channel}")

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
        if track.radii and radius >= 0.995 * track.radii[-1]:
          self._fallback(track, poly)
          return
        track.radii.append(radius)
      site = self._next_site(track, center, radius)
      if self._seen(track, site):
        if last_poly is None:
          raise ApiError(f"repeated site before localization on channel {track.channel}")
        self._fallback(track, last_poly)
        return
      body = self._measure(site, track.channel)
      result = body.get("measure_result")
      if result == "no_signal":
        track.status = "geometry_fault"
        raise ApiError(f"guaranteed follow-up lost signal on channel {track.channel}")
      self._record(track, site, body)
    if len(track.obs) >= 2:
      poly, _, _ = self._region(track)
      self._fallback(track, poly)
      return
    track.status = "geometry_fault"
    raise ApiError(f"measurement cap reached before intersection on channel {track.channel}")

  def run(self, enter_body: dict[str, object] | None = None) -> MissionResult:
    """执行一轮 Q3 定位清除.

    enter_body 非空时表示调用方已经成功调用过 /enter (例如为了等待接口开放
    而先行轮询), 此时不再重复调用, 否则模拟器会以 accepted=false 拒绝本局.
    """
    if enter_body is None:
      enter_body = self.client.enter()
      self.entered_here = True
    self.virtual_s = float(enter_body["virtual_time_s"])
    try:
      for site in q3_scan_sites():
        for channel in range(1, 21):
          track = self.tracks[channel]
          if track.status == "cleared":
            continue
          body = self._measure(site, channel)
          result = body.get("measure_result")
          if result == "no_signal":
            continue
          self._record(track, site, body)
          if track.status != "cleared":
            self._finish_channel(track)
      for track in self.tracks.values():
        if track.status == "unknown":
          track.status = "absent_certified"
          self.absent_mask |= 1 << (track.channel - 1)
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
