# 问题三: 先按骨架把全部频道扫一遍找到源, 再逐频道交会定位直到能清除.
# 每个频道的状态和观测都记在 ChannelTrack 里, 20 个频道全部有结论才允许退出.
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
  # 原点 + 半径 1200 m 的正六边形, 7 个点扫完能保证覆盖整个 1800 m 圆域:
  # 最坏位置的最近站点距离 968.90 m, 比接收半径下限 1000 m 小 31 m, 这就是余量
  angle = np.arange(6) * math.pi / 3.0
  ring = radius * np.column_stack((np.cos(angle), np.sin(angle)))
  return np.vstack((np.zeros(2), ring))


def polygon_clear_sites(poly: Arr, cover_rad: float = 20.0):
  # 兜底用的网格: 步长取 20*sqrt(2), 保证矩形里任何一点到最近格点都不超过 20 m,
  # 所以这个序列里必然有一个点能清掉目标
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
    # 蛇形走位, 省点移动距离
    row = ys if i % 2 == 0 else ys[::-1]
    points.extend(np.array([x, y]) for y in row)
  return np.asarray(points)


def safe_lateral_site(site: Arr, bearing_deg: float, side: float = 1.0) -> Arr:
  # 第二个测站点: 沿示向度前进 400 m, 再往侧面偏 300 m.
  # 步长 500 m, 横向有偏移就保证了不和第一条视线共线
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
    # 把该频道所有观测的测向锥求交, 再取最小覆盖圆. 用外接多边形所以半径偏大, 不会把没收敛当成收敛
    sites = np.vstack([obs.site for obs in track.obs])
    bearings = np.array([obs.bearing_deg for obs in track.obs])
    poly, _ = localization_polygon(
      sites, bearings, err_deg=self.err_deg, n_bound=1440, outer=True)
    if len(poly) == 0:
      # 交集空了说明观测自相矛盾, 记录故障, 不继续认证该频道
      track.status = "geometry_fault"
      raise ApiError(f"empty bearing intersection on channel {track.channel}")
    center, radius = minimum_enclosing_circle(poly)
    return poly, center, radius

  def _next_site(self, track: ChannelTrack, center: Arr, radius: float) -> Arr:
    # 第一次观测后先走侧偏点拿第二条视线; 已经缩得够小就直接去圆心;
    # 还大的话换到另一侧再测一次(这时候只有两次观测, 只能对称补)
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
  def _seen(track: ChannelTrack, site: Arr):
    return any(np.linalg.norm(obs.site - site) <= 1e-7 for obs in track.obs)

  def _fallback(self, track: ChannelTrack, poly: Arr) -> None:
    # 定位卡住了就按多边形铺 20 m 网格硬扫, 代价大但保证能碰到
    track.status = "clear_fallback"
    for site in polygon_clear_sites(poly):
      if self._clear(site, track, certified=False):
        return
    track.status = "geometry_fault"
    raise ApiError(f"fallback grid exhausted on channel {track.channel}")

  def _finish_channel(self, track: ChannelTrack) -> None:
    # 单个频道最多测 max_obs 次, 半径收到 20 m 以内就清除
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
        # 半径基本没再缩小, 说明再加观测也没用, 直接走兜底网格
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
      # 第一轮: 7 个骨架点 x 20 个频道. 已经清掉的频道不用再测
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
      # 扫完还是 unknown 的频道判定为不存在. 这一步成立的前提是接收半径不小于 1000 m
      for track in self.tracks.values():
        if track.status == "unknown":
          track.status = "absent_certified"
          self.absent_mask |= 1 << (track.channel - 1)
      # 20 个频道都拿到 cleared 或 absent_certified 才算完整, 否则不退出
      complete = self.cleared_mask | self.absent_mask == (1 << 20) - 1
      if not complete:
        raise ApiError("mission ended without a channel certificate")
      out = self.client.exit()
      self.virtual_s = float(out["virtual_time_s"])
    except Exception:
      # 出任何问题都要先退出本局, 不然会被判超时
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
