from __future__ import annotations

import math

import numpy as np

from robot.q3_agent_fast import FastTrack, LocalProbe, Q3FastBase, Q3FastPlanner
from robot.client import ApiClient, ApiError
from robot.geometry_solver_fast import (
  Arr, compact_q4_sites, cover_cells, definitely_disjoint, direction_certificate,
  enclosing, packet_localization_bound, packet_sites, path_length, route_order,
)
from robot.q4_agent import q4_scan_sites

def q4_ring_sites() -> Arr:
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
  return np.vstack((np.zeros(2), inner, outer[np.r_[11, np.arange(11)]]))

class Q4FastBase(Q3FastBase):
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
    packet_time = (approach + 2.0 * ring_r) / 5.0 + 15.0
    return cover_time <= packet_time

  def _finish_cover(self, track: FastTrack) -> None:
    if self._ready_clear(track):
      return
    if track.poly is None:
      raise ApiError("missing directional fallback region")
    angle = math.radians(track.obs[0].bearing_deg)
    cells = cover_cells(track.poly, 19.5, angle)
    centers = np.vstack([enclosing(cell)[0] for cell in cells])
    for i in route_order(centers, self.pos):
      cell = cells[i]
      if definitely_disjoint(cell, track.poly):
        continue
      if any(np.max(np.linalg.norm(cell - old, axis=1)) <= 19.999
             for old in track.failures):
        continue
      if any(np.linalg.norm(centers[i] - old) <= 1e-7 for old in track.failures):
        continue
      if self._clear(track, centers[i], certified=False):
        return
    track.status = "geometry_fault"
    raise ApiError(f"directional coverage exhausted on channel {track.channel}")

class Q4FastCertified(Q4FastBase, Q3FastPlanner):
  def __init__(self, client: ApiClient, err_deg: float = 1.01,
               opportunistic: bool = True, inline: bool = True,
               scan: str = "compact") -> None:
    if scan not in {"compact", "rings", "lattice"}:
      raise ValueError("Q4 scan must be compact, rings or lattice")
    self.use_compact = scan == "compact"
    self.certificate_cells = 0
    self.compact_certified = False
    super().__init__(client, err_deg, opportunistic, inline,
                     "rings" if self.use_compact else scan)

  def _scan_sites(self) -> Arr:
    if not self.use_compact:
      return super()._scan_sites()
    sites = compact_q4_sites()
    ok, self.certificate_cells = direction_certificate(sites)
    self.compact_certified = ok
    return sites if ok else q4_ring_sites()

class Q4FastAgent(LocalProbe, Q4FastCertified):
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
        Q4FastBase._finish_cover(self, track)
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
