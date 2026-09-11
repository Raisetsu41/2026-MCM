# 本地轻量级模拟器.
# 复现题面物理时间, 幂等 ID 和同地点固定测向误差.
from __future__ import annotations

import hashlib
import json
import math
import socket
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np


Json = dict[str, Any]


@dataclass
class Jammer:
  channel: int
  x: float
  y: float
  recv_m: float
  heading_deg: float | None = None
  cleared: bool = False


class MockArena:
  def __init__(
      self, jammers: list[Jammer], robot_id: str = "mock-team",
      drop_first_action: bool = False,
  ) -> None:
    self.jammers = jammers
    self.robot_id = robot_id
    self.drop_first_action = drop_first_action
    self.dropped = False
    self.active = False
    self.enter_calls = 0
    self.pos = np.zeros(2)
    self.channel = 1
    self.virtual_s = 0.0
    self.cache: dict[str, tuple[str, str, Json]] = {}
    self.lock = threading.Lock()

  @staticmethod
  def random_q3(seed: int, count: int | None = None) -> "MockArena":
    rng = np.random.default_rng(seed)
    n = int(rng.integers(10, 17)) if count is None else count
    channels = rng.choice(np.arange(1, 21), size=n, replace=False)
    radius = 1800.0 * np.sqrt(rng.random(n))
    angle = rng.uniform(0.0, 2.0 * math.pi, n)
    recv = rng.uniform(1000.0, 1500.0, n)
    jams = [
      Jammer(int(ch), float(r * math.cos(a)), float(r * math.sin(a)), float(rv))
      for ch, r, a, rv in zip(channels, radius, angle, recv)
    ]
    return MockArena(jams)

  def _base(self) -> Json:
    return {
      "accepted": True,
      "real_timestamp_ms": time.time_ns() // 1_000_000,
      "virtual_time_s": round(self.virtual_s, 6),
    }

  def _jammer(self, channel: int) -> Jammer | None:
    for jammer in self.jammers:
      if jammer.channel == channel and not jammer.cleared:
        return jammer
    return None

  @staticmethod
  def _fixed_error(x: float, y: float, channel: int) -> float:
    key = f"{x:.6f},{y:.6f},{channel}".encode("ascii")
    raw = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    return 2.0 * raw / (2 ** 64 - 1) - 1.0

  @staticmethod
  def _visible(jammer: Jammer, pos: np.ndarray) -> bool:
    if jammer.heading_deg is None:
      return True
    ray = pos - np.array([jammer.x, jammer.y])
    if np.linalg.norm(ray) == 0:
      return True
    head = math.radians(jammer.heading_deg)
    return float(ray @ np.array([math.cos(head), math.sin(head)])) >= -1e-12

  def _move(self, pos: np.ndarray) -> None:
    self.virtual_s += float(np.linalg.norm(pos - self.pos)) / 5.0
    self.pos = pos

  def _enter(self) -> Json:
    self.enter_calls += 1
    if self.active:
      # 真实模拟器对同一局的第二次 /enter 返回 accepted=false.
      return {"accepted": False, "real_timestamp_ms": 0, "virtual_time_s": 0}
    self.active = True
    out = self._base()
    out.update({
      "max_virtual_duration_s": 360000,
      "max_real_duration_s": 1200,
      "remaining_real_duration_s": 1200,
    })
    return out

  def _measure(self, payload: Json) -> Json:
    pos = np.array([payload["position"]["x"], payload["position"]["y"]], dtype=float)
    channel = int(payload["channel"])
    self._move(pos)
    if channel != self.channel:
      self.virtual_s += 1.0
    self.channel = channel
    self.virtual_s += 5.0
    out = self._base()
    jammer = self._jammer(channel)
    if jammer is None:
      out["measure_result"] = "no_signal"
      return out
    target = np.array([jammer.x, jammer.y])
    dist = float(np.linalg.norm(target - pos))
    if dist > jammer.recv_m + 1e-9 or not self._visible(jammer, pos):
      out["measure_result"] = "no_signal"
    elif dist <= 5.0 + 1e-9:
      out["measure_result"] = "near"
    else:
      angle = math.degrees(math.atan2(target[1] - pos[1], target[0] - pos[0]))
      angle += self._fixed_error(float(pos[0]), float(pos[1]), channel)
      out["measure_result"] = "direction"
      out["svd_deg"] = round(angle % 360.0, 2)
    return out

  def _clear(self, payload: Json) -> Json:
    pos = np.array([payload["position"]["x"], payload["position"]["y"]], dtype=float)
    channel = int(payload["channel"])
    self._move(pos)
    jammer = self._jammer(channel)
    target = None if jammer is None else np.array([jammer.x, jammer.y])
    success = target is not None and np.linalg.norm(target - pos) <= 20.0 + 1e-9
    self.virtual_s += 5.0 if success else 3.0
    if success and jammer is not None:
      jammer.cleared = True
    out = self._base()
    out["clear_result"] = "success" if success else "no_target_in_range"
    return out

  def _exit(self) -> Json:
    out = self._base()
    out["exit_reason"] = "user_exit"
    self.active = False
    return out

  def handle(self, path: str, payload: Json) -> tuple[int, Json, bool]:
    with self.lock:
      request_id = str(payload.get("request_id", ""))
      raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
      if request_id in self.cache:
        old_path, old_raw, old_body = self.cache[request_id]
        if old_path != path or old_raw != raw:
          return 409, {"accepted": False, "real_timestamp_ms": 0,
                       "virtual_time_s": 0}, False
        return 200, old_body, False
      if payload.get("arena_id") != "default" or payload.get("robot_id") != self.robot_id:
        return 200, {"accepted": False, "real_timestamp_ms": 0,
                     "virtual_time_s": 0}, False
      if path == "/enter":
        body = self._enter()
      elif not self.active:
        body = {"accepted": False, "real_timestamp_ms": 0, "virtual_time_s": 0}
      elif path == "/measure":
        body = self._measure(payload)
      elif path == "/clear":
        body = self._clear(payload)
      elif path == "/exit":
        body = self._exit()
      else:
        return 404, {"accepted": False, "real_timestamp_ms": 0,
                     "virtual_time_s": 0}, False
      if body.get("accepted") is True:
        self.cache[request_id] = (path, raw, body)
      return 200, body, True


class MockServer:
  def __init__(self, arena: MockArena) -> None:
    self.arena = arena
    self.server: ThreadingHTTPServer | None = None
    self.thread: threading.Thread | None = None

  def __enter__(self) -> "MockServer":
    arena = self.arena

    class Handler(BaseHTTPRequestHandler):
      def do_POST(self) -> None:
        size = int(self.headers.get("Content-Length", "0"))
        try:
          payload = json.loads(self.rfile.read(size).decode("utf-8"))
          status, body, fresh = arena.handle(self.path, payload)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
          status = 400
          body = {"accepted": False, "real_timestamp_ms": 0,
                  "virtual_time_s": 0}
          fresh = False
        if (arena.drop_first_action and fresh and self.path == "/measure"
            and not arena.dropped):
          arena.dropped = True
          try:
            self.connection.shutdown(socket.SHUT_RDWR)
          except OSError:
            pass
          self.connection.close()
          return
        data = json.dumps(body, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

      def log_message(self, format: str, *args: object) -> None:
        return

    self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
    self.thread.start()
    return self

  @property
  def url(self) -> str:
    if self.server is None:
      raise RuntimeError("mock server is not running")
    host, port = self.server.server_address
    return f"http://{host}:{port}"

  def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
    if self.server is not None:
      self.server.shutdown()
      self.server.server_close()
    if self.thread is not None:
      self.thread.join(timeout=2.0)
