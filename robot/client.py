# 模拟器 HTTP 客户端.
# 四个动作串行发送, 失败按同一 request_id 重试, 并守住现实时间预算.
from __future__ import annotations

import json
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


Json = dict[str, Any]


class ApiError(RuntimeError):
  """模拟器通信或业务拒绝.

  rejected=True 表示收到过模拟器的 JSON 响应但业务状态拒绝了本动作
  (accepted=false 或 4xx); rejected=False 表示连接层面一直失败.
  """

  def __init__(self, message: str, rejected: bool = False) -> None:
    super().__init__(message)
    self.rejected = rejected


class ApiClient:
  def __init__(
      self, base_url: str, robot_id: str,
      timeout_s: float = 3.0, max_retry: int = 5,
      backoff_s: float = 0.05, deadline_guard_s: float = 2.0,
  ) -> None:
    if not robot_id or not 1 <= len(robot_id.encode("utf-8")) <= 64:
      raise ValueError("robot_id must contain 1 to 64 UTF-8 bytes")
    if timeout_s <= 0 or max_retry < 0 or backoff_s < 0:
      raise ValueError("invalid retry parameters")
    self.base_url = base_url.rstrip("/")
    self.robot_id = robot_id
    self.timeout_s = timeout_s
    self.max_retry = max_retry
    self.backoff_s = backoff_s
    self.deadline_guard_s = deadline_guard_s
    self.seq = 0
    self.deadline: float | None = None
    self.lock = threading.Lock()

  def _new_id(self, tag: str) -> str:
    self.seq += 1
    return f"{tag}-{self.seq:06d}"

  def _base(self, request_id: str) -> Json:
    return {
      "arena_id": "default",
      "robot_id": self.robot_id,
      "request_id": request_id,
    }

  def _action(self, request_id: str, x: float, y: float, channel: int) -> Json:
    if not 1 <= channel <= 20:
      raise ValueError("channel must be in 1..20")
    if not abs(x) <= 2_000_000 or not abs(y) <= 2_000_000:
      raise ValueError("position exceeds simulator limit")
    out = self._base(request_id)
    out["position"] = {"x": float(x), "y": float(y)}
    out["channel"] = int(channel)
    return out

  def _check_deadline(self, path: str) -> None:
    if path == "/exit" or self.deadline is None:
      return
    if time.monotonic() + self.deadline_guard_s >= self.deadline:
      raise ApiError("real-time deadline guard reached")

  def _post_once(self, path: str, payload: Json) -> tuple[int, Json]:
    data = json.dumps(payload, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")
    req = Request(
      self.base_url + path, data=data,
      headers={"Content-Type": "application/json"}, method="POST")
    try:
      with urlopen(req, timeout=self.timeout_s) as resp:
        status = int(resp.status)
        body = json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
      status = int(exc.code)
      raw = exc.read().decode("utf-8", errors="replace")
      try:
        body = json.loads(raw)
      except json.JSONDecodeError:
        body = {"accepted": False, "raw": raw}
    if not isinstance(body, dict):
      raise ApiError("response must be a JSON object")
    return status, body

  def _post(self, path: str, payload: Json) -> Json:
    with self.lock:
      self._check_deadline(path)
      saw_response = False
      last_status = 0
      for attempt in range(self.max_retry + 1):
        try:
          status, body = self._post_once(path, payload)
          saw_response = True
          last_status = status
          if status == 200 and body.get("accepted") is True:
            return body
          if status not in (429, 500):
            detail = body.get("error") or body.get("message") or body.get("raw")
            extra = f", detail={detail}" if detail else ""
            hint = ""
            if path == "/enter" and status == 200:
              hint = " (重复 /enter 会被拒绝, 每局只允许进入一次)"
            raise ApiError(
              f"{path} 被拒绝: status={status}, "
              f"accepted={body.get('accepted')}{extra}{hint}", rejected=True)
        except (TimeoutError, ConnectionError, URLError,
                json.JSONDecodeError):
          pass
        if attempt == self.max_retry:
          break
        self._check_deadline(path)
        time.sleep(min(self.backoff_s * 2 ** attempt, 1.0))
      if saw_response:
        raise ApiError(
          f"{path} 重试 {self.max_retry + 1} 次仍被拒绝: status={last_status}",
          rejected=True)
      raise ApiError(
        f"{path} 连接失败(共 {self.max_retry + 1} 次): 接口可能未开放或已结束")

  def enter_payload(self, payload: Json) -> Json:
    body = self._post("/enter", payload)
    remain = float(body["remaining_real_duration_s"])
    self.deadline = time.monotonic() + remain
    return body

  def enter_when_open(self, wait_s: float = 300.0,
                      request_id: str = "enter-wait-000001") -> Json:
    """等待接口开放后再进入.

    倒计时期间与非测试期间连接会被直接关闭, 因此只能轮询. 全程复用同一个
    request_id: 一旦某次已被接受, 后续重试会命中幂等缓存并原样返回同一响应.
    """
    payload = self._base(request_id)
    end = time.monotonic() + wait_s
    delay = 0.5
    rejected: ApiError | None = None
    while True:
      try:
        return self.enter_payload(payload)
      except ApiError as exc:
        if exc.rejected:
          rejected = exc
          break
      except (TimeoutError, ConnectionError, URLError):
        pass
      if time.monotonic() + delay >= end:
        raise ApiError(
          f"接口在 {wait_s:.0f} 秒内未开放, 请确认已开始测试并等完 5 秒倒计时")
      time.sleep(delay)
      delay = min(delay * 1.5, 2.0)
    raise ApiError(
      f"/enter 被拒绝: {rejected}. 请检查本局是否已被进入过, "
      "以及是否还有另一个程序在运行")

  def measure(self, x: float, y: float, channel: int) -> Json:
    payload = self._action(self._new_id("measure"), x, y, channel)
    return self._post("/measure", payload)

  def clear(self, x: float, y: float, channel: int) -> Json:
    payload = self._action(self._new_id("clear"), x, y, channel)
    return self._post("/clear", payload)

  def exit(self) -> Json:
    payload = self._base(self._new_id("exit"))
    return self._post("/exit", payload)
