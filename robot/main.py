# 问题三：机器狗自动定位与清除程序.
# 按附件 1 要求, 程序自行记录指令序列与响应信息, 落在 code/outputs/q3_<时间戳>.jsonl.
# 用法: python -X utf8 robot\main.py --team <参赛队号>
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from robot.agent import Q3Agent  # noqa: E402
from robot.client import ApiClient, ApiError  # noqa: E402


log_path: Path | None = None


def log_call(path: str, payload: dict, status: int, body: dict,
             ms: int) -> None:
  """记录一次 HTTP 往返; 写日志失败不影响测试进程."""
  if log_path is None:
    return
  try:
    with log_path.open("a", encoding="utf-8") as file:
      file.write(json.dumps({
        "t": ms, "path": path, "request": payload,
        "status": status, "response": body,
      }, ensure_ascii=False) + "\n")
  except OSError:
    pass


def attach_logging(client: ApiClient) -> None:
  original = client._post_once

  def wrapped(path: str, payload: dict) -> tuple[int, dict]:
    status, body = original(path, payload)
    log_call(path, payload, status, body, time.time_ns() // 1_000_000)
    return status, body

  client._post_once = wrapped  # type: ignore[method-assign]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="问题三 机器狗自动定位与清除")
  parser.add_argument(
    "--team", default=os.environ.get("CUMCM_TEAM_NO", ""),
    help="参赛队号, 需与模拟器当前登录账号一致")
  parser.add_argument("--base-url", default="http://127.0.0.1:2026",
                      help="模拟器接口地址")
  parser.add_argument("--wait-s", type=float, default=300.0,
                      help="等待接口开放的最长秒数")
  parser.add_argument("--max-obs", type=int, default=8,
                      help="单个频道的最大观测次数")
  parser.add_argument("--err-deg", type=float, default=1.01,
                      help="测向误差安全余量(度)")
  parser.add_argument("--log-dir", default=None,
                      help="指令日志目录, 默认 code/outputs")
  return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
  global log_path
  args = parse_args(argv)
  if not args.team:
    print("缺少参赛队号, 请用 --team 指定或设置环境变量 CUMCM_TEAM_NO")
    return 2

  log_dir = Path(args.log_dir) if args.log_dir else root / "code" / "outputs"
  log_dir.mkdir(parents=True, exist_ok=True)
  log_path = log_dir / f"q3_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"

  client = ApiClient(args.base_url, args.team)
  attach_logging(client)
  started = time.perf_counter()
  try:
    enter = client.enter_when_open(wait_s=args.wait_s)
  except ApiError as exc:
    print(f"进入失败: {exc}")
    print(f"本次指令日志: {log_path}")
    return 2

  remain = float(enter.get("remaining_real_duration_s", 0.0))
  print(f"已进入, 本局可用现实时间 {remain:.0f} 秒")

  agent = Q3Agent(client, err_deg=args.err_deg, max_obs=args.max_obs)
  error: str | None = None
  result = None
  try:
    result = agent.run(enter_body=enter)
  except Exception as exc:  # noqa: BLE001 - 异常时也要退出本局
    error = f"{type(exc).__name__}: {exc}"
    print(f"运行中断: {error}")
    try:
      client.exit()
    except ApiError:
      pass

  elapsed = time.perf_counter() - started
  out: dict[str, object] = {
    "program_s": round(elapsed, 3),
    "error": error,
    "log": str(log_path),
  }
  if result is not None:
    out.update({
      "cleared": result.cleared,
      "absent_certified": result.absent_certified,
      "measures": result.measures,
      "clear_calls": result.clear_calls,
      "virtual_time_s": result.virtual_time_s,
      "mean_virtual_per_cleared_s": (
        result.virtual_time_s / result.cleared if result.cleared else None),
      "complete": result.complete,
    })
  print(json.dumps(out, ensure_ascii=False, indent=2))
  return 0 if (result is not None and result.complete) else 1


if __name__ == "__main__":
  raise SystemExit(main())
