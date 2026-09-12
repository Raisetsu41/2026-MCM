# 独立快速版入口, 队号只读命令行或环境变量, 默认不写日志文件.
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import TextIO


root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from robot.agent_fast import Q3FastAgent  # noqa: E402
from robot.client import ApiClient  # noqa: E402
from robot.q4_agent_fast import Q4FastAgent  # noqa: E402


def redact(value: object, team: str) -> object:
  if isinstance(value, dict):
    return {key: "<redacted>" if key in {"robot_id", "team_no"}
            else redact(item, team) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [redact(item, team) for item in value]
  if isinstance(value, str) and team:
    return value.replace(team, "<redacted>")
  return value


def attach_logging(client: ApiClient, file: TextIO, team: str) -> None:
  original = client._post_once

  def wrapped(path: str, payload: dict) -> tuple[int, dict]:
    status, body = original(path, payload)
    try:
      record = {"t": time.time_ns() // 1_000_000, "path": path,
                "request": payload, "status": status, "response": body}
      file.write(json.dumps(redact(record, team), ensure_ascii=False) + "\n")
      file.flush()
    except OSError:
      pass
    return status, body

  client._post_once = wrapped  # type: ignore[method-assign]


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="Q3/Q4 独立快速版")
  parser.add_argument("--team", default=os.environ.get("CUMCM_TEAM_NO", ""))
  parser.add_argument("--problem", type=int, choices=(3, 4), default=3)
  parser.add_argument("--base-url", default="http://127.0.0.1:2026")
  parser.add_argument("--wait-s", type=float, default=300.0)
  parser.add_argument("--q4-scan", choices=("rings", "lattice"), default="rings")
  parser.add_argument("--no-opportunistic", action="store_true")
  parser.add_argument("--no-inline", action="store_true")
  parser.add_argument("--log-dir", help="可选日志目录, 默认仅输出控制台汇总")
  args = parser.parse_args(argv)
  if not args.team:
    print("请用 --team 传入队号, 或设置 CUMCM_TEAM_NO")
    return 2
  if not 0 < args.wait_s < float("inf"):
    print("--wait-s 必须是有限正数")
    return 2
  file: TextIO | None = None
  started = time.perf_counter()
  try:
    client = ApiClient(args.base_url, args.team)
    options = {"opportunistic": not args.no_opportunistic,
               "inline": not args.no_inline}
    agent = (Q3FastAgent(client, **options) if args.problem == 3 else
             Q4FastAgent(client, scan=args.q4_scan, **options))
    if args.log_dir:
      folder = Path(args.log_dir).resolve()
      folder.mkdir(parents=True, exist_ok=True)
      name = f"q{args.problem}_fast_{time.time_ns()}.jsonl"
      file = (folder / name).open("x", encoding="utf-8")
      attach_logging(client, file, args.team)
    print(f"Q{args.problem} fast 已就绪, 等待官方测试窗口")
    enter = client.enter_when_open(wait_s=args.wait_s)
    started = time.perf_counter()
    print("已进入, 开始扫描和定位")
    result = agent.run(enter_body=enter)
    out = {"problem": args.problem, "variant": "fast",
           "cleared": result.cleared, "absent_certified": result.absent_certified,
           "measures": result.measures, "clear_calls": result.clear_calls,
           "virtual_time_s": result.virtual_time_s,
           "mean_virtual_per_cleared_s": result.virtual_time_s / result.cleared,
           "program_s": round(time.perf_counter() - started, 3),
           "complete": result.complete}
    print(json.dumps(redact(out, args.team), ensure_ascii=False, indent=2))
    return 0
  except KeyboardInterrupt:
    print("程序已中断")
    return 130
  except Exception as exc:
    print(redact(f"运行失败: {type(exc).__name__}: {exc}", args.team))
    return 1
  finally:
    if file is not None:
      try:
        file.close()
      except OSError:
        pass


if __name__ == "__main__":
  raise SystemExit(main())
