# v2 独立入口, 已验证的 main_fast.py 及其依赖保持不变.
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

from robot.agent_fast_v2 import Q3FastAgentV2  # noqa: E402
from robot.client import ApiClient  # noqa: E402
from robot.fast_geometry import path_length  # noqa: E402
from robot.main_fast import attach_logging, redact  # noqa: E402
from robot.q4_agent_fast_v2 import Q4FastAgentV2  # noqa: E402


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="Q3/Q4 fast v2")
  parser.add_argument("--team", default=os.environ.get("CUMCM_TEAM_NO", ""))
  parser.add_argument("--problem", type=int, choices=(3, 4), default=3)
  parser.add_argument("--base-url", default="http://127.0.0.1:2026")
  parser.add_argument("--wait-s", type=float, default=300.0)
  parser.add_argument("--q4-scan", choices=("compact", "rings", "lattice"), default="compact")
  parser.add_argument("--no-opportunistic", action="store_true")
  parser.add_argument("--no-inline", action="store_true")
  parser.add_argument("--log-dir", help="可选动作日志目录, 默认不写文件")
  args = parser.parse_args(argv)
  if not args.team:
    print("请用 --team 传入队号, 或设置 CUMCM_TEAM_NO")
    return 2
  if not 0 < args.wait_s < float("inf"):
    print("--wait-s 必须是有限正数")
    return 2
  file: TextIO | None = None
  try:
    client = ApiClient(args.base_url, args.team)
    options = {"opportunistic": not args.no_opportunistic, "inline": not args.no_inline}
    agent = (Q3FastAgentV2(client, **options) if args.problem == 3 else
             Q4FastAgentV2(client, scan=args.q4_scan, **options))
    if args.log_dir:
      folder = Path(args.log_dir).resolve()
      folder.mkdir(parents=True, exist_ok=True)
      name = f"q{args.problem}_fast_v2_{time.time_ns()}.jsonl"
      file = (folder / name).open("x", encoding="utf-8")
      attach_logging(client, file, args.team)
    print(f"Q{args.problem} fast v2 已就绪, 扫描点数 {len(agent.sites)}")
    if args.problem == 4:
      print(f"22 点连续域认证: {agent.compact_certified}, 检查单元 {agent.certificate_cells}")
    print("等待官方测试窗口")
    enter = client.enter_when_open(wait_s=args.wait_s)
    started = time.perf_counter()
    result = agent.run(enter_body=enter)
    out = {"problem": args.problem, "variant": "fast_v2",
           "cleared": result.cleared, "absent_certified": result.absent_certified,
           "measures": result.measures, "clear_calls": result.clear_calls,
           "inferred_negative_count": agent.inferred_negative_count,
           "scan_site_count": len(agent.sites),
           "scan_seed_path_m": path_length(agent.sites, agent.sites[0]),
           "virtual_time_s": result.virtual_time_s,
           "mean_virtual_per_cleared_s": result.virtual_time_s / result.cleared,
           "program_s": round(time.perf_counter() - started, 3), "complete": result.complete}
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
