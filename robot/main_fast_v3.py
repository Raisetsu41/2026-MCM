# 唯一提速入口: 动作账只统计 accepted 的唯一请求, 日志不含队号.
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import TextIO


root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from robot.agent_fast_v3 import Q3FastAgentV3  # noqa: E402
from robot.client import ApiClient  # noqa: E402
from robot.fast_geometry_v3 import path_length  # noqa: E402
def redact(value: object, team: str) -> object:
  if isinstance(value, dict):
    return {key: "<redacted>" if key in {"robot_id", "team_no"}
            else redact(item, team) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [redact(item, team) for item in value]
  if isinstance(value, str) and team:
    return value.replace(team, "<redacted>")
  return value
from robot.q4_agent_fast_v3 import Q4FastAgentV3  # noqa: E402


def instrument(client: ApiClient, agent, file: TextIO | None, team: str) -> dict:
  original = client._post_once
  seen: set[tuple[str, str]] = set()
  positions: set[tuple[float, float]] = {(0.0, 0.0)}
  measured: set[tuple[int, float, float]] = set()
  state = {"pos": (0.0, 0.0), "channel": 1}
  stats = {"distance_m": 0.0, "switches": 0, "site_revisits": 0,
           "duplicate_site_channel": 0, "clear_failures": 0,
           "phase": {}, "channels": {}, "segments_m": [], "log_errors": 0}

  def wrapped(path: str, payload: dict) -> tuple[int, dict]:
    phase = getattr(agent, "phase", "startup")
    status, body = original(path, payload)
    key = (path, str(payload.get("request_id")))
    unique = status == 200 and body.get("accepted") is True and key not in seen
    if unique:
      seen.add(key)
      if path in {"/measure", "/clear"}:
        channel = int(payload["channel"])
        p = payload["position"]
        pos = (float(p["x"]), float(p["y"]))
        distance = math.dist(state["pos"], pos)
        switched = int(path == "/measure" and channel != state["channel"])
        stats["distance_m"] += distance
        stats["switches"] += switched
        stats["segments_m"].append(distance)
        if distance > 1e-7 and pos in positions:
          stats["site_revisits"] += 1
        positions.add(pos)
        state["pos"] = pos
        part = stats["phase"].setdefault(phase, Counter())
        part["distance_m"] += distance
        part["switches"] += switched
        part[path.lstrip("/")] += 1
        track = stats["channels"].setdefault(str(channel), Counter())
        track[path.lstrip("/")] += 1
        if path == "/measure":
          state["channel"] = channel
          measurement = (channel, *pos)
          stats["duplicate_site_channel"] += int(measurement in measured)
          measured.add(measurement)
          result = str(body.get("measure_result"))
          part[result] += 1
          track[result] += 1
        elif body.get("clear_result") == "no_target_in_range":
          stats["clear_failures"] += 1
          part["clear_failures"] += 1
          track["clear_failures"] += 1
    if file is not None:
      record = {"t": time.time_ns() // 1_000_000, "variant": "fast_v3",
                "phase": phase, "path": path, "request": payload,
                "status": status, "response": body, "unique_accepted": unique}
      try:
        file.write(json.dumps(redact(record, team), ensure_ascii=False) + "\n")
        file.flush()
      except OSError:
        stats["log_errors"] += 1
    return status, body

  client._post_once = wrapped
  return stats


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="Q3/Q4 fast v3")
  parser.add_argument("--team", default=os.environ.get("CUMCM_TEAM_NO", ""))
  parser.add_argument("--problem", type=int, choices=(3, 4), default=3)
  parser.add_argument("--base-url", default="http://127.0.0.1:2026")
  parser.add_argument("--wait-s", type=float, default=300.0)
  parser.add_argument("--q4-scan", choices=("compact", "rings", "lattice"), default="compact")
  parser.add_argument("--empty-limit", type=int, choices=range(6), default=5)
  parser.add_argument("--no-opportunistic", action="store_true")
  parser.add_argument("--no-inline", action="store_true")
  parser.add_argument("--log-dir", help="可选脱敏动作日志目录")
  args = parser.parse_args(argv)
  if not args.team or not math.isfinite(args.wait_s) or args.wait_s <= 0:
    print("请提供 --team 或 CUMCM_TEAM_NO, --wait-s 必须是有限正数")
    return 2
  file: TextIO | None = None
  try:
    client = ApiClient(args.base_url, args.team)
    options = {"opportunistic": not args.no_opportunistic, "inline": not args.no_inline}
    agent = (Q3FastAgentV3(client, **options) if args.problem == 3 else
             Q4FastAgentV3(client, scan=args.q4_scan, empty_limit=args.empty_limit, **options))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for name in ("agent_fast*.py", "q4_agent_fast*.py", "fast_geometry*.py", "main_fast*.py")
              for p in sorted((root / "robot").glob(name))}
    if args.log_dir:
      folder = Path(args.log_dir).resolve()
      folder.mkdir(parents=True, exist_ok=True)
      file = (folder / f"q{args.problem}_fast_v3_{time.time_ns()}.jsonl").open("x", encoding="utf-8")
      file.write(json.dumps({"variant": "fast_v3", "source_sha256": hashes}) + "\n")
    stats = instrument(client, agent, file, args.team)
    print(f"Q{args.problem} fast v3 已就绪, 扫描点 {len(agent.sites)}, 等待官方演练窗口")
    enter = client.enter_when_open(wait_s=args.wait_s)
    started = time.perf_counter()
    result = agent.run(enter_body=enter)
    lengths = stats.pop("segments_m")
    bins = [0, 20, 100, 300, 1000, math.inf]
    stats["segment_bins"] = {f"({lo},{hi}]": sum(lo < x <= hi for x in lengths)
                             for lo, hi in zip(bins, bins[1:])}
    stats["stationary_actions"] = sum(x <= 1e-7 for x in lengths)
    stats["max_segment_m"] = max(lengths, default=0.0)
    accounted = (stats["distance_m"] / 5 + 5 * result.measures + stats["switches"]
                 + 5 * result.cleared + 3 * stats["clear_failures"])
    out = {"problem": args.problem, "variant": "fast_v3", "source_sha256": hashes,
           "cleared": result.cleared, "absent_certified": result.absent_certified,
           "channel_certificates": {str(k): t.status for k, t in agent.tracks.items()},
           "measures": result.measures, "clear_calls": result.clear_calls,
           "branches": dict(agent.branches), "actions": stats,
           "inferred_negative_count": agent.inferred_negative_count,
           "scan_site_count": len(agent.sites),
           "scan_seed_path_m": path_length(agent.sites, agent.sites[0]),
           "virtual_time_s": result.virtual_time_s,
           "time_account_residual_s": result.virtual_time_s - float(enter["virtual_time_s"]) - accounted,
           "mean_virtual_per_cleared_s": result.virtual_time_s / result.cleared,
           "program_s": time.perf_counter() - started, "complete": result.complete}
    text = json.dumps(redact(out, args.team), ensure_ascii=False)
    print(text)
    if file is not None:
      try:
        file.write(json.dumps({"summary": redact(out, args.team)}, ensure_ascii=False) + "\n")
      except OSError:
        print("任务已完成, 但日志汇总写入失败")
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
