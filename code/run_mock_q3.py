# 在本地 Mock 模拟器上批量验证 Q3 闭环.
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path


root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from robot.agent import Q3Agent  # noqa: E402
from robot.client import ApiClient  # noqa: E402
from robot.mock_server import MockArena, MockServer  # noqa: E402


def run_case(seed: int) -> dict[str, float | int | bool]:
  arena = MockArena.random_q3(seed)
  expected = len(arena.jammers)
  journal = root / "code" / "outputs" / f"q3_mock_{seed}.jsonl"
  if journal.exists():
    journal.unlink()
  start = time.perf_counter()
  with MockServer(arena) as server:
    client = ApiClient(server.url, arena.robot_id, journal=journal)
    result = Q3Agent(client).run()
  elapsed = time.perf_counter() - start
  return {
    "随机种子": seed,
    "干扰源总数": expected,
    "清除数": result.cleared,
    "清除率": result.cleared / expected,
    "虚拟时间(秒)": result.virtual_time_s,
    "平均清除时间(秒)": result.virtual_time_s / result.cleared,
    "检测次数": result.measures,
    "清除调用次数": result.clear_calls,
    "程序时间(秒)": elapsed,
    "闭环完成": result.complete,
  }


def main() -> None:
  out_dir = root / "code" / "outputs"
  res_dir = root / "results"
  out_dir.mkdir(parents=True, exist_ok=True)
  res_dir.mkdir(parents=True, exist_ok=True)
  rows = [run_case(seed) for seed in range(20260911, 20260916)]
  path = res_dir / "q3_Mock演练统计.csv"
  with path.open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  summary = {
    "cases": len(rows),
    "all_complete": all(bool(row["闭环完成"]) for row in rows),
    "min_clear_rate": min(float(row["清除率"]) for row in rows),
    "mean_virtual_s": sum(float(row["虚拟时间(秒)"]) for row in rows) / len(rows),
    "max_program_s": max(float(row["程序时间(秒)"]) for row in rows),
  }
  (res_dir / "q3_mock_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
  print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
  main()
