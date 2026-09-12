from __future__ import annotations

import csv
import json
import statistics
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from robot.q3_agent import Q3Agent
from robot.client import ApiClient
from robot.mock_server import MockArena, MockServer
from robot.q4_agent import Q4Agent

def run_q3(seed: int, schedule: str) -> dict[str, float | int | str | bool]:
  arena = MockArena.random_q3(seed)
  count = len(arena.jammers)
  started = time.perf_counter()
  with MockServer(arena) as server:
    result = Q3Agent(
      ApiClient(server.url, arena.robot_id), schedule=schedule).run()
  return {
    "problem": "Q3",
    "seed": seed,
    "schedule": schedule,
    "jammer_count": count,
    "cleared": result.cleared,
    "complete": result.complete,
    "virtual_time_s": result.virtual_time_s,
    "per_jammer_s": result.virtual_time_s / count,
    "move_m": arena.move_m,
    "measure_count": arena.measure_accepted_count,
    "switch_count": arena.channel_switch_count,
    "clear_failure_count": arena.clear_failure_count,
    "program_s": time.perf_counter() - started,
  }

def run_q4(seed: int) -> dict[str, float | int | str | bool]:
  arena = MockArena.random_q4(seed)
  count = len(arena.jammers)
  directional = sum(jammer.heading_deg is not None for jammer in arena.jammers)
  started = time.perf_counter()
  with MockServer(arena) as server:
    result = Q4Agent(ApiClient(server.url, arena.robot_id)).run()
  return {
    "problem": "Q4",
    "seed": seed,
    "schedule": "batch",
    "jammer_count": count,
    "directional_count": directional,
    "cleared": result.cleared,
    "complete": result.complete,
    "virtual_time_s": result.virtual_time_s,
    "per_jammer_s": result.virtual_time_s / count,
    "move_m": arena.move_m,
    "measure_count": arena.measure_accepted_count,
    "switch_count": arena.channel_switch_count,
    "clear_failure_count": arena.clear_failure_count,
    "program_s": time.perf_counter() - started,
  }

def mean(rows: list[dict[str, object]], key: str) -> float:
  return statistics.fmean(float(row[key]) for row in rows)

def main() -> None:
  rows: list[dict[str, object]] = []
  q3_seeds = list(range(20260911, 20260921))
  for seed in q3_seeds:
    rows.append(run_q3(seed, "immediate"))
    rows.append(run_q3(seed, "batch"))
  q4_seeds = list(range(20261001, 20261006))
  for seed in q4_seeds:
    rows.append(run_q4(seed))

  immediate = [row for row in rows if row["schedule"] == "immediate"]
  batch = [
    row for row in rows
    if row["problem"] == "Q3" and row["schedule"] == "batch"
  ]
  q4 = [row for row in rows if row["problem"] == "Q4"]
  saving = [
    float(old["virtual_time_s"]) - float(new["virtual_time_s"])
    for old, new in zip(immediate, batch)
  ]
  summary = {
    "q3_case_count": len(q3_seeds),
    "q3_all_complete": all(bool(row["complete"]) for row in immediate + batch),
    "q3_all_zero_clear_failure": all(
      int(row["clear_failure_count"]) == 0 for row in immediate + batch),
    "q3_immediate_mean_virtual_s": mean(immediate, "virtual_time_s"),
    "q3_batch_mean_virtual_s": mean(batch, "virtual_time_s"),
    "q3_batch_mean_saving_s": statistics.fmean(saving),
    "q3_batch_mean_saving_ratio": statistics.fmean(
      save / float(old["virtual_time_s"])
      for save, old in zip(saving, immediate)),
    "q3_batch_worst_virtual_s": max(float(row["virtual_time_s"]) for row in batch),
    "q4_case_count": len(q4_seeds),
    "q4_all_complete": all(bool(row["complete"]) for row in q4),
    "q4_mean_virtual_s": mean(q4, "virtual_time_s"),
    "q4_worst_virtual_s": max(float(row["virtual_time_s"]) for row in q4),
    "q4_mean_clear_failures": mean(q4, "clear_failure_count"),
  }

  result_dir = root / "results"
  result_dir.mkdir(parents=True, exist_ok=True)
  csv_path = result_dir / "mock_benchmark.csv"
  keys = sorted({key for row in rows for key in row})
  with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
  output = {"summary": summary, "runs": rows}
  (result_dir / "mock_benchmark.json").write_text(
    json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
  print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
  main()
