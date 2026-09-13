from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[1]
data_dir = root / "Jammers-simulator" / "JammersSimulatorData"
logs = data_dir / "behavior-logs"
db = data_dir / "practice-statistics-queue.sqlite3"
journal_dirs = [
    root / "code" / "outputs",
    root / "archive" / "runs" / "practice",
    root.parent / "ttmp" / "2026MCM-archive" / "_archive" / "runs" / "practice",
]
BEIJING = timezone(timedelta(hours=8))

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="summarize practice-run results")
  parser.add_argument("--problem", choices=["3", "4", "all"], default="3",
                      help="which problem's practice runs (default: 3)")
  return parser.parse_args(argv)

def official() -> dict[str, dict]:
  if not db.is_file():
    return {}
  rows: list[dict] = []
  for uri in (f"file:{db}?mode=ro", f"file:{db}?mode=ro&immutable=1"):
    try:
      con = sqlite3.connect(uri, uri=True, timeout=5)
      con.row_factory = sqlite3.Row
      rows = [dict(r) for r in con.execute(
          "select case_code, jammer_count, cleared_jammer_count,"
          " virtual_time_us, program_run_duration_ms, state"
          " from practice_statistics_tasks order by id")]
      con.close()
      break
    except sqlite3.Error:
      continue
  return {str(r["case_code"]): r for r in rows}

def read_journal(path: Path) -> dict[str, object]:
  cleared = failed = measures = 0
  virtual = None
  kinds: dict[str, int] = {}
  for line in path.read_text(encoding="utf-8").splitlines():
    try:
      item = json.loads(line)
    except json.JSONDecodeError:
      continue
    body = item.get("response") or {}
    if body.get("accepted") is not True:
      continue
    route = item.get("path")
    if route == "/measure":
      measures += 1
      kind = str(body.get("measure_result"))
      kinds[kind] = kinds.get(kind, 0) + 1
    elif route == "/clear":
      if body.get("clear_result") == "success":
        cleared += 1
      else:
        failed += 1
    if body.get("virtual_time_s") is not None:
      virtual = float(body["virtual_time_s"])
  if virtual is None:
    return {}
  return {"cleared": cleared, "failed": failed, "measures": measures,
          "virtual": virtual, "kinds": kinds}

def match(run: dict, pool: list[Path]) -> Path | None:
  best, gap = None, 300.0
  for path in pool:
    diff = abs(path.stat().st_mtime - run["window"].timestamp())
    if diff < gap:
      best, gap = path, diff
  if best:
    pool.remove(best)
  return best

def collect(problem: str) -> list[dict]:
  stats = official()
  runs = []
  for path in logs.glob(f"practice-p{problem}-*.result.json"):
    try:
      info = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
      continue
    try:
      window = datetime.strptime(
        str(info["window_started_at_utc"])[:19],
        "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    except (KeyError, ValueError):
      window = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    runs.append({"info": info, "window": window})
  runs.sort(key=lambda r: r["window"])

  pool: list[Path] = []
  for folder in journal_dirs:
    if folder.is_dir():
      pool.extend(sorted(folder.glob(f"q{problem}_*.jsonl")))
  for run in runs:
    run["journal"] = match(run, pool)
    run["local"] = read_journal(run["journal"]) if run["journal"] else {}
    run["stat"] = stats.get(str(run["info"].get("case_code")))
  return runs

def report(problem: str) -> None:
  runs = collect(problem)
  print("=" * 100)
  print(f"problem {problem} practice-run summary  (cases and truth from simulator behavior logs; metrics prefer the official queue)")
  print("=" * 100)
  if not runs:
    print("  no records yet")
    return
  print(f"{'Beijing time':<15}{'case code':<22}{'truth':>5}{'omni':>5}{'dir.':>5}"
        f"{'cleared':>5}{'empty':>5}{'meas':>6}{'total/s':>10}{'mean/s':>8}"
        f"{'per/src':>8}  verdict")
  total_ok = 0
  for run in runs:
    info = run["info"]
    code = str(info.get("case_code"))
    stamp = run["window"].astimezone(BEIJING).strftime("%m-%d %H:%M:%S")
    truth = info.get("jammer_count")
    omni = info.get("omnidirectional_jammer_count")
    direc = info.get("directional_jammer_count")
    stat, local = run["stat"], run["local"]
    if stat:
      cleared = stat["cleared_jammer_count"]
      virtual = (stat["virtual_time_us"] or 0) / 1e6
    else:
      cleared = local.get("cleared")
      virtual = local.get("virtual")
    failed = local.get("failed")
    measures = local.get("measures")
    mean = f"{virtual / cleared:.1f}" if virtual and cleared else "-"
    per = f"{virtual / truth:.1f}" if virtual and truth else "-"
    note = "official queue" if stat else ("local logs" if local else "no details")
    if cleared is not None and truth is not None:
      if cleared != truth:
        note += " missing source"
      else:
        total_ok += 1
        note += " ok"
    print(f"{stamp:<15}{code:<22}{truth:>5}{omni:>5}{direc:>5}"
          f"{str(cleared if cleared is not None else '-'):>5}"
          f"{str(failed if failed is not None else '-'):>5}"
          f"{str(measures if measures is not None else '-'):>6}"
          f"{(f'{virtual:.1f}' if virtual else '-'):>10}{mean:>8}{per:>8}  {note}")

  print("-" * 100)
  print(f"  valid runs {total_ok}/{len(runs)} with cleared == truth")
  gaps = [r for r in runs if r["stat"] is None]
  if gaps:
    print(f"  note: {len(gaps)} runs are missing from the official queue; metrics come from the local command logs")

def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  if not logs.is_dir():
    print(f"log directory not found: {logs}")
    return 1
  problems = ["3", "4"] if args.problem == "all" else [args.problem]
  for index, problem in enumerate(problems):
    if index:
      print()
    report(problem)
  return 0

if __name__ == "__main__":
  sys.exit(main())
