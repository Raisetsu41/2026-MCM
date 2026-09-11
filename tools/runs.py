# 查看全部演练局.
# 数据来源优先级: 模拟器官方统计库 > 自建指令日志(code/outputs/q3_*.jsonl).
# 官方统计库由模拟器自己记账, 含真值、清除数、定位清除总时间与程序运行时间.
# 用法: python -X utf8 tools\runs.py
from __future__ import annotations

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


def official() -> dict[str, dict]:
  """模拟器自己记录的官方成绩, 按案例编码索引."""
  if not db.is_file():
    return {}
  stats: dict[str, dict] = {}
  try:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    for row in con.execute(
        "select case_code, jammer_count, cleared_jammer_count,"
        " virtual_time_us, program_run_duration_ms, state"
        " from practice_statistics_tasks order by id"):
      stats[row["case_code"]] = dict(row)
    con.close()
  except sqlite3.Error as exc:
    print(f"读取官方统计库失败: {exc}")
  return stats


def read_journal(path: Path) -> dict[str, float]:
  cleared = 0
  virtual = None
  for line in path.read_text(encoding="utf-8").splitlines():
    try:
      item = json.loads(line)
    except json.JSONDecodeError:
      continue
    body = item.get("response") or {}
    if body.get("accepted") is not True:
      continue
    if item.get("path") == "/clear" and body.get("clear_result") == "success":
      cleared += 1
    if body.get("virtual_time_s") is not None:
      virtual = float(body["virtual_time_s"])
  return {"cleared": cleared, "virtual": virtual} if virtual else {}


def main() -> int:
  if not logs.is_dir():
    print(f"未找到日志目录: {logs}")
    return 1
  stats = official()
  runs = []
  for path in logs.glob("*.result.json"):
    try:
      info = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
      continue
    try:
      window = datetime.strptime(
        str(info["window_started_at_utc"])[:19], "%Y-%m-%dT%H:%M:%S").replace(
        tzinfo=timezone.utc)
    except (KeyError, ValueError):
      window = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    runs.append({"info": info, "window": window})
  runs.sort(key=lambda r: r["window"])
  if not runs:
    print("尚无演练记录")
    return 0

  pool = []
  for folder in journal_dirs:
    if folder.is_dir():
      pool.extend(sorted(folder.glob("q3_*.jsonl")))
  for run in runs:
    best, gap = None, 180.0
    for path in pool:
      diff = abs(path.stat().st_mtime - run["window"].timestamp())
      if diff < gap:
        best, gap = path, diff
    run["local"] = read_journal(best) if best else {}
    if best:
      pool.remove(best)

  print(f"{'北京时间':<17}{'案例编码':<22}{'真值':>5}{'清除':>5}{'总时间/s':>11}"
        f"{'平均/s':>9}{'程序/ms':>9}   来源")
  for run in runs:
    info = run["info"]
    code = str(info.get("case_code"))
    stamp = run["window"].astimezone(BEIJING).strftime("%m-%d %H:%M:%S")
    truth = info.get("jammer_count")
    stat = stats.get(code)
    if stat:
      cleared = stat["cleared_jammer_count"]
      total = (stat["virtual_time_us"] or 0) / 1e6
      program = stat["program_run_duration_ms"]
      source = "官方库"
    else:
      local = run["local"]
      cleared = local.get("cleared")
      total = local.get("virtual")
      program = None
      source = "本地日志" if cleared is not None else "无数据"
    mean = f"{total / cleared:.1f}" if total and cleared else "-"
    note = source
    if cleared is not None and truth is not None and cleared != truth:
      note += " 漏源"
    print(f"{stamp:<17}{code:<22}{truth:>5}{str(cleared if cleared is not None else '-'):>5}"
          f"{(f'{total:.1f}' if total else '-'):>11}{mean:>9}"
          f"{str(program if program is not None else '-'):>9}   {note}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
