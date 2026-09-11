# 读取模拟器官方统计队列.
# 这是"官方口径"数据的唯一本地来源: 模拟器自己记录并上报服务端的成绩.
# 用法:
#   python -X utf8 robot\official_stats.py                # 演练 + 正式 全看
#   python -X utf8 robot\official_stats.py --kind practice
#   python -X utf8 robot\official_stats.py --json out.json
#   python -X utf8 robot\official_stats.py --csv out.csv
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path


root = Path(__file__).resolve().parents[1]
DATA_DIR = root / "Jammers-simulator" / "JammersSimulatorData"
BEIJING = timezone(timedelta(hours=8))

PRACTICE_DB = "practice-statistics-queue.sqlite3"
PRACTICE_TABLE = "practice_statistics_tasks"
FORMAL_DB = "formal-statistics-queue.sqlite3"
FORMAL_TABLE = "statistics_tasks"

# 官方字段 -> 中文说明. 这些字段由模拟器自己记账, 不受本方程序影响.
FIELD_DOC = {
    "practice_run_no": "演练局号",
    "case_code": "案例编码(表 1 要填)",
    "problem_no": "题号",
    "jammer_count": "干扰源总数(真题真值, 仅演练可见)",
    "cleared_jammer_count": "清除个数(表 1 要填)",
    "virtual_time_us": "定位清除总时间(微秒, 虚拟时钟)",
    "program_run_duration_ms": "程序运行时间(毫秒, 官方口径)",
    "measure_accepted_count": "被接受的检测次数",
    "channel_switch_count": "频道切换次数(每次计 1 秒虚拟时间)",
    "clear_failure_count": "清除失败次数(/clear 返回未发现)",
    "entered": "是否成功 /enter",
    "end_reason": "结束原因",
    "state": "上报状态",
    "attempt_count": "上报尝试次数",
    "last_error_code": "最后错误码",
    "received_at_ms": "服务器接收时刻",
    "formal_index": "正式测试序号(1/2/3)",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="显示模拟器记录的官方成绩(演练与正式).")
  parser.add_argument("--data-dir", default=str(DATA_DIR),
                      help="JammersSimulatorData 目录")
  parser.add_argument("--kind", choices=["all", "practice", "formal"],
                      default="all", help="查看哪一类统计")
  parser.add_argument("--json", default=None, help="同时导出为 JSON")
  parser.add_argument("--csv", default=None, help="同时导出为 CSV")
  return parser.parse_args(argv)


def _fmt_time(ms: object) -> str:
  if not ms:
    return "-"
  return datetime.fromtimestamp(float(ms) / 1000.0, BEIJING).strftime(
    "%Y-%m-%d %H:%M:%S")


def _read(db: Path, table: str) -> tuple[list[dict[str, object]], str]:
  if not db.exists():
    return [], f"未找到 {db.name}(尚未产生该类记录)"
  # 模拟器可能正在运行并持有写锁, 用只读模式打开避免相互影响.
  for uri in (f"file:{db}?mode=ro", f"file:{db}?mode=ro&immutable=1"):
    try:
      con = sqlite3.connect(uri, uri=True, timeout=2.0)
      con.row_factory = sqlite3.Row
      rows = [dict(r) for r in con.execute(f"select * from {table} order by id")]
      con.close()
      return rows, ""
    except sqlite3.Error as exc:
      last = f"{type(exc).__name__}: {exc}"
  return [], last


def _official(row: dict[str, object]) -> dict[str, object]:
  """把一行的原始字段换算成可直接写进论文的量."""
  us = row.get("virtual_time_us")
  total = float(us) / 1e6 if us is not None else None
  cleared = row.get("cleared_jammer_count")
  out: dict[str, object] = {
    "案例编码": row.get("case_code"),
    "清除个数": cleared,
    "定位清除总时间_s": total,
    "平均定位清除时间_s": (
      round(total / float(cleared), 6) if total is not None and cleared else None),
    "程序运行时间_ms": row.get("program_run_duration_ms"),
  }
  if "jammer_count" in row:
    total_j = row.get("jammer_count")
    out["干扰源总数"] = total_j
    out["清除比例"] = (
      round(float(cleared) / float(total_j), 6)
      if cleared is not None and total_j else None)
  return out


def _print_table(title: str, rows: list[dict[str, object]], note: str) -> None:
  print("=" * 78)
  print(title)
  print("=" * 78)
  if not rows:
    print(f"  (无记录) {note}")
    return
  for row in rows:
    off = _official(row)
    print(f"  案例编码 {row.get('case_code')}"
          + (f"   正式测试第 {row.get('formal_index')} 次"
             if row.get("formal_index") else
             f"   演练局号 {row.get('practice_run_no')}"))
    print(f"    干扰源总数(真值) : {off.get('干扰源总数', '不提供')}")
    print(f"    清除个数         : {off['清除个数']}")
    print(f"    清除比例         : {off.get('清除比例', '需自行核对')}")
    print(f"    定位清除总时间   : {off['定位清除总时间_s']} s")
    print(f"    平均定位清除时间 : {off['平均定位清除时间_s']} s   <- 表 1 要填")
    print(f"    程序运行时间     : {off['程序运行时间_ms']} ms  <- 表 1 要填")
    print(f"    检测被接受次数   : {row.get('measure_accepted_count')}"
          f"   频道切换 {row.get('channel_switch_count')} 次"
          f"   清除失败 {row.get('clear_failure_count')} 次")
    print(f"    进入/结束        : entered={row.get('entered')}"
          f"  end_reason={row.get('end_reason')}")
    print(f"    上报状态         : {row.get('state')}"
          f"  尝试 {row.get('attempt_count')} 次"
          f"  错误码 {row.get('last_error_code') or '无'}"
          f"  接收于 {_fmt_time(row.get('received_at_ms'))}")
    print()


def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  data_dir = Path(args.data_dir)
  collected: dict[str, list[dict[str, object]]] = {}

  if args.kind in ("all", "practice"):
    rows, note = _read(data_dir / PRACTICE_DB, PRACTICE_TABLE)
    collected["practice"] = rows
    _print_table("官方·问题 3/4 演练测试统计"
                 "(来源: practice-statistics-queue.sqlite3)", rows, note)

  if args.kind in ("all", "formal"):
    rows, note = _read(data_dir / FORMAL_DB, FORMAL_TABLE)
    collected["formal"] = rows
    _print_table("官方·正式测试统计"
                 "(来源: formal-statistics-queue.sqlite3)", rows, note)

  if args.kind in ("all", "formal"):
    rows, _ = _read(data_dir / "upload-queue.sqlite3", "upload_tasks")
    print("=" * 78)
    print("行为日志/统计包上报队列(应在确认后清空)")
    print("=" * 78)
    if not rows:
      print("  (空) 说明已上传成功; 若长期非空则检查网络")
    for row in rows:
      print(f"  kind={row.get('kind')} problem={row.get('problem_no')} "
            f"practice_run={row.get('practice_run_no')} "
            f"formal_index={row.get('formal_index')} case={row.get('case_code')} "
            f"state={row.get('state')} attempts={row.get('attempt_count')} "
            f"err={row.get('last_error_code') or '无'}")

  if args.json:
    path = Path(args.json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(collected, ensure_ascii=False, indent=2,
                               default=str), encoding="utf-8")
    print(f"\n已导出 JSON -> {path}")
  if args.csv:
    path = Path(args.csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = [{**{"类别": kind}, **row}
            for kind, rows in collected.items() for row in rows]
    if flat:
      with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
      print(f"已导出 CSV  -> {path}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
