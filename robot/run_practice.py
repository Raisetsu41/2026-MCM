# 问题 3 演练测试运行入口.
# 用法: python -X utf8 robot\run_practice.py --team <参赛队号>
# 本脚本只调用 /enter /measure /clear /exit, 不触碰模拟器界面与正式测试.
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path


root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from robot.agent import Q3Agent  # noqa: E402
from robot.client import ApiClient, ApiError  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="问题 3 演练测试: 等待接口开放后自动完成一轮定位与清除.")
  parser.add_argument(
    "--team", default=os.environ.get("CUMCM_TEAM_NO", ""),
    help="参赛队号, 必须与模拟器当前登录账号一致; "
         "默认取环境变量 CUMCM_TEAM_NO (请勿在源码中写死队号)")
  parser.add_argument("--base-url", default="http://127.0.0.1:2026",
                      help="模拟器接口地址, 端口被占用时按界面设置修改")
  parser.add_argument("--wait-s", type=float, default=300.0,
                      help="等待 5 秒倒计时结束/接口开放的最长秒数")
  parser.add_argument("--max-obs", type=int, default=8,
                      help="单个频道的最大观测次数")
  parser.add_argument("--err-deg", type=float, default=1.01,
                      help="测向误差安全余量(度), 题面为 1.0")
  parser.add_argument("--journal", default=None,
                      help="本地 JSONL 审计文件路径, 默认按时间戳自动命名")
  parser.add_argument("--summary", default=None,
                      help="本轮统计结果写入的 JSON 路径")
  parser.add_argument("--timeout-s", type=float, default=3.0,
                      help="单次 HTTP 请求超时(秒)")
  return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
  journal = Path(args.journal) if args.journal else (
    root / "code" / "outputs" / f"q3_practice_{stamp}.jsonl")
  summary_path = Path(args.summary) if args.summary else (
    root / "results" / f"q3_practice_{stamp}.json")

  if not args.team:
    print("[practice] 缺少参赛队号: 请用 --team <队号> 或设置环境变量 "
          "CUMCM_TEAM_NO (源码中不写死队号)")
    return 2
  print("[practice] 开始前确认: 模拟器已完成在线登录且不在测试中; "
        "同一时刻只能有一个机器狗程序在跑. 程序在接到 Ctrl+C 时会主动 /exit.")

  print(f"[practice] 队号={args.team} 接口={args.base_url}")
  print(f"[practice] 审计日志 -> {journal}")
  print("[practice] 等待接口开放 (请先在模拟器中点击开始演练测试)...")
  client = ApiClient(args.base_url, args.team, journal=journal,
                     timeout_s=args.timeout_s)
  started = time.perf_counter()
  try:
    enter = client.enter_when_open(wait_s=args.wait_s)
  except ApiError as exc:
    print(f"[practice] 进入失败: {exc}")
    return 2

  remain = float(enter.get("remaining_real_duration_s", 0.0))
  print(f"[practice] /enter 成功, 本局可用现实时间 {remain:.0f} 秒")
  if remain <= 0:
    print("[practice] 可用时间为 0, 立即退出以免被判超时")
    try:
      client.exit()
    except ApiError:
      pass
    return 3

  agent = Q3Agent(client, err_deg=args.err_deg, max_obs=args.max_obs)
  error: str | None = None
  result = None
  try:
    result = agent.run(enter_body=enter)
  except Exception as exc:  # noqa: BLE001 - 演练需要完整记录异常并优雅退出
    error = f"{type(exc).__name__}: {exc}"
    print(f"[practice] 运行中断: {error}")
    try:
      client.exit()
    except ApiError:
      pass
  elapsed = time.perf_counter() - started

  payload: dict[str, object] = {
    "team": args.team,
    "base_url": args.base_url,
    "started_at": stamp,
    "enter_remaining_s": remain,
    "program_s": elapsed,
    "error": error,
    "journal": str(journal),
  }
  if result is not None:
    payload.update({
      "cleared": result.cleared,
      "absent_certified": result.absent_certified,
      "measures": result.measures,
      "clear_calls": result.clear_calls,
      "virtual_time_s": result.virtual_time_s,
      "complete": result.complete,
      "mean_virtual_per_cleared_s": (
        result.virtual_time_s / result.cleared if result.cleared else None),
    })
  summary_path.parent.mkdir(parents=True, exist_ok=True)
  summary_path.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
  print(json.dumps(payload, ensure_ascii=False, indent=2))
  print(f"[practice] 统计已写入 {summary_path}")
  if result is not None:
    total = result.cleared + result.absent_certified
    print(f"[practice] 快速核对: 清除 {result.cleared} + 认证不存在 "
          f"{result.absent_certified} = {total} (20 个频道应全部有归属) | "
          f"清除调用 {result.clear_calls} vs 清除数 {result.cleared} | "
          f"现实 {elapsed:.1f}s / 预算 {remain:.0f}s")
    if total != 20:
      print("[practice] 警告: 存在既未清除也未认证的频道, 证书不完整")
    if result.clear_calls > result.cleared:
      print(f"[practice] 提示: {result.clear_calls - result.cleared} 次 /clear "
            "空跑(每次白花 3 秒), 说明定位区域收敛判据可以收紧")
    print("[practice] 请回到模拟器读取本局真值(干扰源总数), 核对 "
          "cleared == jammer_count; 不相等即为漏源(误认证为不存在)")
  return 0 if (result is not None and result.complete) else 1


if __name__ == "__main__":
  raise SystemExit(main())
