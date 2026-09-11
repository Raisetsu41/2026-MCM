# 生成问题 1 和问题 2 的数值证据与图表.
# 输出 CSV, JSON, PDF 和运行日志.
from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from geometry_solver import (  # noqa: E402
  error_propagation,
  localization_diameter_bounds,
  pareto_second_sites_cone,
  robust_candidate_mask,
  second_site_metrics_sources,
  sector_max_distance,
  source_cone_samples,
)


DEBUG = True
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42

fig_dir = root / "figures"
res_dir = root / "results"
out_dir = root / "code" / "outputs"
logs: list[str] = []


def dbg(s: str) -> None:
  logs.append(s)
  if DEBUG:
    print("[debug] " + s)


def save_csv(path: Path, head: list[str], rows: list[list[float | int | str]]) -> None:
  with path.open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(head)
    w.writerows(rows)


def q1_result() -> dict[str, float | int]:
  g = np.array([500.0, 300.0])
  s = np.array([[-300.0, -100.0], [900.0, -500.0], [-100.0, 1000.0]])
  ang = np.degrees(np.arctan2(g[1] - s[:, 1], g[0] - s[:, 0])) % 360.0
  bounds = localization_diameter_bounds(s, ang, err_deg=1.0, n_bound=720)
  p = np.asarray(bounds["outer_poly"])
  hp = np.asarray(bounds["halfplanes"])
  d = float(bounds["upper_m"])
  pair = np.asarray(bounds["upper_pair"])
  vio = float(np.max(hp[:, :2] @ p.T - hp[:, 2, None]))
  save_csv(res_dir / "q1_定位多边形.csv", ["顶点序号", "横坐标(米)", "纵坐标(米)"],
           [[i + 1, x, y] for i, (x, y) in enumerate(p)])

  fig, ax = plt.subplots(figsize=(6.2, 5.2))
  z = np.vstack((p, p[0]))
  ax.fill(z[:, 0], z[:, 1], color="#4C78A8", alpha=0.22)
  ax.plot(z[:, 0], z[:, 1], color="#2F5D8A", lw=1.5, label="定位区域")
  ax.scatter(s[:, 0], s[:, 1], marker="^", s=45, color="#F58518", label="检测点")
  ax.plot(pair[:, 0], pair[:, 1], color="#D62728", lw=2.0, label="区域直径")
  for x, a in zip(s, ang):
    u = np.array([math.cos(math.radians(a)), math.sin(math.radians(a))])
    ax.plot([x[0], x[0] + 1400 * u[0]], [x[1], x[1] + 1400 * u[1]],
            color="#777777", lw=0.8, ls="--")
  ax.set_xlabel("横坐标 米")
  ax.set_ylabel("纵坐标 米")
  ax.axis("equal")
  ax.grid(alpha=0.2)
  ax.legend(frameon=False)
  fig.tight_layout()
  fig.savefig(fig_dir / "q1_定位区域与直径.pdf", bbox_inches="tight")
  plt.close(fig)

  side = 100.0
  tri = np.array([[0.0, 0.0], [side, 0.0], [side / 2.0, math.sqrt(3.0) * side / 2.0]])
  cen = tri.mean(axis=0)
  fig, ax = plt.subplots(figsize=(5.6, 5.2))
  q = np.vstack((tri, tri[0]))
  ax.fill(q[:, 0], q[:, 1], color="#4C78A8", alpha=0.18)
  ax.plot(q[:, 0], q[:, 1], color="#2F5D8A", lw=1.7, label="等边三角形")
  ax.add_patch(plt.Circle((side / 2.0, 0.0), side / 2.0, fill=False,
                          color="#D62728", lw=1.8, label="直径圆"))
  ax.add_patch(plt.Circle(cen, side / math.sqrt(3.0), fill=False,
                          color="#54A24B", lw=1.8, ls="--", label="最小外接圆"))
  ax.scatter(tri[:, 0], tri[:, 1], color="#2F5D8A", s=35)
  ax.set_xlabel("横坐标")
  ax.set_ylabel("纵坐标")
  ax.axis("equal")
  ax.grid(alpha=0.2)
  ax.legend(frameon=False)
  fig.tight_layout()
  fig.savefig(fig_dir / "q1_荣格定理反例.pdf", bbox_inches="tight")
  plt.close(fig)
  dbg(f"q1 顶点数 {len(p)} 直径 {d:.9f} 最大约束违反 {vio:.3e}")
  return {
    "vertex_count": len(p),
    "diameter_m": d,
    "diameter_lower_m": float(bounds["lower_m"]),
    "diameter_gap_m": max(d - float(bounds["lower_m"]), 0.0),
    "constraint_violation": vio,
    "jung_side_m": side,
    "diameter_circle_radius_m": side / 2.0,
    "minimum_cover_radius_m": side / math.sqrt(3.0),
  }


def q2_result() -> dict[str, float | int | list[float]]:
  s1 = np.array([0.0, 0.0])
  ang = 25.0
  ranges = (5.0, 1500.0)
  a = np.linspace(0.0, 1500.0, 151)
  b = np.linspace(-800.0, 800.0, 161)
  aa, bb = np.meshgrid(a, b)
  t = math.radians(ang)
  u = np.array([math.cos(t), math.sin(t)])
  v = np.array([-math.sin(t), math.cos(t)])
  cand = s1 + aa.ravel()[:, None] * u + bb.ravel()[:, None] * v
  ok = robust_candidate_mask(
    cand, s1, ang, ranges, err_deg=1.0,
    recv_rad=1000.0, time_limit_s=260.0)
  p = cand[ok]
  if len(p) == 0:
    raise RuntimeError("candidate region is empty")
  res = pareto_second_sites_cone(
    p, s1, ang, ranges, err_deg=1.0, recv_rad=1000.0,
    quantile=1.0, min_coverage=1.0, n_range=41, n_angle=21)
  ids = np.flatnonzero(res["pareto"])
  if len(ids) == 0:
    raise RuntimeError("pareto set is empty")
  # 20 m 确定性门槛不可行时, 先最小化误差界, 再最小化耗时.
  best_error = float(np.min(res["error_bound"][ids]))
  accurate = ids[res["error_bound"][ids] <= best_error + 1e-9]
  pick = accurate[int(np.argmin(res["time_s"][accurate]))]
  rows: list[list[float | int | str]] = []
  for i in ids:
    rows.append([float(p[i, 0]), float(p[i, 1]), float(res["angle_cost"][i]),
                 float(res["gdop"][i]), float(res["time_s"][i]),
                 float(res["coverage"][i]), float(res["error_bound"][i]),
                 float(res["rms"][i]),
                 int(i == pick)])
  save_csv(res_dir / "q2_Pareto候选点.csv",
           ["横坐标(米)", "纵坐标(米)", "角度代价", "GDOP(米每弧度)",
            "第二次检测耗时(秒)", "接收覆盖率", "确定性误差界(米)",
            "统计位置RMS(米)", "是否折中点"], rows)

  fig, ax = plt.subplots(1, 2, figsize=(10.8, 4.7))
  ax[0].scatter(cand[:, 0], cand[:, 1], s=1.0, color="#D9D9D9", alpha=0.45)
  ax[0].scatter(p[:, 0], p[:, 1], s=3.0, color="#4C78A8", alpha=0.45,
                label="解析候选区")
  ax[0].scatter(p[ids, 0], p[ids, 1], s=18.0, color="#D62728", label="Pareto点")
  ax[0].scatter(p[pick, 0], p[pick, 1], s=65.0, marker="*", color="#F2CF5B",
                edgecolor="#333333", label="折中点")
  ax[0].scatter([s1[0]], [s1[1]], marker="^", s=45, color="#111111", label="第一检测点")
  ax[0].set_xlabel("横坐标 米")
  ax[0].set_ylabel("纵坐标 米")
  ax[0].axis("equal")
  ax[0].grid(alpha=0.2)
  ax[0].legend(frameon=False, fontsize=8)
  ax[1].scatter(res["time_s"], res["error_bound"], s=5.0,
                color="#4C78A8", alpha=0.35)
  ax[1].scatter(res["time_s"][ids], res["error_bound"][ids], s=22.0,
                color="#D62728", label="Pareto前沿")
  ax[1].scatter([res["time_s"][pick]], [res["error_bound"][pick]], s=70.0,
                marker="*", color="#F2CF5B", edgecolor="#333333", label="折中点")
  ax[1].set_xlabel("第二次检测耗时 秒")
  ax[1].set_ylabel("确定性最坏位置误差界 米")
  ax[1].grid(alpha=0.2)
  ax[1].legend(frameon=False, fontsize=8)
  fig.tight_layout()
  fig.savefig(fig_dir / "q2_候选区域与Pareto前沿.pdf", bbox_inches="tight")
  plt.close(fig)

  ortho = error_propagation((0, 0), [(-3, 0), (0, -4)])
  full_src = source_cone_samples(s1, ang, ranges, 1.0, n_range=61, n_angle=31)
  worst = second_site_metrics_sources(
    p[pick:pick + 1], s1, full_src, quantile=1.0)
  far = float(sector_max_distance(p[pick:pick + 1], s1, ang, ranges, 1.0)[0])
  dbg(f"q2 候选数 {len(p)} Pareto数 {len(ids)} 折中点 {p[pick].tolist()}")
  dbg(f"q2 保证最远距离 {far:.6f} 确定性误差界 {float(worst['error_bound'][0]):.6f}")
  dbg(f"q2 正交解析例 GDOP {float(ortho['gdop']):.12f} RMS {float(ortho['rms']):.12f}")
  return {
    "candidate_count": len(p),
    "pareto_count": len(ids),
    "selected_site_m": p[pick].tolist(),
    "selected_angle_cost": float(res["angle_cost"][pick]),
    "selected_gdop_m_per_rad": float(res["gdop"][pick]),
    "selected_rms_m": float(res["rms"][pick]),
    "selected_error_bound_m": float(res["error_bound"][pick]),
    "selected_time_s": float(res["time_s"][pick]),
    "selected_coverage": float(res["coverage"][pick]),
    "error_bound_20m_feasible": bool(np.any(res["error_bound"] <= 20.0)),
    "guaranteed_max_distance_m": far,
    "sampled_worst_rms_m": float(worst["rms"][0]),
    "sampled_worst_error_bound_m": float(worst["error_bound"][0]),
    "orthogonal_test_gdop": float(ortho["gdop"]),
    "orthogonal_test_rms_m": float(ortho["rms"]),
  }


def sensitivity_result() -> dict[str, list[float | int]]:
  g = np.array([500.0, 300.0])
  s = np.array([[-300.0, -100.0], [900.0, -500.0], [-100.0, 1000.0]])
  ang = np.degrees(np.arctan2(g[1] - s[:, 1], g[0] - s[:, 0])) % 360.0
  err = np.array([0.5, 0.75, 1.0, 1.25, 1.5])
  dia = []
  for e in err:
    res = localization_diameter_bounds(s, ang, err_deg=float(e), n_bound=720)
    dia.append(float(res["upper_m"]))
  save_csv(res_dir / "q1_角误差敏感性.csv", ["角误差上界(度)", "定位区域直径(米)"],
           [[float(e), float(d)] for e, d in zip(err, dia)])

  a = np.linspace(0.0, 1500.0, 151)
  b = np.linspace(-800.0, 800.0, 161)
  aa, bb = np.meshgrid(a, b)
  t = math.radians(25.0)
  u = np.array([math.cos(t), math.sin(t)])
  v = np.array([-math.sin(t), math.cos(t)])
  cand = aa.ravel()[:, None] * u + bb.ravel()[:, None] * v
  tol = np.array([0.5, 0.75, 1.0, 1.25, 1.5])
  cnt1, cnt2 = [], []
  for e in tol:
    cnt1.append(int(robust_candidate_mask(
      cand, (0, 0), 25.0, (5.0, 1500.0), float(e), 1000.0, 260.0).sum()))
    cnt2.append(int(robust_candidate_mask(
      cand, (0, 0), 25.0, (5.0, 1500.0), float(e), 1500.0, 260.0).sum()))
  save_csv(res_dir / "q2_候选区敏感性.csv",
           ["测向误差上界(度)", "1000米保证区点数", "1500米机会区点数"],
           [[float(e), a1, a2] for e, a1, a2 in zip(tol, cnt1, cnt2)])

  fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.0))
  ax[0].plot(err, dia, marker="o", color="#4C78A8")
  ax[0].set_xlabel("示向误差上界 度")
  ax[0].set_ylabel("定位区域直径 米")
  ax[0].grid(alpha=0.2)
  ax[1].plot(tol, cnt1, marker="o", color="#4C78A8", label="1000米保证区")
  ax[1].plot(tol, cnt2, marker="s", color="#F58518", label="1500米机会区")
  ax[1].set_xlabel("测向误差上界 度")
  ax[1].set_ylabel("离散候选点数")
  ax[1].grid(alpha=0.2)
  ax[1].legend(frameon=False)
  fig.tight_layout()
  fig.savefig(fig_dir / "问题1与问题2_敏感性.pdf", bbox_inches="tight")
  plt.close(fig)
  dbg(f"敏感性 误差直径范围 {min(dia):.6f} 到 {max(dia):.6f}")
  return {
    "bearing_error_deg": err.tolist(),
    "diameter_m": [float(x) for x in dia],
    "cone_error_deg": tol.tolist(),
    "guaranteed_count": cnt1,
    "opportunity_count": cnt2,
  }


def main() -> None:
  st = time.perf_counter()
  fig_dir.mkdir(parents=True, exist_ok=True)
  res_dir.mkdir(parents=True, exist_ok=True)
  out_dir.mkdir(parents=True, exist_ok=True)
  out = {
    "problem1": q1_result(),
    "problem2": q2_result(),
    "sensitivity": sensitivity_result(),
  }
  out["elapsed_s"] = time.perf_counter() - st
  (res_dir / "geometry_summary.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
  dbg(f"全部结果无 NaN {int(np.all(np.isfinite([out['elapsed_s']])))}")
  dbg(f"运行耗时 {out['elapsed_s']:.6f} 秒")
  (out_dir / "geometry.log").write_text("\n".join(logs) + "\n", encoding="utf-8")


if __name__ == "__main__":
  main()
