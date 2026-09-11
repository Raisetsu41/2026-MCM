# `robot/` 模块说明

机器狗（问题 3 / 问题 4）的在线定位与清除程序。上游依赖仓库根目录的 `geometry_solver.py`（半平面交、旋转卡壳、最小外接圆等几何算法），本身不重复实现几何。

```
robot/
├── __init__.py          模块出口（对外只暴露 4 个名字）
├── client.py            模拟器 HTTP 通信层
├── agent.py             决策/策略层（Q3 闭环 + 部分 Q4 构件）
├── mock_server.py       本地离线模拟器（自测用）
├── run_practice.py      ★ 连真实模拟器的运行入口（演练/正式通用）
└── official_stats.py    ★ 官方成绩读取器（读模拟器统计队列）
```

★ = 需要人工在终端执行的入口脚本；其余为被导入的库。

---

## 1. `__init__.py`（6 行）

**用途**：包出口。只重导出 `ApiClient`、`ApiError`、`MissionResult`、`Q3Agent` 四个名字，使调用方可以 `from robot import Q3Agent`，而不必关心内部文件划分。

---

## 2. `client.py`（210 行）— 通信层

**用途**：把附件 2 的 HTTP+JSON 协议封装成 4 个方法，并负责所有"协议纪律"，让上层策略代码不需要接触 HTTP。

**对外接口**

| 成员 | 作用 |
| --- | --- |
| `ApiError` | 统一异常。带 `rejected` 标志：`True` = 收到过响应但业务拒绝；`False` = 连接层面一直失败 |
| `ApiClient(base_url, robot_id, journal=..., timeout_s, max_retry, backoff_s, deadline_guard_s)` | 构造；`robot_id` **不写死**，由调用方传入 |
| `.enter()` | `POST /enter`，并把响应里的 `remaining_real_duration_s` 记为现实截止时刻 |
| `.enter_payload(payload)` | 用给定请求体调 `/enter` 并登记截止时刻（供下面两者复用） |
| `.enter_when_open(wait_s, request_id)` | **等待接口开放**：5 秒倒计时/非测试期间连接会被直接关闭，因此轮询；全程复用同一 `request_id`，命中幂等缓存不会重复进入 |
| `.measure(x, y, channel)` | `POST /measure` |
| `.clear(x, y, channel)` | `POST /clear` |
| `.exit()` | `POST /exit` |

**内部机制**

- `_post`：串行（`threading.Lock`）+ 有限重试 + 退避；**重试复用同一 `payload`/`request_id`**（附件 2 §5.3 的硬要求，避免重复计时）；对 429/500 重试，其余状态位立即失败。
- `_check_deadline`：每次非 `/exit` 请求前检查现实预算，留 `deadline_guard_s`（默认 2 s）余量，防止被判超时。
- `_write`：把每次往返（含失败）追加写入 JSONL 审计文件——**这是排错用的**，不是提交用的行为日志。
- `_new_id`：生成 `measure-000001` 这类会话内唯一且幂等的 `request_id`。

---

## 3. `agent.py`（279 行）— 策略层

**用途**：问题 3 的完整决策闭环，以及问题 4 需要的基础构件。所有动作都通过 `ApiClient` 发出。

**数据结构**

| 名称 | 含义 |
| --- | --- |
| `Observation` | 一次有效测向：站点坐标 + 示向度 |
| `ChannelTrack` | 单频道状态机：`status` / 观测列表 `obs` / 定位半径序列 `radii` / 清除点 |
| `MissionResult` | 本局汇总：`cleared`、`absent_certified`、`measures`、`clear_calls`、`virtual_time_s`、`complete` |

**Q3 策略函数**

| 函数 | 作用 |
| --- | --- |
| `q3_scan_sites(radius=1200)` | 7 点发现骨架（原点 + 半径 1200 m 的正六边形）。可保证目标圆内任一点到最近骨架点距离 ≤ **968.9016 m**（余量 31.1 m） |

**Q4 预留函数**（已实现，主循环尚未接入）

| 函数 | 作用 |
| --- | --- |
| `q4_scan_sites(step=990)` | 边长 990 m 的三角网格骨架，用于定向源的"凸包可见性"认证 |
| `fallback_clear_sites(...)` | 沿示向度扇形铺 20 m 覆盖网格（盲扫兜底） |
| `polygon_clear_sites(poly)` | 在定位多边形内铺 20 m 覆盖网格（兜底清除） |

**辅助函数**

| 函数 | 作用 |
| --- | --- |
| `safe_lateral_site(site, bearing, side)` | 由首个观测点构造第二观测点：沿示向度前推 400 m、侧偏 ±300 m（形成非共线基线） |

**核心类 `Q3Agent`**

| 方法 | 作用 |
| --- | --- |
| `__init__(client, err_deg=1.01, max_obs=8)` | 20 个频道各建一条 `ChannelTrack`；状态位掩码 `cleared_mask` / `absent_mask`；`entered_here` 记录是否由自己 `/enter` |
| `_measure` / `_clear` | 发动作并同步本地位置、当前频道、虚拟时钟 |
| `_record` | 处理检测结果：`near` → 直接原地清除；`direction` → 存观测并推进状态；其余为异常 |
| `_region` | 用全部观测调 `localization_polygon` 求交会多边形，再求**最小外接圆**（半径即定位不确定度） |
| `_next_site` | 选下一个观测点：首次观测后走横向基线；半径 ≤ 900 m 时直奔圆心；仍不安全则换另一侧 |
| `_fallback` | 定位支线失败时按多边形铺网格盲扫清除 |
| `_finish_channel` | 单频道最多 `max_obs` 次观测：半径 ≤ 20 m 就 `/clear`；半径不再下降或站点重复则走兜底 |
| **`run(enter_body=None)`** | **主循环**：`enter_body` 非空表示调用方已 `/enter`（不再重复）；否则自己进入。之后 7 站点 × 20 频道扫描发现 → 逐频道定位清除 → 未出现的频道认证为"不存在" → 20 个频道全部有归属才 `/exit` |

> `agent.py` 只覆盖问题 3；问题 4 的定向源可见性推理尚未写进主循环。

---

## 4. `mock_server.py`（242 行）— 离线自测模拟器

**用途**：在**不连真实模拟器、不消耗任何测试机会**的前提下验证策略与协议。用进程内 `ThreadingHTTPServer` 起一个临时端口，复现题面物理规则。

**复现的规则**

- 移动耗时 = 直线距离 / 5 m/s；`/measure` 固定 5 s；跨频道切换 1 s（**仅由 `/measure` 的 channel 变化触发，`/clear` 不触发也不改变当前频道**）；
- `/clear` 成功 5 s、未发现 3 s；清除半径 20 m；近距离阈值 5 m 返回 `near`；
- 有效接收半径逐源随机 ∈ [1000, 1500] m；定向源覆盖角 180°；
- **同一地点测向误差固定**（用 `sha256(坐标,频道)` 派生，复现"重复检测不改变误差"）；
- `request_id` 幂等：同 ID 同内容返回缓存响应，同 ID 改内容返回 409；
- **重复 `/enter` 返回 `accepted=False`**（与真实模拟器一致，用于锁死复现过的缺陷）。

**主要成员**

| 名称 | 作用 |
| --- | --- |
| `Jammer` | 一个干扰源：频道、坐标、有效接收半径、定向方向、是否已清除 |
| `MockArena` | 单局状态机：位置、当前频道、虚拟时钟、幂等缓存；`random_q3(seed, count)` 生成随机案例 |
| `MockArena.handle(path, payload)` | 纯函数式处理一条请求，返回 `(HTTP 状态, 响应体, 是否新动作)` |
| `MockServer(arena)` | 上下文管理器，`with MockServer(...) as srv:` 后用 `srv.url` 取临时地址 |

**已知不保真的地方**（重要）：不模拟"接口未开放时直接断连"的启动时序、不模拟 `accepted=false` 的多种业务原因、不模拟网络抖动。因此 **Mock 通过 ≠ 真实模拟器通过**。

---

## 5. `run_practice.py`（139 行）★ — 真实模拟器运行入口

**用途**：把程序接到真实模拟器上的唯一入口，演练与正式测试通用。

**用法**

```powershell
python -X utf8 robot\run_practice.py --team <参赛队号> [选项]
```

| 选项 | 默认 | 说明 |
| --- | --- | --- |
| `--team` | 环境变量 `CUMCM_TEAM_NO` | **参赛队号不入源码**；为空则拒绝启动 |
| `--base-url` | `http://127.0.0.1:2026` | 改过端口时同步 |
| `--wait-s` | 300 | 等待接口开放的最长秒数 |
| `--max-obs` | 8 | 单频道最大观测次数 |
| `--err-deg` | 1.01 | 测向误差安全余量（题面 1.0） |
| `--journal` / `--summary` | 按时间戳自动命名 | 审计 JSONL / 统计 JSON 的落盘位置 |

**执行流程**

1. 校验队号 → 打印赛前确认；
2. `client.enter_when_open()` 轮询等待 5 秒倒计时结束；
3. 读出官方 `remaining_real_duration_s` 作为预算（**不假定 1200**）；
4. `agent.run(enter_body=enter)` —— 把已接受的 `/enter` 响应传进去，**避免重复进入**；
5. 异常时兜底 `/exit`，并记录 `error`；
6. 打印统计 JSON + **快速核对行**（清除 + 认证是否等于 20、`/clear` 是否空跑、现实耗时 vs 预算），并提示回模拟器核对真值与漏源；
7. 退出码：0 = 完整闭环成功。

---

## 6. `official_stats.py`（190 行）★ — 官方成绩读取器

**用途**：读取**模拟器自己记账**的官方统计（唯一"官方口径"本地来源），而不是我们程序的账本。

**用法**

```powershell
python -X utf8 robot\official_stats.py                 # 演练 + 正式 + 上报表
python -X utf8 robot\official_stats.py --kind formal
python -X utf8 robot\official_stats.py --json results\official.json --csv results\official.csv
```

**数据来源**

| 库 | 表 | 内容 |
| --- | --- | --- |
| `practice-statistics-queue.sqlite3` | `practice_statistics_tasks` | 演练成绩（含 `jammer_count` 真值） |
| `formal-statistics-queue.sqlite3` | `statistics_tasks` | 正式成绩（**不含**真值） |
| `upload-queue.sqlite3` | `upload_tasks` | 行为日志/统计包上报状态（应为空） |

**关键行为**

- 用 `mode=ro`（必要时 `immutable=1`）只读打开，**不会干扰正在运行的模拟器**；
- 把原始字段换算成可直接填表 1 的量：`清除比例 = cleared/jammer_count`、`平均定位清除时间 = virtual_time_us/1e6/cleared`、`程序运行时间 = program_run_duration_ms`；
- 同时显示 `entered` / `end_reason` / `state` / `attempt_count` / `last_error_code`，用于判断"这局是否有效、数据是否真的上报成功"。

---

## 7. 依赖与调用关系

```
run_practice.py ──┬─> client.ApiClient ──HTTP──> 真实模拟器 (127.0.0.1:2026)
                  └─> agent.Q3Agent ──> geometry_solver (根目录)
                                            ├ localization_polygon
                                            └ minimum_enclosing_circle

code/run_mock_q3.py ──> agent.Q3Agent ──> client.ApiClient ──> mock_server.MockServer
tests/test_robot.py ──> 上述全部
official_stats.py ──(只读)──> JammersSimulatorData/*.sqlite3
```

## 8. 状态一览

| 文件 | 状态 |
| --- | --- |
| `client.py` | 可用；已按真实模拟器校准（含连接失败/业务拒绝区分） |
| `agent.py` | 问题 3 可用，真机实测一局 15/15 全清；问题 4 主循环未接入 |
| `mock_server.py` | 可用；已知不保真点见 §4 |
| `run_practice.py` | 可用；已修复重复 `/enter` 缺陷 |
| `official_stats.py` | 可用 |
| `__init__.py` | 可用（未导出新增的 `run_practice` / `official_stats`，它们是入口脚本而非库） |
