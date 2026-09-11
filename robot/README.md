# robot 模块

问题三的机器狗自动定位与清除程序。

## 文件

| 文件 | 职责 |
| --- | --- |
| `main.py` | 程序入口。读取队号与接口地址，等待接口开放后进入目标区域，执行一轮完整的定位与清除，结束时主动退出 |
| `client.py` | 模拟器通信。封装 `/enter`、`/measure`、`/clear`、`/exit` 四个动作，负责串行发送、按同一 `request_id` 重试、守住现实时间预算 |
| `agent.py` | 定位清除策略。扫描发现、交会定位、清除判据、频道状态机与终止条件 |
| `__init__.py` | 对外暴露 `ApiClient`、`ApiError`、`Q3Agent`、`MissionResult` |

几何计算（半平面交、凸包与旋转卡壳、最小覆盖圆）在仓库根目录的 `geometry_solver.py`。

## 运行

```powershell
cd D:\2026MCM
D:\Python3.13.12\python.exe -X utf8 robot\main.py --team <参赛队号>
```

队号也可以用环境变量 `CUMCM_TEAM_NO` 提供。其余参数：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--base-url` | `http://127.0.0.1:2026` | 模拟器接口地址 |
| `--wait-s` | 300 | 等待接口开放的最长秒数 |
| `--max-obs` | 8 | 单个频道的最大观测次数 |
| `--err-deg` | 1.01 | 测向误差安全余量 |

程序在开始时会轮询等待接口开放，因此可以先启动程序再去模拟器点击开始测试；倒计时期间的连接失败属正常现象。

## 输出

结束时打印一行 JSON，包含 `cleared`（清除个数）、`absent_certified`（确认不存在的频道数）、`measures`、`clear_calls`、`virtual_time_s`、`mean_virtual_per_cleared_s`、`program_s`、`complete`。

`mean_virtual_per_cleared_s` 即题目要求的平均定位清除时间；只有 `complete` 为真且没有漏源时，它才等于官方统计值。

## 流程

1. 等待接口开放并调用 `/enter`，取 `remaining_real_duration_s` 作为本局现实时间预算。
2. 在 7 个扫描点依次检测 20 个频道完成发现。扫描点取原点与半径 1200 m 的正六边形顶点，对目标圆域的最坏最近点距离为 968.9016 m。
3. 对每个已发现频道追加异地观测，用全部观测的测向锥半平面交求定位区域，取最小覆盖圆。
4. 最小覆盖圆半径不超过 20 m 时在该点执行 `/clear`；返回"距离过近"时直接清除。
5. 全部频道都取得"已清除"或"确认不存在"的结论后调用 `/exit`。
