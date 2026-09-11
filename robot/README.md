# robot 模块

问题三和问题四的机器狗自动定位与清除程序。

## 文件

| 文件 | 职责 |
| --- | --- |
| `main.py` | 程序入口。读取队号与接口地址，等待接口开放后进入目标区域，执行一轮完整的定位与清除，结束时主动退出 |
| `client.py` | 模拟器通信。封装 `/enter`、`/measure`、`/clear`、`/exit` 四个动作，负责串行发送、按同一 `request_id` 重试、守住现实时间预算 |
| `agent.py` | Q3 七点发现证书、批处理调度、交会定位与频道状态机 |
| `q4_agent.py` | Q4 的 31 点定向发现证书与测向扇形覆盖式清除 |
| `mock_server.py` | 本地复现物理时钟、幂等重试、固定测向误差和定向半平面 |
| `__init__.py` | 对外暴露通信客户端与 Q3/Q4 agent |

几何计算（半平面交、凸包与旋转卡壳、最小覆盖圆）在仓库根目录的 `geometry_solver.py`。

## 运行

```powershell
cd D:\2026MCM
D:\Python3.13.12\python.exe -X utf8 robot\main.py --problem 3 --team <参赛队号>
D:\Python3.13.12\python.exe -X utf8 robot\main.py --problem 4 --team <参赛队号>
```

队号也可以用环境变量 `CUMCM_TEAM_NO` 提供。其余参数：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--base-url` | `http://127.0.0.1:2026` | 模拟器接口地址 |
| `--problem` | 3 | 选择问题三或问题四 |
| `--schedule` | `batch` | 默认先发现后批量定位；`immediate` 仅用于基线对照 |
| `--wait-s` | 300 | 等待接口开放的最长秒数 |
| `--max-obs` | 8 | 单个频道的最大观测次数 |
| `--err-deg` | 1.01 | 测向误差安全余量 |

程序在开始时会轮询等待接口开放，因此可以先启动程序再去模拟器点击开始测试。倒计时期间的连接失败属正常现象。日志写盘前会把身份字段替换为 `<redacted>`。

## 输出

结束时打印一行 JSON，包含 `cleared`（清除个数）、`absent_certified`（确认不存在的频道数）、`measures`、`clear_calls`、`virtual_time_s`、`mean_virtual_per_cleared_s`、`program_s`、`complete`。

`mean_virtual_per_cleared_s` 即题目要求的平均定位清除时间；只有 `complete` 为真且没有漏源时，它才等于官方统计值。

## 流程

1. 等待接口开放并调用 `/enter`，取 `remaining_real_duration_s` 作为本局现实时间预算。
2. Q3 用 7 点覆盖证书，Q4 用 31 点三角格局部凸包证书扫描 unknown 频道。已发现频道先入队，避免立即往返。
3. 发现阶段后按下一安全观测点的距离滚动选频道，用测向锥半平面交与最小覆盖圆定位。
4. 最小覆盖圆半径不超过 20 m 时在该点执行 `/clear`；返回"距离过近"时直接清除。
5. Q4 的异地复测若因定向盲区失去信号，改用覆盖半径严格小于 20 m 的扇形三角格执行与朝向无关的 `/clear`。
6. 全部频道都取得"已清除"或"确认不存在"的结论后调用 `/exit`。
