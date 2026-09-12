# robot 模块

问题三和问题四的机器狗自动定位与清除程序。

## 文件

| 文件 | 职责 |
| --- | --- |
| `main.py` | 程序入口。读取队号与接口地址，等待接口开放后进入目标区域，执行一轮完整的定位与清除，结束时主动退出 |
| `client.py` | 模拟器通信。封装 `/enter`、`/measure`、`/clear`、`/exit` 四个动作，负责串行发送、按同一 `request_id` 重试、守住现实时间预算 |
| `q3_agent.py` | Q3 七点发现证书、批处理调度、交会定位与频道状态机 |
| `q4_agent.py` | Q4 的 31 点定向发现证书与测向扇形覆盖式清除 |
| `mock_server.py` | 本地复现物理时钟、幂等重试、固定测向误差和定向半平面 |
| `__init__.py` | 对外暴露通信客户端与 Q3/Q4 agent |
| `main_fast.py` | **提交版本入口**。与 `main.py` 同协议，另加动作级脱敏日志与分支／时间账汇总 |
| `q3_agent_fast.py` | 提交版本的 Q3：基础版全部证书 + 联合选路、站内顺路收获、带误差上界的局部测站 |
| `q4_agent_fast.py` | 提交版本的 Q4：22 点连续域认证集、全局空清除预算、最后单元覆盖证明 |
| `geometry_solver_fast.py` | 提交版本共用的几何工具：路径优化、阴性覆盖推理、测站包与误差上界 |

几何计算（半平面交、凸包与旋转卡壳、最小覆盖圆）在仓库根目录的 `geometry_solver.py`。

## 运行

```powershell
cd D:\2026MCM
D:\Python3.13.12\python.exe -X utf8 robot\main.py --problem 3 --team <参赛队号>
D:\Python3.13.12\python.exe -X utf8 robot\main.py --problem 4 --team <参赛队号>
```

提交版本（提速版）的命令：

```powershell
cd D:\2026MCM
D:\Python3.13.12\python.exe -X utf8 -m robot.main_fast --problem 3 --team <参赛队号> --log-dir D:\2026MCM\fast_logs_practice
D:\Python3.13.12\python.exe -X utf8 -m robot.main_fast --problem 4 --team <参赛队号> --log-dir D:\2026MCM\fast_logs_practice --q4-scan compact --empty-limit 5
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

## 提速版与基础版的关系

`main_fast.py` 把此前多级提速版本合并为单文件，仓库只保留两条可运行入口：`main.py`（未提速的基础版本，Q3/Q4 共用同一套证书与状态机）与 `main_fast.py`（提交版本）。合并后用 `fast_logs_practice/` 里的 30 局官方演练日志做了逐动作回放：30 局的动作序列、位置、频道与虚拟时间全部与日志一致，说明合并没有改变任何决策。

## 流程

1. 等待接口开放并调用 `/enter`，取 `remaining_real_duration_s` 作为本局现实时间预算。
2. Q3 用 7 点覆盖证书，Q4 用 31 点三角格局部凸包证书扫描 unknown 频道；提交版本 `main_fast.py` 的 Q4 改为 22 点连续域认证集（认证不过自动回落到基础版的双环证书）。已发现频道先入队，避免立即往返。
3. 发现阶段后按下一安全观测点的距离滚动选频道，用测向锥半平面交与最小覆盖圆定位。
4. 最小覆盖圆半径不超过 20 m 时在该点执行 `/clear`；返回"距离过近"时直接清除。
5. Q4 的异地复测若因定向盲区失去信号，改用覆盖半径严格小于 20 m 的扇形三角格执行与朝向无关的 `/clear`。
6. 全部频道都取得"已清除"或"确认不存在"的结论后调用 `/exit`。
