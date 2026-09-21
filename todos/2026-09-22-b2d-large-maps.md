# Large Map 的闭环成本，以及 220 条路线实测

状态: **已完成**（2026-09-22 夜，01:17–04:24）。
目标文档: [docs/bench2drive-cost.md](../docs/bench2drive-cost.md)。
背景: [research/carla-efficiency.md](../research/carla-efficiency.md)。

## 为什么有这一轮

昨天的成本模型有两个洞，两个都只在小地图上量过：

1. **一个 tick 在 Large Map 上要多少钱。** `docs/bench2drive-cost.md` 里那个 71 ms/tick
   的优化配置（800×450、每 4 tick 渲染、Qwen3-VL-4B 真的在环）测的是 base package 的路线，
   全是 Town01–10HD。Town12 的 route 1711 跑出来是 288.2 s / 1283 tick，约 225 ms/tick，
   但那是**未优化**配置，两个数不可比。
2. **Large Map 上能并行开几个实例。** 4 个饱和、8 个优化后能跑，都是在 Town10HD 上量的。
   Large Map 要流式加载 tile、持有的 actor 多得多，天花板可能更低。

**220 条路线里 151 条是 Large Map**（Town12 104、Town13 47、Town11 和 Town15 各 7），
所以这两个洞直接决定"一轮评测到底几小时"。

用户后来加的范围：**不要再外推，直接把 220 条跑完，报实测 wall clock。**
于是顺序是：先定 Town12 上的并发上限（它决定用几个 worker），再用那个 worker 数跑满 220 条。
Task 1 的 per-town 成本从全量跑里免费掉出来——每条路线都有自己的 tick profile，
每条路线的 town 都知道——所以不另外测一遍。

## 方法

**只用 `scripts/b2d_run.py` 和未经修改的 Bench2Drive leaderboard。**
昨晚最贵的一个错误就是量了一个手写的 bench 脚本，然后把它自己的 bug 记到 CARLA 头上。
插桩和优化仍然是 `scripts/b2d_hooks.py` 里带开关的 monkeypatch，未优化路径保持可复现。

优化配置（和 `docs/bench2drive-cost.md` 里那 71.1 ms/tick 的一行逐字相同）：

```
--rig front3 --width 800 --height 450 --policy gpu --policy-socket /tmp/b2d-policy.sock \
--decimate 4 --overlap --no-spectator --zero-copy
```

policy server 一个进程供所有 worker：
`b2d_policy_server.py --backbone qwen --width 800 --n-images 3 --layers 18`，
每帧 1050 个 image token。

新增的只有 `scripts/b2d_report.py`：只读，把每次 attempt 的记录和路线 XML 的 town 拼起来，
按 town 报成本、可靠性（R6）和 server age 失败曲线（R7）。它不启动任何东西。

端口：`--server-index 40`，即 RPC 4000+50i、TM 10000+50i。
避开别的 session 正在用的 2000/8000/9000 段，
也顺手满足了 Bench2Drive README 那句「低于 10000 的端口不安全」。

## 负载条件（必须和数字一起报）

- **GPU 是共享的。** Waymo 特征抽取每 ~15 分钟跑一次 ~3.4 分钟的 burst，占空比约 23%，
  还要再持续大约一天。不能杀、不能停，所以它是本轮所有数字的背景负载的一部分。
  policy server 常驻约 9.6 GB 显存。
- **网络被一个下载占满**，到 23 号中午前后。和本轮无关，但一起记着。
- 所有数字都是**共享卡**口径，不能和 `docs/carla.md` 里那张 idle 卡的表合在一起看。

## 开跑之前写下的预测（CLAUDE.md 要求，且事后写的预测不值钱）

从已有的两组数拆出来的推理：

| 量 | Town10HD | Town12 | 比值 |
|---|--:|--:|--:|
| 未优化配置，非 sensor 的 per-tick 工作（`world_tick` + `tree`） | ~33 ms | 70.4 ms | 2.1× |
| 其中 `world_tick` | 7–9 ms | 33.8 ms | ~4× |
| 其中 scenario tree | 24–26 ms | 35.7 ms | 1.4× |
| 未优化配置的阻塞等相机 | 125–135 ms | 108.2 ms | 0.85× |
| 未优化配置的总 tick | 158–170 ms | 178.6 ms | ~1.1× |

**关键推理**：未优化配置下 Town12 只贵 1.1×，是因为四分之三的 tick 是在阻塞等相机帧，
而等相机的时间两张图一样——Large Map 多出来的 `world_tick` 和 tree 大部分藏在那个等待后面。
**优化配置把那个等待删掉了**（sensor wait 0.03 ms），于是藏在后面的东西全部露出来。
Town10HD 优化后 71.1 ms = 33 ms 非 sensor 工作 + 约 38 ms 落回 `world_tick` 的渲染残余；
Town12 的渲染残余应该一样（同样 3 相机、同样 800×450、同样 decimate 4），
所以预测 **Town12 优化后约 108 ms/tick，即小地图的 1.5×**。

220 条的预测（8 个 worker，若 Large Map 的并发上限确实还是 8）：

- 小地图 8 实例实测 54.5 ms/tick，Large Map 按 1.5× 取 81.8 ms/tick
- 每条路线 2615 tick（小地图实测均值，Town12 只有 route 1711 一个点是 1283，先不用）
- 每条路线 tick 之外的开销：Large Map 约 60 s（route 1711 的 288.2 s 减去 tick 时间），小地图约 25 s
- `151 × (2615×0.0818 + 60) + 69 × (2615×0.0545 + 25) = 52917 s`，除以 8 → **1.84 h**

加上重启、server 启动错峰和 Waymo 的 burst，**预测 2–3 小时**，
即文档里现在那个「约 1.1 小时」乐观了 2–3 倍。
如果 Large Map 的并发上限掉到 4，同样的算术给 **3.2 h**。

**预测区间：1.5–3.5 h，中心 2.5 h。** 超过 3 h 的部分就是 CLAUDE.md 说要先 profile 的那段，
而这一轮的 profiling pass 就是下面的并发梯子本身。

## 结果 1：Town12 上的并发梯子（2026-09-22 00:28–01:15）

同样 24 条 Town12 路线（在 104 条里均匀取），`--max-ticks 800`，优化配置，共享卡。

| 实例数 | 24 条的 wall | ms/tick（每实例） | 理想占用下的总 tick/s | 端到端 tick/s | VRAM 中位 | load 中位 | 真失败 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 4 | 898 s | 84.5 | 25.3 | 21.4 | 42.1 GB | 17.6 | 0 |
| 6 | 635 s | 84.1 | 37.7 | 30.2 | 54.4 GB | 23.5 | 0 |
| 8 | 606 s | 102.5 | 43.5 | 31.7 | 72.0 GB | 23.0 | 0（1 次是端口复用，见下） |
| 10 | 572 s | 101.6 | 53.4 | 33.6 | **83.5 GB（峰值 87.4）** | **27.7（峰值 38.9）** | 1 |

「理想占用」= N × 800 / 每条路线的 wall，即所有 worker 都在跑路线时的吞吐；
「端到端」= 24×800 / 整步 wall，被 20 s×N 的错峰启动稀释，24 条路线摊不开。
**220 条的跑批该看前者**，因为启动只付一次。

三条结论：

1. **膝盖在 4→6 之间。** 4→6 per-instance 成本完全不变（84.5 → 84.1），吞吐 1.49×，
   等于白拿。6→8 per-instance 涨 22%，吞吐只多 15%；8→10 per-instance 不变但 VRAM 到顶。
2. **这一次是显存先到顶，第一次。** 10 个实例时 VRAM 中位 83.5 GB、峰值 87.4 GB（96 GB 的卡），
   每个 Town12 实例约 6.3 GB。12 个就装不下。
   `research/carla-efficiency.md` 里「这张卡在闭环上过剩」这句话**只对小地图成立**；
   Large Map 上它是真的会被用满的。同时 load 到 27.7/25，CPU 也超订了。
3. **选 8，不选 10。** 10 比 8 快约 23%，但把 VRAM 吃到 87/96 GB，
   而同一张卡上还有 Waymo 特征抽取在按 burst 要显存——为了 20% 的吞吐去赌一天的活不值得。

### 那个 segfault 不是并发上限，是端口复用

N=8 和 N=10 各有一次 `server_died_rc139`（139 = 128+11，SIGSEGV）。
两次都是 server index 45，server 日志里写得很清楚：

```
LowLevelFatalError [File:Unknown] [Line: 136]
Exception thrown: bind: Address already in use
Signal 11 caught.
CommonUnixCrashHandler: Signal=11
```

**CARLA 在启动时端口被占就会 segfault**，不是优雅报错。
梯子是一步接一步跑的、复用同一个端口段，上一步的 server 刚停、端口还在 TIME_WAIT。

这直接影响 `docs/carla.md` 里那条「第 6 个 server 在饱和负载下 segfault」：
**那很可能也是端口复用，不是 GPU 饱和。** 启动时的 segfault 在排除掉端口之前不能当成并发上限。

### 顺手修掉的一个我们自己的 bug

梯子结束后有一个 CARLA server 活过了启动它的 runner，占着端口和 6 GB 显存半小时。
原因：`CarlaUE4.sh` 是个 wrapper，真正的二进制是它的子进程；二进制 segfault 时 wrapper 先退出，
`Server.stop()` 里的 `os.getpgid(wrapper_pid)` 抛 ESRCH，循环 break，二进制就活下来了。
修法是在启动时把 pgid 记下来（`setsid` 让 wrapper 就是 group leader），停的时候按记下来的 pgid 杀。
`Runner.kill()` 有同一个问题，一起修了。**下一步那个泄漏的 server 正是它自己制造的 bind error 的来源。**

## 结果 2：220 条全量实测

**11196 s = 3.11 小时**，01:17:23 → 04:24:09，8 个 worker，一条命令无人值守，exit 0。
209/220 条完成，245 次 attempt，25 次重启，633881 个 tick，总吞吐 56.6 tick/s，worker 占用率 0.97。
原始数据：`research/results/b2d/full220-results.csv`（每次 attempt 一行）和 `full220-summary.json`。

**对预测**：事前写下的区间是 1.5–3.5 h、中心 2.5 h。实测 3.11 h，落在区间内、偏上。
per-tick 的预测 Town12 1.5×，实测 1.65×，方向和机制都对，量偏乐观。
文档里原来那个「约 1.1 小时」**乐观了 2.8 倍**。

### 每个 town 的成本

| town | 路线 | ms/tick | `world_tick` | tree | tick/路线 | wall/路线 | 占整轮 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| **Town13** | 47 | **156.8** | 94.5 | **55.9** | 2845 | 584.7 s | 33% |
| **Town12** | 104 | **106.2** | 73.9 | 29.2 | 3141 | 408.6 s | 50% |
| Town15 | 7 | 64.5 | 55.8 | 7.4 | 3635 | 258.5 s | 2% |
| Town11 | 7 | 56.3 | 33.9 | 20.9 | 3754 | 251.4 s | 2% |
| Town05 | 9 | 74.1 | 58.6 | 14.0 | 3742 | 293.6 s | 3% |
| Town10HD | 4 | 78.4 | 57.9 | 19.1 | 2681 | 233.0 s | 1% |
| Town06 | 6 | 65.5 | 52.4 | 11.7 | 3131 | 208.4 s | 1% |
| Town03 | 11 | 63.8 | 53.6 | 9.0 | 3022 | 214.0 s | 3% |
| Town07 | 5 | 62.6 | 52.3 | 9.3 | 1984 | 139.5 s | 1% |
| Town04 | 12 | 60.3 | 48.6 | 10.4 | 2892 | 184.6 s | 3% |
| Town01 | 4 | 55.8 | 44.4 | 10.5 | 1327 | 91.3 s | 0.4% |
| Town02 | 4 | 50.6 | 43.7 | 5.9 | 2042 | 129.1 s | 0.6% |

**这一轮最该记的一条：「Large Map」是错的分组。**
Town11 和 Town15 一样流式加载 tile，却是 0.94×，和小地图没区别；
贵的只有 Town12（1.65×）和 Town13（2.44×）这两张具体的图，它们占 69% 的路线、**83% 的 wall clock**。
「Large Map 贵 1.80×」这个合并数字只在「151 条恰好落在那两张贵图上」的意义上成立。
以后写图的名字。

**Town13 比 Town12 贵的那一截在 scenario tree 上**（55.9 对 29.2 ms，占一个 tick 的 36%），
不在 `world_tick` 上。那就是 `RouteLightsBehavior._turn_close_lights_on` 每 tick 把全图路灯
重取一遍——Town13 路灯最多。**所以 `--cache-lights` 要重测**：它当初在小地图单实例上被否掉
（省下的 Python 原样回到阻塞等待里），但现在它是 Town13 一个 tick 的三分之一，
而且 8 worker 下 CPU 真的超订（load 中位 31.8、p90 47.0 对 25 核）。

### 可靠性（R6）：11 条路线跑不完，全在 Town12/Town13

`3048, 11715, 11755, 23687, 23708`（Town12）、`3785, 3800, 23670, 23695, 24041, 24071`（Town13）。

- 每条三次 attempt，每次都在**全新 server、不同端口段**上，
  每次都在**成功跑了 150–360 s 之后** `Signal=11 / CommonUnixCrashHandler` 崩掉，
  日志里**没有** bind error。同一批 server 上前后跑的别的路线都好好的。
- 36 次失败 attempt **全部**是 `server_died_rc139`，只有这一种失败模式。
- 按 town：Town12 5/104、Town13 6/47、**其余 69 条一条没失败**（Town01–10HD 只有 2 次无关紧要的重启）。
- 代价：24.1 worker-hours 里的 2.17 h（9.0%）。

**这是 sensor-dormancy segfault（Bench2Drive #235 / carla #7772）第一次在未经修改的官方
leaderboard 路径上复现出来。** 之前一直够不着它，因为地图没装。

**论文口径**：这台机器上的 Bench2Drive 分数最多是 **209/220** 条上的分数，
而哪 209 条是 CARLA 决定的、不是 policy 决定的。必须和分数一起报出来。

watchdog 在这里赚回了自己：server 死了之后 route 进程会挂在一个 300 s 超时的 RPC 上，
盯 server 进程把每次崩溃从五分钟变成一个 5 s 的轮询。

### R7：245 次 attempt，不支持定期回收

| server age（已跑路线数） | attempts | failed | 失败率 |
|--:|--:|--:|--:|
| 0 | 44 | 23 | 52% |
| 1 | 21 | 3 | 14% |
| 2–9 | 127 | 2 | 1.6% |
| 10–22 | 53 | 8 | 15% |

**age 0 那 52% 是选择效应，不是「新 server 不可靠」**：一条路线崩了就换 server，
所以崩溃路线的每次重试都落在 age 0；11 条注定失败的路线贡献了 36 次失败里的 33 次。
该读的是 age 2–9：127 次 attempt、2 次失败、**平的**。
失败是**十一条具体路线的属性，不是 server 跑久了的属性**。
`--recycle-routes` 继续关着——这次是有证据地关着。
要推翻它需要：把这 11 条已知坏路线排除之后仍然看到失败率随 age 上升——而这一轮做不到，
因为排除之后几乎不剩失败。

### 负载条件（数字的一部分）

| | |
|---|---|
| GPU 利用率 | 中位 100%，10 分位 65% |
| VRAM | 中位 67.9 GB、p90 74.8、峰值 78.8（共 96） |
| load average | 中位 31.8、p90 47.0、峰值 72.1（对 25 核） |
| 同卡的 Waymo 特征抽取 | ~3.4 min burst / ~15 min，占空比约 23% |
| 同机的 Waymo 下载 | 32 streams，占满网络也吃 CPU |
| policy server | Qwen3-VL-4B 常驻约 9.6 GB 显存，8 个 worker 共用 |

**独占这台机器会明显更快，但我们没测快多少。**

### 3.11 小时里最大的那块水分

查 leaderboard 自己的 records（220 条的 `results.json`）：
**118 条 `Failed - TickRuntime`（撞 4000 tick 上限）、90 条 `Failed - Agent got blocked`、
1 条 route deviation，成功 0 条。** route completion 均值 **10.9%**，最好的一条 20.1%。
**没有任何一条路线是因为车开到了终点而结束的。**
路线平均约 105 m，6 m/s 开完约 330 tick，而实测中位数正好是 4000。

**换一个真能开完路线的 policy，省下来的时间会超过这份文档里任何一项工程优化的总和。**
按 600–1200 tick/路线算，同样 8 个 worker 是 **1.0–1.5 小时**；到那时每条路线约 70 s 的固定开销
（world load + scenario build + teardown，按 town mix 加权）就占到 30–50%，成为下一个该打的目标。
3.11 小时该读成「一个永远开不到终点的司机的上界」，不是仿真器的地板。

其中有一部分是我们自己的：`AutonomousAgent.set_global_plan` 给 agent 的是
`downsample_route(…, 50)`，所以 `_steer_to_route` 是照着一条稀疏路线在打方向，转不过弯。

下一步的控制器不用自己写：CARLA 自带
`$CARLA_ROOT/PythonAPI/carla/agents/navigation/controller.py` 里的 `VehiclePIDController`
（纵向 + 横向两个 PID），外面还有 `local_planner.py` / `basic_agent.py`。
它在 tarball 的 `PythonAPI/carla` 里，不在我们 import 的 PyPI wheel 里，加一条 `sys.path` 就能用。
轨迹表示怎么映射到 CARLA 的控制序列，另开了一份研究笔记。

## 落到哪里了

- `docs/bench2drive-cost.md`：全量实测、per-town 成本、并发梯子、R6/R7，以及被这一轮推翻的三条旧结论。
- `docs/carla.md`：sensor-dormancy segfault 在官方路径上可达；启动 segfault 是端口嫌疑；显存只在小地图上过剩。
- `research/carla-efficiency.md`、`research/decisions.md` 第 17 条：就地修正，写清原来说的是什么、为什么改。
- `research/results/b2d/full220-results.csv`：每次 attempt 一行，245 行。
- 代码：`scripts/b2d_report.py`（新，只读），`scripts/b2d_run.py` 的 pgid 修复。
