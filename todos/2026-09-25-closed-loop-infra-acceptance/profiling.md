# CARLA harness profiling：每个 server 的成本、扩展曲线、瓶颈与布局

状态: done（2026-09-25 15:25–17:50 CST）
上级: [闭环基础设施验收](../2026-09-25-closed-loop-infra-acceptance.md)；英文定稿在
[docs/bench2drive-cost.md](../../docs/bench2drive-cost.md) 的 "Harness cost and layout on the five-GPU box (2026-09-25)" 一节
结果文件: [research/results/infra-acceptance/profiling/](../../research/results/infra-acceptance/profiling/)（`rungs.csv` 每档一行，`ab.csv` 并行 A/B）

## 结论

1. **每张卡的瓶颈是 GPU 渲染，不是 CPU。** Town12 上一张 RTX PRO 6000 从 1 个 CARLA server 加到 6 个，aggregate tick rate
   （所有 worker 每秒合计推进的 simulation tick 数，20 tick = 1 秒模拟时间）从 7.9 涨到 34.8，GPU utilization 到 89%；
   8、10、12 个只多出不到 10%，每个 worker 却慢 1.3–1.8 倍。这期间我们的进程最多用 13 核，cgroup 没有被 throttle。
2. **CPU 的账按 tick 算：Town12 每 tick 约 0.33 core-seconds，Town03 约 0.18**，不随 server 数变化。
3. **建议布局：每张卡 6 个 server，每个 worker 预算 2.5 核，先铺满卡再往一张卡上堆；整台 box 30 个 worker、约 75 核，
   且必须带 `--client-threads 8`。** 否则先撞上的是 container 的线程上限（pids.max 20480），不是 GPU。
4. **thread cap 是整台 box 上 worker 数的硬上限。** 默认每个 worker 约 650 个线程（server ~430 + route client ~215），
   route client 那 208 个线程来自 `carla.Client()` 默认按 host 的 208 个硬件线程开 worker；`--client-threads 8`
   把它降到 16 个，RSS 从 3.0 GB 降到 1.2 GB，CPU 和 tick 速度不变。
5. **`--cache-lights` 原实现不等价（夜间留着 ~1600 盏远处路灯），已修；修好后在 Town13 争用条件下的 A/B 里没有可测的收益**，
   保持默认关闭。把 off-screen viewport 缩到 64×64（`--server-args`）也没有可测收益。
6. **可靠性：今天 91 次 server 启动里 14 次在 route setup 阶段死于 `GameThread timed out waiting for RenderThread`
   + Signal 11**，与 thread cap 无关（`pids.events` 在崩溃时没有增加），与 box 上同时存在的 CARLA server 数（21–29 个）同时出现，
   原因未定；另外我们的端口落在 kernel 的 ephemeral port range 里。

**污染窗口。** 17:00 CST 之后 box 被其他 CARLA 任务占满（P5 v1 generation 在 GPU 4、GPU 2 上 11–12 个 server，控制器验收在
GPU 0 上 6 个，box 贴着 125 核上限），16:53 起我的 GPU 4 ladder 与 P5 v1 同卡。**所以 17:00 之后只用并行 A/B（两组同时跑、
背景相同）比较开关，不再报绝对扩展曲线**；GPU 4 上 16:45 之后的 ladder 作废（留在 `$DATA_DIR/runs/infra-acceptance/scale-void/`）。
Town12 ladder（15:27–16:08）和 Town03 ladder（16:18–16:38）在这之前，但也有别人的负载，每档的背景核数都记在表里。

## 方法

**工作负载。** 两个 rig（相机配置）：
(a) Alpamayo 考试 rig，走真正的考试 agent `scripts/b2d_zeroshot_agent.py`，配置 `"replay": "route"`（没有模型、
没有 socket，plan 来自 route oracle，即沿 route 中心线、按 `cruise_mps` 给出的参考轨迹，节奏和考试一样 2 Hz），
4 个相机（1368×784、1384×784、1392×792、608×352）10 Hz，`--decimate 2`（与 `scripts/zeroshot_b2d_alp.sh` 相同，
它让 `sensor_tick` 真正生效），`--no-spectator`，控制器 fixed；
(b) openpilot+TCP partner rig，用成本 stub `scripts/b2d_agent.py --rig op2tcp3`：openpilot 的 road/wide 两个
1928×1208 加 TCP 的三个 1600×900，五个相机每个 tick 都渲染，不跑任何网络，驾驶用 stub 的简单 route 跟随。
两者都经过生产路径：`b2d_run.Server`（`-RenderOffScreen -quality-level=Epic -graphicsadapter=<gpu>`）→
`b2d_run.Runner` → `b2d_route.py`（leaderboard + scenario_runner + traffic manager 都在这个 route 进程里）。

**固定工作量。** 一档（rung）= 同一张卡上 N 个 server，每个 worker 跑**同一条** route、同一个 TM seed、
`--max-ticks 1200`。同一 route 让每个 worker 的工作完全相同，per-worker 成本在不同 N 之间可比。
Town12 用 route 1773（夜间，obstacle scenario 把 ego 挡住，8 m/s 也能跑满 1200 tick）；Town03 用 25378、
Town13 用 3561，这两条 route oracle 以 8 m/s 三四百 tick 就开完了，所以 cruise 降到 1.5 m/s，保证开到 tick 上限
（车速几乎不影响每 tick 的渲染成本）。server 在各档之间复用，按 15 s 间隔错开启动，多余的 server 在该档开始前停掉
（空闲 server 在 async 模式下会自己空转）。

**采样。** `scripts/b2d_scale.py` 每 2 s 记一次：整个 container 的 cgroup `cpu.stat`（usage、throttled）、load
average、我们启动的每个进程（CarlaUE4 server vs route client）的 CPU 时间 / RSS / 线程数（读 `/proc/<pid>/stat`）、
每张卡的显存和 utilization（`nvidia-smi`）、每个 worker 的 heartbeat tick 数。**steady window** = 所有 worker 都过了
40 tick warm-up、且还没有一个写出结果的那段时间；aggregate ticks/s、各进程核数、GPU 占用都只在这段时间里算。
per-worker ms/tick = N / aggregate ticks/s。"背景核数" = container 总用量减去我们的进程。某档里有 server 在 setup
阶段崩了，就用 `b2d_scale.py --recompute` 只在活下来的 worker 上重算（表里注明实际 worker 数）。
`$DATA_DIR/runs/infra-acceptance/pids.log` 每 5 s 记一次 container 线程数。

**CPU 预算。** 我们所有进程用 `taskset` 钉在 45 个 CPU 上（GPU 4 的 ladder 用 52–96，即 GPU 4 所在 NUMA node 1；
GPU 0 上的 A/B 两组各用 0–21 和 22–44）。

**并行 A/B。** `scripts/infra_ab.sh`：同一张卡上同时起两个 `b2d_scale.py`，各 3 个 server，同一条 route，A 组默认、B 组加开关。
两组看到的是同一个背景，所以 17:00 之后 box 被占满时仍能比较开关，但只能比较、不能当绝对值用。

**等价性检查。** CARLA 在这台机器上不能逐位复现：同一配置跑两次，第一帧相机图像的 md5 就不同，400 tick 后 ego
位置差 1.4–3.4 m。所以"行为一致"按"落在两次相同 baseline 之间的差异范围内"判断（`scripts/infra_verify.sh`：
每帧相机图像的 md5 和 16×9 灰度缩略图、每 tick 的真值位姿和控制量），对 `--cache-lights` 这种有明确语义的改动，
再直接检查语义（`$B2D_LIGHTS_CHECK`：每 20 次更新从 server 读回全部路灯，对照"ego 半径内亮、半径外灭"的规则）。

## 每个 server 的成本（N = 1，Town12，Alpamayo rig，GPU 4）

| 项 | 数值 |
|---|---:|
| ms/tick（steady window） | 127 |
| ticks/s（real-time factor） | 7.9（0.39×） |
| CarlaUE4 server 进程 | 2.43 核，RSS 4.2 GB，374 线程（45 CPU affinity；不钉核时约 430） |
| route client 进程 | 0.69 核，RSS 3.0 GB，214 线程（`--client-threads 8`：16 线程，1.2 GB） |
| VRAM | 约 5.1 GB / server |
| GPU utilization | 20% |
| route setup（load_world + scenario 构建，到第一个 tick） | 56 s（N = 12 时 92 s；Town03 14–28 s） |
| tick 内分解（leaderboard 内计时） | world.tick 95 ms，scenario tree 20.5 ms，agent 9.0 ms |

N = 1 时 GPU 只用了 20%、CPU 只用了 3 核，一个 tick 却要 127 ms：单个 server 是延迟受限的（同步模式下每 tick 串行地做
物理、tile streaming、渲染、回读），所以多开 server 在资源用满之前几乎是白拿的。

## 一张卡上的扩展

![Aggregate tick rate and per-worker cost against servers on one GPU](../../research/figs/infra-scale-throughput.png)

左图是 aggregate ticks/s（虚线是从 1 个 server 线性外推），右图是每个 worker 的 ms/tick。Town12 在 6 个 server 处拐平；
Town03 到 8 个还在涨。

**Town12**（route 1773，GPU 4，CPU 52–96，每档 n = N 个相同 route、各 1200 tick，15:27–16:08 CST）

| server 数 | aggregate ticks/s | 相对 N=1 | per-worker ms/tick | GPU util | VRAM | 我们的核数 | server 核/个 | client 核/个 | 背景核数 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.9 | 1.00× | 127 | 20% | 6.3 GB | 3.1 | 2.43 | 0.69 | 23 |
| 2 | 14.6 | 1.86× | 137 | 38% | 11.0 GB | 5.9 | 2.28 | 0.66 | 27 |
| 4 | 27.4 | 3.49× | 146 | 68% | 21.4 GB | 11.2 | 2.18 | 0.63 | 19 |
| **6** | **34.8** | **4.42×** | 173 | 89% | 30.6 GB | 13.2 | 1.71 | 0.49 | 22 |
| 8 | 36.4 | 4.63× | 220 | 96% | 39.7 GB | 13.1 | 1.28 | 0.36 | 32 |
| 10 | 36.3 | 4.61× | 276 | 97% | 49.7 GB | 12.4 | 0.97 | 0.27 | 31 |
| 12 | 38.2 | 4.85× | 315 | 100% | 58.9 GB | 12.4 | 0.81 | 0.23 | 38 |

读法：1 到 4 个时每个 worker 只慢 15%，吞吐接近线性；4 到 6 还能拿到 27%；6 以后吞吐平了，多出来的 server 只是把每个
worker 拖慢（12 个时 315 ms/tick，是 N=1 的 2.5 倍）。每个进程的核数随 N 下降，是因为 worker 的 tick 变慢了，每 tick 的
CPU 成本本身不变（下图左）。

**Town03**（route 25378，1.5 m/s，GPU 4，16:18–16:38 CST，背景 55–88 核，即控制器验收占满 GPU 0 的时候）

| server 数（实际 worker） | aggregate ticks/s | per-worker ms/tick | GPU util | VRAM | 我们的核数 | server 核/个 | client 核/个 | 背景核数 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 26.7 | 75 | 22% | 12.5 GB | 5.6 | 2.33 | 0.46 | 65 |
| 4 | 46.1 | 87 | 43% | 21.1 GB | 9.9 | 2.07 | 0.40 | 56 |
| 6（5） | 51.3 | 97 | 97% | 33.9 GB | 9.9 | 1.67 | 0.32 | 74 |
| 12（6） | 57.0 | 105 | 96% | 38.9 GB | 10.2 | 1.43 | 0.27 | 73 |
| 8 | 72.7 | 110 | 98% | 47.9 GB | 13.1 | 1.38 | 0.26 | 88 |

N = 1 那档 server 在 setup 时崩了，没有数；"12（6）"那档 12 个 server 里 6 个 route 在 `carla.Client()` 处因 thread cap
失败，只剩 6 个在跑（见下文），按 6 个算。Town03 每 tick 只有 Town12 的一半多一点 CPU，tick 也快一倍，8 个时 72.7 ticks/s
仍在涨；GPU utilization 在 5 个以上就显示 97%，但吞吐还在涨，说明这个指标在这里已经饱和、不再是线性的容量度量。

![CPU per tick, cores used and GPU utilisation against servers on one GPU](../../research/figs/infra-scale-resources.png)

左：每推进一个 simulation tick 花多少 core-seconds（圆点 server，三角 route client），基本不随 N 变；中：我们的进程一共用多少核，
6 个以后不再增加；右：GPU utilization，Town12 在 6 个处到 89%。

## 布局建议

| 项 | 建议 | 依据 |
|---|---|---|
| 每张卡 server 数 | **6**（Town12/13 为主的考试）；只跑小 town 可到 8 以上 | Town12 在 6 处拐平；Town12/13 占 69% 的 route、83% 的墙钟 |
| 每个 worker 的 CPU 预算 | **2.5 核**（每卡约 15 核）；小 town 在 8 个/卡时约 2 核 | 实测 Town12 6 个时 2.2 核/worker、Town03 8 个时 1.6 核/worker；route setup 时更吃 CPU，留余量 |
| 跨卡 | 先铺满卡，再往一张卡上堆 | 单卡 6 个以后只换来更慢的 worker |
| route client 线程 | **`--client-threads 8`** | thread cap，见下 |
| 整台 box | **5 × 6 = 30 个 worker，约 75 核、约 150 GB VRAM、约 170 GB RSS** | GPU 是约束；默认线程数下 pids.max 只容得下约 25 个 |

机器可读的一行已追加到 `gpu-plan.md`：`servers_per_gpu=6 cores_per_server=2.5 threads_per_server=650 (450 with
--client-threads 8) max_servers_box=30`。

**跨卡没有直接测。** 原计划在两张卡上各放 N/2 个、和一张卡放 N 个对比，但一直没有空闲的第二张卡（GPU 1 被 reactivity 借走，
GPU 0 一直有控制器验收的 server）。卡与卡之间共享的是 CPU、内存带宽和 thread cap；按上面每 tick 的 CPU 成本，5 张卡
都在 knee 上时 CPU 约 60 核，离 125 核上限还远，所以预期能按卡线性叠加，但这是推测，要在一台空闲的 box 上用
`scripts/infra_scale.sh x2` 验证。

## container 的 thread cap

`/sys/fs/cgroup/pids.max` 是整个 container 的线程上限 **20480**。CARLA 很吃线程：server 的线程数随 CPU affinity 变
（不钉核约 430，45 个 CPU 时 374，8 个 CPU 时 261）；route client 约 215，其中 208 个来自 `carla.Client(host, port)`
默认 `worker_threads=0`，即按 host 的硬件线程数（208）开 worker，和这个进程能用几个核无关。16:30 前后 box 上三个任务一共
21 个 server，container 线程数到 17.9k，`pids.events` 记了 12 次撞到上限；Town03 12 个 server 那档里 6 个 route 在
`carla.Client()` 处报 `RuntimeError: Resource temporarily unavailable`（pthread_create 返回 EAGAIN）。

`--client-threads N`（b2d_run → b2d_route，默认 0 = 原行为）把 `worker_threads` 设成 N。并行 A/B（Town12 route 1773，GPU 0，
各 3 个 server，800 tick，17:18–17:24 CST，背景约 75 核）：

| 组 | route client 线程 | client 核/个 | client RSS | per-worker ms/tick |
|---|---:|---:|---:|---:|
| A 默认 | 216 | 0.22 | 3.0 GB | 263（n = 3） |
| B `--client-threads 8` | 16 | 0.24 | 1.2 GB | 279（n = 1，另外 2 个 server 在 setup 时死于 RenderThread 超时） |

B 组只剩一个 worker，ms/tick 的差（6%）在这种负载下是噪声量级；等价性见下文：同一时段的验证跑里，它和 baseline 在
400 tick 后的最大位姿差 0.06 m，而两次相同 baseline 之间是 0.41 m。

## 可靠性：测的时候看到的失败

- **`GameThread timed out waiting for RenderThread after 60.00 secs` 然后 Signal 11。** 今天我们 91 次 server 启动里有 14 次，
  控制器验收那边另有 2 次；9 月 23 日的 lateral-v2 也有 19 次。全部发生在 route setup 阶段（第一个 tick 之前），
  不是端口冲突（没有 `bind` 那一行）。**不是 thread cap**：17:18 那两次崩溃前后 `pids.current` 在 17.0–18.6k，
  `pids.events` 一直是 12 没有增加。也不只在拥挤的卡上：16:20 一次是在空闲的 GPU 4 上单独一个 server。
  它和"box 上同时有 20 多个 CARLA server"同时出现，原因未定；下一步是在空 box 上复现，并看 Vulkan 驱动日志。
  对跑考试的影响：b2d_run 的重试会把它吃掉（这一类崩溃发生在 setup，不会跑到一半丢结果），代价是每次约 60–90 s。
- **我们的端口在 kernel 的 ephemeral port range 里。** box 上 `ip_local_port_range` 是 32768–60999；server index i 的 RPC
  端口是 2000 + 50i，i ≥ 616 就在里面（schedule 分给我们的 700–799 全在里面），TM 端口 8000 + 50i 从 i ≥ 496 起就在里面。
  出站连接可能正占着这个端口，CARLA 在启动时报 `bind: Address already in use` 然后 Signal 11（index 780 撞上过一次）。
  `b2d_run.port_free` 只检查有没有人在 listen，查不出这种情况。`b2d_scale.py` 改成真 bind 一次；生产里建议 index 用 490 以下。

## 优化：做了什么、等价吗、值不值

| 改动 | 等价性 | 性能 | 结论 |
|---|---|---|---|
| `--client-threads 8`（route client 的 CARLA worker 线程） | 位姿差 0.06 m，同一时段两次 baseline 之间 0.41 m | CPU 不变，线程 216 → 16，RSS 3.0 → 1.2 GB | **采用**：是整台 box 能跑 30 个 worker 的前提 |
| `--cache-lights` 修复版 | 路灯规则 0 处不符（原实现也是 0；旧缓存每次 1633 处不符）；位姿差 1.7 m，缩略图差与 baseline 相同 | Town13 route 3561 并行 A/B：ms/tick 198（n = 3）vs 200（n = 2），tree 20 vs 22 ms | 等价了，但没有可测收益，**保持默认关闭** |
| off-screen viewport 64×64（`--server-args=-ResX=64 -ResY=64`） | 位姿差 1.7 m，缩略图差 0.16（前 10 帧）与 baseline 的 0.16–0.17 相同 | Town12 并行 A/B：266 vs 269 ms/tick（各 n = 2） | 无收益，**不采用** |

`--cache-lights` 旧实现的问题：它在第一次调用时读每盏灯的 `is_on` 建缓存，但夜间 server 会在 route 开始前后自己把路灯
打开，缓存以为是灭的，于是 Town12 上 2699 盏路灯里有 1633 盏远处的灯一直亮着（原实现每 tick 都会把半径外的灯关掉）。
这很可能也是它当初在单实例上"tree 快了、墙钟反而慢了"的一部分原因：server 多渲染了一千多盏灯。修法是第一次更新下发完整状态，
之后只发变化。修好后在这次的 Town13 route 上 tree 本身只有约 20 ms（full220 里 Town13 的 55.9 ms 是那个不会开车的 stand-in
在 4000 tick 里积累的 scenario 负担），缓存省不出可测的时间。

## 没做 / 待做

- openpilot+TCP rig（5 个相机每 tick 渲染）的 ladder 在 17:40 之后的满载 box 上跑，只作每个 worker 成本的参考，见 `rungs.csv`
  的 `t12-op-t8`。
- 跨卡线性叠加（上文）。
- RenderThread 超时崩溃的原因。
- py-spy 剖析 route 进程（`scripts/pyspy_python.sh`、`infra_scale.sh pyspy` 已备好）：route client 每 tick 只占 0.07–0.09
  core-seconds，不在瓶颈上，所以没有跑。
