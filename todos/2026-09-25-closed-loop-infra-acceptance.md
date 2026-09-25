# 闭环基础设施验收：CARLA harness 性能、控制器验收、不明 SIGKILL

状态: done（2026-09-25 15:10–19:10 CST）；验收结论：harness 布局已定；B2D 没有控制器通过（固定控制器 2 Hz 最接近）；HUGSIM 只有 fixed2 通过；SIGKILL 来源未能确证，取证已布好
主题: ../research/trajectory-to-control.md、[docs/carla.md](../docs/carla.md)、[docs/bench2drive-cost.md](../docs/bench2drive-cost.md)、[docs/hugsim.md](../docs/hugsim.md)
子文档: [profiling](2026-09-25-closed-loop-infra-acceptance/profiling.md)、[B2D 控制器验收](2026-09-25-closed-loop-infra-acceptance/b2d-controllers.md)、
[HUGSIM 控制器验收](2026-09-25-closed-loop-infra-acceptance/hugsim-controllers.md)、[SIGKILL 取证](2026-09-25-closed-loop-infra-acceptance/sigkill.md)、
[P5 复验](2026-09-25-closed-loop-infra-acceptance/b2d-controllers-p5.md)、[P6 / P7 复验](2026-09-25-closed-loop-infra-acceptance/b2d-controllers-p7.md)

## 为什么做

到目前为止 Bench2Drive（CARLA）和 HUGSIM 的闭环分数主要是被基础设施和控制器决定的，而不是被模型决定的：
Zoo PID 只执行了 Alpamayo plan 横向偏移的 4%，碰撞后把车顶在障碍物上
（[Alpamayo 闭环诊断](2026-09-24-zeroshot-exam/alpamayo-closed-loop-diagnosis.md) 第 7 节）；TCP partner 的出厂低速油门上限
让它最高 1.5 m/s；HUGSIM 官方控制器让 openpilot 20/20 原地打转；openpilot 适配曾让停着的车倒车；box 的 CPU 长期被
cgroup 限流在约 68/75 核，却没人量过一个 CARLA server 到底要多少核；还有两次查不出来源的 SIGKILL（不是 OOM）。
用户因此暂停了所有闭环模型考试，直到这套基础设施验收通过。**本任务不跑任何模型考试。**

## 要回答什么

1. **CARLA harness 的成本与最佳布局。** 我们的 `scripts/b2d_run.py` + CARLA 0.9.15（按我们的启动方式）每个 server 在
   sync 模式下吃多少核、RAM、VRAM，tick rate 多少；每张卡、整台 box 上加 server 时吞吐怎么变；瓶颈是渲染、CPU、
   traffic manager、Python agent 还是 I/O；在现在这台 box（5 × RTX PRO 6000 96 GB、cgroup 125 核、600 GB RAM）上
   worker 怎么摆最好。瓶颈在我们自己代码里的就优化（在子集上验证行为一致，记前后数字）。
2. **控制器验收。** 给我们用过的每个控制器喂一份已知是好的 plan（专家轨迹），看它能不能跟住、在预注册的小路线集上
   分数接近专家。逐个控制器给出 pass / fail、跟踪误差、DS（或 HD-Score）对专家参考。
3. **不明 SIGKILL 的来源**（openpilot policy server；2026-09-25 13:41 的 nuScenes 特征提取 `decision40-nusc-op` rc 137）。

## 资源

GPU 0、GPU 4 归本任务（schedule.md 2026-09-25 15:15 版），两卡合计 ≤ 70 核；CARLA server index 700–799
（控制器验收 700–739，profiling 740–789）。GPU 1 在 hugsim-scored-op 结束后可按借卡规则借用（≤ 4 h）。

## 结果

每一部分的预注册、数据与细节在各自的子文档里，这里只放结论与总表。英文定稿：[docs/closed-loop-acceptance.md](../docs/closed-loop-acceptance.md)。

### 1. CARLA harness：瓶颈与布局（[profiling.md](2026-09-25-closed-loop-infra-acceptance/profiling.md)）

**瓶颈是每张卡的 GPU 渲染，整台 box 的上限是 container 的线程数（`pids.max`），不是 CPU。** 同一条 Town12 路线、Alpamayo 考试
rig（4 路相机 10 Hz，走真实的考试 agent、不接模型），一张卡上加 server：

| 每卡 server 数 | 1 | 2 | 4 | **6** | 8 | 10 | 12 |
|---|---:|---:|---:|---:|---:|---:|---:|
| aggregate ticks/s（所有 worker 每秒合计推进的 20 Hz tick 数） | 7.9 | 14.6 | 27.4 | **34.8** | 36.4 | 36.3 | 38.2 |
| 单个 worker ms/tick | 127 | 137 | 146 | 173 | 220 | 276 | 315 |
| GPU utilization | 20% | 38% | 68% | 89% | 96% | 97% | 100% |
| 我们用的核数 | 3.1 | 5.9 | 11.2 | 13.2 | 13.1 | 12.4 | 12.4 |

每个 server：CarlaUE4 约 2.4 核、4.2 GB RSS、约 430 线程；route client（leaderboard + scenario tree + traffic manager + agent）
约 0.7 核、3 GB、约 215 线程（`carla.Client` 按 host 的 208 个硬件线程开 worker）；VRAM 约 5 GB；每 tick 约 0.33 core-s
（Town03 0.18）。cgroup 限流基本为零。**建议布局：每卡 6 个 server、每个 worker 预算约 2.5 核，先铺满更多的卡；整台 box
5 × 6 = 30 个 worker、约 75 核，前提是 route client 用 `--client-threads 8`**（线程 216 → 16，RSS 2.7 → 1.2 GB，行为在同配置重跑的
噪声内）；不加的话 20480 的线程上限在约 25 个 worker 时先到（2026-09-25 已经撞了 12 次，表现为 `carla.Client()` 报
`Resource temporarily unavailable`）。openpilot + TCP partner 的五路每 tick 相机 rig 每 tick 成本约是它的 3 倍，按每 worker 3 核、
每卡 ≤ 6 算。优化：`--client-threads 8` 采用；`--cache-lights` 原实现不等价（夜里留着 1633 / 2699 盏远处路灯），已修，
修后无收益、保持关闭；64 × 64 viewport 无收益。17:00 CST 之后其他作业占满 box，那之后的阶梯只做了同卡同时的 A/B，已在子文档标出。
**仍然开放**：91 次 server 启动里 14 次在 route setup 时 `GameThread timed out waiting for RenderThread` 后 Signal 11，与线程上限
无关，原因未查明，每次多花 60–90 s 重试；server index ≥ 616 的端口落在 kernel ephemeral range 里，可能在启动时撞上
`Address already in use`，建议 index < 490 或先 bind 检查。

### 2. 控制器验收

**B2D（[b2d-controllers.md](2026-09-25-closed-loop-infra-acceptance/b2d-controllers.md)）。** 已知是好的 plan = PDM-Lite 专家
（SimLingo 的 Bench2Drive 副本，原样）自己开出来的轨迹，按考试的 rig、节奏和控制器交给各臂；20 条预注册路线，TM seed 0。

| 控制器 | DS（专家 95.5） | 完成 / 20（专家 19） | 卡住 | 多出的碰撞路线 | e_lat p95 | \|e_lon\| median | lat_ratio | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Zoo PID，2 Hz（Alpamayo 考试） | 35.0 | 6 | 13 | 3 | 0.17 m | 8.3 m | 0.32 | **fail** |
| Zoo PID，5 Hz（openpilot 考试） | 30.1 | 5 | 14 | 5 | 0.21 m | 6.7 m | 0.40 | **fail** |
| 固定 20 Hz 跟踪控制器，2 Hz | **83.7** | **19** | **0** | 6 | 0.58 m | 3.1 m | **0.87** | **fail**（只差纵向） |
| 固定控制器，5 Hz | 57.0 | 18 | 1 | 12 | 0.40 m | 3.2 m | 0.21 | **fail** |
| P1：Zoo 纵向 + 固定横向 | 38.2 | 7 | 12 | 4 | 0.08 m | 7.6 m | 0.70 | **fail** |
| P2：Zoo 纵向 + 1.5 s aim 横向 | 35.1 | 6 | 13 | 4 | 0.13 m | 8.3 m | 0.62 | **fail** |

判据（预注册）：e_lat median ≤ 0.3 / p95 ≤ 1 m，|e_lon| median ≤ 2 / p95 ≤ 8 m，lat_ratio ≥ 0.7，DS ≥ 专家 − 5，专家完成的路线
不许卡住。Zoo PID 系的“卡住”被 replay 的 plan 形状放大（子文档“限制”），但它横向只执行 1/3、纵向没有位置反馈这两点不依赖
replay，单独就足以判 fail。固定控制器 2 Hz 横向合格、19/20 完成，输在比专家时刻表落后 3 m（推测：起步等第一个 plan、油门上限
0.75 而专家用到 1.0），晚到冲突点撞上 6 次；同一个控制器换 5 Hz plan 明显更差，所以它只在 2 Hz 下可用。
TCP partner 不是 plan → control 控制器，它的 1.5 m/s 是出厂低速油门上限，不在验收里。

**P5 复验（[b2d-controllers-p5.md](2026-09-25-closed-loop-infra-acceptance/b2d-controllers-p5.md)，20:05–21:32 CST）。** 必改项 1、3 的执行：
tfv6-controller 线的终版纵向 P5（D 横向）原样接进考试 agent（构造与 L1 逐位相同），replay plan 改成按时间索引、平滑追上，
同一 20 条路线跑 P5 @ 2 / 5 / 1 Hz 和新协议下的 F2（f2t，配对参考）；判据与第一轮相同，预注册于运行之前。

| 控制器 | DS（专家 95.5） | ΔDS 对专家 | ΔDS 对 f2t | 完成 | 碰撞（多出的路线） | e_lat p95 | lag | lat_ratio | 判定 |
|---|---:|---|---|---:|---|---:|---:|---:|---|
| f2t 固定，2 Hz | 80.6 | −14.9 [−26.6, −5.4] | — | 19 | 11 (6) | 0.55 m | −3.0 m | 0.88 | **fail** |
| P5，2 Hz | 85.7 | −9.8 [−18.8, −2.4] | +5.1 [−0.4, +11.4] | 19 | 5 (3) | 0.11 m | −2.7 m | 0.91 | **fail** |
| P5，5 Hz | **88.5** | −7.0 [−15.6, +0.2] | **+7.9 [+1.4, +15.0]** | 19 | 3 (2) | 0.11 m | −2.6 m | 0.90 | **fail**（最接近） |
| P5，1 Hz | 86.4 | −9.1 [−17.2, −1.5] | +5.9 [−5.7, +19.2] | 19 | 4 (4) | 0.17 m | −3.3 m | 0.88 | **fail** |

P5 横向在三个节奏下都合格（A1、A3），不过的是纵向：落后 2.6–3.3 m（A2 门槛 2 m），DS 差门槛 2.0–4.8。换成时间看，每次起步
都比专家晚约 0.5 s、之后稳态滞后 0.3 s：plan 只有位置，专家踩油门后约 0.5 s 车才动，只看位置的控制器要等 plan 动了才加速。
丢分来自晚到冲突点的碰撞和三条无信号路口的停车标志违规（后者可能部分来自新协议）。没有调参，下一步由用户定。

**HUGSIM（[hugsim-controllers.md](2026-09-25-closed-loop-infra-acceptance/hugsim-controllers.md)）。** 喂场景自己的 logged 轨迹、
以 ideal tracker 为参考，held-out 12 个场景：

| 控制器 | 横向 @0.5 s median / p95 | heading p95 | HD 对参考（均值 / 最差场景） | 判定 |
|---|---|---:|---|---|
| official + 适配层 | 0.16 / 1.18 m | 16° | −0.065 / −0.52 | **fail** |
| fixed（PR #57）+ 适配层 | 0.04 / 0.84 m | 10° | −0.002 / −0.04 | **fail** |
| fixed2（PR #57 + iLQR 0.25 s 离散化 + 转向速率代价 1） | 0.016 / 0.26 m | 3.4° | +0.006 / −0.001 | **pass** |

原因（离线复现了仿真器的跟踪误差）：iLQR 以 0.5 s 离散、仿真器每 0.25 s 执行一次（上游 issue #75），其次是转向速率代价。
适配层（forward_only、straight_stop）在 90 个 run、3904 个 plan 上是恒等变换。

### 3. 不明 SIGKILL（[sigkill.md](2026-09-25-closed-loop-infra-acceptance/sigkill.md)）

两次都是**单个 pid** 被 SIGKILL（02:15–02:22 openpilot Cinque policy server；13:41 nuScenes openpilot 特征提取主进程），同组的
父 shell 都活着；两次之后、下次重启之前读到的 `memory.events` oom / oom_kill 都是 0，所以**不是 kernel OOM**。排除：memcg 与整机
OOM、我们代码里所有 kill 路径、Mac 上所有 agent 在两个时间窗的命令、rlimit、pids.max、SIGHUP。**最可能**：AutoDL 平台在 kernel
之外按内存执法、杀容器里最大的进程（两次受害者都是当时匿名内存最大的一类，openpilot Cinque 进程约 27 GB；内存都贴着
`memory.high`、且都在大作业刚启动的几分钟内）。没有 root 拿不到 SIGKILL 的发送者，无法确证。已做：`b2d_run.py` 和
`b2d_tfv6_campaign.py` 的 reaper 以前可能误杀继承了 pid 的陌生进程（pid 约 4 h 回绕一次），现在只杀仍往自己 run 目录写日志的组；
`scripts/boxwatch.sh` 每 5 s 采样内存与最大进程，`slot_run.sh` 在作业被信号杀死时写 `<slot>.death-*.txt`。另外发现 container 的
线程上限（见 1）是另一类“查不出来”的失败的来源。

### 4. 闭环考试恢复之前必须做的

| # | 事项 | 为什么 |
|---|---|---|
| 1 | **B2D 不再用 Zoo PID**（包括只换横向的 P1 / P2）；P5 已复验（见上）：横向合格，纵向起步 / 再起步晚约 0.5 s 仍不过，需要修或解释后再复验 | Zoo PID 在专家 plan 上横向 1/3、纵向无位置反馈；P5 只差纵向 |
| 2 | 固定控制器只按 plan 的原生节奏用，5 Hz 的 openpilot 若要用它，先在 5 Hz 下单独过验收 | F5 比 F2 多 6 条碰撞路线 |
| 3 | ~~验收 replay 的 plan 来源改成“按时间索引、平滑追上”~~ 已做（`replay_plan: time`，P5 复验起使用） | ±2 m 窗口会冻住 plan，放大无位置反馈控制器的卡住 |
| 4 | HUGSIM 换 fixed2（`zs_run.py --controller fixed2`）；official 还要不要当 headline 由用户定 | official、PR #57 都不过验收 |
| 5 | CARLA 作业按线程数排：route client 一律 `--client-threads 8`，每卡 ≤ 6 个 server，整台 ≤ 30 | pids.max 20480 是 box 级上限 |
| 6 | 分数里报告重试次数（RenderThread 超时崩溃约 15% 的 server 启动） | 原因未查明 |
| 7 | 内存按容器上限的 85% 以内排（Cinque 进程按 27 GB 算），下一次 SIGKILL 看 death 文件和 boxwatch | 平台执法的假设要靠相关性定案；可以向 AutoDL 客服确认 |
| 8 | 已有的闭环分数（Alpamayo / openpilot 的 B2D、HUGSIM official 下的分数）按“控制器未验收”标注，不作模型结论 | 同上 |
