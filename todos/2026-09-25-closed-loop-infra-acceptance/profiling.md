# CARLA harness profiling：每个 server 的成本、扩展曲线、瓶颈与布局

状态: running（2026-09-25 15:25 CST 开始；本页先放中期结论，后续结果到了就地更新）
上级: [闭环基础设施验收](../2026-09-25-closed-loop-infra-acceptance.md)；英文定稿进 [docs/bench2drive-cost.md](../../docs/bench2drive-cost.md)

## 中期结论（2026-09-25 16:25 CST，给 P5 v1 CARLA generation 排布局用）

**Town12 上，Alpamayo 考试 rig 的瓶颈是 GPU 渲染，不是 CPU。** 一张 RTX PRO 6000 上从 1 个 server 加到 6 个，
aggregate tick rate（所有 worker 每秒合计推进的 simulation tick 数，20 tick = 1 秒模拟时间）从 7.9 涨到 34.8；
再往上加到 8、10、12 个几乎不涨（36.4、36.3、38.2），而 GPU utilization 在 6 个时已经 89%，8 个以上 96–100%。
同一段时间里这些 worker 一共只用 12–13 个核，cgroup throttling 基本为零。所以一张卡的 knee（再加 server 只换来
更慢的单个 worker、不换来吞吐的拐点）在 **6 个 server**。

**建议布局（Town12 类工作、相机 rig 与 Alpamayo 相当）：每张卡 6 个 server，每个 worker 预算约 2.5 核
（每卡约 15 核），先铺满更多的卡，而不是在一张卡上堆到 6 个以上。** 整台 box 5 张卡 × 6 = 30 个 worker，约 75 核。
CPU 的账按 tick 算最稳：不管几个 server，每推进一个 tick 大约花 0.33 core-seconds（server 0.25–0.31，
route client 0.07–0.09），所以一个 GPU 在 knee 处（约 35 ticks/s）大约要 12 核。小 town（Town03）和更重的
openpilot+TCP rig 还在量，小 town 的每 tick 更便宜、tick 更快，有可能先撞 CPU，结果出来后更新这里。

下面所有数字的条件：GPU 4（其余卡上有别人的任务，背景负载见表），我们的进程用 taskset 钉在 45 个 CPU（52–96，
GPU 4 所在的 NUMA node 1）；每个 worker 跑**同一条** route（Town12 route 1773，夜间、有 obstacle scenario，
ego 会被挡住，所以 1200 tick 都在跑），TM seed 0，`--max-ticks 1200`；数字取自 steady window（所有 worker
都已过 40 tick warm-up、且没有一个结束的那段时间，排除了 load_world 和 scenario 构建）。

### 每个 server 的成本（N = 1，Town12，Alpamayo rig）

| 项 | 数值 |
|---|---:|
| ms/tick（steady window） | 127 |
| ticks/s（real-time factor） | 7.9（0.39×） |
| CarlaUE4 server 进程 | 2.43 核，RSS 4.2 GB，374 线程 |
| route client 进程（leaderboard + scenario_runner + traffic manager + agent） | 0.69 核，RSS 3.0 GB，214 线程 |
| VRAM | 约 5.0–5.3 GB / server（6 个 server 30.6 GB） |
| GPU utilization | 20% |
| route setup（load_world + scenario 构建，到第一个 tick） | 56 s（N = 12 时 92 s） |
| tick 内分解（leaderboard 内计时） | world.tick 95 ms，scenario tree 20.5 ms，agent 9.0 ms |

### 一张卡上的扩展（Town12，Alpamayo rig，n = 每档 N 个相同 route，各 1200 tick）

| server 数 | aggregate ticks/s | 相对 N=1 | per-worker ms/tick | GPU util | VRAM | 我们的核数 | server 核/个 | client 核/个 | 背景核数 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.9 | 1.00× | 127 | 20% | 6.3 GB | 3.1 | 2.43 | 0.69 | 23 |
| 2 | 14.6 | 1.86× | 137 | 38% | 11.0 GB | 5.9 | 2.28 | 0.66 | 27 |
| 4 | 27.4 | 3.49× | 146 | 68% | 21.4 GB | 11.2 | 2.18 | 0.63 | 19 |
| **6** | **34.8** | **4.42×** | 173 | 89% | 30.6 GB | 13.2 | 1.71 | 0.49 | 22 |
| 8 | 36.4 | 4.63× | 220 | 96% | 39.7 GB | 13.1 | 1.28 | 0.36 | 32 |
| 10 | 36.3 | 4.61× | 276 | 97% | 49.7 GB | 12.4 | 0.97 | 0.27 | 31 |
| 12 | 38.2 | 4.85× | 315 | 100% | 58.9 GB | 12.4 | 0.81 | 0.23 | 38 |

读法：N 从 1 到 4 时 per-worker ms/tick 只慢 15%，吞吐几乎线性；4 到 6 还能拿到 27%；6 以后吞吐平了，
多出来的 server 只是把每个 worker 拖慢（12 个时单个 worker 315 ms/tick，是 N=1 的 2.5 倍）。每个进程的核数
随 N 下降，是因为每个 worker 的 tick 变慢了，每 tick 的 CPU 成本本身不变。背景核数是整个 container
（cgroup `cpu.stat`）减去我们的进程，也就是 GPU 0–3 上别人的任务；这期间 throttled 时间基本为零。

### 顺手发现的两个问题（已处理）

- **我们的 CARLA 端口在 kernel 的 ephemeral port range 里。** box 上 `ip_local_port_range` 是 32768–60999；
  server index i 的 RPC 端口是 2000 + 50i，所以 i ≥ 616 的端口（schedule 分给我们的 700–799 全在里面，
  TM 端口 8000 + 50i 从 i ≥ 496 起就在里面）可能正被某个出站连接占着。CARLA 这时在启动阶段报
  `bind: Address already in use` 然后 Signal 11；index 780 实际撞上过一次。`b2d_run.port_free` 只检查有没有人在
  listen，查不出这种情况。`scripts/b2d_scale.py` 改成真 bind 一次再用；生产里建议 index 用 600 以下，或者同样先 bind 检查。
- **`--cache-lights` 原来的实现不等价。** 它在第一次调用时读 `is_on` 建缓存，但夜间 server 会在 route 开始前后自己把
  路灯打开，缓存以为是灭的，结果 Town12 上 2699 盏路灯里有 1633 盏远处的灯一直亮着（原实现会把半径外的灯全关掉）。
  这很可能就是它当初在单实例上"tree 快了、墙钟反而慢了"的原因：多渲染了一千多盏灯。已修（第一次更新下发完整状态），
  用 `$B2D_LIGHTS_CHECK`（每 20 次更新从 server 读回所有灯、对照"半径内亮、半径外灭"的规则）验证：修后 0 处不符，
  原实现 0 处不符，旧缓存每次 1633 处不符。

## 方法

（跑完补全：`scripts/b2d_scale.py` 的 steady window、采样内容，`scripts/infra_scale.sh` 各步，等价性检查。）
