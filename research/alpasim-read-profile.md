# AlpaSim read 作业剖析（2026-10-13，perf lane）

对象：GPU 池里占主导的 AlpaSim read 作业（`ol_chain.sh readlist` / `ro_chain.sh run` / `rm_chain.sh read` → `experiments/alpasim/scripts/run.sh sh30` → `run_native.py` 起 driver + AlpaSim 四个服务）。
三个问题：8 分钟花在哪、~25 GB 内存在哪、4 张/卡在 VRAM 上能否成立。

**结论先行：早停闸门触发。** 墙钟时间约 89 % 在 AlpaSim 自己的代码里（渲染器、runtime 启动/聚合、控制器 MPC），我们的 driver 只占 rollout 时间的 6 %，另有约 11 % 是 driver 启动。所以没有改任何代码，也没有提交任何东西。
能动的杠杆都在“我们怎么配置、怎么切分、怎么计量”这一层：wizard overrides、每个作业的场景数、一个 allocator env，以及池子的 RAM 计量口径。下面全部写成提案。

## 数据来源（哪些是实测，哪些是推断）

| 来源 | 内容 | 性质 |
|---|---|---|
| 300 个已完成的 read 作业（route_follow / onpol_long / route_only / route_mem，最近的 300 个，140 个 10-decision、160 个 18-decision） | `native_summary.json`（阶段耗时、按服务的峰值）、`usage.jsonl`（按服务每秒的 RSS / GPU）、AlpaSim 自带的 `telemetry/*.prom`（按 RPC 的耗时直方图） | 实测 |
| 8 个在跑的 `redg-rg10-*` 作业 | 在作业外部读 `/proc/<pid>/status` + `smaps_rollup`：anon / file / Pss_Anon，按进程 | 实测快照 |
| 本 lane 的 7 个池作业（owner `perf`，priority 1） | smoke（2 个场景）+ 6 个配置臂，同一份 64 场景 10-decision 列表（`all_n10/c3.txt` 的前 64 个），`SH30_TAG=RRED-L18-s0`；外部采样器每 2 s 采一次（按进程的 anon/PSS/CPU/GPU）+ `nvidia-smi pmon`，其中 Aspy 臂整棵进程树跑在 py-spy 下（20 Hz，nonblocking，`--subprocesses`） | 实测，每个臂只跑了 1 次 |
| AlpaSim 源码（`0bb4c4b`） | 内存增长和 VRAM 抖动的原因 | 读代码 |

包装脚本和分析脚本放在 `tmp/arp/`（Mac 和 box 都有，gitignored）：`prof.sh`、`sampler.py`、`ana.py`、`spy.py`、`mine.py`、`prom.py`、`ts.py`。原始数据在 `$DATA_DIR/runs/perf/arp/{smoke,A,B,C,D,E,Aspy}/`。没有改动任何 lane 脚本、`run_native.py`、`pool.py`、`profiles.py`。

## 1. 墙钟时间花在哪

### 1a. 作业层面的阶段（臂 A = 现行配置，64 个场景，实测）

| 阶段 | 秒 | 占比 | 属于谁 |
|---|--:|--:|---|
| driver 启动（加载 checkpoint、onnx、torch.compile 预热，`ready in 37-45 s`） | 43.5 | 11 % | **我们** |
| wizard + controller + renderer 就绪（driver 就绪之后才开始，串行） | 19 | 5 % | AlpaSim（串行顺序是 `run_native.py` 定的） |
| runtime 启动（trajdata 索引、scene loader、spawn workers） | 25 | 6 % | AlpaSim |
| 仿真（rollouts） | 277 | 70 % | 见 1b |
| 结束时的 eval 聚合 + 画图 | 31 | 8 % | AlpaSim |
| 收尾（kill、写 summary） | ~0 | 0 | 我们 |
| **合计** | **395** | | |

300 个历史作业：总时长 p50 536 s（10-dec，150 个 rollout：575 s；18-dec，77 个 rollout：455 s），driver 就绪 p50 36 s，服务就绪 p50 14 s。场景越多，固定开销（约 100-120 s）占的比例越小。

### 1b. 仿真内部：rollout 时间按 RPC 拆分（AlpaSim telemetry，300 个作业的中位数，实测）

| 部分 | 10-dec | 18-dec | 属于谁 |
|---|--:|--:|---|
| `sensorsim.render_aggregated`（MTGS 渲染，每次调用 1.7 s，其中大部分是排队） | 76.7 % | 79.7 % | AlpaSim |
| `sensorsim.get_available_cameras / ego_masks`（每个场景一次的 renderer 场景加载） | 10.9 % | 7.1 % | AlpaSim |
| controller `run_controller_and_vehicle`（casadi MPC） | 7.7 % | 8.4 % | AlpaSim |
| driver（drive + 各种 submit_*） | 6.0 % | 6.1 % | **我们** |

每个 drive 调用约 29 ms，比渲染一帧少一个数量级以上。`submit_image_observation` 一共 127 s，接收的 8 路 JPEG 里只解码 CAM_F0，这是 driver 一侧最大的单项，但在作业墙钟里分摊到不足 1 %。

### 1c. CPU 和 GPU 的忙碌度（Aspy 臂：py-spy 活跃线程时间；采样器给出平均核数；pmon 给出 SM 占用，均为实测）

| 进程类 | 活跃线程-秒（py-spy） | 占比 | 平均核数 | pmon SM % |
|---|--:|--:|--:|--:|
| renderer（`mtgs.py:480 render` 35 %，`set_sensors` 25 %） | 848 | 63 % | 2.1 | 12-15 |
| runtime workers + eval 进程池（序列化、deepcopy、地图加载） | 294 | 22 % | 0.6 | – |
| controller（casadi `call` 63 %） | 93 | 7 % | 0.25 | – |
| driver（PIL 解码、gRPC；其中约 1/3 是启动时的 tune/compile） | 77 | 5.7 % | 0.2 | 4-6 |

整个作业平均约 3.5-4 核，GPU 占空比低（renderer 加 driver 不到 20 % SM）。瓶颈在单个 renderer 进程：它是 4 线程的 gRPC 服务，主要卡在 Python/GIL 和 GPU 同步点上。docs/alpasim.md 里“8 并发 2.96 s/场景，16 并发 2.93”的观察与此一致。

**早停闸门：** 70 % 的线是“墙钟时间在 AlpaSim 自己的代码里”。实测约 89 %（仿真里 94 % 的 rollout 时间 + runtime 启动 + 聚合 + 服务启动），所以闸门触发。代码一律不动；第 4 节只列配置层面的提案。

## 2. 内存在哪

### 2a. 先校正口径：RSS 比真实占用多出约 5 GB

池子（`pool.py: rss_gb`，`statm` 第 2 列）和 `run_native.py` 的 `usage.jsonl` 都把整棵进程树的 **RSS** 加起来。对这类作业来说这个数偏大：

- 每个进程都有约 0.33 GB 的 file-backed RSS（共享库），15 个左右的进程合计约 **4.7-5.3 GB**，但 Pss_File 加起来只有 **0.2 GB**。这些页在 page cache 里是所有作业共用的，在 kill line 里只算一次。
- driver 启动时 fork 出 9 个 inductor compile worker（写时复制），RssAnon 合计 3.4 GB，PSS 只有 0.5 GB；历史记录里“driver 峰值 6.5 GB”就是这样来的。预热结束后只剩 1 个 compile worker（0.4 GB），driver 本体稳定在 2.7-2.8 GB anon。
- 作业的峰值出现在快结束时（墙钟的 87-94 % 处），那时 runtime 最大，driver 的启动尖峰早就过去了。

所以对 kill line 有意义的是 **Pss_Anon ≈ RSS − ~5 GB**。题目里说的 ~25 GB 是 RSS，真实的私有内存约 17-23 GB（见下表）。

### 2b. 按进程拆分（臂 A，64 个场景，在作业 Pss_Anon 峰值那一刻，实测）

| 进程 | Pss_Anon GB | 随场景数增长？ | 属于谁 |
|---|--:|---|---|
| runtime workers ×2（trajdata 场景数据源） | 6.7（起步 2.2） | **是，约 0.07 GB/场景（10-dec 缓存）** | AlpaSim |
| renderer（MTGS，CPU 侧） | 3.5 | 否 | AlpaSim |
| runtime eval 进程池（每个 consumer 一个 spawn 进程：并发 8 → 8 个，每个约 0.5 GB；峰值时还剩 6 个） | 2.8 | 否，随并发数变 | AlpaSim |
| driver（sh30_driver，op-train env） | 2.8 | 否 | **我们** |
| runtime 父进程 | 0.7 | 否 | AlpaSim |
| controller | 0.5 | 否 | AlpaSim |
| inductor compile worker（预热后剩 1 个） | 0.4 | 否 | **我们**（torch） |
| **合计** | **17.5**（同一时刻 RSS 约 22.7） | | |

增长的原因（读代码确认）：`alpasim_runtime/scene_loader.py:264-294` 里 trajdata provider 用一个**不做淘汰的 dict** 缓存每个场景的 `TrajdataDataSource`，跑过的场景全部留在 worker 里直到作业结束。历史曲线与此吻合：runtime 的 RSS 在墙钟 10 / 50 / 90 % 处分别是 5.6 / 14.7 / 17.3 GB（10-dec，150 个场景），18-dec（after160 缓存，场景更长）每个场景约 0.1 GB（docs/alpasim.md 给的是“11 GB + 0.1 GB/场景”）。

由此得到的模型（10-dec，现行配置）：**Pss_Anon ≈ 12.9 GB + 0.069 GB × 每个作业的场景数**。150 个场景 → 23.3 GB（历史 RSS 27.8 − 5 ≈ 22.8，对得上）；18-dec 77 个场景：RSS p50 21.5 → anon 约 16.8（推断）。

**能由我们减掉的：** driver 的 2.8 + 0.4 GB（在第 4 节里收益很小），以及每个作业的**场景数**和**并发数**——这两个是我们的配置，内存的大头随它们变。其余都是 AlpaSim 的代码。page cache 部分（trajdata cache 是 file-backed 的读，共享库）不属于某一个作业。

## 3. 4 张/卡在 VRAM 上能否成立

VRAM 几乎全在 AlpaSim 的 renderer 上：

| | 实测 |
|---|---|
| 我们的 driver | nvidia-smi 2.2 GB；torch max_reserved 1.5 GiB，max_allocated 1.43 GiB（`driver-logs/vram.json`），其余是 CUDA context。`empty_cache` 最多能省约 0.1 GB，**没有可减的** |
| renderer | 300 个作业 p50 20.2 GB、p90 23.8 GB，单个作业最高 31 GB；在同一次运行内部会在 10 到 24 GB 之间来回摆动（FIFO 场景缓存 `--cache-size` 淘汰时会 `empty_cache`，`plugins/mtgs/server/servicer.py:209`；渲染时还有临时 buffer） |

所以在我们的代码里，VRAM 峰值没有可减的部分。能减的只有两个配置旋钮（单次运行，64 个场景，scores 一致）：

| 臂 | 配置 | 总时长 s | runtime s | 作业 Pss_Anon 峰值 GB | 作业 VRAM 最大 / p95 / p50 GB | renderer 峰值 GB | 平均分 |
|---|---|--:|--:|--:|---|--:|--:|
| A | 现行：2 workers，8 并发，cache 9 | 395 | 333 | 17.5 | 18.3 / 16.7 / 13.1 | 16.1 | 0.9438 |
| Aspy | 同 A，跑在 py-spy 下 | 451 | 381 | 17.6 | 18.3 / 16.6 / 13.8 | 16.1 | 0.9438 |
| B | 1 worker | 423 | 355 | 16.6 | 18.7 / 16.7 / 13.7 | 16.5 | 0.9437 |
| C | 4 并发，cache 5 | 431 | 365 | 15.7 | 14.5 / 11.8 / 9.7 | 12.3 | 0.9438 |
| D | 1 worker + 4 并发 + cache 5 | 463 | 404 | 14.9 | 14.3 / 12.7 / 10.6 | 12.1 | 0.9438 |
| E | 现行 + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 448 | 387 | 18.1 | 15.4 / 15.3 / 12.3 | 13.3 | 0.9438 |

注意：A、B、C 同时跑在 0 号卡上；D、E、Aspy 晚 7 分钟在别的卡上跑，邻居不同。同一配置的 A 和 Aspy 之间 runtime 就差了 14 %（py-spy 的开销加上邻居），所以**时长的差别在 ±15 % 以内都算噪声**，只有 D 的 +21 % 超出了这个范围。内存和 VRAM 是作业自己的属性，受邻居影响小。

- C（4 并发 / cache 5）：VRAM 最大值 −3.8 GB（−21 %），p95 −4.9 GB；RAM −1.7 GB（eval 池从 8 个进程降到 4 个，renderer −0.6 GB）；时长在噪声范围内。因为瓶颈本来就是那一个串行的 renderer，减半并发的代价比直觉小。
- E（expandable segments）：renderer 峰值 −2.8 GB，RAM 不变，分数相同。只有 1 次运行，是碎片化减少带来的效果，需要复核。
- B（1 worker）：只省 0.9 GB（场景缓存不会在两个 worker 之间重复，省掉的只是一个 worker 的基础占用），不值得做。

**4 张/卡的算术。** 每卡可用约 83.6 − 4 = 79.6 GB，4 个作业每个最多 19.9 GB。现行配置的历史 p95 是 26 GB，**不成立**。C 的最大值约为 A 的 0.79 倍，乘到历史 p95 上约 20.6 GB（推断）；C+E 叠加没有测过，按两者各自的降幅估计在 18-19 GB 左右（推断，未测）。也就是说，4 张/卡在 VRAM 上**只有改了配置（C，最好再加 E）之后才有可能**，而且要先用约 10 个真实 chunk 的历史确认 p95 ≤ 19 GB。GPU 算力不构成限制（每个作业 SM 不到 20 %，4 个作业约 70-80 %，再往上 renderer 的排队会变长，属推断）。CPU 也不构成限制（每个作业约 4 核，24 个作业约 100 核，配额 150）。

## 4. 我们这一侧可减的项（按价值排序）

| # | 改什么（在哪） | 预计收益 | 代价 / 风险 |
|---|---|---|---|
| 1 | **池子的 RAM 计量从 RSS 改成 RSS − shared**（`pool.py: rss_gb` 用 `statm` 的第 2 列减第 3 列；或者用 smaps_rollup 的 Pss_Anon） | 每个 read 作业的计量少约 5 GB（真实占用不变，只是不再把共享库按进程重复计算）；history cap `1.2×max+1` 跟着从约 35.8 降到约 29.6 GB（150 个场景）；对所有多进程作业都更准 | 低。`pool.py` 正在被其他会话使用，我没动；读 statm 第 3 列和现在一样便宜 |
| 2 | **每个作业的场景数**：10-dec 列表从 150 个场景切到 75 个（各 lane 的 list 切分） | 每个作业 −5.2 GB anon（0.069 × 75） | 每多一个作业，固定开销多约 100-120 s（driver 40 + 服务 19 + runtime 启动 25 + 聚合 ~20-30）：150 个场景的作业 575 s → 两个约 650 s，**卡时 +13 %**。只有当 RAM 而不是 VRAM 是瓶颈时才划算 |
| 3 | **wizard overrides 改成 C**：`n_concurrent_rollouts=4`（renderer / driver / controller 三处），`defines.nre_cache_size=5`（`c0b_chain.OVERRIDES`，各 lane 都在 import） | VRAM 峰值 −4 GB，RAM −1.7 GB | 64 个场景上分数一致；时长 +10 %（在噪声内，未复测）。正在跑的 lane 如果要求条件完全相同，应等到 lane 之间的边界再换 |
| 4 | **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** 加进作业 env（作用于 renderer 和 driver） | renderer VRAM 峰值 −2.8 GB（n=1） | 不改数值（分数一致）；属于第三方的运行时 env，不算改代码。需要 3-5 次复测 |
| 5 | **driver 预热和服务启动并行**（`run_native.py`：先起 wizard / controller / renderer，等 driver 端口就绪后再起 runtime） | 每个作业 −15~19 s（墙钟 3-4 %） | 低，但 `run_native.py` 是共用的，要所有 lane 一起换 |
| 6 | driver：`TORCHINDUCTOR_COMPILE_THREADS=1`（不再保留 compile worker 子进程） | −0.4 GB anon | cache 未命中时首次编译会变慢；收益很小 |
| 7 | `runtime.nr_workers=1` | −0.9 GB | 时长 +7 %（噪声内）；不推荐 |

AlpaSim 那一侧最大的一项（不做淘汰的场景缓存，每个场景约 0.07-0.1 GB）不在我们的改动范围内；如果要做，应该向上游提 issue（可以按 LRU 淘汰，或者每个 rollout 用完就丢），而不是本地改。

**第 4 项的 diff（只是准备好，未提交）。** CLAUDE.md 规定线程 / env 只放在 `profiles.py` 里，但现有 profile 的 `num_threads=2` 会把 OMP 等线程数强加给 renderer，所以单独建一个 profile，不设 `num_threads`：

```diff
--- a/jevdrive/cl/profiles.py
+++ b/jevdrive/cl/profiles.py
 PROFILES = {
     "stock": Profile("stock", pools="stock", client_threads=0, num_threads=None),
     "reduced": Profile("reduced", pools="reduced", pool_threads=4, client_threads=8, num_threads=2),
+    # AlpaSim stack (renderer + driver): allocator env only; numeric threads untouched (research/alpasim-read-profile.md)
+    "alpasim": Profile("alpasim", pools="stock", client_threads=0, num_threads=None,
+                       env=(("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),)),
 }
```
使用方式：提交时加 `--profile alpasim`。先要确认池子对非 CARLA 作业也会应用 `environ()`；如果不会，就改用 `--env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，效果一样。

## 5. 池子该怎么声明

| 情况 | `--vram` | `--ram`（按池子现在的 RSS 口径） | `--cpu` |
|---|--:|--:|--:|
| 现行配置，10-dec，约 150 个场景 | 26（3 张/卡） | 30（RSS 最大 29.0；anon 约 23） | 6（实测平均 3.5-4 核，峰值约 6；5 分钟后池子按实测峰值 ×1.2 计费） |
| 现行配置，18-dec，约 77 个场景 | 26 | 24（RSS 最大 22.8；anon 约 17） | 6 |
| 改用 C(+E)，并且约 10 次复测确认 p95 ≤ 19 GB 之后 | 20（4 张/卡） | 10-dec 75 个场景：约 21（anon 约 16.5）；18-dec 77 个场景：约 20（anon 约 15） | 6 |
| 再加上第 1 项（池子按 anon 计量） | 同上 | 各减 5 | 同上 |

要让 RAM 降到 15 GB（anon 口径）：18-dec 77 个场景的作业改用 C 就能做到（推断约 15.1）；10-dec 作业还需要把每个作业的场景数降到约 55 个以下（11.2 + 0.069 × 55 ≈ 15）。按池子现在的 RSS 口径，以上数字都要再加约 5 GB。所以**先做第 1 项，否则无论怎么调，计量上都达不到 15 GB**。

## 下一步（待定，要用户或 lane owner 拍板）

1. 用 C+E 在一个 lane 的边界跑约 10 个真实 chunk，作为 VRAM 的 p95 复测（同一列表，现行配置的历史作为对照）。
2. 池子 RAM 口径的改动（第 1 项），交给维护 `pool.py` 的会话做。
3. 第 2 项（场景数）要等 1、2 做完再看 RAM 是否仍是瓶颈。

## Applied 2026-10-13

**采纳了什么。** (1) `c0b_chain.OVERRIDES` 改成 4 并发 / cache 5（臂 C），所有 import 它的 lane（read 和 rollout）都跟着变；rollout 只是并发数变了，数值不受影响。(2) `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（臂 E）写成 `profiles.py` 里的 `alpasim` profile，由 `run_native.py` 自己读进来、给它起的每个进程（不需要 `--profile`，也不用重启 dispatcher；作业自己设了同名 env 时以作业为准）。(3) `run_native.py`：driver 一启动就去起 wizard 和非 runtime 的服务，runtime 起之前再等 driver 端口；driver 在预热时死掉，`wait_port` 立刻报错，`finally` 里照旧 kill 全部服务。第三方代码没动。提交 `5633b19c`。

**Pilot。** 10 个真实 chunk，与基线 1:1：`ol_chain.sh readlist RRED-L18-s0 onpol_long/lists/all_n10/c<i>.txt 10`，基线是同一 checkpoint、同一列表今天 15:30 跑的 `redg-rg10-<i>`（`route_follow/guard/read10/RRED-L18-s0/b<i>`，旧配置）。新配置从 box 上的一个 worktree 提交（`arp-pack-<i>`，owner perf，`--vram 26 --cpu 6 --ram 30`，`CL_HIST_KEY=arp-pack`），10 个作业同时跑在 6 张卡上（每卡 1-2 个）；基线当时也是 5 张卡、每卡约 2 个，还有别的 lane 在跑。VRAM = 池子测到的作业峰值；墙钟 = 池子的 t1 − t0。

| chunk | 新 VRAM 峰值 GB | 旧 | 新墙钟 s | 旧 | runtime 起跑 s（新 / 旧） | 分数（新 = 旧） |
|---|--:|--:|--:|--:|---|--:|
| 0 | 13.7 | 15.0 | 657 | 575 | 38 / 53 | 0.9448 |
| 1 | 12.3 | 18.3 | 657 | 575 | 38 / 51 | 0.9102 |
| 2 | 13.9 | 18.0 | 657 | 575 | 38 / 53 | 0.9221 |
| 3 | 11.7 | 15.6 | 657 | 596 | 38 / 51 | 0.9359 |
| 4 | 17.4 | 30.6 | 636 | 596 | 37 / 52 | 0.8903 |
| 5 | 11.9 | 15.7 | 636 | 597 | 36 / 52 | 0.9393 |
| 6 | 11.6 | 18.3 | 657 | 618 | 36 / 53 | 0.9405 |
| 7 | 11.8 | 19.4 | 657 | 598 | 36 / 52 | 0.8931 |
| 8 | 13.0 | 18.3 | 636 | 619 | 36 / 53 | 0.8926 |
| 9 | 12.6 | 20.0 | 636 | 622 | 35 / 59 | 0.9119 |
| **汇总** | p50 12.5，**p95 = max 17.4** | p50 18.3，p95 = max 30.6 | 中位数 **657** | 596.5 | −16 s | 均值 0.9181 = 0.9181 |

- **分数**：10 个 chunk 每一个都与基线一致到小数点后 4 位，没有失败的 rollout。闸门通过。
- **VRAM**：p95（10 个作业的最近秩 p95，即最大值）17.4 GB，≤ 19，所以 4 张/卡成立。基线里 30.6 GB 那个尖峰（c4）在新配置下是 17.4。
- **RAM**：池子 RSS 峰值 19.9-20.3 GB，原来 26.3-28.2，每个作业少约 8 GB（比臂 C 单次测的 −1.7 GB anon 多：eval 进程池减半，各进程的共享库 RSS 也跟着少算）。
- **墙钟**：中位数 +10 %（≤ +15 %，通过）。启动重叠让 runtime 提前约 16 s 开跑；runtime 本身慢了约 14 %（中位约 600 s，原来约 530 s），这是 4 并发的代价。按卡算吞吐：3 个/卡 → 4 个/卡、每个慢 10 %，约 +20 %；前提是同卡 4 个时 runtime 不再明显变慢，这次 pilot 每卡最多 2 个，**没测过 4 个同卡**。

**声明。** p95 17.4 ≤ 19，按规则取 `--vram 20 --cpu 6`：每个 read 占 20 / 83.6 ≈ 24 % 张卡（原来 26 GB ≈ 31 %，3 个/卡）。改了 ro_chain.sh、rf_red.sh、rf_stopline.sh、rf_allrows.sh、rm_chain.sh、ol_chain.sh、ol_fast.sh 里所有 `--vram 26 --cpu 8` 的 read 行；rollout 行仍是 32。history key 从 `ol-fast-read` 换成 `alpasim-read4`：旧 key 里存的是 8 并发时的峰值（到 28-31 GB），新作业挂在旧 key 下会被池子判成“声明过低”，按旧峰值 + 1 GB 记账，反而比 26 还多占。`--ram` 没动（30 / 40）；实测 20 GB，等池子改成按私有内存计量之后可以再降。docs/closed-loop-runbook.md 的 “Fixing a queued job's declaration” 已更新。

**下一步**：同卡 4 个 read 并行时的墙钟没测过，头一批 4 张/卡的 lane 作业跑完后，用 `cl usage` / 作业墙钟复核一次；如果 runtime 比这次再慢 15 % 以上，就把声明提到 26（3 个/卡）。
