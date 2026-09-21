# Waymo train split 的冻结特征抽取

状态: running
主题: ../research/frozen-vlm-planner.md

## 目标

把 Qwen3-VL-4B 的冻结中层特征在 Waymo E2E **train** split 上抽完（约 41.4 万帧，263 个 shard），
边下边抽。下游全部堵在这上面：head 目前只在 239 个 val sequence 上训过，
[decisions.md](../research/decisions.md) 第 3d 条那次证伪了第 1 条的测量，要用 train 的 2037 个 sequence 重做。

## Setup

- 数据: `$DATA_DIR/datasets/waymo_e2e/front3/training_*.tfrecord-*`，263 个 shard，前 3 个相机，下载中（2026-09-21 11:08 起，约 13.3 MB/s，ETA 14 h）
- 模型: Qwen/Qwen3-VL-4B-Instruct，bf16，完全冻结，`torch.compile` 默认 mode
- 特征集: `qwen_front3`（和 val 同一个 set），输出 `processed/waymo_e2e/features/qwen_front3/<shard>/`
- 算力: 一张 RTX PRO 6000 Blackwell，抽取期间独占 GPU；download 占着网络和一部分 CPU
- run dir: `$DATA_DIR/runs/waymo_e2e/features_train/20260921-165650/`

## Profiling（跑之前，CLAUDE.md 要求 3 h 以上的任务先做）

单流 15 h 的估计成立，所以先测。测量条件：download 正在跑，box 25 核。

### 一帧（3 个相机、一次 forward）各阶段的代价

| 阶段 | 单线程 ms/帧 | 说明 |
|---|---:|---|
| `pread` 3 段 JPEG | 0.1 | index 里记了 shard 内的字节区间，没有 protobuf，没有整条 record |
| JPEG decode（3 × 972×1079） | 17.8 | |
| Qwen processor（smart_resize → 960×1088、normalize、patchify） | 38.6 | |
| **decode + processor 合计** | **63.1** | 一个 DataLoader worker 每个 item 付的代价 |
| GPU forward，compile=True，batch 4 | **132.0** | |
| GPU forward，compile=True，batch 8 | 132.3 | |
| GPU forward，compile=False，batch 4 | 183.4 | `torch.compile` 本身省 28% |

**结论：这条 pipeline 是 GPU-bound，不是 preprocessing-bound。** GPU 每 132 ms 要一个 item，
一个 worker 每 63 ms 产一个，所以**两个 worker 就足够喂满**。
val 实测端到端 129.2 ms/帧（存在 meta.json 里），和纯 forward 的 132 ms 基本相等，
说明 I/O 和预处理已经被完全 overlap 掉了，pipeline 没有 stall 可捡。

### forward 内部（batch 8）

| 部分 | ms/帧 | 占比 |
|---|---:|---:|
| ViT 24 blocks（12240 个 patch / 帧） | 44.5 | 35% |
| LLM 36 层（3074 token，其中 3060 个是图像 token） | 72.8 | 57% |
| merger、DeepStack、pooling、H2D | 10.3 | 8% |

LLM 那一段：非 embedding 参数约 3.0B，3074 token，即约 18.4 TFLOP / 72.8 ms = **253 TFLOPS bf16**，
已经贴着这张卡 dense bf16 的屋顶。ViT 同量级。
也就是说**不改特征定义就没有可省的**：省时间只能靠减 token（降分辨率）或减层（early exit），
两者都会让 train 的特征和盘上的 val 特征不可比。

### `draft` 快速解码这条线索：测了，不适用

CARLA 那边的发现（preprocessing 占 Qwen 延迟 70%，按模型输入尺寸直接出图把 139 ms 砍到 59.9 ms）
在这里**不成立**，原因有两层：

1. 这里是离线批量抽取，不是 batch-1 在线推理。GPU 吃满了，preprocessing 全部藏在 worker 里，
   省 CPU 等于省了一个本来就闲着的资源。
2. 更根本的：Qwen 的 `smart_resize` 把 972×1079 放到 **960×1088**，几乎就是原生分辨率，
   根本没有降采样给 `draft` 利用。实测 `draft` 到 1/2 是 486×540（14.9 ms），1/4 是 243×270（13.2 ms），
   相对全解码的 17.8 ms 只省 3-5 ms，而且都远小于模型要的尺寸，用了就得再插值放大回去，
   **每一个特征都会变**。所以不用。

### batch size 是特征定义的一部分（这条最意外）

在已经建好的 val shard 0 上重抽前 128 帧，和盘上的数组逐位比较：

| 配置 | ms/帧 | 10 个数组里有几个逐位相同 | 最大相对偏差 |
|---|---:|---:|---|
| batch 4, workers 6 | 125.0 | **10 / 10** | 0 |
| batch 8, workers 6 | 125.7 | **0 / 10** | `L36_last` 上 mean\|Δ\|/mean\|ref\| = 2.0e-2 |

batch size 变了，GEMM 的 tiling 和 kernel 选择就变，误差沿 36 层累积，到 `_last` 那几个特征上
已经是 1-2% 的相对偏差——**远在 float16 存储噪声之上**。速度又完全一样。
所以 **batch 必须钉死在 4**，train 才能和 val 比。这一条写进了 `scripts/waymo_features_watch.sh` 的注释
和 [docs/waymo-e2e.md](../docs/waymo-e2e.md)。

### before / after

瓶颈是 GPU compute，本来就贴着屋顶，所以抽取速率没有变（129.2 → 125.0 ms/帧，是 run 间噪声）。
这一轮 profiling 买到的是另外四件事：

| 项 | before | after | 效果 |
|---|---|---|---|
| DataLoader workers | `n_cpus() // 2` = 12 | `max(2, min(6, n_cpus() // 4))` = 6 | 还给 download 6 个核，抽取速度不变（实测需要 2 个） |
| batch size | 没有约束，watcher 用 `BATCH` 随便传 | 钉死 4，并写明理由 | 避免 train / val 特征不可比 |
| `draft` 解码 | 待验证的线索 | 测完否掉 | 不引入一条会改变特征的解码路径 |
| 每个 pass 重新载入模型 | 是（bash 循环每次调一次 `features_inc`） | 否，模型在整个 run 里只载入编译一次 | 追上下载之后一个 pass 就是一个 shard，省掉每次约 20 s |

## 步骤

- [x] profiling：分阶段计时、forward 内部拆分、`draft` 验证、batch size 等价性
- [x] `jevdrive.waymo` 加 `features_inc --watch`：模型常驻、每个 pass 先重跑 index、按 split 计数、坏 shard 不阻塞
- [x] `scripts/waymo_features_watch.sh` 改成薄启动脚本，接受 split 参数
- [x] test split 单 shard 冒烟（738 帧，129.1 ms/帧，整条 loop 走通）
- [x] 起 train 长跑：`scripts/tmux_run.sh wfeat-train env INTERVAL=300 scripts/waymo_features_watch.sh train`
- [ ] 跑完核对：263 个 shard 全有 meta.json，`load_features` 能按 `frame_name` 合并
- [ ] 把结果交给 stage A，重做 decisions 3d

## 成功标准

263 个 shard 全部建成，每个 shard 自带 `meta.json`、`index.parquet` 和 10 个 float16 数组
（`vit_mean`、`vis_mean`、`L09/L18/L27/L36` 的 `_mean` 和 `_last`），
并且和 val 用的是同一条解码 / 预处理 / batch 路径——已经用逐位比较证明过。

## 时间预算（2026-09-22 00:20 更正，原估计错了一天多）

- 每个 shard 约 1574 帧 × 0.13 s ≈ 3.4 min；263 个 shard 合计约 **14.5 h** 纯 GPU 时间。这一条没变
- **原来写的"下载每 4.5 min 一个 shard、抽取追得上、次日 08:00 收工"是错的。** 错在直接信了 download
  的 status 行：那个 MB/s 和 ETA 是**累计平均**，被最初几小时的高速度拖住，不描述当前链路。
  实测瞬时速率一整晚在掉：19:19 是 6.3 MB/s，21:00 是 5.4，22:50 是 4.3（22:20-22:50 半小时
  371.7 → 379.5 GB）。status 行同一时刻报的是 9.0 MB/s、ETA 17.3 h
- 按 4.3 MB/s 算，剩下的 155 个 shard 约 550 GB，还要 **约 35 h**，也就是后天上午收工。
  用户决定不折腾更快的代理节点，就按这个数排期
- 所以**整个 run 是 download-bound**：一个 shard 到货要 11-13 min，抽它只要 3.4 min，
  抽取永远在等下载，**GPU 大部分时间是空的**，收工时间等于下载收工时间
- `INTERVAL` 从 300 改成 900：追不追得上不由它决定，但每个 pass 都会重建
  index.parquet / past.npy / future.npy / rater.parquet 四个文件，间隔拉长三倍就把这个
  churn 降到三分之一（blast radius 见 docs/waymo-e2e.md）
- 特征体积：47 KB/帧 × 41.4 万帧 ≈ **19.4 GB**

## 两个已修的 watcher 缺陷（2026-09-22）

| 缺陷 | 后果 | 修法 |
|---|---|---|
| `feature_status` 用 `f"{split}_*.tfrecord-*"` glob，train 的分片叫 `training_...`，匹配不上 | 整晚报 "0/0 shards on disk"；没有 split 总数，`built >= of` 的停止条件永远不成立，263 个 shard 抽完之后会**永远每隔 INTERVAL 重建一次 index** | 改用 `split_of()`，别再从 glob 里重造一遍这个映射 |
| stall 告警假定抽取是慢的一侧 | 实际是 download-bound，空闲是常态。固定的"60 min 没进展"迟早误报；而"有分片没建"这个条件在抽取跟得上时永远不成立——**下载死了看起来和健康空闲一模一样** | 阈值改成本次 run 实测 cadence 的 4 倍（不低于 `stall_s`），并把两种故障分开报：有货不抽 vs 没货可抽 |

## 结果

跑完再填。run dir: `$DATA_DIR/runs/waymo_e2e/features_train/20260921-165650/`
