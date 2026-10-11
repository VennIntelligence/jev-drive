# PAI-KW：PAI driver 侧 keyframe + warp 开关（预注册，2026-10-11）

Lane PAI-KW。本文在任何闭环读数之前写完并 push。只动 serving 侧（`pai_core.slots` 加一个默认关闭的开关），不训练，不改 AlpaSim，
不提交任何 submission。

## 问题

`P2H10` 家族的 checkpoint 是在 warp 历史上训练的：2 Hz keyframe，其余 slot 由 ego-motion warp 合成（决策 142，全量协议定 warp）。
AlpaSim nuPlan track 的 driver 正是这样喂的。PAI track 推 10 Hz 真帧，`pai_core.slots` 把每个真帧直接放进 slot。离线读数
（[served_plan_length.md](../../body1/results/served_plan_length.md) 第 7 点，600 个 `lb_hq_navtestX` token，没有第二读者）：
warp 帧 arc / log 0.994、ADE 0.61 m，真 10 Hz 帧 0.895、1.35 m。这是 train / serve mismatch，plan 短约 10%。

两个问题：

1. 这个 mismatch 在 PAI 闭环里值多少分（每个臂 switch on − off 的绝对分变化）。
2. `P2H10S`（S）对 base 的配对差（决策 235 / 247）在换 frame source 之后是否不变。

**这个开关丢信息。** on 状态下每 10 个真帧只用 4 个时刻（t0 − 1.5 / 1.0 / 0.5 / 0 s）的帧当 keyframe，其余 6 个真帧不看，
8 个 slot 里 6 个是 warp 出来的。所以这是 mismatch test，不是 serving ceiling：on 的分数回答「把输入对齐到训练分布值多少」，
不回答「PAI 上这个家族最多能到多少」。真帧里有 warp 给不出的东西（其他 agent 在 0.5 s 内的运动）。

## 已定的东西（引用，不重测）

- 决策 142：全量协议定 warp；warp 帧就是 `P2H10*` 的训练输入。
- 决策 116：对 GIMM 协议的模型，真 10 Hz 历史只比 GIMM 高 +0.51 [−0.51, +1.54]，「历史帧质量不是瓶颈」。那是 open loop、
  另一种训练协议，不回答 warp 训练的 checkpoint 喂真帧的问题，这里不重测。
- LAT1（[lat1_frame_synthesis.md](../results/lat1_frame_synthesis.md)）：`lattice_gpu` / `warp_gpu` 与 CPU warp 逐像素一致
  （12.58 G 像素 0 个不同），nuPlan 闭环 126 scene 逐分一致。这里只验证 PAI 侧调用它得到的 slot 与同样四个 keyframe 下
  nuPlan driver / 训练管线的 slot 一致。
- 决策 226：PAI served 配置是 `JEV_VCONT=1.0 JEV_LEAD=1`，本实验两边都用它，不动。
- 决策 235 / 247：S − base 在 PAI 上为正（281 scene +0.0987 [+0.0594, +0.1394]），增益全部来自零分 scene 变有分。
  `s_b1a` 33 scene 上重训 seed 之间的差由 2 个 scene 的零分翻转构成（决策 247 第 5 点），这是本实验在 33 scene 上的噪声量级。
- served_plan_length.md 第 7 点的离线表：作为 check (iii) 的参照值，不当作已复核的结论。

## 臂

| 臂 | checkpoint | 开关 |
|---|---|---|
| base off / on | `P2H10-F-s0`、`P2H10-F-s1` | `PAI_KEYWARP` 不设 / `=1` |
| S off / on | `P2H10S-F-s0`、`P2H10S-F-s1` | 同上 |

开关：`PAI_KEYWARP=1` 时 `pai_core.slots` 走 `key_slots`：取 t0 − 1.5 / 1.0 / 0.5 / 0 s 的帧（容差 30 ms）当 keyframe，
调 `sh30_core.lattice_gpu`（nuPlan driver 的 `Core.plan` 调的同一个函数，不重写）。cold start 用 `sh30_core` 的规则
（最老 keyframe back-warp）。默认关，关闭时 `slots` 与改动前逐 bit 相同（单元测试 `tests/test_pai_keywarp.py`）。

## 协议

与决策 247 的 `s_b1a` 读数相同：Tokyo box，`pai_eval.sh`（每张 3090 一个 `pai_run.sh` stack），scene list
`/data/runs/alpasim/pai_full/s_b1a.tsv`（33 scene，一个 chunk = 一次 simulator launch），CONC 4，unharmonised renderer，
image `jev-alpasim:p2h10-f-s0-6f3d05d1`，每 scene 一次 rollout。on 臂只多一个 `-e PAI_KEYWARP=1`。代码从 Tokyo 上的一份冻结
拷贝跑（不动那边的共享 checkout）。

配对参照（off）：

- S off：已有 run `pai_full/runs/p2h10s-f-s0_b1a`、`pai2c/runs/s1_b1a_c4`，同一个 list、同一个 image；`pai_eval.sh` 重跑
  s0 曾 33 scene 全同（docs/tokyo-box.md）。复用。另外用新代码在开关关闭下重跑一次 S s0 off，当作 off 路径的闭环 identity
  （预期 33 scene 逐分相同）；不同则 S off 两个 seed 都用新代码重跑并报告。
- base off：已有的 base rollout 在别的 list 里（`full1_ab` 是 99 scene 一次 launch，base s1 是 `pai_base_s1` 的 chunk）。
  simulator 只对相同的 scene list 可复现，所以 base off 两个 seed 用 `s_b1a` list 重跑，并与旧值逐 scene 对比（顺带读出
  list 依赖有多大）。

**与任务书的偏差。** 任务书要求闭环 run 经 `jevdrive.bench` 提交到主 box 的 GPU pool。实际情况：`jevdrive.bench` 没有
AlpaSim / PAI 这个 bench（docs/alpasim.md：「`jevdrive.bench` never goes through these drivers」），已有的 `s_b1a` 协议和全部
off 参照都在 Tokyo box 上（Docker stack，没有 pool），主 box 上没有这 33 个 scene 的 usdz，主 box 与 Tokyo 之间 40 scene 里有
1 个 scene 翻转（pai2_box.md）。为了让 on / off 是同一台机器、同一个 stack、同一个 list 的配对，闭环 run 用已有 runner
`pai_eval.sh` 在 Tokyo 跑（不写新 runner）；离线 check (iii) 在主 box 上经 pool、在 `jevdrive.run.Run` 里跑。

## Equivalence checks（全量之前）

- (i) 录下来的 PAI stream（COL1 抽出的 driver 侧消息 `col1/x/a10/msgs/*.pkl`，经真实的 `pai_driver.Driver` replay）：每个
  decision 上开关关闭的 `slots` 与改动前的 `slots`（`git show ef2f829d:` 的原文件）输出逐 bit 相同。
- (ii) 同一 stream、开关打开：slot tensor 对同样四个 keyframe 经 `sh30_core.lattice`（CPU 参照路径，训练用的
  `op_interp.synth_cpu`）的结果，逐像素比；plan 对 `sh30_core.Core.plan`（nuPlan driver 的入口）在 warm decision 上比，按
  docs/long-runs.md 的 fp16 规则：只看 ≤ 4 s 的 8 个 pose，报超过 0.03 m 的行占比，并给一个对照（同一 decision 上 off 的 plan
  与 on 的 plan 的差）。主 box 上再比一次：on 的 plan 对 bench 的 warp 预测（`P2H10-F-s0-warp__base.npz`）。
- (iii) 离线 600 token（`lb_hq_navtestX`，与 served_plan_length 相同的抽样）：把真 10 Hz 帧拼成 stream 喂 `pai_core.slots`，
  off 应复现真帧行（约 0.895 / 1.35 m），on 应回到 warp 行（约 0.994 / 0.61 m）。判据：on 的 arc / log 在 0.994 ± 0.01、
  ADE 在 0.61 ± 0.03 m 之内。**不过则停**：开关没有做它声称的事，立即报告。

## 分阶段

1. 1 个 scene（`s_b1a` 第一个），base s0，on，`PAI_DUMP=1`。检查单：200 次 `drive` 全部有 inference、`inference_error` 0、
   记录里 `keywarp` 为真、warm 之后 `n_real` = 8、dump 的 slot 里 −1.0 s 与 0 s 两个 slot 等于真帧、其余是 warp；latency
   见下。不过则停。
2. 其余：7 个 run（base on × 2、base off × 2、S on × 2、S s0 off identity），每张卡一条顺序链，同一个 tag 的 run 不同时跑
   （compose project 名按 tag）。

**Latency。** PAI 每 0.1 s 一个 decision。读 `drive.jsonl` 的 `total_ms`（中位数、p95、p99）off 对 on，同卡同 CONC；on 的 p95
超过 100 ms 则报告为未过预算（不改代码去凑）。同时读 `ms.slots`（on 时含等卡的时间）与 `ms.encode`。

## 两个读数

单位是 scene（33），配对；CI 用 `jevdrive.stats.paired`（percentile bootstrap，B = 10 000，seed 0）。每个臂的 scene 分 =
两个 checkpoint seed 的逐 scene 均值；每个 seed 单独也报。除均值差外报 scene 计数：fixed（off 零分 → on 有分）、broken
（off 有分 → on 零分）、better / worse / within 0.01，以及三类零分（at-fault collision、offroad、left corridor）的计数。

**(a) 每个臂 on − off。**

| 结果 | 含义 |
|---|---|
| CI 不含 0，为正 | mismatch 在闭环里要分。开关成为 PAI serving 的候选（要在 281 scene 上确认）；「未启动的备选」一节的判断点被触发，由用户定 |
| CI 含 0 | 33 scene 上量不出 mismatch 的闭环代价。n = 33 的配对 CI 半宽预计 0.08 左右（决策 247 的 33 scene 读数），小于它的效应这里看不到；不据此说「没有代价」 |
| CI 不含 0，为负 | 真 10 Hz 帧带来的信息比 mismatch 的代价大。开关保持关闭；修 mismatch 的方向只剩「在真帧 slot 上训练」，不是丢帧 |

**(b) S − base 在 off 下、在 on 下，以及两者之差（difference of differences，逐 scene 配对）。**

| 结果 | 含义 |
|---|---|
| DiD 的 CI 含 0，且两边 S − base 同号 | S 的效果在 frame source 之间稳定（在 33 scene 的分辨率内）；决策 247 的配对差不需要 frame source 的 caveat 以外的修正 |
| DiD 的 CI 不含 0 | S 在 PAI 上的增益依赖 frame source；决策 235 / 247 的配对差要注明只在真帧 serving 下成立（或只在 warp serving 下成立） |

两个 checkpoint seed 是否一致单独报（每个 seed 的 on − off、S − base）。33 scene、每臂 2 seed，本实验是描述性读数，不晋级
任何东西；servable 换不换由用户定。

## 预算

- Tokyo：2 × RTX 3090。1 scene 约 4 分钟；7 个 run × 约 27 分钟（33 scene，67–79 scene / h / 卡）≈ 3.2 卡时，两卡约 1.8 h。
- 主 box：check (iii) 一个 pool job，`--vram 8 --cpu 8`，几分钟。闭环不占主 box 的卡。
- 不训练。不新增 scene 下载。

## 未启动的备选：`P2H10` 家族的 warp + 真帧混合训练

同一个 mismatch 的另一种修法，**本实验不启动、不搭建、不排队**，只记在这里。

- 做法：训练行里一部分 token 的 8 个 slot 用真 10 Hz 帧（需要 10 Hz sensor 的 navtrain 子集；现有的真帧缓存只有
  `lb_hq_navtestX` 这一份测试侧的），其余仍是 warp 帧，让同一个 checkpoint 在两种 frame source 下都在分布内。serving 时
  PAI 喂真帧（不丢信息），nuPlan 继续喂 warp。
- 代价：整个家族重训（base、S、B、各 drop-one 臂，每个 2–4 seed），要先建真帧的训练缓存，nuPlan / navtest / navhard 上的
  既有读数都要在新 checkpoint 上重读；与「base model stays」不冲突，但所有 paired 参照换一遍。
- 它比开关多给的东西：保留 6 / 10 的真帧。开关只回答「对齐值多少」，混合训练回答「对齐且不丢信息值多少」。
- **判断点 = 本实验读数 (a) 的绝对分变化。** (a) 为正且 CI 不含 0：mismatch 有闭环代价，混合训练值得由用户决定是否立项
  （开关已经是零成本的下界）。(a) 的 CI 含 0 或为负：没有闭环证据支持为这个 mismatch 重训整个家族，备选搁置。
- 谁定：用户。本 lane 到写下这一节为止。
