# OT2-B 预登记：AlpaSim 输入标准下离轨行的剂量—反应（2026-10-09，任何新 checkpoint 的闭环分数读出之前写定并 push）

lane OT2，piece B，decision 210。接 d189（AP2 输入标准）、d198（离轨行，SH30 配方，一个剂量）、d201（700 scene：OT30 两 seed 对 SH30 +0.0156 [−0.0011, +0.0326]，基线只有一个 seed）、d202（零分约一半是直行段 1–2 m 横移后的侧碰 / 出路）。只有本地 run，不向 AlpaSim 注册、warm-up 或提交。

## 问题

1. AP2 的输入标准（冷启动 backwarp、AlpaSim 的 ego 定义与 route command 训练进去）叠加离轨行，700 scene 上的 mean scene score 是否比 AP2 自己高到够当比赛 driver。
2. 剂量—反应的形状：扰动幅度（±0.5 m → ±1.5 m）和 batch 占比（10% → 25%）各自是否继续有用，在哪里拐。
3. 基线自己的 seed 离散有多大（SH30-F-s1、AP2-AB-s1 的闭环此前没有）。

## 与旧条目的差别

- d132 / d143 / d146：on-policy rollout 采状态、纵向也闭环、WOD 片段，副作用是 HUGSIM 起步停滞。这里沿用 d198 的做法：静态扰动、只横向 + yaw、只取 v > 3 m/s 的 navtrain token、不做 rollout、不动纵向、目标是日志未来在扰动位姿下的重表达。
- d198：SH30 配方（NAVSIM 输入标准）、一个幅度（±0.5 m / ±2°）、一个占比（10%）、读 navhard。这里是 AlpaSim 输入标准、2 × 2 剂量、读 AlpaSim 闭环。d198 第 6 点的 m = 1 离线退步在 AP2 标准下是否还在，作为旁读。
- d201：OT30 对 SH30，基线一个 seed，线 +0.010。这里基线两个 seed，线 +0.015。
- 没有新机制要检验；不重测 d92（desire）、d119（no-launch）、d141（plane 引擎优于深度）。

## 臂

全部是 SH30 / AP2 配方（冻结 Cinque、P2 adapter、hinge λ 30 / 0.5 m、navtrain 12 shard、split `navsim/op-parity-full`、10 000 步 × 128），seed 0 / 1。

| 臂 | 训练代码 | 离轨行 |
|:--|:--|:--|
| `AP2-AB-s0`（已有）、`AP2-AB-s1`（新训） | `ap2_train.py --cold backwarp` | 无 |
| `APO-a05m10-s{0,1}` | `ap2_ot.py train --amp a05 --ot-mass 0.10` | ±0.5 m、±2°（d198 的 `ot1` 缓存原样复用），13 / 128 |
| `APO-a05m25-s{0,1}` | 同上，`--ot-mass 0.25` | 同上，32 / 128 |
| `APO-a15m10-s{0,1}` | `--amp a15 --ot-mass 0.10` | ±1.5 m、±5°（新缓存 `ot2`，漂移历史起点允许离日志路径 3 m），13 / 128 |
| `APO-a15m25-s{0,1}` | `--amp a15 --ot-mass 0.25` | 同上，32 / 128 |

离轨行在 AP2 标准下的构造：只作 m = 4（完整真实历史）的决策，即 rollout 漂出去之后所处的状态；ego 用 AlpaSim 定义（vy = 0、ay = 0、ax 录制值）；command 由该 token 的 AlpaSim route 重表达到扰动位姿后按官方 sample 的规则得出；从不作 anchor 行，从不作冷启动行（冷启动行保持不扰动，对应 d198 结尾提出的「把冷启动行排除出扰动」）。普通行与 `ap2_train.py` 逐行相同（`--ot-mass 0` 的 20 步对拍是训练前的闸门）。±1.5 m 对应 d202 测到的闭环横移 1–2 m；±5° 是 8 m/s 下 3 s 漂 1.5 m 所带朝向（约 3.6°）留出的余量。

决策时不用任何特权输入（GT map、GT agents、日志未来只作训练目标）；没有 WA-JEPA 权重或特征；训练只用 navtrain，不在 AlpaSim 里做 on-policy（公开 scene 是 navtest 的 log）。

## 闭环读数

- scene：d201 的固定 700 scene 列表（`c0b/lists/all.txt`，27 个 log），不随新落盘的 shard 改变。driver：`run.sh <dir> ap2`（APO / AP2）与 `sh30`（SH30-F-s1），不改 driver 代码；仿真确定，每个 checkpoint 跑一次。
- 新跑：`AP2-AB-s1`、8 个 `APO-*`、`SH30-F-s1`。已有：`SH30-F-s0`、`AP2-AB-s0`、`OT30-F-s0 / s1`。
- 每个 checkpoint：mean scene score、满分数、零分按类（at-fault 碰撞 / offroad / 出 corridor）、偏慢 scene（0 < score < 1）数、at-fault 事件数（`offroad_or_collision_at_fault` 之和）。
- 配方 = 两个 seed 的逐 scene 平均；计数取两个 seed 的平均。CI：逐 scene 配对差，按 log 整簇 bootstrap，10 000 次，seed 0。

## 线（写定，不事后改）

一个配置是**比赛候选**，当且仅当同时满足：

1. 两 seed 平均 mean scene score 减去 AP2 两 seed 平均 **≥ +0.015**，且该配对差的按 log 整簇 CI 下界 **> 0**；
2. at-fault 事件数（两 seed 平均）**不高于** AP2 的（两 seed 平均）；
3. 偏慢 scene 数（两 seed 平均）**不多于** AP2 的 + 10。

四个配置各自对线，不做多重比较校正，四个结果全部报告。多个过线时取平均分最高者为「最佳配置」；都不过线时照实写，并把平均分最高者记为「未过线的最佳」。

剂量—反应只描述、不设线：2 × 2 表（对 AP2 的差 + CI）、幅度主效应（±1.5 m 两格平均 − ±0.5 m 两格平均）、占比主效应、各格的零分分类与偏慢数。读法事先写明：主效应 CI 含 0 记为「没有分出来」，不记为「没有效应」。

基线 seed 离散：`AP2-AB-s0 − s1`、`SH30-F-s0 − s1` 的配对差与 CI，以及两个 seed 零分 / 非零分不一致的 scene 数。

## 护栏（只对最佳配置跑，定稿前）

HUGSIM 64（`spec_plan_smooth`，经 `jevdrive.bench`，每场景一次）对 AP2 两 seed：起步停滞数（seed 平均）不增加，且 HD ≥ −0.03（点估计）。d143 / d146 的起步副作用来自 on-policy 与纵向；这里没有这两项，但 ±1.5 m 的 plane 重投影畸变更大，护栏照跑。

## 旁读（无线）

- AlpaSim 标准离线读数（`ap2_offline.py`，navtest，m = 1 / 4，2 035 token 子集的 EPDMS，各配置 seed 0 对 AP2-AB-s0）：看 d198 的 m = 1 退步在 AP2 标准下是否还在。
- navtest / navhard（`python -m jevdrive.bench`，APO / AP2 以 NAVSIM 标准输入读，不是它们的训练标准，只看是否掉分）。
- 离轨探针（`ot_ladder.py`）在 ±0.5 m 与 ±1.5 m 留出行上的修回比例。

## 成本与分阶段（docs/long-runs.md）

- 实测参照（pool history，2026-10-09）：离轨 prep 每 shard 5–7 min、VRAM 29–42 GB（申报 44）、19 核、RAM 29 GB；AP2 全量训练 35.7 GB VRAM、RAM 96 GB（host token 走 page cache），OT30 训练 26.3 GB、5.7 it/s 约 30 min。APO 训练申报 48 GB VRAM / 110 GB RAM（离轨行的 hinge raster 也上卡），20 步 smoke 先量。闭环每 driver 700 scene 约 0.8 job-hour，单个 chunk 作业实测 VRAM 19–32 GB（申报 32）、RAM 28–41 GB（申报 42）。
- 估计（共卡 job-hour）：prep 12 × 0.12 ≈ 1.4；训练 9 × 0.6 ≈ 5.4；smoke 0.3；闭环 10 driver × 0.8 ≈ 8；离线与 bench 的 GPU 部分 ≈ 1.5；HUGSIM 护栏 4 checkpoint ≈ 2。合计约 **19 card-hours**，上限 30；超出就先砍 25% 占比的两格的 seed 1。RAM 是并发的约束（训练同时约 2 个）。
- 分阶段：训练 20 步对拍 + 带离轨行的 20 步 smoke → 全部训练；±1.5 m prep 先 1 个 shard，看样例帧并过符号闸门（OT30-F-s0 在 ±1.5 m 行上的 4 s 横向 / yaw 响应 ≥ 0.1）→ 其余 11 个；闭环沿用 d201 已验证的 chain（看门狗 20 min、缺 scene 补跑），每个 driver 3 个 chunk 作业，同时在池中的作业 ≤ 3。
- 磁盘：`ot2` 缓存约 18 GB；闭环只留零分 rollout 的 `.asl`；剩余空间 < 150 GB 立即停并报告。

## 限定（写在前面）

- 每个配方 2 个 seed；27 个 log，CI 约 ±0.03–0.04，+0.015 的线只有效应到 +0.03 左右才大概率过。
- 本地渲染，未与官方环境对分；公开 700 scene 不是 private 评测集。
- plane 引擎对路面以上物体的畸变随幅度变大，与「离轨状态本身」没有分开（d141、d198 同样的限定）。
- 离轨行只在 m = 4；冷启动三步若已漂出，训练里没有对应行。

## 修订 1（2026-10-09 12:50 box 时间；写于任何 APO / AP2-AB-s1 / SH30-F-s1 闭环分数读出之前）

lane M1 的 decision 205（`experiments/alpasim/results/m1_preturn_shift.md`）：AlpaSim 零分的主机制是强 hinge（λ 30 / 0.5 m）触发的闭环朝向漂移，只把 checkpoint 换成 λ 10 配方，同样 700 scene 上 P2H10-F-s0 0.9484 对 SH30-F-s0 0.9140。本表的基底（AP2 = 强 hinge）正是引起漂移的配方，所以按协调者指示裁剪，不重来：

- **保留并按原线判读**：已训完的 `APO-a05m10-s{0,1}`、`APO-a05m25-s{0,1}` 与基线第二个 seed（`AP2-AB-s1`、`SH30-F-s1`）的 700 scene 闭环。线 1–3 原样不动。P2H10-F-s0 / s1 作为 reference 行取自 M1 的表，不参与判线。
- **砍掉**：`APO-a15m10`、`APO-a15m25` 四个 λ 30 的 ±1.5 m 训练（未提交，不训）；因此 2 × 2 剂量表只剩 ±0.5 m 一行，幅度主效应没有读数，占比效应只有 ±0.5 m 下的 10% 对 25%。HUGSIM 护栏、navtest / navhard 与 AlpaSim 标准离线旁读不跑（强 hinge 配方不再是候选基底，护栏只为定稿候选而设）。
- **保留的数据**：±1.5 m 的 `ot2` 离轨 token 缓存 12 个 shard 跑完并留盘（与 hinge 无关），交给新 lane OT3 在 λ 10 配方上训练；±1.5 m 留出行上的 ladder 探针照跑（piece A）。
- 写本修订时已读的东西：piece C 的 ENS-OT30 与成员（d201 已有）的分数、训练日志的 dev ADE、`ot2` shard 2 上的符号闸门探针。APO、AP2-AB-s1、SH30-F-s1 的闭环分数一个都没有读。

