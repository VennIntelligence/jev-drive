# op_parity WOD 档预登记：P2 输入对等配方在 WOD train 上训，WOD val 读（2026-10-07，任何 WOD 训练臂打分之前写定）

## 问题

第 155 条：P2H（navtrain 上训的 P2 + drivable hinge）在 WOD val 上 RFS 7.708 对 shipped 8.005（−0.30 [−0.50, −0.09]），ADE@3s +0.28 m；bias 置零时权重本身中性，损失来自新输入通道（WOD 加速度离散度只有 NAVSIM 的 1/3–1/7）。WOD 是单独一档。

**问的是**：用 WOD 自己的输入（intent → 命令，past_states → ego 状态与 4 位姿历史，映射与 `results/wod_p2h.md` 完全相同）和 WOD train 数据做同一个 P2 配方，WOD val RFS 能否超过 shipped，超多少；并把「在 WOD 上微调」与「WOD 输入」分开。

## 不重测的既有结论

第 155 条（navtrain 训的 P2H 在 WOD 上的读数）直接引用其存档预测，不重跑。第 131 条（shipped 的夜间差距）只作读数口径。

## 数据（prep 已跑，未读任何训练臂分数）

- **token 来源**：`processed/op_adapt/wodtrain`（op_adapt round 1 的缓存，`experiments/op_adapt_r1/lib/op_adapt_cache.py wodtrain`）：shipped Cinque 冻结视觉的 stage-3 输出 `permute_73`，帧由 WOD harness 同一渲染器（`wod_zeroshot_openpilot._maps / _pack`，FRONT / FRONT_LEFT / FRONT_RIGHT → road + wide，真实相机高度）生成，覆盖 WOD train 2 037 个序列的**全部偶数帧**，2 103 条连续流（帧号步长恒为 2），207 678 个 slot；图像对 = (f − 2, f)，即 0.2 s。`wod_parity.py prep` 只补 stage 4 → `view_39`（32 × 512），不重新渲染。
- **帧协议**：与 harness 一致——真实 10 Hz 帧、0.2 s 图像对、policy 9 个 slot 间隔 0.2 s（f − 16 … f）**全部是真实帧**（harness 喂 10 s 帧，9 个 slot 都满；navtrain 配方是 8 个 + 1 个零 slot，这里按 WOD 服务条件改成 9 个真实 slot，属配方偏离，写明）。
- **行**：某条流的 slot j ≥ 9（9 个 slot 和最老 slot 的前一帧都是真实帧），该帧有 5 s 日志未来，序列属于 `wod/r2-train` 或 `wod/r2-dev`。共 188 883 行：r2-train 179 405 行（1 935 序列），r2-dev 9 478 行（102 序列）；intent 直行 159 250 / 左 15 812 / 右 13 821 / UNKNOWN 0。
- **切分**（`jevdrive.data.splits`，按序列）：训练 `wod/r2-train`，训练中 dev 监控 `wod/r2-dev`，打分 `wod/val`（479 序列，官方 val 分片）；prep 里 `check_disjoint(r2-train, r2-dev, val)` 通过。pp_train 对 unit = sequence 的切分按 `tab["log"]`（序列）选行。
- **输入映射**（训练与评测同一函数 `pp_wod.wod_ego`）：命令 GO_LEFT / GO_STRAIGHT / GO_RIGHT → [左, 直, 右] one-hot；4 位姿 = past_states 第 9 / 11 / 13 / 15 步（−1.5 / −1.0 / −0.5 / 0 s），航向用前后一步的弦向（< 0.1 m 时沿用后一个 key），t0 航向 0；vx = 由位置推的 t0 速度，vy = 0；ax、ay = 最后一个 past state 的给定值；present = 1。
- **目标**：future_states 只有 4 Hz 位置（5 s × 20 步）。8 个目标位姿取 0.5、1.0 … 4.0 s（future 第 2、4 … 16 步，格点上精确，无插值）的 x、y；航向 = 该点前后 0.25 s 两点的弦向（rear axle 无侧滑，弦向即车头方向），弦 < 0.1 m 时沿用前一个 key 的航向（t0 为 0）。相机位置 = 该序列 FRONT 相机外参平移（op_calib；pp_train 的 `rear()` 只用 x，y 偏移的影响是 dy·(1 − cos ψ)，可忽略）。
- **教师**：shipped Cinque（port fp16）在同样 9 个 slot 上的输出（锚行与非 plan 头蒸馏目标）。
- **pilot 子集**：与 navtrain pilot 同尺寸，r2-train 行中均匀抽 4 874、r2-dev 行中抽 526（seed 0），`cache/wod_pilot`（ticks 共享）。

## 臂

| 臂 | 训练 | 新输入 | 说明 |
|:--|:--|:--|:--|
| shipped | — | — | 存档 harness 预测 `preds/op_cinque`（RFS 8.005） |
| P2H | navtrain 全量（第 155 条） | ego + 位姿 + 命令 | 存档预测 P2H10-F-s0 / s1，参照 |
| **WP2** | WOD r2-train | ego + 4 位姿 + 命令（P2 adapter） | 主臂 |
| **WP1** | WOD r2-train，同行序同步数 | 全部置零（无 adapter） | 「只是在 WOD 上微调」对照 |

配方 = op_parity P2（`pp_train.py --arm P2 / P1`）：视觉冻结，off-policy plan 通路可训，adapter 零初始化；0.75 模仿行（plan → 8 位姿，非 plan 头蒸馏回 shipped），0.25 锚行（新输入置零，plan 一致性 + 全部头蒸馏回 shipped，**锚在同一批 WOD 帧上**）；λ_i 1、λ_c 3、λ_d 30；lr 3e-5 / 3e-4，wd 0.01，cosine，clip 1。**不加 hinge**（P2H 的 drivable hinge 需要 NAVSIM 地图标签，WOD 没有；本线是 P2 不是 P2H，与 P2H 的对比混了 hinge 一项，写明）。

- pilot：WP2-pilot-s0、WP1-pilot-s0，600 步、batch 64、warmup 100（navtrain pilot 配方）。
- 全量：WP2-full-s0 / s1、WP1-full-s0 / s1，r2-train 全部行，10 000 步、batch 128、warmup 300、eval 每 1 000 步（P2-F 配方；约 7 epoch）。

评测：`pp_hugsim.py onnx`（训练权重 + `intent_bias` 输入的 ONNX）→ `pp_wod.py bias`（WOD 输入经该臂 adapter → bias；WP1 无 adapter → 零 bias）→ `scripts/wod_zeroshot_openpilot.py --set rater extra --onnx --bias`（与第 155 条完全同一 harness、同一接口、同一帧集）。

## 闸门

- **G0 等价（训练前）**：同一 token 路径在 WOD val（`processed/op_adapt/wodval`，同渲染器同协议）上用 port 算 shipped 的 plan，对存档 harness 预测 `preds/op_cinque` 在 slot ≥ 9 的 rater + extra 帧上比：plan xy 差均值 < 0.05 m。不过 → 停，查 token 路径。
- **G1 pilot 早停**（读 WP2-pilot-s0 vs shipped，479 rater 帧）：d RFS ≥ −0.10（约 2 倍 RFS 噪声 0.05）→ 进全量；< −0.10 → 停、报告（WOD 数据在 pilot 尺度上连 shipped 都追不上，算清晰负结果）。WP1-pilot 同时读出但不进闸门。

## 读数（全量；pilot 同口径只报不判）

WOD val，序列 bootstrap（B 4 000）配对：

1. **主读数**：RFS（479 rater 帧，按 cluster 平均），WP2 两 seed 逐帧平均 − shipped。
2. ADE@3s / @5s（1 437 个有未来的帧），同上。
3. 次读数：WP2 − WP1（WOD 输入本身的作用）、WP1 − shipped（WOD 微调本身的作用）、WP2 − P2H（WOD 训练 vs navtrain 训练，混 hinge）；seed 差 s0 − s1（噪声参照）；计划 5 s 路程 / 日志路程中位数。
4. **夜 / 昼**（`leaderboard_audit/results/night_gap/seq_lum.csv` 亮度标签：夜 < 50、昼 ≥ 120，黄昏剔除；rater 帧夜 133 / 昼 325）：各臂夜、昼 RFS 与 ADE；夜间差距变化 dd = (WP2 − shipped)_夜 − (WP2 − shipped)_昼，同一套 bootstrap 抽样。

## 判读（写在结果之前）

- **WP2 超过 shipped**：d RFS 的 CI 下界 > 0；**持平**：CI 含 0 且 |d| < 0.10；**更差**：CI 上界 < 0。
- **输入有用（在 WOD 上）**：WP2 − WP1 的 CI 下界 > 0；若 WP1 ≈ WP2 > shipped，增益来自 WOD 微调而非输入；若 WP1 > shipped 而 WP2 ≤ WP1，WOD 输入即使在 WOD 上训也不帮忙（与第 155 条「损失来自输入通道」同向）。
- **缩小夜间差距**：dd 的 CI 下界 > 0；否则不下夜间结论（夜间 rater 帧只 133 个，第 155 条 CI 半宽 0.5）。
- 只看到 ADE 变好而 RFS 不变：说明在拟合日志位置，RFS 的信任域没受影响（第 155 条里加速度探针就是这种模式）。

## 预算

prep ~15 min GPU（已跑）；pilot 训练 2 × ~2 min + harness 2 × ~15 min；全量训练 4 × ~25 min + harness 4 × ~15 min；合计约 3 GPU·h，3 卡并行约 1.5 h。

## 限定（预先写明）

开环；P2 无 hinge（与 P2H 对比混 hinge）；9 个真实 slot（navtrain 配方 8 + 零）；r2-dev 只用于训练监控，选择不看 val；WOD train 行在序列内 0.2 s 相邻、高度相关，有效样本量远小于行数。
