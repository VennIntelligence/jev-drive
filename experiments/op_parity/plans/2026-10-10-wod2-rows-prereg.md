# WOD2 预登记：把 nuPlan 侧之后做出的训练成分搬到 WOD 本域训练，WLG 之上还能不能涨 val RFS

2026-10-10，lane WOD2。在任何新臂训练之前提交。结果：`results/wod2_rows.md`，表 `results/wod2_rows/`。
基线：WLG（WOD r2-train 上的 P2 配方 + `--stop-gate 0.5`，val RFS 8.187，test 8.099；决策 169、180）。

## 1. WOD-E2E 给了什么标签（已核，训练前）

box 上 `datasets/waymo_e2e/front3` 的 slim shard 只删了另外 5 路相机，其余字段原样。抽查 train 3 个 shard × 150 帧、val 3 个 shard × 150 帧，
`E2EDFrame` 实际填了的字段只有：

| 字段 | train | val |
|---|---|---|
| `frame.images`（front3 JPEG + 每图 pose）、`context.camera_calibrations` | 有 | 有 |
| `past_states`（4 s，4 Hz，pos / vel / accel）、`future_states`（5 s，4 Hz，pos）、`intent` | 有 | 有 |
| `preference_trajectories`（3 条 rater 轨迹 + 分数） | 无 | 每序列 1 帧（479 帧） |
| `laser_labels` / `camera_labels` / `projected_lidar_labels`（agent 框） | 0 / 450 | 0 / 450 |
| `map_features` / `map_pose_offset`（道路、车道几何） | 0 / 450 | 0 / 450 |
| `lasers`、`frame.pose`、`context.stats` | 0 / 450 | 0 / 450 |

即：只有 ego 轨迹、intent、相机和 val 的 rater 分数。没有 agent 框，没有 road / lane 几何，没有可行驶区域（与决策 93「WOD 无地图」、决策 163「WOD 无可行驶标签」一致）。

## 2. 候选成分的可行性

| 成分 | 需要的标签 | WOD 上 | 处理 |
|---|---|---|---|
| BODY1 包（own-plan agent hinge、hinge-only 离轨行、road-and-lane 栅格；决策 228–234） | agent 框 + road / lane 几何 | 都没有；navtrain 训的 contact head 跨域 agent AUC 0.589（决策 231，CARLA），WOD 上也没有真值可验 | 不可行，不做伪标签 |
| drivable SDF hinge（P2H10 / SH30）、几何 tokenizer、path-req | 可行驶区域 / 地图 | 没有 | 不可行 |
| route polyline（op_route_cmd，522 k 帧已标） | hindsight 折线来自本帧之后的 logged ego 路径 | test 的 future 被隐藏、榜上只给 intent，test 帧拿不到这条输入；在 val 上喂它等于泄漏 logged 路径 | 不能作为榜上配方，不做 |
| 速度剖面（决策 218 补记、221） | - | 已定：WOD 训练的 WLG 上该机制没有可测空间 | 不重测 |
| navtrain + WOD 混合（决策 174）、偏好微调（171）、选择 head（175）、selector 迁移（195） | - | 已定 | 不重测 |
| **离轨重投影行**（决策 198 / 210，`ot1` 配方） | 图像 + 相机标定 + ego 历史 / 未来 | 都有；目标是 logged future 在扰动系里的表达，不需要地图 | **可行，臂 WOT** |
| **yaw-rate rows**（决策 213，`yr1` 配方） | 同上 | 都有 | **可行，臂 WYR** |

所以可行的只有两种「重投影行」。决策 217 量的是 navtrain 训练的这两种行零样本搬到 WOD（都无可测变化），不是在 WOD 上训练它们；本 lane 量后者。

**先验（写在前面）**：两臂的机制都是闭环里才显形的（离轨后的横向修回、yaw rate 续接的阻尼），WOD val 是 on-track 的开环帧，RFS 的缺口约一半是纵向决策
（决策 164），所以先验是「无可测变化」。本实验是为了把这条先验换成一次本域训练的读数；负结果即「WOD 上 nuPlan 侧的训练成分没有便宜的可搬项」。

## 3. 臂（最多 2 个）

两臂都是 WLG 配方原样（`pp_train.py --arm P2 --host --split wod/r2 --stop-gate 0.5`，`wod_r2` cache，无 hinge，anchor 行不变）+ 每个 batch 约 10% 来自重投影行：

- **WOT**：`ot1` 的扰动（终点横向 dy ~ U(±0.5 m)、航向 dψ ~ U(±2°)，航向误差从 −1.6 s 线性爬到 dψ，横向误差按 v·ψ 积分；v > 3 m/s；每行一次扰动）。
- **WYR**：`yr1` 的扰动（三段独立航向增量 N(0, 1°) 截 2.5°，40% 为 `recent` 行；v > 3 m/s）。

两臂共同的 WOD 化（与 navtrain 版的差别，全部写明）：
- WOD 的 9 个 policy slot 都是真实 10 Hz 帧对（f − 2，f），所以一个重投影行是 10 个真实帧（f − 18 … f，步 2）各自用 plane engine
  （`jevdrive.op_interp.warp_frame`）warp 一次到 `logged pose(t) ∘ (0, y(t), ψ(t))`，再过冻结的 Cinque encoder。扰动 profile 在 WOD 的 slot 时刻上取值。
- ego 输入：历史位姿在扰动后的 t0 系里重表达，速度 / 加速度 / intent 原样，经 `pp_wod.wod_ego`（评测用的同一映射）。
- 目标：logged future（8 个 pose）在扰动后的 t0 系里重表达。没有 hinge 项（WOD 没有可行驶标签；navtrain 版的 `off_hinge` 不搬）。
- 重投影行永不作 anchor 行；非 plan head 向这些行上的 shipped Cinque 蒸馏。
- 行池：`wod/r2-train` 里 v > 3 m/s、有 logged future 的行的一个随机子集（seed 固定，≥ 24 k 行，在 pilot 训练前定下、两臂各自一份、pilot 与 full 共用）；
  `wod/r2-dev` 上另取约 2 k 行只用于机制探针。

## 4. 分阶段与闸门

**S1 可行性（不训练）。** WOD 版 prep 的 `--zero`（偏移为 0）必须复现 `wod_r2` 里同一行已存的 token：相对平均误差 < 1e-3，fut / ego 最大差 < 1e-4。
不过就修到过为止（修 bug 不算补记）；修不过则报告「WOD 上重投影行的管线做不出来」并停。同时量 prep 吞吐，给出行池与全量的 card-hour 估计。
探针方向检查：shipped / WLG 在 r2-dev 扰动行上，plan 的横向响应对理想修正的斜率符号为正（读了画面）。

**S2 pilot（1 seed，3 000 步 × batch 128，`wod_r2` 全行）。** 三个 run：`WLGp`（同步数的 WLG 配方，配对基线）、`WOTp`、`WYRp`。闸门按臂各自判：

- G-mech（成分学进去了没有，r2-dev 扰动行，留出）：
  - WOT：4 s 横向修回比例（决策 198 / 209 的口径）≥ 0.50（navtrain：0.21 → 0.64；WLGp 基线同表报）。
  - WYR：对注入 yaw rate 的续接 α ≤ 0.60（navtrain：1.03 → 0.30），且未扰动转弯行上正当续接斜率对 WLGp 的差在 ±0.10 以内。
- G-harm：r2-dev 未扰动行 ADE 对 WLGp 不差于 +5%。
- G-rfs（val 479 rater 帧，cluster-mean RFS，按 sequence 配对 bootstrap，B 4000）：臂 − WLGp 的点估计 ≥ +0.02 才进 full。

判读：G-mech 或 G-harm 不过 = 这个臂在 pilot 结束（可用一次补记改行占比 / 步数）；G-mech 过而 G-rfs 点估计 < +0.02 = 「成分学进去了，RFS 平」，该臂结束，
不跑 full；点估计 < −0.10 且 CI 不含 0 = 「伤」。两臂都停则整个 lane 在 pilot 结束。

**S3 full（过闸门的臂，2 seed，10 000 步 × batch 128，与 WLG-full 同配置）。** 对已存的 WLG-full-s0 / s1 预测配对（G0：经同一 report 脚本复现 WLG 8.187 ± 0.002、
shipped 8.005，否则停）。full 前按 S1 / S2 实测时间估 card-hour；单臂超过 3 h 先 profile。

## 5. 指标与判定线（val RFS）

- 主指标：WOD-E2E val RFS，479 rater 帧，cluster-mean（`jevdrive.waymo.rater_feedback_score` / `rfs_by_cluster`），既有 harness
  （`pp_hugsim.py onnx` + `wod_launch.py gbias` + `scripts/wod_zeroshot_openpilot.py --set rater extra` + `wod_slot.py report`），不写新 runner。
  臂 = 两个 seed 逐帧分数的均值；对比 = 臂 − WLG，按 sequence 配对 percentile bootstrap，B 4000。次要：ADE@3s / @5s（1 437 帧）。
- 判定（每臂对 WLG，full）：
  - 「涨」：d RFS > 0 且 95% CI 不含 0。
  - 「伤」：d RFS < 0 且 95% CI 不含 0。
  - 「平」：|d| < 0.05 且 CI 落在 ±0.10 之内。
  - 「分不出」：其余（报 CI 半宽，说明读数能分辨多大的量）。
- 两个对比、不做多重校正：只有一臂贴边过线记「弱候选」，不记「涨」。
- 分辨力：val 对 shipped 的 CI 半宽约 0.17；两个 WOD 训练配方之间的配对 CI 更窄（WLG − WP2 半宽约 0.10），报告里按实际 CI 写清能否分辨。
- 落点（描述性，不贴标签）：10 个 scenario cluster；速度层（standstill < 0.5 m/s / 0.5–3 / 3–12 / > 12 m/s）；intent（直行 / 转弯）；
  WLG 逐帧 RFS 分层（floor 4.0 / 4–7 / 7–9 / > 9）；昼 / 夜。

## 6. 负结果结束什么

两臂都「平」或在 pilot 停：nuPlan 侧自 WLG 之后做出的训练成分里，WOD 的标签能支持的只有重投影行，而它们在本域训练也不动 val RFS；
WOD 档保持 WLG，「WOD 没有便宜的训练成分可加」写成结论，理由 = 标签（无 agent / 道路几何）+ 这次读数。不再在 WOD 上试重投影行的变体（占比、幅度）。
任何一臂「涨」：成为 WOD 档候选，是否提交 test 由 main 决定（本 lane 不提交）。

## 7. 预算与约束

约 40 card-hour、2 天；本设计预计 prep 2 × (行池 + dev) + pilot 3 run + full ≤ 4 run + 评测，远低于预算，S1 后给实测估计。所有 GPU job 经 GPU pool
（`python -m jevdrive.cl submit`）；训练在 `jevdrive.run.Run` 内，split 取自 `jevdrive.data.splits`（`wod/r2-train`、`wod/r2-dev`、`wod/val`）。
不提交 test，数据不出 box。一次尝试 + 至多一次事先声明的补记。
