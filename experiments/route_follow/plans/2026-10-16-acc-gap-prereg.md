# 预登记：PAI ACC 上的跟车间距校准（serve 侧，不训练），配对已存 ACC 读数，2026-10-16

本文件在任何闭环读数之前写好并推送，之后只追加补记。诊断（全部来自已存日志，CPU）见
[results/acc_gap/diagnosis.md](../results/acc_gap/diagnosis.md)；ACC 本身见 [2026-10-16-pai-acc-prereg.md](2026-10-16-pai-acc-prereg.md)、
[d328](../../../research/decisions/328.md)。

## 0. 诊断要点（读数之前）

- ACC 下跟车时距不是 d315 的 1.84 倍：跟车决策 9 706 个（124 个场景），本车 3.02 s 对人同位置 2.28 s，逐决策比中位 1.10（lead 项胜出时 1.16），
  逐场景 1.11。d315 的 4.3 s 是旧 serve 上按「慢且跟车」挑出的 12 个场景，速度是 plan 定的。真实的差是 **4–6 m 的偏移**（稳态跟车，3–10 m/s 时距多 30–45 %）。
- lead MPC 对模型给的前车几乎精确地守着 openpilot 的跟车律（模型视角残差 +0.2–0.75 m）；多出来的距离来自 **lead 头的偏差**：运动前车距离读短约 1 m、
  速度读慢 0.8–1.1 m/s（x_obstacle = d + v_lead² / 5，速度偏差折合 0.32–0.45 v 米）。偏差在 lead 头自己的速度里，不在 radard 的自车速度项（相关 0.00）；
  静止前车没有速度偏差。低速（1–5 m/s）另多 2–3 m（MPC 在步行速度下几乎不追距离）。
- openpilot 默认值本身也比人长：train160 的 75 个场景日志（只读日志，不跑仿真），人停车间距 3.4 m，以此作停车距离时人的时距 1.13 s；openpilot 是 6 m + 1.45 s。
- 不是原因：概率滤波 / 阈值（跟车决策 91 % 有前车；漏检 662 个），MPC 状态（每会话一个 LeadPlanner，热启动，每 0.1 s 两个 tick，不按决策重置）。
- 追尾：带时刻的 31 次里 18 次在接触前 3 s 由 lead 项定速、前车在走廊里，本车间距是人同时刻的 1.2–2.5 倍，几乎都在 0.2–6 m/s（走走停停里停得远、起步晚）。

## 1. 臂（一个联合臂）

serve：`JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7`，其余与 ACC 读数相同（`JEV_TRACK` 不设 = v8 gate，转向权重 10，RSG-L18-s0）。
`JEV_GAP=<T_FOLLOW>,<STOP_DISTANCE>`（`serve_fix.py` (e)，`lead_long.LongMpc(t_follow, stop_distance)`）：只替换 lead MPC 期望距离
d(v) = v²/5 + T_FOLLOW v + STOP_DISTANCE 里的两个常数。不设时逐位相同（`acc_check.py` 检查 5：lead_long 默认参数对 d78c9f1b 3 000 个决策 0 差异，
检查 1–4 照旧 0 差异；代码在 main 786e0d91（被另一个 agent 的 `commit -am` 一并提交，内容完整）+ af3b88d8）。

**取值只用 train 场景，不看读数集**：目标是真实稳态间距 = train 上人的跟车律 3.4 m + 1.13 s × v。在 lead 头的 train 偏差下
（FIX1 读数，P2H10-F，28 个 train 场景：距离 −0.96 m，速度 −1.12 m/s），MPC 稳态时真实间距 = (T + 0.448) v + S + 0.71，
解得 T = 0.68 → **0.7 s**，S = 2.69 → **2.7 m**。用读数集上 RSG 的偏差（−1.13 m、−0.79 m/s，只作一致性对照）代入，预测真实间距约 3.7 m + 1.02 s × v，
仍接近人。静止前车没有速度偏差，停在静止车后的真实间距 ≈ 2.7 + 0.8 ≈ 3.5 m（人 3.4 m）。
**风险**：期望距离变短，接近静止车 / 慢车时开始制动更晚（10 m/s 时 d(v) 从 40.5 m 到 29.7 m）；lead 头把距离读短是偏保守的，速度读慢也偏保守，两者都保留。
阶段 1 的碰撞线管这一点。

## 2. 阶段 1：跟车子集（box，配对已存 ACC 读数）

- 场景（规则在读数之前定）：ACC 读数里 lead 项在 ≥ 80 个跟车决策（真实运动前车 ≤ 40 m、本车 ≥ 2 m/s、有效段）上胜出的 31 个场景，
  加 lon-clips-g10 页上 g10 的 6 个「无故停车」场景（3294623f、4f7cb072、92af888e、a68ad579、cbb22323、fce18b62），共 37 个：
  [results/acc_gap/stage1.txt](../results/acc_gap/stage1.txt)。已存 ACC 在这 37 个上：均分 0.872，零分 3（碰撞 1、出路 2），被追尾 9（g10 21）。
- 运行：`experiments/route_follow/scripts/rf_accgap.sh submit`：`pai_box.sh`，`CHUNK=19`（两个 chunk），`WIZ=controller.gains.rel_front_steering_angle_weight=10`，
  池优先级 14；同一 chunk 表再跑一遍不带 `JEV_GAP` 的 ACC（`s1acc`，同表噪声参照，**只描述，不进判据**）。gate job `--after` 这四个 chunk。
- 判据（对已存 ACC，两边都有分的场景；`rf_accgap.py lines --stage 1`，rf_pai281 的逐场景事实）：
  1. 被追尾（metrics `collision_rear`）个数 < ACC；
  2. at-fault 碰撞零分 ≤ ACC + 1；
  3. 零分 ≤ ACC + 1。
  全部成立 → 阶段 2；否则停，不跑全集。

## 3. 阶段 2：全 281（只有阶段 1 过才跑）

- gate job 自动提交：`CHUNKS_FROM=$DATA_DIR/runs/route_follow/pai_g10/acc/chunks`（与 ACC 读数同 8 个 chunk 表），同 serve、同 WIZ；报告 job `--after` 8 个 chunk。
- 判据（对已存 ACC，按场景配对 bootstrap 10 000 次，seed 0，ACC 预登记的四条）：
  1. 均分 CI 下界 > −0.01 且点估计 > 0；
  2. 考题 72 点估计 ≥ −0.01；
  3. 零分 ≤ ACC + 2；
  4. 交接 spin ≤ ACC + 1。
- 另报（描述）：被追尾、碰撞、新增 / 修好零分逐个、考题 / 对照 / stop17 子表（`rf_pai281.py --vs acc`）。

## 4. 成本与证据等级

阶段 1：4 个 chunk job（约 19 场景各约 30 分钟），阶段 2：8 个（约 52 分钟各，box 1–1.5 小时墙钟）。1 seed、按场景配对；阶段 1 与已存读数的 chunk 表不同，
跨表噪声由 s1acc 描述。这是当前唯一的 PAI 读数。若全过，它是下一次 PAI 提交的候选（需要新镜像，本 lane 不建）。

## 补记 1（读数之后，2026-10-16 晚）

- 阶段 1 PASS（[results/acc_gap/stage1.md](../results/acc_gap/stage1.md)）：被追尾 9 → 2，碰撞 1 对 1，零分 4 对 3；同表 ACC 重跑与已存读数 37 / 37 完全相同。gate 自动提交了阶段 2。
- 阶段 2 PASS（[results/acc_gap/stage2.md](../results/acc_gap/stage2.md)、`stage2_table.md`）：+0.017 [−0.008, +0.044]，考题 +0.005，零分 75 对 75，spin 0 对 0；被追尾 40 → 28，碰撞 9 对 9。
- 决定 [330](../../../research/decisions/330.md)。
