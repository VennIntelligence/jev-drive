# op_parity / turn-prior 预登记：floor head 在转弯意图帧上到底做了什么，是否叠加在 WLG / 偏好微调上，能否写成固定规则或训练 recipe

2026-10-08，任何新打分之前提交。接第 168 / 169 / 171 / 173 条。执行：`scripts/turn_prior.py`，结果 `results/turn_prior.md`、`results/turn_prior/`、
`figs/turn_prior/`。候选集（F20 = WP2 路径族 5 × 速度 4）、重定时、RFS、bootstrap 全部复用 `s2_gohold.py` / `wod_launch_report.Ctx`
（B = 4 000，按序列配对，cluster-mean RFS，臂 = 两个 seed 逐帧平均）。不向 WOD test 提交；oracle 只作上限。

## 已有的读数（不算新打分）

Step 1 只描述第 173 条已经打过分的 floor head（`E` 臂：ego + 指令 + plan 描述，ridge，按序列 5 折 × 10 次重复）的选择：重跑同一套 out-of-fold 流程，
把每帧每次重复的 pick 记下来，按 v0 档（< 0.5 / 0.5-3 / > 3 m/s）× 左 / 右 × 候选（速度档、路径）汇总，并给每个 pick 的实际 5 s 位移比（pick / plan）与相对
plan 的 5 s 横向偏移。同时报 WLG 已经修掉的部分：静止起步转弯（v0 < 0.5）上 floor head 的增益是否还在（对 WP2 与对 WLG）。纯描述，没有新对比，不做假设检验。

## Step 2：可加性（对 WLG 的 F20，token 路径）

- 系统一 = WLG token path 的 plan（`results/wod_pref/plans.npz` 的 `wlg_s0 / wlg_s1`，harness 差 0.011 m）。F20 候选、候选 RFS、plan 描述全部在这组 plan 上重算；
  ego 流不变。
- **2a** floor head（`E` 臂，超参网格与内部折选法与第 173 条逐字相同：k 无（无视觉流）、λ ∈ LAMS 由训练折内 4 折选），按序列 5 折 × 10 次重复，out-of-fold。
  对比：`WLG+floor − WLG`、`WLG+floor − shipped`（shipped = harness 逐帧 RFS）。置换对照：输入行在帧间置换（两个 seed 同一置换），200 次，
  报零分布与实际值的 p。分层：全部 / turn-intent / straight / v0 三档 / 10 个 cluster。
- **2b** 叠在偏好微调的 out-of-fold plan 上（`top`、`f20` 两个目标，`plans.npz` 的 `top_s*`、`f20_s*`）。打分帧所在的 pref 折 k 的 plan 来自没见过折 k 标签的
  模型；为保证「任何被打分的帧都没有被任一阶段拟合或选择」，floor head 用 **WLG plan 的候选**（没有任何偏好标签进入）在折 k 之外的 4 个 pref 折上拟合
  （λ 由这 4 折内的内部 4 折选），再作用在折 k 的 **pref plan 的候选**上。对比：`pref+floor − pref`、`pref+floor − WLG`、`pref+floor − shipped`。
  敏感性（不进判据，标注「不是严格嵌套」）：head 直接在折 k 之外各帧的 pref plan 候选上拟合（那些 plan 的生成模型见过折 k 的标签）。
- 「delivers」的判据（对比对 WLG，全部配对 CI）：对 WLG 的 CI 下界 > 0；没有分层（turn / straight、v0 三档、cluster）的 CI 整体 < 0；置换对照 p < 0.05。
  「additive」另要求叠加后对其底座（WLG 或 pref 臂）的 CI 下界 > 0。

## Step 3：规则与训练 recipe

只有当 Step 1 能用一句话写出规则才做。规则必须 label-free（输入只有指令、v0、plan 本身），参数只能来自 WOD train 日志（`wod/r2-train` 的 turn-intent 行）或 out-of-fold
拟合，不得取自被打分的折。具体规则与参数来源在 Step 1 读完后以补记 A 提交（在打分之前）。

- 后置规则（3a）：对 WLG plan 直接用固定规则；对比 `WLG+rule − WLG`、`− shipped`，同 Step 2 的分层与判据。
- 训练 recipe（3b）：只有规则能表达为 System-1 训练信号（如 turn-intent 行的速度先验 / 目标重定时）时才做；`pp_train.py --stop-gate 0.5`，WLG recipe 上改动一处，
  2 seed，经决策 155 的 harness 服务，对比 `new arm − WLG`、`− shipped`。预算 ≈ 25 card-min / seed。
- 判据同上；另报与 floor head（2a）的相对大小（占 2a 增益比例）。

## 总预算与停止

约 3 card-hour，绝大部分是 CPU。Step 2 的 2a 对 WLG 的 CI 含 0 且点估计 < +0.05 → 「floor 的增益被 WLG 吃掉」，Step 3 仍可做（规则可能在 WLG 上给不同的数）
但不得写成 additive。四个以上对比被选出一个贴边 CI 时按「弱」标注。

## 补记 A（Step 1 读完后、任何规则打分之前提交）

Step 1 的描述读数（已打分的 floor head 的选择，见 `results/turn_prior/desc_picks.csv`）：WP2 上转弯意图帧 52 个，head 选的速度档 follow / hold / creep = 29 / 50 / 20 %，
5 s 位移是 plan 的 0.44（中位数），路径以向转弯外侧偏移为主（外移 3.2 m 的均值），v0 < 0.5 档的增益（+1.80）在 WLG 上缩到 +0.28，0.5-3 档（+0.80 -> +0.84）和 >= 3 档
（+0.95 -> +1.18）不变。规则的形式（速度缩放 + 向外侧偏移）来自这批已打分的帧，这一点在报告里按「形式事后、参数 out-of-fold」标注。

**规则族 R(alpha, nudge, v-range)**：只对 command 为左 / 右的帧、且 v0 落在 v-range 内的帧改写 plan：同一条路径，弧长剖面乘 alpha（`offset_path(plan, v0, alpha * arc(plan), dy, l0, lt)`），
dy = 0 或 1.2 m 向转弯外侧（即 F20 的 nudge 路径，`l0 = 10, lt = 2`）。输入只有 command、v0、plan 本身，label-free。
网格：alpha in {0.5, 0.7, 0.85, 1.0}，dy in {0, 1.2 m 外侧}，v-range in {[0.5, 3), [0.5, inf), [0, inf)} m/s，共 24 个（含恒等）。其余帧不动。

- **3a-oof（主）**：参数由训练折（wod_pref 的 5 个外层折，打分折之外的 4 折）上的转弯意图帧的 cluster-mean RFS 最大化选出，平局取最接近恒等者；作用在打分折。
  系统一 = WLG plan（对比 `WLG+rule − WLG`、`− shipped`）；叠 pref：参数在 WLG plan 上拟合，作用在打分折的 pref plan（与 2b 同一嵌套）。
- **3a-log**：参数只来自 WOD 日志：alpha = `r2-dev` 转弯意图行上（0.5 <= v0 < 3）日志 5 s 位移 / plan 5 s 位移 的中位数，dy = 0，v-range = [0.5, 3)。
  这是「由训练日志导出的速度先验」的字面版本；预期 ~1（plan 在模仿日志），按实际数报。
- 判据同 Step 2：对 WLG 的 CI 下界 > 0；无分层整体 < 0。另报 5 折各自选出的参数（是否稳定）。
- **3b 训练臂的触发条件**：只有当 3a-oof 的 5 个折选出的 (alpha, dy, v-range) 完全一致且 dy = 0（则单个训练臂等价于对每个折都嵌套），才训练一个臂：
  WOD train 上 turn-intent 且 v0 在该 v-range 的行，监督轨迹沿原路径把弧长乘 alpha（label 重定时），其余同 WLG recipe（`--stop-gate 0.5`），2 seed，
  经决策 155 的 harness 服务；对比 `new − WLG`、`− shipped`。条件不满足则 3b 不做并写明原因，不用不一致的 alpha 训练。
