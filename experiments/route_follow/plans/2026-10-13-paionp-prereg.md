# PAIONP 预登记：在策略自己的 PAI 状态上做模仿（2026-10-13 夜）

**问题** P2 分支上限（[results/p2_branch/](../results/p2_branch/README.md)）：把人的入弯状态、人的速度交给 RSG，12 个 PAI 转弯只完成 4 个，
8 个从每个交接点都以同样方式失败：闭环里它不会「怎么转」。P1 开环（[research/route-follow/p1-openloop.md](../../../research/route-follow/p1-openloop.md)）：
在人的状态上 plan 会减速、会按时转，所以失败出现在策略自己状态上的连续决策里。loss 审计（[loss-audit.md](../../../research/route-follow/loss-audit.md)）：
头能拟合人的转弯，但 on-policy 行上只有 hinge，没有任何一项在策略自己的状态上把人的转弯排第一。还没测过的一臂：**在策略自己的 PAI 状态上模仿人**。
另一个 agent 正在测 RSGmixPAIT（S 行 self + PAI 人类日志回放行 both），本臂不重复它：这里的 PAI 行是 RB0-F-s0 自己开出来的状态。

**臂** `RSGPAIonp-L18-s{0,1}`：`--init RB0-F-s0`，RSG-L18 的温和配方（2 500 步、lr × 0.1、`--on-k 13` 等，与 [rf_tschool_mixed.sh](../scripts/rf_tschool_mixed.sh) 的 `REC` 逐字相同），
池 `on1_RSGPAIonp` = `on1_L18RB0`（mode `self`，即 RSG 的行）+ `on1_PAIonp`（mode `both`），`--on-mode mixed`。
`on1_PAIonp` = `on1_PAIRB0`（RB0-F-s0 在 PAI train160 上自己的 rollout，22 695 行，[rf_pai_pool.py](../scripts/rf_pai_pool.py)）中可用的行：
`fut` = onp_rows 的 `imit`（时刻 t0 的人类日志：从自车在日志路径上的投影出发、按日志自己在 t0 + 0.5k s 的推进量取点，横向偏差 e0 在 max(8 m, 2 s·v0) 内淡出：
朝人类路径恢复、用人的时机）；8 个位姿 imit 全有效（rf_pai_pool 把 NaN 填 0，这些行必须去掉，否则模仿的是「原地不动」）；fut 不穿过路沿线段（rf_pait 的 G-road 规则）。
只取 train160 的 scene，断言不含 hard_exam / control / hard_train。
抽样：PAI 行占池抽样的 30 %（`--on-weight pool=p:W`），其中转弯接近行（rf_pai_follow 的转弯定义：30 m 窗内净航向变化 ≥ 60°，在行自己的日志路径上找；入口前 40 m 到出口）× 2。
脚本 [rf_paionp.py](../scripts/rf_paionp.py)（tag / pack / r1）、链 [rf_paionp.sh](../scripts/rf_paionp.sh)。结果 `experiments/route_follow/results/paionp/`。

## 训练前的闸（不过则不训，直接报告）

- **G0 纠正信号**：可用的转弯接近行上，fut 与策略自己 plan（`poses_model`）在 2 s 的位置差 |fut − plan| 中位 ≥ 0.2 m（否则这些行没有可学的纠正）。
  同时报告：转弯行 / 直行行数，2 s 横向差（朝弯心为 +）与 4 s 弧长差的中位和四分位，按入口前 40–20 / 20–0 m / 弯内、左 / 右分。
- **R1**：RB0-F-s0 在池的 token + ego + 路线特征上重算，对记录的 `poses_model`，逐坐标容差 max(0.03 m, 4 个 fp16 ulp) 的超出份额 ≤ 0.01；错 seed（RB0-F-s1）≥ 0.9。
  （核查：`on1_PAIRB0` 的 build.json 只记了路线特征对 served `rmem` 的闸，没有模型 R1 的记录，所以本登记补读，4 096 行随机样本。）

## 判定线（s0 判定，s1 同表报告；参照数取 [results/tschool/summary.md](../results/tschool/summary.md) 的存档读数，不重跑）

| 线 | 定义 | 阈值 |
|:--|:--|:--|
| L-held-mean | held 460 / 18 决策均分 | ≥ RSG-L18 − 0.01 = **0.8566** |
| L-held-zeros | held 460 零分 | ≤ RSG-L18-s0 + 3 = **37** |
| L-guard | 公开 1491 / 10 决策（1 479 个有 rollout）零分 | ≤ RSG-L18 + 5 = **50** |

三线全过：写 Tokyo 的 PAI 闭环请求一行（`RSGPAIonp-L18-s0 exam_turn26`，标准 serving），与 RSG 存档读数比零分和 rf_pai_follow 完成率；**是否真跑由主会话 / 用户决定**。
任一线不过：不写请求。

## 诊断读数（无线）

- PAI 侧开环：`rf_pai_openloop.py --p1`（Tokyo P1 渲染，48 个 exam-hard 转弯 scene，人类状态），只看横向（2 s 横向缺口、4 s 终点内 / 外），
  RSGPAIonp s0 / s1 对 RSG-L18-s0 配对；同表列 RSGPAIT-L18-s0。速度与开始转的时机这个读数分不开（p1-openloop 第 3 条），不读。
- held 转弯完成率（rf_follow）、内切（rf_inside）、held 452 配对确认（ol_confirm），与 RSGboth / RSGPAIT 同口径。

## 会推翻 / 削弱本登记解读的情况

- R1 不过：池的输入与 rollout 不一致，读数不可解释。
- 转弯接近行太少（< 1 500 行或 < 30 个 scene）：×2 加权后仍是小样本，结论只按方向读。
- `fut` 的时机是「t0 时刻人的推进量」，不是「同一位置上人的速度」：自车落后或超前人很多（|ds0| 大）的行，目标速度与位置不对应。报告 ds0 分布。
