# 预登记（草稿，待用户批准后启动）：AWR-ONPOL，用后果给 on-policy 决策加权，不用任何专家标签，2026-10-17

本文件在任何 pilot 训练与读数之前写好并推送，之后只追加补记。链 [scripts/aw.sh](../scripts/aw.sh)，打分 [scripts/aw_score.py](../scripts/aw_score.py)，
闸 [scripts/aw.py](../scripts/aw.py)，训练项 [lib/awr_train.py](../../../lib/awr_train.py)（rm_train `--awr-*`，默认关），结果 `results/`，box `$DATA_DIR/runs/awr_onpol/`。

## 0. 为什么

- 闭环纠正线全都靠手做的标签：人类路径（d259 的模仿行）、规则专家（d321 JOINT-DAGGER、d322 恢复行）、hinge 目标（d264 corridor K、d286 路线 hinge、d303 接近课）。
  唯一被采用的是 S 的两个 hinge（d259 / d264 / d286），之后加剂量（d311 ON-K）、换状态（d333 RSG-ACC）都饱和。瓶颈在标签质量：几何目标在策略自己的状态上会把「别撞」打掉（d321）。
- 用户要的方向：标签来自后果。每个决策按它之后在模拟器里真正发生了什么加权，没有专家、没有手写目标轨迹。
- 和 d270（候选打分线）不同：那里用 d223 接触头给反事实候选打分，头读不出哪条能救（10 / 42）。这里不学头、不造候选，
  只比较**真实发生过**的结果：同一个状态下 6 个策略各自做了什么、后来怎样。

## 1. 数据与打分（aw_score.py，只读存好的 rollout）

- 池 `on1_RSGACC`（d333）：6 个策略（RB0-F-s0 / s1、RSG-L18-s0 / s1、RSGlr03n5000-L18-s0、RSGk51w1-L18-s0）在 ACC+GAP serve 下，nuPlan onpol-train 1 020 个 scene
  （18 决策、0.5 s）与 PAI train160（0.1 s 一个决策）各滚一遍；新行 259 309、去重状态 62 093；旧池 `s`（默认 serve、单策略）没有同伴，不打分。
- **窗口分数**（评测器自己的项，eval/aggregation/scene_score.py 搬到窗口上）：决策时刻 t，窗口 (t, t + H]，H = 3 s。按评测器的时间步顺序走，
  第一个终止步（collision_any、offroad、dist_to_gt_trajectory ≥ 4 m，即评测器的 RemoveTimestepsAfterEvent）结束窗口；那一步若是 at-fault
  （collision_front / lateral、offroad、left_corridor_laterally，即 scene score 的硬失败）→ s = 0；只有追尾或向前开出 corridor → 窗口截断、不算失败。
  否则 s = min(1, 自车沿 GT 前进 / (0.8 × 日志同窗前进))（scene score 的 0.8 饱和）；日志同窗前进 < 2 m × 窗长 / 3 s 时 s = 1（scene score「GT 太短给满分」的窗口版）。
  无效：t 之前已有终止步（评测器已停止计分），或无失败且窗长 < 1 s（scene 末尾）。舒适度：存的 metrics 里没有，不用。
- **基线**：同一 (scene, 决策时刻)、**其他策略**的行里，自车状态近乎相同的（后轴 ≤ 0.5 m、航向 ≤ 3°、速度 ≤ 0.5 m/s；nuPlan 地图系 / PAI scene 系；
  交通回放日志，所以「时刻 + 自车状态」就是状态）的 s 均值 = kind 1；没有这样的同伴 → 同 (scene, 时刻) 其他策略的留一均值 = kind 2（逐 scene / 时刻基线，状态不同）；
  都没有 → 无定义。**A = s − b**。窗口包含本策略之后几个决策，所以 A 是「从这个状态起按本策略开 3 s」对同伴的优势（蒙特卡洛，单条 rollout）。
- **目标（动作）**：该决策上策略**模型自己输出的 plan**（nuPlan = dump 里 plan 均值经 pp_train.rear 取 0.5–4 s 8 个后轴位姿；PAI = rollout cache 的 `poses_model`）。
  ACC 是环境的一部分（模型 plan 的确定性后处理），所以 served（ACC 改过的）plan 不是动作；拿它当目标等于把 ACC 的纵向蒸进模型，而读数用默认 serve。
  已核：默认 serve 的旧池上 rear(mu) 对 served 中位 0.6 mm、最大 2.6 cm（dump 的 fp16 舍入）。
- **覆盖**（H = 3 s，results/score.md，补在下表，pilot 之前）：

  （由 `aw.sh score` 填：每条赛道 train / dev 的行数、有效窗、失败窗、A 有定义 / kind 1 / kind 2、A ≠ 0、A > 0、A ≥ 0.1、A ≥ 0.5 的行数）

## 2. 训练项（rm_train `--awr-pack`，lib/awr_train.py）

- **选 filtered BC，不选 AWR 的 exp(A/β)**：(1) 绝大多数行 A = 0（同伴结果一样），AWR 给它们权重 exp(0) = 1，等于把 6 个行为策略自己的 plan 平均地再模仿一遍
  （集成自蒸馏，不带后果信息，其中 2 / 6 是 init 底座 RB0，会把续训往回拉）；(2) AWR 的权重恒正，表达不了「别学这个」，而我们的失败史（d321、d286、d303）
  恰恰是在自身状态上加几何目标会覆盖掉「别撞」，所以目标集合要小、只放在后果能区分的地方；(3) 只有一个旋钮（阈值），好解释。
  代码里 `--awr-mode awr --awr-beta` 也在（按 exp(A/β) 抽样），本登记不用。
- 行：pool 里 A > τ、kind 1（只用近乎相同状态的同伴基线）、train log（tab `dev` ≠ '1'，dev 行从不抽）、目标有限。每步 K = 16 行，自己的 rng 流
  [seed, 0, 167]，第二次前向（与主 batch 喂 pool 行的方式完全一样：S.ego 带路线记忆、不 drop、不作 anchor）；损失 = 对目标 plan 的模仿距离
  （pp_train.Losses.dist，与模仿项同尺度），求和 / 每 batch 的普通行数 × λ（λ = 1：每行重量等于一条日志模仿行；16 / 102 ≈ 模仿项的 0.16）。
- **底座配方不变（保留）**：RSG-ACC 的 p0k13 命令逐字 = RSG-L18-s0 的命令（`--init RB0-F-s0 --warmup 1 --lr-scale 0.1 --yr-mass 0.1 --label-bank --agent-lam 10
  --agent-on-only --shape --ho-w 3 --rm-lr 3e-4 --on-split alpasim_nuplan/onpol-train --on-mode self --eval-every 500`），`--on on1_RSGACC --on-k 13`，新池行抽样权重 0
  （抽样逐位等于旧池 `on1_L18RB0`，存档 on_hash_60 = RSG-L18-s0 的 `ea8cacc3e451c577`）：普通行、13 行 S hinge 行、yaw-rate 行全不动，AWR 行是附加的。

## 3. 检查（启动 pilot 之前，aw.sh codecheck / smoke）

- **逐位**：RSG-L18 配方 60 步，旧代码 d7675c5a（SELF-ITER 与 AWR 之前）对本代码全部 `--awr` 关：标量、全部训练权重 max |diff| = 0，dev ADE 相等，on_hash_60 = `ea8cacc3e451c577`。不过则停。
- **冒烟**：τ = K1 的臂 40 步走一遍 pool（AwrSet 加载、抽样份额、第二次前向、awr 损失有限）。

## 4. pilot 与闸（aw.sh pilot，一次提交整条链）

- 臂（1 000 步、seed 0）：对照 `AWR0-L18-s0` = 上面的底座命令（= RSG-ACC 的 `RSGACCpp0k13-L18-s0` 重训，同代码，可与之逐位比）；
  `AWRF01-L18-s0`（τ = 0.1）；`AWRF05-L18-s0`（τ = 0.5，基本只剩「同伴失败、它没失败」的行）。
- 离线闸（每臂对 C = `AWR0-L18-s0`，全部成立才过）：
  - **G1** dev ADE ≤ C + 0.01；
  - **G2 横向不变**：rely.json `read` 为真，且 held01 上等弧长转向对 C ≥ −0.3°（rf_recov.gate_turn）；
  - **G3 后果代理 PREF**（命名的开环代理）：dev log（从不抽）上近乎相同状态的同伴对（同 scene、同时刻、不同策略、分数差 ≥ 0.1），
    margin = ADE(plan, 输家的 plan) − ADE(plan, 赢家的 plan)（plan 取模型在两行输入上的均值），臂 − C 的 log 聚类配对 CI 下界 > 0：
    在没训过的 log 上，模型往当时结果更好的那个策略的动作靠。输家失败的子集同表报告。
  - **G4 不更坏**：新 nuPlan 池 dev 行上重算 plan 的接触率（车或道路，0.1–4 s，rf_rsgacc.contact），臂 − C 的 CI 下界 ≤ 0；PAI dev 行同表报告。
- **闭环（只对过闸的臂提交，gate 任务自己提交）**：held 460 @18（默认 serve，6 chunk，rm_chain.sh read）→ rf_follow → 对存档 `RSG-L18-s0` 读数：
  **零分 J ≤ C + 3** 且 **转弯完成 J ≥ C − 2**（C = RSG-L18-s0）。对照 `AWR0-L18-s0` 同时读（只报告，不判；用来把 1 000 步对 2 500 步的差和 AWR 的作用分开）。
- **判词**：两条闭环线都成立 = 「后果加权的 on-policy 克隆在 held 上无害且保住转弯」，可以进 2 seed 正式（另登记）；任一不过 = 停，报告。
  离线没有臂过 = 停，不读闭环。

## 5. 成本（估计，box 5 卡）

- 逐位 2 × 60 步、冒烟 40 步：各约 5–8 min（主要是 on1_RSGACC 70 GB token 的 Store 加载），≈ 0.3 GPU 作业时。
- pilot 3 × 1 000 步（RSG-ACC 同命令 243 s + 第二次前向约 +15 %，加载）：≈ 0.5 作业时；前向 held01 + dev-rec 两个任务 ≈ 0.6 作业时；PREF / 闸 CPU 几分钟。
- 闭环：每个读的 tag 6 个 chunk × 约 0.3 h（1 / 4 卡）≈ 1.8 作业时；两臂都过 + 对照 = 3 个 tag ≈ 5.4 作业时（≈ 1.4 卡时）。
- 合计：离线 ≈ 1.4 作业时（≈ 0.5 卡时），全链最多 ≈ 7 作业时（≈ 2 卡时），墙钟 1.5–3 h（看队列）。

## 6. 会削弱解读的情况（设计疑点）

- 状态来自 ACC+GAP serve，读数用默认 serve：ACC 下纵向主要由 ACC 定，分数里的进度项与模型 plan 的速度关系弱；A 主要来自横向与失败事件。
- 基线只有 ≤ 5 个同伴、单条 rollout：A 有蒙特卡洛噪声；窗口里包含本策略之后的决策，A 是「策略从此状态起」的优势，不是单个动作的。
- 近乎相同只看自车位姿与速度，不看历史帧；同状态下的输入（8 帧历史）不同。
- PAI 行 0.1 s 一个、强相关；一个失败事件让之前约 30 个决策都成失败窗。
- 6 个策略彼此很近（同底座族），能区分的状态少；A > 0 的多是「同伴犯错、它没犯」，学到的是同族里最好的那个，天花板低。
- pilot 1 seed、1 000 步；dev nuPlan 只有 4 个 log（G4 的 CI 窄得可疑，d333 已记）。
- 默认 serve 的多策略 on-policy 池（RB0 旧池、RSG 自己的 L18RSG、RTB1 的 L18RTB1）也能做同伴基线，而且与读数同 serve；本 pilot 没用，若结果有苗头这是下一步。
- held 460 的 RSG-L18 / RTB1 闭环读数**不能**作训练数据（是评测集），所以没用。

## 7. 启动命令（主会话在用户批准后执行，box、repo 根目录）

```
bash experiments/awr_onpol/scripts/aw.sh pilot
```
（需要 `aw-score` 与 `aw-bitcmp` 两个任务已在 jobs.txt 且成功；pilot 链挂在它们后面。K1 / K2 默认 0.1 / 0.5，AWK 默认 16。）
