# wod-pref 预登记：把 rater 偏好训进 System 1（WLG 的 adapter + plan 通路），读 out-of-fold RFS

2026-10-08。打分前提交。接第 163 / 164 / 168 / 169 条。兄弟 lane `s2-thinhead` 训练候选选择器；本 lane 走另一条路：直接微调 System 1。
代码 `scripts/wod_pref.py`，链 `scripts/wod_pref_chain.sh`，结果 `results/wod_pref.md`。

## 问题

WLG（WP2 + 停车门，val RFS 8.187）之后剩下的是偏好（第 164 / 168 条）。val 的 479 个 rater 帧（每帧 3 条打分轨迹）是唯一的偏好信号。
按序列 k-fold 用它微调 System 1，out-of-fold RFS 能比 WLG 高多少？参照：Poutine 用 GRPO 在 416 / 479 帧上训练，test 只比自己的 base 高 +0.08。
**预期效应约 +0.08**；结果按这个量级读。

## 数据与读数路径

- 479 个 rater 帧（478 条序列）。其中 234 帧是奇数帧号，不在现有 `wodval` trunk cache（只有偶数帧）里。`wod_pref.py prep` 对每个 rater 帧
  渲染自己的 10 帧流（f − 18 … f，步长 2，与训练 cache 相同的渲染器和 (f − 2, f) 图像对协议），过冻结 trunk → stage 4 → 9 个 slot 的 token，
  写 `cache/wod_rater/`。闸门 P0：在同时位于 `wodval` 流 slot ≥ 9 的偶数 rater 帧上，新 token 过 WLG-s0 得到的 plan 与 `wodval` token 得到的 plan
  平均差 < 0.01 m（token 的最大绝对差一并记录）。
- **读数全部走 token 路径**（训练时的同一路径），WLG 参照也在 token 路径上重算（配对，同路径）。闸门 P1：WLG token 路径 plan 与已存 harness
  预测（`preds/op_cinque_WLG-full-s{0,1}`）在 479 帧上的平均距离 < 0.05 m，且 token 路径 WLG 的 cluster-mean RFS 与 harness 的 8.187 相差
  < 0.05；两个数都写进结果。不为每个 fold 导 ONNX 跑 harness（CPU 配额被其他 lane 占着，harness 每次 14 核 × 约 25 分钟）。
- 折：`wod/pref5-f0..f4`（已注册，按 cluster 分层，按序列）。外层第 k 折打分；内层折 (k + 1) % 5 只用于 early stopping；其余三折拟合
  （约 287 帧拟合 / 96 帧内层 / 96 帧 out-of-fold）。任何一帧被打分时，产生它的模型没见过它的 rater 标签，也没用它选步数。
- anchor：`wod_r2` 的 r2-train 行，WLG 原配方不变（75% 模仿日志 + 25% 对 shipped 的 anchor 行，停车门 0.5 m/s，lam_i 1 / lam_c 3 / lam_d 30）。

## 模型与训练（除学习率外全部在此固定）

- 初始化：`WLG-full-s{seed}`；可训练 = WP2 的可训练集（ego adapter + plan 通路 16.2 M），视觉冻结。停车门保留（v < 0.5 m/s 时 adapter 关）。
- batch 64 = **16 个偏好行 + 48 个 WOD-train 行**（偏好行占 25%）。400 步，AdamW，wd 0.01，warmup 20 步后学习率恒定，grad clip 1.0。
  总损失 = WLG 原损失（48 行）+ 1.0 × L_pref（16 行均值）+ 30 × 偏好行上非 plan 头对 shipped 的蒸馏。
- 每 25 步（含第 0 步）对 479 帧出 plan 并存下；early stopping = 内层折帧均 RFS 最大的步（并列取最早；第 0 步 = WLG 本身是合法选择）。
  报告的 out-of-fold 数 = 该步的外层折 plan。2 个 seed（各自从同 seed 的 WLG 出发），臂 = 两 seed 逐帧分数平均（沿用惯例）。
- 学习率：小步 pilot（折 0、目标 (a)、seed 0）跑三档（plan 通路 / adapter）：3e-6 / 3e-5、1e-5 / 1e-4、3e-5 / 3e-4；
  **按内层折 early-stopping 步的帧均 RFS 选一档**，对所有目标、折、seed 固定。pilot 的外层折数会被看到（任务要求先看 out-of-fold 动不动），
  但不参与任何选择。

## 目标（偏好行上的 plan 损失；plan 先换成 WOD 的 20 个后轴路点 0.25 … 5 s）

记归一化 Huber 距离 d(p, q) = 对 20 个路点的 huber((x_p − x_q) / (0.3 + 0.2 t)) + huber((y_p − y_q) / (0.1 + 0.1 t)) 取均值（与模仿损失同一尺度，延到 5 s，无 yaw 项）。

- **(a) top**：d(plan, 最高分 rater 轨迹)。
- **(b) rank**：按分数加权的软最小：−log Σ_k π_k exp(−d(plan, traj_k))，π = softmax(score / 2)。允许 plan 落在就近的次高分轨迹上，不被硬拉过去。
- **(c1) hinge**：指标本身在 log 域的信任域 hinge。对 3 s、5 s 两个检查点和每个 rater k，按官方定义算 norm_kh（纵 / 横归一化距离的较大者，
  含速度缩放），L = 两个检查点的均值 [ −max_k ( log10 max(score_k, 0.01) − relu(norm_kh − 1) ) ]。域内无梯度，域外梯度恒定（不随距离指数衰减），floor 忽略。
- **(c2) f20**：采样奖励。每步对模型当前 plan（detach）建第 168 条的 F20 候选（`s2_gohold.factored`，5 条路径 × 4 个速度剖面，codebook 来自 WOD train），
  用精确 RFS 给 20 条候选打分，取最优（并列规则 `first_best`：优先自身，再取改动小的），L = d(plan, 最优候选)。即在小候选集上的贪心策略改进。
- **对照 perm**：标签置换。拟合折 + 内层折内，把每帧的（3 条轨迹，分数）整组换成同一 v0 档（< 0.5、0.5–2、2–5、5–8、8–12、≥ 12 m/s）里另一帧的
  （档内随机置换，seed 固定）；early stopping 也只看置换后的内层标签；out-of-fold 用真标签打分。它保留「这个速度下 rater 一般喜欢什么」，
  去掉「这一帧画面对应哪条偏好」。四个目标各跑 seed 0。

## 读数（全部对 WLG，配对 bootstrap 按序列，B = 4 000，cluster-mean RFS）

1. 目标 × {out-of-fold，in-sample（该帧在拟合折里的那些模型的平均），perm 对照 out-of-fold}，全部帧。
2. 分层：standstill（v0 < 0.5）、launch（v0 < 2 且日志 5 s 位移 > 5 m）、moving、turn（intent 左 / 右）、night / day、10 个场景 cluster（val 无 Spotlight）。
3. ADE 对日志（479 帧 @3 s / @5 s，对 WLG 的变化）与 r2-dev 2 000 行子样上的 8 点 ADE：plan 离模仿多远。
4. 第 168 条上限的份额：out-of-fold 增益 / 1.068；并在 WLG token 路径 plan 上重算 F20 特权上限作同口径分母。
5. 学习曲线：拟合折 / 内层折 / out-of-fold 的 RFS 对步数（图 1）。约 8 帧的相机画面 + BEV（WLG / 微调 plan / 3 条 rater 轨迹与分数）（图 2）。

## 判定规则

- **「偏好训练有效」**：某目标的 out-of-fold 对 WLG 差值的 95% CI 不含 0（下界 > 0），且没有任何分层或 cluster 的 CI 整体在 0 以下。
- **「来自逐帧偏好」**：该目标 out-of-fold − 其 perm 对照 out-of-fold（seed 0 对 seed 0）的 CI 下界 > 0；否则增益记为速度档级的通用偏置。
- **清晰的 null**：四个目标的 out-of-fold CI 全部含 0 或为负 → lane 结束，证据是学习曲线（in-fold 升、out-of-fold 不升）。
- 有效的目标：保存「479 帧全量」检查点（步数 = 该目标 10 个 fold-run 所选步数的中位数，2 个 seed），供以后可能的 test 提交；**本 lane 不提交任何东西**。
  四个目标 × 4 层 + 10 cluster 的对比很多，贴边的 CI 只算弱。

## 预算

prep 约 5 分钟（少量 CPU 渲染 4 790 帧）。每个 fold-run 约 2 分钟（400 步 × batch 64 + 17 次 479 帧评估）。pilot 3 个 run；全量 4 目标 × 2 seed × 5 折
+ perm 4 × 5 折 = 60 个 fold-run，12 个 pool 作业（每个作业串行 5 折，--vram 16 --cpu 3）。合计约 2.5 卡时，上限 4 卡时。
