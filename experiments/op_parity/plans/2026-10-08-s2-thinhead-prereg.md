# op_parity / s2-thinhead 预登记：系统二 = VLM 特征上的薄 head，在 F20 候选里选一条（WOD val，479 rater 帧）

2026-10-08，打分前提交。接第 164 / 168 条。执行：`scripts/s2_thinhead.py` + `scripts/s2_thinhead_chain.sh`，结果 `results/s2_thinhead.md`、
`results/s2_thinhead/`、`figs/s2_thinhead/`。候选集、重定时、RFS、bootstrap 全部复用 `s2_gohold.py` / `wod_launch_report.Ctx`（B = 4 000，按序列配对，
cluster-mean RFS，WP2 = 两个 seed 逐帧平均）。

## 设定

系统一 = WP2（`WP2-full-s0/s1`），唯一的轨迹生成器。系统二 = 选择器，从 F20（WP2 路径族 5 × 速度 4，`s2_gohold.factored(plan, v0, 3, cb)`，
codebook 只用 train log 拟合）里选一条。特权上限 +1.068（第 168 条）。本 lane 不向 WOD test 提交，oracle 只作上限。

## Q0（离线，先报）：事后监督（hindsight）的上限

val 479 帧上，用 **日志未来** 选候选，再按 rater 打分：
- `H-ade`：与日志 20 个 waypoint ADE 最小的候选（主读数）；`H-fde`：3 s 与 5 s 两点距离均值最小的候选（RFS 的检查时刻）；
- `H-cls`：语义类：速度 = 日志的 V3 类（hold / creep / go；与 WP2 自身类相同时取 follow），路径 = 5 条里 5 s 横向偏移最接近日志的；
- 参照：日志本身、F20 oracle。平局一律偏向计划本身（`first_best` 的顺序）。
判据：`H-ade − WP2` 的 CI 下界 > 0 → 事后标签有可证的价值；CI 含 0 且点估计 ≤ +0.1 → 事后监督「按构造帮不上」，(a) 只作为 (c) 的预训练保留；
分层（standstill / moving / turn）单独报，某层 CI 下界 > 0 时 (a) 在该层仍算有效监督。

## Q1：冻结特征上的薄 head

**输入流（每个臂都是多流融合，各流单独标准化 / 降维后拼接；MLP 里每流各自一个线性编码器）**
- `ego` 流（每个臂都有）：`pp_wod.wod_ego` 的 20 维（command one-hot、4 帧位姿历史、速度、加速度），即 P2 配方喂给 adapter 的同一组量；
- `plan` 流（每个臂都有）：系统一计划的描述（1–5 s 弧长 5 维、3 s / 5 s 横向位移、5 s 航向变化、自身 V3 类 one-hot、三个原型在该速度档的 5 s 距离），
  候选是计划的函数，选择器在计划下游，必须知道计划；
- 视觉流：
  - `C` = Cinque `temporal`（512 维，`op_cinque_p3_trainval`，shipped Cinque，系统一已经看到的表征）；
  - `Q1` = 冻结 Qwen3-VL-4B 单帧：`qwen_front3` 的 `L18_mean`（前三相机一次前向，第 18 层 image-token mean；第 5 条：中层 + mean pooling；
    缓存里有 L09 / L18 / L27 / L36，L18 最接近 L20–L24，层曲线作描述性附表）；
  - `QV` = 冻结 Qwen3-VL-4B 多帧（现成缓存 `qwenvid_train_t4`：每相机 4 帧、间隔 0.2 s、跨 0.6 s，Qwen 自己的 video 路径，三相机，`L18_mean`；
    train 约 15.7 万行 + val 全有）；
  - `QL` = 多帧长跨度（新提取，只在 479 帧上）：每相机 4 帧 t0 − 1.5 / −1.0 / −0.5 / 0 s（间隔 0.5 s，与 ego 位姿历史同一组时刻），同一 video 路径、
    同一层、同一 pooling。train 子集只在 (a) 需要时再提。
- 臂：`E0`（只有 ego 流，字面意义的地板）、`E`（ego + plan，实际地板）、`C+E`、`Q1+E`、`QV+E`、`QL+E`、`Q1+C+E`、`QV+C+E`。
  每个视觉臂同时报「对 `E` 的增量」（配对 CI）。缺多帧特征的 rater 帧（缺历史）用 train 均值代替并注明。
- 降维：各视觉流在 **WOD train 行上无监督** 拟合标准化 + PCA（白化），不碰 val 标签；k 个主成分。

**head（各特征臂完全相同的 head 族与超参网格）**
- `L`：多输出 ridge，回归 20 个候选相对 keep / follow 的 RFS 增益，取预测增益最大者（keep / follow 预测增益恒为 0）；k ∈ {8, 32, 128}、λ 网格
  由训练折内部的 4 折（按序列）选；`+C` 臂每流各 k（另报等输入维度的敏感性：每流 k / 2）。
- `M`：每流线性编码到 32 维 → 拼接 → GELU → dropout 0.5 → 20 维；listwise 损失（对 softmax(RFS / τ) 的 KL，τ = 0.5）；weight decay 网格由内部折选；
  k = 64（`+C` 臂每流 32）。
- 主 head = `L`（闭式解，479 帧下最稳）；`M` 为第二 head。

**监督**
- (a) 事后候选类：WOD `r2-train` 行（token 路径算出的 WP2-s0 计划，每序列隔 3 行取 1），标签 = `H-ade` 类；20 类 softmax（`L` 为多项 logistic，`M` 同形）；
  超参 / 早停只看 `r2-dev`（WP2 没见过的序列）的事后 NLL；直接用于 val 479 帧（没用 rater 标签，不需要 k 折）。报 train / dev / val 的标签分布
  （WP2 在 r2-train 是 in-sample，计划更贴日志，follow 类偏多；这是已知偏差）。
- (b) rater 分数回归：val 479 帧，每帧 20 个候选的 RFS = 稠密 20 维目标（两个 seed 各作一个样本），**按序列 5 折 × 10 次重复**，
  标准化、超参选择都只在训练折内；每帧的 out-of-fold 分数 = 10 次重复的平均。
- (c) 先 a 后 b：`L` = 把 (a) 的 20 个 logit 作为额外输入流进 (b) 的 ridge；`M` = 从 (a) 的权重初始化，折内 listwise 微调（对初始化的 L2-SP 正则）。

**读数**
- out-of-fold 的「所选候选 RFS − WP2」，配对 bootstrap（序列，B 4 000）：全部帧、standstill / moving / turn、10 个场景 cluster；占 F20 oracle 上限
  （+1.068 及各层上限）的比例；in-sample（全量拟合）数字并列；常数策略（折内最优常数候选）一行。
- 置换对照（(b)，`L`，每个主臂）：把输入行在帧间打乱（两个 seed 同一置换）后走同一套 out-of-fold 流程，200 次；报零分布均值、95% 区间、实际值的 p。
- **对设计起决定作用的对比**：`Q+E − C+E`（单帧与多帧各一）、`Q+C+E − C+E`（Qwen 在 Cinque 之上加不加）、多帧 − 单帧（`QV+E − Q1+E`、`QL+E − Q1+E`），
  都是同帧配对 CI。

**判据**
- 「head delivers」：主族 = head `L`、监督 (b)、特征臂 {`C+E`, `Q1+E`, `QV+E`, `QL+E`}（4 个）。某臂 out-of-fold 对 WP2 的 CI 排除 0（下界 > 0）且
  moving 帧不亏（moving CI 上界 ≥ 0 且点估计 ≥ −0.05）→ delivers。4 个臂的多重性：另报 98.75% CI（Bonferroni），两者不一致时写「弱」。
  还需实际值超出置换零分布的 95% 区间。
- 「Qwen adds」：`Q+E − C+E` 的 CI 排除 0（> 0），单帧、多帧分别判；`Q+C+E − C+E` 并列报。
- 「多帧有用」：`QV+E − Q1+E` 或 `QL+E − Q1+E` 的 CI 排除 0。
- 「视觉有用」：视觉臂 − `E` 的 CI 排除 0。

## Q2（Q1 有某个臂 out-of-fold CI 排除 0 才做）

候选集大小与词表（F20 / F30 / 只选速度 / 只选路径）；保守选择规则（预测增益超过 margin 才离开 WP2 的计划，margin 在折内拟合）；同一选择器接在
shipped 上。

## Q3：轻量联合微调（计划内的 pilot，不只是后备）

冻结读数之后做：Qwen3-VL-4B 多帧（`QV` 的输入配方）+ ego / plan 流，LoRA 加在截断到 18 层的解码器的最后 4 层（融合层）+ head；先在 r2-train 子集上
用 (a) 预训练，再按同一序列 k 折在 (b) 上微调（5 折 × 1 次重复，pilot 规模）；同样报 out-of-fold 对 WP2、对冻结 `QV+E` 的配对 CI、in-sample 与
out-of-fold 的差、一次置换对照。零初始化 LoRA 的输出须复现缓存的 `L18_mean`（等价性检查）。
**不做的条件**（两条都满足才跳过，并写出证据）：冻结读数里 Qwen 特征在 ego + Cinque 之上没有任何增量（所有 `Q* − C+E`、`Q*+C+E − C+E` CI 含 0 且点估计
≤ 0），**并且** 诊断说明微调改变不了这一点：(i) train 上数据充足的事后任务里 Qwen 特征对 `E` / `C+E` 也没有增量（不是 479 标签不够的问题），
(ii) (b) 的学习曲线（训练帧 25 / 50 / 75 / 100 %）不随数据上升，(iii) in-sample 已接近上限而 out-of-fold 为 0（纯过拟合，瓶颈是标签数）。

## 约束与预算

- rater 标签只在 val：凡是用它拟合的数都按序列 out-of-fold；不在打分帧上拟合任何东西（PCA / 标准化用 train 行或训练折）。
- 预算约 6 卡时：Q0 与冻结 head < 0.5（head 训练是 CPU / 小显存）；train 行上的 WP2 计划 ~5 分钟；`QL` 提取 479 帧 ~5 分钟；延迟 bench ~5 分钟；
  Q3 pilot ~2–3 卡时。所有 GPU 作业走 `jevdrive.cl submit`。优先级：Q0 → 冻结单帧 / 多帧 + ego 流 → 联合微调 pilot → Q2。
- 延迟：特征提取（Qwen 到第 18 层，batch 1）+ head，单独 bench；卡被共享时只作上界并注明。

## 检查

复现 WP2 8.111 / F20 oracle +1.068；特征与帧按 `frame_name` 对齐（特征 → v0 的线性读出作对齐检查）；token 路径的 WP2 计划在 val rater 帧上对
harness 计划的差；`r2-train` / `r2-dev` / `wod/val` 互不相交（`splits.check_disjoint`）。
