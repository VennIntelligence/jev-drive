# 预登记：STACK（约束代替目标的合一臂：拉格朗日约束 + 模仿 dropout + 最小修正偏好 + 体内反射模块），2026-10-16

背景：[决定 321](../../../research/decisions/321.md)（JOINT-DAGGER：自身状态上的专家几何目标打掉避让，held 零分 +38）、[303](../../../research/decisions/303.md)（+ 补记 5：接近课的害处在形状项）、
[286](../../../research/decisions/286.md)、[318](../../../research/decisions/318.md)（BLIND：失败前 3 s plan 已接触，路线不是原因）、[220](../../../research/decisions/220.md)
（碰撞都在 plan 扫掠里；冻结模型的几何头读不出「会撞」，执行 plan AUC ≤ 0.72）、[recovery_envelope](../results/recovery_envelope/)（RCVRL 过 held H1 + 护栏：R + L 可合成）、
[dagger_check](../results/dagger_check/dagger_check.md)、[critic_pilot](../../critic_pilot/README.md)（每个状态只有一个执行过的 plan，critic 排不出 plan）。
这几轮得出的规则：**不等式型的后果损失**（自身状态上的 agent / road hinge、速度上界）有用或可合成；**等式型的几何目标**打掉避让。

用户决定（2026-10-16 ~11:40 JST）：所有组件叠成**一个合一臂**；先便宜检查，再剂量 pilot，再分阶段读；合一臂不帮忙就换方法（现在不做逐项消融）。
用户设计改动（11:55 JST，本文件第 4.5 节；12:10 JST 文献核对后删 PCGrad / L2-SP、加模仿 dropout，第 4.3 / 4.4 节）：原组件 5（serve 端对后果头做梯度步）**取消**，换成**模型内的身体 / 自我模块**（反射），在同一个合一臂里训练；后果头（C2）只作训练信号，不上推理。
本文件在任何模型输出之前写好、提交、推送；之后只追加补记。

## 1. 便宜检查（先做；结果写 `results/stack/checks.md`）

### C1 计划头能不能采样（读代码，无需模型输出）

- 读法：Cinque 的 `plan` 输出是 MDN 一个分量：前一半 33 × 15 是均值（`jevdrive.op_adapt.plan_index` 只取前 495），后一半是同形状的 spread；没有多假设、没有假设概率
  （旧 supercombo 的 5 × 991 + 概率列在 `jevdrive/openpilot/model.py` 是另一个模型）。盒上打印 `net.slices["plan"]` 的长度确认（990 = 一个分量）。
- spread 在我们的配方里**不受训练**（蒸馏只取 MDN 均值，模仿只用均值，`op_adapt.distill_index`）；d220 读过它（2 s 横向 std，AUC 0.786），只是一个弱的不确定性信号，不是候选来源。
- 结论（登记）：候选 = 模型自己 plan 的**结构化扰动**，K = 8：
  横向偏移 d ∈ {−1.0, −0.5, +0.5, +1.0} m（沿 plan 自己的法向，幅度从 0 s 的 0 线性涨到 2 s 的 d、之后保持）；曲率增益 g ∈ {0.8, 1.2}（`perturb.branch_poses(P, g, 1)`）；
  弧长比例 s ∈ {0.75, 0.9}（`branch_poses(P, 1, s)`，更慢）。候选 0 = 模型自己的 plan。C2 的反事实、第 4.2 节的偏好项用**同一套**变换（`lib/stack_cand.py`，numpy 与 torch 两版逐位核对 ≤ 1e-5 m）。

### C2 后果模型学不学得会（不训练 policy）

- 行：own = `on1_L18RSG`（RSG-L18-s0 自己的状态，17 474 行，带 agent 框与道路栅格）；human = navtrain 训练 / dev 行里带 agent 框（`navtrain_all-k32`）和道路栅格（`JT.ROAD`）的行，抽 ≤ 40 000 行。
  dev = 留出 log：own 用 `rf_recov.dev_of`（crc32(log) % 100 < 15），human 用 NAVSIM 的 dev 划分；训练只用 train log。
- 候选中心：RSG-L18-s0 在行自己输入上的前向 plan；每行 1 + 8 个候选（上节）；human 行另加日志未来作第 10 个候选。
- 标签（任何 plan 都能算，训练时用特权框 / 栅格）：rf_dagger.contacts 的几何（agent：扩展 ego 框与物体框有符号距离 < 0 且 counted；road：Pacifica 四角道路栅格 sdf < 0），0.1–4 s 稠密 40 步，
  按 0.5 s 分 8 个时间格，两类各 8 个 0 / 1；t = 0 已接触的行剔除。
- 头：冻结的 policy token（最新槽 + 一个更早的槽）+ ego 20 维 + 候选 8 个位姿 → 2 × 8 个接触 logit（可以复用 `critic_pilot/lib/critic.py` 的结构）；BCE，固定训练表（不在 dev 上挑 checkpoint）。
- 闸（登记，主会话原文）：
  - **held-out-log AUC ≥ 0.85 on counterfactual candidates**（dev own 行，全部 1 + 8 个候选合池，任一时刻任一类接触；log 聚类 bootstrap CI）；同报：agent / road 分开、human dev 行、
    **执行 plan 的 AUC**（只用候选 0，跨状态；d220 参照 ≤ 0.72）、**只看候选类型 + 速度的基线 AUC**（逻辑回归，说明头是否在读场景）、**状态内 AUC**（同一状态的候选之间排序）。
  - **梯度对齐**：dev 上有真实接触的候选，沿 −∇(预测接触概率之和) 对 plan 走一小步（归一化到每个位姿 RMS 0.10 m），**真实几何接触深度**（40 步上 agent 穿透 + 道路 sdf 负部之和）下降的比例
    **≥ 70 %**；同报同步长随机方向的对照比例。
- C2 不过：第 4.1 节的后果 hinge 从合一臂里**拿掉**，如实写明；其余组件照常。C2 过：冻结头存 `$DATA_DIR/runs/route_follow/stack/cons/head.pt`，接口见 `lib/conshead.py`。

## 2. 合一臂（全部组件一起，RSG-L18 配方，`--init RB0-F-s0`，任何地方都没有几何目标）

配方 REC（`rf_jdagger.sh` 的 REC）+ 池 `on1_RCVL`（S 行 on1_L18RB0 + 包络行，RCVL 的抽取权重，`--on-k 20 --on-mode self`）+ `--env-lam 1 --env-lo-scale 0`（**只用上界**）。2 500 步（pilot 1 000 步）。
对照 = 存档 RSG-L18-s{0,1}。

### 4.1 约束代替目标，拉格朗日乘子（`rm_train --dual`）

- 约束 c 与每行值（全部已有几何，形状只梯度 `--shape` 不变）：
  `agent`（自身状态行的 agent hinge，margin 0.3）、`road`（自身状态行的道路 hinge，margin 0.3）、`envU`（包络行的速度上界超出，m/s）、
  `cons`（只在 C2 过时：冻结后果头对**全部行**（human 行也算）当前 plan 的预测接触概率，行值 = relu(max_格,类 p − 0.5)）。
- 损失里每项的系数 λ_c 不再是手设权重：λ_c 初值 = 原配方的手设权重（agent / road：10 × ho_w 3；envU：10 W × ho_w；cons：10 × ho_w），每步对偶上升
  `log λ_c += η (r̂_c − τ_c) / max(τ_c, 0.02)`，η = 0.02，λ_c 夹在 [λ_c0 / 10, 10 λ_c0]；r̂_c = 违反率（该约束适用的行里行值 > 0 的份额）的 EMA（β 0.95）。
- 目标违反率 τ_c = 档位 × 参照率：agent / road / cons 的参照率 = human 行自己（navtrain 训练行的日志未来在同一几何、同一 margin 下的违反率；`rf_stack.py rates` 一次算好写进 flags）；
  envU 没有 human 参照（人的速度剖面不在策略自己的状态上），参照率 = 前 25 步 r̂ 的均值 × 0.5。

### 4.2 最小修正偏好（`--pref W`）

- 自身状态行上，K 个候选（C1 的变换）围绕模型**当前**的 plan（detach）生成；特权几何评分：可行 = 无 agent 接触、无道路接触（margin 0，同 C2 标签）、stop 类包络行上 envU ≤ 0.5 m/s（不冲必停线）。
- 模型自己的 plan 已可行：该行无此项。否则标签 = 可行候选里离自己 plan 最近的那个（8 位姿平均距离）；项 = W × ho_w × 8 位姿 L1 距离之和 / n_imit。没有可行候选：无此项。
- 报告：有此项的行份额（每步日志 `pref/share` 与汇总）、选中候选的类型分布。

### 4.3 模仿 dropout（文献核对后加入，主会话 12:10 JST；ChauffeurNet M4：绕停车车 55 → 90 / 100）

- 任一行（自身状态行或 human 行）上只要本臂的某个后果约束在该行激活（行值 > 0：agent、road、envU、C2 过时的 cons），该行的模仿项与 anchor 项（朝出厂模型的项）置零，
  目标永远不和约束打架。自身状态行本来就没有模仿项（`self` 行只有 hinge 与非 plan 头的 distill_ho），所以实际作用在 human 行（cons 激活时）与 anchor 行上；
  C2 不过（无 cons）时本项对 human 行无作用，如实写明。P2H10 在模仿行上的 drivable hinge 属原配方，不算本臂的约束，不触发 dropout。
  dropout 的行仍算进 n_imit（各 hinge 的归一化不变）。报告每步被置零的模仿行份额。
- 偏好项（4.2）同一规则的落实：偏好标签必须在**训练 margin 下**满足全部约束（agent / road hinge 在 margin 0.3 下为 0、envU = 0；不只是 margin 0 无接触），
  所以偏好目标在它作用的行上不和任何约束打架；偏好项保持单边（只在自己 plan 不可行的行上）。两种可行定义下有项的行份额都报。

### 4.4 已删去：PCGrad 与 L2-SP / EWC（主会话 12:10 JST，文献核对）

- PCGrad 删去（驾驶上无证据；Kurin / Xin NeurIPS 2022：普通加权和与之持平；乘子已在自适应权重）。L2-SP / EWC 删去（LIBERO：EWC 在全部方法里最差；Wołczyk ICML 2024：行为回放胜过 EWC）。
- 保留输出蒸馏 + human 行回放（原配方）。可选的「自身状态行上朝 EMA 老师的 L2」**不用**：那是自身状态上 plan 的等式目标，正是 d286 / d303 / d321 打掉避让的形式。

### 4.5 体内反射模块（用户 11:55 的设计，取代原 serve 端梯度反射；`--rx`）

用户的概念：模型里有一个身体 / 自我组件，与记忆交换并更新身体的动作与状态，给出反射式的纠正与避让，**不是想象未来**。实现（`lib/reflex.py`）：
- **自我状态输入（efference copy）**：上一个决策 served 的 plan（8 位姿，上一帧后轴系）按实测的 ego 运动搬进当前帧（d313 JEV_INTENT_HOLD 在 serve 端做过的同一变换），与实际位姿比较。
  特征（14 维）：有效位；τ ∈ {0, 0.5, 1.0, 1.5} s 时上一意图在当前帧的位姿（x − v τ、y、yaw；τ = 0 即实际状态对上一意图的横向 / 航向 / 纵向偏差）；上一意图在此刻的速度 − 实测速度。
  实测 yaw rate / 速度已在 ego 向量里。human / navtrain 行没有上一 served plan：全零 + 有效位 0（同路线记忆的做法）；自身状态行上 25 % 置零（独立 rng 流 [seed, 0, 113]）。
- **与记忆交换**：自我状态编码成一个 token，先读（cross-attention）路线分支的记忆 token（null、ego token、路线点），再被路线分支的 32 个槽查询读（一层 decoder，零初始化输出），
  结果加在 9 帧上下文的偏置上（与路线分支同一载体）：身体状态与记忆相互作用，而不是在头部拼接。
- **反射输出**：plan 0.5 / 1.0 / 1.5 s 的近场残差（横向与纵向，|Δ| ≤ 0.5 m，0.5 tanh），0 s 处为 0、节点间线性、1.5 s 之后保持 1.5 s 的值；乘一个零初始化的可学标量门与有效位。
  所以 step 0 模型 = 基座；有效位 0 的行（human 行）上模块的贡献恰好为 0。
- **训练信号**：只有本臂的约束型信号（agent / road hinge、速度上界、C2 过时的后果 hinge，含偏好项）作用在**修正后**的 plan 上；模块参数的梯度只来自这些约束项
  （每步对模块参数单独 `autograd.grad` 约束项之和，覆盖总反向给它们的梯度：不取模仿、蒸馏），没有几何目标。
  模块参数单独一组，lr 3e-4、warm-up 100（同路线分支，不乘 `--lr-scale`）。
- **serve**：驱动每个决策把上一 served plan 喂回来（sh30_driver / pai_driver 都记录它），`Core.plan(..., selfstate=...)`；checkpoint 带反射模块（sidecar `route_mem.json` 的 `reflex`）才喂，
  否则路径与现在逐位相同；另有开关 `JEV_REFLEX_OFF=1` 在带模块的 checkpoint 上关掉模块（有效位 0），用于 serve 端消融。
- 后果头（C2）**只作训练信号**，不上推理。

### 开关与逐位闸

- 每个开关关 = 逐位等于 RSG-L18 配方：新代码 60 步全关 `STKG-off` 对旧代码的 `JDGG-off`（标量、最终权重 max |diff| 0、on_hash_60 = RSG-L18-s0 的 `ea8cacc3e451c577`）。
- 反射模块挂上、step 0：同一 batch 的输出与不挂逐位相同（零初始化）。
- serve：不带反射的 checkpoint 经改过的驱动，plan 与改前逐位相同（离线重放存档决策）；带反射 + `JEV_REFLEX_OFF=1` = 同一权重不带模块的 plan。
- 合一臂 60 步 smoke（全部开关开），报每项的大小、λ 的走向、偏好行份额、模仿 dropout 份额、速度。

## 3. 剂量 pilot（1 000 步，seed 0，并行）

档位 A：τ = 1.0 × 参照率、偏好 W 1；档位 B：τ = 0.5 × 参照率、偏好 W 3。× `--lr-scale` {0.1, 0.3}。tag `STK{A,B}L{01,03}-L18-s0`。

闸（登记，主会话原文）：**dev ADE ≤ 0.545；G-rely true；equal-arc turning ≥ −0.3 deg vs RSG-L18-s0（不要求转弯增益）；dev own-state rows 上 served plan 的几何接触率下降 ≥ 30 %**
（on1_L18RSG 留出 log 的行，接触同 C2 标签、任一类、t0 接触剔除，pilot 带反射模块前向，对 RSG-L18-s0 同行同输入；log 聚类 CI 同报）。

过闸的 pilot 直接送 held 460 @18（`rf_recov_post.sh submit-held`，priority 9）。另：过闸里最好的一个（接触降最多且 ADE 过）再读一次 `JEV_REFLEX_OFF=1`（模块在 serve 端关），只作描述（模块在闭环里的贡献）。
Held H1：**zeros ≤ +3, at-fault ≤ +3, GT ≤ +0.03 m**（对 RSG-L18 两 seed 均值，配对，log 聚类；一个晚的、无关的单个零分按噪声处理）。

## 4. 全量

最好的过闸配置 → 2 seed × 2 500 步 → held 460 → 只有 H1 过才进第 2 阶段（公开 1491 护栏 + 盒上 PAI 281 seed 0，对盒上参照 `$DATA_DIR/runs/route_follow/recovery_envelope/pai/RSG-L18-s0`，不重读）。
线（登记，主会话原文）：H1 同上；**guard score CI low > −0.01, zeros ≤ +5；PAI zeros ≤ reference − 5 OR at-fault ≤ reference − 3**（本轮目的是少撞）；**strict turns not below reference**。
合一臂不帮忙（pilot 全不过，或全量 H1 / PAI 线不过）→ 换方法，不做逐项消融。C2 不过且其余组件的 pilot 也全不过 → 停并报告。

## 5. 交付与花费

代码（全部开关）：`lib/stack_cand.py`、`lib/conshead.py`、`lib/reflex.py`、`rm_train.py` 的 `--dual / --imit-drop / --pref / --rx / --cons`、`rf_stack.{py,sh}`、`rf_stack_c2.py`、驱动的反射喂入。
结果 `experiments/route_follow/results/stack/`（checks.md、pilot.md、lines.md），`research/next-round/turn-night.md` 末尾的日期条目，全量读完写决定文件。每阶段提交 + 推送。
花费估计：C2 ≈ 2–3 GPU job-h；pilot 4 × 约 0.4 h（反射参数单独一次约束项反向，约 1.5–2 倍于 RSG 配方）；held 读每个 pilot 6 个作业 × 0.18 h；全量 2 × 1 h + 读数链约 13 GPU job-h。

## 补记规则

所有 GPU 作业经池、`--after` 串链，不轮询。阈值、档位、K 在看到 pilot 结果后不改；改动只以带时间的补记追加，写明改动前看到了什么。
