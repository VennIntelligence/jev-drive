# 预登记：JOINT-DAGGER（规则专家在策略自己全部状态上的横纵合一标签），2026-10-16

背景：[DAGGER-CHECK](../results/dagger_check/dagger_check.md)（R / RL / RLF 在留出 log 自己的交接状态上不朝专家动：k = h ADE −0.2 至 −1.5 %，离专家仍 3.0 m；
昨夜恢复行 952 行里 870 行是专家开到的状态）；规则专家从 on1_L18RSG 全部 17 474 行自己的状态出发，4 s 无接触 0.89（路口 0.84 对策略 0.70，失败前 3 s 0.81 对 0.36–0.41），
转得比策略多；但它的速度不是好老师（人速 + 3 m/s 地板、弯里固定 4.6、无视停止线：停止线前 14 % 的行冲线；让行规则缩短 22 % 的 plan）。
[RECOVERY-ENVELOPE](2026-10-16-recovery-envelope-prereg.md) 的训练器路径（`both` 模仿行、`--on-mode mixed`、on_envelope 的上界、rf_recov 的 pack / fwd / gate、等弧长转弯仪器、
rf_recov_post 读数链）全部复用。用户 2026-10-16 09:10 JST 批准本轮，要求快。本文件在任何 pilot 之前写好、提交、推送；之后只追加补记。

原则（用户）：只有一个合一臂（横向 + 纵向在同一个标签里）；剂量在便宜的 pilot 上扫；完整读数链只给过闸的臂 + 一个对照；读数分阶段（先 held 460，H1 过了才读公开护栏和 PAI 281）；
标签在行自己的时刻、按当时回放的交通计算。

## 1. 老师修补：专家的停止线规则（`rule_expert.py`，开关 `JEV_EXPERT_STOP`）

- 规则 = rf_recov 的必停规则（on_envelope 训练的那条）：前保险杠前方 0–60 m 第一条 LONG-PROBE 合格线（ln_elig）；停车标志：人在该线停过；灯：此刻（最近关键帧 ≤ 0.3 s）红，
  且人在该线停过或 [t0, t0 + 4 s] 一直红（未知允许，至少一个红）。停止点 = 线 − 1.0 m（前保险杠），速度剖面从停止点起为 0、向后按专家 A_DEC 3.0 制动包络（与让行停车同一路径）。
  以 A_MAX 6 也刹不住的线不停（相当于黄灯通过），记录。
- 数据：`rf_jdagger.py stops` 把 onpol_nuplan ctx.npz 的线（origin 系路线 + ln_*、kf_ts）映射到每个场景 pack 的路径站位：仿真系 = nuPlan 地图系平移 O（on_rows：无旋转），
  仿真时间 = 日志时间 − T0；O、T0 取自所有 on-policy 池的 meta（pose_map − anchor、t_log_us − t_us）。线点离路径 > 3 m 的线不合格。检查：ctx 关键帧位姿 − O 对 pack 日志路径
  （同一时刻）的距离，p95 < 0.5 m，否则停。
- 开关关（不设 `JEV_EXPERT_STOP`，或场景没有线文件）= 逐位与原专家相同：在 on1_L18RSG 全部行上重跑专家，与 DAGGER-CHECK 存的 expert_L18RSG.npz 逐位相等（label 作业里断言）。
- 闸（只报告，不挡本轮，除非严重退化）：专家在 held 460 @18 上带开关的闭环读数（rf_expert.sh box-read，决定 314 的路径）`RSG-L18-s0-XP-L12-E46-ST` 对存档 `-XP-L12-E46`（22 / 24 弯）：
  完成转弯不得少（严重退化 = 少 3 个以上）；人停过线的场景里零分应减少；配对的均分 / 零分 / at-fault 报告。

## 2. 标签（合一）

- 行：on1_L18RSG 自己的状态 k ≥ 2（15 434 行），有专家 pack 的 1 019 个场景。
- 专家：XP-L12-E46 + 停止线规则，从行自己的后轴位姿、实测速度、仿真时间规划（一个场景一个 session，按 k 排序，同 DAGGER-CHECK），取 0.1 s 的 41 个模拟状态。
- 目标：专家的路径几何按速度剖面重新计时：v(a) = min(专家在该弧长上的速度, on_envelope 在行自己状态上的上界：停止线 vuf（前保险杠弧长）、弯入口 vur（后轴弧长）、前车按时刻 lsig），
  用专家自己的速度控制（加速 ≤ A_ACC 1.5、减速 ≤ A_MAX 6）从行的速度出发积分；没有界起作用时目标 = 专家 plan（数值上相同，label 作业报差）。只用上界，不用空路下界。
  目标位姿 0.5 .. 4 s 在行自己的后轴系（模型 plan 格式）。
- 有效性：目标的 0.1–4 s 足迹对该行的回放车框（AgentHinge 距离 < 0）或路面栅格（Pacifica 角点 sdf < 0）有接触、t = 0 已接触、或该行缺车框 / 路面标签 → 丢掉（rf_dagger.contacts，rf_rdom 的几何）。
- 权重：失败转弯前 3 s（d319 单元 event / off 的 k_ref − 6 .. k_ref − 1、d318 的失败状态）或路口 / 弯前段（DAGGER-CHECK rows.parquet 的 seg）记 hi，抽取权重 × 3（训练器没有逐行损失权重，
  抽取权重在期望上等价）。
- dev：crc32(log) % 100 < 15 的 log 的行（同前），永不训练；单独建池 on1_JDGdev 做闸读数。
- 今晚的扰动 pass A / B：pass B 的 identity rollout 在 09:20 仍在跑，且要进训练需要 on_rows → 缓存 → 特征整条链（> 30 min），**本轮不用**。

## 3. 训练

- 配方 = RSG-L18（rf_recov / rf_onk 的 REC：`--init RB0-F-s0 --warmup 1 --eval-every 500 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3
  --rm-lr 3e-4 --on-split alpasim_nuplan/onpol-train`），S 行池 on1_L18RB0 每 batch 13 行 `self`，route 分支 lr 规则不变（--rm-lr 3e-4，不随 --lr-scale）。
- 池 on1_JDG = on1_L18RB0（S，self）+ on1_JDGd（DAgger 训练行，`both`：模仿目标 + 两个 hinge），`--on-mode mixed`。
- 剂量：DAgger 行占 128 行 batch 的 {25 %, 50 %} = 每 batch {32, 64} 行（期望值，带权抽取），取代人类模仿行（102 → 70 / 38）；S 13、yaw-rate 13 不变；× `--lr-scale` {0.3, 1.0}（RSG-L18 是 0.1）。
  4 个 pilot：JDGP{25,50}L{03,10}-L18-s0，1 000 步，seed 0，并行。
- 逐位闸：60 步，RSG-L18 配方（JDGG-off）对池 on1_JDG 且 DAgger 行权重 0（JDGG-w0，`--on-mode mixed --on-k 13 --on-weight pool=d:0`）：标量与最终权重 max |diff| 0，
  on_hash_60 = RSG-L18-s0 存的 ea8cacc3e451c577。不过则停。

## 4. Pilot 闸（登记）

1. G-real：dev ADE ≤ 0.56（放宽：本轮有意离开人类模仿；报值）；
2. G-rely 为 true；
3. 同输入等弧长转弯（rf_recov gate_turn，held 决策 0 / 1 的 276 行）pilot − RSG-L18-s0 ≥ +0.3°；
4. dev log 自己状态行上对 DAgger 标签的 ADE 比 RSG-L18-s0 降 ≥ 30 %（失败窗口行、hi 行另报；训练行上同一读数作描述）。
- 过离线闸的 pilot 由闸作业自己提交 held 460 @18 闭环读数（6 块），必须过 H1（对 RSG-L18 两 seed 均值：零分 ≤ +3、at-fault ≤ +3、GT ≤ +0.03 m；按用户的容忍：
  单个晚的、无关的零分算噪声，判读时写明）才能起全量。

## 5. 全量与线

- 过闸 pilot 里最好的剂量（先看 H1，再看 DAgger 标签降幅与转弯）→ 2 seed × 2 500 步（JDG<share>L<lr>-L18-s{0,1}），卡空时加 seed 0 的 5 000 步（…x5k）。
- 读数分阶段：held 460 @18（两 seed）→ 报告 → H1 过才提交公开 1491 @10 护栏（两 seed）+ 盒上 PAI 281 seed 0（与盒上参照 RSG-L18-s0 配对，gate serve、gains 20）→ 报告。
- 线：H1 零分 ≤ +3、at-fault ≤ +3、GT ≤ +0.03 m；H2 held 24 弯两 seed 均值 ≥ +3；PAI 严格 36 完成 ≥ 10（参照 4）、PAI 零分 ≤ 参照 + 3、cor_route_end ≤ 参照；
  护栏均分 CI 下界 > −0.01、零分 ≤ +5。采用：H1 + 护栏 + H2 / PAI 全过。
- 全部 pilot 都不过：不起全量，报告原因。

## 代码

`experiments/alpasim/lib/rule_expert.py`（停止线规则）、`experiments/route_follow/scripts/rf_jdagger.{py,sh}`、`rf_expert.sh --expert-stop`、
`rf_recov_post.{sh,py}`（`submit-held` / `stage2` 分阶段、`ARMROOT` / `RESSUB`、kind J）。盒上 `$DATA_DIR/runs/route_follow/joint_dagger/`，结果 `experiments/route_follow/results/joint_dagger/`。
