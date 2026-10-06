# factor_wm × op_parity：从 P2 出发的 on-policy 训练（预登记，2026-10-06，写在任何读数之前）

接第 141 / 143 条（精确自车引擎 + DAgger，S3）与第 144 / 145 条（输入对齐的 P2）。代码 `scripts/fw_p2.py`（引擎、采集、训练）、
`scripts/fw_p2_chain.sh`（一条自推进的链）、`scripts/fw_p2report.py`（check / gate / report）。结果写 `results/p2-onpolicy.md`。

## 0. 问题

输入对齐（P2：ego 状态、4 位姿历史、路线命令；navtest 88.21，HUGSIM HD 0.396 / 0.393）与 on-policy 数据（S3：navhard 35.77，全部增益在 stage 2）
合在一起，能否在 navtest、navhard、HUGSIM 三个读数上同时追平 WA-JEPA（91.71 / 35.41 / 0.451）。主对比 **X − C**：只差 WOD 行的状态来源
（自己访问到的状态 vs 日志状态），隔离 on-policy 的作用。

## 1. 臂（视觉冻结，第 145 条）

| 臂 | 做法 |
|:--|:--|
| P2 | op_parity 全量 P2-F-s0 / s1（W 协议，10 000 步 × 128），已有读数直接复用 |
| C（off-policy 对照） | 从 P2-F-s 续训 2 400 步，batch 128 = 64 navtrain 行（P2 自己的 W 帧行，25 % 锚行，原配方）+ 64 WOD 日志状态行（G1 的 1 069 段 train 片段，同一标签函数，零偏移） |
| X（on-policy） | 同 C，WOD 行 = 40 访问状态 + 24 日志状态。DAgger 3 轮照 G1：r1 用 P2 采 → 训 800 步（r1）→ r2 用 X-r1 采 → 训 1 600（r1 + r2）→ r3 用 X-r2 采 → 终训 2 400（r1–r3），每次都从 P2 出发 |

每臂 2 seed；seed s 从 P2-F-s 出发，与 P2 的 seed 配对。X 与 C 的步数、lr（P2 的 3e-5 / adapter 3e-4，warmup 100，余弦）、navtrain 行、片段、
帧协议、标签函数、蒸馏都相同。C − P2 =「多训 2 400 步 + WOD 日志行」。

## 2. 相对 G1 / S3 的改动（写在任何数字之前）

1. **引擎执行 plan，不执行 action。** 横向：在 plan 上做纯追踪（前视 max(3, min(20, v × 1 s)) m，|κ| ≤ 0.25）；纵向：跟 plan 的 0.5 s 速度，
   a = (v_plan(0.5) − v) / 0.5，截在 [−3.5, 2] m/s²，不经 openpilot 的 lateral / longitudinal 栈。理由：(a) 第 119 条定了纵向读 plan，
   g1-diag H2 显示 S3 的引擎读 action[1] 而 HUGSIM 不读它，这是 S3 卡死的直接原因；(b) P2 的配方把所有非 plan 头（含 action）蒸馏回 shipped，
   训练只动 plan，用 action 驱动的引擎采到的其实是 shipped 的状态分布，不是 P2 的；(c) navtest、navhard、HUGSIM exam（iLQR 跟 plan）执行的都是 plan。
   限制：HUGSIM spec 的横向读 action 曲率，在本配方下约等于 shipped，spec 上的横向不会被 on-policy 数据直接改变。
2. **策略输入**：引擎里 P2 的 ego 输入来自模拟出来的自车：速度、加速度（vy = ay = 0，同 HUGSIM 客户端），4 个历史位姿 = 日志历史（第 0 帧的
   5 s 未来，dg_common.poses_from_fut20）接自车自己的轨迹，命令 = 同一时刻日志帧的 WOD 路线意图（1 直、2 左、3 右 → NAVSIM 一热；0 → 全零）。
   图像是最新 8 个 0.2 s 帧对的 `view_39` token（P2 训练时就是 8 槽、最老一槽置零）。
3. **损失**：P2 原配方（plan 对 8 个后轴位姿模仿 + 非 plan 头蒸馏到 shipped + navtrain 锚行），WOD 行的目标 = fw_train 的目标路径（日志 5 s
   未来 + 横向恢复 t_rec 4 s + 3 s 纵向追赶）取 0.5–4 s 的 8 个位姿。不用 S3 的 action[0] / action[1] 监督，不训 stage 4。
   **标签修正**（实现时发现，写在任何读数之前）：`op_adapt_h.recover_target` 把 > 1 mm 的段当作有方向，停着的日志车厘米级抖动（常是向后）会让法向翻转，路径里出现 2 × dy 的横跳（例：dy 0.5 m 时 2 s 处 y 跳到 −1.08 m、航向 −40°）。本实验用阈值 5 cm 的副本（`fw_p2.recover`），X 与 C 同一函数；G1 的 S3 标签带着这个缺陷，影响面在 check 里量出来写进结果。
4. **源帧仍是时间同步（H1 的泄漏没有堵）**：本实验问的正是 P2 的显式 ego 输入能否抵消它；HUGSIM 起步停滞是闸门读数。
5. 访问状态只在 fw_train 的有效域内（|dy| ≤ 1 m、|dψ| ≤ 5°、|dx| ≤ 5 m、第一个失败事件之前）收标签，同 G1。

## 3. 帧协议

- 训练：navtrain 行是 W（P2 的训练协议，复用缓存）；WOD 行是真实 5 Hz 帧 + plane 重投影（引擎）。X 与 C 完全相同，所以 X − C 不受帧协议混杂。
- navtest：**W 为主**（P2 88.21 同协议、同 devkit），G 为副。
- navhard：**G 为主**（W 让 shipped 掉 5.3；参照 P0-G 33.54、P2-G 30.34、S3 35.77 都是 G），W 为副。
- HUGSIM：仿真器实时渲染，没有帧协议问题；exam 与 spec 都跑。
每个读数里所有臂用同一协议：协议只改变水平，不改变 X − C。

## 4. 读数与判据（全量，2 seed）

单位与 CI：seed 均值，配对 cluster bootstrap 95%（B 10 000）。navtest 12 146 token 按 136 个 log 聚类；navhard two-stage 225 组按 76 个 stage-1 log 聚类；
HUGSIM 64 场景每场景一次，单位 = 场景。WA-JEPA 的 HUGSIM 是单次，两种预设共用。

- **on-policy 有效**（每个读数单独判）：X − C 的 CI 下界 > 0。读数 = navtest EPDMS（W）、navhard combined（G）、HUGSIM HD（exam、spec 各一）。
- **追平 WA-JEPA**（每个读数单独判）：X − WA-JEPA 的 CI 上界 ≥ 0。四个格子（navtest W、navhard G、HUGSIM exam、HUGSIM spec）全满足才写
  「输入对齐 + on-policy 追平 WA-JEPA」。
- 次要（描述，不判）：X − P2、C − P2；navhard stage 1 / 2；navtest 分项；HUGSIM 完成、卡死、起步停滞、打转、fg / bg 碰撞；G 帧 navtest、W 帧 navhard；
  引擎内 g0b 扰动臂失败率（stall / heading / lane）与 g1s 误起步。
- 护栏（任一不过 → 结论写成「换了一种失败」，不算追平）：navtest NC X − P2 ≥ −0.5 pp；HUGSIM fg 碰撞 X ≤ P2 + 3（seed 均值，任一预设）；
  navtrain dev drift_off ≤ 0.10 m；g1s 误起步 X ≤ P2 + 0.05。

## 5. 早停闸门（seed 0 小读，在 seed 1 与全量读数之前）

seed 0 的 X（3 轮）与 C 训完后读：navtest W 全量（每模型打分约 5 min，比抽子集更便宜也更稳；这是对「navtest 子集」的偏离，理由是成本已可忽略）、
navhard G、HUGSIM 12 场景 × exam / spec（`scripts/p2op_hug12.txt`：P2 两 seed 两预设平均 HD 排序等距取 8 个 + shipped 在 spec 下起步停滞的场景
按 P2 HD 等距取 4 个，即 S3 的失败面；规则选、不看 X）、引擎 g0b + g1s。P2 用已有的同 seed 运行。

**停**（任一成立就停线，写负结果，不跑 seed 1 与全量）：
- (a) navtest X − P2 < −1.0；
- (b) HUGSIM-12 起步停滞或卡死 X > P2 + 2（任一预设）；
- (c) HUGSIM-12 平均 HD X − P2 < −0.10（任一预设）；
- (d) navhard G X − P2 < −3.0。
否则继续 seed 1 全链与全量读数。闸门只防伤害，不判 on-policy 有没有用（12 场景太少）。

## 6. 分段放量与检查

1：prep（logged token 存储与标签）+ 预检（采集 `--limit 2`，训练 5 步）。~10：seed 0 第 1 轮采集后 `fw_p2report.py check`（已标注状态 ≥ 20 000、无 NaN、
token 行数对上），再走完 seed 0 与闸门。全量：seed 1 + 全部读数。

## 7. 预算（实测换算：G1 每分片 1 785 条 rollout 约 5 min / 18 核；op_parity 训练 3–7 it/s；HUGSIM 64 每臂每预设约 20 min）

| 步骤 | 量 | GPU·h |
|:--|:--|--:|
| prep（49 对 × 1 069 段编码 + 标签） | 一次 | 0.2 |
| 采集 | 2 seed × 3 轮 × 3 分片 × 约 6 min | 1.8 |
| 训练 | X 3 次 + C 1 次，× 2 seed（800–2 400 步，batch 128） | 2 |
| HUGSIM | 闸门 2 × 2 × 12 场景 + 全量 4 × 2 × 64 | 3.2 |
| NAVSIM plans / 引擎读数 | 分钟级 | 0.8 |
| 合计 | | 约 8 |

CPU：navtest 打分约 10 个模型 × 5 min（24 核），navhard harness 每模型约 10 min（10 核）。墙钟估 5–6 h（3 卡并行）。没有单步 > 3 h 的作业，不需要 profile。
磁盘：日志 token 1.7 GB + 每轮约 7 GB × 6 ≈ 45 GB。

## 8. 结果（跑完填）

| 线 | 结果 |
|:--|:--|
| 闸门（seed 0） | |
| X − C（navtest W / navhard G / HUGSIM exam / spec） | |
| X − WA-JEPA（同上） | |
| 护栏 | |
