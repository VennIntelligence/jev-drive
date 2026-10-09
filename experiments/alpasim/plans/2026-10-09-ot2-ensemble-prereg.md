# OT2-C 预登记：共享冻结 encoder 上的 adapter ensemble（2026-10-09，任何 ensemble 闭环分数读出之前写定并 push）

lane OT2，piece C，decision 211。接 d201 第 7 点：同配方两个 seed 的逐 scene 事后取最大（OT30-F-s0 + s1）比较好的那个 seed 高 +0.0133，零分落在不同 scene 上。只有本地 run，不向 AlpaSim 提交。

## 问题

encoder 冻结且各 checkpoint 相同，N 个 adapter seed 共用一次 encoder 前向，各自跑 policy，把 plan 输出等权平均，几乎不加时延。平均能拿到 seed 间失败不重合这部分空间的多少。事后取最大是上界，平均可能一点都拿不到。

d202 的仲裁（在两个 driver 之间按规则选一个）已经停了；这里不是选择，是输出平均，没有门控，没有重测那条。

## 做法

- 新 driver `experiments/alpasim/lib/ens_driver.py`（`run.sh <dir> ens`，`ENS_TAGS=a+b`），不改 lane M1 正在改的 `sh30_*` / `ap2_*`：成员是 NAVSIM 标准的 tag（SH30、OT30）时包在 `sh30_driver.Driver` 外，是 AlpaSim 标准的 tag（AP2、APO）时包在 `ap2_driver.Driver` 外；输入标准或冷启动规则不同的成员拒绝混合。平均的是 policy 的 plan 输出（33 × 15，导出到后轴之前）。启动时逐个核对成员的冻结权重相同。
- 每次决策把各成员自己的 8 个位姿写进 `members.jsonl`（同一状态上的成员分歧）。
- 集合：E1 = `OT30-F-s0 + OT30-F-s1`（现在跑）；E2 = piece B 最佳配置的两个 seed（B 出结果后）；E3 = 同一输入标准下的跨配方混合（如 `AP2-AB-s0` + B 最佳配置的一个 seed；AP2 与 OT30 输入标准不同，不混）。

## 闸门（读分数前）

- 身份：`ENS_TAGS=OT30-F-s0`（单成员）在 pilot 8 scene 上的分数与 d201 的 `OT30-F-s0` 逐 scene 相同（仿真确定）。
- 健康：每个 session 10 次 `drive`、10 次真实推理、0 错误。
- 时延：`ens_driver.py bench` 同进程同卡量单成员与 ensemble 的单步中位数；闭环 `drive.jsonl` 的中位数作为带负载的读数。目标仍是 0.1 s。

## 线

ensemble **被采用**，当且仅当：700 scene（d201 的固定列表）上 mean scene score 减去其成员中较好者（700 scene 均分高者）**≥ +0.008**，按 log 整簇的配对 CI 下界 **> 0**，且 at-fault 事件数（`offroad_or_collision_at_fault` 之和）**不高于**该成员。

停止规则：E1 的点估计 < +0.004（线的一半）就停，不跑 E2 / E3，照实报告「平均拿不到」。E1 ≥ +0.004 才在 B 的最佳配置上跑 E2（同一条线，对 E2 自己的较好成员），E3 只在 E2 过线时跑。

## 失败时要回答的问题（事先定义）

seed 间的失败是 plan 层面的分歧被平均抹平，还是分叉被平均弄得更糟：

- 「分裂 scene」= 恰有一个成员零分的 scene。报告 ensemble 在这些 scene 上的零分数（全抹平 = 0，全继承 = 分裂 scene 数），以及在无成员零分的 scene 上新增的零分数。
- 同一状态上的成员分歧（`members.jsonl`）：两个成员 4 s 终点的横向差。某次决策横向差 > 1.0 m 记为「分叉决策」（平均落在两条都没选的路径中间）；≤ 1.0 m 记为「小分歧」。按 ensemble 自己的零分 scene / 非零分 scene 报告分叉决策的占比，以及 ensemble 新增零分 scene 里零分前是否出现过分叉决策。
- 读法：ensemble 零分集中在有分叉决策的 scene 上 → 平均把分叉弄糟；分裂 scene 上 ensemble 多数不零分而总分不升 → 分歧被抹平但新失败在别处出现；两者都不是就照实写。

## 成本

E1：pilot 8 scene + 700 scene ≈ 1 job-hour；E2 同量；bench 几分钟。合计约 2 card-hours。

## 限定

- 成员只有 2 个 seed；700 scene、27 个 log；本地渲染。
- 平均两条 plan 假定它们在同一模态里；d199 / d201 的 oracle 增益若主要来自模态不同的 scene，平均没有理由拿到。
