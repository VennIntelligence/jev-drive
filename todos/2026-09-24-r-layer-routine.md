# R 层测量：没有突发事件时，各模型的 routine 驾驶能力

状态: draft
执行者: Opus 执行代理（在 GPU box 上）
主题: ../research/capability-vs-leaderboard.md
前提: [zoo 进 CARLA](2026-09-24-zoo-in-carla.md) 的关卡结果（没过关卡的模型只做开环，结果标注 domain 混杂）

## 目标

驾驶能力分两层。**R 层**是 ego-based 的 routine 驾驶：保持车道、直行、转弯、跟车、按信号灯停车、起步。
**E 层**是 routine 过程中出现突发事件时的反应：行人横穿时停还是绕、cut-in、转弯让车，由 [P5 v1](2026-09-24-p5-v1-e-layer.md) 测。
本实验单独测 R 层，回答两个问题：

1. 各模型的 R 层能力排序，和它们的榜单分数排序一致吗？
2. 对 TFv6 这种「接口 + 控制器 + 手写规则」贡献很大的模型，R 层能力里有多少是网络本身的，有多少是配方的？

背景：第 31 条测出，TFv6 用 route + target speed 加作者 PID 开（A 臂），比用它自己的 waypoint 开（B 臂）高约 14 DS；
而 P5 里 target speed 这一路对突发事件几乎不反应（翻转率 2%）。这说明高分很可能主要来自 R 层的配方。

## 场景

**闭环（主）**：Bench2Drive 路线**去掉 scenario**，只保留背景车流，使 E 层事件不出现。
路线用 Dev10 加保留集共 16 条，3 个 TM seed（与第 31 条相同，方便对照）；
如果 CI 太宽，再扩到更多路线，因为第 31 条说明方差主要来自路线之间而不是 seed 之间。

**开环（辅）**：在真实数据上，WOD-E2E 验证集 s_ego 第 1–9 档（ego-only 模型残差不在最高 10% 的帧，即先验能解释的 routine 帧）的 ADE，
口径与第 22 条相同。只对能在 WOD-E2E 上 zero-shot 运行的模型做（openpilot 和 Alpamayo 的 smoke run 里记了它们和 WOD-E2E 输入的缺口），
补不齐输入的就不做，并写明原因。

## 考生

| 考生 | 说明 |
|---|---|
| BehaviorAgent | 特权 expert，R 层上限参照 |
| TFv6 A 臂 | route + target speed + 作者 PID，加作者的启发式（完整配方） |
| TFv6 B 臂 | waypoint + 作者 PID |
| TFv6 A 臂去掉启发式 | 关掉 creeping、stop sign 等作者手写规则，其他同 A；用来量出规则的贡献（这是改配置，不是改模型） |
| openpilot ×3、Alpamayo 1.5 | 用 zoo 进 CARLA 的 wrapper；没过关卡的只报开环 |

## 指标（跑之前写死）

| 指标 | 定义 | 方向 |
|---|---|---|
| RC | Route Completion | 高好 |
| 违规率 | 每公里的碰撞、闯红灯、闯 stop、驶出车道、偏离路线次数（分项报） | 低好 |
| 车道保持 | 相对车道中心线的横向偏差 RMS 和 P95，只在直行和转弯路段上分开算 | 低好 |
| 进度 | 平均速度 / BehaviorAgent 在同一路线上的平均速度 | 接近 1 好 |
| 舒适度 | 纵向 jerk、横向加速度、steer rate 的 P95 | 低好 |
| 停滞 | 非红灯、非前车原因的 ≥ 2 s 停车次数 | 低好 |
| DS | 同一批路线上的 Driving Score，仅作对照 | — |

配对差的 CI 按路线整组做 bootstrap（cluster bootstrap），与第 31 条相同。

## 读法（写于任何数字之前）

- **R 层和 DS 的一致性**：用 Spearman 相关，比较各考生在 R 层综合指标（RC 与违规率）上的排序和他们的 DS 排序。
  两者一致，说明在去掉 scenario 的路段上，DS 主要反映 R 层；不一致的考生逐个解释。
- **TFv6 的分解**：「A − B」是接口的贡献，「A − A 去启发式」是手写规则的贡献，分别给出 R 层指标上的差和 CI。
  如果 A − B 的 DS 差在无 scenario 路段上仍然接近 14，说明第 31 条的 +14 主要来自 R 层；
  如果明显变小，说明它主要来自 scenario 段。
- **真实世界模型**：openpilot 和 Alpamayo 如果在 R 层上接近 TFv6 A，说明 R 层能力在不同类型的模型之间是普遍的，
  融合时 R 层可以放心交给小模型或配方；如果差很多，要先确认是不是 domain 造成的（看关卡结果）。

## 交付物

结果表（考生 × 指标，含 CI）、TFv6 分解表、R 层与 DS 的排序对照图，写在本文件的「结果」一节；
小结果文件放 `research/results/r-layer/`，数字进 decisions。

## 预算

闭环：16 路线 × 3 seed × 约 7 个考生 ≈ 340 个 case。按 benchmarks 文档里 8 实例并发的接法，大约半天到一天。
开环：视能补齐输入的模型数而定，半天。先按 CLAUDE.md 估时，超过 3 小时的部分先做 profiling。

## 结果

跑完再填。
