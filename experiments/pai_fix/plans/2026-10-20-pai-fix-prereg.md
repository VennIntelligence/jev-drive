# PAI-FIX 预登记：定向 PAI 小考（停车越点 + 路线转弯，小护栏），写于任何考题读数之前（2026-10-20）

## 目的

下一步要训 ~3 个便宜变体（站位锚定的纵向目标：只停车 / 只转弯 / 两者）。全量 PAI 281 一次约 1 卡时，太贵。要一个
**远低于 0.5 卡时**、可信的筛子：先回答主问题（NTV0-L18-s0 + ACC + GAP 的两类主要失分有没有被修：(a) 越过人的停车点，
无前车的停止线 / 信号，第 381 条 23 个零分；(b) 入弯过快、欠转，直行过弯 11 + 路口 / 弯道出路），选出 1–2 个候选；其他地方有没有坏，
只对这 1–2 个跑全量 281 再查（用户 10-20 的范围说明）。所以 TARGET 是主体、尽量大；GUARD 很小，只做早期报警。

## 场景来源与不相交规则

- **主体 = val160**（curated-val 441 去掉 test281 的 160 个，盒上都有 NuRec 场景）。不在 PAI 281 里；不在 NTV0 / RB0-F / RSG 任何训练池里
  （第 404 条：P160 的行不在三个头的池里；第 367 条 PRACTICE-PAI 用过它训了一个 1 000 步 pilot，非候选，已关）。
- **d402 的 curated-569（盒上 570 个 curated-train）是训练数据，不用**；它们与 val160 / test281 没有同一驾驶（`nurec_select.csv` 的 exam_gap_s 全是 inf，
  即 600 s 内同 session 时钟、可开到的距离内无 val160 / test281 场景）。盒上另有 284 个 curated-train 场景（下载器 10-20 白天还在按
  `nurec_select.csv` 顺序继续下），视为下一批训练数据，也不用。
- **反向排除**：训练方必须排除考题场景，以及与考题 val160 场景同一驾驶 / 同一路段的 curated-train 场景：`scenes/train_exclude.csv`
  （`pai_fix.py drives`，pav.py 的同驾驶规则：session 时钟差 < 600 s 且起点距离可开到；同路段 = 轨迹点 10 m 内）。
- 不够时加 PAI 281 已知失败场景，作**单独一列**（T281，按 A0 失败挑的，有选择偏差），**永不混进主分**。

## 选取规则（只用人的日志事实，与任何待测模型无关；`pai_fix.py select`）

人的事实来自 rf_tax 抽取 + `pai_gap.human / rest_cue`（第 381 条同一代码）；转弯角来自 `exam_nurec.csv`（pav.py 的 6.4 s 窗口）。

- **STOP**：handoff 时人 >= 1 m/s，片段结束时人静止，静止点前方 15 m 内无日志车（第 381 条越点人群；281 上同规则 22 个，A0 越点 17 个）。
- **STOP-L**：同上但静止点前 15 m 内有车（停在前车后；281 上 23 个，A0 越点 3 个，其余多是好的 → 兼作「停在前车后」护栏）。
- **TURN**：非以上两类，人在 6.4 s 内转 >= 25°，且该窗内中位速度 > 1 m/s（pav 的 turn_mid | turn_sharp；281 上同规则 66 个，覆盖 A0 的 23 个转弯 /
  路口失败中的 22 个）。
- **GUARD**（小）：非 TARGET 且六个已存 val160 读数（RSG-L18-s0/s1、RB0-F-s0/s1、RSGk51w1、RSGlr03n5000，全部 ACC + GAP）都 >= 0.98；
  G-LAUNCH = handoff 时人 < 1 m/s 且日志移动 > 5 m；G-LEAD = 在动、不在片段末静止、行进 0.5 s 步中 >= 30 % 有日志车在人的路径 30 m 内。
  各取 6 个（按 scene id 的 crc32 排序）。NTV0 在 val160 上没有读数，所以「NTV0 现在能过」用这一家族代替。
- **T281（单独一列）**：A0（NTV0-L18-s0 + ACC + GAP，第 366 / 381 条存档读数）零分里 cor_route_end 23 个 + 转弯 / 路口 / 弯道类 23 个
  （cor_straight_at_turn 11、cor_turn_under 3、cor_turn_over 1、off_junc_* 7、off_curve_inside 1）。

结果（`scenes/exam.csv`）：val160 STOP 12、STOP-L 12、TURN 43（TARGET 67）、G-LAUNCH 6、G-LEAD 6；T281 46。
固定 chunk 表 `scenes/chunks/`：v000–v002（val160，26–27 个）、t000–t001（T281，23 个），s000 = 1 个场景的冒烟。模拟器只在同一 chunk 表内复现，
所以以后每个读数都用这几张表（`pai_fix.sh read`）。

## 每次读数的指标（`pai_fix.py report`；各子集分开报）

均值；零分按类：overshoot（cor_route_end）/ route-turn（cor_straight_at_turn、cor_turn_*、cor_wrong_exit）/ offroad / AF-front / side（AF-side、AF-rear）/
zero-other；部分分 rear-hit / slow（slow、creep、no-launch）/ hit；起步数（人 handoff < 1 m/s 且日志移动 > 5 m 的场景里，自车走了 >= 2 m 且 >= 日志距离 30 %）。
配对差（对参照读数）：scene bootstrap 95 % CI（jevdrive.stats，10 000 次，seed 0）。

## 校准（先用已存读数，0 计算；`pai_fix.py calib`）

同一规则用在 test281 上得到「281 派生考题」，对 15 个已存全量 281 ACC + GAP 检查点（RSG、JTN s0/s1、NTV0 s0/s1、NTVX、NTVXS、BNC2K、JTXS、
B s0/s1、B2 s0/s1、CPAIC1K、CPAIP1K）看：子集均值与全量均值的秩相关、两两符号一致率；噪声底 = 种子对（NTV0 / JTN / B / B2 的 s0 − s1）与 A0 重滚（R0）；
T281 上每个读数「修好」A0 零分的个数（偶然水平）；同一检查点原生（VC + LEAD）对 ACC + GAP。

## 参照读数（本轮要跑的 GPU）

全部经 pool，`pai_box.sh` 原样（ALPASIM_FAST=1、CONC 4、g10 转向权重、视频关）；分阶段：s000 冒烟 1 个场景（ag 与 vl 各一）→ 查打分 → 全部。
1. NTV0-L18-s0 + ACC + GAP（ag），v + t。T281 部分与存档 A0 对比 = chunk 组成不同带来的差（存档 R0 重滚在同一 chunk 表上逐分相同）。
2. NTV0-L18-s0 原生 VC + LEAD（vl），v + t。
3. NTV0-L18-s1 + ACC + GAP，只 v：val160 上的种子噪声底（T281 上的种子噪声已由存档给出）。

预计：一次 v + t 读数 125 个场景，约 37 s / 场景 / stack（A0 的 chunk：约 35 个场景 1 320 s）+ 每个 chunk 约 75 s 固定开销 ≈ 1.4 stack 时，
按每卡 3 个 stack ≈ 0.45 卡时；只 v（79 个）≈ 0.9 stack 时 ≈ 0.3 卡时。实测记进 README。

## 用于变体的筛选读法（提议，主会话定；本轮只给参照行，不判任何变体）

- 主读数：val160 TARGET 对 NTV0-L18-s0 同 serve 的配对差，及 overshoot + route-turn 零分数；以 NTV0 s0 / s1 在 val160 上的差作噪声底。
- T281：修好 A0 的 46 个零分中几个（存档偶然水平：15 个检查点每个 0–5 个，NTV0-s1 0 个）；新零分数。
- GUARD：G-LAUNCH / G-LEAD 均值与起步数（所有起步场景）、rear-hit 部分分。只报警，不作通过线。
- 进全量 281 的 1–2 个候选：主读数与 T281 都高于噪声底、GUARD 不报警者。

## 已知局限

- val160 的 STOP 只有 12 个场景，TURN 43 个；每类失败数十个上下，单个候选的差 < 2–3 个场景时分不开。
- GUARD 的「通过」由 RSG / RB0 家族代替 NTV0 定义；NTV0 参照读数出来后如实报告它在 GUARD 上的分。
- T281 是按 A0 失败挑的（选择偏差，回归均值方向对变体有利）；只作单独一列。
- 渲染场景与真实 PAI 私榜不同；一个 seed。
