# 预注册：RSGbothLon（on-policy 行加纵向安全项），2026-10-13

背景：[results/rsgboth_v/summary.md](../results/rsgboth_v/summary.md) 与 [diag.md](../results/rsgboth_v/diag.md)。RSGboth（RSG 配方，on-policy 行
`--on-mode both`）开到人的速度、修好了慢，但 held 多 7–12 个零分、public guard 多 12 个；多出的零分在直路和缓弯上，约 75 % 发生时自车比人快：
跟车追尾（collision）、人停着它往前蹭（corridor）、弯里跑宽（offroad）。原因：模仿目标带速度，道路 / agent hinge 是 shape-only（沿航向梯度为 0），
on-policy 行上没有任何一项让模型刹车。A30 / W / C2 都没修好。本 arm 给 on-policy 行加三项**只沿航向**的纵向项。
本文件在训练任何 Lon arm 之前写好、提交、推送；修改只追加 Amendment，不改原文。

## 代码

- [experiments/onpol_nuplan/lib/on_lonsafe.py](../../onpol_nuplan/lib/on_lonsafe.py)：三项（docstring 写定义），`--on-lon` 模块契约
  `load(pool_dir, names, dev)` / `loss(poses, rows, pack)`。位姿经 `on_lon.timing_only` 进入：值不变，梯度只留沿各自（detached）航向的分量，
  yaw 无梯度，所以只改速度剖面、不改路径。名字没用 `on_lon.py`：那个文件已是 L 线的 A/B/C/P 分析项（rf_lonread.py、on_train 默认 `--on-lon` 在用）。
  - **lead**（m²）：pool `agent.npz` 每个 actor（t 与 t+0.5 s 都有效）按 t 时刻速度匀速外推；plan 自己的路径（detached，末端沿 yaw 延长 20 m）
    为走廊，box 角点离路径 ≤ 1.0 m 或跨路径即在走廊内；只取 t=0 时整个 box 在自车前保险杠之前的。要求 plan 前保险杠站位 + g ≤ actor 最近角点站位，
    g = 2 m + 0.8 s × v_plan，若本行起点已更近则放宽到 max(起始间隙 − 0.5, 0)。overshoot 平方，actor 取 max，8 个时刻取均值。
  - **curve**（m²/s⁴）：plan 路径（detached）曲率 × v_plan²，超过 3.0 m/s² 的部分平方，k = 1..7 取均值；只有 v 有梯度。
  - **stop**（m²）：目标 `fut` 4 s 弧长 < 2 m 的行，plan 4 s 弧长超过 fut + 2 m 的部分平方。
  - 单元测试 `python on_lonsafe.py --smoke`：10 m 处静止前车 / 3 m 外邻道车 / 20 m 外同速前车；R = 10 m 弯 8 m/s 与 4 m/s；停着的人。只在该响的行上响，
    lead / stop 每个位姿的下降方向都朝后，curve 净下降朝后，一小步下降后各项与 4 s 弧长都变小，横向与 yaw 梯度为 0，权重 0 时不返回任何项。PASS。
- [experiments/route_mem/scripts/rm_train.py](../../route_mem/scripts/rm_train.py)：`--lon-lead / --lon-curve / --lon-stop`（默认 0 = 不加任何运算）；
  任一 > 0 时设 `--on-lon on_lonsafe --on-lon-lam 1`，每项以 W × ho_w(3) × Σ行 / n_imit 进 total（与 agent / road hinge 同一归一）。
- [scripts/rf_lonprobe.py](../scripts/rf_lonprobe.py)：各 checkpoint 在 pool 状态上的 plan 上算三项（发火率、按 rclass 分），以及第一个 tag 的
  沿航向梯度比。[scripts/rf_lon.sh](../scripts/rf_lon.sh)：训练 + 读的链。

## 发火率与权重（训练前已看，rf_lonprobe.py，pool on1_L18RB0，17 429 行；`$DATA_DIR/runs/route_follow/lon/probe/`）

| plan 来源 | lead | curve | stop | 任一 |
|:--|--:|--:|--:|--:|
| 目标 fut 本身 | 6.5 % | 1.3 % | 0 %（定义上） | 7.7 % |
| served（RB0-F-s0，起点） | 9.2 % | 0.0 % | 1.9 % | 10.5 % |
| RSGboth-L18-s0 | 9.9 %（直 8.9 / 弯 9.5 / 转弯 14.6 %） | 0.01 %（2 行） | 2.0 % | 11.3 % |
| RSG-L18-s0 | 9.0 % | 0.02 % | 1.9 % | 10.3 % |
| RSGbC2-L18-s0 | 9.3 % | 0.01 % | 2.1 % | 10.7 % |

- lead 落在目标区间 10–30 % 的下沿；RSGboth plan 发火的 1 731 行里 826 行目标 fut 自己也发火（匀速外推的前车 / 横穿车，或恢复路径本身贴得近），
  这些行上模仿项与 lead 项方向相反。overshoot（√h）四分位 0.6 / 1.7 / 5.3 m，90 % 分位 11 m。
- **curve < 1 %：几乎不响**（3 m/s² 门槛下 plan 从不超；目标 fut 在转弯行上 7 % 超）。和 RSGB 的停止线项一样是空项：弯道 offroad 不是 plan 层面横向加速度过大造成的。
  照样带上（权重无关紧要），但预期它对零分不起作用。
- stop 2.0 %（停着的目标只占 2.7 % 行），低于 5–10 % 的目标，> 1 %。
- 权重：发火行上 w_r = |∇_沿航向 模仿| / (ho_w × |∇_沿航向 项|) 的中位数（RSGboth-L18-s0 的 plan）：lead 0.054 [IQR 0.023, 0.134]，stop 0.025 [0.015, 0.052]，
  curve 18（只有 2 行）。登记 **lead 0.05、curve 10、stop 0.025**：发火行上纵向推力的中位数约等于模仿项的沿航向推力（1×）。平方 hinge，overshoot 大的行推力更大。

## Arm

**RSGbothLon-L18-s0** = RSGboth-L18-s0 的原命令（pool on1_L18RB0，`--init RB0-F-s0`，2500 步，`--on-mode both`，seed 0）+
`--lon-lead 0.05 --lon-curve 10 --lon-stop 0.025`。pool 任务，卡 1–5，60 步 preflight smoke（同命令 `--steps 60`）。通过则训 s1（同命令 seed 1）。

## 读

held 460 @18（held.py 共同 scene，ol_confirm 配对），rf_follow 24 个 held 转弯完成率与 outcome，rf_inside，public 1491 @10 guard（ol_confirm），
zeros_mech collect + rf_rsgbv.py diag（零分类型 / 路段类别 / 前车 / 速度 vs RSG）。对照：RSG-L18-s0、RSGboth-L18-s0、RSGbC2-L18-s0 的已存读数（s1：RSG-L18-s1、RSGboth-L18-s1）。

## 线（同 seed 对 RSG-L18；全部满足才算过）

1. held 零分 ≤ RSG + 3（s0：≤ 37 / 460）；
2. public guard 零分 ≤ RSG + 5（s0：≤ 51 / 1479）；
3. held 均值 ≥ RSG（s0 0.8656）；
4. 转弯完成率 ≥ 85 %（≥ 21 / 24）；
5. 「太慢」（rf_follow `short` outcome）≤ 2。
单 seed 过线是第二个 seed 的候选，不是结论；s1 用同样的线对 RSG-L18-s1。
