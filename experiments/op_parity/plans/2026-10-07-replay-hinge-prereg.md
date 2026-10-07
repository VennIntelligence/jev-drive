# op_parity：跟踪器回放足迹上的 drivable hinge + 转弯前角余量（replay hinge）预登记草稿（2026-10-07，未启动，待审）

## 问题与依据

[results/four_dirs.md](../results/four_dirs.md) 方向 1、2（navtest，P2H 特有，WA 只挂同一 token 的 17–20%）：

- 急弯（R < 15 m）DAC 失败 54% [42, 65] 在弯内侧：前角在 3.4 s 附近切进路沿，0.40 m 深，66% 原始 plan 就出界；「转不过去」只 26%（0.23 分）。失败 plan 的航向增益 1.04，终点比日志偏内 0.17 m。
- 宽弯 / 缓弯（R ≥ 15 m）DAC 失败 62% 是 < 0.3 m 的擦边，42% 只在 devkit 的 LQR 回放里出界，内侧 53%（WA 28%）。
- 现行 hinge（`lib/drivable_hinge.py`）只看 8 个原始位姿的线性插值，余量 0.3 m，看不见 LQR 回放的轨迹，也不对转弯前角加余量。
- 相关上限（(a) 替换为 WA）：D1 内侧 0.60、D2 内侧 0.62、D2 仅回放 0.38，合计约 1.3（有重叠）。第 147 条：约一半失败在冻结视觉上，本登记只动 head 目标那一半。
- turn-train（[results/turn_train.md](../results/turn_train.md)）已排除训练分布 / anchor：> 45° DAC 失败 −0.03 pp。

问：**把 hinge 换到 devkit 跟踪器回放后的足迹上、并在转弯 token 上给前角 0.5 m 余量，能否收回内侧切角与仅回放擦边。**

## 设计（写代码前定死）

- **可微跟踪器代理**：按 navsim devkit（`navsim main @0a380a9`）的 LQR 跟踪器（`LQRTracker` 参数与 0.1 s 离散、运动学自行车模型）在 torch 里逐步实现，增益为常数，rollout 4 s，输出 41 个位姿的足迹。验证：300 个 navtest token 上，对 P2H 原始 plan 的回放足迹与 devkit 回放差 ≤ 0.1 m（p95），DAC 判定一致率 ≥ 99%；不过则停在此处报告。
- **R**：hinge 作用在回放足迹上（替换原始位姿版），余量 0.3 m。
- **RM**：R + 在日志 4 s 航向变化 > 20° 的 token 上，前两角余量 0.5 m（后两角 0.3 m）。
- λ = 10，与 P2H10 相同；数值不调。

## 第一步：离线闸门（CPU，无训练）

沿 op_probe `decode` 路径（第 147 条 addendum 1 的 thin decoder，接 P2 plan-head hidden + ego），同 token 同配置训三个 decoder：原始位姿 hinge（现行，对照）、R、RM；全 navtest 用 devkit `pdm_score` 打分（`opb_score.py`）。

- **过**：转弯 token（|Δψ| > 20°）DAC 失败率 RM 或 R 比对照降 ≥ 0.4 pp（log 聚类配对 bootstrap），且整体 per-token 分数不降 → 进 pilot。
- **停**：两者都 < 0.4 pp → 停，报告（内侧切角不是目标几何问题，归表征）。

## 第二步：pilot（仅离线过时；GPU，需再确认）

HP-F pilot 配方，seed 0，把 drivable hinge 换成离线胜出的 R / RM；navtest 全量（bench）。判据：DAC 失败比 HP-F-s0 降 ≥ 0.3 pp 且 EPDMS ≥ +0.2 → 全量 P2H10 配方 2 seed + navtest / navhard G / HUGSIM 64 `spec_plan_smooth`；否则停。

## 代价与时间

跟踪器代理与验证约半天 CPU 开发；离线 decoder 三组 CPU 约 1 h；pilot 训练约 10–20 min + 打分约 10 min；全量 < 3 h。经 GPU 池与 `jevdrive.bench`。

## 限定

devkit 跟踪器是 scorer 的一部分，降低「仅回放」擦边是让 plan 适应评分器的执行层（对闭环真实控制器未必同样有用，HUGSIM 的 iLQR 不同）；约一半的弯道 DAC 失败在冻结视觉上（第 147 条），本线最多收回 head 那一半；HUGSIM 方向 1 是入弯速度问题，本线预期不改 HUGSIM。
