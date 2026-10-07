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

## 执行声明与偏差（2026-10-07，打分前写定）

状态：用户已批准，按本登记执行（执行者 replay-hinge lane）。代码 `lib/lqr_proxy.py`（跟踪器代理）、`experiments/op_parity/scripts/rh.py`（proxy / train / merge / report），打分沿用 `experiments/op_probe/scripts/opb_score.py`。

1. **代理验证的读法**：300 个 token = navtest 中 P2H10-F-s0 存档 plan 的随机 300 个（`default_rng(0)`）；「回放足迹差」= 每 token 41 个状态 × 4 角的角点位移最大值，取 300 个的 p95；「DAC 判定一致」= 代理足迹是否出 scorer 可行驶多边形（ROADBLOCK / INTERSECTION / DRIVABLE_AREA / CARPARK_AREA）对 devkit 存档的 P2H10-F-s0 navtest DAC 子分。代理输入 v0 / a0 取 op_parity cache tab 的 ego status（训练时看到的同一来源），转向 0（devkit `ego_status_to_ego_state` 即如此）。全 navtest 同时报告，只作补充。devkit 把时间戳取整到 µs 带来的 ~1 cm 数值差不复现。
2. **decoder 特征**：「P2 plan-head hidden」= 第 147 条 addendum 1 用的 P2-F-s0 的 H（add_54）+ E，非 P2H 的 hidden（同一冻结视觉）；训练行 navtrain s2–s4 去掉 dev logs（25 415 行），4000 步、batch 512、AdamW 1e-3、cosine，与 op_probe `decode` 相同；CPU；三臂各一个进程，同 seed、同 batch 序列。对照 = addendum 1 的 P2-H|hinge 配置（λ 10、余量 0.3、8 个原始位姿线性插值）。
3. **转弯 token** = 日志 4 s 航向变化 |Δψ| > 20°（RM 训练掩码与闸门分层同一定义）。
4. **闸门读法**：DAC 降幅按 paired 差（臂 − 对照，转弯 token，log 聚类 bootstrap B 10 000）的点估计 ≥ 0.4 pp，CI 只报告；「整体 per-token 分数不降」= 全 navtest paired 分数差（opb_score 的 per-token EPDMS 去掉 EC，reactive 交通）95% CI 上界 ≥ 0（即不显著下降）。R 与 RM 都过时，胜者 = 转弯 DAC 降幅大者。评测集 = 全 navtest 12 146 token。
5. 后续若进 HUGSIM：除登记的 64 场景外，按第 149 条加评 `spec_plan_smooth`；B2D 不在范围内。

## 执行结果（2026-10-07）

代理验证过（p95 1.0 cm，DAC 一致 100%）→ 离线闸门过（RM 转弯 DAC 失败 −4.63 pp，胜者 RM）→ pilot 过（RMP-F-s0 对 HP-F-s0：DAC 失败 −0.40 pp，EPDMS +0.43）→ 全量 RMH10-F 两 seed 已跑（navtest +0.52、navhard +0.25 n.s.、HUGSIM `spec_plan_smooth` +0.002 n.s.）。结果页 [results/replay_hinge.md](../results/replay_hinge.md)。
过程性偏差（不改判据）：decoder 打分因池 CPU 预算拆成 4 个 token 分片；navhard 报告时把 hinge lane 的 P2H10 harness 目录软链到 bench 查找的 `navhard_gimm/harness/`；HUGSIM 报告不含 WA-JEPA（它没有 `spec_plan_smooth` 运行，四方向页的 0.451 来自其自身 client）。
