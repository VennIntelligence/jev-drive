# op_dagger 预登记：重投影 rollout 上的 DAgger 式微调（pilot，2026-10-06）

写于任何训练与 held-out 读数之前。引擎检查（零扰动复现、d123 步 1 增益复现）是训练前的前置步骤，结果另记在 results/pilot.md，不改这里的读数和线。

## 问题
论文 §3 图 8：把几何重投影当 rollout 引擎，在真实 WOD 日志上让 openpilot（shipped Cinque）闭环走 1–2 s，在它自己访问到的状态上打「回到日志路径」的标签再微调，能不能缩小只在闭环里出现的两个缺口：
(i) 起步 / 低速的偏航环路增益（d100 / d111 / d123：真实 WOD 起步上步 1 增益 7–9 °/°）；(ii) 横向 / 朝向偏离后的修回（d123：2 s 内没有恢复响应）。
对照：同样的状态、同样的标签，但历史是脚本化漂移（第 3 层静态 O 配对，d98 已做过的那种），只差「历史是不是模型自己闭环走出来的」。

不做：路口选路、纵向起步。

## 引擎（experiments/op_dagger/scripts/dg_common.py, dg_roll.py）
- 片段：20 帧 5 Hz（10 帧历史到 t0 + 10 帧日志未来），WOD-E2E 前三相机渲染成 openpilot road / wide（op_adapt_h 的渲染器）；日志位姿由 t0 的 5 s 未来轨迹三次插值、朝向取弧长弦向（在 d123 的 sceneflow 精确位姿上测：行驶中逐片段最大朝向误差中位 0.20°、p90 0.87°；起步 0.17° / 0.50°；位置 < 1 cm）。
- 步进 0.2 s：port（op_adapt_l LModel / rft RModel，与 ONNX 逐帧一致）→ action[0] / max(1,v)² → lib/op_ctrl.OpLateral（latActive、clip_curvature、lateralDelay 0.2 s，t0 同步到日志曲率）→ 以日志速度、日志位移按朝向偏差旋转 → 与日志位姿之差 (dx, dy, dψ) → 把该时刻的日志帧重投影到这个偏差（op_interp.warp_frame）→ 历史缓冲 → 下一步。
- 有效范围（d123）：K = 10 步（2 s），|dy| ≤ 1 m，|dψ| ≤ 5°；超出后的状态不进训练，读数里报告出界比例。起步片段是重投影最弱的地方（d123），单独报告。

## 数据
- train：WOD-E2E train（split `wod/r2-train`）320 个片段：起步 120（静止 ≥ 1.8 s 后第一个运动帧）、低速 0.5–3 m/s 60、中速 3–8 m/s 60、巡航 ≥ 8 m/s 80；2 s 内日志转角 ≤ 15°；每个序列最多 2 个、相隔 ≥ 4 s。
- heldout：WOD-E2E val（`wod/val`）120 个片段（起步 40、低速 20、中速 20、巡航 40），同样规则，从不进训练。
- sf：d123 的 10 段 14 个 anchor（引擎检查用）。

## 采集（DAgger）
每个 train 片段 5 条闭环 rollout：free（不注入）、kick ±（步 1 朝向突变，起步 / 低速 |δ| ~ U(0.5, 3)°，中速 / 巡航 U(1, 4)°）、swerve ±（步 1–3 外加朝向，结束时偏 |0.2–0.8| m；速度太低不可行时换成另一个 kick）。访问状态 = 步 1..10 且在有效范围内。标签：恢复路径（op_adapt_h.recover_target，与第 3 层 O 配对教师同构）；action 目标按 shipped 头自己的尺度（rft.act_target，v < 1 m/s 不监督 action）。

## 训练臂（单 seed，pilot：400 步、batch 48，约 it_dw3 的 1/6 步数、1/10 数据）
配方 = op_route_ft rft 去掉路线适配器：stage 4 + plan 通路 + action 通路可训，stage 1–3 冻结；每批 R 20（访问状态）、U 12（日志状态模仿）、D 16（日志状态蒸馏到 shipped，dw 3）；shipped 教师在线。
- `dg1`：R 行 = shipped rollout 的访问状态，历史 = rollout 本身的重投影帧。
- `st1`（对照）：同一批状态、同一标签，历史 = 日志帧沿脚本化漂移重投影到该偏差（op_adapt_h drift）。
- `dg2`：用 dg1 在同样 train 片段上再采一轮，与第 0 轮状态合并，从 shipped 重新训（同步数）。
只做一到两轮迭代。

## 读数（heldout，按片段；CI = 按序列的 cluster bootstrap，10 000 次）
- **G1 步 1 开环偏航增益**：步 1 起朝向 ±2°（不闭环），(phi1(+2) − phi1(−2)) / 4，°/°；同时报 action 曲率增益 gk1–3（1e-3/m 每度）。起步组为主读数。
- **K10 闭环 kick**：步 1 朝向突变 ±2°，闭环到 2 s，符号校正后的朝向偏差 / 2°（> 1 放大，< 1 修回）。
- **S10 闭环 swerve**：±0.5 m 横向偏移（速度可行的片段），2 s 时横向偏差 / 步 3 时横向偏差（< 1 修回）。
- **护栏**：free 闭环 2 s 时 |dy|、|dψ|；日志状态（replay）上 phi1 / action 曲率与 shipped 的平均绝对差；出界比例。

## 线（写死，不看结果改）
- L1（起步增益）：dg1 起步组 G1 ≤ 0.5 × shipped，且 dg1 − st1 的 CI 上界 < 0。
- L2（修回）：dg1 行驶组 S10 − shipped ≤ −0.2 且 CI 上界 < 0；dg1 − st1 CI 上界 < 0。K10 同方向报告，不设线。
- L3（不伤）：dg1 free |dy| − shipped 的 CI 上界 ≤ +0.1 m；replay |Δphi1| 中位 ≤ 1°。
- 结论规则：L1 或 L2 过且 L3 过 → 闭环 rollout 有用，进下一步（HUGSIM spec 11 场景 + op_guard 子集，导出 ONNX，按 GPU 情况）；dg1 过而 st1 也过且两者 CI 不分 → 「标签起作用、rollout 不添加东西」；都不过 → 记录为 pilot 未过线，§3 图 8 不成立的证据之一。dg2 对 dg1 只做描述。
- HUGSIM / guard：只在 L1 或 L2 过线时跑（2026-10-06 GPU 让给 p7 的 B2D lane，按租约排队）。

## 已知限制
单 seed；400 步 pilot；起步重投影失真（d123）；action 头尺度约为日志曲率的 0.45（rft ACT_ALPHA），所以闭环本来就欠转，free rollout 的漂移部分来自这个，不全是误差；朝向估计在行驶中 p90 0.87°。
