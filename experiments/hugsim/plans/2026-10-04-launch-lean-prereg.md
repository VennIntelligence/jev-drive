# HUGSIM 起步偏向与闭环放大：小偏航率下的 G、偏向来源、闭环重放（预登记）

2026-10-04 写，跑任何数之前。依据：第 90 条（PR #57 下 Cinque 低速起步打转，起步计划偏 1–2°，20 次里 18 次向左）、
第 92 条（10°/s 假偏航下的 G）、第 96 条（闭环每步约 ×1.5 放大；低速去旋转切断）、第 98 条（适配把 10°/s 的 G 削掉
71–78%，HUGSIM 打转 10 → 15）。卡 2，离线为主，不跑大批 HUGSIM。

## 已知的事实（读日志，不算结果）

- Cinque 的输入只有图像、desire、traffic_convention、action_t；**没有车速 / ego status 输入**。HUGSIM 适配里车速只进
  decode（curvature），不进计划几何。所以「初始速度」只能通过图像里的运动（起步前 5 s 的静止预热帧）进入模型。
- HUGSIM 的 64 个考试场景里，nuScenes 的新加坡场景用左行 traffic flag（[0, 1]），其余右行。
- 考试日志里 step 0（5 s 静止预热之后）计划已有 ±0–3° 的偏向（10 s 终点方向），step 1（第一帧前进，历史里还没有
  偏航）在部分打转场景里跳到 13–43°。所以「起步偏向」主要出现在第一步运动时，不只是静止帧上的固定偏。

## 1. 小偏航率下的 G

**做法**：第 92 条的构造（op_adapt_h 的 port 实现，`hist_rot`：历史帧按 ω·t 旋转，t0 帧不动），ω = ±0.5、1、2、3、5、
10°/s，模型 shipped（O）、pilot-s0、it_dw3-s0，pnav / pwod / pcarla 的探针样本，stop / low / mid 三档每档随机最多 250 个
（固定种子）。G(ω) = (ψ3(+ω) − ψ3(−ω)) / 2，增益 g(ω) = G / (1.5 ω)（每度历史偏航带来的 3 s 计划朝向，度/度）。
CI：cluster bootstrap（log / sequence / route）。

**补充（HUGSIM 帧上）**：在 HUGSIM 10 个打转场景 + 10 个不打转场景的起步帧上做同样的旋转（`OpenpilotFrames.rot_index`，
对三路相机是精确旋转），ω 同上，三个模型（shipped TRT、pilot / it_dw3 的 serving ONNX），读 1 s 计划方向（闭环控制器
跟的量）和 3 s 朝向。

**判读（登记）**：
- 「适配的削减与偏航率有关」= 小偏航率（0.5–2°/s）下 pilot 相对 shipped 的 g 比值的 CI 上界 > 0.6，而 10°/s 下比值
  < 0.4（第 98 条的 71–78% 削减）。否则判「削减与偏航率无关」。
- 闭环环路增益：用 HUGSIM 帧上的 g（1 s 方向 / 历史偏航）乘以从考试日志估出的控制器传递（下一步偏航变化 / 当前 1 s 计划方向），
  代入线性递推 θ_{k+1} − θ_k = c·φ_k，φ_k = λ + s·(θ_k − θ_{k−6})，取主特征根作为每步增长倍数。shipped 的预测值落在
  [1.2, 1.8] 内算与第 96 条的 ×1.5 一致；适配模型的预测值 > 1 即「仍然放大」。

## 2. 起步偏向的来源

**场景**：64 个考试场景的 base 日志（`hugsim-exam/scored-op/cinque-fixed`）的前 3 帧（step 0–2，历史里偏航 < 0.05°），
主读 10 个打转场景 + 偏向 |lean| ≥ 1° 的场景。**读数**：step 0、1、2 的计划 10 s 终点方向和 1 s 方向（+ = 左）。

| 变体 | 改什么 | 含义 |
|:--|:--|:--|
| base | 无（复现日志） | 复现检查：与日志 model_pos 差 < 0.05 m |
| mirror | 模型帧（road + wide）左右镜像，traffic flag 不变 | 模型帧以车辆纵轴为中心（cx = W / 2），镜像帧 = 镜像世界 + 镜像标定 |
| mirror_tc | 镜像 + traffic flag 翻转 | 完整的镜像世界 |
| tc | 只翻 traffic flag | 偏向是不是来自行驶侧先验 |
| single | 不预热：重置后当前帧只走一个 context step（历史为零图） | 历史为零 |
| warm1 | 预热 1 s 而不是 5 s | 静止历史的长度 |
| roll | 预热帧换成按 1 m/s 前进渲染的历史（op_interp 平面 / 60 m 球近似） | 「车已在走」的初始速度 |
| desire_l / desire_r | 当前步给 turnLeft / turnRight | desire 能否抵消 |
| mask_L / mask_R | road + wide 的左半 / 右半换成中灰 | 偏向由哪半边画面驱动 |
| front | 只用 CAM_FRONT 构图 | HUGSIM 三相机拼接路径 |
| yaw±1 | 虚拟相机偏航 ±1° | 标定误差的尺度 |

**判读（登记）**：
- 「偏向随镜像翻转」= mirror_tc 下 step 1–2 的方向与 base 反号的场景占 ≥ 80%（在 |lean| ≥ 1° 的场景里），且
  |mirror_tc + base| < 0.5 |base| 的中位数成立。翻转 → 偏向来自画面内容（场景几何），不是模型或输入路径的固定偏。
- 不翻转（mirror_tc 与 base 同号）→ 来自模型内部先验 / 输入路径；再看 tc、front、yaw 哪个能解释。
- 「traffic flag 携带偏向」= tc 让 step 1–2 方向变化的中位绝对值 ≥ base 的 50%。
- 「历史 / 速度携带偏向」= single 或 roll 让 |lean| 中位数下降 ≥ 50%。
- mask：哪半边被遮住时偏向消失（|lean| 下降 ≥ 50%）就记为该半边驱动；两边都不行记「分散」。
- 与 NAVSIM 路径对比：第 92 条已在 navtrain 停车 / 低速直行帧上测得无固定偏（|mean| ≤ 0.06 m）；这里只比较 HUGSIM 64 场景
  的 step 0–2 偏向的均值和符号分布，均值的 95% CI 不含 0 才说 HUGSIM 路径有系统偏。

## 3. 闭环重放

**场景**：0528-medium-00、0013-medium-00、053-medium-02（base 都打转，pilot 闭环里 0013 / 0528 仍打转）。
**做法**：沿 base 日志的帧序列（video.mp4，同一请求时序：预热 100 reps、之后每步 4 reps），三个模型各自走一遍（开环重放，
同一历史）；每步另做一次「去旋转」重放（第 96 条规则的帧：最近 25 步按当前朝向重渲染）。每步拆分：
φ_rot(k) = φ_normal − φ_derot（历史偏航带来的部分），φ_lean(k) = φ_derot（没有历史偏航时的偏向），历史偏航 H_k 来自日志。
s_k = φ_rot / H_k。同样在 pilot 自己的闭环日志上重放三个模型。

**判读（登记）**：
- 「适配的放大更小」= 在 H_k ∈ [1°, 15°] 的步上，pilot 的 s 中位数 < 0.5 × shipped 的。
- 「适配的打转来自偏向而不是放大」= pilot 的 φ_lean 在打转方向上大于 shipped 的（中位数差 ≥ 1°），而 s 不比 shipped 大。
- 两者都不成立 → 记为「适配在 HUGSIM 帧上没有改变环路」，原因另找（域：HUGSIM 渲染帧不在训练分布里）。

## 预算与产出

卡 2，GPU 合计 < 2 h；CPU ≤ 50 核；每个 openpilot 进程 RSS 超 40 GB 自杀。结果写
`experiments/hugsim/results/launch_lean.md` 和一张图 `experiments/hugsim/figs/launch-lean.png`；不改 decisions。
