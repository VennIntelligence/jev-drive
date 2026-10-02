# op-adapt H：第 3 层适配 pilot 的预登记（2026-10-04，lane L3）

写于任何训练结果之前（smoke test 只测吞吐，不读指标）。依据：第 92、94、96 条（历史偏航外推、真实转动有用、HUGSIM 打转是闭环放大），第 78–81 条（op_adapt_l 的配方），用户 2026-10-04 的决定（第 3 层先做两种配对）。

## 问题

在 Cinque 上做一次轻度适配，只用两种配对：

1. **历史与未来不一致**：扰动历史帧（假偏航 / 重复当前帧 / 丢掉历史），只用在日志未来不跟随这个运动的帧上；教师 = 日志未来。
2. **偏离后修回**：把图像重投影成自车带朝向偏差（绕后轴的旋转，除相机杠杆外精确）和小横向偏移（路面平面近似）的样子，偏差由历史里的一段漂移达到；教师 = 回到日志路径的修正段。

同一批帧不扰动的版本（含真实的历史转动）也在训练里，教师 = 日志未来（第 94 条：真实转动有用，不能教模型忽略它）。

## 数据（全部是训练集，各榜留出集不碰）

| 域 | 来源 | train / dev | split |
|:--|:--|:--|:--|
| nav | navtrain 中不在 op_lb `lb_navtrain`（3 000，第 92 条探针的 nav 样本来源）里的 2 400 个 token，按 t0 速度分层（stop 700 / low 800 / mid 600 / high 300），每 log ≤ 4；op_lb 的 4 关键帧 + 6 GIMM 帧，与 navtest 读数同一协议 | 2 108 / 292（按 log） | `navsim/op-adapt-h-nav-{train,dev}` |
| wod | WOD-E2E train（op_adapt_l 的 train / dev 切分），同样分层 | 2 400 / 380 | `wod/r2-train` 的子集 |
| carla | `processed/carla_p6_v1` 中 route 与 base route 都不在 bench2drive220 的帧（P6 专家），同样分层 | 2 255 / 333（按 route） | `b2d/op-adapt-h-carla-{train,dev}` |

考题不进训练：navtest、navhard、bench2drive220、HUGSIM 场景、第 92 条探针样本（navtrain `lb_navtrain` 1 907、WOD val 1 436、CARLA P4 1 687）。

## 模型与损失

- op_adapt_l 的 PyTorch port（`LModel`，无 intent）：stage 1–3 冻结，在（扰动后的）图像上在线跑；stage 4 + off-policy plan pathway 可训练（op_adapt_l 里各可训练组差 ≤ 0.02，这里选能改时序整合的组合）。
- 每批 48 行：U 12（不扰动，模仿日志未来）、D 8（不扰动，蒸馏到原模型：plan 一致 + 全部输出头）、H 16（历史扰动，模仿日志未来）、O 12（偏离修回）。域权重 nav 0.4 / wod 0.3 / carla 0.3。
- H：rot 0.6（假偏航率 5–15°/s；日志 3 s 朝向 |ψ| > 3° 时符号取反向，保证未来不跟随），repeat 0.2（只在 v0 ≥ 2 m/s，未来在动），single 0.2。
- O：朝向偏差 1–8°、横向 0–1 m，符号随机，v0 ≥ 1 m/s；历史为 1.6 s 线性漂移；修正段按弧长三次 Hermite，长度 max(8 m, 2.5 s × 速度)。
- 非 plan 头只在不扰动的行上蒸馏（偏移会合理地改变车道线等头）。
- 损失尺度沿用 op_adapt_l（imit 的 16 点 rear-axle 网格、lam_d 10、lam_c 1、dw 1），lr 3e-5，cosine，2 500 步。

## 臂

| 臂 | 内容 |
|:--|:--|
| `pilot` | 上述全部 |
| `pilot_ctl` | 对照：H / O 行换成 U 行（同样的帧、同样的模仿、无扰动）。用来把效应归给配对，而不是归给「模仿日志未来」本身 |

## 读数与线（全部对 port O，同一套代码；配对 cluster bootstrap，jevdrive.stats 默认）

| 读数 | 内容 | 线 |
|:--|:--|:--|
| (a) 第 92 条探针（port 上复跑，样本不变：pnav / pwod / pcarla；变体 normal / rotL / rotR 10°/s / repeat / single） | G = 3 s 朝向的左右差之半，按速度档 | **成立**：`pilot` 的 G_low 在三个域都比 O 低，配对 CI 上界 < 0，且相对降幅 ≥ 30%；同时 `pilot − pilot_ctl` 的 G_low 在 ≥ 2 个域 CI 上界 < 0（归因于配对）。mid 同样报告，不设线。护栏：normal 输入 3 s 横向误差（moving）变差 ≤ 0.10 m |
| (b) navtest PDMS | op_adapt_l 的 navtest 管线（port、GIMM 帧、官方 v1 评分器） | Δ ≥ −0.30 |
| (c) navhard two-stage EPDMS | r2 的 nav_plans + 官方 v2 评分器 | 报告合并分与 stage 1 / 2 分；只说方向，不设通过线。每个模型只考一次，只考在 dev 上选出的模型（至多 pilot、pilot_ctl 和 2 个迭代模型） |
| (d) HUGSIM 打转集（第 96 条的 10 个场景，PR #57 控制器，不加 derot 规则） | 打转数（≥ 60° 航向误差，controller_spin.md 定义） | 对照为同代码的 base 复跑 8 / 10：**成立** = ≤ 4 / 10；HD 均值同报 |
| (e) op_adapt_l 的 start / stop 捕获（WOD val 全量，op_adapt_l_readout） | Δ 对 O | 不变差：点估计 ≥ −0.02 |

## 迭代规则

- 迭代只看 dev（三个池的 dev 切分）：G_low / G_mid、`recover_psi_1s` / `recover_y_1s`、drift 中位、ADE。考题读数 (a)–(e) 不用来选。
- 旋钮：配对比例、扰动强度、步数。每次迭代前在本文件末尾追加一行「改了什么、dev 上期望看到什么」。
- dev 漂移护栏：unperturbed plan 对原模型的 drift 中位 ≤ 0.15 m（op_adapt_l 的线是 0.10，这里因为全部帧都在模仿日志未来，放宽并写明）。

## 追加记录

（迭代与偏离按时间追加于此）
