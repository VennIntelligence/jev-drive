# op_parity 预登记：Cinque 补齐 WA-JEPA 的全部输入（2026-10-06，任何打分跑之前写定）

## 问题

WA-JEPA 在 HUGSIM 64 上 HD 0.451 对 Cinque 0.278（第 138 条），navtest EPDMS 91.71（我们 devkit 复现）对 Cinque 的 84 级 PDMS。但输入不对等：WA-JEPA 读 4 路相机 × 4 帧 2 Hz、`ego_status = [cmd L/S/R, vx, vy, ax, ay]` 与 4 个 ego 位姿；Cinque 只读前视 road / wide，无速度、无位姿历史、无导航。第 92、96、111、118 条把缺 ego-motion 列为起步打转环路的根源嫌疑。

**问的是**：输入对等、微调数据对等（navtrain，WA-JEPA stage 2 同一份）之后，「comma 预训练」与「V-JEPA 2 + nuPlan 视频预训练」的差距还剩多少（trick or trade）。

用户 2026-10-06 改定的范围：**只做 NAVSIM 与 HUGSIM**；WOD 环路增益读数、comma1M guard 读数取消；本轮不用 comma1M / WOD 数据；防遗忘锚只用 navtrain（同帧蒸馏回 shipped）。

## 不重测的既有结论（grep decisions 后）

- desire 进得了模型但不帮忙（92）；图像指令（99、102）、路线微调（128）、加宽视场（135、137）都已定，本线不再测这些机制本身。
- 第 137 条：rc-bear-fix 配方（stage 4 + plan + action 全放开，CARLA 1.86 m 视角）在 comma1M 直路 ADE ×2.6。本线：视觉整段冻结，只放开 plan 通路，并在 navtrain 同帧上加锚行。
- 第 139 条：worldmodel-4B 不能零样本当主干，所以在 Cinque 上做。

## 设计（怎么注入）

**冻结**：视觉编码器（stage 1–4，到 `view_39` 的 32 × 512 hidden token）完全冻结，所以 token 一次算好缓存，训练只在缓存上跑。
**可训练**：off-policy plan 通路（ONNX 节点 479–665，op_adapt_l `pol_weights`，fp32 master）+ 新 adapter。action 通路不动（NAVSIM 读 plan；HUGSIM PR #57 控制器跟 plan）。
**注入点**：`lib/parity_adapter.py` 输出一个 (32, 512) bias，加到 9 个 context 帧的 hidden token 上（op_adapt_l IntentAdapter / op_route_ft RouteAdapter 同一位置，闭环由 `op_l_onnx.py build --bias-input` 的 `intent_bias` 输入承载）。
- ego：20 维 `[present, cmd L/S/R, vx/10, vy/10, ax/3, ay/3, 4 位姿 (x/10, y/10, yaw)]`（NAVSIM AgentInput，rear axle，t0 坐标系）→ MLP → 4 个 memory token。
- 侧 / 后相机：CAM_L0 / CAM_R0 / CAM_B0 各自沿安装偏航渲染成 openpilot road + wide 一对（`OpenpilotMaps(yaw_deg)`，水平、原高度），用 **Cinque 自己冻结的视觉编码器**（共享权重）编码；图像对 = (key k−1, key k)，2 Hz，k = 1..3 → 3 相机 × 3 时刻 × 32 token，各加相机 / 时刻 / slot embedding。
- 32 个 slot query 经 2 层 pre-norm transformer decoder（d 256）读 memory，最后 Linear(256 → 512) **零初始化**，再乘 present。
- 初始化即 shipped：bias 恒为 0，H + 0 逐位不变。

**相机几何**：NAVSIM 原生 rig（CAM_F0 1.87 m，op_lb `gimm` 协议，84.2 那一行），不用 vcam 降高。理由：第 104 条——navtrain 上降高只换来进度、DAC 反降，地平线与广角本来就对；训练与 navtest 同一 rig，模型自己适应。侧 / 后相机同样原高度、只转偏航。HUGSIM 前视沿用 wajepa_ref 里 Cinque 那一行的 `spec` 预设；侧 / 后相机用 HUGSIM 的 FRONT_LEFT / FRONT_RIGHT / BACK（KITTI-360 / Waymo 的 BACK 全黑，与 WA-JEPA 同样不补）。

**输出**：plan → NAVSIM 8 个 0.5 s 位姿，沿用 op_openloop `nav-export` 的 `base`（lever）适配器；HUGSIM 沿用 Cinque 的 PR #57 路径。

## 臂（同一配方、同一行序、同一步数、seed 0）

| 臂 | 新输入 | 说明 |
|:--|:--|:--|
| P0 | — | shipped Cinque（port，与 ONNX 等价性单独报） |
| P1 | 全部置零（无 adapter） | 「只是微调」对照 |
| P2 | ego status + 4 位姿 + 指令 | |
| P3 | P2 + L0 / R0 / B0 | 完全对等；训练时每路相机按行 p = 0.15 丢弃，使「侧机关」消融在分布内 |
| P3:noside | 测试时 3 路都 mask | 侧 / 后相机消融（只读不训） |

**配方**：每批 (1 − 0.25) 模仿行：plan → 日志 8 位姿 (x, y, yaw)，Huber 归一（σx 0.3 + 0.2t，σy 0.1 + 0.1t，σψ 1° + 1°·t），非 plan 头全部蒸馏回 shipped；0.25 锚行：新输入置零（present = 0），plan 一致性 + 全部头蒸馏回 shipped（同一 navtrain 帧）。λ_i 1、λ_c 3、λ_d 30（= it_dw3 的 dw 3）；AdamW lr 3e-5（base）/ 3e-4（adapter），wd 0.01，warmup 100，cosine，clip 1，fp16 计算 + GradScaler。

## 数据

- pilot 训练行：`runs/op_lb/lb_navtrain`（3 000，每指令 1 000）∪ `lb_h1train`（2 400，低速加权）——已有 GIMM context 帧的 navtrain token；按 log hash 切 `navsim/op-parity-pilot-{train,dev}`（4 874 / 526，1 030 log）。
- 评测：`lb_navtest` 全量 12 146（keys + GIMM 已缓存）。
- 全量训练数据另报估算后再定（见下）。

## 并行与吞吐计划（用户 2026-10-06 要求，实现前写定；括号内是实测后补的数）

盒子：cgroup 75 核、276 GiB、3 × RTX 6000D 83.6 GiB（`jevdrive.cl probe`），磁盘剩 ~305 GB。

1. **特征缓存（一次）**：`pp_prep.py`，每个数据目录一个池作业。CPU 进程池（作业分到的全部核）渲染侧 / 后相机（JPEG 解码 + 两次查表），渲染块流式喂给同一进程里的 GPU 编码器（fp16 port，batch 256 对）；前视 token 由 8 线程读 memmap、GPU 编码。三个数据目录可分在三张卡上并行。
   估算：侧相机 12 帧 / token × ~30 ms ≈ 0.35 core·s / token → 70 核 ~200 token/s；编码 17 对 / token。navtest 12 146 token ≈ 206 k 对。（实测：见 results/prep.md）
   体积：front 256 KB + side 288 KB / token（fp16）→ navtest 6.6 GB、pilot 2.9 GB、全量 navtrain 103 k ≈ 56 GB。
2. **训练**：全部缓存一次搬进显存（pilot 3 GB；全量 56 GB 也放得下一张 83.6 GB 卡），无 dataloader。三个臂一卡一个并行（臂间无通信，比 DDP 更省）；单臂扩到多卡用 torchrun 路径（梯度 all-reduce，各 rank 独立行流）。
3. **navtest 读数**：plans 在缓存上跑（每臂分钟级）；devkit v2 打分 CPU，每臂 ~5 min（16 线程，metric cache 已有）。
4. **HUGSIM**：闭环 server 把 bias（ego、位姿历史、指令、侧机 token）每步算好喂 `intent_bias`；10 个 spinner 场景每臂 ~1 GPU·h 量级，经 GPU 池。

**全量训练的瓶颈是前视 context 帧**：GIMM 每 token 2.2 s（第 37 条整合文档：navtrain 全量 ~63 GPU·h）。选项：(a) navtrain 子集 N 个 token 补 GIMM（N = 30 k ≈ 18 GPU·h，3 卡 ~6 h）；(b) CPU ego-motion warp 补全量（~20 core·h，盒子上 ~20 min；navtest 2 000 子集 warp 83.1 对 GIMM 84.7 PDMS），训练 warp、测试 GIMM 有协议差；(c) 混合。pilot 结束后带实测吞吐向 main 报估算再决定，超过 6 GPU·h 的不先跑。

## 读数与线（pilot 与全量相同口径）

1. **主读数：NAVSIM navtest EPDMS**，devkit navsim main @ 0a380a9（= v2.2 + fixes），metric cache `v2_navtest`——即复现 WA-JEPA 91.71 的同一 harness；WA-JEPA 用它已存的逐 token CSV（`runs/top10_t2/navsim/wajepa/20260926-122804/v2/`）。报每臂 EPDMS（devkit average 行）与九个子分；配对差（逐 token EPDMS，按 navtest log 聚类 bootstrap，B 10000）：**P3 − P1**（主）、P2 − P1、P3 − P2、P3 − P3:noside、各臂 − WA-JEPA（gap）。
2. **HUGSIM 64**（wajepa_ref 同一 harness：PR #57 控制器 `fixed`、400 步、64 场景、一场一跑），对 WA-JEPA 0.451。机制读数改为：打转数（60° 规则，`spin_analysis.py`）与起步停滞数（前 40 步峰值速度 < 1.6 m/s），P2 / P3 对 P1。

## pilot（早停闸门）

pilot：上述 5 400 行，每臂 600 步、batch 64（~8 epoch），seed 0。

闸门（全部满足才进全量估算；任一不满足 → 停、报告）：
1. **初始化等价**：P1/P2/P3-init 在 navtest 全量上的 plan 与 P0 port 逐位相同（max |Δ| = 0）；P0 port 与 shipped ONNX（op_lb `gimm@cinque`）的 plan：4 s 位置 L2 均值 < 0.05 m，且 v2 EPDMS 相差 < 0.2。
2. **NAVSIM pilot**：P2 − P1 与 P3 − P1 的配对 EPDMS 均值 ≥ 0（CI 一并报告）。
3. **HUGSIM 10 个 spinner 场景**（`derot_spin10.txt`）：P2、P3 的打转数各 ≤ P1 的打转数（shipped 为 10 / 10，并列报告）。

## 判读（写在结果之前）

- P3 − P1 > 0 且 CI 下界 > 0：补输入有用；再看 P3 与 WA-JEPA 的差还剩多少——剩得多 = 预训练 / 架构差（trade 在 WA-JEPA 一侧），剩得少 = 原差距主要是输入。
- P2 ≈ P3：侧 / 后相机不重要（与第 135、136 条一致）；P3 − P3:noside 给出直接消融。
- P1 本身相对 P0 的变化是「只在 navtrain 上微调」的效应，不算输入的功劳。
- HUGSIM：P2 打转 / 停滞少于 P1 = ego-motion 输入确实切断起步环路（第 92 / 111 / 118 条的嫌疑）。

## 限定（预先写明）

单 seed；pilot 数据只有 5 400 个 navtrain token（WA-JEPA 用全量 103 k）；视觉冻结意味着侧机只能用 Cinque 前视编码器的表征；2 Hz 侧机对用 0.5 s 间隔（前视是 0.2 s），对编码器是分布外，只靠 adapter 学。

## 补充 1（2026-10-06，读任何 pilot 分数之前写定）：帧协议要不要插帧——分因子设计

用户问：P2 / P3 已经显式给了速度和位姿历史，还需要插帧吗？严格对等也指向不插（WA-JEPA 只看真实 2 Hz 帧，GIMM 给 Cinque 的是 WA-JEPA 没有的合成信息）。但 N / W / G 三个协议同时改了好几样东西，不能并排比。要分开的混杂：(1) 信息丢失 vs 只是冻结编码器没见过的分布（0.5 s 图像对）；(2) N 里两个时间尺度：视觉对间隔（0.2 vs 0.5 s）与特征队列槽距（0.2 vs 0.5 s）；(3) 训练协议 vs 测试协议；(4) 替代交互：显式 ego 状态（P2）是否让图像运动线索变得不需要——估计量是协议 ×（P1 vs P2）交互，不是协议主效应；(5) 打分器伪影：第 116 条 GIMM 把自车运动读慢、慢计划在 NAVSIM 上 DAC 涨 EP 跌，所以 EPDMS 差可能是速度偏差而不是质量，必须同时报 EP / DAC / NC / TTC / comfort、计划速度比、对日志 ADE；(6) 噪声：同一 token 集、同一初始化、≥ 2 个训练 seed、按 log 聚类的 token 配对 bootstrap，管线其余部分（含第 104 条的相机高度处理：原生 rig）逐字节相同。

### Stage A（无训练，现在跑）：`scripts/pp_stageA.py`

token：`lb_hq_navtestX`（第 116 条，1 499 navtest token / 78 log，nuPlan 真实 10 Hz CAM_F0 在 0.2 s 格点 t0 − {1.4, 1.2, 0.8, 0.6, 0.4, 0.2} s 上，加 4 个 2 Hz key），真实帧缺失的 token 全格剔除。模型：shipped Cinque（port fp16，冻结），零状态，t0 计划。−1.5 s 之前的帧在所有格里一律是零图（op_lb 对最老一对的约定）。

| 格 | 队列 | 图像对间隔 | 帧来源 | 用途 |
|:--|:--|:--|:--|:--|
| R8 | Q8：8 槽，−1.4..0 每 0.2 s | 0.2 | 真实 | 参照（真值协议） |
| G8 | Q8 | 0.2 | GIMM | 合成误差（GIMM） |
| W8 | Q8 | 0.2 | warp | 合成误差（warp） |
| R8g4 | Q8 | 0.4 | 真实 | 间隔变长的分布外效应，无合成（0.5 在 0.2 格点上取不到，0.4 作代理） |
| S8 | Q8 | 0（静态对 (t, t)） | 真实 | 去掉图像对运动线索 |
| R4f | Q4f：4 槽，−0.6..0 每 0.2 s | 0.2 | 真实 | 槽数 / 跨度效应（对 R8） |
| W4f | Q4f | 0.2 | warp | 槽距效应的同源对照（对 W4d） |
| N4 | Q4d：4 槽在 2 Hz key（−1.5, −1.0, −0.5, 0；每槽 0.5 s，时间膨胀） | 0.5 | 真实 key | 纯原生 2 Hz 协议 N |
| W4d | Q4d | 0.2 | warp | 槽距 0.5 + 间隔 0.2 |
| S4d | Q4d | 0 | 真实 key | 槽距 0.5 且无运动线索 |

N 的队列长度定为 4 个 0.5 s 槽（NAVSIM 只给 1.5 s 历史，4 帧全用；policy 前 5 槽置零无效）。

对比（都对同一 token、配对）：合成误差 = G8 − R8、W8 − R8；间隔分布外 = R8g4 − R8；运动线索的价值 = S8 − R8；槽数 = R4f − R8；槽距 = W4d − W4f（同源同间隔，只差槽距）；Q4d 内间隔 = N4 − W4d（混了 warp 误差，用 W8 − R8 的量级参照）；N 总效应 = N4 − R8。

读数（每格）：t0 槽 hidden token 对 R8 的余弦距离（32 token 平均）；8 个 NAVSIM 位姿对 R8 计划的 ADE 与对日志的 ADE；速度比（计划 4 s 路程 / 日志 4 s 路程，日志 > 2 m 的 token 取中位）；EPDMS 与九个子分（v2 devkit，token 子集）。配对差（vs R8）：EPDMS、EP、DAC、速度比，log 聚类 bootstrap。

Stage A 不设决策线，只用来定 Stage B 的格子与解释；预先写下的读法：若 N4 − R8 的损失主要随 S4d 一起出现（即 0.5 s 对也像静态对）→ 0.5 s 对对冻结编码器基本没有运动信息，N 协议的运动必须靠 P2 的 ego 输入补；若 R8g4 / N4 的 token 距离大但计划差小 → 是分布外而非信息丢失；若各格 EPDMS 差主要伴随速度比变化与 EP / DAC 反向 → 打分器伪影（第 116 条）。

### Stage B（训练，等 main 看过设计与 Stage A 结果再跑）

因子：训练协议 {N = N4 的帧协议，W = W8，G = G8} × 臂 {P1, P2} × seed {0, 1}，pilot 同一 5 400 token、同一初始化、同一步数与行序；12 次训练。测试：(i) 匹配协议（可部署的数）在 navtest 全量；(ii) `lb_hq_navtestX` 真实帧（R8 协议，N 训练的模型另测 N4 真实 key）——与测试期合成无关地看学到了什么；(iii) 交叉协议（可选）。
估计量与线：协议主效应（G − N、W − N，P2 上，配对、两 seed 合并）；交互 (P2 − P1)_N − (P2 − P1)_G（ego 输入是否替代运动线索）；训练 vs 测试协议效应（同一模型 (i) 与 (ii) 之差）。
全量协议的决策规则（可部署匹配分）：P2 下 N 与 G 的 EPDMS 差 ≤ 1.0 → 全量用 N（不插帧，无 GIMM 成本）；否则 W 在 1.0 以内 → W；否则 G。护栏：所选协议相对 G 的计划速度比漂移 ≤ 5%，且 EP 与 DAC 不出现一涨一跌的交换（两者配对差异号且各自 CI 不含 0 即判交换）；不过护栏则顺延到下一个协议。
