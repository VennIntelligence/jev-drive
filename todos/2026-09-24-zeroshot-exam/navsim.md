# Zero-shot 考试：Alpamayo 1.5 与 openpilot 在 NAVSIM 上的 PDMS / EPDMS

状态: running（预登记已写死，结果待填）
主题: ../../research/openpilot-and-open-driving-models.md、../../research/benchmarks-and-evaluation.md
姊妹考试: [wod-e2e.md](wod-e2e.md)、[bench2drive.md](bench2drive.md)（同样两个模型，另外两个 agent）

## 目标

把两个开源的驾驶专用大模型原样放到它们没训练过的 NAVSIM 上，量它们的「通用驾驶能力」。
NAVSIM 是基于 nuPlan / OpenScene 真实日志的 non-reactive 开环仿真评测：agent 只给出一条 4 s 的轨迹，devkit 用
LQR 跟踪这条轨迹、和日志里的其他交通参与者一起前推 4 s，再按规则打分。主指标两个：

- **PDMS**（PDM Score，NAVSIM v1）：NC（No at-fault Collision）× DAC（Drivable Area Compliance）×
  加权平均（TTC 5、EP 5、Comfort 2）。TTC 是 time-to-collision 是否越界，EP 是 Ego Progress（相对 PDM-Closed 规划器的前进比例）。
- **EPDMS**（Extended PDMS，NAVSIM v2）：在 PDMS 上多乘 DDC（Driving Direction Compliance，逆行）和 TLC（Traffic
  Light Compliance，闯红灯），加权项多 LK（Lane Keeping）和 EC（Extended Comfort，相邻两帧规划是否一致），Comfort
  换成 HC（History Comfort，和自车历史运动是否衔接）；并且「人类在该项也违规」的情况不扣分。
- navhard two-stage EPDMS：v2 的 pseudo closed-loop 版本，第一阶段在真实场景打分，第二阶段在 3DGS（3D Gaussian
  Splatting，一种可重渲染的场景重建）合成的「偏离后」场景上再打一次，两段按 agent 第一阶段终点的远近加权后相乘。

**不做任何 fine-tuning，不拟合任何参数**，只写输入输出适配层；打分用官方 devkit，原样运行。

## 预登记（2026-09-24，在看到任何 NAVSIM 分数之前写死）

以下选择都在全量推理之前提交。之后的任何改动记到文末「偏离记录」，写清楚改了什么、为什么、对数字的影响。
适配器验证（步骤 2）只看图和 PhysicalAI-AV 上的 ADE，不看任何 NAVSIM 分数；devkit 的端到端检查只跑 devkit 自带的
constant velocity 和 human agent。

### 评测集、devkit 与 n

| 考卷 | devkit | 脚本 / 设置 | n | 已知参照 |
|---|---|---|---:|---|
| **navtest，PDMS** | navsim v1.1（官方 v1 榜） | `run_pdm_score.py`，non-reactive | navtest 全部 token（索引时实数，见下） | human 94.8，CV 20.6（NAVSIM v1 论文） |
| **navtest，EPDMS** | navsim main @ `0a380a9`（v2.2 + 2025-09 human-filter 修复） | `run_pdm_score_one_stage.py`，non-reactive log replay（devkit 默认） | 同上 | human 90.3 |
| **navhard two-stage，EPDMS** | 同上 | `run_pdm_score.py train_test_split=navhard_two_stage`，第二阶段 reactive IDM（devkit 默认） | stage-one + stage-two 全部 token | 见 research 表（DiffusionDrive 27.5 等） |

两个 devkit 共用 nuplan-devkit v1.2（navsim 的 requirements 指定的 tag），装在 `envs/navsim1`、`envs/navsim2`
（`scripts/setup_navsim_devkit.sh`，Python 3.10）。metric cache 各自用自己的 devkit 从 logs + maps 现建。
我们对 devkit 的唯一添加是一个「回放 agent」`jevdrive/navsim_agent.py`：它不看传感器，只按 token 读我们离线算好的
8 个位姿，这样两个 devkit 的打分代码一行不改。模型本身在各自的 venv 里离线跑，输入由 `scripts/navsim_zs_index.py`
在 devkit venv 里用官方 `SceneLoader` 冻结下来（token、4 帧历史、`AgentInput` 里的自车状态和 8 路相机的路径 + 标定），
所以模型看到的就是一个 NAVSIM agent 能看到的全部东西，没有任何未来信息或 NAVSIM 之外的帧。

### 数据上的硬约束：只有 2 Hz

box 上的 OpenScene 日志和相机 blob 都是 2 Hz（`navsim_logs/test/*.pkl` 相邻帧间隔 0.500 s，每帧 8 路 JPEG 都在），
nuPlan 原始的 10 Hz 相机没有下载，OpenScene 也不提供。更关键的是 NAVSIM 规则只给 agent 当前帧往前 4 帧
（t = −1.5、−1.0、−0.5、0 s）。所以两个模型要的高帧率输入都**拿不到**，下面每一项折中都会**让模型吃亏**，
报出来的分数是「模型 + 适配损失」，不是模型的上限。

### 模型与推理配置

| 模型 | 变体 | 配置 |
|---|---|---|
| Alpamayo 1.5（10B，`nvidia/Alpamayo-1.5-10B` @ `7aba829`） | **nav**（主）、**no-nav** | VLM 用 SDPA，`torch.compile` vision 与 expert（smoke 实测对默认 FA2 drift ≤ 0.02 m、CoC 100% 相同），flow matching 默认 10 步；**每个 token 1 条轨迹**（n = 1，一次 reasoning rollout），temperature 0.6、top-p 0.98、reasoning 上限 256 token（官方默认）；seed 取同一 batch 第一个 token 的前 8 位十六进制；按 nav 文本分桶做 batch（同桶 prompt 等长，不需要 padding），batch 大小由优化阶段决定并记入下面的吞吐表 |
| openpilot small（30M） | **none**（主）、cmd | onnxruntime TensorRT EP，图精度（smoke 里开 fp16 flag 会在分布外输入上溢出） |
| openpilot Cinque v3（382M） | **none**（主）、cmd | TensorRT EP fp16 |
| openpilot Lebowski（877M） | **none**（主）、cmd | TensorRT EP fp16，context-rate 步进（每个 5 Hz context 步一次，在输出相位上与 20 Hz modeld 逐位一致，由 B2D 考试的 `scripts/test_zeroshot_openpilot_context.py` 验证） |

**n = 1、无 oracle**：主表每个模型每个变体只交一条轨迹，不做任何「多采几条挑最好」的选择。

### Alpamayo：相机映射

Alpamayo 要 4 路 f-theta 相机（一种按入射角线性映射像素半径的鱼眼模型）：cross-left（index 0）、front-wide（1）、
cross-right（2）各 120°，front-tele（6）30°，每路 1920×1080，processor 缩到 576×320。虚拟 rig 直接用三场考试共用的
`scripts/zeroshot_rigs.py` 的 `ALPAMAYO_CAMERAS`（PhysicalAI-AV 100 个 clip 的标定中位数：cross 相机光轴 yaw ±67°，
front-wide / tele 朝前，pitch 都在 ±0.7° 内）。我们直接渲染 576×320 的模型图（与 processor 从 1920×1080 缩出来的像素网格一致），
processor 不再缩放。

nuPlan 有 8 路针孔相机，1920×1080，带明显桶形畸变（k1 = −0.356），fx = 1545，去畸变后水平约 72°、垂直约 ±21°；
光轴 yaw：F0 0°、L0/R0 ±55°、L1/R1 ±112°、L2/R2 ±141°、B0 180°。

**映射方法**：对每个虚拟相机的每个输出像素，按 f-theta 模型算入射射线（rig 系 = NAVSIM 自车系，x 前 y 左 z 上），
在 8 路 nuPlan 相机里找能看到它的、且射线离光轴最近的那一路，用该相机的内参 + Brown 畸变投影、双线性采样。
只做**纯旋转重投影**（忽略相机之间的平移，等价于场景在无穷远）；nuPlan 相机装在约 1.5 m 高的车顶，PhysicalAI 的 cross
相机在 0.9 m 高的翼子板，近处物体会有视差误差，这是无深度时唯一可做的映射，不修正。120° 的视图从 1/4 解码的源图采样
（DCT 缩放，386 px/rad，高于模型图中心需要的 278 px/rad），tele 从全分辨率 F0 采样。**看不到的地方填黑**：
nuPlan 相机垂直只有约 ±21°，虚拟相机要 ±34°，三路 120° 视图上下各有一条黑带，实测覆盖率约 64%（tele 100%）。
水平方向完全覆盖：front-wide 由 F0 + L0 + R0 拼成，cross-left 由 F0/L0/L1/L2 拼成，cross-right 对称。

### Alpamayo：时间轴与 egomotion

- **图像**：Alpamayo 要每路 4 帧 @10 Hz（t0−0.3 … t0），NAVSIM 只有 2 Hz。两个候选：
  `repeat`（4 个槽都放 t0 帧，场景看起来静止）和 `2hz`（放 t = −1.5、−1.0、−0.5、0 的 4 帧，时间被压缩 5 倍，运动看起来快 5 倍）。
  **选择规则（在看到 ablation 结果前写下）**：在 PhysicalAI-AV 的 31 个 clip 上（原生 10 Hz 数据，有真值，不是 NAVSIM）跑
  `repeat+rt+ego2hz` 和 `2hz+rt+ego2hz` 两种完整的 NAVSIM 式输入（rt = 经过 nuPlan 相机的往返渲染，ego2hz = 下一条的 2 Hz egomotion），
  每个 clip 6 条样本，**取 4 s 平均 ADE（6 条样本的均值，对应 n = 1 的期望误差）更低的那种**作为 NAVSIM 主设置。
- **ablation 结果与选择（2026-09-24 17:53，仍在任何 NAVSIM 分数之前）**：PhysicalAI-AV 31 个 clip（smoke run 的同一批，t0 = 5.1 s），
  每个 clip 每种输入 6 条样本，4 s 平均 ADE 先在 clip 内对 6 条取均值、再对 31 个 clip 取均值，括号是对 clip 的 bootstrap 95% CI。

  | 输入 | ADE@4s（m） | minADE_6@4s（m） | ADE@6.4s（m） |
  |---|---:|---:|---:|
  | native（原生 10 Hz 帧、原生 egomotion、原生相机） | **0.73** [0.59, 0.88] | **0.31** | **1.77** |
  | native + ego2hz | 0.74 [0.59, 0.90] | 0.31 | 1.78 |
  | native + rt（只做 nuPlan 相机往返） | 0.95 [0.73, 1.21] | 0.41 | 2.32 |
  | repeat（只改时间轴） | 1.42 [1.10, 1.73] | 0.61 | 3.65 |
  | 2hz（只改时间轴） | 1.37 [1.17, 1.58] | 0.68 | 3.17 |
  | **repeat + rt + ego2hz**（NAVSIM 式完整输入） | **1.43** [1.14, 1.78] | 0.67 | 3.71 |
  | 2hz + rt + ego2hz（NAVSIM 式完整输入） | 1.44 [1.19, 1.74] | 0.63 | 3.35 |

  读法：egomotion 从 2 Hz 插值几乎不掉（+0.004 m）；相机往返（黑边 + 视差 + 重采样）让 4 s ADE 涨 0.22 m；
  **掉得最多的是时间轴**，无论 repeat 还是 2hz 都让误差翻倍。两种完整输入之差 −0.01 m（repeat − 2hz，配对 bootstrap
  95% CI [−0.33, +0.28]，repeat 在 31 个里赢 14 个），实际上打平；按预先写下的规则取数值更低的 **repeat** 作为 NAVSIM 主设置。
  这张表也给出了适配损失的量级：同一个模型、同一批场景，NAVSIM 式输入的 4 s ADE 是原生输入的约 2 倍。
- **egomotion 历史**：Alpamayo 要 16 步 @10 Hz（t = −1.5 … 0）的 xyz 和旋转，NAVSIM 给 4 个 2 Hz 位姿（x、y、yaw，
  t0 后轴系）和各自的车体系速度。位置用三次 Hermite 插值（节点速度用实测速度转到 t0 系），yaw 用三次样条，z = 0，
  roll = pitch = 0。ablation 里的 `native+ego2hz` 单独量这一步的损失。
- **导航**：NAVSIM 的 driving command（左 / 直 / 右 / 未知）映射成 Alpamayo 的 route 文本，模板与 WOD-E2E 考试相同：
  左 → "Turn left"，直 → "Continue straight"，右 → "Turn right"，未知 → 不给 route。注意 Alpamayo 训练时的 nav 文本带距离
  （如 "Turn left in 11m"），NAVSIM 不提供到路口的距离，所以这个模板本身是分布外的。no-nav 变体完全不给 route。
- **输出**：64 个点 @10 Hz（0.1 … 6.4 s，t0 rig 系 = 后轴），取第 5、10、…、40 个点，正好是 NAVSIM 的 0.5 … 4.0 s，
  不需要插值；heading 取模型输出旋转矩阵的 yaw。

### openpilot：相机、时间轴与输出

- **相机**：只用 CAM_F0。openpilot 的 road（focal 910）和 wide（focal 455）两个 512×256 model frame，calib 系取 NAVSIM
  自车系（水平、正前）；按 openpilot 自己的 warp 做法，在整数 model 像素上最近邻采样，源图是 libjpeg 直接解出的 YCbCr、
  色度取 2×2 均值（与 WOD-E2E 考试的 openpilot runner 一致）。两个 frame 都完全落在 F0 里（覆盖率 100%）。
  相机高度约 1.5 m，高于 comma 设备的 1.22 m，不修正。
- **时间轴**：openpilot 在 20 Hz 时钟上跑，context 是 5 Hz、约 5 s。NAVSIM 只有 1.5 s 的 2 Hz 帧，所以每个 token
  从零状态开始，在 t = −1.5 … 0 的 20 Hz 时钟上走 31 步（Lebowski 走 8 个 context 步），每步喂「当前时刻之前最近的
  那一张 2 Hz 帧」（sample-and-hold），在 t0 帧刚到达的那一步取 plan。后果：模型看到的是一段「每 0.5 s 才动一下」的视频，
  context 里只有 1.5 s，前面约 3.5 s 是零状态。这是最大的一项 handicap。
- **desire**：主变体 none（openpilot 没有导航输入）；cmd 变体把左 / 右 command 映射成 desire turnLeft / turnRight，
  每步用该时刻那一帧的 command。
- **traffic convention**：sg-one-north（新加坡，左行）给 [0, 1]，其他城市 [1, 0]。
- **输出**：plan 的 33 个点（0 … 10 s，非均匀）在相机点的 calib 系里；按刚体关系换到后轴：
  rear(t) = d + p(t) − R(ψ_t)·d，d 是 F0 在后轴系里的 (x, y)，再在 0.5 … 4.0 s 线性插值 x、y、yaw（yaw 取反为逆时针）。

### baselines

- devkit 自带 constant velocity agent 与 human agent，在我们的 metric cache 上重跑（也作为 devkit 端到端检查：
  v1 的 CV 应接近 20.6、human 接近 94.8，v2 human 接近 90.3；差太多就先查 devkit，不跑模型）。
- 文献数字（不重跑）：TransFuser、DiffusionDrive 及已发表的 VLM 条目，取自 research/openpilot-and-open-driving-models.md 的表。

### 判读（预登记）

| 结果（Alpamayo nav，navtest） | 怎么读 |
|---|---|
| EPDMS ≥ 77（≥ TransFuser） | 在 5 倍时间压缩 / 静止帧、36% 黑边、分布外 nav 文本的条件下仍达到专门在 navtrain 上训练的 baseline，说明「通用驾驶能力」可以迁移 |
| 50 ≤ EPDMS < 77 | 部分迁移；按 sub-score 看是哪几项在扣（DAC / EP 说明几何或速度尺度，NC / TTC 说明交互） |
| EPDMS < 50 或低于 CV 之外的简单 baseline | 适配损失主导或模型不会开车；先用 PhysicalAI-AV ablation 的 ADE 退化量判断适配损失有多大 |
| nav − no-nav 的 EP / DAC 在左右转 command 子集上显著为正 | 模型读懂了模板化的导航文本 |

openpilot 没有 route、只看前视、context 被砍掉 70%，预期在路口 command 场景明显落后；它的数字主要回答
「车道跟随 + 纵向控制」这部分能力在 NAVSIM 规则下值多少分。

## GPU 协调与吞吐

三场考试共用一张 96 GB RTX PRO 6000 和 25 核，协调记录在 box 的 `~/data/runs/zeroshot-exam/gpu-plan.md`。
（优化前后吞吐、batch 选择与最终排期在这里补。）

## 结果

（跑完再填。）

## 偏离记录

（暂无。）
