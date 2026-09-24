# Zero-shot 考试：Alpamayo 1.5 与 openpilot 在 WOD-E2E val 上的 RFS / ADE

状态: running（预登记已写死，结果待填）
主题: ../../research/openpilot-and-open-driving-models.md、../../research/benchmarks-and-evaluation.md

## 目标

把两个开源的驾驶专用大模型原样丢到它们没训练过的 benchmark 上，量它们的「通用驾驶能力」到底有多少。
benchmark 是 WOD-E2E（Waymo Open Dataset Vision-based End-to-End driving，Waymo 的纯视觉端到端驾驶数据集）的 val split，
主指标是 RFS（Rater Feedback Score，把预测轨迹和 3 条人工打过分的轨迹比，看是否落在它们的 trust region 里，0–10 分），
副指标是 ADE（Average Displacement Error，逐点平均位移误差）。打分用仓库里已有的 `jevdrive.waymo.rater_feedback_score`
（官方实现的 bit-exact 移植）和 `rfs_by_cluster`，与我们自己的 planner 完全同一口径。**不做任何 fine-tuning，不拟合任何参数。**

姊妹考试（NAVSIM、Bench2Drive，同样两个模型）由另外两个 agent 做，放在同目录下的其他文件里。

## 预登记（2026-09-24，在看到任何分数之前写死）

以下所有选择在跑全量 eval 之前提交。之后的任何改动都记在文末「偏离记录」里，写清楚改了什么、为什么、对数字的影响。
适配器验证阶段（步骤 2）只看图、不算任何 RFS/ADE；验证用的帧不在评测集里（见下）。

### 评测集与 n

| 集合 | 定义 | n | 用来算 |
|---|---|---:|---|
| **rater 帧（主表）** | val 里每个 sequence 唯一一帧带 3 条 rated trajectory 的帧（frame index 147–150，只有 1 帧在 203） | **479** | RFS（cluster mean 为榜单口径，frame mean 并列）、in-trust-region、floored、ADE@3s/5s vs rater_best（官方 ADE）、ADE@5s vs logged future |
| ADE-extra | 每个 val sequence 用 seed 0 均匀抽 2 帧：frame index ≥ 100、有完整 20 步未来、不是 rater 帧、与 rater 帧相距 ≥ 10 帧 | ≤ 958 | 只算 ADE@3s/5s vs logged future（没有 rater 轨迹，不能算 RFS） |
| 适配器验证帧 | 不在上面两个集合里的 val 帧，人工挑 8 帧（直行、左转、右转、停车、夜间各至少一帧） | 8 | 只看图，不算分 |

val 的 93 个 shard 全部在盘上、10 Hz 帧完全连续（docs/waymo-e2e.md），所以两种模型要的历史帧都是**精确命中**，不需要任何 padding；
rater 帧里 99.8% 往前有满 10 s 的帧，剩下 1 帧从 sequence 第一帧开始（见 openpilot 一节）。

### 模型与推理配置

| 模型 | 变体 | 配置 |
|---|---|---|
| Alpamayo 1.5（10B，`nvidia/Alpamayo-1.5-10B` @ `7aba829`） | **nav**、**no-nav** | smoke run 里「输出不变」档的最快配置：VLM 用 SDPA、`torch.compile` vision 和 expert（smoke 实测对默认 FA2 的 drift ≤ 0.02 m、CoC 100% 相同）；flow matching 保持默认 10 步；每帧 **K = 6** 条轨迹（`num_traj_samples=6`，每条配一次独立 reasoning rollout），temperature 0.6、top-p 0.98、reasoning 上限 256 token（官方默认）；每帧固定 seed 42 |
| openpilot small（30M） | — | onnxruntime TensorRT EP，图精度（不开 fp16 flag，smoke 里开了会在分布外输入上溢出） |
| openpilot Cinque v3（382M） | — | TensorRT EP fp16 |
| openpilot Lebowski（877M，0.11.2 big model） | — | TensorRT EP fp16 |

加载路径：Cosmos-Reason2-8B 的 HF gate 已经通过，Alpamayo 走官方 `from_pretrained`（不再需要 smoke run 里的 offline 绕法；如果官方路径仍失败，退回 offline 绕法并记为偏离，二者读的是同一份 sha 校验过的文件）。

### Alpamayo：相机映射

Alpamayo 要 4 路相机：cross-left（index 0）、front-wide（1）、cross-right（2）、front-tele（6），每路 1920×1080，
processor 再缩到 576×320。它的训练数据（NVIDIA PhysicalAI-AV）用的是 f-theta 鱼眼（一种按入射角线性映射像素半径的镜头模型）：
front-wide / cross 是 120° 水平视场，tele 是 30°。从 PhysicalAI-AV 的 `calibration/camera_intrinsics` 与 `sensor_extrinsics`
（chunk 0 的 100 个 clip，取逐项中位数）得到的虚拟 rig 是：

| 虚拟相机 | 光轴 yaw / pitch | 水平覆盖（yaw） | 垂直覆盖（相对光轴） |
|---|---|---|---|
| cross-left | +67° / 0° | +7° … +127° | ±34° |
| front-wide | −1° / −0.5° | −61° … +59° | ±34° |
| cross-right | −66° / 0° | −126° … −6° | ±34° |
| front-tele | 0° / 0° | ±15° | ±8.4° |

WOD-E2E 每帧有 8 路针孔相机，972×1079（后三路更矮），水平视场都是约 47°，光轴 yaw 分别是
FRONT 0°、FRONT_LEFT/RIGHT ±45°、SIDE_LEFT/RIGHT ±90°、REAR_LEFT/RIGHT ±135°、REAR 180°；
主点在第 719 行，所以光轴以上约 33°、以下只有约 18°。

**映射方法**：对每个虚拟相机的每个输出像素，按它的 f-theta 模型算出入射射线（vehicle 系），
在 WOD 的 7 路相机（除 REAR 外全部）里找能看到这条射线的相机，取光轴与射线夹角最小的那一路，
用该相机的内参（含 k1/k2/p1/p2/k3 畸变）投影、双线性采样。只做**纯旋转重投影**：忽略相机之间的平移，
等价于假设场景在无穷远。所以近处物体的位置会有视差误差（WOD 相机装在 1.81 m 高的车顶，
PhysicalAI 的 cross 相机在 0.9 m 高的翼子板、front 在 1.43 m），这是无深度信息时唯一可做的映射，预登记为已知的适配损失，不修正。

**看不到的地方填黑（RGB 0）**。WOD 相机光轴以下只有约 18°，虚拟相机要 34°，所以四路画面底部约一条带子是黑的
（在 PhysicalAI 上这里通常是引擎盖和近处路面）。水平方向 7 路相机覆盖 −158° … +158°，四路虚拟相机的水平范围全部有像。
tele 完全落在 FRONT 里面（FRONT 1115 px/rad，tele 缩到 576 宽之后是 1107 px/rad，分辨率基本不损失）。
每路的覆盖率（非黑像素比例）在适配器验证里测出来补到下面，不是分数。

**数据来源**：本地 slim shard 只留了 FRONT / FRONT_LEFT / FRONT_RIGHT。cross 相机有一半视场（yaw 68°–127°）
只有 SIDE / REAR 相机能提供，所以从 GCS 原始 val shard 按 TFRecord 帧头逐条走位置、只下载需要的记录（完整 8 路），
存成本地的小 shard。不下载任何别的东西。

### Alpamayo：历史帧与 egomotion

- **图像**：每路 4 帧 @10 Hz，即 WOD 的 frame f−3、f−2、f−1、f，全部精确命中（WOD 的 frame index 间隔实测 0.1000 s）。
  这正好是 Alpamayo 的原生帧率，不需要任何折中。
- **egomotion 历史**：Alpamayo 要 16 步 @10 Hz（t = −1.5 … 0 s）的 xyz 和 3×3 旋转，都在 t0 车体系里。
  WOD 只给 16 步 @4 Hz（t = −3.75 … 0 s）的 x/y、速度、加速度，没有 z 和朝向。
  预登记的换算：位置用 16 个 4 Hz 点做三次样条插值到 10 Hz 的 16 个时刻；z = 0；
  朝向只取 yaw，由速度矢量方向得到，再整体平移使 t0 的 yaw 为 0（车体系定义要求 t0 朝向为 +x）；
  速度 < 1 m/s 的步沿用最近一个运动步的 yaw，全程没动就取 0。roll、pitch 为 0。
- **坐标原点**：WOD 与 PhysicalAI 都是后轴中点（PhysicalAI 的 rig 原点在后轴、地面高度），所以不做平移。

### Alpamayo：导航

Alpamayo 训练时的 nav 文本形如 `Turn left in 11m`（`notebooks/nav_demo_samples.json`），放在 `<|route_start|>…<|route_end|>` 之间。
WOD 只有一个 intent（GO_STRAIGHT / GO_LEFT / GO_RIGHT / UNKNOWN），没有距离；用 logged future 推一个距离会泄露答案，所以不给距离。
预登记模板：

| WOD intent | nav 文本 |
|---|---|
| GO_STRAIGHT | `Continue straight` |
| GO_LEFT | `Turn left` |
| GO_RIGHT | `Turn right` |
| UNKNOWN | 不给 nav（rater 帧里没有出现） |

rater 帧里 427 帧直行、23 左、29 右。**no-nav 变体**用官方的无 nav prompt（`create_message(nav_text=None)`），其余完全相同。
两个变体都是主结果；nav 是否帮忙在直行帧和转弯帧上分开报。

### Alpamayo：输出重采样与轨迹选择

- 输出 64 点 @10 Hz（0.1 … 6.4 s）。在 t = 0 补上原点，按时间线性插值到 WOD 的 20 点（0.25 … 5.0 s），丢掉 z。
- **headline**：「一条采样轨迹的期望 RFS」= 每帧 6 条轨迹各自的 RFS 取平均，再按榜单口径聚合。
  它就是「每帧只部署 1 条采样」时 RFS 的无偏估计，方差比只用一条小。**第 0 条样本单独算一遍**，作为字面意义上的一次提交，并排报。
- 非 oracle 的次要读数：**medoid**（6 条里与其余 5 条平均 ADE@5s 最小的那条），标注为「medoid-of-6」。
- oracle 读数，只能单独列、明确标注：best-of-6 RFS、minADE_6 vs rater_best。**不进 headline，不和 baseline 直接比名次。**

### openpilot：相机映射、帧率与输出

- **相机**：模型要两路 512×256 的 model frame，已经是 calib frame（与车体朝向对齐的虚拟针孔相机）：
  narrow 焦距 910（水平约 31°，主点纵坐标 47.6），wide 焦距 455（水平约 59°，主点纵坐标 151.8）。
  用上面同一个纯旋转重投影，直接从 WOD 的 FRONT / FRONT_LEFT / FRONT_RIGHT 渲染 calib frame：
  等价于把一台 comma 装在 WOD FRONT 相机的位置，calibration 的 roll/pitch/yaw 全为 0。
  两路的视场都完全落在这三路相机的覆盖里（narrow 全在 FRONT 内，wide 的左右边缘来自 FRONT_LEFT/RIGHT），没有黑边。
  采样用最近邻，和 modeld 的 warp（tinygrad `compile_warp.py`）同一种插值；YUV 直接取 JPEG 自己的 YCbCr（BT.601 full range），
  色度按 2×2 取均值，按 modeld 的顺序打包成 6×128×256。
- **帧率与历史**：模型 20 Hz 运行、5 Hz 上下文（取 0.2 s 前的帧、4.8 s 的 feature 历史）。
  WOD 是 10 Hz，**每帧连喂两次**得到 20 Hz：这样 t−4 步正好是 2 个 WOD 帧前、0.2 s，与 modeld 的时间语义完全一致，只是同一帧出现两次。
  从 frame max(f − 100, sequence 第一帧) 开始、零状态起步，一直喂到 frame f（10 s 暖机，远多于 feature 队列要的 4.8 s），
  取最后一步的输出。desire 全 0（openpilot 没有路口导航输入，WOD intent 不给它），traffic convention [1, 0]（美国，右侧通行），
  action_t 同 smoke run（0.275 / 0.525 s）。
- **输出**：plan 的 33 个点（0–10 s，非均匀时间）取均值（MDN 的 μ），device 系（x 前、y 右、z 下）转成 WOD 的 x 前、y 左；
  plan 原点在相机，换到后轴：`rear(t) = d + plan(t) − R(ψ_t) d`，d 是 WOD FRONT 相机在车体系的 (x, y)，ψ_t 是 plan 自己的 yaw
  （转成左正）。然后按时间线性插值到 0.25 … 5.0 s。
- **不做尺度修正**：WOD 相机离地 1.81 m，comma 约 1.2 m，同一像素对应的地面距离会变，plan 的纵向可能有系统偏差。
  任何修正都要用带真值的数据估，那就不是 zero-shot 了，所以预登记为已知的适配损失，只在结果里诊断（纵向误差的符号），不修正。

### 指标与统计

- RFS：cluster mean（榜单口径：每个 scenario cluster 内求均值，再对 10 个出现的 cluster 不加权平均）为主，frame mean 并列；
  另报 in-trust-region 比例和 floored 比例（分数恰好等于下限 4.0 的比例，定义同 `rater_table`）。
- ADE：官方 ADE@3s、ADE@5s vs rater_best（评分最高的 rater 轨迹），以及 ADE@5s vs logged future。
- CI：10 000 次 bootstrap，percentile 95%。frame mean 按帧有放回重抽；cluster mean 在每个 cluster 内分层重抽（保证 10 个 cluster 都在）。
  和 baseline 的比较一律用**配对** bootstrap（同一组重抽的帧上算差），配对对象：cv、logged future、`cls ego`（我们 train 训出来的最好 RFS head，
  逐帧预测在 box 上）。ADE-extra 集按 sequence 重抽（每个 sequence 有 2 帧）。
- 按 scenario cluster 拆 RFS（10 个 cluster，n 从 15 到 116，n < 30 的格子标注「读不动」）；按 intent（直行/转弯）拆 nav 的效果。

### Baseline（全部是仓库里已有的数字或已存的逐帧预测，同一 479 帧）

| 行 | 来源 |
|---|---|
| rater_best / rater_worst | `report/rater.csv`（9.59 / 7.71） |
| logged future | 同上（8.13） |
| cv（匀速，零参数） | 同上（7.10） |
| zero（原地不动） | 同上（5.38） |
| ridge ego / `cls ego` / `cls_late` vision+ego | 第 3d / P0 条的 train 训 head（7.06 / 7.31 / 7.30，cluster mean） |
| 公开榜（**test split**，不是 val） | RAP 8.043、Poutine 7.986、AutoVLA 7.556、OpenEMMA 5.158 |

test 与 val 是不同的 sequence，而且榜单数字是各队自己挑的最好提交，所以只能作为量级参照，不能配对比较。

### 怎么判（跑之前写死）

| 结果 | 读法 |
|---|---|
| 模型 RFS 的配对 Δ vs cv 的 CI 整体 > 0 | 通用驾驶能力在这个分布外的 benchmark 上**有增量**，超过了「只看 ego 状态匀速外推」 |
| CI 跨 0 | zero-shot 下没有可测的增量：通用能力被分布差 + 适配损失吃光，或本来就没有超过 ego prior |
| CI 整体 < 0 | zero-shot 比匀速外推还差：适配损失（视差、黑边、相机高度、帧率）或分布差大于模型带来的东西；要看失败案例区分 |
| 模型 RFS ≥ logged future（8.13）的 CI 下界 | 达到「完美复现司机」的水平，和公开榜第一梯队同量级 |
| nav − no-nav 在转弯帧上 > 0、直行帧上 ≈ 0 | 模板化的 intent 文本被模型用上了 |

分数本身不会被用来改任何适配选择。如果全量跑完后发现适配有 bug（例如坐标符号错），修掉重跑，把修正前后的数字都记进偏离记录。

### 预计算力

Alpamayo：smoke 实测 SDPA + compile、K = 6 约 2.2–2.4 s/帧，479 帧 × 2 变体 + 958 帧 × 1 变体（ADE-extra 只跑 nav）≈ 1.2 h GPU。
openpilot：每个 rater 帧约 200 步 × 3 ms，三个模型合计约 15 分钟，瓶颈在 JPEG 解码（多进程）。
GCS 抓取：(479 + 958) × 4 条记录 × 约 2.3 MB ≈ 13 GB，按 10 MB/s 约 25 分钟。全部远低于 3 h，不需要 profiling pass。

## 适配器验证

8 个验证帧（不在评测集里），只看图、不算分。

![adapter views](../../research/figs/wod-zeroshot-adapter.png)

图：验证帧 `f8af7b57…-135` 上两个模型实际看到的输入。上排和左下是 Alpamayo 的四路 f-theta 视图（由 WOD 7 路相机纯旋转重投影），
中下、右下是 openpilot 的 road / wide model frame（由 FRONT / FRONT_LEFT / FRONT_RIGHT 渲染，只画了 RGB，实际喂的是 YCbCr 打包）。
看三件事：相机之间的接缝处车道线和高架是连续的，说明外参、畸变和选相机的逻辑都对；地平线在各路里的高度与 PhysicalAI 的相机一致；
三路 120° 视图底部约 22% 是黑的，这就是预登记里说的「WOD 相机光轴以下只有 18°」。

| 项 | 结果 |
|---|---|
| Alpamayo 各路覆盖率（非黑像素，8 帧均值） | cross-left 0.746、front-wide 0.781、cross-right 0.752、front-tele 1.000；8 帧之间差别 < 0.01（WOD 的标定几乎不随 sequence 变） |
| 缺的是什么 | 几乎全部是画面底部（光轴以下 18°–34° 的带子），外加 cross 视图上角的小块（WOD 相机光轴以上只有 33°） |
| openpilot 两路覆盖率 | 1.000（代码里 assert 过） |
| GPU（torch `grid_sample`）与 CPU（cv2 `remap`）两条渲染路径 | 目视一致（同一帧两张图逐路对照） |
| 坐标符号 | 8 帧 BEV 上两类模型的预测都跟着 logged future 的左右转方向走（左转帧向左、右转帧向右），直行帧 5 s 终点与 log 差 < 2 m |
| Alpamayo 官方加载路径 | Cosmos-Reason2-8B gate 通过后 `from_pretrained` 在线加载成功，不再需要 offline 绕法 |

一个值得记下的观察（不是适配问题）：两个左转验证帧上，Alpamayo 的轨迹向左、和 log 一致，但它的 CoC 文本写的是 "Turn right at the intersection"。
轨迹方向由 egomotion 历史和图像共同决定，文本里的左右词却反了；NVIDIA 自己的 `nav_demo_samples.json` 里也有 nav "Turn left" 配 CoC "Turn right"
的样本。所以这像是模型文本侧的左右混淆，结果部分会在全部转弯帧上统计文本方向和轨迹方向的一致率（推测，待统计）。

### 算力与 GPU 排期（与 NAVSIM、Bench2Drive 两个考试共用一张卡）

共享文件 `~/data/runs/zeroshot-exam/gpu-plan.md` 上的约定：WOD 只起**一个** Alpamayo 进程（K = 6 时峰值 32 GB），
不用 B2D 的常驻 server（它按请求串行，K = 6 的一次调用会把 B2D 的每个 tick 堵 3–5 s）；openpilot 和 Alpamayo 同时跑，不单占时段；
B2D 的全量闭环排在 WOD 和 NAVSIM 的 Alpamayo 任务之后。实测：

| 任务 | 规模 | 单价 | wall | 瓶颈 |
|---|---|---|---|---|
| GCS 抓取 | 5 728 条记录（1 437 个目标帧 × 4 帧），约 13 GB | 走帧头约 1 s/条（经 Clash），之后约 95 条/分钟 | 约 60 min | 网络延迟（每个 shard 要顺序读 ~1 150 个 12 字节帧头）；与下面两项重叠 |
| openpilot 三个模型 | 1 437 目标 × 最多 101 帧 | GPU 0.83 s/目标（small 0.21、Cinque 0.46、Lebowski 0.16） | 2 124 s（1.48 s/目标） | CPU：每目标 303 张 JPEG 解码 + 重投影，16 个进程；GPU 只占 3.7 GB |
| Alpamayo，rater 帧 × 2 变体 | 958 次 K = 6 调用 | 3.2–5.6 s/调用（卡上同时有 B2D/NAVSIM 的 Alpamayo，GPU 100%）；smoke 在空卡上是 2.2–2.4 s | 约 1 h | GPU（与另两个考试共享） |
| Alpamayo，ADE-extra × nav | 958 次调用 | 同上 | 约 1 h | 同上 |

优化只做了配置层面、输出不变的那一档（SDPA + compile vision/expert，smoke 实测 K = 6 时比默认 FA2 快 12%，drift ≤ 0.02 m）；
K = 6 时 HF `generate` 把整段 prompt（16 张图）复制 6 份再 prefill，这 6 倍重复是最大的浪费，但要改模型代码才能去掉，
按 CLAUDE.md「第三方代码原样跑」不动它。总 wall 时间约 2 h，低于 3 h 的 profiling 门槛。输入准备（JPEG 解码 + GPU 重投影 + processor）
在后台线程里提前做 3 帧，与 GPU 推理重叠。

## 结果

（待填。）

## 偏离记录

（暂无。）
