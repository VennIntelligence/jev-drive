# P4：CARLA 特征差距、词表覆盖与 head 迁移

状态: running（预登记 2026-09-23，写于任何 CARLA 帧生成之前）
主题: ../research/prediag-2026-09/README.md（P4 行）；背景是 ../research/decisions.md 第 19、21、24、25 条

## 目标

P5 要在 CARLA 里造「只差一处可见因素」的配对，拿它当 Waymo 训出来的 head 的考卷，将来还要当教材。
这件事有意义的前提是：同一个 backbone 看 CARLA 帧时，特征落在 head 读得懂的地方，词表也装得下 CARLA 里 expert 开出的轨迹。
P4 只回答这一个前提，**不在 CARLA 上训练任何东西**。三个量：

1. **特征差距**：domain classifier（在冻结特征上训一个分类器区分 Waymo 帧和 CARLA 帧，用 AUC 衡量能分多开）
   在 P3 选定的表征上有多大；配上匹配子集和几个域内对照，让 AUC 能读。
2. **词表覆盖**：K=1024 的 Waymo anchor 词表对 CARLA expert 未来轨迹的 oracle minADE（词表里离真值最近那条 anchor 的 ADE，
   任何打分器在这个词表上的下限）和 trust-region 覆盖（WOD-E2E RFS 的容忍区，见第 22 条），与 Waymo val 上同一个数对比。
3. **head 迁移**：在 Waymo 上 fit 的 `ridge ego` / `ridge_late`（P3(d″) 那一套）直接用在 CARLA 帧上，ADE 对 CARLA 真值；
   分类头的 anchor 分布对比；再加一个语义 probe 迁移（Waymo 上训的线性 probe 在 CARLA 上还读不读得出同一件事）。

## Setup

### 表征（与 P3(d″) 逐项相同）

Qwen3-VL-4B-Instruct，原生视频输入，三相机（front、front_left、front_right，按这个顺序），每相机 4 帧、相邻帧间隔 0.2 s
（Waymo 10 Hz 下的 stride 2，跨 0.6 s），oldest first，decoder 停在第 18 层，取 `L18_last`（主 tap）和 `L18_mean`。
代码是 `jevdrive/waymo_qwenvid.make_fx(compile=False)` 本身，只换数据读取；抽之前先用同一条路径重抽 16 个 Waymo 行，
和 `qwenvid_p3` 的存量逐行比对（`waymo_qwenvid.compare`），证明读取以外没有任何差别。

### CARLA rig（模仿 Waymo 的三相机，失配如实记）

Waymo 的标定从 WOD-E2E val 帧的 `context.camera_calibrations` 读出（6 个 sequence，三相机间一致到 ±0.3%）：

| 量 | Waymo（实测） | CARLA 里怎么做 | 残留失配 |
|:--|:--|:--|:--|
| 输出分辨率 | 972 × 1079（竖幅），JPEG | 同尺寸输出，JPEG 质量按 Waymo 的量化表估 | 无 |
| 焦距 | f ≈ 1113 px（1110–1116 逐帧浮动），HFOV 47.2° | f = 1113.5 px 固定 | 逐帧标定抖动没有模仿 |
| 主点 | c_u = 488.1，c_v = 719.2（主点在画面下 1/3，天空多） | 渲染一张主点居中的大图（1088 × 1560，f 相同）再按 Waymo 主点裁 | 无 |
| 畸变 | Brown 径向 k1 = −0.0736，k2 = −0.0366 | 对 pinhole 渲染图做同一组径向畸变的 remap | 切向项 Waymo 为 0，无 |
| 安装 | 后轴前 1.52 / 1.45 / 1.48 m，离地 1.806 m，yaw 0 / +45° / −45°，横向 +0.03 / +0.15 / −0.12 m，pitch ≈ 0.2° | 按后轴换算到 CARLA 车身坐标，同样高度和 yaw，pitch 0 | 车型：CARLA 的 Lincoln MKZ 2020 对 Waymo 的 Jaguar I-PACE |
| 帧率 | 10 Hz，clip 用 stride 2 | 相机 5 Hz（`sensor_tick` 0.2 s），clip 取连续 4 帧 | 无 |
| 成像 | 真实 ISP、rolling shutter、HDR | CARLA 默认自动曝光、bloom、lens flare | **没有模仿**，这是 sim↔real 差距的一部分，本来就是要量的 |

### 驾驶者与路线

- **Expert**：CARLA 0.9.15 自带的 `BehaviorAgent`（`normal` 档），跑在 Bench2Drive 的 leaderboard 里，沿 route 的稠密 plan 开。
  它是特权的：直接读地图、所有车辆 / 行人的状态和红绿灯状态，不看相机。
  盒子上另有 LEAD（2512.20563）的 PDM-Lite 系 expert（`third_party/scout/lead-cvpr2026`），开得更好，
  但要它自己的 Python 3.10 环境和改过的 leaderboard，接入成本超出 P4 的预算，不用；Bench2Drive 官方采数据的 Think2Drive 没有公开。
  BehaviorAgent 会停在堵住车道的静态障碍后面，所以要换道绕行的 scenario 类型不选。
- **背景交通**：leaderboard 2.0 自己的 BackgroundActivity（Traffic Manager 控制的车流）加各 route 的 scenario actor。
- **路线**：从 `bench2drive220.xml` 里按下面的规则选，种子固定，生成前把清单冻结在 `research/results/p4-carla-gap/routes.csv`：
  去掉 11 条已知会让服务器段错误的 Town12/13 路线（docs/carla.md），去掉 BehaviorAgent 过不去的类型
  （Accident*、ConstructionObstacle*、ParkedObstacle*、HazardAtSideLane*、VehicleOpensDoorTwoWays、YieldToEmergencyVehicle、ParkingExit、InvadingTurn），
  然后按 town 分层：12 个 town 每个都进，Town12 / Town13 多取，路口左右转类 scenario 优先，保证有转弯前的帧（pre-onset）。
  Bench2Drive 的路线很短（50–160 m），一条路线只出几十个可用帧，所以用**约 48 条路线**凑约 2 千帧，而不是 20 条 × 100 帧；
  路线多也正好让 domain classifier 的分组交叉验证有更多组。
- **记录**：每个 tick（20 Hz）记 ego 真值位姿、速度、加速度；每 0.2 s 存三相机 JPEG。
  标签按 WOD-E2E 的格式算：未来 20 个点（0.25 s 间隔到 5 s）、过去 16 个点（4 s）的位置 / 速度 / 加速度，
  全部在当前后轴坐标系里（+x 向前、+y 向左；速度和加速度也转到当前帧，与 Waymo 实测一致；
  Waymo 有 76% 的帧最后一个速度 / 加速度样本重复上一个，照抄）。intent（GO_STRAIGHT / LEFT / RIGHT）从 route plan 的路口指令取。
- **关键帧**：从每条路线里取「过去满 4 s、未来满 5 s、clip 完整」的相机帧，间隔 0.4 s；停着不动的帧最多保留到 Waymo 子集里静止帧的比例。
- 预算：`RESOURCE_LEDGER.md` 的 p4 流：vlm 32B 在跑时 ≤ 1 个 CARLA server（约 6 GB），之后 ≤ 2 个；≤ 6 核。
  CARLA server index 70（RPC 5500、TM 11500），和其他流不重叠。

### Waymo 侧

`qwenvid_p3` 覆盖的 P2/P3 冻结子集，19 663 帧 / 479 sequence，半 val 两个方向（第 3d 条的切分，seed 0）。
head 在 fit 半上拟合，在 eval 半和全部 CARLA 帧上各评一次，所以 Waymo 与 CARLA 的数是同一个 head 的数。
词表与 P3e 相同：train split 全部 logged future 上 k-means，K=1024，seed 0。

## 步骤

- [ ] 冻结路线清单；2 条路线的 smoke：确认三相机同帧、图像统计正常（不是 lavapipe 的空图）、位姿与相机帧对齐、
      ego 原点离地高度、每条路线的 wall time，据此给出总时长估计，再开全量
- [ ] 全量生成（`scripts/b2d_run.py --agent scripts/p4_carla_agent.py`，tmux `jev:p4-gen`，ledger 登记）
- [ ] 建 CARLA 索引（关键帧、past / future / intent）并抽 `qwenvid` 特征（先做 16 行 Waymo 等价检查），`features/carla_p4`
- [ ] 分析：Q1 AUC 全集 / 匹配子集 / 去均值 / PCA-k / 对照；Q2 词表覆盖；Q3 head 迁移、anchor 分布、probe 迁移
- [ ] 图（`research/plot_style.py`）进 `research/figs/`，小结果进 `research/results/p4-carla-gap/`，结果填回本文

## 指标定义（跑之前写死）

- **AUC_raw**：L2 logistic regression，特征按训练折标准化，5 折 GroupKFold（Waymo 按 sequence、CARLA 按 route 分组），
  out-of-fold 概率上的 ROC AUC。`L18_last`、`L18_mean` 各报一次，`L18_last` 是主 tap。
- **AUC_matched**：同上，但只在粗化精确匹配（coarsened exact matching）后的子集上：按 v₀ 分档（每 2 m/s 一档）×
  未来 3 s 的动作类（静止 / 直行 / 左转 / 右转，按 `waymo.subsets` 的 chord bearing 与位移判）分层，每层两边各取 min(n_W, n_C) 帧。
- **AUC_centered**：每个域减去自己的均值（在训练折上估）后再训分类器。它回答「差距是不是只是一个平移」。
- **AUC_PCA-k**：只用 Waymo 特征的前 k 个主成分（k = 1、4、16、64），回答差距集中在几维。
- **对照**：CARLA 内部 large map（Town12/13/15）对 small town 的 AUC、CARLA 白天对夜晚（sun altitude < 0）的 AUC；
  Waymo 内部两个 scenario cluster 之间的 AUC（按 sequence 分组）；Waymo 内部随机按 sequence 分两半的 AUC（噪声地板，应为 0.5 左右）。
- **词表覆盖**：oracle minADE、oracle minFDE，以及 uncoverable 比例（词表里没有任何 anchor 落进以真值为中心的 trust region，
  `traj.vocab_coverage`，3 s / 5 s，官方的速度缩放）。CARLA 全部关键帧对 Waymo 冻结子集全部帧；再报 v₀ 匹配后的数，**判据用匹配后的**。
- **head 迁移**：`ridge ego`（ego 96 维 + intent 4 维）、`ridge_late L18_last / L18_mean`（ego 基线残差上的特征 ridge），
  λ 按 P3(d″) 的 grouped CV 在 fit 半选；CTRV（恒速恒 yaw rate 圆弧，不拟合）作跨域不变的尺子。
  主量是视觉增量 Δ_vis = ADE(`ridge_late`) − ADE(`ridge ego`)，CARLA 上按 route bootstrap CI，全部帧和 pre-onset 各一行，
  与同一 head 在 Waymo eval 半上的 Δ_vis 并排。再报 v₀ 匹配后的 ADE 比值 CARLA / Waymo。
  另一行是**无标签自适应**：CARLA 特征用 CARLA 自己的均值和标准差标准化（per-domain standardization，不用任何 CARLA 标签），同一组权重。
- **anchor 分布**：`cls_late L18_last`（P3e 的分类头）top-1 anchor 在 CARLA 与 Waymo eval 半上的分布，JS divergence，
  和 top-1 命中真值最近 anchor 的比例；对照是真值最近 anchor 本身两域间的 JS divergence（轨迹分布本来就不同的部分）。
- **probe 迁移**：Waymo fit 半上训线性 probe（静止 vs 行驶；未来 3 s 左转 / 右转 / 直行），在 Waymo eval 半和 CARLA 上各报 AUC，
  迁移比 = (AUC_CARLA − 0.5) / (AUC_Waymo − 0.5)。

## 成功标准（跑之前写死）

**预期写在前面**：任何 sim 对 real 的线性 domain classifier 在 2560 维上几乎一定能分开，**AUC_raw 预计 > 0.99**，
所以 AUC_raw 不作判据，只作记录；决定判读的是差距的形状（AUC_centered、PCA-k）和 head 实际读出来的东西（Q2、Q3）。

| 问题 | 可用（usable） | 加自适应可用（usable with adaptation） | 不可用（not usable） |
|:--|:--|:--|:--|
| Q1 差距形状 | AUC_matched ≤ 0.80，或不超过最大域内对照 + 0.05 | AUC_centered ≤ 0.75（差距主要是平移，按域标准化能消掉） | AUC_centered > 0.90（CARLA 特征在不同方向上，不只是平移） |
| Q2 词表覆盖（v₀ 匹配） | minADE 比值 CARLA / Waymo ≤ 1.5 **且** uncoverable 比 Waymo 多 ≤ 2 个百分点 | 比值 ≤ 3 且多 ≤ 10 个百分点（补 CARLA anchor 或重聚类可救） | 比值 > 3 或多 > 10 个百分点 |
| Q3 head 迁移 | Waymo 标准化下 CARLA 全部帧 Δ_vis ≤ 0 且 CI 上界 ≤ +0.05 m，**且** 匹配后 `ridge_late` ADE 比值 ≤ 1.5 | Waymo 标准化下不过，按域标准化后过 | 按域标准化后 Δ_vis 仍 > +0.10 m，或 ADE 比值 > 2 |
| Q3b probe 迁移 | 两个 probe 迁移比都 ≥ 0.7 | ≥ 0.3（或按域标准化后 ≥ 0.7） | < 0.3 |

**总判**（对 P5 的含义）：

- **可用**：Q2、Q3 都「可用」。P5 直接把 Waymo 训的 head 放到 CARLA 配对上考，flip 率可以当成这个 head 的性质来读。
- **加自适应可用**：Q2 至少「加自适应可用」，Q3 或 Q3b 只在按域标准化后过。P5 照做，但特征先按域标准化，
  所有 P5 数字都带「经过无标签自适应」的限定；教材方向（CARLA 配对训 head 再回 Waymo）要额外证明，不能从 P4 推出。
- **不可用**：Q2 或 Q3 落在「不可用」。Waymo head 在 CARLA 上的输出不代表它在真实数据上的行为，P5 的「考卷」读不出东西；
  P5 要么改成在 CARLA 上训、在 CARLA 上考（放弃跨域的那半句），要么换真实数据渲染的配对（第 19 条的 HUGSIM 一类）。

补一句为什么 Q1 不单独定生死：P5 读的是配对**差**（x⁺ 与 x⁻ 只差一处因素），一个常数平移在差分里一阶抵消；
所以「AUC 很高但去均值后很低」对 P5 是可以接受的，「去均值后仍然很高」才说明 CARLA 的变化方向和真实数据不在一个子空间里。
Q3b 的 probe 迁移是这句话的直接检验。

## 结果

跑完再填。run dir：`$DATA_DIR/runs/p4_carla/`。
