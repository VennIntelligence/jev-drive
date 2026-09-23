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
| 帧率 | 10 Hz，clip 用 stride 2 | 相机每 tick（20 Hz）渲染、每 4 tick 存一组（5 Hz），clip 取连续 4 组 | 无 |
| 成像 | 真实 ISP、rolling shutter、HDR | CARLA 默认自动曝光、bloom、lens flare | **没有模仿**，这是 sim↔real 差距的一部分，本来就是要量的 |

### 驾驶者与路线

- **Expert**：CARLA 0.9.15 自带的 `BehaviorAgent`（`normal` 档），跑在 Bench2Drive 的 leaderboard 里，沿 route 的稠密 plan 开。
  它是特权的：直接读地图、所有车辆 / 行人的状态和红绿灯状态，不看相机。
  盒子上另有 LEAD（2512.20563）的 PDM-Lite 系 expert（`third_party/scout/lead-cvpr2026`），开得更好，
  但要它自己的 Python 3.10 环境和改过的 leaderboard，接入成本超出 P4 的预算，不用；Bench2Drive 官方采数据的 Think2Drive 没有公开。
  BehaviorAgent 会停在堵住车道的静态障碍后面，所以要换道绕行的 scenario 类型不选。
- **背景交通**：leaderboard 2.0 自己的 BackgroundActivity（Traffic Manager 控制的车流）加各 route 的 scenario actor。
- **路线**（2026-09-23 按用户要求改版，改在生成任何正式数据之前；第一版 48 条、剔除障碍类的清单作废）：
  帧不能大多是直路巡航，所以路线按 Bench2Drive 的**交互 scenario family** 来取，而不是按 town 取：
  `bench2drive220.xml` 的 44 个 scenario 类型（路口左 / 右转、有无信号灯、停车标志、换道 / 汇入 / 驶出高速、
  施工 / 事故 / 路边障碍、行人 / 自行车 / 车辆横穿、急刹、cut-in、闯红灯、让行等）**每类取 2 条**，
  同一类里优先取不同 town（种子 0），去掉 11 条已知段错误的路线，共约 88 条，覆盖全部 12 个 town（含 Town12/13 和小 town），
  天气和昼夜用各路线 XML 自带的设定（sun altitude 从 −90° 到 90°，有雨有雾），不另改。
  清单冻结在 `research/results/p4-carla-gap/routes.csv`。**第二波（预先写死，只在第一波横向 pre-onset 候选帧不足 600 时启动）**：
  路口转弯类和换道 / 汇入类每类再加第 3 条，清单同样先冻结（`routes_wave2.csv`）。
- **第二波已触发，并扩大**（2026-09-23 17:40，wave 1 跑到 29 / 88 条时，只看轨迹标签、没有看任何特征）：
  29 条路线只出了 **39 个横向 pre-onset 候选**，外推到 88 条约 120 个，远低于 600。原因查过：
  Bench2Drive 的路线只有 16–25 s，转弯常在出生后 2–9 s 就开始，而「过去满 4 s」把转弯前那一段整段排除了。
  两处改动，都只依赖轨迹：(1) 过去历史只要求**至少 1 s 是真实的**，不足 4 s 的部分用出生时的状态补（车停在出生点，速度、加速度为 0），
  Waymo 的 1 s 运动护栏仍然看真实运动；每帧记 `past_padded_s`，head 迁移再报一行「只用未补齐的帧」作对照。
  (2) 第二波从「每类加 1 条」扩成**路口转弯类和换道 / 汇入类剩下的全部路线**（清单 `routes_wave2.csv`，生成前冻结），
  32B 结束后用 2 个 server 跑。横向 pre-onset 仍不够 600 的话，N 按公式缩小，缺口如实报。
- **expert 做不到的动作如实记**：BehaviorAgent 遇到挡住本车道的静态障碍（施工、事故、路边停车）只会停下等，不会借道绕行，
  所以「绕障碍 nudge」这一类 pre-onset 预计几乎为零，**不拿直行帧补**，在结果里写明缺口；
  这些路线仍然有用，它们给出「接近障碍时开始刹车」和排队停车的帧。换道只在 route plan 本身带 CHANGELANE 指令时发生。
  LEAD 的 expert 需要 `leaderboard_autopilot` / `scenario_runner_autopilot` 两个 fork，盒子上没有（`3rd_party/` 下只有 CARLA）。
- **记录**：每个 tick（20 Hz）记 ego 真值位姿、速度、加速度；每 0.2 s 存三相机 JPEG。
  标签按 WOD-E2E 的格式算：未来 20 个点（0.25 s 间隔到 5 s）、过去 16 个点（4 s）的位置 / 速度 / 加速度，
  全部在当前后轴坐标系里（+x 向前、+y 向左；速度和加速度也转到当前帧，与 Waymo 实测一致；
  Waymo 有 76% 的帧最后一个速度 / 加速度样本重复上一个，照抄）。intent（GO_STRAIGHT / LEFT / RIGHT）从 route plan 的路口指令取。
- **关键帧：按动作分层抽，配额写死**（见下一节「分层与配额」）。
- 预算：`RESOURCE_LEDGER.md` 的 p4 流：vlm 32B 在跑时 ≤ 1 个 CARLA server（约 6 GB），之后 ≤ 2 个；≤ 6 核。
  CARLA server index 70（RPC 5500、TM 11500），和其他流不重叠。

### 分层与配额（2026-09-23 按用户要求加，生成数据之前定）

候选帧是每条路线里所有「clip 完整（4 帧、间隔 0.2 s）、过去满 4 s、未来满 5 s」的相机帧（0.2 s 一个）。
每个候选帧**只按 expert 录下来的未来轨迹和 ego 历史**打标签，阈值与 Waymo judge 完全相同（`waymo.subsets`，
第 3、3c 条的定义，`waymo_ladder` / P0 用的同一份代码），**不看特征、不看 head 输出**。按下面的优先级归到唯一一层：

| 层 | 定义（Waymo 的判据原样照搬的写「同 Waymo」） | 配额 |
|:--|:--|:--|
| pre-onset，横向 | 同 Waymo `pre_onset`：过去 1 s 走了 ≥ 1 m、当前 \|yaw rate\| < 1°/s，3 s 处 chord ≥ 3 m 且 \|bearing\| > 5° | ≤ 600 帧（0.20 × 3000），不够全要 |
| pre-onset，起步 | 当前 v₀ < 0.5 m/s，3 s 内走出 ≥ 3 m（start-from-stop） | ≤ 225 帧（0.075 × 3000） |
| pre-onset，开始刹车 | v₀ ≥ 3 m/s、当前纵向加速度 ≥ −1 m/s²（还没在刹），3 s 处速度 ≤ 0.5 v₀（brake-for-agent，也含为红灯刹） | ≤ 225 帧（0.075 × 3000） |
| in-turn | 同 Waymo `turn_yaw`：\|yaw rate\| ≥ 5°/s | 25% × N |
| plain straight | 同 Waymo `straight_yaw` | 25% × N |
| stop / queue | v₀ < 0.5 m/s 且 3 s 内移动 < 1 m | 10% × N |
| other | 以上都不是（弯道上的小 yaw rate、护栏条件不满足等） | 5% × N |

先抽 pre-onset 三类（各自不超过上限，候选不够就全要），**它们合起来定为 35%**，N = pre-onset 帧数 / 0.35，
其余四层按上表占满另外 65%。这样 pre-onset ≥ 30%、in-turn ≥ 25%、plain straight ≤ 30%，剩下是停车 / 排队。
上限是为了不让数量很多的「开始刹车」挤掉稀缺的横向 pre-onset（在 Waymo val 上试算：开始刹车占 7.5% 的帧，横向 pre-onset 只占 1.4%）。

横向 pre-onset 再按未来轨迹分子类，只用于报告：**路口转弯**（5 s 处航向变化 ≥ 30°）、**换道**（5 s 处横向偏移 ≥ 2 m 且航向变化 < 15°）、
**其他横向**（绕障碍 nudge、弯道，余下的）。

抽取在每层候选里均匀随机（种子 0），候选不够就全要，缺口写进结果，不拿别层补。
s_ego 式难度（用 Waymo fit 的 `ridge ego` 在每帧上的 5 s ADE，按 Waymo eval 半的十分位切档）作为每帧的一列，只报告不参与抽取。

**组成对照**：CARLA 抽出来的集合与 Waymo val 全集、Waymo 冻结子集三列并排报：v₀ 分位数、\|yaw rate\| 与 5 s 航向变化的分位数、上面七层的占比、
s_ego 十分位占比、intent 占比、昼夜 / 天气（CARLA 侧）。**每一张结果表都带每层的 n**，AUC、词表覆盖、head 迁移除了汇总也逐层报，
逐层时 Waymo 与 CARLA 用同一层的帧比。

### Waymo 侧

`qwenvid_p3` 覆盖的 P2/P3 冻结子集，19 663 帧 / 479 sequence，半 val 两个方向（第 3d 条的切分，seed 0）。
head 在 fit 半上拟合，在 eval 半和全部 CARLA 帧上各评一次，所以 Waymo 与 CARLA 的数是同一个 head 的数。
词表与 P3e 相同：train split 全部 logged future 上 k-means，K=1024，seed 0。

## 步骤

- [x] 2 条路线的 smoke（第一版清单里的 28035 / 2164）：Large Map 上 `sensor_tick` 让相机不按 4 tick 触发（Town12 一条路只收到 4 组），
      改成相机每 tick 渲染、每 4 tick 存一组：Town12 一条 21 s 的路线存 107 组、0 丢失，约 2.5 min / 路线；ego 原点在地面（bbox z = extent z）
- [ ] 按 scenario family 冻结路线清单（约 88 条），估计 88 × 2.5 min ≈ 3.7 h（1 个 server），32B 结束后 2 个 server
- [ ] 全量生成（`scripts/b2d_run.py --agent scripts/p4_carla_agent.py`，tmux `jev:p4-gen`，ledger 登记）
- [ ] 建 CARLA 索引（关键帧、past / future / intent）并抽 `qwenvid` 特征（先做 16 行 Waymo 等价检查），`features/carla_p4`
- [ ] 分析：Q1 AUC 全集 / 匹配子集 / 去均值 / PCA-k / 对照；Q2 词表覆盖；Q3 head 迁移、anchor 分布、probe 迁移
- [ ] 图（`research/plot_style.py`）进 `research/figs/`，小结果进 `research/results/p4-carla-gap/`，结果填回本文

## 指标定义（跑之前写死）

- **AUC_raw**：L2 logistic regression，特征按训练折标准化，5 折 GroupKFold（Waymo 按 sequence、CARLA 按 route 分组），
  out-of-fold 概率上的 ROC AUC。`L18_last`、`L18_mean` 各报一次，`L18_last` 是主 tap。
- **AUC_matched**：同上，但只在粗化精确匹配（coarsened exact matching）后的子集上：按 v₀ 分档（每 2 m/s 一档）×
  「分层与配额」一节的 7 层分格（原来写的是 4 类动作，随分层一起改成 7 层，改在任何 CARLA 特征之前），
  每格两边各取 min(n_W, n_C) 帧。词表覆盖和 head 迁移的「匹配后」也用同一套格子给 Waymo 重新加权。
- **AUC_centered**：每个域减去自己的均值（在训练折上估）后再训分类器。它回答「差距是不是只是一个平移」。
  **分类器用小 MLP（2560 → 256 → 2，class-balanced，30 epoch），不用线性**。原来写的是同一个线性分类器，
  这是个定义错误，在抽任何 CARLA 特征之前、用 Waymo 内部对照试跑时发现的：两类各自去均值后，
  class-balanced logistic loss 在 w = 0 处梯度为零且是凸的，线性分类器的 AUC 恒等于 0.5，量不出任何东西。
  改成 MLP 后它能用协方差（形状）上的差别。为了可比，AUC_raw 也并排报一个 MLP 版本，
  再加一个「按域 z-score（均值和逐维标准差都按域去掉）」的 MLP 版本，对应 Q3 的按域标准化。
  第二处修正（同样在 CARLA 特征之前，用 Waymo 当假 CARLA 的试跑发现）：去均值如果用训练折上的类均值，
  组数少的一类（CARLA 约 90 条路线）在测试折里会带着路线之间的均值差，MLP 把它当成域信号，
  Waymo 对 Waymo 的试跑读出了 0.83；改成每类用它全部帧的均值，null 又读出 0.36（两折被推到均值两侧）。
  最后定为**每类在每个 split 内用自己的统计量**：训练折的行用训练折的类均值、测试折的行用测试折的类均值
  （transductive，和部署时按域标准化一样，不用任何任务标签），null 三次抽取读出 0.46 / 0.46 / 0.63，回到 0.5 附近。
  透明起见：这第三处修正是在一次 12 条路线、353 帧的 dry run（只为跑通代码）之后做的，那次 dry run 也读出了 CARLA 对 Waymo 的数；
  修正的依据是 null 偏到 0.36，不是 CARLA 的数，判据阈值（相对 null 的 0.10 / 0.25）在 dry run 之前已提交，没有改。
  并加**零差距对照（null）**：从 Waymo 里随机取与 CARLA 路线数、帧数相同的 sequence 当「伪域」（主 tap 抽 3 次，报均值），
  所有 AUC 变体都在它上面再跑一遍，读数是「没有域差距、但组数和 CARLA 一样少」时这个估计量给多少。
  Q1 的阈值按「CARLA 的数减去 null 的数」来读：AUC_centered 的判据改为 AUC_centered − null_centered ≤ 0.10 为「加自适应可用」、
  > 0.25 为「不可用」。
- **AUC_PCA-k**：只用 Waymo 特征的前 k 个主成分（k = 1、4、16、64），回答差距集中在几维。
- **对照**：CARLA 内部 Large Map（Town11/12/13/15）对 small town 的 AUC、CARLA 白天对夜晚（sun altitude < 0）的 AUC，
  这两个线性、MLP 各报一次；Waymo 内部最大的 4 个 scenario cluster 各自对其余的 AUC（按 sequence 分组）；
  Waymo 内部随机按 sequence 分两半的 AUC（噪声地板，应为 0.5 左右）。
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
| Q1 差距形状 | AUC_matched ≤ 0.80，或不超过最大域内对照 + 0.05 | AUC_centered（MLP）比 null 高 ≤ 0.10（差距主要是平移，按域标准化能消掉） | AUC_centered（MLP）比 null 高 > 0.25（CARLA 特征在不同方向上，不只是平移） |
| Q2 词表覆盖（v₀ 匹配） | minADE 比值 CARLA / Waymo ≤ 1.5 **且** uncoverable 比 Waymo 多 ≤ 2 个百分点 | 比值 ≤ 3 且多 ≤ 10 个百分点（补 CARLA anchor 或重聚类可救） | 比值 > 3 或多 > 10 个百分点 |
| Q3 head 迁移 | Waymo 标准化下 CARLA 全部帧 Δ_vis ≤ 0 且 CI 上界 ≤ +0.05 m，**且** 匹配后 `ridge_late` ADE 比值 ≤ 1.5 | Waymo 标准化下不过，按域标准化后过 | 按域标准化后 Δ_vis 仍 > +0.10 m，或 ADE 比值 > 2 |
| Q3b probe 迁移 | 两个 probe 迁移比都 ≥ 0.7 | ≥ 0.3（或按域标准化后 ≥ 0.7） | < 0.3 |

Q2、Q3 的判据**同时**用在汇总（按 v₀ × 动作匹配后）和 pre-onset 层（三类合并）上，取两者中更差的那一档：
P5 考的正是 pre-onset 这种时刻，汇总过关而 pre-onset 不过关算不过关。

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
