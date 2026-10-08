# op_parity turn selector 跨榜预登记：第 193 条的 selector（N7，门 B）原封不动搬到 WOD-E2E val，读 RFS（2026-10-08，任何本线 RFS 读数之前写定）

## 问题

只问一件事：**在 navtrain 的 simulator 分数上学出来的 selector，不重拟合、不调、不重选，接在 SH30 的 plan 上，WOD val 479 个 rater 帧的 RFS 有没有变好。** 这是第 177 条的跨榜判据在 selector 上的一次检验：
有增益 = 「从评价信号学驾驶价值」至少在这一个机制上跨榜成立；没有 = 它是 navtrain / NAVSIM 的榜内技巧。

## 已有的、不重测

第 191、193、194 条的权重、族（F19：横向偏移 × 曲率增益 × 速度比例）、门 B（SH30 自己的 plan 4 s 航向变化 ≥ 20°）、配置，**一概不动**。WOD 侧：第 162、168、172、173、175、176 条（SH30 在 WOD val RFS 7.734，shipped 8.005，WLG 8.178–8.187；
特权 best-of-20 候选 +1.07；薄 head +0.148 全在转弯意图帧）。不用 WOD 标签拟合任何东西；不读 WOD test。

## 流水线与设置

- **输入**：WOD 决策 155 harness 的 SH30-F-s{0,1} 服务 ONNX（`runs/op_parity/hugsim/onnx/pp-SH30-F-s*.onnx`，intent bias 同归档），加 taps `view_39 / select_4 / mean`，真实帧、20 Hz、9 个真实 policy slot，只跑 479 个 rater 帧
  （`scripts/turn_selwod.py extract`）。最后一步的原始输出给 plan、路沿；tap 给视觉 token、隐状态。ego 特征取 `runs/op_parity/wod/bias-*.npz` 里归档的 `ego`（`pp_wod.wod_ego`）。
- **selector**：`turn_selbench.load_bundle()`（N7，F19 × pc，5 个初始化平均，191 的配置），边标定取 `self_consist/calibration.json` 里同 seed 的 SH30 项；特征、自路沿 margin、`picks_of(m = 0)` 与 navtest 路径同一份代码
  （`turn_selbench.curb_margins / infer`，`turn_dewater.plan_desc`）。选择器看的仍是 plan 的 8 个后轴位姿（0.5 … 4 s），和 navtest 一样。
- **候选怎么落到 WOD 的 20 个航点**：F19 的 19 个候选对 plan 的 **10 个后轴位姿（0.5 … 5 s）** 作用（`turn_ceiling` 的 curv / offset / speed，speed 的实现按位姿数推广，8 位姿上逐位等于原实现：闸门 G-gen），
  变化量（候选减原位姿，t = 0 为 0，按时间线性插到 0.25 … 5 s）加到 harness 输出的 20 个航点上，所以未改的帧就是 plan 本身。**主变体 = 这个 0.5–5 s 版**。
  敏感性变体 H4：候选只作用于 8 个位姿（0.5–4 s），变化量 4 s 后保持（闭环第 194 条的 `apply_delta` 做法）。两者都在报告里；判定线只用主变体。
- **门**：B（主）= plan 4 s 航向变化 ≥ 20°（门限不调）；A = 每帧都用，只报告；0 = 恒等。
- **种子**：SH30 两个 seed 各一次完整流水线，主读数用逐帧分数的 seed 均值（同归档 7.734 的口径）。

### 与 navtest 设置的差别（selector 看到的分布在这些地方变了）

1. **相机映射**：navtest 的 SH30 吃缓存的 warp 帧（前视）；WOD 吃 FRONT / FRONT_LEFT / FRONT_RIGHT 三相机经 `camgeom` 重投影到 openpilot road / wide 标定帧的真实帧。视觉 token（`view_39`，当前帧）和隐状态由这条路径出，selector 训练时见到的是 fold 模型在 navtrain warp 路径上的。
2. **帧率与历史 slot（第 176 条）**：navtrain 臂训练用 8 个真实 + 1 个零 slot、2 Hz 关键帧 warp；WOD harness 是 20 Hz 实帧、9 个真实 slot（第 176 条已证明把 WOD 改成 8 + 零只会更差，所以保持 9 个真实）。selector 输入里的隐状态 `select_4 / mean` 因此来自与训练不同的 slot 配置。
3. **自路沿 margin**：路沿 head 同一个，footprint 同一个（nuPlan Pacifica 角点，原点后轴；WOD 的车辆系原点也是后轴），相机位置取 WOD FRONT 外参的 (x, y)（navtest 取缓存的 cam），标定 (s, b) 用 navsim 上按 seed 拟合的那一组，没有在 WOD 上重标。
4. **视野**：selector 的输入和门 B 的判据是 4 s（8 位姿），RFS 在 3 s 和 5 s 取点（trust region 横向 ±1.0 / 1.8 m、纵向 ±4.0 / 7.2 m，低速收窄）。主变体把候选延到 5 s；H4 是 4 s 内变换。
5. **打分**：selector 在 navtrain 上学的是 simulator 子分（DAC / NC / TTC / EP / 舒适度的组合，`pc` 口径），WOD 的 RFS 是对三条 rater 轨迹的 trust-region 近似，没有地图概念，对速度剖面更敏感（第 168 条：特权上限里速度占大头）。
6. **时序**：WOD 每序列只有一个 rater 帧，没有 EC 的跨帧配对问题；navtest 上 EC 吃掉增益的 29%，这里不会发生。
7. **门覆盖**：门 B 在 WOD 上触发 55 / 56 帧（11.5% / 11.7%，两个 seed），navtest 是 26.1%；其中 34 / 35 帧是指令为左 / 右的帧（共 52 个转弯意图帧），21 帧是指令直行但 plan 在转。

## 读数

- **R1（主）**：门 B、主变体，seed 均值逐帧 RFS 对同一次运行的 `ts0`（恒等）的差，cluster-mean RFS（10 个场景 cluster 的无权均值，官方聚合），按 sequence 配对 bootstrap 的 95% CI
  （`wod_launch_report.Ctx.ci`，B 与 WOD 各线同一份重采样；另附 `jevdrive.stats.paired` 按 sequence 的逐帧均值 CI 作交叉核对）。两个 seed 的逐 seed 点估计也报告。
- **R2 分层**：10 个 WOD cluster；转弯意图（左 / 右）对其他；门 B 内 / 外；v0 分层（静止 / 起步 / 行进）；night / day。
- **R3 四个背景读数**（让零结果可解读）：
  1. 特权 best-of-F19 的 RFS 上限（总体、门内、转弯意图帧；分轴：O3 / K3 / V3 / F7 / F19；主变体与 H4；两个 seed）；
  2. 门 B 触发率与 selector 的选择（偏移 / 曲率 / 速度三个轴的边际分布，对照 navtest 门 B 的同一组量，来自 `turn_selbench/select/*navtest.npz`）；
  3. selector 的预测增益与候选实际 RFS 增益的秩一致性（门内帧：Spearman、top-1 一致、pick 在 18 个移动候选中的名次），以及「门完美」的增益（只在 pick 的实际 RFS 增益 > 0 的帧上用它）；
  4. 逐候选估值：19 个候选各自在门内帧上的静态实际增益，pick 的直方图；分解 恒等 / selector / 随机移动候选 / 特权最佳单一常数候选 / 特权逐帧最佳，用来区分「族里没东西」和「族里有东西但 pick 错」。
- **R4 次要（只在便宜且干净时，已跑通所以做）**：同一 selector 叠在 WLG-full-s{0,1}（WOD 配方的模型）的 plan 上，**失配情形**：隐状态和 plan 来自与 selector 训练时不同的 adapter 权重（路沿 head 同底座；边标定仍用同 seed 的 SH30 项）。
  同一套读数，只作探索，不设判定线。
- 辅助：门 A 的同一组数（navtest 上它有害）。

## 线（固定，看到 RFS 后不改）

在 R1（门 B、主变体、seed 均值、对 `ts0`）上：

- **迁移（transfers）**：95% CI 下界 > 0，**且**两个 seed 的点估计都 > 0。
- **伤害（harms）**：95% CI 上界 < 0。
- **不迁移（does not transfer）**：其余全部（CI 含 0；或下界 > 0 但某个 seed 点估计 ≤ 0，记为「不一致，按不迁移」）。
- 报告里同时给 CI 上界：「不迁移」只意味着排除了大于上界的增益。H4、门 A、WLG 不进判定线。

### 功效声明（先说清楚这个检验能结论什么）

- 门 B 内 55 / 56 帧（两 seed 取并 57 帧），占 479 帧的 11.9%，集中在 Intersections（19）、Foreign Object Debris（9）、Pedestrian（7）、Cyclist（6）、Single-Lane（5）等；改动只发生在这些帧上，其余帧恒等，所以总体差的噪声只由这 57 帧决定。
- 门内帧上 SH30 的 RFS 均值 6.56（总体逐帧均值 7.73）。把 navtest 门内的相对增益（+2.11 / 81.15 = 2.6%）原样缩放到 WOD：总体 cluster-mean 约 **+0.02**。
- 噪声（只用归档读数做校准，没有读任何新的 RFS）：门内帧上 SH30 与 shipped 的逐帧 RFS 差 SD 2.8（这是大改动的上界，候选改动 ≤ 0.5 m / 15% / 20%，更小），门内帧 RFS 本身 SD 2.3。
  取逐帧差 SD 0.5 / 1.0 / 2.0，总体差的 SE ≈ SD / √57 × 0.12 ≈ 0.008 / 0.016 / 0.032，80% 功效的最小可检出效应（2.8 × SE）≈ **0.023 / 0.045 / 0.09**。
- 所以：**按 navtest 缩放的增益（+0.02）落在最小可检出效应之下或刚好在它上面**；这个检验能检出的是「增益至少 2 倍于 navtest 缩放值」的迁移，检不出 navtest 量级的迁移。
  「迁移」成立需要 WOD 上的增益大于约 0.03–0.09；「不迁移」的含义是 CI 含 0，并同时给出排除的上界，**不是**证明增益为 0。「伤害」的检出同样只对大于 MDE 的伤害有力。
  门内 / 转弯意图子集（34–35 帧）上的 CI 更宽，只作描述。实际 SE 和 MDE 在结果文档里按实测重算。

## 闸门（不过就停下报告）

全部是 plan 层或恒等层的，不读 selector 的 RFS。

- **G-gen**：`transform_n` 在 8 位姿输入上与 `turn_ceiling.transform` 逐位相等（19 个候选，最大绝对差 0）。冒烟与全量均已过（0.0）。
- **G-map**：harness 的 20 个 WOD 航点与由密集 plan 经同一杠杆臂映射重算的航点最大差 ≤ 1e-4 m。已过（< 1e-5 m）。
- **G-eqv**：这里的批量 select 与闭环线的逐帧 `turn_selhug.Selector.run` 在 479 帧上选出相同候选（一致率 1.000）。已过（四个臂都是 1.000）。
- **G-feat（同权重的 plan 层恒等，按 docs/long-runs.md 的 fp16 条：行占比加错模型对照，不用全点最大值）**：带 tap 的运行（独立构建的 TensorRT engine）与归档运行的 WOD 航点最大差 ≤ **0.25 m**（两个 fp16 ulp，在 128–256 m 处，WOD 航点可到这个量级），行占比 ≤ 2%，并要求错模型对照（另一个 seed 的归档航点）有超过 50% 的行越界。
  **容差在看过全量 plan 层噪声之后（不是 RFS）定：** 最初的规则是「每坐标 2 个 fp16 ulp，下限 0.03 m」（把 plan_pos 前 23 点逐坐标比），全量抽取后发现它因近零横向坐标的 ulp 极小而 35–44% 的行越界，
  而航点级的实际差是：中位数 0.03 m，99 分位 0.10–0.12 m，最大 0.19 m，错 seed 对照中位数 0.42 m，越过 0.25 m 的行占 71–72%。所以改成航点级 0.25 m。已过（越界行 0%，对照 71–72%）。
- **G-id（RFS 层恒等）**：`ts0` 的 seed 均值逐帧 RFS 对归档 SH30 的逐帧 RFS：cluster-mean 差 ≤ 0.01（归档 7.734），且 ≥ 95% 的帧差 ≤ 0.05。对照：归档 s0 与 s1 逐帧差在 0.05 内的帧只占 64%。
  不过就停下报告；若只是 engine 噪声所致，允许在**任何非恒等 RFS 被读之前**用附记改成「以归档航点为底、selector 的变化量加在其上」（此时 `ts0` 按构造等于归档，G-id 不再成为检验，G-feat 是唯一的恒等证据）。

## 读数前已看过什么（披露）

抽取与 select 阶段已跑完（plan、门覆盖、选择分布、plan 层闸门），**没有读任何候选、pick 或 `ts0` 的 RFS**。读过的 RFS 只有归档的（SH30 两 seed、shipped：校准功效用的门内 SD、s0 对 s1 的 G-id 对照）。
`processed/wod_zeroshot/sets.json` 在盒子上缺失（harness 的 span 表），本线用同一段索引代码重建了 479 个 rater 序列的 span 到自己的目录（`turn_selwod/spans_rater.json`），共享文件没动；这条缺失要告诉别的线。

## 成本与停止

预算约 3 h 墙钟、2 card-h、40 core-h。实测抽取 0.83 s / 帧（单卡 479 帧约 7 min，含首次 TensorRT 构建几分钟），4 个臂并行；select 每臂 < 1 min；报告 CPU 分钟级。远低于预算，没有超 1.5 倍的风险。

## 不做

在 WOD 标签上训练 selector、混合两个榜、帧间一致的 selector、改 N7 / 族 / 门限、WOD test、任何在 RFS 读数之后的变体选择。这些需要什么写在结果文档里，不开工。
