# op_parity turn selector HUGSIM 预登记：SH30 + selector（第 193 条门 B）的闭环检验（2026-10-08，任何闭环分数读之前写定）

## 问题

第 193 条：SH30 + N7 selector 门 B（只在 SH30 自己的 plan 4 s 转角 ≥ 20° 的步上用 selector）开环官方 navtest EPDMS +0.53 [+0.24, +0.85]，logged 转弯 token +2.07；代价是 EC（相邻帧 pick 不一致）和一点 LK。
第 186 条的保留意见：候选在 4 s 开环上打分，一个在 4 s 内看起来更好的 pick 可能只是把出路推迟；逐帧切换也可能让闭环不稳。闭环没测过。
本线只问：**在 HUGSIM 闭环里，SH30 + selector（门 B，权重和门限与第 193 条逐位相同，不重训、不调）对 SH30 是更好、一样还是更差，尤其在 23 条转弯路线上。**
同一 harness、同一 preset（`spec_plan_smooth`）、同一场景集（all64 / turn23）、两个 SH30 seed，所以和第 170 条的存档 SH30 run 逐场景配对。

## 已有的、不重测

第 170（SH30 HUGSIM 64：HD 0.439，turn23 0.329，强 hinge 在闭环转弯路线上 −0.023 [−0.049, −0.002]）、186、191、193 全部数。selector 权重 = `turn_selbench` 的 `n7_f19_bundle.pt`（G-repro 已过）；每 seed 的路沿校准取 `self_consist/calibration.json`；门限 20° 不调。
第 153、159：HUGSIM 急弯失败主要是入弯速度（9.8 对 WA-JEPA 2.6–2.8 m/s），不是路径；ax 置零只降 1.09 m/s。**这决定了增益的合理量级**：selector 的候选是「侧向偏移 ±0.5 m、曲率增益 0.85/1.15、速度缩放 0.8/1.2」，
HUGSIM 的横向指令只读 plan 0.5–1.5 s 的平均曲率（`spec_plan_smooth`），偏移候选对这个量几乎没有作用力（偏移用 6 m 平滑起坡，0.5–1.5 s 的航向只改一点），曲率增益直接缩放它，速度缩放通过 plan 弧长影响纵向。
所以预期闭环增益**小于**开环的 +2 EPDMS 折算：turn23 HD 上合理区间 0 到 +0.03（SH30 配对 CI 半宽约 ±0.025，见第 170 条），大于 +0.05 不预期；负向风险是逐步切换导致转向抖动（EC 的闭环面）。

## 把 selector 接进 HUGSIM（最小改动，与 navtest 设定的差别）

SH30 在 HUGSIM 里是：bias server（torch，SH30 的 adapter）+ policy server（ONNX 带 `intent_bias`，TensorRT）。selector 需要的量全部来自这一条路径，不加任何模型：

| selector 输入 | navtest（第 193 条） | HUGSIM（本线） |
|---|---|---|
| 未池化冻结视觉 token（32×512，当前帧） | 缓存的 `front[:, -1]`（warp 帧） | policy ONNX 的 `view_39`（新增图输出，`OPModel(taps=...)`；`--taps`），模型自己渲染的 openpilot 帧 |
| SH30 隐状态 | torch 口 `select_4` / `mean` | 同一个 ONNX 的 `select_4` / `mean`（含 adapter bias），同一步的输出 |
| 路沿 | 模型输出 road_edges μ | 同一步的输出（policy reply 新增 `road_edges`） |
| ego 20 维 + plan 描述 15 维 | tab.ego + 导出位姿 | `parity_hugsim` 的 ego 特征（模型时钟，与 bias server 同一输入）+ 由 plan 导出的 8 个后轴位姿（`to_rear`，lever，同 navsim 导出） |
| 自路沿 margin | NAVSIM 自车 footprint，相机 `cam`（NAVSIM CAM_F0 位置） | **同一个 NAVSIM footprint**（HUGSIM 车型随数据集变，不改，记为差别）；相机 = 数据集的 `rear_offset`，y = 0 |
| 门 B | 导出位姿 4 s 航向 ≥ 20° | 同一函数，模型时钟（plan 的 4 s 模型时间 = 3.2 s 仿真时间） |
| 被选候选如何生效 | 8 个位姿写入 pred 文件 | 候选 − 恒等在 8 个位姿上的差，按时间插值到 33 点 plan（0 s 处为 0，4 s 后保持），回到相机系；plan 路点（纵向）和 `curvature_smooth`（横向，0.5–1.5 s 平均曲率）由新 plan 重算（加上变化量，所以无变化时逐位不变） |

另外的差别（都不修，列出）：视觉 token 来自 HUGSIM 渲染而不是 warp 帧（selector 在 warp 帧上训练，与 navhard gimm 帧同类问题）；闭环每步一次决定（0.25 s 仿真步），没有时间一致性；HUGSIM 的 plan 只用到 3 s 仿真时间（模型 3.75 s）；
候选里的偏移在 plan 末端起作用，而控制只读 0.5–1.5 s 曲率，所以 selector 在这里的作用力小于开环（见上）；速度缩放在 4 s 后保持 4 s 处的差。接口从 `ts*` 选项进入，bench 模型 `SH30-F-s{0,1}:tsB`（主）与 `:ts0`（强制恒等，selector 前向照跑、日志照记，但不改 plan）。门 A 不做（第 193 条：A 不可部署）。

## 阶段与闸门（前一阶段不过不进下一阶段）

- **G-eqv（离线，GPU，几分钟）**：逐步 `Selector`（HUGSIM 用的同一份代码）对 navtest 行复现批处理 `select` 阶段：门一致率 = 1.000，pick 一致率 ≥ 0.98，候选位姿经「差值 → 33 点 plan → 重取 8 点」的往返误差记录。不过则修，不开始闭环。
- **G-id（身份闸门）**：`ts0` 对存档 SH30 run（第 170 条的 `SH30-F-s0_spec_plan_smooth`，同 preset），在固定的 11 个场景上（turn23 每隔 4 个取 6 个 + 非转弯 41 个每隔 10 个取 5 个，`scripts/turn_selhug_s11.txt`）。
  HUGSIM 的确定性来自存档：同模型同 seed 的重跑（P2-F-s0/s1 的 rr1 / rr2）逐场景 HD 完全一致 92–97%，|ΔHD| < 0.01 的占 98%，结局相同 98%，最大偏差 0.58（一个混沌场景）。
  但 ts0 的 ONNX 多了三个图输出，TensorRT 引擎不同，数值不会逐位相同。判据：(i) 11 个场景里结局相同 ≥ 9 个；(ii) |ΔHD| < 0.05 的 ≥ 9 个；(iii) 平均 |ΔHD| ≤ 0.02；(iv) 在结局相同的场景里，前 5 步 plan 路点（日志 `model_pos`）最大差 ≤ 0.05 m（fp16 引擎差）。
  **不过的话**：先在隔离里比带 / 不带 tap 的 ONNX 输出；若是 tap 引起的数值差，则把 `ts0` 全量（all64 × 2 seed）当作配对基线，同时报对存档 SH30 的差，并在结果里说明；若是接线问题则修。
- **G-pilot（11 个场景，`tsB`，seed 0）**：清单：全部完成无 crash / 重试；门 B 在转弯场景有触发、在非转弯场景触发率低（<10% 步）；每步 selector 往返 < 100 ms；被选 plan 与原 plan 的曲率变化 |dk| < 0.1 /m；
  没有出现 `ts0` 里没有的 spin / stuck / 起步停滞；日志每步有 `sel` 记录。触发条件是「有任何违反就停，报告，不进全量」。
- **全量**：all64 × 2 seed 的 `tsB`，一个 seed 一个 run，每个 run 由 bench 切成 worker job，池安排。

## 读数

主读数 **R1：turn23 上 HD（HUGSIM 的 hdscore，每个场景两个 seed 的均值），tsB 对存档 SH30 的逐场景配对差，95% bootstrap CI（`jevdrive.bench report`，同第 170 条的做法）**。
其余：R2 all64 与直行 41 条的同一差；R3 失败类别计数（fg / bg 碰撞、off_route、stuck、spin、起步停滞，两 seed 均值）；
R4 gate 触发率（门内步 / 全部步，分转弯 / 直行路线）和 pick 切换率：连续两步都在门内的步对里 pick 不同的比例，以及全部相邻步对里 pick 不同的比例，门内 pick ≠ 恒等的比例；
R5 转向与横向 jerk：逐步 `steer` 的 |Δ|（均值、p95）、由位置和航向差分得到的横向加速度 `v·ω` 的 jerk（rms，v > 1 m/s 的步，turn23 与 all64 分开），tsB 对 SH30 的配对差，仅描述；R6 `ts0` 的「如果应用会选什么」统计（SH30 自己轨迹上的门触发率与 pick 率）。

## 判据（跑之前写定，不事后改）

- **helps**：R1 配对 CI 下界 > 0，且不满足 harms。
- **harmless**：R1 的 CI 含 0，且失败 / 不稳定计数（stuck + spin + 起步停滞 + off_route，两 seed 均值）没有上升 ≥ 2（SH30 当前 2.0：0 + 0 + 1 + 1）；直行路线 HD 的差点估计不低于 −0.03。
- **harms**：R1 CI 上界 < 0；或不稳定计数的两 seed 均值上升 ≥ 2；或直行路线 HD 差的点估计 < −0.03 且 CI 上界 < 0。
- 其余组合（CI 含 0 但计数上升 ≥ 2 已归 harms；直行路线点估计 < −0.03 但 CI 含 0）报告为「不确定，偏负」，不算 harmless。
- 一次 run 每个场景每 seed 只有一次（HUGSIM 的重跑噪声很小，见上，但不是零）；两 seed 是两个不同模型，不是同模型重复；这在结果里写明。第 170 条 turn23 的 CI 半宽约 0.025，所以 R1 能分辨的效应约 ±0.03。

## 成本与停止线

估计（用第 170 条存档的 SH30 run 的墙钟）：一个 SH30 all64 run 约 64 个场景 × 约 190 s / (3 卡 × 6 槽) ≈ 12 分钟加起落；两个 seed 的 tsB 全量约 0.5 卡小时墙钟不到 1 小时；预算 5 h 墙钟 / 8 卡小时。
G-id + G-pilot 实测后按 docs/long-runs.md 估全量，若总计划超出预算 1.5 倍，停下报告实测数。conditional 项：G-id 不过时的 `ts0` 全量（+2 个 run）。

## 范围外

WOD、重训 selector、带时间一致性的 selector、CARLA。写下它们需要什么：时间一致性 selector 需要上一步 pick 作输入或迟滞，在 navtrain 序列上训练；HUGSIM 域内 selector 需要闭环帧上的标签；CARLA 需要同一接线在 B2D 的 policy server 上。本线都不开始。

## 补记（G-id 跑完后、任何 tsB 分数之前）

G-id 第一次评估：(i) 结局相同 11 / 11，(ii) |ΔHD| < 0.05 的 11 / 11，(iii) 平均 |ΔHD| 0.004，最大 0.027：过。(iv) 失败：前 5 步 `model_pos` 最大差 0.187 m。拆开看是我登记时没想到的：`model_pos` 取 plan 的第 4、8、12、16、20、24、32 点，
差随距离增长（各点场景最大值 0.002 / 0.005 / 0.014 / 0.039 / 0.047 / 0.094 / 0.187 m），第 24、32 点在 5.6 s 以后、几十到一百多米外，fp16 的分辨率本身就是 0.03–0.125 m（`tsn_extract.py` 的 NPT 注释：远点一个 ulp 是 0.125 m）。
selector 和 HUGSIM 的 plan 只用到 4 s 以内（点 ≤ 20）。**修订 (iv)：只比前 5 个点（≤ 3.9 s），阈值 0.05 m 不变**；此修订在看到 ts0 与存档的身份数据之后做，没有读任何 tsB 闭环分数；修订前后的数字都记在结果里。

G-pilot 第一次评估（tsB seed 0，11 个场景）：除「每步 selector 往返 < 100 ms」外全过（完成 11 / 11、门触发转弯 10.1% 步 / 直行 0.5% 步、|dk| 最大 0.053、每步有记录、不稳定计数 tsB 0 = ts0 0）。往返 p95 为 264 ms：
这个往返包含排队（6 个场景槽共用一个 bias server），仿真时钟不受影响，该判据的目的只是成本；tsB 与 ts0 的场景墙钟中位数 168 s 对 167 s。**修订该项：改为 tsB / ts0 墙钟中位数之比 ≤ 1.3**（实测 1.0）。此修订在看到 pilot 的 HD 数之后做（json 里同时带了 HD，读到了）；全量的判据、读数和三条线不变，也没有因此改任何门限。
