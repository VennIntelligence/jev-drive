# 横向 v2：后轴速度系 pursuit + Ackermann 反解的预注册闭环检验

状态: frozen（本文件随首例之前的 commit 冻结；之后只在文末"执行记录"追加，不改正文）
主题: [lateral-physics.md](lateral-physics.md)（物理推导与离线筛选），[docs/b2d-controller.md](../../docs/b2d-controller.md)（控制器栈）
上游冻结历史: [../2026-09-23-lateral-followup/pose-followup/protocol.md](../2026-09-23-lateral-followup/pose-followup/protocol.md)（pose-g2 六例协议，本协议沿用其保护量）

## 目标

[lateral-physics.md](lateral-physics.md) 第 4 节的离线闭环筛选（fitted plant 上跑生产 Controller 与 RouteAdapter）把两处模型错配列为 26966 急右弯外偏的主因，合并修正后筛选中的 window CTE RMS（cross-track error，后轴真值到参考线的横向距离）从 .497 降到 .089 m。本实验在 CARLA 闭环里回答三个问题：

1. 候选（两处修正同时打开）在 fixed-k 位姿下能否把 26966 window CTE RMS 的种子中位数压到 .25 m 以下，且配对改善的置信区间不含 0；
2. 它在 24240 左弯、17563 的两个 S，以及 8 个按几何预先选出、从未看过控制结果的 held-out 弯上是否守住既有保护量；
3. 把真值后轴位姿直接喂给控制链时（truth-pose ceiling，仅诊断），生产控制器和候选各自能到哪里，即控制器本身的上限。

## 候选：唯一变化

两项都是 `scripts/b2d_controller.py` 的 opt-in，默认关闭；关闭时与冻结控制器逐位相同（见"实现与测试"）。

| 开关 | 公式 | 常数与来源 |
|---|---|---|
| `pursuit_frame="rear_slip"` | β̂ = atan((v + 1 m/s)·ω / (c·g))；aim 点先乘 R(β̂) 转进后轴速度系，再算 κ = 2y/d² | c = 11.0 /rad：开环 plant 回放在 8 个窗（4 窗 × 2 臂）上拟合的 PhysX 线性区侧偏刚度；g = 9.81；v、ω 取 SPEED 与 IMU gyro（左正），不读真值 |
| `steer_inverse="ackermann"` | 名义角 δ = atan2(L\|κ\|, 1 − w\|κ\|/2)·sign κ，即 cot δ_inner = 1/(Lκ) − w/(2L) | L = 2.8605 m；w = 1.5929 m：`calibration-physics.json` 两前轮中心间距 159.29 cm；PhysX Ackermann accuracy = 1 把名义角作用在内轮 |

β̂ 的形式等价于 lateral-physics.md 的 k(v)·v·ω，k(v) = (1 + 1/v)/(c·g)；写成 (v + 1)·ω/(c·g) 后在 v→0 处有限，不需要额外的低速截断。

其余全部保持 pose-g2 共同基线：pursuit、linear aim、max lookahead max(3 m, .5 s·v)、PI Kp=.5/Ki=.25、near 速度窗、steer_rate 2/s、max_steer .8、实测 MKZ 几何、v2 RouteAdapter（5 Hz rejoin，即每 0.2 s 从估计位姿重建一段接回参考线的 quintic 过渡）、PoseFilter 增益 .05/.1。控制器 odometry 不加侧滑项（lateral-physics.md 的第 3 个候选，本轮不测）。

**已知的非独立性。** c = 11.0 和两处修正都是在看着这 4 个开发窗的日志时得到的，所以开发窗上的改善不是独立证据；held-out 弯是本轮唯一的样本外检验。

## 臂

| 臂 | 位姿来源 | pose 侧向传播系数 k | pursuit_frame / steer_inverse | 用途 |
|---|---|---:|---|---|
| `prod-k0` | 传感器 PoseFilter | 0 | body / nominal | 2×2 因子 |
| `prod-kfix` | 传感器 PoseFilter | .010659832（冻结，不重拟合） | body / nominal | **门槛对照** |
| `slipack-k0` | 传感器 PoseFilter | 0 | rear_slip / ackermann | 2×2 因子 |
| `slipack-kfix` | 传感器 PoseFilter | .010659832 | rear_slip / ackermann | **候选** |
| `prod-truth` | **真值后轴位姿（仅诊断）** | 不适用 | body / nominal | 生产控制器上限 |
| `slipack-truth` | **真值后轴位姿（仅诊断）** | 不适用 | rear_slip / ackermann | 候选控制器上限 |

配置文件在 [lateral_v2/inputs/configs/](lateral_v2/inputs/configs/)，由 [build_inputs.py](lateral_v2/build_inputs.py) 从冻结的 `baseline-zero.json` 生成。truth 臂只在控制器配置写了 `diagnostic_truth_pose_ceiling: true` **且** validate 带 `--allow-truth-pose-diagnostic` 时才生效，每行遥测带 `pose_source: truth_diagnostic_ceiling`；它们不参与任何门槛，也不能作为候选结论的证据。truth 臂把真值后轴位姿送进 RouteAdapter（投影与 rejoin），控制器两次更新之间仍用 SPEED/gyro 积分，所以它测的是"定位完美时控制链能做到的程度"，不是完美 odometry。

## 路线与窗口

**开发窗（沿用冻结定义，站距在 route_reference 的 dense 参考线上）：**

| 窗 | 路线 / town | 类型 | 巡航 m/s | core m | 评价窗 m | 附加 |
|---|---|---|---:|---|---|---|
| 24240 | 24240 / Town10HD | 左弯（core 49°） | 8 | 23–59 | 18–64 | |
| 26966 | 26966 / Town05 | 急右弯 | 8 | 29–46 | 24–51 | post10m 51–61 |
| 17563-S1 | 17563 / Town12 | S | 6 | 32.5–43.5 | 27.5–48.5 | |
| 17563-S2 | 17563 / Town12 | S | 6 | 78.5–90 | 73.5–95 | |

**Held-out 弯的选取规则（只用几何，在任何 v2 运行之前执行）。** [select_heldout.py](lateral_v2/select_heldout.py) 在 GPU box 上对 bench2drive220 的其余 217 条路线逐条做与评测器相同的 densify（GlobalRoutePlanner，1 m hop，逐段连接 XML waypoint），地图用 OpenDRIVE 在客户端构造，不启动 CARLA。在 0.5 m 重采样、5 m 居中弦求航向的路径上算左正曲率，取 |κ| ≥ 1/40 的同号连续段（间隔 < 3 m 合并）：

- 单弯：|转角| ≥ 60°；左正为 left，否则 right。
- S：相邻两段反号，各自 |转角| ∈ [20°, 60°)，中间直段 ≤ 12 m，峰值 |κ| ≥ 1/15（排除换道）。
- 可用条件：core 前至少 25 m、后至少 35 m 路线；core 前 25 m 内没有其他 ≥ 30° 的弯；core 与任一开发窗 core 相距 ≥ 15 m（同 town）；巡航速度取满足 v²/R_min ≤ 9.5 m/s² 的 8 或 6 m/s，都不满足则剔除。
- 取法：按 right、left、S 轮流，每类取路线号最小、且 town 尚未入选的候选；所有 town 都用过后才允许重复 town；每条路线最多一个弯；与已选弯相距 < 15 m 视为同一地点跳过。配额 right 3、left 3、S 2。
- 入选路线裁成 core 前 40 m 到 core 后 45 m 的 XML waypoint（更短的保持原样），去掉 scenarios；窗口 = core ± 5 m，post10m = 窗口末端后 10 m。站距在裁剪后路线自己的 dense 参考线上；分析时逐例核对运行时 `route_reference.json` 与离线参考逐点偏差 ≤ .05 m，否则该窗记为未覆盖。

217 条中的全部候选（121 个，含剔除原因）保存在 [inputs/candidates.json](lateral_v2/inputs/candidates.json)。大多数 B2D 路线的路口弯在 8–30 m 处开始，因"core 前不足 25 m"被剔除。结果：

| 窗 | town | 类型 | 转角 ° | R_min m | 巡航 m/s | core m | 评价窗 m | 裁后长度 m |
|---|---|---|---:|---:|---:|---|---|---:|
| 25358 | Town06 | right（缓弯） | −70 | 34.8 | 8 | 25.0–76.0 | 20.0–81.0 | 119.4 |
| 27506 | Town05 | right | −88 | 6.3 | 6 | 30.0–43.5 | 25.0–48.5 | 79.5 |
| 27515 | Town03 | right | −86 | 7.3 | 8 | 29.0–44.0 | 24.0–49.0 | 81.9 |
| 2084 | Town12 | left | +88 | 5.9 | 6 | 31.0–43.5 | 26.0–48.5 | 79.2 |
| 27494 | Town04 | left | +84 | 9.4 | 8 | 31.5–49.0 | 26.5–54.0 | 86.1 |
| 2881 | Town12 | left | +87 | 5.9 | 6 | 31.0–43.5 | 26.0–48.5 | 78.8 |
| 23695 | Town13 | S | +28 / −29 | 9.3 | 8 | 81.5–93.5 | 76.5–98.5 | 130.9 |
| 17569 | Town12 | S | +33 / −33 | 7.9 | 8 | 37.5–48.5 | 32.5–53.5 | 96.2 |

两处需要说明。17569 与开发路线 17563 同在 Town12 的 road 1157 上，是同一种重复路型的另一处 S，按规则（core 相距 ≥ 15 m）算 held-out，但它和 S1/S2 的几何相近，不能当作完全独立的路型。2084 与 2881 是 Town12 两个不同路口的同型左转。

## 噪声底与配对扰动

这台 box 上的 CARLA 几乎确定（pose-g2 复现到 1e-5 m），同一输入重复跑什么也测不到。本轮用预注册的扰动集合生成可比较的重复：

- **p00（nominal anchor）**：不加任何扰动，等同历史运行。只用于两件事：prod-k0/prod-kfix 必须复现 gpubox-port-v1 的四窗数字（parity），以及渲染。**不进入统计。**
- **p01–p10**：GNSS `noise_seed` = 100+i，IMU `noise_seed` = 200+i（此前 leaderboard 的属性白名单把 `noise_seed` 丢掉，所有运行都用 seed 0；现由 `b2d_hooks` 透传），起点沿右向量平移 U(−.25, .25) m、yaw 偏 U(−1.5°, 1.5°)，由 `random.Random(20260923)` 抽取并写死在 [perturbations.json](lateral_v2/inputs/perturbations.json)：

| id | GNSS / IMU seed | 横移 m | yaw ° |
|---|---|---:|---:|
| p01 | 101 / 201 | −.077 | −.621 |
| p02 | 102 / 202 | +.143 | −.763 |
| p03 | 103 / 203 | −.003 | −1.418 |
| p04 | 104 / 204 | +.072 | +.056 |
| p05 | 105 / 205 | +.152 | +.047 |
| p06 | 106 / 206 | −.222 | +.474 |
| p07 | 107 / 207 | +.093 | −1.315 |
| p08 | 108 / 208 | −.085 | −.824 |
| p09 | 109 / 209 | +.148 | +1.107 |
| p10 | 110 / 210 | −.080 | +.684 |

GNSS 噪声是每轴 σ ≈ .56 m 的逐帧独立采样，经 .05 增益融合后，定位误差在弯道处的实现随 seed 改变；5 Hz rejoin 每次都从估计位姿重新锚定，所以这部分直接进入 CTE。起点扰动再让车辆状态与 nominal 轨迹去相关。六个臂用完全相同的 10 组扰动，逐 seed 配对。truth 臂的控制不读 GNSS，只受起点扰动影响。

规模：11 路线 × 6 臂 × 11 扰动 = 726 case。

## 指标（逐帧，全部帧进入，不按速度或误差删帧）

与 pose-g2 冻结分析同定义，另加 course：

| 量 | 定义 |
|---|---|
| CTE | 真值后轴点对 `route_reference.json` world_xy 的最近线段投影，左正，m |
| body heading | 真值 yaw − 投影站距处 5 m 居中弦方向，CARLA 右正，° |
| **course error**（登记的跟踪航向量） | 后轴真值位移（第 i−1 到 i+1 帧）方向 − 同一弦方向，° |
| β（真值） | body − course，即后轴侧偏角 |
| β̂（传感器） | −atan((v+1)·ω_left/(11·g))，由 control.jsonl 的 speed/gyro 算，CARLA 右正 |
| body − β̂ | 扣掉传感器预测侧偏后的车身航向误差 |
| 速度 | 实际速度 − 参考速度（validator 的截止减速参考）RMS；窗口内平均实际速度 |
| 动作 | emitted steer 相邻连续帧差分 / dt 的 \|·\| P95；横向加速度 = world acceleration · right vector 的 \|·\| P95 |

每窗汇报 RMS、\|·\| P95、\|·\| 最大、均值；窗口需完整进出且 ≥ 20 帧，否则记未覆盖（证据不足，不自动通过）。

## 统计

- 每臂每窗：10 个 seed 的中位数，及 median 的 95% percentile bootstrap CI（10000 次重采样，RandomState(0)）。
- 配对差：d_i = 候选_i − 对照_i（同一扰动 i）；报告均值、均值的 95% percentile bootstrap CI（同上）、d_i > 0 的个数。
- 相对量：r_i = 候选_i / 对照_i − 1，同样取均值与 CI。

## 门槛（首例前冻结；门槛比较只用 slipack-kfix 对 prod-kfix）

**主目标（26966）**
- P1：slipack-kfix 的 window CTE RMS 10-seed 中位数 ≤ .25 m。
- P2：配对差（slipack-kfix − prod-kfix）的 window CTE RMS 均值的 95% CI 上界 < 0。

**26966 附加保护**：CTE P95 配对均值增量 ≤ 0；post10m CTE RMS 配对均值增量 ≤ 0；以及下表中除 CTE 三项外的全部保护。

**逐窗保护**（24240、17563-S1、17563-S2 和 8 个 held-out 窗各自独立判定，不池化）：

| 量 | 条件（配对均值） | 来源 |
|---|---|---|
| CTE RMS | 增量 ≤ +.03 m | pose-g2 协议 |
| CTE \|·\| P95 | 增量 ≤ +.05 m | 同上 |
| CTE \|·\| 最大 | 增量 ≤ +.10 m | 同上 |
| course error \|·\| P95 | 增量 ≤ +1.0° | 取代 body heading 的跟踪航向保护（理由见下） |
| 参考速度误差 RMS | 增量 ≤ +.10 m/s | pose-g2 协议 |
| 平均实际速度 | 下降 ≤ .20 m/s | 同上 |
| 横向加速度 \|·\| P95 | 相对增加 ≤ 10% | 同上 |
| emitted steer-rate \|·\| P95 | 相对增加 ≤ 20% | 同上 |

- 相对量的对照中位数 ≤ .01（对应单位）时记"证据不足"，不通过。
- 任一窗任一 seed 未覆盖，该窗相关保护记"seed 不全"，不通过。
- 点估计满足但 CI 上界越过门槛时标注"uncertain"，仍按点估计判定。
- **G2**：候选不得在对照通过 G2 的 (扰动, 路线) 上出现 G2 失败（G2 = 既有全程门槛：completed、no_collision、全程 CTE RMS ≤ .5 / P95 ≤ 1 m、巡航速度、pose P90 ≤ .5 m、heading-pose P90 ≤ 1°、终点、停车保持、遥测完整）。所有臂的 G2 失败全部列出。

**判定**：候选通过 ⇔ P1 ∧ P2 ∧ 26966 附加保护 ∧ 三个开发窗的全部保护 ∧ 8 个 held-out 窗的全部保护 ∧ G2。任何一项失败即不通过；不改门槛，看到结果后不加网格、不换常数、不重跑取优。基础设施失败（server 崩溃、超时）可以原样重跑并单独记录，行为失败不重跑。

**航向为什么换成 course。** 后轴精确在线上且运动方向与切线重合时，body heading error 恒等于 β_r；26966 上 β_r 的 P95 约 4.8–4.9°，已接近冻结门槛 4.27+1°（lateral-physics.md 1.3 节）。所以 body heading 门槛在结构上惩罚更好的后轴跟踪。本轮登记 course error 为跟踪航向门槛；冻结的 body heading P95 仍逐窗报告，并列 β（真值）P95、β̂ P95、body − β̂ P95 三列，同时给出"若按冻结 body 门槛（增量 ≤ 1°）会怎样判"的信息列，不参与判定。

**次要报告（不设门槛）**：slipack-k0 对 prod-k0 的同套配对统计（候选在 k=0 位姿下的效果，即 2×2 的另一行）；prod-truth 与 slipack-truth 的中位数（控制器上限），以及候选/生产相对各自 ceiling 的差距。

## 事先写下的预期（来自离线筛选，不是门槛）

筛选在 fixed-k 位姿下给出 26966 .497 → .089、24240 .079 → .119（高于 +.03 保护），S .28/.25 → .12/.11；筛选整体比 CARLA 偏悲观约 16%。据此预期 P1、P2 通过，24240 的 CTE 保护有较大可能失败。若出现"26966 通过、24240 失败、held-out 通过"，下一步应处理 fixed-k 下左弯的定位偏差（lateral-physics.md 第 2 节的抵消），而不是撤回两处修正；若 held-out 失败，则说明候选（含 c=11 的标定）过拟合了开发窗。

## 可视化（独立渲染 pass）

campaign 之后，对 26966 的 p00 分别单独跑 prod-kfix 和 slipack-kfix 各一次（各自独立的 validate 进程与 server，避免上一例的 debug 线残留），`--chase-camera` 在车上刚性挂一台离屏 RGB 相机（960×540，FOV 80°，车后 7.5 m、高 4.2 m、俯角 20°），用 debug line 画出参考线（绿）和逐帧后轴真值轨迹点（红），逐帧存 JPEG。相机和画线不进入控制输入；渲染例的 CTE 轨迹与 campaign p00 同臂逐帧对比，最大差值写进报告。box 上用 ffmpeg 合成 mp4；contact sheet 取站距 24、30、35、40、46、51 m 最近帧，两行（生产 / 候选）拼成一张 PNG，只把这张小 PNG 提交进 git。

## 实现与测试（首例前完成）

- `b2d_controller.py`：`pursuit_frame`、`rear_slip_c_per_rad`、`steer_inverse`、`track_width_m`（开关与参数必须成对出现，否则报错；只允许 pursuit preset）；`update(traj, t, trajectory_dt=None)` 支持 (N,2) 与显式间隔，horizon = N·dt 不外推，默认调用仍只接受 20×2。本轮 route oracle 仍送 20 点 / .25 s，N 点路径不被本 campaign 使用，是为接真实 TCP（4 点 / .5 s）准备的。
- `b2d_agent.py`：truth-pose ceiling（双重 opt-in、逐行标注、真值取不到即报错不回退）；GNSS/IMU `noise_seed` 透传。
- `b2d_controller_validate.py`：`--perturbations/--perturbation-ids`（扰动在路线内层循环，输出 `<out>/<id>/<route>/<arm>/pursuit`）、`--allow-truth-pose-diagnostic`、`--chase-camera`。
- 测试：controller 125 → 143（新增 18 个，含：显式关闭在 vendor replay 与 GPU box 记录的 pose-g2 六例 2758 帧闭环回放上逐位等于默认；slip 的符号、零 yaw rate 逐位等于 body、镜像对称、切线几何；Ackermann 往返 PhysX 中心角、对称、小角极限；N 点接受/拒绝/不外推/与线性重采样等价；truth ceiling 的双重 opt-in、标注与缺帧报错；扰动表校验、起点右向量、noise seed 透传）。TCP 17/17、pose 分析 15/15。均以 `OPENBLAS_CORETYPE=Barcelona` 运行。

## 执行

- 冻结：本文件、`lateral_v2/` 下的 inputs、analyze.py、build_inputs.py、select_heldout.py、run_campaign.sh 与控制器改动同一 commit；该 SHA 记在执行记录里。analyze.py 的门槛常数（GUARDS、PRIMARY_*）与本文一致，运行后不改。
- smoke：先在独立目录跑 26966 × 6 臂 × p00（6 例），只检查基础设施（truth 臂标注、扰动透传、新配置加载、遥测完整），不据此改任何参数。之后 campaign 的 p00 应与 smoke 逐帧相同，作为确定性证据。
- campaign：`run_campaign.sh <root> {4|8}`，按 box 上 `~/data/runs/RESOURCE_LEDGER.md` 的约定（lateral 流在有特征提取时 ≤ 4 个 CARLA server）；4 组路线各一个 worker（8 路时再按扰动对半切）。估计：约 36 万 tick，4 路约 80–100 min，8 路约 45–60 min。
- 输出：`$DATA_DIR/runs/b2d/controller/lateral-v2/`；分析输出与图只把小文件拉回 `lateral_v2/results/`，逐帧数据留在 box。
- 报告：结果写进本文件的执行记录和 [lateral-v2-report.md](lateral-v2-report.md)。

## 执行记录

（运行后追加）
