# P5 v0：CARLA 反事实配对与开环考试

状态: draft（预登记 2026-09-24，写于生成任何 pair 之前；结果栏空着）
主题: ../research/prediag-2026-09/README.md（P5 行）；决策背景 ../research/decisions.md 第 20、21、22、25、31 条

## 目标

P4 说 Waymo 训的 head 在 CARLA 帧上读不出东西（去均值后 domain AUC 仍 1.000），所以 P5 改成**在 CARLA 里训、在 CARLA 里考**。
要回答两件事：

1. **考卷本身成不成立**：仿真器渲染、ego state 逐帧相同、只差一处可见因素的配对（pair，下文 x⁺ 有因素、x⁻ 没有），
   由特权 expert 在两个世界里各开一遍，能不能给出「正确动作确实不同」的标签，且有足够多的帧真的需要反应。
   这是第 21 条和 survey §7 里「没人做过」的那一格：物理渲染 + ego 逐帧对齐 + expert 两侧重跑出标签 + per-pair 定向翻转率。
2. **谁过得了这张考卷**：一个公开的 CARLA 强 planner（TFv6，Bench2Drive DS 约 95，第 31 条），
   我们冻结 Qwen3-VL-4B 视频特征上的 CARLA 训 thin head，以及同一特征上的 linear probe（冻结 backbone、只训一个线性分类器，
   测信息在不在特征里）。probe 与 head 的对照决定失败在 readout 还是 representation（第 20 条的两个分支）。

VLM 零样本 meta-action 那一列**不做**：它在 Waymo 上 4B 和 32B 都没过预登记判据（只在复述 route command，
[todos/2026-09-23-p5-vlm-metaaction-proto.md](2026-09-23-p5-vlm-metaaction-proto.md)），第 25 条的第二个标签来源已关，
写结果时记一句「已关闭」。

## Setup

### 配对怎么造

- **路线**：Bench2Drive `bench2drive220.xml` 里 scenario 类型是「可见、需要反应」的那些，每类恰好 5 条，全部不在 11 条已知段错误路线里：

| family | 路线（town） | x⁺ | x⁻ |
|:--|:--|:--|:--|
| PedestrianCrossing | 14194 (T12)、25863 (T07)、27515 (T03)、27529 (T04)、27582 (T11) | XML 原样 | scenario 照常运行，hazard actor 被藏到地下（见下面的修订） |
| DynamicObjectCrossing | 17752 (T12)、24211 (T01)、24224 (T02)、24252 (T11)、24333 (T15) | 同上 | 同上 |
| VehicleTurningRoutePedestrian | 2164、10857、11381 (T12)、3731、3737 (T13) | 同上 | 同上 |
| ParkingCrossingPedestrian | 3248、3255 (T13)、18252 (T12)、24206 (T03)、24294 (T02) | 同上 | 同上 |
| OppositeVehicleRunningRedLight | 2082、2844、2847 (T12)、26944 (T04)、26950 (T03) | 同上 | 同上 |
| HardBreakRoute | 3540 (T13)、24330 (T10HD)、24781 (T01)、26406 (T04)、26456 (T03) | 同上 | 删掉 scenario（因素就是背景车的急刹本身） |
| StaticCutIn | 2709、2715 (T12)、25358 (T06)、26396 (T05)、26405 (T15) | 同上 | 同上 |
| ParkingCutIn | 1711、18305、18311、18356 (T12)、24759 (T05) | 同上 | 同上 |
| HighwayCutIn | 2286、3072、3074、3080 (T12)、3813 (T13) | 同上 | 同上 |
| **Light**（红灯 vs 绿灯） | Red：2373、3144 (T12)、3865、3869、3876 (T13)；Green：14842、14862、14909 (T12)、25968、25975 (T07) | scenario 类型为 `VanillaSignalizedTurnEncounterRedLight` | 同一元素改成 `...GreenLight`（Green 路线反过来构造 x⁺） |

- **x⁻ 的修订（2026-09-24 00:40，第一对 smoke 之后、批量之前；判据、指标、读法一字未改）**：原设计是「XML 里删掉 scenario 元素」。
  smoke 的第一对（27515，PedestrianCrossing，seed 0）显示这不是「只差一处」：Bench2Drive 的 scenario 触发时还会给背景车流发指令
  （`HandleJunctionScenario` 清空路口、`LeaveSpaceInFront`、`LeaveCrossingSpace`、`RemoveRoadLane` 等，9 个 hazard family 里 8 个都有），
  删掉 scenario 就连这些指令一起删了。实测：x⁺ 里路口被清空、ego 开到行人前才停；x⁻ 里路口没清空，ego 在 t = 4.15 s 就停在排队车后面，
  两个世界的 ego 在行人可见之前就分叉了，而且分叉原因是背景车，不是行人。改为：**x⁻ 保留 scenario 原样运行**（同样的 actor 从同一随机流生成、
  同样的触发、同样的背景指令），只把它的 hazard actor（横穿的行人 / 自行车、cut-in 的那辆车、闯红灯的那辆车；停着的遮挡车、集装箱等道具保留，
  它们是情境不是因素）在每个 scenario tick 之后放到地下 500 m（`b2d_hooks.track_hazards`，这正是 scenario 自己在触发前藏 actor 的位置；
  BehaviorAgent 和 Traffic Manager 的距离判断都是三维的，所以看不见也碰不到）。HardBreakRoute 的因素就是背景车急刹本身，仍用删 scenario；
  Light 仍是红绿互换。被藏的 actor 逐 run 记在 `hidden.json`。
- **变体**：每类只有 5 条路线（Light 10 条），不够 10 个，所以每条路线跑 3 个 `--tm-seed`（0、1、2，Traffic Manager 的随机种子，
  决定背景车流的行为）。x⁺ 与 x⁻ 用同一个 seed。于是 9 个 hazard family 各 15 对、Light 30 对，**共 165 对**。
- **null pair**（只改外观、正确动作不变的对照）：每条路线的 x⁺（seed 0）再跑一个只换天气的版本：XML 里的 `<weathers>` 换成另一个预设
  （原路线是白天：换成晴夜，sun altitude −60°、云 30、雾 2、无雨；原路线是夜晚：换成晴天正午，sun altitude 60°、云 5、雾 2、无雨、地面干），
  scenario 与 seed 不变。CARLA 0.9.15 的天气不进动力学（没有天气到摩擦的耦合），所以 expert 的轨迹应当逐 tick 相同，这同时是一个确定性检查。
  **55 个 null pair**。
- **两个世界之间唯一的差别就是 XML**（town、天气、路线点、seed 全同）。一处我们自己加的 hook（`b2d_hooks.py`，monkeypatch，不改
  Bench2Drive 源码）：`RouteScenario.__init__` 建完 scenario 之后，把 `CarlaDataProvider._rng` **原地**重置为 `2000 + tm_seed`。
  理由：背景车的蓝图、颜色、出生点都从这个共享的随机数流里抽，而 scenario actor 在路线开始时就先于背景车生成
  （`INIT_THRESHOLD = 500 m`，路线都比这短），删掉一个 scenario 会让背景车流整体换一套车型，两个世界从第一帧就不一样。
  重置之后背景从同一个随机状态出发。两个世界都装这个 hook，null pair 也装。
  已知的残留差别：背景车会对 x⁺ 里真实存在的 hazard actor 作反应（比如给行人让路），所以触发后背景车流仍可能两侧不同；
  逐帧报视野里有多少非因素 actor 两侧位置不同，作为纯度诊断。
- **Expert**：CARLA 0.9.15 的 `BehaviorAgent(normal)`，与 P4 完全相同（特权：读地图、所有 actor 与红绿灯，不看相机）。
  LEAD 的 PDM-Lite expert 需要作者 fork 的 leaderboard / scenario_runner 和 HD map `.h5`（box 上都没有），按任务书给 2 小时尝试，
  跑不起来就用 BehaviorAgent 并写明。**结果里标明实际用的是哪一个。**
- **记录**（`scripts/p5_pair_agent.py`，复用 `p4_carla_agent.py` 的相机模型）：每 tick（20 Hz）ego 真值位姿 / 速度 / 加速度与 expert control；
  每 tick 从一次 `world.get_snapshot()` 取 100 m 内所有车辆和行人的位姿、速度（id、type、role_name）；每 0.2 s 存三台
  Waymo 标定相机（与 P4 逐项相同：972 × 1079、f = 1113.5 px、Waymo 主点与径向畸变、JPEG q95），同时记 100 m 内红绿灯的状态，
  以及前视相机视锥内 60 m 内每个 actor / 红绿灯的可见性（从相机位置向目标 bbox 中心和上半部各打一条 `world.cast_ray`，
  第一个命中点落在目标 bbox 内即算没被遮挡）。另挂 TFv6 自己的传感器（3 相机、LiDAR、4 个 radar、IMU、GNSS、speedometer）做 shadow 推理，见下。
  录制上限 50 s 仿真时间，停住 30 s 提前结束，scenario 触发后 20 s 结束（smoke 里 27515 在红灯前停了 20 s 以上，
  P4 的 20 s 卡住门槛会在 scenario 发生之前就把路线截掉；触发后 20 s 足够覆盖观测窗口加 5 s 未来）。
- **确定性检查**（每对）：t_div 是 x⁺ 与 x⁻ 的 ego 后轴位置差 ≥ 1 cm 或航向差 ≥ 0.1° 的第一个 tick。
  **因素第一次可见** t_vis：「因素元素」在 Waymo 前视相机视锥内、60 m 内、ray test 未遮挡的第一个相机帧。
  因素元素 = x⁺ 里被 x⁻ 藏起来的那些 hazard actor（HardBreakRoute：x⁺ 里与 x⁻ 同一 tick 位置差 > 0.1 m 的背景车），加上两侧状态不同的红绿灯。
  其余两侧位置不同的 actor 不算因素，记作纯度诊断。
  smoke 里同一 XML 两次运行的 ego 在起步后有毫米级漂移（27515：2 s 时 4 mm、3 s 时 1 cm），所以 1 cm / 0.1° 的门槛是会被物理噪声碰到的，照登记用，
  null pair（同一 XML 只换天气）给出这个噪声本身的 t_div 分布。
  报 **t_div ≥ t_vis 的对的比例**；t_div < t_vis 的对丢掉并记原因：`background_drift`（scenario 触发之前 ego 就分叉）、
  `expert_reacted_before_visible`（触发之后、可见之前 expert 已经在反应，特权 expert 看得见相机看不见的东西）、`never_visible`。
- **观测帧**：每对里 t_vis ≤ t < t_div 的所有 5 Hz 相机帧，且两个世界都有完整的 4 帧 clip（0.6 s）和完整的 5 s 未来。
  这些帧上 ego 过去与现在逐 tick 相同、因素可见。null pair 的观测帧取同一路线 seed 0 那一对的观测帧时刻（再截到 x⁺ 与天气版的 t_div 之前）。

### 标签（expert 给）

每个观测帧、两个世界各有 expert 录下来的 5 s 未来（后轴坐标系，与 P4 同一套约定）。
- **v2**：未来 1.75 s 到 2.0 s 两点的位移 / 0.25 s（2 s 处的速度，m/s）。**Δ_expert = v2(x⁺) − v2(x⁻)**，负号 = 有因素时更慢。
- **stop**：未来 3 s 内速度 < 0.5 m/s 的指示量；Δstop = stop(x⁺) − stop(x⁻)，作第二列报。
- **reactive**：|Δ_expert| > τ_exp，τ_exp = null pair 观测帧上 |Δ_expert| 的 95 分位数。**预先写死一个下限 0.5 m/s**：
  仿真器是确定性的，天气不影响动力学，null 的 Δ_expert 可能全为 0，95 分位数退化成 0，那样任何数值噪声都会被算成「反应」。
  所以 τ_exp = max(null p95, 0.5 m/s)。不满足的观测帧为 non-reactive（因素已可见，expert 还没反应）。
- **label-validity 表**（本身就是交付物）：每个 family 的 pair 数、通过确定性检查的 pair 数、观测帧数、reactive / non-reactive 帧数、
  Δ_expert 分布、Δstop ≠ 0 的帧数。

### 考生（全部开环，在两个世界录下来的同一批帧上）

(a) **TFv6 shadow**：LEAD `cvpr2026` 的 `tfv6_resnet34`，三 seed ensemble（seed 1、2 正在下载；下不来就只用 seed 0 并写明），
作者的 `SensorAgent` 原样实例化在我们的 recorder agent 里，每 tick 喂它自己那套传感器的数据、跑它自己的预处理（GPS 滤波、LiDAR 累积、JPEG），
**只读输出、不用它的 control**（车由 expert 开）。所以它看到的就是闭环里会看到的输入，只是 ego 由 expert 驾驶。
读两个量：**目标速度** v_ts = Σ softmax(`pred_target_speed_distribution`)·[0, 4, 8, 10, 13.9, 16, 17.8, 20] m/s（主读数，
第 31 条说 TFv6 的分数来自 route + target speed 这条路径），以及 **waypoint 隐含的 2 s 速度** = 第 8、7 个 waypoint 的距离 / 0.25 s（副读数）。
Δ_model = v(x⁺) − v(x⁻)。环境 `envs/scout-tfv6`（Python 3.10、torch 2.8 cu128），整个 recorder 跑在这个环境里（已验证 carla、leaderboard、
BehaviorAgent、b2d_hooks、lead 全能 import）。时间盒 3 小时：跑不通就退回「TFv6 在两个世界各自闭环开、在它自己分叉之前的首批可见帧上比 control / waypoint」，并写明。

(b) **CARLA 训的 thin head**：冻结 Qwen3-VL-4B 原生视频特征（P3(d″) 的抽取器 `waymo_qwenvid.make_fx(compile=False)`，三相机、每相机 4 帧、
间隔 0.2 s，`L18_last` 主 tap、`L18_mean` 副 tap），与 P4 同一个 rig、同一套 ego 约定。
- `ridge ego`：ego 历史（16 步位置 / 速度 / 加速度）+ intent → 20 点未来。pair 观测帧上两个世界 ego 相同，所以 **Δ ≡ 0，是地板**。
- `ridge_late`：在 `ridge ego` 的残差上对特征做 ridge（P3(d″) 的 late fusion）。
- 训练帧：**不在任何 pair 观测帧里的 CARLA expert 帧**——P4 的 151 条路线全部 clip 完整的 5 Hz 帧（停车排队层下采样到 1/4），
  加上 P5 全部 run（x⁺、x⁻、null）里观测帧以外的帧（下采样到 2.5 Hz）。**按 base 路线分组做 5 折**：一个 pair 帧只由没见过它那条路线
  （所有 seed、所有世界、P4 里同 id 的路线）的模型来预测。λ 在训练折内按路线分组的内层 CV 选（与 P3(d″) 同一方法）。
- Δ_model = v2(x⁺ 的预测) − v2(x⁻ 的预测)，v2 的定义与 expert 相同。

(c) **linear probe**（同一特征，L2 logistic regression，训练折内标准化）：
- hazard probe：标签 = 当前帧有 scenario actor（行人、自行车、车辆）在前视视锥 60 m 内、未遮挡、且在接近（距离变化率 < −0.5 m/s）。
- light probe：标签 = 本车道前方 60 m 内管辖本车道的红绿灯为红（1）或绿（0），黄灯和看不见的帧不用。管辖关系用灯的 stop waypoint 与路线点匹配。
- 在非观测帧上训（同样按路线分 5 折），在 pair 观测帧上报 **x⁺ 帧对 x⁻ 帧的 AUC**（hazard family 用 hazard probe，Light 用 light probe）。
  它回答「因素在不在表征里」，与 (b) 的「head 用没用上」分开。

(d) VLM：已关闭，不跑。

### 指标（跑之前写死）

- **定向翻转率**（directional flip rate）：reactive 观测帧里，sign(Δ_model) = sign(Δ_expert) 且 |Δ_model| ≥ τ_model 的比例。
  τ_model = 该考生在 null pair 观测帧上 |Δ_model| 的 95 分位数（每个考生有自己的噪声地板）。
- **false-flip**：(i) null pair 上 |Δ_model| ≥ τ_model 的比例（样本内按构造约 5%，报出来只作检查）；
  (ii) **样本外 null false-flip**：null 路线随机两半，τ_model 在一半上定、在另一半上量，两个方向平均——这是判据用的那个；
  (iii) non-reactive pair 帧上 |Δ_model| ≥ τ_model 的比例（注意：因素可见而 expert 还没反应时，模型提前反应不一定是错，只作描述）。
- **probe AUC**（上面 (c)）。
- **expert Δ 分布**：pair 观测帧对 null 观测帧，按 family。
- 全部指标按 family 和合并各报一次，**95% CI 按 base 路线 bootstrap**（重抽路线，seed、世界、帧跟着一起走，2000 次），每格带 n（路线 / pair / 帧）。
  合并数按帧加权，另报一个 family 等权的版本。
- 进入合并数的 family 至少要有 5 个 pair 含 reactive 帧；不够的 family 单独报、不进合并。

## 成功标准 / 读法（写于任何数字之前）

先看考卷本身：

| 条件 | 读法 |
|:--|:--|
| 通过确定性检查的 pair < 50%（t_div ≥ t_vis 的比例） | 生成器本身不成立，先修生成器，不读考试 |
| 合并 reactive 帧 < 100，或 reactive 占观测帧 < 20% | expert 标签太稀，考卷没有题；报 label-validity 表后停 |

再看考生（τ 与 flip 率都是上面定义的；「CI」指路线 bootstrap 95% CI）：

| 结果 | 读法 |
|:--|:--|
| TFv6 合并定向翻转率 ≥ 50%，且样本外 null false-flip ≤ 10% | 考卷能区分，一个公开 DS-95 planner 过得了；我们 head 相对 TFv6 的差距就是要报的数 |
| TFv6 合并定向翻转率 ≤ 20% | 考卷对 DS-95 planner 也难，这是论文的动机图 |
| TFv6 在 20–50% 之间 | 按 family 读：哪些 family 过、哪些不过，不下总结论 |
| `ridge_late`：probe AUC ≥ 0.80 且翻转率 ≤ 20% | **readout**：因素在表征里，head 没用上（第 20 条分支 1），下一步动 head / 目标（第 25 条的配对监督） |
| `ridge_late`：probe AUC ≤ 0.65 且翻转率 ≤ 20% | **representation**：因素不在特征里（第 20 条分支 2），下一步动表征 |
| `ridge_late`：probe AUC ≥ 0.80 且翻转率 ≥ 50% | **fine**：CARLA 内训的薄 head 已经会反应，配对监督的必要性要重新论证 |
| probe 或翻转率落在两档之间 | 如实报，不归入分支 |
| TFv6 与 `ridge_late` 的合并翻转率都 ≤ 10% 且没有任何 family 的 CI 下界 > 10% | **测量没有地板**：两个考生都在 null 噪声里，停下，先解决测量问题（第 21 条的推翻条件） |

`ridge ego` 的翻转率按构造是 0，只作地板列。

## 估计（跑之前）

| 项 | 量 | 估计 | 依据 |
|:--|:--|:--|:--|
| 仿真 run 数 | 165 对 × 2 + 55 null | 385 次路线 | 上表 |
| 每次 run | 45 s 仿真上限，约 600–900 tick；P4 每 tick 约 150–250 ms（3 台 Waymo 相机 1088 × 1560 每 tick 渲染），加 TFv6 传感器和三 seed ensemble（在 3090 上 agent 约 100 ms/tick） | 约 4–5 min / run（含 server 已起的 route setup） | P4 generation summary（80–220 s / 路线），W2 implementation.md |
| 仿真总 wall | 06:00 前 4 个 server（VRAM ≤ 40 GB：CARLA 约 7–9 GB + TFv6 约 2 GB 每 worker），之后 8 个 | 4 worker：约 385 × 4.5 / 4 ≈ 7 h；06:00 后改 8 worker 约 4–5 h 总计 | CPU 也是约束：qv-train 占约 9 核，box 25 核 |
| Qwen 特征 | P4 追加帧约 7k（3000 帧已有特征直接复用）+ P5 观测帧约 8k（165 对 × 2 × ~25 帧）+ null 约 2.5k + P5 非观测训练帧约 8k ≈ 25k clip | GPU 共享时 532 ms/clip（P4 实测），独占时预计约 250 ms → 2–3.5 h | P4 `carla_p4/meta.json` |
| head、probe、考试 | ridge / logistic，CPU | 分钟级 | P3、P4 |
| 磁盘 | P4 152 路线 12 GB | 约 30 GB | 同比例 |

> 3 小时以上，所以批量之前先做 profiling：**2 对端到端**（生成 → 确定性检查 → TFv6 shadow → 观测帧），量每 tick 的分项耗时（tick、TFv6、相机保存、ray test），
> 找瓶颈再定 worker 数，数字补到下面「2 对 smoke」一节。

资源：server index 80 起（RPC 2000 + 50i = 6000 起，TM 8000 + 50i），与 ledger 里 70–73 不重叠；qv-train 在跑时 ≤ 4 个 server、≤ 40 GB VRAM，
不碰它的窗口。每个 job 在 `$DATA_DIR/runs/RESOURCE_LEDGER.md` 登记。

## 步骤

- [ ] 本 todo 提交（写于任何 pair 之前）
- [ ] XML 构造器（x⁺ / x⁻ / null，每个变体一个 route id，seed 编在 id 里）+ `_rng` 重置 hook + recorder agent（Waymo 相机、actor / 灯快照、可见性、TFv6 shadow）
- [ ] 2 对 smoke：端到端 + 确定性检查 + TFv6 shadow 输出 + 每 tick 分项耗时 → 定 worker 数（报告）
- [ ] PDM-Lite 的 2 小时尝试（与批量并行，结果写在这里）
- [ ] 批量生成 385 次 run（tmux `jev:p5-gen`）→ 确定性表、label-validity 表（报告）
- [ ] 建索引（观测帧、null 帧、训练帧），抽特征（先做 16 行 Waymo 等价检查），`processed/carla_p5/`
- [ ] head、probe、考试，路线 bootstrap（报告）
- [ ] 决策第 32 条、prediag README 的 P5 一节、两张图（每 family 每考生的翻转率；expert Δ 对 null 的分布），`research/results/p5-carla-pairs/`

## 2 对 smoke（2026-09-24 00:00–00:55，box 上 `runs/p5_pairs/smoke*`）

跑的是 27515（PedestrianCrossing，Town03，seed 0）的 x⁺ / x⁻ / null 和 25968（Light，Town07，seed 0）的 x⁺ / x⁻。
一路改出来的东西，按发现顺序：

| 问题 | 症状 | 改法 |
|:--|:--|:--|
| TFv6 环境是 Python 3.10 | Bench2Drive 的 route parser 用了 3.9 删掉的 `Element.getchildren()`，evaluator 又要 `items()` 返回 list | `b2d_route.py` 在 3.9+ 用纯 Python 的 ElementTree 并补回这两个方法 |
| x⁻ 不能没有 scenario 元素 | evaluator 用第一个 scenario 的名字给路线命名，删光就 IndexError | HardBreakRoute 的 x⁻ 保留元素、trigger 挪到 10 km 外 |
| **删 scenario 不是「只差一处」** | 见上面「x⁻ 的修订」：x⁻ 的 ego 在 4.15 s 停在排队车后面，原因是 scenario 的背景指令没了 | x⁻ 改为 scenario 照常运行、hazard actor 藏到地下 |
| 射线遮挡判断不可用 | `cast_ray` 在相机前 1 m 处命中无标签几何（`NONE@1.0`），几十米外的车全判成被遮挡 | 改为 instance segmentation 相机（Waymo 前视同位姿、同视场、半分辨率）：把 actor 的 3D 框投影进去，数框内同类语义、同一 instance 的像素。CARLA 的 instance id 不等于 actor id，所以必须投影 |
| TFv6 每 tick 0.47 s | 作者的 `BaseAgent.tick` 里 RANSAC 去地面用 numba `prange`，默认线程数按宿主机的 208 核开，在 25 核的 cgroup 里严重超订 | `NUMBA_NUM_THREADS=3`：`BaseAgent.tick` 从 260 ms 降到 90 ms；只改线程数，不改作者代码 |
| TFv6 只在 5 Hz 相机帧上读 | 作者完整的 `run_step`（预处理 + 三 seed forward）每 tick 都跑太贵 | 相机 tick 跑完整 `run_step`，其余 tick 只跑作者的 `BaseAgent.tick`（GPS 滤波、位姿历史、LiDAR / radar 队列），模型看到的输入与闭环完全一样；跳过的只喂 control（PID、停车标志、creeping），本来就丢弃 |
| 录制停得太早 | 27515 在红灯前停了 20 s 以上，P4 的 20 s 卡住门槛在 scenario 还没演完就截断 | 触发后 20 s 结束，卡住门槛 30 s，上限 50 s |

**确定性**（同一路线同一 seed）：x⁺ 对只换天气的 null，ego 在 26 s 里逐 tick 位置差 0.000 m；x⁺ 对藏了行人的 x⁻，同样 0.000 m（这一对 expert 在行人出现前就停在红灯前，所以全程没分叉）。
只有原来那版「删 scenario」的 x⁻ 从起步后就有毫米级漂移（背景车换了一套），这正是改设计的原因。TFv6 在两个世界输入逐位相同的帧上输出一致（目标速度到 4 位小数），
在只换天气的 null 上目标速度差 |Δ| 均值 0.04 m/s（一帧 4.5 m/s 的离群），在 x⁺ 对 x⁻（行人在 / 不在）上均值 0.16、p95 0.40、最大 7.0 m/s——
**TFv6 对行人有反应，而 expert 此时停在红灯前没有反应**，这一对的观测帧全部是 non-reactive。

**可见性**：27515 的行人 115 在 t = 6.45 s 第一次进入前视画面（3648 像素，半分辨率），111 在 8.25 s；前方灯头 50–450 像素。

**25968（Light）**：红灯版在触发时把一盏灯从绿切到红，只持续 3.6 s，而且不是 ego 要服从的那盏；两个世界的 ego 全程逐 tick 相同，TFv6 输出也相同。
Light family 在 BehaviorAgent 下可能大部分是 non-reactive，等批量的 label-validity 表说话。

**耗时**（3 seed ensemble，GPU 与 qv-train 共享，1–2 个 server）：

| 分项 | 每 tick ms（均值） | 说明 |
|:--|--:|:--|
| TFv6 相机 tick（`run_step` 全量） | 360–380 | 其中 forward 170–220 |
| TFv6 其余 tick（`BaseAgent.tick`） | 85–100 | `NUMBA_NUM_THREADS=3` 之后；之前 250–260 |
| JPEG 三相机 remap + 编码（相机 tick） | 65 | P4 同一份代码 |
| 可见性（相机 tick） | 3–6 | 投影 + 分割图 |
| expert、actor 快照 | 1–2、0.3 | |
| 整个 agent tick | 220–300 | |

一次 run 191–284 s（473–993 tick），每 worker 显存约 10 GB（CARLA 7.2–7.9 GB + TFv6 2.2 GB）。
批量 385 次 run：06:00 前 3 个 worker（≤ 40 GB），约 3.5 min / run → 约 7.5 h；qv-train 结束后加第二个 runner（同一 `--out`，靠 claim 分路线），预计总 6–7 h。
PDM-Lite：LEAD 的 expert 依赖作者 fork 的 `CarlaDataProvider.active_scenarios` 等字段（我们锁定的 Bench2Drive 里没有）和 `autonomous_agent_local`，
换 fork 就等于换了 scenario 实现，与「不改 Bench2Drive」冲突，**不用，expert 是 BehaviorAgent**。

## 结果

（跑完再填。run dir 在 box 的 `$DATA_DIR/runs/p5_pairs/`。）
