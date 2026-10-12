# RULE-ORACLE 范围界定：规则规划器（PDM-Closed 思路）作 AlpaSim nuPlan 18 决策本地闭环的 driver

2026-10-12 写。只读范围界定：没有训练、没有 rollout、没有 GPU 作业。盒子上做过的只有 import / 版本 / 文件存在检查，外加两次 CPU 探针
（在 AlpaSim venv 里经 scene loader 读一个 held scene 的 rig / traffic / map，并对照 gpkg 车道表），结果写在 §1。
本稿登记 §5 的判词线，写定并 push 于任何 oracle rollout 之前。「推」= 由记录的数或源码推出，「估」= 判断。

**一句话结论。** 两种形态都可行。特权 oracle（形态 1）最便宜的路线是**不装 tuplan_garage / nuplan-devkit**，在 AlpaSim 自己的 venv 里写一个
约 400–500 行的 PDM-Closed 重实现，世界状态直接用 runtime 自己的 scene loader 按 scene id 读出（地图、回放车辆、日志路径与仿真器、评测器
用的是同一份对象，同一个坐标系）。held 460 × 1 seed × 18 决策一臂约 35–40 min wall、约 0.6 card-h；登记四臂（L0、A1、A2、A2L）
约 2.4 card-h，并行跑约 45 min。零样本形态（形态 2）可以复用同一个 numpy 核心，跑在 `envs/op-train` 里的 sh30 driver 上，但它要等
§5 的 A2 / A2L 读数和并行 agent 的 head 精度结果出来再建。

## 0. 问题

一个成熟的规则规划器能否以最小工程量作为 driver 跑进本地 AlpaSim nuPlan 18 决策闭环（`experiments/alpasim_long`，136 个原始 log 重建的
cache `after160`，held 460 个 scene，计分 452 个，18 决策），形态有二：

1. **特权 oracle（上限读数，规则只允许它作探针）**：PDM-Closed 或其思路的小型重实现，喂真地图（车道中心线 / 边界、可行驶区域）和重建
   cache 里的回放车辆，按 scene id 查，不看图像。横向用规则规划器；纵向用它的 IDM，或者用我们现有 driver 的纵向。
2. **零样本非特权**：同一个规则规划器喂冻结 openpilot 模型自己的 lane lines / road edges / lead 输出，不用地图。

## 1. driver 每个决策收到什么，以及怎样按 scene id 旁路加载地图和车辆（问题 a）

**收到的东西**（`src/grpc/alpasim_grpc/v0/egodriver.proto` @ 0bb4c4b；`experiments/alpasim/lib/sh30_driver.py`；docs/alpasim.md
「What a nuPlan-track driver actually receives」）：

| RPC | 内容 | 对 oracle 的用处 |
|:--|:--|:--|
| `start_session` | `rollout_spec.vehicle`（相机标定、ego box 5.176 × 2.297 × 1.777 m 及其在 rig 系的位姿），`random_seed`，**`debug_info.scene_id`** | 本地跑能拿到 scene id：sh30 driver 已经把它写进 `drive.jsonl`（如 `2021.05.25.17.54.41_veh-35_01654_01850-1370cea130165b51`）。官方评测不给（docs/alpasim.md：「no scene id in benchmark runs」） |
| `submit_egomotion_observation` | local → rig 位姿（**local 系**）+ rig 系 `DynamicState`（速度 / 加速度；第一帧速度未旋转，要防） | ego 位姿直接在 local 系，规划无需换系 |
| `submit_route` | 20 个 rig 系路点，约 4.2 m 间隔，40–80 m 外，约一半是 NaN 填充 | 路点是 `RouteGeneratorMap` 把日志路径**投影到车道中心线**后的点（推：`route_generator.py` `_determine_waypoints`），即车道级路线，不只是左 / 右 / 直 |
| `submit_image_observation` | 8 路相机 JPEG，2 Hz | oracle 忽略 |
| `drive(time_now_us, time_query_us)` | 2 Hz，`time_now` = 17 000 + 500 000 k us，k = 0..17；query = now + 0.5 s；无 RPC 截止时间 | 返回 local 系带绝对时间戳的位姿序列，覆盖 ≥ 4 s（LTF 样例的 `build_trajectory_from_plan` 不够 4 s 就回退直线） |

左 / 右 / 直指令不是仿真器给的，是我们的 driver 用 LTF 样例的 `command_from_route`（第一个 ≥ 5 m 的路点，y > 2 m 左、< −2 m 右）自己算的。

**local 系 = 世界系（nuPlan 地图坐标，UTM 量级）只平移、不旋转，原点是 ego 日志轨迹第一个点**（`alpasim_utils/trajdata_data_source.py`
`rig` 与 `_apply_coordinate_transform_to_map`）。所以只要用 runtime 自己的 scene loader 按 scene id 取 `SceneDataSource`，地图、回放车辆、
日志路径就已经在 driver 收到的同一个 local 系里。

**旁路加载（已在盒子上探针，CPU，AlpaSim venv）**：

```python
from alpasim_utils.yaml_utils import typed_parse_config
from alpasim_runtime.config import UserSimulatorConfig
from alpasim_runtime.scene_loader import build_scene_loader
L = build_scene_loader(typed_parse_config(f"{RUN}/generated-user-config-0.yaml", UserSimulatorConfig))  # 6.2 s, 1491 scenes
ds = L.get_data_source(scene_id)
ds.rig.trajectory          # 日志 ego 轨迹（local 系，rear axle），held 例：96 点，0..9.5 s，0.1 s
ds.traffic_objects         # 回放车辆（只有 vehicle，csaps 平滑后），例：27 个；仿真器回放的就是它们（physics NONE）
ds.map                     # trajdata VectorMap（local 系），例：4583 条 lane（center / left_edge / right_edge / next_lanes /
                           #   prev_lanes / adj_lanes_left / adj_lanes_right），1383 个 ROAD_AREA 多边形；0.7 s
```

- 帧对齐已核：`ds.rig.trajectory.interpolate_pose(17000)` = (−0.075, −0.004, yaw −3.072)，与存档 `drive.jsonl` 里同一 scene k = 0 的 anchor
  (−0.07458, −0.00445, −3.07217) 一致到 1e-4 m（推）。
- runtime 对车辆的处理要照抄：`traffic_objects.clip_trajectories(egomotion_context_start_us, end_us + 1, exclude_empty=True)` 再
  `filter_short_trajectories(min_traffic_duration_us)`（本配置 `min_traffic_duration_us: 0`），见 `alpasim_runtime/unbound_rollout.py`。
- 已有可复用的代码：`experiments/anchor_offset/scripts/ao_e1.py` 的 `loader(cfg)`（同一个 loader，`OmegaConf.merge` 写法）和用
  `RouteGenerator.create(P, vector_map=ds.map, route_generator_type=MAP, route_start_offset_m=40)` 重算 AlpaSim 路线的写法（第 266 条：
  与日志 route0 99.89% ≤ 0.05 m）；`experiments/zeros_mech/scripts/zm_read.py` 的 `objects(ds)` / `obj_state(objs, t)`（回放车辆按时间插值）
  和车道方向分类 `LaneCls`（用评测器的 `_get_lane_polygon`）。
- **注意启动顺序**：`run_native.py` 先启动 driver、再跑 wizard 写 `generated-user-config-0.yaml`，所以 loader 必须在第一个 `start_session`
  时惰性建（文件那时已在），用 `RO_RUN_DIR=$out` 传 run 目录。

**地图在盒子上是什么形态**：

| 形态 | 路径 | 内容 |
|:--|:--|:--|
| trajdata 矢量图（评测器与路线生成器用的就是它） | `$DATA_DIR/datasets/alpasim_nuplan/nuplan_test/maps/{boston,las_vegas,pittsburgh,singapore}.pb` + `_kdtrees.dill` / `_rtrees.dill`；`alpasim_nuplan_long/after160/nuplan_test/maps` 是指向它的符号链接 | 车道中心线 / 左右边、拓扑、ROAD_AREA（可行驶区域多边形，nuPlan 没有 road edge 折线，所以 `ROAD_EDGE` 为 0） |
| nuPlan 原始 gpkg | `$DATA_DIR/datasets/navsim/maps/<city>/<version>/map.gpkg`（4 个城市，`nuplan-maps-v1.0`） | 全图层，含 `lanes_polygons` / `lane_connectors` 的 `speed_limit_mps`、`turn_type_fid`、`stop_polygons`、`traffic_lights` |
| 重建 cache 本身 | `after160/nuplan_test/<scene>/agent_data_dt0.10.feather` + `scene_index_dt0.10.pkl` | 只有 agent 状态，**不存车道几何**；车道几何全在上面的 `.pb` |

trajdata 的 `RoadLane` **没有限速字段**，但 lane id 就是 gpkg 的 fid（探针：`68571`、`68545` 在 `lanes_polygons`，`69256`、`70517`、`64917` 在
`lane_connectors`），所以限速可以用 `pyogrio.read_dataframe(gpkg, layer=..., read_geometry=False, fid_as_index=True)` 按 id 查（AlpaSim venv 有
geopandas / pyogrio，0.0 s 读完；Las Vegas 车道限速中位 13.4 m/s，2.2–20.1）。缺失时回退 navsim 的 `fallback_target_velocity` 15 m/s。

## 2. 安装 tuplan_garage + nuplan-devkit，还是无依赖重实现（问题 b）

**盒子上现有的环境**（探针）：

| env | Python | torch | nuplan-devkit | navsim（含 `pdm_planner`） | trajdata | grpc / protobuf |
|:--|:--|:--|:--|:--|:--|:--|
| AlpaSim venv `third_party/alpasim/.venv` | 3.12.14 | 2.9.1+cu128 | 无（overlay `envs/nuplan_devkit_overlay` 2.0.0 `--no-deps` 可 import `ego_state`，但 `map_factory` 缺 rasterio） | 无 | 有 | 1.84.0 / 4.25.9 |
| `envs/navsim2` | 3.10.21 | 2.0.1+cpu | 有（`third_party/nuplan-devkit`） | 有（`third_party/navsim`，PDM-Closed 全套，5 850 行） | 无 | 1.84.0 / 4.25.3 |
| `envs/op-train`（sh30 driver） | 3.11.16 | 2.14.0+cu130 | 无 | 无 | 无 | 有 |
| 提交镜像（`experiments/alpasim/docker`） | 3.11-slim | 2.14.0+cu126 | 无 | 无 | 无 | 1.84.0 / 7.36.2；**无 shapely** |

tuplan_garage 不在盒子上；navsim 的 `pdm_planner` 就是它的移植（Apache-2.0）。py123d_garage（`third_party/py123d_garage`）也有一份移植，但依赖
py123d 自己的 MapAPI 和转换后的地图，不适用。

**路线 A：nuplan-devkit + navsim PDM-Closed 原样跑。** 要在 `navsim2` 里起 gRPC 服务（stubs 可用），地图走 nuPlan map API 读 gpkg（世界系 =
local + 原点，需核），车辆要从 trajdata cache 导出成旁车文件（navsim2 没有 trajdata），再拼 `PlannerInput`（`EgoState`、`DetectionsTracks`、
`route_roadblock_ids`、历史缓冲）。代价：两个环境之间的数据导出、gpkg 地图加载（每城市每进程数十秒，估）、PDM 的可行驶区域用的是 nuPlan
图层而评测器用 trajdata ROAD_AREA（两者可能差几厘米到分米，估），以及 nuPlan 的 at-fault 规则（排除停车被撞、追尾）与 AlpaSim 的不同。
估 1–1.5 天，风险在胶水层。

**路线 B（推荐）：AlpaSim venv 里写约 400–500 行的重实现。** 理由：
- 零安装：numpy、scipy、shapely 2.2、csaps、grpc stubs、trajdata、`alpasim_runtime` 都在。
- 世界状态与仿真器 / 评测器**同源同系**：同一个 `SceneDataSource`，同一套回放车辆，同一份 ROAD_AREA；越野检查可以逐字照抄评测器的
  `_is_offroad_using_road_area`（ROAD_AREA 的并集必须 `covers` 整个 ego 多边形），碰撞可以照抄 AlpaSim 的 at-fault 定义（§3）。
- navsim 里纯 numpy 的部分可以近乎原样搬：`proposal/batch_idm_policy.py`（168 行，只依赖 enum）、`simulation/batch_lqr_utils.py`（238 行）；
  `batch_kinematic_bicycle.py` / `batch_lqr.py` 只用了 nuplan 的 `VehicleParameters` 和 `principal_value`，替换成常数（Pacifica：长 5.176、
  宽 2.297、后轴到中心 1.461、轴距 3.089 m，与 AlpaSim ego box 完全一致）即可。`pdm_scorer.py`（700 行，深度绑 nuplan）不搬，按 AlpaSim
  指标重写一个 80–120 行的打分器。
- 分层让形态 2 能复用、让特权代码进不了镜像：`ro_core.py` 只用 numpy（镜像里没有 shapely），可行驶区域检查作为回调由「世界」提供。

估 1 天写 + 半天离线核对（§4 S0）。

**Drive() 示意（路线 B，形态 1）**：

```python
# experiments/rule_oracle/lib/ro_driver.py — PRIVILEGED. AlpaSim venv only; never in closure.txt / any image.
import os, numpy as np
from alpasim_grpc.v0 import common_pb2, egodriver_pb2, egodriver_pb2_grpc
import ro_core as RC          # numpy only: paths, IDM proposals, LQR + kinematic bicycle, box SAT, scorer
import ro_world_priv as RW    # scene loader, map, replayed traffic, log route, gpkg speed limits, road-area union (shapely)

ARM = os.environ["RO_ARM"]    # L0 | A1 | A2 | A2L  (§4)
assert os.environ.get("RO_PRIVILEGED_OK") == "1", "privileged oracle: local probe only"

class Oracle(egodriver_pb2_grpc.EgodriverServiceServicer):
    def start_session(self, req, ctx):
        scene = req.debug_info.scene_id                        # local runs only; empty in the benchmark -> abort
        if not scene:
            ctx.abort(grpc.StatusCode.FAILED_PRECONDITION, "no scene id: the oracle cannot run here")
        w = RW.World(RW.loader(os.environ["RO_RUN_DIR"]).get_data_source(scene), arm=ARM)  # lazy, cached per process
        self.s[req.session_uuid] = RW.Session(w)
        return common_pb2.SessionRequestStatus()

    def submit_egomotion_observation(self, req, ctx):          # poses already in the local frame
        self.s[req.session_uuid].add_poses(req.trajectory.poses, req.dynamic_states)
        return common_pb2.Empty()

    def submit_route(self, req, ctx):                          # rig-frame waypoints at req.route.timestamp_us
        self.s[req.session_uuid].route = (req.route.timestamp_us, [(w.x, w.y) for w in req.route.waypoints])
        return common_pb2.Empty()

    def drive(self, req, ctx):
        s, now = self.s[req.session_uuid], int(req.time_now_us)
        ego = s.ego_state(now)                                 # x, y, yaw (rear axle, local), v, a; v from pose differences
        if ARM == "L0":                                        # harness floor: the logged future, re-timed from now
            xyh = s.w.log_future(now, horizon_s=5.0, dt_s=0.1)
        else:
            ref = s.w.reference(ego, route=None if ARM == "A1" else s.route_local(ego))   # (N, 4): x, y, yaw, v_limit
            agents = s.w.agents(now, horizon_s=4.0, dt_s=0.1, only_lead=(ARM == "A2L"), path=ref)  # (A, 41, 5) CV forecast
            xyh = RC.pdm_closed(ego, ref, agents, drivable=s.w.drivable_near(ego),          # callback: boxes -> bool
                                corridor=s.w.log_xy if ARM == "A1" else None)               # (41, 3) at now + 0.1 i
        return egodriver_pb2.DriveResponse(trajectory=RC.to_traj(xyh, now, z=s.w.z(ego)))  # PoseAtTime, quat from yaw

    def get_version(self, req, ctx):
        return common_pb2.VersionId(version_id=f"PRIVILEGED-rule-oracle-{ARM}", git_hash=os.environ.get("RO_GIT", "local"),
                                    grpc_api_version=API)
```

```python
# ro_core.pdm_closed — the PDM-Closed idea on AlpaSim's metric (numpy only)
OFFSETS, FRACS, T, DT = (-1.0, 0.0, 1.0), (0.2, 0.4, 0.6, 0.8, 1.0), 40, 0.1   # navsim metric_cache_processor values
IDM = dict(min_gap=1.0, headway=1.5, a_max=1.5, b_max=3.0)                        # same source

def pdm_closed(ego, ref, agents, drivable, corridor=None):
    paths = [offset_path(ref, d) for d in OFFSETS]                     # parallel_discrete_path; v_limit = min(lane limit, sqrt(2.5/|kappa|))
    st = np.stack([rollout(ego, p, f, agents) for p in paths for f in FRACS])   # (15, 41, 5): IDM along the path with the lead found by
                                                                       #   box overlap on the path, tracked by LQR + kinematic bicycle
    eb = ego_boxes(st)                                                 # (15, 41, 4, 2), rear axle -> 5.176 x 2.297 box
    hit = contact_front_or_side(eb, agent_boxes(agents))              # AlpaSim at-fault: any overlap not on the rear bumper line
    off = ~drivable(eb)                                                # road-area union covers the box (evaluator's rule)
    cor = (lateral_to(corridor, st[..., :2]) >= 3.5) if corridor is not None else np.zeros_like(off)
    ok = ~(hit | off | cor).any(1)
    score = ok * (5 * progress(ref, st) + 5 * ttc_ok(eb, agents) + 2 * comfort(st)) / 12
    if not ok.any():
        return brake_on(paths[1], ego)                                 # PDM's emergency brake: centre path, decel 4 m/s^2
    return st[int(score.argmax()), :, :3]
```

## 3. 控制器与计分里要紧的细节，以及和 PDM-Closed 假设的错配（问题 c）

| 项 | AlpaSim nuPlan（0bb4c4b，本地配置 `generated-user-config-0.yaml` / `eval-config.yaml`） | PDM-Closed（nuPlan / navsim） | 对 oracle 的含义 |
|:--|:--|:--|:--|
| 决策频率 | 2 Hz（`control_timestep_us: 500000`），`planner_delay_us: 0` | 10 Hz 闭环重规划 | 计划要保持 0.5 s；紧急制动的反应晚 0.5 s |
| 冷启动 | 前 0.5 s 强制 GT（`force_gt_duration_us: 500000`），从首帧快门结束起混合到物理位姿；`skip_driver_during_force_gt: false`，k = 0 也调用 drive | 有完整历史 | 无影响：oracle 不需要历史，k = 0 的速度用位姿差（或特权的日志速度） |
| 跟踪 | 非线性 MPC，20 × 0.1 s 视界，每 0.1 s 在同一参考上重解，**只从视界索引 10 起罚参考**（1.0–2.0 s 前的位置 / 航向），加速度 [−9, 6] m/s²，转角 ±0.785 rad；过猛的制动参考会引起之字形转向（CONTROLLER_TUNING.md） | LQR 跟踪 + 运动学自行车，打分时也用它前向模拟 | 计划的 0–1 s 段不被跟踪；1–2 s 处位置不可达时 MPC 用任意加速度去追。打分用的前向模拟最好按「保持 0.5 s、只跟 1–2 s 处」近似；急刹参考要限幅（估 ≤ 4–5 m/s²） |
| 计划长度 | 返回 ≥ 4 s；sh30 driver 返回 8 × 0.5 s 重采样到 10 Hz | 提案 4 s @ 10 Hz（40 点），输出可延到 8 s | 返回 41–51 个 0.1 s 位姿（4–5 s）即可，不需要 8 s |
| 走廊 | `left_corridor_laterally`：ego 多边形质心到**完整日志路径**的横向距离 ≥ 4.0 m（越过日志终点后只算垂直于末段的分量）；`RemoveTimestepsAfterEvent` 在首个 ≥ 4 m 的步截断 | 无（nuPlan 用 route roadblocks） | 规则规划器必须沿日志走过的车道；日志换道时不换就可能贴近 3.7 m。A1 把走廊作为硬约束（3.5 m）；A2 不知道日志，只靠路线路点 |
| 越野 | ROAD_AREA（25 m 内）的并集，0.001 m 缝隙闭合后必须 `covers` 整个 ego 多边形；附近没有 ROAD_AREA 也算越野 | 可行驶区域：ego 四角都在 nuPlan 可行驶图层内 | 比 PDM 严：整车足迹。oracle 直接用评测器同一个函数 |
| 碰撞 at-fault | 每步 ego 多边形与任一物体相交：碰到 ego **前保险杠线** = front，碰到**后保险杠线** = rear，否则 lateral；`collision_at_fault` = front 或 lateral（只有追尾不算） | nuPlan：停着被撞、被追尾不算；按碰撞类型和车道归责 | 回放车辆**不反应**：侧面被并线车蹭到、停着被横向撞都算我们的。CV 外推躲不开加塞；看真未来的臂（探索）能躲 |
| 进度 | 场景分 = 0（有 at-fault、越野、出走廊之一），否则 `min(progress_rel_to_total / 0.8, 1)`；日志行驶 < 5 m 时为 1 | PDM 打分里 progress 权重 5 | 要跑到日志路程的 80%：IDM 目标速度太保守会扣分。目标速度 = 车道限速（gpkg），缺失时 15 m/s，曲率上限 √(2.5/κ) |
| 交通灯 | 不计分；回放车辆照日志在红灯前停 | PDM-Closed 把红灯车道当障碍 | 第一辆车在红灯前时 oracle 会闯过去，横穿的回放车可能侧撞（at-fault）。可选：特权地用日志 ego 的停车（A1 的走廊 + 限速代替不了这一点，估影响 < 5 个 scene） |
| 时间窗 | 18 决策（`num_timesteps_after=160`，token + 8.0 s）；6–8 个路线折返 scene 不起会话，计分 452 个 | 无 | 同 alpasim_long |

## 4. 臂、阶段与成本（问题 d）

**臂**（单 seed；规则规划器确定性，仿真器在固定 scene 列表下确定性，所以一个 seed 就是读数）：

| 臂 | 横向 | 纵向 | 路线 | 车辆 | 角色 |
|:--|:--|:--|:--|:--|:--|
| **L0** | 日志未来轨迹（按 now 重新取时） | 日志 | — | — | 本线束的下限：日志本身在 MPC + 评测器下有多少零分。「可避免的零分」= 某臂的零分减去 L0 也零的 scene |
| **A1** | PDM 提案 | IDM | **R-log**：`RouteGenerator.create(ds.rig.trajectory.positions, ...)` 的 `route_polyline_in_local`（日志投影到最佳车道序列，再沿车道延伸），换道处的横向跳变在 ≥ 20 m 上平滑；外加 3.5 m 走廊硬约束 | 全部回放车辆，CV 外推 | 上限读数（最强特权） |
| **A2** | PDM 提案 | IDM | **R-wp**（诚实路线）：收到的路点转回 local 系，按中心线距离 ≤ 0.5 m 找出路点所在车道，从 ego 当前车道沿 `next_lanes`（必要时经 `adj_lanes_*` 换道）图搜索到第一个路点车道；分岔前 40 m 内由「能到达路点车道」决定 | 全部回放车辆，CV 外推 | 主读数：「地图完美、路线诚实」时规则头值多少 |
| **A2L** | 同 A2 | IDM | R-wp | **只给同车道前方最近一辆**（100 m 内，即 openpilot lead head 能给的） | 桥：去掉旁车与横穿车后还剩多少 |
| A3（探索，不登记） | 同 A2 | `YR10m10-F-s0` 的速度剖面（sh30 driver 作上游，oracle 当代理，模板 `experiments/alpasim/scripts/driver_tap.py`：转发全部 RPC，用学到的弧长-时间剖面重新取时规则路径） | R-wp | 全部 | 回答「横向规则 + 纵向我们的」 |
| A1F（探索，不登记） | 同 A1 | IDM | R-log | 回放车辆的**真实未来** | 上限之上限：躲加塞 |

不做「只用左 / 右 / 直指令」的路线（R-cmd）：任何 driver 都收到车道级路点，形态 2 也能用；R-cmd 只在路点不能用时才有意义，而指令
本身在路口前几秒亮起、不告诉路口在哪，直路上 2.7° 航向误差就能翻转（第 205 条、第 268 条 H1），作为诚实版本反而低估规则头。若以后需要：
分岔处取 `lane_connectors.turn_type_fid` 与指令一致、航向变化符号一致的后继。

**阶段**（docs/long-runs.md：1 → 约 10 → 全量）：

| 阶段 | 内容 | 门 |
|:--|:--|:--|
| S0（CPU，盒子，AlpaSim venv，不进池） | `ro_world_priv` 离线核对，10 个 train scene（`experiments/onpol_nuplan/results/long/smoke10_train.txt`） | (a) 每个 scene 的 `ds.rig.trajectory.interpolate_pose(17000)` 与存档 `runs/alpasim_long/smoke/runs/b160/driver-logs/drive.jsonl` 的 k = 0 anchor 差 ≤ 0.01 m；(b) 当前车道 id 在 gpkg 里查得到限速 ≥ 95%（查不到的记下）；(c) 日志路径的 ego 足迹在 `drivable_near` 下越野步 = 评测器在 L0 上的越野步（S1 后对照）；(d) 开环：在日志状态上跑 `pdm_closed`，plan 4 s 端点与日志的中位差、提案全零的比例，每次调用耗时 p95 < 0.25 s |
| S1 冒烟（池，10 个 train scene，`after160`） | L0 与 A1 两个作业 | 10 / 10 rollout、0 driver 错误、18 决策；`drive` p95 < 1 s；L0 ≥ 9 / 10 非零 |
| S2 held（池，`experiments/onpol_nuplan/results/split/held.txt`，460 → 计分 452） | L0、A1、A2、A2L 四个作业，`cl fanout` | 读数，§5 |
| S3（条件：§5 的 R1 成立） | A2N：A2 的地图车道几何按并行 agent 测出的 openpilot lane line 误差（偏置 / σ 随距离）加噪，车辆按 A2L；然后才写形态 2 | 另行登记 |

**命令**（S1 / S2；`drivers/ro.sh` 是新加的 5 行文件，照 `garage.sh` 的写法指向 `experiments/rule_oracle/lib/ro_driver.py`，
cwd = `$SRC`，解释器 `$SRC/.venv/bin/python`，环境 `RO_ARM RO_RUN_DIR=$out RO_PRIVILEGED_OK=1 OMP_NUM_THREADS=1`）：

```bash
# box, repo root, tmux jev; BASE / win as in experiments/alpasim_long/scripts/al_chain.sh (c0b_chain.OVERRIDES + after160)
O=$DATA_DIR/runs/rule_oracle/held; SPL=experiments/onpol_nuplan/results/split
.venv/bin/python -m jevdrive.cl fanout --arms L0,A1,A2,A2L --name ro-h18 --owner rule-oracle --vram 32 --cpu 12 --ram 90 \
  --tries 1 --priority 12 --timeout-h 3 --no-check --log-dir "$O/{arm}/pool" \
  -- bash -c "mkdir -p $O/{arm} && RO_ARM={arm} exec bash experiments/alpasim/scripts/run.sh $O/{arm} ro --scene-list $SPL/held.txt ${BASE[*]} $(win 160)"
# read: the same collector and truncation as the baselines
$DATA_DIR/third_party/alpasim/.venv/bin/python experiments/alpasim_long/scripts/long_read.py collect --scenes $SPL/held.txt --out $O/read --jobs 32 \
  --runs L0@18=$O/L0 A1@18=$O/A1 A2@18=$O/A2 A2L@18=$O/A2L \
  YR10m10-F-s0@18=$DATA_DIR/runs/alpasim_long/held/h18/YR10m10-F-s0 \
  YS-L18-s0@18=$DATA_DIR/runs/onpol_long/read/YS-L18-s0/run YS-L18-s1@18=$DATA_DIR/runs/onpol_long/read/YS-L18-s1/run
```

**成本**（引自 alpasim_long 的 held 计时；loop 受渲染器限制，driver 换成 CPU 规则规划器不改变渲染）：

| 项 | 依据 | 估计 |
|:--|:--|:--|
| 一个 460 scene × 18 决策作业 | held h18：34–37 min wall / 作业（4.6 s / scene / stack），8 个作业约 4.5 card-h；runtime RSS 33–34 GiB，渲染器 VRAM 峰 23–32 GiB | 35–40 min wall，约 0.6 card-h；`--vram 32 --cpu 12 --ram 90` |
| driver CPU | 460 × 18 = 8 280 次 `drive`；估每次 50–150 ms（15 提案 × 40 步向量化 + 每决策一次 ROAD_AREA 并集）；8 个并发会话按会话分到 4 个进程 | 14–20 CPU-min / 作业，不进关键路径（每个 rollout 每决策约 2 s wall 预算） |
| 首会话 | loader 6.2 s / 进程；地图 0.7 s / scene | 可忽略 |
| S0 | CPU，10 scene | < 10 min |
| S1 冒烟 | 2 个作业 × 10 scene | 约 10 min wall，约 0.1 card-h |
| S2 held 四臂 | 4 个作业并行 | 约 45 min wall，约 2.4 card-h |
| 磁盘 | 44 MB / scene / 作业剪枝前，剪枝（只留零分的 `rollout.asl`）后约 15 GB / 4 作业 | 剪枝前约 80 GB，剪后约 12 GB（估） |
| 写代码 | `ro_core.py` 约 250 行，`ro_world_priv.py` 约 150 行，`ro_driver.py` 约 100 行，`drivers/ro.sh` 5 行 | 1 天 + S0 半天 |

合计：约 2.5 card-h，约 1.5 h wall（S1 + S2），外加 1.5 天工程。

## 5. 登记的判词线（问题 e）

**比较集**：452 个计分 held scene（held.txt 去掉不起会话的路线折返 scene），全部经 `long_read.py collect` 的同一截断规则重算。基线（ONPOL-LONG
的 452 表，第 264 条）：`YR10m10-F-s0` 0.785 / 零分 74（碰撞 10、越野 33、走廊 35）/ at-fault 零分 43 / GT 1.495 m；`YS-L18`（两 seed 均值）
0.838 / 44（9 / 14.5 / 22）/ 23.5 / 1.357 m。CI：按 log 聚类的 bootstrap（10 000 次，`jevdrive.stats` 约定），配对同一 scene；`YS-L18` 用两 seed 的 scene 均值。

**报告的量**（每臂一行，外加对 `YS-L18`、`YR10m10-F-s0` 的配对差）：平均场景分；零分总数及按类型（at-fault 碰撞 front / lateral、越野、走廊）；
at-fault 零分；at-fault km（`avg_dist_between_incidents_at_fault`）；GT 距离（每 scene 截断后 `dist_to_gt_trajectory` 最大值的均值）；
「可避免零分」（L0 也零的 scene 剔除后）；首个零分事件的决策序号分布。

**判词线**（主读数是 A2）：

- **R0（线束下限可信）**：L0 场景分 ≥ 0.95 且零分 ≤ 15。否则先查 L0 的零分（地图 / 回放车辆 / MPC 跟踪），A1 / A2 只按「可避免零分」读。
- **R1（规则头有明显上限）**：A2 − `YS-L18` 平均分 ≥ +0.05 且 CI 下界 > 0，**并且** A2 零分 ≤ 22（`YS-L18` 的一半），**并且** A2 at-fault 零分 ≤ 12。
- **R2（没有上限）**：A2 − `YS-L18` 的 CI 上界 < +0.02，或 A2 零分 ≥ 44。
- 介于两者之间：**不定**，按零分类型写清楚差在哪里，不升级成支持。
- **R3（路线特权值多少）**：A1 − A2 的走廊零分差；≥ 10 个记为「诚实路线丢了日志车道选择」，形态 2 的路线处理要先解决。
- **R4（只有前车够不够）**：A2L − A2 的 at-fault 碰撞零分差；≥ 8 个（或 A2L 的平均分比 A2 低 ≥ 0.03 且 CI 不含 0）记为「旁车 / 横穿车是缺口」，
  openpilot lead head 单独撑不起形态 2。

**各结果对协调者的含义**（「学到的眼睛 + 规则头」对「现行后果训练线」）：

| 结果 | 含义 | 下一步 |
|:--|:--|:--|
| R1 成立、R4 不成立 | 地图完美时规则头远胜学到的 plan，且只要前车就够：缺口在「车道几何读得准不准」 | 拿并行 agent 的 head 误差做 S3（A2N）；A2N 仍比 `YS-L18` 高 ≥ +0.03 才建形态 2（`envs/op-train` 里的 sh30 driver 加 lane-line 中心线 + lead IDM，路口与 head 不可信时退回学到的 plan） |
| R1 成立、R4 成立 | 规则头的优势来自看见所有车；openpilot 只给前车，零样本版拿不到 | 规则头不是近期杠杆；后果训练线继续（它隐式处理旁车），可考虑把 A1F / A2 当离线诊断用 |
| 不定 | 规则头与 `YS-L18` 同量级 | 后果训练线继续；oracle 只作零分类型的参照 |
| R2 | 即使地图完美，规则头也不比学到的强：零分主要不是车道几何问题（估：不反应的回放车辆、MPC 的 1–2 s 跟踪窗） | 规则头路线关闭；把 A2 剩下的零分类型交给后果训练线作「任何 driver 都会撞」的基线 |
| R0 不成立 | 线束本身（地图、回放、跟踪）给日志也打零分 | 所有本地读数都要减去这部分；先报告 L0 的零分清单 |

## 6. 阻塞与风险（问题 f）

**最大的两个风险**：

1. **2 Hz + MPC 只跟 1–2 s 的错配让 PDM 的闭环优势打折。** PDM-Closed 的强项是 10 Hz 重规划和对 IDM 前车的即时反应；这里每 0.5 s 才重规划，
   计划的前 1 s 不被跟踪，急刹参考还会引起之字形转向。A2 的分数里有一部分会是「规则规划器 × 这个控制器」的交互，不全是规则头的上限。
   缓解：打分前向模拟按 0.5 s 保持近似；制动参考限幅；S1 看 L0 与 A1 的跟踪误差（`drive.jsonl` 里计划与实际的偏差）。
2. **不反应的回放车辆。** AlpaSim 的 at-fault 只排除追尾：被加塞、停着被侧撞都算我们的。CV 外推的规则规划器躲不开这一类，A2 的零分里会有
   一块「任何不看未来的 driver 都躲不开」的地板；A1F（真未来）只作探索参照，**不能**拿来证明规则头可行。另外第一辆车闯红灯后的横穿碰撞同属此类。

**其他风险**：
- 限速：trajdata 没有，靠 gpkg 按 lane id 查；地图版本若与 `.pb` 不一致（`.pb` 来自主办方的 nuplan_test 地图）会有查不到的 id，S0 (b) 量化。
- 日志换道：R-log 的投影路径在换道处有横向跳变，平滑距离影响走廊零分；R-wp 的图搜索对「两条平行车道都能到路点车道」的情况要定规则
  （取 ego 当前车道优先，路点车道要求的换道放在最后一个能换的路段）。
- 速度：shapely 在多线程 gRPC 下受 GIL 限制；按会话分进程（4 个）即可，S0 (d) 量化。
- 路线折返 scene 与 14 个日志提前结束的 scene 同 alpasim_long，口径一致即可。

**特权泄漏的防线**（形态 1 绝不能成为任何推送的镜像，也不能进任何训练或选型）：
- 代码分层：`ro_core.py`（纯 numpy，无数据访问）可以被形态 2 复用；`ro_world_priv.py` 与 `ro_driver.py` 只在 AlpaSim venv 里跑，导入
  `alpasim_runtime` / `trajdata`，读 `$DATA_DIR/datasets/alpasim_nuplan*` 和 run 目录的配置。提交镜像里没有这两个包、没有 scene 数据、
  官方评测也不给 `debug_info.scene_id`，所以即使误装进去也在 `start_session` 就失败（`ro_driver.py` 遇到空 scene id 直接 abort）。
- `RO_PRIVILEGED_OK=1` 只在 `drivers/ro.sh` 里设；`get_version` 返回 `PRIVILEGED-rule-oracle-<arm>`，任何 run 的 `driver-config` / 结果
  一眼可辨。
- 镜像构建：`experiments/alpasim/docker/closure.txt` 不列 `experiments/rule_oracle/` 的任何文件；建议在 `docker/verify.py` 加一条：closure 里出现
  `rule_oracle/` 或任一 closure 文件 import `ro_world_priv` / `alpasim_runtime` / `trajdata` 即构建失败（属 alpasim 线的改动，由协调者决定）。
- 数据：oracle rollout 不进任何训练行、标签池或 checkpoint 选择；结果只写进本 topic 与 decisions。形态 2 若建，是另一个 driver（`envs/op-train`，
  只读 openpilot head 与收到的路点），先过 docs/alpasim.md 的提交前审计，与本稿的特权 driver 无代码依赖（只共用 `ro_core.py`）。

## 7. 形态 2（零样本非特权）的范围

- 输入：sh30 driver 已有的一切（CAM_F0 帧、ego 位姿 / 速度、路点）+ 同一次策略前向里 openpilot 的 `lane_lines`（4 × 33 × 2，`X_IDXS` 上的
  y / z）、`lane_lines_prob`、`road_edges`（2 × 33 × 2）、`lead`（3 个时刻 × 最佳假设 × (x, y, v, a)）、`lead_prob`。这些切片在
  `self.model.net.slices` 里，`experiments/alpasim/scripts/swv1_replay.py` 已经在 AlpaSim 的回放里读过它们（FT 与 P0 两套）；冻结版用 P0
  （去掉 adapter 的同一次策略前向，`sh30_core.Core.base`）。坐标约定见 `lib/vm3_perception.py`（相机在 rig 前 1.786 m；openpilot y 向右为正，
  要翻成 rig 的 y 向左）。
- 规划：`ro_core` 的中心线 = 两条本车道线的中点（两者 prob > 0.5 时），横向偏移受 road edge 约束，碰撞只对 lead（同车道），越野用「足迹在两条
  road edge 之间」的 numpy 检查；路口、车道线消失、或路点要求转弯时退回学到的 plan 的路径，纵向取 min（学到的、IDM on lead）。
- 环境：`envs/op-train`（3.11）与提交镜像都没有 shapely，所以 `ro_core` 必须纯 numpy（本稿的设计已如此）。
- 前提：并行 agent 报告 head 在存档 rollout 上的精度；本稿 §5 的 R1 成立且 R4 不成立。否则不建。

## 8. 交付物与路径

- 本稿；`experiments/rule_oracle/README.md`（planned）；INDEX 一行。
- 实现时新增：`experiments/rule_oracle/lib/{ro_core,ro_world_priv,ro_driver}.py`、`experiments/rule_oracle/scripts/ro_s0.py`（S0 离线核对）、
  `experiments/alpasim/scripts/drivers/ro.sh`；结果 `experiments/rule_oracle/results/`，盒子 `$DATA_DIR/runs/rule_oracle/{s0,smoke,held}`。
- 参考：`experiments/anchor_offset/scripts/ao_e1.py`（loader、路线重算）、`experiments/zeros_mech/scripts/zm_read.py`（回放车辆插值、车道方向）、
  `experiments/alpasim/scripts/driver_tap.py`（代理模板）、`$DATA_DIR/third_party/navsim/navsim/planning/simulation/planner/pdm_planner/`
  （可搬的 numpy 部分）、`$DATA_DIR/third_party/alpasim/src/eval/src/eval/scorers/{collision,offroad,ground_truth}.py` 与
  `aggregation/processing.py`（指标定义）。
