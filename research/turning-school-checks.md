# 转弯学校的两项前置检查（2026-10-13，CPU、只读）

对象：「转弯学校」方案的 P2（分支示范）开关与 P1（沿人类日志渲染）可行性；PAI 转弯 scene 的数量。
只读 Tokyo 的 AlpaSim 源码（`/data/third_party/alpasim`，0bb4c4b）与已存 rollout；没有起仿真，没有用 GPU。
统计脚本是一次性 scratch（Mac `tmp/tsc/`，未提交）：在 `alpasim-base:0.89.0` 里用 run 自己的 `generated-user-config-0.yaml` 建 scene loader，
读每个 scene 的日志自车轨迹 `ds.rig.trajectory`（位置 + yaw），命令取 driver 的 `drive.jsonl` 决策 0 的 `cmd`。

## 1. 数字（检查 2：转弯计数）

**定义**：日志自车轨迹在 rollout 窗口（开头 20 s = 200 个 0.1 s 步）内，按 1 m 弧长取 yaw，任意 30 m 弧长内 yaw 变化 ≥ 60° 即「转弯 scene」，
正负号分左右。「静止起步」= 日志在交接时刻 t = 1.7 s（force-GT 结束）车速 < 0.5 m/s。「命令已亮」= 决策 0（t = 0，仍在 force-GT 里，自车在日志上，
所以与 checkpoint 无关）的 `cmd` 与转弯方向相同（L = 0、R = 2；`navigation.command_from_route`：第一个 ≥ 5 m 的路点横向 > 2 m）。
命令来源：train160 用 RB0-F-s0 重滚（`/data/runs/route_night/roll/rb0` + `rb0_retry`），281 用 RSG-L18-s0 读数（`/data/runs/route_night/r281/rsg`）。
入口 = 第一个达到 60° 的 30 m 窗口里 yaw 偏离窗口起点 5° 的位置，按离起点的弧长与日志到达的时间计。

| 列表 | scene | 转弯 scene（≥ 60° / 30 m） | 左 / 右 | 其中 > 125°（掉头） | 静止起步 | 静止起步且命令已亮 | 命令在 t = 0 已亮（全部） | 入口在起点 15 m 内 | 入口中位（弧长 / 时间） |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| train160 | 160 | **23**（14 %） | 13 / 10 | 2 | 6 | **6** | 15 | 15 | 10 m / 7.7 s |
| test281 | 281 | **46**（16 %） | 22 / 24 | 3 | 9 | **8** | 28 | 15 | 31 m / 8.9 s |
| exam hard | 131 | **42**（32 %） | 20 / 22 | 3 | 8 | 7 | 25 | 13 | 34 m / 9.3 s |
| exam control | 75 | **0** | 0 / 0 | 0 | 0 | 0 | 0 | 0 | — |
| exam trainable | 75 | **4**（5 %） | 2 / 2 | 0 | 1 | 1 | 3 | 2 | 12 m / 4.9 s |

阈值敏感性：≥ 45° / 30 m 时 train160 24、281 52（hard 47 / control 0 / trainable 5）；不限半径、窗口内净转角 ≥ 60° 时 train160 27、281 56（49 / 2 / 5）。
静止起步的转弯里几乎全部 t = 0 命令已亮（train160 6 / 6、281 8 / 9）；入口离起点远的转弯在 t = 0 多半还是 S，命令随后才亮（route-follow 2.4：入口前中位 37 m / 4.3 s）。

**这些 scene 现在的分**（`experiments/route_follow/results/taxonomy/pai281_*.csv`）：281 里 46 个转弯 scene，RSG-L18-s0 均分 0.232、零分 31（其余 235 个 0.639、零分 39）；
ys0 0.159 / 35。左转 22 个 RSG 0.185 / 16，右转 24 个 0.275 / 15。转弯是 281 上最集中的失分处：16 % 的 scene 占了 44 % 的零分。

**route-follow 2.4 的 48 个转弯 scene**（「在 60–125° 转弯处失败的零分 rollout」所在的 scene）：原始列表没有存下（子智能体的 `/tmp/pw` 已清），
用它留在 Mac `tmp/pai_watch/` 的 `ext.py / feat.py / cls.py` 原样在 Tokyo 重跑（规则不变；pai281 的 7 个 arm，现在 738 个留 asl 的零分 rollout，
其中 270 个落在转弯处），得到 **48 个 scene，与原数相同**：**hard 47、control 0、trainable 1**（`2c0ef536`）。
这一点是构造上的：hard 的定义就是「任一参照零分」，48 个都来自参照 arm 的零分，所以只可能落在 hard，唯一的例外是 trainable 的 `2c0ef536`
（它的零分来自 ynone / ys2g 这些不是 exam 参照的 arm）。
48 个与上表日志口径的 46 个重合 35 个：48 里另外 13 个是路线转了、日志在片段内没转满 60° / 30 m 的（路线比日志多伸 40–80 m）、45–60° 的和变道；
46 里另外 11 个（hard 7、trainable 4）是没有参照在转弯处零分的转弯。

**含义**：control 75 里一个转弯 scene 都没有。exam 若用 hard + control，转弯能力只在 hard 一侧读得到，control 不能当转弯的对照；trainable 只有 4 个转弯，
PAI 自己的转弯训练料在 train160 的 23 个（另加 281 trainable 的 4 个）。静止起步、命令已亮的转弯只有 6（train160）+ 8（281）个。

## 2. 检查 1：能不能从 rollout 中途的状态开始、由我们的 driver 接着开

**结论：没有现成的「中途起步 / 快照 / 回放」接口；但可以做到，最便宜的精确做法是改 AlpaSim runtime 的 force-GT 轨迹（约 20 行，不用重建镜像），
最便宜的零改 AlpaSim 做法是 log-following driver 在决策 k 交给策略（约 5 行，状态近似）。P2 不被卡死。**

### 2.1 源码事实（AlpaSim 0bb4c4b，`src/runtime/alpasim_runtime/`）

- **起点固定在场景开头**：`unbound_rollout.py::_build_rollout_timing` 把 `egomotion_context_start_us` 设成 `data_source.rig.trajectory.time_range_us.start`，
  渲染起点是第一帧相机快门结束，`closed_loop_start_us = render_start + simulation_config.force_gt_duration_us`。usdz scene provider（PAI 用 `scene_provider.kind: usdz`）
  没有起始偏移参数；`runtime.proto` 的 `SceneMetadata.start_time_us` 只是元数据。grpc proto（runtime / physics / controller / egodriver / traffic / sensorsim）里
  没有 initial_state、reset、snapshot、checkpoint、rewind、branch 一类的 RPC 或字段；`autoresume.py` 只是续跑没跑完的整条 rollout。
- **唯一的「把自车放到指定状态」机制是 force-GT**：`SimulationConfig.force_gt_duration_us`（PAI dev preset 1 700 000；wizard 覆盖
  `runtime.simulation_config.force_gt_duration_us=<us>`，须是 100 000 的倍数，`sensorsim_service.validate_timing_alignment`）。force-GT 期间
  `events/controller.py` 用 `state.force_gt_trajectory` 作参考、`controller_service.run_controller_and_vehicle(force_gt=True)` 直接返回该轨迹的位姿；
  交接那一步（`handoff`）用 `force_gt_dynamics`（`events/force_gt_utils.force_gt_dynamic_trajectory`，三次样条速度 / 加速度）给控制器灌初始动力学状态。
  driver 在 force-GT 期间照常被调用并收到渲染帧（`skip_driver_during_force_gt=false`），所以它的历史缓存在交接时是满的。
- **force-GT 轨迹来自哪里**：`event_loop.py::_build_force_gt_physics_blend_trajectory` 从 `unbound.gt_ego_trajectory`（日志）插值，再只在
  z / roll / pitch 上做地面修正（PAI `physics_update_mode: EGO_ONLY`）；`_create_rollout_state` 由它算 `force_gt_dynamics`。
  `gt_ego_trajectory` 另外还被路线生成器（`route_generator`，MAP、offset 40 m）和评估器（走廊、到日志的距离）用，所以不能整体替换。
- **他车与自车无关**：PAI 的 `endpoints.trafficsim.skip: true`、`physics_update_mode: EGO_ONLY`，他车是日志回放。分支时刻 t_k 的他车状态
  只取决于 t_k，与自车之前怎么开无关；原 rollout 在 t_k 的他车就是分支的他车。
- **runtime 源码是挂载的**：pai_run.sh 生成的 compose 把 `$ALPASIM_SRC/src` 挂到每个服务的 `/repo/src`（runtime、controller、physics 都是
  `alpasim-base:0.89.0` 镜像 + 挂载源码），所以一份打了补丁的源码拷贝用 `ALPASIM_SRC=<拷贝>` 即可生效，不用重建镜像。

### 2.2 四条路，按推荐排序

| 做法 | 改动 | 分支状态 | 粒度 |
|:--|:--|:--|:--|
| **A. force-GT 轨迹覆盖（精确）** | runtime 补丁：`event_loop.py::_build_force_gt_physics_blend_trajectory` 里，t ≤ t_k 段用原 rollout 的自车轨迹替换 `gt_trajectory`（`controller/alpasim_controller_<rid>.csv` 10 Hz x / y / yaw，或 asl 的 egomotion；z 取日志），`gt_ego_trajectory` 本身不动；加 `force_gt_duration_us = t_k − render_start` | 自车位姿逐位等于原 rollout 到 t_k；速度由样条差分（与平时 1.7 s 交接同一机制）；他车精确；driver 历史是这段轨迹上的新渲染帧 | `force_gt_duration_us` 是整个 run 一个值；每 scene / 每 rollout 不同的 k 要在补丁里按 scene 读一个表 |
| **B. 日志状态起步（零改动）** | 只加 wizard 覆盖 `runtime.simulation_config.force_gt_duration_us=<k·100000>` | 自车精确在**人类日志**的 t_k 状态（不是策略的中途状态），他车精确 | 每 run 一个 k；就是 onp_pass2.sh `MODE=force` 把 30 s 换成 t_k |
| **C. onp_log_driver 在决策 k 交给策略（零改 AlpaSim）** | `experiments/onpol_pai/lib/onp_log_driver.py::LogDriver.drive` 现在先调 `super().drive`（模型照常推理、写记录）再丢掉它的 plan、返回日志参考；改成 `now ≥ t_switch(scene)` 时返回模型 plan，约 5 行，t_switch 按 `s.scene` 从一个环境变量指定的表读 | 自车由 MPC 跟日志参考到 t_k，状态**近似**日志（偏差由 `onp_track.py` 量）；模型状态（帧历史、VCONT、lead、路线记忆）全程在热；交接没有 coerce，是 MPC 的连续切换 | 每 scene 任意 k；需要 `send_recording_ground_truth=true` |
| D. 重放存下的 plan 到 t_k | driver 变体前 k 步返回存档 plan | 只有闭环逐位可复现时才精确（见 2.3），否则近似 | 每 scene 任意 k；不如 A |

**协调方的问题（onp_log_driver 能否在选定的 k 交给策略）**：能，就是 C；跟日志开到路口入口、再让策略接着开，是零改 AlpaSim 的最便宜分支。
它的局限：分支点是「日志附近」而不是「策略自己开到的状态」，所以它示范的是「从正确的入口状态起转」，不是「从策略犯错的状态救回来」；
后者要 A。B 是 C 的精确版（自车逐位在日志上，交接用 runtime 自己的 coerce），代价是一个 run 只能一个 k，按入口时间把 scene 分组即可。

### 2.3 确定性证据（D 能不能用）

- 第 286 条：pai_driver 加路线记忆支路后，在 PAI 上 2 个 scene 重滚，398 / 398 个决策与存档 ys0 **逐位相同**（同一台机、同一镜像）。
- 第 241 条：PAI 上同一 tag 重跑（CONC 4、多卡），均分差 0.017 / 0.010：**闭环一般不逐位可复现**。
- 第 215 条：nuPlan 容器化 stack 自己重复逐 scene 相同；换硬件后 driver 侧差 1 个 fp16 步长，到决策 9 plan 差 0.16–0.51 m（闭环放大）。
- steer_long `sl_mpc.md`：离线逐步重放 AlpaSim 的 MPC，转向命令与日志差中位 0.0005 rad（接近、非逐位）。

所以 D 只在「同机同卡同并发」时大概率精确，不能当前提；A 不依赖确定性（位姿是强制的）。

### 2.4 代价（每个分支）

PAI 吞吐：RB0 在 train160 上 160 scene / 5 102 s、两张 3090、每卡 4 并发 → 约 56 scene / 卡时，一条 200 步 rollout 约 64 卡秒（含 stack 启动）；
渲染占 81 %，而渲染每一步都做（force-GT 期间也渲染，driver 要帧）。没有快照，所以每个分支都要把前缀重新渲染一遍，同一前缀的多个分支不能共享。
分支代价 ≈ (k + H) / 200 × 64 卡秒，H 为续开步数；`runtime.simulation_config.n_sim_steps` 可以把 run 截到 k + H。
例：路口入口中位约 9 s（k ≈ 90）、续开 8 s（H = 80）→ 约 54 卡秒，约 0.85 条整 rollout；46 个转弯 scene × 每个 4 个分支 ≈ 2.8 卡时（Tokyo 两卡约 1.4 h 墙钟）。
A 和 B / C 代价相同，差别只在状态是否精确、要不要改 AlpaSim。

### 2.5 P1：沿人类日志渲染

已有，不用新写：`onp_log_driver.py`（`onp_pass2.sh` 默认 `MODE=driver`：driver 返回日志、MPC 跟踪，`send_recording_ground_truth=true`）或
`MODE=force`（`force_gt_duration_us=30 000 000`，runtime 自己把自车钉在日志上，pai_driver 不改）；ONPOL-PAI 已经这样滚过 80 个 train scene
（`/data/runs/onpol_pai/pass2`）。AlpaSim 自带同类预设 `wizard/configs/exp/sim/force_gt.yaml`（20 s force-GT + `perfect_control` + 关物理）。
注：任务里提到的 nuPlan `smoke10e` 不是日志跟随 driver，而是 R1 闸门用的**解释执行（不编译）**同一 checkpoint 的 rollout（`experiments/onpol_nuplan/results/rows/smoke10e`）。

## 3. 没核实的

- A 的补丁没有写、没有跑；z 取日志、速度用样条差分后，交接那一步的动力学与原 rollout 在 t_k 的控制器状态（转向角、加速度）不完全一样，
  需要一个 smoke（同一 scene，k 取原 rollout 的若干点，比较分支前缀的位姿与原 rollout 的差 = 0、以及交接后 1 s 的偏差）。
- C 跟日志到 t_k 的偏差没有读（`onp_track.py` 在 pass2 上有现成读法）。
- 转弯计数用的是日志 yaw，不是路线；路线在片段末尾之外还有 40–80 m，日志没转满的路口不计入（48 集合里这类 13 个）。
