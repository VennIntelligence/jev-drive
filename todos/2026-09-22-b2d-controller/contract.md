# 模块协作契约

Python 3.8；code/comments/logs English，工作笔记中文。所有输出的 steer 为 CARLA 正方向（右转为正）。
轨迹内部 x 前 y 左、后轴原点；yaw_rate_rps 参数采用该坐标系（左转为正），由 agent 边界转换 IMU 符号。

## Controller

`Controller(preset='carla', wheelbase=..., max_steer_deg=..., steering_curve=..., lookahead='...', speed_window='near')`
参数中标定默认由主代理稍后提供；先用明确标注 synthetic 的值开发。
`update(traj_xy, t_frame)` 接收 20×2/4 Hz/5 s 轨迹；`step(t_now, speed_mps, yaw_rate_rps)` 推进历史并返回
`(throttle, steer, brake)`；第一 tick 可先 update 再 step，延迟帧使用之前 step 的历史。
`reset()` 清空跨 route 状态；`diagnostics` 为 JSON-compatible dict，字段至少：
`trajectory_time`, `trajectory_age_s`, `target_speed_mps`, `reference_speed_mps`, `aim_xy`,
`cross_track_m`, `heading_error_rad`, `reason`。无有效数值用 null，不能用 NaN。
额外 constructor 参数由 controller 与 agent 子代理直接协商并同步本契约；主代理标定文件传给 agent。

## Agent / runner config

runner 和 route 透传 `--drive controller`、`--controller-preset carla|tcp|pursuit`、`--cruise-mps 8`、
`--tm-seed 0`、`--controller-config <JSON>`。config JSON 中用 `drive`, `controller_preset`, `cruise_mps`,
`controller_config`（参数 JSON 路径）, `out`（attempt 目录）。CLI 的校验和细节由 report 子代理实现。
`policy=none` 的 route 轨迹同步更新，每 decimate tick 一次；控制及运动传感器每 tick 一次。
控制器模式先拒绝尚未支持的真实 policy，不假称已有 planner 轨迹输出。

## control.jsonl

每有效 tick 一条，顶层字段：`frame`, `sim_time`, `speed_mps`, `yaw_rate_rps`, `throttle`, `steer`, `brake`,
`trajectory_time`, `trajectory_age_s`, `target_speed_mps`, `reference_speed_mps`, `aim_xy`, `cross_track_m`,
`heading_error_rad`, `reason`；`sensor_frames` 字典；`pose_xy`, `pose_yaw`（CARLA world yaw）；
`raw_pose_error_m`, `pose_error_m`, `pose_heading_error_rad`（诊断真值 logger 计算）；
`route_cross_track_m`（估计位置相对 dense route）；`truth_cross_track_m`（仅 logger，用真值相对 route）。
主指标优先 `truth_cross_track_m`；缺失时报告缺失并可单列 estimated 指标，不能默认为同一量。
controller diagnostic cross-track 是相对短时参考，不可混同 route cross-track。

官方驾驶结果由 reporter 从 attempt 的 `results.json` 读取。runner finished 仍仅表示 evaluator 返回。
所有异常/缺失/重试均保留，不挑最佳 attempt。
