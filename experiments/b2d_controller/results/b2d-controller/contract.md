# 模块协作契约

Python 3.8；code/comments/logs English，工作笔记中文。所有输出的 steer 为 CARLA 正方向（右转为正）。
轨迹内部 x 前 y 左、后轴原点；yaw_rate_rps 参数采用该坐标系（左转为正），由 agent 边界转换 IMU 符号。

## Controller

`Controller(preset='carla', wheelbase=..., max_steer_deg=..., steering_curve=..., lookahead='...', speed_window='near')`
标定默认已实测：MKZ2020轴距2.860471491m、后轴x=−1.388633220m、最大转角约70°；
steering_curve横轴km/h，(0,1)/(20,.9)/(60,.8)/(120,.7)。见results/controller_config.json。
`update(traj_xy, t_frame)` 接收 20×2/4 Hz/5 s 轨迹；`step(t_now, speed_mps, yaw_rate_rps)` 推进历史并返回
`(throttle, steer, brake)`；第一 tick 可先 update 再 step，延迟帧使用之前 step 的历史。
`reset()` 清空跨 route 状态；`diagnostics` 为 JSON-compatible dict，字段至少：
`trajectory_time`, `trajectory_age_s`, `target_speed_mps`, `reference_speed_mps`, `aim_xy`,
`cross_track_m`, `heading_error_rad`, `reason`。无有效数值用 null，不能用 NaN。
longitudinal_mode默认vendor；可选pi使用SI误差与条件积分抗饱和，pi_kp/pi_ki默认1/.25。
v4显式冻结.5/.25，pursuit lookahead=max；CARLA参考配同一PI，TCP参考保留vendor。
PI安全状态清空积分，超过.2s运动间隔触发motion_gap制动。默认CLI仍carla/vendor，不能把候选配置当新默认。

## Agent / runner config

runner 和 route 透传 `--drive controller`、`--controller-preset carla|tcp|pursuit`、`--cruise-mps 8`、
`--tm-seed 0`、`--controller-config <JSON>`。config JSON 中用 `drive`, `controller_preset`, `cruise_mps`,
`controller_config`（参数 JSON 路径）, `out`（attempt 目录）。CLI 的校验和细节由 report 子代理实现。
`policy=none` 的 route 轨迹同步更新，每 decimate tick 一次；控制及运动传感器每 tick 一次。
控制器模式先拒绝尚未支持的真实 policy，不假称已有 planner 轨迹输出。

## control.jsonl

每个已接收运动帧一条，包含异常制动帧；正常帧顶层字段：`frame`, `sim_time`, `speed_mps`, `yaw_rate_rps`, `throttle`, `steer`, `brake`,
`trajectory_time`, `trajectory_age_s`, `target_speed_mps`, `reference_speed_mps`, `aim_xy`, `cross_track_m`,
`heading_error_rad`, `reason`；`sensor_frames` 字典；`pose_xy`, `pose_yaw`（CARLA world yaw）；
`raw_pose_error_m`, `pose_error_m`, `pose_heading_error_rad`（诊断真值 logger 计算）；
`route_cross_track_m`（估计位置相对 dense route）；`truth_cross_track_m`（仅 logger，用真值相对 route）。
主指标优先 `truth_cross_track_m`；缺失时报告缺失并可单列 estimated 指标，不能默认为同一量。
controller diagnostic cross-track 是相对短时参考，不可混同 route cross-track。

官方驾驶结果由 reporter 从 attempt 的 `results.json` 读取。runner finished 仍仅表示 evaluator 返回。
所有异常/缺失/重试均保留，不挑最佳 attempt。


## 原始运动输入与姿态异常

motion.jsonl在姿态验证之前写入frame、sim_time和sensors.{GPS,IMU,SPEED}.{frame,data}；
非有限数值以nan/inf/-inf字符串保存，不输出非标准JSON数值。正常control.jsonl的pose_status记录
reason、degraded、compass_valid、compass_age_s。已初始化且其他运动输入有效时，缺失compass最多.2s，
由陀螺仪预测，跳过绝对航向校正；GNSS杠杆臂使用预测航向，真值不参与。

无有效初始compass、超过期限、GPS/速度/陀螺仪/时间无效时，agent输出throttle=0/steer=0/brake=1，
reason=invalid_pose，保留pose_error与验证失败前的pose_status，pose_xy/pose_yaw为null；
无效输入数值在该诊断帧也用显式字符串保留。随后清空定位/控制历史与轨迹ID，恢复后立即生成新轨迹，
不等四tick相位，也不沿用异常前的控制积分。报告将缺失真值或指标标为缺失，不能填零。

## 结果边界

finished、driving_completed（路线到100%）、官方记录语义的子集SR和各G2 gate分别报告。
DS不直接包含加速度/jerk，官方Smoothness另算；物理舒适性诊断保存完整运动学、采样时间和滤波规则。
控制器、原始route oracle与真实TCP模型不是同一个实验入口；原生4点TCP接入见agents/tcp-controller-integration-plan.md。
