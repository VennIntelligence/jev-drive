# 后轴侧向传播补偿：独立验收分析

遵循上一级 `protocol.md`；两臂为 `baseline-zero`、`candidate-fixed-k`，共同 linear aim、max(3 m,.5 s×speed)、PI .5/.25。固定四个旧几何窗口，保留完整路线、停止、失效和缺帧。与旧 Hermite 分析源及结果分目录保存；没有修改旧冻结文件。

```bash
/data/envs/carla/bin/python todos/2026-09-23-lateral-followup/pose-followup/analysis/analyze_pose.py \
  --run-root /absolute/closed-six-cases --out /new/analysis-edition \
  --protocol todos/2026-09-23-lateral-followup/pose-followup/protocol.md
```

每例需要 `control.jsonl`、`motion.jsonl`、`validation_trace.json`（JSON数组）、`validation.json`、`route_reference.json`、`agent_config.json` 及其指向的实际 controller config。每个输入和分析依赖记录 SHA256。拒绝已有输出目录。缺例默认在创建输出前退出；`--allow-incomplete`只用于中间检查，不能让缺例通过。

性能条件恢复原转弯协议：26966 主窗口 CTE RMS 至少降低15%，P95及post10m RMS不升；四窗CTE/heading/速度/实际横向加速度/emitted steer-rate守护不变。两个S分别检查。没有 Hermite 20%动作收益、jerk收益或新增恢复门槛。恢复censor与physical jerk完整输出，但仅诊断。缺失、样本不足20或比例基线P95≤.01标证据不足，不能当通过。

保留每例12个G2门槛、非法输入/输出、未知reason、完整control/truth帧和linear aim schema核验。GameTime与独立world clock允许每例恒定offset，但逐帧对应/时间增量必须一致。合法`trajectory_behind`安全制动逐帧索引、保留站距和终点距离，不当非法输出；`invalid_lateral_prediction`等真实输入/pose故障判失败。

新增补偿契约不使用真值：直接从每帧原始 SPEED/IMU 重算死区后的速度、world gyro、前后均值、dt、中点方向与 `-k*v²*omega*dt*right`；检查首帧/reset/fault的null字段，k=0有效区间的disabled/零位移，以及固定k有效区间的applied。逐帧验证补偿先进入预测XY、再经`.05` GNSS融合。原始motion与control必须逐帧同时间、同GPS/IMU/SPEED帧。缺字段、错误系数、错符号、错误均值或错误融合顺序均不通过。

另从记录的GPS投影参数、原始GPS/compass/SPEED/gyro **独立重建全序列姿态**，状态只由该重建自身传播，不拿记录的pose或truth作为反馈；逐帧核对raw/fused XY与yaw，容差1e-8。投影器/lever-arm/.05和.1融合参数来自已归档参考设置。保留短compass dropout规则，失效重置。生产滤波器和CARLA均不被导入。`pose-recomputed.csv`保存预期位移、原传播预测、独立全序列pose和差值；`pose-contract.json`保存全部错误位置。

`frames.csv`新增定位误差字段：position norm、raw position norm、heading、世界XY有符号偏差、**按真值车身heading的left/forward偏差**。body-left不是reference-left，也不是原运动模型残差。truth只用于这些独立评分量。`metrics.csv`提供RMS/P90/P95/max、均值、缺失计数；四窗entry/core/exit/post10m之外，还给每路full_route、endpoint_last5m、terminal_hold、固定straight_guard。17563的approach与gap分别保存。左弯定位退化须显式报告，不新增逐窗pose非恶化门槛。

physical jerk采用世界加速度先差分、再投当前right；只对连续帧按实际dt计算，窗口首帧使用全路前一帧。`applied_control`不与同帧新命令混为执行器误差。CTE仍reference左正，26966右弯正值为外侧。

验证入口：`test_pose_analysis.py`覆盖字段、系数、符号、平均/时序、融合顺序、输入帧、truth隔离、时钟offset及性能门槛；`verify_analysis.py`产生历史双别名、删帧与缺schema的失败fixture；`verify_sensor_replay.py`从旧六例全传感器重建k=0姿态并保存输入哈希。`--legacy-fixture`严格是metrics-only测试开关，输出`live_qualification_allowed=false`，不放宽真实两臂的系数/姿态契约。它不表示相同旧物理轨迹可以同时满足不同k的生产pose。

最终分析包含156条必要条件，其中新增schema/系数/全传感器重建/共同配置属于契约完整性检查，不是新增性能目标。15项单元验证通过；旧六例2761帧k=0独立重建的最大位置差为0。另用当前生产PoseFilter生成两系数、三路线共2760帧的离线pose/status正例，独立重建最大差为0；每组单删`lateral_dt_s`均被拒绝。该正例保留历史控制占位，只用于pose契约验证，绝不称为固定k的闭环控制结果。

大逐帧验证输出保留`/data/runs/b2d/controller/lateral-followup/pose-analysis-verification-v1`及最终`pose-analysis-verification-v2`两版。最终metrics-only双别名只有26966的15%主收益不满足；删帧同时触发case/coverage失败。旧日志缺新schema的live路径也被拒绝，单个字段故障另由正例验证隔离。`verification-v1`保存小型验证摘要与全输入哈希。这里的验证不启动CARLA、不构成新闭环结果；真实六例与条件性反序确认由根代理按已冻结协议执行。
