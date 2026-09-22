# 离线分析入口与字段约定

本目录的[analyze_tcp.py](analyze_tcp.py)只依赖Python标准库，不导入模型、PyTorch或CARLA，也不启动GPU。冻结protocol.md保持不变；本文件说明日志绑定和辅助脚本的验证。

```bash
python3 todos/2026-09-23-tcp-controller/results/analyze_tcp.py \
  --case 24211:native_common=/absolute/closed/native/attempt \
  --case 24211:pi_common=/absolute/closed/pi/attempt \
  --out todos/2026-09-23-tcp-controller/results/analysis-edition-001
```

可按案例完成逐渐增加--case，但每次必须使用新edition目录。仅接受已有最终attempt.json状态的closed attempt；未结束的源日志不做正式hash分析。每个route/arm显式指定一条已选择的attempt，不自动找“最好的一次”；全部失败和重试仍由campaign报告保留。六例未齐或缺行为数据，不做整组优胜判断。

每例输出JSON与逐帧CSV；comparison.json保存原官方record、attempt、完整指标及配对C−B差值，analysis-source.py保存实际分析代码。文件缺失/解析错误/真值错帧/非20Hz连续间隔/来源变化均显式记录。source_sha256保存分析所读的闭合文件身份。

## 实际字段映射

- 当前schema的native_pid_calls位于`prediction`内；normal=1，非有限预测/target/speed故障=0。首帧neutral且prediction为空允许没有desired，计missing覆盖率，不补0。
- `prediction.raw_waypoints`是交换坐标前的forward/right四点；相邻三段距离/.5复算desired，排除origin桥。`prediction.metadata.desired_speed`为控制器使用的值。
- `truth.velocity`和`truth.acceleration`为同frame世界向量；`rotation_deg`依次roll/pitch/yaw。速度主误差使用同帧车头forward投影值；native输入速度与3D速度另保留。
- 主jerk=`(a_world[t]-a_world[t-1])/dt`后点乘当前forward/right。`body_accel_lon_derivative_mps3`仅辅助展示“先投影再差分”的不同量，不替代主jerk。
- forward/right公式与[CARLA0.9.15 Math.cpp](https://github.com/carla-simulator/carla/blob/0.9.15/LibCarla/source/carla/geom/Math.cpp)一致，包含roll/pitch，不只按平面yaw近似。正lateral为CARLA右侧。
- `selected_control`顺序throttle/steer/brake。与comparison选中分支、native steer、上一tick已发命令对应的truth.applied_control分别核对；API pedal一致不证明力瞬时响应。
- `criterion_events.json`的事件frame界定严格碰撞前prefix；若官方碰撞存在但事件frame缺失，不把全程当无碰撞prefix。
- 低速≥5s用实际观测跨度last_time−first_time，20Hz连续样本至少101帧。sample_bin_occupancy_s另存，但不得将100帧4.95s跨度标作5s。时间/帧gap不跨段差分或拼接低速区间。

## 质量与验证

[analysis-tests-v3/tests.log](analysis-tests-v3/tests.log)9项通过；原v1/v2测试日志保留。测试涵盖世界固定加速度在旋转车身上的主jerk仍为0、正确body投影、缺帧不求导、desired排除origin、missing非0、pitch/roll坐标正交、prediction内PID调用计数、5秒观测跨度边界。

统计主表无滤波、无warmup删除；moving与预测stop、首次碰撞前均另列。期望速度变化率描述模型重规划，不是直接可执行的期望车辆加速度。尚未验证物理原点，不伪造模型路径的绝对CTE。角速度原值仍在raw，本文脚本不自动计算有版本疑点的官方Driving Smoothness。

## v1发现后报告补充（冻结协议外）

逐帧保留所有raw负速度、prediction fault、comparison reason、forward/PID调用次数。required sensor为GPS/IMU/SPEED/前方三RGB：统计每路缺失/错帧，optional异步bev不计错帧。模型没有独立source-frame字段，仅同run_step记录与一次forward/captured PID的链路证据；RGB hash记录输入身份而非独立时序证明。

control差保留原数值，native steer与selected branch以1e-7作数值容差；selected与official尾部的油门/刹车变化另列，它们可能是协议的有意公共envelope变化，不归为舍入。`cases.csv`包含每例状态、帧数、故障和安全计数，不能用配对平均掩盖某例失败。route_equal_arm_means在缺例时仍明确标未完整配对，不作主结论。

paired-v1为接入故障发现过程，paired-v2重新完整六例；不拼接。每版manifest保存所读raw、当次脚本字节与输出hash，旧版不覆盖。
