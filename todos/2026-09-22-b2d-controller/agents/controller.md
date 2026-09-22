# Controller 子任务工作记录

状态：T2 / 离线 T3 已实现；实车 plant 和 G2 由主代理独立验证。未修改第三方目录，未启动 server，未做 git 操作。

## 文件和接口

- `scripts/b2d_controller.py`：Python 3.8 / NumPy + 标准库；`Controller.update/step/reset/diagnostics`；carla、tcp、pursuit。
- `scripts/test_b2d_controller.py`：12 项契约测试，包含从锁定本机 CARLA/TCP 文件 AST 提取标量方法运行的等价测试。
- `scripts/b2d_controller_selftest.py`：独立解析圆弧、正弦 S 弯、定时停车参考；合成自行车 + 油门/刹车/阻力和静摩擦。
- `results/offline-selftest.json`：完整的 baseline、速度窗口、延迟、扰动和饱和行，保留所有失败。

接口补充：`lookahead=additive|max|fixed4`，`speed_window=near|reference`；可配置 dt、trajectory_dt、stale_timeout、history_seconds、公共执行限幅与 steer_rate。
正常 update 不重置 PID。重复/倒序参考保留旧参考但暴露 update_rejection；无效/未来/超历史参考制动。`update()` 入队，`step()` 推进当前运动后解析源时刻 pose，保证同 tick update→step 与迟到参考都可用。

stock Lincoln 轴距 2.8604714913890885 m、最大转角 69.99999237060547°、steering_curve 使用主代理实测默认；常数出处写在模块 docstring。曲线 x 为 km/h，主代理的独立响应试验已验证。yaw_rate 输入左正，steer 输出 CARLA 右正。

## 验证命令

```sh
/data/envs/carla/bin/python -m unittest discover -s scripts -p test_b2d_controller.py -v
/data/envs/carla/bin/python scripts/b2d_controller.py --selftest --output todos/2026-09-22-b2d-controller/results/offline-selftest.json
```

12/12 通过。`--selftest` 退出 0 表示候选主测试通过，不意味着基线全部通过或实车通过。
测试包括 0/1/2/6 tick 延迟和插值源帧的 SE(2) 精度、镜像和输出方向、PID 启动/恒值/阶跃/换符号、NaN/Inf/空/全零/重复点、timestamp/reset 和 1200 tick 随机控制不变量。

## 当前合成结果

| preset | R20、6m/s 左/右圆稳态 RMS | 主圆弧门槛 |
|---|---:|---|
| carla | 0.4671 m | 失败，原增益保留 |
| tcp | 0.1546 m | 通过 |
| pursuit | 0.00906 m | 通过 |

- pursuit near 停车：停点误差 -0.0284 m；整个减速段速度 RMS 0.1712 m/s；减速+后续全段 RMS 0.1059 m/s；命令速度 RMS 0.1438 m/s。
- 计划 t=6s 停车时速度仍 0.1516 m/s；实际静止 t=6.45s；从真正静止开始 5s 最大速度和位移均 0。此处明确报告计划停时与真实停时，不能声称 t=6s 已静止。
- 修复了停车参考首段由 ego 原点连接近处 stationary endpoint 导致反复产生小速度命令的爬行：近处终点、低速条件满足后 latch brake hold；新移动参考或显著移动的终点释放。所有 preset 共用，未改 baseline PID 增益。额外单测覆盖释放与 reset。
- 原 reference 速度窗口导致 CARLA/pursuit 约 5.09m 提前停车，失败完整保留；near 显著改善。
- 2/6/8/12/14m/s 与 R20/R40 中预先定义 v²/R≤5m/s² 的可行组合：0.30s 延迟最大 RMS 0.0342m，所有 delay RMS≤0.3m、增量≤0.1m。
- ±0.5m / ±5° 初始偏差、轴距/转向 ±10%、0.15s 转向延迟全部保留，当前矩阵最大 RMS 0.098m。
- 解析 S 弯零延迟 RMS 0.0271m，0.30s 延迟 0.0517m。
- 此次主圆弧 step p99 最大约 0.09ms（会随机器负载变化，不替代实车 telemetry）。

## 边界

plant 力学是明确标记的 synthetic：油门系数3、刹车系数8、阻力0.08，不能当实测。无 tire slip。运动传感器在主 selftest 使用准确区间平均速度/yaw rate，隔离重投影代数；实车端点速度采样和传感器误差需要 G2 验证。GNSS 噪声/融合在 agent 子任务独立覆盖，不伪造实测噪声参数。

坡道仅单位级静摩擦演示：有足够 brake 保持，去掉 brake 会负向运动；因此不依赖 v=max(v,0) 假定。然而这不是 CARLA 坡度停车验收。

补充：diagnostics 返回脱离内部状态的只读语义快照，测试证明调用方改 reason/aim list 不影响控制器。results 新增 additive/max 两条预先声明 lookahead 的 R20、6/8m/s 对照；默认保持 additive，未据此调参。


## G2 validator 二次任务

主代理授权后接管并修复 `scripts/b2d_controller_validate.py`，已交回：

- 独立 true pose / route progress 投影，完全不读取 estimated route.progress；另报 segment Euclidean distance，避免 normal distance 掩盖端点方向错误。真值后轴使用含 pitch 的 forward vector。
- full / steady lateral 分开，G2 lateral gate 用 full；恒速指标固定前5s起步豁免，基于真值剩余距离的制动参考，另报全部 cruise/reference/command 误差。
- None target、无 telemetry、无 heading、错位 frame 都显式失败，不抛 TypeError；raw/fused pose、heading 分布、sample count 均有。
- 停车保持记录5s真实后轴的最大两点距离，gate<=.1m；速度用3D真实范数，因此不能在侧滑/倒溜时误判停住。
- 异常 case、map setup失败、cleanup失败与半写入 telemetry 保存后继续，取消保留当前结果。config只读取一次；同地图复用 world，同次任务共用 server。
- 15/15离线测试通过（原12 + G2三项 fixture）：验证漂移.15m即使speed<.1仍失败、缺heading失败、错frame失败、None速度安全、空结果不冒充成功。

主代理运行CARLA；此子任务未启动任何server。原 handoff feedforward+bearing PID 消融暂缓，优先完成实际阻塞的 G2 修正。

G2 修正交回之后补完原 handoff 横向公式离线消融（生产 Controller 未更改）：R20 / 6m/s，左右镜像，同样 max lookahead + reference speed、相同历史和投影，pursuit RMS 0.00852m，pursuit + 完整 CARLA bearing PID 为 0.12843m；速度 RMS 两者均0.04445m/s。额外 PID 令此例误差约15倍，但两者都低于0.2m，不能说此例已不稳定。`offline-selftest.json.original_bearing_pid_ablation` 保存公式、参数、限定和所有行。消融精确复现原横向相加公式，统一使用修正后的时间/投影，不冒称完整复现原文未指定的 waypoint-index 细节。
