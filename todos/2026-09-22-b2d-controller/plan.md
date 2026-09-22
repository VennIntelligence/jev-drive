# B2D trajectory controller：实现、验证与验收方案

状态: running，2026-09-22；用户已授权按本方案实践，由多个子代理协作。
主题: [trajectory-to-control](../../research/trajectory-to-control.md)
原始交接: `tmp/2026-09-22-handoff-b2d-controller.md`

## 目标

把 planner（输出未来轨迹的规划模型）与 CARLA 的执行连接起来：轨迹以 5 Hz 更新，控制以 20 Hz 更新，
明确坐标、时间、反馈误差和失败处理。先证明合理轨迹可以被稳定执行，再评估它对路线完成率和运行成本的影响。
固定控制器、不训练控制器的方向保留；原 research 的具体公式和验收口径需要先修正，不能当成已验证结论。

本轮交付覆盖原 handoff 的 A–C：独立控制器、route trajectory adapter（把路线变成定时轨迹的适配器）、
完整 Dev10 对照、报告和默认配置。接真实 planner 是随后独立的一步；只有固定 planner 的闭环对照才能说明分数提高。

## Setup

| 项 | 已核实状态 / 执行约束 |
|---|---|
| 本仓库 | 当前分支 `tokyo-headed-smoke`，HEAD `5ec5915`；已有大量未提交的 smoke、TCP 和 runtime 改动，先保留并审阅，不能照旧 handoff 的“五个文件”盲目提交 |
| 实际评测 checkout | `/data/third_party/Bench2Drive`，`0.0.4`，`7ec25d1c9f7522d923ce5f3420986cef1cb2d956`，本次查看工作树干净 |
| 另一份 checkout | `~/mycode/Bench2Drive` 是 `main`，`21d85ee…`，有自己的改动；不用它评测，也无需切它的分支 |
| CARLA | `/data/third_party/carla/CARLA_0.9.15`；每次实验记录 client/server version、地图和车辆型号 |
| Python | `/data/envs/carla/bin/python`，3.8.20；controller 仅 NumPy 和标准库，不导入 CARLA/Torch |
| TCP 参考 | `/data/third_party/Bench2DriveZoo`，`8a08b07883f10b7d83f6bf5dd475bda91a91c50a` 的 `TCP/model.py`、`TCP/config.py` |
| 硬件 | Tokyo，仅物理 GPU 1；`CUDA_VISIBLE_DEVICES=1`，CARLA `--gpu-rank 0`，启动后用 UUID 核验；一个 worker |
| 长任务 | tmux `jev`，独立端口和 run dir，保留 `log.txt`、`events.jsonl`；不改第三方源码 |
| 实验协议 | 固定 Epic、0.05 s timestep、车辆、路线、天气、seed、rig、窗口/离屏模式；正式对照不用人为 tick cap |

现有 `drive_runtime/sensors.py` 可以借用按 frame 收集数据的思路，但它不是现成的双频率控制调度器。
现有 TCP adapter 的预处理和 preview（实时预览）继续复用现有路径；本任务只扩展自身 controller agent 所需的接口。

## 探索得到的修正

| 原建议 / 当前代码 | 问题与证据 | 本方案处理 |
|---|---|---|
| `max(3, 0.5v)` 来自 CARLA 默认 | 实际 `local_planner.py` 是 `3 + 0.5v`，8 m/s 时分别为 4 m、7 m | 两种规则显式命名；CARLA 参考采用加法；候选规则在开发集比较，不再误标出处 |
| pure pursuit + 对 aim bearing 的完整 PID | pure pursuit（根据前方目标点求圆弧曲率的几何跟踪法）已经产生转向；车辆即使贴着圆弧，目标点夹角仍非零，直接相加会重复转向 | CARLA/TCP PID 单独作为对照；新增 pure pursuit 单独作为几何候选。原相加式只做离线消融，不预设为默认 |
| aim point 的 y 是横向误差 | 半径 20 m、沿弧 3 m 的目标点，在完美跟踪时 y = 0.2246 m；会直接违反原 0.2 m selftest | 主指标改为后轴到参考路径的有符号法向距离；aim_xy、bearing 仅诊断 |
| 重投影自动修正目标速度的延迟 | 刚体旋转和平移不改变相邻点距离。固定取旧轨迹第 1–4 点，速度仍是旧时间段的平均 | 几何重投影与时间裁剪分开做，按 `age=t_now-t_frame` 查询剩余轨迹的时间区间 |
| 0.25–1 s 平均速度代表现在速度 | 8 m/s 在 3 s 内匀减速，前述平均速度比当前速度低约 1.67 m/s；控制器追上自身命令仍可能提前停 | 分开报告命令速度误差与原轨迹的定时速度误差；近时段速度读法做受控消融 |
| 两套 PID 只换增益和窗口 | CARLA 用 `sum(error)*dt`、差分 `/dt`；TCP 用零填充窗的 `mean(error)`、不除 dt 的差分 | 两种离散语义分别实现并与来源函数比对；TCP 的轨迹适配版不冒称完整 TCP 复现 |
| 当前 decimate + overlap 可直接接控制器 | `StubAgent.__call__` 只在 policy tick 读传感器，其他 tick hold control；线程内才取 GameTime；收集器可能混用不同帧 | 主线程每 tick 收当前 SPEED/IMU/GNSS 并 step；后台只产带源帧时间的轨迹，不更新 PID 或控制状态 |
| runner `finished` 表示车跑完 | `b2d_route.py` 的含义是 evaluator 正常返回；现有 smoke 中就有 `Failed - TickRuntime` | 从 `results.json` 的 `_checkpoint.records` 读真实 completion/status；独立报告进程完成、驾驶完成和截断 |
| MinimumSpeed 事件数必须为零 | 固定版本的 criterion 定期写事件；`statistics_manager.py` 把该扣分项设为 `[0.7, 'unused']` | 报事件数、percentage 和实际 penalty；不能将事件数或该 criterion 的 FAILURE 直接解释成爬行失败 |
| Dev10 卡住说明 lookahead 不够 | Dev10 有 `ConstructionObstacleTwoWays`、`BlockedIntersection`、`HazardAtSideLane` 等；中心线轨迹可能穿过障碍物 | 先隔离控制能力，再运行原场景；保留全部失败，按证据归因，不自动换 PID |
| 不读 hero transform 就可以声称跑榜合规 | dense route 本身是诊断 oracle（额外提供的参考路径）；传感器频率 monkeypatch、地图访问也有协议边界 | route 实验标为控制诊断；正式 planner 只接允许的输入，届时单独核对提交协议 |

圆弧例子还有一个直接推论：aim bearing = 0.075 rad，仅 CARLA 的 P 项便额外给出 0.14625 steer。
这是解析计算，不是 CARLA 实测；足以说明不能把这项叫作“只修小残差”。

上述判断主要来自本机锁定版本源码：本仓库 `b2d_agent.py`、`b2d_route.py`；CARLA
`agents/navigation/{controller,local_planner}.py`；Bench2Drive 的 `atomic_criteria.py`、
`statistics_manager.py`、`route_manipulation.py`、`sensor_interface.py`；TCP 的 model/config。
车辆 API 单位以 [CARLA 0.9.15 API](https://carla.readthedocs.io/en/0.9.15/python_api/#carla.VehiclePhysicsControl)
为参考，并通过独立标定核验。官方对传感器和特权输入的边界见
[Leaderboard 2.0 evaluation](https://leaderboard.carla.org/evaluation_v2_0/)；本地 route 诊断不据此宣称具备提交资格。

## 推荐实现

### 1. 冻结接口和数据流

保留 handoff 接口：`update(traj_xy, t_frame)` 与 `step(t_now, speed_mps, yaw_rate_rps)`，返回
`(throttle, steer, brake)`。另有 `reset()` 和只读 diagnostics（用于日志的内部诊断快照）。
输入固定为后轴原点、x 前 y 左、20×2、首点 +0.25 s、末点 +5 s；首帧原点作为 t=0 的辅助点保留。
输出 steer 明确采用 CARLA 的正方向；坐标反射只在边界做一次，并通过左右转测试证明。

每 tick 的顺序固定为：收集本帧运动传感器 → 推进运动历史 → 接收新 trajectory → controller step →
写 telemetry（逐 tick 记录）→ 返回 VehicleControl。`update` 只替换带时间的参考，不清空 PID 历史；
每条 route 重建或 reset 全部历史。第一帧没有有效轨迹时停车。

controller 在没有 trajectory 时也记录运动历史，至少覆盖 2 s。延迟到达的 update 使用源帧到当前帧的
完整历史进行 SE(2) 重投影（二维刚体位姿变换），不能把接收时刻当作起点。相同/倒序 trajectory、
超出历史的旧帧、未来时间戳、非有限值、退化点、只剩车后方路径都要有明确处理及 reason code。
初始 stale timeout 定为 0.5 s：超过便退出正常跟踪并制动，记录故障；该值作为配置写进 manifest。

5 Hz 是轨迹更新频率，20 Hz 是运动传感器和控制频率。相机帧与运动帧分开路由：推理需要的相机必须
来自同一个已完成帧，不能在非相机 tick 等一个不会到达的图像，也不能读运动数据时丢掉待消费的相机包。
源 frame ID 和仿真时间在主线程采样时冻结。`policy=none` 的 route adapter 同步出轨迹；人为延迟在测试中
按 tick 显式注入，不能用空后台线程的偶然调度模拟 planner 延迟。

### 2. 控制器对照与候选

| preset | 横向 | 纵向 | 用途 |
|---|---|---|---|
| `carla` | 角度 PID 1.95/0.05/0.2，窗 10，rad，lookahead `3+0.5v` | 1.0/0.05/0，km/h，CARLA 离散语义 | 清楚定义的 CARLA 风格参考；max brake 提到 1 |
| `tcp` | 0.75/0.75/0.3，窗 40，bearing/(π/2)，固定 4 m | 5/0.5/1，窗 40，delta clip [0,0.25]，超速比例 1.1 或目标速度 <0.4 时布尔制动 | 20×2 统一输入上的 TCP 风格参考；不含官方 control branch 混合、target fallback |
| `pursuit` | `κ=2y/(x²+y²)`，`δ=atan(Lκ)`，用测得的转向映射转成命令 | 先与 `carla` 相同 | 推荐优先验证的 v1 候选；不额外叠加 aim-bearing PID |

公共执行限制先用 throttle≤0.75、brake≤1、|steer|≤0.8、每 0.05 s 的 Δsteer≤0.1，油门和制动互斥。
因此上表是统一执行边界下的适配对照；尤其 TCP 的公共限幅与原实现不同，必须写进配置，不能标为逐控制量完全复现。

先固定同一种目标速度读法和定位，比较横向方法；再固定横向方法比较速度读法，避免一次改变所有变量。
保留原 0.25–1.0 s 窗口作为 reference，时间坐标改为 `[age+0.25, age+1.0]`。
候选取 `[age, age+0.25]` 的弧长差除以 0.25 s，缩短刹车预见偏差；窗口必须按轨迹绝对时间插值。
轨迹尾部只在明确停车尾段允许常位置延拓；其他越界触发 stale/invalid，不能偷偷生成继续巡航轨迹。
两种速度读法都需要独立 stop-position 验证，不只看命令误差。

lookahead 从当前后轴在剩余路径的局部投影位置起算弧长，不能从旧的第一个 waypoint 起算。
首轮只比较 `max(3,0.5v)` 和 `3+0.5v` 两个预先列出的规则。若均不足，再试 research 列出的 PDM-Lite
速度 lookahead 与对应 PID，先核对其离散语义；不开展无界调参。若后续确需 feedforward + feedback，
feedback 必须作用于横向偏差和路径切线 heading error，重新验证，不能沿用原相加式。

### 3. 车辆与定位边界

标定脚本保存完整 physics JSON、前后轮位置、车辆 transform、地图、server version 和 GPU UUID。
从轮位置推轴距和后轴相对 actor 原点的偏移，核对位置的坐标系与单位；记录左右前轮最大转角、
`steering_curve` 横轴单位、转向随速度的响应。用数个速度/转向阶跃检查实际 yaw rate 对 `v*tan(δ)/L` 的偏差。
三个常数不足以证明真实 plant（被控车辆）等同于理想自行车；转向执行延迟、轮胎侧偏和油门/刹车响应仍需测。

route adapter 使用 GNSS（卫星定位）、IMU（惯性测量）的 compass/gyro 和 speedometer。
建立同一局部坐标系下的 route 与 GNSS，使用锁定版本 Mercator 变换；诊断路线如使用 GPS/world 对求原点，
明确记录来源。正式路径不调用 world map/OpenDRIVE 取得额外信息。
compass→yaw、gyro 正负、CARLA y 右→轨迹 y 左、GNSS 安装点 x=-1.4 与实测后轴的差都写进契约测试。
用运动预测加 GNSS 修正的轻量滤波稳定定位，参数在独立标定上固定，不能直接把每帧 GNSS 当准确位置。

hero 真值仅进入独立评测记录；统一到后轴和同一时刻后再比较误差。正常 controller 模式关闭真值 logger
仍应可运行；同一段传感器重放时，真值 logger 开关或更换真值内容不得影响输出。
控制器无法处理的碰撞后大侧滑等状态要在报告中暴露，不能用真值重置估计来隐藏问题。

### 4. route 轨迹与终点

保留 dense route；局部投影搜索结合单调弧长进度、heading 和有界前向搜索，覆盖交叉、自近邻和发卡弯，
不只保留“下一个点更近”的贪心索引。每 4 tick 按 cruise_mps=8 生成 20 点、5 s 的轨迹。
原路线若有初始横向错位，不强行把首个参考路径点拉到 ego；单独处理 t=0 后轴原点与参考路径的关系。

终点处理分清 actor 与后轴：停止目标必须与 evaluator 的完成区相容，不能在最后 1 m 提前刹停导致永远差一点完成。
独立停车测试按指定 stop position 验收，route 终点按官方 completion 验收。固定 8 m/s route 只提供跟踪测试输入，
不包含绕障、让行、信号灯策略，也不代表 8 m/s 对所有曲率都物理可行。

## 步骤与交付

| 任务 | 依赖 | 需要改动 / 产物 | 完成条件 |
|---|---|---|---|
| T0 整理基线与修订契约 | 无 | 审阅已有改动，分批保存已完成工作；固定源码/环境/config hash；把本方案中的已确认修正同步到 research | 能明确复现之前的 smoke/runtime；没有误收其他工作或改第三方 checkout |
| T1 标定车辆、坐标和传感器 | T0 | `scripts/b2d_calibrate.py`，physics.json，定位/转向响应摘要 | 常数、单位、后轴偏移、正负号有实测证据；记录无法建模的误差 |
| T2 纯 NumPy controller | T0；实车默认值依赖 T1 | `scripts/b2d_controller.py`，含三个 preset、reset、history、diagnostics、`--selftest` | G1 通过；参考 preset 失败可保留原值，但不得选为合格默认 |
| T3 独立控制验证工具 | T1、T2 | 合成轨迹/plant、延迟注入、`scripts/test_b2d_controller.py`；自有 CARLA 空场景验证脚本 | 无同源误差指标自证；G2 通过 |
| T4 双频率 agent 与 route adapter | T2 | `b2d_agent.py`；必要的小型 pose/route 辅助模块；sensor frame 路由 | 运动传感器和 step 每 tick 更新；replan 每 4 tick；无特权控制输入；reset/cleanup 正常 |
| T5 CLI、日志和报告 | T4 | `b2d_run.py`、`b2d_route.py`、`b2d_report.py`；`--drive controller --controller-preset … --cruise-mps … --tm-seed …` 端到端透传 | attempt 内有可重算 telemetry；report 对 finished/failed/capped/missing 均正确 |
| T6 一条路线 smoke 与协议核验 | T3–T5 | 固定版本的一条无 cap 完整路线；另做短延迟/传感器测试 | G3 通过；既有 runtime/sensor/preview 测试无回归 |
| T7 Dev10 与默认选择 | T6 | carla/tcp 完整两轮；合格 pursuit 再一轮；冻结后重复与保留路线确认 | G4 全覆盖报告，按预定规则选默认或明确失败，不凭单条成功宣布完成 |
| T8 结论和复现材料 | T7 | todo 实测、research 修正、成本模型、默认配置、小结果文件；后续 planner 任务边界 | G5 材料齐全，分批 commit/push 的 hash 可追溯 |

T2 的合成测试开发可以先于车辆标定，临时参数须明确标为 synthetic；不能把假定轴距写成实测默认值。
T0 的代码归档是实施阶段任务。实施前先保存此前 smoke/runtime 基线，随后分批提交本轮工作。

## 验证与成功标准

以下是跑之前确定的验收门槛，不是已取得的成绩。RMS 为均方根误差，p95 为第 95 百分位。
最终默认必须通过适用的门槛；参考 preset 不通过仍完整保留其结果，不为实现“两套都过”偷偷改 baseline。

### G1：离线数学、状态机与延迟

| 测试 | 验收 |
|---|---|
| 坐标/时间 | 四个 cardinal heading、左右镜像、后轴偏移、GNSS round trip、时间戳转换；确定性转换误差 <1e-6 m/rad（地理转换数值误差 <1e-3 m） |
| PID 语义 | 固定 error 序列含启动、常值、阶跃和换符号；与提取的 CARLA/TCP 标量算法分别一致，浮点容差 1e-6 |
| 重投影 | 匀速直线与常 yaw-rate 的解析位姿，0/1/2/6 tick 延迟；误差 <1e-6，含 update 在数个 tick 后才到达 |
| 状态与边界 | 空轨迹、重复点、全零停车、NaN/Inf、乱序/未来/stale、时间回退、route reset；不产非有限控制，不跨 route 泄漏历史 |
| 调度 | 可控假传感器/假 policy；每有效运动 tick 恰好 step 一次，4 tick 一次 replan；无跨帧拼图、无丢包导致的永久等待 |
| 单位/控制输出 | 左右转物理方向正确、曲线横轴单位正确、throttle/brake 互斥、限幅和限速成立 |

无噪声理想自行车做主 selftest：半径 20 m 左/右圆弧、6 m/s；直行先保持 8 m/s，再于 3 s 内停下，
停车后保持 5 s。横向评价用独立解析曲线，不能用 controller 自己裁剪/重投影后的轨迹作唯一 ground truth。
圆弧稳态剔除预先固定的前 2 s，但同时报告全程；减速段不丢弃任何 warmup。
验收：横向 RMS <0.2 m、定时参考速度 RMS <0.5 m/s；停点误差 ≤1 m，停止后 |v|<0.1 m/s、位移≤0.1 m。
plant 需包含明确的油门/制动到加速度映射，不能直接把 v_des 填进车速；倒溜验证不能依赖 `v=max(v,0)` 的实现强行通过。

再做小规模鲁棒性矩阵：速度 2/6/8/12/14 m/s，半径 20/40 m 的物理可行组合，S 弯、变速、
初始横向 ±0.5 m / heading ±5°，0–0.30 s 延迟，已测 GNSS 噪声，转角执行延迟和轴距/转向增益 ±10%。
明显不可行的速度/曲率单列为饱和测试，不混入精度平均值。0.30 s 延迟时可行用例 RMS≤0.3 m；
相对零延迟的横向 RMS 增量≤0.1 m。失败则先定位时间/控制问题。

### G2：CARLA 无交互控制验收

使用自有脚本在独立 server 中构造没有交通场景的直线、左右转、S 弯和停车试验，不改官方 Dev10 XML。
在这些开发场景完成有限参数选择，再冻结配置。只有开发场景调参，不根据 Dev10 单条路线特判。

默认候选须完成全部开发路线；每条后轴 cross-track（后轴到路径法向距离）RMS≤0.5 m、|error| p95≤1 m，
无驶离参考路线或控制导致的卡死。恒速段速度 RMS≤0.5 m/s；停车误差≤1 m，停车后保持 5 s 不倒溜。
另跑存在实际坡度的停车保持测试，报告坡度和停车位移；其结论不由理想模型替代。

GNSS 原始误差、融合后误差分别报中位/p90/p95；后轴定位 p90≤0.5 m、heading p90≤1° 作为初始目标。
未达到则先修定位，不能把定位误差当横向控制器缺陷调增益。真值 logger 关闭、开启和篡改输入的传感器重放
控制输出应一致。冻结条件下 controller.step p99<1 ms（CPU 时间单测），确认没有引入明显的每 tick 开销。

### G3：agent/harness 集成验收

一条完整 smoke 路线无 tick cap，官方 completion=100%；agent、传感器、日志都无异常退出。
一个有效控制 tick 对应一条 telemetry，frame ID 单调；命令、轨迹源帧、age、传感器帧可重建。
测试异常 policy、超时和 SIGTERM：错误能显式终止或触发定义的 stale 状态；取消保留部分日志和正确状态。
既有 `test_b2d_runtime.py`、`test_b2d_sensor_pipeline.py`、`test_b2d_preview.py` 全部通过。
失败/半写入/重试 fixture（预制报告输入）证明汇总不会把进程退出 0 当作 completion=100。

### G4：原版 Dev10 对照与验收

先用同一 seed=0 跑 carla/tcp 各 10 条；pursuit 通过 G1–G3 后增加一轮 10 条。
三种 preset 使用同一定位、相机协议和最终选定的速度读法，比较的是适配后控制器。
至少对选出的默认与最强参考追加 seed=1 的完整配对复跑；保存所有可控随机种子，不能声称 CARLA 完全确定。
Dev10 只有 10 条，均值改善但失败增多时不选默认。

| 层面 | 验收 / 决策规则 |
|---|---|
| 数据完整性 | 每个 preset 的全部 10 条均有最终结果或明确基础设施失败；缺失不能当成功或静默排除；未补齐则评测未完成 |
| 控制能力 | G1–G3 已通过；Dev10 每条报告全程和固定规则截取的首次碰撞前指标，不能只展示表现好的片段 |
| 实际驾驶 | 全部 route completion、blocked、deviation、collision、timeout 保留；不要求无避障 planner 的中心线 driver 通过所有交互场景 |
| 卡住归因 | 原始轨迹/估计位姿/真值/事件时间能证明跟丢或路径穿障碍；无法判定标 unknown。任何失败仍计入总结果 |
| 候选替代默认 | 两个 seed 的 route 平均 completion 分别不低于最强参考；控制导致的 blocked/deviation 不增加；横向路线均值 RMS 至少降低 20% 或所有参考均未过 G2 而候选通过；速度 RMS 不劣化超过 0.1 m/s |
| 近似持平 | 若无明确优势或差异小于复跑波动，保留通过门槛的最简单参考；不能为了“新控制器”必选新方法 |
| 最低速度 | 原始事件数与 percentage 如实报；固定版本 actual penalty 按 evaluator 计算。另报 v_des≥2 时 speed<0.5 连续≥5 s 的低进展片段，帮助定位爬行 |
| 升级条件 | 两个 PID 在无交互可行路径均 RMS>0.5 m 或控制导致 blocked，才触发 PDM-Lite lookahead 对照；传感器/时间错误先修复 |

Dev10 本身兼有 ablation（方法对照与选择）用途。选完参数后，增加预先固定的非 Dev10 六条路线
（直路、左转、右转、S 弯/换道、小地图、大地图覆盖）验证默认与最强参考；在 T0 生成 route ID 清单和 hash，
选择只看几何/场景类型，不看运行成绩。保留集只验收，不回头调参。失败则报告泛化不足。

中心线 driver 不能用全部 Dev10 的成功率证明提交分数改进。真实 planner 接入后必须固定 checkpoint、anchor、
输入和推理节拍，做 controller-only 替换；最终跑榜使用冻结后的 full220，不能在最终评测集逐路线调参。

### G5：报告、成本和复现验收

每个 attempt 保存 `control.jsonl`：frame/time、trajectory ID/source time/age、各运动传感器帧、估计后轴位姿、
speed/yaw rate、reference/command speed、aim_xy、cross-track/heading error、control、各分项与饱和、
stale/rejection reason。route 参考能从输入文件重建；接 planner 时需逐次保存真正执行的 20×2 轨迹。
真值和 GNSS 对照单独记录 frame，避免用不同帧的位姿算误差。缺失/截断日志要标数据质量，不能默认为零误差。

report 从 `results.json` 重建官方分数与事件，从 telemetry 重建跟踪指标，与 attempt/profile 合并。
所有 attempt 留记录；每条 route 的评分只采用预定重试策略的正式结果，不能从多次驾驶失败里挑最好一次。
基础设施重试可沿用 runner，须报首尝试、重试数、最终状态与全部消耗。

结果至少包含这张逐路线长表（route × preset × seed），并生成按 preset 的汇总：

| 字段组 | 内容 |
|---|---|
| 身份与状态 | route ID、town、scenario、preset、seed、attempt、官方状态、进程状态、是否 capped |
| 驾驶 | completion、score_composed、blocked、deviation、collision 分类、MinimumSpeed 事件/percentage/penalty |
| 跟踪 | 横向 RMS/p95/max、heading、v_command 与 v_reference 两种速度 RMS、停点误差、有效样本数 |
| 定位与时间 | raw/fused pose 中位/p90、heading 误差、trajectory age 分布、stale 数、饱和比例 |
| 成本 | ticks、sim time、ms/tick、attempt wall、初始化/清理开销、controller p50/p95/p99、重启与重试 wall |

每个 preset 同时报按 tick 加权和按 route 等权的误差，首碰撞前/全程分开，成功与失败路线 tick 分布分开。
报告 tick 的 min/median/max，与 600–1200 的旧预测对照；提前失败的短路线不能解释为更快。

成本用 `T_route = T_setup + N_tick × T_tick + T_cleanup`，总 wall 另包含 server 启动、重试和排队。
Tokyo/3090 与 GPU box、policy=none 与真实推理、窗口与离屏均分表。
用 Dev10 更新的是 route driver 的范围；220 条估计按地图和场景构成加权并列假设，不把简单 `×22` 当实测。

`research/results/b2d/controller/` 收录小型 results.csv、summary.json、manifest 与参数；完整日志和录像留 `/data`。
更新 research 的错误结论、车辆常数和反证条款；更新 `docs/bench2drive-cost.md`，新坑写 `docs/tokyo-box.md`。
最终回报列出 selftest 数字、所有 Dev10 行、默认选择、pose 误差、版本、commit hash、偏离原计划及未完成项。

## 跑之前的预测与预算

**预测，尚未验证：** 解析反例提示原 pure-pursuit-plus-bearing-PID 会在弯道有系统性过度转向；
单独 pursuit 在模型与定位正确时更可能稳定，但真实转向响应可能推翻这个判断。
CARLA/TCP 谁更好目前不押定胜负；假设 CARLA 较激进、TCP 转向较平缓，需分别核实过冲和恢复速度。
GNSS 抖动、时间历史和后轴偏移很可能比进一步调增益更先成为限制。

成功且交互较少的路线，仍暂估 600–1200 tick；原始 Dev10 可能因中心线无法绕障出现更长尾甚至 blocked。
用 0.10 s/tick、70 s/route 固定开销作启动预算，10 条约 22–32 min，未含重试；这不是测得的新耗时。
首个完整 smoke 重新估时。每轮超过 1 h 暂停该轮并查原因，保留已完成结果；不因为超时就缩小评测分母。

| 工作 | 初步预算 |
|---|---|
| 契约、controller、单测、定位/双频率集成与报告 | 约 1–2 个工作日；如果已有改动归档或坐标问题较多，需要重新估计 |
| 标定、CARLA 无交互开发、smoke | 1–2 h server 时间 |
| 首轮三种 preset | 约 1.1–1.6 h server 时间 |
| 默认/最强参考第二 seed + 六条保留集配对 | 约 1.2–1.8 h server 时间 |
| 初始总 server 预算 | 约 4–6 h，分短阶段运行；超过预算先解释失败源，不开展大规模搜索 |

## 结果

尚未运行。本文的 0.2246 m、0.075 rad、0.14625 steer 和 1.67 m/s 为解析核查，版本信息为本次只读检查。
本轮没有车辆常数实测、selftest 成绩、Dev10 controller 成绩或控制器带来的分数提升结论。

建议 run 根目录：`/data/runs/b2d/controller/`，子目录按 calibration/development/smoke/dev10/holdout、preset、seed 区分。
