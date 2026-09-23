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

## 可选实车坡道保持 helper

新增 `scripts/b2d_controller_slope.py` 与 `scripts/test_b2d_controller_slope.py`。API：

```python
from b2d_controller_slope import run
summary = run(client, "/data/runs/b2d/controller/calibration/controller_config.json", out)
```

调用方必须在自己的评测 actor 清理后调用。helper 不启动/停止进程，只管理自己创建的 `slope_diagnostic` MKZ；恢复最终地图的 world settings。如果当前地图找不到 drivable |pitch|>=3° 位置，可查询并加载 Town12（若已有 vehicle 会破坏调用方状态，则拒绝地图替换并标明 untested）。地图可能改变，已在接口 docstring 明示。

静态位置去重，最多8个候选；只对 spawn 不可用、实际坡度不足或无法安定的 setup 重试，已经测得的 hold 失败不会另挑更好的位置覆盖。先settle至少2s且连续1s speed<.1，再记录整整5s（101个20Hz样本）的真实后轴3D最大两点距离、速度范数、完整带符号速度与坡度/grade。门槛：全段实际|pitch|>=3°、位移<=.1m、speed<.1m/s。100个样本仅4.95s，测试明确判失败。

控制器用5Hz全零轨迹、20Hz step；其输入 speed 是物理速度范数以满足非负API，评测 signed velocity 不做deadband或截断，负值全部保留。该工具明确是 privileged stationary-trajectory plant diagnostic，验证刹车保持，不能替代接近停车或sensor-score实验。

输出 `summary.json`、`slope.jsonl`、`events.jsonl`、`log.txt`。无合适坡度返回明确 untested。5/5离线测试通过（含无服务器所有权的fakeclient、signed reverse/位移超限、返回原点却中间漂移、坡度不足、4.95s边界）。未运行CARLA，等待主代理在复用server上按需调用。

## 冻结 Dev10 seed0 失败分析（只读）

产物：`results/failure-analysis.json`（已移出 git，GPU box：`$DATA_DIR/runs/b2d/controller/git-offload-v1/todos/2026-09-22-b2d-controller/results/failure-analysis.json`）；完整30个官方结果（carla/tcp/pursuit各10），另保留TCP27494首次`server_died_rc139`基础设施失败（22s、无官方record），不混入成功次数。没有更改冻结controller/adapter/report。

### 25424：支持“中心线路径穿过施工障碍”归因

| preset | 官方completion/status | 首次静态碰撞frame | 碰撞前truth横向RMS | 连续\|raw速度\|<.5区间 | 区间真实净位移 |
|---|---|---:|---:|---|---:|
| carla |47.26 / Failed-TickRuntime|3282|.0454m|3303–7123，191.05s|.169m|
| tcp |49.14 / Failed-TickRuntime|15955|.1031m|15982–19797，190.80s|.0627m|
| pursuit |47.26 / Failed-TickRuntime|4953|.0550m|4973–8794，191.10s|.2026m|

三个preset都碰到`static.prop.trafficwarning`。碰撞前最近一次记录的20点世界轨迹位于原dense route上（最大重建偏差接近浮点零），没有绕障参考。停滞中位油门.75，真值横向误差很小，说明此例的主要问题是未规划绕障，不能自动判横向controller跟丢。源码`ConstructionObstacleTwoWays`明确要求占用对向车道，warning prop沿施工车道布置，与事件吻合。

严格限制：criterion message中的x/y是**ego actor位置**，已核对`atomic_criteria.py`；不是障碍物中心。日志没有other actor的连续位姿、bbox或contact force，因而只标`path_through_obstacle_supported`，不假称重建了精确多边形相交。官方失败全部保留。

4000tick来自锁定版本scenario_manager本身的固定保护门槛，hooks保留了同一逻辑；CLI max_ticks=0 / capped=False仅表示没有另加tick cap，不能解释为不存在任何上游上限。

### 2091：碰撞后约180s低进展，原因保守保留

三个preset均官方Completed100%，tick分别3868/3881/3871（carla/tcp/pursuit）。首次车辆碰撞frame7507/20240/9241。连续\|raw speed\|<.5m/s分别177.9/179.15/178.95s，净位移.410/5.298/.519m，之后才继续完成。

统一窗口“首碰撞+2s到最后sample-10s”的truth cross-track p95：carla .843m、tcp 1.866m、pursuit .202m。碰撞前RMS分别.121/.157/.065m。说明TCP碰撞后恢复/侧向运动更差，不能把这条路线所有损失都从controller责任中排除。没有其他车辆占用/移除时间序列，不推断何时脱困由何机制触发。归类`collision_associated_long_dwell`，控制恢复与外部接触各自贡献仍unknown。

### 状态/定位异常

30条内没有`trajectory_behind`或stale。invalid_motion tick总数carla881、tcp977、pursuit873；25424就占651/702/683，主要是碰撞接触中出现小负signed速度触发非负输入防护（例如carla -.057到-.010m/s），而非已证实的NaN/Inf。2091负向速度幅度更大（carla最小约-3.57m/s），不能静默夹成零。

25381三个preset都在起步有2tick invalid_motion和2tick trajectory_outside_history，随后正常恢复并完成。所有异常区间frame边界与示例保留在JSON。全30条pose_error最大.51274m，未出现>1m定位跳变，因此没有证据把长停滞归因为大的定位发散。

### origin → firstpoint 速度桥接限制

3514三个preset第一条轨迹完全相同：firstpoint=[3.474663,3.376833]m，norm/.25=19.380933m/s，下一条future→future段却为8m/s。真实起始route横向偏差3.14558m，pose error仅.41219m，主要来自ParkingExit的初始几何错位。控制器将ego原点加作t=0，第一段因此包含接回centerline的空间间隙，不能解释为单纯8m/s巡航。

5Hz replan使每个20Hz控制tick的age始终<.2s，reference_speed一直处在这个+.25s首段；near目标是首段与下一段的混合。target/reference两个diagnostic同源，彼此一致不能证明“配置8m/s已被准确跟踪”。bridge>8.5m/s的replan数：carla1811/2779、tcp1997/2801、pursuit1607/2779，主要长停滞与起步贡献；逐路线数值均保存。

这不证明实际车速达到19.38m/s。起步时8与19.38的目标都会把油门限到.75，因此不能仅用首个数字尖峰断言它改变了第一步执行或造成碰撞。本轮冻结数据保留原行为；应在后续版本单独定义空间连接与定时速度，并重跑完整对照，不能在本轮中途修后混合统计。

## 成本文档与 Tokyo 环境文档

已更新 `docs/bench2drive-cost.md` 和 `docs/tokyo-box.md`，只引用Dev10 seed0冻结30条结果及1次基础设施失败；seed1/保留集没有混入。每preset区分全部、官方Completed和Failed的tick min/median/max；600–1200预测实际每组7条低于600、1条区间内、2条超过1200。Tokyo物理GPU1 /3090、windowed Epic、policy-none、front3 800×450、5Hz与旧GPUbox模型/8worker数据分开。

从原始attempt/profile/event重算：finalized attempt总wall1027.0s；TCP崩溃attempt22.0s；初始化/readiness+preflight7.088s；restart→retry7.008s；其余runner/report约11.553s。manifest→end marker共1074.650s/17.91min。**最终server.stop在end marker之后且没有单独计时，所以未冒充包含退出清理的完整进程wall。**

profile按每条自己的ticks_used×total_ms_mean计算，共409.354s；finalized attempt余项617.646s含setup/cleanup、首20tick及未计时工作，不能全叫启动成本。各presetweighted ms/tick为12.345/12.760/12.337。Town04 TCP27494在第18次attempt崩溃，记录server_age_routes17，原server复用跨preset；重启并换端口后复跑成功。只说明此次观察，不推断固定寿命或最优回收间隔。

旧“分数最多209/220”“这些失败完全不是我们的问题”“11条路线不能运行”的绝对结论均改为旧run观察范围和根因限制；也移除3.1h是普遍上界的暗示。文档明确禁止把新Tokyo数字直接外推full220或归因成controller-only加速。用户目视看到的后续seed1交叉口碰撞停滞标记为独立观察，不计作新的定量测量。

本次检查：两文档相对链接目标均存在；数据表与原始JSON/事件重算一致。未运行CARLA、未操作git、未修改其他代理文件。

## 转弯覆盖只读审计

产物 `results/turn-coverage.json`。只用静态reference world_xy，按1m弧长重采样、5m chord算路径切线；不使用车辆实际heading或GPS。CARLA世界坐标正转角=右转。弧段累计至少10°且存在左右两种符号才算明显交替曲率。

| G2 route | 5m chord净转角 | 正累计 / 负累计 | 结论 |
|---|---:|---:|---|
|1773|约0°|+.010 / -.010°|直路|
|24240|-89.915°|约0 / -89.915°|左转|
|26966|+90.577°|+90.577 / 约0°|右转|
|25854|+51.548°|+51.552 / -.0038°|单方向弯道，**不是S**|

六条预选holdout（2050、2084、25318、27529、28154、3072）也只有单方向弯或直路，无左右各>=10°的交替弧段。因此不能说测试全是直线，但确实缺G2专用的无交互S验收。Dev10中17569实际dense reference有+70.09/-70.09°的交替换道曲率，说明完整campaign已有非直线交替路径；它不能替代无交互G2。

已按静态XML几何选定补充候选17563（Town12 SequentialLaneChange，117.175m；不在Dev10或预选holdout）：四个明显弧段约−35.49/+35.50/−35.69/+34.42°。原始XML文件及route element的SHA256写入JSON。该路线是交替换道形状，适合作S/lane-change控制试验；不是声称街道本身必有连续弯曲。

5m chord最大曲率代理约.1287/m，对应8m/s侧加速度代理8.23m/s²，6m/s时4.63m/s²。已建议主代理在看结果前预定6m/s、冻结全部控制器参数、三个preset各跑一次scenario-free补测；8m/s只能另列stress。此代理是平滑几何估计，不是精确曲率或轮胎极限。CARLA实际插值dense route生成后仍需确认交替弧段存在。

没有更改任何core/report/其他doc、没有CARLA或git操作。

补充S基线输入已提取：`results/supplemental-s.xml`，SHA256 `d012120d75784dc3d8d6b253bfddd24ae28b1278c29d1c0e2763804bea111730`。逐字节校验waypoints element与锁定原XML一致，route attributes/weather保留，仅清空一个SequentialLaneChange scenario。`turn-coverage.json`记录提取hash以及先验6m/s、三个preset、冻结v1控制参数协议。主代理安排运行，当前primary baseline core未改。

v2独立设计审阅已提前发送给agent子代理：不能通过平移整条轨迹抹掉实际横向偏移；几何回归路径与定时速度信息应分开，元数据仍须使用同一源时间并覆盖延迟/乱序/NaN/停车释放测试；20×2原API应保留明确定义的兼容行为。显式速度信息可以修复诊断速度，但不能把初始偏离3m的参考自动变成动力学可行轨迹。

### V2 adapter review and controlled matrix extension

- Supplemental S route 17563 extracted from the locked XML into `results/supplemental-s.xml`, preserving weather/waypoints and clearing only the scenario child. `results/turn-coverage.json` records source/output hashes and the geometry-based, predeclared 6 m/s choice for all three frozen presets.
- Read-only v2 adapter review exposed collinear cusp masking in three-point curvature and misleading zero-curvature reports for two-point terminal/short-remainder paths. Agent added minimum 33 samples, terminal heading-aware Hermite geometry, honest curvature concerns and standard-branch folding rejection, with fixtures. Terminal Hermite reversal is a separate final edge; communicated to agent/root for its owned patch. Geometry revision changes the commanded path as well as eliminating the artificial first-point speed bridge; do not attribute it solely to the controller.
- Implemented matrix CLI only in `/data/worktrees/jev-drive-controller-v2/scripts/b2d_controller_validate.py`: optional `--variants` ordered label→{absolute controller_config, presets}; optional `--route-cruises` route→positive speed plus default. Loop order remains route then variant then preset on one server/world, with fresh actors/agent per case. Legacy paths remain route/preset; matrix paths route/variant/preset. Independent truth gates unchanged.
- Every case uses its own rear axle and stop deceleration, reports config source/hash, variant and cruise, and runs exact archived configuration bytes. Inputs and expanded ordering are archived in `inputs/`. Malformed inputs fail before simulator imports/start.
- Validation: existing 15 controller/metric tests plus four new matrix parsing/order/isolation tests pass under CARLA Python 3.8, with no server. Handed back and frozen before root's 20-case launch.
