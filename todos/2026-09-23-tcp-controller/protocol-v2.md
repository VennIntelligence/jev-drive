# 协议v2修订：真实静止速度容差

2026-09-23，在paired-v1第五例1773发现停车附近微小负速度触发共同reverse guard，根代理主动中断本轮；不是用户取消或驾驶正常完成。前四完整结果和第五部分结果全部保留，当前162文件索引已封存。

唯一修订：TCPControlComparison对有限|raw_speed|<.01m/s归零用于有效速度与倒车判断，原始speed单独保留；>=.01m/s负速度仍故障制动。模型与原生PID仍读取原始速度，原生预测/desired/转向不动。两臂应用相同有效速度边界；这是oracle适配器已有的数值静止容差，不是提高运动速度阈值或改变路线。fault字段与deadband字段分开。

17项测试通过；已对捕获的真实输入前缀重放旧/新规则，保存每帧输出及源码。重放不证明闭环恢复。因此在独立paired-v2目录重新执行完整六例，不与v1成功案例拼接；仍同一checkpoint/20Hz/公共envelope/route顺序/seed0/600s上限。不得因1773未完成改路线或增益。

原协议SHA256：a9e6a2d979bb9d76d0aa43ce94ae46608ff69b66a35dd7f7abc6d8ccfcdad2a3。以下完整保留原协议文字；其中静止/倒车边界以本修订为准，原协议文件不改写。

---

# 协议：真实TCP、共享执行限幅、固定纵向两臂

版本：2026-09-23，阶段一执行前。范围为3条路线×2臂=6个请求案例；TM seed=0，0.05s同步仿真步长、20Hz推理与控制。阶段二转弯研究另列，当前不更换横向。

## 1. 回答的问题及不可混淆的边界

B/C比较同一真实TCP模型输出所驱动的原生纵向PID与固定PI，在**相同公共执行限幅**下的行为。它不是已完成oracle-controller实验的重命名，不使用dense route生成、替换、重接或修复网络预测。官方导航路线仍作为原生模型的导航输入；独立路线/车辆真值只作记录和诊断。

模型每次输出4×2个点，对应未来+.5/+1/+1.5/+2s。不补点到5s、不复制尾点伪造停车、不改变预测时间。原生横向、导航target-point仲裁、steer clipping保持相同；暂不接完整pursuit。预测物理原点未被完全确认，纵向只用相邻预测点距离，不凭preview原点假设给出毫米级横向执行结论。

## 2. 冻结输入、处理与两个实验臂

固定权重为真实TCP checkpoint `/data/models/bench2drive/tcp/tcp_b2d.ckpt`，已有SHA256 `e6573ff1f8ea910b9a53eddfb68f69cac469bf5bfa253a516578f6126110b4fe`；运行前重核文件与加载missing/unexpected keys，并保存实际源代码字节。仅权重存在不算加载匹配。

显式`PLANNER_TYPE=only_traj`。三路原生相机1600×900、同一个已经验证像素等价的JPEG/crop/resize/normalization模式、相同传感器/天气/地图/车型；BEV/chase不进入模型。所有优化环境变量在启动manifest冻结，不在两臂之间改变。原生首帧中性动作/初始化行为相同。

| 项 | native_common（B） | pi_common（C） |
|---|---|---|
| 模型/预测/导航 | 原生TCP、每步一次forward | 相同代码/权重/节拍 |
| 横向 | 原生目标点仲裁与PID/clipping | 相同实现，不调参 |
| desired speed | 原生4点的三段速度均值 | 完全相同的定义 |
| 纵向 | 原生speed PID及原生desired/brake判断 | 固定SI PI Kp=.5、Ki=.25及已验证的状态边界 |
| 最后执行限幅 | throttle[0,.75]、brake[0,1]、互斥 | 相同 |
| 原官方尾部低速限油门 | 双方移除 | 双方移除 |
| 原官方任意正brake变1 | 双方移除；原生PID本身输出binary brake仍保持 | 双方移除；PI连续制动保留 |

这改变了官方最终执行策略。历史`official_anchor`仅作背景，没有本轮匹配的A臂，不把A→B差异算作PI收益，不默默扩成9例。

原生desired按`mean(||p[i+1]-p[i]||/.5), i=0..2`计算，不含origin→first point；记录metadata并从交换坐标前复制的raw预测独立复算。raw网络坐标forward/right与交换后的metadata分开记录，不能被in-place操作覆盖。

正常有效模型tick只运行网络一次，native PID与PI各更新一次（native_pid_calls=1），用被选中分支驱动车辆，另一个仅作shadow日志。异常例外：raw prediction、target或speed非有限时，两臂共同fault brake；跳过native PID更新并清零其窗口，PI reset，日志native_pid_calls=0。首帧初始化按原生规则不要求一次模型推理，需单列初始化状态。影子控制值属于同一观测下的输出比较，不是另一个车辆闭环的成绩。PI使用真实有效dt，route/reset/故障/停止状态的积分边界需先验证。双方保留相同NaN compass输入规则并记录异常，不把oracle PoseFilter修复自动移植到单臂。

## 3. 六例与路线几何

固定运行顺序按路线相邻配对：24211/native_common→24211/pi_common；1711/native_common→1711/pi_common；1773/native_common→1773/pi_common。同地图可复用CARLA进程，但每例重建ego、agent/model与控制状态；不得同时有多个server owner。仅根代理运行CARLA/GPU。

路线来自固定Bench2Drive XML：24211 Town01 DynamicObjectCrossing、1711 Town12 ParkingCutIn、1773 Town12 ParkedObstacle。每条XML有68位置点、约134m；24211完全共线，1711/1773距端点直线最大约.075/.066m。此为XML检查，不是活体地图插值证明。它们是交互情形覆盖，不是道路弯道覆盖；实际绕障是否转向在报告中独立说明。

seed0只指TM种子，不保证同一NPC生成/动作。每臂保存对手车型、首交互帧、传感器异常、生成失败与spawn/retry信息；不得把不同接触对象称为严格相同世界的反事实对照。

## 4. 执行前质量门槛

- 验证单步仅一次forward、原生steer逐值等价、两臂公共envelope相同、首帧行为与route reset一致；基线原生PID仍用其原离散语义。
- 原始4点副本不被日志/原生交换修改，模型raw、metadata、desired speed相互可核对；finite/shape/time检查失败有明确状态。
- 用合成输入验证PI恒速、加減速、停止、饱和恢复、互斥/限幅、dt和清零边界；不凭合成测试证明CARLA变速箱响应。
- 根代理先验证checkpoint兼容与真实一帧契约，保存日志及实际缺失/多余keys；若影响控制的加载或模型时域/坐标假设不成立，暂停六例并定位，不改协议混跑。
- 真值日志有明确frame、时间、速度、加速度及向量单位，不能回灌网络或控制。原始日志schema到位后再绑定分析字段，未知字段保持missing，不猜单位。

## 5. 行为指标：全程为主，prefix为辅

每个原始样本记录source sensor/model/motion frame、timestamp、raw4点、metadata、native desired、选择臂、native/PI shadow、envelope前/后控制、积分/饱和/gear、实际已应用control、独立真值位置/方向/速度/加速度及官方事件帧。首帧初始化、无效/缺失、gap单独计数，不能默默剔除失败数据。

**时间范围。** 主统计覆盖首个控制tick到该attempt终止；没有根据结果延长warmup，也不剔除加速、减速、停车或碰撞后段。速度误差仅在有效desired与真值同时存在的帧计算，同时报告有效帧数/覆盖率；首帧无desired是missing，不能补0。已完成与未完成的时域长短不同，必须与进度/终止原因/时长一起解释，短早停不能被当作误差改善。

**速度。** `e_v=truth longitudinal speed−native desired speed`，真值纵向速度为世界速度点乘同帧车头forward向量，单位m/s。另报native SPEED输入与同desired的误差、3D速度模长与raw signed speed，防止符号/投影差异混入比较。输出mean bias、RMS、|error|p95/max、target变化率分布与速度时间曲线；原生三段desired独立复算误差也报告。全程、实际moving(|v|>.5m/s)和预测stop(desired<.4m/s)为预定子集；后两者只诊断，不替代主表。

**加速度和jerk。** 真值世界加速度点乘同帧forward/right，得到a_lon/a_lat，单位m/s²；输出signed p05/p95、绝对p95/max与RMS。主jerk先用相邻有效且frame连续的世界加速度向量差除实际dt，再投影到当前帧forward/right，单位m/s³；世界向量jerk范数另列。无滤波为主，并保留原始向量与dt；不跨缺帧或不连续时戳差分，不把gap插成平滑轨迹。a_lon/a_lat先投影再差分的结果可另列为身体轴变化诊断；它包含坐标旋转影响，不替代主world-derivative jerk。任何平滑曲线只能是另列展示，需固定滤波参数且不能覆盖原值。

**执行与安全。** 记录throttle/brake duty、同时非零次数、饱和比例、steer变化率、API gear变化和执行前后差异。安全完成同时报告官方status/completion/DS、碰撞及首接触帧、越线/偏离/blocked/timeout、真实低进展段、未完成与重试；officialblocked=0不代表没有stall。真实低进展固定观察speed<.5m/s持续≥5s，另列desired≥2子集和故障导致desired缺失段，保留事件前后上下文。

**prefix。** 首次碰撞前frame严格小于首接触frame；无可用事件帧则missing，不能默认为全程无碰撞。可补一张配对共同仿真时长prefix：截止=min(B/C首接触时刻、B/C终止时刻)，使用各自从首受控tick计时；它仍不代表相同NPC状态。所有prefix带样本数/持续时间，不能据此替换全程、删除碰撞后失败或选择赢家。

**路径。** 模型原预测、同帧原生目标/heading、真值车辆轨迹与独立路线一起展示。距模型自身预测的误差是局部跟踪诊断；密集导航路线距离在绕障时不是唯一正确性指标。原点/未来杆臂尚未证实时，不将误差数字当绝对轨迹精度证明。

## 6. 汇总与验收措辞

每条路线固定B/C配对，先给全程完整表，再给C−B差值。三路线汇总按route等权，不让长时间卡住的一条凭tick数主导；同时保留tick加权值、样本数和缺失ness。B值为0时只报绝对差，比例设missing。所有实际attempt计成本，不挑最好的重试；沿用first-harness-finished选择，若无finished则保留最后attempt及原因，基础设施失败与驾驶失败分开。

“六例对照完成”要求六个请求案例都有明确终态或基础设施失败及完整归档；缺少可计算行为数据的pair标不可比较，不能只用剩余成功pair宣布优势。安全未完成本身不因日志齐全而变通过。

“值得继续验证的行为信号”预先要求：C没有新增未完成/碰撞/偏离/blocked等安全恶化；三个route等权全程速度RMS、|a_lon|p95、|j_lon|p95分别不升，至少一项降低。逐route上升仍要标为tradeoff，不可只呈现均值。若信号与NPC实现差异或未知接触责任纠缠，仅描述结果，不宣称已证明控制器因果收益。此处不沿用oracle的固定巡航.5m/s门槛，不以单项DS更高替代行为条件。

这只是预定小样本的有限继续依据，不是显著性检验、默认升级或全220成绩承诺。若速度误差下降但jerk上升，或更平顺但进度/安全变差，明确报告取舍，不自动调整权重挑胜。后续有价值的工作必须定位预测轨迹是否更激进/穿障碍，还是车辆未执行模型意图；不能把所有差异归因PI。

## 7. 停止边界与保留规则

每例墙钟上限600s，原官方TickRuntime上限继续生效，不额外缩短仿真tick cap换取成功。任何超时/取消/异常/碰撞/未完成均保存；崩溃重试保留全部attempt和额外成本，不自动增加路线或增益搜索。若不可恢复运行/传感器/模型契约错误，停止依赖部分，先归档定位；普通驾驶失败继续执行其余已批准配对，不因不好看删除。

一个交互若决定结论，是否双方各追加一次复跑由根代理另行确认并记录范围，不能自行扩展此六例。源文件、权重、配置、环境、日志、官方事件/结果、图表及分析代码均记录hash；只在run闭合后全量索引。新增实验/图表用新目录，旧oracle结果与真实TCP历史不覆盖。

## 8. 阶段二：少量转弯案例单独开发

用户另授权横向转弯小范围优化。阶段一六例仍只切纵向，不能混进新横向gain/lookahead。阶段二由路线几何预先定义turn windows，冻结route ID、起止弧长/几何阈值、前后缓冲与不重叠规则；不能看控制误差或某臂成绩后挑有利窗口。

候选路线/窗口正在由独立agent审查，包括已有26966、17563等的真实几何；这些旧oracle几何证据不等于真实TCP已驶过同一窗口。选定后另存协议附录，明确模型4点/.5s时域、原点与原生target arbitration是否保留。不把三个近直XML承诺成转弯验证。

先在少数冻结窗口开发，再运行包含进入/退出与交互的完整路线回归；窗口与完整路线指标同时通过才可支持局部改动继续。只降低全路线均值可能掩盖转弯瞬态，只有窗口改善也不能替代全路线完成/安全回归。本阶段不预先宣布横向改善。

## 冻结时的执行前检查记录

根代理在真实GPU1完成checkpoint strict加载：missing/unexpected keys均0，参数26,593,444，单帧输出4×2且finite；13项TCP单测及9项runtime测试通过。这些检查验证接入契约，不代表六例行为已通过。相关原始验证日志由根代理归档。

本协议在真实六例启动前冻结。每例核对协议hash；之后发现、异常解释或阶段二窗口以新文件/附录保存，不改写本文件或已经完成案例的定义。
