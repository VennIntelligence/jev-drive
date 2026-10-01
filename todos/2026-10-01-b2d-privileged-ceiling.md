# B2D 特权上限：路口冲突、绕障与红灯（预登记）

状态：预登记。本文在本任务任何闭环运行或新结果之前提交并push；仅查了XML场景类型与资源元数据，没有查看这些新路线的旧成绩。任务来自[main提示词](../tmp/2026-10-01-b2d-privileged-ceiling-prompt.md)。背景已读decisions第47/49/52/57/61/74/76/81条、[op-drive](2026-09-29-op-drive.md)、[适配闭环登记](2026-10-01-op-adapt-L-b2d-prereg.md)、[量具](../research/behavior-layer-instruments.md)及[录像诊断](../research/openpilot-seed0-video-diagnosis.md)；decisions只读，由main维护。

## 问题、共用管线与臂

B2D（Bench2Drive，按路线在CARLA仿真器里闭环驾驶）在这里量特权决策能挽回多少分，以及现有控制器能否执行。原openpilot Cinque保持冻结，不训练，不换模型、相机、desire（原生离散意图输入）或导航信息。特权只进仲裁，当前其他actor（交通参与者与障碍物）的真值位置、朝向、速度以及当前灯色可以用，active_scenarios（场景脚本登记）和脚本未来轨迹不可用。

所有臂复用op_arb_agent.py / op_arb_server.py / op_arb.sh和P7（已固定的路径跟踪控制器）。drive采用原curv横向执行、R1 resume（红/黄灯前不按驾驶员恢复键）、8m/s设定速度、原路线命令区/分歧兜底、lead IDM（前车跟驰减速）、行驶中plan约束、intent停车锁存、5s恢复与低速滑行，原参数不变。特权分支关掉时须保持逐数组恒等，不复制另一套drive代码。

| 臂 | 唯一改动 |
|:--|:--|
| drive | 同批重新运行的原管线；仅增加所有臂共有的评价日志 |
| pjunc | 路口冲突时添加纵向停车约束，清空后允许恢复；横向不变 |
| pbyp | 真值静止障碍触发邻车道几何路径，不查对向间隙；纵向约束与恢复保持原drive |
| pbypgap | 同pbyp，但仅当前真值对向间隙够时开启借道路径；间隙不够保留原路径及原纵向，不额外加停车规则 |
| pred | 沿用R3a真值红/黄灯约束，补绿灯后的明确放行；非灯相关停车仍是原R1 |
| pall | pjunc、pbypgap、pred同时开；冲突/灯优先于放行 |

pbypgap的范围有意严格：它是横向候选的间隙门，不额外给纵向真值障碍刹车。这意味着“等间隙时原纵向能否停住”也是现有仲裁的执行诊断；若因此撞原障碍，不能把该失败掩盖成借道决策没有价值。各臂可能变快或变慢，都报均速、等待、卡死和超时；无配速对照，本实验不能分离纯变慢贡献。

## 精确定义与固定参数

真值每0.20s更新，控制仍20Hz；用同一world snapshot（当前仿真状态）取位置/速度，不能访问未来。查询范围100m，忽略与ego（自车）高度差>2m的actor。向量化路径投影与预测须在合成子集上和逐actor参考计算对齐，几何坐标误差≤1e-6m、全部开关一致。所有臂记录独立collision sensor（碰撞传感器，仅评价），接触对象ID、时刻和冲量；不把它送入模型或用于提前控制。

**路口**：路线地图waypoint（车道中心点）的is_junction区连成冲突区，当前进度往前60m内查区。仅对当前速度≥0.5m/s、朝向与该区路线切向差≥30°（含对向）的其他车辆检查；同行跟车仍归原lead。用当前速度匀速外推0.25…5.0s，ego沿给定路线、按原base governor（设定速度/曲率/加减速）的5s剖面前进，双方二维有向包围盒加0.5m缓冲，用SAT（分离轴检验，判断两个矩形是否重叠）判断同一时刻冲突。不能用场景的将来轨迹。

纵向停车位置是冲突首次预测路线位置之前，给前保险杠留2m；用原IDM零速目标生成约束。冲突不存在持续0.8s后释放，若该特权约束导致站住，最多2s暂时去掉plan与latch（锁存）阻止放行的约束，保留base、lead和仍生效的灯/冲突约束；达到1m/s即结束这段放行。已经进入停车点、来不及刹住要照实记late，不调大时间窗追结果。

**静止障碍**：vehicle或static.prop对象速度≤0.2m/s连续2s，包围盒与原本车道路线路径走廊（ego半宽+对象投影半宽+0.15m）相交，位于前保险杠前50m至后5m，路线位置不在junction；车辆若处于路线红灯50m内则不当绕行障碍。同路线上相距≤20m的挡路物并为一组，用整组最前/最后包围盒确定绕行区。不能按scenario类别或active_scenarios定位。

邻道只取地图Driving车道，高度差≤0.75m、中心距2.5–4.5m；优先同向左、再同向右、最后对向左/右，固定顺序不看运行结果。不可达就记no_adjacent_lane。横移量用该邻道真实中心与原车道中心差；路线法线沿给定路线变化。首障碍前20m开始smoothstep（端点斜率为0的平滑插值），15m内移到邻道；最后障碍后8m开始回原道，15m内回正。原路线进度、原纵向信号保持；开启时临时改为路线P7跟踪该路径、curvature置空，结束后恢复原横向归属；只改这段几何，不改openpilot输出。

pbypgap借对向道时，以当前速度/位置外推对向车辆，与借道走廊相交的最早到达时间必须大于`(到整组末端+23m的距离)/max(ego速度,2m/s)+2s`，且当前最近纵向间隙≥15m才允许开启。已开始借道后完成回正，不因新出现车辆突然跳回原路；记录此时gap违规和最小二维包围盒表面间距。无间隙臂不做这道门。上述门只使路径保留原道或开放邻道，不暗加真值纵向刹车。

**红灯**：原R3a route stop waypoint定义，红灯距前保险杠≤50m、能以4m/s²停下时添加约束，停止线前0.5m停；黄灯沿原可停性判断。红/黄阻止plan、lead或timer释放灯相关停车。只有先前真值灯约束已触发且对应灯变绿，才启动上述最多2s的plan/latch放行窗口，保留其他特权冲突与lead，达到1m/s即退出。非红灯相关锁存不得无限等待绿灯。

所有参数现在写死；调试只允许下列非评测路线上的一次参数修订，若需要先追加理由/新值、commit/push，再重新完成调试阶段；调试后记录配置hash（文件校验）与commit锁定，正式评测不得调参。真bug修复记录偏离，原判定线永不放宽。

## 路线与功效

从bench2drive_0.0.4_val.xml按类型选，不看成绩、不优先小town：下表每个目标类型取XML顺序中前3条；不够3条取全部。剔除原held-out19条、调试8条、dev10条后选，再原样加入dev10条。冲突新增24、障碍新增24、dev10，共58条，每臂TM seed（交通随机种子）0/1，六臂共696次正式运行。每技能预计48–60个以上机会，但实际机会数按真值日志判，不能把场景数冒充机会数；每臂不足30个则功效门失败、不判技能无效，不依据效果挑路线补样本。

| 组 | 类型 | 固定路线与town |
|:--|:--|:--|
| junction | NonSignalizedJunctionLeftTurn | 7616 (Town12)、6999 (Town12)、7157 (Town12) |
| junction | NonSignalizedJunctionLeftTurnEnterFlow | 34183 (Town13) |
| junction | NonSignalizedJunctionRightTurn | 7841 (Town12) |
| junction | OppositeVehicleTakingPriority | 8859 (Town12)、9102 (Town12)、9218 (Town12) |
| junction | SignalizedJunctionLeftTurn | 4721 (Town12)、4104 (Town12)、4183 (Town12) |
| junction | SignalizedJunctionLeftTurnEnterFlow | 34391 (Town13)、35243 (Town13)、35330 (Town13) |
| junction | T_Junction | 25051 (Town01)、28180 (Town11)、27823 (Town15) |
| junction | MergerIntoSlowTrafficV2 | 36792 (Town13)、24162 (Town12)、37223 (Town13) |
| junction | SignalizedJunctionRightTurn | 5423 (Town12) |
| junction | VehicleTurningRoute | 10364 (Town12)、10255 (Town12)、9646 (Town12) |
| obstacle | Accident | 19324 (Town12)、19186 (Town12)、19359 (Town12) |
| obstacle | AccidentTwoWays | 31045 (Town13)、31100 (Town13)、30939 (Town13) |
| obstacle | ConstructionObstacle | 2520 (Town12)、18868 (Town12)、18512 (Town12) |
| obstacle | ConstructionObstacleTwoWays | 29893 (Town13)、29770 (Town13)、2608 (Town12) |
| obstacle | ParkedObstacle | 19832 (Town12)、19753 (Town12)、19681 (Town12) |
| obstacle | ParkedObstacleTwoWays | 2667 (Town12)、21546 (Town12)、21496 (Town12) |
| obstacle | HazardAtSideLane | 28387 (Town13)、29183 (Town13)、29073 (Town13) |
| obstacle | HazardAtSideLaneTwoWays | 21037 (Town12)、21188 (Town12)、21009 (Town12) |
| dev | SignalizedJunctionRightTurn | 27043 (Town15) |
| dev | VanillaSignalizedTurnEncounterGreenLight | 15102 (Town12) |
| dev | T_Junction | 24944 (Town02) |
| dev | VanillaNonSignalizedTurn | 27870 (Town04) |
| dev | StaticCutIn | 22535 (Town12) |
| dev | MergerIntoSlowTrafficV2 | 37969 (Town13) |
| dev | ConstructionObstacle | 24497 (Town06) |
| dev | VehicleTurningRoutePedestrian | 27297 (Town11) |
| dev | OppositeVehicleTakingPriority | 9196 (Town12) |
| dev | SignalizedJunctionLeftTurnEnterFlow | 28147 (Town10HD) |

调试路线8条为334、27787、24721、26872、26537、17749、25169、24955，均不进入58条评测：前6条沿用旧tuning（只调试的集），后2条增加事故/施工几何，仍按预先类型规则选。原held-out19条完全不运行。主技能统计用对应新增类型组，连续性dev单列；58条总体DS亦报，不能事后挑某一组的显著结果当主结论。

## 事件定义与读数

事件机会不以特权开关是否触发来定义，否则drive无机会。所有臂都由同一评价器独立检查真值。路口机会：当前或前60m路口区上，其他moving车辆的0–5s匀速轨迹与路线走廊相交且方向差≥30°；同一路口合并全部相关车辆，直到ego越过区末5m，或运行终止。报告机会数、涉及车辆数、接触碰撞、v<0.2m/s连续≥1s的等待、等待总时长；未越过区且连续等待≥60s或blocked/timeout终止算失败，碰撞也算失败。预测停车与真实安全通过是两个不同指标。

障碍机会按上述真值静止/挡路定义，同组只计一次；报告过去、过去且回正、停住卡死、与组对象碰撞、no_adjacent_lane。主成功要求越过组末6m、并在回正区末后恢复到原路线横向≤0.75m；碰撞、到运行终止仍没完成回正、或原地≥60s都算失败。借对向道的表面最小间隙、开始时gap、开放/关闭次数另报。

红灯机会：路线真值灯首次为红、距离(0,50]m，按灯ID/停止线合并至越过后5m或运行结束；报告停住、穿过时仍红、绿灯后首次v>1m/s的时差，未起步为右删失（只知道等待至少这么久）。黄灯控制保留但黄灯机会单列、不混红灯主分母。红灯主失败=闯红或仍未通过且blocked/timeout；放行≤3s比例另报，不让“没闯灯但永远不动”当成功。

碰撞采用额外sensor任意冲量≥1N的事件相关接触，同对象1s内接触合并；B2D违规计数另报，二者定义不同，不能混用。卡死/超时属于驾驶结局，不是程序崩溃。每事件保存开始/结束时刻、route进度、对象ID、几何与控制状态，未完成不悄悄从分母剔除。

汇总DS（Driving Score，路线完成率乘违规罚分）、RC（Route Completion，完成率）、各类违规、blocked/timeout、均速、横向执行偏差与控制时延。丢分分解用RC×0.5^撞人×0.6^撞车×0.65^撞静物×0.7^闯红重建，报与实际DS的差；每次删掉一项罚分反推该项可挽回的平均DS，路线未走完项把RC设100；这是代数反事实而非实测因果，各项乘法重叠，不能相加，stop sign等未建模罚分留在残差。

## 可见性：几何上界，不做遮挡

仅drive的自己的运行读数。0.2s真值轨迹中记录100m内对象3D包围盒，投影到原road/wide的native（原始相机图像）1928×1208平面，使用现有rig位置与焦距。包围盒前方角点的投影与图像边界相交且裁后面积≥4px²就算几何进入视野；记录两个相机分别和union（任一相机）首次时刻、距离、框宽/高/面积，终点取相关碰撞最早时刻，否则与ego二维表面间距最小时刻。20Hz相机、5Hz真值采样产生≤0.2s时刻量化误差。

不读depth/语义分割，也不做遮挡raycast（射线检查），所以首次几何可见提前量是实际无遮挡可见的上界；native图像含模型warp crop（输入裁剪）之外部分，又是对模型可见性的宽松上界。很早几何可见不证明模型读得出；上界仍<2s才支持输入来得太晚。首个记录已经在视野里的对象标left-censored（更早是否可见未知），不把截断的<2s算确定的不足；始终未进入两相机视野则提前量按0单列。报告完整分布、可估与删失数、<2s比例的下/上界，不把删失当成0。

## 统计、登记线与结局读法

全部CI（confidence interval，置信区间）按route聚类bootstrap（整条路线连两个交通seed一起重抽），2000次、seed0、两侧percentile 95%。DS先每路线平均两个seed后配对。事件比例每次重抽累加失败/机会再取比例，特权对drive配对重抽同一路线；不同臂可能遇到不同对象/时机，报告分母变化，这是策略改变曝光后的路线配对效应，不是相同个体事件的随机试验。控制条件为同类型组的drive，其他组为连续性/副读数。pbyp和pbypgap分别判，不依据成绩选最好当主臂。

| 线 | 固定判据与读法 |
|:--|:--|
| I0 功效/完整性 | 58×2×6结果齐，程序崩溃为0，每主技能组各臂实际机会≥30；不足则无法判、不能写无价值 |
| J1 路口值得注入 | pjunc在junction组失败率差的CI上界<0，且同组DS差点估计>0；DS CI下界>0另称强证据 |
| B1/B2 绕障值得注入 | pbyp/pbypgap在obstacle组分别满足同样的失败率与DS条件 |
| R1 红灯有价值 | pred在全58条的红灯失败率差CI上界<0且总体DS差>0；绿灯≤3s放行比例与删失照实报 |
| X1 执行可用 | 对应特权臂主组成功且回正/放行的比例≥0.80、因执行失败占比≤0.10；事件无改善则按触发→约束/几何→控制→放行→完成的链逐步定位，不据单个阴性CI断言无技能价值 |
| V1 观测不足 | drive对应类型组确定union提前<2s比例的整段CI下界>0.50，且机会≥30；几何上界如此短才标观测问题，其他情况不证明输入足够 |
| A1 联合上限 | pall总体DS差>0且总体事件失败率下降CI上界<0；单项和使用pjunc+pbypgap+pred在总体58条的DS差，报告pall差减三项和及CI：CI上界<0是遮挡/冗余，下界>0是协同，跨0无法区分相加；不把丢分分解项相加 |

事件改善而DS不增：按丢分分解和新增失败说明别的损失抵消。没有事件改善且触发错误/执行不动：诊断感知定义或执行仲裁，不能证明学习决策没用；只有特权信号和执行链经检查却仍没收益，才说“当前这项设计未显示上限收益”。所有统计结论受匀速预测非完美未来、几何视野不含遮挡、历史dev已见与两seed限制。

## 分级检查、profile与自推进链

先pred在27787、seed0、一个worker一个特权臂（非评测、录制）。S1：配置Cinque原模型/R1正确；结果与日志齐、无程序崩溃/非有限数；灯约束至少一次在红灯前触发且到停止线前v<0.2，变绿后至少一次v>1；非灯锁存仍保留5s通路；几何开关关闭逐位恒等，合成SAT/路径参考对齐；时延median≤75ms、p99≤200ms。若该route没有红→绿观测或检查不过，停止诊断，不用另一条更好成绩偷偷替换。

再约10单位：调试drive/pred在27787，drive/pjunc/pall在26872，drive/pbyp/pbypgap/pall在25169，drive/pbyp/pbypgap在24955，共12个路线臂单位、seed0；先前pred保留为调试证据，不进正式。按适用项查：冲突约束至少一次且清空恢复；至少一个障碍被检出、可用邻道、平滑路径最大横移2.5–4.5m、路径有限、绕后回正；全部程序崩溃0、每route最多3次infra启动尝试；可正常终止的Completed/blocked/TickRuntime算驾驶结局。任何执行检查失败则ERROR停链、保留日志、修bug，不进全批。

profile与中间阶段合并：before每卡1slot×2worker，after每卡2slot×4worker，先看显存/PID才加负载；两个路径同合成子集数值对齐，参数不因打包而变。每5s记录GPU利用率/显存与CPU忙核，报告单位墙钟、sim tick/s、计划/真值查询时延、卡尾空闲与瓶颈。全批切成每臂每seed的4条路线小shard（分片），六slot动态取队列，复用各自Cinque服务，完成一个shard即自动读数；每卡最多8 CARLA、2模型server，各slot12核，全部CPU0–74内，实际资源查过后登记自己的lane与index范围。PID软门17000、硬线17500；资源不足就等，不改参数或事件线。WORKERS登录变量不改，自己的并行变量PC_WORKERS/PC_SLOTS，只在子进程接口env传旧脚本需要的WORKERS。

预计正式696route runs、每次2–10min，24worker约2–5h；大地图/卡死尾部与启动预算乘1.5，预计4–8h，加调试/profile约1h，上限计划12h墙钟。不是预算到点就删失败或改线；实际profile后更新估计。队列带DONE/ERROR/STATUS，tmux jev通过tmux_run.sh和slot_run.sh；log.txt/events.jsonl/tb/俱全；每个job边界重查自己的GO。box当前3卡、75核/约276GiB，以实测为准；以前B2D lane已归档且无CARLA进程，才允许登记这些空闲资源，不改别人的记录。

## 视频、交付与执行日志

每个特权臂和对应drive先录调试视频，再在正式结果出来后按route ID数值顺序各取首个成功/首个失败做诊断重跑（最多每臂2段；无该结局则记不存在），同route/seed drive对照也录。这些附加重跑只为录像，不覆盖正式成绩，不把新轨迹冒充原运行录像。沿用op_drive_record_agent/record_video的追车相机与编码做法，模型相机不变，MP4留box，QA（文件与渲染检查）与路径写入todo。

代码在scripts/，所有新的大产物只写$DATA_DIR/runs/b2d_privileged_ceiling/；op_adapt_r2、op_adapt_L、op_l_b2d只读。小结果下载到[research/results/b2d-privileged-ceiling/](../research/results/b2d-privileged-ceiling/README.md)，论文风格PNG由此todo引用，每图后1–3句中文读法，PDF留box；不发布Artifact（网页产物）、不提交密钥，不改decisions。每次变更在main显式stage（暂存）自己的文件，commit/push后box pull；最终按精确PID关闭自己的server/window并finish自己的lane。

执行日志从此处追加；偏离须保留原因、发生时刻、是否已看正式结果及对解释的影响。最终写全每臂表、每条线的判定、verified（已实际核验）数字和inference（推断）边界，回答技能值多少、执行是否可用、相机可见多早。
