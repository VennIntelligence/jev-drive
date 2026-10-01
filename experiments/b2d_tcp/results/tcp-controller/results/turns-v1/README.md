# 阶段二转弯实验：short-max 未通过预先冻结的必要条件

日期：2026-09-23。结论：**本轮候选不通过，126 项中 122 PASS、4 FAIL、0 证据不足。** 急右弯主要窗口 RMS 仅改善 **7.56%**，未达到 15% 主目标；同时付出两个窗口 steer-rate P95 增幅超限、一个 S 窗口实际横向加速度 P95 增幅超限的代价。原 G2 六例全部通过，但不覆盖这些新增转弯门槛。

根代理决定：保留负结果，**不追加反序六例，不放宽阈值，不扩展参数网格**。本轮不采纳 short-max，也不据此赋予 baseline 新默认资格或宣称真实 TCP 获益。

## 实际执行与证据范围

协议：[冻结协议](../../turns/protocol.md)，实际运行副本 `/data/runs/b2d/controller/turns-v1/provenance/inputs/00-protocol.md`。三条无交互 oracle 路线分别为 24240@8 m/s、26966@8 m/s、17563@6 m/s，每条 baseline 后接 candidate，总六例；S 路线保留两个独立窗口。六次首次运行均 completed，未通过重试取优。

唯一控制配置差异为 `max_lookahead_time_s`：baseline `.5`，candidate `.375`，共同公式 `max(3 m, coefficient × speed)`；3 m 下限、steer_rate=2/s、max_steer=.8、PI=.5/.25、adapter、定位和实测车辆参数均保持不变。不是旧 `max(2.5,.4v)` 双参数提案。

固定主要窗口：26966 `[24,51]m`；24240 `[18,64]m`；17563 `[27.5,48.5]m` 与 `[73.5,95]m`。窗口按原路线几何预先划定，未根据新误差移动。24240 只评价约49°的高曲率核心及边界，整条路线约90°，因此另保留全路线图。

输入是 `policy none` 的路线 oracle，不是 TCP 模型。无 NPC、已见开发路线、每配置每路线一次，不能把结果写成真实模型、交互驾驶、官方跑榜或统计显著提升。训练 waypoint 的物理原点问题未在本实验中解决，也未用新假设平移模型点。

## 每类必要门槛

完整逐项表：[all-126-conditions.csv](all-126-conditions.csv)；机器判定：[required-conditions.json](analysis/required-conditions.json)。没有用时间池化平均替代逐窗口判定。

| 门槛组 | 条件数 | 结果 | 说明 |
| --- | ---: | --- | --- |
| 原 G2：完成、无碰撞、全程横向、巡航速度、定位、终点和5秒保持等 | 72 | 全 PASS | 六例各12项，原阈值未改 |
| 非法控制 | 6 | 全 PASS | 无非有限/越界/踏板冲突或明确 invalid/stale 故障 |
| 全程帧、truth 完整性 | 6 | 全 PASS | 2,761 个控制帧与 validation 帧逐帧对应 |
| 四窗口 × 两配置完整覆盖 | 8 | 全 PASS | 全部进入/退出；70–117帧/窗，关键字段无缺失 |
| 26966 主目标：RMS 降≥15%、P95不升、post10m RMS不升 | 3 | 2 PASS / **1 FAIL** | 仅RMS改善幅度不足 |
| 逐窗 CTE P95/峰值、航向 P95、其余三窗 RMS 回归限制 | 15 | 全 PASS | 包括两个 S 分别验收 |
| 每窗参考速度误差 RMS 与实际均速限制 | 8 | 全 PASS | 改善不是靠明显降低窗口均速取得 |
| 每窗横向加速度 P95 与 emitted steer-rate P95 代价限制 | 8 | 5 PASS / **3 FAIL** | 详见下表 |
| **合计** | **126** | **122 PASS / 4 FAIL** | **候选不通过** |

原 G2 保持全程 CTE RMS≤.5m/P95≤1m、5秒启动宽限后巡航误差 RMS≤.5m/s、定位P90≤.5m/航向定位P90≤1°、终点≤1m、保持≥5s/位移≤.1m/速度<.1m/s、遥测完整等原门槛。新增舒适比例比较使用协议规定的至少20有效帧和近零基线≤.01单位“证据不足”规则，本轮没有触发缺失或近零豁免。

### 四个失败项

| 必要条件 | baseline | candidate | 实际变化 | 允许值 | 判定 |
| --- | ---: | ---: | ---: | ---: | --- |
| 26966 窗口 CTE RMS，m | .558694 | .516447 | **降低7.56%** | ≤.474890，即降低≥15% | FAIL |
| 26966 emitted steer-rate P95，1/s | .800106 | 1.079711 | **增加34.95%** | ≤.960127，即增加≤20% | FAIL |
| 24240 emitted steer-rate P95，1/s | .218380 | .377941 | **增加73.07%** | ≤.262056，即增加≤20% | FAIL |
| 17563 S1 横向加速度绝对P95，m/s² | 5.827806 | 6.613695 | **增加13.49%** | ≤6.410587，即增加≤10% | FAIL |

这些是本次开发的预设取舍阈值，不是通用车辆安全标准。没有为 jerk 新增隐藏门槛，也没有因原 G2 全通过而撤销转弯失败。

## 各窗口误差、速度与代价

下表使用全部主要窗口帧；本轮窗口内没有速度<2m/s的样本，但全路线停车及回退帧仍完整保留。辅助 moving 表也保留在 [metrics.csv](analysis/metrics.csv)。

| 窗口 | CTE RMS，m（B→C） | CTE P95，m（B→C） | CTE峰值，m（B→C） | 实际均速，m/s（B→C） |
| --- | --- | --- | --- | --- |
| 26966 右急弯 | .5587→.5164 | .8492→.7850 | .8565→.7894 | 7.8851→7.8869 |
| 24240 左弯 | .1078→.1010 | .1799→.1701 | .1826→.1718 | 7.8825→7.8826 |
| 17563 S1 | .2252→.2207 | .4670→.4543 | .5863→.5728 | 5.9661→5.9662 |
| 17563 S2 | .2734→.2688 | .4836→.5048 | .5617→.5985 | 6.0012→6.0012 |

S2 的 RMS 略降，但 P95 和峰值反而上升，只是尚未超过协议的绝对增量上限；不能写成所有误差指标都改善。

| 窗口 | 横向加速度绝对P95，m/s²（B→C） | emitted steer-rate 绝对P95，1/s（B→C） | 横向 jerk 绝对P95，m/s³（B→C，仅报告） |
| --- | --- | --- | --- |
| 26966 | 8.5982→9.1978 | .8001→1.0797 | 33.8112→39.5417 |
| 24240 | 1.9532→2.0288 | .2184→.3779 | 11.8918→12.0088 |
| 17563 S1 | 5.8278→6.6137 | 2.0000→1.9640 | 41.1735→58.4974 |
| 17563 S2 | 6.9071→6.9147 | 2.0000→2.0000 | 50.6602→62.2294 |

速度误差 RMS 增量全部≤.10m/s，均速下降全部≤.20m/s。6m/s处两种公式都落在3m下限，但实际速度会略过6m/s，且两次运行的定位/动力学轨迹并非逐帧相同，不能把 S 窗口视为严格相同控制输入的反事实实验。

## 入弯、出弯与停止行为

[metrics.csv](analysis/metrics.csv) 分别列出 entry/core/exit/post10m；[recovery.csv](analysis/recovery.csv) 保留删失记录。26966 的 post10m CTE RMS 从 **.3170 降到 .2856 m**，满足“不升”；但两组在该观察范围内均未观察到连续≥.5s同时满足 |CTE|≤.25m、|heading|≤5° 的恢复区间。未把删失写成0秒或通过。

24240 两组首次恢复时间均0s；17563 S1均.05s，S2从.70s变为.60s。这是协议固定的首次持续恢复诊断，不代表以后永久收敛。

共11帧 `trajectory_behind` 发生在停车阶段，原始 reference 的剩余投影弧长为0：26966 B/C分别3/2帧，17563 B/C分别3/3帧。这些帧输出的是协议已允许的安全制动，全部保留在全路线图和 [control-reasons.json](analysis/control-reasons.json)，没有归为非法控制，也没有删除。六例终点和5秒保持门槛均通过。

## 数据支持的机制线索，以及不能推断的部分

签名 CTE 统一为**左正**。26966右弯的正残差位于弯外侧，当前结果是外侧残差略减，不是已证明的“内侧切弯改善”。

独立的事后描述诊断 [per-window-replan.csv](replan-diagnostic/per-window-replan.csv) 使用同一固定窗口，定义轨迹更新为 `trajectory_frame` 相比前一连续帧变化。只检查同期关联，不改变任何验收条件：

- 24240 candidate有30/117帧更新轨迹，但最高10% emitted rate 的12/12帧都位于更新帧；baseline为11/12。较短前视组更多更新产生较大变化，但不能由此证明具体原因是 adapter。
- 26966 candidate有18/71帧更新轨迹，最高10%变化率的8帧中7帧位于更新帧；baseline为5/7。candidate出现1帧输出限制，baseline为0帧。
- 两个 S 的部分大变化也与更新同帧，且两组本来都有少量限速帧。所有主要窗口的已记录 rejoin curvature concern 计数为0；这不等价于动态可行性证明。

[最高五个变化事件](replan-diagnostic/top5-rate-events.csv) 保留原始 frame、station、raw/emitted、限幅、定位误差与更新标志。现有数据支持“较大控制变化集中于5Hz轨迹更新时刻”，但不能隔离 rejoin 几何、进度、定位噪声、短前视反馈之间的因果贡献。定位噪声是否被短前视放大仍是待证假设；本轮不再新跑或据此调参。

## 英文图与精确数据

所有图为未滤波原始信号；每张同时保存 PNG/PDF，图轴明确 CTE左正、steer/横向加速度右正。窗口时间分别以各自进入窗口为0，不声称两次运行 frame号相同；原始frame/time在CSV中保留。

| 图 | PNG | PDF |
| --- | --- | --- |
| 26966 固定右弯窗口 | [PNG](figures/route-26966-window-1.png) | [PDF](figures/route-26966-window-1.pdf) |
| 24240 固定左弯窗口 | [PNG](figures/route-24240-window-1.png) | [PDF](figures/route-24240-window-1.pdf) |
| 17563 第一个 S | [PNG](figures/route-17563-window-1.png) | [PDF](figures/route-17563-window-1.pdf) |
| 17563 第二个 S | [PNG](figures/route-17563-window-2.png) | [PDF](figures/route-17563-window-2.pdf) |
| 26966 全路线 | [PNG](figures/route-26966-full.png) | [PDF](figures/route-26966-full.pdf) |
| 24240 全路线 | [PNG](figures/route-24240-full.png) | [PDF](figures/route-24240-full.pdf) |
| 17563 全路线 | [PNG](figures/route-17563-full.png) | [PDF](figures/route-17563-full.pdf) |

三份逐帧精确绘图CSV留在 `/data/runs/b2d/controller/turns-v1-figures-v1/route-<id>-samples.csv`，原始大日志留在 `/data/runs/b2d/controller/turns-v1`。本目录复制的小结果共26文件、3,311,372字节（不含本说明和清单）；[复制映射](copied-artifacts.json) 验证源与副本哈希一致。旧 v1/v2 的8份CSV和全部旧图版均未修改。

分析与绘图产物分别位于 `/data/runs/b2d/controller/turns-v1-analysis-v1`、`turns-v1-figures-v1`，事后关联诊断位于 `turns-v1-replan-diagnostic-v1`。可复算代码：[冻结分析器](../../agents/turns-analysis-freeze-v2/analyze_turns.py)、[绘图](../../agents/plot_turns.py)、[更新关联诊断](../../agents/turns_replan_diagnostic.py)。

冻结 helper SHA-256：`8e0f3238bdb9af5d2969a7562a88f4701860cf602a4ad7b5d552efeda3f0ef64`。分析用 Python3.12.3/NumPy2.4.6；绘图用 `/data/envs/carla/bin/python`、Matplotlib3.7.5/NumPy1.23.5。manifest记录完整源哈希和输出哈希；本轮源与复制副本均再次核验。

真值 `validation_trace.json` 按 frame 与 control 连接，是 JSON 数组。速度误差=真值速度−参考速度；横向加速度为世界加速度点乘当前right_vector；jerk先在世界坐标对加速度按实际dt差分，再投影到当前right_vector，没有对旋转后的横向标量直接求导，也未事后滤波改善指标。相同frame的 `applied_control` 属于旧命令，没有当作当前新命令的执行误差。

根代理补充的 GPU 运行时采样因 server 已退出而为空；该空文件不能作为独立GPU验证证据。启动配置、源码和协议归档仍保留，此报告不新增运行时GPU实测结论。
