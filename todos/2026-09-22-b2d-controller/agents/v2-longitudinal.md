# v2纵向诊断：6m/s附近的反馈与传动循环

2026-09-22；只读完整v1/v2/匹配速度诊断数据。未修改controller/agent、未启动CARLA。核心结论：**6m/s直路也失败，稳定目标下仍有显著油门/制动循环，并伴随API报告的挡位反复变化；问题不能只归于S弯、目标速度抖动或GNSS原点桥接。** 单个低增益PI候选值得在开发集测试，尚无证据支持固定前馈映射或强制挡位。

## 可复算数据与不变口径

- `v2-longitudinal-analyze.py` → [v2-longitudinal-data.json](v2-longitudinal-data.json)：v1真实S、v2真实S与v2直路8m/s，包含完整制动pulse和松刹后谷值事件、稳定目标子集、原始SHA256。
- `v2-longitudinal-gear-analyze.py` → [v2-longitudinal-gear-data.json](v2-longitudinal-gear-data.json)：新6case匹配诊断，包含每次API挡位转移、前一tick控制一致性、分挡位速度/加速度和raw SHA256。
- 旧raw位于`/data/runs/b2d/controller/development-s-v1/`、`development-v2/`。新增匹配诊断位于`development-v2-speed-diagnostic/`，6例均completed、无基础设施失败，全部保留。

速度gate完全复用旧validator：独立真值参考≥cruise−0.1、elapsed≥5s，误差为真值speed减固定cruise。所有helper重算RMS与对应validation.json差小于1e-12。后文稳定目标子集/pulse窗口是诊断，不改变验收gate。闭合六例的[原summary、精确CSV与口径核验](../results/v2-speed-diagnostic/README.md)已归档。

精度更正：先前快速消息中的1773/TCP `0.61155`不是官方结果，也不是正确四舍五入；正式值为`0.6116222639146198`。同一gate行用control的signed speed得到`0.6116799703752411`，用3D truth减动态reference得到`0.6115065143563292`。这说明约1e-4差异可能来自速度/参考定义，不能笼统归因float datatype；旧快报值没有足够证据追溯唯一原因，不继续使用。最终helper固定使用3D truth减固定cruise，并逐例断言匹配原验证结果。

## 目标已基本稳定，6m/s失败仍存在

原v1 S的CARLA/pursuit速度RMS为0.6633/0.6363m/s；去除原点桥接后的首次v2 S为0.6389/0.6444m/s，max-lookahead pursuit为0.7547m/s。

v2 S CARLA巡航320tick的target中位5.999966、p95约5.999999、p05约5.955m/s；少量更低值来自几何弦长/接近减速边界。进一步只取`|target−6|<0.01`，CARLA/pursuit RMS仍为0.6321/0.6485m/s。目标基本恒定仍失败，因此不能用目标变稳就声称纵向问题解决。

近零speed归一化仅在|speed|<0.01m/s工作，这些失败样本在5–6m/s附近，与该安全边界相隔几个数量级；全部巡航reason是tracking，没有invalid_motion/stale。

### 匹配6m/s诊断：不是S弯独有

根代理新增只读`actor.get_control()`记录，放在本tick调用agent/应用新命令之前；runtime controller未变。以下两条路线均配置6m/s，三个preset各完整一例：

| route / shape | CARLA速度RMS m/s | TCP速度RMS m/s | pursuit速度RMS m/s |
|---|---:|---:|---:|
| 1773 / straight | 0.62849 | 0.61162 | 0.69169 |
| 17563 / true S | 0.63028 | 0.62236 | 0.67315 |

六例均高于0.5门槛。相反，已完成的相同1773直路@8m/s CARLA/pursuit RMS为0.22769/0.22308m/s。直路6也复现，支持速度工作区间相关的plant/反馈问题；S弯会影响具体波动，但不是必要条件。

## 实际时间顺序：API控制已更新，但速度仍滞后

首次v2 S CARLA/pursuit各14次制动pulse，中位长度0.25s。速度峰值出现在第一次制动命令之后中位0.10s；松刹之后仍继续降速，中位0.35s后到谷值，额外下降0.961/0.936m/s。两者各有15段持续≥0.25s的“满油门、零刹车、车速仍低于target0.5m/s以上”。

直路8m/s也不是完全平滑：CARLA/pursuit各21个较短pulse，中位0.15s；松刹后谷值延迟0.20s、额外下降约0.332/0.317m/s，没有上述长时间满油门低速平台。这说明新候选应降低控制循环幅度，并保留8m/s已经达标的表现，而不是只让6m/s单条看起来变好。

**这些pulse时差不是测得的纯制动执行器delay。** 它包含已知的一tick命令/观测边界、传动系统、车辆惯性、输入平滑和轮胎响应，不能直接把0.35s写成brake delay参数。

新增6m/s诊断的API throttle/brake/steer与上一tick命令逐项最大差约3e-8，符合float32量化。这排除了“harness把旧刹车又hold了数tick”作为主要解释；不证明底层制动力与API pedal即时成正比。

### API报告的挡位确实在运动中循环

在1773/CARLA满足巡航gate的384tick内，API gear1/2/0分别170/138/76个样本，约17个`2→0→1→0→2`循环。其他五例也反复出现同一转移序列。**gear0样本出现在5m/s左右，不只是停车时的0挡。** 这里按API报告称呼，不推断内部离合器状态、真实发动机RPM或轮胎滑移。

一段1773/CARLA raw时间序列（与本tick新命令不同，API列是刚结束物理步使用/返回的control）：

| elapsed s | speed m/s | API gear | API throttle / brake | 当前新throttle / brake |
|---|---:|---:|---|---|
| 5.65 | 6.157 | 2 | .75 / 0 | 0 / .50 |
| 5.75 | 6.350 | 2 | 0 / 1 | 0 / 1 |
| 5.90 | 5.984 | 2 | 0 / .56 | .07 / 0 |
| 6.00 | 5.782 | 2 | .48 / 0 | .75 / 0 |
| 6.05 | 5.798 | 0 | .75 / 0 | .73 / 0 |
| 6.15 | 5.388 | 1 | .75 / 0 | .75 / 0 |
| 6.35 | 5.167 | 1 | .75 / 0 | .75 / 0 |
| 6.60 | 5.118 | 0 | .75 / 0 | .75 / 0 |
| 6.70 | 5.783 | 2 | .75 / 0 | .75 / 0 |
| 6.85 | 6.351 | 2 | 0 / .89 | 0 / 1 |

因此有实测支持“强反馈踩刹车、降速跨传动工作区、恢复又超调、再次制动”这个循环形态；不能再只说看起来像gear hunting而没有数据。
不过这仍不是反事实因果实验：没有锁挡对照，也没有发动机RPM/离合器torque日志，不给挡位控制加特判。

## 坡度与物理参数能解释多少

首次v2 S CARLA巡航body pitch绝对p95约0.342°、最大0.418°；即使全部当纵向坡度，对应重力分量最多约0.072m/s²。
同期真值速度差分加速度p05/p95约−5.42/+8.48m/s²，最大单tick增速0.667m/s（约13.33m/s²）。悬架动态pitch和路面坡度未完全分离，但记录不支持大坡度是这些脉冲的主要来源。

stock physics只说明存在自动变速：`use_gear_autobox=true`、gear_switch_time0.1s、first/second ratio4.584/2.964、final_ratio3.21，
轮半径0.355m、maxRPM6500、up/down ratio0.46/0.23。锁定无滑移假设下，二挡1495RPM对应约5.84m/s，确实靠近6m/s工作区；
这是参数相容性核查，**不是实际换挡阈值标定**，不用于实现依赖5.84的控制分支。

## 一个值得测试的有限候选

当前CARLA纵向`WindowPID(1.0,0.05,0,10,dt=.05)`的error是km/h，随后直接分配normalized throttle/brake。
忽略小I项后，m/s error实际比例系数为3.6；只要超速约0.278m/s就请求满刹，欠速约0.208m/s就触及0.75油门上限。
10样本滑动I在常值误差下只保留0.5s历史：m/s尺度的I贡献约0.09×error，不能像持续积分那样消除稳态负偏差。
它没有严重长时间积分windup的证据，**原故障不能归咎windup**；antiwindup是新true-PI所需的工程边界。

建议只增加一个显式实验纵向模式（例如`longitudinal=pi_soft_v1`），默认和原CARLA/TCP实现不改：

```text
error = target_speed_mps - observed_speed_mps
I_try = I + error * dt
u_raw = 1.0 * error + 0.25 * I_try
accept I_try only if -1 <= u_raw <= 0.75,
    or u_raw > 0.75 and error < 0,
    or u_raw < -1 and error > 0
u = clip(1.0 * error + 0.25 * I, -1, 0.75)
throttle = max(u, 0)
brake = max(-u, 0)
```

Kp=1.0的单位为normalized pedal/(m/s)，Ki=0.25为normalized pedal/m；I是speed error的时间积分（m）。
**u是signed normalized pedal demand，不是m/s²加速度**。这样明确单位，避免继续把PID输出名为acceleration却实际直接写pedal。
比例反馈约为旧版本的28%，积分时间尺度Kp/Ki=4s，慢于约1s的循环，让静态负偏差缓慢补偿而不每次立即从满油门切满刹。
这些是预先选定的单个试验参数，不是拟合出的最优值，也不保证通过。

- 每route/reset清I；invalid/stale时不积分；进入原stop_hold时清I并保留原满刹保持动作。正常trajectory update不清I。
- 保留现有throttle≤.75、brake≤1及互斥；不降低故障/停车制动上限，不读取API gear/真值来控制。
- 不同时添加deadband、纵向slew、feedforward、档位覆盖或速度分段gain。否则不知道改善来自哪一项。
- 没有独立throttle/brake→acceleration标定支持从当前闭环拟合可靠前馈：观测中同样满油门既会掉速也会强加速，gear相位高度混杂。
  因此这轮先不引入“测得plant前馈”。若要用明确加速度输入，应另做固定gear/速度的plant辨识，不能拿理想3/8m/s²比例当真实标定。

## 已批准的运行边界：固定18例开发矩阵

根代理已批准实现Kp=1.0、Ki=0.25的单个PI候选。为复用同地图server，原建议的“三例先筛选”被完整18例一次运行取代；没有扩大gain搜索范围。[预声明协议](../results/v3-inputs/protocol.json)在结果前固定。

- 三列：PI+CARLA、PI+pursuit additive、PI+pursuit max。各自对照旧相同横向/adapter列，只切换纵向模式。
- 六组：1773@8、1773@6、17563@6、24240@8、26966@8、25854@8，共18例；优先Town12直路8、直路6、S6。新增直路6是额外必过probe，不混入原五路线等权横向均值。
- 单元/离线先验证PI状态、antiwindup、reset、stop_hold/stale、互斥/限幅及原停车语义；保留全部raw、API gear与本tick/前tick控制。
- 每例使用原完整G2 gate：独立速度RMS、横向、pose、停点、hold等，不能以换挡减少替代误差门槛。新增舒适性原始字段只用于后续独立分析，不追改既有gate或旧分数。
- 任何既有通过case变失败或6m/s问题未解除，都保留并报告；不随即追加Ki/Kp网格。若单候选通过全部开发门槛，再考虑固定第二seed，Dev10/holdout不用于调参。

本分析不宣布新默认。是否替代、lookahead如何选择，由完整预声明矩阵决定，不能挑选某次停车时刻或合并新旧有利结果。

## 评分边界：官方DS、Driving Smoothness与开发gate分开

B2D并非只罚碰撞。当前0.0.4的[官方统计源码](https://github.com/Thinklab-SJTU/Bench2Drive/blob/7ec25d1c9f7522d923ce5f3420986cef1cb2d956/leaderboard/leaderboard/utils/statistics_manager.py)还处理红灯、停车标志、场景超时、紧急车辆让行、出界等；Driving Score由route completion乘infraction penalty得到。本版本MinimumSpeed标记unused，不能自行把它算成DS扣分。

官方另有[Driving Smoothness脚本](https://github.com/Thinklab-SJTU/Bench2Drive/blob/7ec25d1c9f7522d923ce5f3420986cef1cb2d956/tools/efficiency_smoothness_benchmark.py)，检查加速度、jerk与yaw类指标，输出满足条件的片段比例，未直接进入上述DS公式。当前G2的speed RMS≤0.5是自建跟踪验收指标，**不是官方舒适性指标**；油门/刹车循环减少也不自动等于官方smoothness提高。

已直接核查两点版本限定的实现疑点：`_z_yaw_acc`的Savgol调用未传导数参数；metric_info直接记录CARLA angular velocity，其[API单位为deg/s](https://carla.readthedocs.io/en/0.9.15/python_api/#carla.Actor)，而脚本使用标注rad/s的阈值且未见单位转换。只记录源码观察，不改vendor、不静默修分。[本地源码快照、哈希及官方固定commit链接](../results/scoring-source-observations/README.md)可复核；远程逐字节hash因HTTP403未核验。

后续trace补完整真值运动学字段后，可分别报告“指定版本官方脚本结果”和“明确SI单位/时间步的物理诊断”。旧run缺少完整向量，不能靠speed差分无损补回官方原始metric_info；旧G2结果继续保持原口径。
