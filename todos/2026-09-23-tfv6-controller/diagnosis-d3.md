# TFv6 W2b：D3 定点机制诊断

2026-09-24。按 [`diagnosis-plan-d3.md`](diagnosis-plan-d3.md) 分两段：D3a 复查 W2/W2b 的 192 个 canonical case；D3b 对保留集 2084、27529 和 Dev10 2091 各三个 seed，重跑 B/C 与诊断混合臂 E/F。诊断臂不替换正式 W2b 对照，也不改变 B/C 的驾驶控制律。D3b 的原始逐 tick 日志、官方结果与 CARLA recorder 保存在 `/data/runs/b2d/tfv6-d3/`；这里的 CSV 与 contact sheet 为可核对的摘要。

## evaluator 路线与坐标校验

原预期 `_global_plan_world_coord` 是约 1 m 的 dense route，但传给 sensor agent 前，Bench2Drive 的 `leaderboard/autoagents/autonomous_agent.py:128–134` 调用 `downsample_route(..., 50)`，把 world/GPS plan 同步抽样。本任务三条路线各只给 TFv6 **4 个稀疏目标点**，目标点的跳变幅度因而可达十几至几十米。evaluator 自己在 `leaderboard/scenarios/route_scenario.py:120–121` 调用 `interpolate_trajectory(config.keypoints)`；该函数在 `leaderboard/utils/route_manipulation.py:136–161` 默认 `hop_resolution=1.0`，使用 `GlobalRoutePlanner(..., 1.0)` 从同一 XML 生成带 `RoadOption` 的 dense route。这些文件来自本次固定 Bench2Drive runtime，绝非把 XML 稀疏节点直连所得的近似。

[`b2d_tfv6_d3_dense.py`](../../scripts/b2d_tfv6_d3_dense.py) 在驾驶进程外各重建一次路线，保存 `/data/runs/b2d/tfv6-d3/dense/{level}-{route}.json`。它用每个稀疏世界路点及与之配对的 GPS 规划坐标拟合 planner→world 的平移、旋转与尺度。三路线的尺度与 1 相差 <`2e-11`，旋角 <`1.4e-11` rad，最大配对残差 <`2e-9 m`，远低于预设 0.1 m；主要平移 x 分别约 `−1416.009`、`+291.295`、`+2773.272 m`。目标点、Kalman 位置和 GPS 位置随后都输出为世界坐标。真实 agent 的稀疏目标点及命令也逐项对照重建的抽样点；诊断日志逐 tick 检查目标在 dense route 的 1 m 内及其 dense index 不回退。

| route | town | dense/稀疏点 | 已完成 W2 A/B case | 真值轨迹 ≤2 m 的 tick 比例 | 最大距离 |
|---|---|---:|---:|---:|---:|
| 2084 | Town12 | 79/4 | 6 | 各 100% | 1.808 m |
| 27529 | Town04 | 65/4 | 6 | 各 100% | 1.105 m |
| 2091 | Town12 | 62/4 | 5 | 各 100% | 1.068 m |

逐 case 的分数、最大与 p99 距离见 [`dense-validation.csv`](results/diagnosis/d3/dense-validation.csv)。按答复 12，ego 到 dense route 的距离是**驾驶结果**，逐 tick 记录首次 >3 m、最大值与官方终态，绝不作为遥测有效性 gate。答复 13 允许离线 I3 rear-waypoint 重算使用绝对 `1e-12 m`、相对 `0` 的比较，并记录各 case 最大残差；`1e-16 m` 扰动通过、`1e-9 m` 扰动失败。I1 phantom、I2、其余 I3 和混合臂组成检查保留。

## D3a：全量旧日志及 dense 分支修正

[`diagnosis-d3a.md`](diagnosis-d3a.md) 的全量扫描发现官方路线偏离仅在 2084/27529 的 C/D，六组各两个臂，共 **12/192**；≥5 s 停滞 **115 段**，散布在十条路线，不能把所有停滞归成 2091 的起步问题。2091 起步段的 C 首段隐含速度中位数约 `0.12–0.14 m/s`，8 点平均约 `2.12–2.38 m/s`；相同 tick 的 B shadow 大多给油，而 C 的 `stop_hold` 经常生效。离线替换期望速度可使同一输入多次要求起步，仅是控制律反事实，不代表闭环成功。

原 [`route-first-wrong.csv`](results/diagnosis/d3/route-first-wrong.csv) 以 XML 稀疏节点补直线弯道，几何误差很大。另一次交叉核对发现早期分析脚本还错误地翻转了 TFv6 预测的横向坐标；现已按作者 `inverse_conversion_2d` 所用 CARLA 局部旋转修正，并重生原 CSV。按答复 11，用真正的 evaluator dense 转弯分支重新计算了旧日志中模型 route/waypoint 末点到正支路及入口直行延长线的距离，保存在独立的 [`d3a-dense-branch.csv`](results/diagnosis/d3/d3a-dense-branch.csv) 与 [`d3a-dense-first-wrong.csv`](results/diagnosis/d3/d3a-dense-first-wrong.csv)。旧 XML 指标把 2084/C 的路口接近窗口约 step 197–201 判为偏直行；dense 指标在该窗口未达到“比正确分支近 ≥2 m”的门槛，首次满足出现在随后偏离前窗口约 step 385–430（2084/C seed 1 因长时间停滞为 step 3336）。27529/C 的 dense 首次 route 倾向直行约 step 379–426；匹配的 B 在两条路线都未触发此 dense 门槛。旧日志没有 target point、command 或 Kalman 状态，故 D3a 即使修正几何，也不能独立裁决 R1/R2。

## D3b：闭环因子实验

E = C steer + B throttle/brake，F = B steer + C throttle/brake；均经过相同的作者后处理与 brake 互锁。逐 case 官方终态/DS、目标 pop、首次离线、最大偏离和控制改写计数见 [`d3b-cases.csv`](results/diagnosis/d3/d3b-cases.csv)。每个官方偏离 case 的关键 tick、同 seed B 的同时间及同 dense-route 进度 tick 见 [`d3b-deviation-evidence.csv`](results/diagnosis/d3/d3b-deviation-evidence.csv)；首次 >3 m 前 10 s 的逐 tick 对照见 [`d3b-deviation-timeline.csv`](results/diagnosis/d3/d3b-deviation-timeline.csv)。同时间对照显示交通时钟，同进度对照更适合看转弯几何；两者不能当作完全相同的模型输入。

官方终态与 DS 总览见 [`d3b-factorial-outcomes.png`](results/diagnosis/d3/d3b-factorial-outcomes.png)；2084/27529 的 evaluator dense route 与四臂真值轨迹见 [`d3b-2084-paths.png`](results/diagnosis/d3/d3b-2084-paths.png)、[`d3b-27529-paths.png`](results/diagnosis/d3/d3b-27529-paths.png)。空心圆为首次离 dense route >3 m 的点，包括最终未被官方判路线偏离的短暂越界。所有图还提供同名矢量 PDF，均由 [`research/plot_style.py`](../../research/plot_style.py) 的版式导出。

### 2084：三个 seed

| seed | B | C | E（C 横向） | F（B 横向） |
|---:|---|---|---|---|
| 0 | 完成，DS 100 | 偏离，RC 50.76 | 偏离，RC 50.76 | 阻塞，RC 67.20；未官方偏离 |
| 1 | 完成，DS 60 | 偏离，RC 50.76 | 偏离，RC 50.76 | 完成，DS 100 |
| 2 | 完成，DS 100 | 偏离，RC 50.76 | TickRuntime，RC 49.13；路口前长停 | 完成，DS 4.93；曾短暂 >3 m 离线 |

三个 seed 的 C 都官方偏离，B 都完成。E 在 seed 0/1 也偏离；seed 2 在路口附近停到 4000 tick，最大离 dense route 仅 1.04 m，因没有驶过转弯，不能把该超时当作“横向 C 安全”。F seed 0 在转弯后阻塞，seed 1/2 完成，但 seed 2 DS 很低且曾短暂离线 5.93 m；因此 F 只提供**官方路线偏离未重现**的证据，不保证无其他失败。

**target/模型/控制时间线：** 四臂的 RoutePlanner 目标从转弯前稀疏点跳到下一个稀疏点时，ego 距 evaluator dense 转弯起点约 7.97–8.09 m，跨三 seed 基本相同；跳变目标始终落在 dense route 上，index 单调。seed 0 的 C 首次 >3 m 在 step 319，此时 target 为 dense index 44、command LEFT，模型 route 末点距正确/直行延长线分别 3.31/5.75 m，实际 steer −0.218；同 dense 进度 B 在 step 405，target 和 command 相同，route 末点距正确/直行分别 0.17/10.85 m，steer −0.401。C 的 route 末点直到 step 322 才首次达到“更靠近直行线 ≥2 m”，比真值首次离线晚 3 tick；E seed 0 的对应顺序是首次离线 step 537、模型 route 偏直行 step 538。seed 1/2 C 的规划偏直行与离线几乎同时出现（见 CSV），不能用终态附近一帧倒推最初原因。

**候选裁决（各 seed）：** R1 作为 C 相对 B 的主要差异被排除：所有臂在同一空间阈值附近 pop，未见 C 特有的提前/回退。R3 得到支持：C/E 的横向控制组合在能驶过转弯的 seed 0/1 一起偏离，B/F 的组合未在这两 seed 官方偏离；seed 2 的 E 因停滞无法检验这一对照。R2 作为*最初*的独立模型错误未获直接支持，但不能完全排除 waypoint 近端形状对 C steer 的影响；一旦 ego 离线，模型 route 随闭环状态继续偏直行，属于观察到的反馈。R4 对 seed 0 的 F 阻塞、seed 2 的 E 停滞和 F 低 DS 明显相关；它没有单独解释 C 三 seed 的一致偏离。录像的独立可见事实与限制在下文记录。

### 27529：三个 seed

| seed | B | C | E（C 横向） | F（B 横向） |
|---:|---|---|---|---|
| 0 | 完成，DS 100 | 偏离，RC 50.02，DS 24.01 | 偏离，RC 42.05，DS 25.23 | 完成，DS 30.85 |
| 1 | 完成，DS 100 | 偏离，RC 50.02 | 完成，DS 100 | 完成，DS 100 |
| 2 | 完成，DS 100 | TickRuntime，RC 55.69，DS 30.69 | 完成，DS 100 | 完成，DS 100 |

**没有一个单臂规律可覆盖三 seed。** seed 0 复制了 2084 的 C/E 偏离、B/F 完成格局，但 F 因其他违规只有 DS 30.85。seed 1 只有完整 C 组合偏离，E/F 都 100，说明这一次横向或纵向单独替换都不足以复制失败。seed 2 的 C 曾短暂离 dense route 至 4.46 m（step 441 首次 >3 m），最后在转弯后长时间停住而非官方偏离；B/E/F 都完成。官方终态与真值离线距离已分列，避免把 timeout 当作路线偏离。

四臂的稀疏目标跳到下一点时，ego 距 dense 转弯起点约 4.84–5.04 m；C 没有跨臂特有的大幅提前 pop。seed 0 的 C 首次 >3 m 在 step 374，target 仍是 dense index 31、command RIGHT，执行 steer +0.024；同路线进度 B 在 step 399，target/command 相同、steer +0.368。C 模型 route 末点直到 step 383 才首次比正确分支更靠近直行延长线 ≥2 m。seed 0 的 E 则在 step 521 已出现该模型指标，至 step 601 才首次 >3 m，说明“规划偏直行”与“真值离线”的顺序**跨诊断臂不同**。seed 1 的 C 两事件相隔仅 3 tick；seed 2 的 C 始终没有达到模型末点偏直行门槛。故单凭终态时刻的直线路径，无法一概断定是先规划错还是先跟踪出错。

预声明的 Kalman 输入隔离臂 K 仅在 R1 有迹象的组执行。上述 2084/27529 六组中，C 的 target index 均单调，跨臂 pop 的空间位置接近，且 target/command 始终对应 evaluator dense route；未见 C 特有的提前 pop 或回退，所以本次不启动 K。这个判据仅排除“导航 target 错跳是 C/B 差异主因”，不证明 Kalman 状态对其他模型输出完全无影响。

**候选裁决须逐 seed 看：** R1 作为 C/B 差异解释在三个 seed 都缺少支持（pop 空间位置几乎相同，index 无回退）。seed 0 的 B/F 与 C/E 因子对照及首离线 tick 支持 R3 横向跟踪贡献，但 E 的模型规划提前偏直行，使 R2 也保持未定；seed 1 的 E/F 均完成，支持两控制分支经闭环**共同**造成 C 的失败，而不支持单独归责横向或纵向；seed 2 官方路线偏离未重现，R2/R3 对该终态不适用，长停要结合 stop sign、灯和 actor 记录判为 R4 或控制律交互。R4 对 seed 0 的低 DS 与 seed 2 长停有影响，但现有局部录像与 actor 日志不足以把 seed 2 长停精确归因于某一个交通对象。

| route / seed | R1 目标 pop | R2 模型独立错支路 | R3 跟踪 | R4 其他交互 |
|---|---|---|---|---|
| 2084 / 0 | 排除作为差异主因：各臂同位置 pop | 未定：C 首离线比规划偏直行早 3 tick | 支持：C/E 官方偏离，B/F 未偏离 | 支持关联：F 后续阻塞 |
| 2084 / 1 | 排除作为差异主因：同位置 pop | 未定：缺少相同模型输入反事实 | 支持：C/E 偏离，B/F 完成 | 未定：B 虽完成但 DS 60 |
| 2084 / 2 | 排除作为差异主因：同位置 pop | 未定：C 偏离后闭环规划变化 | 未定：E 在弯前长停，未完成对照 | 支持关联：E 超时、F 完成但 DS 4.93 |
| 27529 / 0 | 排除作为差异主因：同位置 pop | 未定：E 的规划偏直行先于离线，但 B 不同输入 | 支持贡献：C/E 偏离，B/F 完成 | 支持关联：F 虽完成但 DS 30.85 |
| 27529 / 1 | 排除作为差异主因：同位置 pop | 未定：无同输入 B 模型反事实 | 排除单独充分性：E/F 完成，只有 C 偏离 | 未定：闭环组合仍影响终态 |
| 27529 / 2 | 排除作为差异主因：同位置 pop | 不支持超时解释：C 未触发偏直行门槛 | 未定：无官方路线偏离供因子复现 | 支持关联：C 转弯后长停 |

### 2091：起步停滞

| seed | B | C | E（C 横向） | F（B 横向） |
|---:|---|---|---|---|
| 0 | 完成，DS 100，442 tick | TickRuntime，RC 42.86，4000 tick | 完成，DS 100，543 tick | TickRuntime，RC 42.86，4000 tick |
| 1 | TickRuntime，RC 39.90，4000 tick | 完成，DS 60，556 tick | 完成，DS 100，493 tick | TickRuntime，RC 42.86，4000 tick |
| 2 | 完成，DS 100，437 tick | TickRuntime，RC 42.86，4000 tick | 完成，DS 100，446 tick | TickRuntime，RC 44.34，4000 tick |

[`d3b-2091-launch.csv`](results/diagnosis/d3/d3b-2091-launch.csv) 给出每臂前 1200 tick 的首段/8 点/作者期望速度、B/C shadow、原始与执行 control、stop sign、creep、灯及前方 actor 距离。所有 C/F 首次超过 0.5 m/s 均在 step 23，B/E 在 step 13，故没有一臂从零帧完全不动。图 [`d3b-2091-launch-speed.png`](results/diagnosis/d3/d3b-2091-launch-speed.png) 展示前三个 seed 的前 60 s 速度曲线。以下讨论的是起步后的重复停止和 4000-tick 终态。

| seed | S1 首段速度律 | S2 真实停车理由 | S3 后处理交互 |
|---:|---|---|---|
| 0 | 支持早期控制分歧：C/F 的低首段速度伴随 B shadow 给油；不足以单独解释终身停滞 | 支持有真实前车：C 在约 5.05 m 处遇近乎静止车辆；停车时长合理性未定 | 支持交互存在：C/F 后期分别 1344/1026 tick 纵向 control 被改写；未救出超时 |
| 1 | 排除“所有超时由 C 速度律充分导致”：B 也超时而 C 完成；F 的局部低首段仍可造成减速 | 支持有真实前车：B 有 3715 tick 在 10 m 内见前车；B/F 的相遇相位不同 | 支持交互存在：F 后处理改写 581 tick；是否造成超时未定 |
| 2 | 支持早期控制分歧：C/F 多次低首段而 B shadow 给油，B/E 完成、C/F 超时 | 支持有真实前车：C 在约 5.1 m 处遇近乎静止车辆；无法判定是否可安全绕过 | 支持交互存在：C/F 改写 261/63 tick；不构成充分原因 |


seed 0 的 B/E 完成，C/F 都在 4000 tick 上限终止，显示 C 的纵向分支与该 seed 的超时关联。四臂都曾起步：B/E 首次 >0.5 m/s 在 step 13，C 在 step 23，F 也在前 30 tick 运动。C 在 step 100 车速约 5.04 m/s，故本次的 4000-tick 失败不是从第零帧连续刹车。C 在 step 100–400 的首段隐含速度中位数 `0.239 m/s`、8 点平均 `2.261 m/s`；其 300 tick 中 B shadow 有 218 tick 请求油门，C shadow 有 96 tick 请求刹车，支持 S1 在这一段造成控制分歧。C 在 step 400–800 的 400 tick 中有 277 tick 为 `stop_hold`，step 800–1200 全 400 tick 都是 `stop_hold`；但后一个窗口的 8 点平均速度中位数也仅 `0.242 m/s`，不能套用前一窗口“中远端要求走”的解释。其后长期停在约 `(2771.4,1601.4)`，正前方约 5.05 m 有近乎静止的 `vehicle.chevrolet.impala`，actor 遥测记录相对速度约 0.01 m/s。这支持 S2 存在真实障碍，尚不能判定停多久才合理。交通灯状态虽多次为 Red，`at_light=False`，不能把红灯当作已确认的停车原因。后处理 creep 在 step 1200 后多次把 C 原始 `brake=1` 改写为执行 `throttle=0.4`，但车仍停滞，需把 S3 的控制改写与最终可行驶性分开判读。B/E 的不同轨迹和交通遭遇不能被视为同一障碍物的严格反事实。

seed 1 出现相反的因子格局：B/F 均在 4000 tick 超时，C（DS 60）/E（DS 100）完成。B 轨迹日志里的 C shadow `stop_hold` 只有 40 tick，却有 3715 tick 的 10 m 内前方车辆记录，长期停在约 `(2771.94,1604.30)`；因此此 seed 的 B 超时不能归咎 C 首段速度律。F 也长期停在另一处约 `(2771.24,1601.13)`，其 C 纵向分支仍有大量 `stop_hold`。四臂在独立闭环中所遇车辆及相位有差别，不能把 B/F 的共同超时直接解释为 B 横向控制的确定效应；但它已排除“S1 足以解释所有 2091 超时”的统一说法。


seed 2 与 seed 0 的终态格局相同：B/E 分别 437/446 tick 完成，C/F 都在 4000 tick 超时。C 在 step 100–400 的首段隐含速度中位数为 0.24 m/s、8 点平均为 2.31 m/s；B shadow 有 230/300 tick 请求油门，C shadow 有 97/300 tick 请求刹车。C 随后驶近停在正前方约 5.1 m 的 Chevrolet Impala，step 800 和 1200 的相对速度接近零，在同一位置停至上限。F 也在路口后方长期停住，虽在前 1200 tick 多次短暂前进，终态仍未完成。C/F 的纵向后处理改写分别为 261/63 tick，说明 S3 确实发生，但两者停滞不能只由后处理次数解释。


## 录像与逐帧对照

以下各行均是同 seed 的 B/C 追车录像，叠加 TFv6 route/waypoint、目标点、command 和执行 control；contact sheet 为每臂三个时刻的 2×3 缩略图，单张均 <500 KiB。2084/27529 的中心时刻取该 seed C 首次超过 dense route 3 m 的时刻；2091 固定在起步后的 10 s 路口窗口。录像文件留在 /data，缩略图纳入仓库。

| route / seed | 中心 s | contact sheet | B 录像 | C 录像 |
|---|---:|---|---|---|
| 2084 / 0 | 16.00 | [图](results/diagnosis/d3/d3b-2-2084-0-contact-sheet.png) | [B](/data/runs/b2d/tfv6-d3/render/2-2084-0-B/window.mp4) | [C](/data/runs/b2d/tfv6-d3/render/2-2084-0-C/window.mp4) |
| 2084 / 1 | 19.35 | [图](results/diagnosis/d3/d3b-2-2084-1-contact-sheet.png) | [B](/data/runs/b2d/tfv6-d3/render/2-2084-1-B/window.mp4) | [C](/data/runs/b2d/tfv6-d3/render/2-2084-1-C/window.mp4) |
| 2084 / 2 | 21.75 | [图](results/diagnosis/d3/d3b-2-2084-2-contact-sheet.png) | [B](/data/runs/b2d/tfv6-d3/render/2-2084-2-B/window.mp4) | [C](/data/runs/b2d/tfv6-d3/render/2-2084-2-C/window.mp4) |
| 27529 / 0 | 18.70 | [图](results/diagnosis/d3/d3b-2-27529-0-contact-sheet.png) | [B](/data/runs/b2d/tfv6-d3/render/2-27529-0-B/window.mp4) | [C](/data/runs/b2d/tfv6-d3/render/2-27529-0-C/window.mp4) |
| 27529 / 1 | 21.40 | [图](results/diagnosis/d3/d3b-2-27529-1-contact-sheet.png) | [B](/data/runs/b2d/tfv6-d3/render/2-27529-1-B/window.mp4) | [C](/data/runs/b2d/tfv6-d3/render/2-27529-1-C/window.mp4) |
| 27529 / 2 | 22.05 | [图](results/diagnosis/d3/d3b-2-27529-2-contact-sheet.png) | [B](/data/runs/b2d/tfv6-d3/render/2-27529-2-B/window.mp4) | [C](/data/runs/b2d/tfv6-d3/render/2-27529-2-C/window.mp4) |
| 2091 / 0 | 10.00 | [图](results/diagnosis/d3/d3b-1-2091-0-contact-sheet.png) | [B](/data/runs/b2d/tfv6-d3/render/1-2091-0-B/window.mp4) | [C](/data/runs/b2d/tfv6-d3/render/1-2091-0-C/window.mp4) |
| 2091 / 1 | 10.00 | [图](results/diagnosis/d3/d3b-1-2091-1-contact-sheet.png) | [B](/data/runs/b2d/tfv6-d3/render/1-2091-1-B/window.mp4) | [C](/data/runs/b2d/tfv6-d3/render/1-2091-1-C/window.mp4) |
| 2091 / 2 | 10.00 | [图](results/diagnosis/d3/d3b-1-2091-2-contact-sheet.png) | [B](/data/runs/b2d/tfv6-d3/render/1-2091-2-B/window.mp4) | [C](/data/runs/b2d/tfv6-d3/render/1-2091-2-C/window.mp4) |

**录像直接可见：** 2084 的三个 C 窗口都显示车以较高速度穿过左转口、继续接近入口道路的直行方向；相同时钟的 B 常仍在路口前等待横向车流，随后按指定左转进入支路。27529 seed 0/1 的 C 在右转口继续靠近直行道路，而 B 正在或已经右转；seed 2 的 C 曾短暂驶出 dense 路线附近，随后实际右转并在交通中变慢，与其官方 TickRuntime 而非官方路线偏离一致。2091 的三组 10 s 窗口可见 stop sign 与穿行车辆，B/C 都出现减速，C 的刹车条在部分帧亮起。录像是 4.4 s 的局部窗口，不能显示 2091 数千 tick 后是否仍有前车；该部分依据逐 tick actor 距离和相对速度记录。

**由录像之外的遥测作出的推断：** 2084 的 C/E 官方偏离与同空间位置 target pop、C 较弱转向以及首离线早于或贴近模型偏直行的顺序一起支持 R3 贡献；27529 仅 seed 0 有相近的清晰因子格局，seed 1/2 需要保留闭环耦合或停滞解释。2091 的首段速度与 shadow control 支持 S1 早期分歧，长期前车支持 S2 的真实障碍，后处理改写支持 S3 交互存在；单靠录像的路口画面不足以裁决后三者谁单独足以导致超时。18 个视频均为 960×540、44 帧，录像轨迹与真值的第 60 百分位对齐残差报告值为 0 m；九张缩略图最大 340329 bytes。

## 有效性、偏离与限制

正式 B/C 的驾驶路径继续使用 cb0357c 的控制分支；E/F 仅是显式开启的诊断臂，不进入 W2b 正式对照。36/36 个 D3b case 均有官方终态、逐 tick 日志和 recorder，并通过 I1–I3、混合臂组成、导航目标语义及坐标对齐检查。离线 rear-waypoint 重算的跨 case 最大残差为 8.88e-16 m，55 tick 出现非零浮点舍入残差，均低于答复 13 的绝对 1e-12 m 门槛。诊断专用 30 个单元测试通过。D3a 先于 D3b 提交；D3b 开始前按 W2b 计时预估全流程 2.5–3.0 h，36 个驾驶 case 实际墙钟 6003.5 s（约 100.1 min，双 CARLA worker）。

按答复 10–13 和运行核验，实际方案调整如下。答复 10 允许在任何 case 驾驶前修复 LEAD/leaderboard/scenario_runner 的导入路径，并加了与真实启动方式相同的 smoke test。答复 11 确认 agent 只收 4 点稀疏计划，因此离线调用 evaluator 自己的插值函数重建 dense route，且使用配对 world/GPS 路点拟合坐标系；旧 XML 直连分支指标不再用于裁决。答复 12 删除把 ego 离 dense route 超过 3 m 当硬 gate 的误设，保留为驾驶结果指标。答复 13 只放宽离线 I3 rear-waypoint 数值比较，零相对误差、绝对 1e-12 m，并通过 1e-16/1e-9 扰动正反单测。D3a 的世界坐标投影还修正了预测横向符号，重生旧日志分支 CSV。中断的预修复尝试归档在 /data/runs/b2d/tfv6-d3/aborted/ 下，最终 36 个有效 case 均为新预算。录像重放在 Town12/Town04 切换时需按路线使用新 CARLA server，视频窗口也按本次 C 首次超过 3 m 的 tick 调整；这些只影响离线展示，不触碰驾驶路径。

这些因子臂是各自独立的闭环驾驶，尽管使用同 seed，同一 tick 的车速、交通相位、周围 actor 和模型输入未必相同。同 seed B 同路线进度对照也不是严格同输入模型反事实。因此 R2 的“B 也会在完全相同情境规划错”在本实验中一般不能直接验证；R3 的证据是控制组合与首离线时序，不是对独立交通随机性的消除。模型 route 末点到正确/直行线的 2 m 门槛是可核对的几何代理，而非 evaluator 官方违规规则。报告始终把官方终态、DS/RC、真值首次 >3 m 与录像所见分别记录。


## 唯一下一步建议

在固定 NPC 时空轨迹、信号灯相位和初始状态的可重复 CARLA 场景中，对 27529/seed 1 与 2091/seed 1 重做同一 B/C/E/F 闭环因子实验，并按 evaluator dense-route 进度配对模型输入及控制。两组恰好展示目前最难区分的横纵耦合与交通相位反转；控制交通后，才能更有力地判定 R2/R3 和 S1/S2 哪些差异由控制律本身产生。此处仅提建议，本轮不实施。
