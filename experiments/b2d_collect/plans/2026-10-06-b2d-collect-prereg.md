# b2d_collect：B2D 版 op_parity P2 的模仿数据采集（预登记，2026-10-06，冒烟之前写定；括号内的数在测完后补）

## 目的

给 op_parity P2（Cinque + ego / pose / route-command adapter，第 144 条；+ 可行驶区 SDF hinge，第 148 条；action head 联合训练在讨论中）造一份
CARLA / Bench2Drive 上的模仿学习数据：PDM-Lite 开车，openpilot road + wide 两路相机按我们对齐的机位（第 127 条：1.86 m，与开环榜一致）以
原生 20 Hz 录下，存成 Cinque 实际吃进去的模型帧；同时记 ego、路线指令、专家轨迹与控制、地图与他车 / 红绿灯。规模对标 Bench2Drive-base
（~1000 段）。路口转弯加权（第 121、127、128、133、134、137、147、148 条的未决问题）。不训练；只造数据并验证它能原样进 P2 的输入管线。

## 不重测的既有结论

- 机位：B2D `spec` 相机（x 1.59 m 在后轴前、z 1.86 m、水平），第 127 条；宽角就是 openpilot 自己的 wide 传感器（f 567，118.9°），CARLA 里直接渲染，
  不需要三路拼接（那条规矩针对 NAVSIM / HUGSIM 的现成相机）。
- 打包：闭环 policy server 的 `OpenpilotModel.pack`（BT.601 limited YUV，modeld 最近邻 warp，色度 2×2 均值），采集端逐位复刻
  （`tests/test_b2dc_frames.py` 已对比：逐位相同）。
- 20 Hz：`op_camera_tick 0.05` 是闭环默认（road / wide 同帧，5 Hz 上下文步恰为 4 帧），所以 0.2 s 帧对（step s − 4, s）无需插帧。
- 专家：PDM-Lite 原样（SimLingo 的 B2D 副本，DS 97.0），`scripts/b2d_expert_agent.py` 同一路径；GPU 池、reduced 线程档、端口由池分配。

## 路线（`scripts/b2dc_routes.py build`，CPU，离线 carla.Map；已跑，`$DATA_DIR/runs/b2d_collect/routes/v1/`）

一条路线一个场景（Bench2Drive 的 clip 单位），三个来源：

| 来源 | 内容 | 候选 | 留下（hold-out 后） |
|:--|:--|--:|--:|
| LB | CARLA LB2.0 长路线（routes_training Town12，38 型；routes_validation Town13）每个场景，按 SimLingo `split_route_files.py` 的前 / 后距离截取（前 ≥ 30 m，后按其表 + 10 m），两端移出路口；场景元素（含参数）原样 | 6415 | 5508 |
| JT | 12 个 B2D 城镇所有路口的（路口，进入道路，驶出道路）连接：进口前 15 / 30 / 50 m（随机），出口后 35 m；规划路径必须真走这条连接；按信号灯 / 停车牌 / T 字 / 转向分配一个合适的路口场景型（含 8 个 B2D 独有型：Vanilla*、T_Junction、*LeftTurnEnterFlow），参数取同型 LB 实例（B2D 独有型取 bench2drive220 实例的参数值，不取位置） | 5380 | 4547 |
| SLC | SequentialLaneChange：多车道路 125 m，车道 i → 相邻 → i | 146 | 125 |

**评测 hold-out**（对 bench2drive220 与 bench2drive_0.0.4_val 的每条路线，同一城镇内）：(a) 经过任何评测路线经过的路口 → 删；(b) 触发点在
任一评测触发点 50 m 内 → 删；(c) 路径有 > 20 m 落在评测路径 3 m 内 → 删。删除计数：LB 路口 460 / 触发点 282 / 路径 165，JT 路口 758 / 路径 75，
SLC 路径 3 / 触发点 18（overlap.json）。B2D 的 220 条里 104 条在 Town12、47 条在 Town13，正是从 LB2.0 路线上截的，所以 (b) 删掉的 282 个 LB
场景就是评测原位置。规则 (a) 保证 op_route_ft 的 25 个 B2D 路口转弯所在路口一个都不进训练。

**选择**（n = 1000）：每个 B2D 型先给 min(12, 可用)；然后路口转弯（路口内出现 LEFT / RIGHT RoadOption）补到 55%，左右交替、城镇按
bench2drive220 的城镇分布加权轮转；剩下按场景型轮转。一个路口连接只用一次。天气用 SimLingo 的随机档（每条路线一个种子）。

结果：1000 条，46 型（B2D 44 型全有 + LB 独有的 EnterActorFlowV2 / PriorityAtJunction），路口转弯 60.6%（左 308 / 右 298），城镇 Town12 547、
Town13 238、其余 10 镇 215；平均长 112 m（p10 68，p90 188；JT 84 m，LB 134 m），合计 112 km；夜间 28.8%。登记为 split
`b2d/b2dc-train@v1`（unit route）。不足 12 的型：EnterActorFlowV2 3、MergerIntoSlowTrafficV2 9、HighwayExit 11（hold-out 后可用的就这么多）。

**clip 长度**：一条路线一段，跑到路线完成（或 leaderboard 超时）。不再切：P2 的样本只需 t0 前 1.6 s 帧（9 个上下文槽）与 t0 后 4 s 标签，
整段里任一 t0 都可取；按 112 m、PDM-Lite 均速估 ~25–35 s 一段（冒烟后补实测）。

## 每 tick（20 Hz）存什么（`scripts/b2dc_agent.py`，attempt 目录下 `clip/`）

| 文件 | 内容 |
|:--|:--|
| frames.mp4 | 模型帧：每 tick 一张 512×512 yuv420p（上半 road、下半 wide；即模型的 YUV420 平面，无颜色转换），libx264 crf 0（无损，测试里逐位还原） |
| chase.mp4 | 第三人称 480×270，每 4 tick 一张（只给人看，模型看不到） |
| ego.npz | t、frame、actor 位置 / 姿态、世界系速度 / 加速度 / 角速度、速度、前左轮转角、实际控制与专家控制（DAgger 时不同）、driver、PDM-Lite 内部量（目标速度、路口 / 停车牌 / 行人标志、它真正跟的 64 m 剩余路径）、影响自车的灯 id / 状态、限速、帧索引 |
| route.npz | leaderboard 给 agent 的稠密路线（x、y、z、yaw、RoadOption） |
| actors.npz + kinds.json | 80 m 内每辆车 / 行人每 tick 的 (id, xyz, yaw, 速度)，静态道具每 20 tick；id → 类型、bbox |
| lights.json + lights.npz | 路线附近的灯（位置、触发区、停止线航点、组）、停车牌；120 m 内灯每 tick 的状态 |
| scen.jsonl | PDM-Lite 看到的 active_scenarios 变化 |
| meta.json | 城镇、天气、机位（传感器规格、内参、模型 K、mount、后轴）、车辆尺寸、配置、计时 |

足以复现：路线 XML + TM 种子 + 天气 + 机位 / 内参 / 打包定义都在；原始 1928×1208 不存（两路 20 Hz 原图约 700 GB）。闭环本身不逐位确定，所以「复现」指同条件重采。

## 标签（`scripts/b2dc_labels.py`，CPU，`lib/b2dc_labels.py`；与 op_parity 同约定）

右手系、后轴、t0 车体系（x 前、y 左、yaw 左正）：fut (8, 3) 0.5…4 s；hist (4, 3) −1.5…0 s（开头之前重复首帧，同 WA-JEPA / HUGSIM 侧）；
vel / acc (4, 2) 各自车体系；cmd 四维 NAVSIM one-hot（`route_command`：前方 30 m 内出现路口 LEFT / RIGHT → 左 / 右，否则直；同时存
下一个转弯的类型与距离，换前瞻距离不必重采）；ego (20,) = `parity_adapter.ego_features`；route_poly (16, 2) 10 m 顶点（op_route_ft 口径）；
动作标签 act_kappa / act_accel = t0 + 0.2 s（openpilot lateralDelay）处的实际曲率（航向率 / 速度，±0.1 s 平滑）与纵向加速度；
专家控制、轮转角、PDM-Lite 路径（t0 系）。SDF：op_probe 栅格（x −8..56、y −24..24、0.5 m、内正），可行驶 = CARLA 车道类型
Driving / Parking / Bidirectional 的车道多边形，0.25 m 世界栅格上 EDT，每 2 tick 一张；另存 MKZ 外廓（hinge 的 CORNERS 现在是 Pacifica，
训练时换成 `footprint`）。

## 检查（`scripts/b2dc_check.py`，GPU；每个阶段都跑，门槛写死在 `b2dc_lane.py`）

1. 完整性：picture 序号 = tick、帧号连续、|dt − 0.05| < 1e-4、冻结帧 < 1%。
2. 对齐（核心）：按 op_parity 的 0.2 s 协议取 8 个上下文槽（帧对 (t0 − 4k − 4, t0 − 4k)），过 Cinque 冻结编码器 → P0 与 P2-F-s0 →
   8 个后轴位姿。**帧滞后测试**：P0 的 ADE 在帧栈整体平移 −8 / −4 / +4 / +8 tick 时都应高于 0 偏移（帧与标签时间对齐）；**转向符号**：
   |记录的 4 s yaw| > 20° 的样本上，P2 规划的 yaw 符号一致率 ≥ 0.8（镜像 / 坐标错会掉到 ~0）。另报 ADE、4 s 路长比（速度尺度）。
3. 标签：记录的未来 4 s 车身四角在 SDF ≥ −0.3 m 的比例 ≥ 0.9（栅格坐标对齐）；路线指令与之后 10 s 实际转向（> 30°）一致率 ≥ 0.8。
4. 地平线：车身俯仰 p95 → road 帧地平线偏移行数；GIF 上画标称地平线行（47.6 / 151.8）和投影到地面的记录未来（绿）/ P2 规划（红）。
5. GIF：第三人称 + 两路模型实际输入，中文字幕（Mac 上合成，盒子没有 CJK 字体）。

## 吞吐与存储（实测，results/stages.md）

- 1 worker（卡上同时有 HUGSIM）：RTF 0.41；agent 每 tick：PDM-Lite 21 ms、打包 7.5 ms、记录 5 ms、H.264 管道 0.8 ms；每条路线加载 ~50 s；
  7.2 GB 显存、2.2 核。5 worker 一个作业：31.6 GB（6.3 / worker）、11.4 核（2.3 / worker），每 worker RTF 0.33–0.64。
- 存储：无损模型帧 88 KB / tick（比原始 393 KB 小 4.5 倍）；1000 段 × ~500 tick ≈ 45–60 GB，标签 / SDF / actors 几 GB。盘余 587 GB。
- 全量：3 个作业 × 6 worker（每卡一个，CARLA 上限 6），稳态 ~650 条 / h → ~1.5–2 h，所以不需要 > 3 h 的 profile。
- 冒烟后改动（都是实测失败）：路线 v2（路口内不放关键点 + 重规划校验，id 920000+）；站立 45 s / 仿真 240 s 提前结束；路线指令改由几何标注；
  对齐门槛改为结构性（相机帧号 = 状态帧号）+ 出生起步时刻，模型转弯能力只报告不设门槛。

## 分阶段（docs/long-runs.md）

1. **smoke**：1 条 Town12 左转 JT 路线，1 worker；看实时因子、显存、核、每段耗时、存储；跑 labels + check + GIF，人工看。
2. **ten**：10 条，覆盖 10 个场景型 / 尽量不同城镇（5 个路口转弯型 + Accident / ParkingExit / DynamicObjectCrossing /
   MergerIntoSlowTraffic / SequentialLaneChange），按冒烟实测装 worker；门槛全过才进全量。
3. **all**：其余 ~989 条，J 个作业共享一个路线表（b2d_run 的 claim），池有空才上，不抢占；超过 3 h 先做 profile；每 3–5 h 报告。

门槛（任一失败写 ERROR 停）：完成率 ≥ 0.9（smoke 1/1）；完整性；帧滞后最小在 0（≥ 20 个移动样本时）；转向符号 ≥ 0.8（≥ 5 个转弯样本时）；
SDF 内比例 ≥ 0.9；指令一致 ≥ 0.8。专家自身的碰撞 / 闯灯照记（index.csv），不作门槛，训练时按需剔除或截断。

## DAgger（不做，只是不堵死）

agent 每 tick 先在真实状态上跑 PDM-Lite（专家控制 = 标签），再由 `_drive()` 决定实际施加的控制；`driver: policy` 时换成学到的策略
（经 socket 拿模型帧 + ego，同一打包），ego.npz 同时存 ctl_applied / ctl_expert / driver。标签全部由记录量离线算，与谁在开无关。

## 已知限制 / 待定

- 路线指令的前瞻距离（30 m）是按第 93 条（路口前 30 m 内 78% 一致）定的默认，未对 NAVSIM 的生成规则逐条校准；存了原始距离可改。
- JT 场景型是按路口属性指派的，场景能否在该位置成功实例化看运行时（scen.jsonl 记了真正激活的场景）；起不来的退化为普通路口转弯。
- 车身外廓是 MKZ，不是 nuPlan 的 Pacifica；hinge 训练用 `footprint`。
- 渲染：两路都按全尺寸 1928×1208 渲染以与闭环评测逐位同管线；wide 帧只用传感器中间 ~638×380，裁小渲染可能省很多但会改自动曝光，未做。
