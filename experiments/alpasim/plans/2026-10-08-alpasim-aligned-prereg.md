# AP2 预登记：把 openpilot adapter 模型的输入对齐到 AlpaSim 的接口标准（2026-10-08，任何 AP2 分数之前写定）

## 问题

SH30（Cinque 冻结视觉 + op_parity P2 adapter + drivable hinge）已能当 AlpaSim nuPlan track driver（decision 185），但它对齐的是 NAVSIM 接口，在 AlpaSim 上靠 serving 侧补丁喂进去。本线对 AlpaSim 做 P2 对 NAVSIM / HUGSIM 做过的事：以 AlpaSim 交给 driver 的东西为固定输入标准，把训练输入做成 AlpaSim 实际会给的样子，再用 SH30 配方（navtrain、公开数据与权重）训练，做 driver。

## 不重测（decisions 已定）

侧 / 后相机不加分（144）→ 只用 CAM_F0；原生 2 Hz 协议 N 输给 W（142）→ 保留 W（0.2 s slot，CPU ego-motion warp）；放开 encoder 不赚（op_parity unfreeze）→ 视觉冻结、沿用缓存 token；desire 不是选择信号（92）→ 指令只走 adapter；停车门不训进配方（174）；hinge 强度不再扫（172）→ λ 30 / margin 0.5。

## 写代码前实测到的 AlpaSim 输入定义（48 个 public scene，LTF sample 带 tap 跑一遍 `ltf_full48_tap/20261008-160440`，加读源码 0bb4c4b）

| 输入 | AlpaSim 的定义 | 证据 |
|:--|:--|:--|
| 路线 | 只由本 scene 5.5 s 录制轨迹（56 点）投到车道中心线，再沿车道图（航向最接近的后继）延长；每步从自车投影点重采样 20 点 × 4.21 m，丢掉 40 m 以内，恒为 10 个有效点（40–80 m）+ 10 个 NaN。**40 m 以外大多是地图延长，不是日志的真实去向** | `route_generator.py`；用 NAVSIM 2 Hz 日志 + 同一份 trajdata 地图 + 他们自己的 `RouteGeneratorMap` 重建：tap 位姿处 480 / 480 条消息 waypoint 均差 0.3 mm、最大 1.4 cm，指令 480 / 480 一致；k = 0, 1 用日志位姿均差 5 cm、指令 96 / 96 |
| 速度 / 加速度 | k = 0：nuPlan 录制值（= NAVSIM 的 vx, vy, ax, ay），但被多转了一次 −yaw（旋回后 v 差 0.004 m/s、a 差 0.027 m/s²）；k = 1：vy ≈ 0（0.026），ax = 轨迹 d\|v\|/dt（差 0.055，DB 值差 0.132），ay = vx·ω（差 0.042）；k ≥ 2：车辆模型状态，vy ≈ 0（0.004），ay ≈ 0（0.007），ax = 自身 dvx/dt（与中心差分相关 0.97） | tap 的 DynamicState 对 NAVSIM 日志 `ego_dynamic_state` |
| 偏航率 | DynamicState.angular_velocity.z，k = 0 / 1 与日志偏航中心差分差 0.003 / 0.001 rad/s | 同上 |
| 历史 | t = 0 前没有；决策 k 有 min(k + 1, 4) 个 2 Hz keyframe；位姿时间 = 帧曝光结束（k·0.5 s + 17 ms） | tap |
| 相机 | 8 路 1920 × 1080 针孔、无畸变；CAM_F0 每个 scene 的内外参不同（fx 1542–1582，fy 1466–1570，x 1.67–1.80 m，z 1.52 m），MTGS 渲染 | start_session |

## 设计（AP2）

模型结构、冻结范围、损失、hinge、anchor 全部沿用 SH30（`pp_train.PModel` arm P2，`--frames warp --hinge-lam 30 --hinge-margin 0.5`），只改**训练行的输入怎么造**（`experiments/alpasim/scripts/ap2_train.py`，复用 pp_train 的 Store / Losses / PModel，不改 pp_train）：

1. **决策序号混合**：每行抽 m ∈ {1, 2, 3, 4}，概率 0.1 / 0.1 / 0.1 / 0.7（一个 rollout 10 次决策的真实占比）。navtrain token 当作「起点在它之前 k 帧的那个 scene 的第 k 次决策」，k = 0, 1, 2（m = 1, 2, 3）或 3 + crc32(token) % 7（m = 4）。
2. **历史截断（serving 规则写死）**：m 个 keyframe 时，只有不早于最老 keyframe 的 policy slot 是真的（1 / 3 / 6 / 8 个），更老的 slot 置零且 invalid；最老真 slot 的图像对从零图开始（与 pp_prep 对 m = 4 最老 slot 的约定同一条规则）；缺的历史位姿 = 最老已知状态按恒定体速度与偏航率倒推（`sh30_core.fill_history`，偏航率用日志偏航样条导数，对应 AlpaSim 的 angular_velocity）。m < 4 的 slot token 用 serving 的同一个函数（`sh30_core.lattice`，cold = zero）在 navtrain 上重算并缓存（`ap2_prep.py`），m = 4 用已有 W 缓存。
3. **ego 状态按 AlpaSim 的定义**：m = 1 用录制值；m = 2：vy = 0、ax = 轨迹 d|v|/dt、ay = vx·ω；m ≥ 3：vy = 0、ay = 0、ax = 轨迹 d|v|/dt（日志轨迹代替闭环里车辆模型的 dvx/dt）。
4. **指令**：用 AlpaSim 自己的路线生成器在 navtrain 上重建路线（`ap2_route.py`），再套 shipped sample 的 4 向规则（首个 ≥ 5 m 的 waypoint，y > 2 m 左、< −2 m 右）；取代 NAVSIM `driving_command`。
5. driver 侧 k = 0 的速度 / 加速度旋回沿用 SH30 driver 的数据驱动判别（不写死旋转，官方环境若修了这个 bug 也不坏）。

不可用 / 替代（结果笔记给全表）：MTGS 渲染域（navtrain 没有渲染，训练用真实图像，不可对齐）；闭环里的 ax / 位姿是模型自己上一步计划的后果（开环日志只能给日志自己的）；7 路其余相机、交通灯、地图、agent 都不用。

## 臂（pilot：`navtrain_full.s{2,3,4}of12`，split `navsim/op-parity-s234`，3 000 步 × 64，seed 0，即 SHP-F-s0 的配方）

| 臂 | 输入 |
|:--|:--|
| SHP-F-s0（已有）/ SH30-F-s0（已有，全量） | NAVSIM 对齐；离线按今天 driver 的方式喂：m < 4 用 backwarp，指令走路线规则，ego 按 AlpaSim 定义 |
| N0 | 用 ap2_train 重跑 SHP 配方（m 恒 4、NAVSIM 输入）：只为验证新 trainer 与 pp_train 等价（dev ADE 对 SHP-F-s0） |
| A | AP2：上面 1–4 |
| AR | A + 路线 waypoint 本身进 adapter（ego MLP 输入加 60 维：10 + 10 个点的 [valid, x, y]） |
| AB | A，但 m < 4 用 backwarp 规则训练（slot 全 valid，token 用 backwarp 重算） |

## 离线读数（navtest，训练从不见 navtest；48 个 scene 是 navtest scene）

对每个 m = 1..4 按 AlpaSim 输入标准造 navtest 输入（k = m − 1；m = 4 即该 token 自己的 AlpaSim scene 的决策 3），各臂出 8 个位姿：
- 全部 12 146 token：对日志的 ADE（8 位姿均值）、4 s 纵向误差；按 136 个 log 聚类的配对 bootstrap。
- EPDMS：`python -m jevdrive.bench score-poses --traffic non_reactive`，子集 = seed 0 抽 2 000 个 token + 48 个 scene token；不含 EC。
- 参照：SH30 / SHP 在 NAVSIM 标准输入下（m = 4）的同口径数。

## 闸门与选择规则（先写死）

- **G1 学到了对齐输入**：A 对 SHP（今天的喂法）在 m = 1 的 ADE 更低且 CI 不含 0，m = 2、3 不更差（差 ≤ +0.02 m）。
- **G2 m = 4 不退步**：AlpaSim 标准输入下 A − SHP 的 ADE ≤ +0.02 m，子集 EPDMS ≥ −0.3。
- **路线编码**：AR 只有在 m = 4 上对 A 的子集 EPDMS ≥ +0.3 且 CI 下界 > 0，或 ADE ≤ −0.02 m 且 CI 不含 0 时才采用；否则 4 向指令。
- **冷启动规则**：默认零 slot（A）；AB 只有在 m = 1 的 ADE 比 A 低 ≥ 0.05 m 且 CI 不含 0 时才替换。
- G1、G2 都过 → 选中的臂按 SH30 全量配方（12 shard、10 000 步 × 128、warmup 300、seed 0）训练；全量闸门同 G1 / G2，对照换成 SH30-F-s0。全量不过或超预算 → serve 选中的 pilot 臂，如实报告。
- 闭环：1 scene（`--tap`）→ 3 scene → 48 scene（8 并发），看 frame dump / BEV 再信分数。48 个 scene 单 seed 只是执行证据，不在这 48 个 scene 上调任何东西。

## 预算

GPU ≈ 10 card-hour、墙钟 ≈ 10 h。估算：路线重建 CPU ≈ 0.3 core·s / token·变体；m < 4 token 重算（6 次 warp + 10 对编码 / token）全量 navtrain ≈ 9 core·h + 编码；pilot 每臂 ≈ 10–15 min；全量训练 ≈ 30–60 min；闭环 48 scene ≈ 4 min。全量 prep + 训练 > 1 h 的部分按 1 shard → 其余 shard、smoke 训练 → 全量的分级启动。
