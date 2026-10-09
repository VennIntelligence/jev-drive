# M1 预登记：「转弯前横移」的修法在 AlpaSim 本地闭环上的判定线（2026-10-09，任何修过的 driver 的闭环分数之前写定）

lane M1。接 decision 202（C1：零分过半是 M 类横移，原因待定）、199（400 scene 基线）、198（OT30）。只有本地 run，不向 AlpaSim 注册、warm-up 或提交任何东西。
写这份文件时已经看过的东西：C0 / C0b 里**未修** driver（SH30-F-s0、AP2-AB-s0、OT30-F-s0 / s1）在 400 个诊断 scene 上的分数与 driver log，离线 replay（`scripts/m1_replay.py`，stage a，24 scene）。
没有看过：任何带修法的 driver 的闭环分数；OT30 / WA-JEPA 在 C0b 新增 300 scene 上的分数。

## 诊断到这一步的结论（修法的依据）

1. C1 说的「向 route 所示转弯的反方向横移」因果是反的。269 个 log 直行 scene 里，route 首个 waypoint 在 rig frame 里横移 > 2 m 的 scene 有约 2 / 3 在世界系里 route 根本没动（< 1 m）：
   是 ego 自己 yaw 了几度，42 m 外的 waypoint 经力臂在 rig frame 里移到另一侧，command 随后翻转。M 是直路上 ego 自己的航向漂移。
2. 漂移是闭环的 yaw 不稳定。直行 scene 上 ego 相对 log 的航向误差 sd 逐次决策近似线性增长（k = 2 … 8：0.5° → 5.3°，SH30；AP2、OT30 同形），均值偏右（−1.5°）。
3. 来源在图像通路里合成的 slot，不在 adapter 的 ego 特征、不在 command。离线 replay 一次只换一个输入：
   ego 特征换成拉直的历史、注入 ±0.05 rad/s 的假 yaw rate、换成 NAVSIM 的 ego 特征，plan 都不变（< 0.01 m）；
   把 warp 用的 ego 轨迹拉直（图像帧本身不变），plan 在 0.5 s 的 yaw 对「过去 0.5 s 转过的 yaw」的斜率从 0.97 掉到 0.10（AP2 0.90 → 0.01），sd 从 1.88° 掉到 0.58°；
   决策 3 把视觉 token 换成同一 token 的 NAVSIM 缓存（真实相机帧、log 轨迹），4 s 横向偏 ≥ 0.7 m 的比例从 29% 掉到 8%。
   最新的一对图像（t0 − 0.2 s，t0）是同一张 keyframe 沿 ego 轨迹 warp 出来的，模型从它读到 ego 最近的转动并原样续下去。
4. NAVSIM 一侧同样的续接存在而且是对的：navtest 上 log 未来 0.5 s 的 yaw 对过去 0.5 s 转过的 yaw 斜率 0.97，SH30 / AP2 的 plan 也是 0.97；直行 token 上 4 s 横向偏 ≥ 0.7 m 只有 3.3%（闭环决策 3 是 17–19%）。
   开环里这是 log 的统计规律，闭环里 MPC 把 plan 的 yaw 一比一执行（斜率 1.01–1.03），yaw rate 成了没有阻尼的积分器。

## 修法候选（都不重训、不用特权输入）

输入侧，driver 里的一个标量（`lib/sh30_core.py` 的 `damp_history`，默认 1 = 不改）：合成 slot 的 warp 轨迹只呈现 ego 历史转动的 `SH30_MOTION` = α 份额（α = 0：沿 t0 航向的直线、间距不变）。ego 特征、keyframe、command、输出不动。

| 名字 | 设置 |
|:--|:--|
| W0 | α = 0，每次决策都用 |
| W50 | α = 0.5，每次决策都用 |
| W0G | α = 0，只在 route 消息本身是一条经过 ego 的直线时用（`sh30_driver.route_line`：≥ 3 个 waypoint 离自己的最小二乘直线 ≤ 0.5 m、直线离 ego ≤ 2.5 m、方向与 ego 航向差 ≤ 20°），其余 α = 1 |

诊断集上还可以加的候选（只许用诊断 scene 选）：其他 α；官方允许每次提交调的 MPC gains（`idx_start_penalty`、`heading_weight`、`rel_front_steering_angle_weight`，范围按 docs/alpasim.md）；
OT30 checkpoint 上叠同一个 α（训练侧的修回行加输入侧的阻尼）。不在候选里：改 plan 的执行层规则、任何用地图 / 其他车 / log 未来的东西、WA-JEPA。

与已定结论的区别：d189 的 AR 臂是把 route waypoint 喂进 adapter 训练（未采用），这里不训练、route 只用来判「前方是直路」；d161 是 plan 学会预补偿 devkit tracker，这里不动输出；
d113 / d114 / d157 是执行层对低速打转和前车余量的规则，这里改的是模型看到的自运动，不是 plan 的后处理。

## scene 划分

- **诊断集**：C0 的 400 个 scene（`scenes_public_landed4.txt`）。C1 与本 lane 的全部诊断、候选筛选、α 的选择只用它。分两步跑：先 S1（M 集 82 个：49 个横移 / 零分直行 scene + 33 个对照，再加转弯 scene，约 110 个），再诊断集其余。
- **held-out**：C0b 新增的 300 个 scene（`c0b/lists/new.txt`）。本 lane 没有在上面做过任何诊断；C1 第 7 节读过未修 SH30 / AP2 在它上面的计数。
- **全量**：已落盘的 700 个（诊断 400 + held-out 300），只作汇报，不作判定。

## 流程（冻结）

1. S1 与诊断集其余：各候选在 SH30-F-s0 上跑，对照是 C0 的未修 SH30-F-s0（仿真确定，C0b 已验证逐 scene 复现）。
2. 选定**一个**设置：诊断 400 scene 上 mean scene score 最高者；差 < 0.002 时取零分少者，再相同取更简单的（不带门控 > 带门控；输入侧 > 加 gains）。选定后写进本文件末尾并 push，再跑 held-out。
3. held-out 只跑选定的那一个设置，基座 SH30-F-s0（主读数）；同一设置在 AP2-AB-s0、OT30-F-s1 上各跑一次作次读数。读一次，不回头改设置。

## 判定线（held-out 300 scene，修后的 SH30-F-s0 对未修的 SH30-F-s0，同时满足才算「修法成立」）

1. mean scene score 的配对差 ≥ **+0.015**，95% CI 下界 > 0。CI 是按 scene 配对的 bootstrap（10 000 次，seed 0，重采样单位 = scene）；按 log（`日期_车辆`）整簇重采样的 CI 一并报告，不作判定。
2. M 类零分至少减半：未修的 n_M → 修后的 ≤ n_M / 2。M 类按 C1 的规则（`c1_review.mechanism`，先匹配 S / H / F）：log 航向变化 < 8°、route 首个 waypoint 在 rig frame 里向一侧移 > 2 m、ego 在事件时刻向另一侧偏离 log 路径 ≥ 0.4 m。
   另报一个不含 route 条件的版本（log 航向变化 < 8°、偏离 ≥ 0.4 m），因为第 1 条结论说 route 条件是横移的结果。
3. 没有 route 横移的 scene 上零分不增加：scene 集合取未修 SH30 的 run 里 route 首个 waypoint 在 rig frame 里全程横移 ≤ 2 m 的那些（C1 的定义），修后在这些 scene 上的零分数 ≤ 未修的（C1：未修为 0）。

另报不作判定：零分按 flag（at-fault 碰撞 / offroad / 出 corridor）、转弯 scene（log 转角 ≥ 20°）上的均分与零分（α < 1 让模型看不到自己在转，转弯是它最可能伤到的地方）、偏慢 scene 数、
直行 scene 上航向误差 sd 的逐决策曲线、次读数（AP2、OT30-F-s1）、全量 700 的均分。

读法：三条都过 → 修法成立，进比赛 driver 候选；1 过而 2 或 3 不过 → 有增益但不是修掉了 M，写明；1 不过 → 输入侧阻尼修不动，结论改为训练侧（闭环 / on-policy 的航向恢复行），不再在 held-out 上换设置。

## 局限（事先写明）

- 单训练 seed；本地渲染与官方环境的一致性未验证；held-out 300 与诊断 400 来自同一批 nuPlan log 的不同 shard，不是独立城市。
- 仿真确定，单次 run 没有采样噪声；CI 只反映 scene 抽样。
- α 是一个闭环阻尼旋钮，不是把输入对齐到训练分布：α < 1 的 slot 在转弯时是训练里没有的呈现。

## 补记 1（2026-10-09，诊断集读完之后、任何候选的 held-out 分数之前写定并 push）

**候选扩充及原因。** S1 排队时 lane IG1（decision 206）给了一条线索：navtest 开环里「日志延续」的 token 上，强 hinge（λ 30 / margin 0.5 m：SHP、SH30、OT30、AP2）的 plan 横向虚构变化约 0.14 m，λ 10 的臂约 0.08 m。
所以 S1 里加跑了同一输入标准下的两个现成 checkpoint：`P2-F-s0`（无 hinge）、`P2H10-F-s0`（λ 10 / 0.3 m）。它们不在上面的候选表里；加入发生在读 S1 分数之前，选择只用诊断集。driver 代码与输入标准完全不变（`run.sh <dir> sh30`，`SH30_TAG=...`）。

**诊断集 400 scene 的读数**（对未修 SH30-F-s0 0.9306、零分 23；CI 为按 scene 配对 bootstrap）：

| 设置 | mean scene score | 零分（碰撞 / offroad / corridor） | 差 | log 直行 269 scene 的零分（SH30 14） |
|:--|--:|:--|:--|--:|
| SH30 + W0G（输入侧阻尼，带 route 门控） | 0.9353 | 18（5 / 5 / 8） | +0.0048 [−0.0089, +0.0193] | 9 |
| SH30 + W0 / W50（不带门控，只跑了 S1 的 126 scene） | S1：0.7731 / 0.7966（SH30 0.8105） | 转弯 scene 上 13 / 6 个由非零变零 | 弃 | |
| P2-F-s0 | 0.9591 | 10（1 / 6 / 3） | +0.0285 [+0.0067, +0.0518] | 1 |
| P2-F-s1 | 0.9491 | 12（1 / 7 / 5） | +0.0185 [−0.0045, +0.0426] | 2 |
| **P2H10-F-s0** | **0.9631** | 9（1 / 3 / 5） | +0.0325 [+0.0110, +0.0553] | 2 |
| P2H10-F-s1 | 0.9622 | 9（0 / 4 / 5） | +0.0317 [+0.0107, +0.0539] | 2 |
| P2H10-F-s0 + W0G | 0.9601 | 9（1 / 3 / 5） | +0.0295 [+0.0080, +0.0525] | 2 |
| 参照，未修：AP2-AB-s0 / OT30-F-s0 / OT30-F-s1 | 0.9335 / 0.9399 / 0.9476 | 24 / 19 / 17 | +0.0029 / +0.0093 / +0.0170 | 16 / 9 / 6 |

直行 scene 上航向误差 sd（S1 的直行 scene，决策 5）：SH30 4.9°、OT30 5.2–5.7°、W0G 3.9°、P2H10 1.7°、P2 1.2°；均值 SH30 −2.9°（偏右）、P2H10 −0.03°、P2 +0.4°。
读法：强 hinge 是触发（每次决策向远离近侧路沿的方向推一点，并带来向右的均值偏置），yaw rate 续接的无阻尼闭环是放大器；去掉触发比加阻尼有效得多，两者叠加没有额外收益。

**选定（按上面冻结的规则：诊断 400 scene 均分最高）：`P2H10-F-s0`，输入标准不变，不加阻尼。**

**held-out 运行（300 scene，各一次）：** 主读数 `P2H10-F-s0` 对未修 `SH30-F-s0`，判定线三条照旧不改。次读数（不作判定）：`P2H10-F-s1`（训练 seed）、`P2-F-s0`（无 hinge）、`SH30-F-s0 + W0G`（输入侧候选，换下原定的「同一设置叠到 AP2 / OT30」：选定的设置是换 checkpoint，没有可叠的东西）。
C0b 已有的未修 `AP2-AB-s0`、`OT30-F-s0 / s1` 在 held-out 上的分数作为参照并列（本 lane 在写这段时没有读过它们在 held-out 300 上的逐 scene 分数；读过 C0b 报告的 700 scene 汇总行）。

**代价，事先写明：** P2H10 → SH30 的强 hinge 在开环上是成立的收益（d170：navtest EPDMS +0.87 [+0.64, +1.15]，navhard +1.83）。在 AlpaSim 闭环上换回 λ 10 等于放弃这部分开环收益，换零分减少；两个榜用不同的 checkpoint 属于「逐榜配方」，要声明。

**运行事故：** S2 第一次同时提交了 9 个仿真栈，主机内存被耗尽，renderer 被杀，5 个 run 挂住（与 10-08 的挂起同型）；已取消重跑，chain 加了最多 3 个并发的上限。丢弃的 run 没有读过分数。
