# 第三层（判断之后的行为）量具调研：现有量具、零成本一版、P6 草案

状态: 调研报告，2026-09-26 由 Opus 子代理只读写成，主会话落盘；数字以所引出处为准，结论已进 [decisions.md](decisions.md) 第 46、47 条（就地修正见第 35 条）。

2026-09-26，只读调研。没有跑 GPU、没有碰 box。WOD 的数字是在 Mac 上直接读仓库里已提交的
`results/wod-zeroshot/per_frame.npz`（479 个 rater 帧，每帧 3 条打过分的 rater 轨迹、log 未来和各考生的预测）算出来的，脚本放在 `/tmp`，没进仓库，判定规则写在 §2.1。
「第三层」指 E 层判断之后的行为模式：bypass（绕行）、negotiation / yield（让行、谁先走）、unprotected turn、recovery（恢复）。

## 0. 结论先行

1. 公开量具里，**能把「停下」和「绕过去」分开的只有闭环 SR，而且只是间接的**：Bench2Drive 的 obstacle_bypass 路线，停着不走会被 60 s blocked 终止，拿不到 SR。
   开环榜（WOD RFS、NAVSIM EPDMS）都不按行为模式计分。negotiation 没有一个榜单单独报，最接近的是 interPlan 的 assertive / conservative agent 和 B2D 的 unprotected turn SR。
2. **PDM-Lite 确实会绕，但绕法是特权的**：`autopilot.py` 读 scenario_runner 写进 `CarlaDataProvider.active_scenarios` 的登记，在距障碍 50 m 内把路线整体平移到相邻车道中心；
   TwoWays 类用对向车的真值速度做 gap check，不够就在障碍后面用 IDM 等。BehaviorAgent 完全不会绕（只有「后车更快时让到同向车道」的 tailgating 变道），而且它只看 `*vehicle*` 和 walker，锥桶、警示牌这类 static prop 对它根本不存在。
   公开逐路线数据：PDM-Lite 的 obstacle_bypass SR 92%，YieldToEmergencyVehicle 100%，所有学习方法在后一项上都是 0。
3. **WOD-E2E val 上不生成新数据能做出第三层的第一版读数，但样本很少**：rater 最高分轨迹是严格 S 形 nudge（横向 ≥ 1 m 且回摆或回正、不算变道 / 转弯）的帧有 **21 / 479（4.4%）**；
   宽口径（5 s 处 |y| ≥ 0.75 m，不算转弯）有 133 帧，但弯道混在里面，而 WOD-E2E 不带 HD map，分不开。在两种模式都被 rater 打过分的 18 帧里，nudge 的最高分高于 stop 的有 11 帧，stop 高的 7 帧，平均高 0.67 分。
   也就是说，rater 经常认为停下也可以接受，WOD 在「停 vs 绕」上的区分力很弱。
4. 已有的零成本信号：在这 21 帧上，我们的 `cls_late` 预测出 nudge 的是 **0 / 21**，预测停车的占 29%。Alpamayo 1.5 的 6 个 sample 里有 17% 是 nudge，openpilot 是 14–24%；
   这些模型在 rater 答「keep」的帧上 nudge 率都 ≤ 1%。n 太小，只能算提示，不能当证据。
5. P6 可以做，但**x⁻ 的造法必须和 P5 不同**：P5 把 hazard 藏到地下，这对 PDM-Lite 的绕行无效，因为登记还在，它照样会平移路线，得到一个「对空气绕行」的假标签。x⁻ 必须同时删掉 `active_scenarios` 里那一项登记。

## 1. 已有量具逐个看

| 量具 | 量什么 | 规模 / 噪声 | 停 vs 绕 | 博弈（谁先走） | 出处 |
|:--|:--|:--|:--|:--|:--|
| Bench2Drive multi-ability：Overtaking | 9 个障碍类 scenario 的 SR（Accident、Construction、ParkedObstacle、HazardAtSideLane 及各自 TwoWays、VehicleOpensDoorTwoWays） | 220 条路线里 45 条；单次 SR 的 95% 带约 ±5（BLUE 6 次重复） | **间接**：停着会被 blocked 终止，所以 SR 高就意味着过去了；但「等一会儿再绕」和「立刻绕」分不开 | TwoWays 要让对向车，SR 把「等对了」和「冲过去没撞」混在一起 | `jevdrive/tfv6_rules.py:31`；`results/tfv6-rules-interface/noise.csv` |
| B2D Give_Way | InvadingTurn + YieldToEmergencyVehicle | 各 5 条 | InvadingTurn 是横向让出（压线）；学习方法大多 100% | YieldToEmergency 所有学习方法 SR 0、DS 恰好 70，量的是计分规则，不是能力 | `family.csv`；决策第 38 条 |
| B2D unprotected turn family | 6 个路口左 / 右转 scenario | 30 条 | 不适用 | **最接近 negotiation 的公开量**：TFv6 82%，其余 43–57%，PDM-Lite 90% | `results/tfv6-rules-interface/family.csv` |
| NAVSIM v2 navhard，EPDMS | `NC×DAC×DDC×TLC×(5EP+5TTC+2LK+2HC+2EC)/16`，两阶段 | 450 个 Stage 1 观测、5462 个 Stage 2 观测 | 不能：只对一段固定的未来打分 | 背景车不反应 | arXiv 2506.04218；`leaderboard-vs-ability.md` §2 |
| navhard Stage 2（合成偏离） | 在人类 4 s 终点周围 **横向 ±2.0 m（0.5 m 步长）**、纵向 5 m 步长取起点，用 3DGS 渲染，按 `exp(−d²/2σ²)`、σ² = 0.1 加权 | 每个场景最多约 20 个起点 | 量的是 **recovery**：从偏离车道的状态回到车道；**不是 bypass**，障碍内容就是原 log 里有的东西 | 无 | arXiv 2506.04218 |
| WOD-E2E RFS | 3 s 和 5 s 两个点落在 rater 轨迹 trust region 里的程度，按分数加权，区外下限 4.0 | val 479 个 rater 帧，11 个 cluster（Construction 15、Foreign Object Debris 78 等） | **弱**：考生拿到的是它最接近那条 rater 轨迹的分数；stop 和 nudge 常常都被打了 ≥ 7 分（§2） | 无交互 | arXiv 2510.26125；`results/wod-zeroshot/clusters.csv` |
| nuPlan closed-loop reactive | 背景车走 IDM，CLS 分数 | Val14 | 不按模式计分 | IDM 会让，所以「挤进去」是被奖励的 | — |
| interPlan | 在 nuPlan 上改出 8 类长尾场景，各 10 个：construction、accident、jaywalker、nudging、overtaking（有对向车）、三档密度的变道 | 80 个场景；背景 agent 分 conservative / assertive（反应延迟）/ mixed | **有**：没通过障碍的场景直接记 0 | **有一点**：assertive agent 会考验挤入 | arXiv 2404.07569：PDM-Closed Val14 92 → interPlan 42，GameFormer 75 → 11 |
| Waymax | 在 WOMD log 上闭环，metric 为 overlap / offroad / log divergence 等 | WOMD 规模 | 不按模式计分 | 论文自己指出 RL 会对 sim agent 过拟合 | arXiv 2310.08710（metric 清单未逐项核对） |
| WOSAC | sim agent 的分布真实性：kinematic、interactive（碰撞、最近距离、TTC）、map（offroad、离路沿距离）加权成 0–1 的 meta-metric | WOMD | 不适用：量的是 agent 像不像人，不量 ego 的决策 | 间接（interactive 特征的分布） | arXiv 2305.12032 |
| BehaviorBench | 在 PufferDrive 上从 WOMD 挑出交互密集的子集，配多种交互式 traffic agent | 未查到场景数 | 未查到 | 核心结论：纯 self-play 的 policy 会对训练对手过拟合，换一种对手就泛化不了 | arXiv 2605.10034 |
| HUGSIM | 3DGS 闭环，HD-Score = `NC×DAC×(5TTC+2C)/7 × RC` | 436 个场景 | 不能：EP 不进单帧分数，静止或刹车能保住分；只对 car 框做碰撞检查 | 插入的车有行为难度分级，但不单独计分 | arXiv 2412.01718；`leaderboard-vs-ability.md` §2 |

**读法**：第三层在公开量具里只以两种形式出现：(a) 闭环 SR，隐含「过去了」；(b) interPlan 这种按场景类型拆开的闭环分。
开环量具里没有一个把「行为模式」当计分单位。B2D obstacle_bypass 的公开逐路线数据可以再拆细一格（`results/b2d-family/public_routes.csv` ⋈ `routes.csv`，SR% / DS）：

| 方法 | 1W（同向相邻车道，20 条） | 2W（要借对向车道，25 条） | InvadingTurn（5 条） | unprotected turn（30 条） |
|:--|:--|:--|:--|:--|
| PDM-Lite | 95 / 98 | 92 / 95 | 80 / 99 | 90 |
| TFv6（第三方重跑） | 95 / 95 | 87 / 88 | 80 / 100 | 82 |
| BLUE r1 | 70 / 86 | 84 / 93 | 100 / 100 | 47 |
| SimLingo（第三方重跑） | 70 / 87 | 76 / 88 | 100 / 100 | 47 |
| SparseDriveV2 | 85 / 95 | 68 / 87 | 100 / 100 | 57 |
| Orion | 100 / 100 | 48 / 66 | 60 / 81 | 10 |
| UniAD-base | 25 / 45 | 12 / 42 | 20 / 75 | 0 |

2W 比 1W 低的方法（Orion −52、SparseDriveV2 −17、Hydra-NeXt −37），缺的是「借对向车道时判断对向车」的那一步，也就是 negotiation；只有 PDM-Lite 和 TFv6 两格都 ≥ 87%。
每格只有 5 条路线 / scenario，单格差 20 pp 就是一条路线，不能逐格下结论。

## 2. 零成本第一版：WOD-E2E val rater 轨迹

**数据**：`per_frame.npz` 里的 `traj`（479 × 3 × 20 × 2，4 Hz、0.25–5 s，ego 坐标系，x 向前、y 向左）、`scores`、`logged`、`pred/*`。
帧名和 cluster 从 `results/openpilot-openloop/wod_rfs_per_frame.npz` 对齐（speed 与 logged RFS 两列逐帧相同）。rater 轨迹的读取器是 `jevdrive/waymo.py:116–150`（`preference_trajectories`，`preference_score ≥ 0`）和 `waymo.load_rater`。

### 2.1 判定规则（在看任何分布之前写的，事后没改）

对一条轨迹取 3 点滑动平均的航向角 h，末端航向 h_end，全程最大 |h| 记为 h_max，横向峰值 peak = max|y|，末端横向 y_end：

- **turn**：|h_end| > 25°。
- **nudge_return**：peak ≥ 1 m 且 |y_end| < 0.5·peak（出去又回来的 S 形）。
- **nudge_hold**：peak ≥ 1 m，1 ≤ |y_end| < 2.5 m，|h_end| < 5°，且 h_max > 2·|h_end|（偏出去后已经回正）。
- **lane_change**：|y_end| ≥ 2.5 m、|h_end| < 10°、h_max > 2·|h_end|。
- **curve_or_other**：peak ≥ 1 m、航向单调（弯道，或 5 s 内还没回正的偏移）。**没有 map 就分不开**。
- **stop**：末速 < 0.5 m/s，或 v0 > 3 m/s 且末速 < 0.3·v0；其余记为 **keep**。

仓库里已有一个更宽的判定（`jevdrive/waymo_p5vlm.py:49`，5 s 处 |y| ≥ 0.75 m 记 nudge，代码注释自己写了弯道和变道在那里看起来一样）。它在 rater 帧上给出 log nudge 76 帧（`results/p5-vlm-metaaction/final/label_distribution.csv`）。

### 2.2 结果（479 帧）

| 模式 | rater 最高分轨迹 | log 未来 |
|:--|--:|--:|
| keep | 210 | 214 |
| stop | 85 | 92 |
| **nudge（return + hold）** | **21（4.4%）** | 15 |
| lane_change | 5 | 3 |
| turn | 71 | 55 |
| curve_or_other | 87 | 100 |

| cluster | n | 最高分为 nudge | 最高分为 stop | 两种模式都被打过分 |
|:--|--:|--:|--:|--:|
| Foreign Object Debris | 78 | 6 | 7 | 5 |
| Cut_ins | 20 | 3 | 2 | 1 |
| Special Vehicles | 25 | 2 | 5 | 3 |
| Pedestrian | 52 | 2 | 16 | 0 |
| Cyclist | 71 | 2 | 11 | 1 |
| Interections | 116 | 2 | 18 | 3 |
| Multi-Lane Maneuvers | 42 | 2 | 9 | 2 |
| Construction | 15 | 1 | 4 | 1 |
| Single-Lane Maneuvers | 38 | 1 | 10 | 2 |
| Others | 22 | 0 | 3 | 0 |

- 最高分和 log 都是 nudge 的只有 4 帧；最高分是 nudge、log 却停下的有 0 帧。
- 在 21 个最高分为 nudge 的帧上，最好的非 nudge 替代轨迹平均 7.62 分，其中 13 / 21 帧的替代轨迹 ≥ 7 分。也就是说，**在 RFS 口径下，绕行帧上不绕通常只少 1–2 分**。
- 两种模式都被打过分的 18 帧：nudge 胜 11、stop 胜 7；stop 的最高分里有 3 帧是 10 分。
- **Alpamayo 的 CoT 与它的轨迹不一致**：6 个 sample 里至少 4 个说了「nudge」的帧有 180 / 479，但这些帧里 rater 最高分是 nudge 的只有 12 帧、是 stop 的有 31 帧（`results/wod-zeroshot/cot.json`）。

各考生在 21 个 nudge 帧上的预测模式（同一套判定）：

| 考生 | 预测为 nudge | 预测为 stop | rater 答 keep 的帧上预测 nudge |
|:--|--:|--:|--:|
| cv | 0% | 14% | 0% |
| ours `cls_late` vision+ego（K = 1024） | **0%** | 29% | 0% |
| Alpamayo 1.5 nav（6 个 sample） | 17% | 3% | 1% |
| openpilot Cinque | 14%（3 帧） | 0% | 0% |
| openpilot Lebowski | 24%（5 帧） | 0% | 0% |

**判读**：WOD-E2E val 能提供一个真实数据上的第三层「人类答案」子集，但严格口径只有 21 帧，最多只能当 sanity check，不够做主考卷；
而且 RFS 本身几乎不惩罚「该绕时停下」。宽口径的 133 帧要先解决弯道混入的问题：WOD-E2E 没有 map，只能靠相机里的车道线检测，或者用 train split 的 log 按 ego 历史挑。
NAVSIM navtrain / navtest 有 map 和 GT box，可以挖「本车道有静止障碍 + log 做了 S 形 nudge」的 token（这是 E3 已有代码的一个变体，还没做）。

## 3. P6「行为模式考试」草案（供讨论，不是 todo）

### 3.1 配对

在 B2D obstacle_bypass 的 10 个 scenario 上，按 2 × 2 造世界：障碍 {有, 无} × 对向车流 {有, 无}（后者只对 TwoWays 类）。

| 世界 | 障碍 | 对向车 | PDM-Lite 的预期动作 |
|:--|:--|:--|:--|
| x₀₀ | 删（actor 藏起来，**并删掉 `active_scenarios` 里的登记**） | 删 | keep |
| x₁₀ | 在 | 删 | bypass（绕行） |
| x₁₁ | 在 | 在 | 等 gap 再绕（negotiation） |
| x₀₁ | 删 | 在 | keep（对照：对向车本身不应该引起反应） |

- **bypass 对比** = x₁₀ − x₀₀；**negotiation 对比** = x₁₁ − x₁₀（主要看纵向 Δ 和横向起动时刻的差）。1W 类只有前两个世界。
- 对向车流：PDM-Lite 的 `is_overtaking_path_clear` 只看目标车道上的车，所以 x₀₁ / x₁₁ 要给对向车道布一个确定的车流（TM seed 固定，spawn 表写死），不能依赖背景交通碰巧出现。
- **放置 null**：障碍移到路肩、不占车道（expert 应该 keep）。这测的是「看到障碍」和「障碍挡路」有没有被区分开。另保留 P5 的天气 null。
- **镜像题**：同一个障碍，但相邻车道不可用（实线 + 护栏，或对向车流一直不断），expert 应该停下。用来防「永远绕」这一类攻击，对称于 P5 里防「永远停」的题。
- YieldToEmergencyVehicle 单独做一对（救护车 {有, 无}）。它是唯一「横向让出 + 所有学习方法 0 分」的格子；由于计分规则把 DS 固定在 70，闭环上看不出能力，开环配对反而能看。

### 3.2 expert 标签

| 候选 | 会绕吗 | 问题 | 结论 |
|:--|:--|:--|:--|
| PDM-Lite | 会：2W SR 92%，1W 95% | 靠 `active_scenarios` 的特权登记触发；路线在距障碍 50 m 内就平移（`default_max_distance_to_process_scenario = 50`，`config.py:228`），真正的横向运动发生在障碍前约 8 m 的过渡段（`transition_smoothness_distance`）；gap check 用对向车真值速度，并假设超车速度 50 km/h | **主 expert**。横向分叉时刻是由几何决定的，不像 P5 行人那样早于视觉证据（E4），但仍要按 P5 的规则逐对检查 t_div ≥ t_vis |
| BehaviorAgent | 不会（只有 tailgating 变道）；锥桶、static prop 不在它的检测表里 | 在 Construction 类上会直接撞上去或卡住 | 只能当「只会停 / 不反应」的反例考生，不能当 expert |
| 人类 | WOD 严格口径 21 帧；NAVSIM 需要另挖 | 数量太少 | 真实数据侧的外部效度检查 |

### 3.3 判读量

- **ridge / 回归类考生**（输出一条均值轨迹）：Δ_lat(t) = y⁺(t) − y⁻(t)，取 t = 2 s / 3 s，沿「远离障碍」的方向记号；横向翻转率 = expert 横向反应的帧里，考生的 Δ_lat 同号且超过考生自己 null p95 的比例，另报 Δv₂。
  局限：均值轨迹会把「停」和「绕」平均成「减速 + 半个偏移」，所以还要报**模式一致率**，即把 §2.1 的判定用到考生输出上，得到 keep / stop / bypass / turn 的 4 × 4 混淆矩阵。
- **`cls_late` K = 1024 词表类考生**：把 1024 个 anchor 按 §2.1 分进 {keep, stop, bypass_L, bypass_R, lane_change, turn}，读 softmax 在每个模式上的质量 m_mode。
  主读数是 Δm_bypass = m_bypass(x₁₀) − m_bypass(x₀₀)，negotiation 读 Δm_stop(x₁₁ − x₁₀)。**前置检查**：词表里 bypass 形状的 anchor 有多少、oracle 覆盖率多少。
  WOD 的严格 nudge 只占 4.4%，k-means 很可能几乎不给它 anchor；词表在 box 上，这里未查到。如果 bypass anchor 太少，零读数就是 vocabulary 造成的，不是 representation 的问题。
- **null 地板**：(i) 天气 null 上的 |Δm| p95 或 |Δ_lat| p95（同 P5）；(ii) 放置 null 上的 bypass 率；(iii) 用 x₀₁ − x₀₀ 作为 negotiation 的 null。
  τ 取考生自己的 null p95，下限按 P5 的写法；横向下限建议 0.3 m（未测，需要在 null 上定）。

### 3.4 进考卷的 scenario 和规模

| 类 | scenario | 220 集路线 | 世界 / case | 备注 |
|:--|:--|--:|--:|:--|
| 1W bypass | Accident、ConstructionObstacle、ParkedObstacle、HazardAtSideLane | 20 | 2 | HazardAtSideLane 是慢速骑车人，同时考「跟还是超」 |
| 2W bypass + negotiation | 以上四类的 TwoWays 版 + VehicleOpensDoorTwoWays | 25 | 4 | 主考点 |
| InvadingTurn | 1 | 5 | 2 | 压线让出 |
| YieldToEmergencyVehicle | 1 | 5 | 2 | 横向让出 |
| negotiation（可选第二期） | 6 个 unprotected turn，路口车流 {有, 无} | 30 | 2 | 纵向「等 / 走」，接近 P5 的纵向判读 |

- 只用 220 集、3 个 TM seed、单 expert（PDM-Lite）：1W 20 × 3 × 2 = 120，2W 25 × 3 × 4 = 300，InvadingTurn + Emergency 10 × 3 × 2 = 60，null 约 55 + 放置 null 约 45，**合计约 580 个世界**。
  加上 unprotected turn（30 × 3 × 2 + 30）约 790 个。val 集里这些 scenario 有多少条路线未查到（xml 在 box 上）。
- 成本参考 P5 v1：1029 个新 run、每个 run 中位约 276 s（v0 实测），11 个实例并行。580 个世界约 580 × 4.6 min / 11 ≈ **4 h**，与 P5 v1 同一量级。
  图像量按 P5 v1「101 条路线、29 757 张」线性外推约 1.5–2 万张（推测）。
- 生成器改动：x₀₀ 删登记（改 scenario 的 `active_scenarios.append` 那一行，或者在 recorder 里过滤）；对向车流 spawn 表；放置 null 的位移。P5 v1 的 recorder 已经有 `driver: pdm_lite`（`scripts/p5_pair_agent.py`）。

### 3.5 这个考卷会推翻 / 证实什么

| 结果 | 含义 |
|:--|:--|
| TFv6 waypoint / path 在 x₁₀ − x₀₀ 上有显著的 Δ_lat，而且模式一致率高 | 证实 TFv6 的 obstacle_bypass SR（95 / 87）来自网络本身，而不是规则或接口；**推翻**第 38 条里「TFv6 高分主要不是 E 层」在这一格的读法，第三层是它的强项 |
| TFv6 的 Δ_lat ≈ 0，但闭环 SR 高 | 它的绕行是 route 输入、creeping 规则或闭环反馈兜出来的，开环读不出来。此时 P6 的开环口径本身要打折（与 P5 target speed 通道 2% 同构） |
| 我们的 `cls_late` / M-C：Δm_bypass ≈ 0 且词表 bypass 覆盖 ≥ 足够 | 第 25 条的 reaction decoder 目前只会「刹」，第三层是空的；配对差分得换成横向目标重训（与第 42 条同一套机制） |
| 同上，但词表里没有 bypass anchor | 瓶颈在 vocabulary，K = 1024 不够，或 anchor 取自以直行为主的分布；不涉及 representation |
| M-C 在 bypass 上翻，在 negotiation（x₁₁ − x₁₀）上不翻 | 第三层拆成两半：「看见障碍就偏」可以学，「看对向车再决定」学不到，需要时序或交互信息 |
| 所有考生在放置 null 上的 bypass 率 ≈ 在 x₁₀ 上的 bypass 率 | 考生只对「有东西」起反应，不管挡不挡路；对 P5 里行人翻转的解读也是一个警告 |
| PDM-Lite 在 x₀₀ 上仍然偏移（删登记失败） | 生成器 bug，考卷作废，先修 |

## 4. 候选考生

| 考生 | bypass / negotiation 的证据 | 出处 |
|:--|:--|:--|
| TFv6（waypoint / path 通道） | 闭环：obstacle_bypass 1W 95%、2W 87%，unprotected turn 82%（第三方重跑，每格 5 条路线 / scenario）；开环：无证据 | `public_routes.csv`、`family.csv` |
| SimLingo | 闭环：1W 70%、2W 76%，unprotected turn 47%，Emergency 0；开环：未查到 | 同上；arXiv 2503.09594 |
| Alpamayo 1.5 | WOD 上 21 个 nudge 帧有 17% 的 sample 是 nudge（keep 帧上 1%）；CoT 里常说「Nudge left due to the stopped vehicle」，但和轨迹对不上（§2.2）；B2D 闭环只有 n = 5 的 smoke（第 33 条） | `per_frame.npz`、`cot.json` |
| Qwen3-VL 读出 | WOD meta-action 的横向三分类准确率：4B 0.638、32B 0.640，route-command 基线 0.644，**不比基线好**；nudge 类单独的准确率未查到 | `results/p5-vlm-metaaction/final/rater_agreement.csv` |
| openpilot | WOD 21 个 nudge 帧上 Cinque 3 帧、Lebowski 5 帧预测 nudge，keep 帧上 0；n 太小，产品定位是 ADAS，「会绕」没有文献证据 | `per_frame.npz` |
| ours `cls_late` / M-C | WOD 上 nudge 0 / 21、stop 29%；CARLA 上只测过纵向（第 42 条） | `per_frame.npz` |
| Gigaflow（state-space self-play） | 论文正文描述会「squeeze around an obstacle」、完成 unprotected left turn 和 U-turn、为躲 cut-in 变道；在 CARLA / nuPlan / Waymax 上零样本 SOTA。公开权重：未查到 | arXiv 2502.03349 |
| PufferDrive / V-Max 上训的 RL policy | 都是 state-space（roadgraph + 邻车）；BehaviorBench 说纯 self-play 对训练对手过拟合；绕行 / 让行的逐项指标：未查到。要当考生必须接特权状态，与相机考生不可比 | arXiv 2605.10034、2503.08388；PufferDrive GitHub |
| PDM-Lite | 闭环上限：2W 92%、Emergency 100%，但靠的是特权登记 | `autopilot.py`（SimLingo 仓库） |

## 5. 给主会话的三条建议

1. **先花一个 CPU 小时把 WOD 的第三层读数定死**：把 §2.1 的判定写进 `jevdrive/`（取代 `waymo_p5vlm.py` 里 0.75 m 的宽口径，或者两种口径并列），
   在全部 479 帧和 NAVSIM navtest（有 map，可以剔掉弯道）上报每个考生的「绕 / 停 / keep」混淆矩阵。同时在 box 上数一下 `cls_late` K = 1024 词表里有多少 bypass 形状的 anchor。后者一条命令就能回答「我们的 head 零 nudge 是不是 vocabulary 造成的」。
2. **P6 值得做，定位是 P5 的横向 / 交互版，第一期只做 obstacle_bypass 的 10 类 + YieldToEmergency，单 expert PDM-Lite，约 580 个世界、约 4 h 生成**。
   登记前必须先做一个 smoke：x₀₀ 删掉 `active_scenarios` 登记后，PDM-Lite 不再偏移；x₁₀ 的横向分叉满足 t_div ≥ t_vis。negotiation 用 2W 的 2 × 2 来测，不要另起 unprotected turn 的考卷（第二期再说）。
3. **别指望公开开环榜来量第三层**：WOD RFS 在绕行帧上对停下只扣 1–2 分，NAVSIM navhard 的 Stage 2 量的是 recovery 而不是 bypass。
   对外能引用的第三层数字，目前只有 B2D 的 1W / 2W / unprotected turn SR（带 ±5 SR 的噪声带）和 interPlan（PDM-Closed 92 → 42）。论文里的第三层主张要靠 P6 自己的开环配对，再加上闭环恢复后跑一小批 2W 路线确认。

## 来源

- 仓库：`results/tfv6-rules-interface/{family,routes,noise}.csv`、`results/b2d-family/public_routes.csv`、`results/wod-zeroshot/{per_frame.npz,cot.json,clusters.csv}`、
  `results/openpilot-openloop/wod_rfs_per_frame.npz`、`results/p5-vlm-metaaction/final/*.csv`、`jevdrive/{tfv6_rules,waymo,waymo_p5vlm}.py`、`todos/2026-09-25-reactivity-program/i1-p5v1.md`、决策第 32/35/38/42/44 条。
- 代码（raw GitHub，2026-09-26 取）：[SimLingo `Bench2Drive/leaderboard/team_code/autopilot.py`](https://github.com/RenzKa/simlingo)（`_manage_route_obstacle_scenarios`）、同目录 `config.py`、`privileged_route_planner.py`、
  `scenario_runner/srunner/scenarios/construction_crash_vehicle.py:98–105`（`active_scenarios.append`）；carla_garage `leaderboard_2` 和 DriveLM `pdm_lite` 中的同名函数逻辑相同；
  [CARLA 0.9.15 `behavior_agent.py`](https://github.com/carla-simulator/carla/blob/0.9.15/PythonAPI/carla/agents/navigation/behavior_agent.py)（`_tailgating`、`collision_and_car_avoid_manager`）。
- 论文：Bench2Drive 2406.03877；NAVSIM v2 pseudo-simulation 2506.04218；WOD-E2E 2510.26125；interPlan 2404.07569；Waymax 2310.08710；WOSAC 2305.12032；HUGSIM 2412.01718；
  Gigaflow 2502.03349；V-Max 2503.08388；BehaviorBench 2605.10034；SimLingo 2503.09594；Alpamayo-R1 2511.00088。
