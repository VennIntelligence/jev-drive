# 别人会转弯吗：第三方策略在我们同一把尺子上的闭环转弯（2026-10-15）

问题（用户）：WA-JEPA、Bench2Drive baseline、AlpaSim baseline 好像没有精心训练就会在路口转弯；我们为什么这么难？能不能在空闲的 box 上跑几个看它们闭环到底转不转？
约束：学生仍是 Cinque；借来的模型只作测量 / 参照。只测量，不训练，第三方代码原样跑。
代码 / 预登记 / 结果：[experiments/others_turn/](../experiments/others_turn/README.md)（预登记 `plans/2026-10-15-others-turn-prereg.md`，闸 G1–G3 全过）。

## 一句话

**在我们的尺子上，没有一个第三方策略比 RSG-L18 会转弯。**held 452 / 18 决策的 24 个转弯，完成数：RSG-L18-s0 16，YR10m10 9（桥接，与 follow_ref 一致），WA-JEPA 7，
主办方公开的 py123d-garage TransFuser 1，官方 sample LTF 0、DiffusionDrive 0。它们的失败都在弯里（入口到出口 +10 m），只转了路线角度的一半左右（航向覆盖中位 0.45–0.62）就出走廊或出路。
「别人不用专门训练就会转」在 AlpaSim nuPlan 上不成立。在 CARLA 里别人确实会转，但条件不同：几何目标点、规则专家的数据、专家标签里统一的路口减速（第 3 节）。

## 1. 清单：能原样闭环跑的第三方策略（第一部分）

| 策略 | 在哪 | 能跑的 harness | 输入 | 原样跑的代价 | 本次 |
|:--|:--|:--|:--|:--|:--|
| WA-JEPA（released 权重 + 自带 NAVSIM agent） | box `third_party/wajepa`、`models/wajepa` | AlpaSim nuPlan（我们的 `lib/wajepa_driver.py`，d188）；仓库自带 HUGSIM 闭环 client；PAI 没有 driver | L0 / F0 / R0 / B0 512×256、4 帧历史、自车速度 / 加速度、**4 值命令（sample 规则：首个 ≥ 5 m 路点、±2 m）** | 0：held 460 / 18 决策已跑（alpasim_long 补记 3），driver 记了每步 anchor | 读了 |
| LTF sample（SimScale） | box `models/alpasim_ltf` | AlpaSim nuPlan | L0 / F0 / R0 + 同一 4 值命令 + DynamicState | 已跑 460，但没记 anchor → 24 个转弯重跑约 0.1 卡时 | 读了 |
| DiffusionDrive sample（SimScale） | box `models/alpasim_dd` | 同上 | 同上 | 同上 | 读了 |
| py123d-garage resnet34 v0.1.0（主办方公开的 latent TransFuser；榜首条目本身未公开） | box `models/garage` | AlpaSim nuPlan（`garage.sh`） | F0 / L0 / R0，路线形式未公开 | 同上 | 读了 |
| GTRS-Dense、VaVAM sample | alpasim checkout 里 | AlpaSim | NAVSIM 式 / 视频模型 | 没装，各约半天 | 没做 |
| 主办方 PAI 参考（alpamayo1、vavam-linear / nonlinear、alternative_1..5） | GitHub LFS（alpasim 0bb4c4b）；Tokyo `pai1/ref` 有 alternative_* | 只有逐 scene 汇总（441 curated val × 3 rollout），无轨迹 | – | 0（只读得分） | 附读 |
| Alpamayo 1.5（自己 serve） | box `third_party/alpamayo1.5` | AlpaSim 自带 driver 没装；10B | 多相机 + 路线 | > 1 天 | 没做 |
| Bench2Drive：TCP 官方权重（Tokyo + box）、BridgeDrive 权重（box）、SimLingo gate（box）；UniAD / VAD（Zoo）、TFv6 / LEAD（Tokyo）只有代码，权重公开 | Tokyo（两卡都忙） | B2D / CARLA | 目标点 + 命令 | Tokyo 忙；B2D 分不是我们的转弯尺 | 只看文献 |
| NAVSIM 规划器（drivor、ztrs、sparsedrivev2、autovla、qwen-drive，Hydra-MDP 部分权重公开） | box `third_party` | 只有开环 | NAVSIM 输入 | 每个要写一个 WA-JEPA 式 driver，约 0.5–1 天 | 没做 |

要紧的一点：**AlpaSim 里所有 NAVSIM 系策略拿到的路线和我们一样弱**：40–80 m 外路点推出的左 / 直 / 右命令（OPTED 论文同样说 AlpaSim 给的是 40 m 外的 one-hot）。
所以 nuPlan 上这是一次输入公平的比较；我们多了路线记忆（d286）。

## 2. 测量（第三部分）

held 452 / 18 决策，`rf_follow.py` 的 24 个转弯（log 在 18 决策内自己完成、路线净转角 ≥ 60°），完成 = 零分事件前越过出口 +10 m、离路线 ≤ 2.5 m。
WA-JEPA 读存档的 drive.jsonl；LTF / DD / garage 在 24 个 scene 上原样重跑，anchor 取自 rollout.asl（G1：WA-JEPA 85 个 rollout、1 527 个决策上 asl 与 driver 记录逐位相同；
G3：重跑与存档 460 的零 / 非零一致 24 / 23 / 24）。

| driver | 24 转弯完成 | 配对差 vs RSG（pp，log 聚类 CI） | completed / event / off / short | 24 个转弯 scene 的零分（走廊 / 出路） | 失败时航向覆盖（中位） | 入口速度中位（m/s） | held 452 均分 / 零分 |
|:--|--:|:--|:--|:--|--:|--:|:--|
| RSG-L18-s0 | **16** | – | 16 / 1 / 0 / 7 | 1（1 / 1） | 1.04（失败都是慢） | 4.19 | 0.881 / 26 |
| YR10m10-F-s0（桥接） | 9 | – | 9 / 6 / 5 / 4 | 6 | 0.94 | 4.44 | 0.785 / 74 |
| WA-JEPA | 7 | −37.5 [−68.8, 0.0] | 7 / 14 / 1 / 2 | 13（8 / 6） | 0.45 | 4.61 | 0.766 / 85 |
| py123d-garage | 1 | −62.5 [−83.3, −40.9] | 1 / 18 / 4 / 1 | 18（5 / 12） | 0.62 | 4.19 | 0.628 / 121 |
| LTF | 0 | −66.7 [−85.7, −45.5] | 0 / 23 / 0 / 1 | 23（16 / 9） | 0.56 | 4.83 | 0.569 / 176 |
| DiffusionDrive | 0 | −66.7 [−85.7, −45.5] | 0 / 24 / 0 / 0 | 23（18 / 8） | 0.49 | 4.47 | 0.540 / 192 |

（零分按种类计数可重叠。）按预登记判读：「它们不经专项训练就会转」（≥ 16 / 24）没有一个成立；「它们也不会转」（≤ 12 / 24）四个都成立，LTF / DD / garage 的配对差 CI 上界 < 0；
WA-JEPA 的 CI 上界正好 0.0。入口速度大家差不多（4.2–4.8 m/s，都比 PAI 上我们的 7.4 慢）。所以 nuPlan 上它们失败不是因为快，是弯里只转了一半，和我们「晚而宽」那一类同形（d308），只是多得多。
文件：[follow.md](../experiments/others_turn/results/follow.md)、[turn_mech.md](../experiments/others_turn/results/turn_mech.md)、[held18.md](../experiments/others_turn/results/held18.md)。

**怎么失败（turn_mech，逐 scene 看 `results/turn_mech.csv`）**：WA-JEPA 是两极：要么完成（航向覆盖 ≈ 1.0），要么几乎不转（6 个 scene 失败时航向覆盖 ≈ 0，直着开出走廊 / 出路），
很少有「转一半」。LTF / DD / garage 是转一部分（0.3–0.8）就出走廊或出路，连 RSG 完成的 16 个弯里也几乎全失败。
规则专家 XP-L12-E46（有地图特权，只作上界）22 / 24。对照页的重跑把完成数原样复现（WA-JEPA 7、RSG 16、规则专家 22）。
plan 抽取核对：从 rollout.asl 取的 plan 与 RSG driver 自己记的 plan 在 72 个决策上差 < 0.1 mm，所以第三方的 plan 几何就是它们返回的样子。

**对照页**（`~/Desktop/others_turn/index.html`，Tokyo `/home/ujs/Desktop/others_turn/index.html`；24 行 × 6 列：RSG、规则专家、WA-JEPA、LTF、DD、garage；
每格：结果与入口速度、鸟瞰（各决策 plan 按时间着色）、四格读数、driver 自己收到的前视帧视频；生成：`experiments/others_turn/scripts/ot_viz.py`）。
以下是看图的**印象**（看了 4 个场景的鸟瞰拼图 + 命令序列，没有逐格看完）：
- WA-JEPA：要么沿车道转得很好（7 个完成，路线贴得和 RSG 一样），要么根本不转，plan 从头就是直的，直穿路口（760d5a11、016a5795）。
  它收到的命令在这些弯里每个决策都是对的左 / 右（drive.jsonl 的 cmd），所以是**拿到对的命令也不转**。不慢，入口速度与人相近。
- LTF / DiffusionDrive：每个弯都开始转，前半段在车道里，后半段新 plan 一个比一个更靠外（plan 间 2 s 点横跳约 1 m / 决策，比我们的蠕变大一个量级），
  在出口附近出走廊 / 出路；有的右转直走（016a5795）。速度与人相近。plan 抽取核对过，这是它们返回的 plan 本身。
- garage：转得早、切内侧，在入口线附近就压到弯内路沿（出路零分 12 个），和其他几家方向相反。
- RSG：沿路线在车道里转，失败主要是慢（7 个到 rollout 结束没过出口），只有 1 个零分。规则专家：车道中心线，22 / 24。

**附读（PAI，只有得分）**：主办方参考在我们 36 个严格转弯 scene 上（[pai_ref.md](../experiments/others_turn/results/pai_ref.md)）：
alpamayo1 均分 0.421、通过 56 / 108 rollout（52 %）、≥ 2 / 3 通过的 scene 19 / 36；vavam-linear 0.195（27 %）；vavam-nonlinear 0.055；alternative_1..5 0.03–0.07。
RSG-L18-s0（path 0，1 rollout）0.287、通过 14 / 36（39 %）、按轨迹完成 5 / 36。只有 Alpamayo 1（10B VLA，主办方自己的模型、自家数据域）在 PAI 转弯上比我们强，差距不大；
VaVAM 和其他参考都更差。通过 ≠ 完成，配置也不同（主办方的 run），只作方向。

## 3. 他们怎么转弯（第二部分，文献 / 代码）

| 方法 | 路线怎么进 | 怎么训 | 闭环转弯 | 靠什么「承诺」转弯、弯速从哪来 |
|:--|:--|:--|:--|:--|
| TransFuser++（Hidden Biases, ICCV 2023；carla_garage） | GNSS 目标点（TP，相邻约 30 m，车到 7.5 m 内切下一个），可选离散命令 | 模仿规则专家 750k 帧；感知预训 + 规划两段；相机 ±1 m / ±5° 增广（Longest6 DS 57 → 66） | Longest6 DS 69；**同一模型 TP 换成命令：DS 39 → 32、RC 84 → 56、路线偏离 0 → 0.86**，加增广补不回 | 专家轨迹总经过 TP，模型学会朝最近 TP 开，TP 周期性重置横向误差；路径（1 m 间距 10 个检查点）与速度（4 类目标速度）解耦；弯速来自专家标签（路口 5 m/s、平常 8 m/s）；TP 远在弯后时 GRU 航点会切弯 |
| TFv6 / LEAD（CVPR 2026） | 前 / 当前 / 下一共 3 个 TP，各一个 token，提前切换；NAVSIM 上用命令 | 模仿 73 h，只有视角扰动；作者自认大偏差后的恢复弱 | B2D DS 95.2、SR 86.8；1 → 3 个 TP：B2D 87.3 → 89.3；去 GRU 解码器 +2.3 | 单个远 TP 规定不了多步动作；速度是独立 query，不看 TP |
| TCP（NeurIPS 2022） | 离散命令 + 目标点坐标 + 速度 | 模仿 Roach RL 专家 + 特征蒸馏 | 纯轨迹分支的布局碰撞 64 %、出路 63 % 发生在转弯中（PID 跟航点跟不住大弯）；B2D TCP-traj DS 59.9 | 转弯时切到直接控制分支 |
| UniAD / VAD（Bench2Drive） | 6 类命令选一组 plan；目标点只在 PID 里（角度更小或 10 m 内与 plan 不一致时直接朝 TP 打方向） | 模仿 Think2Drive 专家，B2D-base 1000 clip | DS 45.8 / 42.4，SR 16.4 / 15.0；没有转弯分析 | 命令分头 + 控制器朝 TP 的覆盖：控制器里藏着一个几何路线目标 |
| CaRL（2025，RL） | BEV 栅格；CARLA 里路线只画在路口内 | PPO 从零，3–10 亿样本；奖励 = 路线完成 × 车道中心 / 超速 / 舒适（横向加速度 ±4.89 m/s²）惩罚 | Longest6 v2 DS 64；nuPlan Val14 CLS 91.3；去掉路线栅格路口行为不一致 | 路线完成奖励 + 路口内的几何路线；弯速隐含在横向加速度惩罚里 |
| DiffusionDrive / Hydra-MDP（NAVSIM） | one-hot 命令（前方 20 m 中心线点、±2 m）+ 速度 + 加速度 | 从零；锚点 / 8192 条轨迹词表 + PDM 子分蒸馏 | **NAVSIM 是非反应式开环 / 伪仿真，不是闭环证据**；本次在 AlpaSim 闭环里：DD 0 / 24 | 多模态候选 + 选择，不是回归平均 |
| AlpaSim 挑战 2026 | 40 m 外 L / S / R one-hot（与我们同一信号） | 榜单方法未公开（技术报告 10-31 截止）；py123d_garage = TransFuser 两段训练 | OPTED（arXiv 2609.20756）：TransFuser / VaVAM 在 AlpaSim 里 RL 后训练，LTFv6 PAI 26.5 → 41.8；**VaVAM 最大单项增益来自控制器加摩擦圆（横向加速度）约束，37.1 → 50.6** | 车道对齐门控的进度奖励；横向加速度上限放在控制器里 |

出处：Hidden Biases arXiv 2306.07957（表 1、3.1 节、附录 C.4）；carla_garage `team_code/config.py`；LEAD arXiv 2512.20563；TCP arXiv 2206.08129；
Bench2Drive arXiv 2406.03877 与 Bench2DriveZoo `team_code/pid_controller.py`；CaRL arXiv 2504.17838；DiffusionDrive arXiv 2411.15139；Hydra-MDP arXiv 2406.06978；
OPTED arXiv 2609.20756；SimScale arXiv 2511.23369（文献由子 agent 读网页整理，未逐条复核原文）。

## 4. 差异清单

**他们有、我们没有**
1. 几何路线目标。CARLA 系用目标点（TF++ 1 个、LEAD 3 个，UniAD / VAD 的 PID 也朝 TP 覆盖）。TP 在路口另一侧时，唯一一致的路径就是转弯，而且每过一个 TP 横向误差清零一次。
   我们是 40 m 外一点的三值符号 + 路线记忆（d284 / d286）。Hidden Biases：同一模型 TP 换命令，RC −28。
2. 干净一致的路口速度标签。规则专家在路口统一降到 5 m/s，路径和速度分头学。我们模仿的是人类日志：速度不一致，还会从自身历史复制（d285 棘轮、d294）。
3. 数据里转弯多。B2D / PDM-Lite 路线 60 % 是路口转弯（b2dc-train@v2，d152）；nuPlan / PAI 只有 10–15 %。
4. 输出接口在转弯时绕开航点跟踪。TCP 转弯时出控制；OPTED 在控制器里加横向加速度上限；UniAD / VAD 的 PID 朝 TP。我们是 plan → MPC、每 0.5 s 从车的位姿重新规划，蠕变（d308）就出在这里。
5. 多模态候选 + 选择（DiffusionDrive / Hydra-MDP / GTRS）。我们是回归头，flow head 试过，闭环灾难（d307）。

**我们有、他们没有**
1. 预训练的视觉先验（Cinque，冻结）。但 d283：在路口走廊上它比 WA-Cf 看得差（15 m 处 0.76 对 0.59）。
2. 路线记忆（d286）：唯一在闭环里真被依赖的路线输入；held 上也是它把 RSG 从 YR10m10 的 9 / 24 推到 16 / 24。
3. on-policy 行与 18 决策闭环训练。AlpaSim 里的第三方 sample 都只有日志模仿（SimScale 的伪专家恢复除外）。

**在「Cinque 留下」内能借的**
- 几何路线目标代替 / 补充三值命令：把路线记忆的若干点（如 +10 / +20 / +30 m）当显式目标点 token，做 LEAD 式多 TP。它是 adapter 的输入，Cinque 不动。
  注意 TF++ 的教训：远 TP 会切弯，要近点。
- 路径与速度解耦，弯速独立：用一个独立的速度头 / 目标速度类，标签取路口前 26 m 起的人类减速剖面（d293）。注意 d303 的教训：hinge 式「往哪拉」会学成切内侧。
- 控制侧：MPC 加横向加速度（摩擦圆）上限（OPTED 最大单项增益），以及 serve 侧意图保持（`JEV_INTENT_HOLD`，在跑）。这正对着蠕变。都不碰策略权重。
- 转弯数据密度：转弯 scene 过采样，或把 B2D 的 60 % 转弯数据（d152）作为另一个域的行。
- RL / 路线完成奖励（CaRL、OPTED）在 Cinque 之上训 adapter。和用户的「不要 DAgger」不冲突，但代价大，排后。

## 5. 为什么对我们难（一段话）

前提先改一下：在 AlpaSim nuPlan 上，「别人不用专门训练就会转」不成立。同一把尺子、同一个弱路线信号下，WA-JEPA 7 / 24、主办方 TransFuser 1 / 24、官方 LTF / DiffusionDrive 0 / 24，
都比 RSG-L18 的 16 / 24 差，失败也同形：弯里只转一半就出走廊。唯一在 PAI 转弯上略强的是主办方自己的 10B Alpamayo 1（通过 52 % 对 39 %）。
别人「会转」的地方是 CARLA，那里的条件我们都没有：路口另一侧的几何目标点（只要朝它开就是转弯，每过一个点误差清零）、规则专家给的一致路口减速、60 % 是转弯的数据、
转弯时绕开航点跟踪的输出或控制器。我们难，是因为三样东西叠在一起：路线只以 40 m 外一点的左右符号进来、人类日志里转弯少且速度不一致、plan → MPC 每步从自己的位姿重规划（欠量累积成蠕变）。
这三样在 AlpaSim 里人人都有，所以人人都转不好。最便宜的借法是把几何目标点和独立弯速做成 adapter 输入 / 头，再在控制侧加横向加速度上限，Cinque 不动。
