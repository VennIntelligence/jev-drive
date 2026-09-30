# 人类参照线与只收轨迹的榜单能不能被"手开"（2026-09-29）

起因：WOD-E2E（Waymo Open Dataset Vision-based End-to-End Driving，Waymo 的视觉端到端驾驶开环榜）test 榜只给上传的轨迹打分，从不重跑模型，
所以它无法核实一份提交是不是声称的模型产出的：人可以把 1,505 个 test 帧逐帧手画轨迹，LLM 也可以直接生成。
用户的推测是：一个中等水平的人，在不熟悉车辆动力学的模拟器里开这些场景，未必比 openpilot 好。
我们冻结的 openpilot Cinque（加验证集选出的纵向 ×1.06）在 test 上拿到 RFS 7.921，按方法排 115 个中的第 19
（[结果记录](../tmp/2026-09-29-wod-e2e-cinque-test-result.md)，[榜单对照](../tmp/2026-09-29-leaderboard-wod.md)）。

本文只做文献和规则调查，不跑实验。每个数字标注来源：**[读]** 表示从原文读到（给 arXiv 号或 URL 和章节），
**[摘要/检索摘录]** 表示只在摘要或搜索摘录里看到、没核对正文，**[我们]** 表示我们自己在 val 上测的，**[推]** 表示推断。

## 0. 结论先行

1. **WOD-E2E 没有人类上限。** 论文不报 logged 轨迹的 RFS，也不报 rater 之间互评；3 条 rater 轨迹不是人开的，是 Wayformer 采样后由人挑选、打分的候选。
   logged future 在 val 上拿 8.13（DIAL 报的数，与我们测的 8.131 一致），而且这条 log 本身是"自动驾驶与人工驾驶混合采集"的，不能叫人类基线。
   在 RFS 这把尺子上，logged 驾驶员、榜首方法（test 8.167）和我们的 openpilot（val 8.12 校准后 / test 7.92）挤在 0.3 分以内。
2. **有人类参照的榜，参照都是"日志回放"，不是"人在模拟器里重新开"。** NAVSIM human PDMS 94.8、navtest EPDMS 90.3，nuPlan log replay CLS-NR 94 / CLS-R 80，Waymax expert 碰撞 0.61%。
   这些分数被评测设计本身抬高或压低（下文说明），而顶尖方法已追平或超过（NAVSIM EP 超过 human，nuPlan 上 PDM-Closed 的 CLS-R 高于 log）。
3. **没有找到任何人在 Bench2Drive 或 CARLA Leaderboard 上用官方 DS（Driving Score）给人类驾驶打过分（第 5 节二次检索后仍成立，但找到了一篇更弱的 CARLA 人机对比，见第 5 节）。** 最接近的是一篇 BCI 论文：20 名健全被试用手柄在改过的 CARLA LB 2.0 城区路线上开，限速 5 mph，自定义 DS 平均 0.823（满分 1）。
4. **完整性**：WOD-E2E 和 NAVSIM 都是"参赛方在本地拿到 test 传感器数据、自己推理、上传轨迹"，服务器只打分，没有任何机制区分模型输出和人工 / LLM 输出；
   常设榜不要求技术报告或代码。CARLA Leaderboard 2.x 是提交 docker 镜像在组织方 AWS 上闭环跑，结构上排除了手画。
   没找到针对这些驾驶榜的"手标提交"或 test 泄漏的公开讨论；最近的先例是 ILSVRC 2015 Baidu 多账号刷提交被禁赛。
5. **建议**：我们那条 test 分的可信度，最便宜的加固是公开推理代码和提交包哈希（冻结公开权重，任何人可复现逐字节相同的包），这是纯上传榜能给的最强证据。
   人类基线值得做，但分两步：先在 WOD val 上做"人手画轨迹"的小实验（直接回答用户的假设，零 GPU），
   再视需要在 Tokyo box 上做 CARLA 人类闭环基线（B2D 路线，官方计分）。协议见第 4 节。

## 1. 各榜的人类 / 专家参照

### 1.1 WOD-E2E 的 RFS 怎么构成

RFS（Rater Feedback Score）的构造 **[读，arXiv 2510.26125 §3.4–3.5]**：

- 每个 rater 帧有 3 条参考轨迹，每条带一个 0–10 分的人工评分；并保证至少一条高于 6 分。这些标签只随 val 发布，test 的标签留在服务器上。
- 这 3 条轨迹**不是人开出来的**。流程是：Wayformer 之类的模型先采最多 64 条候选，按速度、加速度、换道自动分桶，挑出少于 12 条多样候选（通常取最左、中间、最右），
  标注员从中选 3 条（要求包括最优、合理替代和次优），再由训练过的 rater 在可视化工具里对着 20 s 场景和其他 agent 的 logged 未来打分（§3.4.2–3.4.3）。
  Rank 1 轨迹的最低分是 6，这是"安全可行"的门槛（§3.4.3，Figure 6）。
- 打分：在 t = 3 s 和 5 s 各算一次，预测落在某条 rater 轨迹的 trust region（按初速缩放的矩形，基准阈值 3 s 为横向 1.0 m / 纵向 4.0 m，5 s 为 1.8 / 7.2 m）内就拿该条的分，
  区域外按距离指数衰减，三条取最大，两个时刻平均，最后**下限截到 4**（§3.5）。我们的移植证实 rater 分本身低于 4 时 RFS 也可以低于 4（decisions.md 第 2 条）。
- logged future **不在 3 条候选里**。论文只用它算 ADE（Average Displacement Error）。论文**没有报告** logged 轨迹的 RFS、rater 之间的一致性或任何人类上限 **[读，全文检索无]**。
- WOD-E2E 的 logged 轨迹来源是 "a mixture of autonomous and manual driving" **[读，§1 贡献段]**。所以"logged future 的分数"不是纯人类司机的分数，可能部分是 Waymo Driver 自己的。

在这把尺子上各条参照线（全部是 val 479 个 rater 帧，cluster mean，除非另注）：

| 轨迹 | RFS | 来源 |
|:--|--:|:--|
| 最高分 rater 轨迹（oracle 上限，拿到了 test 标签才可能达到） | 9.59 | [我们] decisions.md 第 2 条 |
| DIAL：intent 条件 best-of-128 池化 | 9.14 | [读] arXiv 2605.12625，经 [research/lit/2026-09-22-round5-log-as-target.md](lit/2026-09-22-round5-log-as-target.md) 转引 |
| **logged future（自动 / 人工混合）** | **8.13 / 8.131** | [读] DIAL 2605.12625；[我们] decisions.md 第 2 条 |
| 榜首 ZSD-Titan（**test**） | 8.167 | [读] 官方榜 2026-09-29 快照 |
| openpilot Cinque ×1.06（val 交叉拟合） | 8.119 | [我们] |
| openpilot Cinque 原生 | 8.005 | [我们] |
| openpilot Cinque ×1.06（**test**） | 7.921 | [读] Waymo 结果页 |
| constant velocity | 7.10 | [我们] |
| 原地不动 | 5.38 | [我们] |

读法：RFS 被下限 4 和 trust region 压缩得很厉害，"原地不动"都有 5.38。logged future 和我们的 openpilot 在 val 上只差约 0.01–0.13，
而在 s_ego 最高的十分位（车辆行为最出乎自身先验的那 10% 帧），logged future 只有 5.92、41.7% 被截到下限 **[我们，decisions.md]**。
所以用户的推测在 RFS 上**已经有一半答案**：已有的 logged 驾驶（混合了人工驾驶）并不比 openpilot 显著好 **[推]**。这里不需要新人类实验就能说；
但它回答不了"在 test 上"（test 没有 log 的分，未来被隐藏）和"一个陌生人在模拟器里开"这两件事。

### 1.2 其他榜的人类参照

| 榜 / 指标 | 人类参照 | 数值 | 顶尖方法 | 来源 |
|:--|:--|:--|:--|:--|
| NAVSIM v1 navtest，PDMS | Human（logged 轨迹） | NC 100，DAC 100，TTC 100，Comf 99.9，EP 87.5，**PDMS 94.8** | CLOVER 94.5、DrivoR 94.7、ReflectDrive 94.7 | human：[读] arXiv 2406.15349 Table 1；方法：[检索摘录] 2605.15120、2601.05083，未核正文 |
| NAVSIM v2 navtest，EPDMS（单阶段，官方称 unofficial） | Human | NC 100，DAC 100，DDC 99.8，TLC 100，EP 87.4，TTC 100，LK 100，HC 98.1，EC 90.1，**EPDMS 90.3**（修复后 94.5） | SimWAM 90.2，DriveFine 89.7（EP **88.7** > human 87.4） | [读] arXiv 2608.07468 Table 2；修复后值见 [榜单对照](../tmp/2026-09-29-leaderboard-wod.md) B2 节 |
| NAVSIM v2 navhard 两阶段，EPDMS | 未见 human 行 | — | SimWAM 37.6 | [读] 2608.07468 Table 3 没有 human 行；未找到其他来源 |
| nuPlan Val14 | Log Replay | CLS-R **80**，CLS-NR **94**，OLS 100 | PDM-Closed CLS-R 92 / CLS-NR 93 | [读] arXiv 2306.07962 Table 2 |
| Waymax（WOMD 闭环） | Expert（log playback） | off-road 0.32%，collision 0.61% | 最好的 BC 碰撞 4.59% | [读] arXiv 2310.08710 Table 3–4 |
| MetaDrive 安全驾驶 | 人类专家现场驾驶 | test success **0.98**，safety violation 0.16 | HACO 0.83；PVP 0.857 | [读] arXiv 2202.10341 Table 1；2502.03369 Table（human demo success 0.97） |
| Bench2Drive（220 条） | 无人类 | — | PDM-Lite 专家 DS 约 97（检索摘录）；TFv6 约 95 | [检索摘录]；TFv6 见 decisions.md |
| CARLA Leaderboard 2.x | 无人类 | — | — | 官网无 human 行 [读 leaderboard.carla.org] |

这些"人类分"要这样读：

- **NAVSIM 的 human 分是被构造抬高的。** v1 过滤 navtest 时就把 human 轨迹 PDMS < 0.8 的场景删了（为了去掉标注噪声），所以 human 在剩下的集合上高分是选择出来的 **[读，2406.15349 §3.1]**。
  v2 的 EPDMS 更进一步：某项规则违规如果 human 在同一场景也犯了，agent 这项豁免（"human flag filtering"），于是 human 在所有惩罚项上结构性接近满分 **[读，arXiv 2506.04218 §3.1 及 Limitations]**。
  human 唯一明显丢分的是 EP（Ego Progress），因为 EP 以特权规则规划器 PDM-Closed 的最大进度归一化，论文原话是 EP "cannot be solved purely by human imitation" **[读，2406.15349 §5]**。
  所以"方法的 EP 超过 human"只说明它比 log 更激进地贴近 PDM-Closed 的进度上限，不说明开得比人好 **[推]**；PDMS 追平 94.8 同理，是在一个替 human 过滤过的考卷上追平。
- **nuPlan 的 log replay 在 reactive 闭环里只有 80**，低于规则规划器，因为 log 不会对被重新模拟的 agent 做出反应 **[读 + 推]**。这是"回放不等于驾驶"的直接例子。
- **Waymax 的 expert 碰撞率 0.61% 是标注噪声的下限**，论文自己这么解释 **[读，2310.08710 §5]**。
- 真正"人坐在模拟器里重新开"的参照只有 MetaDrive 那一类人在回路 RL 论文，而且都是同一位（作者团队的）"expert"，单人、非基准指标 **[读，2202.10341 §4.1 "One human subject participates in each experiment"]**。

## 2. 人在模拟器里开基准场景的研究

| 研究 | 设置 | 结果 | 与我们的关系 |
|:--|:--|:--|:--|
| BCI 驾驶（arXiv 2508.11805） | 20 名健全被试，手柄（右摇杆控速度 + 转向，左键全刹），CARLA LB 2.0 改的城区路线，**限速 5 mph**，练 15 min，每人 2 次 | 自定义 DS = C·0.8^Nc·0.9^Nl·0.9^Ns，平均 **0.823**（SD 0.160）；BCI 被试 0.924 **[读，Task 5 节]** | 唯一找到的"人按 CARLA LB 的违规计数被打分"的对照组；5 mph 下仍有碰撞；公式不是官方 DS，也没和任何 AV agent 比 |
| PersonaDrive（arXiv 2606.12616） | 8 名被试，Logitech G923 方向盘 + 三块 50" 屏（约 130° FOV），21 个 CARLA LB 场景 × 3 种风格指令 | 数据只用作风格检索库，**没有报告人类的 DS** **[读，§4.1；正文检索无 human DS]** | 证明 wheel rig 在 CARLA LB 场景上可行；没有人类分数 |
| DReyeVR（arXiv 2201.01931） | CARLA 上的 VR 驾驶 + 眼动平台，硬件 < 5000 USD | 平台论文，无基准分 **[检索摘录]** | 可选的沉浸式方案，对我们过重 |
| HACO / PVP（2202.10341，2502.03369） | MetaDrive / CARLA，人实时接管 | 人类专家 success 0.97–0.98 **[读]** | 人是专家、单人、非标准 DS |

**结论：没有找到任何人让普通被试开 Bench2Drive 或 CARLA Leaderboard 路线、用官方 DS 打分并和学习型 agent 并排比较的工作。** 这个空白是真的，不是检索不到（BCI 那篇最接近，但限速 5 mph 和自定义公式让它不可比）。

关于"陌生车辆动力学和模拟器本身"会让人开得多差：

- 模拟器效度：一篇 44 项"模拟器 vs 实车"对照研究的系统综述里，约一半的模拟器达到绝对或相对效度，约三分之一无效；速度是最稳定达到绝对效度的量 **[摘要/检索摘录，Wynne et al., Safety Science 2019]**。
  言下之意：人在模拟器里的速度选择可信，横向控制和错误数就不一定。
- 模拟器晕动（simulator sickness）导致的退出率约 13–17%，年长者更高 **[检索摘录，Brooks et al., AAP 2010，二手引用]**。
- 对陌生转向特性的适应：一项线控转向研究的摘要说新手司机对转向系统变化的适应"没有显著变化"，归因于经验少、不敏感 **[摘要，PMC11397869]**；这与"陌生车更难开"是两回事，只能说明新手不太会调整。
- CARLA 特有的问题（我们自己的观察，**[我们]** decisions.md 第 57 条）：CARLA 车辆在 1–2 m/s 轻刹就会刹停，纵向手感和真车差很多；键盘是开关量油门 / 转向，比方向盘更难平顺。

## 3. 只收轨迹的榜单的完整性

### 3.1 WOD-E2E

| 项 | 内容 | 来源 |
|:--|:--|:--|
| test 规模 | 1,505 个 test 片段，每片段一个目标帧（清单 `test_sequence_frames_for_submission.json` 共 1,505 帧） | [读] 2510.26125 §3.1；[我们] 提交时核对清单 |
| 参赛方拿到什么 | 目标时刻前约 12 s 的数据（我们实测中位 139 张 10 Hz 帧，约 14 s）、8 路相机、自车历史、routing 指令；之后 8 s 保留用于评测 | [读] 挑战页 waymo.com/open/challenges/2025/e2e-driving/；[我们] 榜单对照 A3 |
| 能不能手标 | **能**。test 相机帧就在参赛方手里，提交的是 20 个 (x, y) 点，服务器只看点 | [推] 由上一行 |
| 配额 | test 每 30 天 6 次，报错不计 | [读] 挑战页 |
| 核验 | 常设榜不要求技术报告或代码；元数据（参数量、是否用公开预训练模型）全部自报；挑战页明确允许用 MLLM 自动标注扩充**训练**数据，对推理阶段是否可有人参与没有任何条款 | [读] 挑战页；[读] 提交 proto |
| 已知讨论 | 没有找到关于 WOD-E2E 手标提交、LLM 生成提交或 test 泄漏的公开讨论 | 检索无结果 |

手标的成本很低 **[推]**：按每帧看 12 s 视频加画一条 5 s 轨迹约 1 分钟算，1,505 帧约 25 人时。
更实际的威胁是半自动：val 的 rater 标签是公开的，可以训练一个"像 rater 一样挑轨迹"的选择器，再对 test 帧做 best-of-K；
这在规则上合法（属于"用 val 训练"），但它把"模型能力"换成了"对打分器的拟合"。LLM 直接生成的提交确实存在且是公开的合法方法（LightEMMA 6.52、OpenEMMA 有公开提交代码），
它们目前的分数远低于榜首。

### 3.2 NAVSIM

NAVSIM 的私有 test 同样是上传式：参赛方用官方脚本在本地对 private test 的传感器数据推理，生成每个场景一条轨迹的 pickle，作为 HuggingFace 模型上传；
服务器端打分约 2 小时，每天 1 次 **[读，github.com/autonomousvision/navsim docs/submission.md]**。navhard 两阶段的第二阶段"合成观测"是预先生成好发给参赛方的，
不按 agent 的动作重新渲染 **[读，2506.04218 §3.2]**，所以同样可以离线手画 **[推]**。
约束来自社区规则而非技术：2024 榜会定期移除没有开源训练 / 推理代码和 checkpoint 的条目 **[读，同上 docs]**；
OpenDriveLab 2024 挑战要求 4 页技术报告，获奖队伍应要求提供代码或 docker 供核验，"attempting to hack the test set" 取消资格 **[读，opendrivelab.com/challenge2024]**。

### 3.3 闭环、服务器端的榜怎么避开

CARLA Leaderboard 2.x 要求把 agent 封装成 docker 镜像，通过 EvalAI 提交，在组织方的 AWS g5.12xlarge 节点上闭环运行，每月 20 次；资格赛在参赛方没见过的地图上跑
**[读，leaderboard.carla.org 首页与 submit_v2_1、evaluation_v2_1]**。agent 每一帧的观测取决于它自己之前的动作，参赛方拿不到固定的 test 输入，
所以离线手画在结构上不可能 **[推]**；剩下的理论漏洞只有"镜像里藏一个远程遥控通道"，这取决于评测节点的网络隔离，官网未说明 **[推]**。

Bench2Drive 不是服务器端评测：220 条路线公开，参赛方本地跑、自报 DS **[读，B2D 仓库与论文]**。它防不了"改评测代码或挑 seed"，但能防离线手画
（闭环里人要实时开，而且 B2D 自带的 `human_agent.py` 恰好就是键盘人类 agent）。它的可信度靠开源代码和第三方复现，和 NAVSIM 同一档 **[推]**。

先例：ILSVRC 2015，Baidu 用约 30 个账号在约半年内提交约 200 次，超出每周 2 次的配额，被禁赛 12 个月 **[检索摘录，MIT Technology Review 2015-06-04 等]**。
这是配额滥用（对 test 的自适应过拟合），不是手标，但说明上传式榜单的防线只在配额和账号层。

可信度排序（**[推]**）：CARLA LB 2.x（服务器端闭环）> Bench2Drive ≈ NAVSIM（本地或上传，靠代码公开）> WOD-E2E 常设榜（上传，无代码要求）。

## 4. 对我们的含义

**闭环服务器端榜更可信，但眼下它不是我们的主场。** 我们的闭环证据现在是负的或弱的：openpilot 在 B2D dev 路线上让它横向驾驶 DS 掉 30–48，只做纵向 modifier 时对无感知 base 的 +9.7 可以用"开得慢"解释（decisions.md 第 57 条，待定）。
CARLA LB 2.x 服务器是否仍在正常收提交没有核实。因此"换到更可信的榜"不是一个近期能兑现的叙事。

**WOD-E2E 这条分的防线应该是可复现性，而不是换榜。** 我们的提交用公开冻结权重、没有在 WOD 上训练、唯一拟合的是 val 上的一个 ×1.06 系数；
公开推理脚本、tar 包 SHA-256（`8d98dc2d…1bad`）和"重跑得到逐字节相同 protobuf"的说明，审稿人就可以验证这份提交确实来自 openpilot。这是上传式榜能给出的最强证据，成本约半天 **[推]**。

**人类基线：值得做，分两级。**

第一级（直接回答用户的假设，零 GPU）：在 WOD **val** 上让人手画轨迹，用我们已有的 RFS 实现打分。绝不把人工结果交 test（违背榜单精神，也浪费配额）。

- 被试 3–5 人（有驾照，不看 rater 标签）；从 val 479 帧按 cluster 分层抽 60 帧，其中 20 帧取 s_ego 最高十分位（logged future 在那里只有 5.92）。
- 工具：给出目标帧前 12 s 的 FRONT / FRONT_LEFT / FRONT_RIGHT 视频和 routing 指令，在 BEV 草图上拖出 3 s、5 s 两个点和速度，插值成 20 点（与 openpilot 同一输入契约）。
- 指标：RFS（cluster mean）、floored 比例，对 openpilot / logged future / constant velocity 做逐帧配对差，按帧 cluster bootstrap；另记每帧用时，给出"手画整个 test 要多少人时"的实测值。
- 成本：写标注页约半天，每人约 1–1.5 h（60 帧 × 1 min），合计约 1.5 人日。

第二级（闭环、可写进论文的新数据）：在 Tokyo box 的 CARLA 上让人开 Bench2Drive 路线，官方 DS 计分。

- 被试 4–6 人，有驾照、非游戏玩家与玩家混合；Logitech G29/G923 级方向盘（约 300 USD，没有就用键盘，但键盘作为单独一臂报，不混）。
- 路线：B2D 220 条里按能力类别分层抽 20 条（与 openpilot 仲裁臂的 dev 10 条有重叠，便于配对），另 2 条练习路线不计分；每人每条开 2 次，顺序随机。
- 实现：B2D / leaderboard 仓库自带 `leaderboard/autoagents/human_agent.py`（键盘）；按 CARLA `manual_control_steeringwheel.py` 加方向盘输入，约 0.5–1 天。Tokyo 3090 单卡，单实例即可。
- 指标：官方 DS / SR / 各类违规，和 openpilot 各臂、PDM-Lite、TFv6 按路线配对；练习前后做 SSQ（Simulator Sickness Questionnaire，模拟器晕动量表），晕动退出单独报。
- 成本：B2D 路线约 150 m，每条 1–2 min 加重置，一人 40 次约 1.5 h；6 人约 9 人时，加 1 天工程，约 2.5 人日。
- 预先写下的读法：人类 DS 若低于 openpilot 最好臂，支持"模拟器里的普通人并不是更高的参照线"；若远高于所有学习型 agent，说明 B2D 主要考的是"按路线开完"，也值得写。无论哪种，都要写明这是"人在 CARLA 里"，不是"人在真车里"（第 2 节的模拟器效度问题）。

优先级建议：第一级先做，它直接对着 RFS 回答用户的问题，而且结果决定第二级是否值得占用 Tokyo box；第二级是填空白的新数据，但只有在闭环线有正结果时才能进故事主线。

### 4.1 补录人类演示：静止障碍与行人（2026-09-30，用户回学校后自己录）

这一节和上面两级不是一回事：两级是**给分数**（人类参照线），本节是**补数据**（人类演示）。op-adapt r2 不用 expert 轨迹（预登记 Q2：行为监督只用规则打分），所以这些演示不进 r2；只有 r2 之后新开一个对照臂（暂称 A-human）或加一套读数考卷时才用。

优先录两类，因为候选集与规则给不出：

| 类 | 场景 | 录什么 | 为什么只有人能给 |
|:--|:--|:--|:--|
| 静止障碍 | 路上停着的车或障碍，占住本车道 | 停下，等约 5 s，确认它不动，再绕过并回到本车道；另录一批「等到它动了就跟着走」的 | r2 v5 把绕行候选整族拿掉了（V6 是不可行 offset 的产物），规则里只剩「等」，没有「绕」；绕行怎么起、多宽、什么时候回，只能从人的演示里学 |
| 行人 | 行人横穿、从路边突然进入路面、成组横穿 | 何时开始减速、停在离行人多远、行人走完后多快起步；同一场景各有「行人横穿」与「无行人」两版 | 可与 x⁺ / x⁻ 配对考卷对上；规则打分对「让行时机」只有二值判断（P 在 1 与 0.16 之间跳），人的减速曲线是连续的参照 |

后备（有余力再录，不在本次范围）：红灯与路口起步、带左右导航的转弯、偏离车道后的恢复。

约束：

- 录在 dev 与 train 的路线 / 场景上，**不碰那 19 条 held-out 路线**，否则闭环考卷被污染。
- 相机装置与我们生成 x⁺ / x⁻ 配对用的一致，特征才能直接复用。
- 每段 20–60 s，加重置约 1–2 min；每类 30–50 段，两类合计约 60–100 段，纯操作约 2–3 h，一天内够 **[推，没测]**。这个量够做小对照臂或额外考卷，不够从头训练。
- 如果要同时当人类基线（第二级），另按第二级的被试与计分流程录，不与本节演示混用。

## 5. 二次定向检索：CARLA 里人类驾驶得分（2026-09-29）

上一轮结论"没找到"过窄：它只覆盖了用官方 DS 打分的工作。这一轮放宽到任何 CARLA 基准指标、人机同场对比、以及近亲模拟器，结论分两层：
**用官方 DS 给人类开车打分并和 agent 并排的工作，仍然没有找到；但存在一篇"人 vs DRL agent，同一奖励函数打分"的 CARLA 论文（Yurtsever 2020），此前漏掉了。** 除此之外的命中都是"人开过车，但没报人类分数"。

### 5.1 直接命中与近似命中（CARLA）

| 来源 | 模拟器 / 基准 | 人数与设备 | 路线 / 场景 | 指标与人类得分 | 与 AV 的对比 | 核验 |
|:--|:--|:--|:--|:--|:--|:--|
| Yurtsever et al., *Integrating DRL with Model-based Path Planners for Automated Driving*, arXiv 2002.00434 | CARLA（自建 7 条路线，非官方基准） | 4 人（25–30 岁），键盘，看屏幕绿线导航，看不到分数 | 7 类路线（直道高速 / 城区 / 桥下 / 缓弯 / 急弯 / 路口右转 / 路口左转），每人每条 5 次 | 训练 DRL 用的累计 reward，不是 DS。人类均值：43.4 / 38.1 / 45.2 / 49.5 / -8.9 / -12.1 / -25.5 | 同表 Hybrid-DQN：21.1 / 27.6 / 31.6 / 30.4 / -74.4 / -136.9 / -385.9，人类在全部 7 类上更高（有碰撞即终止的 reward，转弯类差距最大） | **读全文**（Sec. IV-D 与 Table I） |
| Duan et al., *Enhancing E2E AD Through Synchronized Human Behavior Data*, arXiv 2408.10908（BCIMM'24） | CARLA 0.9.13 + DReyeVR，Longest6 评测 | 12 人，Logitech G920 方向盘 + 踏板，VR 眼动 + 64 通道 EEG | Town04 / Town07 默认路线上人机同路线开 | 人类只用作眼动 / 脑电 / 刹车数据源；**没有报告人类的 RC / IS / DS**；表中只有 MaskFuser 系 DS，最高 51.39 | 无人机对比 | **读全文**（相关段落与 Table 1） |
| PersonaDrive, arXiv 2606.12616 | CARLA LB 场景，Bench2Drive 评测 agent | 8 人，G923 + 三屏，非职业司机 | 21 个场景 × 3 种风格指令 | 人类数据用作检索库，无人类 DS | 只评 agent（no-style DS 88.95，SimLingo 85.07） | **读全文**（附录 E） |
| BCI 驾驶, arXiv 2508.11805 | CARLA LB 2.0 改编，限速 5 mph | 20 人，手柄 | Town12 城区 | 自定义 DS 0.823 | 无 | 读全文（上一轮） |
| Hi-OBC / LangAuto-Human, arXiv 2606.08170 | CARLA LangAuto | 人数未写，键盘接管 | 8 个 town、32 长 / 16 短 / 16 微型轨迹 | 人是接管训练信号，没有独立人类 DS | 只比 agent 加不加 MOBC | 读摘要 + 网页全文摘录，人数未披露 |

### 5.2 检索到但不算命中的 CARLA 工作

| 来源 | 为什么不算 |
|:--|:--|
| CARLA 原论文（Dosovitskiy 2017，arXiv 1711.03938） | 没有人类驾驶成绩；只提到 imitation 数据 80% 来自自动 agent、20% 来自人类司机 **[读，ar5iv 全文]** |
| QED（Quantitative Evaluation of Autonomous Driving in CARLA，OSTI 1814365） | "30 个 driver × 6 个 town"是 agent，人类是评分者，QED 与人类评分者的相关系数 0.96/0.97（easy）、0.84/0.74（hard）**[摘要 / 检索摘录]** |
| HABIT（arXiv 2511.19109） | 人体动作数据用于行人，没有人开车；InterFuser 5.24 碰撞/km，TransFuser 7.43 碰撞/km **[读网页全文]** |
| 神经认知奖励建模（arXiv 2603.25968） | 20 名被试在驾驶模拟器里记 EEG，用于 RL 奖励，没有人类驾驶分数 **[摘要]** |
| CARLA LB 2.0 官方"人工执行日志" | 官方训练数据里每个场景有人工执行的 100% 分示范，这是场景可解性证明，不是人类基线 **[官网文字，检索摘录]** |
| Take-over 研究（PMC9782608）、TeleCARL 等 | 人类接管 / 遥操作研究，关注反应时间和注视，没有 DS 类指标；**只读到搜索摘要，未读全文** |

### 5.3 近亲（其他模拟器，单独列）

| 来源 | 模拟器 | 人 | 指标与人类得分 | AV 对比 | 核验 |
|:--|:--|:--|:--|:--|:--|
| HACO, arXiv 2202.10341（ICLR 2022） | MetaDrive | 单个人类专家，方向盘 / 踏板 | test return 358.19±86.00，safety violation 0.16，success 0.98 | 同表 SAC-RS 386.77 / 0.73 / 0.82，HACO 等 | **读全文**（Table 1） |
| NAVSIM "Human Agent" | NAVSIM v2 navtest | 不是人在开，是 log 回放 | EPDMS 90.3（检索摘录） | 与各方法同榜 | 检索摘录；本文档第 1 节已讨论"回放不等于驾驶" |
| krishnasurya9/autonomous-driving-lab | MetaDrive | WASD 键盘 | README 里没有实际结果，仍在开发 | 计划比 LLM / RL | 读 README |
| Sky-Drive（arXiv 2504.18010） | 多人分布式模拟平台 | 人机协同平台论文 | 未看到人类基准分 | - | 只读标题 / 摘要 |

### 5.4 结论

- **官方 DS 下的人类基线：没有。** Bench2Drive、Longest6、Town05 Long、NEAT / TransFuser 路线都没有找到人类 DS 或 SR。Bench2Drive 论文里的"人类"只出现在舒适度阈值（人类专家数据设阈值）和 Think2Drive 专家数据，与人类开车得分无关 **[检索摘录，未读原文]**。
- **最接近的先例有三个，层级不同：** Yurtsever 2020（4 人键盘，同一 reward，人类全面强于 DQN，但不是官方指标）；BCI 2508.11805（20 人手柄，5 mph，自定义 DS）；Duan 2024（12 人方向盘，真人开了 Longest6 类路线，但没有报人类分数）。这三篇都不能直接拿来当"人类 DS"。
- 这个空白对我们仍是机会：第 4 节的第二级方案（Tokyo box 上人开 B2D 路线，官方 DS 计分）依然是新数据。Duan 2024 说明"方向盘 + G920 + CARLA 0.9.13 + 几十人时"的采集流程可行，PersonaDrive 的 rig（G923 + 三屏）是可以直接复制的配置。
- 局限：GitHub issue / Zhihu / CSDN / Bilibili 中文检索只走了搜索引擎摘要，没有逐帖翻；`human_agent` 关键词没有在 issue 里检索到人类得分讨论。这些只能算"没搜到"，不能算"不存在"。

### 5.5 本轮检索词

英文：human drivers CARLA "driving score" human baseline autonomous agents leaderboard；Bench2Drive human expert manual driving driving score human_agent；"human drivers" CARLA "route completion" user study steering wheel；CARLA Leaderboard human performance baseline Longest6 Town05 Long manual control；CARLA Dosovitskiy 2017 human driving performance；"driver-in-the-loop" CARLA Bench2Drive participants；human vs autonomous agent CARLA TransFuser/InterFuser/TCP；github Bench2Drive issue human driving manual_control；CARLA leaderboard 2.0 human drivers infraction score；human baseline CARLA closed-loop success rate；LangAuto-Human；QED human evaluators；human vs AI MetaDrive / highway-env / NAVSIM / nuPlan；teleoperated driving CARLA leaderboard；shared control / human-AI copilot CARLA Town05 Longest6；MetaDrive human expert success rate PVP HACO；human study perceived driving quality CARLA wheel。
中文：人类驾驶 CARLA 驾驶分数 Bench2Drive 人工驾驶 人类基线；知乎 CARLA 真人 手动 开 Leaderboard 路线 得分；Bilibili CARLA 人类 手动驾驶 Bench2Drive 得分 对比。

