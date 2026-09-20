# Frozen VLM + 薄 head 的轨迹规划

状态: 主线主题，2026-09-20 立项。前身是 [qwen-latent-driving.md](qwen-latent-driving.md)（probe 路线，
已降级为本主题的 analysis 部分，结论见那份文档）。

## 一句话

一个**完全冻结、没有做过任何 driving adaptation** 的 Qwen3-VL-4B，把它中间层的 hidden state 接一个
薄 head，在固定的 trajectory vocabulary（从训练集聚类出来的 K 条候选轨迹）上打分，选一条输出。
问的是：这样一个不训练 backbone、可以把特征一次性缓存下来的系统，能在 Waymo E2E 上走到哪里，
代价是多少毫秒、多少可训练参数。

## 为什么换掉原来的 probe 路线

原来的问题是"Qwen 的 latent 里有没有 driving intelligence"。两轮 deep research 之后有两个发现：

- **这个问题已经被人做过一半。** arXiv:2603.06054（2026-03）在 CARLA 上对 Qwen3-VL-2B 做了逐层
  linear probe（冻结 backbone，只训一个线性分类器，用来测某个量是否线性可读），还比较了 probe、
  自由生成和 constrained output，甚至做了 activation steering。我们能剩下的新意只有
  "信息是丢了、迁移了，还是读取方式不对"，那是一篇 interpretability 论文。
- **那种论文在自动驾驶组里拿不到 credit。** 结果是一堆 ablation，没有一个能打榜的系统。
  做 AD 的人看它不像 AD，做 ML 的人看它不够新。

所以主线改成一个**有榜单数字的 system paper**，probe 的结论（见下面"已知结论"）降级为论文里
解释"为什么取这一层"的 analysis section。

## 方法

| 部件 | 选择 | 为什么 |
|---|---|---|
| Backbone | Qwen3-VL-4B-Instruct，bf16，**完全冻结**，无 LoRA、无 SFT、无 RL | 特征可以一次性抽完缓存，训练只在薄 head 上，单卡可复现 |
| 取哪一层 | LLM 中层（nuScenes 上 L19-L22 最好，Waymo 上重测） | 见"已知结论" |
| 输入 | 前 3 个相机（FRONT / FRONT_LEFT / FRONT_RIGHT）+ ego past state + intent | Waymo E2E 官方提供的输入 |
| 对照 backbone | DINOv2（单帧自监督）、V-JEPA 2（视频自监督）、SigLIP2（图文预训练） | 见下面「对照 backbone 怎么选」 |
| Head | 在 K 条 anchor trajectory 上打分，取 argmax；可选一步 refinement | Jev-style 的"一次决策"，K 从 3 类 maneuver 换成 K 条轨迹 |
| 输出 | 未来 5 s、4 Hz、20 个 XY waypoint | Waymo E2E 提交格式 |

Trajectory vocabulary 这个做法本身不是我们发明的，Hydra-MDP（arXiv:2406.06978）、DiffusionDrive
（arXiv:2411.15139）这一系在 NAVSIM 上早就在用。我们的组合是：**冻结的通用 VLM 中间层 + 固定词表打分**，
并且把每一项增益拆开报。

## 主张（按证据强度排序，不是按好看排序）

1. **视觉增益。** 在同一套词表和 head 下，图像特征相对 ego-only / constant-velocity 的增益有多少。
   Waymo E2E 上**没有任何已发表的 ego-only baseline**，这张表我们第一个填。
2. **分类 vs 回归。** 同 backbone 下，固定词表分类相对连续回归损失多少 ADE。
   Waymo 上也没人做过 K sweep，没人报过 oracle minADE 与实际 ranking error 的差。
3. **冻结的代价。** 完全冻结相对做过 driving adaptation 的同规模模型差多少。
   对照是 Qwen-Drive-1.0-4B（arXiv:2609.00111），它自己报过一个未适配 Qwen3.5-4B + Planning Expert
   的 val RFS 7.88，是最近的先例。
4. **效率。** 同一张 RTX PRO 6000 上，batch 1 的完整延迟与可训练参数量，和竞品并排。
   实测参考：Qwen-Drive-1.0-4B 直接规划 702 ms、带 reasoning 1258 ms（见 [docs/baselines.md](../docs/baselines.md)）。

## 已知结论（来自 probe v1，nuScenes trainval，n=26491）

这些是我们自己跑出来的，直接决定上面的层选择，细节见 [todos/2026-09-19-probe-v0.md](../todos/2026-09-19-probe-v0.md)。

| feature | macro-F1 | turn recall | hard turn recall |
|---|---:|---:|---:|
| ego_rule（不训练，外推当前 yaw rate） | 0.783 | 0.712 | 0.000 |
| ego probe | 0.774 | 0.574 | 0.010 |
| DINOv2 patch mean | 0.596 | 0.342 | 0.191 |
| Qwen vision encoder | 0.550 | 0.282 | 0.165 |
| **Qwen L19-L22 mean** | **0.650-0.659** | 0.42-0.45 | 0.25-0.26 |
| Qwen L36 last | 0.565 | 0.301 | 0.188 |

三条：layer curve 是干净的倒 U，LLM 中层比它自己的 vision encoder 高约 10 个点，比 DINOv2 高约 6 个点，
末层又还回去；全集被 ego 惯性统治（图像最好 0.66 < ego-only 0.77）；在 hard subset（当前还没开始转、
但未来要转的样本）上，ego probe 的 turn recall 塌到 0.010 而 Qwen 中层是 0.246。

**但第三条不要写成"视觉的增量集中在 hard subset"。** planner v0 在同一批数据上做了轨迹任务的
difference-of-differences（onset 的增量减去 straight 的增量），结果是 **−0.004 m，CI 半宽 0.035 m**，
也就是四个子集的增量几乎一样，probe 那个"集中"的模式**在轨迹任务上没有复现**；
信任域口径上 onset 甚至反而更差（DiD +0.060，显著）。两个任务问的不是同一件事
（"会不会转" vs "轨迹长什么样"），而 nuScenes 的 onset 子集只有 135 帧，先天测不动。
**这正是 claim 1 要在 Waymo 上重做的事**，判据已经预登记在 [decisions.md](decisions.md) 第 3d 条。

## 对照 backbone 怎么选

原计划里的 DINOv3 用不了：**访问申请被作者拒绝**（2026-09-20，理由未给；社区记录显示 Meta 系门禁
对中国大陆申请普遍拒绝）。被拒之后不能重新申请，我们也不去找第三方转存——作者是明确拒绝，
绕过去在合规上站不住，论文里也不好写。详见 [decisions.md](decisions.md) 第 12 条。

替换成三个都没有门禁的 backbone，而且这三个各自回答一个具体问题，不是凑数：

| backbone | 回答什么问题 |
|---|---|
| DINOv2（已在盘上） | 纯视觉的单帧自监督特征能做到多少？probe v1 和 planner v0 的对照就是它 |
| **V-JEPA 2** | **时间预训练能不能补上单帧缺的那一块？** 单帧结构上估不出速度，而我们的核心争议正是
"转弯开始之前视觉够不够"。这个对照比 DINOv3 更贴题 |
| SigLIP2 | 增益来自语言对齐，还是来自视觉预训练本身？ |

V-JEPA 2 这一条其实比原计划更强：文献里 Drive-JEPA 那条线用的就是它，对照有现成的外部参照。

## Benchmark

| Benchmark | 角色 | 说明 |
|---|---|---|
| Waymo E2E | 主 | 有公开 leaderboard，有 FROST-Drive 这个 frozen-vision 的直接对手。**RFS 本地能算**：每段 val sequence 恰好有一帧带 rater label（在 12 s 处，和 test 提交帧同一时点），完整 val 有 479 帧，`jevdrive/waymo.py` 里是官方实现的 bit-exact 移植。test 分数仍要提交（每 30 天 6 次） |
| NAVSIM v2 navhard | 次 | AD 社区更认的 planning 评测，能跑 EPDMS。只下相机，不要 LiDAR |
| nuScenes | 预演 | 已在盘上，用来把 pipeline、词表、指标表跑通，不作为论文结果 |

## Open questions

- **词表 K 取多少、怎么建。** 5 s / 20 点的 Waymo 上没有公开的最优 K，NAVSIM 上 Hydra 用 4096-8192、
  WoTE 用 256。要自己扫，并且拆开 coverage error（oracle minADE）和 ranking error。
- **三个相机怎么喂。** 一次 forward 三张图，还是每张单独抽特征再拼？token 数和延迟差很多。
- **图像历史。** Waymo 的分片里是打乱的单帧，要按序列名重组才能拿到历史。历史是先做轻量
  temporal head（Stage B），还是直接进 VLM 的长上下文（Stage C），要看 oracle 增益。
- **早退是不是真的。** 只在中层取特征、但仍然跑完整个 LLM，不算节省。要真截断才能声称。
  planner v0 已经实测：真截断到 L22，batch 8 省 31%，但 batch 1 只省 7%（launch overhead 主导）。
- **ADE 已经饱和，主指标必须是 RFS。** 官方 ADE 是对着评分最高的 rater 轨迹算的，
  logged future 自己只有 2.63 m，而公开最好的 test ADE 是 2.65 m。所以 ADE 上没有空间，
  RFS 才有（rater 上限 9.53，logged future 8.08）。见 [decisions.md](decisions.md) 第 2 条。

## 分阶段

第一篇论文做 A 和 B，C 只做到 oracle 分析为止，gate（D）留给第二篇。理由：没有 C 稳定赢 B 的证据，
gate 无从谈起；而 fast/slow gate 这个方向已经很挤（ASSCG、AdaThinkDrive、FIVE-VLA）。

| 阶段 | 加什么 | 进入下一阶段的条件 |
|---|---|---|
| A | 当前帧视觉 + ego/intent，固定词表 scorer | 视觉增益稳定超过 ego-only |
| B | 同样特征，加轻量 temporal state | 增益来自真实历史，不是泄漏或参数变多 |
| C | 同样历史、同样 K，换成 frozen LLM 的长上下文 scorer | 只做 oracle 分析：C 相对 B 的增益集中在哪些样本 |
| D | gate，在 B 和 C 之间分配计算 | 留给第二篇 |

## 参考

两轮 deep research 的完整报告在 `research/lit/`（不进 git）：
`driving_intelligence_research_2026-09-19.md`（领域地图）、
`2026-09-round2-waymo-thin-head.md`（榜单、词表、数据、延迟口径）。

- Waymo E2E 官方协议: https://waymo.com/open/challenges/2025/e2e-driving/
- FROST-Drive（frozen vision encoder + adapter + GRU，test RFS 7.856）: https://arxiv.org/abs/2601.03460
- Qwen-Drive-1.0（Qwen3.5-4B + Planning Expert）: https://arxiv.org/abs/2609.00111
- Hydra-MDP（trajectory vocabulary + 多目标蒸馏）: https://arxiv.org/abs/2406.06978
- DiffusionDrive（20 条 anchor + 截断扩散）: https://arxiv.org/abs/2411.15139
- Is Ego Status All You Need（open-loop 的 ego shortcut）: https://arxiv.org/abs/2312.03031
