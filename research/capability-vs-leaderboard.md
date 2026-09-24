# 榜单分数与驾驶能力：两层能力、分数里的配方，以及定向提取

状态: 骨架（2026-09-24）。论证部分已写；结果部分等下面四个 todo 回来再填。

## 问题

一个模型在开环或闭环榜单上分数高，不等于它会开车；一个很会开车的模型，在榜单上也可能表现一般。
专用驾驶模型和 VLM 在榜上往往只差几分（见 [openpilot-and-open-driving-models.md](openpilot-and-open-driving-models.md)），
这个差距反映的多半是针对榜单的调校，而不是智能上的差别。
如果能把「智能」和「针对榜单的配方」分别量出来，就可以从下载来的模型里有目标地提取各自擅长的部分，再合在一起。

## 驾驶能力分两层

**R 层**（routine，ego-based）：保持车道、直行、转弯、跟车、按信号灯停车、起步。这一层主要靠 continuation prior
（按 ego 自身运动和道路结构往前外推的先验），专用模型乃至只看 ego 历史的模型就能做得很好。P0/P1 里，RFS 口径上 ego-only 模型排第一。

**E 层**（emergent reaction）：routine 过程中出现突发事件时的反应，比如直行时有行人横穿要停下或绕开，转弯时有车要让。
这一层才需要对场景的理解，也是大模型应该发挥作用的地方。它对应第 25 条里的 reaction decoder。

两层之间有模糊地带：很多「突发」其实被 R 层的手写规则兜住了（TFv6 的 creeping、按规则停红灯），测 E 层时必须把这些剔掉。

## 已有的证据：同一个模型里，分数和能力分离

TFv6（Bench2Drive DS 约 95）有两个输出通道：

| 通道 | 用它开车时的 DS | P5 配对考试上的反应 |
|---|---|---|
| route + target speed | 比 waypoint 高约 14（第 31 条） | 定向翻转率 2%，天气一变就整档跳 |
| waypoint | 低 | 39%（逐帧），按对算 73% |

拿分的那一路不怎么对突发事件做反应，会反应的那一路开起来反而分低。

## 四个测量

| 测量 | 回答 | todo |
|---|---|---|
| 榜单 hack 审计 | 高分里有多少来自针对榜单的配方，这些配方都是什么 | [2026-09-24-hack-audit](../todos/2026-09-24-hack-audit/README.md) |
| zero-shot 考试（进行中，同事负责） | 真实世界模型不训练、原样进 B2D / NAVSIM / WOD-E2E 能拿多少分；它的 rig 和接法是其他测量的前提 | [2026-09-24-zeroshot-exam](../todos/2026-09-24-zeroshot-exam/bench2drive.md) |
| R 层测量 | routine 能力的排序是否等于榜单排序；TFv6 的分数里接口、规则、网络各占多少 | [2026-09-24-r-layer-routine](../todos/2026-09-24-r-layer-routine.md) |
| E 层测量（P5 v1） | 谁在突发事件时反应的方向和方式都对 | [2026-09-24-p5-v1-e-layer](../todos/2026-09-24-p5-v1-e-layer.md) |

四项合起来给出一张三轴图：榜单分 × R 层 × E 层。

## 定向提取的候选方法（等三轴图出来再选，现在不押注）

| 方法 | 提取什么 | 前提 |
|---|---|---|
| 差分蒸馏（Δ-distillation）：student 学 teacher 在 x⁺ 与 x⁻ 上输出之差，而不是它的绝对输出 | 只取 teacher 的 E 层反应 | 有配对；teacher 的 Δ 高于它自己的噪声地板 |
| Hydra 式多目标蒸馏：一个 trajectory vocabulary，多个 head 分别学人类示范和各项 metric 子分数 | R 层加上 metric 偏好 | NAVSIM 或 B2D 规模的数据 |
| 双系统（fast/slow）：R 层用小模型或配方，E 层用大模型，中间一个突发事件 gate | 两层分开来源 | gate 信号；大模型时延（Alpamayo 约 0.7 s）对高速 cut-in 可能不够 |
| 直接移植配方 | R 层里被审计确认可迁移的部分 | 审计结果 |
| 权重 merge | — | 必须同 base、同架构；我们的 zoo 高度异构，基本用不上 |

一个判断（推测，待三轴图验证）：R 层的配方可以直接搬过来，或者直接对 metric 做优化，并不稀缺；
稀缺的是 E 层，值得费力提取的也是它。

## 结果

- **榜单 hack 审计**（两轮只读材料的综合判读，2026-09-24）：见 [leaderboard-vs-ability.md](leaderboard-vs-ability.md)，45 个遗留问题的逐条回应在
  [synthesis_answers.md](results/leaderboard-text-analysis/synthesis_answers.md)。一句话：三类榜单三种增益构成（nuScenes = ego prior，NAVSIM v2 = EPDMS 代理评分器，
  Bench2Drive = 真能力 + 接口 + 规则）；能归到 E 层的增益全表只有四条，全在闭环或反应式协议上；上面"R 层配方不稀缺、E 层稀缺"的判断成立（decisions 第 35 条，待定）。
- zero-shot 考试、R 层测量、E 层测量：待填。
