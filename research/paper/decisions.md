# 论文决定记录

只放论文本身的决定：主线、标题、论点、被测对象、论文实验（[experiments/wm_surplus/](../../experiments/wm_surplus/README.md) 的 P0–P6）的结论、投稿事务。
打榜与训练配方的决定仍在 [../decisions.md](../decisions.md)（总表），这里引用总表的条目写「总表第 N 条」，本表的条目写 PD N。
格式与维护规则同总表：一行一条，写错就地改并写明原来说了什么、为什么变；证据：强 / 中 / 弱；状态：已确认 / 待定。
论文骨架（给人看）：[../paper-outline/index.html](../paper-outline/index.html)。

更早的论文方向条目留在总表：第 21 条（AD 方法论文、交付闭环 driver）、第 177 条（世界模型训练场废弃、后果学习候选，已被 PD1 取代）。

| PD | 决定 | 证据 | 状态 |
|:--|:--|:--|:--|
| 1 | 论文主线重定（用户 10-22，取代总表第 177 条的未定稿候选与 10-08 的 trick or trade 骨架）：标题 **Do Driving Policies Need Foundation World Models Yet? Body Schema Learning for End-to-End Autonomous Driving**。主张：现阶段端到端驾驶的瓶颈是开不出来（转弯、变道、到点停车、起步时机），不是看不懂；把通用基础世界模型（Qwen、V-JEPA / WA-JEPA）直接嫁接到驾驶上现在是过剩的；方法 = 身体图式学习，即现有 S 行（自己闭环状态上、无专家目标、对当前 plan 车身扫掠的后果 hinge），Cinque 不换、测试时无新模块。公平对比（自训世界模型）不做，改为用量探针（嫁接模型的驾驶头读了多少维 / 删层 / 因果归因）+ Cinque 加身体图式追平 WA-JEPA；措辞只说「不需要嫁接的通用世界模型」，不说「不需要预测」。查新 [lit/2026-10-22-do-we-need-world-models.md](../lit/2026-10-22-do-we-need-world-models.md)：驾驶里无人以此为主张、无标题撞车，最近的是 CoRL 2026 workshop *Do Robots Need World Models?*；威胁 Fast-WAM（训练时视频共训有用）、Richens et al. ICML 2025、2610.09695（更好的视觉表征会涨分）。骨架 [paper-outline/](../paper-outline/index.html)；实验 P1–P6 在 10-31 关榜后启动；WA-JEPA 在 navtest 上仍领先约 2.2（91.71 对 SH30 89.55，总表第 144、170 条），navhard / HUGSIM 已基本追平（总表第 144、148 条），当前谱系没读过；P2 不成立则降为分析文（原为总表第 452 条，2026-10-23 移到本表） | 方向决定（用户） | 生效；10-31 前仍先拿分 |
| 2 | 被测对象只用纯视觉模型（用户 10-23）：我们的模型是纯相机输入，对比对象也必须是纯相机；不下载 LiDAR。WoTE 与原版 DiffusionDrive（相机 + LiDAR）退出；改为 ReCogDrive（若推理不消耗 LiDAR，只改传感器配置并验证逐位一致后才用）、AutoVLA（Qwen2.5-VL）、DriveVLA-W0（Emu3 基础世界模型）作被测对象，纯相机 DiffusionDrive（SimScale）/ LTF 作非基础模型对照；清单 [shortlist.md](../../experiments/wm_surplus/results/p0_subjects/shortlist.md) | 规则（用户） | 生效 |
| 3 | 投稿事务：CVPR 2027 OpenReview 注册（标题、摘要、作者）2026-11-10，全文 11-16，补充材料 11-23，都按 AoE、不延期；作者名单 11-10 后只能调序不能增删；每位作者要有完整的 OpenReview 主页与 CVPR 作者注册表（缺则 desk reject）；全文须有 LLM Usage Statement；11-10 前至少一位作者够格当互审审稿人（按主页发表记录）。作者沿用已有 OpenReview 账号并加机构邮箱，不新建 | 官方 CFP / Author Guidelines（10-22 抓取，标题摘要可改一条来自摘要抓取未见原文） | 生效 |
