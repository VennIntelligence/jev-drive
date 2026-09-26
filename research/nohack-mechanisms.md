# 没用技巧也拿高分的方法：机制、分类、能否在冻结特征 + 薄 head 里复现

状态: 调研报告，2026-09-26 由 Opus 子代理只读写成，主会话落盘；数字以所引出处为准，结论已进 [decisions.md](decisions.md) 第 46、47 条（就地修正见第 35 条）。

状态：只读调研（2026-09-26）。没跑模型、没上 box。数字出处写在每格或表后；账本行指 `results/leaderboard-text-analysis/w1_ablation_ledger.csv`（下称 W1），论文披露行指 `w5_paper_only.csv`（W5-编号）。
arXiv：FIVE-VLA 2609.18623、RoG-DAgger 2608.24525、LinkVLA 2603.01441、SteerVLA 2602.08440、BLUE 2606.08684、WA-JEPA 2608.20974、Poutine 2506.11234、NTR 2605.31116、OmniSpace 2606.22617。

## 0. 结论先行

1. 九个方法里没有一个是 (a) 类（只改分数、不改行为）。但有三处 (a) 类风险直接压在它们的分数上：SimLingo 定制 B2D 目录（simlingo#43，DS +11 而 SR 不变），它会波及 BLUE、RoG-DAgger、FIVE-VLA、LinkVLA、SteerVLA 五个 SimLingo 衍生方法；BLUE 静止 800 帧后强制给油门（B2D-BLUE-001）；NTR 用了一个没公开训练目标的 DrivoR 式 scorer（W5-037）。
2. 能落到 (c) 类（E 层 / 判断后行为）且超出噪声带的只有三条：**BLUE 的 gate**（突发 hazard SR 比 SimLingo +10.8 [3.1, 18.5]，第 38 条）、**FIVE-VLA 的 RAM**（同权重推理 +2.72 DS / +4.85 SR，但 Give_Way +26.7 这一项有口径疑点，见 1.2）、**RoG-DAgger**（SR +7.15，Fail2Drive SR +11，单次评测）。LinkVLA、SteerVLA、WA-JEPA、Poutine、NTR、OmniSpace 都归 (b)：改了行为，但增益落在总分或 R 层子项上。
3. 在我们的范式里，**tokenization ↔ `cls_late` 已经复现了**：NAVSIM 上 ridge 73.5 → cls 77.9（+4.4 PDMS）→ Hydra 式打分头 84.2（+6.3）（`openpilot-openloop-standing.md` 第 98–101 行；第 40 条 6），量级和 LinkVLA 的 +4.5 DS 相当。**V-JEPA 2 在我们手里没有复现**：冻结、pooled、ridge 的设定下它与 Qwen 同档（第 24 条 d‴、第 40 条），而 WA-JEPA 的 +5.7 是在 encoder 微调、全 token 网格读出、NAVSIM EPDMS 这三个条件下测到的，三个条件我们一个都不满足。
4. 能力包里最大的未知是 **Hydra 打分头和 gate 会不会把 P5 的反应 Δ 压掉**。这件事 CPU 上一天就能测，应该排在任何新 backbone、新数据之前。
5. 4.4 节「90→95 = LiDAR + 接口 + 规则」这句里，**「接口」一项的证据用错了比较对象**：第 31 条的 +14 是 TFv6 自己的 route+speed 对 TFv6 自己的 waypoint，而 SimLingo 系本来就是 path + speed waypoint 加两个 PID（FIVE-VLA p.7）。传感器那一项有 B 级证据（TFv6 纯相机 91.6 → +LiDAR 94.7），规则那一项只有 README 给的约 1 DS。另外，95.2 这个数本身在第三方单 ckpt 重跑里是 89.6（第 38 条）。

## 1. 逐方法机制与分类

分类依据（先定死再归类）：
- **(a) score/protocol-only**：策略输出不变、分数变。例子：评测目录、早停、重试、重打分。
- **(b) 改行为但对准 metric / R 层**：改变的是表征、读出、接口或训练数据，增益出现在总分（DS / PDMS / L2 / RFS），或者 R 层子项（DAC、EP、Merge、Overtake、nuScenes L2）上；也包括训练目标本身就是评测 scorer 的情况。
- **(c) E 层 / 判断后行为**：增益落在突发 hazard、恢复或泛化子集上（Emergency_Brake、Give_Way、突发 hazard family SR、Fail2Drive Gen、Longest6 RC），**超出噪声带**，并且机制本身讲的是「什么时候、怎么反应」。

噪声带：B2D 单次 209 条的 DS SD 0.80，两次之差 95% 带 ±2.3 DS / ±5.0 SR（第 38 条，BLUE 6 次）；3 个评测 seed 取均值时带宽约缩到 ±1.3 DS / ±2.9 SR（按 1/√3 推算，推测）；训练 run 间约 1.5 DS（`leaderboard-vs-ability.md` 6.2）。NAVSIM 训练 seed 间 SD 约 1.1 PDMS（LTF 3 seed，同上）。WOD RFS 我们自己 `cls_late` 的词表 seed 极差 0.18–0.26（第 40 条 R40）。

### 1.1 总表

| 方法 | 改了训练的什么 | 主增益（出处） | 增益落在哪 | 噪声带 | 类 |
|---|---|---|---|---|---|
| FIVE-VLA RAM | 结构：上一时刻 action token 经 2 层 transformer 编码成记忆，当前 action query 对它 cross-attn；第二阶段微调加入，跨时间 stop-grad；推理用双实例「accumulation」 | 88.49→90.95 DS、+4.24 SR（retrained，3 个评测 seed 的均值，p.7）；**同权重** bypass→accumulation +2.72 DS / +4.85 SR；预算配平后 +2.25（W1 T4/T7/T8/A.15） | Give_Way +26.67、Overtaking +7.4、Emergency_Brake +6.11、Smoothness +2.64、Efficiency −7.5（T4） | 同权重那一对在带外；EB +6.1 在 60 条路线 × 3 seed 上约 1.6σ（推测） | (c) 候选 |
| RoG-DAgger | 数据：在 student 闭环访问到的状态上让特权 expert 接管并重标，加轨迹 / 速度扩展、可挽回性触发、FoV 对齐 | 同 backbone 86.59→90.34 DS、+7.15 SR；Longest6 +22 DS / RC +18；Fail2Drive SR +11（W1 T1/T2/T4） | SR 涨幅大于 DS；RC（耐力）大涨；各组件单删 1.4–2.5 DS（在带内）、SR 3.6–6.9 | DS / SR 主对比都在带外；单次评测、无 seed | (c) |
| LinkVLA | 表征 / 读出：action 与 language 共用一个离散 codebook（action tokenization），再加 C2F 并行解码、反向语言对齐 | 85.07→89.57 DS（+4.50）/ SR +5.91；C2F +0.28、对齐 +1.16（W1 T5） | 没有分能力的结果；action dreaming 上 stop 60.9→99.9、换道 +13（T3） | 第一级在带外，后两级在带内 | (b) |
| SteerVLA | 结构：高层 VLM 输出 meta-action，低层 VLA 执行；VLM 事后生成的细粒度标签 | +2.87（分层），+1.90（标签 + reasoning）（W1 T1） | 未查到分能力结果；高层延迟 2.51 s（W5-005），没说仿真是否同步等待 | +2.87 贴带边，+1.90 在带内 | (b) |
| BLUE | 结构：冻结 SimLingo，最后一层 hidden state 接一个 0.11M 单隐层 MLP gate，逐帧决定生不生成语言；标签 = 训练路线（与评测路线不重叠）上「有语言 / 无语言」闭环成功率之差，再细化到帧 | 85.07→90.58 DS、+8.91 SR，**3 seed**（W1 T1）；Longest6 +14 DS / RC +14 | Overtake +12.6、EmBrake +11.6、Merge +7.7、TSign +7.5、GiveWay 0（BLUE Table 2）；突发 hazard SR 比 SimLingo +10.8 [3.1, 18.5]，集中在 junction_violator 和 cut-in（第 38 条） | 带外 | (c) 为主，混有「关掉有害模块」 |
| WA-JEPA | 预训练：V-JEPA 2 ViT-L 初始化，nuPlan 上做 future-masked 预训练（Stage 1）；Stage 2 在 navtrain 上 **encoder 以 lr 1e-5 微调**；4 路相机 × 4 帧历史的 token 进 MMDiT，flow matching 联合预测 scene 与 action | 换 encoder：MAE 83.8 / SigLIP2 83.1 / DINOv3 83.8 → V-JEPA 2 89.5 EPDMS（+5.7–6.4）；Stage 1 +2.2；joint + FM +1.8（W1 T4a–c；论文 p.6–7） | 只报 EPDMS 总分，encoder 消融的子项未查到；HUGSIM HDS 0.446 对 DrivoR 0.325 | 每个配置只训一次；论文的「10 seed」只是采样噪声，SD 0.05（论文 Table 7），不是训练噪声；+5.7 大于 NAVSIM 训练噪声 | (b) |
| Poutine | 数据：CoVLA 83 h + WOD 11 h 的 VLT 预训练（语言由 72B VLM 自动生成）；GRPO 以 RFS 为奖励 | 对未训练的 Qwen2.5-VL-3B 基线 5.59→7.95（+2.36）；CoVLA 再 +0.17；GRPO test +0.08（W1 T1；2506.11234） | 总分；对照组是「没在驾驶数据上训过」 | 带外，但比的是「训过 vs 没训过」 | (b)；GRPO 属 metric |
| NTR | 表征：masked latent 重建只经过一个紧凑的 scene-token 瓶颈，EMA teacher，SAM3 语义决定在哪里重建；推理时去掉重建分支 | 16 token 时 RFS +0.32；1 token 时 +0.01（W1 T4/T5）；token 两两相似度 0.95→0.41 | 总分 | +0.1–0.3 RFS，与我们自己的 seed 极差同量级，**大概率在带内** | (b)，另有 scorer 未审计 |
| OmniSpace | 表征：训练时用 VGGT 3D teacher 蒸馏、Plücker ray 注入 pose、epipolar attention；推理不用 teacher | L2 0.36–0.37→0.28（−0.08~−0.09 m），collision −0.09~−0.17 %，intersection −1.8~−2.3 %（W1 T1/T4） | nuScenes 3 s 开环，这个榜本身由 ego prior 主导（`leaderboard-vs-ability.md` 5.3） | 无 seed；不能当任何一层的证据 | (b)（R 层几何） |

### 1.2 必须同时读的限定

- **FIVE-VLA 的 Give_Way +26.67 有口径疑点。** B2D 的 Give_Way 只包含 InvadingTurn 与 YieldToEmergencyVehicle 两个 scenario，共 10 条路线（`jevdrive/tfv6_rules.py` 第 43 行）。按第 38 条，公开逐路线结果里所有学习式方法在 YTEV 上 SR 都是 0，所以 Give_Way 的上限是 50%；TF++ 0.4、SparseDriveV2 0.5、R2SE 0.5（`results/b2d-family/public/*/…ability.json`），BLUE 50.00±0.00（BLUE Table 2）。76.67 在 3 个 seed 下等于 23/30，意味着 YTEV 过了一半以上。BLUE Table 2 里 BevAD 也报了 76.67，所以这不是不可能，但 FIVE-VLA 没有逐路线结果可核。在核实之前，RAM 的 E 层证据应降级成「同权重推理 +2.7 DS / EB +6.1 / Overtake +7.4」，不能再以 Give_Way 为主。
- **BLUE 的增益有一半来自「关掉语言」。** 同权重下全开语言 SR 66.91、全关 69.55、gate（θ = 0.66）76.18（W1 F6）。gate 比两种固定模式里更好的那个还高 +6.6 SR，和噪声带边缘齐平。在**相同语言激活率**下，速度、加速度、转向、复杂度、随机五种启发式 gate 比学出来的 gate 低 0.7–3.9 DS、4.4–8.8 SR（W1 T7，同权重、3 seed）。所以「在哪一帧开」这个判断本身有价值，不只是开得少。跨 backbone 迁移：gate 挪到 SimLingo 上 +4.16 DS，挪到 CriticVLA 上 SR −0.22（T6）。
- **SimLingo 系共用的协议风险。** 第 35 条推测 SimLingo 系的 B2D 分数偏高，上限约 11 DS（simlingo#43）。但上表的方法内部 Δ 都是在同一个 agent、同一个目录下比出来的，协议偏差会大部分相消；受影响的是「跨方法的绝对名次」，不是 Δ（推测）。
- **RAM 与 ego 历史方向相反。** FIVE-VLA 显式喂 3 / 10 个自车历史 waypoint 时 DS −3.95 / −2.54（`leaderboard-vs-ability.md` 4.5）；RAM 记的是自己上一步的 action latent，DS +2.46。同一篇论文里，「记状态」和「记意图」符号相反。

## 2. 能否在冻结 backbone + 薄 head 里复现或检验

范式固定：openpilot `temporal`（512 维）冻结，head 是线性、MLP 或打分头；行人 / 近处 hazard 走 YOLO26x-seg embedding；反应靠配对差分 Δ（第 42 条）。GPU·h 按 RTX PRO 6000 估，CPU 活记 0。

| # | 文献机制 ↔ 我们的对应 | 能否 | 要什么数据 | 成本 | 预期读数与判据雏形 |
|---|---|---|---|---|---|
| T1 | **LinkVLA tokenization ↔ `cls_late` 词表 vs `ridge_late`** | 能，而且一半已经做过：NAVSIM ridge 73.5 → cls 77.9（+4.4）；WOD RFS +0.19 / +0.23（seed 0），3 seed 下随 seed 变（第 40 条）；静止帧上回归 head 比 cv 低 0.64–0.74，cls ≈ 0 | **缺 P5**：`cls_late` 在 P5 v1 BA 配对上的翻转率从没测过（`reactivity-program` 里没有这一行）。数据都在盘上 | CPU，< 0.5 GPU·h | 读数：行人 / cut-in 翻转、null false-flip。判据：cls 翻转 ≥ ridge + 5 pp 且 null ≤ 7% → 「离散化本身携带反应」；cls 翻转 < ridge → tokenization 只买 R 层（与 LinkVLA 在 action dreaming 上 stop 那一项的读法相反） |
| T2 | **BLUE gate ↔ G1 门控** | 能。G1 已登记 g₁（第 23 条 (e) 的 s_ego gate）、g₂（hazard probe）、g₃（lead TTC），但**三个都不是 BLUE 式的结果标签** | 加一个 g₄：标签 = 「这一帧加 Δ 是否变好」。CARLA 侧用 P5 配对（Δ 翻对为正，null 上动了为负）；真实侧用 WOD 按 s_ego 分档的 ΔADE 符号，rater 帧只有 479 个，太少，只能用来验证 | CPU，分钟级 | 判据在 G1 原判据之上补两条：(i) **gate-on 子集**上 Δ 的 CI 整体 > 0（避免 BLUE 那种「增益 = 关掉」）；(ii) gate 用在 P5 帧上时，行人翻转 ≥ 不加 gate 的 80%。只做到「无害」算安全过滤，不算能力 |
| T3 | **RAM ↔ openpilot temporal 的时间深度** | 部分能。`temporal` 本来就是循环特征，历史 ≥ 4.8 s 时 pre-onset 增益从 −0.14 变到 −0.20（backbones todo 敏感性），所以视觉时间深度我们已经有。RAM 的新东西是**记 action latent**，对应做法是把 head 自己上一步的输出（ŷ_{t−1}、Δ_{t−1}）作为输入，训练时 stop-grad，与 RAM 同构 | P5 v1 BA 的连续观测帧（obs / stream 表）；WOD 连续帧 | < 1 GPU·h | 读数：首次翻转相对 expert onset 的滞后（E4c 现在是 0.8 s）、逐帧翻转、null。判据：滞后缩短 ≥ 0.2 s 或翻转 +5 pp，null ≤ 7%，WOD pre-onset ADE 不变差 0.02 以上；副作用是 copycat（喂 ego 历史 −3.95 DS 的前车之鉴），所以同时报静止帧上相对 cv 的 Δ |
| T4 | **DAgger 恢复数据 ↔ 我们的开环配对** | 开环量不到。P5 的构造保证 ego 逐 tick 相同（第 32 条 96%），测的是「hazard 出现后反应」，不是「自己偏了之后回来」 | 两条路：(i) 开环代理：在 P5 生成器里加**扰动起点**的对（x⁺ = ego 横向 / 航向被扰动，x⁻ = 原状，expert 两侧重跑给出恢复修正），对应 RAP 的扰动数据（v1 0.0 / v2 +4.4）；(ii) 真 DAgger：反应通道接进控制器闭环，student 跑过的状态交给 PDM-Lite / BehaviorAgent 重标，2–3 轮 | (i) 按 P5 v1 的生成量估约 10–20 GPU·h（CARLA server，推测）；(ii) 每轮 220 条约 3 h × 若干 server（第 17 条），加重标，合计 20–60 GPU·h，另需先修好闭环（第 41 条） | (i) 恢复对上的定向翻转与幅度比；(ii) B2D SR 与 Longest6 RC。判据雏形：SR 增益 ≥ 5（超出单次带）、RC 增益 ≥ 10 |
| T5 | **WA-JEPA V-JEPA 2 预训练 ↔ 我们阶梯里的 V-JEPA 2** | 按 WA-JEPA 的条件不能复现（需要 encoder 微调和 token 网格读出）；按我们的条件已经测过：V-JEPA 2 ViT-L 冻结、mean、ridge，WOD 主表 pre-onset −0.049、全部帧 −0.030（比 Qwen A 的 −0.044 还差）、RFS +0.012；train 训协议下 −0.030 CI 跨零，而 openpilot 是 −0.294（backbones todo 主表；第 24 条 d‴）。**排名：通用 backbone 那一档，与 Qwen 同档，在 openpilot 和 Alpamayo 之后。** | 能做的最小对照：NAVSIM 上用冻结 V-JEPA 2 与 SigLIP2 的 token，接和 E6 同一个 Hydra 式打分头、同一个 K = 1024 候选集 | 抽特征 2–6 GPU·h（参照 WOD 50 万行约 70 min / 模型），打分头 CPU，devkit 约 1 h | 对得上 / 对不上的三个原因见表后。判据：V-JEPA − SigLIP2 ≥ +3 PDMS（WA-JEPA 差距的一半）且 CI 不含 0 → 「预训练目标的效应冻结后仍在」；否则 → 它需要微调才能体现，**不进我们的范式** |
| T6 | **Poutine 领域预训练 ↔ openpilot 对通用 VLM** | 已经复现了同一件事：驾驶视频训过的表征比通用 backbone 强一个量级（第 40 条，WOD 与 nuScenes 两个数据集，NAVSIM 第三个）。GRPO（RFS 奖励）只 +0.08，属于 metric 对准 | — | 0 | 不再测 |
| T7 | **NTR latent reconstruction** | 冻结范式里不能做：它改的是 encoder / 瓶颈的训练 | 只能在 YOLO / Qwen token 上接一个小 adapter 去做 | — | 增益 +0.3 RFS 在我们 479 个 rater 帧上测不动（seed 极差 0.18–0.26），**不建议** |
| T8 | **OmniSpace 3D 几何蒸馏 ↔ 第 45 条的 BEV 放置缺口** | 能，而且正对我们的瓶颈：20–40 m 行人图像上看见 46–61%，BEV 里只剩 12–22%（第 45 条） | 用 VGGT 或按标定尺度的深度 teacher 离线给 YOLO 框打接地点 / 深度，训一个小几何 head | 3–8 GPU·h（VGGT 离线跑 P5 与 nuScenes 子集，推测） | 判据就是第 45 条写好的推进条件：P5 / nuScenes 行人召回向 oracle 高度上界（0.70–0.81 / 0.52–0.67）靠近 ≥ 一半 |
| T9 | **SteerVLA meta-action ↔ 第 43 条 Q7 的纵向模式词表** | 能：decision head 在 {继续, 让, 停} 上分类，Δ 以决策为条件 | P5 的 expert Δ 可以直接离散成三类 | CPU | 警告：TFv6 的 target-speed 通道就是一个近二值的决策通道，它的翻转只有 2%，而只换天气就有 10% 的帧整档跳（第 32 条）。离散决策通道必须报 null 地板。判据：决策正确率 ≥ 连续 Δ 翻转率，且 null ≤ 7% |

**WA-JEPA 的 +6 与我们 V-JEPA 的 ≈0 为什么对不上**（按可能性排序，都是推测，T5 可以分开）：
1. **冻结 vs 微调**。WA-JEPA 的 encoder 以 lr 1e-5 参与训练（论文 Implementation details），我们完全冻结，而且 mean-pool 成一个向量。我们 P2(c) 在 Qwen 网格 token 上接 attention 读出没有买回东西，但 V-JEPA 的网格读出没测过。
2. **比较对象不同**。WA-JEPA 比的是 V-JEPA 2 对图像级 encoder（MAE / SigLIP2 / DINOv3，三者互相差 0.7 以内），我们比的是 V-JEPA 对 Qwen VLM。第 24 条 d″ 已经发现「因子是时间，不是 JEPA」：Qwen 原生视频与 V-JEPA 同档。WA-JEPA 的 +6 与「视频预训练 > 图像预训练」一致，与「V-JEPA > 视频 VLM」无关。
3. **指标**。EPDMS 被 DAC / EP 这类 R 层子项主导，我们的主判是 pre-onset ADE；V-JEPA 在我们表里唯一稳的信号是第 10 档对 rater_best 的 ADE −0.31~−0.34（第 24 条），也就是多模态顶档往 rater 偏好的那一支挪。它说明 V-JEPA 帮的是「选哪一支」，与 NAVSIM 打分同向，与 pre-onset 无关。

## 3. 能力包候选清单

目标：同一个冻结 openpilot backbone，同时 (i) 在 NAVSIM / WOD / B2D 上名次好，(ii) 在 P5 配对上保住反应。标记：**[文]** 文献证据，**[我]** 我们量过，**[推]** 推测。

| 组件 | 作用于 | 已知增益 | 与反应 Δ 的关系 | 叠加性 |
|---|---|---|---|---|
| K = 1024 词表 + `cls_late` | 候选 + 模仿选择 | [我] NAVSIM +4.4 PDMS（对 ridge）、WOD +0.2 RFS（随 seed 变）；[文] LinkVLA +4.5 DS | 未知（T1） | 和 Hydra 共用同一个候选集，可叠 |
| Hydra-MDP 式 PDM 子分打分头 | 选择 | [我] +6.3 PDMS，主要来自 DAC 87→93，EC 80→75（第 40 条 6）；只在有 PDM scorer 的榜（NAVSIM）上成立 | **[推] 有冲突风险**：如果把 prior+Δ 作为候选交给它，它按非反应式 log 上的 NC / TTC 选，行人帧上没被 log 触发的反应会被丢掉。PDM 对「继续 / 刹停」与人类的一致率只有逐 token 63% / 逐对 79%（第 44 条 E3） | 和 `cls_late` 是「谁来选」的**互斥**，只能像 Hydra-MDP 那样加权混合；和 Δ 的顺序要定（见下） |
| 配对差分 Δ（M-C / 20 Hz student） | 选完之后的修正 | [我] CARLA 行人 43–53%；真实数据上直接相加有害（WOD RFS −1.0~−1.5，NAVSIM PDMS −8，第 44 条） | 本体 | 必须配 gate |
| 真实数据 gate（G1，加 BLUE 式 g₄） | Δ 的开关 | [文] BLUE +5.5 DS，一半是「关」；[我] 第 23 条 (e) 的 gate 学得会「什么时候先验失灵」，但开了不买东西 | **[推] 有冲突风险**：用真实数据训的 gate 如果学成「几乎总关」（E1 里 Δ 在真实特征上处处有害，最省事的解就是关），P5 反应就丢了 | 判据要写成两边都保：P5 保留率 ≥ 80%，真实数据无害 |
| action memory（RAM 式） | head 输入 | [文] +2.3–2.7 DS，同权重 | [推] 可能缩短反应滞后，也可能引入 copycat | 和 `temporal` 可能部分冗余（T3 测） |
| 恢复数据（扰动对 / DAgger） | 训练数据 | [文] RoG SR +7、RAP v2 +4.4 | 与配对**互补**：配对教「看见了怎么反应」，恢复教「偏了怎么回来」 | 可叠，要闭环 |
| 测距传感器 | 输入 | [文] TFv6 +3.1 DS / +6.1 SR | — | **不在范式内**（openpilot 只有相机），B2D 上这 3 分我们拿不到 |
| 几何蒸馏（接地点 / 深度） | YOLO 通道 | [文] OmniSpace 只在 L2 上；[我] BEV 缺口已定位（第 45 条） | 抬高远处行人的可用率 | 可叠 |

组装顺序建议（**[推]**；T10 = 「榜单 head × 反应 Δ」相容性检验，内容见第 5 节建议 1）：proposals（K = 1024）→ 选择（NAVSIM 用 Hydra、WOD 用 `cls_late`）→ 在**选出之后**加 g(x)·Δ(x)。这样打分头没有机会丢掉反应；代价是 Δ 可能把轨迹推出可行驶区域、掉 DAC。反过来把 prior+Δ 作为候选交给打分头，DAC 安全，但反应由 PDM 决定。两种顺序都要在 P5 与 NAVSIM 行人 token 上测。

**BLUE 的 gate 一半是「关掉语言」，对我们意味着什么**：我们的 Qwen 流就是 BLUE 意义上的「语言 / 慢通道」。E1 显示 Qwen 那一路的 Δ 在真实数据上有害，I3 上 Qwen `ridge_late` 只有 2.8%（real-data-transfer todo 起点表）。所以一个 gate 在榜上的增益很可能大部分是「把 Qwen 关掉」，和 BLUE 的结构一模一样。这不是坏事，但要把「关掉有害模块」和「在对的帧打开」分开报（T2 判据 (i)）。20 Hz student（不含 Qwen，第 42 条 E5）已经是「永远关语言」那一臂的现成版本，可以直接当 BLUE 那个 69.55 的对照。

## 4. 「90–91 对 95.2 = LiDAR + 接口 + 规则」的证据强度

| 分项 | 4.4 / 第 31、38 条里的证据 | 等级 | 问题 |
|---|---|---|---|
| 差距本身 | 作者 TFv6 95.2（3 个独立训练 seed，tfv6 T5）；**第三方单 ckpt 在 209 条上 89.6，与 BLUE 90.6 在噪声内**（第 38 条） | 冲突 | 90→95 这 5 分是不是真实存在都没定；lead#72 说作者的 Longest6 均值 "not true mean"，lead#89 有失败路线重试。我们自己的 TFv6 A1 在 209 条上的两次（T2 / T3）还没出结果 |
| LiDAR / 测距 | TFv6 Table 5：360° 纯相机 91.6 → +LiDAR 94.7（+3.1，SR +6.1）；只加 radar +2.6；两者叠加只再 +0.3；换 RegNetY +0.5；FoV −0.3 | B（3 seed，retrained） | **纯相机的 TFv6（91.6）已经和 VLA 的 90–91 同档**，所以 TFv6 论文内部的分解支持「差距 ≈ 测距传感器」。但第 38 条显示 TFv6 的优势在 unprotected_turn（82 对 47）和 obstacle_bypass（89 对 80），突发 hazard 上反而低 10.9 SR。7.1 节把 LiDAR 标成「E（感知）」与此不符，它更像是帮了让行时的 gap 判断和绕行（推测） |
| 接口 | 第 31 条：同一 TFv6 checkpoint，route+speed 对 waypoint +14.3 DS [5.1, 25.9] | 这个比较本身 A | **比较对象用错了**：SimLingo 系本来就输出 path + speed waypoint，用两个 PID（FIVE-VLA p.7），已经是解耦接口。+14 解释的是「TFv6 为什么不用自己的 waypoint」，解释不了「VLA 为什么比 TFv6 低」。TFv6 的 target-speed 分类与 SimLingo 的 speed waypoint 之间还有没有差别，没测 |
| 规则 | TFv6 README：三项启发式联合开关 95→94（B2D-TFV6-002/003）；BLUE / SimLingo 有静止蠕行（B2D-BLUE-001），影响 `none` | C | 双方都有规则，差别不大；SimLingo 系规则的触发率与影响都没有数字 |
| 专家数据 | TFv6 T1：LEAD 数据 vs PDM-Lite 数据，B2D +1.38、Longest6 +11.5 | B | 4.4 没把它算进去；VLA 用的是 PDM-Lite（SimLingo 数据集） |
| 评测目录 | simlingo#43：DS +11 而 SR 不变 | 推测 | 方向和上面相反：如果成立，VLA 的 90–91 是**偏高**的，真实差距比 5 分更大 |

判读：4.4 那句应改成「TFv6 论文内部，纯相机 → 加测距传感器是 +3.1，纯相机版本已与 VLA 同档；接口 +14 是 TFv6 内部两种读法的差，不是它与 VLA 的差；规则约 1 且双方都有」。**证据强度：传感器 B；接口对这个问题不适用；规则 C；差距本身在我们的 TFv6 数字出来之前未定。**

还缺：(1) 我们自己的 TFv6 A1 在 209 条上的两次评测，加 family 拆分（已在跑）；(2) SimLingo 同一 ckpt 在原版目录与定制目录下各跑 3 次（第 35 条实验 2，未跑）；(3) TFv6 纯相机 ckpt（91.6 那个配置）的 family 拆分，用来把「LiDAR 帮的是哪一类路线」变成数字；(4) FIVE-VLA / LinkVLA / RoG-DAgger 都没有逐路线结果，只能用各自报的 multi-ability，没有 CI。

## 5. 给主会话的三条建议

1. **先跑「榜单 head × 反应」相容性矩阵（T1 + T10，CPU，约 1 天）**。在 P5 v1 BA、I3、NAVSIM 897 个行人 token 上测三件事：`cls_late` 与 Hydra 打分头各自的翻转率；g·Δ 放在选择之前还是之后；NAVSIM PDMS。判据：P5 行人翻转 ≥ 不加榜单 head 时的 80%，null ≤ 7%，NAVSIM PDMS ≥ 83.7（E6 的 CI 下端）。这一步决定「能力包」是能叠还是互斥，比任何新 backbone 都便宜，而且会改变 G1 / G2 的设计。
2. **G1 加一个 BLUE 式的结果标签 gate（g₄），判据里加「gate-on 子集为正」和「P5 保留率」两条**。否则 G1 的「成立」可能只是学会了把 Qwen 关掉，与 BLUE 那一半增益同质；20 Hz student 已经是现成的「永远关」对照臂。改的是登记，不增加算力。
3. **就地修正 4.4 与第 35 条里的「接口」归因，并把 RAM 的 E 层证据从 Give_Way 改到同权重推理与 EB / Overtake**。SimLingo 系已经是 path + speed PID，+14 用错了比较对象；Give_Way 76.67 超过所有公开学习式方法的 50% 上限，没有逐路线结果之前不能当主证据。V-JEPA 冻结 NAVSIM 对照（T5）和 DAgger 闭环（T4-ii）暂缓：前者预期落在「需要微调」的那一格；后者要等反应通道接进控制器之后再做。
