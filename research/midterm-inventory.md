# 中期成果盘点（2026-09-29）

这份文档是为约 10-09 的中期汇报（向领导汇报，争取下一阶段经费）做的成果盘点：现在手上有什么、哪些拿得出手、故事还缺什么、哪些条目经不起追问。
来源是 [decisions.md](decisions.md) 全部条目（第 1–59 条，含 3b / 3c / 3d，另有第 8–10 条编号重复，见第 6 节）、[README.md](../README.md) 索引里的文档，
以及它们引用的 `todos/`、`research/results/`。排期依据 [tmp/2026-09-29-midterm-plan.md](../tmp/2026-09-29-midterm-plan.md)（D0–D10）。
没有跑任何实验，没有动 box；所有数字抄自 decisions 和它引用的结果表，个别「推算」处已标明。

论文主线是「trick or trade」：在工业数据上训练的模型（openpilot、V-JEPA）轻度适配，比为榜单调过的模型更能开；要把「分数」和「能力」分开量。
主方法是 openpilot 加 JEPA latent 世界模型，CARLA 只提供干预数据和最终验证；验收原则是「真学会了，虚拟和真实场景都应该会开」。

## 0. 结论先行

1. **最硬的东西集中在「分数不是能力」和「openpilot 换个喂法 / 换个 head 就有大变化」两块。** 领导能听懂、我们数字也扎实的三件事：
   同一个 openpilot，只修输入契约（input contract：榜单给 agent 的输入是什么、几帧、什么时间轴）NAVSIM navtest PDMS 从 52.1 到 84.2；
   冻结 openpilot `temporal` 特征加薄 head，在 WOD 和 nuScenes 两个数据集上都比所有通用 backbone 好一个量级；
   同一份冻结特征，只把选轨方式换成「对准 metric 的打分头」就加 6.3 PDMS，而行为没有任何改善。
2. **验收原则本身现在没有任何一个正例。** 配对差分（对同一场景的有 / 无 hazard 两个版本，只对输出之差做监督）在 CARLA 里能激发行人反应（43%，3 seed），
   但加到真实数据上有害（第 44 条，四条路都没通）；只用真实数据改 openpilot vision 层，行人可读性 +0.08，没过线，也带不到 CARLA（第 55 条）。
   「sim 和 real 都会开」在中期只能作为目标和 sim+real 混训的在跑实验汇报，不能作为结果。
3. **闭环和世界模型两块目前没有正面结果。** openpilot 进 B2D 闭环：五种仲裁都不满意，唯一看起来加分的 e2e（对 base +9.7）被同均速对照解释掉（第 57 条）；
   HUGSIM zero-shot 考试只有预登记，没有分数。世界模型第一次实验（W）判据全不过，事后诊断归因于数据里的动作-场景混淆，WL 重做的前置检查还没出数（第 54 条）。
4. **决策日志里有一个叙事缺口。** 第 21 条（2026-09-22）写的是「方法论文，交付物是闭环里开得更好的 driver，主表 B2D + HUGSIM + NAVSIM」，
   之后没有一条记录改成「trick or trade」。README 和 memory 里已经是新主线，decisions 没有。领导或 reviewer 翻日志会看到两套说法（第 6 节 W15）。

## 1. 读法

- **R 层 / E 层**：R 层（routine）是保持车道、跟车、转弯、按灯停这类靠 continuation prior（沿自车运动和道路结构外推的先验）就能做的部分；
  E 层（emergent reaction）是行驶中突发事件的反应，比如行人横穿要减速。榜单分数大多由 R 层配方决定（第 35 条）。
- **指标**：RFS（Rater Feedback Score，WOD-E2E 的主指标，0–10，人工评分过的轨迹）；PDMS / EPDMS（NAVSIM v1 / v2 的规则化驾驶分，0–100）；
  DS / SR（Bench2Drive 的 Driving Score 和 Success Rate，闭环）；ADE / L2（对 log 轨迹的位移误差，开环）。
- **我们的考卷**：P5 v1 BA 是 CARLA 里 101 条路线 × 3 seed 的 x⁺ / x⁻ 配对（同一世界，一版有 hazard 一版藏掉），expert 是 BehaviorAgent；
  「翻转率」是考生的输出在 x⁺ 与 x⁻ 之间朝对的方向变化超过噪声门槛 τ 的比例；null false-flip 是只换天气时的误翻率，地板约 5%。
  P6 是同构的行为模式考卷（绕行、让行）；I3 是 HUGSIM 3DGS 真实外观的车辆配对。
- **证据强度**（我按下面的规则给，decisions 里没有这个字段）：
  **A** = 全量或大 n，CI 不跨零，且有独立复现、公开数字对得上，或确定性方法；
  **B** = 多 seed 或大 n 二者有一，但只在单一数据集 / 单一环境；
  **C** = 单 seed 且小 n，或只有目检 / smoke，或依赖第三方自报；
  **D** = 已被后续条目推翻或作废，只能当过程记录。
- 「状态」列写 decisions 里的原状态（已定 / 待定），如果我认为它高估或低估了，写在「缺什么」里。

## 2. 盘点

支撑部分的缩写：**榜** = 榜单分数的构成与噪声；**分≠能** = 分数与能力分离的分析（含我们自己的考卷和提取实验）；**适配** = openpilot 可适配性；
**闭环**；**WM** = 世界模型；**—** = 动机、工程事实或已搁置的支线。

### 2.1 分数不是能力（榜 + 分≠能）

| 条 | 一句话 claim | 关键数字 | 状态 · 强度 | 支撑 | 缺什么才能拿出手 |
|:--|:--|:--|:--|:--|:--|
| 35 | 榜单分数里能确证属于 E 层的成分很少，高分主要是 R 层配方和 metric 代理 | NAVSIM v2 只换评分器不搜索 +10.9；B2D 第 2–6 名间距 < 0.7 对噪声 ≈ 1.5；账本里 E 层增益只剩 4 条（RAM、RoG-DAgger、TFv6 LiDAR、RAP 恢复数据）；nuScenes 速度 ×0.5 的伤害是全黑图像的 35 倍 | 待定 · C（公开论文与代码读数，没跑模型；3 处就地降级） | 榜、分≠能 | 登记的三个实验（TFv6 规则 × 接口、SimLingo 目录对照、v1/v2/HUGSIM 同 ckpt 重调权重）都没做；Q5 速度置零（DrivoR −22.1 PDMS、WA-JEPA −8.2）是「GPT 按登记判格，待人复核」，要人复核后才能引用 |
| 38 | B2D 前几名总分差在评测噪声内；突发 hazard 近乎饱和，差距在规划 / 让行类路线 | 单次评测 SD 0.80 DS；两次评测之差 95% 带 ±2.3 DS / ±5.0 SR；BLUE 六次总分只差 0.8 但 58 / 206 条路线成败翻过；相邻名次 \|ΔDS\| < 2.3 | 待定 · B（22 个条目的公开逐路线结果，209 路线；噪声带只借自 BLUE / Orion-Lite / DriveMoE） | 榜 | 我们自己的 TFv6 在 209 条上跑两遍（T2 / T3）没见结果；噪声带套到别的方法是假设 |
| 46 | 没有一个方法族同时在真实开环榜和 CARLA 闭环榜的前 10；交集由训练数据生态和输入契约决定 | 8 类榜 12 张子榜；BridgeDrive 96.34 才是 B2D 第一（上一轮漏了）；跨榜 ρ：B2D↔Longest6 n=5 ρ=0.05，NAVSIM v1↔v2 顶部 n=11 ρ=0.21 | 待定 · C（榜单快照 2026-09-26；n=5 的 ρ 没有统计意义） | 榜 | 数字只能定性引用，不要放 ρ 值 |
| 46 T1–T3 | 榜单前 10 族的新成员在我们的考卷上没有超过族内老成员；表征驱动的 WA-JEPA 是唯一两半都部分成立的 | WOD 零样本对 cv 的 RFS Δ：SparseDriveV2 −0.68、ZTRS −0.40、DrivoR −0.63、WA-JEPA +0.33 [+0.03, +0.63]；I3 车辆翻转 DrivoR 33.7%、WA-JEPA 66.1%（非反应帧误翻 40%）、SparseDriveV2 / ZTRS 23.6% / 45.5%；BLUE − SimLingo 合并 −8.1 pp [−12.4, −4.5] | 待定 · C（单 ckpt 单 seed；NAVSIM 模型只出 4 s 轨迹外推到 5 s；缺后视相机） | 分≠能、榜 | 一张统一的零样本表（见第 5 节 G5）；适配损失对两边是否对称要写清 |
| 40 第 6 点 | 同一份冻结 512 维 Cinque `temporal` 特征，只把选轨从模仿 softmax 换成 Hydra 式 PDM 子分打分头：navtest PDMS 77.9 → 84.2，达到 TransFuser 水平；这是对准 metric，不是 E 层能力 | PDMS 84.2 [83.7, 84.7]，同候选集配对 +6.3 [+5.7, +6.9]；3 seed 84.2 / 83.8 / 84.3（Cinque）、84.4 / 84.2 / 84.2（Lebowski）；主要在 DAC 87 → 93，comfort 80 → 75 | 待定（正文按 3 seed 已去掉「单 seed」限定）· **A-**（受控、3 seed、CI 窄） | 分≠能、榜 | 这是「分数由配方决定」最干净的受控演示，缺的只是一张图；EPDMS 在 seed 1 低 4.6 分（comfort 抖动）要在图注写明 |
| 53 | 榜单最优的 Hydra 打分头在 CARLA 配对考卷上不可比；把 CARLA 激发的 Δ 经真实数据 gate 叠上去，NAVSIM 掉 5–8 分，「能力包」不成立 | top-10 anchor 重叠 0–10%（门槛 30%）；PDMS Δ −5.14 [−5.52, −4.76]（Cinque）/ −8.33（Lebowski）；Q4a 零约束版 PDMS 不掉（−0.23 到 +0.02），但 Lebowski 行人翻转不达门、Cinque 随 seed 变 | 待定 · B | 分≠能 | Q4a 是「GPT 按登记判格，待人复核」；「能力包」如果不进中期主线可以只用兼容检查那一半 |
| 58 | 四个榜单族（TFv6、BridgeDrive、BLUE、SimLingo）都不「位置记忆」，扰动不崩塌；第 46 条 T3 的 0.2% 是开环回放的分布偏移 | ghost 幽灵率 5.0–13.3%，各自门槛 16.7–30.0%；shift / swap 八格无一过 10 pp 崩塌线，BLUE.swap +18.0 pp [+6, +30] 是唯一 CI 不跨零的格 | 待定 · C（shift / swap 单 seed；**没有阳性对照**，ghost test 的灵敏度没有独立验证） | 榜、闭环 | 要有阳性对照才能说「不背题」；否则只能说「没测出背题」。BLUE.swap 补 seed |
| 31 | TFv6 的高分来自 route + target speed 表示，不是 waypoint 跟踪 | A − B +14.3 [+5.1, +25.9]（48 对，路线整组 bootstrap）；控制器主效应 C − B +1.0 [−12.2, +15.4]，未检出（不是等效） | 已确认（表示效应）· B | 榜 | 第 35 条已就地修正：+14 只解释「TFv6 为什么不用自己的 waypoint」，不解释「VLA 为什么比 TFv6 低」，不能拿来讲榜首差距 |
| 34 | Alpamayo 1.5 和 openpilot 在 WOD-E2E val 上零样本都明显超过 ego-only，也超过我们在 Waymo 上训的 head | RFS：Cinque 8.005 [7.79, 8.22]、Lebowski 7.886、Alpamayo medoid-of-6 8.034，cv 7.103，我们最好 head 7.31；logged future 8.131；Cinque − cv +0.90 [+0.62, +1.18] | 待定 · B（val 479 帧全量，bit-exact RFS；Alpamayo 单 seed 42；三处已知适配损失都让分偏低） | 适配、分≠能 | test split 提交（配额 6 次 / 30 天，要用户决定）；Alpamayo 3 seed；「超过我们的 head」这个比较对象太弱，见 W10 |
| 37 | NAVSIM 上 openpilot 原生 plan 的低分不是模型能力，是输入契约 | hold 输入 PDMS 52.1、navhard EPDMS 9.3；见 2.3 适配 | 待定（全量已补，主体 A-） | 适配、榜 | 见 2.3 |
| 39 | nuScenes L2 上两个模型都比匀速直行差，collision 更好；PhysicalAI-AV 上 openpilot 输给 Alpamayo | L2 Δ vs CV：Alpamayo +0.25 [+0.16, +0.35]，openpilot +0.14 到 +0.36；collision Δ 只有两个 CI 不跨零；PAI 31 个 clip Alpamayo ADE 1.77 对 openpilot 2.35–2.86 | 待定 · C（Alpamayo 只跑 1/4，n=1159；PAI 仅 31 clip，可能与 Alpamayo 训练集重叠） | 榜（负面读数） | 只作「L2 奖励数据集速度剖面」的旁证（与 AD-MLP 批评一致）；被问到「openpilot 在 nuScenes 输了」时的回答 |
| 47 | 「判断之后的行为」（绕行、让行、恢复）没有量具；WOD 上人类明确要绕的只有 21 / 479 帧，`cls_late` 0 / 21 | 公开榜单都不按行为模式计分；PDM-Lite obstacle_bypass SR 92% 但靠特权登记 | 待定 · C（调研 + 零成本读数，n=21） | 分≠能 | 仅作动机；`cls_late` 词表里没有绕行 anchor（第 52 条），0 / 21 至少部分是 vocabulary 造成的 |
| 51 | 公开 state-space RL policy 只有 BehaviorBench 两个权重能跑；negotiation 上 PPO 反应是 IDM 的 3 倍；bypass 由 reward 系数决定；recovery 没有一个会 | PPO 让行反应 43.0%（null 2.0%）；conditioned aggressive 档绕 79.5% | 待定 · C（WOMD val interactive，反应式对手那一臂没跑） | 分≠能（第三层） | 不进中期主线；只当「行为层是可调策略不是感知问题」的一个旁证 |
| 52 | P6 v0 考卷成立：PDM-Lite 9 类障碍 100% 绕、删登记后 0% 绕；对向车流造出 negotiation | x₁₀ 绕行 1.00；x₀₀ 0 / 165；negotiation 先等 0.65；放置 null keep 0.889（门 0.90，**按登记不过**）；镜像题 0.80（踩线）；`cls_late` 词表 0 个 bypass anchor | 待定 · B（220 集 55 路线 × 3 seed；单 expert） | 分≠能 | 放置 null 未过门；对向车流是我们自布的，量的是 PDM-Lite gap check 对这张表的反应，不是 B2D 的 2W 难度 |

### 2.2 能力侧：配对考卷与提取（分≠能）

这一组是我们自己的仪器和方法，结论都限定在 CARLA。

| 条 | 一句话 claim | 关键数字 | 状态 · 强度 | 支撑 | 缺什么才能拿出手 |
|:--|:--|:--|:--|:--|:--|
| 32 | P5 v0：CARLA 配对考卷成立；公开 planner 只在 waypoint 通道上过四成，我们的 CARLA 内薄 head 翻转率为 0 | 96% 的对在因素可见后 ego 逐 tick 相同；null 上 Δ_expert p95 0.04 m/s；TFv6 目标速度翻转 2.0%（噪声地板），waypoint 39.4%（按对 73%）；`ridge_late` 0 / 75 | 待定 · B | 分≠能 | 只有 BehaviorAgent 一个 expert；红绿灯只有 9 个 reactive 帧 |
| 42 | 行人信息在 openpilot vision 层就没有（**只在 CARLA**）；双流 reaction head 用配对差分让行人翻转到 43%，hard-example 重加权不能 | vision 层行人 AUC 0.515 [0.503, 0.535]；M-C 双流行人 43.3% [35.0, 50.7]（Cinque）/ 41.6%，3 seed 43.3 / 43.6 / 42.4；hard-example 0–3%；null 5.1%；20 Hz student 51–57%，端到端 p95 ≈ 30 ms；代价：DynamicObjectCrossing 非反应帧误翻 0.3% → 11.7% | 待定 · **B+**（101 路线 × 3 seed，两模型；但只有 BehaviorAgent 集，PDM-Lite 集所有考生贴地板，见 W9） | 分≠能 | 第 55 条已限定：真实 nuScenes 上原 `temporal` 读走廊行人 AUC 0.71，所以「vision 层没有」不能推广；真实数据上没有对应读数 |
| 43 | 融合前诊断：Qwen ⊕ openpilot 在均匀 imitation 下冗余、在配对差分下互补；SAM 规则门被学出来的 head 超过 | 均匀 imitation 拼接行人翻转 0.2–1.2%；配对差分 43%；SAM 状态规则门 29.2% [13.2, 47.0] | 待定 · C（2 处就地翻案，原结论已被推翻） | — | 不要引用原标题；它是「预登记问错了问题」的教学例子 |
| 45 | 快通道感知：YOLO26x-seg 640 以 SAM 3.1 的 1/25 延迟拿到不劣的行人召回；缺口在 BEV 放置，单目 metric depth 可修 | 三路 p95 20–30 ms 对 SAM 550 ms；召回配对差 +0.005；nuScenes 20–40 m BEV 召回 0.22 → 0.57（UniDepth v2） | 待定 · B（P5 v0 + nuScenes 1/3 scene；SAM 规则门没在 YOLO 状态上复跑） | 分≠能（快通道） | 工程性结果，非主线 |
| 48 | 冻结的视频 / 图像自监督特征里有行人反应信号，只有配对差分激发得出；V-JEPA 2 与 Qwen 同一水平，SigLIP2 低一档，DINOv2 与 openpilot small 没有 | pair-Δ 行人翻转 V-JEPA 2 47.0% [45.8, 47.8]、Qwen 41.5%、SigLIP2 33.8%、DINOv2 7.1%、openpilot small 3.0%；3 seed；单帧 vs 4 帧：0.35–0.40 对 0.46–0.48 | 待定 · B（CARLA BA 集、pooled 特征、线性 head） | 分≠能 | **V-JEPA 2 没有显示出比 Qwen 更好**（W11）；Q6 单帧对照是「待人复核」读数 |
| 49 | openpilot 冻结特征里线性可读「前方静止障碍」「邻道有车」「对向来车」；desire 脉冲方向总对但幅度不够当绕行执行器 | probe a 0.972–0.992（P5 v1）、N1 重跑 0.884–0.911，速度一维 0.50；锥桶 0.95–0.99；desire 5–10 m/s 横移中位 0.61–0.68 m（要 ≥ 1.5 m） | 待定 · B | 分≠能、适配 | 只在 CARLA；desire 执行器路线不成立 |
| 50 | 快通道检测输入不需要平地抬升和走廊筛，image-plane token 同一水平 | 行人翻转 57.4 / 48.5 对 52.7 / 55.7，CI 重叠；cut-in 上走廊几何有 +6.5 pp（Lebowski） | 待定 · B | — | 工程细节 |
| 44 | CARLA 激发的 reaction head 直接加到真实特征上有害；log 孪生对、真实帧抹人对、HUGSIM 3DGS 重训三条路都没带过去 | E1 WOD RFS Δ −1.02 [−1.21, −0.82]（Cinque）/ −1.52；NAVSIM PDMS −8.2 [−8.9, −7.5]；20 Hz student 12 / 12 组「有害」但小一个量级（NAVSIM −1.3 到 −2.7）；G1 gate 只是整体缩小 Δ；E3 log 里挖不出场景解释的分叉；E2 编辑对 navtrain 训 WOD RFS −2.16 | 待定 · B（多个受控负结果；E1 只在 19 663 帧子集） | 分≠能 | 这是**诚实的负结果**，也是验收原则未满足的直接证据；其中 P3 3DGS 插入 / 删除路线已被用户目检否决 |
| 25 | 方法假设：jev 是反应通道（continuation prior + 由配对差分监督的 reaction decoder） | 设计假设；(e) gated head 实测 gate 学得会「何时」、学不会「怎么」 | 待定（设计假设）· — | 分≠能（方法） | 结构在 CARLA 开环有正结果（第 42 条），真实数据没有；现在的主方法已经不是它，见 W15 |

### 2.3 openpilot 适配（适配）

| 条 | 一句话 claim | 关键数字 | 状态 · 强度 | 支撑 | 缺什么才能拿出手 |
|:--|:--|:--|:--|:--|:--|
| 36 | openpilot 换相机 rig 基本不掉，掉的是朝向标定和时间轴 | comma1M 8 段 8800 帧；60–120° pinhole、鱼眼、Waymo 前三路拼接等都在 +8% 内；yaw 标定错 2° 横向 ×4.6；NAVSIM 式 2 Hz 时间轴横向 ×9、纵向 ×25 | 待定 · B-（8 段） | 适配 | 4–8 段 small / Lebowski；渲染 3D 场景复测（`shadow-h*`）没出 |
| 37 + integration | NAVSIM 上 openpilot 原生 plan 的低分几乎全是输入契约；补帧后与 TransFuser 同量级 | navtest 全量 n=12 146：hold 52.1 → **GIMM-VFI 84.2** [83.7, 84.6] / ego-motion warp 82.0；navhard two-stage EPDMS 9.3 → 33.3 / 27.7；直行 87.9、左 / 右转 77.5 / 73.6；WOD 上 GIMM 收回帧率差 89%；2 000 子集与全量相差 0.5–1.1 | 待定（原「30–40 分」标题已被推翻，正文有 4 层就地修正）· **A-**（全量、devkit 复现 log / cv 分数） | 适配、榜 | 见 W2；GIMM-VFI 是 S-Lab 非商用许可；navhard 的 EPDMS 聚合口径第一次算错过；navhard 33.3 与文献 TransFuser 23.1 / DiffusionDrive 27.5 **看起来更高，但 devkit 版本不同（我们 human 94.5，文献常引 90.3），不能写超过** |
| 40 | 冻结特征加薄 head，openpilot `temporal` 比所有通用 backbone 好一个量级；Alpamayo 1.5 中层 VLM 特征不是 | pre-onset ΔADE（train 训、完整 val 评）Cinque −0.294 [−0.424, −0.168]、Lebowski −0.318；对 arm A / 原生视频 Qwen RFS +0.26 到 +0.31；nuScenes 独立复现 −0.091 / −0.087 m（对 A −0.076 / −0.072），相对量 ADE −18% 对 WOD −22%；NAVSIM `cls_late` 77.9 / 77.3，对 `cls ego` +9.6 / +8.8 EPDMS；碰撞率 0.81% → 0.39% | **已定**（openpilot 部分）· **A-**（两个数据集 + 第三个 NAVSIM 复现；`ridge_late` 确定性；`cls_late` 3 seed 极差 0.18–0.26 RFS） | 适配 | 见 W1（native plan 反而更好）；Alpamayo `L27_last` 要独立复现；没有测其他驾驶预训练 backbone（WA-JEPA、DrivoR encoder） |
| 55 | openpilot vision 层可以便宜地改而不坏；但只用真实数据监督行人可读性 +0.08，没过线，完全带不到 CARLA | PyTorch port 对 fp32 图 plan 位置 max 2e-4 m；stage 4 解冻 + 蒸馏 0.07 GPU·h / 10 万样本；漂移中位 0.058 / 0.061 m，WOD ADE +0.36% [−0.28, +1.03]；nuScenes 走廊行人 AUC 0.709 → 0.786（Δ +0.076 [+0.004, +0.166]，线 +0.10）；3 seed Δ +0.076 / +0.094 / +0.073；CARLA D0 +0.009 | 待定 · B（3 seed；正例只 203 个；只测线性可读性和开环漂移，没测行为、闭环；Lebowski 没参照） | 适配 | 这是 D2–D5 op-adapt B 第二轮的基线；「不坏」证据强，「变好」证据没有；见 G1 |
| 56 | Cosmos-Transfer2.5 把 CARLA 配对重画成真实感：独立翻译两边会让 x⁺ / x⁻ 在行人外也不同（v1 no-go）；v2「x⁻ 翻一次 + 行人区锚定」后区外逐像素相同 | v1：openpilot `temporal` 差 80% 落在行人区外（CARLA 原图 7%）；v2 G4 10 对：召回 0.982（CARLA 0.976），lead 一致 95.5%，2 000 对约 118 GPU·h；逐对判据只过 1 / 10，用户事后作废「环 MAD ≤ ¼ 换 seed」一条（记为偏离），G4 go | 待定 · C（10 对、1 seed、单相机、BehaviorAgent 集） | 适配 | 全量 2 000 对在跑；「只差行人」在数字上只在锚定区外「按构造」成立；车内痕迹（仪表台）1 / 10 明显、接受 |
| 59 | 真实片段上用同一 ControlNet 生成配对两边：删人不干净，插入偏假但可接受；整条线暂停 | 19 场景 93 帧；区外像素差 1.5–2.5（0–255）；无自动检查 | **已确认为暂停**（不是 no-go）· C（只有用户目检） | 适配（数据） | 不要作为成果；只作为「真实外观配对暂时没有干净来源」的事实 |
| 26–31、41 | 控制器线：固定轨迹控制器、TCP / TFv6 上的执行层对照 | 无合格替代默认；C / D 在 L1 又准又平顺，但传不到闭环；TFv6 DS：A 94.1、B 95.0、C 86.5、D 86.0、P2 87.8；L3 相对原生 C −7.6 [−16.8, +1.2] | 多数已确认（机制）、结论为「没有控制器被判更好」· B | 闭环（基础设施） | 消耗了大量工作量，与主线无关；只用于回答「执行层为什么不动」 |

### 2.4 闭环与世界模型（闭环、WM）

| 条 | 一句话 claim | 关键数字 | 状态 · 强度 | 支撑 | 缺什么才能拿出手 |
|:--|:--|:--|:--|:--|:--|
| 16、17 | CARLA 能在无头机器上跑；Bench2Drive 220 条全量实测 3.11 h（不是天级） | 4 个实例是旧卡的最优并发（86.5 FPS），新 box 每卡约 6 个；旧全量 3.11 h | 已确认（旧卡口径；新 box 容量扫描 `knee-6000d` 被污染要重跑）· A | 闭环（基础设施） | 工程事实，不是科学结果 |
| 33 | Alpamayo 1.5 能在 B2D 闭环开；openpilot 的 smoke 分数无效 | Alpamayo n=5 DS 60.8、RC 70.1、SR 2 / 5；openpilot DS 2.7 作废（适配 bug：plan 原点差 1.78 m，静止时等于 7.1 m/s 指令） | 待定 · C（n=5 单 seed，全量 220 停在 17 / 220） | 闭环 | 没有任何模型的正式闭环分数；见 G3 |
| 57 | openpilot 进 B2D 闭环：起不了步是静止先验，不转弯是 turn desire 在路口前不改 plan；无感知路线 base 就有 DS 57；openpilot 只做纵向 modifier 对 base +10 [−6, +27]，让它横向驾驶 −30 到 −48 | native 6 / 6 从不动；base 56.6；e2e 66.2；switch 26.1；oplat 8.5；**同均速对照 base 67.7 > e2e 66.2**，「+9.7 可以用开得慢解释」；无 20 s 兜底 e2e 降到 36.4 | 待定（**设计搁置**，用户 2026-09-28 决定）· C（10 条 dev、单 seed，同配置重复在一条路线上差 47 DS） | 闭环 | 没有可汇报的正面结果；这是中期最大的空洞，见 G3 |
| 54 | W：冻结 latent（openpilot `temporal` ⊕ V-JEPA 2 `mean`）上的 action-conditioned 世界模型失败，原因是训练日志里动作—场景因果混淆，不是 latent 装不下 hazard；改用 CARLA 动作分叉数据重做 | 原判：配对 AUC 只有障碍过（0.712），行人 0.541、cut-in 0.694；动作敏感刹停 > 保持只有 6.5%；诊断：latent MSE 比 persistence 低 60%，打乱动作后误差 ×2.6–2.9；专家日志里「接下来刹车」的窗口 2 s 后前车比匀速窗口近 6.9–10.4 m（运动学应远 5.3–5.9 m） | 待定 · C（seed 0 诊断；解释是事后的） | WM | WL 干预数据的 C1–C4 出数（C1 判据 ≥ 85%）；中期只能讲「第一次失败 + 有检验条件的诊断」，见 G6 |

### 2.5 动机、工程与已搁置的支线（—）

这些条目状态多为「已确认」，但都不在现在的故事线上。可以在中期用一页讲「我们怎么走到这里」，不要占主线。

| 条 | 内容 | 状态 · 强度 | 用处 |
|:--|:--|:--|:--|
| 1、3、3b–3d、20、22–24 | Waymo 上的 pre-diagnostic：视觉增量并不集中在 ego prior 最弱处，反而最小；线性读出 Qwen3-VL 特征在 pre-onset 上榨干；换表征阶梯里只有 Qwen 原生视频有信号 | **已确认 / 已证伪** · A（train 训、完整 val 评：DiD +0.111 [+0.059, +0.162]；pre-onset −0.016 [−0.062, +0.030]，n=1 510；(d‴) V-JEPA 的 −0.030 CI 跨零「测不动」） | 一页动机：通用 VLM 冻结特征在关键时刻没用。但它测的是 Qwen，不是现在的主角，容易被误读成「所有冻结特征都没用」 |
| 2、4–15 | RFS 口径、分辨率、K、late fusion、延迟、backbone 门禁、帧间隔等 | 已确认 / 待定混合 · A（实现事实） | 工程背景；第 2 条的 RFS bit-exact 复现是后面所有 WOD 数字的地基 |
| 18、19、21 | 文献判断：低延迟不再是卖点；real→sim 迁移已有人做，缺 HUGSIM 一列；论文是方法论文 | 已确认（文献事实，带覆盖度限定） | 第 21 条的「交付物是闭环 driver」已不是现在的说法，见 W15 |
| 26–31、41 | 控制器线 | 见 2.3 | — |

## 3. 最强的 7 个「拿得出手」结果与故事顺序

排序标准：受控程度、样本、能不能一句话讲清楚、能不能经得起追问，以及是否直接支撑「trick or trade」。表中排名按强度，故事顺序另列。

| 强度排名 | 结果 | 一句话 | 为什么强 | 必须同时说的限定 |
|:--:|:--|:--|:--|:--|
| 1 | 输入契约（第 37 条 + integration 第 9 节） | 同一个 openpilot，只改喂法：NAVSIM navtest PDMS 52.1 → 84.2，与 TransFuser 84.0 同量级 | 全量 12 146 token；devkit 复现 log / cv 分数；零学习的 ego-motion warp 就有 82.0；「trick」与「trade」之间的差别被一个数字说完 | 没有 route，转弯 PDMS 74–78 对直行 88；navhard 只有 33.3；GIMM-VFI 非商用许可；不能写超过文献 navhard |
| 2 | 冻结 `temporal` 是最好的 backbone（第 40 条） | 驾驶视频训出来的 512 维特征加线性 head，比 Qwen / V-JEPA / SigLIP 一类通用 backbone 强一个量级，两个数据集都成立 | 三个数据集独立复现（WOD、nuScenes、NAVSIM）；相对量一致（ADE −18% / −22%）；碰撞率减半 | 只与已测的通用 backbone 比；冻结特征 + head 的 RFS（7.45–7.7）低于 openpilot 原生 plan（8.00）（W1） |
| 3 | 榜单打分头的受控演示（第 40 条第 6 点 + 第 53 条） | 同一份冻结特征，只换选轨目标，PDMS +6.3，DAC 上升、comfort 下降，行为不变 | 3 seed、2 模型、配对 CI 窄；直接对应「分数由配方决定」 | 只在 NAVSIM；结论是「对准 metric」，不能推广成「所有榜单都这样」 |
| 4 | 零样本对比：没见过 Waymo 的驾驶模型 vs 榜单模型（第 34 条 + 第 46 条 T1/T2） | WOD RFS：openpilot Cinque 8.005 与 Alpamayo 7.86–8.03 高于 cv 7.10；SparseDriveV2、DrivoR 在 NAVSIM navtest 上复现 92.2 / 93.7 PDMS、ZTRS 在 navhard 48.1 EPDMS，却在 WOD 上低于 cv（由 Δ 推算约 6.4–6.7）；WA-JEPA 是唯一高于 cv 的榜单族（推算约 7.43） | 直接是「trick or trade」的一张图；479 帧全量、bit-exact RFS | **推算值**，需在同一批帧上统一重算配对 CI；榜单模型有适配折中（4 s 外推 5 s）；单 ckpt 单 seed；Alpamayo 单 seed |
| 5 | 榜单噪声与构成审计（第 38 条 + 第 35 条 + 第 58 条） | B2D 前几名总分差在噪声内（两次评测 ±2.3 DS）；开环 / 闭环 / 代理榜的增益构成不同；四个榜单族没有位置记忆 | 22 个条目的逐路线数据；BLUE 六次重复量出噪声；ghost test 是我们独有的闭环仪器 | 噪声带只借自 BLUE；ghost test 没有阳性对照；n=5 相关不能引 |
| 6 | openpilot 可以便宜地改而不坏（第 55 条） | PyTorch port 与 fp32 图差 2e-4 m；stage 4 解冻 + 蒸馏 0.07 GPU·h / 10 万样本，正常帧 plan 只漂 6 cm，WOD ADE 不变 | 3 seed；成本、漂移都是硬数 | 「变好」没过线（+0.08 < +0.10，CARLA +0.009）；没测行为 |
| 7 | 配对考卷与提取（第 32、42、48、50、52 条） | 在 CARLA 里配对差分让冻结特征的行人翻转从 0 到 43–57%，hard-example 重加权不行；考卷有 null 地板、有 expert 标签 | P5 v1 101 路线 × 3 seed；null 5%；已推广到三类预训练特征、20 Hz 快通道（30 ms） | 只在 CARLA；只有 BehaviorAgent 集有效；PDM-Lite 集所有考生贴地板；真实数据上有害（第 44 条） |

### 建议的故事顺序

1. **榜单分数不是能力**（排名 5、3）：B2D 噪声、增益构成、打分头 +6.3 的受控演示。开场，让听众先接受「分数 ≠ 能力」。
2. **trick vs trade**（排名 4）：一张零样本表：没见过 Waymo 的 openpilot / Alpamayo 高于 cv，榜单前 10 的模型在同一批帧上不如 cv。这是主论点。
3. **trade 的代价来自契约，不是模型**（排名 1）：NAVSIM 52.1 → 84.2。回答「为什么 openpilot 在榜上看起来不行」。
4. **openpilot 为什么值得作主角**（排名 2）：冻结特征最好，跨数据集成立；配上 R40 的诚实读数（native plan 更好）。
5. **能改、便宜、不坏**（排名 6）：适配成本硬数，加上「变好」还没过线和 D2–D5 在做什么。
6. **能力怎么量、怎么提**（排名 7）：配对考卷与配对差分，CARLA 里的正结果。
7. **诚实的限制与下一步**：真实数据上有害（第 44 条）、世界模型第一次失败与诊断（第 54 条）、闭环接入进度（第 57 条）。放在最后，明确写「这是下一阶段要经费做的事」。

这个顺序把最强的结果放在前半，把没有过验收线的东西放在最后当作「下阶段要解决的问题」，而不是当结论。
**领导问「所以能不能在虚拟和真实里都会开」时，诚实回答是：目前没有一个结果同时满足，这是下一阶段的核心。**

## 4. 每个故事段落各缺什么（汇总）

| 段落 | 已有 | 缺 | 成本 |
|:--|:--|:--|:--|
| 榜 | 噪声、构成、ghost test | 我们自己的 TFv6 两遍；ghost test 阳性对照；BLUE.swap 补 seed | GPU 闭环，中 |
| 分≠能 | 打分头受控演示、配对考卷 | 统一零样本表（G5）；Q5 速度置零人复核 | Mac CPU，小 |
| 适配 | 契约 + 特征 + 不坏 | 适配后的 WOD / NAVSIM 分数（G1、G2、G4） | GPU，D2–D5 |
| 闭环 | 基础设施、仲裁 pilot、ghost test | 任何一个 openpilot 的正式闭环分数；HUGSIM 考试（G3） | GPU，D5–D8，风险最高 |
| WM | 第一次失败 + 诊断 | WL C1–C4（G6） | 几 GPU·h，2 卡阶段插空跑 |

## 5. 缺口：故事需要而现在没有的结果，对应 D2–D10

| # | 缺口 | 现状 | 对应计划 | 「够用」的样子 | 风险 |
|:--|:--|:--|:--|:--|:--|
| G1 | **适配后的 openpilot**：sim + real 混训的 vision 层，行人可读性在 real 和 CARLA 上都涨，且正常帧不漂 | 第 55 条：只用真实数据 +0.08，CARLA +0.009；已有 B（stage 4 解冻 + 蒸馏）的配置，0.07 GPU·h / 10 万样本，瓶颈是监督信号，不是算力 | D2–D5 卡 A：op-adapt B 第二轮（Cosmos 配对 + WOD / navtrain，stage 4 + hazard loss + 蒸馏）。前置：Cosmos 2 000 对（约 118 GPU·h，外推墙钟 ≤ 36 h 才放行）和 CARLA 重渲染 | P5 D0 AUC ≥ 0.60 且 nuScenes / WOD 走廊行人 Δ ≥ +0.10、正常帧漂移仍 ≤ 0.10 m；**最好加行为读数**（plan 对行人的停车率） | ① 判据只测线性可读性，第 55 条自己说 stage 3 输出已有 0.83、B 只到 0.79，上限空间小；② 全量 Cosmos 的「只差行人」只在锚定区外按构造成立，车内痕迹和亮度差（4 / 303 对 > 20）可能仍是捷径；③ 考题场景必须与训练对严格分开（Town13、小地图、B2D 行人路线），否则「把 P5 当 held-out」无效；④ 第 55 条和第 42 条已指出 CARLA 行人读不出可能是渲染域差，Cosmos 是否补上要用 D0 AUC 直接测 |
| G2 | **NAVSIM v1 / v2 加导航**后的 openpilot | navtest 84.2（v1，无 route）；navhard 33.3（v2）；左 / 右转 PDMS 74–78 对直行 88 | D2–D5 卡 A：补左右导航后测 NAVSIM v1 / v2 | 转弯 PDMS 追近直行；navhard 与 navtest 之间不再差 50 分 | **导航通道在我们三次测量里都没用**：Alpamayo nav 文本无效（第 34、37、39 条：−2.0 EPDMS、+0.002 m）；openpilot turn desire 拖分（第 37 条 −1.0 到 −3.4；第 40 条 WOD routing desire RFS 不变好、Cinque −0.13）。所以「补导航」不该是 desire 脉冲，要是 ego 侧的 route-conditioned 修正头；这一步的设计需要先登记 |
| G3 | **闭环**：适配后 openpilot 的 B2D 子集与 HUGSIM | 没有任何 openpilot 正式闭环分数；起不了步（缺 resume）、不转弯（desire 不改 plan）、纵向 modifier 的增益被同均速对照解释掉；HUGSIM 只有预登记（4 Hz 帧率代价、`h4-dilate` 选择规则），rate study 排队中 | D2–D5 卡 B：闭环接入修复；D5–D8：B2D 子集 + HUGSIM | 至少：一个不靠「开得慢」的对 base 正配对差（同均速对照下 CI 下界 > 0），加一个 HUGSIM HD-Score 与官方基线的对比 | **这是最大的排期风险**：用户 2026-09-28 因五种仲裁都不满意已搁置集成设计；3 天修复 + 3 天出分，中间没有设计迭代余量。建议 D5 设 go / no-go：过不了就把闭环这一段改成「基础设施 + ghost test + 诊断（起不了步、不转弯的机制）」，不要拿 e2e +9.7 充数 |
| G4 | **WOD-E2E**：适配后 openpilot 的分数，以及能对外说的榜单数字 | val 479 帧：Cinque 8.005；没有 test split 提交；WOD 榜首 8.167（test）；val 与 test 不能配对 | D5–D8 开环榜 | val 上适配前后配对差；如果要提交，需用户批 test 配额（6 次 / 30 天）；Alpamayo 3 seed | 提交前要确认适配损失（视差、底部黑带、相机高度）已量化；否则「openpilot 8.0」会被当成下限还是上限说不清 |
| G5 | **统一的零样本对比表**（trick vs trade 的主图） | 数字散在第 34、46 条，T1 / T2 的 WOD RFS 是「对 cv 的 Δ」，不是同一批帧上的并列表 | 不需要新 GPU：用已存的逐帧预测在 Mac 上并列重算（D0–D1.5 的「Mac 上盘点」窗口） | 一张表：同一 479 帧，同一 bit-exact RFS，每行配对 Δ vs cv 的 CI，标出适配折中 | 如果重算后 WA-JEPA / DrivoR 与 cv 的关系变了，故事要跟着改；先算再讲 |
| G6 | **世界模型**：WL 的 C1–C4 | W 第一次失败；WL 特征在 GPU 5 上出 z，尚未训练、无数字 | 计划：只做 C1–C4 前置检查（几 GPU·h，2 卡阶段插空跑）；想象训练放到中期之后 | C1（动作效应方向）≥ 85%（W 是 6.5%），并在分叉点上 W 原版作阳性对照 | 中期只能汇报到「前置检查」，而主方法是想象训练；要在汇报里明说主方法的核心一步还没开始。若 C1 也不过，诊断的解释（数据混淆）被推翻，主线要重新论证 |
| G7 | **真实外观的行人 E 层考卷** | P3 3DGS 插入 / 删除路线被用户否决；ControlNet 配对暂停；I3 只有车辆、规则标签；CARLA 版行人剂量 - 反应考卷（dose-response）用户同意方向，但在 GPU 6 空出后才派人搭 | 不在 D2–D10 明细里 | 至少 CARLA 原图与 Cosmos G4 两种画面各考一遍同一批行人格 | 没有它，「真实数据上会不会对行人反应」这个问题在中期只能靠 nuScenes 的线性可读性（AUC 0.71–0.83）回答，回答不了行为 |
| G8 | **验收原则的一张综合图**：同一个模型在 sim 和 real 上都比原版好 | 不存在 | D8–D10 出图 | G1 + G2 + G4 + G3 各取一格并排 | 如果 G1 不过线，这张图就是「real 上没坏，sim 上没好」；要预先决定怎么讲 |

## 6. 经不起追问的条目

按被问到时的危险程度排。「怎么答」是现在能诚实给出的回答。

| 编号 | 条目 | 追问 | 问题 | 怎么答 / 怎么补 |
|:--|:--|:--|:--|:--|
| W1 | **第 40 条 vs 第 37 条 integration** | 「为什么不直接用 openpilot 的原生 plan，还要薄 head？」 | 修好输入之后，原生 plan 比冻结特征 + head 更好：NAVSIM 84.2 对同批 `temporal` + `cls_late` 77.5（子集上高 7.2 [5.6, 8.8]）；WOD 8.00 对冻结 head 7.45–7.7，`cls_late` 三个 seed 都没补上 Cinque 的原生（G −0.06 到 0.70）。第 40 条「已定」的是「特征比通用 backbone 好」，这在「用来当反应通道的输入」上成立，在「当规划 backbone」上被自己的后续结果削弱 | 明说：`temporal` 的价值是当**读出与适配的接口**（可读行人、车、障碍，可蒸馏），不是替代原生 plan；原生 plan 修好契约后就够强，所以适配的目标是 hazard 而非通用规划 |
| W2 | 第 37 条标题 | 领导读日志 | 标题仍写「只比 CV 高 15–20 分，离 specialist 差 30–40」，正文已就地改了 3 次（2 000 子集 → 全量，84.2 / 33.3），结论翻了 | 汇报前把标题改成当前结论，把旧标题降到「原文」；同样处理第 42、43、44、46 条的长标题 |
| W3 | 第 57 条 e2e「+9.7」 | 「openpilot 在闭环里有增益吗？」 | 同均速对照（base 按 e2e 的平均车速慢开）DS 67.7 ≥ e2e 66.2；CI [−5.5, +27.0] 跨零；e2e 42 次 plan 引起的停车里 30 次真值无障碍，24 / 41 次放行靠 20 s 兜底；同配置重复在一条路线上差 47 DS | 不引用 +9.7。引用「起不了步是静止先验、不转弯是 desire 不改 plan」这两个机制结论，以及「无感知 base 就 57」这个说明 B2D 分数含义的读数 |
| W4 | 第 33 条 | 「Alpamayo 闭环 DS 60.8」 | n=5，单 seed，全量停在 17 / 220；openpilot 的 2.7 已作废 | 不作为结果；只作为「zero-shot 闭环适配 bug 多」的教训 |
| W5 | 第 54 条 | 「世界模型是主方法，第一次实验失败了？」 | W 的 3 个判据里 2 个不过，是事后诊断（seed 0，「专家日志里刹车窗口 2 s 后前车更近」）救回来的；诊断本身合理但没被检验，WL 还没出数；同时 V-JEPA 块的行人 probe（0.77 / 0.82）低于 Qwen（0.87）和 YOLO image-plane token（0.82） | 汇报为「原假设被证伪、诊断给出可检验的预测（C1 ≥ 85%）、检验在跑」，不要写「失败不是 latent 的问题」 |
| W6 | 第 44、55 条与验收原则 | 「你们的验收原则满足了吗？」 | 没有一个结果同时在 sim 和 real 有效；本轮四条通往真实数据的路（零样本、log 孪生对、编辑对、3DGS 重训）都没通；op-adapt B 只用真实数据带不到 CARLA | 当作贡献而不是失败讲：这是把「sim 训的东西能不能带到 real」量出来的第一批受控负结果；下一阶段用 sim + real 同时在场的监督 |
| W7 | 第 58 条 | 「你们证明了榜单模型不背题？」 | 没有阳性对照；ghost 率 5–13% 是「没测出」不是「测出没有」；shift / swap 单 seed；BLUE.swap +18 pp [+6, +30] 未过 10 pp 线但是唯一 CI 不跨零 | 说「四个族在我们的两类扰动上没有测出位置记忆」，不说「不背题」；补 BLUE.swap seed 和一个阳性对照 |
| W8 | 第 35、46 条 | 「跨榜相关性多少？」 | n=5 的 ρ 无意义；「E 层增益只有四条」是从论文表读出来的；RAM 的 Give_Way +26.7 超过上限 50% 后已降级；很多「多榜前 10」的「前 10」含匿名队 | 不放 ρ 数值，只放定性结论和我们自己量的噪声（第 38 条） |
| W9 | 第 42 条的考卷有效性 | 「行人反应考卷靠什么标签？」 | 只有 BehaviorAgent 集有效；PDM-Lite 集所有考生贴地板（连特权 GT 几何规则门也只有 4–11%），因为 PDM-Lite 靠特权状态在可见前就反应；本条修正过一次（v0 的「0」被证明是数据量，不是表征）。且第 55 条指出 openpilot 在真实 nuScenes 上读得出近处行人（AUC 0.83），CARLA 上读不出（0.51）更像渲染域差，那么 P5 行人题测的可能是渲染域差，而不是 openpilot 的能力缺口 | 汇报时把 P5 行人题限定为「CARLA 渲染下的反应考卷」；Cosmos G4 的价值正是把这一点测出来 |
| W10 | 第 34 条「超过我们的 head」 | 「比你们自己的 baseline 高有什么意义？」 | 我们的 head 是 Qwen 冻结特征 + ridge，本来就是弱基线（RFS 7.31 ≈ cv 7.10 + 0.2）；有意义的对比是对公开榜首（8.043 test / 8.167 ZSD-Titan test）；val 与 test 不能配对 | 用 cv、logged future、rater_best 作参照，不用我们自己的 head 作主要对比对象 |
| W11 | 「V-JEPA」在主线里的证据 | 「V-JEPA 有什么用？」 | 主线写「openpilot + JEPA」，但我们量到的 V-JEPA 2：pre-onset (d‴) 测不动；配对差分行人翻转与 Qwen 同水平（47% 对 41.5%，CI 重叠，「不写更好」）；行人 probe 比 Qwen 低。没有任何结果显示 V-JEPA 比通用模型更好。V-JEPA 2 也未必是「工业数据」 | 现在支持「工业数据训练的模型更好」这句话的只有 openpilot 一个例子；JEPA 的论据要靠 WL 和 WA-JEPA（榜单族里唯一两半成立）；措辞上不要说 openpilot + V-JEPA 两者都被验证 |
| W12 | 第 56 条 | 「Cosmos 配对可靠吗？」 | 逐对登记判据只过 1 / 10，用户事后作废其中一条并按偏离记；10 对、1 seed、单相机；「只差行人」只在锚定区外「按构造」成立；G4 是训练数据生成器，决定者是用户，不是证据 | 汇报时写「用户接受的数据源，质量检查在全量的 stage 10 checklist」 |
| W13 | 第 41、26–31 条 | 「这一大块和主线什么关系？」 | 大量工作量、结论是「没有控制器被判更好」；L3 相对原生 C −7.6 [−16.8, +1.2] | 一页带过；把它作为「闭环接口比控制器更重要（+14）」的支持，并注意第 35 条对「接口」归因的修正 |
| W14 | 第 39 条 | 「openpilot 在 nuScenes 输给匀速直行？」 | 是。L2 Δ vs CV +0.14 到 +0.36（CI > 0） | 说 nuScenes L2 奖励速度剖面（AD-MLP 批评），并附 collision 更低；第 40 条里同一批冻结特征 head 在 nuScenes 上 L2 降 18%、碰撞减半 |
| W15 | 叙事缺口 | 领导翻 decisions | 第 21 条（AD 方法论文，闭环 driver，主表 B2D + HUGSIM + NAVSIM）之后没有一条记「trick or trade」；`frozen-vlm-planner.md` 在 README 里写「不再是主线」，但第 1、25 条仍在按旧框架 | 加一条 decisions（约第 60 条），记录主线变更的日期、依据（第 40、46、55 条）、被取代的说法 |
| W16 | 编号 | 内部整洁 | 第 8、9、10 条各出现两次（第 8 条与第 9 条「K 取多少」是同一结论的两个版本，都写「待定」，其中一个已用真 RFS 复现）；第 22 条排在第 25 条之后 | 汇报材料不用条号引用旧的 8–10；有空时合并 |

## 7. 中期前建议先做的小事（不占 GPU）

1. 在 Mac 上用已存逐帧预测做 G5 的统一零样本表（先算，再决定故事第 2 段怎么讲）。
2. 改写第 37 条标题，补一条 decisions 记录主线变更（W2、W15）。
3. 把第 40 条里 native plan 与冻结 head 并列在一张表里（W1），领导问到时能直接回答。
4. 提前写好 D5 的闭环 go / no-go 判据（G3），并在 D2 就写进 `todos/`，避免看到数字再挑说法。
5. 对 op-adapt B 第二轮，登记里加一个行为读数（plan 对行人的停车率），不只测线性可读性（G1）。
6. 「待人复核」的 GPT 按登记判格读数（Q5 速度置零、Q6 单帧、Q4a、K 能力读数）要先有人复核，否则不能引用。
