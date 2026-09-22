# 反事实配对、生成式编辑与视频生成表征：2026-09 的现状

状态: 文献现状，2026-09-22 汇总。只写「别人做到了哪里」，不写我们的计划。
来源: 四份 round-4 deep-research 原始报告（`research/lit/2026-09-22-round4-*.md`，不进 git）：
intervention pairs、generative editing、video-gen as encoder、H3 deployment。
背景: [decisions.md](decisions.md) 第 18–20 条，[trajectory-to-control.md](trajectory-to-control.md)。

**覆盖度限定**（引用本文任何「没人做过」时必须带上）：四个 agent 的 WebSearch 配额都在中途或开工前耗尽，
arXiv API 大部分时间返回 429，因此是关键词驱动的定向检索，不是按日期穷举；workshop 论文、
未上 arXiv 的 camera-ready、Reddit 一类社区实测帖基本没扫到。「没人做过」一律读作「按这些关键词没找到」。
原始报告里每条都标了 VERIFIED（agent 读了源页面）或 REPORTED（二手），本文只在关键处重标。

## 一句话

三件事在 2026 年夏天同时挤了进来：**反事实配对诊断**（counterfactual pair，同一帧改一处、看 planner 输出变不变）
四周内出了三篇，其中一篇四天前；**视频生成模型当 planner 表征**已经有并排数字，video latent 赢了 VLM hidden state；
**MiniMax H3** 开了权重，但它的理解侧就是 Qwen3-VL-32B。还空着的格子集中在三处：
标签由 simulator 里的 privileged expert 推导（而不是断言或人工）、闭环下的 per-pair 定向翻转率、
以及把配对差当训练目标并对照 hard-example reweighting。

## 1. 问题本身：planner 对可见因素几乎不反应，已经被多方独立量出来

| 来源 | 设置 | 量出来的数 |
|---|---|---|
| Beyond the Leaderboard，arXiv 2609.22582（2026-09-18） | 246 个 NAVSIM 近行人帧，SAM + LaMa 抹掉行人，6 个公开 checkpoint（LTF、DiffusionDrive、DiffusionDriveV2、SimLingo、AutoVLA、Alpamayo-1.5），全冻结 | 行人正落在规划路径上时，**只有 1.9% 的响应算 genuine avoidance**；clearance 变化中位数 ≤ 0.03 m；无关的夜景重打光让 plan 动得远 2–4 倍，方向随机 |
| CVAA / Counter-nuScenes，2607.16938（2026-07-18） | 3,062 对逐物体 inpainting 删除，Alpamayo-1 单模型 | 删红灯 vs 删绿灯的轨迹位移 0.758 vs 0.229 m，是目前唯一一组真实帧上的 traffic-light 敏感度数字；rank 稳定性只有 6.4% |
| CADET，2606.14438（2026-06-12） | query-level 扰动（改 detection query，不重渲染），SparseDrive，nuScenes-mini 81 帧 | 因果 agent 对无关 agent 的影响比 9.5，但单帧里最有影响的无关 agent 和真因果 agent 一样能推动 plan |
| Auto-JEPA，2607.29031（2026-07-31） | 同一帧 mask 动态 agent vs 等面积随机 mask | intent embedding 变化量是随机 mask 的 2.97× |
| Bench2Drive，2406.03877（2024-06） | AD-MLP（只吃 ego 历史、不看图）闭环 | DS 9.14、SR 0.00%，而开环 L2 只有 3.64 |
| 本项目，[decisions.md](decisions.md) 3d / 20 | Waymo E2E 半 val，冻结 Qwen3-VL-4B + 薄 head | 视觉相对 ego prior 的增量在 pre-manoeuvre-onset 子集上最小；线性读出在该子集上已榨干；s_ego 最高十分位的 logged future 本身不被 rater 认可 |

这一类失败的名字是 **shortcut / causal confusion**（模仿学习从相关量而非因果量学策略）：
de Haan 2019（1905.11979）、copycat（2010.14876、2106.06452）、ChauffeurNet 的 past-motion dropout（1812.03079）、
Is Ego Status All You Need（2312.03031）、PlanTF 的 state dropout（2309.10443）。
**经典线全部是「同一场景砍输入通道 / 加噪 / 重加权 / 改架构」，没有一篇构造成对场景**；配对是 2025 年之后才出现的。
2020 年的 Who Make Drivers Stop（2003.02425，IROS 2020）是「删一条 tracklet 并 inpaint、看 Stop/Go 翻转」的机制来源，
当时就带动作标签，也当时就发现 inpainting artifact 会污染周边。

## 2. 考卷一侧：反事实配对评测

### 2.1 编辑真实图像造对

| 论文 | id / 日期 | 编辑什么、怎么编 | ego state 一致 | 动作标签来源 | 测什么 | 用于训练 |
|---|---|---|---|---|---|---|
| Who Make Drivers Stop / DROID | 2003.02425 / 2106.13201，2020–21 | 删 tracklet + partial-conv inpaint | 是 | 有，Stop / Go 二分类 | 自家分类器的翻转 | 否 |
| CVAA / Counter-nuScenes | 2607.16938，2026-07 | 逐物体删除，Gemini imgen 与 LaMa + FLUX Fill 两条管线 | 是 | 无 | 轨迹位移、中间层表征 | 否 |
| CoLT-Drive（NVIDIA，EMNLP 2026） | 2609.00242，2026-08-31 | Gemini-3-Pro-Image 插入罕见物体，29 个基场景 3,536 样本 | 是 | **人工**：3 名标注者标可接受的纵×横 meta-action 集合，Krippendorff α = 0.825 | 11 个模型的 meta-action 准确率；「pair」指 v_full vs v_clean（剥离交通上下文），不是 factual vs counterfactual | 否，明确禁止 |
| Beyond the Leaderboard | 2609.22582，2026-09-18 | SAM + LaMa 删行人（同时删 LiDAR 点）；固定光度变换做夜景当无关对照 | 是 | **无**，只看 policy 自身输出差 | faithfulness 位移 F、clearance、genuine-avoidance 判据 F ≥ 0.5 m 且 F > I（I 为无关编辑的干扰） | 否 |
| Challenger | 2505.15880，2025-05 | 一个 agent 改成对抗轨迹，MagicDriveDiT 重渲；Adv-nuSc 156 场景 | 是 | 无，但验 solvability（ego 按 GT 轨迹能安全通过，否则整场丢弃） | **三臂**：original / re-rendered / re-rendered + edit；纯重渲染让碰撞率 0.29% → 0.32%，对抗 agent 让它到 3.95–7.05% | 否 |
| Probing Visual Concepts in Lightweight VLMs | 2603.06054，2026-03 | 只差一个 visual concept 的 counterfactual image set | 不明 | 无，只有 concept label | linear probe 准确率；perceptual vs cognitive failure 二分 | 否 |

**编辑一侧已经确立的三件事**：机制是 2020 年的；跨多个公开 checkpoint 的 model-agnostic 考试已经做了（2609.22582 六个、CoLT-Drive 十一个）；
「较大的无关外观变化不应翻转」这个对照已经做了（2609.22582 的 F > I，CADET 的 CSI）。
**还没有人做的**：红绿灯状态切换（红 ↔ 绿）的定量；语义安慰剂对照（行人换成等面积的无关物体，编辑量相同但正确动作不变）；
在 3DGS（3D Gaussian Splatting，真实场景的可编辑重建）里插删 agent 并报 per-scene 配对差（HUGSIM 机制支持，无人用）。

### 2.2 仿真里的配对

| 论文 | id / 日期 | 配对什么 | ego state 逐帧一致 | 标签来源 | 测什么 | 用于训练 |
|---|---|---|---|---|---|---|
| Fail2Drive（Geiger 组） | 2604.08535，2026-04 | CARLA 100 对 route，17 类 shift；固定地图、几何、spawn、交通目标，只改一个 targeted shift | **否**（闭环各自 rollout，到场景点时已发散） | 无；PDMLite-F2D 特权专家做 solvability check | 聚合 DS / SR / HM 的相对下降，7 个模型，平均 HM −16.3%、SR −22.8% | **Rule 1 明确禁止** |
| Safe2Drive | 2606.00191，2026-05 | 100 个难场景 + SafeDriving Score | 否，不配对 | 无 | LEAD 94.70 → 39.95，SimLingo 85.07 → 41.00 | 否 |
| Bench2Drive-Robust | 2605.18059，2026-05 | 同 route ± 系统层扰动（丢帧、定位噪声、控制延迟） | 否 | 无 | 闭环分下降 | 否 |
| ICR-Drive | 2604.05378，2026-04 | 同一 CARLA route，seed / map / weather / traffic 全钉死，只换 instruction 文本 | 是（显式重放） | 无 | per-family 的 DS / RC / IS 下降 | 否 |
| How Can Driving World Models Do Counterfactual Prediction | 2608.11601，2026-08 | 同一 CARLA world 共享 15 帧 history，只改 ego action 与 event 是否触发 | 是 | **simulated**（重跑即 ground truth） | 评的是 world model（Vista、DrivingWorld），不评 planner | 否 |
| Causality9k | 2504.14709，2025-04 | WOD 上同 1 s ego 历史，DFS 生成 alternative goal | 是 | 算法构造 | 自研闭环 simulator 上暴露 copycat | 否 |
| HOIST | 2312.02467，2023-12 | CARLA 里改 object 运动打重要性分 | 是 | 无 | object importance | 否 |
| Exploring the Causality of E2E AD | 2407.06546，2024-07 | CARLA 控制变量 + BEV feature masking | 是 | 无 | map 扰动几乎无影响；BEV mask 后大量 collision / 闯红灯 | 否 |

Fail2Drive 的四类 shift 里三类是 invariance test（正确动作不变），只有 Behavioral 一类要求不同行为；
它开源了自定义 CARLA 0.9.15 build、参数化场景工具箱和 PDMLite-F2D。
**仿真一侧没有人做的**：物理渲染、ego state 逐帧对齐、intervention 落在可见场景因子上、
且「正确动作确实不同」由 privileged expert 两侧重跑给出的配对；闭环下的 per-pair 定向翻转率
（开环有：2609.22582 的 1.9%，CADET 的 CRI）。

### 2.3 文献已经记录的 validity threat 与对策

| 威胁 | 证据 | 已有对策 | 出处 |
|---|---|---|---|
| 编辑痕迹成为 shortcut | inpainting artifact 占效应的 5–17%（2609.22582，原文两处数字不一致，引用要标注） | 安慰剂编辑：patch 贴到空路面，量它单独造成的位移当下界 | 2609.22582 |
| 编辑没编干净 | 删除后仍有 8.5% 被独立检测器检出 | COCO Faster R-CNN 独立验证 | 2609.22582 |
| 渲染漂移被误读成因果效应 | 纯重渲染碰撞率 +0.03–0.11 pp，效应 1–7 pp | 三臂设计 original / re-rendered / re-rendered + edit | Challenger 2505.15880 |
| 动作标签是断言不是推导 | SafeMVDrive、SafeGen、PhyGenesis 都由构造断言；NeuroNCAP 不做可解性检查 | 特权规则专家解场景（KING、Fail2Drive）；优化求解（STRIVE）；GT 轨迹可安全通过否则丢弃（Challenger）；人工标注可接受集合（CoLT-Drive） | 多篇 |
| 生成出无解场景 | AlignADV 点名 collision-driven generator 常合成 unsolvable 场景 | DPO 把生成器推向「关键但可解」 | 2606.14032 |
| 多帧时序一致 | layout-conditioned diffusion 整帧重采样 | DDIM inversion（UniScene）、在线修 artifact（ReconDreamer）、点云投影当几何锚（GeoDrive） | 多篇 |
| 几何 / 阴影 | 2D inpainting 插入物普遍缺阴影 | R3D2（2506.07826）一步扩散补阴影光照；Dream4Drive 用 depth / normal / edge guidance | 多篇 |
| 合成训练增益被 epoch 预算偷走 | 「合成预训练 + 真实微调」悄悄多给一倍 epoch | 匹配 epoch 预算的 baseline；Dream4Drive 匹配后增益从 +4.6 缩到 +1.4 mAP | 2510.19195 |
| 删掉某个输入只让 shortcut 搬家 | 拿掉速度后带历史的 BC 反超单帧 BC；PlanTF 删历史后从单帧 kinematic 学出新 shortcut，要 0.75 的 state dropout 才压住 | 任何「删输入」补救都欠一个「shortcut 没搬到别处」的证明 | 2106.06452、2309.10443、1812.03079 |
| saliency ≠ causal object | 注意力 blob 里 58–62% 是 spurious（Kim & Canny，ICCV 2017） | 不能用注意力图当因果证据 | 1703.10631 |
| 特征级删除比像素级乐观 | CADET 在 query 层删 agent，只测一个 checkpoint 81 帧 | 像素级或渲染级删除；闭环祖先是 PlanT 的 RFDS | 2606.14438、2210.14222 |

## 3. 教材一侧：用配对训练

| 工作 | id / 日期 | 目标形式 | 与「交互差」的距离 |
|---|---|---|---|
| CF-Driver / Good Data Is All IL Needs | 2409.17605，2024-09 | DICE 在 actor 状态向量的输入空间做最小改动使动作分类翻转；rule-based CARLA expert 重跑打标签；CF 样本当**普通 imitation 样本**混进数据集，Town05 long DS 84.2 | 训练侧最近的一篇。有配对 + simulated label，但 loss 无配对结构，ego state 不保持一致，**没有 hard-mining 对照** |
| Hydra-MDP / MDP++ | 2406.06978 / 2503.12820 | 离线把 vocabulary 每条轨迹在 simulator 里打分，多头 BCE 蒸馏 | 提供了 J(x, a)，但只有一个 x |
| CRAFT | 2605.04470，2026-05 | 同一 state 上一组 action 的 group-normalized counterfactual advantage | J(x, a) − mean_b J(x, b)，即单场景内层差；场景先验减不掉 |
| DriveDPO | 2509.17940，NeurIPS 2025 | 轨迹级 DPO | 单场景内 pairwise 一次差 |
| CounterPlay | 2609.21617，2026-09 | 失败后 backtrack、换 driving style 重跑，成功分支蒸馏回来（KL） | pair 是同 state 两条 rollout，改的是 ego 策略不是场景 |
| Policy Contrastive Decoding | 2505.13255，ICLR 2026 | 原始观测与 object-masked 观测的 action 分布相减 | 正是 π(·\|x⁺) − π(·\|x⁻)，但在 decoding 不在 training |
| OmniDrive / nuReasoning / VeriDrive | 2405.01533 / 2605.31572 / 2606.07338 | counterfactual QA（假想动作后果）进训练集 | 文本级，不是像素级场景对 |
| ReconDreamer-RL | 2508.08170，2025-08 | 3DGS + diffusion 交互 sim 里造 corner case 做闭环 RL，碰撞率 ÷5 | 真的训了 planner，但不是配对目标 |
| ChatScene / CAT / KING / AdvSim | 2405.14062 / 2310.12432 / 2204.13683 / 2101.06549 | 对抗场景生成后微调或对抗训练 | 两侧是不同 rollout，无配对 loss |
| Keyframe-Focused Visual IL | 2106.06452，ICML 2021 | 给 action changepoint 的 keyframe 加权 | **re-weighting 的先例**：同一个 copycat 问题，纯加权在 CARLA 上就把 history-based BC 救活了 |
| CoDA / MoCoDA / ICIL / IRM / Swamy 2022 | 2007.02863 等 | RL / IL 里的 counterfactual data、invariance | 造的是数据或 invariance 约束，不是 per-pair outcome difference |

**训练一侧的现状**：用配对干预样本训 driving planner 已做（CF-Driver）；
**双差**（某个可见因子改变了多少 action 之间的相对优劣）没有任何论文当 training target；
**没有一篇 counterfactual training 工作对照过 hard-example reweighting**；
sim 配对训练 → 真实数据（NAVSIM / Waymo）的迁移收益没有检索到。

## 4. 视频生成模型当 planner 表征

### 4.1 谁做了

| 论文 | id / 日期 | 视频模型 | 冻结 | benchmark | 对照 | 结果 |
|---|---|---|---|---|---|---|
| **DriveLaW**（CVPR 2026） | 2512.23421，2025-12 | LTX-Video DiT | 否（Video DiT 与 Planning DiT 一起更新） | NAVSIM Navtest | BEV 特征、VLM hidden state（Qwen2.5-VL） | **video latent 89.1 PDMS > VLM 86.5 > BEV 84.1**；denoise step t = 1 / 5 / 10 → 89.1 / 86.9 / 23.2；video 预训练 0 → 7.6M clips → 85.9 → 89.1 |
| DriveWAM | 2605.28544，2026-05 | Wan2.2-TI2V-5B | 否，full fine-tune | NAVSIM + PhysicalAI-AV | 无 DINOv2 / SigLIP 对照 | PDMS 90.1；4k → 100k clips 有 scaling |
| DriveVA | 2604.04198，2026-04 | 未点名的大规模视频生成模型 | 未说明 | NAVSIM 90.9 + nuScenes + **Bench2Drive** | 只比 world-model 类 planner | 唯一把 video-gen prior 推到 Bench2Drive 闭环的公开结果，无 encoder 对照 |
| DriveVLA-W0 | 2510.12796，2025-10 | AR / diffusion world model 当 dense supervision | 否，训练范式 | NAVSIM + 内部数据 | BEV、VLA baseline | claim 是放大 data scaling 的斜率 |
| PerceptDrive | 2607.20175，2026-07 | frozen self-supervised video encoder（未点名，很可能是 V-JEPA 类） | **是** | NAVSIM 90.4 | 无 | 唯一明确 frozen 的，但大概率不是生成式 |
| Cosmos Policy（NVIDIA + Stanford） | 2601.16163，2026-01 | Cosmos-Predict2 | 否，post-train | 机器人 LIBERO 98.5% 等 | 无 | code / models / data 全开 |
| Alpamayo-R1 | 2511.00088，2025-10 | 不是生成模型：backbone 是 Cosmos-Reason（VLM） | — | — | — | NVIDIA 自家做驾驶 VLA 选的是 reasoning 分支 |

**完全冻结通用视频生成模型 + 薄 head**，以及 **video-gen 对 DINOv2 / V-JEPA / SigLIP2 在同一 driving benchmark 上并排**，都没有找到。
DriveLaW 的 denoise-step 表是最可操作的一条：要在高噪声端 tap，靠近干净像素时灾难性塌缩（Comfort 归零）；
DeepMind 的 2502.07001 给出的经验法则是约 2/3 深度、timestep ≈ 200。

### 4.2 生成式表征到底编码了什么

| 层次 | 证据 | 结论 |
|---|---|---|
| 低层几何、对应、tracking | DIFT 63.5 vs DINO 45.0（SPair-71k）；DiTracker 里 WAN-14B 51.2 > DINOv3-B 41.1 > DINOv2-B 40.0 > V-JEPA2 33.8（TAP-Vid）；2507.13942 里 W.A.L.T. 在 pixel / depth forecasting 上把 masking 模型甩开 3–4 倍 | **generative 明确赢** |
| 语义 | REPA（2410.06940）：SiT 的 linear probe「well below DINOv2」，灌入 DINOv2 表征训练加速 17.5×；Chen & He（2401.14404）：去 class-conditioning 后 probe 升、FID 降，「SSL 性能与生成质量不相关」；2502.07001：video diffusion 相对 image diffusion 在 tracking +68%，semantic 几乎不涨，1.9B V-WALT 语义仍输 300M DINOv2 | **discriminative 赢** |
| intuitive physics | 2606.09646（frozen probing）：IntPhys2 上 VideoMAE 73.9 > V-JEPA2 66.0 > **LTX-Video 48.4，随机水平**；MVP 上 V-JEPA 95.0 > LTX-Video 83.0 | **generative 最弱** |
| 视觉真实度 ⇒ 物理？ | Physics-IQ（2501.09038）：真实度与物理分 r = −0.46 不显著，Sora 最真实最差（10/100）；PhyGenBench：VEnhancer 提画质、物理分不动；2411.02385：泛化是 case-based，检索优先级 color > size > velocity > shape，OOD 失败；PhysWeep（2609.06207）：参数由 seed 而非 prompt 决定；PAWBench（2608.27345）：没有模型能复现结果分布 | **已证伪** |
| 反方 | Veo 3（2509.20328）：scale 改变结论，5×5 maze 78% vs 14%，但测的是感知与 puzzle | 唯一实质反方 |

**综合**：generative video pretraining 给的是低层几何、对应、短时运动先验，不是可靠的物理或高层语义先验；两者互补不是替代。
VaVAM（2502.15672，decisions 第 19 条已记）的形状是：scaling video 预训练改善 open-loop、抬高 closed-loop 碰撞。

## 5. MiniMax H3 事实表

| 项 | 事实 |
|---|---|
| 开放状态 | 2026-07-31 API 发布，**2026-08-03 权重上 HF**（`MiniMaxAI/MiniMax-H3`，Diffusers），Community License 排除美 / 欧 / 英 / 韩自托管 |
| 结构 | 理解侧 **Qwen3-VL-32B 取第 50 层 hidden state** 当 conditioner；生成侧 H3-Omni-Transformer，33B dense 单流 DiT，约 13B 在 AdaLN 分支（推理可预计算，不必加载）；VisualVAE f16t4d24；AudioVAE |
| checkpoint | FL2VA（text / 首帧 / 末帧 / 首末帧）、Ref2VA（图 ≤ 9、视频 ≤ 3、音频 ≤ 3 做 reference）；Context-IR、Regenerate-2K 只有托管版 |
| 输出 | 768p、24 fps、4–15 s、32 kHz 立体声 |
| 官方 tech report | **没有**；HF 上 221 个相关仓库；第三方 arXiv 六篇 |
| 物理评测 | 2609.18323（2026-09-16）：四类物理世界推理任务、517 个实例，**整体成功率 41.97%** |
| 当 world model | H3-World（2609.01560）：0.2% 参数的 LoRA 变成交互式 world model，原话 directly reuses the semantic representations learned during video pretraining；SolarWM（2609.02886）：把 Wan2.2 / LTX-2.5 / H3 改成 causal 世界模型，数据、pipeline、权重全开 |
| 量化 | GGUF 从 Q2_K 6.72 GB 起，pruned Q5_K_M 14.1 GB；INT8 / FP8 / NVFP4 / NF4 / W4A8；SGLang 官方 4090 配方峰值约 18 GB |
| 「实时」 | **不成立，差 35–90 倍**：4090 D 24 GB 上 1344×768、107 帧（4.46 s）、20 NFE，BF16 405.6 s，最快配置（int8 + 近似注意力）163.8 s；官方 50 步外推约 1000 s。快过播放的只有 FastH3 4-step 在 4× B300（RTF 0.79）和 Video DeltaNet 在 8× B200（6.7 s 出 14.3 s）；RAVEN streaming LoRA 作者原话 reaching real-time will require further inference acceleration。量化只给 1.34×，杠杆是步数蒸馏 |
| 当 editor | V2V 是 ref2va 的用法，不是独立 task；输出时长限死 5–15 s；diffusers 原话 references do not bind the generated geometry，是参考重生成不是 inpainting |
| 取特征 | 单帧 768p = 1008 token，4 帧 = 2016 token；pruned bf16 40.2 GB 常驻单卡，text encoder 预编码一次；估 30–100 ms / forward（推算，未实测）；diffusers 的 blocks 卡 num_frames 为 17n+5，要绕过 blocks 直接调 transformer |
| 生成成本（RTX PRO 6000） | VDN-H3 8-step：160 s denoise + 20.6 s decode，峰值 95,790 MB，单卡时 text encoder 必须 offload |

## 6. 可用的开源工具

| 用途 | 工具 | 关键规格 |
|---|---|---|
| 仿真配对场景 | Fail2Drive 工具箱 + PDMLite-F2D（Geiger 组，开源） | 参数化场景、solvability check、特权示范；自定义 CARLA 0.9.15 build |
| nuScenes 帧里按 3D box 插删物体 | **DriveEditor**（2412.19458，MIT） | SVD + SV3D，10 帧 576×1024，插入 mRecall 0.94、mATE 0.66 m；>32 GB；自陈小目标 / 夜景失效 |
| 相机 + LiDAR 联合插入 | **MObI**（2501.03173） | Paint-by-Example + 3D box 条件 |
| 插入物补阴影 / 光照 | **R3D2**（2506.07826，Apache-2.0，HF gated） | sd-turbo 一步，逐帧实时 |
| 3D 标注 → 照片级重画 | **Cosmos-Transfer2.5 Edge Distilled** | RTX PRO 6000 单卡 93 帧 720p 实测 78.5 s，65.4 GB；整帧重画，背景不是原像素；Cosmos-Drive-Dreams 有 Waymo → RDS-HQ 转换脚本 |
| 单帧指令编辑 | Mage-Flow-Edit-Turbo 4B（约 1 s，18–20 GB）、Step1X-Edit v1p2（Apache-2.0）、SDXL-inpaint + ControlNet（秒级，4–8 GB） | 多帧一致性全军覆没；Qwen 系列有 pixel drift |
| 视频 inpainting | Wan2.1-VACE-14B（720p 峰值 122 GB，放不下）、DiffuEraser（Apache-2.0）、ProPainter（非扩散，背景逐像素保持，BeyondMasks 上反超 VACE） | Wan 家族帧数必须 4n+1、训练在 81 帧；最后要自己用膨胀 mask 做像素域 alpha composite |
| 开权重视频生成模型 | Wan2.2-TI2V-5B（Apache-2.0，24 GB 可跑，DriveWAM 基座）、Wan2.1-14B（DiTracker 最强）、LTX-Video（DriveLaW 与 2606.09646 共用，可比性最好）、Cosmos-Predict2.5、H3 | 闭源 API（Kling、Veo、Sora、Runway、Seedance）拿不到 latent |

## 7. 空白表

| 命题 | 判决 | 决定性引用 |
|---|---|---|
| ego state 一致的最小可见差配对考试 | 已做 | 2609.22582；机制 2003.02425 |
| 同上，由 simulator 物理渲染 | 没人做过 | 最近三篇全是图像编辑；2608.11601 改的是 ego action、评 world model |
| 「正确动作确实不同」由 privileged expert 两侧重跑给出 | 没人做过 | CoLT-Drive 人工；2609.22582 不建立 ground truth |
| 跨多个公开 checkpoint 的 model-agnostic exam | 已做 | 2609.22582（6）、CoLT-Drive（11）、Fail2Drive（7） |
| per-pair 定向翻转率 | 部分做过 | 开环有（1.9%、CADET CRI）；闭环无 |
| 无关外观变化不应翻转的对照 | 已做 | 2609.22582 的 F > I；CADET 的 CSI |
| 红绿灯状态切换、语义安慰剂对照 | 没人做过 | 零命中 |
| 3DGS 真实场景插删 agent 报 per-scene 配对差 | 没人做过 | HUGSIM 支持但无人用 |
| 用配对干预样本训 planner | 已做 | CF-Driver 2409.17605 |
| 交互差（双差）目标训 planner | 没人做过 | CRAFT 只到内层括号 |
| 配对训练对照 hard-example reweighting | 没人做过 | CF-Driver 等全无；先例 2106.06452 |
| sim 配对训练 → 真实数据迁移收益 | 没人做过 | 未检索到 |
| CARLA 出标签 + 编辑真实图当外部效度的混合 | 没人做过 | 两边论文互不引用 |
| 视频生成模型 latent 喂 driving planner | 已做，赢 VLM +2.6 PDMS | DriveLaW 2512.23421 |
| 完全冻结通用视频生成模型 + 薄 head | 没人做过 | 全表主干都更新 |
| video-gen 对 DINOv2 / V-JEPA / SigLIP2 同 driving benchmark 并排 | 没人做过 | DriveLaW 只比到 VLM / BEV |
| video-gen prior 上 Bench2Drive 闭环 | 部分做过 | DriveVA，基座未点名、无对照 |
| 视觉真实度蕴含可用物理 | 已证伪 | Physics-IQ、PhyGenBench、2606.09646 |
| generative 特征全面弱于 discriminative | 也证伪（过强） | DiTracker vs REPA |
| generative 特征能修 pre-onset 那一格 | 没人做过 | 先验证据为负（VaVAM、2411.02385、2606.09646） |
| H3 理解侧相对 Qwen3-VL 有增量 | 基本否定 | stock Qwen3-VL-32B 第 50 层 |
| 从 H3 抽特征做下游 | 没人做过 | 开权重才一个多月 |

## 参考

编辑与配对：2003.02425 · 2106.13201 · 2607.16938 · 2609.00242 · 2609.22582 · 2505.15880 · 2603.06054 · 2604.08535 · 2606.00191 ·
2605.18059 · 2604.05378 · 2608.11601 · 2504.14709 · 2312.02467 · 2407.06546 · 2606.14438 · 2607.29031 · 2606.14032 · 2510.19195 ·
2506.07826 · 2412.19458 · 2501.03173 · 2511.00062 · 2412.01718 · 2404.07762
训练：2409.17605 · 2406.06978 · 2605.04470 · 2509.17940 · 2609.21617 · 2505.13255 · 2405.01533 · 2605.31572 · 2606.07338 · 2508.08170 ·
2106.06452 · 2309.10443 · 1812.03079 · 1905.11979 · 2010.14876 · 2312.03031 · 1703.10631 · 2210.14222
视频生成表征：2512.23421 · 2605.28544 · 2604.04198 · 2510.12796 · 2607.20175 · 2601.16163 · 2511.00088 · 2306.03881 · 2512.20606 ·
2502.07001 · 2507.13942 · 2410.06940 · 2401.14404 · 2501.09038 · 2411.02385 · 2410.05363 · 2609.06207 · 2608.27345 · 2606.09646 · 2509.20328 · 2502.15672
H3：2609.18323 · 2609.01560 · 2609.02886 · 2609.20744 · 2609.15810 · 2608.25927
