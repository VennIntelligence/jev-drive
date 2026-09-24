# 留给综合阶段的问题

以下仅列待综合判读的问题，未对因果、驾驶能力或指标可信度下结论；每项均指向主表记录、论文表图或代码/issue。

## W1 增益归因

1. **估计的 expert 分数如何使用？** `TFv6` 的 `expert_reference` 行（`tfv6.pdf`, Table 5, PDF p.7）把 LEAD expert 96.8 与 TFv6 95.2 同列，但作者在 [issue #90](https://github.com/kesai-labs/lead/issues/90#issuecomment-4779487301) 澄清前者是按另一 leaderboard 成绩估计，未直接跑 Bench2Drive。综合阶段是否只将这行作背景参照，而不作严格同协议差值？这条说明不指向 Table 1 中 TFv5 的 84.94。
2. **同一搜索后处理能否跨基模型、跨榜归因？** `TOAD`, `TOAD+DrivoR`, `DrivoR+TOAD` 的 `test_time_cem_search` 行（`toad.pdf`, Tables 1–2, PDF p.5；Table 4, PDF p.6）对若干基模型的 PDMS、EPDMS、HDS 变化不同。如何界定搜索本身与 proposal 质量、测试协议、不同 checkpoint 的作用？
3. **早停分数与驾驶完成度如何权衡？** `CarLLaVA`、`SimLingo-BASE` 的 `early_termination` / `early_termination_distance` 行（`carllava.pdf`, Table 2, PDF p.4；`simlingo_base.pdf`, Tables 9–10, PDF p.17）显示不同距离阈值的官方 DS 变化；应怎样结合 W4 的 CARLA 计分实现解释？
4. **逐级消融是否足以分解联合增益？** `LinkVLA` 的 `token_c2f_alignment_sequence` 行（`linkvla.pdf`, Table 5, PDF p.8）按 token→C2F→alignment 逐级加入。次序是否造成交互或训练预算混杂，导致各级增量不能视为独立贡献？
5. **效率与闭环得分如何一起比较？** `FIVE-VLA` 的 `ram_and_backbone`、`ram_inference_mode` 行（`five_vla.pdf`, Tables 7–8, PDF p.11）同时报告 DS/SR 与 T4 FPS，部分配置得分相同但吞吐不同。综合时是否应把这些作为多目标权衡，而非合成一个“增益”？
6. **测试时训练属于同权重比较吗？** `DriveVLA-M0` 的 `memory_injection`, `ttt_learning_rate`, `ttt_gradient_steps`, `memory_scale_and_ttt` 行（`drivevla_m0.pdf`, Tables 1/4/8, PDF pp.7/8/12）共用起始模型但推理时更新参数。跨榜汇总时是否需要单列测试时优化的预算与数据访问条件？
7. **多 seed 的默认配置能否和单个变体比较？** `LTF/TransFuser` 的输入与任务消融行（`ltf.pdf`, Table 2, PDF p.8）默认配置按三个 seed 分别给 83.3/84.0/84.4，变体给单一值，因此账本将基线和差值记 `N/A`。综合阶段是否能从原文确认可比较的配对 seed 或汇总统计？
8. **图上无精确数字的实验可给多大权重？** `SparseOccVLA` 的 `distillation_loss`（`sparseoccvla.pdf`, Figure 4b），`RAP-DINO` 的 `feature_alignment`、`cross_agent_synthetic_data_scale`（`rap.pdf`, Figures 5–6），以及 `Poutine` 的 `grpo_posttraining`（`poutine.pdf`, Figure 5）有趋势但未可靠精确数字化；SparseOccVLA `occupancy_query_count` 与 TOAD `candidate_count_M` 的端点有直接标注，已录入数值。后续是否仅对前一类使用定性证据？
9. **C 档论文自报实验的证据等级如何标注？** `LinkVLA`, `SteerVLA`, `FIVE-VLA`, `RoG-DAgger`, `Kyber-E2E`, `DriveFuture`, `OmniSpace`, `LVLDrive`, `NTR`, `Poutine` 等行只能核论文表图，无法核训练代码或评测代码。是否需将同幅度但可复核性不同的增量分层呈现？
10. **联合改变的数据和训练预算如何归因？** `FIVE-VLA` 的 `RAM_vs_longer_training`（`five_vla.pdf`, Table 7, PDF p.11）、`DrivoR` 的 `synthetic_training_data`（`drivor.pdf`, Table 3, PDF p.6）及 `Poutine` 的 `CoVLA_pretraining` / `grpo_posttraining`（`poutine.pdf`, Table 1, PDF p.4；Figure 1, PDF p.1）跨训练阶段或数据源。需要哪些匹配预算的控制才能解释其单因素效果？
11. **辅助协议的结果能推广到主榜吗？** `AutoVLA` 的 `action_tokenization_method` 在 `aux:action_tokenization`（`autovla.pdf`, Table 4），`TFv6` 的 `aux:town13`、`aux:longest6_v2`（`tfv6.pdf`, Tables 1–5）和部分主榜同名指标并列。应采用什么协议对齐条件后才作跨协议比较？
12. **开放代码应该在哪里合并？** `memory_injection`、`recurrent_action_memory`、`ram_inference_mode`，以及 `sensor_lidar`、`lidar_modality` 等代码语义相近但实施不同。综合阶段是按方法内实验保留细粒度，还是根据可证明相同的处理和协议再合并？

## W2 跨榜一致性

1. Bench2Drive 官方 Table 3 中，`UniAD-Base` 的开环 Avg L2 为 0.73、闭环 DS/SR 为 45.81/16.36；`VAD` 分别是 0.91、42.35/15.00。正文却称 UniAD-Base 闭环较差，可能指其低于 VAD 的 Efficiency/Comfortness。综合阶段应如何解释该语句与不同闭环指标的关系，而不把机械 Spearman 当作因果或部署能力？涉及 `w2_cross_board.csv` 中 Bench2Drive `base-set open-loop 2s@2Hz` 与 `base-set closed-loop 220 routes` 的对应行，以及 `w2_correlations.md` 首四项。
2. NAVSIM v1 `navtest` PDMS 与 NAVSIM v2 `navhard-two-stage` 修复后 EPDMS 的排序变化，究竟来自场景分布、指标、两阶段合成观测、训练目标，还是 checkpoint/重评版本？涉及 `w2_cross_board.csv` 中这两列、TOAD Tables 5/6、DrivoR Tables 13/14 及 `w2_correlations.md` 第五项。
3. 对同方法跨榜的统计，什么证据足以确认同一 checkpoint？TOAD 说使用公开 checkpoint 且不更新权重，但没有逐一对应 v1/v2 的文件或 hash；RAP-DINO 自述直接评估同一训练模型，而其 WOD 和 Bench2Drive 分别微调和更换骨干。涉及 `RAP-DINO`、`RAP-ResNet`、`DrivoR`、`iPad` 等行的 `same_checkpoint` 字段。
4. CARLA LB2 官方 `MAP`/`SENSORS` 的 DS、Bench2Drive 220 短路线 DS 与 Longest6 v2 长路线 DS 对同一方法分别代表哪些可迁移行为？是否应按路线长度、早停规则、传感器轨道和专家数据来源分层？涉及 `SimLingo-BASE`、`TF++`、TFv6 Table 5 的变体、RoG-DAgger Table 1，以及 `w2_correlations.md` 的 Longest6 四项。
5. NAVSIM v2 旧 `navtest EPDMS*`、修复后 `navtest EPDMS` 和官方 `navhard-two-stage EPDMS` 应如何进入统一综合表？NTR 的 `navtest` 90.9 未说明修复状态。涉及 `SparseDriveV2`、`DriveFuture`、`WA-JEPA`、`NTR`、`LTF` 和 `DrivoR` 行的 `protocol/split`。
6. `WOD-E2E test RFS` 与 `NAVSIM v1 navtest PDMS` 的五个重名方法中，有两个 DriveMA 模型规模，且至少部分跨数据集重新训练。这个样本级相关是否值得用于跨榜一致性判断，还是只应作已发表分数索引？涉及 `AutoVLA`、`DriveMA-2B`、`DriveMA-4B`、`NTR`、`RAP-DINO`，以及 `w2_correlations.md` 最后一项。
7. HUGSIM WA-JEPA 的统一 436 场景协议，与 TOAD 分数据源的四组 HDS、HUGSIM 官方论文原始难度分组如何对齐？是否可对某一固定场景/控制器的结果比较 NAVSIM 排序？涉及 `hugsim` 行及 `w2_correlations.md` 的小样本配对。
8. nuScenes 的 L2 表中 ST-P3、UniAD、OmniSpace 等报告采用不同评测实现、ego 输入或修正，和 Bench2Drive 的重训模型能否视为一对可比 checkpoint？涉及 `AD-MLP`、`VAD-Base`、`AutoVLA`、`OmniSpace`、`SteerVLA` 的 `nuscenes` 与 `bench2drive` 行。

## W3 复现差距

1. [TFv6 #90](https://github.com/kesai-labs/lead/issues/90) 中作者称论文 Table 5 的 LEAD expert Bench2Drive 数字是估计，而非实际 Bench2Drive 评测。综合阶段应如何标注该行与直接测得的 DS 的可比性？相关记录：`kesai-labs/lead#90`。
2. [AD-MLP #4](https://github.com/E2E-AD/AD-MLP/issues/4) 的用户指控未来 GT 混入 CAN bus，作者确认重审数据并修订训练流程，但未逐项确认用户的具体指控。应如何把旧/新论文分数、发布 pkl 和代码证据对应起来？相关记录：`E2E-AD/AD-MLP#4,#5`。
3. [NAVSIM #151](https://github.com/autonomousvision/navsim/issues/151) 明确承认 human penalty filter bug；挑战期保留，后修复。哪些榜单快照与论文分数对应修复前或修复后协议？相关记录：`autonomousvision/navsim#151,#158,#172`、`valeoai/DrivoR#31,#47`。
4. [DrivoR #54](https://github.com/valeoai/DrivoR/issues/54) 多名用户在旧 345 场景 HUGSIM 上得到不同复现值，作者指定权重和 LTF 管线后仍有未解释差距。哪些配置差异可由原文核实，哪些仍不可判定？相关记录：`valeoai/DrivoR#34,#38,#54`。
5. [SimLingo #43](https://github.com/RenzKa/simlingo/issues/43) 中改用 Bench2Drive 定制目录后 DS 75.50→86.53，且 Town13 进入训练。综合阶段如何与 CARLA LB2 和 Bench2Drive 的其他分数分开比较？相关记录：`RenzKa/simlingo#43,#72`、`autonomousvision/carla_garage#95,#103,#108`。
6. [AutoVLA #42](https://github.com/ucla-mobility/AutoVLA/issues/42) 中作者最终说 nuScenes 附录结果需要额外 RFT，发布的 checkpoint 是 NAVSIM RFT；[B2D #43](https://github.com/ucla-mobility/AutoVLA/issues/43) 的评测管线也有未发布依赖。哪些已发表数字能对应到可取得的权重与代码？相关记录：`ucla-mobility/AutoVLA#30,#42,#43,#48,#56`。
7. [GTRS #4](https://github.com/NVlabs/GTRS/issues/4) 与 [NAVSIM #140](https://github.com/autonomousvision/navsim/issues/140) 提出 `pinv`/`solve`、NumPy 版本影响 EP。该差别影响哪些已发表 EPDMS/PDMS 对照，是否有固定环境可重算？相关记录：`NVlabs/GTRS#4`、`autonomousvision/navsim#81,#140`。
8. [SparseDriveV2 #7](https://github.com/swc-17/SparseDriveV2/issues/7) 作者称推理评分设计为最大化 PDMS，[#13](https://github.com/swc-17/SparseDriveV2/issues/13) 称 Bench2Drive 未用该监督。综合阶段如何解读该模型两榜分数的训练目标差异？相关记录：`swc-17/SparseDriveV2#7,#13,#16`。

## W4 榜单指标

1. **同名榜单的版本分数能否横比？** `navsim_v1.md` 记录 v1.0 将 DDC 乘入 PDMS，而 v1.1 把 DDC 设为零权；`navsim_v2.md` 记录 v2.0 与 v2.2 的后续场景汇总公式不同。W1/W2 的每个 PDMS/EPDMS 数字实际使用哪个 devkit commit？如缺少版本，该数字应如何标注不可比？
2. **提前停车的净收益范围是多少？** `carla_lb2.md` 的 `LB2-TFPP-001` 有条件地从 `DS=RC×IP` 推出提分可能；需结合 W1 的同一模型消融、路线长度和实际停止配置，判定它在官方提交分数中占多少，以及是否在短路线 Bench2Drive 也存在类似激励。
3. **规则后处理贡献与模型能力如何拆开？** `bench2drive.md` 的 `B2D-TFV6-003`、`B2D-BLUE-001`、`B2D-TFV6-002` 与 `carla_lb2.md` 的 `LB2-TFPP-002/003` 只说明官方计分对违规、blocked 有敏感性；需要同 checkpoint 开/关规则的配对结果，判断是修补模型缺陷还是一般安全控制，以及对实际驾驶的影响。
4. **NAVSIM 子分数代理是否过拟合榜单？** `navsim_v1.md` 的 `NAV1-DVM0-001`、`NAV1-DRIVOR-002`、`NAV1-TOAD-001`、`NAV1-RAP-001` 与 `navsim_v2.md` 的 `NAV2-GTRS-002/003`、`NAV2-DRIVOR-001/002`、`NAV2-TOAD-001/003` 均可对有限评分目标选轨。W1 中代理评分 ablation 与 W2 跨闭环指标是否支持或反驳“只优化代理”的解释？
5. **v2.2 伪闭环权重能否被候选轨迹有意影响？** `navsim_v2.md` 的“首段终点改变合成后续场景计分权重”属于代码支持的可能性；需要固定同一首段场景、改变终点和真实行驶质量的配对分析，区分协议局限与实际攻击。
6. **nuScenes 开环 L2 的信息泄漏在各论文评测中实际发生了吗？** `nuscenes.md` 的 UniAD 评分代码只接收预测与 GT；第一轮 `REAL-NUS-004/006` 涉及未来 GT 派生命令，应沿每个方法的数据加载、输入可用性和实际评测脚本追踪。若不同论文采用不同碰撞率实现，也需逐篇对齐，避免把本页的 UniAD 版本泛化。
7. **WOD RFS 的几何空缺是否映射到真实安全问题？** `wod_e2e.md` 的 3/5 秒离散检查、区外 4.0 下限、跨 rater 最佳匹配是代码事实；结合 W1 的 `REAL-WOD-001/002/003`，需要具体候选轨迹和人工标签分布，才可判断是否出现分数高而路径不可驾驶的例子。官方服务器还有哪些公开的提交约束？
8. **HUGSIM 的 HD-Score 是否奖励失败制动或漏检对象？** `hugsim.md` 的 EP 不入单帧公式，`HUG-WA-002` 的 fallback 可能保持部分子分数；需路线完成率分布与失败前后关键帧分数。非 car 目标框的覆盖范围还需核查实际 `scene_xyz` 是否包含这些对象。
9. **“未观察到”的攻击面应如何在综合结论中呈现？** `w4_attack_surface.csv` 中标记 `no` 的行只有代码支持的可能性、没有首轮方法对应实例；综合阶段需明确是否只把它们作为评测设计局限，而不暗示样本方法已利用。

## W5 C 档论文

- `W5-013` 和 `W5-017`：CarLLaVA / SimLingo-BASE 的主动提前停路，对官方 LB2 DS 的收益及路线完成率损失该如何共同呈现？重复提交方差见 CarLLaVA PDF p.5。
- `W5-023`–`W5-025`：DriveFuture 的 future latent 使用训练期 GT、测试期预测值；与直接读取测试期未来真值的机制应如何分界？GTRS-Dense scorer 对 navhard 55.5 的独立贡献未量化。
- `W5-026`–`W5-028`：Senna 带 ego status 的 0.22 L2 与不带 ego status、不同碰撞实现的比较如何处理？
- `W5-032`–`W5-034`：LVLDrive 的 LiDAR 输入和自定义碰撞计算与 camera-only nuScenes 排名是否属于同协议？
- `W5-025`、`W5-037`：DriveFuture 与 NTR 的 trajectory scorer 各自是否以榜单总分或子分数训练？两篇论文未交代足以直接归入榜单分数代理的细节。
- `W5-040`：Poutine 的 GRPO 使用 416 个 preference-labeled validation scenarios，对正式 test 成绩的解释边界是什么？
- `W5-041`：Poutine-Base 验证集 no-CoT 优于 CoT，但最终推理段仍描述低温 CoT；具体提交配置如何对应？
- `W5-042`–`W5-045`：DrivoR 论文称 NAVSIM-v1 模型零样本用于 HUGSIM；NAVSIM 的预训练、压缩与评分消融能否解释 HUGSIM 分数？确切 checkpoint 文件和适配入口未公开。
