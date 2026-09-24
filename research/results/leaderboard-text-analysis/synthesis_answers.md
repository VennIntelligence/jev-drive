# 对 `questions_for_synthesis.md` 45 项的逐条回应

状态: 综合稿（2026-09-24）。这是 [leaderboard-vs-ability.md](../../leaderboard-vs-ability.md) 的附录：第二轮留下的 45 个问题（W1 12、W2 8、W3 8、W4 9、W5 8 组）逐条给判断、理由和引用；判不了的写明缺什么。
所有数字来自 `w1_ablation_ledger.csv`、`w2_cross_board.csv`、`w3_issues.csv`、`w5_paper_only.csv`、`w4/*.md` 和第一轮 `findings.jsonl`，没有运行任何模型或评测。**结论**是表能直接支持的；**推测**是我们外推的，并给出验证方法。
引用格式：账本行写 `method / component / board / split / metric / baseline→variant (Δ) / source`；issue 写 `repo#编号`；论文披露写 `W5-编号`；第一轮机制写 finding id（如 `NAV2-TOAD-001`）。
证据等级（W1 部分）：**A** 同权重且有 seed 或样本数；**B** retrained 但多 seed；**C** 单次；**D** 混 checkpoint、图读数或多因素同变。

## W1 增益归因（12 问）

### Q1 估计的 expert 分数如何使用？
判断：只作背景参照，不进任何差值。理由：`TFv6 / expert_reference / bench2drive / DS / 96.8→95.2 (−1.6) / tfv6.pdf T5 p.7` 的权重字段是 `different_policy`，notes 明写 96.8 是按另一 leaderboard 成绩「异常路线记零」估算，未跑 B2D。Table 1 的 84.94（TFv5 on LEAD data，3 seeds）是实测，不受影响。建议：把该行从任何「模型 vs 专家差距」的计算中剔除；如需专家上界，需要在 B2D 220 routes 上实跑 LEAD。

### Q2 同一搜索后处理能否跨基模型、跨榜归因？
判断：不能；账本证据支持把增益归给「DrivoR 评分器定义的 EPDMS 代理」而非搜索本身，且该代理在闭环上方向不定。理由分三层：
(a) 跨基模型：v2 上五个基模型 +TOAD 后收敛到 49.0–51.7（iPad +15.1 … ZTRS +1.1），增益 = 常数 − 基模型分；v1 上从 +2.1 递减到 +0.1。
(b) 搜索 vs 评分器：`TOAD / postprocessing_and_scorer_choice / navhard / EPDMS / iPad 34.7→45.6 (+10.9)` 仅重排；换 GTRS 评分器搜索 →23.9 (−10.8)；Hydra-MDP + SparseDriveV2 重排 →9.5 (−31.4)。搜索放大评分器的偏差，不产生独立价值。
(c) 跨榜：HUGSIM 四源 HDS −1.4…+10.2，RC 三降一升（1.7 节）。
缺什么：同一评分器下 CEM 搜索 vs 简单 top-k 重排的 HUGSIM 对照；DrivoR 评分器在非 NAVSIM 数据上的 EPDMS 相关性。

### Q3 早停分数与驾驶完成度如何权衡？
判断：能判。早停是纯 metric-specific，唯一有 RC 的对照显示归一化分数下降。理由：`TF++ / early_termination / aux:town13 / Town13 trained / DS 0.96→5.10 (+4.14), RC 68.53→11.47, normalized_DS 4.94→2.27 (−2.67)`，withheld town 同向（DS +2.65，normDS −0.82），same_weights，3 evaluations。官方榜的 `CarLLaVA=SimLingo-BASE / early_termination_distance / carla_lb2 / DS 2100 m 6.87 vs 1300 m 3.93 (−2.94), 2400 m 6.35 (−0.52)` 说明作者把停车距离调到 DS 极值附近（2100 m），量程占比 45%。结合 W4 LB2-TFPP-001 的 DS = RC × IP：早停在 IP 端换取的乘数增益大于 RC 端损失，只要路线后半段违规密度高于前半段。建议：LB2 榜单内 CarLLaVA/SimLingo-BASE 的 DS 应同时列 RC，或按 TF++ 论文自己提出的 normalized DS 重排。缺什么：官方提交没有 RC 列。

### Q4 逐级消融是否足以分解联合增益？
判断：不足以；只能读第一级。理由：`LinkVLA / token_c2f_alignment_sequence / bench2drive / DS 85.07→89.57 (+4.50)→89.85 (+0.28)→91.01 (+1.16)`，SR 对应 +5.91 / −0.91 / +2.28。第二级 DS 微升、SR 下降，第三级两者都升；在没有 seed 的情况下 ±1 以内的差值不可解读（同榜 SimLingo-BASE 3 seeds 的 language_training_mixture 行波动就是 ±0.7）。次序问题：C2F 在没有 alignment 时可能没有可对齐的目标，所以 +0.28 不能解读为 C2F 的边际贡献。缺什么：至少 2×2 的 C2F × alignment 交叉，以及 seed。

### Q5 效率与闭环得分如何一起比较？
判断：应作多目标，不合成。理由：`FIVE-VLA / efficient_direct_action_mode / bench2drive / DS 88.49→88.49 (0.00), SR 73.03→73.03 (0.00)`，而 fps_T4 行另计；`test_time_reasoning_mode / DS FIVE-VLA CoT 88.74→efficient 90.95 (+2.21)` 显示在闭环里更快本身就是更高分（延迟即控制误差）。因此「效率」在闭环榜上不是独立目标，而是通过时延进入 DS。建议：报告 (DS, FPS) 对，并对 CoT 类方法注明推理时延是否被仿真器同步等待（若仿真等待模型，则 FPS 不影响 DS，+2.21 就要另找解释）。缺什么：FIVE-VLA 的评测是否同步。

### Q6 测试时训练属于同权重比较吗？
判断：不属于同权重，但也不是 retrained；应单列 `test_time_updated` 并标注数据访问条件。理由：`DriveVLA-M0 / memory_injection / navval / PDMS no memory 91.0→TTT full 92.4 (+1.4), TTT LoRA 92.3 (+1.3), offline LoRA 10 epochs 91.2 (+0.2)`：同一记忆库，离线训进去只有 +0.2，测试时更新 +1.3，说明增益来自「对当前测试样本邻域做梯度更新」而非记忆内容。子分数上增益全在 EP（87.7→89.6），NC/TTC 不动——TTT 让模型更敢开，这在非反应式 v1 回放中恰好有利。`memory_scale_and_ttt / navtest / 92.3→94.1 (+1.8)` 是记忆库扩到 10k + TTT 同变，D 级。缺什么：TTT 用的检索邻居是否包含 navtest 自身样本（数据访问边界）；TTT 在 v2 反应式协议上的 EP/NC 变化。

### Q7 多 seed 默认配置能否与单变体比较？
判断：不能给点估计，可以给方向。理由：LTF Table 2 默认 83.3/84.0/84.4（极差 1.1），变体：goal only 81.8、goal+velocity 82.3、60° 单相机 80.3、LiDAR 16 m 79.1 都低于三 seed 最小值 83.3，方向可信；240° 五相机 84.1、64/32/32/32 m 84.3、无 3D det 84.0 落在 seed 范围内，不可判。缺什么：变体的 seed 值，或论文明说变体对应哪个 seed。

### Q8 图上无精确数字的实验可给多大权重？
判断：仅定性，不进任何表。理由：账本 13 行两端 N/A（SparseOccVLA distillation_loss F4、RAP feature_alignment / cross_agent_synthetic_data_scale F5–6、Poutine grpo F5、AutoVLA training_data_scale F4 / grpo_group_size F5(b)、BLUE gate_training_data_size F7、TOAD CEM_iterations_K / elite_count_E F4(b–d)、DrivoR manual_subscore_weights F7）。有直接标注的（SparseOccVLA occupancy_query_count 300→750 CIDEr 0.732→0.778；TOAD candidate_count_M 38.5→48.5）已入数值行，可用。Poutine GRPO 有正文数字 test 7.91→7.99，用正文不用图。

### Q9 C 档论文自报实验的证据等级如何标注？
判断：需要分层，并且本文已按 A–D 分。C 档方法（LinkVLA、SteerVLA、FIVE-VLA、RoG-DAgger、Kyber-E2E、DriveFuture、OmniSpace、LVLDrive、NTR、Poutine）的所有主榜行 notes 都带「仅论文披露，无代码可核」，seed 全无，样本数只有 FIVE-VLA（220）、NTR/Poutine（479）有。它们的增益量级（FIVE-VLA +2.46、RoG-DAgger +3.75、LinkVLA +4.50）与有 seed 的 BLUE +5.51、TFv6 +3.1 在同一档，但后者能给 seed 方差。建议在综合表加一列「可复核性：代码 / 权重 / 评测脚本」，同幅度不同层。

### Q10 联合改变的数据和训练预算如何归因？
判断：三例中只有 FIVE-VLA 做了预算配平。
- FIVE-VLA `RAM_vs_longer_training`：No RAM 6ep 88.49、No RAM 10ep 87.83、No RAM 6ep + same-budget FT 88.70 → RAM 90.95 (+2.25)。配平后增益保留 90%，可归因 RAM。
- DrivoR `synthetic_training_data / navhard / EPDMS 48.3→52.3 (+4.0)→54.6 (+2.3)`：notes 无预算说明；65k→134k 边际递减，不能排除「多训一倍」。需要「真实数据重复采样到同 token 数」的对照。
- Poutine `CoVLA_pretraining`：WOD only 7.95 → CoVLA+WOD 8.12 (+0.17)，但 CoVLA only 7.74 说明 CoVLA 单独也能到 7.74；两者并集是否加了训练步数未知。GRPO +0.08 在 test，与验证集 8.12 不同划分。
缺什么：DrivoR 与 Poutine 的训练步数/样本数表。

### Q11 辅助协议的结果能推广到主榜吗？
判断：只有在同一论文内同一改动同时给了主榜与 aux 数字时，才能用 aux 解释主榜的**方向**，不能用 aux 数字替代。理由：
- TFv6 同一改动 B2D vs Longest6：LiDAR +3.1 vs +9.0；data alignment +1.38 vs +11.54；GRU/目标 token +2.32 vs +6.65；camera FoV −0.3 vs +3.0（RC 89→99）；backbone +0.5 vs +5.0（RC 99→91）。方向大多一致，量级差 3–8 倍，FoV 甚至反向。长路线放大所有效应，且视野在长路线才重要。
- AutoVLA `action_tokenization_method` 在 aux:action_tokenization 是 token 重建 ADE（K-disk vs FAST 0.0687 vs 0.1708 @256），主榜是 PDMS +12.91；重建误差解释了方向但不解释量级。
- TF++ aux:town13 早停行是唯一能解释 LB2 官方分数机制的对照，但 Town13 短路线 vs LB2 长路线不可数值对齐。
建议：aux 行只作「机制存在性」证据。

### Q12 开放代码应该在哪里合并？
判断：方法内保留细粒度；跨方法只在「同一权重字段 + 同一协议 + 同一操作」三者都满足时合并。账本里的具体情况：
- `memory_injection`（DriveVLA-M0：检索记忆 + TTT 更新参数）、`recurrent_action_memory`（FIVE-VLA：训练进去的循环记忆）、`ram_inference_mode`（FIVE-VLA 同权重改推理）：三者权重字段分别是 test_time_updated / retrained / same_weights，不可合并。
- `sensor_lidar`（TFv6 B2D retrained，+3.1）、`lidar_modality`（LVLDrive nuScenes retrained，collision −0.04）、`lidar_range`（LTF、TF++）：协议不同（闭环 vs 开环 L2），LVLDrive 那条还是 camera-only 榜单上的协议争议（W5-032），不可合并。
- 可合并的例子：`early_termination`（TF++）与 `early_termination_distance`（CarLLaVA/SimLingo）都是 same_weights、同一机制，只是一个有 RC 一个没有——可以合并成一组「早停」证据，并以 TF++ 的 RC 解释 CarLLaVA 的曲线。
- `ego_state_bev / ego_state_planner / ego_state_navigation_input / planning_fusion(Traj-Ego)`：都是 nuScenes、retrained、同一操作（把 ego 运动学喂给规划头），可以合并到「ego prior」组做 §6 的汇总，但 BEV-Planner++ 的 mixed_checkpoints 段要剔除。

---

## W2 跨榜一致性（8 问）

### Q1 UniAD-Base vs VAD 那句话怎么解释
**判断**：该句在 DS/SR 上不成立（45.81/16.36 vs 42.35/15.00），只在 Efficiency（129.21 vs 157.94）和 Comfortness（43.58 vs 46.01）上成立；两边差距都没有 CI。
**理由**：n=9 的 Spearman 矩阵里 L2 只与 Efficiency 同向（−0.69），与 DS/SR 无关；DS 与 Comfortness 反向（−0.75）。四个闭环量不是一件事：DS/SR 是 E 层过关率，Efficiency/Comfortness 是 R 层 continuation 的两个侧面。综合阶段应把四个量分开写，并说明 L2 与 Efficiency 同向不是「开环预测闭环」，而是两者都在测 continuation。
**引用**：csv `bench2drive | base-set open-loop 2s@2Hz` 与 `base-set closed-loop 220 routes` 的 UniAD-Base/VAD 行；`w2_correlations.md` 首四项；本文 3.1。

### Q2 v1→v2 排序变化来自哪里
**判断**：幅度（−32.5～−63.9 分）来自 navhard 场景 + 两阶段；顶部**排序**来自配方是否按 v2 重调；指标扩展本身贡献 ≤2 分；checkpoint 漂移不能排除但不需要它来解释。
**理由与分解**：
1. 指标：同一 navtest 上 PDMS→EPDMS：DriveFuture 90.7→89.9、SparseDriveV2 92.0→90.1、WA-JEPA 91.8→91.7（n=3）。
2. 场景 + 两阶段：同一方法 navtest EPDMS vs navhard EPDMS：DriveFuture 89.9 vs 55.5（−34.4）。20 对 v1→v2 分差 32.5–63.9。
3. 配方重调：TOAD 在 v2 上 +1.1～+15.1、在 v1 上 +0.1～+2.1（1.4 表）；DrivoR/TOAD v2 命令的 10/13/6/14/15 权重在与 navhard 交集的 warmup 上选（NAV2-DRIVOR-001/003、NAV2-TOAD-002/003）。yes 最多者秩不掉、未重调者（RAP-DINO）掉 9 秩。
4. checkpoint：唯一 same_checkpoint=yes 的 RAP-DINO 也掉 54.2 分——同权重下塌陷照样发生，所以塌陷不需要 checkpoint 差异来解释；但其余 19 对的**相对**排序里 checkpoint 混杂无法排除。
5. 训练目标：DrivoR 的 `long_trajectory_additional_poses=2`（NAV1-DRIVOR-001）navval PDMS +0.6 而 warmup EPDMS −1.6，是「v1 目标在 v2 上反向」的直接例子，但只有 1 个方法有此消融。
**引用**：csv navsim_v1/navsim_v2 三列；TOAD Tables 5/6（toad.pdf PDF p.14/15）；DrivoR Table 13/14；findings NAV1-DRIVOR-001、NAV2-DRIVOR-001/003、NAV2-TOAD-001/002/003。

### Q3 什么证据足以确认同一 checkpoint
**判断**：表内 NAVSIM v1/v2 只有 RAP-DINO 一对达到「自述同一模型直接评」；TOAD 的「公开 checkpoint、不更新权重」是**权重相同、推理策略不同**（v2 命令另带重调权重 + CEM），不是同一 policy。
**建议分级**（写入 csv 的 `same_checkpoint` 字段，替代 yes/no/unknown 三值）：
- A：同文件 hash + 同推理配置（表内 0 对）；
- B：同文件、不同推理配置（TOAD/DrivoR 的 v1 vs v2；RAP-DINO 自述属此级或 A 级）——可用于「感知相同、配方不同」的比较，这正是量配方项的理想条件；
- C：同训练、另行微调（RAP WOD、LTFv6 WOD、AutoVLA WOD、NTR）——不可配对；
- D：同名、重训/换骨干（RAP-ResNet on B2D、OmniSpace、SparseDriveV2 B2D 分支、SteerVLA、AD-MLP B2D 重训）——不同模型，只作索引。
**缺什么**：TOAD nav1 分支 README 指定了 `nav1_30epochs_with_134k_simscale_...pth`（NAV1-TOAD-002），v2 README 说复制 agent 并覆写权重（NAV2-DRIVOR-002）；要升到 A/B 级需要两边加载的文件 hash。
**引用**：csv `same_checkpoint` 列的 RAP-DINO、RAP-ResNet、DrivoR、iPad、DrivoR + TOAD 行；findings NAV1-TOAD-002、NAV2-DRIVOR-002。

### Q4 LB2 官方 MAP/SENSORS、B2D 220、Longest6 各代表什么可迁移行为
**判断**：应按（路线长度、终止/早停规则、track、expert 来源）四维分层，不能放进一列。
**理由**：第 6 节。要点：B2D DS≈SR（E 层过关率，短程）；Longest6 DS = RC（R 层耐力）× IP（E 层复利），ρ(RC,DS)=0.97；LB2 官方 DS 是早停后前 1.5 km 的 IP × 长度占比。MAP vs SENSORS：SimLingo-BASE 6.25/6.87、TF++ 5.56/5.18，两 track 差 0.4–0.6 分且方向相反（SimLingo SENSORS 高、TF++ MAP 高），在同一模型早停参数能造成的 0.96→5.10 区间内。expert：TFv6/TF++/SimLingo 都模仿 PDM-Lite 系 expert，PDM-Lite 97/73/36.3 是它们三个长度上的上界，TFv6 分别达到 98%/85%/10%。
**引用**：csv carla_lb2 四个协议列；bench2drive SimLingo-BASE/TF++/TFv6/PDM-Lite 行；findings LB2-TFPP-001。

### Q5 v2 的 navtest EPDMS*、修复后 navtest EPDMS、navhard EPDMS 如何进综合表
**判断**：三列分开，永不合并；NTR 90.9 单独标「修复状态未注明」。
**理由**：同一方法 navtest 与 navhard 差 34 分（DriveFuture 89.9/55.5），比修复前后差（表内无配对；`w2_correlations.md` 注 LTF 修复前 23.12 vs 修复后 25.1，约 2 分）大一个数量级。navtest EPDMS 与 navtest PDMS 几乎相同（Q2 第 1 点），综合表里可把 navtest EPDMS 视作 PDMS 的近似替代，但 navhard 是另一个榜。
**引用**：csv navsim_v2 三个 `protocol/split`；SparseDriveV2、DriveFuture、WA-JEPA、NTR、LTF、DrivoR 行。

### Q6 WOD RFS ↔ PDMS 五方法相关是否值得用
**判断**：只作已发表分数索引，不进一致性判断。
**理由**：3.3 的四点：区间 0.52 且下限 4.0；DriveMA 两规模；全部 Waymo 专训；2/5 含 RFS 直接优化（REAL-WOD-001/002/003）。
**引用**：csv wod_e2e 行；`w2_correlations.md` 末项；findings REAL-WOD-*。

### Q7 HUGSIM 的 436 场景协议与 TOAD 分数据源 HDS 如何对齐
**判断**：不能对齐；只能在 WA-JEPA Table 2 内部（同控制器、同 436 场景）比 NAVSIM 排序，且 n=4。
**理由**：同一 DrivoR 在 WA-JEPA 表是 0.3252，在 TOAD Table 4 是 17.7/39.7/35.1/44.2%（KITTI360/nuScenes/PandaSet/Waymo），量纲与场景集都不同，无逐场景输出无法换算。WA-JEPA 表用修正后 controller commit（HUG-WA-001）对全表重测，所以表内可比、跨表不可比。表内：HDS 排序 WA-JEPA 0.4462 > DrivoR 0.3252 > UniAD 0.3124 > LTF 0.231 > VAD 0.1393；PDMS 排序 DrivoR 94.6 > WA-JEPA 91.8 > LTF 83.8 > UniAD 83.4；n=4 ρ=+0.6（p=0.4）。值得记一笔的是 UniAD：PDMS 比 DrivoR 低 11.2 分，HDS 只低 0.013——HUGSIM 单帧无 EP，NAVSIM 的 EP 差距在这里不计。
**缺什么**：逐场景 HDS 与失败次数（HUG-WA-002 指出零轨迹回退仍计分，仓库未给 0.4462 的失败日志）。
**引用**：csv hugsim 两类协议行；findings HUG-WA-001/002；w4/hugsim.md。

### Q8 nuScenes L2 与 B2D 重训模型能否视作可比 checkpoint
**判断**：不能。五个方法 same_checkpoint 全为 no/unknown，三种 L2 实现，AD-MLP 的 B2D 分是基准作者重训。
**能说的**：存在性——nuScenes L2 0.28–0.40 与 B2D DS 18–91 共存；且机制上 L2 的可分辨区间被 ego-state 通道占满（REAL-NUS-001/003/005/006）。这足以说明 nuScenes L2 不是任何一层的证据，不需要可比 checkpoint 来证明。
**引用**：csv nuscenes 三个 `protocol/split`；AD-MLP、VAD-Base、AutoVLA、OmniSpace、SteerVLA 的 bench2drive 行；w4/nuscenes.md。

---


## W3 复现差距（8 问）

### Q1. TFv6 #90：LEAD expert 的 B2D 96.8 如何标注？
判断：**只作背景参照，不作同协议差值；每次出现都标"estimated, not measured on Bench2Drive"。**
理由：作者原话 "We did not evaluate LEAD expert on Bench2Drive since the porting would take a lots of wo…"，数字是从标准 leaderboard 结果对异常路线记零推出，且两协议 target point 采样不同（lead#90）。任何"TFv6 距 expert 仅 1.6"或"学生逼近教师"的叙述不成立。此说明不涉及 Table 1 的 TFv5 84.94（questions 文件注）。

### Q2. AD-MLP #4/#5：旧/新分数、pkl、代码如何对应？
判断：**三者目前对不上，只能分层标注；泄漏指控既未证实也未排除。**
可确认的：(a) 作者承认重审数据并修订训练流程，结论"与旧技术报告略有差异"（#4）；(b) 修订后 0.29 = 论文/README 完整模型，SOTA2 的 0.35 = 无高层命令消融（sampling.csv）；(c) 论文 Table 1 本身显示命令由未来 3 s 真值生成，无命令 0.35→有命令 0.29（`REAL-NUS-006`，medium confidence）；(d) 预处理沿 ST-P3，pkl 生成代码曾丢失后补 `generate_feng.py`（#5）；(e) 前两帧排除是沿用先例（#6）。
对应方式：旧报告数字 → "修订前，泄漏疑点未澄清，不可用"；0.29 → "修订后论文值，含未来派生命令（论文自述），pkl 字段链未闭合"；0.35 → "修订后无命令消融，是 SOTA2 采用的那条"。
缺什么：用 `generate_feng.py` 从原始 nuScenes 重新生成 pkl，逐字段对比发布 pkl 的速度/加速度是否等于未来 GT 差分——这是唯一能判定 #4 指控的实验，表里没有。

### Q3. NAVSIM #151：哪些快照对应修复前/后？
判断（按表可定）：
- 修复前（v2.0/2.1，`human_penalty_filter` bug）：GTRS 论文/README 42.1（sampling.csv "old protocol"）；#158（2025-09-15）那类"子分数全 1 总分 0"的用户结果。
- 修复后（v2.2，navhard 榜已更新）：GTRS 45.4（sampling.csv "revised"）；DrivoR Table 3 "修 bug 后 Nav2 48.3"（#31 的 paper_numbers）；DrivoR 54.6（#47 用户用官方 v2 重缓存后一致）。
- 不能定：DriveFuture 55.5、NTR、PDM-Closed 56.6、TOAD 56.3 的修复状态不在 W3/W5（W2-Q5 已列）。
规则建议：任何 EPDMS 若论文/仓库未写 devkit 版本且日期早于 2025-09-03（#151 日期），默认标"pre-fix"。
附加：#172（2025-11-11，修复后）仍报 navtest 若干 token 子分数全 1 而 EPDMS 0，维护者要求先复现基线，**未确认**；#200/#204 两个 devkit 疑点无回复。修复后协议也不能视为完全干净。

### Q4. DrivoR #54：HUGSIM 复现差距哪些可核、哪些不可判？
可由原文/issue 核实：场景集 = 旧 345 场景 commit（#34）；ckpt = `nav1_25epochs`（#54）；camera-only、默认相机（#7）；评测管线 = hyzhou404/NAVSIM 的 LTF（#7/#54）；未用 HUGSIM 序列训练（#34）；作者向 HUGSIM 官方提过舒适度边界和 heading 的 PR（#7）。
不可判定：适配代码（未发布）；HUGSIM devkit / 控制器版本及 PR 是否在论文运行中生效；HUGSIM 上用的子分数权重（W5-044 明说未说明）；用户 B 的管线为何差 19.6。
判断：**论文值被用户 A 复到 −0.9 HD，说明数字大概率真实；但同一权重下用户 B 差 19.6 说明结果由未发布的管线细节决定，"reproduced once, not reproducible from official code"。** 分层标注时 HUGSIM 那一行的可信度低于 DrivoR 的 NAVSIM 行。

### Q5. SimLingo #43：定制目录 + Town13 训练，如何与 LB2/B2D 其他分数分开？
判断：**B2D 85.94/87.4 与 LB2 6.25 必须分列，不能作同一模型的跨榜对；B2D 分数还应附注"用仓库 Bench2Drive 定制目录评测"。**
理由：(a) Table 2 模型训练含 Town13（#43/#72），而 LB2 官方协议 Town13 是验证城（W2-Q4；carla_garage#103 说 TF++ 训练用 Town12/13 LB2 路线切段是另一回事），所以 B2D 模型 ≠ LB2 提交模型；(b) B2D 与 LB2 计分软件不同（carla_garage#73/#108）；(c) carla_garage#95 作者自己说"精确路线不在训练集 ≠ 无空间重叠"，B2D 本就是 training-town benchmark。
**推测（表未写）**：#43 里定制目录使 DS 75.50→86.53 而 **SR 恒为 67.12%（=147/219）**，成功路线数一根未变，DS 却 +11——这与"目录改变了失败路线的 RC/penalty 计法"一致，而与"驾驶更好"不一致。同日 #44 发现 4000 tick 截断被注释，是最直接的候选原因：截断关闭后超时路线继续跑并积累 RC。若成立，SimLingo 系列（及沿用该目录的 BLUE、RoG-DAgger、FIVE-VLA 等 SimLingo 衍生方法）的 B2D 分数对官方协议偏高，幅度上限约 11 DS。缺：用同一 ckpt 在原版 Bench2Drive 目录（含截断）与定制目录各跑一次。

### Q6. AutoVLA #42/#43：哪些已发表数字能对应到可取得的权重与代码？
判断：

| 榜 | 已发表 | 可对应？ | 依据 |
|---|---|---|---|
| NAVSIM navtest | 89.11 | **部分**：发布 ckpt 是 NAVSIM RFT；一位用户 83.69，作者称另一位 ≈89；需 CoT 开/LoRA 关/metric cache | #48, #42 |
| nuScenes 附录 | 未转录 | **否**：需额外 nuScenes RFT，配方未发（reward/epoch/KL/采样数） | #42, #56（open） |
| Bench2Drive | 未在 W3 | **否**：训练与闭环评测有内部依赖 | #30, #43 |
| WOD-E2E | 7.5566 | **否**：提交/评测入口未公开；训练设置只口头给（≈10 epoch、大 LR、短 warmup） | sampling.csv 注；#47 |
另：#22 发布 config 曾指向 nuPlan `sensor_blobs/test`，作者称误贴并已改；#44/#45 标注四视角、推理三视角。AutoVLA 是 B 档，14 条 issue 里 7 partial 2 no。

### Q7. GTRS #4 / NAVSIM #140：pinv/solve 影响哪些对照？有固定环境吗？
判断：**影响范围不可从表判定；有一个作者自报的固定环境（DrivoR：NumPy 1.23.4），NAVSIM 维护者环境表中未写。**
理由：EP 是 PDMS 与 EPDMS 的加权子项，`pinv` 沿自 nuplan-devkit（#140）；单帧 EP `solve` 1.0 vs `pinv` ≈0.90（GTRS#4）；用户升级 NumPy 后问题消失（#140）；DrivoR#47 用户"NumPy 修后仅 EP 不匹配，重缓存后一致"——说明 EP 对 NumPy 版本和缓存都敏感。数据集级偏移量表中无。GTRS 用户看到 EP **高于**论文，方向是向上。
缺：在 NumPy 1.23.4 与新版各跑一次 navtest 基线（LTF/CV）取 EP 差；没有这个数字前，跨论文 PDMS 差 <1 的比较应视为环境噪声内。

### Q8. SparseDriveV2 #7/#13/#16：两榜训练目标不同怎么解读？
判断：**NAVSIM 分数 = 含 PDMS 代理选轨的分数（作者承认设计目的是最大化 PDMS）；B2D 89.15 = 无代理、但含 `B2D-SPARSE-001`（读 CARLA actor 位姿，high）且发布权重只复到 86.03。两个数字测的不是同一个东西，不能用于跨榜一致性样本。**
理由：#7 "Yes, this is to maximize the PDMS"；#13 "supervision is closely tied to the evaluation metric"，B2D 未用 PDM 监督，碰撞感知监督"差异不显著"（说明 B2D 上换监督目标不敏感，而 NAVSIM 上作者选择了指标本身）；#16 −3.1 无回复。
推测：作者在 B2D 上试碰撞感知监督"不显著"意味着 B2D 的 DS 由 R 层（路线完成、不卡住）而非 E 层碰撞监督主导——与 lead#89（YieldToEmergencyVehicle 4/4 失败仍 DS 70）方向一致。

## W4 榜单指标（9 问）

### Q1 同名榜单的版本分数能否横比？
**判断**：v1.0 与 v1.1 的 PDMS、v2.0 与 v2.2 的 EPDMS 都不能横比；但两处版本差的量级都远小于榜单间差，横比错误主要影响 1–2 分内的排序。
**理由**：`navsim_v1.md`：v1.0 的 `MultiMetricIndex` 含 DDC 并乘进 PDMS，v1.1 把 DDC 设零权、移出乘积门；官方 main 明确把 v1 / navtest 指向 v1.1，所以 W1/W2 里"navtest PDMS"默认按 v1.1 读，早于 v1.1 发布的数字需回查。`navsim_v2.md`：v2.0 的后续场景权重和全局加权平均与 v2.2 的 `exp(−d²/0.2)` 逐组乘积不同，且 v2.2 才修了 human penalty filter（navsim#151，2025-09-03）。W1/W2 的每个数字实际用哪个 devkit commit：**表里没有**，只有 GTRS（旧协议 42.1 / 修复后 45.4）、DrivoR（Table 13/14 分列）、SparseDriveV2 / DriveFuture / WA-JEPA（navtest 的 EPDMS* vs EPDMS）显式区分；NTR 90.9 未注明。修复前后的量级：LTF navhard 23.12 → 25.1（约 2 分）；对比 navtest→navhard 是 34 分。
**标注规则**：无版本信息且日期早于 2025-09-03 的 EPDMS 标 `pre-fix?`；无版本信息的 PDMS 标 `v1.1 assumed`；综合表里三列（navtest EPDMS*、navtest EPDMS、navhard EPDMS）永不合并。
**缺什么**：各论文提交时的 devkit commit；一个基线（LTF / CV）在 v1.0 / v1.1、v2.0 / v2.2 上各跑一遍的差值表。

### Q2 提前停车的净收益范围是多少？
**判断**：官方 LB2 提交分数里早停占多少**不能判定**，因为官方榜没有 RC 列；能给的是上下界。
**理由**：同模型 Town13 validation：无早停 DS 0.96 → 早停 5.10（×5.3），RC 68.5 → 11.5，normalized DS 4.94 → 2.27（`LB2-TFPP-001`；W1 `TF++ / early_termination / aux:town13`，same_weights，3 次评测）。官方榜上 CarLLaVA 同权重扫停止距离：1300 m 3.93 / 1800 m 4.49 / 2100 m 6.87 / 2400 m 6.35（W5-013），极差 2.94 占榜上最高分 6.87 的 43%。乘积公式 `DS = RC × IP`、未到终点仍记正分（`carla_lb2.md`）给出条件：只要后半程的预期罚分乘积损失大于 RC 增量，停车就加分。
**Bench2Drive 是否有同类激励**：公式相同（`bench2drive.md`），但 220 条路线每条只嵌一个 scenario 且很短，在 scenario 之前停车会丢掉大部分 RC，激励弱；第一轮 3 个 B2D 单元都没有早停机制（`matrix.csv` 全 `no`）。B2D 上真正的同类空缺是 `MIN_SPEED_INFRACTION = unused`：慢开不扣分，且 SR 汇总排除该项。**推测**：B2D 上"慢而不停"比"早停"更划算；验证：一个刻意限速的 agent 跑 220 路线，看 DS/SR 与超时数。
**缺什么**：LB2 官方路线长度与 RC；CarLLaVA p.5 的重复提交方差数值。

### Q3 规则后处理贡献与模型能力如何拆开？
**判断**：Bench2Drive 短路线上规则层的贡献约 1 DS（小），控制接口的贡献 14 DS（大）；两者都不是"网络学会了开车"，但性质不同。
**理由**：TFv6 README 三项启发式（creeping、stop sign、Kalman）联合开关约 95 → 94（`B2D-TFV6-002/003` 的 impact 字段）；我们第 31 条在四臂里把启发式**固定开启**、只换控制接口，测得 route + target speed 比 waypoint 高 14.3 DS [5.1, 25.9]，SR 高 29 个百分点。所以"规则 vs 模型"这个问题在 B2D 上的答案是：规则 ≈1，接口 ≈14，网络输出本身（waypoint）走出来是 80 分档。TF++ / BLUE 的停牌与蠕行规则（`LB2-TFPP-002/003`、`B2D-BLUE-001`）没有任何数字。
**是修补缺陷还是一般安全控制**：停牌规则替代的是一个**被计分的网络决策**（stop infraction ×0.8），属于替代 E 层的一格；蠕行是反 blocked 终止（R 层脱困），与计分公式的 60 s / 180 s 卡住阈值直接对应（`bench2drive.md`、`carla_lb2.md`），是 metric 邻接。**推测**：长路线上蠕行的贡献远大于 1 DS（Longest6 的 RC 主导 DS，ρ=0.97）。
**需要的配对**：同 checkpoint、同 220 路线、同 seed：{接口 A, B} × {规则全开, 全关, 逐项关}，同时记 blocked、stop infraction、行人/cut-in 反应（即 7.4 节实验 1）。

### Q4 NAVSIM 子分数代理是否过拟合榜单？
**判断**：W1 和 W2 **支持**"顶部差距主要是对代理的优化"，但**不支持**"这些方法的分数全是代理"。
支持的证据：(a) TOAD 在 v2 上让五个不同基座收敛到 49–52，增益 = 常数 − 基座分（W1 `TOAD / test_time_cem_search / navhard`）；(b) 只换评分器不搜索，iPad 34.7 → 45.6（+10.9），换 GTRS 评分器搜索 → 23.9，Hydra-MDP 用 SparseDriveV2 评分器 → 9.5（W1 `postprocessing_and_scorer_choice`）：分数跟着评分器走，不跟着基座走；(c) DrivoR 长目标 v1 +0.6 / v2 −1.6（`NAV1-DRIVOR-001`）；(d) AutoVLA RFT 把 TTC 打到 98.04（W1 `reinforcement_fine_tuning`）；(e) W2：PDMS ≥ 90 的 11 个方法 v1↔v2 ρ=0.21，PDM-Closed 从 #12 到 #1，唯一同 checkpoint 的 RAP-DINO 掉 54 分；(f) HUGSIM 上同一 TOAD 后处理 HDS −1.4…+10.2 方向不定。
不支持"全是代理"的证据：DrivoR 合成数据 +6.3 EPDMS、RAP 恢复扰动 +4.4、WA-JEPA 预训练 +6 都不是代理机制；DrivoR 的 v2 分数只被 TOAD 再抬 1.7。
**推测**：v2 榜前排里 10–15 分是"scorer 对不对准 v2 公式"；验证见 7.4 节实验 3。

### Q5 v2.2 伪闭环权重能否被候选轨迹有意影响？
**判断**：只能说"代码支持"，材料里没有任何方法利用它的证据，也没有配对分析；应作为协议局限呈现。
**理由**：`scene_aggregator.py` 用首段预测终点到每个预生成后续场景起点的距离 `exp(−d²/(2×0.1))` 加权，后续得分乘进总分（`navsim_v2.md`）。因此选一条终点靠近"容易的后续场景"起点的首段轨迹，会改变第二阶段的计分混合。第一轮 3 个 v2 单元的评分器都不显式读后续场景起点（`w4_attack_surface.csv` 该行 `observed_in_round1 = no`）。
**需要的分析**：固定同一首段场景，人工构造终点不同、真实驾驶质量相同的首段轨迹，报权重分布的熵和最终 EPDMS；若 EPDMS 随终点位置系统变化，就是协议漏洞。
**对我们的含义**：P5 的按对计分也有"观察窗口由谁定义"的同类问题（7.3 节）。

### Q6 nuScenes 开环 L2 的信息泄漏在各论文评测中实际发生了吗？
**判断**：BEV-Planner++ **确实发生**（测试管线代码），幅度未知；AD-MLP **论文自述发生**（−0.06 m），pkl 链未闭合；SparseOccVLA **不能判**；Senna / OmniSpace / LVLDrive 无代码。
**逐方法**：
| 方法 | 未来标签进入测试输入？ | 证据 | 幅度 |
|---|---|---|---|
| BEV-Planner++ | 是 | `LoadGTPlaner` 由随后 6 帧 GT 末端横向位移造命令，test pipeline 调用，规划头用它条件化并选模态（`REAL-NUS-004`，high） | 无消融 |
| AD-MLP | 论文说是 | 数据代码由未来 3 s 造命令；论文 Table 1 无命令 0.35 → 有命令 0.29（`REAL-NUS-006`，medium）；推理读预制 21 维 pkl，生成脚本曾丢失后补（AD-MLP#5） | −0.06 m / −0.04 pp |
| SparseOccVLA | 不能判 | 推理消费 `gt_planning_command`，仓库未给生成链（矩阵记 NA） | — |
| Senna / OmniSpace / LVLDrive | 不能判 | 无规划代码；Senna 带 ego status 0.22 vs 不带 0.59（W5-026） | — |
碰撞率实现：UniAD 的 0.5 m 栅格且 GT 已撞时刻免计（`nuscenes.md`）、LVLDrive 的"碰撞后持续记碰撞"严格口径（W5-034）、ST-P3 口径（AD-MLP、AutoVLA）三者不同，逐篇对齐**从材料里做不到**。
**结论**：nuScenes 榜前六名没有一条能从公开材料完整重算（W3 §4），所以"实际发生了多少"这个问题在现有材料下只能回答到上表。**缺**：AD-MLP 用 `generate_feng.py` 重生成 pkl 并逐字段比对；BEV-Planner++ 换可部署导航命令 / 乱序命令的消融。

### Q7 WOD RFS 的几何空缺是否映射到真实安全问题？
**判断**：材料里没有候选轨迹级数据，不能给出"分数高而路径不可驾驶"的实例；能给的是两个间接信号。
**代码事实**（`wod_e2e.md`）：只评 3 s 和 5 s 两个点；区外候选下限 4.0；两个时刻可分别匹配不同 rater；中间 18 个点不计。
**间接信号**：(a) DriveMA 的 RL（RFS 奖励）段 overall +0.085 而 spotlight 难例 −0.102（W1 `drivema T3`）：直接优化 RFS 抬的是均值不是难例；(b) 我们自己第 34 条：val 上 cv 基线 7.10、logged future 8.13，量程只有 1.0，而榜首五名挤在 0.5 里（7.56–8.08，W2）；4.0 下限把区分度压得很扁。
**官方提交约束**：材料里**没有**（W4 页只覆盖公开的 RFS 函数，不覆盖服务器校验）。
**对我们的含义**：我们的 WOD 提交应自查 18 个未计分点的运动学可行性（加速度、曲率上界），避免自己无意中做出"两点对、中间乱"的轨迹；这也是 P0/P1 用 ADE 并列汇报的理由（第 2、22 条）。

### Q8 HUGSIM 的 HD-Score 是否奖励失败制动或漏检对象？
**判断**：代码上**可能**，材料里**未证明发生**。
**理由**：单帧公式没有 EP，静止轨迹可保住 TTC 与 C，最后只乘 `min(max RC, 1)`（`hugsim.md`）；WA-JEPA 客户端异常时发全零轨迹让场景继续计分，其聚合器自己承认静止也能得分并另列可信均值（`HUG-WA-002`）。0.4462 的逐场景失败次数未公开。非 car 目标：只有 `obj_names == 'car'` 进框碰撞检查，其余是否在 `scene_xyz` 点云里**材料里查不到**。
**一个旁证**：UniAD 在 NAVSIM v1 PDMS 垫底（83.4，#28/28）、在 HUGSIM HDS 中游（0.3124，仅比 DrivoR 低 0.013），与"HUGSIM 不计每帧进度"一致（W2 §2.2）。
**缺**：436 场景逐场景 `n_failures`、`planner_stats.json`；同一失败场景制动回退 vs 记零的配对。

### Q9 "未观察到"的攻击面应如何在综合结论中呈现？
**判断**：作为**评测设计局限**列出，明确写"在 20 个审计单元里未见对应实例"，不暗示任何样本方法利用了它；它们对**我们自己的考试设计**比对评判已发表方法更有用。
`w4_attack_surface.csv` 里 `observed_in_round1 = no` 的 11 行：B2D 最低速度不扣分；NAVSIM v1 的 DDC 零权、低速 TTC 免罚；NAVSIM v2 的首段终点改后续权重、human 同项豁免、交叉口跳 LK；nuScenes 的 GT 已撞免计；WOD 的只拟合 3/5 s、区外 4.0 下限、跨 rater 匹配；HUGSIM 的非关键帧排除、非 car 不检。
呈现方式：主文第 2 节的"结构性盲区"列只写机制，不写方法名；7.3 节把其中与 P5 同构的四条（停着不动是安全、低速免罚、窗口由谁定义、非 car 不检）翻译成我们的防范项。

## W5 C 档论文（8 组）

### W5-013 / W5-017：早停的 DS 收益与 RC 损失怎么共同呈现？
判断：**呈现三元组 (停止距离, 官方 DS, 隐含 RC 上限)，并注明该距离是扫描后选的。** 表内数字：1300 m 3.93 / 2100 m 6.87 / 2400 m 6.35（CarLLaVA Table 2c）；对照 TF++ 同模型 Town13 val 无早停 0.96 → 早停 5.10（`LB2-TFPP-001`）。RC 损失需要路线长度，表中无 → 缺。重复提交方差（CarLLaVA p.5）未转录 → 缺；但 2100 vs 2400 m 差 0.52 DS 已接近 LB2 前三名间距（6.87/6.25/5.56）。这是纯 metric 增益：它把 R 层能力（能开多远）换成 DS。

### W5-023–025：DriveFuture 的 future latent 与测试时读真值如何分界？GTRS-Dense scorer 贡献？
分界：训练期用 GT 未来 latent、推理期换预测 latent 属于特权蒸馏，不是 `future_label_conditioning`（后者要求测试时读同一样本未来真值，taxonomy）。论文明说推理不用 GT（W5-023 notes）。附加论据（推测）：navhard 二阶段是合成场景，没有真实未来观测，至少第二阶段推理只能用预测 latent。但无代码，无法排除"缓存的 future latent 从 GT 生成"这类实现泄漏。
scorer：Table 4 排除 GTRS-Dense scorer，主表 55.5 含它 → 贡献未量化。因 GTRS scorer 已是 `metric_proxy_candidate_selection`（`NAV2-GTRS-003`），55.5 继承了一个未知大小的代理成分。缺：同一批 100 proposal 有/无 scorer 的 navhard 配对。

### W5-026–028：Senna 带 ego status 的 0.22 如何处理？
判断：**0.22 只能与其他带 ego status 的行比（AD-MLP 0.29 含未来派生命令，BEV-Planner++ 0.35 含 ego 状态直连+未来命令）；与无 ego status 方法比应用 0.59。** 0.22→0.59 的 0.37 m 差是 ego status 单项，大于榜上前六名总差距。碰撞率实现：Senna 的口径表中无 → 缺；LVLDrive 用严格口径（W5-034），两者碰撞率不可比。并且 E2E 代码未发（#10），两个数字都不能重算。

### W5-032–034：LVLDrive 的 LiDAR + 自定义碰撞是否同协议？
判断：**不是。** LiDAR 输入使其与 camera-only 排名分属不同传感器层（W5-032）；"improved collision rate"是更严口径（W5-034），碰撞率不可比但也不虚高；L2 0.29 不受碰撞口径影响，若要比 L2 还需知道是否用 ego status → 表中无 → 缺。建议单列"LiDAR+camera"层。

### W5-025 / W5-037：DriveFuture 与 NTR 的 scorer 是否以榜单分数训练？
DriveFuture：**间接是**——它用 GTRS-Dense scorer，而 GTRS scorer 以 PDM 子分数监督（`NAV2-GTRS-003`）。可以 medium confidence 升级为 `metric_proxy_candidate_selection`，前提是 DriveFuture 未重训 scorer（论文未写）。
NTR：**不能判定**——"DrivoR 式 trajectory-conditioned scorer"在 WOD 上没有 PDM 可学；若像 RAP 那样拟合 RFS（`REAL-WOD-002`）就是代理，若拟合 L2/模仿分数就不是。论文未写目标。缺：scorer 的损失定义一句话。

### W5-040：Poutine GRPO 用 416 个 validation scenarios，对 test 成绩的解释边界？
判断：**可解释为"在官方 RFS 上对 val 做 RL，test +0.08"；边界是 (a) 奖励即榜单指标（`metric_reward_finetuning`），(b) 数据是 val 非 test，无 test 泄漏证据，(c) +0.08 无重复评测支撑，小于同榜 DriveMA 的复现差 0.61，不应当作显著。** Fig. 5 的 63 例保留曲线是 val 内部保留，不是 test。RFS 是 rater 打分，直接对 rater 分做 RL 是 W5 中最贴近"metric-specific"的训练机制。

### W5-041：Poutine-Base val 上 no-CoT 8.12 > CoT 8.08，最终为何仍用 CoT？提交配置怎么对应？
判断：**表中只能确认最终推理段描述 CoT 温度 1e-6 + 轨迹 greedy + 去 intent；榜上 7.986 ≈ 文中后 RL 7.99，推断提交是后 RL + CoT 配置（推测）。CoT 的贡献是 −0.04（val），在噪声内；保留 CoT 的理由（格式奖励、可解释性）论文未在表中交代。** 缺：提交时的确切配置文件。

### W5-042–045：DrivoR NAVSIM-v1 模型零样本评 HUGSIM，NAVSIM 消融能解释 HUGSIM 分数吗？
判断：**不能。** HUGSIM 上没有任何消融（W5-042/043/045 notes 均写明），NAVSIM 上的预训练（4a）、压缩（4b）、评分（Table 6/7）消融不能分解 HUGSIM 的 35.7。可核的只有 W3 给出的运行条件（Q4）。机制上，PDM 子分数代理是在非反应式 NAVSIM 上学的进度/碰撞/可行驶区域预测，零样本迁移到闭环重建场景后它是否仍是"好轨迹选择器"完全未验证——用户 A 复到 −0.9 说明数字可再现，但不说明 NAVSIM 上的机制归因可迁移。checkpoint 文件名（`nav1_25epochs`）有了，适配入口仍无。

---
