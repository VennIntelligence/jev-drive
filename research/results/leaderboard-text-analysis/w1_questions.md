# W1 留给综合阶段的问题

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
