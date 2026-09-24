# W1 增益归因账本：开放编码与读表约定

本文件对应 [`w1_ablation_ledger.csv`](./w1_ablation_ledger.csv)。账本覆盖抽样清单中 31 个有可定位论文实验的独立方法标签；`PDM-Closed` 是 NAVSIM 论文与榜单中的 privileged 规则基线，没有独立方法论文消融，故无 W1 行。账本另保留 `TOAD`、`LTF/TransFuser` 这两个论文中的方法或基线名称，用于追踪组合方法的对照。表中一行是**一个配置对 × 一个指标 × 一个评测划分**；同一实验的 DS、SR 等会占多行，行数不能当作独立实验数。

## 字段、比较与缺失值

| 字段 | 约定 |
| --- | --- |
| `method` | 尽量用 `sampling.csv` 的方法名；论文在多个基模型上评测的 TOAD 用 `TOAD` 表示一般搜索实验，DrivoR 组合的专属行用抽样名。 |
| `board` | 七个主榜固定为 `bench2drive`, `carla_lb2`, `navsim_v1`, `navsim_v2`, `nuscenes`, `wod_e2e`, `hugsim`。`aux:` 前缀表示论文自己的其他评测协议，例如 `aux:action_tokenization`, `aux:town13`, `aux:drivex`；这些**不是第八个主榜**，不可并入七榜的 W2 跨榜相关计算。 |
| `split` / `metric` | 保留论文报告的划分及指标名或足以定位的标准缩写；原始分值的单位由指标名、配置或原表决定，不将百分比、比例与米混用。 |
| `component` | 表后列出的原子开放代码。它是「实验变化」的标签，不保证只有一个变量；多因素同变的行会在配置或 `notes` 标明。 |
| `baseline_config` / `variant_config` | 实际比较的两种配置。前后是记录顺序，不表示预设优劣。`expert_reference` 等行是对照，不等同于可单独归因的模块消融。 |
| `baseline_score` / `variant_score` / `delta` | 原表可读的数值直接转录；`delta = variant_score − baseline_score`，按原指标单位保留方向。碰撞率、L2 等越低越好的指标，其负差值可能是改善；账本不把方向统一翻转。 |
| `same_weights_or_retrained` | `same_weights`：固定基础 checkpoint，仅改推理输入、规则、阈值或搜索；若比较已训练 gate 与规则 gate，`notes` 必须说明推理栈并非逐参数相同。`test_time_updated`：从同一起始 checkpoint 在测试时更新参数；`retrained`：训练条件、模块或数据改变，另行训练，含冻结 backbone 后单独训练 gate；`different_policy`：与独立 expert 策略比较；`mixed_checkpoints`：配对含不同来源的 checkpoint，如 Official→Reproduce 或固定主干加外来已训练 gate；`unknown`：论文不足以判断。此字段依论文可确认的信息填写，不表示论文已披露全部随机性。 |
| `seeds_or_runs` / `n_samples` | 仅填**该论文明确写出的**种子、运行数或评测样本数，并尽量在值内写明 PDF 页；其余填 `N/A`。不能把榜单常见路线数自动套到另一实验、划分或论文。当前 `n_samples` 307 行有论文支持的数值，814 行为 `N/A`。 |
| `source` | 均以本地 PDF 文件名、Table/Figure 编号及 **PDF 实际页序** 定位；章节叙述可作为补充。PDF 页序不一定等于论文印刷页码。 |
| `notes` | 记录非单因素变化、协议差异、缺失值原因、只凭论文无法核代码等限定。默认文本表示该行没有额外摘录，不表示排除了未报告的混杂因素。 |
| `hack_finding_id` | 与第一轮 `findings.jsonl` 可直接对应时填原 id；否则 `N/A`。第一轮发现不自动证明本行增益的因果解释。 |

数值 `N/A` 的两种主要情况：15 行基线分数 `N/A`、变体有分数，其中 10 行是 NAVSIM 论文 TransFuser 默认模型按三个 seed 分列、不能任选一个数充当配对基线，5 行是 SparseOccVLA 原表以横线留空；另 13 行两端都是 `N/A`，属于图中趋势、训练曲线或无法可靠精确读数的实验。对应行的 `notes` 给出原因，`delta` 也为 `N/A`。Figure 4(a) 的 SparseOccVLA query 数与 TOAD Figure 4(b) 的候选数有直接标注，已转为数值行；不从未标注的曲线高度估造数据。

## 抽样名称与论文名称

| `sampling.csv` 名称 | W1 记录方式 | 追溯说明 |
| --- | --- | --- |
| `TOAD+DrivoR` | `TOAD+DrivoR` | TOAD 论文 Table 1 的 NAVSIM v1 DrivoR→DrivoR+TOAD 专属行；同论文对其他基模型的搜索实验记 `TOAD`。 |
| `DrivoR+TOAD` | `DrivoR+TOAD` | TOAD 论文 Table 2 的 NAVSIM v2 DrivoR→DrivoR+TOAD 行，以及 Table 4 的 HUGSIM 对照行；两榜名称顺序照抽样表保留。 |
| `LTF` | `LTF/TransFuser` | 抽样表的 LTF 为 TransFuser 系列基线；本地 `ltf.pdf` 是 NAVSIM 官方论文，其 Table 2 做 TransFuser 输入及任务消融。LTF 在 HUGSIM 作为 WA-JEPA 论文的比较基线，那个榜单分数不应与 NAVSIM 消融按同一 checkpoint 合并。 |
| `PDM-Closed` | 无 W1 行 | 抽样表本身标为 privileged 规则 baseline、仅参照。本地 NAVSIM 论文介绍它作参考规划器，但没有一篇独立 PDM-Closed 方法论文及可配对的 PDM-Closed 方法消融；其已发表成绩由 W2 收集。 |
| `DriveMA-4B` | `DriveMA-4B` | 论文文件名缩写为 `drivema.pdf`，方法名仍取抽样标签。 |

## 开放编码的分组边界

下表给出读者查找代码时的概念分组。最终记录使用更细的 `component` 原子代码；**相似字词不等于同一处理**，例如 `memory_injection`、`recurrent_action_memory` 与 `ram_inference_mode` 对应不同论文、不同操作。附录列出账本实际用到的全部代码及行数。

| 组 | 判断边界 | 代表代码 |
| --- | --- | --- |
| 训练数据与来源 | 更改真实、合成、专家或跨数据集训练样本的来源、数量、筛选与混合；实验同时改变训练预算时另注。 | `synthetic_training_data`, `expert_data_state_alignment`, `training_dataset_size`, `data_filtering`, `CoVLA_pretraining` |
| 训练目标与监督 | 损失、蒸馏、教师目标、偏好/RL、DAgger、语言或空间监督变化。 | `distillation_loss`, `direct_future_MSE`, `grpo_posttraining`, `offline_dagger_posttraining`, `language_CoT_training` |
| 结构与容量 | Backbone、解码器、融合、查询数、门控或世界模型容量。 | `backbone_scale`, `planning_fusion`, `global_query_count`, `gate_policy`, `world_model_query_size` |
| 输入与传感器 | 摄像头/LiDAR/radar、位姿、速度、历史、导航输入、传感器范围或损坏。 | `sensor_lidar`, `camera_fov`, `ego_state_navigation_input`, `input_image_frames`, `image_corruption` |
| 表征与输出 | 轨迹/动作 token、词表、目标速度/路点、occupancy 与 ray 等编码。 | `output_representation`, `trajectory_vocabulary`, `action_tokenization_method`, `token_c2f_alignment_sequence`, `ray_representation` |
| 记忆与时序 | 历史帧、RAM、检索、长期记忆、时间教师等；训练与推理条件分别记录。 | `recurrent_action_memory`, `memory_injection`, `retrieval_key`, `temporal_history` |
| 辅助任务与推理 | 感知/语言/规划 QA、辅助地图与 occupancy、CoT 与目标选择。 | `auxiliary_task`, `perception_QA`, `planning_oriented_QA_mix`, `test_time_CoT`, `semantic_prior_target_selection` |
| 测试时计算与后处理 | 不重新训练主体权重时的候选搜索、重评分、TTT、轨迹处理、阈值或规则。TTT 参数更新标 `test_time_updated`。 | `test_time_cem_search`, `ttt_gradient_steps`, `postprocessing`, `early_termination_distance` |
| 控制与协议敏感性 | 扰动、oracle 上界、对照 expert、人工权重、不同 checkpoint 或联合改动。它们提供对照证据，不强行解释为单成分收益。 | `oracle_best_of_N`, `expert_reference`, `manual_subscore_weights`, `ego_velocity_perturbation` |

### 辅助评测、归因限制

`aux:` 行保留论文中的对照实验，以免把其结果遗漏，但只在**相同辅助协议**的行内计算差值。若实验同时改变了多个因素（如数据量与训练时长、backbone 与 RAM、记忆库规模与 TTT），`component` 表示该配置束，并在 `notes` 或配置文字指出；不能把 `delta` 解释成其中任一单独因素的因果贡献。不同论文即使都写 `PDMS`、`DS` 或碰撞率，未核实 split、指标实现和 checkpoint 前不能直接相加或合并。

### 实际使用的原子代码

下面按字母排序列出账本中全部 `component` 值。括号为**指标行数**，并非独立实验数。

| `component` | 指标行数 |
| --- | ---: |
| `3d_geometry_distillation` | 6 |
| `action_centric_pretraining` | 3 |
| `action_codebook_size` | 6 |
| `action_dreaming_training` | 1 |
| `action_token_representation` | 4 |
| `action_tokenization_method` | 19 |
| `adaptive_language_execution` | 2 |
| `aggressive_second_target` | 2 |
| `augmentation_and_target_refinement` | 3 |
| `auxiliary_task` | 2 |
| `backbone` | 2 |
| `backbone_scale` | 11 |
| `camera_fov` | 3 |
| `camera_frequency_alignment` | 2 |
| `camera_modality` | 3 |
| `camera_pose_injector` | 3 |
| `camera_view` | 1 |
| `candidate_count_M` | 1 |
| `CEM_iterations_K` | 1 |
| `cem_search_component` | 5 |
| `CoVLA_pretraining` | 2 |
| `cross_agent_synthetic_data_scale` | 1 |
| `cross_model_gate_transfer` | 4 |
| `cross_view_attention` | 6 |
| `dagger_expert_and_takeover_design` | 10 |
| `data_efficient_language_action_alignment` | 1 |
| `data_filtering` | 3 |
| `denoising_steps` | 8 |
| `diffusion_anchor_count` | 8 |
| `direct_future_MSE` | 7 |
| `distillation_loss` | 5 |
| `dynamic_proposals` | 3 |
| `early_termination` | 8 |
| `early_termination_distance` | 6 |
| `efficient_direct_action_mode` | 6 |
| `ego_state` | 2 |
| `ego_state_bev` | 9 |
| `ego_state_navigation_input` | 6 |
| `ego_state_planner` | 9 |
| `ego_velocity_perturbation` | 12 |
| `elite_count_E` | 1 |
| `ema_teacher_targets` | 6 |
| `ensemble` | 5 |
| `epipolar_attention` | 6 |
| `expert_data_state_alignment` | 2 |
| `expert_reference` | 1 |
| `explicit_ego_waypoint_history` | 16 |
| `factorized_vocabulary_size` | 3 |
| `feature_alignment` | 1 |
| `feature_alignment_direction` | 2 |
| `feature_normalization` | 4 |
| `finetuning_data_mix` | 3 |
| `future_horizon_tf` | 2 |
| `future_prediction` | 1 |
| `future_prediction_flow_matching` | 1 |
| `future_prediction_objective` | 1 |
| `future_prediction_regression` | 1 |
| `gate_activation_threshold` | 2 |
| `gate_dropout` | 2 |
| `gate_hidden_dimension` | 4 |
| `gate_policy` | 18 |
| `gate_training_data_size` | 1 |
| `global_query_count` | 12 |
| `grounded_reasoning_labels` | 1 |
| `grpo_group_size` | 1 |
| `grpo_posttraining` | 2 |
| `guidance_inflection_eo` | 2 |
| `hierarchical_policy` | 1 |
| `history_length` | 8 |
| `image_corruption` | 15 |
| `image_token_count` | 5 |
| `implicit_future_constraint` | 7 |
| `input_image_frames` | 15 |
| `input_or_pretraining` | 3 |
| `joint_world_action_model` | 1 |
| `kinematic_and_GT_guidance` | 7 |
| `language_action_consistency_reward` | 6 |
| `language_CoT_training` | 1 |
| `language_gate` | 10 |
| `language_training_mixture` | 6 |
| `latent_reconstruction` | 16 |
| `learned_trajectory_scoring` | 3 |
| `lidar_modality` | 3 |
| `lidar_range` | 6 |
| `manual_subscore_weights` | 1 |
| `map_auxiliary_task` | 2 |
| `map_pretraining` | 2 |
| `matched_language_gate` | 4 |
| `memory_injection` | 24 |
| `memory_robustness` | 32 |
| `memory_scale_and_ttt` | 1 |
| `memory_type_and_training_budget` | 36 |
| `meta_action_reward` | 6 |
| `model_scale` | 3 |
| `motion_module_design` | 12 |
| `multihead_gate` | 3 |
| `multimodal_fusion_and_spatial_training` | 3 |
| `navigation_conditioning` | 2 |
| `navigation_input_form` | 2 |
| `object_detection_range` | 3 |
| `occupancy_language_component` | 20 |
| `occupancy_module_design` | 12 |
| `occupancy_prediction_task` | 2 |
| `occupancy_query_count` | 1 |
| `occupancy_query_dimension` | 6 |
| `offline_dagger_posttraining` | 8 |
| `oracle_best_of_N` | 7 |
| `output_meta_action` | 3 |
| `output_representation` | 4 |
| `path_scene_attention` | 2 |
| `perception_QA` | 6 |
| `planning_decoder_gru` | 2 |
| `planning_fusion` | 24 |
| `planning_module_design` | 18 |
| `planning_oriented_QA_mix` | 6 |
| `postprocessing` | 3 |
| `postprocessing_and_scorer_choice` | 14 |
| `pretraining` | 3 |
| `pretraining_data_mix` | 2 |
| `privileged_localization` | 3 |
| `privileged_sign_detection` | 3 |
| `ram_and_backbone` | 18 |
| `ram_inference_mode` | 8 |
| `RAM_vs_longer_training` | 6 |
| `raster_background` | 1 |
| `raster_depth_decay` | 1 |
| `raster_face_style` | 1 |
| `ray_representation` | 6 |
| `reasoning_QA` | 6 |
| `recovery_perturbations` | 2 |
| `recurrent_action_memory` | 40 |
| `register_compression` | 2 |
| `register_count` | 4 |
| `register_initialization` | 1 |
| `reinforcement_fine_tuning` | 7 |
| `reinforcement_learning` | 3 |
| `retrieval_key` | 18 |
| `retrieval_trigger_threshold` | 3 |
| `route_target_points` | 2 |
| `scene_token_budget` | 8 |
| `score_representation_disentanglement` | 1 |
| `semantic_prior_target_selection` | 6 |
| `sensor_camera_fov` | 4 |
| `sensor_lidar` | 4 |
| `sensor_localization` | 3 |
| `sensor_radar` | 8 |
| `sensor_sign_detection` | 3 |
| `separate_decoders` | 1 |
| `soft_action_labels` | 2 |
| `speed_loss_weight` | 3 |
| `stage1_masking_strategy` | 4 |
| `static_vocabulary_size` | 4 |
| `structured_planning_interface` | 3 |
| `subscore_prediction` | 1 |
| `synthetic_training_data` | 2 |
| `target_points` | 3 |
| `task_coordination` | 4 |
| `temporal_history` | 3 |
| `temporal_memory_mechanism` | 20 |
| `temporal_teacher_sampling` | 12 |
| `test_time_cem_search` | 36 |
| `test_time_chain_of_thought` | 2 |
| `test_time_CoT` | 1 |
| `test_time_reasoning_mode` | 4 |
| `token_c2f_alignment_sequence` | 45 |
| `token_compression` | 2 |
| `token_interleaving` | 4 |
| `tracking` | 3 |
| `train_time_3d_geometry_distillation` | 12 |
| `training_budget_and_accumulation` | 2 |
| `training_data_scale` | 1 |
| `training_dataset_size` | 3 |
| `training_duration` | 18 |
| `trajectory_postprocessing` | 3 |
| `trajectory_query_count` | 5 |
| `trajectory_reconditioning` | 2 |
| `trajectory_tokenization` | 1 |
| `trajectory_vocabulary` | 6 |
| `ttt_gradient_steps` | 6 |
| `ttt_learning_rate` | 6 |
| `turn_level_credit_assignment` | 3 |
| `unified_qformer` | 3 |
| `upstream_perception_and_motion` | 2 |
| `vision_backbone` | 3 |
| `vision_encoder_pretraining` | 3 |
| `vision_finetuning` | 4 |
| `vision_pretraining` | 6 |
| `vlm_backbone` | 2 |
| `vocabulary_dropout` | 3 |
| `Waymo_pretraining_and_CoT` | 1 |
| `WOD_pretraining` | 1 |
| `world_model_query_size` | 2 |
| `zero_initialized_gate` | 3 |
