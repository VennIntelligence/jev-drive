# 独立抽样验收

对 `out/review.csv` 的 26 个随机抽样格（seed `20260924`，原值 7 个 yes、19 个 no）逐格静态复核。依据仅为 `out/taxonomy.md`、初始抽样表、`out/repos.csv` 及相应固定本地仓库和论文；未读取其他审计结论，未运行模型、仿真器或评测，未下载权重或数据。

同意 **25/26（96.15%）**；原 yes 同意 **7/7**，原 no 同意 **18/19**。唯一分歧是 WOD-E2E RAP 的 `future_label_conditioning`：原 no，独立判 **NA**。

## 逐格判定

| # | 榜单 / 仓库 / 类别 | 原值→复核 | 源码及论文定位 | 判定理由 |
|---:|---|---|---|---|
| 1 | bench2drive / BLUE / `manual_control_override` | yes→**yes** | BLUE gate/evaluation/eval_blue_full.sh:23,237-254; team_code/agent_simlingo.py:861-879; config_simlingo.py:13-15; blue.pdf Table 1 | 正式闭环agent在车速低于0.1持续逾800帧后覆盖PID：油门至少0.4并取消制动；论文未单列分数效应。 |
| 2 | navsim_v1 / DriveVLA-M0 / `metric_proxy_candidate_selection` | yes→**yes** | DriveVLA-M0 action_decoder.py:179-214; episode_drive_loss.py:187-219; run_create_submission_pickle.py:79-87; drivevla_m0.pdf §3.1 | PDM子分数监督评分头；提交从64条在线候选按预测聚合分argmax。直接PDMS CPU入口另有字段名缺陷。 |
| 3 | navsim_v2 / DrivoR / `metric_proxy_candidate_selection` | yes→**yes** | DrivoR drivor_model.py:175-203; drivor_loss.py:187-230; agent_lightning_module.py:127-145; drivor.pdf §3.4 | 预测PDM子分数对固定候选排序，最终输出最高分轨迹；v2评测命令设定聚合权重。 |
| 4 | navsim_v2 / GTRS / `manual_scorer_reweighting` | yes→**yes** | GTRS docs/gtrs_inference.md:51-79,83-115; gtrs_dense/hydra_model.py:311-328; gtrs_aug/hydra_model.py:289-307; gtrs.pdf §3 | Navhard scorer流程以手写系数重排静态库与扩散候选，包含模仿项且偏离EPDMS默认聚合；不推断曾在测试集调参。 |
| 5 | nuscenes / AD-MLP / `future_label_conditioning` | yes→**yes** | AD-MLP stp3/datas/NuscenesData.py:505-532,626-629; planner.py:245-271; eval_weight.py:42-50; ad_mlp.pdf §2.1 | 论文明确用未来三秒横向位移生成转向标签，评测模型读取预制21维特征；预制pkl字段来源未公开。 |
| 6 | navsim_v2 / TOAD / `metric_proxy_test_search` | yes→**yes** | TOAD README.md:135-179; drivoR.yaml:43-55; drivor_model.py:225-249,352-393; agent_lightning_module.py:88-110; toad.pdf §3.2 | 正式v2评测启用5轮64样本CEM，以预测PDM子分数反复评估新轨迹并可输出收敛均值。 |
| 7 | nuscenes / AD-MLP / `metric_grid_alignment` | yes→**yes** | AD-MLP planner.py:47-50,64-78,234-243; train.py:68-71; stp3/planning_metrics.py:194-204; ad_mlp.pdf §2.2 | 训练L1对同一0.5m栅格内误差折半，对齐评测占用栅格尺度；论文明确写出动机。 |
| 8 | bench2drive / SparseDriveV2 / `inference_failure_brake_fallback` | no→**no** | SparseDriveV2 sparsedrive_b2d_agent.py:399-404,489-532; scenario_manager.py:185-199; sparsedrivev2.pdf Appendix A | 正式agent推理和PID后直接返回控制；异常由场景管理器停止路线，未见发送制动轨迹继续计分。 |
| 9 | navsim_v1 / RAP / `metric_reward_finetuning` | no→**no** | RAP navsim_v1 README.md:139-155; pad_agent.py:254-255,365-387,419-472,490-533; pad_model.py:155-167; rap.pdf §4 Training | PDMS计分器对detach候选产监督标签，策略用轨迹模仿损失，评分头用BCE；未以榜单计分函数作策略奖励微调。 |
| 10 | hugsim / UniAD_SIM / `simulator_pose_access` | no→**no** | UniAD_SIM tools/closeloop/e2e.py:231-249; dataparser.py:22-27,57-68,91-105; uniad.pdf §2.4 | FIFO协议info里的世界系ego位姿进入can_bus和l2g；固定agent未直接查询仿真器world/actor，未满足精确判据。 |
| 11 | navsim_v1 / DriveVLA-M0 / `offline_candidate_library` | no→**no** | DriveVLA-M0 action_decoder.py:61-67,154-169,198-213,228-272; README.md:44-54; drivevla_m0.pdf §3.1 / Algorithm 1 | 可学习query在线解码候选；离线memory存案例特征供检索适配，未见大型固定轨迹候选库进入正式选轨。 |
| 12 | nuscenes / SparseOccVLA / `metric_proxy_candidate_selection` | no→**no** | SparseOccVLA sparseoccvla.py:273-285,539-561,730-766; evaluation/eval_planning.py:44-86; sparseoccvla.pdf §3.3,§4.1 | 候选分类由最近未来GT anchor监督，L2/碰撞只在评测计算；无榜单子分数代理评分头。 |
| 13 | navsim_v1 / TOAD / `future_label_conditioning` | no→**no** | TOAD nav1 run_create_submission_pickle.py:61-83; drivor_features.py:38-65,199-262; dataclasses.py:369-394; toad.pdf §3.2 | 正式提交只取历史相机/ego和已有导航指令；未来轨迹仅供target监督。原始指令上游来源未公开。 |
| 14 | navsim_v2 / TOAD / `metric_proxy_candidate_selection` | no→**no** | TOAD v2 README.md:135-179; drivoR.yaml:43-46; drivor_model.py:209-249,352-393; toad.pdf §3.2 Eq.(2)-(4) | 固定候选argmax是CEM初始化及回退，最终可输出新轨迹；按codebook只归metric_proxy_test_search。 |
| 15 | hugsim / WA-JEPA / `simulator_pose_access` | no→**no** | WA-JEPA hugsim_client.py:182-194; hugsim_planner.py:197-200,242-268; docs/hugsim_closed_loop.md:68-80; wa_jepa.pdf Planning inference / Appendix A | FIFO info提供world-frame ego_box构造历史运动输入；固定agent未直接查询world/actor，未满足精确判据。 |
| 16 | navsim_v1 / DriveVLA-M0 / `future_label_conditioning` | no→**no** | DriveVLA-M0 drivevla_base_agent.py:358-387,473-490; drivevla_features.py:80-109,152-192; drivevla_m0.pdf §3.1 | 测试提示指令来自历史ego_status，future仅作训练target；原始scene_dict指令生成史未公开。 |
| 17 | navsim_v2 / TOAD / `metric_reward_finetuning` | no→**no** | TOAD v2 drivor_model.py:225-249,285-286; drivor_agent.py:202-253; drivor_loss.py:248-318; toad.pdf Abstract / §3.2 | CEM仅在eval且no_grad时优化轨迹，训练时榜单分数只监督评分头；无奖励式策略微调。 |
| 18 | hugsim / UniAD_SIM / `control_interface_selection` | no→**no** | UniAD_SIM planning_head.py:16-17,51-55,190-200; tools/closeloop/e2e.py:270-273; uniad.pdf §2.4 / supplementary Planning | 单模规划头只向闭环管道输出一条sdc_traj，未从同一网络多个控制接口中选择。 |
| 19 | wod_e2e / RAP / `future_label_conditioning` | no→**NA** | RAP WOD dataset.py:620-688,744-751,1069-1107; pad_features.py:46-57; run_waymo_submission.py:58-84; rap.pdf §4.1 | 测试缓存代码拼接future_states并平滑计算输入朝向；若test字段非空则泄入未来真值，本地无test TFRecord或预制缓存可判触发。 |
| 20 | hugsim / UniAD_SIM / `metric_proxy_candidate_selection` | no→**no** | UniAD_SIM base_e2e.py:440-451; planning_head.py:166-200,240-248; tools/closeloop/e2e.py:255-273; uniad.pdf §2.4 | 单轨迹规划头以ADE/碰撞监督，检测score_list不作为HUGSIM榜单代理重排规划候选。 |
| 21 | hugsim / NAVSIM / `inference_failure_brake_fallback` | no→**no** | LTF ltf_e2e.py:21-35,56-80; ltf.pdf §4.2 | 推理RuntimeError时发送None并退出；未发送制动或零轨迹让场景继续。 |
| 22 | hugsim / UniAD_SIM / `metric_proxy_test_search` | no→**no** | UniAD_SIM base_e2e.py:48,440-450; planning_head.py:190-238; collision_optimization.py:92-110; uniad.pdf §2.4 Eq.(9)-(11) | 确有测试时避撞优化，但目标是跟随参考轨迹和避开预测占用，不是HUGSIM榜单分数代理。 |
| 23 | bench2drive / BLUE / `simulator_pose_access` | no→**no** | BLUE configs/blue_eval.yaml:5-10; eval_blue_full.sh:244-254; agent_simlingo.py:382-405,468-494; scenario_logger.py:243-275; blue.pdf Bench2Drive setup | 正式决策以GNSS/IMU/车速经UKF估位；actor真位姿仅在日志器，未流入策略。 |
| 24 | bench2drive / SparseDriveV2 / `multi_checkpoint_candidate_selection` | no→**no** | SparseDriveV2 evaluate_mutil_gpu_wyh.py:25-27,47-65; sparsedrive_b2d_agent.py:73-76,112-116,496-503; sparsedrivev2.pdf §3.5 / Appendix A | 多GPU按路线并行但用同一weights_path；单模型内部选轨，未汇合多checkpoint候选。 |
| 25 | nuscenes / BEV-Planner / `benchmark_target_shaping` | no→**no** | BEV-Planner nuscenes_converter.py:478-507,579-582; naive_planner.py:301-311,331-351; bev_planner_plus_plus.py:184-187,233-245; bev_planner.pdf §3,§4.2 | 训练是标准三秒未来位移L1与有效mask/转向mode；论文改碰撞评测栅格，非训练目标按榜单塑形。 |
| 26 | navsim_v2 / DrivoR / `benchmark_target_shaping` | no→**no** | DrivoR README.md:92-120,174-208,285-315; drivoR.yaml:18; drivor_features.py:222-255; drivor_loss.py:252-271; drivor.pdf §4.2.4 Table 7 | v2正式10轮及SimScale训练命令均未启用长目标，默认-1；长目标只在v1命令/消融且warmup EPDMS下降。 |

## 分歧及限定条件

- **WOD-E2E RAP：`no→NA`。** 正式提交读取 `test` 缓存。缓存构建代码读取 `future_states`，拼在历史轨迹后用中心差分及 Savitzky–Golay 平滑生成输入朝向；若 test TFRecord 中未来字段非空，就会让同样本未来影响历史末帧朝向。固定材料没有真实 test TFRecord 或预制缓存，不能确认该字段是否非空。确认非空可判 yes，确认空可判 no；`intent` 指令本身未见未来派生。
- **HUGSIM 位姿两格维持 `no`。** UniAD 的 FIFO `info[ego_pos/ego_rot]` 进入 `can_bus` 和 `l2g`，WA-JEPA 的 FIFO `info[ego_box]` 转成历史 ego 轨迹。这是接收仿真协议提供的世界系位姿。现行 `simulator_pose_access` 明限闭环 agent **直接查询仿真器 world/actor** 的真实位姿或变换；两个固定方法仓只有消息接收与字段消费，未见此 API 查询，协议字段生成端也不在固定仓。若扩大类别至经协议接收真实位姿，两格应重编。
- **DriveVLA-M0 的入口缺陷。** `run_create_submission_pickle.py:82-87` 和多 GPU `AgentLightningModule.predict_step_drivor:161-168` 正确读取评分头输出 `trajectory`，但 `drivevla_base_agent.py:490` 的 `compute_trajectory` 读取未定义的 `pred_traj`。因此 `eval.sh` 所指直接 PDMS CPU 入口不能视为可运行证据；候选评分选轨在提交和多 GPU 路径仍可定位。
- **AD-MLP 未来命令为中等置信度。** 论文说明命令由同样本未来三秒位移定义，随仓 ST-P3 数据类也这样构造；实际评测模型读预制 `data_nuscene.pkl` 21 维特征，但该 pkl 的制作脚本与字段来源未公开。
- **TOAD 和 GTRS 路径边界。** TOAD 固定候选 argmax 及回退属于正式 CEM 搜索流程，只记 `metric_proxy_test_search`；GTRS 手写细排权重在发布的 Navhard 单模型推理路径可见，固定材料不足以还原完整榜单 ensemble 的全部私有权重组合。
- **静态证据范围。** `no` 表示公开关键路径未见适用机制，不能证明私有运行和未公开权重历史；`yes` 表示机制存在且接入相关路径，不能单独量化分数收益、触发频率或违规性。
