# 阶段 2 codebook 横向审阅建议

审阅范围：`out/findings/` 的 4 份 JSONL（31 条、19 个阶段 1 类别）及 `out/audits/` 的全部审计页，包括被排除的 CarLLaVA、SimLingo-BASE、Senna。仅提出重编码建议；本页不改原始发现、taxonomy 或 matrix。这里的“代码机制成立”和“已证明它提高已报分数”须分开判断。`impact_evidence=none` 可保留，但不能在报告中转述成已量化贡献。

## 主要合并与拆分

| 阶段 1 类别 | 建议的最终处理 | 对应 ID 与判定边界 |
| --- | --- | --- |
| `metric_proxy_optimization` | **拆成** `metric_proxy_candidate_selection` 与 `metric_proxy_test_search`。前者指训练代理评分器并从已有候选 argmax；后者指以代理评分为目标反复生成/优化新候选。 | `NAV1-DVM0-001`、`NAV1-RAP-001`、`NAV1-DRIVOR-002`、`NAV2-DRIVOR-002` 属选择；`NAV1-TOAD-001`、`NAV2-TOAD-001` 属 CEM 搜索。TOAD 的 54.6→56.3 是搜索的增量，不能概括成训练评分器的增量。 |
| `metric_reward_training` | 将 `REAL-WOD-002` 移入 `metric_proxy_candidate_selection`：RAP 用 RFS 作**评分头监督标签**并在推理时选轨；`REAL-WOD-001` 独留并改名 `metric_reward_finetuning`：DriveMA 用官方 RFS 作 RL 奖励。 | 两者都接触榜单函数，但一个训练排序头、一个改生成策略；与 NAVSIM 的 PDMS 评分头应跨榜单同码。 |
| `stuck_recovery_override`、`infraction_rule_override` | 可合并为 `manual_control_override`，在定义/正例区分“卡住后蠕行”和“停车牌制动”，报告再按子机制计数。若希望保留规则目标差异，也可保留两类，但须互斥：静止计时器只归前者，具体违规场景检测只归后者。 | `B2D-TFV6-002/003`、`B2D-BLUE-001`、`LB2-TFPP-002/003` 都是在模型控制后由手写规则覆盖油门/刹车；与异常回退 `HUG-WA-002` 的触发条件不同。 |
| `ego_kinematics_shortcut` | 建议拆成 `ego_only_open_loop_planning` 与 `ego_state_fusion`，或保留父类并在定义中强制标明是否仍看环境。 | `REAL-NUS-001` 是完全不读感知的 AD-MLP，部署不成立；`REAL-NUS-003/005` 是视觉规划器中融合 ego 状态，部署部分成立。一个二值“ego shortcut=yes”会抹掉关键差别。 |
| `benchmark_weight_tuning` | 可改名 `manual_scorer_reweighting`，判定依据为**推理公式相对评分头/官方指标的手工改权**，不预设作者以测试集调过。 | `NAV2-DRIVOR-001` 有论文披露 warmup 调权；`NAV2-GTRS-002` 只见手写细排系数，尚无证据这些系数用 navhard 调优。后一条保持中/低置信度，别写成已证实测试集调参。 |
| `test_time_candidate_selection` | 改成 `multi_checkpoint_candidate_selection`，与单模型内的候选重排区分。 | `REAL-WOD-003` 扫描 checkpoint 文件，合并所有候选并全局 argmax。论文称两 checkpoint NMS，代码没有距离 NMS；属于实现差异，非新的“候选选择”机制。 |
| `simulator_controller_patch` | 建议列为**评测协议背景**，或保留单独 `simulator_protocol_dependency`，不要并入模型后处理 hack 频次。 | `HUG-WA-001` 修补公共 HUGSIM 控制器，并对同表基线统一重测。代码能证明绝对分数依赖版本，不能证明 WA-JEPA 相对优势来自该修补。 |
| `inference_failure_brake_fallback` | 保持单独但作为**潜在污染路径**，报告和 matrix 不应表述为已报 0.4462 包含此增益。 | `HUG-WA-002` 有代码和聚合器说明，却无本次 436 场景逐场景失败次数。异常回退是否真的进入报告分数为未知。 |
| `answer_conditioned_rationale` | 建议不计入已确认 hack 矩阵；保留在 AutoVLA 审计页的训练标签解释可信度备注。若保留类别，必须写清“训练阶段答案提示”，绝不与测试阶段泄漏合并。 | `REAL-WOD-004` 的教师 CoT 看见 GT 动作，但学生测试时不看；行为克隆训练本来也使用未来动作作为监督，缺少“该提示额外提高 RFS”的对照。与 `REAL-NUS-004` 的测试时未来真值命令有本质区别。 |

其余类别暂可保留单独定义：`control_interface_selection`、`simulator_pose_access`、`metric_early_termination`、`benchmark_prompt_specification`、`benchmark_target_shaping`、`benchmark_split_adaptation`、`offline_candidate_library`、`metric_grid_alignment`、`future_label_conditioning`。`benchmark_target_shaping` 和 `metric_grid_alignment` 都是训练目标设计，但前者改变未来目标时域并有跨 v1/v2 反向消融，后者按 0.5 m 栅格改损失、没有单项消融；仅在报告中归为训练目标家族，暂不强行合并。

## 重编码时优先核查的证据边界

1. **官方配置无法由固定代码完全对应**：`NAV1-TOAD-001` 的 main commit 缺 README 要求的 `nav1` 分支评测入口；`NAV1-DVM0-001/002` 公开 Base 路径无法证明 Scale 94.1；`NAV2-GTRS-001/002` 的消融是旧指标协议，45.4 来自后续修复口径；`LB2-TFPP-001/002/003` 的 5.56 是 MAP，而公开评测脚本默认 SENSORS，`STOP_AFTER_METER` 代码默认关闭。matrix 若表示“该固定代码含机制”可编码 yes，但报告不可把它们直接量化为相应榜单 headline 的贡献。
2. **只有关联、缺触发或独立增益**：`HUG-WA-002` 不知是否在 0.4462 中触发；`NAV2-TOAD-002` 的最终 YAML 只能证实参数，warmup/navhard 交集与选择过程来自论文，不能证明交集增加了分数；`NAV1-DVM0-002` 的 prompt 进入模板，但无有/无提示对照；`REAL-WOD-004` 的解释教师见答案，无提示增益未隔离。建议这些条目只列待测/协议风险，不作为 5–10 条“最值得注意的已证实得分来源”。
3. **消融数字不能外推为本项净贡献**：`B2D-TFV6-002/003` 共享三项启发式联合约 95→94，不能各记 +1；`NAV1-DRIVOR-002` 的 +1.8 是总分头对六子分数头，不是取消整个评分器；`NAV2-DRIVOR-002` 的 +2.5 涉及不同评分管线，不是 54.6 的同权重消融；`REAL-WOD-001` 的 7.893→8.009 含 SFT→RL 阶段变化；`REAL-NUS-003/005` 的融合消融不能拆成速度或加速度单项；`NAV2-GTRS-001` 的 +1.1 属旧协议；`HUG-WA-001` 没有修补前后同模型数字。
4. **强证据宜保留并突出限制**：`LB2-TFPP-001` 明确利用长路线 DS 公式主动停止（论文与代码均在）；`B2D-SPARSE-001` 在 SENSORS agent 直接用 CARLA world 中 LiDAR actor 真值位姿定位；`REAL-NUS-004` 在 test pipeline 用待预测未来真值造命令；`REAL-NUS-001` 完全无感知输入仍在开环榜取得低 L2。前三条的官方提交/触发量化仍分别有上述配置或消融限制。
5. **零发现审计页仍是有效 no 证据**：HUGSIM LTF 固定直行命令更可能损分、UniAD 公开客户端六相机与论文复测四相机不一致，均无有利分数证据；CarLLaVA/SimLingo-BASE 是同一 Base 的旧新名称且无对应官方闭环入口，已排除。不要把这几页转为 yes，也不要将它们当作独立、可比的入选模型。

## 对 matrix 与报告的操作建议

- `yes` 表示固定源码中确有该机制、可合理联系到该审计单元；`no` 表示检查了相应入口且未见该机制；源码缺失、官方配置不能对应或效应方向不清时优先用 `NA`，并在审计页/报告解释。`none` 仅代表论文未给量化，不自动把 yes 变 no。
- 报告中的数字务必带分割与轨道：navtest PDMS、navhard 两阶段 EPDMS、CARLA MAP/SENSORS、nuScenes ST-P3 与各论文实现、WOD RFS、HUGSIM 同一 436 场景/控制器。共享 repo 跨榜单分别编码；同一 Base 的新旧名字不重复计数。
- 可把最明确的反事实实测列为：TFv6 同 checkpoint 两控制接口与逐项规则关断；SparseDriveV2 actor 真值 pose 对 GPS/IMU；TF++ 早停开/关并同时报 RC、IS、DS、normalized DS；WA-JEPA 排除异常场景后的 HD-Score；BEV-Planner++ 真值未来命令对可部署导航命令。
