# 榜单审计进度

| 榜单 | 完成审计单元 | 新类别 | 耗时 | 负责人 |
| --- | ---: | --- | --- | --- |
| navsim_v1 | 4 | `metric_proxy_optimization`, `benchmark_prompt_specification`, `benchmark_target_shaping` | 约 0.3 小时 | agent_nav |
| navsim_v2 | 3 | `benchmark_split_adaptation`, `benchmark_weight_tuning`, `offline_candidate_library` | 约 0.2 小时（复用 TOAD/DrivoR） | agent_nav |
| bench2drive | 3 | `control_interface_selection`, `stuck_recovery_override`, `infraction_rule_override`, `simulator_pose_access` | 约 0.5 小时 | agent_carla |
| carla_lb2 | 1（另 2 条排除核验） | `metric_early_termination` | 约 0.3 小时 | agent_carla |
| nuscenes | 3 | `ego_kinematics_shortcut`, `metric_grid_alignment`, `future_label_conditioning` | 约 1 小时 | agent_real |
| wod_e2e | 3 | `metric_reward_training`, `test_time_candidate_selection`, `answer_conditioned_rationale` | 约 1 小时 | agent_real |
| hugsim | 3 | `simulator_protocol_dependency`, `inference_failure_brake_fallback` | 约 0.5 小时 | root |

- 阶段 0 复核修正：Senna、HUGSIM DrivoR 缺对应榜单实现降为 C，分别以 BEV-Planner、LTF 递补；BLUE、AutoVLA 降为 B；SparseDriveV2 切到 bench2drive 分支；TF++ 5.56 DS 核为 MAP，公开脚本默认 SENSORS。
- CARLA-LB2 复核：CarLLaVA/SimLingo-BASE 固定公开提交无对应闭环入口，降 C 不入选；该榜仅 TF++ 满足入选条件，按规则如实报告不足 3 项。AD-MLP 的 SOTA2 0.35 与纠错后论文/README 0.29 分开记录。
- 阶段1 HUGSIM：WA-JEPA、LTF、UniAD 三页完成；2 条 WA-JEPA 发现；DrivoR 无 HUGSIM 客户端改列抽样排除；静态核查无模型运行。
- 阶段2：统一 19 类 codebook；20 个单元回扫，补 AD-MLP 未来命令及 GTRS 学得评分头发现、修正 TOAD v1 专用分支、排除 AutoVLA 训练解释候选；合并 32 条最终发现并生成 380 格矩阵（yes 30/no 193/NA 157）。静态验收前四项通过；31 个固定代码链接 HTTP 检查无警告。
- 阶段3 首轮：独立新代理按 seed 20260924 抽查 6 个 yes、20 个 no，24/26 一致（92.3%）；两处 false no 已补 `NAV1-TOAD-002`、`NAV2-TOAD-003`，正式矩阵重建为 34 条发现、yes 32/no 191/NA 157。原始裁定见 `out/sources/review_round1.*`；需据最终矩阵再抽样复核。
- 阶段3 第二轮：另一个无上下文新代理在 34 条发现版本抽查 7 个 yes、20 个 no，27/27 同意；报告交叉核查另补 DrivoR v2 交叠 warmup 验证集 `NAV2-DRIVOR-003`，正式矩阵现为 35 条发现、yes 33/no 190/NA 157，需重抽最终矩阵。
- 阶段3 最终验收：第三位无上下文新代理按 seed 20260924 抽查最终矩阵 7 个 yes、19 个 no，25/26 一致（96.2%）；唯一 `wod_e2e__rap:future_label_conditioning` 原 no、复核 NA，主审据 Waymo 官方测试集扣留未来驾驶记录的协议说明保留 no，分歧和非标准缓存风险已入报告。
- 阶段4：七榜总报告、10 条重点发现、7 组候选实测已写；`python3 out/sources/validate.py --check-http` 五项全通过，33 个唯一固定链接 HTTP 无警告，原样输出见 `out/validation.txt` 和 `out/report.md` 末尾。未运行模型、仿真器或评测。
