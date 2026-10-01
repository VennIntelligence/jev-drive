# 阶段 2 重编码裁决

- `REAL-WOD-004`：不纳入最终 `findings.jsonl` 或 matrix。代码只证明 AutoVLA 的训练解释教师见该训练样本未来 GT 行动；这是已披露的训练监督方式，缺少独立分数增益与测试时未来输入证据。保留在原代理 JSONL 和审计页作为审计轨迹。
- `HUG-WA-001`：保留为 `simulator_protocol_dependency`，只编码共享控制器版本依赖；论文已说明同表方法重测，不把它列为 WA-JEPA 相对优势。
- `HUG-WA-002`：保留为 `inference_failure_brake_fallback`，编码潜在路径；是否在 0.4462 主分数中触发未知。
- `REAL-NUS-006`：由阶段 2 回扫新增；AD-MLP 的未来真值高层命令证据来自论文与数据集函数，预制推理 pkl 内容缺失，故中置信度。
- `NAV2-GTRS-003`：复核矩阵 no 时补出 GTRS-Aug 的预计算 PDM 子分数监督和推理评分头择轨，归 `metric_proxy_candidate_selection`；与手工重加权 `NAV2-GTRS-002` 并列，不能将旧协议消融外推到修复后的 45.4。
- `NAV1-TOAD-002`、`NAV2-TOAD-003`：首轮独立复核 26 格中发现两处原矩阵 false no，分别为 TOAD v1 基座的长目标训练命令、TOAD v2 评测命令显式改权。已补 findings 和两页审计，重建矩阵后重新随机抽样。TOAD v1 训练示例缺 `drivoR_speed_up` Hydra 配置，故只记中置信度。
- TOAD v1/v2 的 `metric_proxy_candidate_selection=no` 经交叉检查与最终独立复核讨论后保留：初始 DrivoR argmax/失败回退是 CEM 测试搜索实现环节，正式路径会用代理生成新候选；codebook 明确两类按最终轨迹来源判定，不重复计这一环节。独立的手调评分权重与基座训练目标仍分别编码。
- `NAV2-DRIVOR-003`：报告交叉核查补出 DrivoR v2 论文的 warmup-two-stage 改权验证及与 navhard 交叠事实；从 `benchmark_split_adaptation=no` 改 yes，影响无单项数字。因最终矩阵变化再抽样独立复核。
