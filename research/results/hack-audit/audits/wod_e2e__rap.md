# WOD-E2E / RAP-DINO

- 仓库：[vita-epfl/RAP，固定 commit 5fd8630](https://github.com/vita-epfl/RAP/tree/5fd8630ae54442dd41827de4a9afe1690e3f02bb)。
- 论文：[RAP](../../papers/rap.pdf)，核对 §4 Waymo、Table 3、附录 A.4。
- 读过：根 README 的 Waymo Fine-tuning 与 Leaderboard Submission、`navsim/planning/script/run_waymo_dataset_caching.py`、`run_waymo_submission.py`、`navsim/planning/training/dataset.py` 的 Waymo 分支、`navsim/agents/rap_dino/rap_agent.py`、`rap_features.py`、`rap_model.py`、`navsim/common/waymo_utils.py`。
- 未读：RAP 的 NAVSIM/B2D 全部训练流程、上游 nuPlan 仿真器及 DINO 内部；未运行代码或评测。

## 整体印象

Waymo 专用路径并非普通 NAVSIM 的 PDMS 提交路径。Waymo 训练关闭 `pdm_scorer` 并用 RFS 给 proposal 评分头提供目标；val 分割携带人工偏好轨迹及分数。最终测试提交扫描同目录全部 checkpoint，拼接多模型候选后用最大预测 score 选一条。论文和 README 披露了 RFS 微调和双 checkpoint ensemble，但论文所称 NMS 与脚本实际 argmax 不同。

## 发现

- **REAL-WOD-002**，`metric_proxy_candidate_selection`：以 RFS 训练候选评分头。
- **REAL-WOD-003**，`multi_checkpoint_candidate_selection`：测试时在多个 checkpoint 的候选间按 score 选轨迹；公开实现未做距离 NMS。

完整记录及固定 commit 行号见 [agent_real.jsonl](../findings/agent_real.jsonl)。

## 局限

代码默认从检查点目录的 `*.ckpt` 扫描，README 和论文称双模型，但仓库没有固定该目录的实物清单；不能确认公开提交确切使用哪两个 seed。多模型对 RFS 的单独增益未消融，属于候选实测。Ego 历史和 route 是 Waymo 任务允许的输入，本审计不因其存在而单独判 hack。

最终独立复核指出通用缓存构建函数把 `future_states` 拼入位置序列求输入朝向，若测试字段非空则有未来影响；仓库未附实际 test TFRecord/预制缓存。Waymo 官方 E2E 数据说明测试集扣留未来驾驶记录，且提交脚本读取 `test` 缓存，故正式协议下 `future_label_conditioning` 保留 `no`；非标准含未来缓存要重判。复核员因不能直接查看缓存判 `NA`，该分歧已在总报告逐项记录。
