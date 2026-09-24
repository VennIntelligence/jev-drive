# nuScenes / Senna：排除记录

- 仓库：[hustvl/Senna，固定 commit 31a3a23](https://github.com/hustvl/Senna/tree/31a3a2336e9494c254b612665fe55e95f67e9cae)。
- 论文：[Senna](../../papers/senna.pdf)，重点核对 §III-C、Table II 和附录。
- 读过：仓库根目录 README、`data_tools/senna_nusc_data_converter.py`、`data_tools/senna_qa_utils.py`、`eval_tools/senna_plan_cmd_eval_multi_img.py`，以及训练脚本清单。
- 未读：`llava/` 和 `llava_next/` 下与 nuScenes 轨迹规划无关的上游通用模块；没有下载模型权重、运行代码。

## 结论

论文 Table II 报告 Senna 在 nuScenes 轨迹规划的 L2 与碰撞率。固定 commit 的 README 只称已发布 **Senna-VLM** 的代码、权重及训练/评测脚本；仓库评测入口算的是元动作分类准确率。论文 §III-C 称轨迹规划由扩展 VADv2 的 Senna-E2E 完成，但此仓库没有相应规划器的训练、推理或 nuScenes L2 评测实现。原先的 A 档抽样不成立，主代理已降为未选 C 档并递补 BEV-Planner。

## 发现与局限

本排除记录没有 hack 发现。元动作 QA 中存在未来动作监督，但公开代码无法追踪其如何进入 Senna-E2E 轨迹规划，不能据此解释 Table II 的 0.22 m 分数。该分数也不能与其他 nuScenes 工作在未核对协议时直接横比。
