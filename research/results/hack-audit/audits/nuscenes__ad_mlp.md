# nuScenes / AD-MLP

- 仓库：[E2E-AD/AD-MLP，固定 commit 4b93ba0](https://github.com/E2E-AD/AD-MLP/tree/4b93ba085ee47474152f282177865796ea577fc0)。
- 论文：[Rethinking the Open-Loop Evaluation of End-to-End Autonomous Driving in nuScenes](../../papers/ad_mlp.pdf)，核对 §2、§3、§4、Table 1。
- 读过：根 README、`pytorch/admlp/planner.py`、`train.py`、`eval_weight.py`、`evaluate_for_mlp.py`、`generate_fengze.py`、`stp3/datas/NuscenesData.py`，以及 ST-P3 评测依赖的关键入口。
- 未读：全部 ST-P3 上游实现和 Paddle 细节；未运行模型、仓库代码或评测。

## 整体印象

这是有意揭示开环规划指标弱点的 toy baseline。训练和推理使用预制 ego 运动特征，不读视觉输入；论文及 README 对这一点明确披露。README 在 2023-10-20 承认早期训练数据有误，并把更正后的全输入结果从 avg L2 0.23 m 更新为 0.29 m、碰撞率从 0.12% 更新为 0.19%。原抽样表的 0.35 m 对应论文去掉高层命令的消融，而非最终 Ours 成绩；抽样表已同时保留排行转录和论文纠错值。

## 发现

- **REAL-NUS-001**，`ego_only_open_loop_planning`：模型不读环境感知。论文 Table 1 中只加入历史 ego 状态而无未来命令时 avg L2 为 0.35 m；完整 21 维含未来命令的 0.29 m 由 `REAL-NUS-006` 另行解释，不能全归因于可部署运动外推。
- **REAL-NUS-002**，`metric_grid_alignment`：训练损失按 0.5 m 占用栅格重加权，同评测离散化耦合；论文披露但无独立消融。
- **REAL-NUS-006**，`future_label_conditioning`：`NuscenesData.py` 从待预测未来三秒的真值轨迹生成高层命令；论文说该命令进入 21 维输入，Table 1 的有/无命令对照是 avg L2 0.35→0.29 m。公开推理路径读取预制特征 pkl，但该 pkl 的逐字段构造未公开，所以记为中置信度。完整记录见 [root.jsonl](../findings/root.jsonl)。

前两条完整记录及固定 commit 行号见 [agent_real.jsonl](../findings/agent_real.jsonl)。论文表中的 ST-P3 指标、别的论文的 L2 和碰撞率实现未必一致，审计不把它们混合排序。

## 局限

`stp3_val/data_nuscene.pkl` 的生成过程和全部字段未随仓库公开；代码能证明 token→预制特征→MLP 的路径，具体 21 维语义由论文解释。README 推荐对近零预测轨迹做阈值归零，以处理碰撞栅格误判；当前已读评测入口未见默认启用，因此没有把建议当作实际分数 hack。各发现对真实闭环的影响仍需实测。
