# NAVSIM v1：TOAD+DrivoR

- 榜单口径：`navtest`，PDMS。抽样 94.7 来自 TOAD 论文 Table 1（DrivoR 94.6→TOAD 94.7）；不能与 NAVSIM-v2 的 EPDMS 混用。
- 固定版本：TOAD `nav1` 分支 [`0110647`](https://github.com/valeoai/TOAD/tree/0110647142888dd56644565588510812bdafaa27)。只读代码和论文；未运行仓库代码、模型或评测。

## 读过

- `README.md` 的 NAVSIM-v1 Evaluation/Results 和 Submission 命令；`navsim/planning/script/config/common/agent/drivoR.yaml` 的 PDMS 子分数权重与 CEM 默认值。
- `navsim/planning/script/run_pdm_score_multi_gpu.py`、`navsim/planning/script/run_create_submission_pickle_warmup_gpu.py`、`navsim/planning/training/agent_lightning_module.py`、`navsim/agents/drivoR/drivor_agent.py` 的预测与提交路径；`navsim/agents/drivoR/drivor_model.py` 的初选、CEM 目标和最终选轨。
- `papers/toad.pdf` 的提取文本：§3–5、Table 1、Figure 4。
- 独立复核后补读 README 的 checkpoint 训练命令、`drivor_features.py` 长轨迹目标构造和 `drivor_loss.py` 对该目标的附加回归损失。

## 代码路径与论文对照

- README [评测命令](https://github.com/valeoai/TOAD/blob/0110647142888dd56644565588510812bdafaa27/README.md#L165-L195) 固定 `train_test_split=navtest`、64 样本、10 次 CEM 迭代、`use_cem=true`；[提交命令](https://github.com/valeoai/TOAD/blob/0110647142888dd56644565588510812bdafaa27/README.md#L245-L278) 也启用同一设置并调用写出 `submission.pkl` 的脚本。YAML 中的 5 轮是默认值，README 的命令覆盖为 10 轮；论文 §4.1 对在线生成候选的方法写通常使用 5 轮，故论文分数不能视为此命令的精确复现。
- 评测入口的 `trainer.predict` 经 `AgentLightningModule.predict_step_drivor` 调用 `agent.forward` 并取 `predictions["trajectory"]`；模型在非训练模式下调用 `cem_refine`，再把 CEM 轨迹与原 argmax 轨迹按同一舒适度修正目标比较后选出输出。CEM 循环把运动学展开轨迹交给学习的 PDM 子评分器，同时减舒适、控制锚定、位置锚定惩罚，并按精英样本更新高斯分布。这是 **NAV1-TOAD-001** 的直接代码依据。
- 论文 §3 描述测试时 CEM 与冻结评分器，Table 1 给出 navtest PDMS 94.6→94.7（+0.1）。nav1 README 的后来版本写 94.6→94.9（+0.3），同时明确说明额外提高涉及 random bouquet 扩充候选和训练更强评分器，且命令迭代数是 10 而非论文通常设置的 5；不能把两份结果的差额都算作相同配置上 CEM 的独立增益。

## 未读与局限

- 未读全量 `nuplan-devkit/` 依赖、资产、未下载的 checkpoint 和运行输出；没有验证实际耗时或真实道路收益。
- nav1 README 提交示例先设 `EXPERIMENT=drivoR_nav1`，后面却传 `experiment_name=${experiment_name}`；静态文本中没有给小写变量赋值。此处可能需要调用者自行设置，未运行故不声称该命令可以原样提交。
- 该分支的 `proposal_augment_random=64` 在评测和提交命令中开启，模型 `forward` 在非训练时也会追加候选；README 将 94.9 的额外提高与此及更强评分器联系起来。现有论文消融不能单独量化该分支中 CEM、候选增强与评分器各自对 94.9 的贡献。
- README 将被评测 checkpoint 关联到 `long_trajectory_additional_poses=2` 的训练命令；该命令写 `agent=drivoR_speed_up`，但固定仓库未见同名 Hydra 配置。公开源码可确认长目标分支及损失、README 可确认评测权重文件名，实际权重的训练史无法逐步复现，故 **NAV1-TOAD-002** 仅记中置信度，不能把 DrivoR 较小配置的消融数字当成 TOAD 此 checkpoint 的贡献。

## 整体印象与发现

NAVSIM-v1 的 TOAD 发现由专用 `nav1` 分支和明确的 `navtest` 评测、提交路径支撑。测试时搜索本身可以用于部署；分数中可能不迁移的部分来自以 NAVSIM PDMS 学得的代理目标选择轨迹。论文报告版本的可见增益很小（+0.1 PDMS），后续 README 数字另有训练、候选和迭代数变化。发现：**NAV1-TOAD-001、NAV1-TOAD-002**。
