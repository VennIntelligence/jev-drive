# NAVSIM v1：RAP-DINO

- 榜单口径：navtest，PDMS 93.8（论文 Table 1）。
- 固定版本：[`5fd8630`](https://github.com/vita-epfl/RAP/tree/5fd8630ae54442dd41827de4a9afe1690e3f02bb)。只读，未运行仓库代码、模型或评测。

## 读过

- `README.md` 的 NAVSIM 训练命令与结果表；`navsim/agents/rap_dino/rap_agent.py` 的 metric-cache 监督、分数损失，`rap_model.py` 的候选选择，`navsim_config.py`；`process_data/create_openscene_metadata_aug.py` 的 CV 过滤片段；`navsim/planning/training/dataset.py` 的 score mask。
- `papers/rap.pdf` 提取文本的 §3–4、Table 1、Table 5/6、附录的相关段落。

## 未读与局限

- 论文称 ego 训练集按 PDMS 过滤，但本固定代码未找到完整可追踪的 ego PDMS 过滤入口与阈值，所以没有把这项写成独立发现。
- 论文 Table 6 的 recovery perturbation 在 **v1 92.5→92.5**，无 v1 提分证据；即使它改善 v2，也不能当作 v1 hack。
- 未读全部渲染细节、外部数据、权重、Waymo/B2D 训练路径；未实测候选重排的因果增量。

## 整体印象与发现

RAP 的主要方法是扩大和对齐合成视觉训练样本，这可能改善可迁移能力，不能仅因合成数据就认定 hack。明确可审的榜单专用部分是以 NAVSIM PDMS 教师分数训练重排头，并用该预测分数选轨。发现：**NAV1-RAP-001**。
