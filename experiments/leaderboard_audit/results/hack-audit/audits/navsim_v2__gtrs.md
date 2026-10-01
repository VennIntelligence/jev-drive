# NAVSIM v2：GTRS

- 榜单口径：抽样日志的 **45.4 是 navhard 双阶段合并 EPDMS**，不是 Stage 2 的单项数字。
- 固定版本：[`92a740d`](https://github.com/NVlabs/GTRS/tree/92a740def80610e4096962d25cc837b21e72ef78)。只读，未运行仓库代码、模型或评测。

## 读过

- `README.md`；`docs/gtrs_inference.md`、`docs/gtrs_training.md`；`navsim/agents/gtrs_dense/hydra_model.py`、`gtrs_aug/hydra_model.py` 的候选合并和评分细排；相应 YAML 配置；`traj_final/gather_traj.py`、`kmeans.py`；`navsim/planning/script/run_pdm_score_gpu_v2.py` 的双阶段入口。
- `papers/gtrs.pdf` 提取文本的 §2–4、Table 1/2。

## 未读与局限

- 本固定 GTRS 论文是 2025 年早期版本，Table 2 的 V2-99 GTRS-Aug 为 **42.1**；README 也写 42.1。DrivoR 后续论文 Table 3 在官方指标 bug 修复后列 GTRS-A V2-99 为 **45.4**。没有同一固定提交直接对应 45.4 的修复后运行记录，因此只审计可见的 GTRS 推理机制，不把旧协议 ablation 的数字宣称为 45.4 的分解。
- GTRS 论文还列 ViT-L GTRS-Aug 45.3、六模型 ensemble 49.4，均与抽样 45.4 不同。
- 未读全部模型骨干、资产中的候选数组内容、外部权重/数据、评测日志；未运行任何模型或评测。

## 整体印象与发现

GTRS 由扩散模型提出动态轨迹，再与离线大词表合并评分；Aug 细排使用代码中的手定子指标权重。阶段 2 回扫还确认：`gtrs_aug_agent.py` 从预计算 PDM 子分数读取训练标签，`hydra_model.py` 由子分数预测头合成候选分数再 `argmax` 选轨迹。这与其他 NAVSIM 方法的评分代理选择同类，补充 **NAV2-GTRS-003**（最终记录见 [findings.jsonl](../findings.jsonl)）。旧协议 Table 1 可量化候选池的贡献，但该数字不可移到修复后的 45.4。发现：**NAV2-GTRS-001、NAV2-GTRS-002、NAV2-GTRS-003**。
