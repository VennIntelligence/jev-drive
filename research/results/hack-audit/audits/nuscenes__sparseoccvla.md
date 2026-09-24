# nuScenes / SparseOccVLA

- 仓库：[MSunDYY/SparseOccVLA，固定 commit e41b48e](https://github.com/MSunDYY/SparseOccVLA/tree/e41b48ebe35b78635d533500d1f433e8143d9754)。
- 论文：[SparseOccVLA](../../papers/sparseoccvla.pdf)，核对 §3.3、§4、Table 2、Table 4。
- 读过：根 README、`projects/configs/SparseOccVLA/` 中 nuScenes 配置、`projects/mmdet3d_plugin/datasets/nuscenes_dataset_v2.py`、`models/detectors/sparseoccvla.py` 的训练与推理路径、`evaluation/eval_planning.py`、`data_gen/planning_anchor.py` 与提示生成入口。
- 未读：仓库携带的 InternVL 通用子树、所有 CUDA kernel、完整数据生成依赖；未运行代码或评测。

## 整体印象

nuScenes 规划轨迹通过命令分组的聚类 anchor、两步扩散和语言隐藏状态选模输出。代码还从预制 `ad_info` 表注入历史 ego 状态。论文 Table 4 的去除 Traj-Ego Fusion 消融说明这一融合对开环 L2 影响显著。其 paper Table 2 的 avg L2 0.23 m 与 AD-MLP 的 ST-P3 数字不能直接混比。

## 发现

- **REAL-NUS-003**，`ego_state_fusion`：`temporal_ego_states` 从 AD-MLP 型预制表进入规划器。Table 4 给出完整模型与去融合模型在 1/2/3 秒 L2 的对照。

完整记录及固定 commit 行号见 [agent_real.jsonl](../findings/agent_real.jsonl)。

## 局限

固定代码用到 `gt_planning_command` 并按命令选轨迹分组，但仓库未公开该字段如何生成；无法判断它是否像 BEV-Planner++ 那样来自待测未来真值。最终矩阵的 `future_label_conditioning` 因此记 NA，不凭字段名判 yes。

预制 `ad_info` 的生成与字段模式没有完整公开，不能据此精确分解历史轨迹、速度和加速度各自贡献。`nuscenes_dataset_v2.py` 还把未来帧 CAN bus 输入占用预测分支；这些信号发生在规划轨迹输出之后，不能据此称它提高本次 nuScenes **规划**榜单分数，故未入 findings。规划评测只统计有效 6 帧未来样本；缺少具体筛除样本数，未推断分数被抬高。
