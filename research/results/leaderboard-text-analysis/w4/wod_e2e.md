# WOD-E2E：Rater Feedback Score（RFS）

官方实现位于 Waymo Open Dataset 仓库 [`99a4cb3ff07e2fe06c2ce73da001f850f628e45a`](https://github.com/waymo-research/waymo-open-dataset/tree/99a4cb3ff07e2fe06c2ce73da001f850f628e45a) 的 `src/waymo_open_dataset/metrics/python/rater_feedback_utils.py`，核心入口 `get_rater_feedback_score`。本页对应公开的 **RFS 函数**；未声称覆盖服务器对提交 protobuf 的其他校验、最终榜单汇总方式或保密测试标签。未运行评测。

## 指标公式

- 默认采样 4 Hz、5 秒、20 个 waypoint，评分只取第 3 秒与第 5 秒的预测点相对每条人工评定轨迹的**横向和纵向投影绝对误差**。基准阈值 1.0 m、1.8 m；纵向阈值默认再乘 4；阈值随初速以 `clip(0.5+0.5(v−1.4)/(11−1.4),0.5,1)` 缩放。[阈值 `rater_feedback_utils.py:21–43`](https://github.com/waymo-research/waymo-open-dataset/blob/99a4cb3ff07e2fe06c2ce73da001f850f628e45a/src/waymo_open_dataset/metrics/python/rater_feedback_utils.py#L21-L43)、[默认参数 `:167–181`](https://github.com/waymo-research/waymo-open-dataset/blob/99a4cb3ff07e2fe06c2ce73da001f850f628e45a/src/waymo_open_dataset/metrics/python/rater_feedback_utils.py#L167-L181)、[投影与取样 `:304–327`](https://github.com/waymo-research/waymo-open-dataset/blob/99a4cb3ff07e2fe06c2ce73da001f850f628e45a/src/waymo_open_dataset/metrics/python/rater_feedback_utils.py#L304-L327)。
- 对每个人工轨迹 `p`、每个候选 `i`、3/5 秒位置，归一化距离 `u=max(|d_lon|/τ_lon,|d_lat|/τ_lat)`；该时刻轴的评分为 `label_p × 0.1^{max(u−1,0)}`。代码随后**按 3 秒、5 秒轴分别取最大人工轨迹分数，再取两轴均值**。若候选没有完整落入任一人工轨迹的两时刻 trust region，单候选分数被下限截为至少 4.0。最终 RFS 为各候选分数乘其报告概率的和。[`rater_feedback_utils.py:359–419`](https://github.com/waymo-research/waymo-open-dataset/blob/99a4cb3ff07e2fe06c2ce73da001f850f628e45a/src/waymo_open_dataset/metrics/python/rater_feedback_utils.py#L359-L419)。
- 本函数将人工轨迹数量和 waypoint 数裁剪或复制末项至默认形状；不由此推断隐藏测试集的人工标签数量。[`rater_feedback_utils.py:96–164,215–225`](https://github.com/waymo-research/waymo-open-dataset/blob/99a4cb3ff07e2fe06c2ce73da001f850f628e45a/src/waymo_open_dataset/metrics/python/rater_feedback_utils.py#L96-L164)。

## 攻击面与第一轮对照

| 攻击面 | 代码所证机制与边界 | 第一轮 |
|---|---|---|
| 用 RFS 奖励、评分头或候选搜索直接优化提交轨迹 | 该函数把候选轨迹与标签映射为单一可计算标量，最终可在多候选概率和中选权重；研究者可训练其代理或在公开验证集调候选。代码只证明“目标可被直接优化”，不证明真实驾驶有或没有改善。[`rater_feedback_utils.py:359–419`](https://github.com/waymo-research/waymo-open-dataset/blob/99a4cb3ff07e2fe06c2ce73da001f850f628e45a/src/waymo_open_dataset/metrics/python/rater_feedback_utils.py#L359-L419) | `REAL-WOD-001/002/003` |
| 只对 3 秒、5 秒位置精确拟合 | 路径中其余 18 个采样位置并不进入评分函数的距离计算；两条轨迹即便中间运动截然不同，只要两个检查点和评定概率相同，在此 RFS 函数中得分相同。[`rater_feedback_utils.py:320–323,304–318`](https://github.com/waymo-research/waymo-open-dataset/blob/99a4cb3ff07e2fe06c2ce73da001f850f628e45a/src/waymo_open_dataset/metrics/python/rater_feedback_utils.py#L304-L323) | 未观察到 |
| 不在完整 trust region 中仍得最低 4.0 | 代码把区外候选分数下限设为 4，严重偏离人工建议的轨迹不会仅因偏离而在此函数中低于 4；以高探索分支换其他收益的净效果仍取决于标签和概率。[`rater_feedback_utils.py:375–413`](https://github.com/waymo-research/waymo-open-dataset/blob/99a4cb3ff07e2fe06c2ce73da001f850f628e45a/src/waymo_open_dataset/metrics/python/rater_feedback_utils.py#L375-L413) | 未观察到 |
| 两时刻可分别匹配不同 rater | `amax(axis=1)` 在保留 3/5 秒轴的张量上做；两个时刻的最大值可来自不同人工轨迹，平均时不要求候选全程对应同一人工决策。[`rater_feedback_utils.py:390–405`](https://github.com/waymo-research/waymo-open-dataset/blob/99a4cb3ff07e2fe06c2ce73da001f850f628e45a/src/waymo_open_dataset/metrics/python/rater_feedback_utils.py#L390-L405) | 未观察到 |

## 盲区与证据界限

该实现的输入是预测二维轨迹、概率、人工评定轨迹/标签和初速；没有交通场景状态、碰撞仿真、规则判定、舒适度或驾驶员干预计数。它直接测的是相对人工评定建议的有限时刻几何相符性；其他行为只能通过人工标签及与这些建议的接近程度间接体现。[`rater_feedback_utils.py:167–213,304–419`](https://github.com/waymo-research/waymo-open-dataset/blob/99a4cb3ff07e2fe06c2ce73da001f850f628e45a/src/waymo_open_dataset/metrics/python/rater_feedback_utils.py#L167-L213)。
