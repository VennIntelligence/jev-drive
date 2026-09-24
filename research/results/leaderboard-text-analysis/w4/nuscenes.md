# nuScenes 开环规划：UniAD 的 L2 与碰撞率实现

nuScenes 本身没有这里所指的统一“官方规划分数函数”。本页明确选择样本相关、被后续工作复用的 **OpenDriveLab/UniAD 官方实现** [`609ee083ea51c3521c323f1279dfc4cee0e60467`](https://github.com/OpenDriveLab/UniAD/tree/609ee083ea51c3521c323f1279dfc4cee0e60467)，文件 `projects/mmdet3d_plugin/uniad/dense_heads/planning_head_plugin/planning_metrics.py`。不能把它的 `obj_col/obj_box_col/L2` 当成 nuScenes 所有论文的唯一统一实现。只静态审读。

## 指标公式

- 每个未来时刻的位移误差 `L2_t = sqrt(Σ_{d∈{x,y}} ((pred_{t,d}−GT_{t,d})² × mask_{t,d}))`，累加各样本，再除总样本数；代码按时刻输出向量，表格中的 1/2/3 秒取对应时刻或研究实现自己的聚合。[`planning_metrics.py:119–148`](https://github.com/OpenDriveLab/UniAD/blob/609ee083ea51c3521c323f1279dfc4cee0e60467/projects/mmdet3d_plugin/uniad/dense_heads/planning_head_plugin/planning_metrics.py#L119-L148)。
- `obj_col_t`：预测车中心点所在 0.5 m BEV 栅格与时刻 `t` 的目标占用分割相交的样本比例；`obj_box_col_t`：长 4.084 m、宽 1.85 m 的车体多边形离散成占用栅格，与目标分割相交的样本比例。两者在 GT 车体本来碰撞的样本时刻都**不计预测碰撞**，分母仍是所有样本数。[栅格与车体 `planning_metrics.py:22–33,43–82`](https://github.com/OpenDriveLab/UniAD/blob/609ee083ea51c3521c323f1279dfc4cee0e60467/projects/mmdet3d_plugin/uniad/dense_heads/planning_head_plugin/planning_metrics.py#L22-L82)、[预测/GT 过滤 `:84–118`](https://github.com/OpenDriveLab/UniAD/blob/609ee083ea51c3521c323f1279dfc4cee0e60467/projects/mmdet3d_plugin/uniad/dense_heads/planning_head_plugin/planning_metrics.py#L84-L118)、[汇总 `:126–148`](https://github.com/OpenDriveLab/UniAD/blob/609ee083ea51c3521c323f1279dfc4cee0e60467/projects/mmdet3d_plugin/uniad/dense_heads/planning_head_plugin/planning_metrics.py#L126-L148)。

## 攻击面与第一轮对照

| 攻击面 | 代码所证机制与边界 | 第一轮 |
|---|---|---|
| 只用历史 ego 状态拟合短时 GT 轨迹 | L2 只看预测坐标对未来 GT 的距离，接口并不要求场景感知贡献；在易预测片段，一个无需理解交通的外推器也能得低 L2。是否与复杂场景驾驶等价由指标代码无法证明。[`planning_metrics.py:119–148`](https://github.com/OpenDriveLab/UniAD/blob/609ee083ea51c3521c323f1279dfc4cee0e60467/projects/mmdet3d_plugin/uniad/dense_heads/planning_head_plugin/planning_metrics.py#L119-L148) | `REAL-NUS-001`; `REAL-NUS-003`; `REAL-NUS-005` |
| 对 0.5 m 栅格判据调整训练和后处理 | 碰撞率由连续车体位置先量化到 BEV 格点，并用 `astype(np.int32)` 及边界 `clip` 查询；改变亚格点位置或损失栅格可改变碰撞计数，而未必改变真实接触风险。不能由此代码量出收益幅度。[栅格尺寸 `planning_metrics.py:22–29`](https://github.com/OpenDriveLab/UniAD/blob/609ee083ea51c3521c323f1279dfc4cee0e60467/projects/mmdet3d_plugin/uniad/dense_heads/planning_head_plugin/planning_metrics.py#L22-L29)、[量化查询 `:60–80`](https://github.com/OpenDriveLab/UniAD/blob/609ee083ea51c3521c323f1279dfc4cee0e60467/projects/mmdet3d_plugin/uniad/dense_heads/planning_head_plugin/planning_metrics.py#L60-L80) | `REAL-NUS-002` |
| 利用 GT 已碰撞时刻的免计规则 | `gt_box_coll` 为真时，预测中心点与预测车体碰撞均被屏蔽；这些时刻产生的预测碰撞不会进入报告的碰撞分子，即使行为没有变安全。这是场景限定的代码机制，未发现样本方法专门利用。[`planning_metrics.py:97–116`](https://github.com/OpenDriveLab/UniAD/blob/609ee083ea51c3521c323f1279dfc4cee0e60467/projects/mmdet3d_plugin/uniad/dense_heads/planning_head_plugin/planning_metrics.py#L97-L116) | 未观察到 |

## 盲区与证据界限

此实现只比较有限时刻的坐标与给定 occupancy 栅格，不仿真 ego 对其他交通参与者的反应，也不直接计算交通规则、乘坐舒适度、路线到达或行动是否合理。[`planning_metrics.py:84–148`](https://github.com/OpenDriveLab/UniAD/blob/609ee083ea51c3521c323f1279dfc4cee0e60467/projects/mmdet3d_plugin/uniad/dense_heads/planning_head_plugin/planning_metrics.py#L84-L148)。第一轮 `REAL-NUS-004/006` 的“未来 GT 派生高层命令”是**评测输入构造问题**，这一个分数函数无法证实或识别；因此不把它伪装成已由 UniAD 计分代码证明的攻击面。
