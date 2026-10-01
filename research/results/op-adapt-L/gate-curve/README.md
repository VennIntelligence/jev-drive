# op-adapt L gate 与 curve 的小结果

预登记、完整执行日志、偏离与判定集中在 [todo](../../../../todos/2026-10-02-op-adapt-L-gate-and-curve.md)。这里的 CSV（逗号分隔结果表）和 JSON（结构化结果文件）来自 box 的 `runs/op_adapt_L/gate_curve/`，checkpoint（权重快照）、完整预测和输入缓存留在 box。

当前只含已完成 gate（起步开关）及数值检查，learning curve（随训练数据量变化的曲线）仍在执行。不能把现有文件解读为两项实验已全部完成。

| 文件 | 内容 |
|:--|:--|
| `gate_auc.csv` | 分类头在留出静止帧上的 AUC（不依赖阈值的正负排序能力）及整段 bootstrap（整段一起重抽）的区间 |
| `gate_metrics.csv` | 起步开关后的逐 seed（随机种子）及平均指标；RFS（评价者轨迹分）原始和 ×1.06 并列 |
| `training_counts.csv` | 四档采样的段、帧与独立事件数 |
| `curve_profile.json` | 相同批的 loss（损失）与 gradient（梯度）数值对齐、吞吐、显存和前向对齐 |
| `numeric_checks.json` | 独立重建 bootstrap 与加权分位数；gate 原提取路径和驻留路径的损失/梯度与性能 |
