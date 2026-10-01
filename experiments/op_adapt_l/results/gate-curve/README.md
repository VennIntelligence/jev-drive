# op-adapt L gate 与 curve 的小结果

预登记、完整执行日志、偏离与判定集中在 [todo](../../plans/2026-10-02-op-adapt-L-gate-and-curve.md)。这里的 CSV（逗号分隔结果表）和 JSON（结构化结果文件）来自 box 的 `runs/op_adapt_L/gate_curve/`，checkpoint（权重快照）、完整预测和输入缓存留在 box。

gate（起步开关）和八个 learning curve（随训练数据量变化的曲线）run 已全部完成。gate 的 AUC 为0.774，但收益与前沿线失败；110段/类三类行为增益都过线，300段/类达到登记饱和，各档均未满足全部可用线。完整数值、逐线判定、解释限制与D1–D7偏离以todo为准，没有放宽线。

| 文件 | 内容 |
|:--|:--|
| `gate_auc.csv` | 分类头在留出静止帧上的 AUC（不依赖阈值的正负排序能力）及整段 bootstrap（整段一起重抽）的区间 |
| `gate_metrics.csv` | 起步开关后的逐 seed（随机种子）及平均指标；RFS（评价者轨迹分）原始和 ×1.06 并列 |
| `training_counts.csv` | 四档采样的段、帧与独立事件数 |
| `curve_profile.json` | 相同批的 loss（损失）与 gradient（梯度）数值对齐、吞吐、显存和前向对齐 |
| `numeric_checks.json` | 独立重建 bootstrap 与加权分位数；gate 原提取路径和驻留路径的损失/梯度与性能 |
| `metrics.csv` | gate三seed、四档curve两seed及其平均值；捕获、误触发、ADE、漂移、速度与两尺度RFS，全部整段2000次/seed0 CI（置信区间） |
| `lines_by_seed.csv` / `lines_all_seeds.csv` | 每seed及全部seed共同满足的G0–G6、C1–C4判定 |
| `curve_ratios.csv` / `interpretations.json` | 相对全量收益、逐seed比值、整段配对饱和区间、登记读法 |
| `frontier.csv` / `frontier.json` | dw（蒸馏权重）扫描与stayheavy候选的前沿点、gate前沿余量的CI |
| `execution_checks.json` / `run_audit.json` | gate整段划分、锁定SHA256（文件校验）、联合训练计数、单位检查、八run配置核对和报告修复前后逐字节一致性 |
| `unit_trend_bounds.json` | 精确首/末160步单位loss趋势的非负损失保守证明 |
| `timings.csv` | 八run4000步耗时、含dev（选择集）的吞吐、数据等待与显存 |
| `gpu_util_samples.csv` / `gpu_util_summary.csv` | 5s GPU采样与按阶段的三卡利用率汇总；时间戳为Unix epoch（自1970年起的秒数） |

三张论文风格PNG与中文读法在todo中引用，PDF（矢量图格式）留box。没有下载预测、checkpoint或特征缓存。CI只覆盖留出数据的整段抽样不确定性，不覆盖训练seed或不同训练段名单的变化。
