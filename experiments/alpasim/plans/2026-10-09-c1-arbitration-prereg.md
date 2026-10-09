# C1 / C2 预登记：standstill 门控的 SH30 / AP2 仲裁，在 C0b 新 scene 上的 held-out 读数

2026-10-09，lane C1。写于 C0b 的逐 scene 表发布之前；本文件提交并 push 之后才读任何新 scene 的分数。
不向 AlpaSim 提交任何东西；候选 driver 里没有 WA-JEPA；判定时刻的信号不含特权输入。

## 背景（400 scene 上的离线结果，已读）

- 两个 driver 在第 0、1 次决策收到的输入逐 scene 完全相同（前 0.5 s 是 log replay；400 个 scene 上第 1 次决策的 anchor 位姿差为 0）。
  所以「在第 1 次决策时选定一个 driver、之后不换」的选择器，其闭环结果就是被选 driver 自己那次 rollout，分数可以直接从两次已有运行里取，不需要新的仿真。
  假设：第 0 次决策返回的轨迹不影响之后的状态（与上面的 anchor 一致性相符，未另行验证）。
- plan 分歧类的门控（纵向长度差、横向差、最大点距、航向差、AP2 自身 plan 抖动）按 log 分组的交叉验证全部为负（−0.006 到 −0.013），不做。
- 唯一为正的信号是起步速度：第 1 次决策时 ego 速度 < 1 m/s 的 76 个 scene 上 SH30 0.982、AP2 0.921（AP2 的 24 个零分有 6 个在这里，SH30 为 0 个）。
  自由阈值的交叉验证 +0.0079 [−0.0037, +0.0188]，留一 log +0.0085 [−0.0036, +0.0198]，阈值限在 {不切, 0.5, 1, 2, 3} 的交叉验证 +0.0098 [+0.0006, +0.0189]，
  事后固定 1 m/s 的全样本值 +0.0117 [+0.0016, +0.0223]（未交叉验证）。离线估计没有过「>= +0.010 且 CI 下界 > 0」的线，所以不实现门控 driver、不起仿真。

## 这里登记的读数

规则 R（冻结，不再调）：第 1 次决策时 ego 速度（driver 收到的 `DynamicState` 线速度的模）< 1.0 m/s 则整个 scene 用 SH30-F-s0，否则用 AP2-AB-s0。

- **数据**：C0b 发布的逐 scene 表里 SH30-F-s0 与 AP2-AB-s0 都有分数、且不在 C0 的 400 scene 列表（`scenes_public_landed4.txt`）里的全部 scene；速度取自该次运行的 `driver-logs/drive.jsonl`。
- **量**：R 的平均 scene score 减去 held-out scene 上均分较高的那个单 driver；CI 为按 log 的 cluster bootstrap（`jevdrive.stats`，B 10 000，95%）。
- **判定线**：差 >= +0.010 且 CI 下界 > 0。
- **同时报、不参与判定**：阈值 0.5 与 2.0；用第 0 次决策速度的变体；held-out 上的 oracle（逐 scene 取大）增益；起步 scene（< 1 m/s）里两个 driver 的零分与偏慢个数。

## 判定之后

- 过线：实现门控 driver（第 0、1 次决策两个模型都跑，之后只跑选中的），在 held-out 的一个子集上经 pool 起一次真实仿真核对「等于被选 driver 的 rollout」这一假设，再谈进不进提交。
- 不过线：仲裁这条线停；AP2 的起步过快另作为 AP2 自身的问题处理（见 C1 结果）。
- C0b 的表若在本 lane 结束前没有发布：只留这份登记，读数由 main 或后续 lane 按本文件执行（`experiments/alpasim/scripts/c1_arb.py` 的 `heldout` 子命令）。
