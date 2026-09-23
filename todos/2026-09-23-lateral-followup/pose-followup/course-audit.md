# 车身航向与后轴 course：失败解释审查

**冻结结论不变：固定 k 候选未通过右弯 body-heading P95 的 +1° 保护条件，不做反序确认或扩参。** 新 course 指标仅解释运动方向，不能替代该门槛。四窗实际后轴运动方向和车身方向并不相同，但本次超限也不全是这种差异造成的。

## 参考、时间与样本

来源为完整 `pose-g2-v1` 六例和已冻结 `pose-g2-v1-analysis-v1`。脚本独立重算 current body heading / nearest station，与冻结分析逐帧核对；使用同一 canonical reference 和5 m居中弦方向。后轴 `truth_xy` 采用冻结分析实际使用的 control TruthLogger 字段，仍为独立 actor 快照真值。validator trace 的同帧 rear 点只存在浮点表示差异；不能混用不同表示然后称逐值核对。

`course=atan2(truth_xy[k]−truth_xy[k−1])` 自然对应相邻 tick 的区间中点。主要新对照将 body yaw 用 wrapped 两端中点、参考方向用后轴位置中点的独立投影站距，从而在同一中点比较 body/course。另保留 course 对当前帧参考的指标，以暴露约半tick时序差异；未使用后帧。角度均 CARLA 右正。

全部四窗的两臂样本均保留：26966 70/70，24240 117/117，S1 70/70，S2 71/72。没有窗口速度<2 m/s或未定义 course 样本。完整路线的首帧、断帧/无位移 course 保留原行并标原因，不填零或删除后称完整。没有低通或按动作删帧。

## 逐窗结果（baseline → fixed k）

| 窗口 | 冻结 current-body abs P95 ° | 中点对齐 course abs P95 ° | body−course offset RMS ° |
|---|---:|---:|---:|
| 26966 急右弯 | 4.27155 → 5.44955 | 7.17766 → 6.67164 | 2.82777 → 2.88418 |
| 24240 左弯 | 1.70110 → 1.92534 | 1.79021 → 1.59749 | 0.81050 → 0.81071 |
| S1 | 3.86679 → 3.70477 | 6.19282 → 6.38351 | 1.85924 → 1.89565 |
| S2 | 5.77390 → 5.83946 | 8.45521 → 8.44321 | 1.92906 → 1.90998 |

course 并非四窗全面改善：S1 增加。P95 峰值未必位于同一帧，不能将表中两个 P95 直接相减当作侧偏角。右弯 body−course signed 均值 +2.16622→+2.16244°，几乎不变；不是候选突然增加了一个大的平均侧向运动偏移。

## 右弯超限位置

冻结逐窗允许值为 baseline P95 +1° = **5.271546829°**。candidate 有8个连续样本超过这一值：frame **1930–1937**，站距 **43.23–45.93 m**，均在 core 29–46 m 的后段，尚未进入 exit 46–51 m。这些样本用于定位航向 P95 失败发生的位置，不是新增逐帧通过门槛。

这些帧 current body error 为 **+5.31 至 +5.68°**；中点对齐 course error 仍为 **+2.53 至 +4.20°**。所以实际运动方向也向参考方向右侧偏离，不能把全部超限解释为“车身朝向偏了但运动完全正确”。另一方面，body−course 的正向差明确存在，车身航向误差也不能直接等同后轴运动方向误差。

这里没有逆向行驶证据：四窗 course 相对参考远未达到90°。正向几度误差可能出现在外侧残差的纠偏过程中；这批日志不足以将它唯一命名为错误转向、轮胎侧滑或理想纠偏。不得因此免除已冻结的航向限制。

## 与固定经验侧向项的关系

以同区间 mean SPEED² × mean world gyro、原固定 k（不重新拟合）预测 model-minus-rear 右向速度残差，右窗残差 RMS 为 baseline 0.02396、candidate 0.02460 m/s；同号比例两者均94.29%。其余窗 residual RMS 约0.0079–0.0326 m/s，关系继续存在。body−course offset 符号与该项一致：右弯为正、左弯为负，S两侧反转，不能用S的近零均值抹掉运动偏差。

这说明车身方向与后轴 course 的差别有可复查的运动学基础；不证明已识别轮胎参数，也不证明 heading 保护可以取消。候选改善了四窗 CTE、却付出右弯 body-heading 代价；当前只保留这个开发取舍事实。

## 归档

[复算脚本](../diagnostics/pose_course_audit.py)、[全部阶段指标](results/course-audit-v3/phase-metrics.csv)、[右窗8帧上下文](results/course-audit-v3/right-threshold-exceedances.csv)、[输入 SHA/核对记录](results/course-audit-v3/manifest.json)、[输出 SHA](results/course-audit-v3/outputs-sha256.json)。完整全帧 CSV 与脚本副本在 `/data/runs/b2d/controller/lateral-followup/pose-course-audit-v3`。

v1/v2 的前置一致性断言发现不同 reference/后轴浮点表示不能混用，均在输出结果前退出；空目录保留。v3 显式统一为冻结 canonical reference 与 TruthLogger 后轴点，逐帧重现原 heading/station 后才生成结果，没有改动冻结分析或运行源。

```sh
PYTHONDONTWRITEBYTECODE=1 /data/envs/carla/bin/python todos/2026-09-23-lateral-followup/diagnostics/pose_course_audit.py --run-root /data/runs/b2d/controller/lateral-followup/pose-g2-v1 --analysis /data/runs/b2d/controller/lateral-followup/pose-g2-v1-analysis-v1 --out /data/runs/b2d/controller/lateral-followup/pose-course-audit-reproduction
```
