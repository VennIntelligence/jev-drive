# 单参数侧向传播项：描述性检验

**一个固定 signed 系数能解释本批后轴侧向位移残差的大部分变化，值得作为后续可证伪候选；尚未证明定位或驾驶闭环收益。** 未改冻结源、未调 GNSS gain、未启动实验。

目标 `y` 为[传播审查](pose-propagation-audit.md)的无侧向模型减真实后轴位移，投 truth yaw 区间中点 right 并除实际 dt，单位 m/s。输入 `x=(两端平均 signed SPEED)^2 × 两端平均 world gyro`；world gyro 右正，使用前一和当前 tick，未使用未来传感器。唯一拟合为 `y=k*x`，零截距。`k=sum(x*y)/sum(x*x)` 最小化全体 baseline 固定四窗样本的残差平方和；这是一个全局普通最小二乘系数，没有窗口权重、时移搜索、逐窗参数或滤波。若“total least squares”特指同时调整 x/y 的正交误差拟合，此处未采用；该方法还需要预先定义不同量纲观测的误差尺度。

用上轮 baseline-max 的 328 帧拟合，得到 **k=+0.010659832012199118 s²/m**。因此物理意义上的经验后轴侧向速度补偿应为 `−k*v²*world_gyro*right`，不能将残差正号直接加到传播中。该符号解释仅针对本定义。

| 数据 | n | 不加经验项 RMS m/s | 固定 k 后残差 RMS m/s | RMS 降幅 |
|---|---:|---:|---:|---:|
| baseline 四窗拟合样本 | 328 | 0.22850 | 0.02445 | 89.30% |
| short-max 四窗整臂留出 | 329 | 0.23347 | 0.02467 | 89.43% |

两者 prediction/target 相关系数分别 0.99418/0.99437。保留所有四窗帧、转向反转和瞬态，其中 abs world gyro<0.1 rad/s 的 baseline/short 样本分别 83/88；未通过删低信号帧或只取稳态提高结果。所有有符号预测与残差逐帧归档。

## Leave-one-window-out

每次仅在 baseline 另三个窗口训练同一形式的一个系数，预测被留出的整个窗口。下表是泛化描述，不将四个系数分别用作运行参数。

| 留出窗口 | 训练 k s²/m | 留出原 RMS m/s | 留出预测残差 RMS m/s | RMS 降幅 |
|---|---:|---:|---:|---:|
| 26966 右急弯 | 0.01140693 | 0.38631 | 0.04651 | 87.96% |
| 24240 左弯 | 0.01072560 | 0.11169 | 0.00859 | 92.31% |
| 17563 S1 | 0.01048708 | 0.18894 | 0.03407 | 81.97% |
| 17563 S2 | 0.01047792 | 0.19574 | 0.03457 | 82.34% |

全局固定 k 在每窗与 short 对照的精确结果见 [windows.csv](../results/pose-lateral-regression-v1/windows.csv)，LOO 见 [leave-one-window-out.csv](../results/pose-lateral-regression-v1/leave-one-window-out.csv)。短前视臂从未参与拟合。

## 范围与后续证伪

这里的真值目标来自相邻位置的 interval-average 差分；SPEED 参考 actor、位置参考后轴。pitch、转向运动学、轮胎/悬架响应、gyro 与位置的离散相位都可能共同形成该项，不能称已识别轮胎参数。三路线同车、6/8 m/s 的范围有限，邻近帧和两个 S 也不独立，相关系数不等于统计显著性或跨车泛化。

最小下一步仍应冻结一个系数，单独检查原传感器序列重放后的完整 pose 误差、GNSS 噪声抵消变化、两S相位和终点/缺测边界；不能只看拟合残差。即使定位回放改善，闭环中路线重接和动作会随之改变，须另做固定开发闭环，不能以 truth shadow 代替驾驶证据。此次没有实施补偿，也没有据此改变当前 Hermite 六例判定。

[复算脚本](../diagnostics/pose_lateral_regression.py)与[小汇总及输入 SHA](../results/pose-lateral-regression-v1/summary.json)已落盘；完整输入关联、逐帧 CSV、脚本副本及输出 SHA 在 `/data/runs/b2d/controller/lateral-followup/pose-lateral-regression-v1`。

```sh
PYTHONDONTWRITEBYTECODE=1 /data/envs/carla/bin/python todos/2026-09-23-lateral-followup/diagnostics/pose_lateral_regression.py --run-root /data/runs/b2d/controller/turns-v1 --propagation /data/runs/b2d/controller/lateral-followup/pose-propagation-v2 --out /data/runs/b2d/controller/lateral-followup/pose-lateral-regression-reproduction
```
