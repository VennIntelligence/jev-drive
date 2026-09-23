# 固定系数全序列定位重放

**结果有利有弊：右急弯与两个 S 的定位误差下降，左弯反而变大。** 这证伪了“经验项拟合残差降低就必然使融合定位全面改善”的推断；还没有任何驾驶闭环收益结论。原冻结运行源码未修改。

使用父代理冻结的 `k=0.010659832`，没有重新拟合。完整原始 GPS/IMU/SPEED、原路线 GPSProjector 和冻结 PoseFilter 均取自 `turns-v1`。六例共 **2761 帧**：k=0 的 XY、yaw、raw XY 全部逐值等于原记录；加补偿后的 yaw 也逐值等于 k=0。验证为数值严格相等，非宽容差匹配。

区间定义固定为前一/当前 SPEED 均值的平方，乘前一/当前右正 world gyro 均值。方向使用原滤波器当前传播的中点：上一估计 yaw + 当前 gyro × dt/2。新增位移为 `−k * v² * gyro * right * dt`，加入 GNSS 更新前的预测位置，仍由原 `.05` 增益校正一次。没有未来传感器、truth 输入、改 heading 更新、增加 gain 或改控制器。真值只用于独立评分。

## 固定四窗

以下全部为欧氏 position error，保留各窗口所有原始样本。两臂保持各自已记录物理运动；short 是旧短前视日志，而不是本次重新运行。

| 窗口/原臂 | n | 原 RMS m | 补偿 RMS m | 原 P95 m | 补偿 P95 m |
|---|---:|---:|---:|---:|---:|
| 26966 baseline | 70 | 0.35796 | 0.11210 | 0.53419 | 0.18446 |
| 26966 short | 71 | 0.36068 | 0.11526 | 0.54410 | 0.19305 |
| 24240 baseline | 117 | 0.09470 | 0.13591 | 0.15414 | 0.25888 |
| 24240 short | 117 | 0.09488 | 0.13592 | 0.15408 | 0.25889 |
| S1 baseline | 70 | 0.19070 | 0.16494 | 0.29847 | 0.26813 |
| S1 short | 70 | 0.19284 | 0.16610 | 0.30161 | 0.26931 |
| S2 baseline | 71 | 0.12683 | 0.09335 | 0.23960 | 0.15838 |
| S2 short | 71 | 0.12584 | 0.09295 | 0.23826 | 0.15528 |

左弯 baseline RMS 增加 0.04121 m、P95 增加 0.10474 m，不能藏在其他窗口收益里。对应左向误差 RMS 为 0.07269→0.11949 m。[此前误差递推分解](pose-propagation-audit.md)显示该窗传播项与 raw-GNSS 项原本部分抵消；消除前者不会自动降低融合后的总误差。这是对同一记录的代数/重放解释，不等于 GNSS 偏差在别次运行会同样发生。

heading 未改变，完整每窗 RMS/P95/max 均在 CSV 中。它不能被计为补偿额外收益。

## 全路线与固定直线接近/间隙守护

完整 baseline 路线 position RMS：26966 0.21172→0.14510 m，24240 0.12830→0.14134 m，17563 0.14583→0.13879 m；short 同方向，数值分别 0.21196→0.14508、0.12831→0.14129、0.14620→0.13905 m。

额外守护预先固定为转弯 pad 前保留 5 m 余量的 approach：26966 s=0–19 m、24240 s=0–13 m、17563 s=0–22.5 m，再加 S 间隙 53.5–68.5 m。这些是固定站距接近/间隙区间，不按误差或动作选帧，不声称每个参考段曲率严格为零。各例 RMS 变化在 −0.00011 至 +0.00095 m；S 增幅最大仍 <1 mm。末 5 s 全帧 RMS 最大增幅约 0.00022 m。这里只做定位守护描述，不替代真正停车控制验收。

## 证据与下一步边界

[复算脚本](../diagnostics/pose_compensation_replay.py)、[所有窗口/全程/守护 CSV](../results/pose-compensation-replay-v1/metrics.csv)、[严格原输出验证](../results/pose-compensation-replay-v1/verification.json)、[输入及代码 SHA](../results/pose-compensation-replay-v1/manifest.json) 已归档。全部逐帧 XY/误差/补偿向量、冻结 adapter 副本与输出 SHA 在 `/data/runs/b2d/controller/lateral-followup/pose-compensation-replay-v1`。

```sh
PYTHONDONTWRITEBYTECODE=1 /data/envs/carla/bin/python todos/2026-09-23-lateral-followup/diagnostics/pose_compensation_replay.py --run-root /data/runs/b2d/controller/turns-v1 --out /data/runs/b2d/controller/lateral-followup/pose-compensation-replay-reproduction
```

当前只能说固定项改善某些已记录转弯的定位、同时损害左弯；不能据此升级默认。真实闭环会改变参考重接、控制与物理轨迹，不能由这次固定物理轨迹的定位重放推出。若后续实施，应显式保留左弯非恶化条件与新运行 GNSS 噪声的不确定性，不能只挑右急弯展示。
