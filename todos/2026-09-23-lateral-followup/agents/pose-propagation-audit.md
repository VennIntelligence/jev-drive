# 后轴位移与 PoseFilter 传播审查

**结论：旧转弯日志支持“无侧向速度传播存在系统偏差”，不支持“全部定位误差都是噪声”，也不足以把全部残差认定为轮胎侧滑。** 右急弯中，该传播项的累积方向和实际定位偏差高度一致；左弯中 GNSS 项反而抵消一部分传播偏差。没有修改运行源、改变 gain、回灌 truth 或运行 CARLA。

## 数据与参考原点

原始六例为 `/data/runs/b2d/controller/turns-v1/{26966,24240,17563}/{baseline-max,short-max}/pursuit`，固定窗口和所有帧不变。`validation_trace.json` 保存 actor location、forward/right、角速度、速度模长和 signed_speed，**没有保存 actor 三维速度向量**，故不声称直接观测到 `v_actor + omega × lever`。actor 位置连续差分仅作为 interval-average 速度估计。

冻结 validator 将后轴位置写成 `actor_location + rear_offset * forward_vector`，`rear_offset=-1.388633220199954 m`；独立复算六例与 trace `truth_xy` 最大差为零。PoseFilter 输出、raw_pose_xy 和 TruthLogger truth 也都指向后轴 XY。旧 GNSS 安装在 actor x=−1.4 m，raw 地理投影另作两者差值的 lever-arm 修正，不能将 GNSS 点和 actor 原点混为一谈。

本地官方 SpeedometerReader 的实现取 actor velocity 与含 pitch 的 forward 向量点积，而非后轴速度。baseline 三条的 raw SPEED 与 trace signed_speed 最大差 8.31e−7 m/s。角速度字段明确为 deg/s，计算 lever 项前转 rad/s；360 个 abs omega_z>0.1 rad/s 的相邻区间，truth yaw 差分与端点平均 omega_z(rad/s) 比值中位数 1.000406，支持单位和符号一致。连续区间仍含离散/物理响应误差。

## 主证据：直接比较后轴位移

用两个相邻 truth yaw 的 wrapped 中点作为 forward/right，平均 recorded signed SPEED × 实际 dt 作为无侧向模型预测位移。残差定义为 **模型位移减去实际后轴位移**；right-positive，除 dt 后只称 interval-average velocity residual。下表保留全部固定窗口样本，无稳态/低速删选。

| 窗口/臂 | n | 横向残差均值 m/s | 横向残差 RMS m/s | 每 tick 横向残差 RMS cm | 纵向残差 RMS m/s |
|---|---:|---:|---:|---:|---:|
| 26966 baseline | 70 | +0.2963 | 0.3863 | 1.932 | 0.01659 |
| 26966 short | 71 | +0.2886 | 0.3927 | 1.963 | 0.01740 |
| 24240 baseline | 117 | −0.1050 | 0.1117 | 0.558 | 0.00219 |
| 24240 short | 117 | −0.1048 | 0.1120 | 0.560 | 0.00218 |
| S1 baseline | 70 | +0.0035 | 0.1889 | 0.945 | 0.01181 |
| S1 short | 70 | +0.0029 | 0.1961 | 0.980 | 0.01116 |
| S2 baseline | 71 | +0.0053 | 0.1957 | 0.979 | 0.01344 |
| S2 short | 71 | +0.0051 | 0.1995 | 0.997 | 0.01264 |

S 中左右两段使有符号均值接近零，RMS 明确不为零，不能据均值消失称模型正确。此处真值中点航向排除了估计航向误差作为主残差的必要条件。

辅助交叉检查：以 actor location 差分，加两端 `omega × (rear_offset * forward)` 的平均，重建后轴 interval-average 速度。它与后轴位置直接差分的中点 right 分量 RMS 差在 baseline 四窗依次为 0.00912、0.00275、0.02129、0.02125 m/s，小于主残差，但不是零。完整三维 pitch/roll、角速度取样与离散近似都保留。不能把此近似包装成已归档的直接 actor velocity。

## 实际滤波误差的精确恒等分解

按原 PoseFilter 的上一估计 yaw、当前 gyro、两帧 SPEED 和实际 dt 重建本 tick 传播位移 `p`。令 `d` 为真实后轴位移，`n=raw_pose−truth`，`e=filtered_pose−truth`，gain `g=.05`，则：

`e[k] = (1−g)e[k−1] + (1−g)(p[k]−d[k]) + g*n[k]`。

将初始误差、传播强迫项和 raw-GNSS 强迫项分别沿整条历史递推，六例所有帧相加复现原 `e`，最大绝对误差 <3.73e−12 m。各项是在世界 XY 递推后投到当前参考左法向；不是把 CTE 曲线分开滤波，也不是控制反事实。这一分解精确，但传播项仍包含估计航向/速度和离散误差，GNSS 项也可能有偏差，不能直接分别命名为“纯侧滑”和“纯随机噪声”。

| baseline 窗口 | 实际左向 pose error RMS m | 传播历史项 RMS m | raw-GNSS 历史项 RMS m | 传播项/实际误差相关系数 | 同号比例 |
|---|---:|---:|---:|---:|---:|
| 26966 | 0.3300 | 0.2363 | 0.1094 | 0.938 | 85.7% |
| 24240 | 0.0727 | 0.0925 | 0.1155 | −0.163 | 69.2% |
| S1 | 0.0900 | 0.0804 | 0.0627 | 0.785 | 84.3% |
| S2 | 0.0978 | 0.0848 | 0.0656 | 0.776 | 70.4% |

RMS 分量相关，**不能相加或当作方差占比**。26966 实际/传播/raw-GNSS 左向均值分别 −0.2881/−0.1954/−0.0922 m：模型让估计后轴偏右，与实际估计偏差方向相符；24240 分别 +0.0084/+0.0843/−0.0763 m，主要存在抵消。对真实路线跟踪误差与 pose−truth 误差必须分开表述，此表仅审查后者。

**模型推断限定：** 固定世界方向、恒定速度偏差、零均值 GNSS 且已稳定时，传播偏差到位置误差的比例为 `(1−g)/g * dt ≈ 0.95 s`，离散衰减时间常数 `−dt/log(1−g)≈0.975 s`。在右急弯，0.296 m/s ×0.95 s 给出约0.28 m的量级；曲率、姿态和 raw-GNSS 偏差随时间变化，不能把该乘积当实测拟合或调 gain 依据。

## 复算与边界

[脚本](../diagnostics/pose_propagation.py)、[完整八窗 CSV](../results/pose-propagation-v2/windows.csv)、[恒等式验证](../results/pose-propagation-v2/verification.json) 和 input/output SHA 已保留。完整逐帧 CSV 与脚本副本在 `/data/runs/b2d/controller/lateral-followup/pose-propagation-v2`。v1 也保留；v2 仅将辅助 actor/omega 向量的投影统一改为区间中点轴，主后轴残差与滤波分解不变。

```sh
PYTHONDONTWRITEBYTECODE=1 /data/envs/carla/bin/python todos/2026-09-23-lateral-followup/diagnostics/pose_propagation.py --run-root /data/runs/b2d/controller/turns-v1 --out /data/runs/b2d/controller/lateral-followup/pose-propagation-reproduction
```

结论只能支撑下一步辨识传播模型：它不证明提高 GNSS gain 有净收益，不证明加入某个横向速度估计会改善闭环，也没有测出唯一轮胎侧滑参数。当前真实速度参考点、pitch、采样相位、有限差分与传感器误差仍需要分开控制。
