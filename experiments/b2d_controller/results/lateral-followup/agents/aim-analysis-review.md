# 首例前独立分析器审查

只读核对 `protocol.md`、`analysis/analyze_aim.py` 与两臂配置。数值门槛对应：四窗 CTE RMS +0.03 m、P95 +0.05 m、peak +0.10 m、heading P95 +1°、速度误差 RMS +0.10 m/s、平均速度下降最多 0.20 m/s、横向加速度 P95 +10%；左右目标窗 emitted rate P95 各降至少 20%、physical jerk 严格降低；S 两窗 rate +20%、jerk +10%；26966 固定 post10m CTE RMS 不升，其他三窗恢复不新增 censor 且时间 +0.10 s。原 G2 必要条件另读归档 gates，不被转弯结果代替。

主判定只选 `subset=all`，moving≥2 m/s 仅辅助。参考线逐点一致检查、独立 nearest-segment 站距、左正 CTE、5 m 居中参考航向、世界加速度先差分再投当前 right vector 的 jerk，以及相邻帧实际 dt 的 emitted rate 均与协议一致。每窗有效值、进入/退出及全帧完整性作为前置条件；26966 recovery censor 显式保留，不假报恢复成功。两配置唯一差异为 `aim_interpolation`。

首例前发现一项应补的时间故障归类：原 `invalid_control()` 检查了 `timestamp_discontinuity`，却漏实际 Controller 发出的 `motion_gap`、`time_regression`；建议也明确涵盖 `future_trajectory`、`trajectory_outside_history`、`duplicate_tick`。已发 root/report 修正，原因是 frame 连续不能单独证明 sim_time 契约有效。合法 `stationary_trajectory` 和 `trajectory_behind` 保留为单列安全输出。此审查不修改控制器或分析源码；最终冻结版需另核对这些更正。

复算闭环结果时仍需核对每个必要失败数值、分母和全帧覆盖；当前仅是运行前代码/协议审查，不是六例结果结论。
