# CLI 与报告子代理工作记录

状态：实现与离线回归完成；2026-09-22。未启动 server，未做 git 操作。

## 文件与接口

- `scripts/b2d_run.py`：新增 drive/controller_preset/cruise_mps/controller_config/tm_seed CLI 与 subprocess 透传；校验正有限 cruise、正 decimate、JSON object config；controller 暂仅 policy=none / built-in agent。独立 invocation manifest 保留 requested route 与 CLI（兼容共享输出目录的多次启动）。
- `scripts/b2d_route.py`：相同 CLI 校验；agent config 写绝对 controller JSON 路径、out、seed；记录 tick cap 是否实际触发。controller 模式包装本地 StatisticsManager 实例的现有统计调用，把既有 criterion 事件 frame/type/message/percentage 写 `criterion_events.json`，不改变 evaluator 判分或驾驶。
- `scripts/b2d_report.py`：保留既有 per-town stdout、`--json` 行为，新增 `--controller-json` 完整结构；CSV 所有 attempt 与未启动路线占位，展开主要 tracking/pose 指标。
- `scripts/test_b2d_controller_report.py`：10 个离线回归用例；CARLA 不需要启动。

## 报告语义

`status/harness_finished` 与 `official_status/completion/driving_completed` 独立；`results.json` 缺失、半写、空 records、多 records 不被判成功。所有重试保留；正式选择首个 harness finished，若无则最新 attempt，不能取最佳分数。请求但未启动路线保留在分母，attempt 数不计占位。配置 tick cap、实际 capped、partial 分列。

MinimumSpeed 事件数与 percentage 原样报告；固定 0.0.4 的模式标 unused，整体 penalty 读取官方 score_penalty，单项 penalty 未序列化则 null，不从事件推算扣分。

truth route cross-track、estimated route cross-track、controller diagnostic cross-track 三者分开。速度相对 command/reference 两种 RMS；raw/fused pose median/p90/p95、heading、age、controller CPU p99；所有指标保留样本数及 sum_squared 便于重算。truth frame 不匹配的定位/真值样本不计入精度。JSONL 半写、非单调帧、漏帧、传感器 frame 不匹配、无效/互斥控制分别计数；missing truth 不会落到 estimated。

首碰撞前指标使用 criterion snapshot frame；旧结果缺事件帧时明确 unavailable。全程、首碰撞前均给 tick 加权 RMS 与 route 等权 mean RMS；完整/未完成路线的 ticks 分布分开；low target-progress 连续 ≥5 s 片段单列。全部 attempt wall 与 retry wall 保留。

## 验证证据

执行：

```text
DATA_DIR=/data /data/envs/carla/bin/python scripts/test_b2d_controller_report.py
Ran 10 tests in 0.008s — OK
DATA_DIR=/data /data/envs/carla/bin/python scripts/test_b2d_runtime.py
Ran 9 tests in 0.216s — OK
```

最初 runtime 的 __new__ agent fixture 暴露 `set_global_plan` 过早读取 drive；已通知 agent 子代理修复，再跑 9/9 通过。

`agents/report-fixture-summary.json` 为合成输入结果：3 条请求路线、3 次真实 attempt（1 次重试）、1 条未启动；1 条 completion=100，另1条 finished 但 completion=31，观测完成率均值 65.5，不把缺失第3条当成功。文件显式标记 synthetic，不能当实测成绩。

## 边界

实验显式使用 `--decimate 4`：现有 CLI 默认 1 保持兼容。报告没有杜撰未记录的 setup/cleanup 分项或 stop-position；当前 cost 由已有 profile、wall、duration_game 与 control telemetry 给出。Criterion snapshot 在 evaluator 正常计算统计时抓取，强制 SIGKILL 的原始 event frame 可能仍不可得，此时报告 unavailable。尚无 live controller 数据，根代理负责实测整合、冻结默认与归档。

## 根代理追加分工：research修订

已更新 `research/trajectory-to-control.md`：修正aim_y与法向cross-track、CARLA加法lookahead、FF+完整bearing PID重复转向、绝对时间速度裁剪、CARLA/TCP离散语义、MinimumSpeed unused、finished≠completion与成本预算边界；替换原“半天/必能跑完”建议为共享计划G1–G4。写入实际MKZ轮距/后轴偏移/70°/km/h转向曲线，直接读calibration-units JSON核对；明确10 m/s右转sweep速度塌陷不可拟合，不宣称真实plant精确或Dev10已通过。
