# 冻结验收与图表脚本

首例前完成：analyze_aim.py / plot_aim.py与本地依赖，最终准确副本frozen-v2、freeze-v2-manifest.json；首版冻结与验证均保留。原turns分析器未改。variant固定baseline-linear/candidate-hermite；四窗逐项验收，物理jerk按世界加速度先差分再投body，恢复censor完整保留。

正常运行：

```bash
/data/envs/carla/bin/python todos/2026-09-23-lateral-followup/analysis/analyze_aim.py --run-root /absolute/closed-run --out /new/analysis-edition --protocol todos/2026-09-23-lateral-followup/protocol.md
/data/envs/carla/bin/python todos/2026-09-23-lateral-followup/analysis/plot_aim.py --run-root /absolute/closed-run --analysis /new/analysis-edition --out /new/figure-edition
```

所有输出必须新目录。实际运行不能使用--legacy-fixture；此开关只为旧日志无新增mode字段的验证别名，manifest/conditions显式标记live_qualification_allowed=false，图有LEGACY ALIAS提示。requested必须严格等于预期；已求aim的used必须合法，Hermite退回linear必须有理由；safe无aim允许used=null。requested缺失、未求aim、实际fallback不混为false。

最终verification-v4：同baseline原数据两臂只出现4个预期主收益失败（左右各20%速率、各strict jerk），其他保护通过；主要窗口删一帧，expected70/observed69，case/coverage失败且比较证据不足。8项schema/时间故障/物理jerk测试通过；7PNG+7PDF已试绘，附3个完整曲线CSV。v1/v2/v3输出保留。frozen copies和验证hash均在manifest。

frames.csv保留逐帧mode/used/fallback、轨迹更新和aim字段JSON；metrics.csv逐窗计数，null未求值与有理由fallback分开。plot展示完整路线及四窗CTE、steer/raw/rate、速度、加速度/jerk、heading和update/mode/fallback事件；无平滑，不用moving子集判通过。

首例前独立审查落实motion_gap/time_regression/future_trajectory/trajectory_outside_history/duplicate_tick为输入/时间故障；未知或缺失reason判证据不足。合法stationary_trajectory/trajectory_behind仍单列，不新增性能门槛。最终helper从根代理获知完成后保持不变。
