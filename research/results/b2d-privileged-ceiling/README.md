# B2D 特权上限结果

目前保存阶段调试证据，正式58条路线、两个交通seed（交通随机种子）、六臂的696次评测尚未开始。设计、固定判据与完整偏离记录见[实验记录](../../../todos/2026-10-01-b2d-privileged-ceiling.md)。

- `pilot_checks.json`保留27787原pilot（最小试跑）的失败，不由后续成功替换。
- `pilot_diagnostics.json`保留三卡并行诊断的全部结果；`pilot_supplement.json`说明以登记debug路线334补充真实停车/恢复验证的明确偏离。
- `debug_v3_checks.json`和`debug_v3_lock.json`记录修复版中间批的逐项检查与控制源码校验。通过实现通路检查不代表路线成功，路口仍存在卡死。
- `debug_v4_checks.json`和`debug_v4_lock.json`是在修正未来车身中心之后重新跑13单位的检查与控制锁定，是当前正式运行前使用的版本；所有适用检查通过，路口卡死与绕障碰撞仍作为真实驾驶失败保留。
- `junction_hold_smoke.json`记录真实状态机的短验证：保持IDM（基于跟驰距离生成速度剖面的规则）、观测自有停车后放行、禁止绿灯授权释放未拥有的停车。
- `readout_checks.json`核验相连障碍组只计一次机会、同对象连续接触合并、接触导致失败、无接触且完成回正导致成功；用已知差值表检查整条路线连两seed的2000次bootstrap（聚类重抽样，seed0），并将三条调试记录的DS、RC与违规数逐项对齐官方记录。

调试数字只用于工程诊断，不进入正式配对差或置信区间。录像、原始逐帧日志及大产物保存在box的`$DATA_DIR/runs/b2d_privileged_ceiling/`，本目录只收小结果文件。
