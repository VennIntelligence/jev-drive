# B2D 特权上限结果

目前保存阶段调试证据，正式58条路线、两个交通seed（交通随机种子）、六臂的696次评测尚未开始。设计、固定判据与完整偏离记录见[实验记录](../../../todos/2026-10-01-b2d-privileged-ceiling.md)。

- `pilot_checks.json`保留27787原pilot（最小试跑）的失败，不由后续成功替换。
- `pilot_diagnostics.json`保留三卡并行诊断的全部结果；`pilot_supplement.json`说明以登记debug路线334补充真实停车/恢复验证的明确偏离。
- `debug_v3_checks.json`和`debug_v3_lock.json`记录修复版中间批的逐项检查与控制源码校验。通过实现通路检查不代表路线成功，路口仍存在卡死。
- `junction_hold_smoke.json`记录真实状态机的短验证：保持IDM（基于跟驰距离生成速度剖面的规则）、观测自有停车后放行、禁止绿灯授权释放未拥有的停车。

调试数字只用于工程诊断，不进入正式配对差或置信区间。录像、原始逐帧日志及大产物保存在box的`$DATA_DIR/runs/b2d_privileged_ceiling/`，本目录只收小结果文件。
