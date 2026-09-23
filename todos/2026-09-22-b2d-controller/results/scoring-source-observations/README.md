# 评分边界的版本化源码观察

本目录保存本地Bench2Drive 0.0.4相关源码及SHA256，附官方固定commit链接。完整记录见[manifest.json](manifest.json)。本地字节快照与网页观察分别标记；直接urllib下载403，未声称官方远程字节hash匹配。源码观察不代表后续版本行为，不修改vendor。

- `statistics_manager.py`的penalty包括碰撞、红灯、停车标志、场景超时、让行紧急车辆、出界比例；`score_composed=max(score_route*score_penalty,0)`。MinimumSpeed本版本为unused。
- `efficiency_smoothness_benchmark.py`另算Driving Smoothness：按20样本片段检查加速度、jerk、yaw指标，满足全部阈值的片段比例，不进入上述DS公式。
- `_z_yaw_acc`调用savgol_filter未传deriv/delta，实际与平滑yaw rate调用相同，未显式求角加速度。
- `autonomous_agent.get_metric_info()`直接保存get_angular_velocity()；CARLA0.9.15官方API单位deg/s。smoothness读取该字段不转弧度，却标rad/s阈值，并对角速度调用角度unwrap。这是当前版本路径的单位/语义风险，未据此修改官方得分或声称测得实际舒适性。
- raw重算必须明确采样时间、完整运动学向量、滤波/分段/单位以及采用原版还是物理修订版。旧G2缺少完整字段，速度差分无法无损恢复官方metric_info；两种定义必须分开命名。
