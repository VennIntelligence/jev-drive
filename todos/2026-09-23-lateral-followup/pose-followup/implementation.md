# 可选后轴传播项：实现与验证

仅改动 `scripts/b2d_controller_adapter.py`、`scripts/b2d_agent.py` 和新建 `scripts/test_b2d_controller_pose_lateral.py`。未改变Controller、validator、转向限制或GNSS增益。配置 `pose_lateral_coefficient_s2_per_m` 透传到PoseFilter的 `lateral_coefficient_s2_per_m`，默认0，必须有限且非负；顶层与既有adapter嵌套写法都支持。候选固定0.010659832，没有重新拟合。根代理已审阅并本地提交 `b0830a6`，源码已冻结。

新增位移与独立定位重放完全一致：使用前一/当前SPEED均值的平方、world gyro均值和原传播中点yaw；负的right向位移先进入预测，再执行原向前传播与GNSS修正。k=0不向旧算式插入额外位移操作。首帧无区间，reset清空历史gyro；原速度死区、reverse/input契约和短compass dropout（短暂罗盘缺测）保持不变。非有限补偿抛出 `invalid_lateral_prediction`，发生在位置和运动历史写入之前。

新增遥测位于既有 `control.jsonl:pose_status`，字段及其首帧/disabled/applied/fault语义完整列在[冻结协议第7节](protocol.md)。`lateral_delta_xy_m` 是GNSS前的世界XY位移；diagnostics读取会复制列表，避免外部修改影响内部状态。正k有效区间即使位移为0仍标applied，含义是公式已经求值。

六项新增测试覆盖非法系数/默认精确golden、方向与GNSS衰减、旋转/平移/镜像、含溢出与时间异常的写入原子性、reset/缺测/静止，以及两种agent配置透传。完整controller测试 **125/125通过**，包含既有vendor/default、pose dropout及接入回归。实现后执行一次，不重复刷结果：

```sh
PYTHONDONTWRITEBYTECODE=1 /data/envs/carla/bin/python -m unittest discover -s scripts -p 'test_b2d_controller*.py'
# Ran 125 tests in 1.422s — OK
```

完整原始GPS/IMU/SPEED重放覆盖 **2761帧**：默认XY/yaw/raw逐值等于原记录；固定候选XY逐值等于实现前独立shadow，最大误差0；yaw和raw位置也保持逐值一致。[摘要/输入源码SHA](results/production-equivalence-v1/summary.json)、[原始输出SHA](results/production-equivalence-v1/outputs-sha256.json)均已由根代理复核。全部逐帧diagnostics和源码副本在 `/data/runs/b2d/controller/lateral-followup/pose-production-equivalence-v1`。

```sh
PYTHONDONTWRITEBYTECODE=1 /data/envs/carla/bin/python todos/2026-09-23-lateral-followup/diagnostics/verify_pose_lateral.py --run-root /data/runs/b2d/controller/turns-v1 --shadow /data/runs/b2d/controller/lateral-followup/pose-compensation-replay-v1 --out /data/runs/b2d/controller/lateral-followup/pose-production-equivalence-reproduction
```

上述验证确立实现等价和边界，不证明驾驶闭环收益。旧定位重放中左弯退化仍然有效，正式六例必须检验实际CTE与动作取舍。CARLA仅由根代理启动。
