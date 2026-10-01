# 真实 TCP paired-v2 成本归档

六组均已闭合，官方成功/失败独立列示；harness finished和returncode=0不等于驾驶成功。本文基于实际checkpoint inference，非无模型oracle。源码版本 `c09c9688d491859a0d8d9e97c5409a65670add12`。

|route|arm|官方结果|ticks|tick mean ms|GPU forward mean ms|attempt s|group s|
|---|---|---|---:|---:|---:|---:|---:|
|24211|native_common|Completed|393|51.538|4.089|24.9|25.629|
|24211|pi_common|Completed|401|57.516|4.540|32.7|33.435|
|1711|native_common|Completed|520|89.709|4.381|78.0|79.082|
|1711|pi_common|Completed|499|89.936|4.423|78.4|79.684|
|1773|native_common|Failed - TickRuntime|4000|118.095|6.555|504.9|505.735|
|1773|pi_common|Failed - TickRuntime|4000|129.479|7.252|548.7|549.580|

manifest.started→completed end共 **1280.562s（21.34min）**；六group合计1273.145s，记录的attempt合计1267.6s。**该attempt.wall_s被route_result覆盖，实际是evaluator区间**；独立route_start→route_end事件合计1272.738s，才是较完整的监督执行区间，逐例CSV保留两者。profile共9813tick，去掉每例前20tick后9693tick；保留样本mean×count约1114.416s。记录attempt减此数的153.184s包含warmup、地图/模型载入、setup/teardown和未覆盖工作，不能全叫“启动”或“warmup”。初始manifest→server-ready事件7.013s；group外合计7.417s。end事件发生在最终server.stop之前，因此这些总数不包括最终关闭server的耗时。

每例CSV保留profile mean/median/p95、world tick/agent/scenario tree等分项，以及visual性能文件实际提供的sensor_wait/preprocess/GPU forward/policy/preview mean和p95。visual去掉前20帧；本轮样本数是否与profile匹配逐例列出。GPU forward是CUDA event间隔，policy/preprocess/GPU范围有重叠，不能把这些均值相加当总耗时。未单独计时PI或native PID，故不编造控制器自身耗时。

本轮2例官方失败，合计8000tick和1053.6s记录attempt/evaluator成本（占该口径总量83.12%）；这些时间全部保留，可在summary.outcomes和逐例CSV直接复算；不把碰撞后运行当可删除成本。只报告本轮同机观测，不混入paired-v1的前四例或维护中止第五例，也不投射220路线。当前两臂共同移除官方低速尾部；相对历史官方尾部的更少tick不能归为PI加速，B/C差异也含闭环轨迹/速度与场景差异。

本轮实际attempt数6、额外attempt数0；group_start记录server PID集合[240758]，server日志集合数1。各setup记录CUDA_VISIBLE_DEVICES均为1：True。[运行中GPU快照](gpu-live.csv)保留物理GPU UUID及采样时renderer/TCP进程；这是一次采样，不声称全程监控。GPU身份请以该CSV为准。

- [全attempt逐例CSV](attempts.csv)
- [汇总与口径](summary.json)
- [原始输入路径/大小/SHA](inputs-sha256.json)
- [复算脚本](recompute.py)

```bash
PYTHONDONTWRITEBYTECODE=1 /data/envs/carla/bin/python recompute.py --run-root /data/runs/b2d/tcp-controller/paired-v2 --out /tmp/tcp-paired-v2-cost-replay
```

必须使用新输出目录。脚本先确认end=completed和六组顺序，再读取closed证据；不启动CARLA、不加载模型、不改运行源码。
