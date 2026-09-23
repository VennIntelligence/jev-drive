# Agent 集成工作记录

负责：`b2d_agent.py`、`b2d_controller_adapter.py`、`test_b2d_controller_agent.py`。

## 已完成

- controller 模式每个 tick 读取精确 frame 的 GPS/IMU/SPEED，20 Hz 更新运动与 Controller.step；每 decimate tick 同步生成 route trajectory（实验参数需指定 decimate=4）。
- 相机按自己的 sensor_tick 到达；FrameRouter 保留未完成相机包，只交付同 frame 的完整 camera set。policy=none 不等待相机，避免在无图像的 tick 永久等待；尚不支持真实 planner。
- GPS 使用 dense GPS/world 对拟合 vendor Mercator 的 scale/translation，检查全部路线配对残差 ≤1 mm。不访问 map/OpenDRIVE。此 route 输入本身仍是诊断 oracle。
- 定位采用速度/gyro 预测、GNSS 位置修正 gain=.05、compass heading 修正 gain=.1；配置与 route_reference.json 显式保存。增益是开发参数，不冒称已通过实车定位验收。
- 测得后轴偏移从 config 强制读取；GNSS 安装 x=-1.4 m；world y-right → trajectory y-left，steer 已由 controller 输出为 CARLA 方向。
- RouteAdapter 在单调 progress 附近有界投影，带 heading 代价，保留完整 dense route；末尾沿切线延长 3 m，并采用 2 m/s² 终点减速，避免后轴停点在官方 actor 完成区之前。官方 completion 必须实跑核验。
- `control.jsonl` 每 tick 写命令、运动 frame、源 trajectory frame、估计 pose、cross-track、step_ms；`trajectories.jsonl` 保存所有真实执行的 trajectory；`route_reference.json` 保存可重建 route 和投影参数。
- 真值仅由独立 TruthLogger 读取同 frame snapshot，用同一个后轴基准计算；帧不一致显式报错而不产生错误指标，不进入控制状态。
- 核对 vendor evaluator 生命周期发现 `set_global_plan` 在 `setup` 之前。保存 dense GPS/world 后在 setup 初始化，另支持后续 set_global_plan/reset。已补独立回归。
- 保留已有 straight/route stub、overlap、TCP/runtime 路径。

## 验证记录

命令：

```
/data/envs/carla/bin/python -m unittest discover -s scripts -p test_b2d_controller_agent.py -v
DATA_DIR=/data /data/envs/carla/bin/python -m unittest discover -s scripts -p test_b2d_runtime.py -v
```

结果：adapter/agent **8/8**，原 runtime **9/9**。新增检查涵盖四 cardinal heading/左右符号/后轴、四个地理纬度和不同路线方向的 Mercator、固定随机 GNSS 噪声滤波、精确运动帧和迟到相机包、单调有界路线/终点延伸、truth frame 对齐、真实 Controller 的 truth 开关/伪造巨大真值不改变任何控制输出、每 4 tick update 与每 tick step、vendor route-before-setup 生命周期。

主代理实测证据：`/data/runs/b2d/controller/calibration/controller_config.json` 提供轴距 2.8604714913890885 m、rear offset -1.388633220199954 m、最大转角约 70° 与 steering_curve。`calibration-units` 实测 gyro z 与 CARLA world yaw 导数相关系数 +0.999946、比值中位 .9999977，因此边界符号转换得到证实；临时 steering curve 实验确认横轴 km/h。

## 仍需实测/局限

- 主代理独占 CARLA server；本子代理未启动 CARLA 或执行 live route。
- 位置滤波是轻量运动模型，碰撞后侧滑不能自动消除；不得用真值修正。
- 正常 cameras 是 optional 诊断输入；当前控制器只支持 policy=none，真实 planner 的 exact camera source-time 接口另做。
- 没有读取 criterion 碰撞时间；只有官方无时间 collision 字符串时，报告不得推算首次碰撞前区间。
- 延长终点与有界局部投影需要官方路线实跑验证；构造测试不能替代 official completion。

## 首次实车失败后的边界修正

只读审阅 development/1773/carla：660 tick 中 102 个 `invalid_motion`，来自停车状态 SPEED 的微小负浮点量（约 -2e-5 m/s）；首个速度 >.1 m/s 在 tick 61。终点已到时 GNSS 抖动让 ego→endpoint 的辅助线段产生 .5–.62 m/s 假目标速度，停车区域又出现 21 tick 油门 >.1。此 attempt 完整保留，不算通过。

同一数据定位 pose RMS=.13356 m、raw GPS RMS=.77545 m，truth lateral RMS=.03965 m，仅是该直路样本的描述性结果。

主代理确认 server 停止后修正：

- 传感器边界只把 `abs(raw_speed)<.01` 归零，记录 `raw_speed_mps`，真实倒退仍交 controller 报错制动。
- RouteAdapter 在终点剩余弧长 ≤.2 m、速度绝对值 <.1 m/s、估计 Euclidean endpoint 距离 ≤.5 m 时锁存停车，输出全零 trajectory，避免定位抖动重新加速；显式记录 `route_terminal_hold` 和 `route_endpoint_distance_m`。该行为是 route planner 的停车决策，不能用于证明真实停点精度。
- route reset 同时清空 tick/decimation phase、camera frame cache、trajectory frame、旧控制、pose filter、Controller、route parking latch。

新增边界/停车/完整 reset 回归，adapter/agent **10/10**。主代理随后将新建 development2 验证，旧 attempt 不覆盖。

## development2 独立审阅

12 个 route×preset 共 **5713 tick**：无 invalid_motion/stale/invalid_trajectory，无 motion frame 或 truth frame 错位，无 frame gap；所有 trajectory 最大 age=.1500000022 s。reason：tracking 4502、stationary_trajectory 1197、stop_hold 2、trajectory_behind 12（全部 TCP/26966）。后者发生在终点，truth lateral≈-.42 至 -.45 m，定位误差 .03–.15 m，endpoint 被投影到车后方，制动 fail-safe 生效。

9 个案例的 blocked 不是实际路径堵塞：validation 在 100 个低速样本（时间跨度只有 99×.05=4.95 s）时抢先判 blocked，距所需 5 s hold 尚差一个 tick；终点误差 .056–.234 m、实测 hold 位移 <.001 m。主代理已收到修改建议：hold_start 非空时不进入 blocked 检查。旧结果仍完整保留。

TCP/26966 的真正失败：lateral RMS=.7028 m、p95=1.5254 m，以及 cruise speed gate。Pursuit 四条 RMS≈.046/.099/.342/.111 m；26966 p95=.9827 m 接近 1 m 门槛，不能夸大余量。定位 p90 最差约 .3923 m（pursuit）。

终点 parking flag 比每 4 tick 的 trajectory 更新可能早 1–2 tick，因此 3 个案例有 terminal_hold=true 时旧 throttle 尚未替换；已提议停车状态切换即时更新一次 reference，常规更新仍维持每 4 tick。

TruthLogger 的 rear axle xy 真值投影增加 cos(pitch)，避免坡道 pose 误差偏差；新增 15° fake snapshot 检查，adapter/agent 10/10。此修改不会热更新 development2 已载入代码，后续 run 生效。独立 G2 validation 原本已按完整 forward vector 处理 pitch。

## 可复现出图与文章素材

新增 `scripts/b2d_controller_plot.py`，依赖当前 `/data/envs/carla/bin/python` 已有的 NumPy、Matplotlib，无 pandas/CARLA/network 依赖。

调用：

```
/data/envs/carla/bin/python scripts/b2d_controller_plot.py \
  --development /data/runs/b2d/controller/development3 \
  --campaign /data/runs/b2d/controller/dev10 \
  --out /data/runs/b2d/controller/figures-final
```

`--campaign` 可以是包含多个 preset/seed 子目录的总目录，也可直接指定一个 controller-report.json；目录模式递归读取已生成的 report。默认示例路线是预定的 26966，不按成绩自动挑选。

交付四类英文图，每类 PNG 300 dpi 与矢量 PDF：

1. 全部 G2 路线的 truth CTE RMS / pose p90，失败 case 用 hatch 标记。
2. Dev10 逐路线 completion/full truth CTE/precollision truth CTE，全部 attempt 写入 CSV；按 reporter 明确的 selected_attempt 画连接线，retry 仍以 x 显示，missing 留 NA。
3. G2 route26966 全部三个 preset 的完整真值后轴轨迹与原始 dense reference overlay，包括 TCP 失败轨迹。
4. 同一路线完整 speed/endpoint distance/throttle 时间曲线，标注第一次真实低速到达，没有裁掉不好的减速段。

全部图标注 `Diagnostic route oracle; policy = none`。五个相邻 CSV 保存每个绘制数值和轨迹/停车完整原始样本。plot-source-manifest.json 保存所有输入的绝对路径/SHA256/字节数、summary/trace JSON 快照、脚本自身完整源码快照及 hash、依赖版本、图与 CSV 的 hash/行数/列名、失败与缺失的保留规则。

已生成初始 `/data/runs/b2d/controller/figures` 和防覆盖检查用 `/data/runs/b2d/controller/figures-v2`，各 4 组图/5 CSV/无 warning；当前包含 G2 12case 与 CARLA seed0 Dev10 10attempt。已目视检查四类图，并验证 CSV 行数/hash。

用户要求保留中间产物后，CLI **拒绝非空输出目录**，exit 2，并明确要求新版本 `--out`。通过再次调用 figures-v2 验证拒绝覆盖且 manifest 字节完全不变；脚本 snapshot 字节与执行版本一致。后续请用 figures-v3/final 等新目录，不能覆盖初始图。

## 简明英文使用文档

新增 `docs/b2d-controller.md` 并在 `docs/README.md` 添加索引，约 880 words，覆盖纯 NumPy API、后轴坐标/时间节拍、MKZ 标定值、policy-none route oracle 与 truth 边界、origin bridge 造成 trajectory speed 不等于配置巡航速度、当前默认 carla 尚无合格替代、官方 4000 tick 上限与人工 cap 区别、单路 smoke/双 seed campaign/holdout/slope-check 命令、版本化 source snapshot/inventory/plot 归档。

仅引用已完成 G2 development3 与 smoke2390 的结果；最终结论链接工作目录 article-notes.md。通过只读 `--help` 核对新增 CLI，验证示例 Controller import 与所有文档链接，无 CARLA/git 操作。

注意 plot 对目录递归读取所有 report，包含 nested holdout；为避免把 holdout 混在 Dev10 标题下，文档示例用明确的单组 report 文件，并要求组合 Dev10 图使用仅含 Dev10 reports 的目录。当前任务未修改 plot/runtime core。

## Dataset 出图边界修正

按主代理新任务更新 plot discovery：目录模式只读取直接 `*-seed*/controller-report.json`，不递归进入 holdout/archive。新增 `--label`（默认 Dev10）；holdout 使用自己的根目录和 `--label Holdout`。输出名称改为 dataset-neutral `campaign-completion-tracking.{png,pdf}`、`campaign-all-attempts.csv`，其余 G2 素材名称不变。

临时最小 fixture 验证 primary root 排除 nested holdout/archive、holdout root 可单独发现、明确 report 文件可读取；3 项通过。仅检查 CLI 和来源发现，未重新生成任何 G2 图或覆盖已有 edition。同步修改 docs 使用说明。
