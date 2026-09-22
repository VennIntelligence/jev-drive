# 共享进展

## 2026-09-22：实施启动

- 旧工作树已备份 diff 至 `/data/runs/b2d/controller/baseline/pre-controller.patch`。
- 既有 runtime/sensor/preview 测试：Python 3.8 下 18 项通过。
- 评测使用 `/data/third_party/Bench2Drive` 的 `0.0.4`，仅物理 GPU 1，一个 server。
- 分工：controller 子代理负责纯 NumPy 数学和离线验证；agent 子代理负责定位/route/双频率集成；
  report 子代理负责 CLI 与结果汇总。主代理负责基线、车辆标定、无交互实车验证和实验执行。

结果尚未产生；以各子代理记录和本页后续更新为准。

## 基线、标定与首轮离线结果

此前的 smoke/cancellation 和 TCP/runtime 工作分别保存为 `0636c06`、`b1a436a`；共享目录为 `e2f61a9`。
发现远端 main 已有独立 Waymo 工作，已无冲突合并，保留双方历史，当前回到 main。

车辆参数来自 `/data/runs/b2d/controller/calibration/physics.json`：stock CARLA 0.9.15，Town10HD，
server port 5000，GPU UUID `GPU-b90dd90e-394b-7800-f23f-5892a8e3d0f1`。
轴距 2.86047149 m，后轴相对 actor 原点 x=-1.38863322 m，前轮最大转角 69.99999237°；
steering curve 为 (0,1)、(20,.9)、(60,.8)、(120,.7)。[参数与完整 physics](results/calibration-physics.json)。

再次标定 `/data/runs/b2d/controller/calibration-units/` 独立验证了单位和正负号：临时诊断曲线
(0,1)、(10,.1)，实际 2.00147 m/s、steer=.05，前轮平均角 1.22293°，与 km/h 横轴一致。
IMU gyro z 与 actor 的右正 yaw rate 相关系数 .999946，比例中位 .9999977。
诊断后恢复 stock physics；该临时曲线不进入评测。[单位探针](results/steering_unit_probe.json)、
[传感器符号](results/sensor-sign-check.json)。10 m/s 右转响应采样发生速度坍塌，未作为转向拟合数据。

12 项 controller 契约测试、8 项 agent 测试、10 项 report fixture 与既有 18 项测试，全套共 48 项通过。
[离线结果](results/offline-selftest.json) 中 pursuit/TCP/CARLA 圆弧 RMS 分别约 .0091/.1546/.4671 m；
CARLA 参考未通过原 .2 m 门槛，未修改增益隐藏失败。pursuit 近时段速度窗口停点误差约 -.0284 m，
减速 RMS .1712 m/s；动力学为 synthetic，不能替代实车结论。

## 首次 G2 实车开发

`/data/runs/b2d/controller/development/` 的 1773/carla 实际到达终点附近。汇总脚本因 nullable
target speed 比较失败而退出，全部 control/trajectory/validation trace 已保存；此轮标为开发失败。
发现静止时极小负速度导致 invalid_motion，终点 GNSS 抖动导致虚假再起步。agent 子代理正在修复
近零速度边界与 route 停车状态；controller 子代理正在独立修订 G2 指标和失败保存逻辑。
下一轮使用新目录，禁止覆盖第一次失败证据。
