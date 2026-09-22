# 共享进展

## 2026-09-22：实施启动

- 旧工作树已备份 diff 至 `/data/runs/b2d/controller/baseline/pre-controller.patch`。
- 既有 runtime/sensor/preview 测试：Python 3.8 下 18 项通过。
- 评测使用 `/data/third_party/Bench2Drive` 的 `0.0.4`，仅物理 GPU 1，一个 server。
- 分工：controller 子代理负责纯 NumPy 数学和离线验证；agent 子代理负责定位/route/双频率集成；
  report 子代理负责 CLI 与结果汇总。主代理负责基线、车辆标定、无交互实车验证和实验执行。

结果尚未产生；以各子代理记录和本页后续更新为准。
