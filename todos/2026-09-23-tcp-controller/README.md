# 真实TCP纵向对照与独立转弯优化

2026-09-23，本阶段完成。**短前视候选未通过逐弯验收，默认保持carla/vendor。** 真实TCP纵向有局部行为信号，但障碍案例两臂都碰撞失败，不能声称安全或榜单提升。

从[总报告：把转弯单独验收](final-report.md)开始阅读；它汇总实施、反馈修复、完整验收、证据和下一步三个任务。

| 阶段 | 状态 | 协议与结果 |
| --- | --- | --- |
| 实际TCP接入与静止速度反馈修复 | 完成，旧中断和重放保留 | [v1发现](results/v1-discovery.md)、[重放](results/standstill-replay-v1) |
| 真实TCP三路线×两臂 | 六例闭合，两臂各2/3完成 | [冻结v2协议](protocol-v2.md)、[TCP报告](TCP-final-report.md) |
| 几何选取少数左/右/S窗口 | 完成，窗口未按控制误差重选 | [几何附录](turn-window-appendix.md)、[来源审查](agents/lateral-audit.md) |
| 短前视独立CARLA对照 | 六例全程基础通过，逐弯122/126通过，4项失败 | [冻结协议](turns/protocol.md)、[转弯结果](results/turns-v1/README.md)、[独立复核](agents/turns-final-review.md) |
| 本地集成、验证与归档 | 112+17测试通过，无默认升级 | [检查记录](results/merged-checks-v1/manifest.json)、[原始索引](results/raw-indices/README.md) |

- [全图版目录](figure-catalog.md)：历史及最终PNG/PDF，精确CSV和源哈希随图保存。
- [实际TCP成本](results/paired-v2-cost/README.md)：21.34分钟，失败停滞占主要成本，与oracle分开。
- [英文TCP使用文档](../../docs/b2d-tcp-controller.md)、[英文控制器文档](../../docs/b2d-controller.md)。
- [TCP报告冻结时的README/文本](results/paired-v2-report-freeze-v1/relocation.md)：旧audit引用的原始字节均保留；本入口随阶段结束更新。

主运行原始目录为`/data/runs/b2d/tcp-controller/paired-v2`与`/data/runs/b2d/controller/turns-v1`，早期paired-v1、preflight、replay、G1和所有分析图版保留。原始大日志不进git。固定[初版TCP协议](protocol.md)与[修订协议](protocol-v2.md)不改写。

真实TCP保持checkpoint、only_traj四点/2s、20Hz推理、原生横向；两臂共同移除官方尾部低速油门限制。横向六例是无背景交通的route oracle，不能当真实模型横向成绩。当前实验全部结束，没有自动追加参数搜索或反序确认。
