# B2D controller 实践

状态：running。用户已于 2026-09-22 授权按方案实施，并要求多个子代理协作、集中保存中间记录和共享结果。

| 文件 | 用途 |
|---|---|
| [plan.md](plan.md) | 已接受的方向、T0–T8 任务和 G1–G5 验收 |
| [progress.md](progress.md) | 主代理维护的进展、决策、阻塞和交付索引 |
| [article-notes.md](article-notes.md) | 文章素材、实测表、反例、图与原始来源；进行中结果明确标记 |
| [contract.md](contract.md) | 模块间接口和 telemetry 字段约定 |
| `agents/` | 各子代理独立维护自己的开发记录，避免并发覆盖 |
| `results/` | 可提交的 manifest、标定与实验摘要、小型 CSV/JSON |
| [results/raw-file-index](results/raw-file-index/README.md) | 已结束阶段的逐文件SHA256与原始路径，包括失败尝试 |

原始日志、逐 tick 数据和录像：`/data/runs/b2d/controller/`。文件名标明阶段、preset 和 seed。
主代理独占 CARLA server 生命周期和 git 操作；各子代理只改分配的源码文件，交叉修改先沟通。
每项结果注明是否为解析计算、离线模拟或 CARLA 实测；失败结果保留。
每个阶段完成后向用户报告简短结果。原始输出、汇总和图片均使用新目录；已有中间版本不覆盖。
