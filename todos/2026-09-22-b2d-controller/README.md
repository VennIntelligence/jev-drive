# B2D controller 实践

状态：v1–v4开发、正式60条对照与坡道检查结束，证据已归档；没有合格的新默认配置，真实TCP新旧控制器对照尚未执行。用户已于2026-09-22授权多个子代理协作，集中保存中间记录和共享结果。

| 文件 | 用途 |
|---|---|
| [final-report.md](final-report.md) | 本轮最终结论、完整正式表、验收与下一步 |
| [plan.md](plan.md) | 已接受的方向、T0–T8 任务和 G1–G5 验收 |
| [iteration-v2.md](iteration-v2.md) | v2–v4闭环反馈、PI调节、舒适性与集成异常证据 |
| [progress.md](progress.md) | 主代理维护的进展、决策、阻塞和交付索引 |
| [article-notes.md](article-notes.md) | 文章素材、实测表、反例、图与原始来源；进行中结果明确标记 |
| [contract.md](contract.md) | 模块间接口和 telemetry 字段约定 |
| `agents/` | 各子代理独立维护自己的开发记录，避免并发覆盖 |
| `results/` | 可提交的 manifest、标定与实验摘要、小型 CSV/JSON |
| [results/raw-file-index](results/raw-file-index/README.md) | 已结束阶段的逐文件SHA256与原始路径，包括失败尝试 |

原始日志和逐 tick 数值数据（本轮未录制RGB或视频）：`/data/runs/b2d/controller/`。文件名标明阶段、preset 和 seed。
主代理独占 CARLA server 生命周期和 git 操作；各子代理只改分配的源码文件，交叉修改先沟通。
每项结果注明是否为解析计算、离线模拟或 CARLA 实测；失败结果保留。
每个阶段完成后向用户报告简短结果。原始输出、汇总和图片均使用新目录；已有中间版本不覆盖。


v1完整材料：[78个官方结果与81次attempt](results/v1-full/README.md)、[保留集图与来源](figures/v1-holdout/README.md)、
[坡道保持摘要](results/v1-full/slope-summary.json)、[真实S弯失败摘要](results/v1-full/s-curve-summary.json)。
两轮Dev10三组均9/10完成，保留集三组均5/6；坡道静止保持通过，新增S弯三组均未通过巡航速度gate。
详细数值、原始数据边界和v2关系见[文章素材](article-notes.md#v1完整归档保留集坡道与真实s弯)。

2026-09-23 起，本阶段 `results/` 下的逐帧文件（`*frames.csv`、`*samples.csv`、`*.jsonl`、`failure-analysis.json`）已移出 git，原样保存在 GPU box `$DATA_DIR/runs/b2d/controller/git-offload-v1/<仓库相对路径>`，校验和见同目录 `SHA256SUMS`。各 manifest 里记录的原路径和 SHA 不变。
