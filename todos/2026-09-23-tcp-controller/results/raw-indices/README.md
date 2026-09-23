# 原始证据完整性索引

所有运行均在结束后建立SHA256索引；旧索引不覆盖。`verification.json`保留首批四目录核验，`verification-v2.json`补全六目录，全部零不匹配。原始大日志保留在`/data/runs`，本目录是可提交的逐文件清单。

| 原始目录 | 文件数 | bytes | 索引 |
| --- | ---: | ---: | --- |
| `/data/runs/b2d/tcp-controller/paired-v1` | 162 | 15007021 | [paired-v1-inventory.json](paired-v1-inventory.json) |
| `/data/runs/b2d/tcp-controller/paired-v2` | 183 | 45695185 | [paired-v2-inventory.json](paired-v2-inventory.json) |
| `/data/runs/b2d/tcp-controller/preflight-v1` | 12 | 9580 | [preflight-v1-inventory.json](preflight-v1-inventory.json) |
| `/data/runs/b2d/tcp-controller/standstill-replay-v1` | 11 | 5096683 | [standstill-replay-v1-inventory.json](standstill-replay-v1-inventory.json) |
| `/data/runs/b2d/controller/turns-candidate-g1-v1` | 27 | 11551097 | [turns-candidate-g1-v1-inventory.json](turns-candidate-g1-v1-inventory.json) |
| `/data/runs/b2d/controller/turns-v1` | 127 | 11472524 | [turns-v1-inventory.json](turns-v1-inventory.json) |

共522文件、88832090bytes。包含旧TCP中断、修复后完整TCP、预检查、真实记录重放、横向G1与G2；不包含另存的分析/图版及此前v1–v4 oracle原始目录。

复核：`python3 todos/2026-09-23-tcp-controller/results/raw-indices/verify.py`。它只读原始日志，打印新验证结果；重新保存时使用新版本文件。不要向已经索引的运行目录追加文件。

TCP实际运行中GPU快照记录模型和renderer在物理GPU1；转弯目录的`gpu-live.csv`采集过晚、为空，`gpu-sample-note.json`明确其不是独立GPU验证。转弯仍保留启动环境、源码、端口和server日志。没有把空快照当成功检查。

图表需要的逐帧数值、事件、预测、控制和真值均保留；本轮没有完整RGB视频。后续文章优先引用各独立分析目录的精确CSV与PNG/PDF。
