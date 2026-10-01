# 结构与链接验收

运行：`python3 out2/sources/validate.py`。脚本只读 CSV、PDF 元数据与公开 URL；不运行模型或评测。

| 文件 | 数据行 |
| --- | ---: |
| `w1_ablation_ledger.csv` | 1121 |
| `w2_cross_board.csv` | 204 |
| `w3_issues.csv` | 113 |
| `w4_attack_surface.csv` | 24 |
| `w5_paper_only.csv` | 45 |

- 必填字段与来源页码检查：1276 条本地 PDF 页面，0 项总错误（含下述 URL）。
- URL 检查：295 条含页锚引用，170 个不同 HTTP 资源，170 个 HTTP 200。

## 错误

无。
