# research/

叙事层：长期主题、综述和跨实验的分析，用中文写。单个实验的问题、结论和文件在
[experiments/INDEX.md](../experiments/INDEX.md) 对应的主题 README 里，主题自己的叙事文档也从那里链接。
跨 session 的决定记在 [decisions.md](decisions.md)（一行一条，全文在 `decisions/`）。
结论稳定后在 `docs/` 或代码里写英文正式版。

跨实验的文档：[survey-thin-head.md](survey-thin-head.md)（文献综述）、
[survey-counterfactual-video-gen.md](survey-counterfactual-video-gen.md)（反事实与视频生成综述）、
[openpilot-and-open-driving-models.md](openpilot-and-open-driving-models.md)（模型版图）、
[capability-vs-leaderboard.md](capability-vs-leaderboard.md)（方向骨架）、
[nohack-mechanisms.md](nohack-mechanisms.md)、[ablation-matrix-inventory.md](ablation-matrix-inventory.md)、
[midterm-inventory.md](midterm-inventory.md)、[midterm-gaps.md](midterm-gaps.md)；
`articles/`（深度长文）、`roadmap/`、`lit/`（不进 git）只在这台 Mac 上。

## 写作惯例

- 自然语言段落讲清推理，主干中文、术语英文；术语每篇第一次出现加一句注释，缩写写全称。
- 结论和推测分开；推测写明用什么实验验证。
- 对比和结果用表格，写明 n 和 split，统一小数位，最好的加粗，表下一两句说读法。
- 图按 CVPR 标准：图内文字英文、不放标题；单栏 3.25 in / 双栏 6.875 in，8–9 pt serif；Okabe-Ito 配色，同一方法颜色固定，
  baseline 灰色；有 CI 就画并说明是哪种；PDF + 300 dpi PNG，都经同一个 plot style 模块（`research/plot_style.py`）。
- 图放所属实验的 `experiments/<主题>/figs/`（综述的图放 `research/figs/`），只提交 PNG（< 约 500 KB）和小的 WebP 审阅片段；
  PDF、checkpoint、feature 留在 box。每张图有文档收拢并用 1–3 句中文说明该看什么；改名、移动、删除时两边一起改。
- 文献长报告放 `lit/`，引用时摘进主题文档并注明 arXiv 号和年月。

## 叙事与讨论材料
- [articles/](articles/README.md)：面向人的深度文章与入门；[roadmap/](roadmap/README.md)：讨论用路线图；
  [benchmarks-and-evaluation.md](benchmarks-and-evaluation.md)：benchmark 与评测方式总览。
