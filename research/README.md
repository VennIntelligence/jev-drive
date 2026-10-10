# research/

叙事层：长期主题、综述和跨实验的分析，用中文写。单个实验的问题、结论和文件在
[experiments/INDEX.md](../experiments/INDEX.md) 对应的主题 README 里。
面向人的页面一律是 `research/<slug>/index.html`，按 [docs/html-reports.md](../docs/html-reports.md) 写成。
结论稳定后在 `docs/` 或代码里写英文正式版。

## 主题页
- [openpilot-diagnosis/](openpilot-diagnosis/index.html)：openpilot 为什么在各榜上表现不一，接口归因、能力缺口与接入矩阵。
- [four-directions/](four-directions/index.html)：四个方向的综合诊断：急弯内侧切角与入弯速度、宽弯擦边余量、前车近距测距偏差与夜间归属。
- [b2d-closed-loop/](b2d-closed-loop/index.html)：Bench2Drive 闭环里表征和控制哪个是瓶颈，双通道分裂与动力学实测。
- [benchmarks/](benchmarks/index.html)：四大基准地形、分数成分解构与榜单完整性（哪些分数能信）。
- [world-model/](world-model/index.html)：世界模型与反事实配对：因果效应闸门、WL-2 判格、no-go 汇总，以及分解世界模型（ego 精确、外生预训练）的设计与查新。
- [literature/](literature/index.html)：文献现状：薄 head、反事实与视频生成。
- [paper-outline/](paper-outline/index.html)：论文骨架 trick or trade：工业模型跨榜泛化与逐榜技巧归因。2026-10-08 起世界模型训练场一节废弃，候选主线见页首（决策 177）。
- [turn-gain/](turn-gain/index.html)：P2 急转弯失败是轻微比例缩小，不是封顶，控制链路不裁剪。
- [hugsim-specplan/](hugsim-specplan/index.html)：用模型自己的 plan 转向：单点曲率更差，0.5 到 1.5 秒平均曲率更好。
- [leaderboard-recipes/](leaderboard-recipes/index.html)：WOD-E2E、NAVSIM 与 AlpaSim 榜首方法归因：分数构成、大模型与世界模型的作用边界与可迁移配方。
- [alpasim-collisions/](alpasim-collisions/index.html)：AlpaSim 里的 at-fault 碰撞：nuPlan 公开集上是横向漂移，PAI 上是纵向且 lead 头提前看到了前车；PAI 的轨迹速度接口缺陷与诊断臂；31 个片段；第八节（SWV1，决策 220）：按 plan 自身扫掠足迹的碰撞分类与净空反事实。
- [body1/](body1/index.html)：BODY1 车身接触预测：冻结 token 上的 contact head 过离线闸门（AUC 0.905 / 0.978），刹停与单决策横向 re-plan 两个 serving 臂都在 one-chunk 检查处停，登记线无读数；loss 臂（补记 5）过 G3、闭环 L1 过但 L2 / L3 不过，不晋级。
- [wod_scout/](wod_scout/index.html)：WOD-E2E 重开前的侦察：Spotlight 成员不可知、纵向之外只改路径 +0.290 且集中在 30 帧、夜间与行人都是速度问题、评分员偏好的标签只有 479 帧（决策 242）。
- [corridor/](corridor/index.html)：冻结视觉特征上的转弯描述头：急弯 4 秒朝向 8.54 度对规划的 11.24 度，但误差不够独立（指标 0.513，线 0.538），登记的喂回试验没有做（决策 243）；越过判定线的探索性试验里三种接法都让朝向更准、分数不涨（决策 245）。

## 记录与工具
- [decisions.md](decisions.md)：跨 session 的决定，一行一条，全文在 `decisions/`。
- [next-round/plan.md](next-round/plan.md)：下一轮方向与计划（2026-10-09 改写）：论文线与比赛线分开，后果监督的轻量表征支路，判定线、日历。
- [next-round/overnight.md](next-round/overnight.md)：2026-10-09 夜的执行任务书：NC 失败分类、真值几何 oracle、离轨 warp 行。
- `lit/`：文献长报告，只在这台 Mac 上，不进 git。
- [plot_style.py](plot_style.py)：所有图共用的 plot style 模块。

## 写作惯例

- 自然语言段落讲清推理，主干中文、术语英文；术语每篇第一次出现加一句注释，缩写写全称。
- 结论和推测分开；推测写明用什么实验验证。
- 对比和结果用表格，写明 n 和 split，统一小数位，最好的加粗，表下一两句说读法。
- 图按 CVPR 标准：图内文字英文、不放标题；单栏 3.25 in / 双栏 6.875 in，8–9 pt serif；Okabe-Ito 配色，同一方法颜色固定，
  baseline 灰色；有 CI 就画并说明是哪种；PDF + 300 dpi PNG，都经同一个 plot style 模块（`research/plot_style.py`）。
- 图放所属实验的 `experiments/<主题>/figs/`，主题页的图放 `research/<slug>/figs/`，只提交 PNG（< 约 500 KB）和小的 WebP 审阅片段；
  PDF、checkpoint、feature 留在 box。每张图有文档收拢并用 1–3 句中文说明该看什么；改名、移动、删除时两边一起改。
- 文献长报告放 `lit/`，引用时摘进主题文档并注明 arXiv 号和年月。
