# 榜单分数与驾驶能力：综合判读（第 6 项）

状态: draft，待交给综合阶段的专家
主题: ../research/capability-vs-leaderboard.md

## 任务

两轮只读文本的材料已经收齐（都没有运行任何模型或评测）。请在此基础上做综合判读，写成一篇给人读的 research 文章，回答：

1. **高分从哪里来。** 按榜单分别说明：高分主要由哪些成分带来（正面的数据、训练、表示、结构），又由哪些针对榜单的机制带来（hack），
   两类各自的证据强度如何。对没用什么技巧也拿到高分的方法，要特别说明它们做对了什么。
2. **分数和驾驶能力的关系。** 哪些榜单、哪些指标的高分可以当作驾驶能力的证据，哪些不行；跨榜排名是否一致；各指标的攻击面和盲区在实际中被利用了多少。
3. **对我们的含义。** 我们把驾驶能力分成两层：R 层是 routine 驾驶（保持车道、转弯、跟车、按灯停车），E 层是突发事件时的反应（行人横穿、cut-in、转弯让车）。
   请把正面成分和 hack 机制分别对应到 R 层或 E 层；说明哪些可以直接当作配方移植，哪些值得从模型里提取；
   并指出我们自己的评测（CARLA 配对考试）需要防范哪些攻击面。
4. **材料里留下的问题**：逐条回应 `questions_for_synthesis.md` 里的 45 项，能判的给出判断和理由，判不了的说明还缺什么证据。

凡是推测都要标明是推测，并说明用什么证据或实验可以验证。实验只作为建议列出，不要执行。

## 材料（路径相对 repo 根目录）

**背景与框架**
- `research/capability-vs-leaderboard.md`：方向骨架（两层能力、TFv6 双通道证据、候选提取方法）
- `research/decisions.md` 第 25 条（reaction channel 假设）、第 31 条（TFv6 的 route + target speed 比 waypoint 高约 14 DS）、第 32 条（P5 配对考试）
- `research/prediag-2026-09/README.md`：预诊断轮的全部结果，其中 P5 一节是配对考试

**第一轮：hack 审计**（`research/results/hack-audit/`，Tokyo 上原路径为 `/data/hack_audit/out/`）
- `report.md`（总报告，先读这个）、`taxonomy.md`（19 类 codebook）、`findings.jsonl`（35 条发现）、`matrix.csv`（20 单元 × 19 类）、
  `sampling.csv`（7 榜 36 个条目）、`audits/`（每个单元一页）、`review.md`（独立复核）

**第二轮：文本分析**（`research/results/leaderboard-text-analysis/`，Tokyo 上原路径为 `/data/hack_audit/out2/`）
- `HANDOFF.md`（导读，先读这个）、`questions_for_synthesis.md`（45 个待判问题）
- W1 `w1_ablation_ledger.csv`（1,121 行 ablation）+ `w1_component_codebook.md`
- W2 `w2_cross_board.csv`（204 个已发表分数）+ `w2_correlations.md`（10 组 Spearman）
- W3 `w3_issues.csv`（113 条 issue）+ `w3_notes.md`
- W4 `w4/<board>.md`（七个榜单的计分代码、攻击面、盲区）+ `w4_attack_surface.csv`
- W5 `w5_paper_only.csv`（14 个无代码方法的 45 条论文披露）+ `w5_codebook_additions.md`
- `review.md`（五项独立复核，均 ≥ 90%）

所有文件中出现的 `/data/hack_audit/...` 路径都指 Tokyo box，对应关系见上。

## 交付

- 文章写在 `research/leaderboard-vs-ability.md`（中文，技术术语保留英文，写作惯例见 `research/README.md`：术语第一次出现时加一句注释，对比用表格，结论和推测分开写）。
- 对 `questions_for_synthesis.md` 的逐条回应可以作为文章附录，也可以单独写成 `research/results/leaderboard-text-analysis/synthesis_answers.md`。
- 如果结论会改变方向，在 `research/decisions.md` 里提议一条新条目，标**待定**。
