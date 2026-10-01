# 第二轮榜单文本分析任务板

抽样基准：[第一轮 `sampling.csv`](../hack-audit/sampling.csv)，36 个榜单条目；C 档 14 个。状态枚举：todo / in-progress / done / verified。

| 工作 | 状态 | 子条目与交付物 |
| --- | --- | --- |
| W1 增益归因账本 | verified | [`w1_ablation_ledger.csv`](w1_ablation_ledger.csv)（1,121 条）；[`w1_component_codebook.md`](w1_component_codebook.md)（193 类）；独立复核 106/113，已改正错误并专项核查同类行 |
| W2 跨榜一致性 | verified | [`w2_cross_board.csv`](w2_cross_board.csv)（204 条）；[`w2_correlations.md`](w2_correlations.md)（10 组 n≥5）；独立复核 19/21，已改正错误 |
| W3 issue 挖掘 | verified | [`w3_issues.csv`](w3_issues.csv)（113 条）；[`w3_notes.md`](w3_notes.md)（20 仓）；修订版独立复核 11/12，35 条论文定位追加复核通过 |
| W4 榜单侧审计 | verified | [`w4/`](w4/) 七页；[`w4_attack_surface.csv`](w4_attack_surface.csv)（24 条）；独立复核 9/10，已改正错误 |
| W5 C 档纯论文审计 | verified | [`w5_paper_only.csv`](w5_paper_only.csv)（45 条，14 个 C 档）；[`w5_codebook_additions.md`](w5_codebook_additions.md)；初次复核错误已改，最终复核 10/10 |
| 独立复核与交接 | verified | [`review.md`](review.md) 五项均达 90%；[`questions_for_synthesis.md`](questions_for_synthesis.md) 汇总 45 题；[`HANDOFF.md`](HANDOFF.md) 一页导读；[`validation.md`](validation.md) 结构、页码和链接零错误 |

辅助过程文件在 `sources/`，独立抽查明细在 `reviews/`。[`validation.md`](validation.md) 记录必填字段、页码和 URL 验收。`expansion_log.md` 记录范围边界，`questions.md` 留材料取得或执行待决事项。
