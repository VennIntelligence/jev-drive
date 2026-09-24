# 综合阶段导读

本轮只整理可追溯材料，不判断哪个成分最重要、分差是否代表驾驶能力或某个指标是否可信。抽样固定为第一轮 [`sampling.csv`](../hack-audit/sampling.csv) 的 36 个条目；没有扩大方法范围，见 [`expansion_log.md`](expansion_log.md)。未运行模型、仿真器或评测。

| 先读 | 内容与用法 |
| --- | --- |
| [`INDEX.md`](INDEX.md) | 五项工作状态与文件入口；[`review.md`](review.md)、[`validation.md`](validation.md) 分别给独立抽查与结构/链接验收。 |
| [`w1_ablation_ledger.csv`](w1_ablation_ledger.csv) + [`w1_component_codebook.md`](w1_component_codebook.md) | 1,121 个“配置对 × 指标 × 划分”行，按 PDF 表图与页码回查。`delta = variant − baseline`，不统一翻转越低越好的指标；`aux:` 是论文辅助协议。`PDM-Closed` 无独立方法消融。 |
| [`w2_cross_board.csv`](w2_cross_board.csv) + [`w2_correlations.md`](w2_correlations.md) | 204 个已发表分数；10 组样本数至少 5 的机械 Spearman 排序。按 `protocol/split` 与 `same_checkpoint` 过滤；跨论文同名方法不保证同权重或同计分版本。 |
| [`w3_issues.csv`](w3_issues.csv) + [`w3_notes.md`](w3_notes.md) | 20 个方法仓库的 issue/discussion 检索，113 个筛选记录，含用户说法、作者引文、论文表页定位与解决状态；用户猜测不等于事实。 |
| [`w4/`](w4/) + [`w4_attack_surface.csv`](w4_attack_surface.csv) | 七榜官方计分代码的固定提交、公式、条件性攻击面、盲区及第一轮 finding 对照。`未观察到` 只表示本轮代码支持的可能性没有对应首轮实例。 |
| [`w5_paper_only.csv`](w5_paper_only.csv) + [`w5_codebook_additions.md`](w5_codebook_additions.md) | 14 个 C 档方法的 45 条论文披露与代码公开状态；类别优先沿用第一轮 taxonomy。 |

需判读的问题集中在 [`questions_for_synthesis.md`](questions_for_synthesis.md)（W1–W5 共 45 项，逐项指向记录）。特别留意 NAVSIM v1.0/v1.1、v2.0/v2.2 的计分差异，NAVSIM v2 修复前后，Bench2Drive 与 CARLA 长短路线，以及 nuScenes 各论文的不同 L2 实现。W1 的缺失样本数、W2 的未知 checkpoint、W3 的未解决 issue 和 W5 无代码核验均按原表标记；不要把条件性代码机制或相关系数读成已实测提分或因果结论。
