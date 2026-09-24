# 第二轮独立复核

抽样统一使用 `random.Random(20260924)`，按各交付 CSV 原行序抽 `max(10, ceil(0.1 × 行数))` 条。每项由未继承工作上下文的新代理比对原文/代码/issue；原始抽查报告保存在 `out2/reviews/`。不一致项保留初次结果，并在更正后复核。

| 工作 | 初次样本与一致率 | 更正与最终验收 |
| --- | --- | --- |
| W1 | 1,121 条中抽 113 条，106/113=93.81%；详见 [报告](reviews/w1.md) | 7 条错误已逐项修正；[专项复核](reviews/w1_clusters.md) 将 BLUE、BEV-Planner++、DriveMA 的同类行一起校准。独立抽查达到 90% 门槛，通过。 |
| W2 | 204 条中抽 21 条，19/21=90.5%；10 组 Spearman 全部复算一致，详见 [报告](reviews/w2.md) | 已修正 CSV 行 96 的 PDM-Lite 规则 expert checkpoint 标记，并同步同方法及 PDM-Closed 行；行 188/157 的 SparseDriveV2 独立训练断言改为论文确证的 anchor 差异与 checkpoint 未知。补明官方 NAVSIM v2 论文 83/86 内部计数不一致及部分协议推定。独立抽查达到 90% 门槛，通过。 |
| W3 | 113 条中抽 12 条，10/12=83.3%，未过；详见 [初次报告](reviews/w3.md) | 第一轮错误已改；修订版抽 12 条，11/12=91.7%，详见 [第二次报告](reviews/w3_final.md)。第二次指出 #62 方法映射错误，已更正；35 条显式 `paper_locator` 经 [追加逐条复核](reviews/w3_citations.md) 全部可核。最终 91.7%，通过。 |
| W4 | 24 条中抽 10 条，9/10=90%；七页各核一处公式与盲区，详见 [报告](reviews/w4.md) | 已把 NAVSIM v1 的“跳过碰撞查询”改为“查询后忽略低速相交候选的 TTC 失分”，并同步修订 NAVSIM v2 页的同类措辞；更正项按报告代码行复核。独立抽查达到 90% 门槛，通过。 |
| W5 | 44 条中抽 10 条，6/10=60%；详见 [初次报告](reviews/w5.md) | 已更正 W5-004、W5-037、W5-041、W5-043，并对同类的 W5-025 降为中性评分头类别；新增 W5-045 拆分 DrivoR token 压缩。修订版 45 条中抽 10 条，10/10=100%，详见 [最终报告](reviews/w5_final.md)；通过。 |

## W5 初次错误与修订

- `W5-004`：导航输入误归输出表征。改为 `navigation_input_conditioning`。
- `W5-037`：NTR 论文未披露以榜单 RFS 监督评分头。改为 `learned_candidate_scoring`；同类 `W5-025` 也按论文自身披露程度下调。
- `W5-041`：把 Poutine-Base 无 CoT 消融误写成最终提交不生成 CoT。改为仅记录基座模式对照，并注明最终推理段仍描述低温 CoT。
- `W5-043`：把 DINOv2 初始化与 register token 压缩合为一项，且遗漏 HUGSIM 零样本用 NAVSIM-v1 模型的论文说明。现拆为 W5-043/045 并补明论文说法与公开代码核验边界。
- 最终复核另外建议四项不影响判定的精度改进：W5-001、W5-018、W5-036 补方法页与数值定位；W5-010 补跨行比较同时换视觉骨干的限制。已采纳，生成脚本与主 CSV 同步更新；抽样结论中的类别、数字和代码状态未改变。

## W4 初次错误与修订

- `AS-09` / CSV 行 10：原文把在 `pdm_scorer.py:460` 的相交查询误说成跳过查询。代码实际在查询后对低于 `0.005 m/s` 的候选跳过后续失分处理。已改 `w4_attack_surface.csv` 与 `w4/navsim_v1.md`，并同步精确化 `w4/navsim_v2.md` 的对应叙述。

## W3 初次错误与修订

- CSV 行 20，NAVSIM #62：提问者引用 Hydra-MDP 的 TransFuser Score 78.0（其 PDM 实现省略 DDC）与 NAVSIM 的 PDMS 84.0，未报告自己的复现分数。已去掉 `score_gap`，注明两篇已发表数字的实现口径未核同，并补两篇论文 URL。非维护者关于训练设置的推测保留为待核说明。
- CSV 行 9，carla_garage #47：原 `author_reply_url` 只支持 rendering bug 一句，不支持摘要中的碰撞、blocked 和单次评测；原论文 URL 也不含对应 Longest6 表。已改为作者详细回复 `#issuecomment-2377354886`，重取该评论原文短引，并以 TF++ 原论文 Table 6 作带版本限制的对照。
- 第二次独立复核仅第 20 行 #62 的 `related_methods` 与多余 `protocol_split` 不一致；现标明 issue 主体 TransFuser 与样本仓库关联 PDM-Closed 的区别，类别只留 `evaluation_config`。
- 两次报告指出部分论文数字只有 PDF 链接、缺表与页码。已对明确的论文陈述补 `paper_locator`，纠正 carla_garage #11/#46 的 TF++ 论文 v1/v2 与 Table 13 页码、BLUE #4 的 v1 勘误、TFv6 issue 用户引用的 95.28 与论文表 95.2±0.3 的归属；只在 issue 中出现的数字用 issue 作为出处。35 条显式定位已由新代理逐项核准。

## W2 初次错误与修订

- CSV 行 96：PDM-Lite 是规则式特权 expert，没有训练 checkpoint；已把它的五条记录全部改为 `not applicable`，并同步同类 PDM-Closed 的两条记录。
- CSV 行 188、157：SparseDriveV2 原注释断言 v1/v2 分别训练，所引论文 p.13 只支持不同的第二层 velocity anchor 个数（20/10）。现保留 checkpoint 身份 `unknown`，不再宣称独立训练。
- 同类自查：`LEAD` 的 Bench2Drive 分数在作者 issue #90 中被说明为估计；Longest6 列只按 Table 5 记录，不把 Bench2Drive 的估计状态套到该列。

## W1 初次错误与修订

- CSV 行 228、433：SparseOccVLA Figure 4(a) 和 TOAD Figure 4(b) 的端点数字直接标在图上，原账本误填 `N/A`；现分别录入 CIDEr `0.732→0.778`、EPDMS `38.5→48.5` 及差值，更新缺失值说明。
- CSV 行 562、569、571、599：BLUE 冻结基础 VLA backbone，但另训 gate；把无 gate→新 gate 比较标为 `retrained`，外来训练 gate 标为 `mixed_checkpoints`。专项复核覆盖同类行 557–574、599–600，并给规则/随机 gate 与同一 gate 阈值比较补限定语。
- CSV 行 608：UniAD Table 1 的 ID2 为 Official、ID3 为 Reproduce；已对 BEV-Planner++ 相关行 604–615 补来源混杂说明，跨来源行标 `mixed_checkpoints`。
- 非计分关联精度：DriveMA 行 242–253 的两个配对臂均含 `Rtraj=RFS`，新增的是语言一致性或 meta-action 奖励；移除这些边际行的 `REAL-WOD-001`，保留首次引入 RL/RFS 的 236–238 并注明联合变化。
