# fusion_diag: Qwen x openpilot complementarity and SAM 3.1 perception

status: concluded
decisions: 43
headline: Qwen + openpilot is redundant under imitation, complementary under paired-difference training; SAM rule gate flips 29.2%

**Question.** Before fusing, are Qwen and openpilot features complementary, and can off-the-shelf SAM 3.1 supply a structured hazard state (Q1-Q9)?

**Conclusion.** Under uniform imitation Qwen + openpilot is redundant, but under paired-difference training it is complementary (decisions 43, corrected in place after P5 v1: the original "drop Qwen" verdict was withdrawn). SAM 3.1 pedestrian recall on P5 hazards is 0.49 overall, 0.88 within 20 m; the geometric rule gate on SAM state flips 29.2% [13.2, 47.0], below the learned head's 43%, so "pedestrians go through the SAM channel" was withdrawn. Reaction-window p25 is 0.05-0.1 s for lights, so those stay in the fast channel.

**Read more.** research/leaderboard-vs-ability.md, research/behavior-layer-instruments.md, `git show bcbdde4:todos/2026-09-25-fusion-diagnostics.md`

<!-- files:begin -->
<!-- files:end -->
