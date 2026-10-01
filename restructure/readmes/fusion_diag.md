# fusion_diag: Qwen x openpilot complementarity, SAM 3.1

status: concluded
decisions: 43
index: Qwen+openpilot complementary under paired-diff; SAM gate flips 29.2%
key: jevdrive/fusion_q4.py, jevdrive/sam_detect.py, jevdrive/fusion_diag.py, jevdrive/fusion_q1.py, jevdrive/fusion_q2b.py, jevdrive/fusion_q4c.py, jevdrive/fusion_q6sam.py, jevdrive/fusion_q8.py, jevdrive/fusion_q9b.py, jevdrive/fusion_figs.py, scripts/fd_sam_chain.sh

**Question.** Are Qwen and openpilot complementary, and can SAM 3.1 give a hazard state?

**Conclusion.** Redundant under imitation, complementary under paired-difference training (decisions 43). SAM pedestrian recall 0.49 (0.88 within 20 m); rule gate flips 29.2% vs learned 43%.

**Read more.** research/results/fusion-diagnostics/, research/behavior-layer-instruments.md, `git show bcbdde4:todos/2026-09-25-fusion-diagnostics.md`

<!-- files:begin -->
<!-- files:end -->
