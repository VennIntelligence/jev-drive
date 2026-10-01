# real_transfer: Real-data transfer gates G0-G3

status: concluded
decisions: 44
index: No transfer: all 12 student zero-shot cells harmful (PDMS -1.1..-2.7)
key: jevdrive/real_g0.py, jevdrive/real_g1.py, jevdrive/real_g2.py, jevdrive/real_g3.py, scripts/real_g0_detect.sh, scripts/real_g0_navsim.sh, scripts/real_g1_score.sh, scripts/real_g3_detect.sh, scripts/real_g3_pdm.sh

**Question.** Can a student, gates, HUGSIM pairs or better edits bring the CARLA reaction to real data?

**Conclusion.** No (decisions 44): G0 all 12 cells harmful; G1 gates only shrink the harm (NAVSIM -1.61 [-2.14, -1.08]); G2 HUGSIM I3 retraining: no head beats the prior; G3 edit quality is not the bottleneck.

**Read more.** research/results/real-data-transfer/, `git show bcbdde4:todos/2026-09-26-real-data-transfer.md`

<!-- files:begin -->
<!-- files:end -->
