# real_transfer: Real-data transfer gates G0-G3

status: concluded
decisions: 44
index: No transfer: all 12 student zero-shot cells harmful (PDMS -1.1..-2.7)

**Question.** Can a student, gates, HUGSIM pairs or better edits bring the CARLA reaction to real data?

**Conclusion.** No (decisions 44): G0 all 12 cells harmful; G1 gates only shrink the harm (NAVSIM -1.61 [-2.14, -1.08]); G2 HUGSIM I3 retraining: no head beats the prior; G3 edit quality is not the bottleneck.

**Read more.** experiments/real_transfer/results/, `git show bcbdde4:todos/2026-09-26-real-data-transfer.md`

<!-- files:begin -->
## Files

- `real_g0.py` (jevdrive): Real-data transfer G0 and the shared …
- `real_g1.py` (jevdrive): a gate trained on real data times the …
- `real_g2.py` (archive): Real-data transfer G2 and G1c
- `real_g3.py` (archive): Real-data transfer round 2, G3
- `real_g0_detect.sh` (archive): Shared YOLO26x-seg detections of the …
- `real_g0_navsim.sh` (archive): Real-data transfer G0, NAVSIM column
- `real_g1_score.sh` (archive): Official NAVSIM scoring of the G1 gated …
- `real_g3_detect.sh` (archive): [G3]
- `real_g3_pdm.sh` (archive): [G3] 08:36

[archive/](archive/) 7 one-off code · [results/](results/) 82 result files · [figs/](figs/) 9 figures
<!-- files:end -->
