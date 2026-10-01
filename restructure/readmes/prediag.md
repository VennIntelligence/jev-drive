# prediag: Pre-diagnostic round on Waymo E2E

status: concluded
decisions: 3d, 20, 21, 22, 23, 24
index: pre-onset vision delta null (CI [-0.062, +0.030])
key: jevdrive/waymo_l0.py, jevdrive/waymo_p0.py, jevdrive/waymo_p1.py, jevdrive/waymo_ladder.py, jevdrive/waymo_heads.py, jevdrive/waymo_qwenvid.py, jevdrive/dit_features.py, scripts/verify_top_decile.py, scripts/p4_wave2.sh

**Question.** Where does vision help beyond the ego prior at critical moments, and does another readout or backbone change that?

**Conclusion.** Pre-onset delta CI [-0.062, +0.030], DiD +0.111 (decisions 3d); no readout or backbone (Qwen-32B, Wan2.2, V-JEPA 2) buys it (decisions 20-24); Waymo heads fail on CARLA.

**Read more.** research/prediag-2026-09/README.md, research/p3-exam-filter.md

<!-- files:begin -->
<!-- files:end -->
