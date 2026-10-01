# driving_backbones: openpilot / Alpamayo as frozen backbones

status: concluded
decisions: 40
index: openpilot temporal -0.294 vs V-JEPA 2 -0.030, WOD pre-onset
key: scripts/drive_backbones_alpamayo.py, scripts/nusc_backbone_openpilot.py, jevdrive/nusc_ladder.py, scripts/make_driving_backbones_figs.py

**Question.** Do driving-trained features beat generic backbones in the P3 ladder and on nuScenes?

**Conclusion.** openpilot `temporal` better: -0.294 [-0.424, -0.168] vs V-JEPA 2 -0.030, replicates on nuScenes; Alpamayo mid-layer different, not better (decisions 40).

**Read more.** research/openpilot-openloop-standing.md, research/feature-adapter-domain-shift.md, `git show bcbdde4:todos/2026-09-24-driving-backbones/README.md`

<!-- files:begin -->
<!-- files:end -->
