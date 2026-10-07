# driving_backbones: openpilot / Alpamayo as frozen backbones

status: concluded
decisions: 40
index: openpilot temporal -0.294 vs V-JEPA 2 -0.030, WOD pre-onset

**Question.** Do driving-trained features beat generic backbones in the P3 ladder and on nuScenes?

**Conclusion.** openpilot `temporal` better: -0.294 [-0.424, -0.168] vs V-JEPA 2 -0.030, replicates on nuScenes; Alpamayo mid-layer different, not better (decisions 40).

**Read more.** research/openpilot-diagnosis/index.html, research/decisions/055.md, `git show bcbdde4:todos/2026-09-24-driving-backbones/README.md`

<!-- files:begin -->
## Files

- `drive_backbones_alpamayo.py` (lib): Driving backbones in the P3 ladder
- `nusc_backbone_openpilot.py` (archive): openpilot as a frozen feature extractor …
- `nusc_ladder.py` (archive): the frozen-feature + ridge result …
- `make_driving_backbones_figs.py` (archive): cross-fit deltas vs `ridge ego` …

[archive/](archive/) 3 one-off code · [results/](results/) 48 result files · [figs/](figs/) 2 figures · [lib/](lib/) 1 library
<!-- files:end -->
