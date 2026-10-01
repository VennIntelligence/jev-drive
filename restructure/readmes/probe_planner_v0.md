# probe_planner_v0: Frozen Qwen3-VL probe and planner v0

status: concluded
decisions: 1, 2, 3, 3b, 3c, 4, 5, 8, 9, 10, 12, 13, 14
index: pre-onset vision delta null (CI [-0.062, +0.030]); K >= 1024
key: jevdrive/probe_v0.py, jevdrive/planner_v0.py, jevdrive/planner.py, jevdrive/waymo_stage_a.py, jevdrive/labels.py, scripts/probe_v0.sh, scripts/planner_v0.sh, scripts/waymo_stage_a.sh

**Question.** With Qwen3-VL frozen, how much does vision add over an ego prior in a thin planner head?

**Conclusion.** Vision-over-ego at pre-onset falsified on Waymo train (CI [-0.062, +0.030], decisions 3d). Defaults: 800 px, K >= 1024, late fusion ADE -0.029 m (decisions 4-10).

**Read more.** research/frozen-vlm-planner.md, research/qwen-latent-driving.md, docs/waymo-e2e.md

<!-- files:begin -->
<!-- files:end -->
