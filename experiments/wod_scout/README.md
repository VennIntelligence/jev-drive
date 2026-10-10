# wod_scout: WOD-E2E capability map beyond speed

status: concluded
decisions: 242 (inputs: 131, 164, 168, 169, 175, 180, 195, 218, 236)
index: path-only +0.290 of the 1.400 val gap, in 30 frames; 479 rated frames is all the label supply
key: experiments/wod_scout/plans/2026-10-10-wodscout-prereg.md, experiments/wod_scout/results/q1_spotlight.md, experiments/wod_scout/results/q2_path_budget.md, experiments/wod_scout/results/q3_strata.md, experiments/wod_scout/results/q4_label_supply.md, experiments/wod_scout/results/q2_path_budget/budget.md, experiments/wod_scout/results/q2_path_budget/top30.md, experiments/wod_scout/results/q4_label_supply/fields.md, experiments/wod_scout/scripts/ws_val.py, experiments/wod_scout/scripts/ws_test.py, experiments/wod_scout/scripts/ws_board.py, experiments/wod_scout/scripts/ws_fields.py

**Question.** Before the WOD-E2E line is reopened: what is in the test-only Spotlight cluster, how much of WLG's val gap is not
longitudinal and where, how much sits in night / pedestrian / the clusters that lose on test, and which label sources for "what the
rater prefers" exist beyond the 479 val frames?

**Conclusion.** Measurement only, 0 card-hours. Path alone returns +0.290 [+0.166, +0.432] of the 1.400 gap (speed alone +0.657), all of it in
30 frames, half of it low-speed turn geometry already described by decisions 169 / 175; night, pedestrians and the clusters that lose on test are
speed, not path; Spotlight membership is not available to us and the test inputs are not separable from val; WOD-E2E has no box or map on any of its
725 799 frames, no rater label outside the 479 val frames, and no released data pairs a computed label with rater scores (decision 242).

**Read more.** [results/q1_spotlight.md](results/q1_spotlight.md), [results/q2_path_budget.md](results/q2_path_budget.md),
[results/q3_strata.md](results/q3_strata.md), [results/q4_label_supply.md](results/q4_label_supply.md),
[plans/2026-10-10-wodscout-prereg.md](plans/2026-10-10-wodscout-prereg.md); Chinese page `research/wod_scout/index.html`.

<!-- files:begin -->
<!-- files:end -->
