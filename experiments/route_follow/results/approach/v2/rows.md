# Approach course v2 rows (rf_apppool.py v2)

Pool `on1_APP` (unchanged rows and labels) + sidecar `approach_v2.npz`. Handoff rule: t_hand >= 1.5 s (nuPlan k >= 3, PAI k >= 15). Lateral target clipped at 0.75 m inside the route centreline.

| source | turn rows | past the handoff | speed rows v1 -> v2 (ego <= 26 m, past the handoff) | clipped targets (median clip m) | negatives | past the handoff | negatives with the symmetric speed term (human median m/s) |
|:--|--:|--:|:--|:--|--:|--:|:--|
| nup | 2379 | 1935 | 1255 -> 719 | 206 (0.37) | 595 | 511 | 292 (8.4) |
| pai | 712 | 629 | 365 -> 239 | 4 (0.88) | 178 | 177 | 152 (15.2) |
