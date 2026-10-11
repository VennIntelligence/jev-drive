# nuPlan 1491 scenes: P2H10-F, APY10m10-AB and P2H10S-F, three-way paired read (2026-10-11; descriptive, nothing promoted)

Question: which of three checkpoints goes into the submission image. Local closed loop (`sh30` driver for P2H10-F / P2H10S-F, `ap2` for APY10m10-AB), all 1491 public scenes (44 logs).
P2H10-F and APY10m10-AB seeds 0,1 are the CF1 runs ([cf1_confirm.md](cf1_confirm.md)); P2H10S-F seeds 0-3 on the 700 BODY1 scenes are the closed-loop read of
[shape_closed_loop.md](../../body1/results/shape_closed_loop.md), and on the remaining 791 scenes (`cf1fresh` 391 + `ot3new` 400, all 15 asset shards were on the box) they were run
for this read: 8 pool jobs (4 seeds x 2 lists), unchanged `sh30` driver (`SH30_TAG=<tag>`, no serving switch), box checkout `503d1345`, 791 / 791 rollouts each, no resubmission, no driver error.
Code: `scripts/nuplan1491_chain.sh`, `scripts/nuplan1491_report.py`; per-scene scores [nuplan1491/nuplan1491_per_scene.csv](nuplan1491/nuplan1491_per_scene.csv), numbers [nuplan1491/nuplan1491_stats.json](nuplan1491/nuplan1491_stats.json).
Per-scene scores are seed means (P2H10 / APY10m10 two seeds, P2H10S four; the "seeds 0-1 only" rows pair the same two seed indices); zeros and slow = seed means; mean progress = `progress_clipped_rel`.
CIs: scene bootstrap and log-clustered bootstrap (10 000 draws), the log-clustered one is the one to read. The first two recipes reproduce the CF1 tables exactly.

**Summary.** On all 1491 scenes P2H10S-F is 0.9243 against P2H10 0.9196 (+0.0047, log CI [-0.0016, +0.0115]) and APY10m10 0.9335 (P2H10S - APY10m10 -0.0092, log CI [-0.0160, -0.0016]); APY10m10 - P2H10 is +0.0139 [+0.0048, +0.0227].
P2H10S has the fewest at-fault collisions (7.5 against 11 / 12) and the most corridor zeros (27.25 against 19.5 / 22.5); APY10m10 has about half the slow scenes of the other two (136 against 262 / 232) and the highest progress (0.9637 against 0.9150 / 0.9251).

## All (1491): 1491 of 1491 scenes scored by all three, 44 logs

| recipe | seeds (mean) | mean [95% CI, logs] | zeros | collision | offroad | corridor | slow (0<s<1) | mean progress |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|
| P2H10 | 0.9192 / 0.9200 | 0.9196 [0.9024, 0.9387] | 87 | 11 | 44.5 | 19.5 | 262 | 0.9150 |
| APY10m10 | 0.9351 / 0.9320 | 0.9335 [0.9172, 0.9505] | 86 | 12 | 39.5 | 22.5 | 136 | 0.9637 |
| P2H10S | 0.9201 / 0.9251 / 0.9269 / 0.9250 | 0.9243 [0.9070, 0.9423] | 84 | 7.5 | 37.25 | 27.25 | 232.25 | 0.9251 |

| paired difference (per-scene seed means) | point | 95% CI scenes | 95% CI logs | per-seed-index differences |
|:--|--:|:--|:--|:--|
| P2H10S (4 seeds) - P2H10 | +0.0047 | [-0.0032, +0.0124] | [-0.0016, +0.0115] | +0.0010 / +0.0051 |
| APY10m10 (4 seeds) - P2H10 | +0.0139 | [+0.0040, +0.0238] | [+0.0048, +0.0227] | +0.0159 / +0.0120 |
| P2H10S (4 seeds) - APY10m10 | -0.0092 | [-0.0192, +0.0004] | [-0.0160, -0.0016] | -0.0150 / -0.0069 |
| P2H10S (seeds 0-1 only) - P2H10 | +0.0030 | [-0.0052, +0.0109] | [-0.0034, +0.0104] | |
| P2H10S (seeds 0-1 only) - APY10m10 | -0.0109 | [-0.0210, -0.0011] | [-0.0181, -0.0031] | |

## New (791: BODY1 read did not cover): 791 of 791 scenes scored by all three, 22 logs

| recipe | seeds (mean) | mean [95% CI, logs] | zeros | collision | offroad | corridor | slow (0<s<1) | mean progress |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|
| P2H10 | 0.8933 / 0.8951 | 0.8942 [0.8746, 0.9169] | 62.5 | 4.5 | 34 | 12 | 155.5 | 0.8934 |
| APY10m10 | 0.9190 / 0.9073 | 0.9131 [0.8913, 0.9366] | 61 | 5.5 | 29 | 14.5 | 78.5 | 0.9476 |
| P2H10S | 0.8950 / 0.9047 / 0.9031 / 0.8975 | 0.9001 [0.8789, 0.9238] | 61.5 | 3 | 28.5 | 18 | 132 | 0.9061 |

| paired difference (per-scene seed means) | point | 95% CI scenes | 95% CI logs | per-seed-index differences |
|:--|--:|:--|:--|:--|
| P2H10S (4 seeds) - P2H10 | +0.0058 | [-0.0059, +0.0178] | [-0.0040, +0.0166] | +0.0016 / +0.0096 |
| APY10m10 (4 seeds) - P2H10 | +0.0189 | [+0.0034, +0.0348] | [+0.0076, +0.0304] | +0.0256 / +0.0122 |
| P2H10S (4 seeds) - APY10m10 | -0.0131 | [-0.0280, +0.0020] | [-0.0198, -0.0043] | -0.0240 / -0.0026 |
| P2H10S (seeds 0-1 only) - P2H10 | +0.0056 | [-0.0068, +0.0181] | [-0.0051, +0.0181] | |
| P2H10S (seeds 0-1 only) - APY10m10 | -0.0133 | [-0.0289, +0.0024] | [-0.0201, -0.0040] | |

