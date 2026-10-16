# JOINT-DAGGER route check: does the route memory the model sees match the path the expert label follows? (2026-10-16)

Question (main session, after the Amendment 1 pilots in [pilot.md](pilot.md)): even 13 DAgger rows per batch for 1 000 steps collapse route reliance
(G-rely d_junction +0.025 -> +0.001 .. +0.005). Hypothesis: on the DAgger rows the route memory fed to the model and the path the expert label
follows are different paths, so the gradient teaches the model to ignore the route. Alternatives: they agree and the collapse comes from
elsewhere, or a pack / feature join bug pairs rows with the wrong route features.

Code: [scripts/rf_jdroute.py](../../scripts/rf_jdroute.py) `read` (CPU pool job, 21 s). Box output `$DATA_DIR/runs/route_follow/joint_dagger/route_check/`
(`rows.npz` per row, `route_check.json`, `join20.txt`). Copies here: [route_check.json](route_check.json), [route_check_join20.txt](route_check_join20.txt).

**Verdict: no mismatch, no join bug.** The fed route memory and the expert's path are the same polyline on nearly every DAgger row
(median lateral gap a few mm). On the trained junction rows the exits split on 2.8 % of rows. The labels lie *closer* to the route memory than
the policy's own served plans on 95-97 % of rows. So the DAgger gradient does not push the plan away from the route, and the hypothesis is
refuted. The reliance collapse must come from something else (open; see the last section).

## What was compared

All 13 992 DAgger rows (`labels.npz` valid: on1_L18RSG own rows k >= 2 with a clean joint target): 12 178 train-log rows (pool on1_JDGd),
1 814 dev-log rows (on1_JDGdev). Amendment 1's subset (junction segment or the 3 s before a failed turn) is 1 162 train / 114 dev rows.
Everything is in the row's rear-axle frame:

- **(a) route memory, as the trainer feeds it.** `feat/on1_JDG.npz` (pool d rows) or `feat/on1_JDGdev.npz`, f16 -> f32, rows with ok
  False zeroed. 40 slots at -6 .. +72 m of route arc from the ego's projection; old layout, no `--rm-order` (RSG-L18 / the pilots' sidecar).
- **(b) the expert pack's path.** Rule_expert's lane-chain centreline from `rf_expert.py pack`. The row's station uses rule_expert's own
  sequential projection: the scene's rows in k order, window [prev - 10, prev + 40] m.
- **(c) the joint target.** 8 poses, the trained `fut`; and the served plan (meta `poses`).
- **(d) the human log path.** meta `gpath`: s0 - 15 .. s0 + 70 m. It equals the pack's log where the log exists. The log reaches +40 m
  on 70 % of rows; the "which one is right" counts use only those rows.

Measures:
- "(a)-(b) lat." = distance from (a)'s slot at +10 / 20 / 40 m to the polyline (b).
- "Exit split" = the heading of (a) at its +40 m slot and of (b) at s0 + 40 m differ by > 30 deg.
- "Split and > 2 m apart" = the split rows where (a)'s +40 m point is also > 2 m off (b). A heading split alone can be an arc shift on the
  same geometry near a turn.
- Label / served distances = mean over the 8 poses of the distance to each polyline. Distances to (a) count only poses inside (a)'s covered
  span. Medians over rows.

## Join check (all rows; 20 printed)

- **Feature file vs source.** The fed memory equals `feat/on1_L18RSG.npz` for the same row name bit for bit (max |diff| 0). The
  Amendment 1 pool `feat/on1_JDGJ.npz` equals the full pool on all 1 162 subset rows (max 0). Feature `k` equals meta `k` on all
  13 992 rows.
- **Fed vs served memory.** The fed memory against the memory the driver actually served in the rollout (meta `rmem`): per-row max |diff|
  p50 0.0009, p99 0.0010 (f16 rounding). One row has a different valid mask and two rows differ by > 0.01.
- **Pool tab vs labels.** Tab `fut` = `labels.npz` target, tab `served` = meta poses, tab `speed` = meta speed, tab `log` = meta log:
  max |diff| 0 for both pools.
- **Pack frame and time.** The pack's log at t0 + 0 .. 4 s, put in the row frame with the meta anchor, against meta `logfut`: max 1.4 cm
  over all rows.
- **20 random rows** (seed 0; [route_check_join20.txt](route_check_join20.txt)). Same name in the tab, meta and feature file; same scene, k,
  log, anchor and speed. Re-running rule_expert with the stop rule at the meta anchor (the whole scene in k order) reproduces the stored
  expert plan exactly (max |diff| 0) on all 20.

**No join bug.**

## Route memory (a) vs expert path (b), and which one the human took

| set | segment | rows / logs | no slot | first slot med (> +18 m) | (a)-(b) lat. 10 / 20 / 40 m, median (p90) | > 3 m at 40 m | exit split (heading, > 30 deg) [CI] | split and > 2 m apart: rows; pack = human / memory = human |
|:--|:--|--:|--:|--:|:--|--:|:--|:--|
| A1 train | all | 1162 / 58 | 0.3 % | +24 m (68 %) | 0.03 (0.30) / 0.01 (0.78) / 0.00 (0.89) | 7.0 % | 59 / 1145 = 5.2 % [2.8, 7.6] | 36; 13 / 4 of 14 with the log at +40 m |
| A1 train | junction | 986 / 55 | 0.3 % | +24 m (69 %) | 0.01 (0.23) / 0.00 (0.30) / 0.00 (0.30) | 6.2 % | 42 / 972 = 4.3 % [2.0, 6.6] | 27; 10 / 0 of 10 with the log at +40 m |
| A1 train | approach | 94 / 27 | 0.0 % | +24 m (68 %) | 0.14 (0.32) / 0.07 (0.78) / 0.01 (5.37) | 21.5 % | 16 / 93 = 17.2 % [4.3, 31.8] | 9; 3 / 4 of 4 with the log at +40 m |
| A1 train | window | 398 / 47 | 0.0 % | +24 m (72 %) | 0.06 (0.83) / 0.01 (0.84) / 0.01 (1.63) | 8.2 % | 27 / 392 = 6.9 % [2.7, 11.6] | 12; 3 / 4 of 4 with the log at +40 m |
| A1 train | straight | 37 / 12 | 0.0 % | +14 m (38 %) | 0.06 (0.84) / 0.00 (0.83) / 0.00 (0.14) | 0.0 % | 0 / 35 = 0.0 % [0.0, 0.0] | 0; 0 / 0 of 0 with the log at +40 m |
| A1 train | other | 45 / 13 | 0.0 % | +24 m (71 %) | 0.07 (0.16) / 0.01 (2.53) / 0.02 (1.31) | 0.0 % | 1 / 45 = 2.2 % [0.0, 8.6] | 0; 0 / 0 of 0 with the log at +40 m |
| A1 dev | all | 114 / 7 | 0.0 % | +24 m (74 %) | 0.01 (0.09) / 0.01 (0.08) / 0.00 (0.06) | 0.0 % | 0 / 114 = 0.0 % [0.0, 0.0] | 0; 0 / 0 of 0 with the log at +40 m |
| A1 dev | junction | 99 / 7 | 0.0 % | +24 m (78 %) | 0.02 (0.09) / 0.00 (0.09) / 0.00 (0.02) | 0.0 % | 0 / 99 = 0.0 % [0.0, 0.0] | 0; 0 / 0 of 0 with the log at +40 m |
| A1 dev | approach | 5 / 3 | 0.0 % | +20 m (60 %) | - (-) / 0.00 (0.00) / 0.04 (0.12) | 0.0 % | 0 / 5 = 0.0 % [0.0, 0.0] | 0; 0 / 0 of 0 with the log at +40 m |
| A1 dev | window | 34 / 5 | 0.0 % | +24 m (76 %) | 0.01 (0.01) / 0.01 (0.01) / 0.00 (0.07) | 0.0 % | 0 / 34 = 0.0 % [0.0, 0.0] | 0; 0 / 0 of 0 with the log at +40 m |
| A1 dev | straight | 9 / 1 | 0.0 % | +16 m (33 %) | 0.01 (0.01) / 0.01 (0.01) / 0.01 (0.52) | 0.0 % | 0 / 9 = 0.0 % [0.0, 0.0] | 0; 0 / 0 of 0 with the log at +40 m |
| A1 dev | other | 1 / 1 | 0.0 % | +32 m (100 %) | - (-) / - (-) / 0.01 (0.01) | 0.0 % | 0 / 1 = 0.0 % [0.0, 0.0] | 0; 0 / 0 of 0 with the log at +40 m |
| full train | all | 12178 / 81 | 0.4 % | +20 m (54 %) | 0.00 (0.08) / 0.00 (0.15) / 0.00 (0.71) | 4.7 % | 477 / 11596 = 4.1 % [3.0, 5.3] | 334; 133 / 15 of 141 with the log at +40 m |
| full train | junction | 986 / 55 | 0.3 % | +24 m (69 %) | 0.01 (0.23) / 0.00 (0.30) / 0.00 (0.30) | 6.2 % | 42 / 972 = 4.3 % [2.0, 6.6] | 27; 10 / 0 of 10 with the log at +40 m |
| full train | approach | 1517 / 58 | 1.0 % | +22 m (59 %) | 0.00 (0.24) / 0.01 (0.96) / 0.02 (5.67) | 15.8 % | 240 / 1480 = 16.2 % [11.2, 20.9] | 189; 70 / 7 of 74 with the log at +40 m |
| full train | window | 398 / 47 | 0.0 % | +24 m (72 %) | 0.06 (0.83) / 0.01 (0.84) / 0.01 (1.63) | 8.2 % | 27 / 392 = 6.9 % [2.7, 11.6] | 12; 3 / 4 of 4 with the log at +40 m |
| full train | straight | 7757 / 81 | 0.1 % | +20 m (50 %) | 0.00 (0.04) / 0.00 (0.06) / 0.00 (0.25) | 2.3 % | 121 / 7297 = 1.7 % [0.9, 2.6] | 81; 38 / 5 of 42 with the log at +40 m |
| full train | other | 1918 / 65 | 1.2 % | +22 m (58 %) | 0.00 (0.16) / 0.00 (0.18) / 0.01 (0.59) | 4.7 % | 74 / 1847 = 4.0 % [2.1, 6.5] | 37; 15 / 3 of 15 with the log at +40 m |
| full dev | all | 1814 / 16 | 0.0 % | +18 m (50 %) | 0.00 (0.03) / 0.00 (0.08) / 0.00 (0.11) | 1.6 % | 11 / 1733 = 0.6 % [0.1, 1.3] | 7; 5 / 5 of 5 with the log at +40 m |
| full dev | junction | 99 / 7 | 0.0 % | +24 m (78 %) | 0.02 (0.09) / 0.00 (0.09) / 0.00 (0.02) | 0.0 % | 0 / 99 = 0.0 % [0.0, 0.0] | 0; 0 / 0 of 0 with the log at +40 m |
| full dev | approach | 138 / 10 | 0.0 % | +24 m (71 %) | 0.00 (0.21) / 0.00 (0.09) / 0.01 (0.14) | 0.7 % | 3 / 134 = 2.2 % [0.0, 7.8] | 2; 0 / 0 of 0 with the log at +40 m |
| full dev | window | 34 / 5 | 0.0 % | +24 m (76 %) | 0.01 (0.01) / 0.01 (0.01) / 0.00 (0.07) | 0.0 % | 0 / 34 = 0.0 % [0.0, 0.0] | 0; 0 / 0 of 0 with the log at +40 m |
| full dev | straight | 1371 / 15 | 0.0 % | +16 m (45 %) | 0.00 (0.02) / 0.00 (0.04) / 0.00 (0.07) | 0.7 % | 3 / 1307 = 0.2 % [0.0, 0.6] | 0; 0 / 0 of 0 with the log at +40 m |
| full dev | other | 206 / 12 | 0.0 % | +22 m (55 %) | 0.00 (0.04) / 0.00 (0.12) / 0.01 (0.22) | 9.3 % | 5 / 193 = 2.6 % [0.0, 8.1] | 5; 5 / 5 of 5 with the log at +40 m |

- **On the trained rows the two are one path.**
  - Amendment 1 train junction rows: median lateral gap 0.00-0.01 m at every distance, p90 0.30 m.
  - Exits split by heading on 4.3 % [2.0, 6.6] of rows, and are also > 2 m apart on 27 / 972 (2.8 %).
  - Dev rows: 0 splits.
  - At the trained dose (13 DAgger rows per batch) that is about 0.4 disagreeing rows per batch, too few to pull reliance from +0.025 to 0.
- **Where they do split, the expert is right.** Among split rows where the human log reaches +40 m, the pack's exit matches the human on
  10 / 10 junction rows, against 0 / 10 for the memory. Over all train rows the count is 133 against 15 of 141.
- **The largest disagreement is on the turn approach (16 % heading split).** These rows were trained only in the first four pilots (Amendment 1
  dropped them). There too the pack matches the human far more often (70 against 7 of 74).
  - Likely mechanism (not isolated here): the memory merges route batches and extends only along the last segment's direction. Near a turn it
    can keep a stale exit.
- **How the two routes are built.** They are the same AlpaSim MAP route, so they agree by construction:
  - (a) AlpaSim's RouteGeneratorMap projects every logged pose onto the lane centreline of a consistent lane sequence. It is built once per
    rollout from the human log and extended past the log's end along the lane graph. Each decision's message is the 20 points 40-80 m ahead
    of the ego's projection. `rm_build pool` regenerates them at the policy's own anchors and `rmem` merges them; this matches the served
    memory.
  - (b) `rf_expert.py pack` generates the same RouteGeneratorMap route along the log poses, prefixes the log before the route start, and
    matches a lane chain (rf_corridor). It keeps the chain where it lies within 2.5 m of the route and uses the route itself elsewhere.
  - Neither is built from the policy's path.

## Label and served plan against the route, the expert path and the human path

| set | segment | label to (a) / (b) / (d), m | served to (a) / (b) / (d), m | label 4 s pose inside (a)'s span | label closer to (a) / (b) than served | label 4 s arc / served (m) |
|:--|:--|:--|:--|--:|:--|:--|
| A1 train | all | 0.17 / 0.43 / 0.65 | 1.28 / 1.28 / 1.09 | 20 % | 97 % (n 185) / 95 % | 17.7 / 17.9 |
| A1 train | junction | 0.15 / 0.42 / 0.65 | 1.15 / 1.25 / 1.04 | 20 % | 96 % (n 156) / 96 % | 18.0 / 18.2 |
| A1 train | approach | 0.19 / 0.31 / 0.30 | 1.52 / 1.27 / 1.05 | 12 % | 100 % (n 11) / 89 % | 15.2 / 14.8 |
| A1 train | window | 0.28 / 0.57 / 0.93 | 2.34 / 2.06 / 1.71 | 14 % | 98 % (n 47) / 94 % | 16.4 / 17.0 |
| A1 train | straight | 0.82 / 0.85 / 1.54 | 2.78 / 2.54 / 1.71 | 27 % | 100 % (n 9) / 97 % | 17.0 / 11.6 |
| A1 train | other | 0.25 / 0.61 / 1.01 | 2.32 / 2.06 / 1.09 | 22 % | 100 % (n 9) / 98 % | 17.2 / 14.8 |
| A1 dev | all | 0.05 / 0.35 / 0.51 | 1.17 / 0.95 / 0.96 | 28 % | 100 % (n 17) / 97 % | 18.4 / 17.3 |
| A1 dev | junction | 0.09 / 0.38 / 0.55 | 1.08 / 0.99 / 0.96 | 24 % | 100 % (n 17) / 97 % | 18.4 / 17.4 |
| A1 dev | window | 0.01 / 0.38 / 0.68 | 1.15 / 1.49 / 1.79 | 24 % | - % (n 0) / 100 % | 17.2 / 15.8 |
| full train | all | 0.06 / 0.15 / 0.26 | 0.37 / 0.40 / 0.27 | 45 % | 94 % (n 4573) / 96 % | 18.5 / 17.4 |
| full train | junction | 0.15 / 0.42 / 0.65 | 1.15 / 1.25 / 1.04 | 20 % | 96 % (n 156) / 96 % | 18.0 / 18.2 |
| full train | approach | 0.10 / 0.20 / 0.25 | 0.56 / 0.49 / 0.31 | 25 % | 94 % (n 300) / 94 % | 14.7 / 12.9 |
| full train | window | 0.28 / 0.57 / 0.93 | 2.34 / 2.06 / 1.71 | 14 % | 98 % (n 47) / 94 % | 16.4 / 17.0 |
| full train | straight | 0.06 / 0.11 / 0.22 | 0.32 / 0.32 / 0.23 | 56 % | 94 % (n 3706) / 97 % | 22.6 / 20.6 |
| full train | other | 0.08 / 0.17 / 0.29 | 0.62 / 0.48 / 0.31 | 29 % | 92 % (n 411) / 96 % | 15.6 / 13.4 |
| full dev | all | 0.06 / 0.11 / 0.25 | 0.31 / 0.34 / 0.24 | 54 % | 95 % (n 839) / 97 % | 22.4 / 19.3 |
| full dev | junction | 0.09 / 0.38 / 0.55 | 1.08 / 0.99 / 0.96 | 24 % | 100 % (n 17) / 97 % | 18.4 / 17.4 |
| full dev | approach | 0.15 / 0.21 / 0.27 | 0.77 / 0.77 / 0.24 | 20 % | 86 % (n 22) / 96 % | 15.0 / 12.8 |
| full dev | window | 0.01 / 0.38 / 0.68 | 1.15 / 1.49 / 1.79 | 24 % | - % (n 0) / 100 % | 17.2 / 15.8 |
| full dev | straight | 0.06 / 0.10 / 0.22 | 0.30 / 0.29 / 0.21 | 63 % | 96 % (n 734) / 97 % | 26.0 / 22.1 |
| full dev | other | 0.06 / 0.25 / 0.33 | 0.44 / 0.63 / 0.35 | 36 % | 88 % (n 66) / 97 % | 16.4 / 16.8 |

- **The label follows the route, more closely than the policy does.**
  - On trained junction rows the label lies 0.15 m from the route memory (median, poses inside its span) and the served plan 1.15 m.
  - The label is closer to (a) than the served plan on 96 % of rows, and closer to (b) on 96 %.
  - The gradient on these rows therefore points *toward* the route geometry.
  - The label is 0.42 m from (b) over all 8 poses, because pure pursuit cuts the corner, and 0.65 m from the human path.
- **The memory does not cover the near field the label is about.**
  - On the trained rows the first valid slot is at +24 m (median); 69 % of junction rows have none before +18 m.
  - The label's 4 s pose lies inside the memory's span on only 20 % of junction rows (its 4 s arc is about 18 m).
  - This is not specific to the DAgger rows. The NAVSIM training rows (navtrain s0of12) have their first slot at +24 m (61 % beyond +18 m),
    and the S rows (on1_L18RB0) at +22 m (58 %).
  - The cause: the merged memory starts at the oldest message's first point, about 42 m ahead of the ego's pose at decision 0. So the first
    valid slot sits at about 42 m minus the distance driven since then.
  - So on most DAgger rows only the exit beyond about +24 m is in the memory. The 0-18 m shape the label teaches has to come from the frame.

## What this leaves

- Mismatch (refuted): (a) = (b) on 97 % of trained junction rows; the splits are rare and the expert is the one that matches the human.
- Join bug (refuted): every identity above holds bit for bit.
- Still open, not tested here: why 13 route-consistent rows per batch remove the route's marginal value on the NAVSIM dev junction rows that
  G-rely reads. Candidates:
  - the label's near-field shape is the lane centreline, visible in the frame and outside the memory's span;
  - the label's offset to the human path (0.5-0.65 m at junctions) is exactly the residual the route branch used to explain on human rows;
  - an optimisation effect of the `both` rows on the shared adapter.
  - Each needs its own read (for example G-rely of a pilot with the DAgger rows' route input masked, or with the route branch frozen).
