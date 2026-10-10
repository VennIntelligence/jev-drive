# LOWDIAG2: seed spread of the decision-237 failure taxonomy and the decision-239 lead-path read

Written 2026-10-10. Pre-registration: amendment 3 of [plans/2026-10-10-lowdiag-prereg.md](../plans/2026-10-10-lowdiag-prereg.md) (written and pushed
before any new-seed read; definitions, priority order, amendments 1 and 2 and all scripts frozen). Same scripts as decisions 237 / 239
(`lbd_pai_x.py`, `lbd_pai_replay.py`, `lbd_pai.py`, `lbd_hugsim.py`, `lbd_hlead.py`); the only edits are plumbing (output root / dir / arm names via
environment variables, `--tag` of the replay is the seed's checkpoint). New scripts: `lbd2_pai_chain.sh`, `lbd2_pai_post.sh`,
`lbd2_hlead_chain.sh`, `lbd2_pai_seeds.py`. No training, no tuning, no new class or rule.

## What was run

| part | seeds | source |
|:--|:--|:--|
| PAI served (`JEV_VCONT=1.0 JEV_LEAD=1`), 60 scenes, six 10-scene chunks | `P2H10-F-s0` (decision 237's table, unchanged) | stored |
| | `P2H10-F-s1` | stored: FIX1's `ab_*_s1`, zero-score `rollout.asl` still on disk (29 zeros) |
| | `P2H10-F-s2`, `-s3` | **re-run**: BODY1's runs of these two seeds had their rollout logs pruned (summaries only). 12 stacks with the identical command (`pai_native.sh`, CONC 4, same chunk files), zero-score logs kept |
| forced-command replay + shipped-weights replay on every zero with a kept log | s1, s2, s3 | one pool job per seed (`lbd2_pai_post.sh`: extraction, replay, unit table) |
| HUGSIM 64 `spec_plan_smooth` taxonomy | `SH30-F-s1` | stored run (HD 0.4338) |
| HLEAD repeat (`jevdrive.bench`) | `SH30-F-s1` `op_lead` on (new) and off (new repeat `hlead-off`); `SH30-F-s0` `op_lead` on once more (`rhlead2`) | 3 bench runs |

The re-run PAI seeds do not reproduce BODY1's scores bit for bit (closed loop is not bit-reproducible): s2 0.3725 here against 0.3558 in decision 235's
table (31 against 30 zeros), s3 0.3470 against 0.3369 (32 against 32); s0 (0.3590) and s1 (0.3661) are the stored runs and equal decision 235's.
The four-seed mean here is 0.361 against 0.3545 there. This difference (0.017 and 0.010 on a seed) is the same size as the seed spread itself and is part of
what the tables below show.

## 1. PAI: seed spread of every decision-237 count

Per seed: [seeds_pai.md](seeds_pai.md) (generated, includes the full per-seed count table and per-scene tables), [seeds_pai_counts.csv](seeds_pai_counts.csv),
one row per rollout in `pai/units.csv` (s0), `pai_s1/units.csv`, `pai_s2/units.csv`, `pai_s3/units.csv`. Classes are the exclusive classes of decision 237
(other > longitudinal > route > clearance, amended definitions); `fast entry` = L2 without L5, as in decision 237's "4".

| quantity (60 scenes) | s0 (d237) | s1 | s2 | s3 | range | mean | s0 inside the other seeds' range |
|:--|--:|--:|--:|--:|--:|--:|:--|
| mean score | 0.3590 | 0.3661 | 0.3725 | 0.3470 | 0.347 - 0.373 | 0.361 | yes |
| zeros | 33 | 29 | 30 | 32 | 29 - 33 | 31.0 | no (top) |
| lost score | 38.46 | 38.04 | 37.65 | 39.18 | | | |
| **share of lost: longitudinal (zeros + progress loss)** | 0.428 | 0.448 | 0.389 | 0.336 | 0.336 - 0.448 | 0.40 | yes |
| of which zeros | 0.286 | 0.210 | 0.186 | 0.153 | 0.153 - 0.286 | 0.21 | no (top) |
| of which progress loss | 0.142 | 0.238 | 0.203 | 0.183 | 0.142 - 0.238 | 0.19 | no (bottom) |
| share of lost: route | 0.234 | 0.210 | 0.266 | 0.255 | 0.210 - 0.266 | 0.24 | yes |
| share of lost: clearance | 0.182 | 0.184 | 0.159 | 0.255 | 0.159 - 0.255 | 0.20 | yes |
| share of lost: other | 0.156 | 0.158 | 0.186 | 0.153 | 0.153 - 0.186 | 0.16 | yes |
| zeros by class: other / longitudinal / route / clearance | 6 / 11 / 9 / 7 | 6 / 8 / 8 / 7 | 7 / 7 / 10 / 6 | 6 / 6 / 10 / 10 | | | |
| progress-loss rollouts | 12 | 18 | 16 | 14 | 12 - 18 | 15.0 | no (bottom) |
| corridor-flag zeros | 18 | 18 | 17 | 18 | 17 - 18 | 17.8 | yes |
| **overran the logged stop (L5)** | 5 | 5 | 5 | 4 | 4 - 5 | 4.75 | yes |
| **entered the turn too fast (L2 without L5)** | 4 | 3 | 2 | 2 | 2 - 4 | 2.75 | no (top) |
| **straight-road drift** (corridor sub-class 4, spins excluded) | 5 | 5 | 5 | 5 | 5 | 5.0 | yes |
| **command correct, plan does not respond** (sub-class 2) | 5 | 3 | 3 | 6 | 3 - 6 | 4.25 | yes |
| turn made late (sub-class 3) | 1 | 1 | 1 | 0 | 0 - 1 | 0.75 | yes |
| route information missing or wrong (sub-class 1) | 0 | 0 | 0 | 0 | 0 | 0 | yes |
| hand-over spin (O2a) | 4 | 6 | 6 | 4 | 4 - 6 | 5.0 | yes |
| L1 contact where braking clears | 2 | 0 | 0 | 0 | 0 - 2 | 0.5 | no (top) |
| served plan in the event 3 s before, of zeros with a kept log | 31 / 32 | 28 / 29 | 29 / 30 | 30 / 31 | | | |
| plan lead, median (s) | 1.8 | 1.7 | 1.85 | 2.2 | | | |
| shipped-weights plan clears the event (any) / speed / path | 6/29, 2/29, 4/31 | 4/26, 1/26, 2/28 | 4/26, 2/26, 3/29 | 6/28, 2/28, 5/30 | | | |
| same, longitudinal-class zeros (any) | 2 / 10 | 1 / 7 | 0 / 6 | 0 / 5 | | | |

By the first registration (no amendment 1 / 2: the overrun and spin units fall back into route / clearance; reconstructed from the final flags, which
reproduces decision 237's 2 / 8 / 14 / 9 on s0): other / longitudinal zeros / route / clearance = 2 / 8 / 14 / 9 (s0), 0 / 4 / 16 / 9 (s1), 1 / 3 / 17 / 9 (s2),
2 / 3 / 16 / 11 (s3). Under the first registration route is the largest zero class in all four seeds.

Corridor sub-class detail, command forced to the right value against forced straight, 4 s end point lateral difference: the inert units
(sub-class 2) show 0.03 - 0.40 m (s0), 0.01 - 0.18 (s1), 0.02 - 0.27 (s2), 0.03 - 0.89 (s3); `cmd_fed` equals the rule on the logged path in every
unit of sub-classes 1 - 3 of every seed. 213dfdac, the one "turn not made" unit of s0 (effect 2.32 m), has effect 2.8 / 1.74 m in s1 / s2 and 0.89 m in s3
(counted as inert there).

### Per-scene agreement (4 seeds, 60 scenes; [seeds_pai_scenes.csv](seeds_pai_scenes.csv))

| seeds that put the scene in the same class | 4 | 3 | 2 | 1 |
|:--|--:|--:|--:|--:|
| coarse (pass / progress loss / other / longitudinal / route / clearance) | 44 | 7 | 9 | 0 |
| fine (the decision-237 sub-classes) | 42 | 9 | 9 | 0 |

A scene is a zero in 4 of 4 seeds in 25 scenes, in 0 of 4 in 24 (so 49 of 60 never change side), in 1 - 3 seeds in 11. Recurrence of the key fine
labels (cells = scene x seed with the label):

| label | cells | scenes in >= 1 seed | in >= 2 | in all 4 |
|:--|--:|--:|--:|--:|
| overran the logged stop | 19 | 5 | 5 | 4 (1c7e2423, 21626256, 7a824ffa, b0fa4732; 9e3fd12d in 3, clearance in s3) |
| straight drift | 20 | 6 | 6 | 4 (6b986b30, 75c23f07, b988494a, fadc73da; 8457182d is a spin in s1 / s2) |
| command inert | 12 | 5 | 3 | 1 (605bf77a; 24a50fcc and c6c01d25 in 3) |
| fast turn entry | 11 | 4 | 3 | 2 (3a48e906, 59e085d7; 1d6e30bc in 2) |
| hand-over spin | 20 | 6 | 6 | 3 (05f35348, 48a3f74a, 69fc21e8) |
| turn made late | 3 | 1 | 1 | 0 (213dfdac in 3, inert in s3) |
| L1 contact | 2 | 2 | 0 | 0 |
| progress loss | 60 | 19 | 17 | 11 |

Per seed the counts are close, and for overrun, drift and spin they are the same scenes. The inert and fast-entry classes are partly the same scenes:
3a48e906 is both (it carries L2 and sub-class 2; by priority it counts as fast entry), so the two counts trade places between seeds.

## 2. HUGSIM 64: `SH30-F-s1` against `SH30-F-s0` (`spec_plan_smooth`, stored runs)

Tables for s1: [hugsim_s1/tables.md](hugsim_s1/tables.md), [hugsim_s1/units.csv](hugsim_s1/units.csv), [hugsim_s1/failed_units.csv](hugsim_s1/failed_units.csv) (s0 in `hugsim/`).

| quantity | s0 (d237) | s1 |
|:--|--:|--:|
| HD | 0.4432 | 0.4338 |
| ends: fg / bg / off_route / complete | 28 / 9 / 1 / 26 | 29 / 8 / 1 / 26 |
| other (O2b oncoming / crossing actor hit even when stopped 11, O2c rear-ended 1): runs, share of lost | 12, 0.311 [0.166, 0.457] | 12, 0.308 [0.165, 0.453] |
| longitudinal: runs, share | 13, 0.309 [0.172, 0.454] | 15, 0.331 [0.190, 0.478] |
| of which L1 (lead in the driven band, braking avoids) | 9 | 10 |
| of which **L2 (entered the turn too fast)** | 4 | 5 |
| clearance: runs, share | 12, 0.303 [0.160, 0.447] | 10, 0.260 [0.125, 0.400] |
| of which C1 / C2 | 7 / 5 | 7 / 3 |
| route: runs, share | 1, 0.010 | 1, 0.019 |
| complete with penalties, share of lost | 0.066 | 0.082 |
| other / clearance split with actors extrapolated to E + 3 s | 5 / 19 runs (0.126 / 0.488) | 6 / 16 (0.149 / 0.418) |
| lead head asks for >= 1.5 m/s^2 at least 1.5 s before E, L1 | 6 / 9 | 7 / 10 |
| ... first such step before E, median | 6.9 s | 4.0 s |
| served plan in the event 3 s before (failed runs) | 32 / 38 | 34 / 38 |

Per scenario the two seeds give the same fine class in 61 of 64 scenarios and the same end in 59 of 64; 37 scenarios fail in both, 39 in either.
The three-way tie (other, longitudinal, clearance 26 - 33% each, overlapping CIs) and route at 1 - 2% hold; the L1 count is 9 - 10, L2 4 - 5.

## 3. HLEAD repeat (`op_lead` on against off)

Tables: [hlead_s1/tables.md](hlead_s1/tables.md) (A0 = stored s1, A0r = s1 switch-off repeat, A1 = s1 on) and [hlead_s0rep/tables.md](hlead_s0rep/tables.md)
(A0 = stored s0, A0r = decision 239's switch-off repeat, A1 = the new second on-run `rhlead2`, A1first = decision 239's on-run). Paired by scenario, bootstrap over
scenarios (B 10 000, seed 0).

| | d239 (s0, on) | s0, on repeat | s1, on |
|:--|--:|--:|--:|
| HD A0 -> A1 | 0.4432 -> 0.4382 | 0.4432 -> 0.4381 | 0.4338 -> 0.4190 |
| A1 - A0 [95% CI] | -0.005 [-0.068, +0.049] | -0.005 [-0.068, +0.050] | -0.015 [-0.078, +0.037] |
| switch-off repeat against stored (same configuration twice) | +0.001 | +0.001 | +0.001 [-0.002, +0.003] |
| ends of A1: complete / fg / bg / off_route / max_steps | 25 / 24 / 10 / 1 / 4 | 25 / 24 / 9 / 2 / 4 | 22 / 24 / 9 / 4 / 5 |
| L1 units | 9 | 9 | 10 |
| L1 units still ending on their own lead | 0 | 0 | 0 |
| L1: complete / standing to max_steps (registered "fixed") | 1 / 4 | 1 / 4 | 0 / 5 |
| L1: later bg collision / struck by another actor / other | 2 / 2 | 2 / 2 | 3 / 2 |
| L1 class HD A1 - A0 | +0.155 [+0.036, +0.304] | +0.157 [+0.035, +0.309] | +0.106 [+0.046, +0.176] |
| original complete scenarios still complete (others: fg / off_route / bg) | 22 / 26 (2 / 1 / 1) | 22 / 26 (2 / 2 / 0) | 20 / 26 (2 / 3 / 1) |

The second on-run of s0 differs from the first on only in 1 of 64 end states (HD mean 0.4381 against 0.4382); the closed loop is nearly reproducible here.

The three scenarios that lost score in decision 239 (HD A0 -> A1, A0r within 0.05 of A0):

| scenario | d239 | s0 repeat | s1 |
|:--|:--|:--|:--|
| 090-hard-01 (hit head-on) | 1.000 -> 0.036 fg | 1.000 -> 0.036 fg | 0.996 -> 0.038 fg |
| 124-hard-01 (hit head-on) | 0.992 -> 0.074 fg | 0.992 -> 0.074 fg | 0.996 -> 0.061 fg |
| 053-medium-02 (off route after the lead leaves) | 0.975 -> 0.150 off_route | 0.975 -> 0.150 off_route | 0.989 -> 0.160 off_route |

On s1 two more scenarios lose by the same registered rule: 132384196576-medium-01 (0.993 -> 0.605, off_route) and 164701907483-easy-00 (0.633 -> 0.468, bg collision).

## What was not done

- The 23 BEVs the s0 stage-1 check looked at were not redone for s1 - s3; no unit of s1 - s3 was looked at by eye. The "3 of the 5 overruns have a red light in the model frames"
  of decision 237 was not re-read per seed.
- No P0 closed-loop arm on HUGSIM (still cancelled, as in decision 237). PAI non-zero rollouts keep no log (pruned), so progress-loss units are read from the driver's records only, as in s0.
- The L1 / L2 / lost-scenario mechanisms of decision 239 (lead reading against 2.0 m/s^2 cap) were not separated.
- No second repeat of `SH30-F-s1` on; the s1 on-run is one rollout per scenario.

## Reproduce

```bash
scripts/tmux_run.sh lbd2-pai bash experiments/lowboard_diag/scripts/lbd2_pai_chain.sh "2 3"          # re-run served s2, s3 (12 stacks)
scripts/tmux_run.sh lbd2-post-s1 bash experiments/lowboard_diag/scripts/lbd2_pai_post.sh 1 $DATA_DIR/runs/alpasim/fix1/paibox/runs
scripts/tmux_run.sh lbd2-post-s2 bash experiments/lowboard_diag/scripts/lbd2_pai_post.sh 2 $DATA_DIR/runs/lowboard_diag/pai2/runs   # and 3
LBD_OUT=<dir> .venv/bin/python experiments/lowboard_diag/scripts/lbd_hugsim.py extract --arms SH30-F-s1 --workers 4   # then report --arms SH30-F-s1
scripts/tmux_run.sh lbd2-hlead bash experiments/lowboard_diag/scripts/lbd2_hlead_chain.sh                                # 3 bench runs
LBD_OUT=<dir> LBD_FAILED=<hugsim_s1/failed_units.csv> LBD_ARMS='{"A0":..,"A0r":..,"A1":..}' .venv/bin/python experiments/lowboard_diag/scripts/lbd_hlead.py extract   # then report
.venv/bin/python experiments/lowboard_diag/scripts/lbd2_pai_seeds.py                                                      # cross-seed PAI tables
```
