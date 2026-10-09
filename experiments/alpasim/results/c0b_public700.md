# C0b: five drivers on all 700 landed public AlpaSim scenes (2026-10-09, decision 201)

Local closed-loop runs on the box, native AlpaSim at 0bb4c4b; nothing was submitted to AlpaSim. Scenes: every scene of the seven asset shards that had a `.done` marker at 08:41 CST
(part001, 002, 003, 005, 007, 008, 009; 700 scenes from 27 nuPlan logs; list and shard map `$DATA_DIR/runs/alpasim/c0b/lists/`). Shards that landed later are not covered. The shard of a scene is
inferred: the extracted scene dirs form contiguous 100-scene blocks of the sorted names, the block extraction times match the markers (part008 finished last, its 5 late directories sit in the
first block of the last 200), so the 008/009 split is an inference, the others are exact up to the same rule.
Full generated read-out (all tables, the list of WA-JEPA zeros): [c0b_report.md](c0b_report.md). Per-scene table for other lanes: [c0b_per_scene.csv](c0b_per_scene.csv) / [c0b_per_scene.json](c0b_per_scene.json)
(per driver: score, zero class, progress, at-fault events, decisions, plan endpoint of decision 0); numbers in [c0b_stats.json](c0b_stats.json).
Code: `scripts/c0b_chain.py` (pool jobs, stall watchdog), `scripts/c0b_report.py`. Pre-registration of the OT30 line, committed before any OT30 closed-loop score was started:
[plans/2026-10-09-ot30-closedloop-prereg.md](../plans/2026-10-09-ot30-closedloop-prereg.md).

Drivers: `SH30-F-s0` (cold start `backwarp`), `AP2-AB-s0`, `OT30-F-s0`, `OT30-F-s1` (the SH30 driver with `SH30_TAG=OT30-F-s{0,1}`, checkpoints `runs/op_parity/runs/OT30-F-s{0,1}/ckpt-final.pt`, nothing else changed),
and WA-JEPA as a **reference row only** (released weights, shipped NAVSIM agent, unmodified, the d188 setup, **fp32**, cold rule `repeat`; fp32 was practical: 1.3 s per `drive`, three card-shared jobs of 233 scenes took 0.8 h each).
SH30 and AP2 reuse last night's 400-scene runs (d199) plus the 300 new scenes; before that, 24 scenes spread over the 400 were rerun for each driver and **all 24 scores and zero reasons reproduced exactly**
(the simulator is deterministic, d199), so no rerun of the 400 was needed. The 700 scenes of OT30 and WA-JEPA ran as 3 chunks each (stride 3 over the sorted list), 8 concurrent rollouts, one runtime per job.

## Result (700 scenes)

| driver | mean scene score [95% CI] | score 1 | score 0 | at-fault collision | offroad | left corridor | slow (0 < score < 1) | at-fault events | mean progress |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| SH30-F-s0 | 0.9140 [0.8845, 0.9366] | 555 | 50 | 19 | 13 | 18 | 95 | 32 | 0.948 |
| AP2-AB-s0 | 0.9223 [0.9003, 0.9404] | 598 | 49 | 21 | 12 | 16 | 53 | 33 | 0.987 |
| OT30-F-s0 | 0.9258 [0.9029, 0.9447] | 565 | 42 | 11 | 12 | 19 | 93 | 23 | 0.950 |
| OT30-F-s1 | 0.9335 [0.9050, 0.9558] | 580 | 38 | 11 | 10 | 17 | 82 | 21 | 0.959 |
| WA-JEPA (**reference**, fp32) | 0.9111 [0.8753, 0.9436] | 594 | 56 | 1 | 34 | 21 | 50 | 35 | 0.928 |

No zero of any driver has a class other than the three listed. CIs resample whole logs (27 clusters), so they are wide: the drivers are not separable from SH30 on the mean except OT30-F-s1.

| paired difference (per scene, log-clustered) | mean scene score [95% CI] | zeros A / B | at-fault collision zeros A / B |
|:--|:--|--:|--:|
| AP2 - SH30 | +0.0083 [-0.0077, +0.0254] | 49 / 50 | 21 / 19 |
| OT30-s0 - SH30 | +0.0118 [-0.0053, +0.0284] | 42 / 50 | 11 / 19 |
| OT30-s1 - SH30 | +0.0195 [+0.0009, +0.0390] | 38 / 50 | 11 / 19 |
| OT30 mean of 2 seeds - SH30 | +0.0156 [-0.0011, +0.0326] | 40 / 50 | 11 / 19 |
| SH30 - WA-JEPA | +0.0029 [-0.0379, +0.0455] | 50 / 56 | 19 / 1 |
| AP2 - WA-JEPA | +0.0112 [-0.0242, +0.0474] | 49 / 56 | 21 / 1 |
| OT30 mean of 2 seeds - WA-JEPA | +0.0185 [-0.0198, +0.0587] | 40 / 56 | 11 / 1 |

## OT30 against the registered line

Line: mean of the two seeds minus SH30 >= +0.010 with CI lower bound > 0, and at-fault collision zeros not higher.
- Mean difference +0.0156 [-0.0011, +0.0326]: the size passes, **the lower bound does not (-0.0011)**: not met. Seed 1 alone has a positive CI (+0.0195 [+0.0009, +0.0390]), seed 0 does not.
- At-fault collision zeros 11 and 11 against 19: met.
- Verdict by the registered line: **not a candidate** (one of two conditions fails, by 0.001 in the CI bound).

What the data say beyond the line (descriptive, not used to move it): the gain is in at-fault collisions (19 -> 11 / 11, at-fault events 32 -> 23 / 21), not in the classes the question named: offroad + left-corridor zeros are
31 for SH30 and 31 / 27 for OT30 (offroad 13 -> 12 / 10, corridor 18 -> 19 / 17). Per scene, OT30 s0 removes 22 SH30 zeros (12 collision, 5 offroad, 5 corridor) and creates 14 new ones (7 corridor, 4 collision, 3 offroad);
s1 removes 26 and creates 14. 30 scenes are zero in both OT30 seeds. The two seeds differ by 0.0077 in the mean, so the seed spread is of the size of the effect, and SH30 has one seed only.
Best-of-two of the two OT30 seeds gains +0.0133 over the better seed, which is what a selector gets from training-seed noise alone.

Cold start (read-out 4). Decision 0 has identical simulator state for all drivers: the 4 s plan endpoint of OT30 is 0.43 m from SH30's on average (p90 0.80 / 0.83 m), more than 3 m away in 1 (s0) and 2 (s1) of 700 scenes; AP2 differs
by 1.97 m (160 scenes beyond 3 m), WA-JEPA by 2.88 m (268). So the offline m = 1 drop of d198 does not show up as a large first-plan change in the simulator. Zeros do not end a rollout (every rollout of every driver has all 10 decisions), so the
"zero within 3 decisions" column of the generated report is 0 by construction and says nothing; whether OT30's remaining zeros start at the cold-start decisions was not read from the `.asl` logs.

## WA-JEPA (reference): where it sits and what its zeros are

0.9111, between SH30 (0.9140) and AP2 (0.9223) and below both OT30 seeds, none separable (CI +-0.04). It does not hold the 48-scene lead (0.978): that came from part001, where it scores 0.9777 again on the 100 scenes
(1 + 1 zeros), and part002 (0.9758); on part003 / 005 / 007 it is level with ours, on **part008 and part009 it is 0.829 / 0.831** against 0.90-0.96 for ours (16 zeros each).
Its zeros are 34 offroad, 21 left-corridor, 1 at-fault collision (ours: 19 collisions for SH30): a different failure profile, almost no collisions but twice as many offroad. 42 of its 56 zeros are scenes where none of our four drivers is zero;
6 are zero for all five (14 scenes are zero for all four of our drivers). 32 of the 56 sit in part008 / 009, 26 of them in three logs of 2021.09.16 (`13.53.10_veh-42` 11, `14.39.34_veh-42` 11, `15.47.30_veh-45` 4). Cause not investigated (no frame-level reading); the list with classes is in c0b_report.md.
fp32, as in d188; bf16 was not run. Label: reference, never a candidate.

## Per shard (mean scene score / zeros)

| shard | SH30 | AP2 | OT30-s0 | OT30-s1 | WA-JEPA (ref) |
|:--|--:|--:|--:|--:|--:|
| part001 | 0.9169 / 7 | 0.9101 / 8 | 0.9334 / 5 | 0.9189 / 7 | 0.9777 / 2 |
| part002 | 0.9250 / 6 | 0.9338 / 6 | 0.9338 / 5 | 0.9471 / 4 | 0.9758 / 2 |
| part003 | 0.9471 / 4 | 0.9461 / 5 | 0.9495 / 4 | 0.9703 / 2 | 0.9398 / 5 |
| part005 | 0.9333 / 6 | 0.9439 / 5 | 0.9428 / 5 | 0.9541 / 4 | 0.9305 / 6 |
| part007 | 0.8267 / 14 | 0.8641 / 12 | 0.8750 / 9 | 0.8606 / 11 | 0.8941 / 9 |
| part008 | 0.9046 / 8 | 0.9130 / 8 | 0.9093 / 8 | 0.9259 / 6 | 0.8289 / 16 |
| part009 | 0.9444 / 5 | 0.9452 / 5 | 0.9366 / 6 | 0.9573 / 4 | 0.8307 / 16 |

Part001 is not unrepresentative for our drivers: part001 minus the other shards is +0.003 [-0.072, +0.050] (SH30), -0.014 (AP2), +0.009 / -0.017 (OT30); for WA-JEPA it is +0.078 [+0.039, +0.116], i.e. its 48-scene lead was a part001 effect.
Part007 is the hardest shard for all of our drivers (0.83-0.87, 9-14 zeros).

## Best-of-k among our own drivers (per-scene maximum, an oracle selector; upper bounds)

| set | oracle mean [95% CI] | minus best single [95% CI] |
|:--|:--|:--|
| SH30 + AP2 | 0.9501 [0.9327, 0.9651] | +0.0278 [+0.0178, +0.0384] |
| AP2 + OT30-s1 | 0.9642 [0.9459, 0.9782] | +0.0307 [+0.0187, +0.0447] |
| SH30 + AP2 + OT30-s0 | 0.9695 [0.9556, 0.9807] | +0.0437 [+0.0322, +0.0586] |
| SH30 + AP2 + OT30-s1 | 0.9725 [0.9584, 0.9838] | +0.0390 [+0.0250, +0.0564] |
| all four | 0.9756 [0.9640, 0.9852] | +0.0422 [+0.0254, +0.0650] |
| OT30-s0 + OT30-s1 (one recipe, two seeds) | 0.9468 [0.9234, 0.9654] | +0.0133 [+0.0056, +0.0221] |

All pairs and triples: c0b_report.md. The SH30 + AP2 gain on 700 scenes (+0.0278) is close to the 400-scene value of d199 (+0.0292). The seed-noise reference (+0.0133) is about half of the cross-driver gains.

## Execution and cost

Staged launch: 8-scene pilots of WA-JEPA and OT30-F-s0 (driver counters clean: 80 `drive` calls, 0 inference and 0 input errors) and the 24-scene reproduction check, then everything (11 pool jobs, never a card or port picked by hand).
Wall 08:41-10:33 CST. Successful job time 6.1 job-hours (several jobs shared a card; wall on three cards 1.9 h) plus about 1.5 job-hours lost: at 09:23-09:27 a runtime worker was SIGKILLed (exit -9) in `wajepa-c0` and `ot0-c0`,
both on card 2 (cause not identified; load was 80 at the time and the other lanes' jobs shared the card), the runtimes hung as on the night of d199. The watchdog (20 min without a new rollout) would have cancelled them at about 09:45; I cancelled
at 09:40 through the pool after reading the log, and the chain resubmitted both from scratch (no resume in the runtime). Disk: 453 GB free at the start, 389 GB at the end (the downloads share it); outputs of this lane 4.9 GB after pruning
(`rollout.asl` is kept only for rollouts with a failure reason, i.e. the zero scores: 191 files in the new runs). Run dirs: `$DATA_DIR/runs/alpasim/c0b/runs/<job>/<ts>/`; SH30 / AP2 old part in `c0/sh30_400_rerun`, `c0/ap2_400_r1`.

## Limits

- One training seed for SH30 and AP2, two for OT30; one simulator run each (deterministic, so no repeat noise to measure). The SH30 control is a single seed: OT30 minus SH30 contains SH30's own seed variance.
- 700 scenes come from 27 logs; scenes inside a log are correlated, hence the wide log-clustered CIs. Seven shards of fifteen.
- Local MTGS rendering, not checked against the official Docker environment (no nuPlan reference scores); not a leaderboard number.
- Shard labels for part008 / part009 are inferred (above). The `.asl` logs of the zero rollouts of the new runs are kept for case review but were not read here.
- WA-JEPA zeros were not diagnosed; WA-JEPA flow noise is fixed by its config, so there is no second seed.
