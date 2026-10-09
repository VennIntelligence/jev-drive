# alpasim OT2-B: off-track rows under the AlpaSim input standard (trimmed grid), 700 public scenes

Written 2026-10-09. Pre-registration with amendment 1: [plans/2026-10-09-ot2-dose-prereg.md](../plans/2026-10-09-ot2-dose-prereg.md) (the plan and the amendment were both pushed before any score below was read). Decision 210.
Code: `scripts/ap2_ot.py` (prep / train), `scripts/ot2_b_chain.sh`, `scripts/ot2_loop.py`, `scripts/ot2_report.py`. Tables: [ot2/b_report.md](ot2/b_report.md), `ot2/b_per_scene.csv`, `ot2/b_stats.json`. Nothing was submitted to AlpaSim.

## What was planned, what was run

Planned: AP2's recipe (strong hinge, lambda 30 / 0.5 m) + off-track rows on a 2 x 2 grid (amplitude +-0.5 m / +-1.5 m, batch share 10 % / 25 %), two seeds each, plus second seeds of the baselines; HUGSIM guardrail for the best cell; navtest / navhard and offline side reads.
Cut by amendment 1, after lane M1 found that the strong hinge itself triggers the closed-loop heading drift behind most zeros (decision 205): the four +-1.5 m trainings (never submitted), with them the amplitude axis; the HUGSIM guardrail and the side reads.
Run: the two +-0.5 m cells x 2 seeds, `AP2-AB-s1`, `SH30-F-s1`, all on the fixed 700 scenes. The +-1.5 m off-track rows were prepared in full and kept for lane OT3 ([ot2/ot2_cache_keys.md](ot2/ot2_cache_keys.md)).

## Answer

**By the registered lines `APO-a05m10` (AP2 + 10 % of +-0.5 m off-track rows) is a candidate: +0.0177 [+0.0077, +0.0277] over AP2, at-fault events halved. It is not final (no guardrail run) and it is not the best driver we have: the lambda-10 checkpoint P2H10 scores higher on the same scenes without any off-track rows. More rows are worse: at 25 % the score falls below AP2.**

| recipe (2 seeds) | seeds | mean scene score [95% CI] | zeros | at-fault collision | offroad + corridor | slow | at-fault events |
|:--|:--|:--|--:|--:|--:|--:|--:|
| SH30 (lambda 30) | 0.9140 / 0.9219 | 0.9180 [0.8896, 0.9394] | 46.5 | 18 | 28.5 | 96 | 31 |
| AP2 (baseline of the line) | 0.9223 / 0.9240 | 0.9232 [0.9008, 0.9416] | 48.5 | 20.5 | 28 | 53.5 | 32 |
| OT30 (SH30 + 10 %) | 0.9258 / 0.9335 | 0.9296 [0.9046, 0.9494] | 40 | 11 | 29 | 87.5 | 22 |
| **APO-a05m10** (AP2 + 10 %) | 0.9423 / 0.9394 | **0.9409 [0.9187, 0.9590]** | 36 | 8.5 | 27.5 | 56.5 | 16.5 |
| APO-a05m25 (AP2 + 25 %) | 0.9179 / 0.9065 | 0.9122 [0.8867, 0.9331] | 56.5 | 14 | 42.5 | 50 | 28 |
| reference, lane M1's table: P2H10-F-s0 / s1 (lambda 10, no off-track rows) | 0.9484 / 0.9496 | | 25 / 23 | 7 / 6 | 18 / 17 | | |

Counts are seed means. The P2H10 rows are copied from [m1/all700_report.md](m1/all700_report.md) (same 700 scenes, lane M1's runs; slow scenes and events are not in that table); they are not part of the line.

| registered line | APO-a05m10 - AP2 | APO-a05m25 - AP2 |
|:--|:--|:--|
| score: >= +0.015 and CI lower bound > 0 | **+0.0177 [+0.0077, +0.0277]: met** | -0.0109 [-0.0244, +0.0014]: not met |
| at-fault events not higher (recipe / AP2) | 16.5 / 32: met | 28 / 32: met |
| slow scenes <= AP2 + 10 | 56.5 / 53.5: met | 50 / 53.5: met |
| candidate | **yes** | no |

Other paired differences: APO-a05m10 - SH30 +0.0229 [+0.0079, +0.0394]; APO-a05m10 - OT30 +0.0113 [-0.0021, +0.0239]; OT30 - SH30 (now two baseline seeds) +0.0116 [-0.0012, +0.0264]; AP2 - SH30 +0.0052 [-0.0065, +0.0196].

## Dose response (what is left of it)

- Share, at +-0.5 m: 25 % minus 10 % is **-0.0286 [-0.0412, -0.0177]**. The turn is between 10 % and 25 %. What gets worse is offroad + corridor (27.5 -> 42.5 zeros; left corridor alone 19.5 -> 28.5), and collisions go back up (8.5 -> 14); slow scenes do not change (56.5 -> 50).
- The open-loop response keeps rising with the share while the closed-loop score falls: response to a 0.5 m offset within 4 s is 0.28 (AP2), 0.73 (10 %), 1.00 (25 %) ([op_parity ot_ladder.md](../../op_parity/results/ot_ladder.md)). A plan that removes the whole offset within 4 s behind a tracker that executes it one to one is the over-correction decision 132 saw with on-policy labels; here it shows as leaving the corridor. Not verified case by case.
- Amplitude: no read (trainings cut).
- Where the 10 % gain is: at-fault collisions 20.5 -> 8.5, at-fault events 32 -> 16.5; offroad + corridor do not move (28 -> 27.5). The same pattern as OT30 against SH30 (decision 201). AP2's input standard keeps its own effect under the rows: slow scenes 56.5 against OT30's 87.5.

## Seed spread of the baselines (the open point of decision 201)

SH30-F-s0 - s1 -0.0079 [-0.0221, +0.0089], zero / non-zero differs on 27 scenes; AP2-AB-s0 - s1 -0.0017 [-0.0101, +0.0071], 15 scenes; OT30 -0.0077, 20 scenes; APO-a05m10 +0.0029, 14 scenes; APO-a05m25 +0.0114, 33 scenes. Seed differences are 0.002-0.011, the size of the +0.0177 effect is 1.5 to 9 times that.
With SH30's second seed the OT30 effect of decision 201 is +0.0116 [-0.0012, +0.0264] (was +0.0156 [-0.0011, +0.0326] against one seed): still short of zero by 0.001.

## Checks, limits, cost

- Trainer check: 20 steps of `ap2_ot.py --ot-mass 0` and of `ap2_train.py` give the same losses and dev ADEs to the printed digit. Off-track rows are m = 4 decisions only, never cold-start or anchor rows.
- Driver health: every rollout of the six new drivers has its decisions as real inferences, no inference or input error (the chain fails otherwise); no pool error, stall or refill in the chain logs.
- Not run (cut): HUGSIM guardrail, navtest / navhard, AlpaSim-standard offline read. So "candidate" is by the closed-loop lines only; the launch-stall side effect of decisions 143 / 146 was not checked for this recipe (it held for OT30, decision 198).
- Two seeds per recipe, one simulation each, 27 logs, local rendering. The simulator reproduces scores only for a fixed scene list (decision 211); every driver here ran the same three chunk lists, so that term is shared. Two cells against the line, no multiplicity correction.
- The base recipe is the strong hinge, which decision 205 identifies as the trigger of the drift these rows partly repair. Whether off-track rows add anything on the lambda-10 recipe is not known from this data.
- Cost, approximate (shared cards, not tallied job by job): 5 trainings and 3 smokes about 3 job-hours, 12 prep jobs about 1.4, closed loop 18 chunk jobs about 8 (chunks took up to 40 min on loaded cards), ladder 0.3: about 13 job-hours against an estimate of 19 for the full grid. The 20-step smoke with 25 % off-track rows reserved 25.8 GB of VRAM (declared 48; the full trainings' peaks were not read back); RAM was declared as non-reclaimable memory (40 GB) because the host token stores are shared page cache.
- Checkpoints (box): `$DATA_DIR/runs/op_parity/runs/{AP2-AB-s1, APO-a05m10-s0, APO-a05m10-s1, APO-a05m25-s0, APO-a05m25-s1}/ckpt-final.pt`; runs `$DATA_DIR/runs/alpasim/ot2/b/` (manifest.json), 8.3 GB with the C runs.
