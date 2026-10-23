# CMD-PILOT: the heading command on every row (round9, 2026-10-23, decision 459)

Question ([plan](../../plans/2026-10-23-command-design.md), sections 5 and 8, registered before any training output): if one command definition
(`rmem.cmd_heading`, the route turns >= 20 deg within 40 m on the fed float16 route memory) runs through every training row (navtrain, yaw,
dev, every pool part) and serve (`JEV_CMD=heading`) from the shipped weights, how much of the lateral damage that PAI human rows bring into
the base (old command: 37 -> 60 lateral zeros, RB0-F -> RB0-P50) is left?

Arms (seed 0, shipped weights, eager, 10 000 steps): **HF-s0** = RB0-F-s0's stored argv + `--cmd-rule heading-all`; **HP50-s0** = RB0-P50-s0's
stored argv (RB0P.npz draw, decision 430 D1 filter) + `--cmd-rule heading-all`. Old-definition cells: the stored RB0-F-s0 / RB0-P50-s0 reads.
Read: PAI 281 ACC + GAP 0.7,2.7, g10, FAST, CONC 4, Harmonizer OFF. Lane [../../scripts/cpl.sh](../../scripts/cpl.sh) /
[cpl.py](../../scripts/cpl.py); box `$DATA_DIR/runs/round9/cmd_pilot/` (reads, logs, per-row command samples).

## Code and gates

- `rm_train.py --cmd-rule heading-all` (`cmd_all`, off by default): before `--rm-drop`, every Store row's command (ego cols 1:4) :=
  `rmem.cmd_heading` of its fed memory; rows with no valid memory slot keep their stored command (HF 3 608 / 172 860 rows, HP50 6 006 / 701 407).
  Changed rows: navtrain 16.7 %, dev 15.0 %, yaw 15.7 %, PAI human ph / ph05 / ph15 22.0-22.4 %, S parts 25.4-28.2 %.
- G0a, flag off ([identity.json](identity.json)): RB0-F-s0's argv on HEAD 29f2bdf1, eager, steps 0-75: 22 / 22 scalars bit-identical to the stored run.
  The trainings started on box HEAD 9a7e6a03 / b3ae2b89 (pool-pilot plan and lane commits only; no change under route_mem, lib, jevdrive).
- G0b / G5, flag on, 60-step smokes ([smoke_HF.json](smoke_HF.json), [smoke_HP50.json](smoke_HP50.json)): finite, step 60; the fed command on
  20 000 sampled rows == cmdd.py's independent recomputation (`rule_h(20, 40)` on the stored feat files) **100.000 %** in every part, and ==
  rows.parquet's `h20_40` on the navtrain rows 100 %; per-part change shares within 0.02 pt of cmdd.py. A first submission failed this gate on a
  wrong reference in cpl.py (ok rows with an empty memory: the trainer keeps the stored command as designed, the reference had S; 33 / 11 959
  rows); reference fixed, everything resubmitted, first try kept at `runs/round9/cmd_pilot.try1/`.
- G6, served command vs the rule ([cmdagree_HF.json](cmdagree_HF.json), [cmdagree_HP50.json](cmdagree_HP50.json)): 55 494 / 55 693 inferences,
  **100 %** agreement. No RPC errors in any read.

## Result ([verdict.md](verdict.md), [pai281.md](pai281.md))

279 scenes common to the five reads.

| read | mean old | mean new | lateral zeros | offroad / zero-other / side | route-turn | zeros (old) | launch | dev ADE |
|:--|--:|--:|--:|:--|--:|--:|--:|--:|
| RB0-F-s0 (old command) | 0.647 | 0.715 | 37 | 19 / 14 / 4 | 21 | 88 | 32/37 | 0.528 |
| **HF-s0** | 0.655 | 0.727 | 36 | 18 / 14 / 4 | 19 | 87 | 33/37 | 0.534 |
| RB0-P50-s0 (old command) | 0.602 | 0.673 | 60 | 28 / 24 / 8 | 11 | 102 | 27/37 | 0.550 |
| **HP50-s0** | 0.613 | 0.699 | 52 | 20 / 23 / 9 | 11 | 100 | 28/37 | 0.559 |
| NTV0-L18-s0 (reference) | 0.743 | 0.815 | 18 | 12 / 6 / 0 | 15 | 59 | 33/37 | - |

| paired (scene bootstrap) | old scorer | new scorer (primary) |
|:--|--:|--:|
| HF - RB0-F | +0.008 [-0.022, +0.038] | +0.012 [-0.019, +0.042] |
| HP50 - RB0-P50 | +0.011 [-0.023, +0.045] | +0.025 [-0.009, +0.061] |
| HP50 - HF | -0.043 [-0.093, +0.008] | -0.028 [-0.078, +0.022] |
| RB0-P50 - RB0-F | -0.045 [-0.098, +0.007] | -0.042 [-0.095, +0.011] |

- Lateral zeros (fclass offroad + zero-other + side): **D_old = 60 - 37 = +23, D_new = 52 - 36 = +16**. The drop is all offroad (28 -> 20);
  drift / lane-change zeros (zero-other + side) stay 32 vs 32.
- HF guard vs RB0-F: all pass (new mean +0.012, route-turn 19 vs 21, lateral 36 vs 37, launch 33/37, dev ADE 0.534 <= 0.538).
- Route-turn gain kept: 11 (HP50) vs 19 (HF), <= HF - 5.
- **Line: between (8 < D_new = 16 < 17)**: partial. Per the registered table, seed 1 for both arms is proposed (~2.6 GPU h), nothing run.
  D_new sits one zero under the "command not the cause" line, inside the stated seed noise (+-4-5 lateral zeros).
- Listed scenes (old / new score, class): 1a69f757 (long gentle curve) 0 in all four cells (RB0-F / HF AF-front, P50 cells offroad), so the
  heading command does not widen it from the shipped weights; 1e8e939e (S-junction) 0 / 0 / 0 / **1.00** (HP50 passes; F cells route-turn);
  baabc57d 1 / 1 / 0 / **1** (RB0-P50's offroad is gone); 596af0c7 offroad in all four.
- New zeros on NTV0's 184 full-score scenes: RB0-F 24, HF 23, RB0-P50 45, HP50 43.
- Not done (registered): nuPlan check, Harmonizer ON reads, the bend plan-offset table.

GPU: training 1.01 h, gates 0.09 h, reads 6.17 job-h at 4 per card (~1.5 card-h): ~2.6 card-hours.
