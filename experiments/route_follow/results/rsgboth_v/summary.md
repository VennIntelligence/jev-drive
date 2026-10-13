# RSGboth variants (seed 0), 2026-10-13

Prereg: [../../plans/2026-10-13-rsgboth-prereg.md](../../plans/2026-10-13-rsgboth-prereg.md) Amendment 1. Chain [rf_rsgbv.sh](../../scripts/rf_rsgbv.sh),
diagnosis / pools [rf_rsgbv.py](../../scripts/rf_rsgbv.py). Each arm = RSGboth-L18-s0's command (pool on1_L18RB0, `--init RB0-F-s0`, 2500 steps,
`--on-mode both`, seed 0) plus ONE change: **A30** `--agent-lam 30`; **W** non-turn pool rows drawn at 0.3x (`--on on1_L18RB0t --on-weight
rclass=straight:0.3,rclass=curve:0.3`); **C2** non-turn rows' imitation target capped at the row's own speed (`--on on1_L18RB0c2`, R1 PASS).

## Verdict: no variant meets all the lines

| line | RSGbA30 | RSGbW | RSGbC2 | RSGboth-s0 | RSG-L18-s0 |
|:--|:--|:--|:--|:--|:--|
| 1 held zeros <= RSG + 3 (<= 37 of 460) | 44 FAIL | 45 FAIL | **40** FAIL | 41 | 34 |
| 2 guard zeros <= RSG + 5 (<= 51 of 1479) | 58 FAIL | 60 FAIL | **56** FAIL | 58 | 46 |
| 3 held mean >= RSG (0.8656) | 0.8957 PASS | 0.8960 PASS | 0.8973 PASS | 0.9020 | 0.8656 |
| 3 guard mean >= RSG (0.9473 on 1479) | 0.9528 PASS | 0.9527 PASS | 0.9520 PASS | 0.9522 | 0.9473 |
| 4 turns >= 19 / 24 and short <= 2 | 18 / 24 (2 short) FAIL | 20 / 24 (0) PASS | 20 / 24 (1) PASS | 22 / 24 (1) | 16 / 24 (7) |
| inside-cut share @1.0 m (24 turns) | 54.2 % | 62.5 % | 54.2 % | 54.2 % | 54.2 % |
| held GT distance (m) | 1.80 | 1.85 | 1.29 | 1.78 | 1.14 |

Paired (ol_confirm, held18 = 452 scenes; guard = 1479): held zeros vs RSGboth A30 +3 [0, 6.8], W +4 [-4.6, 12.0], C2 -1 [-6.2, 4.3];
vs RSG A30 +10 [2.3, 16.5], W +11 [1.0, 21.3], C2 +6 [-2.8, 14.0]. Guard zeros vs RSGboth A30 0, W +2, C2 -2 [-7.6, 3.6]; vs RSG +12 / +14 /
+10 [1.9, 18.9]. Held mean vs RSGboth: A30 -0.006, W -0.006, C2 -0.005 [-0.017, +0.007]. Turn completion vs RSGboth: A30 -16.7 pp, W and
C2 -8.3 pp [-17.9, 0]; C2 vs RSG +16.7 [0, +34.6].

Reading: C2 is the only one that moves in the right direction (speed over RSG halved: straight +0.65 vs +1.03 m/s, junction +0.26 vs
+0.54; extra zeros vs RSG 12 vs 14; GT distance 1.29 vs 1.78 m) while keeping most of the turn gain (20 / 24, short 1), but the zero
reduction is 1 - 2 and inside the noise; it is 6 (held) / 10 (guard) zeros above RSG, far from the +3 / +5 lines. A30 changes nothing
on zeros and loses turns (as predicted: the extra collisions are rear-ends the shape-only hinge cannot see along the heading). W
changes nothing on speed or zeros (the model's speed is not set by how often it sees non-turn rows) and cuts inside more often.

## Diagnosis (RSGboth vs RSG on held 460 @18, both seeds; [diag.md](diag.md), [zeros.csv](zeros.csv))

| | s0 | s1 |
|:--|:--|:--|
| zeros RSGboth / RSG (452 with driver records) | 33 / 26 | 38 / 26 |
| extra zeros (zero in RSGboth, scored in RSG) / the other way | 14 / 7 | 19 / 7 |
| extra: offroad / collision / corridor | 5 / 4 / 5 | 7 / 8 / 4 |
| extra: straight / curve / junction at the event; in a rf_follow turn scene | 4 / 8 / 2; 0 | 5 / 12 / 2; 0 |
| extra with a lead / box in the corridor (last 3 decisions) | 5 / 5 | 4 / 5 |
| extra where ego speed > log speed at the event | 10 of 14 | 15 of 19 |
| speed RSGboth - RSG, straight / curve / junction (m/s) | +1.03 / +0.92 / +0.54 | +1.03 / +0.93 / +0.60 |
| speed minus the log's, RSGboth / RSG, straight | +0.02 / -1.01 | +0.04 / -0.98 |

1. The extra speed is everywhere, slightly less in junctions: RSGboth drives at the human's speed on average where RSG drove ~1 m/s below it.
2. The extra zeros are on non-turn road (straight + mild curves), none in a turn scene: the turn rows are not the cost.
3. They happen where the ego is faster than the human at that moment: rear-ends behind a lead (collision, struck object ahead within 12 deg),
   creeping on while the log was stopped (corridor, v_log 0.1 - 0.3 m/s), running wide on curves at > log speed (offroad).
4. So the human-speed imitation target on non-turn on-policy rows is the cause, and the shape-only hinges cannot brake (no along-heading
   gradient); a stronger agent hinge (A30) or a lower draw rate (W) does not touch it, capping the target's speed (C2) halves it.
5. Not explained by C2 either: C2 still drives ~0.65 m/s faster than RSG on non-turn road and keeps 12 extra zeros; the base slowness of RSG
   (-1 m/s vs the log) is part of why RSG has few zeros, i.e. part of RSG's zero level is bought with slowness.

## Files

held_raw.txt, held18_confirm.{md,json}, guard_confirm.{md,json}, follow.{md,json}, inside.json, diag.{md,json} + zeros.csv (RSGboth vs RSG,
both seeds, before the arms), diag_variants.{md,json} + zeros_variants.csv (each arm and RSGboth-s0 vs RSG-s0), r1_on1_L18RB0{t,c2}.json,
summary.json. Box: `$DATA_DIR/runs/route_follow/rsgboth_v/` (pool logs, follow decisions), zeros_mech collects 20261013-185351 (diagnosis) and
20261013-203302 (arms); pools `runs/op_parity/cache/on1_L18RB0{t,c2}`; checkpoints `runs/op_parity/runs/RSGb{A30,W,C2}-L18-s0`.
