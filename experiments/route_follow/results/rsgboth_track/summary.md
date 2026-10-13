# RSGboth-L18-s0 with the serving-side controller reference (JEV_TRACK), held 452 @18

Serving-only reads, no training. Reads: `rm_chain.sh submit-read` with `READ_ENV` / `READ_SFX` (cards 1-5, pool), labels `RSGboth-L18-s0-rt{gate,gatep,rttimep}`,
`RSG-L18-s0-rtgatep` (`p` = `JEV_TRACK_CFG=preview=0.5`). Report: [rf_rsgbt.sh](../../scripts/rf_rsgbt.sh). Stored plain reads are the references.
Files: [held18/confirm.md](held18/confirm.md), [follow/follow.md](follow/follow.md), [diag/diag.md](diag/diag.md), [diag/zeros.csv](diag/zeros.csv), inside.json, held_raw.txt.

## Verdict: no arm meets the lines (offroad zeros <= 17 with turns >= 85 % and mean >= RSGboth - 0.005). The track switch does not fix RSGboth's offroad-on-curve zeros.

| arm (held 452 @18) | mean | zeros | offroad / collision / corridor | offroad on curve (460) | GT dist (m) | turns completed (24) | inside-cut @1.0 m |
|:--|--:|--:|:--|--:|--:|:--|--:|
| RSGboth plain (ref) | 0.9179 | 33 | 17 / 7 / 9 | 11 | 1.776 | 91.7 % (22) | 54.2 % |
| (a) RSGboth gate | 0.8936 (-0.024 [-0.041, -0.011]) | 41 | 21 / 9 / 13 | 11 | 1.727 | 79.2 % (19) | 54.2 % |
| (b) RSGboth gate + preview 0.5 | 0.8980 (-0.020 [-0.037, -0.006]) | 39 | 20 / 8 / 13 | 10 | 1.738 | 87.5 % (21) | 54.2 % |
| (c) RSGboth time + preview 0.5 | 0.8937 (-0.024 [-0.042, -0.009]) | 41 | 20 / 10 / 13 | 11 | 1.731 | 87.5 % (21) | 54.2 % |
| RSG plain (ref) | 0.8809 | 26 | 17 / 4 / 6 | 9 | 1.135 | 66.7 % (16) | 54.2 % |
| control: RSG gate + preview 0.5 | 0.8751 (-0.006 [-0.020, +0.010]) | 28 | 18 / 2 / 10 | 10 | 1.080 | 58.3 % (14) | 54.2 % |

Zeros are the 452 scenes with driver records (ol_confirm held18); "offroad on curve" counts offroad zeros with road class `curve` at the event over all 460
(zeros.csv, the rf_rsgbv.py diag classification). Differences are paired against the same checkpoint's plain read.

Pairs: (b) - (a) mean +0.004 [-0.000, +0.011], zeros -2 [-5, 0]; (c) - (b) mean -0.004 [-0.017, +0.005], zeros +2. Zeros vs RSG plain: (a) +15 [+6.3, +23.7],
(b) +13 [+3.3, +22.4], (c) +15 [+6.5, +23.7]; RSGboth plain +7 [0, +13.5].

## Reading

- Offroad zeros do not return to RSG's 17: 20-21 vs 17 (plain RSGboth 17 too). Offroad on curves is 10-11 in every RSGboth arm (plain 11, RSG 9), unchanged. The reference
  the controller gets is not what limits the car on these curves.
- The switch costs RSGboth zeros that were not there: +6..+8 zeros (+4..+6 at-fault), mostly corridor (9 -> 13) and collision, mean -0.020..-0.024 (CIs exclude 0). Turn
  completion falls to 79 % (gate) or 87.5 % (with preview; 21/24, but with one off-route, 1 event, 1 short).
- Preview 0.5 helps slightly over plain gate (zeros 41 -> 39, turns 79 -> 87.5 %, mean +0.004), inside noise. gate and time are indistinguishable with preview (the gate opens on a few % of calls).
- Control: RSG with gate + preview 0.5 matches its plain read (mean -0.006, zeros +2) and does not gain turns (58 % vs 67 %), so the effect is not an RSGboth-only artefact of the MPC, it is that
  re-timing / VCURV (slowing for curvature) hurts what the plan was asking for at human speed.
- The inside-cut share is 54.2 % in all arms (13 / 24 turns; the 24-turn set is too small to move).
- Plan-heading tracking ratio at 2 s: not computed. The P0 geometry script (sf_turn.py) works on PAI per-inference drive logs (turn_cases.csv, turn_drive.jsonl.gz) of the stored
  rollouts; the nuPlan reads dump no equivalent file (and the per-inference plan poses are not kept for these runs).

Conclusion: the RSGboth offroad-on-curve excess is not a tracking-reference problem that `JEV_TRACK` removes; the lever stays in the plan (speed on curves), not in the controller reference.
