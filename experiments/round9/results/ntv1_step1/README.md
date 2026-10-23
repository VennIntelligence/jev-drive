# NTV1 step 1: heading command and two placements of PAI human rows (round 9, 2026-10-24, decision 465)

The registration is [../../plans/2026-10-24-ntv1-step1.md](../../plans/2026-10-24-ntv1-step1.md) (Chinese), written before any training output.
The lane is `scripts/ntv.sh`, with gates and verdict in `scripts/ntv.py` (it reuses `pcl.py`, `p1l.py`, `cpl.py` and `dl.py`). The box dir is
`$DATA_DIR/runs/round9/ntv1_step1/`: reads, scorer_v2 rescoring and logs stay there. Full tables are in [verdict.md](verdict.md) (the new scorer is
the primary scorer) and [pai281.md](pai281.md) (old scorer). Seed 0 only.

## Arms

All three arms start from P1L (PCL1L-F-s0, decision 461): 19 500 steps from the shipped weights, eager, S rows from step T = 10 000, NTV0-L18's
lr schedule from T.

| arm | tag | change vs P1L | batch before T | batch from T |
|:--|:--|:--|:--|:--|
| N1 | NT1N1-F-s0 | `--cmd-rule heading-all` (decision 459); served with `JEV_CMD=heading` | 115 navtrain + 13 yaw | 83 + 13 + 32 S (rb0) |
| N2a | NT1N2A-F-s0 | N1 + merged pool across the run | 57 + 13 + 58 PAI human (as HP50) | 51 + 13 + 64 pool (about 32 S + 32 PAI human) |
| N2b | NT1N2B-F-s0 | N1 + the same merged pool from T only | 115 + 13 (= N1) | the same as N2a |

How the pool and the PAI rows are set up:
- **Merged pool.** on1_JTPH, R5LC's pool. Its first 169 606 rows equal on1_JT's. The draw file NTV1M gives the rb0 S rows (JT00S-rb0's weights) a
  share of 0.5 and the PAI human rows (RB0P's weights) the other 0.5.
- **PAI rows.** RB0P is the cleaned draw HP50 used: MV-LC D1 filter + F1 / F2, ph : ph05 : ph15 = 2 : 1 : 1. R5LC's own draw lacked F1 / F2.
- **How PAI rows are trained.** They are mixed-mode `both` rows without agent, road or lateral labels. In effect that is imitation + distill, with
  25 % memory drop.
- **Post-T batch.** N2a and N2b are identical after T, so they differ only in placement. N2a's PAI rows therefore drop from 58 to about 32 per
  batch at T.

## Code change

`rm_train.py --on-pre-k K --on-pre-wfile F` (commit 97d889d1) is off by default and adds no operation when off.
- With `--on-start T`, each batch before T draws K pool rows with F's weights, on a separate stream [seed, 0, 191].
- The pool stream used from T is untouched before T.

`--on-mix-log` is monitoring only: it logs the S rows per batch.

## Gates (all passed; lane submitted from f6a0d360)

| gate | result |
|:--|:--|
| flag off: RB0-F's argv vs RB0-F-s0, steps 0-75 | bit-identical |
| flag off: P1L's argv vs PCL1L-F-s0, steps 0-100 | bit-identical |
| merged draw file | 13 793 rb0 rows (all `self`) + 193 850 PAI rows (all `both`), no overlap; shares S 0.5 / ph 0.25 / ph05 0.125 / ph15 0.125; probe groups identical to N1's |
| 60-step smokes (T = 30) | batch lines as planned; pool rows per window 58 / 62.8 / 64 (N2a); 0 S rows before T; S share after T 0.50 (N2a, N2b) / 1.0 (N1); no S-term mass before T; lr formula exact |
| command on 20 000 sampled rows (every source: navtrain, yaw, dev, PAI ph / ph05 / ph15, S) | equal to cmdd.py's independent recomputation |
| N2b vs N1 before T (reported) | bit-identical |
| S / imit gradient ratio, 500 steps after T (band 0.21-1.04) | N1 0.642, N2a 0.241, N2b 0.361 (whole post-T segment 0.44 / 0.42 / 0.43) |
| post: DONE, dev ADE <= 0.566, lr over the whole run | all pass; dev ADE 0.533 / 0.557 / 0.552 |
| served command vs rule | 100 % in all three reads (55 494-55 693 inferences), no RPC error |

The first submission failed only on the N2a composition smoke, and the fault was in ntv.py. The pre-T `both` share summed to 1 - 5e-12 against a
tolerance of 1e-12, while every other share check uses 1e-9. I fixed the tolerance and resubmitted the whole chain. The first try is kept at
`runs/round9/ntv1_step1.try1/`.

## PAI 281, ACC + GAP, Harmonizer OFF (278 common scenes)

| read | new | old | zeros new / old | lateral | offroad / zero-other / side | route-turn | launched | new zeros on NTV0's 183 full-score scenes (old / new) |
|:--|--:|--:|--:|--:|:--|--:|--:|--:|
| **N1** | **0.797** | 0.721 | 46 / 67 | 25 | 16 / 7 / 2 | 13 | 34/37 | 15 / 12 |
| **N2a** | **0.722** | 0.642 | 67 / 89 | 53 | 23 / 24 / 6 | 5 | 32/37 | 40 / 35 |
| **N2b** | **0.779** | 0.700 | 54 / 76 | 38 | 19 / 16 / 3 | 7 | 34/37 | 27 / 24 |
| P1L | 0.813 | 0.734 | 39 / 61 | 23 | 14 / 6 / 3 | 10 | 32/37 | 15 / 12 |
| NTV0-s0 | 0.818 | 0.746 | 38 / 58 | 18 | 12 / 6 / 0 | 14 | 33/37 | 0 / 0 |
| R8K-s2 | 0.872 | 0.804 | 28 / 47 | 20 | 14 / 4 / 2 | 2 | 35/37 | 10 / 9 |

Lateral zeros are zeros with fclass offroad, zero-other or side.

| pair | new scorer | old scorer | lateral zeros | route-turn |
|:--|--:|--:|--:|--:|
| N1 - P1L | -0.016 [-0.054, +0.022] | -0.012 [-0.051, +0.026] | +2 [-7, +11] | +3 [-3, +9] |
| N1 - NTV0-s0 | -0.021 [-0.058, +0.015] | -0.025 | +7 [+0, +15] | -1 |
| N1 - R8K-s2 | -0.076 [-0.121, -0.030] | -0.083 | +5 | +11 [+5, +18] |
| N2a - N1 | -0.075 [-0.129, -0.021] | -0.079 | +28 [+15, +41] | -8 [-15, -2] |
| N2b - N1 | -0.018 [-0.068, +0.032] | -0.022 | +13 [+0, +26] | -6 [-12, -1] |
| N2a - N2b | -0.057 [-0.109, -0.007] | -0.057 | +15 [+1, +29] | -2 |

## Registered lines (new scorer)

- **N1 - P1L = -0.016, against the line >= -0.015.** This misses the line by 0.001, so the registered reading is "the heading command costs in the
  S recipe (report)". The old scorer gives -0.012, which would read as adopt. The CI spans [-0.054, +0.022] and seed spread is 0.03-0.04: in
  practice no measurable cost and no gain.
- **N2a - N1 = -0.075: PAI human rows hurt in this placement.** Route-turn zeros fall from 13 to 5, but lateral zeros rise by 28: offroad +7,
  drift (zero-other) +17. NTV0's full-score scenes gain 40 new zeros. Dev ADE worsens to 0.557.
- **N2b - N1 = -0.018: hurt** (registered line <= -0.01; the CI includes 0). Route-turn zeros fall from 13 to 7, but lateral zeros rise by 13
  (drift +9).

This is the pattern of decisions 449 / 459 again, inside the S recipe. PAI human rows fix route turns and cost more in lateral zeros, and the earlier
they enter, the more they cost (N2a - N2b -0.057). The consistent command does not remove that damage here.

## Cost

13.0 GPU job-hours:
- training: 3.64 job-hours, three arms in parallel at 4.3-5.3 it/s;
- reads: 9.19 job-hours, at 4 per card about 2.3 card-hours;
- gates: 0.13.

That comes to about 6 card-hours, and 2.1 h wall. The first, failed submission ran its gates only (about 0.1 job-hours).
