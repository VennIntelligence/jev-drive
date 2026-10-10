# BODY1 arm 4.3 (Amendment 5, w = 3): offline gate G3 at full scale

2026-10-10. Full runs `P2H10B-F-s0` / `P2H10B-F-s1` (P2H10 recipe, 10 000 steps, batch 128, all 12 navtrain shards; hinge-only rows of
`bd4` / `ot1` / `yr1` with weight w = 3 on both hinge terms; the validation part `navsim/body1-val-logs@v1` excluded from them) read against
P2H10-F of the same seed. Registered in `plans/2026-10-10-body1-prereg.md` Amendment 4 item 5 and Amendment 5. **This is the second attempt of
the arm, made after one pilot read of hold logs ([loss_pilot.md](loss_pilot.md)); w = 3 was selected on the validation part before that read.**
Hold logs have been read once at pilot scale before these numbers, a limit of every hold-log number below.

**Verdict: G3 (a), (b), (c), (d) are all met on both seeds.** The checkpoints go to the staged closed loop (Amendment 4 item 6).

Tables: `results/loss/g3_a5_full_{hold,navtest}_s{0,1}.{csv,json}` (every family, class and subset), `g3_a5_full_bench/`, `g3_a5_full_d.csv`,
`g3_a5_full_slope.md`. Readers: `scripts/bd4_g3.py`, `scripts/bd4_g3d.py`. Training logs: `$DATA_DIR/runs/body1/loss/a5/full-s{0,1}/`.

## (a) Hold logs: own-plan agent-contact and boundary rates (needs >= 30 % relative fall, interval excluding 0)

Pooled: on-log + `ot1` + `yr1` + `bd4` states of 121 hold logs, 30 083 states. Boundary = NAVSIM raster, margin < -0.20 m. Cluster bootstrap by log.

| Seed | Rate | New | P2H10-F | Difference [95 % CI] | Relative fall | Line |
|--:|:--|--:|--:|:--|--:|:-:|
| 0 | agent | 0.01403 | 0.02729 | -0.01326 [-0.01576, -0.01081] | 48.6 % | met |
| 0 | boundary | 0.01569 | 0.03377 | -0.01808 [-0.02071, -0.01572] | 53.5 % | met |
| 1 | agent | 0.01380 | 0.02666 | -0.01286 [-0.01516, -0.01041] | 48.3 % | met |
| 1 | boundary | 0.01582 | 0.03384 | -0.01802 [-0.02072, -0.01562] | 53.2 % | met |

By state family (relative fall of agent / boundary rate, seed 0 | seed 1): on-log 24 % / 24 % | 24 % / 21 % (agent interval for seed 0 touches 0:
[-0.0033, 0.0000]); `ot1` 45 % / 44 % | 46 % / 47 %; `yr1` 47 % / 57 % | 46 % / 56 %; `bd4` 56 % / 60 % | 55 % / 58 %. The on-log states, the
population closest to the scored loop, move least. > 45 deg hold states (3 954): agent 40 % / 38 %, boundary 49 % / 46 % (seed 0 / seed 1).
Per user class and the road-and-lane raster rate ("road" rows) are in the csv.

## (b) navtest on-log tokens (12 146, never trained on; needs: neither rate up, their sum down)

**Correction, 2026-10-11.** The first version of this table was read on off-protocol plans: until e7747c0b the reader opened `lb_navtest` on GIMM
frames for these warp-trained checkpoints ([navtest_warp.md](navtest_warp.md)). Re-read on warp frames, the plans `jevdrive.bench` scores
(`navtest_warp/loss/g3_a5_full_navtest_s{0,1}.*`); the verdict of (b) is unchanged, the agent fall is larger and now outside its interval:

| Seed | Rate | New | P2H10-F | Difference [95 % CI] | Relative fall | first read (off-protocol): new, base, fall |
|--:|:--|--:|--:|:--|--:|:--|
| 0 | agent | 0.00947 | 0.01153 | -0.00206 [-0.00311, -0.00104] | 17.9 % | 0.01144, 0.01219, 6.1 % [-0.00216, +0.00047] |
| 0 | boundary | 0.01474 | 0.01712 | -0.00239 [-0.00408, -0.00083] | 13.9 % | 0.03087, 0.03623, 14.8 % |
| 0 | sum | 0.02421 | 0.02865 | -0.00445 [-0.00627, -0.00266] | 15.5 % | 0.04232, 0.04841, 12.6 % |
| 1 | agent | 0.00980 | 0.01169 | -0.00189 [-0.00305, -0.00073] | 16.2 % | 0.01087, 0.01186, 8.3 % [-0.00239, +0.00032] |
| 1 | boundary | 0.01490 | 0.01729 | -0.00239 [-0.00427, -0.00071] | 13.8 % | 0.02997, 0.03672, 18.4 % |
| 1 | sum | 0.02470 | 0.02898 | -0.00428 [-0.00644, -0.00213] | 14.8 % | 0.04084, 0.04858, 15.9 % |

Neither rate rises, the sum falls with an interval excluding 0 on both seeds. The agent rate on the real tokens falls by 16 to 18 % with
intervals excluding 0 (115 against 140 and 119 against 142 tokens; the first read had 6 to 8 %, "not significantly"). It is still well below the
30 to 50 % falls of (a), which are on off-track and launch states. > 45 deg tokens: boundary rate -6 % / -7 % (intervals including 0; first read
-27 % / -29 % with intervals excluding 0, not re-established), agent rate -20 % / -36 % (seed 1's interval excludes 0; first read -7 % / -16 %).

## (c) Heading: decision 205's continuation slope `alpha_05` (needs <= P2H10-F + 0.05)

| Seed | New | P2H10-F |
|--:|:--|:--|
| 0 | 0.833 [0.774, 0.894] | 1.028 [0.977, 1.076] |
| 1 | 0.827 [0.767, 0.888] | 1.037 [0.988, 1.085] |

Met, with room (the slope falls by 0.2). Synthetic yaw-rate slots of shards s2 + s3, 1 137 rows (`ot3_rows.py probe`).

## (d) navtest through `jevdrive.bench` (needs >= P2H10-F - 0.3) and the > 45 deg bucket

Both checkpoints and the baseline were scored by the same bench run (baseline re-scored at this checkout: the stored P2H10-F keys carry readout suffixes).

| Arm | Seeds | EPDMS | Per seed | NC | DAC | EP |
|:--|--:|--:|:--|--:|--:|--:|
| P2H10-F | 2 | 88.67 | 88.58 / 88.77 | 98.58 | 96.17 | 87.16 |
| P2H10B-F (this arm) | 2 | **89.19** | 89.19 / 89.20 | 98.79 | 96.53 | 87.04 |

Difference +0.52 [+0.35, +0.70] (seed 0 +0.61, seed 1 +0.43): the line is met by a wide margin and the arm is better on navtest.
Turn-oracle read (`turn_oracle.py replay` of every DAC-failing token, decision 166's definitions; two-seed mean, difference new - base, cluster
bootstrap by log):

| Stratum | Metric | New | P2H10-F | Difference [95 % CI] |
|:--|:--|--:|--:|:--|
| all (12 146) | EPDMS | 89.19 | 88.67 | +0.52 [+0.35, +0.70] |
| all | DAC failure % | 3.47 | 3.83 | -0.36 [-0.54, -0.21] |
| all | inside-cut % | 1.54 | 1.73 | -0.18 [-0.31, -0.06] |
| all | cannot-make-turn % | 0.84 | 0.86 | -0.02 [-0.12, +0.08] |
| > 45 deg (1 517) | EPDMS | 79.55 | 78.61 | +0.93 [+0.28, +1.62] |
| > 45 deg | DAC failure % | 9.33 | 10.19 | -0.86 [-1.66, -0.12] |
| > 45 deg | inside-cut % | 4.15 | 4.78 | -0.63 [-1.28, -0.11] |
| > 45 deg | cannot-make-turn % | 2.70 | 2.60 | +0.10 [-0.47, +0.77] |

The boundary lesson closes part of the inside-cut failures on large turns (-13 % relative); "does not make the turn" is unchanged (2.70 against
2.60 %, as for every earlier recipe, decision 170), so the > 45 deg bucket stays below WA-JEPA. Replay check: all 420 / 422 (new) and 469 / 461 (base)
DAC-failing tokens replayed, none without a side, none disagreeing (`g3_a5_full_d_check.json`).

Guard `dev_drift_off` (`pp_full_check.py train`, needs <= 0.30): 0.0453 / 0.0446 (new, seed 0 / 1) against 0.0411 / 0.0404 (P2H10-F); dev ADE
0.546 / 0.545 against 0.542 / 0.540.

## Costs, deviations, limits

Costs: prep chains 3 x 4 min each, two full runs 24 min each (7.25 it/s, 28.8 GB VRAM, declared 34), G3 (a) to (c) 3 min each, bench and replay about 12 min;
about 1.3 card-h so far; `bd4` cache 12 GB on the box (disk 354 GB free).
Deviations: (1) the baseline navtest numbers come from a fresh bench run of P2H10-F at this checkout, not from the stored readout-suffixed keys.
(2) The > 45 deg bucket here is the token-level devkit bucket of `TURN_BINS`; the closed-loop read uses the scene's navtest token for the same bucket.
Limits: hold logs were read once at pilot scale; the hinge-only rows of the training set are from the very state families the (a) read uses (hold logs
are disjoint logs, not disjoint distributions); the rates are open loop, on the student's own plan, not a score.
