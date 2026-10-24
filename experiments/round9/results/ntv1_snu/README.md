# NTV1 S-NU: fresh own nuPlan S pool for D3 (round 9, 2026-10-24) -- STOPPED at the raster gate

Pre-registration (Chinese): [../../plans/2026-10-24-ntv1-snu.md](../../plans/2026-10-24-ntv1-snu.md). Lane: [../../scripts/snu.sh](../../scripts/snu.sh),
checks: [../../scripts/snu.py](../../scripts/snu.py). Box: `$DATA_DIR/runs/round9/ntv1_snu/`.

**Status.** The pool is built and every pool gate passed except one registered sub-check of `latchk`. As a result, the arm NT5SNU-F-s0 was not trained and nothing was read.
The lane waits for a decision (section "Open decision").

**Student.** D3 = NT4D3-F-s0. Its 2-seed PAI 281 mean (new scorer, Harmonizer off) is 0.859, at or above D1's 0.852 (decision 475).

## Build (S-NU)

- **Roll.** D3 rolled onpol-train at 18 decisions with `ol_chain.sh roll1`, using RB0-F-s0's job declarations and an empty env. All 1 031 / 1 031 scenes scored.
- **On the same scenes as RB0-F-s0's roll** (old nuPlan scorer, [rollstats.json](rollstats.json)):

| | D3 | RB0-F-s0 |
|:--|--:|--:|
| mean | 0.854 | 0.788 |
| zeros | 88 | 146 |
| full | 653 | 576 |
| offroad | 39 | 60 |
| left corridor | 15 | 47 |
| collision | 24 | 29 |

  Paired D3 − RB0-F-s0: +0.067 [+0.047, +0.088].
- **Rows `L18D3`, pool `on1_L18D3` (part d3s)** ([rowstats.json](rowstats.json)):

| | d3s | rb0 |
|:--|--:|--:|
| rows | 17 590 | 17 429 |
| scenes / logs | 1 020 / 97 | 1 020 / 97 |
| drawable (JT00S rule) | 13 932 | 13 793 |
| held out by log | 3 658 | 3 636 |
| static rows < 0.5 m/s | 4.3 % | 4.6 % |
| median speed | 5.2 m/s | 4.8 m/s |
| \|e0\| > 1 m | 7.4 % | 9.6 % |

- **Arm pool `on1_JTSNU`.** Parts in order: d3s, rtb1, ys, y, tv, pd, pd2 (395 708 rows).
- **Draw file NSNU.** S 0.8, uniform over the 13 932 drawable d3s rows; pd 0.0995 / pd2 0.1005, bitwise NTV1D2's.
- **Lateral raster `SNUv2`.** It covers d3s + rtb1 and was built the way L18NR1v2 was.

## Gates

| gate | result |
|:--|:--|
| c0 (configs vs RB0's c0) | pass. Driver / controller / eval / trafficsim configs are byte-equal. The only other changes are throughput / port / run-dir lines. 172 / 172 scored |
| R0 reproduction (today's geo + rm_build on RB0's own roll vs stored L18RB0) | pass. Every key has a row-equal share of at least 0.999 |
| R1 / R2 | pass. One-ulp share 0.9993; wrong-seed control 0.004; ego max 0.018 m; actors 0.99998 |
| poolgate | pass. Route ok 0.9998, route0_share_005 0.9932, cov20 0.475 (rb0 0.461) |
| afix (arm pool rows vs on1_JTDC2 / on1_L18D3) | pass, 0 mismatches |
| zfix (identity pool prefix) | pass |
| latgate (human-path activations) | pass. Hard union 0.0184 vs 0.0176; soft 0.0367 vs 0.0359 |
| draw files, argv diff | pass |
| weight-0 identity (D3 argv, T = 30, on1_JTDC2 vs on1_JTSNUZ) | pass. 153 / 153 scalars identical. The 92 excluded `pool_held` / `pool_fit` diagnostics differ, as registered. The identity pool was then removed |
| **latchk** | **FAIL** on one sub-check, explained below. The other sub-checks pass |
| G0b, smoke, training, reads | not run (they depend on latchk) |

**Why latchk failed.** The registered sub-check needs the rtb1 segment of the new raster to equal L18NR1v2's bit for bit.
- onc, ok, own_found and junction_share are bitwise equal.
- The soft raster differs on 19 of 17 489 rtb1 rows. These rows come from 3 scenes, and on them the soft mask changes as a whole.

[latdiag_*.json](latdiag_0.json) rebuilt those rows with `df_lat.work`:
- with `PYTHONHASHSEED=1` they equal L18NR1v2 exactly;
- with seed 0, and with the seed unset, they equal the new raster.

So `df_lat build` depends on Python's string-hash order (own-lane selection) on a few rows. Neither raster is wrong. The rtb1 rows carry weight 0 in NSNU, and the d3s rows have the same small ambiguity.

## Open decision

Two ways to go on:
- (a) accept SNUv2 as built. The failed sub-check is a property of the raster builder, not of the new pool. Then resubmit the tail (G0b, smoke, train, reads, report).
- (b) rebuild SNUv2 under a fixed `PYTHONHASHSEED` (1 reproduces L18NR1v2), re-run latchk, then the tail.

Either needs a plan note. About 5 min of CPU for the rebuild, then about 7 GPU job-h for the tail.

**Cost so far.** 2.7 GPU job-hours (rolls 2.6); wall 1.5 h.

**Disk** (all kept):

| item | size |
|:--|--:|
| roll | 9.2 GB |
| pool + tokens | 6.0 GB |
| arm pool | 22 GB |
| raster | 12 GB |

Box free: 647 GB.
