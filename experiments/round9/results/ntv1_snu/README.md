# NTV1 S-NU: refresh the nuPlan S pool with the student's own rollouts (round 9, 2026-10-24, decision 478)

Pre-registration (Chinese): [../../plans/2026-10-24-ntv1-snu.md](../../plans/2026-10-24-ntv1-snu.md), with notes 1 and 2.
- Lane: [../../scripts/snu.sh](../../scripts/snu.sh).
- Checks, draw files and verdicts: [../../scripts/snu.py](../../scripts/snu.py).
- Box: `$DATA_DIR/runs/round9/ntv1_snu/`.

Full tables:
- [verdict.md](verdict.md): PAI 281, the registered read;
- [v160.md](v160.md): val160;
- [pai281.md](pai281.md): old scorer only;
- [build.json](build.json): the lane summary.

The new scorer is primary. One seed.

**Result.** The fresh own nuPlan S pool hurts by the registered line.
- PAI 281: S-NU − D3-s0 = **−0.037 [−0.071, −0.004]**, new scorer; the old scorer agrees (−0.037).
- Against D3's 2-seed mean the arm is −0.017 [−0.050, +0.015].
- On val160 it ties D3-s0 (+0.000).
- The S pool built from D3's own nuPlan rollouts is therefore not adopted; D3's recipe keeps RB0-F-s0's S pool (rb0).

## Student

D3 = NT4D3-F-s0, by the main session's rule. D3's 2-seed PAI 281 mean (new scorer, Harmonizer off) is 0.859, at or above D1's 0.852 (decision 475).

## Build (S-NU)

- **Roll.** D3 rolled nuPlan onpol-train (1 031 scenes, 18 decisions, default nuPlan serve).
  - It used `ol_chain.sh roll1`, unchanged, with RB0-F-s0's job declarations and an empty env.
  - Chunk c0 ran first, behind a config gate.
- **Rows `L18D3`, pool `on1_L18D3` (part d3s).** They were built with RB0's own job bodies: `ol_chain.sh geo / tok / strip` and `rm_build.py pool --workers 32`.
- **Static rows (< 0.5 m/s) are not excluded.** rb0's current draw (JT00S-rb0) does not exclude them either. That rule, applied to on1_JTDC2's rb0 rows, reproduces NTV1D2's 13 793 S rows exactly.
- **Hold-out by log:** crc32(log) % 100 < 15.

Train-scene roll on the same 1 031 scenes (old nuPlan scorer, [rollstats.json](rollstats.json)):

| | D3 | RB0-F-s0 |
|:--|--:|--:|
| mean | 0.854 | 0.788 |
| zeros | 88 | 146 |
| full | 653 | 576 |
| offroad | 39 | 60 |
| left corridor | 15 | 47 |
| collision | 24 | 29 |

Paired D3 − RB0-F-s0: +0.067 [+0.047, +0.088].

Pools ([rowstats.json](rowstats.json)):

| | d3s | rb0 |
|:--|--:|--:|
| rows | 17 590 | 17 429 |
| scenes / logs | 1 020 / 97 | 1 020 / 97 |
| drawable / held out by log | 13 932 / 3 658 | 13 793 / 3 636 |
| static < 0.5 m/s | 4.3 % | 4.6 % |
| speed p10 / p50 / p90 (m/s) | 1.25 / 5.24 / 9.29 | 1.18 / 4.83 / 9.15 |
| \|e0\| > 1 m | 7.4 % | 9.6 % |
| decisions excluded (overlap / corridor / log future / frame) | 587 / 19 / 143 / 13 | 639 / 121 / 150 / 13 |

D3's states sit closer to the logged lane than RB0-F's did (|e0| > 1 m: 7.4 % vs 9.6 %), and corridor exclusions drop from 121 to 19.

- **Arm pool `on1_JTSNU`.** Parts in order: d3s, rtb1, ys, y, tv, pd, pd2. The S part comes first because the lateral raster is a prefix raster.
- **Draw file NSNU.**
  - S 0.8, uniform over the 13 932 drawable d3s rows (32 a batch).
  - DAgger rows pd 0.0995 / pd2 0.1005, bitwise D3's NTV1D2 weights (8 a batch).
- **Lateral raster `SNUv2`.** It covers d3s + rtb1 and was built with `df_lat.py build --dilate 6 --points ra --prefix` under `PYTHONHASHSEED=1` (plan note 2).

## Gates (all pass)

| gate | result |
|:--|:--|
| c0 config check vs RB0-F-s0's c0 | driver / controller / eval / trafficsim configs byte-equal; other diffs only throughput / port / run-dir lines (note 1); 172 / 172 scored |
| R0 reproduction: today's `on_rows geo` + `rm_build pool` on RB0-F-s0's own stored roll vs the stored rb0 files | every key of rows meta, all.npz, pool tab / agent / road / drivable, route.npz and features row-equal on at least 0.999 of rows ([repro_rb0.json](repro_rb0.json)) |
| R1 input reproduction / R2 | R1: one-ulp share 0.9993 (rb0 0.9987); wrong-seed control (NT4D3-F-s1) 0.004 (< 0.20). R2: ego 0.018 m; actors within 0.5 m 0.99998 |
| poolgate | route ok 0.9998; route0_share_005 0.9932; cov20 0.475 (rb0 0.461) |
| afix: arm-pool rows vs on1_JTDC2 / on1_L18D3 | 0 mismatches in every file, the features, the teacher and a token sample; rb0's agent ok in on1_JTDC2 = on1_L18RB0's (no post-merge fix on S parts) |
| zfix: identity pool on1_JTDC2 + d3s | prefix = on1_JTDC2 in every file; part counts right |
| latchk | rtb1 segment of SNUv2 = L18NR1v2 bit for bit (onc, soft, ok, own_found, junction); 0 build errors |
| latgate (human logged path) | hard union 0.0184 vs 0.0176 old; soft 0.0367 vs 0.0359 (line 1.5x) |
| draw files / argv diff | NSNU / NSNUZ checks; the arm differs from D3's stored argv only in --tag / --on / --on-wfile / --lat-file |
| weight-0 identity (T = 30, 60 steps): on1_JTDC2 / NTV1D2 vs on1_JTSNUZ / NSNUZ | 153 / 153 scalars identical; the registered-excluded held-out pool diagnostics (pool_held / pool_fit, 92) differ; identity pool removed afterwards |
| G0b: the arm's pre-T vs stored D3 | 85 / 85 identical (46 excluded diagnostics differ) |
| smoke | batch 75 / 13 / 40 mixed; S share and draw shares as NSNU; lr exact |
| ratio / post | S / imit gradient ratio 0.438 at 500 steps after T (band 0.21–1.04), whole post-T 0.334 (D3 0.419); dev ADE 0.529 (D3 0.530); lr over the whole run exact |

**Attempt 1 (plan note 2).** The first raster was built without a fixed hash seed and failed latchk on one sub-check.
- **Symptom.** The soft raster differed from L18NR1v2 on 19 of 17 489 rtb1 rows, all from 3 scenes.
- **Cause.** `df_lat build` picks the ego's own lane chain in Python string-hash order. [latdiag_1.json](latdiag_1.json) shows the 19 rows rebuild to L18NR1v2 under seed 1, and to attempt 1 under seed 0 or no seed.
- **Fix.** By main-session decision (option b), the whole raster was rebuilt under `PYTHONHASHSEED=1`, explicit in `snu.sh lat`. latchk was rerun unchanged and passed.
- **Record.** Attempt 1's result is in [latchk_attempt1.json](latchk_attempt1.json); its raster is in `$O/attempt1/`.

## Arm

NT5SNU-F-s0 = D3's stored argv with only the S part changed: `--on on1_JTSNU --on-wfile NSNU --lat-file SNUv2`. It trains from the shipped weights, seed 0, eager, 19 500 steps, S from step 10 000. Each batch after T is 75 navtrain + 13 yaw + 32 S + 8 DAgger.

## PAI 281, ACC + GAP, Harmonizer OFF (D3's serve; 276 common scenes)

| read | new | old | zeros new / old | lateral | offroad | route-turn | overshoot | AF-front | rear-hit p | neg. flips A / B |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **S-NU** | **0.842** | 0.763 | 34 / 56 | 17 | 11 | 9 | 23 | 7 | 20 | 16 / 11 |
| D3-s0 | 0.879 | 0.799 | 23 / 45 | 15 | 13 | 3 | 23 | 4 | 23 | 12 / 5 |
| D3-s1 | 0.840 | 0.767 | 32 / 52 | 22 | 17 | 4 | 21 | 5 | 26 | 16 / 8 |
| D1-s0 | 0.867 | 0.798 | 27 / 46 | 18 | 15 | 3 | 20 | 5 | 22 | 14 / 8 |
| R8K-s2 | 0.872 | 0.803 | 28 / 47 | 20 | 14 | 2 | 20 | 5 | 18 | 14 / 9 |
| P1L | 0.811 | 0.732 | 39 / 61 | 23 | 14 | 10 | 24 | 4 | 27 | 16 / 12 |
| NTV0-s0 | 0.813 | 0.741 | 39 / 59 | 18 | 12 | 15 | 23 | 3 | 26 | - |
| INTP75-NR (ref., not clean) | 0.878 | 0.809 | 26 / 45 | 17 | 10 | 3 | 20 | 5 | 18 | 8 / 6 |

Flips are counted on NTV0's full-score scenes:
- **A**: new scorer ≥ 0.98 for NTV0, < 0.98 for the read.
- **B**: old scorer ≥ 0.98 for NTV0, new scorer = 0 for the read.

S-NU keeps 28 of R8K-s2's 40 fixes; D3-s0 keeps 30.

| pair | new scorer | old scorer | zeros (old) | route-turn zeros |
|:--|--:|--:|--:|--:|
| **S-NU − D3-s0 (registered)** | **−0.037 [−0.071, −0.004]** | −0.037 [−0.074, −0.001] | +11 [+1, +21] | +6 [+2, +11] |
| S-NU − D3 2-seed mean | −0.017 [−0.050, +0.015] | −0.021 [−0.056, +0.013] | | |
| S-NU − D3-s1 | +0.002 [−0.036, +0.042] | −0.005 | +4 | +5 |
| S-NU − D1-s0 | −0.024 [−0.060, +0.010] | −0.035 | +10 | +6 |
| S-NU − R8K-s2 | −0.029 [−0.066, +0.007] | −0.040 | +9 | +7 |
| S-NU − P1L | +0.031 [−0.012, +0.074] | +0.031 | −5 | −1 |
| S-NU − NTV0-s0 | +0.029 [−0.012, +0.071] | +0.022 | −3 | −6 |

Zero classes, S-NU against D3-s0:

| route-turn | AF-front | side | zero-other | offroad | rear-hit (partial) | overshoot |
|--:|--:|--:|--:|--:|--:|--:|
| +6 [+2, +11] | +3 | +2 | +2 | −2 | −3 | 0 |

The loss is concentrated in route-turn zeros. The route-turn count goes 3 → 9 on the new pool, while it was 2–4 for every DAgger-recipe read.

**Highway (behavior_stats Q1).** Not slow: median v_ego / v_human 1.02, < 0.85 on 8 % of steps, slow + rear points 0.97. D3-s0: 1.01, 8 %, 0.98.

**Registered line (new scorer): S-NU − D3-s0 = −0.037 ≤ −0.015 → "fresh own S hurts".** The old scorer gives the same line.

## val160 (150 common scenes; v160.sh's chunks and serve)

| read | new | old | zeros new / old |
|:--|--:|--:|--:|
| S-NU | 0.841 | 0.754 | 19 / 32 |
| D3-s0 | 0.841 | 0.754 | 19 / 32 |
| D3-s1 | 0.863 | 0.776 | 15 / 28 |
| D3 2-seed mean | 0.852 | 0.765 | |
| R8K-s2 | 0.852 | 0.766 | 18 / 31 |
| D1-s0 | 0.826 | 0.746 | 21 / 33 |
| P1L | 0.839 | 0.752 | 19 / 32 |

| pair (new scorer) | difference |
|:--|--:|
| S-NU − D3-s0 | +0.000 [−0.045, +0.045] |
| S-NU − D3 2-seed mean | −0.011 [−0.050, +0.028] |
| S-NU − R8K-s2 | −0.011 [−0.055, +0.031] |

val160 has no registered line. It does not reproduce the 281 loss.

## Reading

D3's own nuPlan states are cleaner than RB0-F's:
- the train-scene roll is +0.067;
- |e0| > 1 m falls from 9.6 % to 7.4 %;
- corridor exclusions fall from 121 to 19.

Training D3's recipe on them still lowers the 281 score, with route-turn zeros up from 3 to 9.

A plausible reading: the S hinges act on states where the policy fails. RB0-F's weaker rollouts give more turn and corridor states on which the hinges teach recovery; D3's cleaner pool gives fewer.

Two caveats:
- The registered contrast (−0.037) is about the size of D3's own seed gap (s1 − s0 = −0.039). Against D3's 2-seed mean the arm is −0.017 with a CI containing 0, and val160 shows no difference.
- So "hurts" is the registered verdict on one seed. The weaker but consistent reading is that refreshing the S pool with D3's own rollouts does not help.

## Cost and disk

| | GPU job-hours |
|:--|--:|
| rolls | 2.60 |
| build (tok) | 0.06 |
| gates / smokes | 0.13 |
| training | 1.32 |
| PAI 281 read | 3.09 |
| val160 read | 1.74 |
| **total** | **8.9** |

Wall time was 3.6 h of job time; the latchk stop and the decision added about 1 h. CPU jobs (geo, repro, merges, rasters) are not counted.

Disk on the box is about **56 GiB**, all kept:

| item | GiB |
|:--|--:|
| roll (dumps stripped) | 9.1 |
| pool + tokens | 6.0 |
| arm pool | 21.8 |
| raster | 11.2 |
| reads | 7.6 |

Attempt 1's raster (`$O/attempt1/SNUv2`, about 12 GB) is also kept. The identity pool and the R0 scratch copy (L18RB0X) were removed. Box free: 606 GB.
