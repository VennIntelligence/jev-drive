# NTV1 step 4: DAgger iteration 2 (round 9, 2026-10-24, decision 473)

Pre-registration (Chinese): [../../plans/2026-10-24-ntv1-step4.md](../../plans/2026-10-24-ntv1-step4.md). Lane: [../../scripts/dg4.sh](../../scripts/dg4.sh);
checks, draw files and verdicts: [../../scripts/dg4.py](../../scripts/dg4.py). The roll, rows, teacher, pool, label and input-path job bodies are round 6's
`dag.sh` / `dag.py`, unchanged, as in step 2a. Box: `$DATA_DIR/runs/round9/ntv1_step4/`. Full tables:
- [verdict.md](verdict.md): D3 on PAI 281, the registered read;
- [cand_v160.md](cand_v160.md): the candidate tail, D3 seeds 0 and 1 on val160;
- [build.json](build.json): the step-4a summary;
- [pai281.md](pai281.md): old scorer only.

The new scorer is the primary scorer. One seed for the registered line. No step needed a fix or a resubmission.

## 4a: D1's own rows, same teacher

- **Student and serve.** D1 = NT3D1-F-s0 was rolled under step 2a's submission serve (ACC + GAP, g10, FAST, CONC 4, frame dumps, Harmonizer off).
  It ran on the same 719 curated train scenes. The chunk lists were copied from step 2a and checked equal to `dag.py lists`.
- **Teacher.** The teacher is step 2a's b3, unchanged.
- **Pool.** The rows form pool `on1_PAIDC2` (part `pd2`).
- **Aggregated pool.** `on1_JTDC2` = on1_JTDC (on1_JT's five S parts + iteration 1's `pd`, still with its b3 labels) + `pd2`. The tokens are zero-copy.
- **Draw file NTV1D2.**
  - The S part is NTV1D's, bit for bit: 32 / 40, 13 793 rb0 rows.
  - The DAgger 8 / 40 is uniform over both iterations' drawable rows. So the split follows the row counts: pd 0.0995, pd2 0.1005.
- **Weight-0 identity file.** NTV1D2Z = NTV1D + pd2 at weight 0.

| | on1_PAIDC2 (iteration 2, D1 states) | on1_PAIDC (iteration 1, P1L states) |
|---|--:|--:|
| rows / drawable | 113 321 / **93 421** | 112 620 / 92 561 |
| scenes | 717 (2 scenes gave no rows; row gate >= 90 % per chunk) | 719 |
| drawable by class: stop_nolead / turn_sharp / launch / other | 18 643 / 21 767 / 6 328 / 46 683 | 18 703 / 20 609 / 6 547 / 46 702 |
| abs d0 > 1 m, left / right | 4 664 / 3 586 | 9 450 / 7 103 |
| T4 p99 (position read) | 5.05, 2 052 rows over (2.2 %) -> **fallback** | 5.30, 2.6 % -> fallback |

D1 strays from the lane centre about half as often as P1L did: the rows with |d0| > 1 m fall from 16.6 k to 8.3 k.

**Train-scene roll by-product (old scorer, 717 common scenes).**

| student | mean | zeros |
|:--|--:|--:|
| D1 | 0.707 | 177 |
| P1L | 0.673 | 198 |

- Paired D1 − P1L: +0.033 [+0.006, +0.062].
- Route-turn zeros 8 vs 29, zero-other 6 vs 19.
- AF-front zeros 25 vs 17 and offroad 65 vs 56: these two classes rose.

### 4a gates (all pass)

| gate | result |
|:--|:--|
| chk1 (c000) | 6 / 6 scored, teacher on 100 %, 3.3 card-h projected |
| T1 teacher geometry | 0.299 / 0.369 m (= decision 434 and step 2a exactly) |
| T2 in canvas | 62 / 65 = 95 % |
| T4 | p99 5.05 > 4: registered fallback (2.2 % <= 3 %, those rows weight 0) |
| input path (rf_recov fwd) | D1 median 0.011 m, 3.9 % over; wrong ckpt RB0-F-s1 99.9 % over |
| mfix (on1_JTDC2 prefix = on1_JTDC) | 0 mismatches, part counts = on1_JTDC's + pd2, agent ok already equal (0 fixed) |
| draw files (wcheck) | NTV1D2Z prefix = NTV1D bitwise, tail 0; NTV1D2 S part = NTV1D bitwise; DAgger rows = both pools' drawable rows, uniform, mode both, fut finite; shares 0.8 / 0.0995 / 0.1005 |
| argv diff | D1's built argv = its stored argv; D3 differs only in --tag / --on / --on-wfile |
| T3 (a) D1's argv vs stored NT3D1-F-s0, steps <= 100 | 131 / 131 bit-identical |
| T3 (b) weight-0 pair, T = 30 (on1_JTDC / NTV1D vs on1_JTDC2 / NTV1D2Z) | 245 / 245 identical, pool rows drawn |
| D3 pre-T vs stored D1 | 131 / 131 bit-identical |
| smoke D3 (60 steps, T = 30) | batch 75 / 13 / 40 mixed; S share after T 0.813; draw shares S 0.8 / pd 0.0995 / pd2 0.1005; lr exact |

## 4b: D3 = NT4D3-F-s0

D3 is D1's argv with only `--on on1_JTDC2 --on-wfile runs/round9/ntv1_step4/w/NTV1D2.npz` changed. It trains from the shipped weights, seed 0, eager, 19 500 steps.

- S / imit gradient ratio, 500 steps after T: 0.318 (band 0.21–1.04). Over the whole post-T segment it is 0.419 (D1 0.421).
- Dev ADE 0.530 (D1 0.530).

### PAI 281 ACC + GAP, Harmonizer OFF (276 common scenes)

| read | new | old | zeros new / old | lateral | route-turn | overshoot | AF-front | neg. flips A / B |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| **D3** | **0.879** | 0.799 | 23 / 45 | 15 | 3 | 23 | 4 | **12 / 5** |
| D1-s0 | 0.867 | 0.798 | 27 / 46 | 18 | 3 | 20 | 5 | 14 / 8 |
| D1-s1 | 0.837 | 0.761 | 31 / 52 | 20 | 3 | 22 | 7 | 21 / 11 |
| P1L | 0.811 | 0.732 | 39 / 61 | 23 | 10 | 24 | 4 | 16 / 12 |
| NTV0-s0 | 0.813 | 0.741 | 39 / 59 | 18 | 15 | 23 | 3 | - |
| R8K-s2 | 0.872 | 0.803 | 28 / 47 | 20 | 2 | 20 | 5 | 14 / 9 |
| INTP75-NR (ref., not clean) | 0.878 | 0.809 | 26 / 45 | 17 | 3 | 20 | 5 | 8 / 6 |

The flip counts:
- **A**: NTV0's 200 new-scorer full scenes on which the read scores < 0.98.
- **B**: NTV0's 180 old-scorer full scenes on which the read scores new = 0.

D3 keeps 30 of R8K-s2's 40 fixes. D1 keeps 28.

| pair | new scorer | old scorer | zeros (old) |
|:--|--:|--:|--:|
| **D3 − D1-s0 (registered)** | **+0.012 [−0.014, +0.039]** | +0.001 [−0.030, +0.032] | −1 |
| D3 − D1 seed mean | +0.027 [+0.002, +0.052] | +0.020 [−0.008, +0.047] | |
| D3 − D1-s1 | +0.042 [+0.012, +0.072] | +0.038 | −7 |
| D3 − P1L | +0.067 [+0.030, +0.106] | +0.067 | −16 |
| D3 − NTV0-s0 | +0.066 [+0.027, +0.104] | +0.059 | −14 |
| D3 − R8K-s2 | +0.007 [−0.023, +0.039] | −0.004 | −2 |
| D3 − INTP75-NR | +0.001 [−0.029, +0.031] | −0.010 | 0 |

Zero classes, D3 against D1-s0 (95 % CIs all contain 0):

| overshoot | offroad | AF-front | zero-other | rear-hit (partial) | slow (partial) |
|--:|--:|--:|--:|--:|--:|
| +3 | −2 | −1 | −1 | +1 | −1 |

Against R8K-s2, D3 has more overshoot zeros (+3) and rear-hit partials (+5), and fewer lateral zeros (−5).

**Highway (behavior_stats Q1).**
- D3: median v_ego / v_human 1.01, < 0.85 on 8 %, slow + rear points 0.98 -> **not slow**.
- D1-s0 was "slow" only through 2.20 slow + rear points. D3 is faster than the human more often: > 1.10 on 29 % of the steps (R8K-s2 21 %).

**Registered line (new scorer): D3 − D1-s0 = +0.012, inside ±0.015 -> neutral.** The old scorer gives the same line (+0.001).

### Candidate rule: triggered (D3 0.879 >= 0.875)

D3-s1 (NT4D3-F-s1, only `--seed` and the tag changed) passed its gates: ratio 0.645 and dev ADE 0.536. No 281 read of s1 was run, as registered.

**val160** (151 common scenes, ACC + GAP, Harmonizer off, v160.sh's chunk lists and serve):

| read | new | old | zeros new / old |
|:--|--:|--:|--:|
| D3-s0 | 0.842 | 0.756 | 19 / 32 |
| D3-s1 | 0.864 | 0.777 | 15 / 28 |
| **D3 seed mean** | **0.853** | 0.767 | |
| R8K-s2 (stored) | 0.853 | 0.767 | 18 / 31 |
| D1-s0 (stored) | 0.827 | 0.748 | 21 / 33 |
| P1L (stored) | 0.840 | 0.754 | 19 / 32 |

| pair (new scorer) | difference |
|:--|--:|
| D3 seed mean − R8K-s2 | **−0.000 [−0.041, +0.041]** |
| D3-s0 − R8K-s2 | −0.011 [−0.054, +0.031] |
| D3-s1 − R8K-s2 | +0.010 [−0.036, +0.057] |
| D3 seed mean − D1-s0 | +0.025 [−0.004, +0.059] |
| D3-s1 − D3-s0 | +0.021 [−0.015, +0.060] |

**A note on val160's scorers.** The new scorer changes only 12–13 val160 scenes per read, and those scenes move by almost the same amount in every read. As a result many new-scorer and old-scorer paired differences are identical. This is a property of the scorer on this exam, not a fault in the script. I checked the new and old per-scene scores of all five reads: 12–13 changed scenes each, with mean shifts of +0.079 to +0.086.

## Does iteration 2 differ from round 6's DAG2 (decision 438)?

The design differs in four ways:
- **Teacher.** DAG2 changed the teacher to b4c (longitudinal min(student, human) caps that slowed 16 % of rows) and relabelled iteration 1's rows. Here the teacher stays b3 and iteration 1's rows are kept as they were. The only change is the D1-state rows.
- **Base recipe.** DAG2 used R5LC's argv: PAI human rows, init from NTV0, 9 500 steps, batch 136. Here it is the clean recipe from the shipped weights.
- **Students.** DAG2 pooled two students. Here there is one.
- **Arms.** DAG2 had no clean S-only arm.

The outcome is similar in kind, with a slightly better point estimate:
- DAG2 against same-seed iteration 1 was +0.014 / −0.021, and it settled at iteration 1's two-seed average.
- D3 against D1-s0 is +0.012 (neutral by the line).
- D3 is above D1's seed mean by +0.027 (CI just above 0). It has fewer negative flips (12 vs 14) and a smaller seed gap on val160.

Read this way, iteration 2 mostly reduces D1's seed variance and adds a small gain; it does not add a new jump. One seed on 281.

## Cost and disk

| | GPU job-hours |
|:--|--:|
| rolls | 8.10 (2.6 card-h) |
| pool build (cache / fwd) | 0.39 |
| gates / smokes | 0.14 |
| training (s0 + s1) | 2.35 |
| PAI 281 read | 3.11 |
| val160 reads | 3.55 |
| **total** | **17.6** |

Wall time was 4.5 h from the first job to DONE.

Disk on the box is about **121 GiB**, all kept:

| item | GiB |
|:--|--:|
| rows | 37.8 |
| cache | 26.2 |
| on1_PAIDC2 | 3.5 + tokens 28.3 |
| on1_JTDC2 | 21.5 + 2.3 |
| roll / tax / feat / w | about 1.7 |

The frame dumps and rollout.asl files were deleted per chunk, each only after that chunk's row-file tags checked equal to NT3D1-F-s0. No staging dir is left.
