# NTV1 S-PAI: S rows on the student's own PAI states (round 9, 2026-10-24, decision 477)

Pre-registration (Chinese): [../../plans/2026-10-24-ntv1-spai.md](../../plans/2026-10-24-ntv1-spai.md) (amendment 1: the hinge-geometry fix).
Lane: [../../scripts/spai.sh](../../scripts/spai.sh), helpers [../../scripts/spai.py](../../scripts/spai.py). The roll / rows / teacher / pool job bodies are
round 6's `dag.sh`, unchanged. Box: `$DATA_DIR/runs/round9/ntv1_spai/`. Full tables: [verdict.md](verdict.md). One seed. The new scorer is primary.

**Verdict: neutral.** S-PAI − D3-s0 = **+0.004 [−0.024, +0.032]** on PAI 281 (new scorer; old +0.008). The registered line is ±0.015, so the read
is neutral on both scorers. The mean does not move, but the zero mix does:
- offroad zeros fall from 13 to 6 (−7 [−13, −2]), lateral zeros 15 → 10;
- route-turn zeros rise 3 → 6 (+3 [0, +7]).

On val160 the arm reads +0.025 [−0.016, +0.067] over D3-s0 (offroad 6 → 1). That is over the line but reported only, not judged.

## Step 0: student

D3 = NT4D3-F-s0. D3's 2-seed PAI 281 mean (new scorer, Harmonizer off, `ntv1_step4/results/s1_pai281.md`) is 0.859. That is not below D1's 0.852,
so the rule keeps D3. D3 was rolled fresh; D1's on1_PAIDC2 states were not used.

## States and rows

- **Roll.** D3 on the 719 curated train scenes, step 2a's submission serve, same chunk lists.
  - Train-scene roll (old scorer): 0.729, 163 zeros (50 offroad). D1's step-4 roll: 0.707, +0.022 [+0.002, +0.042].
- **DAgger iteration 3 (kept, not used).** Teacher b3 → pool `on1_PAIDC3`.
  - 115 032 rows, 94 602 drawable, 719 scenes.
  - T4 p99 5.02 → the registered fallback (2.2 % of rows at weight 0).
  - Input-path check passed. Details: [rows_on1_PAIDC3.md](rows_on1_PAIDC3.md).
- **S part `ps`.** The same states at 2 Hz (`k % 5 == crc32(scene) % 5`), before the scoring event and the log end.
  - 23 037 rasters; 22 996 candidate rows in the pool.
  - Merged into `on1_JTDC2S` = on1_JTDC2's seven parts + `on1_PAIDC3:ps:self`, zero-copy tokens.
- **Labels.** Hinge-only, shape-only, on the model's own plan:
  - agent hinge on the replayed actor boxes;
  - road_pai raster hinge, margin 0.3, W 1, with fallbacks (a)–(d) of decision 418. Rule (d): a row whose human logged path triggers gets no road hinge.
  - Of the candidates: 92.4 % carry the road hinge, 4.8 % are dropped by (d), 0.4 % have the t0 footprint off the surface, and 2.4 % have under 1 s of human future.
  - onc / soft: none (PAI has no lane-direction raster). side runs on the actor boxes under the trainer's existing rule.
- **Raster window stretched** to x −8..152 m (128 × 1.25 m) and y −24..24 m (0.5 m). Corners outside the canvas read +20 (no hinge).
  - The trainer reads a per-file `geom` key ([lib/drivable_hinge.py](../../../../lib/drivable_hinge.py)). Without it, the old expression runs unchanged.
  - 22.6 % of the candidates' plans reach beyond the old 56 m edge.
  - The evaluator-rule surface (box-centre component ∪ lane union) is grown in 80 m bands. On one 176 m canvas the component leaked through a junction gap and turned a
    narrow road 99 % "inside". In the near window the banded raster agrees with the stock raster: 0.07–0.6 % of cells differ on 6 checked scenes.

## Gates (all pass)

| gate | value | line |
|:--|--:|:--|
| G1 human path triggers, of primary-ok candidates | 4.9 % | ≤ 7 % |
| G2 own pre-offroad decisions (the last ≥ 2.5 s before each of 50 offroad events), raw hinge | 64 % (after fallbacks 48 %) | ≥ 60 % |
| G3 false trigger on 2 Hz decisions of scenes scored ≥ 0.98 (16 648 rows) | 9.0 % (after fallbacks 6.1 %) | ≤ 10 % |
| G4 ≥ 13 m/s share of the PAI-S draws (16 of 32 S slots) | 32.9 % (all S 17.0 %; rb0 1.1 %) | ≥ 30 % (reading confirmed by the main session) |
| standing share of ps draws / station match / raster read-back | 5.0 % / 100 % / equal | ≤ 5 % |
| gcode: D3's argv, 60 steps, T = 30, old commit vs HEAD | 245 / 245 identical | strict |
| gw0: + on1_JTDC2S / NTV1D2SZ + `--on-road-pai`, vs gcode's HEAD run | 153 / 153 identical, pool drawn | all but `pool_held/*`, `pool_fit/*` |
| gpre: the arm's argv ≤ 100 steps vs stored D3 | 85 / 85 identical | same exception |
| diff / wcheck | only `--tag --on --on-wfile` + appended `--on-road-pai`; shares S 0.4, ps 0.4, pd 0.0995, pd2 0.1005, held scenes 0 | |
| smoke (60 steps, T = 30) | 75 / 13 / 40, self share 0.776, draw shares = file, 21 252 road_pai rows = index, road_ho > 0 after T | |
| S / imit gradient ratio (500 steps after T) | 0.474 (whole segment 0.389; D3 0.419) | 0.21–1.04 |
| dev ADE | 0.5305 (D3 0.5304) | ≤ 0.566 |

The class weights of ps draws (highway 1.42, arterial 1.13, urban 0.80, ramp 0.82) put 19.8 % of the draws on highway, 13.8 % arterial,
65.6 % urban and 0.7 % ramp.

**Attempt 1 (gate stop, fixed in my own trainer change).** gw0 first failed: 67 of 153 scalars differed after T, with the same draws.
- Bisect: the new pool on its own was inert. `--on-road-pai` on the old pool with zero labelled rows reproduced the divergence.
- Cause: CUDA divides by a Python scalar as a multiply by its float32 reciprocal; the geom path divided by a tensor, so old-canvas points moved by an ulp.
- Fix: old rows keep the original expression; geom rows are swapped in with `torch.where`.
- gcode, gw0, gpre and the smoke were all rerun on the fixed code (attempt 1's files are in [attempt1/](attempt1/)). No gate line was changed.

## PAI 281 (ACC + GAP, Harmonizer off, 276 common scenes)

| read | new | old | zeros new / old | lateral | offroad | route-turn | overshoot | AF-front | neg. flips A / B |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **S-PAI** | **0.883** | 0.807 | 22 / 43 | 10 | 6 | 6 | 22 | 5 | 12 / 6 |
| D3-s0 | 0.879 | 0.799 | 23 / 45 | 15 | 13 | 3 | 23 | 4 | 12 / 5 |
| D3-s1 | 0.840 | 0.767 | 32 / 52 | 22 | 17 | 4 | 21 | 5 | 16 / 8 |
| D1-s0 | 0.867 | 0.798 | 27 / 46 | 18 | 15 | 3 | 20 | 5 | 14 / 8 |
| R8K-s2 | 0.872 | 0.803 | 28 / 47 | 20 | 14 | 2 | 20 | 5 | 14 / 9 |
| INTP75-NR (ref.) | 0.878 | 0.809 | 26 / 45 | 17 | 10 | 3 | 20 | 5 | 8 / 6 |

| pair | new | old | lateral zeros | offroad | route-turn |
|:--|--:|--:|--:|--:|--:|
| **S-PAI − D3-s0 (registered)** | **+0.004 [−0.024, +0.032]** | +0.008 | −5 [−11, +1] | −7 [−13, −2] | +3 [0, +7] |
| S-PAI − D3 2-seed mean | +0.024 [−0.002, +0.049] | +0.024 | | | |
| S-PAI − D3-s1 | +0.043 [+0.011, +0.076] | +0.039 | −12 | −11 | +2 |
| S-PAI − R8K-s2 | +0.011 [−0.021, +0.044] | +0.004 | −10 [−18, −3] | −8 | +4 [+1, +8] |
| S-PAI − INTP75-NR | +0.005 [−0.026, +0.036] | −0.003 | −7 | −4 | +3 |

- **Negative flips** on NTV0's full-score scenes: 12 / 6, as D3-s0 (12 / 5). The arm keeps 30 of R8K's 40 fixes.
- **Highway (behavior_stats Q1):** median v / v_human 1.01, 8 % of steps below 0.85, slow + rear-hit 0.94 points → not slow. D3: 1.01 / 8 % / 0.98.

## val160 (151 common scenes, reported, not judged)

| read | new | zeros new / old | lateral | offroad | AF-front |
|:--|--:|--:|--:|--:|--:|
| S-PAI | 0.867 | 15 / 28 | 2 | 1 | 11 |
| D3-s0 | 0.842 | 19 / 32 | 7 | 6 | 9 |
| D3-s1 | 0.864 | 15 / 28 | 5 | 4 | 9 |
| R8K-s2 | 0.853 | 18 / 31 | 9 | 7 | 7 |

Paired differences:
- S-PAI − D3-s0: +0.025 [−0.016, +0.067]
- S-PAI − D3 2-seed mean: +0.014 [−0.021, +0.050]
- S-PAI − R8K-s2: +0.014 [−0.025, +0.053]

On val160 the new and old scorers give identical paired differences, as in step 4 (the new scorer changes only 12–13 scenes per read, by nearly the same amount).

## Reading

The road hinge on the student's own PAI states does what it targets: offroad zeros fall on both exams (281: 13 → 6; val160: 6 → 1). This is the
first PAI-state S arm with a road term that trains on its own rows: decisions 367 / 402 had the agent hinge only, and decision 420 put road_pai on human rows.

The registered mean does not move on 281, for two reasons:
- route-turn zeros rise (+3), along with a few partial-score points (slow +2);
- one seed's spread is larger than the gain: D3-s0 − D3-s1 is +0.039.

The 2-seed comparison (+0.024 [−0.002, +0.049]) and val160 (+0.025) point the same way as the offroad shift, but neither is the registered read.

## Cost and disk

- GPU job-hours (lane jobs, both attempts): roll 8.10, gates 0.63, training 1.08, PAI 281 read 3.07, val160 read 1.73; total 14.6.
  The bisect runs add about 0.2 outside the lane log.
- Wall time: 3.0 h from submission (17:30) to DONE (20:30). That includes the gw0 stop at 18:43 and the resume at 18:51; the resumed tail took 1.6 h.
- Disk on the box: about 130 GiB, all kept. Rows 38.0, cache 26.2, on1_PAIDC3 3.5 + tokens 28.8, on1_JTDC2S 27.8 + 3.0, rasters 0.7.
- Frame dumps and rollout.asl files were deleted per chunk after the tag check. The box has 613 GB free (91 % used).
