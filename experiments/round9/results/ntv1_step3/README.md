# NTV1 step 3: P1L + its own DAgger rows (round 9, 2026-10-24, decision 471)

Pre-registration (Chinese): [../../plans/2026-10-24-ntv1-step3.md](../../plans/2026-10-24-ntv1-step3.md), with amendments 1-2. Lane:
[../../scripts/dgt.sh](../../scripts/dgt.sh); gates and verdicts: [../../scripts/dgt.py](../../scripts/dgt.py). Box: `$DATA_DIR/runs/round9/ntv1_step3/`.
Full tables:
- [verdict.md](verdict.md): seed 0, Harmonizer OFF, the registered read;
- [cand.md](cand.md): the candidate tail, seeds 0 / 1, Harmonizer OFF / ON;
- [pai281.md](pai281.md): old scorer only.

The new scorer is the primary scorer.

## Arms (from the shipped weights, P1L's argv, eager, 19 500 steps, S rows from T = 10 000)

| arm | tag | change vs P1L (PCL1L-F-s0) | batch from T |
|:--|:--|:--|:--|
| D1 | NT3D1-F-s0 (+ NT3D1-F-s1) | `--on on1_JTDC --on-wfile runs/round9/ntv1_step2/w/NTV1D.npz --on-k 40` | 75 navtrain + 13 yaw + 40 pool (32 S + 8 DAgger) |
| D2 | NT3D2-F-s0 | D1 + `--ho-distill 0` (the S rows' distill_ho x 0, decision 464) | same |

Before T, both arms run the same batches as P1L (115 + 13).

- **S part.** The S rows of NTV1D are JT00S-rb0's 13 793 rb0 rows, row for row, uniform, scaled to a share of 0.8.
- **DAgger part.** The DAgger part is on1_PAIDC's 92 561 drawable rows (step 2a, teacher b3), uniform, share 0.2.
- **Row treatment, checked against R6DAG-L18-s0.** Every per-row flag is identical: on-mode mixed, on-lon-all, on-normal, ho-w 3, agent-lam 10,
  agent-on-only, shape, all lat-* flags, on-arc / on-arc-rows and on-hold-pct. In both pools the DAgger rows are `both` rows, which gives them:
  - imitation to the b3 label, plus distill;
  - the agent hinge (agent ok 100 %);
  - no road hinge (road ok 0 %) and no lateral terms: the L18NR1v2 raster covers only the first 34 918 pool rows.

  `--ho-distill` does not touch them. The argv differences that are not per-row are listed in
  [gr_row_treatment.json](gr_row_treatment.json): R6DAG has `--init NTV0`, 9 500 steps, batch 136 and yr-mass 0.095, and also
  `--ego-scale-*`, which acts on non-pool rows only. It also uses a different pool and different shares.

## Gates (all pass; lane submitted from f5d4662f)

| gate | result |
|:--|:--|
| G0a P1L's argv vs stored PCL1L-F-s0, steps <= 100 | 131 / 131 bit-identical |
| G0b D1 before T vs stored P1L | 131 / 131 bit-identical; D2 the same (reported) |
| G0c weight-0 pair (step 2a's T3 b on HEAD, T = 30) | 231 / 231 identical, pool rows drawn |
| G-W draw file / G-R row treatment | pass (see above) |
| 60-step smokes, T = 30 | batch 75 / 13 / 40 mixed; S share after T 0.813; draw shares S 0.8 / pd 0.2; no S terms before T; lr formula exact |
| D2 switch | smoke scalars <= 30 equal to D1 (79 / 79); 59 post-T loss scalars differ |
| S / imit gradient ratio, 500 steps after T (band 0.21-1.04) | D1 0.310, D2 0.351, D1-s1 0.597 |
| post: dev ADE <= 0.566, lr over the run | D1 0.530, D2 0.529, D1-s1 0.537 |

**Process fault (my lane script).** Attempt 1 of the report job failed at scorer_v2: the lane's reads had no entry in scorer_v2.py's READS. I added
the entries and resubmitted only the report job (`dgt.sh rereport`); the trainings and reads were kept.

## PAI 281 ACC + GAP, Harmonizer OFF, seed 0 (276 common scenes, new / old scorer)

| read | new | old | zeros new / old | lateral | route-turn | AF-front | neg. flips A / B on NTV0's full scenes |
|:--|--:|--:|--:|--:|--:|--:|--:|
| **D1** | **0.867** | 0.798 | 27 / 46 | 18 | 3 | 5 | 14 / 8 |
| **D2** | **0.829** | 0.746 | 36 / 59 | 24 | 3 | 8 | 22 / 13 |
| P1L | 0.811 | 0.732 | 39 / 61 | 23 | 10 | 4 | 16 / 12 |
| NTV0-s0 | 0.813 | 0.741 | 39 / 59 | 18 | 15 | 3 | - |
| R8K-s2 | 0.872 | 0.803 | 28 / 47 | 20 | 2 | 5 | 14 / 9 |
| INTP75-NR (ref., not clean) | 0.878 | 0.809 | 26 / 45 | 17 | 3 | 5 | 8 / 6 |

The two flip counts:
- **A** (INTERP's count): NTV0 new score >= 0.98 on 200 scenes, and the read's new score < 0.98.
- **B** (step 1's count): NTV0 old score >= 0.98 on 180 scenes, and the read's new score = 0.

Lateral zeros = offroad + zero-other + side.

| pair (new scorer) | mean | zeros (old) | route-turn |
|:--|--:|--:|--:|
| D1 - P1L | **+0.055 [+0.016, +0.096]** (old +0.066) | -15 [-27, -4] | -7 [-12, -2] |
| D1 - NTV0-s0 | +0.054 [+0.012, +0.095] | -13 | -12 |
| D1 - R8K-s2 | -0.005 [-0.039, +0.030] | -1 | +1 |
| D1 - INTP75 | -0.012 [-0.044, +0.021] | +1 | 0 |
| D2 - D1 | **-0.038 [-0.065, -0.013]** (old -0.052) | +13 [+5, +22] | 0 |

Against D1, D2 has more overshoot (+4), offroad (+4), side (+4) and AF-front (+3) zeros.

## Registered lines (new scorer; the old scorer gives the same reading)

- **D1 - P1L = +0.055 >= +0.02: "own-state DAgger helps".**
  - It fixes route turns (10 -> 3) and overshoot / slow, and lateral zeros fall slightly (23 -> 18).
  - It matches R8K-s2 with clean data (one seed).
  - Negative flips on NTV0's full-score scenes: 14 / 8, the same as R8K-s2's 14 / 9 and above INTP75's 8.
- **D2 - D1 = -0.038 < -0.01: "distill_ho still needed in the DAgger recipe".** Decision 464 found it not needed without DAgger rows (one seed).
- **Candidate rule triggered: D1 seed 0 reads 0.867 >= 0.86**, so seed 1 and the Harmonizer-ON reads ran automatically.

## Candidate tail: D1 seeds 0 / 1, Harmonizer OFF / ON ([cand.md](cand.md))

| | OFF new / old | ON new / old | (OFF + ON) / 2 new |
|:--|--:|--:|--:|
| D1-s0 | 0.867 / 0.798 | 0.851 / 0.772 | 0.859 |
| D1-s1 | 0.837 / 0.761 | 0.824 / 0.748 | 0.831 |
| seed mean | **0.852** / 0.779 | **0.837** / 0.760 | 0.845 |
| P1L | 0.811 | 0.804 | 0.808 |
| NTV0-s0 | 0.813 | 0.809 | 0.811 |
| R8K-s2 | 0.872 | 0.866 | 0.869 |

The OFF reads use 276 common scenes and the ON reads 278.

- **Seed 1 is lower by 0.030 [-0.057, -0.004] (OFF).** Its extra losses are rear-hit partials (+9) and two drift zeros.
- **Seed mean against the references.**
  - vs P1L: OFF +0.040 [+0.002, +0.081], ON +0.034 [-0.003, +0.071].
  - vs NTV0-s0: OFF +0.039 [-0.001, +0.079], ON +0.029.
  - vs R8K-s2: OFF -0.020 [-0.054, +0.014], ON -0.028 [-0.058, +0.001].
- **Negative flips (count A).** s1 OFF has 21. With Harmonizer ON: s0 18, s1 21, R8K-s2 11.
- **Reading.** The DAgger gain over P1L holds across both seeds, at about +0.04. Seed 0's parity with R8K is partly a good seed: the two-seed mean is
  about 0.02 under R8K-s2, and further under it with Harmonizer ON.

## Highway (behavior_stats Q1)

| read | free-road median v / v_human | share < 0.85 | slow + rear-hit points | rule result |
|:--|--:|--:|--:|:--|
| D1 | 1.00 | 8 % | 2.20 | "slow on highway" |
| D2 | 1.00 | 10 % | 2.27 | "slow on highway" |
| P1L | 1.00 | 10 % | 1.10 | not slow |
| NTV0 | 1.00 | 9 % | 1.06 | not slow |
| R8K-s2 | 1.00 | 11 % | 0.20 | not slow |

D1 trips only the points rule (>= 2.0). The speed ratio is not low: the median is 1.00 and the > 1.10 share is 22 %, as R8K's. The points come
from a few highway-end slow / rear-hit partials.

## Cost

33.9 GPU job-hours in total, wall 6.2 h:
- training 6.2 job-hours (3 runs);
- reads 27.4 job-hours: 5 reads, the ON reads book 30 GB each, 3-4 reads a card;
- gates 0.24 job-hours.

The seed-0 part alone was 10.8 job-hours over 2.7 h.
