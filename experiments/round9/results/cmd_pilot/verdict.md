# CMD-PILOT verdict (PAI 281, ACC + GAP, Harmonizer OFF; HF / HP50 served with JEV_CMD=heading)

Old scorer: 279 scenes common to the five reads without RPC error; RPC scenes per read {'HF': 0, 'HP50': 0, 'RB0F': 0, 'RB0P50': 0, 'NTV0': 0} (> 5 = re-read before judging).

| read | mean (old) | lateral zeros | zeros | z_offroad | z_zero-other | z_side | z_route-turn | z_overshoot | z_AF-front | p_rear-hit | p_slow | launch | dev ADE |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| RB0F | 0.647 | 37 | 88 | 19 | 14 | 4 | 21 | 22 | 8 | 23 | 8 | 32/37 | 0.5275979042053223 |
| HF | 0.655 | 36 | 87 | 18 | 14 | 4 | 19 | 22 | 10 | 22 | 6 | 33/37 | 0.5343566536903381 |
| RB0P50 | 0.602 | 60 | 102 | 28 | 24 | 8 | 11 | 21 | 10 | 18 | 3 | 27/37 | 0.5504305362701416 |
| HP50 | 0.613 | 52 | 100 | 20 | 23 | 9 | 11 | 26 | 11 | 15 | 4 | 28/37 | 0.5589806437492371 |
| NTV0 | 0.743 | 18 | 59 | 12 | 6 | 0 | 15 | 23 | 3 | 26 | 11 | 33/37 | - |

Lateral zeros = zeros with fclass in {offroad, zero-other, side} (old-scorer facts; the scorer fix does not touch these classes).

| lateral-zero table | F (all-row base) | P50 (58 of 115 imitation rows PAI human) | P50 - F |
|:--|--:|--:|--:|
| old command (RB0-F-s0 / RB0-P50-s0) | 37 | 60 | D_old = +23 |
| heading command, all rows (HF / HP50) | 36 | 52 | **D_new = +16** |
| route-turn zeros old / heading | 21 / 19 | 11 / 11 | |

## Paired differences

| pair | old scorer | new scorer |
|:--|--:|--:|
| HF - RB0F | +0.008 [-0.022, +0.038] | +0.012 [-0.019, +0.042] |
| HP50 - RB0P50 | +0.011 [-0.023, +0.045] | +0.025 [-0.009, +0.061] |
| HP50 - HF | -0.043 [-0.093, +0.008] | -0.028 [-0.078, +0.022] |
| RB0P50 - RB0F | -0.045 [-0.098, +0.007] | -0.042 [-0.095, +0.011] |

New scorer (scorer_v2, decision 453): 279 scenes common to the five reads; means RB0F 0.715, RB0P50 0.673, NTV0 0.815, HF 0.727, HP50 0.699; zeros RB0F 69, RB0P50 82, NTV0 39, HF 67, HP50 76.

## HF guard against RB0-F-s0 (design section 5)

- new-scorer mean >= RB0-F - 0.03: pass
- route-turn <= RB0-F + 3: pass
- lateral zeros <= RB0-F + 5: pass
- launch >= 30 / 37: pass
- dev ADE <= 0.538: pass
  (new-scorer mean diff +0.012; route-turn 19 vs 21; lateral 36 vs 37; launch 33/37; dev ADE 0.5343566536903381)

**Line: between: partial -> propose seed 1 for both arms (~2.6 GPU h), nothing run**

## Listed scenes (old score / new score, fclass)

| scene | what | RB0F | HF | RB0P50 | HP50 |
|:--|:--|--:|--:|--:|--:|
| 1a69f757 | long gentle right curve (off_straight) | 0.00 / 0.00 AF-front | 0.00 / 0.00 AF-front | 0.00 / 0.00 offroad | 0.00 / 0.00 offroad |
| 1e8e939e | S-junction (off_junc_inside) | 0.00 / 0.00 route-turn | 0.00 / 0.00 route-turn | 0.00 / 0.00 offroad | 1.00 / 1.00 ok |
| baabc57d | command flip before the junction | 1.00 / 1.00 ok | 1.00 / 1.00 ok | 0.00 / 0.00 offroad | 1.00 / 1.00 ok |
| 596af0c7 | gentle right curve | 0.00 / 0.00 offroad | 0.00 / 0.00 offroad | 0.00 / 0.00 offroad | 0.00 / 0.00 offroad |

New zeros on NTV0-L18-s0's full-score scenes (>= 0.98, 184 scenes): RB0F 24, HF 23, RB0P50 45, HP50 43

## Served command vs the rule (G6, >= 99.9 %)

- HF: HF-s0-ag 55494 inferences, agree 1.00000 -> pass
- HP50: HP50-s0-ag 55693 inferences, agree 1.00000 -> pass

## Training command record (rm_train cmd_all, full runs)

- HF: rows 172860, changed 28164, kept (no memory) 3608; navtrain_full 16.7% (L/R 36.7% -> 41.1%), dev 15.0% (L/R 31.5% -> 36.0%), yr1_navtrain_full 15.7% (L/R 37.1% -> 39.0%)
- HP50: rows 701407, changed 151967, kept (no memory) 6006; navtrain_full 16.7% (L/R 36.7% -> 41.1%), dev 15.0% (L/R 31.5% -> 36.0%), yr1_navtrain_full 15.7% (L/R 37.1% -> 39.0%), rb0 26.2% (L/R 54.1% -> 30.5%), rtb1 25.4% (L/R 52.6% -> 29.9%), ys 28.2% (L/R 55.9% -> 30.6%), y 27.5% (L/R 55.6% -> 31.1%), tv 24.7% (L/R 16.9% -> 31.9%), ph 22.2% (L/R 44.5% -> 24.3%), ph05 22.4% (L/R 44.9% -> 24.6%), ph15 22.0% (L/R 45.6% -> 25.5%)
