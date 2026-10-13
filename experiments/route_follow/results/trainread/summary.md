# Training-scene read (TRAINREAD), 2026-10-13

Question: do the policies drive well in closed loop on the scenes they were trained on? Reads @18 decisions, same serving as the held reads (rm_chain readlist). Train = `alpasim_nuplan/onpol-train` scenes: turn class = rf_follow's turn rule on the log's travel (policy-free pre-screen, all turn scenes or a fixed-seed 80 sample) + 40 fixed-seed non-turn controls. Held = the 460 held reads. Scene counts (common to the models) train {'turn': 64, 'nonturn': 40}, held {'turn': 24, 'nonturn': 428}. Class = the policy-free pre-screen, identical across models and sets; turn completion / outcomes / inside-cut use rf_follow's own per-rollout turn flag (counts below). CIs: log-clustered bootstrap, 10 000 draws.

## Train vs held

| model | set | turn n | turn mean | turn zeros (col / off / cor / other) | non-turn mean | non-turn zeros | rf_follow turns | completion % | completed / event / off / short | short % | inside-cut % |
|:--|:--|--:|:--|:--|:--|:--|--:|:--|:--|:--|:--|
| RSG-L18-s0 | train | 64 | 0.618 [0.511, 0.710] | 19 (4 / 10 / 5 / 0) | 0.909 [0.833, 0.967] | 2 / 40 | 64 | 32.8 [22, 43] | 21 / 23 / 9 / 11 | 17.2 [8, 28] | 54.7 [43, 66] |
| RSG-L18-s0 | held | 24 | 0.921 [0.805, 0.980] | 1 (0 / 1 / 0 / 0) | 0.879 [0.853, 0.904] | 25 / 428 | 24 | 66.7 [45, 86] | 16 / 1 / 0 / 7 | 29.2 [10, 50] | 54.2 [30, 75] |
| RSGboth-L18-s0 | train | 64 | 0.719 [0.612, 0.816] | 16 (4 / 12 / 0 / 0) | 0.871 [0.737, 0.974] | 5 / 40 | 64 | 57.8 [47, 69] | 37 / 18 / 4 / 5 | 7.8 [2, 15] | 56.2 [44, 68] |
| RSGboth-L18-s0 | held | 24 | 0.955 [0.847, 1.000] | 1 (0 / 1 / 0 / 0) | 0.916 [0.892, 0.942] | 32 / 428 | 24 | 91.7 [80, 100] | 22 / 1 / 0 / 1 | 4.2 [0, 11] | 54.2 [33, 70] |
| RB0-F-s0 | train | 64 | 0.497 [0.392, 0.596] | 29 (7 / 14 / 8 / 0) | 0.890 [0.775, 0.974] | 3 / 40 | 64 | 28.1 [17, 40] | 18 / 31 / 9 / 6 | 9.4 [3, 16] | 48.4 [37, 60] |
| RB0-F-s0 | held | 24 | 0.805 [0.643, 0.930] | 4 (1 / 2 / 1 / 0) | 0.838 [0.799, 0.876] | 47 / 428 | 24 | 54.2 [38, 72] | 13 / 4 / 2 / 5 | 20.8 [6, 33] | 54.2 [29, 76] |

Train - held (unpaired: different scenes), turn class unless stated; completion / short / inside-cut in pp:

| model | turn mean | turn zero rate (pp) | non-turn mean | non-turn zero rate (pp) | completion | short | inside-cut |
|:--|:--|:--|:--|:--|:--|:--|:--|
| RSG-L18-s0 | -0.303 [-0.431, -0.153] | +25.5 [+10, +40] | 0.030 [-0.050, 0.095] | -0.8 [-7, +7] | -33.9 [-56, -10] | -12.0 [-35, +10] | +0.5 [-24, +27] |
| RSGboth-L18-s0 | -0.236 [-0.360, -0.091] | +20.8 [+6, +33] | -0.045 [-0.182, 0.062] | +5.0 [-6, +19] | -33.9 [-49, -18] | +3.6 [-6, +13] | +2.1 [-18, +25] |
| RB0-F-s0 | -0.308 [-0.474, -0.118] | +28.6 [+9, +46] | 0.052 [-0.071, 0.148] | -3.5 [-13, +9] | -26.0 [-47, -7] | -11.5 [-26, +5] | -5.7 [-31, +22] |

Paired model comparisons (same scenes, A - B):

| pair | turn mean | turn zero rate (pp) | non-turn mean | non-turn zero rate (pp) | completion (pp) |
|:--|:--|:--|:--|:--|:--|
| RSGboth-L18-s0 - RSG-L18-s0 @train | 0.100 [0.004, 0.203] | -4.7 [-16, +5] | -0.038 [-0.136, 0.041] | +7.5 [+0, +16] | +25.0 [+13, +39] |
| RSG-L18-s0 - RB0-F-s0 @train | 0.121 [0.059, 0.184] | -15.6 [-23, -7] | 0.019 [-0.013, 0.076] | -2.5 [-8, +0] | +4.7 [-3, +14] |
| RSGboth-L18-s0 - RB0-F-s0 @train | 0.222 [0.111, 0.330] | -20.3 [-31, -10] | -0.019 [-0.103, 0.043] | +5.0 [+0, +13] | +29.7 [+16, +43] |
| RSGboth-L18-s0 - RSG-L18-s0 @held | 0.034 [0.010, 0.072] | +0.0 [+0, +0] | 0.037 [0.015, 0.063] | +1.6 [+0, +3] | +25.0 [+5, +48] |
| RSG-L18-s0 - RB0-F-s0 @held | 0.115 [0.002, 0.253] | -12.5 [-27, +0] | 0.040 [0.016, 0.067] | -5.1 [-8, -3] | +12.5 [-4, +30] |
| RSGboth-L18-s0 - RB0-F-s0 @held | 0.149 [0.040, 0.289] | -12.5 [-27, +0] | 0.078 [0.049, 0.109] | -3.5 [-6, -1] | +37.5 [+16, +57] |

## Notes

- Pick: 1031 onpol-train scenes screened (policy-free, scripts/rf_trainread.py pick); 1021 screened ok (10 without a route), 64 strict turns (all taken, under the cap of 80, so no sampling) + 40 fixed-seed (seed 0) non-turn controls; ids in scenes.txt, class in scenes.csv, all screens in prescreen.csv. rf_follow found the same 64 turns in every model's rollouts.
- Reads: @18 decisions, rm_chain readlist (the held reads' serving), 3 chunks per model; held numbers come from the stored held reads (held 460: 24 turns, 428 non-turn common scenes). Chain: scripts/rf_trainread.sh. Raw: follow.{json,md}, inside.json (rf_inside, train pairs), summary.json.
- Train vs held are different scenes, so their differences are unpaired (logs resampled independently); model-vs-model rows are paired on the same scenes. Train turns come from 64 scenes of many logs; all CIs log-clustered.
- Caveats: the train-turn sample is not matched to the held turns in difficulty (the untrained-on-turns base RB0-F-s0 is equally lower on train turns, -0.31), so a train - held gap in level mixes scene difficulty with fit; the within-set model ordering (RSGboth > RSG > RB0) is the cleaner read. Train scenes supplied the on-policy rows of RSG / RSGboth, RB0-F-s0 saw none of them.
