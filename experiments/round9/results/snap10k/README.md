# SNAP10K: PCL1-F-s0x10000 on PAI 281 ACC + GAP

Plan: [../../plans/2026-10-24-snap10k.md](../../plans/2026-10-24-snap10k.md). Purpose: split one run into the offline part to step 10 000 and the S segment after it.
Snapshot PCL1-F-s0x10000 is tensor-identical to RB0L-F-s0x10000 (206 + 97 tensors, `torch.equal` on CPU; only config metadata differs), so the read stands for both.
Sidecars (adapter.pt, route_mem.json) were present in the snapshot. Serve and chunk lists are P1L's, Harmonizer OFF, scorer_v2 new (old in brackets).
Files: `snap10k.json` (full numbers). Code: `experiments/round9/scripts/snap10k.{sh,py}`. Box dir `$DATA_DIR/runs/round9/snap10k/`.

279 common scenes (0 RPC scenes excluded). New scorer (scorer_v2 new) primary, old in brackets; zero classes from old-scorer facts.

| read | new mean (old) | zeros new (old) | overshoot | route-turn | offroad | AF-front | side | zero-other |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| PCL1-F-s0x10000 | 0.736 (0.665) | 63 (83) | 23 | 16 | 23 | 10 | 4 | 7 |
| RB0-F-s0 | 0.719 (0.651) | 68 (87) | 21 | 21 | 19 | 8 | 4 | 14 |
| RB0L-F-s0 | 0.681 (0.599) | 79 (102) | 24 | 18 | 38 | 10 | 1 | 11 |
| PCL1-F-s0 | 0.794 (0.719) | 44 (65) | 23 | 14 | 16 | 5 | 1 | 6 |
| PCL1L-F-s0 | 0.817 (0.738) | 38 (60) | 23 | 10 | 14 | 4 | 3 | 6 |

| pair (95 % bootstrap CI, scenes as units) | new | old | zeros (old) |
|:--|--:|--:|--:|
| PCL1-F-s0x10000 - RB0-F-s0 | +0.018 [-0.021, +0.057] | +0.014 [-0.028, +0.057] | -4 |
| PCL1-F-s0x10000 - RB0L-F-s0 | +0.055 [+0.019, +0.091] | +0.066 [+0.027, +0.105] | -19 |
| PCL1-F-s0x10000 - PCL1-F-s0 | -0.057 [-0.098, -0.016] | -0.054 [-0.098, -0.009] | +18 |
| PCL1-F-s0x10000 - PCL1L-F-s0 | -0.081 [-0.121, -0.040] | -0.073 [-0.116, -0.032] | +23 |
| PCL1-F-s0 - PCL1-F-s0x10000 | +0.057 [+0.016, +0.098] | +0.054 [+0.009, +0.098] | -18 |
| PCL1L-F-s0 - PCL1-F-s0x10000 | +0.081 [+0.040, +0.121] | +0.073 [+0.032, +0.116] | -23 |
| RB0L-F-s0 - RB0-F-s0 | -0.038 [-0.082, +0.009] | -0.052 [-0.099, -0.005] | +15 |
| PCL1-F-s0 - RB0L-F-s0 | +0.113 [+0.069, +0.157] | +0.120 [+0.074, +0.166] | -37 |
| PCL1L-F-s0 - PCL1-F-s0 | +0.023 [-0.010, +0.057] | +0.020 [-0.015, +0.056] | -5 |

GPU: 3.22 card-hours of job time (3.22 in read chunks).
