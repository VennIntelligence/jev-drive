# junction_eyes step 0: the frozen eye on the new rulers

Pre-registration: [plans/2026-10-12-prereg-A.md](../plans/2026-10-12-prereg-A.md) (pushed before any label or read; amendments 1-2). Scripts `scripts/je0.py`, `scripts/je0_read.py`, `lib/je_ruler.py`; box `$DATA_DIR/runs/junction_eyes/step0/` (read `20261012-215143`). Figures [figs/je0_ra.png](../figs/je0_ra.png), [figs/je0_rb.png](../figs/je0_rb.png). Rows: [rows.md](rows.md).

## Gates

- G_lab_cover: **pass**; {"held452": 0.9431009957325747, "navtest16": 0.9448767833981842, "zero98": 0.9725085910652921}
- G_lab_sign: **pass**; {"held452": {"L": 1.529674768447876, "C": -0.18275438249111176, "R": -1.9325469732284546, "ok": true}, "navtest16": {"L": 1.6073853969573975, "C": -0.07250278443098068, "R": -1.7903637886047363, "ok": true}, "zero98": {"L": 3.831037998199463, "C": 1.6997897624969482, "R": -0.6525657773017883, "ok": true}}
- G_lab_sides: **pass**; {"scenes": 20, "tokens": 155, "points": 2088, "median": 0.0, "p90": 0.010296211577951908, "both_defined_share": 0.8980645161290323, "rb_agree": 1.0}
- G_split: **pass**; {"token": {"overlap": 0, "first": []}, "scene": {"overlap": 0, "first": []}, "log": {"overlap": 0, "first": []}, "unit": {"overlap": 0, "first": []}}
- G_dump: **FAIL**; {"run": false, "wording": "reconstructed slot pairs", "reason": "G-dump not run on J_on: the held-452 validation of the reconstruction (600 decisions) is not within fp16 noise (token mean abs 8.5e-3 against a noise floor of 8.5e-4, per-row max p99 1.4); the coordinator's rule stops before J_on. Numbers: runs/junction_eyes/slots/validate.json and validate_fwd.json"}
- G_ref: **FAIL**; {"note": "[Errno 2] No such file or directory: '/root/autodl-tmp/ujs/runs/op_probe/feats/WA/lb_navtest-gref.npz'"}
- G_repro: **pass**; {"P_row": {"old": {"held_turn20": 0.9342325269808995, "navtest_near": 1.1072536179643684}, "old-s0": {"held_turn20": 1.0738983308530747, "navtest_near": 1.3135999476630618}, "old-s1": {"held_turn20": 0.9925097546938555, "navtest_near": 1.0627397428398284}}, "held_range": [0.85, 1.046], "navtest_range": [1.126, 1.398], "P_row_in_range": false, "c": {"held_turn20": 0.9231268433654183, "held_seeds": 

## Z-0 line (registered)

Close iff R-A s = 20 m, turn decisions, near bucket (in junction + 0-20 m) median < 0.5 m on held-452 and on navtest-16, and R-B off-road IoU (0-20 m) in junction / 20-60 m >= 0.8 on both.

| quantity | held-452 | navtest-16 |
|:--|--:|--:|
| R-A s = 20 near, median (m) | 1.52 [1.33, 1.74] (n 1150) | 0.99 [0.81, 1.18] (n 776) |
| R-B off-road IoU in junction / 20-60 m | 0.76 [0.72, 0.81] (0.693 / 0.910) | 0.89 [0.84, 0.96] (0.782 / 0.878) |

**Verdict: go to step 1.**

## R-A: turn decisions, error at s = 20 m by distance to the junction

| read set | predictor | in junction | 0-10 m | 10-20 m | 20-40 m | 40-60 m | none within 60 m | near | far |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| navtest16 | probe, Cinque tokens (2 seeds) | 0.90 (220) | 1.00 (433) | 1.19 (123) | 0.46 (185) | 0.45 (85) | 0.52 (491) | 0.99 [0.81, 1.18] | 0.46 [0.40, 0.53] |
| navtest16 | blind (ego + command) | 1.58 (220) | 2.48 (433) | 2.53 (123) | 0.95 (185) | 1.43 (85) | 1.31 (491) | 2.13 [1.74, 2.58] | 1.09 [0.71, 1.59] |
| navtest16 | Cinque slot 7 only | 0.90 (220) | 1.11 (433) | 1.29 (123) | 0.50 (185) | 0.57 (85) | 0.59 (491) | 1.06 [0.92, 1.22] | 0.53 [0.39, 0.73] |
| held452 | probe, Cinque tokens (2 seeds) | 1.67 (372) | 1.50 (538) | 1.34 (240) | 0.62 (141) | 0.61 (84) | 0.90 (725) | 1.52 [1.33, 1.74] | 0.61 [0.50, 0.78] |
| held452 | blind (ego + command) | 3.01 (372) | 4.56 (538) | 3.65 (240) | 2.52 (141) | 1.39 (84) | 2.30 (725) | 3.78 [3.27, 4.30] | 1.81 [1.47, 2.53] |
| zero98 | probe, Cinque tokens (2 seeds) | 3.19 (117) | 3.24 (73) | 9.70 (12) | 1.99 (6) | 1.15 (8) | 3.29 (75) | 3.24 [2.28, 4.47] | 1.76 [1.28, 1.93] |
| zero98 | blind (ego + command) | 7.95 (117) | 8.93 (73) | 17.85 (12) | 5.53 (6) | 6.41 (8) | 7.37 (75) | 8.61 [6.24, 13.17] | 5.61 [3.52, 10.13] |

Cells: median over decisions of the per-decision median point error (C, L, R), m; (n decisions); brackets: 95 % cluster bootstrap (log segments / scenes), B 2000 on the registered cells. WA-Cf and Cinque-slot-7 rows are trained on J_nav only and read on real frames only (reference rows, not methods).

R-A by arc length, turn decisions, near bucket (exploration):

| read set | predictor | s = 5 | 10 | 15 | 20 | 30 |
|:--|:--|--:|--:|--:|--:|--:|
| navtest16 | probe, Cinque tokens (2 seeds) | 0.42 | 0.56 | 0.73 | 0.99 | 1.47 |
| navtest16 | blind (ego + command) | 0.56 | 0.87 | 1.49 | 2.13 | 4.01 |
| navtest16 | Cinque slot 7 only | 0.42 | 0.57 | 0.76 | 1.06 | 1.64 |
| held452 | probe, Cinque tokens (2 seeds) | 0.56 | 0.78 | 1.13 | 1.52 | 2.09 |
| held452 | blind (ego + command) | 0.87 | 1.44 | 2.46 | 3.78 | 6.55 |
| zero98 | probe, Cinque tokens (2 seeds) | 1.76 | 2.64 | 2.85 | 3.24 | 4.34 |
| zero98 | blind (ego + command) | 3.61 | 5.79 | 7.03 | 8.61 | 10.89 |

Straight decisions, s = 20 m (G2 baseline for step 1):

| read set | predictor | near | far | all |
|:--|:--|--:|--:|--:|
| navtest16 | probe, Cinque tokens (2 seeds) | 0.28 | 0.14 | 0.17 |
| navtest16 | blind (ego + command) | 0.43 | 0.22 | 0.27 |
| held452 | probe, Cinque tokens (2 seeds) | 0.44 | 0.30 | 0.31 |
| held452 | blind (ego + command) | 0.75 | 0.63 | 0.57 |

## R-B: IoU, cells 0-20 m ahead, turn decisions

| read set | predictor | class | in junction | 0-10 m | 10-20 m | 20-40 m | 40-60 m | none within 60 m | in / far |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| navtest16 | probe | off-road | 0.78 | 0.81 | 0.83 | 0.86 | 0.91 | 0.89 | 0.89 [0.84, 0.96] |
| navtest16 | probe | drivable | 0.89 | 0.91 | 0.89 | 0.85 | 0.85 | 0.90 | 1.05 [1.00, 1.11] |
| navtest16 | probe | oncoming | 0.55 | 0.47 | 0.57 | 0.77 | 0.69 | 0.74 | 0.74 [0.60, 0.90] |
| navtest16 | blind | off-road | 0.50 | 0.45 | 0.54 | 0.62 | 0.68 | 0.60 | 0.78 [0.69, 0.91] |
| navtest16 | blind | drivable | 0.76 | 0.75 | 0.70 | 0.63 | 0.57 | 0.68 | 1.24 [1.14, 1.36] |
| navtest16 | blind | oncoming | 0.12 | 0.05 | 0.15 | 0.36 | 0.34 | 0.25 | 0.34 [0.18, 0.72] |
| held452 | probe | off-road | 0.69 | 0.73 | 0.78 | 0.90 | 0.92 | 0.82 | 0.76 [0.72, 0.81] |
| held452 | probe | drivable | 0.84 | 0.88 | 0.85 | 0.85 | 0.89 | 0.84 | 0.97 [0.91, 1.03] |
| held452 | probe | oncoming | 0.38 | 0.38 | 0.46 | 0.69 | 0.67 | 0.56 | 0.55 [0.45, 0.71] |
| held452 | blind | off-road | 0.39 | 0.36 | 0.43 | 0.53 | 0.67 | 0.52 | 0.68 [0.59, 0.77] |
| held452 | blind | drivable | 0.71 | 0.71 | 0.66 | 0.53 | 0.57 | 0.61 | 1.31 [1.13, 1.51] |
| held452 | blind | oncoming | 0.10 | 0.07 | 0.04 | 0.12 | 0.19 | 0.08 | 0.67 [0.33, 1.54] |
| zero98 | probe | off-road | 0.64 | 0.66 | 0.67 | 0.88 | 0.93 | 0.61 | 0.71 [0.64, 0.81] |
| zero98 | probe | drivable | 0.71 | 0.80 | 0.75 | 0.67 | 0.83 | 0.62 | 0.95 [0.79, 1.31] |
| zero98 | probe | oncoming | 0.29 | 0.23 | 0.40 | 0.41 | 0.70 | 0.23 | 0.57 [0.30, 47.64] |
| zero98 | blind | off-road | 0.33 | 0.36 | 0.25 | 0.65 | 0.48 | 0.38 | 0.61 [0.47, 0.84] |
| zero98 | blind | drivable | 0.55 | 0.64 | 0.48 | 0.44 | 0.38 | 0.46 | 1.35 [0.92, 2.11] |
| zero98 | blind | oncoming | 0.02 | 0.03 | 0.14 | 0.11 | 0.11 | 0.05 | 0.20 [0.04, 2.35] |

IoU pooled over the decisions of a bucket (sum of intersections / sum of unions); in junction every lane cell is class 1 by definition, so only the off-road class is read there.

## R-C: share of R-A label points outside the wide slot (|azimuth| > 29.4 deg) and the road slot (> 15.7 deg), turn decisions

| read set | bucket | decisions | wide s10 | wide s15 | wide s20 | road s15 | road s20 | centre only, wide s20 | R-B off-road + oncoming cells beyond 29.4 deg |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| navtest16 | in junction | 221 | 0.28 | 0.40 | 0.47 | 0.75 | 0.83 | 0.45 | 0.59 |
| navtest16 | 0-10 m | 437 | 0.31 | 0.59 | 0.79 | 0.85 | 0.91 | 0.80 | 0.74 |
| navtest16 | 10-20 m | 123 | 0.07 | 0.13 | 0.29 | 0.41 | 0.61 | 0.31 | 0.87 |
| navtest16 | 20-40 m | 185 | 0.03 | 0.06 | 0.10 | 0.19 | 0.21 | 0.12 | 0.74 |
| navtest16 | 40-60 m | 85 | 0.11 | 0.16 | 0.22 | 0.44 | 0.54 | 0.20 | 0.69 |
| navtest16 | none within 60 m | 491 | 0.09 | 0.16 | 0.24 | 0.45 | 0.60 | 0.21 | 0.71 |
| navtest16 | near | 781 | 0.26 | 0.46 | 0.62 | 0.75 | 0.84 | 0.62 | 0.72 |
| navtest16 | far | 270 | 0.05 | 0.09 | 0.14 | 0.27 | 0.32 | 0.14 | 0.72 |
| held452 | in junction | 374 | 0.36 | 0.42 | 0.46 | 0.64 | 0.71 | 0.46 | 0.55 |
| held452 | 0-10 m | 545 | 0.30 | 0.51 | 0.67 | 0.74 | 0.84 | 0.68 | 0.76 |
| held452 | 10-20 m | 240 | 0.12 | 0.15 | 0.27 | 0.42 | 0.61 | 0.30 | 0.82 |
| held452 | 20-40 m | 141 | 0.11 | 0.21 | 0.27 | 0.40 | 0.46 | 0.27 | 0.71 |
| held452 | 40-60 m | 84 | 0.09 | 0.13 | 0.17 | 0.37 | 0.50 | 0.13 | 0.67 |
| held452 | none within 60 m | 725 | 0.15 | 0.18 | 0.25 | 0.45 | 0.55 | 0.23 | 0.71 |
| held452 | near | 1159 | 0.28 | 0.41 | 0.52 | 0.64 | 0.75 | 0.53 | 0.70 |
| held452 | far | 225 | 0.10 | 0.18 | 0.23 | 0.39 | 0.47 | 0.22 | 0.69 |
| zero98 | in junction | 117 | 0.81 | 0.81 | 0.80 | 0.88 | 0.86 | 0.81 | 0.50 |
| zero98 | 0-10 m | 73 | 0.63 | 0.66 | 0.71 | 0.80 | 0.88 | 0.70 | 0.64 |
| zero98 | 10-20 m | 12 | 0.75 | 0.65 | 0.62 | 0.88 | 0.74 | 0.58 | 0.60 |
| zero98 | 20-40 m | 6 | 0.56 | 0.61 | 0.78 | 0.94 | 1.00 | 1.00 | 0.54 |
| zero98 | 40-60 m | 8 | 0.62 | 0.62 | 0.62 | 0.67 | 0.71 | 0.62 | 0.58 |
| zero98 | none within 60 m | 75 | 0.64 | 0.61 | 0.65 | 0.83 | 0.83 | 0.63 | 0.57 |
| zero98 | near | 202 | 0.74 | 0.75 | 0.76 | 0.85 | 0.86 | 0.76 | 0.55 |
| zero98 | far | 14 | 0.60 | 0.62 | 0.69 | 0.79 | 0.83 | 0.79 | 0.56 |

Registered 4.1 rule (navtest-16, turn, near): share beyond 29.4 deg at s = 15 0.46, at s = 20 0.62 -> field arms (L-W90, L-3V) RUN; F0 G1 at s = 15 only: True. Recommended step-1 arm set: L-F0, L-F0-nav, U-F0, L-W90, L-3V.

