# PLAN CREEP replication (rf_creep_rep.py)

Prereg `plans/2026-10-15-creep-replication-prereg.md`. Held 460 @ 18 decisions (0.5 s), the 24 strict held turns of 17 checkpoints decision 308 did not use. Measure = rf_switching Amendment 1 unchanged, sign flipped: **c = creep per metre, + = each new plan's 2 s point lies outside the previous plan's path** (per-turn median of j / ds over the window entry - 20 .. entry + 20 m before the first zero); C = summed creep (m); x2 = car - plan heading toward the turn over 2 s (deg, per-turn median; - = the car turns less). Failed = every non-completed turn. Medians failed / completed, difference [95 % CI], bootstrap B 10 000 over nuPlan-log clusters (every turn of a scene, across checkpoints, in one cluster).

Turns 408 (17 checkpoints; failed 178, completed 230; 24 scenes in 14 log clusters).

Gates: G-code pass (d308 held_all6 recomputed: j per m +0.003 / -0.006, x2 -1.6 / +0.4, 144 turns; switching.md +0.003 / -0.006, -1.6 / +0.4); G-frame pass (worst checkpoint median |anchor - controller| 0.000 m / 0.00 deg); G-human pass (human log as a plan: c_h +0.0025 [+0.0020, +0.0033] m / m, failed +0.0037 / completed +0.0019); G-n pass (408 / 408).

## R1 / R2 (pooled)

| claim | quantity | n f / c | median f / c | diff [CI] | ratio f / c [CI] | verdict |
|:--|:--|:--|:--|:--|:--|:--|
| R1 | creep per m (m / m) | 178 / 230 | +0.038 / +0.000 | +0.037 [+0.009, +0.069] | 99.01 [4.32, 4927.77] (3858 replicates with completed <= 0 left out) | **supported** |
| - | summed creep (m) | 178 / 230 | +0.73 / -0.09 | +0.82 [+0.46, +1.16] | - | descriptive |
| R2 | 2 s heading shortfall (deg) | 156 / 230 | -1.8 / +0.4 | -2.2 [-3.4, -0.5] | - | **supported** |

Reference (d308, PAI RSG-L18-s0, signed j): -0.063 / -0.015 m / m, diff -0.048 [-0.104, -0.016]; x2 -2.5 / +1.6, diff -4.1 [-6.6, -2.3]; held late-wide -0.104.

## R3: classes and outcomes (failed of the class vs all completed turns; descriptive)

| group | n | c f / c | diff [CI] | C f / c (m) | diff [CI] | x2 f / c | diff [CI] |
|:--|--:|:--|:--|:--|:--|:--|:--|
| cls:late_wide | 44 | +0.099 / +0.000 | +0.099 [+0.074, +0.172] | +1.41 / -0.09 | +1.51 [+0.65, +2.69] | -4.2 / +0.4 | -4.6 [-6.2, -1.9] |
| cls:straight | 32 | +0.265 / +0.000 | +0.265 [+0.184, +0.443] | +2.79 / -0.09 | +2.88 [+1.75, +4.39] | -16.1 / +0.4 | -16.5 [-19.2, -11.0] |
| cls:over_turn | 8 | +0.014 / +0.000 | +0.013 [-0.057, +0.051] | +0.09 / -0.09 | +0.18 [-0.82, +0.42] | +2.1 / +0.4 | +1.7 [+1.0, +3.3] |
| cls:wrong_exit | 2 | -0.028 / +0.000 | -0.028 [-0.033, -0.015] | -0.16 / -0.09 | -0.07 [-0.44, +0.23] | +0.0 / +0.4 | -0.4 [-1.1, +0.3] |
| cls:no_zero | 92 | +0.007 / +0.000 | +0.007 [-0.029, +0.025] | +0.28 / -0.09 | +0.37 [-0.41, +0.84] | -0.8 / +0.4 | -1.2 [-2.6, +0.6] |
| outcome:event | 85 | +0.135 / +0.000 | +0.135 [+0.099, +0.218] | +1.47 / -0.09 | +1.56 [+0.69, +3.04] | -5.6 / +0.4 | -6.0 [-10.2, -4.4] |
| outcome:off | 27 | +0.019 / +0.000 | +0.019 [-0.021, +0.040] | +0.58 / -0.09 | +0.68 [-0.34, +1.12] | -0.8 / +0.4 | -1.3 [-2.3, +1.0] |
| outcome:short | 66 | -0.003 / +0.000 | -0.004 [-0.040, +0.022] | +0.25 / -0.09 | +0.34 [-0.67, +0.72] | -0.8 / +0.4 | -1.2 [-3.0, +0.8] |

Families: failed vs completed of the same family (own, where both have >= 3 turns) and failed vs the completed turns of the other families (vs pooled completed).

| family | tags | n f / c | failed classes | own: c diff [CI] | own: x2 diff [CI] | c failed / own completed | vs pooled completed: c f / c, diff [CI] | x2 diff [CI] |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|
| lr | 9 | 73 / 143 | no_zero 62, late_wide 9, over_turn 2 | +0.007 [-0.033, +0.041] | -1.2 [-3.1, +0.7] | +0.006 / -0.001 | +0.006 / +0.004 | +0.002 [-0.037, +0.035] | -1.3 [-2.6, +0.5] |
| RV2c | 2 | 17 / 31 | no_zero 10, over_turn 5, late_wide 2 | +0.012 [-0.031, +0.059] | +0.1 [-2.3, +3.1] | +0.016 / +0.004 | +0.016 / -0.000 | +0.016 [-0.024, +0.062] | +0.3 [-2.3, +2.9] |
| APP0C | 1 | 5 / 19 | no_zero 3, late_wide 2 | +0.058 [-0.000, +0.097] | -3.3 [-5.4, -0.7] | +0.053 / -0.005 | +0.053 / +0.001 | +0.052 [-0.006, +0.077] | -3.3 [-4.8, -0.5] |
| APP2W05 | 1 | 10 / 14 | no_zero 6, late_wide 3, over_turn 1 | +0.011 [-0.019, +0.095] | -2.5 [-3.3, +0.8] | +0.028 / +0.017 | +0.028 / +0.000 | +0.028 [+0.007, +0.105] | -3.0 [-3.5, +0.2] |
| FMH | 2 | 47 / 1 | straight 32, late_wide 15 | - | - | +0.211 / +0.080 | +0.211 / +0.000 | +0.211 [+0.176, +0.330] | -15.9 [-18.7, -9.5] |
| RGH | 2 | 26 / 22 | late_wide 13, no_zero 11, wrong_exit 2 | +0.026 [-0.011, +0.111] | -1.9 [-3.8, -0.1] | +0.030 / +0.004 | +0.030 / +0.000 | +0.030 [+0.002, +0.093] | -1.8 [-3.8, -0.4] |

Sensitivity (registered, descriptive): R1 / R2 quantities on subsets.

| subset | n f / c | c f / c | diff [CI] | x2 f / c | diff [CI] |
|:--|:--|:--|:--|:--|:--|
| zero only (event / off) | 112 / 230 | +0.085 / +0.000 | +0.085 [+0.054, +0.146] | -2.9 / +0.4 | -3.3 [-5.1, -1.6] |
| scene clusters | 178 / 230 | +0.038 / +0.000 | +0.037 [+0.010, +0.075] | -1.8 / +0.4 | -2.2 [-3.8, -0.8] |
| without lr | 105 / 87 | +0.085 / +0.004 | +0.082 [+0.034, +0.147] | -2.8 / +0.5 | -3.2 [-4.7, -1.9] |
| without RV2c | 161 / 199 | +0.039 / -0.000 | +0.039 [+0.014, +0.085] | -1.8 / +0.4 | -2.2 [-3.7, -0.7] |
| without APP0C | 173 / 211 | +0.037 / +0.001 | +0.036 [+0.009, +0.067] | -1.8 / +0.4 | -2.2 [-3.4, -0.5] |
| without APP2W05 | 168 / 216 | +0.039 / +0.000 | +0.038 [+0.007, +0.073] | -1.8 / +0.5 | -2.2 [-3.5, -0.5] |
| without FMH | 131 / 229 | +0.017 / +0.000 | +0.017 [-0.012, +0.043] | -0.8 / +0.4 | -1.3 [-2.8, +0.1] |
| without RGH | 152 / 208 | +0.040 / +0.000 | +0.039 [+0.008, +0.073] | -1.8 / +0.4 | -2.3 [-3.6, -0.4] |

Each checkpoint alone (24 turns; descriptive):

| checkpoint | n f / c | c f / c | diff [CI] | x2 f / c | diff [CI] |
|:--|:--|:--|:--|:--|:--|
| APP0C-L18-s0 | 5 / 19 | +0.053 / -0.005 | +0.058 [-0.000, +0.097] | -2.9 / +0.4 | -3.3 [-5.4, -0.7] |
| APP2W05-L18-s0 | 10 / 14 | +0.028 / +0.017 | +0.011 [-0.019, +0.095] | -2.6 / -0.1 | -2.5 [-3.3, +0.8] |
| FMH-F-s0 | 23 / 1 | +0.209 / +0.080 | +0.130 [+0.056, +0.292] | -15.6 / -1.1 | -14.5 [-18.6, -11.6] |
| FMH-F-s1 | 24 / 0 | - / - | - [-, -] | - / - | - [-, -] |
| RGH-F-s0 | 14 / 10 | +0.030 / +0.005 | +0.025 [-0.011, +0.111] | -1.0 / +0.6 | -1.6 [-3.9, +0.7] |
| RGH-F-s1 | 12 / 12 | +0.030 / +0.004 | +0.026 [-0.016, +0.149] | -1.7 / +0.5 | -2.3 [-4.6, -0.2] |
| RSGlr03n1250-L18-s0 | 10 / 14 | +0.026 / -0.003 | +0.029 [-0.034, +0.075] | -0.7 / +0.4 | -1.1 [-4.2, +0.9] |
| RSGlr03n2500-L18-s0 | 8 / 16 | +0.008 / -0.001 | +0.009 [-0.016, +0.074] | -1.4 / +0.6 | -2.0 [-4.5, +0.3] |
| RSGlr03n5000-L18-s0 | 6 / 18 | +0.013 / +0.000 | +0.013 [-0.046, +0.063] | -1.8 / +0.3 | -2.1 [-4.3, +1.0] |
| RSGlr03n5000-L18-s1 | 6 / 18 | +0.015 / -0.003 | +0.018 [-0.132, +0.063] | -1.5 / +0.4 | -1.9 [-3.5, +1.1] |
| RSGlr10n1250-L18-s0 | 9 / 15 | +0.029 / -0.000 | +0.029 [-0.046, +0.091] | -0.7 / +0.4 | -1.1 [-4.6, +0.5] |
| RSGlr10n5000-L18-s0 | 8 / 16 | +0.003 / -0.008 | +0.011 [-0.034, +0.036] | -1.3 / +0.8 | -2.1 [-3.8, +0.7] |
| RSGlr30n1250-L18-s0 | 10 / 14 | +0.013 / +0.003 | +0.009 [-0.071, +0.040] | -0.6 / +0.2 | -0.8 [-3.6, +1.6] |
| RSGlr30n2500-L18-s0 | 8 / 16 | -0.016 / -0.001 | -0.015 [-0.044, +0.056] | -0.2 / +0.3 | -0.5 [-4.1, +2.0] |
| RSGlr30n5000-L18-s0 | 8 / 16 | -0.012 / +0.005 | -0.017 [-0.045, +0.050] | -0.1 / +0.9 | -1.0 [-2.7, +1.6] |
| RV2c-L18-s0 | 9 / 15 | +0.017 / +0.003 | +0.014 [-0.030, +0.075] | -0.1 / +0.5 | -0.6 [-2.8, +2.7] |
| RV2c-L18-s1 | 8 / 16 | +0.013 / +0.004 | +0.010 [-0.058, +0.052] | +2.1 / +0.6 | +1.5 [-2.2, +4.1] |

## R4: where the heading shortfall starts

Per-turn medians over the window decisions, then the group median [cluster-bootstrap CI], deg toward the turn. x05 = car - plan over 0-0.5 s (plan k alone drives the car), x1 over 0-1 s, x12 over the 1-2 s segment (car's yaw(2 s) - yaw(1 s) vs the plan's psi(2 s) - psi(1 s)), x2 over 0-2 s. Plan / car columns: the plan's own heading change and the car's over 0-1 s and 1-2 s.

| group | turns | x05 | x1 | x12 | x2 | plan 0-1 s / car | plan 1-2 s / car |
|:--|--:|:--|:--|:--|:--|:--|:--|
| completed | 230 | +0.1 [+0.0, +0.2] | +0.3 [-0.1, +0.5] | +0.2 [-0.2, +0.7] | +0.4 [-0.3, +1.2] | +15.1 / +15.3 | +14.2 / +14.8 |
| failed | 156 | -0.1 [-0.2, -0.0] | -0.6 [-0.9, -0.2] | -1.2 [-1.9, -0.0] | -1.8 [-2.8, -0.1] | +12.5 / +11.5 | +13.7 / +12.4 |
| failed late_wide | 35 | -0.3 [-1.4, -0.2] | -1.2 [-3.6, -0.9] | -2.9 [-4.2, -2.0] | -4.2 [-5.7, -1.4] | +15.3 / +12.5 | +19.3 / +14.2 |
| failed straight | 21 | -0.9 [-1.2, -0.5] | -4.5 [-6.0, -3.0] | -12.7 [-13.7, -5.4] | -16.1 [-18.9, -10.3] | +2.9 / -2.3 | +8.0 / -2.5 |
| failed over_turn | 6 | +0.5 [+0.3, +0.9] | +1.1 [+0.6, +3.6] | +1.3 [+1.2, +1.3] | +2.1 [+2.1, +3.4] | +19.1 / +20.1 | +21.3 / +22.3 |
| failed no_zero | 92 | -0.0 [-0.2, +0.1] | -0.2 [-0.6, +0.2] | -0.3 [-1.4, +0.7] | -0.8 [-1.8, +1.0] | +12.6 / +12.2 | +12.7 / +12.5 |

Registered reading on the failed turns: **both**.

## Amendment 1 (post hoc, descriptive)

(a) Without the flow head FMH (15 checkpoints): failed subsets vs the completed turns of the same 15.

| subset | n f / c | c f / c | diff [CI] | C f / c (m) | diff [CI] | x2 f / c | diff [CI] |
|:--|:--|:--|:--|:--|:--|:--|:--|
| zero only (event / off) | 65 / 229 | +0.038 / +0.000 | +0.038 [+0.004, +0.063] | +0.58 / -0.10 | +0.68 [+0.23, +1.13] | -1.2 / +0.4 | -1.7 [-3.2, -0.2] |
| late_wide | 29 / 229 | +0.098 / +0.000 | +0.097 [+0.051, +0.120] | +0.95 / -0.10 | +1.05 [+0.40, +1.51] | -3.9 / +0.4 | -4.3 [-5.5, -1.0] |
| no_zero | 92 / 229 | +0.007 / +0.000 | +0.007 [-0.029, +0.025] | +0.28 / -0.10 | +0.38 [-0.41, +0.84] | -0.8 / +0.4 | -1.2 [-2.7, +0.6] |
| over_turn | 8 / 229 | +0.014 / +0.000 | +0.013 [-0.057, +0.051] | +0.09 / -0.10 | +0.19 [-0.82, +0.43] | +2.1 / +0.4 | +1.7 [+1.0, +3.3] |
| all failed | 131 / 229 | +0.017 / +0.000 | +0.017 [-0.012, +0.043] | +0.46 / -0.10 | +0.55 [+0.10, +0.85] | -0.8 / +0.4 | -1.3 [-2.8, +0.1] |

(b) The 1-2 s shortfall x12 split through plan k+2 (served 1.0 s after plan k; 3702 decisions): replan = plan k+2's own 0-1 s heading - plan k's 1-2 s segment (- = the newer plan, drawn from where the car is, asks for less turn over the same second); track = car 1-2 s - plan k+2's 0-1 s (- = the car under-executes the plan driving it). Group medians of per-turn medians [CI], deg.

| group | turns | x12 | replan | track | plan k 1-2 s | plan k+2 0-1 s | car 1-2 s |
|:--|--:|:--|:--|:--|:--|:--|:--|
| completed | 230 | +0.2 [-0.4, +1.1] | +0.2 [-0.4, +0.6] | +0.3 [-0.1, +0.4] | +17.0 | +17.2 | +17.7 |
| failed | 156 | -1.3 [-1.9, -0.0] | -0.6 [-1.3, +0.2] | -0.4 [-0.6, -0.0] | +14.7 | +14.0 | +13.0 |
| failed without FMH | 128 | -0.5 [-1.6, +0.5] | -0.1 [-1.2, +0.3] | -0.3 [-0.5, +0.2] | +15.0 | +14.4 | +13.5 |
| failed late_wide | 35 | -2.9 [-4.2, -2.0] | -1.9 [-2.8, +0.6] | -1.0 [-1.5, -0.6] | +19.3 | +16.2 | +14.2 |
| failed straight | 21 | -12.7 [-13.7, -5.4] | -5.2 [-11.8, -3.4] | -2.6 [-4.2, +1.4] | +8.0 | -0.1 | -2.5 |
| failed over_turn | 6 | +1.3 [+1.2, +1.3] | +0.6 [+0.1, +0.7] | +1.1 [+0.9, +1.3] | +21.3 | +22.3 | +22.3 |
| failed no_zero | 92 | -0.3 [-1.4, +0.9] | +0.0 [-0.9, +0.4] | -0.2 [-0.6, +0.4] | +13.8 | +13.9 | +13.2 |
| failed late_wide without FMH | 28 | -2.7 [-3.7, -0.8] | -2.0 [-2.4, +0.0] | -0.7 [-1.0, -0.1] | +19.8 | +17.0 | +15.6 |
