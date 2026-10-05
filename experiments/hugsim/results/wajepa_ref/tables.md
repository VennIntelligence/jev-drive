## T1 metrics on the 64 scenarios (mean over scenarios; PR #57 controller; one run each)

| agent | HD-Score | NC | DAC | TTC | Comfort | RC | PDMS | complete | spin | stuck (max_steps) | bg coll | fg coll | off_route |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| wajepa | 0.451 | 0.700 | 0.953 | 0.647 | 0.833 | 0.565 | 0.616 | 28 | 4 | 0 | 1 | 26 | 5 |
| cinque | 0.278 | 0.791 | 0.914 | 0.741 | 0.930 | 0.349 | 0.701 | 9 | 10 | 14 | 5 | 26 | 0 |
| ltf | 0.279 | 0.495 | 0.941 | 0.426 | 0.949 | 0.433 | 0.409 | 14 | n/a | 0 | 14 | 34 | 2 |
| cv | 0.292 | 0.734 | 0.935 | 0.698 | 1.000 | 0.340 | 0.658 | 3 | n/a | 9 | 15 | 33 | 4 |

LTF / cv rows: spin column not recomputed here (controller_spin.md T1: 0 / 64 for both under PR #57); their end classes are the runner's.

## T2 paired difference in HD-Score (WA-JEPA minus other), bootstrap 95% CI (jevdrive.stats.paired, B = 10000)

| versus | mean diff scenarios (unit = scenario) | mean diff scene clusters (unit = scene, ratio of sums) | WA better / worse / tie (|d| < 0.02) |
|---|---|---|---|
| cinque | 0.173 [0.081, 0.268] | 0.173 [0.071, 0.280] (45 scenes) | 29 / 22 / 13 |
| ltf | 0.172 [0.090, 0.258] | 0.172 [0.091, 0.257] (45 scenes) | 34 / 10 / 20 |
| cv | 0.158 [0.072, 0.247] | 0.158 [0.067, 0.260] (45 scenes) | 24 / 20 / 20 |

## T3 HD-Score by difficulty and by dataset (n per cell = 16)

| difficulty | wajepa | cinque | ltf | cv | WA - Cinque [95% CI] |
|---|--:|--:|--:|--:|---|
| easy | 0.719 | 0.584 | 0.678 | 0.652 | 0.135 [-0.069, 0.348] |
| medium | 0.641 | 0.191 | 0.199 | 0.378 | 0.450 [0.293, 0.613] |
| hard | 0.360 | 0.229 | 0.165 | 0.101 | 0.132 [-0.022, 0.317] |
| extreme | 0.081 | 0.106 | 0.073 | 0.039 | -0.025 [-0.156, 0.098] |

| dataset | wajepa | cinque | ltf | cv | WA - Cinque [95% CI] |
|---|--:|--:|--:|--:|---|
| nuscenes | 0.422 | 0.458 | 0.380 | 0.322 | -0.036 [-0.189, 0.117] |
| kitti360 | 0.330 | 0.212 | 0.103 | 0.081 | 0.118 [-0.006, 0.264] |
| waymo | 0.548 | 0.157 | 0.298 | 0.318 | 0.392 [0.203, 0.577] |
| pandaset | 0.503 | 0.283 | 0.333 | 0.448 | 0.219 [0.027, 0.422] |

## T4 by scenario content (interaction = the scenario has actors; ahead = an actor in front of the ego route)

| stratum | n | WA-JEPA | Cinque | LTF | cv | WA - Cinque [95% CI] |
|---|--:|--:|--:|--:|--:|---|
| no actors (n_actors = 0) | 16 | 0.719 | 0.584 | 0.678 | 0.652 | 0.135 [-0.069, 0.348] |
| actors, none ahead | 18 | 0.335 | 0.215 | 0.219 | 0.120 | 0.121 [0.004, 0.263] |
| actor ahead (n_ahead >= 1) | 30 | 0.376 | 0.152 | 0.102 | 0.204 | 0.225 [0.079, 0.369] |
| medium + actors | 16 | 0.641 | 0.191 | 0.199 | 0.378 | 0.450 [0.293, 0.613] |
| medium + actor ahead | 14 | 0.699 | 0.204 | 0.197 | 0.392 | 0.495 [0.325, 0.660] |
| turning route (turn != straight) | 22 | 0.424 | 0.189 | 0.137 | 0.195 | 0.235 [0.107, 0.379] |

Share of the total paired gap (sum of WA - Cinque over the 64) by difficulty: easy 0.20, medium 0.65, hard 0.19, extreme -0.04 (sum 11.08).

## T5 per scenario (sorted by WA - Cinque)

| scenario | diff | tier | actors / ahead | WA HD | WA class | WA steps | Cinque HD | Cinque class | Cinque steps | LTF HD | cv HD |
|---|---|---|---|--:|---|--:|--:|---|--:|--:|--:|
| scene-0254-extreme-00 | -0.759 | extreme | 2 / 1 | 0.062 | fg_coll | 38 | 0.821 | spin | 76 | 0.012 | 0.055 |
| scene-0051-easy-00 | -0.566 | easy | 0 / 0 | 0.382 | complete | 65 | 0.949 | complete | 53 | 0.954 | 0.905 |
| scene-034-easy-00 | -0.412 | easy | 0 / 0 | 0.543 | complete | 51 | 0.955 | stuck | 400 | 0.530 | 0.893 |
| scene-039-easy-00 | -0.281 | easy | 0 / 0 | 0.719 | complete | 60 | 1.000 | complete | 60 | 0.964 | 1.000 |
| scene-2800_3000-easy-00 | -0.229 | easy | 0 / 0 | 0.342 | spin | 124 | 0.570 | bg_coll | 77 | 0.608 | 0.202 |
| scene-034-hard-00 | -0.115 | hard | 1 / 1 | 0.037 | fg_coll | 22 | 0.152 | fg_coll | 25 | 0.048 | 0.091 |
| scene-881121006469-hard-00 | -0.095 | hard | 1 / 0 | 0.272 | fg_coll | 37 | 0.367 | fg_coll | 46 | 0.101 | 0.318 |
| scene-0411-hard-00 | -0.079 | hard | 1 / 0 | 0.809 | complete | 53 | 0.888 | complete | 42 | 0.380 | 0.027 |
| scene-0411-easy-00 | -0.079 | easy | 0 / 0 | 0.809 | complete | 53 | 0.888 | complete | 42 | 0.667 | 0.994 |
| scene-034-hard-01 | -0.078 | hard | 1 / 1 | 0.064 | fg_coll | 28 | 0.142 | fg_coll | 26 | 0.061 | 0.066 |
| scene-0528-extreme-00 | -0.078 | extreme | 1 / 1 | 0.042 | fg_coll | 14 | 0.120 | fg_coll | 20 | 0.026 | 0.084 |
| scene-095-extreme-01 | -0.070 | extreme | 1 / 0 | 0.013 | fg_coll | 11 | 0.083 | fg_coll | 27 | 0.030 | 0.100 |
| scene-3400_3600-hard-00 | -0.065 | hard | 1 / 0 | 0.072 | fg_coll | 44 | 0.137 | fg_coll | 43 | 0.119 | 0.142 |
| scene-152217047339-extreme-00 | -0.052 | extreme | 1 / 0 | 0.003 | fg_coll | 15 | 0.055 | fg_coll | 15 | 0.005 | 0.053 |
| scene-3000_3200-hard-00 | -0.044 | hard | 1 / 1 | 0.023 | fg_coll | 34 | 0.067 | fg_coll | 20 | 0.000 | 0.033 |
| scene-2510_2710-hard-00 | -0.036 | hard | 1 / 1 | 0.008 | spin | 79 | 0.044 | fg_coll | 86 | 0.000 | 0.000 |
| scene-0166-easy-00 | -0.032 | easy | 0 / 0 | 0.968 | complete | 63 | 1.000 | complete | 49 | 0.997 | 0.948 |
| scene-0138-extreme-00 | -0.031 | extreme | 2 / 1 | 0.076 | fg_coll | 35 | 0.106 | fg_coll | 35 | 0.047 | 0.065 |
| scene-0930-hard-00 | -0.030 | hard | 1 / 0 | 0.970 | complete | 47 | 1.000 | complete | 39 | 0.842 | 0.077 |
| scene-250_450-hard-00 | -0.025 | hard | 1 / 1 | 0.009 | fg_coll | 31 | 0.034 | fg_coll | 26 | 0.000 | 0.000 |
| scene-5980_6180-easy-00 | -0.023 | easy | 0 / 0 | 0.302 | off_route | 45 | 0.325 | spin | 118 | 0.024 | 0.130 |
| scene-3400_3600-extreme-00 | -0.022 | extreme | 1 / 1 | 0.005 | fg_coll | 21 | 0.027 | fg_coll | 26 | 0.000 | 0.000 |
| scene-150623512729-extreme-00 | -0.019 | extreme | 1 / 0 | 0.030 | fg_coll | 14 | 0.049 | fg_coll | 24 | 0.036 | 0.045 |
| scene-0167-easy-00 | -0.012 | easy | 0 / 0 | 0.960 | complete | 71 | 0.972 | complete | 71 | 0.980 | 0.727 |
| scene-0013-extreme-00 | -0.011 | extreme | 1 / 0 | 0.003 | fg_coll | 11 | 0.014 | fg_coll | 15 | 0.005 | 0.015 |
| scene-5980_6180-extreme-00 | -0.004 | extreme | 1 / 1 | 0.008 | fg_coll | 19 | 0.012 | fg_coll | 22 | 0.017 | 0.005 |
| scene-6580_6780-medium-01 | -0.003 | medium | 1 / 1 | 0.000 | bg_coll | 13 | 0.003 | bg_coll | 15 | 0.000 | 0.001 |
| scene-144248042870-extreme-00 | +0.000 | extreme | 1 / 0 | 0.000 | fg_coll | 17 | 0.000 | fg_coll | 14 | 0.000 | 0.000 |
| scene-021-extreme-02 | +0.001 | extreme | 6 / 1 | 0.017 | fg_coll | 23 | 0.016 | fg_coll | 28 | 0.007 | 0.028 |
| scene-166085257829-extreme-00 | +0.001 | extreme | 1 / 0 | 0.074 | fg_coll | 31 | 0.073 | fg_coll | 26 | 0.070 | 0.025 |
| scene-090-hard-01 | +0.002 | hard | 1 / 1 | 0.032 | fg_coll | 15 | 0.030 | fg_coll | 24 | 0.018 | 0.024 |
| scene-053-extreme-02 | +0.005 | extreme | 6 / 2 | 0.029 | fg_coll | 21 | 0.025 | fg_coll | 22 | 0.004 | 0.036 |
| scene-2510_2710-extreme-00 | +0.005 | extreme | 1 / 1 | 0.007 | fg_coll | 21 | 0.002 | fg_coll | 25 | 0.002 | 0.000 |
| scene-0418-hard-00 | +0.009 | hard | 1 / 0 | 0.223 | complete | 57 | 0.214 | complete | 90 | 0.000 | 0.194 |
| scene-770_970-extreme-00 | +0.014 | extreme | 2 / 0 | 0.024 | fg_coll | 26 | 0.010 | fg_coll | 27 | 0.000 | 0.023 |
| scene-0254-hard-00 | +0.040 | hard | 2 / 1 | 0.102 | fg_coll | 57 | 0.062 | stuck | 400 | 0.021 | 0.098 |
| scene-0528-medium-00 | +0.055 | medium | 1 / 1 | 0.100 | fg_coll | 13 | 0.045 | spin | 17 | 0.024 | 0.186 |
| scene-0411-medium-00 | +0.072 | medium | 1 / 1 | 0.145 | fg_coll | 248 | 0.073 | stuck | 400 | 0.040 | 0.223 |
| scene-040-easy-00 | +0.100 | easy | 0 / 0 | 0.906 | complete | 53 | 0.806 | spin | 175 | 0.847 | 1.000 |
| scene-0041-medium-00 | +0.109 | medium | 1 / 0 | 0.224 | off_route | 25 | 0.115 | bg_coll | 21 | 0.182 | 0.269 |
| scene-3000_3200-medium-00 | +0.131 | medium | 1 / 1 | 0.867 | complete | 69 | 0.737 | complete | 51 | 0.170 | 0.027 |
| scene-164701907483-hard-00 | +0.138 | hard | 1 / 0 | 0.344 | off_route | 53 | 0.206 | stuck | 400 | 0.041 | 0.419 |
| scene-152217047339-medium-00 | +0.168 | medium | 1 / 0 | 0.251 | off_route | 18 | 0.083 | spin | 27 | 0.250 | 0.287 |
| scene-322492347634-easy-00 | +0.187 | easy | 0 / 0 | 0.788 | complete | 109 | 0.600 | stuck | 400 | 0.930 | 0.755 |
| scene-8440_8640-easy-00 | +0.203 | easy | 0 / 0 | 0.760 | spin | 111 | 0.557 | spin | 256 | 0.193 | 0.126 |
| scene-164701907483-easy-00 | +0.249 | easy | 0 / 0 | 0.355 | off_route | 38 | 0.106 | stuck | 400 | 0.303 | 0.419 |
| scene-032-medium-00 | +0.259 | medium | 1 / 1 | 0.379 | spin | 400 | 0.120 | stuck | 400 | 0.131 | 0.433 |
| scene-053-medium-02 | +0.301 | medium | 1 / 1 | 0.734 | fg_coll | 91 | 0.432 | spin | 400 | 0.526 | 1.000 |
| scene-1290_1490-medium-01 | +0.580 | medium | 1 / 1 | 0.973 | complete | 191 | 0.393 | bg_coll | 209 | 0.136 | 0.127 |
| scene-100613054308-hard-00 | +0.613 | hard | 1 / 0 | 0.841 | complete | 106 | 0.228 | fg_coll | 34 | 0.967 | 0.039 |
| scene-124-extreme-01 | +0.616 | extreme | 1 / 0 | 0.906 | complete | 64 | 0.290 | fg_coll | 139 | 0.904 | 0.089 |
| scene-113-easy-00 | +0.661 | easy | 0 / 0 | 0.784 | complete | 92 | 0.123 | stuck | 400 | 0.813 | 0.805 |
| scene-570_770-medium-00 | +0.674 | medium | 1 / 1 | 0.927 | complete | 131 | 0.253 | bg_coll | 41 | 0.222 | 0.066 |
| scene-570_770-easy-00 | +0.726 | easy | 0 / 0 | 0.951 | complete | 129 | 0.225 | spin | 45 | 0.153 | 0.419 |
| scene-032-medium-02 | +0.726 | medium | 1 / 1 | 0.944 | complete | 69 | 0.218 | stuck | 400 | 0.369 | 0.613 |
| scene-132384196576-medium-01 | +0.791 | medium | 1 / 1 | 0.912 | complete | 168 | 0.122 | stuck | 400 | 0.030 | 0.347 |
| scene-0013-medium-00 | +0.815 | medium | 1 / 1 | 0.870 | complete | 54 | 0.055 | spin | 22 | 0.911 | 0.280 |
| scene-398895700423-easy-00 | +0.833 | easy | 0 / 0 | 0.966 | complete | 84 | 0.133 | stuck | 400 | 0.944 | 0.932 |
| scene-150623512729-medium-01 | +0.838 | medium | 1 / 1 | 0.992 | complete | 207 | 0.153 | stuck | 400 | 0.081 | 0.365 |
| scene-113792265837-easy-00 | +0.841 | easy | 0 / 0 | 0.975 | complete | 79 | 0.133 | stuck | 400 | 0.937 | 0.173 |
| scene-095-medium-01 | +0.843 | medium | 1 / 1 | 0.946 | complete | 203 | 0.103 | stuck | 400 | 0.040 | 0.949 |
| scene-102751446607-medium-01 | +0.846 | medium | 1 / 1 | 0.996 | complete | 227 | 0.150 | spin | 66 | 0.073 | 0.871 |
| scene-106762673266-hard-00 | +0.925 | hard | 1 / 0 | 0.975 | complete | 96 | 0.050 | fg_coll | 29 | 0.006 | 0.039 |
| scene-124-hard-01 | +0.947 | hard | 1 / 1 | 0.987 | complete | 65 | 0.039 | fg_coll | 30 | 0.032 | 0.042 |

## T6 where Cinque fails and WA-JEPA succeeds (Cinque HD < 0.5, WA-JEPA HD >= 0.5 and complete)

16 scenarios. Cinque failure classes among them: stuck 7, fg_coll 4, spin 3, bg_coll 2.
The reverse (WA-JEPA HD < 0.5, Cinque HD >= 0.5 and complete): 1 scenarios; WA-JEPA classes: complete 1.

Confusion of failure class, Cinque (rows) x WA-JEPA (columns), all 64:

| cc       |   bg_coll |   complete |   fg_coll |   off_route |   spin |
|:---------|----------:|-----------:|----------:|------------:|-------:|
| bg_coll  |         1 |          2 |         0 |           1 |      1 |
| complete |         0 |          9 |         0 |           0 |      0 |
| fg_coll  |         0 |          4 |        21 |           0 |      1 |
| spin     |         0 |          4 |         3 |           2 |      1 |
| stuck    |         0 |          9 |         2 |           2 |      1 |

## T7 the 10 PR #57 Cinque spinner scenarios

| scenario | Cinque HD | Cinque class | Cinque max heading err (deg) | WA HD | WA class | WA max heading err (deg) | WA v_max in first 10 s (m/s) | WA end speed | WA steps |
|---|--:|---|--:|--:|---|--:|--:|--:|--:|
| scene-0013-medium-00 | 0.055 | spin | 171 | 0.870 | complete | 3 | 8.6 | 6.5 | 54 |
| scene-0254-extreme-00 | 0.821 | spin | 178 | 0.062 | fg_coll | 13 | 2.5 | 2.4 | 38 |
| scene-040-easy-00 | 0.806 | spin | 130 | 0.906 | complete | 3 | 7.3 | 7.8 | 53 |
| scene-0528-medium-00 | 0.045 | spin | 114 | 0.100 | fg_coll | 5 | 8.5 | 8.5 | 13 |
| scene-053-medium-02 | 0.432 | spin | 179 | 0.734 | fg_coll | 1 | 5.5 | 7.0 | 91 |
| scene-102751446607-medium-01 | 0.150 | spin | 180 | 0.996 | complete | 9 | 3.4 | 3.8 | 227 |
| scene-152217047339-medium-00 | 0.083 | spin | 151 | 0.251 | off_route | 56 | 5.2 | 4.9 | 18 |
| scene-570_770-easy-00 | 0.225 | spin | 63 | 0.951 | complete | 4 | 6.4 | 7.6 | 129 |
| scene-5980_6180-easy-00 | 0.325 | spin | 70 | 0.302 | off_route | 53 | 7.4 | 6.2 | 45 |
| scene-8440_8640-easy-00 | 0.557 | spin | 150 | 0.760 | spin | 64 | 6.4 | 4.2 | 111 |

Spinners: Cinque 10 / 10 spin (definition check), WA-JEPA 1 / 10. Mean HD on the 10: WA-JEPA 0.593, Cinque 0.350. Launch stall (v_max over the first 40 steps < 1.6 m/s): WA-JEPA 0 / 10, Cinque 0 / 10; end = max_steps: WA-JEPA 0, Cinque 0.

## T8 spin and launch behaviour on all 64 (heading error >= 60 deg, definition of controller_spin.md; launch stall = v_max over the first 40 steps < 1.6 m/s)

| agent | spins >= 60 deg | >= 45 deg not recomputed | launch stalls | standing > 50% of steps | max_steps |
|---|--:|--|--:|--:|--:|
| WA-JEPA | 4 | - | 2 | 3 | 1 |
| Cinque (PR #57) | 10 | - | 9 | 18 | 15 |

WA-JEPA fallbacks (inference exceptions, braking plan): 0 steps in 0 scenarios. Padded-history steps per scenario: 6-6.

BACK-camera tile brightness (mean pixel at step 10, 0 = black), by dataset: kitti360 0.0, nuscenes 99.0, pandaset 104.4, waymo 0.1; front tile: kitti360 99.5, nuscenes 108.8, pandaset 103.7, waymo 103.8.
