| arm | n | spins (all) | spins on the 10 | new spins | stood to max_steps | HD mean | HD on the 10 | HD on non-spin (paired delta, 95% CI) | n non-spin |
|---|---|---|---|---|---|---|---|---|---|
| base | 64 | 10 | 10 / 10 | 0  | 15 | 0.278 | 0.350 |  | 54 |
| base_rerun | 10 | 8 | 8 / 10 | 0  | 1 | 0.315 | 0.315 |  | 0 |
| derot3 | 63 | 3 | 1 / 10 | 2 ['scene-0041-medium-00', 'scene-570_770-medium-00'] | 21 | 0.276 | 0.442 | -0.011 [-0.030, +0.008] | 53 |
| replay3 | 10 | 8 | 8 / 10 | 0  | 1 | 0.363 | 0.363 |  | 0 |
| sel3 | 11 | 0 | 0 / 5 | 0  | 0 | 0.285 | 0.486 | -0.000 [-0.003, +0.003] | 6 |

Per spin scenario (max heading error vs route, HD, end, steps):

| scenario | base | base_rerun | derot3 | replay3 | sel3 |
|---|---|---|---|---|---|
| scene-0013-medium-00 | SPIN 172 deg, HD 0.055, bg_collision 22 st | SPIN 172 deg, HD 0.055, bg_collision 22 st | 3 deg, HD 1.000, complete 49 st | SPIN 172 deg, HD 0.055, bg_collision 22 st | 3 deg, HD 1.000, complete 46 st |
| scene-0528-medium-00 | SPIN 114 deg, HD 0.045, bg_collision 17 st | SPIN 113 deg, HD 0.045, bg_collision 17 st | 3 deg, HD 0.141, max_steps 400 st | SPIN 113 deg, HD 0.045, bg_collision 17 st | 13 deg, HD 0.840, complete 49 st |
| scene-0254-extreme-00 | SPIN 178 deg, HD 0.821, complete 76 st | SPIN 178 deg, HD 0.818, complete 76 st | 0 deg, HD 0.026, fg_collision 39 st | SPIN 178 deg, HD 0.818, complete 76 st | 0 deg, HD 0.024, fg_collision 42 st |
| scene-102751446607-medium-01 | SPIN 180 deg, HD 0.150, bg_collision 66 st | SPIN 177 deg, HD 0.141, bg_collision 66 st | 3 deg, HD 0.154, max_steps 400 st | SPIN 177 deg, HD 0.141, bg_collision 66 st | - |
| scene-152217047339-medium-00 | SPIN 150 deg, HD 0.083, bg_collision 27 st | SPIN 151 deg, HD 0.083, bg_collision 27 st | 2 deg, HD 0.125, max_steps 400 st | SPIN 151 deg, HD 0.083, bg_collision 27 st | - |
| scene-570_770-easy-00 | SPIN 63 deg, HD 0.225, bg_collision 45 st | 15 deg, HD 0.225, bg_collision 38 st | 45 deg, HD 0.244, bg_collision 42 st | 15 deg, HD 0.225, bg_collision 38 st | 52 deg, HD 0.260, bg_collision 43 st |
| scene-5980_6180-easy-00 | SPIN 70 deg, HD 0.325, bg_collision 118 st | 49 deg, HD 0.304, bg_collision 92 st | 57 deg, HD 0.336, off_route 120 st | 54 deg, HD 0.336, off_route 115 st | 49 deg, HD 0.304, bg_collision 92 st |
| scene-8440_8640-easy-00 | SPIN 150 deg, HD 0.557, bg_collision 256 st | SPIN 177 deg, HD 0.098, bg_collision 107 st | SPIN 60 deg, HD 0.807, max_steps 400 st | SPIN 140 deg, HD 0.555, bg_collision 235 st | - |
| scene-040-easy-00 | SPIN 130 deg, HD 0.806, bg_collision 175 st | SPIN 89 deg, HD 0.945, off_route 193 st | 4 deg, HD 0.999, max_steps 400 st | SPIN 89 deg, HD 0.945, off_route 193 st | - |
| scene-053-medium-02 | SPIN 179 deg, HD 0.432, max_steps 400 st | SPIN 179 deg, HD 0.431, max_steps 400 st | 9 deg, HD 0.586, max_steps 400 st | SPIN 179 deg, HD 0.431, max_steps 400 st | - |

Early lean (first 2 s; + right):

| scenario | base: plan direction at 1 s, steps 1-8 (deg) | derot3 | base heading at step 8 | derot3 |
|---|---|---|---|---|
| scene-0013-medium-00 | -1 -1 -2 -4 -5 -9 -16 -24 | -1 -1 -0 -1 -0 0 -1 0 | -9.7 | -1.4 |
| scene-0528-medium-00 | -1 -2 -6 -11 -17 -23 -31 -37 | -1 -2 -2 -3 -4 -1 -1 -2 | -13.6 | -2.1 |
| scene-0254-extreme-00 | -0 -1 -1 -4 -8 -16 -23 -30 | -0 -1 2 1 -1 -0 -0 -0 | -6.9 | 0.0 |
| scene-102751446607-medium-01 | -2 -2 -3 -5 -8 -15 -22 -28 | -2 -2 -4 -2 -0 -1 -1 -0 | -5.0 | -0.8 |
| scene-152217047339-medium-00 | -0 -0 -2 -4 -5 -6 -8 -9 | -0 -0 1 -1 -0 -0 -0 0 | -2.6 | -0.5 |
| scene-570_770-easy-00 | -1 -1 -1 -1 -1 -1 -1 -1 | -1 -1 -1 -1 -0 -0 -0 -0 | -0.6 | -0.4 |
| scene-5980_6180-easy-00 | 0 1 2 3 3 3 3 2 | 0 1 1 1 1 0 0 0 | 3.3 | 2.0 |
| scene-8440_8640-easy-00 | 0 1 1 1 1 1 1 1 | 0 1 0 1 0 0 0 1 | 1.7 | 1.2 |
| scene-040-easy-00 | -1 -1 -1 -1 -1 -1 -1 -1 | -1 -1 -1 -1 -1 -0 -0 -1 | -0.6 | -0.4 |
| scene-053-medium-02 | -0 -6 -15 -19 -25 -31 -31 -34 | -0 0 0 -0 -0 -0 -0 -0 | -30.9 | -0.5 |
