| arm | n | spins | spins on the 10 | new spins | initial-launch spins / n | re-launch spins / n | max_steps | HD mean | non-spin HD paired vs exam base (95% CI) | vs same-day rerun | n non-spin |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 64 | 10 | 10 / 10 | 0  | 6 / 64 | 2 / 27 | 15 | 0.278 |  |  | 54 |
| base_rerun | 64 | 9 | 8 / 10 | 1 ['scene-2800_3000-easy-00'] | 6 / 64 | 0 / 25 | 14 | 0.274 | -0.005 [-0.015, +0.003] |  | 54 |
| lowspeed | 64 | 1 | 0 / 10 | 1 ['scene-2800_3000-easy-00'] | 0 / 64 | 0 / 32 | 21 | 0.268 | -0.028 [-0.058, -0.006] | -0.023 [-0.052, -0.004] | 54 |
| base_rerun2 | 64 | 9 | 8 / 10 | 1 ['scene-2800_3000-easy-00'] | 6 / 64 | 0 / 20 | 15 | 0.280 | +0.002 [-0.000, +0.005] | +0.007 [-0.001, +0.017] | 54 |
| lowsel | 64 | 6 | 5 / 10 | 1 ['scene-0041-medium-00'] | 2 / 64 | 1 / 18 | 18 | 0.286 | +0.005 [-0.021, +0.038] | +0.009 [-0.013, +0.042] | 54 |

Line (i) spins <= 2 (base 10): **6** -> FAIL
Line (ii) non-spin HD paired CI lower bound > -0.02 (vs exam base): **-0.021** -> FAIL (crashed none; crash = 0: +0.005 [-0.021, +0.038])
Near gate (spins <= 4 and lower bound > -0.04): no

Largest non-spin HD losses vs exam base: scene-124-extreme-01 0.290 -> 0.053 (fg_collision); scene-0418-hard-00 0.214 -> 0.002 (max_steps); scene-322492347634-easy-00 0.600 -> 0.419 (max_steps); scene-034-easy-00 0.955 -> 0.865 (max_steps); scene-150623512729-medium-01 0.153 -> 0.089 (max_steps); scene-0138-extreme-00 0.106 -> 0.049 (fg_collision)

Surviving spins (raw plan 1 s direction |a| deg, 12 steps up to the divergence start): {"scene-0041-medium-00": {"start": 20, "raw_plan_dir_deg": [1.1, 1.3, 1.4, 1.5, 1.3, 0.5, 0.7, 1.7, 3.6, 6.8, 11.2, 17.1, 23.3]}, "scene-0528-medium-00": {"start": 8, "raw_plan_dir_deg": [0.0, 0.9, 1.5, 2.0, 3.3, 6.8, 12.5, 18.5, 26.1]}, "scene-053-medium-02": {"start": 5, "raw_plan_dir_deg": [0.0, 0.2, 5.9, 14.7, 19.1, 24.5]}, "scene-152217047339-medium-00": {"start": 14, "raw_plan_dir_deg": [0.4, 1.2, 2.3, 2.6, 2.7, 3.0, 3.8, 5.1, 6.4, 7.6, 8.5, 9.7, 10.5]}, "scene-5980_6180-easy-00": {"start": 83, "raw_plan_dir_deg": [0.0, 0.0, 0.0, 0.9, 0.9, 0.1, 2.0, 5.2, 8.8, 13.5, 16.3, 24.9, 31.8]}, "scene-8440_8640-easy-00": {"start": 49, "raw_plan_dir_deg": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.4, 3.5, 4.0, 6.7, 9.5]}}

Per baseline-spin scenario (max heading error vs route, HD, end, steps):

| scenario | base | base_rerun | lowspeed | base_rerun2 | lowsel |
|---|---|---|---|---|---|
| scene-0013-medium-00 | SPIN 172 deg, HD 0.055, bg_collision 22 st | SPIN 166 deg, HD 0.056, bg_collision 22 st | 3 deg, HD 1.000, complete 50 st | SPIN 172 deg, HD 0.055, bg_collision 22 st | 3 deg, HD 1.000, complete 46 st |
| scene-0528-medium-00 | SPIN 114 deg, HD 0.045, bg_collision 17 st | SPIN 115 deg, HD 0.045, bg_collision 17 st | 2 deg, HD 0.143, max_steps 400 st | SPIN 113 deg, HD 0.045, bg_collision 17 st | SPIN 120 deg, HD 0.052, bg_collision 19 st |
| scene-0254-extreme-00 | SPIN 178 deg, HD 0.821, complete 76 st | SPIN 178 deg, HD 0.826, complete 82 st | 0 deg, HD 0.022, fg_collision 40 st | SPIN 178 deg, HD 0.818, complete 76 st | 0 deg, HD 0.024, fg_collision 41 st |
| scene-102751446607-medium-01 | SPIN 180 deg, HD 0.150, bg_collision 66 st | SPIN 179 deg, HD 0.150, bg_collision 66 st | 1 deg, HD 0.150, max_steps 400 st | SPIN 177 deg, HD 0.141, bg_collision 66 st | 1 deg, HD 0.150, max_steps 400 st |
| scene-152217047339-medium-00 | SPIN 150 deg, HD 0.083, bg_collision 27 st | SPIN 161 deg, HD 0.083, bg_collision 29 st | 2 deg, HD 0.124, max_steps 400 st | SPIN 151 deg, HD 0.083, bg_collision 27 st | SPIN 167 deg, HD 0.112, bg_collision 37 st |
| scene-570_770-easy-00 | SPIN 63 deg, HD 0.225, bg_collision 45 st | 18 deg, HD 0.224, bg_collision 38 st | 24 deg, HD 0.234, bg_collision 41 st | 15 deg, HD 0.225, bg_collision 38 st | 16 deg, HD 0.219, bg_collision 39 st |
| scene-5980_6180-easy-00 | SPIN 70 deg, HD 0.325, bg_collision 118 st | 48 deg, HD 0.305, bg_collision 95 st | 51 deg, HD 0.334, off_route 118 st | 49 deg, HD 0.304, bg_collision 92 st | SPIN 78 deg, HD 0.278, bg_collision 91 st |
| scene-8440_8640-easy-00 | SPIN 150 deg, HD 0.557, bg_collision 256 st | SPIN 72 deg, HD 0.568, bg_collision 273 st | 44 deg, HD 0.795, bg_collision 358 st | SPIN 179 deg, HD 0.472, bg_collision 271 st | SPIN 177 deg, HD 0.529, bg_collision 333 st |
| scene-040-easy-00 | SPIN 130 deg, HD 0.806, bg_collision 175 st | SPIN 98 deg, HD 0.819, bg_collision 169 st | 5 deg, HD 1.000, complete 269 st | SPIN 89 deg, HD 0.945, off_route 193 st | 5 deg, HD 1.000, complete 165 st |
| scene-053-medium-02 | SPIN 179 deg, HD 0.432, max_steps 400 st | SPIN 174 deg, HD 0.438, max_steps 400 st | 2 deg, HD 0.583, max_steps 400 st | SPIN 179 deg, HD 0.431, max_steps 400 st | SPIN 180 deg, HD 0.440, max_steps 400 st |
