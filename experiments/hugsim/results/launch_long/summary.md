| arm | n | spins | on the 10 | new spins | initial-launch spins / n | re-launch spins / n | fg_collision | max_steps | HD mean | RC mean | below 3 m/s in first 10 s, median s | v at +1 / +2 / +3 s, median | non-spin HD paired vs exam base | vs rerun 1 | vs rerun 2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 64 | 10 | 10 / 10 | 0  | 6 / 64 | 2 / 27 | 26 | 15 | 0.278 | 0.349 | 6.38 | 1.31 / 2.15 / 2.65 |  |  |  |
| base_rerun | 64 | 9 | 8 / 10 | 1 ['scene-2800_3000-easy-00'] | 6 / 64 | 0 / 25 | 25 | 14 | 0.274 | 0.343 | 6.38 | 1.30 / 2.14 / 2.62 |  |  |  |
| base_rerun2 | 64 | 9 | 8 / 10 | 1 ['scene-2800_3000-easy-00'] | 6 / 64 | 0 / 20 | 25 | 15 | 0.280 | 0.350 | 6.38 | 1.31 / 2.15 / 2.64 |  |  |  |
| launch_long | 64 | 11 | 9 / 10 | 2 ['scene-034-hard-00', 'scene-2800_3000-easy-00'] | 6 / 64 | 1 / 21 | 25 | 17 | 0.272 | 0.343 | 3.88 | 1.78 / 3.29 / 3.74 | +0.011 [-0.029, +0.048] | +0.015 [-0.024, +0.054] | +0.008 [-0.031, +0.045] |

Line (i) spins <= 2 (base 10, reruns 9, 9): **11** -> FAIL
Line (ii) non-spin HD paired CI lower bound > -0.02 (vs exam base): **-0.029** -> FAIL
Front collisions: 25 (base 26, reruns 25, 25); scenes with fg_collision where a baseline arm has none: vs base: ['scene-0254-extreme-00', 'scene-0418-hard-00', 'scene-3000_3200-medium-00']; vs base_rerun: ['scene-0254-extreme-00', 'scene-0418-hard-00', 'scene-3000_3200-medium-00']; vs base_rerun2: ['scene-0254-extreme-00', 'scene-0418-hard-00', 'scene-3000_3200-medium-00']; of those, collision within 15 s with the assist on in the last 3 s: ['scene-3000_3200-medium-00'] -> safety check ok
Assist reasons over all steps of the 64 runs: {'stop': 6704, 'lead': 200, 'on': 511, 'disarmed': 1444, 'model_faster': 55, 'slowing': 1}

Largest non-spin HD losses vs exam base: scene-3000_3200-medium-00 0.737 -> 0.042 (fg_collision, ll on 5 steps); scene-124-extreme-01 0.290 -> 0.073 (fg_collision, ll on 8 steps); scene-0418-hard-00 0.214 -> 0.071 (fg_collision, ll on 8 steps); scene-095-extreme-01 0.083 -> 0.020 (fg_collision, ll on 9 steps); scene-152217047339-extreme-00 0.055 -> 0.023 (fg_collision, ll on 8 steps); scene-3400_3600-extreme-00 0.027 -> 0.000 (fg_collision, ll on 7 steps)
Largest gains: scene-034-hard-01 0.142 -> 0.780 (max_steps); scene-032-medium-00 0.120 -> 0.353 (max_steps); scene-034-hard-00 0.152 -> 0.344 (bg_collision); scene-322492347634-easy-00 0.600 -> 0.786 (max_steps)
Surviving / new spins: scene-0013-medium-00 (max err 143.8 deg, base spin True, ll on 7); scene-0254-extreme-00 (max err 179.0 deg, base spin True, ll on 6); scene-034-hard-00 (max err 105.1 deg, base spin False, ll on 11); scene-040-easy-00 (max err 87.2 deg, base spin True, ll on 26); scene-0528-medium-00 (max err 111.4 deg, base spin True, ll on 7); scene-053-medium-02 (max err 179.2 deg, base spin True, ll on 4); scene-102751446607-medium-01 (max err 175.0 deg, base spin True, ll on 6); scene-152217047339-medium-00 (max err 148.6 deg, base spin True, ll on 8); scene-2800_3000-easy-00 (max err 110.0 deg, base spin False, ll on 8); scene-5980_6180-easy-00 (max err 94.8 deg, base spin True, ll on 16); scene-8440_8640-easy-00 (max err 156.8 deg, base spin True, ll on 47)

Per baseline-spin scenario:

| scenario | base | rerun 1 | rerun 2 | launch_long |
|---|---|---|---|---|
| scene-0013-medium-00 | SPIN 172 deg, HD 0.055, bg_collision | SPIN 166 deg, HD 0.056, bg_collision | SPIN 172 deg, HD 0.055, bg_collision | SPIN 144 deg, HD 0.064, off_route |
| scene-0528-medium-00 | SPIN 114 deg, HD 0.045, bg_collision | SPIN 115 deg, HD 0.045, bg_collision | SPIN 113 deg, HD 0.045, bg_collision | SPIN 111 deg, HD 0.034, bg_collision |
| scene-0254-extreme-00 | SPIN 178 deg, HD 0.821, complete | SPIN 178 deg, HD 0.826, complete | SPIN 178 deg, HD 0.818, complete | SPIN 179 deg, HD 0.060, fg_collision |
| scene-102751446607-medium-01 | SPIN 180 deg, HD 0.150, bg_collision | SPIN 179 deg, HD 0.150, bg_collision | SPIN 177 deg, HD 0.141, bg_collision | SPIN 175 deg, HD 0.048, bg_collision |
| scene-152217047339-medium-00 | SPIN 150 deg, HD 0.083, bg_collision | SPIN 161 deg, HD 0.083, bg_collision | SPIN 151 deg, HD 0.083, bg_collision | SPIN 149 deg, HD 0.072, bg_collision |
| scene-570_770-easy-00 | SPIN 63 deg, HD 0.225, bg_collision | 18 deg, HD 0.224, bg_collision | 15 deg, HD 0.225, bg_collision | 49 deg, HD 0.240, bg_collision |
| scene-5980_6180-easy-00 | SPIN 70 deg, HD 0.325, bg_collision | 48 deg, HD 0.305, bg_collision | 49 deg, HD 0.304, bg_collision | SPIN 95 deg, HD 0.261, bg_collision |
| scene-8440_8640-easy-00 | SPIN 150 deg, HD 0.557, bg_collision | SPIN 72 deg, HD 0.568, bg_collision | SPIN 179 deg, HD 0.472, bg_collision | SPIN 157 deg, HD 0.588, off_route |
| scene-040-easy-00 | SPIN 130 deg, HD 0.806, bg_collision | SPIN 98 deg, HD 0.819, bg_collision | SPIN 89 deg, HD 0.945, off_route | SPIN 87 deg, HD 0.783, bg_collision |
| scene-053-medium-02 | SPIN 179 deg, HD 0.432, max_steps | SPIN 174 deg, HD 0.438, max_steps | SPIN 179 deg, HD 0.431, max_steps | SPIN 179 deg, HD 0.425, max_steps |
