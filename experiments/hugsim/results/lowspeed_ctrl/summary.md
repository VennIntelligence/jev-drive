| arm | n | spins (all) | spins on the 10 | new spins | initial-launch spins / n | re-launch spins / n | stood to max_steps | HD mean | HD on non-spin (paired delta, 95% CI) | n non-spin |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 64 | 10 | 10 / 10 | 0  | 6 / 64 | 2 / 27 | 15 | 0.278 |  | 54 |
| base_rerun | 64 | 9 | 8 / 10 | 1 ['scene-2800_3000-easy-00'] | 6 / 64 | 0 / 25 | 14 | 0.274 | -0.005 [-0.015, +0.003] | 54 |
| lowspeed | 64 | 1 | 0 / 10 | 1 ['scene-2800_3000-easy-00'] | 0 / 64 | 0 / 32 | 21 | 0.268 | -0.028 [-0.058, -0.006] | 54 |

Line (i) spins <= 2 (base 10): **1** -> PASS
Line (ii) non-spin HD paired CI lower bound > -0.02: **-0.058** -> FAIL (crashed scenarios none; reading 2, crash = 0: -0.028 [-0.058, -0.006])

Per baseline-spin scenario (max heading error vs route, HD, end, steps):

| scenario | base | base_rerun | lowspeed |
|---|---|---|---|
| scene-0013-medium-00 | SPIN 172 deg, HD 0.055, bg_collision 22 st | SPIN 166 deg, HD 0.056, bg_collision 22 st | 3 deg, HD 1.000, complete 50 st |
| scene-0528-medium-00 | SPIN 114 deg, HD 0.045, bg_collision 17 st | SPIN 115 deg, HD 0.045, bg_collision 17 st | 2 deg, HD 0.143, max_steps 400 st |
| scene-0254-extreme-00 | SPIN 178 deg, HD 0.821, complete 76 st | SPIN 178 deg, HD 0.826, complete 82 st | 0 deg, HD 0.022, fg_collision 40 st |
| scene-102751446607-medium-01 | SPIN 180 deg, HD 0.150, bg_collision 66 st | SPIN 179 deg, HD 0.150, bg_collision 66 st | 1 deg, HD 0.150, max_steps 400 st |
| scene-152217047339-medium-00 | SPIN 150 deg, HD 0.083, bg_collision 27 st | SPIN 161 deg, HD 0.083, bg_collision 29 st | 2 deg, HD 0.124, max_steps 400 st |
| scene-570_770-easy-00 | SPIN 63 deg, HD 0.225, bg_collision 45 st | 18 deg, HD 0.224, bg_collision 38 st | 24 deg, HD 0.234, bg_collision 41 st |
| scene-5980_6180-easy-00 | SPIN 70 deg, HD 0.325, bg_collision 118 st | 48 deg, HD 0.305, bg_collision 95 st | 51 deg, HD 0.334, off_route 118 st |
| scene-8440_8640-easy-00 | SPIN 150 deg, HD 0.557, bg_collision 256 st | SPIN 72 deg, HD 0.568, bg_collision 273 st | 44 deg, HD 0.795, bg_collision 358 st |
| scene-040-easy-00 | SPIN 130 deg, HD 0.806, bg_collision 175 st | SPIN 98 deg, HD 0.819, bg_collision 169 st | 5 deg, HD 1.000, complete 269 st |
| scene-053-medium-02 | SPIN 179 deg, HD 0.432, max_steps 400 st | SPIN 174 deg, HD 0.438, max_steps 400 st | 2 deg, HD 0.583, max_steps 400 st |
