| arm | n | spins | on the 10 | new spins | initial-launch spins / n | re-launch spins / n | fg_collision | bg_collision | max_steps | HD mean | RC mean | non-spin HD paired vs base | non-spin HD paired vs base_rerun | non-spin HD paired vs base_rerun2 | non-spin HD paired vs base_rerun3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 64 | 10 | 10 / 10 | 0  | 6 / 64 | 2 / 27 | 26 | 13 | 15 | 0.278 | 0.349 |  |  |  |  |
| base_rerun | 64 | 9 | 8 / 10 | 1 ['scene-2800_3000-easy-00'] | 6 / 64 | 0 / 25 | 25 | 13 | 14 | 0.274 | 0.343 |  |  |  |  |
| base_rerun2 | 64 | 9 | 8 / 10 | 1 ['scene-2800_3000-easy-00'] | 6 / 64 | 0 / 20 | 25 | 12 | 15 | 0.280 | 0.350 |  |  |  |  |
| base_rerun3 | 64 | 9 | 8 / 10 | 1 ['scene-2800_3000-easy-00'] | 6 / 64 | 0 / 20 | 25 | 12 | 15 | 0.280 | 0.350 | +0.002 [-0.000, +0.005] | +0.007 [-0.001, +0.017] | +0.000 [+0.000, +0.000] |  |
| opctrl | 64 | 0 | 0 / 10 | 0  | 0 / 64 | 0 / 35 | 24 | 5 | 24 | 0.286 | 0.322 | -0.009 [-0.058, +0.042] | -0.004 [-0.051, +0.046] | -0.011 [-0.060, +0.040] | -0.011 [-0.060, +0.040] |

Line (i) spins <= 2: **0** -> PASS
Line (ii) non-spin HD paired CI lower bound > -0.02 (vs exam base): **-0.058** -> FAIL

Largest non-spin HD losses vs exam base: scene-0930-hard-00 1.000 -> 0.256 (fg_collision); scene-3000_3200-medium-00 0.737 -> 0.273 (max_steps); scene-322492347634-easy-00 0.600 -> 0.334 (max_steps); scene-0418-hard-00 0.214 -> 0.002 (max_steps); scene-124-extreme-01 0.290 -> 0.099 (fg_collision); scene-034-hard-00 0.152 -> 0.071 (fg_collision)
Largest gains: scene-3400_3600-hard-00 0.137 -> 0.798 (off_route); scene-034-hard-01 0.142 -> 0.787 (max_steps); scene-2800_3000-easy-00 0.570 -> 0.793 (complete); scene-095-extreme-01 0.083 -> 0.277 (fg_collision)
Spins: 

Closed-loop c of the opctrl arm (heading over the next step per degree of the plan's 1 s direction): 0-1 0.033 [0.024, 0.041] (n 640); 1-2 0.066 [0.053, 0.079] (n 780); 2-3 0.146 [0.119, 0.176] (n 426); 0-3 0.064 [0.048, 0.078] (n 1846)

Per baseline-spin scenario:

| scenario | base | base_rerun | base_rerun2 | base_rerun3 | opctrl |
|---|---|---|---|---|---|
| scene-0013-medium-00 | SPIN 172 deg, HD 0.055, bg_collision | SPIN 166 deg, HD 0.056, bg_collision | SPIN 172 deg, HD 0.055, bg_collision | SPIN 172 deg, HD 0.055, bg_collision | 2 deg, HD 1.000, complete |
| scene-0528-medium-00 | SPIN 114 deg, HD 0.045, bg_collision | SPIN 115 deg, HD 0.045, bg_collision | SPIN 113 deg, HD 0.045, bg_collision | SPIN 113 deg, HD 0.045, bg_collision | 3 deg, HD 0.153, max_steps |
| scene-0254-extreme-00 | SPIN 178 deg, HD 0.821, complete | SPIN 178 deg, HD 0.826, complete | SPIN 178 deg, HD 0.818, complete | SPIN 178 deg, HD 0.818, complete | 2 deg, HD 0.030, fg_collision |
| scene-102751446607-medium-01 | SPIN 180 deg, HD 0.150, bg_collision | SPIN 179 deg, HD 0.150, bg_collision | SPIN 177 deg, HD 0.141, bg_collision | SPIN 177 deg, HD 0.141, bg_collision | 1 deg, HD 0.150, max_steps |
| scene-152217047339-medium-00 | SPIN 150 deg, HD 0.083, bg_collision | SPIN 161 deg, HD 0.083, bg_collision | SPIN 151 deg, HD 0.083, bg_collision | SPIN 151 deg, HD 0.083, bg_collision | 2 deg, HD 0.125, max_steps |
| scene-570_770-easy-00 | SPIN 63 deg, HD 0.225, bg_collision | 18 deg, HD 0.224, bg_collision | 15 deg, HD 0.225, bg_collision | 15 deg, HD 0.225, bg_collision | 43 deg, HD 0.224, bg_collision |
| scene-5980_6180-easy-00 | SPIN 70 deg, HD 0.325, bg_collision | 48 deg, HD 0.305, bg_collision | 49 deg, HD 0.304, bg_collision | 49 deg, HD 0.304, bg_collision | 53 deg, HD 0.341, off_route |
| scene-8440_8640-easy-00 | SPIN 150 deg, HD 0.557, bg_collision | SPIN 72 deg, HD 0.568, bg_collision | SPIN 179 deg, HD 0.472, bg_collision | SPIN 179 deg, HD 0.472, bg_collision | 53 deg, HD 0.778, max_steps |
| scene-040-easy-00 | SPIN 130 deg, HD 0.806, bg_collision | SPIN 98 deg, HD 0.819, bg_collision | SPIN 89 deg, HD 0.945, off_route | SPIN 89 deg, HD 0.945, off_route | 6 deg, HD 1.000, complete |
| scene-053-medium-02 | SPIN 179 deg, HD 0.432, max_steps | SPIN 174 deg, HD 0.438, max_steps | SPIN 179 deg, HD 0.431, max_steps | SPIN 179 deg, HD 0.431, max_steps | 3 deg, HD 0.703, max_steps |
