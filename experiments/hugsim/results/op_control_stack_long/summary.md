| arm | n | spins | on the 10 | initial-launch spins / n | re-launch spins / n | fg | bg | off_route | max_steps | complete | HD mean | RC mean | non-spin HD paired vs base | non-spin HD paired vs base_rerun | non-spin HD paired vs base_rerun2 | non-spin HD paired vs base_rerun3 | non-spin HD paired vs opctrl | non-spin HD paired vs base_rerun4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 64 | 10 | 10 / 10 | 6 / 64 | 2 / 27 | 26 | 13 | 0 | 15 | 10 | 0.278 | 0.349 |  |  |  |  |  |  |
| base_rerun | 64 | 9 | 8 / 10 | 6 / 64 | 0 / 25 | 25 | 13 | 0 | 14 | 12 | 0.274 | 0.343 |  |  |  |  |  |  |
| base_rerun2 | 64 | 9 | 8 / 10 | 6 / 64 | 0 / 20 | 25 | 12 | 1 | 15 | 11 | 0.280 | 0.350 |  |  |  |  |  |  |
| base_rerun3 | 64 | 9 | 8 / 10 | 6 / 64 | 0 / 20 | 25 | 12 | 1 | 15 | 11 | 0.280 | 0.350 |  |  |  |  |  |  |
| opctrl | 64 | 0 | 0 / 10 | 0 / 64 | 0 / 35 | 24 | 5 | 2 | 24 | 9 | 0.286 | 0.322 |  |  |  |  |  |  |
| base_rerun4 | 64 | 9 | 8 / 10 | 6 / 64 | 0 / 20 | 25 | 12 | 1 | 15 | 11 | 0.280 | 0.350 |  |  |  |  |  |  |
| opctrl_long | 64 | 1 | 1 / 10 | 0 / 64 | 0 / 0 | 28 | 3 | 1 | 24 | 8 | 0.215 | 0.241 | -0.077 [-0.171, +0.014] | -0.072 [-0.165, +0.018] | -0.079 [-0.174, +0.012] | -0.079 [-0.174, +0.012] | -0.068 [-0.160, +0.020] | -0.079 [-0.174, +0.012] |

Line (i) spins <= 2: **1** -> PASS
Line (ii) non-spin HD paired CI lower bound > -0.02 (vs exam base): **-0.171** -> FAIL
  vs base_rerun3: -0.174 -> FAIL
  vs base_rerun4: -0.174 -> FAIL

Stuck runs (max_steps): opctrl_long 24, opctrl 24, exam base 15, same-day base 15. opctrl_long stuck and not in same-day base: 14; opctrl stuck and not in opctrl_long: 6; opctrl_long stuck and not in opctrl: 6

Largest non-spin HD losses vs exam base: scene-0167-easy-00 0.972 -> 0.044 (max_steps); scene-0930-hard-00 1.000 -> 0.072 (fg_collision); scene-039-easy-00 1.000 -> 0.079 (max_steps); scene-0051-easy-00 0.949 -> 0.076 (max_steps); scene-0411-hard-00 0.888 -> 0.109 (fg_collision); scene-3000_3200-medium-00 0.737 -> 0.033 (max_steps)
Largest gains: scene-095-medium-01 0.103 -> 1.000 (complete); scene-113-easy-00 0.123 -> 0.843 (complete); scene-132384196576-medium-01 0.122 -> 0.669 (max_steps); scene-032-medium-02 0.218 -> 0.609 (bg_collision)
Spins: scene-5980_6180-easy-00 (max err 62.6 deg, base spin True)

Closed-loop c of the arm (heading over the next step per degree of the plan's 1 s direction): 0-1 0.015 [0.007, 0.032] (n 491); 1-2 0.112 [0.047, 0.153] (n 826); 2-3 0.140 [0.086, 0.163] (n 574); 0-3 0.079 [0.029, 0.126] (n 1891)
