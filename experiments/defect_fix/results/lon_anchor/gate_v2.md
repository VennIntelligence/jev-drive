# LON-ANCHOR offline gates, supplement 1 (G1 PASS)

G1 (amended): the human's path started at the ego as the plan, the trainer's term (projected progress): 137 of 134033 labelled rows active (max |d| > 0.5 m) = 0.102 % (line <= 1 %); term 10 / 50 / 90 / 99 % [0.0, 0.0, 0.001, 0.045]; nuPlan 0.037 %, PAI 0.191 %.

Label outcomes (S rows): {"nuplan": {"ok": 77479, "lead": 6091, "clock_go": 1885, "clock_stop": 1486, "no_reach": 47, "log_end": 24}, "pai": {"ok": 56554, "lead": 10564, "log_end": 8249, "clock_go": 4620, "no_reach": 2771, "clock_stop": 1120, "lead_unknown": 164}}

Clock exclusion by pool and kind / class (S rows with a target):

| rows | S rows | clock_go | clock_stop |
|:--|--:|--:|--:|
| nuplan/launch | 4179 | 254 | 1252 |
| nuplan/stop | 3658 | 1067 | 234 |
| nuplan/turn | 11504 | 79 | 0 |
| nuplan/move | 67600 | 485 | 0 |
| pai/launch | 14337 | 1378 | 127 |
| pai/stop | 5448 | 1046 | 743 |
| pai/turn | 9812 | 267 | 0 |
| pai/move | 48826 | 1135 | 0 |
| pai/bend | 8732 | 3 | 0 |
| pai/stop_nolead | 29914 | 1911 | 834 |
| pai/turn_sharp | 31059 | 1328 | 159 |

Kinds: launch = ego t0 speed < 0.5 m/s; stop = the target's last 0.5 s step < 0.5 m/s; turn = |path heading change from s_e to s_e + s_t,8| >= 30 deg; move = the rest; none = no target.

Stored plans against the anchored target (labelled rows after both exclusions; reported only):

| rows | n | target longer (push faster) | target shorter (push slower) | within 0.5 m | active | d4 10 / 50 / 90 / 99 % |
|:--|--:|--:|--:|--:|--:|--:|
| all | 134033 | 65.1 % | 28.2 % | 6.6 % | 96.5 % | [-15.677, -2.569, 5.103, 19.975] |
| nuplan | 77479 | 73.2 % | 19.8 % | 6.9 % | 95.8 % | [-14.763, -3.396, 2.707, 9.927] |
| pai | 56554 | 54.0 % | 39.8 % | 6.2 % | 97.5 % | [-18.052, -1.129, 8.331, 27.971] |
| pai/bend | 6947 | 71.1 % | 24.4 % | 4.5 % | 97.2 % | [-30.796, -4.252, 6.556, 25.053] |
| pai/launch | 3226 | 22.4 % | 68.4 % | 9.2 % | 95.5 % | [-5.838, 1.753, 5.549, 7.816] |
| pai/stop_nolead | 20081 | 38.6 % | 54.0 % | 7.4 % | 97.1 % | [-7.955, 0.958, 11.648, 37.439] |
| pai/turn_sharp | 22993 | 61.6 % | 32.8 % | 5.6 % | 98.0 % | [-18.232, -2.319, 7.065, 19.828] |
| nuplan/launch | 2065 | 69.1 % | 24.5 % | 6.4 % | 96.6 % | [-13.484, -3.383, 2.938, 6.206] |
| nuplan/stop | 2139 | 44.7 % | 45.6 % | 9.7 % | 96.0 % | [-4.532, -0.0, 8.719, 19.435] |
| nuplan/turn | 10681 | 70.1 % | 21.7 % | 8.2 % | 95.2 % | [-12.521, -2.928, 2.894, 10.93] |
| nuplan/move | 62594 | 74.9 % | 18.5 % | 6.6 % | 95.9 % | [-15.466, -3.624, 2.433, 8.979] |
| pai/stop | 3233 | 18.8 % | 72.0 % | 9.2 % | 96.1 % | [-1.805, 2.2, 23.879, 50.72] |
| pai/turn | 8628 | 63.1 % | 31.1 % | 5.8 % | 98.4 % | [-14.613, -2.451, 6.213, 14.104] |
| pai/move | 41467 | 57.4 % | 36.8 % | 5.8 % | 97.5 % | [-21.005, -1.646, 8.474, 25.64] |

By the ego's lead gap on the human's path (AF-front-relevant; rows whose lead is closer than the human's are already excluded):

| rows | n | target longer (push faster) | target shorter | d4 10 / 50 / 90 / 99 % |
|:--|--:|--:|--:|--:|
| nuplan/gap 0-15 m | 3215 | 51.3 % | 36.4 % | [-4.969, -0.593, 3.871, 13.095] |
| nuplan/gap 15-30 m | 16479 | 75.3 % | 17.8 % | [-10.226, -3.28, 2.097, 9.475] |
| nuplan/gap 30-61 m | 13766 | 78.1 % | 15.6 % | [-17.147, -4.32, 1.784, 8.842] |
| nuplan/no lead | 44019 | 72.6 % | 20.7 % | [-16.187, -3.484, 3.048, 10.142] |
| pai/gap 0-15 m | 4302 | 26.6 % | 60.9 % | [-2.708, 1.042, 3.826, 10.905] |
| pai/gap 15-30 m | 8788 | 54.3 % | 36.3 % | [-6.311, -0.89, 5.203, 15.413] |
| pai/gap 30-61 m | 5727 | 62.7 % | 30.6 % | [-17.038, -2.847, 5.787, 20.712] |
| pai/no lead | 37737 | 55.8 % | 39.5 % | [-23.334, -1.612, 10.044, 31.152] |

stored plans = nuPlan rows' served plans (rows meta `poses`), cp rows' model plans (`poses_model`, NTV0-L18-s0); d4 = projected 4 s progress - anchored target (m); target longer = d4 < -0.5 m (the term pushes faster), shorter = d4 > +0.5 m; labelled rows after both exclusions; reported, not gated
