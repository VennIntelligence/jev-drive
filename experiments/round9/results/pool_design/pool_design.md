# POOL-DESIGN offline evidence (CPU, stored files)

## nuPlan S pool on1_JT by part (live rows = drawn in NTV0; activity on each row's input with the named checkpoint's eval plan)

| part | roller | rows / live / scenes | draws per live row (32 x 9 500) | standing | log-replay k<=1 | > 13 m/s | act RB0 / LA RB0 | act NTV0 / LA NTV0 | share of pool hinge mass | LA share of own mass | onc / soft rasters |
|:--|:--|:--|--:|--:|--:|--:|:--|:--|--:|--:|:--|
| rb0 | RB0-F-s0 | 17429 / 13793 / 1020 | 22.0 | 4.5 % | 11.7 % | 1.07 % | 22.4 / 10.4 % | 26.4 / 14.9 % | 21.1 % | 37.8 % | yes |
| rtb1 | RTB1-F-s0 | 17489 / 13854 / 1020 | 21.9 | 6.7 % | 11.7 % | 0.98 % | 20.4 / 9.4 % | 24.8 / 14.2 % | 17.8 % | 43.2 % | yes |
| y | YR10m10-F-s0 | 17306 / 13717 / 1020 | 22.2 | 4.1 % | 11.8 % | 1.02 % | 24.1 / 11.9 % | 22.9 / 11.1 % | 20.7 % | 31.0 % | no |
| ys | YS-L18-s0, YS-L18-s1 | 34788 / 27575 / 1020 | 11.0 | 4.0 % | 11.7 % | 0.94 % | 20.9 / 11.6 % | 20.2 / 11.3 % | 40.4 % | 32.0 % | no |
| all | RB0-F-s0, RTB1-F-s0, YR10m10-F-s0, YS-L18-s0, YS-L18-s1 | 87012 / 68939 / 1020 | 4.4 | 4.6 % | 11.7 % | 0.99 % | 21.8 / 11.0 % | 22.9 / 12.6 % | 100.0 % | 35.0 % | yes |

Speed bands of live rows (%):

| part | <0.5 | 0.5-3 | 3-8 | 8-13 | 13-20 | >20 |
|:--|--:|--:|--:|--:|--:|--:|
| rb0 | 4.5 | 23.6 | 53.3 | 17.5 | 1.1 | 0.0 |
| rtb1 | 6.7 | 20.0 | 52.3 | 20.1 | 1.0 | 0.0 |
| y | 4.1 | 24.1 | 54.1 | 16.7 | 1.0 | 0.0 |
| ys | 4.0 | 25.4 | 53.0 | 16.6 | 0.9 | 0.0 |
| all | 4.6 | 23.7 | 53.2 | 17.5 | 1.0 | 0.0 |

rb0 filters (live rows): `{"standing": {"rows": 614, "share": 0.0445, "mass_share": 0.0305}, "off_t0": {"rows": 270, "share": 0.0196, "mass_share": 0.355}, "logreplay": {"rows": 1616, "share": 0.1172, "mass_share": 0.0179}, "unlearnable_ntv0": {"rows": 1584, "share": 0.1148, "mass_share": 0.6219}}`
rb0 kept after standing + off_t0: `{"rows": 12922, "draws_per_row_k32": 23.5, "la_ntv0": 0.1545, "act_ntv0": 0.2524}`
RB0-F own states vs other parts: `{"act_r_rb0": 0.2242, "act_r_rest": 0.216, "la_r_rb0": 0.1038, "la_r_rest": 0.1112, "act_n_rb0": 0.264, "act_n_rest": 0.2206, "la_n_rb0": 0.1492, "la_n_rest": 0.1196}`
RB0-F rollouts on the train scenes: `{"scenes": 1031, "mean": 0.7877, "zero": 146, "full": 552, "atfault": 29, "offroad": 60, "corridor": 50}`
rb0 stored route: `{"ok": 0.9999, "length_m_p5_p50_p95": [82.0, 98.0, 100.0], "x_far_m_p50": 83.8, "points": 51}`

## PAI states: speed bands (%)

| source | rows / steps | <0.5 | 0.5-3 | 3-8 | 8-13 | 13-20 | >20 | > 13 | > 20 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| PAID: R5LC own states, ACC + GAP, 719 train scenes, 10 Hz | 110981 | 10.5 | 10.0 | 27.6 | 25.1 | 14.0 | 12.8 | 26.8 | 12.8 |
| PAIH: human states, train scenes | 135793 | 13.7 | 11.1 | 26.9 | 22.9 | 13.7 | 11.7 | 25.4 | 11.7 |
| 281 exam, RB0-F-s0 read (0.5 s steps) | 8305 | 7.7 | 9.7 | 22.6 | 27.6 | 16.4 | 15.9 | 32.3 | 15.9 |
| 281 exam, NTV0-L18-s0 read | 8667 | 7.5 | 9.9 | 23.8 | 28.8 | 14.5 | 15.3 | 29.9 | 15.3 |
| nuPlan rb0 (live) | 13793 | 4.5 | 23.6 | 53.3 | 17.5 | 1.1 | 0.0 | 1.1 | 0.0 |
| nuPlan on1_JT (live) | 68939 | 4.6 | 23.7 | 53.2 | 17.5 | 1.0 | 0.0 | 1.0 | 0.0 |

PAID near-duplicates: `{"dk_mode": 1, "dv_lag1_median": 0.076, "dv_lag5_median": 0.377, "rows_2hz": 22440}`
281 exam human stations: `{"stations": 59694, "speed": {"<0.5": 0.0011, "0.5-3": 0.0155, "3-8": 0.1206, "8-13": 0.2552, "13-20": 0.1938, ">20": 0.4138}, "cls": {"urban": 0.4859, "arterial": 0.1578, "highway": 0.3501, "ramp": 0.0062}, "fallback": 0.3185}`

## Road class (behavior_stats Q1 rules; distance-weighted 1 m stations of the human path)

| set | scenes | km | urban | arterial | highway | ramp | fallback (human-speed class) | scene majority |
|:--|--:|--:|--:|--:|--:|--:|--:|:--|
| train | 719 | 137.7 | 60.6 | 14.0 | 24.7 | 0.8 | 34.0 | {'urban': 543, 'highway': 87, 'arterial': 85, 'ramp': 4} |
| test281 | 281 | 59.7 | 48.6 | 15.8 | 35.0 | 0.6 | 31.9 | {'urban': 197, 'highway': 49, 'arterial': 34, 'ramp': 1} |

median human speed by class: train `{'urban': 10.91, 'arterial': 16.61, 'highway': 27.75, 'ramp': 17.88}`, 281 `{'urban': 10.6, 'arterial': 18.23, 'highway': 27.06, 'ramp': 21.45}`
train fallback by class `{'urban': 0.2837, 'arterial': 0.5047, 'highway': 0.3959, 'ramp': 0.0}`; errors 0, lon missing 0
