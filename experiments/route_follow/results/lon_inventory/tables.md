## Policy store (decisions / scenes / logs / with frames)

### red

| subset | decisions / scenes / logs / frames | replay k <= 1 |
|:--|--:|--:|
| all | 120 / 53 / 44 / 120 | 27 |
| plan not <= 1 m/s before the line | 119 / 53 / 44 / 119 | 27 |
| plan crosses within 4 s | 29 / 13 / 11 / 29 | 5 |
| plan cannot stop (<= 3 m/s^2) | 32 / 13 / 11 / 32 | 5 |
| human stopped at the line | 62 / 28 / 24 / 62 | 15 |
| not <= 1 m/s AND human stopped | 62 / 28 / 24 / 62 | 15 |
| cannot stop AND human stopped | 0 / 0 / 0 / 0 | 0 |
| plan end speed >= 0.7 x speed (no intent) | 24 / 14 / 14 / 24 | 8 |
| no intent AND human stopped | 7 / 5 / 5 / 7 | 2 |
| plan decel < 0.5 x needed to stop at the line | 24 / 10 / 10 / 24 | 6 |
| under-braking AND human stopped | 2 / 1 / 1 / 2 | 0 |
| red through the 4 s window | 101 / 44 / 37 / 101 | 19 |

| line distance | all |
|:--|--:|
| 0-10 m | 5 / 3 / 3 / 5 |
| 10-20 m | 19 / 11 / 10 / 19 |
| 20-40 m | 96 / 44 / 38 / 96 |

### stop

| subset | decisions / scenes / logs / frames | replay k <= 1 |
|:--|--:|--:|
| all | 73 / 24 / 14 / 73 | 14 |
| plan not <= 1 m/s before the line | 73 / 24 / 14 / 73 | 14 |
| plan crosses within 4 s | 19 / 9 / 7 / 19 | 6 |
| plan cannot stop (<= 3 m/s^2) | 33 / 12 / 7 / 33 | 10 |
| human stopped at the line | 7 / 2 / 2 / 7 | 0 |
| not <= 1 m/s AND human stopped | 7 / 2 / 2 / 7 | 0 |
| cannot stop AND human stopped | 0 / 0 / 0 / 0 | 0 |
| plan end speed >= 0.7 x speed (no intent) | 20 / 9 / 6 / 20 | 2 |
| no intent AND human stopped | 1 / 1 / 1 / 1 | 0 |
| plan decel < 0.5 x needed to stop at the line | 17 / 7 / 4 / 17 | 1 |
| under-braking AND human stopped | 0 / 0 / 0 / 0 | 0 |
| red through the 4 s window | 0 / 0 / 0 / 0 | 0 |

| line distance | all |
|:--|--:|
| 0-10 m | 3 / 3 / 2 / 3 |
| 10-20 m | 15 / 9 / 7 / 15 |
| 20-40 m | 55 / 19 / 13 / 55 |

### qual

| subset | decisions / scenes / logs / frames | replay k <= 1 |
|:--|--:|--:|
| all | 77 / 36 / 31 / 77 | 19 |
| plan not <= 1 m/s before the line | 76 / 36 / 31 / 76 | 19 |
| plan crosses within 4 s | 1 / 1 / 1 / 1 | 0 |
| plan cannot stop (<= 3 m/s^2) | 1 / 1 / 1 / 1 | 0 |
| human stopped at the line | 64 / 28 / 24 / 64 | 15 |
| not <= 1 m/s AND human stopped | 64 / 28 / 24 / 64 | 15 |
| cannot stop AND human stopped | 0 / 0 / 0 / 0 | 0 |
| plan end speed >= 0.7 x speed (no intent) | 7 / 6 / 6 / 7 | 3 |
| no intent AND human stopped | 6 / 5 / 5 / 6 | 2 |
| plan decel < 0.5 x needed to stop at the line | 0 / 0 / 0 / 0 | 0 |
| under-braking AND human stopped | 0 / 0 / 0 / 0 | 0 |
| red through the 4 s window | 65 / 30 / 26 / 65 | 16 |

| line distance | all |
|:--|--:|
| 0-10 m | 0 / 0 / 0 / 0 |
| 10-20 m | 1 / 1 / 1 / 1 |
| 20-40 m | 76 / 35 / 30 / 76 |

### free

| subset | decisions / scenes / logs / frames | replay k <= 1 |
|:--|--:|--:|
| all | 5349 / 522 / 87 / 5349 | 577 |
| slow_h | 2006 / 319 / 81 / 2006 | 0 |
| slow_x | 79 / 21 / 15 / 79 | 0 |
| has_pack | 232 / 36 / 26 / 232 | 0 |
| slow_p | 1924 / 326 / 78 / 1924 | 14 |

| human speed | all | slow_h (< 0.85 x human) | slow_p (< 0.85 x proxy) | with pack | slow_x (< 0.85 x expert) |
|:--|--:|--:|--:|--:|--:|
| < 8 | 2259 / 321 / 77 / 2259 | 802 / 163 / 58 / 802 | 749 / 164 / 57 / 749 | 172 / 33 / 23 / 172 | 37 / 13 / 10 / 37 |
| 8-12 | 2693 / 294 / 74 / 2693 | 1055 / 170 / 63 / 1055 | 1026 / 167 / 61 / 1026 | 60 / 14 / 12 / 60 | 42 / 11 / 9 / 42 |
| 12-16 | 397 / 42 / 19 / 397 | 149 / 22 / 14 / 149 | 149 / 22 / 14 / 149 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| >= 16 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |

| expert speed | rows with pack | slow_x |
|:--|--:|--:|
| < 8 | 208 / 36 / 26 / 208 | 61 / 19 / 14 / 61 |
| 8-12 | 24 / 8 / 7 / 24 | 18 / 8 / 7 / 18 |
| 12-16 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| >= 16 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |

### turn

| subset | decisions / scenes / logs / frames | replay k <= 1 |
|:--|--:|--:|
| all | 2234 / 260 / 68 / 2234 | 383 |
| fast_x | 23 / 19 / 15 / 23 | 22 |
| has_pack | 277 / 57 / 34 / 277 | 114 |
| fast_p | 114 / 40 / 24 / 114 | 35 |
| fast_h | 60 / 25 / 20 / 60 | 0 |
| fast_p_and_h | 29 / 12 / 9 / 29 | 0 |

## Human navtrain (rows / logs)

tokens 103288, logs 1192; speed q50 / q90 / q99 / max [4.4315361976623535, 9.266329956054689, 12.783636960983277, 17.114574432373047]

- red: all {'rows': 1711, 'logs': 335}, human stopped {'rows': 894, 'logs': 171}, decelerating {'rows': 1191, 'logs': 290}
- stop: all {'rows': 819, 'logs': 171}, human stopped {'rows': 124, 'logs': 26}, decelerating {'rows': 257, 'logs': 93}
- qual: all {'rows': 1418, 'logs': 269}, human stopped {'rows': 960, 'logs': 180}, decelerating {'rows': 977, 'logs': 231}
- free: {'rows': 20889, 'logs': 1010}; by own speed {'< 8': {'rows': 9949, 'logs': 889}, '8-12': {'rows': 9197, 'logs': 719}, '12-16': {'rows': 1735, 'logs': 240}, '>= 16': {'rows': 8, 'logs': 1}}
- turn approach: {'rows': 20492, 'logs': 785}; faster than proxy + 1 {'rows': 873, 'logs': 268}
