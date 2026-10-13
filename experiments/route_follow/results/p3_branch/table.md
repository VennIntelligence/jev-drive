### handover d10

| scene | side | g1s1 | g1.25s1 | g1.5s1 | g2s1 | g1s0.8 | g1s0.6 | g1.5s0.8 | g1.5s0.6 |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| 302c5c99 | L | corr | corr | corr | corr | corr | rear | corr | rear |
| 382a7247 | L | **done** | corr | corr | corr | offr | coll | corr | coll |
| ce161092 | L | offr | end | corr | corr | rear | rear | rear | rear |
| e37aa159 | L | corr | **done** | corr | corr | rear | rear | rear | rear |
| e5e47fd2 | L | corr | corr | corr | offr | rear | rear | rear | rear |
| fd01e9ab | L | corr | coll | corr | offr | corr | corr | corr | corr |
| 2f227f2c | R | **done** | **done** | offr | offr | **done** | **done** | **done** | **done** |
| 3ce13eed | R | corr | corr | corr | coll | corr | offr | corr | offr |
| 94c1d722 | R | offr | offr | offr | offr | offr | offr | end | offr |
| a174faed | R | corr | corr | **done** | offr | rear | rear | rear | rear |
| eaba3f6a | R | **done** | offr | offr | offr | **done** | **done** | **done** | **done** |
| f9e361aa | R | corr | offr | offr | corr | **done** | rear | coll | rear |

- d10@g1s1: completed 3 / 12, zeros 10, mean score 0.167
- d10@g1.25s1: completed 2 / 12, zeros 9, mean score 0.250
- d10@g1.5s1: completed 1 / 12, zeros 11, mean score 0.083
- d10@g2s1: completed 0 / 12, zeros 12, mean score 0.000
- d10@g1s0.8: completed 3 / 12, zeros 5, mean score 0.371
- d10@g1s0.6: completed 2 / 12, zeros 4, mean score 0.321
- d10@g1.5s0.8: completed 2 / 12, zeros 5, mean score 0.332
- d10@g1.5s0.6: completed 2 / 12, zeros 4, mean score 0.321

### handover d0

| scene | side | g1s1 | g1.25s1 | g1.5s1 | g2s1 | g1s0.8 | g1s0.6 | g1.5s0.8 | g1.5s0.6 |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| 302c5c99 | L | corr | corr | **done** | offr | corr | rear | **done** | offr |
| 382a7247 | L | **done** | corr | corr | corr | **done** | offr | corr | corr |
| ce161092 | L | offr | coll | coll | coll | rear | rear | coll | rear |
| e37aa159 | L | corr | **done** | corr | corr | end | rear | corr | rear |
| e5e47fd2 | L | offr | offr | offr | offr | offr | rear | offr | offr |
| fd01e9ab | L | **done** | corr | offr | offr | **done** | rear | offr | **done** |
| 2f227f2c | R | **done** | **done** | **done** | offr | corr | end | offr | **done** |
| 3ce13eed | R | coll | **done** | offr | offr | end | end | offr | offr |
| 94c1d722 | R | offr | **done** | offr | offr | offr | offr | offr | offr |
| a174faed | R | corr | **done** | offr | offr | corr | rear | offr | offr |
| eaba3f6a | R | **done** | offr | offr | offr | coll | **done** | offr | offr |
| f9e361aa | R | corr | offr | offr | offr | corr | coll | offr | offr |

- d0@g1s1: completed 4 / 12, zeros 8, mean score 0.273
- d0@g1.25s1: completed 5 / 12, zeros 7, mean score 0.395
- d0@g1.5s1: completed 2 / 12, zeros 10, mean score 0.167
- d0@g2s1: completed 0 / 12, zeros 12, mean score 0.000
- d0@g1s0.8: completed 2 / 12, zeros 8, mean score 0.206
- d0@g1s0.6: completed 1 / 12, zeros 3, mean score 0.379
- d0@g1.5s0.8: completed 1 / 12, zeros 11, mean score 0.083
- d0@g1.5s0.6: completed 2 / 12, zeros 8, mean score 0.182

ceiling (>= 1 completing cell over 2 handovers x 8 transforms): 10 / 12; identity only: 4 / 12

| scene | completing cells | needed |
|:--|:--|:--|
| 302c5c99 | d0@g1.5s1, d0@g1.5s0.8 | both, curvature |
| 382a7247 | d10@g1s1, d0@g1s1, d0@g1s0.8 | none (identity completes) |
| ce161092 | - | - |
| e37aa159 | d10@g1.25s1, d0@g1.25s1 | curvature |
| e5e47fd2 | - | - |
| fd01e9ab | d0@g1s1, d0@g1s0.8, d0@g1.5s0.6 | none (identity completes) |
| 2f227f2c | d10@g1s1, d10@g1.25s1, d10@g1s0.8, d10@g1s0.6, d10@g1.5s0.8, d10@g1.5s0.6, d0@g1s1, d0@g1.25s1, d0@g1.5s1, d0@g1.5s0.6 | none (identity completes) |
| 3ce13eed | d0@g1.25s1 | curvature |
| 94c1d722 | d0@g1.25s1 | curvature |
| a174faed | d10@g1.5s1, d0@g1.25s1 | curvature |
| eaba3f6a | d10@g1s1, d10@g1s0.8, d10@g1s0.6, d10@g1.5s0.8, d10@g1.5s0.6, d0@g1s1, d0@g1s0.6 | none (identity completes) |
| f9e361aa | d10@g1s0.8 | speed |

window end (k_eff + win): offset from the log / lateral (+ = ego right of the log), distance to the route, fed speed; mean over
completing vs failing cells of each transform

| cell | n done | w_off done | w_droute done | w_v done | w_off fail | w_droute fail | w_v fail |
|:--|--:|--:|--:|--:|--:|--:|--:|
| d10@g1s1 | 3 | 2.73 | 1.43 | 7.32 | 3.69 | 1.97 | 6.26 |
| d10@g1.25s1 | 2 | 3.02 | 2.76 | 7.50 | 3.80 | 2.51 | 6.16 |
| d10@g1.5s1 | 1 | 1.81 | 1.81 | 5.64 | 4.57 | 3.87 | 6.12 |
| d10@g2s1 | 0 | - | - | - | 5.99 | 4.27 | 4.25 |
| d10@g1s0.8 | 3 | 4.88 | 0.40 | 1.35 | 7.34 | 0.58 | 0.80 |
| d10@g1s0.6 | 2 | 7.03 | 0.38 | 0.26 | 9.94 | 0.37 | 0.14 |
| d10@g1.5s0.8 | 2 | 4.39 | 0.36 | 1.06 | 7.18 | 0.75 | 0.90 |
| d10@g1.5s0.6 | 2 | 7.03 | 0.38 | 0.25 | 9.98 | 0.39 | 0.12 |
| d0@g1s1 | 4 | 4.41 | 1.11 | 7.09 | 4.57 | 3.24 | 6.53 |
| d0@g1.25s1 | 5 | 3.34 | 1.51 | 6.72 | 5.57 | 4.29 | 5.64 |
| d0@g1.5s1 | 2 | 3.10 | 4.29 | 4.98 | 8.43 | 6.34 | 4.88 |
| d0@g2s1 | 0 | - | - | - | 9.84 | 6.44 | 4.38 |
| d0@g1s0.8 | 2 | 11.18 | 3.56 | 1.39 | 6.56 | 1.02 | 0.94 |
| d0@g1s0.6 | 1 | 8.19 | 0.47 | 0.34 | 10.28 | 1.00 | 0.20 |
| d0@g1.5s0.8 | 1 | 2.65 | 4.37 | 2.42 | 7.19 | 3.83 | 2.15 |
| d0@g1.5s0.6 | 2 | 10.57 | 2.46 | 0.33 | 9.01 | 1.70 | 0.52 |
