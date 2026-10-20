# ROUND3: PAI human rows by state (own fed speed and own 4 s label)

Rule: standing v0 < 0.3 m/s and label path < 0.5 m; creeping 0.3 <= v0 < 1.0, path < 0.5 m; launching v0 < 1.0, path >= 0.5 m; stopping v0 >= 1.0, last 1 s of the label < 0.5 m; moving otherwise. `drawable` = R2JF weight > 0 (outside held and the exam links); `per batch` = expected rows of that state in J's batch (32 PAI human rows).

| part | drawable | standing | creeping | launching | stopping | moving | standing per batch | launching per batch |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| ph | 114453 | 9216 (8.1%) | 779 (0.7%) | 8924 (7.8%) | 5319 (4.6%) | 90215 (78.8%) | 1.29 | 1.25 |
| ph05 | 104415 | 11371 (10.9%) | 407 (0.4%) | 9707 (9.3%) | 1829 (1.8%) | 81101 (77.7%) | 0.87 | 0.74 |
| ph15 | 83671 | 3956 (4.7%) | 534 (0.6%) | 7218 (8.6%) | 5866 (7.0%) | 66097 (79.0%) | 0.38 | 0.69 |
| all (per batch) | | 2.54 | 0.19 | 2.68 | 1.44 | 25.15 | 2.54 | 2.68 |

Standing rows: the human's wait after the label window (log s until it has moved 0.5 m; stretch rows on their own clock)

| part | standing scenes | <=2 s | 2-6 s | 6-10 s | >10 s | not in clip |
|:--|--:|--:|--:|--:|--:|--:|
| ph | 207 | 2099 (23%) | 1624 (18%) | 53 (1%) | 19 (0%) | 5421 (59%) |
| ph05 | 258 | 3021 (27%) | 2502 (22%) | 110 (1%) | 25 (0%) | 5713 (50%) |
| ph15 | 123 | 436 (11%) | 77 (2%) | 16 (0%) | 0 (0%) | 3427 (87%) |

The 32 S / trainval pool rows of a batch (JT00S, both arms): standing 0.14, creeping 0.05, launching 2.47, stopping 1.71, moving 27.63

navtrain_full (103288 rows): standing 22 (0.02%), creeping 363 (0.35%), launching 13467 (13.04%), stopping 9124 (8.83%), moving 80312 (77.76%)

Per batch (navtrain 51 / 83 + 32 S rows + J's 32 PAI human rows; the 13 yaw-rate rows not counted), J vs CTRL: standing 2.69 vs 0.15, creeping 0.42 vs 0.34, launching 11.81 vs 13.30, stopping 7.66 vs 9.04, moving 92.43 vs 92.16; standing / (standing + launching) 18.5% vs 1.1%.
