# vred Phase A: zero-shot Qwen3-VL-4B, one forward pass, 1153 visual tokens, both cameras, as served

3281 labelled requests with saved frames, 21 routes (13 with an ego red within 50 m). Each request went through the closed-loop serving path (HTTP, JPEG bytes, GPU preprocessing, option scoring), one at a time on a quiet card. Estimate [95% route-cluster bootstrap CI, 2000 resamples, seed 0]. Gate = point estimate against the registered line.

| readout | value [95% CI] | requests | routes | registered line | pass |
|:--|:--|--:|--:|:--|:--|
| ego red / yellow answered red, 0-50 m | 91.2% [85.1, 97.9] | 398 | 13 | >= 80% | yes |
| ... 0-20 m | 95.4% [89.4, 99.1] | 324 | 13 |  |  |
| ... 20-50 m | 73.0% [65.5, 100.0] | 74 | 2 |  |  |
| ... 50-80 m (listed, not scored) | 0.0% [0.0, 0.0] | 39 | 1 |  |  |
| ego red / yellow answered green, 0-50 m | 2.5% [0.3, 6.0] | 398 | 13 |  |  |
| ego green answered green, 0-50 m | 95.7% [91.9, 98.5] | 208 | 10 |  |  |
| another direction red, ego not red: answered red | 6.5% [3.4, 10.6] | 1069 | 17 | <= 10% | yes |
| no light: answered red | 0.0% [0.0, 0.0] | 1752 | 7 | <= 2% | yes |
| no light: answered green | 0.0% [0.0, 0.0] | 1752 | 7 |  |  |
| answer light_for_other_lane, any frame | 0.1% [0.0, 0.4] | 3281 | 21 |  |  |

Latency over 3281 requests (client round trip, quiet card, batch 1): p50 127 ms, p95 128 ms, p99 129 ms; server service time p50 126 ms. Registered line p95 <= 600 ms: yes.

Gate: Q-light pass, latency pass. Proceed to the closed loop: **yes**.

Reproduction on the 233-frame sweep subset of decision 85 (fwd at 1153 tokens: ego red answered red 76/85, answered green 5/85, green answered green 59/68, no light answered red 0/80): here 68/73, 1/73, 54/60, 0/80.

Q-light confusion (rows: truth, columns: answer):

| row_0                 |   no_light |   red_or_yellow_for_ego |   green_for_ego |   light_for_other_lane |
|:----------------------|-----------:|------------------------:|----------------:|-----------------------:|
| ego green 0-50 m      |          0 |                       9 |             199 |                      0 |
| ego red/yellow 0-50 m |         23 |                     363 |              10 |                      2 |
| no light              |       1752 |                       0 |               0 |                      0 |
| other                 |        798 |                      78 |              45 |                      2 |

Per route (counts):

|   route |   requests |   red_n |   red_as_red |   green_n |   green_as_green |   nolight_n |   nolight_as_red |   other_n |   other_as_red |
|--------:|-----------:|--------:|-------------:|----------:|-----------------:|------------:|-----------------:|----------:|---------------:|
|   15102 |         58 |      16 |           16 |         2 |                2 |           0 |                0 |        42 |              3 |
|   15483 |        136 |      52 |           52 |        12 |                9 |           0 |                0 |        84 |              4 |
|   15612 |        218 |      49 |           47 |         0 |                0 |           0 |                0 |       166 |              2 |
|   16390 |         68 |       2 |            2 |        32 |               30 |           0 |                0 |        64 |              7 |
|   16508 |         79 |       0 |            0 |        34 |               32 |           0 |                0 |        79 |              6 |
|   16529 |        112 |      36 |           32 |         0 |                0 |           0 |                0 |        74 |              9 |
|   17280 |         60 |       0 |            0 |         0 |                0 |          60 |                0 |         0 |              0 |
|   19324 |        399 |       0 |            0 |         0 |                0 |         349 |                0 |        50 |              0 |
|   19832 |        399 |       0 |            0 |         0 |                0 |         353 |                0 |        46 |              0 |
|   22535 |         98 |       0 |            0 |         0 |                0 |          98 |                0 |         0 |              0 |
|   24497 |        399 |       0 |            0 |         0 |                0 |         399 |                0 |         0 |              0 |
|   24944 |        179 |     108 |           88 |        21 |               20 |           0 |                0 |        21 |              1 |
|    2520 |        399 |       0 |            0 |         0 |                0 |         356 |                0 |        42 |              0 |
|   27043 |         62 |      26 |           26 |        17 |               17 |           0 |                0 |        36 |              3 |
|   27297 |         47 |      12 |            9 |        19 |               19 |           0 |                0 |        34 |              0 |
|   27787 |         45 |       3 |            3 |        16 |               16 |           0 |                0 |        41 |              1 |
|   27870 |         44 |      18 |           17 |         0 |                0 |           0 |                0 |        25 |             12 |
|   28147 |        103 |      11 |            6 |        28 |               27 |           0 |                0 |        92 |             17 |
|     334 |         58 |      10 |           10 |        27 |               27 |           0 |                0 |        48 |              1 |
|   37969 |        137 |       0 |            0 |         0 |                0 |         137 |                0 |         0 |              0 |
|    9196 |        181 |      55 |           55 |         0 |                0 |           0 |                0 |       125 |              3 |
