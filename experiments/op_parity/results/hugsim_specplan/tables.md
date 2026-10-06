## T0 coverage: scenarios per (arm, preset); turning routes 23, straight 41

| arm        |   exam |   spec |   specplan |
|:-----------|-------:|-------:|-----------:|
| P2-F-s0    |     64 |     64 |         64 |
| P2-F-s1    |     64 |     64 |         64 |
| P2H10-F-s0 |     64 |     64 |         64 |
| P2H10-F-s1 |     64 |     64 |         64 |

## T1 per (arm, preset), 64 scenarios, one run each

| arm        | preset   |   n |    HD |    RC |   complete |   spin |   bg |   fg |   off_route |   stuck | osc runs   | flips/100 moving steps (mean)   | w_rms deg/step (mean)   |
|:-----------|:---------|----:|------:|------:|-----------:|-------:|-----:|-----:|------------:|--------:|:-----------|:--------------------------------|:------------------------|
| P2-F-s0    | exam     |  64 | 0.416 | 0.553 |         20 |      8 |    6 |   28 |           2 |       0 | 0          | 0.24488558890372636             | 1.2521179997942924      |
| P2-F-s0    | spec     |  64 | 0.380 | 0.501 |         18 |      2 |   11 |   30 |           3 |       0 | 0          | 0.0                             | 0.29312040262543787     |
| P2-F-s0    | specplan |  64 | 0.334 | 0.454 |         16 |      6 |    8 |   27 |           7 |       0 | 0          | 2.0985402299236693              | 1.0194761320624222      |
| P2-F-s1    | exam     |  64 | 0.376 | 0.508 |         15 |     11 |    9 |   28 |           1 |       0 | 0          | 0.15669515669515638             | 1.5657835692870175      |
| P2-F-s1    | spec     |  64 | 0.407 | 0.519 |         21 |      1 |   11 |   28 |           3 |       0 | 0          | 0.0                             | 0.26399800650664573     |
| P2-F-s1    | specplan |  64 | 0.343 | 0.467 |         15 |      1 |   11 |   28 |           9 |       0 | 0          | 2.024011753226837               | 0.817656180014824       |
| P2H10-F-s0 | exam     |  64 | 0.412 | 0.546 |         22 |      7 |    6 |   29 |           0 |       0 | 0          | 0.17204556445338548             | 1.4303011466505082      |
| P2H10-F-s0 | spec     |  64 | 0.388 | 0.510 |         21 |      2 |   11 |   27 |           3 |       0 | 0          | 0.0                             | 0.27595720109639066     |
| P2H10-F-s0 | specplan |  64 | 0.347 | 0.463 |         17 |      2 |   11 |   30 |           4 |       0 | 0          | 1.9442156544796252              | 0.9038272645189023      |
| P2H10-F-s1 | exam     |  64 | 0.387 | 0.522 |         15 |     11 |   10 |   28 |           0 |       0 | 0          | 0.12419025660610421             | 1.427538457646261       |
| P2H10-F-s1 | spec     |  64 | 0.403 | 0.521 |         21 |      2 |   11 |   27 |           3 |       0 | 0          | 0.0                             | 0.28349168906740463     |
| P2H10-F-s1 | specplan |  64 | 0.331 | 0.455 |         14 |      4 |   12 |   27 |           7 |       0 | 0          | 2.9627513506115872              | 0.8848437704886004      |
| WA-JEPA    | exam     |  64 | 0.451 | 0.565 |         28 |      4 |    1 |   26 |           5 |       0 | -          | -                               | -                       |

## T2 HD by group x preset x stratum (per-scenario mean over the group's seeds / arms; bootstrap 95% CI over scenarios)

| group    | stratum     | exam                 | spec                 | specplan             |
|:---------|:------------|:---------------------|:---------------------|:---------------------|
| P2       | all 64      | 0.396 [0.311, 0.487] | 0.393 [0.302, 0.498] | 0.338 [0.258, 0.433] |
| P2       | turning 23  | 0.301 [0.175, 0.439] | 0.280 [0.158, 0.410] | 0.172 [0.107, 0.245] |
| P2       | straight 41 | 0.449 [0.341, 0.556] | 0.457 [0.333, 0.580] | 0.432 [0.314, 0.548] |
| P2+hinge | all 64      | 0.400 [0.311, 0.491] | 0.395 [0.305, 0.499] | 0.339 [0.260, 0.428] |
| P2+hinge | turning 23  | 0.322 [0.193, 0.454] | 0.278 [0.161, 0.405] | 0.168 [0.101, 0.251] |
| P2+hinge | straight 41 | 0.443 [0.331, 0.554] | 0.461 [0.336, 0.586] | 0.434 [0.319, 0.544] |
| all 4    | all 64      | 0.398 [0.314, 0.489] | 0.394 [0.303, 0.500] | 0.339 [0.260, 0.430] |
| all 4    | turning 23  | 0.312 [0.187, 0.445] | 0.279 [0.160, 0.407] | 0.170 [0.106, 0.245] |
| all 4    | straight 41 | 0.446 [0.335, 0.555] | 0.459 [0.334, 0.583] | 0.433 [0.318, 0.543] |
| WA-JEPA  | all 64      | -                    | -                    | 0.451 [0.357, 0.551] |
| WA-JEPA  | turning 23  | -                    | -                    | 0.442 [0.280, 0.608] |
| WA-JEPA  | straight 41 | -                    | -                    | 0.456 [0.334, 0.574] |

## T3 end classes by stratum (mean count per arm, over the group's arms; spin = heading error >= 60 deg)

| stratum     | preset   |   complete |   spin |   bg_coll |   fg_coll |   off_route |   stuck |    HD |    RC | oscillating runs   | flips/100 moving   |
|:------------|:---------|-----------:|-------:|----------:|----------:|------------:|--------:|------:|------:|:-------------------|:-------------------|
| turning 23  | exam     |       3.25 |   5.5  |      4.5  |      9    |        0.75 |       0 | 0.312 | 0.467 | 0.0                | 0.2                |
| turning 23  | spec     |       2    |   1.5  |      9    |      7.5  |        3    |       0 | 0.279 | 0.407 | 0.0                | 0.0                |
| turning 23  | specplan |       0.25 |   2.25 |      4.75 |     11.25 |        4.5  |       0 | 0.17  | 0.311 | 0.0                | 2.7                |
| turning 23  | WA-JEPA  |       8    |   1    |      0    |      9    |        5    |       0 | 0.442 | 0.509 | -                  | -                  |
| straight 41 | exam     |      14.75 |   3.75 |      3.25 |     19.25 |        0    |       0 | 0.446 | 0.569 | 0.0                | 0.2                |
| straight 41 | spec     |      18.25 |   0.25 |      2    |     20.5  |        0    |       0 | 0.459 | 0.573 | 0.0                | 0.0                |
| straight 41 | specplan |      15.25 |   1    |      5.75 |     16.75 |        2.25 |       0 | 0.433 | 0.543 | 0.0                | 2.0                |
| straight 41 | WA-JEPA  |      20    |   3    |      1    |     17    |        0    |       0 | 0.456 | 0.596 | -                  | -                  |

## T4 paired HD difference (scenario = unit, seed / arm mean within the group), bootstrap 95% CI; wins / losses / ties at |d| < 0.02

| group      | stratum     | comparison         | mean diff [CI]          |   wins |   losses |   ties |
|:-----------|:------------|:-------------------|:------------------------|-------:|---------:|-------:|
| P2         | all 64      | specplan - spec    | -0.055 [-0.107, -0.007] |     13 |       22 |     29 |
| P2         | all 64      | specplan - exam    | -0.057 [-0.118, +0.001] |     15 |       24 |     25 |
| P2         | all 64      | specplan - WA-JEPA | -0.112 [-0.200, -0.027] |     24 |       24 |     16 |
| P2         | turning 23  | specplan - spec    | -0.108 [-0.190, -0.037] |      2 |       11 |     10 |
| P2         | turning 23  | specplan - exam    | -0.129 [-0.212, -0.055] |      1 |       13 |      9 |
| P2         | turning 23  | specplan - WA-JEPA | -0.270 [-0.411, -0.142] |      4 |       14 |      5 |
| P2         | straight 41 | specplan - spec    | -0.026 [-0.092, +0.036] |     11 |       11 |     19 |
| P2         | straight 41 | specplan - exam    | -0.018 [-0.100, +0.068] |     14 |       11 |     16 |
| P2         | straight 41 | specplan - WA-JEPA | -0.024 [-0.130, +0.081] |     20 |       10 |     11 |
| P2+hinge   | all 64      | specplan - spec    | -0.056 [-0.115, -0.002] |     14 |       18 |     32 |
| P2+hinge   | all 64      | specplan - exam    | -0.061 [-0.131, +0.004] |     17 |       21 |     26 |
| P2+hinge   | all 64      | specplan - WA-JEPA | -0.112 [-0.204, -0.021] |     26 |       24 |     14 |
| P2+hinge   | turning 23  | specplan - spec    | -0.110 [-0.212, -0.019] |      4 |       11 |      8 |
| P2+hinge   | turning 23  | specplan - exam    | -0.154 [-0.252, -0.055] |      2 |       10 |     11 |
| P2+hinge   | turning 23  | specplan - WA-JEPA | -0.273 [-0.426, -0.137] |      5 |       13 |      5 |
| P2+hinge   | straight 41 | specplan - spec    | -0.027 [-0.098, +0.039] |     10 |        7 |     24 |
| P2+hinge   | straight 41 | specplan - exam    | -0.009 [-0.097, +0.073] |     15 |       11 |     15 |
| P2+hinge   | straight 41 | specplan - WA-JEPA | -0.021 [-0.130, +0.090] |     21 |       11 |      9 |
| all 4      | all 64      | specplan - spec    | -0.056 [-0.108, -0.007] |     15 |       23 |     26 |
| all 4      | all 64      | specplan - exam    | -0.059 [-0.122, -0.001] |     15 |       22 |     27 |
| all 4      | all 64      | specplan - WA-JEPA | -0.112 [-0.201, -0.025] |     25 |       25 |     14 |
| all 4      | turning 23  | specplan - spec    | -0.109 [-0.198, -0.029] |      3 |       12 |      8 |
| all 4      | turning 23  | specplan - exam    | -0.141 [-0.228, -0.062] |      1 |       11 |     11 |
| all 4      | turning 23  | specplan - WA-JEPA | -0.271 [-0.413, -0.137] |      4 |       14 |      5 |
| all 4      | straight 41 | specplan - spec    | -0.026 [-0.092, +0.037] |     12 |       11 |     18 |
| all 4      | straight 41 | specplan - exam    | -0.013 [-0.095, +0.069] |     14 |       11 |     16 |
| all 4      | straight 41 | specplan - WA-JEPA | -0.023 [-0.128, +0.084] |     21 |       11 |      9 |
| P2-F-s0    | all 64      | specplan - spec    | -0.046 [-0.110, +0.014] |     14 |       23 |     27 |
| P2-F-s0    | all 64      | specplan - exam    | -0.082 [-0.160, -0.006] |     13 |       24 |     27 |
| P2-F-s1    | all 64      | specplan - spec    | -0.064 [-0.115, -0.020] |     10 |       22 |     32 |
| P2-F-s1    | all 64      | specplan - exam    | -0.033 [-0.096, +0.026] |     16 |       19 |     29 |
| P2H10-F-s0 | all 64      | specplan - spec    | -0.041 [-0.103, +0.022] |     14 |       18 |     32 |
| P2H10-F-s0 | all 64      | specplan - exam    | -0.066 [-0.151, +0.015] |     16 |       23 |     25 |
| P2H10-F-s1 | all 64      | specplan - spec    | -0.071 [-0.132, -0.017] |     13 |       23 |     28 |
| P2H10-F-s1 | all 64      | specplan - exam    | -0.056 [-0.123, +0.004] |     17 |       21 |     26 |

## T5 the turning routes: HD per preset (mean over the 4 arms), specplan end classes of the 4 arms

| scenario                      |   route turn deg |   exam |   spec |   specplan | specplan classes (P2 s0, s1, hinge s0, s1)   |
|:------------------------------|-----------------:|-------:|-------:|-----------:|:---------------------------------------------|
| scene-0041-medium-00          |               86 |  0.138 |  0.166 |      0.126 | fg_coll fg_coll fg_coll fg_coll              |
| scene-100613054308-hard-00    |               37 |  0.55  |  0.182 |      0.135 | fg_coll fg_coll fg_coll fg_coll              |
| scene-102751446607-medium-01  |               92 |  0.784 |  0.95  |      0.382 | off_route off_route fg_coll off_route        |
| scene-113792265837-easy-00    |               86 |  0.742 |  0.829 |      0.762 | spin off_route off_route spin                |
| scene-1290_1490-medium-01     |              154 |  0.156 |  0.147 |      0.116 | fg_coll fg_coll fg_coll fg_coll              |
| scene-132384196576-medium-01  |               40 |  0.803 |  0.984 |      0.376 | spin complete spin spin                      |
| scene-144248042870-extreme-00 |               64 |  0     |  0.004 |      0     | fg_coll fg_coll fg_coll fg_coll              |
| scene-152217047339-extreme-00 |               44 |  0.064 |  0.051 |      0.046 | fg_coll fg_coll fg_coll fg_coll              |
| scene-152217047339-medium-00  |               44 |  0.78  |  0.222 |      0.261 | off_route off_route off_route off_route      |
| scene-164701907483-easy-00    |              108 |  0.135 |  0.439 |      0.113 | spin spin spin spin                          |
| scene-164701907483-hard-00    |              108 |  0.401 |  0.433 |      0.128 | fg_coll fg_coll fg_coll fg_coll              |
| scene-166085257829-extreme-00 |               82 |  0.092 |  0.039 |      0.155 | fg_coll fg_coll fg_coll fg_coll              |
| scene-3000_3200-hard-00       |              237 |  0.044 |  0.062 |      0.053 | bg_coll bg_coll bg_coll bg_coll              |
| scene-3000_3200-medium-00     |              237 |  0.08  |  0.079 |      0.053 | bg_coll bg_coll bg_coll bg_coll              |
| scene-3400_3600-extreme-00    |               80 |  0     |  0     |      0     | fg_coll fg_coll fg_coll fg_coll              |
| scene-3400_3600-hard-00       |               80 |  0.057 |  0.053 |      0.05  | fg_coll fg_coll fg_coll fg_coll              |
| scene-570_770-easy-00         |               96 |  0.211 |  0.205 |      0.205 | off_route bg_coll off_route off_route        |
| scene-570_770-medium-00       |               96 |  0.2   |  0.215 |      0.19  | off_route off_route bg_coll bg_coll          |
| scene-5980_6180-easy-00       |              102 |  0.44  |  0.254 |      0.219 | off_route off_route off_route off_route      |
| scene-5980_6180-extreme-00    |              102 |  0.009 |  0.008 |      0.008 | fg_coll fg_coll fg_coll fg_coll              |
| scene-770_970-extreme-00      |              107 |  0.024 |  0.022 |      0.02  | fg_coll fg_coll fg_coll fg_coll              |
| scene-8440_8640-easy-00       |               87 |  0.594 |  0.712 |      0.102 | bg_coll bg_coll bg_coll bg_coll              |
| scene-881121006469-hard-00    |               73 |  0.862 |  0.36  |      0.417 | bg_coll bg_coll bg_coll bg_coll              |

## T6 requested curvature (what the lateral path is fed), mean per run then over runs: |kappa| (1/m), mean |kappa_t - kappa_t-1| (1/m per step), sign flips of kappa per step

| turning   | preset   |   k_abs |   k_step |   k_sign_flips |
|:----------|:---------|--------:|---------:|---------------:|
| False     | exam     |  0.0045 |   0.0015 |         0.076  |
| False     | spec     |  0.0036 |   0.0011 |         0.0673 |
| False     | specplan |  0.0126 |   0.0071 |         0.131  |
| True      | exam     |  0.0113 |   0.003  |         0.0765 |
| True      | spec     |  0.0066 |   0.0018 |         0.0686 |
| True      | specplan |  0.022  |   0.009  |         0.1159 |

