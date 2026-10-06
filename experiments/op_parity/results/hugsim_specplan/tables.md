## T0 coverage: scenarios per (arm, preset); turning routes 23, straight 41

| arm        |   exam |   mpc |   smooth |   spec |   specplan |
|:-----------|-------:|------:|---------:|-------:|-----------:|
| P2-F-s0    |     64 |    64 |       64 |     64 |         64 |
| P2-F-s1    |     64 |    64 |       64 |     64 |         64 |
| P2H10-F-s0 |     64 |    64 |       64 |     64 |         64 |
| P2H10-F-s1 |     64 |    64 |       64 |     64 |         64 |

## T1 per (arm, preset), 64 scenarios, one run each

| arm        | preset   |   n |    HD |    RC |   complete |   spin |   bg |   fg |   off_route |   stuck | osc runs   | flips/100 moving steps (mean)   | w_rms deg/step (mean)   |
|:-----------|:---------|----:|------:|------:|-----------:|-------:|-----:|-----:|------------:|--------:|:-----------|:--------------------------------|:------------------------|
| P2-F-s0    | exam     |  64 | 0.416 | 0.553 |         20 |      8 |    6 |   28 |           2 |       0 | 0          | 0.24488558890372636             | 1.2521179997942924      |
| P2-F-s0    | spec     |  64 | 0.380 | 0.501 |         18 |      2 |   11 |   30 |           3 |       0 | 0          | 0.0                             | 0.29312040262543787     |
| P2-F-s0    | specplan |  64 | 0.334 | 0.454 |         16 |      6 |    8 |   27 |           7 |       0 | 0          | 2.0985402299236693              | 1.0194761320624222      |
| P2-F-s0    | smooth   |  64 | 0.419 | 0.538 |         24 |      0 |    9 |   29 |           2 |       0 | 0          | 0.0                             | 0.5452782284174134      |
| P2-F-s0    | mpc      |  64 | 0.388 | 0.508 |         20 |      0 |   10 |   31 |           3 |       0 | 0          | 0.0                             | 0.2589345959358501      |
| P2-F-s1    | exam     |  64 | 0.376 | 0.508 |         15 |     11 |    9 |   28 |           1 |       0 | 0          | 0.15669515669515638             | 1.5657835692870175      |
| P2-F-s1    | spec     |  64 | 0.407 | 0.519 |         21 |      1 |   11 |   28 |           3 |       0 | 0          | 0.0                             | 0.26399800650664573     |
| P2-F-s1    | specplan |  64 | 0.343 | 0.467 |         15 |      1 |   11 |   28 |           9 |       0 | 0          | 2.024011753226837               | 0.817656180014824       |
| P2-F-s1    | smooth   |  64 | 0.428 | 0.538 |         24 |      0 |    9 |   30 |           1 |       0 | 0          | 0.08370535714285703             | 0.5165232299892943      |
| P2-F-s1    | mpc      |  64 | 0.404 | 0.520 |         22 |      0 |    7 |   31 |           4 |       0 | 0          | 0.0                             | 0.2245972755768033      |
| P2H10-F-s0 | exam     |  64 | 0.412 | 0.546 |         22 |      7 |    6 |   29 |           0 |       0 | 0          | 0.17204556445338548             | 1.4303011466505082      |
| P2H10-F-s0 | spec     |  64 | 0.388 | 0.510 |         21 |      2 |   11 |   27 |           3 |       0 | 0          | 0.0                             | 0.27595720109639066     |
| P2H10-F-s0 | specplan |  64 | 0.347 | 0.463 |         17 |      2 |   11 |   30 |           4 |       0 | 0          | 1.9442156544796252              | 0.9038272645189023      |
| P2H10-F-s0 | smooth   |  64 | 0.418 | 0.533 |         25 |      0 |    6 |   31 |           2 |       0 | 0          | 0.02297794117647047             | 0.5371892341991241      |
| P2H10-F-s0 | mpc      |  64 | 0.380 | 0.500 |         20 |      0 |   10 |   30 |           4 |       0 | 0          | 0.0                             | 0.256649329754417       |
| P2H10-F-s1 | exam     |  64 | 0.387 | 0.522 |         15 |     11 |   10 |   28 |           0 |       0 | 0          | 0.12419025660610421             | 1.427538457646261       |
| P2H10-F-s1 | spec     |  64 | 0.403 | 0.521 |         21 |      2 |   11 |   27 |           3 |       0 | 0          | 0.0                             | 0.28349168906740463     |
| P2H10-F-s1 | specplan |  64 | 0.331 | 0.455 |         14 |      4 |   12 |   27 |           7 |       0 | 0          | 2.9627513506115872              | 0.8848437704886004      |
| P2H10-F-s1 | smooth   |  64 | 0.448 | 0.560 |         26 |      0 |    7 |   29 |           2 |       0 | 0          | 0.022321428571428437            | 0.5138199077518734      |
| P2H10-F-s1 | mpc      |  64 | 0.409 | 0.524 |         22 |      0 |   10 |   29 |           3 |       0 | 0          | 0.0                             | 0.24737047288864272     |
| WA-JEPA    | exam     |  64 | 0.451 | 0.565 |         28 |      4 |    1 |   26 |           5 |       0 | -          | -                               | -                       |

## T2 HD by group x preset x stratum (per-scenario mean over the group's seeds / arms; bootstrap 95% CI over scenarios)

| group    | stratum     | exam                 | spec                 | specplan             | smooth               | mpc                  |
|:---------|:------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|
| P2       | all 64      | 0.396 [0.311, 0.487] | 0.393 [0.302, 0.498] | 0.338 [0.258, 0.433] | 0.424 [0.326, 0.528] | 0.396 [0.304, 0.502] |
| P2       | turning 23  | 0.301 [0.175, 0.439] | 0.280 [0.158, 0.410] | 0.172 [0.107, 0.245] | 0.313 [0.181, 0.451] | 0.284 [0.155, 0.423] |
| P2       | straight 41 | 0.449 [0.341, 0.556] | 0.457 [0.333, 0.580] | 0.432 [0.314, 0.548] | 0.486 [0.356, 0.610] | 0.459 [0.334, 0.581] |
| P2+hinge | all 64      | 0.400 [0.311, 0.491] | 0.395 [0.305, 0.499] | 0.339 [0.260, 0.428] | 0.433 [0.335, 0.534] | 0.395 [0.305, 0.494] |
| P2+hinge | turning 23  | 0.322 [0.193, 0.454] | 0.278 [0.161, 0.405] | 0.168 [0.101, 0.251] | 0.352 [0.204, 0.512] | 0.299 [0.171, 0.439] |
| P2+hinge | straight 41 | 0.443 [0.331, 0.554] | 0.461 [0.336, 0.586] | 0.434 [0.319, 0.544] | 0.478 [0.350, 0.601] | 0.448 [0.327, 0.570] |
| all 4    | all 64      | 0.398 [0.314, 0.489] | 0.394 [0.303, 0.500] | 0.339 [0.260, 0.430] | 0.428 [0.331, 0.531] | 0.395 [0.306, 0.497] |
| all 4    | turning 23  | 0.312 [0.187, 0.445] | 0.279 [0.160, 0.407] | 0.170 [0.106, 0.245] | 0.333 [0.193, 0.481] | 0.291 [0.163, 0.430] |
| all 4    | straight 41 | 0.446 [0.335, 0.555] | 0.459 [0.334, 0.583] | 0.433 [0.318, 0.543] | 0.482 [0.353, 0.605] | 0.454 [0.329, 0.576] |
| WA-JEPA  | all 64      | -                    | -                    | 0.451 [0.357, 0.551] | nan                  | nan                  |
| WA-JEPA  | turning 23  | -                    | -                    | 0.442 [0.280, 0.608] | nan                  | nan                  |
| WA-JEPA  | straight 41 | -                    | -                    | 0.456 [0.334, 0.574] | nan                  | nan                  |

## T3 end classes by stratum (mean count per arm, over the group's arms; spin = heading error >= 60 deg)

| stratum     | preset   |   complete |   spin |   bg_coll |   fg_coll |   off_route |   stuck |    HD |    RC | oscillating runs   | flips/100 moving   |
|:------------|:---------|-----------:|-------:|----------:|----------:|------------:|--------:|------:|------:|:-------------------|:-------------------|
| turning 23  | exam     |       3.25 |   5.5  |      4.5  |      9    |        0.75 |       0 | 0.312 | 0.467 | 0.0                | 0.2                |
| turning 23  | spec     |       2    |   1.5  |      9    |      7.5  |        3    |       0 | 0.279 | 0.407 | 0.0                | 0.0                |
| turning 23  | specplan |       0.25 |   2.25 |      4.75 |     11.25 |        4.5  |       0 | 0.17  | 0.311 | 0.0                | 2.7                |
| turning 23  | smooth   |       5.25 |   0    |      6    |     10.25 |        1.5  |       0 | 0.333 | 0.462 | 0.0                | 0.1                |
| turning 23  | mpc      |       3    |   0    |      7    |      9.5  |        3.5  |       0 | 0.291 | 0.413 | 0.0                | 0.0                |
| turning 23  | WA-JEPA  |       8    |   1    |      0    |      9    |        5    |       0 | 0.442 | 0.509 | -                  | -                  |
| straight 41 | exam     |      14.75 |   3.75 |      3.25 |     19.25 |        0    |       0 | 0.446 | 0.569 | 0.0                | 0.2                |
| straight 41 | spec     |      18.25 |   0.25 |      2    |     20.5  |        0    |       0 | 0.459 | 0.573 | 0.0                | 0.0                |
| straight 41 | specplan |      15.25 |   1    |      5.75 |     16.75 |        2.25 |       0 | 0.433 | 0.543 | 0.0                | 2.0                |
| straight 41 | smooth   |      19.5  |   0    |      1.75 |     19.5  |        0.25 |       0 | 0.482 | 0.587 | 0.0                | 0.0                |
| straight 41 | mpc      |      18    |   0    |      2.25 |     20.75 |        0    |       0 | 0.454 | 0.569 | 0.0                | 0.0                |
| straight 41 | WA-JEPA  |      20    |   3    |      1    |     17    |        0    |       0 | 0.456 | 0.596 | -                  | -                  |

## T4 paired HD difference (scenario = unit, seed / arm mean within the group), bootstrap 95% CI; wins / losses / ties at |d| < 0.02

| group      | stratum     | comparison         | mean diff [CI]          |   wins |   losses |   ties |
|:-----------|:------------|:-------------------|:------------------------|-------:|---------:|-------:|
| P2         | all 64      | specplan - spec    | -0.055 [-0.107, -0.007] |     13 |       22 |     29 |
| P2         | all 64      | specplan - exam    | -0.057 [-0.118, +0.001] |     15 |       24 |     25 |
| P2         | all 64      | specplan - WA-JEPA | -0.112 [-0.200, -0.027] |     24 |       24 |     16 |
| P2         | all 64      | smooth - spec      | +0.030 [+0.003, +0.063] |     15 |        7 |     42 |
| P2         | all 64      | smooth - exam      | +0.028 [-0.018, +0.074] |     18 |        9 |     37 |
| P2         | all 64      | smooth - specplan  | +0.085 [+0.029, +0.149] |     22 |       14 |     28 |
| P2         | all 64      | smooth - WA-JEPA   | -0.027 [-0.105, +0.043] |     30 |       16 |     18 |
| P2         | all 64      | mpc - spec         | +0.002 [-0.008, +0.014] |      8 |        9 |     47 |
| P2         | all 64      | mpc - exam         | +0.000 [-0.050, +0.051] |     18 |       17 |     29 |
| P2         | all 64      | mpc - specplan     | +0.058 [+0.008, +0.111] |     21 |       13 |     30 |
| P2         | all 64      | mpc - WA-JEPA      | -0.055 [-0.133, +0.018] |     28 |       20 |     16 |
| P2         | turning 23  | specplan - spec    | -0.108 [-0.190, -0.037] |      2 |       11 |     10 |
| P2         | turning 23  | specplan - exam    | -0.129 [-0.212, -0.055] |      1 |       13 |      9 |
| P2         | turning 23  | specplan - WA-JEPA | -0.270 [-0.411, -0.142] |      4 |       14 |      5 |
| P2         | turning 23  | smooth - spec      | +0.033 [+0.001, +0.078] |      7 |        3 |     13 |
| P2         | turning 23  | smooth - exam      | +0.012 [-0.069, +0.084] |      9 |        4 |     10 |
| P2         | turning 23  | smooth - specplan  | +0.141 [+0.060, +0.231] |     12 |        2 |      9 |
| P2         | turning 23  | smooth - WA-JEPA   | -0.129 [-0.284, +0.003] |      8 |        8 |      7 |
| P2         | turning 23  | mpc - spec         | +0.004 [-0.014, +0.024] |      5 |        5 |     13 |
| P2         | turning 23  | mpc - exam         | -0.016 [-0.103, +0.061] |      9 |        7 |      7 |
| P2         | turning 23  | mpc - specplan     | +0.112 [+0.034, +0.198] |     11 |        2 |     10 |
| P2         | turning 23  | mpc - WA-JEPA      | -0.158 [-0.305, -0.037] |      8 |        9 |      6 |
| P2         | straight 41 | specplan - spec    | -0.026 [-0.092, +0.036] |     11 |       11 |     19 |
| P2         | straight 41 | specplan - exam    | -0.018 [-0.100, +0.068] |     14 |       11 |     16 |
| P2         | straight 41 | specplan - WA-JEPA | -0.024 [-0.130, +0.081] |     20 |       10 |     11 |
| P2         | straight 41 | smooth - spec      | +0.028 [-0.006, +0.080] |      8 |        4 |     29 |
| P2         | straight 41 | smooth - exam      | +0.037 [-0.020, +0.089] |      9 |        5 |     27 |
| P2         | straight 41 | smooth - specplan  | +0.054 [-0.021, +0.136] |     10 |       12 |     19 |
| P2         | straight 41 | smooth - WA-JEPA   | +0.030 [-0.051, +0.114] |     22 |        8 |     11 |
| P2         | straight 41 | mpc - spec         | +0.001 [-0.010, +0.015] |      3 |        4 |     34 |
| P2         | straight 41 | mpc - exam         | +0.009 [-0.064, +0.074] |      9 |       10 |     22 |
| P2         | straight 41 | mpc - specplan     | +0.027 [-0.037, +0.092] |     10 |       11 |     20 |
| P2         | straight 41 | mpc - WA-JEPA      | +0.003 [-0.089, +0.091] |     20 |       11 |     10 |
| P2+hinge   | all 64      | specplan - spec    | -0.056 [-0.115, -0.002] |     14 |       18 |     32 |
| P2+hinge   | all 64      | specplan - exam    | -0.061 [-0.131, +0.004] |     17 |       21 |     26 |
| P2+hinge   | all 64      | specplan - WA-JEPA | -0.112 [-0.204, -0.021] |     26 |       24 |     14 |
| P2+hinge   | all 64      | smooth - spec      | +0.038 [+0.001, +0.078] |     16 |        8 |     40 |
| P2+hinge   | all 64      | smooth - exam      | +0.033 [-0.007, +0.078] |     17 |        9 |     38 |
| P2+hinge   | all 64      | smooth - specplan  | +0.094 [+0.026, +0.166] |     21 |       15 |     28 |
| P2+hinge   | all 64      | smooth - WA-JEPA   | -0.018 [-0.096, +0.054] |     31 |       16 |     17 |
| P2+hinge   | all 64      | mpc - spec         | -0.001 [-0.022, +0.017] |      8 |        9 |     47 |
| P2+hinge   | all 64      | mpc - exam         | -0.005 [-0.054, +0.042] |     19 |       17 |     28 |
| P2+hinge   | all 64      | mpc - specplan     | +0.056 [-0.003, +0.120] |     22 |       16 |     26 |
| P2+hinge   | all 64      | mpc - WA-JEPA      | -0.056 [-0.130, +0.013] |     29 |       21 |     14 |
| P2+hinge   | turning 23  | specplan - spec    | -0.110 [-0.212, -0.019] |      4 |       11 |      8 |
| P2+hinge   | turning 23  | specplan - exam    | -0.154 [-0.252, -0.055] |      2 |       10 |     11 |
| P2+hinge   | turning 23  | specplan - WA-JEPA | -0.273 [-0.426, -0.137] |      5 |       13 |      5 |
| P2+hinge   | turning 23  | smooth - spec      | +0.074 [+0.010, +0.158] |      8 |        4 |     11 |
| P2+hinge   | turning 23  | smooth - exam      | +0.030 [-0.057, +0.123] |      8 |        4 |     11 |
| P2+hinge   | turning 23  | smooth - specplan  | +0.184 [+0.068, +0.307] |     11 |        3 |      9 |
| P2+hinge   | turning 23  | smooth - WA-JEPA   | -0.090 [-0.260, +0.068] |      8 |        8 |      7 |
| P2+hinge   | turning 23  | mpc - spec         | +0.021 [-0.007, +0.056] |      4 |        3 |     16 |
| P2+hinge   | turning 23  | mpc - exam         | -0.024 [-0.119, +0.065] |      8 |        6 |      9 |
| P2+hinge   | turning 23  | mpc - specplan     | +0.130 [+0.034, +0.239] |     12 |        3 |      8 |
| P2+hinge   | turning 23  | mpc - WA-JEPA      | -0.143 [-0.295, -0.015] |      8 |       10 |      5 |
| P2+hinge   | straight 41 | specplan - spec    | -0.027 [-0.098, +0.039] |     10 |        7 |     24 |
| P2+hinge   | straight 41 | specplan - exam    | -0.009 [-0.097, +0.073] |     15 |       11 |     15 |
| P2+hinge   | straight 41 | specplan - WA-JEPA | -0.021 [-0.130, +0.090] |     21 |       11 |      9 |
| P2+hinge   | straight 41 | smooth - spec      | +0.017 [-0.024, +0.067] |      8 |        4 |     29 |
| P2+hinge   | straight 41 | smooth - exam      | +0.035 [-0.006, +0.078] |      9 |        5 |     27 |
| P2+hinge   | straight 41 | smooth - specplan  | +0.044 [-0.035, +0.133] |     10 |       12 |     19 |
| P2+hinge   | straight 41 | smooth - WA-JEPA   | +0.023 [-0.052, +0.092] |     23 |        8 |     10 |
| P2+hinge   | straight 41 | mpc - spec         | -0.013 [-0.041, +0.006] |      4 |        6 |     31 |
| P2+hinge   | straight 41 | mpc - exam         | +0.005 [-0.057, +0.059] |     11 |       11 |     19 |
| P2+hinge   | straight 41 | mpc - specplan     | +0.014 [-0.056, +0.090] |     10 |       13 |     18 |
| P2+hinge   | straight 41 | mpc - WA-JEPA      | -0.007 [-0.092, +0.067] |     21 |       11 |      9 |
| all 4      | all 64      | specplan - spec    | -0.056 [-0.108, -0.007] |     15 |       23 |     26 |
| all 4      | all 64      | specplan - exam    | -0.059 [-0.122, -0.001] |     15 |       22 |     27 |
| all 4      | all 64      | specplan - WA-JEPA | -0.112 [-0.201, -0.025] |     25 |       25 |     14 |
| all 4      | all 64      | smooth - spec      | +0.034 [+0.005, +0.068] |     16 |        8 |     40 |
| all 4      | all 64      | smooth - exam      | +0.031 [-0.009, +0.073] |     18 |        9 |     37 |
| all 4      | all 64      | smooth - specplan  | +0.090 [+0.032, +0.155] |     21 |       13 |     30 |
| all 4      | all 64      | smooth - WA-JEPA   | -0.022 [-0.102, +0.046] |     31 |       16 |     17 |
| all 4      | all 64      | mpc - spec         | +0.001 [-0.011, +0.012] |     10 |        8 |     46 |
| all 4      | all 64      | mpc - exam         | -0.003 [-0.050, +0.047] |     20 |       15 |     29 |
| all 4      | all 64      | mpc - specplan     | +0.057 [+0.006, +0.112] |     21 |       16 |     27 |
| all 4      | all 64      | mpc - WA-JEPA      | -0.055 [-0.132, +0.013] |     29 |       20 |     15 |
| all 4      | turning 23  | specplan - spec    | -0.109 [-0.198, -0.029] |      3 |       12 |      8 |
| all 4      | turning 23  | specplan - exam    | -0.141 [-0.228, -0.062] |      1 |       11 |     11 |
| all 4      | turning 23  | specplan - WA-JEPA | -0.271 [-0.413, -0.137] |      4 |       14 |      5 |
| all 4      | turning 23  | smooth - spec      | +0.054 [+0.006, +0.116] |      8 |        3 |     12 |
| all 4      | turning 23  | smooth - exam      | +0.021 [-0.059, +0.102] |      8 |        4 |     11 |
| all 4      | turning 23  | smooth - specplan  | +0.162 [+0.065, +0.268] |     11 |        2 |     10 |
| all 4      | turning 23  | smooth - WA-JEPA   | -0.109 [-0.273, +0.035] |      8 |        8 |      7 |
| all 4      | turning 23  | mpc - spec         | +0.013 [-0.006, +0.033] |      6 |        3 |     14 |
| all 4      | turning 23  | mpc - exam         | -0.020 [-0.105, +0.061] |      8 |        6 |      9 |
| all 4      | turning 23  | mpc - specplan     | +0.121 [+0.036, +0.214] |     12 |        3 |      8 |
| all 4      | turning 23  | mpc - WA-JEPA      | -0.150 [-0.299, -0.026] |      8 |        9 |      6 |
| all 4      | straight 41 | specplan - spec    | -0.026 [-0.092, +0.037] |     12 |       11 |     18 |
| all 4      | straight 41 | specplan - exam    | -0.013 [-0.095, +0.069] |     14 |       11 |     16 |
| all 4      | straight 41 | specplan - WA-JEPA | -0.023 [-0.128, +0.084] |     21 |       11 |      9 |
| all 4      | straight 41 | smooth - spec      | +0.023 [-0.012, +0.072] |      8 |        5 |     28 |
| all 4      | straight 41 | smooth - exam      | +0.036 [-0.012, +0.082] |     10 |        5 |     26 |
| all 4      | straight 41 | smooth - specplan  | +0.049 [-0.024, +0.130] |     10 |       11 |     20 |
| all 4      | straight 41 | smooth - WA-JEPA   | +0.026 [-0.052, +0.101] |     23 |        8 |     10 |
| all 4      | straight 41 | mpc - spec         | -0.006 [-0.022, +0.008] |      4 |        5 |     32 |
| all 4      | straight 41 | mpc - exam         | +0.007 [-0.057, +0.065] |     12 |        9 |     20 |
| all 4      | straight 41 | mpc - specplan     | +0.020 [-0.044, +0.089] |      9 |       13 |     19 |
| all 4      | straight 41 | mpc - WA-JEPA      | -0.002 [-0.089, +0.079] |     21 |       11 |      9 |
| P2-F-s0    | all 64      | specplan - spec    | -0.046 [-0.110, +0.014] |     14 |       23 |     27 |
| P2-F-s0    | all 64      | specplan - exam    | -0.082 [-0.160, -0.006] |     13 |       24 |     27 |
| P2-F-s0    | all 64      | smooth - spec      | +0.040 [+0.008, +0.078] |     14 |        8 |     42 |
| P2-F-s0    | all 64      | smooth - exam      | +0.003 [-0.059, +0.064] |     16 |       15 |     33 |
| P2-F-s0    | all 64      | mpc - spec         | +0.008 [-0.006, +0.024] |      8 |        9 |     47 |
| P2-F-s0    | all 64      | mpc - exam         | -0.028 [-0.091, +0.030] |     17 |       16 |     31 |
| P2-F-s1    | all 64      | specplan - spec    | -0.064 [-0.115, -0.020] |     10 |       22 |     32 |
| P2-F-s1    | all 64      | specplan - exam    | -0.033 [-0.096, +0.026] |     16 |       19 |     29 |
| P2-F-s1    | all 64      | smooth - spec      | +0.021 [-0.011, +0.061] |     12 |       10 |     42 |
| P2-F-s1    | all 64      | smooth - exam      | +0.052 [+0.009, +0.102] |     17 |        8 |     39 |
| P2-F-s1    | all 64      | mpc - spec         | -0.003 [-0.017, +0.011] |      9 |       13 |     42 |
| P2-F-s1    | all 64      | mpc - exam         | +0.029 [-0.030, +0.090] |     19 |       15 |     30 |
| P2H10-F-s0 | all 64      | specplan - spec    | -0.041 [-0.103, +0.022] |     14 |       18 |     32 |
| P2H10-F-s0 | all 64      | specplan - exam    | -0.066 [-0.151, +0.015] |     16 |       23 |     25 |
| P2H10-F-s0 | all 64      | smooth - spec      | +0.030 [-0.017, +0.076] |     17 |        8 |     39 |
| P2H10-F-s0 | all 64      | smooth - exam      | +0.006 [-0.043, +0.054] |     15 |       13 |     36 |
| P2H10-F-s0 | all 64      | mpc - spec         | -0.008 [-0.047, +0.018] |     10 |        9 |     45 |
| P2H10-F-s0 | all 64      | mpc - exam         | -0.033 [-0.088, +0.021] |     18 |       17 |     29 |
| P2H10-F-s1 | all 64      | specplan - spec    | -0.071 [-0.132, -0.017] |     13 |       23 |     28 |
| P2H10-F-s1 | all 64      | specplan - exam    | -0.056 [-0.123, +0.004] |     17 |       21 |     26 |
| P2H10-F-s1 | all 64      | smooth - spec      | +0.045 [+0.013, +0.085] |     14 |        8 |     42 |
| P2H10-F-s1 | all 64      | smooth - exam      | +0.061 [+0.015, +0.112] |     20 |        8 |     36 |
| P2H10-F-s1 | all 64      | mpc - spec         | +0.007 [-0.004, +0.021] |     11 |       13 |     40 |
| P2H10-F-s1 | all 64      | mpc - exam         | +0.022 [-0.032, +0.077] |     19 |       15 |     30 |

## T5 the turning routes: HD per preset (mean over the 4 arms), end classes of the 4 arms

| scenario                      |   route turn deg |   exam |   spec |   specplan |   smooth |   mpc | specplan classes (P2 s0, s1, hinge s0, s1)   | smooth classes (P2 s0, s1, hinge s0, s1)   | mpc classes (P2 s0, s1, hinge s0, s1)   |
|:------------------------------|-----------------:|-------:|-------:|-----------:|---------:|------:|:---------------------------------------------|:-------------------------------------------|:----------------------------------------|
| scene-0041-medium-00          |               86 |  0.138 |  0.166 |      0.126 |    0.194 | 0.17  | fg_coll fg_coll fg_coll fg_coll              | bg_coll bg_coll bg_coll bg_coll            | fg_coll off_route off_route off_route   |
| scene-100613054308-hard-00    |               37 |  0.55  |  0.182 |      0.135 |    0.169 | 0.165 | fg_coll fg_coll fg_coll fg_coll              | bg_coll bg_coll bg_coll bg_coll            | bg_coll bg_coll bg_coll bg_coll         |
| scene-102751446607-medium-01  |               92 |  0.784 |  0.95  |      0.382 |    0.997 | 0.98  | off_route off_route fg_coll off_route        | complete complete complete complete        | bg_coll complete bg_coll bg_coll        |
| scene-113792265837-easy-00    |               86 |  0.742 |  0.829 |      0.762 |    0.998 | 0.99  | spin off_route off_route spin                | complete complete complete complete        | complete complete complete complete     |
| scene-1290_1490-medium-01     |              154 |  0.156 |  0.147 |      0.116 |    0.161 | 0.162 | fg_coll fg_coll fg_coll fg_coll              | fg_coll fg_coll fg_coll fg_coll            | fg_coll fg_coll fg_coll fg_coll         |
| scene-132384196576-medium-01  |               40 |  0.803 |  0.984 |      0.376 |    0.993 | 0.993 | spin complete spin spin                      | complete complete complete complete        | complete complete complete complete     |
| scene-144248042870-extreme-00 |               64 |  0     |  0.004 |      0     |    0.001 | 0.026 | fg_coll fg_coll fg_coll fg_coll              | fg_coll fg_coll fg_coll fg_coll            | bg_coll bg_coll bg_coll bg_coll         |
| scene-152217047339-extreme-00 |               44 |  0.064 |  0.051 |      0.046 |    0.054 | 0.045 | fg_coll fg_coll fg_coll fg_coll              | fg_coll fg_coll fg_coll fg_coll            | fg_coll fg_coll fg_coll fg_coll         |
| scene-152217047339-medium-00  |               44 |  0.78  |  0.222 |      0.261 |    0.819 | 0.289 | off_route off_route off_route off_route      | bg_coll complete complete complete         | bg_coll bg_coll bg_coll bg_coll         |
| scene-164701907483-easy-00    |              108 |  0.135 |  0.439 |      0.113 |    0.706 | 0.491 | spin spin spin spin                          | complete bg_coll complete complete         | off_route complete off_route bg_coll    |
| scene-164701907483-hard-00    |              108 |  0.401 |  0.433 |      0.128 |    0.596 | 0.535 | fg_coll fg_coll fg_coll fg_coll              | complete fg_coll complete complete         | bg_coll fg_coll complete complete       |
| scene-166085257829-extreme-00 |               82 |  0.092 |  0.039 |      0.155 |    0.075 | 0.042 | fg_coll fg_coll fg_coll fg_coll              | bg_coll bg_coll bg_coll bg_coll            | bg_coll bg_coll bg_coll bg_coll         |
| scene-3000_3200-hard-00       |              237 |  0.044 |  0.062 |      0.053 |    0.052 | 0.053 | bg_coll bg_coll bg_coll bg_coll              | fg_coll fg_coll fg_coll fg_coll            | fg_coll fg_coll fg_coll fg_coll         |
| scene-3000_3200-medium-00     |              237 |  0.08  |  0.079 |      0.053 |    0.029 | 0.004 | bg_coll bg_coll bg_coll bg_coll              | fg_coll fg_coll fg_coll fg_coll            | fg_coll fg_coll fg_coll fg_coll         |
| scene-3400_3600-extreme-00    |               80 |  0     |  0     |      0     |    0     | 0     | fg_coll fg_coll fg_coll fg_coll              | fg_coll fg_coll fg_coll fg_coll            | fg_coll fg_coll fg_coll fg_coll         |
| scene-3400_3600-hard-00       |               80 |  0.057 |  0.053 |      0.05  |    0.06  | 0.056 | fg_coll fg_coll fg_coll fg_coll              | fg_coll fg_coll fg_coll fg_coll            | fg_coll fg_coll fg_coll fg_coll         |
| scene-570_770-easy-00         |               96 |  0.211 |  0.205 |      0.205 |    0.208 | 0.202 | off_route bg_coll off_route off_route        | bg_coll off_route off_route off_route      | off_route off_route bg_coll off_route   |
| scene-570_770-medium-00       |               96 |  0.2   |  0.215 |      0.19  |    0.217 | 0.216 | off_route off_route bg_coll bg_coll          | off_route bg_coll off_route off_route      | bg_coll bg_coll off_route bg_coll       |
| scene-5980_6180-easy-00       |              102 |  0.44  |  0.254 |      0.219 |    0.215 | 0.21  | off_route off_route off_route off_route      | bg_coll bg_coll bg_coll bg_coll            | bg_coll off_route bg_coll bg_coll       |
| scene-5980_6180-extreme-00    |              102 |  0.009 |  0.008 |      0.008 |    0.009 | 0.007 | fg_coll fg_coll fg_coll fg_coll              | fg_coll fg_coll fg_coll fg_coll            | fg_coll fg_coll fg_coll fg_coll         |
| scene-770_970-extreme-00      |              107 |  0.024 |  0.022 |      0.02  |    0.025 | 0.025 | fg_coll fg_coll fg_coll fg_coll              | fg_coll fg_coll fg_coll fg_coll            | fg_coll fg_coll fg_coll fg_coll         |
| scene-8440_8640-easy-00       |               87 |  0.594 |  0.712 |      0.102 |    0.64  | 0.677 | bg_coll bg_coll bg_coll bg_coll              | bg_coll bg_coll bg_coll bg_coll            | off_route off_route off_route off_route |
| scene-881121006469-hard-00    |               73 |  0.862 |  0.36  |      0.417 |    0.434 | 0.362 | bg_coll bg_coll bg_coll bg_coll              | fg_coll fg_coll fg_coll fg_coll            | fg_coll fg_coll fg_coll fg_coll         |

## T6 requested curvature (what the lateral path is fed), mean per run then over runs: |kappa| (1/m), mean |kappa_t - kappa_t-1| (1/m per step), sign flips of kappa per step

| turning   | preset   |   k_abs |   k_step |   k_sign_flips |
|:----------|:---------|--------:|---------:|---------------:|
| False     | exam     |  0.0045 |   0.0015 |         0.076  |
| False     | mpc      |  0.002  |   0.0003 |         0.0104 |
| False     | smooth   |  0.0028 |   0.0009 |         0.0594 |
| False     | spec     |  0.0036 |   0.0011 |         0.0673 |
| False     | specplan |  0.0126 |   0.0071 |         0.131  |
| True      | exam     |  0.0113 |   0.003  |         0.0765 |
| True      | mpc      |  0.0089 |   0.0013 |         0.0229 |
| True      | smooth   |  0.0141 |   0.0035 |         0.0501 |
| True      | spec     |  0.0066 |   0.0018 |         0.0686 |
| True      | specplan |  0.022  |   0.009  |         0.1159 |

