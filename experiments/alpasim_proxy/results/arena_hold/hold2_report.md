# ARENA-HOLD readout: RSG-L18-s0 with JEV_HOLD on the arena's held-split windows

Held windows 2246; stored RSG read on 2246; 18 dropped where a read failed before driving (route-fail). Paired = variant minus the stored RSG read on the common windows; CIs 95 %, bootstrap clustered by log (jevdrive.stats).

| variant | windows | mean score | RSG same windows | paired diff [CI] | zero rate | RSG zero rate | zero-rate diff [CI] | gain (sum) | loss (sum) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| leash3 | 2228 | 0.9343 | 0.7857 | +0.1486 [+0.0925, +0.2078] | 0.040 | 0.206 | -0.1652 [-0.2236, -0.1092] | +366.0 | -34.9 |
| stop3 | 2228 | 0.9438 | 0.7857 | +0.1581 [+0.1012, +0.2183] | 0.029 | 0.206 | -0.1768 [-0.2358, -0.1195] | +391.7 | -39.5 |

By stratum (RSG and variant: mean score / zero rate; contribution = the stratum's summed paired difference / all held windows, so the strata add up to the overall paired difference; new zeros / fixed zeros):

| variant | stratum | windows | RSG | variant | paired diff [CI] | contribution | gain | loss | new / fixed zeros |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| leash3 | static | 878 | 0.562 / 0.438 | 0.975 / 0.025 | +0.413 [+0.276, +0.546] | +0.1629 | +363.0 | +0.0 | 0 / 363 |
| leash3 | launch | 200 | 0.937 / 0.045 | 0.777 / 0.020 | -0.159 [-0.193, -0.126] | -0.0143 | +3.0 | -34.9 | 0 / 5 |
| leash3 | braking | 372 | 0.930 / 0.062 | 0.930 / 0.062 | -0.000 [-0.000, -0.000] | -0.0000 | +0.0 | -0.0 | 0 / 0 |
| leash3 | turn | 213 | 0.893 / 0.094 | 0.893 / 0.094 | +0.000 [-0.000, +0.000] | +0.0000 | +0.0 | -0.0 | 0 / 0 |
| leash3 | cruise | 116 | 0.984 / 0.000 | 0.984 / 0.000 | +0.000 [+0.000, +0.000] | +0.0000 | +0.0 | +0.0 | 0 / 0 |
| leash3 | other | 449 | 0.935 / 0.047 | 0.935 / 0.047 | +0.000 [-0.000, +0.000] | +0.0000 | +0.0 | -0.0 | 0 / 0 |
| stop3 | static | 878 | 0.562 / 0.438 | 0.982 / 0.018 | +0.420 [+0.283, +0.553] | +0.1656 | +369.0 | +0.0 | 0 / 369 |
| stop3 | launch | 200 | 0.937 / 0.045 | 0.777 / 0.020 | -0.159 [-0.193, -0.126] | -0.0143 | +3.0 | -34.9 | 0 / 5 |
| stop3 | braking | 372 | 0.930 / 0.062 | 0.960 / 0.032 | +0.029 [+0.009, +0.055] | +0.0049 | +11.0 | -0.0 | 0 / 11 |
| stop3 | turn | 213 | 0.893 / 0.094 | 0.887 / 0.089 | -0.005 [-0.018, +0.005] | -0.0005 | +0.7 | -1.9 | 0 / 1 |
| stop3 | cruise | 116 | 0.984 / 0.000 | 0.981 / 0.000 | -0.002 [-0.008, +0.000] | -0.0001 | +0.0 | -0.3 | 0 / 0 |
| stop3 | other | 449 | 0.935 / 0.047 | 0.948 / 0.029 | +0.012 [+0.002, +0.026] | +0.0025 | +8.0 | -2.4 | 0 / 8 |

Paired against leash3 (the same pairing with leash3's read in place of RSG's):

| variant | stratum | windows | leash3 | variant | paired diff [CI] | contribution | gain | loss | new / fixed zeros |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| stop3 | all | 2228 | 0.9343 / 0.040 | 0.9438 / 0.029 | +0.0095 [+0.0048, +0.0146] | +0.0095 | +25.7 | -4.6 | 0 / 26 |
| stop3 | static | 878 | 0.9749 / 0.025 | 0.9818 / 0.018 | +0.0068 [+0.0011, +0.0151] | +0.0027 | +6.0 | +0.0 | 0 / 6 |
| stop3 | launch | 200 | 0.7773 / 0.020 | 0.7773 / 0.020 | -0.0000 [-0.0000, +0.0000] | -0.0000 | +0.0 | -0.0 | 0 / 0 |
| stop3 | braking | 372 | 0.9303 / 0.062 | 0.9598 / 0.032 | +0.0295 [+0.0094, +0.0549] | +0.0049 | +11.0 | -0.0 | 0 / 11 |
| stop3 | turn | 213 | 0.8928 / 0.094 | 0.8874 / 0.089 | -0.0054 [-0.0185, +0.0054] | -0.0005 | +0.7 | -1.9 | 0 / 1 |
| stop3 | cruise | 116 | 0.9836 / 0.000 | 0.9813 / 0.000 | -0.0024 [-0.0080, +0.0000] | -0.0001 | +0.0 | -0.3 | 0 / 0 |
| stop3 | other | 449 | 0.9351 / 0.047 | 0.9476 / 0.029 | +0.0124 [+0.0023, +0.0258] | +0.0025 | +8.0 | -2.4 | 0 / 8 |

Zero types on the held windows (AF-front / AF-side / offroad / corridor / zero-other) and failure types of partial scores:

| read | AF-front | AF-side | offroad | corridor | zero-other | rear-hit | hit | no-launch | creep | slow |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| RSG (stored) | 23 | 1 | 47 | 387 | 0 | 8 | 0 | 0 | 0 | 159 |
| leash3 | 12 | 1 | 44 | 33 | 0 | 23 | 0 | 0 | 30 | 175 |
| stop3 | 12 | 1 | 43 | 8 | 0 | 23 | 0 | 0 | 31 | 185 |

- leash3: 2228 sessions past the handoff, 1004 latched held (0 by the stop rule); 7250 decisions cut; releases {}
- stop3: 2228 sessions past the handoff, 1146 latched held (142 by the stop rule); 8097 decisions cut; releases {}
