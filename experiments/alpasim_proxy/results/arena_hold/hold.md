# ARENA-HOLD readout: RSG-L18-s0 with JEV_HOLD on the arena's held-split windows

Held windows 2246; stored RSG read on 2246; 18 dropped where a read failed before driving (route-fail). Paired = variant minus the stored RSG read on the common windows; CIs 95 %, bootstrap clustered by log (jevdrive.stats).

| variant | windows | mean score | RSG same windows | paired diff [CI] | zero rate | RSG zero rate | zero-rate diff [CI] | gain (sum) | loss (sum) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| full | 2228 | 0.9224 | 0.7857 | +0.1367 [+0.0821, +0.1955] | 0.035 | 0.206 | -0.1710 [-0.2281, -0.1156] | +377.1 | -72.5 |
| lead | 2228 | 0.9229 | 0.7857 | +0.1371 [+0.0822, +0.1956] | 0.035 | 0.206 | -0.1706 [-0.2279, -0.1151] | +377.1 | -71.6 |
| leash3 | 2228 | 0.9343 | 0.7857 | +0.1486 [+0.0925, +0.2078] | 0.040 | 0.206 | -0.1652 [-0.2236, -0.1092] | +366.0 | -34.9 |

By stratum (RSG and variant: mean score / zero rate; contribution = the stratum's summed paired difference / all held windows, so the strata add up to the overall paired difference; new zeros / fixed zeros):

| variant | stratum | windows | RSG | variant | paired diff [CI] | contribution | gain | loss | new / fixed zeros |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| full | static | 878 | 0.562 / 0.438 | 0.990 / 0.010 | +0.428 [+0.296, +0.558] | +0.1688 | +376.0 | +0.0 | 0 / 376 |
| full | launch | 200 | 0.937 / 0.045 | 0.580 / 0.020 | -0.357 [-0.413, -0.296] | -0.0320 | +1.1 | -72.5 | 0 / 5 |
| full | braking | 372 | 0.930 / 0.062 | 0.930 / 0.062 | -0.000 [-0.000, -0.000] | -0.0000 | +0.0 | -0.0 | 0 / 0 |
| full | turn | 213 | 0.893 / 0.094 | 0.893 / 0.094 | -0.000 [-0.000, +0.000] | -0.0000 | +0.0 | -0.0 | 0 / 0 |
| full | cruise | 116 | 0.984 / 0.000 | 0.984 / 0.000 | -0.000 [-0.000, +0.000] | -0.0000 | +0.0 | -0.0 | 0 / 0 |
| full | other | 449 | 0.935 / 0.047 | 0.935 / 0.047 | +0.000 [-0.000, +0.000] | +0.0000 | +0.0 | -0.0 | 0 / 0 |
| lead | static | 878 | 0.562 / 0.438 | 0.990 / 0.010 | +0.428 [+0.296, +0.558] | +0.1688 | +376.0 | +0.0 | 0 / 376 |
| lead | launch | 200 | 0.937 / 0.045 | 0.584 / 0.025 | -0.352 [-0.405, -0.295] | -0.0316 | +1.1 | -71.6 | 1 / 5 |
| lead | braking | 372 | 0.930 / 0.062 | 0.930 / 0.062 | -0.000 [-0.000, +0.000] | -0.0000 | +0.0 | -0.0 | 0 / 0 |
| lead | turn | 213 | 0.893 / 0.094 | 0.893 / 0.094 | -0.000 [-0.000, +0.000] | -0.0000 | +0.0 | -0.0 | 0 / 0 |
| lead | cruise | 116 | 0.984 / 0.000 | 0.984 / 0.000 | -0.000 [-0.000, +0.000] | -0.0000 | +0.0 | -0.0 | 0 / 0 |
| lead | other | 449 | 0.935 / 0.047 | 0.935 / 0.047 | +0.000 [-0.000, +0.000] | +0.0000 | +0.0 | -0.0 | 0 / 0 |
| leash3 | static | 878 | 0.562 / 0.438 | 0.975 / 0.025 | +0.413 [+0.276, +0.546] | +0.1629 | +363.0 | +0.0 | 0 / 363 |
| leash3 | launch | 200 | 0.937 / 0.045 | 0.777 / 0.020 | -0.159 [-0.193, -0.126] | -0.0143 | +3.0 | -34.9 | 0 / 5 |
| leash3 | braking | 372 | 0.930 / 0.062 | 0.930 / 0.062 | -0.000 [-0.000, -0.000] | -0.0000 | +0.0 | -0.0 | 0 / 0 |
| leash3 | turn | 213 | 0.893 / 0.094 | 0.893 / 0.094 | +0.000 [-0.000, +0.000] | +0.0000 | +0.0 | -0.0 | 0 / 0 |
| leash3 | cruise | 116 | 0.984 / 0.000 | 0.984 / 0.000 | +0.000 [+0.000, +0.000] | +0.0000 | +0.0 | +0.0 | 0 / 0 |
| leash3 | other | 449 | 0.935 / 0.047 | 0.935 / 0.047 | +0.000 [-0.000, +0.000] | +0.0000 | +0.0 | -0.0 | 0 / 0 |

Zero types on the held windows (AF-front / AF-side / offroad / corridor / zero-other) and failure types of partial scores:

| read | AF-front | AF-side | offroad | corridor | zero-other | rear-hit | hit | no-launch | creep | slow |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| RSG (stored) | 23 | 1 | 47 | 387 | 0 | 8 | 0 | 0 | 0 | 159 |
| full | 1 | 1 | 42 | 33 | 0 | 35 | 0 | 54 | 0 | 141 |
| lead | 1 | 1 | 43 | 33 | 0 | 33 | 0 | 54 | 0 | 140 |
| leash3 | 12 | 1 | 44 | 33 | 0 | 23 | 0 | 0 | 30 | 175 |

- full: 2228 sessions past the handoff, 1004 latched held; 9036 decisions cut; releases {}
- lead: 2228 sessions past the handoff, 1003 latched held; 9019 decisions cut; releases {'lead_moved': 2}
- leash3: 2228 sessions past the handoff, 1004 latched held; 7250 decisions cut; releases {}
