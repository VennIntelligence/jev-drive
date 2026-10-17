# JT arena held-split readout (2246 windows)

Paired = tag minus RSG-L18-s0 under the same serve (off: the arena's stored RSG read; leash3: ARENA-HOLD's stored RSG + leash3 read) on the common windows without a route-fail in either; 95 % CIs, bootstrap clustered by log (jevdrive.stats). Contribution = the stratum's summed paired difference / all common windows.

| tag @ serve | windows | mean | RSG same serve | paired diff [CI] | zero rate | RSG zero rate | zero-rate diff [CI] |
|---|--:|--:|--:|--:|--:|--:|--:|
| JT10-L18-s0@leash3 | 2228 | 0.9049 | 0.9343 | -0.0294 [-0.0517, -0.0122] | 0.076 | 0.040 | +0.0359 [+0.0178, +0.0600] |
| JT10-L18-s0@off | 2228 | 0.7602 | 0.7857 | -0.0255 [-0.0507, -0.0034] | 0.238 | 0.206 | +0.0323 [+0.0093, +0.0589] |

By stratum (RSG and tag: mean / zero rate; new / fixed zeros against RSG under the same serve):

| tag @ serve | stratum | windows | RSG | tag | paired diff [CI] | contribution | new / fixed zeros |
|---|---|--:|--:|--:|--:|--:|--:|
| JT10-L18-s0@leash3 | static | 878 | 0.975 / 0.025 | 0.936 / 0.064 | -0.039 [-0.102, -0.001] | -0.0153 | 34 / 0 |
| JT10-L18-s0@leash3 | launch | 200 | 0.777 / 0.020 | 0.767 / 0.030 | -0.010 [-0.024, -0.000] | -0.0009 | 2 / 0 |
| JT10-L18-s0@leash3 | braking | 372 | 0.930 / 0.062 | 0.855 / 0.145 | -0.076 [-0.111, -0.041] | -0.0126 | 32 / 1 |
| JT10-L18-s0@leash3 | turn | 213 | 0.893 / 0.094 | 0.868 / 0.131 | -0.025 [-0.064, +0.006] | -0.0023 | 11 / 3 |
| JT10-L18-s0@leash3 | cruise | 116 | 0.984 / 0.000 | 0.991 / 0.009 | +0.008 [-0.016, +0.029] | +0.0004 | 1 / 0 |
| JT10-L18-s0@leash3 | other | 449 | 0.935 / 0.047 | 0.942 / 0.056 | +0.007 [-0.010, +0.027] | +0.0013 | 9 / 5 |
| JT10-L18-s0@off | static | 878 | 0.562 / 0.438 | 0.533 / 0.467 | -0.028 [-0.099, +0.021] | -0.0112 | 47 / 22 |
| JT10-L18-s0@off | launch | 200 | 0.937 / 0.045 | 0.925 / 0.060 | -0.012 [-0.045, +0.024] | -0.0011 | 5 / 2 |
| JT10-L18-s0@off | braking | 372 | 0.930 / 0.062 | 0.855 / 0.145 | -0.076 [-0.111, -0.041] | -0.0126 | 32 / 1 |
| JT10-L18-s0@off | turn | 213 | 0.893 / 0.094 | 0.868 / 0.131 | -0.025 [-0.064, +0.006] | -0.0023 | 11 / 3 |
| JT10-L18-s0@off | cruise | 116 | 0.984 / 0.000 | 0.991 / 0.009 | +0.008 [-0.016, +0.029] | +0.0004 | 1 / 0 |
| JT10-L18-s0@off | other | 449 | 0.935 / 0.047 | 0.942 / 0.056 | +0.007 [-0.010, +0.027] | +0.0013 | 9 / 5 |

Zero types and partial-score types on the common windows:

| read | AF-front | AF-side | offroad | corridor | zero-other | rear-hit | hit | no-launch | creep | slow |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| RSG @ leash3 | 12 | 1 | 44 | 33 | 0 | 23 | 0 | 0 | 30 | 175 |
| JT10-L18-s0@leash3 | 51 | 2 | 65 | 52 | 0 | 15 | 0 | 0 | 30 | 45 |
| RSG @ off | 23 | 1 | 47 | 387 | 0 | 8 | 0 | 0 | 0 | 159 |
| JT10-L18-s0@off | 71 | 2 | 68 | 389 | 0 | 0 | 0 | 0 | 0 | 27 |
