# JT arena held-split readout (2246 windows)

Paired = tag minus RSG-L18-s0 under the same serve (off: the arena's stored RSG read; leash3 / stop3: ARENA-HOLD's stored RSG + leash3 / ARENA-HOLD 2's RSG + stop3 read) on the common windows without a route-fail in either; 95 % CIs, bootstrap clustered by log (jevdrive.stats). Contribution = the stratum's summed paired difference / all common windows.

| tag @ serve | windows | mean | RSG same serve | paired diff [CI] | zero rate | RSG zero rate | zero-rate diff [CI] |
|---|--:|--:|--:|--:|--:|--:|--:|
| JTB-L18-s0@stop3 | 2228 | 0.9409 | 0.9438 | -0.0029 [-0.0086, +0.0035] | 0.037 | 0.029 | +0.0081 [+0.0033, +0.0125] |
| JTB-L18-s0@off | 2228 | 0.8216 | 0.7857 | +0.0359 [+0.0123, +0.0625] | 0.174 | 0.206 | -0.0319 [-0.0596, -0.0078] |

By stratum (RSG and tag: mean / zero rate; new / fixed zeros against RSG under the same serve):

| tag @ serve | stratum | windows | RSG | tag | paired diff [CI] | contribution | new / fixed zeros |
|---|---|--:|--:|--:|--:|--:|--:|
| JTB-L18-s0@stop3 | static | 878 | 0.982 / 0.018 | 0.982 / 0.018 | +0.000 [-0.006, +0.005] | +0.0000 | 2 / 2 |
| JTB-L18-s0@stop3 | launch | 200 | 0.777 / 0.020 | 0.753 / 0.030 | -0.024 [-0.043, -0.008] | -0.0022 | 2 / 0 |
| JTB-L18-s0@stop3 | braking | 372 | 0.960 / 0.032 | 0.932 / 0.067 | -0.028 [-0.050, -0.006] | -0.0046 | 14 / 1 |
| JTB-L18-s0@stop3 | turn | 213 | 0.887 / 0.089 | 0.895 / 0.094 | +0.007 [-0.035, +0.043] | +0.0007 | 7 / 6 |
| JTB-L18-s0@stop3 | cruise | 116 | 0.981 / 0.000 | 0.998 / 0.000 | +0.016 [+0.005, +0.031] | +0.0009 | 0 / 0 |
| JTB-L18-s0@stop3 | other | 449 | 0.948 / 0.029 | 0.959 / 0.033 | +0.012 [-0.003, +0.029] | +0.0024 | 5 / 3 |
| JTB-L18-s0@off | static | 878 | 0.562 / 0.438 | 0.667 / 0.333 | +0.106 [+0.045, +0.169] | +0.0417 | 8 / 101 |
| JTB-L18-s0@off | launch | 200 | 0.937 / 0.045 | 0.899 / 0.055 | -0.038 [-0.065, -0.015] | -0.0034 | 3 / 1 |
| JTB-L18-s0@off | braking | 372 | 0.930 / 0.062 | 0.889 / 0.110 | -0.041 [-0.063, -0.018] | -0.0069 | 20 / 2 |
| JTB-L18-s0@off | turn | 213 | 0.893 / 0.094 | 0.900 / 0.099 | +0.007 [-0.035, +0.043] | +0.0007 | 7 / 6 |
| JTB-L18-s0@off | cruise | 116 | 0.984 / 0.000 | 1.000 / 0.000 | +0.016 [+0.005, +0.031] | +0.0009 | 0 / 0 |
| JTB-L18-s0@off | other | 449 | 0.935 / 0.047 | 0.950 / 0.049 | +0.014 [-0.001, +0.033] | +0.0029 | 5 / 4 |

Zero types and partial-score types on the common windows:

| read | AF-front | AF-side | offroad | corridor | zero-other | rear-hit | hit | no-launch | creep | slow |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| RSG @ stop3 | 12 | 1 | 43 | 8 | 0 | 23 | 0 | 0 | 31 | 185 |
| JTB-L18-s0@stop3 | 14 | 1 | 52 | 15 | 0 | 15 | 0 | 4 | 33 | 52 |
| RSG @ off | 23 | 1 | 47 | 387 | 0 | 8 | 0 | 0 | 0 | 159 |
| JTB-L18-s0@off | 23 | 1 | 57 | 306 | 0 | 1 | 0 | 3 | 0 | 37 |

The same in the braking-stratum windows only:

| read | AF-front | AF-side | offroad | corridor | zero-other | rear-hit | hit | no-launch | creep | slow |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| RSG @ stop3 [braking] | 0 | 0 | 7 | 5 | 0 | 1 | 0 | 0 | 0 | 42 |
| JTB-L18-s0@stop3 [braking] | 3 | 1 | 11 | 10 | 0 | 0 | 0 | 0 | 0 | 2 |
| RSG @ off [braking] | 0 | 0 | 7 | 16 | 0 | 1 | 0 | 0 | 0 | 42 |
| JTB-L18-s0@off [braking] | 4 | 1 | 11 | 25 | 0 | 0 | 0 | 0 | 0 | 2 |
