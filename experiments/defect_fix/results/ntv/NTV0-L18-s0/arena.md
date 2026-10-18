# JT arena held-split readout (2246 windows)

Paired = tag minus RSG-L18-s0 under the same serve (off: the arena's stored RSG read; leash3 / stop3: ARENA-HOLD's stored RSG + leash3 / ARENA-HOLD 2's RSG + stop3 read) on the common windows without a route-fail in either; 95 % CIs, bootstrap clustered by log (jevdrive.stats). Contribution = the stratum's summed paired difference / all common windows.

| tag @ serve | windows | mean | RSG same serve | paired diff [CI] | zero rate | RSG zero rate | zero-rate diff [CI] |
|---|--:|--:|--:|--:|--:|--:|--:|
| NTV0-L18-s0@stop3 | 2228 | 0.9357 | 0.9438 | -0.0081 [-0.0149, -0.0022] | 0.037 | 0.029 | +0.0085 [+0.0028, +0.0151] |
| NTV0-L18-s0@off | 2228 | 0.7699 | 0.7857 | -0.0158 [-0.0250, -0.0077] | 0.222 | 0.206 | +0.0162 [+0.0084, +0.0252] |

By stratum (RSG and tag: mean / zero rate; new / fixed zeros against RSG under the same serve):

| tag @ serve | stratum | windows | RSG | tag | paired diff [CI] | contribution | new / fixed zeros |
|---|---|--:|--:|--:|--:|--:|--:|
| NTV0-L18-s0@stop3 | static | 878 | 0.982 / 0.018 | 0.976 / 0.024 | -0.006 [-0.020, +0.000] | -0.0022 | 5 / 0 |
| NTV0-L18-s0@stop3 | launch | 200 | 0.777 / 0.020 | 0.767 / 0.030 | -0.011 [-0.026, +0.000] | -0.0010 | 2 / 0 |
| NTV0-L18-s0@stop3 | braking | 372 | 0.960 / 0.032 | 0.949 / 0.046 | -0.011 [-0.026, +0.000] | -0.0019 | 6 / 1 |
| NTV0-L18-s0@stop3 | turn | 213 | 0.887 / 0.089 | 0.869 / 0.108 | -0.018 [-0.065, +0.017] | -0.0018 | 9 / 5 |
| NTV0-L18-s0@stop3 | cruise | 116 | 0.981 / 0.000 | 0.981 / 0.000 | -0.000 [-0.004, +0.002] | -0.0000 | 0 / 0 |
| NTV0-L18-s0@stop3 | other | 449 | 0.948 / 0.029 | 0.941 / 0.036 | -0.006 [-0.023, +0.011] | -0.0013 | 5 / 2 |
| NTV0-L18-s0@off | static | 878 | 0.562 / 0.438 | 0.541 / 0.459 | -0.021 [-0.043, -0.004] | -0.0081 | 19 / 1 |
| NTV0-L18-s0@off | launch | 200 | 0.937 / 0.045 | 0.922 / 0.060 | -0.015 [-0.037, +0.008] | -0.0013 | 4 / 1 |
| NTV0-L18-s0@off | braking | 372 | 0.930 / 0.062 | 0.914 / 0.081 | -0.017 [-0.037, -0.001] | -0.0028 | 8 / 1 |
| NTV0-L18-s0@off | turn | 213 | 0.893 / 0.094 | 0.870 / 0.117 | -0.023 [-0.070, +0.014] | -0.0022 | 10 / 5 |
| NTV0-L18-s0@off | cruise | 116 | 0.984 / 0.000 | 0.983 / 0.000 | -0.000 [-0.004, +0.002] | -0.0000 | 0 / 0 |
| NTV0-L18-s0@off | other | 449 | 0.935 / 0.047 | 0.928 / 0.053 | -0.007 [-0.023, +0.010] | -0.0014 | 5 / 2 |

Zero types and partial-score types on the common windows:

| read | AF-front | AF-side | offroad | corridor | zero-other | rear-hit | hit | no-launch | creep | slow |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| RSG @ stop3 | 12 | 1 | 43 | 8 | 0 | 23 | 0 | 0 | 31 | 185 |
| NTV0-L18-s0@stop3 | 18 | 0 | 55 | 10 | 0 | 24 | 0 | 0 | 32 | 156 |
| RSG @ off | 23 | 1 | 47 | 387 | 0 | 8 | 0 | 0 | 0 | 159 |
| NTV0-L18-s0@off | 37 | 0 | 61 | 396 | 0 | 9 | 0 | 0 | 0 | 131 |

The same in the braking-stratum windows only:

| read | AF-front | AF-side | offroad | corridor | zero-other | rear-hit | hit | no-launch | creep | slow |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| RSG @ stop3 [braking] | 0 | 0 | 7 | 5 | 0 | 1 | 0 | 0 | 0 | 42 |
| NTV0-L18-s0@stop3 [braking] | 2 | 0 | 10 | 5 | 0 | 1 | 0 | 0 | 0 | 26 |
| RSG @ off [braking] | 0 | 0 | 7 | 16 | 0 | 1 | 0 | 0 | 0 | 42 |
| NTV0-L18-s0@off [braking] | 2 | 0 | 10 | 18 | 0 | 1 | 0 | 0 | 0 | 25 |
