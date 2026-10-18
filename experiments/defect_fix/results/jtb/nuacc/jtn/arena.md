# JT arena held-split readout (2246 windows)

Paired = tag minus RSG-L18-s0 under the same serve (off: the arena's stored RSG read; leash3 / stop3: ARENA-HOLD's stored RSG + leash3 / ARENA-HOLD 2's RSG + stop3 read) on the common windows without a route-fail in either; 95 % CIs, bootstrap clustered by log (jevdrive.stats). Contribution = the stratum's summed paired difference / all common windows.

| tag @ serve | windows | mean | RSG same serve | paired diff [CI] | zero rate | RSG zero rate | zero-rate diff [CI] |
|---|--:|--:|--:|--:|--:|--:|--:|
| JTN-L18-s0@stop3 | 2228 | 0.9246 | 0.9438 | -0.0192 [-0.0292, -0.0068] | 0.044 | 0.029 | +0.0148 [+0.0024, +0.0251] |

By stratum (RSG and tag: mean / zero rate; new / fixed zeros against RSG under the same serve):

| tag @ serve | stratum | windows | RSG | tag | paired diff [CI] | contribution | new / fixed zeros |
|---|---|--:|--:|--:|--:|--:|--:|
| JTN-L18-s0@stop3 | static | 878 | 0.982 / 0.018 | 0.995 / 0.005 | +0.014 [+0.001, +0.037] | +0.0054 | 0 / 12 |
| JTN-L18-s0@stop3 | launch | 200 | 0.777 / 0.020 | 0.685 / 0.035 | -0.092 [-0.118, -0.070] | -0.0083 | 3 / 0 |
| JTN-L18-s0@stop3 | braking | 372 | 0.960 / 0.032 | 0.904 / 0.094 | -0.056 [-0.083, -0.030] | -0.0093 | 24 / 1 |
| JTN-L18-s0@stop3 | turn | 213 | 0.887 / 0.089 | 0.823 / 0.164 | -0.064 [-0.103, -0.036] | -0.0061 | 17 / 1 |
| JTN-L18-s0@stop3 | cruise | 116 | 0.981 / 0.000 | 0.993 / 0.000 | +0.012 [-0.001, +0.027] | +0.0006 | 0 / 0 |
| JTN-L18-s0@stop3 | other | 449 | 0.948 / 0.029 | 0.940 / 0.036 | -0.007 [-0.028, +0.013] | -0.0014 | 7 / 4 |

Zero types and partial-score types on the common windows:

| read | AF-front | AF-side | offroad | corridor | zero-other | rear-hit | hit | no-launch | creep | slow |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| RSG @ stop3 | 12 | 1 | 43 | 8 | 0 | 23 | 0 | 0 | 31 | 185 |
| JTN-L18-s0@stop3 | 6 | 1 | 65 | 25 | 0 | 28 | 0 | 16 | 37 | 89 |

The same in the braking-stratum windows only:

| read | AF-front | AF-side | offroad | corridor | zero-other | rear-hit | hit | no-launch | creep | slow |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| RSG @ stop3 [braking] | 0 | 0 | 7 | 5 | 0 | 1 | 0 | 0 | 0 | 42 |
| JTN-L18-s0@stop3 [braking] | 2 | 0 | 13 | 20 | 0 | 1 | 0 | 0 | 0 | 8 |
