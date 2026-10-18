# JT arena held-split readout (2246 windows)

Paired = tag minus RSG-L18-s0 under the same serve (off: the arena's stored RSG read; leash3 / stop3: ARENA-HOLD's stored RSG + leash3 / ARENA-HOLD 2's RSG + stop3 read) on the common windows without a route-fail in either; 95 % CIs, bootstrap clustered by log (jevdrive.stats). Contribution = the stratum's summed paired difference / all common windows.

| tag @ serve | windows | mean | RSG same serve | paired diff [CI] | zero rate | RSG zero rate | zero-rate diff [CI] |
|---|--:|--:|--:|--:|--:|--:|--:|
| RSG-L18-s0@stop3 | 2228 | 0.9260 | 0.9438 | -0.0179 [-0.0265, -0.0101] | 0.045 | 0.029 | +0.0162 [+0.0082, +0.0248] |

By stratum (RSG and tag: mean / zero rate; new / fixed zeros against RSG under the same serve):

| tag @ serve | stratum | windows | RSG | tag | paired diff [CI] | contribution | new / fixed zeros |
|---|---|--:|--:|--:|--:|--:|--:|
| RSG-L18-s0@stop3 | static | 878 | 0.982 / 0.018 | 0.982 / 0.018 | +0.000 [+0.000, +0.000] | +0.0000 | 0 / 0 |
| RSG-L18-s0@stop3 | launch | 200 | 0.777 / 0.020 | 0.709 / 0.040 | -0.068 [-0.090, -0.047] | -0.0061 | 4 / 0 |
| RSG-L18-s0@stop3 | braking | 372 | 0.960 / 0.032 | 0.918 / 0.081 | -0.042 [-0.073, -0.016] | -0.0070 | 18 / 0 |
| RSG-L18-s0@stop3 | turn | 213 | 0.887 / 0.089 | 0.832 / 0.155 | -0.055 [-0.102, -0.019] | -0.0053 | 15 / 1 |
| RSG-L18-s0@stop3 | cruise | 116 | 0.981 / 0.000 | 0.993 / 0.000 | +0.011 [-0.001, +0.027] | +0.0006 | 0 / 0 |
| RSG-L18-s0@stop3 | other | 449 | 0.948 / 0.029 | 0.947 / 0.029 | -0.000 [-0.015, +0.016] | -0.0001 | 3 / 3 |

Zero types and partial-score types on the common windows:

| read | AF-front | AF-side | offroad | corridor | zero-other | rear-hit | hit | no-launch | creep | slow |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| RSG @ stop3 | 12 | 1 | 43 | 8 | 0 | 23 | 0 | 0 | 31 | 185 |
| RSG-L18-s0@stop3 | 16 | 0 | 59 | 25 | 0 | 30 | 0 | 8 | 36 | 94 |

The same in the braking-stratum windows only:

| read | AF-front | AF-side | offroad | corridor | zero-other | rear-hit | hit | no-launch | creep | slow |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| RSG @ stop3 [braking] | 0 | 0 | 7 | 5 | 0 | 1 | 0 | 0 | 0 | 42 |
| RSG-L18-s0@stop3 [braking] | 1 | 0 | 8 | 21 | 0 | 1 | 0 | 0 | 0 | 7 |
