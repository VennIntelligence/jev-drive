# JT arena held-split readout (2246 windows)

Paired = tag minus RSG-L18-s0 under the same serve (off: the arena's stored RSG read; leash3 / stop3: ARENA-HOLD's stored RSG + leash3 / ARENA-HOLD 2's RSG + stop3 read) on the common windows without a route-fail in either; 95 % CIs, bootstrap clustered by log (jevdrive.stats). Contribution = the stratum's summed paired difference / all common windows.

| tag @ serve | windows | mean | RSG same serve | paired diff [CI] | zero rate | RSG zero rate | zero-rate diff [CI] |
|---|--:|--:|--:|--:|--:|--:|--:|
| JTN-L18-s1@stop3 | 2228 | 0.9428 | 0.9438 | -0.0010 [-0.0085, +0.0101] | 0.027 | 0.029 | -0.0013 [-0.0123, +0.0058] |
| JTN-L18-s1@off | 2228 | 0.8909 | 0.7857 | +0.1052 [+0.0666, +0.1509] | 0.094 | 0.206 | -0.1113 [-0.1580, -0.0719] |
| JTN-L18-s0@stop3 | 2228 | 0.9426 | 0.9438 | -0.0013 [-0.0087, +0.0096] | 0.027 | 0.029 | -0.0018 [-0.0124, +0.0051] |
| JTN-L18-s0@off | 2228 | 0.8744 | 0.7857 | +0.0887 [+0.0502, +0.1350] | 0.111 | 0.206 | -0.0947 [-0.1424, -0.0555] |

By stratum (RSG and tag: mean / zero rate; new / fixed zeros against RSG under the same serve):

| tag @ serve | stratum | windows | RSG | tag | paired diff [CI] | contribution | new / fixed zeros |
|---|---|--:|--:|--:|--:|--:|--:|
| JTN-L18-s1@stop3 | static | 878 | 0.982 / 0.018 | 0.997 / 0.003 | +0.015 [+0.001, +0.041] | +0.0058 | 0 / 13 |
| JTN-L18-s1@stop3 | launch | 200 | 0.777 / 0.020 | 0.739 / 0.025 | -0.038 [-0.056, -0.022] | -0.0034 | 1 / 0 |
| JTN-L18-s1@stop3 | braking | 372 | 0.960 / 0.032 | 0.943 / 0.051 | -0.016 [-0.034, +0.000] | -0.0027 | 8 / 1 |
| JTN-L18-s1@stop3 | turn | 213 | 0.887 / 0.089 | 0.890 / 0.089 | +0.002 [-0.034, +0.032] | +0.0002 | 6 / 6 |
| JTN-L18-s1@stop3 | cruise | 116 | 0.981 / 0.000 | 0.981 / 0.000 | +0.000 [-0.008, +0.006] | +0.0000 | 0 / 0 |
| JTN-L18-s1@stop3 | other | 449 | 0.948 / 0.029 | 0.943 / 0.033 | -0.005 [-0.018, +0.012] | -0.0009 | 6 / 4 |
| JTN-L18-s1@off | static | 878 | 0.562 / 0.438 | 0.847 / 0.153 | +0.286 [+0.193, +0.382] | +0.1127 | 0 / 251 |
| JTN-L18-s1@off | launch | 200 | 0.937 / 0.045 | 0.856 / 0.055 | -0.080 [-0.113, -0.052] | -0.0072 | 3 / 1 |
| JTN-L18-s1@off | braking | 372 | 0.930 / 0.062 | 0.922 / 0.073 | -0.008 [-0.029, +0.014] | -0.0014 | 8 / 4 |
| JTN-L18-s1@off | turn | 213 | 0.893 / 0.094 | 0.898 / 0.089 | +0.006 [-0.035, +0.041] | +0.0005 | 6 / 7 |
| JTN-L18-s1@off | cruise | 116 | 0.984 / 0.000 | 0.984 / 0.000 | +0.000 [-0.008, +0.005] | +0.0000 | 0 / 0 |
| JTN-L18-s1@off | other | 449 | 0.935 / 0.047 | 0.938 / 0.042 | +0.003 [-0.012, +0.022] | +0.0006 | 5 / 7 |
| JTN-L18-s0@stop3 | static | 878 | 0.982 / 0.018 | 0.997 / 0.003 | +0.015 [+0.001, +0.041] | +0.0058 | 0 / 13 |
| JTN-L18-s0@stop3 | launch | 200 | 0.777 / 0.020 | 0.744 / 0.025 | -0.033 [-0.050, -0.019] | -0.0030 | 1 / 0 |
| JTN-L18-s0@stop3 | braking | 372 | 0.960 / 0.032 | 0.944 / 0.048 | -0.016 [-0.031, -0.004] | -0.0027 | 8 / 2 |
| JTN-L18-s0@stop3 | turn | 213 | 0.887 / 0.089 | 0.891 / 0.085 | +0.004 [-0.032, +0.031] | +0.0003 | 5 / 6 |
| JTN-L18-s0@stop3 | cruise | 116 | 0.981 / 0.000 | 0.978 / 0.000 | -0.004 [-0.013, +0.002] | -0.0002 | 0 / 0 |
| JTN-L18-s0@stop3 | other | 449 | 0.948 / 0.029 | 0.940 / 0.036 | -0.008 [-0.025, +0.010] | -0.0016 | 6 / 3 |
| JTN-L18-s0@off | static | 878 | 0.562 / 0.438 | 0.804 / 0.196 | +0.243 [+0.151, +0.341] | +0.0956 | 0 / 213 |
| JTN-L18-s0@off | launch | 200 | 0.937 / 0.045 | 0.872 / 0.050 | -0.064 [-0.098, -0.037] | -0.0058 | 3 / 2 |
| JTN-L18-s0@off | braking | 372 | 0.930 / 0.062 | 0.925 / 0.067 | -0.005 [-0.024, +0.014] | -0.0009 | 8 / 6 |
| JTN-L18-s0@off | turn | 213 | 0.893 / 0.094 | 0.895 / 0.089 | +0.003 [-0.034, +0.030] | +0.0002 | 5 / 6 |
| JTN-L18-s0@off | cruise | 116 | 0.984 / 0.000 | 0.980 / 0.000 | -0.004 [-0.013, +0.002] | -0.0002 | 0 / 0 |
| JTN-L18-s0@off | other | 449 | 0.935 / 0.047 | 0.934 / 0.047 | -0.001 [-0.020, +0.019] | -0.0003 | 6 / 6 |

Zero types and partial-score types on the common windows:

| read | AF-front | AF-side | offroad | corridor | zero-other | rear-hit | hit | no-launch | creep | slow |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| RSG @ stop3 | 12 | 1 | 43 | 8 | 0 | 23 | 0 | 0 | 31 | 185 |
| JTN-L18-s1@stop3 | 2 | 0 | 52 | 7 | 0 | 23 | 0 | 11 | 31 | 135 |
| RSG @ off | 23 | 1 | 47 | 387 | 0 | 8 | 0 | 0 | 0 | 159 |
| JTN-L18-s1@off | 7 | 0 | 58 | 145 | 0 | 11 | 0 | 10 | 3 | 129 |
| RSG @ stop3 | 12 | 1 | 43 | 8 | 0 | 23 | 0 | 0 | 31 | 185 |
| JTN-L18-s0@stop3 | 2 | 0 | 49 | 9 | 0 | 23 | 0 | 7 | 34 | 156 |
| RSG @ off | 23 | 1 | 47 | 387 | 0 | 8 | 0 | 0 | 0 | 159 |
| JTN-L18-s0@off | 6 | 0 | 54 | 187 | 0 | 10 | 0 | 7 | 4 | 146 |

The same in the braking-stratum windows only:

| read | AF-front | AF-side | offroad | corridor | zero-other | rear-hit | hit | no-launch | creep | slow |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| RSG @ stop3 [braking] | 0 | 0 | 7 | 5 | 0 | 1 | 0 | 0 | 0 | 42 |
| JTN-L18-s1@stop3 [braking] | 2 | 0 | 12 | 5 | 0 | 0 | 0 | 0 | 0 | 22 |
| RSG @ off [braking] | 0 | 0 | 7 | 16 | 0 | 1 | 0 | 0 | 0 | 42 |
| JTN-L18-s1@off [braking] | 2 | 0 | 12 | 13 | 0 | 0 | 0 | 0 | 0 | 22 |
| RSG @ stop3 [braking] | 0 | 0 | 7 | 5 | 0 | 1 | 0 | 0 | 0 | 42 |
| JTN-L18-s0@stop3 [braking] | 2 | 0 | 10 | 6 | 0 | 0 | 0 | 0 | 0 | 33 |
| RSG @ off [braking] | 0 | 0 | 7 | 16 | 0 | 1 | 0 | 0 | 0 | 42 |
| JTN-L18-s0@off [braking] | 2 | 0 | 10 | 13 | 0 | 0 | 0 | 0 | 0 | 33 |
