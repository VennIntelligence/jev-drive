# JT arena held-split readout (2246 windows)

Paired = tag minus RSG-L18-s0 under the same serve (off: the arena's stored RSG read; leash3: ARENA-HOLD's stored RSG + leash3 read) on the common windows without a route-fail in either; 95 % CIs, bootstrap clustered by log (jevdrive.stats). Contribution = the stratum's summed paired difference / all common windows.

| tag @ serve | windows | mean | RSG same serve | paired diff [CI] | zero rate | RSG zero rate | zero-rate diff [CI] |
|---|--:|--:|--:|--:|--:|--:|--:|
| JT00-L18-s0@leash3 | 2228 | 0.9037 | 0.9343 | -0.0307 [-0.0532, -0.0133] | 0.077 | 0.040 | +0.0368 [+0.0189, +0.0610] |
| JT00-L18-s0@off | 2228 | 0.7516 | 0.7857 | -0.0341 [-0.0583, -0.0144] | 0.246 | 0.206 | +0.0404 [+0.0200, +0.0661] |

By stratum (RSG and tag: mean / zero rate; new / fixed zeros against RSG under the same serve):

| tag @ serve | stratum | windows | RSG | tag | paired diff [CI] | contribution | new / fixed zeros |
|---|---|--:|--:|--:|--:|--:|--:|
| JT00-L18-s0@leash3 | static | 878 | 0.975 / 0.025 | 0.935 / 0.065 | -0.040 [-0.104, -0.001] | -0.0157 | 35 / 0 |
| JT00-L18-s0@leash3 | launch | 200 | 0.777 / 0.020 | 0.766 / 0.025 | -0.012 [-0.025, -0.002] | -0.0010 | 1 / 0 |
| JT00-L18-s0@leash3 | braking | 372 | 0.930 / 0.062 | 0.849 / 0.151 | -0.081 [-0.116, -0.046] | -0.0135 | 34 / 1 |
| JT00-L18-s0@leash3 | turn | 213 | 0.893 / 0.094 | 0.864 / 0.136 | -0.029 [-0.075, +0.005] | -0.0028 | 13 / 4 |
| JT00-L18-s0@leash3 | cruise | 116 | 0.984 / 0.000 | 0.991 / 0.009 | +0.008 [-0.016, +0.029] | +0.0004 | 1 / 0 |
| JT00-L18-s0@leash3 | other | 449 | 0.935 / 0.047 | 0.945 / 0.053 | +0.010 [-0.005, +0.028] | +0.0019 | 7 / 4 |
| JT00-L18-s0@off | static | 878 | 0.562 / 0.438 | 0.514 / 0.486 | -0.048 [-0.118, +0.000] | -0.0189 | 59 / 17 |
| JT00-L18-s0@off | launch | 200 | 0.937 / 0.045 | 0.922 / 0.055 | -0.015 [-0.048, +0.020] | -0.0014 | 4 / 2 |
| JT00-L18-s0@off | braking | 372 | 0.930 / 0.062 | 0.849 / 0.151 | -0.081 [-0.116, -0.046] | -0.0135 | 34 / 1 |
| JT00-L18-s0@off | turn | 213 | 0.893 / 0.094 | 0.864 / 0.136 | -0.029 [-0.075, +0.005] | -0.0028 | 13 / 4 |
| JT00-L18-s0@off | cruise | 116 | 0.984 / 0.000 | 0.991 / 0.009 | +0.008 [-0.016, +0.029] | +0.0004 | 1 / 0 |
| JT00-L18-s0@off | other | 449 | 0.935 / 0.047 | 0.945 / 0.053 | +0.010 [-0.005, +0.028] | +0.0019 | 7 / 4 |

Zero types and partial-score types on the common windows:

| read | AF-front | AF-side | offroad | corridor | zero-other | rear-hit | hit | no-launch | creep | slow |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| RSG @ leash3 | 12 | 1 | 44 | 33 | 0 | 23 | 0 | 0 | 30 | 175 |
| JT00-L18-s0@leash3 | 52 | 1 | 65 | 54 | 0 | 15 | 0 | 3 | 31 | 36 |
| RSG @ off | 23 | 1 | 47 | 387 | 0 | 8 | 0 | 0 | 0 | 159 |
| JT00-L18-s0@off | 77 | 1 | 68 | 402 | 0 | 0 | 0 | 3 | 0 | 17 |
