events 122 in 119 segments

yaw-rate sign check: corr(phi1_08, yaw_rate_left+) over 3 s after onset = nan

## launch lean |phi1| (deg), steps m = 1..4 at onset + 5(m-1) frames; median [cluster CI], mean, share >= 1 deg

| cond | m | n | median | mean | >=1 deg |
|---|---|---|---|---|---|
| real | 1 | 122 | 0.409 [0.299, 0.536] | 0.886 [0.63, 1.2] | 0.213 [0.139, 0.287] |
| real | 2 | 122 | 0.375 [0.297, 0.485] | 1.35 [0.71, 2.53] | 0.238 [0.164, 0.311] |
| real | 3 | 122 | 0.342 [0.271, 0.462] | 1.62 [0.767, 3.15] | 0.23 [0.156, 0.311] |
| real | 4 | 122 | 0.315 [0.267, 0.474] | 1.07 [0.752, 1.47] | 0.221 [0.156, 0.295] |
| real | 1-2 | 244 | 0.392 [0.299, 0.498] | 1.12 [0.686, 1.8] | 0.225 [0.164, 0.291] |
| frozen | 1 | 122 | 0.573 [0.431, 0.7] | 1.73 [0.817, 3.16] | 0.287 [0.213, 0.369] |
| frozen | 2 | 122 | 0.338 [0.279, 0.474] | 1.01 [0.736, 1.32] | 0.246 [0.172, 0.32] |
| frozen | 3 | 122 | 0.326 [0.245, 0.408] | 1.07 [0.71, 1.52] | 0.221 [0.148, 0.295] |
| frozen | 4 | 122 | 0.31 [0.259, 0.455] | 1.04 [0.7, 1.41] | 0.23 [0.156, 0.303] |
| frozen | 1-2 | 244 | 0.452 [0.369, 0.552] | 1.37 [0.833, 2.17] | 0.266 [0.201, 0.332] |

real - frozen history, |phi1| steps 1-2, paired: median -0.00867 [-0.0456, 0.0177], mean -0.253 [-0.584, -0.0404]
signed phi1 real - frozen, steps 1-2: rms 5.07 deg, median |diff| 0.19 deg

## c: heading change over the next 0.25 s per deg of plan direction (phi1), through the origin

| source | cond | bin | steps | c [cluster CI] |
|---|---|---|---|---|
| comma1M launches | real | 0-1 m/s | 495 | 0.0501 [0.0351, 0.0665] |
| comma1M launches | real | 1-2 m/s | 280 | 0.125 [0.106, 0.146] |
| comma1M launches | real | 2-3 m/s | 231 | 0.181 [0.16, 0.201] |
| comma1M launches | real | 0-3 m/s | 1006 | 0.125 [0.103, 0.145] |
| comma1M launches | frozen | 0-1 m/s | 495 | 0.0449 [0.0286, 0.0616] |
| comma1M launches | frozen | 1-2 m/s | 279 | 0.124 [0.101, 0.147] |
| comma1M launches | frozen | 2-3 m/s | 231 | 0.181 [0.16, 0.201] |
| comma1M launches | frozen | 0-3 m/s | 1005 | 0.121 [0.0963, 0.143] |

## c from first differences: change of the next-step heading change on the change of phi1 between consecutive 0.25 s steps

- real, 0-1 m/s: c_diff 0.0442 [0.016, 0.0751] (n 373)
- real, 1-2 m/s: c_diff 0.108 [0.0575, 0.152] (n 280)
- real, 2-3 m/s: c_diff 0.186 [0.144, 0.228] (n 231)
- real, 0-3 m/s: c_diff 0.0985 [0.0661, 0.133] (n 884)
- frozen, 0-1 m/s: c_diff 0.0275 [0.011, 0.0526] (n 372)
- frozen, 1-2 m/s: c_diff 0.0924 [0.0466, 0.15] (n 279)
- frozen, 2-3 m/s: c_diff 0.186 [0.143, 0.226] (n 231)
- frozen, 0-3 m/s: c_diff 0.0692 [0.0384, 0.106] (n 882)

## history before the onset (real stream), localizer motion

2 s before onset: yaw-rate std median 0.020 deg/s (p90 0.054); net |heading change| median 0.024 deg (p90 0.113); max |yaw rate - mean| median 0.065 deg/s; max speed in the window median 0.089 m/s, share > 0.05 m/s 1.00
static before onset (s): median 10.2, min 1.8
phi1 while static (last 2 s before onset), real history: |phi1| median 6.95 deg; real - frozen: rms 27.57 deg (frozen = identical frame repeated, HUGSIM warm-up)

## the real car right after the onset

+1 s: v median 1.49 m/s; |net heading change| median 0.07 deg, p90 0.69, max 3.7
+2 s: v median 3.36 m/s; |net heading change| median 0.26 deg, p90 3.63, max 25.5
+3 s: v median 5.05 m/s; |net heading change| median 0.54 deg, p90 15.97, max 61.1

max(|phi1|) over steps 1-2 per event: share >= 1 deg 0.31, >= 2 deg 0.13, >= 4 deg 0.07, p90 2.4
- lean < 1: n 84, |net heading change| over 3 s: median 0.35, p90 10.23, share > 10 deg 0.11
- 1 <= lean < 2: n 22, |net heading change| over 3 s: median 0.87, p90 11.30, share > 10 deg 0.14
- lean >= 2: n 16, |net heading change| over 3 s: median 6.29, p90 39.16, share > 10 deg 0.38
- corr(|lean| steps 1-2, |heading change 3 s|): Spearman-like rank corr 0.38

## realised kernel: phi1(t) on real heading change over the previous 1.25 s (0.25 s steps m = 1..4 and 5..12), through the origin

- real, m 1-4: slope phi1 per deg heading change 6.07 [4.33, 8.14] (n 488)
- real, m 5-12: slope phi1 per deg heading change 1.2 [0.992, 1.39] (n 972)
- frozen, m 1-4: slope phi1 per deg heading change 6.32 [4.59, 8.51] (n 488)
- frozen, m 5-12: slope phi1 per deg heading change 1.2 [0.972, 1.39] (n 972)
