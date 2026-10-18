# Do third-party policies read the red light in AlpaSim nuPlan renders?

Diagnostic, CPU readout plus one new read (WA-JEPA on the wait set). Code: [`scripts/red_light_readers.py`](../scripts/red_light_readers.py) (waitset / arena
modes); the WA-JEPA read is `wait_set.sh read WAJEPA` (tag added to the third-party list: `run.sh wajepa`, the same driver path as decision 312 / wajepa_smoke.md,
pool job 1018-163501-df6f, 312 windows, 0 failed rollouts). The render's lamp never switches (wait_cap census_red_rear.md), so "reads red" can only show as:
more waiting at a red light than at a standstill without a light (red_first vs other_free), and fewer launches at red than elsewhere. One read per model, no seeds.
Other benchmark-shipped models with an AlpaSim driver: Alpamayo (`pai` driver) is the SH30 family on PAI cameras, not a shipped checkpoint, and has no wait-set
read; not run. JTN-L18-s0 (own model, already read) is included as a reference.

## 1-2. Wait set (set v2, 312 windows; a = static, b = launch at 3.0 s)

`wait ok a` = the ego rolled < 3 m (corridor) while the human waits; `launched b` = ego passed 1 m/s in the window; delay = ego launch minus human launch (negative = earlier).

| model | class | n a | wait ok a | zeros a | n b | launched b | median delay b (s) |
|---|---|--:|--:|--:|--:|--:|--:|
| P0 | red_first | 35 | 28 (80%) | 0 | 41 | 11 / 41 (27%) | -0.78 |
| P0 | red_queue | 39 | 39 (100%) | 0 | 35 | 5 / 35 (14%) | 1.62 |
| P0 | other_lead | 36 | 35 (97%) | 0 | 48 | 4 / 48 (8%) | 0.54 |
| P0 | other_free | 36 | 22 (61%) | 1 | 42 | 17 / 42 (40%) | -0.73 |
| P0 | ALL | 146 | 124 (85%) | 1 | 166 | 37 / 166 (22%) | -0.53 |
| RSG-L18-s0 | red_first | 35 | 0 (0%) | 0 | 41 | 41 / 41 (100%) | -0.88 |
| RSG-L18-s0 | red_queue | 39 | 15 (38%) | 3 | 35 | 33 / 35 (94%) | -0.13 |
| RSG-L18-s0 | other_lead | 36 | 18 (50%) | 1 | 48 | 47 / 48 (98%) | -0.23 |
| RSG-L18-s0 | other_free | 36 | 1 (3%) | 3 | 42 | 42 / 42 (100%) | -0.93 |
| RSG-L18-s0 | ALL | 146 | 34 (23%) | 7 | 166 | 163 / 166 (98%) | -0.51 |
| RTB1ARO-L18-s0 | red_first | 35 | 1 (3%) | 0 | 41 | 39 / 41 (95%) | -0.88 |
| RTB1ARO-L18-s0 | red_queue | 39 | 28 (72%) | 2 | 35 | 34 / 35 (97%) | -0.08 |
| RTB1ARO-L18-s0 | other_lead | 36 | 30 (83%) | 0 | 48 | 48 / 48 (100%) | -0.03 |
| RTB1ARO-L18-s0 | other_free | 36 | 1 (3%) | 4 | 42 | 40 / 42 (95%) | -1.03 |
| RTB1ARO-L18-s0 | ALL | 146 | 60 (41%) | 6 | 166 | 161 / 166 (97%) | -0.63 |
| G0014 | red_first | 35 | 29 (83%) | 1 | 41 | 9 / 41 (22%) | -1.28 |
| G0014 | red_queue | 39 | 37 (95%) | 1 | 35 | 14 / 35 (40%) | 1.04 |
| G0014 | other_lead | 36 | 31 (86%) | 0 | 48 | 17 / 48 (35%) | 0.47 |
| G0014 | other_free | 36 | 12 (33%) | 5 | 42 | 28 / 42 (67%) | -1.03 |
| G0014 | ALL | 146 | 109 (75%) | 7 | 166 | 68 / 166 (41%) | -0.26 |
| DD-sample | red_first | 35 | 1 (3%) | 0 | 41 | 33 / 41 (80%) | -1.68 |
| DD-sample | red_queue | 39 | 0 (0%) | 0 | 35 | 33 / 35 (94%) | -0.28 |
| DD-sample | other_lead | 36 | 0 (0%) | 1 | 48 | 47 / 48 (98%) | -1.33 |
| DD-sample | other_free | 36 | 0 (0%) | 2 | 42 | 40 / 42 (95%) | -1.78 |
| DD-sample | ALL | 146 | 1 (1%) | 3 | 166 | 153 / 166 (92%) | -1.58 |
| LTF-sample | red_first | 35 | 1 (3%) | 0 | 41 | 36 / 41 (88%) | -1.68 |
| LTF-sample | red_queue | 39 | 1 (3%) | 1 | 35 | 32 / 35 (91%) | -0.33 |
| LTF-sample | other_lead | 36 | 3 (8%) | 2 | 48 | 48 / 48 (100%) | -1.38 |
| LTF-sample | other_free | 36 | 0 (0%) | 3 | 42 | 42 / 42 (100%) | -1.73 |
| LTF-sample | ALL | 146 | 5 (3%) | 6 | 166 | 158 / 166 (95%) | -1.63 |
| APY10m10-AB-s0 | red_first | 35 | 0 (0%) | 0 | 41 | 41 / 41 (100%) | -1.23 |
| APY10m10-AB-s0 | red_queue | 39 | 0 (0%) | 16 | 35 | 35 / 35 (100%) | -0.83 |
| APY10m10-AB-s0 | other_lead | 36 | 0 (0%) | 9 | 48 | 48 / 48 (100%) | -0.78 |
| APY10m10-AB-s0 | other_free | 36 | 0 (0%) | 3 | 42 | 42 / 42 (100%) | -1.18 |
| APY10m10-AB-s0 | ALL | 146 | 0 (0%) | 28 | 166 | 166 / 166 (100%) | -1.08 |
| YR10m10-F-s0 | red_first | 35 | 0 (0%) | 0 | 41 | 41 / 41 (100%) | -0.88 |
| YR10m10-F-s0 | red_queue | 39 | 13 (33%) | 2 | 35 | 33 / 35 (94%) | -0.08 |
| YR10m10-F-s0 | other_lead | 36 | 18 (50%) | 1 | 48 | 48 / 48 (100%) | -0.13 |
| YR10m10-F-s0 | other_free | 36 | 1 (3%) | 3 | 42 | 41 / 42 (98%) | -0.93 |
| YR10m10-F-s0 | ALL | 146 | 32 (22%) | 6 | 166 | 163 / 166 (98%) | -0.53 |
| JTN-L18-s0 | red_first | 35 | 15 (43%) | 0 | 41 | 24 / 41 (59%) | 0.27 |
| JTN-L18-s0 | red_queue | 39 | 36 (92%) | 0 | 35 | 28 / 35 (80%) | 0.32 |
| JTN-L18-s0 | other_lead | 36 | 32 (89%) | 0 | 48 | 39 / 48 (81%) | 0.37 |
| JTN-L18-s0 | other_free | 36 | 11 (31%) | 2 | 42 | 32 / 42 (76%) | -0.41 |
| JTN-L18-s0 | ALL | 146 | 94 (64%) | 2 | 166 | 123 / 166 (74%) | 0.22 |
| WAJEPA | red_first | 35 | 0 (0%) | 0 | 41 | 41 / 41 (100%) | -0.88 |
| WAJEPA | red_queue | 39 | 36 (92%) | 0 | 35 | 31 / 35 (89%) | 0.22 |
| WAJEPA | other_lead | 36 | 33 (92%) | 0 | 48 | 42 / 48 (88%) | 0.29 |
| WAJEPA | other_free | 36 | 0 (0%) | 3 | 42 | 42 / 42 (100%) | -0.86 |
| WAJEPA | ALL | 146 | 69 (47%) | 3 | 166 | 156 / 166 (94%) | -0.53 |
| starter | red_first | 35 | 34 (97%) | 0 | 41 | 0 / 41 (0%) | nan |
| starter | red_queue | 39 | 39 (100%) | 0 | 36 | 0 / 36 (0%) | nan |
| starter | other_lead | 36 | 34 (94%) | 0 | 48 | 0 / 48 (0%) | nan |
| starter | other_free | 37 | 33 (89%) | 2 | 44 | 0 / 44 (0%) | nan |
| starter | ALL | 147 | 140 (95%) | 2 | 169 | 0 / 169 (0%) | nan |
| replay | red_first | 35 | 35 (100%) | 0 | 41 | 41 / 41 (100%) | -0.03 |
| replay | red_queue | 39 | 39 (100%) | 0 | 36 | 33 / 36 (92%) | -0.03 |
| replay | other_lead | 36 | 36 (100%) | 0 | 48 | 45 / 48 (94%) | -0.08 |
| replay | other_free | 37 | 37 (100%) | 1 | 44 | 42 / 44 (95%) | -0.06 |
| replay | ALL | 147 | 147 (100%) | 1 | 169 | 161 / 169 (95%) | -0.03 |

Key contrast, red_first (light, no lead) vs other_free (no light, no lead):

| model | wait ok a: red_first | other_free | zeros a: red_first | other_free | launched b: red_first | other_free | median delay b: red_first | other_free |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| P0 | 80% | 61% | 0/35 | 1/36 | 11/41 | 17/42 | -0.78 | -0.73 |
| RSG-L18-s0 | 0% | 3% | 0/35 | 3/36 | 41/41 | 42/42 | -0.88 | -0.93 |
| RTB1ARO-L18-s0 | 3% | 3% | 0/35 | 4/36 | 39/41 | 40/42 | -0.88 | -1.03 |
| G0014 | 83% | 33% | 1/35 | 5/36 | 9/41 | 28/42 | -1.28 | -1.03 |
| DD-sample | 3% | 0% | 0/35 | 2/36 | 33/41 | 40/42 | -1.68 | -1.78 |
| LTF-sample | 3% | 0% | 0/35 | 3/36 | 36/41 | 42/42 | -1.68 | -1.73 |
| APY10m10-AB-s0 | 0% | 0% | 0/35 | 3/36 | 41/41 | 42/42 | -1.23 | -1.18 |
| YR10m10-F-s0 | 0% | 3% | 0/35 | 3/36 | 41/41 | 41/42 | -0.88 | -0.93 |
| JTN-L18-s0 | 43% | 31% | 0/35 | 2/36 | 24/41 | 32/42 | 0.27 | -0.41 |
| WAJEPA | 0% | 0% | 0/35 | 3/36 | 41/41 | 42/42 | -0.88 | -0.86 |
| starter | 97% | 89% | 0/35 | 2/37 | 0/41 | 0/44 | nan | nan |
| replay | 100% | 100% | 0/35 | 1/37 | 41/41 | 42/44 | -0.03 | -0.06 |

## 3. Arena (held split, 2 228 valid windows, labels from the census rule)

Labels join the arena windows (3 s grid) to the census `windows.parquet` uniform 0.5 s grid on (log, round(t0/0.5)); all matched. red-static = human stops at red and
does not launch in the window; red-launch = stops at red and launches inside it; nolight-static = human static (maxdisp < 1 m) and no light line ahead;
other = the rest (including static at a green/unknown light). The WA-JEPA and JTN reads do not exist on the arena. Nothing here is a lamp reading: red-static and
nolight-static windows both ask for a standstill.

```
unmatched census rows: 0; unlabelled arena rows: 0

windows per label (RSG rows, valid): other 1325, red-static 682, nolight-static 147, red-launch 74
| model | red-static mean / zero (682) | red-launch mean / zero (74) | nolight-static mean / zero (147) | other mean / zero (1325) | all mean / zero |
|---|--:|--:|--:|--:|--:|
| RSG-L18-s0 (n 2228) | 0.546 / 45.3% | 0.957 / 1.4% | 0.762 / 23.8% | 0.902 / 8.5% | 0.786 / 20.6% |
| NR1F-L18-s0 (n 2228) | 0.592 / 40.8% | 0.939 / 2.7% | 0.925 / 7.5% | 0.886 / 10.3% | 0.800 / 19.2% |
| YR10m10-F-s0 (n 2228) | 0.514 / 48.5% | 0.978 / 0.0% | 0.687 / 31.3% | 0.866 / 12.2% | 0.750 / 24.1% |
| APY10m10-AB-s0 (n 2228) | 0.268 / 73.2% | 0.905 / 9.5% | 0.415 / 58.5% | 0.850 / 14.3% | 0.645 / 35.1% |
| G0014 (n 2106) | 0.900 / 9.8% | 0.633 / 8.7% | 0.871 / 12.9% | 0.752 / 20.8% | 0.803 / 16.4% |
| DD-sample (n 2228) | 0.584 / 41.6% | 0.873 / 0.0% | 0.367 / 63.3% | 0.725 / 26.0% | 0.663 / 32.4% |
| LTF-sample (n 2228) | 0.553 / 44.7% | 0.873 / 2.7% | 0.333 / 66.7% | 0.741 / 24.2% | 0.661 / 32.5% |
| starter (n 2228) | 0.908 / 9.2% | 0.523 / 2.7% | 1.000 / 0.0% | 0.550 / 38.6% | 0.688 / 25.9% |

zero types per label (count of zeros: corridor / at-fault front / offroad):

| model | red-static | red-launch | nolight-static | other |
|---|--:|--:|--:|--:|
| RSG-L18-s0 | 295 / 20 / 6 | 0 / 0 / 1 | 35 / 0 / 0 | 71 / 3 / 40 |
| NR1F-L18-s0 | 262 / 14 / 6 | 0 / 0 / 2 | 11 / 0 / 0 | 90 / 5 / 43 |
| YR10m10-F-s0 | 295 / 41 / 7 | 0 / 0 / 0 | 39 / 0 / 7 | 91 / 7 / 64 |
| APY10m10-AB-s0 | 412 / 154 / 6 | 1 / 6 / 0 | 64 / 18 / 9 | 124 / 25 / 53 |
| G0014 | 50 / 1 / 14 | 3 / 0 / 3 | 10 / 0 / 9 | 48 / 13 / 194 |
| DD-sample | 266 / 40 / 6 | 0 / 0 / 0 | 83 / 10 / 0 | 179 / 14 / 162 |
| LTF-sample | 279 / 40 / 6 | 1 / 1 / 0 | 75 / 23 / 0 | 174 / 23 / 135 |
| starter | 40 / 25 / 8 | 0 / 2 / 0 | 0 / 0 / 0 | 200 / 68 / 302 |
```

## Reading

Nobody reads the red lamp. In the wait set the red_first and other_free classes get nearly the same behaviour from RSG, V2, YR, APY, DD, LTF, WA-JEPA and NR1F
(wait success 0-3 % in both, 98-100 % launched in b, launch earlier than the human by about 0.9-1.8 s in both): they roll in both classes, and the roll costs a zero in the arena where the human stands still (red-static zero rate RSG 45 %, NR1F 41, YR 49, APY 73, DD 42, LTF 45; nolight-static 24, 8, 31, 59, 63, 67: higher at red for the L18 family, lower for DD / LTF, so no consistent red effect; the nolight-static class is small, n 147). Two models separate the classes, both for reasons other than the lamp. P0 and garage G0014 wait more at red_first than at other_free (P0 80 vs
61 %, G0014 83 vs 33 %) and launch less (P0 27 vs 40 %, G0014 22 vs 67 %): G0014 is also the only model with a low red-static zero rate in the arena (9.8 %)
but it is no better at nolight-static (12.9 %) and pays elsewhere (other 20.8 %), so it is a generally conservative, slow-starting policy; P0 waits everywhere and
launches in only 22 % of b windows. The starter (go-straight) shows the same pattern trivially. JTN waits at red_first 43 % vs 31 % at other_free and launches
59 % vs 76 %: a weak same-sign difference (n 35 / 36, not significant), delays are positive at red. The difference between the red and no-light classes is small
or absent against the 20-50 point spread between models, so with a lamp that never changes the data give no evidence that any model reads it; if one does, G0014 and
P0 are the candidates, and a clean test needs the lamp or a light-line control (a no-light window at the same standstill geometry), which the wait set's other_free class
only approximates (the lead-free flag, not the intersection geometry).
