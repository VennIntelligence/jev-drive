# LEAD-AWARE HOLD tables (lead_aware.py)

Held windows 2228; reference RSG-L18-s0 + stop3 0.9438. Paired differences: mean per held window [95 % CI, bootstrap clustered by log, jevdrive.stats].

## Checks

- Latch class (RSG stop3 calls): {'free': 1082, 'static': 1004, 'stop': 142}; JTN stop3's own latch differs on 0 windows.
- Re-read noise (free windows: the stop3 read is the plain driver there): RSG native vs stop3 differ on 22 of 1082 free windows, summed +0.00; JTN off vs stop3 9, +0.02.
- Serve-time flag against the ground truth on the 1004 static-latched windows: oracle lead 578, serve 675, both 565, oracle only 13, serve only 110; stop-latched: oracle 72 / 142, serve 77. RSG lead head at the handoff on oracle-lead static windows: gap median 3.8 m (logged 4.2 m), p median 1.00.
- The 110 serve-only static latches (head: a lead within 30 m, log: none in the lane within 30 m at the window start): logged lead gap 0 with any gap, still end 76, head gap median 17.6 m; RSG native - stop3 there -67.7.

## Static-latched windows by lead (oracle): native vs stop3 means (zeros)

| lead | class | windows | RSG native | RSG stop3 | JTN off | JTN stop3 | NR1F native | NR1F - RSG stop3 (sum) | JTN off - JTN stop3 (sum) | RSG n - s3 (sum) |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| False | launch, human < 5 m | 29 | 0.966 (1) | 0.966 (1) | 0.931 (2) | 0.966 (1) | 0.931 (2) | -1.0 | -1.0 | +0.0 |
| False | launch, human >= 5 m | 52 | 0.883 (5) | 0.491 (1) | 0.773 (4) | 0.437 (1) | 0.847 (6) | +18.5 | +17.5 | +20.4 |
| False | moving end (static) | 22 | 0.864 (3) | 0.909 (2) | 0.909 (2) | 0.955 (1) | 0.864 (3) | -1.0 | -1.0 | -1.0 |
| False | still end | 323 | 0.084 (296) | 0.994 (2) | 0.536 (150) | 0.997 (1) | 0.105 (289) | -287.0 | -149.0 | -294.0 |
| True | launch, human < 5 m | 36 | 0.972 (1) | 1.000 (0) | 0.972 (1) | 1.000 (0) | 0.972 (1) | -1.0 | -1.0 | -1.0 |
| True | launch, human >= 5 m | 30 | 0.925 (0) | 0.508 (0) | 0.771 (0) | 0.433 (0) | 0.877 (0) | +11.1 | +10.1 | +12.5 |
| True | moving end (static) | 28 | 1.000 (0) | 1.000 (0) | 1.000 (0) | 1.000 (0) | 1.000 (0) | +0.0 | +0.0 | +0.0 |
| True | still end | 484 | 0.835 (80) | 0.975 (12) | 0.965 (17) | 0.998 (1) | 0.895 (51) | -39.0 | -16.0 | -68.0 |

Lead-present (oracle) still-end static-latched windows served natively, zero types (stop3 zeros in brackets):

- RSG-L18-s0: 484 windows, native zeros 80 {'corridor': 58, 'AF-front': 21, 'offroad': 1} (stop3: 12 {'AF-front': 11, 'offroad': 1})
- JTN-L18-s0: 484 windows, native zeros 17 {'corridor': 13, 'AF-front': 3, 'offroad': 1} (stop3: 1 {'offroad': 1})
- NR1F-L18-s0: 484 windows, native zeros 51 {'corridor': 35, 'AF-front': 15, 'offroad': 1} (stop3 (RSG credit): 12 {'AF-front': 11, 'offroad': 1})
- with the serve flag (471 of them): native zeros RSG 67, JTN 11, NR1F 44

## Rules vs RSG-L18-s0 + stop3

| model | rule | switched | sum vs RSG+stop3 | per window [CI] | rule increment vs the model's stop3: sum | per window [CI] | new / fixed zeros vs RSG+stop3 | zeros | AF-front | corridor |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| RSG-L18-s0 | model + stop3 | 0 | +0.0 | +0.0000 [+0.0000, +0.0000] | +0.0 | +0.0000 [+0.0000, +0.0000] | 0 / 0 | 64 | 12 | 8 |
| RSG-L18-s0 | A static latch, lead -> native [oracle] | 578 | -56.5 | -0.0253 [-0.0413, -0.0115] | -56.5 | -0.0253 [-0.0413, -0.0115] | 69 / 0 | 133 | 23 | 66 |
| RSG-L18-s0 | A' any latch (static + stop), lead -> native [oracle] | 650 | -56.0 | -0.0251 [-0.0411, -0.0113] | -56.0 | -0.0251 [-0.0411, -0.0113] | 69 / 0 | 133 | 23 | 66 |
| RSG-L18-s0 | B static latch, lead and handoff plan 2 s arc >= 0.5 m -> native [oracle] | 275 | -54.1 | -0.0243 [-0.0406, -0.0104] | -54.1 | -0.0243 [-0.0406, -0.0104] | 65 / 0 | 129 | 23 | 62 |
| RSG-L18-s0 | B static latch, lead and handoff plan 2 s arc >= 1 m -> native [oracle] | 9 | +1.4 | +0.0006 [-0.0010, +0.0022] | +1.4 | +0.0006 [-0.0010, +0.0022] | 2 / 0 | 66 | 12 | 10 |
| RSG-L18-s0 | B static latch, lead and handoff plan 2 s arc >= 2 m -> native [oracle] | 1 | +0.7 | +0.0003 [+0.0000, +0.0010] | +0.7 | +0.0003 [+0.0000, +0.0010] | 0 / 0 | 64 | 12 | 8 |
| RSG-L18-s0 | C static latch, lead: release when the held plan's 2 s arc >= 2 m (UB) [oracle] | 173 | -56.5 | -0.0253 [-0.0413, -0.0115] | -56.5 | -0.0253 [-0.0413, -0.0115] | 69 / 0 | 133 | 23 | 66 |
| RSG-L18-s0 | C' ... at a call where also RSG's lead head has the lead present and moving > 1 m/s (UB) [oracle] | 49 | +8.3 | +0.0037 [+0.0000, +0.0069] | +8.3 | +0.0037 [+0.0000, +0.0069] | 4 / 0 | 68 | 12 | 12 |
| RSG-L18-s0 | A static latch, lead -> native [serve] | 675 | -111.2 | -0.0499 [-0.0778, -0.0238] | -111.2 | -0.0499 [-0.0778, -0.0238] | 130 / 0 | 194 | 19 | 129 |
| RSG-L18-s0 | A' any latch (static + stop), lead -> native [serve] | 752 | -114.7 | -0.0515 [-0.0806, -0.0246] | -114.7 | -0.0515 [-0.0806, -0.0246] | 134 / 0 | 198 | 19 | 133 |
| RSG-L18-s0 | B static latch, lead and handoff plan 2 s arc >= 0.5 m -> native [serve] | 371 | -108.8 | -0.0488 [-0.0763, -0.0232] | -108.8 | -0.0488 [-0.0763, -0.0232] | 126 / 0 | 190 | 19 | 125 |
| RSG-L18-s0 | B static latch, lead and handoff plan 2 s arc >= 1 m -> native [serve] | 16 | +0.6 | +0.0003 [-0.0024, +0.0030] | +0.6 | +0.0003 [-0.0024, +0.0030] | 5 / 0 | 69 | 12 | 12 |
| RSG-L18-s0 | B static latch, lead and handoff plan 2 s arc >= 2 m -> native [serve] | 2 | +1.4 | +0.0006 [+0.0000, +0.0019] | +1.4 | +0.0006 [+0.0000, +0.0019] | 0 / 0 | 64 | 12 | 8 |
| RSG-L18-s0 | C static latch, lead: release when the held plan's 2 s arc >= 2 m (UB) [serve] | 270 | -111.2 | -0.0499 [-0.0778, -0.0238] | -111.2 | -0.0499 [-0.0778, -0.0238] | 130 / 0 | 194 | 19 | 129 |
| RSG-L18-s0 | C' ... at a call where also RSG's lead head has the lead present and moving > 1 m/s (UB) [serve] | 71 | -3.2 | -0.0014 [-0.0105, +0.0064] | -3.2 | -0.0014 [-0.0105, +0.0064] | 18 / 0 | 82 | 12 | 26 |
| RSG-L18-s0 | A static latch, lead -> native [serve, gap <= 20 m] | 612 | -69.1 | -0.0310 [-0.0526, -0.0115] | -69.1 | -0.0310 [-0.0526, -0.0115] | 86 / 0 | 150 | 14 | 90 |
| JTN-L18-s0 | model + stop3 | 0 | -2.8 | -0.0013 [-0.0087, +0.0096] | +0.0 | +0.0000 [+0.0000, +0.0000] | 20 / 24 | 60 | 2 | 9 |
| JTN-L18-s0 | A static latch, lead -> native [oracle] | 578 | -9.7 | -0.0043 [-0.0135, +0.0072] | -6.9 | -0.0031 [-0.0088, +0.0019] | 36 / 23 | 77 | 6 | 22 |
| JTN-L18-s0 | A' any latch (static + stop), lead -> native [oracle] | 650 | -9.4 | -0.0042 [-0.0134, +0.0073] | -6.6 | -0.0030 [-0.0087, +0.0020] | 36 / 23 | 77 | 6 | 22 |
| JTN-L18-s0 | B static latch, lead and handoff plan 2 s arc >= 0.5 m -> native [oracle] | 22 | +2.6 | +0.0011 [-0.0070, +0.0128] | +5.4 | +0.0024 [+0.0002, +0.0045] | 22 / 24 | 62 | 2 | 11 |
| JTN-L18-s0 | B static latch, lead and handoff plan 2 s arc >= 1 m -> native [oracle] | 7 | -1.8 | -0.0008 [-0.0088, +0.0110] | +1.0 | +0.0005 [-0.0013, +0.0021] | 22 / 24 | 62 | 2 | 11 |
| JTN-L18-s0 | B static latch, lead and handoff plan 2 s arc >= 2 m -> native [oracle] | 1 | -2.1 | -0.0009 [-0.0085, +0.0098] | +0.7 | +0.0003 [+0.0000, +0.0010] | 20 / 24 | 60 | 2 | 9 |
| JTN-L18-s0 | C static latch, lead: release when the held plan's 2 s arc >= 2 m (UB) [oracle] | 110 | -9.7 | -0.0043 [-0.0135, +0.0072] | -6.9 | -0.0031 [-0.0088, +0.0019] | 36 / 23 | 77 | 6 | 22 |
| JTN-L18-s0 | C' ... at a call where also RSG's lead head has the lead present and moving > 1 m/s (UB) [oracle] | 36 | +6.0 | +0.0027 [-0.0053, +0.0141] | +8.8 | +0.0040 [+0.0020, +0.0059] | 21 / 24 | 61 | 2 | 10 |
| JTN-L18-s0 | A static latch, lead -> native [serve] | 675 | -43.6 | -0.0196 [-0.0349, -0.0029] | -40.8 | -0.0183 [-0.0323, -0.0054] | 76 / 23 | 117 | 6 | 59 |
| JTN-L18-s0 | A' any latch (static + stop), lead -> native [serve] | 752 | -45.4 | -0.0204 [-0.0360, -0.0034] | -42.6 | -0.0191 [-0.0335, -0.0060] | 78 / 23 | 119 | 6 | 61 |
| JTN-L18-s0 | B static latch, lead and handoff plan 2 s arc >= 0.5 m -> native [serve] | 29 | +4.4 | +0.0020 [-0.0065, +0.0136] | +7.2 | +0.0032 [+0.0006, +0.0062] | 23 / 24 | 63 | 2 | 11 |
| JTN-L18-s0 | B static latch, lead and handoff plan 2 s arc >= 1 m -> native [serve] | 8 | -0.7 | -0.0003 [-0.0084, +0.0113] | +2.1 | +0.0010 [-0.0008, +0.0028] | 22 / 24 | 62 | 2 | 10 |
| JTN-L18-s0 | B static latch, lead and handoff plan 2 s arc >= 2 m -> native [serve] | 2 | -1.4 | -0.0006 [-0.0083, +0.0102] | +1.4 | +0.0006 [+0.0000, +0.0019] | 20 / 24 | 60 | 2 | 9 |
| JTN-L18-s0 | C static latch, lead: release when the held plan's 2 s arc >= 2 m (UB) [serve] | 182 | -43.6 | -0.0196 [-0.0349, -0.0029] | -40.8 | -0.0183 [-0.0323, -0.0054] | 76 / 23 | 117 | 6 | 59 |
| JTN-L18-s0 | C' ... at a call where also RSG's lead head has the lead present and moving > 1 m/s (UB) [serve] | 43 | +5.6 | +0.0025 [-0.0058, +0.0142] | +8.4 | +0.0038 [+0.0006, +0.0071] | 23 / 24 | 63 | 2 | 12 |
| JTN-L18-s0 | A static latch, lead -> native [serve, gap <= 20 m] | 612 | -24.1 | -0.0108 [-0.0253, +0.0050] | -21.3 | -0.0096 [-0.0224, +0.0009] | 56 / 24 | 96 | 4 | 40 |
| NR1F-L18-s0 | model + stop3 (latched: RSG + stop3 credit) | 0 | -2.3 | -0.0010 [-0.0037, +0.0017] | +0.0 | +0.0000 [+0.0000, +0.0000] | 15 / 8 | 71 | 13 | 13 |
| NR1F-L18-s0 | A static latch, lead -> native [oracle] | 578 | -31.2 | -0.0140 [-0.0277, -0.0024] | -28.9 | -0.0130 [-0.0268, -0.0016] | 56 / 9 | 111 | 18 | 48 |
| NR1F-L18-s0 | A' any latch (static + stop), lead -> native [oracle] | 650 | -31.4 | -0.0141 [-0.0277, -0.0027] | -29.1 | -0.0131 [-0.0269, -0.0019] | 57 / 9 | 112 | 18 | 49 |
| NR1F-L18-s0 | B static latch, lead and handoff plan 2 s arc >= 0.5 m -> native [oracle] | 109 | -27.8 | -0.0125 [-0.0256, -0.0015] | -25.5 | -0.0115 [-0.0245, -0.0008] | 51 / 9 | 106 | 18 | 43 |
| NR1F-L18-s0 | B static latch, lead and handoff plan 2 s arc >= 1 m -> native [oracle] | 11 | -4.9 | -0.0022 [-0.0056, +0.0012] | -2.6 | -0.0012 [-0.0039, +0.0011] | 20 / 8 | 76 | 13 | 18 |
| NR1F-L18-s0 | B static latch, lead and handoff plan 2 s arc >= 2 m -> native [oracle] | 1 | -1.5 | -0.0007 [-0.0034, +0.0019] | +0.7 | +0.0003 [+0.0000, +0.0010] | 15 / 8 | 71 | 13 | 13 |
| NR1F-L18-s0 | A static latch, lead -> native [serve] | 675 | -95.8 | -0.0430 [-0.0700, -0.0177] | -93.5 | -0.0420 [-0.0694, -0.0163] | 127 / 9 | 182 | 17 | 117 |
| NR1F-L18-s0 | A' any latch (static + stop), lead -> native [serve] | 752 | -99.0 | -0.0444 [-0.0715, -0.0191] | -96.7 | -0.0434 [-0.0709, -0.0180] | 131 / 9 | 186 | 18 | 120 |
| NR1F-L18-s0 | B static latch, lead and handoff plan 2 s arc >= 0.5 m -> native [serve] | 204 | -87.4 | -0.0392 [-0.0640, -0.0156] | -85.1 | -0.0382 [-0.0636, -0.0143] | 117 / 9 | 172 | 17 | 107 |
| NR1F-L18-s0 | B static latch, lead and handoff plan 2 s arc >= 1 m -> native [serve] | 23 | -6.9 | -0.0031 [-0.0075, +0.0009] | -4.7 | -0.0021 [-0.0064, +0.0015] | 25 / 8 | 81 | 13 | 21 |
| NR1F-L18-s0 | B static latch, lead and handoff plan 2 s arc >= 2 m -> native [serve] | 2 | -0.9 | -0.0004 [-0.0032, +0.0023] | +1.4 | +0.0006 [+0.0000, +0.0019] | 15 / 8 | 71 | 13 | 13 |
| NR1F-L18-s0 | A static latch, lead -> native [serve, gap <= 20 m] | 612 | -51.8 | -0.0232 [-0.0433, -0.0061] | -49.5 | -0.0222 [-0.0428, -0.0049] | 81 / 9 | 136 | 14 | 75 |
| RSG-L18-s0 | D static latch, lead -> JEV_HOLD=lead's read (full hold, lead-head release; a real read) [oracle] | 578 | -3.9 | -0.0018 [-0.0078, +0.0078] | -3.9 | -0.0018 [-0.0078, +0.0078] | 0 / 11 | 53 | 1 | 8 |
| RSG-L18-s0 | D static latch, lead -> JEV_HOLD=lead's read (full hold, lead-head release; a real read) [serve] | 675 | -9.2 | -0.0041 [-0.0106, +0.0060] | -9.2 | -0.0041 [-0.0106, +0.0060] | 1 / 13 | 52 | 1 | 8 |
| NR1F-L18-s0 | NR1F native on every lead window, RSG + stop3 elsewhere [oracle] | 903 | -27.7 | -0.0124 [-0.0264, -0.0012] | - | - | 43 / 3 | 104 | 17 | 43 |
| NR1F-L18-s0 | NR1F native on every lead window, RSG + stop3 elsewhere [serve] | 1037 | -96.0 | -0.0431 [-0.0702, -0.0181] | - | - | 118 / 3 | 179 | 17 | 114 |

Shorter static leash with a lead (E; offline, diag.md (b) / residual.md's estimator: a lead-present static-latched window whose stop3 read is an AF-front zero with the ego past the new leash is credited as held (score 1 when the human travels < 5 m, else the progress estimate); other non-zero, non-rear-hit windows get the progress change min((pre + min(reach, L)) / (0.8 gt), 1) minus the same at 3 m, reach = the native read's travel past the handoff; E + C' = released windows credited with the native read, the rest shortened):

| model | flag | leash with a lead (m) | AF-front avoided | launch progress (sum) | increment vs the model's stop3 (sum) | per window | E + C' (sum) | per window |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| RSG-L18-s0 | oracle | 2 | 11 | -3.3 | +7.7 | +0.0035 | +19.0 | +0.0085 |
| RSG-L18-s0 | oracle | 2.3 | 11 | -2.3 | +8.7 | +0.0039 | +19.1 | +0.0086 |
| RSG-L18-s0 | oracle | 2.5 | 11 | -1.6 | +9.4 | +0.0042 | +19.2 | +0.0086 |
| RSG-L18-s0 | serve | 2 | 11 | -5.2 | +5.8 | +0.0026 | +6.1 | +0.0027 |
| RSG-L18-s0 | serve | 2.3 | 11 | -3.6 | +7.4 | +0.0033 | +6.6 | +0.0030 |
| RSG-L18-s0 | serve | 2.5 | 11 | -2.6 | +8.4 | +0.0038 | +7.0 | +0.0031 |
| JTN-L18-s0 | oracle | 2 | 0 | -2.4 | -2.4 | -0.0011 | +8.6 | +0.0039 |
| JTN-L18-s0 | oracle | 2.3 | 0 | -1.6 | -1.6 | -0.0007 | +8.7 | +0.0039 |
| JTN-L18-s0 | oracle | 2.5 | 0 | -1.2 | -1.2 | -0.0005 | +8.7 | +0.0039 |
| JTN-L18-s0 | serve | 2 | 0 | -4.3 | -4.3 | -0.0019 | +6.5 | +0.0029 |
| JTN-L18-s0 | serve | 2.3 | 0 | -3.0 | -3.0 | -0.0013 | +7.1 | +0.0032 |
| JTN-L18-s0 | serve | 2.5 | 0 | -2.1 | -2.1 | -0.0009 | +7.4 | +0.0033 |

Release grid (C' rule; release at the first held call from 1 s where the held plan's 2 s arc >= arc and RSG's lead head has the lead present (p > 0.5) at speed > v; released windows credited with the native read: an in-sample upper bound). Top 3 per model and flag by the increment over the model's stop3:

| model | flag | arc (m) | v (m/s) | released | increment (sum) | per window | new / fixed zeros vs the model's stop3 |
|---|---|--:|--:|--:|--:|--:|--:|
| RSG-L18-s0 | oracle | 2 | 1 | 49 | +8.3 | +0.0037 | 4 / 0 |
| RSG-L18-s0 | oracle | 1 | 1 | 53 | +8.3 | +0.0037 | 4 / 0 |
| RSG-L18-s0 | oracle | 3 | 1 | 40 | +7.3 | +0.0033 | 4 / 0 |
| RSG-L18-s0 | serve | 0 | 2 | 20 | +2.6 | +0.0012 | 4 / 0 |
| RSG-L18-s0 | serve | 1 | 2 | 19 | +2.0 | +0.0009 | 4 / 0 |
| RSG-L18-s0 | serve | 3 | 2 | 11 | +2.0 | +0.0009 | 2 / 0 |
| JTN-L18-s0 | oracle | 1 | 1 | 41 | +8.9 | +0.0040 | 1 / 0 |
| JTN-L18-s0 | oracle | 2 | 1 | 36 | +8.8 | +0.0040 | 1 / 0 |
| JTN-L18-s0 | oracle | 3 | 1 | 31 | +8.2 | +0.0037 | 1 / 0 |
| JTN-L18-s0 | serve | 2 | 1 | 43 | +8.4 | +0.0038 | 3 / 0 |
| JTN-L18-s0 | serve | 3 | 1 | 35 | +8.1 | +0.0037 | 2 / 0 |
| JTN-L18-s0 | serve | 2 | 1.5 | 30 | +5.2 | +0.0023 | 3 / 0 |

Reference RSG-L18-s0 + stop3: zeros 64, AF-front 12, corridor 8.

## By stratum (model + stop3 and rule A; sum / contribution = summed difference / all held windows; increment = vs the model's stop3)

| model | rule | stratum | windows | sum vs RSG+stop3 | contribution | per window [CI] | increment (sum) | new / fixed zeros |
|---|---|---|--:|--:|--:|--:|--:|--:|
| RSG-L18-s0 | A static latch, lead -> native [oracle] | static | 878 | -68.0 | -0.0305 | -0.077 [-0.113, -0.045] | -68.0 | 68 / 0 |
| RSG-L18-s0 | A static latch, lead -> native [oracle] | launch | 200 | +11.5 | +0.0052 | +0.058 [+0.037, +0.083] | +11.5 | 1 / 0 |
| RSG-L18-s0 | A static latch, lead -> native [oracle] | braking | 372 | +0.0 | +0.0000 | +0.000 [+0.000, +0.000] | +0.0 | 0 / 0 |
| RSG-L18-s0 | A static latch, lead -> native [oracle] | other | 449 | +0.0 | +0.0000 | +0.000 [+0.000, +0.000] | +0.0 | 0 / 0 |
| RSG-L18-s0 | A static latch, lead -> native [oracle] | turn | 213 | +0.0 | +0.0000 | +0.000 [+0.000, +0.000] | +0.0 | 0 / 0 |
| RSG-L18-s0 | A static latch, lead -> native [oracle] | cruise | 116 | +0.0 | +0.0000 | +0.000 [+0.000, +0.000] | +0.0 | 0 / 0 |
| RSG-L18-s0 | A static latch, lead -> native [serve] | static | 878 | -127.0 | -0.0570 | -0.145 [-0.211, -0.080] | -127.0 | 127 / 0 |
| RSG-L18-s0 | A static latch, lead -> native [serve] | launch | 200 | +15.8 | +0.0071 | +0.079 [+0.051, +0.106] | +15.8 | 3 / 0 |
| RSG-L18-s0 | A static latch, lead -> native [serve] | braking | 372 | +0.0 | +0.0000 | +0.000 [+0.000, +0.000] | +0.0 | 0 / 0 |
| RSG-L18-s0 | A static latch, lead -> native [serve] | other | 449 | +0.0 | +0.0000 | +0.000 [+0.000, +0.000] | +0.0 | 0 / 0 |
| RSG-L18-s0 | A static latch, lead -> native [serve] | turn | 213 | +0.0 | +0.0000 | +0.000 [+0.000, +0.000] | +0.0 | 0 / 0 |
| RSG-L18-s0 | A static latch, lead -> native [serve] | cruise | 116 | +0.0 | +0.0000 | +0.000 [+0.000, +0.000] | +0.0 | 0 / 0 |
| JTN-L18-s0 | model + stop3 | static | 878 | +13.0 | +0.0058 | +0.015 [+0.001, +0.041] | +0.0 | 0 / 13 |
| JTN-L18-s0 | model + stop3 | launch | 200 | -6.6 | -0.0030 | -0.033 [-0.050, -0.019] | +0.0 | 1 / 0 |
| JTN-L18-s0 | model + stop3 | braking | 372 | -6.0 | -0.0027 | -0.016 [-0.031, -0.004] | +0.0 | 8 / 2 |
| JTN-L18-s0 | model + stop3 | other | 449 | -3.5 | -0.0016 | -0.008 [-0.025, +0.010] | +0.0 | 6 / 3 |
| JTN-L18-s0 | model + stop3 | turn | 213 | +0.8 | +0.0003 | +0.004 [-0.032, +0.031] | +0.0 | 5 / 6 |
| JTN-L18-s0 | model + stop3 | cruise | 116 | -0.4 | -0.0002 | -0.004 [-0.013, +0.002] | +0.0 | 0 / 0 |
| JTN-L18-s0 | A static latch, lead -> native [oracle] | static | 878 | -3.0 | -0.0013 | -0.003 [-0.024, +0.023] | -16.0 | 15 / 12 |
| JTN-L18-s0 | A static latch, lead -> native [oracle] | launch | 200 | +2.5 | +0.0011 | +0.013 [-0.012, +0.040] | +9.1 | 2 / 0 |
| JTN-L18-s0 | A static latch, lead -> native [oracle] | braking | 372 | -6.0 | -0.0027 | -0.016 [-0.031, -0.004] | +0.0 | 8 / 2 |
| JTN-L18-s0 | A static latch, lead -> native [oracle] | other | 449 | -3.5 | -0.0016 | -0.008 [-0.025, +0.010] | +0.0 | 6 / 3 |
| JTN-L18-s0 | A static latch, lead -> native [oracle] | turn | 213 | +0.8 | +0.0003 | +0.004 [-0.032, +0.031] | +0.0 | 5 / 6 |
| JTN-L18-s0 | A static latch, lead -> native [oracle] | cruise | 116 | -0.4 | -0.0002 | -0.004 [-0.013, +0.002] | +0.0 | 0 / 0 |
| JTN-L18-s0 | A static latch, lead -> native [serve] | static | 878 | -40.0 | -0.0180 | -0.046 [-0.091, -0.001] | -53.0 | 52 / 12 |
| JTN-L18-s0 | A static latch, lead -> native [serve] | launch | 200 | +5.6 | +0.0025 | +0.028 [-0.011, +0.063] | +12.2 | 5 / 0 |
| JTN-L18-s0 | A static latch, lead -> native [serve] | braking | 372 | -6.0 | -0.0027 | -0.016 [-0.031, -0.004] | +0.0 | 8 / 2 |
| JTN-L18-s0 | A static latch, lead -> native [serve] | other | 449 | -3.5 | -0.0016 | -0.008 [-0.025, +0.010] | +0.0 | 6 / 3 |
| JTN-L18-s0 | A static latch, lead -> native [serve] | turn | 213 | +0.8 | +0.0003 | +0.004 [-0.032, +0.031] | +0.0 | 5 / 6 |
| JTN-L18-s0 | A static latch, lead -> native [serve] | cruise | 116 | -0.4 | -0.0002 | -0.004 [-0.013, +0.002] | +0.0 | 0 / 0 |
| NR1F-L18-s0 | model + stop3 (latched: RSG + stop3 credit) | static | 878 | +0.0 | +0.0000 | +0.000 [+0.000, +0.000] | +0.0 | 0 / 0 |
| NR1F-L18-s0 | model + stop3 (latched: RSG + stop3 credit) | launch | 200 | -1.3 | -0.0006 | -0.006 [-0.017, -0.000] | +0.0 | 1 / 0 |
| NR1F-L18-s0 | model + stop3 (latched: RSG + stop3 credit) | braking | 372 | -2.7 | -0.0012 | -0.007 [-0.017, +0.002] | +0.0 | 6 / 2 |
| NR1F-L18-s0 | model + stop3 (latched: RSG + stop3 credit) | other | 449 | +3.8 | +0.0017 | +0.008 [+0.000, +0.018] | +0.0 | 1 / 3 |
| NR1F-L18-s0 | model + stop3 (latched: RSG + stop3 credit) | turn | 213 | -2.8 | -0.0012 | -0.013 [-0.033, +0.003] | +0.0 | 7 / 3 |
| NR1F-L18-s0 | model + stop3 (latched: RSG + stop3 credit) | cruise | 116 | +0.7 | +0.0003 | +0.006 [+0.002, +0.010] | +0.0 | 0 / 0 |
| NR1F-L18-s0 | A static latch, lead -> native [oracle] | static | 878 | -39.0 | -0.0175 | -0.044 [-0.075, -0.018] | -39.0 | 40 / 1 |
| NR1F-L18-s0 | A static latch, lead -> native [oracle] | launch | 200 | +8.8 | +0.0040 | +0.044 [+0.022, +0.069] | +10.1 | 2 / 0 |
| NR1F-L18-s0 | A static latch, lead -> native [oracle] | braking | 372 | -2.7 | -0.0012 | -0.007 [-0.017, +0.002] | +0.0 | 6 / 2 |
| NR1F-L18-s0 | A static latch, lead -> native [oracle] | other | 449 | +3.8 | +0.0017 | +0.008 [+0.000, +0.018] | +0.0 | 1 / 3 |
| NR1F-L18-s0 | A static latch, lead -> native [oracle] | turn | 213 | -2.8 | -0.0012 | -0.013 [-0.033, +0.003] | +0.0 | 7 / 3 |
| NR1F-L18-s0 | A static latch, lead -> native [oracle] | cruise | 116 | +0.7 | +0.0003 | +0.006 [+0.002, +0.010] | +0.0 | 0 / 0 |
| NR1F-L18-s0 | A static latch, lead -> native [serve] | static | 878 | -106.0 | -0.0476 | -0.121 [-0.191, -0.057] | -106.0 | 107 / 1 |
| NR1F-L18-s0 | A static latch, lead -> native [serve] | launch | 200 | +11.2 | +0.0050 | +0.056 [+0.026, +0.085] | +12.5 | 6 / 0 |
| NR1F-L18-s0 | A static latch, lead -> native [serve] | braking | 372 | -2.7 | -0.0012 | -0.007 [-0.017, +0.002] | +0.0 | 6 / 2 |
| NR1F-L18-s0 | A static latch, lead -> native [serve] | other | 449 | +3.8 | +0.0017 | +0.008 [+0.000, +0.018] | +0.0 | 1 / 3 |
| NR1F-L18-s0 | A static latch, lead -> native [serve] | turn | 213 | -2.8 | -0.0012 | -0.013 [-0.033, +0.003] | +0.0 | 7 / 3 |
| NR1F-L18-s0 | A static latch, lead -> native [serve] | cruise | 116 | +0.7 | +0.0003 | +0.006 [+0.002, +0.010] | +0.0 | 0 / 0 |
