# CB2 step 0: b3 DAgger candidates on human-standing / -stopping states (cb2.py creep)

Candidates = k >= 2, fed >= 2 m/s, teacher label, not in the 15 % held-out logs. HS = the human's logged 4 s arc < 0.5 m; HP = not HS and his end speed < 0.5 m/s (stops within 4 s). Conflict = HS / HP rows whose b3 label (longitudinal = the served arc) ends beyond the STAND cap max(his arcs) + 1 m. Lead = a logged vehicle at t0 within |y| < 2 m ahead (no lead = line stop: red light / stop sign proxy).

| pool | cand | HS | HP | HS no lead | HP no lead | conflict rows (no lead) | excess m p50 / p90 | served - human 4 s p50 / p90 on HS+HP | R1 keep / lat / lat>0.5 / conflict left | R2 keep / lat / lat>0.5 / conflict left |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| L18RB0 | 10019 | 79 | 332 | 59 | 160 | 265 (152) | 6.92 / 15.68 | 4.97 / 16.09 | 0.992 / 0.978 / 0.986 / 0.614 FAIL | 0.959 / 0.936 / 0.956 / 0.000 PASS |
| L18D3 | 10249 | 83 | 383 | 58 | 193 | 310 (182) | 7.87 / 16.79 | 4.71 / 19.11 | 0.992 / 0.982 / 0.985 / 0.606 FAIL | 0.955 / 0.925 / 0.938 / 0.000 PASS |
| PN0 | 740 | 48 | 66 | 46 | 23 | 63 (53) | 16.41 / 21.32 | 16.16 / 23.68 | 0.935 / 0.962 / 0.962 / 0.230 FAIL | 0.846 / 0.897 / 0.908 / 0.000 PASS |
| PN1 | 967 | 133 | 76 | 113 | 39 | 159 (138) | 17.27 / 21.26 | 18.76 / 25.95 | 0.862 / 0.814 / 0.756 / 0.078 FAIL | 0.784 / 0.750 / 0.714 / 0.000 FAIL |
| PN2 | 2495 | 62 | 186 | 52 | 75 | 101 (75) | 10.21 / 17.21 | 0.69 / 18.21 | 0.975 / 0.954 / 0.954 / 0.181 FAIL | 0.901 / 0.920 / 0.918 / 0.000 PASS |
| PN3 | 2286 | 121 | 133 | 109 | 63 | 155 (128) | 12.88 / 17.49 | 15.09 / 19.96 | 0.947 / 0.941 / 0.914 / 0.215 FAIL | 0.889 / 0.902 / 0.878 / 0.000 PASS |
| PN4 | 483 | 83 | 35 | 39 | 29 | 92 (47) | 16.97 / 20.88 | 20.14 / 25.49 | 0.828 / 0.649 / 0.657 / 0.035 FAIL | 0.756 / 0.563 / 0.604 / 0.000 FAIL |
| PN5 | 1670 | 47 | 131 | 38 | 49 | 72 (41) | 10.85 / 17.52 | 0.48 / 18.96 | 0.972 / 0.903 / 0.937 / 0.072 PASS | 0.893 / 0.863 / 0.912 / 0.000 PASS |

G0 lines: keep >= 0.8, lateral mass and rows > 0.5 m kept >= 0.8, conflict excess left <= 0.1. Simplest rule passing on every pool: **none**.
