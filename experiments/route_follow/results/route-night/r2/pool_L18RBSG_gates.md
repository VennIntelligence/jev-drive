# ONPOL-LONG pool gates, rows `L18RBSG`: PASS

R1 as amended (one fp16 ulp or 0.03 m on the first 15 plan points, >= 0.99): {'RBSG-L18-s0': 0.9989067894131185}; interpreted control [[100, 1.0]]; wrong-seed control (8 poses within 0.03 m, < 0.20): {'RBSG-L18-s0': 0.017894131185270425}. R2: ego at the log-replay decisions max 0.018 m (<= 0.3); actor samples within 0.5 m 0.99998 of 377414 (>= 0.9999). Rollouts without a time alignment to the navsim log (frame error > 0.1 m; Amendment 2), no rows: [{'scene': '2021.09.09.17.18.51_veh-48_00657_00876-798842f0b1c253b4', 'tag': 'RBSG-L18-s0', 'frame_err': 2.1068653683766074, 't0_rel': -2.0}].

Lane raster present on 0.9998 of the rows.

| term (served plans) | active rows | share | scenes | mean | mean when active | along-heading gradient max | share active on the log's own future |
|:--|--:|--:|--:|--:|--:|--:|--:|
| kcorr | 2890 | 0.1663 | 429 | 0.1159 | 0.697 | 7.5e-09 | 0.0063 |
| klane | 8744 | 0.5031 | 841 | 0.1319 | 0.262 | 7.5e-09 | 0.4606 |

Active share by decision index k: kcorr: 0 0.107, 1 0.091, 2 0.104, 3 0.123, 4 0.131, 5 0.148, 6 0.159, 7 0.158, 8 0.174, 9 0.180, 10 0.191, 11 0.210, 12 0.199, 13 0.209, 14 0.211, 15 0.212, 16 0.219, 17 0.205; klane: 0 0.542, 1 0.509, 2 0.518, 3 0.510, 4 0.522, 5 0.514, 6 0.500, 7 0.496, 8 0.488, 9 0.486, 10 0.490, 11 0.479, 12 0.478, 13 0.497, 14 0.507, 15 0.498, 16 0.511, 17 0.506

Items trained (K0: active on >= 0.2 % of the served plans, on <= 2 % of the logged futures): ['kcorr']

All gates pass.
