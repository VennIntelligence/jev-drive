# ONPOL-LONG pool gates, rows `L18Y`: PASS

R1 as amended (one fp16 ulp or 0.03 m on the first 15 plan points, >= 0.99): {'YR10m10-F-s0': 0.999133248584306}; interpreted control [[100, 1.0]]; wrong-seed control (8 poses within 0.03 m, < 0.20): {'YR10m10-F-s0': 0.015717092337917484}. R2: ego at the log-replay decisions max 0.018 m (<= 0.3); actor samples within 0.5 m 0.99998 of 377414 (>= 0.9999). Rollouts without a time alignment to the navsim log (frame error > 0.1 m; Amendment 2), no rows: [{'scene': '2021.09.09.17.18.51_veh-48_00657_00876-798842f0b1c253b4', 'tag': 'YR10m10-F-s0', 'frame_err': 2.1068653683766074, 't0_rel': -2.0}].

Lane raster present on 0.9999 of the rows.

| term (served plans) | active rows | share | scenes | mean | mean when active | along-heading gradient max | share active on the log's own future |
|:--|--:|--:|--:|--:|--:|--:|--:|
| kcorr | 3783 | 0.2186 | 544 | 0.2193 | 1.003 | 7.5e-09 | 0.0060 |
| klane | 10113 | 0.5844 | 914 | 0.1778 | 0.304 | 7.5e-09 | 0.4607 |

Active share by decision index k: kcorr: 0 0.129, 1 0.088, 2 0.097, 3 0.135, 4 0.166, 5 0.185, 6 0.205, 7 0.214, 8 0.238, 9 0.252, 10 0.259, 11 0.271, 12 0.273, 13 0.287, 14 0.298, 15 0.300, 16 0.304, 17 0.310; klane: 0 0.571, 1 0.544, 2 0.548, 3 0.543, 4 0.549, 5 0.559, 6 0.562, 7 0.580, 8 0.598, 9 0.593, 10 0.605, 11 0.607, 12 0.615, 13 0.629, 14 0.632, 15 0.598, 16 0.601, 17 0.612

Items trained (K0: active on >= 0.2 % of the served plans, on <= 2 % of the logged futures): ['kcorr']

All gates pass.
