# ONPOL-LONG pool gates, rows `L18RSG`: PASS

R1 as amended (one fp16 ulp or 0.03 m on the first 15 plan points, >= 0.99): {'RSG-L18-s0': 0.9989126702529473}; interpreted control [[100, 1.0]]; wrong-seed control (8 poses within 0.03 m, < 0.20): {'RSG-L18-s0': 0.10203731257868834}. R2: ego at the log-replay decisions max 0.018 m (<= 0.3); actor samples within 0.5 m 0.99998 of 377414 (>= 0.9999). Rollouts without a time alignment to the navsim log (frame error > 0.1 m; Amendment 2), no rows: [{'scene': '2021.09.09.17.18.51_veh-48_00657_00876-798842f0b1c253b4', 'tag': 'RSG-L18-s0', 'frame_err': 2.1068653683766074, 't0_rel': -2.0}].

Lane raster present on 1.0000 of the rows.

| term (served plans) | active rows | share | scenes | mean | mean when active | along-heading gradient max | share active on the log's own future |
|:--|--:|--:|--:|--:|--:|--:|--:|
| kcorr | 2565 | 0.1468 | 347 | 0.1032 | 0.703 | 7.5e-09 | 0.0060 |
| klane | 9378 | 0.5367 | 843 | 0.1370 | 0.255 | 7.5e-09 | 0.4597 |

Active share by decision index k: kcorr: 0 0.094, 1 0.057, 2 0.063, 3 0.080, 4 0.097, 5 0.118, 6 0.126, 7 0.142, 8 0.153, 9 0.159, 10 0.173, 11 0.185, 12 0.195, 13 0.197, 14 0.208, 15 0.215, 16 0.210, 17 0.222; klane: 0 0.550, 1 0.529, 2 0.527, 3 0.515, 4 0.526, 5 0.531, 6 0.531, 7 0.527, 8 0.525, 9 0.526, 10 0.519, 11 0.523, 12 0.534, 13 0.543, 14 0.570, 15 0.563, 16 0.568, 17 0.568

Items trained (K0: active on >= 0.2 % of the served plans, on <= 2 % of the logged futures): ['kcorr']

All gates pass.
