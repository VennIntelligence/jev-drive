# ONPOL-LONG pool gates, rows `L18P`: PASS

R1 as amended (one fp16 ulp or 0.03 m on the first 15 plan points, >= 0.99): {'P2H10S-F-s0': 0.9991341991341991}; interpreted control [[100, 1.0]]; wrong-seed control (8 poses within 0.03 m, < 0.20): {'P2H10S-F-s0': 0.016565656565656565}. R2: ego at the log-replay decisions max 0.018 m (<= 0.3); actor samples within 0.5 m 0.99998 of 377414 (>= 0.9999). Rollouts without a time alignment to the navsim log (frame error > 0.1 m; Amendment 2), no rows: [{'scene': '2021.09.09.17.18.51_veh-48_00657_00876-798842f0b1c253b4', 'tag': 'P2H10S-F-s0', 'frame_err': 2.1068653683766074, 't0_rel': -2.0}].

Lane raster present on 0.9999 of the rows.

| term (served plans) | active rows | share | scenes | mean | mean when active | along-heading gradient max | share active on the log's own future |
|:--|--:|--:|--:|--:|--:|--:|--:|
| kcorr | 3765 | 0.2173 | 513 | 0.2232 | 1.027 | 7.5e-09 | 0.0060 |
| klane | 10507 | 0.6065 | 921 | 0.1768 | 0.292 | 7.5e-09 | 0.4586 |

Active share by decision index k: kcorr: 0 0.134, 1 0.084, 2 0.096, 3 0.145, 4 0.160, 5 0.189, 6 0.221, 7 0.236, 8 0.232, 9 0.235, 10 0.252, 11 0.258, 12 0.269, 13 0.270, 14 0.292, 15 0.297, 16 0.304, 17 0.305; klane: 0 0.586, 1 0.559, 2 0.556, 3 0.568, 4 0.572, 5 0.584, 6 0.613, 7 0.612, 8 0.616, 9 0.617, 10 0.622, 11 0.638, 12 0.635, 13 0.632, 14 0.639, 15 0.639, 16 0.623, 17 0.631

Items trained (K0: active on >= 0.2 % of the served plans, on <= 2 % of the logged futures): ['kcorr']

All gates pass.
