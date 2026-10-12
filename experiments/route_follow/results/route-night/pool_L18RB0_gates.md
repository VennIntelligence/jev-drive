# ONPOL-LONG pool gates, rows `L18RB0`: PASS

R1 as amended (one fp16 ulp or 0.03 m on the first 15 plan points, >= 0.99): {'RB0-F-s0': 0.9987377359573125}; interpreted control [[100, 1.0]]; wrong-seed control (8 poses within 0.03 m, < 0.20): {'RB0-F-s0': 0.013770153192954271}. R2: ego at the log-replay decisions max 0.018 m (<= 0.3); actor samples within 0.5 m 0.99998 of 377414 (>= 0.9999). Rollouts without a time alignment to the navsim log (frame error > 0.1 m; Amendment 2), no rows: [{'scene': '2021.09.09.17.18.51_veh-48_00657_00876-798842f0b1c253b4', 'tag': 'RB0-F-s0', 'frame_err': 2.1068653683766074, 't0_rel': -2.0}].

Lane raster present on 1.0000 of the rows.

| term (served plans) | active rows | share | scenes | mean | mean when active | along-heading gradient max | share active on the log's own future |
|:--|--:|--:|--:|--:|--:|--:|--:|
| kcorr | 3066 | 0.1759 | 401 | 0.1498 | 0.852 | 7.5e-09 | 0.0058 |
| klane | 9739 | 0.5588 | 882 | 0.1501 | 0.269 | 7.5e-09 | 0.4603 |

Active share by decision index k: kcorr: 0 0.093, 1 0.057, 2 0.065, 3 0.092, 4 0.122, 5 0.146, 6 0.163, 7 0.181, 8 0.193, 9 0.207, 10 0.215, 11 0.219, 12 0.231, 13 0.236, 14 0.241, 15 0.254, 16 0.253, 17 0.259; klane: 0 0.563, 1 0.539, 2 0.536, 3 0.525, 4 0.540, 5 0.543, 6 0.540, 7 0.541, 8 0.551, 9 0.559, 10 0.567, 11 0.585, 12 0.568, 13 0.581, 14 0.593, 15 0.594, 16 0.578, 17 0.576

Items trained (K0: active on >= 0.2 % of the served plans, on <= 2 % of the logged futures): ['kcorr']

All gates pass.
