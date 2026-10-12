# ONPOL-LONG pool gates, rows `L18YS`: PASS

R1 as amended (one fp16 ulp or 0.03 m on the first 15 plan points, >= 0.99): {'YS-L18-s0': 0.9986187050359713, 'YS-L18-s1': 0.9991385746281514}; interpreted control [[100, 1.0]]; wrong-seed control (8 poses within 0.03 m, < 0.20): {'YS-L18-s0': 0.025553956834532373, 'YS-L18-s1': 0.030494458163441106}. R2: ego at the log-replay decisions max 0.018 m (<= 0.3); actor samples within 0.5 m 0.99998 of 754828 (>= 0.9999). Rollouts without a time alignment to the navsim log (frame error > 0.1 m; Amendment 2), no rows: [{'scene': '2021.09.09.17.18.51_veh-48_00657_00876-798842f0b1c253b4', 'tag': 'YS-L18-s0', 'frame_err': 2.1068653683766074, 't0_rel': -2.0}, {'scene': '2021.09.09.17.18.51_veh-48_00657_00876-798842f0b1c253b4', 'tag': 'YS-L18-s1', 'frame_err': 2.1068653683766074, 't0_rel': -2.0}].

Lane raster present on 0.9999 of the rows.

| term (served plans) | active rows | share | scenes | mean | mean when active | along-heading gradient max | share active on the log's own future |
|:--|--:|--:|--:|--:|--:|--:|--:|
| kcorr | 7017 | 0.2017 | 573 | 0.1597 | 0.792 | 7.5e-09 | 0.0043 |
| klane | 19835 | 0.5702 | 920 | 0.1647 | 0.289 | 7.5e-09 | 0.4610 |

Active share by decision index k: kcorr: 0 0.095, 1 0.068, 2 0.092, 3 0.117, 4 0.161, 5 0.171, 6 0.183, 7 0.203, 8 0.208, 9 0.223, 10 0.237, 11 0.254, 12 0.265, 13 0.264, 14 0.268, 15 0.279, 16 0.299, 17 0.317; klane: 0 0.547, 1 0.523, 2 0.521, 3 0.543, 4 0.544, 5 0.559, 6 0.551, 7 0.562, 8 0.573, 9 0.582, 10 0.593, 11 0.589, 12 0.594, 13 0.598, 14 0.594, 15 0.603, 16 0.608, 17 0.606

Items trained (K0: active on >= 0.2 % of the served plans, on <= 2 % of the logged futures): ['kcorr']

All gates pass.
