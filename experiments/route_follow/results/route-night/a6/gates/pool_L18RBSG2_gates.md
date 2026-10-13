# ONPOL-LONG pool gates, rows `L18RBSG2`: PASS

R1 as amended (one fp16 ulp or 0.03 m on the first 15 plan points, >= 0.99): {'RBSG2-L18-s0': 0.9993150684931507}; interpreted control [[100, 1.0]]; wrong-seed control (8 poses within 0.03 m, < 0.20): {'RBSG2-L18-s0': 0.026883561643835617}. R2: ego at the log-replay decisions max 0.018 m (<= 0.3); actor samples within 0.5 m 0.99998 of 377414 (>= 0.9999). Rollouts without a time alignment to the navsim log (frame error > 0.1 m; Amendment 2), no rows: [{'scene': '2021.09.09.17.18.51_veh-48_00657_00876-798842f0b1c253b4', 'tag': 'RBSG2-L18-s0', 'frame_err': 2.1068653683766074, 't0_rel': -2.0}].

Lane raster present on 1.0000 of the rows.

| term (served plans) | active rows | share | scenes | mean | mean when active | along-heading gradient max | share active on the log's own future |
|:--|--:|--:|--:|--:|--:|--:|--:|
| kcorr | 2298 | 0.1312 | 334 | 0.0774 | 0.590 | 7.5e-09 | 0.0063 |
| klane | 8437 | 0.4816 | 809 | 0.0992 | 0.206 | 7.5e-09 | 0.4588 |

Active share by decision index k: kcorr: 0 0.098, 1 0.082, 2 0.080, 3 0.098, 4 0.108, 5 0.113, 6 0.123, 7 0.137, 8 0.144, 9 0.145, 10 0.152, 11 0.154, 12 0.151, 13 0.149, 14 0.151, 15 0.165, 16 0.167, 17 0.166; klane: 0 0.552, 1 0.528, 2 0.517, 3 0.517, 4 0.508, 5 0.494, 6 0.486, 7 0.491, 8 0.480, 9 0.456, 10 0.462, 11 0.456, 12 0.446, 13 0.450, 14 0.447, 15 0.446, 16 0.460, 17 0.449

Items trained (K0: active on >= 0.2 % of the served plans, on <= 2 % of the logged futures): ['kcorr']

All gates pass.
