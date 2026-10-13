# ONPOL-LONG pool gates, rows `L18RBSG2allheld`: PASS

R1 as amended (one fp16 ulp or 0.03 m on the first 15 plan points, >= 0.99): {'RBSG2-L18-s0': 0.9984717269485481}; interpreted control [[100, 1.0]]; wrong-seed control (8 poses within 0.03 m, < 0.20): {'RBSG2-L18-s0': 0.026235354049923585}. R2: ego at the log-replay decisions max 0.052 m (<= 0.3); actor samples within 0.5 m 1.00000 of 130660 (>= 0.9999). Rollouts without a time alignment to the navsim log (frame error > 0.1 m; Amendment 2), no rows: [{'scene': '2021.06.03.12.02.06_veh-35_00233_00609-be7331d3f05e5d16', 'tag': 'RBSG2-L18-s0', 'frame_err': 0.10109809264760687, 't0_rel': -1.5}, {'scene': '2021.06.03.17.06.58_veh-35_02943_03220-c2736d03415f515b', 'tag': 'RBSG2-L18-s0', 'frame_err': 0.10820006840263371, 't0_rel': -1.5}, {'scene': '2021.06.28.15.02.02_veh-38_02398_02848-0c8b9f08bfa25dc2', 'tag': 'RBSG2-L18-s0', 'frame_err': 0.19546061620398686, 't0_rel': -1.5}, {'scene': '2021.06.28.15.02.02_veh-38_02398_02848-9bfa838de21f5d25', 'tag': 'RBSG2-L18-s0', 'frame_err': 0.19909840986653246, 't0_rel': -1.5}, {'scene': '2021.06.28.15.02.02_veh-38_02398_02848-a4c23fa815945fe3', 'tag': 'RBSG2-L18-s0', 'frame_err': 0.15960321072664893, 't0_rel': -1.5}].

Lane raster present on 1.0000 of the rows.

| term (served plans) | active rows | share | scenes | mean | mean when active | along-heading gradient max | share active on the log's own future |
|:--|--:|--:|--:|--:|--:|--:|--:|
| kcorr | 965 | 0.1229 | 154 | 0.0801 | 0.651 | 7.5e-09 | 0.0062 |
| klane | 3892 | 0.4957 | 367 | 0.1039 | 0.210 | 7.5e-09 | 0.4400 |

Active share by decision index k: kcorr: 0 0.083, 1 0.049, 2 0.076, 3 0.094, 4 0.101, 5 0.110, 6 0.132, 7 0.146, 8 0.147, 9 0.161, 10 0.149, 11 0.143, 12 0.146, 13 0.150, 14 0.142, 15 0.119, 16 0.133, 17 0.139; klane: 0 0.550, 1 0.537, 2 0.512, 3 0.506, 4 0.497, 5 0.519, 6 0.508, 7 0.512, 8 0.519, 9 0.525, 10 0.506, 11 0.493, 12 0.459, 13 0.472, 14 0.455, 15 0.449, 16 0.434, 17 0.454

Items trained (K0: active on >= 0.2 % of the served plans, on <= 2 % of the logged futures): ['kcorr']

All gates pass.
