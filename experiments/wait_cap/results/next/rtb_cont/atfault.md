# At-fault collisions on held 460 @18: RTB1O-L18-s0 vs RSG-L18-s0

| scene | log | at-fault J / C | contact J (s) | class J | struck (label, x / y m at contact) | J speed 2 s before / at contact | C contact (s) | C dist (m) | J dist (m) | human dist (m) |
|---|---|:-:|--:|---|---|--:|--:|--:|--:|--:|
| 7bbb6cfc4f135e50 | 2021.06.03.12.02.06_veh-35 | - / Y | - | no contact | - | - | 9.0 | 39.4 | 39.3 | 50.1 |
| 0722adecc3d45ac4 | 2021.06.28.16.57.59_veh-26 | Y / Y | 5.0 | other | VEHICLE, 1.6 / 2.2 | 6.7 / 3.9 | 5.0 | 40.2 | 39.9 | 92.3 |
| de4ace24443b5b2d | 2021.06.28.16.57.59_veh-26 | Y / - | 6.5 | other | VEHICLE, 1.6 / 2.4 | 2.4 / 1.9 | 6.5 | 18.9 | 14.6 | 52.8 |
| daf38cecd5045cf6 | 2021.09.09.18.04.06_veh-40 | Y / Y | 5.5 | other | VEHICLE, 3.7 / 0.9 | 4.7 / 6.1 | 5.5 | 16.8 | 17.3 | 25.7 |
| 6d2b59f78d995d9c | 2021.09.09.18.04.06_veh-40 | Y / - | 6.0 | other | VEHICLE, 5.1 / -1.7 | 4.5 / 8.4 | - | 41.5 | 18.9 | 41.2 |
| a3c4e4bf10d65ddb | 2021.09.16.14.39.34_veh-42 | Y / - | 6.5 | other | VEHICLE, 5.2 / 0.4 | 5.1 / 2.3 | - | 14.2 | 19.0 | 27.2 |
| 365c0937de9d5885 | 2021.09.16.15.47.30_veh-45 | Y / - | 5.5 | other | VEHICLE, 3.8 / 2.2 | 4.4 / 6.9 | - | 38.5 | 19.2 | 38.5 |
| 47e0ff3bc36a5f69 | 2021.09.16.17.40.09_veh-45 | Y / Y | 7.5 | other | VEHICLE, 2.5 / 3.2 | 5.9 / 5.3 | 8.0 | 27.8 | 25.7 | 24.7 |
| 9a675656c3c85f4c | 2021.09.16.19.49.00_veh-42 | Y / - | 6.0 | other | VEHICLE, 4.1 / -0.6 | 3.5 / 5.6 | - | 24.7 | 16.5 | 21.7 |
| 84b38cdaa93e5dcd | 2021.10.06.08.16.17_veh-52 | Y / - | 7.0 | other | VEHICLE, 4.9 / -0.5 | 1.5 / 1.7 | 6.5 | 37.1 | 32.1 | 77.7 |

## Geometry against the scene map (per contact)

| scene | tag | struck: dpsi (deg) / kind / speed (m/s) | struck's lane vs ego heading (deg; faces ego) | ego lane at contact (max lane gap deg) | 2 s before: opposite / off-lane share | ego offset from human (m) | other tag at same station: offset / lane |
|---|---|---|---|---|--:|--:|---|
| 7bbb6cfc4f135e50 | RSG-L18-s0 | -8 / same direction / 1.3 | 2 (no) | own (2) | 0.00 / 0.00 | +0.10 | -0.00 / own |
| 0722adecc3d45ac4 | RTB1O-L18-s0 | -6 / same direction / 7.1 | 0 (no) | own (1) | 0.00 / 0.00 | -0.08 | -0.14 / own |
| 0722adecc3d45ac4 | RSG-L18-s0 | -7 / same direction / 7.1 | 1 (no) | own (0) | 0.00 / 0.00 | -0.14 | -0.09 / own |
| de4ace24443b5b2d | RTB1O-L18-s0 | -15 / same direction / 2.6 | 0 (no) | own (0) | 0.00 / 0.00 | +0.15 | +0.07 / own |
| de4ace24443b5b2d | RSG-L18-s0 | - (asl pruned) | - | own (7) | 0.00 / 0.00 | +0.25 | +0.11 / own |
| daf38cecd5045cf6 | RTB1O-L18-s0 | 50 / crossing / 3.8 | 40 (no) | opposite (92) | 1.00 / 0.00 | -2.15 | -2.64 / opposite |
| daf38cecd5045cf6 | RSG-L18-s0 | 54 / crossing / 3.8 | 44 (no) | opposite (96) | 0.95 / 0.00 | -2.52 | -2.06 / opposite |
| 6d2b59f78d995d9c | RTB1O-L18-s0 | 5 / same direction / 9.2 | 4 (no) | own (4) | 0.00 / 0.00 | -0.77 | -1.00 / own |
| a3c4e4bf10d65ddb | RTB1O-L18-s0 | -16 / same direction / 0.0 | 18 (no) | own (18) | 0.00 / 0.00 | +1.32 | -1.05 / own |
| 365c0937de9d5885 | RTB1O-L18-s0 | 8 / same direction / 0.1 | 6 (no) | own (6) | 0.57 / 0.05 | +1.77 | +0.23 / own |
| 47e0ff3bc36a5f69 | RTB1O-L18-s0 | -177 / head-on / 0.1 | 175 (yes) | opposite (175) | 1.00 / 0.00 | +5.90 | +6.74 / opposite |
| 47e0ff3bc36a5f69 | RSG-L18-s0 | -177 / head-on / 5.5 | 176 (yes) | opposite (176) | 1.00 / 0.00 | +6.26 | +8.55 / opposite |
| 9a675656c3c85f4c | RTB1O-L18-s0 | 147 / head-on / 1.2 | 148 (yes) | opposite (148) | 0.38 / 0.00 | +3.01 | +0.71 / own |
| 84b38cdaa93e5dcd | RTB1O-L18-s0 | -98 / crossing / 5.3 | 1 (no) | own (2) | 0.00 / 0.00 | -0.26 | -0.05 / own |
| 84b38cdaa93e5dcd | RSG-L18-s0 | - (asl pruned) | - | own (1) | 0.00 / 0.00 | -0.03 | -0.36 / own |
