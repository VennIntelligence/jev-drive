# PAI-HOLD-DIAG per-rollout tables (ACC-GAP, decision 330 stage 2)

Rollouts 281; zeros 76; by failure: left_corridor_laterally 53, offroad 14, collision_at_fault 9

## Corridor zeros by class

| class | n | median ego v at trip (m/s) | median human v at handoff | median ego past rest (m) |
|---|--:|--:|--:|--:|
| A still from handoff to end (leash3 latch) | 1 | 3.8 | 0.0 | +4.1 |
| B braking to a stop at handoff (stop3 latch) | 1 | 0.1 | 3.7 | +4.0 |
| C human stops mid-scene, ego rolls past (no latch) | 18 | 11.0 | 10.8 | +4.0 |
| D turn / turn exit | 20 | 6.1 | 8.2 | -43.2 |
| E high-speed lateral drift | 4 | 34.5 | 35.2 | -210.9 |
| F other (moving human, mid-path, < 20 m/s, straight) | 9 | 15.5 | 12.1 | -51.6 |

## Corridor zeros (53)

| id | score | fail | cls | v_call | a_call | a_call5 | latch_leash | latch_stop | cap | h_disp_max | h_rest_from_xy0 | h_v_end | h_t_stop | h_still_end | gt_len | t_trip | ego_v_trip | ego_past_rest | ego_v_at_hstop | ego_gap_at_hstop | turn_deg | gt_frac_trip | ego_maxd_xy0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3fe2bac8 | 0.00 | left_corridor_laterally | A still from handoff to end (leash3 latch) | 0.00 | -0.07 | -0.01 | True | False | 3.00 | 0.02 | 0.02 | 0.00 | 0.24 | True | 0.19 | 5.26 | 3.75 | 4.11 | 0.29 | -0.02 | 0.75 | 0.02 | 18.65 |
| 2bbf1efc | 0.00 | left_corridor_laterally | B braking to a stop at handoff (stop3 latch) | 3.73 | -1.34 | -1.18 | False | True | 8.18 | 5.80 | 5.80 | 0.00 | 4.55 | True | 13.35 | 18.24 | 0.15 | 4.02 | 0.95 | -0.59 | 0.21 | 0.25 | 9.99 |
| 054b5901 | 0.00 | left_corridor_laterally | C human stops mid-scene, ego rolls past (no latch) | 14.80 | 0.42 | -0.05 | False | False | nan | 108.13 | 108.11 | 0.00 | 13.67 | True | 132.34 | 9.20 | 14.76 | 3.95 | 14.71 | -68.78 | 1.54 | 0.71 | 264.68 |
| 127c4f85 | 0.00 | left_corridor_laterally | C human stops mid-scene, ego rolls past (no latch) | 10.88 | -0.16 | -0.11 | False | False | nan | 93.97 | 93.96 | 0.00 | 12.81 | True | 111.50 | 11.83 | 3.81 | 4.31 | 2.24 | -7.23 | 0.22 | 0.65 | 106.12 |
| 2466266e | 0.00 | left_corridor_laterally | C human stops mid-scene, ego rolls past (no latch) | 10.80 | 0.29 | 0.24 | False | False | nan | 92.06 | 92.06 | 0.00 | 15.21 | True | 109.70 | 8.43 | 8.27 | -30.73 | 3.98 | -17.63 | 21.45 | 0.42 | 110.42 |
| 444fca69 | 0.00 | left_corridor_laterally | C human stops mid-scene, ego rolls past (no latch) | 8.81 | 0.73 | 0.64 | False | False | nan | 91.45 | 91.45 | 0.00 | 15.60 | True | 104.82 | 12.46 | 8.39 | 4.63 | 4.54 | -20.56 | 0.01 | 0.97 | 112.62 |
| 44df49fe | 0.00 | left_corridor_laterally | C human stops mid-scene, ego rolls past (no latch) | 10.76 | -0.20 | -0.07 | False | False | nan | 65.33 | 65.33 | 0.00 | 11.30 | True | 82.58 | 8.02 | 12.53 | 5.03 | 9.15 | -44.22 | 1.14 | 0.68 | 148.88 |
| 47db2eb6 | 0.00 | left_corridor_laterally | C human stops mid-scene, ego rolls past (no latch) | 8.73 | 0.29 | 0.45 | False | False | nan | 104.31 | 104.31 | 0.00 | 13.60 | True | 117.58 | 14.55 | 6.21 | 4.13 | 6.08 | 1.45 | 0.01 | 0.84 | 133.17 |
| 4cb7f7ac | 0.00 | left_corridor_laterally | C human stops mid-scene, ego rolls past (no latch) | 10.73 | -0.45 | -0.02 | False | False | nan | 92.55 | 92.54 | 0.00 | 15.86 | True | 109.74 | 10.37 | 10.89 | 4.94 | 9.40 | -62.68 | 0.03 | 0.91 | 182.62 |
| 4ecc114d | 0.00 | left_corridor_laterally | C human stops mid-scene, ego rolls past (no latch) | 13.61 | 0.89 | 0.17 | False | False | nan | 204.43 | 204.43 | 0.02 | 19.27 | True | 226.19 | 16.48 | 14.54 | 3.22 | 13.73 | -41.73 | 9.11 | 0.98 | 256.24 |
| 559d4ad6 | 0.00 | left_corridor_laterally | C human stops mid-scene, ego rolls past (no latch) | 12.20 | -0.02 | -0.45 | False | False | nan | 56.61 | 56.61 | 0.00 | 10.59 | True | 76.61 | 6.71 | 12.86 | 4.94 | 12.21 | -59.48 | 0.01 | 0.70 | 227.88 |
| 62cfe9d3 | 0.00 | left_corridor_laterally | C human stops mid-scene, ego rolls past (no latch) | 10.77 | 0.02 | 0.06 | False | False | nan | 104.30 | 104.29 | 0.01 | 17.08 | True | 131.79 | 11.14 | 7.96 | -22.13 | 11.30 | -18.12 | 69.48 | 0.55 | 168.43 |
| 68c3be44 | 0.00 | left_corridor_laterally | C human stops mid-scene, ego rolls past (no latch) | 13.24 | -0.17 | -0.52 | False | False | nan | 81.22 | 81.22 | 0.00 | 11.92 | True | 103.14 | 8.04 | 15.65 | 3.48 | 13.23 | -57.30 | 0.11 | 0.61 | 242.53 |
| 7750265d | 0.00 | left_corridor_laterally | C human stops mid-scene, ego rolls past (no latch) | 19.57 | 0.92 | -0.42 | False | False | nan | 197.35 | 197.33 | 0.00 | 15.78 | True | 229.23 | 14.68 | 13.34 | 3.39 | 14.64 | -18.79 | 2.32 | 0.80 | 287.41 |
| 7d994ed3 | 0.00 | left_corridor_laterally | C human stops mid-scene, ego rolls past (no latch) | 10.96 | -0.19 | -0.13 | False | False | nan | 80.62 | 80.62 | 0.00 | 12.83 | True | 98.44 | 9.25 | 11.03 | 2.68 | 13.44 | -46.51 | 0.20 | 0.67 | 206.26 |
| 9b6c68b4 | 0.00 | left_corridor_laterally | C human stops mid-scene, ego rolls past (no latch) | 11.53 | -0.36 | -0.05 | False | False | nan | 73.53 | 73.52 | 0.00 | 17.52 | True | 93.03 | 15.55 | 1.87 | -6.23 | 5.04 | -0.55 | 34.25 | 0.47 | 84.02 |
| 9f8acd00 | 0.00 | left_corridor_laterally | C human stops mid-scene, ego rolls past (no latch) | 13.88 | 0.35 | 0.27 | False | False | nan | 141.84 | 141.84 | 0.00 | 16.28 | True | 163.94 | 11.51 | 14.66 | 3.99 | 13.98 | -71.34 | 1.19 | 0.83 | 263.52 |
| ba40be62 | 0.00 | left_corridor_laterally | C human stops mid-scene, ego rolls past (no latch) | 10.64 | -0.17 | -0.04 | False | False | nan | 50.95 | 50.94 | 0.00 | 8.41 | True | 68.08 | 7.01 | 8.53 | 4.71 | 6.38 | -14.92 | 1.32 | 0.44 | 72.94 |
| bc309fd6 | 0.00 | left_corridor_laterally | C human stops mid-scene, ego rolls past (no latch) | 10.49 | -1.06 | -1.40 | False | False | nan | 27.85 | 27.85 | 0.00 | 6.70 | True | 46.44 | 4.66 | 12.42 | 5.02 | 14.78 | -32.27 | 0.01 | 0.94 | 213.22 |
| fc9fde87 | 0.00 | left_corridor_laterally | C human stops mid-scene, ego rolls past (no latch) | 9.35 | 0.08 | -0.08 | False | False | nan | 122.92 | 122.91 | 0.00 | 18.08 | True | 138.07 | 14.99 | 5.42 | 3.64 | 8.77 | -13.17 | 5.88 | 0.91 | 138.60 |
| 00fabf60 | 0.00 | left_corridor_laterally | D turn / turn exit | 9.38 | -2.09 | -2.18 | False | False | nan | 123.59 | 123.59 | 6.95 | 20.00 | False | 152.35 | 14.44 | 9.09 | -26.83 | 5.86 | 41.44 | 76.02 | 0.81 | 130.26 |
| 095cf563 | 0.00 | left_corridor_laterally | D turn / turn exit | 9.77 | 0.92 | 1.05 | False | False | nan | 127.30 | 127.30 | 7.90 | 20.00 | False | 142.55 | 17.50 | 4.61 | -0.50 | 4.95 | -10.71 | 36.61 | 1.00 | 139.02 |
| 1e8e939e | 0.00 | left_corridor_laterally | D turn / turn exit | 14.36 | -2.37 | -0.84 | False | False | nan | 157.66 | 157.66 | 17.24 | 20.00 | False | 198.30 | 12.74 | 6.89 | -86.89 | 15.56 | 9.91 | 61.03 | 0.68 | 154.88 |
| 26004ca8 | 0.00 | left_corridor_laterally | D turn / turn exit | 27.42 | -2.23 | -0.55 | False | False | nan | 455.60 | 455.60 | 24.62 | 20.00 | False | 521.79 | 6.24 | 20.01 | -271.50 | 10.27 | 190.43 | 35.36 | 0.27 | 238.63 |
| 2f227f2c | 0.00 | left_corridor_laterally | D turn / turn exit | 1.07 | 1.55 | 1.14 | False | False | nan | 62.14 | 62.14 | 3.86 | 20.00 | False | 72.12 | 6.94 | 9.80 | -44.25 | 0.04 | 9.96 | 63.55 | 0.43 | 114.45 |
| 302c5c99 | 0.00 | left_corridor_laterally | D turn / turn exit | 4.27 | 0.83 | 0.92 | False | False | nan | 92.06 | 92.06 | 18.52 | 20.00 | False | 160.65 | 8.43 | 4.32 | -121.06 | 4.40 | 191.94 | 162.75 | 0.38 | 102.41 |
| 3232fdb7 | 0.00 | left_corridor_laterally | D turn / turn exit | 7.92 | 1.06 | 0.78 | False | False | nan | 100.95 | 100.95 | 13.14 | 20.00 | False | 135.41 | 15.23 | 7.01 | -31.60 | 12.43 | -14.45 | 80.79 | 0.86 | 113.56 |
| 403d558a | 0.00 | left_corridor_laterally | D turn / turn exit | 9.76 | -1.10 | -1.08 | False | False | nan | 47.22 | 47.22 | 3.97 | 20.00 | False | 79.49 | 4.55 | 6.41 | -38.56 | 3.99 | 12.01 | 94.10 | 0.25 | 115.00 |
| 52003cee | 0.00 | left_corridor_laterally | D turn / turn exit | 6.09 | 0.72 | 0.45 | False | False | nan | 107.27 | 107.27 | 4.87 | 20.00 | False | 118.72 | 17.24 | 4.11 | -3.03 | 5.68 | -9.04 | 65.59 | 0.98 | 114.08 |
| 98f88a4e | 0.00 | left_corridor_laterally | D turn / turn exit | 13.14 | -0.19 | -0.10 | False | False | nan | 329.13 | 329.13 | 22.55 | 20.00 | False | 358.00 | 7.13 | 13.23 | -268.81 | 14.10 | 101.74 | 31.65 | 0.34 | 228.44 |
| a4d2f7cf | 0.00 | left_corridor_laterally | D turn / turn exit | 8.43 | 0.59 | 0.58 | False | False | nan | 73.92 | 73.92 | 2.55 | 20.00 | False | 114.88 | 10.03 | 5.15 | -35.85 | 1.86 | 3.84 | 85.44 | 0.58 | 78.49 |
| abd45a30 | 0.00 | left_corridor_laterally | D turn / turn exit | 12.31 | -0.31 | -0.18 | False | False | nan | 118.55 | 118.55 | 8.57 | 20.00 | False | 167.93 | 12.93 | 5.41 | -38.36 | 5.91 | 21.67 | 93.14 | 0.70 | 121.17 |
| ddc3e8df | 0.00 | left_corridor_laterally | D turn / turn exit | 2.87 | 0.52 | 0.70 | False | False | nan | 213.33 | 213.33 | 14.95 | 20.00 | False | 220.14 | 5.84 | 9.30 | -191.05 | 8.23 | 55.87 | 44.40 | 0.29 | 157.73 |
| de9db5dc | 0.00 | left_corridor_laterally | D turn / turn exit | 0.02 | -0.17 | -0.52 | True | False | 3.00 | 37.57 | 37.57 | 0.63 | 20.00 | False | 40.99 | 15.65 | 3.48 | -12.64 | 6.83 | -1.09 | 54.90 | 0.65 | 40.86 |
| e0f6ebe1 | 0.00 | left_corridor_laterally | D turn / turn exit | 6.53 | -0.54 | -0.65 | False | False | nan | 73.88 | 73.88 | 5.32 | 20.00 | False | 100.70 | 15.61 | 5.43 | -10.89 | 10.65 | -24.50 | 45.76 | 0.86 | 87.83 |
| e37aa159 | 0.00 | left_corridor_laterally | D turn / turn exit | 0.76 | -2.31 | -2.53 | False | True | 3.12 | 76.71 | 76.71 | 12.96 | 20.00 | False | 103.38 | 15.76 | 8.29 | -42.10 | 12.27 | -2.48 | 77.61 | 0.81 | 83.57 |
| e5e47fd2 | 0.00 | left_corridor_laterally | D turn / turn exit | 10.48 | 0.87 | 1.75 | False | False | nan | 58.27 | 37.77 | 1.85 | 20.00 | False | 145.51 | 6.75 | 6.50 | -48.48 | 10.81 | 182.16 | 88.97 | 0.34 | 181.48 |
| eaba3f6a | 0.00 | left_corridor_laterally | D turn / turn exit | 9.46 | -0.74 | -0.88 | False | False | nan | 90.12 | 90.12 | 10.66 | 20.00 | False | 133.04 | 7.03 | 4.96 | -76.40 | 7.55 | 58.39 | 82.21 | 0.53 | 120.92 |
| f9e361aa | 0.00 | left_corridor_laterally | D turn / turn exit | 8.02 | 0.62 | 0.64 | False | False | nan | 117.02 | 117.02 | 7.63 | 20.00 | False | 156.09 | 9.34 | 5.78 | -90.18 | 12.57 | -22.27 | 77.86 | 0.47 | 137.64 |
| fd01e9ab | 0.00 | left_corridor_laterally | D turn / turn exit | 0.01 | 0.00 | 0.00 | True | False | 3.00 | 226.39 | 226.39 | 25.73 | 20.00 | False | 235.73 | 7.34 | 4.77 | -217.21 | 0.33 | 183.51 | 99.11 | 0.43 | 42.83 |
| 0e05cb0c | 0.00 | left_corridor_laterally | E high-speed lateral drift | 32.45 | 0.57 | 0.45 | False | False | nan | 603.05 | 603.05 | 32.44 | 20.00 | False | 654.55 | 15.04 | 32.50 | -166.38 | 32.50 | 7.15 | 0.77 | 0.74 | 595.88 |
| 0feb3afa | 0.00 | left_corridor_laterally | E high-speed lateral drift | 29.52 | -0.09 | -0.53 | False | False | nan | 470.45 | 470.45 | 24.53 | 20.00 | False | 518.85 | 11.67 | 26.95 | -189.76 | 26.14 | -46.21 | 0.36 | 0.62 | 516.68 |
| 32be03e4 | 0.00 | left_corridor_laterally | E high-speed lateral drift | 37.99 | -0.38 | -0.34 | False | False | nan | 666.12 | 666.12 | 35.53 | 20.00 | False | 727.56 | 13.06 | 36.47 | -231.94 | 34.97 | 3.66 | 3.34 | 0.68 | 662.28 |
| 4cb05d65 | 0.00 | left_corridor_laterally | E high-speed lateral drift | 38.54 | -0.01 | -0.08 | False | False | nan | 703.26 | 703.26 | 38.23 | 20.00 | False | 765.16 | 13.24 | 38.53 | -254.59 | 38.53 | -3.25 | 1.86 | 0.66 | 706.32 |
| 098e2482 | 0.00 | left_corridor_laterally | F other (moving human, mid-path, < 20 m/s, straight) | 2.32 | -0.01 | 0.02 | False | False | nan | 43.50 | 43.50 | 1.85 | 20.00 | False | 47.20 | 12.81 | 6.31 | -14.44 | 2.36 | -38.11 | 1.11 | 0.65 | 81.62 |
| 13d646af | 0.00 | left_corridor_laterally | F other (moving human, mid-path, < 20 m/s, straight) | 16.59 | -0.06 | 0.12 | False | False | nan | 268.29 | 268.29 | 7.23 | 20.00 | False | 300.42 | 14.83 | 15.64 | -51.58 | 6.98 | -6.10 | 24.65 | 0.76 | 273.23 |
| 2d10a08d | 0.00 | left_corridor_laterally | F other (moving human, mid-path, < 20 m/s, straight) | 14.96 | 0.22 | -0.04 | False | False | nan | 162.80 | 162.79 | 0.01 | 16.02 | False | 187.01 | 12.74 | 16.02 | 4.67 | 14.94 | -54.83 | 0.17 | 0.79 | 275.85 |
| 35e082a4 | 0.00 | left_corridor_laterally | F other (moving human, mid-path, < 20 m/s, straight) | 7.43 | -1.91 | -1.93 | False | False | nan | 17.21 | 17.20 | 0.01 | 6.64 | False | 32.10 | 4.77 | 6.93 | 4.55 | 6.92 | -17.69 | 0.08 | 0.91 | 119.04 |
| 6e6cbe67 | 0.00 | left_corridor_laterally | F other (moving human, mid-path, < 20 m/s, straight) | 21.67 | 0.32 | 0.34 | False | False | nan | 426.89 | 426.89 | 22.28 | 20.00 | False | 461.40 | 15.52 | 16.69 | -225.66 | 19.16 | 149.70 | 2.15 | 0.51 | 277.18 |
| 7cfa81de | 0.00 | left_corridor_laterally | F other (moving human, mid-path, < 20 m/s, straight) | 10.62 | 0.36 | 0.08 | False | False | nan | 205.51 | 205.51 | 11.77 | 20.00 | False | 222.61 | 10.15 | 15.50 | -88.82 | 10.22 | -47.99 | 1.16 | 0.62 | 253.55 |
| b40f4729 | 0.00 | left_corridor_laterally | F other (moving human, mid-path, < 20 m/s, straight) | 2.33 | -1.74 | -1.61 | False | True | 4.56 | 64.42 | 64.42 | 0.40 | 20.00 | False | 74.13 | 10.93 | 13.33 | -1.68 | 2.20 | -93.24 | 0.99 | 0.92 | 157.09 |
| e904e9c0 | 0.00 | left_corridor_laterally | F other (moving human, mid-path, < 20 m/s, straight) | 12.11 | -0.51 | -0.26 | False | False | nan | 177.91 | 177.91 | 2.21 | 20.00 | False | 204.35 | 7.72 | 12.07 | -104.75 | 0.82 | 9.30 | 28.82 | 0.39 | 167.14 |
| eeba2aa9 | 0.00 | left_corridor_laterally | F other (moving human, mid-path, < 20 m/s, straight) | 15.96 | 2.74 | 1.13 | False | False | nan | 341.83 | 341.83 | 18.97 | 20.00 | False | 365.77 | 14.86 | 18.44 | -116.80 | 17.64 | 25.01 | 0.67 | 0.69 | 316.81 |

## Offroad zeros (14)

| id | score | fail | cls | v_call | a_call | a_call5 | latch_leash | latch_stop | cap | h_disp_max | h_rest_from_xy0 | h_v_end | h_t_stop | h_still_end | gt_len | t_trip | ego_v_trip | ego_past_rest | ego_v_at_hstop | ego_gap_at_hstop | turn_deg | gt_frac_trip | ego_maxd_xy0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2a882d89 | 0.00 | offroad | - | 8.89 | 0.31 | 0.30 | False | False | nan | 125.11 | 125.11 | 7.07 | 20.00 | False | 170.62 | nan | nan | nan | nan | nan | nan | nan | 123.56 |
| 2eaf1872 | 0.00 | offroad | - | 25.67 | 0.54 | 0.27 | False | False | nan | 503.98 | 503.98 | 28.89 | 20.00 | False | 546.12 | nan | nan | nan | nan | nan | nan | nan | 472.13 |
| 300465f7 | 0.00 | offroad | - | 5.28 | -0.27 | 0.06 | False | False | nan | 90.82 | 89.77 | 2.90 | 20.00 | False | 105.91 | 16.45 | 3.61 | -7.20 | 3.06 | 15.10 | 148.54 | 0.90 | 104.27 |
| 33c8976b | 0.00 | offroad | - | 1.49 | -0.12 | -0.04 | False | False | nan | 104.02 | 104.02 | 10.03 | 20.00 | False | 114.13 | 4.85 | 6.08 | -102.61 | 0.49 | 71.41 | 66.87 | 0.35 | 40.62 |
| 4f7cb072 | 0.00 | offroad | - | 23.02 | 0.28 | 0.16 | False | False | nan | 416.16 | 416.16 | 22.43 | 20.00 | False | 462.28 | nan | nan | nan | nan | nan | nan | nan | 364.70 |
| 596af0c7 | 0.00 | offroad | - | 26.41 | 0.21 | 0.03 | False | False | nan | 455.48 | 455.48 | 26.19 | 20.00 | False | 528.10 | nan | nan | nan | nan | nan | nan | nan | 344.00 |
| 7381c9fa | 0.00 | offroad | - | 3.35 | 0.37 | 0.44 | False | False | nan | 54.35 | 54.35 | 0.00 | 13.64 | True | 60.59 | 4.65 | 7.80 | -40.14 | 7.78 | -15.34 | 42.45 | 0.27 | 84.24 |
| 94c1d722 | 0.00 | offroad | - | 16.46 | -2.99 | -2.61 | False | False | nan | 183.51 | 183.51 | 23.16 | 20.00 | False | 257.48 | 11.64 | 7.66 | -158.72 | 5.76 | 113.89 | 75.04 | 0.56 | 128.30 |
| a5599793 | 0.00 | offroad | - | 3.62 | -0.18 | -0.34 | False | False | nan | 113.00 | 113.00 | 3.77 | 20.00 | False | 121.21 | 18.64 | 10.86 | 4.51 | 7.23 | -16.26 | 3.92 | 1.00 | 129.31 |
| a68ad579 | 0.00 | offroad | - | 16.98 | -0.38 | -0.60 | False | False | nan | 250.22 | 250.22 | 12.81 | 20.00 | False | 306.63 | nan | nan | nan | nan | nan | nan | nan | 209.16 |
| b661da67 | 0.00 | offroad | - | 4.20 | -0.60 | -0.30 | False | False | nan | 96.76 | 96.76 | 6.27 | 20.00 | False | 103.98 | 17.75 | 5.52 | 4.14 | 6.20 | -16.95 | 0.53 | 1.00 | 113.70 |
| ba62e10d | 0.00 | offroad | - | 20.36 | -0.59 | -0.48 | False | False | nan | 158.41 | 158.41 | 5.72 | 20.00 | False | 193.32 | 17.34 | 6.47 | 4.16 | 9.94 | -22.55 | 55.91 | 1.00 | 169.22 |
| d8bfaaf2 | 0.00 | offroad | - | 10.66 | 0.81 | 0.28 | False | False | nan | 177.44 | 177.44 | 10.83 | 20.00 | False | 231.48 | 19.75 | 11.54 | 4.21 | 11.21 | -6.48 | 30.43 | 1.00 | 177.93 |
| e848c843 | 0.00 | offroad | - | 6.53 | 0.93 | -0.35 | False | False | nan | 89.53 | 89.53 | 7.96 | 20.00 | False | 131.36 | nan | nan | nan | nan | nan | nan | nan | 82.77 |

## Other zeros (9)

| id | score | fail | cls | v_call | a_call | a_call5 | latch_leash | latch_stop | cap | h_disp_max | h_rest_from_xy0 | h_v_end | h_t_stop | h_still_end | gt_len | t_trip | ego_v_trip | ego_past_rest | ego_v_at_hstop | ego_gap_at_hstop | turn_deg | gt_frac_trip | ego_maxd_xy0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 09528162 | 0.00 | collision_at_fault | - | 8.05 | -0.85 | -0.72 | False | False | nan | 109.98 | 109.98 | 7.52 | 20.00 | False | 141.45 | 8.14 | 7.68 | -92.43 | 7.91 | 2.89 | 42.77 | 0.44 | 107.13 |
| 1a69f757 | 0.00 | collision_at_fault | - | 5.85 | -0.94 | -1.02 | False | False | nan | 124.54 | 124.54 | 10.71 | 20.00 | False | 143.57 | nan | nan | nan | nan | nan | nan | nan | 119.17 |
| 22fa5c92 | 0.00 | collision_at_fault | - | 0.00 | -0.02 | -0.00 | True | False | 3.00 | 36.12 | 36.12 | 6.43 | 20.00 | False | 43.23 | 9.54 | 4.32 | -17.19 | 0.63 | -1.33 | 29.76 | 0.84 | 34.48 |
| 2c0ef536 | 0.00 | collision_at_fault | - | 7.41 | -2.44 | -1.17 | False | False | nan | 29.71 | 29.71 | 4.41 | 20.00 | False | 42.93 | 18.80 | 4.55 | 4.11 | 4.47 | -9.12 | 2.25 | 1.00 | 38.85 |
| 3ce13eed | 0.00 | collision_at_fault | - | 3.26 | 0.81 | 0.80 | False | False | nan | 93.58 | 93.58 | 9.28 | 20.00 | False | 114.33 | 6.63 | 3.24 | -93.55 | 8.02 | 53.20 | 83.61 | 0.35 | 53.10 |
| b680a689 | 0.00 | collision_at_fault | - | 7.35 | 0.41 | -0.05 | False | False | nan | 62.36 | 62.36 | 3.83 | 20.00 | False | 79.41 | 12.64 | 6.41 | 4.26 | 3.32 | -45.03 | 54.97 | 1.00 | 85.37 |
| dbfd6c72 | 0.00 | collision_at_fault | - | 4.28 | -0.37 | -0.10 | False | False | nan | 12.17 | 12.17 | 0.00 | 6.20 | True | 18.61 | nan | nan | nan | nan | nan | nan | nan | 15.82 |
| f8ee31cb | 0.00 | collision_at_fault | - | 5.77 | -0.33 | -0.22 | False | False | nan | 65.60 | 65.60 | 9.72 | 20.00 | False | 75.32 | 10.14 | 13.38 | 4.59 | 5.70 | -110.01 | 2.88 | 1.00 | 175.56 |
| fce18b62 | 0.00 | collision_at_fault | - | 2.42 | 0.11 | -0.09 | False | False | nan | 28.47 | 28.47 | 0.01 | 18.64 | False | 32.31 | nan | nan | nan | nan | nan | nan | nan | 29.65 |

## Every latched rollout (leash3 or stop3) (43)

| id | score | fail | cls | v_call | a_call | a_call5 | latch_leash | latch_stop | cap | h_disp_max | h_rest_from_xy0 | h_v_end | h_t_stop | h_still_end | gt_len | t_trip | ego_v_trip | ego_past_rest | ego_v_at_hstop | ego_gap_at_hstop | turn_deg | gt_frac_trip | ego_maxd_xy0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2bbf1efc | 0.00 | left_corridor_laterally | B braking to a stop at handoff (stop3 latch) | 3.73 | -1.34 | -1.18 | False | True | 8.18 | 5.80 | 5.80 | 0.00 | 4.55 | True | 13.35 | 18.24 | 0.15 | 4.02 | 0.95 | -0.59 | 0.21 | 0.25 | 9.99 |
| 4e4a2a51 | 1.00 |  | - | 4.68 | -3.20 | -3.09 | False | True | 6.42 | 79.95 | 79.95 | 0.98 | 20.00 | False | 90.97 | nan | nan | nan | nan | nan | nan | nan | 78.71 |
| a100665e | 1.00 |  | - | 3.19 | -1.32 | -1.28 | False | True | 6.86 | 4.70 | 4.69 | 0.00 | 4.70 | True | 11.42 | nan | nan | nan | nan | nan | nan | nan | 5.75 |
| b40f4729 | 0.00 | left_corridor_laterally | F other (moving human, mid-path, < 20 m/s, straight) | 2.33 | -1.74 | -1.61 | False | True | 4.56 | 64.42 | 64.42 | 0.40 | 20.00 | False | 74.13 | 10.93 | 13.33 | -1.68 | 2.20 | -93.24 | 0.99 | 0.92 | 157.09 |
| dc401449 | 1.00 |  | - | 3.51 | -2.01 | -2.42 | False | True | 6.06 | 150.48 | 150.48 | 11.13 | 20.00 | False | 168.80 | 17.64 | 8.85 | 4.32 | 3.04 | -16.96 | 0.64 | 1.00 | 166.40 |
| e37aa159 | 0.00 | left_corridor_laterally | D turn / turn exit | 0.76 | -2.31 | -2.53 | False | True | 3.12 | 76.71 | 76.71 | 12.96 | 20.00 | False | 103.38 | 15.76 | 8.29 | -42.10 | 12.27 | -2.48 | 77.61 | 0.81 | 83.57 |
| e7551f4e | 1.00 |  | - | 1.08 | -0.80 | -1.46 | False | True | 3.73 | 49.90 | 49.90 | 5.17 | 20.00 | False | 53.84 | 16.23 | 5.99 | 4.41 | 1.55 | -13.84 | 1.08 | 1.00 | 63.77 |
| ea468790 | 1.00 |  | - | 0.55 | -1.87 | -2.09 | False | True | 3.08 | 97.02 | 97.02 | 16.90 | 20.00 | False | 109.53 | nan | nan | nan | nan | nan | nan | nan | 88.16 |
| fda39805 | 1.00 |  | - | 0.64 | -0.73 | -0.72 | False | True | 3.28 | 35.81 | 35.81 | 1.76 | 20.00 | False | 47.04 | 16.83 | 6.25 | 4.52 | 1.72 | -13.07 | 3.53 | 1.00 | 46.24 |
| 132ae6cb | 0.10 |  | - | 0.00 | -0.00 | 0.00 | True | False | 3.00 | 190.58 | 190.58 | 25.76 | 20.00 | False | 220.65 | nan | nan | nan | nan | nan | nan | nan | 63.50 |
| 21829be5 | 0.60 |  | - | 0.00 | 0.01 | 0.00 | True | False | 3.00 | 44.59 | 44.59 | 3.66 | 20.00 | False | 44.61 | nan | nan | nan | nan | nan | nan | nan | 30.81 |
| 22fa5c92 | 0.00 | collision_at_fault | - | 0.00 | -0.02 | -0.00 | True | False | 3.00 | 36.12 | 36.12 | 6.43 | 20.00 | False | 43.23 | 9.54 | 4.32 | -17.19 | 0.63 | -1.33 | 29.76 | 0.84 | 34.48 |
| 268949d2 | 0.82 |  | - | 0.02 | -0.35 | -1.06 | True | False | 3.00 | 66.73 | 66.73 | 2.66 | 20.00 | False | 76.20 | nan | nan | nan | nan | nan | nan | nan | 60.34 |
| 2a47ea01 | 0.99 |  | - | 0.00 | -0.00 | 0.00 | True | False | 3.00 | 138.21 | 138.21 | 14.98 | 20.00 | False | 138.22 | nan | nan | nan | nan | nan | nan | nan | 112.25 |
| 2b577fc2 | 0.85 |  | - | 0.00 | 0.01 | -0.00 | True | False | 3.00 | 61.93 | 61.93 | 3.76 | 20.00 | False | 61.95 | nan | nan | nan | nan | nan | nan | nan | 48.69 |
| 2d023524 | 1.00 |  | - | 0.00 | -0.01 | -0.00 | True | False | 3.00 | 93.01 | 93.01 | 15.65 | 20.00 | False | 93.04 | 19.34 | 14.55 | 4.18 | 14.07 | -12.82 | 1.42 | 1.00 | 105.82 |
| 3fe2bac8 | 0.00 | left_corridor_laterally | A still from handoff to end (leash3 latch) | 0.00 | -0.07 | -0.01 | True | False | 3.00 | 0.02 | 0.02 | 0.00 | 0.24 | True | 0.19 | 5.26 | 3.75 | 4.11 | 0.29 | -0.02 | 0.75 | 0.02 | 18.65 |
| 5266c7c9 | 1.00 |  | - | 0.01 | 0.03 | -0.00 | True | False | 3.00 | 25.96 | 25.96 | 1.82 | 20.00 | False | 25.98 | nan | nan | nan | nan | nan | nan | nan | 26.31 |
| 5fcb2ba7 | 1.00 |  | - | 0.00 | 0.00 | -0.00 | True | False | 3.00 | 0.00 | 0.00 | 0.00 | nan | True | 0.00 | nan | nan | nan | nan | nan | nan | nan | 2.91 |
| 68e53160 | 1.00 |  | - | 0.00 | -0.00 | -0.00 | True | False | 3.00 | 36.36 | 36.36 | 8.35 | 20.00 | False | 49.93 | 11.84 | 11.09 | 3.36 | 0.22 | -44.16 | 35.92 | 1.00 | 71.80 |
| 76f7e025 | 1.00 |  | - | 0.00 | 0.00 | 0.00 | True | False | 3.00 | 108.43 | 108.43 | 10.99 | 20.00 | False | 108.47 | 15.35 | 11.53 | 4.79 | 0.22 | -26.75 | 0.45 | 1.00 | 135.17 |
| 788c4ad9 | 1.00 |  | - | 0.00 | -0.01 | 0.00 | True | False | 3.00 | 146.84 | 146.84 | 12.39 | 20.00 | False | 146.87 | nan | nan | nan | nan | nan | nan | nan | 135.19 |
| 7b5c9fd6 | 1.00 |  | - | 0.25 | 0.07 | 0.09 | True | False | 3.00 | 69.07 | 69.07 | 9.36 | 20.00 | False | 69.40 | nan | nan | nan | nan | nan | nan | nan | 59.60 |
| 7c6529f6 | 0.50 |  | - | 0.06 | -0.52 | -0.45 | True | False | 3.00 | 11.93 | 11.93 | 0.64 | 20.00 | False | 12.64 | nan | nan | nan | nan | nan | nan | nan | 4.80 |
| 81dc4ece | 1.00 |  | - | 0.01 | -0.00 | -0.00 | True | False | 3.00 | 21.70 | 21.70 | 5.82 | 19.97 | False | 21.77 | nan | nan | nan | nan | nan | nan | nan | 20.09 |
| 903731dd | 1.00 |  | - | 0.00 | -0.00 | -0.00 | True | False | 3.00 | 173.94 | 173.94 | 13.44 | 20.00 | False | 173.94 | 19.72 | 15.60 | 4.78 | 15.61 | -7.91 | 0.11 | 1.00 | 181.84 |
| 9fa1d6eb | 1.00 |  | - | 0.01 | 0.00 | 0.00 | True | False | 3.00 | 34.78 | 34.78 | 2.38 | 20.00 | False | 34.84 | 16.97 | 4.89 | 4.00 | 1.48 | -13.90 | 0.16 | 1.00 | 48.69 |
| a1b166cc | 1.00 |  | - | 0.00 | -0.00 | -0.00 | True | False | 3.00 | 45.92 | 45.92 | 7.11 | 20.00 | False | 56.04 | 18.34 | 5.30 | 4.12 | 7.82 | -14.65 | 4.51 | 1.00 | 58.41 |
| a1f07c94 | 1.00 |  | - | 0.01 | -0.03 | -0.07 | True | False | 3.00 | 97.99 | 97.99 | 9.31 | 20.00 | False | 98.22 | nan | nan | nan | nan | nan | nan | nan | 81.10 |
| a686c7d0 | 0.19 |  | - | 0.00 | -0.07 | 0.00 | True | False | 3.00 | 98.62 | 98.62 | 9.42 | 20.00 | False | 98.65 | nan | nan | nan | nan | nan | nan | nan | 45.27 |
| b44a4a44 | 0.82 |  | - | 0.00 | -0.00 | -0.00 | True | False | 3.00 | 99.40 | 99.40 | 1.43 | 20.00 | False | 99.72 | nan | nan | nan | nan | nan | nan | nan | 64.80 |
| c7956ed6 | 1.00 |  | - | 0.14 | 0.67 | 0.26 | True | False | 3.00 | 90.66 | 90.66 | 10.30 | 20.00 | False | 90.69 | nan | nan | nan | nan | nan | nan | nan | 85.37 |
| c8641276 | 0.95 |  | - | 0.01 | 0.00 | 0.00 | True | False | 3.00 | 147.59 | 147.59 | 15.32 | 20.00 | False | 147.62 | nan | nan | nan | nan | nan | nan | nan | 111.18 |
| c8cab401 | 1.00 |  | - | 0.00 | -0.00 | 0.00 | True | False | 3.00 | 44.76 | 44.76 | 6.07 | 20.00 | False | 44.77 | nan | nan | nan | nan | nan | nan | nan | 43.53 |
| ca173de1 | 1.00 |  | - | 0.00 | 0.02 | 0.00 | True | False | 3.00 | 163.53 | 163.53 | 15.11 | 20.00 | False | 163.70 | nan | nan | nan | nan | nan | nan | nan | 162.35 |
| ca7accbc | 1.00 |  | - | 0.00 | 0.00 | -0.00 | True | False | 3.00 | 39.48 | 39.48 | 5.23 | 20.00 | False | 40.18 | 13.34 | 10.66 | 4.31 | 0.30 | -27.46 | 11.58 | 1.00 | 66.43 |
| cbb22323 | 1.00 |  | - | 0.00 | 0.00 | -0.00 | True | False | 3.00 | 11.96 | 11.96 | 0.00 | 14.65 | True | 11.99 | nan | nan | nan | nan | nan | nan | nan | 10.33 |
| ce161092 | 0.49 |  | - | 0.05 | 0.26 | 0.12 | True | False | 3.00 | 82.98 | 82.98 | 10.01 | 20.00 | False | 105.92 | nan | nan | nan | nan | nan | nan | nan | 72.82 |
| de9db5dc | 0.00 | left_corridor_laterally | D turn / turn exit | 0.02 | -0.17 | -0.52 | True | False | 3.00 | 37.57 | 37.57 | 0.63 | 20.00 | False | 40.99 | 15.65 | 3.48 | -12.64 | 6.83 | -1.09 | 54.90 | 0.65 | 40.86 |
| ec1d0b2b | 0.84 |  | - | 0.00 | -0.00 | -0.00 | True | False | 3.00 | 168.62 | 168.62 | 15.48 | 20.00 | False | 169.01 | nan | nan | nan | nan | nan | nan | nan | 115.22 |
| f1069db5 | 1.00 |  | - | 0.01 | 0.01 | 0.00 | True | False | 3.00 | 165.47 | 165.47 | 16.50 | 20.00 | False | 165.59 | nan | nan | nan | nan | nan | nan | nan | 153.68 |
| f18c6340 | 0.62 |  | - | 0.00 | -0.00 | 0.00 | True | False | 3.00 | 19.48 | 19.48 | 0.00 | 17.86 | True | 19.51 | nan | nan | nan | nan | nan | nan | nan | 12.59 |
| fd01e9ab | 0.00 | left_corridor_laterally | D turn / turn exit | 0.01 | 0.00 | 0.00 | True | False | 3.00 | 226.39 | 226.39 | 25.73 | 20.00 | False | 235.73 | 7.34 | 4.77 | -217.21 | 0.33 | 183.51 | 99.11 | 0.43 | 42.83 |

## Offline hold estimate (latched rollouts only change)

Held ego stays within `cap` of the handoff point xy0. Fix = the human is still at the end and its rest point is within cap + 4 m of xy0 (the ego can end inside the 4 m disc) -> score 1 if GT < 5 m else min(1, cap / (0.8 GT)) as an upper bound. Launch = the human later goes farther than cap from xy0 -> held score <= min(1, cap / (0.8 GT)) (progress bound, collisions not modelled).

| rule | latched | human still to end | fix candidates (score 0 now) | gain (all positive deltas) | launches | of which scored > 0 now | cost | net / 281 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| leash3 | 34 | 4 | 1 | +1.19 | 32 | 29 | -22.52 | -0.0759 |
| stop3 extra | 9 | 2 | 1 | +0.88 | 7 | 5 | -4.91 | -0.0143 |
| stop3 total | 43 | 6 | 2 | +2.08 | 39 | 34 | -27.43 | -0.0902 |
