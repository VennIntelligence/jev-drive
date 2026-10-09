Scenes scored in every run: 10. Reference subjects: mean of 3 rollouts on the same scenes (e2e_challenge/local_evaluation/data/pai).

| Row | Mean scene score | Score 1 | Zeros | at-fault collision | offroad | left corridor | Slow (0 < s < 1) |
|---|--:|--:|--:|--:|--:|--:|--:|
| **P2H10 10 Hz** | 0.1538 | 1 | 8 | 4 | 2 | 2 | 1 |
| **P2H10 2 Hz** | 0.1570 | 1 | 8 | 3 | 3 | 2 | 1 |
| **P2H10 10 Hz harmonizer** | 0.1539 | 1 | 8 | 3 | 2 | 3 | 1 |
| alpamayo1 (reference) | 0.4382 | 3 | 4 | 3% | 10% | 37% | |
| alternative_1 (reference) | 0.1005 | 1 | 8 | 20% | 30% | 30% | |
| alternative_2 (reference) | 0.1000 | 1 | 9 | 10% | 40% | 50% | |
| alternative_3 (reference) | 0.2001 | 2 | 7 | 10% | 30% | 30% | |
| alternative_4 (reference) | 0.2186 | 1 | 7 | 10% | 20% | 40% | |
| alternative_5 (reference) | 0.0000 | 0 | 10 | 10% | 50% | 50% | |
| vavam-linear (reference) | 0.4265 | 2 | 3 | 20% | 0% | 23% | |
| vavam-nonlinear (reference) | 0.0995 | 0 | 7 | 13% | 33% | 47% | |

| Scene | Reference mean | alpamayo1 | P2H10 10 Hz | why | P2H10 2 Hz | why | P2H10 10 Hz harmonizer | why | GT distance m | P2H10 10 Hz driven m | P2H10 2 Hz driven m | P2H10 10 Hz harmonizer driven m |
|---|--:|--:|--:|---|--:|---|--:|---|--:|--:|--:|--:|
| 7a824ffa | 0.00 | 0.00 | 0.00 | left_corridor_laterally | 0.00 | left_corridor_laterally | 0.00 | left_corridor_laterally | 51 | 46 | 55 | 45 |
| 9ea70552 | 0.04 | 0.00 | 0.00 | collision_at_fault | 0.00 | collision_at_fault | 0.00 | collision_at_fault | 151 | 155 | 155 | 154 |
| 213dfdac | 0.08 | 0.65 | 0.00 | left_corridor_laterally | 0.00 | left_corridor_laterally | 0.00 | left_corridor_laterally | 149 | 71 | 72 | 70 |
| 94877a4a | 0.11 | 0.00 | 0.00 | offroad | 0.00 | offroad | 0.00 | offroad | 239 | 95 | 85 | 48 |
| a28b6685 | 0.12 | 1.00 | 0.00 | collision_at_fault | 0.00 | offroad | 0.00 | collision_at_fault | 59 | 3 | 51 | 3 |
| 5d794411 | 0.17 | 0.00 | 1.00 |  | 1.00 |  | 1.00 |  | 213 | 217 | 218 | 217 |
| 4f779a92 | 0.24 | 0.31 | 0.00 | offroad | 0.00 | offroad | 0.00 | offroad | 504 | 55 | 55 | 53 |
| 13fb89b9 | 0.29 | 1.00 | 0.00 | collision_at_fault | 0.00 | collision_at_fault | 0.00 | left_corridor_laterally | 190 | 68 | 83 | 117 |
| 49597f01 | 0.38 | 0.42 | 0.54 | progress 0.43 | 0.57 | progress 0.46 | 0.54 | progress 0.43 | 422 | 181 | 192 | 181 |
| 071d15c4 | 0.54 | 1.00 | 0.00 | collision_at_fault | 0.00 | collision_at_fault | 0.00 | collision_at_fault | 86 | 16 | 26 | 10 |

**P2H10 10 Hz** (a10_c4): {"scenes": 10, "conc": 4, "wall_s": 621, "gpu": 1, "driver_gpu": 1, "tag": "P2H10-F-s0", "harmonizer": 0, "image": "jev-alpasim:p2h10-f-s0-6f3d05d1", "base": "alpasim-base:0.89.0"}
- 1986 `drive` calls, 1986 inferences; per call median 48.0 ms, p95 96.0, max 296; stage medians encode 32.1, policy 12.8, export 0.1, prep 1.0, slots 0.3, wait 0.0 ms
- commands fed (left / straight / right / unknown): [922, 408, 656, 0]; real slots per inference: {1: 20, 2: 20, 3: 20, 4: 20, 5: 20, 6: 20, 7: 20, 8: 1846}; history keys: {1: 45, 2: 50, 3: 50, 4: 1841}
- session counters: drive 1986, inference 1986, inference_error 0, input_error 0, state_rotated 1, cold 140, images 1996; delivered sizes [(1920, 1080)]; model-frame coverage (road, wide) [1.0, 1.0] (minimum over sessions)
- peaks: c4-controller-0-1 VRAM 0.0 GiB, RAM 0.5 GiB; c4-drv VRAM 3.0 GiB, RAM 1.9 GiB; c4-physics-0-1 VRAM 1.1 GiB, RAM 0.6 GiB; c4-renderer-0-1 VRAM 18.1 GiB, RAM 5.9 GiB; c4-runtime-0-1 VRAM 0.0 GiB, RAM 9.3 GiB

**P2H10 2 Hz** (b10_every5): {"scenes": 10, "conc": 4, "wall_s": 569, "gpu": 1, "driver_gpu": 1, "tag": "P2H10-F-s0", "harmonizer": 0, "image": "jev-alpasim:p2h10-f-s0-6f3d05d1", "base": "alpasim-base:0.89.0"}
- 1986 `drive` calls, 400 inferences; per call median 47.6 ms, p95 93.2, max 312; stage medians encode 32.2, policy 12.6, export 0.1, prep 1.0, slots 0.5, wait 0.0 ms
- commands fed (left / straight / right / unknown): [123, 126, 151, 0]; real slots per inference: {1: 10, 3: 10, 6: 10, 8: 370}; history keys: {1: 10, 2: 10, 3: 10, 4: 370}
- session counters: drive 1986, inference 400, inference_error 0, input_error 0, state_rotated 0, cold 30, images 1996; delivered sizes [(1920, 1080)]; model-frame coverage (road, wide) [1.0, 1.0] (minimum over sessions)
- peaks: every5-controller-0-1 VRAM 0.0 GiB, RAM 0.5 GiB; every5-drv VRAM 3.0 GiB, RAM 2.0 GiB; every5-physics-0-1 VRAM 1.1 GiB, RAM 0.7 GiB; every5-renderer-0-1 VRAM 18.7 GiB, RAM 6.0 GiB; every5-runtime-0-1 VRAM 0.0 GiB, RAM 9.1 GiB

**P2H10 10 Hz harmonizer** (c10_harm): {"scenes": 10, "conc": 4, "wall_s": 4436, "gpu": 1, "driver_gpu": 0, "tag": "P2H10-F-s0", "harmonizer": 1, "image": "jev-alpasim:p2h10-f-s0-6f3d05d1", "base": "alpasim-base:0.89.0"}
- 1986 `drive` calls, 1986 inferences; per call median 141.7 ms, p95 209.2, max 399; stage medians encode 124.5, policy 15.4, export 0.1, prep 1.0, slots 0.2, wait 0.0 ms
- commands fed (left / straight / right / unknown): [734, 527, 725, 0]; real slots per inference: {1: 20, 2: 20, 3: 20, 4: 20, 5: 20, 6: 20, 7: 20, 8: 1846}; history keys: {1: 45, 2: 50, 3: 50, 4: 1841}
- session counters: drive 1986, inference 1986, inference_error 0, input_error 0, state_rotated 0, cold 140, images 1996; delivered sizes [(1920, 1080)]; model-frame coverage (road, wide) [1.0, 1.0] (minimum over sessions)
- peaks: harm-controller-0-1 VRAM 0.0 GiB, RAM 0.5 GiB; harm-drv VRAM 3.0 GiB, RAM 2.0 GiB; harm-physics-0-1 VRAM 1.1 GiB, RAM 0.6 GiB; harm-renderer-0-1 VRAM 18.4 GiB, RAM 6.6 GiB; harm-runtime-0-1 VRAM 0.0 GiB, RAM 5.4 GiB
