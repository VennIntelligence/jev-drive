# default: jev-alpasim:ap2-ab-s0-70b25326

| check | value | limit | |
|---|--:|--:|---|
| image size | 7.48 GiB | 40 GiB | pass |
| read-only root, no network, uid 10001 | 0 write errors, 0 changed files | 0 | pass |
| peak GPU memory, 2 concurrent rollouts | 3.04 GiB | 16 GiB | pass |
| /tmp high-water (probe / 48-scene smoke) | 0.2 / 0.8 MiB | 2048 MiB | pass |
| /run high-water | 0.00 MiB | 64 MiB | pass |
| cold start until the service answers | 19.2 s | - | |
| `drive`, 2 concurrent synthetic rollouts (NVIDIA GeForce RTX 3090) | p50 89.4 / p90 100.9 / p99 115.9 / max 133.2 ms | target 100 ms | |
| `drive`, 48-scene smoke at 8 concurrent rollouts (server side) | p50 56.0 / p90 67.5 / max 111.3 ms | | 0 inference errors |
| 48 scenes, containerised | mean 0.9521, zeros 2, at 1 41 (n 48) | | |
| 48 scenes, native reference | mean 0.9320, zeros 3, at 1 41 | | 5 of 48 scenes differ, max 0.9719 |

# default

containerised: n 48, mean scene score 0.9521, zeros 2, at 1 41
driver: 480 drive calls, 0 inference errors, total p50 / p90 / p99 / max 56.0 / 67.5 / 105.7 / 111.3 ms (encode p50 21.1, frames p50 25.0, wait p50 0.0)
native reference: n 48, mean 0.9320, zeros 3, at 1 41
scenes that differ: 5 of 48 (max |diff| 0.9719)

| scene | containerised | native | why (containerised / native) |
|---|--:|--:|---|
| 2021.05.25.14.16.10_veh-35_00083_00485-c9b12b21fa7c57fd | 0.8124 | 0.8185 | - / - |
| 2021.05.25.14.16.10_veh-35_01100_01664-368cb65e8fef57b7 | 0.9719 | 0.0000 | - / collision_at_fault |
| 2021.05.25.14.16.10_veh-35_01100_01664-6fbe0e06902e5304 | 0.9978 | 0.9989 | - / - |
| 2021.05.25.14.16.10_veh-35_01100_01664-82cd122751085a80 | 0.9616 | 0.9616 | - / - |
| 2021.05.25.14.16.10_veh-35_01690_02183-54e1cb577c0a5f7e | 0.9592 | 0.9578 | - / - |

