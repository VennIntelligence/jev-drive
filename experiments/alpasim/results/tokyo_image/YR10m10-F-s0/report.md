# default: jev-alpasim:yr10m10-f-s0-073dea33

| check | value | limit | |
|---|--:|--:|---|
| image size | 7.55 GiB | 40 GiB | pass |
| read-only root, no network, uid 10001 | 0 write errors, 0 changed files | 0 | pass |
| peak GPU memory, 2 concurrent rollouts | 1.56 GiB | 16 GiB | pass |
| /tmp high-water (probe / 48-scene smoke) | 0.2 / 0.8 MiB | 2048 MiB | pass |
| /run high-water | 0.00 MiB | 64 MiB | pass |
| cold start until the service answers | 20.4 s | - | |
| `drive`, 2 concurrent synthetic rollouts (NVIDIA GeForce RTX 3090) | p50 50.0 / p90 52.6 / p99 69.5 / max 77.8 ms | target 100 ms | |
| `drive`, 48-scene smoke at 8 concurrent rollouts (server side) | p50 35.8 / p90 44.8 / max 148.1 ms | | 0 inference errors |
| 48 scenes, containerised | mean 0.9838, zeros 0, at 1 35 (n 48) | | |
| 48 scenes, native reference | mean 0.9633, zeros 1, at 1 34 | | 14 of 48 scenes differ, max 1.0 |

# default

containerised: n 48, mean scene score 0.9838, zeros 0, at 1 35
driver: 480 drive calls, 0 inference errors, total p50 / p90 / p99 / max 35.8 / 44.8 / 84.8 / 148.1 ms (encode p50 19.9, frames p50 4.3, wait p50 0.0)
native reference: n 48, mean 0.9633, zeros 1, at 1 34
scenes that differ: 14 of 48 (max |diff| 1.0)

| scene | containerised | native | why (containerised / native) |
|---|--:|--:|---|
| 2021.05.25.14.16.10_veh-35_00083_00485-4b594a8e80915943 | 0.9625 | 0.9591 | - / - |
| 2021.05.25.14.16.10_veh-35_00083_00485-c9b12b21fa7c57fd | 0.9878 | 0.9872 | - / - |
| 2021.05.25.14.16.10_veh-35_01100_01664-1893fb783df95146 | 0.7880 | 0.7862 | - / - |
| 2021.05.25.14.16.10_veh-35_01100_01664-6fbe0e06902e5304 | 0.9708 | 0.9696 | - / - |
| 2021.05.25.14.16.10_veh-35_01100_01664-6fed9368351f54d5 | 0.9192 | 0.9244 | - / - |
| 2021.05.25.14.16.10_veh-35_01100_01664-82cd122751085a80 | 0.9254 | 0.9278 | - / - |
| 2021.05.25.14.16.10_veh-35_01100_01664-9215555823945665 | 0.9896 | 0.9888 | - / - |
| 2021.05.25.14.16.10_veh-35_01100_01664-a41f538fa8e25be0 | 0.8992 | 0.9021 | - / - |
| 2021.05.25.14.16.10_veh-35_01100_01664-dd9b1479609c5c59 | 0.9764 | 0.9817 | - / - |
| 2021.05.25.14.16.10_veh-35_01100_01664-fe800ded24045b44 | 0.9870 | 0.9887 | - / - |
| 2021.05.25.14.16.10_veh-35_01690_02183-30f09e001cf55013 | 1.0000 | 0.0000 | - / offroad |
| 2021.05.25.14.16.10_veh-35_01690_02183-54e1cb577c0a5f7e | 0.9591 | 0.9593 | - / - |
| 2021.05.25.14.16.10_veh-35_01690_02183-75f9d978b4d757a2 | 0.9692 | 0.9711 | - / - |
| 2021.05.25.14.16.10_veh-35_02482_02649-6f11adda2af357ff | 0.8879 | 0.8916 | - / - |

