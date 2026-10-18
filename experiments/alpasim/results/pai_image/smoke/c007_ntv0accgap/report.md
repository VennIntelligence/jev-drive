# smoke: jev-alpasim:ntv0-l18-s0-0d417ee4-pai-vc1.0-lead-acc-gap0.7_2.7

| check | value | limit | |
|---|--:|--:|---|
| image size | 7.55 GiB | 40 GiB | pass |
| read-only root, no network, uid 10001 | 0 write errors, 0 changed files | 0 | pass |
| peak GPU memory, 2 concurrent rollouts | 1.44 GiB | 16 GiB | pass |
| /tmp high-water (probe / 48-scene smoke) | 2.8 / 25.2 MiB | 2048 MiB | pass |
| /run high-water | 0.00 MiB | 64 MiB | pass |
| cold start until the service answers | 17.6 s | - | |
| `drive`, 2 concurrent synthetic rollouts (NVIDIA GeForce RTX 3090) | p50 46.8 / p90 52.7 / p99 115.2 / max 190.5 ms | target 100 ms | |
| 4 replicas on one card, 2 concurrent rollouts each | card peak 5.76 GiB, per replica max 1.43 GiB; `drive` p50 167.3 / p90 173.3 / max 387.5 ms (worst replica); cold start max 20.5 s; versions 1 | 16 GiB per replica | pass |
| `drive`, 29-scene smoke (server side) | p50 56.5 / p90 75.7 / max 198.8 ms | | 0 inference errors |
| 29 scenes, containerised | mean 0.5403, zeros 13, at 1 14 (n 29) | | |
| 29 scenes, native reference | mean 0.6090, zeros 11, at 1 16 | | 4 of 29 scenes differ, max 1.0, paired mean diff -0.0687, zeros agree 27 |

# smoke

containerised: n 29, mean scene score 0.5403, zeros 13, at 1 14
driver: 5752 drive calls, 0 inference errors, total p50 / p90 / p99 / max 56.5 / 75.7 / 136.7 / 198.8 ms (encode p50 33.7, frames p50 None, wait p50 0.0)
native reference: n 29, mean 0.6090, zeros 11, at 1 16
scenes that differ: 4 of 29 (max |diff| 1.0); paired mean diff -0.0687, within 0.01: 27, within 0.05: 27, zero / non-zero agree: 27 of 29

| scene | containerised | native | why (containerised / native) |
|---|--:|--:|---|
| clipgt-2bbf1efc-55e9-4d87-bf74-0740353e8268 | 0.0000 | 1.0000 | left_corridor_laterally / - |
| clipgt-3232fdb7-2b72-4e25-8726-4d9b28281e1e | 0.0000 | 1.0000 | left_corridor_laterally / - |
| clipgt-404541da-5297-4c32-b0af-21740de150b0 | 0.7650 | 0.7570 | - / - |
| clipgt-478d6ec4-f02b-4c82-97af-ddc0fbe5027f | 0.9028 | 0.9028 | - / - |

