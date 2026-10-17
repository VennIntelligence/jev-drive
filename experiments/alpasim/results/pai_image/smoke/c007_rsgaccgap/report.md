# smoke: jev-alpasim:rsg-l18-s0-ea61e9df-pai-vc1.0-lead-acc-gap0.7_2.7

| check | value | limit | |
|---|--:|--:|---|
| image size | 7.48 GiB | 40 GiB | pass |
| read-only root, no network, uid 10001 | 0 write errors, 0 changed files | 0 | pass |
| peak GPU memory, 2 concurrent rollouts | 1.44 GiB | 16 GiB | pass |
| /tmp high-water (probe / 48-scene smoke) | 2.7 / 25.2 MiB | 2048 MiB | pass |
| /run high-water | 0.00 MiB | 64 MiB | pass |
| cold start until the service answers | 17.7 s | - | |
| `drive`, 2 concurrent synthetic rollouts (NVIDIA GeForce RTX 3090) | p50 47.1 / p90 52.7 / p99 113.6 / max 191.8 ms | target 100 ms | |
| 4 replicas on one card, 2 concurrent rollouts each | card peak 5.76 GiB, per replica max 1.43 GiB; `drive` p50 166.1 / p90 172.3 / max 457.2 ms (worst replica); cold start max 20.3 s; versions 1 | 16 GiB per replica | pass |
| `drive`, 29-scene smoke (server side) | p50 56.2 / p90 73.4 / max 239.3 ms | | 0 inference errors |
| 29 scenes, containerised | mean 0.5519, zeros 12, at 1 13 (n 29) | | |
| 29 scenes, native reference | mean 0.5502, zeros 12, at 1 13 | | 4 of 29 scenes differ, max 0.0633, paired mean diff +0.0017, zeros agree 29 |

# smoke

containerised: n 29, mean scene score 0.5519, zeros 12, at 1 13
driver: 5752 drive calls, 0 inference errors, total p50 / p90 / p99 / max 56.2 / 73.4 / 136.8 / 239.3 ms (encode p50 33.5, frames p50 None, wait p50 0.0)
native reference: n 29, mean 0.5502, zeros 12, at 1 13
scenes that differ: 4 of 29 (max |diff| 0.0633); paired mean diff +0.0017, within 0.01: 27, within 0.05: 28, zero / non-zero agree: 29 of 29

| scene | containerised | native | why (containerised / native) |
|---|--:|--:|---|
| clipgt-268949d2-3e20-4972-8785-c15e4fcb5fde | 0.8044 | 0.8187 | - / - |
| clipgt-404541da-5297-4c32-b0af-21740de150b0 | 0.9478 | 0.9473 | - / - |
| clipgt-478d6ec4-f02b-4c82-97af-ddc0fbe5027f | 0.9031 | 0.9030 | - / - |
| clipgt-52bd89ce-db30-4525-9fa1-b0c0dd72f5f4 | 0.3505 | 0.2872 | - / - |

