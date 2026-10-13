**Unpaced, 2x fast-forward** (session 1, run /data/runs/alpasim_human/pai_1013-2255 on Tokyo, before HUMAN_REALTIME existed: the sim ran at about 0.5 wall s per sim s, so the user drove a world moving twice as fast as the clock).

| scene | human | why | progress | driven / GT m | sim s | wall per sim s | stale | wait s | P2H10 (model) | alpamayo1 (organisers) | organisers' mean |
|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 724-f7de1d45ab65 | 0.023 | slow | 0.018 | 1.6 / 86.4 | 19.8 | 0.507 | 0 | 7.1 | 0.000 | 1.000 | 0.542 |
| bb4-c3029d89fcbe | 1.000 | pass | 1.002 | 194.4 / 190.0 | 19.7 | 0.532 | 0 | 3.9 | 0.000 | 1.000 | 0.292 |
| dc9-aaef6d0a7b92 | 0.000 | left_corridor_laterally | 0.336 | 50.7 / 149.4 | 19.8 | 0.513 | 0 | 20.5 | 0.000 | 0.648 | 0.081 |
| 447-95741690d9d8 | 0.000 | collision_at_fault | 0.594 | 249.9 / 421.9 | 19.8 | 0.503 | 0 | 3.4 | 0.538 | 0.424 | 0.380 |
| 09e-18fa15cbbf97 | 0.680 | slow | 0.544 | 273.3 / 503.7 | 19.8 | 0.553 | 0 | 3.4 | 0.000 | 0.310 | 0.238 |
| f33-0118cbb902e5 | 1.000 | pass | 1.001 | 217.5 / 212.6 | 19.7 | 0.55 | 0 | 0.8 | 1.000 | 0.000 | 0.167 |
| f20-667d342cbc69 | 0.000 | left_corridor_laterally | 1.0 | 53.9 / 50.7 | 19.7 | 0.483 | 0 | 49.7 | 0.000 | 0.000 | 0.000 |
| e4a-f4453def52cc | 0.000 | offroad | 0.194 | 46.2 / 239.0 | 19.8 | 0.488 | 0 | 1.5 | 0.000 | 0.000 | 0.113 |
| 368-9a906a333f6e | 0.000 | left_corridor_laterally | 0.697 | 105.6 / 151.0 | 19.7 | 0.517 | 0 | 2.0 | 0.000 | 0.000 | 0.042 |
| 257-3d992516a2f7 | 0.000 | collision_at_fault | 0.129 | 7.8 / 58.7 | 19.8 | 0.62 | 0 | 134.6 | 0.000 | 1.000 | 0.125 |

pai, 10 scenes: human mean 0.2703; P2H10 (model) 0.1538; alpamayo1 (organisers) 0.4382; organisers' mean 0.1979; wall s per sim s (after the start) 0.527
