# smoke_c001

containerised: n 36, mean scene score 0.7666, zeros 6, at 1 22
driver: 7151 drive calls, 0 inference errors, total p50 / p90 / p99 / max 58.7 / 77.0 / 150.3 / 224.6 ms (encode p50 33.6, frames p50 None, wait p50 0.0)
native reference: n 36, mean 0.7939, zeros 5, at 1 23
scenes that differ: 9 of 36 (max |diff| 1.0); paired mean diff -0.0273, within 0.01: 34, within 0.05: 35, zero / non-zero agree: 35 of 36

| scene | containerised | native | why (containerised / native) |
|---|--:|--:|---|
| clipgt-91d84204-d511-464e-b986-6cab844794cc | 0.9632 | 0.9679 | - / - |
| clipgt-94c1d722-1fce-46b4-9aa9-c78389fae1ac | 0.9732 | 0.9600 | - / - |
| clipgt-9d9e7f39-0435-41ad-bd7e-6a672119183c | 0.6697 | 0.6711 | - / - |
| clipgt-9f393c8a-4e24-4b04-b98a-099a28bde89a | 0.3552 | 0.3552 | - / - |
| clipgt-b44a4a44-0d00-4795-8a66-7acb0fdf46a2 | 0.7657 | 0.7658 | - / - |
| clipgt-b65e55e9-f3d5-4445-bdbd-325b07a5c493 | 0.4894 | 0.4805 | - / - |
| clipgt-ba62e10d-7d77-4707-96a6-207e6f6aaecc | 0.0000 | 1.0000 | offroad / - |
| clipgt-bd97de34-6f59-455a-93c5-9f469f243b4a | 0.4908 | 0.4908 | - / - |
| clipgt-c8641276-c887-465d-b88e-df70bd7214b6 | 0.8887 | 0.8886 | - / - |
