# smoke_off_c001

containerised: n 36, mean scene score 0.8154, zeros 6, at 1 27
driver: 7151 drive calls, 0 inference errors, total p50 / p90 / p99 / max 57.9 / 74.9 / 140.7 / 221.0 ms (encode p50 33.9, frames p50 None, wait p50 0.0)
native reference: n 36, mean 0.8431, zeros 5, at 1 28
scenes that differ: 4 of 36 (max |diff| 1.0); paired mean diff -0.0277, within 0.01: 35, within 0.05: 35, zero / non-zero agree: 35 of 36

| scene | containerised | native | why (containerised / native) |
|---|--:|--:|---|
| clipgt-91d84204-d511-464e-b986-6cab844794cc | 0.9688 | 0.9694 | - / - |
| clipgt-ba62e10d-7d77-4707-96a6-207e6f6aaecc | 0.0000 | 1.0000 | offroad / - |
| clipgt-bbc53aa8-7009-4139-b7c9-38d71ecfac7f | 0.8954 | 0.8919 | - / - |
| clipgt-bd97de34-6f59-455a-93c5-9f469f243b4a | 0.4911 | 0.4910 | - / - |
