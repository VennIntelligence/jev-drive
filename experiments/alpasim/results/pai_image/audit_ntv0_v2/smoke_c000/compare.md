# smoke_c000

containerised: n 36, mean scene score 0.6931, zeros 10, at 1 21
driver: 7082 drive calls, 0 inference errors, total p50 / p90 / p99 / max 57.2 / 75.1 / 157.0 / 382.8 ms (encode p50 32.7, frames p50 None, wait p50 0.0)
native reference: n 36, mean 0.6923, zeros 10, at 1 21
scenes that differ: 5 of 36 (max |diff| 0.0238); paired mean diff +0.0009, within 0.01: 35, within 0.05: 36, zero / non-zero agree: 36 of 36

| scene | containerised | native | why (containerised / native) |
|---|--:|--:|---|
| clipgt-596af0c7-c33f-4160-84d9-5ab525e9675e | 0.6149 | 0.6148 | - / - |
| clipgt-5ddb96bd-7221-4107-8987-29b9928c58c7 | 0.7009 | 0.6940 | - / - |
| clipgt-6250146a-7f9f-4508-851e-e25b7d9dc36a | 0.9398 | 0.9398 | - / - |
| clipgt-6e6cbe67-bccd-4d4d-9943-933a272b7bf0 | 0.8485 | 0.8247 | - / - |
| clipgt-7c346fd5-4223-40da-8785-c49fe69f06a1 | 0.8479 | 0.8479 | - / - |
