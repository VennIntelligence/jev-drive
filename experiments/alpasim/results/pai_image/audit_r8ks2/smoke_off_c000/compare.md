# smoke_off_c000

containerised: n 36, mean scene score 0.7660, zeros 8, at 1 25
driver: 7082 drive calls, 0 inference errors, total p50 / p90 / p99 / max 56.3 / 75.7 / 139.6 / 202.4 ms (encode p50 33.0, frames p50 None, wait p50 0.0)
native reference: n 36, mean 0.7662, zeros 8, at 1 25
scenes that differ: 3 of 36 (max |diff| 0.0075); paired mean diff -0.0002, within 0.01: 36, within 0.05: 36, zero / non-zero agree: 36 of 36

| scene | containerised | native | why (containerised / native) |
|---|--:|--:|---|
| clipgt-6250146a-7f9f-4508-851e-e25b7d9dc36a | 0.9408 | 0.9411 | - / - |
| clipgt-6e6cbe67-bccd-4d4d-9943-933a272b7bf0 | 0.8385 | 0.8381 | - / - |
| clipgt-7c346fd5-4223-40da-8785-c49fe69f06a1 | 0.7954 | 0.8030 | - / - |
