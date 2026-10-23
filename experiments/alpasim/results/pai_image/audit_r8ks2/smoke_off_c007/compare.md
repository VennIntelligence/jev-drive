# smoke_off_c007

containerised: n 29, mean scene score 0.6865, zeros 9, at 1 19
driver: 5752 drive calls, 0 inference errors, total p50 / p90 / p99 / max 57.4 / 77.8 / 142.0 / 194.9 ms (encode p50 33.9, frames p50 None, wait p50 0.0)
native reference: n 29, mean 0.7103, zeros 8, at 1 19
scenes that differ: 2 of 29 (max |diff| 0.6898); paired mean diff -0.0238, within 0.01: 28, within 0.05: 28, zero / non-zero agree: 28 of 29

| scene | containerised | native | why (containerised / native) |
|---|--:|--:|---|
| clipgt-268949d2-3e20-4972-8785-c15e4fcb5fde | 0.0000 | 0.6898 | collision_at_fault / - |
| clipgt-478d6ec4-f02b-4c82-97af-ddc0fbe5027f | 0.9097 | 0.9092 | - / - |
