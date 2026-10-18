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
