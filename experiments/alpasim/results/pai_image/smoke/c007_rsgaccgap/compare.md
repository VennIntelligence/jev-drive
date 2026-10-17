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
