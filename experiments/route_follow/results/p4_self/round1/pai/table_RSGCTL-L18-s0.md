# P4 self-branch read (27 rollouts, 27 scenes)

`done` = turn completed (P2's rule). Otherwise the zero type (corr / offr / coll / oth) or rear (rear-ended before the target) / end.
Cells g<gain>-<profile>: raw = the plan's own speed, ve = profile to v_e, ve08 = to 0.8 v_e. `id` = the policy alone (phase A).
(log) = unreached distance, handover from the log (src=log). `=D<n>` = same handover decision as D<n> (run once).

| scene | side | id | D | src | h | g1-raw | g1-ve | g1-ve08 | g1.25-raw | g1.25-ve | g1.25-ve08 | g1.5-raw | g1.5-ve | g1.5-ve08 |
|:--|:--|:--|--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 0d10d2e3 | L | coll | 40 | self | 17 | coll | - | - | - | - | - | - | - | - |
| 0d10d2e3 | L | coll | 30 | self =D40 | 17 | coll | - | - | - | - | - | - | - | - |
| 0d10d2e3 | L | coll | 20 | self =D40 | 17 | coll | - | - | - | - | - | - | - | - |
| 0d10d2e3 | L | coll | 10 | self | 29 | coll | - | - | - | - | - | - | - | - |
| 0d10d2e3 | L | coll | 0 | self | 68 | coll | - | - | - | - | - | - | - | - |
| 213dfdac | L | offr | 40 | self | 19 | offr | - | - | - | - | - | - | - | - |
| 213dfdac | L | offr | 30 | self | 29 | offr | - | - | - | - | - | - | - | - |
| 213dfdac | L | offr | 20 | self | 42 | offr | - | - | - | - | - | - | - | - |
| 213dfdac | L | offr | 10 | self | 77 | offr | - | - | - | - | - | - | - | - |
| 213dfdac | L | offr | 0 | self | 111 | offr | - | - | - | - | - | - | - | - |
| 33c8976b | L | offr | 40 | self | 17 | offr | - | - | - | - | - | - | - | - |
| 33c8976b | L | offr | 30 | self =D40 | 17 | offr | - | - | - | - | - | - | - | - |
| 33c8976b | L | offr | 20 | self =D40 | 17 | offr | - | - | - | - | - | - | - | - |
| 33c8976b | L | offr | 10 | self =D40 | 17 | offr | - | - | - | - | - | - | - | - |
| 33c8976b | L | offr | 0 | self | 34 | offr | - | - | - | - | - | - | - | - |
| 8a365ac2 | L | offr | 40 | self | 17 | offr | - | - | - | - | - | - | - | - |
| 8a365ac2 | L | offr | 30 | self =D40 | 17 | offr | - | - | - | - | - | - | - | - |
| 8a365ac2 | L | offr | 20 | self =D40 | 17 | offr | - | - | - | - | - | - | - | - |
| 8a365ac2 | L | offr | 10 | self =D40 | 17 | offr | - | - | - | - | - | - | - | - |
| 8a365ac2 | L | offr | 0 | self | 52 | offr | - | - | - | - | - | - | - | - |
| 96eb4700 | L | **done** | 40 | self | 17 | **done** | - | - | - | - | - | - | - | - |
| 96eb4700 | L | **done** | 30 | self =D40 | 17 | **done** | - | - | - | - | - | - | - | - |
| 96eb4700 | L | **done** | 20 | self =D40 | 17 | **done** | - | - | - | - | - | - | - | - |
| 96eb4700 | L | **done** | 10 | self =D40 | 17 | **done** | - | - | - | - | - | - | - | - |
| 96eb4700 | L | **done** | 0 | self =D40 | 17 | **done** | - | - | - | - | - | - | - | - |
| a4be61d6 | L | **done** | 40 | self | 17 | **done** | - | - | - | - | - | - | - | - |
| a4be61d6 | L | **done** | 30 | self =D40 | 17 | **done** | - | - | - | - | - | - | - | - |
| a4be61d6 | L | **done** | 20 | self =D40 | 17 | **done** | - | - | - | - | - | - | - | - |
| a4be61d6 | L | **done** | 10 | self =D40 | 17 | **done** | - | - | - | - | - | - | - | - |
| a4be61d6 | L | **done** | 0 | self | 49 | **done** | - | - | - | - | - | - | - | - |
| a4d2f7cf | L | end | 40 | self | 42 | end | - | - | - | - | - | - | - | - |
| a4d2f7cf | L | end | 30 | self | 62 | end | - | - | - | - | - | - | - | - |
| a4d2f7cf | L | end | 20 | self | 108 | end | - | - | - | - | - | - | - | - |
| a4d2f7cf | L | end | 10 | self | 139 | end | - | - | - | - | - | - | - | - |
| a4d2f7cf | L | end | 0 | self | 165 | end | - | - | - | - | - | - | - | - |
| c6c01d25 | L | **done** | 40 | self | 17 | **done** | - | - | - | - | - | - | - | - |
| c6c01d25 | L | **done** | 30 | self =D40 | 17 | **done** | - | - | - | - | - | - | - | - |
| c6c01d25 | L | **done** | 20 | self =D40 | 17 | **done** | - | - | - | - | - | - | - | - |
| c6c01d25 | L | **done** | 10 | self =D40 | 17 | **done** | - | - | - | - | - | - | - | - |
| c6c01d25 | L | **done** | 0 | self =D40 | 17 | **done** | - | - | - | - | - | - | - | - |
| e5e47fd2 | L | corr | 40 | self | 17 | corr | - | - | - | - | - | - | - | - |
| e5e47fd2 | L | corr | 30 | self | 25 | corr | - | - | - | - | - | - | - | - |
| e5e47fd2 | L | corr | 20 | self | 33 | corr | - | - | - | - | - | - | - | - |
| e5e47fd2 | L | corr | 10 | self | 41 | corr | - | - | - | - | - | - | - | - |
| e5e47fd2 | L | corr | 0 | self | 48 | corr | - | - | - | - | - | - | - | - |
| 1594e89e | R | coll | 40 | self | 17 | coll | - | - | - | - | - | - | - | - |
| 1594e89e | R | coll | 30 | self =D40 | 17 | coll | - | - | - | - | - | - | - | - |
| 1594e89e | R | coll | 20 | self =D40 | 17 | coll | - | - | - | - | - | - | - | - |
| 1594e89e | R | coll | 10 | self =D40 | 17 | coll | - | - | - | - | - | - | - | - |
| 1594e89e | R | coll | 0 | self | 22 | coll | - | - | - | - | - | - | - | - |
| 1e8e939e | R | coll | 40 | self | 43 | coll | - | - | - | - | - | - | - | - |
| 1e8e939e | R | coll | 30 | self | 50 | coll | - | - | - | - | - | - | - | - |
| 1e8e939e | R | coll | 20 | self | 55 | coll | - | - | - | - | - | - | - | - |
| 1e8e939e | R | coll | 10 | self | 64 | coll | - | - | - | - | - | - | - | - |
| 1e8e939e | R | coll | 0 | log | 121 | - | - | - | - | - | - | - | - | - |
| 232d240e | R | offr | 40 | self | 17 | offr | - | - | - | - | - | - | - | - |
| 232d240e | R | offr | 30 | self =D40 | 17 | offr | - | - | - | - | - | - | - | - |
| 232d240e | R | offr | 20 | self =D40 | 17 | offr | - | - | - | - | - | - | - | - |
| 232d240e | R | offr | 10 | self =D40 | 17 | offr | - | - | - | - | - | - | - | - |
| 232d240e | R | offr | 0 | self | 26 | offr | - | - | - | - | - | - | - | - |
| 268949d2 | R | **done** | 40 | self | 17 | **done** | - | - | - | - | - | - | - | - |
| 268949d2 | R | **done** | 30 | self =D40 | 17 | **done** | - | - | - | - | - | - | - | - |
| 268949d2 | R | **done** | 20 | self =D40 | 17 | **done** | - | - | - | - | - | - | - | - |
| 268949d2 | R | **done** | 10 | self =D40 | 17 | **done** | - | - | - | - | - | - | - | - |
| 268949d2 | R | **done** | 0 | self | 62 | **done** | - | - | - | - | - | - | - | - |
| 3232fdb7 | R | corr | 40 | self | 64 | corr | - | - | - | - | - | - | - | - |
| 3232fdb7 | R | corr | 30 | self | 87 | corr | - | - | - | - | - | - | - | - |
| 3232fdb7 | R | corr | 20 | self | 115 | corr | - | - | - | - | - | - | - | - |
| 3232fdb7 | R | corr | 10 | self | 149 | corr | - | - | - | - | - | - | - | - |
| 3232fdb7 | R | corr | 0 | log | 148 | - | - | - | - | - | - | - | - | - |
| 37d660b1 | R | corr | 40 | self | 17 | corr | - | - | - | - | - | - | - | - |
| 37d660b1 | R | corr | 30 | self =D40 | 17 | corr | - | - | - | - | - | - | - | - |
| 37d660b1 | R | corr | 20 | self =D40 | 17 | corr | - | - | - | - | - | - | - | - |
| 37d660b1 | R | corr | 10 | self | 30 | corr | - | - | - | - | - | - | - | - |
| 37d660b1 | R | corr | 0 | self | 54 | corr | - | - | - | - | - | - | - | - |
| 403d558a | R | corr | 40 | self | 17 | corr | - | - | - | - | - | - | - | - |
| 403d558a | R | corr | 30 | self =D40 | 17 | corr | - | - | - | - | - | - | - | - |
| 403d558a | R | corr | 20 | self =D40 | 17 | corr | - | - | - | - | - | - | - | - |
| 403d558a | R | corr | 10 | self | 26 | corr | - | - | - | - | - | - | - | - |
| 403d558a | R | corr | 0 | self | 36 | corr | - | - | - | - | - | - | - | - |
| 77fdd75c | R | **done** | 40 | self | 17 | **done** | - | - | - | - | - | - | - | - |
| 77fdd75c | R | **done** | 30 | self =D40 | 17 | **done** | - | - | - | - | - | - | - | - |
| 77fdd75c | R | **done** | 20 | self =D40 | 17 | **done** | - | - | - | - | - | - | - | - |
| 77fdd75c | R | **done** | 10 | self =D40 | 17 | **done** | - | - | - | - | - | - | - | - |
| 77fdd75c | R | **done** | 0 | self =D40 | 17 | **done** | - | - | - | - | - | - | - | - |
| 96da4f1a | R | **done** | 40 | self | 17 | **done** | - | - | - | - | - | - | - | - |
| 96da4f1a | R | **done** | 30 | self =D40 | 17 | **done** | - | - | - | - | - | - | - | - |
| 96da4f1a | R | **done** | 20 | self =D40 | 17 | **done** | - | - | - | - | - | - | - | - |
| 96da4f1a | R | **done** | 10 | self =D40 | 17 | **done** | - | - | - | - | - | - | - | - |
| 96da4f1a | R | **done** | 0 | self | 47 | **done** | - | - | - | - | - | - | - | - |
| 98694f91 | R | corr | 40 | self | 17 | corr | - | - | - | - | - | - | - | - |
| 98694f91 | R | corr | 30 | self | 27 | corr | - | - | - | - | - | - | - | - |
| 98694f91 | R | corr | 20 | self | 43 | corr | - | - | - | - | - | - | - | - |
| 98694f91 | R | corr | 10 | self | 90 | corr | - | - | - | - | - | - | - | - |
| 98694f91 | R | corr | 0 | self | 128 | corr | - | - | - | - | - | - | - | - |
| a174faed | R | corr | 40 | self | 17 | corr | - | - | - | - | - | - | - | - |
| a174faed | R | corr | 30 | self | 44 | corr | - | - | - | - | - | - | - | - |
| a174faed | R | corr | 20 | self | 63 | corr | - | - | - | - | - | - | - | - |
| a174faed | R | corr | 10 | self | 80 | corr | - | - | - | - | - | - | - | - |
| a174faed | R | corr | 0 | self | 97 | corr | - | - | - | - | - | - | - | - |
| b40f4729 | R | **done** | 40 | self | 17 | **done** | - | - | - | - | - | - | - | - |
| b40f4729 | R | **done** | 30 | self =D40 | 17 | **done** | - | - | - | - | - | - | - | - |
| b40f4729 | R | **done** | 20 | self =D40 | 17 | **done** | - | - | - | - | - | - | - | - |
| b40f4729 | R | **done** | 10 | self =D40 | 17 | **done** | - | - | - | - | - | - | - | - |
| b40f4729 | R | **done** | 0 | self | 34 | **done** | - | - | - | - | - | - | - | - |
| d0b9a872 | R | offr | 40 | self | 17 | offr | - | - | - | - | - | - | - | - |
| d0b9a872 | R | offr | 30 | self =D40 | 17 | offr | - | - | - | - | - | - | - | - |
| d0b9a872 | R | offr | 20 | self =D40 | 17 | offr | - | - | - | - | - | - | - | - |
| d0b9a872 | R | offr | 10 | self =D40 | 17 | offr | - | - | - | - | - | - | - | - |
| d0b9a872 | R | offr | 0 | self | 24 | offr | - | - | - | - | - | - | - | - |
| dc1966f4 | R | **done** | 40 | self | 19 | **done** | - | - | - | - | - | - | - | - |
| dc1966f4 | R | **done** | 30 | self | 30 | **done** | - | - | - | - | - | - | - | - |
| dc1966f4 | R | **done** | 20 | self | 46 | **done** | - | - | - | - | - | - | - | - |
| dc1966f4 | R | **done** | 10 | self | 101 | **done** | - | - | - | - | - | - | - | - |
| dc1966f4 | R | **done** | 0 | self | 131 | **done** | - | - | - | - | - | - | - | - |
| ddc3e8df | R | corr | 40 | self | 17 | corr | - | - | - | - | - | - | - | - |
| ddc3e8df | R | corr | 30 | self =D40 | 17 | corr | - | - | - | - | - | - | - | - |
| ddc3e8df | R | corr | 20 | self =D40 | 17 | corr | - | - | - | - | - | - | - | - |
| ddc3e8df | R | corr | 10 | self =D40 | 17 | corr | - | - | - | - | - | - | - | - |
| ddc3e8df | R | corr | 0 | self =D40 | 17 | corr | - | - | - | - | - | - | - | - |
| ea468790 | R | **done** | 40 | self | 17 | **done** | - | - | - | - | - | - | - | - |
| ea468790 | R | **done** | 30 | self =D40 | 17 | **done** | - | - | - | - | - | - | - | - |
| ea468790 | R | **done** | 20 | self =D40 | 17 | **done** | - | - | - | - | - | - | - | - |
| ea468790 | R | **done** | 10 | self =D40 | 17 | **done** | - | - | - | - | - | - | - | - |
| ea468790 | R | **done** | 0 | self | 96 | **done** | - | - | - | - | - | - | - | - |
| eaba3f6a | R | corr | 40 | self | 17 | corr | - | - | - | - | - | - | - | - |
| eaba3f6a | R | corr | 30 | self | 25 | corr | - | - | - | - | - | - | - | - |
| eaba3f6a | R | corr | 20 | self | 36 | corr | - | - | - | - | - | - | - | - |
| eaba3f6a | R | corr | 10 | self | 45 | corr | - | - | - | - | - | - | - | - |
| eaba3f6a | R | corr | 0 | self | 53 | corr | - | - | - | - | - | - | - | - |
| f9e361aa | R | corr | 40 | self | 23 | corr | - | - | - | - | - | - | - | - |
| f9e361aa | R | corr | 30 | self | 34 | corr | - | - | - | - | - | - | - | - |
| f9e361aa | R | corr | 20 | self | 44 | corr | - | - | - | - | - | - | - | - |
| f9e361aa | R | corr | 10 | self | 54 | corr | - | - | - | - | - | - | - | - |
| f9e361aa | R | corr | 0 | self | 63 | corr | - | - | - | - | - | - | - | - |

## Coverage

- turns: 27; identity (the policy alone) completes 9 / 27
- turns with >= 1 completing branch cell at some distance: 0 / 27; at >= 2 distinct handovers: 0 / 27
- determinism: self branches whose plans before h differ from phase A's identity at the same k: 0 of 0
- **line: turns with completing cells at >= 2 distinct handovers with realized entry speed <= v_e + 1 m/s: 0 / 27 = 0 % -> FAIL** (pass >= 60 %, fail <= 35 %); self handovers only: 0 / 27
- turns with >= 1 completing cell including the identity: 9 / 27
- row yield (completing cells that pass the full shape gate: class p95 kappa / dkappa, a_lat <= human p95, entry speed <= v_e + 1): turns with >= 1: 0 / 27; at >= 2 distinct handovers: 0 / 27
- per distance (turns with >= 1 completing cell): D40 0, D30 0, D20 0, D10 0, D0 0

| cells | rollouts | done | done + gate | zeros | rear-ended | crawl (approach) | crawl inside | median entry speed (done) | mean score |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|

- human entry speed (log, strict entry) median 4.01 m/s over these turns; fallback (src=log) distances: 2 of 83
