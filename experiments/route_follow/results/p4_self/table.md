# P4 self-branch read (693 rollouts, 27 scenes)

`done` = turn completed (P2's rule). Otherwise the zero type (corr / offr / coll / oth) or rear (rear-ended before the target) / end.
Cells g<gain>-<profile>: raw = the plan's own speed, ve = profile to v_e, ve08 = to 0.8 v_e. `id` = the policy alone (phase A).
(log) = unreached distance, handover from the log (src=log). `=D<n>` = same handover decision as D<n> (run once).

| scene | side | id | D | src | h | g1-raw | g1-ve | g1-ve08 | g1.25-raw | g1.25-ve | g1.25-ve08 | g1.5-raw | g1.5-ve | g1.5-ve08 |
|:--|:--|:--|--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 0d10d2e3 | L | coll | 40 | self | 17 | coll | coll | coll | coll | coll | coll | corr | corr | corr |
| 0d10d2e3 | L | coll | 30 | self =D40 | 17 | coll | coll | coll | coll | coll | coll | corr | corr | corr |
| 0d10d2e3 | L | coll | 20 | self =D40 | 17 | coll | coll | coll | coll | coll | coll | corr | corr | corr |
| 0d10d2e3 | L | coll | 10 | self | 29 | coll | coll | coll | corr | corr | corr | corr | corr | corr |
| 0d10d2e3 | L | coll | 0 | self | 68 | coll | coll | coll | coll | coll | coll | coll | coll | coll |
| 213dfdac | L | offr | 40 | self | 19 | offr | offr | offr | corr | corr | corr | offr | offr | offr |
| 213dfdac | L | offr | 30 | self | 29 | offr | offr | offr | corr | corr | corr | offr | offr | offr |
| 213dfdac | L | offr | 20 | self | 42 | offr | offr | offr | offr | offr | offr | offr | offr | offr |
| 213dfdac | L | offr | 10 | self | 77 | offr | offr | offr | offr | offr | offr | offr | offr | offr |
| 213dfdac | L | offr | 0 | self | 111 | offr | offr | offr | **done** | end | end | offr | end | end |
| 33c8976b | L | offr | 40 | self | 17 | offr | offr | offr | offr | offr | offr | offr | offr | offr |
| 33c8976b | L | offr | 30 | self =D40 | 17 | offr | offr | offr | offr | offr | offr | offr | offr | offr |
| 33c8976b | L | offr | 20 | self =D40 | 17 | offr | offr | offr | offr | offr | offr | offr | offr | offr |
| 33c8976b | L | offr | 10 | self =D40 | 17 | offr | offr | offr | offr | offr | offr | offr | offr | offr |
| 33c8976b | L | offr | 0 | self | 34 | offr | offr | offr | offr | offr | offr | offr | offr | offr |
| 8a365ac2 | L | offr | 40 | self | 17 | offr | offr | offr | **done** | **done** | offr | **done** | **done** | **done** |
| 8a365ac2 | L | offr | 30 | self =D40 | 17 | offr | offr | offr | **done** | **done** | offr | **done** | **done** | **done** |
| 8a365ac2 | L | offr | 20 | self =D40 | 17 | offr | offr | offr | **done** | **done** | offr | **done** | **done** | **done** |
| 8a365ac2 | L | offr | 10 | self =D40 | 17 | offr | offr | offr | **done** | **done** | offr | **done** | **done** | **done** |
| 8a365ac2 | L | offr | 0 | self | 52 | offr | offr | offr | **done** | **done** | **done** | corr | **done** | **done** |
| 96eb4700 | L | **done** | 40 | self | 17 | **done** | **done** | **done** | **done** | **done** | **done** | **done** | offr | offr |
| 96eb4700 | L | **done** | 30 | self =D40 | 17 | **done** | **done** | **done** | **done** | **done** | **done** | **done** | offr | offr |
| 96eb4700 | L | **done** | 20 | self =D40 | 17 | **done** | **done** | **done** | **done** | **done** | **done** | **done** | offr | offr |
| 96eb4700 | L | **done** | 10 | self =D40 | 17 | **done** | **done** | **done** | **done** | **done** | **done** | **done** | offr | offr |
| 96eb4700 | L | **done** | 0 | self =D40 | 17 | **done** | **done** | **done** | **done** | **done** | **done** | **done** | offr | offr |
| a4be61d6 | L | **done** | 40 | self | 17 | **done** | **done** | **done** | **done** | **done** | **done** | corr | corr | corr |
| a4be61d6 | L | **done** | 30 | self =D40 | 17 | **done** | **done** | **done** | **done** | **done** | **done** | corr | corr | corr |
| a4be61d6 | L | **done** | 20 | self =D40 | 17 | **done** | **done** | **done** | **done** | **done** | **done** | corr | corr | corr |
| a4be61d6 | L | **done** | 10 | self =D40 | 17 | **done** | **done** | **done** | **done** | **done** | **done** | corr | corr | corr |
| a4be61d6 | L | **done** | 0 | self | 49 | **done** | **done** | **done** | end | **done** | **done** | corr | corr | end |
| a4d2f7cf | L | end | 40 | self | 42 | end | end | end | end | end | end | end | end | end |
| a4d2f7cf | L | end | 30 | self | 62 | end | end | end | end | end | end | corr | corr | corr |
| a4d2f7cf | L | end | 20 | self | 108 | end | end | end | end | end | end | end | end | end |
| a4d2f7cf | L | end | 10 | self | 139 | end | end | end | corr | corr | corr | corr | corr | corr |
| a4d2f7cf | L | end | 0 | self | 165 | end | end | end | end | end | end | corr | corr | corr |
| c6c01d25 | L | **done** | 40 | self | 17 | **done** | **done** | **done** | **done** | **done** | **done** | corr | coll | corr |
| c6c01d25 | L | **done** | 30 | self =D40 | 17 | **done** | **done** | **done** | **done** | **done** | **done** | corr | coll | corr |
| c6c01d25 | L | **done** | 20 | self =D40 | 17 | **done** | **done** | **done** | **done** | **done** | **done** | corr | coll | corr |
| c6c01d25 | L | **done** | 10 | self =D40 | 17 | **done** | **done** | **done** | **done** | **done** | **done** | corr | coll | corr |
| c6c01d25 | L | **done** | 0 | self =D40 | 17 | **done** | **done** | **done** | **done** | **done** | **done** | corr | coll | corr |
| e5e47fd2 | L | corr | 40 | self | 17 | corr | corr | rear | corr | corr | **done** | corr | corr | rear |
| e5e47fd2 | L | corr | 30 | self | 25 | corr | corr | rear | corr | corr | **done** | corr | corr | rear |
| e5e47fd2 | L | corr | 20 | self | 33 | corr | corr | corr | corr | corr | corr | corr | corr | corr |
| e5e47fd2 | L | corr | 10 | self | 41 | corr | corr | corr | corr | corr | corr | corr | corr | corr |
| e5e47fd2 | L | corr | 0 | self | 48 | corr | corr | corr | corr | corr | corr | corr | corr | corr |
| 1594e89e | R | coll | 40 | self | 17 | coll | coll | coll | coll | coll | coll | coll | coll | coll |
| 1594e89e | R | coll | 30 | self =D40 | 17 | coll | coll | coll | coll | coll | coll | coll | coll | coll |
| 1594e89e | R | coll | 20 | self =D40 | 17 | coll | coll | coll | coll | coll | coll | coll | coll | coll |
| 1594e89e | R | coll | 10 | self =D40 | 17 | coll | coll | coll | coll | coll | coll | coll | coll | coll |
| 1594e89e | R | coll | 0 | self | 22 | coll | coll | coll | coll | coll | coll | offr | offr | offr |
| 1e8e939e | R | coll | 40 | self | 43 | coll | coll | coll | coll | coll | coll | coll | corr | corr |
| 1e8e939e | R | coll | 30 | self | 50 | coll | coll | coll | coll | coll | coll | coll | coll | coll |
| 1e8e939e | R | coll | 20 | self | 55 | coll | coll | coll | coll | coll | coll | coll | coll | coll |
| 1e8e939e | R | coll | 10 | self | 64 | coll | coll | coll | coll | coll | coll | coll | coll | coll |
| 1e8e939e | R | coll | 0 | log | 121 | **done** | **done** | **done** | **done** | **done** | **done** | offr | **done** | **done** |
| 232d240e | R | offr | 40 | self | 17 | offr | offr | offr | **done** | **done** | **done** | **done** | offr | offr |
| 232d240e | R | offr | 30 | self =D40 | 17 | offr | offr | offr | **done** | **done** | **done** | **done** | offr | offr |
| 232d240e | R | offr | 20 | self =D40 | 17 | offr | offr | offr | **done** | **done** | **done** | **done** | offr | offr |
| 232d240e | R | offr | 10 | self =D40 | 17 | offr | offr | offr | **done** | **done** | **done** | **done** | offr | offr |
| 232d240e | R | offr | 0 | self | 26 | offr | offr | offr | **done** | **done** | **done** | **done** | offr | offr |
| 268949d2 | R | **done** | 40 | self | 17 | **done** | corr | **done** | **done** | corr | **done** | **done** | corr | **done** |
| 268949d2 | R | **done** | 30 | self =D40 | 17 | **done** | corr | **done** | **done** | corr | **done** | **done** | corr | **done** |
| 268949d2 | R | **done** | 20 | self =D40 | 17 | **done** | corr | **done** | **done** | corr | **done** | **done** | corr | **done** |
| 268949d2 | R | **done** | 10 | self =D40 | 17 | **done** | corr | **done** | **done** | corr | **done** | **done** | corr | **done** |
| 268949d2 | R | **done** | 0 | self | 62 | **done** | corr | **done** | **done** | **done** | **done** | offr | offr | offr |
| 3232fdb7 | R | corr | 40 | self | 64 | corr | corr | corr | corr | corr | corr | corr | corr | corr |
| 3232fdb7 | R | corr | 30 | self | 87 | corr | corr | corr | corr | corr | corr | corr | corr | corr |
| 3232fdb7 | R | corr | 20 | self | 115 | corr | corr | corr | corr | corr | corr | corr | corr | corr |
| 3232fdb7 | R | corr | 10 | self | 149 | corr | corr | corr | corr | corr | corr | corr | corr | corr |
| 3232fdb7 | R | corr | 0 | log | 148 | coll | coll | corr | offr | offr | end | offr | offr | offr |
| 37d660b1 | R | corr | 40 | self | 17 | corr | corr | corr | corr | corr | corr | corr | corr | corr |
| 37d660b1 | R | corr | 30 | self =D40 | 17 | corr | corr | corr | corr | corr | corr | corr | corr | corr |
| 37d660b1 | R | corr | 20 | self =D40 | 17 | corr | corr | corr | corr | corr | corr | corr | corr | corr |
| 37d660b1 | R | corr | 10 | self | 30 | corr | corr | corr | **done** | **done** | corr | **done** | offr | **done** |
| 37d660b1 | R | corr | 0 | self | 54 | corr | corr | corr | **done** | **done** | **done** | offr | offr | offr |
| 403d558a | R | corr | 40 | self | 17 | corr | corr | rear | corr | corr | rear | corr | corr | **done** |
| 403d558a | R | corr | 30 | self =D40 | 17 | corr | corr | rear | corr | corr | rear | corr | corr | **done** |
| 403d558a | R | corr | 20 | self =D40 | 17 | corr | corr | rear | corr | corr | rear | corr | corr | **done** |
| 403d558a | R | corr | 10 | self | 26 | corr | corr | corr | corr | corr | corr | corr | corr | corr |
| 403d558a | R | corr | 0 | self | 36 | corr | corr | corr | corr | corr | corr | corr | corr | corr |
| 77fdd75c | R | **done** | 40 | self | 17 | **done** | **done** | **done** | offr | offr | offr | offr | offr | offr |
| 77fdd75c | R | **done** | 30 | self =D40 | 17 | **done** | **done** | **done** | offr | offr | offr | offr | offr | offr |
| 77fdd75c | R | **done** | 20 | self =D40 | 17 | **done** | **done** | **done** | offr | offr | offr | offr | offr | offr |
| 77fdd75c | R | **done** | 10 | self =D40 | 17 | **done** | **done** | **done** | offr | offr | offr | offr | offr | offr |
| 77fdd75c | R | **done** | 0 | self =D40 | 17 | **done** | **done** | **done** | offr | offr | offr | offr | offr | offr |
| 96da4f1a | R | **done** | 40 | self | 17 | **done** | corr | corr | offr | corr | corr | offr | **done** | corr |
| 96da4f1a | R | **done** | 30 | self =D40 | 17 | **done** | corr | corr | offr | corr | corr | offr | **done** | corr |
| 96da4f1a | R | **done** | 20 | self =D40 | 17 | **done** | corr | corr | offr | corr | corr | offr | **done** | corr |
| 96da4f1a | R | **done** | 10 | self =D40 | 17 | **done** | corr | corr | offr | corr | corr | offr | **done** | corr |
| 96da4f1a | R | **done** | 0 | self | 47 | **done** | coll | corr | offr | **done** | **done** | offr | offr | offr |
| 98694f91 | R | corr | 40 | self | 17 | corr | corr | corr | corr | corr | corr | corr | corr | corr |
| 98694f91 | R | corr | 30 | self | 27 | corr | corr | corr | corr | corr | corr | corr | corr | corr |
| 98694f91 | R | corr | 20 | self | 43 | corr | corr | corr | corr | corr | corr | corr | corr | corr |
| 98694f91 | R | corr | 10 | self | 90 | corr | corr | corr | corr | corr | corr | corr | corr | corr |
| 98694f91 | R | corr | 0 | self | 128 | corr | corr | corr | **done** | **done** | end | offr | offr | offr |
| a174faed | R | corr | 40 | self | 17 | corr | corr | corr | corr | corr | corr | corr | corr | corr |
| a174faed | R | corr | 30 | self | 44 | corr | corr | corr | corr | corr | corr | corr | corr | corr |
| a174faed | R | corr | 20 | self | 63 | corr | corr | corr | corr | corr | corr | corr | corr | corr |
| a174faed | R | corr | 10 | self | 80 | corr | corr | corr | corr | corr | corr | coll | end | **done** |
| a174faed | R | corr | 0 | self | 97 | corr | corr | corr | corr | corr | corr | corr | **done** | **done** |
| b40f4729 | R | **done** | 40 | self | 17 | **done** | corr | corr | **done** | **done** | **done** | coll | coll | coll |
| b40f4729 | R | **done** | 30 | self =D40 | 17 | **done** | corr | corr | **done** | **done** | **done** | coll | coll | coll |
| b40f4729 | R | **done** | 20 | self =D40 | 17 | **done** | corr | corr | **done** | **done** | **done** | coll | coll | coll |
| b40f4729 | R | **done** | 10 | self =D40 | 17 | **done** | corr | corr | **done** | **done** | **done** | coll | coll | coll |
| b40f4729 | R | **done** | 0 | self | 34 | **done** | corr | corr | **done** | **done** | **done** | coll | coll | coll |
| d0b9a872 | R | offr | 40 | self | 17 | offr | offr | offr | rear | rear | rear | rear | rear | rear |
| d0b9a872 | R | offr | 30 | self =D40 | 17 | offr | offr | offr | rear | rear | rear | rear | rear | rear |
| d0b9a872 | R | offr | 20 | self =D40 | 17 | offr | offr | offr | rear | rear | rear | rear | rear | rear |
| d0b9a872 | R | offr | 10 | self =D40 | 17 | offr | offr | offr | rear | rear | rear | rear | rear | rear |
| d0b9a872 | R | offr | 0 | self | 24 | offr | offr | offr | rear | rear | rear | rear | rear | rear |
| dc1966f4 | R | **done** | 40 | self | 19 | **done** | end | end | **done** | **done** | **done** | **done** | **done** | end |
| dc1966f4 | R | **done** | 30 | self | 30 | **done** | end | end | **done** | **done** | end | **done** | end | end |
| dc1966f4 | R | **done** | 20 | self | 46 | **done** | end | end | end | end | end | **done** | end | end |
| dc1966f4 | R | **done** | 10 | self | 101 | **done** | end | end | coll | coll | coll | offr | offr | offr |
| dc1966f4 | R | **done** | 0 | self | 131 | **done** | end | end | **done** | **done** | **done** | coll | coll | coll |
| ddc3e8df | R | corr | 40 | self | 17 | corr | corr | corr | **done** | **done** | **done** | offr | offr | offr |
| ddc3e8df | R | corr | 30 | self =D40 | 17 | corr | corr | corr | **done** | **done** | **done** | offr | offr | offr |
| ddc3e8df | R | corr | 20 | self =D40 | 17 | corr | corr | corr | **done** | **done** | **done** | offr | offr | offr |
| ddc3e8df | R | corr | 10 | self =D40 | 17 | corr | corr | corr | **done** | **done** | **done** | offr | offr | offr |
| ddc3e8df | R | corr | 0 | self =D40 | 17 | corr | corr | corr | **done** | **done** | **done** | offr | offr | offr |
| ea468790 | R | **done** | 40 | self | 17 | **done** | **done** | end | **done** | **done** | end | **done** | end | end |
| ea468790 | R | **done** | 30 | self =D40 | 17 | **done** | **done** | end | **done** | **done** | end | **done** | end | end |
| ea468790 | R | **done** | 20 | self =D40 | 17 | **done** | **done** | end | **done** | **done** | end | **done** | end | end |
| ea468790 | R | **done** | 10 | self =D40 | 17 | **done** | **done** | end | **done** | **done** | end | **done** | end | end |
| ea468790 | R | **done** | 0 | self | 96 | **done** | **done** | end | **done** | **done** | rear | offr | offr | offr |
| eaba3f6a | R | corr | 40 | self | 17 | corr | corr | corr | corr | corr | corr | corr | corr | corr |
| eaba3f6a | R | corr | 30 | self | 25 | corr | corr | corr | corr | corr | corr | corr | corr | corr |
| eaba3f6a | R | corr | 20 | self | 36 | corr | corr | corr | corr | corr | corr | corr | corr | corr |
| eaba3f6a | R | corr | 10 | self | 45 | corr | corr | corr | corr | corr | corr | corr | corr | corr |
| eaba3f6a | R | corr | 0 | self | 53 | corr | corr | corr | corr | corr | corr | corr | corr | corr |
| f9e361aa | R | corr | 40 | self | 23 | corr | corr | coll | corr | corr | coll | corr | corr | coll |
| f9e361aa | R | corr | 30 | self | 34 | corr | corr | coll | corr | corr | coll | corr | corr | coll |
| f9e361aa | R | corr | 20 | self | 44 | corr | corr | coll | corr | **done** | **done** | corr | offr | offr |
| f9e361aa | R | corr | 10 | self | 54 | corr | corr | corr | corr | corr | corr | corr | corr | corr |
| f9e361aa | R | corr | 0 | self | 63 | corr | corr | corr | corr | corr | corr | corr | corr | corr |

## Coverage

- turns: 27; identity (the policy alone) completes 9 / 27
- turns with >= 1 completing branch cell at some distance: 20 / 27; at >= 2 distinct handovers: 11 / 27
- determinism: self branches whose plans before h differ from phase A's identity at the same k: 0 of 648
- **line: turns with completing cells at >= 2 distinct handovers with realized entry speed <= v_e + 1 m/s: 10 / 27 = 37 % -> INCONCLUSIVE** (pass >= 60 %, fail <= 35 %); self handovers only: 10 / 27
- row yield (completing cells that pass the full shape gate: class p95 kappa / dkappa, a_lat <= human p95, entry speed <= v_e + 1): turns with >= 1: 13 / 27; at >= 2 distinct handovers: 7 / 27
- per distance (turns with >= 1 completing cell): D40 14, D30 2, D20 2, D10 2, D0 13

| cells | rollouts | done | done + gate | zeros | rear-ended | crawl (approach) | crawl inside | median entry speed (done) | mean score |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| profile raw | 168 | 33 | 18 | 123 | 6 | 55 | 11 | 3.77 | 0.225 |
| profile ve | 249 | 37 | 23 | 187 | 6 | 82 | 21 | 3.76 | 0.213 |
| profile ve08 | 249 | 36 | 20 | 173 | 24 | 82 | 25 | 3.74 | 0.228 |
| gain 1 | 168 | 17 | 13 | 128 | 6 | 54 | 21 | 3.76 | 0.200 |
| gain 1.25 | 249 | 65 | 36 | 154 | 18 | 81 | 17 | 3.68 | 0.309 |
| gain 1.5 | 249 | 24 | 12 | 201 | 12 | 84 | 19 | 4.02 | 0.150 |
| D40 | 216 | 50 | 30 | 143 | 15 | 115 | 17 | 3.74 | 0.256 |
| D30 | 80 | 4 | 2 | 64 | 3 | 40 | 6 | 4.41 | 0.175 |
| D20 | 80 | 3 | 2 | 62 | 0 | 32 | 8 | 3.90 | 0.215 |
| D10 | 104 | 5 | 3 | 94 | 1 | 32 | 4 | 4.40 | 0.090 |
| D0 | 186 | 44 | 24 | 120 | 17 | 0 | 22 | 3.66 | 0.279 |
| src self | 648 | 98 | 53 | 474 | 33 | 219 | 56 | 3.68 | 0.215 |
| src log (fallback) | 18 | 8 | 8 | 9 | 3 | 0 | 1 | 4.59 | 0.463 |

- human entry speed (log, strict entry) median 4.01 m/s over these turns; fallback (src=log) distances: 2 of 83
