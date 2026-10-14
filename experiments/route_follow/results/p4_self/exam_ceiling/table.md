# P4 self-branch read (465 rollouts, 15 scenes)

`done` = turn completed (P2's rule). Otherwise the zero type (corr / offr / coll / oth) or rear (rear-ended before the target) / end.
Cells g<gain>-<profile>: raw = the plan's own speed, ve = profile to v_e, ve08 = to 0.8 v_e. `id` = the policy alone (phase A).
(log) = unreached distance, handover from the log (src=log). `=D<n>` = same handover decision as D<n> (run once).

| scene | side | id | D | src | h | g1-raw | g1-ve | g1-ve08 | g1.25-raw | g1.25-ve | g1.25-ve08 | g1.5-raw | g1.5-ve | g1.5-ve08 |
|:--|:--|:--|--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 22fa5c92 | L | coll | 40 | self | 17 | coll | coll | coll | coll | coll | coll | coll | coll | coll |
| 22fa5c92 | L | coll | 30 | self =D40 | 17 | coll | coll | coll | coll | coll | coll | coll | coll | coll |
| 22fa5c92 | L | coll | 20 | self =D40 | 17 | coll | coll | coll | coll | coll | coll | coll | coll | coll |
| 22fa5c92 | L | coll | 10 | self =D40 | 17 | coll | coll | coll | coll | coll | coll | coll | coll | coll |
| 22fa5c92 | L | coll | 0 | self | 72 | coll | coll | coll | coll | coll | coll | coll | coll | coll |
| 382a7247 | L | **done** | 40 | self | 17 | **done** | offr | offr | **done** | offr | offr | corr | corr | corr |
| 382a7247 | L | **done** | 30 | self =D40 | 17 | **done** | offr | offr | **done** | offr | offr | corr | corr | corr |
| 382a7247 | L | **done** | 20 | self | 22 | **done** | offr | offr | coll | **done** | **done** | corr | corr | corr |
| 382a7247 | L | **done** | 10 | self | 39 | **done** | offr | offr | corr | corr | corr | corr | corr | corr |
| 382a7247 | L | **done** | 0 | self | 51 | **done** | **done** | **done** | corr | corr | corr | corr | corr | corr |
| a1b166cc | L | corr | 40 | self | 17 | corr | corr | rear | **done** | rear | rear | rear | rear | rear |
| a1b166cc | L | corr | 30 | self =D40 | 17 | corr | corr | rear | **done** | rear | rear | rear | rear | rear |
| a1b166cc | L | corr | 20 | self =D40 | 17 | corr | corr | rear | **done** | rear | rear | rear | rear | rear |
| a1b166cc | L | corr | 10 | self =D40 | 17 | corr | corr | rear | **done** | rear | rear | rear | rear | rear |
| a1b166cc | L | corr | 0 | self | 76 | corr | corr | rear | offr | offr | end | corr | corr | corr |
| abd45a30 | L | rear | 40 | self | 93 | rear | rear | rear | rear | rear | rear | rear | rear | rear |
| abd45a30 | L | rear | 30 | self | 124 | rear | rear | rear | rear | rear | rear | rear | rear | rear |
| abd45a30 | L | rear | 20 | self | 160 | rear | rear | rear | rear | rear | rear | rear | rear | rear |
| abd45a30 | L | rear | 10 | self | 189 | rear | rear | rear | rear | rear | rear | rear | rear | rear |
| abd45a30 | L | rear | 0 | log | 115 | offr | offr | offr | **done** | **done** | **done** | coll | coll | coll |
| ce161092 | L | offr | 40 | self | 59 | offr | offr | offr | offr | offr | offr | offr | offr | offr |
| ce161092 | L | offr | 30 | self | 72 | offr | offr | offr | offr | offr | offr | **done** | offr | offr |
| ce161092 | L | offr | 20 | self | 84 | offr | offr | offr | coll | **done** | rear | coll | coll | **done** |
| ce161092 | L | offr | 10 | self | 97 | offr | offr | rear | coll | rear | corr | coll | coll | coll |
| ce161092 | L | offr | 0 | self | 109 | offr | **done** | **done** | coll | coll | corr | coll | coll | coll |
| d90fe123 | L | **done** | 40 | self | 61 | **done** | **done** | rear | **done** | **done** | rear | **done** | **done** | rear |
| d90fe123 | L | **done** | 30 | self | 71 | **done** | **done** | rear | **done** | **done** | rear | **done** | **done** | **done** |
| d90fe123 | L | **done** | 20 | self | 82 | **done** | **done** | rear | end | **done** | **done** | corr | corr | corr |
| d90fe123 | L | **done** | 10 | self | 94 | **done** | **done** | **done** | corr | corr | corr | corr | corr | corr |
| d90fe123 | L | **done** | 0 | self | 106 | **done** | **done** | **done** | corr | corr | corr | corr | corr | corr |
| e37aa159 | L | corr | 40 | self | 17 | corr | corr | corr | corr | corr | corr | corr | corr | corr |
| e37aa159 | L | corr | 30 | self | 74 | corr | corr | corr | corr | rear | rear | corr | rear | rear |
| e37aa159 | L | corr | 20 | self | 97 | corr | corr | corr | corr | corr | end | corr | rear | rear |
| e37aa159 | L | corr | 10 | self | 114 | corr | corr | corr | **done** | corr | end | corr | rear | rear |
| e37aa159 | L | corr | 0 | self | 129 | corr | corr | corr | **done** | coll | coll | corr | corr | rear |
| fd01e9ab | L | corr | 40 | self | 17 | corr | corr | corr | coll | coll | corr | corr | corr | corr |
| fd01e9ab | L | corr | 30 | self =D40 | 17 | corr | corr | corr | coll | coll | corr | corr | corr | corr |
| fd01e9ab | L | corr | 20 | self =D40 | 17 | corr | corr | corr | coll | coll | corr | corr | corr | corr |
| fd01e9ab | L | corr | 10 | self =D40 | 17 | corr | corr | corr | coll | coll | corr | corr | corr | corr |
| fd01e9ab | L | corr | 0 | self | 53 | corr | corr | corr | **done** | corr | rear | corr | corr | corr |
| 09528162 | R | rear | 40 | self | 17 | rear | rear | rear | rear | rear | rear | **done** | **done** | **done** |
| 09528162 | R | rear | 30 | self =D40 | 17 | rear | rear | rear | rear | rear | rear | **done** | **done** | **done** |
| 09528162 | R | rear | 20 | self =D40 | 17 | rear | rear | rear | rear | rear | rear | **done** | **done** | **done** |
| 09528162 | R | rear | 10 | self | 25 | rear | rear | rear | rear | rear | rear | coll | coll | coll |
| 09528162 | R | rear | 0 | self | 46 | rear | rear | rear | rear | rear | rear | **done** | **done** | **done** |
| 2f227f2c | R | corr | 40 | self | 17 | corr | corr | corr | corr | corr | corr | corr | corr | corr |
| 2f227f2c | R | corr | 30 | self =D40 | 17 | corr | corr | corr | corr | corr | corr | corr | corr | corr |
| 2f227f2c | R | corr | 20 | self | 43 | corr | corr | corr | corr | corr | corr | corr | corr | corr |
| 2f227f2c | R | corr | 10 | self | 57 | corr | corr | corr | corr | corr | corr | corr | corr | corr |
| 2f227f2c | R | corr | 0 | log | 88 | **done** | **done** | **done** | **done** | **done** | **done** | **done** | **done** | **done** |
| 3ce13eed | R | corr | 40 | self | 17 | corr | offr | offr | corr | corr | offr | corr | corr | corr |
| 3ce13eed | R | corr | 30 | self =D40 | 17 | corr | offr | offr | corr | corr | offr | corr | corr | corr |
| 3ce13eed | R | corr | 20 | self =D40 | 17 | corr | offr | offr | corr | corr | offr | corr | corr | corr |
| 3ce13eed | R | corr | 10 | self | 25 | corr | offr | offr | corr | corr | corr | corr | corr | offr |
| 3ce13eed | R | corr | 0 | self | 59 | corr | corr | corr | **done** | **done** | **done** | **done** | **done** | coll |
| 94c1d722 | R | offr | 40 | self | 22 | offr | offr | offr | offr | offr | offr | offr | offr | offr |
| 94c1d722 | R | offr | 30 | self | 30 | offr | offr | offr | offr | offr | offr | offr | offr | offr |
| 94c1d722 | R | offr | 20 | self | 41 | offr | offr | offr | offr | offr | offr | offr | offr | offr |
| 94c1d722 | R | offr | 10 | self | 89 | offr | offr | offr | offr | offr | offr | offr | offr | offr |
| 94c1d722 | R | offr | 0 | self | 134 | offr | offr | offr | end | end | end | end | offr | offr |
| 9f393c8a | R | coll | 40 | self | 17 | coll | coll | coll | coll | coll | coll | offr | offr | offr |
| 9f393c8a | R | coll | 30 | self =D40 | 17 | coll | coll | coll | coll | coll | coll | offr | offr | offr |
| 9f393c8a | R | coll | 20 | self =D40 | 17 | coll | coll | coll | coll | coll | coll | offr | offr | offr |
| 9f393c8a | R | coll | 10 | self =D40 | 17 | coll | coll | coll | coll | coll | coll | offr | offr | offr |
| 9f393c8a | R | coll | 0 | self =D40 | 17 | coll | coll | coll | coll | coll | coll | offr | offr | offr |
| e0f6ebe1 | R | end | 40 | self | 57 | end | corr | corr | end | corr | corr | end | corr | corr |
| e0f6ebe1 | R | end | 30 | self | 79 | end | corr | corr | corr | end | corr | corr | end | corr |
| e0f6ebe1 | R | end | 20 | self | 95 | end | corr | corr | offr | end | corr | offr | offr | offr |
| e0f6ebe1 | R | end | 10 | self | 108 | end | corr | corr | offr | offr | offr | offr | offr | offr |
| e0f6ebe1 | R | end | 0 | self | 121 | end | corr | end | offr | **done** | **done** | offr | offr | offr |
| e848c843 | R | end | 40 | self | 25 | end | end | end | end | end | end | end | end | end |
| e848c843 | R | end | 30 | self | 44 | end | end | end | end | end | end | end | end | end |
| e848c843 | R | end | 20 | self | 69 | end | end | end | end | end | end | end | end | end |
| e848c843 | R | end | 10 | self | 102 | end | end | end | offr | offr | offr | offr | offr | offr |
| e848c843 | R | end | 0 | self | 135 | end | end | end | **done** | **done** | end | offr | offr | offr |

## Coverage

- turns: 15; identity (the policy alone) completes 2 / 15
- turns with >= 1 completing branch cell at some distance: 12 / 15; at >= 2 distinct handovers: 5 / 15
- determinism: self branches whose plans before h differ from phase A's identity at the same k: 0 of 432
- **line: turns with completing cells at >= 2 distinct handovers with realized entry speed <= v_e + 1 m/s: 2 / 15 = 13 % -> no pass / fail (ceiling read)**; self handovers only: 2 / 15
- turns with >= 1 completing cell including the identity: 12 / 15
- row yield (completing cells that pass the full shape gate: class p95 kappa / dkappa, a_lat <= human p95, entry speed <= v_e + 1): turns with >= 1: 7 / 15; at >= 2 distinct handovers: 1 / 15
- per distance (turns with >= 1 completing cell): D40 4, D30 2, D20 3, D10 2, D0 11

| cells | rollouts | done | done + gate | zeros | rear-ended | crawl (approach) | crawl inside | median entry speed (done) | mean score |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| profile raw | 114 | 19 | 6 | 73 | 13 | 19 | 6 | 4.43 | 0.293 |
| profile ve | 168 | 24 | 15 | 105 | 36 | 28 | 9 | 5.69 | 0.298 |
| profile ve08 | 168 | 16 | 8 | 99 | 42 | 28 | 18 | 4.67 | 0.307 |
| gain 1 | 114 | 14 | 8 | 70 | 27 | 18 | 11 | 5.75 | 0.301 |
| gain 1.25 | 168 | 27 | 14 | 92 | 34 | 30 | 13 | 5.68 | 0.353 |
| gain 1.5 | 168 | 18 | 7 | 115 | 30 | 27 | 9 | 4.43 | 0.247 |
| D40 | 120 | 10 | 3 | 79 | 25 | 51 | 13 | 5.70 | 0.262 |
| D30 | 56 | 7 | 4 | 25 | 18 | 8 | 0 | 5.71 | 0.454 |
| D20 | 72 | 7 | 7 | 42 | 16 | 8 | 0 | 4.80 | 0.349 |
| D10 | 88 | 3 | 2 | 65 | 19 | 8 | 7 | 5.80 | 0.179 |
| D0 | 114 | 32 | 13 | 66 | 13 | 0 | 13 | 4.43 | 0.327 |
| cell g1-ve | 56 | 8 | 5 | 37 | 12 | 9 | 3 | 5.75 | 0.267 |
| cell g1-ve08 | 56 | 5 | 2 | 32 | 15 | 9 | 8 | 8.01 | 0.328 |
| cell g1.25-raw | 56 | 11 | 4 | 33 | 7 | 10 | 4 | 6.04 | 0.328 |
| cell g1.25-ve | 56 | 10 | 7 | 30 | 13 | 10 | 4 | 5.68 | 0.375 |
| cell g1.25-ve08 | 56 | 6 | 3 | 29 | 14 | 10 | 5 | 4.56 | 0.355 |
| cell g1.5-raw | 56 | 7 | 1 | 39 | 6 | 9 | 2 | 4.43 | 0.251 |
| cell g1.5-ve | 56 | 6 | 3 | 38 | 11 | 9 | 2 | 4.02 | 0.252 |
| cell g1.5-ve08 | 56 | 5 | 3 | 38 | 13 | 9 | 5 | 4.42 | 0.236 |
| src self | 432 | 47 | 20 | 271 | 91 | 75 | 30 | 5.68 | 0.285 |
| src log (fallback) | 18 | 12 | 9 | 6 | 0 | 0 | 3 | 4.43 | 0.667 |

- human entry speed (log, strict entry) median 4.89 m/s over these turns; fallback (src=log) distances: 2 of 56
