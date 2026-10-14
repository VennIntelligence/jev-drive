# P4 self-branch read (4 rollouts, 1 scenes)

`done` = turn completed (P2's rule). Otherwise the zero type (corr / offr / coll / oth) or rear (rear-ended before the target) / end.
Cells g<gain>-<profile>: raw = the plan's own speed, ve = profile to v_e, ve08 = to 0.8 v_e. `id` = the policy alone (phase A).
(log) = unreached distance, handover from the log (src=log). `=D<n>` = same handover decision as D<n> (run once).

| scene | side | id | D | src | h | g1-raw | g1-ve | g1-ve08 | g1.25-raw | g1.25-ve | g1.25-ve08 | g1.5-raw | g1.5-ve | g1.5-ve08 |
|:--|:--|:--|--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| f9e361aa | R | corr | 40 | self | 23 | corr | - | - | - | - | - | - | - | - |
| f9e361aa | R | corr | 30 | self | 34 | corr | - | - | - | - | - | - | - | - |
| f9e361aa | R | corr | 20 | self | 44 | corr | corr | - | corr | - | **done** | - | - | - |
| f9e361aa | R | corr | 10 | self | 54 | corr | - | - | - | - | - | - | - | - |
| f9e361aa | R | corr | 0 | self | 63 | corr | - | - | - | - | - | - | - | - |

## Coverage

- turns: 1; identity (the policy alone) completes 0 / 1
- turns with >= 1 completing branch cell at some distance: 1 / 1; at >= 2 distinct handovers: 0 / 1
- determinism: self branches whose plans before h differ from phase A's identity at the same k: 0 of 3
- **line: turns with completing cells at >= 2 distinct handovers with realized entry speed <= v_e + 1 m/s: 0 / 1 = 0 % -> FAIL** (pass >= 60 %, fail <= 35 %); self handovers only: 0 / 1
- row yield (completing cells that pass the full shape gate: class p95 kappa / dkappa, a_lat <= human p95, entry speed <= v_e + 1): turns with >= 1: 1 / 1; at >= 2 distinct handovers: 0 / 1
- per distance (turns with >= 1 completing cell): D40 0, D30 0, D20 1, D10 0, D0 0

| cells | rollouts | done | done + gate | zeros | rear-ended | crawl (approach) | crawl inside | median entry speed (done) | mean score |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| profile raw | 1 | 0 | 0 | 1 | 0 | 0 | 0 | nan | 0.000 |
| profile ve | 1 | 0 | 0 | 1 | 0 | 0 | 0 | nan | 0.000 |
| profile ve08 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 3.90 | 1.000 |
| gain 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | nan | 0.000 |
| gain 1.25 | 2 | 1 | 1 | 1 | 0 | 0 | 0 | 3.90 | 0.500 |
| D20 | 3 | 1 | 1 | 2 | 0 | 0 | 0 | 3.90 | 0.333 |
| src self | 3 | 1 | 1 | 2 | 0 | 0 | 0 | 3.90 | 0.333 |

- human entry speed (log, strict entry) median 5.25 m/s over these turns; fallback (src=log) distances: 0 of 5
