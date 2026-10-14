# P4 self-branch read (5 rollouts, 1 scenes)

`done` = turn completed (P2's rule). Otherwise the zero type (corr / offr / coll / oth) or rear (rear-ended before the target) / end.
Cells g<gain>-<profile>: raw = the plan's own speed, ve = profile to v_e, ve08 = to 0.8 v_e. `id` = the policy alone (phase A).
(log) = unreached distance, handover from the log (src=log). `=D<n>` = same handover decision as D<n> (run once).

| scene | side | id | D | src | h | g1-raw | g1-ve | g1-ve08 | g1.25-raw | g1.25-ve | g1.25-ve08 | g1.5-raw | g1.5-ve | g1.5-ve08 |
|:--|:--|:--|--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 302c5c99 | L | corr | 40 | self | 17 | corr | - | - | corr | - | - | corr | - | - |
| 302c5c99 | L | corr | 30 | self =D40 | 17 | corr | - | - | corr | - | - | corr | - | - |
| 302c5c99 | L | corr | 20 | self | 35 | corr | - | - | - | - | - | - | - | - |
| 302c5c99 | L | corr | 10 | self | 55 | corr | - | - | - | - | - | - | - | - |
| 302c5c99 | L | corr | 0 | self | 75 | corr | - | - | corr | - | - | corr | - | - |

## Coverage

- turns: 1; identity (the policy alone) completes 0 / 1
- turns with >= 1 completing branch cell at some distance: 0 / 1; at >= 2 distinct handovers: 0 / 1
- determinism: self branches whose plans before h differ from phase A's identity at the same k: 0 of 4
- **line: turns with completing cells at >= 2 distinct handovers with realized entry speed <= v_e + 1 m/s: 0 / 1 = 0 % -> no pass / fail (ceiling read)**; self handovers only: 0 / 1
- turns with >= 1 completing cell including the identity: 0 / 1
- row yield (completing cells that pass the full shape gate: class p95 kappa / dkappa, a_lat <= human p95, entry speed <= v_e + 1): turns with >= 1: 0 / 1; at >= 2 distinct handovers: 0 / 1
- per distance (turns with >= 1 completing cell): D40 0, D30 0, D20 0, D10 0, D0 0

| cells | rollouts | done | done + gate | zeros | rear-ended | crawl (approach) | crawl inside | median entry speed (done) | mean score |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| profile raw | 4 | 0 | 0 | 4 | 0 | 0 | 0 | nan | 0.000 |
| gain 1.25 | 2 | 0 | 0 | 2 | 0 | 0 | 0 | nan | 0.000 |
| gain 1.5 | 2 | 0 | 0 | 2 | 0 | 0 | 0 | nan | 0.000 |
| D40 | 2 | 0 | 0 | 2 | 0 | 0 | 0 | nan | 0.000 |
| D0 | 2 | 0 | 0 | 2 | 0 | 0 | 0 | nan | 0.000 |
| cell g1.25-raw | 2 | 0 | 0 | 2 | 0 | 0 | 0 | nan | 0.000 |
| cell g1.5-raw | 2 | 0 | 0 | 2 | 0 | 0 | 0 | nan | 0.000 |
| src self | 4 | 0 | 0 | 4 | 0 | 0 | 0 | nan | 0.000 |

- human entry speed (log, strict entry) median 3.73 m/s over these turns; fallback (src=log) distances: 0 of 4
