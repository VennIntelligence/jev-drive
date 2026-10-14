# P4 exam ceiling read: the P4 branch family on the hard_exam turn scenes (2026-10-14, read-only)

Prereg [../../../plans/2026-10-14-p4-exam-ceiling-prereg.md](../../../plans/2026-10-14-p4-exam-ceiling-prereg.md) (Amendments 1-3). This read
takes the P4 branch family as it stands (prereg [2026-10-14-p4-self-branch](../../../plans/2026-10-14-p4-self-branch-prereg.md): handovers
D40..D0, 9 cells, log fallback) and runs it on the hard_exam turns. The result is an **upper bound**, to be quoted next to RSGSELF's PASS1
result. It changes no design value, gets no pass / fail mark, and no row is built from it: the root `/data/runs/route_night/p4_exam`
(Tokyo) holds `NO_ROWS`, which `rf_selfpool.py` refuses, and every exam rollout.asl was deleted after its scene's read (0 left).

Files: `table.md` (the scene x handover x cell table, per-D / per-cell / per-profile / per-gain rows, written by `rf_p4.py table --ceiling`),
`read.json` (465 rollouts, every field of `rf_p4.py read`), `summary.json` (coverage per scene and D), `plan.json` (the 15 definable
turns plus the 11 dropped ones, with reasons), `s1.txt` (phase A against r281/rsg), `exam_log.txt` / `run_log.txt` (per-scene and per-launch
log), `code.json` (served code: frozen Tokyo clone `/data/runs/route_night/p4_exam/code` at 05f3c5c5, dirty 0).

## Headline

Each count is given over the 26 exam turn scenes. A dropped scene counts as not covered, because its turn is not definable there. The same
count over the 15 definable turns is in brackets.

| line | count |
|:--|--:|
| identity: turns the policy alone completes | **2 / 26** (2 / 15): 382a7247, d90fe123, both entering at 8.3-9.0 m/s, above the class gate of 5.6 / 6.3 |
| ceiling, any branch cell at any handover (P4's rule: the identity is no branch) | **12 / 26** (12 / 15); self handovers only 10; including the identity 12 |
| coverage definition (completing branch cells at >= 2 distinct handovers, each with realized entry <= class v_e + 1 m/s) | **2 / 26** (2 / 15): d90fe123 (D40 / 30 / 20 / 10), 09528162 (D40 / D0); self handovers only 2 |
| shape gate (decision 293 as `rf_p4.py read` computes it), turns with gate-passing completions at >= 1 handover | **7 / 26** (7 / 15) |
| shape gate at >= 2 distinct handovers | **1 / 26** (1 / 15): d90fe123 |
| turns with completing cells at >= 2 distinct handovers (any speed) | 5 / 15 |
| phase A bit-identical to r281/rsg (served plans, track, score) | **15 / 15** |
| self branches' plans before h identical to phase A | 432 / 432 |

What the table shows: on most exam turns the branch family completes the turn from at least one state (12 / 15). That is almost always a
single handover, and most often D0, where the policy reaches the entry on its own and the push only has to bend it through (D0: 11 turns;
D40 4, D30 2, D20 3, D10 2). Only two turns are covered at two or more handovers at human-like entry speed, and only d90fe123 passes the
full shape gate at two or more. That turn is also one the policy completes alone, although too fast. For comparison, the coverage
definition reached 10 / 27 on the training scenes. Here it is 2 / 26 (2 / 15 definable).

No branch cell completes 22fa5c92 (collision at every handover), 94c1d722 (offroad everywhere, approach crawl in the profile cells) or
9f393c8a (one distinct handover, at k 17).

## Per scene

`h` = handover decision, `src` = self (the policy's own state) or log (fallback, unreached distance). Cells done at that handover: all /
entry speed <= v_e + 1 / passing the shape gate. Distances that share a handover decision are counted once (the larger D).

| scene | side | class | v_e | identity | D: done / slow / gate |
|:--|:--|:--|--:|:--|:--|
| 382a7247 | L | straight_exit | 4.6 | done (8.95 m/s) | D40 1/0/0, D20 2/2/2, D10 0, D0 2/0/0 |
| a1b166cc | L | other_no_lane | 4.6 | corridor | D40 1/1/0, D0 0 |
| abd45a30 | L | straight_exit | 4.6 | rear-ended | D40-D10 0; D0 (log) 3/0/0 |
| ce161092 | L | straight_road | 5.3 | offroad | D40 0, D30 1/0/0, D20 2/2/2, D10 0, D0 2/0/0 |
| d90fe123 | L | straight_road | 5.3 | done (8.34 m/s) | D40 5/3/3, D30 6/4/4, D20 3/3/3, D10 2/2/2, D0 2/0/0 |
| e37aa159 | L | curve_single | 3.9 | corridor | D40-D20 0, D10 1/0/0, D0 1/0/0 |
| fd01e9ab | L | no_straight_exit | 4.6 | corridor | D40 0, D0 1/1/1 |
| 22fa5c92 | L | straight_exit | 4.6 | collision | D40 0, D0 0 |
| 09528162 | R | straight_exit | 4.6 | rear-ended | D40 3/3/0, D10 0, D0 3/3/0 |
| 2f227f2c | R | straight_road | 5.3 | corridor | D40-D10 0; D0 (log) 9/9/9 |
| 3ce13eed | R | curve_single | 3.9 | corridor | D40 0, D10 0, D0 5/5/1 |
| e0f6ebe1 | R | straight_exit | 4.6 | end (no completion) | D40-D10 0, D0 2/0/0 |
| e848c843 | R | straight_exit | 4.6 | end (no completion) | D40-D10 0, D0 2/2/2 |
| 94c1d722 | R | no_straight_exit | 4.6 | offroad | all 0 |
| 9f393c8a | R | curve_single | 3.9 | collision | D40 (only handover) 0 |

## Cells, fallback, crawl

Branch rollouts: 450, of which 59 complete. Their median realized entry speed is 4.80 m/s; the human median over these 15 turns is 4.89 m/s.

| cells | rollouts | done | done + gate | rear-ended | crawl approach | crawl inside |
|:--|--:|--:|--:|--:|--:|--:|
| g1-ve / g1-ve08 | 56 / 56 | 8 / 5 | 5 / 2 | 12 / 15 | 9 / 9 | 3 / 8 |
| g1.25-raw / -ve / -ve08 | 56 / 56 / 56 | 11 / 10 / 6 | 4 / 7 / 3 | 7 / 13 / 14 | 10 / 10 / 10 | 4 / 4 / 5 |
| g1.5-raw / -ve / -ve08 | 56 / 56 / 56 | 7 / 6 / 5 | 1 / 3 / 3 | 6 / 11 / 13 | 9 / 9 / 9 | 2 / 2 / 5 |
| src self | 432 | 47 | 20 | 91 | 75 | 30 |
| src log (fallback) | 18 | 12 | 9 | 0 | 0 | 3 |

- **Fallback**: 2 of 56 distinct handovers were unreached, both at D0 (abd45a30 and 2f227f2c: the policy left the route or zeroed before
  reaching the entry), and they gave 18 log-fallback rollouts. Those 2 turns are covered only through the fallback (12 completions).
- **Crawl** (min speed between h and the entry < 0.5 v_e): 75 of 432 self branches, most of them at D40 (51); crawl inside the turn: 33 of 450.
- Rear-ended (the log's following car does not react; not disqualifying): 91 of 450.

## Run and timing

Tokyo, two RTX 3090, one rollout at a time per launch; 465 rollouts read (15 phase A + 450 branches) between 14:40 and 19:30 JST.

| stage | scenes | layout | wall | notes |
|:--|--:|:--|--:|:--|
| 1 | 1 | card 0, phase B as 3 launches | 31 min | 41 rollouts; card 0 peak 22.6 GB |
| 5 | 5 | 3 workers per card | 15:12-16:57 | 4 DONE; d90fe123's renderer OOMed at start-up, and every later rollout start in that launch then waited 600 s in `get_available_cameras`; the launch was stopped and set aside (9 rollouts not read) |
| rest (3 per card) | - | 3 workers per card | 2 min | second renderer OOM (CUDA OOM in 2f227f2c's phase A with three renderers warming on card 0); stopped, nothing read |
| rest2 | 10 | 2 workers per card | 17:03-19:30 | all DONE, 0 OOM |

Per rollout: 53-77 s of simulation plus start-up and read. Scene jobs took 1 076 s (9 rollouts) to 5 759 s (41 rollouts, 1 launch). **Three
launches per 3090 is not safe**: it ran OOM twice, both times while several renderers were warming up together. Two per card ran clean.
Disk after pruning: 17 GB under the root.

## Not claimed

15 definable turns of 26; one seed, one checkpoint, a deterministic simulator. A completing cell is one draw of a chaotic closed loop. This
is an upper bound on what this branch family reaches on the exam turns, not a result for any trained policy.
