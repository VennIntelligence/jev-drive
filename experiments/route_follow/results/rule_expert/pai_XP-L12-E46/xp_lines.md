# Lines (pai)

- E2: completed 29 / 36 (line 29 / 36)
- E3 (control, 75 scenes scored by every label): xp-l12-e46 0.8116; rsg 0.7779; xp-l12-e46 - rsg +0.0337 (line >= -0.05)
- xp-l12-e46: scored 281, mean 0.7646, zeros 36, collision at fault 7, offroad 8
- rsg: scored 281, mean 0.5720, zeros 70, collision at fault 9, offroad 14
- xp-l12-e46 zeros 36: shared with rsg 21, new 15, removed 49; new zeros by reason: collision_at_fault 3, left_corridor_laterally 8, offroad 4
- new zeros whose last expert decision is at or past the human log's end station - 2 m (the evaluator's corridor / GT reference is a straight extension there; the expert does not stop where the human stopped): 9
