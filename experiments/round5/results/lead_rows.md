# R5-N: the LEAD rule on the PAI human rows (offline, before training)

lead = an object (box length >= 1.5 m) whose centre lies within half the ego width + half its width + 0.3 m of the human's 4 s future path (extended along the last heading), heading within 60 deg of the path, gap (path station - L/2 - front bumper) in [-1, max(40 m, 4 s x v0)]; plain rows: their own t0 boxes, label and fed speed; stretched rows: their base plain row's flag.

R3MV draws only PAI rows fed >= 2 m/s; R5-N sets the lead rows among them to 0 and renormalises each part to its R3MV share (ph 0.25, ph05 0.125, ph15 0.125 of the 64 pool rows).

| part | R3MV drawable | lead (removed) | share | draw-weighted | gap <= 15 m | gap median (m) | v0 median removed / kept (m/s) | scenes kept / drawable |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| ph | 90631 | 36073 | 39.8% | 39.8% | 12309 | 20.1 | 8.6 / 10.0 | 524 / 705 |
| ph05 | 74250 | 28602 | 38.5% | 38.5% | 7706 | 22.2 | 5.0 / 5.3 | 497 / 680 |
| ph15 | 69095 | 28478 | 41.2% | 41.2% | 10795 | 19.0 | 11.7 / 14.5 | 490 / 707 |

Removed / drawable by label class (r3.label_class; launch and stand are 0 under MV's >= 2 m/s cut):

| part | launch | stand | stop-approach | turn | cruise |
|:--|--:|--:|--:|--:|--:|
| ph | 0 / 0 | 0 / 0 | 1955 / 4564 | 5161 / 18092 | 28957 / 67975 |
| ph05 | 0 / 0 | 0 / 0 | 93 / 376 | 2811 / 9369 | 25698 / 64505 |
| ph15 | 0 / 0 | 0 / 0 | 3069 / 6749 | 5607 / 18367 | 19802 / 43979 |

PAI human rows a batch (64 pool rows x draw mass) by class:

| draw | PAI rows | launch | stand | stop-approach | turn | cruise | unique PAI rows |
|:--|--:|--:|--:|--:|--:|--:|--:|
| R3MV | 32.00 | 0.00 | 0.00 | 1.63 | 6.33 | 24.04 | 233976 |
| R5N | 32.00 | 0.00 | 0.00 | 1.54 | 7.45 | 23.01 | 140823 |

Lead share over all plain rows (any speed): 42.4%. Files: w_R5N.json, w_R5NI.json (R5NI = R3MV bit for bit).
