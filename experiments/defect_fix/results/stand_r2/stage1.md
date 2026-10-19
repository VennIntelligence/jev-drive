# STAND-R2 pilot stage 1 (NR1 prereg supplement 23; native reads; control = the stored B2 pilot STANDFMXH1K-L18-s0)

Arena subset (chunks 0 / 3 / 6 / 9 / 12 / 15) on the frozen board-mix items every read has: 716 windows. STAND-R R1 / R2 (decision 399) are shown on the same subset for comparison only.

## L1 `STANDRLH1K-L18-s0`: FAIL

| read | STANDFMXH1K-L18-s0 | STANDRLH1K-L18-s0 | line | STAND-R R1 / R2 |
|:--|--:|--:|:--|--:|
| main 1: easy red-line zeros | 30 | 15 | <= control - 6: ok | 15 / 4 |
| main 2: subset PCS (frozen board-mix) | 1246 | 1285 | >= control + 15: ok | 1282 / 1312 |
| guard 1: held 460 zeros | 42 | 37 | <= control + 2: ok | 38 / 36 |
| guard 2: held 460 front collisions | 3 | 1 | <= control + 2: ok | 1 / 3 |
| guard 2: subset AF-front windows | 1 | 1 | <= control + 2: ok | 0 / 0 |
| guard 3: easy misses outside the red line | 23 | 21 | <= control + 2: ok | 21 / 22 |
| guard 4: easy partial: free-road ratchet / slow | 7 | 7 | <= control + 2: ok | 7 / 7 |
| guard 4: easy partial: launch late / not launching | 2 | 2 | <= control + 2: ok | 2 / 2 |
| guard 4: easy partial: near-miss 0.98-0.999 | 3 | 3 | <= control + 2: ok | 3 / 4 |
| guard 4: easy partial: other slow | 3 | 3 | <= control + 2: ok | 3 / 1 |
| guard 4: easy partial: rear-ended | 0 | 1 | <= control + 2: ok | 1 / 1 |
| guard 4: easy partial: stop / lock | 2 | 2 | <= control + 2: ok | 2 / 2 |
| guard 4: easy zero: corridor, moving non-turn | 2 | 2 | <= control + 2: ok | 2 / 2 |
| guard 4: easy zero: offroad, non-turn | 1 | 1 | <= control + 2: ok | 1 / 2 |
| guard 4: easy zero: static, other cue | 3 | 0 | <= control + 2: ok | 0 / 0 |
| guard 5: subset slow partial windows | 36 | 37 | <= control + 3: ok | 37 / 39 |
| guard 5: subset k = 9 speed ratio | 0.9068 | 0.892 | >= control - 0.03: ok | 0.8883 / 0.8725 |
| guard 6: wait a other_lead | 0.917 | 0.917 | >= control - 0.03: ok | 0.889 / 0.889 |
| guard 6: wait a red_queue | 0.923 | 0.923 | >= control - 0.03: ok | 0.923 / 0.949 |
| guard 6: wait b median launch delay other_lead (s) | 0.37 | 0.52 | <= control + 0.15: ok | 0.37 / 0.44 |
| guard 6: wait b median launch delay red_queue (s) | 0.42 | 0.57 | <= control + 0.15: FAIL | 0.47 / 0.72 |
| guard 6: wait b launched other_lead | 0.812 | 0.729 | >= 0.95 x control: FAIL | 0.688 / 0.583 |
| guard 6: wait b launched red_queue | 0.657 | 0.629 | >= 0.95 x control: ok | 0.571 / 0.429 |
| guard 7: held 460 right drift, non-turn k 11-17 (arm - control, m) |  | -0.006 | > -0.05: ok | -0.01 / -0.004 |

- subset mean 0.9127 -> 0.9389, zeros 55 -> 33; easy misses 53 -> 36 (of 525 easy windows); held 460 mean 0.8514 -> 0.8424
- red-line easy zeros fixed 15, new 0

## L2 `STANDRL2H1K-L18-s0`: FAIL

| read | STANDFMXH1K-L18-s0 | STANDRL2H1K-L18-s0 | line | STAND-R R1 / R2 |
|:--|--:|--:|:--|--:|
| main 1: easy red-line zeros | 30 | 7 | <= control - 6: ok | 15 / 4 |
| main 2: subset PCS (frozen board-mix) | 1246 | 1301 | >= control + 15: ok | 1282 / 1312 |
| guard 1: held 460 zeros | 42 | 38 | <= control + 2: ok | 38 / 36 |
| guard 2: held 460 front collisions | 3 | 2 | <= control + 2: ok | 1 / 3 |
| guard 2: subset AF-front windows | 1 | 0 | <= control + 2: ok | 0 / 0 |
| guard 3: easy misses outside the red line | 23 | 22 | <= control + 2: ok | 21 / 22 |
| guard 4: easy partial: free-road ratchet / slow | 7 | 7 | <= control + 2: ok | 7 / 7 |
| guard 4: easy partial: launch late / not launching | 2 | 2 | <= control + 2: ok | 2 / 2 |
| guard 4: easy partial: near-miss 0.98-0.999 | 3 | 2 | <= control + 2: ok | 3 / 4 |
| guard 4: easy partial: other slow | 3 | 3 | <= control + 2: ok | 3 / 1 |
| guard 4: easy partial: rear-ended | 0 | 1 | <= control + 2: ok | 1 / 1 |
| guard 4: easy partial: stop / lock | 2 | 2 | <= control + 2: ok | 2 / 2 |
| guard 4: easy zero: corridor, moving non-turn | 2 | 2 | <= control + 2: ok | 2 / 2 |
| guard 4: easy zero: offroad, non-turn | 1 | 2 | <= control + 2: ok | 1 / 2 |
| guard 4: easy zero: static, other cue | 3 | 0 | <= control + 2: ok | 0 / 0 |
| guard 4: easy zero: turn window, offroad / corridor | 0 | 1 | <= control + 2: ok | 0 / 1 |
| guard 5: subset slow partial windows | 36 | 40 | <= control + 3: FAIL | 37 / 39 |
| guard 5: subset k = 9 speed ratio | 0.9068 | 0.8778 | >= control - 0.03: ok | 0.8883 / 0.8725 |
| guard 6: wait a other_lead | 0.917 | 0.917 | >= control - 0.03: ok | 0.889 / 0.889 |
| guard 6: wait a red_queue | 0.923 | 0.949 | >= control - 0.03: ok | 0.923 / 0.949 |
| guard 6: wait b median launch delay other_lead (s) | 0.37 | 0.47 | <= control + 0.15: ok | 0.37 / 0.44 |
| guard 6: wait b median launch delay red_queue (s) | 0.42 | 0.67 | <= control + 0.15: FAIL | 0.47 / 0.72 |
| guard 6: wait b launched other_lead | 0.812 | 0.604 | >= 0.95 x control: FAIL | 0.688 / 0.583 |
| guard 6: wait b launched red_queue | 0.657 | 0.429 | >= 0.95 x control: FAIL | 0.571 / 0.429 |
| guard 7: held 460 right drift, non-turn k 11-17 (arm - control, m) |  | 0.005 | > -0.05: ok | -0.01 / -0.004 |

- subset mean 0.9127 -> 0.9506, zeros 55 -> 21; easy misses 53 -> 29 (of 525 easy windows); held 460 mean 0.8514 -> 0.8099
- red-line easy zeros fixed 23, new 0
