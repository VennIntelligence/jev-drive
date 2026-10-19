# STAND-R pilot stage 1 (NR1 prereg supplement 20; native reads; control = the stored B2 pilot STANDFMXH1K-L18-s0)

Arena subset (chunks 0 / 3 / 6 / 9 / 12 / 15) on the frozen board-mix items every read has: 716 windows.

## R1 `STANDRH1K-L18-s0`: FAIL

| read | STANDFMXH1K-L18-s0 | STANDRH1K-L18-s0 | line |
|:--|--:|--:|:--|
| main 1: easy red-line zeros | 30 | 15 | <= control - 6: ok |
| main 2: subset PCS (frozen board-mix) | 1246 | 1282 | >= control + 15: ok |
| guard 1: held 460 zeros | 42 | 38 | <= control + 2: ok |
| guard 2: held 460 front collisions | 3 | 1 | <= control + 2: ok |
| guard 2: subset AF-front windows | 1 | 0 | <= control + 2: ok |
| guard 3: easy misses outside the red line | 23 | 21 | <= control + 2: ok |
| guard 4: easy partial: free-road ratchet / slow | 7 | 7 | <= control + 2: ok |
| guard 4: easy partial: launch late / not launching | 2 | 2 | <= control + 2: ok |
| guard 4: easy partial: near-miss 0.98-0.999 | 3 | 3 | <= control + 2: ok |
| guard 4: easy partial: other slow | 3 | 3 | <= control + 2: ok |
| guard 4: easy partial: rear-ended | 0 | 1 | <= control + 2: ok |
| guard 4: easy partial: stop / lock | 2 | 2 | <= control + 2: ok |
| guard 4: easy zero: corridor, moving non-turn | 2 | 2 | <= control + 2: ok |
| guard 4: easy zero: offroad, non-turn | 1 | 1 | <= control + 2: ok |
| guard 4: easy zero: static, other cue | 3 | 0 | <= control + 2: ok |
| guard 5: subset slow partial windows | 36 | 37 | <= control + 3: ok |
| guard 5: subset k = 9 speed ratio | 0.9068 | 0.8883 | >= control - 0.03: ok |
| guard 6: wait a other_lead | 0.917 | 0.889 | >= control - 0.03: ok |
| guard 6: wait a red_queue | 0.923 | 0.923 | >= control - 0.03: ok |
| guard 6: wait b median launch delay other_lead (s) | 0.37 | 0.37 | <= control + 0.15: ok |
| guard 6: wait b median launch delay red_queue (s) | 0.42 | 0.47 | <= control + 0.15: ok |
| guard 6: wait b launched other_lead | 0.812 | 0.688 | >= 0.95 x control: FAIL |
| guard 6: wait b launched red_queue | 0.657 | 0.571 | >= 0.95 x control: FAIL |
| guard 7: held 460 right drift, non-turn k 11-17 (arm - control, m) |  | -0.01 | > -0.05: ok |

- subset mean 0.9127 -> 0.9390, zeros 55 -> 32; easy misses 53 -> 36 (of 525 easy windows); held 460 mean 0.8514 -> 0.8396
- red-line easy zeros fixed 15, new 0

## R2 `STANDR2H1K-L18-s0`: FAIL

| read | STANDFMXH1K-L18-s0 | STANDR2H1K-L18-s0 | line |
|:--|--:|--:|:--|
| main 1: easy red-line zeros | 30 | 4 | <= control - 6: ok |
| main 2: subset PCS (frozen board-mix) | 1246 | 1312 | >= control + 15: ok |
| guard 1: held 460 zeros | 42 | 36 | <= control + 2: ok |
| guard 2: held 460 front collisions | 3 | 3 | <= control + 2: ok |
| guard 2: subset AF-front windows | 1 | 0 | <= control + 2: ok |
| guard 3: easy misses outside the red line | 23 | 22 | <= control + 2: ok |
| guard 4: easy partial: free-road ratchet / slow | 7 | 7 | <= control + 2: ok |
| guard 4: easy partial: launch late / not launching | 2 | 2 | <= control + 2: ok |
| guard 4: easy partial: near-miss 0.98-0.999 | 3 | 4 | <= control + 2: ok |
| guard 4: easy partial: other slow | 3 | 1 | <= control + 2: ok |
| guard 4: easy partial: rear-ended | 0 | 1 | <= control + 2: ok |
| guard 4: easy partial: stop / lock | 2 | 2 | <= control + 2: ok |
| guard 4: easy zero: corridor, moving non-turn | 2 | 2 | <= control + 2: ok |
| guard 4: easy zero: offroad, non-turn | 1 | 2 | <= control + 2: ok |
| guard 4: easy zero: static, other cue | 3 | 0 | <= control + 2: ok |
| guard 4: easy zero: turn window, offroad / corridor | 0 | 1 | <= control + 2: ok |
| guard 5: subset slow partial windows | 36 | 39 | <= control + 3: ok |
| guard 5: subset k = 9 speed ratio | 0.9068 | 0.8725 | >= control - 0.03: FAIL |
| guard 6: wait a other_lead | 0.917 | 0.889 | >= control - 0.03: ok |
| guard 6: wait a red_queue | 0.923 | 0.949 | >= control - 0.03: ok |
| guard 6: wait b median launch delay other_lead (s) | 0.37 | 0.44 | <= control + 0.15: ok |
| guard 6: wait b median launch delay red_queue (s) | 0.42 | 0.72 | <= control + 0.15: FAIL |
| guard 6: wait b launched other_lead | 0.812 | 0.583 | >= 0.95 x control: FAIL |
| guard 6: wait b launched red_queue | 0.657 | 0.429 | >= 0.95 x control: FAIL |
| guard 7: held 460 right drift, non-turn k 11-17 (arm - control, m) |  | -0.004 | > -0.05: ok |

- subset mean 0.9127 -> 0.9525, zeros 55 -> 19; easy misses 53 -> 26 (of 525 easy windows); held 460 mean 0.8514 -> 0.8077
- red-line easy zeros fixed 26, new 0
