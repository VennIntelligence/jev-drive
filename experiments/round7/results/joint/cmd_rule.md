# Change 3: command from the route's heading change (rmem.cmd_heading, look 40 m, 20 deg), offline (joint.py cmdcheck)

Rows fed >= 4 m/s with a finite label; label turn = |4 s label heading|; L/R share of the old command (the shipped rule on the latest route message, as stored) and the new rule on the same rows' stored float16 memory. navtrain: one shard (s0of12), its own NAVSIM command as `old` (the target profile).

| rows | rule | n | label turn 0-10 deg | 10-30 | 30-60 | >= 60 | sign agrees (>= 30 deg) |
|:--|:--|--:|--:|--:|--:|--:|--:|
| ph | old | 96676 | 26.9 % | 86.3 % | 99.2 % | 99.9 % | 100.0 % |
| ph | new | 96676 | 5.5 % | 40.0 % | 95.4 % | 99.7 % | 99.1 % |
| ph05 | old | 59385 | 29.6 % | 99.5 % | 99.8 % | 100.0 % | 100.0 % |
| ph05 | new | 59385 | 4.8 % | 65.7 % | 100.0 % | 100.0 % | 99.8 % |
| ph15 | old | 76119 | 26.8 % | 73.8 % | 92.2 % | 98.5 % | 98.8 % |
| ph15 | new | 76119 | 5.6 % | 30.1 % | 79.9 % | 97.6 % | 98.4 % |
| pd | old | 83573 | 25.4 % | 87.0 % | 99.8 % | 98.5 % | 99.9 % |
| pd | new | 83573 | 5.0 % | 40.7 % | 98.3 % | 99.8 % | 99.6 % |
| rb0 | old | 10511 | 31.9 % | 97.5 % | 99.7 % | 99.3 % | 99.9 % |
| rb0 | new | 10511 | 6.8 % | 49.5 % | 94.8 % | 94.8 % | 99.5 % |
| rtb1 | old | 10988 | 30.3 % | 97.7 % | 99.5 % | 99.6 % | 100.0 % |
| rtb1 | new | 10988 | 6.7 % | 50.1 % | 94.2 % | 95.4 % | 99.6 % |
| y | old | 10317 | 32.0 % | 97.7 % | 99.8 % | 99.4 % | 99.5 % |
| y | new | 10317 | 7.3 % | 50.9 % | 94.7 % | 96.1 % | 99.2 % |
| ys | old | 20640 | 32.9 % | 96.8 % | 99.5 % | 98.7 % | 99.6 % |
| ys | new | 20640 | 6.9 % | 45.6 % | 94.6 % | 94.4 % | 99.3 % |
| navtrain s0of12 | old | 4728 | 4.5 % | 60.1 % | 96.0 % | 99.3 % | 100.0 % |
| navtrain s0of12 | new | 4728 | 7.9 % | 57.3 % | 94.1 % | 96.4 % | 99.6 % |

**G2 (offline): PASS** (PAI parts (ph, ph05, ph15, pd), new rule: L/R <= 10 % at 0-10 deg, >= 75 % at 30-60, >= 90 % at >= 60, sign >= 97 %; {"nonturn": true, "turn30": true, "turn60": true, "sign": true}).
