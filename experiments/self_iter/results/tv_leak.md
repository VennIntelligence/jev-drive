# TV-AUDIT leak (1310 trainval log segments, 171 drives, 723019 tokens)

| evaluation set | segments | drives | segment overlap | drive overlap |
|:--|--:|--:|--:|--:|
| navtest (navsim test logs) | 147 | 46 | 0 | 0 |
| navhard_two_stage | 76 | 34 | 0 | 0 |
| held460 | 39 | 16 | 0 | 0 |
| public1491 | 136 | 44 | 0 | 0 |
| arena (all windows) | 136 | 44 | 0 | 0 |
| arena held windows | 39 | 16 | 0 | 0 |
| waitset v2 | 91 | 40 | 0 | 0 |
| waitcap_test | 130 | 43 | 0 | 0 |

token overlap: {'navtest': 0, 'navhard_two_stage': 0, 'navhard original scene tokens': 0}; PAI 281 (281 scenes) is another dataset (no nuPlan log); navtrain dev logs inside trainval 32 (excluded from the splits)
clean: True; registered navsim/tv-clean-train-logs@v1:816f0ad017ee (1154 logs), navsim/tv-clean-hold-logs@v1:1b0767f6cabc (124 logs)
