# VAL160: D1 / R8K-s2 / P1L on PAI val160

Descriptive, one read each. Plan: [../../plans/2026-10-24-val160.md](../../plans/2026-10-24-val160.md) (Chinese). Code: `scripts/v160.sh`, `scripts/v160.py`.
Full tables: [val160.md](val160.md), [val160.csv](val160.csv), [per_scene.csv](per_scene.csv). Scenes: [chunks/](chunks/), dropped 8: [dropped8.txt](dropped8.txt).

Setup: 152 val160 scenes (160 minus 8 drive-linked to on-box train scenes); v = the 72 PAI-FIX scenes (selected on NTV0 failures + guard), w = 80
never-selected. Serve as the 281 reads (ACC + GAP, g10, FAST, CONC 4, Harmonizer off, no JEV_CMD). No stored val160 reads existed for the three; all new.
152 common scenes, no RPC errors. New scorer = scorer_v2 (zero rollouts re-evaluated; 42-58 nonzero rollouts per read lack an ASL and keep the stored value).

| scorer | subset | n | D1 | R8K-s2 | P1L | D1 - R8K-s2 | D1 - P1L |
|:--|:--|--:|--:|--:|--:|:--|:--|
| new | all | 152 | 0.828 | 0.854 | 0.841 | -0.026 [-0.071, +0.019] | -0.012 [-0.067, +0.043] |
| old | all | 152 | 0.750 | 0.769 | 0.755 | -0.019 [-0.063, +0.024] | -0.006 [-0.059, +0.047] |
| new | v (selected) | 72 | 0.754 | 0.799 | 0.780 | -0.045 [-0.116, +0.025] | -0.027 [-0.113, +0.058] |
| new | w (unselected) | 80 | 0.896 | 0.904 | 0.895 | -0.008 [-0.067, +0.051] | +0.001 [-0.068, +0.070] |

Zeros (new / old scorer): D1 21 / 33, R8K-s2 18 / 31, P1L 19 / 32. Old-scorer zero classes (D1 / R8K-s2 / P1L): overshoot 12 / 13 / 13, route-turn 1 / 2 / 4,
offroad 7 / 7 / 5, AF-front 9 / 7 / 8, side 3 / 2 / 1, zero-other 1 / 0 / 1.

Ordering vs test281: D1 - R8K-s2 is -0.005 on 281 (old scorer, 279 scenes) and -0.019 on val160; same sign, both CIs contain 0. D1 - P1L is +0.065 on 281
(old) but -0.006 on val160: the 281 gain over P1L does not show on val160 (w: +0.001). No contrast excludes 0; D1 does not beat R8K-s2 on either exam.

Cost: 21 chunk jobs, 5.2 job-hours (about 1.3 card-hours at 4 per card), 28 min wall; plus CPU tax / rescoring.
Note: attempt 1 of the report step failed (ModuleNotFoundError, a lane-script fault); the reads were fine, report re-run by hand after the fix.
