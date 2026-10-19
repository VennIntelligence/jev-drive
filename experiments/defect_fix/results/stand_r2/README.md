# STAND-R2 (NR1 prereg supplement 23): results, 2026-10-20

Plan: [plans/2026-10-20-nr1-supp23-standr2.md](../../plans/2026-10-20-nr1-supp23-standr2.md) (registered 02:40 CST, before any label, gate
or arm read). Code: `lib/nr1_free.py` (`launch_label`, `launch_excl`, `stand_load`), `lib/nr1_lat.py`, `rm_train --on-stand-launch`,
`scripts/stand_r2.{py,sh}`. STAND-R (supplement 20, [../stand_r/](../stand_r/README.md)) with every standing and rolling row dropped
when the logged human launches (> 0.5 m or >= 1 m/s) within 4 + 2 s, judged on the whole log rather than the 5.5 s window.

**Verdict: both arms FAIL stage 1 on the launch guard again, so stage 2 (the PAI half) was not run.** Dropping the pre-launch rows gives
back only part of what STAND-R cost. At 1x, wait-set b launches go from 33 / 48 (STAND-R R1) to 35 / 48 on other_lead and from 20 / 35 to
22 / 35 on red_queue. The control has 39 / 48 and 23 / 35, and the line is 0.95 x control. The red-line fix is unchanged: easy red-line
zeros go 30 -> 15, the same as R1. So the launch leak does not come mainly from rows where the human launches right after the label window.
Those rows are only 8 % of the standing rows and 10 % of the rolling rows.

## Pre-checks (all passed)
| check | result | file |
|:--|:--|:--|
| launch labels on on1_JTX (128 798 S rows, all labelled) | supplement 20 rows: standing 19 324 -> 17 837 kept (1 487 dropped, 17 of them unknown); rolling 3 568 -> 3 228 (340 dropped, 12 unknown). Launch time quantiles after t (10 / 25 / 50 %): 5.6 / 8.1 / 13.6 s standing, 5.4 / 7.5 / 13.1 s rolling | [launch.json](launch.json) |
| G1: the human's own 4 s paths as the plan, kept rows | 0 fires on 21 065 rows | [gate.json](gate.json) |
| G2: rolling decisions in B2's easy red-line zero windows (19 window-reads), kept rows | 65 / 65 fire; kept rows in 19 / 19 windows (line 10) | gate.json |
| G3: rolling **fires** on windows B2 scores 1 on (32 396 decisions, 4 060 windows) | 143 fires = 0.44 % (line 1 %), in 63 windows = 1.6 % (line 5 %); supplement 20's rule on the same full-log tracks: 224. Standing fires 3 951 (old rule 4 519), no line | gate.json |
| G3w: the control's wait-set v2 b reads, other_lead / red_queue, decisions within 3 s before the human's launch | 0 / 418 fire (old rule 0 / 418: no row, his 4 s arc already moves); a variant: rows 524 -> 0, fires 379 -> 0 | gate.json |
| clock check: the 5.5 s window track's displacement against the raw log's | max 4e-6 m | gate.json |
| switch-off identity (B2 pilot command, 60 steps eager, 4ad8034c vs d7aeb311) | bit-identical | [identity.json](identity.json) |
| smoke (L2 command, 60 steps) | finite; `on/lon_standr_pos` max 0.014; labels show the dropped rows | [smoke.json](smoke.json) |
| DRY command diff | only the tag, (L2) `--on-stand 0.0136`, `--on-stand-roll 3.0 --on-stand-launch 2.0` | [cmd_diff.txt](cmd_diff.txt) |

As the plan said in advance, G3w could not catch the leak. Within 3 s of the launch, the human's own 4 s arc already moves, so neither rule
has a row there. The rule does change one thing on the wait set. In the a variant (the human waits and leaves 0.5 s after the window), the
old rule had 524 rows and 379 fires on the control's plans, and the new rule has none. The a-variant wait rates did recover at 1x
(other_lead 0.889 -> 0.917).

## Stage 1 (gate table: [stage1.md](stage1.md), [stage1.json](stage1.json); stand.py tables [s17table_L1.md](s17table_L1.md), [s17table_L2.md](s17table_L2.md))
716-window subset, the same as STAND-R's. Columns: control / STAND-R2 arm / STAND-R arm at the same dose (decision 399).

| read | control | L1 (1x) | R1 (1x) | L2 (2x) | R2 (2x) | line |
|:--|--:|--:|--:|--:|--:|:--|
| main 1: easy red-line zeros | 30 | **15** | 15 | **7** | 4 | <= 24 |
| main 2: subset PCS | 1246 | **1285** | 1282 | **1301** | 1312 | >= 1261 |
| held 460 zeros / front collisions | 42 / 3 | 37 / 1 | 38 / 1 | 38 / 2 | 36 / 3 | <= 44 / <= 5 |
| easy misses outside the red line | 23 | 21 | 21 | 22 | 22 | <= 25 |
| subset slow windows / k = 9 speed ratio | 36 / 0.907 | 37 / 0.892 | 37 / 0.888 | **40** / 0.878 | 39 / 0.873 | <= 39 / >= 0.877 |
| wait a other_lead / red_queue | 0.917 / 0.923 | 0.917 / 0.923 | 0.889 / 0.923 | 0.917 / 0.949 | 0.889 / 0.949 | >= control - 0.03 |
| wait b launched other_lead (of 48) | 39 | **35** | 33 | **29** | 28 | >= 37 |
| wait b launched red_queue (of 35) | 23 | 22 | 20 | **15** | 15 | >= 22 |
| wait b median launch delay red_queue (s) | 0.42 | **0.57** | 0.47 | **0.67** | 0.72 | <= 0.57 |
| held 460 mean | 0.851 | 0.842 | 0.840 | 0.810 | 0.808 | |
| verdict | | FAIL (other_lead launches; red_queue delay a tie) | FAIL | FAIL (launches, delay, slow) | FAIL | |

L1 misses the other_lead line by 2 launches (35 against 37 of 48). The red_queue delay ties the line exactly: 0.567 against
0.417 + 0.15, and it fails only on floating-point rounding. That does not change the verdict, because other_lead fails on its own. L1 recovers 2 launches in each class over R1, which is within the noise of n = 48 / 35. L2 behaves like R2.

## Reading
- The leak is not mainly the "human launches within 2 s after the label" rows. Dropping them, about 8-10 % of the term's rows, leaves the
  launch cost about where it was. The rest plausibly comes from the term itself: "stay inside your braking reach while the logged human
  stands" is taught on states the model cannot tell apart from the seconds before a launch, whenever the launch cue (light, lead moving)
  is not readable at the decision. This is the same input limit as decisions 380 / 386. Not tested.
- Dose sets the trade-off. 1x halves the easy red-line zeros at about -4 / -1 launches; 2x removes about 3/4 of them at -10 / -8 launches.
- Next options (not tested, not registered): keep only rows whose human stands >= 10 s after t (t_L quantiles leave about half of them);
  restrict the term to the rolling rows and leave the standing dose at B2's; or accept a serve-side stop3 for this class.

## Cost
About 5.7 job-hours, or about 1.5 card-hours: two trainings of about 3 min each, the arena subset and held 460 reads at four per card,
the wait-set chunks, and CPU jobs of a few minutes. The chain ran 02:40-03:16 CST on an empty pool. No PAI half was read.
