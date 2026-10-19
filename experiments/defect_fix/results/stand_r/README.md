# STAND-R (NR1 prereg supplement 20): results, 2026-10-20

Plan: [plans/2026-10-20-nr1-supp20-standr.md](../../plans/2026-10-20-nr1-supp20-standr.md) (registered 00:45 CST, before any label, gate or arm
read). Code: `lib/nr1_free.py` (`rolling`, `roll_cap`, `stand_term_split`), `lib/nr1_lat.py`, `rm_train --on-stand-roll`,
`scripts/stand_r.{py,sh}`. Scope set by the main session: offline gate plus two 1 000-step pilots plus the control's PAI half read, no
full-length arm. One diagnostic read was added: B2 s1 + stop3.

**Verdict: both arms FAIL stage 1 on the registered launch guard, so stage 2 (the PAI half) was not run.** The term does what it was
designed to do: red-line easy zeros fall 30 -> 15 (R1) and 30 -> 4 (R2), and subset PCS rises +36 / +66. The cost is launches. On the
wait set v2 b variant, the share of launches drops with dose: other_lead 0.812 -> 0.688 / 0.583 and red_queue 0.657 -> 0.571 / 0.429.
That is 39 -> 33 / 28 of 48 and 23 -> 20 / 15 of 35. Held 460 mean falls too: -0.012 [-0.022, +0.002] for R1 and -0.044 [-0.065, -0.024]
for R2, while held zeros improve (42 -> 38 / 36).

## Pre-checks (all passed)
| check | result | file |
|:--|:--|:--|
| gap labels on on1_JTX (128 798 S rows, all finite) | 0.0 m max difference to stoppt's h4c on on1_JT's 87 012 comparable rows | [gap.json](gap.json) |
| G1: the human's own 4 s paths used as the plan | 0 fires on 22 892 rows (19 324 standing, 3 568 rolling) | [gate.json](gate.json) |
| G2: rolling decisions in B2's easy red-line zero windows (s0 10 + s1 9 windows) | 76 / 76 fire (standing decisions there 52 / 52) | gate.json |
| G3: rolling decisions in windows B2 scores 1 on | 0.94 % (303 / 32 396 decisions, both seeds; line 1 %); 224 of them fire (static 122, braking 75, other 27) | gate.json |
| switch-off identity (B2 pilot command, 60 steps eager, dfba9d04 vs new code) | bit-identical (scalars, weights, dev ADE, pool hash) | [identity.json](identity.json) |
| smoke (R2 command, 60 steps) | finite; `on/lon_standr_pos` > 0 at each logged step (max 0.014) | [smoke.json](smoke.json) |
| DRY command diff against the stored B2 pilot command | only the tag, (R2) `--on-stand 0.0136`, `--on-stand-roll 3.0` | [cmd_diff.txt](cmd_diff.txt) |

The draft said most rolling rows sit behind the human's standing point. The labels show the opposite. Gap quantiles over the 3 568 rolling
rows are -2.7 / -1.7 / -0.8 / -0.1 / +3.4 m (10 / 25 / 50 / 75 / 90 %). In 2 705 rows (76 %) the ego is already past the human's standing
station, and only 541 rows (15 %) get a cap above b_8 from the gap. The draft's 74 % came from STOPPT's `rest` label, which is a different
quantity. So the term mostly says "stop within your braking reach once you are level with or past where the human stands". That is the
rule that also bites in the launch phase.

In hindsight G3 was the warning. On windows B2 already scores 1 on, the term fired on 224 of 303 rolling decisions, and the unchanged
standing rows fired on 4 519 decisions. These are launches at the end of a stand that the model gets right and the human made later. The
gate counted only the domain share (0.94 %, under the 1 % line), not the fire count on good driving.

## Stage 1 (gate table: [stage1.md](stage1.md), [stage1.json](stage1.json); stand.py tables [s17table_R1.md](s17table_R1.md), [s17table_R2.md](s17table_R2.md))
| read | control STANDFMXH1K | R1 (W_S x 1) | R2 (W_S x 2) | line |
|:--|--:|--:|--:|:--|
| main 1: easy red-line zeros (716-window subset) | 30 | **15** | **4** | <= 24 |
| main 2: subset PCS | 1246 | **1282** | **1312** | >= 1261 |
| held 460 zeros / front collisions | 42 / 3 | 38 / 1 | 36 / 3 | <= 44 / <= 5 |
| easy misses outside the red line (each class <= +2) | 23 | 21 | 22 | <= 25 |
| subset slow windows / k = 9 speed ratio | 36 / 0.907 | 37 / 0.888 | 39 / **0.873** | <= 39 / >= 0.877 |
| wait a other_lead / red_queue | 0.917 / 0.923 | 0.889 / 0.923 | 0.889 / 0.949 | >= control - 0.03 |
| wait b launched other_lead / red_queue | 0.812 / 0.657 | **0.688 / 0.571** | **0.583 / 0.429** | >= 0.95 x control |
| wait b median launch delay red_queue (s) | 0.42 | 0.47 | **0.72** | <= control + 0.15 |
| held 460 right drift (k 11-17) | | -0.010 | -0.004 | > -0.05 |
| verdict | | FAIL (launched) | FAIL (launched, delay, k = 9) | |

Description only: the subset mean goes 0.9127 -> 0.9390 / 0.9525 and zeros 55 -> 32 / 19. Easy misses go 53 -> 36 / 26. No new red-line
zeros appeared.

## B2 s1 + stop3 (diagnostic read, 18 arena held chunks; [easy_H_s1_stop3.json](easy_H_s1_stop3.json), windows [arena_windows_H_s1_stop3.csv.gz](arena_windows_H_s1_stop3.csv.gz))
| | B2 s1 native | B2 s1 + stop3 |
|:--|--:|--:|
| frozen-pool PCS (board-mix) / mean / zeros | 1398 / 0.9365 / 96 | 1409 / 0.9449 / 58 |
| easy misses (red-line zeros) | 47 (9) | 39 (0) |
| easy launch late / stop-lock partials | 5 / 2 | 9 / 7 |
| easy turn / moving corridor / straight offroad zeros | 8 / 7 / 5 | 7 / 2 / 4 |

stop3 still holds all 9 red-line easy windows on B2. It hands back about half of them as launch and stop-lock partials, though, so PCS rises
only +11.

## Cost
About 13 job-hours, or about 4.6 card-hours: arena / held / wait-set reads at four per card, two trainings, the control's PAI half at
about 1.5 card-hours, and the stop3 read at 4.3 job-hours. The control's PAI half
(`runs/defect_fix/stand_r/pai/STANDFMXH1K-L18-s0`) is on disk but unused, because no arm reached stage 2.
