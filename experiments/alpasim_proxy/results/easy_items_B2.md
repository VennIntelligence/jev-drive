# EASY-ITEMS-B2: where B2's remaining PCS is, against full B (2026-10-20)

Diagnostic, CPU only, no training and no new rollouts. The decision 390 diagnosis ([easy_items_B.md](easy_items_B.md)), re-run unchanged
on the half-dose candidate B2 (`STANDFMXH-L18-s1`, decision 391's nuPlan candidate, and `STANDFMXH-L18-s0`), with full B's seed 1
(`STANDFMX-L18-s1`) for the seed spread. Same 2 106 frozen board-mix items, same strata, same first-match cause rules, frozen PCS on
the Mac (`slow_rescore.Frozen`, log-clustered bootstrap, 2 000 draws).

Inputs: the stored native arena held reads ([stand_b/arena_windows_full_H_s1.csv.gz](../../defect_fix/results/stand_b/arena_windows_full_H_s1.csv.gz),
`_H`, `_s1`) and the per-decision tables of the same reads from the box (`runs/defect_fix/stand_b/ext_full_H`, `ext_full_H_s1`,
`ext_full_s1`). Commands (Mac):

```
P=tmp/pcs_local/.venv/bin/python; S=experiments/alpasim_proxy/scripts/easy_items.py; R=experiments/defect_fix/results/stand_b
$P $S subject --rows $R/arena_windows_full_H_s1.csv.gz --tag STANDFMXH-L18-s1 --dec <ext_full_H_s1> --out easy_items_B2_s1 --misses misses_STANDFMXH-s1
$P $S subject --rows $R/arena_windows_full_H.csv.gz    --tag STANDFMXH-L18-s0 --dec <ext_full_H>    --out easy_items_B2_s0 --misses misses_STANDFMXH-s0
$P $S subject --rows $R/arena_windows_full_s1.csv.gz   --tag STANDFMX-L18-s1  --dec <ext_full_s1>   --out easy_items_B_s1  --misses misses_STANDFMX-s1
```

Outputs: [easy_items_B2_s1.json](easy_items_B2_s1.json), [easy_items_B2_s0.json](easy_items_B2_s0.json), [easy_items_B_s1.json](easy_items_B_s1.json);
per-window classes in `easy_items/misses_STANDFMXH-s{0,1}.csv.gz` and `misses_STANDFMX-s1.csv.gz`. The group rows of the second table come from
the same pricer and the per-window class files.

## Reading

- **The half dose removed fewer easy misses than hoped, and it moved some of them from zeros to partials.** Easy misses go from 48 / 48 (B s0 / s1)
  to 54 / 47 (B2 s0 / s1). Easy zeros fall from 36 / 39 to 30 / 31. Easy partials rise from 12 / 9 to 24 / 16, because the free-road ratchet partials
  come back (0 -> 6 / 3) and so do near-misses and stop / lock. PCS rises only on s1 (1398 vs 1368-1379). B2 s0 is level with B (1369).
- **The overspeed-driven lateral zeros mostly remain.** These are the moving corridor, turn offroad / corridor, straight offroad and AF-front classes.
  Easy windows: 25 / 28 for B, 19 / 20 for B2. AF-front is gone (2 -> 0), corridor 8 / 9 -> 6 / 7, turn 10 / 9 -> 8 / 8, straight offroad unchanged
  at 5. 11 of B2's 19-20 lateral zeros still run above 1.05 x the human's speed (B 13-14 of 23-26). Halving FREE-M took out about 5-8 windows.
  The rest is not dose-driven, or needs much less dose than 0.001.
- **The static red-line roll-outs did not move: 10 / 10 / 10 / 9 easy windows.** 6 windows are shared by all four reads, and 7 by both B2 seeds.
  JTN-s1 + stop3 holds 9-10 of them. Each window fails the same way:
  - at k = 0 the ego stands, the human stands, and the plan already spans 2-4 m in 4 s;
  - the ego creeps, passes 0.5 m/s at about k = 4-5 and then accelerates (plan arc 13-20 m by k = 7);
  - about half of the decisions before the zero (40 / 79 on s1, 41 / 92 on s0) have the ego at 0.5-3 m/s while the human still stands. That is
    outside STAND's rows, which require ego speed < 0.5 m/s. The rest are inside STAND and still plan 4-5 m, so the dose is short too.
- **Largest class now.**
  - By the d390 single-class rule, it is the static red line on both seeds: B2 s1 +29 [+6, +60] board-mix / +54 g15 (9 windows), B2 s0 +25 / +41.
    Next come the turn zeros (+25 / +46 on s1) and the launch partials (+28 / +30 on s1, 5 windows).
  - As groups, the lateral group (19-20 windows, +54 to +80 board-mix) and the easy partials (16-24 windows, +97 to +117) are larger. Each is
    several mechanisms, though, and the lateral group is mostly not overspeed any more.
- **Why each fix alone stays small:** the a-ordered curve still needs about 27-32 fixed windows (B2) to pass garage's 1553, against 33-35 for B.

## Tables

| cause (easy items only) | B s0 | B s1 | B2 s0 | B2 s1 | class PCS easy-only, B s0 (bm / g15) | B2 s0 | B2 s1 | B2 easy speed ratio s0 / s1 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| zero: static, red line, no lead | 10 | 10 | 10 | 9 | +26 / +42 | +25 / +41 | +29 / +54 | - / - |
| zero: turn window, offroad / corridor | 10 | 9 | 8 | 8 | +26 / +42 | +19 / +32 | +25 / +46 | 1.16 / 1.14 |
| zero: corridor, moving non-turn | 8 | 9 | 6 | 7 | +23 / +40 | +16 / +28 | +22 / +41 | 1.26 / 1.16 |
| zero: offroad, non-turn | 5 | 8 | 5 | 5 | +13 / +19 | +12 / +21 | +19 / +39 | 1.04 / 1.05 |
| zero: AF-front collision | 2 | 2 | 0 | 0 | +7 / +12 | - | - | - / - |
| zero: AF-side collision | 1 | 1 | 1 | 1 | +1 / +1 | +1 / +1 | +1 / +2 | 0.88 / 0.88 |
| zero: static, other cue | 0 | 0 | 0 | 1 | - | - | +2 / +3 | - / - |
| partial: launch late / not launching | 6 | 6 | 6 | 5 | +22 / +21 | +22 / +21 | +28 / +30 | 0.82 / 0.07 |
| partial: rear-ended | 1 | 1 | 1 | 1 | +14 / +10 | +13 / +10 | +19 / +15 | - / - |
| partial: near-miss 0.98-0.999 | 3 | 1 | 6 | 4 | +9 / +16 | +12 / +21 | +6 / +9 | 0.93 / 0.61 |
| partial: stop / lock | 2 | 1 | 4 | 2 | +5 / +6 | +9 / +12 | +3 / +4 | 0.42 / 0.43 |
| partial: free-road ratchet / slow | 0 | 0 | 6 | 3 | - | +9 / +12 | +4 / +5 | 0.68 / 0.63 |
| partial: following a lead | 0 | 0 | 1 | 1 | - | +2 / +2 | +2 / +3 | 0.77 / 0.80 |
| **all easy misses (zeros / partials)** | 48 (36 / 12) | 48 (39 / 9) | 54 (30 / 24) | 47 (31 / 16) | | | | |

| | B s0 | B s1 | B2 s0 | B2 s1 |
|:--|--:|--:|--:|--:|
| frozen-pool PCS board-mix / g15 | 1368 / 1475 | 1379 / 1481 | 1369 / 1482 | 1398 / 1526 |
| mean, zeros, partials (2 106 items) | 0.9368, 94, 72 | 0.9318, 108, 64 | 0.9386, 86, 102 | 0.9365, 96, 86 |
| lateral group: moving corridor + turn + straight offroad + AF-front: windows, PCS fixed alone bm [CI] / g15 | 25, +85 [+39, +208] / +163 | 28, +121 [+50, +410] / +232 | 19, +54 [+24, +123] / +100 | 20, +80 [+30, +234] / +178 |
| static red line: windows, PCS fixed alone bm [CI] / g15 | 10, +26 [+3, +51] / +42 | 10, +30 [+4, +55] / +47 | 10, +25 [+3, +54] / +41 | 9, +29 [+6, +60] / +54 |
| easy partials (launch, rear-end, near-miss, stop / lock, ratchet, follow): windows, PCS fixed alone bm [CI] / g15 | 12, +69 [+34, +147] / +68 | 9, +58 [+23, +163] / +49 | 24, +117 [+63, +305] / +126 | 16, +97 [+41, +325] / +91 |
| lateral zeros (not AF) with speed ratio > 1.05 | 13 / 23 | 14 / 26 | 11 / 19 | 11 / 20 |
| easy red-line zeros that JTN-s1 + stop3 scores 1 on | 10 | 10 | 10 | 9 |
| board-mix greedy: windows to pass garage 1553 / #1 1899 (by a) | 35 / 44 | 33 / 40 | 32 / 46 | 27 / 39 |

- Class PCS is each class set to 1 on its easy windows only, priced at the subject's own base. Group rows set the whole group to 1 together.
- The speed ratio is the median over moving decisions of ego speed / human speed at the same station (d384), per window, then the median over the
  class's easy windows.
- Seed spread: B s0 / s1 differ by 6 easy windows each way (42 shared). B2 s0 / s1 share 40 (14 / 7 only in one). Class differences of 1-2 windows
  between B and B2 are within seed noise; the red-line constancy and the AF-front 2 -> 0 hold on both seeds.

## Next

The arm aimed at the red-line class is drafted, not launched, in
[NR1 prereg supplement 20 DRAFT](../../defect_fix/plans/2026-10-20-nr1-supp20-standr-DRAFT.md): STAND-R extends STAND's rows to a rolling
ego (0.5 to 3 m/s) while the human stands, with a station-aligned cap, and can double the dose.
