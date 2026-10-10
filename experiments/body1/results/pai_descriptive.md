# Descriptive PAI read of the BODY1 checkpoints (Amendment 6 note (j))

**This is a descriptive read; the registered PAI read does not exist (no arm met the nuPlan lines) and this read replaces nothing; it cannot promote any arm, and the 60 PAI scenes are now development scenes for this lane.** In plain numbers (60 PAI scenes, one rollout per scene, tag and serving): under the plain driver the four-seed mean is 0.2026 for the base `P2H10-F`, 0.2780 for `P2H10S-F` (+0.0755 [+0.0216, +0.1363] paired by seed) and 0.2615 for `P2H10B-F` (two seeds, +0.0465 [-0.0256, +0.1249]); under decision 226's PAI serving (`JEV_VCONT=1.0 JEV_LEAD=1`) the base is 0.3545, `P2H10S-F` 0.4211 (+0.0667 [-0.0077, +0.1479]) and `P2H10B-F` 0.4291 (+0.0665 [-0.0108, +0.1509]). Against the base's own seed spread (largest base-pair difference 0.0957 plain, 0.0292 served) the plain-driver `P2H10S` difference is inside the spread and the served differences of both arms are outside it. The per-seed picture is mixed: plain `P2H10S` s1 and s2 are +0.03 / +0.02 with CIs including zero; served `P2H10B` s0 is +0.0086. The served `P2H10B-F-s1` (0.4904) is the single highest tag, 0.013 under alpamayo1's 0.5033 (a 441-scene reference, not comparable in sample). Nothing here is selected or promoted; the driver stays `P2H10-F`.

Tables: [pai/pai_tables.md](pai/pai_tables.md) (all numbers below), `pai/pai_by_tag.csv`, `pai/pai_paired.csv`, `pai/pai_paired.json`; code `scripts/pai_report.py`, `scripts/pai_chain.sh`.

## What was run

60 scenes = the six 10-scene chunk files of PAI2 (`a10 b1 b2a b2b e1 e2`), unharmonised renderer, `pai_native.sh`, CONC 4, one pool job per tag x serving x chunk (`--vram 24 --cpu 6 --ram 40`). Tags: `P2H10-F-s{0..3}`, `P2H10S-F-s{0..3}`, `P2H10B-F-s{0,1}`; servings (i) plain, (ii) `JEV_VCONT=1.0 JEV_LEAD=1`. Reused: base s0 / s1 plain = PAI2's runs (`pai2/base`), base s0 / s1 served = FIX1's arm "ab" (`fix1/paibox/runs/ab_*`), both with the identical command and chunk lists. Newly run: 96 chunk stacks (base s2, s3; all arm tags) plus 3 reruns.

## Scores and zeros by class (mean over the seeds of a family; per-tag rows in pai_tables.md)

| serving | family (seeds) | mean | zeros | at-fault collision | offroad | corridor | slow | progress |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| plain | base (4) | 0.2026 | 43.25 | 12.50 | 14.50 | 16.00 | 9.50 | 0.54 |
| plain | `P2H10S` (4) | 0.2780 | 36.75 | 12.50 | 8.75 | 14.75 | 13.75 | 0.58 |
| plain | base (s0, s1) | 0.2150 | 42.5 | 12.5 | 13.5 | 16.0 | 9.5 | 0.54 |
| plain | `P2H10B` (2) | 0.2615 | 39.0 | 12.0 | 11.0 | 15.5 | 10.5 | 0.58 |
| served | base (4) | 0.3545 | 31.25 | 4.00 | 10.00 | 16.25 | 15.25 | 0.60 |
| served | `P2H10S` (4) | 0.4211 | 27.75 | 3.75 | 8.50 | 15.50 | 14.50 | 0.64 |
| served | base (s0, s1) | 0.3626 | 31.0 | 4.0 | 10.0 | 16.5 | 15.0 | 0.61 |
| served | `P2H10B` (2) | 0.4291 | 27.5 | 4.0 | 10.5 | 13.0 | 14.0 | 0.65 |
| reference | alpamayo1 (441 scenes) | 0.5033 | | | | | | |

Zeros "other" (not collision / offroad / corridor) are driver exceptions (`Expected all tensors to be on the same device`, 8 rollouts of 960 outside the reruns, one scene `77fdd7` three times including PAI2's base); progress = mean `progress_clipped_rel`. Reading: the zeros the arms remove are offroad zeros (plain 14.5 -> 8.75 for `P2H10S`), at-fault collision zeros do not move under either serving (12.5 -> 12.5 plain; 4.0 -> 3.75 / 4.0 served), corridor zeros fall by one to three. The served driver already removes most collision zeros of every tag (12.5 -> 4.0); the arms add on top of that through offroad / corridor zeros and progress (0.60 -> 0.64 / 0.65), not through collisions, the class the open-loop hinges target.

## Arm minus base, paired by seed index and scene (scene bootstrap, 10 000 draws)

| serving | comparison | difference [95 % CI] | outside the base-pair spread |
|:--|:--|:--|:--|
| plain | `P2H10S` s0 / s1 / s2 / s3 | +0.1146 [+0.0504, +0.1883] / +0.0298 [-0.0449, +0.1062] / +0.0206 [-0.0556, +0.0948] / +0.1370 [+0.0622, +0.2196] | s0, s3 yes; s1, s2 no |
| plain | `P2H10S` seed mean | +0.0755 [+0.0216, +0.1363] | no |
| plain | `P2H10B` s0 / s1 | +0.0587 [-0.0125, +0.1377] / +0.0343 [-0.0733, +0.1434] | no / no |
| served | `P2H10S` s0 / s1 / s2 / s3 | +0.0192 [-0.0746, +0.1149] / +0.0695 [-0.0388, +0.1764] / +0.0966 [-0.0062, +0.2036] / +0.0814 [-0.0053, +0.1701] | s0 no; s1-s3 yes |
| served | `P2H10S` seed mean | +0.0667 [-0.0077, +0.1479] | yes |
| served | `P2H10B` s0 / s1 | +0.0086 [-0.0734, +0.0934] / +0.1243 [+0.0225, +0.2317] | no / yes |
| served | `P2H10B` seed mean (2) | +0.0665 [-0.0108, +0.1509] | yes |

Null, base seed against base seed (six pairs of four seeds): plain -0.0957 to +0.0831 (max |.| 0.0957; s1 - s0 +0.0831 [+0.0054, +0.1638], s3 - s1 -0.0957 [-0.1761, -0.0270]); served -0.0292 to +0.0071 (max |.| 0.0292). "Outside the spread" means beyond every base-pair difference of that serving; the plain base spread is wide (0.161 to 0.257), the served one narrow (0.337 to 0.366), which is why the same +0.07 is inside the first and outside the second. Of the four seed-mean CIs three include zero (served `P2H10S` -0.0077, served `P2H10B` -0.0108, plain `P2H10B` -0.0256); the one that excludes zero (plain `P2H10S`) is inside the base spread.

## Figure

![per-seed PAI scores](../figs/pai/pai_per_seed.png)

What to look at: the grey band is the range of the four base seeds; in the right panel (served, the configuration PAI is actually served with) all four `P2H10S` seeds and both `P2H10B` seeds sit above the band (`P2H10B` s0 by only 0.002), and nothing reaches the dashed alpamayo1 line; in the left panel (plain) four of six arm points sit above the wide band (`P2H10S` s2 and `P2H10B` s0 inside it).

## Costs

99 stacks (96 + 3 reruns) x about 12 to 13 minutes, about 20 card-hours (main's brief enlarged the 8 card-hour budget by the second chain; the lane's own estimate for the first 48 stacks was 9.6); wall 10:19 to 13:20 box time, up to 14 stacks in flight (chain 1 CAP 6, chain 2 CAP 8, margin guard 100 GiB, never below 230 GiB observed, pids at most about 10 400 of 20 480). Box: 5.4 GB under `$DATA_DIR/runs/body1/pai` (summaries, logs; videos, renderer caches and rollout logs pruned), 330 GB free on the data disk.

## Deviations and limits

- Two chain instances (second added on the main session's instruction) started from the same script version; the second shares STATUS / log / jobs files with the first (the planned suffix variable was not applied), so `DONE` was written by the first one; the reader checks every run directory instead.
- 10 rollouts of `P2H10S-F-s1` plain (chunks b2b, e1, e2) failed with `No space left on device` in `/tmp` (the box's root overlay, 30 GB, was at 100 % from other lanes' files, 26 GB in `/tmp`); the three stacks were rerun with `TMPDIR` on the data disk and replace the originals (the original summaries are in `runs/bad_enospc/`; the first read of that tag was 0.2133, the rerun 0.2863). All other stacks show no such failure. The root disk remains full and is not this lane's to clean.
- Plain-driver `P2H10S-F-s1` therefore differs from the other stacks in `TMPDIR` only. Eight device-mismatch exceptions remain as scored zeros in all tags alike.
- No run on card 2 or card 3 directly; everything went through the pool.
- One rollout per scene, tag and serving; the simulator reproduces only per identical scene list, the lists are the base's. Chunk-level numerics differ from Tokyo's by decision 215's cm-level effect. PAI2's 60 scenes are a difficulty-stratified sample, so the means do not compare with alpamayo1's 441-scene 0.5033. `P2H10B` has two seeds only. Seeds are paired by index only through the shared scenes, not by any shared training state.
