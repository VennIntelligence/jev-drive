# SLOW-RESCORE: frozen board-mix PCS of the stored native reads, and why JTN-L18-s1 is slow (2026-10-19, d384)

CPU only, no new rollouts. Decision 382 found that on the private PCS the dominant loss is "slow" partial scores on moving windows. This
re-scores stored native arena held reads with the frozen board-mix items of d382 and asks two questions. First, is FREE-M (d378), which
removed the speed ratchet, PCS-positive despite its mean-score guards? Second, does a static-only mean gain (the BN control continuation,
d379) carry PCS?

Code: [scripts/slow_rescore.py](../../scripts/slow_rescore.py) (Mac, `tmp/pcs_local/.venv`). Inputs are the `arena.py collect` rows of
pcs_local (`rows_1`, `rows_3`) and `$DATA_DIR/runs/best_of_n/cont_rows.csv.gz`, plus native_gap's decision table (`tmp/native_gap`).
Outputs: [rescore.json](rescore.json) and [slow_causes_jtn1.csv](slow_causes_jtn1.csv) (one row per short window).

Definitions:
- **PCS**: items frozen (`boardmix.npz`; `g15` = garage weight 15). Each read is scored as one more submission: MAP ability, then the
  6-anchor map (`pcs_local.Scored`). The CI on a difference is a log-clustered bootstrap (2 000 draws) with items fixed.
- **The 6-chunk pilot**:
  - "fill": the 1 390 missing items take JTN-L18-s0 native's scores in both arms.
  - "subset": the MAP uses only the 716 shared items.
- **Window classes**:
  - slow: 0 < score < 0.98, no collision (types slow / creep / no-launch), on a non-static, non-launch window. This is d382's "slow
    progress" class.
  - launch-short: the same, on launch windows.
  - near-1: 0.98 <= score < 1.

## Re-score

| pair (a - b) | windows | mean a / b | d mean [CI] | PCS a / b (board-mix) | d PCS board-mix [CI] | d PCS g15 [CI] | zeros a / b | static zeros | slow | launch-short | near-1 | AF-front |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| FREEM1K - JTNS1K (d378, 6 chunks), fill | 753 | 0.829 / 0.862 | -0.033 [-0.066, -0.000] | 1191 / 1185 | **+5 [-9, +21]** | +11 [-7, +32] | 125 / 96 | 108 / 79 | **10 / 40** | 8 / 12 | 1 / 13 | 8 / 2 |
| same, subset MAP | 716 items | | | 1169 / 1159 | +10 [-25, +68] | +22 [-23, +96] | | | | | | |
| BNC2K-s1 - JTN-s1 (d379) | 2228 | 0.905 / 0.891 | +0.014 [+0.000, +0.034] | 1226 / 1247 | **-20 [-43, +4]** | -34 [-72, +0] | 174 / 210 | 97 / 134 | 114 / 98 | 42 / 44 | 25 / 18 | 6 / 7 |
| BNC2K-s0 - JTN-s0 (d379 suppl. 1) | 2228 | 0.899 / 0.874 | +0.025 [+0.006, +0.052] | 1216 / 1198 | **+17 [-5, +49]** | +23 [-8, +64] | 188 / 247 | 110 / 172 | 116 / 117 | 41 / 40 | 23 / 26 | 6 / 6 |
| BNC4K-s1 - JTN-s1 (d379 suppl. 1) | 2228 | 0.905 / 0.891 | +0.015 [-0.000, +0.034] | 1206 / 1247 | **-41 [-70, -15]** | -74 [-119, -34] | 169 / 210 | 94 / 134 | 140 / 98 | 45 / 44 | 23 / 18 | 6 / 7 |
| JTN-s1 - RSG (native) | 2228 | 0.891 / 0.786 | +0.105 [+0.067, +0.151] | 1247 / 1106 | +140 [+110, +177] | +200 [+160, +247] | 210 / 458 | 134 / 385 | 98 / 139 | 44 / 20 | 18 / 29 | 7 / 23 |
| JTN-s1 - NTV0 (native) | 2228 | 0.891 / 0.770 | +0.121 [+0.084, +0.165] | 1247 / 1104 | +142 [+112, +180] | +201 [+159, +249] | 210 / 494 | 134 / 403 | 98 / 108 | 44 / 23 | 18 / 26 | 7 / 37 |
| NTV0 - RSG (native) | 2228 | 0.770 / 0.786 | -0.016 [-0.025, -0.008] | 1104 / 1106 | -2 [-9, +4] | -1 [-10, +6] | 494 / 458 | 403 / 385 | 108 / 139 | 23 / 20 | 26 / 29 | 37 / 23 |

How to read it:
- **FREE-M is PCS-neutral, not PCS-positive.** It fixes 30 slow partials and 12 near-1s, but it adds 29 static zeros and 6 AF-front zeros.
  The net is +5 / +11 PCS with CIs across 0, far below the +30 line set before the run (below). At native ability a slow fix is worth
  about +0.4 PCS per window (JTN-s1: 98 windows, +39), and a static corridor zero costs about 0.7 (d382: 115 windows, +82). The d382 value of +145 for "slow -> 1" was for
  JTN-s1 + stop3, which has no static zeros left and a higher ability. Natively, all 98 slow partials of JTN-s1 are worth only +39 / +41
  (cause table below). So the pitch "the mean guards hide a PCS gain" does not hold for d378. It would need a model that already stands
  still (STAND) or stop3. On d371's native ruler that is not this arm.
- **A static-only mean gain carries little PCS, and it can be negative.** The BN control continuation gains its mean on static windows: 37
  to 62 fewer static zeros. On s0 that is worth +17 / +23 (CI across 0). On s1 the same recipe adds slow partials (98 -> 114 at 2K,
  -> 140 at 4K), and PCS falls by 20 / 34 at 2K and by 41 / 74 at 4K (that CI excludes 0), while the mean rises +0.014. **Continuing the JTN
  recipe likely re-grows the ratchet** (inferred from the slow counts; there are no decision dumps for these reads). The mean and PCS disagree in sign on 2 of the 3 continuation pairs.
- **Native JTN-s1 over RSG / NTV0 is a real PCS gain** (+140 / +200), almost all from static zeros (385 / 403 -> 134). NTV0 against RSG
  is PCS-flat even though its mean is -0.016: it has 31 fewer slow partials but 14 more AF-front zeros.
- k = 9 same-station speed ratio (native_gap / free_m reads; no decision dumps exist for the BN reads):
  - FREEM1K 1.025, JTNS1K 0.841.
  - JTN-s1 0.912, JTN-s0 0.891, NTV0 0.880, RSG 0.831.

## Why JTN-L18-s1 native is slow

The population is the 142 short windows (98 slow + 44 launch-short). Causes are assigned per window in order, first match, using d372's
decision table (valid decisions k >= 2; ratio = ego / human speed at the same station):
1. late launch: the human starts below 1 m/s;
2. stop / lock: the ego is below 0.3 m/s where the human moves at 2 m/s or more, after the ego had reached 1 m/s;
3. following a lead: a lead within 25 m on at least half the decisions;
4. moving ratchet: last ratio < 0.9;
5. other.

| cause | windows (non-launch) | points lost | median last ratio | PCS if -> 1 (board-mix / g15) | strata |
|:--|--:|--:|--:|:--|:--|
| moving ratchet, free road | 76 (76) | 9.3 | **0.37** | **+30 / +32** | other 38, braking 16, turn 12, cruise 10 |
| late launch | 44 (0) | 15.8 | 0.83 | +1 / +2 | launch 44 |
| following a lead | 13 (13) | 1.6 | 0.52 | +1 / +2 | other 10 |
| stop / lock | 5 (5) | 0.6 | 0.01 | +3 / +3 | braking 5 |
| other | 4 (4) | 1.5 | - | +1 / +1 | |
| all 98 slow (non-launch) | 98 | 13.0 | | +39 / +41 | |
| all 142 short | 142 | 28.8 | | +41 / +44 | |

- **Slow PCS is the free-road ratchet, in its tail.** 76 of the 98 slow windows have no lead at all (lead share 0), start at a median
  5.6 m/s and keep the human's speed (5.1 -> 5.0 m/s), while the ego decays from 5.0 to 1.8 m/s. The median ratio by k is 0.97, 0.90,
  0.81, 0.72, 0.62, 0.52, 0.43, 0.36 (k 2..9). The plan asks for 7-17 % less than the current speed at every step: plan(0.5 s) / v is
  0.93 at k = 2 and 0.83 at k = 9. This is the d372 / d375 loop, at 4-5 times the median per-step drop (-0.12 against -0.029 log). It
  is not one sudden brake: the worst single step carries a median of 24 % of the total drop.
- The late launch costs the most mean (15.8 points) and almost no PCS, because launch items have a of about 0.13 (d382).
- Natively the whole slow class is worth +39 / +41. That is about half of what the 115 static corridor zeros are worth (+82 / +119, d382).
  Slow becomes the dominant lever only once static is held (stop3: +145 / +223).

## Step 2 gate: not met, so no pilot

The line was written into `slow_rescore.py` before it ran: FREEM1K - JTNS1K >= +30 PCS (fill), under both board-mix and g15, with the
subset MAP of the same sign. The result is +5 / +11 (subset +10 / +22), so the gate fails. No dose supplement was written and no job was
submitted. A lower dose would cut the 29 extra static zeros, but by d378 it would also cut the slow fix. Even if it removed the static
leak entirely, the slow fix is worth little at native ability: the whole slow class of JTN-s1 over all 18 chunks is +39 / +41,
and the 6 chunks hold about a third of it, and the 6 extra AF-front windows still
fail the driving guard. Worth re-asking only on a parent that already stands still at red lines (STAND, supplement 17). There each
slow fix is worth about 1.4 PCS (d382, stop3), against about 0.4 here.

Limits:
- 1 seed per arm. FREEM1K covers 6 of 18 chunks, and "fill" assumes JTN-s0's scores for the other 12.
- The garage weight is not identified (d382: ±100 PCS).
- The cause rules are heuristic: first match, thresholds 1 / 0.3 / 2 m/s, 25 m and 0.9.
