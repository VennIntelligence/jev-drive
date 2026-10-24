# NTV1 K / H: D1 + R8K's two differences, one at a time (ROUND9, 2026-10-24)

Pre-registration (Chinese): [../../plans/2026-10-24-ntv1-kh.md](../../plans/2026-10-24-ntv1-kh.md), amendment 1 (D1-H dropped). Code:
[../../scripts/kh.sh](../../scripts/kh.sh), [../../scripts/kh.py](../../scripts/kh.py). The b5 teacher and the G1 label gates are round 7's
`joint.py`, called as written with its path constants re-pointed. Box: `$DATA_DIR/runs/round9/ntv1_kh/`. Full verdict: [verdict.md](verdict.md)
([verdict.json](verdict.json)). One seed (seed 0).

**Result.** K (R8K's pacing floor on D1's DAgger rows) is **neutral**. D1-K reads 0.868 on the new scorer with Harmonizer OFF, the same as
D1-s0's 0.868 (+0.000 [-0.022, +0.022], line +-0.015). With Harmonizer ON it reads 0.846, against D1-s0's 0.851 (-0.005 [-0.028, +0.018]).
The ON - OFF gap does not shrink either: -0.022 against D1's -0.017, a difference of -0.005 [-0.035, +0.024]. **D1-H was not run: cost estimate
7-10 card-hours plus an inexact replay driver; dropped by the user** (decision 458: the board does not run the Harmonizer).

## What K changed

The arm is NTK1-F-s0. It takes D1's argv and changes only `--on on1_JTDCK --on-wfile runs/round9/ntv1_kh/w/NTV1K.npz`; the token diff against
NT3D1-F-s0's stored argv is exactly `--tag`, `--on` and `--on-wfile`.

- **Teacher b5** on P1L's stored DAgger states (on1_PAIDC, 719 scenes): R7J's rule, verbatim. Where all its conditions hold, the arc lengths
  become max(student, min(human, v0 t + t^2)); otherwise the b3 label is kept.
- **Rows floored:** 17 515 of the 92 561 drawable rows (18.9 %; R7J 16.3 %).
  - The 4 s arc grows by p10 / p50 / p90 0.78 / 5.64 / 18.4 m.
  - Straight rows: 11 372 of 63 599 floored.
  - By class: launch 561 / 6 547, stop_nolead 1 941 / 18 703, turn_sharp 4 800 / 20 609, other 10 213 / 46 702.
- **Rows not floored:**
  - no human window: 12 541
  - human slows: 7 742
  - clock lag: 9 421
  - student not slower: 33 183
  - lead object: 16 829
  - reverted at the T4 line: 1 235
- **F2 on the b5 labels:** 6 060 drawable rows go to weight 0 (the same rule on the b3 labels would take 9 862). The drawable set falls from
  92 561 to 86 501, still uniform, with the pd share kept at 0.2. The S weights are bit-equal to NTV1D's.

## Gates (all pass)

| gate | result |
|---|---|
| G1 b5 labels (R7J's lines) | G1a b3 reproduced: max diff 0 over 100 % of the drawable rows. G1b one-sided: min arc change -0.9 mm, change minus floor max +4.2 mm, floor over max(student, human) 0. G1c median clock lag 0.000 s at k = 0. T4 p99 3.72 (b3 3.57) |
| G-W draw file / pools | S weights bit-equal to NTV1D; pd non-zero = drawable minus F2, uniform, mode both, finite labels; shares 0.8 / 0.2. on1_JTDCK differs from on1_JTDC only in the fut of the 17 515 floored rows. on1_JTDCI tab is identical to on1_JTDC |
| D1's argv vs stored NT3D1-F-s0 (to step 100) | 131 / 131 scalars bit-identical |
| D1-K pre-T vs stored NT3D1-F-s0 | 131 / 131 bit-identical |
| switch off after T (on1_JTDCI vs on1_JTDC, 60 steps, T = 30) | 245 / 245 bit-identical, pool drawn |
| weight 0 after T (on1_JTDCK vs on1_JTDC on NTV1D0) | 231 / 231 bit-identical, pool drawn |
| smokes D1 / D1-K (60 steps) | batch 75 / 13 / 40 mixed, S share after T 0.813, draw shares S 0.8 / pd 0.2, lr matches the formula |
| K switch acts | scalars up to step 30 equal (79 / 79); 67 loss scalars differ after T |
| ratio (500 steps after T) | 0.352 (band 0.21-1.04); whole post-T segment 0.423 |
| post | dev ADE 0.5336 <= 0.566 (D1-s0 0.5297); lr over the whole run as the formula |
| twin coverage | PAIHTW rows matching D1's pool: 0 of 282 226, so `--twin PAIHTW:1.0` would swap nothing; R8K's twin-rate gate does not apply |

## Reads (PAI 281, ACC + GAP, D1's serve; new scorer primary)

| read | OFF new | OFF old | ON new | ON old | OFF zeros new / old |
|---|--:|--:|--:|--:|--:|
| D1-K | **0.868** | 0.792 | **0.846** | 0.771 | 28 / 49 |
| D1-s0 | 0.868 | 0.799 | 0.851 | 0.772 | 27 / 46 |
| R8K-s2 | 0.872 | 0.804 | 0.866 | 0.794 | 28 / 47 |
| P1L | 0.813 | 0.734 | 0.804 | 0.733 | 39 / 61 |
| NTV0-s0 | 0.814 | 0.742 | 0.809 | 0.745 | 39 / 59 |

OFF has 278 common scenes and ON 279. Paired differences, D1-K minus each other read (new scorer; old scorer in brackets):

| | OFF | ON |
|---|---|---|
| - D1-s0 | +0.000 [-0.022, +0.022] (old -0.007) | -0.005 [-0.028, +0.018] (old -0.001) |
| - R8K-s2 | -0.005 [-0.039, +0.030] | -0.020 [-0.055, +0.015] |
| - P1L | +0.055 [+0.018, +0.094] | +0.042 [+0.002, +0.081] |
| - NTV0-s0 | +0.053 [+0.012, +0.096] | +0.037 [-0.004, +0.076] |

**Classes, D1-K - D1-s0** (old-scorer fclass counts, 95 % CI). K does what it was built for on the longitudinal partial classes, and gives that back
in zeros:

| | slow | rear-hit | offroad | overshoot | AF-front |
|---|--:|--:|--:|--:|--:|
| OFF | -4 [-8, -1] | -4 [-9, 0] | +4 [0, +9] | +2 [0, +5] | |
| ON | | -6 [-12, -1] | | | +3 [0, +7] |

Total old-scorer zeros: +3 OFF and +3 ON.

**Negative flips** on NTV0-s0's full-score scenes:

| | count A (202 / 203 scenes) | of them to 0 | count B | R8K-s2's fixes kept |
|---|--:|--:|--:|--:|
| D1-K OFF | 14 | 9 | 9 | 33 / 40 |
| D1-s0 OFF | 14 | 8 | 8 | 28 / 40 |
| R8K-s2 OFF | 14 | 9 | 9 | 40 / 40 |
| D1-K ON | 18 | 11 | | |
| D1-s0 ON | 18 | 10 | | |
| R8K-s2 ON | 11 | 8 | | |

**ON - OFF gap** (277 scenes common to both blocks; new scorer): D1-K -0.022, D1-s0 -0.017, R8K-s2 -0.007, P1L -0.009, NTV0-s0 -0.006. D1-K's
gap minus D1-s0's is -0.005 [-0.035, +0.024]; on the old scorer +0.006 [-0.028, +0.039]. The pacing floor does not explain R8K's small ON loss.

## Lines (registered: D1-K - D1-s0, new scorer, OFF; +-0.015)

- OFF new: +0.000 -> **neutral** (registered). OFF old -0.007, ON new -0.005, ON old -0.001: all neutral.
- D1-H: not run. Cost estimate in plan section 2: twins for on1_PAIDC's 719 scenes would take about 30 render job-hours, about 7.4-9.9 card-hours
  (harm-mix measured 148 job-s per scene). They would also need an approximate replay driver, because the student states cannot be replayed
  exactly: poses are stored at 1e-4 and `rollout.asl` was deleted. Dropped by the user.

## Cost

14.2 GPU job-hours: training 1.38, reads 12.66 (OFF + ON), gates / smokes 0.20. Wall time 3.1 h. The b5 teach ran on 46 CPU workers.

## Process note

The first report printed every S-row name prefix into `twin_cover.json` / `verdict.md` (1.8 MB each). That was a fault in my check script only,
with the counts unchanged. `twincov` now summarises the prefixes per part (commit 3605213f), and only twincov + verdict were rerun (`kh-rereport`).
No read or training was redone.
