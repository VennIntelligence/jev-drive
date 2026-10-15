# ON-K pilot: on-policy rows per batch on the RSG-L18 recipe (held 452 / 18 decisions, base RSG-L18-s0)

Prereg [plans/2026-10-15-onk-pilot-prereg.md](../../plans/2026-10-15-onk-pilot-prereg.md) (Amendment 1: the first-arm imit tolerance), chain
[scripts/rf_onk.sh](../../scripts/rf_onk.sh). RSG-L18-s0's exact command with `--on-k` {26, 51} at `--ho-w 3` and `--on-k 51 --ho-w 1`; k13 is
RSG-L18-s0 itself (stored checkpoint and read). Pressure = expected on-policy hinge weight per batch, ho_w x k / n_imit with n_imit = 0.75 nN + 13,
relative to k13. Raw outputs: [held18_confirm.md](held18_confirm.md) (log-clustered paired CIs, 39 logs), [held_raw.txt](held_raw.txt) (460
common scenes), [follow.md](follow.md) (24 held turns), [onk.json](onk.json).

## Seed 0

| arm (on-k, ho-w) | rows normal / yr / on | pressure x k13 | tag | zeros 452 | d zeros vs RSG-L18-s0 [95 % CI] | at-fault | d at-fault | col / off / cor | mean score | d score | turn completion (24) | dev ADE |
|:--|:--|--:|:--|--:|:--|--:|:--|:--|--:|:--|--:|--:|
| (13, 3) | 102 / 13 / 13 | 1.00 | RSG-L18-s0 | 26 | - | 21 | - | 4 / 17 / 6 | 0.8809 | - | 66.7 % | 0.5284 |
| (26, 3) | 89 / 13 / 26 | 2.24 | RSGk26-L18-s0 | **22** | -4.0 [-9.6, +1.2] | 18 | -3.0 [-8.8, +2.7] | 7 / 11 / 5 | 0.8910 | +0.0101 [-0.0008, +0.0231] | 66.7 % | 0.5298 |
| (51, 3) | 64 / 13 / 51 | 5.76 | RSGk51-L18-s0 | 25 | -1.0 [-6.4, +4.1] | 20 | -1.0 [-7.1, +5.9] | 7 / 13 / 5 | 0.8890 | +0.0081 [-0.0044, +0.0215] | 58.3 % | 0.5355 |
| (51, 1) | 64 / 13 / 51 | 1.92 | RSGk51w1-L18-s0 | 23 | -3.0 [-7.7, +1.1] | **17** | -4.0 [-8.6, +0.0] | 6 / 11 / 6 | **0.8939** | **+0.0130 [+0.0037, +0.0240]** | **70.8 %** | 0.5298 |
| (13, 3) s1, seed reference | 102 / 13 / 13 | 1.00 | RSG-L18-s1 | 26 | +0.0 [-4.7, +4.9] | 19 | -2.0 [-7.0, +2.6] | 6 / 13 / 8 | 0.8830 | +0.0021 [-0.0097, +0.0137] | 58.3 % | 0.5270 |

Read: d = arm - RSG-L18-s0 on the 452 paired scenes. Turn-completion pairs vs RSG-L18-s0: k26 +0.0, k51 -8.3 [-20.8, +9.1], k51w1 +4.2 [-10.5, +18.2] pp.

**Lines (prereg).** Candidate = zeros CI high < 0 or score CI low > 0. Stop (no seed 1) = zeros CI low > +5.
- k26: no gain (zeros -4.0 [-9.6, +1.2], score CI reaches 0). Not stopped.
- k51: no gain (-1.0 [-6.4, +4.1], +0.0081 [-0.0044, +0.0215]). Not stopped.
- k51w1: **candidate on the score clause** (+0.0130 [+0.0037, +0.0240]); zeros -3.0 [-7.7, +1.1] do not meet the zero line. Not stopped.
- No arm is worse on zeros. All three go to seed 1.

**Reading.** More on-policy rows per batch do not hurt the held read: every arm points to fewer zeros (-1 to -4) and a higher score (+0.008 to
+0.013), but only k51w1's score CI excludes 0. Of three comparisons at 95 %, one passing by chance is plausible; seed 1 is the guard. The arm with
the most hinge pressure (k51 at ho-w 3, 5.8x) is the weakest of the three: least gain, the only drop in turn completion, and dev ADE +0.007
(within the 0.02 line). The same 51 rows at a third of the weight (k51w1, 1.9x) are the best. So the share by itself is not the limit. The
hinge pressure near 2x looks fine; near 6x the gain starts to fade. The k51w1 score gain has the same size as lrsteps' (0.03, 5000) +0.0139,
whose held gain replicated in seed 1 but tied RSG-L18 on public 1491 (lrsteps.md, Amendment 2).

**Checks.** Batch accounting logged as registered (64 / 13 / 51, 89 / 13 / 26; pool 17 429 rows); init dev ADE 0.5276 in every arm; dev ADE
@500 0.5338-0.5386 vs 0.5372, final 0.5298-0.5355 vs 0.5284. Batch imit shifts up with k because the yaw-rate share of the imitation rows
grows (Amendment 1). The registered k26 prediction (steps 100-500 mean 0.745-0.775) reads 0.768. k51w1 at a third of k51's pressure reads
0.811 vs 0.818, so the shift is the mix, not the hinge.

**Cost.** 3 trainings (~7 min each at 6.2-6.9 it/s) + 18 read chunks, 4 per card. Wall 14:58-15:33 on the 3 otherwise idle cards, about 1.8 card-h
(estimate: training 0.4 job-h, reads 3.3 job-h).
