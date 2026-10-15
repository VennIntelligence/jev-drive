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

## Seed 1

Paired against RSG-L18-s1 (same-seed pair, log-clustered 95 % CI on the 452 scenes), not against RSG-L18-s0.

| arm (on-k, ho-w) | rows normal / yr / on | pressure x k13 | tag | zeros 452 | d zeros vs RSG-L18-s1 [95 % CI] | at-fault | d at-fault | col / off / cor | mean score | d score | turn completion (24) | dev ADE |
|:--|:--|--:|:--|--:|:--|--:|:--|:--|--:|:--|--:|--:|
| (13, 3) | 102 / 13 / 13 | 1.00 | RSG-L18-s1 | 26 | - | 19 | - | 6 / 13 / 8 | 0.8830 | - | 58.3 % | 0.5270 |
| (26, 3) | 89 / 13 / 26 | 2.24 | RSGk26-L18-s1 | 25 | -1.0 [-5.8, +3.1] | 19 | +0.0 [-4.1, +3.8] | 5 / 14 / 6 | 0.8855 | +0.0025 [-0.0066, +0.0122] | 66.7 % | 0.5282 |
| (51, 3) | 64 / 13 / 51 | 5.76 | RSGk51-L18-s1 | 24 | -2.0 [-9.6, +6.7] | 20 | +1.0 [-6.7, +10.1] | 7 / 13 / 4 | 0.8875 | +0.0045 [-0.0125, +0.0197] | 62.5 % | 0.5355 |
| (51, 1) | 64 / 13 / 51 | 1.92 | RSGk51w1-L18-s1 | 22 | -4.0 [-10.9, +3.2] | 18 | -1.0 [-7.9, +6.1] | 6 / 12 / 4 | 0.8934 | +0.0104 [-0.0044, +0.0248] | 66.7 % | 0.5320 |

Turn-completion pairs vs RSG-L18-s1: k26 +8.3 [-9.1, +27.3], k51 +4.2 [-13.8, +23.8], k51w1 +8.3 [-12.0, +27.8] pp. The 24-turn follow read exists for all four
seed-1 tags ([follow.md](follow.md)); nothing was submitted.

**Verdict (seed 1 alone, vs RSG-L18-s1).** No seed-1 arm meets the candidate line (zeros CI high < 0 or score CI low > 0): all three CIs reach 0 on both. k51w1 keeps the seed-0 sign and size (zeros -4.0, score +0.0104 vs +0.0130) but its score CI low is -0.0044. Seed-1 RSG-L18 itself is +0.0021 [-0.0097, +0.0137] above s0 on score, so the two bases differ by about a fifth of the k51w1 gain.

## Two seeds

Per arm, same-seed pairs (s0 vs RSG-L18-s0, s1 vs RSG-L18-s1) and the 2-seed mean minus the RSG-L18 2-seed mean (s0, s1), 452 scenes, log-clustered 95 % CI (`ol_confirm --mean`; [held18_confirm.md](held18_confirm.md)). The RSG-L18 2-seed mean is 26.0 zeros, score 0.8820.

| arm | pressure x k13 | d zeros s0 | d zeros s1 | d zeros 2-seed mean [95 % CI] | d score s0 | d score s1 | d score 2-seed mean [95 % CI] | d at-fault 2-seed mean | d GT (m) 2-seed mean | turn completion s0 / s1 |
|:--|--:|--:|--:|:--|--:|--:|:--|:--|:--|:--|
| RSGk26-L18 (26, 3) | 2.24 | -4.0 | -1.0 | -2.5 [-6.2, +0.5] | +0.0101 | +0.0025 | +0.0063 [-0.0001, +0.0140] | -1.5 [-5.2, +1.8] | -0.044 [-0.095, +0.012] | 66.7 / 66.7 % |
| RSGk51-L18 (51, 3) | 5.76 | -1.0 | -2.0 | -1.5 [-7.2, +4.7] | +0.0081 | +0.0045 | +0.0063 [-0.0067, +0.0183] | +0.0 [-6.2, +7.4] | +0.012 [-0.067, +0.087] | 58.3 / 62.5 % |
| RSGk51w1-L18 (51, 1) | 1.92 | -3.0 | -4.0 | -3.5 [-8.8, +1.8] | +0.0130 | +0.0104 | **+0.0117 [+0.0015, +0.0222]** | -2.5 [-7.8, +2.9] | +0.024 [-0.035, +0.084] | 70.8 / 66.7 % |

**Lines (prereg, on the 2-seed mean).** Candidate = zeros CI high < 0 or score CI low > 0.
- k26: not a candidate (zeros -2.5 [-6.2, +0.5], score +0.0063 [-0.0001, +0.0140]; score CI low is -0.0001).
- k51: not a candidate (zeros -1.5 [-7.2, +4.7], score +0.0063 [-0.0067, +0.0183]).
- k51w1: **candidate on the score clause** (+0.0117 [+0.0015, +0.0222]); the zero clause is not met (-3.5 [-8.8, +1.8]).
- No arm is worse than RSG-L18 on zeros in either seed.

**Reading.** The seed-0 direction replicates in sign for every arm: zeros are lower (-1 to -4) and score higher (+0.0025 to +0.0130) in both seeds, and no 2-seed zero CI excludes 0. The size does not replicate for k26: its score gain falls from +0.0101 to +0.0025, so its 2-seed CI low sits at -0.0001. k51w1 keeps its score clause on 2 seeds, narrowly (CI low +0.0015, against +0.0037 on seed 0), and each seed alone is a pair with the same sign but, in seed 1, a CI that includes 0. It is a pair of seeds with the same sign, not two independent passes. The pressure argument survives: the 5.76x arm (k51) is the weakest on score in both seeds (+0.0081 / +0.0045 against +0.0130 / +0.0104 for the 1.92x arm with the same 51 rows) and has the lowest turn completion of the three arms in both seeds (58.3 / 62.5 % against 70.8 / 66.7 % for k51w1 and 66.7 / 66.7 % for k26), while k26 (2.24x) has a gain that is mostly seed 0 (+0.0101 against +0.0025). The ordering holds in both seeds, but with 2 seeds and overlapping CIs between arms (not tested) it is a description, not an established pressure law. dev ADE final 0.5282-0.5355 in seed 1 against 0.5270 for RSG-L18-s1, inside the 0.02 line. Open: the public 1491 / 10-decision guard read for the 2 seeds of k51w1 is not run (lrsteps' held gain also replicated and then tied RSG-L18 on public 1491).

**Checks.** Batch accounting logged as registered (64 / 13 / 51, 89 / 13 / 26; pool 17 429 rows); init dev ADE 0.5276 in every arm; dev ADE
@500 0.5338-0.5386 vs 0.5372, final 0.5298-0.5355 vs 0.5284. Batch imit shifts up with k because the yaw-rate share of the imitation rows
grows (Amendment 1). The registered k26 prediction (steps 100-500 mean 0.745-0.775) reads 0.768. k51w1 at a third of k51's pressure reads
0.811 vs 0.818, so the shift is the mix, not the hinge.

**Cost.** 3 trainings (~7 min each at 6.2-6.9 it/s) + 18 read chunks, 4 per card. Wall 14:58-15:33 on the 3 otherwise idle cards, about 1.8 card-h
(estimate: training 0.4 job-h, reads 3.3 job-h).
