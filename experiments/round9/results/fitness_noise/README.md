# FITNESS-NOISE: how many practice scenes a PBT fitness needs before it ranks signal, not noise (2026-10-22)

CPU only, no new reads. The question comes from the round-9 draft, section 1 ([plan](../../plans/2026-10-22-round9-draft.md)). It
proposes a PBT controller that scores learners on a fixed practice slice of 60-100 training scenes with Harmonizer-OFF reads, and asks
whether that slice can rank learners whose true gap is about one arm effect (0.02-0.05). As a proxy for the noise structure, this uses
the stored PAI 281 ACC + GAP OFF reads. Code: [scripts/fitness_noise.py](../../scripts/fitness_noise.py). Full numbers:
[fitness_noise.json](fitness_noise.json).

## Inputs

- 22 OFF reads from `$DATA_DIR/runs/pai_fix/facts/<tag>-281ag.csv`, covering 12 recipes: NTV0, R2C, R2J, R5A (1 seed each); R3MV, R5LC, R6FIX,
  R6REC, R6DAG2, R7J (2 seeds each); R6DAG, R8K (3 seeds each). The analysis uses the 272 scenes that every read covers.
- OFF reads repeat bit for bit (decision 441 serve gate; decision 442), so a fixed slice carries no read noise. The only noise is which
  scenes are in the slice, plus the seed / snapshot difference between checkpoints. ON reads would add very little (decision 442: sd about
  0.003 on 279 scenes).
- Truth for a checkpoint pair is D, the mean per-scene difference over all 272 scenes. Truth for a recipe pair is the difference of the
  all-seed recipe means.
- Slices are drawn with replacement, which treats the 281 as a large population, as the training-scene pool is. Drawing without
  replacement from the 281 alone is in the JSON (`ckpt_without_replacement`); it looks better at K >= 140 only because the slice then
  covers most of the truth set.
- Fitness = slice mean of `score - lam * break`. Here `break` = NTV0 scores >= 0.98 on the scene and the learner scores < 0.98, and
  lam is 0, 0.5 or 1.

## 1. Noise structure

| quantity | value |
|---|--:|
| per-scene paired difference sd, close unrelated checkpoints (\|D\| < 0.06) | 0.32 |
| per-scene paired difference sd, same-recipe seed pairs | 0.22 |
| scenes on which two checkpoints differ (median pair) | 62 / 272 (23 %) |
| sd of the slice mean difference, close pairs, K = 40 / 60 / 100 / 200 | 0.053 / 0.044 / 0.034 / 0.024 |
| seed sd of the recipe means on 272 scenes (pooled within recipe, 10 df) | 0.014 |
| scene x seed interaction sd per scene | 0.169 |
| share of that seed-mean variance that is only 272-scene sampling | 51 % |
| population seed sd (two-way moment estimate, rough) | 0.010 |
| persistence of the per-scene difference pattern across snapshot pairs (rho, corr of d over scenes, disjoint seeds) | 0.38 (IQR 0.25-0.51) |

The "seed noise about 0.03 on 281" breaks into two parts. About half the variance is scene sampling inside the 281 itself. The other
half is a real checkpoint difference with sd about 0.010 per checkpoint, or about 0.014 for a pair. A larger slice removes the first
part. No slice size removes the second part when the question is which θ is better.

## 2. Question 1: P(slice ranks a checkpoint pair as the full set does)

Uniform slice, lam = 0. Each cell gives P for pairs with |D| in [0.01, 0.02) / [0.02, 0.035) / [0.035, 0.06). There are 30 / 30 / 56 of
the 231 read pairs in those bands. NTV0 vs the R6DAG / R8K / R7J reads (D 0.034-0.064) falls mostly in the last band.

| design | K = 40 | 60 | 80 | 100 | 140 | 200 |
|---|---|---|---|---|---|---|
| uniform, lam 0 | 0.63 / 0.72 / 0.82 | 0.65 / 0.76 / 0.86 | 0.68 / 0.80 / 0.90 | 0.70 / 0.82 / 0.92 | 0.73 / 0.86 / 0.95 | 0.76 / 0.90 / 0.97 |
| stratified by NTV0 solved / failed | 0.63 / 0.72 / 0.83 | 0.66 / 0.77 / 0.87 | 0.67 / 0.80 / 0.90 | 0.70 / 0.83 / 0.93 | 0.73 / 0.87 / 0.96 | 0.77 / 0.91 / 0.98 |
| Neyman by NTV0 failure class (ok / partial / overshoot / zero-other) | 0.63 / 0.73 / 0.83 | 0.66 / 0.78 / 0.88 | 0.69 / 0.81 / 0.91 | 0.70 / 0.83 / 0.93 | 0.73 / 0.87 / 0.96 | 0.77 / 0.91 / 0.98 |
| stratified by stratum (stop/lead, straight, launch, turn) | 0.63 / 0.72 / 0.82 | 0.66 / 0.77 / 0.87 | 0.68 / 0.80 / 0.90 | 0.70 / 0.83 / 0.93 | 0.73 / 0.87 / 0.95 | 0.77 / 0.90 / 0.97 |
| uniform, lam 0.5 (P vs the score truth) | 0.65 / 0.74 / 0.79 | 0.68 / 0.79 / 0.83 | 0.71 / 0.82 / 0.86 | 0.73 / 0.84 / 0.88 | 0.76 / 0.88 / 0.91 | 0.80 / 0.91 / 0.93 |
| uniform, lam 1 (P vs the score truth) | 0.66 / 0.74 / 0.76 | 0.69 / 0.79 / 0.80 | 0.71 / 0.82 / 0.82 | 0.73 / 0.84 / 0.83 | 0.76 / 0.87 / 0.85 | 0.80 / 0.89 / 0.87 |

- Stratifying adds at most 1-2 points. The paired difference variance is about the same in every stratum, and the stratum means of d are
  close to 0, so there is nothing for stratification to remove.
- The new-break penalty makes the estimate noisier: slice sd at K = 100 is 0.034 with lam 0, 0.045 with lam 0.5 and 0.059 with lam 1. It
  also moves the ranking away from the score. In the 0.035-0.06 band at K = 100, P falls from 0.92 to 0.88 (lam 0.5) and 0.83 (lam 1).
  Ranked against its own full-set fitness instead of the score, lam 1 is 0.85, so it is no easier to estimate either. It helps slightly
  only in the 0.01-0.035 bands (+0.02-0.03), where breaks on NTV0-solved scenes are much of the difference.
- False differences. Same-recipe seed pairs (12 pairs, full-set differences -0.041 ... +0.030) show a slice difference of at least
  0.03 with P 0.43 / 0.41 / 0.35 / 0.33 / 0.29 / 0.26 at K = 40 / 60 / 80 / 100 / 140 / 200. P(at least 0.02) is 0.65 at K = 40 and 0.46
  at K = 200. Even at K = 200, one comparison of single checkpoints from the same recipe looks like an arm effect a quarter of the time.

Recipe level (one random seed per learner; truth = all-seed recipe mean). For the 21 recipe pairs with gap 0.035-0.06, P is 0.81 / 0.86 /
0.89 / 0.91 / 0.94 / 0.96 at K = 40 ... 200. For the 7 pairs with gap 0.01-0.02, P is 0.59 ... 0.69, and only 0.88 on the full 272
scenes. For the 14 pairs with gap below 0.01, P is 0.74 on the full set, which is a coin flip weighted by seed luck.

## 3. Question 2: informative slices (scenes where checkpoints disagree)

The design uses a temporal split, as the real loop would. The informative set is picked from the older reads (NTV0, R2C, R2J, R3MV,
R5A, R5LC, R6FIX, R6REC; 136 of 272 scenes have a score range >= 0.02 among them). It is then tested on the 55 newer pairs: R6DAG,
R6DAG2, R7J and R8K reads among themselves and against NTV0.

| K | uniform | random from old disagreement pool | top-K by old-read variance (fixed) | 70 % pool + 30 % uniform |
|--:|--:|--:|--:|--:|
| P(correct), test pairs with \|D\| >= 0.02 (21), truth on all 272 | | | | |
| 40 | 0.78 | 0.84 | 1.00 | 0.82 |
| 100 | 0.89 | 0.92 | 1.00 | 0.91 |
| 200 | 0.95 | 0.97 | 1.00 | 0.96 |
| P(correct), split-half: slice from a random scene half A, truth on the other half B (300 splits) | | | | |
| 40 | 0.63 | | 0.68 | 0.72 |
| 60 | 0.64 | | 0.67 | 0.69 |
| 80 | 0.65 | | 0.69 | 0.69 |
| 100 | 0.66 | | 0.70 | 0.70 |

- The 1.00 for the fixed top-variance slice is optimistic. The slice is part of the truth set, and the truth is concentrated on exactly
  those scenes. In the split-half test the truth comes from scenes the slice never saw, and the half-set truth (136 scenes) is itself
  noisy, so every P there is lower. In that honest test the informative slice still wins by +0.04-0.05 at equal K. Top-variance with
  K = 40 (0.68) beats uniform with K = 100 (0.66), so per scene the informative slice is worth about 2.5 uniform scenes.
- Selecting the slice on the same reads it later ranks inflates the result further (`topvar_all_insample` vs `topvar_old`). The
  disagreement pool must come from earlier rounds' reads, never from the reads being ranked.
- Coverage risk. 12 % of the newer pairs' absolute difference mass, and 12 of the 159 new breaks of the 10 newer reads on NTV0-solved scenes, fall on scenes
  where the older reads never disagreed. A slice made only of informative scenes cannot see a learner that breaks something every past
  model solved. Keeping about 30 % uniform scenes keeps the power (column 4) and restores that coverage.
- Selection overfitting in the loop. Picking the best of 4 learners on a fixed slice overstates the winner by about se x E[max of 4
  normals] ≈ 0.024 x 1.03 ≈ +0.025 at K = 100. With rho ≈ 0.4 persistence, part of that luck carries into the next round. Slice-score
  gains across rounds are therefore not progress; only the final 281 read (kept untouched in the plan) is.

## 4. Question 3: practical answer

Normal model `P = Phi(gap * sqrt(K) / sd_d)` for checkpoints. For recipes (θ, one learner each),
`P = Phi(gap / sqrt(sd_d^2 / K + 2 * 0.010^2))`. The model reproduces the empirical bands above within about 0.02.

| true gap | sd_d | K for 80 %, checkpoint | K for 80 %, θ (one learner each) | P at K = 60 / 100 / 140 | P, 3-round average, fixed slice, K = 100 | P, 3-round average, rotating slice, K = 100 | P(top of 4 is best), K = 100 |
|--:|--:|--:|--:|---|--:|--:|--:|
| 0.02 | 0.32 (unrelated) | 185 | 287 | 0.69 / 0.73 / 0.77 | 0.79 | 0.86 | 0.68 |
| 0.03 | 0.32 | 82 | 98 | 0.76 / 0.82 / 0.87 | 0.89 | 0.95 | 0.81 |
| 0.05 | 0.32 | 30 | 32 | 0.89 / 0.94 / 0.97 | 0.98 | 1.00 | 0.94 |
| 0.02 | 0.22 (siblings) | 89 | 138 | 0.76 / 0.82 / 0.86 | 0.88 | 0.94 | 0.80 |
| 0.03 | 0.22 | 40 | 47 | 0.85 / 0.91 / 0.94 | 0.96 | 0.99 | 0.91 |
| 0.05 | 0.22 | 15 | 16 | 0.96 / 0.99 / 1.00 | 1.00 | 1.00 | 0.99 |

(4-learner case: the learners' true values are spaced one gap apart.)

- **Minimum slice.** Use 100 scenes as the floor if learners can be as different as unrelated recipes; 60 is enough if they are
  siblings from one exploit copy. Per scene, about 70 % should come from the disagreement pool of earlier rounds and 30 % should be
  uniform, which is worth about 1.5-2.5x its size in uniform scenes. With that slice:
  - gap 0.05: >= 0.94 in every setting;
  - gap 0.03: >= 0.82 pairwise and about 0.80 for top-of-4;
  - gap 0.02: needs about 185 uniform scenes in a single round, or 100 scenes averaged over 3 rounds on a rotating slice (0.86);
  - a single round of 60-100 scenes is never >= 0.80 at gap 0.02 between unrelated learners.
- **Fitness.** Rank on the plain mean score. Use new breaks on NTV0-solved scenes as a veto: drop a learner whose break count is far
  above the incumbent's, for example more than +3 per 100 scenes. Do not use them as a lam term inside the ranking, because lam 1 costs
  about 0.09 of P in the 0.035-0.06 band. If a penalty is wanted, lam <= 0.5.
- **Combining rounds.** This helps, but only where the error is not persistent.
  - A fixed slice keeps the persistent part of the slice error: the variance factor over R rounds is rho + (1 - rho) / R = 0.59 for
    R = 3.
  - A slice rotated each round makes the errors independent (factor 1/3). Pairing within a round is unchanged, because all learners in
    a round read the same scenes.
  - Empirically, recipe lineages over 3 matched-seed rounds at K = 60, gap 0.035-0.06: single round 0.86, fixed-slice average 0.94,
    rotating average 0.97. At K = 100: 0.92 / 0.98 / 0.99.
  - Recommended: a fixed informative core (about 70 %) plus a uniform part re-drawn each round (about 30 %), with fitness = EMA over
    2-3 rounds that carries through exploit copies.
  - Caveat: these "rounds" were independent seeds. Successive PBT snapshots are more alike, so rho will be higher and the gain from
    averaging a fixed slice smaller. The EMA also lags real improvement by about 1 / alpha rounds.
- **θ vs weights.** Choosing θ (the controller's purpose) has a floor no slice removes: with one learner per θ and a population seed sd
  of 0.010, P never exceeds 0.92 at gap 0.02 or 0.98 at gap 0.03, even with K = ∞. Gap-0.02 θ decisions need repeated wins across
  rounds. Copying weights (the exploit step) has no such floor.

## 5. Transfer from 281 to a practice slice of training scenes

- **Ceiling and sparsity.** Learners have rows and rollouts from training scenes. If the practice scenes, or close neighbours, appear in
  the buffer, more of them score 1.0 for every learner. Fewer scenes then carry the comparison, sd_d shrinks, and true gaps shrink too.
  Gaps measured there track fit to the training distribution, not generalisation. The plan holds the practice scenes out of rows and
  rollouts; keep them out of PLR sampling and the DAgger buffer as well.
- **Class mix.** PAI 718 training scenes may have a different stop / turn / launch mix than the 281. The per-class sd of d is similar
  across strata here, so the K numbers move little, but gap sizes per class would.
- **Siblings.** PBT learners share weights after every exploit, so their per-scene differences should be closer to the same-recipe sd
  (0.22) or below. Their true gaps after one 1 000-1 500-step round are also likely below 0.02. The open question is whether
  round-level gaps of 0.03 exist at all. Estimate it in the first rounds from the slice itself, using the spread of learner scores
  against the K-scene noise floor (sd 0.022 x sqrt(100 / K) for siblings).
- **nuPlan practice scenes** (mean about 0.93) are much nearer the ceiling, with a different difference structure. These numbers do
  not carry over; repeat this analysis on stored nuPlan held-460 reads before using a nuPlan slice.
- **Pool drift.** The disagreement pool drifts as the model improves. Refresh it every few rounds from earlier rounds' slice reads,
  never from the reads being ranked.
