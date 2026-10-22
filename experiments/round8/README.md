# round8: R6DAG + keep-pace floor + Harmonizer twins

status: live
decisions: 442, 443
index: R8K not adopted; 3-seed avg vs R6DAG -0.001; ON noise sd 0.003

**Question.** Does R6DAG plus only round 7's longitudinal hygiene (keep-pace floor, F2) and Harmonizer-ON twin tokens beat R6DAG with the
Harmonizer both OFF and ON, without R7J's lateral loss? Is a Harmonizer-ON closed-loop read reproducible?

**Conclusion.** Harmonizer-ON re-reads are not bit-identical, but the noise on a 279-scene mean is about 0.003, so d439 stands (decision 442). R8K is not adopted (decision 443). OFF it reads 0.799 / 0.808 against R6DAG 0.782 / 0.819, and lateral recovers to R6DAG's level. ON it reads 0.776 / 0.770 against 0.799 / 0.783: the twin tokens did not recover the ON loss.

**Next.** None in this lane. Third seeds (amendment 1): the two recipes cannot be told apart. R8K - R6DAG on 3-seed means is OFF +0.006 [-0.013, +0.025], ON -0.007 [-0.024, +0.010] and (OFF + ON) / 2 -0.001. The best single checkpoint by (OFF + ON) / 2 is R8K-s2 at 0.803, against R6DAG-s1 at 0.801; its ruler is -145 against NTV0 (results/s2/).

**Read more.** Plan (Chinese, registered before any read; amendment 1 = third seed): [plans/2026-10-22-r8k.md](plans/2026-10-22-r8k.md). R8K full table and verdict: [results/r8k/pai281_R8K.md](results/r8k/pai281_R8K.md); ruler: [results/r8k/pcs_r8k.md](results/r8k/pcs_r8k.md); gates: [results/r8k/identity.json](results/r8k/identity.json), [results/r8k/smoke.json](results/r8k/smoke.json); noise check: [results/noise/noise.md](results/noise/noise.md); third seeds: [results/s2/seeds3.md](results/s2/seeds3.md), ruler [results/s2/pcs_s2.md](results/s2/pcs_s2.md). Box: `$DATA_DIR/runs/round8/`.

<!-- files:begin -->
## Files

- `2026-10-22-r8k.md` (plans): 第 8 轮 R8K：R6DAG 配方 + 保持人步调底线 / F2 + …
- `r8k.sh` (scripts): Box, repo root
- `r8k.py` (scripts): checks and reports of the R8K arm and …
- `joint.py` (experiments/round7/scripts): R6DAG's recipe with three changes, one …
- `README.md` (experiments/round7/results/joint): round7 joint arm, 2026-10-21 / 22
- `README.md` (experiments/round7/results/harm_mix): Harmonizer-ON twins of the PAI human …
- `harmonizer_closedloop.md` (experiments/round7/results): Harmonizer closed-loop paired read
- `twin_tokens.py` (lib): the same pool row rendered a second way

[results/](results/) 12 result files · [plans/](plans/) 1 live plans · [scripts/](scripts/) 2 entry points
<!-- files:end -->

Layout: `scripts/` entry points (live), `results/` small result files, `plans/` live plan notes.
