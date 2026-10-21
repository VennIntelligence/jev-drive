# round8: R6DAG + keep-pace floor + Harmonizer twins

status: live
decisions: 442, 443
index: R8K (R6DAG + keep-pace/F2 + Harmonizer twins) and the Harmonizer render-noise check

**Question.** Does R6DAG plus only round 7's longitudinal hygiene (keep-pace floor, F2) and Harmonizer-ON twin tokens beat R6DAG with the
Harmonizer both OFF and ON, without R7J's lateral loss? Is a Harmonizer-ON closed-loop read reproducible?

**Conclusion.** Harmonizer-ON re-reads are not bit-identical but the 279-scene mean noise is about 0.003, so d439 stands (decision 442). R8K is not adopted: OFF 0.799 / 0.808 vs R6DAG 0.782 / 0.819, s1 fails the mean and offroad rules, s0 ON 0.776 vs 0.799 (decision 443, interim).

**Next.** Copy the final report (s1 ON read, ruler, verdict) from `$DATA_DIR/runs/round8/r8k/results/` into `results/r8k/` and finish row 443.

**Read more.** Plan (Chinese, registered before any read): [plans/2026-10-22-r8k.md](plans/2026-10-22-r8k.md). Box: `$DATA_DIR/runs/round8/`.

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

[results/](results/) 6 result files · [plans/](plans/) 1 live plans · [scripts/](scripts/) 2 entry points
<!-- files:end -->

Layout: `scripts/` entry points (live), `results/` small result files, `plans/` live plan notes.
