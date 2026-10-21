# round8: R6DAG + keep-pace floor + Harmonizer twins

status: live
decisions: 439, 440, 441
index: R8K (R6DAG + keep-pace/F2 + Harmonizer twins) and the Harmonizer render-noise check

**Question.** Does R6DAG plus only round 7's longitudinal hygiene (keep-pace floor, F2) and Harmonizer-ON twin tokens beat R6DAG with the
Harmonizer both OFF and ON, without R7J's lateral loss? Is a Harmonizer-ON closed-loop read reproducible?

**Conclusion.** Pending.

**Next.** R8K s0 / s1 gates -> training -> PAI 281 OFF / ON -> ruler -> verdict; the noise check runs alongside.

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

[results/](results/) 0 result files · [plans/](plans/) 1 live plans · [scripts/](scripts/) 2 entry points
<!-- files:end -->

Layout: `scripts/` entry points (live), `results/` small result files, `plans/` live plan notes.
