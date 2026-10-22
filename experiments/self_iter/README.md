# self_iter: open-loop self-iteration by hard-example mining

status: paused
decisions: 338
index: round 1 pilot STOP: trainval rows teach standstill holds, launches lag; round 2 shelved for this board by the user (2026-10-17 afternoon)

**Question.** Can drawing the rows a policy gets wrong more often (labels from the human log, the right rows still mixed in) fix those errors
without losing anything else, on navtrain plus the trainval logs on disk?

**Conclusion.** Round 1 pilot STOP: the hard decile drops 46-53 %, but dev ADE rises by 0.017-0.044 m and standstill launches lag by 1.0-1.7 m. In the unseen trainval rows, 64 % of the hard decile are rows where the human holds and RSG drives off. Adding those rows, with or without mining, teaches the hold and delays the launch (decision 338).

**Next.** For the user to decide: a navtrain-only mining pilot (no trainval rows), or trainval rows rebalanced to navtrain's class mix, before any
full-scale rerun (`SFX=2 si.sh launch-full`) or round 2.

**Read more.** [plans/2026-10-17-self-iter-prereg.md](plans/2026-10-17-self-iter-prereg.md); the full trainval download audit (integrity, leakage, class census, safe uses): [research/next-round/trainval-use.md](../../research/next-round/trainval-use.md), `scripts/tv_audit.py`

<!-- files:begin -->
## Files

- `2026-10-17-self-iter-prereg.md` (plans): SELF-ITER 预登记：开环自迭代（难例挖掘），标签 = …
- `si.py` (scripts): open-loop hard-example mining with the …
- `si.sh` (scripts): Box, repo root
- `self_iter.py` (lib): SELF-ITER hard-example mining
- `rm_train.py` (experiments/route_mem/scripts): onpol_nuplan's on_train.py `train` (its …

[archive/](archive/) 1 one-off code · [results/](results/) 16 result files · [plans/](plans/) 1 live plans · [scripts/](scripts/) 3 entry points
<!-- files:end -->

Layout: `scripts/` entry points (live), `lib/` code other topics import, `archive/` one-off code of a concluded
experiment, `results/` small result files, `figs/` figures, `plans/` live plan notes. Refresh the file list and
INDEX.md with `python tools/topic_index.py --topic <name>` (your topic only; a bare run or `--all` would rewrite
other lanes' READMEs, so `--all` is a dry run unless `--write`; `--check` is read-only).
