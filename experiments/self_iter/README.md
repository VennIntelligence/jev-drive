# self_iter: open-loop self-iteration by hard-example mining

status: live
decisions: none
index: SELF-ITER round 1 running: scan RSG-L18-s0, top-10 % x m, trainval rows

**Question.** Can drawing the rows a policy gets wrong more often (labels from the human log, the right rows still mixed in) fix those errors
without losing anything else, on navtrain plus the trainval logs on disk?

**Conclusion.** None yet (round 1 submitted 2026-10-17).

**Next.** Round 1: scan, three 1 000-step pilots (m 1 / 3 / 6), gate, the long run L and closed loop for passing arms; rerun at full scale when
the trainval download completes (`SFX=2 si.sh launch-full`); round 2 rescans with the round-1 model.

**Read more.** [plans/2026-10-17-self-iter-prereg.md](plans/2026-10-17-self-iter-prereg.md)

<!-- files:begin -->
## Files

- `2026-10-17-self-iter-prereg.md` (plans): SELF-ITER 预登记：开环自迭代（难例挖掘），标签 = …
- `si.py` (scripts): open-loop hard-example mining with the …
- `si.sh` (scripts): Box, repo root
- `self_iter.py` (lib): SELF-ITER hard-example mining
- `rm_train.py` (experiments/route_mem/scripts): onpol_nuplan's on_train.py `train` (its …

[plans/](plans/) 1 live plans · [scripts/](scripts/) 2 entry points
<!-- files:end -->

Layout: `scripts/` entry points (live), `lib/` code other topics import, `archive/` one-off code of a concluded
experiment, `results/` small result files, `figs/` figures, `plans/` live plan notes. Refresh the file list and
INDEX.md with `python tools/topic_index.py --topic <name>` (your topic only; a bare run or `--all` would rewrite
other lanes' READMEs, so `--all` is a dry run unless `--write`; `--check` is read-only).
