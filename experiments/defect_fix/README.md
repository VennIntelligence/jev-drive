# defect_fix: NR1 joint arm against the user's defects

status: live
decisions: pending
index: NR1 prereg written; pilots (3 base doses, 2 lateral doses) running

**Question.** On top of V2 (RTB1AR base -> RTB1ARO-L18), does one joint arm aimed at the defects the user saw in the zero-score videos (slow
launches and waiting: trainval rows with a visible reason to stand, balanced by launch rows; creep: history-speed noise; lateral failures: a
hard / soft constraint hinge on kerbs, other vehicles and opposite-direction lanes) beat RSG-L18-s0 clearly (prereg section 5)?

**Conclusion.** Not read yet.

**Next.** Pilots: base doses NR1B05 / NR1B10 / NR1BALL-F-s0 (open-loop lines, plain continuations on held 460 / public 1491 / the waiting set),
L18 pilots NR1H0P / H15P / H30P-L18-s0 after the label gate; then the full chain NR1F-L18-s0 / s1 (prereg 4-6).

**Read more.** [plans/2026-10-17-nr1-prereg.md](plans/2026-10-17-nr1-prereg.md) (Chinese; design, gates, bar, branch rules).

<!-- files:begin -->
## Files

- `2026-10-17-nr1-prereg.md` (plans): NR1 预登记：下一轮联合训练臂（对准用户看出的缺陷）
- `df.sh` (scripts): Box, repo root
- `df.py` (scripts): NR1, the next-round joint arm against …
- `df_lat.py` (scripts): NR1 lateral constraint labels
- `nr1_lat.py` (lib): NR1 hard / soft lateral constraint …
- `rm_train.py` (experiments/route_mem/scripts): onpol_nuplan's on_train.py `train` (its …
- `handoff-2026-10-17-losses-and-plan.md` (tmp): 交接：2026-10-17 失分诊断汇总、下一轮训练方案、在跑任务
- `census_red_rear.md` (experiments/wait_cap/results/next): Red-light and rear-end census
- `explore_g.md` (experiments/wait_cap/results/next): does the retain line track turns worse …
- `wait_set.md` (experiments/alpasim_proxy/results): a local closed-loop set where the …

[results/](results/) 0 result files · [plans/](plans/) 1 live plans · [lib/](lib/) 1 library · [scripts/](scripts/) 3 entry points
<!-- files:end -->

Layout: `scripts/` entry points (live), `lib/` code other topics import, `archive/` one-off code of a concluded
experiment, `results/` small result files, `figs/` figures, `plans/` live plan notes. Refresh the file list and
INDEX.md with `python tools/topic_index.py --topic <name>` (your topic only; a bare run or `--all` would rewrite
other lanes' READMEs, so `--all` is a dry run unless `--write`; `--check` is read-only).
