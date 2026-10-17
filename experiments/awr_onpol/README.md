# awr_onpol: on-policy plans weighted by their consequences

status: live
decisions: none yet
index: prepared, not launched: outcome score + near-identical-peer advantage on on1_RSGACC, filtered cloning pilot awaits the user
key: experiments/awr_onpol/plans/2026-10-17-awr-onpol-prereg.md, experiments/awr_onpol/scripts/aw_score.py, experiments/awr_onpol/scripts/aw.py, experiments/awr_onpol/scripts/aw.sh, lib/awr_train.py, experiments/route_mem/scripts/rm_train.py, experiments/awr_onpol/results/score.md, experiments/route_follow/plans/2026-10-17-rsg-acc-prereg.md

**Question.** Can labels come from consequences instead of hand-made targets: clone the plan an on-policy policy actually output at a
decision only where, in the simulator, it did better over the next 3 s than the other policies that stood in the same state, with no
expert, human path or hinge target?

**Conclusion.** None yet. The pilot is prepared and waits for the user (prereg section 7).

**Next.** Fill the coverage table of the prereg from `results/score.md`, check the identity and smoke runs, then the user decides on
`aw.sh pilot` (two keep thresholds, 1 000 steps, held 460 @18 only on a pass).

**Read more.** [plans/2026-10-17-awr-onpol-prereg.md](plans/2026-10-17-awr-onpol-prereg.md); the pool: decision 333 and
`experiments/route_follow/plans/2026-10-17-rsg-acc-prereg.md`; the hand-label lines this replaces: decisions 259, 264, 311, 321, 322;
the scorer line it is not: decision 270.

<!-- files:begin -->
<!-- files:end -->

Layout: `scripts/` entry points (live), `results/` small result files, `plans/` live plan notes. The training term lives in
`lib/awr_train.py` (rm_train `--awr-*`, off by default).
