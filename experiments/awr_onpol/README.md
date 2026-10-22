# awr_onpol: on-policy plans weighted by their consequences

status: paused
decisions: none yet
index: prepared, never launched: 2 688 train rows with advantage >= 0.1 vs near-identical peers; shelved for this board by the user (2026-10-17 afternoon)

**Question.** Can labels come from consequences instead of hand-made targets: clone the plan an on-policy policy actually output at a
decision only where, in the simulator, it did better over the next 3 s than the other policies that stood in the same state, with no
expert, human path or hinge target?

**Conclusion.** None yet. The pilot is prepared and waits for the user (prereg section 7).

**Next.** Score, identity (bit-identical, on_hash_60 ea8cacc3e451c577) and 40-step smoke are done; 2 688 train rows have advantage
>= 0.1 against near-identical peers ([results/score.md](results/score.md)). The user decides on `aw.sh pilot` (keep 0.1 / 0.5, 1 000 steps,
held 460 @18 only on a pass).

**Read more.** [plans/2026-10-17-awr-onpol-prereg.md](plans/2026-10-17-awr-onpol-prereg.md); the pool: decision 333 and
`experiments/route_follow/plans/2026-10-17-rsg-acc-prereg.md`; the hand-label lines this replaces: decisions 259, 264, 311, 321, 322;
the scorer line it is not: decision 270.

<!-- files:begin -->
## Files

- `2026-10-17-awr-onpol-prereg.md` (plans): 预登记（草稿，待用户批准后启动）：AWR-ONPOL，用后果给 …
- `aw_score.py` (scripts): AWR-ONPOL outcome score and advantage …
- `aw.py` (scripts): identity, the PREF outcome proxy, pilot …
- `aw.sh` (scripts): consequence-weighted cloning of …
- `awr_train.py` (lib): consequence-weighted cloning of …
- `rm_train.py` (experiments/route_mem/scripts): onpol_nuplan's on_train.py `train` (its …
- `2026-10-17-rsg-acc-prereg.md` (experiments/route_follow/plans): 预登记：RSG-ACC，把 RSG-L18 的配方搬到 ACC serve …

[results/](results/) 4 result files · [plans/](plans/) 1 live plans · [scripts/](scripts/) 3 entry points
<!-- files:end -->

Layout: `scripts/` entry points (live), `results/` small result files, `plans/` live plan notes. The training term lives in
`lib/awr_train.py` (rm_train `--awr-*`, off by default).
