# zeros_mech: mechanisms behind the S-family zeros

status: live
decisions: 268
index: H3 arm (31% of YS zeros in an oncoming lane); H2 not shown; H1 half

**Question.** On the stored 18-decision held rollouts (452 scenes, YS-L18 / YS2G-L18 / YNONE-L18 / YR10m10-F, two seeds each), do the
zeros of the on-policy S-hinge checkpoints come from turn timing and the route command (H1), a sideways cut-around-the-lead reflex taught by
the shape-only agent hinge (H2), or driving in the oncoming lane that the direction-free road raster does not penalise (H3)?

**Conclusion.** By the registered lines H3 gets an arm (18 / 59 = 0.31 of YS's corridor + collision zeros had the rear axle in an oncoming
lane in the 2 s before the event; base 0.17), H2 is not established (lead-conditioned YS - base |d(3)| +0.079 [-0.069, +0.260] m,
cut-around 0.28 of at-fault collisions, no collision hits the lead) and H1-command holds only for its straight-road half (12 / 12, but the
base also 7 / 7; early-inside 10 < late-outside 14 on turns); exploratory, S flipped turn failures across the oncoming lanes from late-wide to
early-inside cuts (all zero types 6 / 19 -> 21 / 8) while kerb-side turns stay wide (decision 268).

**Next.** Arm B (direction-aware road raster on the on-policy rows) is the candidate these reads support; A and C are not supported. The
coordinator decides whether to train it.

**Read more.** [plans/2026-10-12-zeros-mech-prereg.md](plans/2026-10-12-zeros-mech-prereg.md) (Chinese, pushed before any read);
[results/zm_report.md](results/zm_report.md) (all tables, gates, the 43 events, exploratory section); the human page
[research/zeros-mech/index.html](../../research/zeros-mech/index.html); the clips the hypotheses come from:
[research/onpol-long/zeros-ys-l18.html](../../research/onpol-long/zeros-ys-l18.html).

Figures: [figs/delta_hist.png](figs/delta_hist.png) (turn timing by group), [figs/lead_offset_gap.png](figs/lead_offset_gap.png) (plan offset
under a slower lead by gap), [figs/oncoming_by_k.png](figs/oncoming_by_k.png) (oncoming-lane share by decision), [figs/bev_examples.png](figs/bev_examples.png)
(one decision per class); each is explained on the HTML page. Box: `$DATA_DIR/runs/zeros_mech/collect/20261012-155215` (per-decision parquet,
BEV pickle), `rows/20261012-154945`.

<!-- files:begin -->
## Files

- `2026-10-12-zeros-mech-prereg.md` (plans): ZEROS-MECH 预登记：on-policy S …
- `zm_report.md` (results): what the S-family zeros come from
- `zm_read.py` (scripts): what the zeros of the on-policy S-hinge …
- `events_YS-L18-s1.csv` (results): 
- `index.html` (research/zeros-mech): 
- `268.md` (research/decisions): ZEROS-MECH：YS 零分的三个机制假设在已存 18 决策 held …
- `zeros_YS_s1.md` (experiments/onpol_long/results/read): the zero-score scenes of the held read
- `ao_e1.py` (experiments/anchor_offset/scripts): does the served plan's lateral offset …
- `agent_hinge.py` (lib): the plan's footprint must stay out of …
- `on_rows.py` (experiments/onpol_nuplan/scripts): finished on-policy rollouts -> one …

[results/](results/) 5 result files · [figs/](figs/) 4 figures · [plans/](plans/) 1 live plans · [scripts/](scripts/) 1 entry points
<!-- files:end -->
