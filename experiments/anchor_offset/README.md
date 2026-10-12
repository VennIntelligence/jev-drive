# anchor_offset: near-anchor plan offset as a drift warning

status: live
decisions: 266
index: E1: plan offset 10 m ahead warns corridor zeros AUC 0.953, 70% >= 1.5 s; E2 closed

**Question.** On the stored 18-decision held rollouts, does the served plan's lateral offset from the route at a near anchor
(5 / 10 / 20 m ahead) warn of corridor and offroad zeros early and cleanly enough to leave room for a serving near-anchor correction (E2)
or a short-range anchor label in training (E3)? Motivated by Anchored Planning (arXiv 2609.30036).

**Conclusion.** The route is rebuilt exactly from AlpaSim's own `RouteGeneratorMap` (99.89% of 48 721 decisions within 0.05 m of
the logged first waypoint), so no re-roll was needed. On 452 held scenes x 6 Y-family labels, the served plan's offset from the route 10 m
ahead warns of corridor zeros with AUC 0.953 [0.923, 0.982] and 70% of them >= 1.5 s ahead at 5% clean false alarms (current distance to the
log: AUC 0.940, 1%); offroad only 0.757 / 25%. By the registered lines E2 (serving near-anchor correction) is closed (AUC margin over the
log-distance baseline 0.013 < 0.05) and E3 (short-range anchor label) has room at N = 5 m (70.5% of pre-zero plans do not shrink a >= 1 m
offset vs 47.3% on clean rollouts) (decision 266).

**Next.** E3 (short-range anchor hinge on on-policy rows) if the coordinator schedules it; reopening E2 needs a new registration with lead time as the main quantity.

**Read more.** [plans/2026-10-12-e1-prereg.md](plans/2026-10-12-e1-prereg.md) (Chinese, written before any read);
[research/literature/anchored-planning-2026.md](../../research/literature/anchored-planning-2026.md).

<!-- files:begin -->
## Files

- `2026-10-12-e1-prereg.md` (plans): E1 预登记：近锚点偏差能否提前预警漂移零分（AlpaSim nuPlan …
- `anchored-planning-2026.md` (research/literature): 文献笔记：Aim Short to Reach Far（Anchored …
- `ol_mech.py` (experiments/onpol_long/scripts): ONPOL-LONG Amendment 3, descriptive …
- `long_read.py` (experiments/alpasim_long/scripts): per-scene and per-decision metrics of …

[results/](results/) 0 result files · [plans/](plans/) 1 live plans · [scripts/](scripts/) 0 entry points
<!-- files:end -->
