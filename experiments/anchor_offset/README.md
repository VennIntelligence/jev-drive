# anchor_offset: near-anchor plan offset as a drift warning

status: live
decisions: pending
index: E1 pre-registered: near-anchor offset as a drift warning, no read yet

**Question.** On the stored 18-decision held rollouts, does the served plan's lateral offset from the route at a near anchor
(5 / 10 / 20 m ahead) warn of corridor and offroad zeros early and cleanly enough to leave room for a serving near-anchor correction (E2)
or a short-range anchor label in training (E3)? Motivated by Anchored Planning (arXiv 2609.30036).

**Conclusion.** Pending.

**Next.** Run E1 as registered (CPU only), then the decision row.

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
