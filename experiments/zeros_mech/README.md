# zeros_mech: mechanisms behind the S-family zeros

status: live
decisions: (pending)
index: three mechanism reads on YS-L18 zeros, pre-registered; no read yet

**Question.** On the stored 18-decision held rollouts (452 scenes, YS-L18 / YS2G-L18 / YNONE-L18 / YR10m10-F, two seeds each), do the
zeros of the on-policy S-hinge checkpoints come from turn timing and the route command (H1), a sideways cut-around-the-lead reflex taught by
the shape-only agent hinge (H2), or driving in the oncoming lane that the direction-free road raster does not penalise (H3)?

**Conclusion.** Pending (pre-registered 2026-10-12, before any read).

**Next.** Run the registered reads (CPU, box), then the decision entry and the HTML page.

**Read more.** [plans/2026-10-12-zeros-mech-prereg.md](plans/2026-10-12-zeros-mech-prereg.md) (Chinese, written before any read);
the clips the hypotheses come from: [research/onpol-long/zeros-ys-l18.html](../../research/onpol-long/zeros-ys-l18.html).

<!-- files:begin -->
## Files

- `2026-10-12-zeros-mech-prereg.md` (plans): ZEROS-MECH 预登记：on-policy S …
- `zeros_YS_s1.md` (experiments/onpol_long/results/read): the zero-score scenes of the held read
- `ao_e1.py` (experiments/anchor_offset/scripts): does the served plan's lateral offset …
- `long_read.py` (experiments/alpasim_long/scripts): per-scene and per-decision metrics of …
- `ol_clips.py` (experiments/onpol_long/scripts): ONPOL-LONG zero-score clips of one …
- `agent_hinge.py` (lib): the plan's footprint must stay out of …
- `on_rows.py` (experiments/onpol_nuplan/scripts): finished on-policy rollouts -> one …
- `ap2_inputs.py` (experiments/alpasim/lib): The AlpaSim nuPlan-track input standard …

[results/](results/) 0 result files · [figs/](figs/) 0 figures · [plans/](plans/) 1 live plans · [scripts/](scripts/) 0 entry points
<!-- files:end -->
