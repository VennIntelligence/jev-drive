# route_follow: does the served policy use its route

status: live
decisions: (none yet; inputs 237, 245, 268, 282, 283, body1 Amendment 7)
index: route used but weak at junctions (0.56 m at 2 s, 18% of need); lane change unused (0.05 m)

**Question.** At junction turns and lane changes, does the plan of the served checkpoints move at the controller's horizon (1-2 s) when the
route they are given changes, i.e. is the route unused, used too weakly, or used too late?

**Conclusion.** The route reaches every served checkpoint only as a 3-way command from one waypoint ~40 m ahead. Swapping it on YS-L18-s1's
held-452 tokens moves the plan 0.56 m [0.42, 0.63] at 2 s at junction turns (straights 0.04 m), i.e. 18 % of the route's need at 10 m: used but too
weak by the registered line, and the effect sits at 3-4 s (0.10 m at 1 s, 3.9 m at 4 s) beyond the MPC's 1-2 s; at lane changes it is unused
(0.05 m [0.03, 0.15]). Training side: NAVSIM's command tracks the 4 s target (99 % on >= 45 deg turns) and reads S on 85 % of logged lane shifts.
PAI zeros at 60-125 deg turns: straight on 133, late-wide 96, early-inside 24; right turns completed 6 %. Proposals ranked in the note section 3.

**Next.** User's call on R-CF (paired counterfactual junction rows on navtrain, offline G-swap gate before any closed loop).

**Read more.** [research/next-round/route-follow.md](../../research/next-round/route-follow.md) (Chinese; train / serve chain, registration,
reads, proposals). Box: `$DATA_DIR/runs/route_follow/` (cf/, nav.json, read/<ts>/). Inputs reused unedited: `$DATA_DIR/runs/road_probe/tok/{held,zero}`
(served tokens) and `labels/{held,zero}.parquet` (decision 275).

<!-- files:begin -->
## Files

- `route-follow.md` (research/next-round): 路线遵循：策略为什么不按导航走（诊断与方案稿）
- `rf_cf.py` (scripts): ROUTE-FOLLOW counterfactual
- `sh30_driver.py` (experiments/alpasim/lib): SH30 as an AlpaSim nuPlan-track driver
- `pai_driver.py` (experiments/alpasim/lib): The SH30 family as an AlpaSim PAI-track …
- `parity_adapter.py` (lib): Input-parity adapter for openpilot …
- `rp.py` (experiments/road_probe/scripts): is the road geometry of nuPlan city …

[results/](results/) 2 result files · [figs/](figs/) 20 figures · [scripts/](scripts/) 1 entry points
<!-- files:end -->
