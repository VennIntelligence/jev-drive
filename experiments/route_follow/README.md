# route_follow: does the served policy use its route

status: live
decisions: 284, 286
index: fresh route-memory base relied on; RSG-L18 held zeros 34 vs base 59; route hinge cuts right turns

**Question.** At junction turns and lane changes, does the plan of the served checkpoints move at the controller's horizon (1-2 s) when the
route they are given changes; and (route-night, 2026-10-13) does a base retrained from the shipped weights with the route memory as input and
the route itself as a target follow it?

**Conclusion.** Diagnosis (decision 284): the route reaches every served checkpoint only as a 3-way command from one waypoint ~40 m ahead; a
command swap moves the plan 0.56 m at 2 s at junctions (18 % of need), at 3-4 s rather than 1-2 s, and lane changes not at all. Route-night
(decision 286): the fresh base RB0-F (YR10m10 recipe + route-memory branch) is the first checkpoint that relies on the route memory offline
(junction dev ADE +0.025 / +0.022 m when masked; d282 +0.0004 / +0.0011) and reads 59 / 59 held-460 zeros (YR10m10 82 / 79); one gentle
S-only on-policy round on it, RSG-L18, reads 34 / 34 zeros, 0.866 / 0.868, GT 1.14 m and passes L3 (zeros -25 [-37, -14] vs RB0), the lowest
18-decision held zeros so far. The route hinge as trained tonight is not adopted: every arm with it gains right-turn-inside offroad zeros
(17-21 vs 1-2; RSG 1 / 1, hinge on cov20 rows only 5 / 10) and the registered arm RBS-L18 reads 122 / 99 zeros; counterfactual rows (RB-F)
teach route switching at junctions (G-swap 55 % of need, 94 % on the counterfactual side) but not lane changes (5-7 %), and cost G-real.

**Next.** Round 2 (prereg Amendments 4-5: RBSG2 with RBSG's own over-turning states; RS2G / RBS2Gc from RSG rollouts) is running, results
~07:00 JST; Tokyo: RB0-F-s0 PAI re-roll with gains 20, RSG-L18-s0 PAI 281 read queued. Before RSG-L18 is called a base: paired CI against
YS2G-L18, the public 1491 / 10-decision guard, PAI 281.

**Read more.** [research/decisions/286.md](../../research/decisions/286.md) (Chinese; full table, six line verdicts, branches, card-hours);
human page [research/route-follow/route-night.html](../../research/route-follow/route-night.html); task book
[research/next-round/route-night.md](../../research/next-round/route-night.md); registration
[plans/2026-10-13-route-night-prereg.md](plans/2026-10-13-route-night-prereg.md) (Amendments 1-5); diagnosis
[research/next-round/route-follow.md](../../research/next-round/route-follow.md).

- [research/route-follow-offroad.md](../../research/route-follow-offroad.md): why route-hinge arms go offroad; look at table 1's
  "right-turn inside" column (17-21 vs 1-2) and the H1-H5 verdicts (labels on road, memory not drifted, over-rotation in small right turns).
- [results/route-night/](results/route-night/): held reads (`held_raw.txt`, `a3/held_raw_a3.txt`), `ol_report` tables (`report_RBS.md`,
  `report_RBSG.md`, `a3/report_RSG.md`, `a3/report_RBSGc.md`), route following (`follow_held.md`, `follow_train.md`, `a3/follow_a3.md`),
  G-swap (`gswap/`), G-rely (`rely/`, `a3/rely/`), pool gates (`pool_L18RB0_gates.md`), inside-of-turn counts for all arms (`a3/offroad_a3.md`).
- [results/cf_gates.json](results/cf_gates.json): counterfactual pool on1_CF size (14 765 train rows) and G-cf1 / G-cf2 gates.
- [figs/cf_bev_0.png](figs/cf_bev_0.png) - [cf_bev_5.png](figs/cf_bev_5.png) (G-cf3): one counterfactual row each (0-2 junction, 3-5 lane
  change); look at whether the red counterfactual label stays inside the grey road and leaves at a different exit (junction) or lane (lane
  change) than the blue real route.
- [figs/offroad_1.png](figs/offroad_1.png) - [offroad_6.png](figs/offroad_6.png): new offroad zeros of RBSG-L18 that the base does not have;
  look at where the orange served plan starts bending toward the turn centre and on which side of the dashed blue label the car leaves the
  road (inside of a right turn in 1-5; the label itself always lies in the grey).

Box: `$DATA_DIR/runs/route_mem/` (feat/, read/, train-*), `$DATA_DIR/runs/route_follow/` (report/, offroad/, offroad_a3/, cf/, nav.json),
pools `$DATA_DIR/runs/op_parity/cache/on1_{CF,CFdev,L18RB0,L18RB0c}`. Inputs reused unedited: `$DATA_DIR/runs/road_probe/tok/{held,zero}`
and `labels/{held,zero}.parquet` (decision 275).

<!-- files:begin -->
## Files

- `2026-10-13-route-night-prereg.md` (plans): route-night, 2026-10-13 night
- `286.md` (research/decisions): ROUTE-NIGHT：从出厂权重重训的路线记忆底座 RB0-F …
- `route-night.md` (research/next-round): 今晚的任务：跟着路线 waypoint 开的新底座（2026-10-13 夜）
- `route-follow-offroad.md` (research): 路线 hinge 臂的 offroad 零分诊断（2026-10-13 …
- `on_route.py` (experiments/onpol_nuplan/lib): the lateral distance of the student's …
- `rf_cfpool.py` (scripts): ROUTE-NIGHT counterfactual pool `on1_CF`
- `rf_follow.py` (scripts): ROUTE-FOLLOW closed-loop read
- `rf_gates.py` (scripts): does a ROUTE-MEM checkpoint's plan …
- `rf_offroad.py` (scripts): ROUTE-FOLLOW offroad diagnosis
- `rf_pai_pool.py` (scripts): Route-night section 9 step 5
- `rm_train.py` (experiments/route_mem/scripts): onpol_nuplan's on_train.py `train` (its …
- `route-follow.md` (research/next-round): 路线遵循：策略为什么不按导航走（诊断与方案稿）
- `rf_cf.py` (scripts): ROUTE-FOLLOW counterfactual
- `pai_driver.py` (experiments/alpasim/lib): The SH30 family as an AlpaSim PAI-track …

[results/](results/) 44 result files · [figs/](figs/) 32 figures · [plans/](plans/) 1 live plans · [scripts/](scripts/) 6 entry points
<!-- files:end -->
