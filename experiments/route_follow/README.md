# route_follow: does the served policy use its route

status: live
decisions: 284, 286
index: route-memory base relied on; RSG-L18 held zeros 34 vs 59, public ties YS2G

**Question.** At junction turns and lane changes, does the plan of the served checkpoints move at the controller's horizon (1-2 s) when the
route they are given changes; and (route-night, 2026-10-13) does a base retrained from the shipped weights with the route memory as input and
the route itself as a target follow it?

**Conclusion.** Diagnosis (decision 284): the route reaches every served checkpoint only as a 3-way command from one waypoint ~40 m ahead; a
command swap moves the plan 0.56 m at 2 s at junctions (18 % of need), at 3-4 s rather than 1-2 s, and lane changes not at all. Route-night
(decision 286): the fresh base RB0-F (YR10m10 recipe + route-memory branch) is the first checkpoint that relies on the route memory offline
(junction dev ADE +0.025 / +0.022 m when masked; d282 +0.0004 / +0.0011) and reads 59 / 59 held-460 zeros (YR10m10 82 / 79). The one
candidate that stands is RSG-L18 (one gentle S-only on-policy round on RB0-F): held 34 / 34 zeros, L3 pass (-25 [-37, -14] vs RB0),
beats YS2G-L18 paired on held 452 (zeros -15.5 [-24.9, -5.3]); on the public 1491 / 10 decisions it passes the guardrail (+0.016 over
YR10m10-F-s0) and ties YS-L18 / YS2G on score and zeros with GT 0.09-0.11 m closer. The route hinge as trained tonight is not adopted:
with rows from the base's own states it makes the car cut inside right turns (right-turn-inside offroad zeros 17-21 vs 1-2; RBS-L18
122 / 99 zeros); adding the hinge policy's own over-turning states (RBSG2-L18) fixes this on held (4 / 4, 37 / 37 zeros, corridor 3 / 4)
but the offroad cost returns on the public set (77.5 zeros, at-fault 69; vs RSG +32.5 [+20.8, +45.5]); an earlier version of this
README called RBSG2 the route-following candidate, withdrawn. Amendment 5: S-only rows from RSG's rollouts (RS2G) add nothing over RSG,
the cov20-only hinge (RBS2Gc) is worse. Counterfactual rows (RB-F) teach route switching at junctions (G-swap 55 % of need, 94 % on the
counterfactual side) but not lane changes (5-7 %), and cost G-real.

PAI 281 (Tokyo, controller gain rel_front_steering_angle_weight 20 after expert B's pilot passed): RSG-L18-s0 0.572 / 70 zeros, the best
stored PAI read (YS-L18-s0 0.517 / 96, paired +0.056 [+0.019, +0.093]; YR10m10-F-s0 +0.086), but the references ran with gains 5 (the pilot
gives ~19 spins and ~3 zeros of a YS-L18 read to the gains); same-gains references are queued. RSG-PAI-s0 (the same gentle S-only
round on RB0-F-s0's own PAI rows) reads 0.565 / 77, level with RSG-L18 (-0.007 [-0.034, +0.021]): RSG-L18 is the single candidate on
both tracks (`results/pai281/`).

**Next.** Tokyo PAI 281 queue: RBSG2-L18-s0 (~11:25 JST, mechanism), RB0-F-s0 (~13:35, same-gains base reference), RBSGc-PAI (~15:45);
a YS-L18-s0 reference at gains 20 is still to be run. The mechanism
lead (the hinge needs the policy's own over-turning states) is for a next round, with a public-set read before any held claim (held 452
could not separate RBSG2 from RSG; the public set separates them by 32.5 zeros).

**Read more.** [research/decisions/286.md](../../research/decisions/286.md) (Chinese; full table, six line verdicts, branches, card-hours);
human page [research/route-follow/index.html](../../research/route-follow/index.html) (four figures copied to research/route-follow/figs/); task book
[research/next-round/route-night.md](../../research/next-round/route-night.md); registration
[plans/2026-10-13-route-night-prereg.md](plans/2026-10-13-route-night-prereg.md) (Amendments 1-5); diagnosis
[research/next-round/route-follow.md](../../research/next-round/route-follow.md).

- [research/route-follow-offroad.md](../../research/route-follow-offroad.md): why route-hinge arms go offroad; look at table 1's
  "right-turn inside" column (17-21 vs 1-2) and the H1-H5 verdicts (labels on road, memory not drifted, over-rotation in small right turns).
- [results/route-night/](results/route-night/): held reads (`held_raw.txt`, `a3/held_raw_a3.txt`), `ol_report` tables (`report_RBS.md`,
  `report_RBSG.md`, `a3/report_RSG.md`, `a3/report_RBSGc.md`), route following (`follow_held.md`, `follow_train.md`, `a3/follow_a3.md`),
  G-swap (`gswap/`), G-rely (`rely/`, `a3/rely/`), pool gates (`pool_L18RB0_gates.md`), inside-of-turn counts for all arms (`a3/offroad_a3.md`); round 2 in `r2/` (`report_RBSG2.md`, `held_raw_r2.txt`, `offroad_counts.json`).
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

[results/](results/) 84 result files · [figs/](figs/) 32 figures · [plans/](plans/) 1 live plans · [scripts/](scripts/) 7 entry points
<!-- files:end -->
