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

Day-time addendum (2026-10-13, decision 286): +45 % rows (RSGall) give no public gain (zeros +3); the speed-history drop (RSGallV)
removes ~75 % of the plan shortness but costs public zeros (+8.5) and GT (+0.056 m); the stop-line arm RSGB fails (B active on ~0.5 % of
rows, P dominates, corridor 7.5 -> 12.5, GT +0.39 m, route reliance lost); holding the fed speed history (VHIST) removes 86 % of the
shortness, so the slowness ratchet runs through that input, and holdlead reads held 0.894 (+0.028), 32 zeros, rear-ended -20 but GT +0.10
(line M fails on GT); a learned plan critic fails all three lines (AUC 0.846, R^2 0.445, flip 0.41; one plan per state); round 3 RBSG3
reads 53 public zeros (damped iteration toward RSG); with the memory fed empty in closed loop RSG loses 24 zeros and RBSG2 17, mostly
corridor, so the memory is relied on in closed loop and offline G-rely does not predict it; RSG's PAI gap to 0.94 is 36.8 points
(slow 13.5, route-following corridor 10.7, drove past the human's stop 6.0; research/zero-taxonomy.md); PAI exam split v1 hard 131 /
control 75 / trainable 75; RBSG2-L18-s0 on PAI 281 0.599 / 65, the best PAI mean, though excluded on public nuPlan.

**Next.** Turning-school reads P0-P2 (`research/turning-school-checks.md`, to appear); the holdlead PAI exam read (coordinator exception);
spin-fix lateral bound v2; Tokyo PAI 281: RB0-F-s0 (same-gains base reference), RBSGc-PAI.

**Read more.** [research/decisions/286.md](../../research/decisions/286.md) (Chinese; full table, six line verdicts, branches, card-hours);
human page [research/route-follow/index.html](../../research/route-follow/index.html) (four figures copied to research/route-follow/figs/); task book
[research/next-round/route-night.md](../../research/next-round/route-night.md); registration
[plans/2026-10-13-route-night-prereg.md](plans/2026-10-13-route-night-prereg.md) (Amendments 1-5); diagnosis
[research/next-round/route-follow.md](../../research/next-round/route-follow.md).

- [research/zero-taxonomy.md](../../research/zero-taxonomy.md): every zero / low score of RSG-L18-s0 on PAI 281 and held 460 by mechanism
  (`scripts/rf_tax.py`, per-scene rows in `results/taxonomy/`, figures `figs/tax_*.png`, review sheet `research/route-follow/review-sheet-rsg.md`).
- [research/route-follow/pai-follow.md](../../research/route-follow/pai-follow.md): PAI 281 turn completion with the real route (RSG 13.9 %,
  RBSG2 25.0 % of 36 turns) and the map topology of every corridor zero (14 / 22 route zeros have a straight road ahead; the ego never slows
  into the entry); `scripts/rf_pai_follow.py` (+ `.sh`, Tokyo container), `results/pai_follow/`, figure `figs/pai_follow_zeros.png`.
- [research/route-follow-offroad.md](../../research/route-follow-offroad.md): why route-hinge arms go offroad; look at table 1's
  "right-turn inside" column (17-21 vs 1-2) and the H1-H5 verdicts (labels on road, memory not drifted, over-rotation in small right turns).
- [research/route-follow/loss-audit.md](../../research/route-follow/loss-audit.md): loss audit of the turn objective (verdict: objective, not
  representation; the route hinge's lane-centre label prefers wide / inside-cut variants over the human, the S hinges tie, nothing on on-policy
  rows rewards slowing; the frozen-token head fits the human turns to 0.32 m); `scripts/rf_loss_audit.py`, `results/loss_audit/`, figures
  `figs/loss_audit_variants_moving.png` (look at J_S: H ties with no-slow and inside-cut) and `figs/loss_audit_grad_moving.png` (route: outward
  before the entry, inward inside right turns; only imitation has an along-heading push).
- [research/route-follow/rsgboth-ledger.md](../../research/route-follow/rsgboth-ledger.md): RSGboth vs RSG solved / unsolved ledger on held 460
  (slowness solved, turn shape not: left turns late / wide, right turns early / inside; the lane-centre "inside cut" is mostly the human's own line);
  `scripts/rf_ledger.py`, `results/ledger/`, figures `figs/ledger_bars.png` (look at "slow" and the turn-zero outside bar) and
  `figs/ledger_turn_shape.png` (look at the orange lines: outward after u = 0.5 on left turns, inward on right turns).
- [results/route-night/](results/route-night/): held reads (`held_raw.txt`, `a3/held_raw_a3.txt`), `ol_report` tables (`report_RBS.md`,
  `report_RBSG.md`, `a3/report_RSG.md`, `a3/report_RBSGc.md`), route following (`follow_held.md`, `follow_train.md`, `a3/follow_a3.md`),
  G-swap (`gswap/`), G-rely (`rely/`, `a3/rely/`), pool gates (`pool_L18RB0_gates.md`), inside-of-turn counts for all arms (`a3/offroad_a3.md`); round 2 in `r2/`, Amendment 6 in `a6/` (`report_RBSG2.md`, `held_raw_r2.txt`, `offroad_counts.json`).
- [results/vhist/](results/vhist/): JEV_VHIST serving switch on held 460 (hold: 86 % of the shortness removed, +36 zeros, all front collisions; holdlead: score +0.028, zeros -2, rear-ended -20, but GT +0.10 fails line M; zero on RSGallV does not help); prereg `plans/2026-10-13-vhist-prereg.md`.
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
