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

lr x steps pilot (2026-10-15): 3 x 3 lr-scale {0.03, 0.1, 0.3} x steps {1250, 2500, 5000} on RSG-L18-s0's exact command, held 452 zeros
vs RSG-L18-s0: **flat**, so RSG-L18 stays the base. Best cell (0.03, 5000) reads 23 vs 26, -3.0 [-7.6, +1.0], score +0.014 [+0.003, +0.027]. Short
or weak schedules are worse: (0.1, 1250) +9, (0.03, 2500) +6. Addendum, seed 1 of (0.03, 5000): 21 vs 26 zeros. The 2-seed mean vs RSG-L18 is zeros -4.0 [-9.0, +0.0] and score
+0.013 [+0.003, +0.026]: the score gain replicates, the zero line is not met. Public 1491 / 10 guard (Amendment 2): passes, and it ties RSG-L18 (2-seed
score +0.0005 [-0.0032, +0.0041], zeros +0.0 [-5.5, +5.8], at-fault +0.0, GT -0.012 m). See [results/lrsteps/lrsteps.md](results/lrsteps/lrsteps.md),
prereg `plans/2026-10-15-lrsteps-pilot-prereg.md`, chain `scripts/rf_lrsteps.sh`.

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
- [research/route-follow/late-turns.md](../../research/route-follow/late-turns.md): turn completion by class (late >= 40 m / delayed 20-40 m /
  early / fork) on nuPlan @18 and PAI: late turns barely exist @18 (4 of 1473 scenes; window median 47 m); entry >= 20 m RSG 18 %, RSGboth 68 %,
  RB0 14 % (22), mostly "too slow"; forks no harder than other turns; PAI late 1 / 15 for RSG; `scripts/rf_lateturn.py` (+ `.sh`), `results/lateturn/`.
- [research/route-follow/p1-openloop.md](../../research/route-follow/p1-openloop.md): P1 open-loop read on the Tokyo log-following render (36 exam-hard
  PAI turns): on the human's states RSG already slows with the human and commits on time; RSGboth is not earlier (20 m before the entry +0.43 m/s
  faster, commit identical, lateral -0.06 m); residual = straight-exit turns +0.56 m/s at the entry and 0.7 m outside at 4 s; `scripts/rf_pai_openloop.py --p1`
  (offline replay of any checkpoint on the dumped inputs), `results/pai_openloop/p1_exam48.md`, figures `figs/p1_exam48_speed.png`, `figs/p1_exam48_lat.png`.
- [results/p3_branch](results/p3_branch/README.md): turning school P3. A 3 s curvature / speed push of RSG-L18-s0's own plan at the PAI
  junction entry raises the branch ceiling from 4 to 10 / 12 (passes the 8 line, but 3 of the 6 new scenes complete in a single cell
  and only 2 of 9 survive a 1.5 s push); prereg `plans/2026-10-14-p3-branch-prereg.md`, `scripts/rf_p3.sh`, `onp_log_driver.py` ONP_BRANCH.
- [results/p4_self/smoke.md](results/p4_self/smoke.md): turning school P4 instrument (self-branch search on RSG-L18-s0's own junction
  states, 27 PAI training turns x D 40-0 x 9 cells, `scripts/rf_p4.sh` / `rf_p4.py`, `onp_log_driver.py` ONP_CELLS + `onp_p4.py`); smoke
  S1 / S2 / S3 pass, NRE renderer race (one rollout at a time per launch), box timing and the launch command; prereg
  `plans/2026-10-14-p4-self-branch-prereg.md`, plan `results/p4_self/plan.json`.
- [results/p4_self/pool_dev](results/p4_self/pool_dev/README.md): the P4 row builder `scripts/rf_selfpool.py` (self-branch rollouts ->
  pool `on1_SELF`, gates G-in / route / rmem / road / speed / r1 / shape / crawl) run on the P3 archive: 666 rows, 13 branches, 6 scenes (dev
  only, exam scenes inside); figure `figs/self_bev_dev.png` (red = realized branch demo vs green = the policy's own plan).
- [results/p4_self/train_dry.md](results/p4_self/train_dry.md): the phase 2 training arm RSGSELF-L18 (RSG-L18 + SELF rows as `both` rows in
  `--on-mode mixed`, 30 % of draws, per-scene cap `rm_train --on-cap-scene`), chain `scripts/rf_self_train.sh`, lateral guard + ladder table
  `scripts/rf_selfread.py`; dry run on the dev pool (W, cap shares, mixcheck pass, 3.3 it/s), estimate ~1.5 h, the submit command.
- [results/p4_self/box_det.md](results/p4_self/box_det.md): P4 on the GPU box before the fanout (`scripts/rf_p4det.sh` / `rf_p4det.py`):
  box vs box bit-identical (115 / 115 sessions), box vs Tokyo never (first plan differs at k 0, fixed per host, same score / zero class);
  per-rollout wall 190 / 217.5 / 231.5 s at 1 / ~4.5 / ~7.5 launches per card; host RAM binds before VRAM; disk margin.
- [results/v2gate/](results/v2gate/slot.md): route v2 (prereg `plans/2026-10-15-route-v2-prereg.md`, Amendments 1-3; `scripts/rf_v2.sh`,
  `rf_v2gate.py`): the progress-anchored ordered memory fixes decision 299's slot defect (G-slot: start error median -4.51 m -> 0.00 m on
  53 / 53 turn zeros) at no gate cost; lane-change pairs at weight 10 break dev ADE (+0.035), at weight 1 they pass the guard gates but
  not G-LC (0.10 m vs 0.50); G-swap fails in every arm (no junction counterfactual rows). Amendment 5: weight 3, adapter-only
  gradient and `--rm-latfeat` give G-LC 0.14-0.15 (relaxed line 0.30), weight 10 adapter-only 0.23 but breaks G-real; no lane-change
  setting qualifies. Full round not launched (arms a / b are the coordinator's).
- [results/p4_self/README.md](results/p4_self/README.md): P4 box read, 27 turns / 693 rollouts: identity 9, any completing cell 20, the registered line
  10 / 27 = 37 % (inconclusive), gate-passing 13 (7 at >= 2 handovers); identity fails before the entry on 2 of 27; on1_SELF row count 4 528 (67 branches, 14 turns);
  timing in `timing.md`, per-rollout read `read.json`, secondary numbers `secondary.md` (`scripts/rf_p4_summary.py`).
- [results/p4_self/exam_ceiling/README.md](results/p4_self/exam_ceiling/README.md): P4 exam ceiling read (read-only, Tokyo, `scripts/rf_p4exam.sh`):
  the P4 branch family on the 26 hard_exam turns, 15 definable; identity 2 / 26, any branch cell 12 / 26, coverage definition 2 / 26,
  shape gate 7 / 26 (1 at >= 2 handovers); phase A 15 / 15 bit-identical; 3 launches per 3090 OOMs, 2 per card clean. No rows from it.
- [results/p4_self/round1](results/p4_self/round1/README.md): P4 training round 1 (prereg Amendments 3-4, d297): RSGSELF-L18-s{0,1} = RSG-L18 + 753 self-branch rows (g1.25-raw, D40 / D0, 8 turns, in-junction / exit only) at 30 % of draws; **refuted**: identity completion on the 27 train turns 8 / 9 vs control 9 (L1a fail; RSGCTL-L18-s0 bit-identical to RSG-L18-s0 and reproduces P4 phase A), held turns -4.2 pp [-23.1, +10.7], held zeros +3.5 [-1.2, +7.9], public zeros +11.0 [+2.8, +20.3] (offroad 39.5 vs 30.5; L3 fail), GT lateral pass; `scripts/rf_round1.py`, `rf_self_train.sh stage1 / rest`.
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
- [research/route-follow/side-bias.md](../../research/route-follow/side-bias.md): where RSGboth's left / right bias comes from (verdict: the
  target via the pool's state mix, not tracking, not row balance: RB0's right-turn rows sit outside 4 : 1, so `both` pulls right turns in 0.72 m
  at 2 s vs 0.29 m on left turns; held plan 2 s RSGboth - RSG right -0.33 m, driven - plan <= 0.1 m); `scripts/rf_sidebias.py`,
  `results/sidebias/` (`tables.md`), figure `figs/sidebias.png` (look at the right column: plan and driven overlap; orange below green on right turns).
- [results/route-night/](results/route-night/): held reads (`held_raw.txt`, `a3/held_raw_a3.txt`), `ol_report` tables (`report_RBS.md`,
  `report_RBSG.md`, `a3/report_RSG.md`, `a3/report_RBSGc.md`), route following (`follow_held.md`, `follow_train.md`, `a3/follow_a3.md`),
  G-swap (`gswap/`), G-rely (`rely/`, `a3/rely/`), pool gates (`pool_L18RB0_gates.md`), inside-of-turn counts for all arms (`a3/offroad_a3.md`); round 2 in `r2/`, Amendment 6 in `a6/` (`report_RBSG2.md`, `held_raw_r2.txt`, `offroad_counts.json`).
- [results/vhist/](results/vhist/): JEV_VHIST serving switch on held 460 (hold: 86 % of the shortness removed, +36 zeros, all front collisions; holdlead: score +0.028, zeros -2, rear-ended -20, but GT +0.10 fails line M; zero on RSGallV does not help); prereg `plans/2026-10-13-vhist-prereg.md`.
- [results/vhist/pai_holdlead/](results/vhist/pai_holdlead/README.md): RSG-L18-s0 + holdlead on PAI 281 (prereg Amendment 2, user override): score +0.052 [+0.030, +0.075], front-collision zeros +1, rear-ended -13, turn completion 5 / 36 unchanged: pass; on the 72 exam scenes +0.001 [-0.046, +0.042], zeros +3, GT +0.16 m.
- [results/slow_novelty](results/slow_novelty/README.md): does the residual slowing track state novelty (offset / heading off the log, junction) once the speed-history ratchet is held or controlled for (prereg `plans/2026-10-14-slow-novelty-prereg.md`, d294): no, "pure input loop": the novelty block adds 0.11 % of variance (registered line 5 %), only the offset is clear of 0 (-0.041 SD, about 5 cm of plan per 0.57 m), heading error goes the other way; onset: offset rises with the slowing ramp, not ahead of it; `scripts/rf_slownov.py` (+ `.sh`), figures `figs/slow_novelty_bins.png` (look at the middle column: longer plans at larger heading error) and `figs/slow_novelty_onset.png` (look at the first three panels: the ramp starts while the fed trend is still positive).
- [results/entry_speed_rows](results/entry_speed_rows/README.md): are the hinge rows of fast turn entries waste (prereg `plans/2026-10-14-entry-speed-rows-prereg.md`, d295)? **Flat, speed cap dropped**: on 304 nuPlan @18 turn cells (6 row pools) the usable share of hinge-active rows is 0.36 in the slowest and the fastest quartile of v_entry / v_human (D +0.000 [-0.112, +0.110], line +0.20); P3's designed slowing gives +0.22 (half of it rows after rear-ends), but the slowest nuPlan quartile completes 0.14 of turns against 0.52 (floor ~0.86 x the human's entry speed); `scripts/rf_entryspeed.py` (+ `.sh`), figures `figs/entry_speed_rows_bins.png` (look at the flat left panel against the rising completion) and `figs/entry_speed_rows_cells.png`.
- [results/turn_gallery/](results/turn_gallery/plan_strict36.json): Tokyo desktop video gallery `~/Desktop/turn_gallery/index.html` of RSG-L18-s0's 70 PAI 281 zeros (A route-following corridor zeros 30, B other zeros 40) plus the strict turns completed at 1.0 with a straight alternative (C, 1): HUMAN / RSG / best completing P4 cell per row, PUSH / SELF overlays (decision k = clip time (k + 1) x 0.1 s); `scripts/turn_gallery.py` (ffmpeg only, no simulation), `plan_strict36.json` = `rf_p4.py plan` of the 36 strict turns (topo, entry, human entry speed).
- [results/turn_gallery/p4_fill.md](results/turn_gallery/p4_fill.md): P4 camera fill of the gallery (prereg Amendment 5, read-only, NO_ROWS): the 17 clip-less rows with a strict turn (5 train, 12 exam) got identity + g1 / g1.25 / g1.5 raw at D40 / D0 (16 box + 59 Tokyo rollouts, all re-runs bit-identical, 12 / 12 identities = r281/rsg); 22 / 71 rows now carry a P4 cell, 10 completing; 49 rows have no strict turn.
- [results/progress/progress.md](results/progress/progress.md): reading A (prereg `plans/2026-10-15-progress-lane-readings-prereg.md`,
  read-only): are the over-turn / shortcut turn zeros of RSG-L18 the ones whose route memory had no near route (0-20 m) 3 s before the
  failure? **Unsupported**: pooled over PAI 281 s0 and held 460 s0 / s1 (72 turn zeros) RR 0.72 [0.22, 2.29] (T 12 / 53 uncovered vs 6 / 19
  covered; geometric coverage, Amendment 2: 0.67 [0.20, 2.09]); on PAI the 3 over-turn zeros and 3 of the shortcut flags all had the near route
  (0 / 21 uncovered vs 6 / 19 covered), held turn zeros are all uncovered (they sit in the first 40 m of an 18-decision rollout). Completed strict
  turns are even more often uncovered at their entry (97 %) than turn zeros at t_f - 3 s (74 %): the first-40 m gap does not separate failures
  from completions. Side finding: when the ego stands before the memory's start and the route turns in between, `rmem.window` serves the
  far points in nearer slots (median 4.5 m, p90 15.7 m too near; de9db5dc 14 vs 41 m). `scripts/rf_progress.py` (`rmem` extractor + `report`),
  figure `figs/progress_bev.png` (look at the red rings: the near route the memory lacked at t_f - 3 s, and whether the orange path leaves the
  route where the rings are; in 2466266e and 382a7247 the memory is full and the car still leaves).
- [results/laneoff/laneoff.md](results/laneoff/laneoff.md): reading B (same prereg): do failed turns sit off the route's lane before the turn,
  and does the car make the lane changes its route asks for? **Inconclusive** at the registered point (PAI strict, entry - 20 m: median
  |offset| failed 0.48 vs completed 0.06 m, D +0.42 [+0.08, +0.81] (line 0.5), out-of-lane share +0.16 [+0.00, +0.33]; only 3 completed
  turns have the point, held turns start 3-19 m before the entry); at the entry pooled 44 / 33 turns it is unsupported (D -0.08 [-0.64, +0.31]).
  Exploratory, signed toward the turn: failed PAI turns sit on the side away from the turn at entry - 20 m (-0.48 vs +0.06 m, 7 / 19 >= 0.75 m,
  3 >= 1.75 m, completed 0 / 3), and 7 / 26 failed vs 0 / 5 completed PAI turns touch an oncoming lane. Lane changes: the registered geometric
  proxy is dominated by road geometry (Amendment 3); on the map-validated sites (route leaves its lane chain for an adjacent lane, human moved
  2-6 m) the ego reached 4 PAI + 4 held sites and shifted >= 2 m on 1 / 4 and 0 / 4 (in the route's lane after 2 / 4 each). `scripts/rf_laneoff.py`
  (`lanes` extractor + `report`), figure `figs/laneoff_bev.png` (look at the titles: offset / heading error at the triangle (entry - 20 m)
  and the square (entry); the failed turns start their turn from the lane or line away from the turn).
- [results/pai281/RV2a-L18-s0.md](results/pai281/RV2a-L18-s0.md): route v2 arm a (ordered memory only) on PAI 281 vs RSG-L18-s0, gains 20
  (read of record): +0.010 [-0.007, +0.028], zeros 67 vs 70 (offroad 16 vs 14, corridor 41 vs 47), spins 2 vs 2; exam 72 +0.030
  [-0.005, +0.078], zeros 36 vs 38; strict turns 5 / 36 both (gained e848c843, ea468790; lost 132ae6cb, 382a7247), shape gate 3 vs 1.
  Look at the all281 / exam72 rows and the per-scene table.
- [results/pai281/RV2a-L18-s0_unreliable.md](results/pai281/RV2a-L18-s0_unreliable.md) and
  [RV2a_spins_unreliable.md](results/pai281/RV2a_spins_unreliable.md): the first RV2a-L18-s0 PAI 281 read and its spin reading.
  **Unreliable**: the run drove at the default controller gains, not gains 20 (route v2 prereg Amendment 18); its -0.035 / 87 zeros /
  13 spins are not a comparison with r281/rsg. Kept for the record; the gains-20 reread replaces it.
- [results/pai281/RV2b-L18-s1.md](results/pai281/RV2b-L18-s1.md): route v2 arm b (ordered memory + lane hinge) on PAI 281 vs RSG-L18-s0
  (gains 20): +0.010 [-0.012, +0.031], zeros 67 vs 70, spins 2 vs 2; exam 72 +0.023 [-0.025, +0.071], zeros 35 vs 38; strict turns 7 / 36
  vs 5 / 36, shape gate 4 vs 1. Look at the all281 / exam72 rows and the per-scene table.
- [results/turn_gain/g1.25.md](results/turn_gain/g1.25.md): TURN-GAIN serve-side switch (diagnostic, not adopted; prereg
  `plans/2026-10-15-turn-gain-switch-prereg.md`, Amendment 1): RSG-L18-s0 + `JEV_TURN_GAIN=1.25` (plan yaw / course x1.25 whenever its 2 s turn
  is < 60 % of what its own route memory shows, from the policy's own states, no handover) on the 36 strict PAI turns + the 26 exam turn
  scenes (47). **Refuted** (turn magnitude is not the bottleneck): strict 6 / 36 vs 5 / 36 (gained e0f6ebe1 only, which fails the shape gate),
  exam turn scenes 3 / 26 vs 2 / 26, shape gate 2 vs 1; the switch fired on 30 / 36 strict scenes, mostly 20-40 m before the entry, and the
  outcome class changed in 3 scenes. No new zero (31 -> 28 on the 47, mean +0.059 [+0.000, +0.128]: 68e53160, a1b166cc, cdf95fd1 corridor
  zeros became scored off-route finishes). No g = 1.5 run (needs >= 3 more). Look at the per-scene table: failures stay where they were
  (corridor / offroad 0-20 m past the entry, several entering at 10-14 m/s). `scripts/rf_v2pai.py` (`tgcheck`, `turns`, `table`),
  `scripts/rf_v2pai.sh tg`, switch `experiments/alpasim/lib/turn_gain.py`.
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
