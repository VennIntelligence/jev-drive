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

ON-K pilot (2026-10-15, decision 311): on-policy rows per batch on RSG-L18-s0's exact command, never varied before (13 = round(0.1 x 128)).
Arms are `--on-k` 26 and 51 at ho-w 3 (hinge pressure 2.2x and 5.8x of k13, because n_imit shrinks as k grows) and 51 at ho-w 1 (1.9x). Held 452
zeros vs RSG-L18-s0, seed 0: k26 -4.0 [-9.6, +1.2], k51 -1.0 [-6.4, +4.1], k51w1 -3.0 [-7.7, +1.1]. Score: k51w1 +0.0130 [+0.0037, +0.0240] was a
candidate on the score clause, the others' CIs reached 0; the 5.8x arm was the weakest (turn completion -8 pp). Seed 1 (vs RSG-L18-s1) did not repeat the
size: zeros -1.0 / -2.0 / -4.0 and score +0.0025 / +0.0045 / +0.0104, all CIs reaching 0. Corrected on 2 seeds (2-seed mean minus RSG-L18 2-seed mean):
k26 zeros -2.5 [-6.2, +0.5], score +0.0063 [-0.0001, +0.0140] (seed-0 gain mostly not replicated, no candidate); k51 -1.5 [-7.2, +4.7], +0.0063
[-0.0067, +0.0183] (no gain); k51w1 -3.5 [-8.8, +1.8], **+0.0117 [+0.0015, +0.0222]** (keeps the score clause, narrowly; zero clause not met). Direction
replicates in sign for all arms, and the 5.8x arm stays the weakest on score in both seeds. Open: the public 1491 / 10-decision guard read for the 2
seeds of k51w1 is not run (lrsteps' held gain also replicated and then tied RSG-L18 on public 1491). See [results/onk/onk.md](results/onk/onk.md), prereg `plans/2026-10-15-onk-pilot-prereg.md`, chain `scripts/rf_onk.sh`.

RECOVERY-EXPERT REWIND (2026-10-16, decision 314): no training. The box replays RSG-L18-s0's own failing PAI strict-turn sessions bit-identically up to 1 / 2 / 3 s before the failure, and from there the rule expert XP-L12-E46 drives. The expert takes over the policy's pose, speed and controller state; the rewind check passes on 80 / 80 sessions. Path JEV_TRACK 0. On the 27 failures (26 at 3 s) the expert completes 23 / 22 / 22, i.e. 85 / 81 / 85 %, against 0 for the policy. That passes the registered lines at every Delta. No creep: 92-100 % of rollouts keep >= 2 m/s 3 s after the handover. The d293 shape gate passes on 7 / 23, 15 / 22 and 16 / 22 of the completed paths, so Delta 1 s recoveries are hard pulls (lateral acceleration up to 8 m/s²). Caveats: the expert from the start already completes 24 of the 27, and at 2-3 s about half the handovers are before the entry. See [results/rewind/rewind.md](results/rewind/rewind.md), figure [figs/rewind_bev.png](figs/rewind_bev.png), prereg `plans/2026-10-16-recovery-rewind-prereg.md`.

RECOVERY-REWIND train 64 (2026-10-16, decision 316): the same rewind on the 64 strict turns of the nuPlan @18 train pool. Serving is the held read's sh30 path; the rule expert takes over the policy's own session from a per-scene decision (`JEV_EXPERT_FROM`). nuPlan sessions are not bit-identical across launches, because the renderer sometimes sends different frames. So the handover check is 'within the identity's run-to-run spread': 115 / 123 sessions pass, 104 of them bit-identical. Every session runs 30 decisions. On the 42 failed turns, the expert handed over 2 s before the stored failure completes 33 / 41 = 0.80 [0.66, 0.93], against 10 / 41 for the policy at the same length (+0.56). It recovers every off unit and 16 / 22 zero-event units. The short units are not turn errors: with 15 s the policy completes 9 / 10 of them itself. The d293 gate passes on 7 / 30, 15 / 33 and 23 / 31 completions at 1 / 2 / 3 s. Rows in the turn window after the handover: 345 at 2 s, of which 122 are in completed and gated units (230 at 3 s). The B stratum (completed but violating the gate) is nearly empty: 4 / 21 violate, only 1 can be handed over. The expert from the start completes 0.76, below its 0.8 line. The PAI-derived gate is too strict on entry speed for nuPlan humans: only 37 / 63 human turns pass it. See [results/rewind_train64/rewind_train64.md](results/rewind_train64/rewind_train64.md), figures [figs/rewind_train64_A.png](figs/rewind_train64_A.png) and [figs/rewind_train64_B.png](figs/rewind_train64_B.png), prereg `plans/2026-10-16-rewind-train64-prereg.md`.

ROUTE-DOMINANCE (2026-10-16, decision 318): no training, open loop. The question is whether the route input pushes RSG-L18-s0's plans into the curbs and cars of its failing turns. At its own pool states 0.5-3 s before a turn zero (held 108 + train 362 states), served route memory is compared with zeroed memory and with the other exit's route. The read is BLIND. 65 % of the served plans already contact the road edge or an agent box. Zeroing the route frees only 5.6 % [3.0, 8.6] of them (301 states, 45 logs; BLIND in held, train and all three failure classes), against 12.8 % on completed turns, and it pushes 10.6 % of contact-free failing plans into contact. The route moves the 1-2 s plan by 3 cm. APP1X-L18-s0's weights at the same states are also BLIND (9.5 %); its route pulls plans away from contact. APP1X's own failing states were not dumped and cannot be read. See [results/route_dominance/route_dominance.md](results/route_dominance/route_dominance.md), figure [figs/route_dominance.png](figs/route_dominance.png), prereg `plans/2026-10-16-route-dominance-prereg.md`.

SELF-REWIND (2026-10-16, decision 317): does the policy have an AlphaNIST-style frontier on its own failures? Same instrument as decision 316 (30-decision sh30 sessions, after280 cache), but the car goes back to RSG-L18-s0 itself. On the 32 failed train turns (event / off) and 10 completed controls, the policy drives to h = k_ref - 2 Delta (Delta 3 / 5 / 8 s). From h, for 2 decisions, the served plan is perturbed by action (`JEV_PERTURB`, perturb.py: curvature gain g 0.7-1.3 x arc scale s 0.96 / 1.04). After that the policy's raw plan is served again. 8 variants per (unit, Delta). The switch off is bit-identical to d316's stored identity, and every perturbed session is identical before h. Displacement at the window end: lateral median 2-10 cm, speed ratio 0.92 / 1.07; 439 / 512 variants are inside the bound. Registered verdict: UNDECIDED. At Delta 3 s exactly 8 failed units have p-hat in [0.2, 0.8], but the controls reach only 0.55 [0.25, 0.89] (line 0.7). Mean p-hat on failed units is 0.12 / 0.17 / 0.17. Units with p-hat > 0: 17 / 32, all of them units the expert also recovers. The completions are not due to slowing (12 / 22 come from s < 1). Strict self rows: 757 (623 pass the gate), 57 % from 3 turns. See [results/rewind_self/rewind_self.md](results/rewind_self/rewind_self.md), figure [figs/rewind_self_bev.png](figs/rewind_self_bev.png) (left: a frontier unit, 4 / 7 variants complete; middle: p-hat 0 while the expert completes from the same h; right: a control that loses half its completions under the perturbation), prereg `plans/2026-10-16-rewind-self-prereg.md`.

REWIND-ONPOL-ALL (2026-10-16, decision 319): decision 316's rewind on the whole onpol-train split, to collect recovery labels. 1 031 scenes, 17 of which fail in the simulator. Sessions are 30 decisions, the turn window runs to decision 29, and the rewind is at Delta 2 and 3 s. There are 166 strict turns, 95 of them failed; 78 event / off units are handed over to the rule expert. The expert completes 0.65 / 0.68 of them, with a zero after h in 25 / 23. Label rows (completed and zero-free after h, junction stretch, 2 Hz): 1 064 from 49 turns in 36 logs. 542 of them pass the PAI gate and 493 the nuPlan-recalibrated gate. The trainer's row list is on the box at `$DATA_DIR/runs/route_follow/rewind_onpol_all/rows/rows.csv`, format in the README beside it. See [results/rewind_onpol_all/rewind_onpol_all.md](results/rewind_onpol_all/rewind_onpol_all.md), prereg `plans/2026-10-16-rewind-onpol-all-prereg.md`.

**Next.** Turning-school reads P0-P2 (`research/turning-school-checks.md`, to appear); the holdlead PAI exam read (coordinator exception);
spin-fix lateral bound v2; Tokyo PAI 281: RB0-F-s0 (same-gains base reference), RBSGc-PAI.

**Read more.** [research/decisions/286.md](../../research/decisions/286.md) (Chinese; full table, six line verdicts, branches, card-hours);
human page [research/route-follow/index.html](../../research/route-follow/index.html) (four figures copied to research/route-follow/figs/); task book
[research/next-round/route-night.md](../../research/next-round/route-night.md); registration
[plans/2026-10-13-route-night-prereg.md](plans/2026-10-13-route-night-prereg.md) (Amendments 1-5); diagnosis
[research/next-round/route-follow.md](../../research/next-round/route-follow.md).

- [research/zero-taxonomy.md](../../research/zero-taxonomy.md): every zero / low score of RSG-L18-s0 on PAI 281 and held 460 by mechanism
  (`scripts/rf_tax.py`, per-scene rows in `results/taxonomy/`, figures `figs/tax_*.png`, review sheet `research/route-follow/review-sheet-rsg.md`).
- LON-DIAG readings 1-2 ([plans/2026-10-15-longitudinal-diag-prereg.md](plans/2026-10-15-longitudinal-diag-prereg.md)): the 17 route-end zeros
  ([results/lon_diag/route_end.md](results/lon_diag/route_end.md), contact sheet `figs/lon_diag_route_end.png`) and the slow scenes (onset,
  triggers, fixed point, following gap: [results/lon_diag/slow.md](results/lon_diag/slow.md), `figs/lon_diag_fixed_point.png`,
  `figs/lon_diag_onsets.png`); `scripts/rf_londiag.py`, per-decision table on Tokyo `/data/runs/route_follow/lon_diag/decisions.parquet`.
- [research/route-follow/pai-follow.md](../../research/route-follow/pai-follow.md): PAI 281 turn completion with the real route (RSG 13.9 %,
  RBSG2 25.0 % of 36 turns) and the map topology of every corridor zero (14 / 22 route zeros have a straight road ahead; the ego never slows
  into the entry); `scripts/rf_pai_follow.py` (+ `.sh`, Tokyo container), `results/pai_follow/`, figure `figs/pai_follow_zeros.png`.
- [results/hard_train14/hard_train14.md](results/hard_train14/hard_train14.md): stored r281 reads (path 0) on the 14 strict hard_train turns
  (+ 3 trainable): completed / completed + d293 shape gate RSG 0 / 0 of 14, RBSG2 4 / 4 of 14; of 17 RSG 2 / 2, RBSG2 6 / 6; `scripts/rf_ht14.py`.
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
- Plan vs execution viewer `~/Desktop/plan_exec/index.html` (Mac and Tokyo desktops, not committed; the first 15-row page is `index_15.html`): every turn of RSG-L18-s0, grouped completed first then by class: the 36 PAI 281 strict turns + the 4 non-strict PAI scenes of the first page, and the 24 held-460 nuPlan turns (BEV + traces only, 2 Hz, approaches 3-19 m). Per row the BEV of every served plan from entry - 20 m (>= 3 s) to the event / exit + 10 m over the route lane chain (rf_corridor lanes), the stored front-camera clip with HANDOFF / ENTRY / EVENT flags (PAI), and traces (speed vs the human at the same station, plan heading change at 2 / 4 s vs the human, car lateral offset from the plan served 1 s earlier, plan k vs k-1 lateral jump at 2 s); frame check: anchor = controller pose (<= 0.011 m; held from decision 2), plan 1.0 s point to the executed track 0.04 m (PAI) / 0.02 m (held) median on the completed turns; `scripts/plan_exec_viz.py` (extract / clips on Tokyo, held records + lanes slimmed on the box, render on the Mac).
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
- [results/lc_hyst/reading.md](results/lc_hyst/reading.md): LC-HYSTERESIS (prereg `plans/2026-10-15-lc-hysteresis-prereg.md`): does the
  3-way command flip L/R -> S mid lane change and the plan snap back? **No on RSG** (C1 0 sites; RV2b 4), so no command-latch pilot; the
  command is a poor lane-change signal (8a143cb2 completes with cmd R / S only) and 415e950b drifts and snaps back with L held. Figure
  `figs/lc_hyst_timeline.png` (look at the red plan line against the command dots around station 0).
- [results/turn_gain/g1.25.md](results/turn_gain/g1.25.md): TURN-GAIN serve-side switch (diagnostic, not adopted; prereg
  `plans/2026-10-15-turn-gain-switch-prereg.md`, Amendment 1): RSG-L18-s0 + `JEV_TURN_GAIN=1.25` (plan yaw / course x1.25 whenever its 2 s turn
  is < 60 % of what its own route memory shows, from the policy's own states, no handover) on the 36 strict PAI turns + the 26 exam turn
  scenes (47). **Refuted** (turn magnitude is not the bottleneck): strict 6 / 36 vs 5 / 36 (gained e0f6ebe1 only, which fails the shape gate),
  exam turn scenes 3 / 26 vs 2 / 26, shape gate 2 vs 1; the switch fired on 30 / 36 strict scenes, mostly 20-40 m before the entry, and the
  outcome class changed in 3 scenes. No new zero (31 -> 28 on the 47, mean +0.059 [+0.000, +0.128]: 68e53160, a1b166cc, cdf95fd1 corridor
  zeros became scored off-route finishes). No g = 1.5 run (needs >= 3 more). Look at the per-scene table: failures stay where they were
  (corridor / offroad 0-20 m past the entry, several entering at 10-14 m/s). `scripts/rf_v2pai.py` (`tgcheck`, `turns`, `table`),
  `scripts/rf_v2pai.sh tg`, switch `experiments/alpasim/lib/turn_gain.py`.
- [results/intent_hold/intent_hold.md](results/intent_hold/intent_hold.md): INTENT HOLD serve-side switch (diagnostic, not adopted; prereg
  `plans/2026-10-15-intent-hold-prereg.md`, Amendments 1-3): RSG-L18-s0 + `JEV_INTENT_HOLD=1` on PAI (49 scenes: the 47 above + d299's PAI
  over-turn zeros; `JEV_TRACK=0`, the identity's no-gate path). While the route memory or the command shows a turn, the served path is held
  to a world-frame intent I = (1 - a) I + a plan, matched by arc length from the rear axle, with a cosine ramp onto it over 1 s and a 1 m
  guard; it resets on no turn, standing or a plan 1.5 m off I. Unswitched serving is bit-identical (2 / 2). **Completion lines fail for
  both variants.** a 0.3: strict 8 / 36, runway 5 / 22, loses 8a143cb2. a 0.15: 7 / 36, 3 / 22, loses 382a7247 and 8a143cb2. Neither adds a
  non-turn zero or a spin, and zeros fall from 33 to 30 / 27. The creep falls (failed turns -0.063 -> -0.038 / -0.029 m per m on the
  served plan) but the model plan still widens at -0.034 to -0.047 with the car held on its intent. So decision 308's creep is mostly the
  plan drifting against its own previous plan, which a hold only lags. Figure `figs/intent_hold_bev.png`: compare the dashed intent with the
  grey model plan, which lie together at every snapshot. `scripts/rf_intent.py` (replay / ihcheck / extract / creep / lines / fig),
  `scripts/rf_intent.sh` (chain / resume / read), switch `experiments/alpasim/lib/intent_hold.py`.
- [results/approach/](results/approach/): APPROACH COURSE (prereg `plans/2026-10-15-approach-course-prereg.md`, Amendments 1-3): the
  policy's own states 0-40 m before a >= 45 deg turn (nuPlan RSG-L18-s0 rollouts on1_L18RSG + PAI train160 on1_PAIonp; 3 091 turn rows,
  773 straight negatives, [rows.md](results/approach/rows.md)) pulled toward the human's entry state (`onpol_nuplan/lib/on_approach.py`: L1
  entry pose, L2 1-2 s blend, L3 speed not above the human's; `rm_train.py --app-lam`, flags off bit-identical, [ident.json](results/approach/ident.json)).
  **Pilot gate failed, stopped before any arm** ([gate/](results/approach/gate/)): at W 1 / 3 the held-out entry lateral and speed excess
  do not drop (-2 / -8 %, line +30 %) and dev ADE rises past the line (+0.020 / +0.037 m); the control W 0 changes nothing. On their own
  rows the pilots do fit (`*_trainrows.md`: lateral -28 / -30 %, speed excess -34 / -40 %, PAI speed excess 1.8 -> 1.0-1.1 m/s), so the
  terms learn without transferring. Look at the read tables: the dev base is already near the dead zones (lateral 0.26 m vs 0.69 m on the
  train rows; speed excess 0.006 m/s, the PAI dev has one scene with a speed read). Scripts `rf_apppool.py`, `rf_appgate.py`, `rf_app.sh`,
  `rf_appheld.py`.
  **EXPLORATORY APP1X-L18-s0** (Amendment 4; A1 at W 1, seed 0, a mechanism read and not a candidate): held 460 H1 fails.
  - Zeros 70 vs 26, +44 [+30, +58]; at-fault +26; offroad 41 vs 17.
  - Held turns 8 / 24 vs 16 / 24. The car enters 0.17 m further inside; entry speed is unchanged.
  - Look at [held/confirm.md](results/approach/held/confirm.md) and [held/entry.md](results/approach/held/entry.md).
  - PAI 281 ([pai/mechanism_APP1X-L18-s0.md](results/approach/pai/mechanism_APP1X-L18-s0.md),
    [pai281/APP1X-L18-s0.md](results/pai281/APP1X-L18-s0.md)). On the 22 runway turns the mechanism moves:
    - entry speed -2.96 [-4.45, -1.54] m/s (7.38 -> 4.80, human 4.99)
    - speed excess over the human -1.30 [-2.21, -0.56] m/s
    - inward offset at entry - 20 m +0.29 [+0.05, +0.59] m
  - Runway completion barely changes: 4 / 22 vs 3 / 22. The other 14 strict turns go 7 vs 2; strict 36 11 vs 5; shape gate 8 vs 1.
  - Harm: all 281 -0.062 [-0.108, -0.016], spins 29 vs 2, offroad zeros 32 vs 14, control 75 6 zeros vs 0.
  - Look at the per-scene table in the mechanism file: compare entry speed and completion on the runway rows.
  **APPROACH COURSE v2 (APP2)** (prereg `plans/2026-10-15-approach-v2-prereg.md`, Amendments 1-2; [results/approach/v2/](results/approach/v2/)):
  the same rows with the term scoped to its purpose (`rm_train.py --app-v2`; on_approach v2 branch; sidecar `rf_apppool.py v2`):
  - no term in the first 1.5 s after the handoff
  - the speed term only with the ego <= 26 m before the entry
  - a symmetric speed term to the human on the straights
  - a 0.5 m lateral dead zone to the human entry offset, clipped inward at 0.75 m
  - a 2 s over-turn guard

  Flags off and the v1 path are bit-identical ([ident2off.json](results/approach/v2/ident2off.json), [ident2app1.json](results/approach/v2/ident2app1.json)).
  - Pilot guard gate ([gate/](results/approach/v2/gate/)): W 0.5 passes (G-real 0.5403, line 0.5435); W 1 fails (0.5508).
  - **APP2W05-L18-s0 fails H1 on held 460, stopped before PAI** ([held/confirm.md](results/approach/v2/held/confirm.md)):
    - zeros 70 vs 26, +44 [+29, +60]; at-fault +24; offroad 34 vs 17, corridor 27 vs 6; dev ADE +0.025
    - held turns 14 / 24 vs 16 / 24
    - the car still enters further inside, +0.16 [+0.08, +0.23] m ([held/entry.md](results/approach/v2/held/entry.md))
    - 35 of its 50 new zeros are also APP1X's new zeros, so the harm is common to both term designs

  Look at the confirm table next to APP1X's [held/confirm.md](results/approach/held/confirm.md): the zero count and the kind split barely move.
  - **W 0 control APP0C-L18-s0** (Amendment 3; same pool, `--app-lam 0`; [w0/held/confirm.md](results/approach/w0/held/confirm.md)) is level
    with RSG-L18-s0:
    - zeros 24 vs 26, -2 [-7, +3]; mean +0.015; held turns 19 / 24; dev ADE +0.0004
    - against APP2W05: zeros -46 [-64, -30]

    So the pool mix is harmless, and the held harm comes from the approach gradient itself, in both v1 and v2.
  - **Seed 1 of the control (Amendment 4)** does not repeat seed 0's gain: score +0.004, zeros +3 [-4, +11], turns 14 / 24 vs 14. Over two
    seeds: score +0.009 [-0.001, +0.021], zeros +0.5, turns 33 / 48 vs 30 / 48. The control is neutral; seed 0's gain was seed noise.
  - **Zero attribution (decision 303 Amendment 5; [app1x_zeros.md](results/approach/app1x_zeros.md), `scripts/rf_appzeros.py`,
    figure [figs/app1x_zeros.png](figs/app1x_zeros.png)).** APP1X's +44 zeros are mostly outside the turn: +27 outside, mostly lateral
    corridor exits, a median 15 m past the entry.
    - At the zeros the speed matches RSG's (x0.99) while the car under-turns (heading -4.8 deg, lateral -0.75 m).
    - On identical inputs the plan turns about 1 deg less by 4 s before the entry, and its arc does not change.
    - So the shape terms (L1 / L2) did the harm and the speed term did not. APP2 repeats the pattern.
- [results/spin_v9/](results/spin_v9/README.md): SPIN v9 (prereg `plans/2026-10-15-spin-v9-prereg.md`, Amendments 1-3; serve side,
  `JEV_TRACK=v9` in `experiments/alpasim/lib/track_ref.py`; scripts `sv9_diag.py`, `sv9_check.py`, `sv9_read.py`, `sv9.sh`).
  - Mechanism: APP1X's 29 handoff spins come from a fed - plan speed gap of 7-12 m/s at the 1.7 s handoff. JEV_VCONT turns the gap into a
    7-12 m/s^2 demand at the MPC's bound. v8 was never served in that read.
  - v9a (v8 + a -2.5 m/s^2 bound for the first 2.7 s) gives spins 29 -> 1 (APP1X) and 26 -> 1 (RSG). It fails the control line on both
    (zeros +1 net, CI upper +4 / +5, APP1X mean -0.011), so it is not adopted.
  - Plain v8 already gives 29 -> 1 on APP1X.
  - Look at the per-case notes in the README: 2fe500f7 is a reversing plan, d8bfaaf2 the pose fault, and 268949d2 / 09528162 are v8's
    low-speed control zeros.
- [results/corridor/](results/corridor/check.md): ROUTE-LANE CORRIDOR labels and the offline check (turn-night.md section 6; prereg
  `plans/2026-10-15-corridor-check-prereg.md`, Amendments 1-4; `scripts/rf_corridor.py` lanes | plans | build | check, `scripts/rf_corridor.sh`).
  Labels: the route's own lane chain (entry lane -> connector -> exit lane, the target lane at a lane change) + 0.3 m, classes corridor /
  adjacent same-direction / oncoming / off-road / other drivable, per decision a 48 x 48 R-B-frame grid and the corridor polygon (box
  `$DATA_DIR/runs/route_follow/corridor/build/labels_*.pkl`); found on 0.98 of held and 1.00 of PAI decisions ([coverage.md](results/corridor/coverage.md)).
  **Registered read: the label gate fails, so C1 / C2 are void.** The human's own 1-4 s footprint leaves the corridor into a hard class in
  0.13 [0.04, 0.22] (held) and 0.33 [0.22, 0.45] (PAI) of strict-turn window decisions (line 0.05). Taken at face value, 0.86 of the
  inside / oncoming / over-turn zeros have a hard-exit plan by t_f - 1 s (60 / 70), but so do 0.37 of the completed turns' decisions.
  The exploratory sweep ([sweep.md](results/corridor/sweep.md), Amendment 4) shows the trade-off: at margins that make the held human pass
  (1.0 m, or the axle line), C1 falls to 0.48-0.69 and held C2 stays at 0.12-0.18. PAI never passes (0.12-0.25; PAI lane polygons are
  narrow and there is no road-area layer, so the human's corner cuts read as off-road). The hard-exit term does not separate failing plans
  from the completed turns' plans. Look at [figs/corridor_bev.png](figs/corridor_bev.png): 1a69f757 and APP1X 53d37873 have plans in the
  oncoming lane / off-road 2 s before the zero, but so does the completed held left turn 6f393be3. 382a7247 and de9db5dc are collision zeros
  whose plans stay in the corridor.
- [results/receding/](results/receding/receding.md): RECEDING-TURN fingerprint (turn-night.md section 9; prereg
  `plans/2026-10-15-receding-turn-prereg.md`, Amendment 1; `scripts/rf_receding.py` extract-pai | extract-held | read, `scripts/rf_receding.sh`).
  Per strict turn, the slope of the served plan's 15 deg onset arc against the distance to the entry over the approach (1 = the onset stays
  at one world point, 0 = it stays at one distance ahead of the car) and the world onset's forward drift. The human log read as a plan
  gives 0.92 and 1.8 m (gate passed). **Registered: C1 inconclusive, C2 refuted.** On PAI RSG-L18-s0 the failed turns' median slope is 0.67
  and the completed turns' 0.79 (21 / 3 turns with a slope; D -0.13 [-0.41, +0.33]). Drift is 3.3 vs 4.0 m. RV2a and RV2b refute C1 and C2.
  APP1X is the only set in the predicted direction (0.53 vs 0.96, D -0.42 [-0.72, -0.19]), but its failed median is above the 0.3 line, so
  it reads inconclusive. Held approaches are 3-19 m long (about 5 decisions at 2 Hz) and pooled they refute both claims (0.79 vs 0.57).
  The plan does not push the turn ahead of the car. Within 15 m of the entry the plan's onset sits 1.1 m *before* the route's own 15 deg
  point in failed turns (completed: 0.2 m before). The classes differ in other ways:
  - **straight-through:** the plan never turns. Its onset is visible on 0.12 of the approach decisions. On every decision whose 4 s horizon
    already covers the route's onset, the plan has no onset (unseen 1.00). Its 4 s heading change at 0-10 m is 5.6 deg; the human's is 74 deg.
  - **late-wide:** the onset is in place (slope 0.70, lag -1.4 m), but the plan asks for too little heading. At 10-20 m the plan asks
    5.3 deg vs the human's 27.6 deg. Inside the turn, the plan's remaining heading follows only half of the true remaining heading
    (OLS 0.49; completed 0.93), and the car never reaches the exit heading.
  - **over-turn** (the 11 d299 inside zeros, Amendment 1; none of the failed strict turns is one): the onset is fixed and early (slope 0.72,
    lag -1.9 m). The zero comes before the car reaches the exit heading, and the plan then asks 5.4 deg less than the remaining heading.
    Completed plans that pass the exit heading ask to turn back (-4.4 deg). So there is no evidence that the plan keeps asking for heading.
  On the 22 runway turns APP1X's slope matches RSG's (0.72 vs 0.74, paired +0.04 [-0.17, +0.16]): the human entry speed does not change
  where the plan places the turn. Look at [figs/receding_onset.png](figs/receding_onset.png): on the left, the traces follow the slope-1
  line from 20 to about 5 m and then flatten in both groups. On the right, failed (dashed / dotted) and completed (solid) world onsets
  are equally flat until the last 5 m. The straight-through traces shown are the 2 of 8 whose plans turned often enough for a slope.
  Extracts: Tokyo `/data/runs/route_night/receding/pai.pkl`, box `$DATA_DIR/runs/route_follow/receding/held.pkl`
  (+ `tax/` for RV2a / RV2b, `decisions.parquet` = every decision's onset / dpsi / human values).
- [results/mode_avg/](results/mode_avg/mode_avg.md): MODE AVERAGING, Reading 1 (prereg `plans/2026-10-15-mode-averaging-prereg.md`,
  Amendments 1-3; `scripts/rf_modeavg.py read` on the RECEDING extracts, the rf_progress route-memory extracts and the lane extracts in
  `tmp/progress/`). deficit = 1 - plan 4 s heading change / human 4 s heading change, 0-30 m before the entry; exits = distinct outgoing lane
  directions (>= 20 deg apart) from the same-direction lanes at the junction mouth (gate: the route's own direction is among them on 97 % /
  100 % of PAI / held turns). **Registered: M1 inconclusive, M2 untestable.** Exits: 3+ vs 2 exits differ by +0.01 [-0.49, +0.54] (PAI
  RSG-L18-s0, 15 / 16 turns), +0.08 [-0.03, +0.20] (held RSG-L18-s0 / -s1), +0.05 [-0.06, +0.23] pooled; the 5 PAI curves with one exit fail
  with a negative deficit. Command: inside 30 m the served command names the turn on every decision (PAI 1 980, held 290; never S). Where
  a rollout starts far enough out, it switches from S to the turn at 36.6 m (median, 28.7-40.6). The plan's shortfall is therefore not
  a weak command. Memory coverage of the junction has no consistent relation to the deficit once split by distance (M3). M4 (0-10 m, PAI
  failed turns): straight 0.13, partial 0.18, full 0.69 of decisions, median ratio 0.85, read "mixed"; held near 1. **Amendment 3
  changes the reading of d306's "heading deficit".** The failed turns approach slower, so their 4 s plan covers less road. Measured over
  the plan's own arc, the late-wide plans follow the road: at 10-20 m the plan asks 5.3 deg, the human 27.6 deg, the route over the plan's
  9.8 m arc (2.6 m/s) turns 3.5 deg. Their arc-matched ratio has median 0.83 and is full on 0.76 of decisions. no_zero turns read the same.
  Only straight-through (8 turns) is a true heading miss: the route turns 68.5 deg over the plan's arc, and the plan asks 1.9 deg. Its
  decisions split into two clusters, not a smear (10-20 m: straight 0.61 / partial 0.38 / full 0.01; 0-10 m: 0.60 / 0.09 / 0.31). That is
  the shape of a choice between modes, not an average. Look at [figs/mode_hist.png](figs/mode_hist.png). Left: the registered ratio near
  the entry. Middle: late-wide's ratio against the human (dashed, piled at 0) moves to about 0.8-0.9 against the road over its own arc
  (solid), while straight-through (green) stays at 0 with a second cluster near 0.45. Right: each turn's decisions; the green
  straight-through rows sit at 0 and jump, the orange rows scatter around 1.
  Reading 2 (FLOW1's flow head vs its regression twin in closed loop) is in [../flowhead/README.md](../flowhead/README.md): the flow head
  completes 0.5 / 24 held turns vs 11, F1 / F2 refuted.
- [results/switching/](results/switching/switching.md): PLAN SWITCHING (prereg `plans/2026-10-15-plan-switching-prereg.md`, Amendment 1
  post hoc; `scripts/rf_switching.py read` on the RECEDING extracts, CPU). Window entry - 20 .. entry + 20 m to the first zero. Per decision:
  r = plan / route heading over the plan's own arc, j = jump of the plan's 2 s point off the previous plan's path (+ inside), e = the car's
  offset from the plan of 1 s earlier (+ inside). Gates pass: anchor = controller pose to 0.000 m; the human log treated as a plan switches
  0.000 with |j_h| 0.015 m. **The plan does not switch.** There are 0 switching pairs (r >= 0.6 next to r <= 0.3) in 1 596 PAI RSG-L18-s0 pairs,
  1 in 1 580 on APP1X and 0 on every held set. So S1 is 0 / 0, inconclusive by the registered rule. APP1X's "supported" is degenerate
  (0.001 vs 0). **S2 is refuted** on PAI RSG: median |j| is 0.030 vs 0.058 m, ratio 0.53 [0.21, 1.02]. It is also refuted on RV2a,
  RV2b, APP1X and held RSG. Failed plans are smoother than completed ones. The car follows each plan closely: |e| is 0.02 vs 0.03 m.
  d307 M4's 31 % straight / partial was measured against the human's 4 s. On the arc-matched r, failed turns 0-10 m before the entry are
  S 0.09 / P 0.04 / F 0.86, and 76 % of the variance lies between turns: straight-through turns stay straight throughout. S3 by the
  registered rule reads "averages the sequence" on PAI (executed / route = 0.83 x mean r - 0.09, R^2 0.73, vs p90 r slope 0.49). The
  sequence is smooth, though, so the average is just the plans themselves, under-executed. Amendment 1 finds what does separate the
  groups: **plan creep**. Each new plan starts from the car's actual pose. In failed PAI turns its 2 s point lies outside the previous
  plan's path by -0.063 vs -0.015 m per metre of travel (diff -0.048 [-0.104, -0.016]); summed over the window that is -2.1 vs -0.4 m
  (-1.70 [-4.86, -0.81]). In late-wide turns it is -0.133 m / m and -3.7 m. The control, 40-80 m before the entry, is +0.003. Over-turn
  zeros creep inward (+0.14 m / m). Over the next 2 s the car turns 2.5 deg less than each plan asks, against 1.6 deg more in completed
  turns (diff -4.1 [-6.6, -2.3] deg). Replications point the same way but the CIs reach 0 (APP1X at matched approach speed: -0.041 vs
  -0.007 m / m). Late-wide on all 4 PAI checkpoints: -0.100 [-0.138, -0.034]; the 6 held late-wide turns: -0.104. Look at
  [figs/plan_switching.png](figs/plan_switching.png). Row 1: failed r traces are smooth and never jump between the shaded bands. Row 2:
  failed j stays small and below 0, while completed j is larger and scatters around 0. Row 4: summed j falls steadily to about -2 m in
  the failed turns and stays near 0 in the completed ones (382a7247 creeps inward).
- [results/creep_rep/](results/creep_rep/creep_rep.md): CREEP REPLICATION (prereg `plans/2026-10-15-creep-replication-prereg.md`, Amendment 1
  post hoc; `scripts/rf_creep_rep.py read`, box stages `scripts/rf_creep_rep.sh`, CPU). d308's creep measure, unchanged and with its sign
  flipped (+ = outward), on 408 held 460 @18 turns of 17 checkpoints d308 did not use: 9 lr x steps cells, RV2c s0 / s1, APP0C, APP2W05,
  FMH-F and RGH-F s0 / s1. 178 failed, 230 completed; log-clustered bootstrap. Gates pass: the code reproduces d308's held_all6 to the
  printed digit; the human log treated as a plan creeps +0.0025 m / m. **As registered, R1 and R2 are supported.** Creep per metre is
  +0.038 vs +0.000 (diff +0.037 [+0.009, +0.069]); summed over the window that is +0.73 vs -0.09 m. The 2 s heading shortfall is -1.8 vs
  +0.4 deg (-2.2 [-3.4, -0.5]). **But the flow head carries the pooled result.** FMH's 47 failed turns include all 32 straight-through
  turns, which creep +0.265 m / m. Without FMH the difference is +0.017 [-0.012, +0.043] and x2 -1.3 [-2.8, +0.1]. Amendment 1 finds
  where creep lives without FMH. Late-wide creeps +0.097 [+0.051, +0.120] m / m (n 29; d308 held late-wide 0.104, PAI 0.133) with x2
  -4.3 deg. Failures with a zero creep +0.038 [+0.004, +0.063]. Slow / short turns (no zero, 92) do not creep: +0.007 [-0.029, +0.025].
  Over-turns creep 0 and turn more than the plan (+2.1 deg). So creep is the signature of turns that end wide, not of every failure.
  Families: RGH +0.030 vs +0.004, APP0C +0.053, the lr cells +0.006 (their failures are 62 / 73 short). R4: in the first 0.5 s, when plan
  k alone drives the car, the shortfall is -0.1 deg on failed turns and -0.3 on late-wide. Most of it builds in 1-2 s (late-wide x1
  -1.2, x12 -2.9). The registered reading on all failed turns is "both". Each late-wide plan's far second asks more than its near second
  (19.3 vs 15.3 deg). Splitting the late-wide 1-2 s shortfall through plan k+2, served 1 s later: the newer plan, drawn from where the car
  is, asks 1.9 deg less for that second than plan k did; the car then under-executes plan k+2 by a further 1.0 deg. So about two thirds
  is re-planning and one third tracking. Look at [figs/creep_rep.png](figs/creep_rep.png). (a): completed turns sit on 0, late-wide and
  straight-through float above it, and open circles (FMH) fill the straight-through column. (b): only FMH separates by family. (c):
  every failed class starts at 0 at 0.5 s and fans out by 2 s.
- [results/speed_geo/](results/speed_geo/speed_geo.md): SPEED-GEOMETRY (turn-night.md section 10, the user's reading "over-guided by the
  route, ignores what the road geometry implies"; prereg `plans/2026-10-16-speed-geometry-prereg.md`, Amendment 1; `scripts/rf_speedgeo.py`
  brake-extract (box) | read (Mac), CPU, on the RECEDING extracts and the PAI rf_tax extract). **A, wide turn = speed: A1 supported, A2
  supported on PAI, but the effect is a mode switch, not a curvature cap.** r = plan peak curvature over 0-2 s / human peak curvature over
  the same stretch of road, q = ego / human speed at the same station. On PAI RSG-L18-s0 the slope of log r on log q is -1.68 [-2.48, -0.71]
  (34 turns; the control A1c, log v with log v_h and log kappa_h as covariates, -1.77 [-2.59, -0.74]). It replicates on 0-4 s (-1.49), the
  time-matched human (-1.52), held RSG s0 + s1 (-0.88 [-1.64, -0.24]) and APP1X (-0.64). But within a turn the slope is -0.14
  [-0.70, +0.31]. Straight-through turns carry it: r 0.05 at q 1.36. Without them the slope is -0.24 [-0.48, -0.02] (within +0.01), and
  they make up 28 / 39 / 50 % of the decisions at q > 1 / 1.25 / 1.5. Late-wide plans match the human's curvature on the same road
  (r 0.92) while slower than the human (q 0.89; 3.4 vs 4.4 m/s), so late-wide is not a speed effect (d308 creep stands). A2: the plan's
  0-4 s lateral acceleration has p90 2.89 [2.26, 3.18] m/s^2 (human 3.91). Following the human's line at the ego's speed would need p90
  8.94 [5.11, 11.13] on the 841 decisions where the ego is faster. held refutes A2 (2.71: where the ego is faster on held turns, the human's line asks little); APP1X is
  inconclusive. A3: on 31 shared turns APP1X moves r toward 1 (median |log r| 0.17 vs 0.29, paired log r +0.09 [+0.01, +0.21]; q 0.94 vs
  0.99; completed 11 vs 4). **B, no braking for what it sees: B1 untestable, B2 refuted, B3 descriptive.** dP = v - plan speed at 2 s,
  dH = the human's 2 s drop at the same log station. At the 9 collision zeros the human did not brake at those stations (mean dH -0.33,
  dP -0.00), so the registered ratio is undefined. The exploratory read: of the 3 zeros whose hit object was closing at TTC <= 3 s 2 s
  before contact, the plan speeds up into both oncoming cars. In 1a69f757 an oncoming car 35 m away is at TTC 2.4 s; the plan asks for
  +2.2 m/s and the car goes from 6.3 to 8.0 m/s. In 382a7247 the plan asks +0.7. For the cyclist (1e8e939e) the plan cuts 3.6 m/s at
  13.5 m/s, but the car only slows from 14.6 to 11.6. In 229791a5 the human waited at the stop line that the ego ran at 4-5 m/s,
  speeding up into a crossing truck. The other 5 are low-speed sideswipes and a pedestrian walking into the stopped ego. With a
  same-direction lead the plan does brake. The human braked for a lead in 83 scenes, but the ego came within 3 s TTC of one in only 14
  (it is slower). There, R2 = 0.81 [0.15, 1.64], 1.11 when both face a lead, and 1.20 at their own times. At a lead the plan asks for
  2.36 m/s (half its speed) against 0.57 without one, and the car drops 3.4 m/s. The plan drops at least half the human's in 6 / 8
  scenes. The exception is b40f4729: at 14.6 m/s, with a lead 19 m ahead at TTC 2.1 s, the plan speeds up. B3 (7 oncoming-lane entries):
  the plan does not slow before entering (dP -0.06 vs dH +0.02) and speeds up inside (-0.76). The human speeds up there too (-1.42,
  leaving the turn), and there is rarely an in-path vehicle. So: it brakes for cars ahead. It does not slow for crossing or oncoming
  traffic or for stop lines, and it opts out of turns it reaches fast. Look at [figs/speed_geo.png](figs/speed_geo.png). Top left: turn
  medians sit on the r = 1 line across q. The green straight-through turns fall far below it, most of them at q > 1. Top middle: the arrows
  from RSG to APP1X climb toward r = 1. Top right: the plan's lateral acceleration (orange) stays almost entirely under the 4.5 line where the human line
  at the ego's speed (blue) climbs to 8-15. Bottom: 1a69f757's plan speed (orange) climbs as the range to the oncoming car (purple) falls.
  In 229791a5 the human's speed at the station (blue dashed) drops to 0 at the stop line. The cyclist's plan brakes but the car (black)
  lags.
- [results/rule_expert/](results/rule_expert/README.md): RULE EXPERT (prereg `plans/2026-10-15-rule-expert-prereg.md`, Amendment 1): a
  privileged map-based rule planner (`experiments/alpasim/lib/rule_expert.py`, `JEV_EXPERT`: the route-lane chain from rf_corridor's
  matcher, pure pursuit, human-speed cruise with a 26 m ramp to 4.6 m/s at every >= 45 deg turn, swept-footprint yield on the replayed
  boxes, a nudge around parked boxes; stop lines ignored) replaces the model's plan in the unchanged AlpaSim loop; a teacher check for
  DAgger, no training. **Held 460: E1 passes, 22 / 24 held turns** (RSG-L18 16 / 14; both failures speed, no zero), mean 0.918 vs 0.881,
  zeros 22 vs 26, at-fault 13 vs 21, E3 passes. **PAI 281: E2 passes at the line, 29 / 36 strict turns** (RSG 5; shape gate 24 vs 1,
  runway 15 / 22 vs 3, entry speed 4.6 vs 7.4 m/s), all-281 mean +0.193 [+0.136, +0.250]; failures speed / yield 4, PAI corridor 2,
  collision 1, wrong chain 0. **Verdict: qualifies as a teacher** (caveats: privileged, stop lines ignored, E2 on the line). Look at [figs/rule_expert_smoke.png](figs/rule_expert_smoke.png)
  (the expert's red path on the dashed chain against the human's blue path through six turns RSG fails or barely makes) and the per-turn
  list `results/rule_expert/held_XP-L12-E46/xp_turns.md`; [figs/rule_expert_pai_fails.png](figs/rule_expert_pai_fails.png) for the
  PAI failures (yield waits behind same-direction traffic). Scripts `rf_expert.py` (pack | fig | report), `rf_expert.sh`.
- [results/slow/](results/slow/pilot.md): SLOW S1 / S2 round (prereg `plans/2026-10-15-slow-s1s2-prereg.md`, Amendments 1-3; running): bit-identity gates (`gates.json`, all 0), PAITV pool gates (`gates_PAITV*.json`, `pool.json`; BEV sheets `figs/pait_bev_PAITV80.png`, `figs/pait_bev_PAITVn.png`: look at the red demo on the blue logged path with no box on it within 40 m), pilot gate table (`pilot.md`: S1 / S2 / S3 pass, Sall fails dev ADE); chain `scripts/rf_slow.sh`, line 3 `rf_londiag.py pai`.
- [results/lon_inventory/](results/lon_inventory/inventory.md): LON-INVENTORY (plan `plans/2026-10-16-lon-inventory.md`, CPU only): material for longitudinal envelope rows on RSG-L18-s0's own nuPlan states (on1_L18RSG, 17 474 rows with frames): qualifying stop-line approaches 77 decisions / 36 scenes at >= 6 m/s (451 / 85 at >= 3, 0 at >= 12; the plan brakes where the human stopped), free road < 0.85 x human 2 006 / 319 (none at >= 16 m/s), turn approach > expert + 1 m/s ~80 own-state decisions; expert packs 64 / 1 021 scenes (proxy v_ref agrees, median 0.11 m/s); on1_PAITV the only rows at PAI handover speeds. `scripts/rf_loninv.py`.
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

[results/](results/) 668 result files · [figs/](figs/) 79 figures · [plans/](plans/) 30 live plans · [scripts/](scripts/) 94 entry points
<!-- files:end -->
