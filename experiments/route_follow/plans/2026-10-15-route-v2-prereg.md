# ROUTE v2: ordered route input, near-field lane hinge with negatives, lane-change counterfactual pairs (pre-registration)

2026-10-15 night, written before any pilot read. Task book: research/next-round/turn-night.md section 2 (training-design candidates, to be cut
by readings A and B). Context: decisions 282, 284, 286 (RB0-F / RSG-L18 / RBSG2), 287, 293, 297. Code: `experiments/route_mem/lib/rmem.py`,
`experiments/route_mem/scripts/rm_build.py order`, `rm_train.py` (flags below), `experiments/route_follow/scripts/rf_v2gate.py`, `rf_v2.sh`.

## 1. The three changes (each behind a flag, all off = RSG-L18's recipe operation for operation)

| flag | what | why (symptom in turn-night.md section 0) |
|:--|:--|:--|
| `--rm-order` | memory window cut at +40 m (24 points, -6 .. +40 m at 2 m from the ego's projection; far points dropped, not masked) + two scalars: arc distance to the next route heading change >= 30 deg (cap 60 m, unknown -> mask bit) and arc length travelled since the first route message | near route empty -> the plan heads for the far points (shortcut, 180 instead of 90 deg: 9f393c8a, 2466266e); progress / order missing (d287's 40 m gap) |
| `--lane-lam` | lateral hinge of plan poses 1.0-2.0 s (w 2) and 2.5-4 s (w 1) to the carried route, dead zone `--lane-dz` 0.5 m, shape-only (lateral) gradient, on every non-anchor row whose memory is fed (log, yaw-rate and pool rows); negatives `--lane-neg` 0.2: fake input memory (parallel shift +-3.5 m, or the straight continuation on curved non-junction NAVSIM rows) with the ORIGINAL route as hinge target and the imitation target unchanged; over-turn guard: relu(heading excess into the turn at 2 s - 10 deg) | right lane never penalised (1a69f757); over-turn into kerbs (382a7247, de9db5dc); RBSG2's hinge over-turned and cost public offroad zeros (d286 correction 2) |
| `--lc-cf` | lane-change counterfactual pairs: per batch k tokens, the real row (real route memory, imitation of the log) and a stay-lane row (same camera tokens, stay-lane route memory + its command, lane hinge to the stay-lane label with 0.5 m dead zone, no imitation) | lane-change commands ignored (d284: swing 0.05 m; d286: CF rows taught junction switching but lane changes only 5-7 %) |

Implementation facts settled while building (recorded here as the decisions they are):

- D1 (ordered layout). The existing window (rmem.window) is already ordered by arc from the ego's projection on the merged route and
  resampled every 2 m; the ordered layout is therefore its first 24 points (bit for bit) plus the scalars, a pure function
  `rmem.to_order(old window, trav)` shared by the offline rows, the served Tracker and the trainer's on-the-fly negatives. A progress anchor
  for the near gap (ego arc = -42.1 m + odometry instead of the geometric extrapolation) was considered and rejected: with a 40 m horizon it
  leaves the first decisions with no point at all, whereas the geometric anchor keeps the post-turn points of a route that starts beyond a
  turn in the 0-40 m slots. The d_turn reference heading is the route's at +0 m when that point is valid, else the ego's own (the gap).
- D2 (train / serve gap). Training rows keep the serve-time near gap: NAVSIM rows are decision k = crc32 % 18 of an 18-decision rollout that
  follows the log (rm_build nav), pool rows are rebuilt at the driver's anchors (G-serve of d282: 90 / 90 masks equal). trav uses the same
  rule on both sides (chords between the route-message poses, then to the anchor). Verified by `rm_build.py order`'s near-field summary
  (share of rows with no valid point in 0-20 m) and, for any ordered checkpoint that is read, `rm_build.py served` (now compares d_turn and
  trav too).
- D3 (hinge target). The lane hinge's target is the old-layout memory polyline (-6 .. +72 m) of the row, whatever the input layout: the
  carried route, not the log path, so the hinge is 0 where the memory does not reach the pose (no target in the gap). The pose weights are
  on_route's W_T (0 at 0.5 s).
- D4 (straight negatives). Only on NAVSIM training rows whose route turns >= 15 deg within 40 m and whose token has no alternative-exit
  counterfactual in runs/route_follow/cf/rows (j0 / j1 = a junction choice exists): a straight continuation at a junction can be a legal
  route, and teaching the plan to ignore it would contradict G-swap. Half of the negatives on such rows are straight, the rest shifts.
- D5 (lane-change rows). rf_cfpool already builds stay-lane rows: lane-graph lane changes within 4 s ending >= 2 m off the t0 lane, 3 972
  candidates, 2 615 ok (2 589 train in on1_CF + 26 dev in on1_CFdev); 1 140 fail because the real route memory already equals the stay-lane
  route (the generator kept the lane), 262 on the lane snap. d284's 4 668 is a geometric proxy (lateral shift >= 2 m in 4 s) that also
  counts curves. So `--lc-cf` reuses the 2 589 ok rows (no duplicate pool; the rows are drawn straight from cf/rows/<shard>.npz with the
  token's own Store row), and the extension is the pairing: the real row is drawn in the same batch. No new rows tonight.
- D6 (weights). `--lane-lam 10` (per row-metre, summed / non-anchor rows: a third of RBSG's 30 per pool row, now on every row), over-turn
  weight = `--lane-lam` per radian, `--lc-lam` = `--lane-lam`, `--lc-k` 8 at batch 128 (4 at the pilot's 64). On-k 7 at batch 64.
- D7 (init conversion). An old-layout branch (RSG-L18-s0) initialises the ordered head with its first 24 position embeddings and a zero
  scalar map, so the converted init differs from the base only by the dropped far points (step-0 dev ADE logged).

## 2. Which reading gates which flag

| reading (turn-night.md section 1, other agents) | supports -> | does not support -> |
|:--|:--|:--|
| A (progress: turn zeros by near 0-20 m memory coverage) | `--rm-order` on | off |
| B (lane: lateral offset to the route in the 20 m before the entry, completed vs failed) | `--lane-lam` on | off; the lever is steering amplitude / speed (separate line) |
| B's lane-change share of the failures (route asked a lane change) | `--lc-cf` on | off |

The pilot below runs all three on; the gates are read per flag only through the pilot table (no per-flag pilot unless a gate fails).

## 3. Gates (offline, `rf_v2gate.py`, pilot = 1 000 steps, seed 0, batch 64, gentle schedule, init RSG-L18-s0; base = RSG-L18-s0)

| gate | rows | line |
|:--|:--|:--|
| G-rely | dev junction rows (route enters a junction <= 40 m) | mean ADE(memory masked) - ADE(on) >= 0.01 m |
| G-swap | on1_CFdev junction rows (j0 / j1) | median share of the needed move at 1.5-2 s >= 0.50 and 4 s pose on the counterfactual side >= 70 % |
| G-ST | dev straight rows +-3.5 m shift; dev curve rows (no junction, turn >= 15 deg) straight continuation | median lateral move at 1.5-2 s <= 0.10 m, both |
| G-LC | on1_CFdev lc rows | median |y(2 s) stay - y(2 s) real| >= 0.50 m |
| G-real | dev rows | ADE (memory on) <= base + 0.01 m |
| G-over | dev junction rows | share with plan heading at 2 s > route heading + 10 deg toward the inside <= base's share |

Pass on all six -> the flag set may go to `train` (2 seeds) once the user names the base and flags. G-swap < 0.20 at 1 000 steps -> the route
input is not read: stop the training change (R-CF's stop rule). G-real / G-ST / G-over fail -> no full training with that flag set;
diagnose with single-flag 1 000-step pilots (each <= 0.1 card-h) before proposing.

## 4. Held 460 lines (after `train`, 2 seeds, 18 decisions; `rf_v2.sh report`)

- H1 no harm: zeros and at-fault zeros (452 paired, seed mean) vs the base and vs RSG-L18: log-clustered 95 % CI upper bound <= +5 each.
- H2 turns: completion on the 24 held >= 60 deg turns (rf_follow), seed mean >= base + 3 turns; right-turn-inside offroad zeros not above the
  base + 2 (the RBSG failure).
- H3 public guard later (1491 / 10 decisions, separate submission after H1 and H2): zeros and at-fault vs the base, CI upper bound <= +5
  (RBSG2 failed here with +32.5 zeros); then PAI 281 on Tokyo.

## 5. Cost (training 0.5 card-h / 10 000 steps; held read 0.7 card-h / checkpoint, closed-loop.md)

feature build CPU only (~5 min, 24 cores); bit identity 2 x 50 steps ~0.15 card-h (setup dominated); pilot 1 000 steps x 64 ~0.05 card-h +
gate ~0.1; total tonight <= 0.4 card-h. Full round: 2 seeds x 2 500 steps ~0.25 card-h + gates 0.2 + held 2 x 0.7 = ~1.9 card-h; public
guard ~2 card-h per checkpoint.

## 6. Stop rules

- Bit identity with every flag off fails (any tensor differs after 50 eager steps vs the code before route v2): nothing runs until fixed.
- Pilot gate rules in section 3; full training only on the user's word.
- Held: H1 fails on either seed mean -> not adopted, whatever H2 says.

## Amendment 1 (before any pilot read, 2026-10-15 00:15)

- d_turn must persist 4 m (the point two slots on also differs >= 30 deg, or the memory ends there): ~6 % of NAVSIM rows carry a one-segment
  merge kink (e.g. 3, 3, -50, -50, 3 deg) that read as a turn (commit "rmem.turn_dist: a heading change must persist 4 m").
- `rm_build.py order` built (CPU, 3 min): rows with d_turn known 0.99, d_turn < 60 m 0.41-0.43 (NAVSIM), 0.38-0.40 (yaw-rate), 0.33 (pool
  L18RB0), 0.73-0.76 (on1_CF / on1_CFdev); trav median 14 m (NAVSIM), 19 m (pool); 57 % / 54 % of the rows have no route point in 0-20 m.
- D2 verified (the near gap of the training rows against the served memory of RSG-L18-s0's held 460 read, `rm_build cover`): share with no
  route point in 0-20 m by decision k

  | k | served RSG-L18-s0 held | pool L18RB0 | NAVSIM rows |
  |:--|--:|--:|--:|
  | 0 | 0.951 | 0.965 | 0.947 |
  | 1-2 | 0.948 | 0.961 | 0.950 |
  | 3-5 | 0.857 | 0.892 | 0.885 |
  | 6-9 | 0.480 | 0.520 | 0.570 |
  | 10-17 | 0.195 | 0.216 | 0.303 |
  | all (n 8 129 / 17 429 / 101 313) | 0.495 | 0.536 | 0.570 |

  Same construction, same profile over k; the late-k excess of the NAVSIM rows is their slower logs (less distance covered by decision k),
  not a different rule. Kept as is.
- The pilot is chained behind the identity check (`AFTER=<ident-cmp id> rf_v2.sh pilot`): it starts only if the 50-step flags-off run is
  bit-identical to the code before route v2 (`6551f10b`).

## Amendment 2 (after readings A / B, decision 299; before any pilot read, 2026-10-15 00:35)

Decision 299: A (near-route gap -> shortcut / over-turn) unsupported; B (lane offset before the entry) inconclusive, unsupported at the
entry; required lane changes essentially never done (map-validated sites: PAI 4 reached / 1 done, held 4 / 0); and a representation defect:
while the ego is short of the memory's start and the route bends in between, the projection-anchored slots put far points into near slots
(the start shown a median 4.5 m too near, p90 15.7 m).

- Deviation from section 2: `--rm-order` is no longer the A hypothesis (A is refuted) but the fix of that slot defect, and it changes
  accordingly. D1 is withdrawn: the ego's arc on the merged memory is now PROGRESS-anchored, s_e = arc(newest message's first waypoint) -
  42.105 m + chord driven since (AlpaSim resamples 20 points every 80 / 19 m from its projection of the ego and drops those < 40 m, so the
  first kept waypoint is 42.105 m of route arc ahead; PAI drive.jsonl shows 42.1057). The 40 m horizon cut is dropped with A (its only
  reason was to stop aiming at far points): the ordered layout keeps the 40 slots (-6 .. +72 m) at their true arc + the d_turn / trav
  scalars (DIM_O 203). The head conversion keeps every tensor (scalar map at 0).
- One builder (b of the request): `rmem.memory_o` is called by the served `Tracker(order=True)`, `rm_build.py order` (NAVSIM rows rebuilt
  from the messages at frames i - k .. i; pool rows from regen at the decision anchors; yaw-rate rows = the source block moved into the
  perturbed frame and re-scored) and `rf_cfpool.py build --order` (the counterfactual / stay-lane rows); `rm_build.py served` compares the
  served ordered memory with the offline rebuild (points, d_turn, trav) for any ordered checkpoint that is read. The trainer's hinge
  target, negatives and over-turn reference use the ordered block when `--rm-order` is on.
- New gate G-slot (layout gate, model independent; `rf_v2gate.py slotmem` on Tokyo / box + `slot` on the Mac): the 53 turn zeros of
  results/progress/zeros.csv with the served first slot ahead (first_off_m3 > 0), at the decision 3 s before the failure; error = the
  arc where the layout puts the memory's start minus its true arc (station of the merged memory's first point on rf_tax's rebuilt route
  minus the anchor's station); line: median |error| <= 0.5 m. Read before any pilot (it needs no model), results/v2gate/slot.md:
  projection (served today) median -4.51 m, |.| p90 15.67, 26 / 53 >= 5 m (decision 299 reproduced); ordered +0.00, p90 0.06, 0 / 53:
  PASS. de9db5dc: true start 39.7 m ahead (d299's 41.1 was the station of the first valid point at t, 0.1 s later), projection 12.5 m
  (first slot 14), ordered 39.7 m (first slot 40). 2466266e is not a gap case (memory starts 25 m behind the ego; both layouts within
  0.21 m; results/v2gate/slot_2466266e.json). The served window was reproduced from the logged messages on 53 / 53 (max |diff| <= 2e-4).
- Arms (coordinator, 2026-10-15): V2a = `--rm-order --lc-cf`; V2b = V2a + `--lane-lam 10 --lane-dz 0.5 --lane-neg 0.2` (negatives and
  over-turn guard). `--lc-cf` is on in every arm (lane changes certain untrained). The lane hinge is the second arm, not the main one: its
  only support is decision 299's exploratory lead (failed turns 0.48 m on the side away from the turn at entry - 20 m vs completed +0.06).
  Two pilots (RV2aP / RV2bP-L18-s0: 1 000 steps, seed 0, batch 64, init RSG-L18-s0), all seven gates per arm; full round = 2 arms x 2
  seeds, gentle schedule, on the base the coordinator names, held 460 paired vs the base and vs RSG-L18-s<seed> (H1-H3 unchanged).
- Cost: pilots + gates ~0.3 card-h; full round 4 x 2 500 steps ~0.5 + 4 gates ~0.4 + 4 held reads ~2.8 = ~3.7 card-h.

## Amendment 3 (pilot reads, 2026-10-15 01:15; the lines of section 3 unchanged)

Bit identity (every flag off, 50 eager steps of RSG-L18-s0's command, this code vs `6551f10b`): 109 + 97 tensors identical, dev ADE 0.5284
-> 0.5320 in both (results/v2gate/ident.json, ident_dev.txt). Pilots: 1 000 steps, seed 0, batch 64, init RSG-L18-s0; gate vs RSG-L18-s0
(results/v2gate/RV2*.md). Fixes made between reads (code, not lines): the ordered head's forward is picked by input width (first a / b run
crashed at step 0); negatives and the gate's straight fakes also on near-gap rows (with the +0 m slot required only ~15 % of rows
qualified: 2 % negatives, 0 dev curve rows; the straight fake runs from the ego along its heading there).

| pilot (flags) | G-rely >= 0.01 | G-swap >= 0.50 / 0.70 | G-ST <= 0.10 (shift / straight) | G-LC >= 0.50 | G-real <= 0.5385 | G-over <= 0.0067 | G-slot <= 0.5 |
|:--|--:|:--|:--|--:|--:|--:|--:|
| base RSG-L18-s0 | 0.0248 | 0.176 / 0.691 | 0.004 / 0.023 | 0.067 | 0.5285 | 0.0067 | 4.51 (old) |
| RV2aP (order + lc, lc-lam 10) | **-0.016** | **0.291** / 0.743 | 0.022 / **0.124** | **0.217** | **0.5658** | 0.0067 | 0.0002 |
| RV2bP (+ lane 10, narrow negatives) | **-0.035** | **0.379** / 0.798 | **0.124 / 0.154** | **0.233** | **0.5806** | 0.0056 | 0.0002 |
| RV2bP2 (+ lane 10, wide negatives) | **-0.028** | **0.331** / 0.771 | 0.071 / 0.076 | **0.213** | **0.5745** | 0.0056 | 0.0002 |
| RV2cP (no flag, control) | 0.0228 | **0.200** / 0.751 | 0.003 / 0.023 | **0.070** | 0.5264 | 0.0067 | (old) |
| RV2oP (order only) | 0.0212 | **0.206** / 0.739 | 0.003 / 0.021 | **0.075** | 0.5277 | 0.0067 | 0.0002 |
| RV2lP (lc only, lc-lam 10) | **-0.014** | **0.310** / 0.749 | 0.023 / **0.107** | **0.179** | **0.5635** | 0.0067 | (old) |
| **RV2aP3 (order + lc, lc-lam 1)** | 0.0179 | **0.227** / 0.746 | 0.005 / 0.027 | **0.098** | 0.5280 | 0.0067 | 0.0002 |
| **RV2bP3 (+ lane 1, wide negatives)** | 0.0180 | **0.215** / 0.746 | 0.005 / 0.016 | **0.092** | 0.5284 | 0.0067 | 0.0002 |

Reading: the ordered layout costs nothing (oP = cP on every gate) and fixes the slot defect. The lane-change pairs at weight 10 (D6) are
what breaks G-real / G-rely (lP alone: dev ADE +0.035, masking the memory then helps): the stay-lane hinge at 10 per row-metre on 4 of 64
rows moves the trunk (ade without memory 0.5433 -> 0.5495, control 0.5409). At weight 1 every guard gate passes (a3 / b3) but the lane-change
swing on the 26 dev lc rows is 0.09-0.10 m (line 0.50; at weight 10 0.18-0.23). G-swap fails in every arm, including the control (0.20),
as expected: no arm trains junction counterfactual rows (RB-F did, 0.55); the R-CF stop line (G-swap < 0.20) is not crossed (0.215-0.227).
The lane hinge at weight 10 makes the plan follow a shifted route (0.12 m) until the negatives cover the gap rows (0.07); at weight 1 it is
inert offline (b3 = a3 on every gate).

Decisions (recorded, not launched): D6 is replaced by weight 1 for `--lc-lam` and `--lane-lam` in `rf_v2.sh` (the only setting that passes
G-real); with it G-LC and G-swap fail, so by section 3 neither arm qualifies for the full round as pre-registered. Proposals for the
coordinator: (1) launch V2a as the slot-defect fix only (the lane-change term is inert at weight 1; G-LC fails), or (2) one more 1 000-step
pilot each at `--lc-lam 3` and with the stay-lane hinge's gradient kept off the trunk (branch + adapter only), ~0.1 card-h each, before
deciding on `--lc-cf`; (3) G-swap needs the junction counterfactual pool (on1_CF j0 / j1) in the arm, which no arm has. GPU used tonight
~0.85 card-h (2 identity runs, 10 pilot trainings incl. two crashed at step 0, 10 gate reads).

## Amendment 4 (coordinator, before these reads, 2026-10-15 01:25): lane-change weight / gradient pilots, `--rm-latfeat`, G-LC 0.30

- Pilots (1 000 steps, seed 0, batch 64, init RSG-L18-s0, arm a = `--rm-order --lc-cf` + the flags; all seven gates each):
  (i) `--lc-lam 3` (gradient through the whole trainable net, as before); (ii) `--lc-lam 3 --lc-adapter-only` (the stay rows' term
  enters through torch.autograd.grad w.r.t. the adapter + route branch only, added to their .grad after the main backward; the shared
  network gets nothing from the stay rows); (iii) (ii) at `--lc-lam 10`; (iv) (ii) + `--rm-latfeat`.
- `--rm-latfeat` (coordinator's explanation: a 3.5 m parallel shift is a small change of a 40-point polyline for a zero-init decoder,
  a junction alternative ~85 m): the route's signed lateral offset (left +, ego heading frame) at 10 / 20 / 30 / 40 m of route arc =
  the ordered slots' y / 10 (clip +-4) and 4 validity bits (an invalid slot -> 0, 0), through a zero-initialised linear map added to the
  branch's ego token. Computed inside the head from the fed ordered block (`rmem.lat_feats`), so no feature file changes and the served
  numbers are the trained ones; needs `--rm-order`; sidecar head `latfeat`.
- G-LC line relaxed from 0.50 to 0.30 m (the registered 0.50 exceeds what a 2 s window shows of a 4 s lane change). An LC setting
  qualifies for a third arm only if G-LC >= 0.30 AND G-rely, G-real and G-ST all pass. G-swap is reported, not required (no arm has
  junction counterfactual rows; Amendment 3).

## Amendment 5 (Amendment 4's pilot reads, 2026-10-15 01:30)

| pilot (arm a + ...) | G-rely >= 0.01 | G-swap (reported) | G-ST <= 0.10 (shift / straight) | G-LC >= 0.30 [CI] | G-real <= 0.5385 | G-over | G-slot | qualifies |
|:--|--:|:--|:--|:--|--:|--:|--:|:--|
| (a3) lc-lam 1 | 0.0179 | 0.227 / 0.746 | 0.005 / 0.027 | 0.098 | 0.5280 | 0.0067 | 0.0002 | no (G-LC) |
| (i) lc-lam 3 | 0.0147 | 0.253 / 0.756 | 0.010 / 0.048 | 0.147 [0.03, 0.60] | 0.5335 | 0.0067 | 0.0002 | no (G-LC) |
| (ii) lc-lam 3, adapter-only | 0.0175 | 0.260 / 0.756 | 0.011 / 0.044 | 0.136 [0.03, 0.58] | 0.5313 | 0.0067 | 0.0002 | no (G-LC) |
| (iii) lc-lam 10, adapter-only | **-0.0019** | 0.318 / 0.754 | 0.026 / **0.126** | 0.231 [0.13, 0.64] | **0.5569** | 0.0056 | 0.0002 | no |
| (iv) (ii) + `--rm-latfeat` | 0.0161 | 0.263 / 0.756 | 0.011 / 0.045 | 0.143 [0.03, 0.59] | 0.5322 | 0.0067 | 0.0002 | no (G-LC) |

No lane-change setting qualifies for a third arm. G-LC moves with the weight (0.10 -> 0.14 -> 0.23 at 1 / 3 / 10) while G-real and
G-rely degrade with it; detaching the shared network halves the dev-ADE cost at weight 3 (0.5313 vs 0.5335) but not at weight 10
(0.5569; the adapter's bias reaches every row, so the damage does not need the trunk). The lateral-offset features change nothing at
1 000 steps (iv = ii within noise; their zero-init map barely moves). G-LC rests on 26 dev rows (CIs 0.03-0.60). Training-side swing on
the pairs at step 1 000: 0.12 / 0.23 / 0.22 / 0.55 / 0.22 m. GPU tonight ~1.15 card-h in total (this amendment ~0.3).

## Amendment 6 (main session, 2026-10-15 01:40, before the full round)

- `--lc-cf` is dropped from both arms: no weight passes G-LC (>= 0.30) while G-rely / G-real / G-ST pass (Amendment 5), so the term
  would be in the recipe without a working mechanism. Lane changes stay unsolved by counterfactual pairs; the on-policy lane hinge of arm b
  is the only term left that can pull the car toward the route's lane.
- Full round: arm a = `--rm-order` (slot-defect fix only, the control for b); arm b = `--rm-order --lane-lam 1 --lane-dz 0.5 --lane-neg 0.2`.
  2 seeds each, gentle schedule, base = the lr x steps pilot's winner (RSG-L18 if flat), held 460 paired vs the base and vs RSG-L18-s<seed>,
  rf_follow turn completion on the 24 held turns. Lines H1-H3 unchanged.

## Amendment 7 (coordinator, 2026-10-15 02:15, before any arm-c read): arm c = ordered memory + junction counterfactual rows

- Why: G-swap is 0.18-0.26 in every arm and pilot (Amendments 3, 5) because nothing trains junction route switching; RB-F reached 0.55 with
  the on1_CF rows, but as a from-scratch base with a G-real cost; a gentle round on RSG-L18 with them is untested.
- Arm c = `--rm-order` + the on1_CF junction rows (classes j0 / j1, 12 176 rows; lane-change rows excluded), served as RB-F served them:
  route hinge to the counterfactual label (on_route, dead zone 0.5 m, lateral only, poses 1-2 s weight 2) + road-grid hinge, no imitation,
  no agent hinge. One pool (rm_train takes one): `rf_v2cpool.py` -> on1_L18RB0J = on1_L18RB0 (RSG's S rows, route hinge off on them) +
  the CF junction rows (their ordered memory from rf_cfpool build --order), tab `pool` s / cf; per batch of 128: 25 pool draws with
  `--on-weight pool=cf:1.3213`, i.e. in expectation 13 S rows (= RSG) + 12 CF rows (= RB-F's on-k 12); normal rows 90 instead of 102.
  `--on-lon on_route` at its default weight (10 x ho_w 3, / n_imit; RB-F's). Pilot batch 64: on-k 13 (7 S + 6 CF).
- Rule: the 1 000-step pilot (RV2cPJ-L18-s0; seed 0, init RSG-L18-s0) must pass G-swap >= 0.50 / >= 0.70 AND G-real, G-ST, G-over,
  G-rely. Pass -> arm c for 2 seeds (`rf_v2.sh train c RSG-L18-s<seed> <seed>`, PRIO 10, behind b) with the same held chain, in the
  final report. Fail -> report the table and stop.
- Unrelated fix the same night: `rm_train.lane_prep` built the lane-change tables (and their ordered map) whatever the flags; arm b of
  Amendment 6 (`--lane-lam` without `--lc-cf`) crashed at the first line that read them; they are now built only with `--lc-cf`
  (the flags-off path never calls lane_prep, so the identity of Amendment 3 stands).

## Amendment 8 (reads, 2026-10-15 02:20)

- Arm c pilot (RV2cPJ-L18-s0, results/v2gate/RV2cPJ-L18-s0.md; pool on1_L18RB0J = 17 429 S + 12 176 junction CF rows, 13 pool draws per 64
  with ~6 CF, normal rows 45): G-swap 0.626 / 0.961 PASS (base 0.176 / 0.691), G-rely 0.031 PASS, G-ST 0.008 / 0.062 PASS, G-over 0.0056
  PASS, G-slot PASS, **G-real 0.5521 FAIL** (line 0.5385; base 0.5285; without memory 0.5719), G-LC 0.109 (not required). By Amendment 7's
  rule arm c stops: not launched. The junction rows teach route switching on a gentle round too, at RB-F's dev-ADE cost (+0.024 here).
  Not run, a candidate if wanted: the same arm at a lower `--on-lon-lam` (10 -> 3) or a smaller CF share, read on the same gates.
- Arm a held 460 / 18 (results/v2held/: held_raw.txt, confirm.md, follow.md; report job with RSG-L18-s0 / s1): zeros 33 / 33 (RSG 34 / 34),
  paired 452 vs RSG-L18-s<seed>: zeros -1.0 [-7.0, +4.9] / -1.0 [-5.6, +3.3], at-fault +0.0 [-4.9, +4.9] / -1.0 [-4.4, +2.3], mean score
  +0.000 / +0.003, GT -0.013 / -0.020 m; turn completion (24 held turns) 75.0 % / 62.5 % vs 66.7 % / 58.3 % (+8.3 [+0.0, +20.8] /
  +4.2 [+0.0, +11.1] pp). H1 passes (CI upper +4.9 / +3.3 zeros, +4.9 / +2.3 at-fault, line <= +5); H2 (>= base + 3 turns, seed mean) not
  met (+2 / +1 turns). G-serve of the ordered memory on the held chunk 0: 1 368 / 1 368 decisions per seed, masks equal, max point
  difference 1.6e-4 m, d_turn equal 100 %, trav within 5e-4 m.

## Amendment 9 (coordinator, 2026-10-15 02:25, before these reads): arm c weight / share variants

Three 1 000-step pilots of arm c (seed 0, init RSG-L18-s0, batch 64), same seven gates: (i) `--on-lon-lam 3` (route-hinge weight 3, CF
share as in Amendment 7: on-k 13, `pool=cf:1.3213`, ~6 CF of 64); (ii) weight 10, CF share halved: `--on-k 10 --on-weight pool=cf:0.6607`
(~3 CF + 7 S of 64; at batch 128: on-k 19 = 13 S + 6 CF); (iii) weight 3 and share halved. A variant qualifies if G-swap >= 0.50 / >= 0.70
AND G-real <= base + 0.01 (0.5385) AND G-ST / G-over / G-rely pass; if several, the highest G-swap ratio. One qualifies -> arm c with it,
2 seeds (`rf_v2.sh train c RSG-L18-s<seed> <seed> <variant flags>`, PRIO 10, held chain + served check), into the final combined report;
none -> report and stop.

## Amendment 10 (Amendment 9's reads, 2026-10-15 02:55)

| arm c pilot | G-swap >= 0.50 / 0.70 | G-real <= 0.5385 | G-rely | G-ST | G-over | qualifies |
|:--|:--|--:|--:|:--|--:|:--|
| Amendment 7: weight 10, ~6 CF / 64 | 0.626 / 0.961 | **0.5521** | 0.031 | 0.008 / 0.062 | 0.0056 | no (G-real) |
| (i) weight 3, ~6 CF / 64 | 0.554 / 0.956 | **0.5408** | 0.029 | 0.007 / 0.048 | 0.0056 | no (G-real, +0.0023 over) |
| (ii) weight 10, ~3 CF / 64 | 0.544 / 0.922 | **0.5404** | 0.025 | 0.007 / 0.052 | 0.0056 | no (G-real, +0.0019 over) |
| (iii) weight 3, ~3 CF / 64 | **0.411** / 0.753 | 0.5324 | 0.022 | 0.006 / 0.039 | 0.0067 | no (G-swap ratio) |

No variant qualifies; arm c is not launched (rule of Amendment 9). The trade is smooth: halving either the weight or the share keeps
G-swap above 0.54 and cuts the dev-ADE cost from +0.024 to +0.012; halving both keeps the cost at +0.004 but the switch at 0.41.

## Amendment 11 (combined held read of arms a and b, 2026-10-15 03:00; results/v2held/ now holds both arms)

| held 460 / 18 | zeros | paired zeros vs RSG-L18-s<seed> [95 % CI] | at-fault vs RSG | GT (m) | turns completed / 24 | turns vs RSG (pp) |
|:--|--:|:--|:--|--:|:--|:--|
| RV2a-L18-s0 | 33 | -1.0 [-7.0, +4.9] | +0.0 [-4.9, +4.9] | 1.122 | 18 (75.0 %) | +8.3 |
| RV2a-L18-s1 | 33 | -1.0 [-5.6, +3.3] | -1.0 [-4.4, +2.3] | 1.124 | 15 (62.5 %) | +4.2 |
| RV2b-L18-s0 | 32 | -2.0 [-6.7, +2.9] | -1.0 [-5.2, +3.5] | 1.097 | 16 (66.7 %) | +0.0 |
| RV2b-L18-s1 | 30 | -4.0 [-9.6, +1.0] | -1.0 [-6.8, +4.8] | 1.131 | 17 (70.8 %) | +12.5 |
| RSG-L18-s0 / s1 | 34 / 34 | | | 1.135 / 1.144 | 16 / 14 | |

H1 passes for both arms (every CI upper bound <= +5). H2 (seed mean >= base + 3 turns) is not met: arm a +2 / +1 turns (mean +1.5),
arm b +0 / +3 (mean +1.5). Offline gates of the trained checkpoints (vs their base): RV2b-L18-s0 / s1 pass G-rely / G-ST / G-real /
G-over / G-slot, fail G-swap (0.18 / 0.19) and G-LC (0.08), as the pilots predicted. G-serve of the ordered memory: 1 368 / 1 368
decisions per seed for both arms (max point difference 5.3e-4 m). Next registered step (not run here): the public 1491 guard (H3).
