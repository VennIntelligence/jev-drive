# APPROACH COURSE: the policy's own approach states pulled toward the human's entry state (pre-registration)

2026-10-15 morning, written before any pilot or training read (the row labeller was smoke-run on 40 nuPlan scenes for bugs only; no
model was evaluated). Design fixed by the main session and approved by the user (research/next-round/turn-night.md sections 4-5 and the
09:40 line). Context: decisions 293 (human entry speed 4.6 m/s median, deceleration from ~26 m before the entry), 295, 296, 297, 298
(log handover to the entry completes 11 / 26 exam turns, the policy's own approach almost never), 299 (failed turns 0.48 m on the side
away from the turn 20 m before the entry; late-wide is the largest failure class, 34 / 72), 301 (amplitude is not the lever; 5 scenes
enter at > 10 m/s), 302 (arms a / b: harmless, +1.5 turns, below the +3 line). P4's rows had 0 rows before 20 m: the approach was never
taught. Code: `experiments/onpol_nuplan/lib/on_approach.py`, `experiments/route_follow/scripts/rf_apppool.py`, `rf_appgate.py`,
`rf_app.sh`, `experiments/route_mem/scripts/rm_train.py --app-lam`.

## 1. Rows

| source | what | rows used |
|:--|:--|:--|
| nuPlan | `on1_L18RSG`: RSG-L18-s0's own 18-decision rollouts on the 1 020 train scenes (decision 286 Amendment 5, the RS2G rows) | approach-window rows |
| PAI | `on1_PAIonp`: the RSG-PAI line's train160 rows (RB0-F-s0's own PAI rollouts, rf_paionp.py pack; no exam / control / hard_train scene) | approach-window rows |

- Turn row: the human's logged path has a turn >= 45 deg ahead and the ego's foot on that path is 0 < d_e <= 40 m before the turn's entry.
  The turn and entry rule is rf_pai_follow.find_turn (rf_shape.py / rf_entryspeed.py / decision 293), with the threshold at 45 deg: 30 m
  window, net heading change >= 45 deg, change over [a - 10, a + 40] m >= 0.75 x 45 deg, entry / exit where 10 % / 90 % of the change is
  reached. Only the first entry ahead counts.
- Straight negative: no turn entry in (-10, +80] m of the foot, the human path within 10 deg of the foot's heading over the next 60 m, and
  the route label equally straight. 20 % of the approach rows (negatives = 0.25 x the turn rows of the source, drawn with rng 1).
- Kept only if: ego speed >= 1 m/s (P4's G-crawl: lateral terms on a standing car move nothing real); |ego offset to the human path| <= 3 m;
  the row's route label ok; for turn rows, the human's entry point <= 2.5 m from the route the policy is given (the route takes the
  human's turn); nuPlan rows are the pool's kept rows; PAI rows before any zero event.
- Dev split for the pilot gate: logs with crc32(log) % 100 < 15 (nuPlan log unit, PAI scene) are held out of the pool entirely.
- If the train turn rows of both sources together are < 1 500: add log-handover states (labelled as such, an amendment before use).

Labels (rf_apppool.py label; per row, in the row's own rear-axle frame):

- H: the human's logged path at arcs -10 .. +80 m every 1 m from the ego's foot (nuPlan: on_rows' path G rebuilt from the navsim log
  exactly as the pool was built, gate: the foot reproduces meta s0 / e0 to <= 1e-6 m, 40-scene smoke 5.7e-10; PAI: the cache `path`),
  its heading Hh, the human's speed when it first passed each point Hv; d_e = the entry's arc from the foot.
- The human's entry state relative to the route centreline (lateral offset, heading, speed) is computed and reported per row; it is the
  point of H at d_e, so H carries it.
- l3 (the speed term applies): the human's speed >= 2 m/s over the whole 26 m before the entry (rf_entryspeed's VH_MIN). A human who stopped
  there (queue, red light) gives no speed target: time-dependent traffic is not a turn speed.
- Negatives: H = the route centreline (route.npz, the lane centre), no speed; d_e = a virtual entry drawn from the source's turn-row d_e.

## 2. Losses (on_approach.py, an `--on-lon` module; all lateral-only or longitudinal-only)

Each plan pose (rear axle, 0.5 .. 4 s) is projected on H: station s from the ego's foot and signed lateral e; e0 = the ego's own offset.

| term | what | dead zone | gradient |
|:--|:--|:--|:--|
| L1 | the plan at the entry line (first plan point with station >= d_e, linear between the bracketing poses): \|e\| and the plan segment's heading vs H's heading at d_e; entry beyond the plan's 4 s: the end pose vs H at its own station | 0.3 m, 5 deg | lateral (shape-only) |
| L2 | the 1.0 / 1.5 / 2.0 s poses: \|e_k - e0 (1 - clip(s_k / d_e, 0, 1))\|, mean of three, x 2 (MPC horizon, decision 284): from where the policy is, the straight blend of its offset into the human's entry state | 0.3 m | lateral (shape-only) |
| L3 | each 0.5 s plan segment's speed at its distance-to-entry d in [0, 26] m vs the human's speed at the same station, one-sided relu(v - v_h - 0.5), mean over the segments in range; l3 rows only | 0.5 m/s | longitudinal (along each pose's own heading) |

- Negatives: L1 / L2 against the lane centre (no drift learning), no L3.
- Weight W: `rm_train.py --app-lam W` = `--on-lon on_approach --on-lon-lam 10 W`: each term summed over the batch's pool rows / n_imit x
  ho_w (3) x 10 W. W = 1 is the per-row weight of RB-F's route hinge and of the S agent hinge (agent-lam 10), W = 3 that of RBSG's 30.
- S hinges (agent, road raster) stay on every pool row exactly as in RSG-L18 (`--on-mode self`); PAI rows have no road raster (as in the
  RSG-PAI line). No imitation term on any pool row.
- Flags off: no operation is added (`--app-lam 0` default). Bit identity: 50 eager steps of RSG-L18-s0's command with this code and with
  the code before the approach course (`107e2002`), every trained tensor equal (on_train.py ident) and equal dev ADE logs.

## 3. Arms and the pilot gate

Pool `on1_APP` = on1_L18RB0 (RSG-L18's S rows, unchanged, 17 429) + the train approach rows of both sources; `--on-weight` puts ~7 approach
rows into every batch of 128 next to RSG's 13 S rows (on-k 20; the normal rows shrink by 7). Base RSG-L18-s<seed>, gentle schedule
(2 500 steps, lr x 0.1, warm-up 1, rm-lr 3e-4, batch 128) = RSG-L18's recipe.

- A1 = approach rows + `--app-lam W`.
- A2 = A1 + arm b's lane hinge `--lane-lam 1 --lane-dz 0.5 --lane-neg 0.2` (no `--rm-order`).
- 2 seeds each (APP1-L18-s{0,1}, APP2-L18-s{0,1}).

Pilot (1 000 steps, seed 0, batch 64, on-k 10, init RSG-L18-s0), W in {1, 3}; a control pilot with W = 0 on the same pool is read as a
diagnostic (what the extra rows do without the terms), not a gate. Gate vs RSG-L18-s0 (`rf_appgate.py` + `rf_v2gate.py`):

| gate | rows | line |
|:--|:--|:--|
| A-lat | dev turn rows (held-out logs) | mean \|lateral to the human path\| of the plan at the entry line (end pose if not reached) drops >= 30 % vs the base |
| A-speed | dev turn rows with the speed term and a plan segment in the 26 m window | mean speed excess over the human (m/s, no dead zone) drops >= 30 % |
| A-neg | dev straight negatives | end-pose \|lateral to the lane centre\| not above the base + 0.10 m |
| G-real | NAVSIM dev rows | ADE <= base + 0.01 m |
| G-ST | dev straight rows +-3.5 m shifted memory; curve rows straight memory | median lateral move at 1.5-2 s <= 0.10 m, both |
| G-over | dev junction rows | over-turn share <= the base's |
| G-rely | dev junction rows | memory reliance (ADE masked - on) >= 0.01 m |

W = the largest weight that passes every gate; none passes -> report and stop (no arm is trained). The drops carry log-clustered CIs;
the lines read the point estimates.

## 4. Closed-loop lines (after `train`)

- Held 460 / 18 decisions, paired vs RSG-L18-s<seed> (held.py, ol_confirm.py, rf_follow.py as in rf_v2.sh report):
  - H1 no harm: zeros and at-fault zeros, log-clustered 95 % CI upper bound <= +5, each seed.
  - H2 turns: completion on the 24 held turns >= base + 3 (2-seed mean; RSG 16 / 14).
  - Reported: entry speed and entry lateral offset (signed toward the inside) of the 24 held turns per label (rf_follow decisions at the
    entry station, rf_entryspeed's cross_speed), paired by scene.
- PAI 281 on Tokyo (rf_v2pai.sh run / read, gains 20) for the better arm's better seed (H2 turns, then H1 zeros): strict-36 completion
  >= 10 (RSG 5, RV2b-s1 7); exam-72 zeros not above RSG's 38 + 5; the decision-293 shape gate count reported.
- Public 1491 / 10 decisions (rf_app.sh guard) only if H1 and H2 pass: zeros and at-fault vs RSG-L18, CI upper bound <= +5.

## 5. Cost and stop rules

Rows ~0.5 card-h of CPU (labelling ~5 min on 48 cores, pack ~10 min); identity 2 x 50 steps ~0.15 card-h; pilots 3 x (1 000 x 64 +
gates) ~0.3 card-h; 4 trainings x 2 500 steps ~0.5 card-h + gates 0.4; 4 held reads ~2.8 card-h; PAI 281 ~2.3 h on both Tokyo cards;
public guard ~2 card-h per checkpoint. Total ~7 card-h on the box.

- Identity fails: nothing runs until fixed.
- Pilot: no weight passes -> stop and report.
- H1 fails on either seed -> no PAI read, no public guard, not adopted (whatever H2 says).

## Amendment 1 (rows built, identity read; before any pilot read, 2026-10-15 10:30)

- Bit identity (every flag off, 50 eager steps of RSG-L18-s0's command, this code vs `107e2002`): 109 net + 97 adapter tensors identical,
  dev ADE 0.5284 -> 0.5320 in both (`$DATA_DIR/runs/route_follow/approach/ident.json`, `ident_dev.txt`). PASS.
- Route consistency (section 1) by geometry, not by the entry point: the PAI route labels (on1_PAIonp route.npz) carry the 40 m near gap
  and arc offsets shifted from the ego's foot, so "the human's entry point <= 2.5 m from the route" rejected 478 PAI turn rows whose route
  polyline lies on the human path. Now: the route's valid points within 15 m of the human's entry point (else within 60 m of the ego) have
  a median distance to the human path <= 2.5 m (rf_apppool.route_check). PAI turn rows 430 -> 769, nuPlan 2 583 -> 2 628.
- Dev split and clusters by the nuPlan log file, not the log unit (the unit split left 4 dev clusters). PAI: the scene.
- Counts (`results/approach/rows.md`): train turn rows 3 091 (nuPlan 2 379 in 300 scenes / 63 logs; PAI 712 in 17 scenes), 40-20 m 1 075,
  20-0 m 2 016, speed term on 1 620; negatives 773; dev 306 turn rows (nuPlan 249 / 11 logs, PAI 57 / 2 scenes) + 313 negatives.
  >= 1 500: no log-handover supplement. Pool on1_APP = 17 429 S rows + 3 864 approach rows, --on-weight app=turn:2.4288,app=neg:2.4288
  (~7 approach rows per batch of 128; ~3.5 of the pilot's 64).
- Human entry state of the train turn rows (median): nuPlan 4.0 m/s, +0.14 m toward the inside of the route, +2.5 deg; PAI 3.3 m/s,
  -0.52 m, +1.8 deg (PAI lateral is to the PAI route label, see the near-gap note).
- Held 460 caveat (read from RSG-L18-s0's stored rf_follow, before any new read): the 24 held turns start a median 5.0 m (q25 3.8, q75 7.0)
  before the entry; the approach course has at most a few metres of runway there. H2 stays as registered; the PAI 281 read (rollouts from
  well before the entry) is where the approach is exercised.
- Pilots launched: APP1P1 / APP1P3-L18-s0 (W 1 / 3) and the control APP1P0-L18-s0 (W 0, same pool and draws).

## Amendment 2 (coordinator, 2026-10-15 ~10:45, before any pilot, held or PAI read): where H2 is read

Held 460 rollouts start a median 5 m before the turn entry (Amendment 1), so held cannot test the approach.
- Held 460 keeps H1 (no harm, unchanged) and the descriptive turn table (completion on the 24 held turns, entry speed and lateral via
  rf_appheld.py). It no longer gates anything about turns.
- H2 moves to PAI 281 (rf_v2pai.sh run / read, gains 20, paired against the stored r281/rsg = RSG-L18-s0). Primary subset: the strict
  turns with >= 20 m of runway before the entry, i.e. the strict turns rf_laneoff.py (decision 299) could define at entry - 20 m
  (results/laneoff/turns.csv, set pai, kind strict, point m20, why ok). That is **22** turns, not 19: decision 299's "19" counted only
  the failed ones (19 failed + 3 completed). The list is `results/approach/pai_runway22.txt` (all 22 are in strict36). RSG-L18-s0
  completes **3 / 22** there (132ae6cb, 382a7247, d90fe123) and 2 / 14 of the other strict turns (3c2822d9, 8a143cb2).
  - H2 line: completion on the 22 >= RSG + 3 (>= 6 / 22).
  - No loss on the other 14 strict turns (completion >= 2 / 14).
  - Exam-72 zeros and the decision-293 shape gate count are reported. The old line "strict-36 >= 10" is reported too but does not gate.
- Scheduling. A1-s0 and A2-s0 each go to Tokyo (one card each, in parallel) as soon as that checkpoint exists, provided the pilot gate passed
  cleanly. The PAI run goes alongside the held read and is discarded if H1 then fails on held. Seed-1 PAI runs only for an arm that passes
  the primary line on seed 0.

## Amendment 3 (pilot read, 2026-10-15 ~11:40): the gate fails at every weight, stop

The gate is rf_appgate.py + rf_v2gate.py vs RSG-L18-s0, 306 dev turn rows (13 logs) and 313 negatives
(`results/approach/gate/APP1P{0,1,3}-L18-s0.md`):

| pilot | A-lat drop | A-speed drop | A-neg | G-real (line 0.5385) | G-ST | G-over | G-rely | all |
|:--|--:|--:|--:|--:|:--|--:|--:|:--|
| W 1 | -0.023 [-0.33, +0.23] | -0.024 | -0.009 PASS | 0.5490 FAIL | 0.004 / 0.026 PASS | 0.0056 PASS | 0.0196 PASS | FAIL |
| W 3 | -0.083 [-0.47, +0.21] | -0.039 | -0.018 PASS | 0.5655 FAIL | 0.007 / 0.034 PASS | 0.0045 PASS | 0.0128 PASS | FAIL |
| control W 0 (diagnostic) | -0.001 | -0.084 | +0.001 | 0.5286 | 0.003 / 0.020 | 0.0067 | 0.0216 | (no terms) |

- Base on the dev rows: entry lateral 0.256 m, heading error 2.9 deg, speed excess 0.006 m/s.
- No weight passes, so by section 5 no arm is trained and there is no held or PAI read.
- The gate reproduces the served plans: the dev nuPlan base lateral is 0.267 m whether computed from the stored served plans or from the gate.

Diagnostic, not a gate (`*_trainrows.md`, `rf_appgate.py --rows train`, 2 000 of the pool's own rows per source):
- The pilots do fit their training rows.
  - Entry lateral 0.685 -> 0.492 / 0.481 m (-28 / -30 %).
  - Speed excess 0.532 -> 0.352 / 0.320 m/s (-34 / -40 %).
  - On the PAI rows the speed excess goes 1.81 -> 1.14 / 1.04 m/s, and the lateral 1.06 -> 0.62 m.
- So the terms are learnt but do not transfer to held-out logs, and they cost NAVSIM dev ADE. The held-out dev draw is easy: lateral
  0.27 m against 0.60 m on the nuPlan train rows, and the PAI dev has 2 scenes, only one with a speed read and no speed excess at all. A
  30 % drop was therefore hard to show there. This is a property of the split, recorded here, not a reason to move the line after reading.

## Amendment 4 — EXPLORATORY (coordinator, 2026-10-15 ~11:50, after the failed pilot gate; not a registered arm, not a candidate)

The prereg verdict stands: the pilot gate failed and there is no registered arm. One exploratory mechanism read follows, because the
offline dev draw cannot exercise the approach (base already 0.27 m / 0.006 m/s there; PAI dev = 2 scenes) and PAI closed loop is the only
place the approach is tested. G-real (+0.020 m at W 1) is the known harm proxy.
- **Training.** EXPLORATORY APP1X-L18-s0 = arm A1 at W 1, seed 0 only, 2 500 gentle steps on RSG-L18-s0 (rf_app.sh train with
  TAG_PREFIX=APP1X, PRIO 12). Gates are logged. The held 460 read gives H1 only.
- **PAI 281.** Run on Tokyo in parallel with the held read, without waiting for H1: rf_v2pai.sh run / read, gains 20, both cards, paired
  against r281/rsg.
- **Primary read on PAI.**
  1. Mechanism at the 22 runway turns (results/approach/pai_runway22.txt): entry speed; speed excess over the human from 26 m before the
     entry; signed inward offset at entry - 20 m (rf_laneoff / rf_entryspeed machinery). Each paired with a CI.
  2. Completion on the 22 (RSG 3) and on the other 14 (RSG 2); exam-72 zeros; the shape gate; spins; zeros by kind.
- **Reading.**
  - Speed excess and offset both within noise: the approach course as rows-plus-hinge is closed.
  - Mechanism moves but completion does not: that is a finding about which entry state is not enough.

### Amendment 4 reads — EXPLORATORY

- **Training.** APP1X-L18-s0: dev ADE 0.5284 -> 0.5600 (+0.032); junction reliance 0.015. Offline gate (`results/approach/gate/APP1X-L18-s0.md`):
  A-lat -15 %, A-speed +9 %, G-real FAIL.
- **Held 460 H1 FAILS** (`results/approach/held/confirm.md`, held18, 452 paired, 39 logs, vs RSG-L18-s0):
  - zeros 70 vs 26, +44.0 [+30.4, +58.3]
  - at-fault zeros 47 vs 21, +26.0 [+15.5, +36.9]
  - collision / offroad / corridor 6 / 41 / 25 vs 4 / 17 / 6
  - mean score -0.126 [-0.156, -0.098]
  - held.py on all 460: zeros 78 vs 34
- **Held turns 8 / 24 vs 16 / 24** (`held/entry.md`).
  - Entry speed unchanged: -0.02 [-0.18, +0.10] m/s.
  - Inward offset at the entry +0.17 [+0.09, +0.25] m.
  - Reading: the plan is pushed inward and leaves the road, the RBSG2 direction.
- **PAI 281** (mechanism read): below, when it lands.
