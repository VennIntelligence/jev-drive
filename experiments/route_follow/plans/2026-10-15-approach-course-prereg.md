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
