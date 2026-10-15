# RECOVERY-REWIND on the nuPlan @18 train turns (64), RSG-L18-s0, expert XP-L12-E46 (no training)

Prereg: [plans/2026-10-16-rewind-train64-prereg.md](../../plans/2026-10-16-rewind-train64-prereg.md). Amendment 1 stopped the first attempt at
the smoke: nuPlan sessions are not bit-identical across launches, because the renderer sends different frames. Amendment 2 is the user's
decision and was written before any of the reads below:
- the handover check is "within the identity's run-to-run spread";
- every session runs 30 decisions (15 s);
- units and k_ref come from the stored 18-decision rollouts;
- short is reported as its own outcome.

Code:
- [scripts/rf_rewind.sh](../../scripts/rf_rewind.sh) `train64`, [scripts/rf_rewind.py](../../scripts/rf_rewind.py) `n-*`;
- driver `experiments/alpasim/lib/sh30_driver.py` with `rule_expert.py`'s `JEV_EXPERT_FROM`;
- served tree 15e40af8, read 3ec3f5f9.

Files:
- [table_raw.md](table_raw.md): every table;
- [rewind_train64.csv](rewind_train64.csv): one row per unit, Delta and xp0;
- [read.json](read.json): every rollout;
- [tfail.json](tfail.json): units, k_ref, h;
- box `$DATA_DIR/runs/route_follow/rewind_train64/{smoke3,pilot10,rest54,all}`.

**Setup.**
- Serving is the held read's: sh30 driver, c0b_chain overrides, plain RSG-L18-s0. The trajdata cache is `after280_rw` (token + 14 s, 64 scenes).
- Units are the 64 strict turns of the train pool's stored 18-decision RSG-L18-s0 rollouts:
  - **A** = the 43 failed turns. k_ref is:
    - event: k_z (23 units);
    - off: the first decision > 2.5 m off the route after entry - 20 m (9 units);
    - short: decision 17 (11 units).
  - **B** = the 21 completed turns. k_ref = the decision of the first decision-293 shape violation on the stored 10 Hz track. Only 4 of the 21
    violate, all on entry speed, at k_ref 1-4. The other 17 are the "completed and human-like" control.
  - **C** = the expert from decision 0 on all units.
- The handover is h = k_ref - 2 Delta, kept only if h >= 1. The expert takes over the policy's own session.
- Policy = RSG-L18-s0 alone in the same 30-decision setting. It is therefore no longer 0 by construction.
- One scene, 8d06ea88 (A, short), never started a driver session under the long cache. So n = 63: A 42, B 21.

| stratum | Delta | units | expert completes [95 % CI] | expert c / e / o / s | policy c / e / o / s | paired expert - policy | within tolerance | gate pass / completed | rows after h (all; completed + gated) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A | 1 s | 41 | 30 = 0.73 [0.58, 0.87] | 30 / 11 / 0 / 0 | 10 / 23 / 8 / 0 | +0.49 [+0.31, +0.66] | 35 / 41 | 7 / 30 | 267; 48 |
| **A** | **2 s** | 41 | **33 = 0.80 [0.66, 0.93]** | 33 / 7 / 0 / 1 | 10 / 23 / 8 / 0 | **+0.56 [+0.36, +0.74]** | 40 / 41 | **15 / 33** | **345; 122** |
| A | 3 s | 40 | 31 = 0.78 [0.62, 0.90] | 31 / 8 / 0 / 1 | 10 / 22 / 8 / 0 | +0.53 [+0.33, +0.71] | 39 / 40 | 23 / 31 | 405; 230 |
| A-event | 2 s | 22 | 16 = 0.73 [0.52, 0.91] | 16 / 5 / 0 / 1 | 0 / 22 / 0 / 0 | +0.73 | 21 / 22 | 5 / 16 | 205; 47 |
| A-off | 2 s | 9 | 9 = 1.00 | 9 / 0 / 0 / 0 | 1 / 1 / 7 / 0 | +0.89 | 9 / 9 | 6 / 9 | 79; 50 |
| A-short | 2 s | 10 | 8 = 0.80 [0.50, 1.00] | 8 / 2 / 0 / 0 | 9 / 0 / 1 / 0 | -0.10 [-0.50, +0.20] | 10 / 10 | 4 / 8 | 61; 25 |
| B | 1 s | 1 / 4 | 1 | 1 / 0 / 0 / 0 | 1 / 0 / 0 / 0 | 0 | 1 / 1 | 1 / 1 | 16; 16 |
| B | 2 / 3 s | 0 / 4 | - | - | - | - | - | - | 0 |
| C (xp0) | - | 63 | 48 = 0.76 [0.65, 0.86] | 48 / 14 / 0 / 1 | 31 / 24 / 8 / 0 | | | 46 / 48 | |

Outcomes are completed / event / off / short (rf_follow). CI: bootstrap over logs (10 000). Rows = the expert's decisions at k >= h inside
[entry - 20, exit + 10] m before any zero (2 Hz). "Completed + gated" = rows of the units that completed and pass the gate: the label pool of
a recovery lesson.

**Registered lines.**
- **A PASS:** 0.80 [0.66, 0.93] at Delta 2 s against >= 0.6. Delta 1 / 3 s give 0.73 / 0.78.
- **B not evaluable:** 0 handovers at Delta 2 s, so it is FAIL by the rule.
- **C FAIL:** 0.76 [0.65, 0.86] against >= 0.8.
- Creep: v(h + 3 s) >= 2 m/s in 37 / 40, 38 / 41 and 38 / 40 rollouts.
- At-fault zeros after the handover: 11 / 11 / 11. The expert's events are 9 / 7 / 8 collisions and 2 / 0 / 0 offroad. Rear contact after h: 11 / 11 / 9.

**Handover check (Amendment 2).**
- Before h, 104 / 123 expert sessions are bit-identical to the identity. 115 / 123 are within the registered tolerance (plan <= 0.15 m,
  track <= 5 cm, v(h) <= 0.05 m/s).
- 6 of the 8 outliers are at Delta 1 s. The worst is 131c5de1: plan 3.9 m, v(h) 1.03 m/s.
- Identity replays inside the expert launches give the same score and outcome as `id` in 59 / 66.
- Restricted to within-tolerance units, A completes 27 / 35, 33 / 40 and 31 / 39. This changes nothing.

**Flags.**
- The identity runs 30 decisions on 59 / 63 scenes. The logs end earlier on four: 84ea5053 (15), b5895541 (26), 55475ba6 (28) and ef305e77 (22).
- The MTGS camera coverage ends at token + 9.6 s (median; >= 14 s on 1 / 63). After about decision 20 the policy sees frames rendered
  outside the reconstruction. So the policy column late in the run is confounded. The expert reads the map, so its column is not.
- At 30 decisions the identity changes its stored outcome on 12 turns: 9 short -> completed (the turn is merely slow), off -> completed 1,
  off -> event 1, short -> off 1.

**Reading.**
- The rule expert recovers the policy's own training-turn failures about as well as on the exam set (d314: 81 % at 2 s): 80 % at 2 s.
  - It recovers every off unit (9 / 9).
  - It recovers 73 % of the zero-event units, whose failures are mostly collisions. The expert's own failures are 7-9 collisions; it yields
    to logged agents only within 3 s.
  - Against the policy at the same length, the gain is +0.56 [+0.36, +0.74] at 2 s.
  - On short units there is no gain (-0.10): with 15 s the policy completes 9 / 10 of them itself. Those were not turn errors.
- Most of this is the expert being good at these turns: from the start it completes 33 of the 42 A units. Rewinding adds 45c25a1d
  (Delta 1-2 s), 227d5c07 (1 s) and 131c5de1 (2-3 s, out of tolerance at 1 s).
- Gate and labels:
  - Delta 1 s recoveries are hard pulls: 7 / 30 pass the gate (median |lat| at h is 1.2 m).
  - Delta 2 s: 15 / 33 pass. The failing item is mostly the curvature rate (dk, 12 of 18).
  - Delta 3 s: 23 / 31 pass, again failing only on dk.
- Row yield for a recovery lesson:
  - 345 decisions after the handover at Delta 2 s (median 8 per unit). 122 of them are in completed and gated units (15 turns).
  - 230 rows at Delta 3 s (23 turns).
  - That is 2 Hz serving rows. A 10 Hz label would be 5x.
- B is empty as a stratum. On the train turns the policy rarely breaks the d293 gate once it completes (4 / 21, all on entry speed, all at
  decisions 1-4, before any rewindable h).
- The gate's PAI-derived entry speed (5.6 m/s) is too strict for nuPlan humans: 25 / 63 human logs fail it, and only 37 / 63 human turns pass
  the full gate. A nuPlan-calibrated gate would change B and the gate columns. This is not done here.

Figures:
- [figs/rewind_train64_A.png](../../figs/rewind_train64_A.png): two A units. d3f25947 is off; the policy's line runs wide, and the expert
  handed over 2.2 m off at Delta 2 s completes, gated. 27e7523c is a zero event; every Delta completes, gated. Dots mark the handovers, x the
  policy's zero, dashed the human log.
- [figs/rewind_train64_B.png](../../figs/rewind_train64_B.png): the only B handover, de2955ff at Delta 1 s (entry-speed violation). The
  expert takes the corner on the route instead of the policy's 2.3 m wide line; it completes and is gated.
