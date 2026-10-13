# SPIN-FIX: making the ego never spin on PAI by changing only the driver's control path

Pre-registration: [plans/2026-10-13-spin-fix-prereg.md](../plans/2026-10-13-spin-fix-prereg.md) (with amendment 1). Code:
`experiments/alpasim/lib/track_ref.py` (switch `JEV_TRACK`, hooked into `pai_driver.py` and `sh30_driver.py`, default off). Scripts:
`scripts/sf_collect.py`, `sf_drive.py` (Tokyo extraction), `sf_detect.py` (spin detector), `sf_cases.py` (case table), `sf_read.py` (read),
`sf_arm.sh` (one arm on card 0), `sf_replay.py` (offline MPC replay), `sf_turn.py` (turn geometry). Sets and per-case data in `spin_fix/`.
Tokyo reads: `/data/runs/steer_long/spinfix/reads/<arm>`; box (nuPlan): `$DATA_DIR/runs/steer_long/spinfix/`.

**Status (2026-10-13 17:20 JST): line 1 passed, line 2 failed, nothing adopted.** `JEV_TRACK=full` (v6) gets **0 spins on the 38-scene
spin set**, with zeros 15 vs 16 and mean 0.4518 vs 0.4437. The 46 controls fail the registered no-regression line on the low-speed turns
(2 new zeros, mean -0.035). nuPlan held 452 fails too (zeros +11), and the ablation pins that cost on the path-tracking law, not on the
curvature cap. The amin -7 and weight-5 spin-set reads and the under-turn closed-loop reads did not run: card 0 was reserved for the user from
17:25 JST and the amin -7 read was terminated at 16/38 rollouts.

## 1. Case set and mechanism (step 1-2)

38 scenes (`spin_fix/spin_set.txt`, facts in `spin_fix/cases.csv`): 26 with a spin in the stored RSG-L18-s0 PAI 281 read (weight 20) + the
14 handoff-spin scenes of the gains pilot (weight 5), 2 overlap. 15 spins at the force-GT handoff (median 28.9 m/s), 23 mid-rollout
(median 8.2 m/s; includes all low-speed in-turn spins; full-lock circling below 5 m/s only appears together with a fast spin, 2 / 38).

**One mechanism in all 38**: in the second before the spin the served plan's 2 s arc length is >= 3 m short of 2 x ego speed (median deficit
6.5 m at the handoff, 7.7 m mid-rollout; mid-rollout the plan's 1-2 s speed collapses to 0.6-3 m/s) and the acceleration sits at the MPC's
-9 bound or -4..-8. The plan's 1-2 s curvature x v^2 stays < 3 m/s^2 in 26 / 38. AlpaSim's MPC tracks positions only (no steering-size,
steering-rate or lateral-acceleration cost, linear tyres without saturation): when braking cannot remove the distance it steers, and at
speed any steering is a spin.

Offline replay caveat (`sf_replay.py`, B's handoff replay + reference arms): at the handoff step every non-GT reference, even a constant
extrapolation of the car's own motion anchored at the driver's t0 pose, reproduces the jump, while the same extrapolation anchored at the
replayed step's own pose does not (|u| <= 0.02): the replay's force-GT pose (GT trajectory) and the driver's anchor disagree, so the
handoff replay amplitudes (also B's) are partly a replay artefact. Closed loop decides; no conclusion rests on the replay.

## 2. Variants (closed loop, Tokyo card 0, RSG-L18-s0, VCONT 1.0, LEAD 1, weight 20 unless stated; baseline = stored RSG read)

| run | switch | spin set: spins (handoff / fast / slow) | zeros | at-fault / front / hit from behind | mean |
|:--|:--|:--|--:|:--|--:|
| base | stored RSG read | 26 (2 / 26 / 1) | 16 | 1 / 1 / 6 | 0.4437 |
| 1 full v1 | curvature cap ALAT 4 / JLAT 5, reference starts at the measured curvature | 0 | 21 | 4 / 4 / 6 | 0.3113 |
| 2 time | re-timing only, AMIN -5, path unchanged | 21 (0 / 20 / 3) | 17 | 2 / 2 / 5 | 0.4332 |
| 3 full v2 | heading from plan yaw, first curvature = desired, ALAT 7 / JLAT 10 | 16 (0 / 16 / 0) | 16 | 2 / 2 / 5 | 0.4599 |
| 4 full v4 | + VCURV, ALAT 4, cap 0.085 above 5.5 m/s | 8 (0 / 8 / 0) | 17 | 2 / 2 / 5 | 0.4346 |
| 5 full v5 | cap 0.075 from 5 m/s | 1 (0 / 1 / 0) | 17 | 2 / 2 / 5 | 0.4346 |
| 6 full v6 | + longitudinal jerk 10 m/s^3 from the measured acceleration | **0** | 15 | 1 / 0 / 6 | 0.4518 |

Per scene: [spin_fix/case_table.md](spin_fix/case_table.md); case facts (speed, deficit, steering peak) in `spin_fix/cases.csv`.

What each step showed:
- 1: zero spins but a design bug: the reference restarted every 0.1 s from the car's own curvature and ramped toward the plan, so the
  receding-horizon reference lagged every turn (60cc35db left the corridor 1.1 s after the handoff).
- 2: re-timing alone removes every handoff spin (the plan's speed deficit) but not the mid-rollout ones: the car, faster than the plan, is
  driven through the plan's turn at a lateral demand the linear-tyre vehicle turns into side slip.
- 3: ALAT 7 is above the detector's 6 m/s^2 line; the remaining 16 are sustained |v wz| >= 6 at speed or side slip near 5 m/s.
- 4-5: the 8, then 1, remaining were side slip >= 0.15 while slowing into tight turns at 5.0-5.5 m/s, where AlpaSim switches to the
  linear-tyre model; the last one (d8bfaaf2, 18.4 s) had a straight, feasible reference and still drew a 0.69 rad command from a step in the
  reference's deceleration caused by plan-speed flicker.
- 6: the jerk limit removes that step.

Controls, v6 (`full6_ctrl2`; the first control read was destroyed by a compose-name clash with another lane's run of the same tag and
chunk, see section 5):

| set | scenes | spins | zeros (base) | at-fault / front / behind (base) | mean (base) | line |
|:--|--:|--:|:--|:--|:--|:--|
| ctrl_fast (>= 20 m/s handoff) | 20 | 0 | 0 (0) | 0 / 0 / 13 (0 / 0 / 13) | 0.6119 (0.5958) | pass |
| ctrl_slow (low-speed turns) | 16 | 0 | **2** (0) | 0 / 0 / 5 (0 / 0 / 6) | **0.6382** (0.6727) | **fail** |
| ctrl_brake (hard braking) | 10 | 0 | 6 (6) | 0 / 0 / 1 (0 / 0 / 1) | 0.3405 (0.3394) | pass |

The registered line 2 (36 fast + slow controls: spins 0, zeros not above 0, mean not below base - 0.01) **fails**. The new zeros are
268949d2 (corridor, accelerating out of a right turn through 5 m/s, where the reference's curvature cap and lag leave the car about 0.5 m
less turned than the plan) and e848c843 (offroad after a late right turn at 2.8 m/s). The low-speed turn scenes otherwise gain
(b727530b +0.25, 09528162 +0.52, c4a58f83 +0.17). Brake controls are unchanged, so the -5 m/s^2 reference bound did not cost a collision
there; on the spin set front collisions went 1 -> 0.

## 3. nuPlan

- smoke5, pool, RSG-L18-s0 (run 5 code): with the switch on, 0 spins, 0 zeros, 0 at-fault, mean 0.938 vs 0.929. With the switch off,
  against the parent commit, the plans are not bit-identical: even the first decision differs by up to 1.6 cm with identical ego inputs (the
  jobs ran on cards 1 and 5). That is GPU noise, not the code: scores agree within 0.001, and with the switch off the code path is the
  unchanged else-branch.
- Guard reads, v6 `JEV_TRACK=full` vs the stored RSG-L18-s0 (s1 shown as seed noise), `ol_confirm.py`:

| read | mean (diff, 95 % CI) | zeros | at-fault zeros | GT distance (m) |
|:--|:--|:--|:--|:--|
| public 1479 @10 | 0.9398 (-0.0075 [-0.016, +0.001]) | 52 vs 46 | 46 vs 39 | 0.650 vs 0.792 |
| held 452 @18, full v6 | 0.8571 (-0.024 [-0.045, -0.005]) | 37 vs 26 | 32 vs 21 | 0.898 vs 1.135 |
| held, time (re-timing + jerk only) | 0.8730 (-0.008 [-0.023, +0.007]) | 29 vs 26 | 20 vs 21 | 1.073 |
| held, full without the lateral cap | 0.8568 (-0.024) | 38 | 33 | 0.923 |
| held, full with cap 0.15 / ALAT 6 | 0.8571 (-0.024) | 37 | 32 | 0.910 |
| held, RSG-L18-s1 (seed noise) | 0.8830 (+0.002) | 26 | 19 | 1.144 |

  The public read passes the 0.02 guard. Held fails, and the ablation says the cause is the path-tracking law (the reference converges to the
  plan's polyline over max(v x 1 s, 4 m)), not the curvature cap: removing or raising the cap changes nothing. Re-timing alone (`time`) ties
  the stored read on held (mean within 0.01, zeros +3). nuPlan has 0 spins without any switch (d285).

## 4. Turn under-steer split (P0, coordinator scope)

Offline geometry (`sf_turn.py`, `spin_fix/turn_geom.csv`): the 17 "turned too little" and 11 "straight at turn" RSG zeros, at every
inference in the 6 s before the event where the model plan points >= 30 deg into the turn at 4 s. The quantity is the heading the MPC's
1-2 s window is asked for:
- Median over the 20 cases with such inferences, as a ratio to the plan's own 2 s heading: served reference (VCONT + LEAD re-timing, as run)
  0.98; track_ref full 0.96. The model plan at its own timing gives the same headings as served (the 1.5 s values agree to within a degree).
- The plan itself turns about 30 deg by 2 s against about 60 deg at 4 s. The car reaches 0.74 of the plan's 2 s heading.
- Split: in 0 / 17 under-turn cases does (b) or (c) ask for materially more heading than (a). So the VCONT hypothesis is not supported, and
  the under-steer is not serve-side re-timing.
- The 26 % car-versus-reference shortfall is controller tracking lag. A preview switch (`JEV_TRACK_CFG=preview=0.5`: curvature fed forward
  v x 0.5 s ahead) and weight 5 vs 20 were queued for the 17 cases. They did not run (card 0 reserved).

## 5. Final switch set and recommendation

- Code (all default off): `JEV_TRACK=time|full`, `JEV_TRACK_CFG` keys `amin amax jlon alat jlat kmax kdyn krate vcurv latcap preview
  tau_lat d_min zeta v_min v_drop`. v6 defaults: AMIN -5, AMAX 3, JLON 10, ALAT 4, JLAT 10, KMAX 0.2 (below 3.5 m/s) -> KDYN 0.075
  (from 5 m/s), KRATE 0.1, VCURV on, LATCAP on, PREVIEW 0.
- Verdict against the registered lines: **line 1 pass** (0 / 38 spins at weight 20). **Line 2 fail** (low-speed turn controls 2 zeros,
  -0.035). The weight-5 question (is the fix alone enough without the gains knob) is unanswered.
- **Recommendation: adopt nothing as default yet.** `JEV_TRACK=full` (v6) is the only setting that reached 0 spins, and it does it without
  the gains knob doing the work at the handoff. But it costs low-speed turns on PAI and zeros on nuPlan held. The next read (one card, about
  3 h) is a hybrid: the plan's own path re-timed (`time`, which ties nuPlan held and removes every handoff spin) plus the VCURV speed limit
  (slow for the plan's curvature instead of tracking a smoothed path). Read it on the 38 + 46 at weight 20 and 5, then on nuPlan held.
- Infrastructure:
  - `pai_eval.sh` takes `RUNPFX`, because two lanes running one tag and chunk at once share the compose project `pai-<tag>-c000`, and the
    second start tears the first stack down.
  - B's handoff replay mixes the GT pose and the driver's anchor at the handoff step, so its amplitudes are not evidence.
