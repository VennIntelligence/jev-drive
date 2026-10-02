# Tracker-lag pre-compensation of the submitted trajectory (day 2026-10-04, lane TRK; pre-registration)

Written before any score of this run was read. User decision 2026-10-04: allowed as a per-board trick, labelled and reported
separately (tmp/2026-10-03-overall-plan.md). Follows decision 88 (lateral gain calibration lost 3.4 navtest PDMS).

## What the scorer does (read from the devkit, navsim main @ 0a380a9 and v1.1: the simulator files are identical up to formatting)
- The 8 submitted poses (0.5-4 s) are linearly interpolated, with the t0 ego state prepended, to 41 states at 0.1 s.
- `BatchLQRTracker`: per 0.1 s step, longitudinal one-step LQR to the reference velocity 1 s ahead (q 10, r 1); lateral
  one-step LQR over a 10-step (1 s) horizon on [lateral error, heading error, steering angle], Q = diag(1, 10, 0), R = 1,
  velocity / curvature profiles fitted from the poses with jerk / curvature-rate penalties.
- `BatchKinematicBicycleModel` (Pacifica wheelbase): steering angle low-pass 0.05 s, acceleration low-pass 0.2 s, Euler.
- Start state: x, y, heading, velocity and acceleration of the t0 ego state; **steering angle, steering rate, yaw rate are 0
  for every token** (checked on navhard both stages and navtest v1 caches). The tracker therefore starts from straight wheels.
- Check done before this plan: our own proposal construction equals the devkit's `get_trajectory_as_array` to 1e-5 m (float32
  pose files) and the simulated states agree to 2e-5 (30 tokens each of navhard, navtest, navtrain).

## Transform (reference-free)
Per token, with the uncompensated 8 poses p as the model's intended plan (dense target = the scorer's own interpolation of p):

    u* = argmin_u  sum_{t=0.1..4} |e_lon(t)|^2 + |e_lat(t)|^2 + 2^2 wrap(h_sim(t) - h_p(t))^2 + 0.01 |u - p|^2

where (e_lon, e_lat) is the simulated rear-axle position minus the plan position in the plan's frame at time t, and h_sim
comes from the devkit's own `PDMSimulator` started from the t0 velocity / acceleration (the agent's ego_status) with
straight wheels. Levenberg-Marquardt, finite-difference Jacobian over the 24 pose values, at most 12 iterations.
Submission: u_alpha = p + alpha (u* - p). Inputs: the plan and the agent's own ego status; no reference, no map, no
scorer output, no stage label. (A y-and-heading-only variant was tried on 30 navtrain tokens for runtime and dropped before
any score: with free y and heading it still removes the longitudinal error, so it does not isolate lateral.)

## Free parameter, fitted on navtrain only
alpha in {0.25, 0.5, 0.75, 1.0}; native Cinque on lb_navtrain (3 000 real tokens, official v1 PDMS, `v1_navtrain_oplb`).
alpha* = the alpha with the largest navtrain PDMS delta vs uncompensated (ties: smaller alpha); none if no delta > 0.
N4 has no held-out navtrain predictions (it was fitted on navtrain), so N4 uses the native alpha*.

## Arms scored on the test boards
For native Cinque and N4 (best runnable arm on navhard, 36.07; pose files as shipped): alpha = 1.0 (the principled
"realise the plan" arm, scored regardless of navtrain) and alpha* if it exists and differs. Official scorers:
navhard two-stage EPDMS (v2) and navtest PDMS (v1.1), via navsim_zs_score.sh.

## Readouts
- navhard: combined EPDMS delta vs the same model uncompensated, 95% paired bootstrap over the 225 scene-mapping groups
  (5 000 resamples); stage 1 / stage 2; per sub-metric (NC, DAC, DDC, TLC, EP, TTC, LK, HC, EC) per stage with token bootstrap.
  In-process devkit harness (reproduces the official CSVs); every compensated arm also scored by the official script and the
  combined numbers compared.
- navtest: PDMS delta and per sub-metric (NC, DAC, DDC, EP, TTC, C), per-token paired bootstrap from the official CSVs.
- Tracker response (the "37% at 1 s / 72% at 2 s" fact): through-origin slope of simulated lateral on planned lateral at
  1 / 2 / 4 s (tokens with |plan lateral| > 0.5 m), uncompensated vs compensated; mean position / lateral / longitudinal
  tracking error to the plan; how far the submitted poses move from the plan.
- Realising vs gaming: an **ideal-tracker** arm on navhard scores the plan's own interpolated poses as the simulated
  states (exact tracking, dynamics by finite differences; comfort terms not meaningful for it). Reading:
  - "realises the plan" if the compensated tracked trajectory is within 0.2 m mean of the plan AND the compensated
    non-comfort sub-metric gains (DAC, NC, DDC, EP, TTC, LK) are no larger than the ideal arm's (point estimates);
  - "exploits the scorer" if the compensated arm beats the ideal arm on those sub-metrics by more than the CI half-width,
    or if the gain comes from tokens where the tracked trajectory moves away from the plan.
- Lateral gain (decision 88) on the same tables: navtest per sub-metric of `gain_single` vs native, to place the two.

## Decision rule (fixed now; per model)
- Adopt as a labelled navhard / NAVSIM adapter trick if navhard combined delta > 0 with CI lower bound > 0 AND navtest
  PDMS delta >= -0.30 (point). A navtest gain with CI lower bound > 0 counts on its own as a navtest trick.
- Ambiguous if navhard delta > 0 with CI including 0; reject if navhard delta <= 0 and navtest delta <= 0.
- Arms: alpha 1.0 and alpha* per model (at most 4 navhard comparisons); no multiplicity correction, said in the result.

## Other boards
The transform targets the NAVSIM LQR + bicycle only; it applies to every board scored with that simulator (navtest v1
PDMS, navtest v2 EPDMS, navhard two-stage, navtrain). HUGSIM, Bench2Drive and WOD are discussed, not scored (see result).
