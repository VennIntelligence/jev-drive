# Non-privileged longitudinal launch rule on HUGSIM: speed through the 0-3 m/s window 6.4 s -> 3.9 s (median), spins unchanged (11 vs 9-10), both pre-registered lines fail

Written 2026-10-04. Pre-registration (committed before any closed-loop run, commit e7f0c3fa): [../plans/2026-10-04-launch-long-prereg.md](../plans/2026-10-04-launch-long-prereg.md).
Follows decision 115 (real cars leave 0-3 m/s in ~1.8 s; HUGSIM spends a median 6.4 s of the first 10 s below 3 m/s) and 113 / 114 (lateral rules).
Code: `lib/launch_long.py`, agent hook `experiments/hugsim/lib/zs_agent.py` (opts `launch_long`), lead x / v logged by `hugsim_zs_server.py`, chain `experiments/hugsim/scripts/launch_long_chain.sh`,
report `experiments/hugsim/scripts/launch_long_report.py`; tables `launch_long/{summary.md,summary.json,launch_long_runs.csv}`. Shipped Cinque, PR#57 controller (tree `fixed`), 64 exam scenarios, one run,
card 1 (lease `hugsim-launchlong`, released), 23 min wall. Tags: **[E]** measured, **[I]** inference.

## 1. Diagnosis: why the HUGSIM launch is slow (CPU, from the 64 `cinque-fixed-base2` logs + the ego / actor tracks in `data.pkl`)

Steps below 3 m/s in the first 10 s (n = 1556 steps in 64 runs); "blocked" = an actor in the ego corridor (|lateral| < 2 m) with a front gap below G:

| cause (exclusive, in this order) | G = 10 m | G = 20 m | G = 30 m |
|---|---|---|---|
| actor ahead in the lane | 16% | 38% | 50% |
| the model's own plan is a stop (3 s end < 1 m, `straight_stop`) | 26% | 20% | 16% |
| moving plan, just slow | 58% | 43% | 34% |

- **Controller (PR#57 iLQR): not the limit [E].** Plan-implied speed (3 s end / 3) over actual v is 2.0 / 1.7 / 1.25 / 1.03 at steps 1 / 4 / 8 / 12 (a lag of 1-3 steps, closed by 3 s); the achieved acceleration correlates 0.94 with the one the plan asks for and is above it; offline the same iLQR realises 1.6-1.7 m/s^2 in the first second when asked for 1.2 (section 3).
- **Openpilot plan speed: a large part [E, role I].** Per dataset (steps with v < 3 and no actor within 20 m, non-stop plans): plan 3 s end / 3 = 2.53 (nuScenes) / 2.44 (kitti360) / 2.11 (pandaset) / 1.41 (waymo) m/s; the model's own ego-speed estimate over 1.25 v = 0.99 / 0.96 / 0.88 / 0.84; median seconds below 3 m/s in the first 10 s 2.5 / 5.0 / 7.0 / 8.75. The gradient follows the dataset's camera height (nuScenes lowest), the same direction as decision 104's camera-height scale (0.8 of the logged speed on NAVSIM, camera 1.87 m); the cause is not isolated here. Of the non-stop plans the real-car profile (+2 s 3.4 m/s) is far above any of these.
- **Obstacles ahead: a large part [E].** 66% of the 64 scenes have an actor in the lane within 25 m at launch (decision 110), 38% of the slow steps have one within 20 m, and the model reports a lead (lead_prob > 0.5) on 46% of them. 23 of 64 runs have no actor within 20 m in the first 10 s; they still spend a median 6.25 s below 3 m/s, 40% of those steps with a stop plan and 12% with lead_prob > 0.5.

So: the slow launch is the model's own plan (stop or slow, strongest on waymo / pandaset) plus real blockers; the controller is not the limit.

## 2. Rule (tuned offline, disclosed; the pre-registered values, nothing retuned after the runs)

Agent side, longitudinal only: when a plan is not a stop and the lead head reports no close lead, the model's plan is re-timed along its own path so that it covers at least the distance of a launch profile (a = 1.2 m/s^2 from the current speed, capped at 4 m/s).
Active during a launch only: latched on at start and after every v < 0.5 m/s, off for good at the first v >= 3.5 m/s (no speed floor afterwards).
Gate per step: plan 3 s arc length >= 3 m and not `straight_stop`; plan 1 s arc length >= 0.7 v; not (lead_prob > 0.5 and 0 < lead_x < 20 m). Inputs: ego speed and the model's own plan / lead outputs only; no scene, route or map. Lateral target unchanged, controller unchanged.

| parameter | value | how chosen |
|---|---|---|
| a | 1.2 m/s^2 | offline: HUGSIM `plan2control` + v += a dt on a straight line, v0 0.3 / 0.7 / 1.0, against the comma1M profile (1.5 / 3.4 / 5.1 m/s at +1 / +2 / +3 s, d115, capped at 3.5), a in {0.8, 1.0, 1.2, 1.4, 1.7}, vcap in {3.5, 4, 4.5}: RMSE minimum 0.16 m/s at (1.2, 4.0); v0 = 0.7 gives 2.3 / 3.2 m/s at +1 / +2 s, crossing 3.5 at 2.0 s, max acceleration 1.75 (comfort bound 2.4) |
| vcap | 4.0 m/s | vcap = 3.5 makes v approach 3.5 asymptotically (3.49 after 4 s), the latch would never release |
| v_end | 3.5 m/s | given |
| x_safe, p_lead | 20 m, 0.5 | physical / prior: 5.7 s at 3.5 m/s, 6x the 3 m braking distance from 3.5 m/s at 2 m/s^2; not tuned on any HUGSIM result (the old logs hold lead_prob only, lead_x was added for this run) |
| dmin, keep | 3 m, 0.7 | prior values |

## 3. Result (HUGSIM, all 64, one run; smoke of 4 scenes first, not scored)

| arm | spins | on the 10 | new | initial-launch spins | re-launch spins | fg_collision | max_steps | HD | RC | below 3 m/s in first 10 s, median | v at +1 / +2 / +3 s (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base (exam) | 10 | 10 / 10 | 0 | 6 / 64 | 2 / 27 | 26 | 15 | 0.278 | 0.349 | 6.38 s | 1.31 / 2.15 / 2.65 |
| base rerun 1 | 9 | 8 / 10 | 1 | 6 / 64 | 0 / 25 | 25 | 14 | 0.274 | 0.343 | 6.38 s | 1.30 / 2.14 / 2.62 |
| base rerun 2 | 9 | 8 / 10 | 1 | 6 / 64 | 0 / 20 | 25 | 15 | 0.280 | 0.350 | 6.38 s | 1.31 / 2.15 / 2.64 |
| **launch_long** | **11** | 9 / 10 | 2 (034-hard-00, 2800_3000-easy-00) | 6 / 64 | 1 / 21 | 25 | 17 | 0.272 | 0.343 | **3.88 s** | **1.78 / 3.29 / 3.74** |

Non-spin HD (54 scenes) paired against the exam base: **+0.011 [-0.029, +0.048]**; against rerun 1 +0.015 [-0.024, +0.054]; against rerun 2 +0.008 [-0.031, +0.045].

- **Line (i) spins <= 2: FAIL, 11** (base 10, reruns 9, 9; rerun noise about +-2). Per launch event 6 / 64 initial, the same as every base arm. [E]
- **Line (ii) non-spin HD paired CI lower bound > -0.02: FAIL, -0.029** (point estimate +0.011; the interval is wide, so this is "not shown harmless", not "harmful"). [E]
- **The rule did what it was asked on speed [E]**: median time below 3 m/s in the first 10 s 6.38 -> 3.88 s; v at +1 / +2 / +3 s 1.78 / 3.29 / 3.74 against the real cars' 1.5 / 3.4 / 5.1 (the cap stops it at 3.5-4; the model's own plan takes over). The assist was on in 511 of the steps of the 64 runs, with reasons off: stop plan 6704 (the model plans a stop on 36% of low-speed steps in the logs), disarmed 1444, lead 200, model already faster 55, slowing 1.
- **Spins [E]**: 9 of the 10 baseline spinners spin again (all but 570_770-easy, which is a rerun-noise scene: base 63 deg, reruns 18 / 15 deg). In the spinners where both arms can be read the divergence starts at the same step: 102751446607-medium-01 (5 deg at step 8 in both, v 1.47 -> 2.07), 053-medium-02 (step 5 both, v 2.05 -> 2.21), 034-hard-00 (step 11 -> 10, v 2.16 -> 2.87; spin 8 deg -> 105 deg, a new spin), 152217047339-medium-00 (step 10 -> 7, v 1.80 -> 3.12). The faster launch does not make the loop leave the window before the seed grows; the c of the controller rises with speed (0.066 / 0.146 / 0.307 at < 1 / 1-2 / 2-3 m/s, d113), so a car at 2-3 m/s a step earlier meets a higher c.
- **Front collisions [E]**: 25 (base 26, reruns 25, 25), not more. New fg_collision against all three baseline arms: 0254-extreme-00 (a baseline spinner that completes with HD 0.82; here fg collision, HD 0.06), 0418-hard-00, 3000_3200-medium-00 (HD 0.74 -> 0.04). Only 3000_3200-medium-00 is a collision within 15 s with the assist on during the previous 3 s (safety check of the pre-registration: > 2 such scenes = unsafe; here 1, so ok). Its mechanism in the log: lead_prob 0.44 and lead_x 15 m at assist start (below p_lead), v 3.7 m/s at step 10, then the model's own plan keeps accelerating (5.5 m/s at step 15) into an actor at 21 m that the lead head does not report (lead_prob 0.01). HUGSIM attack actors are scripted in time, so a faster ego meets them at different places; this is a property of the benchmark, not of the lead gate. [I]
- Gains: 034-hard-01 0.14 -> 0.78, 032-medium-00 0.12 -> 0.35 (waymo / pandaset scenes that were stuck on stop / slow plans), 322492347634-easy-00 0.60 -> 0.79; losses: the collision scenes above. Mean HD over all 64 is 0.272 against 0.274-0.280 for the bases.

## Reading

1. [E] A non-privileged, longitudinal-only launch that reproduces the real-car speed profile does not remove HUGSIM launch spins (11 vs 9-10) although it cuts the time below 3 m/s by 40% and the early speed matches d115 at +1 / +2 s. The d115 reading (ii), "real cars only dwell 1.8 s in the sensitive window and the gain has no time to amplify the seed", is not supported by the closed loop; the window is left earlier, the divergence starts at the same step.
2. [I] Decision 90 point 4 (privileged launch assist drives the first 5 s to 5 m/s, 5 of 6 spins removed) was a route follower: it also steered. The lateral part, not the speed, is the likelier reason it worked; the pure speed part, tested here, does not remove spins. This points back to d115 (i), lateral not engaged at launch on real cars (untestable on comma1M), and to the lateral rules of d113 / d114 as the only closed-loop levers that moved the spin count.
3. [E] The rule is not harmful by the point estimate (HD +0.011) but the interval does not clear the -0.02 line and 3 new front collisions appear while 3 old ones go; with launch speed 1.8 s earlier HUGSIM's time-scripted attackers meet the ego at different places.
4. Not run: B2D (the gate was A passing; no lane was written). If it had passed, the lane would be `drive` arm, 19 routes x seed 2, 3 (same as the lowspeed B2D lane), with the same hook in `scripts/b2d_zeroshot_agent.py`.

## Limits

One run per arm (spin noise +-2, HD noise +-0.005 per the two reruns); a, vcap are calibrated on a straight-line open-loop iLQR model, not a closed loop; x_safe / p_lead / dmin / keep are priors, so a stricter lead gate or a different profile is not excluded, but the spin mechanism above does not depend on them (the divergence starts at the same step with the assist on). The ego-lane actor gap of section 1 is computed from `data.pkl` boxes with a 2 m corridor, not from the model's view. The smoke (4 scenes) was read before the full run only to confirm the hook and the lead gate (assist off at lead_x 14-16 m with lead_prob > 0.5; on at 22-24 m); no parameter changed afterwards. Dataset gradient of section 1 is a correlation over 16 scenes per dataset, camera height is the inferred cause.
