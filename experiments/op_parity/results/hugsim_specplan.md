# HUGSIM `spec_plan` family: one-point plan curvature is worse than spec; plan curvature averaged over 0.5-1.5 s is better (turning +0.054); the legacy lateral MPC equals spec

Written 2026-10-06. Question: P2's plan improved but HUGSIM `spec` steers from the action head (anchored to shipped). Replace the action curvature by the curvature of the model's OWN plan, keep clip_curvature, lateralDelay 0.25 s and the iLQR longitudinal unchanged. Preset `spec_plan` (opts `op_ctrl_src: plan`, docs/openpilot-interface.md). Conversion = `modeld.get_curvature_from_plan` (mirrored in `jevdrive/openpilot/model.py curvature_from_plan`): `2 psi(t) / (v t) - yaw_rate(0) / v` at t = action_t 0.275 model-s, psi and yaw rate from plan channels 11 / 14. The sim then sees the same request path as `spec`.

Arms: P2-F-s0/s1, P2H10-F-s0/s1 (hinge), 64 scenarios each, one run per scenario, harness / scoring / classes as [hugsim_full.md](hugsim_full.md). Controls: stored `spec` and `exam` runs of the same arms (not rerun). 23 turning scenarios = route yaw range >= 30 deg, 41 straight. Tables: [hugsim_specplan/tables.md](hugsim_specplan/tables.md) (+ CSVs); script `scripts/pp_specplan_report.py`; Chinese page `research/hugsim-specplan/index.html`.

## Result

| HD-Score (seed / arm mean) | exam | spec | spec_plan | WA-JEPA |
|---|---|---|---|---|
| all 64 (4 arms) | 0.398 | 0.394 | **0.339** [0.260, 0.430] | 0.451 |
| turning 23 | 0.312 | 0.279 | **0.170** [0.106, 0.245] | 0.442 |
| straight 41 | 0.446 | 0.459 | 0.433 | 0.456 |

- Paired spec_plan - spec (4 arms): all -0.056 [-0.108, -0.007]; turning -0.109 [-0.198, -0.029] (3 wins / 12 losses / 8 ties); straight -0.026 [-0.092, +0.037]. vs exam: all -0.059 [-0.122, -0.001], turning -0.141 [-0.228, -0.062]. vs WA-JEPA: all -0.112 [-0.201, -0.025], turning -0.271 [-0.413, -0.137]. P2 and hinge alone give the same numbers within 0.005; all four arms individually are negative vs spec (-0.04 to -0.07).
- Turning routes complete: 0.25 / 23 per arm (spec 2.0, exam 3.25, WA-JEPA 8). End classes per arm on the turning routes (spec_plan / spec): fg 11.3 / 7.5, bg 4.8 / 9.0, off_route 4.5 / 3.0, spin 2.25 / 1.5. Straight routes: complete 15.3 vs 18.3, bg 5.8 vs 2.0, off_route 2.3 vs 0.
- Spins and oscillation: spins overall 6 / 1 / 2 / 4 (spec 2 / 1 / 2 / 2, exam 8 / 11 / 7 / 11). Heading-rate sign flips: 2.0-3.0 per 100 moving steps vs 0.0 under spec (0.12-0.24 exam); no run meets the oscillation definition (>= 8 flips and >= 15% of moving steps). So no spin / oscillation blow-up, but the heading rate is rougher: RMS 0.8-1.0 deg/step vs 0.26-0.29 under spec.
- Why it is worse: the requested curvature is 3.3x larger in magnitude on turning routes (0.022 vs 0.0066 1/m) and 5x jerkier (mean step change 0.0090 vs 0.0018 1/m), straight routes the same (0.0126 vs 0.0036; 0.0071 vs 0.0011). One-point yaw / yaw-rate extrapolation at 0.275 s amplifies plan noise (fig5: a 0.18 1/m spike, 6 deg/step heading rate, then a collision at step 66 where spec drives 300 steps). The clip is a pass-through for spec (turn-gain.md) but not here.

## Verdict

Joint plan + action training is not supported by this test. Steering from the plan through openpilot's own plan-to-curvature conversion does not improve turning routes (HD 0.17 vs 0.28, 0.25 completes vs 2); it is not a spin explosion either, so the plan is not unstable in closed loop in the sense of decision 133, but the single-horizon conversion is too noisy to steer with. This does NOT test a smoothed conversion (e.g. curvature averaged over 0.5-1.5 s of the plan) or a plan tracker with good low-speed behaviour: if the plan is to steer, the conversion would be a declared execution-layer choice, not openpilot's.

## Deviations and caveats

- One-shot curvature conversion exactly as modeld's, with the harness speed (model clock, same as the action path); the conversion is not tuned.
- Scenario sharding: 4 pool jobs, 8 workers each, 33-45 GB per job (two per card was not admitted at 45 GB, resubmitted at 33 GB; real use ~37 GB). `TIMEOUT=600`, `RETRIES=2` for this run; unix-socket connect now retries (`scripts/zeroshot_wire.py connect_retry`). Three P2-F-s0 scenarios crashed twice under load (pre-run sim start failures, not the socket timeout) and were rerun alone; all 768 runs finished.
- Scenario-level differences below ~0.2 are not evidence; single run per scenario; 23 turning scenarios from 17 scenes.

## Follow-up (2026-10-07): smoothed conversion and the legacy lateral MPC

Same 4 arms x 64 scenarios, harness, pool, timeout 600 s / retries 2; 8 new (arm, preset) sets, 512 runs, all finished (a number of runs crashed under CPU contention with the other jobs and were rerun alone; none missing). Full tables: [hugsim_specplan/tables.md](hugsim_specplan/tables.md); figures in `hugsim_specplan/figs/` and the Chinese page.

- **`spec_plan_smooth`**: lateral curvature = mean curvature of the plan over 0.5-1.5 s model time, exactly `(psi(1.5) - psi(0.5)) / (s(1.5) - s(0.5))`, psi = plan yaw (channel 11), s = cumulative length of the plan positions (linear interpolation on T_IDXS; arc length floored at 1 m = MIN_SPEED x 1 s). Open-loop conversion, then the same clip + 0.25 s delay (`model.py curvature_window`).
- **`spec_plan_mpc`**: openpilot's legacy lateral MPC, `lib/op_lat_mpc.py`, from commaai/openpilot v0.9.4 (68aba7ce): `lateral_planner.py` `LateralPlanner.update` (weights PATH 1.0, LATERAL_MOTION 0.11, LATERAL_ACCEL 0.0, LATERAL_JERK 0.04, STEERING_RATE 700; references y, yaw of the model plan in the current vehicle frame each step; x0 = [0, 0, 0, desired yaw rate carried from the previous solution, interpolated one model step ahead]), `lateral_mpc_lib/lat_mpc.py` (state [x, y, psi, psi_rate], control psi_accel, N = 32 at T_IDXS, cost on y, v' psi, v' psi_rate, v' u, u / (v + 0.1), v' = v + 10), and `drive_helpers.get_lag_adjusted_curvature` (actuator-delay compensation `2 psi(delay) / (v delay) - curvature[0]`, delay 0.275 model s, curvature-rate limit 5 m/s^3), then the same clip + 0.25 s delay. **Deviation (declared in the module):** acados SQP_RTI is replaced by the exact solution of the QP of the small-angle linearisation (per-node speeds, exact discretisation, state bounds not imposed); Toyota-class rotation-radius constants; no `get_speed_error`. The feedback is the openpilot one: the plan is re-expressed in the current frame every step, the MPC state carries only its own commanded yaw rate.

HD-Score (4-arm mean, bootstrap 95% CI over scenarios):

| | exam | spec | spec_plan | spec_plan_smooth | spec_plan_mpc | WA-JEPA |
|---|---|---|---|---|---|---|
| all 64 | 0.398 | 0.394 | 0.339 | **0.428** [0.331, 0.531] | 0.395 | 0.451 |
| turning 23 | 0.312 | 0.279 | 0.170 | **0.333** [0.193, 0.481] | 0.291 | 0.442 |
| straight 41 | 0.446 | 0.459 | 0.433 | 0.482 | 0.454 | 0.456 |

Paired (scenario = unit, 4-arm mean; wins / losses / ties at |d| < 0.02):

| | all 64 | turning 23 | straight 41 |
|---|---|---|---|
| smooth - spec | +0.034 [+0.005, +0.068] | +0.054 [+0.006, +0.116] (8 / 3 / 12) | +0.023 [-0.012, +0.072] |
| smooth - exam | +0.031 [-0.009, +0.073] | +0.021 [-0.059, +0.102] | +0.036 [-0.012, +0.082] |
| smooth - spec_plan | +0.090 [+0.032, +0.155] | +0.162 [+0.065, +0.268] | +0.049 [-0.024, +0.130] |
| smooth - WA-JEPA | -0.022 [-0.102, +0.046] | -0.109 [-0.273, +0.035] | +0.026 [-0.052, +0.101] |
| mpc - spec | +0.001 [-0.011, +0.012] | +0.013 [-0.006, +0.033] (6 / 3 / 14) | -0.006 [-0.022, +0.008] |
| mpc - exam | -0.003 [-0.050, +0.047] | -0.020 [-0.105, +0.061] | +0.007 [-0.057, +0.065] |
| mpc - spec_plan | +0.057 [+0.006, +0.112] | +0.121 [+0.036, +0.214] | +0.020 [-0.044, +0.089] |
| mpc - WA-JEPA | -0.055 [-0.132, +0.013] | -0.150 [-0.299, -0.026] | -0.002 [-0.089, +0.079] |

End classes on the turning routes (mean per arm, of 23): complete exam 3.25 / spec 2.0 / spec_plan 0.25 / **smooth 5.25** / mpc 3.0 (WA-JEPA 8); spins 5.5 / 1.5 / 2.25 / 0 / 0 (WA-JEPA 1); bg 4.5 / 9.0 / 4.75 / 6.0 / 7.0; fg 9.0 / 7.5 / 11.25 / 10.25 / 9.5; off_route 0.75 / 3.0 / 4.5 / 1.5 / 3.5. Spins over all 64 per arm: smooth 0 / 0 / 0 / 0, mpc 0 / 0 / 0 / 0 (spec 2 / 1 / 2 / 2, exam 8 / 11 / 7 / 11, spec_plan 6 / 1 / 2 / 4).

Heading-rate sign flips per 100 moving steps (mean): exam 0.12-0.24, spec 0.0, spec_plan 1.9-3.0, smooth 0.0-0.08, mpc 0.0; no run of the new arms oscillates. Requested curvature on turning routes, |kappa| / mean step change (1/m): exam 0.0113 / 0.0030 (the action head, not used for steering there), spec 0.0066 / 0.0018, spec_plan 0.0220 / 0.0090, smooth 0.0141 / 0.0035, mpc 0.0089 / 0.0013 (straight routes: spec 0.0036 / 0.0011, smooth 0.0028 / 0.0009, mpc 0.0020 / 0.0003, spec_plan 0.0126 / 0.0071).

Reading:
- The plan can steer: with a smoothed open-loop conversion the plan's own curvature beats the action head on turning routes (+0.054, CI above 0, 5.25 vs 2.0 completed turns, no spins) and reaches exam level (0.333 vs 0.312, difference within noise). The one-point conversion's failure was noise amplification, not an unstable plan.
- The MPC does not add over spec: it tracks the plan smoothly (the smoothest curvature of all, step change 0.0013) but its HD on turning routes (0.291) is spec's (0.279), less than the open-loop average. The MPC plans a smooth path and its first-step curvature is conservative, which fits the 0.8 gain of the circle test (steady curvature 0.0284 for a plan of 0.03). The HUGSIM fail modes of the turning routes (fg collisions 7.5-11, bg 4.5-9) are not steering-smoothness problems in either arm.
- Neither reaches WA-JEPA on the turning routes (0.442; smooth -0.109 [-0.273, +0.035], mpc -0.150 [-0.299, -0.026]).
- Caveats: 23 turning routes from 17 scenes; per-scenario differences below ~0.2 are not evidence; the smoothing window (0.5-1.5 s) was the pre-specified one, not tuned; the MPC is a declared QP replacement for acados, with Toyota-class constants and no tuning for the P2 plan.
- Process notes: server socket names were shared by two presets of the same arm (a collision that stalled four jobs; fixed with per-preset names, `pp_hugsim.sh`); the four affected jobs were cancelled and rerun, no collided rows kept apart from fully completed runs on identical servers. Jobs declared 33 GB (measured 45-48) until the coordinator's note; reruns declared 50 GB.

## Repeats (2026-10-07): spec vs spec_plan_smooth, 2 more runs of every arm

Question: does decision 149's smooth - spec contrast survive repeats? 4 arms (P2-F-s0/s1, P2H10-F-s0/s1) x 2 presets x 2 new runs (`r1`, `r2`) x 64 scenarios = 16 bench runs (`jevdrive.bench run --repeat r1 r2 --workers 2 --jobs 2`, run keys `<arm>_<preset>-rr1/-rr2`), all finished, no missing scenario. `r0` = the stored run behind the tables above. Pooling: HD averaged over repeats within (arm, scenario), then over the 4 arms; contrast bootstrapped over scenarios (B 10000, seed 0). Script `scripts/pp_specrep_report.py`; tables in [hugsim_specplan/repeats/repeats.md](hugsim_specplan/repeats/repeats.md) (+ CSVs).

**HUGSIM is effectively deterministic.** There is no seed anywhere in the harness; repeats differ only through server / GPU timing. Per (arm, preset), 41-49 of 64 scenarios have identical HD across the three runs, 62-64 of 64 the same end, per-scenario sd mean 0.001-0.006 and median 0.000. Largest spreads: P2-F-s0 smooth, one scenario with range 0.578 (a chaotic collision timing); otherwise max range 0.015-0.068, and only 3 scenarios in 512 arm-scenario cells exceed 0.05. Arm HD varies by at most 0.009 between repeats (smooth P2-F-s0 0.419 / 0.419 / 0.428; all others within 0.004). So the repeats confirm that the earlier numbers are reproducible; they add almost no information about generalisation, which is limited by the 64 scenarios (23 turning from 17 scenes), not by run noise.

Paired smooth - spec (HD, 4-arm and repeat mean; W / L / T at |d| 0.02):

| pool | all 64 | turning 23 | straight 41 |
|---|---|---|---|
| r1 + r2 (new) | +0.034 [+0.003, +0.072] | +0.055 [+0.006, +0.124] | +0.022 [-0.013, +0.069] |
| r0 + r1 + r2 (pooled) | **+0.034 [+0.004, +0.072]** | **+0.055 [+0.006, +0.122]** | +0.022 [-0.013, +0.070] |
| r0 alone (decision 149) | +0.034 [+0.004, +0.071] | +0.054 [+0.007, +0.117] | +0.023 [-0.012, +0.070] |
| W / L / T (pooled) | 16 / 8 / 40 | 8 / 3 / 12 | 8 / 5 / 28 |

Single repeats give turning +0.054 / +0.052 / +0.059 and all +0.034 / +0.033 / +0.035. Means: spec 0.394, smooth 0.429 (all 64); turning 0.279 vs 0.334; straight 0.459 vs 0.482. Per arm (3 runs), turning: P2-F-s0 +0.045 [+0.008, +0.090], P2-F-s1 +0.029 [-0.029, +0.110], P2H10-F-s0 +0.070 [+0.008, +0.151], P2H10-F-s1 +0.075 [+0.010, +0.161]; all positive, three of four CIs above 0.

End classes, mean per arm x run (spec / smooth), turning 23: complete 2.0 / 5.3, fg 7.5 / 10.3, bg 9.2 / 6.0, off_route 2.8 / 1.4, spin 1.5 / 0. All 64: complete 20.3 / 24.7, bg 11.4 / 7.8, spin 1.5 / 0. Spins over 64, identical in all three runs: spec 1 / 1 / 2 / 2, smooth 0 / 0 / 0 / 0.

Reading: the +0.054 turning and +0.034 all-64 gains reproduce to the third decimal, with the same CI, because the pooled data are nearly the same data. The caveat of decision 149 is unchanged: scenario selection is the uncertainty (lower bound +0.004 to +0.006), the 64 scenarios are what generalises, and the gain is concentrated in a few scenarios (best scene-0013-medium-00 +0.75, scene-152217047339-medium-00 +0.64; worst scene-090-hard-01 -0.24). A new CI needs new scenarios or a second dataset, not more repeats of these.

Process: one command (`bench run`, 16 runs) had all 3 cards running a HUGSIM job within about 1 min (submitted 00:21:29 box time, all 3 cards busy at the first look 57 s later). Default 6-slot jobs (59 GB declared) did not fit beside the B2D collection (pool free VRAM 32-37 GB per card), so I used 2 slots per job (25 GB); first attempt cancelled and resubmitted. Bench issue found, not fixed in code: resubmitting 15 s after `cl cancel` reused the still-cancelling jobs of the old `jobs.json`, so 12 of 16 runs got stale ERROR files and ran nothing until a second `run` (2.5 h lost; wait for the cancel or check the job state first). After that, with other lanes' jobs queued ahead and later cards freed, the remaining 12 runs finished 2 h after the resubmission; total wall 3 h 55 min from the first command.
