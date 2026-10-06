# HUGSIM `spec_plan`: steering from the model's own plan makes turning routes worse, not better (no spin or oscillation blow-up)

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
