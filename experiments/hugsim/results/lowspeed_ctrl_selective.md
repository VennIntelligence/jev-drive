# Selective low-speed rule (deadband on the plan's 1 s direction): HUGSIM spins 10 -> 6, both pre-registered lines fail, B2D not run

Written 2026-10-04. Pre-registration (committed before the first closed-loop run, commit b8dc4663): [../plans/2026-10-04-lowspeed-ctrl-selective-prereg.md](../plans/2026-10-04-lowspeed-ctrl-selective-prereg.md).
Follows [lowspeed_ctrl.md](lowspeed_ctrl.md) (uniform low-pass: spins 10 -> 1, non-spin HD -0.028 [-0.058, -0.006]).
Code: `lib/lowspeed_ctrl.py` (`plan_select`), `patches/hugsim/optional/lowspeed-sel-ctrl.patch` + tree `lowsel` in `experiments/hugsim/archive/zs_run.py`,
`experiments/hugsim/scripts/{lowspeed_sel_dist,lowspeed_sel_params,lowsel_report}.py`, `lowspeed_chain.sh` stages 3-5. Tables: `lowspeed_ctrl_selective/`
(`summary.md/json`, `lowsel_runs.csv`, `phi_distributions.json`, `phi_steps.csv`, `param_candidates.md`).
Shipped Cinque, PR #57 controller, 64 exam scenarios, one run per arm, one card, lease `hugsim-lowsel` (card 1, released). Tags: **[E]** measured, **[I]** inference.

## Rule and parameters (tuned offline, disclosed)

The plan is rotated about the car so that its 1 s direction a becomes a * g(|a|), g = clip((|a| - d0) / (d1 - d0), 0, 1) (dead zone below d0, ramp to d1, pass-through
beyond), scaled by the speed taper w(v) (1 up to 2.5 m/s, 0 at 3.5 m/s). **d0 = 2 deg, d1 = 4 deg**; the uniform low-pass and the curvature clip are off, so the
rotation is the only difference to the base. Chosen from baseline logs only (64 runs, no rule-arm result read): at steps 1-4 after a launch event |a| has median 2.2 deg
(56% <= 3 deg, n = 16 steps) for events that spin and 0.41 deg (92% <= 3 deg, 95.5% <= 4 deg, n = 112) for events that do not. Among the candidates in `param_candidates.md`,
(2, 4) is the strongest that changes no non-spin low-speed step by more than 2 deg (share > 2 deg: 0.000; (2, 6) 0.092, (3, 6) 0.113); it leaves 10% of the spin-event lean
(median). Pre-registered concern: the loop gain is > 1, so one seed step above d1 starts an unattenuated growth.

## Results (HUGSIM, all 64, one run per arm)

| arm | spins | on the 10 | new | initial-launch spins | re-launch spins | max_steps | HD mean | RC mean | non-spin HD paired vs exam base | vs same-day rerun |
|---|---|---|---|---|---|---|---|---|---|---|
| base (exam) | 10 | 10 / 10 | 0 | 6 / 64 | 2 / 27 | 15 | 0.278 | 0.349 | | |
| base rerun 1 | 9 | 8 / 10 | 2800_3000-easy | 6 / 64 | 0 / 25 | 14 | 0.274 | 0.343 | -0.005 [-0.015, +0.003] | |
| base rerun 2 (this run) | 9 | 8 / 10 | 2800_3000-easy | 6 / 64 | 0 / 20 | 15 | 0.280 | 0.350 | +0.002 [-0.000, +0.005] | +0.007 [-0.001, +0.017] (vs rerun 1) |
| uniform low-pass (d113) | 1 | 0 / 10 | 2800_3000-easy | 0 / 64 | 0 / 32 | 21 | 0.268 | 0.315 | -0.028 [-0.058, -0.006] | -0.023 [-0.052, -0.004] |
| **selective (d0 2, d1 4)** | **6** | 5 / 10 | 0041-medium | 2 / 64 | 1 / 18 | 18 | 0.286 | 0.334 | **+0.005 [-0.021, +0.038]** | +0.009 [-0.013, +0.042] (rerun 1); +0.002 [-0.024, +0.036] (rerun 2) |

- **Line (i) spins <= 2: FAIL, 6** (base 10, reruns 9). Per launch event 3 / 82 against 8 / 91 (base), 6 / 89 (reruns). Remaining spins: 0041-medium (new, not a spinner in any base run), 0528-medium, 053-medium-02, 152217047339-medium-00, 5980_6180-easy, 8440_8640-easy (rerun-noise on the 10 is about +-2).
- **Line (ii) non-spin HD paired lower bound > -0.02: FAIL, -0.021** (mean +0.005; all 64: +0.009 [-0.035, +0.056]). The "near" gate (spins <= 4 and lower bound > -0.04) also fails on spins. No crashes. [E]
- The cost side improved a lot over the uniform low-pass: mean non-spin change +0.005 against -0.028, RC 0.334 against 0.315, 3000_3200-medium stays complete (0.900). The worst non-spin losses: 124-extreme-01 -0.24 (fg collision), 0418-hard-00 -0.21 (max_steps), 322492347634-easy -0.18, 034-easy -0.09. Gains: 034-hard-01 +0.68, 2800_3000-easy +0.16. [E]
- scene-0254-extreme-00 (a baseline "spinner" that still completes, HD 0.82) goes to fg_collision at step 41, HD 0.02, as in the uniform arm: the rule removes a spin that happened to be a good run; it is in the 10 and in neither the non-spin line nor the spin count.
- Why spins survive [E, `summary.md`]: in 5 of 6 surviving spins the raw plan direction crosses d1 = 4 deg within 1-3 steps of the first visible lean (053-medium-02: 0.2 -> 5.9 -> 14.7 deg; 0528: 3.3 -> 6.8 -> 12.5; 0041: 1.7 -> 3.6 -> 6.8 -> 11.2), after which the rule passes it at full gain, i.e. the pre-registered mechanism concern: the seed is not always small, and once it jumps over d1 nothing damps the growth. 8440_8640 and 5980_6180 spin late (steps 49 and 83), not at a launch.
- B2D: **not run** (gate not met, prereg: no retune, no extra arm). NAVSIM: not applicable (open loop).

## What this says

- [E] The selective rule is nearly free in HD (+0.005) but only halves the spins; the uniform low-pass removes them (1) at -0.028. The two arms sit on one trade-off line: what removes the spin is damping the loop gain at amplitudes above ~4 deg too, which is also what slows legitimate turns.
- [I] A larger d1 would not help (seeds jump past it faster); the lever is the loop gain between ~2 and ~15 deg, not the dead zone. A rule of the form "pass-through gain < 1 for 4-15 deg at launch (first N steps after a launch event only)" would be a third arm, with its own pre-registration; this arm is not retuned.
- [I] Single run per arm: the spins 6 vs 9-10 carries a noise of about +-2, so even the 10 -> 6 reading is weak.

## Method and limits

One run per arm and scenario; the rule rotates the logged plan inside HUGSIM `traj2control` (the zs logs keep the raw plan), checked by the smoke (0013-medium: spin -> complete 1.0; 3000_3200-medium complete 0.88). Same-day reruns of the base (two) bound the run-to-run HD noise (+-0.005) and spin noise (+-2). The parameters are tuned offline on 16 spin-event steps (small n). B2D hook not implemented.
