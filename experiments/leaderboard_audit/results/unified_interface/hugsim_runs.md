# HUGSIM unified-interface re-baseline, small stage (2026-10-05)

Shipped Cinque, 11 scenes (unified_hugsim_small.txt), one run per scene x arm, card 0, run dir `$DATA_DIR/runs/unified/hugsim` (box). Arms: `exam` = legacy exam config (iLQR, 5 s static warm-up); `d118` = decision 118 (lateral op-path, 5 s static warm-up); `spec` = lateral op-path, no static warm-up, dilate clock; `hold` = spec with hold clock, delay 0.2 s. Reference columns are the existing rows of `experiments/hugsim/results/op_control_stack_long/opctrl_long_runs.csv` (`base`, `base_rerun*`, `opctrl` = decision 118 single run, `opctrl_long`). Spin = decision 118 definition (`spin_analysis.analyse`, same as opctrl_report.py). `vmax` is nan in the reference csv for rerun3/4, opctrl and opctrl_long (not logged there).

Smoke (2 scenes, preset spec): both finished; interface.json has board hugsim, preset spec, lateral.exec op-path, history.warmup cold, warmup_s 0.0; sim.log has `op_ctrl {act, des, real, active, kmean}` lines as in cinque-opctrl; first step record reps 4 (d118 has reps 100 at step 0). Scene-0013 hit max_steps already in smoke (v 0.007 at the end).

Headline: removing the static warm-up (spec) breaks the launch. The d118 arm and the exam arm reproduce their reference runs; spec and hold do not launch.

## Per scene x arm (spin | end | HD | steps | vmax)

| scene | base | base_rerun | base_rerun2 | base_rerun3 | base_rerun4 | opctrl | opctrl_long | exam | d118 | spec | hold |
|---|---|---|---|---|---|---|---|---|---|---|---|
| scene-0013-medium-00 | SPIN bg_collision HD 0.055 n=22 vmax 4.5 | SPIN bg_collision HD 0.056 n=22 vmax 4.5 | SPIN bg_collision HD 0.055 n=22 vmax 4.5 | SPIN bg_collision HD 0.055 n=22 vmax nan | SPIN bg_collision HD 0.055 n=22 vmax nan | complete HD 1.000 n=50 vmax nan | max_steps HD 0.059 n=400 vmax nan | SPIN bg_collision HD 0.062 n=22 vmax 4.5 | complete HD 1.000 n=50 vmax 18.1 | max_steps HD 0.062 n=400 vmax 1.8 | max_steps HD 0.031 n=400 vmax 1.0 |
| scene-0051-easy-00 | complete HD 0.949 n=53 vmax 11.7 | complete HD 0.968 n=54 vmax 11.8 | complete HD 0.968 n=53 vmax 11.7 | complete HD 0.968 n=53 vmax nan | complete HD 0.968 n=53 vmax nan | complete HD 1.000 n=55 vmax nan | max_steps HD 0.076 n=400 vmax nan | complete HD 0.968 n=53 vmax 11.7 | complete HD 1.000 n=55 vmax 11.5 | max_steps HD 0.083 n=400 vmax 1.8 | max_steps HD 0.019 n=400 vmax 1.0 |
| scene-0254-extreme-00 | SPIN complete HD 0.821 n=76 vmax 17.1 | SPIN complete HD 0.826 n=82 vmax 17.3 | SPIN complete HD 0.818 n=76 vmax 17.1 | SPIN complete HD 0.818 n=76 vmax nan | SPIN complete HD 0.818 n=76 vmax nan | fg_collision HD 0.030 n=39 vmax nan | fg_collision HD 0.024 n=40 vmax nan | SPIN complete HD 0.818 n=76 vmax 17.1 | fg_collision HD 0.030 n=39 vmax 1.3 | fg_collision HD 0.024 n=41 vmax 1.8 | fg_collision HD 0.006 n=43 vmax 1.0 |
| scene-034-hard-01 | fg_collision HD 0.142 n=26 vmax 3.8 | fg_collision HD 0.142 n=26 vmax 3.8 | fg_collision HD 0.142 n=26 vmax 3.8 | fg_collision HD 0.142 n=26 vmax nan | fg_collision HD 0.142 n=26 vmax nan | max_steps HD 0.787 n=400 vmax nan | fg_collision HD 0.051 n=30 vmax nan | fg_collision HD 0.142 n=26 vmax 3.8 | max_steps HD 0.787 n=400 vmax 8.1 | fg_collision HD 0.041 n=28 vmax 1.8 | fg_collision HD 0.012 n=33 vmax 1.0 |
| scene-040-easy-00 | SPIN bg_collision HD 0.806 n=175 vmax 3.0 | SPIN bg_collision HD 0.819 n=169 vmax 3.4 | SPIN off_route HD 0.945 n=193 vmax 2.9 | SPIN off_route HD 0.945 n=193 vmax nan | SPIN off_route HD 0.945 n=193 vmax nan | complete HD 1.000 n=186 vmax nan | complete HD 0.955 n=79 vmax nan | SPIN complete HD 0.927 n=225 vmax 3.3 | complete HD 1.000 n=186 vmax 2.8 | complete HD 1.000 n=193 vmax 2.9 | max_steps HD 0.164 n=400 vmax 1.0 |
| scene-0418-hard-00 | complete HD 0.214 n=90 vmax 17.7 | complete HD 0.207 n=92 vmax 17.8 | complete HD 0.229 n=90 vmax 17.5 | complete HD 0.229 n=90 vmax nan | complete HD 0.229 n=90 vmax nan | max_steps HD 0.002 n=400 vmax nan | max_steps HD 0.078 n=400 vmax nan | complete HD 0.229 n=90 vmax 17.5 | max_steps HD 0.002 n=400 vmax 2.8 | complete HD 0.735 n=105 vmax 18.8 | max_steps HD 0.029 n=400 vmax 1.0 |
| scene-053-medium-02 | SPIN max_steps HD 0.432 n=400 vmax 6.4 | SPIN max_steps HD 0.438 n=400 vmax 6.3 | SPIN max_steps HD 0.431 n=400 vmax 6.4 | SPIN max_steps HD 0.431 n=400 vmax nan | SPIN max_steps HD 0.431 n=400 vmax nan | max_steps HD 0.703 n=400 vmax nan | complete HD 0.987 n=113 vmax nan | SPIN max_steps HD 0.431 n=400 vmax 6.4 | max_steps HD 0.703 n=400 vmax 4.5 | max_steps HD 0.711 n=400 vmax 4.8 | max_steps HD 0.139 n=400 vmax 1.0 |
| scene-0930-hard-00 | complete HD 1.000 n=39 vmax 11.8 | complete HD 1.000 n=39 vmax 11.6 | complete HD 1.000 n=40 vmax 11.3 | complete HD 1.000 n=40 vmax nan | complete HD 1.000 n=40 vmax nan | fg_collision HD 0.256 n=35 vmax nan | fg_collision HD 0.072 n=27 vmax nan | complete HD 1.000 n=40 vmax 11.3 | fg_collision HD 0.256 n=35 vmax 5.0 | fg_collision HD 0.121 n=27 vmax 1.8 | fg_collision HD 0.009 n=22 vmax 1.0 |
| scene-102751446607-medium-01 | SPIN bg_collision HD 0.150 n=66 vmax 4.2 | SPIN bg_collision HD 0.150 n=66 vmax 4.3 | SPIN bg_collision HD 0.141 n=66 vmax 4.5 | SPIN bg_collision HD 0.141 n=66 vmax nan | SPIN bg_collision HD 0.141 n=66 vmax nan | max_steps HD 0.150 n=400 vmax nan | max_steps HD 0.257 n=400 vmax nan | SPIN bg_collision HD 0.155 n=65 vmax 4.3 | max_steps HD 0.150 n=400 vmax 1.1 | max_steps HD 0.140 n=400 vmax 1.8 | max_steps HD 0.125 n=400 vmax 1.0 |
| scene-3000_3200-medium-00 | complete HD 0.737 n=51 vmax 9.3 | complete HD 0.723 n=51 vmax 9.3 | complete HD 0.723 n=51 vmax 9.3 | complete HD 0.723 n=51 vmax nan | complete HD 0.723 n=51 vmax nan | max_steps HD 0.273 n=400 vmax nan | max_steps HD 0.033 n=400 vmax nan | complete HD 0.723 n=51 vmax 9.3 | max_steps HD 0.273 n=400 vmax 5.0 | max_steps HD 0.278 n=400 vmax 5.0 | max_steps HD 0.000 n=400 vmax 1.0 |
| scene-8440_8640-easy-00 | SPIN bg_collision HD 0.557 n=256 vmax 3.8 | SPIN bg_collision HD 0.568 n=273 vmax 4.0 | SPIN bg_collision HD 0.472 n=271 vmax 4.0 | SPIN bg_collision HD 0.472 n=271 vmax nan | SPIN bg_collision HD 0.472 n=271 vmax nan | max_steps HD 0.778 n=400 vmax nan | max_steps HD 0.377 n=400 vmax nan | SPIN bg_collision HD 0.472 n=271 vmax 4.0 | max_steps HD 0.778 n=400 vmax 4.1 | max_steps HD 0.746 n=400 vmax 3.7 | max_steps HD 0.007 n=400 vmax 1.0 |

## Per arm (11 scenes)

| arm | n | spins | max_steps (stuck) | complete | fg | bg | off_route | crash | mean HD |
|---|---|---|---|---|---|---|---|---|---|
| base | 11 | 6 | 1 | 5 | 1 | 4 | 0 | 0 | 0.533 |
| base_rerun | 11 | 6 | 1 | 5 | 1 | 4 | 0 | 0 | 0.536 |
| base_rerun2 | 11 | 6 | 1 | 5 | 1 | 3 | 1 | 0 | 0.539 |
| base_rerun3 | 11 | 6 | 1 | 5 | 1 | 3 | 1 | 0 | 0.539 |
| base_rerun4 | 11 | 6 | 1 | 5 | 1 | 3 | 1 | 0 | 0.539 |
| opctrl | 11 | 0 | 6 | 3 | 2 | 0 | 0 | 0 | 0.544 |
| opctrl_long | 11 | 0 | 6 | 2 | 3 | 0 | 0 | 0 | 0.270 |
| exam | 11 | 6 | 1 | 6 | 1 | 3 | 0 | 0 | 0.539 |
| d118 | 11 | 0 | 6 | 3 | 2 | 0 | 0 | 0 | 0.544 |
| spec | 11 | 0 | 6 | 2 | 3 | 0 | 0 | 0 | 0.358 |
| hold | 11 | 0 | 8 | 0 | 3 | 0 | 0 | 0 | 0.049 |

## Paired HD differences on scenes where neither arm spins (per scene; bootstrap CI shown for orientation only)

**spec - d118 (warm-up effect)**: n=11 mean -0.185 [-0.474, +0.092]; 0013-medium-00 -0.938, 0051-easy-00 -0.917, 0254-extreme-00 -0.007, 034-hard-01 -0.747, 040-easy-00 +0.000, 0418-hard-00 +0.733, 053-medium-02 +0.008, 0930-hard-00 -0.135, 102751446607-medium-01 -0.010, 3000_3200-medium-00 +0.005, 8440_8640-easy-00 -0.032

**spec - exam**: n=5 mean -0.361 [-0.795, +0.107]; 0051-easy-00 -0.885, 034-hard-01 -0.102, 0418-hard-00 +0.506, 0930-hard-00 -0.879, 3000_3200-medium-00 -0.445

**hold - spec (clock effect)**: n=11 mean -0.309 [-0.500, -0.127]; 0013-medium-00 -0.031, 0051-easy-00 -0.064, 0254-extreme-00 -0.017, 034-hard-01 -0.028, 040-easy-00 -0.836, 0418-hard-00 -0.705, 053-medium-02 -0.572, 0930-hard-00 -0.112, 102751446607-medium-01 -0.015, 3000_3200-medium-00 -0.278, 8440_8640-easy-00 -0.739

## Launch (4 Hz steps; v at +1/+2/+3 s = steps 4/8/12, heading change = theta[5]-theta[0] in deg)

| arm | n | median v +1 s | +2 s | +3 s | median abs heading change first 5 steps (deg) | median signed |
|---|---|---|---|---|---|---|
| exam | 11 | 1.47 | 2.37 | 3.29 | 0.59 | -0.48 |
| d118 | 11 | 1.47 | 2.40 | 2.64 | 0.27 | +0.14 |
| spec | 11 | 1.45 | 0.98 | 0.48 | 0.39 | -0.19 |
| hold | 11 | 0.30 | 0.09 | 0.03 | 0.23 | -0.13 |

Per-scene launch (v+1, v+2, v+3, dtheta5):

- exam: 0013-medium-00 1.6/2.9/4.4/-1.1; 0051-easy-00 1.5/2.8/4.7/-0.0; 0254-extreme-00 1.1/1.6/2.6/-0.7; 034-hard-01 1.0/1.2/1.2/+0.1; 040-easy-00 1.6/2.4/2.3/-0.2; 0418-hard-00 1.3/2.1/2.5/+0.0; 053-medium-02 1.6/3.5/3.6/-6.0; 0930-hard-00 1.9/3.9/6.2/-1.8; 102751446607-medium-01 1.0/1.5/2.7/-0.5; 3000_3200-medium-00 1.3/2.2/3.3/-0.6; 8440_8640-easy-00 1.7/2.8/3.3/+0.6
- d118: 0013-medium-00 1.6/2.9/3.9/-0.2; 0051-easy-00 1.5/2.8/4.7/-0.0; 0254-extreme-00 1.1/1.2/1.0/+0.4; 034-hard-01 1.0/1.1/1.0/+0.3; 040-easy-00 1.6/2.4/2.4/+0.1; 0418-hard-00 1.3/2.2/2.6/+0.0; 053-medium-02 1.6/3.1/4.0/+0.4; 0930-hard-00 1.8/3.6/4.9/-0.5; 102751446607-medium-01 1.0/1.1/0.9/+0.3; 3000_3200-medium-00 1.3/1.8/2.6/+0.0; 8440_8640-easy-00 1.7/2.8/3.1/+0.5
- spec: 0013-medium-00 1.6/1.4/0.8/-2.3; 0051-easy-00 1.5/1.3/1.2/-0.5; 0254-extreme-00 1.4/0.9/0.6/+0.1; 034-hard-01 1.3/0.8/0.3/+0.7; 040-easy-00 1.2/0.6/0.1/-0.3; 0418-hard-00 1.5/0.9/0.3/+0.1; 053-medium-02 1.5/1.9/3.2/-0.2; 0930-hard-00 1.5/1.4/1.4/-1.1; 102751446607-medium-01 1.2/0.6/0.2/-0.0; 3000_3200-medium-00 1.4/1.0/0.5/-0.4; 8440_8640-easy-00 1.5/1.1/0.5/+0.6
- hold: 0013-medium-00 0.3/0.1/0.0/-1.1; 0051-easy-00 0.4/0.1/0.0/-0.6; 0254-extreme-00 0.4/0.1/0.0/+0.0; 034-hard-01 0.3/0.1/0.0/+0.4; 040-easy-00 0.3/0.1/0.1/-0.2; 0418-hard-00 0.3/0.1/0.0/-0.3; 053-medium-02 0.3/0.1/0.0/-0.1; 0930-hard-00 0.3/0.1/0.0/-0.7; 102751446607-medium-01 0.3/0.1/0.0/+0.0; 3000_3200-medium-00 0.5/0.1/0.0/-0.1; 8440_8640-easy-00 0.3/0.1/0.0/+0.1

## Notes

- Noise: `exam` reproduces `base_rerun3` on 10 of 11 scenes (same end, steps, HD to 3 decimals); 040-easy is the one that also varies between base reruns (SPIN complete 0.927 vs bg_collision 0.806 / off_route 0.945). `d118` reproduces the single `opctrl` run on all 11 scenes (same end, same steps, HD equal to 3 decimals). So run-to-run noise is about 0 here; the spec / hold differences are real.
- Why spec stalls (scene-0051-easy, same scene in all arms): d118 step 0 has reps 100 (the warm-up) and the plan speed ramps 0.9 -> 7.4 m/s over 15 steps with accel +1.4. spec has no warm-up: step 0 plan is 9 m/s (cold context, model_v 11), next step the plan speed collapses to 1.7 m/s (model_v 2.3 at 1.75 m/s ego), accel goes to -0.4 and the ego decays 1.7 -> 1.2 m/s over 15 steps; the model reproduces the low ego speed it sees in the 4 Hz render history instead of launching. hold: from step 0 plan speed 0.4 m/s, ego decays to 0.02 m/s by step 14 (hold clock sees each render held 5 steps = stationary-looking history), never moves; 8 of 11 runs end max_steps, vmax 1.0 (the init speed).
- Spin counts: spec has 0 spins like d118, but this is because the ego barely moves (6 max_steps, v <2 m/s), not because the spin mode is fixed. Exceptions with real driving in spec: 0418-hard (complete, HD 0.735, vmax 18.8; exam 0.229), 053-medium (max_steps HD 0.711, vmax 4.8, same as d118).
- Launch metrics are over runs with > 12 steps (all 11 per arm here). Heading change = theta[5] - theta[0].
- Spin scenes in the paired comparison: pairs are restricted to scenes where neither arm spins; exam spins on 6 scenes so spec - exam has n = 5. spec - d118 and hold - spec have n = 11 (no arm spins), bootstrap CI over 11 scenes is for orientation only.
- No crashes, no retries; whole small stage took 24 min on one card (19:55 - 20:20 box time). Servers used: one resident Cinque server.
