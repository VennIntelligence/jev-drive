# HUGSIM closed loop with `P2H10-F-s0` served through the stateless training protocol

2026-10-11. One closed-loop run, 64 scenarios, checkpoint seed 0, preset `spec`, paired against the served `spec` runs of the same
checkpoint. Pre-registration: [plans/2026-10-11-stateless-cl-prereg.md](../plans/2026-10-11-stateless-cl-prereg.md). Follows
[serving_trace.md](serving_trace.md) (decision 249: what the served queue receives) and [frame_ab.md](frame_ab.md) (decision 250: how far
the plan moves offline). Tables: [stateless_cl/](stateless_cl/). Code references are to commit `47221d76`.

This is the closed-loop test of the mismatch, not a change of what is served: the mode is the agent opt `stateless`, off by default.

**Answer.** Served through the protocol it was trained on (8-slot warp lattice, real clock, nothing dilated), the car launches faster,
as decision 250 predicted, and scores **lower, not higher**: HD 0.340 against 0.387, paired -0.047 [-0.104, +0.004] over 64 scenarios,
3 scenarios fixed and 17 broken (|dHD| > 0.05). The CI of HD touches zero; no-collision (-0.112 [-0.159, -0.066]), TTC
(-0.100 [-0.151, -0.049]) and PDMS (-0.092 [-0.139, -0.045]) are clearly worse. Foreground collisions go 27 -> 33 and completes
21 -> 19. The run-to-run spread of the reference is 0.001 in mean HD, so the difference is not repeat noise. Under the registered
readings this is "launch faster / plans longer, HD not better", with the components pointing to a net loss: the frame and clock mismatch
is not what limits `P2H10-F-s0` on HUGSIM, and removing it (together with the static warm-up and the desire pulse, see Limits) costs
collisions. One checkpoint seed, one run.

## What was run

| run | serving | scenarios | run dir (box, `runs/bench/hugsim/`) |
|:--|:--|--:|:--|
| `stateless` | agent opt `stateless`, `OP_CTRL` delay 0.2 | 64 (1 + 7 from the staged pilots, 56 in the full launch) | `P2H10-F-s0_spec-cd7ae2461e390` |
| `r3` (primary reference) | served queue, option off, the same commit, submitted together with `stateless` | 64 | `P2H10-F-s0_spec-rr3` |
| `rr1`, `rr2` | served queue, 2026-10-07 | 64 each | `P2H10-F-s0_spec-rr1`, `-rr2` |
| `r4` | served queue, option off, this commit, 4 scenarios on an otherwise idle card | 4 | `P2H10-F-s0_spec-rr4` |

    B=".venv/bin/python -m jevdrive.bench"
    $B run --model P2H10-F-s0 --bench hugsim --preset spec --opts '{"stateless": {}}' --controller-env '{"OP_CTRL": {"delay": 0.2}}' --scenarios all64
    $B run --model P2H10-F-s0 --bench hugsim --preset spec --repeat r3 --scenarios all64

**The mode.** Per simulator step n the agent keeps the packed frames of the last 7 steps and the state rows, builds the arguments of
`sh30_core.Core.plan` with `lib/parity_hugsim.core_inputs` (the function arm `b` of frame_ab used, moved from `frame_ab_torch.py` into the
lib so both call it): keys = the frames of steps n, n-2, n-4, n-6 at their own rear-axle poses, speed / acceleration / pose history at
face value. `experiments/hugsim/scripts/stateless_server.py` (envs/op-train, one `Core`, `synth = "cpu"`, `cold = "backwarp"`, one lock)
calls `Core.plan` unchanged and replaces the ONNX policy server and the bias server of the bench worker. Two differences from the offline
arm `b`, both registered: `lht` follows the scene's traffic flag (arm `b` used False everywhere), and since `Core.plan` returns the plan
only, the action head that `spec` steers with comes from one more policy pass on the same tokens (the server asserts that its plan equals
`Core.plan`'s within 1e-3 m). No desire input exists on this path.

**Cold start.** `sh30_core`'s own back-warp rule: 1 key at steps 0-1, 2 at 2-3, 3 at 4-5, 4 from step 6; 8 valid slots always. No static
warm-up (there is no queue to warm). This checkpoint was not trained with that rule.

## Time base: what follows the undilated clock

Decision 249's list of consumers, each changed inside the option:

| consumer | served (dilate) | stateless | code |
|:--|:--|:--|:--|
| frames in | one render = 4 model steps = 0.2 model s | no queue stepping; keys every 2 simulator steps = 0.5 s, 0.2 s lattice warped on the real clock | `experiments/hugsim/lib/zs_agent.py:257` (branch), `:243-253`; `lib/parity_hugsim.py:183-209` (`Stateless`), `:46` (`M2_BACK`) |
| speed to the policy (action curvature `act / max(1, v)^2`, plan curvature) | `ego_velo x 1.25` | `ego_velo` | `lib/parity_hugsim.py:204`; `scripts/stateless_server.py:58` |
| ego features (vx, ax, 4 poses) | scale 1.25: vx x 1.25, ax x 1.25^2, poses at simulator -1.875 / -1.25 / -0.625 / 0 s | scale 1.0: poses at -1.5 / -1.0 / -0.5 / 0 s | `lib/parity_hugsim.py:201` (`core_inputs(..., M2_BACK, 1.0, d)`), `:165` |
| plan out | `openpilot_to_plan(dilation = 1.25)` | `openpilot_to_plan(dilation = 1.0)` | `zs_agent.py:246` |
| lateral path (`op_ctrl.hugsim_steer`, `patches/hugsim/optional/op-ctrl.patch`) | dt 0.25 s, simulator speed, `OP_CTRL.delay` 0.25 simulator s = 0.2 model s | dt and speed unchanged (already simulator units); **delay 0.2 s** (the lateralDelay at face value, as `spec_hold`) | `--controller-env '{"OP_CTRL": {"delay": 0.2}}'`; the agent refuses to run the option with any other delay, `zs_agent.py:171-173` |
| longitudinal (iLQR on the plan, simulator time) | unchanged | unchanged | - |
| interface.json | `history.clock = dilate`, ... | `history.frames = warp-lattice`, `rate_hz = 2`, `clock = real`, `warmup = backwarp`, `lateral.delay_s = 0.2`, `command.channel = onehot` | `jevdrive/openpilot/interface.py:287-290` |

Bench wiring: `jevdrive/bench/hugsim.py:312-316` (the stateless server as the job's only server, on the policy socket), `:184` (8 more
cores per worker job). Refused together with the option (`zs_agent.py:166-170`): `parity`, `derot_*`, `lstab`, `op_lead`, `lead_margin`,
`op_clock = hold`. Thresholds that are in model units (`MIN_SPEED` 1 m/s, the 0.275 s lateral action time) now apply at face value.

## Gate: the serving mode reproduces arm `b` (verified)

`scripts/stateless_gate.py`, pool job `hugsim-stateless-gate`, [stateless_cl/gate.json](stateless_cl/gate.json),
[gate_rows.csv](stateless_cl/gate_rows.csv). The 989 recorded decisions of the 16 scenarios of the frame A/B run
(`P2H10-F-s0_spec-c11011ea9047d/.../zs_frames`), fed step by step through `parity_hugsim.Stateless` and the server over the wire, against
`runs/hugsim_frame_ab/out/torch/*.npz`; plan points within 4 s, tolerance 0.03 m.

| reading | result |
|:--|:--|
| against arm `b` (`lht = False`, as `b`) | 0 / 989 decisions over tolerance, max difference 0.0 m (bit-identical); action head max difference 0.0; key count per decision equal in 16 / 16 scenarios |
| wrong-arm control: the same outputs against `wd` (lattice on the dilated clock) | 988 / 989 over tolerance |
| against the logged served plan | 989 / 989 over |
| the 4 Singapore scenes replayed with their own flag (`lht = True`, what the closed loop serves), not gated | 124 / 125 decisions over tolerance against `b`, median 0.31-0.63 m, max 1.07 m |

Pass. The traffic flag moves the plan by about half a metre on left-hand-traffic scenes; arm `b` of frame_ab did not have it.

## Off path: the option off (verified at the level serving_trace.md used; not a bit-identity proof over 64)

With the option off the agent executes the same statements as before plus two `None` checks (`zs_agent.py:166`, `:257`), and bench starts
the same servers. Closed-loop check as serving_trace.md did for `trace`:

| pair | scenarios | step-0 plan identical | every sent plan identical | same end | HD identical |
|:--|--:|--:|--:|--:|--:|
| `r4` (this commit, option off, 4 slots on one card) vs `rr1` | 4 | 4 | 3 | 4 | 4 (0.2061 / 0.2481 / 0.9976 / 0.7913) |
| the pre-change traced run `c85e22e1821d5` vs `rr1` (serving_trace.md) | 4 | 4 | 3 | 4 | 3 |
| `r3` (this commit, option off, 36 slots on 6 cards beside the stateless run) vs `rr1` | 64 | 64 | 33 | 62 | 48 |
| `rr1` vs `rr2` (2026-10-07, run side by side) | 64 | 64 | 54 | 64 | 57 |
| the pre-change recording run `c11011ea9047d` (today, 6 slots) vs `rr1` | 16 | 16 | 5 | 16 | - |
| `r3` vs `rr1` on those 16 | 16 | 16 | 7 | 15 | - |

`r4` matches `rr1` exactly as the pre-change traced run did (3 of 4 scenarios identical over the whole episode, the fourth off by
millimetres from step 1, all four HD equal). `r3` agrees with `rr1` in fewer scenarios than `rr1` with `rr2`, which fails the registered
wording "not larger than `rr1` vs `rr2`"; the pre-change run of today differs from `rr1` at the same rate on its 16 scenarios (5 / 16
against 7 / 16 for `r3`), the differences start mid-episode at 6-10 mm, and the step-0 plan is identical in 64 / 64. So the spread between
repeats depends on the day and the load (TensorRT sessions sharing a card), not on this change. That last sentence is an inference; the
table rows are measured.

## Staged launch (verified)

| stage | check | reading |
|:--|:--|:--|
| 1 scenario (`scene-0051-easy-00`) | finished, scored | complete, HD 0.982 (reference 1.000), 39 steps |
| | interface.json | `warp-lattice`, 2 Hz, `real`, `backwarp`, delay 0.2, `onehot` |
| | plans and curvature finite | yes |
| | step-0 plan against the gate's replay | not done for this scene (left-hand traffic; the gate did not store its `lht` plans); done on the 6 right-hand scenes of stage 2 |
| 8 scenarios | crashes / missing | 0 |
| | launch stalls (stop at >= 6) | 0 / 8 (reference 1 / 8) |
| | completes | 5 (reference 4); HD 0.718 against 0.632 on these 8 |
| | step-0 plan against arm `b` of the same scenario | <= 0.001 m on the 6 right-hand-traffic scenes (0.48 and 0.68 m on the 2 Singapore scenes, the `lht` difference) |
| | plan arc 0-3 s / (3 s x v), warm, v > 3 m/s, in [0.9, 1.4] | 1.03 |
| | step-0 plan arc against the reference's, in [2, 8] | 3.1x to 6.0x per scenario |
| | agent time per decision: median <= 1 s, p99 <= 5 s | 155 ms, 395 ms (max 464 ms) |
| | wall per step <= 2x the reference | 0.58-1.66 s against 0.43-1.34 s |

Passed; the full batch was submitted together with `r3`. The pilot of 8 was better than the reference; the full 64 are not.

## Result: 64 scenarios, paired (verified)

[stateless_cl/paired.csv](stateless_cl/paired.csv), [classes.csv](stateless_cl/classes.csv),
[per_scenario.csv](stateless_cl/per_scenario.csv). `stateless - r3`, `jevdrive.stats.paired` over scenarios (B = 10 000, seed 0).

| metric | stateless | `r3` | difference [95 % CI] |
|:--|--:|--:|:--|
| **HD-Score** | 0.340 | 0.387 | **-0.047 [-0.104, +0.004]** |
| route completion | 0.483 | 0.510 | -0.028 [-0.081, +0.018] |
| no collision | 0.579 | 0.691 | -0.112 [-0.159, -0.066] |
| drivable area | 0.938 | 0.930 | +0.007 [-0.006, +0.024] |
| time to collision | 0.496 | 0.596 | -0.100 [-0.151, -0.049] |
| comfort | 0.998 | 0.991 | +0.007 [+0.002, +0.014] |
| PDMS | 0.478 | 0.569 | -0.092 [-0.139, -0.045] |

Against `rr1` and `rr2` the HD difference is -0.048 [-0.105, +0.002] and -0.048 [-0.104, +0.002].

**Fixed / broken** (|dHD| > 0.05 against `r3`): 3 fixed, 17 broken, 4 identical, 51 / 64 same end class. Fixed: `scene-0013-medium-00`
(fg collision -> complete, 0.248 -> 0.916), `scene-095-medium-01` (complete, 0.591 -> 0.976), `scene-106762673266-hard-00` (fg collision
-> off route, 0.044 -> 0.387). Broken: 8 that completed in the reference (3 now foreground collisions: `scene-090-hard-01` 0.996 ->
0.031, `scene-132384196576-medium-01` 0.974 -> 0.059, `scene-053-medium-02` 0.773 -> 0.098; 5 still complete with a lower score), 5 that
ended in a background collision, 3 foreground collisions that happen earlier (17-20 steps against 58-73), 1 off route.

**Failure classes.**

| run | HD | complete | fg collision | bg collision | off route | stuck | spin | launch stall | never reaches 2 m/s | steps |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| stateless | 0.340 | 19 | 33 | 6 | 5 | 0 | 1 | 1 | 3 | 2 620 |
| `r3` | 0.387 | 21 | 27 | 11 | 3 | 0 | 2 | 1 | 1 | 3 735 |
| `rr1` | 0.388 | 21 | 27 | 11 | 3 | 0 | 2 | 1 | 1 | 3 756 |
| `rr2` | 0.388 | 21 | 27 | 11 | 3 | 0 | 2 | 1 | 1 | 3 759 |

Transitions `r3` -> stateless ([transitions.csv](stateless_cl/transitions.csv)): complete 21 -> 18 complete + 3 fg; fg 27 -> 25 fg +
1 complete + 1 off route; bg 11 -> 5 bg + 5 fg + 1 off route; off route 3 -> 2 + 1 bg; spin 2 -> 1 spin + 1 off route. The one launch
stall is a different scenario in each: `scene-021-extreme-02` in the reference (in the stateless run it passes the 1.6 m/s line, still
collides before 2 m/s), `scene-100613054308-hard-00` in the stateless run (background collision in both). Episodes are shorter in 56 of 64 scenarios (30 % fewer steps in total): the car is faster
(mean speed over decisions 4.82 against 3.64 m/s, share above 10 m/s 14 % against 7 %, mean peak speed 9.1 against 7.7 m/s) and the
collisions come earlier.

By stratum ([paired_strata.csv](stateless_cl/paired_strata.csv)): PandaSet -0.130 [-0.288, -0.002], KITTI-360 -0.030 [-0.055, -0.008],
Waymo -0.055 [-0.188, +0.041], nuScenes +0.027 [-0.032, +0.121]; easy -0.033, medium -0.076, hard -0.064, extreme -0.014; 16 scenarios
each.

## Launch (verified; closed-loop readings on each arm's own states, except step 0)

[launch.csv](stateless_cl/launch.csv), [launch_pairs.csv](stateless_cl/launch_pairs.csv).

| quantity | stateless | `r3` | arm against reference |
|:--|--:|--:|:--|
| time to 2 m/s, median (s) | 1.00 | 1.50 | -0.62 s [-0.69, -0.56] over the 61 scenarios where both reach it |
| time below 2 m/s per scenario (s) | 2.85 | 5.96 | -3.11 s [-4.73, -1.75] |
| mean plan arc 0-3 s on decisions below 2 m/s (m) | 5.58 | 3.82 | ratio 1.46 [1.33, 1.65] |
| step-0 plan arc 0-3 s (m), same initial state | 5.25 | 1.25 | ratio 4.20 [4.01, 4.45] |
| plan arc 0-3 s / (3 s x v), warm, v > 3 m/s | 1.049 | 1.034 | ratio 1.014 [0.994, 1.036] |

Decision 250's launch prediction holds in closed loop: +46 % plan length below 2 m/s (offline +44 %), 4.2x at step 0 (offline 4.2x).
Above 3 m/s the offline +9 to +12 % does not appear: relative to the speed actually driven the plans are 1.4 % longer, CI including 0.
The car is nevertheless faster overall, so the longer launch plans carry over into speed, not into a longer plan per unit of speed.

## Reference spread (verified)

Mean HD of the three reference runs: 0.3870 / 0.3882 / 0.3880. Paired: `r3 - rr1` -0.001 [-0.004, +0.001], `r3 - rr2` -0.001
[-0.003, +0.001], `rr1 - rr2` +0.000 [-0.000, +0.001]; largest single-scenario |dHD| 0.072 / 0.047 / 0.025; fixed / broken counts 0 / 1,
0 / 0, 0 / 0; class counts identical in all three ([identity.csv](stateless_cl/identity.csv)). Registered noise floor: max(0.005,
0.001) = 0.005. The stateless difference is ten times that in the mean and 17 broken against at most 1 in a repeat. HUGSIM has no random
seed on this path, so this spread is execution noise only; the spread across checkpoint seeds is not in it (decision 241 has the
two-seed HUGSIM numbers).

## Latency (verified)

[latency.csv](stateless_cl/latency.csv). Per decision, steps > 0, median / p95 / p99:

| | stateless (64 run) | `rr1` | `r3` (ran beside the stateless run) |
|:--|:--|:--|:--|
| server time (ms) | 116 / 144 / 159 | 29 / 88 / 112 | 44 / 106 / 124 |
| agent round trip (ms) | 126 / 271 / 374 | 31 / 95 / 157 | 60 / 170 / 485 |
| agent total (ms) | 152 / 293 / 397 | 62 / 131 / 202 | 87 / 197 / 509 |
| step 0, agent total (ms) | 224 | 493 (100 warm-up steps) | 794 |
| wall per simulator step (s) | 1.08 | 0.81 | 2.01 |

Stateless server stages (median ms): lattice on the CPU 54, encoder 20, policy 21, second policy pass for the action head 17; a single
stream in the gate takes 88 ms per decision. Waiting for the server lock: median 0, p95 142 ms with 6 scenarios per server. So the mode
costs about 90 ms more per decision (2.5x the agent time), 0.3 s per simulator step, well inside the loop's tolerance (it blocks on the
plan; the bench watchdog is 420 s of silence). The whole 64 took 3.5 min of wall on 6 cards and fewer scenario-seconds than the reference
(2 128 against 2 849 for `rr1`) because the episodes are shorter. Cards used in total (gate, pilots, both full runs, `r4`): about 1.6
card-hours (wall x cards, estimated from job times).

## Limits

- One checkpoint seed, one run. The HD CI includes 0 at its upper end (+0.004); the collision components exclude it.
- The option changes a bundle and the closed loop measures its sum: frame source, 9 -> 8 slots, clock, the desire pulse (gone), the 5 s
  static warm-up (replaced by a back-warp cold start this checkpoint was not trained with), the lateral delay value (0.25 -> 0.2 s), and
  the traffic flag now reaching the policy through `Core.plan`. No attribution was run. In particular the faster launch contains both
  the undilated clock (decision 250, +44 % offline on warm decisions below 2 m/s) and the cold start (4.2x at step 0).
- Not a statement about the protocol on other boards, about other checkpoints (`AP2H10-AB` was trained with the cold-start rule), or
  about a checkpoint trained on HUGSIM-like serving.
- The 8 pilot scenarios are part of the 64 (run minutes earlier, same commit and configuration).
- Launch readings other than step 0 compare decisions at different states.
- `r3` shows more repeat-to-repeat plan divergence than `rr1` / `rr2` (see Off path); its scores do not.

## Reproduce

Box, repo root. Gate: `cl submit --name hugsim-stateless-gate --vram 8 --cpu 12 -- .venv/bin/python
experiments/hugsim/scripts/stateless_gate.py --run-dir $DATA_DIR/runs/bench/hugsim/P2H10-F-s0_spec-c11011ea9047d --ab-out
$DATA_DIR/runs/hugsim_frame_ab/out --out $DATA_DIR/runs/hugsim_stateless_cl/gate` (2 min). Runs: the two `bench run` lines above. Tables:
`.venv/bin/python experiments/hugsim/scripts/stateless_cl_report.py --runs stateless=<dir> r3=<dir> rr1=<dir> rr2=<dir> --out
$DATA_DIR/runs/hugsim_stateless_cl/report`.
