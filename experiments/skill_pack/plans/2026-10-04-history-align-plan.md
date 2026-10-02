# History alignment as a reference-free input rule (night 2026-10-03, lane A; pre-registration)

Written before any number of this run was read. Follows decision 88 (the wrong-direction class follows the ego rotation of
the history; removing it was an intervention on 312 tokens, not a rule).

## Pipeline (as shipped, only the history frames change)
Native Cinque, `scripts/op_lb.py run --frames gimm --model cinque` (TensorRT, desire `none`, 31 steps at 20 Hz, zero state),
`OPI_ROOT=op_lb experiments/op_openloop/lib/op_interp.py nav-export` (adapter `base`), official scorers via
`experiments/op_openloop/archive/op_interp_score.sh` (navhard: v2 two-stage EPDMS; navtest: v1 PDMS).
The rule is `--align <rule>` -> `jevdrive.op_interp.align_history`, applied to every token of both splits; inputs: the four
2 Hz ego poses and velocities and the camera position, i.e. what the agent receives. No reference, no scorer, no stage label.

## Rules
- `rot0` (primary): each of the 9 history frames (3 keys + 6 GIMM context frames) re-projected from its pose (x, y, yaw) to
  (x, y, 0) through the road-plane warp of op_interp (current heading, translation kept). t0 frame unchanged.
- `straight` (second variant): the t0 key warped back along a straight track with the history's arc length (the
  `warp_no_yaw_track` arm of decision 88, in op_lb's 10-frame layout).
Both are run on navhard; navtest for every rule that is not rejected on navhard.

## Readouts
- navhard: official combined EPDMS (+ stage 1 / 2) per arm; paired difference vs baseline with a 95% bootstrap CI over the
  scene-mapping groups (5 000 resamples, as offroad_gain.py), on per-token rows of the official CSVs.
- navtest: official PDMS per arm; paired per-token difference with a 95% bootstrap CI.
- Wrong-direction class: opposite-side rate at stage 2 (definition of the diagnosis plan) per arm; on the 312 baseline
  opposite-side tokens: share still opposite and stage-2 DAC; DAC-failure classes of the diagnosis (fixed / broken).
- Check: `straight` should reproduce decision 88's 15% still-opposite on the 312 within ~5 points (else the
  layout difference matters and is reported).

## Decision rule (fixed now)
- Adopt a rule if navhard combined EPDMS delta > 0 with CI lower bound > 0 AND navtest PDMS delta >= -0.30 (point estimate).
- Ambiguous if navhard delta > 0 with CI including 0, or navtest delta in [-0.60, -0.30): report, try the next variant.
- Reject if navhard delta <= 0 or navtest delta < -0.60.
- If both rules pass, the one with the larger navhard delta is named; both are reported (two variants tested).
- Any further variant chosen after reading these numbers is labelled post hoc.
