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

## Addendum (2026-10-03 02:35 box time, before any score of this run was read)
Read so far: opposite-side rates only (stage 1 / stage 2): base 1.3% / 5.7%, rot0 15.3% / 6.1%, straight 6.2% / 2.75%
(straight reproduces decision 88: 15.4% of the 312 still opposite). rot0 keeps the lateral path while fixing the heading,
which seems to read as sideways drift. Third variant, chosen after these rates (labelled so):
- `straight_keys`: each history frame (keys and context frames) re-projected from its own pose to (-s(t), 0, 0), i.e. the
  real frames moved onto the straight track (keeps the other agents' motion that `straight` drops). Same decision rule.

## Addendum 2 (03:45 box time; post hoc, after the navhard scores of rot0 / straight were read)
Read: official navhard combined base 33.33, rot0 32.67 (-0.66 [-3.80, +2.37]), straight 31.99 (-1.34 [-5.22, +2.40]): both
rejected. Stage 2 rises (46.90 -> 52.69 / 50.75), stage 1 (450 real scenes) falls (71.70 -> 61.69 / 61.70). Stage-2 history
images agree with their pose yaw (phase-correlation slope 0.95, as stage 1 0.97 and navtest 0.99): no image-vs-odometry gate.
On the opposite-side flips, the lower lateral plan std of the two rollouts picks the right one (AUC 0.86 rot0, 0.78 straight).
- `sel-<rule>`: run both rollouts (as shipped and with the rule); per token keep the rule's plan iff the sum of the plan's
  lateral position std over the knots t <= 4 s is lower than the shipped plan's. No threshold, no tuning. Reference-free (the
  model's own output). Scored on full navhard and navtest with the same decision rule; labelled post hoc.
