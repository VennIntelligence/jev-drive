# SLOT-RULE: does the nearest-keyframe slot rule ("breathing" frames) matter for the plan? (2026-10-22)

Diagnosis, no training, open loop only. Prereg (+ addendum 1) [plans/2026-10-22-slot-rule-prereg.md](../plans/2026-10-22-slot-rule-prereg.md).
Code: `scripts/warp_loop.py slot / slotreport`, `scripts/warp_loop.sh slot / slotident`, `scripts/slot_ident.py`; diagnostic switch
`SH30_SLOT` (`experiments/alpasim/lib/sh30_core.py`, default off). Raw tables: [slot_rule/slot_report.json](slot_rule/slot_report.json).
Box outputs: `$DATA_DIR/runs/defect_fix/warp_loop/slot/`, B2 dumps `$DATA_DIR/runs/defect_fix/native_gap/dump/B2/`.

**Short answer.** Yes, the rule matters a lot to the model. It is load-bearing, not cosmetic. Suppose each synthesised slot is warped
forward from the newest keyframe at or before it (the smooth rule, no breathing). Then the 4 s plan of the nuPlan candidate B2
`STANDFMXH-L18-s1` gets 13 % shorter on cruising decisions (arc ratio 0.870 [0.863, 0.878]). The plan speed at 1 s drops by 1.2 m/s, and 97 %
of decisions move by more than 0.3 m. NTV0-L18-s0 gives the same picture (0.881). Backward-only stays near neutral on arc (B2 0.998
cruising), but it still moves most plans by 0.3-0.5 m. The served nearest rule is the one every checkpoint was trained on
(`pp_prep` protocol W = `op_interp.synth_cpu` warp). So the user's prior holds in this sense: the breathing is in-distribution, and the
served path is right to keep it. Changing it serve-side would make the plans shorter, and slowness is already the model's known weak spot (d372 / d375).

## Setup

- Slot offsets (slot time minus source key time), slots -1.4 .. 0: nearest +0.1 / -0.2 / key / +0.2 / -0.1 / +0.1 / -0.2 / key;
  fwd +0.1 / +0.3 / key / +0.2 / +0.4 / +0.1 / +0.3 / key; bwd -0.4 / -0.2 / key / -0.3 / -0.1 / -0.4 / -0.2 / key. Keyframes, the
  fed row, route memory and the warp track (the native ego track) are unchanged.
- Inputs: native arena held closed-loop dumps of native_gap's 270 moving windows (6 shards). B2 was dumped fresh with the same command
  (`native_gap.sh dump B2`), and NTV0 uses the stored WARP-LOOP dumps. The decision set is k >= 3 (4 real keys): 1 890 decisions per model.
- Per decision, the slots are rebuilt with the served GPU lattice under each rule, and the plans go through the served compiled passes.
  Comparisons are always rule minus N (the rebuilt nearest rule).
- Classes: stopped v < 0.5; launching 0.5 <= v < 6 after an earlier stop in the session; turning v >= 2 with plan yaw at 4 s >= 15
  deg or history yaw over 1.5 s >= 10 deg; cruising the other v >= 2; other the rest.

## Gates

- **Identity** (`slot_ident.py`, 24 random cases, warm and cold): with `SH30_SLOT` off, `lattice` and `lattice_gpu` are bit-identical
  to the code before the switch. fwd / bwd change only synthesised slots, never keys or cold-start backwarp slots.
- **Control reproduces the stored plans** (prereg: p99 <= 0.06 m, max <= 0.15 m). The first NTV0 run used the op_torch interpreter
  and missed it (p99 0.079 m). The cause is fp16 output rounding against the served compiled passes: the error grows in steps of 1/32 m
  with plan distance. With the compiled passes (addendum 1, before any rule comparison was read), it passes:

| model | decisions | N vs served plan p99 / max (m) | F (dumped frames) vs served p99 / max | ADE(N, F) > 0.3 m |
|---|--:|--:|--:|--:|
| B2 `STANDFMXH-L18-s1` | 2 700 | 0.031 / 0.063 | 0.031 / 0.055 | 0.0 % |
| NTV0-L18-s0 | 2 700 | 0.031 / 0.063 | 0.031 / 0.058 | 0.0 % |

## Forward-only against nearest (k >= 3; window bootstrap 95 % CI)

| model | class | n | median arc N (m) | arc ratio fwd / N | Δ speed 1 / 2 / 4 s (m/s) | ADE median / p90 (m) | lateral ADE median / p90 (m) | share ADE > 0.3 m |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| B2 | all | 1 890 | 25.5 | 0.873 [0.866, 0.880] | -1.08 / -0.81 / -0.50 | 2.08 / 4.26 | 0.08 / 0.59 | **94.7 %** [92.9, 96.3] |
| B2 | cruising | 1 233 | 27.8 | **0.870** [0.863, 0.878] | -1.22 / -0.91 / -0.52 | 2.32 / 4.38 | 0.06 / 0.27 | 97.2 % |
| B2 | turning | 457 | 25.8 | 0.861 [0.850, 0.873] | -1.08 / -0.88 / -0.72 | 2.06 / 4.28 | 0.38 / 1.29 | 98.5 % |
| B2 | launching | 86 | 17.3 | 0.929 [0.901, 0.958] | -0.46 / -0.35 / -0.12 | 0.93 / 2.24 | 0.08 / 0.81 | 86.0 % |
| B2 | stopped | 36 | 2.9 | 1.038 [0.794, 1.305] | +0.03 / +0.02 / +0.08 | 0.20 / 1.21 | 0.01 / 0.03 | 41.7 % |
| B2 | other | 78 | 5.8 | 1.101 [1.014, 1.199] | -0.08 / +0.14 / +0.43 | 0.41 / 1.11 | 0.02 / 0.13 | 66.7 % |
| NTV0 | all | 1 890 | 20.0 | 0.882 [0.874, 0.891] | -0.97 / -0.66 / -0.23 | 1.60 / 4.05 | 0.08 / 0.59 | **93.7 %** [92.1, 95.2] |
| NTV0 | cruising | 1 199 | 22.1 | **0.881** [0.871, 0.891] | -1.11 / -0.74 / -0.20 | 1.89 / 4.28 | 0.06 / 0.30 | 97.3 % |
| NTV0 | turning | 430 | 22.0 | 0.857 [0.842, 0.872] | -1.01 / -0.80 / -0.58 | 1.85 / 4.11 | 0.36 / 1.23 | 98.1 % |
| NTV0 | launching | 107 | 18.0 | 0.928 [0.907, 0.952] | -0.49 / -0.37 / -0.08 | 0.96 / 2.08 | 0.08 / 0.82 | 87.9 % |
| NTV0 | other | 138 | 5.9 | 1.101 [1.065, 1.144] | -0.13 / +0.13 / +0.50 | 0.37 / 0.89 | 0.04 / 0.13 | 62.3 % |

The table shows a shorter plan at every moving class. The arc shortfall is largest at 1 s and shrinks toward 4 s. NTV0 has fewer than 20
stopped decisions in these windows, so that row is not reported. Lateral moves are small except on turns (median 0.36-0.38 m, p90 1.2-1.3 m).

## Backward-only against nearest (secondary, descriptive)

| model | class | arc ratio bwd / N | Δ speed 1 / 2 / 4 s (m/s) | ADE median / p90 (m) | lateral ADE median / p90 (m) | share ADE > 0.3 m |
|---|---|--:|--:|--:|--:|--:|
| B2 | all | 1.000 [0.995, 1.004] | +0.13 / +0.04 / -0.18 | 0.39 / 0.82 | 0.03 / 0.19 | 63.2 % |
| B2 | cruising | 0.998 [0.993, 1.003] | +0.15 / +0.05 / -0.26 | 0.41 / 0.82 | 0.03 / 0.12 | 66.0 % |
| B2 | turning | 1.012 [1.008, 1.017] | +0.09 / +0.07 / +0.09 | 0.33 / 0.70 | 0.11 / 0.35 | 56.0 % |
| B2 | launching | 0.966 [0.924, 1.001] | +0.08 / -0.12 / -0.39 | 0.46 / 0.97 | 0.04 / 0.22 | 68.6 % |
| NTV0 | all | 1.010 [1.004, 1.016] | +0.16 / +0.10 / -0.10 | 0.45 / 0.92 | 0.04 / 0.18 | 69.7 % |
| NTV0 | cruising | 1.010 [1.003, 1.018] | +0.20 / +0.13 / -0.16 | 0.49 / 0.99 | 0.03 / 0.12 | 74.0 % |

## Verdict against the registered lines

Lines: "matters" if cruising |R - 1| >= 0.03 or the share of decisions with ADE > 0.3 m is >= 10 %. "Does not matter" if both are below
half of that. Both lines are crossed by a wide margin for both models: cruising |R - 1| = 0.130 (B2) / 0.119 (NTV0), and the share is 94.7 % /
93.7 %. So the outcome is **matters**. The control sits at 0 % on the same share, so this is not reconstruction noise.

## What it means

- The plan is very sensitive to which keyframe each slot comes from, even with the keyframes, the ego row and the warp track held
  fixed. This fits WARP-LOOP (d375): the frames are the main speed channel, with gain 0.66-0.69. Forward-only puts the newest image pair
  (-0.2 s = key -0.5 warped +0.3 s, then key 0) on a real image change instead of a pure planar warp of key 0. Open loop, the model then reads
  less ego motion and plans about 1 m/s slower. Mechanism not tested beyond this.
- The breathing the user sees is part of the training distribution (nearest rule in `pp_prep` protocol W and in the served lattice). On
  the user's actual question, "does it matter for the model", the answer is that it does matter, but **as served it is the right input**.
  Replacing it at serve time with a smoother rule would move the plans out of distribution and make them shorter. A human viewer can
  show smoother frames (for example forward-only) without touching what the model gets.
- Backward-only keeps the arc but still moves most plans by 0.3-0.5 m: any rule change is a distribution shift at sub-metre level.

## Follows (per prereg)

- Serve stays on the nearest rule. `SH30_SLOT` stays a diagnostic.
- Proposed, **not launched**: a training-side test. Rebuild the training frames with the forward-only rule (`pp_prep` warp source =
  newest key at or before, plus on-policy rows from the same lattice), train the B2 recipe for a 1 000-step pilot against the same pilot under the
  nearest rule, and read the native arena subset. Question: is a rule trained in from scratch (no breathing) as good or better? Priority is low.
  The serve is consistent with training today, and nothing here shows the nearest rule costs score.

## Limits

- Open loop, one decision's counterfactual. Closed loop with `SH30_SLOT=fwd` was not read; it would be out of distribution anyway.
- One seed per model; 270 moving windows (the stopped class is thin: 36 decisions for B2, < 20 for NTV0).
- The mechanism reading (which image pair carries the speed) is an inference, not measured slot by slot.
