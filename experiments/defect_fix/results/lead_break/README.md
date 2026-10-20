# LEAD-BREAK: why RB0-F's own plan stopped braking for a closing lead (2026-10-20)

This is a diagnosis. Nothing was trained and nothing new was rolled out.
- Prereg: [../../plans/2026-10-20-lead-break-prereg.md](../../plans/2026-10-20-lead-break-prereg.md), pushed at cdf4887d before any model output.
- Code: `scripts/lead_break.py` (`select`, `cf`, `nav`, `report`) and `scripts/lead_break.sh`. Box output: `$DATA_DIR/runs/defect_fix/lead_break/`.
- Compute: one GPU job of 1.5 min, one CPU row rebuild of 1.5 min, one CPU navtrain job of 1 min. One seed, one read.
- Files: [report.md](report.md) (all cells, all paired differences) and [report.json](report.json).

## Answer

**Mechanism (b) holds: the adapter's ego-state input overrides a lead brake that the policy weights still produce.**
- Take RB0-F-s0 on its own crash states with `present = 0` (the anchor / cons condition). Its plan clears the lead as often as P0 does.
- That arm recovers 96 % [92, 99] of the P0 - RB0-F gap.
- Zeroing only the 16 ego-state entries recovers 94 %. Zeroing only the route memory recovers 2.5 %.
- Feeding a fed ax of -2 m/s² alone recovers 77 %: the plan brakes when the history says it is already braking (d218, d331).

**The slot protocol is not the cause.**
- (a1) Serving the trained protocol (keyframes plus warp) makes RB0-F worse, not better: -0.049 [-0.099, -0.006].
- (a2) Warp slots remove part of P0's cue (-0.139, 29 % of the gap; registered as "partial"), and only at TTC < 2 s.
- With warp slots and the adapter off, RB0-F equals P0 on warp slots exactly. The weights read the lead from the warp frames.

**(c) Navtrain does not lack lead-braking labels. It lacks rows where the braking has not started yet.**
- 12.4 % of navtrain rows (12 850 rows, 781 logs) are brake-needed under the same definition.
- 84 % of those rows already have a decelerating fed ego state (median ax -0.96 m/s²). Only 16 % are initiation rows (2 100), and the human brakes more gently on them (median a02 -0.66 against -1.16).
- RB0-F's PAI crash states are the opposite case: median fed ax +0.36, v0 12.1 against 4.3 m/s, closing 5.1 against 3.4 m/s, TTC 1.6 against 3.3 s.
- They are out of distribution by the registered rule: median 1 navtrain neighbour, and 47 % have none.

## Data

**States.** RB0-F-s0's native PAI 281 read (d405), the 50 AF-front zero scenes.
- The rows were rebuilt from the stored rollout.asl with `onp_rows.py`. G-in passed: 9 915 / 9 915 decisions match drive.jsonl.
- Decisions are taken at 5 Hz, before the first event, with v0 >= 2 m/s, all slots real, and |e0| <= 1.5 m.
- The lead is the nearest box within 1.75 m of the logged path. Its future is the box's actual pose in this rollout.

**Windows.**
- BRK (brake needed): an ego holding its speed overruns the lead's tail within 4 s. 324 decisions in 30 scenes.
- FOL (control, following): 87 decisions in 10 scenes.
- The other 20 scenes have no BRK decision under this definition, for example a side or crossing hit, or the ego off the path.

**Gates.** G0 passed: RB0-F's F arm against its logged plan has p99 0.059 m, max 0.063 m, and 99.5 % of rows within 2 fp16 steps.
G1 passed: P0 - RB0-F clear rate on BRK is +0.485 [+0.380, +0.588].

## BRK: share of plans that stay behind the lead over 4 s (95 % scene-cluster CI)

| model | F | OFF (present 0) | ZR (route 0) | ZE (ego state 0) | AXN (ax -2) | KW (warp slots) | KWOFF |
|---|--:|--:|--:|--:|--:|--:|--:|
| P0 | **0.633** [0.516, 0.747] | | | | | 0.494 | |
| RB0-F-s0 | **0.148** [0.095, 0.206] | **0.614** [0.494, 0.728] | 0.160 | 0.602 | 0.522 | 0.099 | 0.494 |
| NTV0-L18-s0 | 0.182 [0.124, 0.240] | 0.614 | 0.201 | 0.509 | 0.515 | 0.127 | 0.494 |

**Plan 0-2 s acceleration.** P0 -2.36, RB0-F -0.29, RB0-F OFF -2.30 m/s². RB0-F's 4 s arc is 38.8 m against P0's 29.0 m.
NTV0 inherits the defect (+0.034 over RB0-F), and with its inputs off it is P0 again.

**By TTC (exploratory).**

| TTC | P0 F | RB0-F F | RB0-F OFF | P0 KW | n (scenes) |
|---|--:|--:|--:|--:|--:|
| 0-1 s | 0.37 | 0.00 | 0.36 | 0.06 | 107 (27) |
| 1-2 s | 0.70 | 0.01 | 0.67 | 0.56 | 86 (23) |
| 2-3 s | 0.68 | 0.18 | 0.65 | 0.69 | 68 (20) |
| 3-10 s | 0.94 | 0.56 | 0.94 | 0.94 | 63 (19) |

On each scene's first 3 BRK decisions (median TTC 3.0 s, v0 7.4 m/s), P0 clears 0.88, RB0-F 0.40 and RB0-F OFF 0.87. So the failure is
in starting to brake.

**FOL control.** Every arm clears about 1.0. RB0-F still accelerates behind a steady lead (a02 +0.65 against P0's -0.32), and turning its
inputs off removes that too.

## What it implies for training (with confidence)

1. **High confidence for this diagnosis.** The lever is the adapter's ego-state path, not the slots, not the route memory, and not the
   policy weights.
   - The cons / anchor term kept the inputs-off model equal to P0 (-0.019 on these states).
   - With the inputs on, the bias from ego speed and acceleration decides the longitudinal plan.
   - Real-frame training slots would at best restore the close-range part of P0's frame cue. That is second order, and serving warp slots
     on PAI is already known to hurt (d376).
2. **Medium confidence for the fix.** Break the shortcut "braking is in the history" on the rows where braking must start.
   - On navtrain brake-needed rows, feed a non-decelerating history (ax 0, constant-velocity poses) and keep the human label.
     There are about 12.8 k candidates, against 2.1 k real initiation rows. This is targeted ego-state dropout, not global removal.
   - Add own-state rows (flat history, closing lead) with a one-sided lead-tail hinge. RB0-F trained with `agent_lam = 0`.
   - Pilot gate: this exact instrument. RB0-F F must rise toward OFF on BRK with FOL and navtest held, before any closed-loop read.
3. **Low confidence / known negatives.** Removing the ax input globally was tried (d221, TR1). The speed continuation moved into the frame
   path, and putting lead outputs into the adapter did nothing. JTB's braking rows with an arc target made front collisions worse (d353).
   A targeted augmentation is a different recipe, but those results argue for a small pilot first.

## Limits

- Open loop, one seed, one read of RB0-F's own states (30 scenes carry BRK decisions).
- P0 here is a counterfactual on RB0-F's states, which are already late (median TTC 1.6 s). Its 0.63 is not a closed-loop number.
- The lead comes from the logged-path corridor and the actual future boxes of the rollout. Crossing or lateral hits are outside BRK.
- The navtrain path is the human's own future plus a straight extension. The neighbour count (c1) mixes "out of distribution" with "the
  model's own late states", which humans rarely reach.
- ZE (speed 0) is outside the training distribution and is exploratory. AXN is a single fixed value.
