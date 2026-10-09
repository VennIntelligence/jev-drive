# COL1: did the frozen model's own lead outputs see the struck object in time? (pre-registration, 2026-10-09)

Lane COL1 (collision research, AlpaSim nuPlan public scenes on the GPU box + PAI scenes on the Tokyo box). Written and pushed before any
lead-output number of a closed-loop AlpaSim rollout is computed or looked at. The collision taxonomy itself is a description of existing
runs and is not covered here. Nothing is trained, no rule is put in the loop, nothing is submitted. Simulator state (object boxes, logged
path) is a label and an oracle only. No WA-JEPA weights or features.

## Question

Shipped openpilot does not avoid a lead through its plan alone: the model's `lead` / `lead_prob` heads feed a separate longitudinal planner.
Our AlpaSim driver returns only the `plan` slice (experiments/alpasim/lib/sh30_core.py: `out[0, self.pi]`); the lead heads are computed in
the same forward pass and discarded. In the rollouts that ended in an at-fault collision, did those heads report the struck object early
enough for a stop, and what is the ceiling of the cheapest rule that would use them?

How this differs from the settled reads: decision 157 (fixed 2 m lead margin on stored plans: HUGSIM / navtest offline), 140 (resume rule,
HUGSIM), 179 and 203 (navtest token-level AUC of lead clearance / native lead outputs against open-loop, non-reactive 4 s NC labels).
None of them is a time-resolved read on closed-loop rollouts with the simulator's gap as the label, and none is on AlpaSim or on PAI's 20 s
scenes behind real queues. No mechanism of those decisions is re-tested; no rule is implemented here.

## Units

- **C**: every at-fault-collision rollout of the taxonomy. nuPlan: P2H10-F-s0 and -s1 on the 1491 public scenes (other recipes with logs are
  reported separately, never pooled into a line). PAI: P2H10-F-s0 on every scene run on the Tokyo box (pai1's `a10_c4` plus COL1's runs of
  further scenes of `pai_scenes_40.tsv`, one plan per call, Harmonizer off).
- **C_A**: the subset whose struck object is class A of the taxonomy (a vehicle ahead in the ego's lane: stopped, or moving the same way).
- **N** (controls): rollouts without any collision flag. nuPlan: P2H10-F-s0 re-run with inputs kept on the scene lists of the first two
  public shards as originally chunked. PAI: every non-collision rollout, decisions up to the first step at which the ego is more than 4 m
  from the logged path (after that the rendering is outside the reconstruction).
- A decision = one `drive` call (nuPlan 2 Hz, PAI 10 Hz).

## Signals (per decision, from an offline replay of the logged driver inputs through the real driver class)

The replay must reproduce the run's plan (4 s end point within 0.05 m, share of decisions reported); decisions that do not are dropped and counted.

- p = sigmoid(lead_prob[0]); d = lead hypothesis 0 at t = 0, x, minus the distance from the camera to the front bumper (session
  calibration and ego box); v_l = the same entry's speed. Parsing as experiments/op_parity/scripts/nc_slow.py (`lead[:72]` -> 3 x 6 x 4).
- Two sources on the same vision tokens: **P0** = the shipped policy weights without the adapter bias (primary: this is openpilot's own
  lead output); **FT** = the served checkpoint's own output vector (secondary: free in the serving forward pass).
- Labels: g = longitudinal bumper gap to the struck object (controls: the nearest object whose box overlaps the ego's straight-ahead
  corridor of ego width + 0.5 m, within 80 m), c = closing speed along the ego heading, v_e = ego speed.
  a_need = c_+^2 / (2 max(g - 0.5 c_+ - 1.0, 0.1)): the constant deceleration that, begun 0.5 s later, stops the closing 1 m short if the
  object keeps its speed.

## Quantities

1. **Seen in time (S_time)**, per rollout of C_A: a decision is a *hit* if p >= 0.5 and |d - g| <= max(2 m, 0.3 g). The rollout is *seen in
   time* if some hit decision in its final approach (the last 8 s before impact, or since the start of the scene) has a_need <= 4 m/s^2.
   Reported with: lead time of the first such decision, p and d - g against g in bins (0-5, 5-10, 10-20, 20-40, > 40 m), and the same for FT.
2. **Guard ceiling**, rule G(p*, a*), model outputs and ego speed only: trigger when p >= p* and
   (v_e - v_l)_+^2 / (2 max(d - 4 m, 0.5)) >= a*, or when v_e < 0.5 m/s, p >= p* and d < 6 m (hold at standstill).
   Primary point p* = 0.5, a* = 1.5 m/s^2; sensitivity grid p* in {0.3, 0.5, 0.7, 0.9} x a* in {1.0, 1.5, 2.5}.
   - *Turned*: a rollout of C in which G triggers at a decision whose label says a stop is still possible (a_need <= 4 m/s^2). Upper bound:
     it assumes the stop causes no other zero and that the struck object is the only threat.
   - *False alarm*: a trigger on a control decision whose label has a_need <= 0.5 m/s^2 or no object in the corridor. Per decision, and per
     scene (a scene counts when false triggers hold for >= 1.0 s: 2 consecutive decisions on nuPlan, 10 on PAI) = "extra slow scenes".
   - Also reported: triggers on controls that the label supports (a_need > 0.5), and the share of C rollouts with a trigger that came too late.

## Lines (per track; a track with fewer than 8 rollouts in C_A is descriptive only for line 1)

| Line | Reading |
|---|---|
| L1: S_time (P0) >= 0.60 | the frozen model sees the lead in time in a large share of class-A collisions |
| L1: S_time (P0) <= 0.30 | it does not; a lead-based rule cannot be the fix, the signal has to be learned |
| L1 in between | partial; reported as such |
| L2: at the primary point, turned >= 0.40 of all rollouts in C **and** false-alarm scenes <= 0.15 of N | an in-loop test of the rule is worth a lane |
| L2 otherwise | not as a rule; whatever the grid shows is reported as post hoc |

Bootstrap 95 % intervals over rollouts (10 000 draws, seed 0) for S_time, the turned share and the false-alarm scene share; with these
sample sizes they will be wide, and the point estimate decides the line.

## Limits known in advance

Offline read on rollouts whose traffic does not react (nuPlan) or replays the log (PAI); a stop changes the future and may cause a rear
contact (not at fault) or lost progress, neither is estimated beyond the false-alarm counts. On nuPlan six of the eight slots per decision
are synthesised frames (decision 205), and decision 202 found no class-A collision among SH30 / AP2 zeros on 400 scenes, so C_A may be
small there. PAI: one checkpoint, one rollout per scene, at most 40 scenes.

## Amendment 1 (2026-10-09, before the run it describes): PAI diagnostic arm for a serving-side defect

Found while checking the PAI inputs (description of existing runs, 20 scenes): the trajectory the driver returns is a linear
resampling of the plan's 0.5 s points, so its first segment has the plan's mean speed whatever the ego's speed is. At the hand-over
(1.7 s) that speed is 6-7 % below the ego's in the two scenes above 23 m/s, and in both the nonlinear MPC's first commands are
saturated braking (-9 m/s^2) and steering (0.66 / 0.73 rad): both rollouts spin and are scored offroad. The four scenes between 16 and
23 m/s (-3 to -11 %) show braking to -4.6 m/s^2 without the steering. At 5-12 m/s the served plan's first segment is 5-28 % above the
ego's speed (the shipped policy on the same tokens: -7 to +5 % in 7 of 8), and the ego accelerates at 1.5-3.8 m/s^2 after the hand-over.

Arm `v1`: `lib/col1_pai_driver.py`, PAI_VCONT = 1.0 s (the plan's path re-timed to start at the ego's speed and join the plan's speed
profile after 1 s), P2H10-F-s0, the 10 scenes and chunking of `a10_c4` (4 concurrent), Harmonizer off. Nothing else changes.
No line; it is a diagnostic, not a candidate. Reported: zeros by flag against `a10_c4`, per-scene score and flag, and for the two
scenes above 23 m/s the largest steering and braking command in the 1.2 s after the hand-over. Stated in advance: if the defect is the
cause, those two rollouts no longer spin (|steer| < 0.1 rad in that window); nothing is expected or claimed for the collisions, whose
approach speeds the arm lowers only through the 1 s ramp. The simulator's repeatability on this track is not known (pai1's three runs
of these scenes gave the same 8 zero scenes), so single-scene changes other than the two spins are not interpreted.

## Amendment 2 (2026-10-09, after arm `v1` on the 10 `a10_c4` scenes was read, before any further `v1` run)

Read so far (10 scenes): mean scene score 0.154 -> 0.333; zeros 8 -> 5; 4f779a92 no longer spins. The arm is extended, unchanged, to the
other 30 scenes of `pai_scenes_40.tsv`, each list with the chunking of its baseline run (`b1`, `b2a`, `b2b`). Still a diagnostic of the
serving-side defect, not a candidate and not a line. Reported on all 40 scenes: mean scene score and zeros by flag, baseline against `v1`,
the paired per-scene difference with a scene-bootstrap 95 % interval (10 000 draws, seed 0), and the hand-over steering / braking of every
scene above 16 m/s. One baseline run per scene; no repeat of the baseline is made, so the interval does not contain the simulator's own
run-to-run spread.
