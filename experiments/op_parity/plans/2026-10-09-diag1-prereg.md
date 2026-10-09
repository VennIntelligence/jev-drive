# Lane DIAG1 pre-registration: (1) FIX1's two serving fixes as a post-processing of WOD open-loop predictions; (2) what nuPlan adaptation changed longitudinally

Written 2026-10-09, committed and pushed before any number of this lane is read. Measurement only: existing checkpoints, no training,
no simulator runs, nothing submitted. No WA-JEPA weights or features. Logged futures and simulator state are labels / oracles only.

Already settled, not re-tested (how this lane differs):
- Decision 162: on WOD val, P2H10 path + shipped speed profile +0.062, shipped path + P2H10 speed profile -0.464 (vs shipped); the loss is the
  adapter bias constant. Here that swap is a reproduction gate, extended to SH30, to lead / no-lead and speed strata, and given a navtest
  counterpart with the official scorer. The bias decomposition itself is not repeated on WOD (stored variant predictions are re-read only).
- Decisions 167 / 169 / 174: the 0.5 m/s adapter gate (navtest -1.50, HUGSIM -0.084; WOD in-domain +0.39 on standstill). Not re-run; here the gate
  is one of the ablation switches whose *plan geometry* (not score) is read.
- Decisions 164 / 169: WOD-trained arms (WP2 / WLG). This lane reads the navtrain-trained arms.
- Decision 203: whether "where to slow down" can be selected from inputs. Here no selector: the plan's own response to the model's lead output.
- Decision 207: swap of plan vs log. Here the swap is adapted plan vs shipped plan.
- Decision 209: lateral recovery. Here longitudinal only.

## Job 1: FIX1's fixes on WOD-E2E val, open loop

Inputs: stored predictions `processed/wod_zeroshot/preds/op_cinque[_<tag>]/<frame>.npz` (`wod`: 20 waypoints at 0.25 .. 5 s, rear axle; decision 155
harness, 9 real slots). Arms: P2H10 = P2H10-F-s0 + s1, SH30 = SH30-F-s0 + s1, control shipped (`op_cinque`). An arm's per-frame score is the
mean of its seeds' per-frame scores (decision 217).

(a) Speed-continuous re-timing. The path of the prediction is kept (polyline through origin + waypoints, straight continuation beyond the end);
the speed profile starts at the ego's current speed v0 and blends linearly into the plan's own segment speeds over tau seconds:
v(t) = v0 + (v_plan(t) - v0) min(t / tau, 1), integrated at 0.05 s, positions read at the 20 waypoint times. If lane FIX1 has committed its
serving function when the read is run, that function and its default are used; otherwise the form of
`experiments/alpasim/lib/col1_pai_driver.py::retime` (tau = 1.0 s), generalised only in the waypoint grid (0.25 s x 20 instead of 0.5 s x 8).
The result doc says which. v0 = the speed the adapter is fed (`jevdrive.waymo.past_kinematics(past)["v"]`), for every arm including shipped.

(b) Lead-aware longitudinal cap from the frozen model's lead outputs (stored in the same npz: `lead`, `lead_prob`). The law is FIX1's, verbatim.
Not invented here: if FIX1 has not committed a law by the time job 2 is finished, (b) is reported as pending and only (a) is read.
If it lands: (b) alone and (a) + (b) are read with the same lines.

Gates: G0 the unmodified arms reproduce 7.708 / 7.734 / 8.005 (RFS). G1 the re-timing operator with v0 := the plan's own first-segment speed
changes no waypoint by more than 0.01 m (operator identity).

Read: RFS = cluster-mean rater feedback score on the 479 rater frames; ADE@3s on the 1 437 frames with futures. Paired difference modified
minus unmodified of the same arm; 95% percentile bootstrap over sequences (B 4 000, `wod_launch_report.Ctx.ci`, as decision 217).
Primary comparisons (two, RFS): P2H10 (a) vs P2H10, SH30 (a) vs SH30; when (b) lands the same two arms for (a) + (b).
Lines: "helps" = RFS difference > 0 with the CI excluding 0; "hurts" = < 0 with the CI excluding 0; else "no measurable change".
ADE@3s is reported next to it, no line. shipped (a) is the control row, no label.
Secondary, no label: strata standstill (v0 < 0.5 m/s) / moving, lead-present (shipped lead head sigmoid(lead_prob[0]) > 0.5) / no-lead,
and speed bins 0.5-5 / 5-12 / >= 12 m/s; sensitivity tau in {0.5, 2.0} and v0 = `init_speed`; per-seed differences.
Prior, written before the read: decision 164 (raters prefer going further) and 169 (standstill gap) make a cost plausible: from standstill
the re-timed plan covers less distance, and a lead cap shortens plans; decision 162's creep on stopped frames makes a gain on SS frames
plausible. A negative RFS difference is a finding about the board, not a failure of the fix.

## Job 2: longitudinal audit, shipped plan vs adapted plan on identical frames

No registered lines except the definitions below (fixed now); this is measurement. Arms: shipped (`P0`: shipped policy weights, no adapter, same
vision tokens), P2H10-F s0 / s1, SH30-F s0 / s1; APY10m10-AB / AP2H10-AB only where their input standard is served by existing code
(navtest through the stored `@warp` bench plans; the AlpaSim nuPlan rollouts whose driver they were).
Domains: (N) navtest 12 146 tokens, pp_prep warp cache, torch parity path; (W) WOD val 1 437 frames with futures (479 rater), stored harness
predictions; (A-nu) logged decisions of the AlpaSim nuPlan rollouts COL1 extracted (collision cases + P2H10-F-s0 control scenes), replayed
open loop through the real driver class; (A-pai) the PAI rollouts COL1 extracted and replayed (served checkpoint `ft` and shipped `p0` on
the same tokens, as stored). In (A-*) the state at each decision is the rollout's (our driver's closed loop), the "logged future" is the
scene's recorded ego trajectory from the same sim time (a loose reference once the rollout has diverged; reported, flagged).
First check, reported before anything else if it fails: units and frames per domain (rear-axle frame, waypoint grid, the ego feature vector
fed per domain: vx / 10, ax / 3, pose scaling; fed speed vs the speed implied by the fed poses).

Definitions (all in the rear-axle frame at t0; arc length along the plan polyline from the origin):
- D1 first-segment ratio r0 = (arc length at 0.5 s / 0.5) / v0, frames with v0 >= 2 m/s; speed bins [0.5, 2), [2, 5), [5, 12), [12, 16), >= 16 m/s
  (and < 0.5 for the standstill reads). Reported as median and mean of (r0 - 1) per bin, domain and arm, paired adapted - shipped with a cluster
  bootstrap (log / sequence / scene).
- D2 arc-length ratio at 1 / 2 / 4 s vs the logged future: sum of plan arc / sum of logged arc per bin (ratio of sums), and the median per-frame
  ratio on frames whose logged arc at that horizon is >= 1 m.
- D3 lead response. Lead frame: shipped lead head p = sigmoid(lead_prob[0]) > 0.5. Gap d = lead hypothesis 0 at t = 0, x minus the
  camera-to-front-bumper distance; lead speed v_l = same hypothesis, channel 2; closing speed c = v0 - v_l. Plan acceleration
  a = (plan speed over 1.5-2.0 s - v0) / 1.75 s; the logged human's a by the same formula on the logged future. Closing-lead frame: lead frame
  with v0 >= 2 m/s, c >= 1 m/s and d / max(c, 0.1) <= 8 s. "Slows" = a <= -0.3 m/s^2. Reported: mean a of shipped / adapted / log per cell of
  time-to-contact d / c in {< 2, 2-4, 4-8 s}; the share of closing-lead frames where shipped slows and adapted does not, and the reverse (with
  both / neither), per domain and arm, CI by cluster bootstrap; the slope of a on a_need = c^2 / (2 max(d - 2, 0.5)) per arm.
  Sensitivity: p > 0.7, slows = a <= -0.5.
- D4 standstill: frames with v0 < 0.5 m/s; plan displacement at 2 s (and 4 s); split by stopped lead (lead frame with d < 15 m and v_l < 1 m/s)
  vs no lead (p <= 0.5); logged displacement alongside.
- D5 path of the difference, existing switches only, on the torch parity path (navtest; AlpaSim nuPlan replays where the replay is re-run) and
  from stored variant predictions on WOD: `nobias` (adapter off: `inputs_on=False`; what remains is the fine-tuned plan pathway reading
  the frozen vision tokens), `const` (bias replaced by its mean over the domain's frames: input-independent part, decision 162),
  `resid` (bias minus that mean), `posecv` (the 4-pose history replaced by a straight constant-speed history at the fed speed: no history
  information beyond speed), `gate` (adapter off below 0.5 m/s, decision 169). None of the arms has a memory channel (all are arm P2), so
  "memory-mask" does not apply; that is stated instead of run. Read: D1 and D2 (2 s) of each variant; the share of the adapted - shipped
  difference in r0 - 1 and in the 2 s arc ratio that remains with `nobias` (vision / plan-weight path) vs what `const` and `resid` carry.
- D6 swaps (construction of decisions 162 / 207: a trajectory's own arc length at each waypoint time imposed on the other's path):
  AS = adapted path + shipped speed profile, SA = shipped path + adapted speed profile. Scored on WOD by RFS (479 frames; `pp_wod_diag.retime`
  polyline form) and on navtest by `python -m jevdrive.bench score-poses --traffic non_reactive` (no-EC EPDMS, all 12 146 tokens;
  `pt_swap.Curve` with its arc extension). One number per domain: longitudinal share = (score(SA) - score(S)) / (score(A) - score(S)), path
  share = (score(AS) - score(S)) / (score(A) - score(S)), with the paired CIs of the numerators (navtest: cluster bootstrap over logs; WOD: over
  sequences). Gate: the unswapped keys reproduce the stored bench sub-scores per token (navtest) and the stored RFS (WOD). Secondary on navtest:
  the job-1 re-timing (a) of P2H10-F-s0 and SH30-F-s0 scored the same way (what speed-continuous serving costs on the nuPlan board, with v0 =
  the cache's `speed`). In the AlpaSim domains there is no offline board metric: D1-D4 only, no swap score.
- Ceilings for the candidate changes are read from D6 and D5 only: "adapter restricted to path, base speed profile kept" = AS; "speed residual
  gated by lead prob" = AS on lead frames and A elsewhere (and the reverse split), scored on both boards. No training.

Deliverable question: is the adapter's speed behaviour a nuPlan prior that overrides the base model's lead-aware longitudinal behaviour; in
which domains does that cost score; the cheapest training-free or training-side change that keeps the nuPlan-board gain without exporting the
prior. If the audit finds a frames / units / ego-feature-scaling fault instead, that is reported first.

Budget: <= 10 card-hours, ~6 h wall, <= 8 jobs at a time, lane cores <= 100; all GPU work through the pool; board scores only through
`jevdrive.bench` / the `wod_slot` (`mixed_domain.Wod`) path.
