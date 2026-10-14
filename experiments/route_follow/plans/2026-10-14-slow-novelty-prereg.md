# Pre-registration: SLOW-NOVELTY, does the residual slowing track state novelty (no training, no GPU), 2026-10-14

Context: [decision 285](../../../research/decisions/285.md) (the plan brakes where the log did not; a ratchet through the fed speed history),
[vhist](../results/vhist/README.md) / decision 286 (holding the fed speed history at serving removes 0.86 of the plan shortness on held 460).
The user's hypothesis, to test and not to confirm: **the part of the slowing the ratchet does not explain, and what starts the ratchet, is
the model slowing when it finds itself in states it has not experienced** (off the log path, odd heading, inside junctions, far from the
training distribution), copying the human's caution without the human's release condition. Written and committed before any statistic below
is computed. Amendments are appended, never edited in place.

**Question.** Once the ratchet is removed (the `vhold` read) or controlled for (the fed-speed covariates in the plain read), does the plan's
shortness at a decision grow with the novelty of the state at that decision?

## Data (stored reads, box, read-only; held 460 scenes x 18 decisions, RSG-L18-s0, one checkpoint, one seed)

| label | read | role |
|:--|:--|:--|
| RSG | `runs/route_mem/read/RSG-L18-s0/c*/run` | ratchet present (analyses 2, 3, 4) |
| RSG+hold | `runs/route_mem/read/RSG-L18-s0-vhold/c*/run` | ratchet removed where held (analyses 1, 4; **primary**) |
| RSG+holdlead | `runs/route_mem/read/RSG-L18-s0-vholdl/c*/run` | replication of analysis 1 only (secondary) |

Known limit of the hold, fixed before the read: `JEV_VHIST=hold` releases the hold (feeds the measured row) whenever the route command is
left / right or the previous plan is below 2 m over 2 s. The release is on turns, i.e. on junction decisions, so in the vhold read the ratchet
is still live exactly where novelty (c) lives. The primary model therefore keeps the fed-speed covariates (e) in its base block in vhold too;
the brief's model without (e) is reported as a secondary row.

## Unit and outcome

Unit: one decision (scene x decision index k) of the last driver session of a scene (`ao_e1.drive_records`), kept by rf_deficit's rule
(measured speed >= 1 m/s and k <= k_z, the first zero-score event step of `long_read.one`), and with a valid projection on the log path.

- **Y1 (primary): plan 2 s deficit** = rf_deficit's ruler, unchanged: `rf_follow.plan_arc2(poses) - 2 x v_meas` (m; v_meas from the logged,
  measured `ego` row; negative = plan shorter than the car's own speed). To first order Y1 = 2 x the plan's mean acceleration over 2 s.
- **Y2 (secondary): speed ratio** = v_meas / v_log(s), v_log(s) = the log's speed where the log was at the ego's arc position s (ego anchor
  projected on the log rig path; log speed from the rig trajectory's station-vs-time, central difference over +-0.25 s). Decisions with
  v_log(s) >= 1 m/s only.

## Covariates (per decision)

Novelty block N (the hypothesis):
- (a) `lat`: |distance| of the ego anchor to the log's rig path (m), projection local to the previous decision's station (rf_follow's -30 /
  +60 m window).
- (b) `head`: |ego anchor yaw - log path tangent heading at the projection| (deg, wrapped).
- (c) `junc`: rf_follow's junction flag (route turn stretch, entry - 20 m .. exit + 10 m, turns found by `rf_follow.find_turn` within the log's
  own travel to decision 17); scenes without a route rebuild drop out.
- (f) feature-space distance to training states: **not measurable here** (checked before registering): the drive records hold the ego row,
  the plan, `rmem` and `vhist`, no tokens; `images.jsonl` keeps only byte counts and no frames are stored, so the closed-loop views would have
  to be re-rendered and encoded on a GPU. Not done (brief: no GPU); its cost is stated in the result.

Base block B (what else is known to move Y1):
- (h) `logdef`: the log's own 2 s deficit at the same arc position (log arc over [t_log(s), t_log(s) + 2 s] - 2 x v_log(s)): the human's
  legitimate slowing at that place (junctions are where humans brake); and `vlog` = v_log(s).
- (d) `tstop`: seconds since the last decision with v_meas < 0.5 m/s (0.5 s per decision); `stopped` = 1 if any such decision before k (when 0,
  `tstop` = time since the rollout start).
- (e) ratchet (fed row = `vhist.ego_fed` when logged, else `ego`): `vfed` = fed speed (m/s), `dvfed` = fed speed - mean fed speed over the
  1.5 s history (fed history poses' arc / 1.5 s; negative = the fed history shows deceleration); in hold reads also `held` = v_fed > v_meas.
- (g) `k`: decision index.

Preprocessing (fixed): continuous covariates and outcomes winsorised at the label's 1st / 99th percentile; every covariate and the outcome
z-scored within the label (standardised coefficients); `junc`, `stopped`, `held` enter as 0/1 then z-scored. For Y2, `vfed` and `dvfed` are
left out (v_fed / v_log is the outcome itself in RSG); `held` stays.

## Analyses

1. **Primary (vhold, Y1):** OLS Y1 ~ B (h, vlog, d, stopped, e, g) + N (a, b, c). Statistic: delta R^2 = R^2(B + N) - R^2(B), and the three
   standardised N coefficients. CIs: log-clustered bootstrap (resample held logs = scene id without its last "-<hash>", B = 2000 draws,
   `default_rng(0)`), 95 % percentile. Secondary rows: the same without (e) (the brief's spec), the same with scene fixed effects (all
   variables demeaned within the scene: within-rollout novelty), Y2, and RSG+holdlead.
2. **RSG with the ratchet in:** the same model on the RSG read (e included). Does N still carry once the ratchet variables are in.
3. **Onset (RSG):** excess deficit x = Y1 - logdef (the plan's shortness beyond the human's own at that place). Threshold thr = the 10th
   percentile of x over all kept RSG decisions (computed and printed by the script; a rule, not a number chosen after seeing anything else).
   Onset of a rollout = its first kept decision with x <= thr; the rollout enters the onset contrasts only if at least 2 kept
   decisions precede that onset. Never-slow rollouts: no kept
   decision with x <= thr. For each of a, b, c and for `dvfed`, `vfed`: (i) onset value - mean over the same rollout's preceding kept
   decisions (paired, per rollout), (ii) onset value - the mean over never-slow rollouts' kept decisions at the same k (per onset rollout,
   then averaged). Means with log-clustered bootstrap CIs.
4. **Binned plot:** Y1 against each of a, b, c (and Y2 against a, b), deciles of the pooled covariate (junc: 0 / 1), mean with log-clustered CI,
   RSG and RSG+hold overlaid. `figs/slow_novelty_bins.png`, onset `figs/slow_novelty_onset.png`.

## Decision rule (registered)

- **"Novelty carries"**: in the primary model (vhold, Y1, B with e), delta R^2 >= 0.05 **and** at least one of the a, b, c coefficients has a
  95 % CI entirely below 0 (more novel -> shorter plan).
- **"Pure input loop"** otherwise. Within it, "novelty is a minor term" when a coefficient is clear of 0 in the slower direction but delta
  R^2 < 0.05; "refuted outright" when delta R^2 < 0.02 and no a, b, c coefficient is clear of 0 in the slower direction.
- Evidence **against** the hypothesis also: a novelty coefficient clear of 0 in the opposite direction (more novel -> longer plan).
- Onset ("novelty starts the ratchet"): supported if for at least one of a, b, c both (i) and (ii) are clear of 0 in the more-novel
  direction; not supported if neither is. Read beside the same contrasts for `dvfed` (a ratchet-started onset shows the fed history already
  decelerating before the onset decision).
- RSG (analysis 2), Y2, within-scene and holdlead rows are reported and interpreted, but do not change the verdict.

Threats named in advance: novelty may be a consequence of slowing (a slow car is off the log's timing, and turns at low speed may cut the
corner), so the regression is association, not cause; the onset analysis is the only time-ordered read. Junctions are where the log brakes
(h controls for it) and where the hold is released (e controls for it). One checkpoint, one seed: evidence grade low.

Cost (est.): one CPU job, ~10-30 min at 12 workers (route rebuild per rollout as in rf_follow), outside the GPU pool, inside the box's CPU
margin (`cl top`).
