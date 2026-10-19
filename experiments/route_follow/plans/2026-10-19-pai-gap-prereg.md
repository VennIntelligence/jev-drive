# PAI-GAP prereg (2026-10-19, written before any new number exists)

Diagnosis, no training. The PAI counterpart of decision 372's native gap: where NTV0-L18-s0 + ACC + GAP (decision 366: PAI 281, g10,
0.744, 59 zeros) loses points, what each loss is worth, and whether each top loss is the model's plan, the serve side, or unattainable.
Code: `experiments/route_follow/scripts/pai_gap.sh` (pool chain) and `pai_gap.py` (box read-outs). Results:
`experiments/route_follow/results/pai_gap.md` + `pai_gap/` (json, tables). Box root: `$DATA_DIR/runs/route_follow/pai_gap/`.

## Inputs

- **A0** = the stored read `runs/defect_fix/pai_accgap/NTV0-L18-s0` (code 2e2d0e46, VCONT 1.0 + LEAD + ACC + GAP 0.7,2.7, g10, CONC 4,
  ALPASIM_FAST=1). Its rf_tax extract `runs/route_follow/pai_input_loop/tax/A0.pkl` (controller trace, log path, logged traffic at 0.5 s)
  and its `drive.jsonl`. 280 scenes (1059fc66 has a driver RPC error, excluded as in d376).
- **N0** = the same model without ACC + GAP (VCONT + LEAD), stored by d376 (`pai_input_loop/N0`), the serve counterfactual.
- **Other stored PAI 281 reads with the same serve and g10**: NTV0-L18-s1, NTV0-L18-s0-hl, JTN-L18-s0, JTN-L18-s1, NTVX-L18-s0
  (`runs/defect_fix/pai_accgap/*`) and RSG-L18-s0 (`runs/route_follow/acc_gap/read/s2`). Scores only, for best-of-reads.
- **H (new, the only new rollouts)** = a log-replay read of the same 281 scenes (A0's 8 chunk lists): `onpol_pai/lib/onp_log_driver.py`
  (the ego is steered along the logged path by the MPC; the model runs, is fed and is recorded at the human's states, with the A0 serve:
  VCONT 1.0 + LEAD + ACC + GAP, so `poses_model`, `poses`, `fix.lead`, `fix.acc` are logged at on-log states; `log_poses` is the human's
  future in the same frame), wizard `runtime.simulation_config.send_recording_ground_truth=true` + g10, NTV0-L18-s0, priority 12.
  Staged: chunk c000, a CPU gate (every drive record has `ret == "log"` after the force-GT window, 0 RPC errors, >= 30 rollouts), then
  c001..c007. H's scores are the human trajectory as this simulator and scorer score it (with the MPC tracking it).

## Item 1: per-scene decomposition of A0

Loss of a scene = 1 - score (1 point = one scene; mean = 1 - points / 280). Class, first match:
1. zero by type (pai281's `zclass`): AF-front, AF-side, AF-rear, offroad, corridor, other; corridor and offroad zeros sub-typed by
   rf_tax `classify` (cor_route_end, cor_straight_at_turn, cor_turn_under / over, cor_wrong_exit, cor_lane_change, cor_drift;
   off_junc_* / off_curve_* / off_straight / off_at_start); AF zeros by the object (stopped / moving ahead / crossing / side).
2. partial (0 < score < 0.98) by pai281's `ptype`: rear-hit, hit, no-launch, creep, slow; with rf_tax's primary label.
3. scored (>= 0.98): residual.

Strata of the scene (first match, from the log and the logged traffic, not from the policy): static (log < 1 m); launch (log speed at the
handoff, 1.7 s, < 1 m/s); stop/lead (moving at the handoff and the human stops, v < 0.5 m/s for >= 1 s, later; or a logged vehicle in
the human's own path within 30 m ahead on >= 30 % of the human's moving 0.5 s steps); turn (log heading change > 30 deg); straight.

Per class and stratum: points lost; points recovered by **best-of-reads** (per scene the best score among the 7 stored reads, A0 included:
what some existing model / seed already scores) and points the **human** also loses (H's loss on that scene, min(loss A0, loss H)).
"Arithmetic to target", as d372: ranked by points-to-gain = points lost - human-lost points, cumulative mean; target named = +0.05
(the RSG -> NTV0 step; the top-3 private-board gap is ~100 PCS, about one such step).

## Item 2: are the JEV_LEAD cuts real leads or phantoms (d376 open item)

Every A0 inference decision with the ego >= 1 m/s and `fix.lead.cut` > 0 (metres removed from the first 2 s; d376's definition), and
separately cut >= 0.5 m. Ego pose at t0 from the controller trace; logged traffic interpolated at t0 (rf_tax `objs_at`). The ego's
corridor = the SERVED plan (rig frame -> world), extended 40 m straight along its end heading; an object is **in path** when its centre is
within hw + W / 2 + 0.3 m of it laterally and its rear edge 0 .. D ahead of the front bumper (rf_tax `in_path` rule), D = d + max(5 m,
0.3 d), d = the reported lead distance `fix.lead.d`.

- **no lead reported**: `fix.lead.d` is None (filtered probability <= 0.5): the cut is the MPC's own transient (fake lead), not a lead.
- **real**: a logged object is in path within D.
- **adjacent / off-path**: none in path, but an object whose rear edge is 0 .. D ahead with lateral offset <= lim + 3.7 m (one lane).
- **nothing**: neither.
Phantom = adjacent + nothing + no lead reported; reported per decision, per cut metre, and per scene.
Gate (before the classification is read): logged traffic against the rollout.asl actor poses on the A0 rollouts that kept an asl
(rf_tax extract `--asl-actors`): the same ids within 0.05 m on >= 99 % of the pairs (zeros_mech G-act). Fail -> classify only those
rollouts with the asl actors and say so.

**Worth (upper bound from the decomposition, no serve change):** the longitudinal loss (rear-hit, slow, creep, no-launch partials and AF
zeros are excluded from the bound) of scenes whose phantom cuts carry >= 50 % of the scene's cut metres before its first scored event
(rear contact / truncation); a looser bound = all longitudinal loss x the pooled phantom share of cut metres. Removing a cut may also add
AF-front zeros; that is not counted, so both numbers are ceilings.

## Item 3: attribution of the top 2-3 loss classes (by points-to-gain)

Per losing scene, the A0 event time t_ev (rf_tax: the first zero event; for partials the rear contact or, for slow, the whole rollout to
its truncation); the human's time t_h at A0's ego station at t_ev; H's decisions in [t_h - 4 s, t_h] are "the same place, on the human's
state" (open loop against the human, as d372 k = 0 / d376 2c).

- **(iii) unattainable**: H loses >= 0.5 point on the scene (the human trajectory scores badly here; or a sim / scorer artefact: H's own
  event type is reported).
- **(ii) serve side**: lateral zeros: N0 (same model, no ACC + GAP) scores >= 0.98 on the scene AND A0's served speed in the 3 s before
  t_ev exceeds both the model plan's own speed (`fix.acc.v_model`) and the human's speed at the same station by >= 10 %. Longitudinal
  partials: in the 3 s before the rear contact (slow: over the valid rollout), the term that set the served speed (`fix.acc.won`) is `lead`
  (a cut) or `curve` on >= 50 % of the decisions; for `lead`, item 2's class says whether it was a phantom.
- **(i) the model's plan**: not (iii) / (ii), and on the human's state the model plan already departs from the human: lateral (corridor /
  offroad / AF-side): median |lateral error of `poses_model` against `log_poses` at 4 s| >= 1.0 m over the H window ("plan wrong on the
  human's state"); < 1.0 m -> "(i-cl) the plan is right on the human's state and goes wrong on its own" (closed-loop compounding; still the
  model, but a remedy is on its own states, not its labels). Longitudinal: `fix.acc.won` is `set` / `ramp` / `model` and on H the model
  plan's 2 s arc against the human's 2 s arc (`log_poses`) has median < 0.9 (the model under-plans the human's acceleration; ACC's set
  speed v_set = max(handoff speed, the plan's far speed) follows the model's plan, so a short plan caps the speed).
- otherwise **mixed / unexplained**, counted.

For each top class: the share of its points under (i) / (i-cl) / (ii) / (iii) / mixed, and one model-side remedy consistent with the closed
lines (decisions 352 / 353 / 365: no two-sided speed target; waiting and launching share one low-speed axis).

## Not claimed

Anything about the private board beyond direction; one seed (NTV0-L18-s0; s1 is used only inside best-of-reads); H is one read (the MPC
tracking the log is not the log itself: small deviations are reported, H's own zeros are typed); item 2's worth is a ceiling, not a serve
change.

## Amendment 1 (13:08, after the A0-only dry run of items 1-2, before H has any rollout)

The dry run (H absent) shows 23 of the 44 corridor zeros are rf_tax `cor_route_end`: the human ends the 20 s still and the ego passes
its rest point (decision 357's class C, a 4 m disc). That zero is longitudinal in mechanism, so the lateral test of item 3 does not fit it.
For `cor_route_end` zeros only: the H window is the human's last 4 s before it reaches the station of A0's event (= its approach to the
rest point); **(i)** = on H the model plan's 4 s arc exceeds the log's 4 s arc by >= 4 m (median over the window: the plan on the
human's own state does not stop where the human stops); otherwise **(i-cl)**. (ii) / (iii) unchanged. Added descriptions (not lines):
for `cor_route_end`, a logged vehicle in the human's path within 15 m at the rest point (queue stop) or not, and the map wait lines
within 12 m (rf_tax `waitlines`); for real-lead cuts, the served 1.5 s speed against the human's speed at the ego's station.
