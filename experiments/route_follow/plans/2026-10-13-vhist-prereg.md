# Pre-registration: VHIST, the slowness causal split at serving time (no training), 2026-10-13

Context: [steer-long.md](../../../research/next-round/steer-long.md) 4.4 / 5 (2) (expert B: the plan follows its own speed history; feeding the
log's speed history offline restores 83 % of the shortness), [decision 285](../../../research/decisions/285.md),
[allrows](../results/allrows/README.md) (training-side fix `--ego-drop-speed 0.5`, RSGallV-L18: plan 2 s deficit median -0.433 -> -0.247 on public
1491, about 75 % of the rollout-mean shortness, at public zeros +8.5 and GT +0.056 m), [zero-taxonomy](../../../research/zero-taxonomy.md)
(slow = 13.5 of 36.8 points on PAI, 4.2 of 7.4 on held). Written, committed and pushed before any read below is submitted. Amendments are
appended, never edited in place.

**Question.** Is the slowness a closed-loop feedback through the policy's own speed-history input, such that holding that input at a reference
value at serving time (no training) removes the same share of the plan shortness as the training-side drop, at the same or lower cost in zeros
and GT distance?

## The switch: `JEV_VHIST` (experiments/alpasim/lib/vhist.py; nuPlan sh30 and PAI drivers; default off)

Only the ego row fed to the adapter changes. The synthesised frames, the served trajectory, the controller and the logged `ego` (the measured
row, so rf_deficit's "fed speed" stays the car's own speed) are untouched; the fed row is logged as `vhist.ego_fed`. Unset / `off` = no
operation is added (the plan call is the same call).

Speed-history entries = rm_train's EGO_SPEED, ego_features columns 4-9, 11, 12, 14, 15, 17, 18 (vx, vy, ax, ay, the x / y of the 4 history
poses; `present`, command and history yaws stay).

- `zero`: those 12 entries set to 0 at every decision (the serving counterpart of `--ego-drop-speed`; meant for RSGallV, which saw that row in
  training).
- `hold`: the entries a car would have that drove the same path at a reference speed v_ref. Only quantities the driver has (no log, no map
  speed limit: neither the nuPlan nor the PAI driver is given one):
  - v_ref = the running maximum of the measured ego speed over the session's decisions so far (the first decision's speed included; a cruise
    set-speed: the fastest the car itself has gone in this rollout). Updated at every decision, released or not.
  - release (feed the measured row unchanged) when (a) the route command is left or right (a turn ahead: the log slows there for a reason) or
    (b) the previous decision's model plan has a 2 s arc (rear axle through the 0.5 .. 2.0 s poses) below 2.0 m (the model itself asks to
    stop or crawl). B's third release (an openpilot-distance lead from JEV_LEAD) is not used: it needs the lead outputs and would make the arm
    depend on a second switch.
  - otherwise v_fed = max(v_meas, v_ref); if v_fed = v_meas the row is unchanged. Else (vx, vy) scaled by v_fed / v_meas (v_meas < 0.5 m/s:
    (v_fed, 0)), ax = 0 (steady speed), ay unchanged, and the 3 past poses re-spaced along their own polyline (newest to oldest, extended
    straight beyond the oldest) to arc lengths v_fed x (0.5, 1.0, 1.5) s, yaw interpolated the same way (parity_adapter.cv_history's rule
    at v_fed; the counterfactual's arm B with v_ref in place of the log speed).
  Scenes that start from rest have v_ref = the speed reached so far, so the hold acts once the car has moved.

## Reads (main box, GPU pool, held 460 at 18 decisions, the 6 held_n6 chunks on cards 1-5; `rm_chain.sh submit-read` with env + suffix)

Held 460 is now training-side (its rollouts are in the all-1491 pool): a mechanism read, not evidence for the exam.

| label | checkpoint | switch | status |
|:--|:--|:--|:--|
| RSG | RSG-L18-s0 | off | stored (`runs/route_mem/read/RSG-L18-s0`) |
| RSG+hold | RSG-L18-s0 | `JEV_VHIST=hold` | new, `read/RSG-L18-s0-vhold` |
| allV | RSGallV-L18-s1 | off | new, `read/RSGallV-L18-s1` |
| allV+zero | RSGallV-L18-s1 | `JEV_VHIST=zero` | new, `read/RSGallV-L18-s1-vzero` |

Smoke before the reads: 5 held scenes (first 5 of `held_n6/c0.txt`, so the stored read has them), RSG-L18-s0: switch off -> every logged plan
equal to the stored read's at the same (scene, k) (bit-identical; if compiled-graph noise breaks equality the off-path code identity is shown
instead and the size of the difference reported), `hold` and `zero` run without driver errors and their logged `ego_fed` follows the rules above.

## Measures (one script, `rf_vhist.py`; rf_deficit unchanged)

- Plan 2 s deficit: rf_deficit (served plan arc to 2 s minus measured speed x 2 s, decisions >= 1 m/s up to the first zero event): pooled
  decision median and mean of rollout means; share removed = 1 - mean(arm) / mean(reference) on the rollout means.
- Zeros: scenes scored 0, the 8 no-rollout scenes counted (zero-taxonomy's convention: stored RSG = 34); collision / offroad / corridor split.
- GT distance: mean over scenes with a rollout of the aggregate dist_to_gt_trajectory.
- Mean score; rear-ended: rollouts with collision_rear > 0 (all and non-zero-score); mean progress_score and progress_rel.
- Paired differences over common scenes with log-clustered 95 % CIs (jevdrive.stats.paired).

## Lines (registered)

- **M (mechanism, RSG+hold vs RSG)**: deficit median >= -0.25 m (from -0.433); zeros <= RSG + 5 (<= 39); GT <= RSG + 0.05 m. All three = pass.
- **S (same share)**: RSG+hold's share removed >= 0.60 (training-side drop: about 0.75 on public 1491) = "the serving hold reproduces the
  training-side effect"; 0.30-0.60 = partial; < 0.30 = the shortness is not (mainly) carried by the fed speed history.
- **Z (matched, allV+zero vs allV)**: reported with CIs; zero "helps" if the deficit median rises by >= 0.10 m with zeros <= allV + 5 and GT <=
  allV + 0.05.
- **Cost comparison**: hold's zeros / GT cost against RSG is compared with allV's against RSG (the training-side cost on the same scenes).

PAI recommendation (rule fixed now, applied by the coordinator on Tokyo): M passes -> RSG-L18-s0 + `JEV_VHIST=hold`; else Z helps -> RSGallV-L18-s1
+ `JEV_VHIST=zero`; else no VHIST run on PAI. On PAI the read to watch is the rear-ended count and progress score (zero-taxonomy: 70 non-zero
rollouts truncated by rear contact).

Cost (est.): 18 chunk jobs x ~8 min, about 2.5 card-hours; smoke ~10 min.

## Amendment 1 (2026-10-13, after the first read, before any `holdlead` read): hold with a lead and a brake release

The first read (results/vhist/README.md): `hold` removed 0.86 of the shortness but zeros 34 -> 70, all 39 collision zeros front collisions (the
car runs into its lead), GT +0.60 m; M failed, Z did not help, no PAI run. Coordinator's request: the lead-release arm.

New mode `JEV_VHIST=holdlead` = `hold` (same v_ref, same feed, releases (a) turn command and (b) previous plan 2 s arc < 2 m) plus two releases,
checked in this order after (a), (b); all read from the PREVIOUS decision's outputs of the same model (the current pass depends on the fed row,
so its own outputs cannot gate it):
- (c) lead: the previous decision's lead / lead_prob outputs inside openpilot's following distance, evaluated at that decision's measured speed
  with serve_fix.LeadPlanner.inside (JEV_BASE=2's rule, lead_long.py from openpilot source): radard's asymmetric lead-probability filter
  (rc 0.2 s) > 0.5 and dRel + v_lead^2 / (2 x 2.5) < v^2 / (2 x 2.5) + 1.45 v + 6, with dRel = lead x - camera-to-front-bumper and
  v_lead = v + lead v - the model's own speed (plan velocity at t = 0). This is B's gap ("JEV_LEAD's radard-filtered lead inside openpilot's
  following distance").
- (d) brake: the previous model plan decelerates: segment speeds s1 (0 .. 0.5 s) and s4 (1.5 .. 2.0 s) of the plan, (s1 - s4) / 1.5 s >= 1.0 m/s^2.
  The threshold is a real brake, not the small braking that starts the ratchet (releasing on every decrease would switch the hold off).
Released = the measured row is fed (v_ref keeps updating). The lead outputs are switched on in the core (`lead_out`; output slicing only, plans
unchanged); the served trajectory is unchanged (no JEV_LEAD).

Reads (held 460 / 18, 6 chunks, cards 1-5): `RSG-L18-s0` + holdlead (`read/RSG-L18-s0-vholdl`), and `RSGallV-L18-s1` + holdlead
(`read/RSGallV-L18-s1-vholdl`) queued alongside (the pool runs it when cards are free). Smoke first: 5 held scenes, holdlead, 0 rule
mismatches (`rf_vhist.py check`, the lead flag taken from the log) and switch off still bit-identical.

Lines: **M as registered** for RSG + holdlead vs RSG (deficit median >= -0.25, zeros <= 39, GT <= 1.135 + 0.05), plus S's share reported.
RSGallV + holdlead is compared with allV plain (same M thresholds relative to allV: zeros <= allV + 5, GT <= allV + 0.05) and with RSG.
PAI rule: RSG + holdlead passes M -> it is the PAI-exam switch set (`SH30_TAG=RSG-L18-s0`, `JEV_VHIST=holdlead`); else if RSGallV + holdlead
passes against RSG's thresholds (zeros <= 39, GT <= 1.185, deficit median >= -0.25) -> that; else no VHIST on PAI.

## Amendment 2 (2026-10-14, written and pushed before any run below): RSG + holdlead on the PAI 281, user override

**Override.** The PAI rule above ("RSG + holdlead fails M -> no VHIST run on PAI") is overridden by the user on 2026-10-14: read
RSG-L18-s0 + `JEV_VHIST=holdlead` on the full PAI stage-1 281 set. Held 460 (Amendment 1): score +0.028 [+0.019, +0.040], zeros 32 vs 34
(front-collision zeros 2 vs 4), rear-ended 38 -> 18, GT +0.097 m (line M failed on GT alone).

**Known before this run (disclosed).** A coordinator exception on 2026-10-13 (`/data/runs/route_night/r281/exam_rsg_holdlead`, the same
checkpoint, switch and serving) drove 99 of these 281 scenes (exam206 rows that are in test281): +0.083 [+0.030, +0.135] against the
stored reference, zeros 26 vs 23, rear-ended 27 vs 37, GT +0.22 m (rf_vhist_pai.py on the stored roots). On the 38 of them in
hard_exam.txt: +0.004 [-0.085, +0.082], zeros 14 vs 11. Those 99 scenes are driven again here (PAI reruns are scene-deterministic,
docs/tokyo-box.md); the read below counts all 281.

**Run.** Tokyo card 1 only (card 0 is reserved for the user). Same stack and image as the reference `/data/runs/route_night/r281/rsg`:
`pai_eval.sh` (-> `pai_run.sh`, driver image `jev-alpasim:p2h10-f-s0-6f3d05d1`, `alpasim-base:0.89.0`, official `+e2e_challenge=dev`
stack), checkpoint RSG-L18-s0, `controller.gains.rel_front_steering_angle_weight=20`, CONC 4, the reference's own 8 chunk lists (copied
from `r281/rsg/chunks/`), `SERVE="-e JEV_VCONT=1.0 -e JEV_LEAD=1 -e JEV_VHIST=holdlead"`, `RUNPFX=hl-`, code = a clean Tokyo worktree at
the commit of this amendment. Output `/data/runs/route_night/r281/rsg_holdlead`. Full rollouts, one seed.

**Estimate.** The reference's chunks took 31-34 min per 36 scenes on one 3090 (c000 1859 s); the v8 spin read ran 84 scenes in 71 min on
card 1. 281 scenes = 8 chunks -> **about 4.2 h** (4.0-4.5 h) on card 1, plus a 10 min smoke. Under the 1 h staged-launch rule this is
smoke (2 scenes x 2 arms) -> one chunk -> the rest, the chunks running back to back in one tmux window; no profile (< 3 h per stage
after the first chunk is measured: if chunk c000 runs over 45 min the ETA is restated in the results README).

**Smoke (before the full read, same worktree, card 1).** The first 2 scenes of the reference's c000: (1) switch off: every logged plan
bit-identical to the stored reference at the same (scene, k) (`rf_vhist.py same`; PAI logs read since this amendment's commit); if not
bit-identical the full read waits and the difference is reported. (2) holdlead: `rf_vhist.py check` 0 rule mismatches (release, v_fed,
previous-plan brake / arc, the fed row; the logged rows are rounded to 5 decimals, so a held row's tolerance scales with v_fed / v_meas and
rows differing only in the re-spaced history poses' lateral y are counted apart; on the stored 99-scene exam read: 19 653 records,
0 mismatches, 2 near-ties, 7 lateral-y-only rows, largest 0.074 m). The same check runs on the full read.

**Measures** (`scripts/rf_vhist_pai.py`, numpy, paired per scene against the stored reference over the scenes both drove; a rollout without
metrics is rerun once, alone): mean score difference with a scene-bootstrap 95 % CI (10 000 draws, seed 0); zeros by kind (collision =
at-fault, offroad, corridor, other; rf_pai281's order) and front-collision zeros (score 0 with collision_front > 0) separately, since the
plain hold produced 39 front-collision zeros on nuPlan held; rear-ended (collision_rear > 0, all / scored > 0); progress score
(score_metrics.progress_score) with CI; GT distance with CI; spins (sf_detect.py); plan 2 s deficit, rf_deficit's quantity on PAI logs as
rf_pai281.py computes it (served plan arc through its 0.5 .. 2.0 s poses minus the measured speed x 2 s, decisions >= 1 m/s, whole rollout:
PAI has no 18-decision window), pooled decision median and mean of rollout means, paired difference with CI. Turn completion:
`rf_pai_follow.py` extract (Tokyo container, both roots) and report (its registered `completed` rule on the scenes where the route turns,
plus its @open / @bend variants).

**Lines.** **Pass** = the mean score difference is > 0 with its 95 % CI clear of 0 **and** front-collision zeros (arm - reference) <= +2.
Otherwise **fail**. Everything above is reported either way. The PAI exam subset (`experiments/alpasim/results/pai_exam/hard_exam.txt`,
72 never-trained scenes, all in the 281) is reported separately with the same measures; it is not gated. No research/decisions entry.
