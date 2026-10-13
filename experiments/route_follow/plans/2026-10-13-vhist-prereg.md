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
