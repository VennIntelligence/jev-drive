# rc-*-near: near, low-speed CARLA turn-entry poses (decision 129 follow-up), pilot + 9-turn staged read

Pre-registration: [plans/2026-10-05-route-ft-prereg.md](../plans/2026-10-05-route-ft-prereg.md), sections "2026-10-06 near", "更正" (decision-130 action
target) and "范围变更" (scope cut to pilot + 9 turns, verdict rule), all written before the reads. Code: `scripts/near_plan.py`, `near_pack.py`,
`near_eval.py`, `near_chain.sh`, `near_lane.py`, `near_pilot_cl.sh`, `near_report.py`; `scripts/rft.py` arms `rc-bear-fix`, `rc-bear-near`;
test `tests/test_rft_act_target.py`.

**Verdict: include near / low-speed targets in the merged training: NO (no closed-loop evidence for it at pilot scale).** By the pre-registered rule:
on the 9 turns both 400-step pilots take 0 / 9 (rc-bear-near - rc-bear-fix = 0, rule "yes" needs >= +2), and neither turns in earlier (both are
mostly "not chosen"). Open loop the near rows are learnable and do not cost the real-row gain, so the material is not harmful; it is just not shown to
help. The deliverables for the merged run are the corrected action target (which every merged arm needs regardless) and the packed near set, which
the merged run can add later as an ablation if the forced-turn test points to turn-entry states.

## What was built

- **Corrected action target (decision 130), all rows.** `rft.act_target` / `act_target_path`: action[0] = -kappa_inst(t + 0.275 s) * max(1, v_model)^2,
  gain 1, right +, |kappa| <= 0.2, v_model = the original's plan speed at t0 (`rft.vmodel`). kappa_inst = cubic fit of the timed target (real rows) or
  the dense path's curvature at arc v0 * 0.275 s (CARLA rows). Replaces -0.45 x 1 s pure pursuit x max(1, v0)^2, which every rc-* arm of decisions
  128-129 carried. Real train rows: new / old target slope nav 1.24, wod 1.12 (corr 0.94 / 0.95). DAgger imports the same function.
- **Near set** `$DATA_DIR/runs/op_route_ft/carla_near/packed` (`rft.CARLA_ROOTS["near"]`, bank `$R/bank/near`): 4000 poses, 7941 (pose, exit) rows,
  914 junctions (none of the 172 B2D junctions), splits `b2d/route-carla-near-train@v1` (834 junctions) / `-dev@v1` (80), same sha1 dev rule as
  `route-carla-*` and disjoint from their train / dev. Approach poses 2681 (rear axle 0-10 m before the connector start, every exit legal from the
  ego lane, 1018 three-exit poses), in-turn poses 1319 (0.5-6 m along a taken turning connector); speed rolling 80% (v0 0.3-3 m/s, constant
  acceleration -1..+1 m/s^2 over the 1.8 s history), halted 10%, stopped 10%. Each row carries the 1 m dense lane-centre path (`dense`, `dmask`) the
  targets use. Rendered with `carla_pairs_render_ol.py` unchanged (open-loop-aligned rig, 6 servers on one card, 0 restarts, 56 unsettled = 1.4%,
  frames kept); sheet of 5 poses checked (horizon rows, exits fan into the branches, no body).
- **Arms** (400-step pilots, seed 0, rc-bear recipe): `rc-bear-fix` = rc-bear + corrected target; `rc-bear-near` = rc-bear-fix + 6 of the 16 CARLA P
  rows from the near set (action supervised from 0.3 m/s; plan target speed ramps from v0 at 1.5 m/s^2 to 4 m/s).

## Pilot (400 steps, open loop)

| readout | shipped | pilot-bear (old target) | pilot-bear-fix | pilot-bear-near |
|---|--:|--:|--:|--:|
| loss imit, step 100 -> 400 | | | 2.64 -> 1.79 | 2.60 -> 2.07 |
| loss act, step 100 -> 400 | | | 0.024 -> 0.022 | 0.025 -> 0.013 |
| real-row action gain vs the corrected target: nav approach (n 56) | 1.04 | 0.79 | 0.97 | **0.99** |
| wod approach (n 60) | 0.75 | 0.70 | 0.70 | **0.73** |
| wod in turn (n 7) | 0.90 | 0.73 | 0.77 | 0.74 |
| nav / wod straight | 0.97 / 0.73 | 0.63 / 0.58 | 0.75 / 0.58 | 0.78 / 0.60 |
| near dev, turn rows, gain k / k_target: approach d 0-3 m (n 121) | 0.06 | 0.20 | 0.07 | **0.19** |
| approach d 3-6 m (n 69) | 0.13 | 0.41 | 0.09 | 0.30 |
| in turn 0.5-6 m (n 93) | 0.54 | 0.88 | 0.68 | **0.75** |
| in turn, sign = command side | 0.83 | 0.92 | 0.90 | **0.94** |
| left minus right command, same approach pose (1/m) | 0 | 0.0084 | 0.0023 | **0.0098** |
| near dev approach rows, plan exit correct | 0.25 | 0.59 | 0.60 | **0.94** |
| CARLA ol dev exits correct per row (line 0.8; rc-bear-s0 0.583) | 0.156 | 0.445* | 0.447 | 0.472 |
| no-command drift CARLA / nav jct / wod jct, median m (line 0.15 pilot) | | | 0.090 / 0.034 / 0.035 | 0.097 / 0.039 / 0.042 |
| negatives CARLA N1 / navtrain screened / WOD, mean m (line 0.3) | | | 0.62 / 0.15 / 0.23 | 0.73 / 0.24 / 0.34 |

Gain = least-squares slope through 0 of the model's curvature on the target's (near rows: k = -action / max(1, v_model)^2); `pre_diag.py --out
near_pilot_diag.json`, `near_eval.py eval`, `rft.py evalol` (box: `$R/evalol_ol/`, `$R/evalol_near/`). *pilot-bear's CARLA number is from the
2026-10-05 pilot (prereg). Pilot lines (prereg "更正"): all four pass (losses fall; real approach gain near >= fix - 0.05: 0.99 vs 0.97, 0.73 vs 0.70;
near in-turn gain 0.75 > 0.68 with sign 0.94; CARLA exits 0.47 >= shipped + 0.10, drift <= 0.15).

Reading: the near rows do what they are built for open loop: at 0-3 m before the mouth the head's gain on the target is 2.8 x the fix arm's (0.19 vs 0.07),
left-right spread 4 x, and the plan picks the commanded exit on 94% of near approach rows (fix 60%), without lowering the real-row gain (the decision 129
failure mode). Costs: negatives get worse (CARLA N1 0.73 vs 0.62, WOD 0.34 vs 0.23, both over the 0.3 line), and imit loss falls less (2.07 vs 1.79).

## Closed loop: 9 of the 25 B2D turns (desire off, zones off, lat_exec curv, seed 2)

Turns (routes 10255, 5423, 15102, 34183, 25051, 28180, 27994, 28008 x 2): choice turns rc-bear entered late (tight and wide), a forced late, a forced
tight crawl, and two rc-bear took. Full table and per-turn causes: [near_small.md](near_small.md), split [near_small_split.md](near_small_split.md)
(`near_report.py`, same scoring as pre_report.py / rft_split.py).

| arm | took (9) | choice (5) | forced (4) | late | turn-in arc vs turn start, median m (n steered) | v median in window |
|---|--:|--:|--:|--:|--:|--:|
| shipped | 2 | 1 | 1 | 1 | 3.0 (4) | 1.6 |
| rc-ctl-s0 (4000 steps, old target) | 2 | 2 | 0 | 3 | 3.5 (8) | 1.1 |
| rc-bear-s0 (4000 steps, old target) | 2 | 0 | 2 | 6 | 4.8 (9) | 1.6 |
| pilot-bear-fix | **0** | 0 | 0 | 0 | -2.5 (1) | 2.1 |
| pilot-bear-near | **0** | 0 | 0 | 1 | 3.2 (2) | 2.2 |

Both pilots fall back to shipped's failure: 6 of the 8 entered turns "not chosen" (curvature towards the exit never reaches 0.5 / R_min); the
other two are 28008#0 crawl / stop and 5423 (fix: wrong side, near: late); both never enter 28008#1. They drive faster in the window (2.1-2.2 vs 1.1-1.6 m/s median). The near arm's open-loop turn-in at
0-3 m does not appear in the closed loop.

Verdict rule (prereg): near - fix = 0 (<= 1) and no earlier turn-in -> **no**.

## Doubts

- **400 steps is not the merged recipe.** Neither pilot matches the 4000-step arms (rc-bear-s0 2 / 9, 6 of them late), so the read compares two
  under-trained heads; a 4000-step rc-bear-fix was never run (scope cut), so whether the corrected target alone changes the closed loop is open.
  Pilot-bear (old target, 400 steps) was not run closed loop either, so "0 / 9" cannot be attributed to the corrected target vs pilot length.
- n = 9 turns, one seed, one run per cell.
- The near rows supervise curvature at the car (pure local curvature at arc v0 * 0.275 s): 3-10 m before the mouth the target is ~0 by construction
  (lane-centre paths turn at the connector), so these rows teach "turn when at the corner", not "turn in early".
- v_model scaling: on the B2D spec camera the model sees 0.87 of its speed (decision 130), and the target is in the model's speed, so the executed
  curvature (decoded with the true v) is ~0.76 of the target; the near rows below 1 m/s are unaffected (max(1, v)^2 = 1).
- Negatives regress with the near rows (N1 / WOD over the line).
- The guard tool's `peak / needed` column from rft_split (`peak` values 4-25 for every arm incl. shipped) is not a curvature ratio; it is left out.

## Files

- Box: `$DATA_DIR/runs/op_route_ft/carla_near/{plan.path, render/, packed/}` (frames 30 GB redundant after packing, pano kept), `$R/bank/near`,
  `$R/runs/pilot-bear-{fix,near}`, `$R/onnx/pilot-bear-{fix,near}.onnx`, B2D units `$R/desire_off/pilot-bear-{fix,near}/b2d/turns-small-s2-k*`,
  pool logs `$R/nrp/`. Plan run `$R/near_plan_near4k/20261005-103125/` (summary.json, target_stats.json).
- Candidates `pilot-bear-fix`, `pilot-bear-near` in experiments/op_guard/candidates.json.
