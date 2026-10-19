# PAI-INPUT-LOOP prereg (2026-10-19, written before any new read exists)

Diagnostic, no training. Checkpoint NTV0-L18-s0 (the PAI candidate with ACC + GAP, decision 366: PAI 281 0.744, progress 0.88, rear-hit
partials 26). Code: `experiments/route_follow/scripts/pai_input_loop.sh` (reads, gate, extraction, counterfactual) and `pai_input_loop.py`
(read-out). Results: `experiments/route_follow/results/pai_input_loop.md`.

## Questions

1. **Train / serve frame mismatch.** PAI serves 8 rendered 10 Hz frames as the policy slots; the checkpoint was trained on 2 Hz keyframes plus
   warps. `PAI_KEYWARP=1` builds the slots as training did (pai_keywarp.md: equivalence checks passed 10-11; the unit tests
   `tests/test_pai_keywarp.py` pass on the current code, 3 / 3, 10-19). Does the training-matched input help or hurt on PAI 281, and where?
2. **Speed loop.** On PAI, how does ego / human speed at the same station evolve over the 20 s rollout, with and without ACC + GAP, and how
   much of the per-step drop is the plan (model plan, then the serve fix) against execution (decision 372's A2 split)? Does the slowdown come
   through the speed-history row or through the frames?

Note on the premise: decision 285's "the log speed history recovers 83 %" (`experiments/steer_long/scripts/sl_cf.py`) was measured on the
**nuPlan held-452** zero rollouts' driver messages, not on PAI; decision 372 cites it as "d285 PAI 83 %". There is no PAI history
counterfactual yet; item 2c below is the first.

## Reads (pool, priority 12, g10 = `controller.gains.rel_front_steering_angle_weight=10`, CONC 4, ALPASIM_FAST=1, the 8 chunk lists of the
stored read `runs/defect_fix/pai_accgap/NTV0-L18-s0/chunks`, code = this commit; root `$DATA_DIR/runs/route_follow/pai_input_loop/`)

| arm | serve | slots | role |
|---|---|---|---|
| A0 (stored) | VCONT 1.0 + LEAD + ACC + GAP 0.7,2.7 | rendered 10 Hz | reference, `runs/defect_fix/pai_accgap/NTV0-L18-s0` (code 2e2d0e46; pai_driver / serve_fix / sh30 unchanged since) |
| KA | same as A0 | `PAI_KEYWARP=1` | contrast 1a: KA - A0 |
| N0 | VCONT 1.0 + LEAD | rendered 10 Hz | no stored NTV0 no-ACC read exists, so read here (`KEEP_ASL=1`, the counterfactual's input) |
| KN | VCONT 1.0 + LEAD | `PAI_KEYWARP=1` | contrast 1b: KN - N0 |

Staging: KA chunk c000 first; a CPU gate job checks its drive log (every `drive` record has `keywarp`, 0 inference errors, at least 30
sessions) before the other 7 KA and all 8 KN chunks start. N0 does not touch the switch and starts at once.

## Item 1 read-out

`defect_fix/scripts/pai281.py` unchanged (scenes scored by both arms without a driver RPC error, log-clustered paired bootstrap, 10 000 draws,
seed 0; zero types, partial types, strata static / launch / moving). Added in `pai_input_loop.py`: a **turn** stratum (the human's logged
path turns > 30 deg over the rollout; moving scenes are split into moving-straight / turn) and per-scene progress.

Lines (per contrast, 1a and 1b): **helps** = CI low > 0; **hurts** = CI high < 0; otherwise **no measurable effect** at this size (expected
half-width ~0.03). The strata are descriptive. The two contrasts together: same sign in both = a frame-source effect that does not depend on
the serve fix.

## Item 2 read-out (all four arms)

Per rollout from the controller trace and the logged rig path (`rf_tax.py extract`, the `Roll` alignment: the ego's rear axle projected on the
logged path, the log speed at that station), on the 0.5 s grid after the handoff (t >= 1.7 s):

- **2a ratio vs time**: median over rollouts of ego speed / log speed at the ego's station, steps with the log >= 2 m/s at the station, by 1 s
  bin of time since the handoff (0-18 s); also the share of steps with ratio < 0.8.
- **2b split** (decision 372 A2, at 0.5 s steps k -> k+1 between inference decisions, log >= 2 m/s at both stations, ego >= 1 m/s):
  `plan_model = log(vpm(0.5) / v) - log(vh[k+1] / vh[k])`, `fix = log(vps(0.5) / vpm(0.5))` (served against model plan; 0 when no serve fix
  acts), `exec = log(v[k+1] / vps(0.5))`; vp(0.5) = |p(1.0 s)| / 1 s, as decision 372. The three terms sum to the step change of
  log(ego / log). Pooled means, plan share = (plan_model + fix) / total; split by phase (handoff + 0-5 s, 5-10 s, 10-18 s).
- **2c history vs frames (open loop on N0's stored messages)**: N0's driver messages replayed in arrival order through the real
  `pai_driver.Driver` (NTV0, the same serve env), as `steer_long/scripts/sl_cf.py` did on nuPlan; at every inference the policy is called on
  the same slot frames three times: F as served (gate: the model plan equals the stored `poses_model` within 0.03 m on >= 99 % of the
  decisions); A the history velocities rescaled to the log speed at the ego's station at each history key (direction kept), longitudinal
  acceleration from the log; B = A plus the history key positions scaled radially by log / ego mean speed (clip 0.5-3). r = plan 1-2 s speed /
  log speed now; recovery = (r_A - r_F) / (1 - r_F) on decisions with the log >= 2 m/s, the ego >= 1 m/s and r_F < 0.95, at t >= 3 s.
  Line (sl_cf's): recovery_B >= 0.5 -> **through the history row**; <= 0.2 -> **through the frames** (or elsewhere in the input); between ->
  partial. Reported also on the subset ego < 0.8 x log.
- The KW contrast is the closed-loop frame check: the 2a ratio of KN vs N0 at the same time bins.

Not claimed: anything about the leaderboard; one seed; 2c is open loop (the frames stay those of the slow rollout).
