# HUGSIM ax probe: does the closed-loop ax feed drive P2H's fast entries into sharp turns?

Written 2026-10-07. The set rule, the readouts and the verdict rule below were written and committed before any probe run was started;
results are appended under "Result". Candidate under test (ego_history_probe.md, caveats; four_dirs/hugsim.md direction 1): HUGSIM feeds the sim's
acceleration as ax, P2H's plan speed rides on ax (ehp `acc0` = `cv`), so an accelerating history could raise the plan, which raises the
acceleration (positive feedback), giving the 9.8 m/s median entry into R 8.9 m corners (WA-JEPA 2.6-2.8 m/s).

## Design

Two arms per (P2H seed, scenario), preset `spec_plan_smooth`, P2H10-F-s0 and -s1, bench repeat label `axp` (fresh runs of both arms, same code tree, same pool):
`base` = unmodified; `ax0` = option `parity.zero_acc` (`lib/parity_hugsim.py`, default off, bench opts `{"parity": {"zero_acc": true}}`, a separate run
identity): ax = ay = 0 in the 20 ego features sent to the bias server (the adapter's only ax input). HUGSIM reports `info["accelerate"]` as ax and no lateral
components, so ay is already 0 in the base arm; nothing else (speed, 4-pose history, command, images, controller, plan-to-actuation path) changes. Runs are
effectively deterministic (hugsim_specplan.md repeats), one run per arm and scenario and seed; seeds are averaged per scenario before the paired contrast.
Determinism guard: the `base` arm is compared with the stored r0 runs of the same arms (HD and end).

## Scenario set rule (fixed; code `scripts/ax_probe.py select`, from `four_dirs/hugsim_events.csv`, the stored P2H `spec_plan_smooth` runs, 6 per scenario)

- **sharp (5)**: the D1 scenarios of four_dirs/hugsim.md: scene-0041-medium-00, scene-570_770-easy-00, scene-570_770-medium-00, scene-5980_6180-easy-00,
  scene-8440_8640-easy-00 (R 7.9-9.8 m). Reference route position `s_ref` = median over the 6 stored runs of the route position of the departure
  onset (last step with lateral distance to the route < 1.5 m before the bg / off-route end; the "onset" of four_dirs/hugsim.md): 30.0, 62.2, 63.8, 48.7, 108.2 m.
- **control (5 straight)**: scenarios with `turning == False` (route yaw range < 30 deg), all 6 stored runs ending `complete`, at most one per scene,
  ranked by the median over the 6 runs of the peak speed, top 5 (the fast straights: where an ax feedback would show if it exists, and where a
  slowing would hurt): scene-0418-hard-00, scene-0013-medium-00, scene-0411-easy-00, scene-0167-easy-00, scene-0166-easy-00 (stored peak 15-19 m/s,
  all nuScenes). Their `s_ref` = median of the five sharp `s_ref` = 62.2 m.
  Table: [hugsim_ax_probe_sets.csv](hugsim_ax_probe_sets.csv).

## Readouts (per run; seeds averaged per scenario; paired ax0 - base)

- **v_ref** (primary): speed at the first step with route progress >= `s_ref` (route projection of the ego box centre, the recorded route as in
  fd_hugsim.py); if the run ends earlier, its last speed (`reached` counts runs that got there). On the sharp scenarios this is the entry speed at the
  turn, same position as the four_dirs onset speed (and `v_onset` itself is reported where the run departs the route).
- peak speed `vmax`, mean speed over route progress 10 m .. `s_ref` (`v_win_mean`), mean simulated acceleration there, HD-Score, end class,
  launch stall (peak speed over the first 40 steps < 1.6 m/s) and stuck (`max_steps` end) as guards.
- Integrity: ax0 runs must log ax = ay = 0 on every step (`zs_steps.jsonl` parity ego), base runs a nonzero ax.

## Verdict rule (fixed)

d = ax0 - base on v_ref, per scenario (mean over the 2 seeds), n = 5 + 5 scenarios, so sign counts and medians, no CI claimed.
- **Cut**: median d on the 5 sharp <= -2.0 m/s and d < -1.0 m/s in >= 4 of 5 scenarios.
- **No harm**: on the 5 controls median d(v_ref) >= -1.5 m/s, mean d(HD) >= -0.05, no new launch stall or stuck run, end class not worse in more than 1 scenario.
- Cut and no harm: **supported** (the ax feed is a mechanism of the fast entry and can be switched off at no cost on straights).
  Cut and harm: **non-specific** (ax drives speed everywhere). No cut: **not supported** (the ax feedback does not explain the fast entries; a
  median d between -2.0 and -1.0 m/s or a 3 / 5 count is reported as **weak**). Whether the turn is then made (end class, HD) is reported but is not part of the verdict.

## Offline check before the closed loop (one quick check)

Stored HUGSIM ego features (all approach steps up to the departure onset of the 5 sharp scenarios, stored P2H10 r0 runs, both seeds) go through the torch
port (`pp_train.PModel`, fp16) on navtest front tokens of matching speed (the closed-loop frames are not stored), ax as fed vs ax = 0; readout planned speed
v13 = (x(3 s) - x(1 s)) / 2 (model clock). Expected from ego_history_probe.md: ax = 0 lowers the plan where the fed ax is positive.
