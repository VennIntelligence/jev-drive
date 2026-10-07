# HUGSIM ax probe: does the closed-loop ax feed drive P2H's fast entries into sharp turns?

Written 2026-10-07. Verdict: weak and not turn-specific, the ax feedback does not explain the fast entries (Result below). The set rule, the readouts and the verdict rule below were written and committed before any probe run was started;
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

## Result

Runs 2026-10-07 on the pool (4 bench run dirs `P2H10-F-s{0,1}_spec_plan_smooth[-cab5dd41e9fdc]-raxp`, 80 scenario runs, all finished, no retry). Code `scripts/ax_probe.py`
(`report`); per-run rows [hugsim_ax_probe_runs.csv](hugsim_ax_probe_runs.csv), paired rows [hugsim_ax_probe_paired.csv](hugsim_ax_probe_paired.csv),
medians / counts [hugsim_ax_probe_summary.json](hugsim_ax_probe_summary.json).

Checks. (a) Integrity: every ax0 step logged ax = ay = 0 (max |ax fed| 0.0000, max |ay| 0.0000); base runs fed ax up to a median of 4.7 m/s^2 (model clock).
(b) Determinism guard: the fresh `base` arm vs the stored r0 runs of the same arms: same end in 19 / 20 (570_770-easy s0 off_route vs bg_collision), HD max |diff| 0.024,
mean 0.003: the between-run noise to read the contrasts against (HD ~0.02, one end flip per 20 runs). (c) Offline check, passed (below).

### Offline check (before the closed loop)

365 stored approach steps of the 5 sharp scenarios (both seeds, up to the departure onset; fed ax mean +1.34, p10 -1.74, p90 +3.89 m/s^2) through the
torch port on speed-matched navtest tokens ([hugsim_ax_probe_offline.csv](hugsim_ax_probe_offline.csv)): ax = 0 changes the planned v13 as expected from the
navtest probe, lowering it where the fed ax is positive and raising it where it is negative.

| fed ax (m/s^2) | steps | mean fed ax | plan v13 as fed (m/s) | d v13 with ax = 0 (m/s) |
|---|---|---|---|---|
| <= 0.3 | 117 | -1.32 | 7.61 | +0.83 |
| 0.3 - 1.5 | 57 | +0.96 | 10.78 | -0.64 |
| > 1.5 | 191 | +3.09 | 8.61 | -1.57 |

So the input does move the plan, by about 1.6 m/s where HUGSIM's own acceleration is above 1.5 m/s^2 (a 15-20% lower plan speed): a feedback of this size is possible.

### Closed loop: paired ax0 - base (seeds averaged, v_ref in m/s)

| scenario | set | s_ref (m) | v_ref base | v_ref ax0 | d v_ref | v_onset base / ax0 | vmax base / ax0 | HD base / ax0 | end base | end ax0 | stall b/a | stuck b/a |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| scene-0041-medium-00 | sharp | 30.0 | 9.93 | 8.39 | -1.55 | 9.76 / 7.54 | 10.89 / 9.38 | 0.188 / 0.214 | bg_collision/bg_collision | bg_collision/bg_collision | 0/0 | 0/0 |
| scene-570_770-easy-00 | sharp | 62.2 | 11.47 | 10.39 | -1.09 | 11.53 / 10.39 | 13.13 / 11.41 | 0.205 / 0.231 | bg_collision/off_route | bg_collision/bg_collision | 0/0 | 0/0 |
| scene-570_770-medium-00 | sharp | 63.8 | 12.11 | 10.29 | -1.82 | 12.17 / 10.29 | 13.38 / 11.78 | 0.218 / 0.236 | off_route/off_route | bg_collision/bg_collision | 0/0 | 0/0 |
| scene-5980_6180-easy-00 | sharp | 48.7 | 6.62 | 7.21 | +0.59 | 6.70 / 7.08 | 9.16 / 8.07 | 0.215 / 0.227 | bg_collision/bg_collision | bg_collision/bg_collision | 0/0 | 0/0 |
| scene-8440_8640-easy-00 | sharp | 108.2 | 4.50 | 6.37 | +1.87 | 4.88 / 6.99 | 10.23 / 9.48 | 0.647 / 0.605 | bg_collision/bg_collision | bg_collision/off_route | 0/0 | 0/0 |
| scene-0418-hard-00 | control | 62.2 | 15.57 | 14.57 | -1.00 | - / - | 19.24 / 18.51 | 0.282 / 0.247 | complete/complete | complete/complete | 0/0 | 0/0 |
| scene-0013-medium-00 | control | 62.2 | 15.46 | 4.33 | -11.13 | - / 3.66 | 19.09 / 4.33 | 1.000 / 0.159 | complete/complete | bg_collision/bg_collision | 0/0 | 0/0 |
| scene-0411-easy-00 | control | 62.2 | 15.41 | 13.79 | -1.62 | - / - | 16.57 / 15.26 | 0.879 / 0.892 | complete/complete | complete/complete | 0/0 | 0/0 |
| scene-0167-easy-00 | control | 62.2 | 15.35 | 13.92 | -1.43 | - / - | 16.17 / 16.08 | 1.000 / 1.000 | complete/complete | complete/complete | 0/0 | 0/0 |
| scene-0166-easy-00 | control | 62.2 | 14.36 | 13.11 | -1.25 | - / - | 15.09 / 14.56 | 1.000 / 1.000 | complete/complete | complete/complete | 0/0 | 0/0 |

(v_ref of scene-8440_8640 is at s_ref = 108 m, after the car slowed from a peak of 10 m/s; scene-0013-medium-00 ax0 never reached s_ref, v_ref = its last speed.)

Verdict-rule quantities:

| | sharp (5) | control (5) |
|---|---|---|
| median d v_ref | **-1.09** m/s (mean -0.40) | -1.43 m/s (mean -3.29, -1.33 without scene-0013) |
| scenarios with d v_ref < -1.0 | **3 / 5** (-1.55, -1.09, -1.82; +0.59, +1.87 faster) | 4 / 5 (+ scene-0013) |
| median d vmax | -1.51 | -0.73 |
| mean d HD | +0.008 (4 of 5 up by 0.01-0.03, 8440 down 0.04) | **-0.173** (all from scene-0013-medium-00, -0.841; the other four -0.035 .. +0.013) |
| new launch stall / stuck | 0 / 0 | 0 / 0 |
| end class worse / better | 0 / 0 (bg / off_route swaps only) | 1 (scene-0013, complete -> bg_collision) / 0 |

## Verdict

**Not supported as the explanation: the cut is weak and not turn-specific.** By the fixed rule the sharp-turn cut is **weak** (median -1.09 m/s, between the -2.0 bar and
-1.0, and 3 of 5 below -1.0; the median sits on the line and a repeat could move it). The ax feed does matter for speed, a bit: where the car is accelerating, ax = 0 takes
about 1.1-1.8 m/s (10-16%) off the speed at the turn in three scenarios (0041, both 570_770), but the entry stays fast (6.4-10.4 m/s in all five against WA-JEPA 2.6-2.8) and every
sharp scenario still ends in the same bg / off_route failure (HD +0.008, no scenario completes). The same switch takes 1.0-1.6 m/s (6-10%) off the four straights that
still complete, i.e. it is a generic shave on speed under acceleration, not a corner effect; two sharp scenarios (5980_6180, 8440_8640) get faster. On the controls the
"harm" criterion trips on one scenario: scene-0013-medium-00 with ax = 0 stops accelerating at 4.4 m/s (peak speed 4.3-4.4 in both seeds) and ends in a background
collision at step 22 (HD 1.00 -> 0.16); I did not look into that run, so whether it is a bad low-speed launch without ax or a scene effect is open (no launch stall by the
40-step definition: the car moved at 4.4 m/s). So P2H's fast entries into HUGSIM sharp turns are not mainly a closed-loop positive feedback through ax: removing the loop
leaves the entry speed at 84-91% of the old one in the three scenarios where it falls. What remains is the model's speed plan itself (four_dirs/hugsim.md: the plan keeps accelerating until the route
command switches 2.5-6.6 m before the turn), consistent with the navtest probe (ego_history_probe.md), where the history explains at most a third of the planned slowdown.

## Caveats

- n = 5 + 5 scenarios, one scenario = one cluster, 2 seeds averaged: sign counts and medians only, no CI; the median -1.09 sits at the weak-cut boundary. The
  five controls are the fastest complete straights of one dataset (nuScenes, one per scene), a deliberate stress set, not a sample of straights.
- ax = 0 is the strongest version of the intervention (it also removes the true deceleration cue where the car brakes); a partial version (ax filtered or clipped at a
  positive value) was not tested. ay was already 0 in HUGSIM. The offline check uses navtest tokens at matched speed with HUGSIM's ego features, not the closed-loop frames.
- The baseline is a fresh run of the same code tree, not the stored r0 runs; between-run noise measured by the determinism guard is HD ~0.02 and 1 end flip in 20 runs, speeds not compared.
- Ends are bg / off_route swaps in the sharp scenarios (one 570_770-medium flips off_route -> bg_collision under ax0): these are the same failure, not an improvement.
