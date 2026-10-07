# Lead standstill margin (execution-layer rule): the offline gate fails on both boards; no closed-loop run

Written 2026-10-07. Pre-registration [plans/2026-10-07-lead-margin-prereg.md](../plans/2026-10-07-lead-margin-prereg.md) (approved by the
user; the declarations and deviations section was committed in 1e33c723, before any score was read). Driver P2H (P2H10-F-s0 / s1), HUGSIM
lateral path `spec_plan_smooth` (decision 149). Rule `jevdrive/openpilot/lead_margin.py`, one function shared by the HUGSIM client
(`zs_agent.py` opt `lead_margin`) and the NAVSIM export (op_interp adapter `lm`, bench model option `:lm`); interface trick `lead_margin`
([docs/openpilot-interface.md](../../../docs/openpilot-interface.md)). Offline script `scripts/lm_offline.py`; tables in
[lead_margin/](lead_margin/).

**Outcome.** Step 1 (offline) failed two of its three gates. The prereg says to stop, so the 5-scenario HUGSIM read, HUGSIM 64 and navhard
were not run. No GPU HUGSIM job was submitted. The only pool jobs were the two navtest `:lm` runs (plans on the GPU for about 1 min each,
then CPU devkit scoring).

| gate (prereg step 1) | threshold | result | pass |
|---|---|---|---|
| HUGSIM D3b: scenarios where the rule turns the pre-contact plans into avoiding ones (primary: every step of the last 3 s keeps clear) | >= 7 / 10 | **1 / 10** (and that one, 3000_3200, the rule never changed: its baseline plans were already clear) | no |
| HUGSIM complete runs: steps the rule changes (a plan point pulled back > 0.05 m) | < 5 % | 4.71 % (16 570 steps, 150 runs, 27 scenarios) | yes, barely |
| navtest EPDMS, LM - P2H | >= -0.2 and NC + TTC failures down | **-0.75 [-0.96, -0.55]**; NC + TTC failures 531 -> 565 | no |

## The rule (as registered, values not tuned)

Trigger lead_prob > 0.5. Corrected gap g = lead_x - b(lead_x), with b = 2.0 m below 6 m, falling linearly to 0 at 10 m (from
`four_dirs/hugsim_lead_calib.csv`). Cap s_max(t) = max(0, g + max(lead_v, 0) t - 2.5 m). The plan becomes s'(t) = min(s(t), s_max(t)) along
its own path: the shape is kept, only the speed profile changes, and the rule never adds motion. Lead head: selection 0 at t = 0, the
decoding the HUGSIM server uses. Declared readings (prereg section "执行中的声明与偏离"):

- "Pulled back" is taken as the min. The literal "set everything after the first violation to s_max" would push a plan forward wherever it
  later falls below the cap.
- HUGSIM lead_v is divided by 1.25 because of the dilated clock.
- d_min is measured from the camera. On HUGSIM that leaves a 1.0 m gap at the bumper; on NAVSIM it leaves 0.12 m (CAM_F0 sits 2.38 m behind
  the Pacifica's bumper).
- NAVSIM: the stored plans had no lead columns, so the bench parity plans stage now saves them. The `:lm` plans are bit-identical to the
  stored P2H plans (max |d plan_pos| = 0). Scoring used the full-navtest bench devkit rather than the fd_navsim failing-token replay.

## HUGSIM offline replay (stored traces, 56 D3b runs, 10 scenarios)

Setup. At each of the 12 steps before contact, the rule is applied to the stored plan with the stored lead head. A step "keeps clear" if
the ego box at the plan's 0.5 / 1.0 / 1.5 s points stays off the hit actor's recorded box at the matching steps.

| step before contact (0.25 s) | -12 | -10 | -8 | -6 | -5 | -4 | -3 | **-2** | **-1** |
|---|---|---|---|---|---|---|---|---|---|
| plan keeps clear, stored (base) | 0.89 | 0.95 | 0.84 | 0.43 | 0.34 | 0.18 | 0.11 | 0.05 | 0.05 |
| plan keeps clear, with the rule | 0.95 | 1.00 | 0.84 | 0.73 | 0.89 | 0.75 | 0.66 | **0.07** | **0.11** |

Reading:

- Between -1.5 and -0.75 s the rule clears most plans: the share of clear steps over the window rises from 0.53 to 0.74. Under the
  secondary definition (at least half of the steps clear) all 56 runs and 10 / 10 scenarios convert.
- In the last 0.5 s it fixes almost nothing. Only 3 / 56 runs are clear at every step, and those 3 are the 3000_3200 runs whose stored
  plans were already clear.
- Primary gate: 1 / 10 scenarios.

Why the rule plan still touches (175 failing steps; [offline_d3b_steps.csv](lead_margin/offline_d3b_steps.csv)):

| cause | all failing steps | last 0.5 s |
|---|---|---|
| in-lane actor, rule triggered, but lead_x reads much further than the 2 m the table removes | 100 | 72 |
| actor laterally offset (\|lateral\| >= 1 m, partly in the lane): the lead head reports another, moving object (median lead_x 13.2 m at a true gap of 2.8 m) | 51 | 24 |
| lead head below 0.5 (not triggered) | 24 | 6 |

The offset-actor cases are 0254-hard, 0528, 2510_2710 and 3000_3200. In the last 0.5 s before contact, the in-lane lead_x error is a
median +4.4 m at a median true gap of 1.9 m. That is about twice the four_dirs mean for gaps < 3 m (+1.95 m), which averages over all P2H
steps and presets. So the near-range bias grows exactly where the contact happens. A fixed 2 m correction leaves the stop point inside the
actor, and lead heads that track another object cannot be corrected by any bias table.

Complete runs (stop-risk proxy). The rule triggers on 38.8 % of steps and changes 4.71 %. It turns a moving plan into a stop on 0.02 % of
steps. The changes come almost entirely from one scenario, 095-medium-01, a lead-following drive where 66 % of steps are capped; the
median scenario has 0 %. In closed loop this scenario is the one to watch for a slowdown or a stall (decision 140's failure mode).

## navtest (full, 12 146 tokens, v2 devkit through `jevdrive.bench`)

| arm | EPDMS | per seed | NC | DAC | EP | TTC | EC |
|---|---|---|---|---|---|---|---|
| LM (P2H + rule) | 87.92 | 87.85 / 88.00 | 98.43 | 96.25 | 86.04 | 98.05 | 85.15 |
| P2H | 88.67 | 88.58 / 88.77 | 98.58 | 96.17 | 87.16 | 97.96 | 88.62 |

- **LM - P2H = -0.75 [-0.96, -0.55]** (log-cluster bootstrap). Shapley split of the gap: EP 0.35, EC 0.34, NC 0.15
  ([navtest_shapley.md](lead_margin/navtest_shapley.md)).
- The rule triggers on 48 % of tokens and shortens the 4 s plan by > 0.05 m on about 935 per seed.
- On those changed tokens: EP -14.6, two-frame comfort -47 (the capped plan disagrees with the previous frame's plan), NC -1.7 to -2.2.
- NC + TTC failures per seed: s0 274 -> 290, s1 257 -> 275. The rule fixes 18 / 16 tokens and creates 34 / 34, 31 of them new NC
  failures.
- The new failures are near-standstill queue tokens (plan start speed median 0.8 m/s, lead_x median 6 m) where the rule holds the car
  about 6 m short of the plan. The at-fault collision type behind them was not extracted (open issue).
- The 0.12 m NAVSIM bumper margin is not the cause: the new failures are tokens where the rule stopped the car early, not late.

## Guards

- No closed-loop run, so no stuck or launch-stall reading exists.
- Offline, the rule never adds motion. It makes a stop out of a moving plan on 0.02 % of complete-run steps, but caps 66 % of
  095-medium-01. That is where decision 140's standing failure would show.

## Deviations (all declared in the prereg before scoring)

NAVSIM lead columns were added and scored with the full-navtest bench devkit rather than the replay subset. The min reading of "pull
back". lead_v / 1.25. The primary and secondary offline definitions. Nothing was changed after scoring. The failing-step breakdown and the
per-step table were added to the script after the gate had been read; they are diagnosis only and are disclosed here.

## Caveats and open issues

- Offline replay holds the stored ego trajectory fixed. A rule active through the whole approach would have arrived slower and earlier, so
  "last 0.5 s" is a pessimistic proxy for closed loop. The gate was registered on this proxy, and it fails on both boards.
- The navtest NC increase (31 new at-fault failures per seed on near-standstill tokens) is not explained. The next step would be a
  collision-type extraction on those tokens with the fd_navsim replay instrumentation, adding pred-file selection for the `lm` export.
- What the readings point to, not tested here: a rule on this lead head would need a bias that grows as the gap closes, or a different
  near-range distance signal. Laterally offset actors are outside what a lead-head rule can see. That supports the prereg's own caveat
  that the fix lives in the near-range distance representation (four_dirs fix 4) or the agent hinge.
- The code stays in: the option is off by default and both boards' default behaviour is unchanged (tests
  `tests/test_openpilot_lead_margin.py`).
