# op_resume pre-registration: an emulated driver resume from a held standstill (shared rule, HUGSIM + B2D)

Written 2026-10-05, committed before any scored closed-loop run. Code: `jevdrive/openpilot/resume.py` (rule), interface key
`lon.resume` (`jevdrive/openpilot/interface.py`), hooks in `experiments/hugsim/lib/zs_agent.py` (opt `resume`) and
`lib/op_arb_agent.py` (arb `resume_rule`), test `tests/test_openpilot_resume.py`. Design data (read before writing this, logs of
earlier runs only): `scripts/or_offline.py` -> `$DATA_DIR/runs/op_resume/offline/{episodes,first_fire}.csv`.

## 1. Question and premise

On many cars openpilot's exit from a held standstill needs the driver to press resume or tap the gas, so Cinque never had to
launch from a held standstill on its own. In HUGSIM `spec` (decision 118's arm; decision 124) 24 of 64 runs end at max_steps,
the car standing 74-96 s on a stop plan (decision 119: the model wants to stop; no lead in most). Earlier mechanical rules
(decisions 90, 96, 101/107, 113, 117) keyed on speed or plan direction and fired in the wrong places. This rule keys on *why a
driver would resume*: the car has been held for a while and the model's own lead head sees nothing close ahead. No ground-truth
light, route, map or actor state is read; the light stays invisible to the rule (declared; guard 2 measures the cost).

Settled things not re-tested (decisions grepped): desire does not help (92); the 5 s static warm-up stays (124); action
acceleration -> LongControl fails launches (119); lateral stays the action curvature through openpilot's path (118).

## 2. The rule (defaults = these values; `resume.DEFAULT`)

States idle -> launch -> cooldown. Inputs: board time, ego speed, the model's lead head (lead_prob, lead x at t = 0 of the first
hypothesis, metres ahead of the device), the distance the board's plan covers in 1 s.

| Parameter | Value | Why |
|---|---|---|
| `v_stand` | 0.1 m/s | standstill. HUGSIM stuck runs stand at 0.00-0.04 m/s, B2D at 0.0 |
| `t_stand` | 10 s continuous | (a) > B2D's existing timer resume (`latch_max_s` 5 s), so on B2D the rule only acts where the timer did not get the car moving and the two never release the same stop; (b) B2D red light remaining after the car stops: p50 9.3 s, p75 20.5, p90 27.6 s (491 stops, 58 logged runs) - with the timer's pulses the 10 s continuous standstill is not reached at a red in the logs (first-fire replay: 0 of 58 B2D runs fire at a red / yellow light); (c) HUGSIM: standstills >= 10 s that the model ended itself: 4 episodes (322492347634 x2, 124-extreme-01, 3400_3600-hard) against 24 runs that stood to the end |
| `p_lead` | 0.5 | openpilot's own threshold for a vision lead (radard uses model leads with prob > 0.5); also our B2D IDM's `lead_p` |
| `d_lead` | 15 m | launch to 2.5 m/s at 1.2 m/s^2 covers 2.6 m, braking from 2.5 m/s at 2 m/s^2 1.6 m, + IDM s0 2.5 m + device-to-bumper ~2 m = ~9 m; 15 m leaves ~6 m. Data: it holds the 3 HUGSIM stuck runs whose lead head reads 5-12 m (0254-hard, 2510_2710-hard, 3000_3200-medium) and the B2D queue standstills (lead 4-10 m); it passes the stuck runs whose lead head reads >= 16.7 m (0528-medium: 16.7 m, ground-truth box ~25 m) |
| `t_clear` | 1 s | the lead head must have been clear for the last second (4 HUGSIM steps, 20 B2D ticks): no firing on one flicker |
| `a`, `v_end` | 1.2 m/s^2 to 2.5 m/s | the launch profile of decision 117's pre-registration (matched to comma1M real launches, 1.5 / 3.4 m/s at +1 / +2 s); v_end in the brief's 2-3 m/s, reached in ~2.1 s |
| hand-back | v >= v_end ("speed"); the model's own plan covers at least the profile's 1 s distance ("plan"); lead head stops being clear ("lead_abort"); `t_max` 6 s ("timeout") | the plan takes over as soon as it asks for as much as the rule; t_max = ~3 x the nominal launch, a car that is not at 2.5 m/s by then is blocked |
| `t_cool` | 10 s after a hand-back; the standstill clock restarts after it | >= 20 s between launches (<= ~4 per 100 s HUGSIM episode); with the lead gate it cannot pulse at a lead the head sees |

Board action while launching. HUGSIM: after forward_only / straight_stop, the plan is re-timed along its own path (straight ahead
for a stop plan) to max(its own arc, the launch profile); iLQR tracks it longitudinally, lateral stays op_ctrl's action
curvature. No other protection than the lead gate exists on HUGSIM (as on the car with a resume press). B2D (`drive` / `spec`):
the plan and latch constraints are dropped and replaced by max(profile, plan); the lead-head IDM and the set-speed governor stay;
the timer resume stays (B2D's declared deviation); interface value `timer+rule`.

Counterfactual replay of these values on the logged runs (first firing only; after it the log no longer applies): HUGSIM
20 of 24 stuck runs fire (t 14-39 s; not: the 3 held by the lead gate and 8440_8640-easy, which never stands still, it crawls), 2 of 40 others (124-extreme-01, whose ground truth has a box in the lane ~11 m ahead the
head missed: the guard-3 risk case; 3400_3600-hard); B2D 7 of 58 runs, none at a red / yellow light.

## 3. Readouts (definitions fixed here)

- **fired**: the rule entered launch at least once (`rr == "fire"` in the step / plan log).
- **launch**: after the first firing the ego reaches >= 2.0 m/s. **Launch rate** = launched / runs of the set.
- HUGSIM HD / RC / end class from zs_run's `results.csv`; B2D DS / RC / infractions from the official record.
- Paired differences rule - spec, `jevdrive.stats.paired` (scene / route-seed units, 10 000 resamples, seed 0).
- **Collision after firing** (HUGSIM): the run ends fg / bg collision within 10 s after a firing.
- **Firing at a light** (B2D, ground truth used only here, never by the rule): a firing tick with a red / yellow light within 40 m.

## 4. Stages

Arms on every case: `spec` (preset as it is, rerun the same day on the same code) and `rule` (spec + `resume` defaults).

**Pilot (~10 cases, <= 3 GPU h; est. ~1).** HUGSIM (`scripts/pilot_hugsim.txt`): 6 stuck under decision 118 -
0418-hard, 0411-medium (nuScenes), 034-easy-00, 095-medium-01 (PandaSet), 113792265837-easy (Waymo), 3000_3200-medium (KITTI-360,
lead head 5-8 m: the rule must hold it); 2 not stuck - 0051-easy (completes), 124-extreme-01 (the missed-lead risk case).
B2D seed 2: 24944 (red light, long reds), 9196 (red light + 20-47 s lead standstill), 17280 (stop sign).

**Pilot gate (stop and report if any holds):**
1. fewer than 3 of the 6 stuck HUGSIM cases launch;
2. a HUGSIM guard case regresses: 0051-easy or 124-extreme-01 HD rule < spec - 0.02, or a collision ending within 10 s after a
   firing in any pilot scene, or 3000_3200-medium fires while its lead head reads < 15 m (a code bug);
3. a B2D guard case regresses: in a run where the rule fired, red_light + stop_infraction + collisions_vehicle rule > spec,
   or any firing at a light. (Runs where the rule never fired run the same agent; their differences are repeat noise, d118.5.)

**Full (est. ~3 GPU h; report before anything above 8).** HUGSIM all 64 (`derot_all64.txt`), both arms. B2D guard subset, seeds
2 and 3, both arms: light routes 16390 15612 15483 27043 15102 28147 24944 9196, stop-sign / non-signalised 17280 16529 16508, lead
queues 26153 26365.

**Lines (full):**
- F1 (fix): stuck set = decision 118's 24 max_steps scenes (fixed list, `results/op_control_stack/opctrl_runs.csv`): launch rate
  reported; HD paired rule - spec, **lower CI > 0**. RC paired reported.
- G1 (HUGSIM no harm): the other 40 scenes, HD paired **lower CI > -0.02** (decision 118's line).
- G3 (stopped lead, HUGSIM): runs ending in a collision within 10 s after a firing **<= 1**, and collision endings (fg + bg)
  rule - spec **<= +1** over 64.
- G2 (B2D): over runs where the rule fired, red_light, stop_infraction and collisions_vehicle each **rule <= spec**; firings at a
  light reported (expected 0). Totals over all runs and DS paired CI reported without a line.

GIF: one fixed stuck case, scene-0418-hard-00, rule arm, third-person render next to the model's input frames (repo convention),
rerun with `dump_every 1`.

Deviations from this plan are listed in the result note, with the reason.
