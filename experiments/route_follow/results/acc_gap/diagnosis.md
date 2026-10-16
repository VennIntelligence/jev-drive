# ACC-GAP diagnosis: why the ACC car follows farther back than the human (PAI, stored reads, CPU only)

Prereg: [plans/2026-10-16-acc-gap-prereg.md](../../plans/2026-10-16-acc-gap-prereg.md). Script `scripts/rf_accgap.py` (slim / table / report /
humanlog). Inputs on the box `$DATA_DIR/runs/route_follow/acc_gap/`: `rf_tax.py extract` of the ACC read (`pai_g10/acc`, logged traffic),
of g10 (`g10L.pkl`, logged traffic) and of the FIX1 train-scene runs (`alpasim/fix1/paibox/runs/{ab,b}_*`, P2H10-F-s0 / -s1, 60 scenes, all in
train160); per-decision `fix.lead` / `fix.acc` from drive.jsonl. True lead = the nearest vehicle in AlpaSim's route corridor ahead of the
front bumper (rf_londiag.corridor_vehicles); human gap = the logged ego's gap to the same actor (same time and same place).

## 1. Size of the gap

| read | following decisions (true moving lead <= 40 m, ego >= 2 m/s) | ego time gap s | human, same place s | per-decision ratio median | ratio >= 1.5 | per-scene ratio median |
|:--|--:|--:|--:|--:|--:|--:|
| ACC | 9 706 in 124 scenes | 3.02 | 2.28 | 1.10 (lead term winning: 1.16) | 0.24 | 1.11 (115 scenes) |
| g10 | 8 406 in 119 scenes | 3.22 | 2.30 | 1.16 | 0.32 | 1.17 (109 scenes) |

Not 1.84x. Decision 315's 1.84x (4.3 s vs 2.25 s) was the old serve on the 12 scenes classified `low_slow_behind_lead`, i.e. scenes chosen
because the car was slow there; the plan, not the lead limit, set that speed. Under ACC the lead term sets the speed on 32 % of valid
decisions (set 49 %, model 7 %, ramp 7 %, curve 6 %).

What the gap looks like in metres, steady following (|v - v_lead| < 1 m/s, lead term winning), ACC vs the human of the same read set
(descriptive) and openpilot's own law (gap = 6 + 1.45 v when the lead is read exactly):

| ego speed m/s | 1-3 | 3-5 | 5-8 | 8-12 | 12-18 |
|:--|--:|--:|--:|--:|--:|
| ACC true gap m | 12.9 | 16.9 | 17.8 | 25.9 | 31.4 |
| human (read set, steady) m | 8.9 | 11.6 | 14.1 | 19.9 | 25.0 |
| openpilot law m | 8.9 | 11.3 | 15.4 | 19.6 | 26.3 |

The excess is a 4-6 m offset, which as a time gap is +30-45 % at 3-10 m/s and small above 12 m/s.

## 2. Where the excess comes from

- **The MPC holds openpilot's law on the model's lead, almost exactly.** Residual (d + vl^2 / 5) - (v^2 / 5 + 1.45 v + 6) in steady following:
  +0.2 / +0.75 / +0.2 m at 5-8 / 8-12 / 12-18 m/s. The emulation itself converges to the law (acc_check.py check 5: 20.5 m at 10 m/s).
- **The lead head reads a moving lead short and slow.** d - true gap and vl - true lead speed (moving leads, filtered p > 0.5):

  | read | d - gap m | vl - v_lead m/s | the model's own ego speed vm - v0 m/s |
  |:--|--:|--:|--:|
  | ACC (RSG-L18-s0, read set) | -1.13 | -0.79 | +0.09 |
  | g10 (RSG-L18-s0, read set) | -0.93 | -0.90 | +0.15 |
  | FIX1 (P2H10-F, 28 train scenes) | -0.96 | -1.12 | +0.15 |

  The speed bias is in the lead head's own speed (lead v - true -0.68 m/s), not in the ego-speed term of radard's vLead (correlation with
  vm - v0: 0.00). Stationary leads are read without a speed bias (vl +0.2 m/s). Since x_obstacle = d + vl^2 / 5, a lead read 0.8-1.1 m/s slow
  costs 0.32-0.45 v metres: about 3-4 m at 10 m/s, plus the ~1 m distance bias. True gap minus openpilot's law in steady following:
  +4.4 / +6.4 / +6.7 m at 5-8 / 8-12 / 12-18 m/s.
- **Low speed (1-5 m/s) adds 2-3 m more** even in the model's view: the MPC barely closes a too-large gap at walking speed (a_mpc about 0).
- **openpilot's defaults are themselves longer than this human.** The human's following on 75 train160 scene logs (`humanlog`, no rollout,
  288 steady steps in 25 scenes): standstill gap 3.4 m (54 steps, 6 scenes); with that as stop distance, time headway 1.13 s (scene median 1.12);
  binned 7.6 / 12.4 / 14.1 / 28.1 m at 4.5 / 6.9 / 9.8 / 13.3 m/s against openpilot's 12.5 / 16.0 / 20.2 / 25.2 m.
- **Not the cause:** the probability filter / threshold (a lead is present on 91 % of following decisions; true moving lead <= 30 m with
  p <= 0.5 on 662 decisions); the MPC state (one LeadPlanner per session, warm-started, 2 ticks per 0.1 s call, never reset per decision;
  the jerk guess is reset only when |v_des - v| > 2 m/s).
- **Phantom leads (separate issue, not calibrated here):** 18 % of the decisions with a present lead have no corridor vehicle within d + 15 m;
  the lead term wins on 1 249 of them in 65 scenes (vehicles outside the route corridor on bends / adjacent lanes are part of this).

## 3. The rear-ends

The rf_tax rear contacts of the ACC read with a contact time: 31. In 18 the lead term won in the 3 s before the contact, the true lead was in the
corridor and the ego's gap was 1.2-2.5x the human's at the same time (132ae6cb 13.0 vs 8.2 m, 404541da 30.0 vs 18.4, 5c4a277a 25.5 vs 12.4,
fb138448 28.6 vs 13.9, cbb22323 8.4 vs 3.3), almost all at 0.2-6 m/s: in stop-and-go the ego stops and pulls away farther back than the human and
the logged follower, driving the human's trajectory, closes on it. This is what the clips show. 9 more were the set term (the human sped up
past the handoff speed later), 3 the model guard.

## 4. Unexplained mid-drive stops (for the user)

The clip page's 6 g10 "unexplained stops" were classified on an extract that had traffic only for zero-score rollouts (`--asl-actors`: the
rollout.asl is kept only for zeros). With the logged traffic, g10 has **3** (3294623f, 4f7cb072, a68ad579); 92af888e is following a heavy
truck, cbb22323 and fce18b62 are queues. The ACC read has **0** primary unexplained stops (g10 3). A stop >= 1 s where the human keeps >= 2 m/s
and no lead / queue explains it appears in 7 ACC scenes (g10 5 by the same rule), all late in the clip (12-19 s), 1.0-3.0 s long, and 6 of
the 7 score >= 0.98 (5ddb96bd is a zero by a side collision). Of g10's 3: under ACC 3294623f scores 1.0, 4f7cb072 and a68ad579 become
offroad zeros (they no longer stop; they leave the road).
