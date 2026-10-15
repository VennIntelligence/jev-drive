# Pre-registration: RECEDING-TURN fingerprint on stored closed-loop logs (no training, no rollouts, CPU only), 2026-10-15

**Why.** research/next-round/turn-night.md section 9 ("the turn recedes"): the policy re-plans every decision from the current pose and
places the turn ~3 s ahead (d284: the command's effect lands at 3-4 s of the plan, the MPC tracks 1-2 s); the car advances without turning
enough; the next plan places the turn ~3 s ahead again, until the junction is upon the car, which then turns late and wide (d299: late-wide
34 / 72) or, once turning, continues its own yaw rate past the target (over-turn; d205 / d218). Supporting: d305 (plan geometry 2 s before
the failure does not separate failed and completed turns), d303 (human entry state did not rescue the 22 runway turns, 4 / 22), d209.
The mechanism proposal (intent memory + self-deviation head) is built only if this fingerprint holds. Written and committed before any
statistic below is computed. Amendments are appended, never edited in place.

## Data (stored reads, read-only)

| set | checkpoint | where | turns | plans |
|:--|:--|:--|:--|:--|
| pai_RSG (primary) | RSG-L18-s0, PAI 281 gains 20 | Tokyo `/data/runs/route_night/r281/rsg`, rf_pai_follow extract `/data/runs/route_night/pai_follow/r281.pkl` (label `rsg`) | strict 36 (`results/pai281/strict36.txt`) | every `infer` drive record (10 Hz), `poses` (served) and `poses_model` |
| pai_RV2a, pai_RV2b, pai_APP1X | RV2a-L18-s0, RV2b-L18-s1, APP1X-L18-s0 (EXPLORATORY, d303) | Tokyo `/data/runs/route_night/v2pai/<tag>` (runs + rerun), its `follow.pkl` | the same 36 scenes | same |
| held_<tag> (replications) | RSG-L18-s{0,1}, RV2a-L18-s{0,1}, RV2b-L18-s{0,1}, held 460 @18 | box `$DATA_DIR/runs/route_mem/read/<tag>/c*/run`; rf_tax extracts (`route_follow/ledger/held_RSG-*.pkl`; RV2a / RV2b extracted now with the same `rf_tax.py extract`, classes by `rf_tax.py report`) | strict turns of `route_follow/v2/report/follow/scenes.csv` label `<tag>@18` (24 each) | `driver-logs/drive.jsonl` `drive` records of the scene's last session (2 Hz), `poses` |

Per rollout: the route = the stored rebuilt AlpaSim route (MAP, 40 m; PAI: the label's own route in the rf_pai_follow extract; held: the
rf_tax extract's route), the controller trace (10 Hz), the human log path, t_f = the first zero-score event (PAI: rf_pai_follow `tz` when
the score is 0; held: `T0 + 0.5 s x kz` of scenes.csv; none -> +inf). Turn = rf_pai_follow.one on the label's route (PAI, as rf_appread) /
the stored scenes.csv row (held): side, entry, exit (10 % / 90 % of the heading change), outcome; completed = outcome `completed`, failed =
any other outcome. Turn sign sigma = +1 left, -1 right. Ego station s_e(k) = the decision anchor (rear axle, local frame) projected on the
route with rf_follow's local window (-30 / +60 m around the controller's station at t0); d_e(k) = entry - s_e(k).

## Per-decision quantities

Plan = the served `poses` (8 poses at 0.5 .. 4.0 s, anchor frame: x, y, yaw), prefixed by the origin (0, 0, 0); arc a_i = cumulative
chord length; heading psi_i = sigma x unwrap(yaw_i) (positive = into the turn).
- **s_on(k)**: the arc where psi first exceeds 15 deg, linear interpolation between the two poses around the crossing; NaN if it never does.
- **dpsi(k)**: psi_8 (deg), the plan's total heading change over 4 s toward the turn.
- **a4(k)**: the plan's arc length at 4 s (horizon).
- **route onset s_onR(k)**: the first route station s > s_e where sigma x (h(s) - h(s_e)) > 15 deg (h = rf_pai_follow.heading, chord +-5 m),
  minus s_e; **lag(k)** = s_on(k) - s_onR(k).
- **world onset W(k)**: the plan point at s_on(k) (interpolated) composed with the anchor, projected on the route (window -30 / +120 m
  around s_e) -> station W_st(k).
- **human "plan"** (label sanity): the human log's pose when its route station first reaches s_e(k), and its poses 0.5 .. 4.0 s later (log
  clock) in that pose's frame; heading = the log path's chord heading over +-1 m of log arc; the same s_on / dpsi / W.
Approach decisions of a turn: d_e(k) in [0, 40] m, t0 < t_f, and before the ego first crosses the entry station.

## Statistics

Per turn, over its approach decisions with s_on defined:
- **slope** = Theil-Sen slope of s_on on d_e (all pairs with |delta d_e| >= 1 m); defined when >= 3 such decisions span >= 5 m of d_e
  (else undefined, counted). Slope ~ +1: the onset is fixed in the world; ~ 0: the onset is fixed relative to the ego (receding).
- **drift** = Theil-Sen slope of W_st on s_e (same pairs rule) x the s_e span of those decisions (m; + = the world onset moves forward
  along the route with the car). A fixed onset gives 0; a receding one gives ~ the span. (Routine reading of the brief's "drifts toward
  the entry": the onset placed before / beyond the entry moves forward along the route as the car advances; the sign is forward.)
- Descriptive per turn: visible share (approach decisions with s_on defined), d_e at first visibility, median lag over d_e <= 15 m,
  "unseen" share = decisions whose horizon reaches the route onset (a4 >= s_onR) with s_on NaN or > s_onR + 5 m.

Bootstrap: B = 10 000, seed 0, percentile 95 %, clusters = scenes (held pooled: the log id, both seeds and all checkpoints of a log
together); a replicate resamples clusters and recomputes both group medians; replicates with an empty group are dropped (counted).

**C1 (primary: pai_RSG, strict 36).** M_f, M_c = median slope of failed / completed turns; D = M_c - M_f with its CI.
- supported: M_f <= 0.3, M_c >= 0.7 and D's CI lower > 0;
- refuted: M_f >= 0.7, or D's CI upper < 0.4;
- inconclusive: otherwise, or < 3 turns with a defined slope in either group.
Replications (same rule, own verdicts): pai_RV2a, pai_RV2b, pai_APP1X, each held set, held pooled.

**C2 (same population, same rule shape).** Medians of drift, D = median failed - median completed.
- supported: failed median >= 10 m, completed median <= 3 m and D's CI lower > 0;
- refuted: failed median <= 5 m, or D's CI upper < 5 m;
- inconclusive: otherwise, or < 3 turns per group.

**Gates (before reading C1 / C2).**
- G-frame: PAI and held, median |anchor - controller position at t0| <= 0.3 m (else the frames are fixed first).
- G-entry (held): the strict turn recomputed on the rf_tax route with rf_pai_follow.find_turn over [s0, s_log_end] has |entry - stored
  entry| <= 1.0 m on >= 90 % of the turns (else the recomputed entry is used and said so).
- G-human: on pai_RSG strict 36 the human "plan"'s median slope lies in [0.8, 1.2] and its median |drift| <= 3 m. Failing it, C1 / C2
  read inconclusive (the measure does not see a fixed onset as fixed) and only the descriptives are reported.

**C3 (descriptive).** dpsi(k) median per d_e bin (0-10, 10-20, 20-30, 30-40 m) by class: completed, late-wide, straight-through,
over-turn, other failed; human dpsi in the same bins.

**C4 (descriptive, the over-turn side).** Inside decisions: s_e in [entry, exit + 10], t0 < t_f. Plan remaining = dpsi(k); true remaining =
sigma x wrap(psi_exit - yaw_ego(k)), psi_exit = the route tangent at exit + 10 m, yaw_ego = the anchor yaw. Per class (over-turn vs
completed first): the share of decisions with true remaining <= 0 (at / past the exit heading) whose plan remaining > +5 deg, the median
plan remaining there, the median (plan - true) over all inside decisions, and the OLS slope of plan on true remaining.

**Classes of failed turns.** Taxonomy primary through rf_progress.TURN_CLS (straight / late_wide / over_turn / wrong_exit) where a taxonomy
exists (pai_RSG: `results/taxonomy/pai281_rsg.csv`; held: `results/ledger/tax/RSG-L18-s{0,1}.csv`, RV2a / RV2b from the new rf_tax report);
a zero with a non-turn primary, and the PAI sets without a taxonomy (RV2a, RV2b, APP1X): rf_tax's corridor heading rule at t_f (log heading
change over log stations [s(t_f - 3 s), s(t_f) + 15 m] vs ego yaw change over [t_f - 3 s, t_f]; the log heading from its path chord, +-1 m);
failed without a zero event (short / off with score > 0): `no_zero`. The rule's agreement with the taxonomy on pai_RSG / held RSG is
reported. The fingerprint (slope, drift, lag, unseen) is tabled per class; C1 / C2 verdicts use failed vs completed only.

**Also reported.** The same slope / drift on `poses_model` (PAI; before the lead fix) and on the human "plan"; APP1X vs RSG on the 22
runway turns (`results/approach/pai_runway22.txt`): does the onset still recede with the human's entry speed (per-turn slope paired,
median difference, scene CI).

## What the outcomes mean

- C1 and C2 supported -> the receding turn is a fingerprint of the failures: the intent-memory / self-deviation mechanism (turn-night.md
  section 9) is built and tested.
- Refuted -> failed turns place the onset as fixed in the world as completed ones: the failure is not re-planning drift; the mechanism is
  not built on this argument.
- Inconclusive -> no design decision rests on it; the counts are reported.

## Outputs

`scripts/rf_receding.py` (extract-pai on Tokyo, extract-held on the box, read on the Mac), results `results/receding/` (turns.csv per turn,
decisions.parquet, receding.json, receding.md with CIs and verdicts), figure `figs/receding_onset.png` (s_on vs d_e traces of the 5
completed and 6 failed pai_RSG turns, the first two per class among late-wide / straight / over-turn by scene id with a defined slope, plus
the world-onset W_st - entry vs d_e), one README paragraph. research/decisions.md is not edited.

## Amendments (appended after the prereg commit; each says whether it came before or after a statistic was seen)

1. (After the first read; C1 / C2 / C3 numbers were seen.) C4's registered population holds no over-turn: none of the failed strict
   turns of the four PAI sets or the six held sets is classed over-turn (taxonomy or rule); the d299 over-turn zeros sit on turns outside
   the strict lists. C4 (descriptive) is extended to the d299 over-turn zeros that have a near turn (`results/progress/zeros.csv`, cls
   `over_turn`, near_turn set: PAI RSG-L18-s0 2, held RSG-L18-s0 5, RSG-L18-s1 4), with the zero-topology turn stored there (entry,
   exit, degrees; side from the sign) and its stored t_f; their approach fingerprint is tabled as class `over0_d299`. They enter no C1 /
   C2 / C3 statistic; nothing registered changes.
