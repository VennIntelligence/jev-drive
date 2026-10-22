# ROUTE-CHECK: is the route branch worth a base retrain for PAI? (ROUND9, 2026-10-22)

Diagnosis, CPU only: stored PAI 281 ACC + GAP reads plus open-loop CPU forwards on cached rows. No training, no closed-loop reads,
no GPU. Question (round9 draft section 2): do the remaining straight / high-speed losses of R6DAG come from the route input (command flips,
route coverage too short for the plan at speed, mis-anchored memory)? If they do, would a speed-scaled look-ahead window, anchored memory or
a heading command rule recover points, and would that justify a base retrain?

**Verdict.** No base retrain for the route input on this evidence.
- The demonstrable gain is about 1 scene point per seed, from the heading command rule. That rule needs no new base: R7J already trained
  with it in the R6DAG recipe.
- A speed-scaled window and anchored memory explain 0 of the 14 / 10 target events.
- The largest route-related block is different: in 4-5 scenes per seed the AlpaSim route runs in the lane next to the one the human drove.
  None of the three changes touches this. It belongs to the DAgger lateral teacher, which currently labels the route's lane.

Code [scripts/rtc.py](../../scripts/rtc.py) (`extract`, `events`, `band`, `report`) and [rtc.sh](../../scripts/rtc.sh). Box outputs are in
`$DATA_DIR/runs/round9/route_check/`. Generated tables are in [tables.md](tables.md), with csv files alongside.

## Inputs and definitions

- Reads (PAI 281, ACC + GAP, Harmonizer off): R6DAG-L18 s0 / s1 (the subject), NTV0-L18-s0, and R7J-L18 s0 / s1 (served with
  `JEV_CMD=heading`). Every inference record of `driver/drive.jsonl` is used: 277 869 decisions, about 55.5 k per read. Facts, classes and
  event times come from `pai_fix` (`primary`, `t_ev_s`).
- Every model feeds the projected route memory: 40 slots at -6 .. +72 m of route arc from the ego's projection, 2 m apart. The command is
  the shipped rule (|y| > 2 m at the ~42 m point), except R7J, which uses the heading rule (`rmem.cmd_heading`: 20 deg within 40 m).
  `cmd_h` is the heading rule recomputed offline on the same stored memory.
- Coverage (`cov`) is the largest valid slot ahead, at most 72 m. `arc4` is the served plan's 4 s arc length. "Beyond" means arc4 > cov.
  `latcov` is the plan's lateral offset from the route at the last covered arc (+ = plan left of route).
- `y_rt0` is the route's lateral offset at the ego's own station. `d_log` is the ego's signed offset from the human log (+ left).
  `rh = y_rt0 + d_log` is therefore the route's offset from the human's path. Map columns (`m_*`) come from round6 `rmc.state_metrics` on
  the NuRec map: ego lane offset, the route's lane offset at 20 / 40 m, and the lane bend.
- Target events: R6DAG primaries `cor_drift`, `off_straight`, `off_junc_straight`, `cor_lane_change` and `cor_straight_at_turn`.
  That is 14 scenes for s0 and 10 for s1, 15 distinct. "Leave" is the first decision in the 15 s before the event at which the ego is
  more than 1 m from the human log. "Lead-up" is the 5 s before leave.

## A. The target events

Per event (full table in [events.csv](events.csv); other reads over the same human-log stretch in
[events_contrast.csv](events_contrast.csv)). Points are scene points, one per zero.

| mechanism (rule in `rtc.ev_summary`, checked by hand on every time series) | s0 | s1 | scenes |
|---|--:|--:|---|
| Route in the lane next to the human's (rh 3.1-3.8 m at the start). The ego follows the route and ends 4.0-4.1 m from the human. | 3 | 3 | 26004ca8 (R, 22 m/s), 4cb05d65 (L, 38.5 m/s), 7dc59d0d (L, 28 m/s) |
| Route in the next lane, the human merges into it, the ego stays in its own lane | 1 | 1 | 0e05cb0c (32 m/s) |
| Human leaves the route's lane, the ego stays on the route and slows (21 -> 5 m/s) | 1 | 0 | 404cc92f |
| Ego on the route and within 1.5 m of the human, offroad at a junction / lane edge / early lead avoidance | 5 | 4 | 4f7cb072, 7750265d, a5599793, a68ad579, e1c6ae9f; s1: b661da67 instead of 7750265d / e1c6ae9f |
| Route turn not taken | 2 | 2 | 302c5c99 (U-turn: the route starts 10 m to the left, on the opposite carriageway), e5e47fd2 (a 117 deg left turn 5 s into the episode, inside the start gap) |
| Policy drift with the route fine and the command straight | 1 | 0 | 596af0c7 (gentle right bend, 19 m/s, takes the right lane) |
| Command flip (old rule R, heading rule S), ego moves right of the human | 1 | 0 | baabc57d (9 m/s, junction approach) |

Findings:

1. **Speed.** At leave the ego drives at 22-38.5 m/s in the four lane-disagreement events and at 2.4-21 m/s in the rest. Every
   high-speed event in the set is a lane-disagreement event.
2. **Command.** In the lane-disagreement events the old command pointed to the route's side on 80-95 % of lead-up decisions (0e05cb0c
   100 %). The heading rule gives STRAIGHT on every one of them. Flips in the lead-up: 0-1. Elsewhere the old rule fires L/R on 11-12 % of
   straight decisions at >= 15 m/s (section B).
3. **Coverage.** The plan runs past the 72 m route on 86-100 % of lead-up decisions in every event at >= 20 m/s. However, the decisive
   geometry is never beyond 72 m. The route's lateral offset (3.1-3.8 m, a whole lane) is visible at the ego's own station (`y_rt0`) and
   at 20 m.
4. **Anchoring.** No event shows mis-anchoring. In the disagreement events `m_d_rt20` / `m_d_rt40` put the route on the centre line of an
   adjacent lane (offset 3.5-3.6 m), not shifted along the track. The route is "wrong" because of the lane it chose, not because of where
   the memory is anchored.
5. **Direction relative to the route.** In 3 of the 4 high-speed events the ego moves toward the route (and away from the human). In
   0e05cb0c it stays put while the human moves into the route's lane.
6. **What decides the zero.** The drift zeros fire at about 4 m from the human path: the target events end at |d_log| 3.96-4.14 m, while
   NTV0 passes 4cb05d65 with a 3.8 m peak after following the same route. A route one lane over (3.1-3.6 m) plus any overshoot crosses
   the line. On 4cb05d65 R6DAG and R7J overshoot the route's lane centre by 1.1-1.5 m at 38.5 m/s; NTV0 stops within 0.5 m of it.
7. **Contrast reads.**
   - 7dc59d0d: R7J (heading rule) stays in the human's lane and passes with both seeds. NTV0 and R6DAG s0 / s1, all fed LEFT for about
     5 s, follow the route and fail. This is the only scene where the command rule plausibly decides the outcome.
   - 4cb05d65 and 26004ca8: R7J follows the route with STRAIGHT and fails, so the route's geometry is enough to pull the ego across.
   - 404cc92f, 7750265d, baabc57d, e1c6ae9f, b661da67: the other R6DAG seed, NTV0 and R7J split without a route-input pattern. These look
     like seed-level lateral noise.

### Where the lane disagreement comes from

AlpaSim's map route generator and the DAgger teacher (round6 `dag.py`, same rule) project each human pose onto a lane among those within
4 m and pi / 8, and **keep the previous lane while it stays a candidate**. A human driving 3.1-3.8 m away, in the next lane, keeps the old
lane as a candidate. On the 281 test human log the route sits >= 1.75 m from the human's lane centre on 10 % of states (decision 432).

At decisions on straight roads the route is >= 2.5 m to the ego's side on 9.5 % of decisions at 20-25 m/s and 14.5 % at >= 25 m/s,
against 0.7-5 % below 20 m/s. These are multi-lane roads at highway speed. 12 R6DAG scenes have this on >= 20 % of their non-turn
decisions at >= 15 m/s; 6 / 4 of them are zeros (s0 / s1), and NTV0 and R7J give the same picture (5-6 zeros). The DAgger lateral teacher
labels the route lane's centre plus the human's offset clipped to +-0.5 m. In these states it therefore teaches the ego to join the
route's lane, 3.5 m from where the human drove. This follows from the teacher's definition; it has not been checked on the DAgger rows
themselves.

## B. Population, all decisions (R6DAG; NTV0 / R7J alike, [pop_speed.csv](pop_speed.csv))

| speed (m/s) | n (s0) | plan arc4 p50 / p90 (m) | plan beyond the route | beyond by > 10 m | median overrun (m) | no slot at the ego |
|---|--:|---|--:|--:|--:|--:|
| 0.5-5 | 9 964 | 11.6 / 25.0 | 0.5 % | 0.3 % | 0 | 62 % |
| 5-10 | 16 034 | 29.7 / 40.8 | 0.0 % | 0.0 % | 0 | 36 % |
| 10-15 | 12 056 | 46.4 / 59.3 | 0.2 % | 0.1 % | 0 | 30 % |
| 15-20 | 4 923 | 68.7 / 78.8 | 32 % | 3 % | 0 | 14 % |
| 20-25 | 4 019 | 88.1 / 97.1 | 94 % | 77 % | 16 | 11 % |
| 25-45 | 4 573 | 118.7 / 147.2 | 99.8 % | 99.5 % | 47 | 9 % |

s1 is within 1-2 points of s0 in every cell.

- **The plan runs past the route.** Above 20 m/s nearly every plan extends past the route memory, by a median 16 m at 20-25 m/s and
  47-50 m above 25 m/s. Coverage itself is almost always the full 72 m when moving.
- **But the plan does not drift from the route at speed.** With the route on the ego's lane (|y_rt0| < 1 m, straight), the plan's
  offset from the route at 72 m exceeds 1 m on 2.2-5.1 % of decisions at >= 15 m/s, against 9.7-11 % at 5-15 m/s. The p90 is
  0.58-0.76 m against 0.98-1.1 m ([pop_lat.csv](pop_lat.csv)). High-speed plans follow the covered route more tightly than slow ones.
- **No slot at the ego.** 27 % of all decisions have no route slot at the ego, 62 % below 5 m/s. This is the 40 m start gap of AlpaSim's
  messages (decision 287): early in an episode the memory starts about 40 m ahead. A speed-scaled window does not change this.
- **Command on straight segments** (|route heading change within 40 m| < 3 deg, [pop_cmd.csv](pop_cmd.csv)):

  | segment | old rule L / R (fed to R6DAG) | heading rule |
  |---|--:|--:|
  | v >= 15 m/s | 6.6 / 5.2 % (s0), 5.5 / 5.4 % (s1), 1.7 command changes per minute | 0 / 0 |
  | 10-15 m/s | 1.9 / 8.2 % | 0 / 0 |
  | gentle 3-15 deg, v >= 15 m/s | 15.2 / 10.3 % | 0 / 0 |

  R7J's fed command changes 0.1 times per minute on the same segments.

## C. Route dependence by distance band (CPU forwards)

PENDING

## D. Points plausibly recoverable on PAI 281 (scene points per seed, s0 / s1)

| lever | s0 | s1 | evidence | needs a base retrain? |
|---|--:|--:|---|---|
| (i) heading command rule | ~1 (upper bound 3) | ~1 (upper bound 3) | 7dc59d0d: R7J passes with both seeds, every old-rule read fails. 26004ca8 / 4cb05d65 had the same wrong command, but R7J still fails them, so they are not credited. Cost: R7J's s1 offroad +9 (bundled arm, not separable). | No. It is a row and serve input change that R7J already trained in the R6DAG recipe, so it can ride on the next DAgger iteration. |
| (ii) speed-scaled look-ahead, fixed point count | 0 | 0 | No target event's decisive geometry lies beyond 72 m. On covered straight road the plan tracks the route better at speed. The plan overruns the route above 20 m/s only by construction. | Yes (input definition), for no shown gain. |
| (iii) anchored memory | 0 (at most 1: the U-turn 302c5c99) | 0 (same) | Route offsets in the events are lane choice, not along-track anchoring. Decision 342: re-anchoring moves plans by <= 0.1 m (median); U-turns have never been solved. | Yes, for no shown gain. |
| not touched by (i)-(iii): the route lies in the lane next to the human's | 4-5 | 4 | Section A: the ego follows the route one lane over and crosses the ~4 m line, or the reverse (0e05cb0c). | No. This is a DAgger teacher matter. |

Recommendation:

1. Do not spend a base retrain on the route input definition (window, anchoring). Together with the heading rule they are worth about
   1 point per seed (~0.004 of the mean) on these classes.
2. Fold the heading command rule into the next DAgger iteration's rows and serve. It is cheap, and R7J showed it trains in this recipe.
   Judge it by 7dc59d0d and the cor_drift / lane-change count.
3. The larger route-related block (4-5 points per seed, all at 22-38 m/s) is the route's lane differing from the human's lane. Address it
   in the lateral teacher: where the route's lane is a parallel neighbour of the student's current lane with no turn or fork within the
   window, label the current lane instead of joining the route's lane. This rests on the teacher's definition and has not been verified on
   rows (open). It cannot fix 0e05cb0c, where following the route would have been right.
4. Lateral overshoot at very high speed (R6DAG / R7J 1.1-1.5 m past the target lane centre at 38.5 m/s, NTV0 < 0.5 m) is a separate,
   smaller effect, seen on 4cb05d65.

## Caveats

- One read per checkpoint. Zeros near the 4 m line are fragile, and seed-level noise is visible (7750265d, b661da67, baabc57d).
- The mechanism rules are my own thresholds (rh >= 2.5 m, |d_log| > 1 m at leave, |d_log| <= 1.5 m for "on the human path"). Each
  event was checked by hand on its 0.5-1 s time series; the hand reading agrees with the rule labels in the table.
- R7J bundles three changes (decision 441). Crediting 7dc59d0d to the command rule is a judgement, not an ablation.
- The `d_log` and lane columns use round6 `rmc`'s lane chain. They are reliable on straight multi-lane roads and noisy inside junctions
  (`m_dth40` jumps).
