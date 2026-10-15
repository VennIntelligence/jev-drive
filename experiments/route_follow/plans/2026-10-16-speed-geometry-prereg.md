# Pre-registration: SPEED-GEOMETRY reading on stored closed-loop logs (no training, no rollouts, CPU only), 2026-10-16

**Why.** research/next-round/turn-night.md section 10, the user's reading of the plan_exec visualization: the policy has learned to
turn, but the route over-guides it and it ignores what the visible road geometry implies. It does not slow for cars, it drives into the
oncoming lane, it rarely turns sharply, and it takes wide turns at speed, so it touches the edge and fails. Prior evidence: d293
(completed P3 turns at 1.33x the human speed and 1.6x the lateral acceleration), d303 (entry 7.4 vs human 5.0 m/s; APP1X slows entry
to 4.8 and strict turns go 5 -> 11), d285 / d294 (speed is copied from its own history: the ratchet, a pure input loop), d299 (7 / 26
failed strict turns in the oncoming lane at the entry), d308 (plan creep in late-wide turns). Two testable parts: **A** wide turn = speed,
**B** no braking for what it sees. Written and committed before any statistic below is computed (the data layouts and the taxonomy's
label counts were inspected; no outcome number was). Amendments are appended, never edited in place.

## Data (stored reads, read-only)

| part | set | checkpoint | source | units |
|:--|:--|:--|:--|:--|
| A primary | pai_RSG | RSG-L18-s0, PAI 281 gains 20 | RECEDING extract, Tokyo `/data/runs/route_night/receding/pai.pkl` (route, controller 10 Hz, human log, every `infer` decision's anchor + served `poses`, 10 Hz) | the 36 strict turns (`results/pai281/strict36.txt`), + the 2 d299 over-turn zeros (class only) |
| A replication | held_RSG | RSG-L18-s0 / -s1, held 460 @18 | RECEDING extract, box `$DATA_DIR/runs/route_follow/receding/held.pkl` (decisions 2 Hz) | 24 strict turns per seed, + the 9 d299 over-turn zeros (class only) |
| A3 | pai_APP1X | APP1X-L18-s0, same PAI 281 | same pai.pkl | the same 36 scenes |
| B | pai_RSG | RSG-L18-s0, PAI 281 | box rf_tax extract `$DATA_DIR/runs/route_follow/corridor/in/pai_rsg.pkl` (controller, human log, rollout.asl actor boxes at 0.5 s, route), plans `.../plans_pai_rsg.pkl` (every `infer` decision, 10 Hz), `results/taxonomy/pai281_rsg.csv` (zero class, t_ev, hit object), `results/laneoff/turns.csv` (d299 oncoming events) | all 281 rollouts |

Classes (Part A): the RECEDING turn classes (`results/receding/turns.csv`, column `cls`): completed / late_wide / straight / over_turn
(d299 zeros) / no_zero / other. Turn entry / exit / side and the zero time t_f as in the RECEDING prereg.

## Part A: wide turn = speed

**Decisions.** Approach: d_e = entry - s_e in [0, 40] m, t0 < t_f, before the ego first reaches the entry (RECEDING). Turn: s_e in
[entry, exit + 10 m], t0 < t_f. s_e = the anchor projected on the route (RECEDING).

**Curvature operator (one operator for plan and human).** Points p_0 = origin, p_1..p_8 = the plan's poses (0.5 .. 4.0 s, anchor frame).
Segment chord headings h_i = atan2(p_i - p_{i-1}) at the mid arcs m_i, plus h_0 = 0 (the anchor heading) at arc 0. Curvature samples
kappa_j = |unwrap(h_j - h_{j-1})| / (m_j - m_{j-1}), j = 1..8 (m_0 = 0); undefined if either segment is < 0.3 m. Peak over 0-2 s =
max over j <= 4; over 0-4 s = max over j <= 8. Plan implied lateral acceleration a_j = kappa_j x v_j^2, v_j = mean of the two segment
speeds around the sample (segment length / 0.5 s; j = 1: segment 1's speed); peak over 0-4 s.
- **Human, arc-matched (primary):** the human log path from the point where its route station first reaches s_e (log arc q0), sampled at
  log arc q0 + a_i where a_i are the plan's own cumulative arcs; h_0 = the log's chord heading over +-1 m at q0; the same operator and
  the same windows. Both curvatures then cover the same stretch of road, so a plan that copies the human path has ratio 1 at any speed.
- **Human, time-matched (secondary):** the log's own positions 0.5 .. 4.0 s after it passes s_e (RECEDING Log.plan_at_station), same
  operator.
- **Speeds:** v = the controller's vx at t0; v_h = the human's speed at its first passage of s_e (log 10 Hz positions, step speed,
  5-sample box). Speed ratio q = v / v_h; curvature ratio r_T = kappa_plan,T / kappa_h,T (T = 2 s, 4 s), the plan's curvature floored at
  0.001 1/m.

**Eligible decisions (A1):** v >= 2 and v_h >= 2 m/s, the 0-T plan samples defined, kappa_h,T >= 0.02 1/m (the human is turning on
that stretch: radius <= 50 m).

**A1 (primary: pai_RSG, T = 2 s).** Weighted least squares of log r_2 on log q over the eligible approach + turn decisions of the 36
strict turns, each turn's decisions weighted 1 / n_turn (every turn counts once). 95 % CI by turn-cluster bootstrap (B 10 000, seed 0,
percentile; held clusters = the log scene, both seeds together).
- supported: slope <= -0.5 and the CI upper < 0;
- refuted: slope > -0.5 and the CI lower > -0.5, or slope >= 0;
- inconclusive: otherwise, or < 8 turns with an eligible decision.
Replications with their own verdicts: T = 4 s; held_RSG (s0 + s1 pooled); the time-matched human.
**A1c (registered control, read with A1).** Human slowing at sharp corners makes q large exactly where kappa_h is large, so a plan whose
curvature does not follow the geometry at all also gives a negative A1 slope. Control: the same weighted regression of log kappa_plan on
log v + log v_h + log kappa_h (coefficient on log v; a lateral-acceleration cap gives about -2, a speed-blind plan about 0), and the A1
slope within turns (turn-demeaned x and y). A1's mechanism reading ("the plan opens the turn because it is fast") needs A1 supported **and**
the A1c log-v coefficient's CI upper < 0; A1 supported with A1c CI including 0 is read "ratio falls with speed, not attributable to the
ego's speed".

**A2.** Over all approach + turn decisions of pai_RSG with v >= 2 m/s and the plan's 0-4 s samples defined: p90 of the plan's peak
0-4 s lateral acceleration, and over those with v > v_h: p90 of kappa_h,4 (arc-matched) x v^2 (what following the human's line at the
ego's speed would need). Turn-cluster bootstrap CIs.
- supported: plan p90 <= 4.5 m/s^2 (d293's human gate) and the counterfactual p90's CI lower > 4.5;
- refuted: plan p90's CI lower > 4.5, or the counterfactual p90's CI upper <= 4.5;
- inconclusive: otherwise.
Descriptive: the human's own kappa_h,4 x v_h^2 p90; the same on held_RSG; plan a_lat against v.

**A3 (descriptive).** pai_APP1X vs pai_RSG on the scenes where both define the turn: median over turns of the turn's median r_2 and
median q; the paired difference of turn medians (APP1X - RSG) with a scene bootstrap CI. Expected by the reading: APP1X's q nearer 1 and
r_2 nearer 1.

Per class (completed / late_wide / straight / over_turn / no_zero / other), per set: turns, eligible decisions, median r_2 and q (of turn
medians), the A1 slope when >= 5 turns, plan a_lat p90. Per-turn csv: one row per turn with those numbers.

## Part B: no braking for what it sees

**Quantities per decision** (pai_RSG, 10 Hz plans):
- **plan 2 s drop** dP = v - v_plan(2 s), v = the controller speed at t0, v_plan(2 s) = mean of segment speeds 4 and 5 (1.5-2.5 s);
- **human 2 s drop at the same station** dH = v_h(t_s) - v_h(t_s + 2 s), t_s = the human's first passage of the log-path station of
  the anchor (rf_tax poly_proj on the log path), v_h the human's 10 Hz speed (5-sample box);
- **leads:** traffic = the rollout.asl actor boxes at 0.5 s, interpolated by id (rf_tax Roll.objs_at). In-path = centre within
  hw + W/2 + 0.3 m of the subject's next 5 s path (+ 20 m straight extension), near edge -1 .. 40 m ahead of the front bumper (rf_tax
  in_path rule, G = 40); lead = the nearest in-path vehicle / cyclist moving within 60 deg of the path heading. TTC = gap / (v - v_lead
  cos dpsi) when closing (else inf). Ego: its own controller path at t0; human: its log path at t_s.
- Event windows: decisions with t0 in [t_ev - 2 s, t_ev).

**B1 (collision zeros).** The 9 PAI RSG-L18-s0 zeros with a coll_* primary (taxonomy; t_ev = the taxonomy's event time). "Rear-end by
the ego" = coll_veh_ahead / coll_stationary primaries: the taxonomy has none on this set, so the set is the 9. Per scene: mean dP and
mean dH over the window. R1 = mean over scenes of dP / mean over scenes of dH, scene bootstrap CI (B 10 000).
- untestable: mean dH < 0.5 m/s (the human did not brake at those stations either);
- supported: R1 <= 0.25 and the CI upper < 1;
- refuted: R1 > 0.25 and the CI upper >= 1;
- inconclusive: otherwise.
Descriptive per scene: the executed drop v(t_ev - 2 s) - v(t_ev), the human's speed at the event station, the range and range-rate TTC
to the hit object (taxonomy hit_id) at t_ev - 2 s, the plan's minimum v_plan(2 s) in the window.

**B2 (human braked for a lead, the ego did not collide).** Matched scenes: rollouts with no coll_* primary where the human log has a
brake-for-lead moment: dH(t) >= 1.0 m/s at some t in the rollout's window, v_h(t) >= 1 m/s, with a lead (human frame, rule above) at a gap
<= 25 m. Eligible decisions: t0 < t_f (first zero, any type), v >= 1 m/s, ego lead TTC <= 3 s. Per scene mean dP and dH over its
eligible decisions; R2 = mean dP / mean dH over scenes, scene bootstrap CI.
- untestable: < 5 scenes with an eligible decision, or mean dH < 0.5 m/s;
- supported: R2 <= 0.5 and the CI upper < 1;
- refuted: R2 > 0.5 and the CI upper >= 1;
- inconclusive: otherwise.
Descriptive: R2 on the subset where the human also had a lead within 3 s TTC at t_s (both facing a lead); the own-time comparison (plan dP
at ego TTC <= 3 s vs human dH at human TTC <= 3 s, each at its own times, same scenes); the taxonomy lead / queue scenes
(low_slow_behind_lead / low_queue / low_follow_lead / low_stopped_lead) as a subset; the plan's baseline drop on lead-free decisions
(no lead within 40 m, v >= 2) of the same rollouts. Note: d285's ratchet makes dP > 0 even with no lead, which biases R1 / R2 upward
(against the claims); the baseline is reported so the response above baseline can be read.

**B3 (descriptive).** The 7 d299 oncoming-lane events (`results/laneoff/turns.csv`: set pai, point e0, oncoming, failed; t_e = the
point's time): mean dP and dH over [t_e - 2 s, t_e) and over [t_e, min(t_e + 2 s, t_f)); does the plan slow while in the oncoming lane.

## Outputs

`scripts/rf_speedgeo.py` (brake-extract on the box, read on the Mac), `results/speed_geo/` (speed_geo.md, speed_geo.json, turns.csv,
brake_scenes.csv), `figs/speed_geo.png` (A: curvature ratio against speed ratio by class; B: brake-response traces for 4 collision
scenes), a README paragraph. research/decisions.md is not edited by this reading.

## Amendment 1 (2026-10-16, after the registered read; routine choices and EXPLORATORY items, post hoc)

Routine choices made while coding (no number was seen before them):
- B "own times": the human's states are those at the stations the ego's decisions visit (the decision rows' t_s with the human's own lead
  TTC <= 3 s), not every log time; the plan side is the B2 eligible decisions.
- Figure picks: the first scene by id of each type (oncoming, crossing, side, cyclist / pedestrian) among the 9 collision zeros.
- Per-class A1 slopes use 2 000 cluster resamples (registered sets: 10 000).
- The Part A per-decision table (`decisions_partA.csv.gz`) and the Part B decision table (`brake_dec.csv.gz`) stay on the box,
  `$DATA_DIR/runs/route_follow/speed_geo/`; the taxonomy's integer actor ids are read back as asl id strings.

Exploratory, computed after the registered verdicts were seen (not verdicts):
- A1 without the straight-through (and other) turns, and the share of straight-through decisions by speed ratio: where the pooled
  slope comes from.
- B2 against the plan's own lead-free decisions (pooled), and per scene "the plan drops at least half the human's". A per-scene
  baseline subtraction was tried first and dropped: scene lead-free means are too noisy (e.g. turn slowdowns count as lead-free).
- B1 against the hit object's approach: scenes whose hit object closed at range-rate TTC <= 3 s at t_ev - 2 s, the plan's dP there;
  ego vs human speed at the event station.
