# Pre-registration: PROGRESS / LANE readings on stored rollouts (no training, no GPU, CPU only), 2026-10-15

Context: the user watched RSG-L18-s0's PAI 281 turn gallery (`results/turn_gallery/`, `plan_strict36.json`) and concluded that the policy
does turn, but (i) cuts toward far route points when the near route is missing (9f393c8a: a right turn then a left turn ahead, the car cut
straight to the left turn; 2466266e: a lane change then a 90 deg left, the car turned 180 deg), (ii) turns in the wrong lane or onto the
oncoming lane (1a69f757), (iii) over-turns into curbs / walls (382a7247, de9db5dc), (iv) ignores required lane changes (d284: swing 0.05 m),
(v) tries to overtake (fce18b62). The route reaches the model as the ROUTE-MEM polyline (-6 .. +72 m, mask; `experiments/route_mem/lib/rmem.py`)
built from AlpaSim messages that only carry the route 40-80 m ahead (`route_start_offset_m` 40); d287: the memory covers +20 m on only part of
the decisions and most held turns lie in the first-40 m gap. Two hypotheses are tested before any training is designed. Written and committed
before any statistic below is computed. Amendments are appended, never edited in place.

## Data (stored reads, read-only)

| set | checkpoint | where | per rollout |
|:--|:--|:--|:--|
| PAI 281 | RSG-L18-s0 (gains 20, VCONT 1.0, LEAD 1) | Tokyo `/data/runs/route_night/r281/rsg` (16 chunks) | driver `drive.jsonl` at 10 Hz with the served `rmem` (200 numbers), `anchor`, `poses`; rf_tax extract `/data/tmp_tax/pai_rsg.pkl` (controller 10 Hz, evaluator series, AlpaSim route rebuilt at the ego's 0.5 s poses, log path, lane class / junction flag at 0.5 s, road polygons) |
| held 460 @18 | RSG-L18-s0, RSG-L18-s1 | box `$DATA_DIR/runs/route_mem/read/RSG-L18-s{0,1}/c*/run` | `driver-logs/drive.jsonl` at 2 Hz with `rmem`; rf_tax extracts `$DATA_DIR/runs/route_follow/ledger/held_RSG-L18-s{0,1}.pkl` |

Taxonomy (primary labels, never re-run): `results/taxonomy/pai281_rsg.csv`, `results/ledger/tax/RSG-L18-s{0,1}.csv`.
Turn outcomes (never re-run): PAI `results/pai_follow/scenes.csv` (label `rsg`: the 36 strict turns = `plan_strict36.json`; `rsg@bend`),
held `$DATA_DIR/runs/route_follow/report/follow_a3/scenes.csv` (labels `RSG-L18-s{0,1}@18`, the 24 strict turns of d286).
Clocks: driver `t0`, controller `t`, evaluator series share the simulator clock (us).

Shared definitions.
- Ego = the controller trace (rear axle, 10 Hz). Route = the rf_tax rebuilt AlpaSim route polyline (MAP, 40 m), stations from its start;
  ego station / signed lateral (left +) by projection with rf_follow's local window (-30 / +60 m of the previous station).
- t_f = the first zero-score event time on the evaluator series (rf_tax's rule: earliest flagged at-fault collision / offroad / corridor).
- Memory at time t = the `rmem` of the last decision with t0 <= t (decision 0 when t precedes it, flagged). Points at offsets
  -6, -4, .., +72 m in the decision's rear-axle frame (x = 20 f1, y = 20 f2), valid = f0 > 0.5.
  cov_0_20 = share valid of the 10 offsets 0, 2, .., 18 m; cov_20_40 = share valid of 20, 22, .., 38 m.
  First valid point = the smallest valid offset >= 0 m: its offset, lateral y (m, left +) and bearing atan2(y, x) (deg).
- Clusters for every bootstrap: the scene (held: both seeds' rollouts of a scene go together). B = 10 000, percentile 95 % CI, seed 0.

## Reading A: progress / near-route coverage

Population: zero rollouts (score 0 with a flagged zero type) of PAI RSG-L18-s0 and held RSG-L18-s0 / s1, excluding taxonomy primaries
`infra`, `off_at_start`, `cor_route_end` (evaluator / route-end artefacts). A zero is a **turn zero** if its taxonomy primary is a turn class
(`cor_straight_at_turn`, `cor_turn_under`, `cor_turn_over`, `cor_wrong_exit`, `off_junc_inside`, `off_junc_outside`, `off_junc_straight`,
`off_curve_inside`, `off_curve_outside`) or if the route has a turn at the failure: rf_pai_follow's zero-topology turn (the first route turn,
strict 60 deg / 30 m else bend 45 deg / 60 m, with exit + 10 m > s_reach and entry <= s_reach + 40 m; s_reach = the last route station the ego
held within 2.5 m of the route before t_f). `cor_lane_change` zeros are tabled separately (lane change, not turn).

Failure class of a turn zero:
- taxonomy primary in a turn class -> straight-through (`cor_straight_at_turn`, `off_junc_straight`), late-wide (`cor_turn_under`,
  `off_junc_outside`, `off_curve_outside`), over-turn / inside (`cor_turn_over`, `off_junc_inside`, `off_curve_inside`), wrong exit
  (`cor_wrong_exit`);
- otherwise (collisions, `cor_drift`, `off_straight`): rf_tax's corridor heading rule applied at t_f (log heading change over log stations
  [s(t_f - 3 s), s(t_f) + 15 m] vs ego yaw change over [t_f - 3 s, t_f]: log >= 30 deg and ego < 10 deg or opposite -> straight-through;
  same sign less -> late-wide; more -> over-turn; log < 30 and ego >= 30 -> wrong exit; else other). Gate G-A1: on the corridor zeros the
  rule as coded reproduces the taxonomy's `cor_*` primary on >= 95 % (else the code is fixed before the read).
- **shortcut** (the taxonomy has no such class; flag, computed on every turn zero, independent of the class above): at t_f let s_f be the ego
  station, c_F the centroid of the route stations [s_f + 40, s_f + 80] (the far segment AlpaSim sends), psi_R the route tangent at s_f,
  psi_e the ego yaw, beta_e the bearing from the ego to c_F. The ego deviates toward the far segment when |wrap(psi_e - psi_R)| >= 20 deg and
  sign(wrap(psi_e - psi_R)) = sign(wrap(beta_e - psi_R)). The log at the same arc length (log pose at log station = the ego's driven arc length
  since its start, plus the log station of the ego's start) does not, by the same test with its own pose, bearing and the route tangent at its
  station. shortcut = ego deviates and log does not. (Far segment cut at the route end; < 10 m of far segment -> flag undefined, counted.)
- Target set T = over-turn / inside class OR shortcut flag.

Coverage bucket: covered = cov_0_20 >= 0.5 at t_f - 3 s (primary); t_f - 1 s secondary.

**Claim A (primary).** Pooled over the three sets, RR = P(T | uncovered) / P(T | covered) among turn zeros.
- supported: RR >= 2 and the CI lower bound > 1;
- unsupported: RR < 1, or the CI upper bound < 2;
- inconclusive: otherwise, or either bucket holds < 5 turn zeros.
RR's bootstrap: a replicate with P(T | covered) = 0 and P(T | uncovered) > 0 counts as RR = +inf; replicates with an empty bucket or 0 / 0
are dropped (their number reported). Per-set RRs (PAI; held s0 + s1) are secondary.

Reported alongside (no verdict): counts class x bucket (both times); cov_0_20 / cov_20_40 / first valid offset, lateral, bearing medians per
class; the ego's driven distance since its start at t_f - 3 s (coverage is mechanically a function of it: the first message starts ~42 m
ahead, so 0-20 m reaches >= 50 % only after ~32 m of driving); **completed strict turns** (PAI `rsg` completed, held 24 x 2 completed):
the same coverage statistics at the time the ego first crosses the entry station and 3 s before it; P(uncovered) completed vs turn zeros
(difference, CI). Per turn zero: whether the far segment lies on a different road leg than the near route at t_f - 3 s
(|route heading at s + 60 m - at s| >= 45 deg, chords +-5 m; s = the ego station), whether the route over [s, s + 80] holds two turns of
opposite sign (rf_pai_follow `all_turns` with 30 deg / 30 m) (the "right turn then left turn" pattern), and whether a turn's entry lies in
[s, s + 40] (the maneuver sits in the gap the messages never show).

## Reading B: lane / lateral offset

Population: the 36 strict PAI turns (`rsg`, = plan_strict36.json) and the held strict turns of RSG-L18-s0 / s1 (24 each). Entry = the
rf_pai_follow / rf_follow entry station (10 % of the turn's heading change; d293 / rf_shape / rf_p4 use the same). Completed = the stored
outcome `completed`; failed = any other outcome (event / off / short; failed without `short` secondary).
Measurement points: the first controller sample whose route station >= entry - 20 m, and >= entry, before t_f. Undefined when the ego
starts past the point (counted) or never reaches it before t_f (counted, "not reached").
Measures: lat_route = signed lateral offset to the route polyline (left +); lat_log = signed lateral offset to the log rig path;
heading error = wrap(ego yaw - route tangent); speed = controller |v|; lat_mem = signed lateral offset to the served memory polyline
(valid points of the decision's `rmem`, in its rear-axle frame, ego carried to the sample time) where >= 2 valid points span the ego's
projection, else undefined (secondary: tells how often the model even has the near route).
Out-of-lane = |lat_route| > 1.75 m.

Known structure fixed before the read: held turns begin 3-19 m after the rollout start (rf_ledger), so entry - 20 m is undefined on held;
on PAI 12 of the 36 entries lie < 20 m from the start and 3 of the 5 completed turns have entry - 20 m defined.

**Claim B (primary).** PAI strict turns at entry - 20 m: D_med = median |lat_route| failed - completed; D_out = out-of-lane share failed -
completed; scene bootstrap.
- supported: (D_med >= 0.5 m and its CI lower > 0) or D_out's CI lower > 0;
- unsupported: D_med's CI upper < 0.5 m and D_out's CI upper <= 0.10;
- inconclusive: otherwise, or either group < 3 turns with the point defined.
Secondary (no verdict): the same at the entry, PAI and pooled PAI + held; the `rsg@bend` PAI turns (45 deg / 60 m) at entry - 20 m;
lat_log, heading error, speed per group; share of turns whose ego stood on an oncoming lane (rf_tax lane class 2 at any 0.5 s step with
route station in [entry - 20, exit + 10] before t_f) failed vs completed.

**Lane changes.** Every scene of PAI 281 (RSG-L18-s0) and held 460 (both seeds), route stations from the ego's start to the log's end - 40 m:
LC site = the first station s where the route point at s + 40 m lies >= 2 m to the side of the route tangent at s and the route heading
changes < 15 deg over [s, s + 40] (d284's proxy with the brief's 2 m; d284 used 2.5 m); stricter variant: heading change < 5 deg (parallel
tangents; d284 noted that gentle bends pass the 15 deg proxy). For each LC site: shift = the ego's lateral displacement, in the frame of the
route tangent at s, from its position at station s, toward the side of the route's shift; "shifted" = max shift >= 2 m over route stations
[s, s + 60] before t_f; reached = the ego reached station s before t_f. Table per scene and the shares (shifted / reached).

## What the outcomes mean for training design

- A supported -> the cut-across / over-turn zeros are a missing-input failure: route input ordered and progress-anchored with a 40 m near
  horizon that is never empty (back-fill the first-40 m gap from the first message, or train with gap-aware rows that carry the gap as it
  is served), before any new turn objective.
- B supported -> the lane error exists before the turn: a near-field lateral-offset hinge to the carried route (0.5 m dead zone) on
  approach rows, with straight-road negatives so the hinge does not buy offroad cuts (d286's lesson).
- Both unsupported -> the input and the approach lane are not where the failure lives; it is in the turn magnitude / speed (d293, d296,
  d298), not in what the route input shows.
- A or B inconclusive -> no design decision rests on that reading; the counts are reported for the user.

## Outputs

`scripts/rf_progress.py` (reading A, also the `rmem` extractor), `scripts/rf_laneoff.py` (reading B), results `results/progress/` and
`results/laneoff/` (json + md with CIs), figures `figs/progress_bev.png`, `figs/laneoff_bev.png` (4-6 scenes each incl. 9f393c8a, 2466266e),
one paragraph each in the README. research/decisions.md is not edited.

## Amendments (appended after the prereg commit 77810b16; each says whether it came before or after a statistic was seen)

1. (Before the read, while coding.) The first valid memory point at offset 0 m sits beside the rear axle, so its bearing is ~ +-90 deg by
   construction; the first valid point at >= 10 m (offset, lateral, bearing) is reported next to it. Descriptive only.
2. (After the first read; the registered slot-based numbers are kept as the primary verdict.) The BEV of de9db5dc showed that the slot index
   of the served memory is not the distance along the route when the ego stands before the start of the memory and that start lies past a
   turn: `rmem.window` extrapolates the ego's arc position along the memory's first segment, so points ~40-50 m ahead along the route are
   served in the 14-20 m slots (de9db5dc: slot cov 0-20 m 0.3, geometric 0.0). Added a geometric coverage: the valid memory points projected
   on the rebuilt route span [lo, hi] (+-1 m); share of the stations s_e + 0, 2, .., 18 (20, .., 38) inside it. Claim A is reported under
   both; the verdict of record stays the registered slot version. Gate G-A2 added at the same time: served memory points lie on the rebuilt
   route (median distance 0.00 m), so the memory and the route are the same object and only the slot labelling differs.
3. (After the first lane-change read.) The registered geometric LC proxy (route at +40 m >= 2 m to the side, heading change < 15 / 5 deg)
   fires on 116 / 98 PAI rollouts and is dominated by road geometry (S-bends, junction offsets, curves leaving the tangent): the "log
   shifted" column is ~90 % by construction (the route is the log on lane centres) and does not validate anything. Added a map version: every
   4 m the lane under the route and its successor chains (rf_pai_follow lane_at / chains, PAI lanes from the rf_pai_follow extract, held lanes
   from `rf_laneoff.py lanes` on the box); a site when the route point 40 m ahead is >= 2 m from every chain and on another lane; validated =
   that lane is adjacent (gap <= 6 m) and the human's log itself moved 2-6 m to that side (> 6 m = the log turned there; added after
   the first map read showed turn-lane picks with 14-49 m log shifts among the sites). Ego outcomes: shifted >= 2 m (registered), in the
   route's lane after (|lat_route| <= 1.75 m in [s + 30, s + 60]), stayed in its own lane chain at s + 40. Both versions are reported; the
   registered proxy's shares are not read as "lane changes done". A first map version that used the route's lane-id steps to a neighbour
   (13 PAI sites, the log moved >= 2 m at 2 of 5 reached) was dropped before the final read as lane-assignment noise.
4. (After the read; exploratory, no verdict.) Reading B's offsets signed toward the turn's inside (+) vs away from it (-), since the user's
   observation (ii) is a wrong-lane / wrong-side observation and |offset| cannot tell the side.
