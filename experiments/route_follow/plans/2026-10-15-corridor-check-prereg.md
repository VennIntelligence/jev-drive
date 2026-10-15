# Pre-registration: ROUTE-LANE CORRIDOR labels and the offline corridor check (no training, no GPU, CPU only), 2026-10-15

**Why.** research/next-round/turn-night.md section 6: every S hinge so far (RBS, RBSG2 d286, arm c d302, APP1X d303) bounds the plan only
with a drivable-area grid in which the whole junction polygon and the oncoming lane are drivable, so a pull toward the route cuts the
inside corner with no boundary term firing. The user's direction: pull toward the route, but with a lane-line / boundary concept and no
overshoot. Proposed loss: the plan's 1-4 s car footprint outside a corridor = the route's own lane sequence (+ 0.3 m), hard outside it
into the oncoming lane or off the road, soft into a same-direction neighbour lane, zero gradient inside. Before any training: on stored
rollouts, does the policy's own served plan already leave that corridor before its turn zeros (the hinge would have had a gradient), while
completed turns stay inside (the hinge would not disturb them)? Written and committed before any label or statistic below is computed.
Amendments are appended, never edited in place.

## Data (stored reads, read-only)

| set | checkpoint | rollouts | served plans | rf_tax extract / classes | strict turns (rf_follow) |
|:--|:--|:--|:--|:--|:--|
| pai | RSG-L18-s0, PAI 281 (gains 20) | Tokyo `/data/runs/route_night/r281/rsg` | `runs/*/driver/drive.jsonl` records with `infer` (10 Hz) | `/data/tmp_tax/pai_rsg.pkl`, `results/taxonomy/pai281_rsg.csv` | `results/pai_follow/scenes.csv` label `rsg` (36) |
| held_RSG_s0 / s1 | RSG-L18-s{0,1}, held 460 @18 | box `$DATA_DIR/runs/route_mem/read/RSG-L18-s{0,1}/c*/run` | `driver-logs/drive.jsonl` `drive` records (2 Hz) | `runs/route_follow/ledger/held_RSG-L18-s{0,1}.pkl`, `results/ledger/tax/RSG-L18-s{0,1}.csv` | `report/follow_a3` `RSG-L18-s{0,1}@18` |
| held_RBSG2_s0 | RBSG2-L18-s0 | `read/RBSG2-L18-s0` | same | new extract `runs/route_follow/corridor/tax/held_RBSG2-L18-s0.pkl` + `rf_tax.py report` (no ref) | `report/follow_r2` `RBSG2-L18-s0@18` |
| held_RV2c_s0 / s1 | RV2c-L18-s{0,1} | `read/RV2c-L18-s{0,1}` | same | new extracts, same | `route_follow/v2/report/follow` `RV2c-L18-s{0,1}@18` |
| held_APP1X_s0 | APP1X-L18-s0 (EXPLORATORY, d303) | `read/APP1X-L18-s0` | same | new extract, same | `route_follow/approach/report/follow` `APP1X-L18-s0@18` |

Maps: held = the AlpaSim nuPlan map of each scene (trajdata VectorMap via the run's `generated-user-config-0.yaml`, the map rf_tax /
rf_cfpool read); PAI = the AlpaSim PAI (USDZ) map of each clip (the lane graph rf_laneoff / rf_pai_follow used). Extracted once per
scene by `rf_corridor.py lanes`: every lane within 150 m of the logged path (centre, left / right edge, the evaluator's lane polygon
`eval.scorers.offroad._get_lane_polygon`, next / prev / adjacent ids) and the road-area polygons (nuPlan; PAI maps have none).
Feasibility probe before this file (no statistic): nuPlan lanes all carry edges and adjacency, road areas exist; PAI lanes all carry
edges and adjacency, no road areas, no ROAD_AREA layer at all.

## Labels (`rf_corridor.py build`)

Route = rf_tax's rebuilt AlpaSim route of the rollout (MAP generator, 40 m; the route the policy was given), stations from its start.
Ego station s_e(t) = rf_follow's local-window projection of the controller trace (rf_progress `Ego`).

1. **Route-lane chain** (per rollout; the matching rules of d283's R-A labeler `je_ruler.LocalMap.route`, applied to the route
   polyline instead of the log): route samples every 1 m; candidate lanes = lane polygons within 1.0 m with centre-line heading within
   60 deg of the route tangent, score = heading error (rad) + 0.5 x distance to the polygon; best score wins. **Ambiguous sample**: a
   second candidate whose centre-line distance is within 0.5 m of the best's, both headings within 30 deg, and whose centre point lies
   > 1.0 m from the best's (two distinct lanes, not overlapping connectors at a fork); then the lane whose centre point is closer to the
   human log path wins and the sample is flagged. A-B-A flicker with B < 2 samples -> A; one missing connector between consecutive lanes
   is filled from the successor graph; transitions are successor / change (adjacent or <= 6 m from the previous centre) / break.
2. **Pieces**: each chain lane clipped to the arc range the route uses (whole lane between successors; cut at a change sample), with
   its station range. At a lane change the corridor is the target lane; for a transition zone the source piece is extended 10 m past
   the cut and the target piece starts 30 m before it (both lanes are corridor in [cut - 30, cut + 10]). Piece polygon = the lane's
   left / right edges between the arc limits (the evaluator polygon when the piece is the whole lane).
3. **Corridor at a decision** (time t0, anchor = the served decision's rear-axle pose): union of the pieces whose station range meets
   [s_e - 15, s_e + 100] m, buffered by 0.3 m. Found = the chain covers >= 90 % of the 1 m route samples in [s_e, s_e + 40] (clipped at
   the route end). Ambiguous = any flagged sample in the window.
4. **Classes** of a world point p outside the corridor: reference heading = the route-lane heading at the nearest corridor centre
   sample; covering lanes = lane polygons (evaluator's) containing p, heading = their centre-line heading at p; d = |heading - reference|.
   Code 0 = in corridor; 1 = **adjacent same-direction** (some covering lane d < 60 deg); 2 = **oncoming** (else some d > 120 deg);
   4 = **other drivable** (else covered by a crossing lane, or inside a road area); 3 = **off-road** (no lane polygon, no road area).
   Hard = {2, 3}; soft = {1, 4}. (rf_tax's lane-class precedence: same > oncoming > crossing.)
5. **Stored per decision**: the 48 x 48 grid at 0.5 m in the anchor's rear-axle frame (d283 R-B frame: x 0.25 .. 23.75 m, y -11.75 ..
   11.75 m, left +) with codes 0-4; the corridor polygon in the same frame (WKB, for a longer-range hinge); found / ambiguous flags.
   Coverage table = share of decisions found (and ambiguous) per source, all decisions and the strict-turn windows.

## Check (`rf_corridor.py check`)

Footprint of a pose = the rig box from rf_tax `dims` (rear axle origin; 6 points along each side + front and rear centre). Plan = the
served `poses` of the decision (8 poses at 0.5 s in the anchor frame); its 1-4 s poses (1.0, 1.5, .., 4.0 s) are classed against the
corridor of that decision. Plan **hard exit** = any footprint point of a 1-4 s pose in code 2 or 3; soft exit = in 1 or 4 (not hard);
also oncoming-only and off-road-only.

- t_f = rf_tax's first zero event time (rf_progress `Ego.t_fail`). Probe decisions at t_f - 1 s, - 2 s, - 3 s = the last decision with
  t0 <= t_f - D and t0 > t_f - D - 1 s (else undefined). A zero **hits** when the plan is a hard exit at any defined probe.
- Turn zeros and their class: rf_progress reading A (`zero_job`, unchanged): drop `infra`, `off_at_start`, `cor_route_end`; turn zero =
  taxonomy turn class or rf_pai_follow zero-topology turn; class over_turn (cor_turn_over, off_junc_inside, off_curve_inside = the inside
  cuts; rf_tax has no separate "early-inside" label), late_wide, straight, wrong_exit, else the corridor heading rule.
  Ego touches the oncoming lane = rf_tax lane class 2 at any 0.5 s step before t_f with station in [entry - 20, exit + 10] of the
  zero-topology turn (no such turn: in [t_f - 5 s, t_f]) (rf_laneoff's rule).
- **C1** (registered line): population = turn zeros of class over_turn, plus turn zeros with zero type corridor whose ego touches the
  oncoming lane; share that hits >= 0.70. Verdict on the pooled population of all seven sets (scene clusters; held scenes shared across
  checkpoints are one cluster); per set and per class reported. Zeros with no defined probe are counted and left out.
- **C2** (registered line): completed strict turns (rf_follow outcome `completed`), decisions with s_e in [entry - 20, exit + 10]
  and t0 before the ego first crosses exit + 10 m; share of decisions whose plan is a hard exit <= 0.10, pooled over the seven sets
  (verdict) and per set.
- **C3** (descriptive): the same hit share for late_wide, straight (and wrong_exit, other) turn zeros, and for all decisions of failed
  strict turns. **Label sanity G-H**: the human log's own 1-4 s footprint after t0 (log poses at t0 + 1.0 .. 4.0 s, classed against the
  corridor around the human's own station at t0), decisions in the strict-turn windows (human station in [entry - 20, exit + 10]), per
  source (pai, held; held deduplicated by scene x decision time): hard exit share <= 0.05 (the human is inside >= 95 %), and the human
  rear-axle point inside the corridor reported. **If G-H fails for a source, its labels are wrong and C1 / C2 on that source are void**
  (reported, not counted in the pooled verdict).
- CIs: scene-clustered bootstrap (`jevdrive.stats.bootstrap(groups=)`, ratio of sums, B = 10 000, seed 0). A line "holds" on the point
  estimate; "robust" when the 95 % CI lies on the same side of the line. Decision rule from turn-night section 6: train the corridor
  hinge only if C1 and C2 both hold.
- Secondary (not registered): every decision in [t_f - 3 s, t_f - 1 s] (PAI is 10 Hz); the ego's own footprint at the probe in code 2;
  hit at each probe separately; soft exits; first hard-exit pose time.

## Outputs

`results/corridor/`: `coverage.md` / `.json` (label coverage per source), `check.md` / `check.json` (C1 / C2 / C3 / G-H per set and
class), `zeros.csv`, `turns.csv`; `figs/corridor_bev.png` (4-6 scenes incl. PAI 382a7247, de9db5dc, 1a69f757 and one completed turn:
corridor classes at t_f - 2 s, human path, driven path, plan at t_f - 2 s). Labels (per decision grids and polygons) stay on the box:
`$DATA_DIR/runs/route_follow/corridor/labels/`. Code `scripts/rf_corridor.py` (lanes | plans | build | check), `scripts/rf_corridor.sh`.

## Amendments (appended after the prereg commit 7e1eac00; each says whether it came before or after a statistic was seen)

1. (Before any statistic; from a smoke BEV of 7 held RSG-L18-s0 rollouts, labels only.) The rebuilt AlpaSim route starts a few metres
   ahead of the log's first pose, so the first decisions stood behind the chain (the human's rear axle classed "adjacent" at t0). The
   chain is now matched on the route prefixed with the human log before the route's start (log points with log station < the
   projection of the route's first point); corridor windows use stations on this prefixed route (`s_x`); the turn windows keep
   rf_follow's route stations (`s_e`, `s_h`). Smoke result after the fix: the human axle is inside at every early decision of the 7.
2. (Before any statistic.) Two descriptive columns per decision: `plan_len` (served plan arc to 4 s) and `cor_left` (route-chain end
   minus s_x). A plan longer than `cor_left` runs past the end of the labelled route; how often that happens in the C2 windows is
   reported, the registered measure is unchanged.
