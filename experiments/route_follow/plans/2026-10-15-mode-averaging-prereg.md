# Pre-registration: is the plan head's turn deficit MODE AVERAGING? Two readings, 2026-10-15 night

**Why.** Decision 306: the served plan starts the turn in the right place but asks for a fraction of the human's heading (straight-through
class 5.6 vs 74 deg over 4 s at 0-10 m before the entry; late-wide 5.3 vs 27.6 deg at 10-20 m; inside the turn it follows half of the
remaining heading, slope 0.49 vs 0.93). Hypothesis: a regression head at a junction with several exits outputs the average of the modes
(straight plus a slight bend) when the conditioning (command / route memory) is weak; inside the turn it averages "keep turning" and
"straighten". Related: d246 (FLOW1: flow head vs its regression twin, > 45 deg off-road -4.98 pp on navtest, never read in closed loop),
d301 (x1.25 lateral gain: 5 -> 6 deg), d284 (the 3-way command's effect sits at 3-4 s), d240 / d093 (navtrain: many tokens with alternative
exits). Written and committed before any statistic below is computed. Amendments are appended, never edited in place.

## Reading 1 (CPU, stored reads): deficit against ambiguity

**Data** (all read-only; the RECEDING extracts of d306):
- `decisions.parquet` of `results/receding/` (box `runs/route_follow/receding/`, Mac copy `tmp/receding/`): per decision the served plan's
  4 s heading change toward the turn `dpsi`, the human log's `h_dpsi` at the same route station, `d_e`, the approach flag, class.
- Turns and routes: the RECEDING extracts `pai.pkl` / `held.pkl` (route, entry, exit, side, rollout id).
- Served command and route memory per decision: rf_progress `rmem` extracts (`rmem_pai_rsg.pkl`, `rmem_held_RSG-L18-s{0,1}.pkl`: drive.jsonl
  `cmd` = argmax of NAVSIM [L, S, R, unknown], `rmem` 40 slots at -6 .. +72 m, validity + rig-frame xy), joined on (rollout id, t0).
- Lanes: rf_corridor / rf_laneoff lane extracts (PAI `lanes_pai.pkl` with `nx`; held `lanes_held.pkl`).

**Populations.** Primary: PAI 281 RSG-L18-s0, the strict 36 turns + the 2 d299 over-turn zeros of that set (RECEDING Amendment 1).
Replication: held 460 @18 RSG-L18-s0 and -s1, 24 strict turns each + their 5 / 4 over-turn zeros (clusters: the log id, both seeds
together). Pooled: both. The 11 over-turn zeros are tabled as their own class as well.

**Window.** Approach decisions (RECEDING definition: before the ego first reaches the entry, t0 < t_f) with d_e in [0, 30] m.

**Deficit.** Per decision with h_dpsi >= 15 deg (the human asks for a material turn within 4 s): delta = 1 - dpsi / h_dpsi, winsorised to
[-1, 2]; decisions with h_dpsi < 15 deg have no deficit (counted). Per turn: D = median delta over its window decisions (turns with no such
decision are dropped, counted).

**Exits (map).** For each turn: s_a = entry - 5 m on the route; the route's lane at s_a from rf_corridor's SceneMap.match (PAI) or, where
match gives no lane (held maps have no adjacency / polygons in the slim extract), the nearest lane centre within 2.5 m whose heading is
within 30 deg of the route's. The approach group = every lane with a centre point within 7 m of the route point at s_a and heading within
30 deg of the route's (the same-direction lanes at the junction mouth). From each group lane, starting at the projection of the route
point, every successor path (`nx`, depth <= 8, <= 256 paths) is walked to arc L = (exit - s_a) + 15 m (or its end); its heading change
at that arc relative to the route heading at s_a is one outgoing direction. Directions are clustered single-linkage at 20 deg; n_x = the
number of clusters (1 = a curve with no alternative). Secondary (descriptive): n_x from the matched lane alone.
- G-map: the route's own heading change over the same arc lies within 20 deg of one cluster on >= 80 % of the turns with a matched lane, and
  a lane is matched on >= 90 % of the turns. Failing it, M1 reads inconclusive and only the table is reported.

**Memory coverage.** Per decision: the valid route-memory slots carried to world through the anchor, projected on the route; n_past =
valid slots with station > entry; cov = share of the stations entry .. exit (1 m) inside [min, max] of the projected valid slots.

**Statistics.** Cluster bootstrap, B = 10 000, seed 0, percentile 95 %, clusters = scenes (held: log id); replicates with an empty group
are dropped (counted).
- **M1** (turn level): Delta1 = median D (n_x >= 3) - median D (n_x = 2). Supported: Delta1 >= 0.2 and CI lower > 0; refuted: CI upper < 0.2;
  else inconclusive; < 4 turns in a group: inconclusive. n_x = 1 turns are tabled, not tested.
- **M2** (decision level): the served command agrees with the turn (L on a left turn, R on a right turn) or says S. Delta2 = median delta (S) -
  median delta (agree). Supported: Delta2 >= 0.2 and CI lower > 0; refuted: CI upper < 0.2; else inconclusive; < 20 decisions or < 4 turns in
  a group: inconclusive. Also reported (no verdict): the distance-matched difference (same difference inside d_e bins 0-10, 10-20, 20-30,
  averaged with weights = the smaller group's count), and the opposite command's deficit.
- **M3** (descriptive): delta by cov bins (0, (0, 0.5), [0.5, 1), 1) and by n_past bins (0, 1-5, 6-15, > 15), per class.
- **M4** (shape of the plan near the entry; descriptive with a fixed reading): decisions with d_e in [0, 10) and h_dpsi >= 15 deg, ratio
  r = dpsi / h_dpsi. Shares: straight r < 0.25, partial 0.25 <= r < 0.6, full r >= 0.6, for failed and completed turns. Reading per
  population (failed turns): **averaging** if the partial share >= the full share and the modal 0.1-wide bin of r is in [0.1, 0.6);
  **mode selection** if the partial share < 0.2 while straight and full shares are each >= 0.2; **straight-dominant** if the straight
  share >= 0.6; else **mixed**. The per-turn picture (each turn's decisions as a strip) is shown next to it. Figure `figs/mode_hist.png`.
- Distance: delta by d_e bin per class, reported.

Outputs: `scripts/rf_modeavg.py` (read on the Mac), `results/mode_avg/` (mode_avg.md, mode_avg.json, turns.csv, decisions.csv.gz),
`figs/mode_hist.png`, one README paragraph.

## Reading 2 (box, GPU pool): FLOW1's flow head vs its regression twin in closed loop

**Checkpoints** (d246, `experiments/flowhead`): FMH-F-s0, FMH-F-s1 (flow head, `thead.pt` kind fm) and their twins RGH-F-s0, RGH-F-s1
(same net, kind rg), under `$DATA_DIR/runs/op_parity/runs/<tag>/`. No route memory (SH30 recipe: P2 arm + drivable hinge).

**Serving.** The AlpaSim sh30 driver (experiments/alpasim/lib/sh30_core.py) gains the head path that jevdrive.bench serves on navtest: a
checkpoint with `thead.pt` next to it sets the policy's tap `select_4`, and the served plan = the head's 8 rear-axle poses (fm: 4 Euler
steps from the fixed noise row 0, as bench; rg: the regression). For a flow head the driver also logs the head's samples from noise rows
0 .. 7 (`th_k` in drive.jsonl, K = 8, SH30_THEAD_K), so the "K = 8 samples at the failing decisions" reading uses the exact hidden state of
each closed-loop decision. Checkpoints without thead.pt are served exactly as before.
- **G-eq** (before any read): `sh30_check.py` parity on 300 navtest tokens per tag: the core's served poses against the bench prediction
  file (`preds/<tag>-warp__base.npz`, the head poses of d246): mean of per-token max |diff| <= 0.05 m and p99 <= 0.15 m (SH30-F-s0 measured
  0.011 / 0.031 m). Failing it on a tag, that tag is not read and the blocker is reported.

**Runs.** The standard held 460 / 18-decision read (`rm_chain.sh read <tag> i 6`, 6 chunks, pool), into `route_mem/read/<tag>`; report:
ol_confirm (held18: score, zeros, at-fault), rf_follow (@18: strict held turns, completion), rf_tax extract + RECEDING extract-held
(per-decision plan / human heading), th_k extraction. References (context only, no claim): RSG-L18-s0 / -s1 held @18 (16 / 14 of 24).

**Claims** (24 strict held turns = the turn rows of rf_follow's scenes.csv; per scene, completion = mean over the seeds):
- **F1**: completed turns FM - RG (2-seed mean, summed over the 24 turns). Supported: >= +4 and CI lower > 0 (paired cluster bootstrap over
  logs, B 10 000); refuted: point <= +1 or CI upper < +4; else inconclusive. One seed only: the same rule on that seed, said so.
- **F2**: approach-window deficit (Reading 1's D per turn, held window), median over turns FM - RG. Supported: <= -0.3 and CI upper < 0;
  refuted: CI lower > -0.3; else inconclusive.
- **F3** (no-harm, descriptive): held 460 zeros, at-fault zeros, mean score, FM - RG (2-seed means, ol_confirm's paired CIs).
- **K-samples** (descriptive with a fixed reading): at the failing decisions (FM rollouts, approach-window decisions of failed turns whose
  served deficit delta >= 0.5), the number of the 8 samples with dpsi >= 0.6 h_dpsi. "The mode exists in the head, selection is missing"
  if >= 1 of 8 is full on >= 50 % of those decisions; "the mode is absent" if on < 10 %; else partial. Also the same count on the
  completed turns' window decisions, and the samples' spread (sd of dpsi) against the served deficit.

Outputs: `experiments/flowhead/scripts/flow_cl.sh` (pool chain), `experiments/flowhead/results/closed_loop/` (cl.md, cl.json, tables),
the figure next to it, a README paragraph in experiments/flowhead. research/decisions.md is not edited by this lane.

## Amendments (appended after the prereg commit; each says whether it came before or after a statistic was seen)

1. (Before any Reading 1 statistic.) Lane code: rf_corridor's SceneMap.match needs the lane polygons and junction_eyes' je_ruler, which
   the held lane extract does not carry; both populations use rf_pai_follow's lane-graph helpers on the same lane extracts instead
   (`chains`: every successor chain from a lane's point nearest the route point, cap 256; end heading by the module's chord heading;
   chains that end short of the arc L and are shorter than 25 m, the map extent's cut, are dropped as in rf_pai_follow MIN_CHAIN). The
   matched lane = the nearest group lane within 2.5 m. Everything else in "Exits" is as registered.
