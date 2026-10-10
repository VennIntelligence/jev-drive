# STOPLBL: what labels supervise "slow down / stop for what the picture shows", sized by what each board loses to it

Written 2026-10-10. Read-only inventory: no training, no GPU, no closed-loop run. Scripts [`scripts/stoplbl_label.py`](../scripts/stoplbl_label.py) (box, CPU),
[`scripts/stoplbl_part0.py`](../scripts/stoplbl_part0.py) (Mac); tables in [stoplbl/](stoplbl/).

## Part 0: points each board loses to the three situations

Situation labels are mine (recomputed): for every log frame the ego's *hindsight route* is the driven path of the next 30 s (+15 m straight), and a token is in a
situation when the map target is crossed by that route within 50 m: **red** = a traffic-light stop line whose lane connector is red in the log's
`traffic_lights` (the connector the route takes after the line); **stop sign** = a type-1 stop polygon; **yield** = a type-3 turn-stop or type-4 yield polygon;
**ped** = a marked crosswalk on the route with a pedestrian box within 3 m of it; **turn > 45** = the bench strata bin (logged 4 s heading change, 1 441 of 12 146
valid navtest tokens; the map-based route turn >= 45 deg within 50 m gives 3 457, a looser definition). Sets overlap (552 of the 2 191 yield tokens are turn > 45 tokens: turn-stop
polygons sit on turning connectors); the exclusive partition is in `stoplbl/part0_navtest_excl_turnfirst_SH30.csv`. **loss in set** = sum over the set of (100 - EPDMS) / N,
in board points; **excess** = fraction x (EPDMS of the rest - EPDMS of the set), the points the set costs *beyond* what the rest of the board loses per token,
cluster bootstrap by log (B 1000). Navtest = seed mean per token, all 12 146 tokens, bench `units.csv` (SH30 s0 s1, P2H10 s0 s1, P2H10S s0 s1; SH30 shown, others within
0.1 point, `part0_situations.csv`). Navhard = 225 groups, situation of the group's stage-1 tokens (stage-2 frames inherit it; approximation), `combined` score.

| board | situation | n units | loss in set (pts of board) | share of board loss | excess beyond rest [95% CI] | how attributed |
|---|---|--:|--:|--:|--:|---|
| navtest SH30 (EPDMS 89.54, loss 10.46) | traffic-light compliance (TLC) term | 30 tokens fail (0.25%) | 0.25 | 2.4% | n.a. | Shapley split of the per-token EPDMS to perfect, recomputed |
| | red light within 50 m | 1 277 | 0.65 | 6.2% | -0.49 [-0.70, -0.31] | map labels, recomputed; set EPDMS 93.8 against 89.1 for the rest |
| | any traffic-light line within 50 m | 2 575 | 1.41 | 13.5% | -1.02 [-1.38, -0.66] | same |
| | stop sign within 50 m | 821 | 0.89 | 8.5% | +0.20 [+0.06, +0.33] | same; EP 0.807 against 0.877, DAC 0.956 against 0.973 |
| | yield / turn-stop within 50 m | 2 108 | 2.01 | 19.3% | +0.25 [-0.04, +0.53] | same; overlaps turn by 552 tokens |
| | pedestrian crosswalk within 50 m | 1 464 | 1.07 | 10.2% | -0.22 [-0.41, -0.04] | same |
| | turn > 45 deg (logged) | 1 441 | 2.37 | 22.6% | +1.29 [+0.93, +1.61] | strata bin; DAC term carries 0.71 of the 1.29, EP 0.10, EC 0.32 |
| navtest P2H10 / P2H10S | turn > 45 excess | 1 441 | | 21.7% | +1.28 / +1.24 | same; stop sign +0.21 / +0.20, red -0.58 / -0.54 |
| navhard SH30 (33.67, loss 66.33) | TLC | stage 1: 0 tokens, stage 2: 62 of 5 462 (1.1%) | 0.75 of the stage-2 token mean | 1.4% of stage-2 token loss | n.a. | stage-2 token Shapley, recomputed |
| | red light | 5 groups | 1.27 | 1.9% | -0.20 [-0.78, +0.30] | not separable from zero |
| | stop sign / yield / ped | 27 / 43 / 26 groups | 8.0 / 13.5 / 7.4 | 12% / 20% / 11% | +0.04 / +0.96 / -0.32, all CIs contain 0 | |
| | turn > 45 | 56 groups | 19.17 | 28.9% | +3.54 [+1.66, +5.55] | P2H10 +3.65, P2H10S +3.61 |
| AlpaSim nuPlan 700 (P2H10-F s0 s1, mean 0.9483, loss 72.4 scene-seeds) | red | 108 scenes x 2 seeds | 3.25 | 4.5% | scene score 0.970 against 0.946 | navtest tokens of the scenes, recomputed; 2 zeros, 17 partial |
| | stop sign | 96 | 4.02 | 5.5% | | 3 zeros, 12 partial |
| | yield / turn-stop | 178 (non-exclusive) | 18.2 | 25.2% | 0.898 against 0.956 | 15 zeros, 23 partial; most are also turns |
| | ped crosswalk | 156 | 6.03 | 8.3% | | 2 zeros |
| | turn > 45 (logged) | 122 | 19.2 | 26.5% | 0.843 against 0.958 | 17 zeros, 19 partial; zeros = 18 of 49 by decision 227 |
| AlpaSim PAI (60 scenes, 1 seed, lost 38.46) | overrun of the logged stop, red light | 3 zeros | 3.0 | 7.8% | | quoted: decision 237 / pai.md (L5); scene ids 21626256, b0fa4732, 9e3fd12d |
| | overrun at a give-way T-junction | 1 | 1.0 | 2.6% | | quoted (7a824ffa, 12.8 against 2.8 m/s) |
| | stop sign passed at 15.2 against 6.4 m/s | 1 | 1.0 | 2.6% | | quoted (59e085d7, flagged L2 in pai.md) |
| | overrun, reason unknown | 1 | 1.0 | 2.6% | | quoted (1c7e2423) |
| | fast turn entry (flags L2), 1 also command unused | 3 | 3.0 | 7.8% | | quoted: 1d6e30bc, 3a48e906, 96da4f1a (the last two never attempt the turn) |
| | lead vehicle (L1, both lateral in fact) | 2 | 2.0 | 5.2% | | quoted |
| | slow progress (L4), 8 without + 4 behind the lead limit | 12 | 5.46 | 14.2% | | quoted; situation not determinable from stored results (rollout logs pruned) |
| HUGSIM 64 (SH30, 1 seed, lost 35.63 HD) | red light / stop line | | | | | HD has no light term (rc, nc, dac, ttc, comfort); scenario yaml carries no signal state: **not determinable**, loss 0 by construction unless a collision follows |
| | yield / stop sign | | | | | not determinable from stored results |
| | fast turn entry (L2) | 4 | 3.21 | 9.0% | | quoted: decision 237 / hugsim.md |
| | lead vehicle stopped (L1) | 9 | 7.81 | 21.9% | | quoted |
| WOD-E2E val | red light | | | ~0 | | quoted decision 169: red-light stops tie across arms (8.3-8.4 RFS), "red lights are not where WOD loses" |
| | stop-sign junction launch | 45 frames | | 35% of the stopped-frame loss | | quoted decision 169 (hand check of the auto labels: light 92.5%, stop sign 87.5%) |

### Reading

- **Red light / stop line costs almost nothing on navtest and navhard.** TLC is 0.25 points of 10.46 (2.4%) on navtest; 30 of 12 146 tokens fail, 13 of them in the 1 293 red-within-50-m tokens (1.0%).
  The red-light set scores *higher* than the rest (93.8 against 89.1) because the log driver stops there and so does the model (in 912 of 1 277 the log driver is stopped at the line; the model's
  EP, NC, DAC are no worse). The navhard stage-2 TLC (1.1% of tokens, 0.75 points of the stage-2 token mean) is a consequence of the synthetic pose perturbation, not of a stop prior.
  On the closed-loop boards the same situation is not small: PAI 4 to 5 of 60 scenes (3 certain red-light overruns plus an unknown one, 7.8 to 10.4% of the PAI loss).
  nuPlan 700: 108 scenes with a red light on the route lose 3.25 of 72.4 (4.5%), a below-average share.
- **Yield / stop sign** is a modest navtest cost (stop sign +0.20 points beyond the rest, EP and DAC; yield not separable) and a larger share on the closed-loop nuPlan board
  (178 yield scenes lose 25% of the loss, but they are mostly the same scenes as the turns).
- **Turns above 45 degrees are the one situation that costs on every EPDMS board**: +1.29 [0.93, 1.61] points beyond the rest on navtest (22.6% of the loss in 11.9% of the tokens),
  +3.5 [1.7, 5.6] on navhard (28.9% of the loss in 24.9% of the groups), 26.5% of the nuPlan 700 loss, 7.8 to 9.0% of the PAI and HUGSIM loss as fast entries.
  **Speed against path**: on navtest the turn excess is DAC 0.71 (55%), EC 0.32, EP 0.10 (7%): the cost is in the path, not in progress (recomputed). Decision 203 /
  op_parity pt-swap (quoted, not recomputed): putting the log timing on the plan's path removes 12.6% of the > 20 deg DAC failures, the log path with the plan's timing removes 94.6%, the lever is the path shape.
  The speed part is visible on the closed-loop boards (PAI 4 and HUGSIM 4 fast entries at 6.4 to 10.4 m/s against 2.6 to 2.8, decisions 153 / 159).
- What cannot be attributed from stored results: PAI progress loss (12 units, 5.46), HUGSIM stop / yield, any board's "slow" scenes by situation except nuPlan 700 (token labels recomputed by me).

Sub-score split, all situations and recipes: `stoplbl/part0_situations.csv`, `part0_subscore_loss_navtest.csv`, `part0_term_excess_navtest.csv`, `part0_tlc.csv`; nuPlan 700: `alpasim_nuplan_700_*.csv`, `alpasim_nuplan_zeros_labels.csv`.

## Part 1: inventory of labels (sized by Part 0: turns > 45 deg first, stop sign / yield second, red light last)

Everything below is **measured on the box** unless a row says "read from code / docs". Sampled: 300 random navtrain logs (seed 0), 267 of which contain navtrain tokens, 23 901 navtrain tokens
(of 103 288), 96.3% with a complete route (30 s hindsight or >= 50 m); counts are fractions with a log-cluster bootstrap CI, then scaled to 103 288 (`stoplbl/navtrain_counts.csv`). All of navtest
(147 logs, 75 122 frames, 12 146 tokens) was labelled for Part 0. Code: `scripts/stoplbl_label.py` (labeler, 267 logs in about 4 minutes on 32 workers, so full navtrain is about 15 to 20 minutes and
about 60 MB), `stoplbl_counts.py`, `stoplbl_sheet.py`.

### Fields that exist

| field | navtrain / navtest (nuPlan logs + maps), measured | in a training cache? |
|---|---|---|
| per-frame light state | log pkl `traffic_lights` = list of (lane-connector id, is_red); populated in 51.7% [48.0, 55.4] of frames; only red / not red (yellow and green are not separated); lights listed only for connectors the log tracked | no. Only `experiments/op_probe` `attrs_navtest.csv` (a navtest red flag); raw log only |
| stop-line / stop-polygon geometry | map layer `stop_polygons`, types 0 pedestrian crossing, 1 stop sign, 2 traffic-light stop line, 3 turn stop (unprotected-turn yield), 4 yield (Singapore only, 3 polygons); 4 cities, e.g. Boston 678 polygons, Las Vegas 567, Pittsburgh 305, Singapore 546; `lane_connectors.traffic_light_stop_line_fids` ties a light line to its connectors | no (raw maps in `datasets/navsim/maps`) |
| lane connector / intersection geometry | `lane_connectors` (turn type, linked light lines), `intersections`; cached only as `jct_s`, `jct_dist`, `n_turn`, `turn_s` ... in `processed/op_route_cmd/navtrain/route.npz` (103 288 rows) | yes (route fields), light lines no |
| crosswalks | map layer `crosswalks` (polygons, marked flag); pedestrians come from the log boxes (`anns`) | partly: `op_adapt/navtrain_labels.parquet` has pedestrian-in-corridor and distance, not the crosswalk |
| stop-sign locations | only as type-1 stop polygons (the stop line), no sign positions | no |
| route curvature ahead | hindsight path: `route.npz` has `turn_deg`, `turn_s`, `turn_end_s`, `turn_rmin` for all 103 288 tokens (decision 93 / op_route_cmd; 32 269 samples with a turn >= 25 deg) | yes |

Other datasets (code / file reading, one measurement where stated):

| dataset | red light / stop line | stop sign / yield | turn | usable for |
|---|---|---|---|---|
| AlpaSim nuPlan public scenes (navtest logs, 12 148 scene dirs in `alpasim_nuplan/nuplan_test`) | **yes**: `tls_data_dt0.05.feather` per scene (lane id, scene_ts, status; 84 rows in the scene I opened, status codes not decoded) plus the nuPlan map of the city | same nuPlan map | same | scoring / diagnosis of the board; label rows for training are not allowed (public scenes = navtest) |
| AlpaSim PAI (67 `.usdz` scenes on the box, measured over all 67) | static only: `map.xodr` has 1 315 TrafficLight signals (966 dynamic = real lights, 349 static), 325 `stopline`, 41 `yieldline`, 791 RegulatorySigns of which 188 are R1 (stop sign), plus `roadMark` objects named stop; 61 of 67 scenes have at least one signal. No time-varying light state in `map.xodr` (checked `sequence_tracks.json`, `datasource_summary.json`, `data_info.json` for light fields: none found). Parsing map to route: not done | yes (R1 stop signs, yield lines) | lanes and edges only (pai.md) | diagnosis only; the logged driver's stop (L5) already marks 5 overruns |
| HUGSIM 64 scenarios (4 datasets x 16) | none: HD = rc, nc, dac, ttc, comfort; the scenario yamls hold plan, start pose, difficulty, no signal | none | route reference only | only nuScenes (16 of 64 runs, 8.32 of 35.63 lost) could be queried through the nuScenes map expansion (`stop_line`, `ped_crossing`, `traffic_light` layers exist in `datasets/nuscenes/maps/expansion`, no light state); waymo / pandaset / kitti360 scenes: nothing |
| WOD-E2E | none (decision 236, `docs/waymo-e2e.md`: front3 images, calibration, ego history / future, `intent`, rater trajectories on val; agent boxes and road geometry 0 in train / val). Confirmed for lights: no field | none | `intent` only | image-only supervision |
| nuScenes (box has v1.0-trainval + map expansion) | map has `traffic_light`, `stop_line` layers, no light state | | yes | stop lines only; read from the file listing, not used here |

### Counts on navtrain (sample, per 103 288 tokens)

| situation (route target within 50 m) | frames in sample | fraction [95% CI] | scaled to navtrain | log driver behaviour |
|---|--:|--:|--:|---|
| red light, state at t0 | 2 176 | 9.5% [7.9, 10.9] | 9 763 [8 199, 11 226] | stopped (min speed < 1 m/s) 6 609 (6.4%); passes at >= 3 m/s 2 521 (2.4%) |
| **red light, state when the ego reaches the line (hindsight)** | 1 369 | 6.0% [5.0, 7.0] | 6 142 [5 137, 7 185] | stopped 5 433 (5.3%); passes at >= 3 m/s 332 (0.3%): the log driver almost never runs a red |
| red at t0 but not at arrival | 1 286 | 5.6% | 5 770 | the light turns green on the way: **59% of the red-at-t0 frames** |
| any light line (red or not) | 4 989 | 21.7% [19.2, 23.9] | 22 383 | |
| stop sign | 1 853 | 8.1% [6.6, 9.7] | 8 313 [6 842, 9 983] | stopped 6 784 (6.6%); slowed to < 0.6 v0 677 (0.7%) |
| yield / turn-stop | 3 013 | 13.1% [11.3, 15.3] | 13 518 | stopped 2 773 (2.7%); the turn-stop line is conditional (see sample check) |
| crosswalk with a pedestrian near it | 3 029 | 13.2% [11.3, 15.3] | 13 590 | stopped 4 154 (4.0%) |
| any stop target (red t0 / sign / yield / ped) | 8 003 | 34.8% [32.7, 37.0] | 35 905 | |
| turn >= 45 deg starting within 50 m (hindsight route) | 7 143 | 31.0% [28.8, 33.1] | 32 047 [29 701, 34 184] | entry speed of the log driver, percentiles 10 / 25 / 50 / 75 / 90: 1.0 / 2.4 / 3.7 / 4.8 / 6.2 m/s; v0 median 3.6; curvature speed sqrt(2 m/s2 / kappa_max): median 3.65, so the log's entry speed is above it in 49% and above 1.5x in 24% |
| turn >= 45 deg within 30 m, v0 > 3 m/s | 4 384 / 6 293 | 19.0% / 27.3% | 19 669 / 28 233 | |

(`stoplbl/navtrain_counts.csv`, `navtrain_turn_entry_speed.csv`.) The PAI fast entries (6.4 to 10.4 m/s, decision 153 / 237) lie beyond the 90th percentile (6.2 m/s) of the log driver's entry.
The turn count is looser than the bench strata (11.9% of navtest tokens have a logged 4 s heading change > 45 deg): the route turn is read from the 30 s hindsight path.

### Label quality, 20 frames per situation, front image next to the label (judged by me, one reader)

Contact sheets: `figs/stoplbl/{red,stopsign,yield,turn}_{0..4}.jpg` (4 frames per image, label in the yellow header; `*_meta.csv` lists token and log).
Look at: the header text against the lights and signs in the picture, and against the driver's minimum speed on the route.

| situation | right | wrong | cannot tell | what the failures are |
|---|--:|--:|--:|---|
| red light (t0 state), stop line 6 to 40 m ahead | 13 / 20 | 1 | 6 | wrong: a green light is visible and the driver passes (red belongs to another movement); the 6 cannot-tell frames are all drivers that pass at 5 to 9 m/s, the red at t0 is not the red at arrival. The geometry (stop line distance) was never visibly off. Consistent with the 59% above, so the target must use the state at arrival, which only a log (not a single frame) gives |
| stop sign, line 6 to 40 m ahead | 12 / 20 | 2 | 6 | wrong: a parking-lot polygon (driver passes at 6.4 m/s) and a Singapore junction without a visible sign (driver passes at 4.8 m/s); cannot tell: sign not visible in the frame at 36 m, the driver stops (garage exits, one car-following stop) |
| yield / turn-stop line, 6 to 40 m ahead | line is on the route in 20 / 20, but the picture demands a yield in 3 / 20 | | | 14 of 20 are green-light protected intersections where the driver passes at speed, 3 are standstill frames, 3 yield (min speed 0.7 / 1.5 / 4.1 m/s). The polygon says "yield if turning against traffic", not "stop"; as a label it is nearly all negatives |
| turn >= 45 deg, starts 8 to 40 m ahead | 17 / 20 | 0 | 3 | turn visible in 17; the 3 are turns that start 30 to 40 m ahead and are not visible yet |

### The natural supervision target

- **s_stop(t): arc distance along the ego's route to the next point where it must stop** = min over the traffic-light line whose lane connector is red *when the ego reaches it*, the stop-sign line, and optionally the yield line when a conflicting agent is present.
  Computable from the log and the map for every frame that has a hindsight route: 96.3% of the sampled navtrain tokens (route complete), of which 14.0% have a target within 50 m
  (19.1% within 80 m, 25.0% anywhere on the route); the other frames get "no stop target ahead", also a computable label. Of the 14.0%, 11.8% of all frames are the ones where the log driver also stops there (2 717 frames in the sample), the remaining 2.2% are lights that turn or signs the driver rolls (0.7% decelerate only).
  The state at the frame itself is the wrong input to the label (59% of "red now within 50 m" are green on arrival): the label needs frames t+k of the log, so it is hindsight, as the route polyline of decision 93 is.
- **Turns**: v_req(s) = sqrt(a_lat / kappa(s)) from the hindsight path heading, with the backwards braking limit; computable on the 96.3% too (31.0% have a >= 45 deg turn within 50 m). a_lat = 2 m/s2 puts the median log entry exactly at the limit, so the log driver's own entry-speed distribution (percentiles above) is the better "required speed" than a physical limit: the label is a percentile of what the log did at that curvature.
- Not computable: yellow (only red / not red is logged, so yellow-onset timing is unknown), lights with no state in the log (48% of frames carry no light entry; 5.8% of stop-line approaches have state `unk`), a yield that depends on a conflicting agent (needs the agents, available in nuPlan logs but not tested), stop-sign locations other than their stop line.

### What is missing

- Light state in any cache and the state at the arrival frame; the stop-polygon distance in any cache (all derived on the fly from raw logs and maps here).
- Light state for the board scenes: PAI has static lights and stop lines but no per-time state; HUGSIM has none (nuScenes scenes only via the map expansion, no state); WOD-E2E none. For these boards the stop line / sign geometry is the only available score-side label, and the logged driver's own stop (PAI L5) the state proxy.
- A yield label that says whether yielding is required (needs conflicting-agent logic).

### Sampled / measured / read

Measured by me: all navtest and navtrain-sample numbers, the PAI map signal counts (67 usdz, `map.xodr`), the nuScenes / HUGSIM / WOD file inventory, the 80 sample judgements. Quoted: PAI and HUGSIM class numbers (decision 237), WOD (decision 169, 236), pt-swap (decision 203 family).
Not measured: label noise beyond the 20-frame checks (one reader), stage-2 navhard situations (inherited from the stage-1 token), light state codes in the AlpaSim tls feather.
Files: `stoplbl.md`, `results/stoplbl/*.csv`, `figs/stoplbl/*`, `scripts/stoplbl_*.py`. Box: `$DATA_DIR/runs/lowboard_diag/stoplbl/` (parquet labels, sheets).
