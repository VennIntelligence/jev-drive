# op_route_ft report: route-choice adapter fine-tune of openpilot (2026-10-05 night, single seed 0)

Pre-registration: [plans/2026-10-05-route-ft-prereg.md](../plans/2026-10-05-route-ft-prereg.md). Serving and harness: [harness.md](harness.md). Arms: rc-bear (bearing + distance adapter), rc-poly (16-point polyline adapter),
rc-ctl (same fine-tune, adapter input zero), rc-all (rc-bear + T4 start / stop + T5 history + T6 drivable area; exploratory). Shipped = Cinque as in decision 127 (zones-off arm).

## Results

**The adapter was fed in the closed loop (wiring check, done first).** rc-bear: interface.json says `command.route_geometry: nav-polyline` and carries the `route_adapter` path; plans.jsonl has `ra.f` on every plan step
(10255, 27297, 5423, 24758 checked). `ra.f` = [present, has, d / 50, sin b, cos b - 1, b / 90]: has = 1 on the approach, d falls 0.50 -> 0.07 (25 m -> 3.5 m) before the junction, and for the right turn 10255 sin b = -1.0 (b = -90 deg, right),
for the left turn 24758 sin b = +0.88..1.0; after the car has left the route the maneuver is gone (has = 0, d = 3.0). rc-poly (27297): the 16 polyline vertices vary (std 0.1-0.47); rc-all: bear features vary like rc-bear;
rc-ctl: no `ra` (zero bias by design). Commanded side = route turn side on every checked turn, and the logged desire pulse matches the side (rft_split: 0.94-0.95 for rc-bear / shipped). The 4/25 is a valid measurement of an adapter that was told the right exit.

**Primary (25 B2D junction turns of decision 127, zones off, seed 2, route-derived command, no noise): the line fails for every arm.** Line: rc-bear >= 13/25, above shipped and above rc-ctl (paired CI lower bound > 0).

| arm | took exit (all) | choice (13) | forced (12) | paired diff vs shipped [95% CI, route bootstrap] | leaves lane (of entered) | collisions in turn windows | route DS mean (20 routes) |
|---|--:|--:|--:|---|--:|--:|--:|
| shipped | 1 | 0 | 1 | | 18 / 20 | 5 | 26.3 |
| rc-ctl | 6 | 2 | 4 | +0.20 [+0.07, +0.36] | 17 / 21 | 7 | 34.0 |
| rc-bear (main) | 4 | 1 | 3 | +0.12 [0.00, +0.30] | 18 / 21 | 5 | 36.2 |
| rc-poly | 7 | 2 | 5 | +0.24 [+0.07, +0.44] | 18 / 22 | 6 | 39.0 |
| rc-all (exploratory) | 6 | 3 | 3 | +0.20 [+0.04, +0.40] | 19 / 21 | 8 | 31.0 |

Command effect = arm minus rc-ctl, paired by turn: rc-bear -0.08 [-0.28, +0.15] (2 gained, 4 lost), rc-poly +0.04 [-0.11, +0.23], rc-all vs rc-bear +0.08 [0.00, +0.21]. In closed loop the command adds nothing measurable over the fine-tune itself.
Exit-taking by R_min (< 7 / 7-10 / >= 10 m, of 6 / 5 / 14 turns): shipped 0 / 0 / 1, ctl 2 / 1 / 3, bear 0 / 0 / 4, poly 2 / 0 / 5, all 0 / 1 / 5. Out-of-lane share and window collisions are not above shipped (rc-all 8 vs 5 window collisions is the one exception, n = 25, no test).
Route DS (zones off) is higher for every fine-tuned arm than shipped (+5 to +13) and unplanned: 20 routes, one seed, not a pre-registered line; read it only as "the fine-tune did not make driving worse".

### Failure split (rft_split.py; [split.md](split.md), per turn in the same file)

Which of the 25 turns the car entered, and why the entered ones were lost (first matching cause; thresholds in split.md; steering = desired curvature act_k signed to the commanded side, read from 6 s before entry to the end of the turn span):

| arm | took | not chosen | wrong side | chosen late (> 3 m of arc past the turn start) | in time, crawl / stop | in time, moving | never entered |
|---|--:|--:|--:|--:|--:|--:|--:|
| shipped | 1 | 10 | 2 | 4 | 3 | 0 | 5 |
| rc-ctl | 6 | 0 | 2 | 9 | 4 | 0 | 4 |
| rc-bear | 4 | 4 | 0 | 6 | 7 | 0 | 4 |
| rc-poly | 7 | 6 | 1 | 6 | 1 | 1 | 3 |
| rc-all | 6 | 1 | 0 | 8 | 6 | 0 | 4 |

Entered turns, medians: speed in the first 15 s after entry 0.9-2.0 m/s with 31-42% of that window stopped (shipped 1.0 m/s, 41%); peak curvature towards the commanded side / needed 1/R_min: shipped 0.31, ctl 1.91, bear 1.34, poly 1.23, all 3.39;
turn-in (first 0.5 / R_min towards the commanded side) 2.5-4.2 m of arc after the turn start (shipped 3.1 m, only 8 of 20 turns steer that far). Peak steering towards the commanded side (choice / forced turns): shipped 4 of 11 / 6 of 9, ctl 9 / 11 and 9 / 10, bear 8 / 12 and 8 / 9, poly 8 / 12 and 8 / 10, all 11 / 12 and 9 / 9.

Readings:
1. The fine-tuned head does not fail by being unable to steer: the heads reach 1.2-3.4 x the needed curvature (shipped 0.3 x), and never in the wrong direction (wrong side 0-2 turns). Not choosing is the shipped failure (10 of 20 entered turns) and mostly disappears with the fine-tune (rc-ctl 0, rc-all 1, rc-bear 4 choice turns).
2. The dominant remaining failure is timing: 6-9 turns per arm steer only after the car has passed the turn start (median turn-in 5-6 m of arc after it for ctl / all / shipped-lost), e.g. rc-bear 10255: the command reads right 90 deg at d = 3.5 m, k stays about 0 for the next 30 s, the car drives straight to 28 m off the route and only then steers (k = 0.17-0.19 at t = 54 s). The step from the command to the action head is weak at the approach poses, where action supervision is about zero by construction (the pure-pursuit look-ahead is shorter than the distance to the junction, see openloop.md); the plan head moves with the command (open loop 0.58-0.59 exits correct), the action head does not follow in time.
3. Crawling and stopping are a harness-wide property, not a fine-tune effect: every arm including shipped sits at about 1 m/s with 40% of the window stopped, and speed does not separate taken from lost turns (ctl took turns at median 0.7 m/s, shipped lost at 0.9). It does hide the effect of steering (7 of rc-bear's 17 lost turns steer in time but are stopped for more than 25% of the window or move below 1.5 m/s), so the longitudinal side (plan-idm-latch scheduler, red lights with zones off) is a second lever; I did not isolate its cause.
4. Visibility: the rc-bear / ctl losses are not concentrated on the turns the camera cannot see. By R_min, rc-bear takes none of the 11 turns with R_min < 10 m (CARLA visibility table, wide camera, d = 20 m: 88% of the exit window visible for tight, 100% for wide turns, whole window 31% / 99%; at d = 10 m only 3% of tight exits are fully inside the wide frame) and 4 of 14 wide ones, so tight turns do pay the visibility cost; but rc-poly and rc-ctl take 2 of 6 tight turns, so visibility does not explain the whole gap and the evidence for it is weak at n = 6.
5. rc-bear versus rc-ctl: the fine-tune (action pathway + T1-T3 data, no command) is where the gain over shipped comes from (+5 turns); the adapter does not add to it in closed loop. Because the route desire pulse (turnLeft / turnRight 20 m before the turn) still reaches the model in these runs and rc-ctl also steers to the commanded side in 9 / 11 choice turns, the side information in rc-ctl comes from the desire pulse, not from the adapter.

### Figures

- [figs/b2d_turn_panels.png](../figs/b2d_turn_panels.png): BEV tracks of four turns for shipped (grey), rc-ctl (blue), rc-bear (red) on the route lane. Look at 10255 (rc-bear drives straight past the junction, then curves right; rc-ctl turns in earlier but stops short), 27297 (rc-ctl takes the right turn, shipped and rc-bear lose it), 28008 turn 1 (rc-ctl turns left, rc-bear and shipped go straight), 24758 turn 0 (a forced curve: rc-bear follows it, rc-ctl and shipped stall).
- [figs/rc-bear-s0_10255_turn0.gif](../figs/rc-bear-s0_10255_turn0.gif): rc-bear on route 10255 turn 0 (right turn, 89 deg, R_min 6.4 m, choice): third-person camera beside the model input frames (road and wide view; the route command enters through the adapter bias and the desire pulse, so nothing command-specific is drawn in the pixels). Look at when the car reaches the junction mouth and how long k stays near zero.

## Open loop (rft_eval.py, [openloop.md](openloop.md))

| readout | line | O | rc-bear-s0 | rc-poly-s0 | rc-ctl-s0 | rc-all-s0 |
|---|---|---:|---:|---:|---:|---:|
| CARLA dev exits: correct per (pose, exit) row | >= 0.8 | 0.156 [0.14, 0.17] (n 1505) | 0.583 [0.55, 0.61] (n 1505) | 0.591 [0.56, 0.62] (n 1505) | 0.291 [0.27, 0.31] (n 1505) | 0.579 [0.55, 0.61] (n 1505) |
|   among rows whose plan reaches d + 12 m |  | 0.408 [0.39, 0.43] (n 576) | 0.881 [0.85, 0.91] (n 997) | 0.885 [0.86, 0.91] (n 1005) | 0.372 [0.36, 0.39] (n 1176) | 0.872 [0.85, 0.90] (n 1000) |
|   share of rows whose plan stops short |  | 0.617 | 0.338 | 0.332 | 0.219 | 0.336 |
|   left commands |  | 0.128 [0.10, 0.16] (n 485) | 0.487 [0.44, 0.53] (n 485) | 0.470 [0.43, 0.51] (n 485) | 0.270 [0.22, 0.32] (n 485) | 0.460 [0.42, 0.50] (n 485) |
|   straight commands |  | 0.306 [0.26, 0.36] (n 500) | 0.792 [0.76, 0.82] (n 500) | 0.800 [0.77, 0.83] (n 500) | 0.468 [0.41, 0.52] (n 500) | 0.828 [0.80, 0.86] (n 500) |
|   right commands |  | 0.038 [0.02, 0.06] (n 520) | 0.473 [0.43, 0.52] (n 520) | 0.502 [0.46, 0.54] (n 520) | 0.140 [0.11, 0.17] (n 520) | 0.452 [0.41, 0.49] (n 520) |
|   3-exit poses with all three right |  | 0.000 [0.00, 0.00] (n 269) | 0.375 [0.32, 0.43] (n 269) | 0.353 [0.30, 0.41] (n 269) | 0.000 [0.00, 0.00] (n 269) | 0.335 [0.29, 0.39] (n 269) |
|   signed lateral at d + 10 m, turn rows (m) |  | 0.388 [0.21, 0.60] (n 1005) | 3.407 [3.19, 3.62] (n 1005) | 3.526 [3.31, 3.74] (n 1005) | 0.168 [0.09, 0.26] (n 1005) | 3.361 [3.15, 3.58] (n 1005) |
|   action[0] sign = command side, turn rows |  | 0.495 [0.48, 0.51] (n 1005) | 0.527 [0.51, 0.55] (n 1005) | 0.526 [0.50, 0.55] (n 1005) | 0.484 [0.47, 0.50] (n 1005) | 0.536 [0.52, 0.56] (n 1005) |
| no-command drift |dy(4 s)|, CARLA dev poses (median, m) | <= 0.10 | 0.000 (n 600) | 0.037 (n 600) | 0.037 (n 600) | 0.063 (n 600) | 0.042 (n 600) |
|   navtrain dev, turn within 60 m | <= 0.10 | 0.000 (n 65) | 0.012 (n 65) | 0.014 (n 65) | 0.016 (n 65) | 0.027 (n 65) |
|   navtrain dev, straight | <= 0.10 | 0.000 (n 141) | 0.009 (n 141) | 0.010 (n 141) | 0.010 (n 141) | 0.015 (n 141) |
|   WOD dev, turn within 60 m | <= 0.10 | 0.000 (n 39) | 0.009 (n 39) | 0.010 (n 39) | 0.017 (n 39) | 0.018 (n 39) |
|   WOD dev, straight | <= 0.10 | 0.000 (n 201) | 0.010 (n 201) | 0.013 (n 201) | 0.014 (n 201) | 0.018 (n 201) |
| negative offset |dy(4 s)| vs original, CARLA N1 (mean, m) | <= 0.3 | 0.000 (n 349) | 0.472 (n 349) | 0.442 (n 349) | 0.288 (n 349) | 0.507 (n 349) |
|   navtrain screened N1-N4 | <= 0.3 | 0.000 (n 140) | 0.206 (n 140) | 0.232 (n 140) | 0.042 (n 140) | 0.250 (n 140) |
|     N1 exit |  | 0.000 (n 10) | 0.132 (n 10) | 0.413 (n 10) | 0.024 (n 10) | 0.139 (n 10) |
|     N2 side |  | 0.000 (n 26) | 0.115 (n 26) | 0.108 (n 26) | 0.037 (n 26) | 0.121 (n 26) |
|     N3 wrong side |  | 0.000 (n 81) | 0.285 (n 81) | 0.296 (n 81) | 0.050 (n 81) | 0.352 (n 81) |
|     N4 u-turn |  | 0.000 (n 23) | 0.061 (n 23) | 0.067 (n 23) | 0.031 (n 23) | 0.084 (n 23) |
|   WOD online N3 / N4 | <= 0.3 | 0.000 (n 106) | 0.238 (n 106) | 0.157 (n 106) | 0.079 (n 106) | 0.309 (n 106) |
| real turn frames, nav: |y(4 s) - logged| with the route (m) | report | 1.98 (orig 1.98, n 14) | 1.94 (orig 1.98, n 14) | 1.75 (orig 1.98, n 14) | 2.04 (orig 1.98, n 14) | 1.46 (orig 1.98, n 14) |
| real turn frames, wod: |y(4 s) - logged| with the route (m) | report | 0.70 (orig 0.70, n 12) | 0.85 (orig 0.70, n 12) | 0.75 (orig 0.70, n 12) | 0.68 (orig 0.70, n 12) | 0.80 (orig 0.70, n 12) |

CI: 95% cluster bootstrap (junction / log), 2000 resamples. Lines from plans/2026-10-05-route-ft-prereg.md.

Prereg lines: CARLA dev exit correct >= 0.8 fails for all four arms (0.58 / 0.59 / 0.58, rc-ctl 0.29, shipped 0.16; 0.87-0.89 among rows whose plan reaches the junction); drift passes (<= 0.063 m); CARLA N1 negatives fail (0.44-0.51 m vs 0.3), navtrain screened pass, WOD online rc-all 0.309 borderline fail.
The adapter is read by the plan head (+0.43 rows correct over shipped, +0.29 over rc-ctl at equal fine-tuning); the open-loop to closed-loop transfer is what fails.

## Guard subset (experiments/op_guard, [rc-bear-s0](../../op_guard/results/rc-bear-s0/subset/guard.md), [rc-all-s0](../../op_guard/results/rc-all-s0/subset/guard.md))

Subset mode, paired with shipped of the same mode; verdict of the tool: both FAIL (rc-bear 8 pass / 2 fail, rc-all 5 pass / 3 fail). The rules are the guard set's (merged plan section 4).

| line | readout | rule | rc-bear-s0 | rc-all-s0 | shipped |
|---|---|---|---|---|---|
| navtest | PDMS v1 (4026 tokens), delta vs shipped | not below shipped - 0.3 | 84.26, +0.23 [-0.13, +0.61] pass | 84.81, +0.78 [+0.33, +1.27] pass | 84.03 |
| navtest | sub-score deltas (pp) | | NC +0.04, DAC +0.40, EP +0.24, TTC -0.12 | NC +0.14, DAC +0.82, EP +0.75, TTC +0.25 | |
| navhard | EPDMS two-stage (5912 tokens) | no drop | 32.90, -0.43 [-1.67, +0.84] FAIL | 35.23, +1.90 [-0.50, +4.29] pass | 33.33 |
| navhard | EPDMS early-turn set | no drop | 43.59, -0.30 [-0.66, +0.04] FAIL | 43.24, -0.65 [-1.58, +0.28] FAIL | 43.89 |
| navhard | DAC, token mean (stage 1 / stage 2 / pooled), diff vs shipped pooled | report | 88.4 / 76.3 / 77.2; -0.54 [-0.86, -0.22] | 88.2 / 77.3 / 78.1; +0.32 [-0.69, +1.29] | 89.3 / 76.8 / 77.8 |
| navhard | EP, token mean (stage 1 / stage 2 / pooled), diff vs shipped pooled | report | 74.9 / 66.6 / 67.2; +0.47 [+0.40, +0.53] | 75.1 / 70.3 / 70.7; +3.94 [+3.51, +4.42] | 75.0 / 66.1 / 66.8 |
| wod | RFS (479 rater frames) | no drop | 8.07, +0.07 pass | 8.02, +0.01 pass | 8.005 |
| wod | false start, 4116 stay rows | <= shipped + 2 pp | 4.3%, +0.2 pp pass | 5.2%, +1.1 pp pass | 4.1% |
| drift | 4 s lateral drift, max of set medians | <= 0.10 m | 0.038 pass | 0.041 pass | 0 |
| negatives | offset on negative frames (m) | <= shipped + 0.3 | 0.743 (shipped 0.779) pass | 0.728 pass | 0.779 |
| hugsim | spins / stuck / completes of 11; HD-Score | spins not up | 0 / 4 / 4; 0.479 (-0.065 [-0.23, +0.10]) pass | 1 / 1 / 5; 0.537 (-0.006) FAIL (one spin) | 0 / 6 / 3; 0.544 |
| b2d_turns | window collisions | not up vs shipped | 5 pass | 8 FAIL | 5 |
| b2d_ds | DS, 19 routes paired | delta >= -7.6 | 66.8, -2.4 [-10.8, +6.6] pass | 62.5, -6.7 [-19.7, +4.9] pass (near the edge) | 69.3 |

rc-all (T6 drivable area + T4 + T5), asked separately: navhard DAC is flat (+0.32, CI includes 0) while navhard EP rises 3.9 pp (stage 2: 66.1 -> 70.3); navtest DAC +0.82 and EP +0.75 both rise. So T6 did not buy DAC with EP; the EP gain on navhard is the larger effect and DAC does not move there.
The navhard EPDMS gain (+1.90, CI includes 0) comes from EP, not DAC; the early-turn set (heading change >= 5 deg) still drops for both arms (rc-all -0.65 vs rc-bear -0.30, both CIs include 0 for rc-all). HUGSIM has no route plumbing, so both arms run there with zero bias; rc-all's one spin is a single scene. DAC / EP rows are per-token means of the harness's sub-scores (harness_tokens.csv), diff CI = group bootstrap over 225 mapping groups, 1000 resamples.

## Doubts

- One seed per arm, one run per cell, 25 turns (13 choice / 12 forced) on 20 routes: a difference of 2-3 turns is inside the noise (rc-bear vs rc-ctl: 2 gained, 4 lost). Shipped 1/25 is a single run of decision 127's unit.
- Desire still reaches the model in closed loop (decision 127 olnz config) while the trainer fed desire 0: the fine-tuned heads see a pulse they never saw; rc-ctl's side-correct steering suggests it drives the side information. A no-desire closed-loop arm would test the adapter alone; not run (the guard line fixes DESIRE = true to match the cached shipped run).
- CARLA training targets were timed at >= 4 m/s through the junction; real stop-and-go in the harness is about 1 m/s with 40% stops, a longitudinal regime the training data does not cover. The stop / crawl share is the same for shipped, so it is a harness property, but I did not find its cause (red lights vs lead traffic vs scheduler).
- Action supervision at approach poses is about zero by construction (open loop action sign agreement 0.53); this is the likely reason the command reaches the plan but not the steering in time. Not tested by an ablation (no new arms were allowed).
- The failure split is heuristic (first matching cause, thresholds in split.md); causes are not exclusive. Steering is read over the whole turn span, which for lost cars runs up to 25 m off the route, so "late" includes steering after leaving the road.
- rc-all's guard and DS numbers are from the same 20 routes / 11 scenes as the others, one run each; DS differences of +5 to +13 over shipped are not pre-registered.
- Open-loop CARLA exit readout counts "short" plans as wrong; shipped is 62% short there. CARLA exit set has no traffic.
- Guard bring-up notes: guard.py crashed on a missing logs directory (fixed, one line) and its drift / negatives / wod lines need the op-train python (`onnx`); those three lines for rc-bear / rc-all were run with `$DATA_DIR/envs/op-train/bin/python guard.py --lines drift,negatives,wod --force`.
