# Pair inventory for the layer-3 adaptation (2026-10-03, lane D)

Inventory only: no training set was built. It counts the raw material for the three pairing types of the overall plan
(different command on the same frame; different history, same plan; offset start, recovery) per source. Everything ran on
the box on CPU (navtrain 20 workers, CARLA 16 workers, WOD 1 process).

Scripts (all in `experiments/op_common_cause/scripts/`) and raw outputs (`results/*.json`, same directory tree):

| Source | Script | Env | Output |
|---|---|---|---|
| navtrain | `pair_inv_navtrain.py` | `envs/navsim2` (nuPlan map API) | `navtrain_summary.json` (per-frame table stays on the box: `processed/op_common_cause/pair_inventory/navtrain_frames.parquet`) |
| WOD-E2E | `pair_inv_wod.py` | `envs/jevdrive` | `wod_summary.json` |
| CARLA / B2D | `pair_inv_carla.py` | `envs/carla` (`carla.Map` built from the xodr, no server) | `carla_summary.json`, `carla_traversals.json` |

## Result in one table

| Source | Type 1: alternative command from a map | Type 2: history perturbation (>= 2 s history) | Type 3: heading offset |
|---|---|---|---|
| navtrain (103 288 frames, 1192 logs, 10 550 scenes) | **11 310 frames / 2 482 approach segments / 2 554 scenes / 744 logs / 434 distinct junctions** (13 529 frame x alternative-class pairs); strict (>= 2 m before the lane end) 8 090 frames / 2 036 segments | 103 083 frames (4 s: 102 282) | 103 288 front-camera frames |
| WOD-E2E (train + val, 522 023 frames, 2 516 sequences) | **none**: no map, no route; only the per-frame `intent` | 470 469 frames (4 s: 419 688) | 522 023 front-camera frames |
| CARLA B2D (438 routes of the two route files, 47 km) | 226 junction traversals on 226 routes; 179 with an alternative branch in the map; **existing logs: 165 traversals recorded, 131 with alternative** (216 955 sampled ticks at 2 Hz, 12 recordings per route, no images) | 11.2 M ticks at 20 Hz in 5 489 recordings; no camera frames stored | no images on disk |

Short reading: navtrain is the only source with real images and a map; it yields ~2.5 k distinct approach segments (about 11 k frames at
2 Hz) with at least one alternative branch. WOD cannot produce type 1 at all. CARLA has few junctions per route and no stored images; type 1 there
means new closed-loop rollouts, not reuse.

## Definitions

- **Junction approach** (navtrain): the ego pose is matched to a nuPlan LANE (polygon contains the point; if several, the one whose
  baseline heading is closest to the ego yaw, rejected if > 60 deg off). Going downstream along that lane and through single-exit connectors
  (at most 4 lanes), the first lane with >= 2 distinct exits is the branching point. The frame is an approach frame when the arc length from ego
  to that branching point is <= **30 m**. Frames already inside a connector polygon (or within 1 m of the lane end while inside one) are `in_junction`
  and are not counted.
- **Branch class**: heading change of the connector baseline (end heading minus lane end heading): left > +25 deg, right < -25 deg, U-turn beyond +-135 deg, else straight.
  A junction approach has an **alternative** when the lane offers >= 2 classes (two exits with the same class, e.g. a fork of two straight lanes, do not count).
  The teacher trajectory for an alternative is the baseline path lane -> connector -> next lane (a map centreline, speed profile still to be chosen).
- **Taken branch**: which connector of that branching lane the ego actually enters in the next 20 frames (10 s). Connector polygons overlap,
  so the connector is picked by heading alignment (mean |dyaw| < 30 deg). A frame is **pair-eligible** when the taken branch is known and >= 1 other class exists
  (11 310 of 13 649 frames with an alternative; the other 2 339 never enter a connector within 10 s, e.g. the car stands still). The logged `driving_command` agrees
  with the taken class in 78% of pair-eligible frames (the logged command looks further ahead than this junction), which is the sanity check of the branch detection.
- **Segment**: one (log, branching lane) group of consecutive approach frames; median 4 frames (2 s). **Distinct junction**: (map, branching lane id).
- **Exam exclusion**: only `navsim_logs/trainval` logs of the navtrain filter (1192 of its 1200 log names are on disk, 103 288 of its tokens) are read.
  navhard synthetic scenes and the test logs are never opened.
- **Speed**: ego speed from `ego_dynamic_state` (navtrain), `past_states` last velocity (WOD), `v` in `ticks.jsonl` (CARLA, abs value).
- **History-perturbation eligibility, N = 2.0 s** (4 s also reported): the frame has >= N s of consecutive earlier frames of the same log / sequence
  (navtrain: 4 earlier frames at 2 Hz; WOD: 20 earlier frame indices at 10 Hz, exact steps of 1; CARLA: `t >= N` in the tick file at 20 Hz). The perturbations
  (repeat the current frame, drop frames, fake yaw) need the history images, so the frame must have them. WOD `past_states` (16 x 4 Hz, always present) also allows kinematic-only perturbations on all frames.
- **Heading-offset eligibility**: any frame with the front camera; a pose offset changes only the projective warp, so no history is required. Whether
  a fixable lane-level geometry exists for the lateral part is not checked here.

## 1. navtrain

Frame status (103 288 frames): `in_junction` 41 372, `no_junction_30m` 38 673, `branch` 22 932 (any branching lane within 30 m; 947 logs, 4 452 segments),
heading mismatch 158, off map 153. Of the 22 932 branch frames, 13 649 have >= 2 classes (alternative available; 779 logs, 2 885 scenes, 2 619 segments, 442 junctions),
and 11 310 of those are pair-eligible.

Pair-eligible frames by junction type (class set of the branching lane's exits), counted over the 13 649 alternative frames:

| Exits | Frames | Segments | Distinct junctions |
|---|---:|---:|---:|
| straight + right | 5 196 | 1 084 | 152 |
| straight + left | 3 408 | 749 | 145 |
| left + right | 2 514 | 350 | 59 |
| left + straight + right | 2 456 | 424 | 83 |
| with U-turn (4 sets) | 75 | 12 | 4 |

Pair-eligible frames by taken branch (the "from" command): straight 5 390, right 3 623, left 2 296, U-turn 1.
Segments by taken branch: straight 1 235, right 763, left 483.
Frame x alternative pairs (13 529): by alternative class left 5 016, right 4 763, straight 3 681, U-turn 69;
by taken -> alternative: straight->left 3 081, straight->right 3 286, right->straight 2 183, right->left 1 934, left->straight 1 497, left->right 1 476, left->U 52, right->U 15.

By ego speed (m/s), pair-eligible frames / alternative pairs / segments (median speed of the segment):

| Speed | Frames | Pairs | Segments |
|---|---:|---:|---:|
| < 1 | 2 899 | 3 914 | 581 |
| 1-3 | 3 207 | 4 054 | 637 |
| 3-6 | 2 815 | 2 985 | 595 |
| 6-10 | 2 081 | 2 251 | 565 |
| 10-15 | 306 | 323 | 103 |
| >= 15 | 2 | 2 | 1 |

Distance to the branching point (alternative frames): quantiles 5/25/50/75/95% = 0 / 2.2 / 6.5 / 17.4 / 27.1 m. Many frames sit right at the stop line
(8 090 frames are >= 2 m away). No single log contributes more than 0.9% of the pair-eligible frames.

Not counted (upper bound, tier B): frames whose own lane has a single exit but whose roadblock holds lanes with other classes (needs a lane change inside 30 m):
11 320 more frames (705 logs). A teacher for those is a lane change plus a turn; treat as a stretch option, not as clean material.

Caveats: a map centreline is not what a driver does at the junction (speed, stop, yield); the teacher for "another command" would be geometry only.
Navtrain cameras are 2 Hz, openpilot wants a 20 Hz history, so history perturbation on navtrain works on the 2 Hz history the adapter already builds.

## 2. WOD-E2E

WOD-E2E has no map, no route and no lane graph: the frame carries `intent` (straight 440 637, left 43 805, right 37 581 over train+val frames with futures), the
rear-axle ego past (16 x 4 Hz) and future (20 x 4 Hz), plus 3 front cameras. A different-command teacher trajectory cannot be generated. What exists:

| Quantity (train + val, with futures) | Frames | Sequences |
|---|---:|---:|
| all frames | 522 023 | 2 516 |
| history >= 2 s (20 exact earlier indices) | 470 469 | |
| history >= 4 s | 419 688 | |
| low speed (< 3 m/s) and history >= 2 s | 165 066 | |
| intent = turn (left or right) | 81 386 | 1 424 |
| not turning yet, turns within 3 s (`pre_onset`, the nearest thing to an approach) | 8 296 (hist >= 2 s: 7 660) | 994 |
| already turning (`turn_yaw`) | 54 393 | 1 313 |

`pre_onset` by speed: <1 118, 1-3 3 104, 3-6 2 845, 6-10 1 882, 10-15 332, >=15 15; by intent: straight 2 719, left 2 765, right 2 812.
Frames by speed (all): <1 113 958, 1-3 67 173, 3-6 111 145, 6-10 141 076, 10-15 64 486, >=15 24 185.
The index covers all 622 shards on disk (train 263, val 93, test 266; test has no futures and is excluded). The only way to get map-derived pairs near this data would be
Waymo Motion (`datasets/womd/validation_interactive` is on the box, 2.5 GB, vector maps) but it has no cameras and no scene overlap with WOD-E2E, so it does not give image pairs.
A heuristic alternative for type 1 on WOD (future ego trajectory of other frames of the same intent class) is not a different command on the same scene, so it is not counted.

## 3. CARLA / Bench2Drive

Route sources: `bench2drive220.xml` (220 routes) and `bench2drive_0.0.4_val.xml` (220, 2 shared) -> 438 distinct routes; the recordings use ids from both files. The XML
keypoints are sparse and can skip a junction, so each route was densified with the leaderboard's `GlobalRoutePlanner` (1 m) on the `carla.Map` built from the town xodr
(checked against a recorded `route.json`: route 26872, 82.8 m, same junction 4901, same left turn). Junction roads are taken from the xodr (`road@junction != -1`);
branches from the junction's `connection`/`laneLink` entries for the route's entry lane, class from the connecting road's net heading change.
CARLA yaw is clockwise-positive, which is handled in `cls_of`.

Routes per town: Town12 233, Town13 88, Town11 17, Town15 17, Town03 16, Town04 18, Town05 15, Town06 9, Town07 8, Town01 6, Town10HD 6, Town02 5. Total 47.4 km (median 78 dense points).

| Quantity | Count |
|---|---:|
| routes with a clean junction traversal (route does not start or end inside it) | 226 (each has exactly one) |
| traversals by branch taken | left 109, right 61, straight 56 |
| traversals with a different-class branch in the map for the same entry lane | 179 (left->straight 108, left->right 88, right->straight 57, right->left 43, straight->right 11, straight->left 8) |
| distinct junctions / entry lanes visited | 172 / 198 |
| junctions visited by >= 2 routes | 38 |
| entry lanes where >= 2 existing routes leave by different branches | **4** (9 routes; left/right x2, left/straight, right/straight) |

So "same junction, different route" already in the route files is almost nonexistent (4 entry lanes, 9 routes). The type-1 material in CARLA is the map itself: 179 traversals
where a rerun with a different route would give the other command, which needs new closed-loop rollouts (and cameras on).

Existing recordings: 5 489 `ticks.jsonl` files under `~/data/runs` (20 skipped without ego truth), 11.2 M ticks at 20 Hz, 292 distinct routes, median 12 recordings per route
(replicates of one agent/arm/seed matrix, not independent scenes). The files are closed-loop logs with ego pose, speed and brake/throttle; **no camera frames are stored**
(`frames/` empty, a few mp4/jpg for debugging). Matching ego positions to the dense route (within 5 m, sampled every 10th tick = 2 Hz):

| Quantity | Count |
|---|---:|
| approach ticks (route distance to junction entry 0-30 m), 2 Hz | 243 624 on 165 distinct traversals (165 routes) |
| of which with an alternative class in the map | 216 955 on 131 traversals (left 133 799, right 71 029, straight 12 127) |
| ... with >= 2 s / >= 4 s history | 207 567 / 198 195 |
| by speed (all approach ticks) | <1 205 597, 1-3 20 206, 3-6 15 684, 6-10 1 988, 10-15 148 |

84% of approach ticks are below 1 m/s: the recordings are dominated by agents waiting at the start / brake-held, and 12 replicates inflate every tick count.
The number that matters is the 131 recorded traversals (with alternative), not the ticks. For type 2 and 3 on CARLA the recordings are usable only as trajectories;
images have to be re-rendered with the same route and seed (the closed-loop infrastructure does it; `carla_rewind` is not equivalent, see decision 65).

## 4. History-perturbation and heading-offset eligibility per source

| Source | Frames with front camera | History >= 2 s | History >= 4 s | Low speed (< 3 m/s) and >= 2 s |
|---|---:|---:|---:|---:|
| navtrain | 103 288 | 103 083 | 102 282 | 33 716 frames below 3 m/s (history complete for all but 205 frames) |
| WOD-E2E train + val | 522 023 | 470 469 | 419 688 | 165 066 |
| CARLA recordings | 0 stored | 11.2 M ticks (trajectory only), images missing | | |

navtrain speed histogram (all frames): <1 13 852, 1-3 19 864, 3-6 35 792, 6-10 26 561, 10-15 7 181, >=15 38, so the data is a low-speed urban set. The navtrain
history rule is the dataset filter itself (4 history frames at 0.5 s); the 205 frames below 2 s sit at the start of a log pickle.

## Data gaps

1. WOD-E2E: no map or route, so type 1 is impossible there; only intent-labelled frames and history/offset pairs exist.
2. CARLA recordings have no camera frames; every CARLA pair needs a new rollout with sensors (cost and cores to be planned; not done tonight).
3. B2D routes are short (median 78 m) with at most one junction, and only 4 entry lanes are shared by routes with different exits.
4. Alternatives in navtrain are same-lane alternatives only (tier A); lane-change-needed alternatives (tier B, 11 320 frames) are counted but not clean.
5. Teacher trajectories were not generated: only the existence of an alternative branch and its centreline was checked. Speed profiles, stop behaviour at junctions and
   the effect of traffic lights / agents on the alternative path are open.
6. navtrain taken-branch inference uses the next 10 s of the same log; 2 339 alternative frames have no entry within that window and are not counted as pairs.
