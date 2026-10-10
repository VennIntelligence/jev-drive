# Q4. Label supply for "what the rater prefers": 479 frames is all there is on WOD-E2E; no released box or map, no link to other Waymo data, and no setting where a computed label can be checked against rater scores

Written 2026-10-10, lane WODSCOUT. An inventory with counts, not a method. Pre-registration:
[plans/2026-10-10-wodscout-prereg.md](../plans/2026-10-10-wodscout-prereg.md). Code: `scripts/ws_fields.py` (full-set field scan, 13 s on 32
processes), `scripts/ws_val.py` (kinematic agreement). Tables: [q4_label_supply/](q4_label_supply/). Web claims were collected by a sub-agent
through a page summariser and are marked STATED (relayed from the primary page), INFERRED or NOT FOUND; nothing was downloaded.

## Answer

1. **WOD-E2E carries exactly the 479 rated frames.** Every frame of every shard was scanned (725 799 frames, 622 shards): 30 field paths are
   present (5 of them all zero: the timestamps and the per-image pose) and none of them is an agent box, a map feature, a lidar return or a pose. Decision 236's sample of 450 frames per split holds on the
   full set. Train has 0 rated frames in 415 663.
2. **No link to WOD Perception or WOMD.** No shared id, the timestamps and poses that could align the data are zeroed, the camera rig is a different
   one, and Waymo states it does not release location for WOD-E2E. Boxes and maps exist inside Waymo (the raters saw them) and are not released.
3. **A computed rater-like label on nuPlan cannot be validated against rater scores, because no released data has both.** WOD-E2E val has rater scores
   without maps or boxes; navtrain has maps and boxes without rater scores. The one public set that would have both (DriveCritic, 5 730 human-preferred
   trajectory pairs built on NAVSIM) is not released as of today.
4. **What can be measured without a map: no single kinematic quantity reproduces the rater ordering.** Over 1 429 within-frame pairs of rated
   trajectories with different scores, "the higher-scored one goes further in 5 s" holds in 56.2 % [53.1, 59.3], "is closer to the logged future" in
   59.8 % [56.9, 62.7]; braking and lateral acceleration are at chance, and jerk is reversed. None reaches the pre-registered 0.70.

## (a) What a WOD-E2E frame contains, on the full set

Schema-free walk of the protobuf wire format over all slim shards (the slim copy drops five of the eight cameras and nothing else,
`docs/waymo-e2e.md`); "non-zero" = the value is not all zeros (for a pose: neither zeros nor the identity). Full table:
[q4_label_supply/fields.md](q4_label_supply/fields.md), per split [fields_by_split.md](q4_label_supply/fields_by_split.md).

| | train | val | test |
|:--|--:|--:|--:|
| frames scanned (= frames in our index) | 415 663 | 106 360 | 203 776 |
| sequences | 2 037 | 479 | 1 505 |
| frames with a full logged future (20 steps) | 415 663 | 106 360 | 0 |
| frames with rated trajectories (score >= 0) | **0** | **479** (478 sequences) | 0 |
| intent straight / left / right | 349 956 / 35 837 / 29 870 | 90 681 / 7 968 / 7 711 | 178 079 / 15 378 / 10 319 |

| field path | populated | note |
|:--|:--|:--|
| `frame.context.name`, `frame.context.camera_calibrations` (8) | every frame | |
| `frame.images` (`name`, `image`, `velocity`, `shutter`) | every frame, 3 kept of 8 | `velocity` is non-zero: per-image vehicle velocity, the only motion field besides the ego states |
| `frame.images.pose`, `pose_timestamp`, `camera_trigger_time`, `camera_readout_done_time`, `frame.timestamp_micros` | present, **all zero** on every frame | no global pose, no time |
| `past_states` (pos x / y, vel x / y, accel x / y), `intent` | every frame | no `pos_z` in the past |
| `future_states` (pos x / y / z) | train and val, every frame; absent in test | |
| `preference_trajectories` | val only: 3 placeholders per frame, real positions on 479 frames | train and test: field absent |
| `frame.pose`, `frame.lasers`, `frame.laser_labels`, `frame.camera_labels`, `frame.projected_lidar_labels`, `frame.map_features`, `frame.map_pose_offset`, `frame.context.stats`, `frame.context.laser_calibrations` | **absent on all 725 799 frames** | no box, no map, no lidar, no weather / time-of-day / location tag |

The 479 rated frames sit in 478 sequences: one sequence has two and one has none (the same two sequences are named in the public issue #942 of the
waymo-open-dataset repository; STATED). Supervision per frame is therefore: images, calibration, 4 s of ego history, a 3-way intent, the 5 s logged
future (train and val), and on 479 val frames three scored trajectories. The scenario cluster tag exists for val sequences only; test has none locally
and train has none (see Q1).

## (b) Linking WOD-E2E to WOD Perception / WOMD

| check | finding | status |
|:--|:--|:--|
| ids | WOD-E2E: 32-hex sequence id + frame index. Perception: `<uint64>_<start>_<length>_<start>_<length>` (798 training segments of labels on the box, `datasets/waymo_perception/v2`). WOMD: its own scenario ids. No common key. | measured on the box |
| time and pose | Perception frames carry microsecond timestamps and poses; in WOD-E2E both are zero on all 725 799 frames, so nothing can be aligned by time or place. | measured |
| camera rig | Perception: 5 cameras, 1920 x 1280 (three) and 1920 x 886 (two). WOD-E2E: 8 cameras, 972 x 1079 / 972 x 587 / 972 x 551. A different sensor suite. | measured |
| proto | `E2EDFrame` has fields 1 (frame), 5-8 (future, past, intent, preference); the comment says only context name, calibrations, timestamp and images of `frame` are used. No field for boxes, maps or a foreign id. | STATED (github.com/waymo-research/waymo-open-dataset, `protos/end_to_end_driving_data.proto`) |
| paper | Mined from 6.39 M miles of driving logs; no statement of correspondence to Perception or WOMD segments; raters saw "annotations for all on-road agents" and map elements in an internal tool. | STATED (arXiv 2510.26125 v2) / INFERRED that these are fleet logs, not the public segments |
| maintainers | "The WOMD and WOD-E2E datasets do not include identifying location data at this time"; "no current plans to add city labels or other location data" (issue #963, 2025-07). | STATED |
| a later release with boxes, maps or train-split rater labels | nothing found; the 2026 challenge page could not be fetched (404). | NOT FOUND |

Conclusion: not linkable. The only route left would be matching by image content against Perception segments, which the different rig and the
mining source make pointless; it was not tried and needs no download.

Licence and rules (STATED from waymo.com/open/terms and the 2025 challenge page; relayed by a summariser, so re-read before relying on the wording):
non-commercial licence; for the vision-based end-to-end challenge "Submissions may be created using any public data made available to the
academic/research community", with data sources disclosed; "Submissions may not be created using any form of manual labeling" while automated
labelling (e.g. by MLLMs) is allowed; a reply by an author account on issue #945 says the val preference trajectories may be used as a training signal
(not an official rule). So nuPlan and WOD Perception / WOMD are admissible training data; hand-made rater-style labels of our own are not.

## (c) Rater-like labels computed on nuPlan (navtrain)

What exists on our side:

| source | unit and count | label it carries | reaches "rater preference"? | obstacle |
|:--|:--|:--|:--|:--|
| WOD-E2E val rater frames | 479 frames / 478 sequences, 3 scored trajectories each (1 437 scored trajectories, 1 429 ordered pairs) | human score 0-10 on model-sampled candidates | yes, it is the definition | exhausted: every route that trains on it ends at about +0.09 to +0.15 out of fold (decisions 171, 173, 175) |
| WOD-E2E train | 2 037 sequences, 415 663 frames | logged future, intent | no: the log is not what raters rank first (log RFS 8.13, decision 164); hindsight labels are negative on RFS (decision 173) | no rater score, no box, no map |
| WOD-E2E val, unrated frames | 105 881 frames in the same 479 sequences | logged future | no | same as train; the rated frame is one per sequence |
| WOD-E2E test | 1 505 sequences, 203 776 frames, 1 505 scored by the server | none locally | no | labels hidden; 6 submissions per 30 days; not a label source |
| navtrain (nuPlan) | 103 k frames (decision 93 / op_route_cmd) | map, agent boxes, traffic lights -> the PDM sub-scores (collision, drivable area, direction, progress, TTC, comfort) for any candidate trajectory | computable, never checked against a human | no rater score anywhere on nuPlan; a selector trained on such scores does not transfer to WOD RFS (decision 195: -0.0004 [-0.031, +0.030]; pick against the realised RFS optimum Spearman 0.11) |
| navtrain candidate scores already cached | 28 323 turn tokens with simulator scores of a 19-candidate family (decision 191) | EPDMS-type score per candidate | same as above | turn tokens only |
| DriveCritic (arXiv 2510.13108) | 5 730 trajectory pairs "sampled and constructed from NAVSIM" with expert pairwise preference (STATED on the project page) | human preference between two trajectories on scenes that have map and boxes | would be the missing overlap | not released: the project page says code "coming soon", the linked repository returns 404 (checked 2026-10-10); licence unknown |
| WOD Perception / WOMD | on the box: 798 label segments, 10 scene-flow records, 10 WOMD shards | boxes, maps (WOMD) | no | no rater score, no link to WOD-E2E, another camera rig |

**Agreement of a computed label with rater scores.** There is no setting where both exist, so it cannot be measured. Plainly: rater scores exist only
on WOD-E2E val, where there is no map and no box; map and boxes exist on nuPlan, where there is no rater score. The rater rubric (STATED, arXiv
2510.26125 section 3.4.3: start at 10; minus 2 per violation of safety, legality or reaction time; minus 1 for braking necessity or efficiency;
cumulative; candidates sampled from a Wayformer-type model, up to 64, three picked by the rater) has five dimensions of which two (braking,
efficiency) need no map. For those two the pre-registered substitute read is below; safety, legality and reaction time cannot be computed on WOD.

**Map-free kinematic quantities against the rater ordering** (no fitted parameter; pairs of rated trajectories in the same frame with different
scores; concordance = share of pairs where the higher-scored trajectory has the expected relation, ties count half; bootstrap over sequences).
Table: [q4_label_supply/kinematic_concordance.md](q4_label_supply/kinematic_concordance.md).

| the higher-scored trajectory ... | all 1 429 pairs | stopped frames (357) | moving frames (1 072) | pairs with the top-rated (958) | reading |
|:--|:--|:--|:--|:--|:--|
| goes further in 5 s (efficiency) | 0.562 [0.531, 0.593] | 0.571 [0.500, 0.644] | 0.559 [0.525, 0.591] | 0.545 [0.509, 0.578] | weak |
| is closer to the logged future (ADE) | 0.598 [0.569, 0.627] | 0.611 [0.552, 0.672] | 0.593 [0.558, 0.627] | 0.598 [0.562, 0.635] | weak |
| brakes less hard (min longitudinal accel) | 0.537 [0.508, 0.567] | 0.577 [0.518, 0.637] | 0.524 [0.490, 0.559] | 0.519 [0.481, 0.557] | weak to unrelated |
| has lower peak lateral accel | 0.502 [0.474, 0.532] | 0.459 [0.396, 0.521] | 0.517 [0.485, 0.550] | 0.513 [0.479, 0.548] | unrelated |
| has lower peak jerk | 0.380 [0.352, 0.408] | 0.445 [0.386, 0.504] | 0.358 [0.326, 0.391] | 0.339 [0.305, 0.375] | opposite |

Chance is 0.5; the pre-registered line for "moves with the rater ordering" was a lower bound of 0.70 and nothing comes near it ("weak" = lower bound
above 0.5). Progress, the direction decision 164 found for the top-rated trajectory against the log, orders only 56 % of rated pairs: among the
candidates a rater was shown, going further is not what separates the scores; the deductions are for things that need the scene (safety, legality,
reaction). Higher-scored trajectories have more jerk, which is mostly the 4 Hz second difference of model-sampled trajectories that accelerate.

## Feasibility summary

| route to more "rater preference" supervision | exists today | size | verdict |
|:--|:--|:--|:--|
| more rater frames on WOD-E2E | no | 0 beyond 479 | closed unless Waymo releases train or test labels |
| WOD-E2E boxes / maps to compute rubric terms on WOD | no | 0 | closed; pseudo-labels from our own detectors would be unverifiable (no truth on WOD; decision 231: agent AUC 0.589 across domains) |
| link to WOD Perception / WOMD | no | 0 | closed |
| computed rubric on nuPlan (map + boxes) | yes | 103 k frames, any number of candidates | available as a training signal; its agreement with raters is unmeasurable, and the one transfer test we have is null (decision 195) |
| human preference on NAVSIM scenes (DriveCritic) | announced, not released | 5 730 pairs | the only candidate for an overlap; watch for the release |
| map-free kinematic terms on WOD train (progress, comfort) | yes | 415 663 frames | computable, but the best two order rated pairs at 56 % and 60 %, the rest are at chance or reversed: not a stand-in |
| test server as a label source | 6 submissions per 30 days | 11 cluster means per submission | excluded by our own rule (decision 180) and too coarse |

## Limits

- Web statements passed through a summarising fetch tool; quotes are as relayed. The CVPR supplemental, the dataset page (login) and the 2026
  challenge page were not readable.
- The field scan reads the slim shards; the five dropped cameras are the only difference from the bucket by construction of the slimming step,
  which was not re-verified here.
- The kinematic read uses the three trajectories a rater chose out of up to 64 candidates: a selected, small candidate set, in sample, 479 frames.
- "Not released" for DriveCritic is the state on 2026-10-10.
