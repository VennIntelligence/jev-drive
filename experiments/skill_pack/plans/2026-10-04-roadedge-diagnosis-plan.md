# Why openpilot's road edge disagrees with NAVSIM's drivable area (day 2026-10-04, lane EDGE; pre-registration)

Written before any of the measurements below was run. Follows decision 88 (model edge > 1 m beyond the map at the
departing corner in 59% / 74% of navhard stage-2 DAC failures) and decision 103 (drivable area is the largest loss block
left: navhard 13.0, navtest 3.8 points). User's point: the gap shows only on the NAVSIM boards; suspect adapter, protocol
or scorer map, not the model.

## Facts read before measuring (code, no numbers)
- Adapter (`jevdrive/navsim_zs.py` OpenpilotMaps): road / wide 512x256 frames use openpilot's own MEDMODEL_K (f 910,
  cy 47.6) and SBIGMODEL_K (f 455), calib = NAVSIM ego axes (level, straight); rays go through the real CAM_F0 extrinsic
  rotation, K and Brown distortion. CAM_F0 sits 1.52 m above the ego origin (comma devices: ~1.2-1.4 m). The warp is a pure
  rotation, so the only geometric mismatch it can carry is camera height / position, not intrinsics.
- Scorer map (`navsim/.../pdm_occupancy_map.py` PDMDrivableMap.from_simulation): the cached map holds ROADBLOCK,
  ROADBLOCK_CONNECTOR interiors (lanes, connectors), INTERSECTION and CARPARK_AREA; nuPlan's generic DRIVABLE_AREA layer is
  left out ("TODO: Fix SemanticMapLayer.DRIVABLE_AREA problems"). DAC counts a corner as drivable inside ROADBLOCK,
  INTERSECTION, DRIVABLE_AREA (never cached) or CARPARK_AREA.
- Decision 88's edge statistic is measured at the departing corner of failing plans. A plan that leaves the polygon while
  staying inside its own edge implies an edge beyond the polygon there, so this number is conditioned on the outcome.

## Measurements (navsim2 env, CPU; cached native Cinque heads, no new inference unless stated)
M1. Per token, at ego x = 5, 10, 15, 20, 25, 30 m (rear axle frame) on the lateral cross-section through x:
  model lane lines (4, mean of the MDN) and road edges (2) moved into the ego frame with the CAM_F0 position; their z;
  map intervals: (a) scorer polygon (DAC layers, as cached), (b) the same plus nuPlan DRIVABLE_AREA (generic drivable
  areas, from the map API), (c) the lane polygons (LANE / LANE_CONNECTOR) and their boundaries; ego lane = the lane
  interval containing y = 0. Boards: navtest (12 146, real), navhard stage 1 (real) and stage 2 (synthetic).
  Valid lane section: ego lane interval exists, no INTERSECTION polygon on the section, model's two ego lines prob > 0.5.
  Valid edge section: the section's centre is inside the scorer polygon and no INTERSECTION on the section.
M2. Visual review sheet: CAM_F0 with the model's edges, the scorer polygon boundary and the DRIVABLE_AREA boundary
  projected on the ground plane, for random disagreeing sections (model edge > 1 m beyond the scorer boundary): >= 40 cases.
M3. Unconditional rate: share of all valid edge sections, and of DAC passes vs failures, with the model edge > 1 m beyond.
M4. CARLA reference: ego-lane width ratio from any existing B2D run that logged openpilot lane lines with map truth.

## Hypotheses and what decides each
H1 geometry / calibration. Confirmed if on navtest valid lane sections the median ratio (model ego-lane width / map
  ego-lane width) is outside [0.92, 1.08] at >= 3 of the 6 distances, or the median ego-lane centre offset is > 0.3 m,
  AND the same factor shows in the road-edge width. Error growing with x: |ratio - 1| at 30 m > at 10 m by > 0.05.
  Asymmetry: median outward error left minus right > 0.3 m. Also read: the model's lane-line z at 10-20 m vs 1.52 m
  (the camera height the model infers). Rejected if the ratio is within [0.92, 1.08] at all distances and the offset < 0.3 m.
H2 scorer map narrower than the road. Confirmed if (i) on navtest valid edge sections the model edge is beyond the scorer
  boundary by a median > 0.5 m on the side where the scorer boundary is the outer lane boundary, and (ii) adding
  DRIVABLE_AREA moves the boundary outwards to within 0.5 m of the model edge in >= half of the disagreeing sections, or
  (iii) the review sheet shows the model edge on paved surface (shoulder, parking strip, curb lane, driveway) in >= 50%.
  Rejected if the model edge is within +-0.5 m of the scorer boundary in median on straight sections (then the
  departure-corner number is selection, see H5).
H3 protocol (synthetic history / GIMM frames / 2 Hz). Confirmed if the median outward edge error on navhard stage 2 exceeds
  stage 1 by > 0.5 m, or the lane-width ratio differs by > 0.05. Interpolated vs non-interpolated previous frame: a GPU
  re-run on ~300 tokens with the previous frame = the current frame, only if H1-H3 leave it open.
H4 trajectory, not edge. Among DAC failures (navtest, navhard stage 1 / 2): share where the plan's lateral position at the
  departure time is within 0.5 m of the model's own ego-lane centre (plan follows its lane; the lane is off the map) vs
  > 0.5 m off it (plan leaves its own lane: corner cutting / overshoot). Plan-off-own-lane > 50% means the edge is not
  the cause.
H5 selection (added after reading decision 88's definition). Confirmed if the rate of "edge > 1 m beyond" on DAC passes
  is within 10 points of the rate on failures at the same sections.

## Fix rule (only if H1 or H2 is confirmed)
H1: correct the geometry in the adapter (e.g. a virtual camera at a comma height, or a lateral rescale of the plan by the
  measured factor fitted on navtrain), pre-registered in an addendum before scoring.
H2: no model fix exists; an adapter margin (keep the plan's lateral offset from the model edge >= m, m fitted on navtrain)
  is a labelled per-board trick. Score navhard two-stage EPDMS and navtest PDMS on CPU with the official scorers;
  adopt if navhard combined delta > 0 with CI lower bound > 0 and navtest delta >= -0.30 (as lane TRK).

## Addendum 1 (after M1 / M2, before any fix was scored)
What M1 / M2 showed: on real navtest frames the model's ego lane is 0.67-0.70 of the map lane and its road 0.72-0.77 of the
scorer road at every distance, symmetric; its edges sit inside the scorer polygon (median -0.6 to -2.0 m); the plan speed at
t0 is 0.81 of the logged speed. The NAVSIM ego origin is the rear axle at axle height: nearby vehicle boxes end at a median
z = -0.35 m (9 781 boxes, 40 test logs), so CAM_F0 is ~1.87 m above the road, not 1.52. Projected with the ground at -0.35 m
the map lane boundaries land on the painted lines, and the model's lane lines taken as camera rays land on the same pixels;
the model's own lane-line z puts the camera 1.34 m above the road. Reading: the model sees the image right and scales the
metric world by about 1.34 / 1.87 = 0.72 (a roof camera is outside openpilot's windshield-height training range). H1 holds
as a camera-height scale error, not a warp error; H2 is rejected (the model is narrower than the map, not wider).

Fix arms (native Cinque only; N4 picks among anchors and is not rebuilt). Per token, reference-free:
  h_model = cam_z - median over the two ego lane lines and the knots with 5 <= x <= 30 m of the model's z (calib frame);
  h_true = cam_z + 0.35;  k = h_true / h_model  (clipped to [1, 2]).
  U(k): the 33 plan positions (calib frame, camera origin) times k, yaw unchanged, then the usual lever-arm conversion to
        the 8 rear-axle poses. The inverse of a uniform scale error: path and speed both scaled.
  P(k): path geometry times k, but each knot keeps its original arc length along the path (speed profile unchanged);
        yaw from the scaled path's tangent.
  Constants k in {1.15, 1.30, 1.45} for both families, and the per-token k ("own").
Selection on navtrain only (official v1 PDMS, the 3 000-token lb_navtrain subset): per family the arm with the largest
PDMS delta vs native (none if no delta > 0). Scored on navhard two-stage EPDMS and navtest PDMS (official scorers): U(own),
P(own) regardless (the principled arms) and the navtrain pick per family if different. Adoption rule as in the main plan
(navhard combined delta > 0 with CI lower bound > 0 and navtest delta >= -0.30; a navtest gain with CI lower bound > 0 counts
on its own). Labelled as a NAVSIM rig adapter: it uses the known rig height, not a scorer quantity.
