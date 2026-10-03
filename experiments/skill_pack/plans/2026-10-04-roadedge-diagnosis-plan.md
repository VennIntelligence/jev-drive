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
  h_model = median over the two ego lane lines and the knots with 5 <= x <= 30 m of the model's z (calib frame, the height
            of the road below the camera as the model reads it; wording fixed after the run, the code did this);
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

## Addendum 2 (intervention, before it ran; navtrain picks of addendum 1 were known: U1.15 +0.38, P1.15 +0.14, U-own -3.82)
Show the model a lower camera. Seed-0 random 400 navtest tokens; keys rendered from CAM_F0 with the ground-plane homography
of a camera lowered from h_true to h_true / r (rays below the horizon sample CAM_F0 at r times their depression slope; above
the horizon unchanged); the 6 context-rate frames by the CPU ego-motion warp at the virtual height for every arm (so arms
differ only by r); r = 1.0 (control) and 1.44 (virtual camera ~1.30 m). Script edge_height.py, card 1, a few minutes.
Readouts on the same tokens: ego-lane width ratio on valid lane sections, road width ratio vs the scorer polygon, plan speed
at t0 over the logged speed, the model's lane-line z.
Confirmed (camera height sets the metric scale) if lane-width ratio(r 1.44) / ratio(r 1.0) >= 1.25 and the speed ratio
moves the same way; rejected if < 1.10. If confirmed, the image-side adapter is the candidate fix; scoring it needs
GIMM frames re-made at the virtual height for both boards (GPU hours), outside this lane's budget: it is reported as the next
step with its cost, not run here.

## Addendum 3 (scored fix, written before any scored run of it; lane EDGE-FIX)
Known before writing: addendum 2 (400 navtest tokens: lane ratio 0.69 -> 0.97, speed 0.84 -> 1.00 at 1.30 m); the output-side
rescale arms (all negative on the test boards); navtrain warp (road plane 1.52 m) 81.17 vs GIMM 82.12 (shipped runs). No PDMS of
any virtual-camera input has been read.

Pipeline (`scripts/op_lb.py run --vcam H`, `--frames vh<cm>`): no frame cache. Per token the 4 CAM_F0 keyframes are rendered for
a virtual camera H m above the road at the same x, y and axes (ground homography: a model ray below the horizon samples CAM_F0
along (x, y, r z), r = h_true / H, h_true = CAM_F0 z + 0.35 m; rays above the horizon unchanged); the 6 context-rate frames are
the CPU ego-motion warp of those keys with the road plane H below the camera. Control `vh187` = r 1 (the true height, the same
code; its keys are bit-identical to the shipped frame cache, checked on 2 tokens), so every comparison below differs only in H.
The history frames are therefore at the same virtual height by construction. Plans are in metric ego coordinates of the real
world (the virtual camera sees the same ground points), exported by the usual lever-arm conversion. Cinque, zero state,
desire none, op_lb step schedule, as the shipped runs.

Height choice, navtrain only: H in {1.22, 1.30, 1.40} and the control vh187 on the lb_navtrain 3 000-token subset, official v1
PDMS (`v1_navtrain_oplb` cache). H* = the H with the largest navtrain PDMS delta vs vh187 (taken even if no delta is > 0; then the
fix fails its line on navtrain already and the test scores are read as confirmation only).

Arms (test boards: navtest v1 PDMS n 12 146, navhard two-stage v2 EPDMS n 5 912, official scorers):
- A0 `vh187` native Cinque (control); A1 `vh<H*>` native.
- A2 = A1 + tracker pre-compensation, decision 97's `path` variant (`trk_precomp.py --modes path`), alpha in {0.25, 0.5, 0.75, 1}
  refitted on navtrain against `vh<H*>` native (largest navtrain delta, taken even if negative). A0T = A0 + the same alpha
  (interaction: does pre-compensation only pay once the plan is correctly scaled).
- B0 / B1: it_dw3-s0 serving ONNX at vh187 / vh<H*>; B0s / B1s: + decision 94's selector (rot0 rollout at the same height,
  ratio fixed at 0.6, not refitted).
Lines (paired; navtest per-token bootstrap, navhard bootstrap over the 225 scene-mapping groups, 5 000 draws, 95% CI):
- Fix (A1 vs A0), primary: adopt if navhard combined delta > 0 with CI lower bound > 0 and navtest delta >= -0.30; a navtest gain
  with CI lower bound > 0 counts on its own (as addendum 1).
- TRK (A2 vs A1): the same line.
- Adapted model (B1s vs B0s): the same line. Gains add if the interaction (B1 - B0) - (A1 - A0) has a CI that contains 0 or is
  > 0 on both boards and B1 > B0; sub-additive if its CI is < 0 on a board.
- Context only (pipeline differs): every arm against the shipped GIMM rows (84.18 / 33.33; it_dw3 + selector 84.70 / 35.76).
Frame-interpolation check: vh187 vs the shipped GIMM run on all three splits. The cheap warp path "matches" if |navtrain delta|
<= 0.5 and |navtest delta| <= 0.5. If not, GIMM context frames at H* are made for navtrain only (cost ~1.2 GPU-s/token) and
vh<H*>-GIMM vs shipped GIMM is scored there, to say whether the fix gain depends on the interpolator.
Side checks (CPU, existing dumps, no new runs): HUGSIM and CARLA / B2D camera heights above the road and openpilot's lane-width
ratio there; whether Alpamayo / N-series NAVSIM inputs take the same CAM_F0 geometry (stated from code).

## Addendum 4 (scope extension: one virtual rig with switches; written after the navtrain height sweep, before any test score)
Read so far (navtrain only, official v1 PDMS, 3 000 tokens): control vh187 80.29; vh122 78.32, vh130 80.20, vh140 81.01, so
H* = 1.40 m (+0.72). The warp control is 1.83 below the shipped GIMM run (82.12), so the frame-interpolation check of addendum 3
has failed its 0.5 line and its conditional GIMM arm runs (below). No navtest / navhard number of any vh arm has been read.
navhard vh187 native was killed (out of memory: orphaned render workers of the --vcam pool, fixed in c82d070) and is re-run.

The user's rig points (wide too narrow, horizon wrong) checked against openpilot's own model-frame definition before any arm:
- openpilot's model inputs are fixed calibrated frames (common/transformations/model.py, ported in jevdrive/openpilot/frames.py):
  road f 910, cy 47.6 (hfov 31.4 deg, horizon at 18.6% of the rows, 3.0 deg up / 12.9 deg down); wide f 455, cy 151.8 (hfov 58.7
  deg, horizon at 59.3% of the rows, 18.5 deg up / 12.9 deg down). The 120 deg physical wide camera is cropped to 58.7 deg by
  modeld; "wide = mostly sky, road = mostly ground" is openpilot's own layout. Our frames use the same K with calib = the level
  ego axes, so the horizon rows are 47.6 / 151.8 by construction.
- CAM_F0 (f 1545, hfov 63.7, vfov 38.5) covers 100% of both model frames at every height and pitch used here (measured on the
  ray maps); CAM_L0 / CAM_R0 add zero pixels to a 58.7 deg frame. The 3-camera wide switch is therefore a no-op for openpilot's
  input and is not run (stated, not scored). A stitched wide would only matter for a model whose wide frame is wider.
- Implied pitch from the model itself (navtest / navtrain heads): lane-line z slope over 5-40 m gives -0.15 to +0.03 deg;
  road_transform pitch -0.27 to -0.33 deg (< 5 px in the road frame). No evidence of a horizon error.
Horizon switch anyway (as asked), chosen on navtrain only: virtual pitch in {-1, -0.3, +0.3, +1} deg (> 0 = camera up, horizon
lower) at H = 1.40, warp context frames as the keys (the warp assumes a level road frame; at <= 1 deg the error is small, stated).
P* = largest navtrain delta vs vh140; the rig "height + horizon" = vh140 + P* is scored on the test boards regardless (factor
arm); the "best rig" for the TRK / it_dw3 arms is vh140 + P* if its navtrain delta > 0, else vh140.
Test arms (otherwise as addendum 3): A0 vh187, A1 vh140 (height), A1h vh140 + P* (height + horizon), on the best rig: A2 + TRK
(path alpha refit on navtrain), B1 / B1s it_dw3 (+ selector 0.6); controls A0T, B0, B0s. Lines and contrasts as addendum 3; the
factor contributions are A1 - A0 (height) and A1h - A1 (horizon), paired.
GIMM check (conditional arm of addendum 3, now triggered): keys at 1.40 m for lb_navtrain rendered to a run dir, GIMM context
frames on card 0, Cinque rollout, official navtrain PDMS vs the shipped GIMM run (82.12): says whether the height gain holds with
the shipped interpolator. Test boards stay on the warp pipeline (GIMM there costs ~1.2 GPU-s/token, ~6 GPU-h per arm).
Other boards (CPU, no reruns): camera height, horizon row and wide-frame coverage of our HUGSIM and CARLA / B2D renders.
