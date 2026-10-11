# FAMILY task 1: what each nuPlan-track driver reads from the per-scene / per-vehicle request metadata (2026-10-12)

Code read: AlpaSim 0bb4c4b on the box (`$DATA_DIR/third_party/alpasim`), its MTGS renderer plugin (`plugins/mtgs/server/`), the shipped
SimScale samples (`e2e_challenge/sample_submission_simscale_navsim_{transfuser,diffusiondrive}`), `py123d_garage` v0.1.1 (288d34c), our
served driver (`experiments/alpasim/lib/sh30_driver.py`, `sh30_core.py`, `ap2_driver.py`, `jevdrive/navsim_zs.py`, `jevdrive/op_interp.py`).
Numbers per public asset: [asset_calib_public.csv](asset_calib_public.csv) (`scripts/asset_calib.py`, all 1491 assets).

## Where the calibration comes from, and what it omits (new)

- The runtime asks the renderer (`get_available_cameras`); MTGS answers from the asset's `video_scene_dict.pkl`, `frame_infos[0].cams`,
  preferring `colmap_param.cam_intrinsic` (`artifact_adapter.py:get_available_cameras_from_data_source`). It fills K (fx, fy, cx, cy),
  resolution, and the camera pose in the rig (`rig_to_camera`, which despite its name is camera -> rig). **It never fills the distortion
  coefficients.**
- The renderer renders a pinhole image at `getOptimalNewCameraMatrix(K, D)` and then applies `initInverseRectificationMap(K, D, ...)`
  (`server/engine/mtgs.py:set_sensors`, `render`): the JPEG is **barrel-distorted with the asset's colmap D**, like a raw nuPlan frame.
  Public assets: k1 -0.42 .. -0.12 (median -0.338), k2 -0.07 .. 0.20 (median 0.113); raw nuPlan D (-0.356, 0.1725, ..., k3 -0.052).
  Checked on one stored render (veh-35 0c318d79, k = 0) against the real nuPlan frame of the same instant: it overlays the distorted real
  frame (edge NCC 0.37, periphery 0.40) and not the undistorted one (0.31 / 0.33): [../figs/render_vs_real.jpg](../figs/render_vs_real.jpg)
  (left: render blended with the real frame, sharp; right: with the undistorted real frame, doubled edges at the left border).
  AUD1 / AP2 wrote "MTGS renders an undistorted pinhole camera": they read the distortion from the request, where it is always zero. Wrong.
- The two code paths unwrap the pickle differently: rendering accepts a flat or a per-scene-nested dict, the camera report only a nested
  one. A flat asset would be rendered correctly while the driver is told the adapter's fallback camera (fx 1920, fy 1080, cx 960, cy 560,
  identity rotation = optical axis up, t = 0). All 1491 public assets are nested; nothing is known about the private ones.
- Production: per the e2e_challenge config README the nuPlan base config overrides the renderer to `alpasim-mtgs-server` from the trusted
  base image (aud1_web_research.md); the adapter code above is what the public image ships, the production build was not seen.

## Dependence table

| request field | our openpilot driver (sh30 / ap2) | when missing / out of range / other convention | LTF / DD samples | garage 0014 |
|---|---|---|---|---|
| CAM_F0 present | required | FAILED_PRECONDITION at start_session, rollout fails | L0 / F0 / R0 required, same | announced cams mapped by name; missing -> static rig file |
| camera model | `opencv_pinhole` only; rational / thin-prism rejected | FAILED_PRECONDITION | not read | f-theta only (PAI); pinhole announcement ignored |
| K (fx, fy, cx, cy) | **used**: every model pixel's ray is projected into CAM_F0 through K (`OpenpilotMaps`) | any value accepted; a wrong K samples the wrong pixels silently; per-scene maps cached by K / R / D | not read: fixed pixel crops `[28:-28]`, sides `[28:-28, 416:-416]`, resize 2048 x 512 | not read on nuPlan |
| distortion | **used if given**; the evaluator gives none, so the driver treats a distorted render as pinhole (all scenes, public and private) | zero -> assumed pinhole; non-monotonic polynomials are cut | not read; trained on raw (distorted) frames, so the render matches training | not read |
| mount rotation R | **used**: full rotation in the remap (pitch, roll, yaw all reach the frames) | any value accepted silently; convention camera -> rig assumed; the transposed convention would double / flip the pitch (~3 deg) | not read | not read on nuPlan |
| mount translation t | t.z = camera height for the road-plane warp of synthesised history slots; t.x, t.y = lever arm camera -> rear axle of the plan | t = 0 (fallback camera): plane warp degenerate | not read | not read |
| image size | hard-coded 1920 x 1080 (`NUPLAN_WH`, flat-index sampling) | smaller: IndexError, every `drive` aborted; larger: silently wrong rows | asserts 1920 x 1080, else INVALID_ARGUMENT | reads the decoded size |
| ego box / vehicle dims | only with serving switches (off in both images) | not used | not read | static config (wheelbase 3.089, rear axle -> centre 1.461), not from the request |
| route | 4-way command by the shipped rule (first waypoint >= 5 m; y > 2 left, < -2 right); waypoints not fed (non-route arm) | NaN / short / behind / empty all reduce to a command (AUD1) | same rule | one target point 45 m along the route, with bridge / pad |
| egomotion / DynamicState | velocity rotated when it matches the pose difference better (the -yaw quirk at k = 0); 17 ms companion pose needed | one start pose: k = 0 plan off by tens of degrees (AUD1) | DynamicState as given | rear-axle pose, DynamicState |
| timing | 0.5 s key grid, exact frame / pose timestamp equality | other step: always cold; 1 ms offset: every `drive` aborted | own buffers | own buffers |
| left-hand traffic | flag exists, never served (`SH30_LHT=0`) | Singapore scenes served right-hand | none | none |
| frame convention | rig frame in, rig poses out anchored in the local frame (shipped LTF functions) | — | same | same |

Read: on the nuPlan track **only our driver consumes the camera calibration**; the three comparison drivers read raw pixels with fixed crops
and would be untouched by any calibration error, while a change of the rendered lens itself would hit them and not (much) us.
