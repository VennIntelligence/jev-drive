# Render probe: what breaks the PAI image when the human falls behind the log (2026-10-22)

**Question.** The user reports that when they fall behind the human log, even on the logged path, the whole rendered image
(road, background, other cars) turns into "shattered glass". Is that the time, the pose or the actors?

**Setup.** `scripts/render_probe.py` reads the user's rollout ASL (run `exam_pai_1022-154753`, scenes 213dfdac and 071d15c4),
takes the runtime's own front-camera `render_rgb` request at t = 3 s and sends edited copies to a renderer started from the
stack's NRE image (`nre-ga:26.04`, same command line, unharmonised) on Tokyo card 1. Same camera, intrinsics, 1920 x 1080,
shutter, JPEG quality and ego mask; the camera pose is the log's ego pose composed with the request's rig_to_camera. Frame
check: on the first frame (force-GT replay) the runtime's ego pose equals the log's to 1e-8 m.

| Condition | Ego pose | Frame time | Actors |
|---|---|---|---|
| log | log(t) | t | at t |
| log_late_D | log(t + D) | t + D | at t + D |
| lag_D (what the runtime sends when the human is D s behind on the path) | log(t) | t + D | at t + D |
| lag_time_D | log(t) | t + D | at t |
| lag_actors_D | log(t) | t | at t + D |
| shift_S | log(t) moved S m left | t | at t |

**Result.** Only the frame time breaks the image. PSNR against the `log` image (same pose), and mean gradient (sharpness):

| Scene (log camera moves t -> t + D) | lag_time 2 s | 5 s | 10 s | lag_actors 2 s | 5 s | 10 s |
|---|--:|--:|--:|--:|--:|--:|
| 213dfdac (8 m/s: 16, 31, 36 m) | 24.8 dB | 18.1 dB | 19.6 dB | 51.9 dB | 51.7 dB | 36.1 dB |
| 071d15c4 (standing until 5 s: 0, 3, 30 m) | 38.7 dB | 31.1 dB | 25.4 dB | 36.0 dB | 33.0 dB | 33.5 dB |

- `lag` and `lag_time` are the same image to within 0.1-2 dB: moving the actors costs little; the timestamp costs everything.
- In 213dfdac the whole frame smears at 5 and 10 s (sky, trees, houses, the parked cars that are part of the background): the
  "shattered glass". Sharpness falls 7.0 -> 6.6 -> 6.3. In 071d15c4, where the log car stood still until 5 s, lag_time 2-5 s is
  mild and 10 s (log camera 30 m ahead) is clearly degraded.
- Off the path at the right time is fine: shifts of 1 and 2 m are clean, 4 m shows edge artefacts only near the image border.
- lag_actors_10s shows a smeared actor box (a car of t + 10 s inserted where it is in the log): that is the actor model, small.

**Why.** The runtime puts the simulation time in `frame_start_us` / `frame_end_us` of every request
(`alpasim_runtime/services/sensorsim_service.py` `construct_rgb_render_request`, lines 229-311, `frame_start_us=start_us, frame_end_us=end_us`; the pose is interpolated from
the ego trajectory, the time is the clock trigger's). The NuRec renderer (`pycena_nrm_full`) evidently
renders the background as reconstructed at that time: the closer a region was to the recording camera at time T, the sharper
it is at T. A camera that is D s behind on the path looks at the near scene the recording camera left D s earlier, which at
time T is represented only as seen from far away. Hence the dependence on how far the log camera moved, not on D itself
(071d15c4 standing: little damage at 2-5 s). This is inferred from the probe. Correction (2026-10-22): the renderer's Python source ships inside the image; its scene model gives the whole
scene's appearance a time input (`nre/models/gaussians/collect/collector.py`) and the server config has no time option, which fits the probe
(read from the code, not traced per scene; [../../alpasim/results/version_and_frametime.md](../../alpasim/results/version_and_frametime.md)).

**Consequence.** Any driver slower than the log (human or model) gets a degraded image, the more so the further behind. A fix
would be in the renderer request (frame time = the log time whose pose is nearest the ego, actors at sim time = the
`lag_actors` condition, which renders clean), i.e. a runtime change; it is not applied.

Figures: [figs/render_lag_213dfdac.png](../figs/render_lag_213dfdac.png), [figs/render_lag_071d15c4.png](../figs/render_lag_071d15c4.png).
Rows: (log, shift 1 / 2 / 4 m), then for D = 2, 5, 10 s (log_late, lag, lag_time, lag_actors). Look at columns 2-3 against
column 4 in the lower rows. Raw images and probe.json: Tokyo `/data/runs/alpasim_human/test/render_probe/`.
