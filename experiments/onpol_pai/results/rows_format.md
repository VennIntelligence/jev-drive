# ONPOL-PAI rows: file format (contract of record)

Written by the row-pipeline agent, 2026-10-11. This file is the interface between `scripts/onp_rows.py` (writer) and the trainer
(reader). Any change to a key, shape, unit or convention is written here first. Status of every label item is in the last section.

One file per scene: `/data/runs/onpol_pai/rows/<pass>/<scene_id>.npz` (Tokyo box; `pass` = `pol` for pass 1, the on-policy rollouts).
`np.savez_compressed`; float32 unless a dtype is given. A scene's file appears atomically (written as `.tmp`, then renamed), so any
file that exists is complete (both stages done; see `onp_rows.py`). Scenes whose build failed have `rows/<pass>/failed/<scene_id>.txt`
and no npz. `rows/<pass>/DONE` or `ERROR` exists when a build pass over the given run dirs has ended; `rows_<pass>.md` (next to the
pass directory, copied to `experiments/onpol_pai/results/`) is the generated summary.

Deviations from the task's wording, all additive or dtype-only: `P`, `V`, `acc`, `anchor` are float64 (the driver's own arrays,
unrounded); `stop_kind` has three values; extra keys `ego_in`, `z0`, `veh_sim`, `box_cls`, `box_label`, `box_id`, `log0`, `log_s`,
`path_len`, `wl_*`, `has_edges`, `t_rel_us`, `t_end_us`, `after_end`, `corridor_m`, `dlat_c`, `run`, `meta`.

N = decisions with `infer = true` of the session the scorer read (the session directory that holds `metrics.parquet`), in `k` order.
M = distinct model frames of the session, L = samples of the logged ego path (10 Hz, about 202), E = scene road-edge vertices.

## Frames and conventions

- **Local frame** = the rollout's local frame (what `rollout.asl` and `drive.jsonl: anchor` use): origin at the logged rig pose of the
  first log sample, x along its heading.
- **Rig frame at t0** = the frame the driver's plan is expressed in: x forward, y left, origin = rig origin = rear axle (the driver's
  assumption, pai_driver.py docstring), at `anchor` = the rig pose the driver interpolated for t0. Every per-row geometric label is in
  this frame; yaw is relative to the ego yaw at t0, counter-clockwise positive.
- **Time**: microseconds on the simulator's clock (`t0_us`); `t0` = `frame_end_us` of the newest frame = the Drive call's `time_now_us`.
  Plan pose k (0-based index k in arrays of 8) is at t0 + 0.5 (k + 1) s.
- **Boxes** follow `/data/runs/op_parity/agent_labels/navtrain_all-k32.npz` (lib/agent_hinge.py): `box[..., :] = x, y, yaw, length, width`
  with (x, y) the box **centre**, yaw the heading of the box's long axis (unwrapped over the 9 times), length along the heading, width
  across; times t0 + 0, 0.5, .. 4.0 s (9); K = 32 nearest by centre distance to the rig origin at t0, among actors present at t0.
- `max int64` (9223372036854775807) = "never"; NaN = "not present".
- Two ego boxes exist and differ; both are stored (see `veh`, `veh_sim`).

## Keys

### Identity
| key | shape, dtype | meaning |
|---|---|---|
| `scene`, `session`, `pass`, `tag`, `run` | str | scene id (`clipgt-...`), session uuid, pass name, checkpoint tag (from `--tag`), run directory |
| `t0_us` | (N,) int64 | decision time |
| `k` | (N,) int32 | the driver's Drive-call counter of the session |
| `meta` | str (JSON) | build info: script args, cold rule, counts, seconds, gate result of this scene |

### Inputs (exactly what the driver fed the model)
| key | shape, dtype | meaning |
|---|---|---|
| `frames` | (M, 2, 6, 128, 256) uint8 | every model-format frame the session received (`Session.frames` values), time order |
| `frame_ts` | (M,) int64 | `frame_end_us` of each |
| `slot` | (N, 8) int32 | index into `frames` of each of the 8 policy slots, oldest first (the order `pai_core.slots` stacks them: t0 - 1.4 .. t0 at 0.2 s); -1 = the slot was not a rendered frame (cold start: synthesised by back-warp) |
| `valid`, `real` | (N, 8) bool | as `pai_core.slots` returns them |
| `cold_rows` | (C,) int32 | row indices that have a -1 slot |
| `cold_cur` | (C, 8, 2, 6, 128, 256) uint8 | the full `cur` of those rows (use it instead of `frames[slot]`); computed in the driver image, where the back-warp is the served one bit for bit |
| `P` | (N, 4, 3) float64 | ego history x, y, yaw at t0 - 1.5 / 1.0 / 0.5 / 0 s in the rig frame at t0 (`C.fill_history` output) |
| `V` | (N, 4, 2) float64 | body velocity at the same times |
| `acc` | (N, 2) float64 | `dyn[-1][1]` |
| `cmd` | (N, 4) | one-hot [left, straight, right, unknown] |
| `cam_t` | (3,) float64 | camera position in the rig frame |
| `lht` | bool | left-hand traffic flag passed to the model |
| `ego_in` | (N, 20) | `PA.ego_features(P, V, tile(acc), cmd)` recomputed in the replay, unrounded (the model's ego input) |
| `ego` | (N, 20) | the same from `drive.jsonl` (rounded to 5 decimals) |
| `poses` | (N, 8, 3) | served plan (after serve_fix), rig frame at t0, from `drive.jsonl` (4 decimals) |
| `poses_model` | (N, 8, 3) | adapter plan before serve_fix (= `poses` where the record has no `fix`) |
| `anchor` | (N, 3) float64 | rig pose x, y, yaw at t0 in the local frame (replayed, unrounded; `drive.jsonl` rounds to 4 decimals) |
| `z0` | (N,) | rig z at t0 (local frame), from the egomotion sample nearest t0 |
| `v0` | (N,) | `hypot(V[-1])` m/s |

Building the model input of row i: `cur = frames[slot[i]]` if `(slot[i] >= 0).all()` else `cold_cur[j]` with `cold_rows[j] == i`;
then `pai_core.plan(core, cur, valid[i], P[i], V[i], acc[i], cmd[i], cam_t, lht)`.

### Labels (privileged; never an input)
| key | shape, dtype | meaning |
|---|---|---|
| `veh` | (4,) | ego length, width, rear-axle-to-box-centre offset, wheelbase from `start.jsonl: rollout_spec.vehicle` (wheelbase NaN: the spec carries none) |
| `veh_sim` | (3,) | ego length, width, rig-to-box-centre offset as the simulator / scorer used them (`rollout_metadata`: EGO aabb, `transform_ego_coords_rig_to_aabb`) |
| `box` | (N, 9, 32, 5) | simulator actors as moved in THIS rollout (`actor_poses` of `rollout.asl`), rig frame at t0, convention above |
| `box_ok` | (N, 9, 32) bool | the actor has a pose at that time (false after the rollout's last step: rows of the last 4 s have a truncated horizon) |
| `box_cls` | (N, 32) int8 | index into agent_labels' classes [vehicle, generic_object, pedestrian, bicycle]; -1 = empty slot |
| `box_label` | (N, 32) str | the simulator's own actor label |
| `box_id` | (N, 32) str | actor id ("" = empty) |
| `log_xyyaw` | (L, 3) float64 | logged ego RIG path, local frame |
| `log_t_us` | (L,) int64 | its times |
| `log_v` | (L,) | its speed (arc-length gradient over time) |
| `log_s` | (L,) | its cumulative arc length |
| `path_len` | float | total arc length of the logged path |
| `log0` | (N, 3) | the log's rig pose at t0, rig frame at t0 |
| `log_fut`, `log_fut_ok` | (N, 8, 3), (N, 8) bool | the log's rig pose at t0 + 0.5 (k + 1) s, rig frame at t0; false beyond the log's last sample |
| `s_ego` | (N,) | arc position of the ego's projection on the logged path. Tracked forward in time (each projection is searched from 5 m behind to 40 m ahead of the previous one); past the path's end the path continues straight along the log's last heading, so `s_ego` may exceed `path_len` |
| `e0` | (N,) | signed lateral offset of the rig origin from the path (or from its straight continuation), left + |
| `dpsi0` | (N,) | ego yaw minus path heading at `s_ego`, wrapped |
| `ds0` | (N,) | `s_ego` minus the log's own arc position at t0 (positive = the ego is ahead of the log) |
| `path`, `path_ok` | (N, 121, 2), (N, 121) bool | logged path at arcs `s_ego - 20 .. s_ego + 100` m (1 m), rig frame at t0; false beyond either end of the path |
| `path_v` | (N, 121) | the log's speed at those arcs (minimum over the log samples that share an arc: a logged stop reads 0) |
| `path_h` | (N, 121) | path heading at those arcs, relative to the ego yaw at t0 (unwrapped along the arc) |
| `imit`, `imit_ok` | (N, 8, 3), (N, 8) bool | RoaD / DAgger recovery target, definition below |
| `edge_box`, `edge_ok` | (N, 96, 5), (N, 96) bool | the 96 nearest road-edge segments within 40 m of the rig origin, at the ego's level (|z - z_ego| <= 3 m where the map has z), as thin boxes x, y, yaw, length (about 1.0), width 0.1; rig frame at t0; nearest first |
| `edges_xyz` | (E, 3) | scene road-edge polylines resampled at 1 m, concatenated, local frame (z NaN if the map has none) |
| `edges_off` | (n_poly + 1,) int32 | polyline j = `edges_xyz[edges_off[j]:edges_off[j + 1]]` |
| `has_stop`, `has_edges` | bool | the map's wait lines / road edges were read for this scene (false = the scene loader failed; the arrays are then stubs and `meta.map_error` says why) |
| `stop_s` | (N, 4) | arc from `s_ego` (rear axle) to the eligible stop lines that cross the logged path ahead within 100 m, nearest first; NaN pad |
| `stop_tcross_us` | (N, 4) int64 | scene time at which the LOGGED ego's front bumper crossed that line; max int64 if never inside the clip |
| `stop_kind` | (N, 4) int8 | wait-line type: 1 = STOP, 2 = YIELD, 3 = any other (the maps' `UNKNOWN`, all of them implicit lines); 0 = pad |
| `wl_s`, `wl_tcross_us`, `wl_tstop_us`, `wl_kind`, `wl_implicit` | (N, 4) | the same for ALL wait lines crossed ahead within 100 m (eligible or not); `wl_tstop_us` = first time the log is slower than 1 m/s with the rear axle in [line - 15 m, line - front + 1.5 m] before crossing |
| `turn_s` | (N,) | arc from `s_ego` to the curvature peak of the first >= 45 deg turn of the logged path ahead, within 100 m; NaN if none |
| `turn_vpk` | (N,) | the log's speed at that peak |
| `score` | float | scene score from `results-summary.json` |
| `fail_reason` | str | the summary's `failure_reason` ("" if none) |
| `zero_kind` | (3,) bool | summary flags [collision_at_fault, offroad, left_corridor_laterally] != 0 |
| `t_event_us` | (3,) int64 | first time the scorer's per-step series flags each kind (same order; at-fault collision = front or lateral, as the scorer defines it), inside the window the scorer reads (`t_rel_us` .. `t_end_us`); max int64 if never |
| `after_event` | (N,) bool | `t0 >= min(t_event_us)` |
| `t_rel_us`, `t_end_us` | int64 | the window the scorer reads: from the first step with `eval_relevant` > 0 up to and including the first step with offroad, ANY collision, or `dist_to_gt_trajectory >= corridor_m` (the plain distance to the logged path, which also grows past its end; max int64 if none). A flag outside the window is not in the score (an at-fault collision after a rear-end one, an offroad after the corridor exit), and `t_event_us` follows that rule |
| `after_end` | (N,) bool | `t0 > t_end_us`: rows the scorer never read |
| `corridor_m` | float | the scorer's corridor half-width (`eval-config.yaml: max_dist_to_gt_trajectory`) |
| `dlat_c` | (N,) | the scorer's own quantity recomputed: lateral distance of the ego BOX CENTRE to the logged box-centre path (compare with `corridor_m`) |

### Definitions

- **imit** (pre-registered by the main session). With s_log(t) the log's arc position at time t, d_k = s_log(t0 + 0.5 (k + 1) s) - s_log(t0)
  (the arc the log advances), D = max(8 m, 2 s x v0), w_k = max(0, 1 - d_k / D): pose k sits on the logged path at arc `s_ego + d_k`,
  displaced laterally (left +) by `e0 * w_k`. Heading = direction of the displaced curve c(d) = path(s_ego + d) + e0 w(d) n(s_ego + d),
  central differences on a 0.1 m grid; where d_k < 0.5 m the heading is `path heading + dpsi0 * w_k`. Not ok where t0 + 0.5 (k + 1) s is
  after the log's last sample or `s_ego + d_k` is beyond the path's end.
- **stop-line eligibility on PAI** (the map has no light state; lc.py / lc_ctx.py define eligibility only for nuPlan maps, so this is the
  stop-sign rule of lc_ctx.py carried over): a wait line crossed by the logged path (or by its 30 m straight extension past the last
  sample) is eligible at row t0 when the logged driver makes a stop for it (`wl_tstop_us` finite) and has not yet crossed it at t0
  (`t0 < wl_tcross_us`). A line the log drives through without stopping is not eligible. The log never violates "do not pass the line
  before the log does". "Front bumper" = rig origin + `veh_sim[2] + veh_sim[0] / 2`.
- **turn** = `lc_ctx.turn` (first run of |curvature| > 0.025 / m, merged over gaps <= 5 m, turning >= 45 deg, peak at least 1 m ahead)
  on the logged path heading sampled at 1 m from `s_ego` to `s_ego + 130 m` (or the path's end), kept if the peak is within 100 m.
- **Path vertices**: the logged samples that advance more than 0.05 m from the previous vertex (a parked log's position noise is not
  path); `log_s` is the arc of each sample's vertex, `log_v` the speed from the raw positions.
- **Cold-start rows** (the first 14 decisions, 1.4 s; all inside the simulator's force-GT window of about 1.7 s, during which the ego is
  held on the log whatever the plan says): `real` is false on the synthesised slots. `box_ok` is false at times after the rollout's
  last step, so rows of the last 4 s have a truncated box horizon.
- **Scorer geometry**: the scorer measures the corridor between the ego box centre and the logged path of the box centre (both
  `veh_sim[2]` ahead of the rig origin), not between rig origins. `path` / `e0` are rig quantities; `dlat_c` is the scorer's.

## Gate G-in (inputs equal what the driver fed)

Per scene, in `meta`: `gin_dump_n` / `gin_dump_eq` = PAI_DUMP files compared / with `cur`, `valid`, `real` bit-equal and the dumped
`ego` vector (float32, unrounded) equal; `gin_rec_n` / `gin_rec_eq` = decisions compared with `drive.jsonl` / with `hist` equal to P
at 4 decimals, `ego` equal to `ego_in` at 5 decimals, `anchor` at 4 decimals, command equal and t0 equal. Totals: `rows_<pass>.md`.

## Status of the label items

See `rows_<pass>.md` (generated) for what was real and what was stubbed in a given build. Stub convention: arrays keep their shape,
filled with NaN / false / max int64, and the scene-level flag (`has_stop`, `has_edges`) is false.
