# ONPOL-NUPLAN: the on-policy pool as `on_train.py` reads it (contract, 2026-10-11)

Written by the trainer agent for the pool agent. A pool is one `pp_train.Store` family: `on_train.py --on <POOL>` appends the dir
`<POOL>` to the Store's data dirs (after the 12 navtrain shards and the ot1 / yr1 / bd4 dirs) and reads nothing else. N = rows of the
pool, one per visited decision that survived the prereg's exclusions (a) - (c); the row order is free but identical in every file.
All paths under `$DATA_DIR/runs/op_parity/cache/`. Everything metric, radians, float32 unless said; "own frame" = the rear-axle frame of
the visited state at the decision time t (origin at the rear axle, x forward, y left, yaw left-positive, heading 0), the frame in which
the driver expressed its plan (`sh30_core.Core.plan` -> `poses`).

## Files

| File | Key | Shape, dtype | Content |
|:--|:--|:--|:--|
| `<POOL>/tab.npz` | `names` | (N,) str | unique row id, must start with `on:` (e.g. `on:<scene>:<ckpt seed>:<k>`); the lookup key of the two label files; never a NAVSIM token |
| | `log` | (N,) str | nuPlan log of the scene (`cf1_report.log_of`) |
| | `ego` | (N, 20) f4 | the adapter input exactly as the driver fed it: `Core.plan`'s `o["ego"]` (`parity_adapter.ego_features`), unrounded (not drive.jsonl's 5 decimals) |
| | `fut` | (N, 8, 3) f4 | **imitation target**: rear-axle x, y, yaw at t + 0.5 .. 4.0 s in the own frame (the prereg's recovery path). Finite on every row |
| | `cam` | (N, 3) f4 | `cam_t` passed to `Core.plan` (camera position in the ego frame); the trainer reads `cam[:, 0]` for the plan -> rear-axle conversion |
| | `lht` | (N,) bool | traffic convention fed (`tc` = [0, 1] if True else [1, 0]); False unless SH30_LHT=1 |
| | `speed` | (N,) f4 | hypot of the body velocity at t (m/s); Store field, not in the S loss |
| | `nv` | (N,) int, optional | number of valid (newest) slots the driver fed; absent = 8 everywhere (SH30_COLD=backwarp feeds 8 at every decision, cold start included) |
| | `served` | (N, 8, 3) f4, optional | `o["poses"]` of that decision (gate R1, activation statistics); not read in training |
| | `pose`, `vel`, `acc`, `cmd` | as the navtrain tabs, optional | not read; if present use the base shapes (4, 3) / (4, 2) / (4, 2) / (4, 4) |
| | `off` | - | **must be absent** in a real pool (the labels are in the own frame). Only the stand-in carries it (below) |
| `<POOL>@warp/front.npy` | | (N, 8, 32, 512) f2, plain .npy (memory-mapped) | the hidden tokens the policy received: `o["tokens"]` (`H[0]`, encoder output before the adapter), slots oldest -> newest (`sh30_core.SLOT_T`). With `nv` < 8 the real tokens are right-aligned in the last `nv` slots, zeros before |
| `<POOL>@warp/teacher.npz` | `out`, `plan`, `di`, `pi` | (N, 1086) f4, (N, 33, 15) f4, (1086,) i8, (495,) i8 | shipped Cinque (port fp16, no adapter) on the row's tokens, computed exactly as `ot_rows.py` `make_teacher` (9 slots, slot 0 zero and invalid; with `nv` < 8 also the older invalid slots; `tc` from `lht`). `di` / `pi` = `A.distill_index` / `A.plan_index`, equal to the navtrain teachers' (asserted) |
| `<POOL>/agent.npz` | `tokens`, `ok`, `box`, `valid` | (N,) str = `names`, (N,) bool, (N, 9, 32, 5) f4, (N, 9, 32) bool | **actor boxes**, the `runs/op_parity/agent_labels/navtrain_all-k32.npz` layout: times t + 0, 0.5 .. 4.0 s; `box` = centre x, centre y, yaw (unwrapped over time per object), length, width in the own frame at t (every time step in that one frame); the 32 actors nearest at t followed by track; `valid` False (box zeros) where a track is absent. `ok` False = no label for the row (it is then trained without the agent hinge; rule (c) should leave none) |
| `<POOL>/road.npz` | `tokens`, `ok`, `sdf` | (N,) str = `names`, (N,) bool, (N, 128, 96) f2 | **road-and-lane raster**, the `runs/body1/labels_road/` layout (`bd4_prep.py`'s rasteriser, layers ROADBLOCK, ROADBLOCK_CONNECTOR, INTERSECTION, LANE, LANE_CONNECTOR): signed distance in metres, + inside, in the own frame; cell [i, j] centre at x = -8 + 0.5 (i + 0.5), y = -24 + 0.5 (j + 0.5) (x -8 .. 56 m over the 128 rows, y -24 .. 24 m over the 96 columns; sampled with `grid_sample(align_corners=False)`) |

## What the trainer reads per on-policy row

| Mode | Inputs | Target / labels | Loss terms on the row |
|:--|:--|:--|:--|
| every mode | `front` (+ `nv`), `ego`, `lht`, `cam[:, 0]` | `teacher.out` | non-plan heads distilled to the shipped teacher on the same tokens (lam_d 30), as every hinge-only and yaw-rate row of S |
| `imit` | | `fut` | imitation term, weight of one imitation row; **no** hinge: `agent.npz` / `road.npz` are not opened |
| `self` | | `agent.npz`, `road.npz` | S's two hinges on the own plan (shape-only, x ho_w 3, lam 10, summed over the rows / number of imitation rows of the batch); `fut` is not read |
| `both` | | all three | the imitation term and the two hinges x 3 |

`teacher.plan` is never read on these rows (they are never anchor rows and never dev rows) but the Store needs the key.
The trainer asserts: names unique, all start with `on:`, none in the split; `fut` finite; `di` / `pi` equal the base teacher's; label
`tokens` cover `names`; shapes above. It logs: rows, logs, rows with agent / road labels, rows whose footprint at t is off the raster.
Unlike S's hinge-only rows, a row that is off the road raster at t keeps that raster (S swaps in the NAVSIM drivable raster there; the
pool has none): say so if rule (b) leaves many.

## Optional longitudinal labels (lane L)

`on_train.py --on-lon` imports `experiments/onpol_nuplan/lib/on_lon.py`: `load(pool_dir: Path, names (N,), dev) -> pack` (optional; without
it `pack = dict(dir=pool_dir, names=names, dev=dev)`) and `loss(poses (n, 8, 3) torch, rear-axle x, y, yaw of the own plan in the own
frame, with gradient; rows (n,) long = row index in the pool; pack) -> {term: (n,) per-row loss}`. Each term is added as
lam 10 x ho_w x sum over the rows / imitation rows of the batch, in `self` and `both`. Its files live beside the others in `<POOL>/`.

## Stand-in

`on_train.py standin --data navtrain_full.s2of12` writes `onstand_<data>`: the bd4 rows of that shard in this format (front / teacher
linked, names `on:bd4:<token>`, `fut` = bd4's own-frame target, agent boxes moved into the own frame, the raster kept in the logged
frame with the tab's `off`, which is why `off` exists as a key). It is the known-answer test of gate T3 and a file-level example.
