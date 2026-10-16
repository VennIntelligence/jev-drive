# ALPASIM_FAST: fast local PAI reads, same scores (2026-10-16)

Switch: `ALPASIM_FAST` in `experiments/alpasim/lib/sim_fast.py`. `run_native.py` applies it on the GPU box (`pai_native.sh`, `pai_box.sh`)
and `pai_run.sh` on Tokyo (`pai_eval.sh`); docs/alpasim.md "Fast local reads". It is the default for PAI local reads since 2026-10-16.
`ALPASIM_FAST=0` restores the official camera set. Submission images do not go through these scripts.

| Mode | Cameras requested | Eval video |
|:--|:--|:--|
| `0` | the preset's 6 cameras, 1080 px, 10 Hz | on |
| `1` (default) | all 6: the driver's own camera exactly as the preset asks (PAI: `camera_front_wide_120fov` 1900x1080, 10 Hz, 30 ms shutter); the other 5 at 64 px with a 100 s frame interval, i.e. one frame per rollout | off |
| `solo` | the driver's camera alone | off |

- The driver's camera comes from the driver's own `CAM` (`pai_driver.py`, honouring `PAI_CAM`; `sh30_driver.py`: CAM_F0).
- The video is drawn after scoring; the official `ec2` presets run with it off. `ALPASIM_VIDEO=1` keeps it (`pai_box.sh` sets that from
  `KEEP_VIDEO=1`).

**Why the 5 unread cameras stay, at one frame each.** The runtime anchors the rollout clock on the first frame end over the requested
cameras (`alpasim_runtime/unbound_rollout.py`, `render_start_us`). On PAI scenes the 6 cameras' first frames differ. Requesting the front
camera alone (`solo`) therefore moves every decision time by 6 us to 30 ms, so the rollout is a perturbed copy. Mode `1` keeps the clock and
renders each unread camera once. Keeping them at 64 px but at 10 Hz is bit-identical but no faster: on this renderer each camera costs
about the same however small it is.

## Timing

Box rows: GPU box, `pai_box.sh`, CONC 4, cards shared with two other PAI stacks each (as in the g10 read itself), RSG-L18-s0, the full
36-scene g10 chunk lists.

| Variant | Chunk | Wall s / scene | vs g10, same chunk | Core s / scene | Peak VRAM GB (stack) | Peak RSS GiB (stack) |
|:--|:--|--:|--:|--:|--:|--:|
| g10, official (`0`) | c000 / c001 (all 8: 83-94) | 83.3 / 88.8 | 1 | 258 / 281 | 20.0 / 19.7 | 23.6 / 24.2 |
| `solo` | c000 | 36.6 | 2.28x | 87 | 19.2 | 19.3 |
| `1` | c001 | 34.8 | 2.55x | 88 | 20.2 | 18.2 |
| 6 cameras, unread ones 64 px at 10 Hz (dropped) | first 16 scenes of c000 | 75.6 | 1.1x | 191 | 19.5 | 16.2 |

Mean seconds per rollout, from the runtime log (`experiments/alpasim/scripts/pai_fast_cmp.py`):

| Variant | Load | Event loop (render + driver + controller) | Eval | Start-up + tail per job |
|:--|--:|--:|--:|--:|
| g10, official | 9-19 (33 for the 4 scenes that start together) | 191-215 | 103-123 | 21 + 35 |
| `solo` | 25 | 93 | 11 | 22 + 35 |
| `1` | 22 | 94 | 11 | 26 + 35 |

- Two thirds of the gain is the eval video. It took 103-123 s per rollout while holding a rollout slot.
- The other third is rendering one camera instead of six.
- Scene load, start-up and the gaps between scenes are small at CONC 4 (under 6 s per scene of wall). The renderer's PTX JIT cache
  already persists natively on the box (`~/.nv`, now `$DATA_DIR/cache/cuda`), so no load lever was worth adding.

Tokyo rows: `pai_eval.sh`, one 3090 stack, CONC 4, the 6 scenes of the ACC smoke read
`/data/runs/route_follow/pai_acc_smoke/off` (RSG-L18-s0, gains 10). The reference ran on card 1 and the fast runs on card 0. Card 1 was
partly busy with another stack during the fast runs.

| Variant | Compose wall s (6 scenes) | Scores equal | Controller traces equal |
|:--|--:|--:|--:|
| official (reference) | 433 | | |
| `solo` | 280 | 5 / 6 (one +0.0005) | 1 / 6 |
| 6 cameras, unread ones 64 px at 10 Hz | 428 | 6 / 6 | 6 / 6 |
| `1` | 271 | 6 / 6 | 6 / 6 (every decision of every scene) |

## Verification against g10 (`$DATA_DIR/runs/route_follow/pai_g10/g10`, scene by scene)

The chunk lists are identical, and so are the checkpoint, WIZ (`controller.gains.rel_front_steering_angle_weight=10`) and SERVE
(`JEV_VCONT=1.0 JEV_LEAD=1`). g10 ran on code c37e7150; the fast runs ran on 6f1a9de5, where the driver differs only in `JEV_ACC`, which
is off and bit-identical when off.

| Variant | Scenes | g10 mean | Mean | Diff (95 % CI, paired bootstrap) | Identical scores | Within 0.05 | Zeros g10 / read | Zero flips | Identical traces |
|:--|--:|--:|--:|:--|--:|--:|--:|--:|--:|
| `1`, c001 | 36 | 0.5509 | 0.5509 | +0.0000 [+0.0000, +0.0000] | 36 | 36 | 10 / 10 | 0 | 36 |
| `solo`, c000 | 36 | 0.5263 | 0.5270 | +0.0006 [-0.0005, +0.0024] | 26 | 36 | 12 / 12 | 0 | 4 |
| 64 px at 10 Hz, 16 scenes of c000 | 16 | 0.4989 | 0.4989 | +0.0000 | 16 | 16 | 5 / 5 | 0 | 15 (one scene has no trace) |

Reference noise, box vs Tokyo parity (experiments/route_follow/results/pai_acc/parity.md): +0.005 [-0.008, +0.019], 162 / 281
identical, about 2 % zero flips.

- **Mode `1` is the same run as the official set, 2.5x faster.**
- **`solo` passes the statistical check** (CI contains 0, no zero flips, 10 scenes moved by at most 0.026). But it is not faster than
  `1`, and it is not the same run.
- Scene `clipgt-13d646af` scores 0 in every read of c000 (g10, `solo`, the 16-scene check). g10 and `solo` both log a renderer
  exception at its first frame (`ego_hood.py overlay_on_image`: tensors on cuda:0 and cpu). That zero comes from the renderer, not from
  the driving.

## Packing

- **VRAM:** a stack's peak barely moves (19-20 GB in every mode). The renderer's scene cache (`nre_cache_size` = CONC + 1 scenes, about
  3 GB each) dominates, not the camera count. The default booking is lowered from 28 to 22 GB (RAM from 40 to 30). Three stacks still
  fit an 84 GB card by booking; four would need about 21 GB each against a 20.2 GB peak, so that is not done.
- **CPU:** the gain is mainly here. A scene costs 88 core seconds instead of 258-300, so 6-core jobs no longer saturate their cores.
- **RAM:** at 18-19 GiB per stack, the box (754 GiB, about 110 GiB used at 17:00) is not RAM-bound either.

## Scope

- **Tokyo:** `pai_eval.sh` / `pai_run.sh` take the same switch and default.
- **Videos:** Tokyo reads that are watched (`pai_videos.py`) need `ALPASIM_VIDEO=1`.
- **nuPlan reads:** `run.sh` → `run_native.py` take it with `ALPASIM_FAST=1`. The default there stays 0 because it is unmeasured.
  nuPlan uses MTGS with `render_bundling: RENDER_AGGREGATED` (all cameras in one pass) and synthesised first-frame times, so whether
  dropping 7 of 8 cameras saves time, and whether it is identical, needs its own check against a scored nuPlan read.
