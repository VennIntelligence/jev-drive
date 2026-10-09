# LAT1: frame synthesis of the AlpaSim driver on the GPU (2026-10-09)

Result: the slot warp runs as one batched GPU call and reproduces the CPU reference bit for bit; with a faster (also bit-identical)
frame packing, a `drive` call drops from 99 to 35 ms median (p95 163 -> 66 ms) in closed loop at 8 concurrent rollouts, with the same
scores, ego poses and plans. The drivers now default to it (`SH30_SYNTH=gpu`); `SH30_SYNTH=cpu` is the reference path, unchanged.
What is left of a `drive` call is 25 ms of Python overhead in the model runner, not synthesis (last section).

Code: `jevdrive/op_interp.py` (`warp_gpu`, `_remap_gpu`, `_plane_gpu`), `experiments/alpasim/lib/sh30_core.py` (`lattice_gpu`,
`pack_fast`, `Core(synth=...)`), `lib/sh30_driver.py`, `lib/ap2_core.py`, `lib/ap2_driver.py`. Checks: `scripts/lat1_check.py`,
box jobs `scripts/lat1_chain.sh`. Numbers: `results/lat1/summary.json`. Box output: `$DATA_DIR/runs/alpasim/lat1`.

## Survey: what the standard options are

| family | standard implementations | what it would mean here |
|:--|:--|:--|
| Plane-induced inverse warp + bilinear resampling (what we already do: every destination pixel's ray hits the road plane or the 60 m sphere, moves with the ego, is looked up in the source frame) | the sampler of spatial transformers, `torch.nn.functional.grid_sample` (kornia `remap` / `warp_perspective` wrap it); `cv2.cuda.remap`, NPP `nppiRemap`, CV-CUDA `Remap`, DALI `remap` | same algorithm as the checkpoints were trained on, no retraining. **Picked.** |
| Cached warp maps | precomputed `map_x`, `map_y` per pose | the map depends on the relative pose of every decision, so only its pose-independent half is cacheable (the plane / sphere points per camera position): done in `_plane_gpu` |
| GPU JPEG decode | nvJPEG (`torchvision.io.decode_jpeg(device="cuda")`, DALI) | faster, not bit-identical to libjpeg: measured below, not adopted |
| Frame interpolation | classical optical flow (Farneback, DIS, TV-L1); learned VFI: RIFE (real time, TensorRT / ncnn ports), IFRNet, FILM, GIMM-VFI | a different input for the model. On navtest GIMM is worth +0.71 over the plane warp (decisions 116, 142). Serving it needs checkpoints trained on GIMM frames (the `@gimm` row caches) and a VFI forward pass per decision; neither was measured in this lane |

Why the pick is written by hand and not a `grid_sample` call: `grid_sample` interpolates in float, `cv2.remap` in fixed point (source
coordinates rounded to 1/32 px, integer weights of sum 1024). On the CPU path's own maps `grid_sample` differs from `cv2.remap` in 4.1 % of
the pixels (1 624 214 of 39.3 M, up to 4 grey levels). `_remap_gpu` does cv2's arithmetic with integer tensors (four gathers and a
weighted sum), so the frames are the ones the checkpoints saw. No library on the box offers that (no kornia, DALI, CV-CUDA or CUDA OpenCV
in `envs/op-train`, and none of them reproduces cv2's fixed point either).

## Design

- `warp_gpu(src, cam, pose_dst, pose_src)`: the 6-7 warped slots of a decision, both views, in one call. `warp_map`'s operations in
  float64 in the same order (no fused multiply-add: every step is its own kernel), then `_remap_gpu` for luma and for both chroma planes.
  The ego track (scipy splines, 0.4 ms) and the choice of source keyframe stay on the CPU, shared with the reference.
- Asynchronous in the CUDA sense: the warp is queued on the device and the encoder is queued behind it; the host blocks once per stage
  for the stage timers. Dropping those syncs (`SH30_STAGE_SYNC=0`) changes nothing measurable (below), so they stay.
- `pack_fast`: the CAM_F0 JPEG -> model frames step, bit-identical to `pack`: libjpeg's YCbCr read without two image copies, byte
  gathers from the flat image, the 2x2 chroma mean in integers (round half to even, as `np.rint` of the float mean).
- Not on the card: JPEG decode (libjpeg-turbo, 8-12 ms, releases the GIL) and frame packing. A GPU packing (`pack_gpu`, also
  bit-identical) was written and measured: one blocking 6 MB upload per image, slower under load than `pack_fast`.
- The CPU path is untouched: `lattice`, `warp_frame`, `warp_map`, `pack`, `OpenpilotMaps` are byte-for-byte as before; training prep
  and the replay tools keep `Core(synth="cpu")`.

## Before / after (closed loop, `m1/lists/s1.txt`, 126 scenes, 8 concurrent rollouts, job pinned to 8 cores, card shared with the renderer)

| stage, ms per `drive` (median / p95 / max) | CPU, M1 run of 10:11 | CPU, rerun right after the GPU run | GPU, 16:27 |
|:--|--:|--:|--:|
| frames: slot synthesis | 63.9 / 97.6 / 171.3 | 60.4 / 97.8 / 150.0 | **5.1 / 13.7 / 42.4** |
| encode: vision encoder, 8 image pairs, fp16 | 13.8 / 21.8 / 36.9 | 13.5 / 20.4 / 41.6 | 11.0 / 24.8 / 37.4 |
| policy: adapter + policy | 16.2 / 26.7 / 47.2 | 15.0 / 21.3 / 46.1 | 14.0 / 23.1 / 48.0 |
| wait for the single inference lock | 0.0 / 47.3 / 318.0 | 0.0 / 53.8 / 253.1 | 0.0 / 7.8 / 86.7 |
| total inside `drive` | 99.3 / 162.8 / 435.7 | 94.5 / 156.3 / 341.6 | **34.7 / 65.6 / 136.0** |
| CAM_F0 JPEG decode + packing (in `submit_image_observation`) | 30.0 / 61.3 / 103.2 | 27.3 / 57.9 / 132.0 | 16.4 / 47.8 / 85.8 |
| runtime-side `drive` RPC, mean | 113.3 | 109.2 | 43.3 |
| runtime, 126 rollouts (s) | 526 | 453 | 449 |
| driver CPU time (s) | 374 | 349 | 96 |
| driver VRAM, nvidia-smi peak / torch max allocated (GiB) | 3.52 / - | 3.52 / 2.77 | 3.72 / 2.96 |
| driver RSS (GiB) | 2.7 | 2.7 | 3.4 |

The organisers' target (0.1 s of model work per `drive`) is met at the median and at p95; the maximum (136 ms) is lock queueing at 8
rollouts per driver, four times the official 2 per replica. VRAM +0.2 GiB (float64 map temporaries, about 13 MB each), far below 16 GiB.

Replay of logged inputs through the driver class, no simulator (`lat1_check.py load`, 64 scenes, 640 decisions, 8 cores; the streams
fire without pause, so the lock is saturated and `wait` is queue length, not a latency):

| variant | frames (ms, median / p95) | decode + packing | wall for 640 decisions, 8 streams / 2 streams (s) |
|:--|--:|--:|--:|
| CPU reference | 53.8 / 66.1 | 26.0 / 55.6 | 57.7 / 56.4 |
| GPU warp + `pack_fast` (served) | 6.0 / 18.8 | 15.1 / 50.6 | 27.6 / 24.8 |
| same, no per-stage CUDA sync | 5.9 / 18.3 | 16.1 / 53.4 | 26.6 / 27.0 |
| GPU warp + `pack_gpu` | 5.5 / 20.6 | 16.3 / 47.6 | 34.8 / - |

Alone on 3 cores (`lat1_check.py prof`): the 8-thread CPU lattice 109 ms, `lattice_gpu` with sync 2.9 ms (`warp_gpu` itself 1.7 ms);
`pack` 31.9 ms, `pack_fast` 19.8 ms (measured before its two image copies were removed), of which libjpeg 12.7 ms. On a card another process is using, a blocking GPU call takes
7-10 ms instead of 2-3 (time slicing): the p95 of every GPU stage above is that, not compute.

## Equivalence

| check | sample | result |
|:--|:--|:--|
| resampler alone: `_remap_gpu` vs `cv2.remap` on the CPU maps | 100 scenes, 39.3 M px | 0 differing pixels |
| `pack_fast` and `pack_gpu` vs `pack` | 1 100 logged JPEGs, 865 M px | 0 differing pixels |
| slot frames, GPU vs CPU, logged inputs of the 400-scene SH30 run (`P2H10-F-s0`) | 4 000 decisions (1 200 cold-start), 12.58 G px | 0 differing pixels, 0 frames |
| plans on the same decisions: 4 s endpoint, yaw at 0.5 s, raw plan output | 4 000 decisions | all 0 (bit-identical); CPU run twice: 0 |
| same for the AP2 core (`AP2-AB-s0`) | 600 decisions, 1.89 G px | 0 differing pixels, plans identical |
| replay through the driver, 8 streams, GPU vs CPU | 640 decisions | 0 differing plans |
| closed loop, same list and chunking as M1's `s1-p2h10` | 126 scenes, 1 260 decisions | GPU vs the M1 run: all scene scores, ego poses and plans identical (mean score 0.9390 both) |
| closed loop, CPU rerun vs the M1 run (repeatability of the simulator) | 126 scenes | scores identical; 1 scene differs in 9 decisions (plans up to 0.47 m) |
| closed loop, a second GPU run (32 gRPC workers) vs the first | 126 scenes, 1 260 decisions | all identical |

The CPU rerun's one differing scene starts at decision 1 with identical ego pose and history but a rendered CAM_F0 JPEG of 371 368
instead of 371 369 bytes: the renderer, not the driver. So the simulator repeats exactly in 125-126 of 126 scenes for a fixed list
(decision 211), and the GPU path sits inside that.

The frames can differ from the reference only if a float64 map value differs in its last bit (numpy's matmul against separate
multiply and add) and that lands on a 1/32 px rounding tie. Not observed in 12.58 G pixels.

## Tried and dropped

- `torch.nn.functional.grid_sample` as the sampler: 4.1 % of pixels differ from cv2 (see survey).
- `pack_gpu`: bit-identical, slower under load (34.8 s against 27.6 s per 640 decisions) and touches the card from every gRPC thread.
- No per-stage CUDA sync: no gain (26.6 / 27.0 s against 27.6 / 24.8 s).
- nvJPEG decode (`lat1_check.py nvjpeg`, 154 images, 140 decisions; measurement only): decode + packing 16.5 -> 2.0 ms per image; 1.4 %
  of model-frame pixels differ (mean 0.015 levels, max 10); plans move by 0.012 m at the 4 s endpoint (median; p95 0.075, max 0.27 m),
  yaw at 0.5 s by at most 0.01 deg. For scale, two training seeds differ by 0.05 m (decision 211). Not identical, so it needs a
  closed-loop non-inferiority run on the 700 scenes and nvJPEG in the submission image. Worth 14 ms per step outside `drive`.
- Caching warped slots across decisions: the yaw spline of the ego track is global over the four keyframes, so a slot between two old
  keyframes changes when a new keyframe arrives. Not exact, and the warp costs 2 ms now.
- Returning from `submit_image_observation` before the decode finishes: `drive` needs the frame anyway, so the step does not get
  shorter. Not built.

## Waste in the shared baseline (measured, driver side)

| item | measured | status |
|:--|:--|:--|
| Slot warp on the CPU: 27 ms per frame, most of it the float64 map in numpy | 109 -> 2.9 ms per decision | done for the drivers. The row-cache builders (`ap2_prep.py`, `ot3_rows.py`, `pp_prep`) still call the CPU `lattice` / `synth_cpu` per token; `lattice_gpu` gives the same bytes |
| Frame packing in `OpenpilotMaps.__call__`: a float mean over five axes for a 2x2 average | numpy packing 14.5 ms on 3 cores, the three byte gathers that replace it 0.9 ms; decode + packing 27-30 -> 16 ms in closed loop (`pack_fast`, same bytes); `OpenpilotMaps.decode` also copies the image twice | done for the drivers; `jevdrive/navsim_zs.py` itself is unchanged, so every NAVSIM keyframe renderer still pays it |
| Model runner: `jevdrive/op_torch.py` interprets the ONNX graph node by node in Python under `vmap` (`lat1_check.py model`) | 837 graph nodes, 6 739 aten calls per decision. Encoder 9.9 of 10.2 ms and policy 14.3 of 14.5 ms are host time spent queueing kernels; the card is idle most of a `drive`. Policy for 4 decisions at once: 14.6 ms (same as 1); encoder for 2: 13.3 ms | **open, the largest item left (25 of 35 ms)** |
| - compiled graph (exploratory, one run) | `torch.compile` (inductor): policy 13.9 -> 3.0 ms, encoder 10.0 -> 6.6 ms, 23 s to compile, outputs differ (max 0.047 / 0.023 in fp16). `aot_eager`: identical outputs, 12.1 / 8.3 ms. `torch.jit.trace`: fails | not adopted: needs the plan and closed-loop checks, and a compiler plus a writable cache in a read-only image |
| - cross-session batching | two decisions in one pass cost about 28 ms instead of 2 x 25 | not built; exact per row only if verified |
| Single inference lock | wait p95 47 -> 8 ms at 8 rollouts after this lane | fine at the official 2 rollouts per replica |
| gRPC side: `submit_egomotion_observation` and `submit_route` take 9-10 ms each on the runtime's clock for handlers that do nothing; image RPCs 11-13 ms mean although 7 of 8 are dropped unread | unchanged by this lane (9.2 / 10.0 / 11.4 ms), and unchanged with 32 gRPC worker threads instead of 8 (10.0 / 10.5 / 11.9 ms, `drive` 44.5 ms, same scores and plans): not the pool size | open: whether it is the runtime's own event loop or Python gRPC under the GIL was not separated; about 10 ms per step outside `drive` |
| Per-stage CUDA syncs, per-call JSON log line | no measurable cost (`prep` 0.2 ms) | keep |

## Limits

- Latency was measured on a box that other lanes were loading (load average 50-70 on 208 cores, all three cards shared); the closed-loop
  columns share one scene list and core count, not one moment. The official hardware and CPU count are unknown.
- Bit-identity was shown for cv2 4.11, torch 2.14 + cu130, numpy 2.4 on the RTX 6000D. Another build (the submission image: torch
  +cu126) should rerun `lat1_check.py remap` and `equiv`; both need only logged messages.
- `_plane_gpu` and the packing index tables are cached per calibration (6 MB and 6 MB each; the plane cache is cleared at 32 entries).
  A private suite with hundreds of calibrations costs host memory for the index tables (not bounded).
- The ensemble driver (`ens_driver.py`) was not run on the GPU path.

## Proposed decision text (main numbers it)

AlpaSim driver, frame synthesis on the GPU (LAT1, 2026-10-09). The slot warp as one batched GPU call that repeats `cv2.remap`'s
fixed-point arithmetic (`op_interp.warp_gpu`) and an integer frame packing (`sh30_core.pack_fast`) give the CPU reference's frames bit
for bit: 0 differing pixels in 12.58 G (4 000 logged decisions of 400 scenes, 1 200 cold-start), plans identical, and a closed-loop run on
126 scenes with the list and chunking of M1's `s1-p2h10` repeats every scene score, ego pose and plan (mean 0.9390). `drive` falls from
99 to 35 ms median and from 163 to 66 ms at p95 at 8 concurrent rollouts (frames 64 -> 5 ms; JPEG decode + packing 30 -> 16 ms outside
`drive`); driver CPU time 374 -> 96 s per 126 scenes, VRAM +0.2 GiB. The drivers default to it (`SH30_SYNTH=gpu`); the CPU path is the
unchanged reference for training prep. A CPU rerun of the same list differs from the first in 1 of 126 scenes through a rendered frame,
which bounds the simulator's repeatability. Remaining `drive` time is 25 ms of Python dispatch in the ONNX interpreter
(`jevdrive/op_torch.py`: 6 739 aten calls per decision, card idle); a compiled policy runs in 3 ms but is not bit-identical. Evidence:
one box, one checkpoint each for SH30 and AP2 cores, latency under shared load.
[results](../experiments/alpasim/results/lat1_frame_synthesis.md)
