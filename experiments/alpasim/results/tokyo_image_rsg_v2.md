# Submission images for RSG-L18-s0 and RTB1ARO-L18-s0 (V2): build, containerised smoke (2026-10-17)

Read this before you push or submit either image to the nuPlan track. Both checkpoints are ROUTE-MEM family ([decision 282](../../../research/decisions/282.md),
[286](../../../research/decisions/286.md)): the adapter reads a memory of the route messages the driver has received in the episode. `RTB1ARO-L18-s0` is
"V2" of [decision 339](../../../research/decisions/339.md) (exploratory: the retention term without the standing-wait rows, on the RSG-L18 recipe). The last nuPlan
image, `YR10m10-F-s0` of 2026-10-12 ([tokyo_image_yr10m10.md](tokyo_image_yr10m10.md)), predates the route-memory serving code, so these are the first nuPlan images
whose driver builds a route memory. Each image serves one checkpoint through `sh30_driver.py` with the configuration of the box reads that judged it (public 1491 @10
and held 460 @18), except that the model passes are interpreted (section "Configuration served"). Built and tested on the Tokyo box with `docker/new_tag.sh` from a
frozen clone at `d47f2692`. Nothing was pushed, nobody logged in to the registry, nothing was submitted; no job was paused or deleted on either box. The independent
audit is pending. Run dirs on the Tokyo box: `/data/runs/alpasim/tokyo_image/RSG-L18-s0-20261017-150558/` and `/data/runs/alpasim/tokyo_image/RTB1ARO-L18-s0-20261017-151621/`
(each: build, test, smoke, smoke_c2); native references `/data/runs/alpasim/native_ref/{RSG-L18-s0,RTB1ARO-L18-s0}/`. Files added and left in place (nothing deleted): on Tokyo the frozen clone
`/data/runs/alpasim/tokyo_image/code-rsg-v2/`, the `RTB1ARO-L18-s0` checkpoint and sidecar under `/data/runs/op_parity/runs/`, and `/data/runs/alpasim/tokyo_image/rsg_v2_chain/` (`chain.sh` that ran both images, and the
uncommitted helper scripts `facts.sh`, `cmp_drive.py`, `rmem_unit.py`, `plan_k0.py`, `scene_table.py` with their outputs); on the GPU box `$DATA_DIR/tmp/rsg_v2/` (`ref48.py` and the reference extracts).

## Images

| | RSG-L18-s0 | RTB1ARO-L18-s0 (V2) |
|---|---|---|
| Local name | `jev-alpasim:rsg-l18-s0-d47f2692` | `jev-alpasim:rtb1aro-l18-s0-d47f2692` |
| Image id | `sha256:f99095eafd5453afe38c245b9d8b804a22c4af9e5a546bc5671ae7cd51ee92ee` | `sha256:18a85838204ecc53050ecf1992134ee25c779ddcb035087fe04417910951ce27` |
| Size | 8 105 610 044 bytes (7.55 GiB; limit 40 GiB) | 8 105 611 459 bytes (7.55 GiB) |
| Built | 2026-10-17 15:06 JST, 13 s | 2026-10-17 15:16 JST, 11 s |
| Repo state | `d47f2692` (origin/main when the frozen clone was made), clean in the image's paths (no `-dirty` suffix) | same |
| Layers | 18. The lowest 12 (Debian, Python, the locked packages, the gRPC package, the base model, the sample-code files; 7.9 GB) equal those of the YR10m10 and the PAI images; the 6 above are new (driver code, entry points, user, stable checkpoints, extra checkpoint, verify output; about 230 MB uncompressed) | 18. The lowest 16 equal those of the RSG image (the lowest 12 also those of the YR10m10 and PAI images); only the extra-checkpoint layer (77.6 MB) and the verify output differ |
| Also inside | the stable checkpoint layer of every nuPlan image (`P2H10-F-s0`, `AP2-AB-s0`), not served; the other candidate's checkpoint is in neither image | same |

## Checkpoints

| | RSG-L18-s0 | RTB1ARO-L18-s0 (V2) |
|---|---|---|
| What | decision 286: RB0-F-s0 (route-memory base) + one gentle on-policy round, S rows only (`nx.sh` `$RSG`, 2 500 steps) | decision 339 V2: base RTB1AR-F-s0 (`nx.sh train rtb1ar 0 10000`: RB0 recipe + retention term `--kd-lam 1 --d2-lam 1 --kd-groups AR:1`) + the RSG-L18 recipe (`train rtbo=RTB1AR-F-s0 0 2500 RTB1ARO-L18-s0`) |
| File | `$DATA_DIR/runs/op_parity/runs/RSG-L18-s0/ckpt-final.pt`, 77 549 791 bytes, 2026-10-13 | `.../RTB1ARO-L18-s0/ckpt-final.pt`, 77 551 199 bytes, 2026-10-17 |
| sha256 | `fbe7ac9ff0d32ee6636369e6fcac248bf517dff9db528c5be689d18a04f05143`. Equal on the GPU box, on the Tokyo box and inside the image | `7a7a85f71742ad8e192d302ca783f169d7fe6c1f1f666d4723d02751e12e601b`. Equal on the GPU box (pulled to Tokyo with rsync, checked), on the Tokyo box and inside the image |
| Sidecar `route_mem.json` sha256 | `672ef154639ab9201093d2af54b0570695cd1dd1d19c313b1a968b02ad394ade` (box = Tokyo = image) | `0697fe6143a454a6fb4f81788a46e5e328c7b67dea5a6a40b703e3ab20a43697` (box = Tokyo = image) |
| Identity | the `RSG-L18-s0` of the stored 10-13 reads and of the PAI images (`382164ae`) | the checkpoint of `nx.sh rtb-wide` / `cont-reads` (public read 2026-10-17 11:29 box time) |
| `cinque.ort.onnx` in the image | `3f12a0cd266cb827d0a3edd81cf86ed70ddbeea29a99c868026a1ba56b7a4673` (the YR image's) | same |

## Configuration served, against the box reads that judged the checkpoints

The reads: public 1491 @10 and held 460 @18 on the GPU box. RSG-L18-s0: the stored reference reads of 2026-10-13
(`runs/route_follow/guard/read10/RSG-L18-s0/b0..b9`, `runs/route_mem/read/RSG-L18-s0/c0..c5`). RTB1ARO-L18-s0: `nx.sh rtb-wide` and `cont-reads` of 2026-10-17
(`runs/wait_cap/next/cont/wide/pub/RTB1ARO-L18-s0/b0..b9`, `runs/route_mem/read/RTB1ARO-L18-s0/c0..c5`). Both go through `scripts/run.sh <dir> sh30`
(`ol_chain.sh readlist` / `rm_chain.sh read`). The driver command each run stored in its `native_summary.json` is `env ALPASIM_SRC=.. TMPDIR=.. TORCHINDUCTOR_CACHE_DIR=..
ALPASIM_DRIVER_LOG_DIR=.. ALPASIM_DRIVER_GRPC_WORKERS=8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 SH30_TAG=<tag> SH30_COLD=backwarp SH30_DUMP=0 SH30_LHT=0
.../envs/op-train/bin/python experiments/alpasim/lib/sh30_driver.py`: no `JEV_*` and no `SH30_COMPILE`. The pool jobs add only `CL_HIST_KEY` (submission flag) and, through the pool's `alpasim`
profile (`jevdrive/cl/profiles.py`, applied by `run_native.py` to every process it starts), `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and scratch / cache paths on the data disk: allocator and
paths, no effect on values. The driver's own log lines of those runs are the record of what it served.

| Setting | Box reads (driver log, stored command) | Image | |
|---|---|---|---|
| Driver | `sh30_driver.py` (`run.sh <dir> sh30`) | `JEV_DRIVER=sh30` baked in; `serve.py` prints `serve: driver sh30, tag <tag>` and execs `sh30_driver.py` (the YR image used `auto`, which resolves to `sh30` for both tags in `tags.json`) | same |
| Served checkpoint | `SH30_TAG=<tag>` | `JEV_TAG=<tag>` -> `SH30_TAG` | same |
| ROUTE-MEM branch | log: `serving the ROUTE-MEM branch, memory on, layout -6..+72` (the 10-13 RSG-L18-s0 reads print it without the layout words: older log text); `SH30_RMEM` unset | same line; `SH30_RMEM` unset; the checkpoint's `route_mem.json` is in the image | same |
| Cold start | `SH30_COLD` unset = `backwarp`; log `backwarp, motion 1.00` | unset; log `backwarp, motion 1.00`; `version_id` `jev-<tag>-backwarp` | same |
| Frame synthesis, JPEG decode | log `synth gpu`, `jpeg libjpeg` | same log | same |
| Model passes | log `compiled True` (`SH30_COMPILE` unset = 1: `torch.compile`) | `SH30_COMPILE=0`; log `compiled False` (interpreted) | **differs** (known, below) |
| Speed profile | log `JEV_VCONT 0, JEV_LEAD 0, JEV_BASE 0` (the V2 reads also `JEV_VHIST off, JEV_ROUTE off`) | `JEV_VCONT=0`, `JEV_LEAD=0`, `JEV_ACC=0`, `JEV_VHIST=` (off); same log line | same; no ACC (decision 329) |
| Other switches (`JEV_TRACK`, `JEV_EXPERT*`, `JEV_PERTURB`, `JEV_STOP` / `JEV_REPLAN`, `JEV_TTS`, `JEV_DESIRE`, `JEV_GAP`, `SH30_MOTION`) | not set; the reads' `drive.jsonl` rows carry none of their keys (`fix`, `track`, `route`, `vhist`, `expert`, `perturb`, `body`, `tts`, `desire`, `rx`, `poses_model`) | not set; the smoke's `drive.jsonl` rows have exactly the keys of the reads' rows | same |
| Traffic | `SH30_LHT=0` | `SH30_LHT` unset: right-hand traffic | same |
| gRPC workers, threads | `ALPASIM_DRIVER_GRPC_WORKERS=8`, `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=2` | the same values in the image `ENV` | same |
| Cameras, simulator | official camera set (`ALPASIM_FAST` is 0 for nuPlan), `+e2e_challenge_nuplan=full`, `SH30_SYNTH` unset | the official stack's `full` preset | same |
| Allocator, scratch | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `TMPDIR` and cache dirs on the data disk (pool profile) | not set; `/tmp` tmpfs | differs, no numeric effect |

**The one setting that differs: compiled against interpreted passes.** The reads did not use interpreted passes: every driver log of the reads (public and held, both checkpoints)
says `compiled True`, because `SH30_COMPILE` defaults to 1 on the box. The image cannot compile (no C compiler, read-only triton cache), so every image serves `SH30_COMPILE=0`, as the YR10m10
image did; compiled and interpreted passes differ at fp16 rounding level (`results/lat1_frame_synthesis.md`: plans move by about 1 cm; P2H10-F-s0 on 700 scenes 0.9481 compiled against
0.9484 interpreted, the same 25 zeros). The smokes below measure this together with the host difference (RTX 3090, torch `+cu126`, against RTX 6000D, `+cu130`); no native interpreted run
was made to separate them, because the means are within 0.0005 of native.

Every environment variable the image sets (`docker image inspect`; identical in both images except `JEV_TAG`): `PATH`, `LANG=C.UTF-8`, `GPG_KEY`, `PYTHON_VERSION=3.11.17`,
`PYTHON_SHA256`; `PYTHONDONTWRITEBYTECODE=1`, `PYTHONUNBUFFERED=1`, `HOME=/app`, `TMPDIR=/tmp`, `XDG_CACHE_HOME=/tmp/.cache`, `TORCH_HOME=/tmp/torch`,
`TORCH_EXTENSIONS_DIR=/tmp/torch_extensions`, `HF_HOME=/tmp/huggingface`, `HF_HUB_OFFLINE=1`, `MPLCONFIGDIR=/tmp/matplotlib`, `CUDA_CACHE_PATH=/tmp/nv`, `NUMBA_CACHE_DIR=/tmp/numba`,
`DATA_DIR=/app/data`, `ALPASIM_SRC=/app/alpasim`, `ALPASIM_DRIVER_LOG_DIR=/tmp/alpasim-driver`, `ALPASIM_DRIVER_HOST=0.0.0.0`, `ALPASIM_DRIVER_PORT=6789`, `ALPASIM_DRIVER_GRPC_WORKERS=8`,
`OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`, `JEV_TAG=<tag>`, `JEV_DRIVER=sh30`, `SH30_GIT_HASH=d47f2692`, `JEV_VCONT=0`, `JEV_LEAD=0`, `JEV_ACC=0`, `JEV_VHIST=`,
`SH30_COMPILE=0`, `TRITON_CACHE_DIR=/tmp/triton`, `TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor`. No `PAI_*`, `AP2_*`, `SH30_LHT`, `SH30_SYNTH`, `SH30_COLD`, `JEV_BASE`, `JEV_TRACK`, `JEV_STOP`, `JEV_REPLAN`.
The runtime adds only `ALPASIM_CONTESTANT_REPLICA_INDEX` / `ALPASIM_CONTESTANT_REPLICAS`. Entry point: `CMD ["python","/app/serve.py"]`, no `ENTRYPOINT`, `EXPOSE 6789`, `STOPSIGNAL SIGTERM`, no `HEALTHCHECK`,
no volumes, `WORKDIR /app`, `USER 10001:10001`.

## Route memory: does the image's nuPlan driver build it as the box driver does?

Yes, on every check below; nothing was patched. The path is one piece of code: `sh30_driver.py` -> `sh30_core.Core` (`rmem.load_pmodel` attaches the branch from the sidecar
`route_mem.json`) -> one `rmem.Tracker` per session, fed by `submit_route` (message time and rig-frame waypoints), read at each decision with the pose of every message
(`Tracker.features`) and fed after the ego row.

| Check | RSG-L18-s0 | RTB1ARO-L18-s0 |
|---|---|---|
| Code: the 29 files of `closure.txt` in the image, hashed, against the repo at `d47f2692` (and `serve.py`, `probe.py`, `verify.py`, `pai_spec.json` against `docker/`) | equal | equal |
| Build-time `verify.py` (CPU, inside `docker build`) | `branch RSG-L18-s0 dim 200 order False reflex False`; `tags.json`: `RSG-L18-s0` -> `sh30` | `branch RTB1ARO-L18-s0 dim 200 order False reflex False`; `tags.json`: `RTB1ARO-L18-s0` -> `sh30` |
| Driver log at start (test, smoke, smoke_c2) | `serving the ROUTE-MEM branch, memory on, layout -6..+72`, the line of the box reads | same |
| `rmem.Tracker` on a synthetic episode (a route with a 90 deg turn, 10 route messages, a driven curve; unordered and ordered layout: 20 feature vectors) in the box's `envs/op-train` (python 3.11.16, numpy 2.4.6) and inside the image (python 3.11.17, numpy 2.4.6) | 20 of 20 bit-identical | 20 of 20 (the V2 image's output equals the RSG image's) |
| Logged `rmem` of the 48-scene smoke against the box read's `drive.jsonl` rows of the same scenes: rows with a non-empty memory | 475 of 480 (box: 475 of 480; the 5 empty rows are decisions 0-4 of one scene, on both sides) | 475 of 480 (box: 475 of 480; same scene) |
| Decisions 0 and 1 of the 48 scenes (state identical on both sides): ego row, command, first route waypoint, fed memory | max abs difference 0.0 in 96 of 96 decisions | 0.0 in 96 of 96 |
| All 480 decisions: valid-point mask of the memory equal; command equal | mask 474 of 480, command 479 of 480; the differences are at decisions 7-9 (mask) and 9 (command), after the ego pose has drifted by centimetres | mask 476 of 480 (decisions 8-9), command 480 of 480 |
| Decisions whose measured ego row agrees to 1e-3 on both sides: memory difference p50 / p90 / max (feature units; x is in 20 m) | 157 decisions: 0 / 2.4e-4 / 1.9e-2 | 154 decisions: 0 / 2.2e-4 / 2.7e-3 |

Where the two sides see the same state (decision 0, identical ego row, command and memory) the first plans differ by a median 0.11 / 0.18 cm at 0.5 s and 1.2 / 1.0 cm at 4 s
(RSG / V2; max 1.2 / 0.6 cm and 6.2 / 4.9 cm): GPU / CPU numerics, as in decision 215. The closed loop amplifies them: by decision 2 to 8 the ego row of every scene differs by more than 1e-3.

## Constraint test and smoke (one RTX 3090 per role)

`test.sh`: the organisers' `run_local_container.sh` flags plus `--network none`, 2 concurrent synthetic rollouts x 8 (the probe sends a route message every step, so the route memory
is exercised under the read-only flags). `smoke.sh`: the 48 public scenes of `scenes_navtest_full_part001_48.txt` (all of the log `2021.05.25.14.16.10_veh-35`) in the official containerised stack (AlpaSim
0bb4c4b, `alpasim-base:0.89.0`), run twice per image: at 8 concurrent rollouts (the wizard overrides of the native reference runs) and with the driver endpoint at 2 concurrent rollouts
(the official per-replica count, `WIZ_EXTRA=runtime.endpoints.driver.n_concurrent_rollouts=2`). The native references are the 48 rollouts of these scenes filtered from each checkpoint's public
1491 @10 read (10 chunk runs, `compiled True`; RSG at 8 concurrent / render cache 9, V2 at 4 / 5), not 48-scene runs.

| | RSG 8 concurrent | RSG driver 2 conc. | RSG native | V2 8 concurrent | V2 driver 2 conc. | V2 native | limit / expected |
|---|--:|--:|--:|--:|--:|--:|---|
| Mean scene score, 48 scenes | **0.9597** | **0.9597** | 0.9592 | **0.9597** | **0.9597** | 0.9596 | within ~0.01 of native |
| Zeros (collision / offroad / corridor) | 1 (0 / 1 / 0) | 1 (0 / 1 / 0) | 1 (0 / 1 / 0) | 1 (0 / 1 / 0) | 1 (0 / 1 / 0) | 1 (0 / 1 / 0) | |
| Scenes at 1 | 33 | 33 | 33 | 38 | 38 | 38 | |
| Identical to native | 34 of 48 | 34 of 48 | | 39 of 48 | 39 of 48 | | most |
| Differ from native by <= | 0.0104 (14 scenes) | 0.0104 (14) | | 0.0073 (9) | 0.0073 (9) | | |
| Zero / non-zero flips against native | 0 | 0 | | 0 | 0 | | |
| Paired mean difference | +0.0005 | +0.0005 | | +0.0000 | +0.0000 | | |
| 8-concurrent vs 2-concurrent run | | 0 of 48 scenes differ | | | 0 of 48 scenes differ | | |
| `drive` calls / inference errors / input errors | 480 / 0 / 0 | 480 / 0 / 0 | | 480 / 0 / 0 | 480 / 0 / 0 | | 0 errors |
| Tracebacks in the driver log | 0 | 0 | | 0 | 0 | | |
| `drive`, server side, p50 / p90 / p99 / max | 36.6 / 47.4 / 106.3 / 163.1 ms | 33.6 / 38.6 / 46.0 / 74.8 ms | | 36.2 / 46.1 / 85.6 / 114.2 ms | 33.3 / 39.2 / 59.6 / 72.1 ms | | p50 < 100 ms |
| `drive`, `test.sh` probe, 2 concurrent, client side, p50 / p90 / p99 / max | 50.4 / 52.6 / 63.1 / 70.2 ms | | | 50.5 / 52.4 / 61.9 / 70.5 ms | | | target 100 ms |
| Cold start, `docker run` -> listening (`test.sh`) | 17.6 s | | | 17.9 s | | | runtime waits 240 s |
| GPU memory of the driver | 1.57 GiB (probe), 1.72 GiB card peak in the smoke | 1.72 GiB | | 1.57 GiB, 1.72 GiB | 1.72 GiB | | 16 GiB |
| `/tmp` high-water: probe / smoke | 0.36 / 1.42 MiB | 1.42 MiB | | 0.36 / 1.42 MiB | 1.42 MiB | | 2 048 MiB |
| `/run` high-water | 0 | | | 0 | | | 64 MiB |
| Read-only root: write errors / changed files; stop (SIGTERM -> exit 0) | 0 / 0; 0.6 s | | | 0 / 0; 0.6 s | | | 0 |
| Wall time of the smoke | 209 s | 268 s | | 199 s | 265 s | | |

The zero of every run, native and containerised, is the same scene: `...01690_02183-30f09e001cf55013`, offroad (the knife-edge scene of the YR report, which the YR container run scored 1.0
against a native 0; here host and container agree). The scenes that differ from native are exactly the progress-limited ones (RSG 14 = 48 - 33 - 1, V2 9 = 48 - 38 - 1): a scene at
the cap of 1.0 stays at 1.0, a scene below it moves in the third or fourth decimal with a centimetre-level change of the trajectory. Driver concurrency 2 against 8 changes no score for RSG
(48 of 48 scenes identical for V2). Latency and VRAM are those of the earlier images (YR10m10: probe p50 50.0 ms, 1.56 GiB, cold start 18-20 s).

### Scenes that differ from the native read (8-concurrent smoke; native / container)

**RSG-L18-s0**: 14 of 48 scenes differ; the other 34 (33 at 1.0 and the one zero) are identical.

| scene (prefix `2021.05.25.14.16.10_veh-35_` omitted) | native | container, 8 concurrent | container, driver 2 concurrent | container - native |
|---|--:|--:|--:|--:|
| 00083_00485-4b594a8e80915943 | 0.9776 | 0.9880 | 0.9880 | +0.0104 |
| 00083_00485-c9b12b21fa7c57fd | 0.9845 | 0.9827 | 0.9827 | -0.0018 |
| 01100_01664-1893fb783df95146 | 0.7667 | 0.7670 | 0.7670 | +0.0004 |
| 01100_01664-6fbe0e06902e5304 | 0.9729 | 0.9731 | 0.9731 | +0.0002 |
| 01100_01664-6fed9368351f54d5 | 0.9062 | 0.9044 | 0.9044 | -0.0019 |
| 01100_01664-82cd122751085a80 | 0.6831 | 0.6828 | 0.6828 | -0.0003 |
| 01100_01664-9215555823945665 | 0.9881 | 0.9862 | 0.9862 | -0.0019 |
| 01690_02183-0c77bbb199f3589f | 0.9780 | 0.9829 | 0.9829 | +0.0049 |
| 01690_02183-2313aa310e16503d | 0.9837 | 0.9844 | 0.9844 | +0.0008 |
| 01690_02183-54e1cb577c0a5f7e | 0.9702 | 0.9723 | 0.9723 | +0.0021 |
| 01690_02183-75f9d978b4d757a2 | 0.9752 | 0.9756 | 0.9756 | +0.0004 |
| 02482_02649-6507522e38405857 | 0.9435 | 0.9534 | 0.9534 | +0.0099 |
| 02482_02649-6f11adda2af357ff | 0.9173 | 0.9164 | 0.9164 | -0.0009 |
| 02482_02649-8e9f743d92c05d10 | 0.9945 | 0.9946 | 0.9946 | +0.0001 |

**RTB1ARO-L18-s0**: 9 of 48 scenes differ; the other 39 (38 at 1.0 and the one zero) are identical.

| scene (prefix `2021.05.25.14.16.10_veh-35_` omitted) | native | container, 8 concurrent | container, driver 2 concurrent | container - native |
|---|--:|--:|--:|--:|
| 00083_00485-83fd1c194c0c5441 | 0.9645 | 0.9572 | 0.9572 | -0.0073 |
| 00083_00485-c9b12b21fa7c57fd | 0.8376 | 0.8415 | 0.8415 | +0.0039 |
| 01100_01664-025b657634505df3 | 0.9701 | 0.9672 | 0.9672 | -0.0029 |
| 01100_01664-1893fb783df95146 | 0.7771 | 0.7755 | 0.7755 | -0.0015 |
| 01100_01664-6fbe0e06902e5304 | 0.9977 | 0.9984 | 0.9984 | +0.0007 |
| 01100_01664-82cd122751085a80 | 0.9138 | 0.9205 | 0.9205 | +0.0067 |
| 01690_02183-75f9d978b4d757a2 | 0.9281 | 0.9271 | 0.9271 | -0.0010 |
| 02482_02649-6f11adda2af357ff | 0.8197 | 0.8214 | 0.8214 | +0.0017 |
| 02482_02649-8e9f743d92c05d10 | 0.8544 | 0.8565 | 0.8565 | +0.0021 |

## For the auditor: what changed since the YR10m10 image (073dea33 -> d47f2692)

- **Driver code.** 55 commits touch the 29 files of `docker/closure.txt`: +2 401 / -101 lines in 18 files, 7 new files (`rmem.py`, `perturb.py`, `route_path.py`, `rule_expert.py`, `track_ref.py`,
  `turn_gain.py`, `vhist.py`). `sh30_driver.py`, `sh30_core.py` and `serve_fix.py` carry +404 / -41 of it: the ROUTE-MEM branch (`4bd6c3a5`: `rmem.Tracker` per session, sidecar `route_mem.json`) and
  switches that are all off here (`JEV_ACC`, `JEV_GAP`, `JEV_TRACK`, `JEV_VHIST`, `JEV_ROUTE`, `JEV_EXPERT*`, `JEV_PERTURB`, `JEV_TTS`, `JEV_DESIRE`, `JEV_STOP` / `JEV_REPLAN`, `SH30_RMEM`). `ap2_*` and `ens_driver.py` are unchanged.
  The base-model files (`op_interp.py`, `op_torch.py`, `op_adapt.py`, `parity_adapter.py`, `pp_train.py`, `lead_long.py`) changed by refactors and optional arguments (`warp_view` split out of
  `warp_frame`, LoRA / merged-weight helpers, `desire=None`, `t_follow` defaults, `noego=False`); their docstrings state unchanged defaults (`desire=None`: "the call unchanged (bit for bit)"; `safe_distance`: "the arithmetic as before, bit for bit"). The last closure change before the V2 reads
  is `a9380ddd`. The box checkout during the V2 reads (`6a20777b`; the next pull, `d7675c5a`, came at 11:29:39 box time) has the same 29 closure files and `docker/` as `d47f2692` (`git diff` empty), so the V2 reads
  ran the code of this image with its switches off. The RSG-L18-s0 reads of 10-13 ran older code (public read on `d4f19687`, held read on `f402f620`): 14 of the 29 closure files differ from this image (+1 953 / -91 lines; `rmem.py`
  +245 / -14), but the unordered memory path (`merge`, `window` without `prog`, `memory`, `Tracker.features` with `order=False`) is unchanged, today's code reproduced the stored held read score by score (decision 329, 452 of 452)
  and chunk b0 of the stored public read (`pai_fast.md`: 150 of 150 scores identical), and the memory those reads logged equals the image's bit for bit at decisions 0 and 1 (table above).
- **Not in the image** (imported only when their switch is set, none is): `tts.py`, `route_desire.py`, `intent_hold.py`, `reflex*.py`, `experiments/body1`.
- **Build files.** `Dockerfile`: two more `ARG` / `ENV` pairs, `JEV_ACC=0` and `JEV_VHIST=` (empty), baked next to `JEV_VCONT=0` / `JEV_LEAD=0` (they parse as off in `serve_fix.py` / `vhist.py`); `serve.py` prints the switches
  at start; `verify.py` also builds every checkpoint with a `route_mem.json` through `pai_driver.load_branch`; `build.sh` copies the sidecar; `test.sh` / `compare.py` gained PAI paths. The nuPlan path of `new_tag.sh` is the YR one.
  `requirements.lock` is unchanged. Origin/main moved on to `e5a9310e` (10 more commits) while this was written; none touches the closure or `docker/`, so the images are the closure of main.
- **Deviations from the official run** (as for the earlier images): the organisers' GPU and CPU are not ours (cm-level numerics; knife-edge scenes may flip either way); the reads used compiled passes and the image runs interpreted ones;
  the smoke covers the 48 public scenes of one log; the trusted image `alpasim-base:0.89.0` is Tokyo's 2026-10-09 build of floating package versions; one RTX 3090 renders and one drives.

## What the user runs (not run by an agent)

On the Tokyo box, with the CLI already configured for the earlier submissions (`docs/alpasim.md`, "What the user runs"; quota: 3 official submissions per UTC month, shared with the PAI track, and 3 warm-ups per 30 days).
The tag is public on the leaderboard; `latest` is refused. The team id in the URI is the one `ecr-login` prints: the nuPlan images went to `teams/lao-siji`, but the Tokyo Docker store also holds
`teams/ujs-iv:rsg-l18-s0-acc-v1` (the PAI push), so check which team repository the account is logged in to before tagging. The push should upload only the layers that are new (RSG image: 6 layers, about 230 MB uncompressed;
V2 image after the RSG image: 2 layers, 77.6 MB); the 12 lowest are the same as in the YR10m10 and PAI images already pushed (expected, not verified against the registry).

Once:

```bash
C="python3 /data/third_party/alpasim-cli/e2e_challenge/competitor_cli/alpasim_challenge.py"
$C me; $C limits                      # token valid 12 h; re-run auth-url / configure-token if expired; check the remaining official quota
$C ecr-login
```

RSG-L18-s0:

```bash
IMG=jev-alpasim:rsg-l18-s0-d47f2692                 # image id sha256:f99095eafd5453afe38c245b9d8b804a22c4af9e5a546bc5671ae7cd51ee92ee
URI=696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/lao-siji:rsg-l18-s0-v1
docker image inspect "$IMG" --format '{{.Id}}'      # must print sha256:f99095ea...
docker tag "$IMG" "$URI"
until docker push "$URI"; do sleep 10; done         # note the digest it prints
$C submit --track nuplan "$URI"                     # official: one of the 3 submissions of the UTC month (shared with PAI); a warm-up first: $C submit --track nuplan-warmup "$URI" (32 private scenes, status only)
$C status <submission_id>; $C submissions --track nuplan
```

RTB1ARO-L18-s0 (V2):

```bash
IMG=jev-alpasim:rtb1aro-l18-s0-d47f2692             # image id sha256:18a85838204ecc53050ecf1992134ee25c779ddcb035087fe04417910951ce27
URI=696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/lao-siji:rtb1aro-l18-s0-v1
docker image inspect "$IMG" --format '{{.Id}}'      # must print sha256:18a85838...
docker tag "$IMG" "$URI"
until docker push "$URI"; do sleep 10; done         # note the digest it prints
$C submit --track nuplan "$URI"                     # official; warm-up first: $C submit --track nuplan-warmup "$URI"
$C status <submission_id>; $C submissions --track nuplan
```

## Paused jobs

None. Both Tokyo cards were idle when the chain started at 15:04 JST (49 / 18 MiB, no compute process; the chain waited for under 1 000 MiB before each image, `test.sh` saw 18 MiB) and idle again at 15:26; nothing was stopped, so there is nothing to restart.

Last verified: 2026-10-17
