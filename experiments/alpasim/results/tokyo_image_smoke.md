# Submission image: constraint tests and the first containerised closed loop (2026-10-09, decision 215)

Read this when you build, test or submit the AlpaSim driver image. One image (`experiments/alpasim/docker/`) serves any openpilot + adapter
checkpoint of the `sh30` / `ap2` / `ens` drivers, chosen at run time; it was built on the Tokyo box, passed the organisers' published
container limits, and drove the 48 public scenes of `scenes_navtest_full_part001_48.txt` inside the official containerised stack
(AlpaSim unmodified at 0bb4c4b, their `alpasim-base` image built from their Dockerfile). Nothing was pushed, authenticated or submitted.
How to run it: [docs/alpasim.md](../../../docs/alpasim.md#submission-image-measured-2026-10-09-decision-215). Evidence files: `results/tokyo_image/`;
run dirs on the Tokyo box under `/data/runs/alpasim/tokyo_image/`.

## Result

Image `jev-alpasim:<tag>-6f3d05d1`, 8 027 812 004 bytes (7.48 GiB; 4.86 GB as `docker save | gzip -1`), one RTX 3090 for the driver. The card
differs from the organisers' (unknown, their preset is named `ec2`) and from the GPU box's RTX 6000D: latency and VRAM are 3090 numbers.

| | P2H10-F-s0 (`sh30` driver, image default) | AP2-AB-s0 (`ap2` driver, `-e JEV_TAG=AP2-AB-s0`) | limit |
|---|--:|--:|--:|
| Read-only root, `--network none`, uid 10001, caps dropped | pass (0 write errors, 0 changed files) | pass | |
| Image size | 7.48 GiB | same image | 40 GiB |
| Peak GPU memory, 2 concurrent rollouts (card total, `nvidia-smi`) | 3.04 GiB | 3.04 GiB | 16 GiB |
| `/tmp` high-water: probe / 48-scene smoke | 0.21 / 0.80 MiB | 0.21 / 0.80 MiB | 2 048 MiB |
| `/run` high-water | 0 | 0 | 64 MiB |
| Cold start, `docker run` -> service answers | 19.4 s | 19.8 s | runtime waits 240 s |
| `drive`, 2 concurrent synthetic rollouts, client side, p50 / p90 / p99 / max | 86.0 / 96.8 / 122.3 / 124.1 ms | 87.2 / 95.3 / 112.3 / 121.3 ms | target 100 ms |
| `drive` in the 48-scene smoke (8 concurrent rollouts, server side), p50 / p90 / p99 / max | 56.2 / 70.9 / 99.7 / 131.6 ms | 55.9 / 72.1 / 129.0 / 156.8 ms | |
| 48 scenes containerised: mean scene score / zeros / at 1 | **0.9839** / 0 / 37 | **0.9521** / 2 / 41 | |
| 48 scenes native (GPU box) | 0.9840 / 0 / 37 (from the 700-scene chunks) | 0.9320 / 3 / 41 (same 48-scene list) | |
| Scenes whose score differs | 11 of 48, max 0.0062 | 5 of 48: four <= 0.0061, one 0 -> 0.9719 | |
| `drive` calls / inference errors | 480 / 0 | 480 / 0 | |

Reading: every limit holds with a wide margin, and the containerised stack reproduces the native scores to the third decimal on 47 of 48
scenes per driver. It is not bit-identical (next section). In the 2-rollout probe the two rollouts fire without pause, so its 86 ms holds
about 29 ms of waiting for the other rollout's inference; model work per call (frame synthesis + encoder + policy) is 55-60 ms here.
The P2H10 reference is not a 48-scene-list run (none exists): its rows are taken from the three 700-scene chunk runs of lane OT3, and the
simulator is deterministic only for a fixed scene list (decision 211), so its 11 small differences mix that with what follows.

The whole chain for a tag (thin-layer rebuild, constraint tests, 48-scene smoke, comparison): **355 s** on the Tokyo box
(`new_tag.sh`, drill with AP2-AB-s0 placed in the extra layer; `results/tokyo_image/newtag_drill_report.md`).

## Why containerised != native, to the last digit

The containerised stack repeats itself exactly: two AP2 runs gave the same 48 scores scene by scene (`AP2-AB-s0/run2_vs_run1.md`), and two
more (with input dumps on, and the `new_tag.sh` drill) the same mean, zeros and count at 1. Against the native
run, 5 AP2 scenes differ; the one large change is `...01100_01664-368cb65e8fef57b7`, an at-fault collision natively, 0.9719 here.

| Stage | Check | Result |
|---|---|---|
| Renderer (their image: torch 2.8.0, gsplat 1.5.3 built for sm_86, RTX 3090; native: torch 2.9.1, sm_90 PTX build, RTX 6000D) | CAM_F0 JPEG of decision 0, same scene, both runs (4 scenes) | **byte-identical** (`renderdiff.jsonl`) |
| Driver CPU side (JPEG -> model frames) | the native JPEG through `sh30_core.pack` in the image | 89-162 of 393 216 values differ (max 11 levels): the nearest-pixel sampling maps round differently on this CPU (Ryzen 9950X vs Xeon 8470Q); same Pillow / libjpeg-turbo 3.1.4.1 |
| Driver GPU side (fp16 encoder + policy) | native slot frames + ego features through the model in the image (8 decisions) | plan differs by up to 1 fp16 step (0.0625 at 64-128 m); exported poses by 0.8 cm median, 2.4 cm max (`replay_ap2.jsonl`) |
| Closed loop | same scenes at decision 9 | frames differ in 64-95 % of pixels, plans by 0.16-0.51 m: the ego is somewhere else by then |

So the renderer is not the source: the first frame is the same file, and the first plans differ by 0.1-1.6 cm from driver numerics on a
different GPU / CUDA build and CPU. The loop amplifies that (decision 211 saw a 1.5 cm plan difference change a scene score), which is
enough to move a knife-edge collision either way. Not separated: torch `+cu126` against `+cu130` on one card, and RTX 3090 against RTX
6000D with one build. The organisers' cards are a third case; a public-suite score from the box is an estimate of the official one to
within this noise, not a prediction of individual scenes.

## What the tests are

- `docker/test.sh <image> <out> [JEV_TAG] [gpu]`: the container with the flags of the organisers' `run_local_container.sh`
  (`--read-only --cap-drop ALL --security-opt no-new-privileges:true --pids-limit 1024 --memory 32g --cpus 8 --tmpfs /tmp:size=2g
  --tmpfs /run:size=64m`, uid 10001) plus `--network none`; `probe.py` inside it drives 2 concurrent rollouts x 8 (10 decisions each, 8
  JPEGs of 1920 x 1080 per step). Exact command lines: `tokyo_image/<tag>/test_cmd.txt`. `docker diff` lists only what the NVIDIA
  runtime injects.
- `docker/smoke.sh <image> <out> [JEV_TAG] [reference]`: the same container on a Docker network created with `--internal` (the smoke checks
  that it cannot reach the internet), the wizard run inside `alpasim-base:0.89.0` with `wizard.run_method=NONE`, then
  `docker compose up --exit-code-from runtime-0` (renderer, controller, runtime + eval in their image), with the wizard overrides of the
  native reference runs (`tokyo_image/wizard_cmd.txt`). `compare.py` writes the table.

## Found on the way

- **Per-thread first-inference cost.** In the image the first `drive` served by each gRPC worker thread took 1.35 s in the encoder
  (8 threads: 8 calls of 1.4-2.8 s in the first 12 s of a replica); the native logs show none (max encode 37.5 ms in 2 340 calls). The
  drivers now run one inference in every worker thread before the port opens (`sh30_driver.warm_workers`): cold start 7.5 -> 19 s,
  probe p99 2 745 -> 122 ms. The only driver change; outputs are untouched.
- **The official nuPlan config cannot render on an RTX 3090.** `e2e_challenge_nuplan_common/base.yaml` sets
  `TORCH_CUDA_ARCH_LIST=8.9;9.0+PTX` for gsplat's kernel build; a 3090 is sm_86 ("no kernel image is available"). `smoke.sh` reads the
  render card's capability and, below 8.9 only, overrides that one list on the wizard command line. A deviation of the Tokyo smoke.
- **Wheel downloads.** Through the Tokyo proxy one connection to the wheel CDNs gave 0.25 MB/s (torch alone 40 min with pip, and a failed
  try starts over). `wheels.py` fetches the files of `requirements.lock` in 16 parallel ranges and checks each sha256 against the lock
  (4.13 GiB in 867 s); the image installs from that directory with `uv pip install --require-hashes`. The torch wheel's hash equals the
  one published on download.pytorch.org.
- One scene asset on the Tokyo box was truncated during the first P2H10 smoke (renderer: "failed finding central directory", one failed
  rollout, 0.9630); the bytes lane repaired it and the rerun is the row above. That first run dir was deleted in the cleanup.

## What the image contains

`tokyo_image/layers.txt` (layers), `tokyo_image/MANIFEST.sha256` (every file staged into the build). No repo history, no data, no
secrets, no training code beyond the three modules the model class imports.

| Component | What | Licence |
|---|---|---|
| Base | `python:3.11-slim-bookworm` (Debian 12.15, CPython 3.11.17) | Debian packages (various free licences); PSF-2.0 |
| torch 2.14.0+cu126, triton 3.8.0, sympy, networkx, jinja2, fsspec, filelock, mpmath, typing_extensions | PyTorch and its dependencies | BSD-3-Clause / Apache-2.0 (torch), MIT, BSD, PSF-2.0 |
| `nvidia-*-cu12` (cuBLAS, cuDNN 9.10.2, cuFFT, cuSPARSE, cuSOLVER, NCCL, nvJitLink, runtime, ...), `cuda-toolkit` | CUDA 12.6 libraries from PyPI, 3.6 GiB | **NVIDIA proprietary** (redistributable inside an application under the CUDA / cuDNN EULAs; not Apache-2.0) |
| numpy 2.4.6, scipy 1.17.1, pandas 3.0.5, python-dateutil, six | | BSD-3-Clause (and bundled 0BSD / MIT / Zlib), Apache-2.0 / BSD |
| opencv-python-headless 4.11.0.86, onnx 1.23.0, ml_dtypes, grpcio 1.84.0, cuda-bindings | | Apache-2.0 (the OpenCV wheel bundles FFmpeg libraries under LGPL) |
| pillow 12.3.0, protobuf 7.36.2, setuptools, pip, wheel, packaging | | MIT-CMU, BSD-3-Clause, MIT, Apache-2.0 / BSD |
| `alpasim_grpc` 0.54.0 (`src/grpc`), `navsim_transfuser_challenge/{navigation,trajectory}.py` | NVlabs/alpasim at 0bb4c4b, unmodified | Apache-2.0 (NVIDIA) |
| `cinque.ort.onnx`, 766 354 813 bytes, sha256 `3f12a0cd...` | openpilot driving model "Cinque" (comma.ai), the frozen base of every driver | MIT (openpilot) |
| `ckpt-final.pt` per tag (P2H10-F-s0 `07153f0d...`, AP2-AB-s0 `4547a7b3...`), 75 MB each | our adapter + plan-pathway weights, trained on navtrain (nuPlan / OpenScene logs) | ours; the training data is nuPlan (CC BY-NC-SA 4.0), which the user has to weigh against any open-licence requirement in the terms |
| 18 repo files (`docker/closure.txt`), `serve.py`, `probe.py`, `verify.py` | driver code | ours (the repo has no licence file) |

## Not verified

- The organisers' host: its GPU model and driver version (by NVIDIA's release notes, quoted from memory, CUDA 12.6 libraries need driver
  >= 560.28 and the CUDA 12.8 ones in their own image >= 570.26, so ours is the weaker requirement), its CPU allowance, and the exact `docker run` flags of their runner
  (not in the repo; the flags above are those of their local script, which their README calls "the official container restrictions
  except outbound network blocking").
- 16 replicas on 4 GPUs: four instances per card were not run together (4 x 3.04 GiB by arithmetic). The throughput budget of the track
  is shown only in a submission's status.
- The private scenes, and anything about the warm-up or official run itself.
- The ensemble driver (`JEV_TAG=tagA+tagB`): it is in the image and imports at build time, but no container run was made.
- The terms and conditions (need an approved account): the "public under Apache-2.0" requirement is second-hand.

Last verified: 2026-10-09
