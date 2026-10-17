# PAI submission image: build, limits, replicas, containerised smoke against native, audit (2026-10-16)

Read this before you push or submit a PAI-track image. The image path (`experiments/alpasim/docker/`) gained a `pai` driver family on
2026-10-16 (commit 382164ae): one image serves a PAI checkpoint picked at build or run time through `pai_driver.py` with the serving switches
of the box reads. How to build and what the user runs: [docs/alpasim.md](../../../docs/alpasim.md#pai-submission-image-2026-10-16). Small
files of every step: [pai_image/](pai_image/) (test / replicas `result.json`, smoke `compare.md` / `smoke.json` / `cmp_*.md`, build manifests
and layers). Tokyo artefacts: `/data/runs/alpasim/pai_image/`. Nothing was pushed, logged in or submitted by an agent.

## Images (Tokyo box, built from 382164ae, `TRACK=pai`)

| label | build command (on Tokyo, repo root) | image | id |
|---|---|---|---|
| rsg | `env DATA_DIR=/data TRACK=pai TAG=RSG-L18-s0 experiments/alpasim/docker/build.sh` | `jev-alpasim:rsg-l18-s0-382164ae-pai-vc1.0-lead` | `sha256:4f3bd3f4...` |
| rsghl | `... TAG=RSG-L18-s0 VHIST=holdlead ...build.sh` | `jev-alpasim:rsg-l18-s0-382164ae-pai-vc1.0-lead-vhholdlead` | `sha256:a5a93c3c...` |
| rsgacc | `... TAG=RSG-L18-s0 ACC=1 ...build.sh` | `jev-alpasim:rsg-l18-s0-382164ae-pai-vc1.0-lead-acc` | `sha256:67f002e9...` |
| rbsg2 | `... TAG=RBSG2-L18-s0 ...build.sh` | `jev-alpasim:rbsg2-l18-s0-382164ae-pai-vc1.0-lead` | `sha256:bd48b58d...` |
| rsgaccgap | `... TAG=RSG-L18-s0 ACC=1 GAP=0.7,2.7 ...new_tag.sh` (built from ea61e9df, 2026-10-18, section below) | `jev-alpasim:rsg-l18-s0-ea61e9df-pai-vc1.0-lead-acc-gap0.7_2.7` | `sha256:328433cb...` |

Full ids in [pai_image/images.txt](pai_image/images.txt). All four share every layer up to the checkpoints (both candidates are in every
image, `RSG-L18-s0` and `RBSG2-L18-s0` with their `route_mem.json`; sha256 equal to the box: `fbe7ac9f` / `672ef154`, `5572c7e8` / `b667704a`)
and differ only in the last `ENV` layer. Every image: `JEV_DRIVER=pai`, `JEV_VCONT=1.0`, `JEV_LEAD=1`, `JEV_TRACK` unset (pai_driver.py then
serves the v8 `gate` reference, as every box read), `SH30_COMPILE=0` (interpreted passes; the PAI driver never compiles, natively either).
A combination (RBSG2 + holdlead, ACC + holdlead) is the same command with both variables, one new `ENV` layer, seconds.
`verify.py` (inside `docker build`, CPU only) loads both ROUTE-MEM checkpoints through pai_driver's own loader (memory dim 200, unordered
layout, no reflex module).

## Limits (`test.sh`, card 1, RTX 3090; organisers' flags + `--network none`; PAI probe: 2 concurrent rollouts x 2, 200 decisions at 10 Hz, 6 f-theta cameras)

| check | rsg | rsghl | rsgacc | rbsg2 | limit |
|---|--:|--:|--:|--:|--:|
| image size | 7.48 GiB | same | same | same | 40 GiB |
| read-only root: write errors / changed files | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 |
| GPU memory of the driver | 1.44 GiB | 1.44 | 1.44 | 1.44 | 16 GiB |
| `/tmp` high-water (probe; 29-scene smoke) | 2.6; 24 MiB | 2.8; 26 | 2.7; 25 | 2.6; 24 | 2048 MiB |
| `/run` high-water | 0 | 0 | 0 | 0 | 64 MiB |
| cold start (`docker run` -> listening) | 18.8 s | 17.9 | 17.9 | 17.7 | runtime waits 240 s |
| `drive`, client p50 / p90 / p99 / max (ms) | 48 / 54 / 117 / 187 | 48 / 56 / 115 / 188 | 47 / 54 / 115 / 190 | 49 / 57 / 116 / 188 | target 100 |
| SIGTERM -> exit 0 | 0.6 s | 0.6 | 0.6 | 0.6 | |
| `get_version` | `jev-pai-RSG-L18-s0-backwarp-vc1-lead-trackgate` | `...-trackgate-vhholdlead` | `...-vc1-lead-acc-trackgate` | `jev-pai-RBSG2-L18-s0-...-trackgate` | |

`/tmp` grows by the driver's `drive.jsonl`, about 0.85 MB per 20 s rollout: 2 GiB holds about 2 400 rollouts per replica (the official run
gives a replica a few dozen to a few hundred).

**Replicas** (`replicas.sh`, rsg): 4 replicas of the image on the one 3090 at once (the ec2 topology `8gpu_32rollouts` puts 16 replicas on 4
cards, 4 per card, 2 rollouts each), each `--network none` with its own replica index, probe running in all four: pass. 1.43 GiB per replica
(5.8 GiB card), cold start 19.8-20.8 s, one version string across replicas, 0 tracebacks, 3 200 `drive` calls. Client-side `drive` p50 /
p90 / max 167 / 174 / 449 ms: 8 rollouts sent as fast as the driver answers saturate one 3090, every call waits behind the others. The
organisers' cards (p5: H100) and pacing (the runtime waits for the renderer between calls) differ; the per-track throughput budget is shown
only in the submission status. In the real stack (one replica, 4 concurrent rollouts on a 3090 shared with the renderer) the server-side
`drive` is p50 55 / p90 73-78 / p99 139-144 ms.

## Smoke in the official containerised PAI stack (Tokyo card 1, steering weight 10, the images as built, no driver env)

`pai_smoke.sh`: the driver container exactly as the organisers start it (own `CMD`, no mounts, replica variables only, test.sh limits, a Docker
network without a route out; a reach of pypi.org fails), the wizard in `alpasim-base:0.89.0` with `+e2e_challenge=dev`, harmonizer off as in
the ec2 preset, `controller.gains.rel_front_steering_angle_weight=10`, 4 concurrent rollouts, renderer + physics + driver on the one card.
0 inference / input errors in every run (25 191 `drive` calls), 0 driver tracebacks. Comparison: `scripts/pai_image_cmp.py` (scores and the
served plan of every decision).

**Same host, same code path: bit-identical.** The 6-scene list of the ACC prereg's Tokyo smoke, against those runs (the nuPlan image with
HEAD's PAI files mounted, `pai_run.sh`, same 3090 model):

| run | image | Tokyo mounted driver | scenes with identical scores | scenes with identical served plans, all decisions |
|---|--:|--:|--:|--:|
| six_rsg (default serve) vs `pai_acc_smoke/off` (card 1) | 0.0876 | 0.2543 | 5 of 6 | 5 of 6 |
| six_acc (JEV_ACC=1) vs `pai_acc_smoke/acc` (card 0) | 0.4179 | 0.4179 | 6 of 6 | 6 of 6 |

The one difference, 49bcab6f in six_rsg (0 against 1.0), is not the driver: the NRE renderer raised `Expected all tensors to be on the same
device, cuda:0 and cpu` in `nre/grpc/ego_hood.py:99` (`overlay_on_image`), right after two concurrent rollouts logged an `EgocarHood: resize
cache miss` in the same millisecond, and the runtime failed the rollout (not retryable). The driver logged no error and its last `drive`
returned normally; in six_acc the same scene is 1.0. The same renderer error is in 4 of 88 native box chunk logs (one rollout each, about
0.13 % of rollouts, among them the g10 reference's c000, scene 13d646af): the box reads carry it too. The ec2 preset runs the same renderer
image with the same hood option, so an official run can lose a rollout to it (scored 0; `allow_aggregation_with_failed_rollouts: true`); the
ec2 renderer serves one rollout per replica, which should make this race rarer (not verified). No driver can prevent it.

**Cross host: the image on Tokyo against native on the box** (chunk c007 of the gains-10 reads, 29 scenes, identical list, `pai_g10/<label>`):

| image | box native | image mean (zeros) | native mean (zeros) | paired diff [95 % CI] | equal / within 0.01 / within 0.05 | zero agreement | zero flips |
|---|---|--:|--:|--:|--:|--:|---|
| rsg | g10 | 0.4932 (10) | 0.4300 (12) | +0.063 [-0.002, +0.162] | 18 / 25 / 27 | 27 / 29 | 1bfad87c 1.0 vs 0, 4e4a2a51 0.87 vs 0 |
| rsgacc | acc | 0.5109 (13) | 0.4774 (14) | +0.033 [-0.003, +0.103] | 25 / 27 / 28 | 28 / 29 | 4e4a2a51 1.0 vs 0 |
| rsghl | g10hl | 0.4491 (12) | 0.4502 (12) | -0.001 [-0.104, +0.102] | 18 / 25 / 27 | 27 / 29 | 2c263e19 0 vs 1.0, 4e4a2a51 1.0 vs 0 |
| rbsg2 | rbsg2g10 | 0.3630 (13) | 0.3368 (14) | +0.026 [-0.001, +0.080] | 18 / 28 / 28 | 28 / 29 | 4e4a2a51 0.77 vs 0 |

Decision by decision (`pai_image/smoke/c007_*/cmp_box.md`): no scene has identical served plans across hosts (the box runs torch
`+cu130` on an RTX 6000D, the image `+cu126` on a 3090). The first decision, where both sides see the same rendered frame and ego state, differs
by a median 0.1-0.7 cm (rsg 0.54, rsgacc 0.12, rsghl 0.54, rbsg2 0.66 cm), the cm-level numerics of decision 215. One scene, 35e082a4, differs by
13.6-18.8 m already at decision 0 in all four (identical ego features, route and command on both sides; the image plans 5 m in 4 s
where the box plans 19 m): a knife-edge choice between the plan's stop and go modes flipped by those numerics (inferred, the frames were not
dumped); it is a corridor zero on both hosts.

Reading: this is decision 215's cross-host drift (the ACC prereg's parity check found box vs Tokyo +0.005 [-0.008, +0.019] on 281 scenes,
zeros agreeing 276 / 281, 1bfad87c among the box-only zeros). 4e4a2a51 is an offroad zero on the box in all four configurations and scored
in the image in all four: a host property of that scene, not of the image. Means of a 29-scene chunk are not a read; the configuration is
chosen on the box's 281-scene reads (plans/2026-10-16-pai-gains10-prereg.md, pai-acc-prereg.md).

## ACC-GAP image (2026-10-18, decision 330: RSG-L18-s0 + `JEV_ACC=1` + `JEV_GAP=0.7,2.7`, steering weight 10 at submit)

Built on Tokyo from ea61e9df (build.sh gained the `GAP` build argument there), one chain:
`env DATA_DIR=/data TRACK=pai TAG=RSG-L18-s0 ACC=1 GAP=0.7,2.7 SCENES=/data/runs/alpasim/pai_image/lists/c007.txt OUT=/data/runs/alpasim/pai_image/rsgaccgap TGPU=1 experiments/alpasim/docker/new_tag.sh RSG-L18-s0 /data/runs/alpasim/pai_image/native/accgap/aggregate/results-summary.json`.
Image `jev-alpasim:rsg-l18-s0-ea61e9df-pai-vc1.0-lead-acc-gap0.7_2.7`, id `sha256:328433cbab611095543e1846ad379f5245c06f5f82f212fbe068a4e3d3ef2679`
(8 033 245 840 bytes). Files: [build/rsgaccgap/](pai_image/build/rsgaccgap/), [test/rsgaccgap/](pai_image/test/rsgaccgap/),
[replicas/rsgaccgap/](pai_image/replicas/rsgaccgap/), [smoke/c007_rsgaccgap/](pai_image/smoke/c007_rsgaccgap/) (`report.md` = the chain's table).

**Against rsgacc.** Layers 1-12 (base, packages, Cinque, AlpaSim files) are the same layers. Metadata: one new `ARG GAP` + `ENV JEV_GAP=0.7,2.7`
after the switches `ENV`, and `GIT_HASH` 382164ae -> ea61e9df. Code layer (`COPY jev-drive`, 508 -> 530 kB): 9 closure files changed between
382164ae and ea61e9df on purpose (serve_fix.py and lead_long.py carry `JEV_GAP`; pai_core / pai_driver / sh30_* / rmem / pp_train / ap2_driver
gained flag-gated switches, all off in this image; the serve code equals the d330 native read's 94711ca9 for serve_fix.py / lead_long.py).
The layers after it (`serve.py`..., user, checkpoints, `verify.py`) were rebuilt because the cache broke above them; their contents are unchanged
(manifest sha256 equal; checkpoints inside the image `fbe7ac9f` / `672ef154`, `5572c7e8` / `b667704a`, as the box). `get_version`
`jev-pai-RSG-L18-s0-backwarp-vc1-lead-acc-gap0.7_2.7-trackgate` (one string across 4 replicas).

| check | rsgaccgap | limit | |
|---|--:|--:|---|
| image size | 7.48 GiB | 40 GiB | pass |
| read-only root: write errors / changed files | 0 / 0 | 0 | pass |
| GPU memory of the driver | 1.44 GiB | 16 GiB | pass |
| `/tmp` high-water (probe; 29-scene smoke) | 2.7; 25 MiB | 2048 MiB | pass |
| `/run` high-water | 0 | 64 MiB | pass |
| cold start | 17.7 s | runtime waits 240 s | pass |
| `drive`, client p50 / p90 / p99 / max (ms) | 47 / 53 / 114 / 192 | target 100 | as the others |
| 4 replicas on one 3090 | 1.43 GiB per replica (card 5.76 GiB), cold start max 20.3 s, 1 version, p50 / p90 / max 166 / 172 / 457 ms | 16 GiB | pass |
| smoke `drive`, server side | p50 56 / p90 73 / max 239 ms, 5 752 calls, 0 inference errors | | pass |

**Smoke** (chunk c007, 29 scenes, Tokyo card 1, official containerised stack, weight 10, image as built) against the box's native ACC-GAP read
(`route_follow/acc_gap/read/s2/runs/rsg-l18-s0_c007`, d330 stage 2; results-summary + drive.jsonl copied to `pai_image/native/accgap`):

| image | box native | image mean (zeros) | native mean (zeros) | paired diff [95 % CI] | equal / within 0.01 / within 0.05 | zero agreement | zero flips |
|---|---|--:|--:|--:|--:|--:|---|
| rsgaccgap | accgap | 0.5519 (12) | 0.5502 (12) | +0.002 [-0.001, +0.007] | 25 / 27 / 28 | 29 / 29 | none |

The zero set and its reasons are identical (2 at-fault collisions 1a69f757 / 3ce13eed, 1 offroad, 9 corridor). Decision 0 plan difference median
0.11 cm (decision 215 numerics); 35e082a4 differs by 1.9 m at decision 0 (the knife-edge stop / go scene above; a corridor zero on both).
The four differing scenes are non-zero on both sides (largest 52bd89ce 0.35 vs 0.29). 4e4a2a51, the box-only offroad zero of the four earlier
images, scores 1.0 on both hosts under ACC-GAP. Nothing pushed, tagged to ECR or submitted by an agent.

## Audit

AUDIT_PENDING

## Deviations of the local checks from the official run

- One RTX 3090 (sm_86) for renderer, physics and driver; the official run uses H100s (driver on GPUs 4-7, renderer on 0-3) and the `ec2`
  preset (16 replicas x 2 rollouts, renderer 8 replicas x 1 rollout per card); the smoke ran the `dev` preset's 1-GPU topology at 4 concurrent
  rollouts with the harmonizer removed as `ec2` does. `ec2` also sets `render_bundling: BATCH_RENDER_RGB` (images rendered as one batch per
  step); the dev preset does not, so the per-camera image arrival order differs; the driver aligns by timestamp and decodes only the front
  wide camera.
- The trusted image is Tokyo's 2026-10-09 build of `alpasim-base:0.89.0` (floating package versions).
- The scenes are curated-validation scenes (the public local set); the official PAI suite is private.
