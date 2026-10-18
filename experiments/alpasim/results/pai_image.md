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
| ntv0accgap | `... ACC=1 GAP=0.7,2.7 ...new_tag.sh NTV0-L18-s0` (built from 0d417ee4, 2026-10-19, section "NTV0 ACC-GAP image") | `jev-alpasim:ntv0-l18-s0-0d417ee4-pai-vc1.0-lead-acc-gap0.7_2.7` | `sha256:37183c04...` |

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

## NTV0 ACC-GAP image (2026-10-19, decision 366: NTV0-L18-s0 + `JEV_ACC=1` + `JEV_GAP=0.7,2.7`, steering weight 10 at submit)

The probe's serve configuration (decision 356) with the checkpoint swapped. Built on Tokyo from 0d417ee4 (main; the driver closure is
byte-equal to the native read's 2e2d0e46 and, apart from the nuPlan-only hold files, to ea61e9df), one chain:
`env DATA_DIR=/data TRACK=pai ACC=1 GAP=0.7,2.7 SCENES=/data/runs/alpasim/pai_image/lists/c007.txt OUT=/data/runs/alpasim/pai_image/ntv0accgap TGPU=1 experiments/alpasim/docker/new_tag.sh NTV0-L18-s0 /data/runs/alpasim/pai_image/native/ntv0accgap/aggregate/results-summary.json`.
Image `jev-alpasim:ntv0-l18-s0-0d417ee4-pai-vc1.0-lead-acc-gap0.7_2.7`, id `sha256:37183c048a06965ef588d1b7c6810865018956bf926ebf8e0718f0da616a0558`
(8 110 811 511 bytes). Files: [build/ntv0accgap/](pai_image/build/ntv0accgap/), [test/ntv0accgap/](pai_image/test/ntv0accgap/),
[replicas/ntv0accgap/](pai_image/replicas/ntv0accgap/), [smoke/c007_ntv0accgap/](pai_image/smoke/c007_ntv0accgap/) (`report.md` = the chain's
table, `cmp_box.md` decision by decision), [smoke/c007_ntv0accgap_rep/](pai_image/smoke/c007_ntv0accgap_rep/) (same-host repeat).

**Checkpoint.** NTV0-L18-s0 enters through the thin `ckpt-extra` layer (not in the stable RSG / RBSG2 layer, which is unchanged):
`ckpt-final.pt` sha256 `5fa544ac...655b`, `route_mem.json` `f5c25900...7816`, equal on the AutoDL box, Tokyo `/data/runs` and the build
manifest (copied box -> Mac -> Tokyo on 2026-10-19 because `tokyo_pull.sh` found no Clash node that reached the box). The sidecar's `head` is
`{d 128, layers 2, heads 4}`, with no `order`, `anchor` or `reflex` key: the projected route-memory layout (rmem `SERVE_ORDER` / `SERVE_ANCHOR`
False, as RSG-L18-s0), init RB0-F-s0, `on1_JT` / `nr1_lat`, 9 500 steps.

**Against rsgaccgap.** Build manifest: only the two NTV0 files are new, plus `sh30_hold.py` / `static_hold.py` (nuPlan hold, in the closure since
e7db399e, imported only by verify.py and by serve.py when `JEV_HOLD` is set) and serve.py / verify.py (the HOLD dispatch: with `JEV_HOLD` empty serve.py
execs `pai_driver.py` as before). Every other closure file (pai_core, pai_driver, serve_fix, lead_long, rmem, sh30_core, ...) and the stable checkpoints
are byte-equal to rsgaccgap. Metadata: `JEV_TAG=NTV0-L18-s0`, `SH30_GIT_HASH=0d417ee4`, plus the empty `ENV JEV_HOLD=` layer; `JEV_VCONT=1.0
JEV_LEAD=1 JEV_ACC=1 JEV_VHIST= JEV_GAP=0.7,2.7`, `JEV_TRACK` unset (driver logs `JEV_TRACK gate`), `SH30_COMPILE=0`, identical to rsgaccgap. The driver's start line in the
smoke: `switches JEV_VCONT=1.0 JEV_LEAD=1 JEV_VHIST= JEV_ACC=1 JEV_GAP=0.7,2.7 JEV_HOLD=`, ROUTE-MEM branch on. `get_version`
`jev-pai-NTV0-L18-s0-backwarp-vc1-lead-acc-gap0.7_2.7-trackgate`, the same string as the native read's (one string across 4 replicas).

| check | ntv0accgap | limit | |
|---|--:|--:|---|
| image size | 7.55 GiB | 40 GiB | pass |
| read-only root: write errors / changed files | 0 / 0 | 0 | pass |
| GPU memory of the driver | 1.44 GiB | 16 GiB | pass |
| `/tmp` high-water (probe; 29-scene smoke) | 2.8; 25 MiB | 2048 MiB | pass |
| `/run` high-water | 0 | 64 MiB | pass |
| cold start | 17.6 s | runtime waits 240 s | pass |
| `drive`, client p50 / p90 / p99 / max (ms) | 47 / 53 / 115 / 191 | target 100 | as the others |
| SIGTERM -> exit | 0.6 s, code 0 | | pass |
| 4 replicas on one 3090 | 1.43 GiB per replica (card 5.76 GiB), cold start max 20.5 s, 1 version, p50 / p90 / max 167 / 173 / 388 ms | 16 GiB | pass |
| smoke `drive`, server side | p50 57 / p90 76 / max 199 ms, 5 752 calls, 0 inference / input errors | | pass |

**Smoke** (chunk c007, 29 scenes, the same list as rsgaccgap's smoke; Tokyo card 1, official containerised stack, weight 10, image as built, no driver env)
against the box's native NTV0 + ACC + GAP read (`defect_fix/pai_accgap/NTV0-L18-s0/runs/ntv0-l18-s0_c007`, code 2e2d0e46, the d366 PAI 281 read;
results-summary + drive.jsonl copied to Tokyo `pai_image/native/ntv0accgap`):

| image | box native | image mean (zeros) | native mean (zeros) | paired diff [95 % CI] | equal / within 0.01 / within 0.05 | zero agreement | zero flips |
|---|---|--:|--:|--:|--:|--:|---|
| ntv0accgap | ntv0 accgap | 0.5403 (13) | 0.6090 (11) | -0.069 [-0.172, +0.001] | 25 / 27 / 27 | 27 / 29 | 2bbf1efc 0 vs 1.0, 3232fdb7 0 vs 1.0 (both corridor) |
| ntv0accgap, repeat (card 0) | image run 1 | 0.5403 (13) | 0.5403 (13) | +0.000 | 29 / 29 / 29 | 29 / 29 | none; served plans identical at every decision in 29 / 29 |

Decision 0 (same rendered frame and ego state on both hosts): plan difference median 0.06 cm (rsgaccgap 0.11 cm); the cm-level numerics of decision 215,
so code, checkpoint and switches agree with the native read. One scene, 1a5da90a, differs by 1.28 m at decision 0 (scored 1.0 on both).
The two flips are not at decision 0: on 2bbf1efc the served plans part by > 0.1 m at decision 4 and > 1 m at decision 16, on 3232fdb7 at
decisions 24 and 69, the closed loop amplifying the cross-host numerics. Both are corridor zeros of RSG + ACC + GAP on both hosts (rsgaccgap
smoke table), i.e. scenes that NTV0 only just recovers. The other two differing scenes are non-zero on both sides (404541da 0.765 vs 0.757,
478d6ec4 0.9028 on both to 4 places); 4e4a2a51 is a zero on both with another reason (at-fault collision vs corridor). The repeat on the other Tokyo card is bit-identical to run 1, so the image is deterministic on
one host and the gap is the box (`+cu130`, RTX 6000D) against Tokyo (`+cu126`, RTX 3090), as rsg's image showed +0.063 with 2 flips the other way on
this chunk (section above). Reading: the image serves what the box read served; this chunk's -0.069 is two knife-edge corridor scenes of 29,
not a read (d366's PAI 281 margin is +0.0545 [+0.024, +0.088] over 280 scenes on one host). The organisers' H100s are a third host, so the
private score carries this kind of drift on both sides. Nothing pushed, tagged to ECR or submitted by an agent.

**What the user runs** (Tokyo, profile acct3 = the vennai-org team of the decision 356 probe; none of it by an agent; quota: 3 official per UTC
month shared with nuPlan, 3 `pai-warmup` per 30 days):

```bash
# Tokyo. ~/bin/ap <profile> = the competitor CLI (or docker) with that profile's config / ECR login
~/bin/ap acct3 me; ~/bin/ap acct3 limits                      # session alive? (re-auth: ~/bin/ap acct3 auth-url; ~/bin/ap acct3 configure-token)
~/bin/ap acct3 ecr-login
IMG=jev-alpasim:ntv0-l18-s0-0d417ee4-pai-vc1.0-lead-acc-gap0.7_2.7
URI=696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/vennai-org:ntv0-l18-s0-acc-gap-v1   # team slug as `ap acct3 me` prints it; public tag, not `latest`
docker image inspect "$IMG" --format '{{.Id}}'                # must print sha256:37183c048a06965ef588d1b7c6810865018956bf926ebf8e0718f0da616a0558
docker tag "$IMG" "$URI"; until ~/bin/ap acct3 docker push "$URI"; do sleep 10; done   # the layers below the code layer are already in the registry
echo '{"rel_front_steering_angle_weight": 10}' > /data/runs/alpasim/pai_image/gains10.json
~/bin/ap acct3 submit --track pai-warmup --controller-gains /data/runs/alpasim/pai_image/gains10.json "$URI"   # optional status-only check
~/bin/ap acct3 status <submission_id>                         # first rollout's version must be jev-pai-NTV0-L18-s0-backwarp-vc1-lead-acc-gap0.7_2.7-trackgate
~/bin/ap acct3 submit --track pai --controller-gains /data/runs/alpasim/pai_image/gains10.json "$URI"          # official
```

## Audit

**ntv0accgap (`sha256:37183c04...`), independent pre-submission audit, 2026-10-19: GO.** Checked on the image on Tokyo (inspect, `docker run
--network none` with entrypoint overrides, one own `test.sh` run on card 1, own runs of `pai_image_cmp.py` and an own bootstrap), against the
AutoDL box originals, not only on the records above.

1. Identity and config: pass. Id `sha256:37183c048a06...0558`, 8 110 811 511 bytes. Sorted ENV against rsgaccgap differs in exactly three
   lines: `JEV_TAG` RSG-L18-s0 -> NTV0-L18-s0, `SH30_GIT_HASH` ea61e9df -> 0d417ee4, new `JEV_HOLD=` (empty). `JEV_DRIVER=pai JEV_VCONT=1.0
   JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7 JEV_VHIST= SH30_COMPILE=0`, no `JEV_TRACK` (driver logs `JEV_TRACK gate`), no other `JEV_*` / `SH30_*`.
   CMD `python /app/serve.py`, no entrypoint, user 10001:10001, port 6789, SIGTERM, workdir /app: as rsgaccgap. No steering weight baked in.
2. Layers and code: pass. RootFS layers 1-12 (base, packages, wheels, models 766 363 005 B, AlpaSim files) identical to rsgaccgap; 13-18 (code,
   serve files, user, ckpt, ckpt-extra, verify) differ. sha256 of every file under /app except models (48 vs 44 files): new NTV0
   `ckpt-final.pt` / `route_mem.json`, new `sh30_hold.py` / `static_hold.py` (absent at ea61e9df), changed serve.py / verify.py / tags.json
   (+ `"NTV0-L18-s0": "sh30"`); everything else, including RSG / RBSG2 checkpoints, pai_spec.json, probe.py and every other closure file,
   byte-equal. serve.py's diff ea61e9df..0d417ee4 is the HOLD dispatch only: `JEV_HOLD` empty -> `pai_driver.py` exec'd as before (start line
   `serve: driver pai (pai_driver.py)`); verify.py only adds `import sh30_hold`. All 35 baked code files equal `git show 0d417ee4`; all
   /app/jev-drive files equal `git show 2e2d0e46` (the box read's `config.txt`: `code=2e2d0e46`, `SERVE=JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1
   JEV_GAP=0.7,2.7`, `WIZ=...rel_front_steering_angle_weight=10`).
3. Checkpoint: pass. `ckpt-final.pt` `5fa544ac...155b`, `route_mem.json` `f5c25900...7816`, equal in the image, on Tokyo `/data/runs` and on
   the AutoDL original. Sidecar `head` `{d 128, layers 2, heads 4}`, no `order` / `anchor` / `reflex` key; pai_driver sets
   `RMEM.SERVE_ORDER` / `SERVE_ANCHOR` from `head_order` / `head_anchor` (-> False, projected layout) and attaches reflex only on a sidecar
   `reflex`. `verify.py` under `--read-only --network none`, no GPU: `branch NTV0-L18-s0 dim 200 order False reflex False`; only the
   build-time tags.json write fails, as it should.
4. Limits: pass. Own `test.sh` (organisers' hardening flags, read-only, `--network none`, card 1, card idle at 18 MiB): listening after 17.6 s,
   800 `drive` calls server p50 46.3 / p99 119 ms, 0 inference errors, 0 tracebacks, 0 read-only errors, 0 changed files, /tmp 2.8 MiB,
   /run 0, GPU 1.47 GiB, SIGTERM -> exit 0 in 0.6 s, version `jev-pai-NTV0-L18-s0-backwarp-vc1-lead-acc-gap0.7_2.7-trackgate`, git 0d417ee4,
   `pass: true` (`/data/runs/alpasim/pai_image/audit_ntv0accgap/test`). Size 7.55 GiB < 40 GiB.
5. Secrets: pass. `docker history --no-trunc`: no token, proxy, credential URL or host address (only the Python base's public `GPG_KEY` id).
   Filesystem: no .env, .netrc, .git-credentials, .alpasim, ssh key / authorized_keys / known_hosts, .git; the only *.pem are public CA
   bundles, none with a private key; no hf_ / AKIA / ghp_ / sk- / nvapi- / private-key / user:pass@ URL / box address / proxy port / team or
   ECR account id in /app, /etc, /root (as root), /home, /tmp (one pattern hit, a float in pai_spec.json). /root holds the base image's
   .bashrc / .profile / empty .python_history / a .wget-hsts with www.python.org, all in the unchanged base layers.
6. Smoke evidence: pass. Native reference on Tokyo is byte-equal to the box's results-summary (`029b236f...`). Recomputed from the two
   results-summary files: image 0.5403 (13 zeros), native 0.6090 (11), paired -0.0687, bootstrap 95 % [-0.1724, +0.0005]; zero flips only
   2bbf1efc and 3232fdb7 (image corridor 0, native 1.0), none the other way; other differing scenes 404541da 0.765 / 0.757 and 478d6ec4
   (equal to 6 places); 4e4a2a51 a zero on both (collision vs corridor). Driver run without `-e JEV_*` (only the replica variables), wizard
   `controller.gains.rel_front_steering_angle_weight=10`; all 29 rollout.asl carry `...-acc-gap0.7_2.7-trackgate`. Flips: decision 0 plan
   difference 0.3 mm / 3.1 mm; plans part > 0.1 m at decision 4 (> 1 m at 16) on 2bbf1efc and at 24 (> 1 m at 69) on 3232fdb7, and both are
   corridor zeros of rsgaccgap on both hosts: closed-loop amplification of cross-host numerics on knife-edge scenes, not an image defect.
   Own `pai_image_cmp.py` of smoke_rep vs smoke: 29 / 29 scores equal, served plans identical at every decision in 29 / 29 (the builder's
   bit-identical claim holds). Submit block: `~/bin/ap acct3 ...` sets `ALPASIM_CONFIG_DIR` / `DOCKER_CONFIG` to the acct3 profile (exists on
   Tokyo) and execs the competitor CLI or docker; `me`, `limits`, `ecr-login`, `status`, `submit --track pai|pai-warmup --controller-gains`
   all exist in the CLI; gains10.json on Tokyo is `{"rel_front_steering_angle_weight": 10}` (bound [0, 10]); image id in the check line is
   this image's; tag `ntv0-l18-s0-acc-gap-v1` is new (d356 used `rsg-l18-s0-acc-gap-v1` on the same vennai-org repo, so layers 1-12 are
   already there). Not run.

Concerns (none blocking): 1a5da90a's decision 0 plan differs by 1.28 m longitudinally (end point 59.07 vs 57.78 m, scored 1.0 on both; rsgaccgap
had 0.001 m on this scene), so NTV0 has at least one first-frame knife edge that RSG did not; the -0.069 on this chunk is cross-host drift, which
the organisers' H100s will add on top of the d366 single-host margin. The team slug `vennai-org` was not re-read (`ap acct3 me` contacts the
challenge); the user's first command prints it. The `until ... docker push` loop retries forever on a persistent error (e.g. an expired ECR
login); stop it by hand if it does not converge. As for rsgaccgap, only the version string and the serve start line show `JEV_GAP`; check the
version on the submission's first rollout.

**rsgaccgap (`sha256:328433cb...`), independent pre-submission audit, 2026-10-18: GO.** Checked on the image on Tokyo (inspect, `docker run
--network none` with entrypoint overrides, one own `test.sh` run on card 1), not only on the records above.

1. Identity and config: pass. Id `sha256:328433cbab61...2679`, 8 033 245 840 bytes. ENV = rsgacc's block plus `JEV_GAP=0.7,2.7`, with
   `SH30_GIT_HASH` 382164ae -> ea61e9df; `JEV_TAG=RSG-L18-s0 JEV_DRIVER=pai JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1`, `JEV_VHIST=` (empty = off),
   no `JEV_TRACK` (driver logs `JEV_TRACK gate`), no other `JEV_*` / `SH30_*` switch. CMD `python /app/serve.py`, no entrypoint, user
   10001:10001, port 6789, SIGTERM: all as rsgacc. The steering weight is not baked in.
2. Layers and code: pass. RootFS layers 1-12 identical to rsgacc; 13-18 (code, serve files, user, ckpt, ckpt-extra, verify) differ. File
   sha256 of the whole of /app (except models, unchanged layer): only the 9 known closure files differ from rsgacc; serve.py / probe.py /
   verify.py / pai_spec.json / checkpoints / sidecars / tags.json are equal. All 29 baked code files equal `git show ea61e9df`. Against the
   d330 stage-2 read's 94711ca9, serve_fix.py and lead_long.py are equal; 7 differ (ap2_driver, pai_core, pai_driver, sh30_core, sh30_driver,
   pp_train, rmem). Read: JEV_DESIRE (unset: module not imported, `desire=None`, the old call taken), the rmem anchored layout (`SERVE_ANCHOR`
   from the sidecar's `head.anchor`; RSG-L18-s0's route_mem.json has neither `order` nor `anchor` -> False -> old `merge` / `memory` path),
   P0 allowed in sh30_core's assert, an ap2-only t_first fix. None touches the served path of this image. The d330 read's version string
   is the image's, `jev-pai-RSG-L18-s0-backwarp-vc1-lead-acc-gap0.7_2.7-trackgate`.
3. Checkpoints: pass. In image RSG `fbe7ac9f` / route_mem `672ef154`, RBSG2 `5572c7e8` / `b667704a`, equal to both Tokyo `/data/runs` and the
   AutoDL box originals.
4. Start and serving: pass. `verify.py` under `--read-only --network none`, no GPU: both ROUTE-MEM branches load on the CPU (dim 200, order
   False, reflex False); only the build-time tags.json write fails, as it should on a read-only root. Own `test.sh` (organisers' hardening
   flags, read-only, `--network none`, card 1): listening after 17.9 s, 800 `drive` calls p50 46.7 ms, 0 inference errors, 0 tracebacks,
   0 changed files, /tmp 2.7 MiB, GPU 1.47 GiB, version as above, git ea61e9df (`/data/runs/alpasim/pai_image/audit_rsgaccgap/test`).
5. Secrets: pass. `docker history --no-trunc`: no token, proxy, credential URL or build-host address (only the Python base's public
   `GPG_KEY` id). Filesystem: no .env, .netrc, .git-credentials, ~/.alpasim, ssh key, .git; the only *.pem are public CA bundles; no
   hf_ / AKIA / ghp_ / sk- / nvapi- / private-key / user:pass@ URL / proxy port / box address in /app, /etc, /root, /home, /tmp.
6. Smoke evidence: pass. Recomputed from smoke.json: paired diff +0.0017, bootstrap 95 % [-0.0014, +0.0066], zero set identical (12, with the
   same reasons per scene in cmp_box.md, incl. both at-fault collisions 1a69f757 / 3ce13eed), 4 differing scenes all non-zero on both sides;
   0 inference / input errors over 5 752 calls; no driver env injected (`drv_env ""`); the smoke rollouts carry the `-gap0.7_2.7` version, so
   GAP was live.

Concerns (none blocking): serve.py's start line and pai_driver's `serving:` line do not print `JEV_GAP`; the version string is the only log
evidence that GAP is on. Check that string on the submission's first rollout.

## Deviations of the local checks from the official run

- One RTX 3090 (sm_86) for renderer, physics and driver; the official run uses H100s (driver on GPUs 4-7, renderer on 0-3) and the `ec2`
  preset (16 replicas x 2 rollouts, renderer 8 replicas x 1 rollout per card); the smoke ran the `dev` preset's 1-GPU topology at 4 concurrent
  rollouts with the harmonizer removed as `ec2` does. `ec2` also sets `render_bundling: BATCH_RENDER_RGB` (images rendered as one batch per
  step); the dev preset does not, so the per-camera image arrival order differs; the driver aligns by timestamp and decodes only the front
  wide camera.
- The trusted image is Tokyo's 2026-10-09 build of `alpasim-base:0.89.0` (floating package versions).
- The scenes are curated-validation scenes (the public local set); the official PAI suite is private.
