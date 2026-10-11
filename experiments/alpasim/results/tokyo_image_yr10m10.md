# Submission image for YR10m10-F-s0: build, containerised smoke, pre-submission audit (2026-10-12)

Read this before you push or submit `YR10m10-F-s0` to the nuPlan track. It is the image of the first official submission (APY10m10-AB-s0,
[docs/alpasim.md](../../../docs/alpasim.md#submission-image-measured-2026-10-09-decision-215), [aud1_driver_assumptions.md](aud1_driver_assumptions.md)
section 6) with only the checkpoint changed. The checkpoint was chosen in [aud1_drift_by_decision.md](aud1_drift_by_decision.md). Built and
tested on the Tokyo box with `docker/new_tag.sh` from a frozen clone at `073dea33`. Nothing was pushed, nobody logged in to the registry,
and nothing was submitted. Evidence: `results/tokyo_image/YR10m10-F-s0/`. Run dirs on the Tokyo box:
`/data/runs/alpasim/tokyo_image/YR10m10-F-s0-20261012-003448/` (test, smoke, smoke_c2) and `/data/runs/alpasim/tokyo_image/yr10m10_chain/`
(chain, live audit).

## Image

| | |
|---|---|
| Local name | `jev-alpasim:yr10m10-f-s0-073dea33` |
| Image id | `sha256:3a00b88c60776845010431aa15492bf8ee4e397dec5544a7931e0554869e0c17` |
| Registry digest | `sha256:3f55a7dcdec32d8a624b01d12390256278b8616869654df1cbb78de7c5bfc722` (manifest size 4094; pushed by the user 2026-10-12 ~01:10 JST; 2 layers uploaded, checkpoint and build defaults, 16 layers already in the team repository from `apy10m10-ab-s0-v1`) |
| Size | 8 102 839 375 bytes (7.55 GiB; limit 40 GiB) |
| Built | 2026-10-12 00:17 JST, 3 min. All layers below the thin checkpoint and defaults layers are cached and shared with the pushed `apy10m10-ab-s0-v1` |
| Repo state | `073dea33` (main HEAD), clean in the image's paths (no `-dirty` suffix) |

## Checkpoint

| | |
|---|---|
| Tag | `YR10m10-F-s0`: P2H10 lambda-10 base + yaw-rate rows, seed 0 ([ot3_lambda10.md](ot3_lambda10.md), [cf1_confirm.md](cf1_confirm.md)) |
| Training | `ot3_rows.py train --data navtrain_full.s{0..11}of12 --split navsim/op-parity-full --batch 128 --warmup 300 --std navsim --ot-mass 0.10 --hinge-lam 10 --hinge-margin 0.3 --steps 10000 --seed 0`, finished 2026-10-09 14:32 on the box (git `8a937a38`, dirty). Run dir `$DATA_DIR/runs/op_parity/train-YR10m10-F-s0/20261009-140623`, dev ADE 0.539 |
| File | `$DATA_DIR/runs/op_parity/runs/YR10m10-F-s0/ckpt-final.pt`, 74 965 804 bytes |
| sha256 | `a20d5ba97bf37d150e3d51f06e23304b5da51d83a4f9488456b6770addce404b`. Equal on the GPU box, on the Tokyo box (`tokyo_pull.sh ckpt`) and inside the image |
| Identity | The seed-0 entry of the `ot3/a` manifest, the native runs that CF1 and `aud1_drift_by_decision.md` read. Its native driver log says `sh30: YR10m10-F-s0 (backwarp, motion 1.00)` |

## Configuration served

The same configuration as the first official submission except for the checkpoint and the driver family that follows from it (`sh30`
here, `ap2` there).

| Setting | Value in the image | Meaning |
|---|---|---|
| `JEV_TAG` | `YR10m10-F-s0` | served checkpoint |
| `JEV_DRIVER` | `auto` -> `sh30` (`tags.json`) | driver family |
| `SH30_COMPILE` | `0` | interpreted passes (driver log: `compiled False`) |
| Cold start | `backwarp` (the checkpoint's own rule) | `version_id` `jev-YR10m10-F-s0-backwarp` |
| `JEV_VCONT` / `JEV_LEAD` | `0` / `0` | off |
| `JEV_BASE`, `JEV_STOP`, `JEV_REPLAN` | unset | off (log: `serving: JEV_VCONT 0, JEV_LEAD 0, JEV_BASE 0`) |
| `SH30_LHT` | unset | right-hand traffic |
| `SH30_SYNTH` / JPEG decode | unset | GPU frame synthesis, libjpeg (log: `synth gpu`, `jpeg libjpeg`) |
| `ALPASIM_DRIVER_GRPC_WORKERS` | `8` | |
| `SH30_GIT_HASH` | `073dea33` | reported as `git_hash` by `get_version` |

Every environment variable the image sets (`docker image inspect`): `PATH`, `LANG=C.UTF-8`, `GPG_KEY`, `PYTHON_VERSION=3.11.17`,
`PYTHON_SHA256` (python base image); `PYTHONDONTWRITEBYTECODE=1`, `PYTHONUNBUFFERED=1`, `HOME=/app`, `TMPDIR=/tmp`,
`XDG_CACHE_HOME=/tmp/.cache`, `TORCH_HOME=/tmp/torch`, `TORCH_EXTENSIONS_DIR=/tmp/torch_extensions`, `HF_HOME=/tmp/huggingface`,
`HF_HUB_OFFLINE=1`, `MPLCONFIGDIR=/tmp/matplotlib`, `CUDA_CACHE_PATH=/tmp/nv`, `NUMBA_CACHE_DIR=/tmp/numba`, `DATA_DIR=/app/data`,
`ALPASIM_SRC=/app/alpasim`, `ALPASIM_DRIVER_LOG_DIR=/tmp/alpasim-driver`, `ALPASIM_DRIVER_HOST=0.0.0.0`, `ALPASIM_DRIVER_PORT=6789`,
`ALPASIM_DRIVER_GRPC_WORKERS=8`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`, `JEV_TAG=YR10m10-F-s0`,
`JEV_DRIVER=auto`, `SH30_GIT_HASH=073dea33`, `JEV_VCONT=0`, `JEV_LEAD=0`, `SH30_COMPILE=0`, `TRITON_CACHE_DIR=/tmp/triton`,
`TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor`. No `PAI_*`, `AP2_*`, `SH30_LHT`, `SH30_SYNTH`, `JEV_BASE`, `JEV_STOP` or `JEV_REPLAN`.
The runtime adds only `ALPASIM_CONTESTANT_REPLICA_INDEX` / `ALPASIM_CONTESTANT_REPLICAS`.

## Constraint test and smoke (one RTX 3090 per role)

`test.sh`: the organisers' `run_local_container.sh` flags plus `--network none`, with 2 concurrent synthetic rollouts x 8. `smoke.sh`: the
48 public scenes of `scenes_navtest_full_part001_48.txt` in the official containerised stack (AlpaSim 0bb4c4b, `alpasim-base:0.89.0`). It was
run twice: once at 8 concurrent rollouts (the wizard overrides of the native reference runs, as in decision 215) and once with the driver
endpoint at 2 concurrent rollouts (the official per-replica count, `WIZ_EXTRA=runtime.endpoints.driver.n_concurrent_rollouts=2`).

| | containerised, 8 concurrent | containerised, driver 2 concurrent | native (GPU box) | limit / expected |
|---|--:|--:|--:|---|
| Mean scene score, 48 scenes | **0.9838** | **0.9838** | 0.9633 | within ~0.01 of native |
| Zeros (collision / offroad / corridor) | 0 (0 / 0 / 0) | 0 (0 / 0 / 0) | 1 (0 / 1 / 0) | |
| Scenes at 1 | 35 | 35 | 34 | |
| Identical to native | 34 of 48 | 34 of 48 | | most |
| Differ by <= 0.0054 | 13 | 13 | | |
| Flipped | 1 (`...01690_02183-30f09e001cf55013`: native offroad 0 -> 1.0) | same | | |
| Mean of the other 47 scenes | 0.9834 | 0.9834 | 0.9838 (difference -0.0003) | |
| 8-concurrent vs 2-concurrent run | | 0 of 48 scenes differ | | |
| `drive` calls / inference errors / input errors | 480 / 0 / 0 | 480 / 0 / 0 | | 0 errors |
| Tracebacks in the driver log | 0 | 0 | | |
| `drive`, server side, p50 / p90 / p99 / max | 35.8 / 44.8 / 84.8 / 148.1 ms | 33.6 / 38.0 / 57.3 / 102.0 ms | | p50 < 100 ms |
| `drive`, `test.sh` probe, 2 concurrent, client side, p50 / p90 / p99 / max | 50.0 / 52.6 / 69.5 / 77.8 ms | | | target 100 ms |
| Cold start, `docker run` -> listening | 20.4 s (`test.sh`); 18.2 / 18.0 s (live audit, 2 runs) | | | < 30 s; runtime waits 240 s |
| GPU memory of the driver | 1.56 GiB (`test.sh` probe), 1.71 GiB card peak in the smoke | 1.71 GiB | | 16 GiB |
| `/tmp` high-water: probe / smoke | 0.21 / 0.80 MiB | 0.80 MiB | | 2 048 MiB |
| `/run` high-water | 0 | | | 64 MiB |
| Wall time of the smoke | 213 s | 267 s | | |

Reading: the mean is 0.02 above native, which misses the "within ~0.01" expectation. The whole gap is one scene. On the other 47 scenes the
two means differ by 0.0003, and no scene differs by more than 0.0054. The flipped scene is a known knife edge. On OT3's 700-scene native
runs it ends offroad for about half the checkpoints and seeds, and the outcome does not follow the recipe: offroad for P2H10-F-s1, P2-F s0 / s1,
OT10a05-F-s1, YR10m10-F-s0, AP2H10-AB-s1, OT10a15-F s0 / s1 and APY10m10-AB-s0; 1.0 for P2H10-F-s0, YR10m10-F-s1, OT10a05-F-s0 and the
rest. That is the cm-level GPU / CPU numerics that the loop amplifies (decision 215, [tokyo_image_smoke.md](tokyo_image_smoke.md)). The
other side is a numerics change too: the native reference is not a 48-scene-list run but 48 rollouts filtered from the three 700-scene
chunk runs, and the simulator repeats itself only for a fixed scene list (decision 211).
Driver concurrency 2 against 8 changes no score. Latency and VRAM equal those of the image already pushed
(`apy10m10-ab-s0-2d7d7fdf`: probe p50 48.9 ms, 1.56 GiB, cold start 19.0 s). The driver is faster than the 86 ms of decision 215 because of
the GPU frame synthesis of LAT1.

## Pre-submission audit (on the exact image `3a00b88c...`)

Live checks: `live_audit.sh` runs the image with the `test.sh` flags (`--read-only --network none --cap-drop ALL --security-opt
no-new-privileges:true`, tmpfs `/tmp` and `/run`, uid from the image) and inspects the running container. Output: `live_audit.txt`.

| # | Check | Result | |
|---|---|---|---|
| 1 | Entry point: `CMD ["python","/app/serve.py"]`, no `ENTRYPOINT`, no arguments needed, `EXPOSE 6789`, `STOPSIGNAL SIGTERM`, no `HEALTHCHECK`, no volumes, `WORKDIR /app` | as the shipped samples | pass |
| 2 | Checkpoint inside the image `/app/data/runs/op_parity/runs/YR10m10-F-s0/ckpt-final.pt` = `a20d5ba9...` (the box's) | equal | pass |
| 3 | The 20 driver files of `closure.txt` in the image, hashed, against the repo at `073dea33`; `serve.py`, `probe.py`, `verify.py` against `docker/`; `cinque.ort.onnx` `3f12a0cd...` | 24 of 24 equal | pass |
| 4 | Serving switches: `JEV_VCONT=0`, `JEV_LEAD=0`; `JEV_BASE`, `JEV_STOP`, `JEV_REPLAN` unset; driver log `serving: JEV_VCONT 0, JEV_LEAD 0, JEV_BASE 0` | off | pass |
| 5 | `SH30_COMPILE=0`; driver log `compiled False` | interpreted | pass |
| 6 | Right-hand traffic: `SH30_LHT` unset | RHT | pass |
| 7 | Served tag and cold-start rule: driver log `YR10m10-F-s0 (backwarp, motion 1.00, synth gpu, compiled False, jpeg libjpeg)`; `serve: driver sh30, tag YR10m10-F-s0` | as intended | pass |
| 8 | uid / gid of the serving process 10001 / 10001, `CapEff 0000000000000000`, `NoNewPrivs 1` | | pass |
| 9 | Read-only root: writes to `/`, `/app`, the checkpoint directory refused (`Read-only file system`); `/run` refused (root-owned tmpfs, as in the organisers' flags; the driver does not use it); `/tmp` writable; `docker diff` lists only the NVIDIA runtime's injections and their parent directories; `test.sh`: 0 write errors, 0 changed files | | pass |
| 10 | No network: interfaces `['lo']` only; connecting to `1.1.1.1:443` gives `Network is unreachable` and `pypi.org` fails name resolution; in the smoke (Docker network `--internal`) `smoke.sh` checks that `https://pypi.org` cannot be reached | | pass |
| 11 | gRPC health: `get_version` answers `version_id: "jev-YR10m10-F-s0-backwarp"`, `git_hash: "073dea33"`, API 0.54.0 | | pass |
| 12 | gRPC drive: `probe.py` in the live container, 2 sessions x 2 rollouts x 10 decisions, every response checked: 40 `drive` calls, 0 inference / input errors, p50 50.5 ms; `test.sh` probe 160 calls, 0 errors; smokes 960 calls, 0 errors | | pass |
| 13 | Clean stop: SIGTERM -> exit 0 in 0.9 s | | pass |
| 14 | Limits: 7.55 GiB image, 1.71 GiB VRAM, `/tmp` 0.8 MiB, `/run` 0, cold start 18-20 s | | pass |

Verdict: **pass, submission-ready.** The image serves `YR10m10-F-s0` with the first submission's configuration, holds every published
container limit, and reproduces the native scores on 47 of 48 scenes to within 0.0054. Known differences from the official run, the same as
for the first submission: the organisers' GPU and CPU are not ours (cm-level numerics, knife-edge scenes may flip either way); the native
1491-scene reads used compiled passes and the image runs interpreted ones; the smoke covers the 48 Las Vegas scenes of one log set only.

## What the user runs (not run by an agent)

On the Tokyo box, with the CLI already configured for the first submission (`docs/alpasim.md`, "What the user runs"). The tag is public on the
leaderboard. The push should upload only the thin checkpoint and defaults layers, because the rest is already in the team repository from
`apy10m10-ab-s0-v1` (expected, not verified).

```bash
C="python3 /data/third_party/alpasim-cli/e2e_challenge/competitor_cli/alpasim_challenge.py"
$C me; $C limits                      # token valid 12 h; re-run auth-url / configure-token if expired; check the remaining official quota
$C ecr-login
IMG=jev-alpasim:yr10m10-f-s0-073dea33        # image id sha256:3a00b88c60776845010431aa15492bf8ee4e397dec5544a7931e0554869e0c17
URI=696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/lao-siji:yr10m10-f-s0-v1
docker image inspect "$IMG" --format '{{.Id}}'   # must print sha256:3a00b88c...
docker tag "$IMG" "$URI"
until docker push "$URI"; do sleep 10; done      # note the digest it prints in this file
$C submit --track nuplan "$URI"                  # official: uses one of the 3 submissions of the UTC month (shared with PAI)
$C status <submission_id>; $C submissions --track nuplan
```

## The paused ONPOL-PAI read (Tokyo cards were needed for this smoke)

At the user's request the ONPOL-PAI 281-scene read (`jev:onp-final`, `onp_final_chain.sh` -> `pai_queue.sh`, runs SELF, IMIT, FT, 843 scenes)
was stopped on 2026-10-12 at 00:31 JST, 31 min after it had started. Killed: process group 1799002, and the compose projects
`pai-p2h10s-self-on-f-s0-c000` / `-c001` with their `-drv` containers. Progress is kept per chunk (a `DONE` in
`runs/<tag>_<chunk>/`), and no chunk had finished: 0 scenes kept. Everything has to be rerun: the two in-flight SELF chunks c000 and c001
(72 scenes, about 31 min of card time lost) and all 843 scenes of the read. Their partial directories were moved to
`/data/runs/onpol_pai/read281/paused-20261012-003154/` and the claims were reset. The chain's earlier stages (trainings, G-load, held read)
were finished before the read began. The chain itself wrote no DONE / ERROR. Note: `/data/runs/onpol_pai/read281/PAUSED.md`.

Restart only the queue, not `onp_final_chain.sh` (that would retrain). A rerun skips finished chunks:

```bash
cd /data/runs/onpol_pai/code/pass1 && S=/data/runs/alpasim/pai_full && tmux new-window -d -t jev -n onp-read281 \
  "CARDS='0 1' CHUNK=36 CONC=4 SERVE='-e JEV_VCONT=1.0 -e JEV_LEAD=1' bash experiments/alpasim/scripts/pai_queue.sh /data/runs/onpol_pai/read281 \
   'self|P2H10S-SELF-ON-F-s0|$S/s_*.tsv' 'imit|P2H10S-IMIT-ON-F-s0|$S/s_*.tsv' 'ft|P2H10S-FT-F-s0|$S/s_*.tsv'"
```

About 6.3 h at 133 scenes / h on both cards. `/data/runs/onpol_pai/final/DONE` is not written by the queue. The queue writes its own
`read281/DONE`.

Last verified: 2026-10-12

## Official submission #2 (user, 2026-10-12)

`submit --track nuplan 696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/lao-siji:yr10m10-f-s0-v1` -> submission id
`cdf145f9-c8fb-474d-ba8a-22499f1b4a59`, status `QUEUED` (queue position 1 of 1) at about 01:10 JST. One official slot remains for October,
held for the final week (board closes 2026-10-31). **Result (SUCCEEDED 16:22 UTC, evaluation 26 min):** average scene score 0.80, PCS 1007 [961, 1053], rank 47-62 of 106, at-fault distance 0.529 km,
GT distance 1.733 m, 1000 / 1000 rollouts, wall time 1166 s of 3000 s. Submission #1 (APY10m10-AB-s0): 0.640 / 788 / 0.132 km / 2.29 m. On the public
1491 scenes the two checkpoints differ by -0.010 score, -0.19 m GT distance, at-fault not separable (aud1_drift_by_decision.md). Decision 260.
