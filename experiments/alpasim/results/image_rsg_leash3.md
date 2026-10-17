# nuPlan submission image: RSG-L18-s0 + serve-side static hold `JEV_HOLD=leash3` (2026-10-18)

Status: built and tested, **independent audit pending, nothing pushed or submitted**. The candidate of
[decision 350](../../../research/decisions/350.md) (arena held windows: RSG-L18-s0 0.786 -> 0.934 with leash3). Built on the Tokyo box with
`docker/build.sh` from a frozen clone at `e7db399e` (`/data/runs/alpasim/tokyo_image/code-rsg-leash3/`, clean, no `-dirty`). No job was paused
or cancelled on either box.

## Image

| | |
|---|---|
| Local name (Tokyo Docker store) | `jev-alpasim:rsg-l18-s0-e7db399e-holdleash3` |
| Image id | `sha256:394b113747aeb834830980467d0b784cef5decf1868e98234ac59e0c8e3d03a7` (no registry digest yet: not pushed) |
| Size | 8 105 635 216 bytes (7.55 GiB; limit 40 GiB) |
| Build | `env DATA_DIR=/data EXTRA=RSG-L18-s0 TAG=RSG-L18-s0 DRIVER=sh30 HOLD=leash3 OUT=... experiments/alpasim/docker/build.sh` (the 10-17 RSG image's arguments + `HOLD=leash3`) |
| ENV (switches) | `JEV_TAG=RSG-L18-s0 JEV_DRIVER=sh30 SH30_GIT_HASH=e7db399e JEV_VCONT=0 JEV_LEAD=0 JEV_ACC=0 JEV_VHIST= SH30_COMPILE=0 JEV_GAP= JEV_HOLD=leash3`; everything else as the 10-17 RSG image ([tokyo_image_rsg_v2.md](tokyo_image_rsg_v2.md)); CMD `python /app/serve.py`, user 10001 |
| Served path | `serve.py` prints `serve: driver sh30 (sh30_hold.py), tag RSG-L18-s0, ...; switches ... JEV_HOLD=leash3` and execs `lib/sh30_hold.py`, which wraps the unchanged `sh30_driver.py` (the box's `sh30hold` driver of the arena-hold reads) |
| Checkpoint in image | `RSG-L18-s0/ckpt-final.pt` sha256 `fbe7ac9f...f05143`, `route_mem.json` `672ef154...4ade`: equal to the AutoDL box and Tokyo originals |
| Same build, HOLD unset | `jev-alpasim:rsg-l18-s0-e7db399e`, `sha256:6823c1a5...2d2e`: the same 18 filesystem layers as the hold image (only `ENV JEV_HOLD=` / `ARG HOLD=` differ); serve line `driver sh30 (sh30_driver.py)`; its `test.sh` passes |

**Against the 10-17 RSG image** (`rsg-l18-s0-d47f2692`, `f99095ea`): layers 1-12 identical; layers 13-18 (code, serve files, user, checkpoints,
verify) rebuilt. File sha256 over `/app` (except the unchanged models layer, [limits/app_sha_hold.txt](image_rsg_leash3/limits/app_sha_hold.txt)):
new `sh30_hold.py`, `static_hold.py`; changed `sh30_driver.py`, `sh30_core.py`, `pai_driver.py`, `ap2_driver.py`, `rmem.py` (d47f2692 -> e7db399e:
the flag-gated rmem anchored layout de3e94c7 / a853fe87 / 14fb2bcc, off for RSG's sidecar; the same set the PAI ACC-GAP audit read), `serve.py`,
`verify.py`. Checkpoints, sidecar, tags.json equal. All 31 closure files and the 4 entry files equal the repo at `e7db399e`. Every served file
(closure, `sh30_hold.py`, `static_hold.py`, `run.sh`, `drivers/sh30hold.sh`) is identical between `89e5e660` (the hold fix the arena reads ran) and `e7db399e`.

## Constraint test (`test.sh`, Tokyo RTX 3090 card 1, organisers' flags + `--network none`)

Pass ([test/result.json](image_rsg_leash3/test/result.json)): cold start 18.2 s; driver GPU 1.57 GiB; `/tmp` 0.38 MiB, `/run` 0; read-only root 0 write
errors, 0 changed files; 160 `drive` calls p50 / p90 / p99 / max 50.5 / 53.1 / 60.8 / 71.7 ms; 0 inference errors, 0 tracebacks; SIGTERM -> exit 0
in 0.6 s; version `jev-RSG-L18-s0-backwarp`, git `e7db399e`. The hold ran: `/tmp/alpasim-driver/hold.jsonl` 17 kB.

Other limits: CPU only (`--network none --read-only`, no GPU): `import static_hold, sh30_hold, sh30_driver` with `JEV_HOLD=leash3` parses mode
`leash` 3.0 m, latch 0.3 m/s at 0.5 s; `verify.py` loads the ROUTE-MEM branch (`dim 200 order False reflex False`) and fails only at the build-time
`tags.json` write, as on a read-only root it must. Secrets: `docker history --no-trunc` has no token, proxy, credential URL or host address;
no `.env`, `.netrc`, `.git-credentials`, ssh key, `.git`, `challenge.json` in the image; no `hf_` / `AKIA` / `ghp_` / `nvapi-` / private key / proxy
port / box address under `/app`, `/etc`, `/root`, `/home`, `/tmp`.

## 48-scene smoke (official containerised stack, `smoke.sh`, render card 0 / driver card 1)

Against a native box read of the same 48 scenes with the same hold (pool job `1018-010637-bd56`: `run.sh <dir> sh30hold` with `JEV_HOLD=leash3
SH30_TAG=RSG-L18-s0` and the c0b overrides, at `e7db399e`; copied to Tokyo `/data/runs/alpasim/native_ref/RSG-L18-s0-holdleash3/`):

| | image (container) | native leash3 (box) | native RSG, no hold (10-17) |
|---|--:|--:|--:|
| mean, 48 scenes | **0.9184** | 0.9180 | 0.9592 |
| zeros / at 1 | 1 / 29 | 1 / 29 | 1 / 33 |

18 of 48 scenes differ from native leash3, max 0.0104, paired mean +0.0005, 47 within 0.01, zero / non-zero agree 48 / 48; 480 `drive` calls,
0 inference / input errors, server p50 / p90 / max 37.1 / 49.7 / 99.4 ms; 0 tracebacks ([smoke/compare.md](image_rsg_leash3/smoke/compare.md)).
Latch parity on these 48 ([parity_public48.md](image_rsg_leash3/parity_public48.md)): latch state 48 / 48 equal (5 held), hold state and cut flag
480 / 480 equal.

**Public cost of the hold:** on this one public log the hold latches 5 of 48 sessions, all where the human launches later in the window, and costs
0.041 (0.959 -> 0.918; 4ccc9e33 0.28, 3debd3d8 0.34, 218adf8c 0.66, b6968f15 0.74, identical on both hosts). This is the launch cost decision 350
accepts for the static gain; the public board will read lower than the RSG image.

## Parity on 12 held arena windows (served plans and latch events)

The 12 windows of the arena-hold smoke list (6 static, 4 launch, 2 other; all in the leash3 reads' chunks c3-c6 of shard 1), run through the image
in the containerised stack (`smoke.sh` with `SCENES=parity12.txt` and `runtime.scene_provider.trajdata.cache_location` = a copy of the box's arena
cache shard `s1`; its `maps` symlink re-pointed to Tokyo's shipped maps, sha256-identical to the box's), against the box's leash3 read logs
(`reads/leash3/c{3,4,5,6}/run/driver-logs/{hold,drive}.jsonl`, results-summary). Script `scripts/hold_parity.py`; table
[parity_arena12.md](image_rsg_leash3/parity_arena12.md).

- Latch events: 12 / 12 equal (9 held: all 6 static + 3 of 4 launch; 3 free), at the same call (0.5 s) with the same measured speed to 1 mm/s.
- Hold state and cut flag per decision: 120 / 120 equal.
- Served plans: decision 0 max xy difference over the 8 poses median 0.7 cm, max 7.9 cm; all decisions median 0.8 cm, p90 8.0 cm, max 0.29 m
  (on a free 14 m/s window). Held windows: max distance from the latch point equal to 2 cm (e.g. 3.00 / 3.00 m, 2.97 / 2.97 m).
- Scores: 10 / 12 identical, 12 / 12 within 0.01 (0.2830 / 0.2829, 0.8574 / 0.8552); mean 0.8450 box / 0.8448 image; the same zero
  (86280cbf, static, held).

Plans are not bit-identical: the image runs interpreted passes on an RTX 3090 (`SH30_COMPILE=0`), the box compiled ones on an RTX 6000D, as for
every earlier image (cm-level, decision 215); two box reads of the same windows already differ up to 0.20 m (arena_hold.py gate).

## For the auditor: what changed (commit `e7db399e`, plus this record)

- `docker/Dockerfile`: `ARG HOLD=""` + `ENV JEV_HOLD=${HOLD}` after GAP's ENV (metadata layers only).
- `docker/build.sh`: `HOLD` env (validated `full | lead | leash<m>`, `off` = unset; refused for TRACK=pai, DRIVER ap2 / ens, ensembles), `--build-arg
  HOLD` only when set, tag suffix `-hold<HOLD>`.
- `docker/serve.py`: JEV_HOLD set and not `off` -> exec `sh30_hold.py` (refuses any family but sh30); else `<fam>_driver.py` as before. The start
  line now names the entry file and also prints `JEV_GAP` and `JEV_HOLD`.
- `docker/closure.txt`: + `sh30_hold.py`, `static_hold.py`. `docker/verify.py`: also imports `sh30_hold` (JEV_HOLD unset at build: patches nothing).
- `docker/new_tag.sh`: header comment only. `docs/alpasim.md`: one build line. New `scripts/hold_parity.py`.
- Look at: (1) `sh30_hold.install` logs `serving JEV_HOLD=...` before `sh30_driver.main()` configures logging, so the line never appears (also on
  the box reads); evidence that the hold is live is serve.py's start line and `hold.jsonl`. (2) The version string is the RSG image's
  (`jev-RSG-L18-s0-backwarp`): `sh30_driver.get_version` does not know the hold, so the leaderboard cannot tell the two images apart by version.
  (3) An uncommitted `stop<m>` mode of `static_hold.py` / `sh30_hold.py` is in the Mac working tree (another session); it is not in this image or
  in `e7db399e`, and `build.sh` rejects `HOLD=stop<m>` until its regex is extended. (4) `hold.jsonl` grows about 200 bytes per decision in `/tmp`
  (2 GiB tmpfs): no risk at 1 000 scenes.

Files left in place: on Tokyo the frozen clone, `/data/runs/alpasim/tokyo_image/RSG-L18-s0-holdleash3/` (test, test_unset, smoke, parity, limits;
`parity_fail1` = a first parity run that failed on the cache's maps symlink), `/data/runs/alpasim/tokyo_image/RSG-L18-s0-holdleash3-build/`, the
cache copy `/data/datasets/alpasim_nuplan/_arena_hold_parity/s1` (471 MB, inside the dataset root because only that root is mounted into the stack),
`/data/runs/alpasim/native_ref/RSG-L18-s0-holdleash3/`, and the unset image `jev-alpasim:rsg-l18-s0-e7db399e`. On the AutoDL box
`$DATA_DIR/runs/alpasim/hold_image/native48_leash3/`.

## What the user runs (not run by an agent)

On the Tokyo box, after the audit, as in [tokyo_image_rsg_v2.md](tokyo_image_rsg_v2.md) and [docs/alpasim.md](../../../docs/alpasim.md) "What the
user runs". Accounts are profiles of `~/bin/ap` (`ap ls` shows which session is alive; each profile has its own token and Docker login, all teams
share one ECR host). The nuPlan images so far went to `teams/lao-siji`; take the team repository that `ecr-login` prints for the chosen profile.
Quota: 3 official submissions per UTC month shared with PAI; warm-ups separate. Only the new layers upload (code, serve files, user, checkpoints,
verify: about 230 MB uncompressed); the lower 12 are those of the images already pushed.

```bash
P=lao-siji                                            # or ujs-iv / acct3: the account this probe is spent on
ap $P me; ap $P limits                                # token valid 12 h; if expired: ap $P auth-url; ap $P configure-token
ap $P ecr-login                                       # prints the team repository URI
IMG=jev-alpasim:rsg-l18-s0-e7db399e-holdleash3
docker image inspect "$IMG" --format '{{.Id}}'        # must print sha256:394b113747aeb834830980467d0b784cef5decf1868e98234ac59e0c8e3d03a7
URI=696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/lao-siji:rsg-l18-s0-holdleash3-v1   # team from ecr-login; public tag, not `latest`
docker tag "$IMG" "$URI"
until ap $P docker push "$URI"; do sleep 10; done     # note the digest it prints
ap $P submit --track nuplan-warmup "$URI"             # optional warm-up: 32 private scenes, status only
ap $P submit --track nuplan "$URI"                    # official
ap $P status <submission_id>; ap $P submissions --track nuplan
```

Last verified: 2026-10-18
