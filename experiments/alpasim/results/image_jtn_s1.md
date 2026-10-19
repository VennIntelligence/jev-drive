# nuPlan submission images: JTN-L18-s1, native and with serve-side hold `JEV_HOLD=stop3` (2026-10-19)

Status: built and tested; **independent audit pending**. Nothing was pushed or submitted. Two images with the same build code and the same
checkpoint (JTN-L18-s1), differing only in `JEV_HOLD`. The procedure is that of [image_jtn_stop3.md](image_jtn_stop3.md) (JTN-L18-s0 + stop3) and
[image_rsg_stop3.md](image_rsg_stop3.md) (audited GO): same frozen clean clone at `ba3ef44d` on Tokyo
(`/data/runs/alpasim/tokyo_image/code-rsg-stop3/`, no `-dirty`), only the checkpoint and `HOLD` swapped. No code change: `build.sh` already maps an empty
`HOLD` to "no hold" (no `-hold...` version suffix, no `HOLD` build arg, `serve.py` runs `sh30_driver.py`), and `tags.json` is generated at build time from
the checkpoint dir. No pool job was paused or cancelled.

## Images

| | native | stop3 |
|---|---|---|
| Local name (Tokyo Docker store) | `jev-alpasim:jtn-l18-s1-ba3ef44d` | `jev-alpasim:jtn-l18-s1-ba3ef44d-holdstop3` |
| Image id | `sha256:1b68246103ec1fd106482eeec04c3e9b9fddb11a9f896148bde7b1a8f03dea22` | `sha256:b547cca3bbec62899785c618eea9b9d0241de88a032a2b2aa02f523b5f8d2440` |
| Size | 8 105 640 419 bytes (7.55 GiB; limit 40 GiB) | 8 105 640 419 bytes |
| Build | `env DATA_DIR=/data EXTRA=JTN-L18-s1 TAG=JTN-L18-s1 DRIVER=sh30 HOLD= OUT=... build.sh` | same with `HOLD=stop3` |
| `JEV_HOLD` in ENV | empty | `stop3` |
| Serve line | `serve: driver sh30 (sh30_driver.py), tag JTN-L18-s1 ...; JEV_HOLD=` | `serve: driver sh30 (sh30_hold.py), tag JTN-L18-s1 ...; JEV_HOLD=stop3` |
| Version (gRPC `get_version`) | **`jev-JTN-L18-s1-backwarp`**, git `ba3ef44d` | **`jev-JTN-L18-s1-backwarp-holdstop3`**, git `ba3ef44d` |
| `static_hold.SUFFIX` in image | `''` | `'-holdstop3'` |

Both: ENV otherwise `JEV_TAG=JTN-L18-s1 JEV_DRIVER=sh30 SH30_GIT_HASH=ba3ef44d JEV_VCONT=0 JEV_LEAD=0 JEV_ACC=0 JEV_VHIST= SH30_COMPILE=0 JEV_GAP=`; CMD
`python /app/serve.py`, user 10001:10001. Build artifacts: [native](image_jtn_s1/native/build), [stop3](image_jtn_s1/stop3/build).

**Checkpoint** (`JTN-L18-s1/` under `runs/op_parity/runs/`): `ckpt-final.pt` sha256 `d9c134bfdc692548aa4c9d3b1f5b5a065a1eaa7fa0c3704a1e06e2ebb062b2dc`
(77 552 223 bytes), `route_mem.json` `c2dbdd307593d528069a3063357bd87783e4b19f393ae6e85d40b94649d1bd44`. The AutoDL box original, the Tokyo copy
(`/data/runs/op_parity/runs/JTN-L18-s1/`, streamed with tar) and the files inside both images are equal (sha256 recomputed in each place). The box dir also holds
`adapter.pt` (`9060480c...`, copied to Tokyo, in no image). `tags.json` in both images: `{"AP2-AB-s0": "ap2", "JTN-L18-s1": "sh30", "P2H10-F-s0": "sh30"}`.

## Sanity: the pair differs only in `JEV_HOLD`

- Docker rootfs layers: all 18 identical between the two images (the `HOLD` ENV is a metadata layer), so they differ only in image config (`JEV_HOLD`).
- File sha256 over `/app` (47 files each): **byte-identical** between native and stop3 ([stop3/limits/app_diff_vs_native.txt](image_jtn_s1/stop3/limits/app_diff_vs_native.txt) is empty).
  What turns the hold on is `serve.py` reading `JEV_HOLD`; the hold code (`sh30_hold.py`, `static_hold.py`) ships in both.
- Version string differs by exactly the `-holdstop3` suffix, as listed above.

**Against the JTN-L18-s0 stop3 image** (`74ad6b91`): rootfs layers 1-16 identical, 17-18 (checkpoint layer, `tags.json`) differ. `/app` file sha256
([native](image_jtn_s1/native/limits/app_diff_vs_jtn_s0_stop3.txt), [stop3](image_jtn_s1/stop3/limits/app_diff_vs_jtn_s0_stop3.txt), same 5 diff lines for both):
the two `JTN-L18-s0/{ckpt-final.pt,route_mem.json}` files replaced by the `JTN-L18-s1` ones, and `tags.json` (JTN-L18-s0 -> JTN-L18-s1 entry). Code,
`cinque.ort.onnx` and the stable checkpoints are byte-equal. Environment differs in `JEV_TAG` (and `JEV_HOLD` for native).

## Constraint test (`test.sh`, Tokyo RTX 3090 card 1, organisers' flags + `--network none`, by image id)

Both pass ([native](image_jtn_s1/native/test/result.json), [stop3](image_jtn_s1/stop3/test/result.json)):

| check | native | stop3 |
|---|--:|--:|
| cold start | 17.4 s | 19.4 s |
| driver GPU | 1 603 MiB | 1 603 MiB |
| `/tmp` / `/run` | 0.37 / 0 MiB | 0.37 / 0 MiB |
| read-only root | 0 errors, 0 files changed | 0 errors, 0 files changed |
| 160 `drive` calls p50 / p90 / p99 / max | 50.1 / 52.3 / 61.1 / 69.5 ms | 50.4 / 53.3 / 63.5 / 70.3 ms |
| inference errors / tracebacks | 0 / 0 | 0 / 0 |
| SIGTERM | exit 0 in 0.6 s | exit 0 in 0.6 s |
| probe version | `jev-JTN-L18-s1-backwarp` | `jev-JTN-L18-s1-backwarp-holdstop3` |

## 48-scene smoke (official containerised stack, `smoke.sh`, render card 0 / driver card 1, by image id)

References: native box reads of the same 48 public scenes with the overrides of the JTN-s0 stop3 reference (pool job `1018-100141-234a`) and
`SH30_TAG=JTN-L18-s1`, submitted through the GPU pool: native `1019-105949-ed5a` (`run.sh ... sh30`, `JEV_HOLD=` empty), stop3 `1019-105949-f7b9`
(`run.sh ... sh30hold`, `JEV_HOLD=stop3`). Box commit `926ae261`; its alpasim `lib/` differs from `ba3ef44d` only by the `SH30_WARP` diagnostic switch in
`sh30_core.py` (default off, same calls as before). Copied to Tokyo `/data/runs/alpasim/native_ref/JTN-L18-s1-{native,holdstop3}/`. Box dirs:
`~/data/runs/alpasim/hold_image/native48_jtn_s1_{native,stop3}/`.

| | image (container) | native (box) |
|---|--:|--:|
| **native**: mean, 48 scenes | **0.9828** | 0.9828 |
| zeros / at 1 | 0 / 37 | 0 / 37 |
| **stop3**: mean, 48 scenes | **0.9417** | 0.9418 |
| zeros / at 1 | 0 / 34 | 0 / 34 |

- Native pair ([compare.md](image_jtn_s1/native/smoke/compare.md)): 11 of 48 scenes differ, max 0.0261 (scene `...00083_00485-83fd1c19`, image 0.8940 vs box 0.9201),
  paired mean -0.0000, 47 within 0.01, all 48 within 0.05, zero / non-zero agree 48 / 48.
- Stop3 pair ([compare.md](image_jtn_s1/stop3/smoke/compare.md)): 14 of 48 differ, max 0.0261 (same scene), paired mean -0.0001, 47 within 0.01, all 48 within 0.05,
  zero / non-zero agree 48 / 48. The scene gaps are the cross-host closed-loop numerics seen in the earlier records, not an offset.
- Driver, both: 480 `drive` calls, 0 inference errors, 0 tracebacks, server p50 / p90 / max 36.3 / 44.7 / 130.2 ms (native), 36.5 / 44.4 / 126.6 ms (stop3).
  The fail-open line (`JEV_HOLD: the hold raised`) is absent in both. The native image writes no `hold.jsonl`.
- Hold live in the stop3 image: `hold.jsonl` 480 records; 5 sessions are held (< 0.3 m/s), 43 free. No stop-rule latch evidence beyond that was examined here.
- On this public log the hold costs 0.041 of mean (0.9828 native -> 0.9417 stop3): 4 scenes drop by more than 0.01 and nothing rises
  (`01100_01664-218adf8c` 1.00 -> 0.65, `-4ccc9e33` 1.00 -> 0.28, `01690_02183-3debd3d8` 0.95 -> 0.34, `-b6968f15` 1.00 -> 0.72; image-vs-image, the box reads agree). Descriptive, one seed, one log
  (JTN-L18-s0 + stop3 scored 0.9383 on the same scenes). The public log is not the private one; the hold's rationale is the private-like arena, see [decision 353](../../../research/decisions/353.md).

Not repeated: the latch parity and fail-open rerun of the RSG stop3 record; the served hold code is byte-identical, so they are inherited.

## Limits and secrets

- Size 7.55 GiB; GPU driver 1.6 GiB; `/tmp` 0.4 MiB; no network needed (`--network none`; smoke on an `--internal` network).
- `docker history --no-trunc` (stop3 image): no token, proxy, credential URL or host address; the only hit of `@` is the Debian base's `@1791158400` timestamp.
- Filesystem scan as root (stop3 image): none of `.env .netrc .git-credentials id_* *.key .git .ssh .docker challenge.json credentials* .pypirc`; the content grep hits only
  `/app/pai_spec.json` (`7890` inside a float, a known false positive, byte-equal to the earlier images). The two images share all files, so the native image is covered by the same result.
  Container env: no token or proxy variable.

## For the auditor

- Same code as the audited RSG stop3 image; inputs that changed are the checkpoint + sidecar, `tags.json`, `JEV_TAG`, version string, and (native) `JEV_HOLD` empty.
- Look at: (1) the checkpoint sha above against the box dir; (2) the two `app_diff_vs_jtn_s0_stop3.txt` files and the empty `app_diff_vs_native.txt`; (3) that the native image
  really serves `sh30_driver.py` with no hold (serve line, `JEV_HOLD=`, no `hold.jsonl`); (4) the native references were made at box commit `926ae261`, not `ba3ef44d`
  (serve code equal, `SH30_WARP` default off).

Files left in place on Tokyo: `/data/runs/alpasim/tokyo_image/JTN-L18-s1-{native,holdstop3}/` (build.log, test, smoke, IMAGE), `...-build/`, `jtn_s1_chain1.sh` (build + test),
`jtn_s1_chain2.sh` (smoke), `/data/runs/alpasim/native_ref/JTN-L18-s1-*/`, `/data/runs/op_parity/runs/JTN-L18-s1/`.

## What the user runs (not run by an agent; after the audit)

On the Tokyo box, account profile `ujs-iv`. `~/bin/ap <profile> ...` exports that profile's `ALPASIM_CONFIG_DIR` and `DOCKER_CONFIG`, so the ECR login is the profile's own
(all teams share one registry host; check that the team printed by `ecr-login` is the one you mean, the Tokyo store also holds images pushed to other teams). Quota: 3 official submissions per UTC month
shared with PAI; only the code layer and the layers above it upload (about 230 MB for the first image, the second one only its metadata layer).

```bash
P=ujs-iv
~/bin/ap $P me; ~/bin/ap $P limits                    # token valid 12 h; if expired: ap $P auth-url; ap $P configure-token
~/bin/ap $P ecr-login                                 # prints the team repository URI (used below as $REPO)
REPO=696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/ujs-iv   # replace with the URI ecr-login printed

# image 1: native (no hold)
IMG=jev-alpasim:jtn-l18-s1-ba3ef44d
docker image inspect "$IMG" --format '{{.Id}}'        # must print sha256:1b68246103ec1fd106482eeec04c3e9b9fddb11a9f896148bde7b1a8f03dea22
URI=$REPO:jtn-l18-s1-native
docker tag "$IMG" "$URI"
until ~/bin/ap $P docker push "$URI"; do sleep 10; done     # note the digest it prints
~/bin/ap $P submit --track nuplan-warmup "$URI"       # optional warm-up: 32 private scenes, status only
~/bin/ap $P submit --track nuplan "$URI"              # official
~/bin/ap $P status <submission_id>; ~/bin/ap $P submissions --track nuplan

# image 2: hold stop3
IMG=jev-alpasim:jtn-l18-s1-ba3ef44d-holdstop3
docker image inspect "$IMG" --format '{{.Id}}'        # must print sha256:b547cca3bbec62899785c618eea9b9d0241de88a032a2b2aa02f523b5f8d2440
URI=$REPO:jtn-l18-s1-holdstop3
docker tag "$IMG" "$URI"
until ~/bin/ap $P docker push "$URI"; do sleep 10; done
~/bin/ap $P submit --track nuplan-warmup "$URI"
~/bin/ap $P submit --track nuplan "$URI"
```

The `until ... push` loop retries forever on a persistent error (for example an expired ECR login); stop it by hand if it does not converge. Remote tags are public on the leaderboard
(`latest` is refused). Counting the quota, the two images are two of the three official submissions of the month.

Last verified: 2026-10-19
