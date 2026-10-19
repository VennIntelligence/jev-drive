# nuPlan submission images: JTN-L18-s1, native and with serve-side hold `JEV_HOLD=stop3` (2026-10-19)

Status: built and tested; independent audit: **GO for both images** ([image_jtn_s1_audit.md](image_jtn_s1_audit.md)), commands below corrected by it. Nothing was pushed or submitted. Two images with the same build code and the same
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

Corrected by the audit ([image_jtn_s1_audit.md](image_jtn_s1_audit.md), item 5): the team repository is taken from what `ecr-login` prints instead of a
hard-coded guess, the local tag is made from the image **id** (so a re-tagged or rebuilt `jev-alpasim:...` name cannot be pushed by mistake), the push
retries a bounded number of times, and the remote digest is checked against the push before submitting.

On the Tokyo box, account profile `ujs-iv`. `~/bin/ap <profile> ...` exports that profile's `ALPASIM_CONFIG_DIR` and `DOCKER_CONFIG`, so the token and the
ECR login are the profile's own (`ap <profile> submit` also resolves the image with that login). All teams share one registry host. Evidence for the team
id: this account's PAI image was pushed on 2026-10-16 to `teams/ujs-iv` (`teams/ujs-iv:rsg-l18-s0-acc-v1`, digest `9b19cc9d...`, still in the Tokyo store);
`ecr-login` prints the authoritative value. Quota: 3 official submissions per UTC month shared with PAI; on 2026-10-19 neither public board lists a
`ujs-iv` submission, so the two images would be two of the account's three October slots (`limits` gives the exact count after sign-in).

```bash
P=ujs-iv
~/bin/ap $P me                                        # token valid 12 h; if expired: ~/bin/ap $P auth-url, sign in, ~/bin/ap $P configure-token
~/bin/ap $P limits                                    # remaining official nuplan slots this UTC month: must be >= 2
REPO=$(~/bin/ap $P ecr-login | tee /dev/stderr | sed -n 's/^Team repository: //p'); echo "REPO=$REPO"
[[ -n $REPO ]] || echo "STOP: ecr-login printed no team repository (terms not accepted? run: ~/bin/ap $P terms status)"
# expected: 696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/ujs-iv; any other team -> stop, wrong profile

push() {   # push <image id> <remote tag>: tag by id, push (bounded retries), print the remote digest
  docker image inspect "$1" >/dev/null || return 1
  docker tag "$1" "$REPO:$2" || return 1
  for i in 1 2 3 4 5; do ~/bin/ap $P docker push "$REPO:$2" && break; [[ $i == 5 ]] && return 1; sleep 15; done
  ~/bin/ap $P docker manifest inspect "$REPO:$2" >/dev/null && docker image inspect "$REPO:$2" --format '{{.Id}} {{.RepoDigests}}'
}

# image 1: native (no hold), version jev-JTN-L18-s1-backwarp
push sha256:1b68246103ec1fd106482eeec04c3e9b9fddb11a9f896148bde7b1a8f03dea22 jtn-l18-s1-native
#   must print sha256:1b682461... and $REPO@sha256:<digest the push printed>
~/bin/ap $P submit --track nuplan "$REPO:jtn-l18-s1-native"          # official (optional first: --track nuplan-warmup, own quota, status only)

# image 2: hold stop3, version jev-JTN-L18-s1-backwarp-holdstop3
push sha256:b547cca3bbec62899785c618eea9b9d0241de88a032a2b2aa02f523b5f8d2440 jtn-l18-s1-holdstop3
~/bin/ap $P submit --track nuplan "$REPO:jtn-l18-s1-holdstop3"

~/bin/ap $P submissions --track nuplan; ~/bin/ap $P status SUBMISSION_ID          # id printed by submit; the first rollout's version must be the one named above
```

Submitting by tag is the path of every earlier successful submission; the tag points at the digest printed by `push` (nobody else pushes to this
repository). The CLI also accepts `$REPO@sha256:<digest>` (it refuses only `latest`), but the board labels a submission by its tag, so the tag form is
kept. The second push uploads only the image config: the two images share all 18 rootfs layers. Remote tags are public on the leaderboard.

Last verified: 2026-10-19

## Submitted (2026-10-19, profile ujs-iv, track nuplan)

| image | remote tag | registry digest | submission id |
|---|---|---|---|
| native `1b682461` | `teams/ujs-iv:jtn-l18-s1-native` | `sha256:81a7deaceb3d55b2d02a76c89d206ff7f730afa8537b45f5db51bd9d0cfcb742` | `91f5b902-e938-49b1-93de-cf20f4c80e88` |
| stop3 `b547cca3` | `teams/ujs-iv:jtn-l18-s1-holdstop3` | `sha256:e81aa87386d140b9718220771379fe91db6f73e8ae333c0f213a2262baab57e2` | `a3836ef6-ae58-4b6d-9274-a3ede5272fb4` |

Script and log: Tokyo `/data/runs/alpasim/submit_jtn_s1/` (`run.sh`, `log.txt`; `log_try1.txt` is the first attempt, which pushed the native image
and then stopped on a `docker image inspect --format '{{join .RepoDigests ...}}'` template error before submitting; the Tokyo docker rejects `join`
on that field, so the record's `push()` helper must print `{{.RepoDigests}}` instead). Monthly official quota for ujs-iv: 3 -> 1.
