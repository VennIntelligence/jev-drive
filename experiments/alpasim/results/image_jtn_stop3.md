# nuPlan submission image: JTN-L18-s0 + serve-side hold `JEV_HOLD=stop3` (2026-10-18)

Status: built and tested; **no independent audit yet**, nothing pushed or submitted. The same procedure and the same code as the RSG-L18-s0 stop3
image ([image_rsg_stop3.md](image_rsg_stop3.md), audited GO), with the checkpoint/tag swapped RSG-L18-s0 -> JTN-L18-s0. JTN is the no-arc-term arm of
[decision 353](../../../research/decisions/353.md) ([readout](../../defect_fix/results/jtb/README.md)): under stop3 it ties RSG (arena held 2 228: -0.001
[-0.009, +0.010]), native +0.089. It is a deploy-equivalent alternative, not an expected improvement. Built on the Tokyo box from the existing frozen clone
at `ba3ef44d` (`/data/runs/alpasim/tokyo_image/code-rsg-stop3/`, clean, no `-dirty`), i.e. the exact code of the RSG stop3 image. No code change was
needed: `tags.json` is generated at build time by `verify.py` from the checkpoint dir, and `serve.py` takes the family from it. No pool job was paused or cancelled.

## Image

| | |
|---|---|
| Local name (Tokyo Docker store) | `jev-alpasim:jtn-l18-s0-ba3ef44d-holdstop3` |
| Image id | `sha256:74ad6b91e269468910a72982d49c2638dfe25db16a2bc7ba6890f03387c9cf7b` (not pushed, no registry digest) |
| Size | 8 105 640 419 bytes (7.55 GiB; limit 40 GiB); the RSG stop3 image is 8 105 637 984 |
| Build | `env DATA_DIR=/data EXTRA=JTN-L18-s0 TAG=JTN-L18-s0 DRIVER=sh30 HOLD=stop3 OUT=... experiments/alpasim/docker/build.sh` ([build/](image_jtn_stop3/build)) |
| ENV | `JEV_TAG=JTN-L18-s0 JEV_DRIVER=sh30 SH30_GIT_HASH=ba3ef44d JEV_VCONT=0 JEV_LEAD=0 JEV_ACC=0 JEV_VHIST= SH30_COMPILE=0 JEV_GAP= JEV_HOLD=stop3`; CMD `python /app/serve.py`, user 10001:10001 |
| Version (gRPC `get_version`) | **`jev-JTN-L18-s0-backwarp-holdstop3`**, git `ba3ef44d` |
| Checkpoint | `JTN-L18-s0/ckpt-final.pt` sha256 `10f84006d2f5152494d5ea5155dee23b635ddc78eacf965182f66fb9f7a73a46`; `route_mem.json` `b9ef092253646893bfbe857205e4cd4f08f97b9f31183d2d16beb398df595b72`. Box originals (`~/data/runs/op_parity/runs/JTN-L18-s0/`) and the Tokyo copy are equal, and equal to the files in the image. The box dir also holds `adapter.pt`, which no image carries |
| `tags.json` in image | `{"AP2-AB-s0": "ap2", "JTN-L18-s0": "sh30", "P2H10-F-s0": "sh30"}` |

**Against the RSG stop3 image** (`51401086`): rootfs layers 1-16 are identical, layers 17-18 (extra checkpoint layer and `tags.json`) differ. File
sha256 over `/app` ([limits/app_diff_vs_rsg_stop3.txt](image_jtn_stop3/limits/app_diff_vs_rsg_stop3.txt), 47 files each): exactly the expected differences.
RSG's `RSG-L18-s0/{ckpt-final.pt,route_mem.json}` are absent (the RSG image carries them in its extra layer; here only JTN does), the two JTN files are added,
and `tags.json` differs (RSG -> JTN entry). Code (`serve.py`, `verify.py`, `sh30_hold.py`, `static_hold.py`, the whole closure), `cinque.ort.onnx`, and the
stable checkpoints `AP2-AB-s0` / `P2H10-F-s0` are byte-equal. Environment differs only in `JEV_TAG`. Hold parse inside the image: `static_hold.SUFFIX` = `-holdstop3`.

## Constraint test (`test.sh`, Tokyo RTX 3090 card 1, organisers' flags + `--network none`, by image id)

Pass ([test/result.json](image_jtn_stop3/test/result.json)):

| check | result |
|---|---|
| cold start | 17.7 s |
| driver GPU | 1 603 MiB |
| `/tmp` / `/run` | 0.38 MiB / 0 |
| read-only root | 0 errors, 0 files changed |
| 160 `drive` calls, p50 / p90 / p99 / max | 50.1 / 52.1 / 65.0 / 69.3 ms |
| inference errors / tracebacks | 0 / 0 |
| SIGTERM | exit 0 in 0.6 s |
| probe version | `jev-JTN-L18-s0-backwarp-holdstop3`, git `ba3ef44d` |

## 48-scene smoke (official containerised stack, `smoke.sh`, render card 0 / driver card 1, by image id)

Reference: native box read of the same 48 public scenes, JTN + stop3 (pool job `1018-100141-234a`, the overrides of the RSG stop3 reference job
`1018-014836-8788` with `SH30_TAG=JTN-L18-s0`, at box commit `d6d7e61f`, whose alpasim serve code equals `ba3ef44d`); copied to Tokyo
`/data/runs/alpasim/native_ref/JTN-L18-s0-holdstop3/`.

| | image (container) | native JTN stop3 (box) |
|---|--:|--:|
| mean, 48 scenes | **0.9383** | 0.9370 |
| zeros / at 1 | 0 / 33 | 0 / 33 |

- Against native ([smoke/compare.md](image_jtn_stop3/smoke/compare.md)): 15 of 48 scenes differ, max 0.0365 (scene `...00083_00485-83fd1c19`, 0.9796
  vs 0.9431), paired mean +0.0012, 47 within 0.01, all 48 within 0.05, zero / non-zero agree 48 / 48. The RSG stop3 smoke had max 0.0104; the larger
  single-scene gap is a closed-loop divergence of the kind seen in RSG's late-rollout plans, not a systematic offset.
- Driver: 480 `drive` calls, 0 inference errors, 0 tracebacks, server p50 / p90 / max 36.3 / 43.1 / 150.5 ms. The fail-open line
  (`JEV_HOLD: the hold raised`) is absent.
- Hold live: `hold.jsonl` 480 records; 5 sessions are held (< 0.3 m/s), 43 free; no stop-rule latch evidence beyond that was examined here.
- RSG stop3 image on the same scenes: 0.9184, 1 zero. JTN scores 0.9383, 0 zeros on this public log (descriptive, 1 seed).

Not repeated for this image: the 12 held-arena-window latch parity and fail-open rerun of the RSG doc. The served code is byte-identical to that image,
so they are inherited, not re-measured.

## Limits and secrets

- Size 7.55 GiB; GPU driver 1.6 GiB (smoke peak 1.76 GiB); `/tmp` 0.4-1.6 MiB; no network needed (`--network none`; smoke on an `--internal` network where the internet probe fails).
- `docker history --no-trunc`: no token, proxy, credential URL or host address; the only patterns hit are the Python base's public `GPG_KEY` and an apt `@` timestamp.
- Filesystem scan as root ([limits/secrets_fs.txt](image_jtn_stop3/limits/secrets_fs.txt)): none of `.env .netrc .git-credentials id_* *.key .git .ssh .docker challenge.json credentials* .pypirc`.
  The pattern grep hits only `/app/pai_spec.json` line 257 on `7890` inside the float `1570.4058837890625` (false positive; the file is byte-equal to the RSG image's).
- Container env has no token or proxy variable; the only `*KEY*` is `GPG_KEY`.

## For the auditor

- Same code as the audited RSG stop3 image; changed inputs are only the JTN checkpoint + sidecar, `tags.json`, `JEV_TAG` and the version string.
- Look at: (1) the checkpoint sha above against the box dir; (2) `app_diff_vs_rsg_stop3.txt`; (3) the JTN `route_mem.json` is read by the sh30 route-memory loader (probe and
  smoke served it without error, 0 inference errors); (4) the box commit for the native reference is `d6d7e61f`, not `ba3ef44d` (alpasim serve code identical).

Files left in place on Tokyo: `/data/runs/alpasim/tokyo_image/JTN-L18-s0-holdstop3/` (build.log, test, smoke, limits, IMAGE), `JTN-L18-s0-holdstop3-build/`,
`/data/runs/alpasim/native_ref/JTN-L18-s0-holdstop3/`, `/data/runs/op_parity/runs/JTN-L18-s0/`, chain script `jtn_chain1.sh`. AutoDL box:
`~/data/runs/alpasim/hold_image/native48_jtn_stop3/`.

## What the user runs (not run by an agent; after the audit)

Same as [image_rsg_stop3.md](image_rsg_stop3.md#what-the-user-runs-not-run-by-an-agent) with
`IMG=jev-alpasim:jtn-l18-s0-ba3ef44d-holdstop3`, expected id `sha256:74ad6b91e269468910a72982d49c2638dfe25db16a2bc7ba6890f03387c9cf7b`, tag e.g. `jtn-l18-s0-holdstop3-v1`.

Last verified: 2026-10-18
