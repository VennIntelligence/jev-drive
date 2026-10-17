# nuPlan submission image: RSG-L18-s0 + serve-side hold `JEV_HOLD=stop3` (2026-10-18)

Status: built and tested; **independent audit GO** ([image_rsg_stop3_audit.md](image_rsg_stop3_audit.md)), nothing pushed or submitted yet. The candidate of
[decision 351](../../../research/decisions/351.md) (arena held windows: leash3 0.934 -> stop3 0.944; RSG-L18-s0 0.786). It supersedes the
leash3 image ([image_rsg_leash3.md](image_rsg_leash3.md)), which stays the audited fallback. It includes the two code fixes that the leash3
audit ([image_rsg_leash3_audit.md](image_rsg_leash3_audit.md), items 8 and 9) asked for in the next image. Built on the Tokyo box with
`docker/build.sh` from a frozen clone at `ba3ef44d` (`/data/runs/alpasim/tokyo_image/code-rsg-stop3/`, clean, no `-dirty`). No pool job was
paused or cancelled.

## Image

| | |
|---|---|
| Local name (Tokyo Docker store) | `jev-alpasim:rsg-l18-s0-ba3ef44d-holdstop3` |
| Image id | `sha256:51401086aaed3ad95f4781428e513559b7ceb4dcbf549aee85a541536389d4e2` (not pushed, so no registry digest yet) |
| Size | 8 105 637 984 bytes (7.55 GiB; limit 40 GiB) |
| Build | `env DATA_DIR=/data EXTRA=RSG-L18-s0 TAG=RSG-L18-s0 DRIVER=sh30 HOLD=stop3 OUT=... experiments/alpasim/docker/build.sh` (the leash3 build with `HOLD=stop3`) |
| ENV | `JEV_TAG=RSG-L18-s0 JEV_DRIVER=sh30 SH30_GIT_HASH=ba3ef44d JEV_VCONT=0 JEV_LEAD=0 JEV_ACC=0 JEV_VHIST= SH30_COMPILE=0 JEV_GAP= JEV_HOLD=stop3`; CMD `python /app/serve.py`, user 10001:10001 |
| Served path | `serve: driver sh30 (sh30_hold.py), tag RSG-L18-s0, ...; switches ... JEV_HOLD=stop3`. Inside the image `static_hold` parses mode `stop` 3.0 m, latch 0.3 m/s, stop rule 0.5 m/s^2 within 3 s, suffix `-holdstop3` ([limits/cpu_import.txt](image_rsg_stop3/limits/cpu_import.txt)) |
| Version (gRPC `get_version`) | **`jev-RSG-L18-s0-backwarp-holdstop3`**, git `ba3ef44d`. The leash3 image reported `jev-RSG-L18-s0-backwarp` |
| Checkpoint | `RSG-L18-s0/ckpt-final.pt` sha256 `fbe7ac9f...f05143`, `route_mem.json` `672ef154...4ade`: unchanged, equal to the Tokyo originals and to the leash3 image |

**Against the leash3 image** (`394b1137`): rootfs layers 1-12 are identical; layers 13-18 are rebuilt (code layer and everything above it). Of
all the files under `/app` (the unchanged models layer excluded), exactly two differ: `experiments/alpasim/lib/sh30_hold.py` and
`static_hold.py` ([limits/app_diff_vs_leash3.txt](image_rsg_stop3/limits/app_diff_vs_leash3.txt)). Every other file is byte-equal: checkpoints,
sidecar, `serve.py`, `verify.py`, `tags.json` and the rest of the closure. The 31 baked jev-drive files equal the clone at `ba3ef44d`.

**Against the stop3 arena reads' code** (`7b709626`, the 69175511 readout): over the closure, only `sh30_hold.py` differs (`git diff 7b709626
ba3ef44d`). `static_hold.py`, `sh30_driver.py` and the rest are identical.

## Code changes (commit `ba3ef44d`)

1. **Fail-open hold** (audit item 8). In `sh30_hold.make_cached_plan`, the per-call hold (session state, speed, `static_hold.step`, the
   `hold.jsonl` write) is wrapped in `try/except Exception`. An exception serves the model's own plan for that `Drive` call. If only the log
   write failed, the hold's plan is served. The first exception per process is logged with its traceback (`JEV_HOLD: the hold raised ...`).
   When nothing raises, the output path is unchanged.
2. **Version string** (audit item 9). `sh30_hold.install` also patches `Driver.get_version` to append `static_hold.SUFFIX`
   (`-holdstop3`).
3. **`build.sh`** accepts `HOLD=stop<m>` (regex `full|lead|(leash|stop)<m>`). The Dockerfile change is a comment only.
4. **`hold_check.py` check 5** (box, `envs/op-train`; all checks pass). The real `sh30_hold.install` runs with `JEV_HOLD=stop3`, and the
   driver's `make_cached_plan` is replaced by a recorder. Across 2 200 calls on random sessions (1 380 held, 1 331 cut, stop latches
   included), the wrapper hands on exactly what `static_hold.step` serves: the same object when uncut, bit-identical when cut. `hold.jsonl`
   holds step's records. With `step` made to raise, 20 of 20 calls hand on the model's plan object, and one log line is written. The
   version is `jev-RSG-L18-s0-backwarp-holdstop3`. Checks 1-4 are unchanged, including "off patches nothing".
5. `scripts/hold_parity.py` (not in the image): for a stop latch, the latch column also shows its deceleration and leash.

**Fail-open does not change served behaviour (box, served path).** The arena stop3 smoke windows (`lists/smoke2.txt`, 12 windows, shard 6)
were rerun natively at `ba3ef44d` (pool job `1018-014835-ec0e`). They were compared with the original stop3 smoke at the code the reads ran
(`$H/smoke/stop3/run`):

- latch 12 / 12;
- hold state and cut flag 120 / 120;
- served plans max diff **0.0000 m**;
- `hold.jsonl` records (session id aside) and the model's `drive.jsonl` plans: **identical, 120 / 120**;
- scores equal 12 / 12.

The fail-open line never appears in any run ([failopen/smoke2_box_rerun.md](image_rsg_stop3/failopen/smoke2_box_rerun.md)). The 69175511
arena reads therefore remain the behavioural reference of this image.

## Constraint test (`test.sh`, Tokyo RTX 3090 card 1, organisers' flags + `--network none`, by image id)

Pass ([test/result.json](image_rsg_stop3/test/result.json)):

| check | result |
|---|---|
| cold start | 17.7 s |
| driver GPU | 1 603 MiB |
| `/tmp` / `/run` | 0.40 MiB / 0 |
| read-only root | 0 errors, 0 files changed |
| 160 `drive` calls, p50 / p90 / p99 / max | 50.0 / 51.6 / 61.3 / 69.1 ms |
| inference errors / tracebacks | 0 / 0 |
| SIGTERM | exit 0 in 0.6 s |
| probe version | `jev-RSG-L18-s0-backwarp-holdstop3`, git `ba3ef44d` |
| hold live | `/tmp/alpasim-driver/hold.jsonl` 17 kB |

## 48-scene smoke (official containerised stack, `smoke.sh`, render card 0 / driver card 1, by image id)

The reference is a native box read of the same 48 public scenes with the same hold at `ba3ef44d` (pool job `1018-014836-8788`:
`run.sh <dir> sh30hold` with `JEV_HOLD=stop3 SH30_TAG=RSG-L18-s0` and the leash3 reference's overrides). It was copied to Tokyo
`/data/runs/alpasim/native_ref/RSG-L18-s0-holdstop3/`.

| | image (container) | native stop3 (box) | leash3 image smoke |
|---|--:|--:|--:|
| mean, 48 scenes | **0.9184** | 0.9180 | 0.9184 |
| zeros / at 1 | 1 / 29 | 1 / 29 | 1 / 29 |

- Against native ([smoke/compare.md](image_rsg_stop3/smoke/compare.md)):
  - 18 of 48 scenes differ, max 0.0104, paired mean +0.0005;
  - 47 are within 0.01, and zero / non-zero agree 48 / 48.
- Driver: 480 `drive` calls, 0 inference errors, server p50 / p90 / max 36.2 / 50.3 / 140.1 ms. The fail-open line is absent.
- Against the leash3 image's smoke, all 48 scene scores are identical. On this public log no session stop-latches: 5 are held at
  < 0.3 m/s (the same 5 as leash3) and 43 are free. The public board should read as the leash3 image would.
- Latch parity on these 48 ([parity_public48.md](image_rsg_stop3/parity_public48.md)): latch state 48 / 48, hold state and cut flag
  480 / 480. Served plans have a median diff of 4.7 cm. The 3.1 m maximum comes from closed-loop divergence late in a free rollout: the
  scores of those scenes agree within 0.0104.

## Parity on 12 held arena windows (served plans and latch events)

The 12 windows come from the stop3 read's shard-1 chunks c3-c6 ([parity12.txt](image_rsg_stop3/parity12.txt)):

- 3 static latches, including the static zero 86280cbf and a latch at 0.274 m/s;
- **5 stop-rule latches**: the two leash3 corridor zeros that stop3 fixed (e1f8bd3d, f831df1e), one at 4.1 m/s with a 6.6 m leash, one at
  0.35 m/s just above the static threshold, and one that stop3 cost 0.004;
- 2 launch windows;
- 2 unlatched windows (2.28 and 0.695 m/s).

They were run through the image in the containerised stack (`smoke.sh`, `SCENES=parity12.txt`, cache = the shard-1 copy
`/data/datasets/alpasim_nuplan/_arena_hold_parity/s1` from the leash3 parity) and compared with the box's stop3 read logs
(`reads/stop3/c{3..6}/run`; `hold_parity.py`; table [parity_arena12.md](image_rsg_stop3/parity_arena12.md)):

- **Latch events: 12 / 12 equal** (10 held, of which 5 are stop latches; 2 free), all at the 0.5 s call with the same measured speed. Each
  stop latch has the same deceleration and leash (e.g. a -1.339 cap 3.359, a -2.333 cap 6.636).
- **Hold state and cut flag: 120 / 120 equal.**
- **Served plans:** at decision 0, median 0.65 cm, max 2.9 cm. Over all decisions, median 0.8 cm, p90 3.3 cm, max 16 cm (on an unlatched
  window). Max distance from the latch point is equal to 1 cm (3.26 / 3.26, 6.17 / 6.17 m).
- **Scores:** 8 / 12 identical, 12 / 12 within 0.01; mean 0.8426 box / 0.8425 image; the same zero (86280cbf). The two leash3 zeros fixed
  by the stop rule score 1.0 on both sides.

The image runs interpreted passes on an RTX 3090 (`SH30_COMPILE=0`), the box ran compiled passes on an RTX 6000D. The plans therefore
agree to cm level but are not bit-identical, as for every earlier image (decision 215).

## Limits and secrets

- Size: 7.55 GiB.
- GPU: driver 1.6 GiB of 16 GiB.
- Read-only root, `/tmp` 0.4-1.6 MiB of 2 GiB, no network access needed (`--network none` in test.sh; smoke on an `--internal` network,
  where the internet probe fails).
- `docker history --no-trunc`: no token, proxy, credential URL or host address. The only pattern hit is `HF_HOME=/tmp/huggingface`.
- Filesystem (as root): no `.env`, `.netrc`, `.git-credentials`, ssh key, `*.key`, `.git`, `.ssh`, `.docker`, `challenge.json`,
  `credentials*` or `.pypirc`. No `hf_` / `AKIA` / `ghp_` / `github_pat_` / `nvapi-` / private key / proxy port / box address / `github-venn`
  / e-mail under `/app /etc /root /home /tmp /usr/local/bin /opt`.
- The container env has no token or proxy variables. The only `*KEY*` is the Python base's public `GPG_KEY` id.

## For the auditor: what changed

- Since the leash3 image (`e7db399e`): `experiments/alpasim/lib/sh30_hold.py` (fail-open + version, the only served-code change beyond
  `static_hold.py`'s stop mode); `experiments/alpasim/lib/static_hold.py` (the `stop<m>` mode, 7b709626, the code the arena reads ran);
  `experiments/alpasim/docker/build.sh` (HOLD regex, header); `experiments/alpasim/docker/Dockerfile` (comment);
  `experiments/alpasim/scripts/hold_check.py` (check 4 stop3 from 7b709626, check 5); `experiments/alpasim/scripts/hold_parity.py`
  (stop-latch column).
- Things to look at:
  1. The except branch in `make_cached_plan` wraps the state creation, `HD.step` and the log write. If only the write fails, the hold's plan
     is served and that decision's record is lost.
  2. `get_version` is patched on the `Driver` class. The suffix comes from `static_hold.SUFFIX` (`-hold` + the raw `JEV_HOLD`).
  3. `install`'s own `serving JEV_HOLD=...` info line still precedes the driver's logging setup and does not appear. This is unchanged and
     non-blocking. The serve line and `hold.jsonl` show that the hold is live.

Files left in place:

- Tokyo:
  - the frozen clone `code-rsg-stop3`;
  - `/data/runs/alpasim/tokyo_image/RSG-L18-s0-holdstop3/` (test, smoke, parity, limits, box_ref = the stop3 read logs c3-c6);
  - `RSG-L18-s0-holdstop3-build/`;
  - `/data/runs/alpasim/native_ref/RSG-L18-s0-holdstop3/`.
- AutoDL box: `$DATA_DIR/runs/alpasim/hold_image/{native48_stop3,smoke2_stop3_failopen}/`.

## What the user runs (not run by an agent)

On the Tokyo box, after the audit. The steps are the same as for the leash3 image (accounts are `~/bin/ap` profiles; quota is 3 official
submissions per UTC month shared with PAI; only the code layer and the layers above it upload, about 230 MB).

```bash
P=lao-siji                                            # or ujs-iv / acct3: the account this probe is spent on
ap $P me; ap $P limits                                # token valid 12 h; if expired: ap $P auth-url; ap $P configure-token
ap $P ecr-login                                       # prints the team repository URI
IMG=jev-alpasim:rsg-l18-s0-ba3ef44d-holdstop3
docker image inspect "$IMG" --format '{{.Id}}'        # must print sha256:51401086aaed3ad95f4781428e513559b7ceb4dcbf549aee85a541536389d4e2
URI=696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/lao-siji:rsg-l18-s0-holdstop3-v1   # team from ecr-login; public tag, not `latest`
docker tag "$IMG" "$URI"
until ap $P docker push "$URI"; do sleep 10; done     # note the digest it prints
ap $P submit --track nuplan-warmup "$URI"             # optional warm-up: 32 private scenes, status only
ap $P submit --track nuplan "$URI"                    # official
ap $P status <submission_id>; ap $P submissions --track nuplan
```

Fallback: the leash3 image (`jev-alpasim:rsg-l18-s0-e7db399e-holdleash3`, `sha256:394b1137...`, audited GO) with the commands in
[image_rsg_leash3.md](image_rsg_leash3.md).

Last verified: 2026-10-18
