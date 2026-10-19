# Independent audit: nuPlan images JTN-L18-s1 native and JTN-L18-s1 + `JEV_HOLD=stop3` (2026-10-19)

Images in the Tokyo Docker store:

| | native | stop3 |
|---|---|---|
| name | `jev-alpasim:jtn-l18-s1-ba3ef44d` | `jev-alpasim:jtn-l18-s1-ba3ef44d-holdstop3` |
| id | `sha256:1b68246103ec1fd106482eeec04c3e9b9fddb11a9f896148bde7b1a8f03dea22` | `sha256:b547cca3bbec62899785c618eea9b9d0241de88a032a2b2aa02f523b5f8d2440` |

The builder's record is [image_jtn_s1.md](image_jtn_s1.md) (d7ea8bfc). The method follows the RSG stop3 audit ([image_rsg_stop3_audit.md](image_rsg_stop3_audit.md), GO).
Intended use: a clean A/B on the private nuPlan board, so the pair must differ only in the hold, and the native image must be the arena "native" read
(decisions 371 / 372) of JTN-L18-s1.

Every check was rerun on the images (by id), on git, on the AutoDL box and on the challenge CLI source. Nothing was taken from the builder's record.
Nobody logged in to a registry, nothing was pushed or submitted, the images were not modified, no pool job was touched. The only challenge-API calls were
the read-only `me` (it answered "expired session") and `leaderboard`.
Auditor's run dir on Tokyo: `/data/runs/alpasim/tokyo_image/JTN-L18-s1-audit/` (`x/` the extracted `/app` trees, `native/` and `stop3/` with `test/` and `smoke/`, `chain*.sh`).

**Verdict: native GO, stop3 GO.** One defect, in the record's submit commands (item 7), fixed in the record. Notes are non-blocking.

| # | item | native | stop3 |
|---|---|---|---|
| 1 | The pair differs only in `JEV_HOLD` | PASS | PASS |
| 2 | Checkpoint is JTN-L18-s1, served at run time | PASS | PASS |
| 3 | Native serves no hold and no other lever | PASS | - |
| 4 | stop3 = the audited RSG stop3 config except the checkpoint | - | PASS |
| 5 | Code and settings equal the arena reads the selection rests on | PASS | PASS |
| 6 | Organiser limits and GPU rerun (test.sh + 48-scene smoke) | PASS | PASS (public drop is by design) |
| 7 | Push / submit commands for `ujs-iv` | FIXED | FIXED |
| 8 | Quota | see item 8 | see item 8 |

## 1. The pair differs only in `JEV_HOLD`: PASS

- Rootfs: both images have the same 18 layer diff ids, in order.
- Full `docker image inspect` minus id / tags / created: the only difference is `JEV_HOLD=` vs `JEV_HOLD=stop3`. CMD `python /app/serve.py`, user
  10001:10001, STOPSIGNAL SIGTERM, workdir `/app`, every other ENV equal. Both have the same `Created` timestamp (one build context).
- `docker history --no-trunc`: the only lines that differ are `ARG HOLD=` / `ENV JEV_HOLD=` vs `...=stop3` (0-byte metadata).
- `/app` copied out of both images and hashed (47 files each): byte-identical. Since the layers are equal, so is every other file.

So the A/B is exactly "hold on / hold off": the hold code ships in both images and only `serve.py`'s reading of `JEV_HOLD` turns it on.

## 2. Checkpoint: PASS (both)

| file | image | AutoDL `~/data/runs/op_parity/runs/JTN-L18-s1/` | Tokyo copy |
|---|---|---|---|
| `ckpt-final.pt` (77 552 223 B) | `d9c134bfdc692548aa4c9d3b1f5b5a065a1eaa7fa0c3704a1e06e2ebb062b2dc` | same | same |
| `route_mem.json` | `c2dbdd307593d528069a3063357bd87783e4b19f393ae6e85d40b94649d1bd44` | same | same |

- `tags.json` in the images: `{"AP2-AB-s0": "ap2", "JTN-L18-s1": "sh30", "P2H10-F-s0": "sh30"}`; `JEV_TAG=JTN-L18-s1`, `JEV_DRIVER=sh30`. No RSG-L18-s0 or
  JTN-L18-s0 file is in the image (the other two checkpoints are the long-standing stable ones and are never selected with `JEV_TAG=JTN-L18-s1`).
- `serve.py` reads `JEV_TAG`, checks it against `tags.json`, sets `SH30_TAG=JTN-L18-s1` and execs the driver; `sh30_core.Core` loads
  `runs/JTN-L18-s1/` through `rmem.load_pmodel` (the sidecar `route_mem.json` is present, so the ROUTE-MEM branch). The test.sh driver logs say
  `serving the ROUTE-MEM branch, memory on, layout -6..+72` and `JTN-L18-s1 (backwarp, motion 1.00, synth gpu, compiled False, jpeg libjpeg) ready`.
- The box dir also holds `adapter.pt`. It is a training by-product (`pp_train.py` saves it next to the checkpoint); no serving code reads it (grep over the
  image and over the box's serving libs), and the adapter weights are inside `ckpt-final.pt`. Its absence from the image is correct.

## 3. Native serves no hold and no other lever: PASS

- `serve.py`: `hold = JEV_HOLD.strip().lower() not in ("", "off")`, so empty means no hold, and the entry is `sh30_driver.py`, not `sh30_hold.py`.
  `static_hold` imported with `JEV_HOLD=` gives `ON False, MODE off, SUFFIX ''`; nothing in `sh30_driver.py` imports `sh30_hold`.
- Every serve-side switch is off by its parser: `JEV_VCONT=0` (`float(...) = 0`), `JEV_LEAD=0` and `JEV_ACC=0` (`== "1"` false), `JEV_GAP=` (empty),
  `JEV_VHIST=` (`"" or "off"`), and unset `JEV_TRACK`, `JEV_ROUTE`, `JEV_EXPERT`, `JEV_PERTURB`, `JEV_DESIRE`, `JEV_TTS`, `JEV_STOP`, `JEV_REPLAN`, `JEV_BASE`.
  No leash: a leash exists only in `static_hold`, which is not loaded.
- Measured in the auditor's test.sh run: serve line `serve: driver sh30 (sh30_driver.py), tag JTN-L18-s1, ...; JEV_HOLD=`; driver line
  `serving: JEV_VCONT 0, JEV_LEAD 0, JEV_BASE 0, JEV_VHIST off, JEV_ROUTE off`; no `serving JEV_ACC` / GAP / TRACK / EXPERT / PERTURB / STOP / TTS / DESIRE /
  trajectory-head line; no `hold.jsonl` in `/tmp/alpasim-driver` (stop3 run: `hold.jsonl` 17 kB). Version `jev-JTN-L18-s1-backwarp`, git `ba3ef44d`.

## 4. stop3 = the audited RSG stop3 configuration except the checkpoint: PASS

- Against the audited RSG stop3 image (`sha256:51401086...`): rootfs layers 1-16 identical, **including the jev-drive code layer and the serve-files layer**;
  layers 17-18 (checkpoint, `tags.json` / verify) differ. Config ENV differs only in `JEV_TAG` (RSG-L18-s0 -> JTN-L18-s1); CMD, user, signal, ports equal.
  So the served code, the hold code and every switch are bit-for-bit those audited GO on 10-18.
- Hold constants read from the image's own `static_hold.py` with `JEV_HOLD=stop3`: mode `stop`, CAP 3 m, V_LATCH 0.3 m/s, T_HAND 0.5 s, A_STOP 0.5 m/s^2,
  T_STOP 3 s, REL_ARC 2 m, V_LEAD_GO 1 m/s, SUFFIX `-holdstop3`, as in the RSG stop3 audit. Version `jev-JTN-L18-s1-backwarp-holdstop3`.
- The latch-parity and hold-semantics probes of the RSG audit (items 8 and 9 there) exercise code that is byte-identical here and independent of the
  checkpoint; they are inherited, not rerun.
- Note: the `install` log line (`serving JEV_HOLD=stop3 ...`) does not appear in the driver log because it is emitted before the driver configures logging;
  same in the audited RSG image. The hold being live is shown by `hold.jsonl` (test and smoke) and the version suffix.

## 5. Code and settings equal the reads the selection rests on: PASS

The selection reads are the arena held split of JTN-L18-s1 (`experiments/defect_fix/results/jtb/JTN-L18-s1/arena.md`: `@off` 0.8909 = decision 371's
"native 0.891"; `@stop3` 0.9428), made by `jt.sh aread` on the AutoDL box between 2026-10-18 21:30 and 22:10 CST (18 chunks per serve).

- **Code.** The 31 jev-drive files baked in the image were hashed against `git show <rev>:<path>`: 31 / 31 equal at `ba3ef44d` (the build, clean clone,
  `git describe --dirty` = `ba3ef44d`) and at every commit the box checkout was at during the reads (`5490eb4b`, `5f9873d3`, `3a04498e`, `4b9b6207`, and
  `0e20f2cf` just after). `serve.py`, `verify.py`, `probe.py`, `pai_spec.json`, `requirements.lock` equal `experiments/alpasim/docker/` at `ba3ef44d` and HEAD.
  The challenge sample's `navigation.py` / `trajectory.py` and `cinque.ort.onnx` equal the box's copies. Today's HEAD differs only in `sh30_core.py`
  (the `SH30_WARP` diagnostic switch, default off, not in the image).
- **Driver and settings of the reads.** `@off` ran `run.sh <dir> sh30hold` with `JEV_HOLD` unset: `sh30_hold.py` with `HD.ON` false calls `sh30_driver.main()`
  with nothing patched, i.e. the plain sh30 driver. All 18 `@off` chunks: no `hold.jsonl`, driver line `serving: JEV_VCONT 0, JEV_LEAD 0, JEV_BASE 0,
  JEV_VHIST off, JEV_ROUTE off`, `backwarp, motion 1.00, synth gpu`, no ACC / GAP / other switch line, ROUTE-MEM on. All 18 `@stop3` chunks: the same plus
  `hold.jsonl`. The images log the same lines.
- **Known deviation (all images, decision 215):** the box reads ran compiled passes on an RTX 6000D (`compiled True`); the image runs interpreted
  (`SH30_COMPILE=0`) on whatever GPU the organisers give. cm-level plan differences, shown by the smoke parity below.

## 6. Organiser limits and GPU rerun: PASS (both)

Auditor's own `test.sh` (organisers' flags, `--read-only`, `--cap-drop ALL`, `--network none`, 32 GiB / 8 CPU, tmpfs `/tmp` 2 GiB, card 1, by image id):

| check | native | stop3 |
|---|--:|--:|
| size | 8 105 640 419 B (7.55 GiB; limit 40 GiB) | same |
| cold start to listening | 17.9 s | 17.6 s |
| driver GPU peak | 1 603 MiB (limit 16 GiB) | 1 603 MiB |
| `/tmp` / `/run` peak | 0.38 MiB / 0 | 0.40 MiB / 0 |
| read-only root: errors / `docker diff` lines | 0 / 0 | 0 / 0 |
| 160 `drive` calls p50 / p90 / p99 / max | 50.0 / 53.1 / 68.0 / 70.0 ms | 50.2 / 52.0 / 68.7 / 70.9 ms |
| inference errors / tracebacks | 0 / 0 | 0 / 0 |
| SIGTERM | exit 0 in 0.6 s | exit 0 in 0.6 s |
| probe version, git, API | `jev-JTN-L18-s1-backwarp`, ba3ef44d, 0.54.0 | `jev-JTN-L18-s1-backwarp-holdstop3`, ba3ef44d, 0.54.0 |

Secrets: the filesystem is the RSG stop3 image's (secrets-clean, audited) with layers 17-18 replaced by two checkpoint files and `tags.json`; the config
ENV has no token / proxy / host variable (only the Python base's public `GPG_KEY`); `docker history` has no credential.

**48-scene smoke** (official containerised stack `alpasim-base:0.89.0`, driver card 1, render card 0, the builder's wizard overrides, by image id), against
the box native reads of the same scenes (`native_ref/JTN-L18-s1-{native,holdstop3}`, pool jobs `1019-105949-ed5a` / `-f7b9`):

| | audit run | builder run | box native read |
|---|--:|--:|--:|
| native: mean, 48 scenes | **0.9828** | 0.9828 | 0.9828 |
| native: zeros / at 1 | 0 / 37 | 0 / 37 | 0 / 37 |
| stop3: mean, 48 scenes | **0.9417** | 0.9417 | 0.9418 |
| stop3: zeros / at 1 | 0 / 34 | 0 / 34 | 0 / 34 |

- Audit vs builder: 0 of 48 scenes differ, for both images (bit-identical scores).
- Audit vs box: native 11 of 48 differ, max 0.0261 (`00083_00485-83fd1c19`), paired mean -0.0000, 47 within 0.01, zero / non-zero agree 48 / 48;
  stop3 14 differ, max 0.0261 (same scene), paired mean -0.0001, 47 within 0.01, 48 / 48 agree. Cross-host compiled-vs-interpreted numerics, no offset.
- Driver: 480 `drive` calls each, 0 inference errors, 0 tracebacks, 0 fail-open lines; server p50 / p90 / max 36.3 / 44.8 / 115.3 ms (native),
  36.8 / 48.4 / 147.8 ms (stop3). Native writes no `hold.jsonl`; stop3 writes 480 records.

**The stop3 public drop (0.9828 -> 0.9417) is by design, not a defect.** The image-vs-image comparison is clean: the two runs differ in exactly 4 scenes,
and all 4 are sessions the hold latched (`held`, ego below 0.3 m/s at the 0.5 s handoff); the other 44 scenes are bit-identical.

| scene | native | stop3 |
|---|--:|--:|
| `01100_01664-218adf8c` | 1.0000 | 0.6492 |
| `01100_01664-4ccc9e33` | 1.0000 | 0.2799 |
| `01690_02183-3debd3d8` | 0.9547 | 0.3408 |
| `01690_02183-b6968f15` | 1.0000 | 0.7150 |

`hold.jsonl`: 5 of 48 sessions held, 43 free; the fifth held session (`00083_00485-431ae299`) scores 1.0 in both. These are the same five sessions the
RSG stop3 image held on this log (RSG audit item 7, scores 0.66 / 0.28 / 0.34 / 0.74 / 1.0), and the drop has the same size (RSG 0.959 -> 0.918). In
them the ego starts at rest and the human drives off; the 3 m leash holds the ego until a release condition, so the score falls to the progress ratio.
The hold buys the private board's human-static windows (arena held: JTN-L18-s1 0.891 -> 0.943, static stratum 0.847 -> 0.997) and pays in launches
(arena launch stratum 0.856 -> 0.739); the public log, at 0.4 % static windows, shows only the cost. This is exactly the trade the A/B is meant to measure.

## 7. Push / submit commands for `ujs-iv`: FIXED (record corrected)

Checked against `~/bin/ap`, the CLI source (`/data/third_party/alpasim-cli/.../alpasim_challenge.py`) and earlier submissions.

Correct in the builder's block: `~/bin/ap <profile> <cli args>` exports `ALPASIM_CONFIG_DIR` and `DOCKER_CONFIG=<profile>/docker`; `ap <p> docker push` uses
the profile's ECR login; `submit` needs `--track` from `pai|nuplan|pai-warmup|nuplan-warmup` and runs `docker manifest inspect` under the same profile's
login before posting; `latest` is refused; `status <id>` / `submissions --track nuplan` exist. The profile `~/.alpasim/profiles/ujs-iv/` exists with a
token (expired now) and an **empty** `docker/` dir (this profile has never done `ecr-login`).

Defects fixed in [image_jtn_s1.md](image_jtn_s1.md):

1. **Team repository hard-coded from a guess** (`REPO=.../teams/ujs-iv  # replace with ...`): now captured from the `Team repository:` line `ecr-login`
   prints, with a stop if it is empty. Evidence that it will be `teams/ujs-iv`: the Tokyo store holds `teams/ujs-iv:rsg-l18-s0-acc-v1` with RepoDigest
   `teams/ujs-iv@sha256:9b19cc9d...`, i.e. a successful push on 2026-10-16 (the global `~/.docker` ECR login made right after the account switch,
   `~/.alpasim/challenge.json` 18:27 JST, before the per-profile setup of 10-17). Teams seen in records: `lao-siji` (nuPlan submissions),
   `vennai-org` (acct3, PAI probe). The token cannot be read off without a fresh sign-in, so the printed line stays authoritative.
2. **Local image selected by name** (`docker tag "$IMG"` with `IMG=jev-alpasim:...`): now tagged from the image id, so a re-tagged or rebuilt name cannot
   be pushed by mistake; after the push the remote manifest is resolved and the local `{{.Id}}` and RepoDigest printed for a check.
3. **Unbounded `until ... push; do sleep 10; done`** (spins forever on an expired ECR login): 5 bounded tries.
4. Warm-ups moved to an optional comment; `status <submission_id>` (not valid shell) -> `status SUBMISSION_ID`. The block passes `bash -n` on Tokyo and
   the `Team repository:` parse was tested on the CLI's output format.

Submission stays by tag (`$REPO:jtn-l18-s1-native`, `$REPO:jtn-l18-s1-holdstop3`), the path of every earlier successful submission; the board labels a
submission by its tag. The CLI would also take `$REPO@sha256:<digest>`.

Final commands (Tokyo, user; the record's block adds a stop line if `REPO` is empty):

```bash
P=ujs-iv
~/bin/ap $P me                                        # if expired: ~/bin/ap $P auth-url, sign in, ~/bin/ap $P configure-token
~/bin/ap $P limits                                    # >= 2 official slots left
REPO=$(~/bin/ap $P ecr-login | tee /dev/stderr | sed -n 's/^Team repository: //p'); echo "REPO=$REPO"   # expect .../teams/ujs-iv
push() {
  docker image inspect "$1" >/dev/null || return 1
  docker tag "$1" "$REPO:$2" || return 1
  for i in 1 2 3 4 5; do ~/bin/ap $P docker push "$REPO:$2" && break; [[ $i == 5 ]] && return 1; sleep 15; done
  ~/bin/ap $P docker manifest inspect "$REPO:$2" >/dev/null && docker image inspect "$REPO:$2" --format '{{.Id}} {{join .RepoDigests " "}}'
}
push sha256:1b68246103ec1fd106482eeec04c3e9b9fddb11a9f896148bde7b1a8f03dea22 jtn-l18-s1-native
~/bin/ap $P submit --track nuplan "$REPO:jtn-l18-s1-native"
push sha256:b547cca3bbec62899785c618eea9b9d0241de88a032a2b2aa02f523b5f8d2440 jtn-l18-s1-holdstop3
~/bin/ap $P submit --track nuplan "$REPO:jtn-l18-s1-holdstop3"
```

## 8. Quota

Not readable without a sign-in (`ap ujs-iv me` / `submissions` answer "missing or expired session"). The public boards (readable with the expired token,
2026-10-19) list no `ujs-iv` team on either track: nuPlan 128 succeeded submissions / 39 teams (ours: `lao-siji` x2, 10-11), PAI 103 / 32 (ours:
`vennai-org` x1, 10-17). The boards show only SUCCEEDED records, so a failed or running `ujs-iv` submission would be invisible. If none exists, the account
has 3 official slots this UTC month (shared nuPlan / PAI) and this A/B uses 2 of them. `~/bin/ap ujs-iv limits` after sign-in is the first command of the block.

Last verified: 2026-10-19
