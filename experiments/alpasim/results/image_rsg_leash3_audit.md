# Independent audit: nuPlan image RSG-L18-s0 + `JEV_HOLD=leash3` (2026-10-18)

Image `jev-alpasim:rsg-l18-s0-e7db399e-holdleash3`, id `sha256:394b113747aeb834830980467d0b784cef5decf1868e98234ac59e0c8e3d03a7`
(Tokyo Docker store), the candidate of [decision 350](../../../research/decisions/350.md); builder record [image_rsg_leash3.md](image_rsg_leash3.md).
Every check below was rerun by the auditor on the image (by id) or on git, not read off the builder's record. Nothing was pushed, nobody
logged in to a registry, the challenge CLI was not run, the image was not modified. Auditor's run dir on Tokyo:
`/data/runs/alpasim/tokyo_image/RSG-L18-s0-holdleash3-audit/{test,smoke}`.

**Verdict: GO.** All nine items pass. No rebuild is needed; two small code changes are recommended for the *next* image (items 8, 9), not this one.

| # | item | verdict |
|---|---|---|
| 1 | Layers | PASS |
| 2 | Checkpoint | PASS |
| 3 | Driver path | PASS |
| 4 | Code drift | PASS |
| 5 | Start-up and limits | PASS |
| 6 | Secrets | PASS |
| 7 | GPU rerun (test.sh + 48-scene smoke) | PASS |
| 8 | Hold semantics / edge cases | PASS (one non-blocking note) |
| 9 | Version string | PASS, no rebuild (fix in code for the next image) |

## 1. Layers: PASS

18 rootfs layers. Compared by diff id (`docker image inspect .RootFS.Layers`):

| layers | vs same build without HOLD (`rsg-l18-s0-e7db399e`, `6823c1a5`) | vs 10-17 RSG (`rsg-l18-s0-d47f2692`, `f99095ea`) | vs last submitted nuPlan (`yr10m10-f-s0-073dea33`, `3a00b88c`) |
|---|---|---|---|
| 1-12 (Python base, wheels, alpasim_grpc, models, alpasim LICENSE / e2e_challenge) | equal | equal | equal |
| 13-18 (jev-drive code 540 kB, serve files 20 kB, user 4 kB, ckpt 150 MB, ckpt-extra 77.6 MB, verify 69 B) | equal | differ | differ |

- Against the unset build: all 18 layers equal; `docker history` differs only in `ARG HOLD=leash3` / `ENV JEV_HOLD=leash3` (vs empty), both
  metadata-only. Config: ENV identical except `JEV_HOLD=leash3`; CMD `python /app/serve.py`, no entrypoint, user 10001:10001, port 6789,
  STOPSIGNAL SIGTERM, workdir /app, no labels: identical.
- Against d47f2692: layers 13-18 differ. Explained by file content (sha256 of every file under /app except the unchanged models layer, 46 vs
  44 files): new `sh30_hold.py`, `static_hold.py`; changed `sh30_driver.py`, `sh30_core.py`, `ap2_driver.py`, `pai_driver.py`, `rmem.py`,
  `serve.py`, `verify.py`; everything else (checkpoints, route_mem.json, tags.json, probe.py, pai_spec.json, requirements.lock, alpasim
  files, the other closure files) byte-equal. The ckpt / user / verify layers carry equal content; their diff ids change because BuildKit
  rebuilt them on top of a changed parent (the code layer), as in every earlier image. ENV vs d47f2692: `SH30_GIT_HASH` d47f2692 -> e7db399e
  and the two new `JEV_GAP=` (empty = off) / `JEV_HOLD=leash3`; nothing else.
- Against YR10m10 (last submitted nuPlan image): same 12 lower layers; ENV differs by `JEV_TAG` YR10m10-F-s0 -> RSG-L18-s0, `JEV_DRIVER`
  auto -> sh30, hash, and the later switches `JEV_ACC=0 JEV_VHIST= JEV_GAP= JEV_HOLD=leash3`, all off except HOLD.

## 2. Checkpoint: PASS

In the image: `RSG-L18-s0/ckpt-final.pt` sha256 `fbe7ac9ff0d32ee6636369e6fcac248bf517dff9db528c5be689d18a04f05143`, `route_mem.json`
`672ef154639ab9201093d2af54b0570695cd1dd1d19c313b1a968b02ad394ade`. Equal to `/home/ujs/data/runs/op_parity/runs/RSG-L18-s0/` on the AutoDL
box and `/data/runs/op_parity/runs/RSG-L18-s0/` on Tokyo. `tags.json` maps RSG-L18-s0 -> sh30. Driver log: "serving the ROUTE-MEM branch,
memory on", "RSG-L18-s0 (backwarp, ... compiled False)".

## 3. Driver path: PASS

- `docker run` of the default CMD prints `serve: driver sh30 (sh30_hold.py), tag RSG-L18-s0, ...; switches JEV_VCONT=0 JEV_LEAD=0 JEV_VHIST=
  JEV_ACC=0 JEV_GAP= JEV_HOLD=leash3` and execs `/app/jev-drive/experiments/alpasim/lib/sh30_hold.py` (the CPU traceback below starts in
  `sh30_hold.py` line 68 = `D.main()`, i.e. after `install`). The unset image prints `driver sh30` and runs `sh30_driver.py`.
- Inside the image `static_hold` parses `mode leash 3.0`, latch 0.3 m/s at the 0.5 s call; after `install`, `sh30_driver.make_cached_plan` is
  the wrapper from `sh30_hold`, `JEV_TRACK` off (the wrapper asserts it).
- The served files (`sh30_hold.py`, `static_hold.py`, `sh30_driver.py` and the whole 31-file closure) are byte-equal to `89e5e660`, the code the
  arena-hold reads ran (`git show 89e5e660:<path>`). Only `serve.py` / `verify.py` differ from 89e5e660 (the e7db399e entry dispatch; not in
  the box's `run.sh sh30hold` path, which calls `sh30_hold.py` directly with the same environment). `git diff 89e5e660 e7db399e` over
  `experiments/alpasim/{lib,scripts}`, `jevdrive`, `lib`, `op_parity`, `route_mem`, `op_adapt_*`: empty.
- `sh30_driver.py` is unchanged by the hold: the wrapper patches `Driver.drive`, `Core.plan` and `make_cached_plan` at run time only.
- In my GPU test the hold was live: `/tmp/alpasim-driver/hold.jsonl` 17 kB next to `drive.jsonl`.

## 4. Code drift: PASS

Each of the 35 baked code / serve files (the 31 closure files + serve.py, verify.py, probe.py, pai_spec.json) was hashed against `git show <rev>:<path>`:

| revision | match |
|---|---|
| e7db399e (the build commit, `SH30_GIT_HASH`) | 35 / 35 |
| 89e5e660 (arena reads) | 33 / 35 (`serve.py`, `verify.py`) |
| 7b709626 / HEAD / Mac working tree | 33 / 35: `static_hold.py` and `sh30_hold.py` differ |

The `stop<m>` mode is no longer uncommitted: it went into 7b709626 (ARENA-HOLD supplement 2), after e7db399e. The image's two hold files are
the pre-stop versions (equal to 89e5e660 / e7db399e, not to 7b709626 / HEAD), and the image's `static_hold` rejects `JEV_HOLD=stop3`
(its assert allows off | full | lead | leash). The build clone `/data/runs/alpasim/tokyo_image/code-rsg-leash3` is at e7db399e and clean.

## 5. Start-up and limits: PASS

Limits from `e2e_challenge/README.md` at 0bb4c4b: image <= 40 GiB, <= 16 GiB VRAM per driver instance, no outbound network, read-only root,
scratch `/tmp` 2 GiB and `/run` 64 MiB, GPU provided; ~0.1 s per `Drive` call target.

- Size 8 105 635 216 B (7.55 GiB).
- CPU only (no GPU, `--network none --read-only`): `verify.py` loads the ROUTE-MEM branch and stops only at the build-time `tags.json` write,
  as on any read-only root; `import static_hold, sh30_hold, sh30_driver` and `sh30_hold.install` work. The default CMD without a GPU exits
  with "Found no NVIDIA driver" after the serve line, exactly as the 10-17 RSG image does: the sh30 family needs the GPU the challenge
  provides; not a regression.
- Own `test.sh` on card 1 (organisers' hardening flags, `--network none`, 2 sessions x 8 rollouts, 160 `drive` calls): pass. Listening 17.7 s after `docker run`; GPU driver peak 1 603 MiB
  (limit 16 GiB; 16 replicas on 4 cards = 4 per card); `/tmp` 0.39 MiB of 2 GiB, `/run` 0; 0 read-only errors, 0 files changed outside the
  tmpfs (the `docker diff` lines are the NVIDIA runtime's injected driver files); 0 tracebacks; `drive` p50 / p90 / p99 / max 50.6 / 53.0 / 68.8 /
  69.2 ms client side (policy p50 7.2 ms); 0 inference errors; SIGTERM -> exit 0 in 0.8 s.

## 6. Secrets: PASS

`docker history --no-trunc`: no token, password, proxy, credential URL or host address; the only URL is the public PyTorch index in the
(offline, `--no-index`) uv install line, the only "key" the Python base's public `GPG_KEY` id. Filesystem (as root, whole root fs minus
/proc, /sys): no `.env`, `.netrc`, `.git-credentials`, ssh keys, `*.key`, `.git`, `.ssh`, `.docker`, `.alpasim`, `challenge.json`,
`credentials*`, `.pypirc`; `*.pem` only the public CA bundle. No `hf_` / `AKIA` / `ghp_` / `github_pat_` / `nvapi-` / `sk-` / private key /
`user:pass@` URL / proxy port 7890 / Tokyo or AutoDL address / `github-venn` / user e-mail in `/app /etc /root /home /tmp /usr/local/bin /opt`.
Container env has no token / secret / proxy variable.

## 7. GPU rerun: PASS

Both on the exact image id, card 1 (driver) / card 0 (render), `/data/runs/alpasim/tokyo_image/RSG-L18-s0-holdleash3-audit/`.

- `test.sh`: pass (numbers in item 5); probe reads version `jev-RSG-L18-s0-backwarp`, git `e7db399e`, API 0.54.0; `hold.jsonl` written.
- `smoke.sh`, 48 public scenes in the official containerised stack (alpasim-base:0.89.0, `+e2e_challenge_nuplan=full`, the same wizard
  overrides as the builder's; same render-arch deviation sm_86), wall 205 s; compared with `compare.py --ref` against the native box read of
  the same hold (pool job `1018-010637-bd56`, `native_ref/RSG-L18-s0-holdleash3`):

| | audit run | builder run | native leash3 (box) |
|---|--:|--:|--:|
| mean, 48 scenes | **0.9184** | 0.9184 | 0.9180 |
| zeros / at 1 | 1 / 29 | 1 / 29 | 1 / 29 |
| zero scene | `...30f09e001cf55013` | same | same |

  Audit vs builder: 0 of 48 scenes differ (bit-identical scores). Audit vs native: 18 differ, max |diff| 0.0104, paired mean +0.0005, 47
  within 0.01, zero / non-zero and at-1 sets agree 48 / 48. Driver: 480 `drive` calls, 0 inference / input errors, server p50 / p90 / p99 / max
  36.8 / 48.5 / 103.8 / 139.8 ms, 0 tracebacks, GPU peak 1 761 MiB (card), `/tmp` peak 1.6 MiB.
- Hold behaviour, from the run's own `hold.jsonl`: all 48 sessions decide at t = 0.5 s; 5 held (`218adf8c...450058eb`, `...47e95c26`,
  `b6968f15...4bfb5a5d`, `3debd3d8...6b5850dd`, `4ccc9e33...aa795ef1`), 43 free; latch state equal to the builder's run and to the native box
  read in 48 / 48 sessions, and the held scenes score identically to native (1.0 / 0.661 / 0.280 / 0.342 / 0.738). The public cost the builder
  reports (0.959 plain RSG -> 0.918) is these 4 launch scenes; it is the cost decision 350 accepts.

## 8. Hold semantics: PASS (one non-blocking note)

Edge cases run inside the image (its own Python and modules, CPU, `--read-only --network none`; scripts in `tmp/aud_hold/` on the
Mac, not committed), 16 / 16 pass:

| case | result |
|---|---|
| very first decision | never latches (t = 0 < 0.5 s); `pre`, plan returned as the same object |
| first call only, then 1 s later | latches at the 2nd call, not the 1st |
| session that never latches (moving at the handoff) | 20 decisions, output `is` the model's plan every time (bit for bit) |
| single-call session | `pre`, no error |
| leash: ego 0 / 1 / 2.5 / 3 / 3.5 / 10 / 100 m from the latch point | `s_lim` = max(0, 3 - d) >= 0; served arc <= `s_lim` (max excess 0) |
| reverse plan (24 m backwards) while held | cut to 3.000 m arc; no error |
| negative speed | unreachable in serving (`sh30_hold` passes `hypot(vx, vy)` >= 0); if passed, `v < 0.3` latches, no error |
| NaN / +inf speed at the handoff | `free`, plan passed through (fail open) |
| NaN ego position at the latch | `s_lim` 0 for the session (stays stopped; finite output). Only reachable if the simulator sends a NaN pose, which breaks plain RSG too |
| NaN / inf plan while held | no exception; the non-finite rows pass through as they would in plain RSG |
| all-zero / repeated / float64 plans | shape and dtype kept |
| fuzz: 20 000 random plans x random `s_lim` | no exception; served arc <= `s_lim` + 2.5e-7 m; poses inside `s_lim` unchanged bit for bit |
| non-monotone clock | no error |
| `hold.jsonl` in a missing nested directory (the 89e5e660 fix) | directory created, file opened |
| unwritable log directory | `OSError` at start-up; `sh30_driver.Driver` itself opens `drive.jsonl` in the same directory and fails identically, so no new failure mode. In the challenge the dir is `/tmp/alpasim-driver` on the 2 GiB tmpfs (`ALPASIM_DRIVER_LOG_DIR`), shown writable by test.sh / smoke |

Note (non-blocking): the hold's `make_cached_plan` wrapper is outside `sh30_driver.drive`'s inference `try`, so an exception in
`static_hold.step` would fail that `Drive` call (gRPC error) instead of falling back to the model's plan. The fuzz above found no input that
raises: the step is pure numpy on the driver's own finite values, and its state is per session (the state dict is created under a lock). Fail-open by construction
would need a `try: ... except: return make0(t0, anchor, poses)` in `sh30_hold.make_cached_plan`: worth adding in code for the next image, not a
reason to rebuild this one. `hold.jsonl` grows ~200 B per decision on the 2 GiB `/tmp` (17 kB per 160 decisions here): no risk.

## 9. Version string: PASS, no rebuild

The served `VersionId.version_id` is `jev-RSG-L18-s0-backwarp`, `git_hash` `e7db399e`. `static_hold.SUFFIX` is `-holdleash3`, but
`sh30_driver.get_version` does not include it and `sh30_hold` does not patch it, so the version alone cannot tell this image from plain RSG.

Is that a problem? Not for tracking this submission: (a) no plain RSG-L18-s0 nuPlan image was ever submitted (the two nuPlan submissions are
APY10m10-AB-s0 and YR10m10-F-s0), so the string collides with nothing on the board; (b) the pair (`version_id`, `git_hash`) is unique among
pushed images (the only other e7db399e build is the unset local image, never to be pushed); (c) the submission record ties id -> pushed URI
(tag `...:rsg-l18-s0-holdleash3-v1`) -> registry digest -> this image id, which is the authoritative mapping. The organisers do not use the
version for scoring. A rebuild would change the image (code layer + all layers above), need a new test / smoke / audit, and buy only a
nicer string. Recommendation: push this image as is; record the digest and `JEV_HOLD=leash3` in the submission record; for the next image,
make `sh30_hold` append `HD.SUFFIX` to `get_version` (one line), together with the fail-open `try` of item 8.
