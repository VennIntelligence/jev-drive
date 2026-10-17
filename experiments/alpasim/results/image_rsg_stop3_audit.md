# Independent audit: nuPlan image RSG-L18-s0 + `JEV_HOLD=stop3` (2026-10-18)

Image `jev-alpasim:rsg-l18-s0-ba3ef44d-holdstop3`, id `sha256:51401086aaed3ad95f4781428e513559b7ceb4dcbf549aee85a541536389d4e2`
(Tokyo Docker store). It is the candidate of [decision 351](../../../research/decisions/351.md). The builder's record is
[image_rsg_stop3.md](image_rsg_stop3.md), and the method follows the leash3 audit ([image_rsg_leash3_audit.md](image_rsg_leash3_audit.md)).

The auditor reran every check on the image (by id) or on git; nothing was read off the builder's record. Nothing was pushed, nobody logged in
to a registry, the challenge CLI was not run and the image was not modified. No pool job was touched.

- Auditor's run dir on Tokyo: `/data/runs/alpasim/tokyo_image/RSG-L18-s0-holdstop3-audit/` (`test`, `smoke`, `parity`, `limits`,
  `edge_stop3.txt`, `parity_arena12_audit.md`, `parity_leash_audit.txt`).
- Probe script on the Mac (not committed): `tmp/aud_stop3/edge_stop3.py`.

**Verdict: GO.** All ten items pass, and no fix is needed before the push. Three non-blocking notes follow (items 8 and 9).

| # | item | verdict |
|---|---|---|
| 1 | Layers | PASS |
| 2 | Checkpoint | PASS |
| 3 | Served path | PASS |
| 4 | No drift | PASS |
| 5 | Limits | PASS |
| 6 | Secrets | PASS |
| 7 | GPU rerun (test.sh + 48-scene smoke) | PASS |
| 8 | Stop-rule parity on arena windows | PASS (one note) |
| 9 | Hold semantics and fail-open | PASS (two notes) |
| 10 | Version string | PASS |

## 1. Layers: PASS

The image has 18 rootfs layers. Compared by diff id with the audited leash3 image (`sha256:394b1137...`):

- **Layers 1-12 are equal:** Python base, wheels, alpasim_grpc, models, and alpasim LICENSE / e2e_challenge.
- **Layers 13-18 differ:** the jev-drive code, serve files, user, ckpt, ckpt-extra and verify layers.

**History and config.** `docker history --no-trunc` differs only in:

- `ARG/ENV HOLD` leash3 -> stop3;
- `ARG GIT_HASH` / `SH30_GIT_HASH` e7db399e -> ba3ef44d;
- the size of the jev-drive COPY, 540 -> 543 kB.

The config JSON differs only in `SH30_GIT_HASH` and `JEV_HOLD`. CMD, user 10001:10001, port, STOPSIGNAL and workdir are equal.

**File content.** Every file of both images was hashed: the whole root filesystem minus /proc, /sys and /dev, 27 882 files each, run as
root. The diff is:

- `experiments/alpasim/lib/sh30_hold.py` and `static_hold.py`: the two expected hold files;
- `/etc/hostname` and `/etc/mtab`: Docker injects these per container at run time, so they are not image content.

Checkpoints, `route_mem.json`, `tags.json`, `serve.py`, `verify.py`, `probe.py`, the closure, `/etc/passwd` and the rest are byte-equal.
Layers 15-18 carry equal content. Their diff ids change only because BuildKit rebuilt them on a changed parent, the code layer, as in
every earlier image.

## 2. Checkpoint: PASS

| file | sha256 in the image |
|---|---|
| `RSG-L18-s0/ckpt-final.pt` | `fbe7ac9ff0d32ee6636369e6fcac248bf517dff9db528c5be689d18a04f05143` |
| `route_mem.json` | `672ef154639ab9201093d2af54b0570695cd1dd1d19c313b1a968b02ad394ade` |

Both are equal to the copies on the AutoDL box (`~/data/runs/op_parity/runs/RSG-L18-s0/`) and on Tokyo (`/data/runs/op_parity/runs/RSG-L18-s0/`).
`tags.json` maps RSG-L18-s0 -> sh30. `verify.py`, run on CPU, reports `branch RSG-L18-s0 dim 200`.

## 3. Served path: PASS

- **Default CMD.** It prints `serve: driver sh30 (sh30_hold.py), tag RSG-L18-s0, ...; switches JEV_VCONT=0 JEV_LEAD=0 JEV_VHIST= JEV_ACC=0
  JEV_GAP= JEV_HOLD=stop3`. With no GPU, it then fails inside `sh30_hold.py` line 83, `D.main()`, which is after `install`.
- **Hold configuration in the image.** `static_hold` reads:
  - mode `stop`, CAP 3.0;
  - V_LATCH 0.3 m/s, T_HAND 0.5 s;
  - A_STOP 0.5 m/s^2, T_STOP 3 s;
  - SUFFIX `-holdstop3`; `JEV_TRACK` off.
- **Patches.** After `install`, four objects come from `sh30_hold`: `Driver.drive`, `Core.plan`, `make_cached_plan` and `Driver.get_version`.
- **Served files against git.**
  - `sh30_driver.py` is byte-equal to ba3ef44d and to 7b709626, so it is unchanged.
  - Against 7b709626, the code the arena stop3 reads ran, the only difference over the 36 baked files is `sh30_hold.py`.
  - `git diff 7b709626 ba3ef44d -- experiments/alpasim/lib/` is exactly the fail-open `try/except` around state, speed, `HD.step` and the
    log write in `make_cached_plan`, plus the `get_version` patch that appends `HD.SUFFIX`. I read it line by line; nothing else changed.
  - The other files changed between the two commits (`nr1_lat.py`, `rf_pait.py`, `rm_train.py`, `build.sh`, `Dockerfile`,
    `hold_check.py`) are not in the image.

## 4. No drift: PASS

The 36 baked code and serve files (the 31-file closure plus `serve.py`, `verify.py`, `probe.py`, `pai_spec.json` and `requirements.lock`)
were hashed against `git show <rev>:<path>`:

| revision | match |
|---|---|
| ba3ef44d (the build, `SH30_GIT_HASH`) | 36 / 36 |
| HEAD (3e69c223) and the Mac working tree | 36 / 36 |
| 7b709626 (arena stop3 reads) | 35 / 36 (`sh30_hold.py`) |
| e7db399e (leash3 image) | 34 / 36 (`sh30_hold.py`, `static_hold.py`) |

The build clone `/data/runs/alpasim/tokyo_image/code-rsg-stop3` is at ba3ef44d with 0 uncommitted files (`git describe --dirty` = `ba3ef44d`).

## 5. Limits: PASS

| limit | measured |
|---|---|
| Size | 8 105 637 984 B (7.55 GiB; limit 40 GiB) |
| Start-up (my test.sh, card 1) | listening 17.7 s after `docker run` |
| GPU memory | driver peak 1 603 MiB (limit 16 GiB) |
| Host RAM, separate run under the same probe load | peak 2.29 GiB of the 32 GiB `--memory` |
| Scratch | `/tmp` peak 0.40 MiB of 2 GiB (`hold.jsonl` 17 kB, `drive.jsonl` 363 kB); `/run` 0 |
| Read-only root | 0 errors; `docker diff` 0 lines, 0 files changed outside the tmpfs |
| `drive` latency | client p50 / p90 / p99 / max 50.5 / 52.1 / 60.0 / 69.9 ms (policy p50 7.2 ms); 0 inference errors, 0 tracebacks |
| SIGTERM | exit 0 in 0.6 s |
| CPU only (`--network none --read-only`) | imports and `install` work; `verify.py` loads the branch and stops only at the build-time `tags.json` write, as on any read-only root. The default CMD exits with "Found no NVIDIA driver", as every sh30 image does: the challenge provides the GPU, so this is not a regression |

## 6. Secrets: PASS

- **Filesystem, as root.** The leash3 audit's `secrets.sh` found no `.env`, `.netrc`, `.git-credentials`, ssh key, `*.key`, `.git`, `.ssh`,
  `.docker`, `.alpasim`, `challenge.json`, `credentials*` or `.pypirc`. The only `*.pem` files are the public CA bundle.
- **Content patterns.** No match under `/app /etc /root /home /tmp /usr/local/bin /opt` for `hf_`, `AKIA`, `ghp_`, `github_pat_`, `nvapi-`,
  `sk-`, private keys, `user:pass@` URLs, proxy port 7890, the Tokyo or AutoDL address, `github-venn` or the user e-mail.
- **Environment.** The only `*KEY*` variable is the Python base's public `GPG_KEY`.
- **History.** The only URLs are the public python.org and PyTorch index URLs. There is no token, password, proxy or host address.
- Item 1 also shows that the filesystem equals that of the already-audited leash3 image, apart from two git-tracked source files.

## 7. GPU rerun: PASS

Both runs used the exact image id: driver on card 1, render on card 0, the same wizard overrides and the same sm_86 render-arch deviation as
the builder.

**test.sh** passes; the numbers are in item 5. The probe reads `jev-RSG-L18-s0-backwarp-holdstop3`, git `ba3ef44d`, API 0.54.0, and
`hold.jsonl` is written.

**48-scene smoke** (official containerised stack, alpasim-base:0.89.0). The native reference is the box read of the same hold at ba3ef44d
(`native_ref/RSG-L18-s0-holdstop3`).

| | audit run | builder run | native stop3 (box) |
|---|--:|--:|--:|
| mean, 48 scenes | **0.9184** | 0.9184 | 0.9180 |
| zeros / at 1 | 1 / 29 | 1 / 29 | 1 / 29 |
| zero scene | `...30f09e001cf55013` | same | same |

- **Audit against builder:** 0 of 48 scenes differ (bit-identical scores). The audit run also equals the leash3 audit smoke in 48 / 48 scenes.
- **Audit against native:**
  - 18 scenes differ, max |diff| 0.0104, paired mean +0.0005;
  - 47 are within 0.01;
  - zero / non-zero agree 48 / 48.
- **Driver:** 480 `drive` calls, 0 inference errors, server p50 / p90 / p99 / max 36.3 / 45.6 / 77.6 / 130.5 ms, 0 tracebacks. The fail-open
  line appears 0 times in the test, smoke and parity driver logs.
- **Hold behaviour on the public 48:**
  - 5 sessions hold at below 0.3 m/s (`218adf8c`, `3debd3d8`, `431ae299`, `4ccc9e33`, `b6968f15`), 43 are free, and none stop-latches;
  - latch state equals the builder's run and the native read in 48 / 48 sessions;
  - the held scenes score as on native (1.0 / 0.6615 / 0.2804 / 0.3417 / 0.7376).

  So the public board should read exactly as the leash3 image would. The stop rule's gain is only visible on held windows like the arena's.

## 8. Stop-rule parity on arena windows: PASS

I chose 12 windows myself from the box stop3 read's shard-1 chunks c3-c6 (`box_ref/c{3..6}/run`, 33 stop latches there). **None overlaps
the builder's 12.**

| kind | windows | why chosen |
|---|---|---|
| stop latch | 7 | the three largest leashes in the read (9.86, 9.05 and 8.31 m); the three weakest decelerations just past the 0.5 m/s^2 line (-0.515, -0.546, -0.557, the last at 0.309 m/s); one at -0.798 |
| static latch | 3 | random |
| free | 2 | random |

All 12 were run through the image in the containerised stack, using the shard-1 cache. The comparison used `hold_parity.py` and an
independent check of the stop fields and the leash per decision (`parity_leash_audit.txt`).

| window | kind | latch box = image (a m/s^2, stop distance m, cap m) | max \|Δ s_lim\| / \|Δ d0\| (m) | max (d0 - cap) image (m) | score box / image |
|---|---|---|--:|--:|--:|
| `...df4e53377e485114` | stop | -1.559, 6.857, 9.857 | 0.036 / 0.036 | -0.56 | 1.0000 / 1.0000 |
| `...ee701f01ee2c588f` | stop | -1.398, 6.049, 9.049 | 0.014 / 0.014 | -1.18 | 0.9821 / 0.9811 |
| `...a6efc718ee815492` | stop | -1.483, 5.310, 8.310 | 0.014 / 0.014 | -2.36 | 0.9451 / 0.9476 |
| `...567608d576035c01` | stop | -0.515, 0.170, 3.170 | 0.002 / 0.003 | -2.91 | 1.0000 / 1.0000 |
| `...e2910aa7298e5eb5` | stop | -0.546, 0.249, 3.249 | 0.014 / 0.015 | -2.55 | 1.0000 / 1.0000 |
| `...bd783fb1c9065de0` | stop | -0.557, 0.086, 3.086 | 0.003 / 0.003 | -2.20 | 1.0000 / 1.0000 |
| `...6aa3dcbb7fd65c44` | stop | -0.798, 3.273, 6.273 | 0.006 / 0.007 | -3.34 | 1.0000 / 1.0000 |
| `...00046052dcf7537e` | static | held at 0.001 m/s, cap 3 | 0.008 / 0.008 | -1.46 | 1.0000 / 1.0000 |
| `...a68371d746bc53f1` | static | held at 0.042 m/s, cap 3 | 0.010 / 0.010 | -0.64 | 1.0000 / 1.0000 |
| `...fafc6df6c07e5bee` | static | held at 0.0 m/s, cap 3 | 0.010 / 0.010 | -0.17 | 1.0000 / 1.0000 |
| `...e138c568ffe65d47` | free | free at 3.337 m/s | - | - | 0.0000 / 0.0000 |
| `...cb5794a6a3cc5724` | free | free at 9.596 m/s | - | - | 1.0000 / 1.0000 |

- **Latch events: 12 / 12 equal.** The state, the call time (0.5 s) and the measured speed match. The stop fields `stop_a`, `stop_s` and
  `cap` are equal to the logged precision in all 7 stop latches.
- **Hold state and cut flag:** 120 / 120 equal.
- **Leash:** over the 90 held decisions, `s_lim` and the distance from the latch point agree within 3.6 cm. No window gets past its leash:
  in the image, the distance from the latch point never exceeds the cap (the largest margin is 0.17 m short).
- **Served plans:** at decision 0, median 0.8 cm and max 2.5 cm. Over all decisions, median 1.6 cm, p90 4.5 cm and max 14 cm.
- **Scores:** 10 / 12 identical, 12 / 12 within 0.01, mean 0.9106 box / 0.9107 image, and the same zero (`e138c568`, a free window that
  is also zero on the box).

The plans are cm-level rather than bit-identical because the image runs interpreted passes on an RTX 3090, while the box ran compiled
passes on an RTX 6000D, as for every image (decision 215). The builder's own 12 windows, read independently from the same box logs, show
the same agreement ([parity_arena12.md](image_rsg_stop3/parity_arena12.md)), so 24 arena windows in all, 12 of them stop latches, agree.

Note (non-blocking): the `hold_parity.py` in the build clone (ba3ef44d) predates the stop-latch column (added in 3e69c223, a script that is
not in the image). The stop fields above therefore come from my own check.

## 9. Hold semantics and fail-open: PASS (two notes)

The probe ran inside the image with its own Python and modules: CPU only, `--read-only --network none --cap-drop ALL`, the baked
`JEV_HOLD=stop3`, and serving warning settings. The real `sh30_hold.install` was used, with `make_cached_plan`'s original replaced by a
recorder. Result: **15 / 15 pass, 0 numpy warnings** (`edge_stop3.txt`).

| case | result |
|---|---|
| First decision | t = 0 and t = 0.499 s are `pre`, and the plan object is returned; a single-call session is `pre` with no error |
| Session that never latches | cruise, accelerating, braking at a = -0.4, braking at a = -0.6 with v/\|a\| = 8.3 s, and a = -0.5 exactly (a strict `<`): `free`, and the model's plan object is served at 20 / 20 decisions |
| Stop latch semantics | 2 -> 1 m/s in 0.5 s gives held, a = -2, cap 3.25, served arc 3.2500 m. v/\|a\| = 3.0 s latches (`<=`); 3.1 s stays free. Walking away keeps served arc <= s_lim = max(0, cap - d), excess 0. A latch at 0.29 m/s is the static leash3 latch (cap 3, no stop fields) |
| NaN / ±inf speed at the handoff and at the call before (NaN / inf a), 25 combinations | no exception, finite outputs. A NaN or +inf handoff speed is `free` (identity). Every held session has a finite cap >= 3. In serving, v is `hypot(vx, vy)`, which is >= 0 or NaN, so -inf cannot occur |
| Zero / negative deceleration | constant speed, accelerating and starting from 0 are all `free`. A previous call 0.5 ms before the handoff (a would be -2000) is skipped by the 1 ms dt guard (`free`) |
| Enormous predicted stop distance | v 1e3, 1e6, 1e200 and 1e300 m/s: no exception, and `json.dumps` of the record works. The cap becomes 128 m, 1.25e5 m, then inf (float overflow). With an inf cap, s_lim is inf and the plan passes uncut |
| Negative predicted stop distance | unreachable: a < -0.5 means -2a > 0. Fuzzing 200 000 random (v_prev, v, dt) gave 73 773 stop latches and 0 caps below 3. A cap forced to -7 or NaN clamps s_lim to 0 (stand still), with no error |
| `limit_arc` fuzz, 20 000 random plans, s_lim up to 12 m | served arc <= s_lim + 5e-7 m; poses inside s_lim unchanged bit for bit |
| Missing log directory | `install` creates a missing nested `/tmp/.../a/b/c` and opens `hold.jsonl`. An unwritable directory (`/app/nope` on the read-only root) raises `OSError` at start-up. `sh30_driver.Driver` opens `drive.jsonl` in the same directory and would fail the same way, so this is no new failure mode. In the challenge the directory is `/tmp/alpasim-driver` on the 2 GiB tmpfs, and items 5 and 7 show it is writable |
| Forced exception in `static_hold.step` | 20 / 20 calls hand the **model's own plan object** to `make_cached_plan`, one log line is written per process, and the next session works normally |
| Exceptions elsewhere in the hold path | 4 / 4 hand on the model's plan: `velocity()` raising, `KeyError` from the session states, a malformed scene, and the real `step` raising on a malformed (1-D) plan while held |
| `hold.jsonl` write failing mid-run (file closed) | a held call still serves the hold's cut plan (arc 3.250 m); an unheld call serves the model's plan |
| `get_version` | `jev-RSG-L18-s0-backwarp-holdstop3`, git `ba3ef44d` |

Notes (non-blocking):

- **The fail-open log line is written once per process.** After the first exception, later exceptions of any kind are silent. Their only
  trace is the missing `hold.jsonl` record for that decision. This is acceptable for a submission, where nothing reads the log. If the hold
  is ever used for a diagnosis run, count the failures rather than log only the first.
- **A +inf speed at the call before the handoff latches a 3 m leash** when the handoff speed is finite (a = -inf, stop distance 0). This
  needs the simulator to report an infinite ego velocity, which would also break plain RSG's inputs. It is not reachable in practice and
  the output is finite.

## 10. Version string: PASS

The served `VersionId` reads `version_id` **`jev-RSG-L18-s0-backwarp-holdstop3`**, `git_hash` `ba3ef44d`, API 0.54.0. This was measured
twice: in the probe of my test.sh run, and in a separate RAM-measurement run. The direct `Driver.get_version` call in item 9 gives the
same string. The leash3 audit's item 9 recommendation is implemented: the suffix comes from `static_hold.SUFFIX` and is appended by the
`sh30_hold` patch.

Last verified: 2026-10-18
