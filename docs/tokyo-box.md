# Tokyo box: the box with eyes

Read this when you need to **see** CARLA rather than measure it: a window on a real monitor, a
manual drive, a screenshot or a screen recording of a route. It also hosts the explicitly
scoped controller diagnostics below. Keep their single-3090, windowed measurements separate
from GPU-box model experiments ([remote-box.md](remote-box.md)).

## What it is

- Login: `ssh ujs@100.108.238.8` (Tailscale peer `tokyo-devbox`). Only the user `ujs` is accepted;
  `gaochengzhi1999`, `snail` and `root` are all refused. No `~/.ssh/config` alias yet.
- The name says Tokyo, the machine is in **Shanghai** (China Mobile). Its clock runs **JST
  (+0900)**, so every timestamp in a log there is one hour ahead of this Mac.
- It has a physical monitor and a real desktop session, which is the whole point of it.

| | |
|---|---|
| OS / kernel | Ubuntu 24.04.3, currently booted on 7.0.0-31-generic |
| CPU / RAM | Ryzen 9 9950X, 16 cores / 32 threads; 60 GB |
| GPU | One RTX 3090 24 GB, driver 580.173.02, PCI `02:00.0` |
| Disks | `/` 1.8 TB (1.4 TB free), `/data` 3.6 TB (nearly empty) |
| Desktop | GNOME Wayland + Xwayland on `:0`, seat0, 3840x2160 |
| Code | `~/mycode/jev-drive` (clean clone, `origin` is plain `github.com`) |
| Data | `/data` — treat it as `DATA_DIR`, same layout as the GPU box |

## GPU selection (measured 2026-09-23)

`nvidia-smi` reports exactly one NVIDIA GeForce RTX 3090 at index 0, UUID
`GPU-b90dd90e-394b-7800-f23f-5892a8e3d0f1`, PCI `0000:02:00.0`. PyTorch reports one device,
`torch.cuda.device_count() == 1`, named `NVIDIA GeForce RTX 3090`, with no
`CUDA_VISIBLE_DEVICES` set. Use CUDA index 0 directly.

CARLA ignores `CUDA_VISIBLE_DEVICES` and takes `-graphicsadapter=<rank>`, a Vulkan physical-device
index. Two headless CARLA starts with the NVIDIA Vulkan ICD both appeared on that same UUID:

| CARLA flag | observed `nvidia-smi` device | PCI |
|---|---|---|
| `-graphicsadapter=0` | index 0, RTX 3090 | `0000:02:00.0` |
| `-graphicsadapter=1` | index 0, RTX 3090 | `0000:02:00.0` |

The box now exposes only one NVIDIA card; the old second-card rank distinction no longer applies.
Use rank 0. This is a one-card mapping;
recheck the process UUID after driver or Vulkan changes with
`nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv` and
`nvidia-smi --query-gpu=index,uuid,pci.bus_id --format=csv`.

The former two-card notes described the removed 580.159.03 setup. They no longer describe this host.
The running kernel module and userspace both report 580.173.02; ordinary `nvidia-smi` and PyTorch
work. The old private userspace directory is retained for its historical evidence only and must not
be sourced by current launch scripts. See [the recovery record](../todos/2026-09-23-lateral-followup/diagnostics/driver-recovery/README.md).

## Looking at CARLA

X access uses `DISPLAY=:0`. In the current GNOME Wayland session, set `XAUTHORITY` to the file
passed to the running Xwayland process with `-auth`; `~/.Xauthority` is stale and fails here. Do
not enable broad `xhost` access.

```bash
ssh ujs@100.108.238.8
cd /data/third_party/carla/CARLA_0.9.15
export DISPLAY=:0
export XAUTHORITY="$(ps -C Xwayland -o args= | sed -n 's/.*-auth \([^ ]*\).*/\1/p' | head -1)"
DISPLAY=:0 VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  ./CarlaUE4.sh -windowed -ResX=1280 -ResY=720 -nosound \
                -carla-rpc-port=3000 -quality-level=Epic -graphicsadapter=0
```

A `CarlaUE4` window appears on the physical monitor and renders at once; the spectator camera is
driven with the mouse and WASD on that machine. Everything [carla.md](carla.md) says about CARLA
still applies here — silence is not a hang, an open RPC port is not a ready server, never
`pgrep -f` a pattern that matches your own command line (it matched three times while this page was
being written).

**Getting the picture back to the Mac.** ImageMagick's `import` and `xwd` are installed;
`scrot`, `gnome-screenshot`, `xdotool` and `wmctrl` are not.

```bash
# on the box
DISPLAY=:0 import -window root -resize 1400x /tmp/shot.png
# from the Mac
scp ujs@100.108.238.8:/tmp/shot.png .
```

`ffmpeg` and `x11vnc` are both installed, so a route can be recorded
(`DISPLAY=:0 ffmpeg -f x11grab -framerate 20 -i :0 out.mp4`) or watched live over VNC. Neither has
been used in anger yet.

## Everything long runs in tmux

Same rule as the GPU box ([long-runs.md](long-runs.md)), and here it has a second purpose: the
person sitting at the monitor sees the same pane you do. Sessions in use: `jev` for project work,
`dl` for downloads and installs.

## Network: Clash in global mode, and nodes that decay

Clash Verge runs on the desktop: mixed port **7890**, **mode `global`**, external controller on
`127.0.0.1:9097`. The controller secret is the `secret:` line of
`~/.local/share/io.github.clash-verge-rev.clash-verge-rev/clash-verge.yaml`; it is not written down
here and must not be committed.

Three things about global mode that cost an hour here:

- **In global mode there is no direct traffic.** Everything, including a plain `curl` with no
  proxy variables set, goes through whatever node the `GLOBAL` selector points at, because Clash
  runs a TUN interface (`utun1024`) and hijacks DNS to fake IPs in `198.18.0.0/16`. A dead node
  therefore presents as `SSL_ERROR_SYSCALL` on an unrelated host, which reads exactly like a
  GFW reset. Check the selector before diagnosing anything else:

  ```bash
  curl -s --noproxy '*' -H "Authorization: Bearer $SECRET" \
    http://127.0.0.1:9097/proxies/GLOBAL | python3 -c 'import json,sys;print(json.load(sys.stdin)["now"])'
  ```

  Switch it with `PUT /proxies/GLOBAL`, body `{"name": "<node>"}`.

- **A node that benchmarks fast dies within a minute or two.** Measured 2026-09-22 against the
  CARLA release mirror, 12 s per node:

  | Node | MB/s |
  |---|---|
  | AWS Japan 09 | 9.29 |
  | Tokyo 07, 0.1x "dedicated line" | 6.87 |
  | `DIRECT` | 6.41 |
  | San Jose 01 | 5.89 |
  | Hong Kong HKT hy2 | 0.47 |
  | US "good for downloading" 0.01x | 0.03-0.41 |

  The winner of that table, AWS Japan 09, then stalled at 1 MB per 45 s and never recovered, while
  `DIRECT` — fourth-ranked — pulled the remaining 7.1 GB in one unbroken stream at ~10 MB/s. So a
  12 s benchmark ranks nothing that matters. **Make the downloader rotate the node on every stall**
  (`--speed-limit 300000 --speed-time 15` plus a `PUT` to the selector on each retry) instead of
  picking a winner up front. The "0.01x, good for downloading" nodes were the slowest on offer.

- **GitHub over SSH works only on `DIRECT`.** Every proxy node tried closes port 22
  (`Connection closed by 198.18.0.7 port 22` — the fake IP gives away that it never left the
  tunnel). So `git pull` here means setting the selector to `DIRECT` first, where it works fine.
  Python packages are the opposite case: install from **official PyPI through the proxy**, because
  in global mode a domestic mirror is routed abroad too and `mirrors.aliyun.com` just times out.

## CARLA and Bench2Drive on this box

| Path | What | Use it? |
|---|---|---|
| `/data/third_party/carla/CARLA_0.9.15` | stock CARLA 0.9.15 + all additional maps | **yes** |
| `/data/third_party/Bench2Drive` | branch `0.0.4` @ `7ec25d1`, bundled leaderboard + scenario_runner | **yes, for evaluation** |
| `/data/envs/carla` | Python 3.8.20, PyPI CARLA 0.9.15 | **yes, for client and runner** |
| `~/mycode/Bench2Drive` | full repo, branch `main` @ `21d85ee`, `bench2drive220.xml` and Dev10 present | leave unchanged; do not use for comparable evaluation |
| `~/mycode/CARLA_Truck_Platoon_0.0.1` | a 0.9.15 **fork**, `VERSION` says `0.9.15-dirty`, Town01-07 + Town10HD only | no |
| `~/mycode/Carla-0.10.0` | CARLA 0.10.0, UE5 | no — Bench2Drive needs 0.9.15 |

Use the venv's Python explicitly: this repository targets Python 3.11 for other work, while
Bench2Drive needs this separate 3.8 environment. The PyPI wheel has no `carla.__version__`;
`carla.Client('localhost', 2000).get_client_version()` and `importlib.metadata.version('carla')`
both report `0.9.15` without requiring a running server.

The runner accepts `--windowed` to show a 1280x720 spectator window on the physical monitor.
Keep all long-running commands inside tmux session `jev`, and on this desktop use:

```bash
DISPLAY=:0 DATA_DIR=/data \
CARLA_ROOT=/data/third_party/carla/CARLA_0.9.15 \
BENCH2DRIVE_ROOT=/data/third_party/Bench2Drive \
  /data/envs/carla/bin/python scripts/b2d_run.py \
  --out /data/runs/b2d/smoke --workers 1 --gpu-rank 0 --windowed \
  --route-ids 24211,1711,1773 --rig front3 --policy none
DATA_DIR=/data /data/envs/carla/bin/python scripts/b2d_report.py \
  --out /data/runs/b2d/smoke --csv /data/runs/b2d/smoke/results.csv
```

Windowed timings include spectator rendering and should not be mixed with off-screen timings.
`finished` in the runner means evaluation ended normally; read the leaderboard checkpoint for
driving success, collisions or an agent-blocked failure.

Install dependencies from official PyPI through the proxy. The old `networkx==2.2` requires
NumPy below 1.24 (`np.int` was removed in 1.24); the installed pair is 2.2 / 1.23.5. The old
OpenCV 4.2.0.32 pin is replaced with 4.11.0.86 for Python 3.8. The exact environment is recorded
in `/data/runs/b2d/evidence/requirements-frozen.txt`. Neither bundled evaluator tree is patched.
NVIDIA's ICD is at `/usr/share/vulkan/icd.d/nvidia_icd.json` here, so the launcher falls back to
that path when the GPU box's `/etc/vulkan/icd.d/nvidia_icd.json` is absent.

The initial stub windowed smoke on 2026-09-22 was **stopped by the user**, not fully passed: route 24211
ended normally (1304 ticks, 45.5 s) but its driving verdict was `Failed - Agent got blocked`;
1711 was interrupted at 758 ticks; 1773 was not started. This run is now archived under
`/data/runs/b2d/smoke-stub-cancelled-20260922`; its `report.md` and
`results.csv` were generated successfully. `summary.json` lists 1711 and 1773 as never finished.
The cancelled attempt's `attempt.json` records the interruption explicitly: the evaluator itself
returned `finished` after SIGINT, so its original `route_result.json` is insufficient evidence of
a completed route. The runner's daemon workers also do not reliably clean up on Ctrl-C; the
remaining server and route process groups were stopped explicitly.

A read-only control probe reproduced a route-following defect in `StubAgent._steer_to_route`:
on a straight plan at x=[0, 10, 20], ego x=8 and yaw=0, it returns steer=-1.0 because it selects
the first waypoint more than 6 m away, including waypoints already behind the ego. This was later
fixed using a monotonic dense-route cursor and regression tests. Tick heartbeats before cancellation advanced from 147 to
583 in 30.44 s, then 742 in 10.13 s; the observed stationary scene is not by itself evidence of
a stalled client or server. Background-vehicle motion was not instrumented.

The stock install was added on 2026-09-22 because neither pre-existing build could run a
Bench2Drive route:

- the truck-platoon fork is a modified build, so nothing measured on it could be quoted; and
- **the additional maps were never imported.** The official
  `AdditionalMaps_0.9.15.tar.gz` (7375946087 bytes, byte-identical in size to the GPU box's copy,
  and it does contain Town11/12/13/15) was sitting unused in that fork's `Import/`, so the box had
  only Town01-07 and Town10HD — that is **165 of the 220 Bench2Drive routes missing**, Town12's 104
  among them. Having the tarball on disk is not having the maps; `ImportAssets.sh` has to run, and
  it consumes the tarball, so feed it a copy.

**`ImportAssets.sh` prints alarming `tar` errors and exits non-zero. Ignore them.** It ends with
`Unexpected inconsistency when making directory` on directories that already exist, then
`Exiting with failure status due to previous errors`, which under `set -e` aborts whatever called
it. The import had in fact finished. The way to settle that is a file-by-file diff against the GPU
box's working install, not the exit code:

```bash
ssh autodl 'cd $DATA_DIR/third_party/carla/CARLA_0.9.15 && find . -type f | sort' > a.txt
ssh ujs@100.108.238.8 'cd /data/third_party/carla/CARLA_0.9.15 && find . -type f | sort' > t.txt
LC_ALL=C comm -23 <(LC_ALL=C sort a.txt) <(LC_ALL=C sort t.txt)    # LC_ALL matters: the two boxes collate differently
```

44712 files here against 44721 there, and the nine extras were all `__pycache__/*.pyc` that autodl
generated by running Python. Identical install.

The separate checkout at `~/mycode/Bench2Drive` is on `main`. Controller diagnostics below
use `/data/third_party/Bench2Drive`, branch `0.0.4`, commit
`7ec25d1c9f7522d923ce5f3420986cef1cb2d956`. Record the actual evaluated path and commit;
do not infer them from the other checkout or switch its branch as part of a comparison.

Historical setup gaps (resolved by the environment and windowed runner described above):

- a Python 3.8 client venv (`uv` is at `~/.local/bin/uv`; system Python is 3.12). Install it from
  **official PyPI through the proxy**, not from a domestic mirror: in global mode the mirror's
  traffic is routed through a Japan node too, and `mirrors.aliyun.com` simply times out.
- a way to point the leaderboard at an already-running windowed server. Bench2Drive's
  `leaderboard_evaluator.py` starts CARLA itself with `-RenderOffScreen`, which is exactly the
  thing this box exists to avoid.
- a working way to pick the map. `./CarlaUE4.sh /Game/Carla/Maps/Town12` is **silently ignored** —
  the server comes up on Town10HD and says nothing about it. Use a client `load_world`, and read
  the map back rather than trusting the launch line.

## Official agents and single-window preview (2026-09-22 update)

The official NPC smoke at `/data/runs/b2d/smoke` completed all three evaluator processes,
one attempt each, with `routes_never_finished=[]`. Driving success is distinct from process
completion: 24211 and 1711 completed their routes with collision penalties; 1773 reached the
4000-tick limit at 36.39% completion. Its `results.csv` and `report.md` are generated.
The runner now handles cancellation using non-daemon workers and process-group cleanup;
an integration cancellation check exited 130 without an orphan server.

The learned policy is official Bench2DriveZoo `tcp/admlp` commit
`8a08b07883f10b7d83f6bf5dd475bda91a91c50a`, using `/data/envs/b2d-tcp` and
`/data/models/bench2drive/tcp/tcp_b2d.ckpt` (SHA256
`e6573ff1f8ea910b9a53eddfb68f69cac469bf5bfa253a516578f6126110b4fe`).
Set `IS_BENCH2DRIVE=1`, `PLANNER_TYPE=only_traj`,
`TORCH_HOME=/data/models/torch`, and
`PYTHONPATH=/data/third_party/Bench2DriveZoo:/data/third_party/Bench2DriveZoo/TCP`.
Pass `--agent scripts/b2d_tcp_visual_agent.py --agent-config <checkpoint>` and
`--python /data/envs/b2d-tcp/bin/python` to the windowed runner. Set `SAVE_PATH` to a
run-local directory (the official validator uses this to allow debug-camera extrinsics),
and `B2D_PREVIEW_DIR=<run>/live`. The wrapper suppresses official per-frame PNG dumps.

`scripts/b2d_viewer.py --dir <run>/live` runs in the carla venv with `DISPLAY=:0`.
It shows exactly one model-input mosaic and one chase view. The wrapper repositions the
existing, non-policy BEV debug camera to x=-6, z=3.5, pitch=-15, 640x360/FOV90;
the three policy cameras, official JPEG preprocessing, weights and controls are unchanged.
Predicted waypoints are reused from CPU PID metadata and projected in the viewer onto an
approximate flat road plane. Yellow lines are model predictions, not ground-truth routes,
and do not implement depth occlusion. Preview bytes use bounded double-buffered shared memory
under `/dev/shm`; slow/closed viewers never wait in the control loop. No preview JPEGs are saved.

The revised run is `/data/runs/b2d/tcp-fast` (stopped by user after two completed routes).
Early Town01 samples were about 11 ticks/s: sensor wait 38 ms, preprocessing 40 ms,
GPU forward 3.8 ms, total policy 46 ms, and preview publication 1.8 ms. Preprocessing/GPU
are components of policy time, not additional costs. These are online phase measurements,
not an A/B speedup claim. Per-route `live/performance-*.json` records mean/p95 after warmup.
The previous `/data/runs/b2d/tcp-smoke` was interrupted and must not be called a clean test;
its first route's timing includes a diagnostic X11 screenshot stall. Use the viewer's own
`--snapshot <png>` instead of grabbing the CARLA/X11 window for screenshots.

Last verified: 2026-09-22

### Preview layout / preprocessing follow-up

For the later same-frame pipelining optimization, measured comparison and enable switch,
see [tokyo-python-optimization.md](tokyo-python-optimization.md).

The top camera crops now form one contiguous labelled left/front/right strip, without
letterbox gaps. Below it are the chase projection and a lightweight vector BEV with a
5 m grid and approximate ego outline. Predictions retain their original values: a fixed
zero waypoint is prepended for display, and the GNSS reference's -1.4 m sensor offset is
applied to place it in vehicle coordinates. This is not smoothing or a policy correction.
Cached planner waypoints supply the blue route line; it is not a lane-boundary map.
No surrounding-actor queries or extra cameras are added.

`b2d_tcp_preprocess.py` parallelizes the three independent official JPEG round trips with
three CPU threads. All official resize/JPEG settings remain unchanged. The benchmark
`benchmark_b2d_preprocess.py` extracts the installed official `tick` and wrapper `tick`,
then checks every return field, including pixel equality and NaN compass handling.
35 synthetic-input comparisons passed; full preprocessing mean was 43.1 ms official vs
21.4 ms parallel in that experiment (not a whole-route speedup claim).
Logs: `/data/runs/b2d/evidence/preprocess-full-benchmark.log`.

This change was loaded for route 1773 in the ongoing `tcp-fast` run; 24211 and 1711
used the earlier sequential preprocessing. Do not aggregate these as a controlled A/B
performance measurement. The official speed-dependent throttle clamp (about 1.5 m/s
straight, 1 m/s turning) is untouched; slow driving is separate from low simulation FPS.

Final stop: routes 24211 and 1711 finished evaluator execution in one attempt each
(1889 ticks / 169.1 s / 87.3 ms per profiled tick, and 1847 ticks / 275.4 s /
132.5 ms per profiled tick). Route 1773 was intentionally cancelled at 661 ticks,
121.9 s wall time, one attempt. `summary.json` correctly lists 1773 as never finished;
the report's failed-attempt count includes this user cancellation, not a server crash.
The runner exited 130, with no remaining CARLA/route/viewer processes; results.csv was
generated. Closing the viewer alone does not remove policy cameras or their sensor waits.


## Controller diagnostic campaign and server reuse (2026-09-22)

The first frozen Dev10 controller comparison is complete under
`/data/runs/b2d/controller/dev10/`: CARLA/TCP/pursuit, seed0, ten official routes each,
31 attempts including one infrastructure retry. Each preset completed 9/10 driving routes;
all evaluator-returned runs remain distinct from official driving outcomes. Source and
validation details are in the [article notes](../todos/2026-09-22-b2d-controller/article-notes.md),
with [cost decomposition](bench2drive-cost.md#tokyo-controller-diagnostic-complete-dev10-seed0-2026-09-22)
and failure evidence (GPU box: `$DATA_DIR/runs/b2d/controller/git-offload-v1/todos/2026-09-22-b2d-controller/results/failure-analysis.json`).

Configuration: stock MKZ2020, stock CARLA0.9.15, Python `/data/envs/carla/bin/python`, Epic,
windowed, front3 at 800×450, cameras/plans at 5 Hz, motion/control at 20 Hz, `policy=none`,
nominal cruise 8 m/s. The route is a diagnostic input; it supplies no obstacle avoidance,
yielding or traffic-light planner. This is not the learned TCP checkpoint experiment above.
The physical GPU was **GPU 1**, UUID `GPU-b90dd90e-394b-7800-f23f-5892a8e3d0f1`, selected
with CARLA `--gpu-rank 0` / `-graphicsadapter=0`. Preserve the UUID check after launch.

| Preset | All-route ticks min / median / max | Completed-route ticks min / median / max | Failed-route ticks | Group wall |
|---|---|---|---:|---:|
| CARLA |253 / 405 / 4000|253 / 395 / 3868|4000|342.6 s|
| TCP |254 / 432 / 4000|254 / 392 / 3881|4000|378.6 s|
| pursuit |256 / 401.5 / 4000|256 / 397 / 3871|4000|345.3 s|

The campaign completion marker is 1074.650 s (17.91 min) after manifest start. That interval
includes initial readiness, the retry and report orchestration, but final server shutdown
occurs after the marker and was not separately timed. Finalized attempts total 1027.0 s;
TCP's failed first attempt adds 22.0 s. Initial readiness/preflight was 7.088 s and crash
restart to retry was 7.008 s. Do not fold these into the 12.34–12.76 ms weighted profiled
tick cost or call all non-tick time server startup.

The old 600–1200 ticks per route prediction misses both ends of this diagnostic: the eight
shorter completed routes span 253–697 ticks, while collision-associated route 2091 takes
3868–3881 ticks and route 25424 fails at 4000. A user also visually observed the crossroads
collision/stall during the later seed1 run; that observation is distinct from the seed0
logged frame evidence and is not an additional timed measurement. The near-180 s seed0
low-progress intervals are in the raw logs. Fast evaluator throughput does not imply good
driving, and these Tokyo/no-model timings must not be substituted for GPU-box inference
costs or directly multiplied by 22 to forecast full220.

### Reuse the process, keep recovery explicit

One server process served all ten CARLA routes and the first seven TCP routes. On the
**18th attempt**, TCP route 27494 in **Town04**, it died with `Signal 11` and
`CommonUnixCrashHandler`; the attempt records `server_died_rc139`, `server_age_routes=17`,
`server_age_s=665`. Source evidence is `dev10/servers/carla-74-1.log` and
`dev10/tcp-seed0/attempts/27494/1/`. The failure does not prove an age-dependent leak or
identify a specific engine defect. It also disproves treating the earlier observed
Town12/Town13 crash set as an exclusive list of maps that can crash.

The runner moved from index74 (RPC5700/TM11700) to index75 (RPC5750/TM11750), retried 27494
successfully, then reused the replacement through the pursuit group. Normal world resets
still run through the evaluator. Reuse avoids unnecessary process starts; it does not remove
map loading, actor cleanup or the need to retain a failed attempt and restart on a confirmed
server death. No fixed recycling interval has been established by this one crash.

### Interpretation and validation traps found in this iteration

- `finished` means evaluator return. All three 25424 attempts are official
  `Failed - TickRuntime` despite returncode0 and no extra CLI tick cap. The locked upstream
  manager still has its own 4000-tick limit.
- Construction warning collisions followed by throttle .75 and almost no motion support
  an obstructed centerline reference. Collision messages give the ego position, not an
  obstacle center or bounding box; those fields alone cannot reconstruct exact contact.
- A 20 Hz window of 100 samples spans **4.95 s**, not 5 s. The intermediate G2 validator
  therefore mislabelled nine parking cases as blocked before the hold gate matured.
  `development2` is retained; `development3` reran all twelve cases after that fix.
- Telemetry may contain a null target speed during a defined fault. Report code must retain
  the sample and mark unavailable metrics, rather than crash or convert null to zero.
- Collision contact can produce negative signed speed. The agent logs raw speed and only
  applies a small resting deadband at its boundary; larger reverse motion triggers the
  controller's defined invalid-motion brake state. It must not disappear from diagnostics.
- For ParkingExit3514, the origin-to-firstpoint bridge implies 19.381 m/s while the next
  future-to-future segment implies 8 m/s. This is a timed-route-adapter limitation when the
  vehicle starts off the centerline, not evidence it physically travels at 19.381 m/s.
  Target/reference speed share that bridge and are not independent checks of nominal cruise.

The seed0 core remained frozen during the comparison; later confirmation and holdout results
belong in their own evidence records. Intermediate failures, source snapshots and precise
provenance limits are indexed by the article notes.

### Controller feedback iteration: compass dropout and completed reuse audit

The later v4 smoke reproduced a single nonfinite IMU compass value at simulation time 9.10 s while GNSS, gyro and speed remained finite. Log raw motion before pose validation. The initialized filter now predicts heading from the gyro for at most 0.2 s, records degraded observations, and skips compass correction; an uninitialized or longer dropout brakes and resets history. A fresh valid pose forces a new trajectory. The recovery smoke completed 100%, and 126 regression tests passed. This is bounded sensor recovery, not simulator-truth substitution.

The completed v4 campaign reused its server across all six groups and restarted only after three recorded rc139 failures. All failed attempts remain archived. This observation still does not establish a fixed safe process lifetime. Its 11 official TickRuntime failures all had contact before prolonged low speed; zero official blocked/deviation events must not be reported as zero real stalls, and ego-only logs do not establish collision responsibility. See the [final report](../todos/2026-09-22-b2d-controller/final-report.md) for the 60 selected records, 63 attempts and all intermediate evidence.
