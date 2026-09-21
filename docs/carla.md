# CARLA

Read this when you need a CARLA simulator on the box, or want to know what a closed-loop
Bench2Drive evaluation would cost us.

**Status: CARLA 0.9.15 runs headless on our Blackwell card and renders real frames on the GPU.**
Closed-loop evaluation is technically possible here. We have not decided to do it.

## Vulkan in the container: the one hard blocker, and its fix

CARLA renders through Vulkan even with `-RenderOffScreen`, so the container needs a working NVIDIA
Vulkan driver. Ours did not have one, and the failure does not point at its cause.

`/etc/vulkan/icd.d/nvidia_icd.json` was already present (mounted from the host driver) and pointed
at `libGLX_nvidia.so.0`, which was also present and exported the right symbols. Yet `vulkaninfo`
reported only `llvmpipe` and said:

```
ERROR: [Loader Message] Code 0 : loader_scanned_icd_add: Could not get 'vkCreateInstance'
       via 'vk_icdGetInstanceProcAddr' for ICD libGLX_nvidia.so.0
```

Calling the ICD entry points by hand showed `vk_icdNegotiateLoaderICDInterfaceVersion` returning
`-3` (`VK_ERROR_INITIALIZATION_FAILED`) for every interface version 1-7. So the driver was refusing
to initialise, not failing to negotiate a version. `strace` gave the reason: NVIDIA's Vulkan driver
`dlopen`s `libEGL.so.1`, and GLVND's EGL dispatch library was not installed. Only the vendor library
`libEGL_nvidia.so.0` was there.

**Installing `libegl1` fixes it.** Nothing else was needed.

**Do not rebuild the Vulkan loader.** Ubuntu 22.04 ships loader 1.3.204, which looks far too old for
a driver advertising Vulkan 1.4.329, so it is the obvious suspect. It is not the problem: we built
loader 1.4.313 from source first and it failed in exactly the same way. The stock `libvulkan1` works
once `libegl1` is present.

Packages added as root, all from the Huawei Cloud mirror already in `/etc/apt/sources.list`:

| Package | Why |
|---|---|
| `libegl1` | **the actual fix**: GLVND EGL dispatch, which NVIDIA's Vulkan ICD dlopens |
| `libvulkan1` | Vulkan loader |
| `vulkan-tools` | `vulkaninfo`, to check the above |
| `libsdl2-2.0-0`, `libomp5`, `xdg-user-dirs` | CARLA's own runtime dependencies |

Check with `vulkaninfo --summary`. It must list `NVIDIA RTX PRO 6000 Blackwell Server Edition`,
Vulkan 1.4.329, driver 595.71.05.

`vulkan-tools` pulls in `mesa-vulkan-drivers`, which adds `llvmpipe` as a second Vulkan device.
CARLA will happily pick it and render on the CPU, so `scripts/carla_server.sh` pins
`VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`.

A re-created instance loses all of this. Re-run the apt install.

## Traps

Each of these cost us time; none is guessable.

- **`SDL_VIDEODRIVER=offscreen` breaks CARLA.** It exits 1 immediately, printing nothing past
  `Disabling core dumps.`. `-RenderOffScreen` already does the headless part and SDL is not involved.
  Unset it, or use `dummy`. Setting it is a natural thing to try, which is why it is listed first.
- **A healthy server prints nothing.** Shipping builds stop at `Disabling core dumps.` and stay
  silent. Silence is not a hang.
- **`Exiting abnormally (error code: 143)` is not a crash.** 143 is `128 + SIGTERM`: something
  killed it, usually your own `timeout`. A real startup failure exits 1 within about four seconds.
- **Do not use `setsid cmd &` and then watch `$!`.** `setsid` forks, the parent exits at once, and
  the liveness check watches a dead PID while CARLA runs fine. This made us report a crash twice.
- **Never `pgrep -f`/`pkill -f` a pattern that appears in your own command line.** We killed our own
  ssh session with `pgrep -f carla-releases`, the same hazard `long-runs.md` describes. Use
  `pgrep -x curl`, explicit PIDs, or a process group.
- **Stop sensors before destroying actors.** A camera callback that fires on a destroyed sensor
  throws inside a CARLA worker thread; an uncaught exception there calls `terminate()` and aborts
  the whole client. Call `sensor.stop()` on every sensor, `world.tick()` once to drain in-flight
  callbacks, then destroy via `client.apply_batch_sync`. Report results before tearing down, so a
  cleanup fault cannot cost a finished measurement.

Two things we suspected and disproved, recorded so nobody re-investigates them: the `CarlaUE4.sh`
wrapper's `echo \"$0\" | xargs readlink -f` line looks broken but is harmless (`xargs` strips the
literal quotes; all five launch variants worked), and the Vulkan loader version was not the problem.

## Python client

The tarball ships **cp27 and cp37 wheels and eggs only** - there is no cp38 artifact, despite
Bench2Drive's README saying "python 3.8 also works well". Bench2Drive's own install (a `carla.pth`
pointing at the shipped egg) therefore pins Python 3.7, while Bench2DriveZoo's `INSTALL.md` insists
on 3.8. Resolve it with the PyPI wheel, which does have cp38:

```bash
uv venv --python 3.8 $DATA_DIR/envs/carla
VIRTUAL_ENV=$DATA_DIR/envs/carla uv pip install \
  --default-index https://mirrors.aliyun.com/pypi/simple carla==0.9.15 numpy'<1.25' pillow
```

Keep this venv separate from `envs/jevdrive` (Python 3.11): the 0.9.15 client cannot coexist with it.

## Starting and stopping a server

```bash
scripts/carla_server.sh start 0     # index 0 -> rpc port 2000; index i -> 2000 + 4i
scripts/carla_server.sh status
scripts/carla_server.sh stop 0      # omit the index to stop every server we started
```

It launches `CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=<port> -quality-level=Epic`,
pins the NVIDIA ICD, and waits for the RPC port. Stopping kills the process group, never `pkill -f`.
Logs go to `$DATA_DIR/runs/carla/carla-<i>.log`.

Check a server really renders, rather than trusting a successful connect:

```bash
$DATA_DIR/envs/carla/bin/python scripts/carla_bench.py --port 2000 --town Town10HD_Opt
```

It spawns a vehicle and cameras and reports per-camera pixel mean/std/unique counts, exiting
non-zero if every camera's std is below 1.0. A server rendering through `lavapipe` on the CPU, or
failing to render at all, answers RPCs perfectly happily and hands out uniform buffers; pixel
statistics are the only check that catches it.

## Measured throughput

Town10HD_Opt, 40 background vehicles, synchronous 20 Hz, `-quality-level=Epic`, one instance.
"Real-time factor" is simulated seconds per wall second; 1.0 means the simulator keeps up with
the 20 Hz clock.

| Cameras | Resolution | FPS | Real-time factor |
|---|---|---|---|
| 1 | 1600x900 | 28.5 | 1.42 |
| 6 | 1600x900 | 9.6 | 0.48 |
| 6 | 800x600 | 10.5 | 0.52 |

**Camera count dominates, resolution barely matters.** Cutting six cameras from 1600x900 to
800x600 is a 4x reduction in pixels and buys 9% more throughput. The cost is per-sensor overhead,
not GPU fill rate. A six-camera agent (UniAD, VAD) runs at about half real time no matter how far
you cut resolution; the only lever that helps is fewer cameras.

One instance uses **4.5 cores, 8.0 GB VRAM, 4.1 GB RAM**.

## Parallelism: the GPU binds first, and then it crashes

One camera at 1600x900, 40 vehicles, clients staggered 20 s apart.

| Instances | Aggregate FPS | Scaling | VRAM | GPU util |
|---|---|---|---|---|
| 1 | 28.2 | 1.00x | 6.8 GB | 48% |
| 2 | 46.8 | 1.66x | 14.1 GB | 77% |
| 4 | **86.5** | 3.07x | 27.5 GB | 94% |
| 5 | 82.0 | 2.91x | 30 GB | 99% |
| 6 | - | - | - | sixth server segfaults at startup |

**Four instances is the operating point.** Throughput peaks there and *falls* at five. A sixth
server dies during startup with `Signal=11` / `CommonUnixCrashHandler`, so it is a hard limit, not
a slow start.

Neither the cores nor the VRAM we sized the box around is the constraint: at five instances VRAM
is 31% used and CPU is 16.7 of 25 cores. (That 16.7 is with all five clients driving; idle servers ticking between routes draw 13.3. Both are well under the quota, which is the point.) The GPU saturates first, and past saturation CARLA falls
over rather than degrading gracefully. **The 96 GB card is heavily over-provisioned for this
workload** - a much smaller GPU would hit the same four-instance limit.

**Stagger instance startup.** Starting four clients at once made one of them time out during
`load_world`; staggering them 20 s apart made all four succeed *and* raised aggregate throughput
from 74.2 to 86.5. Bench2Drive's own multi-task script staggers by 5 s and its evaluator sleeps 30 s
after spawning each server, which suggests its authors hit this too. `scripts/carla_parallel.sh`
does the staggering.

## What a Bench2Drive evaluation would cost

Not measured - we have run no routes. This is arithmetic on the numbers above, and the assumptions
are the load-bearing part.

Assumptions: 220 routes; 20 Hz; `scenario_manager.py` caps a route at 4000 ticks (200 simulated
seconds); routes are short (mean 105 m, median 104 m, measured from `bench2drive220.xml`), so a
typical route is perhaps 1000 ticks, with 4000 the worst case. Four concurrent instances.

| Agent | Sim throughput | 220 routes @ 1000 ticks | @ 4000 ticks |
|---|---|---|---|
| 1 camera | 86.5 ticks/s (measured) | 0.7 h | 2.8 h |
| 6 cameras | ~24 ticks/s (**extrapolated**, not measured at N=4) | 2.5 h | 10 h |

That is the simulator floor with a free agent. A real agent costs more, and the way it costs more
matters: **CARLA already saturates the GPU at four instances**, so model inference competes with
rendering on the same card. Running a heavy BEV model means fewer CARLA instances, not the same
number plus inference.

For calibration, published numbers: carla_garage reports ~4 h for TransFuser++ on **8x 2080Ti**
(≈32 GPU-hours), and the Bench2Drive authors report "several days" on 4x A6000 for UniAD/VAD-class
models. Scaled to our single card at four instances, expect **roughly a day for a light agent and
several days for a six-camera BEV model**. Use the 10-route Dev10 subset
(`drivetransformer_bench2drive_dev10.xml`) to bring a pipeline up; it exists for exactly this.

## What the smoke test does not prove

It proves rendering works. It does not prove a 220-route run survives, and two known problems are
out of its reach:

- **Sensor-dormancy segfault** (Bench2Drive #235, upstream carla #7772, both open): a non-hero
  actor carrying sensors entering dormancy crashes the server. It affects Town12, which is 104 of
  the 220 routes. Reproduces on stock 0.9.15, binary and source build alike.
- **Long unattended runs hang** on Town12/Town13 (Bench2Drive #234), for the model and the expert.

Neither is Blackwell-specific. Bench2Drive's README recommends looping the evaluation until it
completes, because "CARLA is easy to crash", and ships `tools/clean_carla.sh` for the wreckage.

## Prerequisites for a real evaluation

| Item | Size | Status |
|---|---|---|
| `CARLA_0.9.15.tar.gz` | 7.81 GiB (8,386,636,048 B) | **done**, extracted to 19 GB |
| `AdditionalMaps_0.9.15.tar.gz` | 6.87 GiB (7,375,946,087 B) | **paused at 4.98 GB**, resume with `scripts/download_carla.sh maps` |
| Bench2Drive repo (branch `0.0.4`) | 145 MB | **done**, `$DATA_DIR/third_party/Bench2Drive` |
| Python 3.8 client venv | small | **done**, `$DATA_DIR/envs/carla` |
| Bench2Drive **training** dataset | 400 GB - 7.3 TB | **not needed** and not downloaded |

**AdditionalMaps is not optional.** The 220 routes break down by town as:

| Town | Routes | In base package? |
|---|---|---|
| Town12 | 104 | no |
| Town13 | 47 | no |
| Town11 | 7 | no |
| Town15 | 7 | no |
| Town01-10 | 55 | yes |

So 165 of 220 routes need the extra 6.87 GiB. The base package alone covers 55 routes and is enough
for smoke tests and throughput work, which is why we ran today's measurements on Town10HD_Opt.

The harness needs no other downloads: Bench2Drive vendors `leaderboard/` (22.6 MB) and
`scenario_runner/` (39.2 MB) in-tree, and they are **modified** - `leaderboard_evaluator.py` spawns
CARLA itself with `-RenderOffScreen` and `--gpu-rank`, and `scenario_manager.py` adds the 4000-tick
cap. Do not replace them with upstream `leaderboard-2.0`. There are no separate scenario JSON files
either: Leaderboard 2.0 inlines scenario triggers and weather into the route XML, so
`bench2drive220.xml` (677 KB) is the whole definition.

Evaluation needs no clip from the training dataset. A camera-in/control-out agent reads the route
XML and the vendored `srunner` and nothing else. An agent needing BEV/map ground truth would need
`Bench2Drive-Map-V0.0.4`, which is a separate decision.

Note `-graphicsadapter=N` selects a Vulkan physical device, not a CUDA device, and is unrelated to
`CUDA_VISIBLE_DEVICES`. We have one GPU so it does not matter here, but it does on a multi-GPU box.

Last verified: 2026-09-21
