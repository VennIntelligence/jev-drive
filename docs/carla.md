# CARLA

Read this when you need a CARLA simulator on the box.
[bench2drive-cost.md](bench2drive-cost.md) is the doc for what a closed-loop run costs;
this one is about getting a server up and the traps in doing so.

**Status: CARLA 0.9.15 runs headless on our Blackwell card and renders real frames on the GPU.
A real Bench2Drive route runs end to end on Town12, the heaviest map.** All 220 routes are
available. We have not decided to run them.

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
`-3` (`VK_ERROR_INITIALIZATION_FAILED`) for every interface version 1-7, so the driver was refusing
to initialise rather than failing to negotiate a version. `strace` gave the reason: NVIDIA's Vulkan
driver `dlopen`s `libEGL.so.1`, and GLVND's EGL dispatch library was not installed. Only the vendor
library `libEGL_nvidia.so.0` was there.

**Installing `libegl1` fixes it.** Nothing else was needed. We have not found this published
anywhere, and it is a plausible cause of the "incompatible vulkan driver found" reports from other
RTX 50-series users (carla #9502, #9725).

**Do not rebuild the Vulkan loader.** Ubuntu 22.04 ships loader 1.3.204, which looks far too old for
a driver advertising Vulkan 1.4.329, so it is the obvious suspect. It is not: we built loader
1.4.313 from source first and it failed identically. Stock `libvulkan1` works once `libegl1` is in.

Packages added as root, all from the Huawei Cloud mirror already in `/etc/apt/sources.list`:

| Package | Why |
|---|---|
| `libegl1` | **the actual fix**: GLVND EGL dispatch, which NVIDIA's Vulkan ICD dlopens |
| `libvulkan1` | Vulkan loader |
| `vulkan-tools` | `vulkaninfo`, to check the above |
| `libsdl2-2.0-0`, `libomp5`, `xdg-user-dirs` | CARLA's own runtime dependencies |

Check with `vulkaninfo --summary`: it must list `NVIDIA RTX PRO 6000 Blackwell Server Edition`,
Vulkan 1.4.329, driver 595.71.05. A re-created instance loses all of this; re-run the apt install.

`vulkan-tools` pulls in `mesa-vulkan-drivers`, which adds `llvmpipe` as a second Vulkan device.
CARLA will pick it and render on the CPU, so pin `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`.

## Traps

- **A traffic-manager port outlives the server that owned it.** The most expensive trap here. An
  unattended 44-route run needed 12 restarts and *all 12* were traffic-manager port reuse; CARLA
  never crashed. Kill a server, start another on the same RPC port, and the next client's
  `get_trafficmanager()` fails with `trying to create rpc server for traffic manager; but the system
  failed to create because of bind error`. It reads exactly like a server crash, and it masquerades
  as whatever you were testing - it wasted two cells of a spawn experiment here by failing before
  the spawn was reached. Our scripts space RPC ports 50 apart (`2000 + 50i`, TM `8000 + 50i`); they
  used 4, which is the same hazard. `scripts/carla_bench.py --tm-port` overrides it for consecutive
  runs on one RPC port.
  Bench2Drive spaces its task ports **150** apart and its README says to avoid ports below 10000
  ("<10000 could be unsafe"). Our TM ports at `8000 + 50i` are inside that range, so moving both
  ranges above 10000 is still worth doing; the wider spacing alone may not be the whole fix.
- **`SDL_VIDEODRIVER=offscreen` breaks CARLA.** It exits 1 immediately, printing nothing past
  `Disabling core dumps.`. `-RenderOffScreen` already does the headless part; SDL is not involved.
  Unset it, or use `dummy`. Setting it is a natural thing to try, which is why it is listed here.
- **`-quality-level=Low` with `-RenderOffScreen` segfaults** (carla #4940, #4966, #7675). We run
  `Epic`. `scripts/b2d_run.py` exposes `--quality Low`; do not use it.
- **CARLA ignores `CUDA_VISIBLE_DEVICES`.** Select the GPU with `-graphicsadapter=<rank>`, which is
  a Vulkan physical-device index. Bench2Drive's README warns the mapping can be off by one or more:
  on a 4-GPU box they saw GPU0 -> 0, GPU1 -> 2, GPU2 -> 3, GPU3 -> 4. Irrelevant on our one card,
  load-bearing the moment we use two.
- **CARLA silently resets `tile_stream_distance` and `actor_active_distance` on every
  `load_world`.** The leaderboard re-applies both (650 m) after each load, with the comment "Large
  Map settings are always reset, for some reason". On Large Maps (Town11/12/13/15) these govern
  actor dormancy, and getting them wrong crashes the server - see the Town12 section below.
- **An open RPC port is not a ready server.** Bench2Drive sleeps a flat 30 s after starting the
  server before connecting, and retries `load_world` up to 20 times (and the traffic manager 40),
  treating a server that will not come up as routine. Our `carla_server.sh` returns as soon as the
  port accepts, which is 4-7 s. The port check is a liveness gate, not a readiness one.
- **A healthy server prints nothing.** Shipping builds stop at `Disabling core dumps.` and stay
  silent. Silence is not a hang.
- **`Exiting abnormally (error code: 143)` is not a crash.** 143 is `128 + SIGTERM`, usually your
  own `timeout`. A real startup failure exits 1 within about four seconds.
- **`alive=0` plus a client RPC timeout is ambiguous.** A dead server and a healthy one you gave up
  on look identical from the client. Check the server log for `Signal=11` /
  `CommonUnixCrashHandler`, or sample the process, before concluding anything. This signature misled
  us twice in one night.
- **Do not use `setsid cmd &` and then watch `$!`.** `setsid` forks, the parent exits at once, and
  the liveness check watches a dead PID while CARLA runs fine.
- **Never `pgrep -f`/`pkill -f` a pattern that appears in your own command line.** We killed our own
  ssh session with `pgrep -f carla-releases`, the hazard `long-runs.md` describes. Use `pgrep -x`,
  explicit PIDs, or a process group.
- **Stop sensors before destroying actors.** A camera callback firing on a destroyed sensor throws
  inside a CARLA worker thread; an uncaught exception there calls `terminate()` and aborts the
  client. `sensor.stop()` on every sensor, `world.tick()` once to drain, then
  `client.apply_batch_sync`. Report results before tearing down.

Debugging aids that do **not** work here, so nobody spends the time: the container surfaces no host
kernel messages, so `dmesg` never shows the segfault; and UE4 writes no crash report
(`CarlaUE4/Saved/Crashes` never appears) because its own handler suppresses core dumps and
re-raises. A real backtrace needs `gdb` with `-nocrashhandler`.

## Town12 and Large Maps

Town12 (104 of the 220 routes) and Town13 (47) are Large Maps with streamed tiles and actor
dormancy. A real Bench2Drive route runs on Town12 end to end through the unmodified leaderboard:
route 1711 finished in 288 s over 1283 ticks with zero restarts.

Attaching a sensor on a Large Map with the wrong world settings segfaults the server. The backtrace
is specific - the spawn-actor RPC runs a dormancy pass, dormancy *destroys* a sensor, and
`ASensor::EndPlay` dereferences an invalid stream token:

Verbatim, from `gdb` with `-nocrashhandler` (paths under `Plugins/Carla/...` abbreviated, nothing
else changed), so the reading can be checked rather than taken on trust:

```
#0  carla::streaming::detail::token_type::operator carla::streaming::Token() const ()
#1  carla::streaming::detail::Stream<...MultiStreamState>::token (this=0x7ff401a41e40)
        at .../carla/streaming/detail/Stream.h:36
#2  FDataStreamTmpl<...>::GetToken (this=0x7ff401a41e38)
        at .../Carla/Source/Carla/Sensor/DataStream.h:55
#3  ASensor::EndPlay (this=0x7ff401a41c00, EndPlayReason=<optimized out>)
        at .../Carla/Source/Carla/Sensor/Sensor.cpp:120
#4  AActor::RouteEndPlay (this=0x7ff401a41c00, EndPlayReason=EEndPlayReason::Destroyed)
        at Runtime/Engine/Private/Actor.cpp:2257
#5  AActor::Destroyed (this=0x7ff401a41c00)          at Runtime/Engine/Private/Actor.cpp:2332
#6  UWorld::DestroyActor (ThisActor=0x7ff401a41c00)  at Runtime/Engine/Private/LevelActor.cpp:708
#7  AActor::Destroy (this=0x7ff401a41c00, bNetForce=false, bShouldModifyLevel=true)
        at Runtime/Engine/Private/Actor.cpp:4056
#8  FCarlaActor::PutActorToSleep (this=0x7ff9e5d0cb20, CarlaEpisode=0x7ffab3b3c800)
        at .../Carla/Source/Carla/Actor/CarlaActor.cpp:161
#9  FActorRegistry::PutActorToSleep (Id=<optimized out>, CarlaEpisode=0x7ffab3b3c800)
        at .../Carla/Source/Carla/Actor/ActorRegistry.cpp:198
#10 UCarlaEpisode::PutActorToSleep (this=0x7ffd70b7e220, ActorId=8)
        at .../Carla/Source/Carla/Game/CarlaEpisode.h:282
#11 FCarlaServer::FPimpl::BindActions()::$_46::operator()(carla::rpc::ActorDescription,
        carla::geom::Transform const&, unsigned int, carla::rpc::AttachmentType) const
        (InAttachmentType=carla::rpc::AttachmentType::Rigid)
        at .../Carla/Source/Carla/Server/CarlaServer.cpp:767
```

Read bottom-up: #11 is the spawn-actor RPC handler with `AttachmentType::Rigid`, i.e. the
`spawn_actor(..., attach_to=ego)` call. Inside it CARLA runs a dormancy pass and puts ActorId 8 to
sleep (#10-#8); sleeping an actor *destroys* it (#7-#4); destruction routes `EndPlay` into
`ASensor::EndPlay` (#3), which asks the sensor's data stream for its token (#2-#1) and faults on an
invalid one (#0).

This is the sensor-dormancy crash of Bench2Drive #235 / carla #7772.

**What is established:** the mechanism above, from the backtrace; and that the unmodified
leaderboard runs Town12 routes on this box without hitting it.

**What is not:** which difference between our client and theirs is responsible. We diffed the
leaderboard's world setup against `carla_bench.py` and tested the candidates on the failing cell
(Town12, one camera, no traffic, fresh server and distinct TM port per cell):

| Cell | Result |
|---|---|
| baseline (`synchronous_mode` + `fixed_delta_seconds` only) | crash |
| `+ spectator_as_ego=False` | crash |
| `+ tile_stream_distance=650`, `actor_active_distance=650` | crash |
| full bundle: both of the above, `deterministic_ragdolls=True`, `reset_all_traffic_lights()`, settle tick | crash |
| `spectator_as_ego=False` repeat | crash |

So the world settings are **not** the fix, and neither single line is. The remaining differences we
did not test are the order of operations (`_setup_simulation` applies `WorldSettings` to the world
*before* loading, then `load_world(town, reset_settings=False)`; ours loads first and applies
after), the 30 s readiness sleep, `-graphicsadapter`, `find_free_port`,
`set_hybrid_physics_mode(True)`, `set_random_device_seed()`, and the fact that the leaderboard
creates its sensors through `AgentWrapper` inside a `RouteScenario` rather than directly. The
`reset_settings=False` ordering is the most promising of those and was not reached.

**Practical guidance until someone finishes this:** for anything on a Large Map, go through the
leaderboard (`scripts/b2d_run.py`), which works. Do not write direct client code against Town12/13
and expect it to survive attaching a sensor.

One paragraph on how this was found, because the shape of the mistake is the lesson. Our own
`scripts/carla_bench.py` crashed on Town12 and we spent hours treating it as a property of CARLA:
four hypotheses - the map cannot load, spawn clearance, background traffic dormancy, server
readiness - each fitting the observed pattern, each tested with paired designs, each dead. Running
one official route settled the question in five minutes, and the working reference had been on disk
the whole time. Official code that other people have published results with should be the first
thing you run, not the last.

## Python client

The tarball ships **cp27 and cp37 wheels and eggs only** - no cp38, despite Bench2Drive's README
saying "python 3.8 also works well". Their own install (a `carla.pth` pointing at the shipped egg)
therefore pins 3.7, while Bench2DriveZoo's `INSTALL.md` insists on 3.8. Use the PyPI wheel, which
has cp38:

```bash
uv venv --python 3.8 $DATA_DIR/envs/carla
VIRTUAL_ENV=$DATA_DIR/envs/carla uv pip install \
  --default-index https://mirrors.aliyun.com/pypi/simple carla==0.9.15 numpy'<1.25' pillow
```

Keep this venv separate from `envs/jevdrive` (Python 3.11).

## Starting and stopping a server

```bash
scripts/carla_server.sh start 0     # index 0 -> rpc port 2000; index i -> 2000 + 50i
scripts/carla_server.sh status
scripts/carla_server.sh stop 0      # omit the index to stop every server we started
```

It launches `CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=<port> -quality-level=Epic`,
pins the NVIDIA ICD, and waits for the RPC port. Stopping kills the process group, never `pkill -f`.
Logs go to `$DATA_DIR/runs/carla/carla-<i>.log`. It does **not** implement Bench2Drive's 30 s
readiness sleep or their load retries; add those before trusting it for unattended runs.

Check a server really renders, rather than trusting a successful connect:

```bash
$DATA_DIR/envs/carla/bin/python scripts/carla_bench.py --port 2000 --town Town10HD_Opt
```

It reports per-camera pixel mean/std/unique counts and exits non-zero if every camera's std is below
1.0. A server rendering through `lavapipe` on the CPU answers RPCs happily and hands out uniform
buffers; pixel statistics are the only check that catches it.

## Numbers, and where each came from

Two different things have been measured, and they are not comparable. Read the source column before
quoting any of it.

**From `scripts/carla_bench.py`** - a server-side ceiling on **Town10HD_Opt only**, with a
non-blocking consumer (the camera callback keeps whatever arrived; it never waits), on an otherwise
idle card. Not a closed-loop rate, and never validated on a Large Map.

| Cameras | Resolution | FPS | Real-time factor |
|---|---|---|---|
| 1 | 1600x900 | 28.5 | 1.42 |
| 6 | 1600x900 | 9.6 | 0.48 |
| 6 | 800x600 | 10.5 | 0.52 |

One instance: 4.5 cores, 8.0 GB VRAM, 4.1 GB RAM. Concurrency on Town10HD_Opt peaked at **four**
instances (86.5 FPS aggregate, 3.07x) and fell at five; a sixth server segfaulted at startup. At
five instances VRAM was 31% used and CPU 16.7 of 25 cores while the GPU sat at 99%, so the GPU binds
first and the 96 GB card is over-provisioned for this. Starting instances simultaneously made one
time out during `load_world`; staggering them 20 s apart fixed it and raised aggregate throughput
from 74.2 to 86.5.

**From the real leaderboard via `scripts/b2d_run.py`** - closed-loop, real routes, blocking sensor
waits and the scenario tree included. These are the numbers that count. See
[bench2drive-cost.md](bench2drive-cost.md); as a single anchor, Town12 route 1711 took 288 s for
1283 ticks (0.225 s/tick) with one instance and no policy in the loop.

The closed loop is roughly **3x slower** than the server-side ceiling above (1 camera: 0.54x real
time there against 1.42x here), because four fifths of a tick is one blocking wait for camera
frames. Both agree on the useful conclusion: cost is per-sensor, not per-pixel. Cutting pixels 4x
bought 9% here and 16x bought 3% there, both inside that doc's 7-8% noise band. Render fewer
cameras; resolution will not save you.

## Prerequisites for a real evaluation

| Item | Size | Status |
|---|---|---|
| `CARLA_0.9.15.tar.gz` | 7.81 GiB | done, extracted to 19 GB |
| `AdditionalMaps_0.9.15.tar.gz` | 6.87 GiB | done, downloaded and imported |
| Bench2Drive repo (branch `0.0.4`) | 145 MB | done, `$DATA_DIR/third_party/Bench2Drive` |
| Python 3.8 client venv | small | done, `$DATA_DIR/envs/carla` |
| Bench2Drive **training** dataset | 400 GB - 7.3 TB | **not needed**, not downloaded |

Downloading AdditionalMaps is not enough - it must be imported, or the extra towns are simply absent:

```bash
scripts/download_carla.sh maps
mv $DATA_DIR/third_party/carla/AdditionalMaps_0.9.15.tar.gz $CARLA_ROOT/Import/
cd $CARLA_ROOT && bash ImportAssets.sh      # consumes the tarball, takes a few minutes
```

It is also not optional. The 220 routes by town:

| Town | Routes | In base package? |
|---|---|---|
| Town12 | 104 | no |
| Town13 | 47 | no |
| Town11 | 7 | no |
| Town15 | 7 | no |
| Town01-10 | 55 | yes |

So 165 of 220 routes need it. Installed maps are now Town01-07, 10, 11, 12, 13, 15.

The harness needs no other downloads: Bench2Drive vendors `leaderboard/` (22.6 MB) and
`scenario_runner/` (39.2 MB) in-tree, and both are **modified** - `leaderboard_evaluator.py` spawns
CARLA itself with `-RenderOffScreen` and `--gpu-rank`, and `scenario_manager.py` adds a 4000-tick
route cap. Do not replace them with upstream `leaderboard-2.0`. There are no separate scenario JSON
files: Leaderboard 2.0 inlines scenario triggers and weather into the route XML, so
`bench2drive220.xml` (677 KB) is the whole definition. Evaluation needs no clip from the training
dataset; an agent wanting BEV/map ground truth would need `Bench2Drive-Map-V0.0.4`, a separate
decision.

Last verified: 2026-09-22
