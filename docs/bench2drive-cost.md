# Bench2Drive closed-loop cost

Read this when you need to know what a Bench2Drive closed-loop evaluation costs in wall-clock time,
where that time goes, or how to run one without losing the run to a crash.
[carla.md](carla.md) covers getting a CARLA server up at all; this doc is about the loop.

**One-line answer: a 220-route round is about an hour and a half of wall clock on this one card
once the agent is wired to the simulator properly, against roughly 15 hours if it is wired the way
Bench2Drive ships - and in neither case is our model what you are paying for.**

## How these numbers were made

Everything here runs the unmodified Bench2Drive 0.0.4 leaderboard on a real route from
`bench2drive220.xml`. Nothing in the checkout is edited; the instrumentation and every optimisation
is a monkeypatch behind a flag (`scripts/b2d_hooks.py`), so the unoptimised path stays reproducible
and any measurement can be re-run against it.

| Script | What it is |
|---|---|
| `scripts/b2d_route.py` | one route, one process, per-phase timers, optional `--cprofile` |
| `scripts/b2d_agent.py` | the stand-in policy: camera rig, inference cost, control rate, overlap |
| `scripts/b2d_hooks.py` | the instrumented tick loop and the optimisation flags |
| `scripts/b2d_run.py` | many routes: server pool, watchdog, retries, resume, summary |
| `scripts/b2d_sweep.py` | one route under each configuration, printing the comparison table |
| `scripts/b2d_policy_server.py` | real frozen features (DINOv2 / Qwen3-VL) in the project's 3.11 env |

The tick loop is timed by a line-for-line copy of `ScenarioManager._tick_scenario` carrying a
`perf_counter` around each phase. The copy is guarded by the md5 of the original source, so if
Bench2Drive changes the loop the run aborts instead of profiling something that no longer matches.
Image deserialisation is timed inside the CARLA client worker thread where it actually happens.

Unless stated: route 24240 (Town10HD), 300 ticks after 20 warmup ticks, 20 Hz synchronous,
`-quality-level=Epic`, one CARLA instance, an otherwise idle box.

**Noise band: the same configuration measured three times gave 158.8, 170.6 and 170.4 ms per tick,
a spread of 7%.** Nothing below about 8% in the tables below is a result.

## The profile: where a tick goes

Three cameras at 1600x900 (our planner's rig), no policy in the loop. Percentages are of the
158-170 ms tick.

| Phase | ms/tick | share | what it is | how measured |
|---|---:|---:|---|---|
| waiting for sensor data | 125-135 | 79% | `SensorInterface.get_data` blocking until all three frames for this tick have arrived from the server | timer inside the agent's `__call__` |
| scenario tree | 24-26 | 15% | `scenario_tree.tick_once()`: py_trees behaviours and criteria | timer around `tick_once` |
| `world.tick()` | 7-9 | 5% | the synchronous step itself, up to the server acknowledging the frame | timer around `world.tick` |
| image deserialise + copy | 10-13 | (concurrent) | `np.frombuffer` + `copy.deepcopy` in the client's worker threads, summed over three cameras | timer inside the patched `_parse_image_cb` |
| `CarlaDataProvider.on_carla_tick` | 0.35 | 0.2% | actor cache update | timer |
| `apply_control`, snapshot, spectator | 0.08 | 0.05% | three RPCs | timers |

The copy is concurrent with the wait, which is why the column does not add up: it happens in a
CARLA client thread while the main thread is blocked.

**Read it as: four fifths of a tick is one blocking wait for camera frames, and the client is
almost idle for it.** That is the whole story of the closed loop's cost.

## Cameras cost per sensor, not per pixel

| Rig | Resolution | ms/tick | sensor wait | MiB/300 ticks | real time |
|---|---|---:|---:|---:|---:|
| no cameras | - | 29.1 | 0.4 | 0 | 1.72x |
| 1 camera | 1600x900 | 92.4 | 58.7 | 1703 | 0.54x |
| 3 cameras (ours) | 1600x900 | 158.3 | 128.0 | 5109 | 0.32x |
| 6 cameras (UniAD/VAD) | 1600x900 | 263.3 | 227.1 | 10217 | 0.19x |
| 3 cameras | 800x450 | 150.9 | 117.5 | 1277 | 0.33x |
| 3 cameras | 400x225 | 153.4 | 120.3 | 319 | 0.33x |

Two things follow, and the second one is the useful one.

**Each camera costs about 33 ms of wall clock per tick, and the first one costs 63.** Six cameras
are a 0.19x simulation before anything thinks.

**Resolution is worth nothing. A sixteenth of the pixels is 3% faster** (153.4 vs 158.3, inside the
noise band), while the bytes crossing the socket fall by 16x (5109 -> 319 MiB). So the cost is not
fill rate, not the wire and not the Python side: it is a fixed per-sensor, per-frame cost inside
CARLA - a render pass and a GPU readback per camera per tick, whose latency does not care how big
the image is. This matches what the feasibility run saw from the other side (`carla.md`: four times
fewer pixels bought 9%), measured here in a loop that actually consumes the frames.

The consequence is the one lever that works: **you cannot make a camera frame cheaper, so render
fewer of them.**

## What paid, what did not

Each change measured against its own baseline on the same server, same route, same tick budget.

| Change | Before | After | Effect | Verdict |
|---|---:|---:|---|---|
| run the policy every 2nd tick (`--decimate 2`) | 170.6 | 96.8 | **1.76x** | take |
| every 4th tick (`--decimate 4`) | 170.6 | 80.5 | **2.12x** | take |
| every 10th tick (`--decimate 10`) | 170.6 | 50.8 | **3.36x** | take if the policy can |
| render at the model's input size (800x450 vs 1600x900) | 80.5 | 62.3 | 1.29x, and 2.3x off the policy | **take, largest single win** |
| `np.frombuffer(...).copy()` instead of `copy.deepcopy` | 158.3 | 169.0 | none | reject |
| no copy at all: hand out a view of the CARLA buffer | 158.3 | 165.7 | none in wall clock; 12.4 ms -> 0.13 ms of CPU | take for CPU, not for speed |
| stop moving the spectator camera (2 RPCs/tick) | 158.3 | 165.9 | none (the RPCs cost 0.06 ms) | reject |
| cache the street lights (see below) | 157.9 | 193.4 | tree 25.0 -> 7.8 ms, wall clock worse | reject at one instance |
| overlap inference with simulation | 167.4 | 165.1 | hides all of a 129 ms policy | take |

### Decimation only works after fixing a leaderboard bug

Running the policy every Nth tick is worth nothing on its own: the cameras still render 20 times a
second and the frames still arrive. Measured, `--decimate 2/4/10` moved exactly the same 5109 MiB
as every tick and did not change the wall clock at all - **the wait simply moved out of the sensor
queue and into the next `world.tick()`**, which is itself a good demonstration that the loop is
paced by the server rather than by anything Python does.

The cause is `AgentWrapper._preprocess_sensor_spec` (leaderboard), which builds each sensor's
blueprint attributes from a hard-coded whitelist per sensor type. `sensor_tick` is not in that
whitelist for any sensor, so an agent asking for its cameras at 5 Hz is silently given 20 Hz. With
that one attribute passed through (`b2d_hooks._patch_sensor_tick`), the bytes fall exactly as
expected (5109 -> 2554 -> 1285 -> 511 MiB) and so does the time.

Decimation is a change to the agent, not to the evaluation: the simulation still steps at 20 Hz,
the scenario logic still runs every tick and control is still applied every tick. What changes is
that the policy sees the world at its own control rate, which is what it does in a vehicle.

### Rendering at the model's input size is free on one side and large on the other

Because CARLA's per-camera cost does not depend on resolution, asking for 800x450 frames instead of
1600x900 costs the simulator nothing measurable. It costs the policy a great deal less, because the
resize disappears:

| Policy | Frames at 1600x900 | Frames at the model's size | of which forward |
|---|---:|---:|---:|
| Qwen3-VL-4B, 3 cameras, 800 px, layer 18 | 139.0 ms | **59.9 ms** | 38.5 ms |
| DINOv2-base, 1 camera | 21.9 ms | **5.3 ms** | 3.4 ms |

Same tokens, same forward, same features - the difference is entirely the PIL resize of frames that
never needed to be that big. Note what this says about our own published latencies: the 127-216
ms/frame in [waymo-e2e.md](waymo-e2e.md) is offline throughput, with preprocessing hidden in
DataLoader workers. In a closed loop the preprocessing is on the critical path and is **70% of the
latency** at 1600x900 input, 85% for DINOv2.

The caveat to state in any paper: a frame rendered at 800x450 is not the same image as a 1600x900
frame downsampled to 800x450 (no box filter, different aliasing). Train and evaluate on the same
one.

### Why a compiled helper is not the answer

`--cprofile` on the optimised configuration puts one function at the top of the whole run:
`RouteLightsBehavior._turn_close_lights_on` in scenario_runner, **5.3 s of a 36.8 s route, 17.8 ms
of a 71 ms tick, 14.5% of total wall clock** and about three quarters of the scenario tree. It
re-fetches every street light in the map over RPC on every tick, measures each one's distance in a
Python loop, and re-sends `turn_on`/`turn_off` for lights already in that state; the vehicle half
asks the server for locations `CarlaDataProvider` already cached this tick.

So the hot path is one function, and what it spends its time on is RPCs for data that does not
change - which Cython would not touch. Caching it (`--cache-lights`, semantically identical: the
same lights end up on) does exactly what it should to the Python: **the scenario tree falls from
25.0 to 7.8 ms per tick, a 3.2x reduction.**

And the wall clock does not improve. It gets slightly worse: 157.9 -> 193.4 ms at one instance,
62.3 -> 67.6 ms in the decimated configuration, with the removed Python time reappearing almost
one-for-one in the blocking wait. The leaderboard's per-tick Python work was already hidden behind
the sensor pipeline; removing it just means arriving at the same wall sooner.

**Conclusion for H4: the ceiling on optimising the leaderboard's Python is zero wall-clock at one
instance.** The flag is kept because at four instances the freed CPU may be worth something, but
that is untested and should not be assumed. This is the clearest result in this document about
which optimisations are worth writing: the traditional one (compile the hot Python) was measurably
pointless here, and the two that paid were about not asking the simulator for work at all.

## With a real policy in the loop

A `time.sleep` stand-in leaves the GPU idle; the real model competes with CARLA's renderer for the
same card. Both are reported, because the difference is the point. Policy runs in a separate
process (the CARLA client wheel is Python 3.8, torch is 3.11) and is reached over a unix socket.

| Policy | Configuration | ms/tick | real-time ratio | inference visible in the tick |
|---|---|---:|---:|---:|
| none | 3 cameras, 1600x900 | 158-170 | 0.32x | - |
| 129 ms stand-in | every tick | 294.6 | 0.17x | 129.3 |
| 129 ms stand-in | decimate 4 + overlap + zero-copy | 91.9 | 0.54x | 1.5 |
| **DINOv2-base** | 1 camera, every tick, 1600x900 | 196.6 | 0.25x | 116.2 |
| **DINOv2-base** | 448x252, decimate 4, overlap | **44.5** | **1.13x** | 0.8 |
| **Qwen3-VL-4B @L18** | 3 cameras, every tick, 1600x900 | 520.9 | 0.096x | 365.2 |
| **Qwen3-VL-4B @L18** | decimate 4 + overlap + zero-copy | 85.2 | 0.59x | 9.3 |
| **Qwen3-VL-4B @L18** | + render at 800x450 | **71.1** | **0.70x** | 3.2 |

**The naive way of putting our model in the loop costs 7.3x more than the careful way** (520.9 vs
71.1 ms/tick). Three things cause the gap, in order: the policy runs at 20 Hz when it only needs
5 Hz; it blocks the simulator while it thinks instead of thinking alongside it; and it is handed
frames four times larger than its input.

Note the contention the stand-in hides. Qwen costs 139 ms standing alone on a quiet card and
365 ms/tick when it runs every tick against a CARLA server that is already using the GPU - the
`time.sleep(129)` stand-in was optimistic by 2.8x. Once the policy is decimated and overlapped,
the contention mostly disappears with it, and the stand-in and the real model agree to within 20%
(91.9 vs 85.2 ms/tick).

**A cheap backbone, done right, runs faster than real time (1.13x). Our actual model runs at 0.70x.**

## Parallelism: the ceiling is a property of the configuration, not of the box

[carla.md](carla.md) reports four instances as the operating point - 86.5 aggregate FPS against
28.2 for one - with five slower and **a sixth server segfaulting during startup**. That was measured
with one camera rendering at 1600x900 on every tick, which puts the GPU at 99% from four instances
on.

Run the optimised configuration instead (three cameras at 800x450, rendered every fourth tick, real
Qwen3-VL in the loop) and the GPU is no longer saturated, and the ceiling moves. Same routes, same
everything, only the instance count changing:

| Instances | ms/tick per instance | aggregate ticks/s | vs 4 | failures |
|---:|---:|---:|---:|---:|
| 4 | 45.8 | 87.4 | 1.00x | 0 |
| 6 | 50.2 | 119.5 | **1.37x** | 0 |
| 8 | 54.5 | 146.7 | **1.62x** | 0 |

**Six and eight instances run cleanly. The sixth server does not segfault; it segfaulted under a
saturating load.** At eight, VRAM is 59 of 96 GB, GPU utilisation about 67%, and the CPU is the
binding constraint at a load average of 23.8 against the cgroup's 25 cores - which is where the
diminishing returns come from (1.24x per-instance slowdown for 2x the instances). Eight is the
practical ceiling here, and it is a CPU ceiling.

This is the one place where the 96 GB card earns something: only because the per-instance load is
small enough that many instances fit. A card with a quarter of the memory would still hold eight of
these.

Servers must still be started staggered; four launched at once cost a world-load timeout and 17%
throughput. `b2d_run.py` serialises its launches by `--stagger-s` (default 20 s).

## What a 220-route round would cost

Measured: **44 routes, four workers, real Qwen3-VL in the loop, 0.71 h of wall clock** (42 of 44
routes finished; see the reliability section). Mean 2615 ticks and 121.7 s per route, mean 45.8 ms
per tick, and world loading is negligible because consecutive routes in a worker reuse the loaded
map.

Extrapolating to 220 routes at 575,300 ticks (220 x the measured mean):

| Configuration | Instances | aggregate ticks/s | 220 routes |
|---|---:|---:|---:|
| Bench2Drive as shipped, our 3-camera rig, Qwen3-VL every tick at 1600x900 | 4 | 5.9 | **~15 h** |
| optimised: 800x450, every 4th tick, overlapped, zero-copy | 4 | 87.4 | 1.8 h |
| optimised | 8 | 146.7 | **~1.1 h** |

The unoptimised figure is the single-instance 520.9 ms/tick of `gpu_qwen` scaled by the 3.07x
four-instance curve that configuration actually achieves; the optimised ones are measured
end-to-end. The ratio between them, on the same route and the same card, is **13.9x**.

Read that as: **a Bench2Drive round is a couple of hours, and the difference between a couple of
hours and two days is entirely in how the agent is wired to the simulator.** Four RTX PRO 6000s
divide it again, since routes shard cleanly.

Four things make the real number larger than 1.1 h, and they should be stated with it:

- **Town12 and Town13 are 151 of the 220 routes** and are far larger maps with tile streaming. None
  of them can run here. If they are twice the per-tick cost of Town01-10HD, the round is 3-4 h.
- **A competent policy drives further.** Thirteen of the 38 routes hit the leaderboard's 4000-tick
  cap with our deliberately poor stand-in driver; a policy that completes routes uses fewer ticks,
  but one that drives well and far may use more.
- **A six-camera BEV agent** (UniAD, VAD) pays 263.3 ms/tick against our 158.3 at 1600x900, and adds
  a lidar. Roughly double.
- **Restarts.** The 44-route run lost 25 minutes of worker time to hangs; see below.

## Reliability, which is the part that decides whether any of this matters

`scripts/b2d_run.py` gives one route one process and one CARLA server, and the properties were
verified by breaking a run on purpose rather than by reading the documentation.

- **A crash costs one route.** Killing a worker's CARLA server mid-route did not disturb the other
  three workers; that worker restarted its server and retried the route.
- **The watchdog watches ticks, not the route process.** When the server was killed, the route
  process stayed perfectly alive, blocked inside an RPC with a 300 s timeout. Its liveness would
  have said everything was fine. The heartbeat the tick loop writes did not advance, and the route
  was killed and retried after 90 s.
- **It also watches the server, which is the cheap half.** A dead server means the route is over
  whatever the client thinks, and the client will otherwise sit on its RPC until `client_timeout`
  expires - in the Town12 camera crashes that was most of the five to seven minutes each one cost.
  Watching the server process turns that into one poll interval, about 5 s, on every crash. The
  heartbeat check stays for what process death cannot see: a server that is alive and wedged.
- **Resume works after a crash, not only after a clean exit.** SIGKILLing the whole runner, then
  re-running the identical command: the nine finished routes were skipped, the stale claim left by
  the dead process was stolen, and only the unfinished route was redone. A third run is a no-op.
- **Its own death is cleaned up.** A SIGKILLed runner leaves its servers and route processes
  holding the ports; recorded pids are checked against `/proc` and their process groups killed on
  the next start. Never `pkill -f`, which matches our own command line.
- **The summary is built from disk**, so a resumed run still reports what its predecessor did:
  attempts per route, repeat offenders, routes that never finished, routes skipped for a missing
  map.

Two bugs in that mechanism were only found by testing it the way it will be used, which is the
argument for doing so: the runner deadlocked on its first "already done" route (a non-reentrant
lock, reachable only on a second run), and the summary reported zero restarts for a run that had
restarted a server.

### What an unattended run actually did

44 routes, four workers, 0.71 h, unattended:

| | |
|---|---|
| routes finished | 38 of 44 on the first pass, 42 of 44 after one resume |
| restarts | 12 |
| worker time lost to hangs | about 25 min of the 2.8 worker-hours |
| routes that never finished | 24224, 24841 - **and both ran first time, unretried, on a fresh server index** |

Every failure was the same one, and it was not CARLA's: a route that is still dying holds the
traffic manager's RPC port (the TM server lives in the *client* process), the next attempt fails
with a bind error before it ticks once, and because the retry reuses the same port it then hangs
for the full watchdog timeout. Four workers lost slots this way and the routes assigned to them
were abandoned after three attempts each. The fix is two lines - take the first free TM port, and
move a failing worker to the next server index - and the two "never finished" routes then completed
on the first attempt.

**So the honest statement about stability is: in 3.5 worker-hours on the base-package towns, CARLA
itself did not crash, hang or segfault once.** Every restart in that run was caused by our own port
reuse. That is a real result, but it is a narrow one - see the limits below, in particular that the
known Town12 sensor-dormancy segfault cannot be reached without AdditionalMaps.

**Report the restart count and the repeat offenders with any Bench2Drive score.** Scores aggregate
over routes, so silently dropping the routes that would not finish computes a number on a selected
subset that cannot be compared with anyone else's. This run is the example: read only
`routes_finished`, and you would report a score over 38 of 44 routes chosen by which worker happened
to break.

### Recycling a server before it dies (off by default, on purpose)

`--recycle-routes N` stops and restarts a worker's server every N route attempts however healthy it
looks. It is complementary to everything above: resume handles a crash that has happened, recycling
tries to get ahead of the slow death that precedes many of them.

**It defaults to off because the interval is a measurement we have not made.** The arithmetic sets
the price but not the benefit: a recycle costs a server start plus a world load, 30-60 s, against
about four minutes per route per worker, so recycling every route is 15-25% overhead and every
fifth route is 3-5%. Whether that buys anything depends entirely on how the failure rate grows with
the number of consecutive routes one server has served. If failures cluster after some number of
routes, recycling just before it is nearly free; if the rate is flat, recycling only costs.

Every run now records the raw material for that curve: each attempt carries the server's age in
routes served and in seconds, and `summary.json` reports `by_server_age` as attempts and failures
per age bucket. **One run is not the curve** - our own 44-route run says nothing about it, because
all 12 of its restarts were our own port reuse rather than UE4 decay. Accumulate across runs with
the port spacing fixed, then set N.

### Port slots are 50 apart

A CARLA server claims several ports above its RPC port and the traffic manager wants room of its
own, so server index *i* gets RPC `2000 + 50i` and traffic manager `8000 + 50i`, the same spacing
`scripts/carla_server.sh` uses. The 44-route run used 4, which is how two worker slots were lost to
a traffic-manager bind error.

### Multi-GPU

Route sharding, not a split simulation. Several runners can share one `--out` directory, one per
card, each with its own `--gpu-rank` and `--server-index`. A route is claimed with an `O_EXCL` lock
file before it starts, so two cards never run the same route, and `done/<id>.json` means finished on
any card. Claims are released whatever the outcome; a claim whose owner pid is dead is stolen at
once.

## What these numbers do not cover

- **Only 44 of the 220 routes can run here.** The base 0.9.15 package ships Town01-05 and Town10HD
  only - `client.get_available_maps()` says so, and Town06 and Town07 are *not* in it, contrary to
  the obvious assumption. The other 176 routes need AdditionalMaps (6.9 GB), which is deliberately
  not downloaded. Town12 and Town13, which are 151 of the 220, are much larger maps with tile
  streaming; their per-tick cost is unknown and the extrapolation above assumes they behave like
  Town01-10HD, which is optimistic.
- **The Town12 sensor-dormancy segfault could not be reached**, because Town12 is not installed.
  It remains an open risk for 104 of the 220 routes (decisions 16).
- **One route, one map, for the profile.** Route 24240 in Town10HD. The scenario tree cost in
  particular depends on the route's scenarios and on whether it is a night route.
- **The stand-in agent drives badly on purpose.** It holds 6 m/s and steers at the next route
  waypoint. Cost per tick is what is being measured, not driving quality; but a competent policy
  drives further per route and therefore takes more ticks, so a real evaluation is longer than the
  extrapolation from these routes at the same ms/tick.
- **`--cache-lights` is measured at one instance only** and rejected there. Whether the CPU it
  frees is worth anything at eight instances, where the CPU is what binds, is untested and is the
  obvious next measurement.
- **No lidar or radar.** The Bench2Drive rig for UniAD/VAD adds a 64-channel lidar, which is
  another sensor on the same per-sensor cost and is not in any number here.
- **The stability result is 3.5 worker-hours, not overnight.** No CARLA crash, hang or segfault in
  that window on Town01-10HD; the known failure at hour three of a long single-GPU run
  (decisions 16) is not excluded by it.
- **Nothing here says anything about driving scores.** Every number is wall clock. The stand-in
  policy consumes features and throws them away.
- **The instance sweep used the first 12 and 16 base-town routes**, compared route-by-route against
  the same routes in the four-worker run. It is not the full 44 at each instance count.
- **The policy is one process for all workers.** Inference serialises through one model on one
  card, which is the right design on one GPU but means the ladder numbers do not separate model
  latency from queueing behind other workers.

Last verified: 2026-09-21
