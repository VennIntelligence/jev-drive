# Closed-loop infrastructure acceptance

Read this before you score any model in closed loop (Bench2Drive / CARLA or HUGSIM): it says which parts of the
stack are accepted, which are not, and what a closed-loop score can and cannot be attributed to.
Working notes, pre-registrations and every table: [todos/2026-09-25-closed-loop-infra-acceptance.md](../todos/2026-09-25-closed-loop-infra-acceptance.md).

## Verdict (2026-09-25)

| Part | Verdict | Where |
|---|---|---|
| CARLA harness cost and layout | measured; GPU render binds first, **6 servers per GPU**, ~2.5 cores per worker; the container thread cap binds the box | [bench2drive-cost.md](bench2drive-cost.md) "Harness cost and layout on the five-GPU box" |
| B2D controllers (Zoo PID as used for Alpamayo / openpilot, fixed 20 Hz tracker, lateral fixes P1 / P2) | **none passes**; the fixed tracker at 2 Hz plans is closest (lags ~3 m); Zoo PID fails outright | below |
| HUGSIM controllers | official **fail**, PR #57 **fail**, fixed2 **pass** | [hugsim.md](hugsim.md) "Controller acceptance" |
| Unexplained SIGKILLs | not kernel OOM, not our code; most likely the platform's memory enforcement; forensics now armed | [long-runs.md](long-runs.md), todo `sigkill.md` |

## Bench2Drive controller acceptance

**Test.** Each controller gets a known-good plan: the PDM-Lite expert's own driven track (SimLingo's Bench2Drive copy,
`scripts/b2d_expert_agent.py`, run as shipped: DS 97.0 on the public 220), replayed by the exam agent
(`b2d_zeroshot_agent.py` `"replay"`) at the exam's camera rig, plan cadence and controller settings, with the pose from the
agent's own sensor filter. The replayed plan is the expert's track from the car's projected position at the expert's pace,
waiting as long as the expert waited (`_replay_path`). 20 pre-registered routes, TM seed 0, all arms in the same tree
(`scripts/infra_ctl_accept.sh`, scored by `scripts/infra_ctl_score.py`). Pass (pre-registered): cross-track to the expert
path median <= 0.3 m and p95 <= 1 m; |along-track lag| median <= 2 m and p95 <= 8 m; lateral execution ratio (driven /
planned lateral offset 2 s after a plan, the Alpamayo diagnosis metric) >= 0.7; mean DS >= expert - 5; no route stuck that
the expert completed. The expert run twice differs by 0.26 DS per route on average.

| Controller (plan cadence) | DS (expert 95.5) | completed / 20 (expert 19) | stuck | routes with extra collisions | cross-track p95 | lag median | lateral ratio | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Zoo PID, forward-only, 2 Hz (Alpamayo exam) | 35.0 | 6 | 13 | 3 | 0.17 m | 8.3 m | 0.32 | fail |
| Zoo PID, forward-only, 5 Hz (openpilot exam) | 30.1 | 5 | 14 | 5 | 0.21 m | 6.7 m | 0.40 | fail |
| fixed 20 Hz tracker (`b2d_controller.py` carla), 2 Hz | **83.7** | **19** | **0** | 6 | 0.58 m | 3.1 m | **0.87** | fail (longitudinal only) |
| fixed tracker, 5 Hz | 57.0 | 18 | 1 | 12 | 0.40 m | 3.2 m | 0.21 | fail |
| P1: Zoo longitudinal + fixed lateral, 2 Hz | 38.2 | 7 | 12 | 4 | 0.08 m | 7.6 m | 0.70 | fail |
| P2: Zoo longitudinal + 1.5 s-aim lateral, 2 Hz | 35.1 | 6 | 13 | 4 | 0.13 m | 8.3 m | 0.62 | fail |

- **Zoo PID** executes about a third of a plan's lateral offset and has no position feedback: its target speed is the
  spacing of the 0.5 s and 1 s waypoints, so any distance between the car and the first waypoint is never recovered
  (lag 7-8 m). Its many "stuck" routes are inflated by the replay (see limits), but these two properties alone fail it.
  Replacing only the lateral (P1, P2) does not help: the longitudinal still stalls.
- **The fixed tracker at 2 Hz** tracks laterally and finishes 19/20, but runs ~3 m behind the expert's timetable and, since
  the replayed plan does not react, reaches six conflict points late and collides. Suspected causes: it holds the brake
  until the first plan and caps throttle at 0.75 (PDM-Lite uses 1.0). Fed 5 Hz plans the same controller is much worse
  (12 routes with collisions, runs ahead of the plan on some), so it is only validated at its pre-registered 2 Hz.
- **Limits of the test.** The plan is the expert's, non-reactive: a car that lags meets a different scenario. The ±2 m
  wait window can freeze the plan when the car stops more than 2 m short of an expert stop point, which prolongs stalls of
  controllers without position feedback; a time-indexed plan with smooth catch-up (s(t) = s_e(τ + t) − (s_e(τ) − s0)·e^(−t/2 s))
  avoids both the jump and the freeze and should replace it before the next acceptance run.
- Only the fixed tracker at 2 Hz is close to usable; **no closed-loop B2D score obtained with Zoo PID (Alpamayo, openpilot)
  can be attributed to the model.**


## Box limits that decide how many closed-loop workers fit

- **GPU render** binds a card at about six CARLA servers (Town12, Alpamayo-like rig: 34.8 aggregate ticks/s at 6,
  38.2 at 12). Spread workers over cards before stacking one past six.
- **The container thread cap** (`/sys/fs/cgroup/pids.max` = 20480, counts threads) binds the box: a default worker
  holds ~650 threads (server ~430, route client ~215 because `carla.Client` starts one worker per host hardware
  thread, 208 here). It was hit 12 times on 2026-09-25 and shows up as `RuntimeError: Resource temporarily
  unavailable` at `carla.Client()`. Run route clients with `b2d_run.py --client-threads 8` (~16 threads, behaviour
  unchanged within run-to-run noise) and count threads, not only cores, when sizing a CARLA job.
- **Ports.** Server index i uses RPC port 2000 + 50i and TM port 8000 + 50i; from index 616 (RPC) and 496 (TM)
  they fall into the kernel's ephemeral range 32768-60999 and can be held by an outgoing connection, which kills
  CARLA at startup with `bind: Address already in use` + Signal 11. Prefer indices below 490 or bind-test first.
- **Setup crashes of unknown cause**: 14 of 91 server starts in the profiling died in route setup with
  `GameThread timed out waiting for RenderThread` + Signal 11, not correlated with the thread cap. Each costs one
  retry; report retries with every score.

## SIGKILL

Two single-process SIGKILLs on 2026-09-25 (openpilot Cinque policy server ~02:20; nuScenes openpilot feature
extraction 13:41) hit the largest anonymous-memory process in the container while memory sat at `memory.high`,
with `memory.events` oom / oom_kill = 0 before the next restart. Kernel OOM, our reapers, the agents' own commands,
rlimits, pids.max and SIGHUP are ruled out; the leading explanation is AutoDL enforcing the memory limit outside the
kernel. It cannot be confirmed without root. `scripts/boxwatch.sh` (5 s memory / largest-process sampler) and the
`slot_run.sh` post-mortem (`<slot>.death-*.txt`) are armed so the next kill can be attributed by correlation; keep
application memory below ~85% of the container limit and count ~27 GB per openpilot Cinque process. The reapers in
`b2d_run.py` and `b2d_tfv6_campaign.py` could kill a stranger that inherited a recorded pid (pid_max wraps in ~4 h
here); both now kill only groups that still write into their own run directory.

## Before closed-loop exams resume

1. B2D: drop Zoo PID (and P1 / P2). Start from the fixed tracker at 2 Hz, fix its longitudinal lag (no brake hold before
   the first plan, throttle authority, position feedback on the plan's time stamps) and re-run this acceptance
   (`scripts/infra_ctl_accept.sh arms <gpu> <arm>:<index>`, ~40 min for one arm). Use a controller only at the plan cadence
   it was accepted at.
2. Replace the replay's wait window with the time-indexed smooth catch-up plan before using this test again.
3. HUGSIM: switch to fixed2 (`zs_run.py --controller fixed2`); whether official stays the headline is the user's call.
4. Size CARLA jobs by threads: `--client-threads 8` on every runner, <= 6 servers per GPU, <= 30 on the box.
5. Report retries with every score (~15% of server starts die in setup with a RenderThread timeout, cause unknown).
6. Keep application memory under ~85% of the container limit; read `<slot>.death-*.txt` and boxwatch at the next kill.
7. Mark existing closed-loop scores (Alpamayo / openpilot on B2D with Zoo PID; HUGSIM under official) as
   "controller not accepted".

Last verified: 2026-09-25
