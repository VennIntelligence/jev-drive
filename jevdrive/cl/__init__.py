"""jevdrive.cl: closed-loop (CARLA / Bench2Drive) and GPU-job infrastructure on the box. Start at docs/closed-loop-runbook.md.

  pool       the GPU pool: submit jobs, one dispatcher (tmux jev:pool) runs each on any card with room (VRAM, CARLA
             servers, server-index blocks, cores, PIDs), with DONE / ERROR / STATUS, deps, retries, cancel, safe cleanup
  box        live probe: cgroup CPU quota, pids.max, memory, NUMA, cards (nvidia-smi, cached), CARLA servers per card
  profiles   named worker profiles: CARLA thread pools, client threads, numeric-library threads
  capacity   thread model per worker, PID admission from pids.max, per-card sizing, usable server indices
  procs      owned process trees by (pid, start time); stop through pidfd, never by pattern

CLI: python -m jevdrive.cl {submit, queue, show, cancel, top, hold, holds, unhold, dispatch, probe, plan, profiles}.
"""
from .profiles import PROFILES, Profile, get as profile  # noqa: F401
