"""jevdrive.cl: closed-loop (CARLA / Bench2Drive) lane infrastructure. Start at docs/closed-loop-runbook.md.

  box        live probe: cgroup CPU quota, pids.max, memory, NUMA, cards (nvidia-smi, cached), CARLA servers per card
  profiles   named worker profiles: CARLA thread pools, client threads, numeric-library threads
  capacity   thread model per worker, PID admission from pids.max, per-card sizing, usable server indices
  lease      lane rows in the box's schedule table (runs/sched/table.tsv): find free cards / cores / indices, grant
  procs      owned process trees by (pid, start time); stop through pidfd, never by pattern
  sampler    util.csv: GPU, slice cores, threads by role, pids, memory
  lane       Job / b2d() declarations and the Lane scheduler (placement, retries, DONE / ERROR / STATUS, drain, reap)

CLI: python -m jevdrive.cl {probe, plan, profiles, lease, release, run, status, drain, stop} (--help on each).
"""
from .lane import Job, Lane, b2d, b2d_finished  # noqa: F401
from .profiles import PROFILES, Profile, get as profile  # noqa: F401
