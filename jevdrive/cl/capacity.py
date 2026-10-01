"""Capacity model: threads per worker, PID admission and per-card sizing, all derived from the live probe.

One set of constants replaces the ones the lane scripts disagreed on (sch_table 16000 / 400 per worker, b2d_run waits at
17000, nq4_g 16000 / 700, G lane 250, privileged chain 17500 ...): the caps are fractions of the measured pids.max and
the per-worker cost comes from the thread model below, evaluated for the actual profile and core slice.
"""
from __future__ import annotations

from dataclasses import dataclass

from .profiles import Profile

# Threads of one CARLA 0.9.15 server with reduced pools (4 each) at an affinity of C cores. UE4 sizes its TaskGraph and
# PoolThread pools from the affinity mask, so the count follows the slice. Measured: 8 / 16 / 32 cores 2026-09-27
# (scripts/carla_threads.py), 24 / 128 cores in the G lane 2026-09-28 (docs/carla.md), 25 cores 2026-10-01 (this lane).
REDUCED_SERVER_THREADS = ((8, 69), (16, 109), (24, 149), (32, 169), (128, 263))
# Route client: ~21 threads of its own plus carla.Client's worker threads (29 at --client-threads 8, ~215 at the default
# on the 208-thread host, 2026-09-27). An agent adds its own (SimLingo ~40 more; PDM-Lite measured 2026-10-01).
CLIENT_BASE_THREADS = 21
AGENT_THREADS = 40
# Admission: plan new workers up to PLAN_FRACTION of pids.max; b2d_run holds a server start while pids.current is above
# WAIT_FRACTION (B2D_PIDS_WAIT). At pids.max 20480 these are 16384 and 17408 (the old hand-set 16000 and 17000).
PLAN_FRACTION, WAIT_FRACTION = 0.80, 0.85
VRAM_HEADROOM_GB = 8.0            # left free on every card (a CARLA server holds 3-10 GB, a Town12 one ~6.3)
# Per-card defaults (2026-10-01 experiment, docs/closed-loop-runbook.md): the GPU knee for camera agents is ~6 servers
# per card (2026-09-25); CPU-bound agents are limited by cores per worker instead.
GPU_KNEE = 6
CORES_PER_WORKER = 3.0
VRAM_PER_WORKER_GB = 9.0
# Server indices: RPC 2000 + 50 i, traffic manager 8000 + 50 i .. +49. Bench2Drive's README warns against ports < 10000,
# and ports in the kernel's ephemeral range can be held by an outgoing connection (bind error, Signal 11 at start-up).
PORT_BASE, TM_BASE, PORT_STRIDE = 2000, 8000, 50
MIN_PORT = 10000


def _interp(points, x):
    pts = sorted(points)
    if x <= pts[0][0]:
        (x0, y0), (x1, y1) = pts[0], pts[1]
    elif x >= pts[-1][0]:
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
    else:
        (x0, y0), (x1, y1) = next((a, b) for a, b in zip(pts, pts[1:]) if a[0] <= x <= b[0])
    return y0 + (y1 - y0) * (x - x0) / float(x1 - x0)


def server_threads(cores: float, profile: Profile, host_cpus: int) -> int:
    """Threads of one server pinned to `cores` CPUs. Stock pools add 3 x ((host - 2) // 3 - 4) over the reduced count."""
    t = _interp(REDUCED_SERVER_THREADS, max(cores, 1))
    if profile.pools == "stock":
        t += 3 * ((max(host_cpus, 4) - 2) // 3) - 12
    else:
        t += 3 * (profile.pool_threads - 4)
    return int(round(t))


def client_threads(profile: Profile, host_cpus: int, agent: int = AGENT_THREADS) -> int:
    return CLIENT_BASE_THREADS + (profile.client_threads or host_cpus) + agent


def worker_threads(profile: Profile, cores: float, host_cpus: int, agent: int = AGENT_THREADS) -> int:
    """Threads (= PIDs in the cgroup's count) one worker adds: its server plus its route client."""
    return server_threads(cores, profile, host_cpus) + client_threads(profile, host_cpus, agent)


@dataclass(frozen=True)
class Admission:
    pids_max: int
    plan_cap: int     # do not plan workers beyond this many PIDs
    wait_cap: int     # b2d_run waits before a server start while pids.current is above this (B2D_PIDS_WAIT)

    @classmethod
    def from_box(cls, box) -> "Admission":
        m = box.pids_max or 1 << 22          # no limit: effectively unbounded
        return cls(m, int(m * PLAN_FRACTION), int(m * WAIT_FRACTION))

    def room(self, pids_current: int, per_worker: int, pending: int = 0) -> int:
        """New workers that fit, counting `pending` workers already launched but not yet visible in pids.current."""
        return max(0, (self.plan_cap - pids_current) // max(per_worker, 1) - pending)


def index_bounds(ephemeral_lo: int = 32768) -> tuple:
    """Usable server indices: RPC >= MIN_PORT and the whole TM block below the ephemeral range."""
    lo = -(-(MIN_PORT - PORT_BASE) // PORT_STRIDE)
    hi = (ephemeral_lo - TM_BASE - PORT_STRIDE) // PORT_STRIDE
    return lo, hi


def workers_per_card(cores_per_card: float, card_total_gb: float, cores_per_worker: float = CORES_PER_WORKER,
                     vram_per_worker_gb: float = VRAM_PER_WORKER_GB, knee: int = GPU_KNEE) -> int:
    """Default workers on one card: the GPU knee, the core slice and the card's VRAM, whichever binds first."""
    by_cpu = int(cores_per_card / cores_per_worker + 1e-9)
    by_vram = int((card_total_gb - VRAM_HEADROOM_GB) / vram_per_worker_gb)
    return max(1, min(knee, by_cpu, by_vram))


def plan(box, profile: Profile, cards: list = None, cores_per_card: float = None, workers: int = None,
         agent_threads: int = AGENT_THREADS) -> dict:
    """The sizing a lane on `cards` (default: every card) would get, with the numbers it was derived from."""
    cards = [box.card(i) for i in cards] if cards else list(box.cards)
    n = max(len(box.cards), 1)
    cpc = cores_per_card or (box.cores / n)
    total_gb = min((c.mem_total_mib for c in cards), default=0) / 1024
    w = workers or workers_per_card(cpc, total_gb)
    per = worker_threads(profile, cpc, box.host_cpus, agent_threads)
    adm = Admission.from_box(box)
    return dict(profile=profile.describe(), cards=[c.index for c in cards], cores_per_card=round(cpc, 1),
                workers_per_card=w, cores_per_worker=round(cpc / w, 2), threads_per_worker=per,
                server_threads=server_threads(cpc, profile, box.host_cpus),
                pids_plan_cap=adm.plan_cap, pids_wait_cap=adm.wait_cap,
                workers_admitted_by_pids=adm.room(box.pids_current, per), index_bounds=index_bounds(box.ephemeral[0]))
