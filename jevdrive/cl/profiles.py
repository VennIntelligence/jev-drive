"""Named worker profiles: the one place where the thread flags and thread env of a closed-loop worker are chosen.

A worker is one CARLA server plus one route-client process (scripts/b2d_run.py starts both). Three knobs decide how many
threads it has and how hard it oversubscribes its core slice:

  pools          the server's three CARLA asio pools (RPC, sensor streaming, multi-GPU secondary). Stock CARLA sizes each
                 from the HOST's CPU count, (208 - 2) / 3 = 68 here, whatever the cgroup quota or affinity says;
                 "reduced" passes -RPCThreads=N -StreamingThreads=N -SecondaryThreads=N (docs/carla.md, "Threads per server").
  client_threads carla.Client worker threads in the route process; 0 = CARLA's default, one per host hardware thread.
  num_threads    OMP / MKL / OPENBLAS / NUMBA threads of the route process (the agent's numeric libraries); None = unset.

OPENBLAS_CORETYPE is not a profile knob: it is a correctness setting of particular envs (Haswell for the NAVSIM devkit
on the GPU box, Barcelona to replay Tokyo-recorded controller goldens bit for bit). Put it in a job's env.

The default was chosen by the 2026-10-01 experiment (fc65452:todos/2026-10-01-cl-lib.md, docs/closed-loop-runbook.md).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

NUM_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")


@dataclass(frozen=True)
class Profile:
    name: str
    pools: str = "reduced"        # "reduced" | "stock"
    pool_threads: int = 4
    client_threads: int = 8       # 0 = CARLA default (host hardware threads)
    num_threads: int = 2          # None = leave the numeric-library env alone
    env: tuple = ()               # extra (key, value) pairs

    def server_args(self) -> list:
        if self.pools == "stock":
            return []
        n = self.pool_threads
        return ["-RPCThreads=%d" % n, "-StreamingThreads=%d" % n, "-SecondaryThreads=%d" % n]

    def environ(self) -> dict:
        """Env for the runner and its route processes. B2D_CARLA_POOLS=stock stops b2d_run adding its reduced default."""
        e = {"B2D_CARLA_POOLS": "stock" if self.pools == "stock" else "reduced"}
        if self.num_threads is not None:
            e.update({k: str(self.num_threads) for k in NUM_ENV})
        e.update(dict(self.env))
        return e

    def with_(self, **kw) -> "Profile":
        """A copy with knobs overridden; None values are ignored (so CLI flags that were not given change nothing)."""
        kw = {k: v for k, v in kw.items() if v is not None}
        return replace(self, **kw) if kw else self

    def describe(self) -> str:
        ct = self.client_threads or "host"
        return "%s: pools %s, client threads %s, numeric threads %s" % (
            self.name, "stock" if self.pools == "stock" else "%d each" % self.pool_threads, ct,
            "unset" if self.num_threads is None else self.num_threads)


PROFILES = {
    "stock": Profile("stock", pools="stock", client_threads=0, num_threads=None),
    "reduced": Profile("reduced", pools="reduced", pool_threads=4, client_threads=8, num_threads=2),
}
DEFAULT = "reduced"


def get(name: str = None, **overrides) -> Profile:
    p = PROFILES[name or DEFAULT]
    return p.with_(**overrides)
