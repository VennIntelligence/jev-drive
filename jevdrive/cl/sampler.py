"""util.csv: per card every `period` s, GPU util / VRAM, busy cores of the card's slice, CARLA servers on the card (any
owner), threads of this lane's processes on it by role, and the cgroup's pids / memory. The numbers a capacity
decision needs, written while the lane runs instead of reconstructed afterwards."""
from __future__ import annotations

import csv
import threading
import time
from pathlib import Path

from .box import parse_cpus, processes, smi
from .procs import tree_threads

COLS = ["t", "gpu", "util_pct", "mem_mib", "cores_busy", "slice_cores", "carla_on_card", "lane_servers", "server_threads",
        "lane_clients", "client_threads", "other_threads", "pids_current", "mem_current_gib"]


def cpu_times(root: Path = Path("/")) -> dict:
    out = {}
    for line in (root / "proc/stat").read_text().splitlines():
        if line.startswith("cpu") and line[3:4].isdigit():
            f = line.split()
            v = [int(x) for x in f[1:9]]
            out[int(f[0][3:])] = (sum(v) - v[3] - v[4], sum(v))     # (busy, total) jiffies; idle + iowait excluded
    return out


def busy(prev: dict, cur: dict, cores) -> float:
    return sum((cur[i][0] - prev[i][0]) / max(cur[i][1] - prev[i][1], 1) for i in cores if i in cur and i in prev)


class Sampler(threading.Thread):
    """`slices()` -> {card: core list string}; `records()` -> {card: [owned-process records]} (procs.capture format)."""

    def __init__(self, path: Path, slices, records, period: float = 15.0, root: Path = Path("/"), tb=None):
        super().__init__(daemon=True)
        self.path, self.slices, self.records, self.period, self.root, self.tb = Path(path), slices, records, period, root, tb
        self.halt = threading.Event()

    def sample(self, prev: dict) -> tuple:
        cur = cpu_times(self.root)
        rows = processes(self.root)
        g = {}
        for line in smi("gpu=index,utilization.gpu,memory.used", max_age=self.period / 2):
            f = [x.strip() for x in line.split(",")]
            if len(f) == 3 and f[0].isdigit():
                g[int(f[0])] = (int(f[1]) if f[1].isdigit() else 0, int(f[2]) if f[2].isdigit() else 0)
        carla = {}
        for r in rows.values():
            a = r["argv"]
            if a and "CarlaUE4-Linux-Shipping" in a[0]:
                k = next((int(x.split("=")[1]) for x in a if x.startswith("-graphicsadapter=")), -1)
                carla[k] = carla.get(k, 0) + 1

        def cg(name):
            try:
                return int((self.root / "sys/fs/cgroup" / name).read_text())
            except (OSError, ValueError):
                return 0
        pids, mem = cg("pids.current"), cg("memory.current") / 2 ** 30
        out, now = [], round(time.time(), 1)
        recs = self.records()
        for card, spec in sorted(self.slices().items()):
            cores = parse_cpus(spec) if spec else []
            t = {"servers": 0, "server_threads": 0, "clients": 0, "client_threads": 0, "other_threads": 0}
            for rec in recs.get(card, []):
                for k, v in tree_threads(rec, rows).items():
                    t[k] += v
            u, m = g.get(card, (0, 0))
            out.append([now, card, u, m, round(busy(prev, cur, cores), 2) if prev else "", len(cores), carla.get(card, 0),
                        t["servers"], t["server_threads"], t["clients"], t["client_threads"], t["other_threads"], pids,
                        round(mem, 1)])
        return cur, out

    def run(self):
        new = not self.path.exists()
        prev = {}
        with self.path.open("a", buffering=1) as f:
            w = csv.writer(f)
            if new:
                w.writerow(COLS)
            step = 0
            while not self.halt.is_set():
                try:
                    prev, rows = self.sample(prev)
                    if step:                       # the first sample has no CPU baseline
                        w.writerows(rows)
                        if self.tb is not None:
                            for r in rows:
                                self.tb.add_scalar("gpu%d/util_pct" % r[1], r[2], step)
                                self.tb.add_scalar("gpu%d/cores_busy" % r[1], r[4] or 0, step)
                            self.tb.add_scalar("box/pids_current", rows[0][12] if rows else 0, step)
                except Exception as e:             # noqa: BLE001 - sampling never stops the lane
                    with self.path.with_suffix(".err").open("a") as err:
                        err.write("%s %r\n" % (time.strftime("%F %T"), e))
                step += 1
                self.halt.wait(self.period)
