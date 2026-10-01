"""Live resource probe. The box is elastic (1 to 7 cards, a CPU quota and pids.max that change with the instance), so
everything that sizes a closed-loop lane reads it here at run time instead of trusting a number in a doc.

Every path is read under `root` (default "/"), so tests run against a fake tree, and nvidia-smi goes through
`smi()`, which caches each query for 60 s: every call takes the NVIDIA driver's device lock, and on 2026-09-28 per-round
queries from several lanes helped ~50 starting CARLA servers pile up on it (docs/carla.md, "Traps").
"""
from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

CARLA_BIN = "CarlaUE4-Linux-Shipping"


def parse_cpus(spec: str) -> list:
    """'0-3,8,10-11' -> [0, 1, 2, 3, 8, 10, 11]."""
    out = []
    for part in spec.replace(" ", "").split(","):
        if part:
            a, _, b = part.partition("-")
            out.extend(range(int(a), int(b or a) + 1))
    return out


def format_cpus(cpus) -> str:
    """[0, 1, 2, 3, 8] -> '0-3,8' (sorted, deduplicated)."""
    xs, runs = sorted(set(cpus)), []
    for c in xs:
        if runs and c == runs[-1][1] + 1:
            runs[-1][1] = c
        else:
            runs.append([c, c])
    return ",".join(str(a) if a == b else "%d-%d" % (a, b) for a, b in runs)


def _read(root: Path, rel: str, default=None):
    try:
        return (root / rel.lstrip("/")).read_text().strip()
    except OSError:
        return default


_SMI = {}


def smi(query: str, max_age: float = 60.0) -> list:
    """`nvidia-smi --query-<query> --format=csv,noheader,nounits` lines, at most one real call per query per max_age."""
    t, lines = _SMI.get(query, (0.0, None))
    if lines is None or time.time() - t > max_age:
        try:
            lines = subprocess.run(["nvidia-smi", "--query-" + query, "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=60).stdout.strip().splitlines()
        except (OSError, subprocess.TimeoutExpired):
            lines = []
        _SMI[query] = (time.time(), lines)
    return lines


@dataclass
class Card:
    index: int
    uuid: str
    bus: str                      # PCI bus id, sysfs form (0000:3b:00.0)
    mem_used_mib: int
    mem_total_mib: int
    util: int
    numa: int = -1                # NUMA node of the card (-1: unknown)
    compute_pids: list = field(default_factory=list)
    carla: int = 0                # CARLA servers rendering on it (-graphicsadapter, any owner)

    @property
    def free_gb(self) -> float:
        return (self.mem_total_mib - self.mem_used_mib) / 1024


@dataclass
class Box:
    host_cpus: int                # online logical CPUs of the host: what CARLA and carla.Client size their pools from
    quota_cores: float            # cgroup cpu.max (0 = no quota)
    affinity: list                # CPUs this process may run on
    numa: dict                    # node -> CPUs
    siblings: dict                # cpu -> its hyperthread siblings (incl. itself)
    pids_max: int                 # cgroup pids.max (0 = no limit)
    pids_current: int
    mem_max_gb: float             # cgroup memory.max (0 = no limit)
    mem_current_gb: float
    ephemeral: tuple              # net.ipv4.ip_local_port_range
    cards: list
    load: list = field(default_factory=list)
    time: float = 0.0

    @property
    def cores(self) -> float:
        """CPU time the container may use, in cores: the quota if set, else the affinity mask."""
        return self.quota_cores or float(len(self.affinity))

    def card(self, index: int) -> Card:
        return next(c for c in self.cards if c.index == index)

    def primary_cpus(self, node: int = -1) -> list:
        """One CPU per physical core (the lowest sibling), in the affinity mask, on `node` (-1: all nodes)."""
        allowed = set(self.affinity) & (set(self.numa.get(node, [])) if node >= 0 else set(self.affinity))
        return sorted(c for c in allowed if min(self.siblings.get(c, [c])) == c)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cores"] = self.cores
        return d


def processes(root: Path = Path("/")) -> dict:
    """pid -> {argv, threads, start_ticks, ppid, pgid}, non-zombie processes only."""
    out = {}
    proc = root / "proc"
    for p in proc.glob("[0-9]*"):
        try:
            s = (p / "stat").read_text().rsplit(")", 1)[1].split()
            if s[0] == "Z":
                continue
            argv = (p / "cmdline").read_bytes().decode(errors="replace").split("\0")
            out[int(p.name)] = dict(pid=int(p.name), ppid=int(s[1]), pgid=int(s[2]), sid=int(s[3]),
                                    threads=int(s[17]), start_ticks=s[19], argv=[a for a in argv if a])
        except (OSError, IndexError, ValueError):
            continue
    return out


def carla_servers(rows: dict) -> list:
    """(pid, card, rpc_port) of every live CARLA server binary, whoever owns it."""
    out = []
    for pid, r in rows.items():
        a = r["argv"]
        if a and CARLA_BIN in a[0]:
            opt = dict(x.split("=", 1) for x in a if x.startswith("-") and "=" in x)
            out.append((pid, int(opt.get("-graphicsadapter", -1)), int(opt.get("-carla-rpc-port", 2000))))
    return out


def probe(root: Path = Path("/"), rows: dict = None, query=smi) -> Box:
    root = Path(root)
    cpu_max = (_read(root, "sys/fs/cgroup/cpu.max", "max 100000") or "max 100000").split()
    quota = 0.0 if cpu_max[0] == "max" else int(cpu_max[0]) / int(cpu_max[1])
    online = _read(root, "sys/devices/system/cpu/online")
    host = len(parse_cpus(online)) if online else (os.cpu_count() or 1)
    status = _read(root, "proc/self/status", "") or ""
    m = re.search(r"Cpus_allowed_list:\s*(\S+)", status)
    affinity = parse_cpus(m.group(1)) if m else list(range(host))
    numa = {}
    for d in sorted((root / "sys/devices/system/node").glob("node[0-9]*")):
        cl = _read(root, str(d.relative_to(root) / "cpulist"))
        if cl:
            numa[int(d.name[4:])] = parse_cpus(cl)
    siblings = {}
    for c in affinity:
        sl = _read(root, "sys/devices/system/cpu/cpu%d/topology/thread_siblings_list" % c)
        siblings[c] = parse_cpus(sl) if sl else [c]

    def num(rel, default=0):
        v = _read(root, rel)
        return default if v in (None, "max") else int(v)

    eph = (_read(root, "proc/sys/net/ipv4/ip_local_port_range", "32768 60999") or "32768 60999").split()
    rows = processes(root) if rows is None else rows
    cards, by_uuid = [], {}
    for line in query("gpu=index,uuid,pci.bus_id,memory.used,memory.total,utilization.gpu"):
        f = [x.strip() for x in line.split(",")]
        if len(f) < 6 or not f[0].isdigit():
            continue
        bus = f[2].lower()[-12:]
        node = _read(root, "sys/bus/pci/devices/%s/numa_node" % bus)
        c = Card(int(f[0]), f[1], bus, int(float(f[3])), int(float(f[4])), int(float(f[5]) if f[5][:1].isdigit() else 0),
                 int(node) if node not in (None, "") else -1)
        cards.append(c)
        by_uuid[c.uuid] = c
    for line in query("compute-apps=gpu_uuid,pid"):
        f = [x.strip() for x in line.split(",")]
        if len(f) == 2 and f[0] in by_uuid and f[1].isdigit():
            by_uuid[f[0]].compute_pids.append(int(f[1]))
    for _, g, _ in carla_servers(rows):
        for c in cards:
            if c.index == g:
                c.carla += 1
    return Box(host_cpus=host, quota_cores=quota, affinity=affinity, numa=numa, siblings=siblings,
               pids_max=num("sys/fs/cgroup/pids.max"), pids_current=num("sys/fs/cgroup/pids.current"),
               mem_max_gb=num("sys/fs/cgroup/memory.max") / 2 ** 30, mem_current_gb=num("sys/fs/cgroup/memory.current") / 2 ** 30,
               ephemeral=(int(eph[0]), int(eph[1])), cards=sorted(cards, key=lambda c: c.index),
               load=(_read(root, "proc/loadavg", "") or "").split()[:3], time=time.time())
