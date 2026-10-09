"""1 s memory sampler for the SIGKILL study (INFRA2, 2026-10-10): what boxwatch.sh does not record.

One JSON line a second: the container's memory.current, the memory.stat split (anon, file, active / inactive file,
shmem, kernel, reclaim counters), memory.events high, memory.pressure some / full totals, pids, the HOST's
/proc/meminfo (MemFree, MemAvailable, Cached, AnonPages, Committed_AS ...: /proc/meminfo is not virtualised here) and per-node free
memory, and the five largest processes by RSS (pid, rss kB, age s, comm).

    python experiments/cl_infra/scripts/memwatch.py OUT.jsonl [--dt 1] [--hours 6]
"""
import argparse, json, subprocess, time
from pathlib import Path

CG = Path("/sys/fs/cgroup")
STAT = ("anon", "file", "shmem", "kernel", "active_file", "inactive_file", "active_anon", "inactive_anon",
        "file_mapped", "file_dirty", "pgscan", "pgsteal", "pgactivate", "pgdeactivate", "workingset_refault_file")
HOST = ("MemFree", "MemAvailable", "Cached", "AnonPages", "Active(file)", "Inactive(file)", "Shmem", "Dirty",
        "Committed_AS", "CommitLimit", "Mlocked", "Unevictable", "PageTables")


def kv(path, keys):
    out = {}
    for line in Path(path).read_text().split("\n"):
        p = line.replace(":", " ").split()
        if p and p[0] in keys:
            out[p[0]] = int(p[1])
    return out


def sample():
    d = {"t": round(time.time(), 2), "cur": int((CG / "memory.current").read_text())}
    d.update(kv(CG / "memory.stat", STAT))
    d["high_ev"] = kv(CG / "memory.events", ("high",)).get("high")
    for line in (CG / "memory.pressure").read_text().split("\n"):
        if line:
            d["psi_" + line.split()[0]] = int(line.rsplit("total=", 1)[1])
    d["pids"] = int((CG / "pids.current").read_text())
    d["host"] = kv("/proc/meminfo", HOST)  # kB
    d["node_free"] = [int(next(l for l in (p / "meminfo").read_text().split("\n") if "MemFree" in l).split()[3])
                      for p in sorted(Path("/sys/devices/system/node").glob("node[0-9]*"))]
    ps = subprocess.run("ps -eo pid=,rss=,etimes=,comm= --sort=-rss | head -5", shell=True, capture_output=True,
                        text=True).stdout
    d["top"] = [[int(a[0]), int(a[1]), int(a[2]), a[3]] for a in (l.split(None, 3) for l in ps.split("\n")) if len(a) == 4]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--hours", type=float, default=6.0)
    a = ap.parse_args()
    end = time.time() + a.hours * 3600
    with open(a.out, "a", buffering=1) as f:
        while time.time() < end:
            t = time.time()
            try:
                f.write(json.dumps(sample(), separators=(",", ":")) + "\n")
            except Exception as e:  # a vanished /proc entry must not stop the sampler
                f.write(json.dumps({"t": t, "err": repr(e)}) + "\n")
            time.sleep(max(0.0, a.dt - (time.time() - t)))


if __name__ == "__main__":
    main()
