#!/usr/bin/env python
"""Per-CARLA-instance CPU / RAM sampler for the P5 v1 profiling (todos/2026-09-25-reactivity-program/i1-p5v1.md).

Every --every seconds, for the CARLA servers whose RPC port is in [--port-lo, --port-hi] and the b2d_route.py processes
talking to them: CPU cores used since the last sample (utime + stime from /proc, children included once they are
reaped), RSS, and the card's used VRAM. One TSV row per instance per sample; `--summary <tsv>` prints median / p95.
Runs until killed (SIGTERM) or until --until-file exists.

    python scripts/p5v1_prof_sampler.py --out prof.tsv --port-lo 46000 --port-hi 46450 --gpu 2 --until-file done
    python scripts/p5v1_prof_sampler.py --summary prof.tsv
"""
import argparse
import os
import re
import subprocess
import time

HZ = os.sysconf("SC_CLK_TCK")


def procs():
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open("/proc/%s/cmdline" % d, "rb") as fh:
                cmd = fh.read().replace(b"\0", b" ").decode(errors="replace")
            with open("/proc/%s/stat" % d) as fh:
                st = fh.read().rsplit(")", 1)[1].split()
            yield int(d), cmd, int(st[11]) + int(st[12]), int(st[21]) * os.sysconf("SC_PAGE_SIZE")
        except (OSError, IndexError, ValueError):
            continue


def instance_of(cmd, lo, hi):
    """(kind, rpc port) for our CARLA servers and route processes, else None."""
    m = re.search(r"-carla-rpc-port=(\d+)", cmd)
    if m and "CarlaUE4-Linux-Shipping" in cmd and lo <= int(m.group(1)) <= hi:
        return "server", int(m.group(1))
    m = re.search(r"b2d_route\.py .*--port (\d+)", cmd)
    if m and lo <= int(m.group(1)) <= hi:
        return "route", int(m.group(1))
    return None


def vram_mib(gpu):
    try:
        return int(subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used",
                                            "--format=csv,noheader,nounits"], text=True).split()[0])
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return -1


def sample(a):
    last, t_last = {}, time.time()
    with open(a.out, "a", buffering=1) as fh:
        fh.write("t\tport\tserver_cores\troute_cores\tserver_rss_mb\troute_rss_mb\tvram_mib\n")
        while not (a.until_file and os.path.exists(a.until_file)):
            time.sleep(a.every)
            now = time.time()
            per = {}
            seen = {}
            for pid, cmd, ticks, rss in procs():
                k = instance_of(cmd, a.port_lo, a.port_hi)
                if k is None:
                    continue
                kind, port = k
                d = per.setdefault(port, {"server": [0.0, 0.0], "route": [0.0, 0.0]})
                prev = last.get(pid)
                if prev is not None:
                    d[kind][0] += (ticks - prev) / HZ / (now - t_last)
                d[kind][1] += rss / 2 ** 20
                seen[pid] = ticks
            last, t_last = seen, now
            v = vram_mib(a.gpu)
            for port, d in sorted(per.items()):
                fh.write("%.1f\t%d\t%.2f\t%.2f\t%.0f\t%.0f\t%d\n" % (now, port, d["server"][0], d["route"][0],
                                                                    d["server"][1], d["route"][1], v))


def summary(path):
    import pandas as pd
    t = pd.read_csv(path, sep="\t")
    t = t[t.t != "t"].astype(float)
    busy = t[t.route_cores > 0.05]                 # a route is running on this instance
    busy = busy.assign(cores=busy.server_cores + busy.route_cores, rss_gb=(busy.server_rss_mb + busy.route_rss_mb) / 1024)
    n = t.groupby("t").port.nunique()
    q = busy[["server_cores", "route_cores", "cores", "rss_gb"]].quantile([0.5, 0.95]).round(2)
    print("samples with a route running: %d (instances per sample: median %d)" % (len(busy), n.median()))
    print(q.to_string())
    print("VRAM on the card (MiB): median %d, max %d" % (t.vram_mib.median(), t.vram_mib.max()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out")
    p.add_argument("--port-lo", type=int, default=46000)
    p.add_argument("--port-hi", type=int, default=46950)
    p.add_argument("--gpu", type=int, default=2)
    p.add_argument("--every", type=float, default=5.0)
    p.add_argument("--until-file", default="")
    p.add_argument("--summary")
    a = p.parse_args()
    if a.summary:
        summary(a.summary)
    else:
        sample(a)


if __name__ == "__main__":
    main()
