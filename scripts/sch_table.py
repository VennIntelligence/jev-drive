#!/usr/bin/env python3
"""SCH: the box's global schedule table ($DATA_DIR/runs/sched/table.tsv), capacity probe and GO-file grants.

table.tsv (tab separated, one row per lane / grant; '-' = none):
  lane      name (nq3-b, nq4-opl, sch-pilot-b, ...)
  gpus      comma list of physical GPU ids
  workers   CARLA servers per GPU ('-' for GPU-only jobs)
  idx0      first CARLA server index: one number (GPU at position k of `gpus` uses [idx0 + k*span, idx0 + (k+1)*span)),
            or an explicit per-GPU map "g:i,g:i" (lane B)
  idx_span  indices per GPU block
  cpus      taskset core list
  status    free text: debug / pilot / batch / waiting / done ...
  go        the lane's GO file ('-' for chain-owned rows that read none)

GO file (shell-sourceable, re-read by a lane at its step boundaries): GPUS="4 5", WORKERS=6, IDX0=60, IDX_SPAN=12,
<PREFIX>_CPUS=134-145 (+ any lane-specific extras already in the file are kept below a marker line).

  python3 scripts/sch_table.py show            table + live pids / load / CPU / per-GPU VRAM, util, CARLA count + checks
  python3 scripts/sch_table.py check           exit 1 on an index conflict (i, i +- 120 across rows) or a capacity breach
  python3 scripts/sch_table.py grant <lane> --gpus 4,5 --workers 6 --idx0 60 --span 12 --cpus 134-145 \\
          --go $DATA_DIR/runs/nq4/opl/GO --prefix OPL [--status batch]
  python3 scripts/sch_table.py revoke <lane>   GO -> GPUS="" (the lane stops taking new work at its next boundary)

Index rule (docs: tmp/2026-09-26-codex-handoff.md, "CARLA 端口"): index i binds RPC 2000 + 50 i (+1, +2) and its traffic
manager scans TM 8000 + 50 i .. +49 = the RPC block of i + 120, so two rows conflict when an index of one equals an index,
or an index +- 120, of the other. New blocks also stay at index <= 494, so that RPC and TM ports sit below the kernel's
ephemeral range (32768-60999), where an outgoing connection can hold a port and crash a starting server
(docs/closed-loop-acceptance.md); rows whose status starts with "legacy" (lane A's 600-689) are exempt.
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
import time
from pathlib import Path

DATA = Path(os.environ.get("DATA_DIR", Path.home() / "data"))
TABLE = DATA / "runs" / "sched" / "table.tsv"
COLS = ["lane", "gpus", "workers", "idx0", "idx_span", "cpus", "status", "go"]
# capacity model (SCH 2026-09-26 18:25 measurement; see the handoff section)
PIDS_PER_WORKER, PIDS_CAP = 400, 16000          # one CARLA server ~330-430 threads + route client ~15; b2d waits at 17000
VRAM_PER_WORKER_GB, VRAM_CAP_GB = 7.5, 88       # CARLA server 3-9 GB; leave >= 8 GB per card
CPU_CAP = 165                                    # of the cgroup's 175


def load() -> list[dict]:
    if not TABLE.exists():
        return []
    rows = list(csv.DictReader(TABLE.open(), delimiter="\t"))
    for r in rows:
        for c in COLS:
            r[c] = (r.get(c) or "-").strip() or "-"
    return rows


def save(rows: list[dict]) -> None:
    tmp = TABLE.with_suffix(".tmp")
    with tmp.open("w") as f:
        w = csv.DictWriter(f, COLS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows({c: r.get(c, "-") for c in COLS} for r in rows)
    tmp.replace(TABLE)


def gpus(r) -> list[int]:
    return [int(g) for g in r["gpus"].replace(" ", ",").split(",") if g.strip().isdigit()]


def indices(r) -> set[int]:
    if r["idx0"] == "-" or r["idx_span"] == "-":
        return set()
    span = int(r["idx_span"])
    if ":" in r["idx0"]:        # explicit per-GPU map: every block is reserved, including ones for a later expansion
        starts = [int(e.split(":")[1]) for e in r["idx0"].replace(" ", ",").split(",") if ":" in e]
    else:
        starts = [int(r["idx0"]) + k * span for k in range(max(len(gpus(r)), 1))]
    return {s + j for s in starts for j in range(span)}


EPHEMERAL_LO = 32768   # net.ipv4.ip_local_port_range starts here: an outgoing connection can hold a port above it


def conflicts(rows) -> list[str]:
    out, live = [], [r for r in rows if not r["status"].startswith(("done", "revoked"))]
    for r in live:
        top = max(indices(r), default=-1)
        if top >= 0 and 8000 + 50 * top + 49 >= EPHEMERAL_LO and not r["status"].startswith("legacy"):
            out.append(f"{r['lane']}: index {top} puts TM ports in the ephemeral range (keep every index <= 494)")
    for a in range(len(live)):
        for b in range(a + 1, len(live)):
            A, B = indices(live[a]), indices(live[b])
            hit = A & (B | {i + 120 for i in B} | {i - 120 for i in B})
            if hit:
                out.append(f"index conflict {live[a]['lane']} x {live[b]['lane']}: {min(hit)}..{max(hit)} ({len(hit)})")
    return out


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def probe() -> dict:
    cpu0 = int(sh("awk '/usage_usec/{print $2}' /sys/fs/cgroup/cpu.stat") or 0)
    time.sleep(3)
    cpu1 = int(sh("awk '/usage_usec/{print $2}' /sys/fs/cgroup/cpu.stat") or 0)
    g = {}
    for line in sh("nvidia-smi --query-gpu=index,pci.bus_id,memory.used,memory.total,utilization.gpu "
                   "--format=csv,noheader,nounits").splitlines():
        i, bus, used, tot, util = [x.strip() for x in line.split(",")]
        g[bus.lower()[-12:]] = {"gpu": int(i), "used_gb": int(used) / 1024, "total_gb": int(tot) / 1024,
                                "util": int(util), "carla": 0}
    for line in sh("nvidia-smi --query-compute-apps=gpu_bus_id,process_name --format=csv,noheader").splitlines():
        bus, name = [x.strip() for x in line.split(",", 1)]
        if "CarlaUE4" in name and bus.lower()[-12:] in g:
            g[bus.lower()[-12:]]["carla"] += 1
    return {"pids": int(Path("/sys/fs/cgroup/pids.current").read_text()),
            "pids_max": int(Path("/sys/fs/cgroup/pids.max").read_text()),
            "load": Path("/proc/loadavg").read_text().split()[:3], "cores_used": (cpu1 - cpu0) / 3e6,
            "gpus": sorted(g.values(), key=lambda x: x["gpu"])}


def show(check_only=False) -> int:
    rows = load()
    bad = conflicts(rows)
    p = probe()
    if not check_only:
        print("\t".join(COLS))
        for r in rows:
            print("\t".join(r[c] for c in COLS))
        print(f"\npids {p['pids']} / {p['pids_max']} (plan cap {PIDS_CAP}, b2d waits at 17000), load {' '.join(p['load'])}, "
              f"cgroup cores in use {p['cores_used']:.0f} / 175 (cap {CPU_CAP})")
        for x in p["gpus"]:
            print(f"GPU {x['gpu']}: {x['used_gb']:5.1f} / {x['total_gb']:.0f} GB, util {x['util']:3d} %, CARLA servers {x['carla']}")
        free = max(0, (PIDS_CAP - p["pids"]) // PIDS_PER_WORKER)
        print(f"room for about {free} more CARLA workers by pids")
    if p["pids"] > PIDS_CAP:
        bad.append(f"pids {p['pids']} above the planning cap {PIDS_CAP}")
    if p["cores_used"] > CPU_CAP:
        bad.append(f"cgroup CPU {p['cores_used']:.0f} cores above {CPU_CAP}")
    for x in p["gpus"]:
        if x["used_gb"] > VRAM_CAP_GB:
            bad.append(f"GPU {x['gpu']} VRAM {x['used_gb']:.0f} GB above {VRAM_CAP_GB}")
    for b in bad:
        print("CHECK:", b)
    return 1 if bad else 0


def write_go(path: Path, r: dict, prefix: str) -> None:
    keep = ""
    if path.exists() and "# --- lane extras" in path.read_text():
        keep = path.read_text().split("# --- lane extras", 1)[1]
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (f"# GO grant for {r['lane']} (scripts/sch_table.py, {time.strftime('%F %T')}): {r['status']}\n"
            f"GPUS=\"{' '.join(map(str, gpus(r)))}\"\nWORKERS={r['workers'] if r['workers'] != '-' else 0}\n"
            f"IDX0={r['idx0'] if ':' not in r['idx0'] else ''}\nIDX_SPAN={r['idx_span'] if r['idx_span'] != '-' else 0}\n"
            f"{prefix}_CPUS={r['cpus'] if r['cpus'] != '-' else ''}\n")
    if keep:
        body += "# --- lane extras" + keep
    tmp = path.with_suffix(".tmp")
    tmp.write_text(body)
    tmp.replace(path)


def grant(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("lane")
    for k in ("gpus", "workers", "idx0", "span", "cpus", "go", "prefix", "status"):
        ap.add_argument("--" + k, default=None)
    a = ap.parse_args(argv)
    rows = load()
    r = next((x for x in rows if x["lane"] == a.lane), None)
    if r is None:
        r = {c: "-" for c in COLS}
        r["lane"] = a.lane
        rows.append(r)
    for k, c in (("gpus", "gpus"), ("workers", "workers"), ("idx0", "idx0"), ("span", "idx_span"), ("cpus", "cpus"),
                 ("go", "go"), ("status", "status")):
        if getattr(a, k) is not None:
            r[c] = getattr(a, k)
    if a.status is None:
        r["status"] = f"batch granted {time.strftime('%H:%M')}"
    bad = conflicts(rows)
    if bad:
        print("\n".join(bad), "\nnot granted", sep="")
        return 1
    p = probe()
    need = (int(r["workers"]) if r["workers"].isdigit() else 0) * len(gpus(r))
    if p["pids"] + need * PIDS_PER_WORKER > PIDS_CAP:
        print(f"warning: pids {p['pids']} + {need} workers x {PIDS_PER_WORKER} > {PIDS_CAP}; new servers will wait in b2d_run")
    save(rows)
    if r["go"] != "-":
        write_go(Path(os.path.expandvars(r["go"])), r, a.prefix or a.lane.split("-")[-1].upper())
        print("wrote", r["go"])
    print("granted:", "\t".join(r[c] for c in COLS))
    return 0


def revoke(lane: str) -> int:
    rows = load()
    r = next((x for x in rows if x["lane"] == lane), None)
    if r is None:
        print("no such lane")
        return 1
    r["status"] = f"revoked {time.strftime('%H:%M')}"
    save(rows)
    if r["go"] != "-":
        go = Path(os.path.expandvars(r["go"]))
        if go.exists():
            s = go.read_text().splitlines()
            go.write_text("\n".join('GPUS=""' if x.startswith("GPUS=") else x for x in s) + "\n")
        print("GPUS emptied in", go)
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    sys.exit({"show": lambda: show(), "check": lambda: show(True), "grant": lambda: grant(sys.argv[2:]),
              "revoke": lambda: revoke(sys.argv[2])}[cmd]())
