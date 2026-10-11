#!/usr/bin/env python3
"""Wall clock and card time of pai_eval.sh / pai_queue.sh runs, from the run dirs: a chunk starts at the first `driver ... on GPU` line of its
log.txt (pai_run.sh) and ends at the mtime of its DONE; run.json has the card, the scenes and the compose-phase wall. Plain python3, on the Tokyo box
(timestamps in the log are local time).
  pai_queue_timing.py '<queue root>/*/runs/*' [--out md]          one table per run, one per card, the queue totals
"""
import argparse
import glob
import json
import re
import time
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("pattern", nargs="+"), ap.add_argument("--out")
a = ap.parse_args()
rows = []
for d in sorted(Path(x) for p in a.pattern for x in glob.glob(p)):
    if not (d / "DONE").exists() or not (d / "run.json").exists():
        continue
    t0 = next((time.mktime(time.strptime(m[1], "%Y-%m-%d %H:%M:%S")) for ln in (d / "log.txt").read_text().splitlines()
               if (m := re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) driver .* on GPU", ln))), None)
    j = json.loads((d / "run.json").read_text())
    if t0 is not None:
        rows.append(dict(run=d.parent.parent.name + "/" + d.name, card=j["gpu"], scenes=j["scenes"], compose=j["wall_s"], t0=t0, t1=(d / "DONE").stat().st_mtime))
fmt = lambda s: f"{int(s // 3600)} h {int(s % 3600 // 60):02d} min"  # noqa: E731
q0, q1 = min(r["t0"] for r in rows), max(r["t1"] for r in rows)
L = [f"Queue: {len(rows)} chunks, {sum(r['scenes'] for r in rows)} scenes, {time.strftime('%F %T', time.localtime(q0))} to {time.strftime('%F %T', time.localtime(q1))}, wall {fmt(q1 - q0)}.", "",
     "| Card | Chunks | Scenes | Busy (start of stack to DONE) | Compose phase | Idle inside the queue span | Scenes / h of busy time |", "|---|--:|--:|--:|--:|--:|--:|"]
for g in sorted({r["card"] for r in rows}):
    c = [r for r in rows if r["card"] == g]
    busy = sum(r["t1"] - r["t0"] for r in c)
    L.append(f"| {g} | {len(c)} | {sum(r['scenes'] for r in c)} | {fmt(busy)} | {fmt(sum(r['compose'] for r in c))} | {fmt(q1 - q0 - busy)} | {sum(r['scenes'] for r in c) * 3600 / busy:.1f} |")
runs = defaultdict(list)
for r in rows:
    runs[r["run"].split("/")[0]].append(r)
L += ["", "| Run | Chunks | Scenes | First start | Last DONE | Span | Summed chunk time (cards) |", "|---|--:|--:|---|---|--:|--:|"]
for k, c in runs.items():
    s0, s1 = min(r["t0"] for r in c), max(r["t1"] for r in c)
    L.append(f"| {k} | {len(c)} | {sum(r['scenes'] for r in c)} | {time.strftime('%F %T', time.localtime(s0))} | {time.strftime('%F %T', time.localtime(s1))} | {fmt(s1 - s0)} | "
             f"{fmt(sum(r['t1'] - r['t0'] for r in c))} |")
L += ["", "| Chunk | Card | Scenes | Start | End | Elapsed min | Compose s |", "|---|--:|--:|---|---|--:|--:|"]
for r in sorted(rows, key=lambda r: r["t0"]):
    L.append(f"| {r['run']} | {r['card']} | {r['scenes']} | {time.strftime('%T', time.localtime(r['t0']))} | {time.strftime('%T', time.localtime(r['t1']))} | {(r['t1'] - r['t0']) / 60:.1f} | {r['compose']} |")
print("\n".join(L))
if a.out:
    Path(a.out).write_text("\n".join(L) + "\n")
