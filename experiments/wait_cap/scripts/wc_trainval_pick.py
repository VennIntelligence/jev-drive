"""Pick the OpenScene trainval camera archives for the wait-train design (plans/2026-10-16-wait-train-design.md, section 2).

Per trainval log: frames in the W cells (W_L + W_N of wc.cell_of: stationary, human stays put 4 s) that have the 4-frame history
and 8 future frames at 0.5 s spacing (wc.log_rows). Per archive (openscene_sensor_trainval_0-199.json maps archive -> logs):
sum over its logs. Archives are taken by waiting frames, richest first, until the byte budget is reached (deterministic, ties by index).

  python wc_trainval_pick.py pick  --index idx.json --sizes sizes.json [--gb 120] [--workers 12]   (box, envs/jevdrive python, CPU)
Writes results/trainval_logs.txt (logs of the picked archives), results/trainval_archives.tsv, results/trainval_pick.md.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wc  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"


def job(p):
    rows = wc.log_rows(p)
    return p.stem, sum(wc.cell_of(r) in ("W_L", "W_N") for r in rows), len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--sizes", required=True)
    ap.add_argument("--gb", type=float, default=120.0)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    idx = json.load(open(a.index))
    size = json.load(open(a.sizes))
    paths = sorted((wc.NAV / "navsim_logs/trainval").glob("*.pkl"))
    with Pool(a.workers) as pl:
        cnt = {n: (w, e) for n, w, e in pl.imap_unordered(job, paths, chunksize=4)}
    arch = []
    for k, logs in idx.items():
        arch.append(dict(name=k, size=size[k + ".tgz"], w=sum(cnt.get(l, (0, 0))[0] for l in logs), eligible=sum(cnt.get(l, (0, 0))[1] for l in logs),
                         missing=sum(l not in cnt for l in logs), logs=logs))
    arch.sort(key=lambda r: (-r["w"], int(r["name"].rsplit("_", 1)[1])))
    pick, tot = [], 0
    for r in arch:
        if tot + r["size"] <= a.gb * 1e9:
            pick.append(r)
            tot += r["size"]
    pick.sort(key=lambda r: int(r["name"].rsplit("_", 1)[1]))
    (RES / "trainval_logs.txt").write_text("\n".join(sorted(l for r in pick for l in r["logs"])) + "\n")
    (RES / "trainval_archives.tsv").write_text("".join(f"{r['name']}.tgz\t{r['size']}\t{r['w']}\t{len(r['logs'])}\n" for r in pick))
    allw = sum(r["w"] for r in arch)
    (RES / "trainval_pick.md").write_text(
        f"{len(paths)} trainval logs, {allw} W-cell frames with history (all 200 archives); picked {len(pick)} archives, {tot / 1e9:.1f} GB, "
        f"{sum(r['w'] for r in pick)} W frames, {sum(len(r['logs']) for r in pick)} logs\n\n| archive | GB | W frames | logs |\n|---|--:|--:|--:|\n"
        + "".join(f"| {r['name']} | {r['size'] / 1e9:.1f} | {r['w']} | {len(r['logs'])} |\n" for r in pick))
    print((RES / "trainval_pick.md").read_text())


if __name__ == "__main__":
    main()
