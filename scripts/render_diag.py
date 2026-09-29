#!/usr/bin/env python
"""Read-out for scripts/render_diag.sh (todos/2026-09-28-wm-loop.md, "渲染故障的根因").

    .venv/bin/python scripts/render_diag.py frames <exp> [<exp> ...]   per-frame luminance -> <exp>/frames.parquet
    .venv/bin/python scripts/render_diag.py summary <exp> [<exp> ...]  per-run verdicts     -> stdout, <exp>/runs.csv
    .venv/bin/python scripts/render_diag.py pair <exp_a> <exp_b>      same route, same tick: front-luma difference b - a
                                                                           per route (rep0 of each) -> stdout
    .venv/bin/python scripts/render_diag.py scan <name> <b2d_run out> ..  front camera of every finished run of existing
                                                                           b2d_run outputs -> renderfix/scan_<name>.parquet

frames: every saved camera frame of every finished run: mean luma (0-255) of the three cameras and the front camera's
clipped fraction (luma >= 250), decoded at 1/4 resolution.
summary, per run:
  blowout   the front camera's clipped fraction jumps above 0.15 between two consecutive frames and stays there for
            more than half of the remaining frames (the whole-frame bloom failure; a normal frame is < 0.05);
  lights    night routes: the median over the trace of srv_near_on / near from lights_truth.jsonl (the fraction of
            lights within RouteLightsBehavior's radius that the server has on), and srv_on, all lights on;
  dlum      the mean |front luma - per-route per-tick median over all runs of all given exps| over the frames.
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(os.environ["DATA_DIR"]) / "runs/wl/renderfix"
CAMS = ("front", "front_left", "front_right")


def runs(exp):
    for done in sorted((ROOT / exp).glob("rep*/done/*.json")):
        d = json.loads(done.read_text())
        yield exp, done.parent.parent.name, d["route_id"], Path(d["config"]["out"])


def _one(p):
    im = cv2.imread(str(p), cv2.IMREAD_REDUCED_GRAYSCALE_4)
    return (np.nan, np.nan) if im is None else (float(im.mean()), float((im >= 250).mean()))


def frames(exp):
    rows = []
    with ThreadPoolExecutor(48) as ex:
        for e, rep, rid, out in runs(exp):
            fr = [json.loads(l) for l in open(out / "frames.jsonl")]
            fr = [x for x in fr if x["files"]]
            res = {c: list(ex.map(lambda x: _one(out / x["files"][c]), fr)) for c in CAMS}
            for i, x in enumerate(fr):
                rows.append({"exp": e, "rep": rep, "route_id": rid, "tick": x["tick"],
                             **{"lum_" + c: res[c][i][0] for c in CAMS}, "clip_front": res["front"][i][1]})
    df = pd.DataFrame(rows)
    df.to_parquet(ROOT / exp / "frames.parquet")
    return df


def blowout(clip):
    """Index of the jump into the whole-frame bloom state, or -1."""
    c = np.asarray(clip)
    for i in range(1, len(c)):
        if c[i] > 0.15 and c[i - 1] < 0.1 and (c[i:] > 0.15).mean() > 0.5:
            return i
    return -1


def summary(exps):
    df = pd.concat([pd.read_parquet(ROOT / e / "frames.parquet") for e in exps])
    ref = df.groupby(["route_id", "tick"]).lum_front.median().rename("ref")
    df = df.join(ref, on=["route_id", "tick"])
    out = []
    for (e, rep, rid), g in df.groupby(["exp", "rep", "route_id"]):
        g = g.sort_values("tick")
        i = blowout(g.clip_front.values)
        lt = ROOT / e / rep / "attempts" / rid
        tr = [json.loads(l) for p in sorted(lt.glob("*/lights_truth.jsonl"))[-1:] for l in open(p)]
        tr = [t for t in tr if t["tick"] > 20]
        near = np.median([t["srv_near_on"] / t["near"] for t in tr if t["near"]]) if tr else np.nan
        out.append({"exp": e, "rep": rep, "route_id": rid, "frames": len(g), "lum_front": round(g.lum_front.mean(), 1),
                    "clip_front": round(g.clip_front.mean(), 3), "blowout_tick": int(g.tick.values[i]) if i >= 0 else -1,
                    "dlum": round((g.lum_front - g.ref).abs().mean(), 1),
                    "near_on": round(near, 3), "srv_on": int(np.median([t["srv_on"] for t in tr])) if tr else -1,
                    "lights": tr[0]["lights"] if tr else -1, "sun": tr[0]["sun"] if tr else np.nan})
    o = pd.DataFrame(out)
    for e in exps:
        o[o.exp == e].to_csv(ROOT / e / "runs.csv", index=False)
    return o


def scan(name, outs):
    """Front-camera luma / clipped fraction of every saved frame of every finished run under the given b2d_run --out
    directories (base rate of the bloom failure in earlier generations)."""
    rows = []
    with ThreadPoolExecutor(96) as ex:
        for o in outs:
            for done in sorted(Path(o).glob("done/*.json")):
                out = Path(json.loads(done.read_text())["config"]["out"])
                if not (out / "frames.jsonl").exists():
                    continue
                fr = [json.loads(l) for l in open(out / "frames.jsonl")]
                fr = [x for x in fr if x["files"]]
                for x, (lum, clip) in zip(fr, ex.map(lambda x: _one(out / x["files"]["front"]), fr)):
                    rows.append({"src": o, "route_id": done.stem, "tick": x["tick"], "lum_front": lum, "clip_front": clip})
    df = pd.DataFrame(rows)
    df.to_parquet(ROOT / f"scan_{name}.parquet")
    per = df.sort_values("tick").groupby(["src", "route_id"]).clip_front.agg(lambda c: blowout(c.values) >= 0)
    print(name, len(df), "frames", len(per), "runs", int(per.sum()), "with a bloom jump")
    print(per.groupby(level=0).agg(["sum", "count"]))


def pair(a, b):
    """Per route present in both exps (rep0): mean and max over ticks of |front luma b - a|, and the mean signed
    difference (how much a switch changes frames that were fine without it)."""
    f = [pd.read_parquet(ROOT / e / "frames.parquet").query("rep == 'rep0'") for e in (a, b)]
    m = f[0].merge(f[1], on=["route_id", "tick"], suffixes=("_a", "_b"))
    d = m.lum_front_b - m.lum_front_a
    o = m.assign(d=d, ad=d.abs()).groupby("route_id").agg(frames=("d", "size"), lum_a=("lum_front_a", "mean"),
                                                          lum_b=("lum_front_b", "mean"), mean_d=("d", "mean"),
                                                          mean_abs_d=("ad", "mean"), max_abs_d=("ad", "max"))
    print(f"{a} -> {b}")
    print(o.round(2).to_string())


if __name__ == "__main__":
    cmd, exps = sys.argv[1], sys.argv[2:]
    if cmd == "pair":
        pair(*exps)
    elif cmd == "scan":
        scan(exps[0], exps[1:])
    elif cmd == "frames":
        for e in exps:
            print(e, len(frames(e)), "frames")
    else:
        pd.set_option("display.width", 250)
        pd.set_option("display.max_rows", 500)
        print(summary(exps).to_string(index=False))
