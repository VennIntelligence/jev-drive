#!/usr/bin/env python
"""Find stop -> go launches in comma1M segments (localizer only), fetch their videos.
Launch = static (|v| < 0.1 m/s) for >= 1.8 s, then v >= 1 m/s within 4 s; >= 100 frames of history (20 Hz) and >= 2 s after the onset.
Onset j = first frame with |v| >= 0.1 (the WOD / HUGSIM "first moving frame").

  python real_launch_scan.py scan --n-ids 600 --out <events.json> [--fetch]
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from jevdrive.openpilot.dl import download, hf_url  # noqa: E402
from jevdrive.openpilot.frames import load_segment_meta  # noqa: E402

ROOT = Path.home() / "data/datasets/comma1M"
REPO_ID = "commaai/comma1M"


def seg_ids(limit):
    import requests
    r = requests.get(f"https://hf-mirror.com/api/datasets/{REPO_ID}/tree/main/data", params={"limit": limit}, timeout=120)
    return [d["path"].split("/")[-1] for d in r.json() if d["type"] == "directory"][:limit]


def events_of(meta):
    t, v = meta["t_loc"], np.linalg.norm(meta["vel"], axis=1)
    st, i, ev = v < 0.1, 0, []
    while i < len(v):
        if not st[i]:
            i += 1
            continue
        j = i
        while j < len(v) and st[j]:
            j += 1
        if j < len(v) and t[j] - t[i] >= 1.8 and j >= 100 and j + 40 <= len(v):
            k = np.where(v[j:] >= 1.0)[0]
            if len(k) and t[j + k[0]] - t[j] <= 4.0:
                ev.append(dict(onset=int(j), static_s=float(t[j] - t[i])))
        i = j
    return ev


def scan(sid):
    d = ROOT / sid
    try:
        for f in ("frame_info.safetensors", "localizer.safetensors"):
            download(hf_url(REPO_ID, f"data/{sid}/{f}", dataset=True), d / f, streams=1)
        m = load_segment_meta(d)
        if m["fcam_wh"] != (1928, 1208) or not m["has_ecam"]:
            return sid, []
        return sid, events_of(m)
    except Exception as e:  # noqa: BLE001
        print("skip", sid, str(e)[:60], flush=True)
        return sid, []


def fetch(sid):
    for f in ("fcamera.hevc", "ecamera.hevc"):
        download(hf_url(REPO_ID, f"data/{sid}/{f}", dataset=True), ROOT / sid / f, streams=8)
    print("fetched", sid, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["scan"])
    ap.add_argument("--n-ids", type=int, default=600)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--max-events", type=int, default=130)
    a = ap.parse_args()
    with ThreadPoolExecutor(16) as ex:
        rows = list(ex.map(scan, seg_ids(a.n_ids)))
    res, n = {}, 0
    for sid, ev in rows:                       # first segments by id until max-events
        if ev and n < a.max_events:
            res[sid] = ev
            n += len(ev)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"segments with launches {len(res)}, events {n}, scanned {len(rows)}", flush=True)
    if a.fetch:
        with ThreadPoolExecutor(3) as ex:
            list(ex.map(fetch, res))
        print("FETCH DONE", flush=True)
