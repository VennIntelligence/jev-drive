"""Pick and fetch a few comma1M segments (HF commaai/comma1M) for openpilot model smoke tests.

Scans the first --scan segment ids, reads their small frame_info/localizer files, keeps comma 3/3X
segments (1928x1208 road + wide camera) that are moving, ranks them by how much they turn, and downloads
fcamera.hevc + ecamera.hevc for the top --n. Everything goes through hf-mirror.com (direct, parallel).

  python scripts/fetch_comma1m.py --n 4 --out ~/data/datasets/comma1M
"""
import argparse, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevdrive.openpilot.dl import HF_MIRROR, download, hf_url
from jevdrive.openpilot.frames import load_segment_meta

REPO = "commaai/comma1M"


def seg_ids(limit):
    r = requests.get(f"{HF_MIRROR}/api/datasets/{REPO}/tree/main/data", params={"limit": limit}, timeout=60)
    return [d["path"].split("/")[-1] for d in r.json() if d["type"] == "directory"][:limit]


def summarize(sid, root):
    d = root / sid
    try:
        for f in ("frame_info.safetensors", "localizer.safetensors"):
            download(hf_url(REPO, f"data/{sid}/{f}", dataset=True), d / f, streams=1)
        m = load_segment_meta(d)
    except Exception as e:  # noqa: BLE001 - a missing or odd segment is skipped, not fatal
        return dict(sid=sid, ok=False, err=str(e)[:80])
    v = np.linalg.norm(m["vel"], axis=1)
    yaw_rate = m["omega_dev"][:, 2]
    return dict(sid=sid, ok=True, w=int(m["fcam_wh"][0]), h=int(m["fcam_wh"][1]), wide=m["has_ecam"],
                v_mean=float(v.mean()), v_min=float(v.min()), turn=float(np.abs(yaw_rate).mean()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path.home() / "data/datasets/comma1M")
    ap.add_argument("--scan", type=int, default=80)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--min-speed", type=float, default=6.0)
    ap.add_argument("--random-extra", type=int, default=0,
                    help="also fetch this many usable segments drawn at random (seed 0) from the rest, listed in "
                         "extra.json; selected.json is left alone")
    a = ap.parse_args()
    with ThreadPoolExecutor(16) as ex:
        rows = list(ex.map(lambda s: summarize(s, a.out), seg_ids(a.scan)))
    good = [r for r in rows if r["ok"] and r["wide"] and r["w"] == 1928 and r["v_mean"] > a.min_speed]
    good.sort(key=lambda r: -r["turn"])
    print(f"scanned {len(rows)}, usable {len(good)}")
    for r in good[:a.n] if not a.random_extra else []:
        for f in ("fcamera.hevc", "ecamera.hevc"):
            download(hf_url(REPO, f"data/{r['sid']}/{f}", dataset=True), a.out / r["sid"] / f)
        print(json.dumps(r))
    if a.random_extra:
        rest = sorted(good[a.n:], key=lambda r: r["sid"])
        pick = [rest[i] for i in sorted(np.random.default_rng(0).choice(len(rest), a.random_extra, replace=False))]
        for r in pick:
            for f in ("fcamera.hevc", "ecamera.hevc"):
                download(hf_url(REPO, f"data/{r['sid']}/{f}", dataset=True), a.out / r["sid"] / f)
            print(json.dumps(r))
        (a.out / "extra.json").write_text(json.dumps(pick, indent=1))
    else:
        (a.out / "selected.json").write_text(json.dumps(good[:a.n], indent=1))
