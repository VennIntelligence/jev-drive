"""Stage 1 (CPU): WOD sceneflow segments -> 5 Hz openpilot model frames + ego track, then the anchor selection.

  render   per segment (10 pickles of sc_wod_prep.py): every 2nd 10 Hz frame, FRONT / FRONT_LEFT / FRONT_RIGHT rendered into openpilot's road / wide
           model frames exactly as sc_wod_run.sf_render does (base rig, camgeom rays, nearest neighbour), pose (x, y, yaw), frame speed from the
           pose track -> runs/wm_vs_reproj/prep/<seg>.npz
  select   anchors per category -> runs/wm_vs_reproj/anchors.json
Run with $DATA_DIR/envs/openpilot/bin/python (numpy, PIL).
"""
import argparse
import io
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "experiments/model_smoke/lib"), str(Path(__file__).parent)]
import wm_common as C  # noqa: E402

PREP = C.RUN / "prep"
GROUND = -0.05            # ground z in the WOD vehicle frame (sc_wod_ground.py)


def yaw_of(T):
    return float(np.arctan2(T[1, 0], T[0, 0]))


def render_seg(seg):
    from PIL import Image
    from jevdrive import camgeom as G
    import wod_openpilot_rigs as R
    out = PREP / f"{seg}.npz"
    if out.exists():
        return seg
    d = pickle.load(open(C.SF / f"{seg}.pkl", "rb"))
    cal = d["cal"]
    sizes = [(cal[c]["width"], cal[c]["height"]) for c in (1, 2, 3)]
    h_cam = float(cal[1]["extrinsic"][2, 3])
    idx = {}
    for k in ("road", "wide"):
        rays = G.pinhole_rays(np, G.OP_K[k], G.OP_W, G.OP_H)
        src, U, V = G.choose_sources(np, rays, cal)
        idx[k] = G.nn_gather_index(np.where(src >= 0, src, -1), U, V, sizes).ravel()
    sel = list(range(0, len(d["frames"]), 2))
    op = np.empty((len(sel), 2, 6, 128, 256), np.uint8)
    black = np.array([[0, 128, 128]], np.uint8)
    for j, i in enumerate(sel):
        planes = []
        for c in (1, 2, 3):
            im = Image.open(io.BytesIO(d["frames"][i]["jpg"][c]))
            im.draft("YCbCr", im.size)
            planes.append(np.asarray(im.convert("YCbCr")).reshape(-1, 3))
        cat = np.concatenate(planes + [black])
        for m, k in enumerate(("road", "wide")):
            op[j, m] = R._pack(cat[idx[k]].reshape(G.OP_H, G.OP_W, 3))
    T = np.array([d["frames"][i]["pose"] for i in sel])
    p = np.c_[T[:, 0, 3], T[:, 1, 3], np.unwrap([yaw_of(t) for t in T])]
    dp = np.gradient(p[:, :2], C.DT, axis=0)
    cam = cal[1]["extrinsic"][:3, 3] + np.array([0, 0, -GROUND])      # camera mount in the vehicle frame, z above the ground
    np.savez(out.with_suffix(".tmp.npz"), op=op, pose=p, v=np.hypot(*dp.T), cam=cam, frame=np.array(sel))
    out.with_suffix(".tmp.npz").replace(out)
    return seg


def cmd_render(a):
    PREP.mkdir(parents=True, exist_ok=True)
    segs = sorted(p.stem for p in C.SF.glob("*.pkl"))
    with ProcessPoolExecutor(a.workers) as ex:
        for s in ex.map(render_seg, segs):
            print("rendered", s, flush=True)


def track_stats(p, v):
    """Per 5 Hz frame: lateral deviation of the real path from its 3 s Gaussian-smoothed version (m, left +), mean yaw rate over the next 2 s (deg / s)."""
    from scipy.ndimage import gaussian_filter1d
    sm = gaussian_filter1d(p[:, :2], 7.5, axis=0, mode="nearest")
    tan = np.gradient(sm, axis=0)
    tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-6)
    ey = (p[:, 0] - sm[:, 0]) * -tan[:, 1] + (p[:, 1] - sm[:, 1]) * tan[:, 0]
    w = np.array([np.degrees(p[min(i + C.K, len(p) - 1), 2] - p[i, 2]) / (C.K * C.DT) for i in range(len(p))])
    return ey, w


def cmd_select(a):
    rows = []
    for f in sorted(PREP.glob("*.npz")):
        z = np.load(f)
        p, v = z["pose"], z["v"]
        ey, w = track_stats(p, v)
        n = len(p)
        for i in range(C.HIST, n - C.K - 1):
            fut = slice(i, i + C.K + 1)
            rows.append(dict(seg=f.stem, i=i, v=float(v[i]), dv=float(v[i + 5] - v[i]), w=float(w[i]), ey_max=float(np.abs(ey[fut]).max()),
                             w_abs=float(np.abs(np.diff(p[fut, 2])).max() / C.DT * 57.3), vmax=float(v[fut].max())))
    cats = {"launch": lambda r: r["v"] < 3.5 and r["dv"] > 0.6,
            "cruise": lambda r: r["v"] > 8 and r["w_abs"] < 1.5 and r["ey_max"] < 0.15,
            "curve": lambda r: r["v"] > 3 and abs(r["w"]) > 3,
            "natdev": lambda r: r["v"] > 4 and abs(r["w"]) < 3 and r["ey_max"] > 0.2}
    score = {"launch": lambda r: r["dv"], "cruise": lambda r: r["v"], "curve": lambda r: abs(r["w"]), "natdev": lambda r: r["ey_max"]}
    out, used = [], {}
    for cat, ok in cats.items():
        cand = sorted([r for r in rows if ok(r)], key=lambda r: -score[cat](r))
        n = 0
        for r in cand:
            if any(u["seg"] == r["seg"] and abs(u["i"] - r["i"]) < a.gap for u in out if u["cat"] == cat):
                continue
            if sum(u["seg"] == r["seg"] for u in out) >= 2:
                continue
            out.append({**r, "cat": cat})
            n += 1
            if n >= a.per_cat:
                break
        print(cat, len(cand), "candidates,", n, "chosen", flush=True)
    json.dump(out, open(C.RUN / a.out, "w"), indent=1)
    for r in out:
        print(r)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("render").add_argument("--workers", type=int, default=5)
    q = sp.add_parser("select")
    q.add_argument("--per-cat", type=int, default=3)
    q.add_argument("--seg-cap", type=int, default=3)
    q.add_argument("--out", default="anchors.json")
    q.add_argument("--gap", type=int, default=20, help="min spacing (5 Hz frames) of two anchors of one category in one segment")
    a = ap.parse_args()
    {"render": cmd_render, "select": cmd_select}[a.cmd](a)
