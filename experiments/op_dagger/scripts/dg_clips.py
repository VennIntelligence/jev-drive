"""op_dagger clip sets (CPU, op-train env): 20 frames on the 5 Hz lattice per clip (10 history ending at t0 + K = 10 logged future frames).

  train    WOD-E2E train rows (op_adapt_L prep wod.npz, split train = wod/r2-train): launch / low / mid / cruise clips
  heldout  WOD-E2E val rows (wodval.npz = wod/val): the same categories, never trained on
  sf       decision 123's 10 WOD sceneflow segments, its 15 anchors (exact 10 Hz poses), for the engine checks
Categories at t0: launch = first moving frame (v > 0.1 m/s) after >= 1.8 s at v <= 0.1 (op_adapt_h lwod, m = 1); low 0.5-3 m/s, mid 3-8, cruise >= 8,
all with |logged heading change over 2 s| <= MAX_TURN (the gaps here are launch gain and offset recovery, not turns). At most 2 clips per sequence,
>= 4 s apart. Logged poses of frames 9..19 from the t0 future (dg_common.poses_from_fut20); `check` measures that heading estimate on the sf poses.

  $DATA_DIR/envs/op-train/bin/python experiments/op_dagger/scripts/dg_clips.py select [--n-train 320 --n-heldout 120]
  $DATA_DIR/envs/op-train/bin/python experiments/op_dagger/scripts/dg_clips.py render train heldout sf [--workers 40]
  $DATA_DIR/envs/op-train/bin/python experiments/op_dagger/scripts/dg_clips.py check
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dg_common as C  # noqa: E402

MAX_TURN = 15.0           # deg over the 2 s future
QUOTA = {"train": {"launch": 120, "low": 60, "mid": 60, "cruise": 80}, "heldout": {"launch": 40, "low": 20, "mid": 20, "cruise": 40}}
SRC = {"train": ("wod", "train", "wod/r2-train"), "heldout": ("wodval", "val", "wod/val")}


def cat_of(v_hist, v0, v_fut):
    if v0 > 0.1 and np.all(v_hist <= 0.1):
        return "launch"
    if np.min(v_fut) < 0.3:
        return None
    return "low" if 0.5 <= v0 < 3 else "mid" if 3 <= v0 < 8 else "cruise" if v0 >= 8 else None


def cmd_select(a):
    from experiments.op_adapt_l.lib import op_adapt_l as L
    from jevdrive.data import splits
    rng = np.random.default_rng(0)
    for name in ("train", "heldout"):
        f, sp, sid = SRC[name]
        S = splits.load(sid)
        z = np.load(L.lroot("prep") / f"{f}.npz", allow_pickle=True)
        names, seq, split, v, fut20 = z["name"].astype(str), z["seq"].astype(str), z["split"].astype(str), z["v0"].astype(float), z["fut20"]
        keep = (split == sp) & S.mask(seq)
        fr = np.array([int(x.rsplit("-", 1)[1]) for x in names])
        by = {}
        for k in np.flatnonzero(keep):
            by.setdefault(seq[k], {})[fr[k]] = k
        cand = {c: [] for c in QUOTA[name]}
        for q, d in by.items():
            for f0 in sorted(d):
                ks = [d.get(f0 + 2 * (i - C.T0)) for i in range(C.NF)]
                if any(k is None for k in ks):
                    continue
                c = cat_of(v[ks[: C.T0]], v[ks[C.T0]], v[ks[C.T0 + 1:]])
                if c is None:
                    continue
                P = C.poses_from_fut20(fut20[ks[C.T0]])
                if abs(np.degrees(P[-1, 2])) > MAX_TURN:
                    continue
                cand[c].append((q, f0, ks))
        rows = []
        for c, quota in QUOTA[name].items():
            got, per = 0, {}
            for i in rng.permutation(len(cand[c])):
                q, f0, ks = cand[c][i]
                if got >= quota:
                    break
                if len(per.get(q, [])) >= 2 or any(abs(f0 - g) < 40 for g in per.get(q, [])):
                    continue
                if any(r["seq"] == q and abs(r["f0"] - f0) < 40 for r in rows):
                    continue
                per.setdefault(q, []).append(f0)
                rows.append(dict(seq=q, f0=int(f0), cat=c, names=[str(names[k]) for k in ks], rows=[int(k) for k in ks]))
                got += 1
            print(f"{name} {c}: {len(cand[c])} candidates, {got} chosen", flush=True)
        json.dump(dict(split_id=S.id, src=f, clips=rows), open(C.root("clips") / f"{name}.json", "w"))


def _render(names):
    import drive_backbones_openpilot as DB
    return DB.render(list(names))


def _init():
    sys.path[:0] = [str(C.REPO / "experiments/op_adapt_h/scripts")]
    import h_prep
    h_prep._wod_init("train")


def wod_calib():
    sys.path[:0] = [str(C.REPO / "experiments/op_adapt_h/scripts")]
    import h_prep
    return h_prep.wod_calib()


def render_wod(name, workers):
    from experiments.op_adapt_l.lib import op_adapt_l as L
    d = C.root("clips", name)
    if (d / "tab.npz").exists():
        print(name, "exists")
        return
    js = json.load(open(C.root("clips") / f"{name}.json"))
    clips = js["clips"]
    z = np.load(L.lroot("prep") / f"{js['src']}.npz", allow_pickle=True)
    v_all, fut_all = z["v0"].astype(float), z["fut20"].astype(np.float32)
    calib = wod_calib()
    n = len(clips)
    tmp = d / "imgs.tmp.npy"
    out = np.lib.format.open_memmap(tmp, "w+", np.uint8, (n, C.NF, 2, 6, 128, 256))
    t0 = time.time()
    with ProcessPoolExecutor(workers, initializer=_init) as ex:
        for i, fr in enumerate(ex.map(_render, [c["names"] for c in clips], chunksize=2)):
            out[i] = fr
            if i % 50 == 0:
                print(f"{name}: {i + 1}/{n} clips, {time.time() - t0:.0f} s", flush=True)
    out.flush()
    del out
    tmp.replace(d / "imgs.npy")
    ks = np.array([c["rows"] for c in clips])
    fut = fut_all[ks[:, C.T0:]]                                                 # (n, K + 1, 20, 2)
    pose = np.stack([C.poses_from_fut20(f[0]) for f in fut])
    v = v_all[ks]
    tab = dict(id=np.array([c["names"][C.T0] for c in clips]), seq=np.array([c["seq"] for c in clips]), cat=np.array([c["cat"] for c in clips]),
               v=v.astype(np.float32), fut20=fut, pose=pose.astype(np.float64),
               cam=np.array([np.array(calib[c["seq"]]["1"]["extrinsic"]).reshape(4, 4)[:3, 3] for c in clips], np.float32),
               tc=np.tile(np.array([1.0, 0.0], np.float32), (n, 1)), k0=np.array([C.kappa_log(p, vv[C.T0:]) for p, vv in zip(pose, v)]),
               split_id=np.array(js["split_id"]))
    np.savez(d / "tab.npz", **tab)
    print(f"{name}: {n} clips in {time.time() - t0:.0f} s; cats {dict(zip(*np.unique(tab['cat'], return_counts=True)))}", flush=True)


def render_sf():
    """decision 123 anchors: frames i - 9 .. i + 10 of runs/wm_vs_reproj/prep/<seg>.npz, exact poses in the anchor frame."""
    d = C.root("clips", "sf")
    if (d / "tab.npz").exists():
        print("sf exists")
        return
    wm = C.data_dir() / "runs" / "wm_vs_reproj"
    an = json.load(open(wm / "anchors.json"))
    imgs, rows = [], []
    for a in an:
        z = np.load(wm / "prep" / f"{a['seg']}.npz")
        i = a["i"]
        if i - C.T0 < 0 or i + C.K >= len(z["op"]):
            print("skip", a["seg"][:12], i)
            continue
        op, pose, v = z["op"], z["pose"], z["v"]
        p0 = pose[i]
        rel = np.c_[(pose[i: i + C.K + 1, :2] - p0[:2]) @ C.rot(p0[2]), np.unwrap(pose[i: i + C.K + 1, 2]) - p0[2]]
        imgs.append(op[i - C.T0: i + C.K + 1])
        rows.append(dict(id=f"{a['seg'][:12]}_{i}", seq=a["seg"], cat=a["cat"], v=v[i - C.T0: i + C.K + 1], pose=rel, cam=z["cam"]))
    n = len(rows)
    np.save(d / "imgs.npy", np.stack(imgs))
    tab = {k: np.array([r[k] for r in rows]) for k in ("id", "seq", "cat", "v", "pose", "cam")}
    tab |= dict(tc=np.tile(np.array([1.0, 0.0], np.float32), (n, 1)), fut20=np.full((n, C.K + 1, 20, 2), np.nan, np.float32),
                k0=np.array([C.kappa_log(p, vv[C.T0:]) for p, vv in zip(tab["pose"], tab["v"])]), split_id=np.array("wod_sf/wm_vs_reproj anchors (decision 123)"))
    np.savez(d / "tab.npz", **tab)
    print(f"sf: {n} clips; cats {dict(zip(*np.unique(tab['cat'], return_counts=True)))}", flush=True)


def cmd_check(a):
    """Heading estimate from a 5 s future against the exact sceneflow poses: every 5 Hz frame of the 10 segments with >= 5 s ahead."""
    wm = C.data_dir() / "runs" / "wm_vs_reproj" / "prep"
    err, ey = {"stop": [], "launch": [], "moving": []}, []
    for f in sorted(wm.glob("*.npz")):
        z = np.load(f)
        pose, v = z["pose"], z["v"]
        tt = 0.2 * np.arange(len(pose))
        for i in range(0, len(pose) - 26, 3):
            p0 = pose[i]
            tq = tt[i] + C.T_FUT
            xy = np.stack([np.interp(tq, tt, pose[:, c]) for c in range(2)], 1)
            fut = (xy - p0[:2]) @ C.rot(p0[2])
            est = C.poses_from_fut20(fut)
            true = np.unwrap(pose[i: i + C.K + 1, 2]) - p0[2]
            e = np.degrees(np.abs(est[:, 2] - true)).max()
            k = "stop" if v[i: i + C.K + 1].max() < 0.3 else "launch" if v[i] < 1.0 else "moving"
            err[k].append(e)
            ey.append(np.abs(est[:, :2] - (pose[i: i + C.K + 1, :2] - p0[:2]) @ C.rot(p0[2])).max())
    out = {k: dict(n=len(x), median=float(np.median(x)) if x else None, p90=float(np.percentile(x, 90)) if x else None) for k, x in err.items()}
    out["pos_err_m"] = dict(median=float(np.median(ey)), p90=float(np.percentile(ey, 90)))
    print(json.dumps(out, indent=1))
    json.dump(out, open(C.root("check") / "heading_estimate.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "render", "check"])
    ap.add_argument("sets", nargs="*")
    ap.add_argument("--workers", type=int, default=40)
    a = ap.parse_args()
    if a.cmd == "select":
        cmd_select(a)
    elif a.cmd == "check":
        cmd_check(a)
    else:
        for s in a.sets:
            render_sf() if s == "sf" else render_wod(s, a.workers)
