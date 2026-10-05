"""factor_wm G0 clip sets from WOD-E2E val (`wod/val`, never trained on), op-train env, CPU (plans/2026-10-05-stage1-prereg.md section 3.1).

  g0a  20 frames (10 history + 2 s): launch 24 (op_dagger launch), sharp 20 (|heading change over the next 1 s| > 15 deg), cruise 24 (v >= 8,
       |heading change over 2 s| < 5 deg)
  g0b  50 frames (10 history + 8 s): launch 40, turn 40 (heading change over 8 s >= 45 deg, first 1 s <= 10 deg, v0 >= 2), cruise 40 (v0 >= 8,
       |heading change over 8 s| < 20 deg)
At most 2 clips per sequence, >= 4 s apart, seeded draw. Rendering = op_dagger's (drive_backbones_openpilot.render, the WOD front3 calib).

  $DATA_DIR/envs/op-train/bin/python experiments/factor_wm/scripts/fw_clips.py select
  $DATA_DIR/envs/op-train/bin/python experiments/factor_wm/scripts/fw_clips.py render g0a g0b --workers N
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fw_common as C  # noqa: E402
import dg_clips as DGC  # noqa: E402

SETS = {"g0a": dict(kl=10, quota={"launch": 24, "sharp": 20, "cruise": 24}),
        "g0b": dict(kl=40, quota={"launch": 40, "turn": 40, "cruise": 40}),
        "g1s": dict(kl=40, quota={"stay": 40}),                                              # G1 false-go guard: log stands still 8 s
        "train": dict(kl=40, quota={"launch": 400, "turn": 300, "cruise": 150, "mid": 150, "stay": 150}, src="train")}
SRC = {"val": ("wodval", "val", "wod/val"), "train": ("wod", "train", "wod/r2-train")}
SPLIT = "wod/val"


def heading(fut_rows, kl):
    return np.degrees(C.chained_poses(fut_rows, kl)[:, 2])


def cat_of(name, v, has, fut, ks, kl):
    """category of the clip with rows ks (None = not eligible)."""
    T0 = C.T0
    m = [ks[T0 + 20 * i] for i in range((kl + 19) // 20)]
    if not all(has[k] for k in m):
        return None
    vh, v0 = v[ks[:T0]], v[ks[T0]]
    if name in ("g1s", "train") and v0 <= 0.1 and np.all(v[ks[T0:]] <= 0.1):
        return "stay"
    if v0 > 0.1 and np.all(vh <= 0.1):
        return "launch"
    if v0 < 2.0:
        return None
    psi = heading(fut[m], kl)
    if name == "g0a":
        if v0 >= 2.0 and abs(psi[5]) > 15.0:
            return "sharp"
        if v0 >= 8.0 and abs(psi[10]) < 5.0:
            return "cruise"
        return None
    if v0 >= 2.0 and abs(psi[-1]) >= 45.0 and abs(psi[5]) <= 10.0:
        return "turn"
    if v0 >= 8.0 and np.max(np.abs(psi)) < 20.0 and np.min(v[ks[T0:]]) > 3.0:
        return "cruise"
    if name == "train" and 2.0 <= v0 < 8.0 and np.max(np.abs(psi)) < 20.0:
        return "mid"
    return None


def cmd_select(a):
    from experiments.op_adapt_l.lib import op_adapt_l as L
    from jevdrive.data import splits
    for name in (a.sets or list(SETS)):
        cfg = SETS[name]
        if (C.root("clips") / f"{name}.json").exists():
            print(name, "selection exists")
            continue
        src, sp, sid = SRC[cfg.get("src", "val")]
        S = splits.load(sid)
        z = np.load(L.lroot("prep") / f"{src}.npz", allow_pickle=True)
        names, seq, split, v, fut, has = (z["name"].astype(str), z["seq"].astype(str), z["split"].astype(str), z["v0"].astype(float),
                                          z["fut20"].astype(float), z["has"].astype(bool))
        keep = (split == sp) & S.mask(seq)
        fr = np.array([int(x.rsplit("-", 1)[1]) for x in names])
        by = {}
        for k in np.flatnonzero(keep):
            by.setdefault(seq[k], {})[fr[k]] = k
        rng = np.random.default_rng([0, len(name), cfg["kl"]])
        nf = C.NH + cfg["kl"]
        cand = {c: [] for c in cfg["quota"]}
        for q, d in by.items():
            for f0 in sorted(d):
                ks = [d.get(f0 + 2 * (i - C.T0)) for i in range(nf)]
                if any(k is None for k in ks):
                    continue
                c = cat_of(name, v, has, fut, ks, cfg["kl"])
                if c in cand:
                    cand[c].append((q, f0, ks))
        rows = []
        for c, quota in cfg["quota"].items():
            got = 0
            for i in rng.permutation(len(cand[c])):
                q, f0, ks = cand[c][i]
                if got >= quota:
                    break
                mine = [r["f0"] for r in rows if r["seq"] == q]
                if len(mine) >= 2 or any(abs(f0 - g) < 40 for g in mine):
                    continue
                rows.append(dict(seq=q, f0=int(f0), cat=c, names=[str(names[k]) for k in ks], rows=[int(k) for k in ks]))
                got += 1
            print(f"{name} {c}: {len(cand[c])} candidates, {got} chosen", flush=True)
        json.dump(dict(split_id=S.id, src=src, kl=cfg["kl"], clips=rows), open(C.root("clips") / f"{name}.json", "w"))


def render(name, workers):
    from experiments.op_adapt_l.lib import op_adapt_l as L
    d = C.root("clips", name)
    if (d / "tab.npz").exists():
        print(name, "exists")
        return
    js = json.load(open(C.root("clips") / f"{name}.json"))
    clips, kl = js["clips"], js["kl"]
    nf = C.NH + kl
    z = np.load(L.lroot("prep") / f"{js['src']}.npz", allow_pickle=True)
    v_all, fut_all = z["v0"].astype(float), z["fut20"].astype(np.float64)
    calib = DGC.wod_calib()
    n = len(clips)
    tmp = d / "imgs.tmp.npy"
    out = np.lib.format.open_memmap(tmp, "w+", np.uint8, (n, nf, 2, 6, 128, 256))
    t0 = time.time()
    with ProcessPoolExecutor(workers, initializer=DGC._init) as ex:
        for i, f in enumerate(ex.map(DGC._render, [c["names"] for c in clips], chunksize=1)):
            out[i] = f
            if i % 20 == 0:
                print(f"{name}: {i + 1}/{n} clips, {time.time() - t0:.0f} s", flush=True)
    out.flush()
    del out
    tmp.replace(d / "imgs.npy")
    ks = np.array([c["rows"] for c in clips])
    pose = np.stack([C.chained_poses(fut_all[[k[C.T0 + 20 * i] for i in range((kl + 19) // 20)]], kl) for k in ks])
    v = v_all[ks]
    a0 = (v[:, C.T0 + 1] - v[:, C.T0 - 1]) / (2 * C.DT)
    tab = dict(id=np.array([c["names"][C.T0] for c in clips]), seq=np.array([c["seq"] for c in clips]), cat=np.array([c["cat"] for c in clips]),
               v=v.astype(np.float32), pose=pose, a0=a0.astype(np.float32), fut20=fut_all[ks].astype(np.float32),
               cam=np.array([np.array(calib[c["seq"]]["1"]["extrinsic"]).reshape(4, 4)[:3, 3] for c in clips], np.float32),
               tc=np.tile(np.array([1.0, 0.0], np.float32), (n, 1)),
               k0=np.array([DG_k0(p, vv) for p, vv in zip(pose, v)]), split_id=np.array(js["split_id"]))
    np.savez(d / "tab.npz", **tab)
    print(f"{name}: {n} clips in {time.time() - t0:.0f} s; cats {dict(zip(*np.unique(tab['cat'], return_counts=True)))}", flush=True)


def DG_k0(pose, v):
    return C.DG.kappa_log(pose, v[C.T0:])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "render"])
    ap.add_argument("sets", nargs="*")
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "select":
        cmd_select(a)
    else:
        from jevdrive.common import n_cpus
        for s in a.sets:
            render(s, a.workers or max(1, n_cpus() - 2))
