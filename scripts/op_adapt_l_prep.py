"""op-adapt L data preparation (todos/2026-10-01-op-adapt-L-prereg.md).

  vstreams   stream plan of the full WOD-E2E val split (479 sequences, even frame numbers, one stream per contiguous run of
             frames) -> processed/op_adapt/wod_val_plan.json; the trunk cache itself is `scripts/op_adapt_cache.py wodval`
  wodtab     kinematics, human future, intent and slice flags of every WOD sample row: train / dev rows of the r2 `wod`
             domain (read only) and the full val domain (packed here) -> $L/prep/{wod,wodval}.npz
  nustab     the same for nuScenes (r2 `nus` domain, read only; 5 Hz lattice, full 9-slot context) -> $L/prep/nus.npz
  pack       sample table + flat-row map of the full val domain -> $L/t/{samples,trunk}/wodval.*

$L = $DATA_DIR/runs/op_adapt_L. The r2 run dir is only ever read.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from jevdrive.common import data_dir  # noqa: E402
import log_expert_audit as AU  # noqa: E402
from jevdrive import op_adapt_l as L  # noqa: E402

WE = data_dir() / "processed" / "waymo_e2e"


def cmd_vstreams(a):
    ix = pd.read_parquet(WE / "index.parquet", columns=["sequence", "frame", "split"])
    v = ix[(ix.split == "val") & (ix.frame % 2 == 0)].sort_values(["sequence", "frame"])
    streams = []
    for i, (seq, g) in enumerate(v.groupby("sequence", sort=True)):
        fr = g.frame.to_numpy()
        cut = np.flatnonzero(np.diff(fr) != 2) + 1
        for j, part in enumerate(np.split(fr, cut)):
            names = [f"{seq}-{f:03d}" for f in part]
            streams.append({"key": f"{i:04d}{'abcdefgh'[j]}_{seq}", "names": names, "targets": list(range(len(names)))})
    out = data_dir() / "processed" / "op_adapt" / "wod_val_plan.json"
    out.write_text(json.dumps({"streams": streams}))
    print(len(streams), "streams,", sum(len(s["names"]) for s in streams), "slots ->", out)


def cmd_pack(a):
    """Sample table of the full val domain (one sample per even frame; ctx = the 9 previous slots of its stream)."""
    from jevdrive import op_adapt_r2 as R
    root = L.lroot("t")
    rows = []
    d = data_dir() / "processed" / "op_adapt" / "wodval"
    for f in sorted(d.glob("*.npz")):
        if f.stem.endswith(".tmp"):
            continue
        with np.load(f) as z:
            c, names, tg, tc = int(z["stride"]), z["names"].astype(str), z["targets"], z["traffic"]
        for j in tg:
            loc = j - c * np.arange(R.CTX - 1, -1, -1)
            rows.append({"cache": str(f), "row": int(j), "lctx": np.where(loc >= 0, loc, -1), "key": names[j], "name": names[j],
                         "stream": f.stem, "tc0": float(tc[0]), "tc1": float(tc[1]), "split": "val",
                         "group": names[j].rsplit("-", 1)[0], "domain": "wodval"})
    ix = pd.DataFrame(rows)
    ix["uid"] = 9_000_000_000 + np.arange(len(ix))
    R.pack("wodval", ix, root)
    print("wodval:", len(ix), "samples from", ix.cache.nunique(), "streams")


def wod_kin(names: np.ndarray) -> dict:
    """Kinematics and human future of the given WOD frame names (`<sequence>-<frame>`), NaN where the frame has none."""
    ix = pd.read_parquet(WE / "index.parquet", columns=["sequence", "frame", "split", "intent", "has_future", "n_pref"])
    key = ix.sequence + "-" + ix.frame.map("{:03d}".format)
    pos = pd.Series(np.arange(len(ix)), index=key.to_numpy())
    r = pos.reindex(names).to_numpy()
    ok = ~np.isnan(r)
    ri = np.where(ok, r, 0).astype(int)
    past = np.load(WE / "past.npy", mmap_mode="r")
    fut = np.load(WE / "future.npy", mmap_mode="r")
    o = np.argsort(ri)
    p, f = np.empty((len(ri),) + past.shape[1:], np.float32), np.empty((len(ri),) + fut.shape[1:], np.float32)
    p[o], f[o] = np.asarray(past[ri[o]]), np.asarray(fut[ri[o]])
    has = ok & ix.has_future.to_numpy()[ri]
    sp = lambda x: np.linalg.norm(x, axis=-1)  # noqa: E731
    tab = {"has": has, "v0": sp(p[:, 15, 2:4]), "vm05": sp(p[:, 13, 2:4]), "vm1": sp(p[:, 11, 2:4]), "pm05": p[:, 13, :2],
           "pm1": p[:, 11, :2], "fut": f[:, 1:16:2, :2], "fut20": f[:, :, :2],
           "intent": np.where(ok, ix.intent.to_numpy()[ri], 0).astype(np.int8),      # WOD routing intent: 0 unknown, 1 straight, 2 left, 3 right
           "rater": ok & (ix.n_pref.to_numpy()[ri] > 0)}
    tab["intent_audit"] = np.array([-1, 1, 0, 2])[tab["intent"]].astype(np.int8)     # the audit's coding (0 left, 1 straight, 2 right)
    return tab


def flags(tab: dict) -> dict:
    """Slice masks with the audit's definitions (BASE thresholds) plus the L-specific contrast sets."""
    t = {k: tab[k] for k in ("fut", "v0", "vm05", "vm1", "pm05", "pm1")} | {"intent": tab["intent_audit"]}
    has = tab["has"]
    fl = {k: v & has for k, v in AU.slices(t).items() if k in AU.SLICES}
    fut = t["fut"].astype(np.float64)
    seg = np.diff(np.concatenate([np.zeros_like(fut[:, :1]), fut], 1), axis=1)
    sp = np.linalg.norm(seg, axis=-1) / 0.5
    apsi = np.where(sp >= AU.SEG_VALID, np.abs(np.degrees(np.arctan2(seg[..., 1], seg[..., 0]))), 0.0)
    # straight intent, no turn, moving: the "straight with straight intent" contrast of turn onset (a superset of control)
    fl["straight_int"] = has & (tab["intent"] == 1) & (apsi.max(1) < 10) & (tab["v0"] >= 2.0) & ~fl["stop"]
    return fl


def cmd_wodtab(a):
    from jevdrive import op_adapt_r2 as R
    r2 = R.Domain("wod")                                              # read only
    out = L.lroot("prep")
    for dn, names, split in (("wod", r2.col("name").astype(str), r2.col("split").astype(str)),
                             ("wodval", None, None)):
        if dn == "wodval":
            d = R.Domain("wodval", L.lroot("t"))
            names, split = d.col("name").astype(str), d.col("split").astype(str)
        tab = wod_kin(names)
        fl = flags(tab)
        tab = {k: v for k, v in tab.items()} | {f"s_{k}": v for k, v in fl.items()} | {"split": split, "name": names,
                                                                                       "seq": np.array([n.rsplit("-", 1)[0] for n in names])}
        np.savez(out / f"{dn}.npz", **tab)
        print(dn, len(names), {k: int(v.sum()) for k, v in fl.items()})


def cmd_nustab(a):
    from jevdrive import op_adapt_r2 as R
    from jevdrive import op_adapt_data as D
    from concurrent.futures import ProcessPoolExecutor
    d = R.Domain("nus")
    idx = pickle.load(open(data_dir() / "processed/nusc_zs/trainval_index.pkl", "rb"))
    s = d.s
    jobs = []
    for name, g in s.groupby("key"):
        f = g.cache.iloc[0]
        with np.load(data_dir() / f) as z:
            steps = z["steps"]
        plan = D.scene_plan(idx, name)
        st = steps[g.row.to_numpy()]
        keep = (st % 4 == 0) & (g[[f"ctx{k}" for k in range(9)]].min(axis=1).to_numpy() >= 0)
        jobs.append((name, f, g.index.to_numpy()[keep], st[keep], plan["t"][st[keep]]))
    with ProcessPoolExecutor(AU.pool_size(), initializer=AU._nus_init) as ex:
        res = [o for o in ex.map(AU._nus_scene, jobs, chunksize=4) if o is not None]
    rows = np.concatenate([o["rows"] for o in res])
    cat = lambda k: np.concatenate([o[k] for o in res])  # noqa: E731
    tab = {"row": rows, "log": np.concatenate([np.full(len(o["rows"]), i, np.int32) for i, o in enumerate(res)]),
           "scene": np.concatenate([np.full(len(o["rows"]), o["name"]) for o in res]), "t": cat("t"),
           "v0": cat("v0"), "vm05": cat("vm05"), "vm1": cat("vm1"), "pm05": cat("pm05"), "pm1": cat("pm1"), "fut": cat("fut"),
           "intent": np.zeros(len(rows), np.int8), "intent_audit": np.full(len(rows), -1, np.int8),
           "has": np.ones(len(rows), bool), "split": np.concatenate([np.full(len(o["rows"]), o["split"]) for o in res])}
    fl = flags(tab)
    tab |= {f"s_{k}": v for k, v in fl.items()}
    np.savez(L.lroot("prep") / "nus.npz", **tab)
    print("nus", len(rows), {k: int(v.sum()) for k, v in fl.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["vstreams", "pack", "wodtab", "nustab"])
    a = ap.parse_args()
    {"vstreams": cmd_vstreams, "pack": cmd_pack, "wodtab": cmd_wodtab, "nustab": cmd_nustab}[a.cmd](a)
