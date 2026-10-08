"""Offline checks of the SH30 online core (experiments/alpasim/lib/sh30_core.py) on navtest tokens, no simulator.

  parity  the core fed a token's real CAM_F0 JPEGs and index ego states must reproduce the stored serving path: the pp_prep W tokens
          (cache/lb_navtest@warp/front.npy), the bench plan (bench/ol/lb_navtest/plans/<tag>@warp.npz) and its exported poses
  cold    the same tokens with only the m = 1, 2, 3 newest keyframes and states, under both cold-start rules: distance of the 8 exported
          poses to the full-history plan and to the logged future (AlpaSim scenes start without history: decisions 0-2 of 10)

  python experiments/alpasim/scripts/sh30_check.py --out $DATA_DIR/runs/alpasim/sh30_check [--n 300] [--scenes FILE]
Tokens: those of --scenes (AlpaSim scene ids `<log>-<token>`) first, then a seed-0 draw from navtest up to --n. Run in envs/op-train.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import sh30_core as C  # noqa: E402
from jevdrive import navsim_zs as Z  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402


def main(a):
    R = data_dir() / "runs"
    idx = Z.load_index("navtest", slim=True)
    by = {e["token"]: k for k, e in enumerate(idx)}
    first = [s.rsplit("-", 1)[1] for s in Path(a.scenes).read_text().split()] if a.scenes else []
    rest = [idx[k]["token"] for k in np.random.default_rng(0).permutation(len(idx))]
    toks = list(dict.fromkeys([t for t in first if t in by] + rest))[: a.n]
    tab = np.load(R / "op_parity/cache/lb_navtest/tab.npz")
    row = {t: k for k, t in enumerate(tab["names"].tolist())}
    front = np.load(R / "op_parity/cache/lb_navtest@warp/front.npy", mmap_mode="r")
    pz = np.load(R / f"bench/ol/lb_navtest/plans/{a.tag}@warp.npz")
    assert pz["names"].tolist() == tab["names"].tolist()
    pf = R / f"bench/ol/lb_navtest/preds/{a.tag}-warp__base.npz"
    preds = np.load(pf)["poses"] if pf.exists() else None
    cores = {c: C.Core(a.tag, cold=c) for c in C.COLD}
    cores["zero"].model = cores["backwarp"].model                      # one copy of the weights
    ade = lambda x, y: float(np.linalg.norm(x[:, :2] - y[:, :2], axis=1).mean())  # noqa: E731
    rows, ms = [], []
    for n, t in enumerate(toks):
        e, r = idx[by[t]], row[t]
        cam = e["cams"][-1]["CAM_F0"]
        keys = [C.pack(e["cams"][f]["CAM_F0"]["path"], cam) for f in range(4)]
        pose, vel = np.asarray(e["pose"], float), np.asarray(e["vel"], float)
        kw = dict(acc=e["acc"][-1], cmd=e["cmd"][-1], cam_t=cam["t"], lht=e["map"] in ("sg-one-north",))
        full = cores["backwarp"].plan(keys, pose, vel, **kw)
        ms.append(full["ms"])
        tk = full["tokens"].float().cpu().numpy()
        ref = np.asarray(front[r], np.float32)
        fut = tab["fut"][r]
        rec = {"token": t, "v0": float(np.linalg.norm(vel[-1])), "cmd": int(np.argmax(e["cmd"][-1])),
               "tok_max": float(np.abs(tk - ref).max()), "tok_rel": float(np.linalg.norm(tk - ref) / np.linalg.norm(ref)),
               "ego_max": float(np.abs(full["ego"] - tab["ego"][r]).max()),
               "mu_xy_max": float(np.abs(full["mu"][:, :2] - pz["plan_mu"][r][:, :2]).max()),
               "pred_max": float(np.abs(full["poses"] - preds[r]).max()) if preds is not None else None,
               "full_fut": ade(full["poses"], fut) if not np.isnan(fut).any() else None}
        for m in (1, 2, 3):
            w = float(pose[4 - m + 1, 2] - pose[4 - m, 2]) / 0.5 if m > 1 else float(pose[3, 2] - pose[2, 2]) / 0.5
            for c in C.COLD:
                o = cores[c].plan(keys[-m:], pose[-m:] - 0, vel[-m:], yaw_rate=w, **kw)
                rec[f"{c}{m}_full"] = ade(o["poses"], full["poses"])
                rec[f"{c}{m}_fut"] = ade(o["poses"], fut) if rec["full_fut"] is not None else None
                rec[f"{c}{m}_x4"] = float(o["poses"][-1, 0] - full["poses"][-1, 0])
        rows.append(rec)
        if n % 50 == 0:
            print(n, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mean = lambda k: float(np.mean([r[k] for r in rows if r[k] is not None]))  # noqa: E731
    q = lambda k, p: float(np.quantile([r[k] for r in rows if r[k] is not None], p))  # noqa: E731
    S = {"n": len(rows), "tag": a.tag, "parity": {k: {"mean": mean(k), "p99": q(k, 0.99), "max": q(k, 1.0)}
                                                  for k in ("tok_max", "tok_rel", "ego_max", "mu_xy_max", "pred_max") if rows[0][k] is not None},
         "full_fut_ade": mean("full_fut"),
         "cold": {f"{c}{m}": {"ade_vs_full": mean(f"{c}{m}_full"), "p90_vs_full": q(f"{c}{m}_full", 0.9), "ade_vs_log": mean(f"{c}{m}_fut"),
                              "dx4_mean": mean(f"{c}{m}_x4")} for c in C.COLD for m in (1, 2, 3)},
         "ms_warm": {k: float(np.median([x[k] for x in ms[5:]])) for k in ms[0]}}
    (out / "check.json").write_text(json.dumps({"summary": S, "rows": rows}, indent=1))
    print(json.dumps(S, indent=1))
    (out / "DONE").write_text(time.strftime("%F %T") + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="SH30-F-s0")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--scenes", default="")
    main(ap.parse_args())
