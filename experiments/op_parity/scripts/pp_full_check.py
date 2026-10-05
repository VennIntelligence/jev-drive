"""op_parity full-run sanity gates between launch stages (scripts/pp_full_chain.sh; docs/long-runs.md staged launch). Exit 1 = stop the chain.

  cache --k K --shards 0|all [--register navsim/op-parity-full]
        per shard: tab rows = front rows = side rows = teacher rows; no NaN / inf; future present on every row; front / side token RMS within
        25 % of the Stage B navtrain W cache (cache/lb_navtrain@warp, same encoder and protocol); side not all-zero; teacher plan ADE vs the
        log (8 poses, rear axle) in [1.0, 3.5] m (shipped on W frames: 1.57 on the real-frame subset, Stage A W8) and its speed ratio in
        [0.7, 1.3]. --register: the log-disjoint train / dev split over all shards (dev = sha256(log) % 50 == 0), jevdrive.data.splits.
  train --tag T   finite DONE, dev ADE <= 1.2 m (pilot P2 0.83 on its dev rows), drift to shipped with inputs off <= 0.30 m.
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_pl.Path(__file__).parent)]
import argparse, glob, hashlib, json  # noqa: E401,E402

import numpy as np  # noqa: E402

from jevdrive.common import data_dir  # noqa: E402

C = data_dir() / "runs" / "op_parity" / "cache"


def rms(a, n=512):
    x = np.asarray(a[np.linspace(0, len(a) - 1, min(n, len(a))).astype(int)], np.float32)
    return float(np.sqrt((x ** 2).mean()))


def check_cache(a):
    import torch
    from pp_train import rear
    from experiments.op_adapt_r2.lib import op_adapt_r2 as R2
    ref_f, ref_s = rms(np.load(C / "lb_navtrain@warp/front.npy", mmap_mode="r")), rms(np.load(C / "lb_navtrain/side.npy", mmap_mode="r"))
    shards = range(a.k) if a.shards == "all" else [int(x) for x in a.shards.split(",")]
    bad, names, logs = [], [], []
    for i in shards:
        b, w = C / f"navtrain_full.s{i}of{a.k}", C / f"navtrain_full.s{i}of{a.k}@warp"
        tab, tz = np.load(b / "tab.npz"), np.load(w / "teacher.npz")
        F, S = np.load(w / "front.npy", mmap_mode="r"), np.load(b / "side.npy", mmap_mode="r")
        n = len(tab["names"])
        names += tab["names"].tolist()
        logs += tab["log"].tolist()
        fut = tab["fut"]
        cam = tab["cam"][:, 0].astype(np.float64)
        W = torch.as_tensor(R2.t_weights(0.5 * np.arange(1, 9)))
        x, y, _ = rear(torch.as_tensor(tz["plan"], dtype=torch.float64), torch.as_tensor(cam), W)
        P = np.stack([x.numpy(), y.numpy()], -1)
        ade = float(np.linalg.norm(P - fut[:, :, :2], axis=-1).mean())
        L4 = np.linalg.norm(fut[:, -1, :2], axis=-1)
        mv = L4 > 2
        sr = float(np.median(np.linalg.norm(P[mv, -1], axis=-1) / L4[mv]))
        r = {"shard": i, "n": n, "front_rms": rms(F), "side_rms": rms(S), "ade_teacher": ade, "speed_ratio_teacher": sr,
             "nan": bool(np.isnan(fut).any() or not np.isfinite(tz["plan"]).all())}
        ok = (len(F) == len(S) == len(tz["plan"]) == n and not r["nan"] and abs(r["front_rms"] / ref_f - 1) < 0.25
              and abs(r["side_rms"] / ref_s - 1) < 0.25 and 1.0 <= ade <= 3.5 and 0.7 <= sr <= 1.3)
        print(json.dumps(r | {"ok": ok, "ref_front_rms": ref_f, "ref_side_rms": ref_s}))
        if not ok:
            bad.append(i)
    if a.register and not bad:
        from jevdrive.data import splits
        assert len(set(names)) == len(names)
        dev = lambda g: int(hashlib.sha256(g.encode()).hexdigest(), 16) % 50 == 0  # noqa: E731
        o = f"navsim navtrain tokens with a logged future (navtrain_future.npz), {len(names)} tokens; dev = navtrain logs with sha256(log) % 50 == 0"
        tr = splits.define("navsim", a.register.split("/")[1] + "-train", [t for t, g in zip(names, logs) if not dev(g)], unit="token", origin=o,
                           used_by=["experiments/op_parity"], status="frozen", notes="op_parity full run (protocol W)")
        dv = splits.define("navsim", a.register.split("/")[1] + "-dev", [t for t, g in zip(names, logs) if dev(g)], unit="token", origin=o,
                           used_by=["experiments/op_parity"], status="frozen", notes="op_parity full run dev rows (log-disjoint)")
        splits.check_disjoint(tr, dv)
        print("registered", tr.id, dv.id)
    return 1 if bad else 0


def check_train(a):
    fs = sorted(glob.glob(str(data_dir() / "runs" / "op_parity" / f"train-{a.tag}" / "*" / "DONE")))
    if not fs:
        print("no DONE"), exit(1)
    d = json.loads(open(fs[-1]).read())
    ok = d.get("dev_ade", 9) <= 1.2 and d.get("dev_drift_off", 9) <= 0.30
    print(json.dumps({k: d.get(k) for k in ("dev_ade", "dev_drift_on", "dev_drift_off", "train_s")} | {"ok": ok}))
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("cache")
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--shards", default="all")
    p.add_argument("--register", default="")
    p = sp.add_parser("train")
    p.add_argument("--tag", required=True)
    a = ap.parse_args()
    exit({"cache": check_cache, "train": check_train}[a.cmd](a))
