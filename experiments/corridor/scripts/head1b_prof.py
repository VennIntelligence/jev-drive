#!/usr/bin/env python3
"""HEAD1b steps B / C (experiments/corridor, amendment "补记 2026-10-10（HEAD1b）" of plans/2026-10-10-head1-prereg.md). EXPLORATORY: a second
attempt after the missed registered gate G3; nothing here is a registered read.

Heading profiles fed to the policy's memory channel (experiments/op_parity/scripts/path_req.py kind qp): (N, 22) fp32 heading (rad) on the
label grid, one file per token-cache dir, tab order. The step-A lane writes the on-log ones ($H1/final/prof: out-of-fold on navtrain, the
fold-0 model on navtest); this script adds

  smoke   a SMOKE input under $H1B/smoke/prof that no read touches: navtrain = the logged-path label L (forward-filled; privileged, code-path
          test only), navtest = the stage-1 head `L-f0-p1000-s0`. CPU.
  ot      the head's prediction on the hinge-only off-track rows of P2H10S (ot1 / yr1 / bd4 of the given shards): the head is run on the
          row's OWN cached vision tokens (the view re-projected to the displaced pose) and its own ego input, by the fold model(s) that
          never saw the row's log. --match-final: the checkpoints are the ones whose stored held-out predictions reproduce $H1/final/prof
          (a single run or the mean of two seeds per fold), so the off-track rows get exactly the head the on-log rows got; the on-log
          rows of the same shards are re-predicted and compared with the stored files (guard of this script's inference path).
          Also writes quality.json: the fed profile's heading error on those rows against the logged future re-expressed in the row's
          frame, next to the on-log twin rows, and how far the prediction follows the pose offset. GPU, a few GB, seconds.
"""
import argparse
import glob
import itertools
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(Path(__file__).parent)]
from jevdrive.common import data_dir  # noqa: E402

D = data_dir()
H1, H1B, CR = D / "runs/corridor/head1", D / "runs/corridor/head1b", D / "runs/op_parity/cache"
NSH, NFOLD = 12, 5
TRAIN_DIRS = [f"navtrain_full.s{k}of{NSH}" for k in range(NSH)]
GRID = np.r_[np.arange(0.0, 40.001, 2.5), 45.0, 50.0, 60.0, 70.0, 80.0]
EXPL = "EXPLORATORY (HEAD1b: second attempt after the missed registered gate G3, user-authorised; not a registered read)"


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def at(prof, s):
    """prof (n, 22) on GRID, s (n,) -> linear interpolation at arc length s (clamped to the grid)."""
    s = np.clip(s, 0, GRID[-1])
    j = np.clip(np.searchsorted(GRID, s, side="right") - 1, 0, len(GRID) - 2)
    r, f = np.arange(len(s)), (s - GRID[j]) / (GRID[j + 1] - GRID[j])
    return prof[r, j] * (1 - f) + prof[r, j + 1] * f


def save(d, name, x):
    d.mkdir(parents=True, exist_ok=True)
    assert x.ndim == 2 and x.shape[1] == len(GRID) and np.isfinite(x).all(), (name, x.shape)
    np.save(d / f".{name}.tmp.npy", x.astype(np.float32))
    (d / f".{name}.tmp.npy").replace(d / f"{name}.npy")


def cmd_smoke(a):
    out = H1B / "smoke" / "prof"
    lab = np.load(H1 / "labels/navtrain.npz")
    L, o = lab["L"].astype(np.float32), 0
    L[:, 0] = np.nan_to_num(L[:, 0])
    for j in range(1, L.shape[1]):                                              # forward fill beyond the logged path's end
        L[:, j] = np.where(np.isfinite(L[:, j]), L[:, j], L[:, j - 1])
    for d in TRAIN_DIRS:
        n = len(np.load(CR / d / "tab.npz")["names"])
        assert (lab["names"][o:o + n] == np.load(CR / d / "tab.npz")["names"]).all()
        save(out, d, L[o:o + n])
        o += n
    z = np.load(H1 / "pred" / "L-f0-p1000-s0.npz")
    assert (z["names_test"] == np.load(CR / "lb_navtest/tab.npz")["names"]).all()
    save(out, "lb_navtest", z["test"][..., 0])
    print(f"SMOKE profiles (logged-path labels on navtrain: privileged, code-path test only) -> {out}")


def find_final(folds):
    """Per fold the checkpoint(s) whose stored held-out predictions are $H1/final/prof on the fold's held-out rows."""
    tabs = [np.load(CR / d / "tab.npz")["names"] for d in TRAIN_DIRS]
    names = np.concatenate(tabs)
    F = np.concatenate([np.load(H1 / "final/prof" / f"{d}.npy") for d in TRAIN_DIRS])
    assert len(F) == len(names)
    pos = {t: i for i, t in enumerate(names.tolist())}
    T = np.load(H1 / "final/prof/lb_navtest.npy")
    out, info = {}, {}
    for j in folds:
        cand = []
        for f in sorted(glob.glob(str(H1 / f"train-L-f{j}-p1000-s*" / "*" / "ckpt.pt"))):
            p = Path(f).parent / "pred.npz"
            if "smoke" in f or not p.exists():
                continue
            z = np.load(p)
            if z["dev"].shape[1:] != (len(GRID), 2):
                continue
            cand.append((f, F[[pos[t] for t in z["names_dev"].tolist()]], z["dev"][..., 0].astype(np.float64), z["test"][..., 0].astype(np.float64)))
        best = None
        for k in (1, 2):
            for c in itertools.combinations(cand, k):
                if len({x[2].shape for x in c}) > 1:
                    continue
                d = float(np.abs(np.mean([x[2] for x in c], 0) - c[0][1]).max())
                if best is None or d < best[0]:
                    best = (d, [x[0] for x in c], np.mean([x[3] for x in c], 0))
            if best is not None and best[0] < 1e-4:
                break
        assert best is not None and best[0] < 1e-4, f"fold {j}: no run (or pair of runs) among {[c[0] for c in cand]} reproduces final/prof (best max |diff| {best and best[0]})"
        out[j], info[j] = best[1], dict(ckpt=best[1], max_abs_diff_rad=best[0])
        if j == 0:
            info[j]["navtest_max_abs_diff_rad"] = float(np.abs(best[2] - T).max())
            assert info[j]["navtest_max_abs_diff_rad"] < 1e-4, "final/prof/lb_navtest.npy is not the fold-0 model's navtest prediction"
    return out, info


def cmd_ot(a):
    import torch
    import head1_train as HT
    from jevdrive.data import splits
    from jevdrive.run import Run
    out = Path(a.out)
    with Run("corridor", "head1b/prof-ot" + ("-smoke" if a.smoke else ""), config=vars(a)) as run:
        dev = torch.device("cuda")
        folds = list(range(NFOLD))
        if a.match_final:
            ck, info = find_final(folds)
        else:
            spec = dict(x.split("=", 1) for x in a.ckpt)
            ck = {j: (spec.get(str(j)) or spec["all"]).split(",") for j in folds}
            info = {j: dict(ckpt=ck[j]) for j in folds}
        run.info("%s; head checkpoints per fold: %s", EXPL, json.dumps(info))
        fm = [splits.load(f"navsim/op-parity-cf5f{j}-dev") for j in folds]
        nets = {}

        def model(f):
            if f not in nets:
                c = torch.load(f, map_location="cpu", weights_only=False)
                cfg = c["config"]
                assert cfg["arm"] in ("L", "C") and cfg.get("head", "full") == "full", cfg
                net = HT.build(True, drop=cfg.get("drop", 0.1)).to(dev)
                net.load_state_dict(c["model"])
                nets[f] = (net.eval(), c["emu"], c["esd"])
            return nets[f]

        def predict(d):
            """(n, 22) profile of every row of a cache dir, each row by the fold model(s) that never saw its log."""
            t = np.load(CR / d / "tab.npz")
            n = len(t["names"])
            fold = np.stack([s.mask(t["log"]) for s in fm], 1)
            assert (fold.sum(1) == 1).all(), f"{d}: {int((fold.sum(1) != 1).sum())} rows are not held out by exactly one fold"
            fold = fold.argmax(1)
            V = HT.load_tokens([(d, np.arange(n), 0)], n, dev)
            P = np.zeros((n, len(GRID)))
            for j in folds:
                i = np.flatnonzero(fold == j)
                if not len(i):
                    continue
                for f in ck[j]:
                    net, emu, esd = model(f)
                    E = torch.as_tensor((t["ego"].astype(np.float32) - emu) / esd, dtype=torch.float32, device=dev)
                    P[i] += HT.predict(net, V, E, torch.as_tensor(i, device=dev))[..., 0] / len(ck[j])
            del V
            return P, t

        Q = dict(note=EXPL, checkpoints=info, on_log_repredict_max_abs_diff_rad={}, families={})
        base = {}
        for d in a.data:
            P, t = predict(d)
            ref = np.load((H1 / "final/prof" if a.match_final else H1B / "smoke/prof") / f"{d}.npy")
            Q["on_log_repredict_max_abs_diff_rad"][d] = float(np.abs(P - ref).max())
            Q.setdefault("on_log_repredict_mean_abs_diff_rad", {})[d] = float(np.abs(P - ref).mean())
            if a.match_final:           # this script's inference path = the trainer's stored predictions, to the bf16 rounding of the head's output
                assert Q["on_log_repredict_max_abs_diff_rad"][d] < 0.035 and Q["on_log_repredict_mean_abs_diff_rad"][d] < 2e-3, (d, Q["on_log_repredict_max_abs_diff_rad"][d])   # 2 bf16 steps at 2-4 rad
            base[d] = (ref.astype(np.float64) if a.match_final else P, t)
        deg = np.degrees
        rms = lambda x: float(np.sqrt(np.mean(np.square(x)))) if len(x) else None  # noqa: E731
        for fam in a.fams:
            acc = []
            for d in a.data:
                P, t = predict(f"{fam}_{d}")
                save(out, f"{fam}_{d}", P)
                Pb, tb = base[d]
                sr = t["src_row"]
                assert (tb["names"][sr] == t["names"]).all()
                fut, off = t["fut"].astype(np.float64), t["off"].astype(np.float64)
                ok = np.isfinite(fut).all((1, 2)) & np.isfinite(tb["fut"][sr]).all((1, 2))
                x = np.concatenate([np.zeros((len(fut), 1, 2)), np.nan_to_num(tb["fut"][sr][..., :2].astype(np.float64))], 1)
                s4 = np.hypot(*np.moveaxis(np.diff(x, axis=1), -1, 0)).sum(1)       # logged 4 s arc length (the same path in both frames)
                acc.append(dict(ok=ok, dpsi=off[:, 1], dy=off[:, 0], turn=np.abs(deg(tb["fut"][sr][:, 7, 2].astype(np.float64))),
                                e_row=wrap(at(P, s4) - fut[:, 7, 2]), e_log=wrap(at(Pb[sr], s4) - tb["fut"][sr][:, 7, 2]),
                                d0=P[:, 0] - Pb[sr][:, 0], d10=P[:, 4] - Pb[sr][:, 4], d4=at(P, s4) - at(Pb[sr], s4)))
            A = {k: np.concatenate([x[k] for x in acc]) for k in acc[0]}
            ok = A["ok"]
            slope = lambda y: float(np.polyfit(-A["dpsi"][ok], y[ok], 1)[0])  # noqa: E731
            Q["families"][fam] = dict(
                rows=int(len(ok)), rows_with_logged_future=int(ok.sum()), abs_dpsi_deg_mean=float(np.abs(deg(A["dpsi"])).mean()), abs_dy_m_mean=float(np.abs(A["dy"]).mean()),
                heading_err_at_logged_4s_arc_rms_deg={b: dict(n=int((ok & m).sum()), own_view=rms(deg(A["e_row"][ok & m])), on_log_twin=rms(deg(A["e_log"][ok & m])))
                                                      for b, m in (("all", np.ones(len(ok), bool)), ("> 20 deg", A["turn"] > 20), ("> 45 deg", A["turn"] > 45))},
                slope_of_own_view_minus_on_log_profile_on_minus_dpsi={"at 0 m": slope(A["d0"]), "at 10 m": slope(A["d10"]), "at the logged 4 s arc": slope(A["d4"])},
                reading="slope 1 = the head re-expresses the road ahead in the displaced frame (what the logged path does), 0 = it ignores the displacement")
            run.info("%s: %s", fam, json.dumps(Q["families"][fam]))
        (out / "quality.json").write_text(json.dumps(Q, indent=1) + "\n")
        (out / "DONE").write_text("ok\n")
        run.summary.update(out=str(out), quality=Q["families"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke")
    p = sub.add_parser("ot")
    p.add_argument("--out", required=True)
    p.add_argument("--data", nargs="+", default=["navtrain_full.s2of12", "navtrain_full.s3of12"])
    p.add_argument("--fams", nargs="+", default=["ot1", "yr1", "bd4"])
    p.add_argument("--match-final", action="store_true", help="the checkpoints that reproduce $H1/final/prof")
    p.add_argument("--ckpt", nargs="*", default=[], help="J=ckpt.pt[,ckpt.pt] per fold or all=...: explicit checkpoints (smoke)")
    p.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    {"smoke": cmd_smoke, "ot": cmd_ot}[a.cmd](a)
