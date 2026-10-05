"""rc-*-near readouts (op-train env).

  stats --plan <poses.pkl>                     CPU, before rendering: the near action targets (rft._near rules) by pose class -> stdout + <plan dir>/target_stats.json
  eval  --models O rc-bear-s0 ... [--out f]    one leased card: action head on the near DEV rows -> $R/evalol_near/<model>.json
        per moving (v0 >= rft.NEAR_VMIN) row: desired curvature k = -action[0] / max(1, v0)^2 (left +) against the target curvature kt (the same
        rule as training: pure pursuit on the dense lane-centre path, lookahead clip(v0, 4, 15) m);
        gain = slope of k on kt through 0 (1 = the target), by where (app / in) x d bin, turn rows (cmd left / right);
        sign = sign(k) == command side on turn rows; spread = k(left cmd) - k(right cmd) per approach pose with both;
        exits = plan class at arc d + 12 m (approach rows, as rft_eval.carla_exit).
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import rft as F  # noqa: E402

BINS = (("app d0-3", "app", 0, 3), ("app d3-6", "app", 3, 6), ("app d6-10", "app", 6, 10.01), ("in s0-6", "in", -6.01, 0))


def target_k(dense, dmask, v0):
    tp, tm = F.target_path(dense, dmask)
    a, w = F.act_target_path(tp, tm, v0, F.NEAR_VMIN)
    return -a / (F.ACT_ALPHA * max(1.0, v0) ** 2) if w else np.nan


def cmd_stats(a):
    P = pickle.load(open(a.plan, "rb"))
    R = []
    for p in P:
        for e in p["exits"]:
            R.append(dict(where=p["where"], d=p["d"], v0=p["v0"], cls=e["cls"], kt=target_k(e["dense"], e["dmask"], p["v0"]),
                          kmax=float(np.abs(F.RP.curvature(e["dense"][e["dmask"]], 1.0)[:15]).max()) if e["dmask"].sum() > 3 else np.nan))
    out = {}
    for name, wh, lo, hi in BINS:
        for c in ("left", "right", "straight"):
            m = [r for r in R if r["where"] == wh and lo <= r["d"] < hi and r["cls"] == c]
            kt = np.array([r["kt"] for r in m])
            if len(m) < 5:
                continue
            sup = np.isfinite(kt)
            out[f"{name} {c}"] = dict(n=len(m), supervised=float(sup.mean()), kt_signed_mean=float(np.nanmean(kt) if sup.any() else np.nan),
                                      kt_abs_p50=float(np.nanmedian(np.abs(kt))) if sup.any() else None,
                                      kt_abs_p90=float(np.nanpercentile(np.abs(kt), 90)) if sup.any() else None,
                                      path_kmax_15m_p50=float(np.nanmedian([r["kmax"] for r in m])))
    for k, v in out.items():
        print(f"{k:22s} " + "  ".join(f"{kk} {vv:.3f}" if isinstance(vv, float) else f"{kk} {vv}" for kk, vv in v.items()))
    (Path(a.plan).parent / "target_stats.json").write_text(json.dumps(out, indent=1))


def cmd_eval(a):
    import torch
    import route_adapter as RA
    import rft_eval as E
    dev = torch.device("cuda")
    C = F.rows_of(F.Carla("near"))
    rows = np.flatnonzero((C.r["split"] == "dev") & (C.r["dmask"].sum(1) >= 5))
    pose = C.r["pose_row"][rows]
    v0 = C.tab["v0"][pose].astype(float)
    mv = v0 >= F.NEAR_VMIN
    rows, pose, v0 = rows[mv], pose[mv], v0[mv]
    kt = np.array([target_k(C.r["dense"][j], C.r["dmask"][j], v) for j, v in zip(rows, v0)])
    cmd, where, d = C.r["cmd"][rows], C.r["where"][rows], C.r["d"][rows].astype(float)
    cl = C.r["cluster"][rows]
    fb = np.stack([RA.features("bear", C.r["poly"][j], C.r["pmask"][j]) for j in rows])
    fp = np.stack([RA.features("poly", C.r["poly"][j], C.r["pmask"][j]) for j in rows])
    sg = np.where(cmd == "left", 1.0, np.where(cmd == "right", -1.0, 0.0))
    od = F.rroot("evalol_near")
    for name in a.models:
        m = F.load_rmodel(name, dev)
        f = fp if (m.route is not None and m.route.enc == "poly") else fb
        pl, ac = E.run_rows(m, lambda r: C.T[r], pose, f, C.tab["slot_valid"][pose], C.tab["tc"][pose], dev)
        k = -ac / np.maximum(1.0, v0) ** 2
        r = {"n_rows": int(len(rows))}
        for bn, wh, lo, hi in BINS + (("all", None, -99, 99),):
            mm = (sg != 0) & np.isfinite(kt) & ((where == wh) if wh else True) & (d >= lo) & (d < hi)
            if mm.sum() < 5:
                continue
            x, y = kt[mm], k[mm]
            r[bn] = {"n": int(mm.sum()), "gain": float((x * y).sum() / max((x * x).sum(), 1e-12)), "k_signed": E.boot((sg * k)[mm], cl[mm]),
                     "kt_signed": float((sg * kt)[mm].mean()), "sign_ok": float((np.sign(y) == sg[mm]).mean()),
                     "corr": float(np.corrcoef(x, y)[0, 1]) if mm.sum() > 2 else None}
        sp, scl = [], []
        for p in np.unique(pose[where == "app"]):
            mm = pose == p
            kl, kr = k[mm & (cmd == "left")], k[mm & (cmd == "right")]
            if len(kl) and len(kr):
                sp.append(kl.mean() - kr.mean())
                scl.append(C.tab["cluster"][p])
        r["spread_lr_app"] = E.boot(np.array(sp), np.array(scl)) if sp else {"n": 0}
        app = (where == "app") & np.isin(cmd, ["left", "straight", "right"])
        c = E.cls_at_arc(pl[app], C.cam, d[app] + 12.0)
        r["exit_app_row"] = E.boot((c == cmd[app]).astype(float), cl[app])
        r["exit_app_short"] = float((c == "short").mean())
        (od / f"{name}.json").write_text(json.dumps(r, indent=1, default=float))
        print(name, json.dumps({kk: (vv.get("gain", vv.get("mean")) if isinstance(vv, dict) else vv) for kk, vv in r.items()}, default=float), flush=True)
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("stats")
    p.add_argument("--plan", required=True)
    p = sp.add_parser("eval")
    p.add_argument("--models", nargs="+", required=True)
    a = ap.parse_args()
    {"stats": cmd_stats, "eval": cmd_eval}[a.cmd](a)
