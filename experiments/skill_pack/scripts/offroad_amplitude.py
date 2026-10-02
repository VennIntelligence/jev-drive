"""Turn amplitude of the native plan vs the human trajectory on normal scenes (navtrain subset, navtest) and a lateral gain calibration.

navsim2 env, CPU. Plan = the 8-pose file the scorer receives (runs/op_lb/lb_{navtrain,navtest}/preds/gimm-cinque__base.npz), human = the
logged future (runs/navsim_zs/index/<split>_future.npz), both in the ego frame at t0 (x forward, y left). Per horizon t = 1, 2, 3, 4 s:
signed ratio plan / human (tokens with a clear human displacement), regression slope through the origin, spread; for lateral y and heading;
binned by the human path's curvature (|heading change at 4 s|: straight < 0.1 rad, gentle 0.1-0.4, sharp > 0.4), speed and command.
"Near-term extrapolation": the plan's heading change over its first second divided by the path length gives a curvature, extrapolated as a
circular arc over the plan's own arc length. Gains are fitted on navtrain only (regress human y on plan y through the origin) and written
to gains.json for offroad_gain.py. Outputs tables/amp_*.csv, figs/amp_*.png.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(Path(__file__).resolve().parent)]
import offroad_lib as L  # noqa: E402

OUT = REPO / "experiments/skill_pack/results/navhard-offroad"
H = {1: 1, 2: 3, 3: 5, 4: 7}          # horizon (s) -> pose index
SETS = {"navtrain": ("lb_navtrain", "navtrain"), "navtest": ("lb_navtest", "navtest")}
CMD = ["left", "straight", "right", "unknown"]


def load(name):
    d, split = SETS[name]
    m = json.load(open(L.D / f"runs/op_lb/{d}/meta.json"))
    p = np.load(L.D / f"runs/op_lb/{d}/preds/gimm-cinque__base.npz")
    f = np.load(L.D / f"runs/navsim_zs/index/{split}_future.npz")
    gt = dict(zip(f["tokens"].tolist(), f["poses"]))
    toks = p["tokens"].tolist()
    ok = [i for i, t in enumerate(toks) if t in gt]
    P = p["poses"][ok].astype(np.float64)
    G = np.stack([gt[toks[i]] for i in ok]).astype(np.float64)
    mi = {t: i for i, t in enumerate(m["names"])}
    speed = np.array([m["speed"][mi[toks[i]]] for i in ok])
    cmd = np.array([CMD[m["cmd"][mi[toks[i]]]] for i in ok])
    return np.array(toks)[ok], P, G, speed, cmd


def arc_extrapolation(P):
    """Circular arc from the plan's first-second curvature over the plan's arc length at 1..4 s: (n, 4) lateral, heading."""
    seg = np.linalg.norm(np.diff(np.concatenate([np.zeros((len(P), 1, 2)), P[:, :, :2]], 1), axis=1), axis=-1)
    s = np.cumsum(seg, 1)
    kap = P[:, 1, 2] / np.maximum(s[:, 1], 1.0)                                    # heading at 1 s over arc length
    out_y, out_h = [], []
    for t, i in H.items():
        a = kap * s[:, i]
        out_y.append(np.where(np.abs(kap) > 1e-4, (1 - np.cos(a)) / np.where(np.abs(kap) > 1e-4, kap, 1), 0.5 * kap * s[:, i] ** 2))
        out_h.append(a)
    return np.stack(out_y, 1), np.stack(out_h, 1)


def stats(yp, yh, thr):
    m = np.abs(yh) > thr
    if m.sum() < 15:
        return dict(n=int(m.sum()))
    r = yp[m] * np.sign(yh[m]) / np.abs(yh[m])
    return dict(n=int(m.sum()), slope=float((yp * yh).sum() / (yh * yh).sum()), ratio_median=float(np.median(r)), ratio_q25=float(np.quantile(r, .25)),
                ratio_q75=float(np.quantile(r, .75)), frac_under_half=float((r < .5).mean()), frac_over_1p5=float((r > 1.5).mean()),
                frac_wrong_sign=float((r < 0).mean()))


def main():
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "figs").mkdir(parents=True, exist_ok=True)
    rows = []
    data = {n: load(n) for n in SETS}
    for name, (toks, P, G, speed, cmd) in data.items():
        curv = np.abs(G[:, 7, 2])
        cbin = np.where(curv < 0.1, "straight", np.where(curv < 0.4, "gentle", "sharp"))
        sbin = np.select([speed < 3, speed < 6, speed < 9], ["<3", "3-6", "6-9"], ">=9")
        ey, eh = arc_extrapolation(P)
        groups = [("all", np.ones(len(P), bool))] + [(f"curv {b}", cbin == b) for b in ("straight", "gentle", "sharp")] + \
                 [(f"speed {b}", sbin == b) for b in ("<3", "3-6", "6-9", ">=9")] + [(f"cmd {c}", cmd == c) for c in ("left", "straight", "right")]
        for gname, mk in groups:
            for t, i in H.items():
                for qty, plan, ext, hum, thr in (("lateral y", P[:, i, 1], ey[:, t - 1], G[:, i, 1], 0.5), ("heading", P[:, i, 2], eh[:, t - 1], G[:, i, 2], 0.05)):
                    for src, v in (("plan", plan), ("near-term arc", ext)):
                        s = stats(v[mk], hum[mk], thr)
                        rows.append(dict(set=name, group=gname, horizon_s=t, quantity=qty, source=src, **s))
    A = pd.DataFrame(rows)
    A.to_csv(OUT / "tables/amp_bins.csv", index=False)

    # gains fitted on navtrain
    toks, P, G, speed, cmd = data["navtrain"]
    g = {}
    g["single"] = float(sum((P[:, i, 1] * G[:, i, 1]).sum() for i in H.values()) / sum((P[:, i, 1] ** 2).sum() for i in H.values()))
    g["horizon"] = {str(t): float((P[:, i, 1] * G[:, i, 1]).sum() / (P[:, i, 1] ** 2).sum()) for t, i in H.items()}
    # slope of plan on human (the other regression direction), for reference
    g["plan_on_human"] = {str(t): float((P[:, i, 1] * G[:, i, 1]).sum() / (G[:, i, 1] ** 2).sum()) for t, i in H.items()}
    # held-out check: fit on half of navtrain tokens (even), evaluate lateral MSE on the other half and on navtest
    half = np.arange(len(P)) % 2 == 0
    gh = {str(t): float((P[half, i, 1] * G[half, i, 1]).sum() / (P[half, i, 1] ** 2).sum()) for t, i in H.items()}
    mse = lambda a, b, gg: float(np.mean([np.mean((gg[str(t)] * a[:, i, 1] - b[:, i, 1]) ** 2) for t, i in H.items()]))  # noqa: E731
    one = {str(t): 1.0 for t in H}
    tt, Pt, Gt, *_ = data["navtest"]
    g["lateral_mse_heldout_navtrain"] = dict(gain1=mse(P[~half], G[~half], one), fitted=mse(P[~half], G[~half], gh))
    g["lateral_mse_navtest"] = dict(gain1=mse(Pt, Gt, one), fitted_navtrain=mse(Pt, Gt, g["horizon"]))
    g["n_navtrain"] = len(P)
    (OUT / "gains.json").write_text(json.dumps(g, indent=1))
    print(json.dumps(g, indent=1))

    # figures
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4))
    col = {"all": "#222", "curv straight": "#8a949e", "curv gentle": "#1f77b4", "curv sharp": "#e8710a"}
    for name, ls in (("navtrain", "-"), ("navtest", "--")):
        for gname, c in col.items():
            s = A[(A.set == name) & (A.group == gname) & (A.quantity == "lateral y") & (A.source == "plan")].sort_values("horizon_s")
            axs[0].plot(s.horizon_s, s.slope, ls, marker="o", c=c, label=f"{name} {gname}" if name == "navtrain" else None)
            s2 = A[(A.set == name) & (A.group == gname) & (A.quantity == "heading") & (A.source == "plan")].sort_values("horizon_s")
            axs[1].plot(s2.horizon_s, s2.slope, ls, marker="o", c=c)
    axs[0].axhline(1, c="k", lw=.5); axs[1].axhline(1, c="k", lw=.5)
    axs[0].set_title("lateral y: slope plan on human (solid navtrain, dashed navtest)"); axs[1].set_title("heading: slope plan on human")
    axs[0].legend(fontsize=7); axs[0].set_xlabel("horizon (s)"); axs[1].set_xlabel("horizon (s)")
    tt, Pt, Gt, *_ = data["navtest"]
    curv = np.abs(Gt[:, 7, 2]); m = (np.abs(Gt[:, 7, 1]) > 1.0)
    r = (Pt[m, 7, 1] * np.sign(Gt[m, 7, 1]) / np.abs(Gt[m, 7, 1]))
    axs[2].hist(np.clip(r, -2, 3), bins=60, color="#1f77b4", alpha=.8)
    axs[2].axvline(1, c="k", lw=.8); axs[2].axvline(np.median(r), c="#e8710a", lw=2)
    axs[2].set_title(f"navtest, 4 s lateral plan / human (|y_h| > 1 m, n = {m.sum()}), median {np.median(r):.2f}")
    fig.tight_layout(); fig.savefig(OUT / "figs/amp_gain.png", dpi=110)


if __name__ == "__main__":
    main()
