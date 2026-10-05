"""op_dagger readouts (CPU, op-train env), exactly as plans/2026-10-06-dagger-prereg.md defines them.

Per held-out clip (roll/<model>/heldout-eval-*.npz):
  g1        open-loop yaw gain at step s: (phi1(yaw+2) - phi1(yaw-2)) / 4 deg per deg (decision 123's R definition, logged-frame source)
  gk1       the same for the action curvature (1e-3 / m per deg)
  kick10    closed loop, kick +-2 deg at step 1: sign-corrected heading offset at step 10 (2 s) / 2 deg (1 = held, > 1 = amplified, < 1 = restoring)
  sw10      closed loop, swerve +-0.5 m (moving clips): sign-corrected lateral offset at step 10 / at step 3 (< 1 = restoring)
  free_dy   free closed loop (no injection): |dy| at step 10 (m); free_dpsi |dpsi| (deg)
  rep_dphi  replay (logged states): mean |phi1 - phi1_shipped| over steps 0..10 (deg), drift on normal input
  ok10      the rollout stayed inside the cap (|dy| <= 1 m, |dpsi| <= 5 deg) to step 10
CIs: cluster bootstrap over WOD sequences (jevdrive.stats, B = 10000); paired against shipped and against the static control.

  python experiments/op_dagger/scripts/dg_report.py --models shipped dg1 st1 [dg2] --out experiments/op_dagger/results/pilot
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dg_common as C  # noqa: E402

GROUPS = {"launch": ("launch",), "moving": ("low", "mid", "cruise"), "all": ("launch", "low", "mid", "cruise")}


def load(model, name="heldout-eval"):
    fs = sorted(C.root("roll", model).glob(f"{name}-*.npz"))
    if not fs:
        return None
    zs = [dict(np.load(f, allow_pickle=True)) for f in fs]
    return {k: np.concatenate([z[k] for z in zs]) for k in zs[0]}


def per_clip(z, ship=None):
    """{clip_id: metrics} of one model's eval file."""
    out = {}
    idx = {}
    for i, (cid, arm) in enumerate(zip(z["clip_id"], z["arm"])):
        idx.setdefault(cid, {})[str(arm)] = i
    sidx = {}
    if ship is not None:
        for i, (cid, arm) in enumerate(zip(ship["clip_id"], ship["arm"])):
            sidx.setdefault(cid, {})[str(arm)] = i
    for cid, a in idx.items():
        m = dict(cat=str(z["cat"][a["free"]]), seq=str(z["seq"][a["free"]]))
        p, q = a["yaw+2"], a["yaw-2"]
        for s in (1, 2, 3, 5, 10):
            m[f"g{s}"] = (z["phi1"][p][s] - z["phi1"][q][s]) / 4.0
            m[f"gk{s}"] = (z["kappa"][p][s] - z["kappa"][q][s]) / 4.0 * 1e3
        r = [np.sign(z["exo"][a[k]][1]) * z["off"][a[k]][:, 2] / np.radians(2.0) for k in ("ckick+2", "ckick-2")]
        m["kick_curve"] = np.mean(r, 0)
        m["kick10"] = float(m["kick_curve"][C.K])
        m["kick_ok10"] = float(np.mean([z["ok"][a[k]][C.K] for k in ("ckick+2", "ckick-2")]))
        if "cswerve+0.5" in a:
            w = [np.sign(z["exo"][a[k]][1]) * z["off"][a[k]][:, 1] for k in ("cswerve+0.5", "cswerve-0.5")]
            w = np.mean(w, 0)
            m["sw_curve"] = w / max(w[3], 1e-3)
            m["sw10"] = float(m["sw_curve"][C.K])
        f = a["free"]
        m["free_dy"] = float(abs(z["off"][f][C.K, 1]))
        m["free_dpsi"] = float(np.degrees(abs(z["off"][f][C.K, 2])))
        m["free_ok10"] = float(z["ok"][f][C.K])
        if ship is not None and cid in sidx:
            m["rep_dphi"] = float(np.mean(np.abs(z["phi1"][a["replay"]] - ship["phi1"][sidx[cid]["replay"]])))
            m["rep_dkappa"] = float(np.mean(np.abs(z["kappa"][a["replay"]] - ship["kappa"][sidx[cid]["replay"]])) * 1e3)
        out[cid] = m
    return out


def sf_gain():
    """Engine check (c): step gains on decision 123's anchors, both frame sources; median over anchors with a bootstrap CI as decision 123."""
    out = {}
    for src in ("anchor", "log"):
        f = C.root("roll", "shipped") / f"sf-check-{src}.npz"
        if not f.exists():
            continue
        z = dict(np.load(f, allow_pickle=True))
        g = {}
        for cid in np.unique(z["clip_id"]):
            i = {str(a): k for k, a in enumerate(z["arm"]) if z["clip_id"][k] == cid}
            g[cid] = (z["phi1"][i["yaw+2"]] - z["phi1"][i["yaw-2"]]) / 4.0, str(z["cat"][i["yaw+2"]])
        rng = np.random.default_rng(0)
        for s in (1, 2, 3, 5, 10):
            x = np.array([v[0][s] for v in g.values()])
            bs = np.median(x[rng.integers(0, len(x), (10000, len(x)))], 1)
            out[f"{src}_s{s}"] = dict(n=len(x), median=float(np.median(x)), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))
        xl = np.array([v[0][1] for v in g.values() if v[1] == "launch"])
        out[f"{src}_s1_launch"] = dict(n=len(xl), median=float(np.median(xl)) if len(xl) else None)
    return out


def main():
    from jevdrive import stats
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--control", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ship = load("shipped")
    M = {m: per_clip(load(m), ship) for m in a.models if load(m) is not None}
    keys = ["g1", "g2", "g3", "gk1", "gk2", "gk3", "kick10", "sw10", "free_dy", "free_dpsi", "rep_dphi", "rep_dkappa", "kick_ok10", "free_ok10"]
    rows, pairs = [], []
    for gname, cats in GROUPS.items():
        for m, d in M.items():
            ids = [c for c, v in d.items() if v["cat"] in cats]
            for k in keys:
                x = np.array([d[c].get(k, np.nan) for c in ids], float)
                if np.isfinite(x).sum() < 3:
                    continue
                r = stats.bootstrap(x, groups=[d[c]["seq"] for c in ids])
                rows.append(dict(group=gname, model=m, metric=k, **r))
            refs = [r for r in ("shipped", a.control) if r and r != m and r in M]
            for ref in refs:
                common = [c for c in ids if c in M[ref]]
                for k in keys:
                    xa = np.array([d[c].get(k, np.nan) for c in common], float)
                    xb = np.array([M[ref][c].get(k, np.nan) for c in common], float)
                    if (np.isfinite(xa) & np.isfinite(xb)).sum() < 3:
                        continue
                    pairs.append(dict(group=gname, arm=f"{m} - {ref}", metric=k, **stats.paired(xa, xb, groups=[d[c]["seq"] for c in common])))
    stats.write_table(rows, out / "levels")
    stats.write_table(pairs, out / "paired")
    sg = sf_gain()
    eng = {}
    for f in sorted(C.root("check").glob("*.json")):
        eng[f.stem] = json.load(open(f))
    json.dump(dict(sf_gain=sg, engine=eng), open(out / "engine.json", "w"), indent=1)
    fig(M, out)
    print(json.dumps(sg, indent=1))


def fig(M, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = C.DT * np.arange(C.K + 1)
    cols = {"shipped": "#555555", "dg1": "#d1495b", "st1": "#2e86ab", "dg2": "#edae49"}
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
    for m, d in M.items():
        c = cols.get(m, None)
        for k, (cat, key, ylab, ttl) in enumerate([("launch", "g", "dphi1 / dyaw (deg/deg)", "open-loop yaw gain, launch"),
                                                    ("moving", "g", "dphi1 / dyaw (deg/deg)", "open-loop yaw gain, moving"),
                                                    ("all", "kick_curve", "heading offset / kick", "closed loop, 2 deg kick"),
                                                    ("moving", "sw_curve", "dy / dy(step 3)", "closed loop, 0.5 m swerve")]):
            ids = [i for i, v in d.items() if v["cat"] in GROUPS[cat]]
            if key == "g":
                ss = [1, 2, 3, 5, 10]
                y = [np.median([d[i][f"g{s}"] for i in ids]) for s in ss]
                ax[k].plot(np.array(ss) * C.DT, y, "o-", color=c, label=m)
            else:
                ys = np.array([d[i][key] for i in ids if key in d[i]])
                if len(ys):
                    ax[k].plot(t, np.median(ys, 0), "-", color=c, label=m)
                    ax[k].fill_between(t, *np.percentile(ys, [25, 75], 0), color=c, alpha=0.12)
            ax[k].set(title=ttl, xlabel="time after t0 (s)", ylabel=ylab)
            ax[k].axhline(0 if key == "g" else 1, color="k", lw=0.5)
    ax[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "dagger-pilot.png", dpi=130)


if __name__ == "__main__":
    main()
