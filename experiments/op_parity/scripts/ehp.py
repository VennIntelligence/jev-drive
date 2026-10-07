"""Ego-history probe (results/ego_history_probe.md): does P2H's / WA-JEPA's planned speed before a sharp turn ride on the logged ego history?

Inference only. The ego inputs (velocity, acceleration, 4-pose history) of a navtest token are replaced by a constant-velocity history at the
token's current speed, the plan is recomputed, planned distance / speed are compared with the unmodified input.

  select   token sets T (turn approach), P (pre-flip), C (matched straight controls)  -> $DATA_DIR/runs/op_parity/ehp/sets.csv (+ results/ehp_sets.csv)
           The rule is fixed in the result file before any plan is read; this step reads logs / inputs only.
  p2h      P2H10-F-s0 / s1 plans on ALL navtest tokens x variants (orig, cv, cvlong, acc0, pose)  -> ehp/p2h_<arm>_<variant>.npy (n, 8, 3)
  wareq    WA-JEPA request file for the selected tokens x the same variants                    -> ehp/wa_req_<variant>.npz
  report   tables with cluster bootstrap over logs                                             -> results/ehp_*.csv, printed markdown

Variants (all edit only ego inputs; command, images unchanged): v = token speed (|vel| at t0), dt = 0.5 s.
  orig    unmodified
  cv      full constant velocity: vx = v, vy = 0, ax = ay = 0, poses x = -v * dt * (3, 2, 1, 0), y = yaw = 0
  cvlong  longitudinal only: vx = v, ax = 0, past-pose x = -v * dt * k (y, yaw, vy, ay untouched)
  acc0    ax = ay = 0 only (poses and velocity untouched)
  pose    past poses x = -v * dt * k only (velocity / acceleration untouched)
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_R / "experiments/op_openloop/lib"), str(_pl.Path(__file__).parent)]
import argparse, json  # noqa: E401,E402

import numpy as np  # noqa: E402

from jevdrive.common import data_dir  # noqa: E402

OUT = data_dir() / "runs/op_parity/ehp"
RES = _R / "experiments/op_parity/results"
VARIANTS = ["orig", "cv", "cvlong", "acc0", "pose"]
ARMS = ["P2H10-F-s0", "P2H10-F-s1"]
DT = 0.5


def tab():
    return dict(np.load(data_dir() / "runs/op_parity/cache/lb_navtest/tab.npz"))


def edit(pose, vel, acc, v, variant):
    """pose (n, 4, 3), vel / acc (n, 4, 2) body frame (row 3 = t0), v (n,) -> edited copies."""
    pose, vel, acc = pose.copy(), vel.copy(), acc.copy()
    k = np.array([3, 2, 1, 0], np.float32)
    px = -v[:, None] * DT * k[None]
    if variant in ("cv", "cvlong"):
        vel[:, 3, 0] = v
        acc[:, 3, 0] = 0
    if variant == "cv":
        vel[:, 3, 1] = 0
        acc[:, 3, 1] = 0
        pose[:, :, 1] = 0
        pose[:, :, 2] = 0
    if variant == "acc0":
        acc[:, 3, :] = 0
    if variant in ("cv", "cvlong", "pose"):
        pose[:, :, 0] = px
    return pose, vel, acc


def features(t, variant):
    import parity_adapter as PA
    v = np.hypot(t["vel"][:, 3, 0], t["vel"][:, 3, 1])
    pose, vel, acc = edit(t["pose"], t["vel"], t["acc"], v, variant)
    return PA.ego_features(pose, vel, acc, t["cmd"][:, -1])        # cmd of t0 (n, 4); `present` = 1


def cmd_select(a):
    import pandas as pd
    t = tab()
    names = t["names"].tolist()
    pq = pd.read_parquet(data_dir() / "runs/op_probe/joint/navtest_tokens.parquet").set_index("token").loc[names]
    d = pd.DataFrame({"token": names, "log": t["log"], "ts": pq.ts.values.astype(np.int64), "v0": np.hypot(t["vel"][:, 3, 0], t["vel"][:, 3, 1]),
                      "yaw4": np.degrees(t["fut"][:, 7, 2]), "hist_yaw": np.degrees(t["pose"][:, 0, 2]),
                      "dv_hist": np.hypot(*t["vel"][:, 0].T) - np.hypot(*t["vel"][:, 3].T),
                      "cmd": pq.cmd.values})
    d["row"] = np.arange(len(d))
    d = d.sort_values(["log", "ts"]).reset_index(drop=True)
    d["set"] = ""
    base = (d.v0 >= 3) & (d.hist_yaw.abs() < 10)
    turn = d.cmd.isin(["left", "right"])
    d.loc[base & turn & (d.yaw4.abs() > 45), "set"] = "T"
    # P: straight-command tokens 0.5-3.0 s before a straight -> turn switch whose command run contains a token with |yaw4| > 45
    prev = d.groupby("log").cmd.shift(1)
    dtp = d.groupby("log").ts.diff() / 1e6
    sw = np.flatnonzero(turn.values & (prev.values == "straight") & (dtp.values < 0.75))
    for i in sw:
        j = i
        while j + 1 < len(d) and d.log[j + 1] == d.log[i] and d.cmd[j + 1] == d.cmd[i] and (d.ts[j + 1] - d.ts[j]) / 1e6 < 0.75:
            j += 1
        if (d.yaw4[i:j + 1].abs() > 45).any():
            for m in range(max(0, i - 6), i - 0):
                if d.log[m] == d.log[i] and d.cmd[m] == "straight" and 0.45 < (d.ts[i] - d.ts[m]) / 1e6 < 3.05 and base[m] and d["set"][m] == "":
                    d.loc[m, "set"] = "P"
    # controls: straight command, |yaw4| < 5, matched without replacement on (v0, dv_hist), other log, T first, then P
    pool = d[base & (d.cmd == "straight") & (d.yaw4.abs() < 5) & (d["set"] == "")]
    used = set()
    pairs = []
    for S in ["T", "P"]:
        for i in d.index[d["set"] == S]:
            cand = pool[(pool.log != d.log[i]) & ~pool.index.isin(used)]
            dist = np.hypot((cand.v0 - d.v0[i]) / 1.0, (cand.dv_hist - d.dv_hist[i]) / 0.5)
            if len(cand) and dist.min() < 2.0:
                j = dist.idxmin()
                used.add(j)
                pairs.append((i, j))
    d["match"] = ""
    for i, j in pairs:
        d.loc[j, "set"] = "C" + d["set"][i]
        d.loc[j, "match"] = d.token[i]
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "sets.csv", index=False)
    d[d["set"] != ""].to_csv(RES / "ehp_sets.csv", index=False)
    print(d["set"].value_counts().to_string())
    for s in ["T", "P", "CT", "CP"]:
        x = d[d["set"] == s]
        print(s, len(x), "logs", x.log.nunique(), "v0 med %.2f" % x.v0.median(), "dv_hist med %.2f" % x.dv_hist.median(), "yaw4 med %.1f" % x.yaw4.abs().median())


def sel_rows():
    import pandas as pd
    d = pd.read_csv(OUT / "sets.csv")
    return d[d["set"].notna() & (d["set"] != "")].reset_index(drop=True)


def adapt_poses(plan_mu, rows, mt):
    """plan_mu (m, 33, 15) of token rows -> (m, 8, 3) rear-axle poses at 0.5 .. 4 s (op_interp `base` adapter, as the bench export)."""
    import op_interp as OI
    from jevdrive import navsim_zs as Z
    z = {"plan_pos": plan_mu[:, :, 0:3], "plan_yaw": plan_mu[:, :, 11]}
    return np.stack([OI.adapt(z, i, mt, Z.T_OUT, mode="lever")[0] for i in range(len(rows))]).astype(np.float32)


def cmd_p2h(a):
    import torch
    import pp_train as T
    from jevdrive.bench.models import resolve
    from jevdrive.bench import navsim as BN
    from jevdrive.run import Run
    t = tab()
    names = t["names"].tolist()
    mt = json.loads((data_dir() / "runs/op_lb/lb_navtest/meta.json").read_text())
    assert names == mt["names"]
    sel = sel_rows()
    row = np.array([names.index(x) for x in sel.token])
    dev = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    eq = {"ego_orig_vs_cache_max_abs": float(np.abs(features(t, "orig") - t["ego"]).max())}
    assert eq["ego_orig_vs_cache_max_abs"] < 1e-5, eq
    with Run("op_parity", "ego-history-probe", config=vars(a)) as run:
        for arm in a.arms:
            m = resolve(arm, check=True)
            S = T.Store(["lb_navtest"], dev, need_side=False, frames=m.frames)
            assert S.tab["names"].tolist() == names
            model = T.load_pmodel(m.name, dev)
            sl = model.net.slices
            pi = np.arange(sl["plan"].start, sl["plan"].start + 495)

            def plan(rows, ego):
                mu = np.zeros((len(rows), 33, 15), np.float32)
                with torch.no_grad():
                    for i in range(0, len(rows), 128):
                        r = torch.as_tensor(rows[i:i + 128], device=dev)
                        o = model(S.front[r], ego[r], S.tc[r], None, None).float().cpu().numpy()
                        mu[i:i + len(r)] = o[:, pi].reshape(-1, 33, 15)
                return mu

            ego0 = S.ego
            mu_all = plan(np.arange(S.n), ego0)
            stored = BN.plan_file(m, "navtest")
            eq[arm] = {"stored_plan_file": str(stored)}
            if stored.exists():
                d = np.abs(mu_all - np.load(stored)["plan_mu"])
                eq[arm] |= {"orig_vs_stored_max_abs": float(d.max()), "orig_vs_stored_p99_row_max": float(np.percentile(d.max((1, 2)), 99))}
            run.info(f"{arm}: {eq[arm]}")
            out = {}
            for v in VARIANTS:
                ego = ego0 if v == "orig" else torch.as_tensor(features(t, v), device=dev)
                mu = mu_all[row] if v == "orig" else plan(row, ego)
                out[v] = adapt_poses(mu, row, mt)
            np.savez(OUT / f"p2h_{arm}.npz", tokens=np.array(sel.token), **out)
        (OUT / "equivalence.json").write_text(json.dumps(eq, indent=1))


def cmd_wareq(a):
    t = tab()
    names = t["names"].tolist()
    sel = sel_rows()
    z = np.load(a.full)
    krow = {k: i for i, k in enumerate(z["keys"].tolist())}
    zi = np.array([krow[x] for x in sel.token])
    ti = np.array([names.index(x) for x in sel.token])
    assert np.allclose(z["hist"][zi], t["pose"][ti], atol=1e-4), "request history differs from the cached pose history"
    assert np.allclose(z["ego"][zi], np.concatenate([t["vel"][ti, 3], t["acc"][ti, 3]], 1), atol=1e-4)
    v = np.hypot(t["vel"][ti, 3, 0], t["vel"][ti, 3, 1])
    keys, img, hist, ego, cmd = [], [], [], [], []
    for var in VARIANTS:
        p, vl, ac = edit(t["pose"][ti], t["vel"][ti], t["acc"][ti], v, var)
        keys += [f"{x}|{var}" for x in sel.token]
        img.append(z["img"][zi])
        hist.append(p)
        ego.append(np.concatenate([vl[:, 3], ac[:, 3]], 1))
        cmd.append(z["cmd"][zi])
    np.savez(a.out, keys=np.array(keys), img=np.concatenate(img), hist=np.concatenate(hist).astype(np.float32),
             ego=np.concatenate(ego).astype(np.float32), cmd=np.concatenate(cmd))
    print(len(keys), "requests ->", a.out)


def metrics(P):
    """poses (m, 8, 3) -> D2, D4 (m, path length at 2 / 4 s) and v13 (m/s, path length 1 -> 3 s / 2)."""
    xy = P[:, :, :2]
    L = np.cumsum(np.linalg.norm(np.diff(np.concatenate([np.zeros_like(xy[:, :1]), xy], 1), axis=1), axis=-1), 1)
    return {"D2": L[:, 3], "D4": L[:, 7], "v13": (L[:, 5] - L[:, 1]) / 2}


class Boot:
    """Cluster bootstrap over logs: statistic = sum / count of the drawn clusters (token-weighted mean). B 10 000, seed 0."""
    B = 10000

    def __init__(self, logs):
        self.u, self.inv = np.unique(logs, return_inverse=True)
        self.rng = np.random.default_rng(0)
        self.idx = self.rng.integers(0, len(self.u), (self.B, len(self.u)))

    def sums(self, x):
        s = np.bincount(self.inv, weights=np.nan_to_num(x), minlength=len(self.u))
        n = np.bincount(self.inv, weights=np.isfinite(x).astype(float), minlength=len(self.u))
        return s, n

    def mean(self, x):
        s, n = self.sums(x)
        b = s[self.idx].sum(1) / n[self.idx].sum(1)
        return s.sum() / n.sum(), b


def fmt(m, b, f=".2f"):
    lo, hi = np.percentile(b, [2.5, 97.5])
    return f"{m:{f}} [{lo:{f}}, {hi:{f}}]"


def contrast(x1, l1, x2, l2, paired=False):
    """mean(x1) - mean(x2), cluster bootstrap over logs. paired: same tokens, one resample (x1, x2 aligned); else independent."""
    if paired:
        m, b = Boot(l1).mean(x1 - x2)
    else:
        m1, b1 = Boot(l1).mean(x1)
        m2, b2 = Boot(l2).mean(x2)
        m, b = m1 - m2, b1 - b2
    return m, b


def cmd_report(a):
    import pandas as pd
    t = tab()
    names = t["names"].tolist()
    sel = sel_rows()
    ti = np.array([names.index(x) for x in sel.token])
    v0 = np.hypot(t["vel"][ti, 3, 0], t["vel"][ti, 3, 1])
    logd4 = metrics(t["fut"][ti])["D4"]
    P = {}
    for v in VARIANTS:
        for arm in ARMS:
            z = np.load(OUT / f"p2h_{arm}.npz")
            assert z["tokens"].tolist() == sel.token.tolist()
            P[("P2H-" + arm[-2:], v)] = metrics(z[v])
        P[("P2H", v)] = {k: (P[("P2H-s0", v)][k] + P[("P2H-s1", v)][k]) / 2 for k in ("D2", "D4", "v13")}
    wa = np.load(OUT / "wa.npz")
    wk = {k: i for i, k in enumerate(wa["keys"].tolist())}
    for v in VARIANTS:
        P[("WA-JEPA", v)] = metrics(wa["traj"][np.array([wk[f"{x}|{v}"] for x in sel.token])][:, :, :3])
    sets = {s: np.flatnonzero(sel["set"].values == s) for s in ["T", "CT", "P", "CP"]}
    lg = sel.log.values
    models = ["P2H", "P2H-s0", "P2H-s1", "WA-JEPA"]
    rows, ctx = [], []
    for s, ix in sets.items():
        r = {"set": s, "n": len(ix), "logs": len(set(lg[ix])), "v0 median": np.median(v0[ix]), "dv_hist median": np.median(sel.dv_hist.values[ix]),
             "v0 x 4 s (m)": np.mean(v0[ix] * 4), "logged D4": np.mean(logd4[ix])}
        for mo in ["P2H", "WA-JEPA"]:
            r[f"{mo} orig D4"] = np.mean(P[(mo, "orig")]["D4"][ix])
            r[f"{mo} orig v13 / v0"] = np.mean(P[(mo, "orig")]["v13"][ix] / v0[ix])
        ctx.append(r)
    pd.DataFrame(ctx).round(3).to_csv(RES / "ehp_context.csv", index=False)
    eff = {}
    for mo in models:
        for v in VARIANTS[1:]:
            for k in ("D2", "D4", "v13"):
                eff[(mo, v, k)] = P[(mo, v)][k] - P[(mo, "orig")][k]
    for (mo, v, k), e in eff.items():
        for s, ix in sets.items():
            m, b = Boot(lg[ix]).mean(e[ix])
            rows.append({"model": mo, "variant": v, "metric": k, "set": s, "n": len(ix), "mean": m, "lo": np.percentile(b, 2.5), "hi": np.percentile(b, 97.5)})
    E = pd.DataFrame(rows)
    E.round(3).to_csv(RES / "ehp_effects.csv", index=False)
    crow = []
    for (mo, v, k), e in eff.items():
        if mo.startswith("P2H-"):
            continue
        for sa, sb in [("T", "CT"), ("P", "CP")]:
            m, b = contrast(e[sets[sa]], lg[sets[sa]], e[sets[sb]], lg[sets[sb]])
            crow.append({"contrast": f"{sa} - {sb}", "model": mo, "variant": v, "metric": k, "mean": m, "lo": np.percentile(b, 2.5), "hi": np.percentile(b, 97.5)})
        if mo == "P2H":
            for s, ix in sets.items():
                m, b = contrast(e[ix], lg[ix], eff[("WA-JEPA", v, k)][ix], lg[ix], paired=True)
                crow.append({"contrast": f"P2H - WA-JEPA on {s}", "model": "-", "variant": v, "metric": k, "mean": m, "lo": np.percentile(b, 2.5), "hi": np.percentile(b, 97.5)})
    C = pd.DataFrame(crow)
    C.round(3).to_csv(RES / "ehp_contrasts.csv", index=False)
    # closure: share of the gap to constant speed that cv closes, ratio of sums
    clo = []
    for mo in ["P2H", "WA-JEPA"]:
        for s, ix in sets.items():
            num = eff[(mo, "cv", "D4")][ix]
            den = v0[ix] * 4 - P[(mo, "orig")]["D4"][ix]
            B = Boot(lg[ix])
            sn, nn = B.sums(num)
            sd, _ = B.sums(den)
            b = sn[B.idx].sum(1) / sd[B.idx].sum(1)
            clo.append({"model": mo, "set": s, "gap to v0 x 4 s (m)": den.mean(), "closure": sn.sum() / sd.sum(), "lo": np.percentile(b, 2.5), "hi": np.percentile(b, 97.5)})
    pd.DataFrame(clo).round(3).to_csv(RES / "ehp_closure.csv", index=False)
    # dose response: cv D4 effect on dv_hist (slope per m/s) and bins
    dose = []
    dv = sel.dv_hist.values
    for mo in ["P2H", "WA-JEPA"]:
        for s in ["T", "CT"]:
            ix = sets[s]
            e = eff[(mo, "cv", "D4")][ix]
            B = Boot(lg[ix])
            u, inv = B.u, B.inv

            def slope(w):                                  # cluster weights w (resample counts) -> weighted OLS slope
                ww = w[inv]
                xm = (ww * dv[ix]).sum() / ww.sum()
                ym = (ww * e).sum() / ww.sum()
                return (ww * (dv[ix] - xm) * (e - ym)).sum() / (ww * (dv[ix] - xm) ** 2).sum()
            W = np.stack([np.bincount(r, minlength=len(u)) for r in B.idx[:2000]])
            b = np.array([slope(w) for w in W])
            row = {"model": mo, "set": s, "slope (m per m/s of dv_hist)": slope(np.ones(len(u))), "lo": np.percentile(b, 2.5), "hi": np.percentile(b, 97.5)}
            for lab, m in [("dv<=0", dv[ix] <= 0), ("0<dv<=1", (dv[ix] > 0) & (dv[ix] <= 1)), ("dv>1", dv[ix] > 1)]:
                row[f"D4 effect, {lab}"] = f"{e[m].mean():.2f} (n={m.sum()})" if m.any() else "-"
            dose.append(row)
    pd.DataFrame(dose).round(3).to_csv(RES / "ehp_dose.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 500)
    print(pd.DataFrame(ctx).round(2).to_string(index=False))
    print(E[(E.metric == "D4") & (E.variant == "cv")].round(2).to_string(index=False))
    print(C[(C.metric == "D4") & (C.variant == "cv")].round(2).to_string(index=False))
    print(pd.DataFrame(clo).round(3).to_string(index=False))
    print(pd.DataFrame(dose).round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for n, f in [("select", cmd_select), ("p2h", cmd_p2h), ("wareq", cmd_wareq), ("report", cmd_report)]:
        q = sp.add_parser(n)
        q.set_defaults(fn=f)
        if n == "p2h":
            q.add_argument("--arms", nargs="+", default=ARMS)
        if n == "wareq":
            q.add_argument("--full", required=True)
            q.add_argument("--out", required=True)
    a = ap.parse_args()
    a.fn(a)
