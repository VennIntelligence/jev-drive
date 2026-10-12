"""junction_eyes step 0, read and report (plans/2026-10-12-prereg-A.md section 3.3, amendment 1 items 5, 6, 9, 10). Called by je0.py:

  read    (AlpaSim venv or envs/op-train, CPU) [--gdump FILE] -> $W/read/<ts>/ gates.json, ra.csv, rb.csv, rc.csv, je0.json, dec.parquet, bev.npz
  report  (Mac .venv) --read DIR (a local copy of the read dir) -> results/step0.md, results/step0_{ra,rb,rc}.csv, figs/je0_ra.png, figs/je0_rb.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from je0 import BINS, CR, DATA, FAR, NEAR, RP, TOPIC, WA, W

SETS = ("held452", "navtest16", "zero98")
CLUSTER = {"held452": "scene", "navtest16": "scene", "zero98": "scene", "navtest_all": "scene"}
S_IDX = {5: 0, 10: 1, 15: 2, 20: 3, 30: 4}
WIDE, ROAD = 29.4, 15.7
RA_PRED = ("A", "A-blind", "A-s7", "A-wa")
RB_PRED = ("B", "B-blind")


def jbin(jd):
    return np.select([jd == 0, (jd > 0) & (jd <= 10), (jd > 10) & (jd <= 20), (jd > 20) & (jd <= 40), (jd > 40) & (jd <= 60)], list(BINS[:5]), BINS[5])


def boot(cl, fn, B, seed=0):
    """Cluster bootstrap: fn(index array) -> dict of floats; -> {key: dict(v, lo, hi)} (percentile 95 %)."""
    u, inv = np.unique(cl, return_inverse=True)
    mem = [np.flatnonzero(inv == k) for k in range(len(u))]
    pt = fn(np.arange(len(cl)))
    rng = np.random.default_rng(seed)
    bs = [fn(np.concatenate([mem[k] for k in rng.integers(0, len(u), len(u))])) for _ in range(B)]
    out = {}
    for k, v in pt.items():
        x = np.array([b[k] for b in bs], float)
        f = np.isfinite(x)
        out[k] = dict(v=float(v), lo=float(np.quantile(x[f], 0.025)) if f.any() else np.nan, hi=float(np.quantile(x[f], 0.975)) if f.any() else np.nan)
    out["_clusters"] = int(len(u))
    return out


def tabcol(D, col):
    """Per-row column of the token tabs (cam, ego) for rows D."""
    tabs, out = {}, None
    for i, (f, r) in enumerate(zip(D.tab_file, D.tok_row)):
        if f not in tabs:
            p = DATA / "runs" / f / "tab.npz" if f.startswith("road_probe") else CR / f / "tab.npz"
            tabs[f] = np.load(p)[col]
        x = tabs[f][r]
        if out is None:
            out = np.zeros((len(D),) + x.shape, np.float64)
        out[i] = x
    return out


def ra_err(P, L):
    """P, L (n, 30) -> per-decision error (n, 5): median over the defined points C / L / R of the Euclidean errors at each s."""
    e = np.hypot(*(P.reshape(-1, 5, 3, 2) - L.reshape(-1, 5, 3, 2)).transpose(3, 0, 1, 2))
    with np.errstate(all="ignore"):
        import warnings
        warnings.simplefilter("ignore")
        return np.nanmedian(e, 2)


def iou_parts(pc, lab, xmax=40):
    """pred classes / labels (n, 48, 48) -> intersections, unions (n, 3) over the cells with x index < xmax."""
    p, l = pc[:, :xmax], lab[:, :xmax]
    I = np.stack([((p == c) & (l == c)).sum((1, 2)) for c in range(3)], 1).astype(np.float64)
    U = np.stack([((p == c) | (l == c)).sum((1, 2)) for c in range(3)], 1).astype(np.float64)
    return I, U


def read(a):
    import pandas as pd
    from jevdrive.run import Run
    import rp
    with Run("junction_eyes", "step0/read", config=vars(a)) as run:
        out = Path(run.dir)
        D = pd.read_parquet(W / "rows/all.parquet")
        RA = np.load(W / "labels/ra.npy")
        RB = np.load(W / "labels/rb.npy", mmap_mode="r")
        rs = D[D.set.isin(SETS + ("navtest_rest",))].copy()
        rs["bin"] = jbin(rs.jdist.values)
        rs["turn"], rs["straight"] = (rs.turn40 >= 20).values, (rs.turn40 < 5).values
        cam = tabcol(rs, "cam")
        G = {}
        # ---- G-lab
        lab = RA[rs.rid.values].reshape(-1, 5, 3, 2)
        d20 = np.isfinite(lab[:, 3, :, 0]).all(1)
        G["G_lab_cover"] = {s: float(d20[(rs.set == s).values & rs.turn.values].mean()) for s in SETS}
        G["G_lab_cover"]["pass"] = bool(all(v >= 0.9 for v in G["G_lab_cover"].values()))
        sg = {}
        for s in SETS:
            m = (rs.set == s).values & rs.straight.values
            y = np.nanmedian(lab[m, 3, :, 1], 0)
            sg[s] = dict(L=float(y[1]), C=float(y[0]), R=float(y[2]), ok=bool(y[1] > y[0] > y[2]))
        G["G_lab_sign"] = dict(**sg, **{"pass": all(v["ok"] for v in sg.values())})
        G["G_lab_sides"] = json.loads((W / "labels/glab.json").read_text())
        G["G_split"] = json.loads((W / "rows/rows.json").read_text())["G_split"]
        # ---- G-dump (executor 2's check or the same check by executor 1)
        gd = Path(a.gdump) if a.gdump else None
        G["G_dump"] = json.loads(gd.read_text()) if gd and gd.exists() else dict(pass_=None, note="not available")
        G["G_dump"]["pass"] = bool(G["G_dump"].get("pass", G["G_dump"].get("pass_")))
        # ---- G-ref
        try:
            z, c = np.load(WA / "lb_navtest-gref.npz"), np.load(WA / "lb_navtest.s0of3.npz")
            pos = {int(r): j for j, r in enumerate(c["rows"])}
            X = z["Cf"].astype(np.float32)
            Y = c["Cf"][[pos[int(r)] for r in z["rows"]]].astype(np.float32)
            cos = (X * Y).sum(-1) / (np.linalg.norm(X, axis=-1) * np.linalg.norm(Y, axis=-1) + 1e-9)
            G["G_ref"] = dict(tokens=int(len(X)), mean_cos=float(cos.mean()), min_cos=float(cos.min()), sample_mean_min=float(cos.mean(1).min()))
            G["G_ref"]["pass"] = bool(G["G_ref"]["mean_cos"] >= 0.999)
        except FileNotFoundError as e:
            G["G_ref"] = {"pass": False, "note": str(e)}
        # ---- predictions
        Pz = {}
        for f in sorted((W / "pred").glob("*.npz")):
            if "smoke" in f.stem:
                continue
            z = np.load(f)
            Pz[f.stem] = (z["rid"], z["pred"])

        def get(*names):
            ps = [Pz[n] for n in names if n in Pz]
            if len(ps) != len(names):
                return None
            rid = ps[0][0]
            assert all((p[0] == rid).all() for p in ps)
            M = np.full((len(D),) + ps[0][1].shape[1:], np.nan, np.float32)
            M[rid] = np.mean([p[1].astype(np.float32) for p in ps], 0)
            return M
        PR = {"A": get("A-Z-s0", "A-Z-s1"), "A-blind": get("A-blind-s0"), "A-s7": get("A-s7-s0", "A-s7-s1"), "A-wa": get("A-wa-s0", "A-wa-s1"),
              "A-Z-s0": get("A-Z-s0"), "A-Z-s1": get("A-Z-s1"), "old": get("old-Z-s0", "old-Z-s1"), "old-s0": get("old-Z-s0"), "old-s1": get("old-Z-s1")}
        # ---- G-repro (old ruler)
        if PR["old"] is not None:
            rep = {}
            for k in ("old", "old-s0", "old-s1"):
                Pm = PR[k]
                Lh = pd.read_parquet(RP / "labels/held.parquet")
                Ln = pd.read_parquet(RP / "real/labels/navtest.parquet")
                h_ = D[D.set == "held452"]
                ph = Pm[h_.rid.values]
                th = (Lh.turn40.values[h_.old_row.values] >= 20)
                eh = np.abs(ph[:, 4:6] - Lh[["L20", "R20"]].values[h_.old_row.values])[th]
                n_ = D[D.source == "navtest"]
                pn = Pm[n_.rid.values]
                ln = Ln.iloc[n_.old_row.values]
                mn = (ln.turn40.values >= 20) & (ln.jdist.values <= 20)
                en = np.abs(pn[:, 4:6] - ln[["L20", "R20"]].values)[mn]
                rep[k] = dict(held_turn20=float(np.nanmedian(eh)), navtest_near=float(np.nanmedian(en)))
            G["G_repro"] = dict(P_row=rep, held_range=[0.850, 1.046], navtest_range=[1.126, 1.398])
            G["G_repro"]["P_row_in_range"] = bool(0.850 <= rep["old"]["held_turn20"] <= 1.046 and 1.126 <= rep["old"]["navtest_near"] <= 1.398)
            fc = W / "repro/g_repro_c.json"                       # amendment 5: the check on rp_real's own fit rows decides the gate
            G["G_repro"]["c"] = json.loads(fc.read_text()) if fc.exists() else None
            G["G_repro"]["pass"] = bool(G["G_repro"]["c"] and G["G_repro"]["c"]["pass"])
        else:
            G["G_repro"] = {"pass": False, "note": "old-ruler predictions missing"}
        G["pass"] = bool(all(v["pass"] for k, v in G.items() if k.startswith("G_")))
        G["not_passed"] = [k for k, v in G.items() if k.startswith("G_") and not v["pass"]]
        (out / "gates.json").write_text(json.dumps(G, indent=1, default=float))
        run.info(json.dumps(G, default=float))
        if not G["pass"] and not a.force:
            run.info("a gate failed: no registered read")
            return
        # ---- R-A tables
        Lr = RA[rs.rid.values]
        err = {k: ra_err(PR[k][rs.rid.values], Lr) for k in ("A", "A-blind", "A-s7", "A-wa", "A-Z-s0", "A-Z-s1") if PR[k] is not None}
        cl = rs.scene.values.astype(str)
        rows = []
        groups = {s: (rs.set == s).values for s in SETS}
        groups["navtest_all"] = rs.set.isin(["navtest16", "navtest_rest"]).values
        for k, E in err.items():
            for g, gm in groups.items():
                if k in ("A-s7", "A-wa") and not g.startswith("navtest"):
                    continue
                for sub in ("turn", "straight"):
                    m = gm & rs[sub].values
                    for s, j in S_IDX.items():
                        e, b = E[m, j], rs.bin.values[m]
                        ok = np.isfinite(e)
                        e, b, c_ = e[ok], b[ok], cl[m][ok]

                        def fn(ix, e=e, b=b):
                            ee, bb = e[ix], b[ix]
                            r = {x: float(np.median(ee[bb == x])) if (bb == x).any() else np.nan for x in BINS}
                            nm, fm = np.isin(bb, NEAR), np.isin(bb, FAR)
                            r["near"] = float(np.median(ee[nm])) if nm.any() else np.nan
                            r["far"] = float(np.median(ee[fm])) if fm.any() else np.nan
                            r["all"] = float(np.median(ee)) if len(ee) else np.nan
                            return r
                        reg = (s in (15, 20)) and sub == "turn" and g != "navtest_all"
                        bt = boot(c_, fn, a.boot if reg else 200)
                        nb = {x: int((b == x).sum()) for x in BINS}
                        nb.update(near=int(np.isin(b, NEAR).sum()), far=int(np.isin(b, FAR).sum()), all=len(b))
                        rows += [dict(pred=k, set=g, subset=sub, s=s, bin=x, n=nb[x], median=bt[x]["v"], lo=bt[x]["lo"], hi=bt[x]["hi"], B=a.boot if reg else 200,
                                      clusters=bt["_clusters"]) for x in BINS + ("near", "far", "all")]
                        run.info(f"R-A {k} {g} {sub} s{s}: near {bt['near']['v']:.3f} far {bt['far']['v']:.3f}")
        TA = pd.DataFrame(rows)
        TA.to_csv(out / "ra.csv", index=False)
        # ---- R-B tables
        lab_b = np.asarray(RB[rs.rid.values])
        rb_rows = []
        IU = {}
        for k, names in (("B", ("B-Z-s0", "B-Z-s1")), ("B-blind", ("B-blind-s0",)), ("B-Z-s0", ("B-Z-s0",)), ("B-Z-s1", ("B-Z-s1",))):
            if not all(n in Pz for n in names):
                continue
            pc = np.zeros((len(rs), 48, 48), np.uint8)
            prob = None
            for n in names:
                rid, p = Pz[n]
                pos = {int(r): i for i, r in enumerate(rid)}
                x = p[[pos[int(r)] for r in rs.rid.values]].astype(np.float32)
                prob = x if prob is None else prob + x
            pc = prob.argmax(-1).astype(np.uint8)
            IU[k] = {"0-20": iou_parts(pc, lab_b, 40), "0-24": iou_parts(pc, lab_b, 48)}
            if k == "B":
                np.save(out / "rb_pred_classes.npy", pc)
        for k, parts in IU.items():
            for g, gm in groups.items():
                for sub in ("turn", "straight"):
                    m = gm & rs[sub].values
                    for reg_name, (I, U) in parts.items():
                        Ii, Ui, b = I[m], U[m], rs.bin.values[m]

                        def fn(ix, Ii=Ii, Ui=Ui, b=b):
                            r = {}
                            bb, Ix, Ux = b[ix], Ii[ix], Ui[ix]
                            ms = {x: (np.isin(bb, NEAR) if x == "near" else np.isin(bb, FAR) if x == "far" else bb == x) for x in BINS + ("near", "far")}
                            for c in range(3):
                                for x, mm in ms.items():
                                    u = Ux[mm, c].sum()
                                    r[f"{c}|{x}"] = float(Ix[mm, c].sum() / u) if u > 0 else np.nan
                                r[f"{c}|ratio_in_far"] = r[f"{c}|in junction"] / r[f"{c}|far"] if r[f"{c}|far"] and np.isfinite(r[f"{c}|far"]) else np.nan
                            return r
                        reg = reg_name == "0-20" and sub == "turn" and g != "navtest_all"
                        bt = boot(cl[m], fn, a.boot if reg else 200)
                        for key, v in bt.items():
                            if key.startswith("_"):
                                continue
                            c, x = key.split("|")
                            n = int(np.isin(b, NEAR).sum()) if x == "near" else int(np.isin(b, FAR).sum()) if x == "far" else int((b == x).sum()) if x in BINS else len(b)
                            rb_rows.append(dict(pred=k, set=g, subset=sub, region=reg_name, cls=int(c), bin=x, n=n, iou=v["v"], lo=v["lo"], hi=v["hi"],
                                                B=a.boot if reg else 200))
        TB = pd.DataFrame(rb_rows)
        TB.to_csv(out / "rb.csv", index=False)
        # ---- R-C (geometry only)
        az = np.degrees(np.arctan2(lab[..., 1] - cam[:, None, None, 1], lab[..., 0] - cam[:, None, None, 0]))      # (n, 5, 3)
        rc_rows = []
        X_, Y_ = np.meshgrid(np.arange(48) * 0.5 + 0.25, np.arange(48) * 0.5 - 11.75, indexing="ij")
        for g, gm in groups.items():
            m = gm & rs.turn.values
            for x in BINS + ("near", "far", "all"):
                mb = m & (np.isin(rs.bin.values, NEAR) if x == "near" else np.isin(rs.bin.values, FAR) if x == "far" else True if x == "all" else rs.bin.values == x)
                r = dict(set=g, bin=x, decisions=int(mb.sum()))
                for s in (10, 15, 20):
                    v = np.abs(az[mb, S_IDX[s]])
                    v = v[np.isfinite(v)]
                    r[f"n{s}"] = len(v)
                    r[f"wide{s}"] = float((v > WIDE).mean()) if len(v) else np.nan
                    r[f"road{s}"] = float((v > ROAD).mean()) if len(v) else np.nan
                    vc = np.abs(az[mb, S_IDX[s], 0])
                    vc = vc[np.isfinite(vc)]
                    r[f"wideC{s}"] = float((vc > WIDE).mean()) if len(vc) else np.nan
                idx = np.flatnonzero(mb)
                tot = out_ = 0
                for i in idx:
                    L_ = lab_b[i, :40]
                    cm = (L_ == 0) | (L_ == 2)
                    azc = np.degrees(np.abs(np.arctan2(Y_[:40] - cam[i, 1], X_[:40] - cam[i, 0])))
                    ahead = X_[:40] > cam[i, 0]
                    tot += int((cm & ahead).sum())
                    out_ += int((cm & ahead & (azc > WIDE)).sum())
                r["rb_cells"], r["rb_wide"] = tot, (out_ / tot if tot else np.nan)
                rc_rows.append(r)
        TC = pd.DataFrame(rc_rows)
        TC.to_csv(out / "rc.csv", index=False)
        # ---- Z-0 line and the 4.1 rule
        q = lambda g, s=20, k="A": TA[(TA.pred == k) & (TA.set == g) & (TA.subset == "turn") & (TA.s == s) & (TA.bin == "near")].iloc[0]  # noqa: E731
        rbq = lambda g, x: TB[(TB.pred == "B") & (TB.set == g) & (TB.subset == "turn") & (TB.region == "0-20") & (TB.cls == 0) & (TB.bin == x)].iloc[0]  # noqa: E731
        z0 = {}
        for g in ("held452", "navtest16"):
            r = q(g)
            z0[f"ra_near20_{g}"] = dict(median=r["median"], lo=r["lo"], hi=r["hi"], n=int(r["n"]), ok=bool(r["median"] < 0.5))
            rr = rbq(g, "ratio_in_far")
            z0[f"rb_offroad_ratio_{g}"] = dict(ratio=rr["iou"], lo=rr["lo"], hi=rr["hi"], iou_in=rbq(g, "in junction")["iou"], iou_far=rbq(g, "far")["iou"],
                                              ok=bool(rr["iou"] >= 0.8))
        close = all(v["ok"] for v in z0.values())
        verdict = "close: the frozen eye sees on the fair ruler (eye line not opened)" if close else "go to step 1"
        n16 = TC[(TC.set == "navtest16") & (TC.bin == "near")].iloc[0]
        rule = dict(share15=float(n16["wide15"]), share20=float(n16["wide20"]))
        rule["field_arms"] = bool(rule["share15"] > 0.5 or rule["share20"] > 0.5)
        rule["g1_s15_only"] = bool(rule["share20"] > 0.5)
        arms = ["L-F0", "L-F0-nav", "U-F0"] + (["L-W90", "L-3V"] if rule["field_arms"] else [])
        cnt = {g: dict(rows=int(gm.sum()), turn=int((gm & rs.turn.values).sum()), straight=int((gm & rs.straight.values).sum()),
                       clusters=int(rs.scene[gm].nunique())) for g, gm in groups.items()}
        # G1 inputs for pre-registration B (Z, blind, WA-Cf, Cinque-slot-7 on navtest-16 / held-452 turn near, s = 15 / 20)
        g1 = {}
        for k in ("A", "A-blind", "A-s7", "A-wa"):
            for g in ("navtest16", "held452"):
                for s in (15, 20):
                    sel = TA[(TA.pred == k) & (TA.set == g) & (TA.subset == "turn") & (TA.s == s) & (TA.bin == "near")]
                    if len(sel):
                        g1[f"{k}|{g}|s{s}"] = dict(median=float(sel.iloc[0]["median"]), lo=float(sel.iloc[0]["lo"]), hi=float(sel.iloc[0]["hi"]))
        S = dict(gates=G, provisional=G["not_passed"], z0=z0, verdict=verdict, rule41=rule, arms=arms, counts=cnt, g1_inputs=g1)
        (out / "je0.json").write_text(json.dumps(S, indent=1, default=float))
        # per-decision table for figures and BEV examples (zero-98: four decisions with the largest off-road + oncoming share ahead)
        dec = rs[["rid", "set", "scene", "k", "bin", "turn40"]].copy()
        for k, E in err.items():
            for s, j in S_IDX.items():
                dec[f"{k}:e{s}"] = E[:, j]
        dec.to_parquet(out / "dec.parquet", index=False)
        if "B" in IU:
            zm = np.flatnonzero((rs.set == "zero98").values & rs.turn.values)
            frac = np.array([((lab_b[i, :40] == 0) | (lab_b[i, :40] == 2)).mean() for i in zm])
            pick = zm[np.argsort(-frac)][:: max(1, len(zm) // 16)][:4]
            pc = np.load(out / "rb_pred_classes.npy")
            np.savez(out / "bev.npz", rid=rs.rid.values[pick], scene=rs.scene.values[pick], k=rs.k.values[pick], label=lab_b[pick], pred=pc[pick],
                     ra=lab[pick], ra_pred=PR["A"][rs.rid.values[pick]].reshape(-1, 5, 3, 2), cam=cam[pick])
        run.summary.update(verdict=verdict, z0=z0, rule41=rule)
        run.info(f"Z-0: {verdict}; {json.dumps(z0, default=float)}; 4.1 {json.dumps(rule)}")


# ---------------------------------------------------------------- report (Mac)
def report(a):
    import shutil
    import sys
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    sys.path.insert(0, str(TOPIC.parents[1] / "research"))
    import plot_style  # noqa: F401
    Rd = Path(a.read)
    S = json.loads((Rd / "je0.json").read_text())
    TA, TB, TC = pd.read_csv(Rd / "ra.csv"), pd.read_csv(Rd / "rb.csv"), pd.read_csv(Rd / "rc.csv")
    res, figs = TOPIC / "results", TOPIC / "figs"
    for f, g in (("ra.csv", "step0_ra.csv"), ("rb.csv", "step0_rb.csv"), ("rc.csv", "step0_rc.csv"), ("gates.json", "step0_gates.json"), ("je0.json", "step0.json")):
        shutil.copy(Rd / f, res / g)
    f2 = lambda x: "-" if x is None or not np.isfinite(x) else f"{x:.2f}"  # noqa: E731
    f3 = lambda x: "-" if x is None or not np.isfinite(x) else f"{x:.3f}"  # noqa: E731
    xb = list(BINS)
    lbl = {"A": "probe, Cinque tokens (2 seeds)", "A-blind": "blind (ego + command)", "A-s7": "Cinque slot 7 only", "A-wa": "WA-Cf (reference)"}
    col = {"A": "#0072B2", "A-blind": "#999999", "A-s7": "#56B4E9", "A-wa": "#E69F00"}
    # figure 1: R-A s = 20 by junction bin, turn decisions, three read sets
    fig, axs = plt.subplots(1, 3, figsize=(6.875, 2.4), sharey=True)
    for ax, g in zip(axs, ("navtest16", "held452", "zero98")):
        for j, k in enumerate(("A", "A-blind", "A-s7", "A-wa")):
            t = TA[(TA.pred == k) & (TA.set == g) & (TA.subset == "turn") & (TA.s == 20)].set_index("bin")
            if not len(t):
                continue
            y = t.loc[xb, "median"].values
            lo, hi = t.loc[xb, "lo"].values, t.loc[xb, "hi"].values
            x = np.arange(len(xb)) + (j - 1.5) * 0.08
            ax.errorbar(x, y, yerr=[np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)], color=col[k], marker="o", ms=2.5, lw=1, capsize=1.5, label=lbl[k])
        ax.axhline(0.5, color="k", lw=0.5, ls=(0, (2, 3)))
        ax.set_yscale("log")
        ax.set_xticks(np.arange(len(xb)), ["in", "0-10", "10-20", "20-40", "40-60", "none"], fontsize=7)
        ax.set_title({"navtest16": "navtest-16 (real)", "held452": "held-452 (reconstructed)", "zero98": "zero-98 (reconstructed)"}[g], fontsize=8)
        ax.set_xlabel("route distance to junction (m)", fontsize=7)
    axs[0].set_ylabel("R-A error at s = 20 m, median (m)", fontsize=7)
    axs[0].legend(fontsize=6, frameon=False, loc="upper right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figs / f"je0_ra.{ext}", dpi=300)
    # figure 2: R-B IoU by bin (off-road, oncoming), turn decisions, 0-20 m
    fig, axs = plt.subplots(1, 2, figsize=(6.875, 2.4), sharey=True)
    for ax, c in zip(axs, (0, 2)):
        for g, mk in (("navtest16", "o"), ("held452", "s"), ("zero98", "^")):
            for k, ls in (("B", "-"), ("B-blind", ":")):
                t = TB[(TB.pred == k) & (TB.set == g) & (TB.subset == "turn") & (TB.region == "0-20") & (TB.cls == c)].set_index("bin")
                if not len(t):
                    continue
                y = t.loc[xb, "iou"].values
                lo, hi = t.loc[xb, "lo"].values, t.loc[xb, "hi"].values
                ax.errorbar(np.arange(len(xb)), y, yerr=[np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)], marker=mk, ms=2.5, lw=1, ls=ls, capsize=1.5,
                            color={"navtest16": "#0072B2", "held452": "#D55E00", "zero98": "#009E73"}[g] if k == "B" else "#999999",
                            label=f"{g}{' blind' if k != 'B' else ''}")
        ax.set_xticks(np.arange(len(xb)), ["in", "0-10", "10-20", "20-40", "40-60", "none"], fontsize=7)
        ax.set_title({0: "off-road class", 2: "oncoming class (not scored in junction)"}[c], fontsize=8)
        ax.set_xlabel("route distance to junction (m)", fontsize=7)
    axs[0].set_ylabel("R-B IoU, cells 0-20 m ahead", fontsize=7)
    axs[0].legend(fontsize=5.5, frameon=False, ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figs / f"je0_rb.{ext}", dpi=300)
    # markdown
    G = S["gates"]
    L = ["# junction_eyes step 0: the frozen eye on the new rulers", "",
         "Pre-registration: [plans/2026-10-12-prereg-A.md](../plans/2026-10-12-prereg-A.md) (pushed before any label or read; amendments 1-2). "
         f"Scripts `scripts/je0.py`, `scripts/je0_read.py`, `lib/je_ruler.py`; box `$DATA_DIR/runs/junction_eyes/step0/` (read `{Rd.name}`). "
         "Figures [figs/je0_ra.png](../figs/je0_ra.png), [figs/je0_rb.png](../figs/je0_rb.png). Rows: [rows.md](rows.md).", "", "## Gates", ""]
    for k, v in G.items():
        if isinstance(v, dict):
            L.append(f"- {k}: **{'pass' if v.get('pass') else 'FAIL'}**; " + json.dumps({kk: vv for kk, vv in v.items() if kk != "pass"}, default=float)[:400])
    z0 = S["z0"]
    L += ["", "## Z-0 line (registered)", "",
          "Close iff R-A s = 20 m, turn decisions, near bucket (in junction + 0-20 m) median < 0.5 m on held-452 and on navtest-16, and R-B off-road "
          "IoU (0-20 m) in junction / 20-60 m >= 0.8 on both.", "",
          "| quantity | held-452 | navtest-16 |", "|:--|--:|--:|",
          f"| R-A s = 20 near, median (m) | {f2(z0['ra_near20_held452']['median'])} [{f2(z0['ra_near20_held452']['lo'])}, {f2(z0['ra_near20_held452']['hi'])}] "
          f"(n {z0['ra_near20_held452']['n']}) | {f2(z0['ra_near20_navtest16']['median'])} [{f2(z0['ra_near20_navtest16']['lo'])}, "
          f"{f2(z0['ra_near20_navtest16']['hi'])}] (n {z0['ra_near20_navtest16']['n']}) |",
          f"| R-B off-road IoU in junction / 20-60 m | {f2(z0['rb_offroad_ratio_held452']['ratio'])} [{f2(z0['rb_offroad_ratio_held452']['lo'])}, "
          f"{f2(z0['rb_offroad_ratio_held452']['hi'])}] ({f3(z0['rb_offroad_ratio_held452']['iou_in'])} / {f3(z0['rb_offroad_ratio_held452']['iou_far'])}) | "
          f"{f2(z0['rb_offroad_ratio_navtest16']['ratio'])} [{f2(z0['rb_offroad_ratio_navtest16']['lo'])}, {f2(z0['rb_offroad_ratio_navtest16']['hi'])}] "
          f"({f3(z0['rb_offroad_ratio_navtest16']['iou_in'])} / {f3(z0['rb_offroad_ratio_navtest16']['iou_far'])}) |", "",
          f"**Verdict: {S['verdict']}.**", ""]
    # R-A table
    L += ["## R-A: turn decisions, error at s = 20 m by distance to the junction", "",
          "| read set | predictor | " + " | ".join(xb) + " | near | far |", "|:--|:--" + "|--:" * (len(xb) + 2) + "|"]
    for g in ("navtest16", "held452", "zero98"):
        for k in ("A", "A-blind", "A-s7", "A-wa"):
            t = TA[(TA.pred == k) & (TA.set == g) & (TA.subset == "turn") & (TA.s == 20)].set_index("bin")
            if not len(t):
                continue
            L.append(f"| {g} | {lbl[k]} | " + " | ".join(f"{f2(t.loc[x, 'median'])} ({int(t.loc[x, 'n'])})" for x in xb)
                     + f" | {f2(t.loc['near', 'median'])} [{f2(t.loc['near', 'lo'])}, {f2(t.loc['near', 'hi'])}] | {f2(t.loc['far', 'median'])} "
                     f"[{f2(t.loc['far', 'lo'])}, {f2(t.loc['far', 'hi'])}] |")
    L += ["", "Cells: median over decisions of the per-decision median point error (C, L, R), m; (n decisions); brackets: 95 % cluster bootstrap "
          "(log segments / scenes), B 2000 on the registered cells. WA-Cf and Cinque-slot-7 rows are trained on J_nav only and read on real frames "
          "only (reference rows, not methods).", "",
          "R-A by arc length, turn decisions, near bucket (exploration):", "", "| read set | predictor | s = 5 | 10 | 15 | 20 | 30 |", "|:--|:--|--:|--:|--:|--:|--:|"]
    for g in ("navtest16", "held452", "zero98"):
        for k in ("A", "A-blind", "A-s7", "A-wa"):
            t = TA[(TA.pred == k) & (TA.set == g) & (TA.subset == "turn") & (TA.bin == "near")].set_index("s")
            if len(t):
                L.append(f"| {g} | {lbl[k]} | " + " | ".join(f2(t.loc[s, "median"]) for s in (5, 10, 15, 20, 30)) + " |")
    L += ["", "Straight decisions, s = 20 m (G2 baseline for step 1):", "", "| read set | predictor | near | far | all |", "|:--|:--|--:|--:|--:|"]
    for g in ("navtest16", "held452"):
        for k in ("A", "A-blind"):
            t = TA[(TA.pred == k) & (TA.set == g) & (TA.subset == "straight") & (TA.s == 20)].set_index("bin")
            L.append(f"| {g} | {lbl[k]} | {f2(t.loc['near', 'median'])} | {f2(t.loc['far', 'median'])} | {f2(t.loc['all', 'median'])} |")
    # R-B table
    L += ["", "## R-B: IoU, cells 0-20 m ahead, turn decisions", "", "| read set | predictor | class | " + " | ".join(xb) + " | in / far |",
          "|:--|:--|:--" + "|--:" * (len(xb) + 1) + "|"]
    for g in ("navtest16", "held452", "zero98"):
        for k in ("B", "B-blind"):
            for c in (0, 1, 2):
                t = TB[(TB.pred == k) & (TB.set == g) & (TB.subset == "turn") & (TB.region == "0-20") & (TB.cls == c)].set_index("bin")
                if not len(t):
                    continue
                L.append(f"| {g} | {'probe' if k == 'B' else 'blind'} | {('off-road', 'drivable', 'oncoming')[c]} | "
                         + " | ".join(f2(t.loc[x, "iou"]) for x in xb) + f" | {f2(t.loc['ratio_in_far', 'iou'])} [{f2(t.loc['ratio_in_far', 'lo'])}, "
                         f"{f2(t.loc['ratio_in_far', 'hi'])}] |")
    L += ["", "IoU pooled over the decisions of a bucket (sum of intersections / sum of unions); in junction every lane cell is class 1 by "
          "definition, so only the off-road class is read there.", ""]
    # R-C
    L += ["## R-C: share of R-A label points outside the wide slot (|azimuth| > 29.4 deg) and the road slot (> 15.7 deg), turn decisions", "",
          "| read set | bucket | decisions | wide s10 | wide s15 | wide s20 | road s15 | road s20 | centre only, wide s20 | R-B off-road + oncoming cells beyond 29.4 deg |",
          "|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for g in ("navtest16", "held452", "zero98"):
        for x in xb + ["near", "far"]:
            t = TC[(TC.set == g) & (TC.bin == x)]
            if not len(t):
                continue
            r = t.iloc[0]
            L.append(f"| {g} | {x} | {int(r.decisions)} | {f2(r.wide10)} | {f2(r.wide15)} | {f2(r.wide20)} | {f2(r.road15)} | {f2(r.road20)} | "
                     f"{f2(r.wideC20)} | {f2(r.rb_wide)} |")
    rl = S["rule41"]
    L += ["", f"Registered 4.1 rule (navtest-16, turn, near): share beyond 29.4 deg at s = 15 {f2(rl['share15'])}, at s = 20 {f2(rl['share20'])} -> "
          f"field arms (L-W90, L-3V) {'RUN' if rl['field_arms'] else 'not run'}; F0 G1 at s = 15 only: {rl['g1_s15_only']}. "
          f"Recommended step-1 arm set: {', '.join(S['arms'])}.", ""]
    (res / "step0.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
