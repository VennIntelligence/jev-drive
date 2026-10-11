#!/usr/bin/env python3
"""LONG-PROBE reads (plans/2026-10-11-long-probe-prereg.md): every checkpoint of the ladder on body1-hold-logs, the registered table and verdicts.

  predict  (GPU) --runs NAME=RUN_DIR ... [--s0]: each lc_train checkpoint, with the weights selected for each output group (lat, A, B, C), on all
           24 queries of the hold states -> long_contact/read/pred/<NAME>.npz (out_<head> (n, 24, 10) fp16); --s0 also re-reads BODY1 S0's
           checkpoints (step seeds 0 / 1, blind seed 0) through contact_head.predict -> pred/S0-*.npz. Every call appends to read/reads.jsonl.
  report   (CPU) --ladder RUNG=NAME_S0,NAME_S1 ... -> read/report/*.csv, ladder.md tables (copied to experiments/long_contact/results/)
Score of a rung = the mean of its two seeds' logits (S0's gate convention); per-seed reads are reported. AUC with ties at half, interval =
percentile log-cluster bootstrap (contact_head.auc_boot's draws, 10 000 resamples, seed 0); gains are paired (the same draws for both scores).
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "lib"))
import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import lc as LC  # noqa: E402
import lc_head as H  # noqa: E402
import b1 as B  # noqa: E402
import contact_head as C  # noqa: E402
import lc_train as TR  # noqa: E402

FAMS = ("log", "ot1", "yr1")
S0_RUNS = {"S0-step-s0": "full-step/20261010-024906", "S0-step-s1": "full-step/20261010-024909", "S0-blind-s0": "full-blind/20261010-024909"}
NON_PRIV = ("blind", "vis", "nat")
NB = 10_000


def rd():
    return LC.root() / "read"


def hold_rows():
    from jevdrive.data import splits
    ho, tr = splits.load(B.HOLD), splits.load(B.TRAIN)
    R = C.load_rows(FAMS, range(B.NSH), lambda log, hold: hold, steps=False)
    assert ho.mask(R["log"]).all() and not tr.mask(R["log"]).any()
    return R, (ho, tr)


# ---------------------------------------------------------------- predict
def cmd_predict(a):
    import torch
    from jevdrive.run import Run
    dev = torch.device("cuda")
    with Run("long_contact", "read/predict", config=vars(a)) as run:
        R, sp = hold_rows()
        for s in sp:
            run.use_split(s)
        X = TR.load_extra(R, nat=True, priv=True)
        n = len(R["gi"])
        G = lambda x, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(x)).to(dev, dt)  # noqa: E731
        V, Qp = TR.load_tokens_host(R["src"], n), G(R["q"])                    # host memory: 6.7 GB, gathered per batch
        P = {k: G(X[k], torch.float16) for k in ("obj", "rte", "lin", "trn")}
        (rd() / "pred").mkdir(parents=True, exist_ok=True)
        log = open(rd() / "reads.jsonl", "a")
        for spec in a.runs:
            name, d = spec.split("=")
            arr = dict(gi=R["gi"], fam=R["fam"])
            for h in H.HEADS:
                net, ck = H.load_ckpt(_pl.Path(d) / "ckpt.pt", dev, h)
                D = dict(V=V, Qp=Qp, E=G((R["ego"] - ck["emu"]) / ck["esd"]), **P)
                if net.nat:
                    D["nat"], D["hid"] = G((X["nat"] - ck["nmu"]) / ck["nsd"], torch.float16), G(X["hid"], torch.float16)
                arr[f"out_{h}"] = H.predict(net, D).astype(np.float16)
                del net
            np.savez(rd() / "pred" / f"{name}.npz", **arr)
            log.write(json.dumps(dict(t=time.strftime("%Y-%m-%d %H:%M:%S"), name=name, ckpt=d, read="hold rows, all heads", rung=ck["config"]["rung"],
                                      seed=ck["config"]["seed"], val={h: {k: ck["val"][h][k] for k in ("a", "b", "A", "B", "C")} for h in H.HEADS})) + "\n")
            log.flush()
            run.info("%s: predicted (%s)", name, ck["config"]["rung"])
        if a.s0:
            E0 = None
            for name, rel in S0_RUNS.items():
                net, ck = C.load_ckpt(C.sroot() / rel / "ckpt.pt", dev)
                Vd = V.to(dev) if net.vision else None                           # S0's predict indexes on the card
                out, _ = C.predict(net, Vd, G((R["ego"] - ck["emu"]) / ck["esd"]), Qp)
                del Vd
                np.savez(rd() / "pred" / f"{name}.npz", gi=R["gi"], fam=R["fam"], out=out.astype(np.float32))
                log.write(json.dumps(dict(t=time.strftime("%Y-%m-%d %H:%M:%S"), name=name, ckpt=str(C.sroot() / rel), read="hold rows (S0 re-read, positive control)")) + "\n")
                del net, E0
                E0 = None
        log.close()
        run.summary.update(runs=a.runs, s0=a.s0)


# ---------------------------------------------------------------- report
P1_SLOTS, P0_SLOTS = (0, 1, 16, 17, 18, 19), (0, 1)


def row_table(R, X, T, slots=P1_SLOTS):
    """One row per (hold state, query slot in `slots`): labels, masks, strata; pop = own (slots 0, 1) or arc (16-19)."""
    import pandas as pd
    cls, gt45 = C.classes(T)
    Y = C.targets(R)
    n, k = len(R["gi"]), len(slots)
    rep = lambda x: np.repeat(x, k)  # noqa: E731
    o = lambda x: x[:, list(slots)].ravel()  # noqa: E731
    speed = np.concatenate([np.load(B.cache_root() / d / "tab.npz")["speed"][m] for d, m, _ in R["src"]])
    sl = np.tile(np.array(slots), n)
    return pd.DataFrame(dict(
        i=rep(np.arange(n)), slot=sl, pop=np.where(sl < 2, "own", "arc"), log=rep(R["log"]), fam=rep(np.array(R["fams"])[R["fam"]]), cls=rep(cls[R["gi"]]),
        gt45=rep(gt45[R["gi"]]), speed=rep(speed), band=rep(LC.speed_band(speed)), a=o(Y["a"]), a_ok=o(Y["a_ok"]), b=o(Y["b"]), b_ok=o(Y["b_ok"]),
        a_cls=o(R["a_cls"]), A=o(X["a_lab"]), A_sub=o(X["a_sub"]), B=o(X["b_lab"]), B_kind=o(X["b_kind"]), C=o(X["c_lab"]), C_dv=o(X["c_dv"]),
        b_ahead=rep(X["b_ahead"]), c_ahead=rep(X["c_ahead"]), qok=o(R["qok"])))


def label_mask(D, k):
    """(y, mask) of label k on a row table."""
    if k == "a":
        return D.a.to_numpy(), D.a_ok.to_numpy()
    if k == "b":
        return D.b.to_numpy(), D.b_ok.to_numpy()
    lab = D[k].to_numpy()
    return lab == 1, (lab >= 0) & D.qok.to_numpy()


def read(y, scores, logs, n_boot=NB):
    r = H.auc_draws(y, scores, logs, n_boot)
    return {k: dict(auc=v[0], lo=H.ci(v[1])[0], hi=H.ci(v[1])[1], draws=v[1]) for k, v in r.items()}


def verdict(best, gain):
    """Registered lines (prereg section 6)."""
    if best["auc"] >= 0.80 and best["lo"] > 0.70 and gain["d"] >= 0.05 and gain["lo"] > 0:
        return "READABLE"
    if best["auc"] < 0.70 or gain["d"] < 0.02:
        return "NOT READABLE"
    return "PARTIAL"


def cmd_report(a):
    import pandas as pd
    from jevdrive.run import Run
    with Run("long_contact", "read/report", config=vars(a)) as run:
        R, sp = hold_rows()
        for s in sp:
            run.use_split(s)
        X = TR.load_extra(R)
        T = dict(np.load(B.root() / "taxonomy" / "tax.npz"))
        D = row_table(R, X, T)
        own_m = (D["pop"] == "own").to_numpy()
        pops = {"P1 own + arc": np.ones(len(D), bool), "P0 own": own_m}
        out = rd() / "report"
        out.mkdir(parents=True, exist_ok=True)
        lad = dict(x.split("=") for x in a.ladder)                       # rung -> "NAME_S0,NAME_S1"
        pr = {}
        for rung, names in lad.items():
            for s, nm in enumerate(names.split(",")):
                z = np.load(rd() / "pred" / f"{nm}.npz")
                assert (z["gi"] == R["gi"]).all()
                pr[(rung, s)] = {h: z[f"out_{h}"].astype(np.float32) for h in H.HEADS}
        ii, ss = D.i.to_numpy(), D.slot.to_numpy()
        head_of = dict(a="lat", b="lat", A="A", B="B", C="C")

        def score(rung, k, seed=None):
            col = H.IDX[k]
            if seed is not None:
                return pr[(rung, seed)][head_of[k]][ii, ss, col]
            return np.mean([pr[(rung, s)][head_of[k]][ii, ss, col] for s in (0, 1)], 0)
        logs = D.log.to_numpy()
        rows, gains, ver = [], [], {}
        for pname, pm in pops.items():
            for k in ("A", "B", "C", "a", "b"):
                y, m = label_mask(D, k)
                m = m & pm
                sc = {}
                for rung in lad:
                    sc[f"{rung}|mean"] = score(rung, k)
                    for s in (0, 1):
                        sc[f"{rung}|s{s}"] = score(rung, k, s)
                r = read(y[m], {kk: v[m] for kk, v in sc.items()}, logs[m])
                for kk, v in r.items():
                    rung, which = kk.split("|")
                    rows.append(dict(pop=pname, label=k, rung=rung, score=which, auc=v["auc"], lo=v["lo"], hi=v["hi"], n=int(m.sum()), pos=int(y[m].sum()),
                                     logs=len(set(logs[m])), logs_pos=len(set(logs[m][y[m]]))))
                gk = []
                pairs = [(x, "blind") for x in lad if x != "blind"] + [("nat", "vis"), ("priv", "nat"), ("blind_nat", "blind"), ("nat", "blind_nat")]
                for hi_, lo_ in pairs:
                    if hi_ in lad and lo_ in lad:
                        dd = r[f"{hi_}|mean"]["draws"] - r[f"{lo_}|mean"]["draws"]
                        cl, ch = H.ci(dd)
                        gk.append(dict(pop=pname, label=k, rung=hi_, vs=lo_, d=r[f"{hi_}|mean"]["auc"] - r[f"{lo_}|mean"]["auc"], lo=cl, hi=ch))
                gains += gk
                if k in LC.LONG and all(x in lad for x in NON_PRIV):
                    best = max(NON_PRIV, key=lambda x: r[f"{x}|mean"]["auc"])
                    g = next((x for x in gk if x["rung"] == best and x["vs"] == "blind"), dict(d=0.0, lo=0.0, hi=0.0))
                    ver[f"{pname}|{k}"] = dict(pop=pname, label=k, best=best, auc=r[f"{best}|mean"]["auc"], lo=r[f"{best}|mean"]["lo"], hi=r[f"{best}|mean"]["hi"],
                                               gain=g["d"], gain_lo=g["lo"], gain_hi=g["hi"], verdict=verdict(r[f"{best}|mean"], g), pos=int(y[m].sum()),
                                               logs_pos=len(set(logs[m][y[m]])), blind=r["blind|mean"]["auc"],
                                               priv=r["priv|mean"]["auc"] if "priv" in lad else np.nan)
        L = pd.DataFrame(rows)
        L.to_csv(out / "ladder.csv", index=False)
        pd.DataFrame(gains).to_csv(out / "gains.csv", index=False)
        (out / "verdict.json").write_text(json.dumps(ver, indent=1))
        # ---- positive control: S0's own checkpoints re-read (identity with results/s0_gate.md)
        s0 = {}
        for nm in S0_RUNS:
            f = rd() / "pred" / f"{nm}.npz"
            if f.exists():
                z = np.load(f)
                assert (z["gi"] == R["gi"]).all()
                s0[nm] = z["out"]
        pc = []
        if s0:
            ens = np.mean([s0["S0-step-s0"], s0["S0-step-s1"]], 0)
            S0s = {"S0 (mean of seeds)": ens, "S0 step seed 0": s0["S0-step-s0"], "S0 step seed 1": s0["S0-step-s1"], "S0 blind seed 0": s0["S0-blind-s0"]}
            s0sc = lambda o, col: o[ii, ss, col]  # noqa: E731
            for pname, pm in pops.items():
                for k, col in (("a", 0), ("b", 1)):
                    y, m = label_mask(D, k)
                    m = m & pm
                    r = read(y[m], {nm: s0sc(o, col)[m] for nm, o in S0s.items()}, logs[m])
                    pc += [dict(pop=pname, label=k, subset="all", score=nm, auc=v["auc"], lo=v["lo"], hi=v["hi"], n=int(m.sum()), pos=int(y[m].sum()),
                                logs_pos=len(set(logs[m][y[m]]))) for nm, v in r.items()]
                # lateral-by-class split of the agent read: positives by contact type against the same no-contact negatives
                ya, ma = label_mask(D, "a")
                ma = ma & pm
                Al = D.A.to_numpy()
                veh = D.a_cls.to_numpy() == 0
                neg = ma & ~ya
                for nm, pos in (("A-type: vehicle ahead in own lane (A positive)", ya & (Al == 1)),
                                ("other vehicle contact (lateral / go-around / drift / cut-in / crossing)", ya & (Al != 1) & veh),
                                ("other object (pedestrian, cyclist, generic object)", ya & (Al != 1) & ~veh)):
                    mm = ma & (pos | neg)
                    sc = {snm: s0sc(o, 0)[mm] for snm, o in S0s.items() if "seed" not in snm or "blind" in snm}
                    for rung in ("vis", "nat", "priv"):
                        if rung in lad:
                            sc[f"ladder {rung} (lat head)"] = score(rung, "a")[mm]
                    if "vis" in lad:
                        sc["ladder vis (A head)"] = score("vis", "A")[mm]
                    rr = read(pos[mm], sc, logs[mm], 2000)
                    pc += [dict(pop=pname, label="agent split", subset=nm, score=snm, auc=v["auc"], lo=v["lo"], hi=v["hi"], n=int(mm.sum()), pos=int(pos[mm].sum()),
                                logs_pos=len(set(logs[mm][pos[mm]]))) for snm, v in rr.items()]
            # the S0 lateral head as a score for the longitudinal labels (descriptive)
            for pname, pm in pops.items():
                for k in LC.LONG:
                    y, m = label_mask(D, k)
                    m = m & pm
                    r = read(y[m], {"S0 agent logit": s0sc(ens, 0)[m], "S0 boundary logit": s0sc(ens, 1)[m]}, logs[m], 2000)
                    pc += [dict(pop=pname, label=k, subset="S0 head as a score", score=nm, auc=v["auc"], lo=v["lo"], hi=v["hi"], n=int(m.sum()), pos=int(y[m].sum()))
                           for nm, v in r.items()]
        pd.DataFrame(pc).to_csv(out / "positive_control.csv", index=False)
        # ---- strata (descriptive, P1): speed band, sub-class, population, state family, user class, decision-relevant subsets; every rung's two-seed score
        st = []
        strata = [("speed band", LC.SPEED_NAMES[b], D.band.to_numpy() == b) for b in range(4)] + [("states", f, (D.fam == f).to_numpy()) for f in FAMS]
        strata += [("query", p_, (D["pop"] == p_).to_numpy()) for p_ in ("own", "arc")]
        strata += [("user class", B.CLS[c], (D.cls == c).to_numpy()) for c in range(4)] + [("> 45 deg", "yes", D.gt45.to_numpy())]
        for k in LC.LONG:
            y, m = label_mask(D, k)
            extra = []
            if k == "A":
                extra = [("sub-class", "A1 stopped", (D.A_sub == 1).to_numpy() | ~y), ("sub-class", "A2 moving", (D.A_sub == 2).to_numpy() | ~y)]
            if k == "B":
                extra = [("sub-class", "red light", (D.B_kind == 2).to_numpy() | ~y), ("sub-class", "stop sign", (D.B_kind == 1).to_numpy() | ~y),
                         ("decision-relevant", "eligible stop line <= 60 m ahead", D.b_ahead.to_numpy())]
            if k == "C":
                extra = [("decision-relevant", "eligible turn peak <= 60 m ahead", D.c_ahead.to_numpy())]
            for sname, val, sm in strata + extra:
                mm = m & sm
                if y[mm].sum() < 5 or (~y[mm]).sum() < 5:
                    st.append(dict(label=k, stratum=sname, value=val, n=int(mm.sum()), pos=int(y[mm].sum())))
                    continue
                r = read(y[mm], {rung: score(rung, k)[mm] for rung in lad}, logs[mm], 2000)
                for rung, v in r.items():
                    st.append(dict(label=k, stratum=sname, value=val, rung=rung, auc=v["auc"], lo=v["lo"], hi=v["hi"], n=int(mm.sum()), pos=int(y[mm].sum()),
                                   logs_pos=len(set(logs[mm][y[mm]]))))
        pd.DataFrame(st).to_csv(out / "strata.csv", index=False)
        # ---- operating points for the OOD lead times: per label and rung, the two-seed logit at 2 % of the P1 negatives
        thr = {}
        for k in LC.LONG:
            y, m = label_mask(D, k)
            thr[k] = {rung: float(np.quantile(score(rung, k)[m & ~y], 0.98)) for rung in lad}
        (out / "thresholds.json").write_text(json.dumps(thr, indent=1))
        run.summary.update(verdict=ver)
        with pd.option_context("display.width", 250, "display.max_rows", 500):
            print(L[L.score == "mean"].to_string(index=False))
            print(pd.DataFrame(gains).to_string(index=False))
            print(json.dumps(ver, indent=1))
            print(pd.DataFrame(pc).to_string(index=False))


KIN_SLOTS = P1_SLOTS


def kin_feats(R, slots=KIN_SLOTS):
    """Per (state, slot) kinematic features (prereg addendum 1): v0, a0 and the plan's speed-profile / curvature summary -> dict of (n * k,)."""
    a0 = np.concatenate([np.load(B.cache_root() / d / "tab.npz")["acc"][m][:, -1, 0] for d, m, _ in R["src"]]).astype(np.float32)
    v0 = np.concatenate([np.load(B.cache_root() / d / "tab.npz")["speed"][m] for d, m, _ in R["src"]]).astype(np.float32)
    q = R["q"][:, list(slots)].astype(np.float64)                                   # (n, k, 8, 3)
    p0 = np.concatenate([np.zeros_like(q[..., :1, :]), q], -2)
    seg = np.hypot(np.diff(p0[..., 0], axis=-1), np.diff(p0[..., 1], axis=-1))      # (n, k, 8) m per 0.5 s
    spd = seg / 0.5
    dyaw = np.abs(np.diff(np.unwrap(p0[..., 2], axis=-1), axis=-1))
    kap = dyaw / np.maximum(seg, 0.1)
    j = kap.argmax(-1)
    k = len(slots)
    rep = lambda x: np.repeat(x, k)  # noqa: E731
    F = dict(v0=rep(v0), a0=rep(a0), arc4=seg.sum(-1).ravel(), vmin=spd[..., 1:].min(-1).ravel(), vmax=spd.max(-1).ravel(),
             acc=((spd[..., -1] - spd[..., 0]) / 3.5).ravel(), ratio=(seg.sum(-1) / np.maximum(4 * v0[:, None], 2.0)).ravel(),
             kmax=kap.max(-1).ravel(), v_at_kmax=np.take_along_axis(spd, j[..., None], -1)[..., 0].ravel(), alat=(spd ** 2 * kap).max(-1).ravel())
    F |= {f"s{i}": spd[..., i].ravel() for i in range(8)}
    return {x: v.astype(np.float32) for x, v in F.items()}


KIN_MODELS = {"K1 v0": ["v0"], "K2 a0": ["a0"], "K3 v0 + a0": ["v0", "a0"],
              "K4 plan speed profile": ["v0", "a0", "arc4", "vmin", "vmax", "acc", "ratio"] + [f"s{i}" for i in range(8)]}
KIN_MODELS["K5 K4 + plan curvature"] = KIN_MODELS["K4 plan speed profile"] + ["kmax", "v_at_kmax", "alat"]
KIN_SINGLE = {"A": [("plan 4 s arc length", "arc4")], "B": [("plan min speed after 0.5 s", "vmin"), ("plan 4 s arc length", "arc4")],
              "C": [("plan max v^2 kappa", "alat"), ("plan speed at its max curvature", "v_at_kmax")], "a": [], "b": []}


def cmd_kin(a):
    """Prereg addendum 1: kinematic baselines (fit logs -> hold), the hard subset without a kinematic cue, every rung's AUC there."""
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("long_contact", "read/kin", config=vars(a)) as run:
        ho, tr = splits.load(B.HOLD), splits.load(B.TRAIN)
        run.use_split(ho), run.use_split(tr)
        Rf = C.load_rows(FAMS, range(B.NSH), lambda log, hold: ~hold & ~np.array([C.is_val(l) for l in log.tolist()]), steps=False)
        assert tr.mask(Rf["log"]).all()
        Xf, Ff = TR.load_extra(Rf), kin_feats(Rf)
        Rh, _ = hold_rows()
        Xh, Fh = TR.load_extra(Rh), kin_feats(Rh)
        T = dict(np.load(B.root() / "taxonomy" / "tax.npz"))
        Dh = row_table(Rh, Xh, T)
        Dfit = row_table(Rf, Xf, T)
        # hard subset per hold state: not decelerating at t0, and the logged driver's minimum speed over the next 8 s is >= 1 m/s below v0
        Z = np.load(LC.ctx_path())
        vl, tl = Z["vlog"][Rh["gi"]].astype(np.float32), Z["tlog"][Rh["gi"]].astype(np.float32)
        vmin8 = np.where(np.isfinite(vl) & (tl <= 8.0), vl, np.inf).min(1)
        k = len(KIN_SLOTS)
        a0, v0 = Fh["a0"].reshape(-1, k)[:, 0], Fh["v0"].reshape(-1, k)[:, 0]
        hard = np.repeat((a0 >= 0) & (v0 - vmin8 >= 1.0), k)
        hard0 = np.repeat(a0 >= 0, k)
        own = (Dh["pop"] == "own").to_numpy()
        pops = {"P1 own + arc": np.ones(len(Dh), bool), "P0 own": own, "P1 & H (no kinematic cue)": hard, "P0 & H": own & hard, "P1 & H0 (a0 >= 0)": hard0}
        # rung scores
        names = sorted({p.stem.rsplit("-s", 1)[0] for p in (rd() / "pred").glob("*-s[01].npz") if not p.stem.startswith("S0")})
        ii, ss = Dh.i.to_numpy(), Dh.slot.to_numpy()
        head_of = dict(a="lat", b="lat", A="A", B="B", C="C")
        rows = []
        for lab in ("A", "B", "C", "a", "b"):
            yf, mf = label_mask(Dfit, lab)
            yh, mh = label_mask(Dh, lab)
            sc = {}
            for r in names:
                zz = [np.load(rd() / "pred" / f"{r}-s{s_}.npz")[f"out_{head_of[lab]}"].astype(np.float32)[ii, ss, H.IDX[lab]] for s_ in (0, 1)]
                sc[f"rung {r}"] = 0.5 * (zz[0] + zz[1])
            rng = np.random.default_rng(0)
            fit_idx = np.flatnonzero(mf)
            fit_idx = rng.choice(fit_idx, min(len(fit_idx), 600_000), replace=False)
            for nm, cols in KIN_MODELS.items():
                Xtr = np.stack([Ff[c][fit_idx] for c in cols], 1)
                mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced")).fit(Xtr, yf[fit_idx])
                sc[nm] = mdl.decision_function(np.stack([Fh[c] for c in cols], 1))
            for nm, c in KIN_SINGLE.get(lab, []):
                sc[nm] = Fh[c].astype(float)
            for pname, pm in pops.items():
                m = mh & pm
                if yh[m].sum() >= 5 and (~yh[m]).sum() >= 5:
                    r = read(yh[m], {kk: v[m] for kk, v in sc.items()}, Dh.log.to_numpy()[m], 2000)
                    rows += [dict(pop=pname, label=lab, score=kk, auc=v["auc"], lo=v["lo"], hi=v["hi"], n=int(m.sum()), pos=int(yh[m].sum()),
                                  logs_pos=len(set(Dh.log.to_numpy()[m][yh[m]]))) for kk, v in r.items()]
                else:
                    rows.append(dict(pop=pname, label=lab, score="(too few positives)", n=int(m.sum()), pos=int(yh[m].sum())))
            run.info("%s done", lab)
        K = pd.DataFrame(rows)
        K.to_csv(rd() / "report" / "kin.csv", index=False)
        hs = dict(states=int(len(a0)), H=int(((a0 >= 0) & (v0 - vmin8 >= 1.0)).sum()), H0=int((a0 >= 0).sum()))
        (rd() / "report" / "kin_subsets.json").write_text(json.dumps(hs, indent=1))
        run.summary.update(hs)
        with pd.option_context("display.width", 250, "display.max_rows", 600):
            print(K.to_string(index=False))


def cmd_tables(a):
    """report/*.csv (+ OOD csv if present) -> report/tables.md: the Markdown tables of results/ladder.md (numbers straight from the csv)."""
    import pandas as pd
    rp = rd() / "report"
    L, G, PC, ST = (pd.read_csv(rp / f) for f in ("ladder.csv", "gains.csv", "positive_control.csv", "strata.csv"))
    ver = json.loads((rp / "verdict.json").read_text())
    f3 = lambda r: f"{r['auc']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]" if pd.notna(r.get("auc")) else "n/a"  # noqa: E731
    fd = lambda r: f"{r['d']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]"  # noqa: E731
    rungs = [r for r in ("blind", "vis", "nat", "priv", "blind_nat") if r in set(L.rung)]
    out = []
    for pop in ("P1 own + arc", "P0 own"):
        out += [f"### {pop}: AUC, two-seed score [95% log-cluster CI]", "",
                "| label | positives (logs) | " + " | ".join(rungs) + " | best non-priv - blind | nat - vis | verdict |", "|---|---|" + "---|" * (len(rungs) + 3)]
        for k in ("A", "B", "C", "a", "b"):
            x = L[(L["pop"] == pop) & (L.label == k) & (L.score == "mean")].set_index("rung")
            if x.empty:
                continue
            v = ver.get(f"{pop}|{k}", {})
            g = G[(G["pop"] == pop) & (G.label == k)]
            gb = g[(g.rung == v.get("best")) & (g.vs == "blind")]
            gn = g[(g.rung == "nat") & (g.vs == "vis")]
            name = {"a": "agent contact (S0 label)", "b": "boundary (S0 label)"}.get(k, k)
            out.append(f"| {name} | {int(x.iloc[0]['pos'])} ({int(x.iloc[0]['logs_pos'])}) | " + " | ".join(f3(x.loc[r]) for r in rungs)
                       + f" | {(v.get('best', '') + ' ' + fd(gb.iloc[0])) if len(gb) else '-'} | {fd(gn.iloc[0]) if len(gn) else '-'} | {v.get('verdict', '-')} |")
        out += ["", "Per seed (seed 0 / seed 1):", "", "| label | " + " | ".join(rungs) + " |", "|---|" + "---|" * len(rungs)]
        for k in ("A", "B", "C", "a", "b"):
            x = L[(L["pop"] == pop) & (L.label == k)]
            if x.empty:
                continue
            cell = lambda r: " / ".join(f"{x[(x.rung == r) & (x.score == s_)].auc.iloc[0]:.3f}" for s_ in ("s0", "s1"))  # noqa: E731
            out.append(f"| {k} | " + " | ".join(cell(r) for r in rungs) + " |")
        out += [""]
    out += ["### Gains (paired, same log-cluster draws)", "", "| population | label | rung | against | AUC difference [95% CI] |", "|---|---|---|---|---|"]
    out += [f"| {r['pop']} | {r['label']} | {r['rung']} | {r['vs']} | {fd(r)} |" for r in G.to_dict("records")]
    out += ["", "### Positive control and lateral split", "", "| population | label | subset | score | AUC [95% CI] | n | positives (logs) |", "|---|---|---|---|---|--:|---|"]
    for r in PC.to_dict("records"):
        out.append(f"| {r['pop']} | {r['label']} | {r.get('subset', '')} | {r['score']} | {f3(r)} | {int(r['n'])} | {int(r['pos'])}"
                   + (f" ({int(r['logs_pos'])})" if pd.notna(r.get('logs_pos')) else "") + " |")
    out += ["", "### Strata (P1, two-seed score, 2 000 resamples)", "", "| label | stratum | value | positives (logs) | " + " | ".join(rungs) + " |",
            "|---|---|---|---|" + "---|" * len(rungs)]
    for (k, sn, val), x in ST.groupby(["label", "stratum", "value"], sort=False):
        x = x.set_index("rung") if "rung" in x and x.rung.notna().any() else None
        if x is None:
            r0 = ST[(ST.label == k) & (ST.stratum == sn) & (ST.value == val)].iloc[0]
            out.append(f"| {k} | {sn} | {val} | {int(r0['pos'])} | " + " | ".join("too few" for _ in rungs) + " |")
            continue
        out.append(f"| {k} | {sn} | {val} | {int(x.iloc[0]['pos'])} ({int(x.iloc[0]['logs_pos'])}) | " + " | ".join(f3(x.loc[r]) if r in x.index else "" for r in rungs) + " |")
    if (rp / "kin.csv").exists():
        K = pd.read_csv(rp / "kin.csv")
        out += ["", "### Kinematic baselines and the hard subset (prereg addendum 1; descriptive)", "",
                "| population | label | score | AUC [95% CI] | n | positives (logs) |", "|---|---|---|---|--:|---|"]
        for r in K.to_dict("records"):
            out.append(f"| {r['pop']} | {r['label']} | {r['score']} | {f3(r) if pd.notna(r.get('auc')) else 'n/a'} | {int(r['n'])} | {int(r['pos'])}"
                       + (f" ({int(r['logs_pos'])})" if pd.notna(r.get('logs_pos')) else "") + " |")
    (rp / "tables.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


def cmd_pilot(a):
    """Pilot checklist (prereg section 7): reads the VALIDATION reads stored in the two pilot checkpoints (no hold row is touched)."""
    import torch
    res, ok = {}, True
    for spec in a.runs:
        name, d = spec.split("=")
        ck = torch.load(_pl.Path(d) / "ckpt.pt", map_location="cpu", weights_only=False)
        v = {h: {k: ck["val"][h][k] for k in ck["val"][h]} for h in H.HEADS}
        fin = all(np.isfinite(c[k]) for c in ck["curve"] for k in ("a", "b"))
        res[name] = dict(rung=ck["config"]["rung"], val=v, finite=fin)
        ok &= fin
    vis = next(r for r in res.values() if r["rung"] == "vis")
    pri = next(r for r in res.values() if r["rung"] == "priv")
    chk = dict(lateral_agent=vis["val"]["lat"]["a"] >= 0.80, lateral_boundary=vis["val"]["lat"]["b"] >= 0.90)
    for k in LC.LONG:
        chk[f"{k}_priv_ceiling"] = pri["val"][k][k] >= 0.90
        chk[f"{k}_val_positives"] = vis["val"][k][f"{k}_pos"] > 0
    ok &= all(chk.values())
    out = dict(checks=chk, passed=bool(ok), runs=res)
    (rd() / "pilot.json").parent.mkdir(parents=True, exist_ok=True)
    (rd() / "pilot.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    from jevdrive.run import cli_args
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("predict")
    p.add_argument("--runs", nargs="*", default=[], help="NAME=RUN_DIR")
    p.add_argument("--s0", action="store_true")
    cli_args(p)
    p = sp.add_parser("report")
    p.add_argument("--ladder", nargs="+", required=True, help="RUNG=NAME_S0,NAME_S1")
    cli_args(p)
    sp.add_parser("tables")
    cli_args(sp.add_parser("kin"))
    p = sp.add_parser("pilot")
    p.add_argument("--runs", nargs="+", required=True, help="NAME=RUN_DIR (the vis and priv pilots)")
    a = ap.parse_args()
    {"predict": cmd_predict, "report": cmd_report, "pilot": cmd_pilot, "tables": cmd_tables, "kin": cmd_kin}[a.cmd](a)
