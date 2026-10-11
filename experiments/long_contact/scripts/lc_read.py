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
        V, Qp = C.load_tokens(R["src"], n, dev), G(R["q"])
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
                out, _ = C.predict(net, V if net.vision else None, G((R["ego"] - ck["emu"]) / ck["esd"]), Qp)
                np.savez(rd() / "pred" / f"{name}.npz", gi=R["gi"], fam=R["fam"], out=out.astype(np.float32))
                log.write(json.dumps(dict(t=time.strftime("%Y-%m-%d %H:%M:%S"), name=name, ckpt=str(C.sroot() / rel), read="hold rows (S0 re-read, positive control)")) + "\n")
                del net, E0
                E0 = None
        log.close()
        run.summary.update(runs=a.runs, s0=a.s0)


# ---------------------------------------------------------------- report
def own_table(R, X, T):
    """One row per (hold state, own slot): labels, masks, strata."""
    import pandas as pd
    cls, gt45 = C.classes(T)
    Y = C.targets(R)
    n = len(R["gi"])
    rep = lambda x: np.repeat(x, 2)  # noqa: E731
    o = lambda x: x[:, :2].ravel()  # noqa: E731
    speed = np.concatenate([np.load(B.cache_root() / d / "tab.npz")["speed"][m] for d, m, _ in R["src"]])
    return pd.DataFrame(dict(
        i=rep(np.arange(n)), slot=np.tile(C.OWN, n), log=rep(R["log"]), fam=rep(np.array(R["fams"])[R["fam"]]), cls=rep(cls[R["gi"]]), gt45=rep(gt45[R["gi"]]),
        speed=rep(speed), band=rep(LC.speed_band(speed)), a=o(Y["a"]), a_ok=o(Y["a_ok"]), b=o(Y["b"]), b_ok=o(Y["b_ok"]), a_cls=o(R["a_cls"]),
        A=o(X["a_lab"]), A_sub=o(X["a_sub"]), B=o(X["b_lab"]), B_kind=o(X["b_kind"]), C=o(X["c_lab"]), C_dv=o(X["c_dv"]),
        b_ahead=rep(X["b_ahead"]), c_ahead=rep(X["c_ahead"]), qok=o(R["qok"])))


def label_mask(D, k):
    """(y, mask) of label k on the own-plan table."""
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
        D = own_table(R, X, T)
        out = rd() / "report"
        out.mkdir(parents=True, exist_ok=True)
        lad = dict(x.split("=") for x in a.ladder)                       # rung -> "NAME_S0,NAME_S1"
        pr = {}
        for rung, names in lad.items():
            for s, nm in enumerate(names.split(",")):
                z = np.load(rd() / "pred" / f"{nm}.npz")
                assert (z["gi"] == R["gi"]).all()
                pr[(rung, s)] = {h: z[f"out_{h}"].astype(np.float32) for h in H.HEADS}
        own = lambda o, col: o[:, :2, col].ravel()  # noqa: E731
        head_of = dict(a="lat", b="lat", A="A", B="B", C="C")

        def score(rung, k, seed=None):
            col = H.IDX[k]
            if seed is not None:
                return own(pr[(rung, seed)][head_of[k]], col)
            return np.mean([own(pr[(rung, s)][head_of[k]], col) for s in (0, 1)], 0)
        logs = D.log.to_numpy()
        rows, gains, ver = [], [], {}
        for k in ("A", "B", "C", "a", "b"):
            y, m = label_mask(D, k)
            sc = {}
            for rung in lad:
                sc[f"{rung}|mean"] = score(rung, k)
                for s in (0, 1):
                    sc[f"{rung}|s{s}"] = score(rung, k, s)
            r = read(y[m], {kk: v[m] for kk, v in sc.items()}, logs[m])
            for kk, v in r.items():
                rung, which = kk.split("|")
                rows.append(dict(label=k, rung=rung, score=which, auc=v["auc"], lo=v["lo"], hi=v["hi"], n=int(m.sum()), pos=int(y[m].sum()),
                                 logs=len(set(logs[m])), logs_pos=len(set(logs[m][y[m]]))))
            if "blind" in lad:
                for rung in lad:
                    if rung == "blind":
                        continue
                    dd = r[f"{rung}|mean"]["draws"] - r["blind|mean"]["draws"]
                    lo, hi = H.ci(dd)
                    gains.append(dict(label=k, rung=rung, d=r[f"{rung}|mean"]["auc"] - r["blind|mean"]["auc"], lo=lo, hi=hi))
            if k in LC.LONG and all(x in lad for x in NON_PRIV):
                best = max(NON_PRIV, key=lambda x: r[f"{x}|mean"]["auc"])
                g = next((x for x in gains if x["label"] == k and x["rung"] == best), dict(d=0.0, lo=0.0, hi=0.0))
                ver[k] = dict(best=best, auc=r[f"{best}|mean"]["auc"], lo=r[f"{best}|mean"]["lo"], hi=r[f"{best}|mean"]["hi"], gain=g["d"], gain_lo=g["lo"], gain_hi=g["hi"],
                              verdict=verdict(r[f"{best}|mean"], g), pos=int(y[m].sum()), logs_pos=len(set(logs[m][y[m]])))
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
            S0s = {"S0 (mean of seeds)": ens, "S0 blind seed 0": s0["S0-blind-s0"]}
            for k, col in (("a", 0), ("b", 1), ("A", 0)):
                y, m = label_mask(D, k)
                r = read(y[m], {nm: own(o, col)[m] for nm, o in S0s.items()}, logs[m])
                pc += [dict(label=k, score=nm, auc=v["auc"], lo=v["lo"], hi=v["hi"], n=int(m.sum()), pos=int(y[m].sum())) for nm, v in r.items()]
            # lateral-by-class split of the S0 agent read: positives by contact type against the same no-contact negatives
            ya, ma = label_mask(D, "a")
            Al = D.A.to_numpy()
            veh = D.a_cls.to_numpy() == 0
            neg = ma & ~ya
            for nm, pos in (("A-type: vehicle ahead in own lane (A positive)", ya & (Al == 1)), ("other vehicle contact (lateral / go-around / drift / cut-in / crossing)", ya & (Al != 1) & veh),
                            ("other object (pedestrian, cyclist, generic)", ya & (Al != 1) & ~veh), ("all agent contacts (S0 G1)", ya)):
                mm = ma & (pos | neg)
                for snm, o in S0s.items():
                    rr = read(pos[mm], {snm: own(o, 0)[mm]}, logs[mm], 2000)[snm]
                    pc.append(dict(label="agent split", subset=nm, score=snm, auc=rr["auc"], lo=rr["lo"], hi=rr["hi"], n=int(mm.sum()), pos=int(pos[mm].sum()),
                                   logs_pos=len(set(logs[mm][pos[mm]]))))
                for rung in ("vis", "nat"):
                    if rung in lad:
                        rr = read(pos[mm], {rung: score(rung, "a")[mm]}, logs[mm], 2000)[rung]
                        pc.append(dict(label="agent split", subset=nm, score=f"ladder {rung} (lat head)", auc=rr["auc"], lo=rr["lo"], hi=rr["hi"], n=int(mm.sum()),
                                       pos=int(pos[mm].sum()), logs_pos=len(set(logs[mm][pos[mm]]))))
        pd.DataFrame(pc).to_csv(out / "positive_control.csv", index=False)
        # ---- strata (descriptive): speed band, sub-class, state family, "stop / turn ahead" for every rung's two-seed score
        st = []
        strata = [("speed band", LC.SPEED_NAMES[b], D.band.to_numpy() == b) for b in range(4)] + [("states", f, (D.fam == f).to_numpy()) for f in FAMS]
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
        # ---- the S0 lateral head as a score for the longitudinal labels (descriptive)
        if s0:
            rows = []
            for k in LC.LONG:
                y, m = label_mask(D, k)
                r = read(y[m], {"S0 agent logit": own(ens, 0)[m], "S0 boundary logit": own(ens, 1)[m]}, logs[m], 2000)
                rows += [dict(label=k, score=nm, auc=v["auc"], lo=v["lo"], hi=v["hi"]) for nm, v in r.items()]
            pd.DataFrame(rows).to_csv(out / "s0_on_long.csv", index=False)
        # ---- arc family (the own plan re-timed x c, c ~ U(0.5, 1.5); descriptive, more positives)
        rows = []
        qs = list(range(16, 20))
        for k in LC.LONG:
            lab = X[f"{k.lower()}_lab"][:, qs]
            y, m = (lab == 1).ravel(), ((lab >= 0) & R["qok"][:, qs]).ravel()
            lg = np.repeat(R["log"], len(qs))
            sc = {rung: np.mean([pr[(rung, s)][k][:, qs, H.IDX[k]] for s in (0, 1)], 0).ravel() for rung in lad}
            r = read(y[m], {kk: v[m] for kk, v in sc.items()}, lg[m], 2000)
            rows += [dict(label=k, rung=rung, auc=v["auc"], lo=v["lo"], hi=v["hi"], n=int(m.sum()), pos=int(y[m].sum()), logs_pos=len(set(lg[m][y[m]]))) for rung, v in r.items()]
        pd.DataFrame(rows).to_csv(out / "arc_family.csv", index=False)
        run.summary.update(verdict=ver)
        with pd.option_context("display.width", 250, "display.max_rows", 400):
            print(L.to_string(index=False))
            print(pd.DataFrame(gains).to_string(index=False))
            print(json.dumps(ver, indent=1))
            print(pd.DataFrame(pc).to_string(index=False))


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
    p = sp.add_parser("pilot")
    p.add_argument("--runs", nargs="+", required=True, help="NAME=RUN_DIR (the vis and priv pilots)")
    a = ap.parse_args()
    {"predict": cmd_predict, "report": cmd_report, "pilot": cmd_pilot}[a.cmd](a)
