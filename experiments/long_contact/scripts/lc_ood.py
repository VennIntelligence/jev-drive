#!/usr/bin/env python3
"""LONG-PROBE out-of-distribution rows (prereg section 10, descriptive): decision 220's nuPlan decisions (the BODY1 G2 dumps of the served
tokens, ego vector and plan; experiments/body1/scripts/g2_dump.py). Nothing is fitted or chosen here.

  g2         (GPU) the serving student's native outputs re-computed on the dumped tokens (identity: this forward reproduces the dumped plan
             slice), the ladder's A logit of the served plan (rungs blind / vis / nat, seeds 0 / 1, A head) and S0's agent logit
             -> read/ood/g2_scores.npz
  g2-report  (CPU) AUC of each score on the class (i) rollouts ("straight into an in-path object", swerve/classes.csv) against g2's clean
             control decisions (swv1_bc.read_auc unchanged: speed-matched weights, log clusters, 1000 resamples), and on all collision
             rollouts; per class (i) rollout the time before the event of the first decision flagged at the P1 2 % threshold
             (read/report/thresholds.json) -> read/ood/g2_auc.csv, g2_lead.csv
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / "lib"), str(_pl.Path(__file__).resolve().parent)]
import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

import lc as LC  # noqa: E402
import lc_head as H  # noqa: E402
import b1 as B  # noqa: E402
import contact_head as C  # noqa: E402

SETS = ("P2H10-F-s0", "P2H10-F-s1", "ctrl")
STUDENTS = ("P2H10-F-s0", "P2H10-F-s1")
RUNGS = ("blind", "vis", "nat")
S0 = {"S0-step-s0": "full-step/20261010-024906", "S0-step-s1": "full-step/20261010-024909"}
CLASSES = B.REPO / "experiments/alpasim/results/swerve/classes.csv"


def out_dir():
    return LC.root() / "read" / "ood"


def lht_of_logs():
    t = np.load(B.cache_root() / "lb_navtest" / "tab.npz")
    return {str(l): bool(h) for l, h in zip(t["log"], t["lht"])}


def student_forward(model, tok, ego, tc, nv, dev, batch=64):
    """PModel forward on dumped tokens -> (outputs (N, n_out) float32, native (N, 206), hidden (N, 1024))."""
    import torch
    from jevdrive import op_adapt as A
    import lc_native as NA
    stash, orig = {}, model.net.run_batched

    def tap(feeds, want, **kw):
        o = orig(feeds, A.POLICY_OUT, **kw)
        stash.update(o)
        return {k: o[k] for k in want}
    model.net.run_batched = tap
    sl = model.net.slices
    cols = np.concatenate([np.arange(sl[k].start, sl[k].stop) for k in NA.SLICES])
    O, N, Hh = [], [], []
    try:
        with torch.no_grad():
            for i in range(0, len(tok), batch):
                j = slice(i, i + batch)
                o = model(torch.as_tensor(tok[j]).to(dev), torch.as_tensor(ego[j]).to(dev), torch.as_tensor(tc[j]).to(dev), None, None,
                          nv=torch.as_tensor(nv[j]).to(dev)).float()
                O.append(o.cpu().numpy())
                N.append(o[:, cols].cpu().numpy())
                Hh.append(torch.cat([stash["select_4"].reshape(len(o), -1), stash["mean"].reshape(len(o), -1)], 1).float().cpu().numpy())
    finally:
        model.net.run_batched = orig
    return np.concatenate(O), np.concatenate(N), np.concatenate(Hh)


def cmd_g2(a):
    import torch
    import pp_train as T
    from jevdrive.run import Run
    dev = torch.device("cuda")
    with Run("long_contact", "read/ood-g2", config=vars(a)) as run:
        lht = lht_of_logs()
        models = {t: T.load_pmodel(t, dev) for t in STUDENTS}
        res = {}
        for g in SETS:
            z = np.load(B.root() / "g2" / "dump" / f"{g}.npz")
            tok, valid, ego, poses, raw = z["tokens"], z["valid"], z["ego"].astype(np.float32), z["poses"].astype(np.float32), z["plan_raw"]
            assert (valid == (np.arange(8)[None] >= 8 - valid.sum(1)[:, None])).all(), "valid slots are not the newest ones"
            logs = np.array([s.rsplit("-", 1)[0] for s in z["scene"]])
            tc = np.array([[0.0, 1.0] if lht.get(l, False) else [1.0, 0.0] for l in logs], np.float32)
            nv = valid.sum(1).astype(np.int64)
            best = None
            for t, model in models.items():
                o, nat, hid = student_forward(model, tok, ego, tc, nv, dev)
                d = float(np.abs(o[:, model.net.slices["plan"]] - raw).max())
                run.info("%s: %s forward against the dumped plan slice: max |diff| %.4g", g, t, d)
                if best is None or d < best[0]:
                    best = (d, t, nat, hid)
            d, srv, nat, hid = best
            assert d < 0.05, f"{g}: no student reproduces the dumped plan (best {srv}: {d})"
            arr = dict(scene=z["scene"], k=z["k"], served=srv, ident=d)
            V, vm = torch.as_tensor(tok).to(dev), torch.as_tensor(valid).to(dev)
            Qp = torch.as_tensor(poses[:, None]).to(dev)
            for rung in RUNGS:
                for s in (0, 1):
                    rdir = _pl.Path((LC.root() / "train" / "runs" / f"full-{rung}-s{s}.dir").read_text().strip())
                    net, ck = H.load_ckpt(rdir / "ckpt.pt", dev, "A")
                    D = dict(V=V, Qp=Qp, E=torch.as_tensor((ego - ck["emu"]) / ck["esd"]).to(dev))
                    if net.nat:
                        n2 = np.repeat(((nat - ck["nmu"]) / ck["nsd"])[:, None], 2, 1)
                        D["nat"], D["hid"] = torch.as_tensor(n2).to(dev, torch.float16), torch.as_tensor(np.repeat(hid[:, None], 2, 1)).to(dev, torch.float16)
                    arr[f"{rung}-s{s}"] = H.predict(net, D, valid=vm)[:, 0, H.IDX["A"]].astype(np.float32)
            for nm, rel in S0.items():
                net, ck = C.load_ckpt(C.sroot() / rel / "ckpt.pt", dev)
                o, _ = C.predict(net, V, torch.as_tensor((ego - ck["emu"]) / ck["esd"]).to(dev), Qp, valid=vm)
                arr[nm] = o[:, 0, 0].astype(np.float32)
            res[g] = arr
            run.info("%s: %d decisions, served by %s (identity %.4g)", g, len(z["k"]), srv, d)
        out_dir().mkdir(parents=True, exist_ok=True)
        np.savez(out_dir() / "g2_scores.npz", **{f"{g}|{k}": v for g, arr in res.items() for k, v in arr.items()})
        run.summary.update(served={g: str(r["served"]) for g, r in res.items()}, ident={g: float(r["ident"]) for g, r in res.items()})


def cmd_g2_report(a):
    import pandas as pd
    import g2 as G
    import swv1_bc as BC
    from jevdrive.run import Run
    with Run("long_contact", "read/ood-g2-report", config=vars(a)):
        z = np.load(out_dir() / "g2_scores.npz", allow_pickle=True)
        thr = json.loads((LC.root() / "read" / "report" / "thresholds.json").read_text())
        names = [f"{r}-s{s}" for r in RUNGS for s in (0, 1)] + list(S0)
        sc = {}
        for g in SETS:
            for i, (scn, k) in enumerate(zip(z[f"{g}|scene"], z[f"{g}|k"])):
                sc[(g, str(scn), int(k))] = {nm: float(z[f"{g}|{nm}"][i]) for nm in names}
        for (g, scn, k), v in sc.items():
            for r in RUNGS:
                v[f"{r} (mean of seeds)"] = 0.5 * (v[f"{r}-s0"] + v[f"{r}-s1"])
            v["S0 (mean of seeds)"] = 0.5 * (v["S0-step-s0"] + v["S0-step-s1"])
        cols = [f"{r} (mean of seeds)" for r in RUNGS] + names + ["S0 (mean of seeds)"]
        cl = pd.read_csv(CLASSES)
        cls_i = {(r.set, r.scene) for r in cl.itertuples() if r.dom == "nuplan" and r.cls == "i" and r.set in G.POS_SETS}
        table = G.load_table()
        pos, neg = G.split(table)
        rows = []
        for subset, P in (("class (i) straight into an in-path object", [d for d in pos if (d["set"], d["scene"]) in cls_i]), ("all collision rollouts", pos)):
            for nm in cols:
                pp = [dict(d, s=sc[(d["set"], d["scene"], int(d["k"]))][nm]) for d in P if (d["set"], d["scene"], int(d["k"])) in sc]
                nn = [dict(d, s=sc[(d["set"], d["scene"], int(d["k"]))][nm]) for d in neg if (d["set"], d["scene"], int(d["k"])) in sc]
                r = BC.read_auc(pp, nn, "s", 1000)
                rows.append(dict(subset=subset, score=nm, n_pos=len(pp), n_neg=len(nn), rollouts=len({(d["set"], d["scene"]) for d in pp}),
                                 **({k: r[k] for k in ("auc", "lo", "hi")} if r else {})))
        pd.DataFrame(rows).to_csv(out_dir() / "g2_auc.csv", index=False)
        # lead time of the first flagged decision per class (i) rollout (threshold: P1 2 % false positives of the A negatives, two-seed score)
        lead = []
        tdf = pd.DataFrame(table)
        for (g, scn) in sorted(cls_i):
            d = tdf[(tdf.set == g) & (tdf.scene == scn) & (tdf.t_to_ev <= 8.0)].sort_values("now")
            for r in RUNGS:
                nm = f"{r} (mean of seeds)"
                fl = [row.t_to_ev for row in d.itertuples() if (g, scn, int(row.k)) in sc and sc[(g, scn, int(row.k))][nm] >= thr["A"][r]]
                lead.append(dict(set=g, scene=scn, rung=r, threshold=thr["A"][r], decisions=len(d), flagged=len(fl), first_flag_s_before_event=max(fl) if fl else np.nan))
        pd.DataFrame(lead).to_csv(out_dir() / "g2_lead.csv", index=False)
        with pd.option_context("display.width", 200):
            print(pd.DataFrame(rows).to_string(index=False))
            print(pd.DataFrame(lead).to_string(index=False))


if __name__ == "__main__":
    from jevdrive.run import cli_args
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("g2", "g2-report"):
        cli_args(sp.add_parser(nm))
    a = ap.parse_args()
    {"g2": cmd_g2, "g2-report": cmd_g2_report}[a.cmd](a)
