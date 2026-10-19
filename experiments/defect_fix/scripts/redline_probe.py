#!/usr/bin/env python3
"""REDLINE-PROBE (plans/2026-10-19-redline-probe-prereg.md): do the policy's frozen tokens (sh30 2 Hz CAM_F0 serve path, synthesised slots
included) carry "red lamp at the line / must stand" on the arena held static windows where the human stands at a red light's line with no
lead? Against KIN, KIN + lead head, and the plan's own decision-0 arc. Results: results/redline_probe.md.

  .venv/bin/python redline_probe.py select     (box) windows + classes (native_gap_report.common / cue, census light labels) -> $O/windows.csv;
                                                the dump lists of windows not yet dumped by native_gap (JTN1) -> $O/lists/<name>.txt
  $TR redline_probe.py feats                    (box, GPU) per window, decisions k = 0 and 3 of JTN-L18-s1's native dump: Cinque tokens of the
                                                8 slots (8, 32, 512) fp16, the fed ego row, and at k = 0 RSG-L18-s0's and JTN-L18-s1's forward
                                                (lead means + lead_prob, plan 4 s arc; gate against the stored native reads' k = 0 plan)
  $TR redline_probe.py probe                    (box, GPU) log-fold cross-fitted probes, AUCs with log-cluster bootstrap -> $O/probe/probe.json
  python redline_probe.py report [--src DIR]    probe.json -> the markdown tables of results/redline_probe.md (stdout)
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "16")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(HERE), str(REPO / "experiments/route_follow/scripts")]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
NG = DATA / "runs/defect_fix/native_gap"
O = DATA / "runs/defect_fix/redline_probe"
AR = DATA / "runs/alpasim_proxy/arena"
STORED = {"RSG": AR / "reads/RSG-L18-s0", "JTN1": DATA / "runs/defect_fix/jt/arena/JTN-L18-s1@off"}
CKPT = {"RSG": "RSG-L18-s0", "JTN1": "JTN-L18-s1"}
KS = (0, 3)
FOLDS, B_BOOT, FPR, PER_LIST = 5, 2000, 0.10, 60
NEG = ("N1 close lead", "N2 no light no lead", "N3 moving to green/unknown light")
POS = "P red line no lead"
EXTRA = ("G green launch no lead", "U unknown light static", "F far lead static")
SPLIT = "alpasim_nuplan/onpol-held"


# ---------------------------------------------------------------- select
def classes():
    import pandas as pd
    import native_gap_report as R
    W = pd.read_parquet(NG / "win.parquet")
    C = pd.read_parquet(NG / "set_census.parquet").drop_duplicates("id").set_index("id")
    S = pd.read_csv(AR / "set/set.csv.gz").set_index("id")
    ids = R.common(W, C, S)
    S, C = S.loc[ids], C.reindex(ids)
    st = S.stratum == "static"
    cu = R.cue(S, C)
    red = C.red_stop.fillna(False).astype(bool)
    lead = S.lead.fillna(False).astype(bool)
    tl = C.tl.fillna(False).astype(bool)
    n3 = ~st & tl & ~red & (C.red_frac.fillna(1) == 0) & (C.tl_d_front > 0) & (C.tl_d_front <= 60) & (S.v0 >= 2)
    g = red & (C.red_cls.fillna("") == "G") & ~lead
    cls = pd.Series(np.select([st & (cu == "red light at the line, no lead"), st & (cu == "lead <= 8 m"), st & (cu == "no light, no lead"), n3, g,
                               st & (cu == "light of unknown state, no lead"), st & (cu == "lead 8-30 m")], [POS, *NEG, *EXTRA], ""), index=ids)
    T = pd.DataFrame(dict(cls=cls, src=S.src, log=S.log, shard=S.shard, v0=S.v0, red_dfront=C.red_dfront, tl_d_front=C.tl_d_front, st0=C.st0))
    return T[T.cls != ""]


def dumped(root, pattern):
    """scene -> (drive.jsonl, session n) of every SH30_DUMP read under root."""
    out = {}
    for f in sorted(glob.glob(str(root / pattern))):
        for line in open(f):
            if line.startswith('{"kind": "start"'):
                r = json.loads(line)
                out.setdefault(r["scene"], (f, r["n"]))
    return out


def cmd_select(a):
    T = classes()
    have = dumped(NG / "dump/JTN1", "c*/run/driver-logs/drive.jsonl")
    T["dump"] = np.where(T.index.isin(list(have)), "native_gap", "new")
    O.mkdir(parents=True, exist_ok=True)
    T.to_csv(O / "windows.csv")
    (O / "lists").mkdir(exist_ok=True)
    for k, g in T[T.dump == "new"].groupby("shard"):
        for j, c in enumerate(np.array_split(np.array(sorted(g.index)), int(np.ceil(len(g) / PER_LIST)))):
            (O / "lists" / f"s{k}_{j}.txt").write_text("\n".join(c) + "\n")
    print(T.groupby("cls").agg(windows=("log", "size"), logs=("log", "nunique"), reuse=("dump", lambda x: int((x == "native_gap").sum()))).to_string())
    print("lists:", " ".join(sorted(p.stem for p in (O / "lists").glob("*.txt"))))


# ---------------------------------------------------------------- feats (GPU)
_REC = {}


def records(f, n, ks):
    """the drive records (k in ks) of session n of one drive.jsonl -> {k: rec} (each file parsed once)."""
    if f not in _REC:
        ses, idx = {}, {}
        for line in open(f):
            r = json.loads(line)
            if r["kind"] == "start":
                ses[r["session"]] = r["n"]
            elif r["kind"] == "drive" and int(r["k"]) in ks:
                idx.setdefault(ses[r["session"]], {})[int(r["k"])] = r
        _REC[f] = idx
    return _REC[f].get(n, {})


def stored_k0(tag, ids):
    """k = 0 drive records of the stored native read of tag -> {scene: rec}."""
    out = {}
    for f in sorted(glob.glob(str(STORED[tag] / "c*/run/driver-logs/drive.jsonl"))):
        for line in open(f):
            if '"kind": "drive"' in line and '"k": 0,' in line:
                r = json.loads(line)
                if r["scene"] in ids and int(r["k"]) == 0:
                    out.setdefault(r["scene"], r)
    return out


def arc(P):
    P = np.asarray(P, float).reshape(-1, 3)[:, :2]
    return float(np.hypot(*np.diff(np.r_[[[0.0, 0.0]], P], axis=0).T).sum())


def cmd_feats(a):
    import pandas as pd
    import torch
    sys.path[:0] = [str(REPO / "experiments/alpasim/lib"), str(REPO / "lib")]
    import sh30_core as SC
    from jevdrive import navsim_zs as Z
    from jevdrive import op_interp as I
    from jevdrive.run import Run
    T = pd.read_csv(O / "windows.csv").set_index("id")
    with Run("defect_fix", "redline-probe-feats", config=vars(a)) as run:
        src = {**dumped(NG / "dump/JTN1", "c*/run/driver-logs/drive.jsonl"), **dumped(O / "dump", "*/run/driver-logs/drive.jsonl")}
        ids = [i for i in T.index if i in src]
        run.info(f"windows {len(T)}, with a dump {len(ids)}; missing {len(T) - len(ids)}")
        sto = {t: stored_k0(t, set(ids)) for t in STORED}
        cores = {t: SC.Core(CKPT[t], "cuda", "backwarp") for t in CKPT}
        tc = torch.tensor([[1.0, 0.0]], device="cuda")
        n = len(ids)
        F = {f"tok{k}": np.zeros((n, 8, 32, 512), np.float16) for k in KS} | {f"ego{k}": np.zeros((n, 20), np.float32) for k in KS}
        F |= {f"has{k}": np.zeros(n, bool) for k in KS} | {"lead3_JTN1": np.zeros((n, 75), np.float32)}
        for t in CKPT:
            F |= {f"lead0_{t}": np.zeros((n, 75), np.float32), f"arc0_{t}": np.zeros(n), f"gate0_{t}": np.full(n, np.nan),
                  f"rmem_eq_{t}": np.zeros(n, bool)}
        F["gate_dump"] = np.full(n, np.nan)

        def forward(core, H, row, cam_t):
            o = core.model(H, torch.from_numpy(row[None].astype(np.float32)).cuda(), tc)
            o = (o[0] if isinstance(o, tuple) else o).float()
            sl = core.model.net.slices
            ld, lp = o[0, sl["lead"]].cpu().numpy(), o[0, sl["lead_prob"]].cpu().numpy()
            mu = o[0, core.pi].reshape(33, 15).cpu().numpy()
            P = I.to_rear(mu[:, 0:3], mu[:, 11], I.T_IDXS, cam_t[:2].astype(np.float64), Z.T_OUT, "lever")[:, :3]
            return np.r_[ld[:len(ld) // 2], lp], P

        cj = cores["JTN1"]
        for i, sc in enumerate(ids):
            f, ses_n = src[sc]
            recs = records(f, ses_n, KS)
            for k in KS:
                p = Path(f).parent / "dump" / f"s{ses_n:02d}_k{k}.npz"
                if k not in recs or not p.exists():
                    continue
                d = np.load(p)
                cur, valid = d["cur"], d["valid"]
                assert valid.all(), (sc, k, valid)
                prev = np.concatenate([np.zeros((1,) + cur.shape[1:], np.uint8), cur[:-1]])
                with torch.no_grad():
                    pp, cu = (torch.from_numpy(x).cuda() for x in (prev, cur))
                    H = cj.model.net.run_batched(cj.A.vision_feeds(pp, cu), ["view_39"])["view_39"].reshape(1, 8, *cj.A.H_SHAPE)
                    F[f"tok{k}"][i] = H[0].half().cpu().numpy()
                    ego = d["ego"].astype(np.float32)
                    F[f"ego{k}"][i], F[f"has{k}"][i] = ego, True
                    rm_dump = np.asarray(recs[k].get("rmem") or [], np.float32)
                    if k == 3:
                        F["lead3_JTN1"][i] = forward(cj, H, np.r_[ego, rm_dump] if cj.rm else ego, d["cam_t"])[0]
                        continue
                    for t, core in cores.items():
                        s0 = sto[t].get(sc)
                        rm = np.asarray(s0["rmem"], np.float32) if s0 is not None and s0.get("rmem") is not None else rm_dump
                        F[f"rmem_eq_{t}"][i] = bool(len(rm) == len(rm_dump) and np.allclose(rm, rm_dump, atol=1e-4))
                        ld, P = forward(core, H, np.r_[ego, rm] if core.rm else ego, d["cam_t"])
                        F[f"lead0_{t}"][i], F[f"arc0_{t}"][i] = ld, arc(P)
                        if s0 is not None:
                            F[f"gate0_{t}"][i] = float(np.abs(P[:, :2] - np.asarray(s0["poses"], float)[:, :2]).max())
                        if t == "JTN1":
                            F["gate_dump"][i] = float(np.abs(P[:, :2] - d["poses"][:, :2]).max())
            if i % 100 == 0:
                run.info(f"{i} / {n}")
        np.savez(O / "feats.npz", ids=np.array(ids), **F)
        g = {k: (float(np.nanmax(v)), float(np.nanquantile(v, 0.99)), int(np.isfinite(v).sum())) for k, v in F.items() if k.startswith("gate")}
        run.summary.update(windows=n, has0=int(F["has0"].sum()), has3=int(F["has3"].sum()), gates=g,
                           rmem_eq={t: int(F[f"rmem_eq_{t}"].sum()) for t in CKPT})
        run.info(f"gates (max, p99, n): {g}; rmem equal to the dump's: " + ", ".join(f"{t} {int(F[f'rmem_eq_{t}'].sum())}" for t in CKPT))


# ---------------------------------------------------------------- probe
def fit_tok(Ttr, Ktr, ytr, Tte, Kte, seed):
    """rf_stopprobe.fit_tok's net (per-token LN -> 16 -> MLP 256), batch 64, 30 epochs (about 800 training rows here)."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    mu, sd = Ktr.mean(0), Ktr.std(0) + 1e-6
    net = nn.ModuleDict(dict(ln=nn.LayerNorm(512, elementwise_affine=False), red=nn.Linear(512, 16),
                             mlp=nn.Sequential(nn.Linear(32 * 16 + Ktr.shape[1], 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1)))).cuda()
    f = lambda t, k: net["mlp"](torch.cat([net["red"](net["ln"](t.float())).flatten(1), k], 1))[:, 0]  # noqa: E731
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    pw = torch.tensor(float((1 - ytr).sum() / max(ytr.sum(), 1)), device="cuda")
    Kt, Ke = (torch.from_numpy(((x - mu) / sd).astype(np.float32)).cuda() for x in (Ktr, Kte))
    Tt, Te = torch.from_numpy(np.ascontiguousarray(Ttr)).cuda(), torch.from_numpy(np.ascontiguousarray(Tte)).cuda()
    yt = torch.from_numpy(ytr.astype(np.float32)).cuda()
    g = np.random.default_rng(seed)
    for _ in range(30):
        net.train()
        for b in np.array_split(g.permutation(len(ytr)), max(1, len(ytr) // 64)):
            b = torch.from_numpy(b).cuda()
            loss = nn.functional.binary_cross_entropy_with_logits(f(Tt[b], Kt[b]), yt[b], pos_weight=pw)
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return f(Tt, Kt).cpu().numpy(), f(Te, Ke).cpu().numpy()


def pair_auc(s, y, grp, seed=0):
    """AUC over (positive, negative) pairs from the same log only, with a log-cluster bootstrap -> (auc, lo, hi), draws."""
    u = np.unique(grp)
    c, m = np.zeros(len(u)), np.zeros(len(u))
    for j, l in enumerate(u):
        q = grp == l
        a, b = s[q & y], s[q & ~y]
        if len(a) and len(b):
            d = a[:, None] - b[None]
            c[j], m[j] = (d > 0).sum() + 0.5 * (d == 0).sum(), d.size
    rng = np.random.default_rng(seed)
    C = np.stack([np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u)) for _ in range(B_BOOT)]).astype(float)
    dr = (C @ c) / np.maximum(C @ m, 1)
    return (float(c.sum() / max(m.sum(), 1)), float(np.quantile(dr, 0.025)), float(np.quantile(dr, 0.975))), dr, int((m > 0).sum()), int(m.sum())


def cmd_probe(a):
    import pandas as pd
    import rf_stopprobe as RS
    from jevdrive.data import splits
    from jevdrive.run import Run
    out = O / "probe"
    out.mkdir(parents=True, exist_ok=True)
    with Run("defect_fix", "redline-probe", config=vars(a)) as run:
        sp = splits.load(SPLIT)
        run.use_split(sp)
        z = np.load(O / "feats.npz")
        T = pd.read_csv(O / "windows.csv").set_index("id").loc[z["ids"]]
        keep = z["has0"]
        assert set(T.src[keep]) <= set(sp.members), "a window outside the registered held split"
        T = T[keep]
        Z_ = {k: z[k][keep] for k in z.files if k != "ids"}
        cls, log = T.cls.values, T.log.values
        train = np.isin(cls, [POS, *NEG])
        y = cls == POS
        fold = np.array([int(hashlib.sha1(x.encode()).hexdigest(), 16) % FOLDS for x in log])
        n = len(cls)
        kin = Z_["ego0"]
        lin = {"KIN": kin, "LEAD-RSG": np.c_[kin, Z_["lead0_RSG"]], "LEAD-JTN1": np.c_[kin, Z_["lead0_JTN1"]],
               "TOK-lin": np.c_[kin, Z_["tok0"][:, 7].astype(np.float32).mean(1)]}
        has3 = Z_["has3"]
        lin3 = {"KIN@k3": Z_["ego3"], "LEAD-JTN1@k3": np.c_[Z_["ego3"], Z_["lead3_JTN1"]]}
        toks = {"TOK": (Z_["tok0"], 7, kin)} | {f"TOK-s{j}": (Z_["tok0"], j, kin) for j in range(7)} | \
               {f"TOK-s{j}@k3": (Z_["tok3"], j, Z_["ego3"]) for j in range(8)}
        S, TAU = {}, {}
        S["ARC0-RSG"], S["ARC0-JTN1"] = -Z_["arc0_RSG"], -Z_["arc0_JTN1"]
        for k_ in ("ARC0-RSG", "ARC0-JTN1"):
            TAU[k_] = [float(np.quantile(S[k_][train & ~y & (fold != k)], 1 - FPR)) for k in range(FOLDS)]
        jobs = [(nm, "lin", X, None) for nm, X in lin.items()] + [(nm, "lin", X, has3) for nm, X in lin3.items()] + \
               [(nm, "tok", v, has3 if nm.endswith("@k3") else None) for nm, v in toks.items()]
        for nm, kind, X, need in jobs:
            sc, tau = np.full(n, np.nan), []
            ok = np.ones(n, bool) if need is None else need
            for k in range(FOLDS):
                tr, te = train & ok & (fold != k), ok & (fold == k)
                if kind == "lin":
                    s_tr, s_te = RS.fit_lin(X[tr], y[tr], X[te])
                else:
                    Tk, j, K = X
                    s_tr = s_te = 0
                    for seed in range(3):
                        a_, b_ = fit_tok(Tk[np.flatnonzero(tr), j], K[tr], y[tr].astype(int), Tk[np.flatnonzero(te), j], K[te], seed)
                        s_tr, s_te = s_tr + a_ / 3, s_te + b_ / 3
                sc[te] = s_te
                tau.append(float(np.quantile(s_tr[~y[tr]], 1 - FPR)))
            S[nm], TAU[nm] = sc, tau
            run.info(f"{nm} done")
        np.savez(out / "scores.npz", **S, cls=cls, log=log, fold=fold, ids=T.index.values, red_dfront=T.red_dfront.values, tl_d_front=T.tl_d_front.values)

        def table(mask, sets, pair=False):
            """AUC of the sets on rows `mask` (positives y), log-cluster bootstrap; TOK - best LEAD / KIN paired."""
            m = mask & np.all([np.isfinite(S[s]) for s in sets], 0)
            if (y & m).sum() < 5 or (~y & m).sum() < 5:
                return dict(n_pos=int((y & m).sum()), n_neg=int((~y & m).sum()))
            res, dr = RS.boot_auc({s: S[s][m] for s in sets}, y[m], log[m])
            r = dict(n_pos=int((y & m).sum()), n_neg=int((~y & m).sum()), logs_pos=int(len(set(log[m & y]))), logs_neg=int(len(set(log[m & ~y]))), auc=res)
            if "TOK" in sets:
                lb = max(("LEAD-RSG", "LEAD-JTN1"), key=lambda s: res[s][0]) if "LEAD-RSG" in sets else None
                for b in [x for x in ("KIN", lb) if x]:
                    d = dr["TOK"] - dr[b]
                    r[f"TOK-{'LEAD' if b == lb else b}"] = (res["TOK"][0] - res[b][0], float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975)))
                r["lead_best"] = lb
            if pair:
                pr = {s: pair_auc(S[s][m], y[m], log[m]) for s in sets}
                r["pair"] = {s: v[0] for s, v in pr.items()}
                r["pair_logs"], r["pair_n"] = pr[sets[0]][2], pr[sets[0]][3]
                if "TOK" in sets:
                    lb = r["lead_best"]
                    d = pr["TOK"][1] - pr[lb][1]
                    r["pair_TOK-LEAD"] = (pr["TOK"][0][0] - pr[lb][0][0], float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975)))
            return r

        base = ["KIN", "LEAD-RSG", "LEAD-JTN1", "ARC0-RSG", "ARC0-JTN1", "TOK-lin", "TOK"]
        J = dict(rows=n, classes={c: dict(windows=int((cls == c).sum()), logs=int(len(set(log[cls == c])))) for c in [POS, *NEG, *EXTRA]}, tables={})
        isP = cls == POS
        J["tables"]["primary"] = table(train, base, pair=True)
        for c in NEG:
            J["tables"][f"P vs {c}"] = table(isP | (cls == c), base, pair=True)
        J["tables"]["P vs N1+N2 (static only)"] = table(isP | np.isin(cls, NEG[:2]), base, pair=True)
        slots = [f"TOK-s{j}" for j in range(7)] + ["TOK"]
        J["tables"]["slots k0"] = table(train, ["KIN"] + slots)
        J["tables"]["slots k3"] = table(train & has3, ["KIN@k3", "LEAD-JTN1@k3"] + [f"TOK-s{j}@k3" for j in range(8)])
        rd, td = T.red_dfront.values, T.tl_d_front.values
        for lo, hi in ((-99, -1.5), (-1.5, -1.0), (-1.0, -0.5), (-0.5, 99)):
            J["tables"][f"P red_dfront [{lo}, {hi})"] = table((isP & (rd >= lo) & (rd < hi)) | (train & ~y), base)
        for lo, hi in ((0, 20), (20, 40), (40, 60.01)):
            J["tables"][f"N3 light at [{lo}, {hi}) m"] = table(isP | ((cls == NEG[2]) & (td >= lo) & (td < hi)), base)
        # extras: the share called "must stand" at each fold's 10 % FPR threshold; G against P (the lamp colour is constant in the render)
        ex = {}
        for c in [POS, *NEG, *EXTRA]:
            q = cls == c
            ex[c] = {s: float(np.mean([S[s][i] >= TAU[s][fold[i]] for i in np.flatnonzero(q)])) for s in base}
        J["fire_at_10fpr"] = ex
        for c in EXTRA:
            m = isP | (cls == c)
            res, _ = RS.boot_auc({s: S[s][m] for s in base}, isP[m], log[m])
            J["tables"][f"P vs {c} (P = positive)"] = dict(n_pos=int(isP.sum()), n_neg=int((cls == c).sum()), auc=res)
        J["gates"] = {k: (float(np.nanmax(Z_[k])), float(np.nanmedian(Z_[k]))) for k in Z_ if k.startswith("gate")}
        J["tau"] = TAU
        json.dump(J, open(out / "probe.json", "w"), indent=1)
        p = J["tables"]["primary"]
        run.summary.update(rows=n, tok=p["auc"]["TOK"], tok_lead=p.get("TOK-LEAD"), lead_best=p.get("lead_best"))


# ---------------------------------------------------------------- report
def cmd_report(a):
    J = json.load(open(Path(a.src) / "probe.json"))
    f = lambda t: f"{t[0]:.3f} [{t[1]:.2f}, {t[2]:.2f}]"  # noqa: E731
    g = lambda t: f"{t[0]:+.3f} [{t[1]:+.2f}, {t[2]:+.2f}]"  # noqa: E731
    L = [f"rows {J['rows']}; gates (max, median m): {J['gates']}", "", "| class | windows | logs |", "|---|--:|--:|"]
    L += [f"| {c} | {v['windows']} | {v['logs']} |" for c, v in J["classes"].items()]
    for name, r in J["tables"].items():
        L += ["", f"### {name}", ""]
        if "auc" not in r:
            L.append(f"too few rows: {r}")
            continue
        L.append(f"pos {r['n_pos']} ({r.get('logs_pos', '-')} logs), neg {r['n_neg']} ({r.get('logs_neg', '-')} logs)")
        L += ["", "| set | AUC [95 % log CI] | same-log pairs |", "|---|--:|--:|"]
        L += [f"| {s} | {f(v)} | {f(r['pair'][s]) if 'pair' in r else '-'} |" for s, v in r["auc"].items()]
        for k in ("TOK-KIN", "TOK-LEAD", "pair_TOK-LEAD"):
            if k in r:
                L.append(f"| **{k}**{' (' + r['lead_best'] + ')' if 'LEAD' in k else ''} | {g(r[k])} | |")
        if "pair" in r:
            L.append(f"\nsame-log pairs: {r['pair_n']} from {r['pair_logs']} logs")
    L += ["", "### share called 'must stand' at the fold's 10 % FPR threshold", "", "| class | " + " | ".join(next(iter(J["fire_at_10fpr"].values()))) + " |",
          "|---|" + "--:|" * len(next(iter(J["fire_at_10fpr"].values())))]
    L += [f"| {c} | " + " | ".join(f"{x:.2f}" for x in v.values()) + " |" for c, v in J["fire_at_10fpr"].items()]
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("select", "feats", "probe", "report"))
    ap.add_argument("--src", default=str(O / "probe"))
    a = ap.parse_args()
    dict(select=cmd_select, feats=cmd_feats, probe=cmd_probe, report=cmd_report)[a.cmd](a)
