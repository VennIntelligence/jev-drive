"""Skill pack N1 (fc65452:todos/2026-09-29-n1-scorer.md, pre-registered before any N1 label or score): retrain E6's Hydra-style
scoring head on GIMM-interpolated openpilot Cinque `temporal` features, with the native plan in the candidate set.

Candidates per token: E6's K = 1024 anchors + 4 native slots (the native plan stretched along its path by 1.00 / 1.05 /
1.10 / 1.15, skill_pack_n0.stretch). Inputs: standardised [ego (32), GIMM `temporal` (512), native plan (24)]. Heads: five
linear sub-score heads (NC, DAC, EP, TTC, C) over the 1028 candidates, BCE on the devkit's per-candidate sub-scores
(E6's anchor labels + experiments/skill_pack/archive/n1_score_native.py's native labels), lambda per head on held-out logs. Imitation term:
minus the mean (x, y) distance of the candidate to the native plan (replaces E6's cls_late logits). Selection:
argmax_c  w_im (-d_c) + w_mul (log s_NC + log s_DAC) + w_ttc log s_TTC + w_ep log s_EP + w_c log s_C + beta [c native],
all weights and beta by argmax of the held-out-logs PDMS. CPU steps and one GPU (small) step.

    python -m experiments.skill_pack.archive.skill_pack_n1 cands --n N          stretched native candidates of the first N train tokens + navtest
    python -m experiments.skill_pack.archive.skill_pack_n1 fit --n N [--final]  held-out fit / grid / ablations; --final also writes the navtest poses
    python -m experiments.skill_pack.archive.skill_pack_n1 report               navtest N1 vs N0 / native / E6 Hydra (after the one scoring)
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from jevdrive import navsim_heads as H
from jevdrive.common import data_dir, get_logger

log = get_logger(__name__)
RUN = "runs/skill_pack/n1"
FEAT = "runs/skill_pack/n1/feat"
E6_PREP = "runs/elicitation/e6-prep/20260926-003758"
SUBS = ("NC", "DAC", "EP", "TTC", "C")
LAMS = (1e-5, 1e-4, 1e-3, 1e-2)
W_IM, W_MUL, W_TTC, W_EP, W_C, BETA = (0, .1, .25, .5, 1, 2, 4), (1, 2, 5, 10), (0, .5, 1, 2, 5), (0, .5, 1, 2, 5), (0, 1), (0, .25, .5, 1, 2)
SCALES = (1.00, 1.05, 1.10, 1.15)
NS, K = len(SCALES), H.K
SEED_HOLD = 1                                # the held-out log split of E6's fit


def run_dir() -> Path:
    d = data_dir() / RUN
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- candidates

def cmd_cands(n: int):
    from experiments.skill_pack.archive.skill_pack_n0 import stretch
    for name, f in (("train", f"lb_n1train_n{n}"), ("navtest", "lb_navtest_n12146")):
        if not (data_dir() / FEAT / f"{f}.npz").exists():
            print(f"{f}: not extracted yet, skipped")
            continue
        z = np.load(data_dir() / FEAT / f"{f}.npz")
        nat = z["native"]
        c = np.stack([[stretch(p, s) for s in SCALES] for p in nat]).astype(np.float32)
        out = run_dir() / f"cands_{name}_n{len(nat)}.npz"
        np.savez(out, tokens=z["tokens"], cands=c)
        print(f"{out.name}: {c.shape}")


# ---------------------------------------------------------------- data

def _std(train: np.ndarray, *others):
    mu, sd = train.mean(0), train.std(0)
    sd = np.where(sd > 1e-6, sd, 1)
    return [torch.as_tensor((x - mu) / sd, dtype=torch.float32, device=H.DEV) for x in (train, *others)]


def _dist(P: torch.Tensor, ref: torch.Tensor, step: int = 1000) -> torch.Tensor:
    """P (n, C, 8, 3), ref (n, 8, 3): mean over the 8 poses of the (x, y) distance -> (n, C)."""
    return torch.cat([(P[i:i + step, :, :, :2] - ref[i:i + step, None, :, :2]).norm(dim=-1).mean(-1)
                      for i in range(0, len(P), step)])


def load_train(n: int) -> dict:
    """Aligned training rows: features, E6's per-anchor labels, the native candidates and their labels."""
    f = np.load(data_dir() / FEAT / f"lb_n1train_n{n}.npz")
    toks = f["tokens"]
    c = np.load(run_dir() / f"cands_train_n{n}.npz")
    assert (c["tokens"] == toks).all()
    from jevdrive import elicit_e6 as E6
    tr = E6._navtrain(True)
    pos = dict(zip(tr["tokens"].tolist(), range(len(tr["tokens"]))))
    r = np.array([pos[t] for t in toks])
    at = np.load(data_dir() / E6_PREP / "anchors.npz")["anchors"]
    want = {t: i for i, t in enumerate(toks.tolist())}
    sub = np.zeros((len(toks), K, 5), np.float32)
    pd_ = np.zeros((len(toks), K), np.float32)
    seen = 0
    for p in sorted((data_dir() / E6_PREP / "score").glob("chunk_*.npz")):
        z = np.load(p)
        for j, t in enumerate(z["tokens"].tolist()):
            if t in want:
                sub[want[t]], pd_[want[t]] = z["sub"][j], z["pdms"][j]
                seen += 1
    assert seen == len(toks), f"E6 labels found for {seen} / {len(toks)} tokens"
    ns = {}
    for p in sorted((run_dir() / f"native_score_n{n}").glob("chunk_*.npz")):
        z = np.load(p)
        ns.update({t: (z["sub"][j], z["pdms"][j]) for j, t in enumerate(z["tokens"].tolist())})
    assert all(t in ns for t in toks.tolist()), "native labels missing for some tokens"
    sub_n = np.stack([ns[t][0] for t in toks.tolist()])
    pd_n = np.stack([ns[t][1] for t in toks.tolist()])
    return {"tokens": toks, "log": tr["log"][r], "ego": tr["ego"][r], "hold_feat": tr["cinque"][r], "gimm": f["temporal"],
            "native": f["native"], "cands": c["cands"], "anchors": at, "sub": np.concatenate([sub, sub_n], 1),
            "pdms": np.concatenate([pd_, pd_n], 1)}


# ---------------------------------------------------------------- heads

def bce_fit(X, T, rows, lam, iters=100):
    from jevdrive.elicit_e6 import _bce_fit
    return _bce_fit(X, T, rows, lam, iters)


def fit_heads(X: torch.Tensor, T: dict, fit_r, hold_r, tag: str, rec: dict) -> tuple[dict, dict]:
    """Per sub-score: lambda on the held-out BCE, hold logits (fit on fit_r) and the all-rows (W, b)."""
    hold, full, lam = {}, {}, {}
    allr = np.arange(len(X))
    for m in SUBS:
        sc = []
        for lm in LAMS:
            W, b = bce_fit(X, T[m], fit_r, lm)
            sc.append(float(torch.nn.functional.binary_cross_entropy_with_logits(X[hold_r] @ W + b, T[m][hold_r])))
        k = int(np.argmin(sc))
        lam[m] = LAMS[k]
        W, b = bce_fit(X, T[m], fit_r, LAMS[k])
        hold[m] = X[hold_r] @ W + b
        full[m] = bce_fit(X, T[m], allr, LAMS[k])
        rec[f"{tag}/{m}"] = {"lam": LAMS[k], "bce_hold": sc, "edge": k in (0, len(LAMS) - 1)}
        log.info(f"[{tag}] {m}: lam {LAMS[k]:g} hold BCE {np.round(sc, 5).tolist()}")
    return hold, full


def scores(logits: dict, Dm: torch.Tensor, w, beta, nat_mask: torch.Tensor) -> torch.Tensor:
    ls = torch.nn.functional.logsigmoid
    w_im, w_mul, w_ttc, w_ep, w_c = w
    return (-w_im * Dm + w_mul * (ls(logits["NC"]) + ls(logits["DAC"])) + w_ttc * ls(logits["TTC"]) + w_ep * ls(logits["EP"])
            + w_c * ls(logits["C"]) + beta * nat_mask)


def grid_search(logits: dict, Dm: torch.Tensor, P: torch.Tensor, nat_mask: torch.Tensor, cols=None):
    """argmax of the held-out PDMS over the weight grid (first maximum in grid order on ties)."""
    if cols is not None:
        logits = {m: v[:, cols] for m, v in logits.items()}
        Dm, P, nat_mask = Dm[:, cols], P[:, cols], nat_mask[cols]
    ar = torch.arange(len(P), device=P.device)
    best, rows = None, []
    for w_im, w_mul, w_ttc, w_ep, w_c, beta in itertools.product(W_IM, W_MUL, W_TTC, W_EP, W_C, BETA):
        if cols is not None and not nat_mask.any() and beta:
            continue
        w = (w_im, w_mul, w_ttc, w_ep, w_c)
        v = float(P[ar, scores(logits, Dm, w, beta, nat_mask).argmax(1)].mean())
        rows.append((*w, beta, v))
        if best is None or v > best[2]:
            best = (w, beta, v)
    return best, rows


# ---------------------------------------------------------------- fit

def cmd_fit(n: int, final: bool):
    from jevdrive.runlog import RunLog
    H.DEV = "cuda" if torch.cuda.is_available() else "cpu"
    rl = RunLog("skill_pack", "n1", f"fit_n{n}" + ("_final" if final else ""))
    out = run_dir()
    d = load_train(n)
    toks, logs = d["tokens"], d["log"]
    hold = H._group_folds(logs, 5, seed=SEED_HOLD) == 0
    fit_r, hold_r = np.flatnonzero(~hold), np.flatnonzero(hold)
    rl.log.info(f"n = {len(toks)}: {len(fit_r)} fit / {len(hold_r)} held-out tokens ({len(set(logs[hold_r]))} held-out logs)")
    nat = d["native"]
    A = torch.as_tensor(d["anchors"], device=H.DEV)
    C = torch.as_tensor(d["cands"], device=H.DEV)
    Nt = torch.as_tensor(nat, device=H.DEV)
    Dm = torch.cat([_dist(A[None].expand(len(toks), -1, -1, -1), Nt), _dist(C, Nt)], 1)          # (n, K + NS)
    T = {m: torch.as_tensor(d["sub"][:, :, i], device=H.DEV) for i, m in enumerate(SUBS)}
    P = torch.as_tensor(d["pdms"], device=H.DEV)
    nat_mask = torch.zeros(K + NS, device=H.DEV)
    nat_mask[K:] = 1
    feats = {"gimm": d["gimm"], "hold": d["hold_feat"]}
    X = {}
    for nm, F in feats.items():
        Xe, Xf, Xn = _std(d["ego"], d["ego"])[0], _std(F)[0], _std(nat.reshape(len(nat), -1))[0]
        X[nm] = torch.cat([Xe, Xf, Xn], 1)
    rec, tab = {}, []
    ho = torch.as_tensor(hold_r, device=H.DEV)
    Ph, Dh = P[ho], Dm[ho]
    ref = {"oracle (all 1028)": float(Ph.max(1).values.mean()), "oracle (anchors)": float(Ph[:, :K].max(1).values.mean()),
           **{f"native s={s:.2f}": float(Ph[:, K + i].mean()) for i, s in enumerate(SCALES)},
           "native-vs-anchor oracle": float(torch.maximum(Ph[:, K:].max(1).values, Ph[:, :K].max(1).values).mean()),
           "nearest anchor of native s=1.00 (N0's lookup for a native plan)": float(
               Ph[torch.arange(len(ho)), Dh[:, :K].argmin(1)].mean())}
    rl.log.info(f"held-out references: {json.dumps({k: round(v, 4) for k, v in ref.items()})}")
    arms = {"A0 GIMM feats + native slots (N1)": ("gimm", None), "A1 hold feats (E6 tap) + native slots": ("hold", None),
            "A2 GIMM feats, anchors only": ("gimm", np.arange(K))}
    heads_full, best_of = {}, {}
    for name, (fk, cols) in arms.items():
        if fk not in heads_full:
            hd, fl = fit_heads(X[fk], T, fit_r, hold_r, fk, rec)
            heads_full[fk] = (hd, fl)
        hd = heads_full[fk][0]
        b, rows = grid_search(hd, Dh, Ph, nat_mask, None if cols is None else torch.as_tensor(cols, device=H.DEV))
        best_of[name] = b
        tab.append({"arm": name, "w_im": b[0][0], "w_mul": b[0][1], "w_ttc": b[0][2], "w_ep": b[0][3], "w_c": b[0][4],
                    "beta": b[1], "pdms_hold": 100 * b[2]})
        rl.log.info(f"{name}: hold PDMS {100 * b[2]:.3f} at w = {b[0]}, beta = {b[1]}")
        if fk == "gimm" and cols is None:
            np.savetxt(out / f"weight_grid_n{n}.csv", np.array(rows), delimiter=",", comments="",
                       header="w_im,w_mul,w_ttc,w_ep,w_c,beta,pdms_hold")
    # A3: the native plan scored by its nearest anchor's head outputs (N0's lookup) instead of its own slot
    hd = heads_full["gimm"][0]
    near = Dh[:, :K].argmin(1)                                   # per held-out token: nearest anchor of native s = 1.00
    lk = {m: torch.cat([v[:, :K], v[torch.arange(len(ho)), near][:, None].expand(-1, NS)], 1) for m, v in hd.items()}
    b, _ = grid_search(lk, Dh, Ph, nat_mask)
    tab.append({"arm": "A3 GIMM feats, native scored by its nearest anchor", "w_im": b[0][0], "w_mul": b[0][1], "w_ttc": b[0][2],
                "w_ep": b[0][3], "w_c": b[0][4], "beta": b[1], "pdms_hold": 100 * b[2]})
    rl.log.info(f"A3 nearest-anchor lookup: hold PDMS {100 * b[2]:.3f}")
    import pandas as pd
    pd.DataFrame(tab).to_csv(out / f"hold_table_n{n}.csv", index=False, float_format="%.4f")
    (out / f"hold_refs_n{n}.json").write_text(json.dumps({"n": len(toks), "n_hold": len(hold_r), "refs": ref, "heads": rec},
                                                          indent=1, default=float))
    rl.event("hold", table=tab, refs=ref)
    if final:
        _final(rl, n, d, X["gimm"], heads_full["gimm"][1], best_of["A0 GIMM feats + native slots (N1)"], ref, tab)
    rl.close()


def _final(rl, n, d, Xtr_all, full, best, ref, tab):
    """Refit on all rows (done in fit_heads), then the navtest poses: the one N1 output that gets scored."""
    out = run_dir()
    w, beta, v = best
    te = H.load("navtest", False)
    z = np.load(data_dir() / FEAT / "lb_navtest_n12146.npz")
    c = np.load(out / "cands_navtest_n12146.npz")
    assert (z["tokens"] == c["tokens"]).all()
    pos = dict(zip(te["tokens"].tolist(), range(len(te["tokens"]))))
    r = np.array([pos[t] for t in z["tokens"].tolist()])
    nat = z["native"]
    # standardisation statistics of the training rows, as in cmd_fit
    Xe = _std(d["ego"], te["ego"][r])[1]
    Xf = _std(d["gimm"], z["temporal"])[1]
    Xn = _std(d["native"].reshape(len(d["native"]), -1), nat.reshape(len(nat), -1))[1]
    Xt = torch.cat([Xe, Xf, Xn], 1)
    logits = {m: Xt @ full[m][0] + full[m][1] for m in SUBS}
    A = torch.as_tensor(d["anchors"], device=H.DEV)
    Nt = torch.as_tensor(nat, device=H.DEV)
    Ct = torch.as_tensor(c["cands"], device=H.DEV)
    Dm = torch.cat([_dist(A[None].expand(len(Nt), -1, -1, -1), Nt), _dist(Ct, Nt)], 1)
    nat_mask = torch.zeros(K + NS, device=H.DEV)
    nat_mask[K:] = 1
    sel = scores(logits, Dm, w, beta, nat_mask).argmax(1).cpu().numpy()
    pool = np.concatenate([np.broadcast_to(d["anchors"][None], (len(sel), K, 8, 3)), c["cands"]], 1)
    poses = pool[np.arange(len(sel)), sel].astype(np.float32)
    np.savez(out / "navtest_n1.npz", tokens=z["tokens"], poses=poses)
    share = {"native": float((sel >= K).mean()), "native_stretched": float((sel > K).mean()),
             **{f"slot_{s:.2f}": float((sel == K + i).mean()) for i, s in enumerate(SCALES)}}
    (out / "select.json").write_text(json.dumps({"n_train": n, "weights": w, "beta": beta, "pdms_hold": v, "hold_refs": ref,
                                                 "hold_table": tab, "navtest_shares": share}, indent=1, default=float))
    rl.log.info(f"navtest poses written; shares {share}")


# ---------------------------------------------------------------- report

def cmd_report():
    import pandas as pd
    from experiments.skill_pack.archive.skill_pack_n0 import CMD, NAVFULL, V1, per_token
    out = run_dir()
    n1, n0, nat, hyd = (per_token(f"v1_navtest_{s}") for s in ("sp_n1_navtest", "sp_n0_navtest",
                                                                  "opi_navfull_gimm_g0.2-cinque__base", "e6_hydra_cinque"))
    hum = per_token("v1_navtest_human")
    mt = json.loads((data_dir() / NAVFULL / "meta.json").read_text())
    grp = pd.Series({t: ("start (v0<1)" if s < 1 else CMD[int(c)]) for t, c, s in zip(mt["names"], mt["cmd"], mt["speed"])})
    rows, pairs = [], []
    for name, dd in (("N1", n1), ("N0", n0), ("native (84.2 run)", nat), ("E6 Hydra (stored)", hyd), ("human", hum)):
        rows.append({"row": name, "n": len(dd), **{k: 100 * dd[k].mean() for k in V1.values()}})
    rng = np.random.default_rng(1)
    for name, dd in (("N1 - N0", n0), ("N1 - native", nat), ("N1 - E6 Hydra", hyd)):
        x, y = n1["PDMS"].align(dd["PDMS"], join="inner")
        e = (x - y).to_numpy()
        bs = e[rng.integers(0, len(e), (10_000, len(e)))].mean(1)
        pairs.append({"pair": name, "n": len(e), "diff": 100 * e.mean(), "lo": 100 * np.percentile(bs, 2.5),
                      "hi": 100 * np.percentile(bs, 97.5)})
    g = grp.reindex(n1.index)
    bycmd = pd.DataFrame({nm: dd["PDMS"].reindex(n1.index).groupby(g).mean() * 100
                          for nm, dd in (("N1", n1), ("N0", n0), ("native", nat), ("E6 Hydra", hyd))})
    ep_hum = float((n1["EP"] > hum["EP"].reindex(n1.index)).mean())
    res = {"table": pd.DataFrame(rows), "paired": pd.DataFrame(pairs), "by_command": bycmd}
    for k, v in res.items():
        v.to_csv(out / f"report_{k}.csv", float_format="%.3f")
        print(v.round(2).to_markdown())
    print(f"share of tokens with N1 EP > human EP: {ep_hum:.3f}")
    (out / "report.json").write_text(json.dumps({"ep_above_human": ep_hum, "paired": pairs}, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("cands", "fit", "report"))
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    {"cands": lambda: cmd_cands(a.n), "fit": lambda: cmd_fit(a.n, a.final), "report": cmd_report}[a.cmd]()
