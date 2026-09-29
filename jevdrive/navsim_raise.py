"""NAVSIM score raising on top of skill pack N1 (todos/2026-09-30-navsim-raise.md, every arm pre-registered there
before its navtest score). Same data as N1: the 19 968 E6 navtrain tokens, GIMM Cinque `temporal`, 1024 anchors +
native slots, fold 0 of the log split as held-out logs. navtest / navhard are never used for a choice.

    python -m jevdrive.navsim_raise fit --arm n1|n1b [--final]   held-out selection, all-row refit, test poses (--final)
    python -m jevdrive.navsim_raise repro                         N1 refit vs the stored navtest_n1.npz
    python -m jevdrive.navsim_raise report --arm n1b              navtest paired deltas (after the one scoring)
    python -m jevdrive.navsim_raise explore <what>                held-out estimates only (no test data touched)
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from . import navsim_heads as H
from . import skill_pack_n1 as N1
from .common import data_dir, get_logger

log = get_logger(__name__)
RUN = "runs/skill_pack/raise"
N = 19968
SUBS, K = N1.SUBS, N1.K
LAMS = {"n1": (1e-5, 1e-4, 1e-3, 1e-2),
        "n1b": (0, 1e-9, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 1e-4, 1e-3, 1e-2)}
ARMS = {"n1": ("n1", False, 0), "n1b": ("n1b", False, 0), "fam": ("n1b", True, 0),
        "scale": ("n1b", False, 40000)}         # arm -> (lambda grid, extra FAM slots, scale-up rows)
TEST = {"navtest": ("lb_navtest_n12146", "navtest"), "navhard": ("lb_navhard_n5912", "navhard_two_stage")}
# extra native-family slots (stretch s, lateral scale l) on top of N1's four stretches (s, 1.0), s = 1.00 ... 1.15
FAM = ((0.90, 1.0), (1.20, 1.0), (1.25, 1.0), (1.30, 1.0), (1.40, 1.0), (1.00, 0.8), (1.00, 1.2), (1.15, 0.8), (1.15, 1.2))


def run_dir(*sub) -> Path:
    d = data_dir() / RUN / Path(*sub) if sub else data_dir() / RUN
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- scale-up token set (more navtrain rows)

def tokens2(m: int):
    """Extra navtrain tokens for the scale-up arm, fixed order (seed 20260930): stage one with a future, a v1_navtrain
    metric cache, not among E6's 20 000, not in N0's selection set T, and not from a log of N1's held-out fold."""
    import glob
    from . import elicit_e6 as E6
    tr = E6._navtrain(False)
    e6 = (data_dir() / N1.E6_PREP / "tokens.txt").read_text().split()
    lane = set((data_dir() / "runs/op_lb/lb_navtrain/tokens.txt").read_text().split())
    pos = dict(zip(tr["tokens"].tolist(), range(len(tr["tokens"]))))
    e6_logs = tr["log"][[pos[t] for t in e6]]
    hold_logs = set(e6_logs[H._group_folds(e6_logs, 5, seed=N1.SEED_HOLD) == 0])
    cached = {os.path.basename(os.path.dirname(f)) for f in
              glob.glob(str(data_dir() / "runs/navsim/metric_cache/v1_navtrain/*/*/*/metric_cache.pkl"))}
    bad = set(e6) | lane
    pool = [t for t, g in zip(tr["tokens"].tolist(), tr["log"].tolist()) if t not in bad and g not in hold_logs and t in cached]
    pool = [pool[i] for i in np.random.default_rng(20260930).permutation(len(pool))][:m]
    out = run_dir("scale") / "tokens.txt"
    out.write_text("\n".join(pool) + "\n")
    print(f"{len(pool)} tokens (pool before the cut: see log) -> {out}; held-out logs excluded: {len(hold_logs)}")


def pilotcheck():
    """Arm S pilot checklist (todos/2026-09-30-navsim-raise.md): exit 1 if any item fails."""
    R = run_dir("scale")
    f = np.load(data_dir() / N1.FEAT / "lb_n2train_n192.npz")
    ref = np.load(data_dir() / N1.FEAT / f"lb_n1train_n{N}.npz")["temporal"][:2000]
    an, na = _chunks(R / "anchor_score_pilot"), _chunks(R / "native_score_pilot")
    toks = f["tokens"].tolist()
    e6 = [np.load(q)["pdms"] for q in sorted((data_dir() / N1.E6_PREP / "score").glob("chunk_*.npz"))[:8]]
    tf = f["temporal"]
    c = {"C1 label coverage": float(np.mean([t in an and t in na for t in toks])),
         "C2 native s=1.00 mean PDMS": 100 * float(np.mean([na[t][1][0] for t in toks if t in na])),
         "C3 temporal NaN/Inf": int((~np.isfinite(tf)).sum()),
         "C3 norm ratio (new / lb_n1train)": float(np.median(np.linalg.norm(tf, axis=1)) / np.median(np.linalg.norm(ref, axis=1))),
         "C4 mean anchor PDMS new vs E6": [100 * float(np.mean([an[t][1].mean() for t in toks if t in an])),
                                          100 * float(np.concatenate(e6).mean())]}
    vr = R / "vram_pilot.txt"
    c["C5 peak VRAM GB"] = max(float(x) for x in vr.read_text().split()) / 1024 if vr.exists() else -1
    ok = {"C1": c["C1 label coverage"] >= 0.95, "C2": 75 <= c["C2 native s=1.00 mean PDMS"] <= 92,
          "C3": c["C3 temporal NaN/Inf"] == 0 and 0.8 <= c["C3 norm ratio (new / lb_n1train)"] <= 1.25,
          "C4": abs(c["C4 mean anchor PDMS new vs E6"][0] - c["C4 mean anchor PDMS new vs E6"][1]) <= 5,
          "C5": 0 < c["C5 peak VRAM GB"] <= 80}
    c["pass"] = ok
    (R / "pilot_check.json").write_text(json.dumps(c, indent=1))
    print(json.dumps(c, indent=1))
    raise SystemExit(0 if all(ok.values()) else 1)


# ---------------------------------------------------------------- data

def lat(p: np.ndarray, l: float) -> np.ndarray:
    """Lateral scale of a (8, 3) pose sequence: y times l, heading re-derived for the scaled curve."""
    q = p.copy()
    q[:, 1] *= l
    q[:, 2] = np.arctan2(l * np.sin(p[:, 2]), np.cos(p[:, 2]))
    return q


def family(nat: np.ndarray, fam=FAM) -> np.ndarray:
    from .skill_pack_n0 import stretch
    return np.stack([[lat(stretch(p, s), l) for s, l in fam] for p in nat]).astype(np.float32)


def slots(extra: bool) -> list:
    return [(s, 1.0) for s in N1.SCALES] + (list(FAM) if extra else [])


def cands_fam():
    z = np.load(data_dir() / N1.FEAT / f"lb_n1train_n{N}.npz")
    out = run_dir("fam") / "cands_train.npz"
    np.savez(out, tokens=z["tokens"], cands=family(z["native"]))
    print(out)


def _add_fam(d: dict):
    ns = {}
    for f in sorted((run_dir("fam") / "score_train").glob("chunk_*.npz")):
        z = np.load(f)
        ns.update({t: (z["sub"][j], z["pdms"][j]) for j, t in enumerate(z["tokens"].tolist())})
    toks = d["tokens"].tolist()
    assert all(t in ns for t in toks), "family labels missing"
    c = np.load(run_dir("fam") / "cands_train.npz")
    assert (c["tokens"] == d["tokens"]).all()
    d["cands"] = np.concatenate([d["cands"], c["cands"]], 1)
    d["sub"] = np.concatenate([d["sub"], np.stack([ns[t][0] for t in toks])], 1)
    d["pdms"] = np.concatenate([d["pdms"], np.stack([ns[t][1] for t in toks])], 1)


def _chunks(dirp: Path) -> dict:
    ns = {}
    for f in sorted(dirp.glob("chunk_*.npz")):
        z = np.load(f)
        ns.update({t: (z["sub"][j], z["pdms"][j]) for j, t in enumerate(z["tokens"].tolist())})
    return ns


def _add_scale(d: dict, m: int):
    """Append the scale-up rows (lb_n2train, first m tokens): features, E6-style anchor labels, native labels.
    None of them comes from a held-out log (tokens2), so fold 0 stays exactly N1's held-out set."""
    from . import elicit_e6 as E6
    from .skill_pack_n0 import stretch
    f = np.load(data_dir() / N1.FEAT / f"lb_n2train_n{m}.npz")
    toks = f["tokens"].tolist()
    an, na = _chunks(run_dir("scale") / "anchor_score"), _chunks(run_dir("scale") / "native_score")
    keep = [i for i, t in enumerate(toks) if t in an and t in na]       # tokens the devkit could score
    toks = [toks[i] for i in keep]
    tr = E6._navtrain(True)
    pos = dict(zip(tr["tokens"].tolist(), range(len(tr["tokens"]))))
    r = np.array([pos[t] for t in toks])
    nat = f["native"][keep]
    c = np.stack([[stretch(p, s) for s in N1.SCALES] for p in nat]).astype(np.float32)
    sub = np.concatenate([np.stack([an[t][0] for t in toks]), np.stack([na[t][0] for t in toks])], 1)
    pd_ = np.concatenate([np.stack([an[t][1] for t in toks]), np.stack([na[t][1] for t in toks])], 1)
    new = {"tokens": np.array(toks), "log": tr["log"][r], "ego": tr["ego"][r], "hold_feat": tr["cinque"][r],
           "gimm": f["temporal"][keep], "native": nat, "cands": c, "sub": sub, "pdms": pd_}
    for k, v in new.items():
        d[k] = np.concatenate([d[k], v.astype(d[k].dtype)])
    log.info(f"scale-up: + {len(toks)} rows ({m - len(toks)} without devkit labels)")


def prep(n: int = N, extra: bool = False, scale: int = 0) -> dict:
    """Training tensors exactly as skill_pack_n1.cmd_fit builds them (GIMM features, A0); extra = + FAM slots;
    scale = + the first `scale` scale-up tokens (their logs are never held out)."""
    d = N1.load_train(n)
    if extra:
        _add_fam(d)
    if scale:
        _add_scale(d, scale)
    hold = np.zeros(len(d["tokens"]), bool)
    hold[:n] = H._group_folds(d["log"][:n], 5, seed=N1.SEED_HOLD) == 0                # N1's held-out rows, unchanged
    nat = d["native"]
    A = torch.as_tensor(d["anchors"], device=H.DEV)
    C = torch.as_tensor(d["cands"], device=H.DEV)
    Nt = torch.as_tensor(nat, device=H.DEV)
    Dm = torch.cat([N1._dist(A[None].expand(len(nat), -1, -1, -1), Nt), N1._dist(C, Nt)], 1)
    X = torch.cat([N1._std(d["ego"])[0], N1._std(d["gimm"])[0], N1._std(nat.reshape(len(nat), -1))[0]], 1)
    S = d["cands"].shape[1]
    mask = torch.zeros(K + S, device=H.DEV)
    mask[K:] = 1
    return {"d": d, "X": X, "T": {m: torch.as_tensor(d["sub"][:, :, i], device=H.DEV) for i, m in enumerate(SUBS)},
            "P": torch.as_tensor(d["pdms"], device=H.DEV), "Dm": Dm, "mask": mask,
            "fit_r": np.flatnonzero(~hold), "hold_r": np.flatnonzero(hold), "logs": d["log"]}


def test_inputs(split: str, d: dict, extra: bool = False) -> dict:
    """Test-split features standardised with the training statistics, candidates, distances."""
    from .skill_pack_n0 import stretch
    feat, hsplit = TEST[split]
    z = np.load(data_dir() / N1.FEAT / f"{feat}.npz")
    te = H.load(hsplit, False)
    pos = dict(zip(te["tokens"].tolist(), range(len(te["tokens"]))))
    r = np.array([pos[t] for t in z["tokens"].tolist()])
    nat = z["native"]
    X = torch.cat([N1._std(d["ego"], te["ego"][r])[1], N1._std(d["gimm"], z["temporal"])[1],
                   N1._std(d["native"].reshape(len(d["native"]), -1), nat.reshape(len(nat), -1))[1]], 1)
    c = np.stack([[stretch(p, s) for s in N1.SCALES] for p in nat]).astype(np.float32)
    if extra:
        c = np.concatenate([c, family(nat)], 1)
    A = torch.as_tensor(d["anchors"], device=H.DEV)
    Nt = torch.as_tensor(nat, device=H.DEV)
    Dm = torch.cat([N1._dist(A[None].expand(len(nat), -1, -1, -1), Nt), N1._dist(torch.as_tensor(c, device=H.DEV), Nt)], 1)
    return {"tokens": z["tokens"], "X": X, "cands": c, "Dm": Dm, "cmd": te["ego"][r][:, -4:].argmax(1)}


# ---------------------------------------------------------------- fit

def fit_arm(arm: str, final: bool):
    from .runlog import RunLog
    H.DEV = "cuda" if torch.cuda.is_available() else "cpu"
    rl = RunLog("skill_pack", "raise", f"fit_{arm}" + ("_final" if final else ""))
    out = run_dir(arm)
    lg_name, extra, scale = ARMS[arm]
    p = prep(extra=extra, scale=scale)
    N1.LAMS = LAMS[lg_name]                               # n1 vs n1b: only the lambda grid differs
    rec = {}
    hd, full = N1.fit_heads(p["X"], p["T"], p["fit_r"], p["hold_r"], "gimm", rec)
    ho = torch.as_tensor(p["hold_r"], device=H.DEV)
    best, rows = N1.grid_search(hd, p["Dm"][ho], p["P"][ho], p["mask"])
    w, beta, v = best
    rl.log.info(f"[{arm}] held-out PDMS {100 * v:.3f} at w = {w}, beta = {beta}; lambdas "
                f"{ {m: rec['gimm/' + m]['lam'] for m in SUBS} }")
    torch.save({m: {"hold": hd[m].half().cpu()} for m in SUBS}, out / "hold_logits.pt")
    np.savez(out / "heads.npz", **{f"W_{m}": full[m][0].cpu().numpy() for m in SUBS},
             **{f"b_{m}": full[m][1].cpu().numpy() for m in SUBS})
    sel = {"arm": arm, "n": N, "n_hold": len(p["hold_r"]), "lams": LAMS[lg_name], "slots": slots(extra), "weights": w, "beta": beta, "pdms_hold": v,
           "heads": rec}
    if final:
        for split in TEST:
            if not (data_dir() / N1.FEAT / f"{TEST[split][0]}.npz").exists():
                rl.log.info(f"{split}: features not extracted, skipped")
                continue
            t = test_inputs(split, p["d"], extra)
            lg = {m: t["X"] @ full[m][0] + full[m][1] for m in SUBS}
            s = N1.scores(lg, t["Dm"], w, beta, p["mask"]).argmax(1).cpu().numpy()
            pool = np.concatenate([np.broadcast_to(p["d"]["anchors"][None], (len(s), K, 8, 3)), t["cands"]], 1)
            np.savez(out / f"{split}_{arm}.npz", tokens=t["tokens"], poses=pool[np.arange(len(s)), s].astype(np.float32))
            sel[f"{split}_shares"] = {"native": float((s >= K).mean()),
                                      **{f"slot_{a:.2f}x{b:.1f}": float((s == K + i).mean()) for i, (a, b) in enumerate(slots(extra))}}
            rl.log.info(f"{split}: poses written, shares {sel[f'{split}_shares']}")
    (out / "select.json").write_text(json.dumps(sel, indent=1, default=float))
    rl.close()


def repro():
    a = np.load(run_dir("n1") / "navtest_n1.npz")
    b = np.load(data_dir() / N1.RUN / "navtest_n1.npz")
    assert (a["tokens"] == b["tokens"]).all()
    e = np.abs(a["poses"] - b["poses"]).reshape(len(a["poses"]), -1).max(1)
    frac = float((e < 1e-4).mean())
    print(json.dumps({"same_pose_frac": frac, "n": len(e), "max_err": float(e.max())}))
    (run_dir("n1") / "repro.json").write_text(json.dumps({"same_pose_frac": frac, "max_err": float(e.max())}))
    assert frac >= 0.99, "N1 refit does not reproduce the stored navtest selection"


# ---------------------------------------------------------------- report

def paired(a, b, rng_seed=1):
    x, y = a["PDMS"].align(b["PDMS"], join="inner")
    e = (x - y).to_numpy()
    bs = e[np.random.default_rng(rng_seed).integers(0, len(e), (10_000, len(e)))].mean(1)
    return {"n": len(e), "diff": 100 * e.mean(), "lo": 100 * np.percentile(bs, 2.5), "hi": 100 * np.percentile(bs, 97.5),
            "bs": bs}


def report(arm: str, vs=("sp_n1_navtest", "sp_n0_navtest", "opi_navfull_gimm_g0.2-cinque__base"), looks: int = 0):
    import pandas as pd
    from .skill_pack_n0 import CMD, NAVFULL, V1, per_token
    me = per_token(f"v1_navtest_sp_{arm}_navtest")
    rows = [{"row": arm, **{k: 100 * me[k].mean() for k in V1.values()}}]
    pairs = []
    for v in vs:
        o = per_token(f"v1_navtest_{v}")
        rows.append({"row": v, **{k: 100 * o[k].mean() for k in V1.values()}})
        r = paired(me, o)
        bs = r.pop("bs")
        if looks > 1:
            r["lo_bonf"], r["hi_bonf"] = (100 * np.percentile(bs, q) for q in (2.5 / looks, 100 - 2.5 / looks))
        pairs.append({"pair": f"{arm} - {v}", **r})
    mt = json.loads((data_dir() / NAVFULL / "meta.json").read_text())
    grp = pd.Series({t: ("start (v0<1)" if s < 1 else CMD[int(c)]) for t, c, s in zip(mt["names"], mt["cmd"], mt["speed"])})
    g = grp.reindex(me.index)
    bycmd = pd.DataFrame({nm: per_token(f"v1_navtest_{nm}")["PDMS"].reindex(me.index).groupby(g).mean() * 100
                          for nm in (f"sp_{arm}_navtest", *vs)})
    out = run_dir(arm)
    for k, v in {"table": pd.DataFrame(rows), "paired": pd.DataFrame(pairs), "by_command": bycmd}.items():
        v.to_csv(out / f"report_{k}.csv", float_format="%.3f")
        print(v.round(2).to_markdown())


def navhard_read(name: str) -> dict:
    import glob
    import pandas as pd
    f = sorted(glob.glob(str(data_dir() / f"runs/navsim/eval/v2_navhard_two_stage_{name}/*/*.csv")))[-1]
    r = pd.read_csv(f).query("token == 'extended_pdm_score_combined'").iloc[0]
    return {"combined": 100 * r["extended_pdm_score_combined"], "s1": 100 * r["score_stage_one"] if "score_stage_one" in r else None,
            "s2": 100 * r["score_stage_two"] if "score_stage_two" in r else None, "csv": f}


# ---------------------------------------------------------------- held-out exploration (no test data)

def _hold_pdms(lg: dict, p: dict, w=None, beta=None) -> tuple[float, tuple]:
    ho = torch.as_tensor(p["hold_r"], device=H.DEV)
    if w is None:
        (w, beta, v), _ = N1.grid_search(lg, p["Dm"][ho], p["P"][ho], p["mask"])
        return 100 * v, (w, beta)
    s = N1.scores(lg, p["Dm"][ho], w, beta, p["mask"]).argmax(1)
    return 100 * float(p["P"][ho][torch.arange(len(ho)), s].mean()), (w, beta)


def _lams(arm="n1b") -> dict:
    return {m: json.loads((run_dir(arm) / "select.json").read_text())["heads"][f"gimm/{m}"]["lam"] for m in SUBS}


def _fit_on(p, rows, lams, X=None, iters=100) -> dict:
    X = p["X"] if X is None else X
    ho = p["hold_r"]
    out = {}
    for m in SUBS:
        W, b = N1.bce_fit(X, p["T"][m], rows, lams[m], iters)
        out[m] = X[ho] @ W + b
    return out


def explore(what: str):
    from .runlog import RunLog
    H.DEV = "cuda" if torch.cuda.is_available() else "cpu"
    rl = RunLog("skill_pack", "raise", f"explore_{what}")
    p = prep(extra=(what == "fam"))
    lams = _lams()
    ho = torch.as_tensor(p["hold_r"], device=H.DEV)
    res = {}
    if what == "rules":                          # selection rules on N1b's held-out logits
        hd = {m: v.float().to(H.DEV) for m, v in
              ((m, d["hold"]) for m, d in torch.load(run_dir("n1b") / "hold_logits.pt").items())}
        res["grid (N1b)"] = _hold_pdms(hd, p)[0]
        sg = {m: torch.sigmoid(v) for m, v in hd.items()}
        ep = sg["NC"] * sg["DAC"] * (5 * sg["EP"] + 5 * sg["TTC"] + 2 * sg["C"]) / 12
        Ph = p["P"][ho]
        ar = torch.arange(len(ho), device=H.DEV)
        res["expected PDMS (product of sigmoids)"] = 100 * float(Ph[ar, ep.argmax(1)].mean())
        for tau in (0.5, 0.7, 0.8, 0.9, 0.95):
            ok = (sg["NC"] > tau) & (sg["DAC"] > tau)
            sc = torch.where(ok, sg["EP"] + sg["TTC"], -1 + sg["NC"] * sg["DAC"])
            res[f"gate NC,DAC > {tau} then max EP+TTC"] = 100 * float(Ph[ar, sc.argmax(1)].mean())
        res["oracle"] = 100 * float(Ph.max(1).values.mean())
    elif what == "curve":                        # learning curve over the fit logs, lambdas and weights as N1b
        fl = p["logs"][p["fit_r"]]
        u = np.random.default_rng(0).permutation(np.unique(fl))
        for frac in (1 / 8, 1 / 4, 1 / 2, 1):
            keep = set(u[:max(1, round(frac * len(u)))])
            rows = p["fit_r"][np.array([g in keep for g in fl])]
            res[f"{frac:.3f} ({len(rows)} rows)"] = _hold_pdms(_fit_on(p, rows, lams), p)[0]
            rl.log.info(f"curve {frac}: {res[f'{frac:.3f} ({len(rows)} rows)']:.3f}")
    elif what == "bag":                          # fold bagging: average of 4 heads that each drop one fit fold
        f = H._group_folds(p["logs"], 5, seed=N1.SEED_HOLD)
        single = _fit_on(p, p["fit_r"], lams)
        bag = [_fit_on(p, p["fit_r"][f[p["fit_r"]] != k], lams) for k in range(1, 5)]
        avg = {m: torch.stack([b[m] for b in bag]).mean(0) for m in SUBS}
        res = {"single (all fit rows)": _hold_pdms(single, p)[0], "bag of 4 (each 3/4 of fit rows)": _hold_pdms(avg, p)[0],
               **{f"member {k + 1}": _hold_pdms(b, p)[0] for k, b in enumerate(bag)}}
    elif what == "iters":                        # L-BFGS iterations at lambda = the N1b choice
        for it in (100, 300, 1000):
            res[f"iters {it}"] = _hold_pdms(_fit_on(p, p["fit_r"], lams, iters=it), p)[0]
            rl.log.info(f"iters {it}: {res[f'iters {it}']:.3f}")
    elif what == "mlp":
        res = _mlp(p, rl)
    elif what == "feats":                        # extra input views at zero GPU cost: E6's hold-input taps
        from . import elicit_e6 as E6
        tr = E6._navtrain(True)
        pos = dict(zip(tr["tokens"].tolist(), range(len(tr["tokens"]))))
        r = np.array([pos[t] for t in p["d"]["tokens"].tolist()])
        views = {"+ Lebowski hold": [tr["lebowski"][r]], "+ Cinque hold": [tr["cinque"][r]],
                 "+ both hold": [tr["lebowski"][r], tr["cinque"][r]]}
        res["base (N1b features)"] = _hold_pdms(_fit_on(p, p["fit_r"], lams), p)[0]
        for nm, vs in views.items():
            X2 = torch.cat([p["X"], *[N1._std(v)[0] for v in vs]], 1)
            res[nm] = _hold_pdms(_fit_on(p, p["fit_r"], lams, X=X2), p)[0]
            rl.log.info(f"{nm}: {res[nm]:.3f}")
    elif what == "fam":                          # N1b heads over 4 + 9 native-family slots
        Ph = p["P"][ho]
        res["oracle native 4 slots"] = 100 * float(Ph[:, K:K + 4].max(1).values.mean())
        res["oracle native 13 slots"] = 100 * float(Ph[:, K:].max(1).values.mean())
        res["mean PDMS per extra slot"] = {f"{a:.2f}x{b:.1f}": 100 * float(Ph[:, K + 4 + i].mean()) for i, (a, b) in enumerate(FAM)}
        lg = _fit_on(p, p["fit_r"], lams)
        res["N1b heads, 13 native slots"], wb = _hold_pdms(lg, p)
        s = N1.scores(lg, p["Dm"][ho], *wb, p["mask"]).argmax(1)
        res["weights"], res["slot shares"] = wb, {f"{a:.2f}x{b:.1f}": float((s == K + i).float().mean())
                                                  for i, (a, b) in enumerate(slots(True))}
        res["anchor share"] = float((s < K).float().mean())
    else:
        raise SystemExit(f"unknown {what}")
    rl.log.info(json.dumps(res, indent=1))
    (run_dir("explore") / f"{what}.json").write_text(json.dumps(res, indent=1, default=float))
    rl.close()


def _mlp(p, rl, hidden=1024, epochs=60, lr=1e-3, wd=1e-4, drop=0.1, bs=512, seed=0) -> dict:
    """Shared-trunk MLP heads (Hydra-style: one trunk, five sub-score heads over all candidates), BCE on soft targets.
    Epoch by held-out BCE (as N1 picks lambda), then the weight grid on held-out PDMS."""
    torch.manual_seed(seed)
    X, S = p["X"], len(p["mask"])
    Tall = torch.stack([p["T"][m] for m in SUBS], 1)                          # (n, 5, S)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.GELU(), torch.nn.Dropout(drop),
                              torch.nn.Linear(hidden, hidden), torch.nn.GELU(), torch.nn.Dropout(drop),
                              torch.nn.Linear(hidden, 5 * S)).to(H.DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * (len(p["fit_r"]) // bs + 1))
    fr, ho = torch.as_tensor(p["fit_r"], device=H.DEV), torch.as_tensor(p["hold_r"], device=H.DEV)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    best, res = (np.inf, None, -1), {}
    for ep in range(epochs):
        net.train()
        for i in fr[torch.randperm(len(fr), device=H.DEV)].split(bs):
            loss = bce(net(X[i]).view(-1, 5, S), Tall[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            lg = net(X[ho]).view(-1, 5, S)
            hb = float(bce(lg, Tall[ho]))
        if hb < best[0]:
            best = (hb, {m: lg[:, i].clone() for i, m in enumerate(SUBS)}, ep)
        if ep % 10 == 9:
            rl.log.info(f"mlp epoch {ep + 1}: hold BCE {hb:.5f}")
    res["best epoch"], res["hold BCE (mean of 5 heads)"] = best[2] + 1, best[0]
    res["mlp hold PDMS"], wb = _hold_pdms(best[1], p)
    res["weights"] = wb
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("fit", "repro", "report", "explore", "tokens2", "pilotcheck"))
    ap.add_argument("what", nargs="?")
    ap.add_argument("--arm", default="n1b")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--looks", type=int, default=0)
    ap.add_argument("--m", type=int, default=40000)
    a = ap.parse_args()
    {"fit": lambda: fit_arm(a.arm, a.final), "repro": repro, "report": lambda: report(a.arm, looks=a.looks),
     "explore": lambda: explore(a.what), "tokens2": lambda: tokens2(a.m), "pilotcheck": pilotcheck}[a.cmd]()
