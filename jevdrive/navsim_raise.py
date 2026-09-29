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
FAM2 = ((1.50, 1.0), (1.60, 1.0), (1.75, 1.0), (1.30, 0.8), (1.40, 0.8), (1.15, 0.6), (1.40, 0.6), (1.00, 0.9), (1.30, 0.9))
FAMS = (("fam", FAM), ("fam2", FAM2))


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


def fam_list(level: int) -> list:
    return [x for _, f in FAMS[:int(level)] for x in f]


def slots(extra, alt: bool = False) -> list:
    return [(s, 1.0) for s in N1.SCALES] + fam_list(extra) + ([(f"{k}*{x:.2f}", 1.0) for k in ALT for x in ALT_S] if alt else [])


ALT, ALT_S = ("lcL", "lcR", "tL", "tR"), (1.00, 1.15)      # forced-desire alternative plans x stretch


def alt_cands(tag: str = f"lb_n1train_n{N}", out: str = "cands_train.npz") -> np.ndarray:
    """(n, 8, 8, 3): the four forced-desire alternative plans of Cinque, each at stretch 1.00 and 1.15."""
    from .skill_pack_n0 import stretch
    base, n = tag.rsplit("_n", 1)
    z = np.load(data_dir() / N1.FEAT / f"{base}__alt_n{n}.npz")
    c = np.stack([np.stack([stretch(p, s) for k in ALT for p in [z[f"native_{k}"][i]] for s in ALT_S])
                  for i in range(len(z["tokens"]))]).astype(np.float32)
    if out:
        np.savez(run_dir("alt") / out, tokens=z["tokens"], cands=c)
    return c


def cands_fam():
    z = np.load(data_dir() / N1.FEAT / f"lb_n1train_n{N}.npz")
    out = run_dir("fam") / "cands_train.npz"
    np.savez(out, tokens=z["tokens"], cands=family(z["native"]))
    print(out)


def _add_fam(d: dict, level: int):
    toks = d["tokens"].tolist()
    for name, _ in FAMS[:int(level)]:
        ns = _chunks(run_dir(name) / "score_train")
        assert all(t in ns for t in toks), f"{name} labels missing"
        c = np.load(run_dir(name) / "cands_train.npz")
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


def _add_scale(d: dict, m: int, extra: int = 0, alt: bool = False):
    """Append the scale-up rows (lb_n2train, first m tokens): features, E6-style anchor labels, native labels.
    None of them comes from a held-out log (tokens2), so fold 0 stays exactly N1's held-out set."""
    from . import elicit_e6 as E6
    from .skill_pack_n0 import stretch
    f = np.load(data_dir() / N1.FEAT / f"lb_n2train_n{m}.npz")
    toks = f["tokens"].tolist()
    an, na = _chunks(run_dir("scale") / "anchor_score"), _chunks(run_dir("scale") / "native_score")
    srcs = [an, na]
    if extra:
        fa = _chunks(run_dir("scale") / "fam_score")                     # FAM + FAM2 labels of the scale rows
        srcs.append(fa)
    if alt:
        al = _chunks(run_dir("scale") / "alt_score")
        srcs.append(al)
    keep = [i for i, t in enumerate(toks) if all(t in q for q in srcs)]   # tokens the devkit could score
    toks = [toks[i] for i in keep]
    tr = E6._navtrain(True)
    pos = dict(zip(tr["tokens"].tolist(), range(len(tr["tokens"]))))
    r = np.array([pos[t] for t in toks])
    nat = f["native"][keep]
    c = [np.stack([[stretch(p, s) for s in N1.SCALES] for p in nat]).astype(np.float32)]
    parts = [(an, None), (na, None)]
    if extra:
        nf = len(fam_list(extra))
        c.append(family(nat, fam_list(extra)))
        parts.append((fa, nf))
    if alt:
        c.append(alt_cands(f"lb_n2train_n{m}", out="")[keep])
        parts.append((al, None))
    c = np.concatenate(c, 1)
    sub = np.concatenate([np.stack([q[t][0][:k] for t in toks]) for q, k in parts], 1)
    pd_ = np.concatenate([np.stack([q[t][1][:k] for t in toks]) for q, k in parts], 1)
    new = {"tokens": np.array(toks), "log": tr["log"][r], "ego": tr["ego"][r], "hold_feat": tr["cinque"][r],
           "gimm": f["temporal"][keep], "native": nat, "cands": c, "sub": sub, "pdms": pd_}
    for k, v in new.items():
        d[k] = np.concatenate([d[k], v.astype(d[k].dtype)])
    log.info(f"scale-up: + {len(toks)} rows ({m - len(toks)} without devkit labels)")


def _add_alt(d: dict):
    ns = _chunks(run_dir("alt") / "score_train")
    toks = d["tokens"].tolist()
    assert all(t in ns for t in toks), "alt labels missing"
    c = np.load(run_dir("alt") / "cands_train.npz")
    assert (c["tokens"] == d["tokens"]).all()
    d["cands"] = np.concatenate([d["cands"], c["cands"]], 1)
    d["sub"] = np.concatenate([d["sub"], np.stack([ns[t][0] for t in toks])], 1)
    d["pdms"] = np.concatenate([d["pdms"], np.stack([ns[t][1] for t in toks])], 1)


def prep(n: int = N, extra: int = 0, scale: int = 0, alt: bool = False) -> dict:
    """Training tensors exactly as skill_pack_n1.cmd_fit builds them (GIMM features, A0); extra = + FAM (1) or
    FAM + FAM2 (2) slots; alt = + the 8 forced-desire slots; scale = + the first `scale` scale-up tokens (their logs
    are never held out, so the held-out rows stay N1's)."""
    d = N1.load_train(n)
    if extra:
        _add_fam(d, extra)
    if alt:
        _add_alt(d)
    if scale:
        _add_scale(d, scale, extra, alt)
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


def test_inputs(split: str, d: dict, extra: int = 0, alt: bool = False) -> dict:
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
        c = np.concatenate([c, family(nat, fam_list(extra))], 1)
    if alt:
        c = np.concatenate([c, alt_cands(feat, out="")], 1)
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


def report(arm: str, looks: int = 0):
    import pandas as pd
    from .skill_pack_n0 import CMD, NAVFULL, V1, per_token
    vs = ("sp_n1_navtest",) + (("sp_n1b_navtest",) if arm != "n1b" else ()) + ("sp_n0_navtest", "opi_navfull_gimm_g0.2-cinque__base")
    me = per_token(f"v1_navtest_sp_{arm}_navtest")
    hum = per_token("v1_navtest_human")
    print(f"share of tokens with EP > human EP: {float((me['EP'] > hum['EP'].reindex(me.index)).mean()):.3f}")
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
    elif what == "mlpcurve":                     # N2's chosen configuration on 1/4, 1/2, all of the fit logs
        p = prep(extra=2)
        X = _with_views(p["X"], _views("train", p["d"]["tokens"]))
        fl = p["logs"][p["fit_r"]]
        u = np.random.default_rng(0).permutation(np.unique(fl))
        for frac in (1 / 4, 1 / 2, 1):
            keep = set(u[:max(1, round(frac * len(u)))])
            rows = p["fit_r"][np.array([g in keep for g in fl])]
            res[f"{frac:.3f} ({len(rows)} rows)"] = _eval_cfg(p, X, ("mlp", 1, 2), rows, p["hold_r"], lams, rl)["pdms_hold"]
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

# ---------------------------------------------------------------- N2: head x input views x native slots

N2_GRID = [(h, v, f) for h in ("lin", "mlp") for v in (0, 1) for f in (0, 1, 2)]     # table order: simpler first
SEEDS, EPOCHS, LIN_ITERS = 5, 60, 300


def _views(split: str, tokens) -> list:
    """E6's hold-input `temporal` of Cinque and Lebowski for these tokens (train: navtrain; tests: their split)."""
    if split == "train":
        from . import elicit_e6 as E6
        tr = E6._navtrain(True)
    else:
        tr = H.load(TEST[split][1], True)
    pos = dict(zip(tr["tokens"].tolist(), range(len(tr["tokens"]))))
    r = np.array([pos[t] for t in list(tokens)])
    return [tr["cinque"][r], tr["lebowski"][r]]


def _with_views(Xtr, Vtr, Xte=None, Vte=None):
    if Xte is None:
        return torch.cat([Xtr, *[N1._std(v)[0] for v in Vtr]], 1)
    return torch.cat([Xte, *[N1._std(a, b)[1] for a, b in zip(Vtr, Vte)]], 1)


def _mlp_net(din, S, hidden=1024, drop=0.1):  # noqa: D103
    return torch.nn.Sequential(torch.nn.Linear(din, hidden), torch.nn.GELU(), torch.nn.Dropout(drop),
                               torch.nn.Linear(hidden, hidden), torch.nn.GELU(), torch.nn.Dropout(drop),
                               torch.nn.Linear(hidden, 5 * S)).to(H.DEV)


def train_mlp(X, Tall, fit_rows, seed, pick=None, epoch=None, evals=(), bs=512, lr=1e-3, wd=1e-4, hidden=1024):
    """One MLP head set. pick = (rows, ) -> epoch with the lowest BCE on those rows; else the fixed `epoch`.
    Returns the logits {SUB: (n_e, S)} for every matrix in evals at that epoch, and the epoch (1-based)."""
    torch.manual_seed(seed)
    S = Tall.shape[2]
    net = _mlp_net(X.shape[1], S, hidden)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    fr = torch.as_tensor(fit_rows, device=H.DEV)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=EPOCHS * (len(fr) // bs + 1))
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    best = (np.inf, None, -1)
    for ep in range(EPOCHS):
        net.train()
        for i in fr[torch.randperm(len(fr), device=H.DEV)].split(bs):
            loss = bce(net(X[i]).view(-1, 5, S), Tall[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        if pick is None and ep + 1 != epoch:
            continue
        net.eval()
        with torch.no_grad():
            if pick is not None:
                pr = torch.as_tensor(pick, device=H.DEV)
                hb = float(bce(net(X[pr]).view(-1, 5, S), Tall[pr]))
                if hb >= best[0]:
                    continue
            outs = [net(E).view(-1, 5, S) for E in evals]
            best = (hb if pick is not None else 0.0, [{m: o[:, j].clone() for j, m in enumerate(SUBS)} for o in outs], ep + 1)
        if pick is None:
            break
    return best[1], best[2]


def _eval_cfg(p, X, cfg, fit_rows, hold_rows, lams, rl, hidden=1024) -> dict:
    """Held-out PDMS of one N2 configuration on hold_rows (fit on fit_rows)."""
    h, v, f = cfg
    Xh = X[torch.as_tensor(hold_rows, device=H.DEV)]
    if h == "lin":
        lg = {}
        for m in SUBS:
            W, b = N1.bce_fit(X, p["T"][m], fit_rows, lams[m], LIN_ITERS)
            lg[m] = Xh @ W + b
        eps = []
    else:
        Tall = torch.stack([p["T"][m] for m in SUBS], 1)
        runs = [train_mlp(X, Tall, fit_rows, sd, pick=hold_rows, evals=(Xh,), hidden=hidden) for sd in range(SEEDS)]
        lg = {m: torch.stack([r[0][0][m] for r in runs]).mean(0) for m in SUBS}
        eps = [r[1] for r in runs]
    ho = torch.as_tensor(hold_rows, device=H.DEV)
    (w, beta, val), _ = N1.grid_search(lg, p["Dm"][ho], p["P"][ho], p["mask"])
    out = {"cfg": list(cfg), "hidden": hidden, "n_fit": len(fit_rows), "pdms_hold": 100 * val, "weights": list(w), "beta": beta, "epochs": eps}
    rl.log.info(f"N2 {cfg}: held-out {100 * val:.3f} (w {w}, beta {beta}, epochs {eps})")
    return out


def n2dev():
    from .runlog import RunLog
    H.DEV = "cuda" if torch.cuda.is_available() else "cpu"
    rl = RunLog("skill_pack", "raise", "n2dev")
    lams = _lams()
    res, V = [], None
    for f in (0, 1, 2):
        p = prep(extra=f)
        V = V if V is not None else _views("train", p["d"]["tokens"])
        folds = H._group_folds(p["logs"], 5, seed=N1.SEED_HOLD)
        for cfg in [c for c in N2_GRID if c[2] == f]:
            X = _with_views(p["X"], V) if cfg[1] else p["X"]
            res.append(_eval_cfg(p, X, cfg, np.flatnonzero(folds != 0), np.flatnonzero(folds == 0), lams, rl))
        del p
        torch.cuda.empty_cache()
    best = max(res, key=lambda r: r["pdms_hold"])                  # max() keeps the first maximum: simpler on ties
    out = {"grid": res, "chosen": best}
    # fold-1 replicate (descriptive): the chosen configuration and the N1b-style linear baseline
    p = prep(extra=best["cfg"][2])
    folds = H._group_folds(p["logs"], 5, seed=N1.SEED_HOLD)
    fit1, hold1 = np.flatnonzero(folds != 1), np.flatnonzero(folds == 1)
    Xc = _with_views(p["X"], V) if best["cfg"][1] else p["X"]
    out["fold1_chosen"] = _eval_cfg(p, Xc, tuple(best["cfg"]), fit1, hold1, lams, rl)
    p0 = prep()
    out["fold1_linear_n1b"] = _eval_cfg(p0, p0["X"], ("lin", 0, 0), fit1, hold1, lams, rl)
    (run_dir("n2") / "n2dev.json").write_text(json.dumps(out, indent=1, default=float))
    rl.log.info(f"chosen {best['cfg']} held-out {best['pdms_hold']:.3f}")
    rl.close()


def n2final():
    """Refit the chosen N2 configuration on all rows; navtest / navhard poses (each scored once by the chain)."""
    from .runlog import RunLog
    H.DEV = "cuda" if torch.cuda.is_available() else "cpu"
    rl = RunLog("skill_pack", "raise", "n2final")
    dev = json.loads((run_dir("n2") / "n2dev.json").read_text())["chosen"]
    h, v, f = dev["cfg"]
    w, beta = tuple(dev["weights"]), dev["beta"]
    p = prep(extra=f)
    Vtr = _views("train", p["d"]["tokens"]) if v else None
    X = _with_views(p["X"], Vtr) if v else p["X"]
    tests = {}
    for split in TEST:
        t = test_inputs(split, p["d"], f)
        t["X"] = _with_views(p["X"], Vtr, t["X"], _views(split, t["tokens"])) if v else t["X"]
        tests[split] = t
    allr = np.arange(len(X))
    if h == "lin":
        lams = _lams()
        full = {m: N1.bce_fit(X, p["T"][m], allr, lams[m], LIN_ITERS) for m in SUBS}
        lgs = {sp: {m: t["X"] @ full[m][0] + full[m][1] for m in SUBS} for sp, t in tests.items()}
    else:
        Tall = torch.stack([p["T"][m] for m in SUBS], 1)
        names = list(tests)
        runs = [train_mlp(X, Tall, allr, sd, epoch=e, evals=[tests[sp]["X"] for sp in names])[0]
                for sd, e in zip(range(SEEDS), dev["epochs"])]
        lgs = {sp: {m: torch.stack([r[j][m] for r in runs]).mean(0) for m in SUBS} for j, sp in enumerate(names)}
    sel = {"cfg": dev["cfg"], "weights": w, "beta": beta, "slots": slots(f)}
    for sp, t in tests.items():
        s = N1.scores(lgs[sp], t["Dm"], w, beta, p["mask"]).argmax(1).cpu().numpy()
        pool = np.concatenate([np.broadcast_to(p["d"]["anchors"][None], (len(s), K, 8, 3)), t["cands"]], 1)
        np.savez(run_dir("n2") / f"{sp}_n2.npz", tokens=t["tokens"], poses=pool[np.arange(len(s)), s].astype(np.float32))
        sel[f"{sp}_shares"] = {"anchor": float((s < K).mean()),
                               **{f"{a:.2f}x{b:.1f}": float((s == K + i).mean()) for i, (a, b) in enumerate(slots(f))}}
        rl.log.info(f"{sp}: poses written, shares {sel[f'{sp}_shares']}")
    (run_dir("n2") / "select.json").write_text(json.dumps(sel, indent=1, default=float))
    rl.close()


# ---------------------------------------------------------------- N3: N2's configuration + scale-up rows (+ alt slots)

S_ROWS = 40000
N3_GRID = [(alt, hid) for alt in (0, 1) for hid in (1024, 2048)]            # table order: simpler first


def _folds_all(p) -> np.ndarray:
    """N1's log folds on the first N rows; scale rows get -1 (never held out, always fitted)."""
    f = np.full(len(p["logs"]), -1)
    f[:N] = H._group_folds(p["logs"][:N], 5, seed=N1.SEED_HOLD)
    return f


def altdev():
    """Held-out value of the 8 forced-desire slots on N2's chosen configuration (19 968 rows, fold 0)."""
    from .runlog import RunLog
    H.DEV = "cuda" if torch.cuda.is_available() else "cpu"
    rl = RunLog("skill_pack", "raise", "altdev")
    lams, res = _lams(), {}
    for alt in (0, 1):
        p = prep(extra=2, alt=bool(alt))
        X = _with_views(p["X"], _views("train", p["d"]["tokens"]))
        f = _folds_all(p)
        ho = torch.as_tensor(np.flatnonzero(f == 0), device=H.DEV)
        if alt:
            res["oracle alt 8 slots alone"] = 100 * float(p["P"][ho][:, -8:].max(1).values.mean())
            res["oracle native 22 + alt 8"] = 100 * float(p["P"][ho][:, K:].max(1).values.mean())
        else:
            res["oracle native 22"] = 100 * float(p["P"][ho][:, K:].max(1).values.mean())
        res[f"alt={alt}"] = _eval_cfg(p, X, ("mlp", 1, 2), np.flatnonzero(f != 0), np.flatnonzero(f == 0), lams, rl)
        del p
        torch.cuda.empty_cache()
    (run_dir("alt") / "altdev.json").write_text(json.dumps(res, indent=1, default=float))
    rl.close()


def n3dev(noalt: bool = False):
    from .runlog import RunLog
    H.DEV = "cuda" if torch.cuda.is_available() else "cpu"
    rl = RunLog("skill_pack", "raise", "n3dev")
    lams, res, V = _lams(), [], None
    for alt in ((0,) if noalt else (0, 1)):
        p = prep(extra=2, scale=S_ROWS, alt=bool(alt))
        V = _views("train", p["d"]["tokens"])
        X = _with_views(p["X"], V)
        f = _folds_all(p)
        for a_, hid in [g for g in N3_GRID if g[0] == alt]:
            r = _eval_cfg(p, X, ("mlp", 1, 2), np.flatnonzero(f != 0), np.flatnonzero(f == 0), lams, rl, hidden=hid)
            res.append({**r, "alt": alt})
        del p, X
        torch.cuda.empty_cache()
    best = max(res, key=lambda r: r["pdms_hold"])
    out = {"grid": res, "chosen": best}
    p = prep(extra=2, scale=S_ROWS, alt=bool(best["alt"]))
    X = _with_views(p["X"], _views("train", p["d"]["tokens"]))
    f = _folds_all(p)
    out["fold1_chosen"] = _eval_cfg(p, X, ("mlp", 1, 2), np.flatnonzero(f != 1), np.flatnonzero(f == 1), lams, rl,
                                    hidden=best["hidden"])
    (run_dir("n3") / "n3dev.json").write_text(json.dumps(out, indent=1, default=float))
    rl.log.info(f"chosen alt {best['alt']} hidden {best['hidden']}: held-out {best['pdms_hold']:.3f}")
    rl.close()


def n3final():
    from .runlog import RunLog
    H.DEV = "cuda" if torch.cuda.is_available() else "cpu"
    rl = RunLog("skill_pack", "raise", "n3final")
    dev = json.loads((run_dir("n3") / "n3dev.json").read_text())["chosen"]
    alt, hid, w, beta = bool(dev["alt"]), dev["hidden"], tuple(dev["weights"]), dev["beta"]
    p = prep(extra=2, scale=S_ROWS, alt=alt)
    Vtr = _views("train", p["d"]["tokens"])
    X = _with_views(p["X"], Vtr)
    tests = {}
    for split in TEST:
        t = test_inputs(split, p["d"], 2, alt)
        t["X"] = _with_views(p["X"], Vtr, t["X"], _views(split, t["tokens"]))
        tests[split] = t
    Tall = torch.stack([p["T"][m] for m in SUBS], 1)
    names = list(tests)
    runs = [train_mlp(X, Tall, np.arange(len(X)), sd, epoch=e, evals=[tests[sp]["X"] for sp in names], hidden=hid)[0]
            for sd, e in zip(range(SEEDS), dev["epochs"])]
    sel = {"alt": alt, "hidden": hid, "weights": w, "beta": beta, "n_rows": len(X)}
    for j, sp in enumerate(names):
        lg = {m: torch.stack([r[j][m] for r in runs]).mean(0) for m in SUBS}
        t = tests[sp]
        s_ = N1.scores(lg, t["Dm"], w, beta, p["mask"]).argmax(1).cpu().numpy()
        pool = np.concatenate([np.broadcast_to(p["d"]["anchors"][None], (len(s_), K, 8, 3)), t["cands"]], 1)
        np.savez(run_dir("n3") / f"{sp}_n3.npz", tokens=t["tokens"], poses=pool[np.arange(len(s_)), s_].astype(np.float32))
        sel[f"{sp}_shares"] = {"anchor": float((s_ < K).mean()),
                               **{str(x): float((s_ == K + i).mean()) for i, x in enumerate(slots(2, alt))}}
        rl.log.info(f"{sp}: poses written")
    (run_dir("n3") / "select.json").write_text(json.dumps(sel, indent=1, default=float))
    rl.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("fit", "repro", "report", "explore", "tokens2", "pilotcheck", "n2dev", "n2final", "alt_cands", "altdev", "n3dev", "n3final"))
    ap.add_argument("what", nargs="?")
    ap.add_argument("--arm", default="n1b")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--looks", type=int, default=0)
    ap.add_argument("--m", type=int, default=40000)
    ap.add_argument("--noalt", action="store_true")
    a = ap.parse_args()
    {"fit": lambda: fit_arm(a.arm, a.final), "repro": repro, "report": lambda: report(a.arm, looks=a.looks),
     "explore": lambda: explore(a.what), "tokens2": lambda: tokens2(a.m), "pilotcheck": pilotcheck, "n2dev": n2dev, "n2final": n2final, "alt_cands": alt_cands, "altdev": altdev, "n3dev": lambda: n3dev(a.noalt), "n3final": n3final}[a.cmd]()
