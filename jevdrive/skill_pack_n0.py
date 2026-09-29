"""Skill pack N0 (todos/2026-09-29-skill-pack-n0.md, pre-registered before any N0 score): a switch between openpilot
Cinque's native plan (GIMM-VFI context frames, NAVSIM navtest 84.2 PDMS) and E6's Hydra-style scoring head, with
arc-length speed-up candidates of the native plan. CPU only.

  prep     refit E6's scorer on the CPU (sub-score heads on E6's 20 000 tokens at E6's lambdas; cls ego -> cls_late
           imitation on navtrain minus the tuning set T; E6's aggregation weights fixed), build the native candidates
           n_s (s in SCALES) on T and navtest, their nearest-anchor score gaps D_s, and T's five pose files to score
  select   the (mode, delta) grid on T from the devkit per-token CSVs; writes the one navtest pose file
  report   navtest: N0 vs the native plan (84.2 run) and vs E6's stored Hydra run, paired token bootstrap, sub-scores,
           by command

    python -m jevdrive.skill_pack_n0 prep|select|report
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .common import data_dir, get_logger

log = get_logger(__name__)
REPO = Path(__file__).resolve().parents[1]
RUN = "runs/skill_pack/n0"
E6_PREP = "runs/elicitation/e6-prep/20260926-003758"
E6_FIT = "runs/elicitation/e6-fit/20260926-025946"
LANE = "runs/op_lb/lb_navtrain"
NAVFULL = "runs/op_interp/navfull"
NATIVE_TEST = f"{NAVFULL}/preds/gimm_g0.2-cinque__base.npz"
EVAL = "runs/navsim/eval"
SCALES = (1.00, 1.05, 1.10, 1.15)
MODES = (*SCALES, "auto")
DELTAS = (-1.0, 0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, np.inf)
V1 = {"no_at_fault_collisions": "NC", "drivable_area_compliance": "DAC", "ego_progress": "EP",
      "time_to_collision_within_bound": "TTC", "comfort": "C", "score": "PDMS"}
CMD = {0: "left", 1: "straight", 2: "right", 3: "unknown"}


def run_dir() -> Path:
    d = data_dir() / RUN
    d.mkdir(parents=True, exist_ok=True)
    return d


def stretch(p: np.ndarray, s: float) -> np.ndarray:
    """(8, 3) NAVSIM poses (0.5 ... 4 s): the same path with every pose's arc length times s (same time stamps).
    Positions / heading interpolated along the polyline from the origin; beyond the last pose, extrapolated along
    the last segment's direction with the last heading."""
    if s == 1.0:
        return p.copy()
    q = np.vstack([np.zeros(3, p.dtype), p]).astype(np.float64)
    q[:, 2] = np.unwrap(q[:, 2])
    seg = np.linalg.norm(np.diff(q[:, :2], axis=0), axis=1)
    L = np.r_[0, np.cumsum(seg)]
    if L[-1] < 1e-3:
        return p.copy()
    tgt = s * L[1:]
    out = np.stack([np.interp(tgt, L, q[:, k]) for k in range(3)], -1)
    far = tgt > L[-1]
    if far.any():
        j = np.flatnonzero(seg > 1e-6)[-1]
        u = (q[j + 1, :2] - q[j, :2]) / seg[j]
        out[far, :2] = q[-1, :2] + (tgt[far] - L[-1])[:, None] * u
        out[far, 2] = q[-1, 2]
    return out.astype(np.float32)


def nearest_anchor(P: np.ndarray, A: np.ndarray) -> np.ndarray:
    """P (n, 8, 3), A (K, 8, 3): index of the anchor with the smallest max-abs (x, y) difference (E6's check metric)."""
    Pt, At = torch.as_tensor(P[..., :2]), torch.as_tensor(A[..., :2])
    return torch.cat([(c[:, None] - At[None]).abs().amax((2, 3)).argmin(1) for c in Pt.split(2048)]).numpy()


# ---------------------------------------------------------------- prep

def prep():
    from . import elicit_e6 as E6, navsim_heads as H, planner, probe
    H.DEV = planner.DEV = probe.DEV = "cpu"
    torch.set_num_threads(len(os.sched_getaffinity(0)))
    os.environ["OPI_ROOT"] = "op_lb"
    sys.path.insert(0, str(REPO / "scripts"))
    import op_interp as OPI
    from . import navsim_zs as Z
    out = run_dir()
    st = json.loads((data_dir() / E6_FIT / "stats.json").read_text())
    lams, w = tuple(st["cls_lams"]), tuple(st["weights"])
    tr = E6._navtrain(True)
    te = H.load("navtest", True)
    an = np.load(data_dir() / E6_PREP / "anchors.npz")
    anchors, ids = an["anchors"], an["ids"]
    # tuning set T: the lane's navtrain subset minus E6's 20 000 sub-score training tokens
    lane = (data_dir() / LANE / "tokens.txt").read_text().split()
    e6 = set((data_dir() / E6_PREP / "tokens.txt").read_text().split())
    pos = dict(zip(tr["tokens"].tolist(), range(len(tr["tokens"]))))
    T = [t for t in lane if t not in e6 and t in pos]
    log.info(f"T: lane {len(lane)}, minus E6 overlap {sum(t in e6 for t in lane)}, minus not in navtrain features "
             f"{sum(t not in e6 and t not in pos for t in lane)} -> {len(T)}")
    rT = np.array([pos[t] for t in T])
    # sub-score targets of E6's 20 000 tokens
    toks, sub = [], []
    for p in sorted((data_dir() / E6_PREP / "score").glob("chunk_*.npz")):
        z = np.load(p)
        toks.append(z["tokens"]), sub.append(z["sub"])
    toks, sub = np.concatenate(toks), np.concatenate(sub)
    rows = np.array([pos[t] for t in toks])
    assert not set(T) & set(toks.tolist())
    Xe, Xe_te = H._std(tr["ego"], te["ego"])
    Xf, Xf_te = H._std(tr["cinque"], te["cinque"])
    X = torch.cat([Xe, Xf], 1)
    Xev = {"T": X[rT], "navtest": torch.cat([Xe_te, Xf_te], 1)}
    heads = {s: {} for s in Xev}
    for i, m in enumerate(E6.SUBS):
        W, b = E6._bce_fit(X[rows], torch.as_tensor(sub[:, :, i]), np.arange(len(rows)), st["lam"][m])
        for s in Xev:
            heads[s][m] = Xev[s] @ W + b
        log.info(f"sub-score head {m} fitted (lam {st['lam'][m]:g})")
    fit_rows = np.setdiff1d(np.arange(len(tr["tokens"])), rT)
    im = E6._cls_logits(Xe, Xf, ids, tr["log"], fit_rows, {"T": Xe[rT], "navtest": Xe_te},
                        {"T": Xf[rT], "navtest": Xf_te}, lams)
    log.info(f"imitation refitted on {len(fit_rows)} navtrain rows (T excluded)")
    # native plans: T from the lane's plan file (op_interp's base adapter), navtest from the 84.2 run
    mt = OPI.meta("lb_navtrain")
    zp = np.load(data_dir() / LANE / "plans" / "gimm@cinque.npz")
    assert zp["names"].tolist() == mt["names"]
    at = dict(zip(mt["names"], range(len(mt["names"]))))
    nat = {"T": np.stack([OPI.adapt(zp, at[t], mt, Z.T_OUT, **OPI.ADAPTERS["base"])[0] for t in T])}
    zt = np.load(data_dir() / NATIVE_TEST)
    tt = dict(zip(zt["tokens"].tolist(), range(len(zt["tokens"]))))
    nat["navtest"] = zt["poses"][[tt[t] for t in te["tokens"]]]
    keys = {"T": np.array(T), "navtest": te["tokens"]}
    for s in Xev:
        S = _scores(im[s], heads[s], w)
        k = S.argmax(1)
        cand = {sc: np.stack([stretch(p, sc) for p in nat[s]]) for sc in SCALES}
        a = {sc: nearest_anchor(c, anchors) for sc, c in cand.items()}
        ar = torch.arange(len(k))
        D = np.stack([(S[ar, k] - S[ar, torch.as_tensor(a[sc])]).numpy() for sc in SCALES], 1)
        np.savez_compressed(out / f"{s}_cands.npz", tokens=keys[s], k=k.numpy(), D=D, scales=np.array(SCALES),
                            hydra=anchors[k.numpy()], **{f"n{sc:.2f}": c for sc, c in cand.items()})
        log.info(f"{s}: {len(k)} tokens; D (s = 1.00) quantiles 10/50/90 {np.quantile(D[:, 0], [.1, .5, .9]).round(3)}; "
                 f"native nearest anchor == Hydra pick {(D[:, 0] == 0).mean():.3f}")
        if s == "T":
            (out / "T_tokens.txt").write_text("\n".join(T) + "\n")
            for sc in SCALES:
                np.savez(out / f"T_n{sc:.2f}.npz", tokens=keys[s], poses=cand[sc])
            np.savez(out / "T_hydra.npz", tokens=keys[s], poses=anchors[k.numpy()])
        else:
            ref = np.load(data_dir() / E6_FIT / "navtest_hydra_cinque_temporal.npz")
            assert (ref["tokens"] == keys[s]).all()
            same = float((np.abs(ref["poses"] - anchors[k.numpy()]).max((1, 2)) < 1e-4).mean())
            log.info(f"navtest: refitted scorer picks E6's stored anchor on {same:.4f} of tokens")
            (out / "prep_stats.json").write_text(json.dumps({"n_T": len(T), "e6_same_anchor_navtest": same,
                                                             "weights": w, "cls_lams": lams}, indent=1))


def _scores(im: torch.Tensor, heads: dict, w) -> torch.Tensor:
    """E6's weighted log-score sum per anchor (elicit_e6._select without the argmax)."""
    ls = torch.nn.functional.logsigmoid
    w_im, w_mul, w_ttc, w_ep, w_c = w
    return (w_im * torch.log_softmax(im, 1) + w_mul * (ls(heads["NC"]) + ls(heads["DAC"])) + w_ttc * ls(heads["TTC"])
            + w_ep * ls(heads["EP"]) + w_c * ls(heads["C"])).float()


# ---------------------------------------------------------------- select / report

def per_token(ver_split_name: str) -> pd.DataFrame:
    fs = sorted(glob.glob(str(data_dir() / EVAL / ver_split_name / "*" / "*.csv")))
    assert fs, f"no CSV for {ver_split_name}"
    d = pd.read_csv(fs[-1])
    d = d[d["token"].str.fullmatch(r"[0-9a-f]{16,17}") & d["valid"].astype(bool)]
    return d.rename(columns=V1).set_index("token")[list(V1.values())].astype(float)


def choose(D: np.ndarray, mode, delta: float) -> tuple[np.ndarray, np.ndarray]:
    """(use native?, which scale index) per token."""
    j = np.full(len(D), SCALES.index(mode)) if mode != "auto" else D.argmin(1)   # argmin: first (smallest s) on ties
    d = D[np.arange(len(D)), j]
    return d <= delta, j


def select():
    out = run_dir()
    c = np.load(out / "T_cands.npz")
    P = {f"n{sc:.2f}": per_token(f"v1_navtrain_sp_n0_T_n{sc:.2f}") for sc in SCALES}
    P["hydra"] = per_token("v1_navtrain_sp_n0_T_hydra")
    toks = c["tokens"]
    for k in P:
        P[k] = P[k].reindex(toks)
    ok = np.all([~P[k]["PDMS"].isna().to_numpy() for k in P], 0)
    log.info(f"T tokens with all five scores: {ok.sum()} / {len(toks)}")
    Pn = np.stack([P[f"n{sc:.2f}"]["PDMS"].to_numpy() for sc in SCALES], 1)[ok]
    Ph, D = P["hydra"]["PDMS"].to_numpy()[ok], c["D"][ok]
    grid = []
    for m in MODES:
        for dl in DELTAS:
            use, j = choose(D, m, dl)
            v = np.where(use, Pn[np.arange(len(j)), j], Ph)
            grid.append({"mode": str(m), "delta": dl, "pdms": 100 * v.mean(), "native_share": use.mean(),
                         "scaled_share": (use & (np.asarray(SCALES)[j] > 1)).mean()})
    g = pd.DataFrame(grid)
    g.to_csv(out / "T_grid.csv", index=False, float_format="%.4f")
    best = g.iloc[int(g["pdms"].to_numpy().argmax())]          # first maximum = earlier grid cell on ties
    orc = 100 * np.maximum(Pn.max(1), Ph).mean()
    ref = {"native s=1.00": 100 * Pn[:, 0].mean(), "hydra": 100 * Ph.mean(), "oracle": orc,
           **{f"native s={sc:.2f}": 100 * Pn[:, i].mean() for i, sc in enumerate(SCALES)}}
    log.info(f"T references: {json.dumps({k: round(v, 2) for k, v in ref.items()})}")
    log.info(f"T grid best: {best.to_dict()}")
    mode = best["mode"] if best["mode"] == "auto" else float(best["mode"])
    t = np.load(out / "navtest_cands.npz")
    use, j = choose(t["D"], mode, float(best["delta"]))
    nat = np.stack([t[f"n{sc:.2f}"] for sc in SCALES], 1)[np.arange(len(j)), j]
    poses = np.where(use[:, None, None], nat, t["hydra"]).astype(np.float32)
    np.savez(out / "navtest_n0.npz", tokens=t["tokens"], poses=poses)
    (out / "select.json").write_text(json.dumps({"mode": str(best["mode"]), "delta": float(best["delta"]),
                                                 "pdms_T": float(best["pdms"]), "n_T": int(ok.sum()), "refs_T": ref,
                                                 "navtest_native_share": float(use.mean()),
                                                 "navtest_scaled_share": float((use & (np.asarray(SCALES)[j] > 1)).mean())},
                                                indent=1, default=float))


def report():
    out = run_dir()
    n0, nat, hyd = (per_token(f"v1_navtest_{n}") for n in ("sp_n0_navtest", "opi_navfull_gimm_g0.2-cinque__base",
                                                            "e6_hydra_cinque"))
    hum = per_token("v1_navtest_human")
    mt = json.loads((data_dir() / NAVFULL / "meta.json").read_text())
    grp = pd.Series({t: ("start (v0<1)" if s < 1 else CMD[int(c)])
                     for t, c, s in zip(mt["names"], mt["cmd"], mt["speed"])})
    rows, pairs = [], []
    for name, d in (("N0", n0), ("native (84.2 run)", nat), ("E6 Hydra (stored)", hyd), ("human", hum)):
        rows.append({"row": name, "n": len(d), **{k: 100 * d[k].mean() for k in V1.values()}})
    for name, d in (("N0 - native", nat), ("N0 - E6 Hydra", hyd)):
        x, y = n0["PDMS"].align(d["PDMS"], join="inner")
        dd = (x - y).to_numpy()
        bs = dd[np.random.default_rng(1).integers(0, len(dd), (10_000, len(dd)))].mean(1)
        pairs.append({"pair": name, "n": len(dd), "diff": 100 * dd.mean(), "lo": 100 * np.percentile(bs, 2.5),
                      "hi": 100 * np.percentile(bs, 97.5)})
    g = grp.reindex(n0.index)
    bycmd = pd.DataFrame({name: d["PDMS"].reindex(n0.index).groupby(g).mean() * 100
                          for name, d in (("N0", n0), ("native", nat), ("E6 Hydra", hyd))})
    ep_hum = float((n0["EP"] > hum["EP"].reindex(n0.index)).mean())
    res = {"table": pd.DataFrame(rows), "paired": pd.DataFrame(pairs), "by_command": bycmd}
    for k, v in res.items():
        v.to_csv(out / f"report_{k}.csv", float_format="%.3f")
        print(v.round(2).to_markdown())
    print(f"share of tokens with N0 EP > human EP: {ep_hum:.3f}")
    (out / "report.json").write_text(json.dumps({"ep_above_human": ep_hum, "paired": pairs}, indent=1, default=float))


if __name__ == "__main__":
    {"prep": prep, "select": select, "report": report}[sys.argv[1]]()
