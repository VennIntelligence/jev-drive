"""Elicitation E2, training and readouts on the navtrain edit pairs (todos/2026-09-26-elicitation-program.md,
deviation-log entry [E2] 00:57, written before any fit).

  data     valid pairs of jevdrive.elicit_e2 (x+, x-, placebo clips), Qwen `L18_last` (navsim_qwen) and openpilot
           `temporal` (navsim_zs_openpilot feat) per side; the prior is navtrain's `ridge ego` + `ridge_late <model>`
           refit with navsim_heads' recipe; everything on the 16-point 0.25 s grid (0.25 ... 4 s)
  labels   (a) y+ = logged future, y- = CTRA continuation; (b) the three Q6 gates on GT states: brake - continue when
           the gate fires on x+ only; (c) PDM scorer brake vs continue on x+ (optional, when its scores exist)
  arms     pair (dual / qwen / op), uniform imitation, hard-example reweighting; lambda grid of M-C, inner 3-fold by log
  read     R1 |Delta(x+) - Delta(x-)| vs placebo on the E1 head and on the trained heads; R2 P5 v1 BA exam;
           R3 WOD (E1's frames, prior and readouts)

Run on the box: P5_SET=carla_p5v1_ba python -m jevdrive.elicit_e2_train run
"""
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch

from . import elicit_e1 as E1, elicit_e2 as E2, navsim_heads as NH, navsim_qwen as NQ, navsim_zs as Z
from . import p5_exam as E, p5_pairs as P, reactivity_mc as MC, traj
from .common import data_dir, get_logger

log = get_logger(__name__)
MODELS = ("cinque", "lebowski")
T16 = np.arange(1, 17) * 0.25
DECEL = 3.0
LABELS = ("a", "b", "c")


# ---------------------------------------------------------------- trajectories on the 16-point grid

def grid16(p8: np.ndarray) -> np.ndarray:
    """(n, 8, >=2) poses at 0.5 ... 4 s -> (n, 16, 2) (x, y) at 0.25 ... 4 s, linear, through the origin."""
    xy = np.concatenate([np.zeros((len(p8), 1, 2), np.float32), p8[..., :2]], 1)
    t = np.r_[0, Z.T_OUT]
    return np.stack([np.stack([np.interp(T16, t, xy[i, :, j]) for j in range(2)], -1) for i in range(len(p8))]).astype(np.float32)


def extend20(d16: np.ndarray) -> np.ndarray:
    """(n, 16, 2) -> (n, 20, 2): points 17-20 continue the 15 -> 16 step (deviation [E2] 00:57 (2))."""
    step = d16[:, 15] - d16[:, 14]
    return np.concatenate([d16, d16[:, 15:16] + step[:, None] * np.arange(1, 5)[None, :, None]], 1)


def ctra(e: dict, decel: float | None = None) -> np.ndarray:
    """(16, 2) CTRA from a slim index entry: t0 speed, t0 longitudinal acceleration, yaw rate of the last 0.5 s;
    speed clipped at 0. decel: instead of the acceleration, brake at this constant rate to a stop."""
    v0 = float(np.linalg.norm(e["vel"][-1]))
    a0 = -decel if decel is not None else float(e["acc"][-1][0])
    dyaw = e["pose"][-1, 2] - e["pose"][-2, 2]
    w = float(np.arctan2(np.sin(dyaw), np.cos(dyaw)) / 0.5)
    dt = 0.01
    ts = np.arange(0, 4.0 + dt / 2, dt)
    v = np.maximum(v0 + a0 * ts, 0.0)
    th = w * ts
    x = np.r_[0, np.cumsum(v[:-1] * np.cos(th[:-1]) * dt)]
    y = np.r_[0, np.cumsum(v[:-1] * np.sin(th[:-1]) * dt)]
    return np.stack([np.interp(T16, ts, x), np.interp(T16, ts, y)], -1).astype(np.float32)


def cv_path(e: dict) -> np.ndarray:
    """(16, 2) the same arc at constant t0 speed (the "continue" proposal of label (b) / (c))."""
    return ctra({**e, "acc": np.zeros_like(e["acc"])})


# ---------------------------------------------------------------- label (b): Q6's three gates on GT states

def _gates(anns, drop: set, path: np.ndarray, v0: float) -> bool:
    """Any of Q6's gates on the t0 GT agents (minus the tracks in `drop`): TTC < 3 s in the corridor ahead, incursion
    into the 3 s path +-1.2 m, a pedestrian closing on the corridor at > 0.5 m/s within 4 m of it and 30 m of ego."""
    seg = np.diff(path, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    s_cum = np.r_[0, np.cumsum(seglen)]
    s3 = v0 * 3.0 + 2.0
    for nm, bx, vel, tt in zip(anns["gt_names"], anns["gt_boxes"], anns["gt_velocity_3d"], np.asarray(anns["track_tokens"]).astype(str)):
        if tt in drop or nm not in ("vehicle", "pedestrian", "bicycle"):
            continue
        p = np.asarray(bx[:2], float)
        a, ab = path[:-1], seg
        tpar = np.clip(((p - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0, 1)
        foot = a + tpar[:, None] * ab
        dist = np.linalg.norm(p - foot, axis=1)
        k = int(np.argmin(dist))
        lat, s = float(dist[k]), float(s_cum[k] + tpar[k] * seglen[k])
        tang = ab[k] / max(seglen[k], 1e-9)
        if s <= 0 or np.hypot(*p) > 60:
            continue
        if lat <= 1.2:
            closing = v0 - float(np.dot(vel[:2], tang))
            gap = s - bx[3] / 2
            if closing > 0 and gap / closing < 3.0:
                return True
            if s <= s3:
                return True
        if nm == "pedestrian" and np.hypot(*p) < 30 and lat - 1.2 < 4.0:
            normal = (foot[k] - p) / max(lat, 1e-9)
            if float(np.dot(vel[:2], normal)) > 0.5:
                return True
    return False


def _rule_log(args):
    log_path, rows = args
    from .fusion_q2b import extend
    frames = pickle.load(open(log_path, "rb"))
    at = {f["token"]: i for i, f in enumerate(frames)}
    out = {}
    for tok, fut8, drop, v0 in rows:
        a = frames[at[tok]]["anns"]
        path = extend(np.vstack([[0.0, 0.0], fut8[:, :2]]), 60.0)
        out[tok] = (_gates(a, set(), path, v0), _gates(a, drop, path, v0))
    return out


def rule_gates(pairs: pd.DataFrame, fut8: dict, slim: dict, workers: int = 8) -> pd.DataFrame:
    from multiprocessing import Pool
    cands = {c["token"]: c for c in pickle.load(open(E2.out_root("navtrain") / "candidates.pkl", "rb"))}
    logs = data_dir() / "datasets" / "navsim" / "navsim_logs" / "trainval"
    jobs = {}
    for tok, lg in zip(pairs.token, pairs.log):
        c = cands[tok]
        drop = {x[0] for im in c["images"] if im["k"] == 3 for x in im["actors"]} | {c["track"]}
        v0 = float(np.linalg.norm(slim[tok]["vel"][-1]))
        jobs.setdefault(lg, []).append((tok, fut8[tok], drop, v0))
    res = {}
    with Pool(workers) as pool:
        for r in pool.imap_unordered(_rule_log, [(logs / f"{lg}.pkl", rows) for lg, rows in jobs.items()]):
            res.update(r)
    return pd.DataFrame([{"token": t, "gate_plus": a, "gate_minus": b} for t, (a, b) in res.items()])


# ---------------------------------------------------------------- data

def load_data(rl) -> dict:
    tag = "main"
    pairs = pd.read_parquet(E2.out_root("navtrain", tag) / "pairs.parquet")
    slim = {e["token"]: e for e in Z.load_index("navtrain", slim=True)}
    with np.load(Z.root("index") / "navtrain_future.npz") as f:
        fut8 = dict(zip(f["tokens"].tolist(), f["poses"]))
    toks = pairs.token.to_numpy()
    pl_toks = pairs.token[pairs.n_placebo_imgs > 0].to_numpy()
    d = {"pairs": pairs, "tokens": toks, "pl_tokens": pl_toks, "log": pairs.log.to_numpy()}
    for sd, tk in (("plus", toks), ("minus", toks), ("placebo", pl_toks)):
        name = f"e2nav_{sd}"
        d[f"Q {sd}"] = NQ.load(name, tk)["L18_last"]
        for m in MODELS:
            with np.load(Z.root("openpilot", name) / f"{m}_temporal.npz") as z:
                at = dict(zip(z["tokens"].tolist(), range(len(z["tokens"]))))
                d[f"op {m} {sd}"] = z["temporal"][[at[t] for t in tk]].astype(np.float32)
    # labels
    y_plus = grid16(np.stack([fut8[t] for t in toks]))
    y_ctra = np.stack([ctra(slim[t]) for t in toks])
    brake = np.stack([ctra(slim[t], decel=DECEL) for t in toks])
    cont = np.stack([cv_path(slim[t]) for t in toks])
    g = rule_gates(pairs, fut8, slim).set_index("token").loc[toks]
    fire = (g.gate_plus & ~g.gate_minus).to_numpy()
    d["dy"] = {"a": y_plus - y_ctra, "b": np.where(fire[:, None, None], brake - cont, 0.0).astype(np.float32)}
    d["y_single"] = {"a": (y_plus, y_ctra)}
    d["gates"] = g.reset_index()
    sc = E2.out_root("navtrain", tag) / "pdm_scores.csv"            # label (c), when the scorer ran
    if sc.exists():
        s = pd.read_csv(sc).set_index("token").loc[toks]
        d["dy"]["c"] = np.where((s.brake > s["continue"]).to_numpy()[:, None, None], brake - cont, 0.0).astype(np.float32)
    rl.info(f"pairs {len(toks)}, placebo {len(pl_toks)}, rule label fires on {int(fire.sum())} pairs "
            f"(gate x+ {int(g.gate_plus.sum())}, x- {int(g.gate_minus.sum())}), labels {sorted(d['dy'])}")
    # prior: navtrain ridge ego + ridge_late <model>, full-navtrain W (navsim_heads recipe)
    tr = NH.load("navtrain", True)
    keep = (tr["stage"] == "one") & ~np.isnan(tr["fut"]).any((1, 2))
    tr = {k: v[keep] for k, v in tr.items()}
    folds = NH._group_folds(tr["log"], NH.FOLDS)
    Y = torch.as_tensor(tr["fut"].reshape(len(tr["fut"]), -1), device=NH.DEV)
    ego_pairs = NH.ego_features([slim[t] for t in toks])
    Xe, Xe_p = NH._std(tr["ego"], ego_pairs)
    pe, oof_e, _ = NH.fit_ridge(Xe, Y, {"p": Xe_p}, folds)
    base = pe["p"].reshape(-1, 8, 3).cpu().numpy()
    tpos = dict(zip(tr["tokens"].tolist(), range(len(tr["tokens"]))))
    d["s_ego"] = np.linalg.norm(base[:, :, :2] - np.stack([fut8[t] for t in toks])[:, :, :2], axis=-1).mean(1)
    for m in MODELS:
        sides = {sd: d[f"op {m} {sd}"] for sd in ("plus", "minus", "placebo")}
        Xf, *Xs = NH._std(tr[m], *sides.values())
        r, _, st = NH.fit_ridge(Xf, Y - oof_e, dict(zip(sides, Xs)), folds)
        for sd in sides:
            b = base if sd != "placebo" else base[[list(toks).index(t) for t in pl_toks]]
            d[f"prior {m} {sd}"] = grid16(b + r[sd].reshape(-1, 8, 3).cpu().numpy())
        rl.info(f"prior ridge_late {m}: lam {st['lam']}; x+ temporal of the pairs vs navtrain's stored row: "
                f"max |diff| {float(np.abs(tr[m][[tpos[t] for t in toks]] - d[f'op {m} plus']).max()):.3g}")
        del Xf, Xs
    torch.cuda.empty_cache()
    return d


# ---------------------------------------------------------------- fits

def _z(parts_train: list, parts: list):
    """Per-stream standardisation with the training rows' statistics, scaled 1 / sqrt(d); returns z and the stats."""
    stats = []
    for Xt in parts_train:
        mu, sd = Xt.mean(0), Xt.std(0)
        stats.append((mu, np.where(sd > 1e-6, sd, 1.0)))
    z = [np.concatenate([(X - mu) / sd / np.sqrt(X.shape[1]) for X, (mu, sd) in zip(p, stats)], 1) for p in parts]
    return z, stats


def fit_arms(d: dict, m: str, rl) -> dict:
    """Every arm's head for model m: {arm: {"W", "b", "stats", "streams", "lam"}} (Delta = z W + b, 32 outputs)."""
    toks, lg = d["tokens"], d["log"]
    ipl = np.array([list(toks).index(t) for t in d["pl_tokens"]])
    streams_all = {"dual": ("Q", f"op {m}"), "qwen": ("Q",), "op": (f"op {m}",)}
    heads = {}
    for sname, streams in streams_all.items():
        tr_parts = [np.concatenate([d[f"{s} plus"], d[f"{s} minus"]]) for s in streams]
        (zp, zm, zpl), stats = _z(tr_parts, [[d[f"{s} {sd}"] for s in streams] for sd in ("plus", "minus", "placebo")])
        for lab, dy in d["dy"].items():
            if sname != "dual" and lab != "a":
                continue
            pp, pm, ppl = d[f"prior {m} plus"], d[f"prior {m} minus"], d[f"prior {m} placebo"]
            R = (dy - (pp - pm)).reshape(len(toks), -1)
            R0 = (-(pp[ipl] - ppl)).reshape(len(ipl), -1)       # placebo = null pair: y+ = y_placebo
            D = torch.as_tensor(np.concatenate([zp - zm, zp[ipl] - zpl]), dtype=torch.float32)
            Rt = torch.as_tensor(np.concatenate([R, R0]), dtype=torch.float32)
            grp = np.r_[lg, lg[ipl]]
            Zc = torch.zeros(1, D.shape[1])
            score = np.zeros(len(MC.LAMS))
            for a, b in MC._inner_splits(grp):
                Ws = MC._solve_pair(D[a], Rt[a], Zc, 0.0, MC.LAMS)
                score += [float(((D[b] @ W - Rt[b]) ** 2).sum()) for W in Ws]
            best = int(np.argmin(score))
            W = MC._solve_pair(D, Rt, Zc, 0.0, [MC.LAMS[best]])[0]
            arm = f"E2 pair{'' if sname == 'dual' else ' ' + sname} ({lab})"
            heads[arm] = {"W": W.numpy(), "b": 0.0, "stats": stats, "streams": streams, "lam": float(MC.LAMS[best])}
            rl.event("e2_fit", model=m, arm=arm, lam=float(MC.LAMS[best]), edge=best in (0, len(MC.LAMS) - 1),
                     n_pairs=len(toks), n_null=len(ipl))
        if sname == "dual":                                   # single-frame controls on the same z and prior
            y_p, y_m = d["y_single"]["a"]
            Zs = torch.as_tensor(np.concatenate([zp, zm]), dtype=torch.float32)
            Ys = torch.as_tensor(np.concatenate([y_p - d[f"prior {m} plus"], y_m - d[f"prior {m} minus"]]).reshape(2 * len(toks), -1),
                                 dtype=torch.float32)
            grp = np.r_[lg, lg]
            s_ego = np.r_[d["s_ego"], d["s_ego"]]
            for arm, w in (("E2 hard (a)", s_ego / s_ego.mean()), ("E2 uniform (a)", np.ones(len(grp)))):
                w = torch.as_tensor(w, dtype=torch.float32)
                score = np.zeros(len(MC.LAMS))
                for a, b in MC._inner_splits(grp):
                    fits = MC._solve_weighted(Zs[a], Ys[a], w[a], MC.LAMS)
                    score += [float((w[b] * ((Zs[b] @ W + c - Ys[b]) ** 2).sum(1)).sum()) for W, c in fits]
                best = int(np.argmin(score))
                W, c = MC._solve_weighted(Zs, Ys, w, [MC.LAMS[best]])[0]
                heads[arm] = {"W": W.numpy(), "b": c.numpy(), "stats": stats, "streams": streams, "lam": float(MC.LAMS[best])}
                rl.event("e2_fit", model=m, arm=arm, lam=float(MC.LAMS[best]), edge=best in (0, len(MC.LAMS) - 1))
    return heads


def apply(h: dict, feats: dict) -> np.ndarray:
    """Delta (n, 20, 2) of a fitted head on foreign rows; feats = {"Q": ..., "op <m>": ...}."""
    z = np.concatenate([(feats[s] - mu) / sd / np.sqrt(feats[s].shape[1]) for s, (mu, sd) in zip(h["streams"], h["stats"])], 1)
    return extend20((z @ h["W"] + h["b"]).reshape(-1, 16, 2).astype(np.float32))


# ---------------------------------------------------------------- readouts

def _mag(a, b):
    return np.linalg.norm(a - b, axis=-1).mean(-1)


def _ratio_ci(e, p, ge, gp, b=2000, seed=0):
    """median(e) / median(p) with a log bootstrap (edit and placebo pairs resampled by log, jointly)."""
    rng = np.random.default_rng(seed)
    logs = np.unique(np.r_[ge, gp])
    ie = {g: np.flatnonzero(ge == g) for g in logs}
    ip = {g: np.flatnonzero(gp == g) for g in logs}
    rs = []
    for _ in range(b):
        pick = rng.choice(logs, len(logs))
        xe = np.concatenate([e[ie[g]] for g in pick])
        xp = np.concatenate([p[ip[g]] for g in pick])
        if len(xe) and len(xp) and np.median(xp) > 0:
            rs.append(np.median(xe) / np.median(xp))
    return float(np.median(e) / np.median(p)), float(np.quantile(rs, 0.025)), float(np.quantile(rs, 0.975))


def r1(d, deltas: dict) -> pd.DataFrame:
    """|Delta(x+) - Delta(x-)| on edit pairs vs |Delta(x+) - Delta(placebo)| on placebo pairs, per head."""
    ipl = np.array([list(d["tokens"]).index(t) for t in d["pl_tokens"]])
    rows = []
    na = d["pairs"].n_actors.to_numpy()
    for name, (dp, dm, dpl) in deltas.items():
        e, p = _mag(dp, dm), _mag(dp[ipl], dpl)
        for grp, m in (("all", np.ones(len(e), bool)), ("1-3 actors", na <= 3), (">=4 actors", na >= 4)):
            r, lo, hi = _ratio_ci(e[m], p, d["log"][m], d["log"][ipl])
            rows.append({"head": name, "pairs": grp, "n_edit": int(m.sum()), "n_placebo": len(p),
                         "median_edit_m": float(np.median(e[m])), "median_placebo_m": float(np.median(p)),
                         "ratio": r, "lo": lo, "hi": hi, "pass_2x": r >= 2})
    return pd.DataFrame(rows)


def feature_floor(d) -> pd.DataFrame:
    """Gate (d): per stream, |z(x+) - z(x-)| on edit pairs vs |z(x+) - z(placebo)| on placebo pairs (each stream
    standardised over the x+ rows, per-dim RMS), median ratio with a log bootstrap."""
    ipl = np.array([list(d["tokens"]).index(t) for t in d["pl_tokens"]])
    rows = []
    for s in ("Q",) + tuple(f"op {m}" for m in MODELS):
        X = d[f"{s} plus"]
        sd = np.where(X.std(0) > 1e-6, X.std(0), 1.0)
        sh = lambda a, b: np.sqrt((((a - b) / sd) ** 2).mean(1))  # noqa: E731
        e, p = sh(X, d[f"{s} minus"]), sh(X[ipl], d[f"{s} placebo"])
        r, lo, hi = _ratio_ci(e, p, d["log"], d["log"][ipl])
        rows.append({"stream": s, "median_edit": float(np.median(e)), "median_placebo": float(np.median(p)),
                     "ratio": r, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def run(rl):
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 12)))
    d = load_data(rl)
    out = rl.dir
    d["gates"].to_csv(out / "rule_gates.csv", index=False)
    ff = feature_floor(d)
    ff.to_csv(out / "feature_floor.csv", index=False)
    rl.info("feature shift, edit vs placebo\n" + ff.to_markdown(index=False, floatfmt=".3f"))
    # R1 on the E1 head
    r1_rows, heads_all = {}, {}
    for m in MODELS:
        e1h = E1.fold_heads(m, rl)
        sides = [E1.correction(e1h, d[f"Q {sd}"], d[f"op {m} {sd}"]) for sd in ("plus", "minus", "placebo")]
        r1_rows[f"E1 M-C pair [{m}]"] = sides
        heads_all[m] = fit_arms(d, m, rl)
        for arm, h in heads_all[m].items():
            r1_rows[f"{arm} [{m}]"] = [apply(h, {s: d[f"{s} {sd}"] for s in h["streams"]}) for sd in ("plus", "minus", "placebo")]
    t1 = r1(d, r1_rows)
    t1.to_csv(out / "r1_magnitude.csv", index=False)
    rl.info("R1\n" + t1.to_markdown(index=False, floatfmt=".3f"))
    pickle.dump(heads_all, open(out / "heads.pkl", "wb"))
    # R2: P5 v1 BA exam
    t, past, fut, obs, null, pairs = E.load()
    ref = np.load(data_dir() / E1.MC_RUN / "preds_obs.npz")
    rows = ref["rows"]
    Qp = P.load_features(t, ("L18_last",))["L18_last"][rows]
    from . import p5_openpilot
    n = len(t)
    preds = {}
    for m in MODELS:
        Op = p5_openpilot.load(t, (m,), sub="op_streams_vis")[f"op-{m} temporal"][rows]
        prior = ref[f"prior [{m}]"]
        preds[f"prior [{m}]"] = np.full((n, 20, 2), np.nan, np.float32)
        preds[f"prior [{m}]"][rows] = prior
        for arm, h in heads_all[m].items():
            v = np.full((n, 20, 2), np.nan, np.float32)
            v[rows] = prior + apply(h, {s: (Qp if s == "Q" else Op) for s in h["streams"]})
            preds[f"{arm} [{m}]"] = v
    oo, nn = E.deltas(obs, null, t, preds)
    res = E.exam(oo, nn, pairs, list(preds))
    crit = pd.concat([MC.criteria(res, [k for k in preds if k.endswith(f"[{m}]")], f"prior [{m}]") for m in MODELS])
    res["flips"].to_csv(out / "p5_flip_rates.csv", index=False)
    crit.to_csv(out / "p5_criteria.csv", index=False)
    rl.info("R2 P5 v1 BA\n" + crit.to_markdown(index=False, floatfmt=".3f"))
    # R3: WOD
    w = E1.wod_frames()
    tabs, acts = [], []
    for m in MODELS:
        for arm, h in heads_all[m].items():
            delta = apply(h, {s: (w["Q"] if s == "Q" else w[f"op {m}"]) for s in h["streams"]})
            tau = res["taus"][f"{arm} [{m}]"]
            tab, act = E1.readouts(w, w[f"prior {m}"], delta, tau)
            tabs.append(tab.assign(model=m, arm=arm))
            acts.append(act.assign(model=m, arm=arm, tau=tau))
    pd.concat(tabs).to_csv(out / "wod_deltas.csv", index=False)
    pd.concat(acts).to_csv(out / "wod_activation.csv", index=False)
    rl.info("R3 WOD done")


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run",))
    ap.parse_args()
    rl = RunLog("elicitation", "e2-train")
    run(rl)
    rl.close()


if __name__ == "__main__":
    main()
