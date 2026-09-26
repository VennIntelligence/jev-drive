"""Night queue 3, lane D, Q4a: the M-C paired-difference head with a real-frame zero constraint
(todos/2026-09-26-night-queue-3.md, Q4 and the [D] 16:50 entry, written before any Q4a number).

  pred(x) = prior(x) + Delta(x),  Delta(x) = (z(x) - zbar) W  (reactivity_mc's `pair` arm, dual stream Qwen ⊕ openpilot)
  loss    sum_pair |D W - R|^2 + mu sum_train |Zc W|^2
          + lam_r * n_pair * [1/2 mean_nav |(z - zbar) W|^2 + 1/2 mean_wod |(z - zbar) W|^2] + lam |W|^2
          z standardised with the fold's CARLA training rows (the map E1 uses to carry Delta to real data); the real
          rows are clean navtrain / WOD train frames (no pedestrian / cyclist in the +-4 m, 30 m corridor of the logged
          path, no vehicle in the 1.5-4 m side band). lam_r = 0 reproduces the stored M-C fold predictions.

  prep       clean frame sets, the navtrain 90 / 10 log split, the navtrain Qwen extraction index (split nq3_navtrain),
             the held-out scene filter for the v1.1 devkit
  fit        folds x lam_r in {0, 0.1, 1, 10} per model and seed: P5 v1 BA exam (p5_exam + reactivity_mc.criteria),
             I3 rows riding along (elicit_i3's judge), fold heads, Delta on navtrain held-out / navtest / WOD val
  hydra      night2_n3.fit unchanged with the held-out tokens as an extra evaluation matrix (their Hydra selection)
  hold-jobs  held-out predictions (Hydra, Hydra + Delta_lam) and the devkit job list
  select     held-out PDMS per arm -> lam_select.json (before any navtest score)
  nav-jobs   navtest predictions for the selected lam (main: ungated; g2-gated: description) and the job list
  report     navtest paired PDMS, P5 / WOD / I3 readouts, verdicts, PASS + mc_real0 package, small tables

Run on the box: P5_SET=carla_p5v1_ba python -m jevdrive.nq3_q4a <step>
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .common import data_dir, get_logger

log = get_logger(__name__)
MODELS = ("cinque", "lebowski")
SEEDS = (0, 1, 2)
LAM_R = (0.0, 0.1, 1.0, 10.0)             # 0 = the reproduction check, not a candidate
CAND = (0.1, 1.0, 10.0)
HALF_W, REACH, LEAD_W = 4.0, 30.0, 1.5
N_NAV, HOLD_FRAC, SPLIT_SEED = 6000, 0.10, 0
QSPLIT = "nq3_navtrain"
PED_RATIO, PDMS_DROP = 0.8, -1.0
G2_RUN = "runs/real-data-transfer/g2/20260926-090503"
RESULTS = Path(__file__).resolve().parents[1] / "research" / "results" / "nq3" / "q4"


def out(*p) -> Path:
    d = data_dir() / "runs" / "nq3" / "q4a"
    (d / Path(*p)).parent.mkdir(parents=True, exist_ok=True)
    return d / Path(*p)


def _stamp(msg: str):
    with open(out("timeline.txt"), "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


# ---------------------------------------------------------------- clean real frames

def _corridor_hits(path: np.ndarray, pts: np.ndarray, groups: np.ndarray, n_obj: int):
    """Per object (points grouped by object id): any point in the corridor, any point in the lead band."""
    from .fusion_diag import project
    if not n_obj:
        return np.zeros(0, bool), np.zeros(0, bool)
    s, d, _, _ = project(path, pts)
    inc = (np.abs(d) <= HALF_W) & (s > 0) & (s <= REACH)
    lead = inc & (np.abs(d) <= LEAD_W)
    a, b = np.zeros(n_obj, bool), np.zeros(n_obj, bool)
    np.logical_or.at(a, groups, inc)
    np.logical_or.at(b, groups, lead)
    return a, b


def _clean_row(path_xy: np.ndarray, pts: np.ndarray, groups: np.ndarray, cls: np.ndarray) -> dict:
    """cls per object: 0 vehicle, 1 pedestrian, 2 cyclist, 3 other."""
    from .elicit_e3 import extend
    path = extend(np.r_[[[0.0, 0.0]], path_xy], REACH)
    inc, lead = _corridor_hits(path, pts, groups, len(cls))
    vru = inc & np.isin(cls, (1, 2))
    side = inc & ~lead & (cls == 0)
    return {"vru": bool(vru.any()), "side_vehicle": bool(side.any()), "lead_vehicle": bool((lead & (cls == 0)).any()),
            "clean": not (vru.any() or side.any())}


def _nav_worker(args):
    fut, boxes, cls = args
    if not len(boxes):
        return _clean_row(fut, np.zeros((0, 2)), np.zeros(0, int), np.zeros(0, int))
    x, y, ln, w = boxes[:, 0], boxes[:, 1], boxes[:, 3], boxes[:, 4]
    yaw = boxes[:, 6] if boxes.shape[1] > 6 else np.zeros(len(boxes))
    c, s = np.cos(yaw), np.sin(yaw)
    pts = [np.c_[x, y]] + [np.c_[x + a * ln / 2 * c - b * w / 2 * s, y + a * ln / 2 * s + b * w / 2 * c]
                           for a, b in ((1, 1), (1, -1), (-1, 1), (-1, -1))]
    grp = np.tile(np.arange(len(boxes)), 5)
    # E3 classes: 0 vehicle, 1 pedestrian, 2 bicycle, 3 other
    return _clean_row(fut, np.concatenate(pts), grp, cls.astype(int))


def clean_nav() -> pd.DataFrame:
    from multiprocessing import Pool
    from . import elicit_e3 as E3, navsim_heads as H
    tr = H.load("navtrain", False)
    ag = E3.extract("navtrain")
    ok = ~np.isnan(tr["fut"]).any((1, 2))
    tok, fut = tr["tokens"][ok], tr["fut"][ok]
    jobs = [(fut[i, :, :2], ag[t]["boxes"], ag[t]["cls"]) for i, t in enumerate(tok)]
    with Pool(min(20, os.cpu_count())) as p:
        rows = p.map(_nav_worker, jobs, chunksize=512)
    return pd.DataFrame(rows).assign(token=tok, log=tr["log"][ok])


def _wod_worker(args):
    fut, det = args
    if det is None:
        return _clean_row(fut, np.zeros((0, 2)), np.zeros(0, int), np.zeros(0, int))
    cls = det[:, 2].astype(int)
    return _clean_row(fut, det[:, :2], np.arange(len(det)), cls)


def clean_wod() -> pd.DataFrame:
    from multiprocessing import Pool
    from . import real_g0 as G0, waymo
    fr = pd.read_parquet(G0.root("wod_train", "frames.parquet"))
    d = pd.read_parquet(G0.root("wod_train", "dets.parquet"), columns=["frame_id", "prompt", "score", "gx", "gy", "lift_ok"])
    d = d[d.lift_ok & (d.score >= G0.SCORE) & d.prompt.isin(G0.CLS3)]
    d = d.assign(c=d.prompt.map({"vehicle": 0, "pedestrian": 1, "cyclist": 2}).astype(float))
    by = {k: g[["gx", "gy", "c"]].to_numpy(float) for k, g in d.groupby("frame_id", sort=False)}
    _, future = waymo.load_ego()
    fut = future[fr.row.to_numpy(), :, :2]
    jobs = [(fut[i], by.get(k)) for i, k in enumerate(fr.frame_id)]
    with Pool(min(20, os.cpu_count())) as p:
        rows = p.map(_wod_worker, jobs, chunksize=512)
    return pd.DataFrame(rows).assign(frame_id=fr.frame_id.to_numpy(), sequence=fr.sequence.to_numpy())


def prep(rl):
    import pickle
    from . import navsim_qwen as NQ
    nav = clean_nav()
    logs = np.array(sorted(nav.log.unique()))
    hold_logs = set(np.random.default_rng(SPLIT_SEED).permutation(logs)[:int(round(HOLD_FRAC * len(logs)))])
    nav["hold"] = nav.log.isin(hold_logs)
    sub = set((data_dir() / "runs/elicitation/e6-prep/20260926-003758/tokens.txt").read_text().split())
    nav["e6sub"] = nav.token.isin(sub)
    wod = clean_wod()
    rng = np.random.default_rng(SPLIT_SEED)
    pool = nav.index[nav.clean & ~nav.hold].to_numpy()
    nav["constraint"] = False
    nav.loc[rng.choice(pool, min(N_NAV, len(pool)), replace=False), "constraint"] = True
    nav["heldout_eval"] = nav.hold & nav.e6sub
    nav.to_parquet(out("prep", "nav_frames.parquet"), index=False)
    wod.to_parquet(out("prep", "wod_frames.parquet"), index=False)
    need = nav.token[nav.constraint | nav.heldout_eval].to_numpy()
    with open(data_dir() / "runs/navsim_zs/index/navtrain.pkl", "rb") as f:
        idx = {e["token"]: e for e in pickle.load(f)}
    t = pd.DataFrame({"token": need, "files": [[fr[c]["path"] for c in NQ.CAMS for fr in idx[k]["cams"]] for k in need]})
    assert t.files.map(len).eq(12).all()
    t["chunk"] = np.arange(len(t)) // NQ.CHUNK
    t.to_parquet(NQ.root(QSPLIT, "index.parquet"), index=False)
    # the held-out scene filter for the v1.1 devkit (E6's e6sub.yaml format, v1_e6sub metric cache)
    ho = nav[nav.heldout_eval]
    q = lambda xs: "\n".join(f"    - '{x}'" for x in xs)  # noqa: E731
    sp = out("hydra", "train_test_split", "nq3hold.yaml")
    sp.write_text("data_split: trainval\nscene_filter:\n  _target_: navsim.common.dataclasses.SceneFilter\n  _convert_: 'all'\n"
                  "  num_history_frames: 4\n  num_future_frames: 10\n  frame_interval: 1\n  has_route: true\n  max_scenes: null\n"
                  f"  log_names:\n{q(sorted(set(ho.log)))}\n  tokens:\n{q(ho.token)}\n")
    st = {"nav_frames": len(nav), "nav_clean": int(nav.clean.sum()), "nav_logs": len(logs), "hold_logs": len(hold_logs),
          "nav_constraint": int(nav.constraint.sum()), "nav_heldout_eval": int(nav.heldout_eval.sum()),
          "wod_frames": len(wod), "wod_clean": int(wod.clean.sum()), "qwen_tokens": len(t), "qwen_chunks": int(t.chunk.nunique()),
          "nav_vru_share": float(nav.vru.mean()), "nav_side_share": float(nav.side_vehicle.mean()),
          "wod_vru_share": float(wod.vru.mean()), "wod_side_share": float(wod.side_vehicle.mean())}
    (out("prep", "stats.json")).write_text(json.dumps(st, indent=1))
    rl.event("q4a_prep", **st)
    rl.log.info(json.dumps(st))


# ---------------------------------------------------------------- features of the real rows

def real_rows(model: str) -> dict:
    """Qwen L18_last and op temporal of: navtrain constraint rows, WOD clean rows, navtrain held-out, navtest, WOD val."""
    from . import elicit_e1 as E1, navsim_qwen as NQ
    nav = pd.read_parquet(out("prep", "nav_frames.parquet"))
    wod = pd.read_parquet(out("prep", "wod_frames.parquet"))
    zt = np.load(data_dir() / "runs/navsim_zs/openpilot/navtrain" / f"{model}_temporal.npz")
    at = pd.Series(np.arange(len(zt["tokens"])), index=zt["tokens"])
    T = zt["temporal"]
    res = {}
    for k, msk in (("nav", nav.constraint), ("hold", nav.heldout_eval)):
        tok = nav.token[msk].to_numpy()
        res[k] = {"keys": tok, "Q": NQ.load(QSPLIT, tok)["L18_last"], "O": T[at[tok].to_numpy()].astype(np.float32)}
    # WOD train: clean rows of qwenvid_train_t4 (all G0 wod_train frames have both streams)
    root = data_dir() / E1.WOD_FEAT
    keep = pd.Index(wod.frame_id[wod.clean])
    qs, names = [], []
    for sh in sorted((root / "qwenvid_train_t4").iterdir()):
        if sh.name.startswith("training_") and (sh / "index.parquet").exists():
            nm = pd.read_parquet(sh / "index.parquet").frame_name.to_numpy()
            m = pd.Index(nm).isin(keep)
            if m.any():
                qs.append(np.load(sh / "L18_last.npy", mmap_mode="r")[np.flatnonzero(m)].astype(np.float32))
                names.append(nm[m])
    names = np.concatenate(names)
    oi = pd.read_parquet(root / f"op_{model}_p3_trainval/index.parquet").frame_name
    ops = np.load(root / f"op_{model}_p3_trainval/temporal.npy", mmap_mode="r")[E1._rows(pd.Series(names), oi)].astype(np.float32)
    res["wod"] = {"keys": names, "Q": np.concatenate(qs), "O": ops}
    z = np.load(data_dir() / "runs/navsim_zs/openpilot/navtest" / f"{model}_temporal.npz")
    res["navtest"] = {"keys": z["tokens"], "Q": NQ.load("navtest", z["tokens"])["L18_last"], "O": z["temporal"].astype(np.float32)}
    return res


# ---------------------------------------------------------------- fit

def _solve(D, R, M, lams, dev):
    """W for every lam of min |D W - R|^2 + W^T M W + lam |W|^2 (reactivity_mc._solve_pair with M in place of mu Zc'Zc)."""
    ev, V = torch.linalg.eigh((D.T @ D + M).double().to(dev))
    ev, V = ev.float().to(D.device), V.float().to(D.device)
    B = V.T @ (D.T @ R)
    return [V @ (B / (ev[:, None] + lam)) for lam in lams]


def _zmap(X: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
    return (X - mu.float()) / sd.float() / np.sqrt(X.shape[1])


def fit(rl, models=MODELS, seeds=SEEDS, eigh_dev="cuda"):
    from . import elicit_e1 as E1, elicit_i3 as I, night2_n3 as N3, p5_exam as E, p5_openpilot, p5_pairs as P
    from . import reactivity_mc as MC
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
    dev = "cuda"
    with I.p5_set(I.BA):
        t, past, fut, obs, null, pairs = E.load()
        Q = P.load_features(t, ("L18_last",))["L18_last"]
        op = p5_openpilot.load(t, MODELS, sub="op_streams_vis")
    with I.p5_set(I.I3):
        t3a, past3a, _, obs3, null3, pairs3 = E.load()
        keep = t3a.frame_name.isin(set(I.needed())).to_numpy()
        t3, past3 = t3a[keep].reset_index(drop=True), past3a[keep]
        Q3 = P.load_features(t3, ("L18_last",))["L18_last"]
        op3 = p5_openpilot.load(t3, MODELS, sub="op_streams")
    n, n3 = len(t), len(t3)
    ta = pd.concat([t[["frame_name", "role", "base_id", "intent"]],
                    t3[["frame_name", "base_id", "intent"]].assign(role="i3", base_id="i3:" + t3.base_id)], ignore_index=True)
    F = torch.as_tensor(np.r_[fut.reshape(n, -1), np.zeros((n3, 40), np.float32)], device=dev)
    Ego = torch.as_tensor(np.r_[E.ego_input(t, past), E.ego_input(t3, past3)], device=dev)
    Qa = torch.as_tensor(np.r_[Q, Q3], device=dev)
    pos = pd.Series(np.arange(n), index=t.frame_name)
    pr_ip = np.r_[pos[obs.fn_plus].to_numpy(), pos[null.fn_plus].to_numpy()]
    pr_im = np.r_[pos[obs.fn_minus].to_numpy(), pos[null.fn_null].to_numpy()]
    pr_group = np.r_[obs.base_id.to_numpy(), null.base_id.to_numpy()].astype(str)
    obs_rows = np.flatnonzero(t.role.to_numpy() == "obs")
    I3r = np.arange(n, n + n3)
    wd = E1.wod_frames()
    summary = []
    for m in models:
        real = real_rows(m)
        rl.log.info("%s real rows: nav %d, wod %d, hold %d, navtest %d, wod val %d", m, len(real["nav"]["keys"]),
                    len(real["wod"]["keys"]), len(real["hold"]["keys"]), len(real["navtest"]["keys"]), len(wd["Q"]))
        Rq = {k: torch.as_tensor(real[k]["Q"], device=dev) for k in ("nav", "wod")}
        Ro = {k: torch.as_tensor(real[k]["O"], device=dev) for k in ("nav", "wod")}
        Xop = torch.as_tensor(np.r_[op[f"op-{m} temporal"], op3[f"op-{m} temporal"]], device=dev)
        for s in seeds:
            fold = E.folds(t, pairs, s)
            fold_a = np.r_[fold, np.full(n3, -2)]
            ref_run = data_dir() / N3.MC_RUNS[s]
            ref = np.load(ref_run / "preds_obs.npz")
            ref_at = pd.Series(np.arange(len(ref["rows"])), index=ref["rows"])
            ref_lam = {e["fold"]: e["lam"] for e in map(json.loads, open(ref_run / "events.jsonl"))
                       if e.get("kind") == "mc_fold" and e.get("arm") == "pair" and e.get("model") == m}
            p5 = {lr: np.full((n, 20, 2), np.nan, np.float32) for lr in LAM_R}
            p5["prior"] = np.full((n, 20, 2), np.nan, np.float32)
            i3 = {lr: np.zeros((n3, 20, 2)) for lr in (*LAM_R, "prior")}
            heads = {lr: [] for lr in LAM_R}
            for f in range(E.K_FOLDS):
                tr = np.flatnonzero((ta.role.to_numpy() == "train") & (fold_a != f))
                ev = obs_rows[fold[obs_rows] == f]
                prior = MC.fit_fold(f, fold_a, ta, F, Ego, Xop, Qa, pr_ip, pr_im, pr_group, rl, m, arms=())["prior"]
                keepp = fold[pr_ip] != f
                ip, im, grp = pr_ip[keepp], pr_im[keepp], pr_group[keepp]
                Rp = (F[ip] - F[im]) - (prior[ip] - prior[im])
                inner = MC._inner_splits(grp)
                sq, so = E1._stats(Qa, tr), E1._stats(Xop, tr)
                Z = torch.cat([_zmap(Qa, *sq), _zmap(Xop, *so)], 1)
                zbar = Z[tr].mean(0)
                Zc = Z[tr] - zbar
                mu = len(ip) / len(tr)
                D = Z[ip] - Z[im]
                Rg = 0
                for k in ("nav", "wod"):
                    Zr = torch.cat([_zmap(Rq[k], *sq), _zmap(Ro[k], *so)], 1) - zbar
                    Rg = Rg + (Zr.T @ Zr) * (0.5 / len(Zr))
                    del Zr
                base = mu * (Zc.T @ Zc)
                for lr in LAM_R:
                    M = base + (lr * len(ip)) * Rg
                    score = np.zeros(len(MC.LAMS))
                    for a, b in inner:
                        Ws = _solve(D[a], Rp[a], M, MC.LAMS, eigh_dev)
                        score += [float(((D[b] @ W - Rp[b]) ** 2).sum()) for W in Ws]
                    best = int(np.argmin(score))
                    W = _solve(D, Rp, M, [MC.LAMS[best]], eigh_dev)[0]
                    pred = prior + (Z - zbar) @ W
                    p5[lr][ev] = pred[ev].reshape(-1, 20, 2).cpu().numpy()
                    i3[lr] += pred[I3r].reshape(-1, 20, 2).cpu().double().numpy() / E.K_FOLDS
                    heads[lr].append({"W": W.double().cpu(), "zbar": zbar.double().cpu(), "lam": float(MC.LAMS[best]),
                                      "q": tuple(x.cpu() for x in sq), "op": tuple(x.cpu() for x in so)})
                    rl.event("q4a_fold", model=m, seed=s, fold=f, lam_r=lr, lam=float(MC.LAMS[best]),
                             lam_edge=best in (0, len(MC.LAMS) - 1), n_pair=len(ip), n_train=len(tr))
                    if lr == 0.0:
                        diff = float(np.abs(p5[0.0][ev] - ref[f"M-C pair [{m}]"][ref_at[ev].to_numpy()]).max())
                        rl.event("q4a_repro", model=m, seed=s, fold=f, lam=float(MC.LAMS[best]), lam_ref=ref_lam.get(f), max_abs_diff=diff)
                        rl.log.info("%s s%d fold %d lam_r 0: lam %g (stored %s), max |pred - stored| %.2e m", m, s, f,
                                    MC.LAMS[best], ref_lam.get(f), diff)
                        assert np.isclose(MC.LAMS[best], ref_lam[f]) and diff < 1e-3, "lam_r = 0 does not reproduce M-C"
                p5["prior"][ev] = prior[ev].reshape(-1, 20, 2).cpu().numpy()
                i3["prior"] += prior[I3r].reshape(-1, 20, 2).cpu().double().numpy() / E.K_FOLDS
                del Z, Zc, D, Rg, base
                torch.cuda.empty_cache()
            # P5 exam: p5_exam + reactivity_mc.criteria unchanged
            key = lambda lr: f"M-C pair [{m}]" if lr == 0.0 else f"M-C real{lr:g} [{m}]"  # noqa: E731
            preds = {f"prior [{m}]": p5["prior"], **{key(lr): p5[lr] for lr in LAM_R}}
            with I.p5_set(I.BA):
                oo, nn = E.deltas(obs, null, t, preds)
                res = E.exam(oo, nn, pairs, list(preds))
            crit = MC.criteria(res, list(preds), f"prior [{m}]").assign(model=m, seed=s)
            fl = res["flips"].assign(model=m, seed=s)
            # I3: elicit_i3's judge (no TFv6 columns), fold-mean predictions
            pi3 = {f"ridge_late op-{m} temporal": i3["prior"].astype(np.float32),
                   **{key(lr): i3[lr].astype(np.float32) for lr in LAM_R}}
            tf6, E.TFV6 = E.TFV6, {}
            with I.p5_set(I.I3):
                o3, n3_ = E.deltas(obs3, null3, t3, pi3)
                r3 = E.exam(o3, n3_, pairs3, list(pi3))
            E.TFV6 = tf6
            fl3 = r3["flips"].assign(model=m, seed=s)
            d = out("fit", f"{m}_s{s}")
            d.mkdir(parents=True, exist_ok=True)
            crit.to_csv(d / "p5_criteria.csv", index=False)
            fl.to_csv(d / "p5_flip_rates.csv", index=False)
            fl3.to_csv(d / "i3_flip_rates.csv", index=False)
            np.savez_compressed(d / "p5_preds_obs.npz", rows=obs_rows, **{k: v[obs_rows] for k, v in preds.items()})
            # Delta on the real evaluation sets, per lam_r (mean of the fold heads, E1.correction)
            for lr in LAM_R:
                h = heads[lr]
                torch.save(h, d / f"heads_lam{lr:g}.pt")
                dl = {"hold": E1.correction(h, real["hold"]["Q"], real["hold"]["O"]),
                      "navtest": E1.correction(h, real["navtest"]["Q"], real["navtest"]["O"]),
                      "wod": E1.correction(h, wd["Q"], wd[f"op {m}"])}
                np.savez_compressed(d / f"delta_lam{lr:g}.npz", hold_keys=real["hold"]["keys"], hold=dl["hold"],
                                    navtest_keys=real["navtest"]["keys"], navtest=dl["navtest"],
                                    wod_keys=wd["frame_name"], wod=dl["wod"])
                mag = {k: float(np.median(np.linalg.norm(v, axis=-1).mean(-1))) for k, v in dl.items()}
                c = crit.set_index("arm").loc[key(lr)]
                summary.append({"model": m, "seed": s, "lam_r": lr, "ped_flip": float(c.ped_flip), "ped_lo": float(c.ped_lo),
                                "null_ff": float(c.null_ff_oos), "cutin_flip": float(c.cutin_flip),
                                **{f"delta_med_{k}": v for k, v in mag.items()}})
                rl.event("q4a_summary", **summary[-1])
            rl.log.info("%s seed %d\n%s", m, s, crit.to_markdown(index=False, floatfmt=".3f"))
        del Xop, Rq, Ro
        torch.cuda.empty_cache()
    pd.DataFrame(summary).to_csv(out("fit", "summary.csv"), index=False)
    _stamp("fit done")


# ---------------------------------------------------------------- Hydra on the held-out tokens

def hydra(rl, models=MODELS, seeds=SEEDS):
    """night2_n3.fit unchanged, with the held-out navtrain tokens as its extra evaluation matrix."""
    from . import navsim_heads as H, night2_n3 as N3
    nav = pd.read_parquet(out("prep", "nav_frames.parquet"))
    tok = nav.token[nav.heldout_eval].to_numpy()
    tr = H.load("navtrain", True)
    at = pd.Series(np.arange(len(tr["tokens"])), index=tr["tokens"])[tok].to_numpy()
    orig = N3.extra_sets
    try:
        for m in models:
            N3.extra_sets = lambda model, m=m: {"hold": (tr["ego"][at], tr[m][at], tok)}
            for s in seeds:
                N3.fit(rl, m, s)
                z = np.load(rl.dir / f"sel_{m}_s{s}.npz", allow_pickle=True)
                refs = N3._sel_files()
                ref = np.load(refs[(m, s)], allow_pickle=True)
                same = float((z["navtest_hydra"] == ref["navtest_hydra"]).mean())
                rl.event("q4a_hydra_check", model=m, seed=s, navtest_same_anchor=same)
                rl.log.info("%s s%d: refit Hydra navtest selection = stored on %.4f of tokens", m, s, same)
                assert same >= 0.99, "Hydra refit does not reproduce the stored N3 selection"
                np.savez(out("hydra", f"hold_{m}_s{s}.npz"), tokens=tok, poses=z["anchors"][z["hold_hydra"]])
    finally:
        N3.extra_sets = orig
    _stamp("hydra done")


# ---------------------------------------------------------------- devkit jobs

def _add(poses: np.ndarray, delta: np.ndarray, gate=None) -> np.ndarray:
    a = poses.copy()
    g = np.ones(len(a), np.float32) if gate is None else gate
    a[..., :2] += g[:, None, None] * delta[:, 1:16:2]          # 0.5 ... 4.0 s of the 0.25 s grid; heading kept
    return a.astype(np.float32)


def hold_jobs(rl, models=MODELS, seeds=SEEDS):
    jobs = []
    for m in models:
        for s in seeds:
            hz = np.load(out("hydra", f"hold_{m}_s{s}.npz"))
            tok = hz["tokens"]
            arms = {f"hydra_{m}_s{s}": hz["poses"]}
            for lr in CAND:
                dz = np.load(out("fit", f"{m}_s{s}", f"delta_lam{lr:g}.npz"))
                assert (dz["hold_keys"] == tok).all()
                arms[f"hydra_real{lr:g}_{m}_s{s}"] = _add(hz["poses"], dz["hold"])
            for name, p in arms.items():
                path = out("hold", f"{name}.npz")
                np.savez(path, tokens=tok, poses=p.astype(np.float32))
                jobs.append(f"nq3hold {name} {path}")
    out("hold", "jobs.txt").write_text("\n".join(jobs) + "\n")
    rl.log.info("%d held-out jobs", len(jobs))


def _scores(split: str, name: str) -> pd.Series | None:
    from .openloop_standing import _latest
    df = _latest("v1", split, name)
    if df is None:
        return None
    df = df[df["token"].str.fullmatch(r"[0-9a-f]{16,17}") & df["valid"].astype(bool)]
    return df.set_index("token")["score"].astype(float)


def select(rl, models=MODELS, seeds=SEEDS):
    """lam* per model x seed = argmax held-out PDMS of Hydra + Delta_lam; written before any navtest score."""
    assert not out("lam_select.json").exists(), "lam_select.json exists: selection is written once"
    rows, sel = [], {}
    for m in models:
        for s in seeds:
            base = _scores("nq3hold", f"hydra_{m}_s{s}")
            r = {"model": m, "seed": s, "n": len(base), "hydra": 100 * base.mean()}
            for lr in CAND:
                v = _scores("nq3hold", f"hydra_real{lr:g}_{m}_s{s}")
                x, y = v.align(base, join="inner")
                r[f"lam{lr:g}"] = 100 * x.mean()
                r[f"lam{lr:g}_delta"] = 100 * (x - y).mean()
            best = max(CAND, key=lambda lr: r[f"lam{lr:g}"])
            r["lam_star"] = best
            sel[f"{m}_s{s}"] = best
            rows.append(r)
    tab = pd.DataFrame(rows)
    tab.to_csv(out("hold_pdms.csv"), index=False)
    out("lam_select.json").write_text(json.dumps({"written": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "lam_star": sel,
                                                   "rule": "argmax held-out v1.1 PDMS of Hydra_s + Delta_lam"}, indent=1))
    _stamp(f"lam_select written: {sel}")
    rl.log.info("held-out PDMS\n%s", tab.to_markdown(index=False, floatfmt=".2f"))
    # equivalence of this scoring path: Hydra's devkit score vs E6's per-anchor PDMS of the same anchor (seed 0)
    for m in models:
        _check_hydra_hold(rl, m)


def _check_hydra_hold(rl, m: str):
    from . import night2_n3 as N3
    P = N3.prep_dir(0)
    toks, pdms = [], []
    for p in sorted((P / "score").glob("chunk_*.npz")):
        z = np.load(p)
        toks.append(z["tokens"]), pdms.append(z["pdms"])
    at = pd.Series(np.arange(sum(map(len, toks))), index=np.concatenate(toks))
    pdms = np.concatenate(pdms)
    hz = np.load(out("hydra", f"hold_{m}_s0.npz"))
    an = np.load(P / "anchors.npz")["anchors"]
    k = np.abs(hz["poses"][:, None] - an[None]).max((2, 3)).argmin(1)
    ref = pd.Series(pdms[at[hz["tokens"]].to_numpy(), k], index=hz["tokens"])
    dk = _scores("nq3hold", f"hydra_{m}_s0")
    x, y = dk.align(ref, join="inner")
    d = np.abs(x - y)
    rl.event("q4a_score_check", model=m, n=len(d), max_abs_diff=float(d.max()), share_equal=float((d < 1e-9).mean()))
    rl.log.info("%s: devkit held-out score vs E6 per-anchor PDMS: n %d, max |diff| %.2e, equal share %.4f", m, len(d),
                d.max(), (d < 1e-9).mean())


def nav_jobs(rl, models=MODELS, seeds=SEEDS):
    from . import night2_n3 as N3
    sel = json.loads(out("lam_select.json").read_text())["lam_star"]
    g2 = np.load(data_dir() / G2_RUN / "g2_nav_eval.npz", allow_pickle=True)
    jobs = []
    for m in models:
        for s in seeds:
            lr = sel[f"{m}_s{s}"]
            hy = np.load(N3._sel_files()[(m, s)].parent / f"navtest_hydra_{m}_s{s}.npz")
            tok = hy["tokens"]
            dz = np.load(out("fit", f"{m}_s{s}", f"delta_lam{lr:g}.npz"))
            assert (dz["navtest_keys"] == tok).all()
            g = pd.Series(g2["gate"], index=g2["frame_id"].astype(str)).reindex(tok.astype(str)).to_numpy().astype(np.float32)
            for name, p in ((f"nq3_hydra_real_{m}_s{s}", _add(hy["poses"], dz["navtest"])),
                            (f"nq3_hydra_g2real_{m}_s{s}", _add(hy["poses"], dz["navtest"], g))):
                path = out("navtest", f"{name}.npz")
                np.savez(path, tokens=tok, poses=p)
                jobs.append(f"navtest {name} {path}")
    out("navtest", "jobs.txt").write_text("\n".join(jobs) + "\n")
    _stamp("navtest jobs written")


# ---------------------------------------------------------------- report

def _boot(d: np.ndarray, b: int = 10000, seed: int = 0) -> tuple[float, float]:
    bs = d[np.random.default_rng(seed).integers(0, len(d), (b, len(d)))].mean(1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def _rfs_cluster(d: dict, prior: np.ndarray, delta: np.ndarray, b: int = 10000) -> dict:
    """RFS of prior + Delta minus prior on the rater frames: frame mean and cluster mean, cluster-stratified bootstrap."""
    from . import waymo
    pos = d["rater_pos"]
    rfs = lambda p: waymo.rater_feedback_score(p[pos], d["rater_traj"], d["rater_score"], d["speed"][pos])  # noqa: E731
    dd = rfs(prior + delta) - rfs(prior)
    cl = d["cluster"][pos]
    u = np.unique(cl)
    idx = [np.flatnonzero(cl == c) for c in u]
    rng = np.random.default_rng(0)
    cm = lambda v: np.mean([v[i].mean() for i in idx])  # noqa: E731
    bc, bf = [], []
    for _ in range(b):
        sel = [i[rng.integers(0, len(i), len(i))] for i in idx]
        bc.append(np.mean([dd[s_].mean() for s_ in sel]))
        bf.append(dd[np.concatenate(sel)].mean())
    return {"n": len(dd), "rfs_cluster_delta": float(cm(dd)), "rfs_cluster_lo": float(np.percentile(bc, 2.5)),
            "rfs_cluster_hi": float(np.percentile(bc, 97.5)), "rfs_frame_delta": float(dd.mean()),
            "rfs_frame_lo": float(np.percentile(bf, 2.5)), "rfs_frame_hi": float(np.percentile(bf, 97.5))}


def report(rl, models=MODELS, seeds=SEEDS):
    from . import elicit_e1 as E1, night2_n3 as N3
    from .real_g1 import mc_taus
    sel = json.loads(out("lam_select.json").read_text())["lam_star"]
    sc = pd.read_csv(data_dir() / N3.E1_NAV / "navtest_scopes.csv").set_index("token")
    wd = E1.wod_frames()
    taus = mc_taus()
    rows, wod_rows = [], []
    for m in models:
        for s in seeds:
            lr = sel[f"{m}_s{s}"]
            base = _scores("navtest", f"n3_hydra_{m}_s{s}")
            r = {"model": m, "seed": s, "lam_star": lr, "hydra_pdms": 100 * base.mean()}
            for tag in ("real", "g2real"):
                v = _scores("navtest", f"nq3_hydra_{tag}_{m}_s{s}")
                x, y = v.align(base, join="inner")
                dd = (x - y).to_numpy()
                lo, hi = _boot(dd)
                r[f"{tag}_n"], r[f"{tag}_pdms"], r[f"{tag}_delta"], r[f"{tag}_lo"], r[f"{tag}_hi"] = len(dd), 100 * x.mean(), 100 * dd.mean(), 100 * lo, 100 * hi
                for grp in ("ped_cyc_corridor", "straight"):
                    k = sc[grp].reindex(x.index).fillna(False).to_numpy(bool)
                    r[f"{tag}_delta_{grp}"] = 100 * dd[k].mean()
            crit = pd.read_csv(out("fit", f"{m}_s{s}", "p5_criteria.csv")).set_index("arm")
            mc = crit.loc[f"M-C pair [{m}]"]
            c = crit.loc[f"M-C real{lr:g} [{m}]"]
            r.update(mc_ped=float(mc.ped_flip), ped=float(c.ped_flip), ped_lo=float(c.ped_lo), ped_hi=float(c.ped_hi),
                     cutin=float(c.cutin_flip), null_ff=float(c.null_ff_oos))
            fl3 = pd.read_csv(out("fit", f"{m}_s{s}", "i3_flip_rates.csv"))
            p3 = fl3[fl3.scope == "pooled"].set_index("examinee")
            r["i3_flip"] = float(p3.loc[f"M-C real{lr:g} [{m}]"].flip_rate)
            r["i3_flip_mc"] = float(p3.loc[f"M-C pair [{m}]"].flip_rate)
            r["i3_flip_prior"] = float(p3.loc[f"ridge_late op-{m} temporal"].flip_rate)
            r["nav_ok"] = r["real_delta"] >= PDMS_DROP
            r["p5_ok"] = r["ped"] >= PED_RATIO * r["mc_ped"]
            r["package"] = bool(r["nav_ok"] and r["p5_ok"])
            rows.append(r)
            dz = np.load(out("fit", f"{m}_s{s}", f"delta_lam{lr:g}.npz"))
            assert (dz["wod_keys"] == wd["frame_name"]).all()
            prior = wd[f"prior {m}"]
            wr = {"model": m, "seed": s, "lam_star": lr, **_rfs_cluster(wd, prior, dz["wod"])}
            tab, act = E1.readouts(wd, prior, dz["wod"], taus[m])
            a = act.set_index("scope")
            wr["act_straight"], wr["act_ped"] = float(a.loc["straight_yaw", "activation"]), float(a.loc["Pedestrians", "activation"])
            g = tab.set_index(["scope", "judge"])
            wr["rfs_ped_delta"] = float(g.loc[("Pedestrians", "RFS (rater)")].delta)
            wr["delta_med_m"] = float(a.loc["all", "delta_mag_median_m"])
            wod_rows.append(wr)
    T, W = pd.DataFrame(rows), pd.DataFrame(wod_rows)
    verdict = []
    for m in models:
        v = T[T.model == m].package.tolist()
        cell = "能力包成立" if all(v) else ("能力包不成立" if not any(v) else "随 seed 变")
        verdict.append({"model": m, "per_seed": v, "verdict": cell})
    V = pd.DataFrame(verdict)
    RESULTS.mkdir(parents=True, exist_ok=True)
    for dst in (out(), RESULTS):
        T.to_csv(dst / "q4a_main.csv", index=False, float_format="%.4f")
        W.to_csv(dst / "q4a_wod.csv", index=False, float_format="%.4f")
        V.to_csv(dst / "q4a_verdict.csv", index=False)
    for f in ("hold_pdms.csv", "lam_select.json"):
        (RESULTS / f"q4a_{f}").write_text(out(f).read_text())
    pd.read_csv(out("fit", "summary.csv")).to_csv(RESULTS / "q4a_fit_summary.csv", index=False, float_format="%.4f")
    rl.log.info("main\n%s\nWOD\n%s\nverdict\n%s", T.to_markdown(index=False, floatfmt=".3f"), W.to_markdown(index=False, floatfmt=".3f"),
                V.to_markdown(index=False))
    # hand-off to the closed loop: Cinque seed 0 at its lam*
    cin = V.set_index("model").loc["cinque", "verdict"] if "cinque" in models else ""
    pkg = out("mc_real0")
    pkg.mkdir(exist_ok=True)
    lr = sel["cinque_s0"]
    src = out("fit", "cinque_s0", f"heads_lam{lr:g}.pt")
    (pkg / "heads.pt").write_bytes(src.read_bytes())
    (pkg / "README.json").write_text(json.dumps({
        "what": "Q4a M-C (dual stream Qwen L18_last + openpilot Cinque temporal) with the real-frame zero constraint, seed 0",
        "lam_r": lr, "format": "list of 5 fold dicts as elicit_e1.fold_heads: W (3072, 40) float64, zbar, q=(mu, sd) Qwen, "
        "op=(mu, sd) Cinque temporal, lam; Delta(x) = mean over folds of (z(x) - zbar) @ W, see elicit_e1.correction; "
        "prior = the same P5-fitted Cinque ridge_late as M-C (CL4)", "verdict_cinque": cin}, indent=1))
    if cin == "能力包成立":
        out("PASS").write_text(json.dumps({"written": time.strftime("%Y-%m-%d %H:%M:%S"), "package": str(pkg)}) + "\n")
    _stamp(f"report done; cinque verdict {cin}")


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("prep", "fit", "hydra", "hold-jobs", "select", "nav-jobs", "report"))
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    a = ap.parse_args()
    models, seeds = tuple(a.models.split(",")), tuple(int(x) for x in a.seeds.split(","))
    rl = RunLog("nq3", f"q4a-{a.step}")
    fn = {"prep": lambda: prep(rl), "fit": lambda: fit(rl, models, seeds), "hydra": lambda: hydra(rl, models, seeds),
          "hold-jobs": lambda: hold_jobs(rl, models, seeds), "select": lambda: select(rl, models, seeds),
          "nav-jobs": lambda: nav_jobs(rl, models, seeds), "report": lambda: report(rl, models, seeds)}[a.step]
    fn()
    rl.close()


if __name__ == "__main__":
    main()
