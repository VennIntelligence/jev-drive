"""Real-data transfer G1: a gate trained on real data times the reaction correction elicited on CARLA pairs
(todos/2026-09-26-real-data-transfer.md, G1; deviation-log entries [G1], each written before the numbers it affects).

  gates   g1  decision 23 (e)'s gated residual head (waymo_ladder.GatedResidual, mlp trunk) on [ego, op `temporal`],
              trained on WOD train (NAVSIM: navtrain) to correct `ridge ego`; only the gate branch is kept
          g2  logistic probe on the detection embedding, Platt-calibrated (registered after G0's hand-off)
          g3  openpilot's own lead output, no training: sigmoid(lead_prob[0]) * clip((6 - TTC) / 4, 0, 1)
  arm     prior + g(x) * Delta(x), per frame; Delta is E1's M-C correction (or G0's student) exactly as stored
  judges  unchanged: elicit_e1.readouts on WOD, p5_exam.exam on I3, the official devkit on NAVSIM

    python -m jevdrive.real_g1 g1-wod | g1-nav | i3 | wod | nav-write | nav-table | figs
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .common import data_dir, get_logger

log = get_logger(__name__)
MODELS = ("cinque", "lebowski")
OUT = "runs/real-data-transfer"
E1_WOD = "runs/elicitation/e1-wod/20260926-003138"
E1_NAV = "runs/elicitation/e1-navsim/20260926-020742"
I3_EXAM = "runs/elicitation/i3-exam/20260926-012841"
MC_RUN = "runs/reactivity/mc-carla_p5v1_ba/20260925-233126"
WOD_LEAD = "processed/drive_backbones/op_lead_g1eval"
TTC_ON, TTC_OFF, CLOSING = 2.0, 6.0, 0.1          # g3's mapping, fixed in the [G1] 08:40 entry
ACT_HARM = 0.07


def rd(*p) -> Path:
    return data_dir() / OUT / Path(*p)


# ================================================================ gates

class Gate:
    """A trained gate branch: z-score with the gate's own training statistics, then sigmoid(gate(encode(x)))."""

    def __init__(self, net, mu, sd):
        self.net, self.mu, self.sd = net.eval(), mu, sd

    @torch.inference_mode()
    def __call__(self, X: np.ndarray) -> np.ndarray:
        dev = next(self.net.parameters()).device
        out = []
        for c in torch.as_tensor(np.asarray(X, np.float32)).split(8192):
            z = self.net.encode((c.to(dev) - self.mu.to(dev)) / self.sd.to(dev))
            out.append(torch.sigmoid(self.net.gate(z)).squeeze(-1).float().cpu())
        return torch.cat(out).numpy()

    def save(self, path, **meta):
        torch.save({"state": self.net.state_dict(), "mu": self.mu.cpu(), "sd": self.sd.cpu(), **meta}, path)

    @classmethod
    def load(cls, path, dev="cuda" if torch.cuda.is_available() else "cpu"):
        from .waymo_ladder import GatedResidual
        ck = torch.load(path, map_location="cpu")
        net = GatedResidual("mlp", ck["d"], ck["out"], ck["l1"])
        net.load_state_dict(ck["state"])
        return cls(net.to(dev), ck["mu"], ck["sd"])


def _std_stats(X: torch.Tensor, rows) -> tuple[torch.Tensor, torch.Tensor]:
    """planner.standardize's statistics (float64 accumulation, sd <= 1e-6 -> 1), kept for foreign rows."""
    r = torch.as_tensor(rows, device=X.device)
    mu = X[r].double().mean(0)
    sd = X[r].double().std(0, correction=0)
    return mu.float(), torch.where(sd > 1e-6, sd, torch.ones_like(sd)).float()


def fit_gated(X: np.ndarray, R: torch.Tensor, sp, pre: np.ndarray | None, rl, tag: str, seed: int = 0,
              l1s=(0.0, 1e-3, 1e-2)) -> tuple:
    """waymo_ladder.gated_arm's recipe with the fitted net kept: L1 over (0, 1e-3, 1e-2) chosen on the inner split's
    pre-onset ADE (overall inner ADE where there is no pre-onset subset), `_train` unchanged."""
    from . import waymo_ladder as L
    Xt = torch.as_tensor(np.asarray(X, np.float32), device=L.DEV)
    mu, sd = _std_stats(Xt, sp.train)
    Z = (Xt - mu) / sd
    del Xt
    T = R.shape[1] // 2
    res_fut = R.reshape(-1, T, 2).cpu().numpy()
    best = None
    for l1 in l1s:
        nets = []

        def make(l1=l1):
            nets.append(L.GatedResidual("mlp", Z.shape[1], R.shape[1], l1))
            return nets[-1]
        _, st = L._train(make, lambda net, b: net(Z[b]), sp, R, res_fut, seed, pre=pre)
        score = st["sel_pre_ade"] if np.isfinite(st["sel_pre_ade"]) else st["sel_ade"]
        g = st.pop("gate")
        rl.event("g1_fit", tag=tag, l1=l1, score=score, sel_ade=st["sel_ade"], epochs=st["epochs"],
                 gate_mean_val=float(g.mean()), gate_open_val=float((g > 0.5).mean()))
        log.info("%s l1=%.0e: inner score %.4f (overall %.4f), %d epochs, mean gate on val %.3f", tag, l1, score,
                 st["sel_ade"], st["epochs"], g.mean())
        if best is None or score < best[0]:
            best = (score, l1, nets[-1], st)
    _, l1, net, st = best
    del Z
    torch.cuda.empty_cache()
    return Gate(net, mu, sd), {"l1": l1, "d": int(X.shape[1]), "out": int(R.shape[1]), **{k: v for k, v in st.items()
                                                                                        if k != "gate"}}


def wod_train_context():
    """The rows of decision 40 (iii)'s prior: `waymo_ladder.train_context` (P0's join), train fit / val eval."""
    from . import drive_backbones as D, waymo_ladder as L
    return L.train_context(p0_run=D.P0_RUN)


def g1_wod(rl, models=MODELS, oof_rows: str = "", main_run: str = "", seed: int = 0):
    """g1 on WOD train: target = residual of `ridge ego` fitted on the same train rows; input [ego, op temporal].

    oof_rows (a file of WOD frame names, the selection rows): instead of the main fit, 5 sequence-grouped folds over
    the train sequences, each refitting `ridge ego` and the gate (L1 fixed to the main fit's, from `main_run`) without
    that fold, and writing the gate on the fold's selection rows -- the out-of-fold values the AUC selection reads."""
    from . import planner, waymo_ladder as L, waymo_stage_a as sa
    ctx = wod_train_context()
    want = set(Path(oof_rows).read_text().split()) if oof_rows else None
    for m in models:
        a = L.align(ctx, f"op_{m}_p3_trainval", ["temporal"])
        sel = np.flatnonzero(a["covered"])
        seq, h, n = ctx["seq"][sel], ctx["half"][sel], len(sel)
        X = np.concatenate([ctx["ego"][sel], a["temporal"][sel].astype(np.float32)], 1)
        F = torch.as_tensor(ctx["fut"][sel].reshape(n, -1), device=L.DEV)
        E = torch.as_tensor(ctx["ego"][sel], device=L.DEV)
        pre = ctx["sub"]["pre_onset"][sel]

        def fit(fit_mask, eval_mask, tag, l1s):
            sp = sa.Halves(ctx["df"], seq, fit_mask, eval_mask, seed)
            Xe = planner.standardize(E, sp.train)
            _, st_ego, W = sa.ridge_cv(Xe, F, sp, ctx["fut"][sel])
            R = F - planner.linear_apply(W, Xe, np.arange(n))[0]
            del Xe
            log.info("%s: %d fit / %d eval rows, d = %d, ridge ego %s", tag, len(sp.train), len(sp.val), X.shape[1], st_ego)
            gate, st = fit_gated(X, R, sp, pre, rl, tag, seed, l1s)
            del R
            torch.cuda.empty_cache()
            return gate, st, sp

        if want is None:
            gate, st, sp = fit(h == 0, h == 1, f"g1-wod-{m}", (0.0, 1e-3, 1e-2))
            gate.save(rl.dir / f"g1_wod_{m}.pt", **{k: v for k, v in st.items() if k in ("l1", "d", "out")})
            np.savez_compressed(rl.dir / f"g1_wod_{m}_val.npz", frame_name=ctx["fname"][sel][sp.val], gate=gate(X[sp.val]))
            rl.event("g1_saved", tag=f"g1_wod_{m}", **{k: v for k, v in st.items() if isinstance(v, (int, float))})
            continue
        l1 = float(torch.load(data_dir() / main_run / f"g1_wod_{m}.pt", map_location="cpu")["l1"])
        tr_seq = np.unique(seq[h == 0])
        fold_of = dict(zip(np.random.default_rng(seed).permutation(tr_seq), np.arange(len(tr_seq)) % 5))
        fold = np.array([fold_of.get(q, -1) for q in seq])
        target = (h == 0) & pd.Series(ctx["fname"][sel]).isin(want).to_numpy()
        names, gates = [], []
        for f in range(5):
            gate, st, sp = fit((h == 0) & (fold != f), target & (fold == f), f"g1-wod-{m}-oof{f}", (l1,))
            names.append(ctx["fname"][sel][sp.val])
            gates.append(gate(X[sp.val]))
        np.savez_compressed(rl.dir / f"g1_wod_{m}_oof.npz", frame_name=np.concatenate(names), gate=np.concatenate(gates), l1=l1)
        rl.event("g1_oof_saved", model=m, rows=int(sum(map(len, names))), l1=l1)


def g1_nav(rl, models=MODELS, seed: int = 0):
    """g1 on navtrain (stage one): input [32-d ego, op temporal], target = `ridge ego`'s out-of-fold residual (x, y);
    early stopping and L1 on a log-grouped 20% inner split. navtest rows ride along as the `val` rows."""
    from types import SimpleNamespace
    from . import navsim_heads as NH
    tr = NH.load("navtrain", True)
    keep = (tr["stage"] == "one") & ~np.isnan(tr["fut"]).any((1, 2))
    tr = {k: v[keep] for k, v in tr.items()}
    te = NH.load("navtest", True)
    folds = NH._group_folds(tr["log"], NH.FOLDS, seed=seed)
    inner = NH._group_folds(tr["log"], 5, seed=seed + 1) == 0
    Y = torch.as_tensor(tr["fut"].reshape(len(tr["fut"]), -1), device=NH.DEV)
    Xe, _ = NH._std(tr["ego"], te["ego"])
    _, oof, st_ego = NH.fit_ridge(Xe, Y, {}, folds)
    R8 = (Y - oof).reshape(-1, 8, 3)[..., :2].reshape(len(Y), -1)
    del Xe
    ntr, nte = len(Y), len(te["tokens"])
    R = torch.cat([R8, torch.zeros(nte, R8.shape[1], device=R8.device)])
    sp = SimpleNamespace(train=np.arange(ntr), fit=np.flatnonzero(~inner), sel=np.flatnonzero(inner),
                         val=np.arange(ntr, ntr + nte))
    for m in models:
        X = np.concatenate([np.r_[tr["ego"], te["ego"]], np.r_[tr[m], te[m]]], 1).astype(np.float32)
        log.info("g1 NAVSIM %s: %d navtrain / %d navtest rows, d = %d, ridge ego %s", m, ntr, nte, X.shape[1], st_ego)
        gate, st = fit_gated(X, R, sp, None, rl, f"g1-nav-{m}", seed)
        gate.save(rl.dir / f"g1_nav_{m}.pt", **{k: v for k, v in st.items() if k in ("l1", "d", "out")})
        np.savez_compressed(rl.dir / f"g1_nav_{m}_navtest.npz", tokens=te["tokens"], gate=gate(X[sp.val]))
        np.savez_compressed(rl.dir / f"g1_nav_{m}_navtrain.npz", tokens=tr["tokens"], gate=gate(X[sp.train]))
        rl.event("g1_saved", tag=f"g1_nav_{m}", **{k: v for k, v in st.items() if isinstance(v, (int, float))})
        # out-of-fold gate on navtrain (the AUC selection's rows): the same log folds, L1 fixed to the main fit's;
        # the target stays `ridge ego`'s out-of-fold residual, whose fold of a row never saw that row
        oof = np.zeros(ntr, np.float32)
        for f in range(NH.FOLDS):
            tr_f = np.flatnonzero(folds != f)
            inn = NH._group_folds(tr["log"][tr_f], 5, seed=seed + 1) == 0
            sp_f = SimpleNamespace(train=tr_f, fit=tr_f[~inn], sel=tr_f[inn], val=np.flatnonzero(folds == f))
            g_f, _ = fit_gated(X, R, sp_f, None, rl, f"g1-nav-{m}-oof{f}", seed, (st["l1"],))
            oof[sp_f.val] = g_f(X[sp_f.val])
        np.savez_compressed(rl.dir / f"g1_nav_{m}_oof.npz", tokens=tr["tokens"], gate=oof, l1=st["l1"])


# ---------------------------------------------------------------- g3

def lead_decode(lead: np.ndarray, lead_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """fusion_q4c.outputs' decoding: (x, v) of selection 0 at t = 0 and P(lead)."""
    mu = np.asarray(lead, np.float64)[:, :72].reshape(-1, 3, 6, 4)
    p = 1 / (1 + np.exp(-np.clip(np.asarray(lead_prob, np.float64)[:, 0], -11, None)))
    return mu[:, 0, 0, 0], mu[:, 0, 0, 2], p


def g3(lead: np.ndarray, lead_prob: np.ndarray, v_ego: np.ndarray) -> np.ndarray:
    """P(lead) * clip((6 - TTC) / (6 - 2), 0, 1); TTC = x / (v_ego - v_lead) when closing faster than 0.1 m/s, else inf."""
    x, v, p = lead_decode(lead, lead_prob)
    close = v_ego - v
    ttc = np.where(close > CLOSING, np.maximum(x, 0) / np.maximum(close, CLOSING), np.inf)
    return (p * np.clip((TTC_OFF - ttc) / (TTC_OFF - TTC_ON), 0, 1)).astype(np.float32)


def wod_lead(model: str, names: np.ndarray) -> dict:
    """The re-run lead outputs (and `temporal`, for the equivalence check) on the given WOD frames."""
    parts = [np.load(f) for f in sorted((data_dir() / WOD_LEAD / model).glob("*.npz")) if ".tmp" not in f.name]
    nm = np.concatenate([p["name"] for p in parts]).astype(str)
    at = pd.Series(np.arange(len(nm)), index=nm).reindex(names)
    assert at.notna().all(), f"{int(at.isna().sum())} frames without a lead output"
    i = at.astype(int).to_numpy()
    return {k: np.concatenate([p[k] for p in parts])[i] for k in ("temporal", "lead", "lead_prob")}


# ---------------------------------------------------------------- g2

def _wod_label_worker(args):
    """Per frame: (ped / cyclist in the logged-path corridor, nearest in-corridor vehicle range or inf)."""
    from . import elicit_e3 as E3
    from .fusion_diag import project
    out = []
    for fut, pts, cls in args:
        ped, rv = False, np.inf
        if len(pts):
            s, d, _, _ = project(E3.extend(np.r_[[[0.0, 0.0]], fut]), pts)
            inn = (np.abs(d) <= E3.HALF_W) & (s > 0) & (s <= E3.REACH)
            ped = bool((inn & (cls != "vehicle")).any())
            v = inn & (cls == "vehicle")
            if v.any():
                rv = float(np.linalg.norm(pts[v], axis=1).min())
        out.append((ped, rv))
    return out


def wod_labels(name: str) -> pd.DataFrame:
    """[G1] 08:50 (2) on a G0 WOD set: YOLO detections (score > 0.25, lifted) on the logged 5 s path extended to 30 m
    (+-1.5 m); pedestrian / cyclist, or the nearest in-corridor vehicle 0.2 m closer than 0.4 s before (frame - 4)."""
    from multiprocessing import Pool
    from . import real_g0 as G0, waymo as W
    from .common import n_cpus
    fr = pd.read_parquet(G0.root(name, "frames.parquet"))
    d = pd.read_parquet(G0.root(name, "dets.parquet"), columns=["frame_id", "prompt", "score", "gx", "gy", "lift_ok"])
    d = d[d.lift_ok & (d.score > G0.SCORE)]
    _, future = W.load_ego()
    at = pd.Series(np.arange(len(fr)), index=fr.frame_id.to_numpy())
    fi = at.reindex(d.frame_id).to_numpy().astype(int)
    by = pd.Series(np.arange(len(d))).groupby(fi).indices
    pts, cls = d[["gx", "gy"]].to_numpy(float), d.prompt.to_numpy()
    fut = future[fr.row.to_numpy(), :, :2].astype(np.float64)
    jobs = [(fut[i], pts[by[i]] if i in by else np.zeros((0, 2)), cls[by[i]] if i in by else np.zeros(0, object))
            for i in range(len(fr))]
    with Pool(min(48, n_cpus())) as p:
        res = sum(p.map(_wod_label_worker, [jobs[i:i + 512] for i in range(0, len(jobs), 512)]), [])
    ped = np.array([r[0] for r in res])
    rv = pd.Series([r[1] for r in res], index=fr.frame_id.to_numpy())
    seq = fr.frame_id.str.rsplit("-", n=1).str[0]
    num = fr.frame_id.str.rsplit("-", n=1).str[1].astype(int)
    prev = rv.reindex((seq + "-" + (num - 4).map("{:03d}".format)).to_numpy()).to_numpy()
    has_prev = ~np.isnan(prev)
    approach = has_prev & np.isfinite(rv.to_numpy()) & (np.nan_to_num(prev, nan=np.inf) - rv.to_numpy() >= 0.2)
    lab = pd.DataFrame({"frame_id": fr.frame_id, "sequence": seq, "ped_cyc": ped, "vehicle_approach": approach,
                        "has_prev": has_prev})
    lab["label"] = lab.ped_cyc | lab.vehicle_approach
    log.info("WOD %s labels: %d frames, positive %.3f (ped/cyc %.3f, approaching vehicle %.3f), with a previous frame %.3f",
             name, len(lab), lab.label.mean(), lab.ped_cyc.mean(), lab.vehicle_approach.mean(), has_prev.mean())
    return lab


def nav_labels(split: str, tokens: np.ndarray) -> pd.DataFrame:
    """GT: elicit_e3.cause_flags_nav's in_pedestrian | in_bicycle | cause_vehicle on the logged-path corridor."""
    from . import elicit_e3 as E3
    with np.load(data_dir() / "runs/navsim_zs/index" / f"{split}_future.npz") as f:
        pos = dict(zip(f["tokens"].tolist(), range(len(f["tokens"]))))
        fut = f["poses"][[pos[t] for t in tokens]].astype(np.float32)
    fl = E3.cause_flags_nav(tokens, fut[:, :, :2], E3.extract(split))
    lab = pd.DataFrame({"frame_id": tokens, "label": (fl.in_pedestrian | fl.in_bicycle | fl.cause_vehicle).to_numpy(),
                        "ped_cyc": (fl.in_pedestrian | fl.in_bicycle).to_numpy(), "vehicle_approach": fl.cause_vehicle.to_numpy()})
    log.info("NAVSIM %s labels: %d tokens, positive %.3f (ped/cyc %.3f, approaching vehicle %.3f)", split, len(lab),
             lab.label.mean(), lab.ped_cyc.mean(), lab.vehicle_approach.mean())
    return lab


def _folds(groups: np.ndarray, k: int = 5, seed: int = 0) -> np.ndarray:
    u = np.unique(groups)
    f = dict(zip(np.random.default_rng(seed).permutation(u), np.arange(len(u)) % k))
    return np.array([f[g] for g in groups])


class Probe:
    """[G1] 08:50 (3): standardised embedding -> L2 logistic (C by grouped OOF log-loss) -> Platt on the OOF logits."""

    CS = (0.01, 0.1, 1.0, 10.0)

    def fit(self, X, y, groups, rl=None, tag=""):
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import log_loss
        self.mu, sd = X.mean(0), X.std(0)
        self.sd = np.where(sd > 1e-6, sd, 1.0)
        Z = (X - self.mu) / self.sd
        folds = _folds(groups)
        best = None
        for C in self.CS:
            oof = np.zeros(len(y))
            for f in range(5):
                lr = LogisticRegression(C=C, max_iter=3000).fit(Z[folds != f], y[folds != f])
                oof[folds == f] = lr.decision_function(Z[folds == f])
            ll = log_loss(y, 1 / (1 + np.exp(-oof)))
            if rl is not None:
                rl.event("g2_c", tag=tag, C=C, oof_logloss=ll)
            log.info("g2 %s C=%g: OOF log-loss %.4f", tag, C, ll)
            if best is None or ll < best[0]:
                best = (ll, C, oof)
        _, self.C, oof = best
        self.platt = LogisticRegression(C=1e6, max_iter=1000).fit(oof[:, None], y)
        self.lr = LogisticRegression(C=self.C, max_iter=3000).fit(Z, y)
        self.oof = self.platt.predict_proba(oof[:, None])[:, 1].astype(np.float32)
        return self

    def __call__(self, X):
        return self.platt.predict_proba(self.lr.decision_function((X - self.mu) / self.sd)[:, None])[:, 1].astype(np.float32)


def i3_labels(t3: pd.DataFrame, obs3: pd.DataFrame, null3: pd.DataFrame) -> np.ndarray:
    """[G1] 08:50 (2): x+ frames of reactive pairs (|Delta_expert| > tau_exp, p5_exam.exam's rule) -> 1, all else 0."""
    from . import p5_exam as E
    tau_exp = max(float(np.quantile(np.abs(null3.d_expert), 0.95)) if len(null3) else 0.0, E.TAU_EXP_MIN)
    pos = set(obs3.fn_plus[np.abs(obs3.d_expert) > tau_exp])
    return t3.frame_name.isin(pos).to_numpy()


def g2_fit(rl):
    """g2 on WOD train (G0's t4 set) and navtrain; the final probe on WOD val / navtest, OOF values on the training rows."""
    from . import real_g0 as G0
    out = {}
    for ds, tr, ev in (("wod", "wod_train", "wod_val"), ("nav", "navtrain", "navtest")):
        fr, X = G0.load_embed(tr)
        lab = wod_labels(tr) if ds == "wod" else nav_labels(tr, fr.frame_id.to_numpy())
        assert (lab.frame_id.to_numpy() == fr.frame_id.to_numpy()).all()
        groups = lab.sequence.to_numpy() if ds == "wod" else _nav_logs(tr, fr.frame_id.to_numpy())
        y = lab.label.to_numpy().astype(int)
        pr = Probe().fit(X, y, groups, rl, ds)
        fe, Xe = G0.load_embed(ev)
        ge = pr(Xe)
        lab.assign(oof=pr.oof).to_parquet(rl.dir / f"g2_{ds}_train_oof.parquet", index=False)
        np.savez_compressed(rl.dir / f"g2_{ds}_eval.npz", frame_id=fe.frame_id.to_numpy(), gate=ge)
        with open(rl.dir / f"g2_{ds}.pkl", "wb") as f:
            import pickle
            pickle.dump(pr, f)
        out[ds] = {"C": pr.C, "rows": len(y), "positive": float(y.mean()), "platt": [float(pr.platt.coef_[0, 0]),
                   float(pr.platt.intercept_[0])], "eval_rows": len(ge)}
        rl.event("g2_fit", dataset=ds, **out[ds])
        log.info("g2 %s: %s", ds, out[ds])
    return out


def _nav_logs(split: str, tokens: np.ndarray) -> np.ndarray:
    from . import navsim_zs as Z
    lg = {e["token"]: e["log_name"] for e in Z.load_index(split, slim=True)}
    return np.array([lg[t] for t in tokens])


def g2_i3(t3: pd.DataFrame, obs3: pd.DataFrame, null3: pd.DataFrame, rl=None) -> np.ndarray:
    """g2 on I3: probe on I3's own embedding, rule-expert conflict labels, OOF over 5 scene folds, Platt-calibrated."""
    from . import real_g0 as G0
    fr, X = G0.load_embed("i3")
    at = pd.Series(np.arange(len(fr)), index=fr.frame_id.to_numpy()).reindex(t3.frame_name).astype(int).to_numpy()
    X, groups = X[at], fr.base_id.to_numpy()[at]
    y = i3_labels(t3, obs3, null3).astype(int)
    return Probe().fit(X, y, groups, rl, "i3").oof


def select(rl, g1_oof_run: str, g1_nav_run: str, g2_run: str) -> pd.DataFrame:
    """[G1] 08:40 (4) / 08:50 (4): AUC of each gate against the g2 label on the 10% selection rows; the main arm."""
    from sklearn.metrics import roc_auc_score
    from . import navsim_zs as Z, waymo
    D = data_dir() / OUT / "g1"
    rows = []
    # WOD
    names = np.array((D / "sel_wod_frames.txt").read_text().split())
    lab = pd.read_parquet(data_dir() / g2_run / "g2_wod_train_oof.parquet").set_index("frame_id").loc[names]
    df = waymo.load_index()
    past, _ = waymo.load_ego()
    r = pd.Series(np.arange(len(df)), index=waymo.frame_names(df)).reindex(names).astype(int).to_numpy()
    v_ego = np.linalg.norm(past[r, -1, 2:4], axis=1)
    for m in MODELS:
        z = np.load(data_dir() / g1_oof_run / f"g1_wod_{m}_oof.npz")
        g1v = pd.Series(z["gate"], index=z["frame_name"].astype(str)).reindex(names).to_numpy()
        parts = [np.load(f) for f in sorted((data_dir() / "processed/drive_backbones/op_lead_g1sel" / m).glob("*.npz"))
                 if ".tmp" not in f.name]
        nm = np.concatenate([p["name"] for p in parts]).astype(str)
        li = pd.Series(np.arange(len(nm)), index=nm).reindex(names).astype(int).to_numpy()
        g3v = g3(np.concatenate([p["lead"] for p in parts])[li], np.concatenate([p["lead_prob"] for p in parts])[li], v_ego)
        for g, v in (("g1", g1v), ("g2", lab.oof.to_numpy()), ("g3", g3v)):
            rows.append({"dataset": "WOD", "model": m, "gate": g, "n": len(names), "positive": float(lab.label.mean()),
                         "auc": float(roc_auc_score(lab.label.to_numpy(), v))})
    # NAVSIM
    tok = np.array((D / "sel_navtrain_tokens.txt").read_text().split())
    lab = pd.read_parquet(data_dir() / g2_run / "g2_nav_train_oof.parquet").set_index("frame_id").loc[tok]
    vel = {e["token"]: np.linalg.norm(e["vel"][-1]) for e in Z.load_index("navtrain", slim=True)}
    v_ego = np.array([vel[t] for t in tok])
    for m in MODELS:
        z = np.load(data_dir() / g1_nav_run / f"g1_nav_{m}_oof.npz")
        g1v = pd.Series(z["gate"], index=z["tokens"]).reindex(tok).to_numpy()
        ld = np.load(data_dir() / "runs/navsim_zs/openpilot/navtrain" / f"{m}_temporal_lead.npz")
        li = pd.Series(np.arange(len(ld["tokens"])), index=ld["tokens"]).reindex(tok).astype(int).to_numpy()
        g3v = g3(ld["lead"][li], ld["lead_prob"][li], v_ego)
        for g, v in (("g1", g1v), ("g2", lab.oof.to_numpy()), ("g3", g3v)):
            rows.append({"dataset": "NAVSIM", "model": m, "gate": g, "n": len(tok), "positive": float(lab.label.mean()),
                         "auc": float(roc_auc_score(lab.label.to_numpy(), v))})
    t = pd.DataFrame(rows)
    simple = {"g3": 0, "g2": 1, "g1": 2}
    main = []
    for (ds, m), g in t.groupby(["dataset", "model"]):
        top = g.auc.max()
        cand = g[g.auc >= top - 0.005].assign(k=lambda x: x.gate.map(simple)).sort_values("k")
        main.append({"dataset": ds, "model": m, "main": cand.gate.iloc[0]})
    t = t.merge(pd.DataFrame(main), on=["dataset", "model"])
    t.to_csv(rl.dir / "selection_auc.csv", index=False)
    log.info("selection\n%s", t.to_markdown(index=False, floatfmt=".4f"))
    return t


# ================================================================ judges

def gate_desc(g: np.ndarray, scopes: dict) -> list[dict]:
    return [{"scope": k, "n": int(m.sum()), "gate_mean": float(g[m].mean()), "gate_open": float((g[m] > 0.5).mean())}
            for k, m in scopes.items() if m.any()]


def verdict_g1(tab: pd.DataFrame, act: pd.DataFrame, main: str, straight: str, ped: str, judge: str) -> str:
    """G1's two pre-registered cells: holds = straight activation <= 7%, the all-frames delta CI not wholly < 0 and the
    pedestrian subset's CI not wholly < 0; useful = holds and the pedestrian subset's CI wholly > 0."""
    g = tab.set_index(["scope", "judge"])
    a_st = float(act.set_index("scope").loc[straight, "activation"])
    allr, pr = g.loc[(main, judge)], g.loc[(ped, judge)]
    holds = a_st <= ACT_HARM and not allr.hi < 0 and not pr.hi < 0
    return "useful" if holds and pr.lo > 0 else "holds" if holds else "fails"


def run_wod(rl, gates: dict, deltas: dict, taus: dict, tag: str = "mc"):
    """elicit_e1.readouts, unchanged, on prior + g * Delta for every (model, gate) and on the ungated arm."""
    from . import elicit_e1 as E1, waymo
    d = E1.wod_frames()
    df = waymo.load_index()
    past, _ = waymo.load_ego()
    names = waymo.frame_names(df)
    rows = E1._rows(pd.Series(d["frame_name"]), names)
    v_ego = np.linalg.norm(past[rows, -1, 2:4], axis=1)
    scopes = {"all": np.ones(len(rows), bool), "straight_yaw": d["straight_yaw"], "pre_onset": d["pre_onset"],
              "Pedestrians": d["cluster"] == "Pedestrian", "Cyclists": d["cluster"] == "Cyclist"}
    tabs, acts, descs, verdicts, aucs = [], [], [], [], []
    sam = wod_sam_labels(d["frame_name"].astype(str)) if tag == "mc" else None
    for m in MODELS:
        if m not in deltas:
            continue
        gm = gates[m](d, v_ego) if callable(gates.get(m)) else gates[m]
        if sam is not None:              # [G1] 08:50 (2): the non-circular description, not a selection input
            from sklearn.metrics import roc_auc_score
            for gname, g in gm.items():
                for lab in ("any", "ped_cyc"):
                    y = sam[lab].to_numpy()
                    aucs.append({"model": m, "gate": gname, "label": f"SAM {lab}", "n": len(y), "positive": float(y.mean()),
                                 "auc": float(roc_auc_score(y, g))})
            np.savez_compressed(rl.dir / f"wod_gates_{m}.npz", frame_name=d["frame_name"], **gm)
        for gname, g in {"none": np.ones(len(rows), np.float32), **gm}.items():
            tab, act = E1.readouts(d, d[f"prior {m}"], g[:, None, None] * deltas[m], taus[m])
            meta = {"model": m, "gate": gname, "delta": tag}
            tabs.append(tab.assign(**meta))
            acts.append(act.assign(**meta, tau=taus[m]))
            descs += [{**meta, **r} for r in gate_desc(g, scopes)]
            v = verdict_g1(tab, act, "all", "straight_yaw", "Pedestrians", "RFS (rater)")
            verdicts.append({**meta, "verdict": v})
            rl.event("g1_wod_verdict", **meta, verdict=v)
            log.info("WOD %s / %s gate %s: %s\n%s", m, tag, gname, v,
                     tab[tab.judge == "RFS (rater)"].to_markdown(index=False, floatfmt=".3f"))
    for name, rows_ in (("wod_deltas", tabs), ("wod_activation", acts), ("wod_gate_desc", descs), ("wod_verdict", verdicts),
                        ("wod_sam_auc", aucs)):
        pd.DataFrame(rows_).to_csv(rl.dir / f"{name}_{tag}.csv", index=False)


def wod_sam_labels(names: np.ndarray) -> pd.DataFrame:
    """Q2b's SAM 3.1 detections on WOD val by E3 (8)'s rule (score > 0.5, logged-path corridor, single frame)."""
    from . import elicit_e3 as E3, fusion_q4 as Q, waymo
    df = waymo.load_index()
    _, future = waymo.load_ego()
    lst = pd.read_parquet(Q.root("lists") / "wod.parquet")
    f = E3.sam_flags(df, waymo.frame_names(df), future, lst).set_index("key").reindex(names)
    assert f.in_any.notna().all(), f"{int(f.in_any.isna().sum())} eval frames without SAM detections"
    return pd.DataFrame({"any": (f.in_pedestrian | f.in_cyclist | f.in_vehicle).astype(bool).to_numpy(),
                         "ped_cyc": (f.in_pedestrian | f.in_cyclist).astype(bool).to_numpy()})


def run_i3(rl, gate_fns: dict, with_g2: bool = False):
    """p5_exam.exam, unchanged, on the I3 pairs with the M-C correction gated frame by frame (elicit_i3's judge)."""
    from . import elicit_i3 as I, p5_exam as E, p5_openpilot
    with I.p5_set(I.I3):
        t3a, past3a, _, obs3, null3, pairs3 = E.load()
        keep = t3a.frame_name.isin(set(I.needed())).to_numpy()
        t3, past3 = t3a[keep].reset_index(drop=True), past3a[keep]
        op3 = p5_openpilot.load(t3, MODELS, sub="op_streams")
        lead3 = p5_openpilot.load(t3, MODELS, ("temporal", "lead", "lead_prob"), sub="op_streams_lead")
    z = np.load(data_dir() / I3_EXAM / "preds_i3.npz", allow_pickle=True)
    assert (z["frame_name"].astype(str) == t3.frame_name.to_numpy()).all()
    ego = E.ego_input(t3, past3)
    v_ego = np.linalg.norm(past3[:, -1, 2:4], axis=1)
    preds, gdesc, checks = {}, [], []
    g2v = g2_i3(t3, obs3, null3, rl) if with_g2 else None
    fam = pd.Series("none", index=t3.frame_name)
    for w, sub in (("fn_plus", obs3), ("fn_minus", obs3)):
        fam[sub[w].to_numpy()] = np.where(w == "fn_plus", sub.family.to_numpy(), "minus")
    fam[null3.fn_null.to_numpy()] = "null_world"         # not "null": pandas reads that back as NaN
    for m in MODELS:
        d_eq = float(np.abs(lead3[f"op-{m} temporal"] - op3[f"op-{m} temporal"]).max())
        checks.append({"model": m, "temporal_max_abs_diff": d_eq})
        log.info("I3 %s: re-run temporal vs stored max |diff| %.2e", m, d_eq)
        prior, mc = z[f"ridge_late op-{m} temporal"], z[f"M-C pair [{m}]"]
        delta = mc - prior
        preds[f"ridge_late op-{m} temporal"], preds[f"M-C pair [{m}]"] = prior, mc
        gs = {"g1": gate_fns["g1"][m](np.concatenate([ego, op3[f"op-{m} temporal"]], 1)),
              "g3": g3(lead3[f"op-{m} lead"], lead3[f"op-{m} lead_prob"], v_ego)}
        if g2v is not None:
            gs["g2"] = g2v
        x, v, p = lead_decode(lead3[f"op-{m} lead"], lead3[f"op-{m} lead_prob"])
        st = (fam == "static").to_numpy() & (p > 0.5)
        checks.append({"model": m, "check": "static world, P(lead) > 0.5", "n": int(st.sum()),
                       "v_lead_median": float(np.median(v[st])), "v_ego_median": float(np.median(v_ego[st]))})
        for gname, g in gs.items():
            preds[f"M-C pair [{m}] x {gname}"] = prior + g[:, None, None] * delta
            gdesc += [{"model": m, "gate": gname, **r} for r in
                      gate_desc(g, {k: (fam == k).to_numpy() for k in ("static", "cutin", "oncoming", "minus", "null_world")})]
    E.TFV6 = {}
    oo, nn = E.deltas(obs3, null3, t3, preds)
    res = E.exam(oo, nn, pairs3, list(preds))
    fl = res["flips"]
    r, taus, rows = res["obs"][res["obs"].reactive], res["taus"], []

    def flips(sub, ex):
        return ((np.sign(sub[ex]) == np.sign(sub.d_expert)) & E._moved(sub[ex], taus[ex])).astype(float).to_numpy()
    for m in MODELS:
        pr = f"ridge_late op-{m} temporal"
        for ex in [k for k in preds if k.startswith(f"M-C pair [{m}]")]:
            for scope, sub in [("pooled", r[r.family.isin(res["pooled_families"])])] + [(fa, r[r.family == fa]) for fa in sorted(r.family.unique())]:
                dd, lo, hi = E.boot_ratio(flips(sub, ex) - flips(sub, pr), np.ones(len(sub)), sub.base_id.to_numpy())
                rows.append({"examinee": ex, "vs": pr, "scope": scope, "n": len(sub), "delta": dd, "lo": lo, "hi": hi})
    fl.to_csv(rl.dir / "i3_flip_rates.csv", index=False)
    pd.DataFrame(rows).to_csv(rl.dir / "i3_paired_vs_prior.csv", index=False)
    pd.DataFrame(gdesc).to_csv(rl.dir / "i3_gate_desc.csv", index=False)
    pd.DataFrame(checks).to_csv(rl.dir / "i3_checks.csv", index=False)
    log.info("checks %s\n%s\n%s", checks, fl[fl.scope == "pooled"].to_markdown(index=False, floatfmt=".3f"),
             pd.DataFrame(rows).query("scope == 'pooled'").to_markdown(index=False, floatfmt=".3f"))


# ---------------------------------------------------------------- NAVSIM (elicit_e1.run_navsim / navsim_table's code paths)

NAV_PRIOR = "runs/navsim_zs/heads/20260925-232810"


def nav_lead(model: str) -> dict:
    z = np.load(data_dir() / "runs/navsim_zs/openpilot/navtest" / f"{model}_temporal_lead.npz")
    return {k: z[k] for k in ("tokens", "temporal", "lead", "lead_prob")}


def nav_write(rl, gates: dict, deltas: dict, tag: str, taus: dict):
    """prior (`ridge_late`) + g * Delta on navtest's 0.5 ... 4.0 s poses (heading kept, as E1), the activation table, and
    the devkit job list (scripts/navsim_zs_score.sh; names g1_<tag>_<gate>_ridge_late_<model>)."""
    from . import elicit_e1 as E1, p5_pairs as P
    sc = pd.read_csv(data_dir() / E1_NAV / "navtest_scopes.csv")
    scopes = {"all": np.ones(len(sc), bool), "straight": sc.straight.to_numpy(), "ped_cyc_corridor": sc.ped_cyc_corridor.to_numpy(),
              "no ped_cyc": ~sc.ped_cyc_corridor.to_numpy()}
    acts, descs, jobs = [], [], []
    for m, (tok, delta) in deltas.items():
        p = np.load(data_dir() / NAV_PRIOR / f"navtest_ridge_late_{m}_temporal.npz")
        assert (p["tokens"] == tok).all() and (sc.token.to_numpy() == tok).all()
        for gname, g in {"none": np.ones(len(tok), np.float32), **gates[m]}.items():
            arm = p["poses"].copy()
            arm[..., :2] += g[:, None, None] * delta[:, 1:16:2]
            name = f"g1_{tag}_{gname}_ridge_late_{m}"
            if gname != "none" or tag != "mc":          # the ungated M-C arm is E1's, already scored
                np.savez(rl.dir / f"navtest_{name}.npz", tokens=tok, poses=arm.astype(np.float32))
                jobs += [f"{v} navtest {name} {rl.dir / f'navtest_{name}.npz'}" for v in ("v1", "v2")]
            act = (np.abs(P.v2(E1._grid20(arm)) - P.v2(E1._grid20(p["poses"]))) >= taus[m]).astype(float)
            for k, msk in scopes.items():
                acts.append({"model": m, "delta": tag, "gate": gname, "scope": k, "n": int(msk.sum()), "tau": taus[m],
                             "activation": float(act[msk].mean())})
            descs += [{"model": m, "delta": tag, "gate": gname, **r} for r in gate_desc(g, scopes)]
    pd.DataFrame(acts).to_csv(rl.dir / f"navsim_activation_{tag}.csv", index=False)
    pd.DataFrame(descs).to_csv(rl.dir / f"navsim_gate_desc_{tag}.csv", index=False)
    (rl.dir / f"score_jobs_{tag}.txt").write_text("\n".join(jobs) + "\n")
    log.info("NAVSIM %s: %d devkit jobs -> %s\n%s", tag, len(jobs), rl.dir / f"score_jobs_{tag}.txt",
             pd.DataFrame(acts).pivot_table(index=["model", "gate"], columns="scope", values="activation").to_markdown(floatfmt=".3f"))


def nav_table(rl, pairs: dict):
    """elicit_e1.navsim_table's paired token bootstrap for {label: (arm devkit name, prior devkit name)}."""
    from .openloop_standing import _latest
    sc = pd.read_csv(data_dir() / E1_NAV / "navtest_scopes.csv").set_index("token")
    groups = {"all": None, "ped_cyc_corridor": sc.ped_cyc_corridor, "no ped_cyc": ~sc.ped_cyc_corridor, "straight": sc.straight}
    rows = []
    rng = np.random.default_rng(0)
    f = lambda df: df[df["token"].str.fullmatch(r"[0-9a-f]{16,17}") & df["valid"].astype(bool)].set_index("token")["score"].astype(float)  # noqa: E731
    for label, (arm, prior) in pairs.items():
        for ver, metric in (("v1", "PDMS"), ("v2", "EPDMS")):
            a, b = _latest(ver, "navtest", arm), _latest(ver, "navtest", prior)
            if a is None or b is None:
                log.warning("%s %s: devkit scores missing (%s / %s)", label, metric, a is None, b is None)
                continue
            x, y = f(a).align(f(b), join="inner")
            for g, msk in groups.items():
                keep = np.ones(len(x), bool) if msk is None else msk.reindex(x.index).fillna(False).to_numpy(bool)
                d = (x - y).to_numpy()[keep]
                bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
                rows.append({"arm": label, "metric": metric, "group": g, "n": len(d), "prior_score": 100 * y.to_numpy()[keep].mean(),
                             "arm_score": 100 * x.to_numpy()[keep].mean(), "delta": 100 * d.mean(),
                             "lo": 100 * np.percentile(bs, 2.5), "hi": 100 * np.percentile(bs, 97.5)})
    t = pd.DataFrame(rows)
    t.to_csv(rl.dir / "navsim_paired.csv", index=False)
    log.info("navtest\n%s", t.to_markdown(index=False, floatfmt=".2f"))
    return t


# ================================================================ entry points

def load_g1(kind: str, run: str) -> dict:
    return {m: Gate.load(data_dir() / run / f"g1_{kind}_{m}.pt") for m in MODELS}


def mc_taus() -> dict:
    fl = pd.read_csv(data_dir() / MC_RUN / "flip_rates.csv")
    return {m: float(fl[(fl.examinee == f"M-C pair [{m}]") & (fl.scope == "pooled")].tau_model.iloc[0]) for m in MODELS}


def _g2_eval(g2_run: str, ds: str, keys: np.ndarray) -> np.ndarray:
    z = np.load(data_dir() / g2_run / f"g2_{ds}_eval.npz", allow_pickle=True)
    return pd.Series(z["gate"], index=z["frame_id"].astype(str)).reindex(keys.astype(str)).to_numpy().astype(np.float32)


def wod_gates(rl, g1: dict, g2_run: str = ""):
    """{model: f(d, v_ego) -> {gate: (n,)}} on E1's WOD frames; checks the lead re-run's `temporal` against the stored one."""
    from . import elicit_e1 as E1, waymo

    def gates_for(m):
        def f(d, v_ego):
            names = d["frame_name"].astype(str)
            ld = wod_lead(m, names)
            diff = float(np.abs(ld["temporal"] - d[f"op {m}"]).max())
            rl.event("g1_wod_lead_equiv", model=m, temporal_max_abs_diff=diff)
            log.info("WOD %s: re-run temporal vs stored max |diff| %.2e", m, diff)
            df = waymo.load_index()
            past, _ = waymo.load_ego()
            rows = E1._rows(pd.Series(names), waymo.frame_names(df))
            ego = np.concatenate([waymo.ego_state(past[rows]), waymo.intent_onehot(df.iloc[rows])], 1)
            out = {"g1": g1[m](np.concatenate([ego, d[f"op {m}"]], 1)), "g3": g3(ld["lead"], ld["lead_prob"], v_ego)}
            if g2_run:
                out["g2"] = _g2_eval(g2_run, "wod", names)
                assert not np.isnan(out["g2"]).any()
            return out
        return f
    return {m: gates_for(m) for m in MODELS}


def nav_gates(g1_run: str, g2_run: str = "") -> dict:
    """{model: {gate: (n,)}} on navtest in token order: g1 from its fit run, g3 from the lead re-run."""
    from . import navsim_zs as Z
    idx = Z.load_index("navtest", slim=True)
    tok = np.array([e["token"] for e in idx])
    v_ego = np.array([np.linalg.norm(e["vel"][-1]) for e in idx])
    out = {}
    for m in MODELS:
        z = np.load(data_dir() / g1_run / f"g1_nav_{m}_navtest.npz")
        ld = nav_lead(m)
        at = pd.Series(np.arange(len(ld["tokens"])), index=ld["tokens"]).reindex(tok).astype(int).to_numpy()
        assert (z["tokens"] == tok).all()
        out[m] = {"g1": z["gate"], "g3": g3(ld["lead"][at], ld["lead_prob"][at], v_ego)}
        if g2_run:
            out[m]["g2"] = _g2_eval(g2_run, "nav", tok)
            assert not np.isnan(out[m]["g2"]).any()
    return out


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("g1-wod", "g1-nav", "g2", "select", "i3", "wod", "nav-write", "nav-table"))
    ap.add_argument("--g1-run", default="", help="the g1 fit run dir (relative to DATA_DIR)")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--oof-rows", default="", help="g1-wod: selection rows file -> out-of-fold fits")
    ap.add_argument("--tags", default="mc")
    ap.add_argument("--g1-nav-run", default="")
    ap.add_argument("--g1-oof-run", default="")
    ap.add_argument("--g2-run", default="", help="the g2 run dir; given -> g2 joins the arms")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 16)))
    models = tuple(a.models.split(","))
    rl = RunLog("real-data-transfer", a.what)
    if a.what == "g1-wod":
        g1_wod(rl, models, a.oof_rows, a.g1_run)
    elif a.what == "g1-nav":
        g1_nav(rl, models)
    elif a.what == "g2":
        g2_fit(rl)
    elif a.what == "select":
        select(rl, a.g1_oof_run, a.g1_nav_run, a.g2_run)
    elif a.what == "i3":
        run_i3(rl, {"g1": load_g1("wod", a.g1_run)}, with_g2=bool(a.g2_run))
    elif a.what == "wod":
        deltas = {m: np.load(data_dir() / E1_WOD / f"wod_delta_{m}.npz")["delta"] for m in MODELS}
        run_wod(rl, wod_gates(rl, load_g1("wod", a.g1_run), a.g2_run), deltas, mc_taus(), "mc")
    elif a.what == "nav-write":
        deltas = {}
        for m in MODELS:
            z = np.load(data_dir() / E1_NAV / f"navtest_delta_{m}.npz")
            deltas[m] = (z["tokens"], z["delta"])
        nav_write(rl, nav_gates(a.g1_nav_run, a.g2_run), deltas, "mc", mc_taus())
    elif a.what == "nav-table":
        pairs = {}
        for tag in a.tags.split(","):
            for m in MODELS:
                prior = f"heads_ridge_late_{m}_temporal"
                if tag == "mc":
                    pairs[f"mc none {m}"] = (f"e1_ridge_late_{m}_plus_mc", prior)
                for g in ("g1", "g2", "g3") + (("none",) if tag != "mc" else ()):
                    pairs[f"{tag} {g} {m}"] = (f"g1_{tag}_{g}_ridge_late_{m}", prior)
        nav_table(rl, pairs)
    rl.close()


if __name__ == "__main__":
    main()
