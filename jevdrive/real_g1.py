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


def fit_gated(X: np.ndarray, R: torch.Tensor, sp, pre: np.ndarray | None, rl, tag: str, seed: int = 0) -> tuple:
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
    for l1 in (0.0, 1e-3, 1e-2):
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


def g1_wod(rl, models=MODELS, exclude: str = "", seed: int = 0):
    """g1 on WOD train: target = residual of `ridge ego` fitted on the same train rows; input [ego, op temporal].
    `exclude` (a file of sequences) removes those train sequences from the fit (the selection rows' held-out fit)."""
    from . import planner, waymo_ladder as L, waymo_stage_a as sa
    ctx = wod_train_context()
    for m in models:
        a = L.align(ctx, f"op_{m}_p3_trainval", ["temporal"])
        keep = a["covered"].copy()
        if exclude:
            drop = set(Path(exclude).read_text().split())
            keep &= ~pd.Series(ctx["seq"]).astype(str).isin(drop).to_numpy()
        sel = np.flatnonzero(keep)
        seq, h = ctx["seq"][sel], ctx["half"][sel]
        sp = sa.Halves(ctx["df"], seq, h == 0, h == 1, seed)
        n = len(sel)
        F = torch.as_tensor(ctx["fut"][sel].reshape(n, -1), device=L.DEV)
        Xe = planner.standardize(torch.as_tensor(ctx["ego"][sel], device=L.DEV), sp.train)
        _, st_ego, W = sa.ridge_cv(Xe, F, sp, ctx["fut"][sel])
        R = F - planner.linear_apply(W, Xe, np.arange(n))[0]
        del Xe, F
        X = np.concatenate([ctx["ego"][sel], a["temporal"][sel].astype(np.float32)], 1)
        log.info("g1 WOD %s: %d train / %d val rows, d = %d, ridge ego %s", m, len(sp.train), len(sp.val), X.shape[1], st_ego)
        gate, st = fit_gated(X, R, sp, ctx["sub"]["pre_onset"][sel], rl, f"g1-wod-{m}", seed)
        tag = f"g1_wod_{m}" + ("_heldout" if exclude else "")
        gate.save(rl.dir / f"{tag}.pt", **{k: v for k, v in st.items() if k in ("l1", "d", "out")})
        g = gate(X[sp.val])
        np.savez_compressed(rl.dir / f"{tag}_val.npz", frame_name=ctx["fname"][sel][sp.val], gate=g)
        rl.event("g1_saved", tag=tag, **{k: v for k, v in st.items() if isinstance(v, (int, float))})
        del R, X
        torch.cuda.empty_cache()


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
    tabs, acts, descs, verdicts = [], [], [], []
    for m in MODELS:
        if m not in deltas:
            continue
        gm = gates[m](d, v_ego) if callable(gates.get(m)) else gates[m]
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
    for name, rows_ in (("wod_deltas", tabs), ("wod_activation", acts), ("wod_gate_desc", descs), ("wod_verdict", verdicts)):
        pd.DataFrame(rows_).to_csv(rl.dir / f"{name}_{tag}.csv", index=False)


def run_i3(rl, gate_fns: dict):
    """p5_exam.exam, unchanged, on the I3 pairs with the M-C correction gated frame by frame (elicit_i3's judge)."""
    from . import elicit_i3 as I, p5_exam as E, p5_openpilot
    with I.p5_set(I.I3):
        t3a, past3a, _, obs3, null3, pairs3 = E.load()
        keep = t3a.frame_name.isin(set(I.needed())).to_numpy()
        t3, past3 = t3a[keep].reset_index(drop=True), past3a[keep]
        op3 = p5_openpilot.load(t3, MODELS, sub="op_streams")
        lead3 = p5_openpilot.load(t3, MODELS, ("temporal", "lead", "lead_prob"), sub="op_streams_lead")
    z = np.load(data_dir() / I3_EXAM / "preds_i3.npz")
    assert (z["frame_name"].astype(str) == t3.frame_name.to_numpy()).all()
    ego = E.ego_input(t3, past3)
    v_ego = np.linalg.norm(past3[:, -1, 2:4], axis=1)
    preds, gdesc, checks = {}, [], []
    fam = pd.Series("none", index=t3.frame_name)
    for w, sub in (("fn_plus", obs3), ("fn_minus", obs3)):
        fam[sub[w].to_numpy()] = np.where(w == "fn_plus", sub.family.to_numpy(), "minus")
    fam[null3.fn_null.to_numpy()] = "null"
    for m in MODELS:
        d_eq = float(np.abs(lead3[f"op-{m} temporal"] - op3[f"op-{m} temporal"]).max())
        checks.append({"model": m, "temporal_max_abs_diff": d_eq})
        log.info("I3 %s: re-run temporal vs stored max |diff| %.2e", m, d_eq)
        prior, mc = z[f"ridge_late op-{m} temporal"], z[f"M-C pair [{m}]"]
        delta = mc - prior
        preds[f"ridge_late op-{m} temporal"], preds[f"M-C pair [{m}]"] = prior, mc
        gs = {"g1": gate_fns["g1"][m](np.concatenate([ego, op3[f"op-{m} temporal"]], 1)),
              "g3": g3(lead3[f"op-{m} lead"], lead3[f"op-{m} lead_prob"], v_ego)}
        for k, fn in gate_fns.items():
            if k not in gs:
                gs[k] = fn[m](t3)
        x, v, p = lead_decode(lead3[f"op-{m} lead"], lead3[f"op-{m} lead_prob"])
        st = (fam == "static").to_numpy() & (p > 0.5)
        checks.append({"model": m, "check": "static world, P(lead) > 0.5", "n": int(st.sum()),
                       "v_lead_median": float(np.median(v[st])), "v_ego_median": float(np.median(v_ego[st]))})
        for gname, g in gs.items():
            preds[f"M-C pair [{m}] x {gname}"] = prior + g[:, None, None] * delta
            gdesc += [{"model": m, "gate": gname, **r} for r in
                      gate_desc(g, {k: (fam == k).to_numpy() for k in ("static", "cutin", "oncoming", "minus", "null")})]
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


# ================================================================ entry points

def load_g1(kind: str, run: str) -> dict:
    return {m: Gate.load(data_dir() / run / f"g1_{kind}_{m}.pt") for m in MODELS}


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("g1-wod", "g1-nav", "i3", "wod"))
    ap.add_argument("--g1-run", default="", help="the g1-wod run dir (relative to DATA_DIR)")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--exclude", default="")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 16)))
    models = tuple(a.models.split(","))
    rl = RunLog("real-data-transfer", f"g1-{a.what}")
    if a.what == "g1-wod":
        g1_wod(rl, models, a.exclude)
    elif a.what == "g1-nav":
        g1_nav(rl, models)
    elif a.what == "i3":
        run_i3(rl, {"g1": load_g1("wod", a.g1_run)})
    elif a.what == "wod":
        from . import elicit_e1 as E1
        fl = pd.read_csv(data_dir() / MC_RUN / "flip_rates.csv")
        taus = {m: float(fl[(fl.examinee == f"M-C pair [{m}]") & (fl.scope == "pooled")].tau_model.iloc[0]) for m in MODELS}
        d_names = None
        deltas = {}
        for m in MODELS:
            z = np.load(data_dir() / E1_WOD / f"wod_delta_{m}.npz")
            d_names = z["frame_name"].astype(str)
            deltas[m] = z["delta"]
        g1 = load_g1("wod", a.g1_run)

        def gates_for(m):
            def f(d, v_ego):
                assert (d["frame_name"].astype(str) == d_names).all()
                ld = wod_lead(m, d_names)
                diff = float(np.abs(ld["temporal"] - d[f"op {m}"]).max())
                rl.event("g1_wod_lead_equiv", model=m, temporal_max_abs_diff=diff)
                log.info("WOD %s: re-run temporal vs stored max |diff| %.2e", m, diff)
                from . import waymo
                df = waymo.load_index()
                past, _ = waymo.load_ego()
                rows = E1._rows(pd.Series(d_names), waymo.frame_names(df))
                ego = np.concatenate([waymo.ego_state(past[rows]), waymo.intent_onehot(df.iloc[rows])], 1)
                return {"g1": g1[m](np.concatenate([ego, d[f"op {m}"]], 1)),
                        "g3": g3(ld["lead"], ld["lead_prob"], v_ego)}
            return f
        run_wod(rl, {m: gates_for(m) for m in MODELS}, deltas, taus, "mc")
    rl.close()


if __name__ == "__main__":
    main()
