"""Real-data transfer G2 (retrain the reaction head on the HUGSIM 3DGS vehicle pairs) and G1c (g3 x a constant brake)
(todos/2026-09-26-real-data-transfer.md, G2 and deviation-log entry [G2] 10:50, written before any number).

  fit        per model x seed: 5 scene folds of I3; in each fold, with the other four folds' scenes only,
             M-C `pair` (reactivity_mc's closed form), the `hard` / `uniform` controls and E5's students A / B on the
             paired target (y+ - y-) - (p+ - p-) over the frozen CARLA `ridge_late` prior; the I3 exam (p5_exam.exam
             unchanged) on the held-out scenes, next to the CARLA-trained heads, the g2 description and G1c on I3
  transfer   Delta = mean of the 5 fold heads on WOD (E1's 19 663 frames, elicit_e1.readouts) and navtest (predictions
             and devkit jobs); each head standardises with its own I3 mu rows (main) or WOD-train rows (descriptive)
  g1c-wod / g1c-nav-write   the constant-brake control of G1's g3 x M-C arm
  nav-table  paired token bootstrap of the devkit scores (vehicle-approach group added)

    python -m jevdrive.real_g2 fit | transfer | g1c-wod | g1c-nav-write | nav-table | summary | figs
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .common import data_dir, get_logger

log = get_logger(__name__)
MODELS, SEEDS = ("cinque", "lebowski"), (0, 1, 2)
OUT = "runs/real-data-transfer"
G1_WOD = "runs/real-data-transfer/wod/20260926-092810"        # G1's final WOD run: wod_gates_<m>.npz (g1, g2, g3, c2)
LINEAR, STUDENTS = ("pair", "hard", "uniform"), ("A", "B")
ACT_HARM, NULL_FF_MAX = 0.07, 0.07
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _sd_safe(x: torch.Tensor, rows) -> tuple[torch.Tensor, torch.Tensor]:
    """planner.standardize's statistics (float64, sd <= 1e-6 -> 1), kept for foreign rows."""
    mu, sd = x[rows].double().mean(0), x[rows].double().std(0, correction=0)
    return mu.float(), torch.where(sd > 1e-6, sd, torch.ones_like(sd)).float()


def _z(x: torch.Tensor, st) -> torch.Tensor:
    return (x - st[0].to(x.device)) / st[1].to(x.device) / np.sqrt(x.shape[1])


MASK = np.arange(64) % 8 == 7


def _zs(xo: torch.Tensor, e: torch.Tensor, st: dict) -> torch.Tensor:
    """E5's student input [z_op, z_e] (mask bits not standardised), with the sd <= 1e-6 -> 1 rule ([G2] 10:53)."""
    m = torch.as_tensor(MASK, device=e.device)
    ze = torch.where(m, e, (e - st["e_mu"].to(e.device)) / st["e_sd"].to(e.device)) / np.sqrt(e.shape[1])
    return torch.cat([(xo - st["op_mu"].to(xo.device)) / st["op_sd"].to(xo.device) / np.sqrt(xo.shape[1]), ze], 1)


# ================================================================ I3 data

def i3_data() -> dict:
    """The I3 rows with Qwen features (elicit_i3.needed), their features, CARLA-fit predictions and lead outputs."""
    from . import elicit_i3 as I, p5_exam as E, p5_openpilot, p5_pairs as P, real_g0 as G0
    from .real_g1 import I3_EXAM
    with I.p5_set(I.I3):
        ta, pa, fa, obs, null, pairs = E.load()
        keep = ta.frame_name.isin(set(I.needed())).to_numpy()
        t, past, fut = ta[keep].reset_index(drop=True), pa[keep], fa[keep]
        Q = P.load_features(t, ("L18_last",))["L18_last"]
        op = p5_openpilot.load(t, MODELS, sub="op_streams")
        lead = p5_openpilot.load(t, MODELS, ("lead", "lead_prob"), sub="op_streams_lead")
    fr, emb = G0.load_embed("i3")
    emb = emb[pd.Series(np.arange(len(fr)), index=fr.frame_id.to_numpy()).reindex(t.frame_name).astype(int).to_numpy()]
    z = np.load(data_dir() / I3_EXAM / "preds_i3.npz", allow_pickle=True)
    assert (z["frame_name"].astype(str) == t.frame_name.to_numpy()).all()
    n = len(t)
    F = fut.reshape(n, -1).astype(np.float32)
    pos = pd.Series(np.arange(n), index=t.frame_name)
    ip = np.r_[pos[obs.fn_plus].to_numpy(), pos[null.fn_plus].to_numpy()]
    im = np.r_[pos[obs.fn_minus].to_numpy(), pos[null.fn_null].to_numpy()]
    grp = np.r_[obs.base_id.to_numpy(), null.base_id.to_numpy()].astype(str)
    okF = ~np.isnan(F).any(1)
    ok = okF[ip] & okF[im]
    ego = z["ridge ego"].reshape(n, -1)
    d = {"t": t, "past": past, "F": F, "okF": okF, "obs": obs, "null": null, "pairs": pairs, "Q": Q, "emb": emb,
         "ip": ip[ok], "im": im[ok], "grp": grp[ok], "minus": (t.world == "minus").to_numpy(),
         "s_ego": np.where(okF, np.linalg.norm((ego - np.nan_to_num(F)).reshape(n, 20, 2), axis=-1).mean(1), np.nan),
         "v_ego": np.linalg.norm(past[:, -1, 2:4], axis=1), "carla": z}
    for m in MODELS:
        d[f"op {m}"] = op[f"op-{m} temporal"]
        d[f"prior {m}"] = z[f"ridge_late op-{m} temporal"]
        d[f"lead {m}"] = (lead[f"op-{m} lead"], lead[f"op-{m} lead_prob"])
    log.info("I3: %d rows (%d minus), %d training pairs (%d dropped for NaN futures), %d scenes", n, d["minus"].sum(),
             ok.sum(), (~ok).sum(), t.base_id.nunique())
    return d


# ================================================================ fits

def fit_model_seed(D: dict, m: str, s: int, rl) -> tuple[dict, dict]:
    """Held-out Delta (n, 20, 2) per head and the fold heads (for transfer) of one model x seed ([G2] 10:50 (1)-(6))."""
    from . import elicit_e5 as E5, p5_exam as E, reactivity_mc as MC
    MC.EIGH_DEVICE = DEV                              # float64 eigh on the GPU ([G2] 10:53): the box's CPUs are saturated
    t = D["t"]
    n = len(t)
    fold = E.folds(t, D["pairs"], s)
    seq = t.base_id.to_numpy().astype(str)
    F = torch.as_tensor(np.nan_to_num(D["F"]), device=DEV)
    prior = torch.as_tensor(D[f"prior {m}"].reshape(n, -1), device=DEV)
    Q, Xo = torch.as_tensor(D["Q"], device=DEV), torch.as_tensor(D[f"op {m}"], device=DEV)
    Em = torch.as_tensor(D["emb"], device=DEV)
    s_ego = torch.as_tensor(np.nan_to_num(D["s_ego"]), device=DEV)
    held = {k: np.zeros((n, 20, 2), np.float32) for k in (*LINEAR, *STUDENTS)}
    heads, weights = {}, {}
    for f in range(E.K_FOLDS):
        tr = np.flatnonzero(D["minus"] & (fold != f))
        ev = np.flatnonzero(fold == f)
        keep = fold[D["ip"]] != f
        ip, im, grp = D["ip"][keep], D["im"][keep], D["grp"][keep]
        Rp = (F[ip] - F[im]) - (prior[ip] - prior[im])
        sq, so = _sd_safe(Q, tr), _sd_safe(Xo, tr)
        Z = torch.cat([_z(Q, sq), _z(Xo, so)], 1)
        zbar = Z[tr].mean(0)
        # M-C pair: reactivity_mc.fit_fold's solver, lambda grid and inner split, unchanged
        Zc, Dd, mu = Z[tr] - zbar, Z[ip] - Z[im], len(ip) / len(tr)
        score = np.zeros(len(MC.LAMS))
        for a, b in MC._inner_splits(grp):
            Ws = MC._solve_pair(Dd[a], Rp[a], Zc, mu, MC.LAMS)
            score += [float(((Dd[b] @ W - Rp[b]) ** 2).sum()) for W in Ws]
        best = int(np.argmin(score))
        W = MC._solve_pair(Dd, Rp, Zc, mu, [MC.LAMS[best]])[0]
        teach = (Z - zbar) @ W
        held["pair"][ev] = teach[ev].reshape(-1, 20, 2).cpu().numpy()
        heads[m, s, f, "pair"] = {"q": sq, "op": so, "zbar": zbar.cpu(), "W": W.cpu(), "lam": float(MC.LAMS[best])}
        rl.event("g2_fit", model=m, seed=s, fold=f, head="pair", lam=float(MC.LAMS[best]), lam_edge=best in (0, len(MC.LAMS) - 1),
                 n_pair=len(ip), n_mu=len(tr), n_eval=len(ev))
        # hard-example reweighting and uniform imitation (fit_fold's recipe; rows with a finite future)
        rows = np.unique(np.r_[tr, ip, im])
        rows = rows[D["okF"][rows]]
        Y = F[rows] - prior[rows]
        inner_r = MC._inner_splits(seq[rows])
        for arm, w in (("hard", s_ego[rows] / s_ego[rows].mean()), ("uniform", torch.ones(len(rows), device=DEV))):
            sc = np.zeros(len(MC.LAMS))
            for a, b in inner_r:
                fits = MC._solve_weighted(Z[rows[a]], Y[a], w[a], MC.LAMS)
                sc += [float((w[b] * ((Z[rows[b]] @ Wh + c - Y[b]) ** 2).sum(1)).sum()) for Wh, c in fits]
            bb = int(np.argmin(sc))
            Wh, c = MC._solve_weighted(Z[rows], Y, w, [MC.LAMS[bb]])[0]
            held[arm][ev] = (Z[ev] @ Wh + c).reshape(-1, 20, 2).cpu().numpy()
            heads[m, s, f, arm] = {"q": sq, "op": so, "W": Wh.cpu(), "c": c.cpu(), "lam": float(MC.LAMS[bb])}
            rl.event("g2_fit", model=m, seed=s, fold=f, head=arm, lam=float(MC.LAMS[bb]), lam_edge=bb in (0, len(MC.LAMS) - 1),
                     n_rows=len(rows))
        # E5's students on [z_op, z_e], mu-row statistics
        st = dict(zip(("op_mu", "op_sd"), _sd_safe(Xo, tr))) | dict(zip(("e_mu", "e_sd"), _sd_safe(Em, tr)))
        X = _zs(Xo, Em, st).float()
        heads[m, s, f, "student"] = st
        for arm, tch in (("A", None), ("B", teach)):
            dl = E5.train_student(X, ip, im, Rp, tr, grp, tch, s, rl, f"{m} f{f} {arm}", weights)
            held[arm][ev] = dl[ev].reshape(-1, 20, 2).cpu().numpy()
        rl.log.info("%s s%d fold %d: %d pairs, %d mu rows, %d eval rows; pair lam %g", m, s, f, len(ip), len(tr), len(ev),
                    MC.LAMS[best])
    return held, {"heads": heads, "weights": {(m, s, *k): v for k, v in weights.items()}}


def i3_exam(rl, D: dict, held: dict) -> dict:
    """p5_exam.exam unchanged (no TFv6 columns) on every examinee; paired flips vs the prior and vs CARLA's M-C."""
    from . import p5_exam as E, real_g0 as G0
    from .real_g1 import g3, g2_i3
    t = D["t"]
    z = D["carla"]
    preds, gd = {}, []
    g2v = g2_i3(t, D["obs"], D["null"], rl)
    for m in MODELS:
        prior = D[f"prior {m}"]
        preds[f"ridge_late op-{m} temporal"] = prior
        dmc = z[f"M-C pair [{m}]"] - prior
        for arm in LINEAR:
            preds[f"CARLA M-C {arm} [{m}]"] = z[f"M-C {arm} [{m}]"]
        for arm in STUDENTS:
            for sd in SEEDS:
                zs = np.load(G0.g0_dir(f"i3_delta_{m}_{arm}_s{sd}.npz"), allow_pickle=True)
                at = pd.Series(np.arange(len(zs["frame_name"])), index=zs["frame_name"].astype(str))
                preds[f"CARLA student {arm} s{sd} [{m}]"] = prior + zs["delta"][at.reindex(t.frame_name).astype(int).to_numpy()]
        for s in SEEDS:
            for arm in LINEAR:
                preds[f"G2 M-C {arm} s{s} [{m}]"] = prior + held[m, s][arm]
            for arm in STUDENTS:
                preds[f"G2 student {arm} s{s} [{m}]"] = prior + held[m, s][arm]
            for src, k in (("M-C pair", "pair"), ("student A", "A")):     # [G2] 10:50 (10): g2 x the G2 Delta, descriptive
                preds[f"G2 {src} s{s} x g2 [{m}]"] = prior + g2v[:, None, None] * held[m, s][k]
        # G1c on I3 (descriptive): the g3-weighted mean of CARLA's M-C Delta, longitudinal only / 2-D
        g = g3(*D[f"lead {m}"], D["v_ego"])
        c = (g[:, None, None] * dmc).sum(0) / g.sum()
        preds[f"G1c g3 x M-C [{m}]"] = prior + g[:, None, None] * dmc
        preds[f"G1c g3 x const [{m}]"] = prior + g[:, None, None] * (c * [1.0, 0.0])
        preds[f"G1c g3 x const2d [{m}]"] = prior + g[:, None, None] * c
        gd.append({"model": m, "g3_mean": float(g.mean()), "g3_open": float((g > 0.5).mean()), "c_x_2s": float(c[7, 0]),
                   "c_y_2s": float(c[7, 1]), "c_x_4s": float(c[15, 0]), "c_y_4s": float(c[15, 1])})
    E.TFV6 = {}
    oo, nn = E.deltas(D["obs"], D["null"], t, preds)
    res = E.exam(oo, nn, D["pairs"], list(preds))
    r, taus, rows = res["obs"][res["obs"].reactive], res["taus"], []

    def flips(sub, ex):
        return ((np.sign(sub[ex]) == np.sign(sub.d_expert)) & E._moved(sub[ex], taus[ex])).astype(float).to_numpy()
    scopes = [("pooled", r[r.family.isin(res["pooled_families"])])] + [(fa, r[r.family == fa]) for fa in sorted(r.family.unique())]
    for m in MODELS:
        for ref in (f"ridge_late op-{m} temporal", f"CARLA M-C pair [{m}]"):
            for ex in [k for k in preds if k.endswith(f"[{m}]") and k != ref]:
                for scope, sub in scopes:
                    dd, lo, hi = E.boot_ratio(flips(sub, ex) - flips(sub, ref), np.ones(len(sub)), sub.base_id.to_numpy())
                    rows.append({"examinee": ex, "vs": ref, "scope": scope, "n": len(sub), "delta": dd, "lo": lo, "hi": hi})
    res["flips"].to_csv(rl.dir / "i3_flip_rates.csv", index=False)
    pd.DataFrame(rows).to_csv(rl.dir / "i3_paired.csv", index=False)
    pd.DataFrame(gd).to_csv(rl.dir / "i3_g1c_const.csv", index=False)
    np.savez_compressed(rl.dir / "i3_g2_gate.npz", frame_name=t.frame_name.to_numpy(), g2=g2v)
    fl = res["flips"]
    log.info("I3 pooled\n%s", fl[fl.scope == "pooled"][["examinee", "tau_model", "flip_rate", "flip_lo", "flip_hi",
                                                           "false_flip_null_oos", "false_flip_nonreactive"]].to_markdown(index=False, floatfmt=".3f"))
    return res


def run_fit(rl, models=MODELS, seeds=SEEDS):
    """One shard (models x seeds) of the fits; `exam` merges the shards (they run as parallel processes)."""
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
    D = i3_data()
    held, store = {}, {"heads": {}, "weights": {}}
    for m in models:
        for s in seeds:
            h, st = fit_model_seed(D, m, s, rl)
            held[m, s] = h
            store["heads"].update(st["heads"])
            store["weights"].update(st["weights"])
            torch.cuda.empty_cache()
    torch.save(store, rl.dir / "heads.pt")
    np.savez_compressed(rl.dir / "i3_held_delta.npz", frame_name=D["t"].frame_name.to_numpy(),
                        **{f"{m} s{s} {k}": v for (m, s), h in held.items() for k, v in h.items()})


def run_exam(rl, fit_runs: list[str]):
    """Merge the fit shards into one heads.pt / i3_held_delta.npz and run the I3 exam."""
    D = i3_data()
    held, store = {}, {"heads": {}, "weights": {}}
    for r in fit_runs:
        st = torch.load(data_dir() / r / "heads.pt", weights_only=False)
        store["heads"].update(st["heads"])
        store["weights"].update(st["weights"])
        z = np.load(data_dir() / r / "i3_held_delta.npz", allow_pickle=True)
        assert (z["frame_name"].astype(str) == D["t"].frame_name.to_numpy()).all()
        for k in z.files:
            if k != "frame_name":
                m, s, arm = k.split()
                held.setdefault((m, int(s[1:])), {})[arm] = z[k]
    assert set(held) == {(m, s) for m in MODELS for s in SEEDS}, sorted(held)
    torch.save(store, rl.dir / "heads.pt")
    i3_exam(rl, D, held)


# ================================================================ transfer

def lin_delta(hs: list[dict], Qx, Ox, own=None) -> np.ndarray:
    """Mean over the fold heads of the linear Delta, (n, 20, 2). `own` = ((mu, sd) Qwen, (mu, sd) op) of another
    dataset replaces each head's I3 statistics (and the pair head's zbar by that dataset's mean, i.e. 0)."""
    Qx, Ox = torch.as_tensor(Qx, device=DEV).double(), torch.as_tensor(Ox, device=DEV).double()
    out = 0
    for h in hs:
        sq, so = own if own else (h["q"], h["op"])
        Z = torch.cat([(Qx - sq[0].to(DEV).double()) / sq[1].to(DEV).double() / np.sqrt(Qx.shape[1]),
                       (Ox - so[0].to(DEV).double()) / so[1].to(DEV).double() / np.sqrt(Ox.shape[1])], 1)
        W = h["W"].to(DEV).double()
        out = out + ((Z - (0 if own else h["zbar"].to(DEV).double())) @ W if "zbar" in h else Z @ W + h["c"].to(DEV).double())
    return (out / len(hs)).float().cpu().numpy().reshape(-1, 20, 2)


def student_delta(store: dict, m: str, s: int, arm: str, Xop, Emb, own=None) -> np.ndarray:
    from .real_g0 import _mlp_cpu
    Xo, Ee = torch.as_tensor(Xop, dtype=torch.float32), torch.as_tensor(Emb, dtype=torch.float32)
    out = 0
    with torch.no_grad():
        for f in range(5):
            st = own or store["heads"][m, s, f, "student"]
            net = _mlp_cpu()
            net.load_state_dict(store["weights"][m, s, f"{m} f{f} {arm}", s])
            net.eval()
            out = out + net(_zs(Xo, Ee, {k: v.cpu() for k, v in st.items()}))
    return (out / 5).numpy().reshape(-1, 20, 2)


def g2_deltas(store: dict, m: str, Qx, Ox, Emb, own_lin=None, own_st=None) -> dict:
    """{(head, seed): Delta} for the five G2 heads x three seeds on foreign rows."""
    out = {}
    for s in SEEDS:
        for arm in LINEAR:
            out[f"M-C {arm}", s] = lin_delta([store["heads"][m, s, f, arm] for f in range(5)], Qx, Ox, own_lin)
        for arm in STUDENTS:
            out[f"student {arm}", s] = student_delta(store, m, s, arm, Ox, Emb, own_st)
    return out


def _taus(fit_run: str) -> dict:
    fl = pd.read_csv(data_dir() / fit_run / "i3_flip_rates.csv")
    return fl[fl.scope == "pooled"].set_index("examinee").tau_model.to_dict()


def run_transfer(rl, fit_run: str):
    """WOD readouts (elicit_e1.readouts unchanged) and navtest predictions + devkit jobs for the G2 heads."""
    from . import elicit_e1 as E1, elicit_e3 as E3, navsim_qwen as NQ, p5_pairs as P, real_g0 as G0
    from .real_g1 import NAV_PRIOR, E1_NAV
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
    store = torch.load(data_dir() / fit_run / "heads.pt", weights_only=False)
    taus = _taus(fit_run)
    # ---- WOD
    d = E1.wod_frames()
    fr, Emb = G0.load_embed("wod_val")
    Ew = Emb[pd.Series(np.arange(len(fr)), index=fr.frame_id).loc[d["frame_name"]].to_numpy()]
    tabs, acts = [], []
    for m in MODELS:
        gz = np.load(data_dir() / G1_WOD / f"wod_gates_{m}.npz", allow_pickle=True)
        assert (gz["frame_name"].astype(str) == d["frame_name"].astype(str)).all()
        own_lin, own_st = E1.wod_train_stats(m), G0._wod_train_stats(m)
        for stats in ("i3", "wod-train"):
            dl = g2_deltas(store, m, d["Q"], d[f"op {m}"], Ew, *((own_lin, own_st) if stats == "wod-train" else (None, None)))
            for (head, s), delta in dl.items():
                tau = taus[f"G2 {head} s{s} [{m}]"]
                gates = {"none": None} if stats != "i3" or head not in ("M-C pair", "student A") else {"none": None, "g2": gz["g2"]}
                for gname, g in gates.items():
                    tab, act = E1.readouts(d, d[f"prior {m}"], delta if g is None else g[:, None, None] * delta, tau)
                    tag = {"model": m, "head": head, "seed": s, "stats": stats, "gate": gname}
                    tabs.append(tab.assign(**tag))
                    acts.append(act.assign(**tag, tau=tau))
                if stats == "i3":
                    np.savez_compressed(rl.dir / f"wod_delta_{m}_{head.replace(' ', '_')}_s{s}.npz", delta=delta)
            log.info("WOD %s %s stats done", m, stats)
    pd.concat(tabs).to_csv(rl.dir / "wod_deltas.csv", index=False)
    pd.concat(acts).to_csv(rl.dir / "wod_activation.csv", index=False)
    del d
    # ---- NAVSIM (navtest)
    fr, Emb = G0.load_embed("navtest")
    sc = pd.read_csv(data_dir() / E1_NAV / "navtest_scopes.csv")
    fz = np.load(data_dir() / "runs/navsim_zs/index/navtest_future.npz")
    tok = sc.token.to_numpy()
    assert (fz["tokens"] == tok).all() and (fr.frame_id.to_numpy() == tok).all()
    fl = E3.cause_flags_nav(tok, fz["poses"][:, :, :2], E3.extract("navtest"))
    sc["vehicle_approach"] = fl.cause_vehicle.to_numpy()
    sc.to_csv(rl.dir / "navtest_scopes.csv", index=False)
    Qn = NQ.load("navtest", tok)["L18_last"]
    scopes = {"all": np.ones(len(tok), bool), "vehicle_approach": sc.vehicle_approach.to_numpy(),
              "ped_cyc_corridor": sc.ped_cyc_corridor.to_numpy(), "straight": sc.straight.to_numpy()}
    jobs, nacts = [], []
    for m in MODELS:
        z = np.load(data_dir() / "runs/navsim_zs/openpilot/navtest" / f"{m}_temporal.npz")
        p = np.load(data_dir() / NAV_PRIOR / f"navtest_ridge_late_{m}_temporal.npz")
        assert (z["tokens"] == tok).all() and (p["tokens"] == tok).all()
        dl = g2_deltas(store, m, Qn, z["temporal"].astype(np.float32), Emb)
        for (head, s), delta in dl.items():
            arm = p["poses"].copy()
            arm[..., :2] += delta[:, 1:16:2]                    # 0.5 ... 4.0 s, heading kept (E1)
            name = f"g2_{head.replace(' ', '')}_s{s}_{m}"
            tau = taus[f"G2 {head} s{s} [{m}]"]
            act = (np.abs(P.v2(E1._grid20(arm)) - P.v2(E1._grid20(p["poses"]))) >= tau).astype(float)
            mag = np.linalg.norm(delta[:, 1:16:2], axis=-1).mean(-1)
            for k, msk in scopes.items():
                nacts.append({"model": m, "head": head, "seed": s, "scope": k, "n": int(msk.sum()), "tau": tau,
                              "activation": float(act[msk].mean()), "delta_mag_median_m": float(np.median(mag[msk]))})
            if head in ("M-C pair", "student A") or s == 0:     # [G2] 10:50 (8): controls only seed 0
                np.savez(rl.dir / f"navtest_{name}.npz", tokens=tok, poses=arm.astype(np.float32))
                jobs.append(f"v1 navtest {name} {rl.dir / f'navtest_{name}.npz'}")
    pd.DataFrame(nacts).to_csv(rl.dir / "navsim_activation.csv", index=False)
    (rl.dir / "score_jobs.txt").write_text("\n".join(jobs) + "\n")
    log.info("NAVSIM: %d devkit jobs -> %s", len(jobs), rl.dir / "score_jobs.txt")


# ================================================================ G1c

def _const(g: np.ndarray, delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[G2] 10:50: c = sum g3 Delta / sum g3 over the dataset's evaluation frames; longitudinal only, and 2-D."""
    c = (g[:, None, None] * delta).sum(0) / g.sum()
    return c * np.array([1.0, 0.0], np.float32), c


def run_g1c_wod(rl):
    from . import elicit_e1 as E1
    from .real_g1 import E1_WOD, mc_taus
    d = E1.wod_frames()
    taus = mc_taus()
    tabs, acts, cs = [], [], []
    for m in MODELS:
        dz = np.load(data_dir() / E1_WOD / f"wod_delta_{m}.npz", allow_pickle=True)
        gz = np.load(data_dir() / G1_WOD / f"wod_gates_{m}.npz", allow_pickle=True)
        assert (dz["frame_name"].astype(str) == d["frame_name"].astype(str)).all()
        assert (gz["frame_name"].astype(str) == d["frame_name"].astype(str)).all()
        delta, g = dz["delta"], gz["g3"]
        c, c2 = _const(g, delta)
        cs.append({"model": m, "g3_sum": float(g.sum()), "g3_mean": float(g.mean()), "c_x_2s": float(c2[7, 0]),
                   "c_y_2s": float(c2[7, 1]), "c_x_4s": float(c2[15, 0]), "c_y_4s": float(c2[15, 1])})
        G = g[:, None, None]
        arms = {"g3 x M-C": (d[f"prior {m}"], G * delta), "g3 x const": (d[f"prior {m}"], G * c),
                "g3 x const2d": (d[f"prior {m}"], G * c2),
                "M-C - const (paired)": (d[f"prior {m}"] + G * c, G * (delta - c))}
        for name, (pr, dl) in arms.items():
            tab, act = E1.readouts(d, pr, dl, taus[m])
            tabs.append(tab.assign(model=m, arm=name))
            acts.append(act.assign(model=m, arm=name, tau=taus[m]))
    pd.concat(tabs).to_csv(rl.dir / "g1c_wod_deltas.csv", index=False)
    pd.concat(acts).to_csv(rl.dir / "g1c_wod_activation.csv", index=False)
    pd.DataFrame(cs).to_csv(rl.dir / "g1c_wod_const.csv", index=False)
    t = pd.concat(tabs)
    log.info("G1c WOD\n%s", t[t.judge == "RFS (rater)"].to_markdown(index=False, floatfmt=".3f"))


def run_g1c_nav_write(rl):
    """prior + g3 x c on navtest for the devkit (v1 PDMS and v2 EPDMS for the main constant, v1 for the 2-D one)."""
    from . import elicit_e1 as E1, navsim_zs as Z, p5_pairs as P
    from .real_g1 import E1_NAV, NAV_PRIOR, g3, mc_taus, nav_lead
    idx = Z.load_index("navtest", slim=True)
    tok = np.array([e["token"] for e in idx])
    v_ego = np.array([np.linalg.norm(e["vel"][-1]) for e in idx])
    taus = mc_taus()
    jobs, cs, acts = [], [], []
    for m in MODELS:
        dz = np.load(data_dir() / E1_NAV / f"navtest_delta_{m}.npz")
        p = np.load(data_dir() / NAV_PRIOR / f"navtest_ridge_late_{m}_temporal.npz")
        ld = nav_lead(m)
        at = pd.Series(np.arange(len(ld["tokens"])), index=ld["tokens"]).reindex(tok).astype(int).to_numpy()
        assert (dz["tokens"] == tok).all() and (p["tokens"] == tok).all()
        g = g3(ld["lead"][at], ld["lead_prob"][at], v_ego)
        c, c2 = _const(g, dz["delta"])
        cs.append({"model": m, "g3_sum": float(g.sum()), "g3_mean": float(g.mean()), "g3_open_001": float((g > 0.01).mean()),
                   "c_x_2s": float(c2[7, 0]), "c_y_2s": float(c2[7, 1]), "c_x_4s": float(c2[15, 0]), "c_y_4s": float(c2[15, 1])})
        np.savez(rl.dir / f"navtest_g3_{m}.npz", tokens=tok, g3=g)
        for name, cc, vers in (("const", c, ("v1", "v2")), ("const2d", c2, ("v1",))):
            arm = p["poses"].copy()
            arm[..., :2] += g[:, None, None] * cc[None, 1:16:2]
            nm = f"g1c_g3_{name}_ridge_late_{m}"
            np.savez(rl.dir / f"navtest_{nm}.npz", tokens=tok, poses=arm.astype(np.float32))
            jobs += [f"{v} navtest {nm} {rl.dir / f'navtest_{nm}.npz'}" for v in vers]
            act = (np.abs(P.v2(E1._grid20(arm)) - P.v2(E1._grid20(p["poses"]))) >= taus[m]).astype(float)
            acts.append({"model": m, "arm": name, "activation_all": float(act.mean())})
    pd.DataFrame(cs).to_csv(rl.dir / "g1c_nav_const.csv", index=False)
    pd.DataFrame(acts).to_csv(rl.dir / "g1c_nav_activation.csv", index=False)
    (rl.dir / "score_jobs.txt").write_text("\n".join(jobs) + "\n")
    log.info("G1c NAVSIM: %s\n%d devkit jobs -> %s", pd.DataFrame(cs).to_markdown(index=False, floatfmt=".3f"), len(jobs),
             rl.dir / "score_jobs.txt")


# ================================================================ NAVSIM tables

def _scores(ver: str, name: str):
    from .openloop_standing import _latest
    df = _latest(ver, "navtest", name)
    if df is None:
        return None
    return df[df["token"].str.fullmatch(r"[0-9a-f]{16,17}") & df["valid"].astype(bool)].set_index("token")["score"].astype(float)


def nav_paired(pairs: dict, groups: dict, metrics=(("v1", "PDMS"), ("v2", "EPDMS"))) -> pd.DataFrame:
    """elicit_e1.navsim_table's paired token bootstrap (10 000, rng 0) for {label: (arm devkit name, reference name)}."""
    rows = []
    rng = np.random.default_rng(0)
    for label, (arm, ref) in pairs.items():
        for ver, metric in metrics:
            a, b = _scores(ver, arm), _scores(ver, ref)
            if a is None or b is None:
                continue
            x, y = a.align(b, join="inner")
            for g, msk in groups.items():
                keep = np.ones(len(x), bool) if msk is None else msk.reindex(x.index).fillna(False).to_numpy(bool)
                dd = (x - y).to_numpy()[keep]
                bs = dd[rng.integers(0, len(dd), (10000, len(dd)))].mean(1)
                rows.append({"arm": label, "metric": metric, "group": g, "n": len(dd), "ref_score": 100 * y.to_numpy()[keep].mean(),
                             "arm_score": 100 * x.to_numpy()[keep].mean(), "delta": 100 * dd.mean(),
                             "lo": 100 * np.percentile(bs, 2.5), "hi": 100 * np.percentile(bs, 97.5)})
    return pd.DataFrame(rows)


def run_nav_table(rl, transfer_run: str, g1c_run: str):
    sc = pd.read_csv(data_dir() / transfer_run / "navtest_scopes.csv").set_index("token")
    groups = {"all": None, "vehicle_approach": sc.vehicle_approach, "ped_cyc_corridor": sc.ped_cyc_corridor,
              "straight": sc.straight}
    out = []
    # G1c: arm vs prior, M-C vs const, on all / g3-open tokens
    for m in MODELS:
        gz = np.load(data_dir() / g1c_run / f"navtest_g3_{m}.npz")
        gopen = pd.Series(gz["g3"] > 0.01, index=gz["tokens"])
        gg = {**groups, "g3_open": gopen}
        prior = f"heads_ridge_late_{m}_temporal"
        mc = f"g1_mc_g3_ridge_late_{m}"
        pairs = {f"G1c g3 x M-C {m}": (mc, prior), f"G1c g3 x const {m}": (f"g1c_g3_const_ridge_late_{m}", prior),
                 f"G1c g3 x const2d {m}": (f"g1c_g3_const2d_ridge_late_{m}", prior),
                 f"G1c M-C - const {m}": (mc, f"g1c_g3_const_ridge_late_{m}"),
                 f"G1c M-C - const2d {m}": (mc, f"g1c_g3_const2d_ridge_late_{m}")}
        out.append(nav_paired(pairs, gg))
        g2p = {f"G2 {h} s{s} {m}": (f"g2_{h.replace(' ', '')}_s{s}_{m}", prior)
               for h in ("M-C pair", "student A", "M-C hard", "M-C uniform") for s in SEEDS}
        out.append(nav_paired(g2p, groups, (("v1", "PDMS"),)))
    t = pd.concat(out)
    t.to_csv(rl.dir / "navsim_paired.csv", index=False)
    log.info("navtest\n%s", t.to_markdown(index=False, floatfmt=".2f"))


# ================================================================ summary tables and figure

def _ci(r, k=2, scale=1.0):
    return f"{scale * r.delta:+.{k}f} [{scale * r.lo:+.{k}f}, {scale * r.hi:+.{k}f}]"


def summarize(out, fit_run: str, transfer_run: str, g1c_wod_run: str, navtab_run: str):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    D = data_dir()
    fl = pd.read_csv(D / fit_run / "i3_flip_rates.csv")
    pv = pd.read_csv(D / fit_run / "i3_paired.csv")
    wd = pd.read_csv(D / transfer_run / "wod_deltas.csv")
    wa = pd.read_csv(D / transfer_run / "wod_activation.csv")
    na = pd.read_csv(D / transfer_run / "navsim_activation.csv")
    nt = pd.read_csv(D / navtab_run / "navsim_paired.csv")
    for f in ("i3_flip_rates.csv", "i3_paired.csv", "i3_g1c_const.csv"):
        pd.read_csv(D / fit_run / f).to_csv(out / f, index=False)
    for f in ("wod_deltas.csv", "wod_activation.csv", "navsim_activation.csv"):
        pd.read_csv(D / transfer_run / f).to_csv(out / f, index=False)
    nt.to_csv(out / "navsim_paired.csv", index=False)
    for f in ("g1c_wod_deltas.csv", "g1c_wod_activation.csv", "g1c_wod_const.csv"):
        pd.read_csv(D / g1c_wod_run / f).to_csv(out / f, index=False)
    fp = fl[fl.scope == "pooled"].set_index("examinee")
    ref = {m: fp.loc[f"CARLA M-C pair [{m}]", "flip_rate"] for m in MODELS}
    rows, verd = [], []
    for m in MODELS:
        for head in ("M-C pair", "M-C hard", "M-C uniform", "student A", "student B"):
            for s in SEEDS:
                ex = f"G2 {head} s{s} [{m}]"
                r = fp.loc[ex]
                sel = lambda t: t[(t.model == m) & (t.head == head) & (t.seed == s) & (t.stats == "i3") & (t.gate == "none")]  # noqa: E731
                w = sel(wd)
                w = w[w.judge == "RFS (rater)"].set_index("scope")
                a = sel(wa).set_index("scope").activation
                pp = pv[(pv.examinee == ex) & (pv.scope == "pooled")].set_index("vs")
                i3_ok = r.flip_rate >= ref[m] and r.false_flip_null_oos <= NULL_FF_MAX
                wod_ok = not w.loc["Cut_ins", "hi"] < 0 and a["straight_yaw"] <= ACT_HARM
                why = [] if i3_ok else [f"I3 flip {100 * r.flip_rate:.1f}% < {100 * ref[m]:.1f}%" if r.flip_rate < ref[m] else
                                        f"null {100 * r.false_flip_null_oos:.1f}% > 7%"]
                why += [] if wod_ok else ["WOD Cut_ins CI < 0" if w.loc["Cut_ins", "hi"] < 0 else "straight activation > 7%"]
                n = na[(na.model == m) & (na.head == head) & (na.seed == s)].set_index("scope").activation
                nv = nt[(nt.arm == f"G2 {head} s{s} {m}") & (nt.metric == "PDMS")].set_index("group")
                rows.append({"model": m, "head": head, "seed": s, "tau": f"{r.tau_model:.2f}",
                             "I3 flip": f"{100 * r.flip_rate:.1f} [{100 * r.flip_lo:.1f}, {100 * r.flip_hi:.1f}]",
                             "vs prior (pp)": _ci(pp.loc[f"ridge_late op-{m} temporal"], 1, 100),
                             "vs CARLA M-C (pp)": _ci(pp.loc[f"CARLA M-C pair [{m}]"], 1, 100),
                             "null ff": f"{100 * r.false_flip_null_oos:.1f}%", "non-reactive ff": f"{100 * r.false_flip_nonreactive:.1f}%",
                             "WOD RFS all": _ci(w.loc["all"]), "WOD RFS Cut_ins": _ci(w.loc["Cut_ins"]),
                             "WOD RFS Ped.": _ci(w.loc["Pedestrians"]), "WOD act straight": f"{100 * a['straight_yaw']:.1f}%",
                             "WOD Delta med (m)": f"{sel(wa).set_index('scope').delta_mag_median_m['all']:.2f}",
                             "NAV PDMS all": _ci(nv.loc["all"]) if "all" in nv.index else "",
                             "NAV PDMS veh.": _ci(nv.loc["vehicle_approach"]) if "all" in nv.index else "",
                             "NAV act straight": f"{100 * n['straight']:.1f}%",
                             "verdict": "pass" if i3_ok and wod_ok else "fail: " + "; ".join(why)})
                verd.append({"model": m, "head": head, "seed": s, "i3_ok": i3_ok, "wod_ok": wod_ok, "pass": i3_ok and wod_ok})
    t = pd.DataFrame(rows)
    t.to_csv(out / "g2_table.csv", index=False)
    (out / "g2_table.md").write_text(t.to_markdown(index=False))
    pd.DataFrame(verd).to_csv(out / "g2_verdict.csv", index=False)
    # CARLA-trained references on I3
    cr = []
    for ex, r in fp.iterrows():
        if ex.startswith(("CARLA", "ridge_late", "G1c")) or " x g2 " in ex:
            pp = pv[(pv.examinee == ex) & (pv.scope == "pooled")].set_index("vs")
            m = ex.split("[")[-1].rstrip("]") if "[" in ex else ex.split()[1].replace("op-", "")
            cr.append({"examinee": ex, "tau": f"{r.tau_model:.2f}",
                       "I3 flip": f"{100 * r.flip_rate:.1f} [{100 * r.flip_lo:.1f}, {100 * r.flip_hi:.1f}]",
                       "vs prior (pp)": _ci(pp.loc[f"ridge_late op-{m} temporal"], 1, 100) if f"ridge_late op-{m} temporal" in pp.index else "",
                       "null ff": f"{100 * r.false_flip_null_oos:.1f}%", "non-reactive ff": f"{100 * r.false_flip_nonreactive:.1f}%"})
    pd.DataFrame(cr).to_csv(out / "i3_reference.csv", index=False)
    (out / "i3_reference.md").write_text(pd.DataFrame(cr).to_markdown(index=False))


def figs(res_dir, out_dir, model: str = "cinque"):
    """G2, Cinque: I3 held-out flips (CARLA-trained vs I3-trained heads), WOD Cut_ins and all-frame RFS delta, WOD
    straight activation, NAVSIM PDMS delta on vehicle-approach tokens; seed 0 with CI, seeds 1-2 as crosses."""
    import matplotlib.pyplot as plt
    from . import plots
    res_dir = Path(res_dir)
    fl = pd.read_csv(res_dir / "i3_flip_rates.csv")
    fl = fl[fl.scope == "pooled"].set_index("examinee")
    wd, wa = pd.read_csv(res_dir / "wod_deltas.csv"), pd.read_csv(res_dir / "wod_activation.csv")
    nt = pd.read_csv(res_dir / "navsim_paired.csv")
    heads = [("M-C pair", "M-C pair"), ("M-C hard", "Hard-ex."), ("M-C uniform", "Uniform"), ("student A", "Student A"),
             ("student B", "Student B")]
    x = np.arange(len(heads))
    c_carla, c_i3 = "0.55", plots.OKABE_ITO[5]
    with plots.mpl.rc_context(plots.STYLE):
        fig, ax = plt.subplots(1, 4, figsize=(plots.PAGE, 2.0))
        for i, (h, _) in enumerate(heads):
            cex = f"CARLA {'M-C ' + h.split()[-1] if h.startswith('M-C') else h + ' s0'} [{model}]"
            r = fl.loc[cex]
            ax[0].errorbar(i - 0.15, 100 * r.flip_rate, yerr=[[100 * (r.flip_rate - r.flip_lo)], [100 * (r.flip_hi - r.flip_rate)]],
                           fmt="o", mfc="white", color=c_carla, ms=3, lw=0.8, capsize=1.2, label="CARLA-trained" if i == 0 else None)
            for s in SEEDS:
                r = fl.loc[f"G2 {h} s{s} [{model}]"]
                if s == 0:
                    ax[0].errorbar(i + 0.1, 100 * r.flip_rate, yerr=[[100 * (r.flip_rate - r.flip_lo)], [100 * (r.flip_hi - r.flip_rate)]],
                                   fmt="o", color=c_i3, ms=3, lw=0.8, capsize=1.2, label="I3-trained (held-out scenes)" if i == 0 else None)
                else:
                    ax[0].plot(i + 0.2, 100 * r.flip_rate, "x", color=c_i3, ms=3, mew=0.7)
            for k, (scope, col) in enumerate((("Cut_ins", c_i3), ("all", plots.OKABE_ITO[6]))):
                for s in SEEDS:
                    w = wd[(wd.model == model) & (wd["head"] == h) & (wd.seed == s) & (wd.stats == "i3") & (wd.gate == "none")
                           & (wd.judge == "RFS (rater)") & (wd.scope == scope)].iloc[0]
                    xx = i + (k - 0.5) * 0.3 + (0 if s == 0 else 0.08)
                    if s == 0:
                        ax[1].errorbar(xx, w.delta, yerr=[[w.delta - w.lo], [w.hi - w.delta]], fmt="o", color=col, ms=3, lw=0.8,
                                       capsize=1.2, label={"Cut_ins": "Cut-ins (20)", "all": "All rater (478)"}[scope] if i == 0 else None)
                    else:
                        ax[1].plot(xx, w.delta, "x", color=col, ms=3, mew=0.7)
            for s in SEEDS:
                a = wa[(wa.model == model) & (wa["head"] == h) & (wa.seed == s) & (wa.stats == "i3") & (wa.gate == "none")
                       & (wa.scope == "straight_yaw")].iloc[0]
                ax[2].plot(i + 0.08 * s, 100 * a.activation, "o" if s == 0 else "x", color=c_i3, ms=3, mew=0.7)
                n = nt[(nt.arm == f"G2 {h} s{s} {model}") & (nt.metric == "PDMS") & (nt.group == "vehicle_approach")]
                if len(n):
                    n = n.iloc[0]
                    if s == 0:
                        ax[3].errorbar(i, n.delta, yerr=[[n.delta - n.lo], [n.hi - n.delta]], fmt="o", color=c_i3, ms=3, lw=0.8, capsize=1.2)
                    else:
                        ax[3].plot(i + 0.1, n.delta, "x", color=c_i3, ms=3, mew=0.7)
        ref = 100 * fl.loc[f"CARLA M-C pair [{model}]", "flip_rate"]
        ax[0].axhline(ref, color="0.3", ls="--", lw=0.7)
        ax[0].axhline(100 * fl.loc[f"ridge_late op-{model} temporal", "flip_rate"], color="0.3", ls=":", lw=0.7)
        ax[2].axhline(100 * ACT_HARM, color="0.3", ls="--", lw=0.7)
        for a_, yl in zip(ax, ("I3 flip rate (%)", r"WOD $\Delta$RFS", "WOD straight activation (%)",
                               r"NAVSIM $\Delta$PDMS, approaching veh.")):
            if a_ is not ax[0]:
                a_.axhline(0, color="0.5", lw=0.6)
            a_.set_xticks(x, [n for _, n in heads], rotation=35, ha="right")
            a_.set_ylabel(yl)
        ax[0].legend(loc="lower left", fontsize=6)
        ax[1].legend(loc="lower left", fontsize=6)
        fig.tight_layout(w_pad=0.6)
        plots.save(fig, Path(out_dir), "real-g2-i3-retrain")


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("fit", "exam", "transfer", "g1c-wod", "g1c-nav-write", "nav-table", "summary", "figs"))
    ap.add_argument("--fit-run", default="", help="transfer: the exam run (merged heads.pt, i3_flip_rates.csv)")
    ap.add_argument("--fit-runs", default="", help="exam: comma list of fit shard runs")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--transfer-run", default="")
    ap.add_argument("--g1c-run", default="", help="the g1c-nav-write run dir")
    ap.add_argument("--runs", default="", help="summary: fit,transfer,g1c-wod,nav-table run dirs")
    ap.add_argument("--out", default="research/results/real-data-transfer/g2")
    ap.add_argument("--fig-out", default="research/figs")
    a = ap.parse_args()
    if a.what == "summary":
        summarize(a.out, *a.runs.split(","))
        return
    if a.what == "figs":
        figs(a.out, a.fig_out)
        return
    rl = RunLog("real-data-transfer", f"g2-{a.what}")
    if a.what == "fit":
        run_fit(rl, tuple(a.models.split(",")), tuple(int(x) for x in a.seeds.split(",")))
    elif a.what == "exam":
        run_exam(rl, a.fit_runs.split(","))
    elif a.what == "transfer":
        run_transfer(rl, a.fit_run)
    elif a.what == "g1c-wod":
        run_g1c_wod(rl)
    elif a.what == "g1c-nav-write":
        run_g1c_nav_write(rl)
    elif a.what == "nav-table":
        run_nav_table(rl, a.transfer_run, a.g1c_run)
    rl.close()


if __name__ == "__main__":
    main()
