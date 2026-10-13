#!/usr/bin/env python3
"""CRITIC-PILOT train + read (experiments/route_follow/plans/2026-10-13-critic-pilot-prereg.md): the five arms, the held reads (L1, L2),
the counterfactual flip (L3), the blind arm on the RBSG2 reads and the gradient sanity read. envs/op-train, one GPU, a pool job:

  .venv/bin/python -m jevdrive.cl submit --owner critic-pilot --name cp-train --vram 20 --cpu 8 --ram 60 \
      --log-dir $DATA_DIR/runs/critic_pilot/pool -- $DATA_DIR/envs/op-train/bin/python experiments/critic_pilot/scripts/cp_train.py [--arms C2 ...]
  --smoke: 2 epochs x 30 steps, B = 200 (code path only)

Inputs: $DATA_DIR/runs/critic_pilot/labels/<name>.npz (cp_labels.py), the pools on1_<name> (tab, route, agent, road, @warp/front.npy),
rows/<name>/meta.npz (rmem). Out: run dir (ckpt per arm, preds.npz) and experiments/critic_pilot/results/{critic_pilot.json,critic_pilot.md}.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from queue import Queue

_R = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "experiments/critic_pilot/lib"), str(_R / "experiments/route_mem/lib"),
                str(_R / "experiments/body1/lib"), str(_R / "experiments/op_parity/scripts")]
import numpy as np  # noqa: E402

TRAIN = ("L18RB0", "L18RSG", "L18RBSG")
HELD = "L18RSGallheld"
READS = ("RBSG2-L18-s0", "RBSG2-L18-s1", "RSG-L18-s0")
ARMS = dict(C2=dict(pools=("L18RB0", "L18RBSG")), C3=dict(pools=TRAIN), **{"C2-blind": dict(pools=("L18RB0", "L18RBSG"), vision=False)},
            **{"C2-noroute": dict(pools=("L18RB0", "L18RBSG"), route=False)}, **{"C2-aug": dict(pools=("L18RB0", "L18RBSG"), aug=3.0)})
TYPES = ("collision", "offroad", "corridor")
MIN_POS = 20


# ---------------------------------------------------------------- data
def load_pool(D, name, tokens: bool):
    L = dict(np.load(D / f"runs/critic_pilot/labels/{name}.npz"))
    P = D / f"runs/op_parity/cache/on1_{name}"
    tb, rt = np.load(P / "tab.npz"), np.load(P / "route.npz")
    M = np.load(D / f"runs/onpol_nuplan/rows/{name}/meta.npz")
    assert (L["names"] == tb["names"]).all() and (M["names"] == tb["names"]).all()
    rows = np.flatnonzero(L["keep"] & L["labelled"] & M["rmem_ok"])
    d = dict(name=name, rows=rows, names=tb["names"][rows], log=L["log"][rows], scene=L["scene"][rows], ego=tb["ego"][rows],
             rmem=M["rmem"][rows].astype(np.float32), plan=tb["served"][rows].astype(np.float32),
             poly=np.where(rt["ok"][rows, None, None], rt["poly"][rows], np.nan).astype(np.float32), k=M["k"][rows],
             **{k: L[k][rows].astype(np.float32) for k in ("ev", "anyz", "spin", "lat2", "q2", "prog4", "wzpk", "b_now", "b_plan", "b_log")})
    if tokens:
        t0 = time.time()
        f = np.load(P.parent / f"on1_{name}@warp/front.npy", mmap_mode="r")
        d["front"] = np.empty((len(rows), 8, 32, 512), np.float16)
        for i in range(0, len(rows), 2048):
            d["front"][i:i + 2048] = f[rows[i:i + 2048]]
        print(f"{name}: {len(rows)} rows, tokens {d['front'].nbytes / 1e9:.1f} GB in {time.time() - t0:.0f} s", flush=True)
    return d


def cat(ds):
    out = {}
    for k in ds[0]:
        if k == "name":
            continue
        if all(k in d for d in ds):
            out[k] = np.concatenate([d[k] for d in ds])
    out["pool"] = np.concatenate([[d["name"]] * len(d["rows"]) for d in ds])
    return out


def load_reads(D, tag):
    L = dict(np.load(D / f"runs/critic_pilot/labels/{tag}.npz"))
    m = L["keep"] & L["labelled"]
    return dict(names=L["names"][m], log=L["log"][m], ego=L["ego"][m], rmem=L["rmem"][m], plan=L["served"][m].astype(np.float32),
                **{k: L[k][m].astype(np.float32) for k in ("ev", "anyz", "spin", "prog4", "wzpk")},
                lat2=np.full(m.sum(), np.nan, np.float32))


LABS = ("ev", "anyz", "spin", "lat2", "prog4", "wzpk")


def batches(d, idx, bs, shuffle, rng, vision, dev):
    """Background-thread batch builder: numpy gather -> pinned -> GPU."""
    import torch
    order = rng.permutation(idx) if shuffle else idx
    q = Queue(maxsize=3)

    def work():
        for i in range(0, len(order), bs):
            r = np.sort(order[i:i + bs])
            b = {k: torch.from_numpy(d[k][r]) for k in ("ego", "rmem", "plan", "poly", "q2", *LABS) if k in d}
            if vision:
                b["front"] = torch.from_numpy(d["front"][r])
            b = {k: (v.pin_memory().to(dev, non_blocking=True) if dev.type == "cuda" else v) for k, v in b.items()}
            b["idx"] = r
            q.put(b)
        q.put(None)
    threading.Thread(target=work, daemon=True).start()
    while (b := q.get()) is not None:
        yield b


# ---------------------------------------------------------------- metrics
def auc(y, s):
    y, s = np.asarray(y), np.asarray(s, float)
    m = np.isfinite(y) & np.isfinite(s)
    y, s = y[m] > 0.5, s[m]
    npos, nneg = y.sum(), (~y).sum()
    if npos == 0 or nneg == 0:
        return np.nan
    from scipy.stats import rankdata
    r = rankdata(s)
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def r2(y, p):
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    return float(1 - ((y - p) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)) if len(y) > 2 else np.nan


def boot(fn, y, s, log, B, seed=0):
    """Log-clustered bootstrap of fn(y, s): point, 2.5 / 97.5 % percentiles."""
    m = np.isfinite(y) & np.isfinite(s)
    y, s, log = y[m], s[m], log[m]
    u, inv = np.unique(log, return_inverse=True)
    groups = [np.flatnonzero(inv == j) for j in range(len(u))]
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(B):
        g = np.concatenate([groups[j] for j in rng.integers(0, len(u), len(u))])
        v.append(fn(y[g], s[g]))
    v = np.array(v, float)
    return dict(v=fn(y, s), lo=float(np.nanquantile(v, 0.025)), hi=float(np.nanquantile(v, 0.975)), n=int(len(y)),
                pos=int((y > 0.5).sum()) if fn is auc else None, logs=int(len(u)), B=B)


def read(d, P, B_line, B_desc):
    """Predictions P (dict of numpy) on test set d -> metrics dict."""
    out = {}
    lg = d["log"]
    out["any2"] = boot(auc, d["anyz"][:, 0], P["anyz"][:, 0], lg, B_line)
    out["any4"] = boot(auc, d["anyz"][:, 1], P["anyz"][:, 1], lg, B_desc)
    for a, t in enumerate(TYPES):
        for b, T in enumerate((2, 4)):
            y = d["ev"][:, a, b]
            if np.nansum(y) >= MIN_POS:
                out[f"{t}{T}"] = boot(auc, y, P["ev"][:, a, b], lg, B_desc)
            else:
                out[f"{t}{T}"] = dict(pos=int(np.nansum(y)), note=f"< {MIN_POS} positive rows, not read")
    if np.isfinite(d["lat2"]).sum() > 10:
        out["lat2_r2"] = boot(r2, d["lat2"].clip(-10, 10), P["lat2"], lg, B_line)
        out["lat2_mae"] = float(np.nanmean(np.abs(d["lat2"].clip(-10, 10) - P["lat2"])))
        out["abs_lat2_r2"] = boot(r2, np.abs(d["lat2"].clip(-10, 10)), np.abs(P["lat2"]), lg, B_desc)
    out["prog4_r2"] = boot(r2, d["prog4"], P["prog4"], lg, B_desc)
    out["wzpk_r2"] = boot(r2, d["wzpk"], P["wzpk"], lg, B_desc)
    out["spin_pos"] = int(np.nansum(d["spin"]))
    return out


def baselines(d, B):
    """Same-row hand-rule references (not lines)."""
    out = {}
    for b in ("b_plan", "b_now", "b_log"):
        if b in d:
            out[b] = {f"any{T}": boot(auc, d["anyz"][:, j], d[b], d["log"], B)["v"] for j, T in enumerate((2, 4))}
            out[b] |= {f"{t}{T}": auc(d["ev"][:, a, j], d[b]) for a, t in enumerate(TYPES) for j, T in enumerate((2, 4))}
    return out


# ---------------------------------------------------------------- train / predict
def predict(model, d, vision, dev, bs=512):
    import torch
    model.eval()
    P = {k: [] for k in ("ev", "anyz", "spin", "lat2", "prog4", "wzpk")}
    with torch.no_grad(), torch.autocast(dev.type, torch.bfloat16):
        for b in batches(d, np.arange(len(d["ego"])), bs, False, None, vision, dev):
            o = model(b.get("front"), b["ego"], b["rmem"], b["plan"])
            for k in P:
                P[k].append(o[k].float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in P.items()}


def train_arm(arm, cfg, data, a, dev, run, out_dir):
    import torch
    import critic as CR
    vision, route, aug = cfg.get("vision", True), cfg.get("route", True), cfg.get("aug", 0.0)
    m = np.isin(data["pool"], cfg["pools"])
    logs = np.unique(data["log"][m])
    rng = np.random.default_rng(0)
    vlogs = set(rng.permutation(logs)[: max(1, len(logs) // 10)].tolist())
    isv = np.array([g in vlogs for g in data["log"]])
    tr, va = np.flatnonzero(m & ~isv), np.flatnonzero(m & isv)
    torch.manual_seed(0)
    model = CR.Critic(vision=vision, route=route).to(dev)
    npar = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    steps_ep = (a.steps if a.smoke else len(tr) // a.bs)
    total = steps_ep * a.epochs
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=total, pct_start=0.05)
    run.info(f"{arm}: {npar / 1e6:.2f} M params, train {len(tr)} rows / val {len(va)} rows ({len(vlogs)} logs), {total} steps")
    best, best_ep, step, bad = np.inf, -1, 0, 0
    dv = {k: v[va] for k, v in data.items() if len(v) == len(data["ego"]) and k != "pool"}
    for ep in range(a.epochs):
        model.train()
        t0 = time.time()
        for j, b in enumerate(batches(data, tr, a.bs, True, rng, vision, dev)):
            if j >= steps_ep:
                continue                                             # drain the producer thread
            lab = {k: b[k] for k in LABS}
            rmem = b["rmem"]
            if aug > 0:                                              # route shift: memory moved sideways by delta, lat2 relabelled exactly
                n = len(rmem)
                sh = (torch.rand(n, device=dev) < 0.5)
                dl = (torch.rand(n, device=dev) * 2 - 1) * aug * sh
                R = rmem.view(n, 40, 5).clone()
                R[..., 2] = torch.where(R[..., 0] > 0.5, R[..., 2] + dl[:, None] / 20, R[..., 2])
                rmem = R.view(n, -1)
                poly = b["poly"] + torch.stack([torch.zeros_like(dl), dl], -1)[:, None]
                ok = torch.isfinite(poly).all(-1).all(-1) & torch.isfinite(b["q2"]).all(-1) & torch.isfinite(lab["lat2"])
                new = torch.full_like(lab["lat2"], float("nan"))
                if ok.any():
                    new[ok] = CR.signed_off(poly[ok], b["q2"][ok])
                lab["lat2"] = torch.where(sh, new, lab["lat2"])
                for k in ("ev", "anyz", "spin", "prog4", "wzpk"):
                    s = sh.view(-1, *([1] * (lab[k].dim() - 1)))
                    lab[k] = torch.where(s, torch.full_like(lab[k], float("nan")), lab[k])
            with torch.autocast(dev.type, torch.bfloat16):
                o = model(b.get("front"), b["ego"], rmem, b["plan"])
            L, parts = CR.loss(o, lab)
            opt.zero_grad(set_to_none=True)
            L.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            step += 1
            if step % 50 == 0:
                run.scalar(f"{arm}/loss", float(L), step)
                for k, v in parts.items():
                    run.scalar(f"{arm}/{k}", float(v), step)
        Pv = predict(model, dv, vision, dev)
        with torch.no_grad():
            vl = float(CR.loss({k: torch.from_numpy(v) for k, v in Pv.items()}, {k: torch.from_numpy(dv[k]) for k in LABS})[0])
        va_auc, va_r2 = auc(dv["anyz"][:, 0], Pv["anyz"][:, 0]), r2(dv["lat2"].clip(-10, 10), Pv["lat2"])
        run.info(f"{arm} ep {ep}: train loss {float(L):.4f}, val loss {vl:.4f}, val any2 AUC {va_auc:.3f}, val lat2 R2 {va_r2:.3f} ({time.time() - t0:.0f} s)")
        run.scalar(f"{arm}/val_loss", vl, ep)
        if vl < best:
            best, best_ep, bad = vl, ep, 0
            torch.save(model.state_dict(), out_dir / f"{arm}.pt")
        else:
            bad += 1
            if bad >= 3:
                break
    model.load_state_dict(torch.load(out_dir / f"{arm}.pt"))
    return model, dict(params=npar, train_rows=int(len(tr)), val_rows=int(len(va)), best_epoch=best_ep, best_val=best)


# ---------------------------------------------------------------- counterfactual flip (L3)
def cf_rows(D):
    """on1_CFdev junction rows exactly as rf_gates.py pairs them: real ego + real memory + real polyline, cf ego + cf memory + cf polyline."""
    CRD, FEAT, ROWS = D / "runs/op_parity/cache", D / "runs/route_mem/feat", D / "runs/route_follow/cf/rows"
    pool = "on1_CFdev"
    tb, rt = dict(np.load(CRD / pool / "tab.npz")), dict(np.load(CRD / pool / "route.npz"))
    fz = np.load(FEAT / f"{pool}.npz")
    names = tb["names"]
    assert (rt["tokens"] == names).all() and (fz["names"] == names).all()
    tok = np.array([n.split(":")[2] for n in names])
    cls = np.array([n.split(":")[3] for n in names])
    n = len(names)
    f_real, ego_real, Rr, okr = np.zeros((n, 200), np.float32), np.zeros((n, 20), np.float32), np.full((n, 51, 2), np.nan), np.zeros(n, bool)
    key = {(t, c): i for i, (t, c) in enumerate(zip(tok, cls))}
    for rf in sorted(ROWS.glob("*.npz")):
        Z = np.load(rf)
        hit = [(i, key[t, c]) for i, (t, c) in enumerate(zip(Z["token"].tolist(), Z["cls"].tolist())) if (t, c) in key]
        if not hit:
            continue
        zi, pj = map(np.array, zip(*hit))
        src = Z["row"][zi]
        st, ft = np.load(CRD / rf.stem / "tab.npz"), np.load(FEAT / f"{rf.stem}.npz")
        assert (st["names"][src] == tok[pj]).all() and (ft["names"][src] == tok[pj]).all()
        ego_real[pj], f_real[pj] = st["ego"][src], ft["f"][src].astype(np.float32)
        Rr[pj], okr[pj] = Z["real"][zi], ft["ok"][src] & (np.isfinite(Z["real"][zi]).all(-1).sum(-1) >= 2)
    ok = rt["ok"] & fz["ok"] & okr & np.isin(cls, ["j0", "j1"])
    front = np.load(CRD / f"{pool}@warp/front.npy", mmap_mode="r")
    r = np.flatnonzero(ok)
    return dict(rows=r, front=np.asarray(front[r]), ego_real=ego_real[r], f_real=f_real[r], ego_cf=tb["ego"][r], f_cf=fz["f"][r].astype(np.float32),
                Rr=Rr[r], C=rt["poly"][r].astype(np.float64), fut=tb["fut"][r], log=tb["log"][r], cls=cls[r])


def along(Pl, s):
    """Polyline Pl (51, 2) on the -10 .. +90 m arc grid (ego projection near arc 0) -> poses (len(s), 3) at arc lengths s from the ego's
    projection (nearest vertex to the origin)."""
    Pl = Pl[np.isfinite(Pl).all(1)]
    arc = np.r_[0, np.cumsum(np.hypot(*np.diff(Pl, axis=0).T))]
    j = int(np.argmin(np.hypot(*Pl.T)))
    a = arc[j] + np.asarray(s)
    x, y = np.interp(a, arc, Pl[:, 0]), np.interp(a, arc, Pl[:, 1])
    hx, hy = np.interp(a + 1, arc, Pl[:, 0]) - np.interp(a - 1, arc, Pl[:, 0]), np.interp(a + 1, arc, Pl[:, 1]) - np.interp(a - 1, arc, Pl[:, 1])
    return np.c_[x, y, np.arctan2(hy, hx)].astype(np.float32)


def flip_read(models, cf, mu, sd, dev):
    import torch
    arcs = np.cumsum(np.hypot(*np.diff(np.concatenate([np.zeros((len(cf["fut"]), 1, 2)), cf["fut"][..., :2]], 1), axis=1).transpose(2, 0, 1)), axis=1)
    Pr = np.stack([along(R, a) for R, a in zip(cf["Rr"], arcs)])
    Pc = np.stack([along(C, a) for C, a in zip(cf["C"], arcs)])
    gap2 = np.hypot(*(Pr[:, 3, :2] - Pc[:, 3, :2]).T)
    sel = gap2 >= 1.0
    res = dict(junction_rows=int(len(gap2)), rows_gap2_ge_1m=int(sel.sum()), gap2_median=float(np.median(gap2)))
    for arm, (model, vision) in models.items():
        def lat(ego, f, P):
            d = dict(ego=((ego - mu) / sd).astype(np.float32), rmem=f, plan=P, front=cf["front"],
                     **{k: np.full((len(P),) + sh, np.nan, np.float32) for k, sh in (("ev", (3, 2)), ("anyz", (2,)), ("spin", ()), ("lat2", ()), ("prog4", ()), ("wzpk", ()))})
            p = predict(model, d, vision, dev)
            return np.abs(p["lat2"]), 1 / (1 + np.exp(-p["ev"][:, 1:, 1])).sum(1)
        a_rr, z_rr = lat(cf["ego_real"], cf["f_real"], Pr)
        a_cr, z_cr = lat(cf["ego_real"], cf["f_real"], Pc)
        a_cc, z_cc = lat(cf["ego_cf"], cf["f_cf"], Pc)
        a_rc, z_rc = lat(cf["ego_cf"], cf["f_cf"], Pr)
        flip = (a_rr < a_cr) & (a_cc < a_rc)
        zflip = (z_rr < z_cr) & (z_cc < z_rc)
        from jevdrive import stats
        res[arm] = dict(flip=stats.bootstrap(flip[sel].astype(float), cf["log"][sel]),
                        prefers_given_route_real=float((a_rr < a_cr)[sel].mean()), prefers_given_route_cf=float((a_cc < a_rc)[sel].mean()),
                        zero4_flip=float(zflip[sel].mean()), med_abs_lat=dict(rr=float(np.median(a_rr[sel])), cr=float(np.median(a_cr[sel])),
                                                                              cc=float(np.median(a_cc[sel])), rc=float(np.median(a_rc[sel]))))
    return res


# ---------------------------------------------------------------- gradient sanity
def grad_read(model, vision, d, D, mu, sd, dev, n=100):
    import torch
    from agent_hinge import AgentHinge
    from drivable_hinge import Hinge
    from loss43 import shape_only
    P = D / f"runs/op_parity/cache/on1_{HELD}"
    rng = np.random.default_rng(0)
    ev4 = d["ev"][:, :, 1]
    sets = dict(route=rng.permutation(np.flatnonzero((ev4[:, 1] == 1) | (ev4[:, 2] == 1)))[:n], collision=rng.permutation(np.flatnonzero(ev4[:, 0] == 1))[:n])
    names = d["names"]
    AH = AgentHinge(P / "agent.npz", names, dev, margin=0.5)
    RH = Hinge(P / "road.npz", names, dev, margin=0.3)
    ab = np.load(P / "agent.npz")
    apos = {t: i for i, t in enumerate(ab["tokens"].tolist())}
    out = {}
    model.eval()
    for key, r in sets.items():
        if not len(r):
            continue
        rt = torch.as_tensor(r, device=dev)
        plan = torch.tensor(d["plan"][r], device=dev, requires_grad=True)
        with torch.autocast(dev.type, torch.bfloat16):
            o = model(torch.from_numpy(d["front"][r]).to(dev) if vision else None, torch.from_numpy(d["ego"][r]).to(dev),
                      torch.from_numpy(d["rmem"][r]).to(dev), plan)
        risk = o["ev"][:, :, 1].sum(1) + o["lat2"].abs()
        g = -torch.autograd.grad(risk.sum(), plan)[0][..., :2].float().cpu().numpy()        # (n, 8, 2)
        pl = d["plan"][r].copy()
        ps = torch.tensor(pl, device=dev, requires_grad=True)
        xs, ys = shape_only(ps[..., 0], ps[..., 1], ps[..., 2])
        S = 10 * AH.per_step(xs, ys, ps[..., 2], rt).mean(1) * AH.ok[rt] + 10 * torch.relu(0.3 - RH.margins(xs, ys, ps[..., 2], rt)).mean(1) * RH.ok[rt]
        gS = -torch.autograd.grad(S.sum(), ps)[0][..., :2].float().cpu().numpy()
        J = [1, 2, 3]                                                                        # the 1.0 / 1.5 / 2.0 s poses
        cos, cosS, keep = [], [], []
        for i, ri in enumerate(r):
            v = []
            for j in J:
                p = pl[i, j, :2]
                if key == "route":
                    Pl = d["poly"][ri]
                    Pl = Pl[np.isfinite(Pl).all(1)]
                    if len(Pl) < 2:
                        continue
                    seg = np.linspace(0, 1, 11)[:, None]
                    dense = np.concatenate([Pl[k] + seg * (Pl[k + 1] - Pl[k]) for k in range(len(Pl) - 1)])
                    q = dense[np.argmin(np.hypot(*(dense - p).T))]
                    w = q - p
                    if np.hypot(*w) < 0.5:
                        continue
                else:
                    ia = apos.get(names[ri], -1)
                    if ia < 0:
                        continue
                    bx, va = ab["box"][ia, j + 1], ab["valid"][ia, j + 1]
                    if not va.any():
                        continue
                    c = bx[va][np.argmin(np.hypot(*(bx[va][:, :2] - p).T)), :2]
                    w = p - c
                gi = g[i, j]
                if np.hypot(*gi) < 1e-9:
                    continue
                v.append(gi @ w / (np.hypot(*gi) * np.hypot(*w)))
            if v:
                keep.append(i)
                cos.append(np.mean(v))
            nS = np.linalg.norm(gS[i, J])
            if nS > 1e-9:
                cosS.append(float((g[i, J] * gS[i, J]).sum() / (np.linalg.norm(g[i, J]) * nS + 1e-12)))
        gn = np.linalg.norm(g[:, J].reshape(len(r), -1), axis=1)
        gSn = np.linalg.norm(gS[:, J].reshape(len(r), -1), axis=1)
        act = gSn > 1e-9
        out[key] = dict(rows=int(len(r)), rows_read=int(len(keep)), share_toward=float(np.mean(np.array(cos) > 0)) if cos else np.nan,
                        cos_median=float(np.median(cos)) if cos else np.nan, critic_grad_median=float(np.median(gn)),
                        s_active_rows=int(act.sum()), s_grad_median_active=float(np.median(gSn[act])) if act.any() else np.nan,
                        critic_grad_median_on_s_active=float(np.median(gn[act])) if act.any() else np.nan,
                        cos_critic_vs_s_median=float(np.median(cosS)) if cosS else np.nan, cos_critic_vs_s_pos=float(np.mean(np.array(cosS) > 0)) if cosS else np.nan)
    return out


# ---------------------------------------------------------------- main
def main():
    import torch
    from jevdrive.common import data_dir
    from jevdrive.data import splits
    from jevdrive.run import Run
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda", help="cpu: code-path check only")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--B", type=int, default=10_000)
    ap.add_argument("--B-desc", type=int, default=2_000)
    a = ap.parse_args()
    if a.smoke:
        a.epochs, a.B, a.B_desc = 2, 200, 200
    D = data_dir()
    dev = torch.device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    with Run("critic_pilot", "smoke" if a.smoke else "train", config=vars(a)) as run:
        tr_split, he_split = splits.load("alpasim_nuplan/onpol-train"), splits.load("alpasim_nuplan/onpol-held")
        run.use_split(tr_split), run.use_split(he_split)
        out_dir = run.dir / "ckpt"
        out_dir.mkdir(parents=True, exist_ok=True)
        need_tok = any(ARMS[x].get("vision", True) for x in a.arms)
        pools = sorted({p for x in a.arms for p in ARMS[x]["pools"]})
        data = cat([load_pool(D, p, need_tok) for p in pools])
        held = load_pool(D, HELD, need_tok)
        assert tr_split.mask(np.unique(data["scene"])).all() and he_split.mask(np.unique(held["scene"])).all(), "pool scenes outside their split"
        assert not set(data["log"]) & set(held["log"]), "train and held share a log"
        mu, sd = data["ego"].mean(0), data["ego"].std(0) + 1e-3
        data["ego"] = ((data["ego"] - mu) / sd).astype(np.float32)
        held["ego"] = ((held["ego"] - mu) / sd).astype(np.float32)
        for k in ("name", "rows"):
            held.pop(k, None)
        run.info(f"train rows {len(data['ego'])} ({dict(zip(*np.unique(data['pool'], return_counts=True)))}), held rows {len(held['ego'])}")
        res = dict(prereg="experiments/route_follow/plans/2026-10-13-critic-pilot-prereg.md", smoke=a.smoke, arms={}, baselines=baselines(held, a.B_desc))
        res["held_counts"] = dict(rows=len(held["ego"]), logs=int(len(np.unique(held["log"]))), any2_pos=int(np.nansum(held["anyz"][:, 0])),
                                  any4_pos=int(np.nansum(held["anyz"][:, 1])), lat2_rows=int(np.isfinite(held["lat2"]).sum()))
        models = {}
        cf = cf_rows(D)
        for arm in a.arms:
            cfg = ARMS[arm]
            model, info = train_arm(arm, cfg, data, a, dev, run, out_dir)
            vision = cfg.get("vision", True)
            Ph = predict(model, held, vision, dev)
            np.savez(out_dir / f"{arm}_held_preds.npz", names=held["names"], **Ph)
            r = dict(cfg={k: v for k, v in cfg.items()}, **info, held=read(held, Ph, a.B, a.B_desc))
            if not vision:
                r["reads"] = {}
                for tag in READS:
                    rd = load_reads(D, tag)
                    rd["ego"] = ((rd["ego"] - mu) / sd).astype(np.float32)
                    r["reads"][tag] = read(rd, predict(model, rd, vision, dev), a.B_desc, a.B_desc)
            models[arm] = (model, vision)
            fr = flip_read({arm: (model, vision)}, cf, mu, sd, dev) if vision else None
            if fr:
                r["flip"] = fr[arm]
                res["flip_rows"] = {k: v for k, v in fr.items() if k != arm}
            if arm in ("C2", "C3", "C2-aug"):
                r["grad"] = grad_read(model, vision, held, D, mu, sd, dev)
            res["arms"][arm] = r
            run.info(f"{arm}: {json.dumps({k: r['held'][k] for k in ('any2', 'lat2_r2') if k in r['held']})} flip {r.get('flip', {}).get('flip')}")
            del model
            torch.cuda.empty_cache()
        if "C2" in res["arms"]:
            c2 = res["arms"]["C2"]
            res["lines"] = dict(L1=c2["held"]["any2"]["v"] >= 0.90, L2=c2["held"].get("lat2_r2", {}).get("v", np.nan) >= 0.60,
                                L3=c2.get("flip", {}).get("flip", {}).get("mean", np.nan) >= 0.70)
        rd = _R / "experiments/critic_pilot/results"
        rd.mkdir(parents=True, exist_ok=True)
        stem = "smoke" if a.smoke else "critic_pilot"
        (rd / f"{stem}.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
        (run.dir / f"{stem}.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
        run.summary["lines"] = res.get("lines")
        run.info(f"lines: {res.get('lines')}")


if __name__ == "__main__":
    main()
