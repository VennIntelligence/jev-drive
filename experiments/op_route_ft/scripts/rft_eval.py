"""Open-loop readouts of op_route_ft (plans/2026-10-05-route-ft-prereg.md), called as `rft.py evalol --models O rc-bear-s0 ...`.

  carla_exit   CARLA dev junctions (none of them on a B2D route), every (pose, exit) row with the clean exit polyline as the command: the plan's
               class at 4 s (heading > +30 deg left, < -30 deg right, else straight) against the commanded exit; rows whose original-speed arc at
               4 s passes the junction mouth by >= 15 m. Per row and all-exits-correct per 3-exit pose; cluster = junction.
  drift        no command: |y(4 s)| of the model's plan minus the original's, rear axle; junction rows (CARLA dev poses, real rows with a >= 25 deg
               turn starting within 60 m) and straight rows (real, no turn within 150 m).
  negatives    negative command (CARLA N1 on dev poses with a missing class; real N3 / N4 on dev rows): |y(4 s)| of the plan minus the original's
               no-command plan.
  uptake       real dev turn rows (|turn| >= 45 deg starting within 40 m, v0 >= 2): |y(4 s) - logged y(4 s)| with the command vs without.
"""
from __future__ import annotations

import json

import numpy as np
import torch

import rft as F
import route_adapter as RA
import route_neg as RN
from experiments.op_adapt_l.lib import op_adapt_l as L
from jevdrive import op_adapt as A

T4 = np.array([4.0])


def y4(plan, cam):
    return np.array([L.rear_np(p[None], float(c), T4)[0, 0] for p, c in zip(plan, np.broadcast_to(cam, (len(plan),)))])


def yaw4(plan):
    W = A.T_IDXS
    return -np.degrees(np.array([np.interp(4.0, W, p[:, 11]) for p in plan]))


def cls(deg):
    return np.where(deg > 30, "left", np.where(deg < -30, "right", "straight"))


@torch.no_grad()
def run_rows(m, trunk_of, rows, feat, valid, tc, dev, bs=64):
    """plans (n, 33, 15) and action[0] (n,) of model m on `rows` (trunk_of(rows) -> (b, 9, 1024, 8, 16))."""
    pi = A.plan_index(m.net.slices)
    a0 = m.net.slices["action"].start
    P, Ac = [], []
    for i in range(0, len(rows), bs):
        r = rows[i:i + bs]
        tr = torch.from_numpy(np.ascontiguousarray(trunk_of(r))).to(dev)
        f = None
        if m.route is not None and feat is not None:
            f = torch.from_numpy(feat[i:i + bs]).to(dev)
            if getattr(m, "zero_cmd", False):
                f = torch.zeros_like(f)
        o = m(tr, torch.from_numpy(valid[i:i + bs]).to(dev), torch.from_numpy(tc[i:i + bs]).to(dev), f)["outputs"].float()
        P.append(o[:, pi].reshape(-1, 33, 15).cpu().numpy())
        Ac.append(o[:, a0].cpu().numpy())
    return np.concatenate(P), np.concatenate(Ac)


def boot(x, groups):
    gi, Lc, W = L.cluster_boot(np.asarray(groups), B=2000)
    m, ci = L.boot_mean(np.asarray(x, float), gi, Lc, W)
    return {"n": int(len(x)), "clusters": int(Lc), "mean": m, "lo": float(ci[0]), "hi": float(ci[1])}


def build_sets(carla, cap, seed=0):
    """Fixed eval rows and command features (both encodings) for every readout."""
    rng = np.random.default_rng(seed)
    C = F.Carla(carla)
    C.rows_of = {}
    for j, p in enumerate(C.r["pose_row"]):
        C.rows_of.setdefault(int(p), []).append(j)
    C.rows_of = {k: np.array(v) for k, v in C.rows_of.items()}
    S = {}
    dev_p = np.flatnonzero(C.tab["split"] == "dev")
    s4 = np.array([F.plan_arc(C.tea["mu"][p], C.cam)[15] for p in dev_p])
    elig = dev_p[s4 >= C.tab["d"][dev_p] + 15.0]
    rows = np.concatenate([C.rows_of[p] for p in elig if p in C.rows_of])
    S["carla_exit"] = dict(rows=rows, pose=C.r["pose_row"][rows], cmd=C.r["cmd"][rows], cluster=C.r["cluster"][rows],
                           n_exits=C.tab["n_exits"][C.r["pose_row"][rows]],
                           fb=np.stack([RA.features("bear", C.r["poly"][j], C.r["pmask"][j]) for j in rows]),
                           fp=np.stack([RA.features("poly", C.r["poly"][j], C.r["pmask"][j]) for j in rows]))
    dp = dev_p[np.random.default_rng(1).permutation(len(dev_p))[:cap]]
    S["carla_pose"] = dict(pose=dp, cluster=C.tab["cluster"][dp])
    neg = []
    for p in dev_p:
        if p not in C.rows_of or C.tab["profile"][p] == "stopped":
            continue
        miss = F.missing_classes(C, p)
        if not miss:
            continue
        js = C.rows_of[p]
        base = js[np.argmin(np.abs(C.r["angle"][js]))]
        P = F.dense_from_poly(C.r["poly"][base], C.r["pmask"][base])
        n = RN.make("N1_exit", P, rng, s_branch=float(C.tab["d"][p]) + 2.0, missing=miss[0]) if P is not None else None
        if n is not None:
            neg.append((p, n["poly"], n["pmask"]))
    neg = neg[:cap]
    S["carla_neg"] = dict(pose=np.array([x[0] for x in neg]), cluster=C.tab["cluster"][[x[0] for x in neg]],
                          fb=np.stack([RA.features("bear", x[1], x[2]) for x in neg]), fp=np.stack([RA.features("poly", x[1], x[2]) for x in neg]))
    R = {d: F.Real(d) for d in ("nav", "wod")}
    for d, Rd in R.items():
        t, rt = Rd.S.t, Rd.route
        dv = Rd.S.rows("dev")
        dv = dv[rt["has_route"][dv] & (rt["pmask"][dv].sum(1) >= 3)]
        turn = np.isfinite(rt["turn_deg"][dv]) & (rt["turn_s"][dv] <= 60)
        strt = ~np.isfinite(rt["turn_deg"][dv]) & (rt["n_turn"][dv] == 0) & ~rt["in_turn"][dv]
        S[f"{d}_junction"], S[f"{d}_straight"] = dv[turn][:cap], dv[strt][:cap]
        up = dv[(np.abs(np.nan_to_num(rt["turn_deg"][dv])) >= 45) & (rt["turn_s"][dv] <= 40) & (t["v0"][dv] >= 2)]
        S[f"{d}_uptake"] = dict(rows=up, fb=np.stack([RA.features("bear", rt["poly"][i], rt["pmask"][i]) for i in up]) if len(up) else np.zeros((0, 6), np.float32),
                                fp=np.stack([RA.features("poly", rt["poly"][i], rt["pmask"][i]) for i in up]) if len(up) else np.zeros((0, 46), np.float32))
        nn_ = []
        for i in dv[(rt["plen"][dv] >= 60) & (t["v0"][dv] >= 2)]:
            P = F.dense_from_poly(rt["poly"][i], rt["pmask"][i])
            kind = "N4_uturn" if (rt["n_turn"][i] == 0 and not rt["in_turn"][i]) and rng.random() < 0.5 else "N3_wrong"
            n = RN.make(kind, P, rng, lht=bool(t["tc"][i][1] > 0.5)) if P is not None else None
            if n is not None:
                nn_.append((i, kind, n["poly"], n["pmask"]))
            if len(nn_) >= cap // 2:
                break
        S[f"{d}_neg"] = dict(rows=np.array([x[0] for x in nn_]), kind=np.array([x[1] for x in nn_]),
                             fb=np.stack([RA.features("bear", x[2], x[3]) for x in nn_]), fp=np.stack([RA.features("poly", x[2], x[3]) for x in nn_]))
    return C, R, S


def main(a):
    dev = torch.device("cuda")
    C, R, S = build_sets(a.carla, a.cap)
    out_dir = F.rroot("evalol")
    zc = lambda n: np.zeros((n, 6), np.float32)  # noqa: E731
    ctr = lambda r: C.T[r]  # noqa: E731
    cv, ctc = C.tab["slot_valid"], C.tab["tc"]
    res_all = {}
    O = F.load_rmodel("O", dev)
    base = {}

    def real_fn(d):
        Rd = R[d]
        return (lambda r: Rd.B.T[Rd.normal[r]]), (lambda r: Rd.B.v["slot_valid"][Rd.normal[r]]), (lambda r: Rd.S.t["tc"][r]), (lambda r: Rd.S.t["cam"][r, 0])

    # the original's no-command plans on every set
    def plans(m, set_key, feat_key=None):
        s = S[set_key]
        if set_key.startswith("carla"):
            rows = s["pose"] if "pose" in s else s
            fe = s.get(feat_key) if feat_key else None
            return run_rows(m, ctr, rows, fe, cv[rows], ctc[rows], dev), np.full(len(rows), C.cam)
        d = set_key.split("_")[0]
        tf, vf, tcf, camf = real_fn(d)
        rows = s["rows"] if isinstance(s, dict) else s
        fe = s.get(feat_key) if (feat_key and isinstance(s, dict)) else None
        return run_rows(m, tf, rows, fe, vf(rows), tcf(rows), dev), camf(rows)

    for key in ("carla_pose", "carla_neg", "nav_junction", "nav_straight", "wod_junction", "wod_straight", "nav_neg", "wod_neg", "nav_uptake", "wod_uptake"):
        base[key] = plans(O, key)
    for name in a.models:
        m = O if name == "O" else F.load_rmodel(name, dev)
        fk = "fp" if (m.route is not None and m.route.enc == "poly") else "fb"
        r = {}
        # carla exits
        s = S["carla_exit"]
        (pl, ac), cam = run_rows(m, ctr, s["pose"], s[fk], cv[s["pose"]], ctc[s["pose"]], dev), C.cam
        c = cls(yaw4(pl))
        ok = (c == s["cmd"]).astype(float)
        r["carla_exit_row"] = boot(ok, s["cluster"])
        for cm in ("left", "straight", "right"):
            mm = s["cmd"] == cm
            r[f"carla_exit_{cm}"] = boot(ok[mm], s["cluster"][mm]) if mm.any() else {"n": 0}
        three = s["n_exits"] >= 3
        allok = {}
        for p, o_ in zip(s["pose"][three], ok[three]):
            allok.setdefault(int(p), []).append(o_)
        pp = np.array(list(allok))
        r["carla_exit_3pose_all"] = boot(np.array([float(np.all(allok[p])) for p in pp]), C.tab["cluster"][pp]) if len(pp) else {"n": 0}
        turn = s["cmd"] != "straight"
        sg = np.where(s["cmd"] == "left", 1.0, -1.0)
        r["carla_action_sign_turn"] = boot((np.sign(ac[turn]) == sg[turn]).astype(float), s["cluster"][turn])
        r["carla_plan_class_counts"] = {f"{cm}->{k}": int(((s["cmd"] == cm) & (c == k)).sum()) for cm in ("left", "straight", "right")
                                        for k in ("left", "straight", "right")}
        # drift without command
        for key in ("carla_pose", "nav_junction", "nav_straight", "wod_junction", "wod_straight"):
            (pm, _), cam = plans(m, key)
            (po, _), _ = base[key]
            dy = np.abs(y4(pm, cam)[:, 1] - y4(po, cam)[:, 1])
            r[f"drift_{key}"] = {"n": int(len(dy)), "median": float(np.median(dy)) if len(dy) else None, "mean": float(dy.mean()) if len(dy) else None,
                                 "p90": float(np.percentile(dy, 90)) if len(dy) else None}
        # negatives
        for key in ("carla_neg", "nav_neg", "wod_neg"):
            (pm, _), cam = plans(m, key, fk)
            (po, _), _ = base[key]
            dy = np.abs(y4(pm, cam)[:, 1] - y4(po, cam)[:, 1])
            r[f"neg_{key}"] = {"n": int(len(dy)), "mean": float(dy.mean()) if len(dy) else None, "median": float(np.median(dy)) if len(dy) else None,
                               "p90": float(np.percentile(dy, 90)) if len(dy) else None}
        # uptake on real turns
        for d in ("nav", "wod"):
            key = f"{d}_uptake"
            rows = S[key]["rows"]
            if not len(rows):
                continue
            (pm, _), cam = plans(m, key, fk)
            (po, _), _ = base[key]
            fut = L.human_targets(R[d].S.t["fut20"][rows])[:, 15, :2]
            em = np.abs(y4(pm, cam)[:, 1] - fut[:, 1])
            eo = np.abs(y4(po, cam)[:, 1] - fut[:, 1])
            r[f"uptake_{d}"] = {"n": int(len(rows)), "err_model": float(em.mean()), "err_orig": float(eo.mean()),
                                "delta": boot(em - eo, R[d].S.t["cluster"][rows])}
        res_all[name] = r
        (out_dir / f"{name}.json").write_text(json.dumps(r, indent=1, default=float))
        print(name, json.dumps({k: (v.get("mean", v.get("median")) if isinstance(v, dict) else v) for k, v in r.items() if not k.endswith("counts")}, default=float),
              flush=True)
    return res_all
