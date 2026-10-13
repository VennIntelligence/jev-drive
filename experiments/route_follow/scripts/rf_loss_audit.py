#!/usr/bin/env python3
"""ROUTE-FOLLOW loss audit (research/route-follow/loss-audit.md): is the closed-loop training objective able to tell the human's turn from the
wrong turns the policy drives (early inside cut, too wide, no deceleration, straight on)? Three tests on junction-turn states that carry both
the human's future and a model plan:

  1 plug-in    every per-row loss term the route-night trainers use (rm_train.py / on_train.py / loss43.py / on_route.py), evaluated on
               candidate 8-pose plans of the same state: H (the row's human-anchored imitation target, `fut` / PAI `imit`), M (the model's
               plan), H shifted 0.5 / 1.0 m inward and 1.0 m outward (along each pose's normal, yaw kept), H at the model's arc lengths and at
               constant speed v0 (no deceleration), straight on at H's arc lengths.
  2 gradient   d term / d pose at M and at H (with the trainer's shape_only on the shape-only hinges), split per pose into the inward-normal
               and the along-heading component; reported as the descent push (minus the gradient): inward + / outward -, forward (longer
               arc, faster) + / backward (shorter arc, slower) -; binned by the distance to the turn entry.
  3 reach      the RB0-F-s0 model fitted to the human futures of the turn rows only (imitation term only, no on-policy rows, a few hundred
               steps) on the frozen encoder tokens: A = the adapter + route branch alone, B = everything the recipe trains (adapter, branch
               and the trainable net tensors). Train / held-out-scene ADE at 2 / 4 s, inward offset of the fitted plan against H.

Terms (one row; the trainer's per-row weight relative to one imitation row in brackets; every value is the trainer's own function):
  imit    pp_train.Losses.dist: mean over the 8 poses of huber((x - tx) / sx) + huber((y - ty) / sy) + huber((psi - tpsi) / spsi), delta 1,
          sx = 0.3 + 0.2 t, sy = 0.1 + 0.1 t, spsi = 1 + 1 t deg (the "time weights": later poses count less)        [1]
          imit_x = the x part alone (where a speed error shows)
  drv     P2H10's drivable hinge (lib/drivable_hinge.Hinge, margin 0.3, Pacifica corners, 0.1 s densified), on the row's drivable raster
          (pool drivable.npz = P2H10's layers); in training it sits on the offline imitation rows, full gradient        [hinge_lam 10]
  road    S's road-and-lane raster hinge (same Hinge on road.npz, margin 0.3), on the pool rows, shape-only               [road_lam 10 x ho_w 3]
  agent   S's agent hinge (lib/agent_hinge.AgentHinge, margin 0.3, side margin 0, per_step mean over 41 steps), shape-only  [10 x 3]
  route   the route hinge (on_route.loss, the row's dead zone: 1.0 on-policy), shape-only; route05 = the same with dz 0.5    [10 x 3]
  `--label-bank` changes the raster storage only (drivable_hinge.RowBank: same values); `--yr-mass 0.1` and `--ho-w 3` are row-mix weights,
  carried in the recipe totals below. Recipe totals per row (what one row of that kind adds, up to the common 1 / n_imit):
  J_S    = 30 (road + agent)                         RSG-L18's on-policy row (mode self; no imitation)
  J_SR   = 30 (road + agent + route)                 RBSG / RBSG2's on-policy row
  J_both = imit + 30 (road + agent)                  PAIT rows (mode both; road ok False on PAI)
  J_off  = imit + 10 drv                             an offline navtrain / yaw-rate imitation row
  J_all  = imit + 10 drv + 30 (road + agent + route) every term at once

States (`collect`): rows whose label polyline (route.npz poly; PAI pol rows: the logged path, every 2 m from 10 m behind to 90 m ahead, a
proxy) has a turn: a stretch with curvature > 1 / 50 m^-1 (7 m smoothing) and a net heading change >= 45 deg; entry / exit = its ends; the
first such stretch with its exit ahead and its entry within 40 m; bin b40_20 / b20_0 / inside (entry behind the ego, exit ahead). A turn =
one (scene, direction): rows of one rollout at one turn count once. Sets:
  nup_held    on1_L18RSGallheld   RSG-L18-s0 driving held 460 (18 decisions), M = served plan, H = fut (on_rows Amendment 2 recovery target)
  nup_held2   on1_L18RBSG2allheld RBSG2-L18-s0 on held 460, same
  nup_train   on1_L18RSG          RSG-L18-s0 on the 1 020 train scenes, same
  pait        on1_PAIT80          PAI human-log states at turns (the log-pass driver), H = fut, M = RSG-L18-s0's plan on the row's tokens
                                  (`plans`; the row's own `served` is the log)
  pai_pol     Tokyo rows/pol      RB0-F-s0 driving PAI train160 (on-policy, 10 Hz), H = imit, M = served poses; no road raster; route = logged
                                  path (proxy); agent boxes of the rollout

  rf_loss_audit.py pai-extract --src /data/runs/route_night/rows/pol --out F.npz         Tokyo, numpy only (turn-window rows)
  rf_loss_audit.py collect --out DIR [--pai-pol F.npz]                                    box CPU, op-train env -> DIR/states_<set>.npz
  rf_loss_audit.py plans --out DIR --sets pait --tags RSG-L18-s0 RBSG2-L18-s0            one GPU -> DIR/plans_<set>_<tag>.npz
  rf_loss_audit.py audit --out DIR                                                        CPU -> DIR/plugin.csv, grad.csv, tables.md
  rf_loss_audit.py fit --out DIR [--steps 400]                                            one GPU -> DIR/fit.json, fit.md
  rf_loss_audit.py fig --out DIR --figs FIGDIR                                            CPU (matplotlib)
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_P = _pl.Path(__file__).resolve().parents
_R = _P[3] if len(_P) > 3 else _P[0]                                     # pai-extract runs from a scratch dir on Tokyo
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_R / "experiments/op_parity/scripts"),
                 str(_R / "experiments/alpasim/lib"), str(_R / "experiments/alpasim/scripts"), str(_R / "experiments/body1/lib"),
                 str(_R / "experiments/onpol_nuplan/lib"), str(_R / "experiments/route_mem/lib")]
import argparse, json, os  # noqa: E401,E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

T8 = 0.5 * np.arange(1, 9)
SIG = np.stack([0.3 + 0.2 * T8, 0.1 + 0.1 * T8, np.radians(1.0 + 1.0 * T8)])          # pp_train SIG_X, SIG_Y, SIG_PSI (asserted in `fit`)
SETS = {"nup_held": "on1_L18RSGallheld", "nup_held2": "on1_L18RBSG2allheld", "nup_train": "on1_L18RSG", "pait": "on1_PAIT80"}
VARIANTS = ("H", "M", "Hin05", "Hin10", "Hout10", "HvM", "Hv0", "STR")
WRONG = ("Hin05", "Hin10", "Hout10", "HvM", "Hv0", "STR")
TERMS = ("imit", "imit_x", "drv", "road", "agent", "route", "route05")
RECIPES = {"J_S": {"road": 30, "agent": 30}, "J_SR": {"road": 30, "agent": 30, "route": 30}, "J_both": {"imit": 1, "road": 30, "agent": 30},
           "J_off": {"imit": 1, "drv": 10}, "J_all": {"imit": 1, "drv": 10, "road": 30, "agent": 30, "route": 30}}
GW = {"imit": 1, "imit_x": 1, "drv": 10, "road": 30, "agent": 30, "route": 30, "route05": 30}      # trainer weights (per row)
SHAPE = {"road", "agent", "route", "route05"}                                                     # terms that see shape_only poses
BINS = ("b40_20", "b20_0", "inside")
KAPPA, TURN_DEG, ENTRY_MAX = 1 / 50, 45.0, 40.0
MOVING = 3.0                                                                                       # m/s: the moving-rows tables


def data_dir() -> Path:
    return Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))


# ---------------------------------------------------------------- geometry (numpy)
def turn_of(P):
    """Label polyline P (M, 2) in the row frame (NaN padded) -> (dir +1 left / -1 right / 0 none, net angle deg, entry arc, exit arc), arcs
    from the ego's projection; rule in the module docstring."""
    ok = np.isfinite(P).all(1)
    if ok.sum() < 6:
        return 0, 0.0, np.nan, np.nan
    Q = P[ok]
    d = np.diff(Q, axis=0)
    L = np.hypot(*d.T)
    keep = L > 1e-3
    Q, d, L = np.r_[Q[:1], Q[1:][keep]], d[keep], L[keep]
    s = np.r_[0, np.cumsum(L)]
    # the ego's projection (origin) on the polyline
    a, w = Q[:-1], np.clip((-(Q[:-1]) * d).sum(1) / np.maximum(L ** 2, 1e-9), 0, 1)
    j = int(np.hypot(*(a + w[:, None] * d).T).argmin())
    s = s - (s[j] + w[j] * L[j])
    h = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    g = np.arange(np.ceil(s[0]), np.floor(s[-1]) + 1)
    if len(g) < 12:
        return 0, 0.0, np.nan, np.nan
    th = np.interp(g, 0.5 * (s[1:] + s[:-1]), h)
    k = np.convolve(np.gradient(th), np.ones(7) / 7, "same")
    m = np.abs(k) > KAPPA
    i = 0
    while i < len(g):
        if not m[i]:
            i += 1
            continue
        j = i
        sg = np.sign(k[i])
        while j + 1 < len(g) and (m[j + 1] and np.sign(k[j + 1]) == sg or (j + 4 < len(g) and m[j + 4] and np.sign(k[j + 4]) == sg and not m[j + 1])):
            j += 1
        ang = np.degrees(th[j] - th[i])
        if abs(ang) >= TURN_DEG and g[j] > 0 and g[i] <= ENTRY_MAX:
            return int(np.sign(ang)), float(ang), float(g[i]), float(g[j])
        i = j + 1
    return 0, 0.0, np.nan, np.nan


def bin_of(e, x):
    return "inside" if e <= 0 < x else ("b20_0" if 0 < e <= 20 else ("b40_20" if 20 < e <= 40 else ""))


def along(H, s):
    """H (n, 8, 3) path from the origin, s (n, 8) arc lengths -> poses (n, 8, 3) of that polyline at s; beyond its end straight on along the
    last moving segment (pp_train.along's rule, numpy)."""
    out = np.zeros_like(H)
    for r in range(len(H)):
        P = np.r_[np.zeros((1, 3)), H[r]]
        d = np.diff(P[:, :2], axis=0)
        L = np.hypot(*d.T)
        c = np.r_[0, np.cumsum(L)]
        mv = np.flatnonzero(L > 1e-3)
        if not len(mv):
            out[r] = np.c_[s[r] * np.cos(P[-1, 2]), s[r] * np.sin(P[-1, 2]), np.full(8, P[-1, 2])]
            continue
        last = mv[-1]
        u = d[last] / L[last]
        hl = np.arctan2(u[1], u[0])
        for k, sk in enumerate(s[r]):
            if sk >= c[-1]:
                out[r, k] = [P[-1, 0] + (sk - c[-1]) * u[0], P[-1, 1] + (sk - c[-1]) * u[1], hl]
            else:
                i = int(np.clip(np.searchsorted(c, sk, "right") - 1, 0, len(L) - 1))
                f = (sk - c[i]) / max(L[i], 1e-9)
                out[r, k] = P[i] + f * (P[i + 1] - P[i])
    return out


def arcs(P):
    """(n, 8, 3) -> (n, 8) cumulative arc length from the origin."""
    d = np.diff(np.concatenate([np.zeros_like(P[:, :1, :2]), P[..., :2]], 1), axis=1)
    return np.cumsum(np.hypot(d[..., 0], d[..., 1]), 1)


def to_poly(X, poly):
    """Plan poses X (n, 8, 3), polylines (n, M, 2) NaN padded -> (n, 8) signed lateral offset (left +) to the nearest segment (NaN if none)."""
    A, B = poly[:, None, :-1], poly[:, None, 1:]
    d = B - A
    L2 = np.maximum((d ** 2).sum(-1), 1e-9)
    p = X[:, :, None, :2]
    w = np.clip(((p - A) * d).sum(-1) / L2, 0, 1)
    f = A + w[..., None] * d
    dist = np.hypot(*(p - f).transpose(3, 0, 1, 2))
    dist = np.where(np.isfinite(dist), dist, np.inf)
    j = dist.argmin(-1)
    cr = d[..., 0] * (p[..., 1] - A[..., 1]) - d[..., 1] * (p[..., 0] - A[..., 0])
    lat = np.take_along_axis(np.sign(cr) * dist, j[..., None], -1)[..., 0]
    return np.where(np.isfinite(lat), lat, np.nan)


def variants(H, M, sgn, v0):
    """Candidate plans (dict name -> (n, 8, 3)); sgn (n,) +1 left / -1 right turn."""
    n_in = sgn[:, None, None] * np.stack([-np.sin(H[..., 2]), np.cos(H[..., 2])], -1)    # inward unit normal per pose
    sh = lambda d: np.concatenate([H[..., :2] + d * n_in, H[..., 2:]], -1)  # noqa: E731
    sH = arcs(H)
    st = np.zeros_like(H)
    st[..., 0] = sH
    return dict(H=H, M=M, Hin05=sh(0.5), Hin10=sh(1.0), Hout10=sh(-1.0), HvM=along(H, arcs(M)), Hv0=along(H, v0[:, None] * T8[None]), STR=st)


# ---------------------------------------------------------------- PAI on-policy rows (Tokyo, numpy only)
def cmd_pai_extract(a):
    import glob
    out = {k: [] for k in ("scene", "k", "v0", "H", "M", "poly", "box", "val", "e0", "score", "after_event")}
    for f in sorted(glob.glob(f"{a.src}/*.npz")):
        z = np.load(f, allow_pickle=False)
        ok = ~z["after_end"] & z["imit_ok"].all(1)
        P = z["path"][:, 10:111:2].astype(np.float32)                    # s_ego - 10 .. s_ego + 90 m every 2 m, the route.npz layout
        P[~z["path_ok"][:, 10:111:2]] = np.nan
        hh = z["path_h"][:, 10:81].astype(np.float64)
        hh[~z["path_ok"][:, 10:81]] = np.nan
        with np.errstate(invalid="ignore"):
            span = np.nanmax(np.abs(hh - hh[:, 10:11]), 1)                 # max heading change over -10 .. +60 m
        keep = np.flatnonzero(ok & (np.nan_to_num(span) >= np.radians(30)))
        if not len(keep):
            continue
        out["scene"] += [str(z["scene"])] * len(keep)
        out["k"].append(z["k"][keep])
        out["v0"].append(z["v0"][keep])
        out["H"].append(z["imit"][keep])
        out["M"].append(z["poses"][keep])
        out["poly"].append(P[keep])
        out["box"].append(z["box"][keep])
        out["val"].append(z["box_ok"][keep] & (z["box_cls"][keep] >= 0)[:, None])
        out["e0"].append(z["e0"][keep])
        out["score"] += [float(z["score"])] * len(keep)
        out["after_event"].append(z["after_event"][keep])
        print(f, len(keep), flush=True)
    res = {k: (np.array(v) if k in ("scene", "score") else np.concatenate(v)) for k, v in out.items()}
    np.savez_compressed(a.out, **res)
    print("rows", len(res["k"]), "scenes", len(set(res["scene"].tolist())))


# ---------------------------------------------------------------- states
def cmd_collect(a):
    C = data_dir() / "runs/op_parity/cache"
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    summ = {}
    srcs = dict(SETS)
    if a.pai_pol:
        srcs["pai_pol"] = a.pai_pol
    for s, pool in srcs.items():
        if s == "pai_pol":
            z = np.load(pool)
            names = np.array([f"on:pol:{sc}:{k}" for sc, k in zip(z["scene"], z["k"])])
            scene, H, M, poly, v0, cam = z["scene"].astype(str), z["H"], z["M"], z["poly"], z["v0"], np.zeros(len(names))
            dz, rok = np.ones(len(names), np.float32), np.isfinite(poly).sum(1)[:, 0] >= 20
            np.savez(o / "pai_pol_agent.npz", tokens=names, ok=np.ones(len(names), bool), box=z["box"], valid=z["val"])
            idx = np.arange(len(names))
        else:
            t, r = np.load(C / pool / "tab.npz"), np.load(C / pool / "route.npz")
            names = t["names"].astype(str)
            pos = {n: i for i, n in enumerate(r["tokens"].astype(str).tolist())}
            ri = np.array([pos.get(n[3:] if n[3:] in pos else n, -1) for n in names])
            assert (ri >= 0).all(), f"{pool}: route rows missing"
            poly, dz, rok = r["poly"][ri], r["dz"][ri], r["ok"][ri]
            H, v0, cam = t["fut"], t["speed"], t["cam"][:, 0]
            M = t["served"] if s != "pait" else np.full_like(H, np.nan)
            scene = np.array([n.split(":")[-3] if s != "pait" else n.split(":")[2] for n in names])
            idx = np.arange(len(names))
        tt = np.array([turn_of(p) if ok else (0, 0.0, np.nan, np.nan) for p, ok in zip(poly, rok)], dtype=object)
        dirn, ang = tt[:, 0].astype(int), tt[:, 1].astype(float)
        ent, ext = tt[:, 2].astype(float), tt[:, 3].astype(float)
        b = np.array([bin_of(e, x) if d else "" for d, e, x in zip(dirn, ent, ext)])
        sel = np.flatnonzero((b != "") & np.isfinite(H).all((1, 2)))
        np.savez(o / f"states_{s}.npz", name=names[sel], scene=scene[sel], row=idx[sel], pool=pool, dir=dirn[sel], ang=ang[sel], entry=ent[sel],
                 exit=ext[sel], bin=b[sel], H=H[sel].astype(np.float32), M=M[sel].astype(np.float32), v0=v0[sel].astype(np.float32),
                 poly=poly[sel].astype(np.float32), dz=dz[sel].astype(np.float32), cam_x=np.asarray(cam)[sel].astype(np.float32))
        tid = {(sc, d) for sc, d in zip(scene[sel], dirn[sel])}
        summ[s] = dict(pool=str(pool), rows=int(len(names)), turn_rows=int(len(sel)), turns_left=int(sum(d > 0 for _, d in tid)), turns_right=int(sum(d < 0 for _, d in tid)),
                       rows_by_bin={k: {d: int(((b[sel] == k) & (dirn[sel] == (1 if d == "L" else -1))).sum()) for d in "LR"} for k in BINS})
        print(s, json.dumps(summ[s]), flush=True)
    (o / "states.json").write_text(json.dumps(summ, indent=1))


# ---------------------------------------------------------------- model plans on pool rows (GPU)
def load_rows(pool, rows, dev):
    """Pool rows -> (front fp16 (n, 8, 32, 512), ego (n, 220), tc (n, 2), cam_x (n,), nv or None) on dev; ego = tab ego ++ route features."""
    import torch
    C = data_dir() / "runs/op_parity/cache"
    t = np.load(C / pool / "tab.npz")
    fz = np.load(data_dir() / "runs/route_mem/feat" / f"{pool}.npz")
    assert np.array_equal(fz["names"], t["names"]), pool
    f = fz["f"].astype(np.float32)
    f[~fz["ok"]] = 0
    fr = np.load(C / f"{pool}@warp" / "front.npy", mmap_mode="r")
    o = np.argsort(rows)
    fx = np.empty((len(rows),) + fr.shape[1:], np.float16)
    fx[o] = fr[np.asarray(rows)[o]]
    ego = np.concatenate([t["ego"][rows], f[rows]], 1)
    lht = t["lht"][rows].astype(bool)
    tc = np.stack([~lht, lht], 1).astype(np.float32)
    nv = torch.as_tensor(t["nv"][rows].astype(np.int64), device=dev) if "nv" in t.files else None
    T = lambda x: torch.as_tensor(x, device=dev)  # noqa: E731
    return T(fx), T(ego), T(tc), T(t["cam"][rows][:, 0].astype(np.float32)), nv


def plan_fn(pool, dev):
    import torch
    import op_adapt_r2 as R2
    pi = np.load(data_dir() / "runs/op_parity/cache" / f"{pool}@warp" / "teacher.npz")["pi"]
    return torch.as_tensor(pi, device=dev), torch.as_tensor(R2.t_weights(T8), device=dev)


def run_plans(model, front, ego, tc, cam, nv, pi, W, bs=64, grad=False):
    import torch
    import pp_train as T
    out = []
    for i in range(0, len(front), bs):
        sl = slice(i, i + bs)
        with torch.set_grad_enabled(grad):
            p = model(front[sl], ego[sl], tc[sl], None, None, nv=None if nv is None else nv[sl]).float()[:, pi].view(-1, 33, 15)
            x, y, psi = T.rear(p, cam[sl], W)
        out.append(torch.stack([x, y, psi], -1))
    return torch.cat(out)


def cmd_plans(a):
    import torch
    import rmem as RM
    dev = torch.device("cuda")
    o = Path(a.out)
    for s in a.sets:
        st = np.load(o / f"states_{s}.npz")
        pool, rows = str(st["pool"]), st["row"]
        fr, ego, tc, cam, nv = load_rows(pool, rows, dev)
        pi, W = plan_fn(pool, dev)
        for tag in a.tags:
            m = RM.load_pmodel(tag, dev).eval()
            P = run_plans(m, fr, ego, tc, cam, nv, pi, W).cpu().numpy()
            np.savez(o / f"plans_{s}_{tag}.npz", P=P.astype(np.float32), name=st["name"])
            ade = np.hypot(*(P[..., :2] - st["H"][..., :2]).transpose(2, 0, 1)).mean()
            print(f"{s} {tag}: {len(P)} plans, ADE to H {ade:.3f} m", flush=True)
            del m
            torch.cuda.empty_cache()


# ---------------------------------------------------------------- terms (torch, CPU or GPU)
class Terms:
    """Every per-row term of one state set, on candidate plans (n, 8, 3) aligned with the states."""

    def __init__(self, o: Path, s: str, dev):
        import torch
        import on_route as OR
        from agent_hinge import AgentHinge
        from drivable_hinge import Hinge
        st = np.load(o / f"states_{s}.npz")
        self.st, self.dev, self.OR = st, dev, OR
        names = st["name"].astype(str)
        C = data_dir() / "runs/op_parity/cache" / str(st["pool"])
        self.n = len(names)
        t = lambda x: torch.as_tensor(np.asarray(x), device=dev)  # noqa: E731
        self.road = Hinge(C / "road.npz", names, dev, 0.3) if s != "pai_pol" else None
        self.drv = Hinge(C / "drivable.npz", names, dev, 0.3) if s != "pai_pol" and (C / "drivable.npz").exists() else None
        self.agent = AgentHinge(o / "pai_pol_agent.npz" if s == "pai_pol" else C / "agent.npz", names, dev, 0.3, 0.0)
        P = st["poly"].astype(np.float32)
        seg = np.linalg.norm(np.diff(P, axis=1), axis=-1).astype(np.float32)
        ok = np.isfinite(P).all(-1).sum(1) >= 2
        self.pack = dict(poly=t(P), seglen=t(seg), ok=t(ok), dz=t(st["dz"]), drop=t(np.zeros(self.n, bool)))
        self.pack05 = dict(self.pack, dz=t(np.full(self.n, 0.5, np.float32)))
        self.H = t(st["H"].astype(np.float32))
        self.sig = t(SIG.astype(np.float32))
        self.rows = torch.arange(self.n, device=dev)

    def eval(self, P, grad=False):
        """P (n, 8, 3) numpy -> {term: (n,) values} and, with grad, {term: (n, 8, 2) d term / d (x, y)} (shape-only terms through shape_only)."""
        import torch
        from loss43 import shape_only
        F = torch.nn.functional
        val, grd = {}, {}
        for term in TERMS:
            p = torch.as_tensor(P, dtype=torch.float32, device=self.dev).clone().requires_grad_(grad)
            x, y, psi = p[..., 0], p[..., 1], p[..., 2]
            if term in SHAPE:
                xs, ys = shape_only(x, y, psi)
            r = self.rows
            if term in ("imit", "imit_x"):
                h = lambda z: F.huber_loss(z, torch.zeros_like(z), reduction="none", delta=1.0)  # noqa: E731
                ex = h((x - self.H[..., 0]) / self.sig[0])
                v = ex.mean(1) if term == "imit_x" else (ex + h((y - self.H[..., 1]) / self.sig[1]) + h((psi - self.H[..., 2]) / self.sig[2])).mean(1)
            elif term in ("drv", "road"):
                hg = getattr(self, term)
                if hg is None:
                    val[term] = np.full(self.n, np.nan)
                    if grad:
                        grd[term] = np.zeros((self.n, 8, 2))
                    continue
                v = torch.relu(hg.margin - hg.margins(xs if term == "road" else x, ys if term == "road" else y, psi, r)).mean(1) * hg.ok.float()
            elif term == "agent":
                v = self.agent.per_step(xs, ys, psi, r).mean(1) * self.agent.ok.float()
            else:
                v = self.OR.loss(torch.stack([x, y, psi], -1), r, self.pack if term == "route" else self.pack05)["route"]
            val[term] = v.detach().cpu().numpy()
            if grad:
                g, = torch.autograd.grad(v.sum(), p, allow_unused=True)
                grd[term] = np.zeros((self.n, 8, 2)) if g is None else g[..., :2].cpu().numpy()
        return val, grd

    def ok_of(self, term):
        if term in ("drv", "road"):
            h = getattr(self, term)
            return np.zeros(self.n, bool) if h is None else h.ok.cpu().numpy()
        if term == "agent":
            return self.agent.ok.cpu().numpy()
        if term.startswith("route"):
            return self.pack["ok"].cpu().numpy()
        return np.ones(self.n, bool)


def recipe(val, name):
    return sum(w * np.nan_to_num(val[k]) for k, w in RECIPES[name].items())


def cmd_audit(a):
    import torch
    import pandas as pd
    o = Path(a.out)
    torch.set_num_threads(a.threads)
    dev = torch.device("cpu")
    P, G = [], []
    for s in a.sets:
        if not (o / f"states_{s}.npz").exists():
            continue
        tm = Terms(o, s, dev)
        st = tm.st
        M = st["M"]
        if s == "pait":
            M = np.load(o / f"plans_pait_{a.pait_model}.npz")["P"]
        sgn = st["dir"].astype(np.float64)
        V = variants(st["H"].astype(np.float64), M.astype(np.float64), sgn, st["v0"].astype(np.float64))
        base = dict(set=s, name=st["name"], scene=st["scene"], dir=np.where(sgn > 0, "L", "R"), bin=st["bin"], entry=st["entry"], ang=st["ang"], v0=st["v0"])
        oks = {k: tm.ok_of(k) for k in TERMS}
        for vn, X in V.items():
            val, grd = tm.eval(X.astype(np.float32), grad=vn in ("H", "M"))
            df = pd.DataFrame(base | {"variant": vn} | {k: np.where(oks[k], val[k], np.nan) for k in TERMS} | {k: recipe(val, k) for k in RECIPES})
            df["arc4"] = arcs(X)[:, -1]
            n_in = sgn[:, None, None] * np.stack([-np.sin(X[..., 2]), np.cos(X[..., 2])], -1)
            dH = X[..., :2] - st["H"][..., :2]
            df["lat_in_1_2s"] = (dH * n_in)[:, 1:4].sum(-1).mean(1)          # inward offset against H at 1.0 - 2.0 s
            df["route_in_1_2s"] = sgn * np.nanmean(to_poly(X, st["poly"])[:, 1:4], 1)   # inward offset to the route polyline at 1.0 - 2.0 s
            P.append(df)
            if grd:
                tg = np.stack([np.cos(X[..., 2]), np.sin(X[..., 2])], -1)
                g = pd.DataFrame(base | {"at": vn})
                for k in TERMS:
                    pl = -(grd[k] * n_in).sum(-1) * GW[k]                     # descent push, inward +, per pose (n, 8)
                    pf = -(grd[k] * tg).sum(-1) * GW[k]                       # descent push, forward +
                    g[f"{k}_lat"], g[f"{k}_lon"] = np.where(oks[k], pl.sum(1), np.nan), np.where(oks[k], pf.sum(1), np.nan)
                    g[f"{k}_lat_near"], g[f"{k}_lon_far"] = np.where(oks[k], pl[:, :3].sum(1), np.nan), np.where(oks[k], pf[:, 3:].sum(1), np.nan)
                G.append(g)
        print(s, "done", flush=True)
    P, G = pd.concat(P, ignore_index=True), pd.concat(G, ignore_index=True)
    P.to_csv(o / "plugin.csv.gz", index=False)
    G.to_csv(o / "grad.csv.gz", index=False)
    tables(o, P, G, "")
    tables(o, P[P.v0 >= MOVING], G[G.v0 >= MOVING], "_moving")


def tables(o, P, G, sfx):
    """Plug-in and gradient tables (markdown + csv); sfx "_moving" = rows with v0 >= MOVING."""
    import pandas as pd
    L, rows_cmp = [], []
    terms = list(TERMS) + list(RECIPES)
    for (s, d), g in P.groupby(["set", "dir"]):
        h = g[g.variant == "H"].set_index("name")
        for vn in VARIANTS:
            v = g[g.variant == vn].set_index("name").loc[h.index]
            for k in terms:
                a, b = v[k].to_numpy(float), h[k].to_numpy(float)
                m = np.isfinite(a) & np.isfinite(b)
                if not m.any():
                    continue
                dd = a[m] - b[m]
                eps = 1e-6
                rows_cmp.append(dict(set=s, dir=d, variant=vn, term=k, n=int(m.sum()), mean=float(np.mean(a[m])), mean_H=float(np.mean(b[m])),
                                     lower=float((dd < -eps).mean()), tie=float((np.abs(dd) <= eps).mean()), higher=float((dd > eps).mean()),
                                     H_pos=float((b[m] > eps).mean())))
    C = pd.DataFrame(rows_cmp)
    C.to_csv(o / f"plugin_cmp{sfx}.csv", index=False)
    gb = []
    for (s, d, b, at), g in G.groupby(["set", "dir", "bin", "at"]):
        r = dict(set=s, dir=d, bin=b, at=at, n=len(g))
        for k in TERMS:
            for c in ("lat", "lon", "lat_near", "lon_far"):
                x = g[f"{k}_{c}"].to_numpy(float)
                x = x[np.isfinite(x)]
                r[f"{k}_{c}"] = float(x.mean()) if len(x) else np.nan
                if c in ("lat", "lon"):
                    r[f"{k}_{c}_neg"] = float((x < -1e-6).mean()) if len(x) else np.nan
                    r[f"{k}_{c}_pos"] = float((x > 1e-6).mean()) if len(x) else np.nan
        gb.append(r)
    GB = pd.DataFrame(gb)
    GB.to_csv(o / f"grad_bins{sfx}.csv", index=False)
    # per-bin plug-in means (variant x term) for the figures
    PB = P.groupby(["set", "dir", "bin", "variant"])[terms + ["arc4", "lat_in_1_2s"]].mean().reset_index()
    PB.to_csv(o / f"plugin_bins{sfx}.csv", index=False)
    f = lambda x: "-" if not np.isfinite(x) else (f"{x:.3f}" if abs(x) < 100 else f"{x:.0f}")  # noqa: E731
    L += [f"# Loss audit tables (rf_loss_audit.py audit){' -- moving rows only (v0 >= 3 m/s)' if sfx else ''}", ""]
    for s in P.set.unique():
        for d in ("L", "R"):
            c = C[(C.set == s) & (C.dir == d)]
            if c.empty:
                continue
            n = int(c.n.max())
            L += [f"## {s}, {'left' if d == 'L' else 'right'} turns ({n} rows, {P[(P.set == s) & (P.dir == d) & (P.variant == 'H')].scene.nunique()} turns)", "",
                  "Mean per term and variant (per-row value, unweighted; recipe totals weighted). Cell = mean / share of rows where the variant scores "
                  "LOWER than H / share tied with H (|diff| <= 1e-6).", "",
                  "| term | " + " | ".join(VARIANTS) + " |", "|:--|" + "--:|" * len(VARIANTS)]
            for k in terms:
                cells = []
                for vn in VARIANTS:
                    r = c[(c.variant == vn) & (c.term == k)]
                    cells.append("-" if r.empty else (f(r["mean"].iloc[0]) if vn == "H" else f"{f(r['mean'].iloc[0])} / {r.lower.iloc[0]:.2f} / {r.tie.iloc[0]:.2f}"))
                L.append(f"| {k} | " + " | ".join(cells) + " |")
            L.append("")
    L += ["## Gradient direction at the model plan M (descent push = - weight x gradient, summed over the 8 poses; lat: inward +, lon: forward / "
          "faster +). Cell = mean push / share of rows pushed outward (lat) or backward = slower (lon).", ""]
    for s in G.set.unique():
        for d in ("L", "R"):
            L += [f"### {s}, {'left' if d == 'L' else 'right'}", "", "| term | " + " | ".join(f"{b} lat | {b} lon" for b in BINS) + " |",
                  "|:--|" + "--:|" * (2 * len(BINS))]
            for k in TERMS:
                cells = []
                for b in BINS:
                    r = GB[(GB.set == s) & (GB.dir == d) & (GB.bin == b) & (GB["at"] == "M")]
                    if r.empty:
                        cells += ["-", "-"]
                        continue
                    r = r.iloc[0]
                    cells += [f"{f(r[k + '_lat'])} / {f(r[k + '_lat_neg'])}", f"{f(r[k + '_lon'])} / {f(r[k + '_lon_neg'])}"]
                L.append(f"| {k} | " + " | ".join(cells) + " |")
            nn = [int(GB[(GB.set == s) & (GB.dir == d) & (GB.bin == b) & (GB["at"] == "M")].n.sum()) for b in BINS]
            L += ["", f"rows per bin: {dict(zip(BINS, nn))}", ""]
    (o / f"tables{sfx}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:60]))


# ---------------------------------------------------------------- reachability (GPU)
def cmd_fit(a):
    import torch
    import pp_train as T
    import rmem as RM
    assert np.allclose(np.stack([T.SIG_X, T.SIG_Y, T.SIG_PSI]), SIG), "SIG differs from pp_train's"
    dev = torch.device("cuda")
    torch.manual_seed(0)
    o = Path(a.out)
    res = {}
    for s in a.sets:
        st = np.load(o / f"states_{s}.npz")
        pool, rows = str(st["pool"]), st["row"]
        sc = st["scene"].astype(str)
        us = np.unique(sc)
        rng = np.random.default_rng(0)
        va_sc = set(rng.choice(us, max(1, len(us) // 5), replace=False).tolist())
        va = np.array([x in va_sc for x in sc])
        fr, ego, tc, cam, nv = load_rows(pool, rows, dev)
        pi, W = plan_fn(pool, dev)
        H = torch.as_tensor(st["H"], device=dev)
        sig = torch.as_tensor(SIG.astype(np.float32), device=dev)
        sgn = torch.as_tensor(st["dir"].astype(np.float32), device=dev)

        def metr(P, m):
            P, Hm = P[m], H[m]
            e = torch.hypot(P[..., 0] - Hm[..., 0], P[..., 1] - Hm[..., 1])
            nin = sgn[m][:, None, None] * torch.stack([-torch.sin(Hm[..., 2]), torch.cos(Hm[..., 2])], -1)
            lat = ((P[..., :2] - Hm[..., :2]) * nin).sum(-1)
            return dict(n=int(m.sum()), ade=float(e.mean()), fde2=float(e[:, 3].mean()), fde4=float(e[:, 7].mean()), ade_turn_max=float(e.mean(1).max()),
                        lat_in_1_2s=float(lat[:, 1:4].mean()), share_inside_cut=float((lat[:, 1:4].mean(1) > 0.3).float().mean()),
                        arc4_minus_H=float((P[:, 7, :2].norm(dim=-1) - Hm[:, 7, :2].norm(dim=-1)).mean()))
        trm, vam = torch.as_tensor(~va, device=dev), torch.as_tensor(va, device=dev)
        r = dict(rows=int(len(rows)), train_rows=int((~va).sum()), val_rows=int(va.sum()), scenes=int(len(us)))
        for tag in a.ref_tags:
            m = RM.load_pmodel(tag, dev).eval()
            with torch.no_grad():
                P = run_plans(m, fr, ego, tc, cam, nv, pi, W)
            r[f"ref_{tag}"] = dict(train=metr(P, trm), val=metr(P, vam))
            del m
        for arm in ("A_head", "B_all"):
            torch.manual_seed(0)
            m = RM.load_pmodel(a.init, dev)
            m.train()
            base, new = m.groups()
            for p in base:
                p.requires_grad_(arm == "B_all")
            groups = [{"params": new, "lr": a.lr_new, "base": a.lr_new}] + ([{"params": base, "lr": a.lr, "base": a.lr}] if arm == "B_all" else [])
            opt = torch.optim.AdamW(groups, weight_decay=0.01)
            tr = np.flatnonzero(~va)
            g = np.random.default_rng(1)
            hist = []
            for step in range(a.steps):
                b = torch.as_tensor(g.choice(tr, min(a.batch, len(tr)), replace=len(tr) < a.batch), device=dev)
                for q in opt.param_groups:
                    q["lr"] = q["base"] * min(1.0, (step + 1) / 20) * 0.5 * (1 + np.cos(np.pi * step / a.steps))
                p = m(fr[b], ego[b], tc[b], None, None, nv=None if nv is None else nv[b]).float()[:, pi].view(-1, 33, 15)
                x, y, psi = T.rear(p, cam[b], W)
                h = lambda z: torch.nn.functional.huber_loss(z, torch.zeros_like(z), reduction="none", delta=1.0)  # noqa: E731
                Hb = H[b]
                loss = (h((x - Hb[..., 0]) / sig[0]) + h((y - Hb[..., 1]) / sig[1]) + h((psi - Hb[..., 2]) / sig[2])).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([q for gr in groups for q in gr["params"]], 1.0)
                opt.step()
                hist.append(float(loss))
                if (step + 1) % 50 == 0:
                    print(f"{s} {arm} step {step + 1}: imit {np.mean(hist[-50:]):.4f}", flush=True)
            m.eval()
            with torch.no_grad():
                P = run_plans(m, fr, ego, tc, cam, nv, pi, W)
            r[arm] = dict(train=metr(P, trm), val=metr(P, vam), loss_first50=float(np.mean(hist[:50])), loss_last50=float(np.mean(hist[-50:])))
            np.savez(o / f"fit_{s}_{arm}{a.sfx}.npz", P=P.cpu().numpy(), val=va, name=st["name"])
            del m, opt
            torch.cuda.empty_cache()
        res[s] = r
        print(s, json.dumps(r, indent=1), flush=True)
    (o / f"fit{a.sfx}.json").write_text(json.dumps(res, indent=1))


# ---------------------------------------------------------------- figures
def cmd_fig(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    o, fd, sfx = Path(a.out), Path(a.figs), a.sfx
    fd.mkdir(parents=True, exist_ok=True)
    P = pd.read_csv(o / f"plugin_cmp{sfx}.csv")
    sets = [s for s in ("nup_held", "nup_train", "pait", "pai_pol") if s in set(P.set)]
    show = ["imit", "road", "agent", "route", "J_S", "J_SR", "J_off"]
    col = {"H": "#222222", "M": "#d62728", "Hin05": "#ff9896", "Hin10": "#e377c2", "Hout10": "#9467bd", "HvM": "#1f77b4", "Hv0": "#17becf", "STR": "#bcbd22"}
    fig, ax = plt.subplots(len(sets), 2, figsize=(13, 2.6 * len(sets)), squeeze=False)
    for i, s in enumerate(sets):
        for j, d in enumerate("LR"):
            A = ax[i, j]
            c = P[(P.set == s) & (P.dir == d)]
            w = 0.8 / len(VARIANTS)
            for k, vn in enumerate(VARIANTS):
                y = [c[(c.variant == vn) & (c.term == t)]["mean"].mean() for t in show]
                A.bar(np.arange(len(show)) + (k - len(VARIANTS) / 2 + 0.5) * w, y, w, color=col[vn], label=vn)
            A.set_yscale("symlog", linthresh=0.01)
            A.set_xticks(np.arange(len(show)), show)
            A.set_title(f"{s}, {'left' if d == 'L' else 'right'} turns (n={int(c.n.max()) if len(c) else 0})", fontsize=9)
            A.grid(axis="y", alpha=0.3)
    ax[0, 0].legend(ncol=4, fontsize=7)
    fig.suptitle("Per-row loss of each candidate plan (terms unweighted; J_* = recipe totals). H = human future; wrong variants should score higher", fontsize=10)
    fig.tight_layout()
    fig.savefig(fd / f"loss_audit_variants{sfx}.png", dpi=130)
    G = pd.read_csv(o / f"grad_bins{sfx}.csv")
    G = G[G["at"] == "M"]
    terms = ["imit", "drv", "road", "agent", "route"]
    fig, ax = plt.subplots(2, len(sets), figsize=(3.6 * len(sets), 6.4), squeeze=False)
    for j, s in enumerate(sets):
        for i, comp in enumerate(("lat", "lon")):
            A = ax[i, j]
            for k, t in enumerate(terms):
                for dd, ls in (("L", "-"), ("R", "--")):
                    g = G[(G.set == s) & (G.dir == dd)].set_index("bin").reindex(list(BINS))
                    A.plot(np.arange(3) + 0.04 * k, g[f"{t}_{comp}"], ls, marker="o", color=f"C{k}", label=f"{t} {dd}" if j == 0 and i == 0 else None)
            A.axhline(0, color="k", lw=0.6)
            A.set_xticks(range(3), ["40-20 m", "20-0 m", "inside"])
            A.set_yscale("symlog", linthresh=0.1)
            A.set_title(f"{s}: {'lateral push (inward +)' if comp == 'lat' else 'along-heading push (faster +)'}", fontsize=9)
            A.grid(alpha=0.3)
    fig.legend(loc="lower center", ncol=5, fontsize=7)
    fig.suptitle("Descent push of each term at the model's plan, by distance to the turn entry (solid left, dashed right; weighted as in training)", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(fd / f"loss_audit_grad{sfx}.png", dpi=130)
    print("figs ->", fd)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pai-extract")
    p.add_argument("--src", required=True), p.add_argument("--out", required=True)
    p = sub.add_parser("collect")
    p.add_argument("--out", required=True), p.add_argument("--pai-pol", default="")
    p = sub.add_parser("plans")
    p.add_argument("--out", required=True), p.add_argument("--sets", nargs="+", default=["pait"])
    p.add_argument("--tags", nargs="+", default=["RSG-L18-s0", "RBSG2-L18-s0"])
    p = sub.add_parser("audit")
    p.add_argument("--out", required=True), p.add_argument("--sets", nargs="+", default=["nup_held", "nup_held2", "nup_train", "pait", "pai_pol"])
    p.add_argument("--pait-model", default="RSG-L18-s0"), p.add_argument("--threads", type=int, default=8)
    p = sub.add_parser("fit")
    p.add_argument("--out", required=True), p.add_argument("--sets", nargs="+", default=["nup_held", "pait"])
    p.add_argument("--init", default="RB0-F-s0"), p.add_argument("--ref-tags", nargs="+", default=["RB0-F-s0", "RSG-L18-s0"])
    p.add_argument("--steps", type=int, default=400), p.add_argument("--batch", type=int, default=48)
    p.add_argument("--lr", type=float, default=3e-5), p.add_argument("--lr-new", type=float, default=3e-4)
    p.add_argument("--sfx", default="", help="suffix of fit.json / fit_*.npz (e.g. _s3000)")
    p = sub.add_parser("fig")
    p.add_argument("--out", required=True), p.add_argument("--figs", required=True), p.add_argument("--sfx", default="")
    a = ap.parse_args()
    {"pai-extract": cmd_pai_extract, "collect": cmd_collect, "plans": cmd_plans, "audit": cmd_audit, "fit": cmd_fit, "fig": cmd_fig}[a.cmd](a)
