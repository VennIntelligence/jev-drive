"""op_parity lane DIAG1 (plans/2026-10-09-diag1-prereg.md, results/diag1_longitudinal.md): FIX1's serving fixes as a post-processing of the WOD
open-loop predictions (job 1), and the longitudinal audit of shipped vs nuPlan-adapted plans on identical frames (job 2). No training.

  wod-fix      (jevdrive env, CPU) job 1: (a) speed-continuous re-timing, (b) the lead cap when lane FIX1's law is importable, on the stored WOD val
               predictions -> results/diag1/wod_fix_{rfs,ade}.{csv,md}, wod_fix_gate.json
  nav-extract  (op-train env, one GPU: a pool job) navtest plans of shipped / adapted / the D5 switch variants on the pp_prep warp cache, the shipped
               lead outputs -> $DATA_DIR/runs/op_parity/diag1/nav.npz (+ identity gate against the stored plans)
  wod-tab      (jevdrive env, CPU) the WOD val frames in the common table form -> $DATA_DIR/runs/op_parity/diag1/tab_wod.npz
  nav-tab      (.venv, CPU) nav.npz -> tab_nav.npz
  alp-tab      (.venv, CPU) the AlpaSim decision tables of scripts/../alpasim/scripts/diag1_alp.py -> tab_alp_nuplan.npz / tab_alp_pai.npz
  tables       (.venv, CPU) D1 - D5 over every tab_*.npz -> results/diag1/*.{csv,md}, figures
  nav-swap     (.venv, CPU) D6 pose keys for `jevdrive.bench score-poses` -> diag1/swap_poses.npz
  swap-report  (.venv / jevdrive env, CPU) D6: navtest (score CSV) and WOD (RFS) swaps, shares, ceilings -> results/diag1/swap_*.{csv,md}

Common table (one row per frame, every domain): unit (bootstrap cluster), v0, T = (0.5, 1, 1.5, 2, 4) s, arc_<arm> (N, 5) arc length of the plan
polyline from the origin at T, log_arc (N, 5), lead p / d / v_l of the shipped lead head (D3), acc0.
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_pl.Path(__file__).parent)]
import argparse, json, os  # noqa: E401,E402

import numpy as np  # noqa: E402

DATA = _pl.Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
WD = DATA / "runs/op_parity/diag1"
OUT = _R / "experiments/op_parity/results/diag1"
TT = np.array([0.5, 1.0, 1.5, 2.0, 4.0])
ARMS_W = {"shipped": ["shipped"], "P2H10": ["P2H10-F-s0", "P2H10-F-s1"], "SH30": ["SH30-F-s0", "SH30-F-s1"]}
TAU = 1.0                                                   # col1_pai_driver.TAU default = FIX1 serve_fix T_TRACK
WOD_FRONT = 3.9                                             # WOD ego front bumper ahead of the rear axle (m), approximate (+- 0.2 m)


# ---------------------------------------------------------------- re-timing (job 1 a)
def retime_xy(wp, v0, tau=TAU, seg=0.25, dt=0.05):
    """The form of experiments/alpasim/lib/col1_pai_driver.py::retime (= serve_fix.join + place) on an (n, 2) waypoint grid of spacing `seg`: the
    polyline through the origin and the waypoints is kept; the speed profile starts at v0 and blends linearly into the plan's own segment
    speeds over tau seconds; beyond the plan's end the path continues straight along its last moving segment. A given profile v (on the
    dt-midpoint grid) can be passed instead of v0 as `v0` = array."""
    wp = np.asarray(wp, np.float64)
    n, k = len(wp), int(round(seg / dt))
    p = np.concatenate([np.zeros((1, 2)), wp])
    d = np.diff(p, axis=0)
    ds = np.hypot(d[:, 0], d[:, 1])
    s = np.concatenate([[0.0], np.cumsum(np.maximum(ds, 1e-6))])
    t = np.arange(dt / 2, n * seg, dt)
    vp = (ds / seg)[np.minimum((t / seg).astype(int), n - 1)]
    v = np.asarray(v0, np.float64) if np.ndim(v0) else v0 + (vp - v0) * np.minimum(t / max(tau, 1e-3), 1.0)
    sn = np.cumsum(np.maximum(v, 0.0) * dt)[k - 1::k]
    mv = np.flatnonzero(ds > 1e-3)
    u = d[mv[-1]] / ds[mv[-1]] if len(mv) else np.array([1.0, 0.0])
    P, S = np.concatenate([p, (p[-1] + 100.0 * u)[None]]), np.concatenate([s, [s[-1] + 100.0]])
    return np.stack([np.interp(sn, S, P[:, 0]), np.interp(sn, S, P[:, 1])], 1), vp, t


def arc_at(wp, seg, T=TT):
    """(N, n, 2) waypoints at seg, 2 seg, ... -> arc length of the polyline from the origin at times T (N, len(T))."""
    p = np.concatenate([np.zeros_like(wp[:, :1]), wp], 1)
    s = np.concatenate([np.zeros((len(wp), 1)), np.cumsum(np.linalg.norm(np.diff(p, axis=1), axis=-1), 1)], 1)
    tg = seg * np.arange(p.shape[1])
    return np.stack([np.interp(T, tg, r) for r in s])


# ---------------------------------------------------------------- job 1
_MODS = None


def _mods_call(p):
    return _MODS(p)


def wod_ctx():
    from wod_launch_report import Ctx
    from jevdrive import waymo as W
    from jevdrive import wod_zeroshot as Z
    C = Ctx()
    S = Z.load_sets()
    past = np.concatenate([S["rater"]["past"], S["extra"]["past"]])
    kin = W.past_kinematics(past)
    C.vfed, C.v_init, C.acc = kin["v"].astype(np.float64), W.init_speed(past).astype(np.float64), past[:, -1, 4].astype(np.float64)
    z = []
    for nm in C.names:
        with np.load(Z.root("preds", "op_cinque") / f"{nm}.npz") as x:
            z.append({k: np.asarray(x[k]) for k in ("lead_prob", "lead", "dev_xy", "plan_vel")})
    C.lp = np.stack([np.asarray(x["lead_prob"], np.float64).reshape(-1) for x in z])            # (N, 3) probabilities (decode's sigmoid)
    C.lead = np.stack([np.asarray(x["lead"], np.float64).reshape(3, 6, 4) for x in z])          # (N, 3 hyp, 6 times, x y v a), camera frame
    C.dev = np.stack([np.asarray(x["dev_xy"], np.float64) for x in z])
    C.pvel0 = np.stack([np.asarray(x["plan_vel"], np.float64)[0, 0] for x in z])                # the model's own ego speed (plan velocity at t = 0)
    return C


def cmd_wod_fix(a):
    from jevdrive import stats
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("op_parity", "diag1/wod-fix", seed=0, config=vars(a)) as run:
        run.use_split(splits.load("wod/val"))
        C = wod_ctx()
        n, N = C.n, len(C.names)
        P = {k: [C.preds(t, True) for t in tags] for k, tags in ARMS_W.items()}
        fix = None
        if a.lead:
            import diag1_lead as DL                           # thin wrapper over lane FIX1's committed law (experiments/alpasim/lib/serve_fix.py)
            fix = DL
        def mods(p):
            m = {"a": np.stack([retime_xy(p[i], C.vfed[i])[0] for i in range(N)]),
                 "a_tau0.5": np.stack([retime_xy(p[i], C.vfed[i], 0.5)[0] for i in range(N)]),
                 "a_tau2.0": np.stack([retime_xy(p[i], C.vfed[i], 2.0)[0] for i in range(N)]),
                 "a_vinit": np.stack([retime_xy(p[i], C.v_init[i])[0] for i in range(N)]),
                 "id_tau0": np.stack([retime_xy(p[i], C.vfed[i], 1e-3)[0] for i in range(N)]),
                 "id_vp0": np.stack([retime_xy(p[i], np.hypot(*p[i, 0]) / 0.25)[0] for i in range(N)])}
            if fix is not None:
                for nm, vc, tk in (("b", 0.0, 0), ("ab", TAU, 0), ("ab_t2", TAU, 2), ("ab_t20", TAU, 20)):
                    inf = {}
                    m[nm] = fix.wod_cap(C, p, vcont=vc, ticks=tk, info=inf)
                    m["_info_" + nm] = inf
            return m
        from concurrent.futures import ProcessPoolExecutor
        flat = [(k, p) for k, ps in P.items() for p in ps]
        if fix is not None:
            global _MODS
            _MODS = mods
            with ProcessPoolExecutor(len(flat)) as ex:
                res = list(ex.map(_mods_call, [p for _, p in flat]))
        else:
            res = [mods(p) for _, p in flat]
        M = {k: [r for (kk, _), r in zip(flat, res) if kk == k] for k in P}
        capinfo = {k: {nm[6:]: {"share of frames with > 0.1 m removed": float(np.mean([(m[nm]["cut_m"] > 0.1).mean() for m in M[k]])),
                                "mean m removed on those": float(np.mean([m[nm]["cut_m"][m[nm]["cut_m"] > 0.1].mean() for m in M[k]])),
                                "share lead MPC below plan accel": float(np.mean([m[nm]["mpc"].mean() for m in M[k]]))}
                       for nm in M[k][0] if nm.startswith("_info_")} for k in P}
        for k in P:
            for m in M[k]:
                for nm in [x for x in m if x.startswith("_info_")]:
                    del m[nm]
        rfs0 = {k: np.mean([C.rfs(p) for p in ps], 0) for k, ps in P.items()}
        gate = {"G0_rfs": {k: C.cm(v) for k, v in rfs0.items()},
                "G1_registered_v0_eq_first_segment_max_m": {k: float(max(np.abs(m["id_vp0"] - p).max() for m, p in zip(M[k], P[k]))) for k in P},
                "G1_operator_identity_tau0_max_m": {k: float(max(np.abs(m["id_tau0"] - p).max() for m, p in zip(M[k], P[k]))) for k in P}, "lead_cap": capinfo}
        run.info("gate %s", gate)
        lead = C.lp[:n, 0] > 0.5
        st = dict(C.st)
        strata = {"all": st["all"], "standstill (v0 < 0.5)": st["stopped"], "moving (v0 >= 0.5)": st["moving (v>=0.5)"], "lead present": lead, "no lead": ~lead,
                  "0.5-5 m/s": st["slow 0.5-5"], "5-12 m/s": st["mid 5-12"], ">= 12 m/s": st["fast >=12"],
                  "standstill, lead": st["stopped"] & lead, "standstill, no lead": st["stopped"] & ~lead, "moving, lead": st["moving (v>=0.5)"] & lead,
                  "moving, no lead": st["moving (v>=0.5)"] & ~lead}
        primary = ("a", "ab")
        rows, ade = [], []
        err0 = {k: np.mean([np.linalg.norm(p - C.fut, axis=-1)[:, :12].mean(1) for p in ps], 0) for k, ps in P.items()}
        for k in P:
            for mod in [m for m in M[k][0] if not m.startswith("id_")]:
                sc = [C.rfs(m[mod]) for m in M[k]]
                d = np.mean(sc, 0) - rfs0[k]
                for nm, msk in strata.items():
                    p, lo, hi = C.ci(d, msk)
                    lab = ""
                    if nm == "all" and k != "shipped" and mod in primary:
                        lab = "helps" if lo > 0 else "hurts" if hi < 0 else "no measurable change"
                    rows.append({"arm": k, "fix": mod, "stratum": nm, "n": int(msk.sum()), "RFS": C.cm(rfs0[k], msk), "RFS fixed": C.cm(np.mean(sc, 0), msk),
                                 "dRFS": p, "lo": lo, "hi": hi, "seeds": "/".join(f"{C.cm(s - s0, msk):+.3f}" for s, s0 in zip(sc, [C.rfs(q) for q in P[k]])) if len(sc) > 1 else "",
                                 "label": lab})
                e = np.mean([np.linalg.norm(m[mod] - C.fut, axis=-1)[:, :12].mean(1) for m in M[k]], 0)
                dd = e - err0[k]
                b = (C.Ka @ dd) / C.Ka.sum(1)
                mv = np.concatenate([C.v_init >= 0.5])
                ade.append({"arm": k, "fix": mod, "ADE@3s": float(err0[k].mean()), "ADE@3s fixed": float(e.mean()), "d (m)": float(dd.mean()),
                            "lo": float(np.percentile(b, 2.5)), "hi": float(np.percentile(b, 97.5)), "d standstill (m)": float(dd[~mv].mean()), "d moving (m)": float(dd[mv].mean())})
        OUT.mkdir(parents=True, exist_ok=True)
        stats.write_table(rows, OUT / ("wod_fix_rfs" + a.suffix))
        stats.write_table(ade, OUT / ("wod_fix_ade" + a.suffix))
        # geometry of the change: plan arc at 1 / 3 / 5 s before and after (a), per v0 bin
        geo = []
        bins = [("< 0.5", C.vfed < 0.5), ("0.5-5", (C.vfed >= 0.5) & (C.vfed < 5)), ("5-12", (C.vfed >= 5) & (C.vfed < 12)), (">= 12", C.vfed >= 12)]
        T3 = np.array([0.25, 1.0, 3.0, 5.0])
        la = arc_at(C.fut, 0.25, T3)
        for k in P:
            a0 = np.mean([arc_at(p, 0.25, T3) for p in P[k]], 0)
            a1 = np.mean([arc_at(m["a"], 0.25, T3) for m in M[k]], 0)
            a2 = np.mean([arc_at(m["ab"], 0.25, T3) for m in M[k]], 0) if "ab" in M[k][0] else None
            for nm, msk in bins:
                geo.append({"arm": k, "v0 bin": nm, "n": int(msk.sum()), "first-seg speed / v0": float(np.median((a0[msk, 0] / 0.25) / np.maximum(C.vfed[msk], 0.5))),
                            "arc 1 s": float(a0[msk, 1].mean()), "arc 1 s (a)": float(a1[msk, 1].mean()), "arc 3 s": float(a0[msk, 2].mean()), "arc 3 s (a)": float(a1[msk, 2].mean()),
                            "arc 5 s": float(a0[msk, 3].mean()), "arc 5 s (a)": float(a1[msk, 3].mean()),
                            "arc 3 s (a+b)": float(a2[msk, 2].mean()) if a2 is not None else np.nan, "arc 5 s (a+b)": float(a2[msk, 3].mean()) if a2 is not None else np.nan, "log arc 3 s": float(la[msk, 2].mean()), "log arc 5 s": float(la[msk, 3].mean())})
        stats.write_table(geo, OUT / ("wod_fix_geom" + a.suffix))
        (OUT / ("wod_fix_gate" + a.suffix + ".json")).write_text(json.dumps(gate, indent=1))
        run.summary.update(gate=gate)

# ---------------------------------------------------------------- navtest extraction (GPU)
NAV_TAGS = ("P2H10-F-s0", "P2H10-F-s1", "SH30-F-s0", "SH30-F-s1")
VARIANTS = ("main", "nobias", "const", "resid", "posecv", "gate")
PX, PY, PYAW = slice(8, 20, 3), slice(9, 20, 3), slice(10, 20, 3)     # ego feature layout (parity_adapter.ego_features)
T_HIST = np.array([-1.5, -1.0, -0.5, 0.0], np.float32)
T8 = 0.5 * np.arange(1, 9)
NAV_FRONT, NAV_CAM_X = 1.461 + 5.176 / 2, 1.786             # nuPlan ego: front bumper and CAM_F0 ahead of the rear axle (col1_lead.py)


def cmd_nav_extract(a):
    import torch
    from jevdrive import op_interp as I
    from jevdrive.bench import navsim as N
    from jevdrive.bench.models import resolve
    from jevdrive.data import splits
    from jevdrive.run import Run
    out = WD / ("nav.npz" if not a.limit else "nav-smoke.npz")
    with Run("op_parity", "diag1/nav-extract", seed=0, config=vars(a)) as run:
        run.use_split(splits.load("navsim/navtest"))
        N._pp_path()
        import pp_train as T
        dev = torch.device("cuda")
        S = T.Store(["lb_navtest"], dev, need_side=False, frames="warp")
        tb, n = S.tb, (min(S.n, a.limit) if a.limit else S.n)
        ecv = S.ego.clone()
        ecv[:, PX] = ecv[:, 4:5] * torch.from_numpy(T_HIST).to(ecv)[None]
        ecv[:, PY] = 0
        ecv[:, PYAW] = 0

        def fwd(model, ego=None, **kw):
            sl = model.net.slices
            pi = np.arange(sl["plan"].start, sl["plan"].start + 495)
            mu, lead, lp = np.zeros((n, 33, 15), np.float32), np.zeros((n, 144), np.float32), np.zeros((n, 3), np.float32)
            with torch.no_grad():
                for i in range(0, n, 128):
                    r = torch.arange(i, min(i + 128, n), device=dev)
                    o = model(S.front[r], (S.ego if ego is None else ego)[r], S.tc[r], None, None, **kw).float().cpu().numpy()
                    mu[i:i + len(r)], lead[i:i + len(r)], lp[i:i + len(r)] = o[:, pi].reshape(-1, 33, 15), o[:, sl["lead"]], o[:, sl["lead_prob"]]
            return mu, lead, lp

        def poses(mu):
            return np.stack([I.to_rear(mu[i, :, 0:3], mu[i, :, 11], I.T_IDXS, tb["cam"][i, :2], T8, "lever") for i in range(n)])
        R = dict(tokens=tb["names"][:n], log=tb["log"][:n], speed=tb["speed"][:n], fut=tb["fut"][:n], cam=tb["cam"][:n], ego=tb["ego"][:n], acc=tb["acc"][:n, -1])
        gate = {}
        mu, lead, lp = fwd(T.load_pmodel("P0", dev))
        R.update(pose_P0=poses(mu), lead_P0=lead, lp_P0=lp, pvel0_P0=mu[:, 0, 3], pacc0_P0=mu[:, 0, 6])
        for tag in NAV_TAGS:
            model = N._load_ckpt(T, resolve(f"{tag}@warp", check=True).ckpt, dev)
            assert model.adapter is not None and not model.adapter.use_side and not model.mem, tag
            mb = N.nav_mean_bias(model, S, tag, dev, save=False)
            for var in VARIANTS:
                model.gate, model.bias_sub = 0.0, None
                if "apply" in model.adapter.__dict__:
                    del model.adapter.__dict__["apply"]
                kw, ego = {}, None
                if var == "nobias":
                    kw = dict(inputs_on=False)
                elif var == "const":
                    model.adapter.__dict__["apply"] = lambda H, ego, side=None, side_mask=None, _b=mb: H + _b[None, None].to(H.dtype)
                elif var == "resid":
                    model.bias_sub = mb
                elif var == "posecv":
                    ego = ecv
                elif var == "gate":
                    model.gate = 0.5
                mu, lead, lp = fwd(model, ego, **kw)
                R[f"pose_{tag}_{var}"] = poses(mu)
                if var == "main":
                    R[f"lead_{tag}"], R[f"lp_{tag}"], R[f"pvel0_{tag}"] = lead, lp, mu[:, 0, 3]
                    ref = WD.parent / "self_consist/infer" / f"{tag}-warp__lb_navtest.npz"
                    if ref.exists():
                        z = np.load(ref)
                        assert z["names"][:n].tolist() == tb["names"][:n].tolist()
                        gate[tag] = float(np.abs(z["plan_pos"][:n] - mu[:, :, 0:3]).max())
                    pf = N.pred_file(resolve(f"{tag}@warp"), "navtest")
                    if pf.exists():
                        z = np.load(pf)
                        ix = {t: i for i, t in enumerate(z["tokens"].astype(str))}
                        gate[tag + "_poses_vs_bench"] = float(np.abs(z["poses"][[ix[t] for t in tb["names"][:n]]] - R[f"pose_{tag}_main"]).max())
                run.info("%s %s done", tag, var)
            del model
            torch.cuda.empty_cache()
        run.info("gate (max |d| m vs stored plans) %s", gate)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, gate=np.array(json.dumps(gate)), **R)
        run.summary.update(gate=gate, n=int(n))
        assert all(v < 0.03 for v in gate.values()), gate

# ---------------------------------------------------------------- common tables
def sigm(x):
    return 1 / (1 + np.exp(-np.asarray(x, np.float64)))


def save_tab(name, arcs: dict, **kw):
    WD.mkdir(parents=True, exist_ok=True)
    np.savez(WD / f"tab_{name}.npz", arms=np.array(list(arcs)), **{f"arc__{k}": v.astype(np.float32) for k, v in arcs.items()}, **kw)
    print(f"tab_{name}: {len(kw['v0'])} rows, arms {list(arcs)}", flush=True)


def cmd_wod_tab(a):
    from jevdrive import wod_zeroshot as Z
    C = wod_ctx()
    arcs = {"shipped": arc_at(C.preds("shipped", True), 0.25)}
    vmap = {"nobias": "zero", "const": "biasmean", "resid": "biasresid", "posecv": "posecv"}
    for fam in ("P2H10", "SH30"):
        for s in (0, 1):
            tag, k = f"{fam}-F-s{s}", f"{fam}-s{s}"
            main = C.preds(tag, True)
            arcs[k] = arc_at(main, 0.25)
            for var, dv in vmap.items():
                if Z.root("preds", f"op_cinque_dx-{tag}_{dv}").exists() and len(list(Z.root("preds", f"op_cinque_dx-{tag}_{dv}").glob("*.npz"))) >= len(C.names):
                    p = C.preds(f"dx-{tag}_{dv}", True)
                    arcs[f"{k}:{var}"] = arc_at(p, 0.25)
                    if var == "nobias":                               # gate = adapter off below 0.5 m/s fed speed (bias held per target: exact composition)
                        arcs[f"{k}:gate"] = arc_at(np.where((C.vfed < 0.5)[:, None, None], p, main), 0.25)
    ego = np.load(DATA / "runs/op_parity/wod/bias-P2H10-F-s0.npz")
    assert ego["names"].astype(str).tolist() == C.names.tolist()
    rater = np.arange(len(C.names)) < C.n
    save_tab("wod", arcs, unit=C.seq, v0=C.vfed, v0_alt=C.v_init, log_arc=arc_at(C.fut, 0.25), lp=C.lp[:, 0], lead_x=C.lead[:, 0, 0, 0], lead_v=C.lead[:, 0, 0, 2],
             lead_a=C.lead[:, 0, 0, 3], pvel0=C.pvel0, c2f=WOD_FRONT - C.dev[:, 0], ego=ego["ego"], rater=rater, names=C.names)


def cmd_nav_tab(a):
    from jevdrive.bench import navsim as N
    from jevdrive.bench.models import resolve
    z = np.load(WD / "nav.npz")
    tok = z["tokens"].astype(str)
    arcs = {"shipped": arc_at(z["pose_P0"][..., :2].astype(np.float64), 0.5)}
    for tag in NAV_TAGS:
        k = tag.replace("-F-", "-")
        for var in VARIANTS:
            arcs[k if var == "main" else f"{k}:{var}"] = arc_at(z[f"pose_{tag}_{var}"][..., :2].astype(np.float64), 0.5)
    for spec in ("AP2H10-AB-s0", "AP2H10-AB-s1", "APY10m10-AB-s0", "APY10m10-AB-s1"):          # their own input standard, stored bench plans
        pf = N.pred_file(resolve(f"{spec}@warp"), "navtest")
        if pf.exists():
            p = np.load(pf)
            ix = {t: i for i, t in enumerate(p["tokens"].astype(str))}
            if all(t in ix for t in tok):
                arcs[spec.replace("-AB-", "-")] = arc_at(p["poses"][[ix[t] for t in tok]][..., :2].astype(np.float64), 0.5)
    ld = z["lead_P0"][:, :72].reshape(-1, 3, 6, 4)
    save_tab("nav", arcs, unit=z["log"].astype(str), v0=z["speed"].astype(np.float64), log_arc=arc_at(z["fut"][..., :2].astype(np.float64), 0.5),
             lp=sigm(z["lp_P0"][:, 0]), lead_x=ld[:, 0, 0, 0], lead_v=ld[:, 0, 0, 2], lead_a=ld[:, 0, 0, 3], pvel0=z["pvel0_P0"], c2f=NAV_FRONT - z["cam"][:, 0], ego=z["ego"],
             names=tok)


def cmd_alp_tab(a):
    """The per-decision tables of experiments/alpasim/scripts/diag1_alp.py -> one common table per (track, group kind, served family)."""
    for track in ("nuplan", "pai"):
        f = WD / "alp" / f"{track}.npz"
        if not f.exists():
            print("missing", f)
            continue
        z = np.load(f, allow_pickle=True)
        grp, tag = z["group"].astype(str), z["tag"].astype(str)
        fam = np.array([t.split("-s")[0].replace("-F", "").replace("-AB", "") for t in tag])
        kind = np.array(["ctrl" if g in ("ctrl", "extra") else "base" if track == "pai" else "coll" for g in grp])
        if "n_slots" in z.files:                                       # PAI cold start: the first 7 decisions of a rollout have fewer than 8 real slots (and zero ego input at k = 0)
            kind = np.where(z["n_slots"] >= 8, kind, "cold")
        ld = z["lead_p0"][:, :72].reshape(-1, 3, 6, 4)
        front = (NAV_FRONT - NAV_CAM_X) if "c2f" not in z.files else z["c2f"]
        for kd in [k for k in np.unique(kind) if k != "cold"]:
            for fm in np.unique(fam[kind == kd]):
                m = (kind == kd) & (fam == fm)
                arcs = {"shipped": arc_at(z["plan_p0"][m][..., :2].astype(np.float64), 0.5), f"{fm}-s0": arc_at(z["plan_ft"][m][..., :2].astype(np.float64), 0.5)}
                for var in ("nobias", "gate", "posecv"):
                    if f"plan_{var}" in z.files and np.isfinite(z[f"plan_{var}"][m]).all():
                        arcs[f"{fm}-s0:{var}"] = arc_at(z[f"plan_{var}"][m][..., :2].astype(np.float64), 0.5)
                la = np.full((m.sum(), 5), np.nan)
                la[:, [0, 1, 3, 4]] = z["log_arc"][m]
                ra = np.full((m.sum(), 5), np.nan)
                ra[:, [0, 1, 3, 4]] = z["roll_arc"][m]
                kw = dict(unit=np.char.add(np.char.add(grp[m], "/"), z["scene"][m].astype(str)), v0=z["v0"][m].astype(np.float64), log_arc=la, roll_arc=ra, lp=sigm(z["lp_p0"][m][:, 0]),
                          lead_x=ld[m][:, 0, 0, 0], lead_v=ld[m][:, 0, 0, 2], lead_a=ld[m][:, 0, 0, 3], c2f=np.broadcast_to(front, (len(m),))[m] if np.ndim(front) else np.full(m.sum(), front))
                for k in ("v0_sim", "ego", "sim_gap", "sim_closing", "pvel0"):
                    if k in z.files:
                        kw[k] = z[k][m]
                save_tab(f"alp_{track}_{kd}_{fm}", arcs, **kw)


# ---------------------------------------------------------------- D1 - D5
BINS = [("0.5-2", 0.5, 2), ("2-5", 2, 5), ("5-12", 5, 12), ("12-16", 12, 16), (">=16", 16, 1e9)]
NB = 2000


def load_tab(f):
    z = np.load(f, allow_pickle=True)
    t = {k: z[k] for k in z.files if not k.startswith("arc__")}
    t["arc"] = {k[5:]: z[k].astype(np.float64) for k in z.files if k.startswith("arc__")}
    t["dom"] = _pl.Path(f).stem[4:]
    return t


def fams(t):
    """{family: [seed arms]} of the main (variant-free) adapted arms."""
    out = {}
    for k in t["arc"]:
        if k != "shipped" and ":" not in k:
            out.setdefault(k.rsplit("-s", 1)[0], []).append(k)
    return out


def q_r0(arc, v0):
    return (arc[:, 0] / 0.5) / np.maximum(v0, 1e-3) - 1


def q_acc(arc, v0):
    return ((arc[:, 3] - arc[:, 2]) / 0.5 - v0) / 1.75


def cmd_tables(a):
    from jevdrive import stats
    tabs = [load_tab(f) for f in sorted(WD.glob("tab_*.npz"))]
    OUT.mkdir(parents=True, exist_ok=True)
    d1, d2, d3, d3c, d4, d5, chk = [], [], [], [], [], [], []
    for t in tabs:
        dom, v0, u, A = t["dom"], t["v0"].astype(np.float64), t["unit"].astype(str), t["arc"]
        F = fams(t)
        la = t["log_arc"].astype(np.float64)
        gm = lambda ks, f: np.mean([f(A[k]) for k in ks], 0)  # noqa: E731
        # ---- units / frames check
        row = {"domain": dom, "n": len(v0), "units": len(set(u)), "v0 mean": float(v0.mean()), "share v0 < 0.5": float((v0 < 0.5).mean())}
        if "ego" in t:
            e = t["ego"].astype(np.float64)
            mv = v0 >= 2
            row |= {"fed vx*10 / v0 (median, v0 >= 2)": float(np.median(e[mv, 4] * 10 / v0[mv])), "ax/3 std": float(e[:, 6].std()), "ay/3 std": float(e[:, 7].std()),
                    "hist x(-1.5 s)*10 / (-1.5 v0) (median)": float(np.median(e[mv, 8] * 10 / (-1.5 * v0[mv]))), "|hist yaw| p99 (deg)": float(np.degrees(np.percentile(np.abs(e[:, 10]), 99)))}
        if "pvel0" in t:
            mv = v0 >= 2
            row["model's own speed / v0 (median, v0 >= 2)"] = float(np.median(t["pvel0"][mv] / v0[mv]))
        if "v0_sim" in t:
            row["median |v0 - v0_sim|"] = float(np.nanmedian(np.abs(v0 - t["v0_sim"])))
        row["lead frames (p > 0.5)"] = float((t["lp"] > 0.5).mean())
        chk.append(row)
        # ---- D1 / D2
        for nm, lo, hi in BINS + [(">=2 (all moving)", 2, 1e9)]:
            m = (v0 >= lo) & (v0 < hi)
            if m.sum() < 20:
                continue
            rs = q_r0(A["shipped"], v0)
            base = {"domain": dom, "v0 bin": nm, "n": int(m.sum())}
            if lo >= 2:
                d1.append(base | {"arm": "shipped", "median r0 - 1": float(np.median(rs[m])), "mean r0 - 1": float(rs[m].mean())})
                for fm, ks in F.items():
                    ra = gm(ks, lambda x: q_r0(x, v0))
                    p = stats.paired(ra[m], rs[m], groups=u[m], n_boot=NB)
                    d1.append(base | {"arm": fm, "median r0 - 1": float(np.median(ra[m])), "mean r0 - 1": float(ra[m].mean()), "adapted - shipped": p["mean"], "lo": p["lo"], "hi": p["hi"],
                                      "share frames adapted > shipped": float((ra[m] > rs[m]).mean())})
            for arm, ks in [("shipped", ["shipped"])] + list(F.items()):
                ar = np.mean([A[k] for k in ks], 0)
                r = base | {"arm": arm}
                for j, T in ((1, "1 s"), (3, "2 s"), (4, "4 s")):
                    ok = m & np.isfinite(la[:, j])
                    if ok.sum() < 20:
                        continue
                    r[f"arc / log {T} (ratio of sums)"] = float(ar[ok, j].sum() / la[ok, j].sum())
                    ok2 = ok & (la[:, j] >= 1)
                    r[f"arc / log {T} (median)"] = float(np.median(ar[ok2, j] / la[ok2, j])) if ok2.any() else np.nan
                    r[f"arc / shipped {T}"] = float(ar[m, j].sum() / A["shipped"][m, j].sum())
                d2.append(r)
        # ---- D3
        for sens, pth, ath, cform in (("registered", 0.5, -0.3, "v0 - lead v"), ("p > 0.7, a <= -0.5", 0.7, -0.5, "v0 - lead v"), ("radard closing", 0.5, -0.3, "model speed - lead v")):
            if cform != "v0 - lead v" and "pvel0" not in t:
                continue
            d = t["lead_x"].astype(np.float64) - t["c2f"]
            c = (v0 if cform == "v0 - lead v" else t["pvel0"].astype(np.float64)) - t["lead_v"].astype(np.float64)
            ttc = d / np.maximum(c, 0.1)
            cl = (t["lp"] > pth) & (v0 >= 2) & (c >= 1) & (ttc <= 8)
            if cl.sum() < 15:
                continue
            aS, aL = q_acc(A["shipped"], v0), q_acc(la, v0)
            need = np.clip(c ** 2 / (2 * np.maximum(d - 2, 0.5)), 0, 8)
            for fm, ks in F.items():
                aA = gm(ks, lambda x: q_acc(x, v0))
                so = np.mean([(aS <= ath) & ~(q_acc(A[k], v0) <= ath) for k in ks], 0)
                ao = np.mean([~(aS <= ath) & (q_acc(A[k], v0) <= ath) for k in ks], 0)
                bo = np.mean([(aS <= ath) & (q_acc(A[k], v0) <= ath) for k in ks], 0)
                bs, ba, bd = (stats.bootstrap(x[cl], groups=u[cl], n_boot=NB) for x in (so, ao, so - ao))
                okl = cl & np.isfinite(aL)
                sl = lambda y, mm: float(np.polyfit(need[mm], y[mm], 1)[0]) if mm.sum() > 10 else np.nan  # noqa: E731
                d3.append({"domain": dom, "arm": fm, "definition": sens, "closing-lead frames": int(cl.sum()), "units": len(set(u[cl])), "share of all frames": float(cl.mean()),
                           "mean a shipped": float(aS[cl].mean()), "mean a adapted": float(aA[cl].mean()), "mean a log": float(aL[okl].mean()) if okl.any() else np.nan,
                           "shipped slows": float((aS[cl] <= ath).mean()), "adapted slows": float(np.mean([(q_acc(A[k], v0)[cl] <= ath).mean() for k in ks])),
                           "log slows": float((aL[okl] <= ath).mean()) if okl.any() else np.nan,
                           "shipped slows, adapted not": bs["mean"], "s lo": bs["lo"], "s hi": bs["hi"], "adapted slows, shipped not": ba["mean"], "a lo": ba["lo"], "a hi": ba["hi"],
                           "difference": bd["mean"], "d lo": bd["lo"], "d hi": bd["hi"], "both slow": float(bo[cl].mean()),
                           "slope a on a_need shipped": sl(aS, cl), "slope adapted": sl(aA, cl), "slope log": sl(aL, okl)})
                if sens == "registered":
                    for nm, lo, hi in (("< 2 s", -1, 2), ("2-4 s", 2, 4), ("4-8 s", 4, 8.001)):
                        m = cl & (ttc >= lo) & (ttc < hi)
                        if m.sum() >= 8:
                            p = stats.paired(aA[m], aS[m], groups=u[m], n_boot=NB)
                            d3c.append({"domain": dom, "arm": fm, "time to contact": nm, "n": int(m.sum()), "mean gap (m)": float(d[m].mean()), "mean closing (m/s)": float(c[m].mean()),
                                        "mean v0": float(v0[m].mean()), "a shipped": float(aS[m].mean()), "a adapted": float(aA[m].mean()),
                                        "a log": float(np.nanmean(aL[m])), "adapted - shipped": p["mean"], "lo": p["lo"], "hi": p["hi"],
                                        "first-seg r0 - 1 shipped": float(q_r0(A["shipped"], v0)[m].mean()), "first-seg r0 - 1 adapted": float(gm(ks, lambda x: q_r0(x, v0))[m].mean())})
        # ---- D4
        d = t["lead_x"].astype(np.float64) - t["c2f"]
        ss = v0 < 0.5
        for nm, m in (("stopped lead (p > 0.5, gap < 15 m, lead v < 1)", ss & (t["lp"] > 0.5) & (d < 15) & (t["lead_v"] < 1)), ("no lead (p <= 0.5)", ss & (t["lp"] <= 0.5)),
                      ("all standstill", ss)):
            if m.sum() < 8:
                continue
            for arm, ks in [("shipped", ["shipped"])] + list(F.items()) + [("log", None)]:
                ar = la if ks is None else np.mean([A[k] for k in ks], 0)
                r = {"domain": dom, "stratum": nm, "n": int(m.sum()), "arm": arm, "arc 2 s mean (m)": float(np.nanmean(ar[m, 3])), "arc 2 s median": float(np.nanmedian(ar[m, 3])),
                     "arc 4 s mean (m)": float(np.nanmean(ar[m, 4])), "share arc 2 s > 0.5 m": float(np.nanmean(ar[m, 3] > 0.5)), "share arc 4 s > 2 m": float(np.nanmean(ar[m, 4] > 2))}
                if ks is not None and arm != "shipped":
                    p = stats.paired(ar[m, 3], A["shipped"][m, 3], groups=u[m], n_boot=NB)
                    r |= {"arc 2 s adapted - shipped": p["mean"], "lo": p["lo"], "hi": p["hi"]}
                d4.append(r)
        # ---- D5
        mv, m512 = v0 >= 2, (v0 >= 5) & (v0 < 12)
        def meas(arc):
            return [float(q_r0(arc, v0)[mv].mean()), float(q_r0(arc, v0)[m512].mean()) if m512.sum() > 20 else np.nan, float(arc[mv, 3].sum() / A["shipped"][mv, 3].sum()) - 1,
                    float(arc[mv, 4].sum() / A["shipped"][mv, 4].sum()) - 1, float(arc[ss, 3].mean()) if ss.sum() > 8 else np.nan, float(q_acc(arc, v0)[mv].mean())]
        names = ["r0 - 1 (v0 >= 2)", "r0 - 1 (5-12 m/s)", "arc 2 s / shipped - 1 (v0 >= 2)", "arc 4 s / shipped - 1 (v0 >= 2)", "standstill arc 2 s (m)", "plan a 1.5-2 s (v0 >= 2)"]
        mS = meas(A["shipped"])
        d5.append({"domain": dom, "arm": "shipped", "variant": "-"} | dict(zip(names, mS)))
        for fm, ks in F.items():
            mA = np.mean([meas(A[k]) for k in ks], 0)
            d5.append({"domain": dom, "arm": fm, "variant": "main"} | dict(zip(names, mA)))
            for var in VARIANTS[1:]:
                kv = [f"{k}:{var}" for k in ks if f"{k}:{var}" in A]
                if kv:
                    mV = np.mean([meas(A[k]) for k in kv], 0)
                    r = {"domain": dom, "arm": fm, "variant": var} | dict(zip(names, mV))
                    for nm, x, y0, y1 in zip(names, mV, mS, mA):
                        r[f"share of (main - shipped) kept: {nm}"] = float((x - y0) / (y1 - y0)) if abs(y1 - y0) > 1e-6 else np.nan
                    d5.append(r)
    for nm, rows in (("check_units", chk), ("d1_first_segment", d1), ("d2_arc_ratio", d2), ("d3_lead_share", d3), ("d3_lead_cells", d3c), ("d4_standstill", d4), ("d5_switches", d5)):
        if rows:
            stats.write_table(rows, OUT / nm)
    figs(tabs)


def figs(tabs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FD = _R / "experiments/op_parity/figs/diag1"
    FD.mkdir(parents=True, exist_ok=True)
    col = {"shipped": "#444444", "P2H10": "#1f77b4", "SH30": "#d62728", "AP2H10": "#2ca02c", "APY10m10": "#9467bd", "log": "#999999"}
    # fig 1: first-segment speed over ego speed by speed bin, per domain
    fig, axs = plt.subplots(1, len(tabs), figsize=(3.6 * len(tabs), 3.4), sharey=True, squeeze=False)
    for ax, t in zip(axs[0], tabs):
        v0 = t["v0"].astype(np.float64)
        for arm, ks in [("shipped", ["shipped"])] + list(fams(t).items()):
            ys = []
            for nm, lo, hi in BINS[1:]:
                m = (v0 >= lo) & (v0 < hi)
                ys.append(np.mean([q_r0(t["arc"][k], v0)[m].mean() for k in ks]) * 100 if m.sum() >= 20 else np.nan)
            ax.plot(range(4), ys, "o-", color=col.get(arm, "k"), label=arm)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(range(4), [b[0] for b in BINS[1:]])
        ax.set_title(t["dom"], fontsize=8)
        ax.set_xlabel("ego speed bin (m/s)")
        ax.legend(fontsize=7)
    axs[0][0].set_ylabel("plan speed over first 0.5 s vs ego speed (%)")
    fig.tight_layout()
    fig.savefig(FD / "first_segment.png", dpi=130)
    plt.close(fig)
    # fig 2: plan acceleration vs required deceleration on closing-lead frames
    fig, axs = plt.subplots(1, len(tabs), figsize=(3.6 * len(tabs), 3.4), sharey=True, squeeze=False)
    for ax, t in zip(axs[0], tabs):
        v0 = t["v0"].astype(np.float64)
        d = t["lead_x"].astype(np.float64) - t["c2f"]
        c = v0 - t["lead_v"].astype(np.float64)
        cl = (t["lp"] > 0.5) & (v0 >= 2) & (c >= 1) & (d / np.maximum(c, 0.1) <= 8)
        need = np.clip(c ** 2 / (2 * np.maximum(d - 2, 0.5)), 0, 8)
        edges = [0, 0.5, 1, 2, 4, 8.01]
        for arm, ks in [("shipped", ["shipped"])] + list(fams(t).items()) + [("log", None)]:
            y = q_acc(t["log_arc"].astype(np.float64), v0) if ks is None else np.mean([q_acc(t["arc"][k], v0) for k in ks], 0)
            ys = [np.nanmean(y[cl & (need >= lo) & (need < hi)]) if (cl & (need >= lo) & (need < hi)).sum() >= 8 else np.nan for lo, hi in zip(edges[:-1], edges[1:])]
            ax.plot(range(5), ys, "o-", color=col.get(arm, "k"), label=arm, ls="--" if ks is None else "-")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(range(5), ["<0.5", "0.5-1", "1-2", "2-4", ">4"])
        ax.set_title(f"{t['dom']} (n {int(cl.sum())})", fontsize=8)
        ax.set_xlabel("required deceleration from the lead output (m/s^2)")
        ax.legend(fontsize=7)
    axs[0][0].set_ylabel("plan acceleration to 1.5-2 s (m/s^2)")
    fig.tight_layout()
    fig.savefig(FD / "lead_response.png", dpi=130)
    plt.close(fig)
    # fig 3: standstill displacement at 2 s, stopped lead vs no lead
    fig, axs = plt.subplots(1, len(tabs), figsize=(3.6 * len(tabs), 3.4), sharey=True, squeeze=False)
    for ax, t in zip(axs[0], tabs):
        v0 = t["v0"].astype(np.float64)
        d = t["lead_x"].astype(np.float64) - t["c2f"]
        ss = v0 < 0.5
        ms = [ss & (t["lp"] > 0.5) & (d < 15) & (t["lead_v"] < 1), ss & (t["lp"] <= 0.5)]
        arms = [("shipped", ["shipped"])] + list(fams(t).items()) + [("log", None)]
        for i, (arm, ks) in enumerate(arms):
            ar = t["log_arc"].astype(np.float64) if ks is None else np.mean([t["arc"][k] for k in ks], 0)
            ax.bar(np.arange(2) + (i - len(arms) / 2 + 0.5) * 0.8 / len(arms), [np.nanmean(ar[m, 3]) if m.sum() >= 8 else 0 for m in ms], 0.8 / len(arms), color=col.get(arm, "k"), label=arm)
        ax.set_xticks(range(2), [f"stopped lead\n(n {int(ms[0].sum())})", f"no lead\n(n {int(ms[1].sum())})"])
        ax.set_title(t["dom"], fontsize=8)
        ax.legend(fontsize=7)
    axs[0][0].set_ylabel("plan arc at 2 s from standstill (m)")
    fig.tight_layout()
    fig.savefig(FD / "standstill.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- D6 swaps
def retime_pose(poses, v0, tau=TAU, dt=0.05):
    """experiments/alpasim/lib/col1_pai_driver.py::retime, verbatim form ((8, 3) poses at 0.5 .. 4 s)."""
    p = np.concatenate([np.zeros((1, 3)), np.asarray(poses, np.float64)])
    ds = np.hypot(*np.diff(p[:, :2], axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(np.maximum(ds, 1e-6))])
    t = np.arange(dt / 2, 4.0, dt)
    vp = (ds / 0.5)[np.minimum((t / 0.5).astype(int), 7)]
    v = v0 + (vp - v0) * np.minimum(t / max(tau, 1e-3), 1.0)
    sn = np.cumsum(np.maximum(v, 0.0) * dt)[int(round(0.5 / dt)) - 1::int(round(0.5 / dt))]
    far = p[-1] + np.array([60.0 * np.cos(p[-1, 2]), 60.0 * np.sin(p[-1, 2]), 0.0])
    P, S = np.concatenate([p, far[None]]), np.concatenate([s, [s[-1] + 60.0]])
    return np.stack([np.interp(sn, S, P[:, 0]), np.interp(sn, S, P[:, 1]), np.interp(sn, S, np.unwrap(P[:, 2]))], 1)


def cmd_nav_swap(a):
    import pt_swap as PS
    z = np.load(WD / "nav.npz")
    n = len(z["tokens"])
    S = z["pose_P0"].astype(np.float64)
    CS = [PS.Curve(q) for q in S]
    K = {"S": S}
    err = 0.0
    for i, tag in enumerate(NAV_TAGS):
        Apose = z[f"pose_{tag}_main"].astype(np.float64)
        CA = [PS.Curve(q) for q in Apose]
        K[f"A{i}"] = Apose
        K[f"AS{i}"] = np.stack([ca.at(cs.sv[1:]) for ca, cs in zip(CA, CS)])          # adapted path, shipped speed profile
        K[f"SA{i}"] = np.stack([cs.at(ca.sv[1:]) for ca, cs in zip(CA, CS)])          # shipped path, adapted speed profile
        err = max(err, float(max(np.abs(c.at(c.sv[1:])[:, :2] - q[:, :2]).max() for c, q in zip(CA, Apose))))
        if i in (0, 2):
            K[f"Aa{i}"] = np.stack([retime_pose(q, v) for q, v in zip(Apose, z["speed"])])   # job 1 (a) on the nuPlan board
    K["Sa"] = np.stack([retime_pose(q, v) for q, v in zip(S, z["speed"])])
    print("curve identity max err (m)", err, "keys", list(K), flush=True)
    assert err < 0.15                                        # vertices closer than pt_swap.EPS_V (0.05 m) are merged: near-standstill plans only
    np.savez(WD / "swap_poses.npz", tokens=z["tokens"], **{k: v.astype(np.float32) for k, v in K.items()})

def _swap_rows(dom, sc, fam_idx, unit, strata, cm, ci, scale=1.0):
    """sc: {key: (N,) per-frame score}; fam_idx {family: [seed index]}; cm(x, mask) -> board mean; ci(d, mask) -> (point, lo, hi).
    -> rows (arms, contrasts, shares, composites) per stratum."""
    rows = []
    for fm, idx in fam_idx.items():
        Sx = sc["S"]
        Ax, AS, SA = (np.mean([sc[f"{k}{i}"] for i in idx], 0) for k in ("A", "AS", "SA"))
        for nm, m in strata.items():
            if m.sum() < 8:
                continue
            r = {"domain": dom, "arm": fm, "stratum": nm, "n": int(m.sum()), "S": cm(Sx, m) * scale, "A": cm(Ax, m) * scale, "AS (A path, S speed)": cm(AS, m) * scale,
                 "SA (S path, A speed)": cm(SA, m) * scale}
            for lab, d in (("A - S", Ax - Sx), ("AS - S", AS - Sx), ("SA - S", SA - Sx), ("AS - A", AS - Ax), ("SA - A", SA - Ax)):
                p, lo, hi = ci(d, m)
                r[lab], r[lab + " lo"], r[lab + " hi"] = p * scale, lo * scale, hi * scale
            g = r["A - S"]
            r["longitudinal share (SA - S) / (A - S)"] = r["SA - S"] / g if abs(g) > 1e-9 else np.nan
            r["path share (AS - S) / (A - S)"] = r["AS - S"] / g if abs(g) > 1e-9 else np.nan
            rows.append(r)
    return rows


def _composites(dom, sc, fam_idx, masks, cm, ci, scale=1.0):
    """Per-frame switches between A and AS (base speed profile on the frames of `mask`, the adapter's elsewhere), against A and S."""
    rows = []
    allm = np.ones(len(sc["S"]), bool)
    for fm, idx in fam_idx.items():
        Ax, AS = (np.mean([sc[f"{k}{i}"] for i in idx], 0) for k in ("A", "AS"))
        cand = {nm: np.where(m, AS, Ax) for nm, m in masks.items()}
        cand["oracle max(A, AS) per frame (privileged upper bound)"] = np.mean([np.maximum(sc[f"A{i}"], sc[f"AS{i}"]) for i in idx], 0)
        for nm, x in cand.items():
            pa, pl = ci(x - Ax, allm), ci(x - sc["S"], allm)
            rows.append({"domain": dom, "arm": fm, "policy": nm, "frames on base speed": float(masks[nm].mean()) if nm in masks else np.nan, "score": cm(x, allm) * scale,
                         "- A": pa[0] * scale, "lo": pa[1] * scale, "hi": pa[2] * scale, "- S": pl[0] * scale, "S lo": pl[1] * scale, "S hi": pl[2] * scale})
    return rows


def cmd_swap_report(a):
    import pandas as pd
    from jevdrive import stats
    OUT.mkdir(parents=True, exist_ok=True)
    rows, comp, extra, gate = [], [], [], {}
    # ---- navtest (score-poses, no-EC EPDMS, non-reactive)
    f = WD / "swap_score.csv"
    if f.exists():
        z, t = np.load(WD / "nav.npz"), load_tab(WD / "tab_nav.npz")
        tok, logs, v0 = z["tokens"].astype(str), z["log"].astype(str), z["speed"].astype(np.float64)
        df = pd.read_csv(f)
        df = df.rename(columns={"no_at_fault_collisions": "NC", "drivable_area_compliance": "DAC", "driving_direction_compliance": "DDC", "traffic_light_compliance": "TLC",
                                "ego_progress": "EP", "time_to_collision_within_bound": "TTC", "lane_keeping": "LK", "history_comfort": "HC"})
        sub = [c for c in ("NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC") if c in df]
        P = {k: g.set_index("token").reindex(tok) for k, g in df.groupby("key")}
        sc = {k: g["score"].to_numpy(float) for k, g in P.items()}
        cm = lambda x, m: float(np.nanmean(x[m]))  # noqa: E731
        def ci(d, m):
            r = stats.bootstrap(d[m], groups=logs[m], n_boot=4000)
            return r["mean"], r["lo"], r["hi"]
        try:                                                           # identity gate: unswapped keys vs the stored bench sub-scores
            from mixed_domain import nav_units
            for i, tag in enumerate(NAV_TAGS):
                u = nav_units(tag).reindex(tok)
                gate[tag] = {c: float(np.nanmax(np.abs(u[c].to_numpy(float) - P[f"A{i}"][c].to_numpy(float)))) for c in sub if c in u}
                gate[tag]["tokens with any sub-score differing"] = int((np.abs(u[[c for c in sub if c in u]].to_numpy(float) - P[f"A{i}"][[c for c in sub if c in u]].to_numpy(float)) > 1e-6).any(1).sum())
        except Exception as e:                                         # noqa: BLE001
            gate["navtest_gate_error"] = repr(e)
        lead = t["lp"] > 0.5
        d = t["lead_x"] - t["c2f"]
        c = v0 - t["lead_v"]
        closing = lead & (v0 >= 2) & (c >= 1) & (d / np.maximum(c, 0.1) <= 8)
        st = {"all": np.ones(len(tok), bool), "standstill (v0 < 0.5)": v0 < 0.5, "moving (v0 >= 0.5)": v0 >= 0.5, "lead present": lead, "no lead": ~lead, "closing lead": closing,
              "0.5-5 m/s": (v0 >= 0.5) & (v0 < 5), "5-12 m/s": (v0 >= 5) & (v0 < 12), ">= 12 m/s": v0 >= 12}
        fam = {"P2H10": [0, 1], "SH30": [2, 3]}
        rows += _swap_rows("navtest (no-EC EPDMS)", sc, fam, logs, st, cm, ci, 100.0)
        masks = {"AS everywhere (adapter path, base speed profile)": st["all"], "base speed on lead frames": lead, "base speed on closing-lead frames": closing,
                 "base speed when moving (v0 >= 0.5), adapter launch at standstill": v0 >= 0.5, "base speed on no-lead frames": ~lead}
        comp += _composites("navtest (no-EC EPDMS)", sc, fam, masks, cm, ci, 100.0)
        for fm, idx in fam.items():                                    # where the swaps act: sub-scores
            for key in ("S", "A", "AS", "SA"):
                ks = ["S"] if key == "S" else [f"{key}{i}" for i in idx]
                extra.append({"domain": "navtest", "arm": fm, "cell": key, "no-EC EPDMS": 100 * float(np.mean([np.nanmean(sc[k]) for k in ks]))}
                             | {c2: 100 * float(np.mean([np.nanmean(P[k][c2].to_numpy(float)) for k in ks])) for c2 in sub})
        ret = []
        for key, base, nm in (("Aa0", "A0", "P2H10-F-s0"), ("Aa2", "A2", "SH30-F-s0"), ("Sa", "S", "shipped")):
            if key in sc:
                for snm, m in st.items():
                    p, lo, hi = ci(sc[key] - sc[base], m)
                    ret.append({"arm": nm, "stratum": snm, "n": int(m.sum()), "no-EC EPDMS": 100 * cm(sc[base], m), "re-timed (a)": 100 * cm(sc[key], m), "d": 100 * p, "lo": 100 * lo, "hi": 100 * hi}
                               | ({c2 + " d": 100 * float(np.nanmean((P[key][c2].to_numpy(float) - P[base][c2].to_numpy(float))[m])) for c2 in sub} if snm == "all" else {}))
        stats.write_table(ret, OUT / "nav_retime_a")
    # ---- WOD (RFS)
    try:
        from pp_wod_diag import retime
        C = wod_ctx()
        n = C.n
        Sx = C.preds("shipped", True)
        sc = {"S": C.rfs(Sx)}
        tags = ["P2H10-F-s0", "P2H10-F-s1", "SH30-F-s0", "SH30-F-s1"]
        for i, tag in enumerate(tags):
            Ax = C.preds(tag, True)
            sc[f"A{i}"], sc[f"AS{i}"], sc[f"SA{i}"] = C.rfs(Ax), C.rfs(retime(Ax[:n], Sx[:n])), C.rfs(retime(Sx[:n], Ax[:n]))
        gate["wod_retime_self_max_m"] = float(np.abs(retime(Sx[:n], Sx[:n]) - Sx[:n]).max())
        lead = C.lp[:n, 0] > 0.5
        v0 = C.v0
        d = C.lead[:n, 0, 0, 0] - (WOD_FRONT - C.dev[:n, 0])
        c = C.vfed[:n] - C.lead[:n, 0, 0, 2]
        closing = lead & (C.vfed[:n] >= 2) & (c >= 1) & (d / np.maximum(c, 0.1) <= 8)
        st = {"all": np.ones(n, bool), "standstill (v0 < 0.5)": v0 < 0.5, "moving (v0 >= 0.5)": v0 >= 0.5, "lead present": lead, "no lead": ~lead, "closing lead": closing,
              "0.5-5 m/s": (v0 >= 0.5) & (v0 < 5), "5-12 m/s": (v0 >= 5) & (v0 < 12), ">= 12 m/s": v0 >= 12}
        fam = {"P2H10": [0, 1], "SH30": [2, 3]}
        rows += _swap_rows("WOD val (RFS)", sc, fam, C.seq[:n], st, C.cm, C.ci)
        masks = {"AS everywhere (adapter path, base speed profile)": st["all"], "base speed on lead frames": lead, "base speed on closing-lead frames": closing,
                 "base speed when moving (v0 >= 0.5), adapter launch at standstill": v0 >= 0.5, "base speed on no-lead frames": ~lead}
        comp += _composites("WOD val (RFS)", sc, fam, masks, C.cm, C.ci)
        gate["wod_d162_reference"] = "P2H10: AS - S +0.062, SA - S -0.464 (decision 162)"
    except ImportError as e:
        gate["wod_skipped"] = repr(e)
    if rows:
        stats.write_table(rows, OUT / "swap_shares")
        stats.write_table(comp, OUT / "swap_composites")
    if extra:
        stats.write_table(extra, OUT / "swap_nav_subscores")
    (OUT / "swap_gate.json").write_text(json.dumps(gate, indent=1))
    print(json.dumps(gate, indent=1))

def cmd_nav_acc(a):
    """Post-hoc input probe (not registered): the fed longitudinal acceleration on the AlpaSim PAI track has 5.7x the navtest spread
    (ax / 3 std 1.44 vs 0.25). navtest tokens with the acceleration input replaced: acc0 (0), accx{k} (ax scaled by k), accn (ax + N(0, 4.3 m/s^2)),
    accp2 / accm2 (ax + / - 2 m/s^2). Adapted arms only (shipped reads no ego input). -> diag1/nav_acc.npz"""
    import torch
    from jevdrive import op_interp as I
    from jevdrive.bench import navsim as N
    from jevdrive.bench.models import resolve
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("op_parity", "diag1/nav-acc", seed=0, config=vars(a)) as run:
        run.use_split(splits.load("navsim/navtest"))
        N._pp_path()
        import pp_train as T
        dev = torch.device("cuda")
        S = T.Store(["lb_navtest"], dev, need_side=False, frames="warp")
        tb, n = S.tb, S.n
        g = torch.Generator(device="cpu").manual_seed(0)
        noise = (torch.randn(n, generator=g) * 4.3 / 3.0).to(S.ego)
        def ed(f):
            e = S.ego.clone()
            e[:, 6] = f(e[:, 6])
            return e
        V = {"acc0": ed(lambda x: x * 0), "accx3": ed(lambda x: x * 3), "accx57": ed(lambda x: x * 5.7), "accn": ed(lambda x: x + noise), "accp2": ed(lambda x: x + 2 / 3.0),
             "accm2": ed(lambda x: x - 2 / 3.0)}
        R = dict(tokens=tb["names"], noise=noise.cpu().numpy() * 3.0)
        for tag in ("P2H10-F-s0", "SH30-F-s0"):
            model = N._load_ckpt(T, resolve(f"{tag}@warp", check=True).ckpt, dev)
            sl = model.net.slices
            pi = np.arange(sl["plan"].start, sl["plan"].start + 495)
            for var, ego in V.items():
                mu = np.zeros((n, 33, 15), np.float32)
                with torch.no_grad():
                    for i in range(0, n, 128):
                        r = torch.arange(i, min(i + 128, n), device=dev)
                        mu[i:i + len(r)] = model(S.front[r], ego[r], S.tc[r], None, None).float().cpu().numpy()[:, pi].reshape(-1, 33, 15)
                R[f"pose_{tag}_{var}"] = np.stack([I.to_rear(mu[i, :, 0:3], mu[i, :, 11], I.T_IDXS, tb["cam"][i, :2], T8, "lever") for i in range(n)])
                run.info("%s %s", tag, var)
            del model
        np.savez(WD / "nav_acc.npz", **R)


def cmd_acc_report(a):
    """nav_acc.npz + the PAI table: first-segment ratio and arcs under the acceleration interventions; on PAI rows the adapted - shipped
    first-segment gap against the fed acceleration."""
    from jevdrive import stats
    z, zz = np.load(WD / "nav.npz"), np.load(WD / "nav_acc.npz")
    v0, logs = z["speed"].astype(np.float64), z["log"].astype(str)
    aS = arc_at(z["pose_P0"][..., :2].astype(np.float64), 0.5)
    rows = []
    for tag in ("P2H10-F-s0", "SH30-F-s0"):
        base = arc_at(z[f"pose_{tag}_main"][..., :2].astype(np.float64), 0.5)
        for var in ("main", "acc0", "accx3", "accx57", "accn", "accp2", "accm2"):
            ar = base if var == "main" else arc_at(zz[f"pose_{tag}_{var}"][..., :2].astype(np.float64), 0.5)
            for nm, lo, hi in BINS[1:] + [(">=2", 2, 1e9)]:
                m = (v0 >= lo) & (v0 < hi)
                if m.sum() < 20:
                    continue
                p = stats.paired(q_r0(ar, v0)[m], q_r0(aS, v0)[m], groups=logs[m], n_boot=NB)
                rows.append({"arm": tag, "acceleration input": var, "v0 bin": nm, "n": int(m.sum()), "mean r0 - 1": float(q_r0(ar, v0)[m].mean()), "r0 adapted - shipped": p["mean"], "lo": p["lo"],
                             "hi": p["hi"], "arc 2 s / shipped": float(ar[m, 3].sum() / aS[m, 3].sum()), "arc 4 s / shipped": float(ar[m, 4].sum() / aS[m, 4].sum()),
                             "share r0 > 1.05": float((q_r0(ar, v0)[m] > 0.05).mean())})
            if var == "accn":                                          # dose: first-segment response per m/s^2 of injected acceleration error
                mv = v0 >= 2
                sl = np.polyfit(zz["noise"][mv], ((ar[:, 0] - base[:, 0]) / 0.5)[mv], 1)[0]
                rows.append({"arm": tag, "acceleration input": "slope: first-segment speed (m/s) per m/s^2 of injected ax", "v0 bin": ">=2", "n": int(mv.sum()), "mean r0 - 1": float(sl)})
    stats.write_table(rows, OUT / "acc_probe_nav")
    # lead response with the acceleration input removed: does the adapted plan slow for a closing lead by itself, or by reading the ego's own deceleration?
    lr = []
    t = load_tab(WD / "tab_nav.npz")
    d, c = t["lead_x"] - t["c2f"], v0 - t["lead_v"]
    cl = (t["lp"] > 0.5) & (v0 >= 2) & (c >= 1) & (d / np.maximum(c, 0.1) <= 8)
    need = np.clip(c ** 2 / (2 * np.maximum(d - 2, 0.5)), 0, 8)
    axn = z["ego"][:, 6].astype(np.float64) * 3.0
    sl = lambda y, m: float(np.polyfit(need[m], y[m], 1)[0])  # noqa: E731
    for nm, m in (("closing lead, all", cl), ("closing lead, time to contact < 4 s", cl & (d / np.maximum(c, 0.1) < 4)), ("closing lead, fed ax >= -0.3", cl & (axn >= -0.3)),
                  ("closing lead, fed ax < -0.3", cl & (axn < -0.3)), ("not closing, v0 >= 2", ~cl & (v0 >= 2))):
        r = {"domain": "nav", "frames": nm, "n": int(m.sum()), "mean fed ax": float(axn[m].mean()), "share fed ax < -0.3": float((axn[m] < -0.3).mean()), "a shipped": float(q_acc(aS, v0)[m].mean()),
             "a log": float(q_acc(t["log_arc"].astype(np.float64), v0)[m].mean()), "slope shipped": sl(q_acc(aS, v0), m), "shipped slows": float((q_acc(aS, v0)[m] <= -0.3).mean())}
        for tag in ("P2H10-F-s0", "SH30-F-s0"):
            for var in ("main", "acc0"):
                ar = arc_at((z[f"pose_{tag}_main"] if var == "main" else zz[f"pose_{tag}_{var}"])[..., :2].astype(np.float64), 0.5)
                r |= {f"a {tag} {var}": float(q_acc(ar, v0)[m].mean()), f"slope {tag} {var}": sl(q_acc(ar, v0), m), f"slows {tag} {var}": float((q_acc(ar, v0)[m] <= -0.3).mean())}
        lr.append(r)
    for g in sorted(WD.glob("tab_alp_*.npz")):
        t = load_tab(g)
        if "ego" not in t:
            continue
        v = t["v0"].astype(np.float64)
        d, c = t["lead_x"] - t["c2f"], v - t["lead_v"]
        cl = (t["lp"] > 0.5) & (v >= 2) & (c >= 1) & (d / np.maximum(c, 0.1) <= 8)
        ax = t["ego"][:, 6].astype(np.float64) * 3.0
        need = np.clip(c ** 2 / (2 * np.maximum(d - 2, 0.5)), 0, 8)
        fam = [k for k in t["arc"] if k != "shipped" and ":" not in k][0]
        for nm, m in (("closing lead, all", cl), ("closing lead, fed ax >= -0.3", cl & (ax >= -0.3)), ("closing lead, fed ax < -0.3", cl & (ax < -0.3)), ("not closing, v0 >= 2", ~cl & (v >= 2))):
            if m.sum() >= 10:
                aS2, aA2 = q_acc(t["arc"]["shipped"], v), q_acc(t["arc"][fam], v)
                lr.append({"domain": t["dom"], "frames": nm, "n": int(m.sum()), "mean fed ax": float(ax[m].mean()), "share fed ax < -0.3": float((ax[m] < -0.3).mean()), "a shipped": float(aS2[m].mean()),
                           "slope shipped": float(np.polyfit(need[m], aS2[m], 1)[0]), "shipped slows": float((aS2[m] <= -0.3).mean()),
                           f"a {fam} main": float(aA2[m].mean()), f"slope {fam} main": float(np.polyfit(need[m], aA2[m], 1)[0]), f"slows {fam} main": float((aA2[m] <= -0.3).mean())})
    stats.write_table(lr, OUT / "acc_probe_lead")
    if (WD / "acc_score.csv").exists():                               # what the acceleration input is worth on the nuPlan board (score-poses, no-EC EPDMS)
        import pandas as pd
        df = pd.read_csv(WD / "acc_score.csv").rename(columns={"no_at_fault_collisions": "NC", "ego_progress": "EP", "time_to_collision_within_bound": "TTC", "drivable_area_compliance": "DAC"})
        tok = z["tokens"].astype(str)
        P = {k: g.set_index("token").reindex(tok) for k, g in df.groupby("key")}
        sr = []
        for i, tag in ((0, "P2H10-F-s0"), (2, "SH30-F-s0")):
            for var in ("acc0", "accx3", "accn"):
                dd = P[f"{var}_{i}"]["score"].to_numpy(float) - P[f"A{i}"]["score"].to_numpy(float)
                for snm, m in (("all", np.ones(len(tok), bool)), ("standstill (v0 < 0.5)", v0 < 0.5), ("moving (v0 >= 0.5)", v0 >= 0.5)):
                    r = stats.bootstrap(dd[m], groups=logs[m], n_boot=4000)
                    sr.append({"arm": tag, "acceleration input": var, "stratum": snm, "n": int(m.sum()), "d no-EC EPDMS": 100 * r["mean"], "lo": 100 * r["lo"], "hi": 100 * r["hi"]}
                              | {c2 + " d": 100 * float((P[f"{var}_{i}"][c2].to_numpy(float) - P[f"A{i}"][c2].to_numpy(float))[m].mean()) for c2 in ("NC", "TTC", "EP", "DAC")})
        stats.write_table(sr, OUT / "acc_probe_nav_score")
    f = WD / "alp" / "pai.npz"
    if f.exists():
        p = np.load(f, allow_pickle=True)
        v = p["v0"].astype(np.float64)
        ax = p["ego"][:, 6].astype(np.float64) * 3.0
        rf, rp = q_r0(arc_at(p["plan_ft"][..., :2].astype(np.float64), 0.5), v), q_r0(arc_at(p["plan_p0"][..., :2].astype(np.float64), 0.5), v)
        unit = np.char.add(p["group"].astype(str), p["scene"].astype(str))
        out = []
        for g in np.unique(p["group"].astype(str)):
            gm = (p["group"].astype(str) == g) & (v >= 2) & (p["n_slots"] >= 8 if "n_slots" in p.files else True)
            out.append({"group": g, "cell": "all v0 >= 2", "n": int(gm.sum()), "ax mean": float(ax[gm].mean()), "ax std": float(ax[gm].std()), "r0 - 1 adapted": float(rf[gm].mean()), "r0 - 1 shipped": float(rp[gm].mean()),
                        "corr(ax, adapted - shipped r0)": float(np.corrcoef(ax[gm], (rf - rp)[gm])[0, 1]),
                        "slope first-seg speed per m/s^2 ax": float(np.polyfit(ax[gm], ((rf - rp) * v)[gm], 1)[0])})
            for nm, lo, hi in (("ax < -2", -1e9, -2), ("-2..-0.5", -2, -0.5), ("-0.5..0.5", -0.5, 0.5), ("0.5..2", 0.5, 2), ("ax > 2", 2, 1e9)):
                m = gm & (ax >= lo) & (ax < hi)
                if m.sum() >= 10:
                    b = stats.paired(rf[m], rp[m], groups=unit[m], n_boot=NB)
                    out.append({"group": g, "cell": nm, "n": int(m.sum()), "ax mean": float(ax[m].mean()), "r0 - 1 adapted": float(rf[m].mean()), "r0 - 1 shipped": float(rp[m].mean()),
                                "adapted - shipped": b["mean"], "lo": b["lo"], "hi": b["hi"], "mean v0": float(v[m].mean())})
        stats.write_table(out, OUT / "acc_probe_pai")

def _cap_one(x):
    import serve_fix as F
    poses, lead, lp, pv, v0, a0, c2f = x
    out = []
    for vc in (0.0, TAU):
        sv = F.Serve(vc, True, c2f)
        out.append(sv({"poses": poses, "lead": lead, "lead_prob": lp, "mu": np.array([[0, 0, 0, pv]])}, v0, a0, 0)[0])
    return out


def cmd_nav_cap(a):
    """FIX1's switches (serve_fix.Serve, as committed; first decision of a session) on the navtest plans of P2H10-F-s0 / SH30-F-s0 with the
    checkpoint's own lead outputs -> diag1/cap_poses.npz keys Ab<i> (JEV_LEAD), Aab<i> (JEV_VCONT=1 + JEV_LEAD) for score-poses. Post hoc: what
    the lead limit costs on the nuPlan board."""
    from concurrent.futures import ProcessPoolExecutor
    _sys.path.insert(0, str(_R / "experiments/alpasim/lib"))
    z = np.load(WD / "nav.npz")
    K = {}
    for i, tag in ((0, "P2H10-F-s0"), (2, "SH30-F-s0")):
        P = z[f"pose_{tag}_main"].astype(np.float64)
        args = [(P[j], z[f"lead_{tag}"][j], z[f"lp_{tag}"][j], float(z[f"pvel0_{tag}"][j]), float(z["speed"][j]), float(z["acc"][j, 0]), float(NAV_FRONT - z["cam"][j, 0])) for j in range(len(P))]
        with ProcessPoolExecutor(12) as ex:
            r = list(ex.map(_cap_one, args, chunksize=64))
        K[f"Ab{i}"], K[f"Aab{i}"] = np.stack([x[0] for x in r]).astype(np.float32), np.stack([x[1] for x in r]).astype(np.float32)
        K[f"A{i}"] = P.astype(np.float32)
        print(tag, "mean arc change at 4 s (m):", float((arc_at(K[f"Ab{i}"][..., :2].astype(np.float64), 0.5)[:, 4] - arc_at(P[..., :2], 0.5)[:, 4]).mean()), flush=True)
    np.savez(WD / "cap_poses.npz", tokens=z["tokens"], **K)


def cmd_cap_report(a):
    import pandas as pd
    from jevdrive import stats
    z, t = np.load(WD / "nav.npz"), load_tab(WD / "tab_nav.npz")
    tok, logs, v0 = z["tokens"].astype(str), z["log"].astype(str), z["speed"].astype(np.float64)
    df = pd.read_csv(WD / "cap_score.csv").rename(columns={"no_at_fault_collisions": "NC", "ego_progress": "EP", "time_to_collision_within_bound": "TTC", "drivable_area_compliance": "DAC"})
    P = {k: g.set_index("token").reindex(tok) for k, g in df.groupby("key")}
    lead = t["lp"] > 0.5
    st = {"all": np.ones(len(tok), bool), "standstill (v0 < 0.5)": v0 < 0.5, "moving (v0 >= 0.5)": v0 >= 0.5, "lead present": lead, "no lead": ~lead, "standstill, lead": (v0 < 0.5) & lead}
    rows = []
    for i, tag in ((0, "P2H10-F-s0"), (2, "SH30-F-s0")):
        for key, nm in ((f"Ab{i}", "(b)"), (f"Aab{i}", "(a) + (b)")):
            dd = P[key]["score"].to_numpy(float) - P[f"A{i}"]["score"].to_numpy(float)
            for snm, m in st.items():
                r = stats.bootstrap(dd[m], groups=logs[m], n_boot=4000)
                rows.append({"arm": tag, "fix": nm, "stratum": snm, "n": int(m.sum()), "no-EC EPDMS": 100 * float(P[f"A{i}"]["score"].to_numpy(float)[m].mean()), "d": 100 * r["mean"], "lo": 100 * r["lo"], "hi": 100 * r["hi"]}
                            | {c + " d": 100 * float((P[key][c].to_numpy(float) - P[f"A{i}"][c].to_numpy(float))[m].mean()) for c in ("NC", "TTC", "EP", "DAC")}
                            | {"share of tokens changed": float((np.abs(dd[m]) > 1e-9).mean())})
    stats.write_table(rows, OUT / "nav_fix_b")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    w = sp.add_parser("wod-fix")
    w.add_argument("--lead", action="store_true", help="also (b) and (a) + (b): needs lane FIX1's committed serve_fix.py")
    w.add_argument("--suffix", default="")
    x = sp.add_parser("nav-extract")
    x.add_argument("--limit", type=int, default=0)
    for c in ("wod-tab", "nav-tab", "alp-tab", "tables", "nav-swap", "swap-report", "nav-acc", "acc-report", "nav-cap", "cap-report"):
        sp.add_parser(c)
    a = ap.parse_args()
    {"wod-fix": cmd_wod_fix, "nav-extract": cmd_nav_extract, "wod-tab": cmd_wod_tab, "nav-tab": cmd_nav_tab, "alp-tab": cmd_alp_tab, "tables": cmd_tables,
     "nav-swap": cmd_nav_swap, "swap-report": lambda a: cmd_swap_report(a), "nav-acc": cmd_nav_acc, "acc-report": cmd_acc_report, "nav-cap": cmd_nav_cap, "cap-report": cmd_cap_report}[a.cmd](a)


if __name__ == "__main__":
    main()
