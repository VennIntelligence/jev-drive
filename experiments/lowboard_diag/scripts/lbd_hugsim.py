"""LOWDIAG, HUGSIM half: every failed SH30 run under the shared taxonomy of plans/2026-10-10-lowdiag-prereg.md (sections 1-6, 8).

  extract  (box, .venv)  per-run truth quantities, raw flags, exclusive class, in-plan / lead time, base-right columns
                         -> results/hugsim/units.csv (one row per run of the unit arms; reference arms joined as columns)
  report   (box or Mac)  board table, raw-flag counts, in-plan / lead time, base-right shares -> results/hugsim/*.csv, tables.md
  bev      (box)         one BEV (+ video frames) per run with every flag in the title -> figs/hugsim/<name>.png

Units: SH30-F-s0 `spec_plan_smooth` all64 (scope cut on 2026-10-10: one seed; pass --arms to add s1). Geometry, route, ground /
background point sets, plan frame and the struck-actor rule are fd_hugsim's (imported, not rewritten). State k = 0..n, E = state n
(the frame after the last action), DT 0.25 s. Implementation choices that the pre-registration leaves open are the constants and
comments below; they are listed in results/hugsim.md under "Deviations and choices".
"""
import argparse
import json
import os
import sys
from itertools import combinations
from math import factorial
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "experiments/op_parity/scripts"), str(REPO / "experiments/alpasim/scripts")]
import fd_hugsim as FD  # noqa: E402
from fd_hugsim import DT, EGO_L, EGO_W, box_poly, wrap  # noqa: E402

OUT = REPO / "experiments/lowboard_diag/results/hugsim"
FIGS = REPO / "experiments/lowboard_diag/figs/hugsim"
D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
ARMS = ["SH30-F-s0"]
PRESET = "spec_plan_smooth"
REFS = [("WA-JEPA", "exam"), ("P0", "spec"), ("P0", "spec_plan_smooth")]   # reference arms: light records only
KIND = {"fg_collision": "agent", "bg_collision": "boundary", "off_route": "corridor", "max_steps": "none"}
LAT_ON, BRAKE, T_CF, T_BAND, BAND_PAD = 1.5, 4.0, 3.0, 2.0, 0.5      # registered: onset 1.5 m, 4 m/s^2 from E - 3 s, band at E - 2 s
T_EXTRA = 10.0         # choice: past E the struck actor continues at its last recorded velocity (lm_offline's rule) for 10 s, long
                       # enough not to bind ("stopped and still hit"); contact within 3 s and up to E are kept as sensitivity columns
COV_OFF = 0.5          # fd_hugsim's DAC line: footprint coverage < 0.5 = off the drivable ground
DIL = 1.25             # model speed units -> simulator m/s (lm_offline)
SUBS = ["rc", "nc", "dac", "ttc", "c"]
FLAGS = ["O1", "O2b", "O2c", "L1", "L2", "L4", "R1", "R2", "D", "C1", "C2"]
CLASSES = ["other", "longitudinal", "route", "clearance"]


def a_need(g, c):
    """Required deceleration, col1_pai.a_need (same expression; that module needs the AlpaSim log reader to import)."""
    c = np.maximum(c, 0)
    return np.where(np.isnan(g), 0.0, c**2 / (2 * np.maximum(g - 0.5 * c - 1.0, 0.1)))


def ego_box(x, y, yaw):
    return box_poly([x, y, 0, EGO_W, EGO_L, 0, yaw])


def actor_at(objs, i, k, n):
    """Box of actor i at state k: recorded up to n, then constant velocity from its last recorded step; None if absent."""
    if k <= n:
        return np.asarray(objs[k][i], float) if len(objs[k]) > i else None
    a = np.asarray(objs[n][i], float)
    b = np.asarray(objs[n - 1][i], float) if len(objs[n - 1]) > i else a
    o = a.copy()
    o[:2] = a[:2] + (a[:2] - b[:2]) * (k - n)
    return o


def struck(R):
    """fd_hugsim's rule: the actor box intersecting (else nearest to) the ego box at E -> index, (lx, ly) in the ego frame, dh, speed."""
    X, Y, YAW, n = R["X"], R["Y"], R["YAW"], R["n"]
    ob = np.asarray(R["objs"][n], float)
    if not len(ob):
        return None
    ep = ego_box(X[n], Y[n], YAW[n])
    inter = [ep.intersection(box_poly(o)).area for o in ob]
    i = int(np.argmax(inter)) if max(inter) > 0 else int(np.argmin([ep.distance(box_poly(o)) for o in ob]))
    c, s = np.cos(YAW[n]), np.sin(YAW[n])
    q = ob[i, :2] - [X[n], Y[n]]
    prev = actor_at(R["objs"], i, n - 1, n)
    vo = float(np.linalg.norm(ob[i, :2] - prev[:2]) / DT) if prev is not None else np.nan
    return dict(i=i, lx=float(q[0] * c + q[1] * s), ly=float(-q[0] * s + q[1] * c), dh=float(np.degrees(wrap(ob[i, 6] - YAW[n]))),
                vo=vo, inter=float(inter[i]))


def brake_cf(R, i):
    """Ego brakes at 4 m/s^2 from E - 3 s along its driven path (pose by arc length, clamped to the path end); the actor moves as
    recorded, then at constant velocity. -> (contact up to E + T_EXTRA, up to E + 3 s, up to E, stop pose)."""
    X, Y, YAW, V, n = R["X"], R["Y"], R["YAW"], R["V"], R["n"]
    k0 = max(0, n - round(T_CF / DT))
    arc = np.r_[0.0, np.cumsum(np.hypot(np.diff(X[k0:]), np.diff(Y[k0:])))] + 1e-9 * np.arange(n - k0 + 1)
    hit, hit_3, hit_E, pose = False, False, False, None
    for j in range(n - k0 + round(T_EXTRA / DT) + 1):
        t = min(j * DT, V[k0] / BRAKE)
        sc = min(V[k0] * t - 0.5 * BRAKE * t * t, arc[-1])
        pose = [np.interp(sc, arc, a[k0:]) for a in (X, Y, YAW)]
        o = actor_at(R["objs"], i, k0 + j, n)
        if o is not None and ego_box(*pose).intersects(box_poly(o)):
            hit = True
            hit_3 |= k0 + j <= n + round(3.0 / DT)
            hit_E |= k0 + j <= n
    return hit, hit_3, hit_E, pose


def plan_hits(R, route, scene, kind, i, lat):
    """Per decision k < n: does the served plan (0.5 .. 3.0 s, ego box swept) contain an event of kind K? agent: the box at the plan
    interpolated to 0.25 s meets the struck actor's box of the same time; boundary: a plan-point footprint off the ground (coverage <
    0.5), second column fd_hugsim's plan_off (coverage or > 100 background points); corridor: the plan end is >= 1.5 m from the route
    and further out than the ego is now."""
    n = R["n"]
    hit, hit2 = np.zeros(n, bool), np.zeros(n, bool)
    for k in range(max(0, n - 32), n):
        if R["plans"][k] is None:
            continue
        w, yw = R["plans"][k]
        if kind == "agent":
            q = np.r_[[[R["X"][k], R["Y"][k]]], w]
            for j in range(1, 2 * len(w) + 1):
                a, f = (j - 1) // 2, ((j - 1) % 2 + 1) / 2
                p = q[a] + f * (q[a + 1] - q[a])
                o = actor_at(R["objs"], i, k + j, n)
                if o is not None and ego_box(p[0], p[1], yw[a]).intersects(box_poly(o)):
                    hit[k] = True
                    break
        elif kind == "boundary":
            hit[k] = min(scene.coverage(w[j, 0], w[j, 1], yw[j]) for j in range(len(w))) < COV_OFF
            hit2[k] = hit[k] or max(scene.contact(w[j, 0], w[j, 1], yw[j], R["CY"][k])[0] for j in range(len(w))) > 100
        elif kind == "corridor":
            le = abs(route.project(w[-1:])[1][0])
            hit[k] = le >= LAT_ON and le > abs(lat[k])
    return hit, hit2


def lead_time(hit, n):
    """(in-plan, lead time): a hit among the decisions of the last 3 s; E minus the first decision of the last consecutive hit run."""
    w = np.where(hit[max(0, n - round(3.0 / DT)):n])[0]
    if not len(w):
        return False, np.nan
    k = max(0, n - round(3.0 / DT)) + w[-1]
    while k > 0 and hit[k - 1]:
        k -= 1
    return True, min((n - k) * DT, 8.0)


def shapley_loss(ev):
    """1 - HD split over rc / nc / dac / ttc / c (exact Shapley; HD = rc x mean_t nc dac (5 ttc + 2 c) / 7, each factor set to 1)."""
    det = [ev["details"][k] for k in sorted(ev["details"], key=float)]

    def hd(on):
        g = lambda d, k: 1.0 if k in on else d[k]  # noqa: E731
        p = np.mean([g(d, "nc") * g(d, "dac") * (5 * g(d, "ttc") + 2 * g(d, "c")) / 7 for d in det]) if det else 1.0
        return (1.0 if "rc" in on else ev["rc"]) * p

    m = len(SUBS)
    phi = dict.fromkeys(SUBS, 0.0)
    for r in range(m):
        for S in combinations(SUBS, r):
            wgt = factorial(r) * factorial(m - r - 1) / factorial(m)
            for f in set(SUBS) - set(S):
                phi[f] += wgt * (hd(set(S) | {f}) - hd(set(S)))
    return {f"loss_{k}": v for k, v in phi.items()}, hd(set())


def analyse(row, d, route, scene, planners=()):
    """One run of a unit arm -> (record, extras for the BEV)."""
    R = FD.load_run(d)
    X, Y, YAW, V, n = R["X"], R["Y"], R["YAW"], R["V"], R["n"]
    xy = np.stack([X, Y], 1)
    s, lat = route.project(xy)
    end = row["end"]
    K = KIND.get(end, "complete" if end == "complete" else "exception")
    ev = dict(row, n_steps=n, t_E=n * DT, K=K, s_E=float(s[n]), lat_E=float(lat[n]), v_E=float(V[n - 1]), route_len=float(route.L))
    ej = json.load(open(d / "eval.json"))
    sh, hd0 = shapley_loss(ej)
    ev.update(sh, hd_recomputed=hd0)
    ex = dict(R=R, s=s, lat=lat)
    if K == "complete":
        return ev, ex
    # onset (section 2): last state before E with |lat| < 1.5 m; E if the ego never left that band at E
    far = np.abs(lat) >= LAT_ON
    on = (int(np.where(~far)[0][-1]) if (~far).any() else 0) if far[n] else n
    ss = np.linspace(s[on] - 10, s[on] + 20, 61)
    kr = route.kappa(ss)
    j = int(np.abs(kr).argmax())
    Rr, dpsi = 1 / max(abs(kr[j]), 1e-6), float(np.degrees(route.heading(ss[-1]) - route.heading(ss[0])))
    ref = "turn" if Rr < 15 or abs(dpsi) >= 30 else ("bend" if Rr < 50 or abs(dpsi) >= 10 else "straight")
    tsign = int(np.sign(kr[j])) if Rr < 50 else int(np.sign(dpsi))
    herr = np.degrees(np.abs(wrap(YAW - route.heading(s))))
    ev.update(onset=on, t_on=on * DT, s_on=float(s[on]), v_on=float(V[min(on, n - 1)]), R_ref=float(Rr), dpsi_ref=dpsi, ref=ref,
              turn_sign=tsign, herr_max_after_on=float(herr[on:].max()), a_lat_on=float(V[min(on, n - 1)] ** 2 / Rr))
    f = dict.fromkeys(FLAGS, False)
    f["O1"] = K == "exception"
    f["L4"] = K == "none"
    i = None
    if K == "agent":
        st = struck(R)
        i = st["i"]
        cf, cf_3, cf_E, pose = brake_cf(R, i)
        kb = n - round(T_BAND / DT)
        from shapely.geometry import LineString
        tip = xy[n] + EGO_L / 2 * np.array([np.cos(YAW[n]), np.sin(YAW[n])])
        band = LineString(np.r_[xy, [tip]]).buffer((EGO_W + BAND_PAD) / 2, cap_style=2)
        ob = actor_at(R["objs"], i, kb, n) if kb >= 0 else None
        in_band = bool(ob is not None and band.intersects(box_poly(ob)))
        f["O2b"] = abs(st["dh"]) >= 60 and cf
        f["O2c"] = st["lx"] < 0 and abs(st["dh"]) < 45
        f["L1"] = in_band and st["lx"] > 0 and not cf
        ev.update(obj_i=i, obj_lx=st["lx"], obj_ly=st["ly"], obj_dh=st["dh"], obj_v=st["vo"], obj_inter=st["inter"],
                  obj_planner=planners[i] if i < len(planners) else "", in_band=in_band, cf_contact=cf, cf_contact_3s=cf_3, cf_contact_to_E=cf_E)
        ex.update(cf_pose=pose, band=band)
        # base-right (lead, same moment): the run's own lead head asks for >= 1.5 m/s^2 at least 1.5 s before E
        ks = [k for k in range(0, n - round(1.5 / DT) + 1) if R["steps"].get(k, {}).get("lead_prob") is not None]
        lp = np.array([R["steps"][k]["lead_prob"] for k in ks])
        an = np.array([float(a_need(R["steps"][k]["lead_x"] - EGO_L / 2, V[k] - R["steps"][k]["lead_v"] / DIL)) for k in ks])
        ok = (lp >= 0.5) & (an >= 1.5) if len(ks) else np.zeros(0, bool)
        ev.update(lead_seen=bool((lp >= 0.5).any()), base_right_lead=bool(ok.any()),
                  lead_first_s=float((n - ks[int(np.argmax(ok))]) * DT) if ok.any() else np.nan,
                  lead_a_need_max=float(an[lp >= 0.5].max()) if (lp >= 0.5).any() else np.nan)
    edge = K in ("boundary", "corridor")
    f["L2"] = edge and ref != "straight" and ev["a_lat_on"] > 4.0
    f["R1"] = K in ("boundary", "corridor", "agent") and abs(lat[n]) >= LAT_ON and ev["herr_max_after_on"] >= 20
    f["R2"] = edge and ref == "turn"
    f["D"] = ref == "straight" and abs(lat[n]) >= LAT_ON and not f["R1"]
    f["C1"] = K == "agent" and not (f["L1"] or f["O2b"] or f["O2c"])
    f["C2"] = K == "boundary" and ref != "turn"
    if edge:   # side, fd_hugsim's amended rule: bg contact centroid, a frontal one (|ly| < 0.3 m) takes the side of the route deviation
        sl = lat[n]
        if K == "boundary":
            cnt, _, cly = scene.contact(X[n], Y[n], YAW[n], R["CY"][n])
            if cnt == 0:
                cnt, _, cly = scene.contact(X[n], Y[n], YAW[n], R["CY"][n], margin=0.3)
            ev.update(bg_pts=cnt, bg_ly=cly)
            sl = cly if np.isfinite(cly) and abs(cly) >= 0.3 else lat[n]
        ev["side"] = ("left" if sl > 0 else "right") if tsign == 0 or ref == "straight" else ("inside" if np.sign(sl) == tsign else "outside")
    O2 = f["O2b"] or f["O2c"]
    cls = ("other" if f["O1"] or O2 else "longitudinal" if f["L1"] or f["L2"] or f["L4"] else
           "route" if f["R1"] or f["R2"] or K == "corridor" else "clearance")
    sub = next((k for k in {"other": ["O1", "O2b", "O2c"], "longitudinal": ["L1", "L2", "L4"], "route": ["R1", "R2"],
                            "clearance": ["C1", "C2"]}[cls] if f[k]), "corridor_" + ("drift" if ref == "straight" else ref) if cls == "route" else "D")
    ev.update(f, cls=cls, sub=sub, flags="+".join(k for k in FLAGS if f[k]))
    # section 5: in the served plan, lead time, tracking deviation (ego vs the 0.5 s point of the plan two states earlier)
    if K != "none":
        hit, hit2 = plan_hits(R, route, scene, K, i, lat)
        ev["in_plan"], ev["plan_lead_s"] = lead_time(hit, n)
        if K == "boundary":
            ev["in_plan_fd"], ev["plan_lead_fd_s"] = lead_time(hit2, n)
        ex["hit"] = hit
    dev = np.array([np.linalg.norm(xy[k] - R["plans"][k - 2][0][0]) for k in range(2, n + 1) if R["plans"][k - 2] is not None])
    ev.update(track_dev_med=float(np.median(dev)), track_dev_max=float(dev.max()), track_dev_med_3s=float(np.median(dev[-12:])))
    cov = np.array([scene.coverage(X[k], Y[k], YAW[k]) for k in range(n + 1)])
    ev.update(cov_E=float(cov[n]), frac_off_ground=float((cov < COV_OFF).mean()))
    return ev, ex


def light(d, route, wajepa):
    """Reference arm: arc length and speed series only."""
    R = FD.load_run(d, wajepa)
    s, lat = route.project(np.stack([R["X"], R["Y"]], 1))
    return dict(s=s, V=R["V"], lat=lat)


def scene_of(d):
    """ground.ply / scene.ply of a scenario: the run dir if it keeps them, else the WA-JEPA run dir of the same scenario (the point
    sets are per scenario, identical across runs)."""
    d = Path(d)
    return FD.Scene(d if (d / "ground.ply").exists() else D / "runs/hugsim-wajepa/wajepa/wj" / d.name)


def planners_of(scenario, dataset):
    import yaml
    f = D / "datasets/hugsim/scenarios" / dataset / f"{scenario}.yaml"
    return [p[6] for p in yaml.safe_load(open(f)).get("plan_list", [])] if f.exists() else []


def _work(job):
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    sc, items, refs, xz = job
    route, scene, evs = FD.Route(xz), None, []
    ref = {}
    for nm, r in refs:
        try:
            ref[nm] = dict(light(Path(r["run_dir"]), route, nm.startswith("WA-JEPA")), end=r["end"], hd=float(r["hdscore"]))
        except Exception as e:  # noqa: BLE001
            print("reference failed", nm, sc, repr(e), flush=True)
    for arm, r in items:
        d = Path(r["run_dir"])
        scene = scene or scene_of(d)
        row = dict(arm=arm, scenario=sc, dataset=r["dataset"], difficulty=r["difficulty"], scene=r["scene"], end=r["end"],
                   hd=float(r["hdscore"]), lost=1 - float(r["hdscore"]), **{k: float(r[k]) for k in ("rc", "nc", "dac", "ttc", "c")})
        ev, _ = analyse(row, d, route, scene, planners_of(sc, r["dataset"]))
        for nm, q in ref.items():      # reference columns; closed loop: a different state at the same place, not the same moment
            ev.update({f"{nm}_end": q["end"], f"{nm}_hd": q["hd"], f"{nm}_s_max": float(q["s"].max())})
            if ev["K"] in KIND.values():
                ev[f"{nm}_same_kind"] = KIND.get(q["end"]) == ev["K"]
                ev[f"{nm}_reached"] = bool(q["s"].max() >= ev["s_E"])
                ev[f"{nm}_passed"] = bool(q["s"].max() >= min(ev["s_E"] + 5.0, route.L - 1.0) or q["end"] == "complete")
                at = np.where(q["s"] >= ev["s_on"])[0]
                ev[f"{nm}_v_on"] = float(q["V"][at[0]]) if len(at) else np.nan
        evs.append(ev)
    return evs


def extract(a):
    import pandas as pd
    from jevdrive import par
    from jevdrive.bench import tables as T
    from jevdrive.common import n_cpus
    from jevdrive.run import Run
    routes = json.load(open(D / "runs/op_parity/hugsim/routes.json"))
    with Run("lowboard_diag", "hugsim_extract", config=vars(a)) as run:
        by, refs = {}, {}
        for arm in a.arms:
            u, src = T.load("hugsim", arm, PRESET)
            run.info(f"{arm} {PRESET}: {len(u)} runs from {src}")
            for sc, r in u.iterrows():
                by.setdefault(sc, []).append((arm, r.to_dict()))
        for arm, preset in REFS:
            u, src = T.load("hugsim", arm, preset)
            if u is None:
                run.info(f"reference {arm} {preset}: not stored, column left out")
                continue
            run.info(f"reference {arm} {preset}: {len(u)} runs from {src}")
            for sc, r in u.iterrows():
                refs.setdefault(sc, []).append((f"{arm}_{preset}", r.to_dict()))
        jobs = [(sc, it, refs.get(sc, []), routes[it[0][1]["scene"]]["xz"]) for sc, it in by.items()]
        nw = a.workers or max(1, min(4, n_cpus() // 8))
        res = par.pmap(_work, jobs, workers=nw, run=run, desc="scenarios")
        res.raise_if_failed()
        E = pd.DataFrame([e for v in res.values for e in v]).sort_values(["arm", "scenario"])
        OUT.mkdir(parents=True, exist_ok=True)
        E.to_csv(OUT / "units.csv", index=False, float_format="%.4f")
        run.summary.update(runs=len(E), failed=int((E.K != "complete").sum()))
        run.info(f"{len(E)} runs -> {OUT / 'units.csv'}")


# ------------------------------------------------------------------------------------------------------------------ report
def boot_share(num, den, idx):
    """Ratio of sums with a scenario bootstrap (num, den: one value per scenario)."""
    r = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-12)
    return f"{num.sum() / max(den.sum(), 1e-12):.3f} [{np.quantile(r, .025):.3f}, {np.quantile(r, .975):.3f}]"


def report(a):
    import pandas as pd
    E = pd.read_csv(OUT / "units.csv")
    E = E[E.arm.isin(a.arms)].copy()
    for c in FLAGS + ["in_plan", "in_plan_fd", "base_right_lead", "lead_seen", "cf_contact", "cf_contact_3s", "cf_contact_to_E", "in_band"]:
        E[c] = E[c].map({True: True, False: False, "True": True, "False": False}) if c in E else np.nan
    scen = sorted(E.scenario.unique())
    idx = np.random.default_rng(0).integers(0, len(scen), (10000, len(scen)))
    per = lambda m, w=None: (E.lost if w is None else w).where(m, 0.0).groupby(E.scenario).sum().reindex(scen).fillna(0).values  # noqa: E731
    tot = per(E.lost >= 0)
    F = E[E.K != "complete"].copy()
    F[FLAGS] = F[FLAGS].astype(bool)
    L = []
    P = L.append
    P("# LOWDIAG HUGSIM tables (generated by lbd_hugsim.py report)\n")
    P(f"Arms {', '.join(a.arms)} `{PRESET}`: {len(E)} runs, {len(scen)} scenarios, mean HD {E.hd.mean():.4f}, lost score "
      f"{E.lost.sum():.2f}; failed {len(F)} ({F.end.value_counts().to_dict()}), complete {int((E.K == 'complete').sum())}. "
      "CI: scenario bootstrap of the ratio of sums, 10 000 draws, seed 0; cells with n < 8 are descriptive.\n")
    refs = [c[:-7] for c in E.columns if c.endswith("_passed")]
    rows = []
    for cls in CLASSES + ["complete with penalties", "all failed", "all"]:
        m = (E.cls == cls) if cls in CLASSES else (E.K == "complete") if cls.startswith("complete") else (E.K != "complete") if cls == "all failed" else (E.lost >= 0)
        g = E[m]
        r = dict(cls=cls, runs=len(g), scenarios=g.scenario.nunique(), lost=round(g.lost.sum(), 2), share_of_lost=boot_share(per(m), tot, idx),
                 mean_hd=round(g.hd.mean(), 3) if len(g) else np.nan)
        if cls in CLASSES or cls == "all failed":
            ag = g[g.K == "agent"]
            r["base_right_lead (agent units)"] = f"{int(ag.base_right_lead.sum())} / {len(ag)}" if len(ag) else "-"
            r["base_right_lead lost share"] = round(ag.lost[ag.base_right_lead == True].sum() / ag.lost.sum(), 3) if len(ag) else np.nan  # noqa: E712
            for nm in refs:
                q = g[g[f"{nm}_reached"] == True]  # noqa: E712
                ps = q[f"{nm}_passed"] == True  # noqa: E712
                r[f"{nm} passed / reached"] = f"{int(ps.sum())} / {len(q)}"
                r[f"{nm} passed lost share"] = round(q.lost[ps].sum() / q.lost.sum(), 3) if len(q) else np.nan
            r["in_plan"] = f"{int((g.in_plan == True).sum())} / {int(g.in_plan.notna().sum())}"  # noqa: E712
            r["plan_lead_median_s"] = g.plan_lead_s.median()
            r["track_dev_med_m"] = round(g.track_dev_med.median(), 2)
            r["track_dev_max_m"] = round(g.track_dev_max.max(), 2)
        rows.append(r)
    T1 = pd.DataFrame(rows)
    T1.to_csv(OUT / "board.csv", index=False)
    P("## T1 board: exclusive classes (registered priority other > longitudinal > route > clearance)\n")
    P(T1.to_markdown(index=False) + "\n")
    sub = F.groupby(["cls", "sub", "K", "ref"]).agg(runs=("lost", "size"), scenarios=("scenario", "nunique"), lost=("lost", "sum")).reset_index()
    sub["share_of_lost"] = sub.lost / E.lost.sum()
    sub.to_csv(OUT / "board_sub.csv", index=False, float_format="%.3f")
    P("### T1b sub-class x event kind x ref geometry\n")
    P(sub.to_markdown(index=False, floatfmt=".3f") + "\n")
    C = E[E.K == "complete"]
    cs = pd.DataFrame([dict(sub_score=k, lost=C[f"loss_{k}"].sum(), share_of_complete_lost=C[f"loss_{k}"].sum() / C.lost.sum(),
                            share_of_all_lost=C[f"loss_{k}"].sum() / E.lost.sum(), runs_with_loss=int((C[f"loss_{k}"] > 1e-6).sum())) for k in SUBS])
    cs.to_csv(OUT / "complete_split.csv", index=False, float_format="%.4f")
    P(f"### T1c complete with penalties, 1 - HD split by sub-score (exact Shapley over rc / nc / dac / ttc / c; HD recomputed from "
      f"eval.json, max |recomputed - stored| {np.abs(E.hd_recomputed - E.hd).max():.1e})\n")
    P(cs.to_markdown(index=False, floatfmt=".3f") + "\n")
    fa = pd.DataFrame([dict(sub_score=k, lost=F[f"loss_{k}"].sum(), share_of_failed_lost=F[f"loss_{k}"].sum() / F.lost.sum()) for k in SUBS])
    P("The same split on the failed runs (context: an early end costs route completion):\n")
    P(fa.to_markdown(index=False, floatfmt=".3f") + "\n")
    fl = pd.DataFrame([dict(flag=k, runs=int(F[k].sum()), scenarios=F[F[k] == True].scenario.nunique(), lost=round(F.lost[F[k] == True].sum(), 2),  # noqa: E712
                            **{c: int((F[k] & (F.cls == c)).sum()) for c in CLASSES}) for k in FLAGS])
    fl.to_csv(OUT / "flags.csv", index=False)
    P("## T2 raw flags (multi-select; the class columns say where the flagged runs ended up)\n")
    P(fl.to_markdown(index=False) + "\n")
    P("Flag combinations: " + ", ".join(f"{k or '(none)'}: {v}" for k, v in F["flags"].fillna("").value_counts().items()) + "\n")
    sens = []
    for nm, col in (("E + 10 s (primary)", "cf_contact"), ("E + 3 s", "cf_contact_3s"), ("up to E", "cf_contact_to_E")):
        cf = F[col] == True  # noqa: E712
        o2 = ((F.obj_dh.abs() >= 60) & cf) | F.O2c
        l1 = (F.in_band == True) & (F.obj_lx > 0) & ~cf  # noqa: E712
        c2 = np.where(F.K != "agent", F.cls, np.where(o2, "other", np.where(l1, "longitudinal", np.where(F.R1, "route", "clearance"))))
        sens.append(dict(actor_extrapolated_to=nm, **{f"{c} runs": int((c2 == c).sum()) for c in CLASSES},
                         **{f"{c} share": round(F.lost[c2 == c].sum() / E.lost.sum(), 3) for c in CLASSES}))
    S = pd.DataFrame(sens)
    S.to_csv(OUT / "sensitivity_cf_horizon.csv", index=False)
    P("### T2b sensitivity: how long the struck actor is carried on at its last velocity in the brake counterfactual (O2b vs C1)\n")
    P(S.to_markdown(index=False) + "\n")
    on = F[(F.K == "agent") & (F.obj_dh.abs() >= 60)]
    P(f"Oncoming / crossing struck actors (|dh| >= 60 deg): {len(on)} runs, lost share {on.lost.sum() / E.lost.sum():.3f}; by planner and class: "
      + ", ".join(f"{k[0]} / {k[1]}: {v}" for k, v in on.groupby(["obj_planner", "sub"]).size().items()) + "\n")
    ip = F.groupby(["cls", "K"]).agg(runs=("lost", "size"), in_plan=("in_plan", lambda x: int((x == True).sum())),  # noqa: E712
                                    lead_median_s=("plan_lead_s", "median"), lead_min_s=("plan_lead_s", "min"), lead_max_s=("plan_lead_s", "max"),
                                    in_plan_fd=("in_plan_fd", lambda x: int((x == True).sum())), track_dev_med=("track_dev_med", "median"),  # noqa: E712
                                    track_dev_max=("track_dev_max", "max")).reset_index()
    ip.to_csv(OUT / "in_plan.csv", index=False, float_format="%.3f")
    P("## T3 in the served plan (a hit among the decisions of the last 3 s), lead time (s), tracking deviation (m)\n")
    P("in_plan_fd (boundary only) = fd_hugsim's plan_off: footprint coverage < 0.5 or > 100 background points in a plan-point box.\n")
    P(ip.to_markdown(index=False, floatfmt=".2f") + "\n")
    ag = F[F.K == "agent"]
    br = ag.groupby(["cls", "sub"]).agg(runs=("lost", "size"), lead_seen=("lead_seen", "sum"), base_right_lead=("base_right_lead", "sum"),
                                        lead_first_median_s=("lead_first_s", "median"), a_need_max_median=("lead_a_need_max", "median"),
                                        attack_planner=("obj_planner", lambda x: int((x == "AttackPlanner").sum())),
                                        cf_contact=("cf_contact", "sum"), cf_contact_3s=("cf_contact_3s", "sum"), cf_contact_to_E=("cf_contact_to_E", "sum"), in_band=("in_band", "sum")).reset_index()
    br.to_csv(OUT / "base_right_lead.csv", index=False, float_format="%.3f")
    P("## T4 base-right (lead, same moment), K = agent: lead_prob >= 0.5 and a_need >= 1.5 m/s^2 at a step >= 1.5 s before E\n")
    P(br.to_markdown(index=False, floatfmt=".2f") + "\n")
    for nm in refs:
        t = F.groupby("cls").apply(lambda g: pd.Series(dict(runs=len(g), reached=int((g[f"{nm}_reached"] == True).sum()),  # noqa: E712
                                                              passed=int((g[f"{nm}_passed"] == True).sum()),  # noqa: E712
                                                              same_kind_end=int((g[f"{nm}_same_kind"] == True).sum()),  # noqa: E712
                                                              v_on_unit=g.v_on.median(), v_on_ref=g[f"{nm}_v_on"].median()))).reset_index()
        t.to_csv(OUT / f"ref_{nm}.csv", index=False, float_format="%.3f")
        P(f"## T5 reference {nm} (closed loop, its own state): reached = its max route arc >= s_E; passed = >= s_E + 5 m (or complete); "
          "same_kind_end = its run ends with the same event kind; v_on = median speed at the unit's onset arc\n")
        P(t.to_markdown(index=False, floatfmt=".2f") + "\n")
    cols = ["scenario", "arm", "end", "K", "hd", "lost", "t_E", "t_on", "ref", "R_ref", "dpsi_ref", "v_on", "side", "flags", "cls", "sub", "in_plan",
            "plan_lead_s", "in_plan_fd", "track_dev_med", "track_dev_max", "base_right_lead", "lead_first_s", "obj_dh", "obj_lx", "obj_ly", "obj_v",
            "obj_planner", "in_band", "cf_contact", "cf_contact_3s", "cf_contact_to_E", "lat_E", "herr_max_after_on"]
    cols += [c for nm in refs for c in (f"{nm}_end", f"{nm}_passed", f"{nm}_reached", f"{nm}_same_kind")]
    U = F[[c for c in cols if c in F.columns]].sort_values(["cls", "sub", "scenario"])
    U.to_csv(OUT / "failed_units.csv", index=False, float_format="%.3f")
    P("## T6 failed units\n")
    P(U.to_markdown(index=False, floatfmt=".2f") + "\n")
    (OUT / "tables.md").write_text("\n".join(L))
    print("\n".join(L))


# --------------------------------------------------------------------------------------------------------------------- bev
def bev(a):
    """One figure per run: BEV centred on the ego at E (left) and video frames at E - 2 s / E - 1 s / last frame (right)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MP
    from jevdrive.bench import tables as T
    sys.path.insert(0, str(REPO / "research"))
    import plot_style as PS
    PS.apply()
    C = PS.PALETTE
    routes = json.load(open(D / "runs/op_parity/hugsim/routes.json"))
    FIGS.mkdir(parents=True, exist_ok=True)
    for case in a.case:
        arm, sc = case.split("|")
        u, _ = T.load("hugsim", arm, PRESET)
        r = u.loc[sc]
        d = Path(r.run_dir)
        scene, route = scene_of(d), FD.Route(routes[r.scene]["xz"])
        ev, ex = analyse(dict(arm=arm, scenario=sc, end=r.end, hd=float(r.hdscore)), d, route, scene, planners_of(sc, r.dataset))
        R, n = ex["R"], ex["R"]["n"]
        fig = plt.figure(figsize=(PS.DOUBLE_COLUMN_IN, 4.3))
        gs = fig.add_gridspec(3, 2, width_ratios=[1.45, 1], left=0.06, right=0.99, top=0.86, bottom=0.08, wspace=0.05, hspace=0.06)
        ax = fig.add_subplot(gs[:, 0])
        cx, cy, rad = R["X"][n], R["Y"][n], a.radius
        gi = scene.gt.query_ball_point([cx, cy], rad * 1.5)
        si = [i for i in scene.st.query_ball_point([cx, cy], rad * 1.5) if R["CY"][n] < scene.sy[i] < R["CY"][n] + FD.EGO_H]
        ax.scatter(*scene.gxy[gi][::4].T, s=0.3, c="#cfe8d5", rasterized=True, linewidths=0)
        ax.scatter(*scene.sxy[si][::3].T, s=0.3, c="#9e9e9e", rasterized=True, linewidths=0)
        ax.plot(*route.p.T, "--", color="k", lw=0.8, label="recorded route (ref)")
        ax.plot(R["X"], R["Y"], "-", color=C["vermillion"], lw=1.4, label="ego, driven")
        for j, k in enumerate(range(max(0, n - 12), n, 4)):
            if R["plans"][k] is not None:
                w = R["plans"][k][0]
                ax.plot(np.r_[R["X"][k], w[:, 0]], np.r_[R["Y"][k], w[:, 1]], "-o", ms=1.5, color=C["blue"], alpha=0.35 + 0.3 * j, lw=1.0,
                        label="served plan at -3 / -2 / -1 s" if j == 2 else None)
        hi = ev.get("obj_i")
        for kk, al in ((n, 1.0), (max(0, n - 8), 0.35)):
            for i, o in enumerate(R["objs"][kk]):
                ax.add_patch(MP(np.asarray(box_poly(o).exterior.coords), fill=i == hi, fc=C["purple"], ec=C["purple"] if i == hi else "#555555",
                                alpha=al * (0.6 if i == hi else 1.0), lw=0.9))
            ax.add_patch(MP(np.asarray(ego_box(R["X"][kk], R["Y"][kk], R["YAW"][kk]).exterior.coords), fill=False, ec=C["vermillion"], alpha=al, lw=1.1))
        if "cf_pose" in ex:
            ax.add_patch(MP(np.asarray(ego_box(*ex["cf_pose"]).exterior.coords), fill=False, ec=C["green"], lw=1.1, ls=":", label="ego stop, 4 m/s$^2$ from E - 3 s"))
            ax.plot(*ex["band"].exterior.xy, color=C["orange"], lw=0.5, alpha=0.7, label="swept band (width + 0.5 m)")
        if ev.get("onset", n) < n:
            ax.plot(R["X"][ev["onset"]], R["Y"][ev["onset"]], "x", color="k", ms=6, label="onset (|lat| < 1.5 m last)")
        ax.set_xlim(cx - rad, cx + rad)
        ax.set_ylim(cy - rad * 0.8, cy + rad * 0.8)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.legend(fontsize=5.5, loc="best", frameon=True, framealpha=0.8)
        ax.set_xlabel("x (m); purple filled = struck actor; faint = 2 s before E")
        num = lambda k, f=".1f": "-" if ev.get(k) is None or not np.isfinite(ev.get(k)) else format(ev[k], f)  # noqa: E731
        fig.suptitle(f"{sc} / {arm}: {r.end}, HD {r.hdscore:.2f}, E {ev['t_E']:.2f} s, onset {ev['t_on']:.2f} s, ref {ev['ref']} "
                     f"(R {min(ev['R_ref'], 999):.0f} m, dpsi {ev['dpsi_ref']:.0f} deg), v_on {ev['v_on']:.1f} m/s\n"
                     f"class {ev['cls']} / {ev['sub']}; flags {ev['flags'] or '-'}; in-plan {ev.get('in_plan')} (fd {ev.get('in_plan_fd', '-')}) lead {num('plan_lead_s', '.2f')} s; "
                     f"obj dh {num('obj_dh', '.0f')} deg lx {num('obj_lx')} ly {num('obj_ly')} v {num('obj_v')} {ev.get('obj_planner', '')}; "
                     f"band {ev.get('in_band', '-')} cf {ev.get('cf_contact', '-')}; lead-right {ev.get('base_right_lead', '-')}; "
                     f"lat_E {ev['lat_E']:.1f} herr {ev['herr_max_after_on']:.0f}", fontsize=6.2, x=0.01, ha="left")
        try:
            import cv2
            cap = cv2.VideoCapture(str(d / "video.mp4"))
            nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for row, back in enumerate((8, 4, 0)):
                fi = max(0, nf - 1 - round(back * nf / max(n, 1)))
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok, im = cap.read()
                axv = fig.add_subplot(gs[row, 1])
                axv.axis("off")
                if ok:
                    im = im[im.max(axis=(1, 2)) > 8]            # drop the empty camera rows of single-row rigs
                    h, w = im.shape[:2]
                    axv.imshow(cv2.resize(im[:, :, ::-1], (480, round(480 * h / w)), interpolation=cv2.INTER_AREA))
                    axv.text(0.01, 0.97, f"video frame {fi} / {nf} (E - {back * DT:.0f} s)", transform=axv.transAxes, va="top", fontsize=5.5, color="w",
                             bbox=dict(fc="k", alpha=0.5, lw=0, pad=1))
        except Exception as e:  # noqa: BLE001
            print("video frames skipped:", repr(e))
        out = FIGS / f"{a.prefix}{sc}_{arm}.png"
        fig.savefig(out, dpi=a.dpi, pil_kwargs={"optimize": True})
        plt.close(fig)
        print("wrote", out, out.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract")
    e.add_argument("--workers", type=int, default=0)
    r = sp.add_parser("report")
    b = sp.add_parser("bev")
    b.add_argument("--case", nargs="+", required=True, help="arm|scenario")
    b.add_argument("--radius", type=float, default=22.0)
    b.add_argument("--dpi", type=int, default=170)
    b.add_argument("--prefix", default="bev_")
    for q in (e, r):
        q.add_argument("--arms", nargs="+", default=ARMS)
    a = ap.parse_args()
    {"extract": extract, "report": report, "bev": bev}[a.cmd](a)
