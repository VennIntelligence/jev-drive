"""BODY1 diagnosis of the baseline zeros: tables and figures from bd1_diag_read.py truth (called through `bd1_diag_read.py tables | figs`).

Class of a zero (pre = decisions before the first failing sample; "in plan" = the served plan's own 4 s sweep shows the failure):
  collision  in plan = the sweep meets the struck object.   offroad  in plan = a footprint corner leaves the scorer's road union.
  corridor   in plan = the box centre is >= 4 m from the logged path.
  ii   no pre decision has it in plan (drift / tracking)
  iii  offroad: every in-plan decision is at most 0.20 m deep (the head's label band), or the NAVSIM raster calls the exit drivable (margin >= -0.20 m);
       corridor: the executed ego is on the road and touches nothing when it leaves the corridor (a route failure, nothing physical)
  iv   in plan, but the first contact point is outside the front camera's view at every in-plan decision
  F    in plan and flagged by the head (agent logit for collisions, boundary logit otherwise) at an in-plan decision after decision 0
  i    in plan, deep, in view, never flagged at an in-plan decision after decision 0
"""
import csv
import json

import numpy as np

import bd1_diag as G
import bd1_diag_read as T
import swv1_lib as S

OUT, RES, FIG = T.OUT, T.RES, T.FIG
COL = {"collision": "#D55E00", "offroad": "#0072B2", "corridor": "#009E73", "clean": "#999999"}


def num(x, d=np.nan):
    return d if x in ("", None) else float(x)


def load():
    D = list(csv.DictReader(open(OUT / "dec.csv")))
    for d in D:
        for k, v in d.items():
            if k not in ("set", "kind", "reason", "scene", "a_obj"):
                d[k] = num(v)
    SC = list(csv.DictReader(open(OUT / "scenes.csv")))
    C = dict(np.load(OUT / "cand.npz"))
    for i, d in enumerate(D):
        d["i"] = i
    return D, SC, C


def inplan(d):
    return bool({"collision": d["hit_struck"], "offroad": d["b_out"], "corridor": d["c_out"]}[d["reason"]] > 0)


def flagged(d):
    return bool(d["flag_a"] > 0) if d["reason"] == "collision" else bool(d["flag_b"] > 0)


def classify(sc, P):
    """One zero (its scene row and pre-event decisions) -> (class, facts)."""
    why, ip = sc["reason"], [d for d in P if inplan(d)]
    acted = [d for d in ip if d["k"] >= 1]
    f = dict(n_pre=len(P), n_inplan=len(ip), lead_inplan=max((d["t_ev"] - d["t"] for d in ip), default=np.nan), n_flag=sum(flagged(d) for d in P if d["k"] >= 1),
             n_inplan_flag=sum(flagged(d) for d in acted), lead_flag=max((d["t_ev"] - d["t"] for d in acted if flagged(d)), default=np.nan),
             flag_other=sum(bool(d["flag_b"] > 0) if why == "collision" else bool(d["flag_a"] > 0) for d in P if d["k"] >= 1),
             track_err_max=np.nanmax([d["track_err"] for d in P] + [np.nan]) if len(P) > 1 else np.nan)
    if why == "corridor":
        phys = num(sc["ex_depth_at_ev"], 0) > 0 or num(sc["ex_min_obst"], 9) <= 0
        f.update(plan_b_out=sum(d["b_out"] > 0 for d in P), plan_a_hit=sum(d["a_hit"] > 0 for d in P))
        return ("body" if phys else "iii"), f
    if not ip:
        return "ii", f
    if why == "offroad":
        f["depth_max"] = max(d["b_depth"] for d in ip)
        rm = [d["r_margin_at_exit"] for d in ip if np.isfinite(d["r_margin_at_exit"])]
        f["raster_at_exit_min"] = min(rm) if rm else np.nan
        if f["depth_max"] <= T.BAND or (rm and min(rm) >= -T.BAND):
            return "iii", f
    vis = [d["struck_vis" if why == "collision" else "b_vis"] > 0 for d in ip]
    f["n_vis"] = sum(vis)
    if not any(vis):
        return "iv", f
    return ("F" if f["n_inplan_flag"] else "i"), f


def boot_auc(y, s, g):
    y, s, g = np.asarray(y, bool), np.asarray(s, float), np.asarray(g)
    if y.sum() < 3 or (~y).sum() < 3:
        return dict(auc=np.nan, lo=np.nan, hi=np.nan, pos=int(y.sum()), neg=int((~y).sum()), logs=len(set(g[y])))
    a, lo, hi = S.boot_ci(lambda i: S.wauc(y[i], s[i]), g, 2000)
    return dict(auc=a, lo=lo, hi=hi, pos=int(y.sum()), neg=int((~y).sum()), logs=len(set(g[y])))


def write(name, rows):
    RES.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(RES / name, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows([{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in rows])


def cmd_tables(a, run):
    D, SC, C = load()
    by = {}
    for d in D:
        by.setdefault((d["set"], d["scene"]), []).append(d)
    summ = dict(check=dict(off_agree=sum(int(x["off_agree"]) for x in SC), off_n=sum(int(x["off_n"]) for x in SC), lat_err_max=max(float(x["lat_err"]) for x in SC),
                           replay_diff_max=max(float(x["replay_diff"]) for x in SC), reg_res=[round(num(x["reg_res"]), 3) for x in SC if x["reg_res"] != ""].__len__(),
                           reg_res_max=max([num(x["reg_res"]) for x in SC if x["reg_res"] != ""] + [0]), reg_t=sorted({round(num(x["reg_t"]), 1) for x in SC if x["reg_t"] != ""})))
    # ---- per zero
    Z = []
    for sc in SC:
        if sc["kind"] != "zero":
            continue
        dd = sorted(by[sc["set"], sc["scene"]], key=lambda d: d["k"])
        P = [d for d in dd if d["pre"] > 0]
        cls, f = classify(sc, P)
        ceil = {}
        for fam in ("clear_lat", "clear_lat_elig", "clear_slow", "clear_both", "a_clear_lat", "a_clear_slow", "a_clear_both"):
            ks = [d for d in P if d[fam] > 0]
            ceil[fam + "_k"] = min((int(d["k"]) for d in ks), default="")
            ceil[fam + "_lead"] = max((d["t_ev"] - d["t"] for d in ks), default=np.nan)
        anyc = [d for d in P if d["clear_lat"] > 0 or d["clear_slow"] > 0 or d["clear_both"] > 0]
        st0 = P[-1] if P else dd[0]
        fi = next((d for d in P if inplan(d)), st0)
        turn = abs(num(sc["tok_turn_deg"]))
        man = "launch" if float(sc["v_start"]) < 1 else "turn > 45" if turn > 45 else "turn 20-45" if turn > 20 else "straight"
        Z.append(dict(seed=sc["seed"], scene=sc["scene"][-16:], reason=sc["reason"], cls=cls, t_ev=num(sc["t_ev"]), manoeuvre=man, tok_turn_deg=num(sc["tok_turn_deg"]), log_turn_deg=float(sc["log_turn_deg"]),
                      v_start=float(sc["v_start"]), **f, any_clear_lead=max((d["t_ev"] - d["t"] for d in anyc), default=np.nan), **ceil,
                      head_clear_at_flag=sum(d["head_clear"] > 0 for d in P if flagged(d) and d["k"] >= 1), ex_depth_at_ev=num(sc["ex_depth_at_ev"]), ex_raster_at_ev=num(sc["ex_raster_at_ev"]),
                      ex_lat_max=float(sc["ex_lat_max"]), ex_min_obst=float(sc["ex_min_obst"]), v_first=fi["v0"], lat_first=fi["lat_off"], head_first=fi["head_err_deg"], ahead_first=fi["ahead_m"],
                      v_last=st0["v0"], lat_last=st0["lat_off"], head_last=st0["head_err_deg"], struck_v=fi["struck_v"], full=sc["scene"]))
    write("zeros.csv", Z)
    cnt = {}
    for z in Z:
        for key in (z["reason"], z["reason"] + " > 45"):
            if key.endswith("45") and not abs(z["tok_turn_deg"]) > 45:
                continue
            cnt.setdefault(key, {}).setdefault(z["cls"], 0)
            cnt[key][z["cls"]] += 1
    summ["classes"] = cnt
    summ["manoeuvre"] = {m: {r: sum(z["manoeuvre"] == m and z["reason"] == r for z in Z) for r in ("collision", "offroad", "corridor")} for m in ("launch", "straight", "turn 20-45", "turn > 45")}
    # ---- ceilings
    ce = []
    for why in ("collision", "offroad", "corridor"):
        for sub, zz in (("all", [z for z in Z if z["reason"] == why]), ("> 45 deg", [z for z in Z if z["reason"] == why and abs(z["tok_turn_deg"]) > 45])):
            r = dict(reason=why, subset=sub, n=len(zz))
            for fam in ("clear_lat", "clear_lat_elig", "clear_slow", "clear_both", "a_clear_lat", "a_clear_slow", "a_clear_both"):
                L = [z[fam + "_lead"] for z in zz if np.isfinite(z[fam + "_lead"])]
                r[fam], r[fam + "_lead_med"] = len(L), float(np.median(L)) if L else np.nan
            L = [z["any_clear_lead"] for z in zz if np.isfinite(z["any_clear_lead"])]
            r["any"], r["any_lead_med"], r["any_lead_ge1.5"] = len(L), float(np.median(L)) if L else np.nan, sum(x >= 1.5 for x in L)
            ce.append(r)
    write("ceiling.csv", ce)
    # ---- the head on these decisions
    A = []
    seg = np.array([d["scene"].rsplit("-", 1)[0] for d in D])
    arr = lambda k: np.array([d[k] for d in D], float)  # noqa: E731
    za, zb, ah, dep, rmg, v0, lat, he, k_, kind = arr("za"), arr("zb"), arr("a_hit") > 0, arr("b_depth"), arr("r_margin"), arr("v0"), np.abs(arr("lat_off")), np.abs(arr("head_err_deg")), arr("k"), np.array([d["kind"] for d in D])
    env = (lat <= 0.5) & (he <= 2.0) & (v0 > 3)
    subs = {"all decisions": np.ones(len(D), bool), "decision >= 1": k_ >= 1, "zero scenes": kind == "zero", "clean scenes": kind == "clean", "v0 < 3 m/s": v0 < 3, "v0 >= 3 m/s": v0 >= 3,
            "inside the ot1 envelope (|lat| <= 0.5 m, |heading| <= 2 deg, > 3 m/s)": env, "outside it": ~env}
    for name, m in subs.items():
        r = dict(subset=name, n=int(m.sum()))
        for lab, y, ok, s, thr in (("agent", ah, np.ones(len(D), bool), za, T.FA), ("road (scorer map, > 0.20 m)", dep > T.BAND, (dep > T.BAND) | (dep == 0), zb, T.FB),
                                   ("road (NAVSIM raster, < -0.20 m)", rmg < -T.BAND, np.isfinite(rmg) & ((rmg < -T.BAND) | (rmg >= 0)), zb, T.FB)):
            mm = m & ok
            b = boot_auc(y[mm], s[mm], seg[mm])
            A.append(dict(r, truth=lab, **b, recall_at_flag=float((s[mm & y] >= thr).mean()) if (mm & y).any() else np.nan, flag_rate_neg=float((s[mm & ~y] >= thr).mean()) if (mm & ~y).any() else np.nan))
    write("head_auc.csv", A)
    # ---- road truth: scorer map against the raster, plan level
    both = np.isfinite(rmg)
    summ["map_vs_raster"] = dict(n=int(both.sum()), map_out=int((dep[both] > 0).sum()), map_deep=int((dep[both] > T.BAND).sum()), raster_pos=int((rmg[both] < -T.BAND).sum()),
                                 map_deep_raster_not=int(((dep[both] > T.BAND) & (rmg[both] >= -T.BAND)).sum()), raster_pos_map_in=int(((rmg[both] < -T.BAND) & (dep[both] == 0)).sum()),
                                 map_shallow=int(((dep[both] > 0) & (dep[both] <= T.BAND)).sum()))
    # ---- states against the row set
    stt = []
    pre = arr("pre") > 0
    reason = np.array([d["reason"] for d in D])
    for name, m in [(w, (reason == w) & pre) for w in ("collision", "offroad", "corridor")] + [("all zeros", (kind == "zero") & pre), ("clean", kind == "clean")]:
        n = max(int(m.sum()), 1)
        stt.append(dict(group=name, n=int(m.sum()), v_lt1=(v0[m] < 1).sum() / n, v_1_3=((v0[m] >= 1) & (v0[m] < 3)).sum() / n, v_ge3=(v0[m] >= 3).sum() / n,
                        lat_le05=(lat[m] <= 0.5).sum() / n, lat_05_1=((lat[m] > 0.5) & (lat[m] <= 1)).sum() / n, lat_1_2=((lat[m] > 1) & (lat[m] <= 2)).sum() / n, lat_gt2=(lat[m] > 2).sum() / n,
                        head_le2=(he[m] <= 2).sum() / n, head_2_5=((he[m] > 2) & (he[m] <= 5)).sum() / n, head_5_10=((he[m] > 5) & (he[m] <= 10)).sum() / n, head_gt10=(he[m] > 10).sum() / n,
                        on_log=((lat[m] <= 0.15) & (he[m] <= 0.7)).sum() / n, ot1_env=env[m].sum() / n, lat_med=float(np.median(lat[m])) if m.any() else np.nan, head_med=float(np.median(he[m])) if m.any() else np.nan,
                        track_err_med=float(np.nanmedian(arr("track_err")[m])) if m.any() else np.nan))
    write("states.csv", stt)
    # in-plan decisions of the zeros: flagged or not, by state
    ipm = np.array([d["kind"] == "zero" and d["pre"] > 0 and d["k"] >= 1 and inplan(d) for d in D])
    fl = np.array([flagged(d) if d["kind"] == "zero" else False for d in D])
    ms = []
    for why in ("collision", "offroad", "corridor"):
        for name, m in (("all", np.ones(len(D), bool)), ("v0 < 3", v0 < 3), ("v0 >= 3, in ot1 envelope", env), ("v0 >= 3, outside", (v0 >= 3) & ~env)):
            mm = ipm & (reason == why) & m
            ms.append(dict(reason=why, state=name, in_plan_decisions=int(mm.sum()), flagged=int((mm & fl).sum()), za_med=float(np.median(za[mm])) if mm.any() else np.nan, zb_med=float(np.median(zb[mm])) if mm.any() else np.nan))
    write("inplan_flags.csv", ms)
    # clean plans: how often does the truth call a clean scene's plan a contact
    cl = kind == "clean"
    summ["clean_plans"] = dict(n=int(cl.sum()), a_hit=int((ah & cl).sum()), road_out=int(((dep > 0) & cl).sum()), road_deep=int(((dep > T.BAND) & cl).sum()), corridor=int(((arr("c_out") > 0) & cl).sum()),
                               flag_a=int(((za >= T.FA) & cl & (k_ >= 1)).sum()), flag_b=int(((zb >= T.FB) & cl & (k_ >= 1)).sum()), n_k1=int((cl & (k_ >= 1)).sum()))
    summ["n"] = dict(decisions=len(D), zero_scenes=len(Z), clean_scenes=sum(x["kind"] == "clean" for x in SC))
    (RES / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    run.info(json.dumps(summ, default=float))


# ---------------------------------------------------------------- figures
def y_plane(fr):
    """openpilot 6-plane YUV (6, 128, 256) -> luma (256, 512)."""
    y = np.zeros((256, 512), np.uint8)
    y[0::2, 0::2], y[1::2, 0::2], y[0::2, 1::2], y[1::2, 1::2] = fr[0], fr[1], fr[2], fr[3]
    return y


def strip(ax, ctx, o, k, title):
    import shapely
    r, U = ctx["r"], ctx["U"]
    now, anchor = r.now[k], np.array(o["rec"][k]["anchor"], float)
    W = S.dense(anchor, ctx["P8"][k])
    c0 = S.centre(W[:1], r.off)[0]
    R = 32
    for g in (getattr(U, "geoms", [U])):
        if g.geom_type == "Polygon" and g.distance(shapely.Point(c0[:2])) < 2 * R:
            ax.fill(*g.exterior.xy, color="#e9e9e9", zorder=0)
            for h in g.interiors:
                ax.fill(*h.xy, color="white", zorder=0)
    for oid in r.ids:
        p, ok = r.obj_pose(oid, now)
        if ok[0] and np.hypot(*(p[0, :2] - c0[:2])) < 1.5 * R:
            ax.fill(*shapely.get_coordinates(S.boxes(p[0], *r.size(oid))).T, color="#7f7f7f" if oid != ctx.get("struck") else "#D55E00", alpha=0.8, zorder=2)
    gt = r.gt_rig
    ax.plot(gt[:, 1], gt[:, 2], "k--", lw=1, zorder=3)
    e = r.ego[r.ego[:, 0] <= now + 1]
    ax.plot(*CLrig(e, r.off).T, color="#E69F00", lw=2, zorder=4)
    ax.plot(W[:, 0], W[:, 1], color="#0072B2", lw=1.6, zorder=5)
    for j in (0, 20, 40):
        ax.plot(*np.r_[T.corner_pts(S.centre(W[j:j + 1], r.off)[0], r.Le, r.We), T.corner_pts(S.centre(W[j:j + 1], r.off)[0], r.Le, r.We)[:1]].T, color="#0072B2", lw=0.8, zorder=5)
    ax.set_xlim(c0[0] - R, c0[0] + R), ax.set_ylim(c0[1] - R, c0[1] + R), ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(title, fontsize=7)


def CLrig(c, off):
    import col1_lib as CL
    return CL.to_rig(c[:, 1:4], off)[:, :2]


def cmd_figs(a, run):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D, SC, C = load()
    FIG.mkdir(parents=True, exist_ok=True)
    Z = list(csv.DictReader(open(RES / "zeros.csv")))
    by = {}
    for d in D:
        by.setdefault((d["set"], d["scene"]), []).append(d)
    # ---- 1. decision matrix
    order = sorted(Z, key=lambda z: (("collision", "offroad", "corridor").index(z["reason"]), z["cls"], -float(z["t_ev"])))
    fig, ax = plt.subplots(figsize=(7.5, 10))
    cm = {"none": "#f3f3f3", "inplan": "#E69F00", "both": "#009E73", "flag": "#56B4E9", "post": "#bdbdbd"}
    for i, z in enumerate(order):
        for d in by[f"zero-s{z['seed']}", z["full"]]:
            st = "post" if d["pre"] <= 0 else "both" if inplan(d) and flagged(d) else "inplan" if inplan(d) else "flag" if flagged(d) else "none"
            ax.add_patch(plt.Rectangle((d["k"], i), 0.94, 0.9, color=cm[st]))
            if d["pre"] > 0 and (d["clear_lat"] > 0 or d["clear_slow"] > 0 or d["clear_both"] > 0) and inplan(d):
                ax.plot(d["k"] + 0.47, i + 0.45, "k.", ms=3)
        ax.text(10.15, i + 0.45, f"{z['reason'][:4]} {z['cls']:>4}  {z['manoeuvre']}", fontsize=6, va="center", family="monospace")
    ax.set_xlim(0, 14.5), ax.set_ylim(len(order), 0), ax.set_xticks(np.arange(10) + 0.47), ax.set_xticklabels([f"{0.5 * k:.1f}" for k in range(10)], fontsize=7)
    ax.set_yticks([]), ax.set_xlabel("decision time in the scene, s")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in cm.values()], labels=["plan clean, no flag", "failure in the plan, no flag", "in the plan and flagged", "flag, not in the plan", "after the failure"],
              fontsize=6, loc="lower right", frameon=False)
    ax.set_title("P2H10-F baseline zeros (2 seeds): the failure in the served plan against the head's flag; dot = a clear candidate exists", fontsize=8)
    fig.savefig(FIG / "decision_matrix.png", dpi=150, bbox_inches="tight"), plt.close(fig)
    # ---- 2. states
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    for why in ("clean", "collision", "offroad", "corridor"):
        P = [d for d in D if d["reason"] == why and d["pre"] > 0]
        axs[0].scatter([d["lat_off"] for d in P], [d["head_err_deg"] for d in P], s=7 if why == "clean" else 12, color=COL[why], alpha=0.6, label=f"{why} ({len(P)})", zorder=1 if why == "clean" else 2)
        axs[1].hist([d["v0"] for d in P], bins=np.arange(0, 16, 1), histtype="step", color=COL[why], density=True, lw=1.5)
        Q = [d for d in P if d["k"] >= 1]
        axs[2].scatter([d["zb"] for d in Q], [-d["b_depth"] if d["b_depth"] > 0 else min(num(d["r_margin"], 3), 3) for d in Q], s=7, color=COL[why], alpha=0.6)
    axs[0].add_patch(plt.Rectangle((-0.5, -2), 1, 4, fill=False, ec="k", lw=1)), axs[0].set_xlim(-5, 5), axs[0].set_ylim(-30, 30)
    axs[0].set_xlabel("lateral offset from the logged path, m"), axs[0].set_ylabel("heading minus the logged path's, deg"), axs[0].legend(fontsize=7)
    axs[0].set_title("ego state at each decision before the failure; box = ot1 rows (+-0.5 m, +-2 deg)", fontsize=8)
    axs[1].axvline(3, color="k", lw=0.8), axs[1].set_xlabel("ego speed, m/s"), axs[1].set_title("speed (off-track rows exist only right of the line)", fontsize=8)
    axs[2].axvline(T.FB, color="k", lw=0.8), axs[2].axhline(-T.BAND, color="k", lw=0.8, ls=":"), axs[2].set_xlabel("boundary logit of the served plan"), axs[2].set_ylabel("plan's road margin: -depth (scorer map) when out, else raster margin, m")
    axs[2].set_title("boundary logit against the truth of the same plan (decisions >= 1)", fontsize=8)
    fig.savefig(FIG / "states.png", dpi=150, bbox_inches="tight"), plt.close(fig)
    # ---- 3. BEV strips: one per class where it exists
    tok, rows = T.tokens(), list(csv.DictReader(open(OUT / "zeros.csv")))
    picks, seen = [], set()
    for z in sorted(Z, key=lambda z: -float(z["n_inplan"] or 0)):
        key = (z["reason"], z["cls"])
        if key not in seen:
            seen.add(key), picks.append(z)
    cases = []
    for st in ("zero-s0", "zero-s1"):
        L, M, Zr = T.load_set(st)
        for z in [p for p in picks if f"zero-s{p['seed']}" == st]:
            row = next(x for x in rows if x["scene"] == z["full"] and x["seed"] == z["seed"])
            _, sc, _, ctx = T.one_scene(st, "zero", row, L[z["full"]], M[z["full"]], Zr, tok(z["full"].rsplit("-", 1)[1]))
            ctx["struck"] = sc["struck"]
            dd = sorted(by[st, z["full"]], key=lambda d: d["k"])
            P = [d for d in dd if d["pre"] > 0]
            ks = sorted({int(P[0]["k"]), int(next((d["k"] for d in P if inplan(d)), P[len(P) // 2]["k"])), int(P[-1]["k"]), min(int(P[-1]["k"]) + 1, len(dd) - 1)})
            fig, axs = plt.subplots(2, len(ks), figsize=(3.6 * len(ks), 6.2), gridspec_kw=dict(height_ratios=[3, 1.5]))
            axs = axs.reshape(2, -1)
            fr = Zr["frame"][Zr["scene"] == z["full"]][np.argsort(Zr["k"][Zr["scene"] == z["full"]])]
            for j, k in enumerate(ks):
                d = dd[k]
                strip(axs[0, j], ctx, L[z["full"]], k, f"t {d['t']:.1f} s, {d['v0']:.1f} m/s, lat {d['lat_off']:+.1f} m, head {d['head_err_deg']:+.0f} deg\n"
                      f"agent logit {d['za']:+.1f} (flag {T.FA:.1f}), boundary {d['zb']:+.1f} (flag {T.FB:.1f})\nplan truth: object {'hit' if d['a_hit'] > 0 else 'clear'}, road depth {d['b_depth']:.2f} m, corridor {d['c_max']:.1f} m")
                axs[1, j].imshow(y_plane(fr[k]), cmap="gray"), axs[1, j].set_xticks([]), axs[1, j].set_yticks([])
            name = f"bev_{z['reason']}_{z['cls']}.png"
            fig.suptitle(f"{z['reason']} zero, class {z['cls']}, seed {z['seed']}, {z['full'][-16:]}, failure at {float(z['t_ev']):.1f} s ({z['manoeuvre']})", fontsize=9)
            fig.savefig(FIG / name, dpi=130, bbox_inches="tight"), plt.close(fig)
            cases.append(dict(fig=name, seed=z["seed"], scene=z["full"], reason=z["reason"], cls=z["cls"], decisions=" ".join(map(str, ks))))
    write("bev_cases.csv", cases)
    run.info(f"figures -> {FIG}: decision_matrix, states, {len(cases)} strips")
