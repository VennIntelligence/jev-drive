#!/usr/bin/env python3
"""BODY1 progress diagnosis, closed-loop part (results/progress_diagnosis.md sections 1, 3, 4): where the loss arm (P2H10B-F) loses progress
against P2H10-F on the AlpaSim nuPlan track. Reads the existing runs only (700 scenes x 2 seeds each); nothing is run or trained.

  tables   per (seed, scene) pair: scores, progress, driven distance, the served plans of drive.jsonl, joined with the scene's navtest token
           (prog_ol.py's ol_navtest.parquet: proximity group of the base plan at the token, speed, logged turn) -> results/prog/cl_*.csv, cl_summary.json
  raster   the new offroad zeros against the two rasters (NAVSIM: ROADBLOCK + INTERSECTION + CARPARK_AREA; item C: road-and-lane, no car parks):
           footprint margins of both closed-loop traces, left and right corners separately -> results/prog/cl_offroad_raster.csv, figs/prog/offroad_*.png
  figs     figs/prog/{ol_arc,cl_progress,grad_terms}.png from the csv tables

Scene score = (no at-fault collision) x (not offroad) x (corridor kept) x min(progress / 0.8, 1): a scene is "slow" when its progress relative to the
log is under 0.8. Pairs are pooled over the two seeds; intervals by log (jevdrive.stats.paired). Box, envs/op-train python.
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_H = _pl.Path(__file__).resolve()
_sys.path[:0] = [str(_H.parent), str(_H.parents[1] / "lib"), str(_H.parents[2] / "alpasim" / "scripts"), str(_H.parents[3])]
import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
REPO = _H.parents[3]
OUT, FIG = REPO / "experiments/body1/results/prog", REPO / "experiments/body1/figs/prog"
GROUPS = ("contact", "lead", "near obj", "near edge", "open")
ARM, MAN, OLN = "P2H10B-F", "runs/body1/cl/loss-man.json", "ol_navtest"            # --arm / --man / --ol: another arm of the lane (Amendment 6)
ZC = {"collision_at_fault": "collision", "offroad": "offroad", "left_corridor_laterally": "corridor"}
FRONT, REAR, HALF_W = 4.049, -1.127, 1.1485


def run_dirs():
    arm = json.loads((DATA / MAN).read_text())
    base = {f"P2H10-F-s{i}": [sorted(glob.glob(str(DATA / f"runs/alpasim/tr1/a/runs/P2H10-F-s{i}-chunk{j}/*")))[-1] for j in range(3)] for i in (0, 1)}
    return base | arm


def arc4(poses):
    p = np.concatenate([np.zeros((1, 2)), np.asarray(poses, float)[:, :2]])
    return float(np.hypot(*np.diff(p, axis=0).T).sum())


def trace(d, session):
    """Controller trace of a rollout in the frame of its first pose -> (t s, x, y, yaw)."""
    import stop_report as SR
    c = SR.ctrl(d, session)
    if c is None:
        return None
    yaw = 2 * np.arctan2(c["qz"], c["qw"])
    co, si = np.cos(yaw[0]), np.sin(yaw[0])
    dx, dy = c["x"] - c["x"][0], c["y"] - c["y"][0]
    return np.stack([(c["timestamp_us"] - c["timestamp_us"][0]) / 1e6, co * dx + si * dy, -si * dx + co * dy, np.unwrap(yaw - yaw[0])], 1)


def load():
    """-> DataFrame with one row per (seed, scene): both drivers' score / class / progress / distances / k = 0 plan arc, token features."""
    import pandas as pd
    import c0b_report as R
    import stop_report as SR
    from prog_ol import group
    dirs = run_dirs()
    T = pd.read_parquet(DATA / f"runs/body1/prog/{OLN}.parquet")
    T["grp"] = group(T)
    T = T.set_index("name")
    rows, keep = [], {}
    for i in (0, 1):
        rec = {}
        for side, k in (("base", f"P2H10-F-s{i}"), ("arm", f"{ARM}-s{i}")):
            Rr = R.load_driver(dirs[k])[0]
            D, where = SR.drives(dirs[k])
            rec[side] = (Rr, D, where)
            keep[(i, side)] = (D, where)
        for s in sorted(set(rec["base"][0]) & set(rec["arm"][0])):
            tok = s.rsplit("-", 1)[1]
            r = dict(seed=i, scene=s, tok=tok, log=R.log_of(s))
            for side in ("base", "arm"):
                Rr, D, _ = rec[side]
                x, m = Rr[s], Rr[s]["metrics"]
                zc = next((ZC[f] for f in ZC if x["score_metrics"].get(f)), "other" if x["score"] == 0 else "")
                r |= {f"{side}_score": x["score"], f"{side}_zero": zc, f"{side}_prog": x["score_metrics"].get("progress_clipped_rel") or 0.0,
                      f"{side}_dist": m.get("dist_traveled_m"), f"{side}_obs": m.get("min_distance_to_obstacle_m"), f"{side}_lane": m.get("min_distance_to_lane_boundary_m"),
                      f"{side}_arc0": arc4(D[s][0]["poses"]) if s in D else np.nan, f"{side}_arcmean": float(np.mean([arc4(q["poses"]) for q in D[s]])) if s in D else np.nan}
            r["gt_dist"] = Rr[s]["metrics"].get("gt_dist_traveled_m")
            if tok in T.index:
                t = T.loc[tok]
                r |= dict(grp=t.grp, v0=float(t.v0), turn=float(t.dyaw), lead_gap=float(t.lead_gap), clr=float(t["clr|P2H10-F-s0"]), bm=float(t["bm|P2H10-F-s0"]),
                          ol_ratio=float(t[f"arc4|{ARM}-s{i}"] / max(t[f"arc4|P2H10-F-s{i}"], 1e-3)), ol_arc_base=float(t[f"arc4|P2H10-F-s{i}"]))
            rows.append(r)
    P = pd.DataFrame(rows)
    P["dprog"], P["dscore"] = P.arm_prog - P.base_prog, P.arm_score - P.base_score
    slow = lambda x: (x > 0) & (x < 1)  # noqa: E731
    P["base_slow"], P["arm_slow"] = slow(P.base_score), slow(P.arm_score)
    P["to_slow"], P["from_slow"] = (P.base_score == 1) & P.arm_slow, P.base_slow & (P.arm_score == 1)
    P["new_zero"], P["rem_zero"] = (P.base_score > 0) & (P.arm_score == 0), (P.base_score == 0) & (P.arm_score > 0)
    P["vbin"] = np.select([P.v0 < 1, P.v0 < 3, P.v0 < 6, P.v0 < 10], ["v < 1", "1-3", "3-6", "6-10"], "> 10")
    P["tbin"] = np.select([P.turn < 10, P.turn <= 45], ["< 10 deg", "10-45 deg"], "> 45 deg")
    P["lbin"] = np.select([P.base_obs < 0.5, P.base_obs < 1.5, P.base_obs < 4], ["loop obj < 0.5 m", "0.5-1.5 m", "1.5-4 m"], ">= 4 m / none")
    return P, keep


def cmd_tables(a):
    import pandas as pd
    from jevdrive import stats
    P, _ = load()
    OUT.mkdir(parents=True, exist_ok=True)
    assert P.grp.notna().all(), f"{int(P.grp.isna().sum())} scenes without a navtest token"
    both = (P.base_score > 0) & (P.arm_score > 0)
    tot_dprog, tot_dscore, tot_dprog_b = P.dprog.sum(), P.dscore.sum(), P.dprog[both].sum()

    def line(split, name, m):
        q = P[m]
        qb = P[m & both]
        r = stats.paired(q.arm_prog.to_numpy(), q.base_prog.to_numpy(), groups=q.log.to_numpy())
        rb = stats.paired(qb.arm_prog.to_numpy(), qb.base_prog.to_numpy(), groups=qb.log.to_numpy()) if len(qb) > 5 else dict(mean=np.nan, lo=np.nan, hi=np.nan)
        return dict(split=split, subset=name, pairs=len(q), scenes=q.scene.nunique(), logs=q.log.nunique(), base_prog=q.base_prog.mean(), dprog=r["mean"], lo=r["lo"], hi=r["hi"],
                    share_dprog=q.dprog.sum() / tot_dprog, dprog_nz=rb["mean"], lo_nz=rb["lo"], hi_nz=rb["hi"], share_dprog_nz=qb.dprog.sum() / tot_dprog_b,
                    base_slow=int(q.base_slow.sum()), arm_slow=int(q.arm_slow.sum()), to_slow=int(q.to_slow.sum()), from_slow=int(q.from_slow.sum()),
                    base_zero=int((q.base_score == 0).sum()), arm_zero=int((q.arm_score == 0).sum()), dscore=q.dscore.mean(), share_dscore=q.dscore.sum() / tot_dscore,
                    dscore_prog_part=float(qb.dscore.sum() / len(P)), dscore_zero_part=float(q.dscore[~both[m]].sum() / len(P)),
                    arc0_ratio=q.arm_arc0.sum() / q.base_arc0.sum(), arcmean_ratio=qb.arm_arcmean.sum() / qb.base_arcmean.sum(), dist_ratio=qb.arm_dist.sum() / qb.base_dist.sum(),
                    ol_ratio=(q.ol_ratio * q.ol_arc_base).sum() / q.ol_arc_base.sum())
    one = np.ones(len(P), bool)
    L = [line("all", "all", one), line("all", "seed 0", (P.seed == 0).to_numpy()), line("all", "seed 1", (P.seed == 1).to_numpy()),
         line("all", "v >= 1", (P.v0 >= 1).to_numpy()), line("all", "launch (v0 < 1)", (P.v0 < 1).to_numpy())]
    for split, col, vals in (("group", "grp", GROUPS), ("speed", "vbin", ["v < 1", "1-3", "3-6", "6-10", "> 10"]), ("turn", "tbin", ["< 10 deg", "10-45 deg", "> 45 deg"]),
                             ("object in the base run", "lbin", ["loop obj < 0.5 m", "0.5-1.5 m", "1.5-4 m", ">= 4 m / none"])):
        L += [line(split, v, (P[col] == v).to_numpy()) for v in vals if (P[col] == v).sum() > 5]
    L += [line("group, v >= 1", g, ((P.grp == g) & (P.v0 >= 1)).to_numpy()) for g in GROUPS if ((P.grp == g) & (P.v0 >= 1)).sum() > 5]
    G = pd.DataFrame(L)
    G.to_csv(OUT / "cl_groups.csv", index=False, float_format="%.4f")
    P.to_csv(OUT / "cl_pairs.csv", index=False, float_format="%.4f")
    # --- the extra slow scenes: who crosses the 0.8 line
    b1 = P[(P.base_score == 1)]
    ts = P[P.to_slow]
    feat = lambda q: dict(n=len(q), base_prog_med=q.base_prog.median(), base_prog_lt_085=(q.base_prog < 0.85).mean(), dprog_med=q.dprog.median(), v0_med=q.v0.median(),  # noqa: E731
                          launch=(q.v0 < 1).mean(), turn_gt45=(q.turn > 45).mean(), **{f"grp {g}": (q.grp == g).mean() for g in GROUPS}, lead_gap_med=q.lead_gap[np.isfinite(q.lead_gap)].median(),
                          arc0_ratio=q.arm_arc0.sum() / q.base_arc0.sum(), ol_ratio=(q.ol_ratio * q.ol_arc_base).sum() / q.ol_arc_base.sum())
    S = pd.DataFrame([dict(set="base score 1, all", **feat(b1)), dict(set="1 -> slow", **feat(ts)), dict(set="slow -> 1", **feat(P[P.from_slow])),
                      dict(set="slow in both", **feat(P[P.base_slow & P.arm_slow])), dict(set="new zero", **feat(P[P.new_zero])), dict(set="removed zero", **feat(P[P.rem_zero]))])
    S.to_csv(OUT / "cl_slow_sets.csv", index=False, float_format="%.4f")
    # --- counterfactual: scale every base progress by the arm / base ratio of its own group, count the 0.8 crossings
    scale = {g: P[both & (P.grp == g)].arm_prog.sum() / P[both & (P.grp == g)].base_prog.sum() for g in GROUPS}
    glob_ratio = P[both].arm_prog.sum() / P[both].base_prog.sum()
    m1 = (P.base_score == 1)
    cf_g = int((P.base_prog[m1] * P.grp[m1].map(scale) < 0.8).sum())
    cf_u = int((P.base_prog[m1] * glob_ratio < 0.8).sum())
    # noise floor: the same transitions between the two seeds of the base (and of the arm)
    W = P.pivot(index="scene", columns="seed", values=["base_score", "arm_score", "base_prog", "arm_prog"])
    sl = lambda x: (x > 0) & (x < 1)  # noqa: E731
    floor = dict(base_s0_1_to_s1_slow=int(((W.base_score[0] == 1) & sl(W.base_score[1])).sum()), base_s0_slow_to_s1_1=int((sl(W.base_score[0]) & (W.base_score[1] == 1)).sum()),
                 base_seed_dprog=float((W.base_prog[1] - W.base_prog[0]).mean()), arm_seed_dprog=float((W.arm_prog[1] - W.arm_prog[0]).mean()))
    # --- trade or side effect: collision-relevant pairs against the rest
    col_rel = ((P.base_zero == "collision") | (P.arm_zero == "collision"))
    scn_col = P.scene.isin(P.scene[col_rel])                                         # scenes with a collision zero in any of the four runs
    risk = scn_col | (P.base_obs < 0.5)
    tr = {}
    for name, m in (("collision scenes (a collision zero in any of the 4 runs)", scn_col), ("collision scenes or base-run obstacle distance < 0.5 m", risk), ("the rest", ~risk)):
        q, qb = P[m], P[m & both]
        tr[name] = dict(pairs=len(q), scenes=q.scene.nunique(), share_dprog=float(q.dprog.sum() / tot_dprog), share_dprog_nz=float(qb.dprog.sum() / tot_dprog_b), dprog_nz=float(qb.dprog.mean()),
                        to_slow=int(q.to_slow.sum()), from_slow=int(q.from_slow.sum()), dscore_sum=float(q.dscore.sum()), removed_collisions=int(((q.base_zero == "collision") & (q.arm_score > 0)).sum()),
                        new_collisions=int(((q.arm_zero == "collision") & (q.base_score > 0)).sum()))
    rc = P[(P.base_zero == "collision") & (P.arm_score > 0)]
    other = P.set_index(["scene", "seed"]).base_prog
    rem = [dict(seed=int(r.seed), scene=r.scene[-16:], arm_score=r.arm_score, arm_prog=r.arm_prog, base_prog_at_collision=r.base_prog,
                base_prog_other_seed=float(other.get((r.scene, 1 - r.seed), np.nan)), base_score_other_seed=float(P[(P.scene == r.scene) & (P.seed == 1 - r.seed)].base_score.iloc[0]),
                grp=r.grp, v0=r.v0, turn=r.turn) for r in rc.itertuples()]
    pd.DataFrame(rem).to_csv(OUT / "cl_removed_collisions.csv", index=False, float_format="%.4f")
    nz = P[P.new_zero | P.rem_zero].copy()
    nz["change"] = np.where(nz.new_zero, "new", "removed")
    nz["scene"] = nz.scene.str[-16:]
    nz[["seed", "scene", "change", "base_zero", "arm_zero", "base_score", "arm_score", "base_prog", "arm_prog", "turn", "v0", "grp", "lead_gap", "clr", "bm", "base_obs", "arm_obs", "base_lane",
        "arm_lane", "ol_ratio"]].sort_values(["change", "arm_zero", "scene", "seed"]).to_csv(OUT / "cl_zero_flips.csv", index=False, float_format="%.3f")
    t45 = P[P.turn > 45]
    b45 = (t45.base_score > 0) & (t45.arm_score > 0)
    summ = dict(pairs=len(P), scenes=int(P.scene.nunique()), mean_dscore=float(P.dscore.mean()), dscore_from_zero_flips=float(P.dscore[~both].sum() / len(P)),
                dscore_from_progress=float(P.dscore[both].sum() / len(P)), mean_dprog=float(P.dprog.mean()), mean_dprog_both_nonzero=float(P.dprog[both].mean()),
                prog_ratio_both_nonzero=float(glob_ratio), group_prog_ratio=scale, to_slow=int(P.to_slow.sum()), from_slow=int(P.from_slow.sum()),
                slow_base=int(P.base_slow.sum()), slow_arm=int(P.arm_slow.sum()), slow_from_zero_base=int(((P.base_score == 0) & P.arm_slow).sum()), slow_to_zero_arm=int((P.base_slow & (P.arm_score == 0)).sum()),
                counterfactual_to_slow_group_scaling=cf_g, counterfactual_to_slow_uniform_scaling=cf_u, seed_floor=floor, trade=tr,
                k0_arc_vs_openloop=dict(corr_base=float(np.corrcoef(P.base_arc0, P.ol_arc_base)[0, 1]), ratio_base=float(P.base_arc0.sum() / P.ol_arc_base.sum())),
                gt45=dict(pairs=len(t45), dscore=float(t45.dscore.mean()), from_zero_flips=float(t45.dscore[~b45].sum() / len(t45)), from_progress=float(t45.dscore[b45].sum() / len(t45)),
                          dprog_nz=float(t45.dprog[b45].mean()), to_slow=int(t45.to_slow.sum()), from_slow=int(t45.from_slow.sum()), new_zero=int(t45.new_zero.sum()), rem_zero=int(t45.rem_zero.sum()),
                          arc0_ratio=float(t45.arm_arc0.sum() / t45.base_arc0.sum()), ol_ratio=float((t45.ol_ratio * t45.ol_arc_base).sum() / t45.ol_arc_base.sum())))
    (OUT / "cl_summary.json").write_text(json.dumps(summ, indent=1, default=float) + "\n")
    pd.set_option("display.width", 260)
    print(G[["split", "subset", "pairs", "base_prog", "dprog", "lo", "hi", "share_dprog", "dprog_nz", "share_dprog_nz", "base_slow", "arm_slow", "to_slow", "from_slow", "dscore", "arc0_ratio", "dist_ratio", "ol_ratio"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(S.T.to_string(float_format=lambda v: f"{v:.3f}"))
    print(json.dumps({k: v for k, v in summ.items() if k not in ("trade",)}, indent=None, default=float))
    print(json.dumps(tr, indent=1))


def margins(tr, sdf):
    """Trace (n, 4) t, x, y, yaw in the token frame, raster (128, 96) -> SDF at the 4 footprint corners (n, 4): front-left, front-right, rear-right, rear-left."""
    import sweep as SW
    cx, cy = SW.corners(*SW.ego_centre(tr[:, 1:4]), tr[:, 3], np.full(len(tr), SW.HALF_L), np.full(len(tr), SW.HALF_W))
    return SW.sdf_at(np.broadcast_to(sdf, (len(tr),) + sdf.shape), cx, cy)[0]


def cmd_raster(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    P, keep = load()
    nav, road = np.load(DATA / "runs/op_probe/labels/navtest.npz"), np.load(DATA / "runs/body1/labels_road/navtest.npz")
    pos = {t: i for i, t in enumerate(nav["tokens"].tolist())}
    assert (road["tokens"] == nav["tokens"]).all()
    fut = dict(zip(*(lambda t: (t["names"].tolist(), t["fut"]))(np.load(DATA / "runs/op_parity/cache/lb_navtest/tab.npz"))))
    FIG.mkdir(parents=True, exist_ok=True)
    rows = []
    sel = P[(P.new_zero & (P.arm_zero == "offroad")) | (P.rem_zero & (P.base_zero == "offroad"))]
    for r in sel.itertuples():
        i = pos[r.tok]
        sn, sr = nav["sdf"][i].astype(np.float32), road["sdf"][i].astype(np.float32)
        T = {}
        for side in ("base", "arm"):
            D, where = keep[(r.seed, side)]
            T[side] = trace(*where[r.scene])
        if T["base"] is None or T["arm"] is None:
            continue
        n = min(len(T["base"]), len(T["arm"]))
        b, m = T["base"][:n], T["arm"][:n]
        lat = -np.sin(b[:, 3]) * (m[:, 1] - b[:, 1]) + np.cos(b[:, 3]) * (m[:, 2] - b[:, 2])       # arm - base, + = left of the base
        j = int(np.abs(lat).argmax())
        o = dict(seed=r.seed, scene=r.scene[-16:], change="new" if r.new_zero else "removed", turn=r.turn, v0=r.v0, grp=r.grp, lat_end=float(lat[-1]), lat_max=float(lat[j]), t_lat_max=float(b[j, 0]),
                 drift="left" if lat[j] > 0 else "right")
        for side, tr in (("base", T["base"]), ("arm", T["arm"])):
            for rn, s in (("navsim", sn), ("road", sr)):
                mg = margins(tr, s)
                o |= {f"{side}_{rn}_left": float(mg[:, [0, 3]].min()), f"{side}_{rn}_right": float(mg[:, [1, 2]].min()), f"{side}_{rn}_t0": float(mg[0].min())}
        # could C have pushed? On the base's own trace, C's raster is inside the hinge margin (0.3 m) on the side AWAY from which the arm moves, while the NAVSIM raster is not
        away = "left" if o["drift"] == "right" else "right"
        o["c_only_edge_on_far_side"] = bool(o[f"base_road_{away}"] < 0.3 <= o[f"base_navsim_{away}"])
        o["c_edge_on_far_side"] = bool(o[f"base_road_{away}"] < 0.3)
        o["navsim_edge_on_far_side"] = bool(o[f"base_navsim_{away}"] < 0.3)
        o["starts_off_c"] = bool(o["base_road_t0"] < 0)
        rows.append(o)
        if r.new_zero:
            fig, ax = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True, sharey=True)
            ext = (-24, 24, -8, 56)
            for q, (s, nm) in zip(ax, ((sn, "NAVSIM raster (roadblock + intersection + car park)"), (sr, "item C raster (road-and-lane, no car park)"))):
                q.imshow(s, origin="lower", extent=ext, cmap="Greys_r", vmin=-3, vmax=3, alpha=0.55)
                q.contour(np.linspace(-23.75, 23.75, 96), np.linspace(-7.75, 55.75, 128), s, levels=[0.0, 0.3], colors=["k", "tab:red"], linewidths=[1.2, 0.8])
                f = fut.get(r.tok)
                q.plot(f[:, 1], f[:, 0], "k--", lw=1, label="logged 4 s")
                q.plot(b[:, 2], b[:, 1], color="tab:orange", lw=2, label="P2H10-F run")
                q.plot(m[:, 2], m[:, 1], color="tab:blue", lw=2, label="P2H10B-F run")
                q.set_title(nm, fontsize=9)
                q.set_xlim(-14, 14)
                q.set_ylim(-6, max(20.0, float(max(b[:, 1].max(), m[:, 1].max())) + 6))
                q.set_xlabel("y (m), left of the car is left")
            ax[0].invert_xaxis()
            ax[0].set_ylabel("x forward (m)")
            ax[0].legend(fontsize=8, loc="lower left")
            fig.suptitle(f"new offroad zero {r.scene[-16:]} seed {r.seed}: arm drifts {o['drift']} by {abs(o['lat_max']):.1f} m; black = raster edge, red = hinge margin 0.3 m", fontsize=9)
            fig.tight_layout()
            fig.savefig(FIG / f"offroad_{r.scene[-8:]}_s{r.seed}.png", dpi=110)
            plt.close(fig)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "cl_offroad_raster.csv", index=False, float_format="%.3f")
    print(D.to_string(index=False, float_format=lambda v: f"{v:.2f}"))


def cmd_figs(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    FIG.mkdir(parents=True, exist_ok=True)
    col = dict(zip(GROUPS, ["tab:red", "tab:purple", "tab:orange", "tab:olive", "tab:blue"]))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for q, st in zip(ax, ("navtest", "hold")):
        f = OUT / f"ol_{st}_arc.csv"
        if not f.exists():
            continue
        D = pd.read_csv(f)
        D = D[D.split == "group x speed"]
        vb = ["v < 1", "1-3", "3-6", "6-10", "> 10"]
        for gi, g in enumerate(GROUPS):
            for pi, (pn, mk) in enumerate((("s0", "o"), ("s1", "s"))):
                d = D[(D.pair == pn) & D.subset.str.startswith(g + " |")].set_index(D[(D.pair == pn) & D.subset.str.startswith(g + " |")].subset.str.split(" | ", regex=False).str[1]).reindex(vb)
                x = np.arange(5) + (gi - 2) * 0.15 + (pi - 0.5) * 0.05
                q.errorbar(x, d.ratio, yerr=[d.ratio - d.lo, d.hi - d.ratio], fmt=mk, ms=4, color=col[g], lw=0.8, label=g if pi == 0 else None)
        d = D[D.pair == "seed"]
        q.axhspan(float(d.ratio.quantile(0.1)), float(d.ratio.quantile(0.9)), color="0.85", zorder=0, label="base seed 1 / seed 0 (10-90 % of the cells)")
        q.axhline(1, color="k", lw=0.6)
        q.set_xticks(range(5), vb)
        q.set_xlabel("speed at t0 (m/s)")
        q.set_title(f"{st}: 4 s arc length of the own plan, P2H10B-F / P2H10-F", fontsize=10)
    ax[0].set_ylabel("arc ratio (95 % CI by log)")
    ax[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "ol_arc.png", dpi=120)
    plt.close(fig)
    f = OUT / "cl_pairs.csv"
    if f.exists():
        P = pd.read_csv(f)
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
        nz = (P.base_score > 0) & (P.arm_score > 0)
        for g in GROUPS[::-1]:
            m = nz & (P.grp == g)
            ax[0].scatter(P.base_prog[m], P.arm_prog[m], s=6, color=col[g], alpha=0.6, label=f"{g} ({int(m.sum())})")
        ax[0].plot([0, 1], [0, 1], "k-", lw=0.6)
        ax[0].axhline(0.8, color="k", ls=":", lw=0.8)
        ax[0].axvline(0.8, color="k", ls=":", lw=0.8)
        ax[0].set_xlim(0.4, 1.01), ax[0].set_ylim(0.4, 1.01)
        ax[0].set_xlabel("progress, P2H10-F"), ax[0].set_ylabel("progress, P2H10B-F"), ax[0].legend(fontsize=7)
        ax[0].set_title("pairs with no zero on either side; dotted = the 0.8 line of the score", fontsize=9)
        bins = np.linspace(0.6, 1.0, 41)
        ax[1].hist(P.base_prog[P.base_score == 1], bins, color="0.75", label="base score 1")
        ax[1].hist(P.base_prog[P.to_slow], bins, color="tab:red", label="of them: slow with the arm")
        ax[1].axvline(0.8, color="k", ls=":", lw=0.8)
        ax[1].set_xlabel("progress of the base run"), ax[1].set_ylabel("pairs"), ax[1].legend(fontsize=8)
        ax[1].set_title("the extra slow scenes sit just above the 0.8 line in the base run", fontsize=9)
        G = pd.read_csv(OUT / "cl_groups.csv")
        d = G[G.split == "group"].set_index("subset").reindex(GROUPS)
        x = np.arange(len(d))
        ax[2].bar(x, d.dprog_nz, yerr=[d.dprog_nz - d.lo_nz, d.hi_nz - d.dprog_nz], color=[col[g] for g in GROUPS], capsize=3)
        for xi, (s, n) in enumerate(zip(d.share_dprog_nz, d.pairs)):
            ax[2].text(xi, 0.001, f"{s:.0%} of the loss\n{n} pairs", ha="center", va="bottom", fontsize=7)
        ax[2].axhline(0, color="k", lw=0.6)
        ax[2].set_xticks(x, GROUPS)
        ax[2].set_ylabel("progress, arm - base (pairs with no zero; 95 % CI by log)")
        ax[2].set_title("closed-loop progress difference by the scene's group at its token", fontsize=9)
        fig.tight_layout()
        fig.savefig(FIG / "cl_progress.png", dpi=120)
        plt.close(fig)
    f = OUT / "grad_terms.csv"
    if f.exists():
        D = pd.read_csv(f)
        D = D[(D.v == "all") & ~D.term.str.endswith("navsim")]
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        terms = list(dict.fromkeys(D.term))
        cks = list(dict.fromkeys(D.ckpt))
        cc = {c: ("tab:orange" if "B-F" not in c else "tab:blue") for c in cks}
        for i, (c, d) in enumerate(D.groupby("ckpt", sort=False)):
            d = d.set_index("term").reindex(terms)
            x = np.arange(len(terms)) + (i - (len(cks) - 1) / 2) * 0.18
            ax[0].bar(x, d.along_share, 0.17, color=cc[c], alpha=0.5 + 0.5 * (i % 2), label=c)
            ax[1].bar(x, d.pull, 0.17, color=cc[c], alpha=0.5 + 0.5 * (i % 2))
            ax[2].bar(x, d.retime50, 0.17, color=cc[c], alpha=0.5 + 0.5 * (i % 2))
            ax[2].plot(x, d["shift"], "k_", ms=9)
        for q, t in zip(ax, ("share of the position gradient along the path (positive rows)", "arc-length pull per training batch (m per unit step; < 0 shortens)",
                             "positive rows resolved by half arc (bars) / by a +-1 m ramp (marks)")):
            q.set_xticks(range(len(terms)), terms, rotation=30, ha="right", fontsize=8)
            q.set_title(t, fontsize=9)
        ax[1].axhline(0, color="k", lw=0.6)
        ax[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG / "grad_terms.png", dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tables", "raster", "figs"])
    ap.add_argument("--arm", default=ARM, help="arm tag without the seed; its runs in --man, its open-loop plans in --ol")
    ap.add_argument("--man", default=MAN, help="manifest {<arm>-s0: [run dirs], <arm>-s1: [...]} under $DATA_DIR")
    ap.add_argument("--ol", default=OLN, help="prog_ol.py parquet stem under runs/body1/prog")
    ap.add_argument("--out", default="", help="table / figure sub-directory name instead of prog (results/<out>, figs/<out>)")
    a = ap.parse_args()
    ARM, MAN, OLN = a.arm, a.man, a.ol
    if a.out:
        OUT, FIG = REPO / "experiments/body1/results" / a.out, REPO / "experiments/body1/figs" / a.out
    from jevdrive.run import Run
    with Run("body1", f"prog-cl-{a.cmd}", config=vars(a)) as run:
        {"tables": cmd_tables, "raster": cmd_raster, "figs": cmd_figs}[a.cmd](a)
