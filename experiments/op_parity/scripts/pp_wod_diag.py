"""Why P2H loses on WOD-E2E val (results/wod_p2h_diag.md): input interventions on the 20 ego features, bias decomposition, recipe arms,
frame source (protocol W on WOD), and an output decomposition of the plans. Inference only.

  bias    (op-train env, CPU) one intent_bias npz per (arm, variant) -> $DATA_DIR/runs/op_parity/wod/diag/bias-<arm>_<var>.npz
          variants: see VARS; the main mapping is pp_wod.wod_ego (decision 155)
  report  (jevdrive env, CPU) preds/op_cinque_dx-<arm>_<var>[_W] vs shipped (preds/op_cinque) and vs the main arm: RFS (479 rater
          frames), ADE@3s / @5s (1 437 frames), seed means, sequence bootstraps, recovery fractions, output decomposition (retimed
          plans), strata -> results/wod_p2h_diag/{arms,strata,decomp}.csv + printed tables

Serving: scripts/wod_zeroshot_openpilot.py --onnx pp-<arm>.onnx --tag dx-<arm>_<var> ... --bias <npz> ... (one warm-up, the last step per bias),
--frames warp for the frame-source arms (tags dx-<arm>_<var>_W).
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_pl.Path(__file__).parent)]
import argparse, json  # noqa: E401,E402

import numpy as np  # noqa: E402

# ego feature layout (parity_adapter.ego_features): 0 present, 1:4 cmd [L, S, R], 4 vx/10, 5 vy/10, 6 ax/3, 7 ay/3, 8:20 poses 4 x (x/10, y/10, yaw)
PX, PY, PYAW = slice(8, 20, 3), slice(9, 20, 3), slice(10, 20, 3)
T_HIST = np.array([-1.5, -1.0, -0.5, 0.0])
VARS = ("main", "zero", "cmd0", "acc0", "accnav", "velgiven", "vx110", "yaw0", "yawvel", "posecv", "cv", "biasmean", "biasresid", "biasnav", "biasdenav")   # biasdenav: post hoc (added after the first read)
B = 2000


def ddir():
    from jevdrive.common import data_dir
    d = data_dir() / "runs/op_parity/wod/diag"
    d.mkdir(parents=True, exist_ok=True)
    return d


def yaw_from_vel(past):
    """Key yaws from the given velocity direction (op_interp.track_wod's rule), relative to t0."""
    v = past[:, [9, 11, 13, 15], 2:4].astype(np.float64)
    out = np.zeros(v.shape[:2])
    for i in range(len(v)):
        mov = np.linalg.norm(v[i], axis=1) >= 1.0
        if mov.any():
            y = np.unwrap(np.arctan2(v[i, :, 1], v[i, :, 0]))
            mi = np.flatnonzero(mov)
            y = y[mi[np.abs(np.arange(4)[:, None] - mi[None]).argmin(1)]]
            out[i] = y - y[-1]
    return out


def edit(e, var, past, nav):
    """Main-mapping ego features e (n, 20) -> the variant's features (bias-level variants return e unchanged)."""
    e = e.copy()
    if var == "cmd0":
        e[:, 1:4] = 0
    elif var == "acc0":
        e[:, 6:8] = 0
    elif var == "accnav":                                          # ax, ay z-scored onto the navtest moments
        mw, sw, mn, sn = e[:, 6:8].mean(0), e[:, 6:8].std(0), nav[:, 6:8].mean(0), nav[:, 6:8].std(0)
        e[:, 6:8] = mn + (e[:, 6:8] - mw) * sn / sw
    elif var == "velgiven":                                        # the given velocity vector of the last past state
        e[:, 4:6] = past[:, -1, 2:4] / 10.0
    elif var == "vx110":
        e[:, 4] *= 1.1
    elif var == "yaw0":
        e[:, PYAW] = 0
    elif var == "yawvel":
        e[:, PYAW] = yaw_from_vel(past)
    elif var in ("posecv", "cv"):                                  # straight constant-speed history at the input speed
        e[:, PX] = e[:, 4:5] * T_HIST[None]
        e[:, PY] = 0
        e[:, PYAW] = 0
        if var == "cv":
            e[:, 5:8] = 0
    return e


def cmd_bias(a):
    import torch
    import pp_hugsim as H
    from jevdrive import wod_zeroshot as Z
    from pp_wod import wod_ego
    S = Z.load_sets()
    names = np.concatenate([S[k]["name"].astype(str) for k in ("rater", "extra")])
    past = np.concatenate([S[k]["past"] for k in ("rater", "extra")]).astype(np.float64)
    intent = np.concatenate([S[k]["intent"] for k in ("rater", "extra")])
    ego, _ = wod_ego(past, intent)
    from jevdrive.common import data_dir
    nav = np.load(data_dir() / "runs/op_parity/cache/lb_navtest/tab.npz")["ego"].astype(np.float32)
    sf = ddir() / "bias_stats.json"
    stats = json.loads(sf.read_text()) if sf.exists() else {}
    stats |= {"wod_mean": ego.mean(0).round(4).tolist(), "wod_std": ego.std(0).round(4).tolist(),
             "nav_mean": nav.mean(0).round(4).tolist(), "nav_std": nav.std(0).round(4).tolist()}
    for arm in a.arms:
        if arm.startswith("P1"):
            m = None
        else:
            m = H.pmodel(arm, torch.device("cpu"))
            assert m.adapter is not None and not m.adapter.use_side, arm

        def run(e):
            if m is None:
                return np.zeros((len(e), 32, 512), np.float32)
            with torch.no_grad():
                return np.concatenate([m.adapter(torch.from_numpy(np.ascontiguousarray(e[i:i + 512], np.float32)), None, None).numpy()
                                       for i in range(0, len(e), 512)])
        main = run(ego)
        for var in a.vars:
            e = edit(ego, var, past, nav)
            if var == "main":
                b = main
            elif var == "zero":
                b = np.zeros_like(main)
            elif var == "biasmean":
                b = np.broadcast_to(main.mean(0), main.shape)
            elif var == "biasresid":
                b = main - main.mean(0)
            elif var == "biasnav":
                b = np.broadcast_to(run(nav).mean(0), main.shape)
            elif var == "biasdenav":                               # main minus the navtest-mean bias: a fix that needs no WOD statistics
                b = main - run(nav).mean(0)
            else:
                b = run(e)
            b = np.ascontiguousarray(b).astype(np.float16)
            rms = float(np.sqrt(np.mean(b.astype(np.float32) ** 2)))
            dm = float(np.sqrt(np.mean((b.astype(np.float32) - main) ** 2)))
            np.savez(ddir() / f"bias-{arm}_{var}.npz", names=names, bias=b, ego=e)
            stats[f"{arm}_{var}"] = {"rms": rms, "rms_diff_main": dm}
            print(f"{arm}_{var}: rms {rms:.4f}, rms diff to main {dm:.4f}", flush=True)
    sf.write_text(json.dumps(stats, indent=1))


# ---------------------------------------------------------------- report
def retime(path, prof):
    """path (n, 20, 2) plans; prof (n, 20, 2) plans whose arc length at each waypoint is imposed on `path` -> (n, 20, 2)."""
    out = np.empty_like(path)
    for i in range(len(path)):
        p = np.vstack([[0, 0], path[i]])
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        q = np.vstack([[0, 0], prof[i]])
        sq = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))])[1:]
        mv = np.flatnonzero(seg > 1e-3)
        d = (p[mv[-1] + 1] - p[mv[-1]]) / seg[mv[-1]] if len(mv) else np.array([1.0, 0.0])
        keep = np.concatenate([[True], seg > 1e-6])               # strictly increasing arc for interp
        sx, px = s[keep], p[keep]
        x = np.interp(sq, sx, px[:, 0]) + np.maximum(sq - sx[-1], 0) * d[0]
        y = np.interp(sq, sx, px[:, 1]) + np.maximum(sq - sx[-1], 0) * d[1]
        out[i] = np.stack([x, y], -1)
    return out


def cmd_report(a):
    import pandas as pd
    from jevdrive import waymo as W
    from jevdrive import wod_zeroshot as Z
    from pp_wod import load_preds
    S = Z.load_sets()
    r, x = S["rater"], S["extra"]
    nr = len(r["name"])
    names = np.concatenate([r["name"], x["name"]]).astype(str)
    seq = np.concatenate([r["sequence"], x["sequence"]]).astype(str)
    fut = np.concatenate([r["future"], x["future"]])[..., :2].astype(np.float64)
    cl = r["cluster"].astype(str)
    traj, sc, v0 = r["traj"].astype(np.float64), r["scores"].astype(np.float64), W.init_speed(r["past"])
    ccode, cu = pd.factorize(pd.Series(cl))
    nc = len(cu)

    def rfs_agg(f, i):                                            # rfs_by_cluster on rows i (bincount version)
        s = np.bincount(ccode[i], f[i], nc)
        c = np.bincount(ccode[i], None, nc)
        ok = c > 0
        return float((s[ok] / c[ok]).mean())

    out = _R / "experiments/op_parity/results/wod_p2h_diag"
    out.mkdir(exist_ok=True)
    tags = [t for t in a.arms]
    avail = [t for t in tags if Z.root("preds", f"op_cinque_{t}").exists() and len(list(Z.root("preds", f"op_cinque_{t}").glob("*.npz"))) >= len(names)]
    miss = sorted(set(tags) - set(avail))
    if miss:
        print("missing (skipped):", miss)
    P = {"shipped": load_preds("shipped", names), "stored_s0": load_preds("P2H10-F-s0", names), "stored_s1": load_preds("P2H10-F-s1", names)}
    for t in avail:
        P[t] = load_preds(t, names)
    rfs = {t: np.asarray(W.rater_feedback_score(p[:nr], traj, sc, v0), float) for t, p in P.items()}
    err = {t: np.linalg.norm(p - fut, axis=-1) for t, p in P.items()}
    ade3 = {t: e[:, :12].mean(1) for t, e in err.items()}
    ade5 = {t: e.mean(1) for t, e in err.items()}
    # equivalence: multi-bias path vs the stored single-bias runs of decision 155
    eqv = {}
    for s in ("s0", "s1"):
        t = f"dx-P2H10-F-{s}_main"
        if t in P:
            d = np.linalg.norm(P[t] - P[f"stored_{s}"], axis=-1)
            eqv[t] = {"mean": float(d.mean()), "p99": float(np.percentile(d, 99)), "max": float(d.max())}
    print("equivalence vs stored:", json.dumps(eqv))
    # output decomposition (both seeds): P2H path at shipped speed and shipped path at P2H speed
    for s in ("s0", "s1"):
        t = f"dx-P2H10-F-{s}_main"
        if t in P:
            for nm, pp in ((f"dx-P2H10-F-{s}_O1", retime(P[t], P["shipped"])), (f"dx-P2H10-F-{s}_O2", retime(P["shipped"], P[t]))):
                P[nm] = pp
                rfs[nm] = np.asarray(W.rater_feedback_score(pp[:nr], traj, sc, v0), float)
                e = np.linalg.norm(pp - fut, axis=-1)
                ade3[nm], ade5[nm] = e[:, :12].mean(1), e.mean(1)
    # shipped retimed onto itself: the retime operator's own error (should be ~0)
    pp = retime(P["shipped"], P["shipped"])
    print("retime self-check max |d| m:", float(np.abs(pp - P["shipped"]).max()))

    # seed means: arm names with -s0 / -s1 merged to -sm
    def seedmean(base):
        a0, a1 = base.replace("-sX", "-s0"), base.replace("-sX", "-s1")
        if a0 in rfs and a1 in rfs:
            k = base.replace("-sX", "-sm")
            rfs[k], ade3[k], ade5[k] = ((d[a0] + d[a1]) / 2 for d in (rfs, ade3, ade5))
            return k
        return a0 if a0 in rfs else None
    bases = sorted({t.replace("-s0", "-sX").replace("-s1", "-sX") for t in list(rfs) if t not in ("shipped",) and not t.startswith("stored")})
    sm = [k for k in (seedmean(b) for b in bases) if k]
    codes_r, ur = pd.factorize(pd.Series(seq[:nr]))
    codes_a, ua = pd.factorize(pd.Series(seq))
    idx_r = [np.flatnonzero(codes_r == k) for k in range(len(ur))]
    idx_a = [np.flatnonzero(codes_a == k) for k in range(len(ua))]
    rng = np.random.default_rng(0)
    draws_r = [np.concatenate([idx_r[k] for k in rng.integers(len(ur), size=len(ur))]) for _ in range(B)]
    draws_a = [np.concatenate([idx_a[k] for k in rng.integers(len(ua), size=len(ua))]) for _ in range(B)]
    allr = np.arange(nr)

    def ci_rfs(fa, fb, rows=None):
        d = []
        for i in draws_r:
            if rows is not None:
                i = i[rows[i]]
            d.append(rfs_agg(fa, i) - rfs_agg(fb, i))
        return np.percentile(d, [2.5, 97.5])

    def ci_mean(da, db):
        dd = da - db
        return np.percentile([dd[i].mean() for i in draws_a], [2.5, 97.5])

    def main_of(k):
        if k.endswith("_W"):
            return k.split("_")[0] + "_main_W"
        return k.split("_")[0] + "_main"
    rows = []
    for k in sm:
        mk = main_of(k)
        row = {"arm": k, "RFS": rfs_agg(rfs[k], allr)}
        row["dRFS"] = row["RFS"] - rfs_agg(rfs["shipped"], allr)
        row["dRFS_lo"], row["dRFS_hi"] = ci_rfs(rfs[k], rfs["shipped"])
        row["ADE3"], row["ADE5"] = ade3[k].mean(), ade5[k].mean()
        row["dADE3"] = ade3[k].mean() - ade3["shipped"].mean()
        row["dADE3_lo"], row["dADE3_hi"] = ci_mean(ade3[k], ade3["shipped"])
        row["dADE5"] = ade5[k].mean() - ade5["shipped"].mean()
        if mk in rfs and mk != k:
            row["vs_main_RFS"] = row["RFS"] - rfs_agg(rfs[mk], allr)
            row["vs_main_lo"], row["vs_main_hi"] = ci_rfs(rfs[k], rfs[mk])
            row["vs_main_ADE3"] = ade3[k].mean() - ade3[mk].mean()
            row["vs_main_ADE3_lo"], row["vs_main_ADE3_hi"] = ci_mean(ade3[k], ade3[mk])
            dm = rfs_agg(rfs[mk], allr) - rfs_agg(rfs["shipped"], allr)
            row["recovery_RFS"] = row["vs_main_RFS"] / -dm
            row["recovery_ADE3"] = -row["vs_main_ADE3"] / (ade3[mk].mean() - ade3["shipped"].mean())
        rows.append(row)
        print(f"{k:32s} RFS {row['RFS']:.3f} d {row['dRFS']:+.3f} [{row['dRFS_lo']:+.3f}, {row['dRFS_hi']:+.3f}]  ADE3 d {row['dADE3']:+.3f}"
              + (f"  vs main {row['vs_main_RFS']:+.3f} [{row['vs_main_lo']:+.3f}, {row['vs_main_hi']:+.3f}] rec {row['recovery_RFS']:+.2f}"
                 f" / ADE3 {row['vs_main_ADE3']:+.3f} rec {row['recovery_ADE3']:+.2f}" if "vs_main_RFS" in row else ""), flush=True)
    # frame-source interaction: (P2H - shipped)@W - (P2H - shipped)@real
    inter = {}
    for s in ("sm", "s0", "s1"):
        kw, kr = f"dx-P2H10-F-{s}_main_W", f"dx-P2H10-F-{s}_main"
        if kw in rfs and kr in rfs and "dx-shipped_zero_W" in rfs:
            vals = [(rfs_agg(rfs[kw], i) - rfs_agg(rfs["dx-shipped_zero_W"], i)) - (rfs_agg(rfs[kr], i) - rfs_agg(rfs["shipped"], i)) for i in draws_r]
            inter[s] = {"d_W": rfs_agg(rfs[kw], allr) - rfs_agg(rfs["dx-shipped_zero_W"], allr), "d_W_ci": ci_rfs(rfs[kw], rfs["dx-shipped_zero_W"]).tolist(),
                        "d_real": rfs_agg(rfs[kr], allr) - rfs_agg(rfs["shipped"], allr),
                        "I": float((rfs_agg(rfs[kw], allr) - rfs_agg(rfs["dx-shipped_zero_W"], allr)) - (rfs_agg(rfs[kr], allr) - rfs_agg(rfs["shipped"], allr))),
                        "I_ci": np.percentile(vals, [2.5, 97.5]).tolist(),
                        "shipped_W_minus_real": rfs_agg(rfs["dx-shipped_zero_W"], allr) - rfs_agg(rfs["shipped"], allr),
                        "dADE3_W": float(ade3[kw].mean() - ade3["dx-shipped_zero_W"].mean()), "dADE3_W_ci": ci_mean(ade3[kw], ade3["dx-shipped_zero_W"]).tolist()}
    print("frame source:", json.dumps(inter, indent=1))
    pd.DataFrame(rows).to_csv(out / "arms.csv", index=False)

    # ---- plan geometry and strata (seed-mean main arm)
    km = "dx-P2H10-F-sm_main"
    if km in rfs:
        lead = np.array([float(np.asarray(np.load(Z.root("preds", "op_cinque") / f"{n}.npz")["lead_prob"]).reshape(-1)[0]) for n in names[:nr]])
        fr = fut[:nr]
        logd = np.linalg.norm(fr[:, -1], axis=-1)
        intent = r["intent"]
        disp = lambda p, j: np.linalg.norm(p[:, j], axis=-1)       # noqa: E731
        st = {"all": np.ones(nr, bool), "stopped v<0.5": v0 < 0.5, "launch v<2 & log5s>5m": (v0 < 2) & (logd > 5),
              "slow 0.5-5": (v0 >= 0.5) & (v0 < 5), "mid 5-12": (v0 >= 5) & (v0 < 12), "fast >=12": v0 >= 12,
              "turn intent L/R": intent >= 2, "straight intent": intent == 1, "lead_prob>0.5 (shipped)": lead > 0.5,
              "lead_prob<=0.5": lead <= 0.5}
        lum = pd.read_csv(_R / "experiments/leaderboard_audit/results/night_gap/seq_lum.csv").set_index("sequence").l.reindex(seq[:nr]).to_numpy()
        st["night (luma < 50)"], st["day (luma >= 120)"] = lum < 50, lum >= 120
        for c in cu:
            st["cluster " + c] = cl == c
        srows = []
        p0, p1 = P["dx-P2H10-F-s0_main"], P["dx-P2H10-F-s1_main"]
        ps = P["shipped"]
        for nm, msk in st.items():
            i = np.flatnonzero(msk)
            if len(i) < 8:
                continue
            row = {"stratum": nm, "n": len(i), "RFS shipped": rfs_agg(rfs["shipped"], i), "RFS P2H": rfs_agg(rfs[km], i)}
            row["dRFS"] = row["RFS P2H"] - row["RFS shipped"]
            row["dRFS_lo"], row["dRFS_hi"] = ci_rfs(rfs[km], rfs["shipped"], msk)
            for o in ("O1", "zero", "biasresid", "biasmean", "biasdenav"):
                if f"dx-P2H10-F-sm_{o}" in rfs:
                    row[f"dRFS {o}"] = rfs_agg(rfs[f"dx-P2H10-F-sm_{o}"], i) - row["RFS shipped"]
            if "dx-P2H10-F-sm_main_W" in rfs and "dx-shipped_zero_W" in rfs:
                row["dRFS @W (vs shipped@W)"] = rfs_agg(rfs["dx-P2H10-F-sm_main_W"], i) - rfs_agg(rfs["dx-shipped_zero_W"], i)
            # longitudinal / lateral difference of the P2H plan (seed mean of the two) from shipped at 3 s, in shipped's heading frame
            for j, tt in ((11, "3s"), (19, "5s")):
                dp = (p0[i, j] + p1[i, j]) / 2 - ps[i, j]
                hd = ps[i, j] - ps[i, j - 1]
                hd = hd / np.maximum(np.linalg.norm(hd, axis=-1, keepdims=True), 1e-6)
                hd[np.linalg.norm(ps[i, j] - ps[i, j - 1], axis=-1) < 1e-3] = [1, 0]
                row[f"lon {tt} (m)"] = float(np.mean((dp * hd).sum(-1)))
                row[f"|lat| {tt} (m)"] = float(np.mean(np.abs(dp[:, 0] * -hd[:, 1] + dp[:, 1] * hd[:, 0])))
            mvv = logd[i] > 2
            row["plan5s/log5s P2H"] = float(np.median(((disp(p0, 19) + disp(p1, 19)) / 2)[i][mvv] / logd[i][mvv])) if mvv.sum() >= 5 else np.nan
            row["plan5s/log5s shipped"] = float(np.median(disp(ps, 19)[i][mvv] / logd[i][mvv])) if mvv.sum() >= 5 else np.nan
            srows.append(row)
        sdf = pd.DataFrame(srows)
        sdf.to_csv(out / "strata.csv", index=False)
        pd.set_option("display.width", 250, "display.max_columns", 30)
        print(sdf.to_string(float_format=lambda v: f"{v:.3f}"))
    (out / "meta.json").write_text(json.dumps({"equivalence": eqv, "frame_source": inter, "B": B, "missing": miss}, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("bias")
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--vars", nargs="+", default=list(VARS))
    p = sp.add_parser("report")
    p.add_argument("--arms", nargs="+", required=True, help="prediction tags dx-<arm>_<var>[_W]")
    a = ap.parse_args()
    {"bias": cmd_bias, "report": cmd_report}[a.cmd](a)
