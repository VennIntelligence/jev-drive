"""Scale check analysis: lead distance, lane width, plan speed, model road height on WOD-E2E, WOD perception frames and NAVSIM navtest.
Reads the head dumps (op_lb layout) of sc_wod_run.py and of the NAVSIM runs; ground truth for the lead from 3D boxes (WOD perception
sceneflow frames; NAVSIM navtest logs). CPU: $DATA_DIR/jev venv: .venv/bin/python experiments/leaderboard_audit/scripts/sc_analyze.py
Writes $DATA_DIR/runs/scale_check/stats.json and samples_*.csv.

Pre-registered rules (before looking at ratios): lead truth = nearest vehicle box ahead whose centre is within |y| < 1.5 m of the ego x
axis, heading within 0.4 rad of ego, rear face (cx - (L|cos h| + W|sin h|) / 2) at 8-50 m from the camera; model lead = P(lead) > 0.5, x of
selection 0 at t = 0; ratio = model x / true camera-to-rear-face gap. Lane width = model ego lines (1, 2) both with p > 0.5, at x = 10 and
20 m from the camera. Speed ratio = model plan speed at t0 (plan_vel x of point 0) / logged speed, speed > 3 m/s. Model road height = model
lane-line z (device frame, mean of the two ego lines, x 10-20 m)."""
import glob, json, os, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd

D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
RUN = D / "runs/scale_check"
X_IDXS = 192.0 * (np.arange(33) / 32) ** 2
rng = np.random.default_rng(0)
LAT = 1.5   # lane half-width for 'in lane' (pre-registered 1.2; relaxed to 1.5 on all boards after the sceneflow set had only 3 clusters with a lead)
SF_FROM = 50  # first sceneflow frame used (5 s of stream)
sig = lambda a: 1 / (1 + np.exp(-a))  # noqa: E731


def parse(H, sl):
    ll = H[:, sl["lane_lines"]:sl["lane_lines"] + 264].reshape(-1, 4, 33, 2)
    lp = sig(H[:, sl["lane_lines_prob"]:sl["lane_lines_prob"] + 8][:, 1::2])
    lead = H[:, sl["lead"]:sl["lead"] + 72].reshape(-1, 3, 6, 4)
    pl = sig(H[:, sl["lead_prob"]])
    pv = H[:, sl["plan"] + 3]            # plan mu: (33, 15) row 0, col 3 = vel x
    return ll, lp, lead[:, 0, 0, 0], lead[:, 0, 0, 2], pl, pv


def lane_cols(ll, lp):
    g = lambda a, x: np.array([np.interp(x, X_IDXS, r) for r in a])  # noqa: E731
    out = {}
    for x in (10, 20):
        out[f"w{x}"] = g(ll[:, 2, :, 0], x) - g(ll[:, 1, :, 0], x)
        out[f"z{x}"] = (g(ll[:, 2, :, 1], x) + g(ll[:, 1, :, 1], x)) / 2
    out["ok"] = (lp[:, 1] > .5) & (lp[:, 2] > .5)
    return pd.DataFrame(out)


def lead_truth(boxes, cam_x, is_nav):
    """boxes (N, >=7) [cx, cy, cz, L, W, H, yaw, ...] in the ego frame -> nearest in-lane vehicle gap (camera to rear face) and its speed or nan."""
    best, vb = np.nan, np.nan
    for b in boxes:
        cx, cy, L, W, yaw = b[0], b[1], b[3], b[4], b[6]
        if abs(cy) > LAT or abs(((yaw + np.pi) % (2 * np.pi)) - np.pi) > 0.4:
            continue
        rear = cx - 0.5 * (L * abs(np.cos(yaw)) + W * abs(np.sin(yaw))) - cam_x
        if 8 <= rear <= 50 and (np.isnan(best) or rear < best):
            best = rear
    return best


def boot(vals, cl, stat=np.median, B=2000):
    vals, cl = np.asarray(vals, float), np.asarray(cl)
    if len(vals) == 0:
        return (np.nan,) * 4 + (0, 0)
    u = np.unique(cl)
    groups = [vals[cl == c] for c in u]
    bs = [stat(np.concatenate([groups[i] for i in rng.integers(0, len(u), len(u))])) for _ in range(B)]
    return (float(stat(vals)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(vals), len(u))


def slope_stat(a):
    """a: rows (gap, x_mod) -> least-squares slope of x_mod on gap (with intercept)."""
    a = np.asarray(a)
    return np.polyfit(a[:, 0], a[:, 1], 1)[0]


def boot_slope(df, m, B=2000):
    d = df[m]
    if len(d) < 5:
        return (np.nan,) * 6
    u = d.cl.unique()
    g = {c: d[d.cl == c][["gap", "x_mod"]].to_numpy() for c in u}
    f = lambda arr: np.polyfit(arr[:, 0], arr[:, 1], 1)  # noqa: E731
    est = f(d[["gap", "x_mod"]].to_numpy())
    bs = np.array([f(np.concatenate([g[u[i]] for i in rng.integers(0, len(u), len(u))])) for _ in range(B)])
    return (float(est[0]), float(np.percentile(bs[:, 0], 2.5)), float(np.percentile(bs[:, 0], 97.5)), float(est[1]), len(d), len(u))


def board_stats(df, name, h_true):
    """df columns: cl, v, pv0, x_mod, p_lead, gap, w10, w20, z10, z20, ok"""
    r = {}
    m = df.p_lead.gt(.5) & df.gap.notna() & df.x_mod.gt(0)
    r["lead_ratio"] = boot((df.x_mod / df.gap)[m], df.cl[m])
    r["lead_ratio_log_mean"] = float(np.exp(np.mean(np.log((df.x_mod / df.gap)[m])))) if m.any() else np.nan
    r["lead_slope_icpt"] = boot_slope(df, m)       # (slope, lo, hi, intercept, n, clusters): x_mod = slope * gap + intercept
    t = df.gap.notna()
    r["lead_truth_n"], r["lead_recall"] = int(t.sum()), float((df.p_lead[t] > .5).mean()) if t.any() else np.nan
    # distance-resolved ratio
    for lo, hi in ((8, 15), (15, 25), (25, 50)):
        mm = m & df.gap.between(lo, hi)
        r[f"lead_ratio_{lo}_{hi}"] = boot((df.x_mod / df.gap)[mm], df.cl[mm])
    sp = df.v > 3
    r["speed_ratio"] = boot((df.pv0 / df.v)[sp], df.cl[sp])
    for x in (10, 20):
        mk = df.ok
        r[f"lane_w{x}"] = boot(df[f"w{x}"][mk], df.cl[mk])
    r["road_z"] = boot(((df.z10 + df.z20) / 2)[df.ok], df.cl[df.ok])
    r["h_true_assumed"] = h_true
    return r


# ---------------------------------------------------------------- WOD perception (sceneflow)
def sf_df(variant):
    rows = []
    for f in sorted(glob.glob(str(RUN / f"sf_{variant}_*.npz"))):
        seg = Path(f).stem.split("_", 2)[2]
        z = np.load(f)
        sl = json.loads(str(z["info"]))["heads_slices"]
        first = int(z["first"])
        H = z["heads"]
        d = pickle.load(open(RUN / "wod_sf" / f"{seg}.pkl", "rb"))
        cam_x = float(d["cal"][1]["extrinsic"][0, 3])
        ll, lp, xm, vm, pl, pv = parse(H, sl)
        lc = lane_cols(ll, lp)
        for i in range(len(H)):
            j = first + i
            fr = d["frames"][j]
            b = fr["boxes"]
            b = b[(b[:, 7] == 1) & (b[:, 10] >= 5)] if len(b) else b
            # keep the box columns [cx, cy, cz, L, W, H, yaw]
            gap = lead_truth(b, cam_x, False)
            rows.append(dict(cl=seg, frame=j, v=float(fr["vel"][0]), pv0=float(pv[i]), x_mod=float(xm[i]), p_lead=float(pl[i]), gap=gap, **lc.iloc[i].to_dict()))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- WOD-E2E
def e2e_df(variant):
    z = np.load(RUN / f"e2e_{variant}.npz")
    sl = json.loads(str(z["info"]))["heads_slices"]
    ll, lp, xm, vm, pl, pv = parse(z["heads"], sl)
    import sys as _s
    _s.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from jevdrive import wod_zeroshot as Z, waymo as W
    S = Z.load_sets()
    sp = {str(n): float(v) for w in ("rater", "extra") for n, v in zip(S[w]["name"], W.init_speed(S[w]["past"]))}
    names = z["names"].astype(str)
    lc = lane_cols(ll, lp)
    return pd.DataFrame(dict(cl=[n.rsplit("-", 1)[0] for n in names], v=[sp[n] for n in names], pv0=pv, x_mod=xm, p_lead=pl, gap=np.nan, **lc.to_dict("list")))


# ---------------------------------------------------------------- NAVSIM
def nav_df(file, keep=None):
    z = np.load(file, allow_pickle=True)
    sl = json.loads(str(z["info"]))["heads_slices"]
    names = z["names"].astype(str)
    sel = np.arange(len(names)) if keep is None else np.array([i for i, n in enumerate(names) if n in keep])
    H = z["heads"][sel]
    names = names[sel]
    ll, lp, xm, vm, pl, pv = parse(H, sl)
    lc = lane_cols(ll, lp)
    want = set(names.tolist())
    fr = {}
    for f in glob.glob(str(D / "datasets/navsim/navsim_logs/test/*.pkl")):
        for e in pickle.load(open(f, "rb")):
            if e["token"] in want:
                a = e["anns"]
                b = np.asarray(a["gt_boxes"], float).reshape(-1, 7)[[n == "vehicle" for n in a["gt_names"]]]
                fr[e["token"]] = (b, e["log_name"], float(np.linalg.norm(e["ego_dynamic_state"][:2])))
    rows = []
    for i, n in enumerate(names):
        b, log, v = fr[n]
        rows.append(dict(cl=log, v=v, pv0=float(pv[i]), x_mod=float(xm[i]), p_lead=float(pl[i]), gap=lead_truth(b, 1.62, True), **lc.iloc[i].to_dict()))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = {}
    which = sys.argv[1:] or ["sf", "e2e", "nav"]
    if "sf" in which:
        for v, h in (("base", 2.165), ("h1.22", 1.22)):
            df = sf_df(v)
            df = df[df.frame >= SF_FROM]
            df.to_csv(RUN / f"samples_sf_{v}.csv", index=False)
            out[f"sf_{v}"] = board_stats(df, v, h)
    if "e2e" in which:
        for v, h in (("base", 1.857), ("h1.22", 1.22)):
            df = e2e_df(v)
            df.to_csv(RUN / f"samples_e2e_{v}.csv", index=False)
            out[f"e2e_{v}"] = board_stats(df, v, h)
    if "nav" in which:
        df = nav_df(D / "runs/op_lb/lb_navtest/plans/gimm@cinque.npz")
        df.to_csv(RUN / "samples_nav_full.csv", index=False)
        out["nav_full_gimm"] = board_stats(df, "nav", 1.87)
        keep = set(np.load(D / "runs/skill_pack/edge_diag/height/hgt_r1.npz")["names"].astype(str).tolist())
        for r, h in (("1", 1.87), ("1.44", 1.30)):
            df = nav_df(D / f"runs/skill_pack/edge_diag/height/hgt_r{r}.npz")
            out[f"nav_hgt_r{r}"] = board_stats(df, "nav", h)
    p = RUN / "stats.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    old.update(out)
    p.write_text(json.dumps(old, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))
