"""op_parity four-directions lane, direction 4: night / low light on the HUGSIM + NAVSIM tier.

Night label = decision 131 luma rule (experiments/leaderboard_audit/results/night_gap.md): mean luma of the front camera, per log / scene the
median over frames, night < cut (50), day >= 120, dusk in between (kept separate). Cut sensitivity 35 / 50 / 65.

  luma    navtest + navhard_two_stage: CAM_F0 of the token's current frame (all tokens, 1/8 JPEG draft), -> $DATA_DIR/runs/op_parity/four_dirs/luma_navsim.parquet
  hugsim  64 HUGSIM scenarios: front third of the rendered 3-camera video.mp4 (every 4th frame), median over frames -> luma_hugsim.parquet
  report  scores (P2H10-F-s0/s1 seed mean vs WA-JEPA) by night / dusk / day with cluster bootstrap CIs, size, per-city shares, figures
          -> --out (night.md tables, night_units.csv, figs/night_luma.png)
  sheet   contact sheet of night / dusk / day frames per board -> --out/figs/night_sheet.jpg

Runs on the box (needs the sensor blobs and the score files); CPU only. python experiments/op_parity/scripts/fd_night.py <cmd> [--out DIR]
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_pl.Path(__file__).parent)]
import argparse, json, os  # noqa: E401,E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from jevdrive.common import data_dir, n_cpus  # noqa: E402

CUT_DAY = 120.0
CUTS = (35.0, 50.0, 65.0)
CUT = 50.0
WORK = data_dir() / "runs/op_parity/four_dirs"
OUT = _R / "experiments/op_parity/results/four_dirs"
HUG = data_dir() / "runs/bench/hugsim"
HR = _R / "experiments/hugsim/results"
CITY = {"us-ma-boston": "Boston", "us-nv-las-vegas-strip": "Las Vegas", "us-pa-pittsburgh-hazelwood": "Pittsburgh", "sg-one-north": "Singapore"}
LATLON = {"Boston": (42.36, -71.06), "Las Vegas": (36.17, -115.14), "Pittsburgh": (40.44, -80.00), "Singapore": (1.30, 103.79)}
HUG_CROP = (800, 1600, 0, 450)          # front third (x0, x1, y0, y1) of the 2400 x 900 video; the three cameras sit in the top 450 rows


def sun_elevation(log: str, city: str) -> float:
    """Solar elevation (deg, NOAA low-precision formulas) at the city centre for a nuPlan log name '<yyyy.mm.dd.HH.MM.SS>_<veh>_<first>_<last>'
    (start time in UTC; the snippet starts first / 20 s later: lidar frame index at 20 Hz). Independent time-of-day label for navtest / navhard."""
    import datetime as dt
    t0 = dt.datetime.strptime(log.split("_")[0], "%Y.%m.%d.%H.%M.%S")
    t = t0 + dt.timedelta(seconds=int(log.split("_")[2]) / 20.0)
    lat, lon = LATLON[city]
    n = t.timetuple().tm_yday
    h = t.hour + t.minute / 60 + t.second / 3600
    g = 2 * np.pi / 365 * (n - 1 + (h - 12) / 24)
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    dec = 0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g)
    tst = h * 60 + eqt + 4 * lon                                  # true solar time, minutes (UTC clock + longitude)
    ha = np.radians(tst / 4 - 180)
    la = np.radians(lat)
    return float(np.degrees(np.arcsin(np.sin(la) * np.sin(dec) + np.cos(la) * np.cos(dec) * np.cos(ha))))


def workers():
    return max(1, min(n_cpus() // 3, 24))      # another CPU lane runs on the box: leave room


def label(l, cut=CUT):
    return np.where(l < cut, "night", np.where(l >= CUT_DAY, "day", "dusk"))


# ---------------------------------------------------------------- luma
def _luma_jpg(p):
    from PIL import Image
    try:
        im = Image.open(p)
        im.draft("L", (im.width // 8, im.height // 8))
        return float(np.asarray(im.convert("L"), np.float32).mean())
    except Exception:
        return float("nan")


def _luma_chunk(ps):
    return [_luma_jpg(p) for p in ps]


def cmd_luma(a):
    from jevdrive import navsim_zs as Z
    rows = []
    for split in ("navtest", "navhard_two_stage"):
        for e in Z.load_index(split, slim=True):
            rows.append((split, e["token"], e["log_name"], e["map"], e["stage"], e["cams"][-1]["CAM_F0"]["path"]))
    df = pd.DataFrame(rows, columns=["board", "token", "log", "map", "stage", "path"])
    ch = [df.path.tolist()[i:i + 200] for i in range(0, len(df), 200)]
    with ProcessPoolExecutor(workers()) as ex:
        df["luma"] = [v for part in ex.map(_luma_chunk, ch) for v in part]
    WORK.mkdir(parents=True, exist_ok=True)
    df.to_parquet(WORK / "luma_navsim.parquet")
    print(df.groupby(["board", "stage"]).luma.describe().to_string(), "\nnan:", int(df.luma.isna().sum()))


def _luma_video(args):
    scen, p = args
    import cv2
    c = cv2.VideoCapture(p)
    n = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
    x0, x1, y0, y1 = HUG_CROP
    v = []
    for i in range(0, n, 4):
        c.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = c.read()
        if ok:
            v.append(float(cv2.cvtColor(f[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).mean()))
    return scen, float(np.median(v)) if v else float("nan"), len(v), p


def cmd_hugsim(a):
    u = pd.read_csv(HUG / "P2H10-F-s0_spec_plan_smooth-rr1/units.csv")
    jobs = [(r.scenario, str(_pl.Path(r.run_dir) / "video.mp4")) for r in u.itertuples()]
    with ProcessPoolExecutor(workers()) as ex:
        res = list(ex.map(_luma_video, jobs))
    df = pd.DataFrame(res, columns=["scenario", "luma", "n_frames", "video"]).merge(u[["scenario", "dataset", "difficulty", "scene"]], on="scenario")
    WORK.mkdir(parents=True, exist_ok=True)
    df.to_parquet(WORK / "luma_hugsim.parquet")
    print(df.drop(columns="video").sort_values("luma").to_string())


# ---------------------------------------------------------------- scores
def navtest_scores():
    """token -> (log, P2H10 seed mean EPDMS x 100, WA-JEPA x 100)."""
    import pp_hinge_report as H
    E, log, t = H._navtest(["P2H10-F-s0", "P2H10-F-s1"])
    wa = E.read_csv(data_dir() / E.WAJEPA_CSV)[0]
    toks = sorted(set(t["P2H10-F-s0"].index) & set(t["P2H10-F-s1"].index) & set(wa.index))
    p = np.mean([100.0 * t[m].loc[toks, "score"].to_numpy(float) for m in t], 0)
    return pd.DataFrame({"token": toks, "log": [log[k] for k in toks], "p2h": p, "wa": 100.0 * wa.loc[toks, "score"].to_numpy(float)})


def navhard_scores():
    """group -> (stage-1 log of the group, orig token, P2H10 (G) seed mean and WA-JEPA, combined / stage1 / stage2)."""
    from jevdrive import navsim_zs as Z
    from jevdrive.bench.compat import navhard_dir
    import pp_hinge_report as H
    lg = {e["token"]: e["log_name"] for e in Z.load_index("navhard_two_stage", slim=True)}
    P = data_dir() / "runs/op_parity"
    ds = [navhard_dir("P2H10-F-s%d@gimm" % s, H.RUN / f"harness/P2H10-F-s{s}") for s in (0, 1)]
    wd = navhard_dir("WA-JEPA", P / "navhard/harness/wajepa")
    p = [pd.read_csv(d / "harness_groups.csv").set_index("group") for d in ds]
    w = pd.read_csv(wd / "harness_groups.csv").set_index("group")
    assert all((x.orig.values == w.orig.values).all() for x in p), "group order differs"
    out = pd.DataFrame({"group": w.index, "orig": w.orig.values, "log": [lg.get(o, "?") for o in w.orig]})
    for c in ("combined", "stage1", "stage2"):
        out[f"p2h_{c}"] = np.mean([x[c].to_numpy(float) for x in p], 0)
        out[f"wa_{c}"] = w[c].to_numpy(float)
    return out


def hugsim_scores():
    """scenario -> P2H10 spec_plan_smooth HD x 100 (mean of s0 / s1 x rr1 / rr2) and WA-JEPA exam HD x 100."""
    us = [pd.read_csv(HUG / f"P2H10-F-s{s}_spec_plan_smooth-rr{r}/units.csv").set_index("scenario").hdscore for s in (0, 1) for r in (1, 2)]
    W = pd.read_csv(HR / "hugsim-exam/scored_wajepa.csv").query("tag == 'wajepa'").drop_duplicates("scenario", keep="last").set_index("scenario")
    sc = list(W.index)
    return pd.DataFrame({"scenario": sc, "p2h": 100 * pd.concat([u.reindex(sc) for u in us], axis=1).mean(axis=1, skipna=False).to_numpy(), "wa": 100 * W.hdscore.to_numpy(float)})


# ---------------------------------------------------------------- analysis
def gap_table(d, cut, B=10000, seed=0, lab_fn=None):
    """d: columns cluster, luma_c (cluster luma), p2h, wa. Cluster bootstrap (resample clusters, ratio of sums, as jevdrive.stats).
    Returns one dict: shares, bucket means, gaps (WA - P2H), size."""
    d = d[np.isfinite(d.p2h) & np.isfinite(d.wa)]
    lab = label(d.luma_c.to_numpy(float), cut) if lab_fn is None else lab_fn(d)
    codes, uniq = pd.factorize(d.cluster)
    K = len(uniq)
    cl = pd.Series(lab).groupby(codes).first().to_numpy()
    nc = np.bincount(codes, minlength=K).astype(float)
    gap = (d.wa - d.p2h).to_numpy(float)

    def S(x, m):
        return np.bincount(codes, np.where(m, x, 0.0), K)
    idx = np.random.default_rng(seed).integers(K, size=(B, K))

    def stat(ix):
        """ix: (R, K) cluster index arrays -> dict of statistics (R,)."""
        N = nc[ix].sum(1)
        out = {}
        for b in ("night", "dusk", "day", "all"):
            m = (lab == b) if b != "all" else np.ones(len(lab), bool)
            n = (nc * (cl == b))[ix].sum(1) if b != "all" else N
            out[f"n_{b}"] = n
            out[f"share_{b}"] = 100 * n / N
            for nm, x in (("p2h", d.p2h), ("wa", d.wa), ("gap", gap)):
                with np.errstate(invalid="ignore", divide="ignore"):
                    out[f"{nm}_{b}"] = S(np.asarray(x, float), m)[ix].sum(1) / n
        out["gap_diff"] = out["gap_night"] - out["gap_day"]
        out["size_replace"] = out["share_night"] / 100 * out["gap_night"]              # score points gained by giving WA-JEPA's score on night units
        out["size_excess"] = out["share_night"] / 100 * out["gap_diff"]                # part of that above what day's gap would give
        out["gap_share_night"] = 100 * out["size_replace"] / out["gap_all"]            # share of the whole WA - P2H gap that sits on night units
        return out
    pt = stat(np.arange(K)[None])
    bs = stat(idx)
    r = {"cut": cut, "units": len(d), "clusters": K, "night_clusters": int((cl == "night").sum()), "dusk_clusters": int((cl == "dusk").sum()),
         "day_clusters": int((cl == "day").sum())}
    for k, v in pt.items():
        v = float(v[0])
        ok = np.isfinite(bs[k])
        lo, hi = np.nanquantile(bs[k], [0.025, 0.975]) if ok.sum() > 100 else (np.nan, np.nan)
        r[k], r[k + "_lo"], r[k + "_hi"] = v, float(lo), float(hi)
    return r


def city_share(lum, cut):
    """per-city night share over units (tokens) and over logs."""
    lg = lum.groupby("log").agg(luma=("luma", "median"), city=("city", "first"))
    lg["lab"] = label(lg.luma.to_numpy(), cut)
    t = lum.assign(lab=lum.log.map(lg.lab))
    rows = []
    for c, g in t.groupby("city"):
        rows.append({"city": c, "logs": int((lg.city == c).sum()), "night_logs": int(((lg.city == c) & (lg.lab == "night")).sum()),
                     "dusk_logs": int(((lg.city == c) & (lg.lab == "dusk")).sum()), "tokens": len(g),
                     "night_tokens": int((g.lab == "night").sum()), "night_pct": 100 * (g.lab == "night").mean(), "dusk_pct": 100 * (g.lab == "dusk").mean()})
    return pd.DataFrame(rows)


def mdt(df, f=".2f"):
    df = df.copy()
    return "\n".join(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)] +
                     ["| " + " | ".join(f"{x:{f}}" if isinstance(x, float) else str(x) for x in r) + " |" for r in df.itertuples(index=False)])


def ci(r, k, f=".2f"):
    return f"{r[k]:{f}} [{r[k + '_lo']:{f}}, {r[k + '_hi']:{f}}]"


def cmd_report(a):
    out = _pl.Path(a.out)
    (out / "figs").mkdir(parents=True, exist_ok=True)
    L = pd.read_parquet(WORK / "luma_navsim.parquet")
    L["city"] = L["map"].map(CITY).fillna(L["map"])
    H = pd.read_parquet(WORK / "luma_hugsim.parquet")
    boards = {}
    # --- navtest: units = tokens, cluster = log, label = median luma of the log's tokens
    nt = L[L.board == "navtest"].copy()
    nt["luma_c"] = nt.groupby("log").luma.transform("median")
    s = navtest_scores()
    s = s.merge(nt[["token", "luma", "luma_c", "city"]], on="token")
    s["cluster"] = s["log"]
    s["sun"] = [sun_elevation(l, c) for l, c in zip(s["log"], s.city)]
    boards["navtest"] = s.assign(unit=s.token)
    # --- navhard: units = groups (225), cluster = stage-1 log; luma_c = median over the log's stage-1 tokens (stage 1 only)
    nh = L[L.board == "navhard_two_stage"].copy()
    nh1 = nh[nh.stage == "one"]
    lg1 = nh1.groupby("log").luma.median()
    lg2 = nh[nh.stage == "two"].groupby("log").luma.median()
    g = navhard_scores()
    g = g.merge(nh1[["token", "luma", "city"]].rename(columns={"token": "orig"}), on="orig", how="left")
    g["luma_c"] = g["log"].map(lg1)
    g["luma_s2"] = g["log"].map(lg2)
    g["cluster"] = g["log"]
    g["sun"] = [sun_elevation(l, c) for l, c in zip(g["log"], g.city)]
    g["unit"] = g.group.astype(str)
    nhc = g.assign(p2h=g.p2h_combined, wa=g.wa_combined)
    boards["navhard"] = nhc
    # --- hugsim
    h = hugsim_scores().merge(H[["scenario", "luma", "dataset"]], on="scenario")
    h["luma_c"] = h.luma
    h["cluster"] = h.scenario
    h["unit"] = h.scenario
    h["city"] = h.dataset
    boards["hugsim"] = h

    res = []
    for b, d in boards.items():
        for cut in CUTS:
            r = gap_table(d, cut)
            r["board"] = b
            res.append(r)
    R = pd.DataFrame(res)
    R.to_csv(out / "night_gap_cuts.csv", index=False)

    # navhard stage-wise at cut 50
    stg = []
    for c in ("combined", "stage1", "stage2"):
        r = gap_table(nhc.assign(p2h=g[f"p2h_{c}"], wa=g[f"wa_{c}"]), CUT)
        r["board"] = f"navhard {c}"
        stg.append(r)
    ST = pd.DataFrame(stg)

    # low-light proxy: darkest 10% of clusters per board vs the rest (NOT night: the absolute rule finds none on the NAVSIM boards)
    def dim(d, q=0.10):
        cl = d.drop_duplicates("cluster")
        thr = np.quantile(cl.luma_c, q)
        return np.where(d.luma_c.to_numpy(float) <= thr, "night", "day")
    PX = []
    for b, d in boards.items():
        for q in (0.10, 0.25):
            r = gap_table(d, CUT, lab_fn=lambda x, q=q: dim(x, q))
            r["board"], r["q"] = b, q
            r["thr"] = float(np.quantile(d.drop_duplicates("cluster").luma_c, q))
            PX.append(r)
    PX = pd.DataFrame(PX)
    PX.to_csv(out / "lowlight_proxy.csv", index=False)

    # independent label: solar elevation < 0 at the log's time and city (twilight or darker), NAVSIM boards only
    SG = []
    for b in ("navtest", "navhard"):
        r = gap_table(boards[b], CUT, lab_fn=lambda x: np.where(x["sun"].to_numpy(float) < 0, "night", "day"))
        r["board"] = b
        SG.append(r)
    SG = pd.DataFrame(SG)
    SG.to_csv(out / "sun_label_gap.csv", index=False)

    # sun elevation (independent time-of-day check) on the NAVSIM boards: logs below the horizon / civil twilight
    sunrows = []
    for b in ("navtest", "navhard"):
        d = boards[b].drop_duplicates("cluster")
        for c, gg in d.groupby("city"):
            sunrows.append({"board": b, "city": c, "logs": len(gg), "sun_min": gg["sun"].min(), "sun_median": gg["sun"].median(), "logs_sun<0": int((gg["sun"] < 0).sum()),
                            "logs_sun<10": int((gg["sun"] < 10).sum()), "luma_median": gg.luma_c.median(), "luma_min": gg.luma_c.min()})
    SUN = pd.DataFrame(sunrows)
    dd = boards["navtest"].drop_duplicates("cluster")
    cons_sun = float(np.corrcoef(dd["sun"], dd.luma_c)[0, 1])
    lowsun = dd.sort_values("sun").head(8)[["cluster", "city", "sun", "luma_c"]]

    # units csv
    rows = []
    for b, d in boards.items():
        rows.append(pd.DataFrame({"unit": d.unit, "board": b, "luma": d.luma_c.round(1), "label": label(d.luma_c.to_numpy(float)), "unit_luma": d.luma.round(1),
                                  "cluster": d.cluster, "city_or_dataset": d.city, "sun_elev_deg": d["sun"].round(1) if "sun" in d else np.nan}))
    U = pd.concat(rows)
    U.to_csv(out / "night_units.csv", index=False)

    # per city
    nts = city_share(nt, CUT)
    nhs = city_share(nh1, CUT)
    hs = H.assign(lab=label(H.luma.to_numpy())).groupby("dataset").agg(scenarios=("lab", "size"), night=("lab", lambda x: int((x == "night").sum())),
                                                                       dusk=("lab", lambda x: int((x == "dusk").sum())), median_luma=("luma", "median")).reset_index()

    # navhard luma consistency: stage-2 rendered vs stage-1 per log, orig token vs log median
    cons = {"stage-1 orig token luma vs log median: corr": float(np.corrcoef(g.luma.fillna(g.luma_c), g.luma_c)[0, 1]),
            "stage-2 log median vs stage-1 log median: corr": float(np.corrcoef(*pd.concat([lg1, lg2], axis=1).dropna().T.to_numpy())[0, 1]),
            "stage-2 minus stage-1 log median luma, mean": float((lg2 - lg1).dropna().mean())}
    # token-level vs log-level label agreement (navtest)
    agree = float((label(nt.luma.to_numpy()) == label(nt.luma_c.to_numpy())).mean())
    cons["navtest token label == log label"] = agree
    cons["navtest log luma vs sun elevation: corr"] = cons_sun

    # ---- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4), sharey=False)
    col = {"night": "#4c5fd7", "dusk": "#d99a2b", "day": "#8a8a8a"}
    for x, (b, d) in zip(ax, boards.items()):
        u = d.drop_duplicates("cluster")
        bins = np.arange(0, 201, 5)
        x.hist(u.luma_c.clip(upper=199), bins=bins, color="#6b7a99")
        for c in CUTS:
            x.axvline(c, color=col["night"], lw=1, ls=":" if c != CUT else "-")
        x.axvline(CUT_DAY, color=col["day"], lw=1)
        n = (u.luma_c < CUT).sum()
        x.set_title(f"{b}: {len(u)} {'scenarios' if b == 'hugsim' else 'logs'}, {n} night", fontsize=10)
        x.set_xlabel("front-camera mean luma (median over frames)")
    ax[0].set_ylabel("logs / scenarios")
    fig.tight_layout()
    fig.savefig(out / "figs/night_luma.png", dpi=130)

    # ---- tables for the doc
    hd = ["board", "units", "night units (%)", "night clusters", "P2H night / day", "WA night / day", "gap night / day", "gap diff (night - day)", "size: replace night", "size: excess", "night share of gap"]
    T = []
    for r in R[R.cut == CUT].to_dict("records"):
        T.append([r["board"], r["units"], f"{r['n_night']:.0f} ({r['share_night']:.1f}% [{r['share_night_lo']:.1f}, {r['share_night_hi']:.1f}])",
                  f"{r['night_clusters']} / {r['clusters']}", f"{r['p2h_night']:.2f} / {r['p2h_day']:.2f}", f"{r['wa_night']:.2f} / {r['wa_day']:.2f}",
                  f"{r['gap_night']:.2f} / {r['gap_day']:.2f}", ci(r, "gap_diff"), ci(r, "size_replace", ".2f"), ci(r, "size_excess", ".2f"),
                  f"{r['gap_share_night']:.1f}%"])
    main = pd.DataFrame(T, columns=hd)
    sens = pd.DataFrame([[r["board"], r["cut"], f"{r['n_night']:.0f}", f"{r['share_night']:.1f}", r["night_clusters"], f"{r['gap_night']:.2f}", f"{r['gap_day']:.2f}", ci(r, "gap_diff"),
                          ci(r, "size_replace"), ci(r, "size_excess")] for r in R.to_dict("records")],
                        columns=["board", "night cut", "night units", "share %", "night clusters", "gap night", "gap day", "gap diff", "size replace", "size excess"])
    stt = pd.DataFrame([[r["board"], f"{r['p2h_night']:.2f} / {r['p2h_dusk']:.2f} / {r['p2h_day']:.2f}", f"{r['wa_night']:.2f} / {r['wa_dusk']:.2f} / {r['wa_day']:.2f}",
                         ci(r, "gap_diff"), ci(r, "size_replace")] for r in ST.to_dict("records")],
                        columns=["score", "P2H n / dusk / d", "WA-JEPA n / dusk / d", "gap diff night - day", "size replace"])
    full = pd.DataFrame([[r["board"], f"{r['n_night']:.0f} / {r['n_dusk']:.0f} / {r['n_day']:.0f}", f"{r['p2h_night']:.2f} / {r['p2h_dusk']:.2f} / {r['p2h_day']:.2f}",
                          f"{r['wa_night']:.2f} / {r['wa_dusk']:.2f} / {r['wa_day']:.2f}", ci(r, "gap_night"), ci(r, "gap_dusk"), ci(r, "gap_day"), ci(r, "gap_all")]
                         for r in R[R.cut == CUT].to_dict("records")],
                        columns=["board", "units n / dusk / d", "P2H n / dusk / d", "WA-JEPA n / dusk / d", "gap night", "gap dusk", "gap day", "gap all"])
    parts = {"main": mdt(main), "full": mdt(full), "sens": mdt(sens), "navhard_stages": mdt(stt), "city_navtest": mdt(nts, ".1f"), "city_navhard": mdt(nhs, ".1f"),
             "hugsim_ds": mdt(hs, ".1f"), "sun": mdt(SUN, ".1f"), "lowsun": mdt(lowsun, ".1f"),
             "sunlab": mdt(pd.DataFrame([[r["board"], f"{r['n_night']:.0f} ({r['share_night']:.2f}%)", r["night_clusters"], f"{r['p2h_night']:.2f} / {r['p2h_day']:.2f}", f"{r['wa_night']:.2f} / {r['wa_day']:.2f}",
                                           f"{r['gap_night']:.2f} / {r['gap_day']:.2f}", ci(r, "gap_diff"), ci(r, "size_replace")] for r in SG.to_dict("records")],
                                         columns=["board", "sun<0 units", "logs", "P2H sun<0 / rest", "WA sun<0 / rest", "gap", "gap diff", "size replace"])),
             "proxy": mdt(pd.DataFrame([[r["board"], r["q"], r["thr"], r["units"], f"{r['n_night']:.0f}", f"{r['share_night']:.1f}", f"{r['p2h_night']:.2f} / {r['p2h_day']:.2f}", f"{r['wa_night']:.2f} / {r['wa_day']:.2f}",
                                           f"{r['gap_night']:.2f} / {r['gap_day']:.2f}", ci(r, "gap_diff"), ci(r, "size_replace"), ci(r, "size_excess")] for r in PX.to_dict("records")],
                                         columns=["board", "bottom q", "luma thr", "units", "dim units", "share %", "P2H dim / rest", "WA dim / rest", "gap dim / rest", "gap diff", "size replace", "size excess"])), "cons": "\n".join(f"- {k}: {v:.3f}" for k, v in cons.items())}
    (out / "_tables.json").write_text(json.dumps(parts, indent=1))
    for k, v in parts.items():
        print(f"\n== {k}\n{v}")


# ---------------------------------------------------------------- calibration: can the luma rule see night on nuPlan cameras at all?
def cmd_calib(a):
    """navtrain (not a board of this study) logs by solar elevation: luma of one mid-log token for the 12 lowest-sun logs and 12 random high-sun logs."""
    from jevdrive import navsim_zs as Z
    ix = Z.load_index("navtrain", slim=True)
    lg = {}
    for e in ix:
        lg.setdefault(e["log_name"], []).append(e)
    rows = []
    for l, es in lg.items():
        c = CITY.get(es[0]["map"])
        if c is None:
            continue
        e = es[len(es) // 2]
        rows.append((l, c, sun_elevation(l, c), e["cams"][-1]["CAM_F0"]["path"]))
    df = pd.DataFrame(rows, columns=["log", "city", "sun", "path"]).sort_values("sun")
    pick = pd.concat([df.head(12), df.iloc[np.random.default_rng(0).permutation(len(df))[:12]]])
    pick["luma"] = [_luma_jpg(p) for p in pick.path]
    pick["group"] = ["lowest sun"] * 12 + ["random"] * 12
    WORK.mkdir(parents=True, exist_ok=True)
    pick.drop(columns="path").to_csv(WORK / "calib_navtrain.csv", index=False)
    print(len(df), "navtrain logs;", int((df.sun < -6).sum()), "with sun < -6;", int((df.sun < 0).sum()), "with sun < 0")
    print(pick.drop(columns="path").round(1).to_string())


# ---------------------------------------------------------------- contact sheet
def cmd_sheet(a):
    from PIL import Image, ImageDraw
    import cv2
    out = _pl.Path(a.out)
    (out / "figs").mkdir(parents=True, exist_ok=True)
    U = pd.read_csv(out / "night_units.csv")
    L = pd.read_parquet(WORK / "luma_navsim.parquet").set_index("token")
    H = pd.read_parquet(WORK / "luma_hugsim.parquet").set_index("scenario")
    rng = np.random.default_rng(1)
    TW, TH, NC = 288, 162, 5
    rows = []
    for b in ("navtest", "navhard", "hugsim"):
        u = U[U.board == b]
        for lab in ("night", "darkest", "day"):
            # one frame per cluster (distinct logs / scenarios); night and day random, darkest = the NC lowest-luma clusters that are not night
            cand = u[u.label == lab].drop_duplicates("cluster") if lab != "darkest" else u[u.label != "night"].drop_duplicates("cluster").sort_values("luma")
            if not len(cand):
                continue
            pick = cand.iloc[rng.permutation(len(cand))[:NC]] if lab != "darkest" else cand.iloc[:NC]
            ims = []
            for r in pick.itertuples():
                if b == "hugsim":
                    v = cv2.VideoCapture(H.loc[r.unit, "video"])
                    v.set(cv2.CAP_PROP_POS_FRAMES, int(v.get(cv2.CAP_PROP_FRAME_COUNT)) // 3)
                    ok, f = v.read()
                    x0, x1, y0, y1 = HUG_CROP
                    im = Image.fromarray(cv2.cvtColor(f[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
                    txt = f"{r.unit[:26]} {r.luma:.0f}"
                else:
                    log_toks = L[(L.log == r.cluster) & (L.board == ("navtest" if b == "navtest" else "navhard_two_stage")) & (L.stage == "one")]
                    p = log_toks.path.iloc[len(log_toks) // 2]
                    im = Image.open(p).convert("RGB")
                    txt = f"{r.cluster[:19]} {r.luma:.0f}"
                im = im.resize((TW, TH))
                ImageDraw.Draw(im).rectangle([0, 0, TW, 13], fill=(0, 0, 0))
                ImageDraw.Draw(im).text((2, 1), f"{b} {lab} luma {txt}", fill=(255, 255, 255))
                ims.append(im)
            rows.append(ims)
    S = Image.new("RGB", (NC * TW, len(rows) * TH))
    for i, ims in enumerate(rows):
        for j, im in enumerate(ims):
            S.paste(im, (j * TW, i * TH))
    S.save(out / "figs/night_sheet.jpg", quality=85)
    print(S.size, [len(r) for r in rows])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["luma", "hugsim", "report", "sheet", "calib"])
    ap.add_argument("--out", default=str(WORK / "out"))
    a = ap.parse_args()
    {"luma": cmd_luma, "hugsim": cmd_hugsim, "report": cmd_report, "sheet": cmd_sheet, "calib": cmd_calib}[a.cmd](a)
