"""Real vs render analysis (HUGSIM nuScenes): output gaps real vs render against the adjacent-frame floor, image-side fixes, history-yaw
probes. Rules: experiments/leaderboard_audit/plans/2026-10-04-real-vs-render-prereg.md. Reads $DATA_DIR/runs/real_vs_render/hugsim/<scene>/
{stream,probe}.npz and fixes.json / imgstats.json; writes $DATA_DIR/runs/real_vs_render/stats.json and prints the tables.
Project venv, CPU: .venv/bin/python experiments/leaderboard_audit/scripts/rvr_analyze.py"""
import json, os
from pathlib import Path
import numpy as np

D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
RUN = D / "runs/real_vs_render"
X_IDXS = 192.0 * (np.arange(33) / 32) ** 2
T_IDXS = 10.0 * (np.arange(33) / 32) ** 2
FIRST = 36                      # frames >= 3 s
PRIMARY = ("plan_lat4", "edge_y10", "lane_y10", "plan_v0")
METRICS = ("plan_lat4", "edge_y10", "lane_y10", "plan_v0", "plan_lat2", "plan_x4", "yaw3", "lane_y20", "lane_p", "lead_p", "lead_x")
B = 2000
rng = np.random.default_rng(0)
sig = lambda a: 1 / (1 + np.exp(-a))  # noqa: E731


def heads(H, hs):
    """Kept-layout rows (n, K) -> per-frame readouts (dict of arrays; nan where undefined)."""
    n = len(H)
    ll = H[:, hs["lane_lines"]:hs["lane_lines"] + 264].reshape(n, 4, 33, 2)
    lp = sig(H[:, hs["lane_lines_prob"]:hs["lane_lines_prob"] + 8][:, 1::2])
    re = H[:, hs["road_edges"]:hs["road_edges"] + 132].reshape(n, 2, 33, 2)
    lead = H[:, hs["lead"]:hs["lead"] + 72].reshape(n, 3, 6, 4)
    lpr = sig(H[:, hs["lead_prob"]])
    pl = H[:, hs["plan"]:hs["plan"] + 495].reshape(n, 33, 15)
    at = lambda a, x, idx=X_IDXS: np.array([np.interp(x, idx, r) for r in a])  # noqa: E731
    ok = lp[:, 1:3] > 0.5
    lane10 = np.stack([at(ll[:, 1, :, 0], 10), at(ll[:, 2, :, 0], 10)], 1)
    lane20 = np.stack([at(ll[:, 1, :, 0], 20), at(ll[:, 2, :, 0], 20)], 1)
    return dict(lane_y10=np.where(ok, lane10, np.nan), lane_y20=np.where(ok, lane20, np.nan),
                edge_y10=np.stack([at(re[:, 0, :, 0], 10), at(re[:, 1, :, 0], 10)], 1),
                lane_p=lp, lead_p=lpr[:, None], lead_x=np.where(lpr > 0.5, lead[:, 0, 0, 0], np.nan)[:, None],
                plan_lat2=at(pl[:, :, 1], 2.0, T_IDXS)[:, None], plan_lat4=at(pl[:, :, 1], 4.0, T_IDXS)[:, None],
                plan_v0=pl[:, 0, 3][:, None], plan_x4=at(pl[:, :, 0], 4.0, T_IDXS)[:, None],
                yaw3=-np.degrees(at(pl[:, :, 11], 3.0, T_IDXS))[:, None],
                lane_w10=(lane10[:, 1] - lane10[:, 0])[:, None], road_z=np.where(ok.all(1), (at(ll[:, 1, :, 1], 15) + at(ll[:, 2, :, 1], 15)) / 2, np.nan)[:, None])


def absdiff(a, b):
    """Per-frame mean |a - b| over the metric's components (nan components ignored; all nan -> nan)."""
    d = np.abs(a - b)
    with np.errstate(all="ignore"):
        return np.nanmean(d, 1) if d.shape[1] > 1 else d[:, 0]


def load():
    S = {}
    for d in sorted((RUN / "hugsim").iterdir()):
        f = d / "stream.npz"
        if not f.exists():
            continue
        z = np.load(f)
        info = json.loads(str(z["info"]))
        S[d.name] = dict(v=z["v"], **{a: heads(z[a], info["heads_slices"]) for a in info["arms"]})
        if (d / "stream_env.npz").exists():
            S[d.name]["env"] = heads(np.load(d / "stream_env.npz")["env"], info["heads_slices"])
    return S


def pairs(S, a, b, shift=0, scenes=None):
    """{metric: (values, scene ids)} of |a(k) - b(k + shift)| over frames k >= FIRST."""
    out = {m: ([], []) for m in METRICS}
    for i, (s, x) in enumerate(S.items()):
        if scenes is not None and s not in scenes:
            continue
        n = len(x["v"])
        k = np.arange(FIRST, n - shift)
        for m in METRICS:
            d = absdiff(x[a][m][k], x[b][m][k + shift])
            ok = np.isfinite(d)
            out[m][0].append(d[ok]), out[m][1].append(np.full(ok.sum(), i))
    return {m: (np.concatenate(v), np.concatenate(c)) for m, (v, c) in out.items()}


def boot(fn, groups, B=B):
    """Point estimate and 95% CI of fn(list of per-scene arrays), resampling scenes."""
    est = fn(groups)
    bs = [fn([groups[j] for j in rng.integers(0, len(groups), len(groups))]) for _ in range(B)]
    return [float(est), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]


def split(vals, cl):
    return [vals[cl == c] for c in np.unique(cl)]


def ratio_stat(num, den):
    """Paired over scenes: median of pooled num / median of pooled den, scenes resampled together."""
    ids = sorted(set(np.unique(num[1])) & set(np.unique(den[1])))
    g = [(num[0][num[1] == i], den[0][den[1] == i]) for i in ids]

    def f(gs):
        return np.median(np.concatenate([a for a, _ in gs])) / np.median(np.concatenate([b for _, b in gs]))
    return boot(f, g)


def main():
    S = load()
    names = list(S)
    fx = json.loads((RUN / "fixes.json").read_text())
    calib = set(fx["calib_scenes"])
    evals = [s for s in names if s not in calib]
    out = dict(n_scenes=len(S), scenes=names, calib=sorted(calib))
    RR, RS = pairs(S, "real", "real", 1), pairs(S, "real", "render")
    # 1. gap vs floor
    tab = {}
    for m in METRICS:
        r = ratio_stat(RS[m], RR[m])
        tab[m] = dict(rr=float(np.median(RR[m][0])), rs=float(np.median(RS[m][0])), ratio=r, n=int(len(RS[m][0])),
                      meaningful=bool(r[1] > 1))
    out["gap"] = tab
    # added after the pre-registration (disclosed): the 12 Hz source fed at 20 Hz repeats frames 1-2 steps in a 3-frame / 5-step cycle, and
    # nuScenes sweeps come at 100 / 100 / 50 ms, so adjacent frames sit at different phases of openpilot's 0.2 s frame pair; frames k and
    # k + 3 (0.25 s) share the phase. Secondary floor, not the pre-registered line.
    RR3 = pairs(S, "real", "real", 3)
    out["gap_phase_floor"] = {m: dict(rr3=float(np.median(RR3[m][0])), ratio=ratio_stat(RS[m], RR3[m])) for m in METRICS}
    n_yes = sum(tab[m]["meaningful"] for m in PRIMARY)
    out["verdict_gap"] = "yes" if n_yes >= 2 else "partly" if n_yes == 1 else "no"
    # signed shifts render - real (scale-type readouts)
    sh = {}
    for m in ("plan_v0", "plan_x4", "lane_w10", "road_z", "lane_p", "lead_p", "lead_x", "plan_lat4", "yaw3"):
        g = []
        for s, x in S.items():
            k = np.arange(FIRST, len(x["v"]))
            d = (x["render"][m][k] - x["real"][m][k])
            d = d.mean(1) if m != "lane_p" else d[:, 1:3].mean(1)
            g.append(d[np.isfinite(d)])
        sh[m] = boot(lambda gs: np.median(np.concatenate(gs)), [x for x in g if len(x)])
        real_lvl = np.nanmedian(np.concatenate([np.nanmean(x["real"][m][FIRST:], 1) if m != "lane_p" else x["real"][m][FIRST:, 1:3].mean(1) for x in S.values()]))
        sh[m].append(float(real_lvl))
    out["shift_render_minus_real"] = sh
    # 1b. the closed-loop rig (render_env: 0.3 m lower, level, no recorded dynamic objects) against real and the training-view render
    if all("env" in x for x in S.values()):
        RE, VE = pairs(S, "real", "env"), pairs(S, "render", "env")
        out["env"] = {m: dict(real_env=float(np.median(RE[m][0])), render_env=float(np.median(VE[m][0])),
                              ratio_real_env=ratio_stat(RE[m], RR[m])) for m in METRICS}
        she = {}
        for m in ("plan_v0", "plan_x4", "lane_w10", "road_z", "lane_p", "plan_lat4", "yaw3"):
            g = []
            for x in S.values():
                k = np.arange(FIRST, len(x["v"]))
                d = (x["env"][m][k] - x["real"][m][k])
                d = d.mean(1) if m != "lane_p" else d[:, 1:3].mean(1)
                g.append(d[np.isfinite(d)])
            she[m] = boot(lambda gs: np.median(np.concatenate(gs)), [x for x in g if len(x)])
        out["shift_env_minus_real"] = she
    # 2. fixes (eval scenes only)
    RRe, RSe = pairs(S, "real", "real", 1, evals), pairs(S, "real", "render", 0, evals)
    fixes = {}
    for arm in ("render_sharp", "render_stat", "render_all"):
        F = pairs(S, "real", arm, 0, evals)
        fixes[arm] = {}
        for m in METRICS:
            r = ratio_stat(F[m], RSe[m])
            fixes[arm][m] = dict(gap=float(np.median(F[m][0])), closure=[1 - r[0], 1 - r[2], 1 - r[1]])
    harm = {}
    for arm in ("real_sharp", "real_all"):
        F = pairs(S, "real", arm, 0, evals)
        harm[arm] = {m: dict(gap=float(np.median(F[m][0])), ratio_to_floor=ratio_stat(F[m], RRe[m])) for m in METRICS}
    out["fixes"], out["harm"] = fixes, harm
    out["eval_floor_gap"] = {m: [float(np.median(RRe[m][0])), float(np.median(RSe[m][0]))] for m in METRICS}
    works = {}
    for arm, h in (("render_sharp", "real_sharp"), ("render_all", "real_all"), ("render_stat", None)):
        c = [fixes[arm][m]["closure"][0] >= 0.5 for m in PRIMARY]
        nh = [h is None or harm[h][m]["ratio_to_floor"][0] <= 1 for m in PRIMARY]
        works[arm] = dict(closes_half=dict(zip(PRIMARY, c)), no_harm=dict(zip(PRIMARY, nh)))
    out["fix_verdict"] = works
    # 3. per-scene gap vs render PSNR
    st = json.loads((RUN / "imgstats.json").read_text())
    ps = st["psnr_front_by_scene"]
    per = []
    for i, s in enumerate(names):
        g = RS["plan_lat4"][0][RS["plan_lat4"][1] == i]
        e = RS["edge_y10"][0][RS["edge_y10"][1] == i]
        per.append((s, ps.get(s), float(np.median(g)), float(np.median(e))))
    from scipy.stats import spearmanr
    out["per_scene"] = per
    out["spearman_psnr_vs_gap"] = {"plan_lat4": spearmanr([p[1] for p in per], [p[2] for p in per]).statistic,
                                   "edge_y10": spearmanr([p[1] for p in per], [p[3] for p in per]).statistic}
    # 4. probes
    pr = {}
    P = {}
    for d in sorted((RUN / "hugsim").iterdir()):
        f = d / "probe.npz"
        if f.exists():
            P[d.name] = np.load(f)
    if P:
        nm = list(next(iter(P.values()))["names"])
        ix = {n: nm.index(n) for n in nm}

        def gains(z, arm):
            r = z[arm]
            return dict(G1=(r[:, ix["g1L"]] - r[:, ix["g1R"]]) / 2, G10=(r[:, ix["g10L"]] - r[:, ix["g10R"]]) / 2,
                        L=(r[:, ix["launchL"]] - r[:, ix["launchR"]]) / 2, H0=r[:, ix["normal"]])
        for s_, z in P.items():
            z = {k: z[k] for k in z.files}
            for extra, key in (("probe_env.npz", "env"), ("probe_blur.npz", "real_blur")):
                fe = RUN / "hugsim" / s_ / extra
                if fe.exists():
                    z[key] = np.load(fe)[key]
            P[s_] = z
        for arm in ("render", "render_all", "env", "real_blur"):
            if not all(arm in (z.files if hasattr(z, "files") else z) for z in P.values()):
                continue
            for g in ("G1", "G10", "L"):
                for bn, lo, hi in (("stop", -1, 0.5), ("low", 0.5, 3), ("mid", 3, 99), ("all", -1, 99)):
                    grp = []
                    for z in P.values():
                        a, b = gains(z, "real")[g], gains(z, arm)[g]
                        m = (z["v"] >= lo) & (z["v"] < hi)
                        if m.any():
                            grp.append((a[m], b[m]))
                    if not grp:
                        continue
                    est = boot(lambda gs: np.mean(np.concatenate([b for _, b in gs])) / np.mean(np.concatenate([a for a, _ in gs])), grp)
                    pr[f"{arm}|{g}|{bn}"] = dict(real=float(np.mean(np.concatenate([a for a, _ in grp]))),
                                                 arm=float(np.mean(np.concatenate([b for _, b in grp]))), ratio=est,
                                                 n=int(sum(len(a) for a, _ in grp)), scenes=len(grp))
        hd = []
        for z in P.values():
            hd.append(np.abs(gains(z, "render")["H0"] - gains(z, "real")["H0"]))
        pr["normal_heading_absdiff_deg"] = boot(lambda gs: np.median(np.concatenate(gs)), hd)
    out["probes"] = pr
    out["nav"] = nav()
    (RUN / "stats.json").write_text(json.dumps(out, indent=1, default=float))
    pp(out)


def nav():
    """navhard near-pose pairs (rvr_nav_op.py): syn vs real at the same timestamp (pose offset < 0.5 m, < 1 deg) against the floor real vs
    next real (0.5 s), lane lines / road edges compensated rigidly for the pose offset (expressed in the other side's camera frame)."""
    f = RUN / "nav_heads.npz"
    if not f.exists():
        return None
    z = np.load(f)
    info = json.loads(str(z["info"]))
    hs = info["heads_slices"]

    def lines(H):
        n = len(H)
        ll = H[:, hs["lane_lines"]:hs["lane_lines"] + 264].reshape(n, 4, 33, 2)[:, 1:3, :, 0]
        re = H[:, hs["road_edges"]:hs["road_edges"] + 132].reshape(n, 2, 33, 2)[:, :, :, 0]
        return ll, re, sig(H[:, hs["lane_lines_prob"]:hs["lane_lines_prob"] + 8][:, 1::2])

    def moved(y, d):
        """y (2, 33) right-positive at X_IDXS in frame B -> y at x = 10 m in frame A, A = B moved by d = (dlon, dlat left, dyaw left deg)."""
        a = np.radians(d[2])
        out = []
        for row in y:
            px, py = X_IDXS - d[0], -row - d[1]
            xa, ya = np.cos(a) * px + np.sin(a) * py, -np.sin(a) * px + np.cos(a) * py
            out.append(-np.interp(10.0, xa, ya))
        return np.array(out)

    L, E, P = {}, {}, {}
    for side in ("syn", "real", "next", "syn_sharp"):
        L[side], E[side], P[side] = lines(z[side])
    res = {}
    for side in ("syn", "syn_sharp"):
        rs_l, rs_e, rr_l, rr_e, rs_p, rr_p, cl = [], [], [], [], [], [], []
        for i, (d, dn) in enumerate(zip(info["d"], info["d_next"])):
            okl = (P["real"][i, 1:3] > .5) & (P[side][i, 1:3] > .5)
            okn = (P["real"][i, 1:3] > .5) & (P["next"][i, 1:3] > .5)
            a = np.array([np.interp(10.0, X_IDXS, r) for r in L[side][i]])
            rs_l.append(np.nanmean(np.where(okl, np.abs(a - moved(L["real"][i], d)), np.nan)))
            rs_e.append(np.mean(np.abs(np.array([np.interp(10.0, X_IDXS, r) for r in E[side][i]]) - moved(E["real"][i], d))))
            b = np.array([np.interp(10.0, X_IDXS, r) for r in L["next"][i]])
            rr_l.append(np.nanmean(np.where(okn, np.abs(b - moved(L["real"][i], dn)), np.nan)))
            rr_e.append(np.mean(np.abs(np.array([np.interp(10.0, X_IDXS, r) for r in E["next"][i]]) - moved(E["real"][i], dn))))
            rs_p.append(np.mean(np.abs(P[side][i] - P["real"][i]))), rr_p.append(np.mean(np.abs(P["next"][i] - P["real"][i])))
            cl.append(info["log"][i])
        cl = np.unique(np.array(cl), return_inverse=True)[1]
        r = {}
        for m, num, den in (("lane_y10", rs_l, rr_l), ("edge_y10", rs_e, rr_e), ("lane_p", rs_p, rr_p)):
            num, den = np.array(num), np.array(den)
            ok = np.isfinite(num) & np.isfinite(den)
            r[m] = dict(rs=float(np.median(num[ok])), rr=float(np.median(den[ok])), n=int(ok.sum()), logs=int(len(np.unique(cl[ok]))),
                        ratio=ratio_stat((num[ok], cl[ok]), (den[ok], cl[ok])))
        res[side] = r
    res["n_pairs"] = len(info["d"])
    res["imgstats"] = json.loads((RUN / "nav_imgstats.json").read_text())
    return res


def f3(x):
    return "[%.3g, %.3g]" % (x[1], x[2])


def pp(o):
    print(f"scenes {o['n_scenes']}, verdict gap: {o['verdict_gap']}")
    print("| readout | floor RR | real vs render RS | RS / RR [95% CI] | n frames | meaningful |")
    for m, r in o["gap"].items():
        print(f"| {m} | {r['rr']:.3f} | {r['rs']:.3f} | {r['ratio'][0]:.2f} {f3(r['ratio'])} | {r['n']} | {r['meaningful']} |")
    print("\nsigned shift render - real (median [CI], real level)")
    for m, r in o["shift_render_minus_real"].items():
        print(f"| {m} | {r[0]:+.3f} {f3(r)} | {r[3]:.3f} |")
    print("\nfixes (eval scenes): closure on primaries; harm ratio to floor")
    for arm, d in o["fixes"].items():
        print(arm, " ".join(f"{m}: {d[m]['closure'][0]:+.2f} {f3(d[m]['closure'])}" for m in METRICS[:7]))
    for arm, d in o["harm"].items():
        print(arm, " ".join(f"{m}: {d[m]['ratio_to_floor'][0]:.2f} {f3(d[m]['ratio_to_floor'])}" for m in METRICS[:7]))
    print(json.dumps(o["fix_verdict"]))
    print("spearman PSNR vs gap", o["spearman_psnr_vs_gap"])
    for k, v in o["probes"].items():
        print(k, v)
    if "env" in o:
        print("env rig:", {m: (round(r["real_env"], 3), round(r["render_env"], 3), round(r["ratio_real_env"][0], 2), f3(r["ratio_real_env"])) for m, r in o["env"].items()})
        print("env - real shifts:", {m: (round(r[0], 3), f3(r)) for m, r in o["shift_env_minus_real"].items()})
    print("phase-matched floor:", {m: (round(r["rr3"], 3), round(r["ratio"][0], 2), f3(r["ratio"])) for m, r in o["gap_phase_floor"].items()})
    print("nav:", json.dumps(o["nav"], indent=0, default=float)[:3000])


if __name__ == "__main__":
    main()
