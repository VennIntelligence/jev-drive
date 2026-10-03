#!/usr/bin/env python
"""Lane C (history quality) readouts (plans/2026-10-04-history-quality-prereg.md). navsim2 env, CPU.

PDMS per arm on the navtest subset from the official per-token CSVs, paired deltas (token bootstrap, log-cluster
bootstrap); history-yaw G per speed bin from the rotL / rotR probes; plan speed, lane width, plan drift against the real
arm; Y-PSNR of the synthesized context frames against the real ones; the same plan readouts on the navhard stage-1 subset;
timing checks of the real frames; the review figure. Output: experiments/skill_pack/results/history_quality/ +
experiments/skill_pack/figs/history_quality.jpg.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(Path(__file__).resolve().parent)]
from jevdrive import op_interp as I  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

D = data_dir()
P = D / "runs" / "op_lb"
OUT = REPO / "experiments/skill_pack/results/history_quality"
FIG = REPO / "experiments/skill_pack/figs/history_quality.jpg"
ARMS = {"hold": "hold@cinque", "warp": "vh187@cinque", "gimm": "gimm@cinque", "real": "real@cinque"}
SUBS = ["no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort"]
BINS = [("stop", 0, .5), ("0.5-3", .5, 3), ("3-8", 3, 8), (">8", 8, 99)]
B = 10000
rng = np.random.default_rng(0)


NT, NH = "lb_hq_navtest", "lb_hq_navhard1"


def pdms(stem):
    d = D / "runs/navsim/eval" / f"v1_navtest_opi_{NT}_{stem.replace('@', '-')}__base"
    f = sorted(d.glob("*/*.csv"))[-1]
    df = pd.read_csv(f)
    return df[df.token != "average"].set_index("token")


def boot(x, cl=None):
    """mean, token-bootstrap 95% CI, log-cluster-bootstrap 95% CI."""
    x = np.asarray(x, float)
    tb = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    out = [x.mean(), *np.percentile(tb, [2.5, 97.5])]
    if cl is not None:
        codes, u = pd.factorize(cl)
        sums = np.bincount(codes, x)
        cnt = np.bincount(codes)
        pick = rng.integers(0, len(u), (B, len(u)))
        cb = sums[pick].sum(1) / cnt[pick].sum(1)
        out += list(np.percentile(cb, [2.5, 97.5]))
    return [round(float(v), 3) for v in out]


def plans(data, stem):
    z = np.load(P / data / "plans" / f"{stem}.npz")
    return z


def psi3(z):
    return np.degrees([np.interp(3.0, I.T_IDXS, y) for y in z["plan_yaw"]])


def at(z, t):
    return np.stack([[np.interp(t, I.T_IDXS, p[:, k]) for k in range(2)] for p in z["plan_pos"]])


def plan_readouts(data, arms, mt):
    v = np.array(mt["speed"])
    res = {}
    ref = plans(data, ARMS["real"])
    for a in arms:
        z = plans(data, ARMS[a])
        assert z["names"].tolist() == mt["names"]
        pv0 = z["plan_vel"][:, 0, 0]
        mv = v > 3
        r = dict(speed_ratio_med=round(float(np.median(pv0[mv] / v[mv])), 4),
                 x4_mean=round(float(at(z, 4.0)[:, 0].mean()), 3))
        if a != "real":
            d4 = np.linalg.norm(at(z, 4.0) - at(ref, 4.0), axis=1)
            dpsi = np.abs(psi3(z) - psi3(ref))
            r |= dict(drift4_vs_real_mean=round(float(d4.mean()), 3), drift4_vs_real_med=round(float(np.median(d4)), 3),
                      dpsi3_vs_real_mean_deg=round(float(dpsi.mean()), 3),
                      d_speed_ratio_vs_real=boot(pv0[mv] / v[mv] - ref["plan_vel"][mv, 0, 0] / v[mv]))
        g = {}
        fl, fr = P / data / "plans" / f"{ARMS[a]}_al-rotL.npz", P / data / "plans" / f"{ARMS[a]}_al-rotR.npz"
        if fl.exists() and fr.exists():
            G = (psi3(np.load(fl)) - psi3(np.load(fr))) / 2
            for name, lo, hi in BINS:
                m = (v >= lo) & (v < hi)
                g[name] = dict(n=int(m.sum()), G=boot(G[m]) if m.sum() > 2 else None)
            r["G_deg"] = g
            r["_G"] = G
        res[a] = r
    if all("_G" in res[a] for a in ("gimm", "real")):
        dG = res["gimm"]["_G"] - res["real"]["_G"]
        res["G_gimm_minus_real"] = {name: boot(dG[(v >= lo) & (v < hi)]) for name, lo, hi in BINS if ((v >= lo) & (v < hi)).sum() > 2}
        if "_G" in res.get("warp", {}):
            dW = res["warp"]["_G"] - res["real"]["_G"]
            res["G_warp_minus_real"] = {name: boot(dW[(v >= lo) & (v < hi)]) for name, lo, hi in BINS if ((v >= lo) & (v < hi)).sum() > 2}
    for a in arms:
        res[a].pop("_G", None)
    return res


def lane_ratio(data, mt):
    """decision 104's lane-width ratio (model ego-lane width / map) on the subset, per arm."""
    from edge_height_eval import model_cols
    S = pd.read_pickle(D / "runs/skill_pack/edge_diag/sections_navtest.pkl")
    out = {}
    for a, stem in ARMS.items():
        M = model_cols(P / data / "plans" / f"{stem}.npz").merge(S[["token", "x", "inter", "ego0L", "ego0R", "hum_y"]], on=["token", "x"])
        lv = (~M.inter) & M.ego0L.notna() & (M.p1 > .5) & (M.p2 > .5) & ~(M.hum_y.abs() > 1)
        out[a] = dict(n_sections=int(lv.sum()), lane_ratio_med=round(float((M.ow / (M.ego0L - M.ego0R))[lv].median()), 4))
    return out


def _warp_row(args):
    """the vh187 arm's context frames (op_lb._vcam_job at the true height: ground plane at CAM_F0 z + 0.35 m)."""
    keys, pose, vel, cam = args
    cam = [cam[0], cam[1], cam[2] + 0.35]
    return I.synth_cpu(keys, "warp", np.array([-1.4, -1.2, -0.8, -0.6, -0.4, -0.2]), I.track_navsim(pose, vel), cam)


def psnr(data, mt):
    import op_lb as L
    keys = L.Keys(data)
    real = np.load(P / data / "real.npy", mmap_mode="r")
    arms = {a: np.load(P / data / f"{a}.npy", mmap_mode="r") for a in ("hold", "gimm")}
    with ProcessPoolExecutor(32) as ex:
        arms["warp"] = np.stack(list(ex.map(_warp_row, ((keys[i], mt["pose"][i], mt["vel"][i], mt["cam"][i]) for i in range(len(keys))), chunksize=4)))
    res = {}
    for a, X in arms.items():
        r = {}
        for v, view in enumerate(("road", "wide")):
            ya = np.stack([I.unpack(np.asarray(real[:, j, v]))[0] for j in range(6)], 1).astype(np.float32)
            yb = np.stack([I.unpack(np.asarray(X[:, j, v]))[0] for j in range(6)], 1).astype(np.float32)
            mse = ((ya - yb) ** 2).mean((2, 3))
            p = 10 * np.log10(255 ** 2 / np.maximum(mse, 1e-6))
            r[view] = dict(psnr_mean=round(float(p.mean()), 2), by_t=[round(float(x), 2) for x in p.mean(0)])
        res[a] = r
    return res, arms["warp"]


def figure(tok_i, mt, warp, pd_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import op_lb as L
    keys = L.Keys(NT)[tok_i]
    fr = {"hold": np.load(P / NT / "hold.npy", mmap_mode="r")[tok_i], "warp": warp[tok_i],
          "gimm": np.load(P / NT / "gimm.npy", mmap_mode="r")[tok_i], "real": np.load(P / NT / "real.npy", mmap_mode="r")[tok_i]}
    ts = [-1.4, -1.2, -0.8, -0.6, -0.4, -0.2]
    show = [1, 3, 5]                                  # t = -1.2, -0.6, -0.2 (one per keyframe gap) + the t0 key
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(4, 5, width_ratios=[1, 1, 1, 1, 1.15], wspace=0.04, hspace=0.12)
    for r, a in enumerate(fr):
        for c, j in enumerate(show + [None]):
            ax = fig.add_subplot(gs[r, c])
            y = I.unpack(np.asarray(keys[3][0] if j is None else fr[a][j][0]))[0]
            ax.imshow(y[40:200, 64:448], cmap="gray", vmin=0, vmax=255, aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title("t0 key (same in every arm)" if j is None else f"t = {ts[j]:+.1f} s", fontsize=11)
            if c == 0:
                ax.set_ylabel(f"{a}\nPDMS {pd_rows[a]:.0f}", fontsize=12)
    ax = fig.add_subplot(gs[:, 4])
    cols = {"hold": "#999999", "warp": "#e69f00", "gimm": "#0072b2", "real": "#009e73"}
    for a, stem in ARMS.items():
        p = plans(NT, stem)["plan_pos"][tok_i]
        m = I.T_IDXS <= 4.0
        ax.plot(-p[m, 1], p[m, 0], color=cols[a], lw=2.2, label=a)
    ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend(loc="upper left")
    ax.set_xlabel("left (m)"); ax.set_ylabel("forward (m)"); ax.set_title("plan to 4 s (camera frame)")
    fig.suptitle(f"navtest {mt['names'][tok_i]} (v0 {mt['speed'][tok_i]:.1f} m/s): road-view luma of 3 of the 6 synthesized context frames per history arm, and the plans", fontsize=12)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=80, bbox_inches="tight", pil_kwargs={"quality": 85})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    need = json.loads((P / "hq/need.json").read_text())
    dt = np.array([v["dt"] for v in need.values()])
    kdt = np.array([[x if x is not None else np.nan for x in v["key_dt"]] for v in need.values()], float)
    res["timing"] = dict(n_tokens=len(need), ctx_dt_err_max_ms=round(float(np.abs(dt - np.array([-1.4, -1.2, -0.8, -0.6, -0.4, -0.2])).max() * 1e3), 1),
                         key_dt_err_max_ms=round(float(np.nanmax(np.abs(kdt)) * 1e3), 1),      # need.json stores the error already
                         key_missing=int(np.isnan(kdt).sum()))
    mt = json.loads((P / NT / "meta.json").read_text())
    from jevdrive import navsim_zs as Z
    logs = {e["token"]: e["log_name"] for e in Z.load_index("navtest", slim=True)}
    S = {a: pdms(s) for a, s in ARMS.items()}
    toks = sorted(set.intersection(*[set(s.index) for s in S.values()]))
    cl = [logs[t] for t in toks]
    res["n_scored"] = len(toks)
    res["n_logs"] = len(set(cl))
    res["pdms"] = {a: dict(PDMS=round(100 * float(s.loc[toks, "score"].mean()), 2),
                           **{k[:3]: round(100 * float(s.loc[toks, k].mean()), 2) for k in SUBS}) for a, s in S.items()}
    pairs = [("real", "gimm"), ("gimm", "warp"), ("warp", "hold"), ("real", "warp"), ("real", "hold"), ("gimm", "hold")]
    res["paired"] = {}
    for a, b in pairs:
        d = 100 * (S[a].loc[toks, "score"].values - S[b].loc[toks, "score"].values)
        bt = boot(d, cl)
        res["paired"][f"{a}-{b}"] = dict(delta=bt[0], ci_token=bt[1:3], ci_log=bt[3:5],
                                         subs={k[:3]: round(100 * float((S[a].loc[toks, k] - S[b].loc[toks, k]).mean()), 2) for k in SUBS})
    # recovery of the hold -> real gap, as the WOD table of research/openpilot-openloop-integration.md
    gap = res["pdms"]["real"]["PDMS"] - res["pdms"]["hold"]["PDMS"]
    res["recovered"] = {a: round((res["pdms"][a]["PDMS"] - res["pdms"]["hold"]["PDMS"]) / gap, 3) for a in ("warp", "gimm")}
    # by speed / command (real - gimm)
    v = pd.Series(mt["speed"], index=mt["names"])
    cmd = pd.Series(mt["cmd"], index=mt["names"])
    d = pd.Series(100 * (S["real"].loc[toks, "score"].values - S["gimm"].loc[toks, "score"].values), index=toks)
    res["real-gimm_by"] = {}
    for name, lo, hi in BINS:
        m = d[(v[toks] >= lo) & (v[toks] < hi)]
        res["real-gimm_by"][f"v {name}"] = dict(n=len(m), delta=boot(m.values) if len(m) > 2 else None)
    for c, name in enumerate(("left", "straight", "right")):
        m = d[cmd[toks] == c]
        res["real-gimm_by"][name] = dict(n=len(m), delta=boot(m.values) if len(m) > 2 else None)
    res["plan"] = plan_readouts(NT, list(ARMS), mt)
    res["lane"] = lane_ratio(NT, mt)
    res["psnr"], warp = psnr(NT, mt)
    mh = json.loads((P / NH / "meta.json").read_text())
    res["navhard_stage1"] = dict(n=len(mh["names"]), plan=plan_readouts(NH, ["gimm", "warp", "real"], mh))
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    pd.DataFrame({a: S[a].loc[toks, "score"] for a in ARMS}).assign(log=cl).to_csv(OUT / "per_token_pdms.csv")
    # figure token (fixed in advance): the moving token with the largest real - gimm PDMS gain
    mv = [t for t in toks if v[t] > 3]
    t_fig = max(mv, key=lambda t: (d[t], -abs(d[t])))
    figure(mt["names"].index(t_fig), mt, warp, {a: 100 * float(S[a].loc[t_fig, "score"]) for a in ARMS})
    res["figure_token"] = t_fig
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("timing", "n_scored", "n_logs", "pdms", "paired", "recovered")}, indent=1))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--nt", default=NT)
    ap.add_argument("--nh", default=NH)
    ap.add_argument("--tag", default="", help="output subdir / figure suffix")
    a = ap.parse_args()
    NT, NH = a.nt, a.nh
    if a.tag:
        OUT = OUT / a.tag
        FIG = FIG.with_name(f"history_quality_{a.tag}.jpg")
    main()
