"""op_parity self-consist (plans/2026-10-08-self-consist-prereg.md): does the model's own road-edge / lead output predict the devkit's DAC / NC-TTC failures?

  python sc_analyze.py calib    navtrain-fitted (s, b) per model: model edge vs the SDF-raster drivable boundary at x = 5..30 m
  python sc_analyze.py eval     navtest margins (own edge raw / calibrated / map SDF / shuffle control / plan-only), AUC + recall at 10 % FPR, lead clearance
  python sc_analyze.py figs     figures from results/self_consist/*.csv

CPU only, inside jevdrive.run.Run. Inputs: sc_infer.py outputs (fine-tunes), the stored Cinque heads, bench preds (rear-axle poses), bench units.csv.
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R)]
import argparse, json  # noqa: E401,E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.ndimage import map_coordinates  # noqa: E402

from jevdrive.bench import tables as BT  # noqa: E402
from jevdrive.bench.models import data_dir  # noqa: E402
from jevdrive.run import Run  # noqa: E402

D = data_dir()
INF = D / "runs/op_parity/self_consist/infer"
CR = D / "runs/op_parity/cache"
OUT = _R / "experiments/op_parity/results/self_consist"
FIG = _R / "experiments/op_parity/figs/self_consist"
X_IDXS = 192.0 * (np.arange(33) / 32) ** 2
X0, Y0, RES = -8.0, -24.0, 0.5                                  # label raster (rear-axle frame, jevdrive lib/drivable_hinge.py)
CORNERS = np.array([[4.049, 1.1485], [4.049, -1.1485], [-1.127, 1.1485], [-1.127, -1.1485]])   # nuPlan Pacifica, lib/drivable_hinge.CORNERS
FRONT = 4.049
XS = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
SEEDS = (0, 1)
ARMS = ["cinque", "P2H10-F-s0", "P2H10-F-s1", "SH30-F-s0", "SH30-F-s1"]
FAMILY = {"cinque": ["cinque"], "P2H10": ["P2H10-F-s0", "P2H10-F-s1"], "SH30": ["SH30-F-s0", "SH30-F-s1"],
          "P2H10+SH30": ["P2H10-F-s0", "P2H10-F-s1", "SH30-F-s0", "SH30-F-s1"]}
POSES = {"cinque": D / "runs/bench/ol/lb_navtest/preds/cinque-gimm__base.npz",
         **{f"SH30-F-s{s}": D / f"runs/bench/ol/lb_navtest/preds/SH30-F-s{s}-warp__base.npz" for s in SEEDS},
         **{f"P2H10-F-s{s}": D / f"runs/op_lb/lb_navtest/preds/warp-cinque_PPP2H10-F-s{s}__base.npz" for s in SEEDS}}
TABLE = {"cinque": "cinque@gimm"}
BOOT, NMIN = 1000, 20


# ---------------------------------------------------------------- geometry
def t_dense():
    return np.arange(0, 41) * 0.1


def interp_matrix():
    t = np.r_[0, np.arange(1, 9) * 0.5]
    M = np.zeros((41, 9))
    for k, tt in enumerate(t_dense()):
        j = min(np.searchsorted(t, tt, side="right") - 1, 7)
        f = (tt - t[j]) / (t[j + 1] - t[j])
        M[k, j], M[k, j + 1] = 1 - f, f
    return M


def footprint(poses):
    """(N, 8, 3) rear-axle poses -> corner points (N, 41, 4, 2) on the 0.1 s grid (the hinge's interpolation)."""
    P0 = np.concatenate([np.zeros_like(poses[:, :1]), poses], 1)
    d = np.einsum("kj,njc->nkc", interp_matrix(), P0)
    c, s = np.cos(d[..., 2:3]), np.sin(d[..., 2:3])
    cx = d[..., None, 0] + c * CORNERS[None, None, :, 0] - s * CORNERS[None, None, :, 1]
    cy = d[..., None, 1] + s * CORNERS[None, None, :, 0] + c * CORNERS[None, None, :, 1]
    return np.stack([cx, cy], -1)


def sdf_at(sdf, xy):
    """sdf (N, 128, 96) float, xy (N, ..., 2) rear-axle metres -> bilinear samples, border clamp (grid_sample align_corners=False)."""
    out = np.empty(xy.shape[:-1], np.float32)
    for i in range(len(sdf)):
        r = (xy[i, ..., 0] - X0) / RES - 0.5
        c = (xy[i, ..., 1] - Y0) / RES - 0.5
        out[i] = map_coordinates(sdf[i].astype(np.float32), [r.ravel(), c.ravel()], order=1, mode="nearest").reshape(r.shape)
    return out


def edges_ego(re_mu, cam):
    """road_edges (N, 2, 33, 2) [y, z] -> ego-frame (ex (N, 33), ey (N, 2, 33)) left +, as offroad_roadedge.py."""
    return X_IDXS[None] + cam[:, :1], cam[:, 1, None, None] - re_mu[..., 0]


def edge_margin(corners, ex, ey, s=1.0, b=0.0):
    """Own-edge margin per token: (m, mL, mR, usable). corners (N, 41, 4, 2). ey (N, 2, 33); calibration ey' = s ey (+b left, -b right)."""
    n = len(corners)
    yl, yr = s * ey[:, 0] + b, s * ey[:, 1] - b
    mL, mR, ok = np.empty(n), np.empty(n), np.zeros(n, bool)
    for i in range(n):
        cx, cy = corners[i, ..., 0].ravel(), corners[i, ..., 1].ravel()
        mL[i] = np.min(np.interp(cx, ex[i], yl[i]) - cy)
        mR[i] = np.min(cy - np.interp(cx, ex[i], yr[i]))
        k = np.searchsorted(ex[i], cx.max(), side="left") + 1                       # edge points up to and including the first beyond the footprint
        k = min(k, 33)
        w = yl[i, :k] - yr[i, :k]
        ok[i] = bool(np.isfinite(yl[i, :k]).all() and np.isfinite(yr[i, :k]).all() and (w >= 2.5).all())
    return np.minimum(mL, mR), mL, mR, ok


def boundary_from_sdf(sdf, xs):
    """(N, 128, 96) SDF -> (N, len(xs), 2) [left, right] zero crossing around y = 0 at x = xs (NaN if the centre line is outside / no crossing)."""
    ys = Y0 + (np.arange(96) + 0.5) * RES
    out = np.full((len(sdf), len(xs), 2), np.nan)
    for k, x in enumerate(xs):
        r = (x - X0) / RES - 0.5
        r0 = int(np.floor(r)); f = r - r0
        P = ((1 - f) * sdf[:, r0].astype(np.float32) + f * sdf[:, r0 + 1].astype(np.float32))      # (N, 96)
        c0 = (P[:, 47] + P[:, 48]) / 2
        for side, rng in ((0, np.arange(48, 96)), (1, np.arange(47, -1, -1))):
            Q = P[:, rng]
            neg = Q <= 0
            has = neg.any(1) & (c0 > 0)
            j = neg.argmax(1)
            jj = np.maximum(j, 1)
            q0, q1 = Q[np.arange(len(Q)), jj - 1], Q[np.arange(len(Q)), jj]
            fr = q0 / np.maximum(q0 - q1, 1e-6)
            y0, y1 = ys[rng[jj - 1]], ys[rng[jj]]
            ok = has & (j >= 1)
            out[ok, k, side] = (y0 + fr * (y1 - y0))[ok]
    return out


# ---------------------------------------------------------------- inputs
def load_edges(arm, split):
    """(names, re_mu (N, 2, 33, 2), lead_prob, lead_x, lead_v) of an arm on navtest / the navtrain calibration set."""
    if arm == "cinque":
        f = D / ("runs/bench/ol/lb_navtest/plans/cinque@gimm.npz" if split == "test" else "runs/op_lb/lb_navtrain/plans/gimm@cinque.npz")
        z = np.load(f, allow_pickle=True)
        h = z["heads"]
        ld = h[:, 1907:1907 + 72].reshape(-1, 3, 6, 4)
        return (z["names"], h[:, 536:536 + 132].reshape(-1, 2, 33, 2), z["lead_prob"][:, 0], ld[:, 0, 0, 0], ld[:, 0, 0, 2])
    data = "lb_navtest" if split == "test" else "navtrain_full.s0of12"
    z = np.load(INF / f"{arm.replace('@', '-')}-warp__{data}.npz")
    return z["names"], z["road_edges"], z["lead_prob"], z["lead_x"], z["lead_v"]


def tab_of(split):
    return np.load(CR / ("lb_navtest" if split == "test" else "navtrain_full.s0of12") / "tab.npz")


# ---------------------------------------------------------------- calibration
def huber_fit(ye, ym, side_sign, delta=1.0, it=30):
    """y_map = s ye + sign b, Huber IRLS; ye, ym, side_sign (+1 left, -1 right) are flat arrays."""
    A = np.stack([ye, side_sign], 1)
    w = np.ones(len(ye))
    for _ in range(it):
        sw = np.sqrt(w)
        th = np.linalg.lstsq(A * sw[:, None], ym * sw, rcond=None)[0]
        r = np.abs(ym - A @ th)
        w = np.where(r <= delta, 1.0, delta / np.maximum(r, 1e-9))
    r = ym - A @ th
    return float(th[0]), float(th[1]), float(np.median(np.abs(r - np.median(r))) * 1.4826), len(ye)


def cmd_calib(a):
    with Run("self_consist", "calib", seed=0, config=vars(a)) as run:
        lab = np.load(D / "runs/op_probe/labels/navtrain_all.npz")
        pos = {t: i for i, t in enumerate(lab["tokens"].tolist())}
        res, pairs = {}, {}
        for arm in ARMS:
            names, re_mu, *_ = load_edges(arm, "train")
            tab = np.load(CR / ("lb_navtrain" if arm == "cinque" else "navtrain_full.s0of12") / "tab.npz")
            assert (tab["names"][: len(names)] == names).all() or set(names) <= set(tab["names"].tolist())
            tpos = {t: i for i, t in enumerate(tab["names"].tolist())}
            keep = [i for i, t in enumerate(names.tolist()) if t in pos and t in tpos]
            cam = tab["cam"][[tpos[names[i]] for i in keep]]
            sdf = lab["sdf"][[pos[names[i]] for i in keep]]
            ex, ey = edges_ego(re_mu[keep], cam)
            bm = boundary_from_sdf(sdf, XS)                                           # (n, 6, 2) map [left, right]
            em = np.stack([np.stack([np.interp(XS, ex[i], ey[i, s]) for i in range(len(keep))]) for s in (0, 1)], -1)   # (n, 6, 2)
            ok = np.isfinite(bm).all(-1) & (np.abs(bm) < 12).all(-1)
            ye = np.concatenate([em[..., 0][ok], em[..., 1][ok]])
            ym = np.concatenate([bm[..., 0][ok], bm[..., 1][ok]])
            sg = np.concatenate([np.ones(ok.sum()), -np.ones(ok.sum())])
            s, b, mad, n = huber_fit(ye, ym, sg)
            raw_mad = float(np.median(np.abs((ym - ye) - np.median(ym - ye))) * 1.4826)
            res[arm] = dict(s=s, b=b, resid_mad_m=mad, n_pairs=n, n_tokens=len(keep), raw_median_offset=float(np.median(ym - ye)),
                            raw_ratio_median=float(np.median(np.abs(ym) / np.maximum(np.abs(ye), 0.5))))
            pairs[arm] = (ye[::5], ym[::5], sg[::5])
            run.info(f"{arm}: s {s:.3f} b {b:.3f} resid MAD {mad:.2f} m  pairs {n}  tokens {len(keep)}")
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "calibration.json").write_text(json.dumps(res, indent=1))
        np.savez(OUT / "calib_pairs.npz", **{f"{k}_{n}": v for k, p in pairs.items() for n, v in zip(("ye", "ym", "side"), p)})
        run.summary.update(res)


# ---------------------------------------------------------------- metrics
def prep(score, y):
    """Sort order and tie groups of one (score, label) set; reused over every bootstrap weight vector."""
    o = np.argsort(score, kind="stable")
    sc = score[o]
    return o, np.concatenate([[0], np.cumsum(sc[1:] != sc[:-1])]), y[o]


def auc_tpr(P, w, fpr=0.10):
    """Weighted AUC (ties 1/2) and recall at the threshold that lets `fpr` of the negatives through (score high = risky); P = prep(...)."""
    o, gid, ys = P
    wo = w[o]
    wp, wn = wo * (ys == 1), wo * (ys == 0)
    if wp.sum() == 0 or wn.sum() == 0:
        return np.nan, np.nan
    Wn, Wp = np.bincount(gid, wn), np.bincount(gid, wp)
    auc = float((Wp * (np.cumsum(Wn) - Wn + 0.5 * Wn)).sum() / (Wp.sum() * Wn.sum()))
    cum_wn_hi = np.cumsum(wn[::-1])                                                  # negatives at or above each position, from the top
    k = np.searchsorted(cum_wn_hi, fpr * wn.sum(), side="right")                     # number of top positions with <= fpr negatives
    tpr = float(np.cumsum(wp[::-1])[k - 1] / wp.sum()) if k > 0 else 0.0
    return auc, tpr


class Boot:
    """Log-cluster bootstrap multiplicities, shared by every metric (and by the seeds of a family)."""

    def __init__(self, logs, B=BOOT, seed=0):
        u, self.inv = np.unique(logs, return_inverse=True)
        self.W = np.random.default_rng(seed).multinomial(len(u), np.ones(len(u)) / len(u), size=B).astype(float)

    def run(self, scores, ys, mask):
        """scores / ys: lists (one per arm of a family); metric = mean over the arms. -> dict auc, tpr, lo/hi."""
        m = np.flatnonzero(mask)
        inv = self.inv[m]
        Ps = [prep(s[m], y[m]) for s, y in zip(scores, ys)]
        pt = np.nanmean([auc_tpr(P, np.ones(len(m))) for P in Ps], 0)
        bs = np.empty((len(self.W), 2))
        for b in range(len(self.W)):
            w = self.W[b][inv]
            bs[b] = np.nanmean([auc_tpr(P, w) for P in Ps], 0)
        q = np.nanpercentile(bs, [2.5, 97.5], axis=0)
        return dict(auc=pt[0], auc_lo=q[0, 0], auc_hi=q[1, 0], tpr10=pt[1], tpr_lo=q[0, 1], tpr_hi=q[1, 1])


def cmd_eval(a):
    cal = json.loads((OUT / "calibration.json").read_text())
    st = BT.navtest_strata()
    tab = tab_of("test")
    names = tab["names"]
    cam, v0 = tab["cam"], tab["speed"]
    dy = st.reindex(names).dyaw.to_numpy()
    ady = np.abs(dy)
    logs = tab["log"].astype(str)
    lab = np.load(D / "runs/op_probe/labels/navtest.npz")
    assert (lab["tokens"] == names).all()
    sdf = lab["sdf"]
    bk = {"all": np.ones(len(names), bool), "straight <5": ady < 5, "5-20": (ady >= 5) & (ady < 20), ">20": ady > 20, "20-45": (ady > 20) & (ady <= 45),
          ">45": ady > 45}
    bk_dir = {**{f"{k} left": v & (dy > 0) for k, v in bk.items() if k in (">20", "20-45", ">45")},
              **{f"{k} right": v & (dy < 0) for k, v in bk.items() if k in (">20", "20-45", ">45")}}
    boot = Boot(logs)
    rows, cover, tokrows = [], [], {}
    rng = np.random.default_rng(0)
    with Run("self_consist", "eval", seed=0, config=vars(a)) as run:
        per = {}
        for arm in ARMS:
            z = np.load(POSES[arm])
            assert (z["tokens"] == names).all()
            poses = z["poses"].astype(np.float64)
            u = BT.load("navtest", TABLE.get(arm, arm))[0].reindex(names)
            assert not u.score.isna().any(), arm
            y_dac = (u.DAC.to_numpy() < 1).astype(int)
            y_col = ((u.NC.to_numpy() < 1) | (u.TTC.to_numpy() < 1)).astype(int)
            nm, re_mu, lp, lx, lv = load_edges(arm, "test")
            assert (nm == names).all(), arm
            ex, ey = edges_ego(re_mu, cam)
            cor = footprint(poses)
            c = cal[arm]
            m_raw = edge_margin(cor, ex, ey)
            m_cal = edge_margin(cor, ex, ey, c["s"], c["b"])
            m_map = sdf_at(sdf, cor).reshape(len(names), -1).min(1)
            # shuffle control: the road-edge read of another token of the same turn class (plan unchanged), calibrated
            cls = np.digitize(ady, [5, 20, 45])
            perm = np.arange(len(names))
            for k in np.unique(cls):
                i = np.flatnonzero(cls == k)
                perm[i] = rng.permutation(i)
            m_shuf = edge_margin(cor, ex[perm], ey[perm], c["s"], c["b"])
            # plan-only baselines
            yaw_end = np.abs(np.degrees(np.unwrap(np.concatenate([np.zeros((len(poses), 1)), poses[:, :, 2]], 1), axis=1)[:, -1]))
            dist = np.linalg.norm(poses[:, 7, :2], axis=1)
            # lead clearance
            t8 = np.arange(0, 9) * 0.5
            px = np.concatenate([np.zeros((len(poses), 1)), poses[:, :, 0]], 1) + FRONT
            has = lp > 0.5
            def clear(lead_x):
                g = (lead_x[:, None] + cam[:, :1] + np.maximum(lv, -50)[:, None] * t8[None]) - px
                return np.where(has, g.min(1), 1e3)
            bias = np.clip(2.0 * (10 - lx) / 4.0, 0, 2.0)                           # decision 157: 2.0 m below 6 m, linear to 0 at 10 m
            cl_raw, cl_fix = clear(lx), clear(lx - bias)
            per[arm] = dict(y_dac=y_dac, y_col=y_col, score_ok=dict(raw=m_raw[3], cal=m_cal[3], map=np.ones(len(names), bool), shuf=m_shuf[3], plan=np.ones(len(names), bool), side=m_cal[3]),
                            risk=dict(raw=-m_raw[0], cal=-m_cal[0], map=-m_map, shuf=-m_shuf[0], plan_yaw=yaw_end, plan_dist=dist, speed=v0),
                            risk_side=dict(rawL=-m_raw[1], rawR=-m_raw[2], calL=-m_cal[1], calR=-m_cal[2]),
                            col=dict(clear_raw=-cl_raw, clear_fix=-cl_fix, speed=v0, has=has))
            tokrows[arm] = pd.DataFrame(dict(token=names, dac_fail=y_dac, col_fail=y_col, m_raw=m_raw[0], m_cal=m_cal[0], m_map=m_map, ok_raw=m_raw[3], ok_cal=m_cal[3],
                                             mL_cal=m_cal[1], mR_cal=m_cal[2], clear_raw=cl_raw, clear_fix=cl_fix, has_lead=has, lead_x=lx))
            run.info(f"{arm}: DAC fail {y_dac.mean():.3%}, NC|TTC fail {y_col.mean():.3%}, usable raw {m_raw[3].mean():.3f} cal {m_cal[3].mean():.3f}")
        pd.concat(tokrows, names=["arm"]).reset_index(level=0).to_parquet(OUT / "tokens.parquet", index=False)

        def fam_rows(fam, arms, target, variant, key, bks, extra_mask=None, pred_from="risk"):
            for bname, bmask in bks.items():
                y = [per[x][target] for x in arms]
                sc = [per[x][pred_from][key] for x in arms]
                ok = np.ones(len(names), bool)
                for x in arms:
                    ok &= per[x]["score_ok"].get(variant, np.ones(len(names), bool))
                mk = bmask & ok
                if extra_mask is not None:
                    mk = mk & np.logical_and.reduce([extra_mask(per[x]) for x in arms])
                if mk.sum() < NMIN or min(np.asarray(yy)[mk].sum() for yy in y) < 5:
                    continue
                r = boot.run(sc, y, mk)
                rows.append(dict(family=fam, target=target.replace("y_", ""), variant=variant if variant in ("raw", "cal", "map", "shuf", "all-tokens") else key, bucket=bname, n=int(mk.sum()),
                                 n_fail=float(np.mean([np.asarray(yy)[mk].sum() for yy in y])), coverage=float(mk.sum() / max(bmask.sum(), 1)), **r))

        for fam, arms in FAMILY.items():
            for variant, key in (("raw", "raw"), ("cal", "cal"), ("map", "map"), ("shuf", "shuf"), ("plan", "plan_yaw"), ("plan", "plan_dist"), ("plan", "speed")):
                fam_rows(fam, arms, "y_dac", variant, key, {**bk, **bk_dir})
            for k in ("rawL", "rawR", "calL", "calR"):                                  # per-side diagnostics: left turns -> L margin, right turns -> R margin
                fam_rows(fam, arms, "y_dac", "side", k, {b: m for b, m in bk_dir.items() if (("left" in b) == k.endswith("L"))}, pred_from="risk_side")
            # all-token sensitivity: unusable tokens counted as the most risky
            for arm_key in ("cal",):
                for x in arms:
                    per[x]["risk"]["cal_unusable_risky"] = np.where(per[x]["score_ok"]["cal"], per[x]["risk"]["cal"], 1e3)
                fam_rows(fam, arms, "y_dac", "all-tokens", "cal_unusable_risky", bk)
            # collisions
            for key in ("clear_raw", "clear_fix", "speed"):
                fam_rows(fam, arms, "y_col", "lead", key, bk, pred_from="col")
            for key in ("clear_raw", "clear_fix"):
                for x in arms:
                    per[x]["col"][key + "_has"] = per[x]["col"][key]
                fam_rows(fam, arms, "y_col", "lead|has lead", key + "_has", {"all": bk["all"]}, extra_mask=lambda p: p["col"]["has"], pred_from="col")
            run.info(f"family {fam} done")
        df = pd.DataFrame(rows)
        # coverage table
        for arm in ARMS:
            for bname, bmask in {**bk, **bk_dir}.items():
                cover.append(dict(arm=arm, bucket=bname, n=int(bmask.sum()), usable_raw=float(per[arm]["score_ok"]["raw"][bmask].mean()),
                                  usable_cal=float(per[arm]["score_ok"]["cal"][bmask].mean())))
        df.to_csv(OUT / "metrics.csv", index=False)
        pd.DataFrame(cover).to_csv(OUT / "coverage.csv", index=False)
        run.summary.update(n_rows=len(df))


def cmd_figs(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = pd.read_csv(OUT / "metrics.csv")
    d = d[(d.target == "dac") & (d.family == "P2H10+SH30")]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    bks = [">20", "20-45", ">45", ">20 left", ">20 right"]
    vs = [("map", "map SDF margin (hinge's)"), ("raw", "own edge, raw"), ("cal", "own edge, navtrain-calibrated"), ("shuf", "own edge, shuffled control"), ("plan_yaw", "plan-only: |end heading|")]
    for k, (v, lab) in enumerate(vs):
        r = d[d.variant == v].set_index("bucket").loc[bks]
        x = np.arange(len(bks)) + (k - 2) * 0.16
        ax[0].bar(x, r.auc, 0.16, yerr=[r.auc - r.auc_lo, r.auc_hi - r.auc], label=lab, capsize=2)
    for y in (0.65, 0.75):
        ax[0].axhline(y, color="k", ls=":", lw=1)
    ax[0].set_xticks(range(len(bks)), bks); ax[0].set_ylim(0.4, 1); ax[0].set_ylabel("AUC for DAC failure"); ax[0].legend(fontsize=7, loc="upper right")
    ax[0].set_title("P2H10 + SH30 (4 plans), navtest, 95% log-cluster CI")
    z = np.load(OUT / "calib_pairs.npz"); c = json.loads((OUT / "calibration.json").read_text())["P2H10-F-s0"]
    ye, ym, sg = z["P2H10-F-s0_ye"], z["P2H10-F-s0_ym"], z["P2H10-F-s0_side"]
    ax[1].scatter(sg * ye, sg * ym, s=2, alpha=0.15)
    xx = np.array([0, 12]); ax[1].plot(xx, xx, "k--", lw=1, label="y = x"); ax[1].plot(xx, c["s"] * xx + c["b"], "r", label=f"fit s {c['s']:.2f}, b {c['b']:.2f}")
    ax[1].set_xlim(0, 12); ax[1].set_ylim(0, 12); ax[1].set_xlabel("model edge, distance from centre line (m)"); ax[1].set_ylabel("map drivable boundary (m)"); ax[1].legend(); ax[1].set_title("navtrain calibration set, x = 5..30 m")
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(FIG / "self_consist_auc.png", dpi=130)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calib", "eval", "figs"])
    a = ap.parse_args()
    {"calib": cmd_calib, "eval": cmd_eval, "figs": cmd_figs}[a.cmd](a)
