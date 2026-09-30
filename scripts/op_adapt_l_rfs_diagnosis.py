"""Why did RFS not move? Diagnostic on existing op-adapt L results (todos/2026-10-01-op-adapt-L-rfs-diagnosis.md). CPU only, no training.

  run      slice coverage + headroom (Q1), per-frame RFS response by capture flip (Q2), logged future vs raters (Q3), trust-region
           mechanism of the flips (Q4), hypothesis table (Q5)  ->  research/results/op-adapt-L/rfs-diagnosis/*.csv
  figure   research/figs/op-adapt-L-rfs-diagnosis.png

  CUDA_VISIBLE_DEVICES= taskset -c 100-160 python scripts/op_adapt_l_rfs_diagnosis.py run
Reads the rater-frame plans that the op-adapt L readout stored ($L/readout/{O,main-s*}/eval/rater.npz); nothing is recomputed on a GPU.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from jevdrive import op_adapt as A  # noqa: E402
from jevdrive import op_adapt_l as L  # noqa: E402
from jevdrive import waymo as W  # noqa: E402
from jevdrive import wod_zeroshot as Z  # noqa: E402

OUT = REPO / "research" / "results" / "op-adapt-L" / "rfs-diagnosis"
SEEDS = ["main-s0", "main-s1", "main-s2"]
B, TOP = 2000, 8.17
SLICES = ["start", "stay", "stop", "turn_onset", "in_turn", "nudge", "lane_change", "control", "other", "no_future"]
PRIO = ["turn_onset", "start", "stop", "stay", "in_turn", "nudge", "lane_change", "control", "other", "no_future"]
CAPS = {"start": "cap_start", "stop": "cap_stop", "turn_onset": "cap_turn_onset"}
HOR = np.array(W.RFS_HORIZONS) * W.RFS_FREQ - 1                        # waypoint indices of 3 s / 5 s


# ---------------------------------------------------------------- data
def load():
    import op_adapt_l_prep as P
    z = {m: np.load(L.lroot("readout", m, "eval") / "rater.npz", allow_pickle=True) for m in ["O"] + SEEDS}
    names = z["O"]["names"].astype(str)
    for m in SEEDS:
        assert (z[m]["names"].astype(str) == names).all()
    sets = Z.load_sets()["rater"]
    pos = pd.Series(np.arange(len(sets["name"])), index=sets["name"].astype(str))
    k = pos.reindex(names).to_numpy().astype(int)
    calib = json.loads((Z.root() / "op_calib.json").read_text())
    dev_xy = np.stack([np.array(calib[n.rsplit("-", 1)[0]]["1"]["extrinsic"]).reshape(4, 4)[:2, 3] for n in names])
    d = {"names": names, "seg": np.array([n.rsplit("-", 1)[0] for n in names]), "traj": sets["traj"][k].astype(np.float64),
         "scores": sets["scores"][k].astype(np.float64), "speed": W.init_speed(sets["past"][k]), "cat": sets["cluster"][k].astype(str),
         "plan": {m: z[m]["plan"] for m in z}, "dev_xy": dev_xy}
    kin = P.wod_kin(names)
    fl = P.flags(kin)
    d["kin"], d["has"] = kin, kin["has"].astype(bool)
    d["fut20"] = kin["fut20"].astype(np.float64)
    d["slice"] = {s: fl[s] for s in SLICES if s in fl}
    d["slice"]["no_future"] = ~d["has"]
    d["slice"]["other"] = d["has"] & ~np.any([fl[s] for s in ("start", "stay", "stop", "turn_onset", "in_turn", "nudge", "lane_change", "control")], 0)
    d["slice"]["S"] = fl["start"] | fl["stop"] | fl["turn_onset"]
    d["slice"]["all"] = np.ones(len(names), bool)
    lab = np.full(len(names), "", object)
    for s in PRIO[::-1]:
        lab[d["slice"][s]] = s
    d["label"] = lab
    d["cap"] = {}
    for m in z:
        p8 = L.rear_np(z[m]["plan"], L.CAM_X["wod"])
        f8 = d["kin"]["fut"].astype(np.float32)
        d["cap"][m] = {"cap_start": L.cap_disp(p8, f8), "cap_stop": L.cap_stop(p8), "cap_turn_onset": L.cap_lat_end(p8, f8)}
    return d


def waypoints(d, m, xs):
    p = d["plan"][m]
    wp = np.stack([Z.openpilot_to_wod(p[i, :, 0:3], p[i, :, 11], A.T_IDXS, d["dev_xy"][i]) for i in range(len(p))])[:, :, :2].astype(np.float64)
    wp[..., 0] *= xs
    return wp


# ---------------------------------------------------------------- explicit RFS (official formula, with the geometry exposed)
def geometry(pred, traj, speed):
    """Signed longitudinal / lateral offsets (n, P, 2 horizons) of one candidate to every rater trajectory in that trajectory's own
    frame, and the trust-region thresholds (n, 2) (lateral, longitudinal) as in waymo.rater_feedback_score."""
    lng, lat = W._rater_frames(traj)
    v = pred[:, None] - traj
    sl = (lng * v).sum(-1)[..., HOR]
    st = (lat * v).sum(-1)[..., HOR]
    scale = np.clip(0.5 + 0.5 * (speed - 1.4) / (11 - 1.4), 0.5, 1.0)[:, None]
    base = np.array(W.RFS_BASE_THRESHOLDS)
    lat_thr, lng_thr = (scale * (base * m) for m in W.RFS_MULTIPLIERS)
    return sl, st, lng_thr, lat_thr


def score_from(sl, st, lng_thr, lat_thr, scores):
    nl, nt = np.abs(sl) / lng_thr[:, None], np.abs(st) / lat_thr[:, None]
    norm = np.maximum(nl, nt)
    inside = (norm <= 1.0).all(-1).any(1)
    hv = scores[..., None] * W.RFS_DECAY ** np.maximum(norm - 1.0, 0.0)            # (n, P, 2)
    per = hv.max(1).mean(-1)
    per = np.where(inside, per, np.maximum(per, W.RFS_FLOOR))
    return per, inside, nl, nt, hv


# ---------------------------------------------------------------- cluster bootstrap over segments (one resample for everything)
class Boot:
    def __init__(self, groups, B=B):
        self.gi, self.L, self.W = L.cluster_boot(groups, B)

    def vec(self, x, m):
        """mean and its B bootstrap means over the rows of mask m."""
        x, gi = np.asarray(x, float)[m], self.gi[m]
        s = np.bincount(gi, weights=x, minlength=self.L)
        n = np.bincount(gi, minlength=self.L).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            return float(s.sum() / max(n.sum(), 1)), (self.W @ s) / (self.W @ n)

    def stat(self, x, m):
        if m.sum() == 0:
            return dict(n=0, seg=0, mean=np.nan, lo=np.nan, hi=np.nan)
        mean, v = self.vec(x, m)
        lo, hi = np.nanpercentile(v, [2.5, 97.5])
        return dict(n=int(m.sum()), seg=int(len(np.unique(self.gi[m]))), mean=mean, lo=float(lo), hi=float(hi))

    def diff(self, x1, m1, x2, m2):
        a, va = self.vec(x1, m1)
        b, vb = self.vec(x2, m2)
        lo, hi = np.nanpercentile(va - vb, [2.5, 97.5])
        return dict(mean=a - b, lo=float(lo), hi=float(hi))


# ---------------------------------------------------------------- main
def run(a):
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    n = len(d["names"])
    cats = np.unique(d["cat"])
    C = len(cats)
    cnt = pd.Series(d["cat"]).value_counts()
    w = np.array([1.0 / (C * cnt[c]) for c in d["cat"]])                 # leaderboard weight: sum_i w_i x_i = mean over categories of means
    chk = {"frames": n, "categories": C, "rater_frames_with_future": int(d["has"].sum()), "weights_sum": float(w.sum())}
    bf = Boot(d["seg"])                                                  # frame-level bootstrap
    rows = []                                                            # stacked (frame, seed) rows
    seed_of = np.repeat(np.arange(3), n)
    fi = np.tile(np.arange(n), 3)
    br = Boot(d["seg"][fi])
    res = {}
    for xs in (1.0, 1.06):
        geo, sc, nsl = {}, {}, {}
        for m in ["O"] + SEEDS:
            wp = waypoints(d, m, xs)
            sl, st, lt, at = geometry(wp, d["traj"], d["speed"])
            per, ins, nl, nt, hv = score_from(sl, st, lt, at, d["scores"])
            off = W.rater_feedback_score(wp, d["traj"], d["scores"], d["speed"], details=True)
            assert np.abs(per - off[0]).max() < 1e-9 and (ins == off[1][:, 0]).all(), "explicit RFS differs from waymo.rater_feedback_score"
            geo[m] = dict(sl=sl, st=st, lt=lt, at=at, inside=ins, nl=nl, nt=nt, hv=hv)
            sc[m] = per
        res[xs] = (geo, sc)
        agg = {m: W.rfs_by_cluster(sc[m], d["cat"])[0] for m in sc}
        chk[f"rfs_x{xs}"] = agg
        chk[f"rfs_x{xs}_weighted_equals_leaderboard"] = float(max(abs((w * sc[m]).sum() - agg[m]) for m in sc))
    # cross-check against the stored readout (per-seed RFS deltas in metrics_all.csv)
    ma = pd.read_csv(REPO / "research/results/op-adapt-L/metrics_all.csv")
    ma = ma[(ma.set == "rater") & (ma.slice == "all") & (ma.metric == "rfs")]
    chk["stored_vs_recomputed_delta"] = {f"{m}_x{xs}": [float(ma[(ma.model == m) & (ma.xscale == xs)].delta.iloc[0]) if len(ma[(ma.model == m) & (ma.xscale == xs)]) else None,
                                                          float(chk[f"rfs_x{xs}"][m] - chk[f"rfs_x{xs}"]["O"])] for m in SEEDS for xs in (1.0, 1.06)}
    raw_geo, raw_sc = res[1.0]
    O_gap = None
    gap = d["scores"].max(1) - raw_sc["O"]                               # frame headroom to the best rater label
    # ================= Q1 coverage and headroom
    H_all = float((w * gap).sum())
    q1 = []
    for s in SLICES + ["S", "all"]:
        m = d["slice"][s]
        ex = d["label"] == s
        H = float((w[m] * gap[m]).sum())
        q1.append({"slice": s, "frames": int(m.sum()), "segments": int(len(np.unique(d["seg"][m]))), "frames_exclusive": int(ex.sum()) if s in PRIO else np.nan,
                   "rfs_orig_mean": float(raw_sc["O"][m].mean()) if m.any() else np.nan, "gap_mean": float(gap[m].mean()) if m.any() else np.nan,
                   "H_slice": H, "H_share": H / H_all, "H_over_gap_to_top": H / (TOP - chk["rfs_x1.0"]["O"]),
                   "rater_spread_mean": float((d["scores"][m].max(1) - d["scores"][m].min(1)).mean()) if m.any() else np.nan,
                   "frac_frames": float(m.mean())})
    q1 = pd.DataFrame(q1)
    q1.to_csv(OUT / "q1_coverage.csv", index=False)
    cc = []
    for c in cats:
        m = d["cat"] == c
        row = {"category": c, "frames": int(m.sum()), "rfs_orig": float(raw_sc["O"][m].mean()), "gap_mean": float(gap[m].mean()),
               "headroom_lb_units": float(gap[m].mean() / C), "frames_gap_ge2": int((gap[m] >= 2).sum())}
        for s in PRIO:
            mm = m & (d["label"] == s)
            row[f"gapshare_{s}"] = float(gap[mm].sum() / gap[m].sum())
            row[f"frames_gap_ge2_{s}"] = int(((gap >= 2) & mm).sum())
        cc.append(row)
    pd.DataFrame(cc).sort_values("headroom_lb_units", ascending=False).to_csv(OUT / "q1_category_headroom.csv", index=False)
    # ================= Q2 response by capture flip
    q2, contrib, flipinfo = [], [], {}
    for xs in (1.0, 1.06):
        geo, sc = res[xs]
        dR = np.concatenate([sc[m] - sc["O"] for m in SEEDS])            # (3n,) per (seed, frame)
        wr = w[fi] / 3
        # per-slice flip status of every row
        on_any = np.zeros(3 * n, bool); off_any = np.zeros(3 * n, bool); inS = np.zeros(3 * n, bool)
        for s in ("start", "stop", "turn_onset"):
            co = d["cap"]["O"][CAPS[s]]
            ca = np.concatenate([d["cap"][m][CAPS[s]] for m in SEEDS])
            co3 = np.tile(co, 3)
            member = np.tile(d["slice"][s], 3)
            on, off = member & ~co3 & ca, member & co3 & ~ca
            on_any |= on; off_any |= off; inS |= member
            flipinfo[(xs, s)] = dict(on=on, off=off, unch=member & ~on & ~off)
        flipinfo[(xs, "S")] = dict(on=on_any, off=off_any & ~on_any, unch=inS & ~on_any & ~off_any)
        for s in SLICES + ["S", "all"]:
            mem = np.tile(d["slice"][s], 3)
            base = dict(slice=s, xscale=xs)
            q2.append({**base, "flip": "all", **br.stat(dR, mem), "seed_means": json.dumps([float(dR[seed_of == k][d["slice"][s]].mean()) if d["slice"][s].any() else None for k in range(3)])})
            if (xs, s) in flipinfo:
                fl = flipinfo[(xs, s)]
                for k in ("on", "off", "unch"):
                    q2.append({**base, "flip": k, **br.stat(dR, fl[k])})
                for k in ("on", "off"):
                    q2.append({**base, "flip": f"{k}_minus_unch", "n": int(fl[k].sum()), "seg": np.nan, **(br.diff(dR, fl[k], dR, fl["unch"]) if fl[k].any() and fl["unch"].any() else dict(mean=np.nan, lo=np.nan, hi=np.nan))})
        dF = np.mean([sc[m] - sc["O"] for m in SEEDS], 0)
        for s in PRIO:
            m = d["label"] == s
            contrib.append({"xscale": xs, "label": s, "frames": int(m.sum()), "delta_frame_mean": float(dF[m].mean()) if m.any() else np.nan,
                            "contribution_lb": float((w[m] * dF[m]).sum())})
        contrib.append({"xscale": xs, "label": "TOTAL", "frames": n, "delta_frame_mean": float(dF.mean()), "contribution_lb": float((w * dF).sum())})
        chk[f"decomp_sum_minus_total_x{xs}"] = float(sum(r["contribution_lb"] for r in contrib[-len(PRIO) - 1:-1]) - contrib[-1]["contribution_lb"])
    q2 = pd.DataFrame(q2)
    q2.to_csv(OUT / "q2_response.csv", index=False)
    pd.DataFrame(contrib).to_csv(OUT / "q2_contribution.csv", index=False)
    # total delta with CI (frame-level, seed-mean, leaderboard-weighted point estimate; CI by category-stratified cluster resample)
    # ================= Q3 logged future vs raters
    q3 = []
    has = d["has"]
    best = d["traj"][np.arange(n), d["scores"].argmax(1)]
    ties = int(((d["scores"] == d["scores"].max(1, keepdims=True)).sum(1) > 1).sum())
    lng, lat = W._rater_frames(best[:, None])
    dist_rows = []
    for t_i, t in zip((3, 7, 11, 15), (1, 2, 3, 4)):
        e = d["fut20"][:, t_i] - best[:, t_i]
        el = np.abs((lng[:, 0, t_i] * e).sum(-1)); ela = np.abs((lat[:, 0, t_i] * e).sum(-1))
        for s in ["S", "start", "stop", "turn_onset", "all"]:
            m = d["slice"][s] & has
            if m.any():
                dist_rows.append({"slice": s, "t_s": t, "n": int(m.sum()), "dist_med": float(np.median(np.linalg.norm(e, axis=-1)[m])),
                                  "dist_p90": float(np.percentile(np.linalg.norm(e, axis=-1)[m], 90)), "lon_med": float(np.median(el[m])), "lat_med": float(np.median(ela[m]))})
    pd.DataFrame(dist_rows).to_csv(OUT / "q3_logged_vs_rater_best.csv", index=False)
    for xs in (1.0, 1.06):
        geo, sc = res[xs]
        lg_full = np.full(n, np.nan)
        ins_full = np.zeros(n, bool)
        lg, ins = W.rater_feedback_score(np.nan_to_num(d["fut20"]), d["traj"], d["scores"], d["speed"], details=True)
        lg_full, ins_full = lg, ins[:, 0]
        Om = sc["O"]; Mm = np.mean([sc[m] for m in SEEDS], 0)
        for s in ["S", "start", "stop", "turn_onset", "stay", "in_turn", "nudge", "lane_change", "control", "other", "all"]:
            m = d["slice"][s] & has
            if not m.any():
                continue
            q3.append({"slice": s, "xscale": xs, "n": int(m.sum()), "seg": int(len(np.unique(d["seg"][m]))), "rfs_logged": float(lg_full[m].mean()), "rfs_orig": float(Om[m].mean()),
                       "rfs_main": float(Mm[m].mean()), "logged_inside_frac": float(ins_full[m].mean()), "logged_floor_frac": float((lg_full[m] <= W.RFS_FLOOR + 1e-9).mean()),
                       "orig_floor_frac": float((Om[m] <= W.RFS_FLOOR + 1e-9).mean()), "rater_best_score_mean": float(d["scores"].max(1)[m].mean()),
                       **{f"logged_minus_orig_{k}": v for k, v in bf.diff(lg_full, m, Om, m).items()},
                       **{f"main_minus_logged_{k}": v for k, v in bf.diff(Mm, m, lg_full, m).items()},
                       **{f"main_minus_orig_{k}": v for k, v in bf.diff(Mm, m, Om, m).items()}})
    q3 = pd.DataFrame(q3)
    q3.to_csv(OUT / "q3_logged_rfs.csv", index=False)
    chk["rater_best_ties_frames"] = ties
    # ================= Q4 mechanism on flipped rows
    q4, tr = [], []
    for xs in (1.0, 1.06):
        geo, sc = res[xs]
        go = geo["O"]
        for s in ["S", "start", "stop", "turn_onset"]:
            fl = flipinfo[(xs, s)]
            for kind in ("on", "off", "unch"):
                mk = fl[kind]
                if not mk.any():
                    continue
                dl_l, dl_t, dd, ins_o, ins_a, bl_o, bl_a, nl_o, nt_o, nl_a, nt_a, sgn_o, sgn_a = ([] for _ in range(13))
                rec = {"xscale": xs, "slice": s, "flip": kind, "n_rows": int(mk.sum())}
                parts = {k: [] for k in ("d_total", "d_lng", "d_lat", "inter")}
                for k_seed, m_ in enumerate(SEEDS):
                    sel = mk[k_seed * n:(k_seed + 1) * n]
                    if not sel.any():
                        continue
                    ga = geo[m_]
                    base = sc["O"][sel]
                    s_a = sc[m_][sel]
                    s_l = score_from(ga["sl"][sel], go["st"][sel], go["lt"][sel], go["at"][sel], d["scores"][sel])[0]   # lon from main, lat from O
                    s_t = score_from(go["sl"][sel], ga["st"][sel], go["lt"][sel], go["at"][sel], d["scores"][sel])[0]   # lat from main, lon from O
                    parts["d_total"].append(s_a - base); parts["d_lng"].append(s_l - base); parts["d_lat"].append(s_t - base)
                    parts["inter"].append((s_a - base) - (s_l - base) - (s_t - base))
                    ins_o.append(go["inside"][sel]); ins_a.append(ga["inside"][sel])
                    for g, nl_, nt_, bl_, sg_ in ((go, nl_o, nt_o, bl_o, sgn_o), (ga, nl_a, nt_a, bl_a, sgn_a)):
                        bi = g["hv"][sel].argmax(1)                                  # (r, 2) best rater per horizon
                        rr = np.arange(sel.sum())[:, None]; hh = np.arange(2)[None]
                        nl_.append(g["nl"][sel][rr, bi, hh]); nt_.append(g["nt"][sel][rr, bi, hh])
                        bl_.append(g["nl"][sel][rr, bi, hh] > g["nt"][sel][rr, bi, hh])
                        sg_.append(g["sl"][sel][rr, bi, hh])
                cat = lambda v: np.concatenate(v)  # noqa: E731
                for k, v in parts.items():
                    rec[k] = float(cat(v).mean())
                ino, ina = cat(ins_o), cat(ins_a)
                rec.update(out_to_in=float((~ino & ina).mean()), in_to_out=float((ino & ~ina).mean()), stay_in=float((ino & ina).mean()), stay_out=float((~ino & ~ina).mean()),
                           orig_inside=float(ino.mean()), main_inside=float(ina.mean()))
                for tag, nl_, nt_, bl_, sg_ in (("O", nl_o, nt_o, bl_o, sgn_o), ("main", nl_a, nt_a, bl_a, sgn_a)):
                    a_, b_, c_, s_ = cat(nl_), cat(nt_), cat(bl_), cat(sg_)
                    for h, hn in enumerate((3, 5)):
                        rec[f"{tag}_nlng_{hn}s"] = float(a_[:, h].mean()); rec[f"{tag}_nlat_{hn}s"] = float(b_[:, h].mean())
                        rec[f"{tag}_lng_bind_{hn}s"] = float(c_[:, h].mean()); rec[f"{tag}_lng_signed_{hn}s"] = float(s_[:, h].mean())
                # weighted (leaderboard units) sums over the flipped rows for the hypothesis bounds
                ww = np.concatenate([w[mk[k * n:(k + 1) * n]] / 3 for k in range(3)])
                rec["w_sum_d_total"] = float((ww * cat(parts["d_total"])).sum())
                rec["w_neg_min_component"] = float((ww * np.minimum(0, np.minimum(cat(parts["d_lng"]), cat(parts["d_lat"])))).sum())
                rec["w_sum_d_total_not_out_to_in"] = float((ww * cat(parts["d_total"]) * ~(~ino & ina)).sum())
                q4.append(rec)
    q4 = pd.DataFrame(q4)
    q4.to_csv(OUT / "q4_mechanism.csv", index=False)
    # ================= Q5 hypotheses
    Q = lambda s, flip, xs=1.0: q2[(q2.slice == s) & (q2.flip == flip) & (q2.xscale == xs)].iloc[0]  # noqa: E731
    q1S = q1[q1.slice == "S"].iloc[0]
    hyp = []
    r1a = bool(q1S.H_slice < TOP - chk["rfs_x1.0"]["O"] or q1S.H_share < 0.15)
    hyp.append({"id": "H1", "mechanism": "headroom in imitated slices too small", "rule": "H_S < gap to top or H_S/H_all < 0.15", "supported": r1a,
                "numbers": f"H_S={q1S.H_slice:.3f} (gap to top {TOP - chk['rfs_x1.0']['O']:.3f}); share={q1S.H_share:.3f}", "bound_lb_units": float(q1S.H_slice)})
    r2a = {s: bool(Q(s, "on").lo > 0 and Q(s, "on")["mean"] >= 0.3) if Q(s, "on").n else False for s in ("start", "stop", "turn_onset", "S")}
    E_on = float(Q("S", "on")["mean"] * np.sum(w[fi][flipinfo[(1.0, "S")]["on"]] / 3)) if Q("S", "on").n else 0.0
    realised = float(q4[(q4.slice == "S") & (q4.flip == "on") & (q4.xscale == 1.0)].w_sum_d_total.iloc[0]) if ((q4.slice == "S") & (q4.flip == "on") & (q4.xscale == 1.0)).any() else np.nan
    r2c = bool(r2a["S"] and E_on < 0.07)
    hyp.append({"id": "H3", "mechanism": "gains real but flip frames too few", "rule": "R2a (pooled S on-rows CI>0, mean>=0.3) and E_on<0.07", "supported": r2c,
                "numbers": f"R2a per slice {r2a}; E_on={E_on:.4f}; realised sum over on-rows={realised:.4f}", "bound_lb_units": E_on})
    qS = q3[(q3.slice == "S") & (q3.xscale == 1.0)].iloc[0]
    lg = np.nan_to_num(W.rater_feedback_score(np.nan_to_num(d["fut20"]), d["traj"], d["scores"], d["speed"]))
    mS = d["slice"]["S"] & has
    r3b = bool(qS.logged_minus_orig_lo <= 0 or qS.rfs_logged < qS.rfs_orig)
    hyp.append({"id": "H2", "mechanism": "logged future is not rater-preferred", "rule": "CI lower bound of (logged - O) on S <= 0, or logged mean < O mean", "supported": r3b,
                "numbers": f"S: logged {qS.rfs_logged:.3f} vs O {qS.rfs_orig:.3f}; diff {qS.logged_minus_orig_mean:+.3f} [{qS.logged_minus_orig_lo:+.3f}, {qS.logged_minus_orig_hi:+.3f}]",
                "bound_lb_units": float((w[mS] * np.maximum(0, lg[mS] - raw_sc["O"][mS])).sum())})
    k4 = q4[(q4.slice == "S") & (q4.flip == "on") & (q4.xscale == 1.0)]
    if len(k4):
        k4 = k4.iloc[0]
        r4a = bool(np.sign(k4.d_lng) != np.sign(k4.d_lat) and abs(k4.d_lng) >= 0.1 and abs(k4.d_lat) >= 0.1)
        r4b = bool(k4.out_to_in < 0.5)
        hyp.append({"id": "H4", "mechanism": "longitudinal / lateral mismatch", "rule": "on-rows d_lng and d_lat of opposite sign, each |mean| >= 0.1", "supported": r4a,
                    "numbers": f"on-rows d_total {k4.d_total:+.3f}, lng-only {k4.d_lng:+.3f}, lat-only {k4.d_lat:+.3f}, interaction {k4.inter:+.3f}", "bound_lb_units": float(-k4.w_neg_min_component)})
        hyp.append({"id": "H5", "mechanism": "capture flip does not enter the trust region", "rule": "share out->in among on-rows < 0.5", "supported": r4b,
                    "numbers": f"out->in {k4.out_to_in:.3f}, in->out {k4.in_to_out:.3f}, stay in {k4.stay_in:.3f}, stay out {k4.stay_out:.3f}", "bound_lb_units": float(k4.w_sum_d_total_not_out_to_in)})
    hyp = pd.DataFrame(hyp)
    hyp.to_csv(OUT / "q5_hypotheses.csv", index=False)
    (OUT / "selfcheck.json").write_text(json.dumps(chk, indent=1, default=float))
    pd.set_option("display.width", 250, "display.max_columns", 60)
    print(json.dumps(chk, indent=1, default=float))
    print(q1.round(3).to_string(index=False)); print(pd.read_csv(OUT / "q1_category_headroom.csv").round(3).iloc[:, :8].to_string(index=False))
    print(q2[q2.xscale == 1.0].round(3).drop(columns=["xscale", "seed_means"]).to_string(index=False))
    print(pd.DataFrame(contrib).round(4).to_string(index=False))
    print(pd.DataFrame(dist_rows).round(2).to_string(index=False)); print(q3[q3.xscale == 1.0].round(3).T.to_string())
    print(q4[q4.xscale == 1.0].round(3).T.to_string()); print(hyp.to_string(index=False))


def figure(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})
    q1 = pd.read_csv(OUT / "q1_coverage.csv")
    q2 = pd.read_csv(OUT / "q2_response.csv")
    q3 = pd.read_csv(OUT / "q3_logged_rfs.csv")
    q4 = pd.read_csv(OUT / "q4_mechanism.csv")
    cc = pd.read_csv(OUT / "q1_category_headroom.csv")
    fig, ax = plt.subplots(2, 2, figsize=(10.5, 7))
    blue, grey, red, green = "#1f4e9c", "#9a9a9a", "#c0392b", "#2e8b57"
    # (a) headroom per slice
    a = ax[0, 0]
    s = q1[q1.slice.isin(["start", "stop", "turn_onset", "stay", "in_turn", "nudge", "lane_change", "control", "other", "no_future"])].copy()
    x = np.arange(len(s))
    a.bar(x - 0.2, s.frac_frames, 0.4, color=grey, label="share of 479 frames")
    a.bar(x + 0.2, s.H_share, 0.4, color=blue, label="share of RFS headroom")
    a.set_xticks(x); a.set_xticklabels(s.slice, rotation=40, ha="right"); a.set_title("(a) Where the frames and the headroom are"); a.legend(frameon=False)
    # (b) response by flip
    a = ax[0, 1]
    xs_ = q2[q2.xscale == 1.0]
    lab = []
    for j, sl in enumerate(["start", "stop", "turn_onset", "S"]):
        for k, (fl, c) in enumerate((("on", green), ("unch", grey), ("off", red))):
            r = xs_[(xs_.slice == sl) & (xs_.flip == fl)]
            if len(r) and r.n.iloc[0]:
                r = r.iloc[0]
                a.errorbar(j + (k - 1) * 0.22, r["mean"], yerr=[[r["mean"] - r.lo], [r.hi - r["mean"]]], fmt="o", color=c, ms=4, capsize=2,
                           label={"on": "captured on", "unch": "unchanged", "off": "captured off"}[fl] if j == 0 else None)
    a.axhline(0, color="k", lw=0.6); a.set_xticks(range(4)); a.set_xticklabels(["start", "stop", "turn onset", "S (pooled)"])
    a.set_ylabel("per-frame RFS change, main vs O (95% CI)"); a.set_title("(b) RFS change by capture flip"); a.legend(frameon=False)
    # (c) logged vs O vs main
    a = ax[1, 0]
    q = q3[(q3.xscale == 1.0) & q3.slice.isin(["start", "stop", "turn_onset", "S", "stay", "control", "other", "all"])]
    x = np.arange(len(q))
    for k, (col, c, nm) in enumerate((("rfs_orig", grey, "original"), ("rfs_main", blue, "main"), ("rfs_logged", red, "logged future"))):
        a.bar(x + (k - 1) * 0.27, q[col], 0.27, color=c, label=nm)
    a.set_xticks(x); a.set_xticklabels(q.slice, rotation=40, ha="right"); a.set_ylim(4, 10); a.set_ylabel("mean RFS"); a.set_title("(c) Logged future as a prediction"); a.legend(frameon=False, ncol=3)
    # (d) decomposition on flipped rows
    a = ax[1, 1]
    q = q4[(q4.xscale == 1.0) & (q4.slice == "S") & q4.flip.isin(["on", "off", "unch"])].set_index("flip")
    x = np.arange(3)
    for k, (col, c, nm) in enumerate((("d_lng", blue, "longitudinal only"), ("d_lat", "#e08a1e", "lateral only"), ("inter", grey, "interaction"), ("d_total", "k", "total"))):
        a.bar(x + (k - 1.5) * 0.2, [q.loc[f, col] if f in q.index else 0 for f in ("on", "unch", "off")], 0.2, color=c, label=nm)
    a.axhline(0, color="k", lw=0.6); a.set_xticks(x); a.set_xticklabels(["captured on", "unchanged", "captured off"])
    a.set_ylabel("RFS change of the S rows"); a.set_title("(d) Trust-region components of the change"); a.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    p = REPO / "research" / "figs" / "op-adapt-L-rfs-diagnosis"
    fig.savefig(str(p) + ".png", dpi=200); fig.savefig(str(p) + ".pdf")
    print("saved", p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "figure"])
    a = ap.parse_args()
    {"run": run, "figure": figure}[a.cmd](a)
