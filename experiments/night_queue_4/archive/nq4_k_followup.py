"""Night queue 4, K follow-up (asked after the numbers, descriptive only, never enters a verdict): is the K0 -> K1 drop
of the I3 / P5 cut-in flip rate a loss of reaction or an artifact of K1's representation?

No refit: K0 / K1 come from the stored open-loop exports (runs/nq4/k/openloop) and, for K1's class probabilities, the
stored heads (runs/nq4/k/K1/<fold>/head.npz) applied in numpy (nq4_k.KHead) to the same stored inputs, checked
against the exported v. Per exam (I3 R1; P5 v1 BA unseen, cut-in and pedestrian families):

  variants   the model's speed read at 2 s (the registered exam), at 1 s (K1's target: mean speed 0.75-1.25 s), and K1
             decoded by argmax instead of the two-hot expectation; the expert's own 1 s speed as the horizon ceiling
  exam       p5_exam.exam on those Delta columns (tau from each column's own null, as registered)
  free       threshold-free: directional agreement sign(Delta) == sign(d_expert) on reactive frames (ties 1/2) and the
             detection AUC P(Delta * sign(d_expert) on reactive > |Delta| on null) - the area under the hit /
             false-flip curve the registered (tau, flip) pair is one point of; route bootstrap
  output     K1 probability sharpness (max p), exact / near-zero Delta shares, Spearman of K1 vs K0 Delta

    python -m experiments.night_queue_4.archive.nq4_k_followup    -> runs/nq4/k/followup/<stamp>/
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from jevdrive.common import data_dir
from jevdrive import nq4_k as K

NB = 2000


def _v(fut, a, b):
    """Mean speed between future samples a and b (index i = (i + 1) * 0.25 s)."""
    return np.linalg.norm(fut[..., b, :] - fut[..., a, :], axis=-1) / (0.25 * (b - a))


def v2(fut):
    return _v(fut, 6, 7)


def v1(fut):                                   # K1's speed label window, 0.75 -> 1.25 s
    return _v(fut, 2, 4)


def _auc(pos, neg):
    """P(pos > neg) + P(tie) / 2."""
    if not len(pos) or not len(neg):
        return np.nan
    r = rankdata(np.r_[pos, neg])
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _idx(groups: np.ndarray) -> list:
    return [np.flatnonzero(groups == g) for g in pd.unique(groups)]


def _draw(ix: list, rng) -> np.ndarray:
    return np.concatenate([ix[k] for k in rng.integers(len(ix), size=len(ix))])


def _dirq(x, d):
    s = np.sign(x) * np.sign(d)
    return (s > 0) + 0.5 * (s == 0)


def _boot(r: pd.DataFrame, n: pd.DataFrame, c: str, b: int = NB, seed: int = 0) -> dict:
    """Directional agreement and detection AUC with whole routes resampled (reactive and null by their base route)."""
    q, sx, nx = _dirq(r[c].to_numpy(), r.d_expert.to_numpy()), (r[c] * np.sign(r.d_expert)).to_numpy(), np.abs(n[c].to_numpy())
    ri, ni = _idx(r.base_id.to_numpy()), _idx(n.base_id.to_numpy())
    rng = np.random.default_rng(seed)
    D, A = [], []
    for _ in range(b):
        a, m = _draw(ri, rng), _draw(ni, rng)
        D.append(q[a].mean())
        A.append(_auc(sx[a], nx[m]))
    return {"dir_agree": float(q.mean()), "dir_lo": float(np.quantile(D, 0.025)), "dir_hi": float(np.quantile(D, 0.975)),
            "det_auc": _auc(sx, nx), "det_lo": float(np.quantile(A, 0.025)), "det_hi": float(np.quantile(A, 0.975))}


def _paired_boot(r: pd.DataFrame, n: pd.DataFrame, a: str, b: str, bb: int = NB, seed: int = 0) -> dict:
    """a - b on the same frames (routes resampled once for both): directional agreement and detection AUC."""
    d = r.d_expert.to_numpy()
    qa, qb = _dirq(r[a].to_numpy(), d), _dirq(r[b].to_numpy(), d)
    sa, sb = (r[a] * np.sign(d)).to_numpy(), (r[b] * np.sign(d)).to_numpy()
    na, nb = np.abs(n[a].to_numpy()), np.abs(n[b].to_numpy())
    ri, ni = _idx(r.base_id.to_numpy()), _idx(n.base_id.to_numpy())
    rng = np.random.default_rng(seed)
    D, A = [], []
    for _ in range(bb):
        i, m = _draw(ri, rng), _draw(ni, rng)
        D.append(qa[i].mean() - qb[i].mean())
        A.append(_auc(sa[i], na[m]) - _auc(sb[i], nb[m]))
    return {"a": a, "b": b, "dir_diff": float(qa.mean() - qb.mean()), "dir_lo": float(np.quantile(D, 0.025)),
            "dir_hi": float(np.quantile(D, 0.975)), "auc_diff": _auc(sa, na) - _auc(sb, nb),
            "auc_lo": float(np.quantile(A, 0.025)), "auc_hi": float(np.quantile(A, 0.975))}


def _cols(obs, null, t, series: dict):
    pos = pd.Series(np.arange(len(t)), index=t.frame_name)
    ip, im = pos[obs.fn_plus].to_numpy(), pos[obs.fn_minus].to_numpy()
    jp, jn = pos[null.fn_plus].to_numpy(), pos[null.fn_null].to_numpy()
    obs, null = obs.copy(), null.copy()
    for k, v in series.items():
        obs[k], null[k] = v[ip] - v[im], v[jp] - v[jn]
    return obs, null


def _k1_probs(ego, op, folds):
    """(v_expect, v_argmax, max p) from the stored K1 heads; folds: per-row fold name."""
    kh = K.KHead()
    ve, va, mp = (np.full(len(ego), np.nan) for _ in range(3))
    for f in K.FOLDS:
        m = folds == f
        if not m.any():
            continue
        p = kh.params("K1", f)
        xe, xo = (ego[m] - p["ego_mu"]) / p["ego_sd"], (op[m] - p["op_mu"]) / p["op_sd"]
        pr = K._softmax(np.asarray(kh._lin(xe, p["Ce"]) + kh._lin(xo, p["Cp"]), np.float64))
        ve[m], va[m], mp[m] = pr @ K.SPEEDS, K.SPEEDS[pr.argmax(1)], pr.max(1)
    return ve, va, mp


def _one(exam: str, obs, null, pairs, t, fut, k0, k1, ego, op, folds, fams, rl, v_ref=None):
    from jevdrive import p5_exam as E
    ve, va, mp = _k1_probs(ego, op, folds)
    ref = v2(k1) if v_ref is None else v_ref          # exported v where stored, else the path's 2 s chord speed
    chk, chk_med = float(np.abs(ve - ref).max()), float(np.median(np.abs(ve - ref)))
    series = {"K0 @2s": v2(k0), "K1 @2s": v2(k1), "K0 @1s": v1(k0), "K1 @1s": v1(k1), "K1 argmax": va,
              "expert @1s": v1(fut), "expert @2s": v2(fut)}
    oo, nn = _cols(obs, null, t, series)
    E.TFV6 = {}
    res = E.exam(oo, nn, pairs, list(series))
    r = res["obs"][res["obs"].reactive]
    r = r[r.family.isin(fams)]
    n = nn.copy()
    rows = []
    for c in series:
        tau = res["taus"][c]
        hit = ((np.sign(r[c]) == np.sign(r.d_expert)) & E._moved(r[c], tau)).astype(float)
        fr, flo, fhi = E.boot_ratio(hit.to_numpy(), np.ones(len(r)), r.base_id.to_numpy())
        nr = res["obs"][~res["obs"].reactive & res["obs"].family.isin(fams)]
        rows.append({"exam": exam, "column": c, "n_reactive": len(r), "routes": r.base_id.nunique(), "tau": tau,
                     "flip": fr, "flip_lo": flo, "flip_hi": fhi,
                     "false_flip_nonreactive": float(E._moved(nr[c], tau).mean()),
                     "false_flip_null": float(E._moved(n[c].dropna(), tau).mean()),
                     **_boot(r, n, c),
                     "median_abs_delta_reactive": float(np.median(np.abs(r[c]))),
                     "median_abs_delta_null": float(np.median(np.abs(n[c]))),
                     "share_abs_delta_lt_0.05": float((np.abs(r[c]) < 0.05).mean())})
    tab = pd.DataFrame(rows)
    pair = [{"exam": exam, **_paired_boot(r, n, a, b)} for a, b in
            (("K1 @2s", "K0 @2s"), ("K1 @1s", "K0 @1s"), ("K0 @1s", "K0 @2s"), ("K1 argmax", "K1 @2s"),
             ("expert @1s", "expert @2s"))]
    rho = spearmanr(r["K1 @2s"], r["K0 @2s"]).statistic
    rho1 = spearmanr(r["K1 @1s"], r["K0 @1s"]).statistic
    at = pd.Series(np.arange(len(t)), index=t.frame_name)
    sel = np.unique(np.r_[at[r.fn_plus].to_numpy(), at[r.fn_minus].to_numpy()])
    out = {"exam": exam, "k1_v_recompute_vs_export_max_abs": chk, "k1_v_recompute_vs_export_median_abs": chk_med,
           "k1_maxp_median_reactive_frames": float(np.median(mp[sel])),
           "k1_share_maxp_gt_0.9_reactive_frames": float((mp[sel] > 0.9).mean()),
           "spearman_K1_vs_K0_delta_2s": float(rho), "spearman_K1_vs_K0_delta_1s": float(rho1),
           "expert_v2_vs_d_expert_max_abs": float(np.abs(r["expert @2s"] - r.d_expert).max())}
    rl.log.info("%s: %s", exam, out)
    return tab, pd.DataFrame(pair), out


def main():
    from jevdrive import elicit_i3 as I, p5_exam as E, p5_openpilot as PO, p5_pairs as P
    from jevdrive.runlog import RunLog
    rl = RunLog("nq4", "k", "followup")
    ol = lambda kind, lv: np.load(data_dir() / "runs/nq4/k/openloop" / f"{kind}_{lv}.npz", allow_pickle=True)
    CUT = ("HighwayCutIn", "StaticCutIn", "ParkingCutIn")
    tabs, pairs_, info = [], [], []
    with I.p5_set(I.I3):
        ta, pa, fa, obs3, null3, pairs3 = E.load()
        keep = ta.frame_name.isin(set(I.needed())).to_numpy()
        t3, past3, fut3 = ta[keep].reset_index(drop=True), pa[keep], fa[keep]
        op3 = PO.load(t3, (K.MODEL,), sub="op_streams")[f"op-{K.MODEL} temporal"]
    ego3 = E.ego_input(t3, past3)
    fam3 = sorted(obs3.family.unique())
    x = _one("I3 R1 (all families)", obs3, null3, pairs3, t3, fut3, ol("i3", "K0")["R1"], ol("i3", "K1")["R1"],
             ego3, op3, np.full(len(t3), "R1"), fam3, rl)
    tabs.append(x[0]); pairs_.append(x[1]); info.append(x[2])
    with I.p5_set(I.BA):
        t, past, fut, obs, null, pairs = E.load()
        op = PO.load(t, (K.MODEL,), sub="op_streams_vis")[f"op-{K.MODEL} temporal"]
    z0, z1 = ol("ba", "K0"), ol("ba", "K1")
    folds = z1["readout_unseen"].astype(str)
    ego = E.ego_input(t, past)
    for name, fams in (("P5 cut-in (unseen)", CUT), ("P5 pedestrian (unseen)", E.PED_FAMILIES)):
        x = _one(name, obs, null, pairs, t, fut, z0["unseen"], z1["unseen"], ego, op, folds, fams, rl, z1["v_unseen"])
        tabs.append(x[0]); pairs_.append(x[1]); info.append(x[2])
    out = rl.dir
    tab, pr = pd.concat(tabs, ignore_index=True), pd.concat(pairs_, ignore_index=True)
    tab.to_csv(out / "followup_readouts.csv", index=False)
    pr.to_csv(out / "followup_paired.csv", index=False)
    (out / "followup_info.json").write_text(json.dumps(info, indent=1))
    rl.log.info("\n%s\n\n%s", tab.to_markdown(index=False, floatfmt=".3f"), pr.to_markdown(index=False, floatfmt=".3f"))
    rl.close()


if __name__ == "__main__":
    main()
