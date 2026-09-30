"""op-adapt r2 readouts (op-train venv; §4 of todos/2026-09-29-op-adapt-r2-prereg.md). O and the adapted model are read
in the same forward (the O features are computed once and shared) and through the same probes / bootstraps.

  eval     feature pass of one model on every readout set -> R2/readout/<model>/eval/<set>.npz (`uid`, orig_* / adapt_*:
           temporal, vision (512), plan / plan_logstd (33, 15), lead (72), lead_prob (3), head_t / head_v (3));
           --model O writes the shared O pass (R2/readout/O/eval)
  read     every §4 readout that has its inputs -> R2/readout/<model>/{<readout>.csv, summary.json}; a readout whose
           input is not delivered yet is listed as pending with the reason; --model O = M1 (O columns only, R2/m1/)
  verdict  §4.5 outcome per main arm (A, D; every seed must pass a line) + the arm comparisons -> R2/readout/verdict.json
  navsim   the decision-36 harness with the port: native plan of O-port and a model on lb_navtest (N-nav) or, only
           after the verdict exists and with --final, lb_navhard (§4.6 EPDMS combined); plans written in op_lb format
           to runs/op_lb/<data>/plans/gimm@cinque_<tag>.npz, then op_interp nav-export / op_interp_score.sh / nav-report
  rater    trunk cache of the WOD-E2E val rater frames (N-rfs; 18 predecessors each, the WOD stream protocol)
  fullfwd  §6 1-unit item 5: cache -> stage 4 -> policy against the full port forward from the images, 3 Cosmos pairs

  CUDA_VISIBLE_DEVICES=2 taskset -c 40-47 python scripts/op_adapt_r2_readout.py eval --model O
  python scripts/op_adapt_r2_readout.py read --model A-s0-ls1
Every readout look is appended to R2/readout/looks.jsonl (who, what, when).
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from jevdrive import op_adapt as A  # noqa: E402
from jevdrive import op_adapt_r2 as R  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

AT = (0.275, 0.525)
PLAN_T = 0.25 * np.arange(1, 21)
P5_CAM_X = 1.519                                   # jevdrive.p5_openpilot.RIG front camera x (rear-axle frame)
SETS = {  # readout set -> (domain, row filter)
    "nusval": ("nus", lambda d: (d.col("split") == "val") & d.col("labelled", False, bool)),            # keyframes
    "wodval": ("wod", lambda d: ((d.col("part", "") == "val") & d.col("target", True, bool)) if "part" in d.s
               else (d.col("split") == "val")),                                                     # round-1 readout frames
    "p5": ("p5", lambda d: np.ones(len(d), bool)),
    "cosC": ("simC", lambda d: d.col("split") == "test"),
    "cosK": ("simK", lambda d: d.col("split") == "test"),
    "offdev": ("off", lambda d: d.col("split") == "dev"),
    "rater": ("rater", lambda d: np.ones(len(d), bool)),
    "cosC_swap": ("simC_swap", lambda d: np.ones(len(d), bool)),     # §4.4 pedestrian region swapped to x- (package C)
    "cosK_swap": ("simK_swap", lambda d: np.ones(len(d), bool)),
    "p5_swap": ("p5_swap", lambda d: np.ones(len(d), bool)),
}
MAIN = ("A", "D")
N_BOOT = 2000


def rdir(model, *p):
    """R2/readout/<model>/...; the O readout itself (M1) lives in R2/m1/ (its eval pass stays in R2/readout/O/eval)."""
    if model == "O" and not (p and p[0] == "eval"):
        return R.r2("m1", *p)
    return R.r2("readout", model, *p)


def look(model, what, **kw):
    with open(R.r2("readout") / "looks.jsonl", "a") as f:
        f.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "model": model, "what": what, "user": os.environ.get("USER"), **kw}) + "\n")


# ---------------------------------------------------------------- eval (feature pass)
def load_model(model: str, dev):
    """'O' -> the original; else a training run tag (R2/train-<tag>/ckpt-final.pt) or a path to a checkpoint."""
    if model == "O":
        return R.Model(R.ARMS["O"]).to(dev).eval(), None
    p = Path(model) if model.endswith(".pt") else R.r2(f"train-{model}") / "ckpt-final.pt"
    ck = torch.load(p, map_location="cpu", weights_only=False)
    arm = R.ARMS[ck["cfg"]["arm"]]
    det = R.det_adapter() if arm.det else None
    m = R.Model(arm, det=det).to(dev).eval()
    m.load_state(ck["model"])
    return m, (R.DetSource() if arm.det else None)


def cmd_eval(a):
    dev = torch.device("cuda")
    m, det_fn = load_model(a.model, dev)
    who = "orig" if a.model == "O" else "adapt"
    out = rdir(a.model, "eval")
    for name in a.sets:
        dn, filt = SETS[name]
        if not (R.r2("t") / "samples" / f"{dn}.parquet").exists():
            print(f"{name}: domain {dn} not packed, skipped")
            continue
        d = R.Domain(dn)
        rows = np.flatnonzero(filt(d))
        t0 = time.time()
        o = R.forward_rows(m, d, rows, dev, det_fn=det_fn, feats=True)
        np.savez(out / f"{name}.npz", uid=d.uid[rows], **{f"{who}_{k}": v.astype(np.float32) for k, v in o.items()})
        print(f"{name}: {len(rows)} rows in {time.time() - t0:.0f} s")


def load_eval(model, name):
    """uid-aligned O and adapted arrays of one set (the adapted keys absent for M1)."""
    po = rdir("O", "eval") / f"{name}.npz"
    if not po.exists():
        return None
    z = dict(np.load(po))
    if model != "O":
        pa = rdir(model, "eval") / f"{name}.npz"
        if not pa.exists():
            return None
        za = dict(np.load(pa))
        assert np.array_equal(za["uid"], z["uid"]), f"{name}: row order differs between O and {model}"
        z |= {k: v for k, v in za.items() if k != "uid"}
    return z


# ---------------------------------------------------------------- helpers
def boot_ci(stat, groups, b=N_BOOT, seed=0):
    """Percentile CI of stat(idx) under a cluster bootstrap over groups."""
    codes, uniq = pd.factorize(pd.Series(groups).astype(str))
    mem = [np.flatnonzero(codes == c) for c in range(len(uniq))]
    rng = np.random.default_rng(seed)
    v = [stat(np.concatenate([mem[c] for c in rng.integers(len(uniq), size=len(uniq))])) for _ in range(b)]
    v = np.array([x for x in v if np.isfinite(x)])
    return (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))) if len(v) else (np.nan, np.nan)


def rows_of(d: R.Domain, uid):
    return d.pos.reindex(uid).to_numpy().astype(int)


def plan20(plan, cam_x=0.0):
    """(n, 33, 15) openpilot plan (x fwd, y right, camera origin) -> (n, 20, 2) rear-axle x fwd / y left on 0.25 s."""
    return np.stack([np.stack([np.interp(PLAN_T, A.T_IDXS, p[:, 0]) + cam_x, -np.interp(PLAN_T, A.T_IDXS, p[:, 1])], 1)
                     for p in plan])


def line(value, ok, rule, **kw):
    return {"value": value, "pass": None if ok is None else bool(ok), "line": rule, **kw}


# ---------------------------------------------------------------- 4.1 readability
def probe_pair(X_a, X_o, y, g, dist=None, bins=None):
    import op_adapt_readout as R1
    sa, so = R1.oof_probe(X_a, y, g), R1.oof_probe(X_o, y, g)
    r = R1.paired_boot(y, sa, so, g)
    out = {"main": r}
    if dist is not None:
        neg = y == 0
        for lo, hi in bins:
            p = (y == 1) & (dist >= lo) & (dist < hi)
            m = p | neg
            if p.sum() >= 10:
                out[f"{lo}-{hi}"] = R1.paired_boot(p[m].astype(int), sa[m], so[m], g[m])
    return out


def r_real(model, name, dn, thr):
    """R-nus / R-wod: corridor-pedestrian `temporal` probe AUC delta (D0 recipe, scene / sequence clusters)."""
    z = load_eval(model, name)
    if z is None:
        return None
    d = R.Domain(dn)
    r = rows_of(d, z["uid"])
    m = d.col("labelled", False, bool)[r] & ~d.col("uncertain", False, bool)[r]
    y = d.col("ped_corr", False, bool)[r][m].astype(int)
    g = d.col("group")[r][m].astype(str)
    Xo = z["orig_temporal"][m]
    Xa = z["adapt_temporal"][m] if model != "O" else Xo
    res = probe_pair(Xa, Xo, y, g, d.col("ped_dist")[r][m], ((0, 10), (10, 20), (20, 30)))
    mr = res["main"]
    ok = None if model == "O" else (mr["delta"] >= thr and mr["delta_ci"][0] > 0)
    return line(mr, ok, f"delta >= +{thr} and CI low > 0", bins={k: v for k, v in res.items() if k != "main"})


def p5_frames(model):
    """P5 exam tables and uid-aligned eval arrays (both models) in p5_exam's frame order."""
    os.environ["P5_SET"] = "carla_p5v1_ba"
    from jevdrive import p5_exam as E
    t, past, fut, obs, null, pairs = E.load()
    z = load_eval(model, "p5")
    if z is None:
        return None
    d = R.Domain("p5")
    names = d.col("name" if "name" in d.s else "key")[rows_of(d, z["uid"])].astype(str)
    pos = pd.Series(np.arange(len(names)), index=names)
    at = pos.reindex(t.frame_name).to_numpy()
    assert not np.isnan(at).any(), "P5 frames without features"
    at = at.astype(int)
    return E, t, past, fut, obs, null, pairs, {k: v[at] for k, v in z.items() if k != "uid"}


def vis500(obs):
    """P5 obs rows with the pedestrian visible at >= 500 px_eq (package C: index/p5_obs.parquet `vis500`, keyed by
    fn_plus); None while it is not delivered."""
    p = R.r2("index") / "p5_obs.parquet"
    if not p.exists():
        return None, None
    po = pd.read_parquet(p)
    k = po.set_index("fn_plus")
    return obs.fn_plus.map(k.vis500).fillna(False).to_numpy(bool), obs.fn_plus.map(k.px_eq).to_numpy(float)


def s_p5(model, rl):
    """S-p5 (>= 500 px_eq visible, main) and S-p5-all: D0 pedestrian scope, adapted vs O temporal, route bootstrap."""
    f = p5_frames(model)
    if f is None:
        return None
    E, t, past, fut, obs, null, pairs, X = f
    taps = {"orig_temporal": X["orig_temporal"]}
    if model != "O":
        taps["adapt_temporal"] = X["adapt_temporal"]
    fold = E.folds(t, pairs)
    scores, _ = E.probes(t, taps, fold, rl)
    cmp = [("adapt_temporal", "orig_temporal")] if model != "O" else [("orig_temporal", "orig_temporal")]
    res = {}
    allp = E.probe_auc_paired_scopes(obs, t, scores, cmp)
    ped = allp[allp.scope == "pedestrian"].iloc[0].to_dict()
    res["S-p5-all"] = line(ped, None if model == "O" else (ped["auc"] >= 0.60 and ped["lo"] > 0), "AUC >= 0.60 and delta CI low > 0")
    v, px = vis500(obs)
    if v is None:
        res["S-p5"] = {"pending": "package C index/p5_obs.parquet (vis500)"}
    else:
        sub = E.probe_auc_paired_scopes(obs[v], t, scores, cmp)
        p = sub[sub.scope == "pedestrian"].iloc[0].to_dict()
        res["S-p5"] = line(p, None if model == "O" else (p["auc"] >= 0.65 and p["lo"] > 0), "AUC >= 0.65 and delta CI low > 0")
        bins = {}
        for lo, hi in ((0, 500), (500, 1500), (1500, np.inf)):
            m = (px >= lo) & (px < hi) & np.isfinite(px)
            if m.sum() >= 20:
                b = E.probe_auc_paired_scopes(obs[m], t, scores, cmp)
                bins[f"{lo}-{hi}"] = b[b.scope == "pedestrian"].iloc[0].to_dict() if (b.scope == "pedestrian").any() else None
        res["S-p5 size bins"] = bins
    return res


def s_cos(model):
    """S-cos: E1 pooled probe (round-1 oof_probe, 5 folds by instance) on `temporal`, x+ vs x- of the Cosmos test
    readout slots (x+ px_eq >= 100, E1's rows), C and K each; delta adapted - O."""
    out = {}
    for c in "CK":
        z = load_eval(model, f"cos{c}")
        if z is None:
            out[c] = {"pending": f"sim{c} test not packed / evaluated"}
            continue
        d = R.Domain(f"sim{c}")
        r = rows_of(d, z["uid"])
        sign, px, uid = d.col("sign")[r], d.col("px_eq")[r], d.uid[r]
        pos = pd.Series(np.arange(len(r)), index=uid)
        pl = np.flatnonzero((sign == 1) & (px >= 100))
        mi = pos.reindex(d.col("twin_uid")[r][pl]).to_numpy()
        ok = ~np.isnan(mi)
        pl, mi = pl[ok], mi[ok].astype(int)
        idx = np.r_[pl, mi]
        y = np.r_[np.ones(len(pl)), np.zeros(len(mi))].astype(int)
        g = d.col("group")[r][idx].astype(str)
        Xo = z["orig_temporal"][idx]
        Xa = z["adapt_temporal"][idx] if model != "O" else Xo
        res = probe_pair(Xa, Xo, y, g, np.r_[px[pl], px[pl]], ((0, 500), (500, 1500), (1500, 1e9)))
        mr = res["main"]
        out[c] = line(mr, None if model == "O" else (mr["delta"] >= 0.05 and mr["delta_ci"][0] > 0), "delta >= +0.05 and CI low > 0",
                      bins={k: v for k, v in res.items() if k != "main"})
    return out


# ---------------------------------------------------------------- 4.2 behaviour
def b_p5(model):
    """B-p5 (reactive frames visible >= 500 px_eq; all reactive frames descriptive) and N-cutin: p5_exam.exam
    unchanged with the native plans as examinees (plan -> the P5 (20, 2) future, v2 = its 2 s speed)."""
    f = p5_frames(model)
    if f is None:
        return None
    E, t, past, fut, obs, null, pairs, X = f
    preds = {"native_orig": plan20(X["orig_plan"], P5_CAM_X)}
    if model != "O":
        preds["native_adapt"] = plan20(X["adapt_plan"], P5_CAM_X)
    o, n = E.deltas(obs, null, t, preds)
    ex = list(preds)
    ped = o.family.isin(E.PED_FAMILIES)
    out = {}
    v, _ = vis500(obs)
    for tag, sel in (("B-p5", ped & (v if v is not None else False)), ("B-p5 all reactive (descriptive)", ped)):
        if tag == "B-p5" and v is None:
            out[tag] = {"pending": "package C index/p5_obs.parquet (vis500)"}
            continue
        res = E.exam(o[sel], n, pairs, ex)
        fl = res["flips"].set_index(["examinee", "scope"])
        rows = {e: fl.loc[(e, "pooled")].to_dict() for e in ex if (e, "pooled") in fl.index}
        ok = None
        if model != "O" and "native_adapt" in rows:
            a_, o_ = rows["native_adapt"], rows["native_orig"]
            ok = (a_["flip_lo"] > a_["false_flip_null_oos"] + 0.10) and (a_["false_flip_null_oos"] <= o_["false_flip_null_oos"] + 0.03)
        out[tag] = line(rows, ok if tag == "B-p5" else None,
                        "flip CI low > own out-of-sample null false-flip + 10 pp and null false-flip <= O + 3 pp")
    cut = o.family.isin(["HighwayCutIn", "StaticCutIn", "ParkingCutIn"])
    res = E.exam(o[cut], n, pairs, ex)
    fl = res["flips"].set_index(["examinee", "scope"])
    rows = {e: fl.loc[(e, "pooled")].to_dict() for e in ex if (e, "pooled") in fl.index}
    ok = None if model == "O" or "native_adapt" not in rows else rows["native_adapt"]["flip_rate"] >= rows["native_orig"]["flip_rate"] - 0.05
    out["N-cutin"] = line(rows, ok, "cut-in flip rate >= O - 5 pp")
    return out


def b_real(model):
    """B-real: [slow(pedestrian frames) - slow(null frames)]_adapted - [same]_O, 5 ego-speed strata (pooled
    quintiles), stratum weights = pedestrian frames; cluster bootstrap over scene / sequence; per dataset too."""
    parts = []
    for name, dn in (("nusval", "nus"), ("wodval", "wod")):
        z = load_eval(model, name)
        if z is None:
            continue
        d = R.Domain(dn)
        r = rows_of(d, z["uid"])
        lab = d.col("labelled", False, bool)[r] & ~d.col("uncertain", False, bool)[r]
        ped = lab & d.col("ped_corr", False, bool)[r] & (d.col("ped_dist")[r] <= 30)
        nul = lab & ~d.col("vru_wide", False, bool)[r]
        v0 = d.col("ego_speed")[r] if "ego_speed" in d.s else z["orig_plan"][:, 0, 3]
        so = R.slow_flag(z["orig_plan"], v0)
        sa = R.slow_flag(z["adapt_plan"], v0) if model != "O" else so
        m = ped | nul
        parts.append(pd.DataFrame({"ds": name, "g": d.col("group")[r][m].astype(str), "ped": ped[m], "v0": v0[m], "so": so[m], "sa": sa[m]}))
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    df["bin"] = pd.qcut(df.v0.rank(method="first"), 5, labels=False)

    def delta(x, col):
        w = x[x.ped].groupby("bin").size()
        s = x.groupby(["bin", "ped"])[col].mean().unstack().reindex(index=w.index, columns=[False, True])
        d_ = (s[True] - s[False]).to_numpy()
        ok = np.isfinite(d_)                              # strata with both pedestrian and null frames
        return float((d_[ok] * w.to_numpy()[ok]).sum() / w.to_numpy()[ok].sum()) if ok.any() else np.nan

    def stat(sub):
        return delta(sub, "sa") - delta(sub, "so")
    out = {}
    for tag, x in (("pooled", df), *[(k, g) for k, g in df.groupby("ds")]):
        x = x.reset_index(drop=True)
        dlt = stat(x)
        ci = boot_ci(lambda i: stat(x.iloc[i]), x.g.to_numpy(), b=1000)
        nul_pp = float(x[~x.ped].sa.mean() - x[~x.ped].so.mean())
        ok = None if model == "O" or tag != "pooled" else (dlt >= 0.05 and ci[0] > 0 and nul_pp <= 0.02)
        out[tag] = line({"delta": dlt, "ci": ci, "null_slow_adapt_minus_orig": nul_pp, "slow_orig_ped": float(x[x.ped].so.mean()),
                         "slow_orig_null": float(x[~x.ped].so.mean()), "n_ped": int(x.ped.sum()), "n_null": int((~x.ped).sum())},
                        ok, "delta >= +5 pp, cluster CI low > 0, null slow <= O + 2 pp")
    return out


def b_score(model):
    """B-score-p5 / B-score-nus / B-cos / B-off via package S's score_plans on the native plans (adapted and O)."""
    sfn = R.score_api()
    if sfn is None:
        return {"pending": "package S score_plans"}
    out = {}
    for name, dn, kind in (("p5", "p5", "p5"), ("nusval", "nus", "nus"), ("cosC", "simC", "cos"), ("cosK", "simK", "cos"),
                           ("offdev", "off", "off")):
        z = load_eval(model, name)
        sc = R.load_score(dn)
        if z is None or sc is None:
            out[name] = {"pending": "eval or score table missing"}
            continue
        d = R.Domain(dn)
        r = rows_of(d, z["uid"])
        k = sc["pos"].reindex(z["uid"]).to_numpy()
        has = ~np.isnan(k)
        uid = z["uid"][has]
        ro = sfn(dn, uid, z["orig_plan"][has])
        ra = sfn(dn, uid, z["adapt_plan"][has]) if model != "O" else ro
        if ro is None or ra is None:
            out[name] = {"error": "score_plans failed (see log)"}
            continue
        so = {q: np.asarray(v, float) for q, v in ro.items()}
        sa = {q: np.asarray(v, float) for q, v in ra.items()}
        g = d.col("group")[r][has].astype(str)
        dS = sa["S"] - so["S"]
        sign = d.col("sign")[r][has] if "sign" in d.s else np.zeros(has.sum())
        res = {}
        if kind in ("p5", "cos"):
            for s_, tag in ((1, "x+"), (-1, "x-")):
                m = sign == s_
                if m.any():
                    ci = boot_ci(lambda i: dS[m][i].mean(), g[m])
                    res[tag] = {"delta": float(dS[m].mean()), "ci": ci, "nc_fail_orig": float(1 - so["NC"][m].mean()),
                                "nc_fail_adapt": float(1 - sa["NC"][m].mean()), "n": int(m.sum())}
            if kind == "p5" and model != "O" and "x+" in res:
                res["pass"] = bool(res["x+"]["ci"][0] > 0 and res.get("x-", {"delta": 0})["delta"] >= -0.02)
            if kind == "cos":
                kk = k[has].astype(int)
                top = sc["top"][kk]
                dist = np.linalg.norm(R.plan_ch(torch.from_numpy(z["adapt_plan" if model != "O" else "orig_plan"][has]))[:, None, :, :2].numpy()
                                      - sc["traj"][kk][..., :2], axis=-1).max(-1)
                res["within_0.5m_of_top"] = float((np.where(top, dist, np.inf).min(1) <= 0.5).mean())
                v2 = lambda p: R.v_at(p, 2.0)  # noqa: E731
                pl = np.flatnonzero(sign == 1)
                mi = pd.Series(np.arange(has.sum()), index=uid).reindex(d.col("twin_uid")[r][has][pl]).to_numpy()
                ok = ~np.isnan(mi)
                pa = z["adapt_plan" if model != "O" else "orig_plan"][has]
                res["v2_plus_minus"] = float(np.mean(v2(pa[pl[ok]]) - v2(pa[mi[ok].astype(int)])))
        elif kind == "nus":
            vru = sc["valid"][k[has].astype(int)]
            norm = d.col("normal", False, bool)[r][has]
            for tag, m in (("vru", vru), ("normal", norm & ~vru)):
                if m.any():
                    res[tag] = {"delta": float(dS[m].mean()), "ci": boot_ci(lambda i: dS[m][i].mean(), g[m]), "n": int(m.sum())}
            if model != "O" and "vru" in res and "normal" in res:
                res["pass"] = bool(res["vru"]["ci"][0] > 0 and res["normal"]["ci"][0] >= -0.01)
        else:
            corner = d.col("corner", -1)[r][has]
            for c in sorted(set(corner)):
                m = corner == c
                res[f"corner{c}"] = {"dac_orig": float(so["DAC"][m].mean()), "dac_adapt": float(sa["DAC"][m].mean()),
                                     "ddc_orig": float((so["DDC"][m] == 1).mean()), "ddc_adapt": float((sa["DDC"][m] == 1).mean()),
                                     "dS": float(dS[m].mean()), "n": int(m.sum())}
            if "rej" in sc["names"]:
                j = sc["names"].index("rej")
                kk = k[has].astype(int)
                res["rej_pass"] = float((sc["NC"][kk, j] * sc["DAC"][kk, j] * (sc["DDC"][kk, j] == 1)).mean())
        out[name] = res
    return out


def b_log(model):
    """B-log (descriptive): pedestrian frames, native plan ADE to the log future (WOD val, round-1 wod_ade), and
    the share of log-decelerating pedestrian frames where the plan decelerates too."""
    import op_adapt_readout as R1
    z = load_eval(model, "wodval")
    if z is None or model == "O":
        return None
    d = R.Domain("wod")
    r = rows_of(d, z["uid"])
    keys = d.col("key")[r].astype(str)
    ped = d.col("ped_corr", False, bool)[r]
    ev = {"key": keys[ped], "orig_plan": z["orig_plan"][ped], "adapt_plan": z["adapt_plan"][ped]}
    ade = R1.wod_ade(ev)
    dd = ade["adapt"] - ade["orig"]
    g = np.array([k.rsplit("-", 1)[0] for k in ev["key"]])
    fut = np.load(data_dir() / "processed/waymo_e2e/future.npy", mmap_mode="r")
    from jevdrive import waymo as W
    row = pd.Series(np.arange(len(W.load_index())), index=W.frame_names(W.load_index()))
    fr = np.stack([fut[row[k]][:, :2] for k in ev["key"]])
    v_log2 = np.linalg.norm(fr[:, 7] - fr[:, 6], axis=-1) / 0.25
    v0 = d.col("ego_speed")[r][ped] if "ego_speed" in d.s else z["orig_plan"][ped, 0, 3]
    dec = v_log2 < v0 - np.maximum(1, 0.2 * v0)
    return {"wod_ped_ade_delta": float(dd.mean()), "ci": boot_ci(lambda i: dd[i].mean(), g), "n": int(ped.sum()),
            "log_decel_frames": int(dec.sum()),
            "plan_also_decel_orig": float(R.slow_flag(ev["orig_plan"][dec], v0[dec]).mean()) if dec.any() else None,
            "plan_also_decel_adapt": float(R.slow_flag(ev["adapt_plan"][dec], v0[dec]).mean()) if dec.any() else None}


# ---------------------------------------------------------------- 4.3 no-harm
def n_drift(model):
    """N-drift, N-lead, N-ade: round-1 (b) code on nuScenes val normal keyframes and WOD val normal frames."""
    if model == "O":
        return None
    import op_adapt_readout as R1
    out = {}
    for name, dn in (("nusval", "nus"), ("wodval", "wod")):
        z = load_eval(model, name)
        if z is None:
            continue
        d = R.Domain(dn)
        r = rows_of(d, z["uid"])
        m = d.col("normal", False, bool)[r] & d.col("labelled", False, bool)[r]
        ev = {"adapt_plan": z["adapt_plan"], "orig_plan": z["orig_plan"], "adapt_lead_prob": z["adapt_lead_prob"],
              "orig_lead_prob": z["orig_lead_prob"], "adapt_lead": z["adapt_lead"], "orig_lead": z["orig_lead"]}
        out[name] = R1.drift_rows(ev, m, name)
    ok_d = all(v["drift_median"] <= 0.10 and v["drift_p95"] <= 0.50 for v in out.values()) and len(out) == 2
    ok_l = all(v["lead_prob_absdiff_median"] <= 0.02 and (np.isnan(v["lead_x_absdiff_median"]) or v["lead_x_absdiff_median"] <= 0.5)
               for v in out.values()) and len(out) == 2
    res = {"N-drift": line(out, ok_d, "median <= 0.10 m and p95 <= 0.50 m in both"),
           "N-lead": line({k: (v["lead_prob_absdiff_median"], v["lead_x_absdiff_median"]) for k, v in out.items()}, ok_l,
                          "lead_prob |delta| median <= 0.02, lead x |delta| median <= 0.5 m")}
    z = load_eval(model, "wodval")
    if z is not None:
        d = R.Domain("wod")
        r = rows_of(d, z["uid"])
        keys = d.col("key")[r].astype(str)
        ade = R1.wod_ade({"key": keys, "orig_plan": z["orig_plan"], "adapt_plan": z["adapt_plan"]})
        dd = ade["adapt"] - ade["orig"]
        g = np.array([k.rsplit("-", 1)[0] for k in keys])
        ci = boot_ci(lambda i: dd[i].mean() / ade["orig"][i].mean(), g, b=1000)
        res["N-ade"] = line({"rel": float(dd.mean() / ade["orig"].mean()), "ci": ci, "orig": float(ade["orig"].mean())},
                            ci[1] <= 0.02, "relative change CI high <= +2 %")
    return res


def n_rfs(model):
    """N-rfs: WOD-E2E val rater frames, official RFS of the native plan (openpilot_to_wod, rater_feedback_score),
    paired delta adapted - O with a cluster bootstrap over scenario clusters' frames."""
    z = load_eval(model, "rater")
    if z is None:
        return {"pending": "rater cache (readout.py rater) and eval"}
    from jevdrive import waymo as W
    from jevdrive import wod_zeroshot as Z
    sets = Z.load_sets()["rater"]
    d = R.Domain("rater")
    names = d.col("key")[rows_of(d, z["uid"])].astype(str)
    pos = pd.Series(np.arange(len(sets["name"])), index=sets["name"].astype(str))
    k = pos.reindex(names).to_numpy().astype(int)
    calib = json.loads((Z.root() / "op_calib.json").read_text())
    dev_xy = np.stack([np.array(calib[n.rsplit("-", 1)[0]]["1"]["extrinsic"]).reshape(4, 4)[:2, 3] for n in names])
    sp = W.init_speed(sets["past"][k])
    sc = {}
    for who in ("orig", "adapt") if model != "O" else ("orig",):
        p = z[f"{who}_plan"]
        wp = np.stack([Z.openpilot_to_wod(p[i, :, 0:3], p[i, :, 11], A.T_IDXS, dev_xy[i]) for i in range(len(p))])[:, :, :2]
        sc[who] = np.asarray(W.rater_feedback_score(wp, sets["traj"][k], sets["scores"][k], sp), float)
    cl = sets["cluster"][k].astype(str)
    agg = {w: W.rfs_by_cluster(s, cl)[0] for w, s in sc.items()}
    if model == "O":
        return line(agg, None, "O reference (TensorRT exam: 8.005)")
    stat = lambda i: W.rfs_by_cluster(sc["adapt"][i], cl[i])[0] - W.rfs_by_cluster(sc["orig"][i], cl[i])[0]  # noqa: E731
    ci = boot_ci(stat, names, b=1000)
    return line({"rfs": agg, "delta": agg["adapt"] - agg["orig"], "ci": ci, "n": len(names)}, ci[0] >= -0.10, "paired delta CI low >= -0.10")


# ---------------------------------------------------------------- 4.4 shortcut checks
def shortcut(model):
    """Pedestrian-region swap (x+ pedestrian region -> x- pixels): the adapted aux-logit gap and flips should fall to
    the null level; domain AUC per layer (C / K / real) descriptive; scorer sanity from S's tables."""
    out = {}
    sw = {}
    for c in "CK":
        z, zs = load_eval(model, f"cos{c}"), load_eval(model, f"cos{c}_swap")
        if z is None or zs is None:
            sw[c] = {"pending": "swap caches (package C: sim<c>_swap domains)"}
            continue
        who = "adapt" if model != "O" else "orig"
        d = R.Domain(f"sim{c}")
        r = rows_of(d, z["uid"])
        pos = pd.Series(np.arange(len(r)), index=z["uid"])
        ds = R.Domain(f"sim{c}_swap")
        su = ds.col("src_uid")[rows_of(ds, zs["uid"])]                       # swapped copy of which x+ row
        p = pos.reindex(su).to_numpy().astype(int)
        m = pos.reindex(d.col("twin_uid")[r][p]).to_numpy().astype(int)
        gap = z[f"{who}_head_t"][p, 0] - z[f"{who}_head_t"][m, 0]
        gap_s = zs[f"{who}_head_t"][:, 0] - z[f"{who}_head_t"][m, 0]
        v2 = lambda P: R.v_at(P, 2.0)  # noqa: E731
        sw[c] = {"logit_gap": float(gap.mean()), "logit_gap_swapped": float(gap_s.mean()),
                 "v2_gap": float((v2(z[f"{who}_plan"][p]) - v2(z[f"{who}_plan"][m])).mean()),
                 "v2_gap_swapped": float((v2(zs[f"{who}_plan"]) - v2(z[f"{who}_plan"][m])).mean()),
                 "half_or_more_remains": bool(gap_s.mean() >= 0.5 * gap.mean()) if gap.mean() > 0 else None}
    out["pedestrian region swap"] = sw
    # domain AUC per layer (C vs K vs real), descriptive
    feats = {}
    for name, lab in (("cosC", "C"), ("cosK", "K"), ("nusval", "real")):
        z = load_eval(model, name)
        if z is not None:
            feats[lab] = z
    if len(feats) == 3:
        import op_adapt_readout as R1
        res = {}
        for who in (("orig", "adapt") if model != "O" else ("orig",)):
            for layer in ("temporal", "vision"):
                for a_, b_ in (("C", "K"), ("C", "real"), ("K", "real")):
                    Xa, Xb = feats[a_][f"{who}_{layer}"], feats[b_][f"{who}_{layer}"]
                    n = min(len(Xa), len(Xb), 3000)
                    rng = np.random.default_rng(0)
                    ia, ib = rng.choice(len(Xa), n, replace=False), rng.choice(len(Xb), n, replace=False)
                    X = np.r_[Xa[ia], Xb[ib]]
                    y = np.r_[np.ones(n), np.zeros(n)].astype(int)
                    g = np.r_[ia % 50, 50 + ib % 50].astype(str)
                    s = R1.oof_probe(X, y, g)
                    res[f"{who} {layer} {a_}/{b_}"] = R.auc(y, s)
        out["domain AUC"] = res
    # scorer sanity (S tables): share of slots whose original plan is not in Top; all-actor vs visible-actor Top disagreement
    san = {}
    for dn in ("simC", "simK", "nus"):
        sc = R.load_score(dn)
        if sc is None or not (R.r2("t") / "samples" / f"{dn}.parquet").exists():
            continue
        d = R.Domain(dn)
        op = sc["names"].index("op")
        r = rows_of(d, sc["uid"])
        tr = d.col("split")[r] == "train"
        groups = {"x+": d.col("sign")[r] == 1, "x-": d.col("sign")[r] == -1} if dn.startswith("sim") else \
            {"VRU": sc["valid"], "normal": d.col("normal", False, bool)[r] & ~sc["valid"]}
        for k, m in groups.items():
            mm = m & tr
            san[f"{dn} {k} op not in Top"] = float((~sc["top"][mm, op]).mean()) if mm.any() else None
            if "top_all" in sc:
                san[f"{dn} {k} Top all vs visible differ"] = float((sc["top_all"][mm] != sc["top"][mm]).any(1).mean()) if mm.any() else None
    out["scorer sanity"] = san
    return out


# ---------------------------------------------------------------- read
def cmd_read(a):
    from jevdrive.runlog import RunLog
    rl = RunLog("op_adapt_r2", "readout")
    model = a.model
    out = rdir(model)
    look(model, "read", parts=a.parts)
    summ = {}
    jobs = {"R-nus": lambda: r_real(model, "nusval", "nus", 0.10), "R-wod": lambda: r_real(model, "wodval", "wod", 0.05),
            "S-p5": lambda: s_p5(model, rl), "S-cos": lambda: s_cos(model), "B-p5": lambda: b_p5(model),
            "B-real": lambda: b_real(model), "B-score": lambda: b_score(model), "B-log": lambda: b_log(model),
            "N-drift": lambda: n_drift(model), "N-rfs": lambda: n_rfs(model), "shortcut": lambda: shortcut(model)}
    for k, f in jobs.items():
        if a.parts and k not in a.parts:
            continue
        t0 = time.time()
        try:
            summ[k] = f()
        except Exception as e:  # noqa: BLE001  one missing input must not stop the other readouts
            summ[k] = {"error": f"{type(e).__name__}: {e}"}
        rl.info(f"{k} ({time.time() - t0:.0f} s): {json.dumps(summ[k], default=str)[:2000]}")
    nav = R.r2("readout") / "navsim" / f"navtest_{model}.json"
    if nav.exists():
        summ["N-nav"] = json.loads(nav.read_text())
    R.save_json(out / "summary.json", summ)
    rl.info(f"-> {out / 'summary.json'}")


# ---------------------------------------------------------------- 4.5 verdict
def flat_pass(summ: dict) -> dict:
    """Registered lines of one model -> {line: pass}."""
    g = lambda *k: _get(summ, *k)  # noqa: E731
    return {"R-nus": g("R-nus", "pass"), "R-wod": g("R-wod", "pass"), "S-p5": g("S-p5", "S-p5", "pass"),
            "S-p5-all": g("S-p5", "S-p5-all", "pass"),
            "S-cos": (g("S-cos", "C", "pass") and g("S-cos", "K", "pass")) if None not in (g("S-cos", "C", "pass"), g("S-cos", "K", "pass")) else None,
            "B-p5": g("B-p5", "B-p5", "pass"), "B-real": g("B-real", "pooled", "pass"),
            "N-drift": g("N-drift", "N-drift", "pass"), "N-ade": g("N-drift", "N-ade", "pass"), "N-lead": g("N-drift", "N-lead", "pass"),
            "N-nav": g("N-nav", "pass"), "N-rfs": g("N-rfs", "pass"), "N-cutin": g("B-p5", "N-cutin", "pass"),
            "B-score-p5": g("B-score", "p5", "pass"), "B-score-nus": g("B-score", "nusval", "pass")}


def _get(d, *k):
    for x in k:
        if not isinstance(d, dict) or x not in d:
            return None
        d = d[x]
    return d


def outcome(p: dict) -> list:
    """§4.5 table: every row whose condition holds (the rows are readings and can co-occur, e.g. P-rep and P-size);
    ["none"] when no row holds."""
    N = all(p[k] for k in ("N-drift", "N-ade", "N-lead", "N-nav", "N-rfs", "N-cutin"))
    Rr = p["R-nus"] and p["R-wod"]
    S = p["S-p5"] and p["S-cos"]
    rows = []
    if Rr and S and p["B-p5"] and p["B-real"] and N:
        rows.append("P")
    if Rr and S and N and not (p["B-p5"] and p["B-real"]):
        rows.append("P-rep")
    if Rr and N and p["S-p5"] and not p["S-p5-all"]:
        rows.append("P-size (read the < 500 px_eq bins: they must not move)")
    if Rr and N and not (p["S-p5"] or p["S-p5-all"] or p["S-cos"]) and not p["B-p5"]:
        rows.append("real-only")
    if (p["S-p5"] or p["S-p5-all"] or p["S-cos"] or p["B-p5"]) and (not Rr or not p["B-real"] or not N):
        rows.append("sim-dominant")
    return rows or ["none"]


def cmd_verdict(a):
    look("all", "verdict")
    res = {}
    lam = json.loads((R.r2() / "lambda_s.json").read_text())["lam_s"]
    for arm in MAIN:
        runs = sorted(p.name.replace("train-", "") for p in R.r2().glob(f"train-{arm}-s*")
                      if (p / "DONE").exists() and p.name.endswith(f"-ls{lam:g}"))
        per = {}
        for tag in runs:
            s = rdir(tag) / "summary.json"
            if s.exists():
                per[tag] = flat_pass(json.loads(s.read_text()))
        if len(per) < R.ARMS[arm].seeds:
            res[arm] = {"pending": f"{len(per)} / {R.ARMS[arm].seeds} seeds read", "per_seed": per}
            continue
        keys = next(iter(per.values())).keys()
        allp = {k: (all(v[k] for v in per.values()) if all(v[k] is not None for v in per.values()) else None) for k in keys}
        missing = [k for k, v in allp.items() if v is None]
        res[arm] = {"per_seed": per, "all_seeds": allp, "missing": missing,
                    "outcome": None if missing else outcome(allp),
                    "B-score note": None if missing else (
                        "slowed but not necessarily the right slowing" if (allp["B-p5"] and allp["B-real"]) and not (allp["B-score-p5"] and allp["B-score-nus"])
                        else "score better but not via pedestrian slowing" if (allp["B-score-p5"] and allp["B-score-nus"]) and not (allp["B-p5"] and allp["B-real"])
                        else "")}
    R.save_json(R.r2("readout") / "verdict.json", res)
    print(json.dumps({k: v.get("outcome", v.get("pending")) for k, v in res.items()}, indent=1))


# ---------------------------------------------------------------- navsim (N-nav, §4.6 navhard) with the port
@torch.no_grad()
def nav_plans(models: dict, data: str, dev, bs=32):
    """Native plan at t0 of every token of an op_lb run dir, computed with the port on the same frames and step
    schedule as `op_lb.py run` (Cinque, 31 steps at 20 Hz, desire none): the policy context is the steps
    30, 26, .., 2 (and a zero hidden for -2), each hidden = stage 4 of the trunk of (image at step s - 4, image s)."""
    import op_lb as L
    mt = L.meta(data)
    keys = L.Keys(data)
    syn = np.load(L.root(data) / "gimm.npy", mmap_mode="r")
    ts, src = L._steps(0.0, False)
    assert len(ts) == 31 and all(s != "p" for s, _ in src)
    steps = np.arange(30, -1, -4)[::-1]                            # 2, 6, .., 30 (8 frames, oldest first)
    n = len(mt["names"])
    base = next(iter(models.values()))
    out = {k: np.zeros((n, 33, 15), np.float32) for k in models} | {f"{k}_std": np.zeros((n, 33, 15), np.float32) for k in models}
    pi = A.plan_index(base.net.slices)
    ps = np.arange(base.net.slices["plan"].start + 495, base.net.slices["plan"].start + 990)
    from tqdm import tqdm
    for i0 in tqdm(range(0, n, bs), desc=data, mininterval=60):
        rows = np.arange(i0, min(i0 + bs, n))
        kf = keys[rows]
        sf = np.asarray(syn[rows[0]:rows[-1] + 1])
        img = lambda j, s: (kf[j][src[s][1]] if src[s][0] == "k" else sf[j][src[s][1]]) if s >= 0 else np.zeros((2, 6, 128, 256), np.uint8)  # noqa: E731
        cur = np.stack([[img(j, s) for s in steps] for j in range(len(rows))])
        prev = np.stack([[img(j, s - 4) for s in steps] for j in range(len(rows))])
        B = len(rows)
        c, p = torch.as_tensor(cur).to(dev).flatten(0, 1), torch.as_tensor(prev).to(dev).flatten(0, 1)
        tr = base.net.run_batched(A.vision_feeds(p, c), [A.TRUNK_OUT])[A.TRUNK_OUT][:, 0].reshape(B, 8, 1024, 8, 16)
        trunk = torch.cat([torch.zeros_like(tr[:, :1]), tr], 1)
        valid = torch.ones(B, 9, dtype=torch.bool, device=dev)
        valid[:, 0] = False
        tc = torch.tensor([[0.0, 1.0] if l else [1.0, 0.0] for l in np.asarray(mt["lht"])[rows]], device=dev)
        for k, m in models.items():
            o = A.stage4_policy(m.net, trunk, AT, tc, valid)["outputs"].float()
            out[k][rows] = o[:, pi].cpu().numpy().reshape(B, 33, 15)
            out[f"{k}_std"][rows] = np.exp(np.minimum(o[:, ps].cpu().numpy(), 11)).reshape(B, 33, 15)
    return mt["names"], out


def cmd_navsim(a):
    split = {"navtest": "lb_navtest", "navhard": "lb_navhard"}[a.split]
    if a.split == "navhard":
        v = R.r2("readout") / "verdict.json"
        if not a.final or not v.exists():
            raise SystemExit("§4.6: navhard runs once, after every other readout and the verdict (--final)")
    look(",".join(a.models), f"navsim {a.split}")
    dev = torch.device("cuda")
    models = {"port": load_model("O", dev)[0]} | {m: load_model(m, dev)[0] for m in a.models}
    import op_lb as L
    names, out = nav_plans(models, split, dev, a.batch)
    pdir = L.root(split, "plans")
    stems = []
    for k in models:
        mu = out[k]
        stem = f"gimm@cinque_r2{k}"
        np.savez(pdir / f"{stem}.npz", names=np.array(names), plan_pos=mu[:, :, 0:3], plan_vel=mu[:, :, 3:6], plan_yaw=mu[:, :, 11],
                 plan_mu=mu, plan_std=out[f"{k}_std"], steps=31, info=json.dumps({"model": f"port {k}", "source": "op_adapt_r2_readout"}))
        stems.append(stem)
    env = dict(os.environ, OPI_ROOT="op_lb")
    jev = str(data_dir() / "envs" / "jevdrive" / "bin" / "python")
    ver, sp = ("v1", "navtest") if a.split == "navtest" else ("v2", "navhard_two_stage")
    repo = Path(__file__).resolve().parents[1]
    subprocess.check_call([jev, str(repo / "scripts/op_interp.py"), "nav-export", "--data", split, "--adapters", "base",
                           "--plans", *stems], env=env)
    subprocess.check_call([str(repo / "scripts/op_interp_score.sh"), a.cpus, split, ver, sp], env=env)
    rep = subprocess.run([jev, str(repo / "scripts/op_interp.py"), "nav-report", "--data", split, "--ver", ver, "--split", sp,
                          "--refs", "gimm-cinque_r2port__base"], env=env, capture_output=True, text=True)
    (R.r2("readout", "navsim") / f"{a.split}_report.txt").write_text(rep.stdout + rep.stderr)
    res = read_nav_scores(split, ver, sp, [s.replace("@", "-") + "__base" for s in stems])
    for m in a.models:
        o, x = res.get("gimm-cinque_r2port__base"), res.get(f"gimm-cinque_r2{m}__base")
        entry = {"port_O": o, "model": x, "ref_trt_O": {"navtest": 84.18, "navhard": 33.33}[a.split]}
        if a.split == "navtest" and o and x:
            entry |= {"pass": bool(x["score"] >= o["score"] - 1.0), "line": ">= O - 1 (O = port O, same run)"}
        if a.split == "navhard" and o:
            entry["port_O_reproduces_33.33"] = bool(abs(o["score"] - 33.33) <= 0.1)
        R.save_json(R.r2("readout", "navsim") / f"{a.split}_{m}.json", entry)
    print(json.dumps(res, indent=1, default=str))


def read_nav_scores(split, ver, sp, names) -> dict:
    """Mean score (navtest PDMS) or EPDMS combined + stage rows (navhard) of each scored pose file, per-token scores kept."""
    res = {}
    pfx = f"opi_{split}_"
    for n in names:
        fs = sorted((data_dir() / "runs" / "navsim" / "eval").glob(f"{ver}_{sp}_{pfx}{n}/*/*.csv"))
        if not fs:
            continue
        df = pd.read_csv(fs[-1])
        if "token" in df:
            summ = df[df.token.astype(str).str.startswith("extended_pdm_score")]
            tok = df[~df.token.astype(str).str.startswith("extended_pdm_score") & (df.token != "average")]
        else:
            summ, tok = df.iloc[:0], df
        r = {"csv": str(fs[-1]), "n": int(len(tok))}
        if len(summ):
            r |= {row.token: float(row.score) for row in summ.itertuples()}
            r["score"] = r.get("extended_pdm_score_combined")
        else:
            r["score"] = float(100 * tok.score.mean()) if tok.score.max() <= 1 else float(tok.score.mean())
        res[n] = r
    return res


# ---------------------------------------------------------------- rater cache (N-rfs)
def cmd_rater(a):
    """Trunk cache of the WOD-E2E val rater frames: each target with its 18 predecessors (WOD protocol: pair (f-2, f),
    context stride 2 frames), rendered like the round-1 WOD val cache (scripts/op_adapt_cache.py wod_job)."""
    from concurrent.futures import ProcessPoolExecutor
    import op_adapt_cache as C
    import wod_zeroshot_openpilot as WZ
    from drive_backbones_openpilot import bounded_map
    from jevdrive import drive_backbones as DB
    from jevdrive import wod_zeroshot as Z
    names = Z.load_sets()["rater"]["name"].astype(str)
    out = R.r2("t", "cache-rater")
    sts = []
    for n in names:
        h = Z.history_names(n, 18)
        sts.append({"key": n, "names": h, "targets": [len(h) - 1]})
    sts = [s for s in sts if not (out / f"{s['key']}.npz").exists()]
    plan = json.loads((DB.root() / DB.plan_name("subset")).read_text())
    calib = json.loads((Z.root() / "op_calib.json").read_text())
    with ProcessPoolExecutor(a.workers, initializer=WZ._init, initargs=(plan["spans"], calib, str(data_dir() / "datasets" / "waymo_e2e" / "front3"))) as ex:
        list(ex.map(int, range(a.workers)))
        net = A.load("cinque", torch.float16).cuda()
        for k, prev, cur, meta in bounded_map(ex, C.wod_job, sts, 2 * a.workers):
            np.savez(out / f"{k}.npz", trunk=C.trunk(net, prev, cur, 64), **meta)
    rows = []
    for f in sorted(out.glob("*.npz")):
        with np.load(f) as z:
            nm, tg, c = z["names"].astype(str), int(z["targets"][0]), int(z["stride"])
        loc = tg - c * np.arange(A.CONTEXT - 1, -1, -1)
        rows.append({"cache": str(f), "row": tg, "lctx": np.where(loc >= 0, loc, -1), "key": nm[tg], "tc0": 1.0, "tc1": 0.0,
                     "split": "val", "group": nm[tg].rsplit("-", 1)[0], "domain": "rater"})
    ix = pd.DataFrame(rows)
    ix["uid"] = 8_000_000_000 + np.arange(len(ix))
    R.pack("rater", ix)
    print(f"rater: {len(ix)} frames packed")


# ---------------------------------------------------------------- §6 item 5: full port forward vs cache path
def cmd_fullfwd(a):
    """cache -> stage 4 -> policy against the whole vision stack from the images (op_adapt.full_policy) on 3 Cosmos
    pairs x 4 streams, `temporal` correlation per stream >= 0.9999 (round-1 check)."""
    import op_cosmos_probe as CP
    import cosmos_openpilot as CO
    dev = torch.device("cuda")
    net = A.load("cinque", torch.float16).to(dev).eval()
    d = R.Domain("simC")
    pairs = sorted(set(d.col("key").astype(str)))[: 3]
    CP._init(CO.maps())
    res = []
    for pr in pairs:
        _, frames, *_ = CP.pair_job(pr)
        for si in range(4):
            fr = torch.as_tensor(frames[si]).to(dev)
            prev = torch.cat([torch.zeros_like(fr[:1]), fr[:-1]])
            j = torch.arange(9, 24, device=dev)[:, None] + torch.arange(-8, 1, device=dev)[None]
            with torch.no_grad():
                full = A.full_policy(net, prev[j.clamp(min=0)], fr[j.clamp(min=0)], AT)["select_4"].float()
                x = np.load(R.r2("cache-sim") / f"{pr}.npy", mmap_mode="r")[si * 24:(si + 1) * 24]
                T = torch.as_tensor(np.asarray(x)).to(dev)
                cache = A.stage4_policy(net, T[j.clamp(min=0)], AT, None, j >= 0)["select_4"].float()
            full = torch.where((j >= 0).all(1)[:, None], full, cache)                   # slots 9+ have a full context
            cc = float(np.corrcoef(full.cpu().numpy().ravel(), cache.cpu().numpy().ravel())[0, 1])
            res.append({"pair": pr, "stream": si, "corr": cc})
    out = {"rows": res, "min_corr": min(r["corr"] for r in res), "pass": min(r["corr"] for r in res) >= 0.9999}
    R.save_json(Path(a.out), out)
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("eval")
    p.add_argument("--model", required=True)
    p.add_argument("--sets", nargs="+", default=list(SETS))
    p = sp.add_parser("read")
    p.add_argument("--model", required=True)
    p.add_argument("--parts", nargs="*", default=None)
    sp.add_parser("verdict")
    p = sp.add_parser("navsim")
    p.add_argument("--split", choices=("navtest", "navhard"), required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--cpus", default="40-47")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--final", action="store_true")
    p = sp.add_parser("rater")
    p.add_argument("--workers", type=int, default=8)
    p = sp.add_parser("fullfwd")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"eval": cmd_eval, "read": cmd_read, "verdict": cmd_verdict, "navsim": cmd_navsim, "rater": cmd_rater,
     "fullfwd": cmd_fullfwd}[a.cmd](a)


if __name__ == "__main__":
    main()
