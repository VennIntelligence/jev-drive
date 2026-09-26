"""Night queue 3, lane C, Q1: every examinee that needs no re-recording, read on the P6 v0 exam
(todos/2026-09-26-night-queue-3.md, Q1, rule 7, the [C] entries).

  heads    our readouts trained on P5 v1 BA exactly as stored, the P6 exam frames riding along as rows that never enter a
           fit or a standardisation (elicitation I3's protocol, elicit_i3.exam): per route fold `ridge_late` and M-C pair
           (reactivity_mc.fit_fold, checked against the stored M-C run), `cls_late` (night2_n3.p5cls's recipe, seed 0,
           checked against its stored anchors) and the E5 student with image-plane tokens (night2_n4 arm B, seed 0);
           prediction = mean over the 5 fold models (cls_late: mean of the folds' top-1 anchors, as I3)
           -> processed/carla_p6/nq3_p5heads.npz
  collect  every examinee's (n, 20, 2) futures aligned to the P6 index (NaN where not read): openpilot native plan
           (op_streams_plan), TFv6 waypoints (the recorder's shadow, 2 s) and TFv6 target speed (longitudinal only), the
           P5-trained heads, the NAVSIM family (processed/top10_exam/p6/<model>.npz), Alpamayo 1.5 (nq3_alpamayo.npz)
  judge    jevdrive.nq3_p6.judge_one for each examinee -> research/results/nq3/q1/*.csv|md, world-mode agreement,
           Alpamayo's language-vs-trajectory inconsistency
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import nq3_p6 as J
from .common import data_dir, get_logger

log = get_logger(__name__)
SET = "carla_p6"
MODELS = ("cinque", "lebowski")
REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "research" / "results" / "nq3" / "q1"
N4_FIT = "runs/night2/n4-fit/20260926-125031"
P5CLS = "processed/night2/n3/p5cls_s0.npz"
NAV = {"sparsedrivev2": "SparseDriveV2", "ztrs": "ZTRS", "drivor": "DrivoR", "wajepa": "WA-JEPA"}


def proc(*p) -> Path:
    return data_dir() / "processed" / SET / Path(*p)


# ---------------------------------------------------------------- P5-trained readouts, zero-shot

def _p6_rows():
    from . import nq3_feats as NF
    t6 = NF.rows()
    full = pd.read_parquet(proc("index.parquet"), columns=["frame_name"])
    at = pd.Series(np.arange(len(full)), index=full.frame_name)[t6.frame_name].to_numpy()
    return t6, np.load(proc("past.npy"))[at]


def heads(rl, models=MODELS, folds=None, out: Path | None = None):
    """`models` / `folds` / `out` restrict and redirect the run for the subset check before the batch."""
    from sklearn.model_selection import GroupShuffleSplit
    from . import elicit_e5 as E5, elicit_i3 as I, night2_n3 as N3, night2_n4 as N4, nq3_feats as NF
    from . import navsim_heads as H, p5_exam as E, p5_openpilot, p5_pairs as P, planner, reactivity_mc as MC, traj
    dev = "cuda"
    MC.EIGH_DEVICE = "cuda"
    with I.p5_set(I.BA):
        t, past, fut, obs, null, pairs = E.load()
        Q = P.load_features(t, ("L18_last",))["L18_last"]
        op = p5_openpilot.load(t, MODELS, sub="op_streams_vis")
        fold = E.folds(t, pairs)
    tokB = np.load(N4.out("tokB.npy"))
    t6, past6 = _p6_rows()
    with I.p5_set(SET):
        op6 = p5_openpilot.load(t6, MODELS, sub="op_streams_vis")
        fd = proc("features")
        qn = set(pd.concat([pd.read_parquet(c / "index.parquet") for c in sorted(fd.glob("c[0-9]*"))
                            if (c / "meta.json").exists()] or [pd.DataFrame({"frame_name": []})]).frame_name)
        hasq = t6.frame_name.isin(qn).to_numpy()
        Q6 = np.zeros((len(t6), Q.shape[1]), np.float32)
        Q6[hasq] = P.load_features(t6[hasq], ("L18_last",))["L18_last"]
    if proc("nq3_tokB_index.parquet").exists():
        ti = pd.read_parquet(proc("nq3_tokB_index.parquet")).frame_name
        tok6 = np.load(proc("nq3_tokB.npy"))[pd.Series(np.arange(len(ti)), index=ti)[t6.frame_name].to_numpy()]
    else:                                      # the pre-batch check only (P6 rows never enter a fit)
        assert out is not None, "image-plane tokens missing: run nq3_feats tokens first"
        tok6 = np.zeros((len(t6), tokB.shape[1]), np.float32)
    n, n6 = len(t), len(t6)
    rl.log.info("P5 v1 BA %d rows; P6 %d exam rows (%d with Qwen)", n, n6, int(hasq.sum()))
    ta = pd.concat([t[["frame_name", "role", "base_id", "intent"]],
                    t6[["frame_name", "base_id", "intent"]].assign(role="p6", base_id="p6:" + t6.base_id)], ignore_index=True)
    fold_a = np.r_[fold, np.full(n6, -2)]
    F = torch.as_tensor(np.r_[fut.reshape(n, -1), np.zeros((n6, 40), np.float32)], device=dev)
    Ego = torch.as_tensor(np.r_[E.ego_input(t, past), E.ego_input(t6, past6)], device=dev)
    Qa = torch.as_tensor(np.r_[Q, Q6], device=dev)
    Ba = torch.as_tensor(np.r_[tokB, tok6], device=dev)
    pos = pd.Series(np.arange(n), index=t.frame_name)
    pr_ip = np.r_[pos[obs.fn_plus].to_numpy(), pos[null.fn_plus].to_numpy()]
    pr_im = np.r_[pos[obs.fn_minus].to_numpy(), pos[null.fn_null].to_numpy()]
    pr_group = np.r_[obs.base_id.to_numpy(), null.base_id.to_numpy()].astype(str)
    obs_rows = np.flatnonzero(t.role.to_numpy() == "obs")
    role = ta.role.to_numpy()
    ref_mc = np.load(data_dir() / I.MC_RUN / "preds_obs.npz")
    ref_n4 = np.load(data_dir() / N4_FIT / "preds_obs.npz")
    at = pd.Series(np.arange(len(ref_mc["rows"])), index=ref_mc["rows"])
    I6 = np.arange(n, n + n6)
    preds, checks = {}, []
    mB = torch.as_tensor(np.arange(Ba.shape[1]) % N4.TOK_D == N4.TOK_D - 1, device=dev)

    def add(name, v):
        preds.setdefault(name, np.zeros((n6, 20, 2), np.float64))
        preds[name] += v[I6].reshape(-1, 20, 2).cpu().double().numpy() / E.K_FOLDS

    def z(X, tr, mask=None):                       # night2_n4.fit's standardisation
        mu, sd = X[tr].mean(0), X[tr].std(0, correction=0).clamp_min(1e-6)
        s_ = (X - mu) / sd
        return (s_ if mask is None else torch.where(mask, X, s_)) / np.sqrt(X.shape[1])

    MODELS_ = models
    folds = list(folds) if folds is not None else list(range(E.K_FOLDS))
    for m in MODELS_:
        Xop = torch.as_tensor(np.r_[op[f"op-{m} temporal"], op6[f"op-{m} temporal"]], device=dev)
        for f in folds:
            o = MC.fit_fold(f, fold_a, ta, F, Ego, Xop, Qa, pr_ip, pr_im, pr_group, rl, m, arms=("pair",))
            ev = obs_rows[fold[obs_rows] == f]
            for arm, key in (("M-C pair", f"M-C pair [{m}]"), ("prior", f"prior [{m}]")):
                d = float(np.abs(o[arm][ev].reshape(-1, 20, 2).cpu().numpy() - ref_mc[key][at[ev].to_numpy()]).max())
                checks.append({"model": m, "fold": f, "arm": arm, "max_abs_diff": d})
                assert d < 1e-3, f"{m} fold {f} {arm} does not reproduce the stored run ({d})"
            add(f"ridge_late [{m}]", o["prior"])
            add(f"M-C pair [{m}]", o["M-C pair"])
            # E5 student, image-plane tokens (N4 arm B), seed 0, on this fold's prior
            tr = np.flatnonzero((role == "train") & (fold_a != f))
            prior = o["prior"]
            keep = fold[pr_ip] != f
            ip, im, grp = pr_ip[keep], pr_im[keep], pr_group[keep]
            R = (F[ip] - F[im]) - (prior[ip] - prior[im])
            X = torch.cat([z(Xop, tr), z(Ba, tr, mB)], 1).float()
            dlt = E5.train_student(X, ip, im, R, tr, grp, None, 0, rl, f"{m} f{f} B")
            v = prior + dlt
            d = float(np.abs(v[ev].reshape(-1, 20, 2).cpu().numpy() - ref_n4[f"N4 B s0 [{m}]"][at[ev].to_numpy()]).max())
            checks.append({"model": m, "fold": f, "arm": "E5 student B s0", "max_abs_diff": d})
            add(f"E5 student B [{m}]", v)
            rl.log.info("%s fold %d: heads done (student vs stored N4 max |diff| %.3g m)", m, f, d)
            del X
        del Xop
        torch.cuda.empty_cache()
    # cls_late: night2_n3.p5cls (seed 0) with the P6 rows in place of I3's
    seed = 0
    Axy_sum = {m: np.zeros((n6, 20, 2)) for m in MODELS_}
    ref_cls = np.load(data_dir() / P5CLS)
    seq = ta.base_id.to_numpy().astype(str)
    futa = np.r_[fut, np.zeros((n6, 20, 2), np.float32)]
    for f in folds:
        tr = np.flatnonzero((role == "train") & (fold_a != f))
        ev = obs_rows[fold[obs_rows] == f]
        a, b = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(tr, groups=seq[tr]))
        fit_r, sel_r = tr[a], tr[b]
        A = traj.kmeans(F[tr], H.K, seed=seed)
        ids = traj.nearest(F, A, 1)[0][:, 0]
        Axy = A.reshape(H.K, 20, 2).cpu().numpy()
        tgt = (ids[:, None], np.ones((len(ids), 1), np.float32))
        Xe = planner.standardize(Ego, tr)
        lam_e = N3._pick_cls(Xe, tgt, fit_r, sel_r, Axy, futa)
        We, _ = planner.ce_solve(Xe, tgt, tr, [lam_e], H.K)
        off = planner.linear_apply(We, Xe, np.arange(n + n6))[0]
        inner = H._group_folds(seq[tr], 5, seed=seed)
        for k in range(5):
            Wk, _ = planner.ce_solve(Xe, tgt, tr[inner != k], [lam_e], H.K)
            off[tr[inner == k]] = planner.linear_apply(Wk, Xe, tr[inner == k])[0]
        for m in MODELS_:
            Xf = planner.standardize(torch.as_tensor(np.r_[op[f"op-{m} temporal"], op6[f"op-{m} temporal"]], device=dev), tr)
            lam_l = N3._pick_cls(Xf, tgt, fit_r, sel_r, Axy, futa, off)
            Wl, _ = planner.ce_solve(Xf, tgt, tr, [lam_l], H.K, offset=off)
            top = planner.cls_topk(Wl, Xf, np.r_[ev, I6], 1, offset=off)[:, 0, 0]
            rp = pd.Series(np.arange(len(ref_cls["p5_rows"])), index=ref_cls["p5_rows"])[ev].to_numpy()
            d = float(np.abs(Axy[top[:len(ev)]] - ref_cls[f"p5_{m}"][rp]).max())
            checks.append({"model": m, "fold": f, "arm": "cls_late", "max_abs_diff": d})
            Axy_sum[m] += Axy[top[len(ev):]] / E.K_FOLDS
            del Xf, Wl
        del Xe, We, off
        torch.cuda.empty_cache()
    for m in MODELS_:
        preds[f"cls_late [{m}]"] = Axy_sum[m]
    ck = pd.DataFrame(checks)
    ck.to_csv(rl.dir / "head_checks.csv", index=False)
    rl.log.info("head checks (max |diff| vs stored P5 runs)\n%s", ck.groupby(["model", "arm"]).max_abs_diff.max().to_string())
    out_path = out
    res = {k: v.astype(np.float32) for k, v in preds.items()}
    for k in list(res):
        if k.startswith("M-C"):
            res[k][~hasq] = np.nan                   # no Qwen features on the negotiation / mirror frames
    np.savez_compressed(Path(out_path) if out_path else proc("nq3_p5heads.npz"), frame_name=t6.frame_name.to_numpy().astype(str), **res)
    return res


# ---------------------------------------------------------------- collect

def _tfv6_world(adir: str, names: list, frames: list):
    rec = {}
    for line in open(Path(adir) / "tfv6.jsonl"):
        r = json.loads(line)
        rec[r["frame"]] = r
    out = []
    for nm, fr in zip(names, frames):
        r = rec.get(fr)
        if r is None:
            out.append((nm, None, np.nan))
            continue
        out.append((nm, np.asarray(r["pred_future_waypoints"], np.float32),
                    float(np.dot(r["pred_target_speed_distribution"], [0.0, 4.0, 8.0, 10.0, 13.88888888, 16.0, 17.77777777, 20.0]))))
    return out


def tfv6(t: pd.DataFrame, need: set, workers: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """TFv6 shadow outputs at the camera frame (the recorder runs TFv6 on the same tick): waypoints (8 x 0.25 s) and
    expected target speed. Raw TFv6 frame; the lateral sign is fixed against the expert future in `collect`."""
    from joblib import Parallel, delayed
    s = t[t.frame_name.isin(need)]
    adir = s.files.map(lambda f: f[0].rsplit("/cams/", 1)[0])
    jobs = [(d, g.frame_name.tolist(), g.frame.tolist()) for d, g in s.groupby(adir)]
    res = [x for r in Parallel(workers)(delayed(_tfv6_world)(*j) for j in jobs) for x in r]
    pos = pd.Series(np.arange(len(t)), index=t.frame_name)
    wp, ts = np.full((len(t), 20, 2), np.nan, np.float32), np.full(len(t), np.nan, np.float32)
    for nm, w, v in res:
        if w is not None:
            wp[pos[nm], :8] = w
            ts[pos[nm]] = v
    return wp, ts


def collect() -> tuple[pd.DataFrame, dict, dict]:
    """(P6 index, {examinee: (n, 20, 2)}, {examinee: note}); target-speed channel as a constant-speed pseudo-future
    (x = v t, y NaN) so the judge reads only its 2 s speed."""
    t = pd.read_parquet(proc("index.parquet"))
    fut = np.load(proc("future.npy"))
    need = set(pd.read_parquet(proc("nq3_exam_frames.parquet")).frame_name)
    pos = pd.Series(np.arange(len(t)), index=t.frame_name)
    preds, notes = {}, {}
    for m in MODELS:                                            # openpilot native plan
        d = proc("op_streams_plan", m)
        arr = np.full((len(t), 20, 2), np.nan, np.float32)
        for f in sorted(d.glob("*.npz")):
            z = np.load(f)
            ok = pd.Index(z["name"]).isin(pos.index)
            arr[pos[z["name"][ok]].to_numpy()] = z["plan"][ok]
        preds[f"openpilot {m} native plan"] = arr
    wp, ts = tfv6(t, need)
    sel = np.flatnonzero(~np.isnan(wp[:, 7, 1]) & (t.world == "x10").to_numpy())
    c = np.corrcoef(wp[sel, 7, 1], fut[sel, 7, 1])[0, 1]
    if c < 0:
        wp[..., 1] *= -1
    notes["TFv6 waypoint"] = f"y sign {'flipped' if c < 0 else 'kept'} (corr with the expert's y(2 s) on x10 frames {c:+.2f}); horizon 2 s"
    preds["TFv6 waypoint"] = wp
    tsf = np.full((len(t), 20, 2), np.nan, np.float32)
    tsf[..., 0] = ts[:, None] * (0.25 * np.arange(1, 21))[None]
    tsf[..., 1] = np.where(np.isnan(ts), np.nan, 0.0)[:, None]      # no lateral channel: Delta_lat = 0, never a flip
    preds["TFv6 target speed"] = tsf
    notes["TFv6 target speed"] = "longitudinal only (route + target speed channel): stop substitution"
    if proc("nq3_p5heads.npz").exists():
        # Trusted output of heads() above; older caches store pandas names as objects.
        z = np.load(proc("nq3_p5heads.npz"), allow_pickle=True)
        at = pos[z["frame_name"]].to_numpy()
        for k in z.files:
            if k != "frame_name":
                a = np.full((len(t), 20, 2), np.nan, np.float32)
                a[at] = z[k]
                preds[k] = a
    for k, name in NAV.items():
        f = data_dir() / "processed" / "top10_exam" / "p6" / f"{k}.npz"
        if f.exists():
            z = np.load(f, allow_pickle=True)
            a = np.full((len(t), 20, 2), np.nan, np.float32)
            ok = pd.Index(z["frame_name"]).isin(pos.index)
            a[pos[z["frame_name"][ok]].to_numpy()] = z["grid"][ok]
            preds[name] = a
    f = proc("nq3_alpamayo.npz")
    if f.exists():
        z = np.load(f, allow_pickle=True)
        a = np.full((len(t), 20, 2), np.nan, np.float32)
        ok = pd.Index(z["frame_name"]).isin(pos.index)
        a[pos[z["frame_name"][ok]].to_numpy()] = z["grid"][ok]
        preds["Alpamayo 1.5"] = a
    preds["PDM-Lite expert (future)"] = fut
    notes["PDM-Lite expert (future)"] = "reference row: the expert's own driven future"
    return t, preds, notes


# ---------------------------------------------------------------- judge

def judge(rl, out: Path | None = None) -> pd.DataFrame:
    out = out or RESULTS
    out.mkdir(parents=True, exist_ok=True)
    t, preds, notes = collect()
    p = J.pairs(SET)
    rows, per, wms = [], [], []
    for name, pr in preds.items():
        long_only = name == "TFv6 target speed"
        row, pc, s = J.judge_one(name, p, pr)
        if not row["n_frames"]:
            rows.append({**row, "note": notes.get(name, "")})
            rl.log.info("%s: %s", name, row["verdict"])
            continue
        if long_only:
            for k in ("bypass_flip", "lo", "hi", "gate_a", "shoulder_flip", "selective", "has_bypass", "mirror_borrow"):
                row[k] = np.nan
            row["verdict"] = "longitudinal only"
        row["note"] = notes.get(name, "")
        rows.append(row)
        per.append(pc)
        if not long_only:
            wm = J.world_modes(s, pr)
            if len(wm):
                wms.append(J.mode_agreement(wm).assign(examinee=name))
        rl.log.info("%s: flip %.3f [%.3f, %.3f], ff %.3f, shoulder %.3f (ref %.3f), stop-sub %.3f -> %s", name,
                    row["bypass_flip"], row["lo"], row["hi"], row["null_ff_oos"], row["shoulder_flip"],
                    row["shoulder_ref"], row["stop_sub"], row["verdict"])
    tab = pd.DataFrame(rows)
    for c in ("bypass_flip", "lo", "hi", "null_ff_oos", "gate_a", "shoulder_flip", "shoulder_ref", "selective", "stop_sub",
              "stop_lo", "stop_hi", "neg_later_rate", "neg_agree_expert", "mirror_borrow", "routes", "tau_lat"):
        if c not in tab:
            tab[c] = np.nan
    tab.to_csv(out / "summary.csv", index=False)
    pd.concat(per or [pd.DataFrame()]).to_csv(out / "per_scenario.csv", index=False)
    if wms:
        wm = pd.concat(wms)
        wm.to_csv(out / "world_modes.csv", index=False)
        ag = wm.groupby(["examinee", "world"]).agree.agg(["mean", "size"]).round(3).reset_index()
        ag.to_csv(out / "mode_agreement.csv", index=False)
    cot = proc("nq3_alpamayo_cot.parquet")
    if cot.exists() and "Alpamayo 1.5" in preds:
        c = pd.read_parquet(cot)
        s = J.flips(J.score(p, preds["Alpamayo 1.5"]), *J.taus(J.score(p, preds["Alpamayo 1.5"])))
        b = s[(s.reading == "bypass") & s.main].merge(c[["frame_name", "cot_nudge"]], left_on="frame_name_a",
                                                     right_on="frame_name")
        inc = b[b.cot_nudge.astype(bool)]
        pd.DataFrame([{"x10 frames with CoT nudge": len(inc), "of": len(b),
                       "CoT nudge but no bypass flip": float((~inc.flip).mean()) if len(inc) else np.nan,
                       "bypass flip without CoT nudge": float(b.flip[~b.cot_nudge.astype(bool)].mean())}]
                     ).to_csv(out / "alpamayo_cot.csv", index=False)
    cols = ["examinee", "n_frames", "routes", "tau_lat", "bypass_flip", "lo", "hi", "null_ff_oos", "gate_a",
            "shoulder_flip", "shoulder_ref", "selective", "verdict", "stop_sub", "stop_lo", "stop_hi",
            "neg_later_rate", "neg_agree_expert", "mirror_borrow"]
    (out / "summary.md").write_text(tab[cols].to_markdown(index=False, floatfmt=".3f") + "\n")
    rl.log.info("\n%s", tab[cols].to_markdown(index=False, floatfmt=".3f"))
    return tab


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("heads", "judge"))
    a = ap.parse_args()
    rl = RunLog("nq3_c", f"q1-{a.cmd}")
    heads(rl) if a.cmd == "heads" else judge(rl)
    rl.close()


if __name__ == "__main__":
    main()
