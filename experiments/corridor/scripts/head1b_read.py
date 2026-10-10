#!/usr/bin/env python3
"""HEAD1b steps A and A2: selection, final-head assembly and reads (experiments/corridor; amendment "HEAD1b" at the end of
plans/2026-10-10-head1-prereg.md). EXPLORATORY: a second attempt after the missed registered G3 line; nothing here is a registered read.
CPU, .venv. Definitions (sets H / T, the 4 s-arc heading error, R2, the log-cluster bootstrap) are head1_read.py's, imported.

  sel     step A: candidates' fold-0 validation losses -> the amendment's choice (lowest; within 1 % the cheaper)   -> $H1/final/select.json
  ens     step A: seed-ensemble rule on the fold-0 validation rows (mean of two seeds >= 2 % below seed 0)          -> $H1/final/select.json
  final   step A: out-of-fold profiles of the final head for every navtrain row + navtest (fold 0)                  -> $H1/final/prof/<data>.npy, DONE
  feats   step A2: arc-length heading curves of the pose heads' predicted poses (pt_swap.Curve, as the policy plans) -> $H1/report-b/feats_pose.npz
  read    step A: G1 / G2 / G3 once on navtest with the final head, next to stage 1; H rows; buckets                -> $H1/report-b/a_{tables.md,reads.json}
  a2      step A2: the 2 x 2 (target form x structure) + decision 204's stored QH head                               -> $H1/report-b/a2_{tables.md,reads.json}
  fig     figures from the json files (--src / --out)
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).parent)]
import head1_read as R1  # noqa: E402
from head1_read import CB, GRID, H1, at, corr, fmt, md, r2, rms_a, rms_diff, wrap  # noqa: E402
from head1_train import DELTA, NSH, TRAIN_DIRS, TEST_DIR, is_val, pred_name  # noqa: E402

CR, D = R1.CR, R1.D
FIN, RB = H1 / "final", H1 / "report-b"
NOTE = "EXPLORATORY (HEAD1b: second attempt after the missed registered G3 line, user-authorised 2026-10-10); not a registered read."
CAND = {"a16": dict(steps=16000), "a24": dict(steps=24000), "r16": dict(steps=16000, drop=0.2, wd=0.1), "t16": dict(steps=16000, tokdrop=128),
        "rt24": dict(steps=24000, drop=0.2, wd=0.1, tokdrop=128)}
STAGE1 = dict(G1_rms=8.54, G1_diff=(-2.70, -3.57, -1.74), G2=-6.59, G3_r2=(0.513, 0.420, 0.586), G3_corr=0.53, r2_sh30_all=0.278)
js = lambda o: json.loads(json.dumps(o, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))  # noqa: E731


def run_dir(name):
    """Newest finished run directory of train-<name>."""
    ds = sorted(p for p in glob.glob(str(H1 / f"train-{name}" / "*")) if (Path(p) / "DONE").exists())
    assert ds, f"no finished run of train-{name}"
    return Path(ds[-1])


def run_info(name):
    d = run_dir(name)
    cur = json.loads((d / "curve.json").read_text())
    b = min(cur, key=lambda c: c["val"])
    ev = [json.loads(l) for l in (d / "events.jsonl").read_text().splitlines()]
    return dict(name=name, dir=str(d), best_val=b["val"], best_step=b["step"], last_val=cur[-1]["val"], last_train=cur[-1]["train"], curve=cur,
                wall_s=ev[-1]["t"] - ev[0]["t"])


def cost(c):
    return c["steps"] * c.get("tokdrop", 256)


def val_rows():
    """Fold-0 validation rows in token-cache order: names, L label (n, 22), sampling weight."""
    from jevdrive.data import splits
    Lb = np.load(H1 / "labels/navtrain.npz")
    logs = Lb["log"]
    m = splits.load(f"navsim/op-parity-cf5f{R1.FOLD}-train").mask(logs) & np.array([is_val(l) for l in logs.tolist()])
    return Lb["names"][m], Lb["L"][m].astype(np.float64), 1 + np.minimum(np.abs(np.degrees(np.nan_to_num(Lb["dyaw4"][m]))), 90.0) / 30.0


def huber_val(p, y, w):
    """The trainer's weighted Huber validation loss (channel L) in float64."""
    ok = np.isfinite(y)
    e = np.abs(p - np.nan_to_num(y))
    l = np.where(e <= DELTA, 0.5 * e * e, DELTA * (e - 0.5 * DELTA)) * ok * w[:, None]
    return float(l.sum() / (ok * w[:, None]).sum())


def cmd_sel(a):
    ref = run_info(pred_name("L", 0, 1.0, 0))
    C = {k: run_info(pred_name("L", 0, 1.0, 0, tag=k)) | dict(cfg=v, cost=cost(v)) for k, v in CAND.items()}
    lo = min(c["best_val"] for c in C.values())
    near = [k for k, c in C.items() if c["best_val"] < 1.01 * lo]
    pick = min(near, key=lambda k: (C[k]["cost"], C[k]["best_val"]))
    FIN.mkdir(parents=True, exist_ok=True)
    out = dict(note=NOTE, rule="lowest fold-0 validation loss at its best step; candidates within 1 % of the lowest -> the cheapest (steps x vision tokens read per step)",
               reference_stage1=ref, candidates=C, lowest=min(C, key=lambda k: C[k]["best_val"]), within_1pct=near, chosen=pick)
    (FIN / "select.json").write_text(json.dumps(js(out), indent=1))
    print(md([{"candidate": k, **{q: c["cfg"].get(q, d_) for q, d_ in (("steps", 8000), ("drop", 0.1), ("wd", 0.05), ("tokdrop", 256))}, "best val": f"{c['best_val']:.6f}", "at step": c["best_step"],
               "last train": f"{c['last_train']:.5f}", "wall min": f"{c['wall_s'] / 60:.1f}"} for k, c in C.items()], ["candidate", "steps", "drop", "wd", "tokdrop", "best val", "at step", "last train", "wall min"]))
    print(f"stage-1 reference {ref['best_val']:.6f}; lowest {out['lowest']}; within 1 %: {near}; chosen: {pick}")


def cmd_ens(a):
    S = json.loads((FIN / "select.json").read_text())
    names, y, w = val_rows()
    P = []
    for s in (0, 1):
        z = np.load(run_dir(pred_name("L", 0, 1.0, s, tag=S["chosen"])) / "pred.npz")
        assert (z["names_val"] == names).all()
        P.append(z["val"][..., 0].astype(np.float64))
    v0, v1, vm = huber_val(P[0], y, w), huber_val(P[1], y, w), huber_val((P[0] + P[1]) / 2, y, w)
    S["ensemble"] = dict(rule="mean of seeds 0 and 1 on the fold-0 validation rows at least 2 % below seed 0 (same float64 recomputation for both)", val_s0=v0, val_s1=v1, val_mean=vm,
                         rel=vm / v0 - 1, adopted=bool(vm <= 0.98 * v0), n_rows=len(names), seed1=run_info(pred_name("L", 0, 1.0, 1, tag=S["chosen"])))
    (FIN / "select.json").write_text(json.dumps(js(S), indent=1))
    print(json.dumps({k: v for k, v in S["ensemble"].items() if k != "seed1"}, indent=1))


def cmd_final(a):
    from jevdrive.data import splits
    S = json.loads((FIN / "select.json").read_text())
    seeds = (0, 1) if S["ensemble"]["adopted"] else (0,)
    tabs = [np.load(CR / d / "tab.npz") for d in TRAIN_DIRS]
    names, logs = (np.concatenate([t[k] for t in tabs]) for k in ("names", "log"))
    pos = {k: i for i, k in enumerate(names.tolist())}
    assert len(pos) == len(names)
    P, cnt, runs, test = np.zeros((len(names), len(GRID))), np.zeros(len(names), int), {}, []
    for j in range(5):
        tr, dv = splits.load(f"navsim/op-parity-cf5f{j}-train").mask(logs), splits.load(f"navsim/op-parity-cf5f{j}-dev").mask(logs)
        assert not set(logs[tr].tolist()) & set(logs[dv].tolist())
        for s in seeds:
            nm = pred_name("L", j, 1.0, s, tag=S["chosen"])
            z = np.load(run_dir(nm) / "pred.npz")
            assert int(z["fold"]) == j and (z["names_dev"] == names[dv]).all(), f"{nm}: dev rows are not fold {j}'s held-out rows"
            P[dv] += z["dev"][..., 0] / len(seeds)
            if s == seeds[0]:
                cnt[dv] += 1
            if j == 0:
                test.append(z["test"][..., 0].astype(np.float64))
                tn = z["names_test"]
            runs[f"f{j}-s{s}"] = {k: v for k, v in run_info(nm).items() if k != "curve"}
    assert (cnt == 1).all(), f"out-of-fold coverage: {np.bincount(cnt)}"
    tt = np.load(CR / TEST_DIR / "tab.npz")
    assert (tn == tt["names"]).all() and not set(tt["log"].tolist()) & set(logs.tolist())
    Pt = np.mean(test, 0)
    assert np.isfinite(P).all() and np.isfinite(Pt).all() and np.abs(P[:, 0]).max() < 0.5
    (FIN / "prof").mkdir(parents=True, exist_ok=True)
    off = 0
    for d, t in zip(TRAIN_DIRS, tabs):
        n = len(t["names"])
        np.save(FIN / "prof" / f"{d}.npy", P[off:off + n].astype(np.float32))
        off += n
    np.save(FIN / "prof" / f"{TEST_DIR}.npy", Pt.astype(np.float32))
    for d, n in [(d, len(t["names"])) for d, t in zip(TRAIN_DIRS, tabs)] + [(TEST_DIR, len(tt["names"]))]:
        x = np.load(FIN / "prof" / f"{d}.npy")
        assert x.shape == (n, len(GRID)) and x.dtype == np.float32 and np.isfinite(x).all(), d
    done = dict(note=NOTE, chosen=S["chosen"], config=CAND[S["chosen"]], ensemble=bool(S["ensemble"]["adopted"]), seeds=list(seeds), runs=runs,
                val_losses={k: c["best_val"] for k, c in S["candidates"].items()} | {"stage1 L-f0-s0 (reference)": S["reference_stage1"]["best_val"]},
                ensemble_read={k: v for k, v in S["ensemble"].items() if k != "seed1"},
                files=dict(prof="prof/<data>.npy: float32 (N, 22) heading in rad at the label grid, rows in token-cache tab order", grid=GRID.tolist(),
                           navtrain="every row from the fold model(s) that never saw its log (decision 191's folds; out-of-fold)", navtest="fold-0 model(s)"))
    (FIN / ".DONE.tmp").write_text(json.dumps(js(done), indent=1))
    (FIN / ".DONE.tmp").rename(FIN / "DONE")
    print(f"final head {S['chosen']}, seeds {seeds}: {len(names)} navtrain rows out-of-fold, {len(Pt)} navtest rows -> {FIN}")


# ---------------------------------------------------------------- A2: pose heads
POSE = {f"P-{h} s{s}": pred_name("P", 0, 1.0, s, "thin" if h == "thin" else "full") for h in ("full", "thin") for s in (0, 1)}
QH = D / "runs/op_parity/path_req/head/lb_navtest.npy"


def _pose_chunk(c):
    """rows -> [4 s arc length, 4 s yaw, heading on GRID (ext-line beyond the end), heading on GRID (ext-arc beyond the end)]."""
    import pt_swap as PS
    P = R1._P["P"]
    out = np.full((len(c), 2 + 2 * len(GRID)), np.nan)
    for n, i in enumerate(c):
        cv = PS.Curve(P[i])
        sp = float(cv.sv[-1])
        out[n, 0], out[n, 1] = sp, np.unwrap(np.r_[0, P[i][:, 2]])[-1]
        arc = cv.at(GRID)[:, 2] if cv.L > 0 else np.full(len(GRID), cv.yk[-1])
        g = GRID <= min(sp, cv.L) + 1e-9
        out[n, 2:2 + len(GRID)] = np.where(g, arc, cv.yk[-1])
        out[n, 2 + len(GRID):] = arc
    return c, out


def cmd_feats(a):
    from jevdrive import par
    from jevdrive.run import Run
    from scipy.interpolate import CubicSpline  # noqa: F401  (before the fork)
    with Run("corridor", "head1/feats-b", config=vars(a)) as run:
        src = {}
        for k, nm in POSE.items():
            z = np.load(run_dir(nm) / "pred.npz")
            src[f"H|{k}"], src[f"T|{k}"] = z["dev"], z["test"]
        src["T|QH"] = np.load(QH)
        out = {}
        for k, P in src.items():
            R1._P["P"] = P.astype(np.float64)
            idx = np.arange(len(P))
            res = par.pmap(_pose_chunk, [idx[q::128] for q in range(128)], run=run, desc=k)
            res.raise_if_failed()
            F = np.full((len(P), 2 + 2 * len(GRID)), np.nan)
            for c, o in res.values:
                F[c] = o
            assert np.isfinite(F).all()
            out[k] = F.astype(np.float32)
            run.info("%s: %d plans, median 4 s arc %.2f m", k, len(P), np.median(F[:, 0]))
        RB.mkdir(parents=True, exist_ok=True)
        np.savez(RB / "feats_pose.npz", **out)


# ---------------------------------------------------------------- reads
def draws(cb, stat, a, b, mask):
    """CB.__call__ returning the bootstrap draws too (paired differences of two statistics on the same rows and draws)."""
    ok = np.isfinite(a) & np.isfinite(b) & mask
    a, b, c = a[ok].astype(np.float64), b[ok].astype(np.float64), cb.codes[ok]
    M = np.stack([np.ones_like(a), a, b, a * a, b * b, a * b], 1)
    S = np.stack([np.bincount(c, M[:, k], cb.G) for k in range(6)], 1)
    return float(stat(S.sum(0))), stat((cb.C @ S).T)


def paired(cb, stat, a1, b1, a2, b2, mask):
    """stat(a1, b1) - stat(a2, b2) with the paired log-cluster CI (all four columns finite)."""
    mask = mask & np.isfinite(a1) & np.isfinite(b1) & np.isfinite(a2) & np.isfinite(b2)
    (v1, d1), (v2, d2) = draws(cb, stat, a1, b1, mask), draws(cb, stat, a2, b2, mask)
    lo, hi = np.nanquantile(d1 - d2, [0.025, 0.975])
    return dict(v=v1 - v2, lo=float(lo), hi=float(hi), n=int(mask.sum()))


class Sets:
    """Sets H and T with the stored policy plans' features (head1_read's), pooled over policy seeds."""
    GROUPS = {"H": {"CF5f0": R1.POL["H"]}, "T": {"SH30-F": R1.POL["T"][:2], "GH0-F (pilot)": R1.POL["T"][2:]}}

    def __init__(s):
        FE = np.load(R1.RP / "feats.npz")
        s.d = {}
        for S in ("H", "T"):
            d = R1.load_set(S)
            d["dy"] = np.abs(np.degrees(d["fut"][:, 7, 2]))
            d["bk"] = {k: d["dy"] > v if v else np.isfinite(d["dy"]) for k, v in R1.BK.items()}
            d["h4"] = d["fut"][:, 7, 2]
            d["pol"] = {m: dict(sp=FE[f"{S}|{m}"][:, 0].astype(np.float64), e=wrap(FE[f"{S}|{m}"][:, 1] - d["h4"])) for m in R1.POL[S]}
            d["cb"] = {}
            s.d[S] = d
        assert not set(s.d["H"]["log"]) & set(s.d["T"]["log"])

    def pool(s, S, mods, fn):
        d = s.d[S]
        if len(mods) not in d["cb"]:
            d["cb"][len(mods)] = CB(np.tile(d["log"], len(mods)))
        return np.concatenate([fn(m) for m in mods]), {k: np.tile(v, len(mods)) for k, v in d["bk"].items()}, d["cb"][len(mods)]

    def e_pol(s, S, mods):
        return s.pool(S, mods, lambda m: np.degrees(s.d[S]["pol"][m]["e"]))

    def e_head(s, S, mods, prof):
        """Head profile (n, 22) read at each policy seed's own 4 s arc length, minus the logged 4 s heading (deg)."""
        d = s.d[S]
        return s.pool(S, mods, lambda m: np.degrees(wrap(at(prof, d["pol"][m]["sp"]) - d["h4"])))[0]


def stage1_pred(name, S, names):
    z = np.load(H1 / "pred" / f"{name}.npz")
    assert (z["names_dev" if S == "H" else "names_test"].astype(str) == names).all()
    return z["dev" if S == "H" else "test"].astype(np.float64)


def new_pred(name, S, names):
    z = np.load(run_dir(name) / "pred.npz")
    assert (z["names_dev" if S == "H" else "names_test"].astype(str) == names).all()
    return z["dev" if S == "H" else "test"].astype(np.float64)


def head_row(cb, ep, eh, mk, eb=None):
    h = dict(rms=cb(rms_a, eh, mask=mk), diff_policy=cb(rms_diff, eh, ep, mask=mk), corr=cb(corr, ep, eh, mask=mk), r2=cb(r2, ep, eh, mask=mk))
    if eb is not None:
        h["diff_blind"] = cb(rms_diff, eh, eb, mask=mk)
    return h


def cmd_read(a):
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("corridor", "head1/read-b", config=vars(a)) as run:
        for s in ("navsim/navtest", f"navsim/op-parity-cf5f{R1.FOLD}-dev", f"navsim/op-parity-cf5f{R1.FOLD}-train"):
            run.use_split(splits.load(s))
        done, sel = json.loads((FIN / "DONE").read_text()), json.loads((FIN / "select.json").read_text())
        Z = Sets()
        tabs = [np.load(CR / d_ / "tab.npz") for d_ in TRAIN_DIRS]
        nav_names = np.concatenate([t["names"] for t in tabs]).astype(str)
        oof = np.concatenate([np.load(FIN / "prof" / f"{d_}.npy") for d_ in TRAIN_DIRS]).astype(np.float64)
        pos = {k: i for i, k in enumerate(nav_names.tolist())}
        fin = {"H": oof[[pos[k] for k in Z.d["H"]["names"]]], "T": np.load(FIN / "prof" / f"{TEST_DIR}.npy").astype(np.float64)}
        s1, bl = pred_name("L", 0, 1.0, 0), pred_name("blind", 0, 1.0, 0)
        heads = {S: {"final": fin[S], "stage-1 L s0": stage1_pred(s1, S, Z.d[S]["names"])[..., 0], "blind": stage1_pred(bl, S, Z.d[S]["names"])[..., 0]} for S in ("H", "T")}
        if done["ensemble"]:                                                      # the two members of the adopted ensemble, descriptive
            for s in (0, 1):
                heads["T"][f"final member s{s}"] = new_pred(pred_name("L", 0, 1.0, s, tag=done["chosen"]), "T", Z.d["T"]["names"])[..., 0]
        for k in CAND:                                                           # every candidate on held-out navtrain only (never navtest)
            heads["H"][f"candidate {k}"] = new_pred(pred_name("L", 0, 1.0, 0, tag=k), "H", Z.d["H"]["names"])[..., 0]
        R, A = {}, []
        for S, gs in Z.GROUPS.items():
            d = Z.d[S]
            for g, mods in gs.items():
                ep, bk, cb = Z.e_pol(S, mods)
                eh = {k: Z.e_head(S, mods, v) for k, v in heads[S].items()}
                eh4 = {k: np.degrees(wrap(at(v, d["lab"]["s4"].astype(np.float64)) - d["h4"])) for k, v in heads[S].items()}
                if 1 not in d["cb"]:
                    d["cb"][1] = CB(d["log"])
                for b, mk in bk.items():
                    key = f"{S}|{g}|{b}"
                    R[key] = dict(policy=cb(rms_a, ep, mask=mk), heads={})
                    A.append({"set": S, "policy": g, "bucket": b, "predictor": "policy's own plan", "n": R[key]["policy"]["n"], "RMS deg": fmt(R[key]["policy"])})
                    for k in eh:
                        h = head_row(cb, ep, eh[k], mk, None if k == "blind" else eh["blind"])
                        h["rms_at_log_arc"] = d["cb"][1](rms_a, eh4[k], mask=d["bk"][b])
                        if k not in ("stage-1 L s0", "blind"):
                            h["rms_minus_stage1"] = cb(rms_diff, eh[k], eh["stage-1 L s0"], mask=mk)
                            h["r2_minus_stage1"] = paired(cb, r2, ep, eh[k], ep, eh["stage-1 L s0"], mk)
                        R[key]["heads"][k] = h
                        A.append({"set": S, "policy": g, "bucket": b, "predictor": f"head {k}", "n": h["rms"]["n"], "RMS deg": fmt(h["rms"]), "minus policy": fmt(h["diff_policy"], "{:+.2f}"),
                                  "minus blind": fmt(h.get("diff_blind"), "{:+.2f}"), "corr(e_head, e_policy)": fmt(h["corr"]), "R2": fmt(h["r2"], "{:.3f}"),
                                  "0.93 x R2": f"{R1.QS_GAIN * h['r2']['v']:+.2f}", "RMS minus stage 1": fmt(h.get("rms_minus_stage1"), "{:+.2f}"),
                                  "R2 minus stage 1": fmt(h.get("r2_minus_stage1"), "{:+.3f}"), "RMS at the logged 4 s arc": fmt(h["rms_at_log_arc"])})
        f1, f3, o1, o3 = (R[k]["heads"][h] for k, h in (("T|SH30-F|>45", "final"), ("T|GH0-F (pilot)|all", "final"), ("T|SH30-F|>45", "stage-1 L s0"), ("T|GH0-F (pilot)|all", "stage-1 L s0")))
        fa, oa = R["T|SH30-F|all"]["heads"]["final"], R["T|SH30-F|all"]["heads"]["stage-1 L s0"]
        gate = dict(note=NOTE + " Description of the converged head, not a gate.", final=dict(chosen=done["chosen"], ensemble=done["ensemble"]),
                    G1=dict(final=dict(rms=f1["rms"], diff=f1["diff_policy"]), stage1=dict(rms=o1["rms"], diff=o1["diff_policy"]), policy=R["T|SH30-F|>45"]["policy"], final_minus_stage1=f1["rms_minus_stage1"]),
                    G2=dict(final=f1["diff_blind"], stage1=o1["diff_blind"]),
                    G3=dict(final=dict(r2=f3["r2"], corr=f3["corr"], gain=R1.QS_GAIN * f3["r2"]["v"]), stage1=dict(r2=o3["r2"], corr=o3["corr"], gain=R1.QS_GAIN * o3["r2"]["v"]),
                            final_minus_stage1=f3["r2_minus_stage1"], line_r2=R1.R2_PASS, final_at_or_above_line=bool(f3["r2"]["v"] >= R1.R2_PASS)),
                    r2_vs_SH30_all=dict(final=fa["r2"], stage1=oa["r2"], final_minus_stage1=fa["r2_minus_stage1"], gain_final=R1.QS_GAIN * fa["r2"]["v"], gain_stage1=R1.QS_GAIN * oa["r2"]["v"]),
                    stage1_as_reported=STAGE1)
        L = [f"# HEAD1b step A: the arm-L head trained to convergence\n\n**{NOTE}**\n"]
        P = L.append
        P("## Candidates (fold 0, seed 0; selection on the fold-0 validation logs only, navtest never read for it)\n")
        C = sel["candidates"]
        rows = [{"candidate": "stage-1 L-f0-s0 (reference, not a candidate)", "steps": 8000, "dropout": 0.1, "wd": 0.05, "vision tokens per step": 256, "best val": f"{sel['reference_stage1']['best_val']:.6f}",
                 "at step": sel["reference_stage1"]["best_step"], "last train loss": f"{sel['reference_stage1']['last_train']:.5f}", "job wall min": f"{sel['reference_stage1']['wall_s'] / 60:.1f}"}]
        for k, c in C.items():
            hH = R["H|CF5f0|>45"]["heads"][f"candidate {k}"]
            rows.append({"candidate": k + (" (chosen)" if k == sel["chosen"] else ""), "steps": c["cfg"]["steps"], "dropout": c["cfg"].get("drop", 0.1), "wd": c["cfg"].get("wd", 0.05),
                         "vision tokens per step": c["cfg"].get("tokdrop", 256), "best val": f"{c['best_val']:.6f}", "vs lowest": f"{100 * (c['best_val'] / C[sel['lowest']]['best_val'] - 1):+.2f} %",
                         "at step": c["best_step"], "last train loss": f"{c['last_train']:.5f}", "job wall min": f"{c['wall_s'] / 60:.1f}", "H > 45 RMS deg": fmt(hH["rms"]),
                         "H all R2 vs fold policy": fmt(R["H|CF5f0|all"]["heads"][f"candidate {k}"]["r2"], "{:.3f}")})
        rows[0]["H > 45 RMS deg"] = fmt(R["H|CF5f0|>45"]["heads"]["stage-1 L s0"]["rms"])
        rows[0]["H all R2 vs fold policy"] = fmt(R["H|CF5f0|all"]["heads"]["stage-1 L s0"]["r2"], "{:.3f}")
        P(md(rows, ["candidate", "steps", "dropout", "wd", "vision tokens per step", "best val", "vs lowest", "at step", "last train loss", "job wall min", "H > 45 RMS deg", "H all R2 vs fold policy"]))
        P(f"Rule: lowest validation loss; within 1 % of the lowest the cheapest (steps x vision tokens per step). Lowest `{sel['lowest']}`, within 1 %: {sel['within_1pct']}, chosen **`{sel['chosen']}`**. "
          "The H columns (held-out navtrain, fold policy CF5f0-F-s0) are descriptive and were not used for the choice.\n")
        E = sel["ensemble"]
        P(f"Seed ensemble (fold-0 validation rows, {E['n_rows']} tokens, float64 recomputation): seed 0 {E['val_s0']:.6f}, seed 1 {E['val_s1']:.6f}, mean of the two predictions {E['val_mean']:.6f} "
          f"({100 * E['rel']:+.2f} % against seed 0; rule: at least -2 %) -> **{'adopted: final head = mean of seeds 0 and 1 in every fold' if E['adopted'] else 'not adopted: final head = seed 0'}**.\n")
        P("Fold models of the final configuration (out-of-fold predictions for every navtrain row):\n")
        P(md([{"run": k, "name": v["name"], "best val": f"{v['best_val']:.6f}", "at step": v["best_step"], "job wall min": f"{v['wall_s'] / 60:.1f}"} for k, v in done["runs"].items()], ["run", "name", "best val", "at step", "job wall min"]))
        P("## G1 / G2 / G3 read once on navtest with the final head, next to stage 1 (a description, not a gate)\n")
        P(md([{"line": "G1: RMS(head) - RMS(SH30-F plan), > 45 deg", "final head": f"{fmt(f1['rms'])} vs {fmt(R['T|SH30-F|>45']['policy'])}; {fmt(f1['diff_policy'], '{:+.2f}')}",
               "stage-1 head (recomputed here)": f"{fmt(o1['rms'])}; {fmt(o1['diff_policy'], '{:+.2f}')}", "final - stage 1 (paired)": fmt(f1["rms_minus_stage1"], "{:+.2f}") + " deg RMS"},
              {"line": "G2: RMS(head) - RMS(blind), > 45 deg (stage-1 blind head)", "final head": fmt(f1["diff_blind"], "{:+.2f}"), "stage-1 head (recomputed here)": fmt(o1["diff_blind"], "{:+.2f}")},
              {"line": f"G3: R2 against the pilot policy GH0-F, all tokens (stage-1 line {R1.R2_PASS:.3f})", "final head": f"R2 {fmt(f3['r2'], '{:.3f}')}, corr {fmt(f3['corr'])}, 0.93 x R2 = {R1.QS_GAIN * f3['r2']['v']:+.2f}",
               "stage-1 head (recomputed here)": f"R2 {fmt(o3['r2'], '{:.3f}')}, corr {fmt(o3['corr'])}, {R1.QS_GAIN * o3['r2']['v']:+.2f}", "final - stage 1 (paired)": fmt(f3["r2_minus_stage1"], "{:+.3f}") + " R2"},
              {"line": "R2 against SH30-F, all tokens", "final head": f"R2 {fmt(fa['r2'], '{:.3f}')}, 0.93 x R2 = {R1.QS_GAIN * fa['r2']['v']:+.2f}",
               "stage-1 head (recomputed here)": f"R2 {fmt(oa['r2'], '{:.3f}')}, {R1.QS_GAIN * oa['r2']['v']:+.2f}", "final - stage 1 (paired)": fmt(fa["r2_minus_stage1"], "{:+.3f}") + " R2"}],
             ["line", "final head", "stage-1 head (recomputed here)", "final - stage 1 (paired)"]))
        P("## Buckets: heading error at the plan's own 4 s arc length (deg RMS), every head\n")
        P(md(A, ["set", "policy", "bucket", "predictor", "n", "RMS deg", "minus policy", "minus blind", "corr(e_head, e_policy)", "R2", "0.93 x R2", "RMS minus stage 1", "R2 minus stage 1", "RMS at the logged 4 s arc"]))
        P("Sets: H = fold 0's held-out navtrain logs (fold policy CF5f0-F-s0; the final head's rows are its out-of-fold predictions, i.e. the fold-0 model(s)); T = navtest (fold-0 model(s)). "
          "CIs: log-cluster bootstrap, B 10 000, 95 %. Differences against stage 1 are paired on the same token-seeds and draws.\n")
        RB.mkdir(parents=True, exist_ok=True)
        (RB / "a_tables.md").write_text("\n".join(L))
        (RB / "a_reads.json").write_text(json.dumps(js(dict(note=NOTE, gate=gate, select=sel, final=done, reads=R)), indent=1))
        run.summary.update(gate=js(gate))
        run.info("G1 %s | G2 %s | G3 R2 %s | R2 vs SH30-F all %s", fmt(f1["diff_policy"], "{:+.2f}"), fmt(f1["diff_blind"], "{:+.2f}"), fmt(f3["r2"], "{:.3f}"), fmt(fa["r2"], "{:.3f}"))


def cmd_a2(a):
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("corridor", "head1/a2-b", config=vars(a)) as run:
        for s in ("navsim/navtest", f"navsim/op-parity-cf5f{R1.FOLD}-dev", f"navsim/op-parity-cf5f{R1.FOLD}-train"):
            run.use_split(splits.load(s))
        Z, FP, NGR = Sets(), np.load(RB / "feats_pose.npz"), len(GRID)
        cells = {"L-full": ("L", "full"), "L-thin": ("L", "thin"), "P-full": ("P", "full"), "P-thin": ("P", "thin")}
        R, T, info = {}, [], {}
        for S, gs in Z.GROUPS.items():
            d = Z.d[S]
            heads = {}                                                           # name -> dict(prof, arc (ext-arc profile) | None, yaw4 | None, sp | None)
            for c, (arm, hd) in cells.items():
                for s in (0, 1):
                    nm = pred_name(arm, 0, 1.0, s, hd)
                    if arm == "L":
                        heads[f"{c} s{s}"] = dict(prof=(stage1_pred if hd == "full" else new_pred)(nm, S, d["names"])[..., 0])
                    else:
                        F = FP[f"{S}|{c} s{s}"].astype(np.float64)
                        assert (np.load(run_dir(nm) / "pred.npz")["names_dev" if S == "H" else "names_test"].astype(str) == d["names"]).all()
                        heads[f"{c} s{s}"] = dict(prof=F[:, 2:2 + NGR], arc=F[:, 2 + NGR:], yaw4=F[:, 1], sp=F[:, 0])
                    if S == "H":
                        info[f"{c} s{s}"] = {k: v for k, v in run_info(nm).items() if k != "curve"}
            if S == "T":
                F = FP["T|QH"].astype(np.float64)
                heads["QH (decision 204, stored)"] = dict(prof=F[:, 2:2 + NGR], arc=F[:, 2 + NGR:], yaw4=F[:, 1], sp=F[:, 0])
            for g, mods in gs.items():
                ep, bk, cb = Z.e_pol(S, mods)
                for k, h in heads.items():
                    eh = Z.e_head(S, mods, h["prof"])
                    ea = Z.e_head(S, mods, h["arc"]) if "arc" in h else None
                    et = np.tile(np.degrees(wrap(h["yaw4"] - d["h4"])), len(mods)) if "yaw4" in h else None
                    ext = Z.pool(S, mods, lambda m: (d["pol"][m]["sp"] > h["sp"] + 1e-6).astype(float))[0] if "sp" in h else None
                    for b, mk in bk.items():
                        r_ = head_row(cb, ep, eh, mk)
                        r_["policy"] = cb(rms_a, ep, mask=mk)
                        if ea is not None:
                            r_["ext_arc"] = dict(rms=cb(rms_a, ea, mask=mk), r2=cb(r2, ep, ea, mask=mk))
                            r_["timed_yaw"] = dict(rms=cb(rms_a, et, mask=mk), diff_policy=cb(rms_diff, et, ep, mask=mk), r2=cb(r2, ep, et, mask=mk))
                            r_["share_extended"] = float(ext[mk & np.isfinite(ep)].mean())
                        R[f"{S}|{g}|{b}|{k}"] = r_
                        T.append({"set": S, "policy": g, "bucket": b, "head": k, "RMS deg at the plan's 4 s arc": fmt(r_["rms"]), "minus policy": fmt(r_["diff_policy"], "{:+.2f}"), "corr": fmt(r_["corr"], ci=False),
                                  "R2": fmt(r_["r2"], "{:.3f}"), "ext-arc: RMS / R2": f"{r_['ext_arc']['rms']['v']:.2f} / {r_['ext_arc']['r2']['v']:.3f}" if ea is not None else "",
                                  "share of token-seeds extended": f"{100 * r_['share_extended']:.0f} %" if ea is not None else "",
                                  "timed 4 s yaw: RMS / minus policy / R2": f"{fmt(r_['timed_yaw']['rms'])} / {fmt(r_['timed_yaw']['diff_policy'], '{:+.2f}')} / {r_['timed_yaw']['r2']['v']:.3f}" if ea is not None else ""})
        # ---- the 2 x 2 and the factor effects (seed mean; the two seeds' values side by side)
        reads = {"T: > 45 RMS at SH30-F's arc (deg)": ("T|SH30-F|>45", "rms"), "T: > 45 RMS minus SH30-F plan (G1 form)": ("T|SH30-F|>45", "diff_policy"), "T: all RMS at SH30-F's arc (deg)": ("T|SH30-F|all", "rms"),
                 "T: R2 vs GH0-F, all (G3 form)": ("T|GH0-F (pilot)|all", "r2"), "T: R2 vs SH30-F, all": ("T|SH30-F|all", "r2"),
                 "H: > 45 RMS at the fold policy's arc (deg)": ("H|CF5f0|>45", "rms"), "H: > 45 RMS minus fold policy": ("H|CF5f0|>45", "diff_policy"), "H: all RMS (deg)": ("H|CF5f0|all", "rms"),
                 "H: R2 vs fold policy, all": ("H|CF5f0|all", "r2")}
        val = lambda rd, c, s: R[f"{reads[rd][0]}|{c} s{s}"][reads[rd][1]]["v"]  # noqa: E731
        G, Fx, eff = [], [], {}
        for rd in reads:
            f = "{:.3f}" if "R2" in rd else "{:+.2f}" if "minus" in rd else "{:.2f}"
            cell = lambda c: f"{f.format((val(rd, c, 0) + val(rd, c, 1)) / 2)} ({f.format(val(rd, c, 0))}, {f.format(val(rd, c, 1))})"  # noqa: E731
            row = {"read": rd, **{c: cell(c) for c in cells}}
            if rd.startswith("T"):
                row["QH stored (old labels, 1 model)"] = f.format(R[f"{reads[rd][0]}|QH (decision 204, stored)"][reads[rd][1]]["v"])
            G.append(row)
            ef = {}
            for nm, pairs in (("target form: L - P, full structure", ("L-full", "P-full")), ("target form: L - P, thin structure", ("L-thin", "P-thin")),
                              ("structure: full - thin, target L", ("L-full", "L-thin")), ("structure: full - thin, target P", ("P-full", "P-thin"))):
                dd = [val(rd, pairs[0], s) - val(rd, pairs[1], s) for s in (0, 1)]
                ef[nm] = dict(mean=float(np.mean(dd)), seeds=dd)
            if rd.startswith("T"):
                q = R[f"{reads[rd][0]}|QH (decision 204, stored)"][reads[rd][1]]["v"]
                dd = [val(rd, "P-thin", s) - q for s in (0, 1)]
                ef["label quantity and folds: P-thin (HEAD1 folds, 834 logs) - QH stored"] = dict(mean=float(np.mean(dd)), seeds=dd)
            eff[rd] = ef
            g_ = "{:+.3f}" if "R2" in rd else "{:+.2f}"
            Fx.append({"read": rd, **{k: f"{g_.format(v['mean'])} ({g_.format(v['seeds'][0])}, {g_.format(v['seeds'][1])})" for k, v in ef.items()}})
        L = [f"# HEAD1b step A2: target form x structure (+ label quantity and folds), heads only\n\n**{NOTE}**\n"]
        P = L.append
        P("Heads: fold 0, seeds 0 and 1, stage-1 schedule (8 000 steps x 256, lr 3e-4, wd 0.05, same sampling), same frozen tokens + ego. L = 22-point heading-versus-arc-length curve of the logged path "
          "(HEAD1's target); P = 8 timed poses with decision 204's loss (QH's target). full = HEAD1's scene memory + queries (3.4 M); thin = QH's head (slots 7 and 0, 2 x 1024 MLP). "
          "L-full is the stage-1 head. QH stored = decision 204's head as stored (thin, P, its own training rows and schedule; navtest only, its fit rows overlap H).\n")
        P("Pose heads are read as the policy plans are: predicted poses -> pt_swap.Curve -> heading at the policy plan's own 4 s arc length; where the plan's arc exceeds the head's own 4 s arc the heading "
          "is held (ext-line; ext-arc as sensitivity in the full table).\n")
        P("## The 2 x 2 (seed mean (seed 0, seed 1))\n")
        P(md(G, ["read"] + list(cells) + ["QH stored (old labels, 1 model)"]))
        P("## Factor effects (difference along one factor; seed mean (seed 0, seed 1)); RMS: negative = better, R2: positive = better\n")
        P(md(Fx, list(Fx[0])))
        P("## Training\n")
        P(md([{"head": k, "best val (own loss)": f"{v['best_val']:.5f}", "at step": v["best_step"], "last train": f"{v['last_train']:.5f}", "job wall min": f"{v['wall_s'] / 60:.1f}"} for k, v in info.items()],
             ["head", "best val (own loss)", "at step", "last train", "job wall min"]))
        P("## Full table\n")
        P(md(T, ["set", "policy", "bucket", "head", "RMS deg at the plan's 4 s arc", "minus policy", "corr", "R2", "ext-arc: RMS / R2", "share of token-seeds extended",
                   "timed 4 s yaw: RMS / minus policy / R2"]))
        P("CIs: log-cluster bootstrap, B 10 000, 95 %; n as in the step-A tables. No lines are set on these reads.\n")
        (RB / "a2_tables.md").write_text("\n".join(L))
        (RB / "a2_reads.json").write_text(json.dumps(js(dict(note=NOTE, reads=R, grid2x2=G, effects=eff, training=info)), indent=1))
        run.info("\n".join(L[4:12]))


def cmd_fig(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path[:0] = [str(R1.REPO / "research")]
    import plot_style as PSY
    PSY.apply()
    src, out, col = Path(a.src), Path(a.out), PSY.PALETTE
    A, B = json.loads((src / "a_reads.json").read_text()), json.loads((src / "a2_reads.json").read_text())
    # 1: validation curves of the candidates
    fig, ax = plt.subplots(figsize=(PSY.SINGLE_COLUMN_IN, 2.6))
    sel = A["select"]
    c0 = sel["reference_stage1"]["curve"]
    ax.plot([c["step"] for c in c0], [1e3 * c["val"] for c in c0], color=PSY.BASELINE, lw=1.0, label="stage 1 (8 000 steps)")
    for (k, c), cl in zip(sel["candidates"].items(), (col["blue"], col["vermillion"], col["green"], col["orange"], col["purple"])):
        ax.plot([q["step"] for q in c["curve"]], [1e3 * q["val"] for q in c["curve"]], color=cl, lw=1.4 if k == sel["chosen"] else 0.8, label=k + (" (chosen)" if k == sel["chosen"] else ""))
    ax.set_ylim(None, 1e3 * 1.6 * sel["reference_stage1"]["best_val"]); ax.set_xlabel("training step"); ax.set_ylabel(r"fold-0 validation loss ($\times 10^{-3}$)"); ax.legend(fontsize=6.5)
    fig.tight_layout(pad=0.4); PSY.save(fig, out / "h1b_val_curves")
    # 2: the 2 x 2
    R = B["reads"]
    fig, axs = plt.subplots(1, 2, figsize=(PSY.DOUBLE_COLUMN_IN, 2.5))
    cells = [("L-full", col["blue"]), ("L-thin", col["green"]), ("P-full", col["orange"]), ("P-thin", col["vermillion"])]
    for ax, (key, q, yl) in zip(axs, (("T|SH30-F|>45", "rms", "navtest > 45 deg: 4 s heading error RMS at SH30-F's arc (deg)"), ("T|GH0-F (pilot)|all", "r2", r"navtest, all: $R^2$ against the pilot policy"))):
        x = 0
        for c, cl in cells:
            for s in (0, 1):
                r = R[f"{key}|{c} s{s}"][q]
                ax.bar(x, r["v"], 0.8, color=cl, alpha=1.0 if s == 0 else 0.6, yerr=[[r["v"] - r["lo"]], [r["hi"] - r["v"]]], error_kw=dict(lw=0.5), label=c if s == 0 else None)
                x += 1
            x += 0.5
        r = R[f"{key}|QH (decision 204, stored)"][q]
        ax.bar(x, r["v"], 0.8, color=PSY.BASELINE, yerr=[[r["v"] - r["lo"]], [r["hi"] - r["v"]]], error_kw=dict(lw=0.5), label="QH stored")
        if q == "rms":
            ax.axhline(R[f"{key}|L-full s0"]["policy"]["v"], color="#555555", lw=0.6, ls="--")
        ax.set_xticks([]); ax.set_ylabel(yl, fontsize=6.5); PSY.bars(ax)
    axs[0].legend(fontsize=6.5)
    fig.tight_layout(pad=0.4); PSY.save(fig, out / "h1b_a2_grid")
    print(out)


if __name__ == "__main__":
    from jevdrive.run import cli_args
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("sel", "ens", "final"):
        sub.add_parser(c)
    for c in ("feats", "read", "a2"):
        cli_args(sub.add_parser(c))
    f = sub.add_parser("fig"); f.add_argument("--src", required=True); f.add_argument("--out", required=True)
    a = ap.parse_args()
    {"sel": cmd_sel, "ens": cmd_ens, "final": cmd_final, "feats": cmd_feats, "read": cmd_read, "a2": cmd_a2, "fig": cmd_fig}[a.cmd](a)
