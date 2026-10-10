#!/usr/bin/env python3
"""HEAD1b step B2 reads (experiments/corridor, plans/2026-10-10-head1-prereg.md amendment 2026-10-10). EXPLORATORY: a second attempt after
the missed HEAD1 gate G3; nothing here is a registered read.

The arm: the SH30 pilot recipe (pp_train.py, P2 + hinge 30 / 0.5) with lib/heading_aux.py's head on the plan-pathway hidden state
(--aux-lam). The head is dropped at inference; navtest scores come from `python -m jevdrive.bench` only.

  ident  --a TAG --b TAG          the two checkpoints tensor by tensor (identity smoke: --aux-lam 0 against the plain run)
  sel    --tags L=TAG ...         lambda selection on the held-out validation logs (aux_eval.npz of runs trained with --holdout head1val):
                                  plan 4 s heading error RMS on rows with |logged 4 s heading change| > 20 deg, ADE; the amendment's rule
  pred   --tag TAG                (GPU) navtest pass of a trained arm with its head -> $H1/b2/pred/<TAG>.npz; plans checked against bench's
  acc    --tags TAG ... [--set navtest|eval]   the head's own accuracy and the plan's 4 s heading error, log-cluster bootstrap
Definitions (head1_read.py): plan error = unwrapped plan yaw at 4 s - logged yaw at 4 s; head error = profile interpolated at an arc
length - logged yaw at 4 s, at the plan's own 4 s arc length (pt_swap.Curve) and at the logged 4 s arc length (label s4, privileged arc).
Outputs under $DATA_DIR/runs/corridor/head1/b2/.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "lib"), str(REPO / "experiments/op_parity/scripts"), str(Path(__file__).parent)]
import head1_read as HR  # noqa: E402

D = HR.D
B2 = HR.H1 / "b2"
RUNS = D / "runs/op_parity/runs"
LAMS = (1.0, 3.0, 10.0)


def reads(z, lab) -> dict:
    """z: aux_eval / pred npz (names, log, poses, prof, fut); lab: label npz of the same domain -> per-bucket reads."""
    import pt_swap as PS
    names, fut, P = z["names"].astype(str), z["fut"].astype(np.float64)[:, :, :3], z["poses"].astype(np.float64)
    ln = lab["names"].astype(str)
    o = np.argsort(ln)
    j = o[np.searchsorted(ln[o], names)]
    assert (ln[j] == names).all(), "rows without a label row"
    L, s4 = lab["L"][j].astype(np.float64), lab["s4"][j].astype(np.float64)
    ok = np.isfinite(fut).all((1, 2))
    h4 = fut[:, 7, 2]
    dy = np.abs(np.degrees(HR.wrap(h4)))
    ep = np.degrees(HR.wrap(np.unwrap(np.concatenate([np.zeros((len(P), 1)), P[:, :, 2]], 1), axis=1)[:, -1] - h4))
    ade = np.hypot(P[..., 0] - fut[..., 0], P[..., 1] - fut[..., 1]).mean(1)
    sp = np.array([float(PS.Curve(p).sv[-1]) for p in P])
    prof = z["prof"].astype(np.float64)
    e_arc = np.degrees(HR.wrap(HR.at(prof, sp) - h4))
    e_log = np.degrees(HR.wrap(HR.at(prof, s4) - h4))
    e_lab = np.degrees(HR.wrap(HR.at(L, sp) - h4))                       # the label itself at the plan's arc (privileged reference)
    pt = np.degrees(prof - L)[:, :17]                                    # per-point error on the 0-40 m grid
    cb = HR.CB(z["log"].astype(str))
    st = lambda f, a, m: cb(f, a, mask=m)  # noqa: E731
    out = {"n": int(ok.sum()), "logs": int(len(set(z["log"][ok].tolist()))), "arc4_plan_m": float(np.median(sp[ok])), "arc4_log_m": float(np.nanmedian(s4[ok]))}
    for b, t in HR.BK.items():
        m = ok & (dy > t if t else True)
        out[b] = dict(n=int(m.sum()), plan_rms=st(HR.rms_a, ep, m), ade=st(HR.mean_a, ade, m), head_rms_plan_arc=st(HR.rms_a, e_arc, m & np.isfinite(e_arc)),
                      head_rms_log_arc=st(HR.rms_a, e_log, m & np.isfinite(e_log)), label_rms_plan_arc=st(HR.rms_a, e_lab, m & np.isfinite(e_lab)),
                      head_point_rms_0_40m=float(np.sqrt(np.nanmean(pt[m] ** 2))))
    return out


def cmd_ident(a):
    import torch
    A, B = (torch.load(RUNS / t / "ckpt-final.pt", map_location="cpu", weights_only=False)["model"] for t in (a.a, a.b))
    worst, n, diff = 0.0, 0, []
    for grp in ("net", "parity"):
        assert A[grp].keys() == B[grp].keys(), f"{grp}: different tensors"
        for k in A[grp]:
            d = float((A[grp][k].double() - B[grp][k].double()).abs().max())
            n += 1
            if d > 0:
                diff.append(k)
            worst = max(worst, d)
    ev = {t: {k: v for f in sorted(D.glob(f"runs/op_parity/train-{t}/*/DONE"))[-1:] for k, v in json.loads(f.read_text()).items() if k.startswith("dev_")} for t in (a.a, a.b)}
    r = dict(a=a.a, b=a.b, tensors=n, differing=len(diff), max_abs_diff=worst, bit_identical=not diff, dev=ev)
    B2.mkdir(parents=True, exist_ok=True)
    (B2 / "ident.json").write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))
    assert not diff, f"{len(diff)} tensors differ (max {worst:.3e})"


def cmd_sel(a):
    import torch
    lab = np.load(HR.H1 / "labels/navtrain.npz")
    R, names = {}, None
    for spec in a.tags:
        lam, tag = spec.split("=")
        z = np.load(RUNS / tag / "aux_eval.npz")
        assert names is None or (z["names"] == names).all(), "selection runs read different rows"
        names = z["names"]
        ck = torch.load(RUNS / tag / "ckpt-final.pt", map_location="cpu", weights_only=False)["cfg"]
        assert ck["holdout"] == "head1val" and float(ck["aux_lam"]) == float(lam), f"{tag}: not a selection run of lambda {lam}"
        R[float(lam)] = dict(tag=tag, **reads(z, lab))
    assert 0.0 in R, "the lambda = 0 reference is missing"
    r0, a0 = R[0.0][">20"]["plan_rms"]["v"], R[0.0]["all"]["ade"]["v"]
    cand = [l for l in LAMS if l in R]
    okade = [l for l in cand if R[l]["all"]["ade"]["v"] <= 1.02 * a0]
    pick = min(okade or cand, key=lambda l: R[l][">20"]["plan_rms"]["v"])
    sel = dict(rule="lowest plan 4 s heading RMS (> 20 deg rows) among lambda with ADE <= 1.02 x lambda 0's; if none is below lambda 0, still the lowest",
               chosen=pick, ade_ok=okade, below_ref=bool(R[pick][">20"]["plan_rms"]["v"] < r0), any_ade_ok=bool(okade), ref_rms=r0, ref_ade=a0)
    rows = [{"lambda": l, "tag": r["tag"], "rows (> 20 deg)": f"{r['n']} ({r['>20']['n']})", "logs": r["logs"],
             "plan 4 s heading RMS, > 20 deg (deg)": HR.fmt(r[">20"]["plan_rms"]), "vs lambda 0": f"{r['>20']['plan_rms']['v'] - r0:+.2f}",
             "> 45 deg": HR.fmt(r[">45"]["plan_rms"]), "ADE (m)": HR.fmt(r["all"]["ade"], "{:.4f}"), "ADE / lambda 0": f"{r['all']['ade']['v'] / a0:.4f}",
             "head RMS at the plan's arc, > 45 / all (deg)": f"{r['>45']['head_rms_plan_arc']['v']:.2f} / {r['all']['head_rms_plan_arc']['v']:.2f}",
             "chosen": "yes" if l == pick else ""} for l, r in sorted(R.items())]
    B2.mkdir(parents=True, exist_ok=True)
    (B2 / "sel.json").write_text(json.dumps(dict(selection=sel, runs={str(k): v for k, v in R.items()}), indent=1))
    txt = ("EXPLORATORY (HEAD1b B2). Selection runs: seed 0, trained without the validation logs, read on their rows.\n\n" + HR.md(rows, list(rows[0])) +
           f"\nchosen lambda {pick}; ADE within 2 %: {okade}; below the lambda 0 reference: {sel['below_ref']}\n")
    (B2 / "sel.md").write_text(txt)
    print(txt)


def cmd_pred(a):
    import torch
    import heading_aux as HX
    import pp_train as PT
    from jevdrive.run import Run
    with Run("corridor", f"head1/b2-pred-{a.tag}", config=vars(a)) as run:
        dev = torch.device("cuda")
        S = PT.Store(("lb_navtest",), dev, need_side=False, frames="warp", host=True)
        pi = np.load(PT.data_dir() / "runs/op_parity/cache/navtrain_full.s2of12/teacher.npz")["pi"]
        model = PT.load_pmodel(a.tag, dev)
        aux = HX.HeadingAux(HR.H1 / "labels/navtest.npz", S.tab["names"], dev)
        aux.net.load_state_dict(torch.load(RUNS / a.tag / "aux.pt", map_location="cpu"))
        W = torch.as_tensor(PT.R2.t_weights(PT.T8), device=dev)
        (B2 / "pred").mkdir(parents=True, exist_ok=True)
        tmp = B2 / "pred" / f".{a.tag}.tmp.npz"
        aux.dump(model, S, np.arange(S.n), W, PT.rear, tmp, pi=pi)
        z = np.load(tmp)
        f = HR.OL / "lb_navtest/preds" / f"{a.tag}-warp__base.npz"
        if f.exists():                                                   # the same plans as bench's own pass
            b = np.load(f)
            pos = {k: i for i, k in enumerate(b["tokens"].tolist())}
            dmax = float(np.abs(b["poses"][[pos[k] for k in z["names"].tolist()]][..., :2] - z["poses"][..., :2]).max())
            run.info("plans against bench's: max |dxy| %.4f m", dmax)
            run.summary.update(bench_max_dxy=dmax)
            assert dmax < 0.05, "this pass does not reproduce bench's plans"
        tmp.rename(B2 / "pred" / f"{a.tag}.npz")
        run.summary.update(rows=int(S.n), coverage=aux.coverage)


def cmd_acc(a):
    lab = np.load(HR.H1 / ("labels/navtest.npz" if a.set == "navtest" else "labels/navtrain.npz"))
    R = {t: reads(np.load(B2 / "pred" / f"{t}.npz" if a.set == "navtest" else RUNS / t / "aux_eval.npz"), lab) for t in a.tags}
    rows = [{"tag": t, "bucket": b, "n": r[b]["n"], "plan 4 s heading RMS (deg)": HR.fmt(r[b]["plan_rms"]), "head at the plan's 4 s arc": HR.fmt(r[b]["head_rms_plan_arc"]),
             "head at the logged 4 s arc": HR.fmt(r[b]["head_rms_log_arc"]), "label L at the plan's arc (privileged)": HR.fmt(r[b]["label_rms_plan_arc"]),
             "head per-point RMS 0-40 m": f"{r[b]['head_point_rms_0_40m']:.2f}", "ADE (m)": HR.fmt(r[b]["ade"], "{:.3f}")} for t, r in R.items() for b in HR.BK]
    (B2 / f"acc_{a.set}.json").write_text(json.dumps(R, indent=1))
    txt = f"EXPLORATORY (HEAD1b B2). Auxiliary head and plan, set = {a.set}.\n\n" + HR.md(rows, list(rows[0]))
    (B2 / f"acc_{a.set}.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ident"); p.add_argument("--a", required=True); p.add_argument("--b", required=True)
    p = sub.add_parser("sel"); p.add_argument("--tags", nargs="+", required=True, help="lambda=tag")
    p = sub.add_parser("pred"); p.add_argument("--tag", required=True)
    p = sub.add_parser("acc"); p.add_argument("--tags", nargs="+", required=True); p.add_argument("--set", default="navtest", choices=("navtest", "eval"))
    a = ap.parse_args()
    {"ident": cmd_ident, "sel": cmd_sel, "pred": cmd_pred, "acc": cmd_acc}[a.cmd](a)
