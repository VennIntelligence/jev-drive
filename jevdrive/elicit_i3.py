"""Overnight queue item 5 / elicitation [I3-exam]: existing examinees zero-shot on the HUGSIM 3DGS pairs
(todos/2026-09-25-reactivity-program/i3-hugsim-pairs.md; registration: todos/2026-09-26-elicitation-program.md,
deviation-log entry [I3-exam] 01:13, written before any number).

  op-prepare  processed/hugsim_pairs/op_plan.json: one 5 Hz stream per rendered world, each scene with its own
              camgeom calibration from meta.json (c2front = hugsim_zs.calibs' E, K), openpilot rpy = 0
  qwen-plan   features/plan.parquet: the frames the obs / null tables reference (p5_qwen.work extracts them)
  exam        every examinee fitted on the P5 v1 BehaviorAgent set exactly as stored (reactivity_mc.fit_fold per route
              fold, heads verified against the stored run), the I3 frames riding along as rows that never enter a fit
              or a standardisation; prediction = mean of the 5 fold models; judge p5_exam.exam without the TFv6 columns
"""
import json
import os
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from .common import data_dir, get_logger

log = get_logger(__name__)
I3 = "hugsim_pairs"
BA = "carla_p5v1_ba"
MC_RUN = "runs/reactivity/mc-carla_p5v1_ba/20260925-233126"
MODELS = ("cinque", "lebowski")
CAMS = (("front", "CAM_FRONT"), ("front_left", "CAM_FRONT_LEFT"), ("front_right", "CAM_FRONT_RIGHT"))


@contextmanager
def p5_set(name: str):
    old = os.environ.get("P5_SET")
    os.environ["P5_SET"] = name
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("P5_SET")
        else:
            os.environ["P5_SET"] = old


def _root():
    return data_dir() / "processed" / I3


def scene_calib(cams: dict, wh=(800, 450)) -> dict:
    """meta.json cams -> camgeom records keyed like WOD CameraName (1 FRONT, 2 LEFT, 3 RIGHT)."""
    from .hugsim_zs import CV2V, WOD_IN_CV
    out = {}
    for i, (_, c) in enumerate(CAMS, 1):
        K, E = np.asarray(cams[c]["K"], np.float64), np.asarray(cams[c]["c2front"], np.float64)
        ext = np.eye(4)
        ext[:3, :3] = CV2V @ E[:3, :3] @ WOD_IN_CV
        ext[:3, 3] = CV2V @ E[:3, 3]
        out[str(i)] = {"intrinsic": [K[0, 0], K[1, 1], K[0, 2], K[1, 2], 0.0, 0.0, 0.0, 0.0, 0.0],
                       "extrinsic": ext.ravel().tolist(), "width": wh[0], "height": wh[1]}
    return out


def op_prepare() -> dict:
    from PIL import Image
    t = pd.read_parquet(_root() / "index.parquet")
    calibs, streams = {}, []
    for rid, g in t.groupby("route_id", sort=True):
        d = g.files.iloc[0][0].rsplit("/cams/", 1)[0]
        key = g.base_id.iloc[0]
        if key not in calibs:
            wh = Image.open(g.files.iloc[0][0]).size
            calibs[key] = scene_calib(json.loads((_root() / "scenes" / key / "meta.json").read_text())["cams"], wh)
        fr = sorted((json.loads(x) for x in open(f"{d}/frames.jsonl")), key=lambda r: r["frame"])
        names = [f"{rid}-{r['frame']:07d}" for r in fr]
        want = set(g.frame_name)
        last = max(i for i, n in enumerate(names) if n in want)
        fr, names = fr[: last + 1], names[: last + 1]
        tgt = [i for i, n in enumerate(names) if n in want]
        assert len(tgt) == len(want), f"{rid}: indexed frames missing from frames.jsonl"
        streams.append({"key": rid, "seq": key, "names": names, "targets": tgt,
                        "files": [[f"{d}/{r['files'][c]}" for c, _ in CAMS] for r in fr],
                        "gaps": int((np.diff([r["frame"] for r in fr]) != 4).sum())})
    (_root() / "op_plan.json").write_text(json.dumps({"calibs": calibs, "streams": streams}))
    info = {"streams": len(streams), "scenes": len(calibs), "frames": sum(len(s["names"]) for s in streams),
            "targets": sum(len(s["targets"]) for s in streams), "gaps": sum(s["gaps"] for s in streams)}
    log.info("op plan: %s", info)
    return info


def needed() -> pd.Series:
    o, n = pd.read_parquet(_root() / "obs.parquet"), pd.read_parquet(_root() / "null.parquet")
    return pd.Series(pd.unique(np.r_[o.fn_plus, o.fn_minus, n.fn_plus, n.fn_null]))


def qwen_plan(chunk: int = 1500):
    names = needed()
    d = _root() / "features"
    d.mkdir(exist_ok=True)
    pd.DataFrame({"frame_name": names, "chunk": np.arange(len(names)) // chunk}).to_parquet(d / "plan.parquet", index=False)
    log.info("qwen plan: %d frames", len(names))


def exam(rl):
    from . import p5_exam as E, p5_openpilot, p5_pairs as P, planner, reactivity_mc as MC, waymo_stage_a as sa
    dev = "cuda"
    with p5_set(BA):
        t, past, fut, obs, null, pairs = E.load()
        Q = P.load_features(t, ("L18_last",))["L18_last"]
        op = p5_openpilot.load(t, MODELS, sub="op_streams_vis")
        fold = E.folds(t, pairs)
    with p5_set(I3):
        t3a, past3a, _, obs3, null3, pairs3 = E.load()
        keep = t3a.frame_name.isin(set(needed())).to_numpy()
        t3, past3 = t3a[keep].reset_index(drop=True), past3a[keep]
        Q3 = P.load_features(t3, ("L18_last",))["L18_last"]
        op3 = p5_openpilot.load(t3, MODELS, sub="op_streams")
    n, n3 = len(t), len(t3)
    ta = pd.concat([t[["frame_name", "role", "base_id", "intent"]],
                    t3[["frame_name", "base_id", "intent"]].assign(role="i3", base_id="i3:" + t3.base_id)], ignore_index=True)
    fold_a = np.r_[fold, np.full(n3, -2)]
    F = torch.as_tensor(np.r_[fut.reshape(n, -1), np.zeros((n3, 40), np.float32)], device=dev)
    fut_np = F.reshape(-1, 20, 2).cpu().numpy()
    Ego = torch.as_tensor(np.r_[E.ego_input(t, past), E.ego_input(t3, past3)], device=dev)
    Qa = torch.as_tensor(np.r_[Q, Q3], device=dev)
    pos = pd.Series(np.arange(n), index=t.frame_name)
    pr_ip = np.r_[pos[obs.fn_plus].to_numpy(), pos[null.fn_plus].to_numpy()]
    pr_im = np.r_[pos[obs.fn_minus].to_numpy(), pos[null.fn_null].to_numpy()]
    pr_group = np.r_[obs.base_id.to_numpy(), null.base_id.to_numpy()].astype(str)
    obs_rows = np.flatnonzero(t.role.to_numpy() == "obs")
    ref = np.load(data_dir() / MC_RUN / "preds_obs.npz")
    at = pd.Series(np.arange(len(ref["rows"])), index=ref["rows"])
    I = np.arange(n, n + n3)
    preds, checks = {}, []

    def add(name, v):
        preds.setdefault(name, np.zeros((n3, 20, 2), np.float64))
        preds[name] += v[I].reshape(-1, 20, 2).cpu().double().numpy() / E.K_FOLDS

    for m in MODELS:
        Xop = torch.as_tensor(np.r_[op[f"op-{m} temporal"], op3[f"op-{m} temporal"]], device=dev)
        for f in range(E.K_FOLDS):
            o = MC.fit_fold(f, fold_a, ta, F, Ego, Xop, Qa, pr_ip, pr_im, pr_group, rl, m)
            ev = obs_rows[fold[obs_rows] == f]
            for arm, key in (("M-C pair", f"M-C pair [{m}]"), ("prior", f"prior [{m}]")):
                d = float(np.abs(o[arm][ev].reshape(-1, 20, 2).cpu().numpy() - ref[key][at[ev].to_numpy()]).max())
                checks.append({"model": m, "fold": f, "arm": arm, "max_abs_diff": d})
                assert d < 1e-3, f"{m} fold {f} {arm} does not reproduce the stored run ({d})"
            for arm, v in o.items():
                add(("ridge_late op-%s temporal" % m) if arm == "prior" else f"{arm} [{m}]", v)
        del Xop
        torch.cuda.empty_cache()
    # ridge ego and ridge_late on Qwen L18_last, p5_exam.heads' recipe, per fold, averaged
    for f in range(E.K_FOLDS):
        tr = np.flatnonzero((ta.role.to_numpy() == "train") & (fold_a != f))
        sp = SimpleNamespace(train=tr, val=I, seq=ta.base_id.to_numpy())
        Xe = planner.standardize(Ego, tr)
        _, _, We = sa.ridge_cv(Xe, F, sp, fut_np)
        base = planner.linear_apply(We, Xe, np.arange(n + n3))[0]
        R = F - base
        Xi = planner.standardize(Qa, tr)
        _, _, Wq = sa.ridge_cv(Xi, R, sp, R.reshape(-1, 20, 2).cpu().numpy())
        add("ridge ego", base)
        add("ridge_late L18_last", base + planner.linear_apply(Wq, Xi, np.arange(n + n3))[0])
    pd.DataFrame(checks).to_csv(rl.dir / "head_checks.csv", index=False)
    log.info("head checks: max |diff| %.2e m", max(c["max_abs_diff"] for c in checks))
    preds = {k: v.astype(np.float32) for k, v in preds.items()}
    np.savez_compressed(rl.dir / "preds_i3.npz", frame_name=t3.frame_name.to_numpy(), **preds)
    # judge: p5_exam unchanged but for the TFv6 columns I3 does not have
    E.TFV6 = {}
    oo, nn = E.deltas(obs3, null3, t3, preds)
    res = E.exam(oo, nn, pairs3, list(preds))
    fl = res["flips"]
    fl.to_csv(rl.dir / "flip_rates.csv", index=False)
    res["validity"].to_csv(rl.dir / "validity.csv", index=False)
    res["obs"].to_parquet(rl.dir / "obs_scored.parquet", index=False)
    # paired per-frame flip difference against each model's prior (descriptive, scene bootstrap)
    r = res["obs"][res["obs"].reactive]
    taus, rows = res["taus"], []

    def flips(sub, ex):
        return ((np.sign(sub[ex]) == np.sign(sub.d_expert)) & E._moved(sub[ex], taus[ex])).astype(float).to_numpy()
    for m in MODELS:
        pr = f"ridge_late op-{m} temporal"
        for ex in [k for k in preds if k.endswith(f"[{m}]")]:
            for scope, sub in [("pooled", r[r.family.isin(res["pooled_families"])])] + [(fa, r[r.family == fa]) for fa in sorted(r.family.unique())]:
                d, lo, hi = E.boot_ratio(flips(sub, ex) - flips(sub, pr), np.ones(len(sub)), sub.base_id.to_numpy())
                rows.append({"examinee": ex, "vs": pr, "scope": scope, "n": len(sub), "delta": d, "lo": lo, "hi": hi})
    pd.DataFrame(rows).to_csv(rl.dir / "paired_vs_prior.csv", index=False)
    log.info("tau_exp %.2f, pooled families %s\n%s", res["tau_exp"], res["pooled_families"],
             fl[fl.scope == "pooled"].to_markdown(index=False, floatfmt=".3f"))


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("op-prepare", "qwen-plan", "exam"))
    a = ap.parse_args()
    if a.cmd == "op-prepare":
        print(op_prepare())
    elif a.cmd == "qwen-plan":
        qwen_plan()
    else:
        rl = RunLog("elicitation", "i3-exam")
        exam(rl)
        rl.close()


if __name__ == "__main__":
    main()
