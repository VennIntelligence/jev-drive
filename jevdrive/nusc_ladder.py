"""Decisions 40, follow-up (i): the frozen-feature + ridge result replicated on nuScenes.

Pre-registration: todos/2026-09-24-driving-backbones/README.md, section "follow-up (i)". The P3 ladder's head and
judge carried over to nuScenes, with nothing fitted on val:

  rows      every trainval keyframe with >= MIN_HIST_S of its scene before it and 3 s of ego poses after it;
            fit on the 700 train scenes, evaluate on the 150 val scenes (the official split), one direction
  target    rear-axle positions at 0.25 ... 3.0 s in the t0 ego frame (12 x 2), from the 20 Hz ego poses
  ego       Waymo's ego input rebuilt from the poses -- 16 steps at 4 Hz (t0-3.75 ... t0) of position, velocity
            and acceleration in the t0 frame, both by backward differences (no future inside the input) -- plus
            the VAD command one-hot (left / straight / right from the 3 s lateral offset: the literature's leaky
            convention, standing in for WOD's routing intent); the no-command variant is a sensitivity row
  head      `ridge ego`, then `ridge_late` per feature on its residual, lambda by 4-fold scene-grouped CV on train
  judge     decision 22: ADE vs the log on pre-onset frames (not turning yet, turns within 3 s: waymo.subsets'
            thresholds) in s_ego deciles 1-9, all frames in deciles 1-9, straight as a side column; paired scene
            bootstrap. nuScenes has no rater labels, so there is no RFS column.

    .venv/bin/python -m jevdrive.nusc_ladder --steps index        # trainval stream index (CAM_FRONT + poses)
    .venv/bin/python -m jevdrive.nusc_ladder --steps ladder       # after scripts/nusc_backbone_openpilot.py
"""
import pickle

import numpy as np
import pandas as pd
import torch

from . import nuscenes_zs as Z
from . import planner, waymo, waymo_l0 as l0, waymo_p1 as P1, waymo_stage_a as sa
from .common import data_dir, get_logger

log = get_logger(__name__)
MIN_HIST_S = 4.5                          # ego input reaches back to t0 - 4.25 s (acceleration of the oldest step)
STEP, N_PAST, N_FUT = 0.25, 16, 12
BASE, BASE_NC = "ridge ego", "ridge ego (no cmd)"
REF_A = "A Qwen3-VL-4B L18_mean"
PRIMARY = {"op-cinque temporal": "openpilot Cinque", "op-lebowski temporal": "openpilot Lebowski"}
OP_MODELS = ("cinque", "lebowski")


def index(log_fn=log.info) -> dict:
    idx = Z.build_index(log_fn, splits=("train", "val"), cams=("CAM_FRONT",), boxes=False)
    p = Z.index_path("trainval")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(idx, f, protocol=5)
    return {"scenes": len(idx["scenes"]), "keyframes": len(idx["samples"]), "path": str(p)}


def build_rows(idx: dict) -> dict:
    """Targets, ego input, command and subsets for every usable keyframe (see the module docstring)."""
    toks, scene, split, past, fut = [], [], [], [], []
    j = np.arange(-34, 25)                                    # 0.125 s grid, t0 - 4.25 ... t0 + 3.0
    for e in idx["samples"]:
        sc = idx["scenes"][e["scene"]]
        if e["hist_s"] < MIN_HIST_S - 1e-3 or e["t0"] + 3.0e6 > sc["pose_t"][-1]:
            continue
        xyz, R = Z.ego_at(sc, e["t0"] + j * 0.125e6)
        p = ((xyz - xyz[j == 0]) @ R[j == 0][0])[:, :2]                   # t0 ego frame (R0^T applied)
        v = (p[2:] - p[:-2]) / STEP                                        # backward difference, at grid index k+2
        a = (v[2:] - v[:-2]) / STEP
        k = np.arange(-30, 1, 2)                                          # t0 - 3.75 ... t0, 16 steps
        at = lambda arr, off: arr[k - j[0] - off]                         # noqa: E731  (grid index -> array row)
        past.append(np.concatenate([at(p, 0), at(v, 2), at(a, 4)], 1))
        fut.append(p[np.arange(2, 25, 2) - j[0]])
        toks.append(e["token"]), scene.append(e["scene"]), split.append(sc["split"])
    past, fut = np.stack(past).astype(np.float32), np.stack(fut).astype(np.float32)
    y3 = fut[:, -1, 1]
    cmd = np.select([y3 >= 2, y3 <= -2], [0, 2], 1)                        # left / straight / right
    kin = waymo.past_kinematics(past)
    w = np.abs(np.degrees(kin["w"]))
    w_ok = np.linalg.norm(past[:, -1, :2] - past[:, -5, :2], axis=-1) >= waymo.MIN_PAST_DISP
    b, _, chord = waymo.future_maneuver(fut, 3.0)
    b_ok = chord >= waymo.MIN_CHORD
    later = b_ok & (np.abs(b) > waymo.ONSET_BEARING)
    sub = {"all": np.ones(len(fut), bool), "pre_onset": w_ok & (w < waymo.ONSET_YAW_RATE) & later,
           "straight_yaw": w_ok & (w < waymo.ONSET_YAW_RATE) & b_ok & ~later}
    return {"token": np.array(toks), "scene": np.array(scene), "split": np.array(split), "past": past,
            "fut": fut, "ego": past.reshape(len(past), -1), "cmd": np.eye(3, dtype=np.float32)[cmd], "sub": sub}


def op_features(tokens: np.ndarray, model: str) -> dict:
    """temporal / vision taps and the native plan per token, from the per-scene runner files."""
    d = data_dir() / "processed" / "drive_backbones" / "nusc_op" / model
    parts = []
    for f in sorted(d.glob("*.npz")):          # read and close: 850 lazy NpzFiles would exhaust the fd limit
        with np.load(f) as z:
            parts.append({k: z[k] for k in ("token", "temporal", "vision", "native")})
    at = pd.Series(np.arange(sum(len(q["token"]) for q in parts)),
                   index=np.concatenate([q["token"] for q in parts]).astype(str))
    pos = at.reindex(tokens).to_numpy()
    if np.isnan(pos).any():
        raise RuntimeError(f"{model}: {int(np.isnan(pos).sum())} rows without features")
    pos = pos.astype(int)
    return {k: np.concatenate([q[k] for q in parts])[pos].astype(np.float32) for k in ("temporal", "vision", "native")}


def qwen_features(tokens: np.ndarray, array: str = "L18_mean", set_name: str = "qwen_w800") -> np.ndarray:
    d = data_dir() / "processed" / "nuscenes" / Z.VERSION / "features" / set_name
    i = pd.read_parquet(d / "index.parquet")
    pos = pd.Series(np.arange(len(i)), index=i.sample_token.to_numpy()).reindex(tokens).to_numpy()
    if np.isnan(pos).any():
        raise RuntimeError(f"{set_name}: {int(np.isnan(pos).sum())} rows without features")
    return np.load(d / f"{array}.npy", mmap_mode="r")[pos.astype(int)].astype(np.float32)


def fit(r: dict, feats: dict, natives: dict) -> tuple[dict, dict, sa.Halves]:
    """`ridge ego` (+ no-command variant) and `ridge_late` per feature on train; val predictions of every arm."""
    sp = sa.Halves(None, r["scene"], r["split"] == "train", r["split"] == "val", 0)
    fut, n, T = r["fut"], len(r["fut"]), r["fut"].shape[1]
    F = torch.as_tensor(fut.reshape(n, -1), device=planner.DEV)
    preds, stats = {}, {}
    for base, ego in ((BASE, np.c_[r["ego"], r["cmd"]]), (BASE_NC, r["ego"])):
        Xe = planner.standardize(torch.as_tensor(ego, device=planner.DEV), sp.train)
        p, st, W = sa.ridge_cv(Xe, F, sp, fut)
        preds[base], stats[base] = p[:, 0], st
        R = F - planner.linear_apply(W, Xe, np.arange(n))[0]
        off = (F - R)[sp.val].reshape(-1, T, 2).cpu().numpy()
        res_fut = R.reshape(-1, T, 2).cpu().numpy()
        for name, X in feats.items():
            if base == BASE_NC and name not in (REF_A, *PRIMARY):
                continue
            Xi = planner.standardize(torch.as_tensor(X, device=planner.DEV), sp.train)
            p, st, _ = sa.ridge_cv(Xi, R, sp, res_fut)
            key = name if base == BASE else f"{name} (no cmd)"
            preds[key], stats[key] = p[:, 0] + off, {"d": X.shape[1], **st}
            log.info("%-36s lambda %.3g  sel ADE %.4f", key, st["lam"], st["sel_ade"])
    for name, P in natives.items():
        preds[name] = P[sp.val]
    s = l0.ego_surprise(np.c_[r["ego"], r["cmd"]], fut, sp)
    return preds, {"stats": stats, "s_ego": s}, sp


def judge(r: dict, preds: dict, s: np.ndarray, sp) -> dict[str, pd.DataFrame]:
    """Decision 22's readouts on val, every arm vs its base and every primary tap vs the general reference."""
    v = sp.val
    gt, seq = r["fut"][v], r["scene"][v]
    sv = s[v]
    dec = np.clip(np.searchsorted(np.quantile(sv, np.linspace(0, 1, 11))[1:-1], sv, "right"), 0, 9)
    k19 = dec <= 8
    masks = {"(i) pre-onset, deciles 1-9": r["sub"]["pre_onset"][v] & k19, "(ii) all frames, deciles 1-9": k19,
             "side: straight, deciles 1-9": r["sub"]["straight_yaw"][v] & k19,
             "side: all frames, all deciles": np.ones(len(v), bool), "side: decile 10": dec == 9}
    e = {k: l0.ade(p, gt) for k, p in preds.items()}
    fde = {k: np.linalg.norm(p[:, -1] - gt[:, -1], axis=-1) for k, p in preds.items()}

    def compare(a, b, what="ADE"):
        src = e if what == "ADE" else fde
        return [{"arm": a, "vs": b, "judge": what, "readout": name, **P1.paired(src[a], src[b], seq, m)}
                for name, m in masks.items()]

    base_of = lambda a: BASE_NC if a.endswith("(no cmd)") else BASE  # noqa: E731
    vs_base = [x for a in preds if a not in (BASE, BASE_NC) for x in compare(a, base_of(a))]
    vs_base += [x for a in (REF_A, *PRIMARY) if a in preds for x in compare(a, BASE, "FDE@3s")]
    vs_ref = [x for a in PRIMARY for ref, suf in ((REF_A, ""), (f"{REF_A} (no cmd)", " (no cmd)"))
              if a + suf in preds and ref in preds for x in compare(a + suf, ref)]
    t_base, t_ref = pd.DataFrame(vs_base), pd.DataFrame(vs_ref)
    verdict = []
    for a, label in PRIMARY.items():
        g = t_ref[(t_ref.arm == a) & t_ref.readout.str.startswith("(")]
        good, bad = bool((g.hi < 0).any()), bool((g.lo > 0).any())
        call = ("better" if good and not bad else "worse" if bad and not good else "mixed" if good and bad
                else "no detectable difference")
        gb = t_base[(t_base.arm == a) & (t_base.judge == "ADE") & t_base.readout.str.startswith("(")]
        verdict.append({"model": label, "arm": a, "vs A": call, "better than ridge ego": bool((gb.hi < 0).any()),
                        "worse than ridge ego": bool((gb.lo > 0).any())})
    info = pd.DataFrame([{"rows": len(r["fut"]), "train": len(sp.train), "val": len(v),
                          "train_scenes": len(np.unique(r["scene"][sp.train])), "val_scenes": len(np.unique(seq)),
                          **{f"val {k}": int(m.sum()) for k, m in masks.items()},
                          "val pre-onset all deciles": int(r["sub"]["pre_onset"][v].sum()),
                          "val cmd left/straight/right": "/".join(str(int(c)) for c in r["cmd"][v].sum(0))}])
    return {"nusc_vs_ego": t_base, "nusc_vs_general": t_ref, "nusc_verdict": pd.DataFrame(verdict), "nusc_info": info}


def ladder(rl) -> dict:
    idx = Z.load_index("trainval")
    r = build_rows(idx)
    log.info("rows %d (train %d / val %d); pre-onset %d, straight %d", len(r["fut"]), int((r["split"] == "train").sum()),
             int((r["split"] == "val").sum()), int(r["sub"]["pre_onset"].sum()), int(r["sub"]["straight_yaw"].sum()))
    feats, natives = {REF_A: qwen_features(r["token"])}, {}
    for m in OP_MODELS:
        f = op_features(r["token"], m)
        feats[f"op-{m} temporal"], feats[f"op-{m} vision"] = f["temporal"], f["vision"]
        natives[f"native op-{m} (no fit)"] = f["native"]
    feats["fusion op-cinque temporal + A"] = np.c_[feats["op-cinque temporal"], feats[REF_A]]
    preds, aux, sp = fit(r, feats, natives)
    out = judge(r, preds, aux["s_ego"], sp)
    out["nusc_fits"] = pd.DataFrame([{"arm": k, **v} for k, v in aux["stats"].items()])
    np.savez_compressed(rl.dir / "nusc_preds.npz", token=r["token"][sp.val], scene=r["scene"][sp.val],
                        fut=r["fut"][sp.val], s_ego=aux["s_ego"][sp.val],
                        **{f"sub_{k}": m[sp.val] for k, m in r["sub"].items()},
                        **{f"pred_{k}": p.astype(np.float32) for k, p in preds.items()})
    for name, t in out.items():
        t.to_csv(rl.dir / f"{name}.csv", index=False)
        rl.log.info("%s\n%s", name, t.to_markdown(index=False, floatfmt=".4f"))
    return out


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", default="ladder", help="comma list of index,ladder")
    a = ap.parse_args()
    torch.set_num_threads(4)
    rl = RunLog("nusc_backbones", a.steps.replace(",", "-"))
    rl.event("start", args=vars(a))
    for step in a.steps.split(","):
        if step == "index":
            rl.event("index", **index(rl.log.info))
        elif step == "ladder":
            ladder(rl)
    rl.event("end")
    rl.close()


if __name__ == "__main__":
    main()
