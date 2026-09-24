"""Driving-specialised backbones (openpilot, Alpamayo 1.5) as frozen feature extractors in the P3 ladder.

Pre-registration: todos/2026-09-24-driving-backbones/README.md. The ladder itself (subset, split, head, judge) is
`waymo_ladder` unchanged; this module only produces feature sets in the ladder's flat format and the extra
readouts the pre-registration adds on top.

  prepare       per-sequence streaming plan for openpilot (every val frame from the sequence's first indexed
                frame to its last subset frame, split at index gaps) and the Alpamayo target list, as JSON / npz
                readable from the openpilot and alpamayo venvs (no pandas there)
  finalize      the model runners' per-sequence / per-shard npz files -> features/<set>/{index.parquet, *.npy}
  crossfit      the pooled cross-fit readout: both directions' out-of-sample predictions concatenated (the two
                eval halves are disjoint and cover the subset), paired against the references by sequence bootstrap

The runners live in scripts/drive_backbones_openpilot.py and scripts/drive_backbones_alpamayo.py.
"""
import json
from pathlib import Path

import numpy as np

from .common import data_dir, get_logger

log = get_logger(__name__)
OP_MODELS = ("small", "cinque", "lebowski")
OP_TAPS = {  # tap name -> ONNX value (see the pre-registration's tap table)
    "small": {"temporal": "model/p_select", "vision": "model/view"},
    "cinque": {"temporal": "select_4", "vision": "mean"},
    "lebowski": {"temporal": "select", "vision": "view_40"},
}
OP_ARRAYS = ("temporal", "vision", "hidden", "plan")
ALP_SET = "alpamayo15_p3"


def root(*parts) -> Path:
    d = data_dir() / "processed" / "drive_backbones" / Path(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def op_set(model: str) -> str:
    return f"op_{model}_p3"


# ---------------------------------------------------------------- prepare (project venv)

def prepare(split: str = "subset") -> dict:
    """Write op_plan[_trainval].json (streams, JPEG spans) and, for the subset, alp_targets.npz.

    split "subset": the frozen P2/P3 subset. "trainval": every train and val frame with a future (the rows
    `waymo_p0.load_all` fits and evaluates on), for the pre-registered train-split follow-up."""
    import pandas as pd
    from . import waymo as W
    from . import waymo_ladder as L
    df = W.load_index()
    past, future = W.load_ego()
    names = W.frame_names(df)
    if split == "subset":
        t = pd.read_parquet(L.subset_path())
        assert (names[t.row.to_numpy()] == t.frame_name.to_numpy()).all(), "subset rows moved in the index"
    else:
        r = L.trainval_rows(df, past, future)
        t = pd.DataFrame({"frame_name": names[r], "row": r, "sequence": df.sequence.astype(str).to_numpy()[r]})
    seq = df.sequence.astype(str).to_numpy()
    frame = df.frame.to_numpy()
    target = set(t.frame_name)
    streams, spans = [], {}
    by_seq = pd.Series(np.arange(len(df))).groupby(seq).apply(np.asarray)
    for s in sorted(t.sequence.astype(str).unique()):
        rows = by_seq[s]
        rows = rows[np.argsort(frame[rows])]
        last = max(frame[r] for r in rows if names[r] in target)
        rows = rows[frame[rows] <= last]
        cut = np.flatnonzero(np.diff(frame[rows]) != 1) + 1          # an index gap restarts the stream
        for run in np.split(rows, cut):
            run_names = [str(names[r]) for r in run]
            if any(n in target for n in run_names):
                streams.append({"sequence": s, "names": run_names,
                                "targets": [i for i, n in enumerate(run_names) if n in target]})
            for r in run:
                spans[str(names[r])] = [str(df.shard.iloc[r])] + [int(df[f"{c}_{k}"].iloc[r])
                                                                   for c in W.CAMS for k in ("off", "len")]
    got = sum(len(x["targets"]) for x in streams)
    assert got == len(t), f"{got} targets in streams, {len(t)} in the subset"
    (root() / plan_name(split)).write_text(json.dumps({"streams": streams, "spans": spans}))
    out = {"split": split, "streams": len(streams), "stream_frames": sum(len(x["names"]) for x in streams),
           "targets": got, "sequences": int(t.sequence.nunique())}
    log.info("prepare: %s", out)
    if split != "subset":
        return out
    rows = t.row.to_numpy()
    np.savez(root() / "alp_targets.npz", name=t.frame_name.to_numpy().astype(str), row=rows,
             sequence=seq[rows], frame=frame[rows], past=past[rows])
    return out


def write_calib(split: str) -> Path:
    """Front-three calibration of every streamed sequence (op_calib_<split>.json). The exam's op_calib.json only
    covers val; this is the same record read (`wod_zeroshot.write_op_calib`) written to our own file."""
    from . import waymo as W
    from . import wod_zeroshot as Z
    plan = json.loads((root() / plan_name(split)).read_text())
    E2ED, out = W.e2ed_frame(), {}
    df = W.load_index()
    key = dict(zip(W.frame_names(df), range(len(df))))
    for st in plan["streams"]:
        seq = st["sequence"]
        if seq in out:
            continue
        r = df.iloc[key[st["names"][0]]]
        with open(W.shard_dir() / str(r.shard), "rb") as f:
            f.seek(int(r.rec_off))
            fr = E2ED.FromString(f.read(int(r.rec_len))).frame
        out[seq] = {str(c): {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
                    for c, d in Z.calib_dict(fr, Z.OP_SRC).items()}
    p = root() / f"op_calib_{split}.json"
    p.write_text(json.dumps(out))
    log.info("calibration of %d sequences -> %s", len(out), p)
    return p


def plan_name(split: str) -> str:
    return "op_plan.json" if split == "subset" else f"op_plan_{split}.json"


# ---------------------------------------------------------------- finalize (project venv)

def _write_set(name: str, fnames: np.ndarray, arrays: dict, meta: dict) -> dict:
    """The flat layout `waymo.load_flat_features` reads: index.parquet + one float16 .npy per array."""
    import pandas as pd
    from . import waymo as W
    dst = W.out_dir("features", name)
    (dst / "meta.json").unlink(missing_ok=True)
    df = W.load_index()
    at = pd.Series(np.arange(len(df)), index=W.frame_names(df))
    order = np.argsort(fnames)
    fnames = fnames[order]
    for k, v in arrays.items():
        np.save(dst / f"{k}.npy", np.asarray(v)[order].astype(np.float16))
    pd.DataFrame({"frame_name": fnames, "row": at.reindex(fnames).to_numpy(), "cam": meta.get("cams", "")}
                 ).to_parquet(dst / "index.parquet", index=False)
    meta = {"set": name, "rows": len(fnames), "features": sorted(arrays), **meta}
    (dst / "meta.json").write_text(json.dumps(meta, indent=2, default=float))
    log.info("%s: %d rows, arrays %s -> %s", name, len(fnames), {k: np.asarray(v).shape[1] for k, v in arrays.items()}, dst)
    return meta


def finalize_op(model: str, split: str = "subset") -> dict:
    sub = "op" if split == "subset" else f"op_{split}"
    files = sorted(root(sub, model).glob("*.npz"))
    parts = [np.load(f) for f in files]
    fn = np.concatenate([p["name"] for p in parts]).astype(str)
    arrs = {k: np.concatenate([p[k] for p in parts]) for k in OP_ARRAYS}
    arrs_native = {"wod": np.concatenate([p["wod"] for p in parts]), "hist": np.concatenate([p["hist"] for p in parts])}
    sfx = "" if split == "subset" else f"_{split}"
    np.savez(root() / f"op_{model}{sfx}_native.npz", name=fn, **arrs_native)
    t = {f.stem: json.loads(f.read_text()) for f in sorted(root(sub, model).glob("timing_*.json"))}
    return _write_set(op_set(model) + sfx, fn, arrs, {"model": model, "cams": "front,front_left,front_right",
                                                      "timing": t})


def finalize_alp(name: str = ALP_SET) -> dict:
    files = sorted(root("alp", name).glob("part*.npz"))
    parts = [np.load(f) for f in files]
    fn = np.concatenate([p["name"] for p in parts]).astype(str)
    keys = [k for k in parts[0].files if k != "name"]
    arrs = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    return _write_set(name, fn, arrs, {"model": "nvidia/Alpamayo-1.5-10B", "cams": "front,front_left,front_right"})


# ---------------------------------------------------------------- the ladder run and the extra readouts

REF_A = "A ridge_late pooled (qwen4b L18)"
REF_VID = "d2 qwenvid L18_last"
ALP_ARRAYS = ["vit_mean", "vis_mean"] + [f"L{k:02d}_{p}" for k in (9, 18, 27, 36) for p in ("mean", "last")]
PRIMARY = {"op-small temporal": "openpilot small", "op-cinque temporal": "openpilot Cinque",
           "op-lebowski temporal": "openpilot Lebowski", "alp L18_mean": "Alpamayo 1.5"}
LADDER_SETS = {  # arm prefix -> (feature set, arrays); references first, as in the pre-registration
    "d vjepa2": ("vjepa2_p3", ["mean"]),
    "d2 qwenvid": ("qwenvid_p3", ["L18_last", "L18_mean"]),
    **{f"op-{m}": (op_set(m), list(OP_ARRAYS)) for m in OP_MODELS},
    "alp": (ALP_SET, ALP_ARRAYS),
}
MIN_HIST = 48   # frames: a full 4.8 s openpilot feature queue (the sensitivity row)


def op_history(ctx: dict, model: str = "cinque") -> np.ndarray:
    """Frames of streamed history before every context row (-1 where the row is not a target)."""
    import pandas as pd
    z = np.load(root() / f"op_{model}_native.npz")
    return pd.Series(z["hist"], index=z["name"].astype(str)).reindex(ctx["fname"]).fillna(-1).to_numpy()


def ladder(ctx: dict, keep: np.ndarray, rl, tag: str = "p3drive", min_hist: int | None = None):
    """One `waymo_ladder` run with every reference and every driving row, over the intersection of coverage."""
    from . import waymo_ladder as L
    arms, got, keep = {REF_A: L.ridge_arm(ctx["pooled"])}, {}, keep.copy()
    for name, (st, want) in LADDER_SETS.items():
        a = L.align(ctx, st, want)
        got[name], keep = a, keep & a["covered"]
        for w in want:
            arms[f"{name} {w}"] = L.ridge_arm(a[w])
    arms["fusion op-cinque temporal + A"] = L.ridge_arm(L.Concat([got["op-cinque"]["temporal"], ctx["pooled"]]))
    arms["fusion alp L18_mean + A"] = L.ridge_arm(L.Concat([got["alp"]["L18_mean"], ctx["pooled"]]))
    if min_hist:
        keep &= op_history(ctx) >= min_hist
    log.info("%s: %d arms over %d frames", tag, len(arms), int(keep.sum()))
    return L.run_ladder(lambda k: arms, tag, keep, ctx, rl=rl)


P0_RUN = "/root/autodl-tmp/ujs/runs/waymo_p0/train_split/20260922-175708/"   # the s_ego the P3 train-split run reused


def ladder_train(rl, models=("cinque", "lebowski"), p0_run: str = P0_RUN):
    """The pre-registered follow-up: fit on every train frame, evaluate on every val frame (the P3 train-split protocol)."""
    from . import waymo_ladder as L
    ctx = L.train_context(p0_run=p0_run)
    arms, keep = {REF_A: L.ridge_arm(ctx["pooled"])}, np.ones(len(ctx["fname"]), bool)
    for m in models:
        a = L.align(ctx, op_set(m) + "_trainval", ["temporal"])
        keep &= a["covered"]
        arms[f"op-{m} temporal (train fit)"] = L.ridge_arm(a["temporal"])
    log.info("p3drive_train: %d arms over %d rows (%d evaluated)", len(arms), int(keep.sum()),
             int((keep & (ctx["half"] == 1)).sum()))
    return L.run_ladder(lambda k: arms, "p3drive_train", keep, ctx, directions=(0,), rl=rl)


def _pooled(run_dir, tag: str) -> dict:
    """Both directions' out-of-sample predictions, concatenated (the eval halves are disjoint and cover the rows).
    s_ego deciles are taken per direction, inside its own eval half, exactly as `waymo_ladder.rejudge` does."""
    import pandas as pd
    from . import waymo as W
    from . import waymo_ladder as L
    df = W.load_index()
    past, future = W.load_ego()
    sub_all = W.subsets(df, past, future)
    at = pd.Series(np.arange(len(df)), index=W.frame_names(df))
    parts = []
    for f in L._preds_files(run_dir, tag):
        d = np.load(f, allow_pickle=True)
        s = d["s_ego"]
        dec = np.clip(np.searchsorted(np.quantile(s, np.linspace(0, 1, 11))[1:-1], s, "right"), 0, 9)
        arms = {k[5:]: d[k] for k in d.files if k.startswith("pred_")}
        parts.append({"fname": d["frame_name"].astype(str), "gt": d["fut"], "speed": d["speed"], "dec": dec,
                      "dir": np.full(len(s), int(f.stem.rsplit("dir", 1)[1])), **{f"p:{k}": v for k, v in arms.items()}})
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    idx = at.reindex(out["fname"]).to_numpy().astype(int)
    out["seq"] = df.sequence.astype(str).to_numpy()[idx]
    out["cluster"] = df.cluster.astype(str).to_numpy()[idx]
    out["sub"] = {k: sub_all[k][idx] for k in ("pre_onset", "straight_yaw", "all")}
    r, rtraj, rscore = W.load_rater(df)
    pos = pd.Series(np.arange(len(idx)), index=out["fname"]).reindex(W.frame_names(df)[r]).to_numpy()
    ok = ~np.isnan(pos)
    out["rpos"], out["rtraj"], out["rscore"] = pos[ok].astype(int), rtraj[ok], rscore[ok]
    assert len(np.unique(out["fname"])) == len(out["fname"]), "a frame evaluated in both directions"
    return out


def native_preds(fname: np.ndarray) -> dict:
    """openpilot's own plans (no fit) on the given frames, as pseudo-arms."""
    import pandas as pd
    out = {}
    for m in OP_MODELS:
        f = root() / f"op_{m}_native.npz"
        if f.exists():
            z = np.load(f)
            pos = pd.Series(np.arange(len(z["name"])), index=z["name"].astype(str)).reindex(fname).to_numpy()
            if not np.isnan(pos).any():
                out[f"native op-{m} (no fit)"] = z["wod"][pos.astype(int)]
    return out


def crossfit(run_dir, tag: str = "p3drive") -> dict:
    """The pre-registered cross-fit readouts: every arm against `ridge ego`, and every primary driving tap against
    the two general references, on (i) pre-onset deciles 1-9, (ii) all frames deciles 1-9, (iii) rater RFS."""
    import pandas as pd
    from . import waymo as W
    from . import waymo_p1 as P1
    from .waymo_ladder import BASE
    d = _pooled(run_dir, tag)
    preds = {k[2:]: v for k, v in d.items() if k.startswith("p:")} | native_preds(d["fname"])
    gt, seq, pos = d["gt"], d["seq"], d["rpos"]
    e = {k: np.linalg.norm(p - gt, axis=-1).mean(1) for k, p in preds.items()}
    rfs = {k: W.rater_feedback_score(p[pos], d["rtraj"], d["rscore"], d["speed"][pos]) for k, p in preds.items()}
    keep19 = d["dec"] <= 8
    masks = {"(i) pre-onset, deciles 1-9": d["sub"]["pre_onset"] & keep19, "(ii) all frames, deciles 1-9": keep19,
             "side: straight, deciles 1-9": d["sub"]["straight_yaw"] & keep19}
    rater_seq, ones = seq[pos], np.ones(len(pos), bool)

    def compare(a, b):
        rows = [{"arm": a, "vs": b, "readout": name, **P1.paired(e[a], e[b], seq, m)} for name, m in masks.items()]
        rows.append({"arm": a, "vs": b, "readout": "(iii) RFS, rater frames", **P1.paired(rfs[a], rfs[b], rater_seq, ones),
                     "cluster_mean": W.rfs_by_cluster(rfs[a], d["cluster"][pos])[0],
                     "cluster_mean_vs": W.rfs_by_cluster(rfs[b], d["cluster"][pos])[0]})
        return rows

    vs_base = [r for a in preds if a != BASE for r in compare(a, BASE)]
    vs_ref = [r for a in PRIMARY for ref in (REF_A, REF_VID) if a in preds for r in compare(a, ref)]
    t_base, t_ref = pd.DataFrame(vs_base), pd.DataFrame(vs_ref)
    verdict = []
    for a, label in PRIMARY.items():
        if a not in preds:
            continue
        g = t_ref[t_ref.arm == a]
        good = ((g.readout.str.contains("RFS") & (g.lo > 0)) | (~g.readout.str.contains("RFS") & (g.hi < 0)))
        bad = ((g.readout.str.contains("RFS") & (g.hi < 0)) | (~g.readout.str.contains("RFS") & (g.lo > 0)))
        main = ~g.readout.str.startswith("side")
        per_ref = {ref: (bool((good & main & (g.vs == ref)).any()), bool((bad & main & (g.vs == ref)).any()))
                   for ref in (REF_A, REF_VID)}
        any_good, any_bad = any(v[0] for v in per_ref.values()), any(v[1] for v in per_ref.values())
        all_good = all(v[0] for v in per_ref.values())
        call = ("better" if all_good and not any_bad else "worse" if all(v[1] for v in per_ref.values()) and not any_good
                else "mixed" if any_good and any_bad else "no detectable difference" if not any_good and not any_bad
                else "partly better" if any_good else "partly worse")
        verdict.append({"model": label, "arm": a, "verdict": call,
                        **{f"better_than {r}": v[0] for r, v in per_ref.items()},
                        **{f"worse_than {r}": v[1] for r, v in per_ref.items()}})
    info = {"frames": len(gt), "rater": len(pos), "pre19": int(masks["(i) pre-onset, deciles 1-9"].sum()),
            "all19": int(keep19.sum())}
    log.info("crossfit over %s", info)
    return {"crossfit_vs_ego": t_base, "crossfit_vs_general": t_ref, "crossfit_verdict": pd.DataFrame(verdict),
            "_info": info}


def rater_native_front3() -> dict:
    """Check 2: Alpamayo's own K=6 no-nav trajectories from front-three input vs the exam's seven-camera run."""
    import pandas as pd
    from . import waymo as W
    S = Z_sets()
    r = S["rater"]
    names = [str(x) for x in r["name"]]
    speed = W.init_speed(r["past"])
    out = {}
    for label, d in (("7 cameras (exam)", _zroot("preds", "alpamayo_nonav")), ("front three", root("alp_native_front3"))):
        xyz = np.stack([np.load(d / f"{n}.npz")["xyz"] for n in names])
        p = _zresample(xyz[..., :2])
        rfs = np.stack([W.rater_feedback_score(p[:, j], r["traj"], r["scores"], speed) for j in range(p.shape[1])], 1)
        out[label] = rfs.mean(1)                              # expected RFS of one sample, the exam's headline
    from . import traj
    a, b = out["front three"], out["7 cameras (exam)"]
    lo, hi = traj.boot_ci(a - b, np.array(names))
    cl = r["cluster"].astype(str)
    return {"n": len(names), "rfs_7cam_frame": float(b.mean()), "rfs_front3_frame": float(a.mean()),
            "rfs_7cam_cluster": W.rfs_by_cluster(b, cl)[0], "rfs_front3_cluster": W.rfs_by_cluster(a, cl)[0],
            "delta_frame": float((a - b).mean()), "lo": lo, "hi": hi}


def Z_sets():
    from . import wod_zeroshot as Z
    return Z.load_sets()


def _zroot(*p):
    from . import wod_zeroshot as Z
    return Z.root(*p)


def _zresample(xy):
    from . import wod_zeroshot as Z
    return Z.resample(Z.ALP_T, xy)


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", default="prepare",
                    help="comma list of prepare,finalize_op,finalize_alp,ladder,ladder_hist,crossfit,native_front3")
    ap.add_argument("--models", default=",".join(OP_MODELS))
    ap.add_argument("--run", default=None, help="crossfit: the ladder run directory (default: this run)")
    ap.add_argument("--split", default="subset", choices=("subset", "trainval"))
    a = ap.parse_args()
    rl = RunLog("drive_backbones", a.steps.replace(",", "-"))
    rl.event("start", args=vars(a))
    for step in a.steps.split(","):
        if step == "prepare":
            rl.event("prepare", **prepare(a.split))
            if a.split != "subset":
                write_calib(a.split)
        elif step == "finalize_op":
            for m in a.models.split(","):
                rl.event("finalize", **finalize_op(m, a.split))
        elif step == "ladder_train":
            from . import waymo_ladder as L
            ladder_train(rl, tuple(a.models.split(",")))
            for name, t in L.rejudge(rl.dir, "p3drive_train").items():
                t.to_csv(rl.dir / f"{name}_p3drive_train.csv", index=False)
                rl.log.info("%s\n%s", name, t.to_markdown(index=False, floatfmt=".4f"))
        elif step == "finalize_alp":
            rl.event("finalize", **finalize_alp())
        elif step in ("ladder", "ladder_hist"):
            from . import waymo_ladder as L
            ctx = L.base_context()
            tag = "p3drive" if step == "ladder" else "p3drive_hist"
            ladder(ctx, L.load_subset(ctx), rl, tag, MIN_HIST if step == "ladder_hist" else None)
            for name, t in L.rejudge(rl.dir, tag).items():
                t.to_csv(rl.dir / f"{name}_{tag}.csv", index=False)
                rl.log.info("%s\n%s", name, t.to_markdown(index=False, floatfmt=".4f"))
            for name, t in crossfit(rl.dir, tag).items():
                if name.startswith("_"):
                    rl.event("crossfit_info", tag=tag, **t)
                    continue
                t.to_csv(rl.dir / f"{name}_{tag}.csv", index=False)
                rl.log.info("%s\n%s", name, t.to_markdown(index=False, floatfmt=".4f"))
        elif step == "crossfit":
            for name, t in crossfit(a.run).items():
                if not name.startswith("_"):
                    t.to_csv(Path(a.run) / f"{name}_p3drive.csv", index=False)
                    rl.log.info("%s\n%s", name, t.to_markdown(index=False, floatfmt=".4f"))
        elif step == "native_front3":
            r = rater_native_front3()
            rl.log.info("native front3: %s", r)
            (root() / "alp_native_front3.json").write_text(json.dumps(r, indent=2))
            rl.event("native_front3", **r)
    rl.event("end")
    rl.close()


if __name__ == "__main__":
    main()
