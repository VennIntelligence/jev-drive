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

def prepare() -> dict:
    """Write op_plan.json (streams, JPEG spans) and alp_targets.npz (target names, rows, past states)."""
    import pandas as pd
    from . import waymo as W
    from . import waymo_ladder as L
    df = W.load_index()
    past, _ = W.load_ego()
    t = pd.read_parquet(L.subset_path())
    names = W.frame_names(df)
    assert (names[t.row.to_numpy()] == t.frame_name.to_numpy()).all(), "subset rows moved in the index"
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
    (root() / "op_plan.json").write_text(json.dumps({"streams": streams, "spans": spans}))
    rows = t.row.to_numpy()
    np.savez(root() / "alp_targets.npz", name=t.frame_name.to_numpy().astype(str), row=rows,
             sequence=seq[rows], frame=frame[rows], past=past[rows])
    out = {"streams": len(streams), "stream_frames": sum(len(x["names"]) for x in streams),
           "targets": got, "sequences": int(t.sequence.nunique())}
    log.info("prepare: %s", out)
    return out


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


def finalize_op(model: str) -> dict:
    files = sorted(root("op", model).glob("*.npz"))
    parts = [np.load(f) for f in files]
    fn = np.concatenate([p["name"] for p in parts]).astype(str)
    arrs = {k: np.concatenate([p[k] for p in parts]) for k in OP_ARRAYS}
    arrs_native = {"wod": np.concatenate([p["wod"] for p in parts]), "hist": np.concatenate([p["hist"] for p in parts])}
    np.savez(root() / f"op_{model}_native.npz", name=fn, **arrs_native)
    t = {f.stem: json.loads(f.read_text()) for f in sorted(root("op", model).glob("timing_*.json"))}
    return _write_set(op_set(model), fn, arrs, {"model": model, "cams": "front,front_left,front_right", "timing": t})


def finalize_alp(name: str = ALP_SET) -> dict:
    files = sorted(root("alp", name).glob("part*.npz"))
    parts = [np.load(f) for f in files]
    fn = np.concatenate([p["name"] for p in parts]).astype(str)
    keys = [k for k in parts[0].files if k != "name"]
    arrs = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    return _write_set(name, fn, arrs, {"model": "nvidia/Alpamayo-1.5-10B", "cams": "front,front_left,front_right"})


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", default="prepare", help="comma list of prepare,finalize_op,finalize_alp")
    ap.add_argument("--models", default=",".join(OP_MODELS))
    a = ap.parse_args()
    rl = RunLog("drive_backbones", a.steps.replace(",", "-"))
    rl.event("start", args=vars(a))
    for step in a.steps.split(","):
        if step == "prepare":
            rl.event("prepare", **prepare())
        elif step == "finalize_op":
            for m in a.models.split(","):
                rl.event("finalize", **finalize_op(m))
        elif step == "finalize_alp":
            rl.event("finalize", **finalize_alp())
    rl.event("end")
    rl.close()


if __name__ == "__main__":
    main()
