#!/usr/bin/env python
"""junction_eyes WOD canary (research/next-round/junction-eyes-plan.md 4.5 G2 (c)): a merged Cinque ONNX zero-shot on the 479 WOD-E2E val
rater frames through the serving harness scripts/wod_zeroshot_openpilot.py --onnx (openpilot env, TensorRT fp16, real 10 Hz frames fed
twice, zero state per target), paired against the factory run of the same harness (preds/op_cinque: RFS 8.005, decision 183).

Scoring is wod_parity / wa_xboard's (experiments/op_parity/scripts/wod_launch_report.Ctx): per-frame jevdrive.waymo.rater_feedback_score,
RFS = mean over clusters of the cluster means, paired difference arm - factory with the bootstrap over sequences (B 4 000, seed 0).
Gate G2 (c): CI lower bound of the paired RFS difference >= -0.20.

Predictions: $DATA_DIR/processed/wod_zeroshot/preds/op_cinque_je-<tag>/ (the harness skips targets already there). A sidecar je_onnx.json
holds the ONNX's sha256; when it differs (same tag, new weights) the predictions and the tag's TensorRT engine cache are deleted first, since
the engine cache would otherwise serve the old weights. The factory reference is preds/op_cinque when all 479 rater frames exist there,
else the wrapper first runs the factory ONNX through the same harness as tag je-factory.
Outputs: $DATA_DIR/runs/junction_eyes/wod/<tag>/{rfs.npz, summary.json} (+ the jevdrive.run dir junction_eyes/wod-<tag>).

  # box, one pool job (op-train env; serving runs in the openpilot env as a subprocess; ~10 min incl. the TensorRT build):
  python -m jevdrive.cl submit --name je-wod-<tag> --vram 10 --cpu 16 --ram 48 -- \\
      $DATA_DIR/envs/op-train/bin/python experiments/junction_eyes/scripts/je_wod.py --onnx <merged.onnx> --tag <tag>
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "experiments/op_parity/scripts")]
FACTORY = Path.home() / "data/models/openpilot/cinque.ort.onnx"
FACTORY_PREDS = "op_cinque"                 # the harness's own factory run (decision 183: 8.005)
GATE_LO = -0.20


def pred_dir(tag):
    from jevdrive import wod_zeroshot as Z
    return Z.root("preds", f"op_cinque_je-{tag}")


def complete(d, names):
    return d.is_dir() and all((d / f"{n}.npz").exists() for n in names)


def serve(onnx, tag, names, a, run):
    """Run the harness for `onnx` as tag je-<tag> unless its predictions for this exact ONNX are complete."""
    from jevdrive import cache
    from jevdrive.common import data_dir
    d = pred_dir(tag)
    side = d / "je_onnx.json"
    s = cache.file_sig(onnx, content=True)
    old = json.loads(side.read_text()) if side.exists() else None
    if old is not None and old.get("sha256") != s["sha256"] or old is None and d.exists():
        trt = data_dir() / "runs" / "op_interp" / "trt_cache" / f"cinque-je-{tag}-trt"
        run.log.warning("%s holds predictions of another ONNX (%s): deleting it and %s", d, (old or {}).get("sha256"), trt)
        shutil.rmtree(d, ignore_errors=True)
        shutil.rmtree(trt, ignore_errors=True)
    if complete(d, names):
        run.info("%s: complete for this ONNX, not re-run", d)
        return d, 0.0
    d.mkdir(parents=True, exist_ok=True)
    side.write_text(json.dumps(s))
    py = data_dir() / "envs" / "openpilot" / "bin" / "python"
    cmd = [str(py), str(REPO / "scripts" / "wod_zeroshot_openpilot.py"), "--set", "rater", "--workers", str(a.workers), "--onnx", str(onnx),
           "--tag", f"je-{tag}"]
    run.info("+ %s", " ".join(cmd))
    t = time.time()
    subprocess.run(cmd, check=True, cwd=REPO)
    assert complete(d, names), f"{d}: harness finished without all {len(names)} rater predictions"
    return d, time.time() - t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True, help="merged Cinque ONNX (never modified)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workers", type=int, default=12, help="harness decode workers (its default; more were OOM-killed, op_guard line_wod)")
    a = ap.parse_args()
    from jevdrive.common import data_dir
    from jevdrive.run import Run
    from wod_launch_report import Ctx, f3
    onnx = Path(a.onnx).expanduser().resolve()
    assert onnx.exists(), onnx
    with Run("junction_eyes", f"wod-{a.tag}", config=vars(a)) as run:
        C = Ctx()
        names = C.names[: C.n]
        from jevdrive import wod_zeroshot as Z
        ref = Z.root("preds", FACTORY_PREDS)
        ref_src = f"preds/{FACTORY_PREDS}"
        if not complete(ref, names):
            run.log.warning("%s lacks rater frames: running the factory ONNX as tag je-factory", ref)
            ref, _ = serve(FACTORY.resolve(), "factory", names, a, run)
            ref_src = "preds/op_cinque_je-factory"
        d, wall = serve(onnx, a.tag, names, a, run)
        load = lambda p: np.stack([np.load(p / f"{n}.npz")["wod"] for n in names]).astype(np.float64)[..., :2]  # noqa: E731
        P, R = load(d), load(ref)
        f_arm, f_ref = C.rfs(P), C.rfs(R)
        diff = C.ci(f_arm - f_ref)
        ade = lambda p: np.linalg.norm(p[:, :12] - C.fut[: C.n, :12], axis=-1).mean(1)  # noqa: E731
        dade = C.ci_mean(ade(P) - ade(R))
        out = data_dir() / "runs" / "junction_eyes" / "wod" / a.tag
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "rfs.npz", names=names, rfs=f_arm, rfs_ref=f_ref, ade3=ade(P), ade3_ref=ade(R))
        summ = dict(tag=a.tag, onnx=str(onnx), preds=str(d), ref=ref_src, frames=int(C.n), RFS=C.cm(f_arm), RFS_ref=C.cm(f_ref),
                    dRFS=dict(mean=diff[0], lo=diff[1], hi=diff[2]), dADE3=dict(mean=dade[0], lo=dade[1], hi=dade[2]),
                    max_abs_plan_diff_m=float(np.abs(P - R).max()), identical_frames=int((np.abs(P - R).max((1, 2)) == 0).sum()),
                    gate_G2c=dict(line=f"dRFS CI lower bound >= {GATE_LO}", passed=bool(diff[1] >= GATE_LO)), serving_s=round(wall, 1),
                    run_dir=str(run.dir))
        (out / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
        run.summary.update(RFS=summ["RFS"], RFS_ref=summ["RFS_ref"], dRFS=summ["dRFS"], gate=summ["gate_G2c"]["passed"])
        run.info("RFS %.3f vs factory %.3f: d %s (gate lo >= %.2f: %s); ADE@3s d %s m; plans max |diff| %.3g m, %d / %d frames identical",
                 summ["RFS"], summ["RFS_ref"], f3(diff), GATE_LO, summ["gate_G2c"]["passed"], f3(dade), summ["max_abs_plan_diff_m"],
                 summ["identical_frames"], C.n)
        print(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    main()
