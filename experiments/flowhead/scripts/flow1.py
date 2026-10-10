"""FLOW1 (experiments/flowhead/plans/2026-10-10-flow-head-prereg.md): checks and reads of the flow-matching / regression trajectory heads.

  ident    --a TAG --b TAG      every tensor of two pp_train checkpoints equal (exit 1 if not): the default path of the trainer is unchanged
  smoke    FILE ...             prediction files of the smoke heads written by the bench `poses` stage: tokens of navtest, finite (8, 3) poses
  trained  TAG ...              clean DONE, finite losses, head dev ADE <= 1.2 m, thead.pt present
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib")]
import argparse, glob, json  # noqa: E401,E402

import numpy as np  # noqa: E402

from jevdrive.common import data_dir  # noqa: E402

PR = data_dir() / "runs" / "op_parity"
ADE_MAX = 1.2


def cmd_ident(a):
    import torch
    A, B = (torch.load(PR / "runs" / t / "ckpt-final.pt", map_location="cpu", weights_only=False)["model"] for t in (a.a, a.b))
    diff = {}
    for grp in ("net", "parity"):
        assert A[grp].keys() == B[grp].keys(), f"{grp}: different tensors"
        for k in A[grp]:
            d = float((A[grp][k].float() - B[grp][k].float()).abs().max())
            if d > 0:
                diff[f"{grp}.{k}"] = d
    res = dict(a=a.a, b=a.b, tensors=sum(len(A[g]) for g in ("net", "parity")), differing=len(diff), max_abs=max(diff.values(), default=0.0))
    print(json.dumps(res))
    if a.out:
        _pl.Path(a.out).write_text(json.dumps(res, indent=1))
    raise SystemExit(1 if diff else 0)


def cmd_smoke(a):
    names = np.load(PR / "cache/lb_navtest/tab.npz")["names"].astype(str)
    for f in a.files:
        z = np.load(f)
        assert z["tokens"].astype(str).tolist() == names.tolist() and z["poses"].shape == (len(names), 8, 3) and np.isfinite(z["poses"]).all(), f
        print(f"{f}: {z['poses'].shape} finite, tokens = navtest")


def cmd_trained(a):
    bad = []
    for t in a.tags:
        d = sorted(glob.glob(str(PR / f"train-{t}" / "*")))[-1]
        done = json.loads((_pl.Path(d) / "DONE").read_text())
        ok = (PR / "runs" / t / "thead.pt").exists() and done.get("steps") == a.steps and np.isfinite(done["dev_th_ade"]) and done["dev_th_ade"] <= ADE_MAX
        print(json.dumps(dict(tag=t, ok=bool(ok), dev_th_ade=done.get("dev_th_ade"), steps=done.get("steps"), train_s=done.get("train_s"))))
        bad += [] if ok else [t]
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("ident")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--out", default="")
    p = sp.add_parser("smoke")
    p.add_argument("files", nargs="+")
    p = sp.add_parser("trained")
    p.add_argument("tags", nargs="+")
    p.add_argument("--steps", type=int, default=10000)
    a = ap.parse_args()
    {"ident": cmd_ident, "smoke": cmd_smoke, "trained": cmd_trained}[a.cmd](a)
