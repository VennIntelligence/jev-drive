"""op_parity wod-slot check (plans/2026-10-08-wod-slot-prereg.md, results/wod_slot.md): does the 8 real + 1 zero policy-slot training protocol of the
navtrain arms (P2H10, SH30) explain part of their WOD val loss, where the harness serves 9 real slots?

  onnx    (op-train env, CPU) serving ONNX -> the same ONNX with the oldest of the policy's 9 slots zeroed. The policy reads the last 9 of the 33 queue
          slots `cat_3` (GatherND indices -9 .. -1, i.e. 24 .. 32); a Mul with a (1, 33, 1, 1) mask is inserted after the Concat, so the zero is applied
          AFTER the intent bias was added to the past slots (pp_train.PModel.forward: H + bias, then H * valid) and the recurrent queue is untouched.
          --keep: mask of ones (a build-path check: must reproduce the unmasked ONNX bit for bit).
  check   (openpilot env, GPU) unmasked vs masked ONNX on a few WOD targets with the harness stepping: plan shift, and ones-mask identity.
  report  (jevdrive env, CPU) RFS / ADE / strata and the paired contrasts on WOD val -> results/wod_slot.{md} and results/wod_slot/*.csv

Served tags: `<tag>_s8` = oldest slot zeroed (prediction dir preds/op_cinque_<tag>_s8), stored 9-real runs are `<tag>` / `shipped`.
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_pl.Path(__file__).parent)]
import argparse, json  # noqa: E401,E402

import numpy as np  # noqa: E402

OUT = _R / "experiments/op_parity/results/wod_slot"
OLDEST = 24                                                       # oldest of the 9 policy slots inside the 33-slot queue


def cmd_onnx(a):
    import onnx
    from onnx import helper, numpy_helper
    m = onnx.load(a.src)
    g = m.graph
    nodes = list(g.node)
    ci = [i for i, n in enumerate(nodes) if list(n.output) == ["cat_3"]]
    assert len(ci) == 1 and nodes[ci[0]].op_type == "Concat" and nodes[ci[0]].attribute[0].i == 1
    assert sum("cat_3" in n.input for n in nodes) == 1                        # only the policy's slot gather reads it
    mask = np.ones((1, 33, 1, 1), np.float16)
    if not a.keep:
        mask[:, OLDEST] = 0
    nodes[ci[0]].output[0] = "cat_3_pre"
    g.initializer.append(numpy_helper.from_array(mask, "ws_mask"))
    nodes.insert(ci[0] + 1, helper.make_node("Mul", ["cat_3_pre", "ws_mask"], ["cat_3"]))
    del g.node[:]
    g.node.extend(nodes)
    onnx.save(m, a.out)
    print("wrote", a.out, "mask zero at" if not a.keep else "ones mask", "" if a.keep else OLDEST)


def cmd_check(a):
    from jevdrive import wod_zeroshot as Z
    from jevdrive.openpilot.model import T_IDXS, OPModel, decode
    import wod_zeroshot_openpilot as H
    from concurrent.futures import ProcessPoolExecutor
    sets = Z.load_sets()
    spans, _ = Z.load_spans()
    op_calib = json.loads((Z.root() / "op_calib.json").read_text())
    todo = sorted(str(n) for n in sets["rater"]["name"])[:: max(1, 479 // a.n)][: a.n]
    from jevdrive.common import data_dir
    shard_dir = data_dir() / "datasets" / "waymo_e2e" / "front3"
    ms = [OPModel(p, "trt", cache=data_dir() / "runs" / "op_interp" / "trt_cache" / f"cinque-{pathlib_stem(p)}-chk") for p in a.onnx]
    res = []
    with ProcessPoolExecutor(4, initializer=H._init, initargs=(spans, op_calib, str(shard_dir), None)) as ex:
        for name, names, frames in ex.map(H.model_frames, todo):
            seq = name.rsplit("-", 1)[0]
            dev = np.array(op_calib[seq]["1"]["extrinsic"]).reshape(4, 4)[:2, 3]
            res.append([H.run_one(m, name, names, frames, dev, T_IDXS, decode, None)[0][0] for m in ms])
    r = np.array(res)                                                         # (n, models, 20, 3)
    out = dict(onnx=a.onnx, n=len(todo))
    for j in range(1, len(ms)):
        d = np.linalg.norm(r[:, j, :, :2] - r[:, 0, :, :2], axis=-1)
        out[f"plan_shift_m_{j}_vs_0"] = dict(mean=float(d.mean()), mean_at_5s=float(d[:, -1].mean()), max=float(d.max()))
    print(json.dumps(out, indent=1))
    if a.json:
        pathlib_write(a.json, out)


def pathlib_stem(p):
    return _pl.Path(p).stem


def pathlib_write(p, d):
    _pl.Path(p).parent.mkdir(parents=True, exist_ok=True)
    _pl.Path(p).write_text(json.dumps(d, indent=1))


# ---------------------------------------------------------------- report
def cmd_report(a):
    from mixed_domain import Wod, groups
    from jevdrive import stats
    g = groups(a.arms)
    W = Wod()
    W.load(g)
    C = W.C
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for nm, m in W.st.items():
        for k in g:
            rows.append({"stratum": nm, "n": int(m.sum()), "arm": k, "RFS": C.cm(W.rfs(k), m), "d5 mean (m)": float(W.d(k, 19)[m].mean()),
                         "seeds": "/".join(f"{C.cm(s, m):.3f}" for s in W.sco[k]) if len(W.sco[k]) > 1 else ""})
    stats.write_table(rows, OUT / "arms")
    ct = []
    for pair in a.pairs:
        x, y = pair.split(":")
        for nm, m in W.st.items():
            p, lo, hi = C.ci(W.rfs(x) - W.rfs(y), m)
            ct.append({"stratum": nm, "n": int(m.sum()), "contrast": f"{x} - {y}", "dRFS": p, "lo": lo, "hi": hi,
                       "seed_dRFS": "/".join(f"{C.cm(sx - sy, m):+.3f}" for sx, sy in zip(W.sco[x], W.sco[y])) if len(W.sco[x]) == len(W.sco[y]) > 1 else ""})
    stats.write_table(ct, OUT / "contrasts")
    ade = W.ade(list(g), [])
    stats.write_table(ade, OUT / "ade")
    ac = []
    err = {k: np.mean([np.linalg.norm(p - C.fut, axis=-1) for p in W.P[k]], 0) for k in g}
    for pair in a.pairs:
        x, y = pair.split(":")
        for nm, sl in (("ADE@3s", slice(0, 12)), ("ADE@5s", slice(0, 20))):
            dd = err[x][:, sl].mean(1) - err[y][:, sl].mean(1)
            b = (C.Ka @ dd) / C.Ka.sum(1)
            ac.append({"metric": nm, "contrast": f"{x} - {y}", "d (m)": float(dd.mean()), "lo": float(np.percentile(b, 2.5)), "hi": float(np.percentile(b, 97.5))})
    stats.write_table(ac, OUT / "ade_contrasts")
    # plan shift between the two protocols, per arm pair (same checkpoint): mean waypoint distance over the rater frames
    sh = []
    for pair in a.shift:
        x, y = pair.split(":")
        d = np.mean([np.linalg.norm(px[: C.n] - py[: C.n], axis=-1) for px, py in zip(W.P[x], W.P[y])], 0)
        sh.append({"pair": pair, "mean shift over 20 waypoints (m)": float(d.mean()), "at 5 s (m)": float(d[:, -1].mean())})
    stats.write_table(sh, OUT / "plan_shift")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    o = sp.add_parser("onnx")
    o.add_argument("--src", required=True)
    o.add_argument("--out", required=True)
    o.add_argument("--keep", action="store_true")
    c = sp.add_parser("check")
    c.add_argument("--onnx", nargs="+", required=True, help="first = reference, the others are compared with it")
    c.add_argument("-n", type=int, default=8)
    c.add_argument("--json", default="")
    r = sp.add_parser("report")
    r.add_argument("--arms", nargs="+", required=True)
    r.add_argument("--pairs", nargs="+", required=True)
    r.add_argument("--shift", nargs="*", default=[])
    a = ap.parse_args()
    {"onnx": cmd_onnx, "check": cmd_check, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
