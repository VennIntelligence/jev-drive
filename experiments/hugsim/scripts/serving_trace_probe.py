#!/usr/bin/env python
"""Which queue entries the served queued ONNX reads (experiments/hugsim/results/serving_trace.md, plans/2026-10-11-serving-trace-prereg.md).

The fine-tuned HUGSIM arm is one ONNX whose temporal queues are graph states: state_img_q (2, 5, 6, 128, 256) and state_feat_q
(128, 1, 16384), shifted by one entry per 20 Hz step. This probe perturbs one state entry at a time before a single step and records
whether the plan output moves, which gives the entries the vision part and the policy read; together with the per-decision queue CRCs
of the agent opt `trace` it maps every policy slot to the rendered frames behind it. It also checks the queue shift, and that reading the
queues (hugsim_zs_server.queue_trace) leaves the outputs of the serving backend bit-identical.

  $DATA_DIR/envs/openpilot/bin/python experiments/hugsim/scripts/serving_trace_probe.py --onnx $DATA_DIR/runs/bench/hugsim/_onnx/pp-P2H10-F-s0.onnx
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "scripts"), str(_R / "experiments/hugsim/archive")]
import argparse, json, zlib  # noqa: E401,E402

import numpy as np  # noqa: E402

from jevdrive.openpilot.model import OPModel  # noqa: E402
from jevdrive.run import Run, cli_args  # noqa: E402

FRAME = (2, 6, 128, 256)


def frames(rng, n):
    """n smooth random frames (low-frequency noise, uint8): distinct bytes, inside the image range."""
    low = rng.integers(0, 256, (n, 2, 6, 8, 16)).astype(np.float32)
    return np.clip(np.kron(low, np.ones((16, 16), np.float32)) + rng.normal(0, 8, (n,) + FRAME), 0, 255).astype(np.uint8)


def crc(a):
    return zlib.crc32(np.ascontiguousarray(a).tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--warm", type=int, default=140, help="20 Hz steps before the probe (> 128 fills the feature queue)")
    ap.add_argument("--serve-backend", default="trt", help="backend of the read-only check (the HUGSIM server's)")
    cli_args(ap)
    a = ap.parse_args()
    with Run("hugsim_serving_trace", "probe", seed=a.seed, config=vars(a), resume=a.resume) as run:
        rng = np.random.default_rng(0)
        F = frames(rng, a.warm // 4 + 8)
        m = OPModel(a.onnx, "cuda")
        sl = m.slices["plan"]
        bias = rng.normal(0, 1, m.inputs["intent_bias"][0]).astype(np.float16) if "intent_bias" in m.inputs else None
        if bias is not None:
            m.extra = {"intent_bias": bias}
        for s in range(a.warm):                                    # each frame held for 4 steps, as the HUGSIM agent feeds it
            m.step(F[s // 4])
        S = m.snapshot()
        new = F[a.warm // 4]

        def one(edit=None, img=new, extra=None):
            m.restore(S)
            if edit:
                edit(m.state)
            if extra is not None:
                m.extra = extra
            out = m.step(img)
            m.extra = {} if bias is None else {"intent_bias": bias}
            return out, {k: np.copy(v) for k, v in m.state.items()}

        out0, st0 = one()
        res = {"onnx": a.onnx, "states": {k: list(v.shape) for k, v in S["state"].items()}}
        iq, fq = S["state"]["state_img_q"], S["state"]["state_feat_q"]
        res["shift"] = {"img_q": bool(np.array_equal(st0["state_img_q"][:, :-1], iq[:, 1:]) and np.array_equal(st0["state_img_q"][:, -1], new)),
                        "feat_q": bool(np.array_equal(st0["state_feat_q"][:-1], fq[1:]))}
        rep, _ = one()
        res["repeat_max_abs"] = float(np.abs(rep - out0).max())    # determinism of the probe backend
        alt = frames(np.random.default_rng(1), 1)[0]

        def d(out, st):
            return {"plan_max_abs": float(np.abs(out[sl] - out0[sl]).max()), "out_max_abs": float(np.abs(out - out0).max()),
                    "new_feat_row_max_abs": float(np.abs(st["state_feat_q"][-1].astype(np.float32) - st0["state_feat_q"][-1].astype(np.float32)).max())}

        res["new_img"] = d(*one(img=alt))
        res["img_q"] = []
        for i in run.tqdm(range(iq.shape[1]), desc="img_q"):
            def ed(st, i=i):
                st["state_img_q"] = st["state_img_q"].copy()
                st["state_img_q"][:, i] = alt
            res["img_q"].append(dict(entry=i, steps_back=iq.shape[1] - i, **d(*one(ed))))
        res["feat_q"] = []
        perm = np.random.default_rng(2).permutation(fq.shape[-1])
        for r in run.tqdm(range(fq.shape[0]), desc="feat_q"):
            def ed(st, r=r):
                st["state_feat_q"] = st["state_feat_q"].copy()
                st["state_feat_q"][r] = st["state_feat_q"][r][..., perm]      # the row's own values, shuffled: same scale
            res["feat_q"].append(dict(row=r, steps_back=fq.shape[0] - r, **d(*one(ed))))
        if bias is not None:                                       # is the bias inside the stored feature row?
            res["zero_bias"] = d(*one(extra={"intent_bias": np.zeros_like(bias)}))
        thr = 10 * max(res["repeat_max_abs"], 1e-6)
        res["read_img_steps_back"] = [x["steps_back"] for x in res["img_q"] if x["out_max_abs"] > thr]
        res["read_feat_steps_back"] = [x["steps_back"] for x in res["feat_q"] if x["out_max_abs"] > thr]
        run.info("img_q entries read (20 Hz steps before the new frame): %s", res["read_img_steps_back"])
        run.info("feat_q rows read (20 Hz steps before the new row): %s", res["read_feat_steps_back"])
        del m

        # read-only check on the serving backend: the same stream with and without queue_trace after every step
        import hugsim_zs_server as HS
        ms = OPModel(a.onnx, a.serve_backend)
        runs = []
        for traced in (False, True, False):
            ms.reset()
            if bias is not None:
                ms.extra = {"intent_bias": bias}
            outs, ok = [], True
            for s in range(48):
                outs.append(ms.step(F[s // 4]))
                if traced:
                    t = HS.queue_trace(ms, full=True)
                    want = [crc(F[max(0, s - k) // 4]) if s - k >= 0 else crc(np.zeros(FRAME, np.uint8)) for k in range(4, -1, -1)]
                    ok &= t["img"] == want and t["n"] == s + 1
            runs.append(np.stack(outs))
            if traced:
                res["trace_img_crc_matches_fed"] = bool(ok)
                res["trace_last"] = {"n": t["n"], "distinct_feat_rows": len(set(t["feat_q"]))}
        res["serve_backend"] = a.serve_backend
        res["traced_vs_untraced_max_abs"] = float(np.abs(runs[1] - runs[0]).max())
        res["untraced_vs_untraced_max_abs"] = float(np.abs(runs[2] - runs[0]).max())
        run.path("probe.json").write_text(json.dumps(res, indent=1))
        run.summary.update({k: res[k] for k in ("shift", "read_img_steps_back", "read_feat_steps_back", "trace_img_crc_matches_fed",
                                                "traced_vs_untraced_max_abs", "untraced_vs_untraced_max_abs")})
        run.info("%s", json.dumps(run.summary))


if __name__ == "__main__":
    main()
