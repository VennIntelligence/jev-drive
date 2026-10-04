#!/usr/bin/env python
"""Route-choice fine-tuned openpilot as a serving ONNX + adapter for the closed-loop server (results/harness.md).

A checkpoint of the route fine-tune ($DATA_DIR/runs/op_route_ft/runs/<arm>-s0/ckpt-final.pt, op_adapt_l's format: {"model": {"net": {key:
fp32}, "adapter": None}, "cfg": ...}) plus its `adapter.npz` (lib/route_adapter.RouteAdapter.to_npz) becomes
  <out>.onnx           cinque.ort.onnx with the trained initializers copied in and the `intent_bias` (1, 32, 512) fp16 input
                       (experiments/op_adapt_l/scripts/op_l_onnx.py build --bias-input: the bias is added to the hidden tokens of the
                       current frame and of the 8 past policy slots, i.e. all 9 context frames, as in training)
  <out>.adapter.npz    a copy of the adapter; the server (op_arb_server.py) evaluates it per request with lib/route_adapter.NumpyAdapter
                       on the route polyline the agent sends. `--adapter none` (arm rc-ctl: its adapter only ever saw "no command")
                       writes no adapter: the input stays zero, the unconditioned fine-tuned model.

  build  (op-train env)    route_onnx.py build --ckpt CKPT --adapter ADAPTER.npz|none --out $DATA_DIR/runs/op_route_ft/onnx/<arm>-s0.onnx
  synth  (op-train env)    route_onnx.py synth --out DIR          synthetic checkpoint (shipped weights, 3 tensors x 1.01) + bear / poly
                                                                   adapters with random out layers (std 0.01), for the equivalence check
  ref    (op-train env)    route_onnx.py ref --ckpt CKPT --adapter A.npz --out ref.npz     port (LModel + torch RouteAdapter) on the WOD
                                                                   reference frame streams, route command cycled per 3 frames
  check  (openpilot env)   route_onnx.py check --onnx X.onnx --ref ref.npz                onnxruntime + NumpyAdapter vs the port
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "lib"), str(REPO / "experiments/op_adapt_l/scripts")]
import route_adapter as RA  # noqa: E402

REF_DIR = Path.home() / "data/runs/op_adapt/ref"


def adapter_path(onnx_path):
    """The adapter served with an ONNX: <stem>.adapter.npz next to it (None if absent)."""
    p = Path(str(onnx_path)[: -len(".onnx")] + ".adapter.npz")
    return p if p.is_file() else None


# ---------------------------------------------------------------- commands of the check streams
def l_turn(d, R, sign, n_tail=300):
    s = np.arange(0, d, 0.5)
    th = np.linspace(0, np.pi / 2, 60)
    return np.vstack([np.stack([s, 0 * s], -1), np.stack([d + R * np.sin(th), sign * R * (1 - np.cos(th))], -1),
                      np.stack([np.full(n_tail, d + R), sign * (R + np.arange(n_tail) * 0.5)], -1)])


def check_routes():
    """(name, poly, pmask) of the command cycle: none, straight, left in 40 m, right in 15 m."""
    st = RA.route_poly_from_path(np.stack([np.arange(0.5, 200, 0.5), np.zeros(399)], -1))
    out = [("none", None, None), ("straight",) + st]
    for name, d, sign in (("left40", 40.0, 1.0), ("right15", 15.0, -1.0)):
        out.append((name,) + RA.route_poly_from_path(l_turn(d, 10.0, sign)[1:]))
    return out


def feats(enc):
    return np.stack([np.zeros(RA.ENC_DIM[enc], np.float32) if p is None else RA.features(enc, p, m) for _, p, m in check_routes()])


# ---------------------------------------------------------------- build
def build(a):
    import op_l_onnx as L
    out = Path(a.out)
    L.build(argparse.Namespace(ckpt=a.ckpt, out=str(out), no_adapter=True, bias_input=True))
    dst = Path(str(out)[: -len(".onnx")] + ".adapter.npz")
    if a.adapter != "none":
        RA.NumpyAdapter(a.adapter)                          # loads: a valid adapter file
        shutil.copyfile(a.adapter, dst)
        print("adapter", dst, "enc", RA.NumpyAdapter(dst).enc)
    elif dst.exists():
        dst.unlink()
        print("removed stale", dst)


def synth(a):
    import torch
    from jevdrive import op_adapt as A
    from experiments.op_adapt_l.lib import op_adapt_l as OL
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    net = A.load("cinque", torch.float32)
    names = A.stage4_weights()[:2] + OL.pol_weights()[:1]
    st = {k: net.params[k].detach().float().cpu() * 1.01 for k in (L_key(n) for n in names)}
    torch.save({"model": {"net": st, "adapter": None}, "cfg": {"name": "synth"}}, out / "ckpt-final.pt")
    g = torch.Generator().manual_seed(0)
    for enc in ("bear", "poly"):
        m = RA.RouteAdapter(enc)
        with torch.no_grad():
            m.out.weight.copy_(torch.randn(m.out.weight.shape, generator=g) * a.std)
            m.out.bias.copy_(torch.randn(m.out.bias.shape, generator=g) * a.std)
        m.to_npz(out / f"adapter-{enc}.npz")
    print("wrote", out, "tensors", list(st))


def L_key(name):
    return name.replace(".", "__").replace("/", "_")


# ---------------------------------------------------------------- equivalence
def torch_adapter(path, dev):
    import torch
    z = np.load(path)
    m = RA.RouteAdapter(str(z["enc"]))
    m.load_state_dict({k: torch.from_numpy(z[k.replace(".", "_")]) for k in m.state_dict()})
    return m.to(dev).eval()


def ref(a):
    import torch
    from jevdrive import op_adapt as A
    from experiments.op_adapt_l.lib import op_adapt_l as OL
    dev = torch.device("cuda")
    model = OL.LModel(OL.LCfg("ref", s4=False, pol=False, intent="none")).to(dev).eval()
    if a.ckpt != "none":
        st = torch.load(a.ckpt, map_location="cpu", weights_only=False)["model"]
        for k, v in st["net"].items():
            model.net.params[k].data.copy_(v)
    ad = torch_adapter(a.adapter, dev)
    F = torch.from_numpy(feats(ad.enc)).to(dev)
    with torch.no_grad():
        Btab = ad(F)                                        # (n_cmd, 32, 512)

    class Bias(torch.nn.Module):                            # LModel.policy calls adapter(H, intent): here intent = command index
        def forward(self, H, idx):
            return H + Btab[idx][:, None].to(H.dtype)
    model.adapter = Bias()
    res = {"enc": np.array(ad.enc), "feats": F.cpu().numpy(), "bias": Btab.float().cpu().numpy()}
    for k in (0, 1):
        frames = np.load(REF_DIR / f"frames_{k}.npz")["frames"]
        n = len(frames)
        cmd = (np.arange(n) // 3) % len(F)
        fr = torch.as_tensor(frames).to(dev)
        prev = torch.cat([torch.zeros_like(fr[:2]), fr[:-2]])
        with torch.no_grad():
            H = torch.cat([model.net.run_batched(A.vision_feeds(prev[i:i + 32], fr[i:i + 32]), ["view_39"])["view_39"][:, 0]
                           for i in range(0, n, 32)])
            Hp = torch.cat([torch.zeros(2 * (A.CONTEXT - 1), *A.H_SHAPE, dtype=H.dtype, device=dev), H])
            idx = torch.arange(n, device=dev)[:, None] + torch.arange(0, 2 * A.CONTEXT, 2, device=dev)[None]
            valid = (idx >= 2 * (A.CONTEXT - 1)).to(dev)
            tc = torch.tensor([[1.0, 0.0]], device=dev).expand(32, 2)

            def run(c):
                return np.concatenate([model.policy(Hp[idx[i:i + 32]], valid[i:i + 32], tc[:min(32, n - i)], torch.from_numpy(c[i:i + 32]).to(dev))
                                       ["outputs"].float().cpu().numpy() for i in range(0, n, 32)])
            res[f"out_{k}"], res[f"cmd_{k}"] = run(cmd), cmd
            zero = run(np.zeros_like(cmd))                     # command "none" everywhere: the bias' own effect on the plan
        ok = np.arange(n) >= 2 * 8
        pg = (res[f"out_{k}"][ok, :495] - zero[ok, :495]).reshape(-1, 33, 15)[:, :, :2]
        print("stream %d: plan xy change by the command (port, vs no command) max %.3f m, mean %.4f m" %
              (k, np.abs(pg).max(), np.linalg.norm(pg, axis=-1).mean()))
    np.savez(a.out, **res)
    print("wrote", a.out, "bias |max| %.4f" % np.abs(res["bias"]).max())


def check(a):
    from jevdrive.openpilot.model import OPModel
    z = np.load(a.ref)
    adp = adapter_path(a.onnx)
    na = RA.NumpyAdapter(adp)
    assert na.enc == str(z["enc"]), (na.enc, z["enc"])
    B = np.concatenate([na.bias(f) for f in z["feats"]])     # the server's bias per command
    db = float(np.abs(B.astype(np.float32) - z["bias"]).max())
    print("numpy adapter vs torch adapter: bias max diff %.2e (fp16 cast)" % db)
    m = OPModel(str(a.onnx), a.backend)
    sl = m.slices
    cols = np.concatenate([np.arange(sl[k].start, sl[k].stop) for k in ("plan", "lead", "lead_prob", "desire_state", "meta")])
    rows = []
    for k in (0, 1):
        frames = np.load(REF_DIR / f"frames_{k}.npz")["frames"]
        cmd = z[f"cmd_{k}"]
        res = {}
        for tag, bias_of in (("served", lambda f: B[int(cmd[f])][None]), ("bias dropped", lambda f: np.zeros((1, 32, 512), np.float16))):
            m.reset()
            got = []
            for f, fr in enumerate(frames):
                m.extra = {"intent_bias": bias_of(f)}
                m.step(fr)
                got.append(m.step(fr))
            res[tag] = np.stack(got)[:, cols]
        ref_ = z[f"out_{k}"][:, cols]
        ok = np.arange(len(frames)) >= 2 * 8
        for tag, got in res.items():                         # "bias dropped": what the check would see if the bias were lost
            d = np.abs(got[ok] - ref_[ok])
            pg = got[ok][:, :495].reshape(-1, 33, 15)[:, :, :2] - ref_[ok][:, :495].reshape(-1, 33, 15)[:, :, :2]
            r = (k, tag, int(ok.sum()), float(d.max()), float(np.percentile(d, 99)), float(np.abs(pg).max()), float(np.linalg.norm(pg, axis=-1).mean()))
            rows.append(r)
            print("stream %d %-12s frames %d | all cols max %.4f p99 %.5f | plan xy max %.4f m, mean dist %.5f m" % r)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--ckpt", required=True)
    b.add_argument("--adapter", required=True, help="adapter.npz, or none (zero bias)")
    b.add_argument("--out", required=True)
    s = sp.add_parser("synth")
    s.add_argument("--out", required=True)
    s.add_argument("--std", type=float, default=0.01, help="std of the adapter's random out layer")
    r = sp.add_parser("ref")
    r.add_argument("--ckpt", required=True)
    r.add_argument("--adapter", required=True)
    r.add_argument("--out", required=True)
    c = sp.add_parser("check")
    c.add_argument("--onnx", required=True)
    c.add_argument("--ref", required=True)
    c.add_argument("--backend", default="cuda-iob")
    a = ap.parse_args()
    {"build": build, "synth": synth, "ref": ref, "check": check}[a.cmd](a)
