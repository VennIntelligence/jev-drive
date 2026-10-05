"""op_parity arms in the HUGSIM closed-loop harness of experiments/hugsim/results/wajepa_ref.md (results/hugsim_harness.md).

An arm = shipped Cinque with (a) its fine-tuned plan-pathway initializers and (b) lib/parity_adapter's (32, 512) bias on the hidden tokens
of all 9 policy context frames. Served as
  ONNX         cinque.ort.onnx with the arm's trained initializers and an `intent_bias` (1, 32, 512) fp16 input added to the current
               frame's tokens and to the 8 past policy slots (experiments/op_adapt_l/scripts/op_l_onnx.py build --bias-input), run by
               the HUGSIM policy server (experiments/hugsim/archive/hugsim_zs_server.py cinque --onnx, TensorRT like every Cinque run)
  bias server  `serve` below (op-train env, torch): per simulator step the agent (lib/parity_hugsim.py, zs_agent.py opt `parity`) sends
               ego features (+ the side / rear key frames for an arm that reads them); the arm's ParityAdapter returns the bias; side
               tokens come from the arm's own frozen Cinque vision encoder (port, fp16), the encoder pp_prep cached for training
Tags: P0 / P*-init = shipped weights (P0 has no adapter, P*-init an untrained one: bias exactly 0), a pp_train run tag (P2-s0), or a
checkpoint path (.pt).

  onnx   (op-train)   pp_hugsim.py onnx --tag P2-s0 --out X.onnx
  serve  (op-train)   pp_hugsim.py serve --tag P2-s0 --socket S [--ready-file F]
  synth  (op-train)   pp_hugsim.py synth --out DIR          arm-P3 checkpoint: 3 plan-pathway tensors x 1.01, adapter out layer N(0, std)
  ref    (op-train)   pp_hugsim.py ref --tag T --out ref.npz  torch port (pp_train.PModel weights + adapter) on the WOD reference streams
                                                             with synthetic per-frame ego / side inputs; side tokens via pp_prep.encoder
  check  (openpilot)  pp_hugsim.py check --onnx X --ref ref.npz --socket S [--base cinque] [--json out.json]
                      onnxruntime + the running bias server vs the port; --base: the same served model with zero bias vs a stock ONNX
  signs  (any)        pp_hugsim.py signs --runs DIR... --navsim TAB.npz --json out.json   sign conventions of the HUGSIM ego features vs NAVSIM
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_R / "experiments/op_adapt_l/scripts"),
                 str(_pl.Path(__file__).resolve().parent)]
import argparse, json, socket, threading, time  # noqa: E401,E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import parity_adapter as PA  # noqa: E402
import zeroshot_wire as wire  # noqa: E402

REF_DIR = Path.home() / "data/runs/op_adapt/ref"
N_REF = 150


# ---------------------------------------------------------------- models (op-train env)
def is_shipped(tag: str) -> bool:
    return tag == "P0" or tag.endswith("-init")


def pmodel(tag: str, dev):
    import torch
    import pp_train as T
    if tag.endswith(".pt"):
        ck = torch.load(tag, map_location="cpu", weights_only=False)
        m = T.PModel(ck["model"]["arm"]).to(dev).eval()
        m.load_state(ck["model"])
        return m
    return T.load_pmodel(tag, dev)


def ckpt_path(tag: str):
    import pp_train as T
    if tag.endswith(".pt"):
        return Path(tag)
    return None if is_shipped(tag) else T.proot("runs", tag) / "ckpt-final.pt"


def encode_side(net, keys):
    """keys (B, 3, 4, 2, 6, 128, 256) uint8 key frames -> (B, 3, 3, 32, 512) tokens of the pairs (key k - 1, key k), as pp_prep."""
    from jevdrive import op_adapt as A
    B, C = keys.shape[:2]
    prev, cur = keys[:, :, :-1].reshape(-1, *keys.shape[3:]), keys[:, :, 1:].reshape(-1, *keys.shape[3:])
    return net.run_batched(A.vision_feeds(prev, cur), ["view_39"])["view_39"].reshape(B, C, PA.SIDE_T, *A.H_SHAPE)


def onnx_cmd(a):
    import torch
    import op_l_onnx as L
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    src = ckpt_path(a.tag)
    if src is None:                                          # shipped weights: no trained tensors, only the bias input
        src = out.with_suffix(".shipped.pt")
        torch.save({"model": {"net": {}, "adapter": None}, "cfg": {"name": "shipped"}}, src)
    L.build(argparse.Namespace(ckpt=str(src), out=str(out), no_adapter=True, bias_input=True))


# ---------------------------------------------------------------- bias server
def serve(a):
    import torch
    dev = torch.device("cuda")
    m = pmodel(a.tag, dev)
    ad = m.adapter
    info = {"tag": a.tag, "arm": m.arm, "use_ego": bool(ad is not None and ad.use_ego), "use_side": bool(ad is not None and ad.use_side),
            "adapter_params": int(sum(p.numel() for p in ad.parameters())) if ad is not None else 0}
    gpu = threading.Lock()

    @torch.no_grad()
    def bias(arrays):
        if ad is None:
            return np.zeros(A_SHAPE, np.float16)
        ego = torch.from_numpy(np.asarray(arrays["ego"], np.float32).reshape(1, PA.EGO_DIM)).to(dev)
        side = encode_side(m.net, torch.from_numpy(np.ascontiguousarray(arrays["side"])).to(dev)[None]) if ad.use_side else None
        return ad(ego, side, None)[0].to(torch.float16).cpu().numpy()      # fp16, as adapter.apply casts it to H's dtype

    with gpu:                                               # warm-up with the real shapes
        bias({"ego": np.zeros(PA.EGO_DIM, np.float32), "side": np.zeros((3, 4, 2, 6, 128, 256), np.uint8)})

    def conn_loop(conn):
        try:
            while True:
                meta, arrays = wire.recv(conn)
                if meta["cmd"] == "reset":
                    wire.send(conn, {"ok": True, "server": info}, {})
                    continue
                t = time.perf_counter()
                with gpu:
                    b = bias(arrays)
                wire.send(conn, {"ms": 1e3 * (time.perf_counter() - t)}, {"bias": b})
        except ConnectionError:
            pass
        finally:
            conn.close()

    sp = Path(a.socket)
    sp.parent.mkdir(parents=True, exist_ok=True)
    if sp.exists():
        sp.unlink()
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(sp))
    srv.listen(64)
    print("parity bias server ready:", json.dumps(info), flush=True)
    if a.ready_file:
        Path(a.ready_file).write_text(json.dumps(info))
    while True:
        conn, _ = srv.accept()
        threading.Thread(target=conn_loop, args=(conn,), daemon=True).start()


A_SHAPE = (32, 512)


# ---------------------------------------------------------------- equivalence (synthetic inputs on the WOD reference streams)
def synth_ego(n: int, k: int) -> np.ndarray:
    """(n, 20) plausible, varying ego features: speed 0-12 m/s, a curving 4-pose history, the command cycled per 10 frames."""
    rng = np.random.default_rng([7, k])
    t = np.arange(n)
    v = 6 + 6 * np.sin(t / 17.0 + k)
    acc = 1.5 * np.cos(t / 11.0)
    kappa = 0.05 * np.sin(t / 23.0 + 2 * k)
    tau = np.array([-1.5, -1.0, -0.5, 0.0])
    s = v[:, None] * tau[None]
    yaw = kappa[:, None] * s
    x = np.where(np.abs(kappa[:, None]) > 1e-6, np.sin(yaw) / np.where(np.abs(kappa) > 1e-6, kappa, 1)[:, None], s)
    y = np.where(np.abs(kappa[:, None]) > 1e-6, (1 - np.cos(yaw)) / np.where(np.abs(kappa) > 1e-6, kappa, 1)[:, None], 0)
    pose = np.stack([x, y, yaw], -1) + rng.normal(0, 0.02, (n, 4, 3)) * (tau != 0)[None, :, None]
    vel = np.stack([v, rng.normal(0, 0.1, n)], -1)[:, None].repeat(4, 1)
    ac = np.stack([acc, v ** 2 * kappa], -1)[:, None].repeat(4, 1)
    cmd = np.zeros((n, 4), np.float32)
    c = (t // 10) % 4
    cmd[c < 3, c[c < 3]] = 1.0
    return PA.ego_features(pose, vel, ac, cmd)


def synth_side(frames: np.ndarray, f: int) -> np.ndarray:
    """(3, 4, 2, 6, 128, 256) uint8 side key frames of frame f: cam c = the stream at keys f-6, f-4, f-2, f (clamped at 0), its planes
    rolled sideways by 37 c + 11 columns (cam 2 also flipped), so the three cameras differ."""
    out = np.empty((3, 4) + frames.shape[1:], np.uint8)
    for c in range(3):
        for j, i in enumerate((f - 6, f - 4, f - 2, f)):
            x = np.roll(frames[max(0, i)], 37 * c + 11, axis=-1)
            out[c, j] = x[..., ::-1] if c == 2 else x
    return out


def synth(a):
    import torch
    import pp_train as T
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    m = T.PModel("P3")
    net = m.state()["net"]
    big = sorted(net, key=lambda k: -net[k].numel())[:3]
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        m.adapter.out.weight.copy_(torch.randn(m.adapter.out.weight.shape, generator=g) * a.std)
        m.adapter.out.bias.copy_(torch.randn(m.adapter.out.bias.shape, generator=g) * a.std)
    st = {"net": {k: net[k].float() * 1.01 for k in big}, "adapter": None, "parity": m.adapter.state_dict(), "arm": "P3"}
    torch.save({"model": st, "cfg": {"name": "synth", "std": a.std, "perturbed": big}}, out / "ckpt-final.pt")
    print("wrote", out / "ckpt-final.pt", "perturbed x1.01:", big, "adapter out std", a.std)


def ref(a):
    import torch
    import pp_prep
    import pp_train as T
    from jevdrive import op_adapt as A
    dev = torch.device("cuda")
    m = pmodel(a.tag, dev)
    _, enc = pp_prep.encoder(dev)                           # the training cache's side encoder (independent of the server's code)
    res = {}
    for k in (0, 1):
        frames = np.load(REF_DIR / f"frames_{k}.npz")["frames"][:N_REF]
        n = len(frames)
        ego = synth_ego(n, k)
        fr = torch.as_tensor(frames).to(dev)
        prev = torch.cat([torch.zeros_like(fr[:2]), fr[:-2]])
        with torch.no_grad():
            H = torch.cat([m.net.run_batched(A.vision_feeds(prev[i:i + 32], fr[i:i + 32]), ["view_39"])["view_39"].reshape(-1, *A.H_SHAPE)
                           for i in range(0, n, 32)])
            if m.adapter is not None:
                side = None
                if m.adapter.use_side:
                    keys = np.stack([synth_side(frames, f) for f in range(n)])                      # (n, 3, 4, 2, 6, 128, 256)
                    sp, sc = keys[:, :, :-1].reshape(-1, *keys.shape[3:]), keys[:, :, 1:].reshape(-1, *keys.shape[3:])
                    side = torch.from_numpy(enc(sp, sc)).to(dev).reshape(n, 3, PA.SIDE_T, *A.H_SHAPE)
                bias = m.adapter(torch.from_numpy(ego).to(dev), side, None).to(torch.float16)
            else:
                bias = torch.zeros(n, *A.H_SHAPE, dtype=torch.float16, device=dev)
            Hp = torch.cat([torch.zeros(2 * (A.CONTEXT - 1), *A.H_SHAPE, dtype=H.dtype, device=dev), H])
            idx = torch.arange(n, device=dev)[:, None] + torch.arange(0, 2 * A.CONTEXT, 2, device=dev)[None]
            valid = idx >= 2 * (A.CONTEXT - 1)
            tc = torch.tensor([[1.0, 0.0]], device=dev)
            outs = []
            for i in range(0, n, 32):
                ctx = Hp[idx[i:i + 32]] + bias[i:i + 32, None].to(H.dtype)                          # adapter.apply: every context frame
                ctx = ctx * valid[i:i + 32, :, None, None].to(ctx.dtype)
                o = m.net.run_batched(A.policy_feeds(m.net, ctx, T.AT, tc.expand(len(ctx), 2)), ["outputs"])["outputs"]
                outs.append(o.reshape(len(ctx), -1).float().cpu().numpy())
        res[f"out_{k}"], res[f"bias_{k}"], res[f"ego_{k}"] = np.concatenate(outs), bias.float().cpu().numpy(), ego
        print(f"stream {k}: {n} frames, bias |max| {float(bias.abs().max()):.4f} rms {float(bias.float().pow(2).mean().sqrt()):.4f}", flush=True)
    res["tag"] = np.array(a.tag)
    np.savez(a.out, **res)
    print("wrote", a.out)


def check(a):
    from jevdrive.openpilot.model import OPModel
    z = np.load(a.ref)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(a.socket)
    wire.send(sock, {"cmd": "reset"}, {})
    info = wire.recv(sock)[0]["server"]
    m = OPModel(a.onnx, a.backend)
    assert "intent_bias" in m.inputs, "served ONNX has no intent_bias input"
    sl = m.slices
    cols = np.concatenate([np.arange(sl[k].start, sl[k].stop) for k in ("plan", "lead", "lead_prob", "desire_state", "meta")])
    base = OPModel(a.base, a.backend) if a.base else None
    rows, res = [], {"onnx": a.onnx, "ref": a.ref, "ref_tag": str(z["tag"]), "server": info, "backend": a.backend, "streams": []}

    def run(model, frames, bias_of):
        model.reset()
        got = []
        for f, fr in enumerate(frames):
            if bias_of is not None:
                model.extra = {"intent_bias": bias_of(f)[None]}
            model.step(fr)
            got.append(model.step(fr))                    # recorded at the frame's second 20 Hz step, as the stream protocol
        return np.stack(got)

    def cmp(got, ref_, ok):
        d = np.abs(got[ok][:, cols] - ref_[ok][:, cols])
        pg = (got[ok][:, :495] - ref_[ok][:, :495]).reshape(-1, 33, 15)[:, :, :2]
        return {"all_max": float(d.max()), "all_p99": float(np.percentile(d, 99)), "plan_xy_max_m": float(np.abs(pg).max()),
                "plan_dist_mean_m": float(np.linalg.norm(pg, axis=-1).mean())}

    for k in (0, 1):
        frames = np.load(REF_DIR / f"frames_{k}.npz")["frames"][:N_REF]
        ego = z[f"ego_{k}"]
        B = []
        for f in range(len(frames)):
            arr = {"ego": ego[f]}
            if info["use_side"]:
                arr["side"] = synth_side(frames, f)
            wire.send(sock, {"cmd": "bias"}, arr)
            B.append(wire.recv(sock)[1]["bias"])
        B = np.stack(B)
        ok = np.arange(len(frames)) >= 2 * 8
        rec = {"stream": k, "frames": int(ok.sum()), "bias_absmax_server": float(np.abs(B.astype(np.float32)).max()),
               "bias_absmax_ref": float(np.abs(z[f"bias_{k}"]).max()), "bias_maxdiff": float(np.abs(B.astype(np.float32) - z[f"bias_{k}"]).max())}
        served = run(m, frames, lambda f: B[f])
        rec["served_vs_port"] = cmp(served, z[f"out_{k}"], ok)
        dropped = run(m, frames, lambda f: np.zeros(A_SHAPE, np.float16))
        rec["bias_dropped_vs_port"] = cmp(dropped, z[f"out_{k}"], ok)    # what the check would see if the bias were lost
        if base is not None:
            rec["zero_bias_vs_base"] = cmp(dropped, run(base, frames, None), ok)
        res["streams"].append(rec)
        print(json.dumps(rec), flush=True)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(res, indent=1))
    return rows


# ---------------------------------------------------------------- sign conventions (test 4)
def decode(ego: np.ndarray) -> dict:
    """ego_features (N, 20) -> present, cmd (N, 3), vx, vy, ax, ay, pose (N, 4, 3) in metres / rad."""
    p = ego[:, 8:].reshape(-1, 4, 3)
    return {"cmd": ego[:, 1:4], "vx": 10 * ego[:, 4], "ax": 3 * ego[:, 6], "ay": 3 * ego[:, 7],
            "pose": np.concatenate([10 * p[..., :2], p[..., 2:]], -1)}


def sign_stats(D: dict, turn: np.ndarray, moving: np.ndarray, thr: float) -> dict:
    """turn: a left-positive turning signal from another source than the poses (|turn| > thr counts as turning), for the yaw-sign test."""
    P = D["pose"]
    x0, y0, yaw0 = P[:, 0, 0], P[:, 0, 1], P[:, 0, 2]
    mv = moving & (np.abs(yaw0) > np.radians(5))
    tn = moving & (np.abs(turn) > thr)
    return {"n": int(len(x0)), "n_moving": int(moving.sum()),
            "frac_x_oldest_neg_moving": float((x0[moving] < 0).mean()) if moving.any() else None,
            "frac_vx_pos_moving": float((D["vx"][moving] > 0).mean()) if moving.any() else None,
            "n_turning": int(mv.sum()), "frac_sign_y_opposite_yaw_turning": float((np.sign(y0[mv]) == -np.sign(yaw0[mv])).mean()) if mv.any() else None,
            "n_turn_ref": int(tn.sum()), "frac_yaw_oldest_opposite_turn": float((np.sign(yaw0[tn]) == -np.sign(turn[tn])).mean()) if tn.any() else None,
            "cmd_hist": {n: int((D["cmd"].argmax(1)[D["cmd"].sum(1) > 0] == i).sum()) for i, n in enumerate(("left", "straight", "right"))}}


def signs(a):
    res = {}
    # HUGSIM: the agent's logged features; reference turn = the simulator heading change since 1.5 s (hyaw15, + = turned right)
    E, turn, v, cmd_raw = [], [], [], []
    for d in a.runs:
        for f in sorted(Path(d).glob("*/zs_steps.jsonl")):
            for line in open(f):
                r = json.loads(line)
                if "parity" in r:
                    E.append(r["parity"]["ego"])
                    turn.append(-np.radians(r["hyaw15"]))
                    v.append(r["v"])
                    cmd_raw.append(r["cmd"])
    E, turn, v, cmd_raw = np.asarray(E, np.float32), np.asarray(turn), np.asarray(v), np.asarray(cmd_raw)
    D = decode(E)
    h = sign_stats(D, turn, v > 1.0, np.radians(5))
    h["cmd_map"] = {f"hugsim_{c}": {n: int(((D["cmd"].argmax(1) == i) & (cmd_raw == c)).sum()) for i, n in enumerate(("left", "straight", "right"))}
                    for c in (0, 1, 2)}
    mv = (v > 1.0) & (np.abs(turn) > np.radians(5))
    h["yaw_oldest_over_turn15_median"] = float(np.median(D["pose"][mv, 0, 2] / -turn[mv])) if mv.any() else None
    res["hugsim"] = h
    # NAVSIM: the training rows (pp_prep tab.npz); reference turn = the body-frame lateral acceleration ay (left-positive, > 0.5 m/s^2)
    t = np.load(a.navsim)
    D = decode(t["ego"])
    fut = np.nan_to_num(t["fut"])
    res["navsim"] = sign_stats(D, D["ay"], D["vx"] > 1.0, 0.5)
    res["navsim"]["frac_left_cmd_future_left"] = float((fut[D["cmd"].argmax(1) == 0, -1, 2] > 0).mean())
    res["navsim"]["frac_right_cmd_future_right"] = float((fut[(D["cmd"].argmax(1) == 2) & (D["cmd"].sum(1) > 0), -1, 2] < 0).mean())
    print(json.dumps(res, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    o = sp.add_parser("onnx")
    o.add_argument("--tag", required=True)
    o.add_argument("--out", required=True)
    s = sp.add_parser("serve")
    s.add_argument("--tag", required=True)
    s.add_argument("--socket", required=True)
    s.add_argument("--ready-file", default="")
    y = sp.add_parser("synth")
    y.add_argument("--out", required=True)
    y.add_argument("--std", type=float, default=0.01)
    r = sp.add_parser("ref")
    r.add_argument("--tag", required=True)
    r.add_argument("--out", required=True)
    c = sp.add_parser("check")
    c.add_argument("--onnx", required=True)
    c.add_argument("--ref", required=True)
    c.add_argument("--socket", required=True)
    c.add_argument("--base", default="")
    c.add_argument("--backend", default="trt")
    c.add_argument("--json", default="")
    g = sp.add_parser("signs")
    g.add_argument("--runs", nargs="+", required=True)
    g.add_argument("--navsim", required=True)
    g.add_argument("--json", default="")
    a = ap.parse_args()
    {"onnx": onnx_cmd, "serve": serve, "synth": synth, "ref": ref, "check": check, "signs": signs}[a.cmd](a)
