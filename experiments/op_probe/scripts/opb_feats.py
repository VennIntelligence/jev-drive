"""op_probe stage features of op_parity Cinque arms (plans/2026-10-06-dac-localize-prereg.md section 1), from the pp_prep W-frame caches.

  feats   --data lb_navtest | navtrain_full.s2of12 ... --model P2-F-s0 [--limit N]
          -> $DATA_DIR/runs/op_probe/feats/<model>/<data>.npz: tokens, ego (20), V = view_39 of the current frame (32 x 512, the frozen
          vision output; identical for every arm), M = add_40 current frame (after the adapter bias and the two token-MLP blocks),
          T = select_4 (temporal summary), H = add_54 (plan-head hidden), plan_mu (33 x 15), poses (8 x 3: the op_interp `base` adapter,
          i.e. exactly what the devkit scored for op_parity). V and M are float16, the rest float32.
  ablate  --model P2-F-s0 -> $DATA_DIR/runs/op_probe/ablate/<model>.npz: navtest poses (8 x 3) of every input variant (section 4).

op-train env, one GPU (pool job). The forward is pp_train.PModel.forward with extra taps.
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_R / "experiments/op_parity/scripts")]
import argparse, json  # noqa: E401,E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from jevdrive.common import data_dir  # noqa: E402

OUT = data_dir() / "runs" / "op_probe"
TAPS = ["outputs", "add_40", "select_4", "add_54"]


def frames_dir(data, frames):
    cr = data_dir() / "runs" / "op_parity" / "cache"
    return cr / (data if frames == "gimm" else f"{data}@{frames}")


@torch.no_grad()
def forward(model, front, ego, tc, inputs_on=True):
    """pp_train.PModel.forward with taps -> dict of the TAPS (batch first)."""
    from jevdrive import op_adapt as A
    B, n = front.shape[:2]
    H = torch.cat([front.new_zeros(B, A.CONTEXT - n, *front.shape[2:]), front], 1).to(model.net.dtype)
    valid = torch.zeros(B, A.CONTEXT, dtype=torch.bool, device=H.device)
    valid[:, A.CONTEXT - n:] = True
    if model.adapter is not None and inputs_on:
        H = model.adapter.apply(H, ego, None, None)
    H = H * valid[:, :, None, None].to(H.dtype)
    o = model.net.run_batched(A.policy_feeds(model.net, H, (0.275, 0.525), tc.to(model.net.dtype)), TAPS)
    return {k: v.reshape(B, *v.shape[2:]) if v.shape[1] == 1 else v for k, v in o.items()}


def to_poses(plan_mu, cam, t_out):
    from jevdrive import op_interp as I
    return np.stack([I.to_rear(p[:, 0:3], p[:, 11], I.T_IDXS, c[:2], t_out, "lever", "linear") for p, c in zip(plan_mu, cam)]).astype(np.float32)


def load(data, frames):
    tab = dict(np.load(data_dir() / "runs" / "op_parity" / "cache" / data / "tab.npz"))
    front = np.load(frames_dir(data, frames) / "front.npy", mmap_mode="r")
    assert len(front) == len(tab["names"])
    return tab, front


def run_model(model, tab, front, dev, rows, bs, ego_fn=None, front_fn=None, keep=("V", "M", "T", "H")):
    from jevdrive import navsim_zs as Z
    n = len(rows)
    out = {"plan_mu": np.zeros((n, 33, 15), np.float32)}
    if "V" in keep:
        out["V"] = np.zeros((n, 32, 512), np.float16)
    if "M" in keep:
        out["M"] = np.zeros((n, 32, 512), np.float16)
    for k in ("T", "H"):
        if k in keep:
            out[k] = np.zeros((n, 512), np.float32)
    sl = model.net.slices
    pi = np.arange(sl["plan"].start, sl["plan"].start + 495)
    shape_logged = False
    for i in range(0, n, bs):
        r = rows[i:i + bs]
        f = torch.from_numpy(np.ascontiguousarray(front[r])).to(dev)
        e = tab["ego"][r].copy()
        if ego_fn is not None:
            e = ego_fn(e)
        if front_fn is not None:
            f = front_fn(f)
        tc = torch.as_tensor(np.where(tab["lht"][r][:, None], [[0.0, 1.0]], [[1.0, 0.0]]).astype(np.float32), device=dev)
        o = forward(model, f, torch.as_tensor(e, device=dev), tc)
        if not shape_logged:
            print({k: tuple(v.shape) for k, v in o.items()}, flush=True)
            shape_logged = True
        out["plan_mu"][i:i + len(r)] = o["outputs"].float().reshape(len(r), -1)[:, pi].reshape(-1, 33, 15).cpu().numpy()
        if "V" in keep:
            out["V"][i:i + len(r)] = f[:, -1].half().cpu().numpy()
        if "M" in keep:
            m = o["add_40"].reshape(len(r), 9, 32, 512)[:, -1]          # node 489 reshapes add_40 to (9 x 32) tokens, current frame last
            out["M"][i:i + len(r)] = m.half().cpu().numpy()
        if "T" in keep:
            out["T"][i:i + len(r)] = o["select_4"].float().reshape(len(r), -1).cpu().numpy()
        if "H" in keep:
            out["H"][i:i + len(r)] = o["add_54"].float().reshape(len(r), -1).cpu().numpy()
    out["poses"] = to_poses(out["plan_mu"], tab["cam"][rows], Z.T_OUT)
    return out


def cmd_feats(a):
    import pp_train as T
    from jevdrive.data import splits
    from jevdrive.run import Run
    dev = torch.device("cuda")
    with Run("op_probe", f"feats-{a.model}-{a.data}", config=vars(a)) as run:
        run.use_split(splits.load("navsim/navtest" if "navtest" in a.data else "navsim/navtrain"))
        tab, front = load(a.data, a.frames)
        rows = np.arange(len(tab["names"]))[: a.limit or None]
        model = T.load_pmodel(a.model, dev)
        out = run_model(model, tab, front, dev, rows, a.batch, keep=("V", "M", "T", "H") if a.model != "P0" else ("T", "H"))
        d = OUT / "feats" / a.model
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"{a.data}{'-lim' if a.limit else ''}.npz"
        np.savez(f, tokens=tab["names"][rows], log=tab["log"][rows], ego=tab["ego"][rows], fut=tab["fut"][rows], **out)
        # equivalence with op_parity's stored navtest plans of the same model (gate 2)
        ref = data_dir() / "runs" / "op_lb" / "lb_navtest" / "plans" / f"warp@cinque_PP{a.model}.npz"
        if a.data == "lb_navtest" and ref.exists():
            z = np.load(ref)
            dmu = np.abs(out["plan_mu"] - z["plan_mu"][rows])
            pz = np.load(data_dir() / "runs" / "op_lb" / "lb_navtest" / "preds" / f"warp-cinque_PP{a.model}__base.npz")
            dp = np.linalg.norm(out["poses"][..., :2] - pz["poses"][rows][..., :2], axis=-1)
            run.summary.update(plan_mu_max_abs=float(dmu.max()), poses_max_m=float(dp.max()), poses_p99_m=float(np.percentile(dp.max(1), 99)))
        if "M" in out:                                               # M must be the current frame's tokens: residual stream of V
            v, m = out["V"][:256].astype(np.float32), out["M"][:256].astype(np.float32)
            run.summary["cos_M_V"] = float(np.mean(np.sum(v * m, -1) / (np.linalg.norm(v, axis=-1) * np.linalg.norm(m, axis=-1) + 1e-6)))
        run.summary.update(n=len(rows), out=str(f))
        run.info(json.dumps(run.summary, default=str))


VARIANTS = {
    "full": (None, None),
    "no_velacc": (lambda e: _z(e, slice(4, 8)), None),
    "no_pose": (lambda e: _z(e, slice(8, 20)), None),
    "no_cmd": (lambda e: _z(e, slice(1, 4)), None),
    "cmd_straight": (lambda e: _set(e, slice(1, 4), [0, 1, 0]), None),
    "no_bias": (lambda e: _z(e, slice(0, 1)), None),
    "last_frame_only": (None, lambda f: f[:, -1:]),
}


def _z(e, s):
    e[:, s] = 0
    return e


def _set(e, s, v):
    e[:, s] = v
    return e


def cmd_ablate(a):
    import pp_train as T
    from jevdrive.data import splits
    from jevdrive.run import Run
    dev = torch.device("cuda")
    with Run("op_probe", f"ablate-{a.model}", config=vars(a)) as run:
        run.use_split(splits.load("navsim/navtest"))
        model = T.load_pmodel(a.model, dev)
        res = {}
        tab, front = load("lb_navtest", "warp")
        rows = np.arange(len(tab["names"]))[: a.limit or None]
        for v, (ef, ff) in VARIANTS.items():
            res[v] = run_model(model, tab, front, dev, rows, a.batch, ef, ff, keep=())["poses"]
            run.info(f"{v}: mean |d| vs full {np.linalg.norm(res[v][..., :2] - res['full'][..., :2], axis=-1).mean():.3f} m")
        for fr in ("gimm", "keys"):                                   # the front frame protocol swapped (G = GIMM 0.2 s pairs, N = native 2 Hz keys)
            t2, f2 = load("lb_navtest", fr)
            assert (t2["names"] == tab["names"]).all()
            res[f"frames_{fr}"] = run_model(model, t2, f2, dev, rows, a.batch, keep=())["poses"]
            run.info(f"frames_{fr}: mean |d| vs full {np.linalg.norm(res[f'frames_{fr}'][..., :2] - res['full'][..., :2], axis=-1).mean():.3f} m")
        d = OUT / "ablate"
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / f"{a.model}{'-lim' if a.limit else ''}.npz", tokens=tab["names"][rows], **res)
        run.summary.update(variants=list(res), n=len(rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("feats")
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="P2-F-s0")
    p.add_argument("--frames", default="warp")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--batch", type=int, default=128)
    p = sp.add_parser("ablate")
    p.add_argument("--model", default="P2-F-s0")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--batch", type=int, default=128)
    a = ap.parse_args()
    {"feats": cmd_feats, "ablate": cmd_ablate}[a.cmd](a)
