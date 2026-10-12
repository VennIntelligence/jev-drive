"""ROUTE-NIGHT route hinge (research/next-round/route-night.md section 2): the lateral distance of the student's own plan to the row's label
polyline (the full route), minus a per-row dead zone, lateral gradient only. An `--on-lon` module (plans/on_pool_format.md "Optional
longitudinal labels"): `on_train.py / rm_train.py --on-lon on_route`.

Trainer hook:
    pack = on_route.load(pool_dir, names, dev)   # <POOL>/route.npz re-ordered to the trainer's pool row names (compared with / without `on:`)
    L = on_route.loss(poses, rows, pack)         # poses (n, 8, 3) rear-axle x, y, yaw at 0.5 .. 4 s in each row's own frame (with grad),
                                                 # rows (n,) pool row indices -> {"route": (n,)} metres; the trainer applies lambda and ho_w
route.npz (written by route_mem/scripts/rm_build.py pool and route_follow/scripts/rf_cfpool.py):
    tokens (N,) str row names, ok (N,) bool, poly (N, M, 2) f4 label polyline in the row's rear-axle frame (10 m behind the ego's projection
    to 90 m ahead every 2 m, NaN padded, M = 51), seglen (N, M-1) f4, dz (N,) f4 dead zone (1.0 on-policy, 0.5 counterfactual), cov20 (N,) bool.
Rows of the pool missing from route.npz, or with ok False, get loss 0 (and a NaN polyline). If route.npz is absent, every row is 0 (warned).

Rows whose route input was dropped (--rm-drop zeroed their route memory) get loss 0: the hinge would teach a turn without a reason. Two ways,
use either (they are OR-ed):
    loss(poses, rows, pack, dropped=mask)        # mask (n,) bool aligned with poses / rows (i.e. already restricted to the pool rows)
    pack["drop"][:] = False; pack["drop"][pool_rows_dropped] = True     # (N,) bool by POOL row index (store row - on0, the index on_train
                                                 # passes as `rows`), set by the trainer before each step; needs no change to on_train's
                                                 # call site. load() creates it all False. A pool row drawn twice in one batch shares one flag.

Loss per row: near_k = distance of plan pose k to the polyline, over the segments the pose projects onto (along-segment coordinate in
[-0.5, seglen + 0.5]); a pose that projects onto no segment contributes 0. h_k = relu(near_k - dz); L = sum_k W_T[k] h_k / sum W_T with
W_T = [0, 2, 2, 2, 1, 1, 1, 1] (the MPC tracks 1 - 2 s). The poses enter through shape_only (body1 loss43.shape_only, copied here): values unchanged, the gradient
keeps only the component normal to the pose's own detached heading; the yaw gets none.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

NAME = "route.npz"
M = 51
W_T = torch.tensor([0., 2, 2, 2, 1, 1, 1, 1])
SLACK, FILL = 0.5, 1e4


def shape_only(x, y, psi):
    """body1 loss43.shape_only, copied (loss43 imports pp_train): same values, d(x~, y~) / d(x, y) = n n^T with n the unit normal of the
    detached heading. The yaw is not touched."""
    xd, yd, pd = x.detach(), y.detach(), psi.detach()
    nx, ny = -torch.sin(pd), torch.cos(pd)
    s = nx * (x - xd) + ny * (y - yd)
    return xd + nx * s, yd + ny * s


def _bare(n: str) -> str:
    return n[3:] if n.startswith("on:") else n


def load(pool_dir, names, dev) -> dict:
    """<POOL>/route.npz -> tensors in the trainer's pool row order (names (N,)); rows absent from the file are ok False."""
    names = [str(n) for n in np.asarray(names).tolist()]
    N, p = len(names), Path(pool_dir) / NAME
    poly, seg = np.full((N, M, 2), np.nan, np.float32), np.full((N, M - 1), np.nan, np.float32)
    ok, dz, cov = np.zeros(N, bool), np.ones(N, np.float32), np.zeros(N, bool)
    if p.exists():
        z = np.load(p, allow_pickle=False)
        pos = {_bare(str(t)): i for i, t in enumerate(z["tokens"].tolist())}
        assert len(pos) == len(z["tokens"]), f"{p}: duplicate tokens"
        dst = np.array([i for i, n in enumerate(names) if _bare(n) in pos], np.int64)
        src = np.array([pos[_bare(names[i])] for i in dst], np.int64)
        if len(dst):
            assert z["poly"].shape[1:] == (M, 2), f"{p}: poly {z['poly'].shape}, want (N, {M}, 2)"
            poly[dst], seg[dst], ok[dst], dz[dst], cov[dst] = z["poly"][src], z["seglen"][src], z["ok"][src], z["dz"][src], z["cov20"][src]
        print(f"on_route: {p}: {len(dst)} / {N} pool rows labelled, {int(ok.sum())} ok", flush=True)
    else:
        print(f"on_route: WARNING {p} missing, the route hinge is 0 on every row", flush=True)
    t = lambda x: torch.as_tensor(x, device=dev)  # noqa: E731
    return dict(poly=t(poly), seglen=t(seg), ok=t(ok), dz=t(dz), cov20=t(cov), drop=t(np.zeros(N, bool)), names=names, path=str(p))


def near(poses, rows, pack):
    """-> (n, 8) distance of each plan pose to its row's polyline (FILL where the pose projects onto no segment), shape-only gradient."""
    if poses.dtype != torch.float32:
        poses = poses.float()
    P = pack["poly"][rows]                                                     # (n, M, 2), NaN padded
    ok = torch.isfinite(P).all(-1)
    P0 = torch.nan_to_num(P)
    u = F.normalize(P0[:, 1:] - P0[:, :-1], dim=-1)                            # segment tangents (n, M-1, 2); 0 on padded segments
    xs, ys = shape_only(poses[..., 0], poses[..., 1], poses[..., 2])
    d = torch.stack([xs, ys], -1)[:, :, None] - P0[:, None, :-1]               # (n, 8, M-1, 2) pose - segment start
    lon = (d * u[:, None]).sum(-1)
    lat = (d[..., 1] * u[:, None, :, 0] - d[..., 0] * u[:, None, :, 1]).abs()
    seg = torch.nan_to_num(pack["seglen"][rows], nan=-1.0)[:, None]            # (n, 1, M-1)
    cov = (ok[:, None, :-1] & ok[:, None, 1:] & (lon >= -SLACK) & (lon <= seg + SLACK))
    return lat.masked_fill(~cov, FILL).amin(-1)


def loss(poses, rows, pack, dropped=None):
    """The trainer hook -> {"route": (n,)} per-row metres; 0 on rows with ok False, uncovered poses and dropped route input."""
    nr = near(poses, rows, pack)
    h = torch.relu(nr - pack["dz"][rows, None]) * (nr < 1e3)
    w = W_T.to(h)
    keep = pack["ok"][rows]
    if "drop" in pack:
        keep = keep & ~pack["drop"][rows]
    if dropped is not None:
        keep = keep & ~dropped.to(keep.device, torch.bool)
    return {"route": (h * w).sum(1) / w.sum() * keep.to(h.dtype)}


# ---------------------------------------------------------------- smoke (CPU, synthetic rows): python on_route.py --smoke
@torch.no_grad()
def _f(v):
    return float(v)


def _smoke():
    import tempfile
    torch.manual_seed(0)
    s = np.arange(M, dtype=np.float32) * 2 - 10                                # -10 .. 90 m
    def row(y0=0.0, n_ok=M):
        P = np.stack([s, np.full(M, y0, np.float32)], -1)
        P[n_ok:] = np.nan
        return P
    poly = np.stack([row(), row(), row(n_ok=26), row(), row(), row(), np.full((M, 2), np.nan, np.float32)])   # row 2: NaN beyond 40 m, row 6 all NaN
    N = len(poly)
    seglen = np.linalg.norm(np.diff(poly, axis=1), axis=-1).astype(np.float32)
    ok = np.array([1, 1, 1, 1, 0, 1, 1], bool)
    names = np.array([f"on:x:{i}" for i in range(N)])
    Z = dict(tokens=np.array([f"x:{i}" for i in range(N)]), ok=ok, poly=poly, seglen=seglen, dz=np.ones(N, np.float32), cov20=np.ones(N, bool))
    with tempfile.TemporaryDirectory() as td:
        perm = np.random.default_rng(1).permutation(N)
        np.savez(Path(td) / NAME, **{k: v[perm] for k, v in Z.items()})       # file in a shuffled order
        order = np.random.default_rng(2).permutation(N)
        pk = load(td, list(names[order]) + ["on:missing:0"], "cpu")
        inv = np.argsort(order)                                                # pool index of x:i
        for k in ("poly", "seglen", "ok"):
            a, b = pk[k][torch.as_tensor(inv)].numpy(), Z[k]
            assert np.array_equal(np.isnan(a), np.isnan(b)) if a.dtype != bool else True
            assert np.allclose(np.nan_to_num(a), np.nan_to_num(b)), k
        assert not pk["ok"][N] and torch.isnan(pk["poly"][N]).all(), "missing row"
    print("load: re-order + missing row ok")
    t = torch.arange(1, 9, dtype=torch.float32) * 0.5
    def plan(n, y=0.0, v=4.0, x0=0.0):
        return torch.stack([x0 + v * t, torch.full_like(t, y), torch.zeros_like(t)], -1)[None].repeat(n, 1, 1).requires_grad_(True)
    R = lambda *i: torch.as_tensor(inv[list(i)])  # noqa: E731
    out = {}
    p = plan(1); L = loss(p, R(0), pk)["route"]; out["1 on line"] = _f(L)
    assert abs(out["1 on line"]) < 1e-6
    p = plan(1, y=2.0); L = loss(p, R(0), pk)["route"]; L.sum().backward(); g = p.grad[0]
    out["2 shift 2 m"], out["2 |grad x|"], out["2 |grad y|"], out["2 |grad yaw|"] = _f(L), _f(g[:, 0].abs().max()), _f(g[:, 1].abs().sum()), _f(g[:, 2].abs().max())
    assert abs(out["2 shift 2 m"] - 1.0) < 1e-5 and out["2 |grad x|"] < 1e-7 and out["2 |grad yaw|"] == 0 and abs(out["2 |grad y|"] - 1.0) < 1e-5
    p = plan(1, y=2.0, x0=100.0); L = loss(p, R(0), pk)["route"]; L.sum().backward()
    out["3 beyond coverage"] = _f(L)
    assert out["3 beyond coverage"] == 0 and torch.isfinite(p.grad).all()
    p = plan(1, y=2.0, x0=-20.0, v=0.5); L = loss(p, R(0), pk)["route"]; out["3b behind coverage"] = _f(L)
    assert out["3b behind coverage"] == 0
    p = plan(3, y=2.0, v=8.0, x0=20.0); L = loss(p, R(2, 6, 1), pk)["route"]; L.sum().backward()   # row 2: poses at 24 - 52 m, coverage ends at 40 m
    out["4 NaN pad (row 2, row 6 all-NaN, row 1)"] = [round(_f(v), 4) for v in L]
    assert torch.isfinite(p.grad).all() and _f(L[1]) == 0 and abs(_f(L[2]) - 1) < 1e-5 and abs(_f(L[0]) - 0.7) < 1e-5     # 5 of 8 poses covered: (0 + 2 + 2 + 2 + 1) / 10
    p = plan(1, y=2.0); L = loss(p, R(4), pk)["route"]; L.sum().backward(); out["5 ok False"] = _f(L)
    assert out["5 ok False"] == 0 and torch.isfinite(p.grad).all()
    p = plan(2, y=2.0); L = loss(p, R(3, 5), pk, dropped=torch.tensor([True, False]))["route"]; out["6 dropped kwarg [T, F]"] = [_f(v) for v in L]
    assert out["6 dropped kwarg [T, F]"] == [0.0, 1.0]
    pk["drop"][R(5)] = True; L = loss(plan(2, y=2.0), R(3, 5), pk)["route"]; pk["drop"][:] = False; out["6b pack drop [-, T]"] = [_f(v) for v in L]
    assert out["6b pack drop [-, T]"] == [1.0, 0.0]
    p = plan(1, y=-2.0).detach().half().requires_grad_(True); L = loss(p, R(0), pk)["route"]; L.sum().backward(); out["7 fp16 shift -2 m"] = _f(L)
    assert abs(out["7 fp16 shift -2 m"] - 1) < 1e-3 and torch.isfinite(p.grad).all() and bool((p.grad[0, 1:, 1] < 0).all())
    pr = torch.stack([4 * t * torch.cos(torch.tensor(0.3)), 4 * t * torch.sin(torch.tensor(0.3)) + 2, torch.full_like(t, 0.3)], -1)[None].requires_grad_(True)
    loss(pr, R(0), pk)["route"].sum().backward(); g = pr.grad[0, :, :2]
    out["8 heading 0.3: max |grad . heading|"] = _f((g[:, 0] * np.cos(0.3) + g[:, 1] * np.sin(0.3)).abs().max())
    assert out["8 heading 0.3: max |grad . heading|"] < 1e-6
    for k, v in out.items():
        print(f"  {k}: {v}")
    print("on_route smoke: PASS")


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        _smoke()
