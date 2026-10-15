"""RECOVERY-ENVELOPE L (experiments/route_follow/plans/2026-10-16-recovery-envelope-prereg.md): an asymmetric longitudinal speed envelope on
the policy's own pool rows, computed at each row's own state against the traffic as replayed at that moment. The generalisation of
on_approach.py's speed term L3 (the shape terms L1 / L2 are not part of it: this module never loads on_approach's labels).
`rm_train.py --env-lam W` = `--on-lon on_envelope --on-lon-lam 10 W` (the same scale as --app-lam).

Trainer hook:
    pack = on_envelope.load(pool_dir, names, dev)   # <POOL>/envelope.npz re-ordered to the trainer's pool row names (rows absent: ok False)
    L = on_envelope.loss(poses, rows, pack)         # poses (n, 8, 3) rear-axle x, y, yaw at 0.5 .. 4 s in each row's frame (with grad)
                                                    # -> {"envU", "envL"}: (n,) per row, m/s
envelope.npz (experiments/route_follow/scripts/rf_recov.py label / pack):
    tokens (N,) str, ok (N,) bool, route (N, NR, 2) f4 the row's route grid in its own rear-axle frame (on_lon pack `route`: 1 m of arc
    from the route origin), vuf (N, NR) f4 upper bound on the speed at the FRONT-BUMPER arc (stop lines, leads), vur (N, NR) upper bound at
    the REAR-AXLE arc (turn entry), vl (N, NR) lower bound at the rear-axle arc (free road); BIG (1e3) = no upper bound, 0 = no lower bound.
    The bounds already contain the margins and the feasibility floor (rf_recov.py); cls (N,) str the row's reason class (stop / turn / lead /
    free / "") for the draw weights and the reads. Optional (Amendment 1, the lead bound in time): lsig (N, 41) f4 the nearest eligible lead's
    least corner arc at each dense step (BIG none), sf0 (N,) the row's own front-bumper arc, v0 (N,) its fed speed: at step t the upper bound
    min(.., lead(t)), lead(t) = sqrt(2 A_DEC rem) + 1.0 for rem = lsig(t) - STOP_M - sf(t) > 0, else 0, floored at sqrt(max(v0^2 - 2 A_MAX
    (sf(t) - sf0), 0)) (rule_expert's stop margin and braking envelope against the lead where it is at the plan's own time).

Per dense step t = 0.1 .. 4.0 s (on_lon's geometry: the plan through timing_only, i.e. every pose moves only along its own detached
heading and the yaw gets no gradient; dense poses at 0.1 s; front-bumper / rear-axle arcs sf / sr on the row's route, detached for the grid
look-up; path speed v): with the step on the route (|lateral of the rear axle| <= 3.5 m) and inside the route's known grid,
    envU = mean over bounded steps of relu(v - min(vuf(sf), vur(sr)))       (bounded: an upper bound < BIG / 2 at the step)
    envL = mean over bounded steps of relu(vl(sr) - v)                       (bounded: vl > 0)
0 on rows with ok False or no bounded step. Linear interpolation of the grids between the two vertices around the arc.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import on_lon as OL  # noqa: E402

NAME = "envelope.npz"
BIG, LAT_ROUTE = 1e3, 3.5
A_DEC, A_MAX, STOP_M, MARGIN_UP = 3.0, 6.0, 1.5, 1.0


def _bare(n: str) -> str:
    return n[3:] if n.startswith("on:") else n


def load(pool_dir, names, dev) -> dict:
    """<POOL>/envelope.npz -> tensors in the trainer's pool row order; rows absent from the file get ok False."""
    names = [str(n) for n in np.asarray(names).tolist()]
    p = Path(pool_dir) / NAME
    assert p.exists(), f"on_envelope: {p} missing (rf_recov.py pack)"
    z = np.load(p, allow_pickle=False)
    NR = z["route"].shape[1]
    N = len(names)
    R = np.zeros((N, NR, 2), np.float32)
    vuf, vur, vl = np.full((N, NR), BIG, np.float32), np.full((N, NR), BIG, np.float32), np.zeros((N, NR), np.float32)
    ok = np.zeros(N, bool)
    pos = {_bare(str(t)): i for i, t in enumerate(z["tokens"].tolist())}
    assert len(pos) == len(z["tokens"]), f"{p}: duplicate tokens"
    dst = np.array([i for i, n in enumerate(names) if _bare(n) in pos], np.int64)
    src = np.array([pos[_bare(names[i])] for i in dst], np.int64)
    if len(dst):
        R[dst], vuf[dst], vur[dst], vl[dst], ok[dst] = (z[k][src] for k in ("route", "vuf", "vur", "vl", "ok"))
    lsig, sf0, v0 = np.full((N, 41), BIG, np.float32), np.zeros(N, np.float32), np.zeros(N, np.float32)
    if len(dst) and "lsig" in z.files:
        lsig[dst], sf0[dst], v0[dst] = z["lsig"][src], z["sf0"][src], z["v0"][src]
    cls = np.full(N, "", "U8")
    if len(dst) and "cls" in z.files:
        cls[dst] = z["cls"][src]
    print(f"on_envelope: {p}: {len(dst)} / {N} pool rows labelled, {int(ok.sum())} ok; classes "
          + ", ".join(f"{c} {int((ok & (cls == c)).sum())}" for c in ("stop", "turn", "lead", "free")), flush=True)
    t = lambda x: torch.as_tensor(x, device=dev)  # noqa: E731
    return dict(route=t(R), vuf=t(vuf), vur=t(vur), vl=t(vl), ok=t(ok), lsig=t(lsig), sf0=t(sf0), v0=t(v0), names=names, path=str(p), cls=cls)


def _grid(V, s):
    """Per-row grid values V (n, NR) at arcs s (n, T) -> (n, T), linear between the vertices; the arc clamped into the grid."""
    NR = V.shape[1]
    x = s.clamp(0, NR - 1 - 1e-4)
    i = x.floor().long()
    w = x - i
    a, b = V.gather(1, i), V.gather(1, (i + 1).clamp(max=NR - 1))
    return a + w * (b - a)


def terms(poses, rows, pack):
    """-> envU, envL (n,) losses and the per-step reads: v (n, 41), vu (n, 41) the active upper bound (BIG none), vl (n, 41), on (n, 41) the
    steps on the route and inside the grid, sf / sr (n, 41)."""
    if poses.dtype != torch.float32:
        poses = poses.float()
    x, y, psi = poses.unbind(-1)
    xt, yt = OL.timing_only(x, y, psi)
    d = OL.dense(xt, yt, psi.detach())
    R = pack["route"][rows]
    NR = R.shape[1]
    with torch.no_grad():
        sf, _ = OL.project(R, OL.front_point(d))
        sr, lr = OL.project(R, d[..., :2])
    v = OL.path_speed(d)
    tpos = torch.arange(OL.NS, device=x.device)[None] > 0
    on = tpos & (lr.abs() <= LAT_ROUTE) & (sr <= NR - 1) & (sf <= NR - 1) & pack["ok"][rows][:, None]
    vu = torch.minimum(_grid(pack["vuf"][rows], sf), _grid(pack["vur"][rows], sr))
    if "lsig" in pack:                                                          # Amendment 1: the lead bound at the plan's own time
        ls = pack["lsig"][rows]
        rem = ls - STOP_M - sf
        vlead = torch.where(rem > 0, torch.sqrt(2 * A_DEC * rem.clamp_min(0)) + MARGIN_UP, torch.zeros_like(rem))
        floor = torch.sqrt((pack["v0"][rows][:, None] ** 2 - 2 * A_MAX * (sf - pack["sf0"][rows][:, None])).clamp_min(0))
        vu = torch.minimum(vu, torch.where(ls < BIG / 2, torch.maximum(vlead, floor), torch.full_like(vu, BIG)))
    vlo = _grid(pack["vl"][rows], sr)
    bu = on & (vu < BIG / 2)
    bl = on & (vlo > 0)
    eu = torch.where(bu, torch.relu(v - vu), torch.zeros_like(v))
    el = torch.where(bl, torch.relu(vlo - v), torch.zeros_like(v))
    envU = eu.sum(1) / bu.sum(1).clamp_min(1)
    envL = el.sum(1) / bl.sum(1).clamp_min(1)
    return dict(envU=envU, envL=envL, v=v.detach(), vu=torch.where(bu, vu, torch.full_like(vu, BIG)), vl=torch.where(bl, vlo, torch.zeros_like(vlo)),
                on=on, sf=sf, sr=sr, bu_rows=bu.any(1), bl_rows=bl.any(1))


def loss(poses, rows, pack):
    """The trainer hook -> {"envU", "envL"}: (n,) per-row values (0 on rows without a bound)."""
    T = terms(poses, rows, pack)
    return {"envU": T["envU"], "envL": T["envL"]}


# ---------------------------------------------------------------- smoke (CPU, synthetic rows): python on_envelope.py --smoke
def _smoke():
    import tempfile
    NR = 131
    g = np.arange(NR, dtype=np.float32)
    straight = np.stack([g - 20.0, np.zeros(NR, np.float32)], -1)              # the row sits at arc 20 m of its route
    N = 4
    Z = dict(tokens=np.array([f"x:{i}" for i in range(N)]), ok=np.array([1, 1, 1, 0], bool), route=np.stack([straight] * N),
             vuf=np.full((N, NR), BIG, np.float32), vur=np.full((N, NR), BIG, np.float32), vl=np.zeros((N, NR), np.float32),
             cls=np.array(["stop", "free", "turn", "stop"]))
    Z["vuf"][0] = np.where(g < 50, 5.0, 0.0)                                    # row 0: front bumper at most 5 m/s, stop at arc 50
    Z["vl"][1] = 8.0                                                              # row 1: at least 8 m/s
    Z["vur"][2] = 6.0                                                             # row 2: rear axle at most 6 m/s
    Z["vuf"][3] = 0.0                                                             # row 3: not ok
    with tempfile.TemporaryDirectory() as td:
        np.savez(Path(td) / NAME, **Z)
        pk = load(td, [f"on:x:{i}" for i in (3, 2, 1, 0)] + ["on:missing"], "cpu")
    R = lambda *i: torch.as_tensor([3 - j for j in i])  # noqa: E731
    t = torch.arange(1, 9, dtype=torch.float32) * 0.5

    def plan(v):
        return torch.stack([v * t, torch.zeros_like(t), torch.zeros_like(t)], -1)[None].clone().requires_grad_(True)
    p = plan(4.0)
    T = terms(p, R(0), pk)
    assert float(T["envU"]) == 0 and float(T["envL"]) == 0, "4 m/s under 5"
    p = plan(6.0)                                                               # front at 24.049 + 6 t: reaches arc 50 at t ~ 4.3 s: never
    T = terms(p, R(0), pk)
    assert abs(float(T["envU"]) - 1.0) < 1e-3, float(T["envU"])
    T["envU"].sum().backward()
    assert float(p.grad[0, :, 1].abs().max()) < 1e-7 and float(p.grad[0, :, 2].abs().max()) == 0 and float(p.grad[0, :, 0].sum()) > 0
    p = plan(10.0)                                                              # front passes arc 50 at t = 2.6 s: v 10 vs 0 there
    T = terms(p, R(0), pk)
    exp = (5.0 * 25 + 10.0 * 15) / 40
    assert abs(float(T["envU"]) - exp) < 0.3, (float(T["envU"]), exp)
    T = terms(plan(5.0), R(1), pk)
    assert abs(float(T["envL"]) - 3.0) < 1e-3 and float(T["envU"]) == 0
    T = terms(plan(9.0), R(1), pk)
    assert float(T["envL"]) == 0
    T = terms(plan(7.0), R(2), pk)
    assert abs(float(T["envU"]) - 1.0) < 1e-3
    T = terms(plan(20.0), R(3), pk)
    assert float(T["envU"]) == 0 and float(T["envL"]) == 0, "ok False"
    T = terms(plan(20.0), torch.as_tensor([4]), pk)
    assert float(T["envU"]) == 0, "missing row"
    p = torch.zeros(1, 8, 3, requires_grad=True)
    T = terms(p, R(0), pk)
    (T["envU"] + T["envL"]).sum().backward()
    assert bool(torch.isfinite(p.grad).all())
    assert set(loss(plan(1.0), R(0), pk)) == {"envU", "envL"}
    print("on_envelope smoke: PASS")


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        _smoke()
