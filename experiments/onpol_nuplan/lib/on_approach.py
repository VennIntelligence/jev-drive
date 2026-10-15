"""APPROACH COURSE terms (experiments/route_follow/plans/2026-10-15-approach-course-prereg.md): on the policy's own states in the 40 m
before a turn entry, the plan is pulled toward the HUMAN's entry state (lateral position, heading) and kept no faster than the human's speed
profile over the last 26 m before the entry. An `--on-lon` module (plans/on_pool_format.md "Optional longitudinal labels"):
`rm_train.py --app-lam W` = `--on-lon on_approach --on-lon-lam 10 W` (each term summed over the pool rows / n_imit x ho_w x 10 W).

Trainer hook:
    pack = on_approach.load(pool_dir, names, dev)  # <POOL>/approach.npz re-ordered to the trainer's pool row names
    L = on_approach.loss(poses, rows, pack)        # poses (n, 8, 3) rear-axle x, y, yaw at 0.5 .. 4 s in each row's own frame (with grad),
                                                   # rows (n,) pool row indices -> {"app1", "app2", "app3"}: (n,) per row
approach.npz (experiments/route_follow/scripts/rf_apppool.py):
    tokens (N,) str, ok (N,) bool, kind (N,) int8 (1 turn row, 0 straight negative), H (N, 91, 2) f4 the target path in the row's rear-axle
    frame at arcs -10 .. +80 m every 1 m from the ego's foot on it (turn rows: the human's logged path; negatives: the route centreline),
    NaN where invalid; Hh (N, 91) its heading (row frame, rad); Hv (N, 91) the human's speed at that point (m/s, NaN on negatives);
    d_e (N,) arc of the entry line from the foot (m; negatives: a virtual entry); l3 (N,) bool the speed term applies (a turn row whose
    human did not stop in the 26 m before the entry).
Rows absent from the file or with ok False get 0 on every term.

Terms (geometry of each plan pose against the polyline H: the covering segment nearest to the pose, picked on detached values; station s =
arc of the foot from the ego's foot, signed lateral e left +; the ego's own lateral offset e0 = the origin's):
  app1 (L1)  the plan at the entry line: the first plan point whose station reaches d_e (linear between the two poses that bracket it, weight
             detached): |e| - 0.3 m dead zone + |heading of that plan segment - H's heading at d_e| - 5 deg dead zone (rad). Entry beyond the
             plan's 4 s: the end pose against H at the end pose's own station (the human path at the same arc). Lateral only: positions enter
             through shape_only (gradient normal to each pose's own detached heading); the heading is the chord of two shape-only poses.
  app2 (L2)  the 1.0 / 1.5 / 2.0 s poses: |e_k - e0 (1 - clip(s_k / d_e, 0, 1))| - 0.3 m dead zone, mean of the three, x 2 (the MPC horizon,
             decision 284): from where the policy is, the straight-line blend of the lateral offset into the human's entry state. Lateral only.
  app3 (L3)  plan speed of each 0.5 s segment (chord / 0.5 s) at its midpoint's distance-to-entry d = d_e - s in [0, 26] m vs the human's
             speed at the same station: relu(v - v_h - 0.5 m/s) (one-sided: only faster than the human), mean over the segments in range
             (0 when none). Longitudinal only: positions enter through lon_only (gradient along each pose's own detached heading). l3 rows only.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

NAME = "approach.npz"
M, G0 = 91, -10.0                                  # target polyline: 91 points, arcs -10 .. +80 m, 1 m
DZ_LAT, DZ_HEAD, DZ_V = 0.3, np.radians(5.0), 0.5
W2, K2 = 2.0, (1, 2, 3)                            # L2: weight, plan poses 1.0 / 1.5 / 2.0 s
V_WIN, DT = 26.0, 0.5
SLACK, FILL = 0.5, 1e4
MIN_SEG = 0.2                                      # m: a plan segment shorter than this has no heading (standstill)


def shape_only(x, y, psi):
    """on_route.shape_only: same values, the gradient keeps only the component normal to the detached heading."""
    xd, yd, pd = x.detach(), y.detach(), psi.detach()
    nx, ny = -torch.sin(pd), torch.cos(pd)
    s = nx * (x - xd) + ny * (y - yd)
    return xd + nx * s, yd + ny * s


def lon_only(x, y, psi):
    """Same values, the gradient keeps only the component along the detached heading."""
    xd, yd, pd = x.detach(), y.detach(), psi.detach()
    tx, ty = torch.cos(pd), torch.sin(pd)
    s = tx * (x - xd) + ty * (y - yd)
    return xd + tx * s, yd + ty * s


def _bare(n: str) -> str:
    return n[3:] if n.startswith("on:") else n


def load(pool_dir, names, dev) -> dict:
    """<POOL>/approach.npz -> tensors in the trainer's pool row order (names (N,)); rows absent from the file are ok False."""
    names = [str(n) for n in np.asarray(names).tolist()]
    N, p = len(names), Path(pool_dir) / NAME
    H, Hh, Hv = np.full((N, M, 2), np.nan, np.float32), np.full((N, M), np.nan, np.float32), np.full((N, M), np.nan, np.float32)
    ok, kind, de, l3 = np.zeros(N, bool), np.zeros(N, np.int8), np.full(N, np.nan, np.float32), np.zeros(N, bool)
    if p.exists():
        z = np.load(p, allow_pickle=False)
        pos = {_bare(str(t)): i for i, t in enumerate(z["tokens"].tolist())}
        assert len(pos) == len(z["tokens"]), f"{p}: duplicate tokens"
        dst = np.array([i for i, n in enumerate(names) if _bare(n) in pos], np.int64)
        src = np.array([pos[_bare(names[i])] for i in dst], np.int64)
        if len(dst):
            assert z["H"].shape[1:] == (M, 2), f"{p}: H {z['H'].shape}"
            H[dst], Hh[dst], Hv[dst], ok[dst], kind[dst], de[dst], l3[dst] = (z[k][src] for k in ("H", "Hh", "Hv", "ok", "kind", "d_e", "l3"))
        print(f"on_approach: {p}: {len(dst)} / {N} pool rows labelled, {int(ok.sum())} ok ({int((ok & (kind == 1)).sum())} turn, "
              f"{int((ok & (kind == 0)).sum())} negative, {int((ok & l3).sum())} with the speed term)", flush=True)
    else:
        print(f"on_approach: WARNING {p} missing, every term is 0", flush=True)
    t = lambda x: torch.as_tensor(x, device=dev)  # noqa: E731
    return dict(H=t(H), Hh=t(Hh), Hv=t(Hv), ok=t(ok), kind=t(kind), d_e=t(de), l3=t(l3), names=names, path=str(p))


def project(Q, H):
    """Points Q (n, K, 2) (with grad) on polylines H (n, M, 2) NaN padded -> station s (n, K) detached, signed lateral e (n, K) (left +, grad
    through Q), covered (n, K) bool, segment index j (n, K). The segment: among those the point projects onto (along-segment coordinate in
    [-0.5, len + 0.5]), the one with the smallest |lateral| (detached)."""
    ok = torch.isfinite(H).all(-1)
    H0 = torch.nan_to_num(H)
    d = H0[:, 1:] - H0[:, :-1]
    L = torch.linalg.norm(d, dim=-1)                                            # (n, M-1)
    u = F.normalize(d, dim=-1)
    sv = ok[:, :-1] & ok[:, 1:] & (L > 1e-6)
    r = Q[:, :, None] - H0[:, None, :-1]                                        # (n, K, M-1, 2)
    lon = (r * u[:, None]).sum(-1)
    lat = u[:, None, :, 0] * r[..., 1] - u[:, None, :, 1] * r[..., 0]
    cov = sv[:, None] & (lon >= -SLACK) & (lon <= L[:, None] + SLACK)
    j = lat.detach().abs().masked_fill(~cov, FILL).argmin(-1)                   # (n, K)
    covered = cov.gather(-1, j[..., None])[..., 0]
    e = lat.gather(-1, j[..., None])[..., 0]
    lj = lon.detach().gather(-1, j[..., None])[..., 0]
    s = G0 + j.float() + torch.minimum(lj.clamp(min=0.0), L.gather(-1, j)) / L.gather(-1, j).clamp_min(1e-6)   # the grid is 1 m of arc
    return s, e, covered, j


def _interp(V, s):
    """Per-row values V (n, M) on the 1 m grid from G0 at stations s (n, K) -> (n, K), linear, NaN outside."""
    x = (s - G0).clamp(0, M - 1 - 1e-6)
    i = x.floor().long()
    w = x - i
    a, b = V.gather(1, i), V.gather(1, (i + 1).clamp(max=M - 1))
    return a + w * (b - a)


def _angle(Hh, s):
    x = (s - G0).clamp(0, M - 1 - 1e-6)
    i = x.floor().long()
    w = x - i
    a, b = Hh.gather(1, i), Hh.gather(1, (i + 1).clamp(max=M - 1))
    return a + w * torch.atan2(torch.sin(b - a), torch.cos(b - a))


def wrap(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


def terms(poses, rows, pack):
    """-> dict of per-row tensors: app1, app2, app3 (losses) and the raw reads lat_e (|e| at the entry / end pose), head_e (rad), vex
    (mean speed excess over the human in the 26 m window, no dead zone), reach (the plan reaches the entry line), v_rows (rows with a
    segment in the speed window); losses are 0 on rows with ok False."""
    if poses.dtype != torch.float32:
        poses = poses.float()
    n = len(rows)
    H, Hh, Hv, de = pack["H"][rows], pack["Hh"][rows], pack["Hv"][rows], pack["d_e"][rows]
    ok = pack["ok"][rows] & torch.isfinite(de)
    de = torch.nan_to_num(de, nan=1e3)
    x, y, psi = poses[..., 0], poses[..., 1], poses[..., 2]
    z = torch.zeros_like(x[:, :1])
    xs, ys = shape_only(x, y, psi)
    Qs = torch.stack([torch.cat([z, xs], 1), torch.cat([z, ys], 1)], -1)        # (n, 9, 2) lateral-only, origin first
    s, e, cov, _ = project(Qs, H)
    s, e = s.detach(), e
    e0 = e[:, 0].detach()
    # L2: the 1.0 - 2.0 s poses against the blend of the ego's offset into the human's entry state
    k2 = torch.as_tensor(K2, device=x.device) + 1
    sk, ek, ck = s[:, k2], e[:, k2], cov[:, k2]
    tgt = e0[:, None] * (1 - (sk / de[:, None].clamp_min(1e-3)).clamp(0, 1))
    app2 = W2 * (torch.relu((ek - tgt).abs() - DZ_LAT) * ck).sum(1) / len(K2)
    # L1: the plan at the entry line (or its end pose)
    reached = (s[:, 1:] >= de[:, None]) & cov[:, 1:]
    has = reached.any(1)
    k = torch.where(has, reached.float().argmax(1) + 1, torch.full_like(de, 8, dtype=torch.long))
    ar = torch.arange(n, device=x.device)
    s0_, s1_ = s[ar, k - 1], s[ar, k]
    w = torch.where(has, ((de - s0_) / (s1_ - s0_).clamp_min(1e-6)).clamp(0, 1), torch.ones_like(de)).detach()
    e_at = (1 - w) * e[ar, k - 1] + w * e[ar, k]
    st_at = torch.where(has, de, s1_)
    c_at = torch.where(has, cov[ar, k] & cov[ar, k - 1], cov[ar, k])
    seg = Qs[ar, k] - Qs[ar, k - 1]
    sl = torch.linalg.norm(seg.detach(), dim=-1)
    hp = torch.atan2(seg[:, 1], seg[:, 0] + (sl < MIN_SEG).float())             # standstill segment: masked below (no 0 / 0)
    dh = wrap(hp - _angle(torch.nan_to_num(Hh), st_at[:, None])[:, 0])
    hm = (sl >= MIN_SEG) & c_at
    app1 = torch.relu(e_at.abs() - DZ_LAT) * c_at + torch.relu(dh.abs() - DZ_HEAD) * hm
    # L3: segment speeds against the human's speed at the same station, last 26 m before the entry
    xl, yl = lon_only(x, y, psi)
    Ql = torch.stack([torch.cat([z, xl], 1), torch.cat([z, yl], 1)], -1)
    v = torch.linalg.norm(Ql[:, 1:] - Ql[:, :-1] + 1e-9, dim=-1) / DT           # (n, 8)
    sm = 0.5 * (s[:, 1:] + s[:, :-1])
    dd = de[:, None] - sm
    vh = _interp(torch.nan_to_num(Hv, nan=-1.0), sm)
    inw = (dd >= 0) & (dd <= V_WIN) & cov[:, 1:] & cov[:, :-1] & (vh >= 0) & pack["l3"][rows][:, None]
    nw = inw.sum(1)
    app3 = (torch.relu(v - vh - DZ_V) * inw).sum(1) / nw.clamp_min(1)
    vex = (torch.relu(v - vh) * inw).sum(1) / nw.clamp_min(1)
    okf = ok.to(app1.dtype)
    return dict(app1=app1 * okf, app2=app2 * okf, app3=app3 * okf, lat_e=e_at.abs().detach(), head_e=dh.abs().detach(), vex=vex.detach(),
                reach=has, v_rows=nw > 0, lat_cov=c_at, ok=ok)


def loss(poses, rows, pack):
    """The trainer hook -> {"app1", "app2", "app3"}: (n,) per-row values (0 on rows without a label)."""
    T = terms(poses, rows, pack)
    return {k: T[k] for k in ("app1", "app2", "app3")}


# ---------------------------------------------------------------- smoke (CPU, synthetic rows): python on_approach.py --smoke
def _smoke():
    import tempfile
    torch.manual_seed(0)
    g = np.arange(M, dtype=np.float32) + G0
    straight = np.stack([g, np.zeros(M, np.float32)], -1)
    names = np.array([f"on:x:{i}" for i in range(4)])
    H = np.stack([straight, straight + [0, 1.0], straight, straight])
    Hh = np.zeros((4, M), np.float32)
    Hv = np.full((4, M), 4.0, np.float32)
    Z = dict(tokens=np.array([f"x:{i}" for i in range(4)]), ok=np.array([1, 1, 1, 0], bool), kind=np.array([1, 1, 0, 1], np.int8), H=H, Hh=Hh, Hv=Hv,
             d_e=np.array([20.0, 20.0, 30.0, 20.0], np.float32), l3=np.array([1, 1, 0, 1], bool))
    with tempfile.TemporaryDirectory() as td:
        np.savez(Path(td) / NAME, **Z)
        pk = load(td, list(names[::-1]) + ["on:missing:0"], "cpu")            # reversed order + a missing row
    R = lambda *i: torch.as_tensor([3 - j for j in i])  # noqa: E731
    t = torch.arange(1, 9, dtype=torch.float32) * 0.5

    def plan(v=4.0, y=0.0, yaw=0.0):
        P = torch.stack([v * t, torch.full_like(t, y), torch.full_like(t, yaw)], -1)[None]
        return P.clone().requires_grad_(True)
    out = {}
    T = terms(plan(), R(0), pk)
    out["1 on the path, human speed"] = [float(T[k]) for k in ("app1", "app2", "app3")]
    assert max(out["1 on the path, human speed"]) < 1e-6 and not bool(T["reach"])          # 16 m < 20 m: end pose
    p = plan(v=8.0)
    T = terms(p, R(0), pk)
    out["2 v 8 m/s (human 4): app3, vex, reach"] = [float(T["app3"]), float(T["vex"]), bool(T["reach"])]
    assert abs(float(T["app3"]) - 3.5) < 1e-4 and abs(float(T["vex"]) - 4.0) < 1e-4 and bool(T["reach"])
    T["app3"].sum().backward()
    out["2 grad: |y| / |x| / |yaw|"] = [float(p.grad[0, :, 1].abs().max()), float(p.grad[0, :, 0].abs().sum()), float(p.grad[0, :, 2].abs().max())]
    assert out["2 grad: |y| / |x| / |yaw|"][0] < 1e-7 and out["2 grad: |y| / |x| / |yaw|"][1] > 0 and out["2 grad: |y| / |x| / |yaw|"][2] == 0
    p = plan(y=0.0)
    T = terms(p, R(1), pk)                                                      # human path 1 m to the left: ego offset -1 m at the foot
    out["3 path +1 m: e0 blend app2, app1"] = [float(T["app2"]), float(T["app1"])]
    # poses at 4, 6, 8 m, d_e 20: tgt = -1 (1 - s / 20) -> err = 1 - (1 - s/20)... e = -1, tgt = -(1 - s/20): |e - tgt| = s / 20 = 0.2, 0.3, 0.4
    assert abs(out["3 path +1 m: e0 blend app2, app1"][0] - 2 * (0 + 0 + 0.1) / 3) < 1e-4
    assert abs(out["3 path +1 m: e0 blend app2, app1"][1] - 0.7) < 1e-4                    # end pose 1 m off: 1 - 0.3
    (T["app1"] + T["app2"]).sum().backward()
    out["3 grad: |x| / y sign at 2 s"] = [float(p.grad[0, :, 0].abs().max()), float(p.grad[0, 3, 1])]
    assert out["3 grad: |x| / y sign at 2 s"][0] < 1e-7 and out["3 grad: |x| / y sign at 2 s"][1] < 0
    T = terms(plan(v=8.0, y=2.0), R(2), pk)                                     # negative: no speed term, lateral to the centreline
    out["4 negative 2 m off, v 8: app1, app3"] = [float(T["app1"]), float(T["app3"])]
    assert float(T["app3"]) == 0 and float(T["app1"]) > 1.0
    T = terms(plan(v=8.0, y=2.0), R(3), pk)
    out["5 ok False"] = [float(T[k]) for k in ("app1", "app2", "app3")]
    assert max(out["5 ok False"]) == 0
    p = torch.zeros(1, 8, 3, requires_grad=True)
    T = terms(p, R(0), pk)
    (T["app1"] + T["app2"] + T["app3"]).sum().backward()
    out["6 standstill: finite grad, app1"] = [bool(torch.isfinite(p.grad).all()), float(T["app1"])]
    assert out["6 standstill: finite grad, app1"][0]
    yaw = np.radians(20.0)
    T = terms(torch.stack([4 * t * np.cos(yaw), 4 * t * np.sin(yaw), torch.full_like(t, yaw)], -1)[None], R(0), pk)
    out["7 heading 20 deg: head_e deg"] = float(np.degrees(float(T["head_e"])))
    assert abs(out["7 heading 20 deg: head_e deg"] - 20) < 0.5
    for k_, v_ in out.items():
        print(f"  {k_}: {v_}")
    print("on_approach smoke: PASS")


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        _smoke()
