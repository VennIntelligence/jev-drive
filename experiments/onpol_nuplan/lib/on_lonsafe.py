"""ROUTE-FOLLOW longitudinal safety terms on the on-policy rows (experiments/route_follow/plans/2026-10-13-lon-prereg.md). The road / agent
hinges of the S recipe are shape-only (zero gradient along the heading), so on an on-policy row nothing but the imitation target sets the
plan's speed; these three terms give the row a reason to brake. Timing-only: the poses enter through on_lon.timing_only (values unchanged,
the gradient of each pose keeps only its component along its own detached heading, the yaw gets none), i.e. they move the speed profile,
never the path. An `--on-lon` module (plans/on_pool_format.md "Optional longitudinal labels"); rm_train.py turns it on with --lon-lead /
--lon-curve / --lon-stop (any > 0; then --on-lon-lam is 1 and the weights below carry the scale).

Trainer hook:
    configure(lead=, curve=, stop=)               # the three weights (default 0: the term is not returned, nothing changes)
    pack = on_lonsafe.load(pool_dir, names, dev)  # <POOL>/agent.npz + tab fut, in the trainer's pool row order (compared with / without `on:`)
    L = on_lonsafe.loss(poses, rows, pack)        # poses (n, 8, 3) rear-axle x, y, yaw at 0.5 .. 4 s in each row's own frame (with grad),
                                                  # rows (n,) pool row indices -> {"lead", "curve", "stop"}: (n,) per-row weighted losses; the
                                                  # trainer adds ho_w x lam x sum / n_imit
`terms(poses, rows, pack)` returns the unweighted per-row hinges and the per-step tensors (firing reads, unit test).

Notation: arc_k = the plan's cumulative arc length at pose k (k = 1 .. 8, t_k = 0.5 k; arc_0 = 0 at the rear axle), with gradient (timing-only);
v_k = the plan's speed at pose k (mean of the two adjacent 0.5 s segments, one-sided at k = 8), with gradient. The corridor path = the plan's own
detached polyline [origin, 8 poses] extended EXT = 20 m beyond the last pose along its yaw (a stopped plan's corridor is the line ahead).
  lead  (m^2).  Per actor j of the row's agent.npz (valid at t and t + 0.5 s, class >= 0): constant-velocity box c_j(t) = c_j(0) + v_j t,
        v_j = (c_j(0.5) - c_j(0)) / 0.5, yaw and size of t = 0. At each t_k its 4 corners are projected onto the corridor path: station s (m along
        the path from the rear axle) and signed lateral offset; the box is IN the corridor at t_k if a corner is within LAT = 1.0 m of the path
        or the corners straddle it. sigma_j(t) = min corner station. Eligible: sigma_j(0) >= FRONT (wholly ahead of the ego's front bumper at
        the row's time). Gap g_k = G0 + G1 v_k = 2 m + 0.8 s x v_k, relaxed where the row starts closer: g_eff = min(g_k, max(sigma_j(0) -
        FRONT - 0.5, 0)) (never asks a row for more gap than it has); overshoot o_jk = relu(arc_k + FRONT + g_eff - sigma_j(t_k)) on in-corridor
        steps; h = mean_k max_j o_jk^2.
  curve (m^2/s^4).  Lateral acceleration a_k = v_k^2 kappa_k, kappa_k = |wrap(h_{k+1} - h_{k-1})| / (arc_{k+1} - arc_{k-1}) of the detached
        path (h = the path's own direction by central differences, h_0 = 0), k = 1 .. 7, 0 where the two segments span < 0.5 m;
        h = mean_k relu(a_k - A_MAX)^2, A_MAX = 3.0 m/s^2. Only v carries a gradient: the term brakes, it does not straighten.
  stop  (m^2).  Rows whose target `fut` is stopped (its arc at 4 s < STOP_ARC = 2 m): h = relu(arc_8 - (fut arc_8 + STOP_SLACK))^2, slack 2 m.
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from on_lon import FRONT, timing_only  # noqa: E402

TERMS = ("lead", "curve", "stop")
W = dict(lead=0.0, curve=0.0, stop=0.0)
LAT, EXT, G0, G1, G_BACK = 1.0, 20.0, 2.0, 0.8, 0.5
A_MAX, KAPPA_MIN_DS = 3.0, 0.5
STOP_ARC, STOP_SLACK = 2.0, 2.0
TK = 0.5 * torch.arange(1, 9, dtype=torch.float32)
EPS = 1e-9


def configure(lead=0.0, curve=0.0, stop=0.0):
    W.update(lead=float(lead), curve=float(curve), stop=float(stop))
    return dict(W)


def _bare(n: str) -> str:
    return n[3:] if n.startswith("on:") else n


def arc_of(xy):
    """(n, 8, 2) poses -> (n, 8) cumulative arc length from the origin."""
    p = torch.cat([xy.new_zeros(len(xy), 1, 2), xy], 1)
    return torch.sqrt(((p[:, 1:] - p[:, :-1]) ** 2).sum(-1) + EPS).cumsum(1)


def build(box, valid, cls, fut, ok=None):
    """numpy agent boxes (N, 9, K, 5) (centre x, y, yaw, length, width at t + 0, 0.5 .. 4 s, own frame), valid (N, 9, K), cls (N, K),
    target fut (N, 8, 3) -> the pack arrays."""
    aok = valid[:, 0] & valid[:, 1] & (cls >= 0)
    if ok is not None:
        aok &= ok[:, None]
    c0 = np.where(aok[..., None], box[:, 0, :, :2], 0).astype(np.float32)
    vel = np.where(aok[..., None], (box[:, 1, :, :2] - box[:, 0, :, :2]) / 0.5, 0).astype(np.float32)
    f = np.concatenate([np.zeros((len(fut), 1, 2), np.float32), fut[..., :2].astype(np.float32)], 1)
    farc = np.sqrt((np.diff(f, axis=1) ** 2).sum(-1)).cumsum(1).astype(np.float32)
    return dict(c0=c0, vel=vel, yaw=np.where(aok, box[:, 0, :, 2], 0).astype(np.float32), hl=np.where(aok, box[:, 0, :, 3] / 2, 0).astype(np.float32),
                hw=np.where(aok, box[:, 0, :, 4] / 2, 0).astype(np.float32), aok=aok, farc=farc, stop=farc[:, -1] < STOP_ARC)


def load(pool_dir, names, dev) -> dict:
    """<POOL>/agent.npz + <POOL>/tab.npz (fut) -> tensors in the trainer's pool row order (names (N,)); every name must be labelled."""
    pool_dir = Path(pool_dir)
    za, zt = np.load(pool_dir / "agent.npz"), np.load(pool_dir / "tab.npz")
    want = [_bare(str(n)) for n in np.asarray(names).tolist()]
    pa = {_bare(str(t)): i for i, t in enumerate(za["tokens"].tolist())}
    pt = {_bare(str(t)): i for i, t in enumerate(zt["names"].tolist())}
    miss = [n for n in want if n not in pa or n not in pt]
    assert not miss, f"{pool_dir}: {len(miss)} pool rows without agent / tab rows, e.g. {miss[:2]}"
    ia, it = np.array([pa[n] for n in want]), np.array([pt[n] for n in want])
    Z = build(za["box"][ia], za["valid"][ia], za["cls"][ia], zt["fut"][it], za["ok"][ia])
    print(f"on_lonsafe: {pool_dir}: {len(want)} rows, actors eligible for CV {Z['aok'].sum(1).mean():.1f} / row, stopped targets "
          f"{Z['stop'].mean():.4f}; weights {W}", flush=True)
    return {k: torch.as_tensor(v, device=dev) for k, v in Z.items()}


def _project(V, S, Q):
    """Polyline V (n, NV, 2) with cumulative station S (n, NV), points Q (n, P, 2) -> distance to the polyline, station, signed lateral
    (left +), all (n, P); nearest segment by the clamped distance. No gradient."""
    A, u = V[:, :-1], V[:, 1:] - V[:, :-1]                                    # (n, NS, 2)
    L2 = (u ** 2).sum(-1).clamp_min(EPS)
    d = Q[:, :, None] - A[:, None]                                            # (n, P, NS, 2)
    t = ((d * u[:, None]).sum(-1) / L2[:, None]).clamp(0, 1)
    dist = ((d - t[..., None] * u[:, None]) ** 2).sum(-1).sqrt()
    i = dist.argmin(-1, keepdim=True)
    g = lambda z: z.gather(-1, i)[..., 0]  # noqa: E731
    Ln = L2.sqrt()
    lat = (u[:, None, :, 0] * d[..., 1] - u[:, None, :, 1] * d[..., 0]) / Ln[:, None]
    st = S[:, :-1][:, None] + t * Ln[:, None]
    return g(dist), g(st), g(lat)


def _corners(c, yaw, hl, hw):
    """c (..., 2), yaw / hl / hw (...) -> (..., 4, 2)."""
    cy, sy = torch.cos(yaw)[..., None], torch.sin(yaw)[..., None]
    a = torch.tensor([1., 1., -1., -1.], device=c.device) * hl[..., None]
    b = torch.tensor([1., -1., -1., 1.], device=c.device) * hw[..., None]
    return torch.stack([c[..., None, 0] + a * cy - b * sy, c[..., None, 1] + a * sy + b * cy], -1)


def terms(poses, rows, pack):
    """-> per-row (n,) unweighted lead / curve / stop + `steps` (per-step tensors and masks)."""
    poses = poses.float()
    x, y, psi = poses.unbind(-1)
    xt, yt = timing_only(x, y, psi)
    arc = arc_of(torch.stack([xt, yt], -1))                                   # (n, 8) with the along-heading gradient
    seg = torch.diff(torch.cat([arc.new_zeros(len(arc), 1), arc], 1), dim=1) / 0.5
    v = torch.cat([0.5 * (seg[:, :-1] + seg[:, 1:]), seg[:, -1:]], 1)         # (n, 8) speed at the poses
    n, dev = len(x), x.device
    g = lambda k: pack[k][rows]  # noqa: E731
    # ---- lead
    with torch.no_grad():
        P = torch.cat([poses.new_zeros(n, 1, 2), poses[..., :2].detach()], 1)               # (n, 9, 2)
        pd = psi.detach()[:, -1]
        V = torch.cat([P, (P[:, -1] + EXT * torch.stack([torch.cos(pd), torch.sin(pd)], -1))[:, None]], 1)   # (n, 10, 2)
        S = torch.cat([P.new_zeros(n, 1), ((V[:, 1:] - V[:, :-1]) ** 2).sum(-1).sqrt().cumsum(1)], 1)
        c0, vel, aok = g("c0"), g("vel"), g("aok")                            # (n, K, 2), (n, K, 2), (n, K)
        K = c0.shape[1]
        tt = torch.cat([TK.new_zeros(1), TK]).to(dev)                         # t = 0, 0.5 .. 4
        C = c0[:, :, None] + vel[:, :, None] * tt[None, None, :, None]        # (n, K, 9, 2)
        ex = lambda z: z[:, :, None].expand(-1, -1, 9)  # noqa: E731
        Q = _corners(C, ex(g("yaw")), ex(g("hl")), ex(g("hw")))               # (n, K, 9, 4, 2)
        dist, st, lat = (z.view(n, K, 9, 4) for z in _project(V, S, Q.reshape(n, -1, 2)))
        inc = (dist.amin(-1) <= LAT) | ((lat.amin(-1) < 0) & (lat.amax(-1) > 0) & (dist.amax(-1) <= 2 * (g("hl") + g("hw"))[..., None]))
        sig = st.amin(-1)                                                     # (n, K, 9)
        elig = aok & (sig[..., 0] >= FRONT)
        act = elig[..., None] & inc[..., 1:]                                  # (n, K, 8)
        room = (sig[..., 0] - FRONT - G_BACK).clamp_min(0)[..., None]         # (n, K, 1)
    geff = torch.minimum(G0 + G1 * v[:, None], room)                         # (n, K, 8), gradient through v
    over = torch.relu(arc[:, None] + FRONT + geff - sig[..., 1:])
    o2 = torch.where(act, over, torch.zeros_like(over)) ** 2
    lead_k = o2.amax(1) if K else arc * 0                                     # (n, 8)
    # ---- curve
    with torch.no_grad():
        Pd = P
        dpx, dpy = Pd[:, 2:, 0] - Pd[:, :-2, 0], Pd[:, 2:, 1] - Pd[:, :-2, 1]   # central differences at poses 1 .. 7
        hc = torch.atan2(dpy, dpx)                                            # (n, 7)
        hl_ = torch.atan2(Pd[:, -1, 1] - Pd[:, -2, 1], Pd[:, -1, 0] - Pd[:, -2, 0])[:, None]
        h = torch.cat([P.new_zeros(n, 1), hc, hl_], 1)                        # path direction at poses 0 .. 8
        a0 = torch.cat([P.new_zeros(n, 1), arc.detach()], 1)                  # arc at poses 0 .. 8
        ds = a0[:, 2:] - a0[:, :-2]                                           # (n, 7) for poses 1 .. 7
        dh = torch.atan2(torch.sin(h[:, 2:] - h[:, :-2]), torch.cos(h[:, 2:] - h[:, :-2])).abs()
        kap = torch.where(ds >= KAPPA_MIN_DS, dh / ds.clamp_min(EPS), torch.zeros_like(ds))
    alat = v[:, :7] ** 2 * kap
    curve_k = torch.relu(alat - A_MAX) ** 2                                   # (n, 7)
    # ---- stop
    stop = torch.where(g("stop"), torch.relu(arc[:, -1] - g("farc")[:, -1] - STOP_SLACK) ** 2, torch.zeros_like(arc[:, -1]))
    return dict(lead=lead_k.mean(1), curve=curve_k.mean(1), stop=stop,
                steps=dict(lead=lead_k, curve=curve_k, alat=alat.detach(), arc=arc.detach(), v=v.detach(), act=act, elig=elig))


def loss(poses, rows, pack):
    """The trainer hook: per-row (n,) weighted losses of the terms with weight > 0."""
    T = terms(poses, rows, pack)
    return {k: W[k] * T[k] for k in TERMS if W[k] > 0}


# ---------------------------------------------------------------- unit test (CPU, synthetic rows): python on_lonsafe.py --smoke
def _smoke():
    t = TK

    def straight(v):
        return torch.stack([v * t, torch.zeros_like(t), torch.zeros_like(t)], -1)

    def arc_plan(v, R):                                                       # left circle of radius R at speed v
        th = v * t / R
        return torch.stack([R * torch.sin(th), R * (1 - torch.cos(th)), th], -1)
    K = 3
    N = 6
    box = np.zeros((N, 9, K, 5), np.float32)
    valid = np.zeros((N, 9, K), bool)
    cls = np.full((N, K), -1, np.int8)
    fut = np.stack([straight(8.0).numpy()] * N)
    # row 0: a static vehicle whose rear is 10 m ahead of the rear axle (centre at 12.4 m), the plan runs at 8 m/s straight
    box[0, :, 0] = [12.4, 0, 0, 4.8, 2.0]; valid[0, :, 0] = True; cls[0, 0] = 0
    # row 1: the same vehicle 3 m to the left (adjacent lane): never in the corridor
    box[1, :, 0] = [12.4, 3.0, 0, 4.8, 2.0]; valid[1, :, 0] = True; cls[1, 0] = 0
    # row 2: a lead 20 m ahead moving at 8 m/s: the plan at 8 m/s keeps 20 - 4.05 m > 2 + 6.4 m, no overshoot
    for k in range(9):
        box[2, k, 0] = [22.4 + 8.0 * 0.5 * k, 0, 0, 4.8, 2.0]
    valid[2, :, 0] = True; cls[2, 0] = 0
    # row 3: no actor, tight curve R = 10 m at 8 m/s (6.4 m/s^2 > 3); row 4: the same curve at 4 m/s (1.6 m/s^2)
    # row 5: the target is stopped (fut at the origin), the plan runs at 4 m/s
    fut[5] = 0.0
    pack = {k: torch.as_tensor(v) for k, v in build(box, valid, cls, fut).items()}
    plans = torch.stack([straight(8.0), straight(8.0), straight(8.0), arc_plan(8.0, 10.0), arc_plan(4.0, 10.0), straight(4.0)]).requires_grad_(True)
    T = terms(plans, torch.arange(N), pack)
    out = {k: [round(float(v), 4) for v in T[k].detach()] for k in TERMS}
    tot = T["lead"].sum() + T["curve"].sum() + T["stop"].sum()
    tot.backward()
    G = plans.grad
    u = torch.stack([torch.cos(plans[..., 2]), torch.sin(plans[..., 2])], -1).detach()
    along = (G[..., :2] * u).sum(-1)                                          # dL / d(along-heading displacement) per pose
    normal = (G[..., :2] * torch.stack([-u[..., 1], u[..., 0]], -1)).sum(-1)
    for k, v in out.items():
        print(f"  {k}: {v}")
    print("  along-heading dL/ds per row (sum over poses):", [round(float(a), 3) for a in along.sum(1)])
    assert out["lead"][0] > 0 and out["lead"][1] == 0 and out["lead"][2] == 0, "lead fires on row 0 only"
    assert out["curve"][3] > 0 and out["curve"][4] == 0 and out["curve"][0] == 0, "curve fires on row 3 only"
    assert out["stop"][5] > 0 and sum(out["stop"][:5]) == 0, "stop fires on row 5 only"
    for r in (0, 5):                                                          # lead / stop: every pose's descent (-grad) points backwards
        assert bool((along[r] >= -1e-6).all()) and float((-along[r]).sum()) < 0, f"row {r}: along-heading gradient {along[r]}"
    assert float((-along[3]).sum()) < 0, "curve: net descent backwards (per pose it is a speed term: earlier pose pulled on, later held back)"
    with torch.no_grad():                                                     # one small descent step: each firing term and the 4 s arc drop
        T2 = terms(plans - 1e-3 * G / G.norm(dim=(1, 2), keepdim=True).clamp_min(1e-12), torch.arange(N), pack)
    for r, k in ((0, "lead"), (3, "curve"), (5, "stop")):
        assert float(T2[k][r]) < float(T[k][r].detach()) and float(T2["steps"]["arc"][r, -1]) < float(T["steps"]["arc"][r, -1]), f"row {r} {k}: descent step"
    assert float(along[[1, 2, 4]].abs().max()) == 0, "rows that do not fire get no gradient"
    assert float(normal.abs().max()) < 1e-5 and float(G[..., 2].abs().max()) == 0, "timing-only: no lateral / yaw gradient"
    configure(lead=2.0, curve=0.0, stop=1.0)
    L = loss(plans, torch.arange(N), pack)
    assert set(L) == {"lead", "stop"} and torch.allclose(L["lead"], 2 * T["lead"]), "weights / zero-weight terms"
    configure()
    assert loss(plans, torch.arange(N), pack) == {}, "default weights: nothing returned"
    print("on_lonsafe smoke: PASS")


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        _smoke()
