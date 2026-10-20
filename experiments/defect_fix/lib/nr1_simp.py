"""SIMP (block B of plans/2026-10-20-lon-anchor-prereg.md, supplement 2): two merged S-row hinges replacing four terms of the NTV0 recipe.
Carried by nr1_lat (rm_train --simp-region W / --simp-clear W): `simp_region` / `simp_clear` = 10 x W x term, the trainer multiplies by ho_w and
sums over the pool rows / n_imit, as every NR1 term (W 1 = the recipe's road / agent hinge weight: road_lam = agent_lam = 10).

  region  replaces the S rows' road hinge (Losses43 road_ho: the road-and-lane raster, Pacifica footprint corners, margin 0.3) and lat_onc
          (nr1_lat: the oncoming-direction lane raster at the rear axle and the box centre, depth 0) with ONE hinge on their union: per dense
          step j = 1..40 (0.1 .. 4 s) v_j = max(road_j, onc_j), road_j = mean over the 4 corners of relu(0.3 - sdf_road), onc_j = mean over the
          two points of relu(-sdf_onc) (each raster at its own points, as the two terms read them); term = mean_j v_j. A row with only one of the
          rasters reads that one. Shape-only (nr1_lat.shape_only, the recipe's --shape): no gradient along the heading.
  clear   replaces the S rows' agent hinge (lib/agent_hinge: the ego box with its front moved 0.3 m forward, objects ahead, margin 0.3 in the
          corridor, side margin 0 outside it) and lat_side (nr1_lat: the plain box, objects outside the corridor and not behind, buffer 0.4) with
          ONE object-clearance hinge: per dense step j = 0..40 and object k
            lead  (in the ego's corridor, centre ahead of the plain box centre): relu(0.3 - d_ext), d_ext = the distance of the extended box;
            side  (outside the corridor, centre ahead of -(ego half length + object half length) from the plain box centre): relu(0.4 - d_plain);
          v_j = max over valid objects; term = mean_j v_j. Gradient: shape-only for side objects; for leads the along-heading component is kept
          but only in the shortening direction (each pose's along-heading displacement passes its gradient only when that gradient would
          move the pose backward: `shorten`), so a lead can make the plan brake, never longer.
"""
from __future__ import annotations

import numpy as np

ROAD_M, FRONT_M, SIDE_M = 0.3, 0.3, 0.4


def _shorten():
    import torch

    class Shorten(torch.autograd.Function):
        """Identity forward; backward keeps only the gradient that moves the along-heading coordinate backward (dL/du >= 0)."""
        @staticmethod
        def forward(ctx, u):
            return u.view_as(u)

        @staticmethod
        def backward(ctx, g):
            return g.clamp_min(0.0)
    return Shorten


_S = None


def lead_poses(x, y, psi):
    """Plan poses for the lead part: the shape-only displacement (across the heading) + the along-heading displacement through `shorten`."""
    global _S
    import torch
    if _S is None:
        _S = _shorten()
    xd, yd, pd = x.detach(), y.detach(), psi.detach()
    c, s = torch.cos(pd), torch.sin(pd)
    n_x, n_y = -s, c
    lat = n_x * (x - xd) + n_y * (y - yd)
    u = _S.apply(c * (x - xd) + s * (y - yd))
    return xd + n_x * lat + c * u, yd + n_y * lat + s * u


def _dist(ah, P, rows, front):
    """Dense poses P (B, 41, 3) -> signed distance (B, 41, K) of the ego box (front moved forward by `front`) to the object boxes, the object
    centre's longitudinal / lateral offset from the PLAIN box centre, the in-corridor mask, the valid mask (= agent_hinge.distances)."""
    import torch
    from agent_hinge import FRONT, REAR, HALF_W, _box_sdf, _corners
    bx = torch.einsum("kj,bjoc->bkoc", ah.M, ah.box[rows])
    valid = torch.einsum("kj,bjo->bko", ah.Mv, (~ah.val[rows]).float()) < 0.5
    f = FRONT + front
    e_off, e_hl = (f + REAR) / 2, (f - REAR) / 2
    ex, ey, eps = P[..., 0:1], P[..., 1:2], P[..., 2:3]
    c, s = torch.cos(eps), torch.sin(eps)
    ecx, ecy = ex + c * e_off, ey + s * e_off
    hl_e, hw_e = torch.full_like(ecx, e_hl), torch.full_like(ecx, HALF_W)
    ox, oy, oyaw, ohl, ohw = bx[..., 0], bx[..., 1], bx[..., 2], bx[..., 3] / 2, bx[..., 4] / 2
    qx, qy = _corners(ecx, ecy, eps, hl_e, hw_e)
    d1 = _box_sdf(qx, qy, ox[..., None], oy[..., None], oyaw[..., None], ohl[..., None], ohw[..., None]).amin(-1)
    px, py = _corners(ox, oy, oyaw, ohl, ohw)
    d2 = _box_sdf(px, py, ecx[..., None], ecy[..., None], eps[..., None], hl_e[..., None], hw_e[..., None]).amin(-1)
    c0 = (FRONT + REAR) / 2
    rx, ry = ox - (ex + c * c0), oy - (ey + s * c0)
    lon = c * rx + s * ry
    lc = -s[..., None] * (px - ex[..., None]) + c[..., None] * (py - ey[..., None])
    corridor = (lc.amax(-1) > -HALF_W) & (lc.amin(-1) < HALF_W)
    return torch.minimum(d1, d2), lon, corridor, valid, ohl


def clear_steps(x, y, psi, rows, ah, parts=False):
    """x, y, psi (B, 8) plan poses (full gradient) -> (B, 41) per-step clearance violation (module docstring); parts: + lead-part (B, 41)."""
    import torch
    from agent_hinge import FRONT, REAR
    import nr1_lat
    xs, ys = nr1_lat.shape_only(x, y, psi)
    xl, yl = lead_poses(x, y, psi)
    z = lambda X, Y: torch.einsum("kj,bjc->bkc", ah.M, torch.cat([torch.zeros(len(X), 1, 3, device=X.device), torch.stack([X, Y, psi], -1)], 1))  # noqa: E731
    Ps, Pl = z(xs, ys), z(xl, yl)
    dp, lon, cor, val, ohl = _dist(ah, Ps, rows, 0.0)                              # plain box, shape-only poses: the side objects
    de, lon_e, _, _, _ = _dist(ah, Pl, rows, FRONT_M)                               # extended box, lead poses: the leads
    with torch.no_grad():
        lead = val & cor & (lon > 0)
        side = val & ~cor & (lon > -((FRONT - REAR) / 2 + ohl))
    vl = (FRONT_M - de).clamp_min(0).masked_fill(~lead, 0.0)
    vs = (SIDE_M - dp).clamp_min(0).masked_fill(~side, 0.0)
    v = torch.maximum(vl, vs).amax(-1)
    return (v, vl.amax(-1)) if parts else v


def clear(poses, rows, ah, ok):
    """(n,) mean over the 41 steps; rows without labels 0."""
    p = poses.float()
    return clear_steps(p[..., 0], p[..., 1], p[..., 2], rows, ah).mean(1) * ok.to(p.dtype)


def region_steps(poses, rows, pack, parts=False):
    """(n, 40) union violation per step (module docstring); parts: + (road part, onc part)."""
    import torch
    import nr1_lat
    p = poses.float()
    xs, ys = nr1_lat.shape_only(p[..., 0], p[..., 1], p[..., 2])
    n = len(p)
    road = torch.zeros(n, 40, device=p.device)
    R, r0 = pack["road"], pack["road_on0"]
    rk = pack["road_ok"][rows]
    if rk.any():
        m = R.margins(xs[rk], ys[rk], p[rk][..., 2], rows[rk] + r0).view(int(rk.sum()), 41, -1)[:, 1:]   # (k, 40, 4) footprint corners
        road[rk] = torch.relu(ROAD_M - m).mean(-1)
    onc = torch.zeros(n, 40, device=p.device)
    if "onc" in pack:
        ok = pack["ok"][rows]
        if ok.any():
            d = nr1_lat.dense(p[ok], pack["M"])
            m = nr1_lat.region_m(d, rows[ok], pack, "onc")                              # (k, 80) or (k, 40): rear axle [+ box centre]
            onc[ok] = torch.relu(-m).view(int(ok.sum()), -1, 40).mean(1)
    v = torch.maximum(road, onc)
    return (v, road, onc) if parts else v


def region(poses, rows, pack):
    return region_steps(poses, rows, pack).mean(1) * pack["self"][rows].to(poses.dtype)


if __name__ == "__main__":                      # known-answer checks (CPU, a synthetic AgentHinge)
    import sys
    from pathlib import Path
    import torch
    sys.path[:0] = [str(Path(__file__).resolve().parents[3] / "lib"), str(Path(__file__).resolve().parent)]
    from drivable_hinge import interp_matrix
    x = torch.zeros(1, 8, requires_grad=True)                                        # the shorten rule on its own
    y, psi = torch.zeros(1, 8, requires_grad=True), torch.zeros(1, 8)
    xl, yl = lead_poses(x, y, psi)
    (-xl.sum() + yl.sum()).backward()                                                # a loss that wants the poses longer (and left)
    assert x.grad.abs().max() == 0 and (y.grad == 1).all(), (x.grad, y.grad)          # lengthening blocked, lateral kept
    x.grad = None
    lead_poses(x, y, psi)[0].sum().backward()
    assert (x.grad == 1).all()                                                        # shortening passes
    M = torch.as_tensor(interp_matrix(), dtype=torch.float32)
    ah = type("AH", (), {})()
    ah.M, ah.Mv = M, (M > 0).float()
    box = torch.zeros(1, 9, 2, 5)
    box[0, :, 0] = torch.tensor([16.0, 0.0, 0.0, 4.5, 1.9])                          # a standing lead 16 m ahead in the lane (rear at 13.75 m)
    box[0, :, 1] = torch.tensor([6.0, 2.4, 0.0, 4.5, 1.9])                           # a parked car beside the lane (0.3 m gap to the plain box)
    ah.box, ah.val = box, torch.ones(1, 9, 2, dtype=torch.bool)
    t = torch.linspace(0.5, 4.0, 8)
    x = (2.5 * t)[None].requires_grad_(True)                                         # 10 m in 4 s: the front 0.3 m into the lead's rear
                                                                                     # (a deep overlap would want it longer: blocked)
    y = torch.zeros(1, 8, requires_grad=True)
    v, vl = clear_steps(x, y, psi, torch.tensor([0]), ah, parts=True)
    v.mean().backward()
    assert vl.max() > 0 and x.grad[0, -1] > 0 and (x.grad >= 0).all(), x.grad       # descent shortens; never lengthens
    assert y.grad[0, :3].abs().max() > 0 and (y.grad[0, :3] >= 0).all(), y.grad     # dL/dy > 0: descent moves the early poses right, away from the car
    x3 = (1.0 * t)[None].requires_grad_(True)
    v3, vl3 = clear_steps(x3, torch.zeros(1, 8), psi, torch.tensor([0]), ah, parts=True)
    assert vl3.max() == 0                                                             # slow plan: the lead never fires
    print("nr1_simp known-answer ok", float(v.mean()))
