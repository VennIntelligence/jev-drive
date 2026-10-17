"""NR1 hard / soft lateral constraint terms on the on-policy pool rows (rm_train --lat-onc / --lat-soft / --lat-side; prereg
experiments/defect_fix/plans/2026-10-17-nr1-prereg.md section 3.2, the user's hierarchy: kerb / other vehicles / opposite-direction lanes hard,
same-direction empty lanes soft; no target pose, no pull toward a centre line).

Through on_train's --on-lon interface (rm_train --lat-* sets --on-lon nr1_lat, --on-lon-lam 1): load(pool_dir, names, dev) -> pack;
loss(poses, rows, pack) -> {term: (n,)}, each already x 10 x its weight (the road hinge's lam 10 = weight 1, decision 281's scale: its oncoming
hinge at 0.25 was --on-lon-lam 2.5); the trainer multiplies by ho_w and sums over the pool rows / n_imit, as every --on-lon term. The kerb is the
recipe's existing road hinge and vehicle contact its existing agent hinge; both stay as they are.

  onc   the rear axle and the box centre (1.461 m ahead of it; the rear axle only when the raster dir's meta says points = ra, prereg amendment 1)
        inside an opposite-direction lane, judged per plan point at the plan's own heading:
        DIR/lat.npz (df_lat.py build) holds per row and per heading bin (-90 .. +90 deg in 30 deg steps, the row's t0 heading = 0) the signed
        distance (m, + outside) to the cells that zeros_mech's Lanes.cls rule calls oncoming at that heading (the evaluator's lane polygons and
        centre-line yaw; some containing lane within 60 deg wins, else one beyond 120 deg), junction cells (two containing lanes more than 30 deg
        apart) left out; the point reads the bin of its own heading (clamped to +-90 deg). Term: relu(-sdf - depth), mean over the 40 dense steps
        after t0 (0.1 .. 4 s) and the two points.
  soft  the same on the soft raster: same-direction lanes (within 60 deg at the bin heading) that are not the ego's own lane chain (the lanes
        holding the rear axle at t0 with their successors / predecessors), junction cells left out.
  side  the agent hinge's side extension: logged objects outside the ego's lateral corridor and not behind it (centre ahead of -(ego half length +
        object half length) from the ego box centre), every class lib/agent_hinge.AgentHinge keeps, plain ego box: relu(buf - distance), max over
        objects, mean over the 40 dense steps. The existing agent hinge (margin 0.3 ahead in the corridor, contact only beside) is unchanged.
Every term moves the plan's poses only across their own heading (loss43's shape_only; the recipe runs --shape).
  arc   (NR1HVR, rm_train --on-arc W; prereg supplement 8) the two-sided longitudinal term of nr1_lon.py, carried here because the pool has one
        --on-lon module: logged as lon_arc, 10 x W x nr1_lon.term; it moves the poses only along their own heading. W = 0: nothing loaded or added.
"""
from __future__ import annotations

import numpy as np

BC = (4.049 - 1.127) / 2                       # box centre ahead of the rear axle (nuPlan Pacifica), = the scorer's centroid offset
BIN_DEG = 30.0
CFG = dict(onc=0.0, soft=0.0, side=0.0, depth=0.0, soft_depth=0.0, buf=0.4, file="", arc=0.0, host=False, arc_rows="move")


def configure(onc=0.0, soft=0.0, side=0.0, depth=0.0, soft_depth=0.0, buf=0.4, file="", arc=0.0, host=False, arc_rows="move"):
    """host (rm_train --lat-host, JT): the rasters stay a host memmap and each batch's rows are gathered to the card (same values; a pool too big
    for the card)."""
    CFG.update(onc=float(onc), soft=float(soft), side=float(side), depth=float(depth), soft_depth=float(soft_depth), buf=float(buf), file=str(file),
               arc=float(arc), host=bool(host), arc_rows=str(arc_rows))
    return dict(CFG)


def shape_only(x, y, psi):
    """= body1 loss43.shape_only (importing loss43 pulls pp_train)."""
    import torch
    xd, yd, pd = x.detach(), y.detach(), psi.detach()
    nx, ny = -torch.sin(pd), torch.cos(pd)
    s = nx * (x - xd) + ny * (y - yd)
    return xd + nx * s, yd + ny * s


def load(pool_dir, names, dev, file=None):
    """-> pack: the interpolation matrix, the lateral rasters of DIR (df_lat.py build: meta.npz, onc.npy, soft.npy) on `dev` for the terms with
    a weight (all with file=DIR: the label gate), the pool's agent boxes (plain ego box) for the side term."""
    import torch
    from pathlib import Path
    from drivable_hinge import interp_matrix
    names = np.asarray(names).astype(str)
    pack = dict(M=torch.as_tensor(interp_matrix(), dtype=torch.float32, device=dev), dev=dev)
    f = Path(file or CFG["file"]) if (file or CFG["file"]) else None
    if CFG["onc"] > 0 or CFG["soft"] > 0 or file:
        z = np.load(f / "meta.npz")
        nr = len(z["names"])                                                   # JT (df_lat build --prefix): the raster may cover the pool's first rows
        assert np.array_equal(z["names"].astype(str), names[:nr]) and (nr == len(names) or CFG["host"]), f"{f}: rows != the pool's"
        assert np.array_equal(z["bins"], np.arange(-90.0, 91.0, BIN_DEG)), "heading bins"
        pack["bins"] = torch.as_tensor(z["bins"].astype(np.float32), device=dev)
        pack["ok"] = torch.as_tensor(np.r_[z["ok"].astype(bool), np.zeros(len(names) - nr, bool)], device=dev)
        pack["points"] = str(z["points"]) if "points" in z.files else "both"      # prereg amendment 1: "ra" = the rear axle only
        for k in ("onc", "soft"):
            if (CFG[k] > 0 or file) and CFG["host"]:
                pack[k] = np.load(f / f"{k}.npy", mmap_mode="r")
            elif CFG[k] > 0 or file:
                a = np.load(f / f"{k}.npy", mmap_mode="r")
                t = torch.empty(a.shape, dtype=torch.float16, device=dev)
                for i in range(0, len(a), 2048):
                    t[i:i + 2048] = torch.from_numpy(np.ascontiguousarray(a[i:i + 2048])).to(dev)
                pack[k] = t
    if CFG["side"] > 0 or file:
        from agent_hinge import AgentHinge
        pack["ah"] = AgentHinge(Path(pool_dir) / "agent.npz", names, dev, 0.0, None)
    if CFG["arc"] > 0 or CFG["arc_rows"] != "move":                            # NR1HVR: the longitudinal labels from the pool's tab (JTB / JTN:
        import nr1_lon                                                         # --on-arc-rows loads them for the monitoring even at W 0)
        pack["arc"] = nr1_lon.load(pool_dir, names, dev, CFG["arc_rows"])
        print(f"nr1_lat: lon_arc labels {pack['arc']['rec']}", flush=True)
    return pack


def dense(poses, M):
    """(n, 8, 3) rear-axle poses (shape-only x, y) -> (n, 41, 3) at 0 .. 4 s every 0.1 s."""
    import torch
    x, y, psi = poses[..., 0], poses[..., 1], poses[..., 2]
    xs, ys = shape_only(x, y, psi)
    P0 = torch.cat([torch.zeros_like(poses[:, :1]), torch.stack([xs, ys, psi], -1)], 1)
    return torch.einsum("kj,bjc->bkc", M, P0)


def region_m(d, rows, pack, key):
    """Dense poses (n, 41, 3) -> (n, 80) signed distance of the rear axle and the box centre at steps 1..40, each read at its heading's bin."""
    import torch
    import torch.nn.functional as F
    c, s = torch.cos(d[..., 2]), torch.sin(d[..., 2])
    P = torch.cat([d[:, 1:, :2], torch.stack([d[:, 1:, 0] + BC * c[:, 1:], d[:, 1:, 1] + BC * s[:, 1:]], -1)], 1)     # (n, 80, 2)
    h = torch.cat([d[:, 1:, 2], d[:, 1:, 2]], 1).detach()
    nb = len(pack["bins"])
    b = torch.round(torch.rad2deg(h).clamp(-90.0, 90.0) / BIN_DEG).long() + nb // 2
    g = torch.stack([(P[..., 1] + 24.0) / 24.0 - 1.0, (P[..., 0] + 8.0) / 32.0 - 1.0], -1)[:, :, None]           # drivable_hinge's X0 / Y0
    m = F.grid_sample(rows_of(pack[key], rows).float(), g, mode="bilinear", padding_mode="border", align_corners=False)[..., 0]   # (n, B, 80)
    m = m.gather(1, b[:, None]).squeeze(1)
    return m[:, :m.shape[1] // 2] if pack.get("points", "both") == "ra" else m


def rows_of(R, rows):
    """R[rows]: a card tensor, or (CFG host) a host memmap gathered for these rows (rows past its end read its last row; their `ok` is False)."""
    import torch
    if torch.is_tensor(R):
        return R[rows]
    ri = np.minimum(rows.detach().cpu().numpy(), len(R) - 1)
    u, inv = np.unique(ri, return_inverse=True)
    t = torch.from_numpy(np.ascontiguousarray(R[u])).to(rows.device, non_blocking=True)
    return t[torch.as_tensor(inv.reshape(-1), device=rows.device)]


def region(d, rows, pack, key, depth):
    """(n,) mean over steps 1..40 and the two points of relu(-sdf - depth); rows without a raster 0."""
    import torch
    return torch.relu(-region_m(d, rows, pack, key) - depth).mean(1) * pack["ok"][rows].float()


def side_dist(d, rows, ah):
    """Dense poses (n, 41, 3) -> (n, 40) distance of the plain ego box to the nearest side object at steps 1..40 (module docstring; 1e3 when
    none is counted)."""
    import torch
    from agent_hinge import _box_sdf, _corners
    bx = torch.einsum("kj,bjoc->bkoc", ah.M, ah.box[rows])
    valid = torch.einsum("kj,bjo->bko", ah.Mv, (~ah.val[rows]).float()) < 0.5
    ex, ey, eps = d[..., 0:1], d[..., 1:2], d[..., 2:3]
    c, s = torch.cos(eps), torch.sin(eps)
    ecx, ecy = ex + c * ah.e_off, ey + s * ah.e_off
    hl_e, hw_e = torch.full_like(ecx, ah.e_hl), torch.full_like(ecx, ah.e_hw)
    ox, oy, oyaw, ohl, ohw = bx[..., 0], bx[..., 1], bx[..., 2], bx[..., 3] / 2, bx[..., 4] / 2
    qx, qy = _corners(ecx, ecy, eps, hl_e, hw_e)
    d1 = _box_sdf(qx, qy, ox[..., None], oy[..., None], oyaw[..., None], ohl[..., None], ohw[..., None]).amin(-1)
    px, py = _corners(ox, oy, oyaw, ohl, ohw)
    d2 = _box_sdf(px, py, ecx[..., None], ecy[..., None], eps[..., None], hl_e[..., None], hw_e[..., None]).amin(-1)
    dist = torch.minimum(d1, d2)
    lon = c * (ox - ecx) + s * (oy - ecy)
    lc = -s[..., None] * (px - ex[..., None]) + c[..., None] * (py - ey[..., None])
    corridor = (lc.amax(-1) > -ah.e_hw) & (lc.amin(-1) < ah.e_hw)
    counted = valid & ~corridor & (lon > -(ah.e_hl + ohl))
    return dist.masked_fill(~counted, 1e3).amin(-1)[:, 1:]


def side(d, rows, ah, buf):
    """(n,) mean over steps 1..40 of relu(buf - distance to the nearest side object); rows without labels 0."""
    return (buf - side_dist(d, rows, ah)).clamp_min(0).mean(1) * ah.ok[rows].float()


def terms(poses, rows, pack, cfg=None):
    """Raw (unweighted) terms (n,) for the label gate and the loss: onc / soft / side, those with a table loaded."""
    cfg = cfg or CFG
    d = dense(poses, pack["M"])
    out = {}
    if "onc" in pack:
        out["onc"] = region(d, rows, pack, "onc", cfg["depth"])
    if "soft" in pack:
        out["soft"] = region(d, rows, pack, "soft", cfg["soft_depth"])
    if "ah" in pack:
        out["side"] = side(d, rows, pack["ah"], cfg["buf"])
    return out


def loss(poses, rows, pack):
    t = terms(poses, rows, pack)
    out = {f"lat_{k}": 10.0 * CFG[k] * v for k, v in t.items() if CFG[k] > 0}
    if CFG["arc"] > 0:
        import nr1_lon
        out["lon_arc"] = 10.0 * CFG["arc"] * nr1_lon.term(poses, rows, pack["arc"])
    return out
