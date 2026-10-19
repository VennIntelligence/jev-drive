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
  free  (NR1FREE, rm_train --on-free W; prereg supplement 15) the one-sided longitudinal consequence term of nr1_free.py (plan slower than 0.8 x
        the human with the way ahead free): logged as lon_free, 10 x W x nr1_free.term, on the agent hinge's boxes. W = 0: nothing loaded or added.
  stand (NR1STAND, rm_train --on-stand W [--on-stand-margin M]; prereg supplement 17) nr1_free's mirror term on standing rows (the plan's 4 s
        arc beyond the standing human's + M): logged as lon_stand, 10 x W x nr1_free.stand_term. W = 0: nothing loaded or added.
        --on-stand-roll V (STAND-R, supplement 20): also the rolling rows (V_STAND <= v0 < V; labels <pool>/stand_gap.npz); their part is
        logged apart as lon_standr (lon_stand + lon_standr = the term). V = 0: as supplement 17, nothing loaded or added.
  stoppt (NR1STOPPT, rm_train --on-stoppt W; prereg supplement 18) nr1_free's "do not pass the human's stop point" term on approach rows
        (labels <pool>/stoppt.npz): logged as lon_stoppt, 10 x W x nr1_free.stoppt_term. W = 0: nothing loaded or added.
  turncap (TURN-CAP, rm_train --on-turncap W; prereg supplement 21) nr1_free's one-sided upper limit on moving S rows in sharp turns (the plan's
        arc beyond 1.05 x the larger human arc; labels <pool>/turn_cap.npz): logged as lon_turncap, 10 x W x nr1_free.turncap_term. W = 0:
        nothing loaded or added.
  tightin (TIGHT-IN, rm_train --on-tightin W; prereg supplement 22) nr1_tight's two parts on moving S rows in tight turns (labels
        <pool>/tight_in.npz): the plan outside the human's logged line beyond 0.2 m (lon_tightin, across the heading) and TURN-CAP's arc cap
        (lon_tightina, along the heading), each 10 x W x its part. W = 0: nothing loaded or added.
"""
from __future__ import annotations

import numpy as np

BC = (4.049 - 1.127) / 2                       # box centre ahead of the rear axle (nuPlan Pacifica), = the scorer's centroid offset
BIN_DEG = 30.0
CFG = dict(onc=0.0, soft=0.0, side=0.0, depth=0.0, soft_depth=0.0, buf=0.4, file="", arc=0.0, host=False, arc_rows="move", free=0.0)


def configure(onc=0.0, soft=0.0, side=0.0, depth=0.0, soft_depth=0.0, buf=0.4, file="", arc=0.0, host=False, arc_rows="move", free=0.0,
              free_frac=0.8, free_mode="all", stand=0.0, stand_margin=1.0, stoppt=0.0, free_clear=(0.0, 0.0), stand_roll=0.0, turncap=0.0, tightin=0.0):
    """host (rm_train --lat-host, JT): the rasters stay a host memmap and each batch's rows are gathered to the card (same values; a pool too big
    for the card)."""
    CFG.update(onc=float(onc), soft=float(soft), side=float(side), depth=float(depth), soft_depth=float(soft_depth), buf=float(buf), file=str(file),
               arc=float(arc), host=bool(host), arc_rows=str(arc_rows), free=float(free))
    CFG["stand"] = float(stand)
    CFG["_roll"] = float(stand_roll)                                       # NR1 supplement 20 (popped below)
    assert CFG["stand"] > 0 or not CFG["_roll"], "--on-stand-roll: needs --on-stand"
    CFG["_clear"] = tuple(float(x) for x in free_clear)                     # NR1 supplement 19: FREE-M's stricter free check (popped below)
    assert CFG["free"] > 0 or not any(CFG["_clear"]), "--on-free-clear: needs --on-free"
    if turncap > 0:                                                            # TURN-CAP (supplement 21): only the weight, labels from the file
        CFG["turncap"] = float(turncap)
    if tightin > 0:                                                            # TIGHT-IN (supplement 22): only the weight, labels from the file
        CFG["tightin"] = float(tightin)
    if stoppt > 0:                                                             # NR1STOPPT (supplement 18): only the weight, labels from the file
        CFG["stoppt"] = float(stoppt)
        return dict(_configure_rest(free_frac, free_mode, stand_margin), stoppt=CFG["stoppt"])
    return _configure_rest(free_frac, free_mode, stand_margin)


def _configure_rest(free_frac, free_mode, stand_margin):
    clear, roll = CFG.pop("_clear"), CFG.pop("_roll")
    assert CFG["stand"] > 0 or stand_margin == 1.0, "--on-stand-margin: needs --on-stand"
    if CFG["stand"] > 0:                                                       # NR1STAND (supplement 17)
        import nr1_free
        st = dict(stand=CFG["stand"], **nr1_free.configure_stand(stand_margin, roll))
        if CFG["free"] > 0:
            return dict(CFG, **{f"free_{k}": v for k, v in nr1_free.configure(free_frac, free_mode, clear).items()}, **st)
        assert free_frac == 0.8 and free_mode == "all", "--on-free-frac / --on-free-mode: need --on-free"
        return dict({k: v for k, v in CFG.items() if k not in ("free", "stand")}, **st)
    del CFG["stand"]
    if CFG["free"] > 0:                                                        # NR1FREE-M (supplement 16): band and rows of nr1_free
        import nr1_free
        return dict(CFG, **{f"free_{k}": v for k, v in nr1_free.configure(free_frac, free_mode, clear).items()})
    assert free_frac == 0.8 and free_mode == "all", "--on-free-frac / --on-free-mode: need --on-free"
    return {k: v for k, v in CFG.items() if k != "free"}


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
    if CFG["free"] > 0:                                                        # NR1FREE: the reference arcs + the agent hinge's boxes
        import nr1_free
        if "ah" not in pack:
            from agent_hinge import AgentHinge
            pack["ah"] = AgentHinge(Path(pool_dir) / "agent.npz", names, dev, 0.0, None)
        pack["free"] = nr1_free.load(pool_dir, names, dev, pack["ah"])
        print(f"nr1_lat: lon_free labels {pack['free']['rec']}", flush=True)
    if CFG.get("stand", 0.0) > 0:                                              # NR1STAND: the standing rows' caps
        import nr1_free
        pack["stand"] = nr1_free.stand_load(pool_dir, names, dev)
        print(f"nr1_lat: lon_stand labels {pack['stand']['rec']}", flush=True)
    if CFG.get("stoppt", 0.0) > 0:                                             # NR1STOPPT: the approach rows' caps
        import nr1_free
        pack["stoppt"] = nr1_free.stoppt_load(pool_dir, names, dev)
        print(f"nr1_lat: lon_stoppt labels {pack['stoppt']['rec']}", flush=True)
    if CFG.get("turncap", 0.0) > 0:                                            # TURN-CAP: the sharp-turn rows' caps
        import nr1_free
        pack["turncap"] = nr1_free.turncap_load(pool_dir, names, dev)
        print(f"nr1_lat: lon_turncap labels {pack['turncap']['rec']}", flush=True)
    if CFG.get("tightin", 0.0) > 0:                                            # TIGHT-IN: the tight-turn rows' paths and caps
        import nr1_tight
        pack["tightin"] = nr1_tight.load(pool_dir, names, dev)
        print(f"nr1_lat: lon_tightin labels {pack['tightin']['rec']}", flush=True)
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
    if CFG["free"] > 0:
        import nr1_free
        out["lon_free"] = 10.0 * CFG["free"] * nr1_free.term(poses, rows, pack["free"], pack["ah"])
    if CFG.get("stand", 0.0) > 0:
        import nr1_free
        if "roll" in pack["stand"]:                                            # STAND-R: the rolling rows' part logged apart
            vs, vr = nr1_free.stand_term_split(poses, rows, pack["stand"])
            out["lon_stand"], out["lon_standr"] = 10.0 * CFG["stand"] * vs, 10.0 * CFG["stand"] * vr
        else:
            out["lon_stand"] = 10.0 * CFG["stand"] * nr1_free.stand_term(poses, rows, pack["stand"])
    if CFG.get("stoppt", 0.0) > 0:
        import nr1_free
        out["lon_stoppt"] = 10.0 * CFG["stoppt"] * nr1_free.stoppt_term(poses, rows, pack["stoppt"])
    if CFG.get("turncap", 0.0) > 0:
        import nr1_free
        out["lon_turncap"] = 10.0 * CFG["turncap"] * nr1_free.turncap_term(poses, rows, pack["turncap"])
    if CFG.get("tightin", 0.0) > 0:
        import nr1_tight
        vl, va = nr1_tight.term(poses, rows, pack["tightin"])
        out["lon_tightin"], out["lon_tightina"] = 10.0 * CFG["tightin"] * vl, 10.0 * CFG["tightin"] * va
    return out
