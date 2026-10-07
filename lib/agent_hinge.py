"""Agent hinge: the plan's footprint must stay out of the logged boxes of the objects ahead (plans/2026-10-07-agent-hinge-prereg.md, op_parity).

Labels (CPU; label build modelled on experiments/op_probe/scripts/opb_labels.py), envs/op-train on the box:

  $DATA_DIR/envs/op-train/bin/python lib/agent_hinge.py build --split navtrain [--workers 64]   # -> runs/op_parity/agent_labels/navtrain_all.npz
  $DATA_DIR/envs/op-train/bin/python lib/agent_hinge.py build --split navtest                  # -> .../navtest.npz
  $DATA_DIR/envs/op-train/bin/python lib/agent_hinge.py check [--n 300]                        # the prereg's pre-training validation

Per NAVSIM token: the scene annotations (navsim_logs/<split>/<log>.pkl, `anns`, boxes in each frame's ego = rear-axle frame) of the frames at
t0, t0 + 0.5 .. 4 s, objects of class vehicle / generic_object / pedestrian / bicycle; the K = 16 nearest to the ego at t0 are kept and followed
by track token through the 8 future frames, every box moved to the rear-axle frame at t0 (logged, non-reactive: the scorer's view).
Stored: box (N, 9, K, 5) = x, y, yaw (unwrapped over time), length, width; valid (N, 9, K); cls (N, K); ok (N).

Loss (AgentHinge): the 8 plan poses (rear axle, left +) are linearly interpolated to 0.1 s together with the origin (lib/drivable_hinge.py), the
object boxes are interpolated between their two neighbouring annotated times (a step is valid when both are). Ego box = nuPlan Pacifica with its
front edge moved forward by `margin`. Signed distance ego-box <-> object box = min(SDF of the 4 ego corners to the object box, SDF of the 4
object corners to the ego box) (exact for separated rectangles, < 0 when a corner is inside). Only objects ahead count: object centre in the front
half plane of the ego's interpolated pose (relative to the ego box centre) and |lateral offset| < 3 m or overlapping. Per (row, 0.1 s step):
relu(margin - min over counted objects of the distance) (0 when none); the hinge is the mean over rows and the 41 steps.
"""
import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "lib")]
D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
OUT = D / "runs" / "op_parity" / "agent_labels"
LOGDIR = {"navtest": "test", "navtrain": "trainval"}
KEEP = ("vehicle", "generic_object", "pedestrian", "bicycle")
K, NT = 16, 9                                      # objects per token; annotated times t0 + 0, 0.5 .. 4 s
FRONT, REAR, HALF_W = 4.049, -1.127, 1.1485        # nuPlan Pacifica about the rear axle
LAT_MAX = 3.0


def _quat_yaw(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _pose(f):
    return np.array([f["ego2global_translation"][0], f["ego2global_translation"][1], _quat_yaw(f["ego2global_rotation"])])


def _to_t0(b, pose_j, pose_0):
    """boxes (n, 7) in the ego frame of frame j -> (n, 3) x, y, yaw in the rear-axle frame at t0."""
    c, s = np.cos(pose_j[2]), np.sin(pose_j[2])
    gx, gy = pose_j[0] + c * b[:, 0] - s * b[:, 1], pose_j[1] + s * b[:, 0] + c * b[:, 1]
    c0, s0 = np.cos(pose_0[2]), np.sin(pose_0[2])
    dx, dy = gx - pose_0[0], gy - pose_0[1]
    return np.stack([c0 * dx + s0 * dy, -s0 * dx + c0 * dy, b[:, 6] + pose_j[2] - pose_0[2]], -1)


def work(job):
    split, log, toks = job
    fr = pickle.load(open(D / "datasets" / "navsim" / "navsim_logs" / LOGDIR[split] / f"{log}.pkl", "rb"))
    fr = sorted(fr, key=lambda f: f["timestamp"])
    at = {f["token"]: i for i, f in enumerate(fr)}
    out = []
    for t in toks:
        box, val, cls = np.zeros((NT, K, 5), np.float32), np.zeros((NT, K), bool), np.full(K, -1, np.int8)
        i = at.get(t)
        if i is None:
            out.append((t, box, val, cls, False))
            continue
        f0 = fr[i]
        p0 = _pose(f0)
        a = f0["anns"]
        names = np.asarray(a["gt_names"])
        keep = np.flatnonzero(np.isin(names, KEEP))
        b0 = np.asarray(a["gt_boxes"], np.float64)[keep]
        if len(keep):
            sel = keep[np.argsort(np.hypot(b0[:, 0], b0[:, 1]), kind="stable")[:K]]
            trk = np.asarray(a["track_tokens"])[sel]
            cls[:len(sel)] = [KEEP.index(n) for n in names[sel]]
            for j in range(NT):
                if i + j >= len(fr) or abs((fr[i + j]["timestamp"] - f0["timestamp"]) / 1e6 - 0.5 * j) > 0.1:
                    break
                fj = fr[i + j]
                aj = fj["anns"]
                pos = {tk: r for r, tk in enumerate(np.asarray(aj["track_tokens"]).tolist())}
                rr = np.array([pos.get(tk, -1) for tk in trk.tolist()])
                m = rr >= 0
                if m.any():
                    bj = np.asarray(aj["gt_boxes"], np.float64)[rr[m]]
                    box[j, :len(sel)][m, :3] = _to_t0(bj, _pose(fj), p0)
                    box[j, :len(sel)][m, 3:5] = bj[:, 3:5]
                    val[j, :len(sel)][m] = True
            yaw = box[:, :, 2].astype(np.float64)                         # unwrap over time (invalid steps carry the last valid yaw)
            for k in range(len(sel)):
                v = np.flatnonzero(val[:, k])
                if len(v):
                    yaw[v, k] = np.unwrap(yaw[v, k])
            box[:, :, 2] = yaw
        out.append((t, box, val, cls, True))
    return out


def tokens_of(split):
    """(tokens, logs) in the order of the op_parity caches (navtest = lb_navtest, navtrain = navtrain_full.s0..11of12)."""
    cr = D / "runs" / "op_parity" / "cache"
    dirs = ["lb_navtest"] if split == "navtest" else [f"navtrain_full.s{k}of12" for k in range(12)]
    tabs = [np.load(cr / d / "tab.npz") for d in dirs]
    return np.concatenate([t["names"] for t in tabs]), np.concatenate([t["log"] for t in tabs])


def cmd_build(a):
    from jevdrive import par
    from jevdrive.data import splits
    from jevdrive.run import Run
    tag = "navtest" if a.split == "navtest" else "navtrain_all"
    with Run("op_parity", f"agent-labels-{tag}", config=vars(a)) as run:
        run.use_split(splits.load(f"navsim/{a.split}"))
        toks, logs = tokens_of(a.split)
        if a.limit:
            toks, logs = toks[:a.limit], logs[:a.limit]
        jobs = {}
        for t, lg in zip(toks, logs):
            jobs.setdefault(lg, []).append(t)
        run.info(f"{tag}: {len(toks)} tokens in {len(jobs)} logs, {a.workers} workers")
        t0 = time.time()
        res = par.pmap(work, [(a.split, lg, ts) for lg, ts in jobs.items()], run=run, workers=a.workers)
        res.raise_if_failed()
        R = {r[0]: r for part in res.values for r in part}
        box = np.stack([R[t][1] for t in toks])
        val = np.stack([R[t][2] for t in toks])
        cls = np.stack([R[t][3] for t in toks])
        ok = np.array([R[t][4] for t in toks])
        OUT.mkdir(parents=True, exist_ok=True)
        f = OUT / f"{tag}{'-lim' if a.limit else ''}.npz"
        tmp = f.with_name(f".{f.stem}.{os.getpid()}.npz")
        np.savez(tmp, tokens=toks, log=logs, box=box, valid=val, cls=cls, ok=ok, classes=np.array(KEEP))
        os.replace(tmp, f)
        n_obj = (cls >= 0).sum(1)
        run.summary.update(n=len(toks), ok=int(ok.sum()), compute_s=time.time() - t0, out=str(f), objs_mean=float(n_obj.mean()),
                           objs_full=float((n_obj == K).mean()), valid_t4=float(val[:, -1].sum(1).mean()))
        run.info(f"ok {ok.sum()}/{len(toks)} in {time.time() - t0:.0f} s -> {f}; {n_obj.mean():.1f} objects / token, K full on {(n_obj == K).mean():.3f}")


# ---------------------------------------------------------------- loss
def _box_sdf(px, py, cx, cy, yaw, hl, hw):
    """Signed distance of points (px, py) to oriented boxes (centre cx, cy, heading yaw, half extents hl, hw); broadcasting torch tensors."""
    import torch
    c, s = torch.cos(yaw), torch.sin(yaw)
    dx, dy = px - cx, py - cy
    qx = (c * dx + s * dy).abs() - hl
    qy = (-s * dx + c * dy).abs() - hw
    return torch.sqrt(qx.clamp_min(0) ** 2 + qy.clamp_min(0) ** 2 + 1e-12) + torch.maximum(qx, qy).clamp_max(0)


def _corners(cx, cy, yaw, hl, hw):
    """(..., ) box params -> (..., 4) corner x, y."""
    import torch
    c, s = torch.cos(yaw)[..., None], torch.sin(yaw)[..., None]
    u = torch.tensor([1.0, 1.0, -1.0, -1.0], device=cx.device) * hl[..., None]
    v = torch.tensor([1.0, -1.0, 1.0, -1.0], device=cx.device) * hw[..., None]
    return cx[..., None] + c * u - s * v, cy[..., None] + s * u + c * v


class AgentHinge:
    """Labels aligned to a row order (tokens), on `dev`; __call__(x, y, psi, rows) -> the mean agent hinge over the rows with a label."""

    def __init__(self, label_file, tokens, dev, margin: float = 0.5):
        import torch
        from drivable_hinge import interp_matrix
        z = np.load(label_file)
        pos = {t: i for i, t in enumerate(z["tokens"].tolist())}
        idx = np.array([pos.get(t, -1) for t in np.asarray(tokens).tolist()])
        ok = (idx >= 0) & z["ok"][np.maximum(idx, 0)]
        n = len(idx)
        self.box = torch.zeros((n, NT, K, 5), dtype=torch.float32, device=dev)
        self.val = torch.zeros((n, NT, K), dtype=torch.bool, device=dev)
        rows = np.flatnonzero(ok)
        box, val = z["box"], z["valid"]
        for i in range(0, len(rows), 8192):
            r = rows[i:i + 8192]
            rt = torch.as_tensor(r, device=dev)
            self.box[rt] = torch.from_numpy(box[idx[r]]).to(dev)
            self.val[rt] = torch.from_numpy(val[idx[r]]).to(dev)
        self.ok = torch.as_tensor(ok, device=dev)
        M = interp_matrix()
        self.M = torch.as_tensor(M, dtype=torch.float32, device=dev)
        self.Mv = torch.as_tensor(M > 0, dtype=torch.float32, device=dev)
        self.margin = margin
        self.coverage = float(ok.mean())
        f = FRONT + margin
        self.e_off, self.e_hl, self.e_hw = (f + REAR) / 2, (f - REAR) / 2, HALF_W
        self.c_off = (FRONT + REAR) / 2                       # centre of the unextended box: the front half plane is taken about it

    def distances(self, x, y, psi, rows):
        """x, y, psi (B, 8) plan poses -> signed distance (B, 41, K) of the extended ego box to each object box, counted mask (B, 41, K)."""
        import torch
        P = torch.stack([x, y, psi], -1)
        d = torch.einsum("kj,bjc->bkc", self.M, torch.cat([torch.zeros_like(P[:, :1]), P], 1))      # (B, 41, 3)
        bx = torch.einsum("kj,bjoc->bkoc", self.M, self.box[rows])                                   # (B, 41, K, 5)
        bad = torch.einsum("kj,bjo->bko", self.Mv, (~self.val[rows]).float())
        valid = bad < 0.5
        ex, ey, eps = d[..., 0:1], d[..., 1:2], d[..., 2:3]                                           # (B, 41, 1)
        c, s = torch.cos(eps), torch.sin(eps)
        ecx, ecy = ex + c * self.e_off, ey + s * self.e_off
        hl_e, hw_e = torch.full_like(ecx, self.e_hl), torch.full_like(ecx, self.e_hw)
        ox, oy, oyaw, ohl, ohw = bx[..., 0], bx[..., 1], bx[..., 2], bx[..., 3] / 2, bx[..., 4] / 2
        qx, qy = _corners(ecx, ecy, eps, hl_e, hw_e)                                                # ego corners (B, 41, 1, 4)
        d1 = _box_sdf(qx, qy, ox[..., None], oy[..., None], oyaw[..., None], ohl[..., None], ohw[..., None]).amin(-1)
        px, py = _corners(ox, oy, oyaw, ohl, ohw)                                                    # object corners (B, 41, K, 4)
        d2 = _box_sdf(px, py, ecx[..., None], ecy[..., None], eps[..., None], hl_e[..., None], hw_e[..., None]).amin(-1)
        dist = torch.minimum(d1, d2)
        rx, ry = ox - (ex + c * self.c_off), oy - (ey + s * self.c_off)
        lon, lat = c * rx + s * ry, -s * rx + c * ry
        counted = valid & (lon > 0) & ((lat.abs() < LAT_MAX) | (dist < 0))
        return dist, counted

    def per_step(self, x, y, psi, rows):
        """(B, 41) relu(margin - nearest counted distance)."""
        dist, counted = self.distances(x, y, psi, rows)
        return (self.margin - dist).clamp_min(0).masked_fill(~counted, 0.0).amax(-1)

    def __call__(self, x, y, psi, rows):
        ok = self.ok[rows]
        if not ok.any():
            return x.sum() * 0.0
        return self.per_step(x[ok], y[ok], psi[ok], rows[ok]).mean()


# ---------------------------------------------------------------- validation (before training)
def token_hinge(H, poses, rows, bs=2048):
    """poses (n, 8, 3) rear-axle plans of rows -> per-row max step hinge, per-row max excluding t = 0, min distance (n,)."""
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            r = torch.as_tensor(rows[i:i + bs], device=H.box.device)
            P = torch.as_tensor(poses[i:i + bs], dtype=torch.float32, device=H.box.device)
            v = H.per_step(P[..., 0], P[..., 1], P[..., 2], r)
            dist, cnt = H.distances(P[..., 0], P[..., 1], P[..., 2], r)
            dmin = dist.masked_fill(~cnt, 1e3).amin((1, 2))
            out.append(torch.stack([v.amax(1), v[:, 1:].amax(1), dmin], 1).cpu().numpy())
    return np.concatenate(out)


def cmd_check(a):
    import json
    import torch
    from jevdrive.bench import tables as T
    from jevdrive.run import Run
    from jevdrive.bench.compat import pred_file
    from jevdrive.bench.models import resolve
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tab = np.load(D / "runs/op_parity/cache/lb_navtest/tab.npz")
    toks = tab["names"]
    H = AgentHinge(OUT / "navtest.npz", toks, dev, a.margin)
    rng = np.random.default_rng(0)
    has = ~np.isnan(tab["fut"][:, 0, 0]) & H.ok.cpu().numpy()
    with Run("op_parity", "agent-labels-check", config=vars(a)) as run:
        res = {"coverage": H.coverage, "margin": a.margin}
        pick = rng.choice(np.flatnonzero(has), a.n, replace=False)
        hv = token_hinge(H, tab["fut"][pick].astype(np.float32), pick)
        allr = np.flatnonzero(has)
        ha = token_hinge(H, tab["fut"][allr].astype(np.float32), allr)
        res["human"] = {"n": int(a.n), "viol": float((hv[:, 0] > 0).mean()), "viol_after_t0": float((hv[:, 1] > 0).mean()),
                        "overlap": float((hv[:, 2] < 0).mean()),
                        "all_n": int(len(allr)), "all_viol": float((ha[:, 0] > 0).mean()), "all_viol_after_t0": float((ha[:, 1] > 0).mean()),
                        "all_overlap": float((ha[:, 2] < 0).mean())}
        pos = {t: i for i, t in enumerate(toks.tolist())}
        for m in a.models:
            u, src = T.load("navtest", m)
            pf = pred_file(m, "navtest")
            if not pf.exists():                                                   # pre-bench parity runs: the lane's op_lb export
                mm = resolve(m)
                pf = D / "runs/op_lb/lb_navtest/preds" / f"{mm.frames}-cinque_PP{mm.name}__base.npz"
            z = np.load(pf)
            pr = {t: p for t, p in zip(z["tokens"].tolist(), z["poses"])}
            nc = u.index[u["NC"].to_numpy(float) < 1].tolist()
            nt = u.index[(u["NC"].to_numpy(float) < 1) | (u["TTC"].to_numpy(float) < 1)].tolist()
            ok_all = [t for t in u.index if t in pr and t in pos]
            rr = np.array([pos[t] for t in ok_all])
            hh = token_hinge(H, np.stack([pr[t] for t in ok_all]).astype(np.float32), rr)
            hm = dict(zip(ok_all, hh))
            f = lambda ts, j=0: float(np.mean([hm[t][j] > 0 for t in ts if t in hm]))  # noqa: E731
            res[m] = {"src": src, "n_nc_fail": len(nc), "nc_fail_hinge_pos": f(nc), "nc_fail_overlap": float(np.mean([hm[t][2] < 0 for t in nc if t in hm])),
                      "n_ncttc_fail": len(nt), "ncttc_fail_hinge_pos": f(nt),
                      "pass_hinge_pos": f([t for t in ok_all if t not in set(nt)]), "all_hinge_pos": f(ok_all)}
        res["pass"] = bool(res["human"]["viol"] < 0.01 and all(res[m]["nc_fail_hinge_pos"] >= 0.70 for m in a.models))
        run.summary.update(res)
        (OUT / f"check-m{a.margin}.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("build")
    p.add_argument("--split", choices=["navtest", "navtrain"], required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=64)
    p = sp.add_parser("check")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--margin", type=float, default=0.5)
    p.add_argument("--models", nargs="+", default=["P2H10-F-s0", "HP-F-s0"])
    a = ap.parse_args()
    {"build": cmd_build, "check": cmd_check}[a.cmd](a)
