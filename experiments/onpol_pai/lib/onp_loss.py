"""ONPOL-PAI: the on-policy PAI rows of a training batch and the terms on them (plans/2026-10-11-onpol-pai-prereg.md sections 2, 4, 5).
Nothing of the S recipe is edited: the rows of pp_train.Store keep loss43.Losses43; this module only adds terms on extra batch rows that
come from the PAI row files (results/rows_format.md) through scripts/onp_cache.py.

Rows       OnRows: the cached rows of the passes (`pol` = states the rollouts of P2H10S-F-s0 visited, `log` = states along the logged
           path) with their labels on the card and their frozen-encoder tokens on the host (one token per distinct image pair).
Plan       the 8 poses of a row are exported as the PAI driver exports them (rear_xy = op_interp.to_rear `lever` with the camera's x AND y
           in the rig frame, then the 0.5 s grid); pp_train.rear drops the camera's y.
Terms      per row, on the model's own plan, all in the rig frame at t0 (OnLoss.terms):
  agent    S's agent hinge (lib/agent_hinge.AgentHinge.per_step, the same operations) with the row's own ego box against the simulator's
           actor boxes at t0 + 0 .. 4 s: margin 0.3 m inside the ego corridor, 0 outside                                    shape-only
  edge     the same hinge, the same margins (0.3 m inside the ego corridor, 0 outside), against the 96 nearest road-edge segments as
           static thin boxes. With 0.3 m outside the corridor too the log's own future violates it on 17 % of the rows where the ego
           is on the log (three of 16 scenes drive within 0.3 m of an edge); with S's margins on 0.2 %                      shape-only
  corr     route hinge: mean over the 8 poses of relu(distance to the finite logged path - 3.0 m); a pose past the path's end counts by
           its distance to the end point                                                                                    shape-only
  stop     design 3.2 item B, stop-sign rule: front bumper arc s_f(t) <= theta_l = max(d_l - 1.0 m, the log's own furthest front-bumper
           arc while it has not crossed) at every dense step t with t0 + t < the log's crossing time of line l, gated by
           |lateral| <= 3.5 m and s_f not more than 30 m past the path's last point (its straight continuation, where the row files
           still accept a line); lines the ego's front bumper has already passed at t0 are dropped
                                                                                                                            timing-only
  turn     design 3.2 item C: v(t) <= max(v_ref + 1.0, 1.15 v_ref) in the window s_pk - 10 m .. s_pk + 3 m of the first turn peak ahead,
           |lateral| <= 3.5 m, |heading - path heading at the peak| <= 30 deg, log speed at the peak >= 1 m/s; v_ref(s) = the largest
           dense speed of the log's own future within 1 m of arc s, or (outside the log's 4 s reach) the log's speed on the path grid
           within 1 m of s                                                                                                  timing-only
  prog     design 3.2 guard P: mean over the 8 poses of relu(0.95 x the log's same-time progress - 0.5 m - the plan's arc length), only
           on `open` rows (no stop line and no turn peak within 60 m ahead, no actor box within 2 m of the path in the next 60 m)
                                                                                                                            timing-only
  imit     the recipe's imitation distance (pp_train.Losses.dist's Huber terms) to `imit` (pass pol) / `log_fut` (pass log)
Weights    as S weighs its hinge-only rows (loss43): a term is summed over the on-policy rows and divided by the number of imitation rows
           of the batch's normal rows, x lam (10) x ho_w (3); the imitation term the same sum x lam_i (1): an on-policy row counts like
           one on-log imitation row. Non-plan heads of the on-policy rows are distilled to shipped Cinque on the same tokens with the
           per-row weight of a normal row (as loss43's distill_ho), in every arm.
Logged     on_<term> (the weighted sum above, before lam), on/<pass>/<term>_mean | _pos | _posmean over the pass's rows of the batch
           (loss43.row_stats), on/<pass>/<term>_elig (share of rows the term is defined on). Every term is logged in every arm; `arm`
           only decides which of them reach the total.
"""
import math

import numpy as np
import torch

from agent_hinge import LAT_MAX, _box_sdf, _corners
from drivable_hinge import interp_matrix
from loss43 import row_stats, shape_only

AG_M, AG_SIDE, EDGE_M, EDGE_SIDE = 0.3, 0.0, 0.3, 0.0
BAND = 3.0                                     # corridor dead band (the scorer's band is 4 m)
M_B, LAT_IN = 1.0, 3.5                         # stop-line margin; lateral gate of the longitudinal terms
EXT = 30.0                                     # a stop line may lie on the straight continuation of the path past its last sample (rows_format.md)
TURN_WIN, TURN_HEAD, V_PK_MIN = (-10.0, 3.0), math.radians(30.0), 1.0
P_K, P_C = 0.95, 0.5
OPEN_S, OPEN_BOX = 60.0, 2.0
S0 = 20                                        # grid index of the ego's own arc position on `path` (s_ego - 20 m .. s_ego + 100 m at 1 m)
SHAPE_T, TIME_T = ("agent", "edge", "corr"), ("stop", "turn", "prog")
TERMS = SHAPE_T + TIME_T
ARMS = {"imit": ("imit",), "self": TERMS, "both": ("imit",) + TERMS, "none": ()}
TAU = np.arange(41) * 0.1


def timing_only(x, y, psi):
    """Poses (n, K) -> (x~, y~) with the same values and d(x~, y~) / d(x, y) = u u^T, u = the detached heading: a pose can only move along
    its own heading (design.md 3.1; the mirror of loss43.shape_only). The caller passes psi on detached."""
    xd, yd, pd = x.detach(), y.detach(), psi.detach()
    ux, uy = torch.cos(pd), torch.sin(pd)
    a = ux * (x - xd) + uy * (y - yd)
    return xd + ux * a, yd + uy * a


def rear_xy(plan, cam_xy, W):
    """plan (B, 33, 15) camera frame (x fwd, y right, ch 11 yaw), cam_xy (B, 2) camera position in the rig frame, W (K, 33) -> rear-axle
    x, y (left +), yaw (left +) at W's times: op_interp.to_rear(mode "lever", linear), the export of pai_core.plan."""
    px, py, psi = plan[..., 0], -plan[..., 1], -plan[..., 11]
    dx, dy = cam_xy[:, :1], cam_xy[:, 1:2]
    c, s = torch.cos(psi), torch.sin(psi)
    return (dx + px - (c * dx - s * dy)) @ W.T, (dy + py - (s * dx + c * dy)) @ W.T, psi @ W.T


def dense(M, x, y, psi, p0=None):
    """8 poses (B, 8) -> (B, 41, 3) at 0.1 s from the origin (or p0 (B, 3)), linear (lib/drivable_hinge.interp_matrix)."""
    P = torch.stack([x, y, psi], -1)
    return torch.einsum("kj,bjc->bkc", M, torch.cat([torch.zeros_like(P[:, :1]) if p0 is None else p0[:, None], P], 1))


def seg_len(dx, dy):
    return (dx * dx + dy * dy + 1e-12).sqrt()


def speed(d):
    """Dense poses (B, 41, 3) -> speed (B, 41): mean of the adjacent 0.1 s segments, one-sided at the ends (long_contact lc.path_speed)."""
    v = seg_len(d[:, 1:, 0] - d[:, :-1, 0], d[:, 1:, 1] - d[:, :-1, 1]) / 0.1
    return torch.cat([v[:, :1], 0.5 * (v[:, 1:] + v[:, :-1]), v[:, -1:]], 1)


def project(path, ok, p):
    """path (B, NP, 2) at 1 m with its non-finite points zeroed, ok (B, NP), points p (B, n, 2) -> on the nearest finite segment:
    arc s (B, n) in grid units from path[:, 0] (differentiable along that segment, continued linearly past its ends), signed lateral
    offset l (left +), distance d to the finite path (clamped to its end points; gradient = the unit vector away from the nearest point).
    A row without a finite segment returns numbers of the zeroed path: the caller masks it."""
    A, u = path[:, None, :-1], path[:, None, 1:] - path[:, None, :-1]                 # (B, 1, NP - 1, 2)
    seg = ok[:, :-1] & ok[:, 1:]
    seg = (seg | ~seg.any(1, keepdim=True))[:, None]
    L2 = (u ** 2).sum(-1).clamp_min(1e-9)
    w = p[:, :, None] - A                                                              # (B, n, NP - 1, 2)
    tr = (w * u).sum(-1) / L2
    t = tr.clamp(0.0, 1.0)
    j = ((w - t[..., None] * u) ** 2).sum(-1).masked_fill(~seg, float("inf")).argmin(-1, keepdim=True)
    j2 = j[..., None].expand(-1, -1, -1, 2)
    uj, wj = u.expand(-1, p.shape[1], -1, -1).gather(2, j2)[:, :, 0], w.gather(2, j2)[:, :, 0]
    tj, trj, Lj = t.gather(2, j)[..., 0], tr.gather(2, j)[..., 0], L2.expand(-1, p.shape[1], -1).gather(2, j)[..., 0].sqrt()
    r = wj - (tj[..., None] * uj).detach()
    return j[..., 0] + trj, (uj[..., 0] * wj[..., 1] - uj[..., 1] * wj[..., 0]) / Lj, (r ** 2).sum(-1).add(1e-12).sqrt()


def box_steps(d, bx, val, front, rear, hw, margin, side):
    """lib/agent_hinge.AgentHinge.distances + per_step with a per-row ego box. d (B, 41, 3) dense rear-axle poses; bx (B, T, K, 5) object
    boxes and val (B, T, K) at the 41 steps (T = 41) or static (T = 1); front / rear / hw (B,) the ego box about the rear axle
    -> (B, 41) max over counted objects of relu(m - distance), m = margin inside the ego's lateral corridor, side outside."""
    v3 = lambda z: z[:, None, None]  # noqa: E731
    e_off, e_hl, c_off, hw = v3((front + margin + rear) / 2), v3((front + margin - rear) / 2), v3((front + rear) / 2), v3(hw)
    ex, ey, eps = d[..., 0:1], d[..., 1:2], d[..., 2:3]                                                    # (B, 41, 1)
    c, s = torch.cos(eps), torch.sin(eps)
    ecx, ecy = ex + c * e_off, ey + s * e_off
    hl_e, hw_e = e_hl.expand_as(ecx), hw.expand_as(ecx)
    ox, oy, oyaw, ohl, ohw = bx[..., 0], bx[..., 1], bx[..., 2], bx[..., 3] / 2, bx[..., 4] / 2
    qx, qy = _corners(ecx, ecy, eps, hl_e, hw_e)                                                          # ego corners (B, 41, 1, 4)
    d1 = _box_sdf(qx, qy, ox[..., None], oy[..., None], oyaw[..., None], ohl[..., None], ohw[..., None]).amin(-1)
    px, py = _corners(ox, oy, oyaw, ohl, ohw)                                                             # object corners (B, T, K, 4)
    d2 = _box_sdf(px, py, ecx[..., None], ecy[..., None], eps[..., None], hl_e[..., None], hw_e[..., None]).amin(-1)
    dist = torch.minimum(d1, d2)
    rx, ry = ox - (ex + c * c_off), oy - (ey + s * c_off)
    lon, lat = c * rx + s * ry, -s * rx + c * ry
    counted = val & (lon > 0) & ((lat.abs() < LAT_MAX) | (dist < 0))
    lc = -s[..., None] * (px - ex[..., None]) + c[..., None] * (py - ey[..., None])
    m = torch.where((lc.amax(-1) > -hw) & (lc.amin(-1) < hw), margin, side) if side != margin else margin
    return (m - dist).clamp_min(0).masked_fill(~counted, 0.0).amax(-1)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------- rows
class OnRows:
    """The cached rows of `passes` (scripts/onp_cache.py: <root>/<pass>/{tab.npz, tok.npy}) concatenated in that order. Labels and model
    inputs live on `dev` as attributes (row-indexed tensors); tokens stay memory-mapped on the host (front()). ego_box: the label key
    of the ego box (`veh_sim` = the simulator's and the scorer's box, `veh` = the rollout spec's)."""

    def __init__(self, root, passes, dev, ego_box="veh_sim"):
        tabs = [dict(np.load(root / p / "tab.npz")) for p in passes]
        self.root, self.passes, self.dev = root, list(passes), dev
        assert not any("labels_only" in t for t in tabs) or dev.type == "cpu", "a labels-only cache has no tokens: checks on the CPU only"
        self.tok = [np.load(root / p / "tok.npy", mmap_mode="r") for p in passes]
        cat = lambda k: np.concatenate([t[k] for t in tabs])  # noqa: E731
        self.n = sum(len(t["sid"]) for t in tabs)
        self.pas = np.concatenate([np.full(len(t["sid"]), i) for i, t in enumerate(tabs)])
        self.scene = np.concatenate([t["scenes"][t["sid"]] for t in tabs])
        self.tix, self.after, self.usable, self.cold, self.k = cat("tix"), cat("after_event") | cat("after_end"), cat("usable"), cat("cold"), cat("k")
        t = lambda x, dt=torch.float32: torch.from_numpy(np.ascontiguousarray(x)).to(dev, dt)  # noqa: E731
        z = lambda k: t(np.nan_to_num(cat(k).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0))  # noqa: E731
        self.pas_t = t(self.pas, torch.long)
        self.ego, self.cam, self.t_out = t(cat("ego_in")), t(cat("cam_t")), t(cat("t_out"))
        self.tc = t(np.where(cat("lht")[:, None], [[0.0, 1.0]], [[1.0, 0.0]]))
        self.poses, self.poses_model, self.v0, self.e0, self.ds0 = z("poses"), z("poses_model"), z("v0"), z("e0"), z("ds0")
        vb = cat(ego_box).astype(np.float32)                                              # length, width, rear axle -> box centre
        self.front, self.rear, self.hw = t(vb[:, 2] + vb[:, 0] / 2), t(vb[:, 2] - vb[:, 0] / 2), t(vb[:, 1] / 2)
        self.box, self.box_ok = z("box"), t(cat("box_ok"), torch.bool)
        self.edge, self.edge_ok = z("edge_box"), t(cat("edge_ok"), torch.bool)
        self.path, self.path_ok, self.path_h = z("path"), t(cat("path_ok"), torch.bool), z("path_h")
        self.path_v = t(np.nan_to_num(cat("path_v").astype(np.float32), nan=1e3))          # no logged speed there: no cap
        self.has_path = self.path_ok[:, S0 - 1:S0 + 1].all(1)                              # the ego's own arc is on the finite path
        self.rlen = (self.path_ok * torch.arange(self.path_ok.shape[1], device=dev)).amax(1).float()   # grid arc of the last finite point
        self.log0, self.log_fut, lok = z("log0"), z("log_fut"), cat("log_fut_ok")
        l0 = np.isfinite(cat("log0")).all(1)
        self.log_ok = t(lok.reshape(len(lok), -1).all(1) & l0, torch.bool)
        self.log_n = t(np.cumprod(lok.reshape(len(lok), 8, -1).all(-1), 1).sum(1) * l0 - (~l0), torch.long)   # leading logged poses that exist; -1: none
        self.imit, io = z("imit"), cat("imit_ok")
        self.imit_ok = t(io.reshape(len(io), 8, -1).all(-1), torch.bool)
        pas_log = t(np.array([p == "log" for p in passes])[self.pas], torch.bool)
        self.tgt = torch.where(pas_log[:, None, None], self.log_fut, self.imit)            # imitation target of the row's pass
        self.tgt_ok = torch.where(pas_log[:, None], t(lok.reshape(len(lok), 8, -1).all(-1), torch.bool), self.imit_ok)
        ss, tcr = cat("stop_s").astype(np.float32), cat("stop_tcross_us").astype(np.float64) - cat("t0_us").astype(np.float64)[:, None]
        self.stop_ok = t(np.isfinite(ss) & cat("has_stop")[:, None], torch.bool)
        self.stop_g = t(np.nan_to_num(ss) + S0)                                            # grid arc of the line
        self.stop_t = t(np.minimum(tcr / 1e6, 1e6))                                        # s after t0 at which the log crosses it
        tk = cat("turn_s").astype(np.float32)
        self.turn_ok = t(np.isfinite(tk) & (tk > 0) & (np.nan_to_num(cat("turn_vpk")) >= V_PK_MIN), torch.bool)
        self.turn_g = t(np.nan_to_num(tk) + S0)
        self.open = self._open()

    @torch.no_grad()
    def _open(self):
        """Rows without a longitudinal context: no stop line and no turn peak within OPEN_S m ahead, no actor box (centre or corner, any of
        its 9 times) within OPEN_BOX m of the path between the ego's arc and OPEN_S m ahead."""
        far = S0 + OPEN_S
        o = ~(self.stop_ok & (self.stop_g <= far)).any(1) & ~(self.turn_ok & (self.turn_g <= far)) & self.has_path
        near = torch.zeros(self.n, dtype=torch.bool, device=self.dev)
        sl = slice(S0, S0 + int(OPEN_S) + 1)
        for i in range(0, self.n, 512):
            b, v = self.box[i:i + 512], self.box_ok[i:i + 512]
            px, py = _corners(b[..., 0], b[..., 1], b[..., 2], b[..., 3] / 2, b[..., 4] / 2)
            p = torch.stack([torch.cat([px, b[..., :1]], -1), torch.cat([py, b[..., 1:2]], -1)], -1)        # (n, 9, K, 5, 2)
            d = project(self.path[i:i + 512, sl], self.path_ok[i:i + 512, sl], p.flatten(1, 3))[2].view(p.shape[:-1])
            near[i:i + 512] = ((d < OPEN_BOX).any(-1) & v).flatten(1).any(1)
        return o & ~near

    def pool(self, scenes, pas=None, after=False):
        """Row numbers of the scenes (and of one pass), usable, by default without the rows at or after the rollout's first scorer event
        (`after_event`) and without the rows the scorer never read (`after_end`)."""
        m = np.isin(self.scene, np.asarray(list(scenes))) & self.usable & (after | ~self.after)
        return np.flatnonzero(m if pas is None else m & (self.pas == self.passes.index(pas)))

    def tokens(self, r) -> torch.Tensor:
        """Rows (numpy) -> (n, 8, 32, 512) fp16 host tensor, each pass read once in file order."""
        r = np.asarray(r)
        out = np.empty((len(r), 8) + self.tok[0].shape[1:], np.float16)
        for i, tk in enumerate(self.tok):
            m = self.pas[r] == i
            if m.any():
                u, inv = np.unique(self.tix[r[m]], return_inverse=True)
                out[m] = tk[u][inv.reshape(-1, 8)]
        return torch.from_numpy(out)


# ---------------------------------------------------------------- terms
class OnLoss:
    """Terms on the on-policy rows of a batch. R: OnRows; W (8, 33), sig (3 x (8,)): pp_train.Losses.W / .s; di, plan_cols, tstd, dmask:
    its distillation columns; arm: imit | self | both | none (module docstring)."""

    def __init__(self, R: OnRows, LS, arm: str, lam=10.0, ho_w=3.0, shape=True):
        self.R, self.W, self.sig, self.pi, self.di, self.plan_cols, self.tstd, self.dmask = R, LS.W, LS.s, LS.pi, LS.di, LS.plan_cols, LS.tstd, LS.dmask
        self.lam_i, self.lam_d, self.lam, self.ho_w, self.shape, self.on = LS.cfg.lam_i, LS.cfg.lam_d, lam, ho_w, shape, ARMS[arm]
        M = interp_matrix()
        self.M, self.Mv = (torch.as_tensor(x, dtype=torch.float32, device=R.dev) for x in (M, M > 0))
        self.tau = torch.as_tensor(TAU, dtype=torch.float32, device=R.dev)
        self.need = torch.as_tensor(np.ceil(np.arange(41) / 5), dtype=torch.long, device=R.dev)      # logged poses a dense log step interpolates

    def poses(self, out, r):
        return rear_xy(out.float()[:, self.pi].view(-1, 33, 15), self.R.cam[r, :2], self.W)

    def imit(self, x, y, psi, r):
        """-> the recipe's imitation distance per row (mean over the target's valid poses), rows with a target."""
        R = self.R
        t, ok = R.tgt[r], R.tgt_ok[r]
        h = lambda z: torch.nn.functional.huber_loss(z, torch.zeros_like(z), reduction="none", delta=1.0)  # noqa: E731
        d = h((x - t[..., 0]) / self.sig[0]) + h((y - t[..., 1]) / self.sig[1]) + h(wrap(psi - t[..., 2]) / self.sig[2])
        return (d * ok).sum(1) / ok.sum(1).clamp_min(1), ok.any(1)

    def terms(self, x, y, psi, r, shape=None):
        """Plan poses (n, 8) of rows r -> {term: (per-row value (n,), rows the term is defined on (n,) bool)}; 0 where it is not."""
        R = self.R
        shape = self.shape if shape is None else shape
        xs, ys = shape_only(x, y, psi) if shape else (x, y)
        xt, yt = timing_only(x, y, psi)
        pd = psi.detach()
        fr, re, hw = R.front[r], R.rear[r], R.hw[r]
        ds = dense(self.M, xs, ys, psi)
        out = {}
        bx = torch.einsum("kj,bjoc->bkoc", self.M, R.box[r])
        bv = torch.einsum("kj,bjo->bko", self.Mv, (~R.box_ok[r]).float()) < 0.5
        out["agent"] = (box_steps(ds, bx, bv, fr, re, hw, AG_M, AG_SIDE).mean(1), R.box_ok[r].flatten(1).any(1))
        out["edge"] = (box_steps(ds, R.edge[r][:, None], R.edge_ok[r][:, None], fr, re, hw, EDGE_M, EDGE_SIDE).mean(1), R.edge_ok[r].any(1))
        path, pok, hp = R.path[r], R.path_ok[r], R.has_path[r]
        out["corr"] = ((project(path, pok, torch.stack([xs, ys], -1))[2] - BAND).clamp_min(0).mean(1) * hp, hp)
        # longitudinal terms: timing-only poses, the yaw detached
        dt = dense(self.M, xt, yt, pd)
        c, s = torch.cos(dt[..., 2]), torch.sin(dt[..., 2])
        lok = R.log_ok[r]
        lst = self.need[None] <= R.log_n[r][:, None]                                                       # (n, 41) dense log steps that exist
        with torch.no_grad():
            dl = dense(self.M, R.log_fut[r][..., 0], R.log_fut[r][..., 1], R.log_fut[r][..., 2], R.log0[r])
            slf = project(path, pok, torch.stack([dl[..., 0] + fr[:, None] * torch.cos(dl[..., 2]), dl[..., 1] + fr[:, None] * torch.sin(dl[..., 2])], -1))[0]
            slr, vl = project(path, pok, dl[..., :2])[0], speed(dl)
        sf, lf, _ = project(path, pok, torch.stack([dt[..., 0] + fr[:, None] * c, dt[..., 1] + fr[:, None] * s], -1))
        line = R.stop_ok[r] & (R.stop_g[r] > sf[:, :1].detach())                                           # (n, 4): not passed at t0
        mu = line[:, None] & (self.tau[None, :, None] < R.stop_t[r][:, None])                             # (n, 41, 4): the log has not crossed
        far = slf[:, :, None].masked_fill(~(mu & lst[:, :, None]), -1e6).amax(1)                          # the log's furthest point while mu
        theta = torch.maximum(R.stop_g[r] - M_B, far)
        gate = (lf.abs() <= LAT_IN) & (sf <= R.rlen[r][:, None] + EXT) & hp[:, None]
        out["stop"] = ((((sf[:, :, None] - theta[:, None]).clamp_min(0) * mu).amax(-1) * gate).mean(1), line.any(1) & hp)
        sr, lr, _ = project(path, pok, dt[..., :2])
        v, pk = speed(dt), R.turn_g[r][:, None]
        tok = R.turn_ok[r] & hp
        with torch.no_grad():
            hpk = R.path_h[r].gather(1, pk.round().long().clamp(0, path.shape[1] - 1))
            w = (sr >= pk + TURN_WIN[0]) & (sr <= pk + TURN_WIN[1]) & (lr.abs() <= LAT_IN) & (wrap(dt[..., 2] - hpk).abs() <= TURN_HEAD) & tok[:, None]
            near = ((slr[:, None, :] - sr[:, :, None]).abs() <= 1.0) & lst[:, None, :]                    # (n, plan step, log step)
            i = sr.round().long().clamp(1, path.shape[1] - 2)
            pv = R.path_v[r].masked_fill(~pok, 1e3)
            v_grid = torch.stack([pv.gather(1, i + k) for k in (-1, 0, 1)]).amax(0)
            v_ref = torch.where(near.any(-1), vl[:, None, :].expand_as(near).masked_fill(~near, -1e6).amax(-1), v_grid)
            cap = torch.maximum(v_ref + 1.0, 1.15 * v_ref)
        out["turn"] = (((v - cap).clamp_min(0) * w).mean(1), tok)
        sp = seg_len(torch.diff(xt, dim=1, prepend=torch.zeros_like(xt[:, :1])), torch.diff(yt, dim=1, prepend=torch.zeros_like(yt[:, :1]))).cumsum(1)
        lg = torch.cat([R.log0[r][:, None, :2], R.log_fut[r][..., :2]], 1)
        slog = seg_len(lg[:, 1:, 0] - lg[:, :-1, 0], lg[:, 1:, 1] - lg[:, :-1, 1]).cumsum(1)
        op = R.open[r] & lok
        out["prog"] = ((P_K * slog - P_C - sp).clamp_min(0).mean(1) * op, op)
        return out

    def __call__(self, out, r, n_imit, n_norm):
        """out (k, n_out) of the on-policy rows r (k,) -> (the terms' share of the total, {name: scalar}); n_imit / n_norm: the number of
        imitation / normal rows among the batch's Store rows (the denominators of loss43's hinge-only rows)."""
        R, o = self.R, out.float()
        x, y, psi = self.poses(o, r)
        e = ((o[:, self.di] - R.t_out[r]) / self.tstd).pow(2) * self.dmask
        Ls = {"distill_on": (e[:, ~self.plan_cols].sum(1) / e.shape[1]).sum() / n_norm}
        total = self.lam_d * Ls["distill_on"]
        with torch.set_grad_enabled(any(k in self.on for k in TERMS)):                  # logged in every arm, differentiated where used
            vals = self.terms(x, y, psi, r)
        with torch.set_grad_enabled("imit" in self.on):
            vals["imit"] = self.imit(x, y, psi, r)
        for k, (v, ok) in vals.items():
            v = v * ok
            Ls[f"on_{k}"] = v.sum() / n_imit
            if k in self.on:
                total = total + (self.lam_i if k == "imit" else self.ho_w * self.lam) * Ls[f"on_{k}"]
            for i, p in enumerate(R.passes):
                m = R.pas_t[r] == i
                if m.any():
                    Ls |= row_stats(v[m].detach(), f"on/{p}/{k}")
                    Ls[f"on/{p}/{k}_elig"] = ok[m].float().mean()
        return total, Ls


@torch.no_grad()
def held_eval(model, R: OnRows, OL: OnLoss, rows, bs=256) -> dict:
    """The held read of one model on rows (numpy): ADE / FDE of its plan against the row's imitation target, rate / mean of every
    ON-SELF term, the 4 s arc length against the served plan and against the parent's logged plan."""
    acc = {}
    add = lambda k, v: acc.setdefault(k, []).append(v.double().cpu())  # noqa: E731
    arc = lambda px, py: seg_len(torch.diff(px, dim=1, prepend=torch.zeros_like(px[:, :1])), torch.diff(py, dim=1, prepend=torch.zeros_like(py[:, :1]))).sum(1)  # noqa: E731
    for i in range(0, len(rows), bs):
        rn = rows[i:i + bs]
        r = torch.as_tensor(rn, device=R.dev)
        o = model(R.tokens(rn).to(R.dev), R.ego[r], R.tc[r]).float()
        x, y, psi = OL.poses(o, r)
        t, ok = R.tgt[r], R.tgt_ok[r]
        d = torch.hypot(x - t[..., 0], y - t[..., 1])
        full = ok.all(1)
        add("ade_05_4", d[full].mean(1)), add("ade_1_4", d[full][:, 1:].mean(1)), add("fde_4", d[full][:, -1])
        add("imit_loss", OL.imit(x, y, psi, r)[0][ok.any(1)])
        act = torch.zeros(len(r), dtype=torch.bool, device=R.dev)
        tv = OL.terms(x, y, psi, r)
        for k, (v, e) in tv.items():
            add(f"{k}_v", v), add(f"{k}_elig", e.float())
            if k != "prog":
                act |= v > 0
        past = (project(R.path[r], R.path_ok[r], torch.stack([x, y], -1))[0] > R.rlen[r][:, None]).any(1)   # a pose past the path's last point
        add("corr_end", ((tv["corr"][0] > 0) & past).float())
        add("active", act.float())
        a, a_s, a_m = arc(x, y), arc(R.poses[r][..., 0], R.poses[r][..., 1]), arc(R.poses_model[r][..., 0], R.poses_model[r][..., 1])
        add("arc", a), add("arc_served", a_s), add("arc_parent", a_m)
    c = {k: torch.cat(v) for k, v in acc.items()}
    res = dict(n=len(rows), n_target=int(len(c["ade_05_4"])), ade_05_4=float(c["ade_05_4"].mean()), ade_1_4=float(c["ade_1_4"].mean()),
               fde_4=float(c["fde_4"].mean()), imit_loss=float(c["imit_loss"].mean()), active_rate=float(c["active"].mean()),
               arc_m=float(c["arc"].mean()), arc_ratio_served=float(c["arc"].sum() / c["arc_served"].sum()),
               arc_ratio_parent=float(c["arc"].sum() / c["arc_parent"].sum()))
    mv = c["arc_served"] > 1.0
    res["arc_ratio_served_rowmean"] = float((c["arc"][mv] / c["arc_served"][mv]).mean())
    for k in TERMS:
        v = c[f"{k}_v"]
        res |= {f"{k}_rate": float((v > 0).double().mean()), f"{k}_mean": float(v.mean()), f"{k}_elig": float(c[f"{k}_elig"].mean())}
    res["corr_end_rate"] = float(c["corr_end"].mean())           # corr > 0 on a plan that runs past the logged path's end (a longitudinal case)
    res["rate_sum"] = sum(res[f"{k}_rate"] for k in TERMS if k != "prog")
    return res
