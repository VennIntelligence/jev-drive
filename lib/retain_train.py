"""P0 retention, longitudinal and targeted (prereg draft experiments/wait_cap/plans/2026-10-17-retain-anticopy-draft.md, decision 335 option
(iii) as re-aimed by decision 336): distil frozen shipped Cinque's (P0's) SPEED PROFILE, not its path, on the frames where P0 stops on arrival
and the adapter stages do not. Used by experiments/route_mem/scripts/rm_train.py (--kd-lam, --kd-k, --kd-pool, --kd-groups, --kd-nav,
--kd-nav-k, --kd-mode); the desire pathway reuses rm_train's D2 (lib/desire_train.py) unchanged.

Why not the full plan: P0 is a poor driver laterally (held 460: 213 zeros, 98 offroad) and its waiting is partly unconditional (decision 326:
it waits on 0.334 of the frames where the human goes). What the adapters lost is P0's stop on ARRIVAL (decision 336: at the rolling handoff of
the static scene P0 plans 0.43 m over 4 s, the WAIT-trained models 2.0-2.3 m) and the stop behind a stopped car with a flat history (decision
331's S_L cell, P0 0.845 -> P2-F 0.406).

Rows (one second forward per step, K pool rows + K_nav navtrain rows; the stage's own batch, draws and losses are unchanged):
  pool  rows of the WAIT-TRAIN pool on1_WAIT (108 trainval logs, leak-checked against every evaluation set), by --kd-groups (default AR:2,W:1):
        AR  rolling arrival: 0.2 <= v0 <= 6 m/s and W_L (the human comes to a stop within 1 m behind a stopped lead), S_L (stops behind a
            stopped car), S_N (stops, no lead: stop lines), Y_P / Y_V (yields); S / Y rows carry the constant-velocity history (wt.py pack,
            rf_lonol.flat_rows), i.e. an arrival with no braking yet in the history
        W   standing with the human waiting (W_L / W_N, v0 < 0.2): P0's hold (0.978)
        G   standing with the human launching (G_L / G_N): P0 waits on a third of them; off by default
  nav   --kd-nav FILE (nx.py approach): navtrain train-split rows with 0.2 <= v0 <= 6 m/s and a stopped lead (< 0.5 m/s) <= 15 m ahead in the
        lane corridor, or a closing lead with time-to-contact < 4 s; fed the constant-velocity history (vy = ax = ay = 0, the past poses
        respaced at v0 along their own polyline, parity_adapter.cv_history), so the stop has to come from the frames
Loss  --kd-mode lon (default): the plan's own cumulative arc length at the 8 poses against P0's (wc_arc.arc: rotation invariant, the gradient
        moves the poses only along the plan's own segments, the lateral path is untouched), smooth L1 (beta 0.5) summed over the poses (the
        wc_arc scale), mean over the drawn rows x --kd-lam. --kd-mode full: the anchor `cons` distance to P0's whole plan (x, y, yaw).
Draws use rm_train's rng stream [seed, 0, 151].
"""
from __future__ import annotations

import numpy as np

GROUPS = {"W": ("W_L", "W_N"), "G": ("G_L", "G_N"), "SY": ("S_L", "S_N", "Y_P", "Y_V")}
AR_CELLS, V_LO, V_HI, V_STAND = ("W_L", "S_L", "S_N", "Y_P", "Y_V"), 0.2, 6.0, 0.2
BETA, EPS = 0.5, 1e-6


def group_masks(cells, v0):
    cells, v0 = np.asarray(cells).astype(str), np.asarray(v0, float)
    roll = (v0 >= V_LO) & (v0 <= V_HI)
    return dict(AR=np.isin(cells, AR_CELLS) & roll, W=np.isin(cells, GROUPS["W"]) & (v0 < V_STAND), G=np.isin(cells, GROUPS["G"]),
                SY=np.isin(cells, GROUPS["SY"]))


def group_weights(cells, v0, spec="AR:2,W:1"):
    """Per-row draw probabilities (n,): each group of `spec` gets its share, uniform inside (a row in two groups takes both)."""
    M = group_masks(cells, v0)
    w = np.zeros(len(cells))
    for g, s in ((k, float(v)) for k, v in (t.split(":") for t in spec.split(","))):
        assert M[g].any(), f"--kd-groups {g}: no row"
        w[M[g]] += s / M[g].sum()
    return w / w.sum(), {g: int(M[g].sum()) for g in M}


def arc(x, y):
    """(n, 8) rear-axle x, y -> (n, 8) cumulative arc length (experiments/wait_cap/lib/wc_arc.arc)."""
    import torch
    dx = torch.diff(x, dim=1, prepend=torch.zeros_like(x[:, :1]))
    dy = torch.diff(y, dim=1, prepend=torch.zeros_like(y[:, :1]))
    return (torch.sqrt(dx * dx + dy * dy + EPS) - EPS ** 0.5).cumsum(1)


def flat_history(ego):
    """Fed rows (n, >= 20) -> the same rows with the constant-velocity history: vy = ax = ay = 0, the past poses respaced at |vx| along their
    own polyline (parity_adapter.cv_history; the torch form of rf_lonol.flat_rows / vhist.respace); the route memory columns unchanged."""
    import parity_adapter as PA
    e = ego.clone()
    e[:, 5:8] = 0
    e[:, 8:20] = PA.cv_history(e[:, :20])
    return e


class RetainSet:
    """The pool rows (on the device: ego ++ route memory, cam_x, tc, P0's plan; tokens from the host memmap) and the navtrain approach rows."""

    def __init__(self, pool, dev, spec, S=None, nav_file="", n_base=0, train_rows=None, layout=""):
        """layout: the pool's route features, "" = feat/<pool>.npz (the projected layout), "a" = feat/<pool>.a.npz (rm_train --rm-anchor)."""
        import torch
        import pp_train as T
        from jevdrive.common import data_dir
        t = lambda x: torch.as_tensor(np.ascontiguousarray(x), device=dev)  # noqa: E731
        self.rec, self.n = dict(mode=None), 0
        if pool and spec:
            cr = data_dir() / "runs/op_parity/cache"
            tb = np.load(cr / pool / "tab.npz")
            tz = np.load(cr / f"{pool}@warp" / "teacher.npz")
            fz = np.load(data_dir() / "runs/route_mem/feat" / (f"{pool}.{layout}.npz" if layout else f"{pool}.npz"))
            names, cells = tb["names"].astype(str), tb["cell"].astype(str)
            n = len(names)
            assert np.array_equal(fz["names"].astype(str), names), f"{pool}: route feature rows differ from the pool's"
            assert tz["plan"].shape == (n, 33, 15), f"{pool}: teacher plan {tz['plan'].shape}"
            f = fz["f"].astype(np.float32)
            f[~fz["ok"]] = 0
            self.ego = torch.cat([t(tb["ego"].astype(np.float32)), t(f)], 1)
            self.cam_x = t(tb["cam"][:, 0].astype(np.float32))
            self.tc = t(np.where(tb["lht"][:, None], [[0.0, 1.0]], [[1.0, 0.0]]).astype(np.float32))
            self.t_plan = t(tz["plan"].astype(np.float32))
            self.front = T.Tokens([cr / f"{pool}@warp" / "front.npy"], dev, host=True)
            assert self.front.nslot == [8] and int(self.front.off[-1]) == n, f"{pool}: front (N, 8, 32, 512)"
            self.p, cnt = group_weights(cells, tb["speed"], spec)
            self.is_w = t(np.isin(cells, GROUPS["W"]))
            self.n = n
            self.rec |= dict(pool=pool, rows=n, groups=spec, group_rows=cnt, route_ok=float(fz["ok"].mean()))
        self.nav = np.zeros(0, np.int64)
        if nav_file:                                                   # navtrain approach rows (store rows, train split)
            z = np.load(nav_file if nav_file.startswith("/") else data_dir() / nav_file)
            pos = {x: i for i, x in enumerate(np.asarray(S.tab["names"][:n_base]).astype(str).tolist())}
            r = np.array([pos.get(x, -1) for x in z["names"].astype(str).tolist()], np.int64)
            r = r[r >= 0]
            self.nav = np.intersect1d(r, train_rows)
            self.rec |= dict(nav_file=str(nav_file), nav_listed=int(len(z["names"])), nav_rows=int(len(self.nav)))
            assert len(self.nav), "--kd-nav: no train-split row"

    def batch(self, rng, k, k_nav, S):
        """-> (front, ego, tc, cam_x, P0 plan, is_wait_row) of the drawn rows (pool rows first, then the navtrain rows with the flat history)."""
        import torch
        parts = []
        if k and self.n:
            i = rng.choice(self.n, k, replace=True, p=self.p)
            it = torch.as_tensor(i, device=self.ego.device)
            parts.append((self.front[i], self.ego[it], self.tc[it], self.cam_x[it], self.t_plan[it], self.is_w[it]))
        if k_nav and len(self.nav):
            r = self.nav[rng.integers(0, len(self.nav), k_nav)]
            rt = torch.as_tensor(r, device=S.ego.device)
            parts.append((S.front[r], flat_history(S.ego[rt].float()), S.tc[rt], S.cam_x[rt], S.t_plan[rt].float(),
                          torch.zeros(len(r), dtype=torch.bool, device=S.ego.device)))
        return [torch.cat([p[j] for p in parts]) for j in range(6)]

    def loss(self, model, b, LS, mode="lon"):
        """Second forward on the drawn rows -> per-row loss (n,), logging dict (plan - P0 4 s arc on the arrival rows, wait share on W rows)."""
        import torch
        import torch.nn.functional as F
        import pp_train as T
        front, ego, tc, cx, tp, isw = b
        plan = model(front, ego, tc, None, None, nv=None).float()[:, LS.pi].view(-1, 33, 15)
        tx, ty, tpsi = T.rear(tp, cx, LS.W)
        x, y, psi = T.rear(plan, cx, LS.W)
        if mode == "full":
            v = LS.dist(plan, cx, tx, ty, tpsi)
        else:
            d = arc(x, y) - arc(tx, ty).detach()
            v = F.smooth_l1_loss(d, torch.zeros_like(d), beta=BETA, reduction="none").sum(1)
        with torch.no_grad():
            a4, t4 = arc(x, y)[:, -1], arc(tx, ty)[:, -1]
            log = {"kd/arc4_minus_p0": (a4 - t4).mean(), "kd/arc4_p0": t4.mean()}
            if isw.any():
                log["kd/wait_W"] = (a4[isw] < 1.0).float().mean()
        return v, log
