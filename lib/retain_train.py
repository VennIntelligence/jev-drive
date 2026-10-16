"""P0 retention (prereg draft experiments/wait_cap/plans/2026-10-17-retain-anticopy-draft.md, option (iii) of decision 335): LwF-style
plan distillation of the trainable policy toward frozen shipped Cinque (P0) while the base stage is rebuilt. Used by
experiments/route_mem/scripts/rm_train.py (--kd-lam, --kd-k, --kd-pool, --kd-groups, --kd-own); the desire pulses reuse rm_train's D2
(lib/desire_train.py) unchanged.

Retention set   rows the stage never exercises: the WAIT-TRAIN pool on1_WAIT (wt.py pack; 108 trainval logs, leak-checked against every
                evaluation set): W_L / W_N (the human waits), G_L / G_N (the human launches), both with their own standing history;
                S_L / S_N / Y_P / Y_V (the human stops or yields) with the constant-velocity history (rf_lonol.flat_rows). The target is
                P0's own plan on the same warp token (the pool's teacher.npz = shipped Cinque, pp_prep / on_rows make_teacher), never the
                human's: P0 waits on 0.978 of W rows and 0.334 of G rows (decision 326), so the term asks for P0's conditional behaviour,
                not for "wait". The route memory is fed as built (runs/route_mem/feat/<pool>.npz; rows without one get the empty memory).
Draw            K rows per step with replacement, own rng stream (rm_train: [seed, 0, 151]); group shares from --kd-groups (default
                W:1,G:1,SY:1: wait, launch and stop / yield one third each, uniform inside a group).
Loss            per row Losses.dist (the anchor `cons` form: Huber of the sigma-normalised rear-axle x, y, yaw errors, mean over the 8
                poses) of the plan against P0's plan; mean over the K rows x --kd-lam. A second forward on the K rows only: the stage's
                own batch, its draws and its losses are unchanged.
Own rows        --kd-own F (default 0 = off): the same distance on a share F of the non-anchor normal rows of the main forward (own rng
                [seed, 0, 157]); a classical LwF term on the stage's own rows. It pulls those rows' plans toward P0 against their
                imitation target (P0's dev ADE 1.576 vs 0.528), so it is expected to cost dev ADE.
"""
from __future__ import annotations

import numpy as np

GROUPS = {"W": ("W_L", "W_N"), "G": ("G_L", "G_N"), "SY": ("S_L", "S_N", "Y_P", "Y_V")}


def group_weights(cells, spec="W:1,G:1,SY:1"):
    """Per-row draw probabilities (n,) from the cell labels: each group of `spec` gets its share, uniform inside; cells outside every listed
    group get 0."""
    cells = np.asarray(cells).astype(str)
    sh = {k: float(v) for k, v in (t.split(":") for t in spec.split(","))}
    w = np.zeros(len(cells))
    for g, s in sh.items():
        m = np.isin(cells, GROUPS[g])
        assert m.any(), f"--kd-groups {g}: no row"
        w[m] = s / m.sum()
    return w / w.sum()


class RetainSet:
    """The retention rows on the device (ego ++ route memory, cam_x, tc, P0's plan) and their tokens (host memmap)."""

    def __init__(self, pool, dev, spec="W:1,G:1,SY:1"):
        import torch
        import pp_train as T
        from jevdrive.common import data_dir
        cr = data_dir() / "runs/op_parity/cache"
        tb = np.load(cr / pool / "tab.npz")
        tz = np.load(cr / f"{pool}@warp" / "teacher.npz")
        fz = np.load(data_dir() / "runs/route_mem/feat" / f"{pool}.npz")
        self.names, self.cells = tb["names"].astype(str), tb["cell"].astype(str)
        n = len(self.names)
        assert np.array_equal(fz["names"].astype(str), self.names), f"{pool}: route feature rows differ from the pool's"
        assert tz["plan"].shape == (n, 33, 15), f"{pool}: teacher plan {tz['plan'].shape}"
        f = fz["f"].astype(np.float32)
        f[~fz["ok"]] = 0
        t = lambda x: torch.as_tensor(np.ascontiguousarray(x), device=dev)  # noqa: E731
        self.ego = torch.cat([t(tb["ego"].astype(np.float32)), t(f)], 1)
        self.cam_x = t(tb["cam"][:, 0].astype(np.float32))
        self.tc = t(np.where(tb["lht"][:, None], [[0.0, 1.0]], [[1.0, 0.0]]).astype(np.float32))
        self.t_plan = t(tz["plan"].astype(np.float32))
        self.front = T.Tokens([cr / f"{pool}@warp" / "front.npy"], dev, host=True)
        assert self.front.nslot == [8] and int(self.front.off[-1]) == n, f"{pool}: front (N, 8, 32, 512)"
        self.p = group_weights(self.cells, spec)
        self.is_w = t(np.isin(self.cells, GROUPS["W"]))
        self.n, self.pool, self.spec = n, pool, spec
        self.rec = dict(pool=pool, rows=n, groups=spec, route_ok=float(fz["ok"].mean()),
                        draw_share={g: float(self.p[np.isin(self.cells, c)].sum()) for g, c in GROUPS.items()})

    def draw(self, rng, k):
        return rng.choice(self.n, k, replace=True, p=self.p)

    def loss(self, model, idx, LS, nv=None):
        """Second forward on rows idx -> per-row distance (k,) to P0's plan, the share of W rows whose plan waits (4 s arc < 1 m; logging)."""
        import torch
        import pp_train as T
        it = torch.as_tensor(idx, device=self.ego.device)
        out = model(self.front[idx], self.ego[it], self.tc[it], None, None, nv=nv).float()
        plan = out[:, LS.pi].view(-1, 33, 15)
        cx = self.cam_x[it]
        tx, ty, tpsi = T.rear(self.t_plan[it], cx, LS.W)
        d = LS.dist(plan, cx, tx, ty, tpsi)
        with torch.no_grad():
            x, y, _ = T.rear(plan, cx, LS.W)
            arc = torch.hypot(x[:, 0], y[:, 0]) + torch.hypot(x[:, 1:] - x[:, :-1], y[:, 1:] - y[:, :-1]).sum(1)
            w = self.is_w[it]
            wait_w = ((arc < 1.0) & w).float().sum() / w.float().sum().clamp_min(1)
        return d, wait_w


def own_loss(out, S, rows, sel, LS):
    """--kd-own: per-row distance (m,) of the main forward's plans on the selected rows to the store's P0 plan (S.t_plan)."""
    import pp_train as T
    plan = out.float()[sel][:, LS.pi].view(-1, 33, 15)
    r = rows[sel]
    tx, ty, tpsi = T.rear(S.t_plan[r], S.cam_x[r], LS.W)
    return LS.dist(plan, S.cam_x[r], tx, ty, tpsi)
