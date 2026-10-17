"""AWR-ONPOL: consequence-weighted cloning of on-policy plans (experiments/awr_onpol/plans/2026-10-17-awr-onpol-prereg.md). Used by
experiments/route_mem/scripts/rm_train.py (--awr-pack, --awr-lam, --awr-k, --awr-mode, --awr-keep, --awr-beta, --awr-wmax, --awr-kinds).

No expert and no hand-made target: a pool row's target is the plan its own policy output at that decision (the action; ACC was part of the
environment), and the row is weighted by what happened in the simulator in the next H s against the other policies that stood in the same
state (experiments/awr_onpol/scripts/aw_score.py: score s, baseline b, advantage A = s - b, baseline kind 1 = near-identical peers,
2 = per-(scene, time) fallback).

Rows: the pack's entries of the --on pool (names asserted equal to the store's pool rows), train logs only (pool tab `dev` != '1'), A finite,
  baseline kind in --awr-kinds (default 1: near-identical peers only), finite target.
  filt (default): kept iff A > --awr-keep, drawn uniformly (filtered behaviour cloning);
  awr:            every eligible row, drawn with p proportional to min(exp(A / beta), --awr-wmax) (AWR's weights as draw probabilities: the
                  expected loss is AWR's, the per-row weight stays 1).
Loss: K = --awr-k rows per step, own rng stream [seed, 0, 167], a second forward exactly as the main batch feeds a pool row (S.ego with the
  route memory, never dropped, never an anchor; S.tc; nv); per-row imitation distance to the target (pp_train.Losses.dist: Huber on x / y / yaw
  over the 8 poses, the imitation term's own scale), summed over the K rows / the normal rows of the batch x --awr-lam: at lam 1 each drawn row
  weighs one log imitation row. The main batch, its draws and every other term are untouched.
"""
from __future__ import annotations

import numpy as np

STREAM = 167


class AwrSet:
    def __init__(self, a, S, on0, dev):
        import torch
        from pathlib import Path
        from jevdrive.common import data_dir
        p = Path(a.awr_pack) if a.awr_pack.startswith("/") else data_dir() / a.awr_pack
        z = np.load(p)
        names = np.asarray(S.tab["names"][on0:]).astype(str)
        assert np.array_equal(z["names"].astype(str), names), f"{p}: pack rows differ from the store's pool rows"
        A, kind, tgt = z["A"].astype(np.float64), z["kind"].astype(int), z["target"].astype(np.float32)
        kinds = [int(k) for k in str(a.awr_kinds).split(",")]
        ok = np.isfinite(A) & np.isin(kind, kinds) & (z["dev"].astype(str) != "1") & np.isfinite(tgt).all((1, 2))
        if a.awr_mode == "filt":
            sel = ok & (A > a.awr_keep)
            w = np.ones(int(sel.sum()))
        else:
            sel = ok
            w = np.minimum(np.exp(A[sel] / a.awr_beta), a.awr_wmax)
        idx = np.flatnonzero(sel)
        assert len(idx), f"--awr: no row ({a.awr_mode}, keep {a.awr_keep}, kinds {kinds})"
        self.p = w / w.sum()
        self.rows = on0 + idx
        t = lambda x: torch.as_tensor(np.ascontiguousarray(x), device=dev)  # noqa: E731
        self.tgt = t(tgt[idx])
        trk, pol = z["track"].astype(str)[idx], z["pol"].astype(str)[idx]
        share = lambda g: {k: float(self.p[g == k].sum()) for k in np.unique(g)}  # noqa: E731
        self.rec = dict(pack=str(p), h=float(z["h"]), mode=a.awr_mode, keep=a.awr_keep, beta=a.awr_beta, wmax=a.awr_wmax, kinds=kinds, k=a.awr_k,
                        lam=a.awr_lam, eligible=int(ok.sum()), rows=int(len(idx)), draw_share_track=share(trk), draw_share_policy=share(pol),
                        A_mean_drawn=float((self.p * A[idx]).sum()), A_pos_share_drawn=float(self.p[A[idx] > 1e-6].sum()),
                        fail_rows_drawn=float(self.p[z["fail"][idx]].sum()), scenes=int(len(np.unique(z["scene"].astype(str)[idx]))),
                        effective_rows=float(1.0 / (self.p ** 2).sum()))

    def batch(self, rng, k):
        """-> (store rows (k,) np.int64, targets (k, 8, 3) on the device)."""
        import torch
        i = rng.choice(len(self.rows), k, replace=True, p=self.p)
        return self.rows[i], self.tgt[torch.as_tensor(i, device=self.tgt.device)]

    def loss(self, model, S, rows, tgt, LS, nv=None):
        """Second forward on the drawn rows -> per-row imitation distance (k,), logging dict (ADE to the target, 4 s arc minus the target's)."""
        import torch
        import pp_train as T
        rt = torch.as_tensor(rows, device=tgt.device)
        out = model(S.front[rows], S.ego[rt], S.tc[rt], None, None, nv=None if nv is None else nv[rt])
        plan = out.float()[:, LS.pi].view(-1, 33, 15)
        cx = S.cam_x[rt]
        v = LS.dist(plan, cx, tgt[..., 0], tgt[..., 1], tgt[..., 2])
        with torch.no_grad():
            x, y, _ = T.rear(plan, cx, LS.W)
            log = {"awr/ade": torch.hypot(x - tgt[..., 0], y - tgt[..., 1]).mean(), "awr/dx4": (x[:, -1] - tgt[:, -1, 0]).mean(),
                   "awr/dy4_abs": (y[:, -1] - tgt[:, -1, 1]).abs().mean()}
        return v, log
