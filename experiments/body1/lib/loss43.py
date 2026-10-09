"""BODY1 arm 4.3 loss (prereg Amendment 4 items 3 and 5 and its implementation notes): P2H10's loss plus
  A  the agent hinge (lib/agent_hinge.py) on the plan of every imitation row and of every hinge-only row,
  B  hinge-only rows: off-track states (ot1 / yr1 / bd4) that carry the two hinges and no imitation target,
  C  the drivable hinge of the hinge-only rows on the scorer-layer raster (scripts/bd4_prep.py).
pp_train.Losses is subclassed, not edited. With no hinge-only row in the batch and no agent hinge the returned total is pp_train.Losses' own
(the same tensor operations in the same order).

Per-row weight: a hinge-only row counts like an on-log imitation row: its hinges are summed over the hinge-only rows and divided by the number
of imitation rows of the batch (x agent_lam / road_lam), times ho_w (prereg Amendment 5 item 3: the weight of both hinge terms of the
hinge-only rows; 1 = Amendment 4).
Logged per step (floats in the returned dict; `fam/<f>/...` per hinge-only family f, `imit/...` for the imitation rows):
  <term>_mean  mean loss over the rows;  <term>_pos  share of rows with a non-zero hinge;  <term>_posmean  mean loss on those rows (absent if none)
with <term> in {agent, road} for the families and {agent, hinge} for the imitation rows. Unconditional rates of the batch: `agent_pos` (share
of the imitation + hinge-only rows whose plan has a non-zero agent hinge), `agent_ho_pos` / `road_ho_pos` (the same over the hinge-only rows).
"""
import numpy as np
import torch

import pp_train as T


class OffAgentHinge:
    """lib/agent_hinge.AgentHinge for rows whose plan is given in a perturbed frame: the boxes of every row are moved once into the row's own
    frame (rigid: off (n, 2) = (dy, dpsi) of the row's t0 pose in the logged frame; ot_rows.to_frame), so the hinge of a normal row (off = 0)
    is AgentHinge's own, value for value. `tokens` may hold "" for rows that need no label (they are not loaded)."""

    def __new__(cls, label_file, tokens, off, dev, margin, side_margin):
        from agent_hinge import AgentHinge
        h = AgentHinge(label_file, tokens, dev, margin, side_margin)
        o = torch.as_tensor(np.asarray(off), dtype=torch.float32, device=dev)
        mv = torch.nonzero((o != 0).any(1) & h.ok)[:, 0]
        for i in range(0, len(mv), 8192):
            r = mv[i:i + 8192]
            b = h.box[r]
            dy, dp = o[r, 0][:, None, None], o[r, 1][:, None, None]
            c, s = torch.cos(dp), torch.sin(dp)
            x, y = b[..., 0], b[..., 1] - dy
            h.box[r] = torch.stack([c * x + s * y, -s * x + c * y, b[..., 2] - dp, b[..., 3], b[..., 4]], -1)
        h.moved = int(len(mv))
        return h


def row_stats(v: torch.Tensor, pre: str) -> dict:
    """v (n,) per-row hinge -> the three logged numbers (module docstring)."""
    if not len(v):
        return {}
    pos = v > 0
    out = {f"{pre}_mean": v.mean().detach(), f"{pre}_pos": pos.float().mean()}
    if pos.any():
        out[f"{pre}_posmean"] = v[pos].mean().detach()
    return out


class Losses43(T.Losses):
    """T.Losses + hinge-only rows. ho (n_store,) bool marks hinge-only Store rows, fam (n_store,) int their family index (-1 on normal rows);
    agent2 = OffAgentHinge over all rows (or None), road = ot_rows.off_hinge on the scorer-layer raster for the hinge-only rows (or None)."""

    def __init__(self, *a, ho=None, fam=None, fams=(), agent2=None, road=None, road_lam=10.0, ho_w=1.0, **k):
        super().__init__(*a, **k)
        self.ho, self.fam, self.fams, self.agent2, self.road, self.road_lam, self.ho_w = ho, fam, fams, agent2, road, road_lam, ho_w

    def __call__(self, out, S, rows, anchor):
        c = self.cfg
        h = self.ho[rows] if self.ho is not None else None
        if h is None or not h.any():
            if self.agent2 is None:
                return super().__call__(out, S, rows, anchor)
            h = torch.zeros_like(anchor)
        n = ~h
        total, Ls = super().__call__(out[n], S, rows[n], anchor[n])                      # P2H10's loss on the normal rows, unchanged
        o = out.float()
        plan = o[:, self.pi].view(-1, 33, 15)
        imit = n & ~anchor & S.has_fut[rows]
        n_imit = imit.sum().clamp_min(1)
        x, y, psi = T.rear(plan, S.cam_x[rows], self.W)
        if self.hinge is not None and imit.any():                                        # logging only: P2H10's term per row
            m = imit & self.hinge.ok[rows]
            if m.any():
                Ls |= row_stats(torch.relu(self.hinge.margin - self.hinge.margins(x[m], y[m], psi[m], rows[m])).mean(1), "imit/hinge")
        if self.agent2 is not None:
            m = (imit | h) & self.agent2.ok[rows]
            av = torch.zeros(len(rows), device=o.device)
            if m.any():
                av[m] = self.agent2.per_step(x[m], y[m], psi[m], rows[m]).mean(1)
            mi = imit & self.agent2.ok[rows]
            Ls["agent"] = av[mi].mean() if mi.any() else av.sum() * 0.0                  # A on imitation rows: AgentHinge's own mean
            total = total + c.agent_lam * Ls["agent"]
            Ls |= row_stats(av[mi], "imit/agent")
            if h.any():
                Ls["agent_ho"] = av[h].sum() / n_imit
                total = total + self.ho_w * c.agent_lam * Ls["agent_ho"]
                Ls["agent_ho_pos"] = (av[h] > 0).float().mean()
            allpos = av[(imit | h)]
            Ls["agent_pos"] = (allpos > 0).float().mean() if len(allpos) else av.sum() * 0.0
            if (allpos > 0).any():
                Ls["agent_posmean"] = allpos[allpos > 0].mean().detach()                 # the pilot gate's number
        if h.any():
            # non-plan heads of the hinge-only rows distilled to shipped on the same tokens (the plan columns are left free)
            e = ((o[h][:, self.di] - S.t_out[rows[h]]) / self.tstd).pow(2) * self.dmask
            Ls["distill_ho"] = (e[:, ~self.plan_cols].sum(1) / e.shape[1]).sum() / n.sum().clamp_min(1)   # per-row weight of a normal row
            total = total + c.lam_d * Ls["distill_ho"]
            rv = None
            if self.road is not None:
                rv = torch.zeros(len(rows), device=o.device)
                m = h & self.road.ok[rows]
                if m.any():
                    rv[m] = torch.relu(self.road.margin - self.road.margins(x[m], y[m], psi[m], rows[m])).mean(1)
                Ls["road_ho"] = rv[h].sum() / n_imit
                total = total + self.ho_w * self.road_lam * Ls["road_ho"]
                Ls["road_ho_pos"] = (rv[h] > 0).float().mean()
            f = self.fam[rows]
            for j, name in enumerate(self.fams):
                m = h & (f == j)
                if self.agent2 is not None:
                    Ls |= row_stats(av[m], f"fam/{name}/agent")
                if rv is not None:
                    Ls |= row_stats(rv[m], f"fam/{name}/road")
        return total, Ls
