"""STACK joint-arm terms for experiments/route_mem/scripts/rm_train.py (prereg experiments/route_follow/plans/2026-10-16-stack-prereg.md 4.1 - 4.5).
Every term is behind its own flag; with all of them off rm_train never builds this object (no operation added).

  --dual C1,C2,..  --dual-tau c=r,.. --dual-level F --dual-eta 0.02
        Lagrangian multipliers on the constraint terms c in {agent, road, envU, cons}: the coefficient of each term (initially the recipe's
        hand weight lam0: agent / road ho_w x 10, envU ho_w x on_lon_lam, cons ho_w x 10) follows dual ascent
        log lam += eta (r - tau) / max(tau, 0.02), lam clipped to [lam0 / 10, 10 lam0]; r = EMA (0.95) of the batch violation rate (share of
        the rows the constraint applies to with a positive value); tau = level x the reference rate (--dual-tau; `auto` = the mean rate over
        the first 25 steps x 0.5). Without --dual the coefficients stay lam0 (the recipe's).
  --cons PATH      the frozen consequence head (lib/conshead.py) on EVERY row of the batch: value relu(max p - 0.5) per row, summed / n_imit.
  --imit-drop      imitation dropout (ChauffeurNet M4): on any row where a constraint of this arm is active (value > 0), its imitation term and
        its anchor terms (cons + the plan columns' distillation of anchor rows) are removed; the row still counts in n_imit / the means.
  --pref W         one-sided minimal-correction preference on the pool (own-state) rows: K = 8 lib/stack_cand candidates around the row's
        current plan (detached); own plan infeasible = an agent or road contact at margin 0 or envU > 0.5 m/s on a stop-class envelope row;
        label = the candidate nearest to the own plan (mean pose distance) that satisfies every constraint AT ITS TRAINING MARGIN (agent / road
        hinges 0 at 0.3 m, envU 0); term W x ho_w x mean pose distance to the label, summed / n_imit. Logged: pref/share (rows with a term),
        pref/infeasible, pref/cand_<k>.
  reflex (--rx)    the module's parameters take the gradient of the constraint side only (G_C = the dual-weighted constraint terms + pref):
        rm_train computes it with one autograd.grad and overwrites what the full backward gave them.
"""
from __future__ import annotations

import copy

import numpy as np
import torch

import stack_cand as SC

CONSTRAINTS = ("agent", "road", "envU", "cons")


def _rates(spec: str) -> dict:
    out = {}
    for t in (spec.split(",") if spec else []):
        k, v = t.split("=")
        out[k] = v
    return out


class Stack:
    def __init__(self, a, ns, LS, dev):
        from agent_hinge import FRONT, REAR
        self.a, self.ns, self.LS, self.dev = a, ns, LS, dev
        self.on0 = ns.on0
        c = ns.cfg
        assert not c.retime and c.late_lat_w == 1.0, "--imit-drop re-derives pp_train's imitation term: retime / late_lat_w off"
        self.lam0 = dict(agent=a.ho_w * c.agent_lam, road=a.ho_w * a.road_lam, envU=a.ho_w * a.on_lon_lam, cons=a.ho_w * 10.0)
        self.dual = [x for x in a.dual.split(",") if x] if a.dual else []
        assert set(self.dual) <= set(CONSTRAINTS), a.dual
        self.lam = dict(self.lam0)
        self.ema, self.tau, self.auto, self.first = {}, {}, {}, {}
        tau = _rates(a.dual_tau)
        for k in self.dual:
            v = tau.get(k, "auto")
            if v == "auto":
                self.auto[k] = []
            else:
                self.tau[k] = a.dual_level * float(v)
        self.cons = None
        if a.cons:
            import conshead
            self.cons = conshead.load(a.cons, dev)
        self.pref_w = a.pref
        self.idrop = a.imit_drop
        if self.pref_w > 0:                                           # margin-0 agent geometry sharing the trainer's boxes
            ag = ns.agent2
            self.a0 = copy.copy(ag)
            self.a0.margin = self.a0.side_margin = 0.0
            self.a0.e_off, self.a0.e_hl = (FRONT + REAR) / 2, (FRONT - REAR) / 2
            self.lon = LS.lon if LS.lon is not None and LS.lon.__name__ == "on_envelope" else None
            self.stop = None
            if self.lon is not None:
                cls = np.asarray(LS.pack["cls"]).astype(str)
                self.stop = torch.as_tensor(cls == "stop", device=dev)
            self.cnt = torch.zeros(SC.K, dtype=torch.float64, device=dev)

    # ---------------------------------------------------------------- per step
    def terms(self, out, S, rows, anchor, front, total, Ls, n_main=None):
        """-> total (G_I + G_C), total_C (the constraint side, for the reflex parameters), Ls (+ logged numbers). The trainer's total already
        holds the recipe's constraint terms at lam0: they are re-weighted here (subtracted at lam0, added at lam)."""
        a, LS, c = self.a, self.LS, self.ns.cfg
        n = len(rows) if n_main is None else n_main
        out, rows, anchor = out[:n], rows[:n], anchor[:n]
        is_pool = rows >= self.on0
        imit = ~is_pool & ~anchor & S.has_fut[rows]
        n_imit = imit.sum().clamp_min(1).float()
        con, act, rate = {}, {}, {}
        if "agent_ho" in Ls:
            con["agent"], rate["agent"] = Ls["agent_ho"], Ls["agent_ho_pos"]
        if "road_ho" in Ls:
            con["road"], rate["road"] = Ls["road_ho"], Ls["road_ho_pos"]
        if "on/envU_ho" in Ls:
            con["envU"], rate["envU"] = Ls["on/envU_ho"], Ls.get("on/envU_pos", Ls["on/envU_ho"] * 0)
        x, y, psi = __import__("pp_train").rear(out.float()[:, LS.pi].view(-1, 33, 15), S.cam_x[rows], LS.W)
        P = torch.stack([x, y, psi], -1)
        if self.cons is not None:
            lg = self.cons(front[:n], S.ego[rows][:, :20].float(), P)
            pk = torch.sigmoid(lg).flatten(1).amax(1)
            vk = torch.relu(pk - 0.5)
            con["cons"] = vk.sum() / n_imit
            rate["cons"] = (vk > 0).float().mean()
            act["cons"] = vk > 0
            Ls["cons/p_mean"], Ls["cons/term"] = pk.mean().detach(), con["cons"].detach()
        total_C = torch.zeros((), device=out.device)
        for k, v in con.items():
            lam = self.lam[k]
            total = total + (lam - self.lam0[k]) * v if k != "cons" else total + lam * v
            total_C = total_C + lam * v
            Ls[f"dual/lam_{k}"] = torch.tensor(float(lam))
            Ls[f"dual/rate_{k}"] = rate[k].detach().float()
        if self.idrop and act:                                        # imitation dropout: rows with an active constraint lose imitation / anchor
            drop = torch.zeros(n, dtype=torch.bool, device=out.device)
            for v in act.values():
                drop |= v
            di, da = drop & imit, drop & anchor
            plan = out.float()[:, LS.pi].view(-1, 33, 15)
            if di.any():
                f = S.fut[rows][di]
                total = total - c.lam_i * LS.dist(plan[di], S.cam_x[rows][di], f[..., 0], f[..., 1], f[..., 2]).sum() / n_imit
            if da.any():
                n_an = anchor.sum().clamp_min(1).float()
                tx, ty, tp = __import__("pp_train").rear(S.t_plan[rows][da], S.cam_x[rows][da], LS.W)
                total = total - c.lam_c * LS.dist(plan[da], S.cam_x[rows][da], tx, ty, tp).sum() / n_an
                e = ((out.float()[da][:, LS.di] - S.t_out[rows][da]) / LS.tstd).pow(2) * LS.dmask
                total = total - c.lam_d * (e[:, LS.plan_cols].sum(1) / e.shape[1]).sum() / (~is_pool).sum().clamp_min(1)
            Ls["drop/imit_share"] = di.float().sum() / imit.sum().clamp_min(1)
            Ls["drop/anchor_rows"] = da.float().sum()
        if self.pref_w > 0 and is_pool.any():
            pv, Ls = self.pref(P, rows, is_pool, Ls)
            t = self.pref_w * a.ho_w * pv.sum() / n_imit
            total, total_C = total + t, total_C + t
            Ls["pref/term"] = t.detach()
        return total, total_C, Ls

    def _feasible(self, C, rr):
        """C (m, 8, 3) candidate poses, rr (m,) store rows -> contact at margin 0 (m,), all hinges zero at the training margins (m,)."""
        ns = self.ns
        x, y, p = C.unbind(-1)
        okA, okR = ns.agent2.ok[rr], ns.road.ok[rr]
        dist, cnt, _ = self.a0.distances(x, y, p, rr)
        contA = ((dist < 0) & cnt)[:, 1:].any((1, 2)) & okA
        hingeA = (ns.agent2.per_step(x, y, p, rr)[:, 1:] > 0).any(1) & okA
        mg = ns.road.margins(x, y, p, rr).view(len(rr), 41, 4)[:, 1:]
        contR = (mg < 0).any((1, 2)) & okR
        hingeR = (mg < ns.road.margin).any((1, 2)) & okR
        cont, viol = contA | contR, hingeA | hingeR
        if self.lon is not None:
            eu = self.lon.loss(C, rr - self.on0, self.LS.pack)["envU"]
            cont = cont | ((eu > 0.5) & self.stop[rr - self.on0])
            viol = viol | (eu > 0)
        return cont, viol

    def pref(self, P, rows, is_pool, Ls):
        K1 = SC.K + 1
        r = torch.nonzero(is_pool)[:, 0]
        with torch.no_grad():
            Pd = P[r].detach()
            C = SC.cands_t(Pd)                                        # (m, 9, 8, 3), candidate 0 = own plan
            m = len(r)
            rr = rows[r].repeat_interleave(K1)
            cont, viol = self._feasible(C.reshape(m * K1, 8, 3), rr)
            cont, viol = cont.view(m, K1), viol.view(m, K1)
            infe = cont[:, 0]
            d = torch.linalg.norm(C[:, 1:, :, :2] - Pd[:, None, :, :2], dim=-1).mean(-1)   # (m, 8)
            d = torch.where(viol[:, 1:], torch.full_like(d, float("inf")), d)
            best = d.argmin(1)
            has = infe & torch.isfinite(d.min(1).values)
            lab = C[torch.arange(m, device=P.device), best + 1]
        pv = torch.zeros(m, device=P.device)
        if has.any():
            pv[has] = torch.linalg.norm(P[r][has][..., :2] - lab[has][..., :2], dim=-1).mean(1)
            self.cnt += torch.bincount(best[has], minlength=SC.K).double()
        Ls["pref/infeasible"] = infe.float().mean()
        Ls["pref/share"] = has.float().mean()
        Ls["pref/no_label"] = (infe & ~has).float().mean()
        return pv, Ls

    def update(self, Ls, step):
        """Dual ascent after the optimiser step (rates of this batch)."""
        for k in self.dual:
            q = f"dual/rate_{k}"
            if q not in Ls:
                continue
            r = float(Ls[q])
            self.ema[k] = r if k not in self.ema else 0.95 * self.ema[k] + 0.05 * r
            if k in self.auto:
                if step < 25:
                    self.auto[k].append(r)
                    continue
                if k not in self.tau:
                    self.tau[k] = self.a.dual_level * 0.5 * float(np.mean(self.auto[k]))
            t = self.tau[k]
            lam = self.lam[k] * float(np.exp(self.a.dual_eta * (self.ema[k] - t) / max(t, 0.02)))
            self.lam[k] = float(np.clip(lam, self.lam0[k] / 10, self.lam0[k] * 10))

    def summary(self):
        out = dict(lam0=self.lam0, lam=self.lam, tau=self.tau, ema=self.ema, dual=self.dual, cons=bool(self.cons is not None), imit_drop=self.idrop,
                   pref=self.pref_w)
        if self.pref_w > 0:
            c = self.cnt.cpu().numpy()
            out["pref_labels"] = {f"{k}{v:g}": int(n) for (k, v), n in zip(SC.CANDS, c)}
        return out
