"""BODY1 log-consistent margins for the two new hinges of P2H10S (plans/2026-10-11-logmargin-pilot-prereg.md): the "the log never violates"
rule of research/self-consequence/design.md section 3 applied to the body lesson S.

For every row r and new hinge X in {agent, road}:  m_X,r = min(m_X0, c_X,r - SLACK), no lower clip (may be negative), where c_X,r is the
minimum clearance of the row's LOGGED path to the hinge's labels, measured with the hinge's own geometry:
  logged path  [logged t0 pose; the 8 logged future poses S.fut], densified to 0.1 s by the plan's interpolation (41 steps). The logged t0 pose
               is the origin on a log row; on an off-track row (ot1 / yr1 / bd4) it is (0, 0, 0) of the logged frame moved by the row's `off`
               (lib/route.log_t0, design.md 3.1), so the logged path of a hinge-only row is the same world path as its source log row's.
  agent        min over the 41 steps and the counted objects of AgentHinge's signed distance (extended ego box, boxes with motion, the counted
               mask of the logged pose), in the row's own frame (OffAgentHinge's moved boxes); +inf when the logged path counts no object
  road         min over the 41 steps x 4 footprint corners of the row's road raster SDF (after bd4_train note (x)'s fallback), the path
               mapped into the raster's logged frame (ot_rows.off_hinge's mapping)
m_X0 are the current margins: agent 0.3 m inside the ego corridor and 0 m outside (the cap applies to both), road 0.3 m. With these margins
the logged path gives zero hinge by construction; `g0` checks it through the same functions the loss calls.
"""
import numpy as np
import torch

SLACK = 0.05                                     # m below the logged path's own clearance
TOL = 1e-3                                       # m: hinge-only row against its source log row, "equal up to the cache rounding"
QS = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95)


def log_t0(off):
    """off (n, 2) = (dy, dpsi) -> the logged t0 pose (n, 3) in the row's own frame (torch twin of lib/route.log_t0)."""
    dy, dp = off[:, 0], off[:, 1]
    return torch.stack([-torch.sin(dp) * dy, -torch.cos(dp) * dy, -dp], -1)


def to_log(x, y, psi, off):
    """Poses (n, K) in the row's own frame -> the same poses in the logged frame (ot_rows.off_hinge's mapping)."""
    dy, dp = off[:, :1], off[:, 1:]
    c, s = torch.cos(dp), torch.sin(dp)
    return c * x - s * y, dy + s * x + c * y, psi + dp


def agent_steps(h, x, y, psi, rows, cap=None, p0=None):
    """(B, 41) agent hinge per step: AgentHinge.per_step with the margin capped per row (cap (B,): m = min(m0, cap), m0 = margin in the
    ego corridor, side margin outside) and the densification started at p0 (B, 3). cap = p0 = None is per_step itself."""
    if cap is None and p0 is None:
        return h.per_step(x, y, psi, rows)
    dist, counted, corridor = h.distances(x, y, psi, rows, p0)
    m = torch.where(corridor, h.margin, h.side_margin) if h.side_margin != h.margin else torch.full_like(dist, h.margin)
    if cap is not None:
        m = torch.minimum(m, cap[:, None, None])
    return (m - dist).clamp_min(0).masked_fill(~counted, 0.0).amax(-1)


def road_log(road, x, y, psi, rows, off):
    """Road SDF at the footprint corners (B, 164) of a path given in the row's frame and densified from the LOGGED t0 pose, which is the
    raster's origin: drivable_hinge.Hinge.margins on the path mapped into the logged frame (OffHinge.margins starts at the row's own t0)."""
    from drivable_hinge import Hinge
    return Hinge.margins(road, *to_log(x, y, psi, off), rows)


@torch.no_grad()
def clearances(S, agent2, road, off, rows_a, rows_r, bs=2048):
    """c_agent, c_road (S.n,) on the device: the logged path's clearance (module docstring) on rows_a / rows_r (long tensors), +inf elsewhere.
    Also returns the step (0..40) where each minimum sits (-1 where +inf), for the hinge-only check."""
    dev = S.fut.device
    ca, cr = (torch.full((S.n,), float("inf"), device=dev) for _ in range(2))
    ka, kr = (torch.full((S.n,), -1, dtype=torch.long, device=dev) for _ in range(2))
    for i in range(0, len(rows_a), bs):
        r = rows_a[i:i + bs]
        F = S.fut[r]
        dist, counted, _ = agent2.distances(F[..., 0], F[..., 1], F[..., 2], r, log_t0(off[r]))
        v, k = dist.masked_fill(~counted, float("inf")).amin(2).min(1)
        ca[r], ka[r] = v, torch.where(torch.isfinite(v), k, -1)
    for i in range(0, len(rows_r), bs):
        r = rows_r[i:i + bs]
        F = S.fut[r]
        v, k = road_log(road, F[..., 0], F[..., 1], F[..., 2], r, off[r]).min(1)
        cr[r], kr[r] = v, k // 4
    return ca, cr, ka, kr


def margins(ca, cr, road_margin):
    """-> (agent cap (n,) = c - SLACK, +inf where nothing is counted; road margin (n,) = min(road_margin, c - SLACK))."""
    return ca - SLACK, (cr - SLACK).clamp_max(road_margin)


def src_rows(data, fams, nd):
    """Store row each row was built from (itself on log rows), for bd4_train's layout data = base dirs + <fam>_<base dir> (family-major):
    the hinge-only caches carry `src_row`, the row in their base dir's tab."""
    from b1 import cache_root
    nb, o = len(data), np.cumsum([0] + list(nd))
    src = np.arange(o[-1])
    for j, f in enumerate(fams):
        for i, d in enumerate(data):
            k = nb * (1 + j) + i
            s = np.load(cache_root() / f"{f}_{d}" / "tab.npz")["src_row"]
            assert len(s) == o[k + 1] - o[k]
            src[o[k]:o[k + 1]] = s + o[i]
    return src


def _dist(v):
    """Quantiles of the lowered margins (QS), as floats."""
    return {f"q{int(q * 100):02d}": float(np.quantile(v, q)) for q in QS} if len(v) else {}


@torch.no_grad()
def g0(S, agent2, road, off, ca, cr, ka, kr, cap_a, m_r, pop_a, pop_r, fam, fams, src, turn, bs=2048):
    """G0 of the pre-registration on the rows a pilot batch can carry each hinge on (pop_a: imitation rows of the train logs + hinge-only pool
    rows with an agent label; pop_r: hinge-only pool rows with a road label; numpy row indices). -> dict (json-ready).
      b      gate: the logged path, through the loss's own functions with the new margins, gives zero hinge on >= 99.5 % of the rows (each hinge)
      b_ho   gate: a hinge-only row's c equals its source log row's (both +inf or |difference| <= TOL) on >= 99.5 % of the hinge-only rows
      b_op   reported: the logged future fed as the plan of the training loss (densified from the row's OWN t0: the perturbed pose on hinge-only
             rows), share of rows with a non-zero hinge, with the new margins and with S's fixed margins; on hinge-only rows also the clearance
             of that path (c_op, the other reading of the definition) against c: share with c_op < c - TOL and the size of the difference
      c      diagnostic: share of rows whose logged path violates S's fixed margin, share whose margin is lowered, the lowered margins
    """
    dev = S.fut.device
    T = lambda a: torch.as_tensor(a, device=dev)                                            # noqa: E731
    name = np.array(["log"] + list(fams))[fam + 1]
    res = dict(slack=SLACK, tol=TOL, agent_margin=agent2.margin, agent_side_margin=agent2.side_margin, road_margin=road.margin)

    def run(rows, fn):
        out = []
        for i in range(0, len(rows), bs):
            r = T(rows[i:i + bs])
            F = S.fut[r]
            out.append(fn(r, F[..., 0], F[..., 1], F[..., 2]).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)
    lt = lambda r: log_t0(off[r])                                                           # noqa: E731
    H = {"agent": dict(rows=pop_a,
                       new=run(pop_a, lambda r, x, y, p: agent_steps(agent2, x, y, p, r, cap_a[r], lt(r)).amax(1)),
                       fix=run(pop_a, lambda r, x, y, p: agent_steps(agent2, x, y, p, r, None, lt(r)).amax(1)),
                       new_op=run(pop_a, lambda r, x, y, p: agent_steps(agent2, x, y, p, r, cap_a[r]).amax(1)),
                       fix_op=run(pop_a, lambda r, x, y, p: agent2.per_step(x, y, p, r).amax(1)),
                       c_op=run(pop_a, lambda r, x, y, p: (lambda d: d[0].masked_fill(~d[1], float("inf")).amin((1, 2)))(agent2.distances(x, y, p, r))),
                       c=ca[T(pop_a)].cpu().numpy() if len(pop_a) else np.zeros(0)),
         "road": dict(rows=pop_r,
                      new=run(pop_r, lambda r, x, y, p: torch.relu(m_r[r][:, None] - road_log(road, x, y, p, r, off[r])).amax(1)),
                      fix=run(pop_r, lambda r, x, y, p: torch.relu(road.margin - road_log(road, x, y, p, r, off[r])).amax(1)),
                      new_op=run(pop_r, lambda r, x, y, p: torch.relu(m_r[r][:, None] - road.margins(x, y, p, r)).amax(1)),
                      fix_op=run(pop_r, lambda r, x, y, p: torch.relu(road.margin - road.margins(x, y, p, r)).amax(1)),
                      c_op=run(pop_r, lambda r, x, y, p: road.margins(x, y, p, r).amin(1)),
                      c=cr[T(pop_r)].cpu().numpy() if len(pop_r) else np.zeros(0))}
    # the source log row's clearance of every hinge-only row: agent on the source row itself (its own labels, logged frame); road on the
    # hinge-only row's raster (the same token's; the source row carries no road label), with the source row's logged future and off = 0
    ho_a = pop_a[fam[pop_a] >= 0]
    ho_r = pop_r
    zero = torch.zeros((1, 2), device=dev)
    c_src_a = ca[T(src[ho_a])].cpu().numpy() if len(ho_a) else np.zeros(0)
    k_src_a = ka[T(src[ho_a])].cpu().numpy() if len(ho_a) else np.zeros(0, int)
    src_r = []
    for i in range(0, len(ho_r), bs):
        r, s = T(ho_r[i:i + bs]), T(src[ho_r[i:i + bs]])
        F = S.fut[s]
        v, k = road_log(road, F[..., 0], F[..., 1], F[..., 2], r, zero.expand(len(r), 2)).min(1)
        src_r.append(torch.stack([v, (k // 4).float()], 1).cpu().numpy())
    src_r = np.concatenate(src_r) if src_r else np.zeros((0, 2))
    chk = {"agent": (ho_a, H["agent"]["c"][fam[pop_a] >= 0], c_src_a, ka[T(ho_a)].cpu().numpy() if len(ho_a) else np.zeros(0, int), k_src_a),
           "road": (ho_r, H["road"]["c"], src_r[:, 0], kr[T(ho_r)].cpu().numpy() if len(ho_r) else np.zeros(0, int), src_r[:, 1].astype(int))}
    gate = True
    for X, h in H.items():
        rows, c = h["rows"], h["c"]
        nm, at = name[rows], np.abs(turn[rows]) > 45
        hob = fam[rows] >= 0
        exc = np.flatnonzero(h["new"] > 0)
        c0_ = 0.3 if X == "road" else agent2.margin
        b = dict(n=int(len(rows)), zero_share=float(1 - len(exc) / max(len(rows), 1)), n_exceptions=int(len(exc)),
                 exceptions=[dict(row=int(rows[j]), fam=str(nm[j]), hinge=float(h["new"][j]), c=float(c[j])) for j in exc[:50]])
        b["pass"] = bool(b["zero_share"] >= 0.995)
        rr, cc, cs, kh, ks = chk[X]
        eq = (np.isinf(cc) & np.isinf(cs)) | (np.abs(cc - cs) <= TOL)
        fin = np.isfinite(cc) & np.isfinite(cs)
        bad = np.flatnonzero(~eq)
        bh = dict(n=int(len(rr)), equal_share=float(eq.mean()) if len(rr) else float("nan"), n_unequal=int(len(bad)),
                  max_abs_diff_equal=float(np.abs(cc - cs)[eq & fin].max()) if (eq & fin).any() else 0.0,
                  unequal_min_in_first_segment=int(((kh[bad] < 5) | (ks[bad] < 5)).sum()),
                  unequal=[dict(row=int(rr[j]), fam=str(name[rr[j]]), src=int(src[rr[j]]), c=float(cc[j]), c_src=float(cs[j]), step=int(kh[j]),
                                step_src=int(ks[j])) for j in bad[:50]])
        bh["pass"] = bool(len(rr) == 0 or bh["equal_share"] >= 0.995)
        co, ch = h["c_op"][hob], c[hob]
        lower = co < ch - TOL
        bh["op_reading"] = dict(n=int(hob.sum()), c_op_lower_share=float(lower.mean()) if hob.any() else float("nan"),
                                c_minus_c_op=_dist(np.minimum(ch[lower], 10.0) - co[lower]),
                                c_op_lower_and_lowered=float((lower & (co < c0_ + SLACK)).mean()) if hob.any() else float("nan"))
        subsets = [("all", np.ones(len(rows), bool))] + ([("log rows", ~hob)] if (~hob).any() else []) + [("hinge-only", hob)] + \
                  [(f"fam {f}", nm == f) for f in fams] + [("> 45 deg", at), ("> 45 deg, hinge-only", at & hob)] + \
                  ([("> 45 deg, log rows", at & ~hob)] if (at & ~hob).any() else [])
        c0 = 0.3 if X == "road" else agent2.margin
        cs_ = {}
        for lab, m in subsets:
            if not m.any():
                continue
            low = c[m] < c0 + SLACK
            mm = np.minimum(c0, c[m][low] - SLACK)
            cs_[lab] = dict(n=int(m.sum()), viol_fixed=float((h["fix"][m] > 0).mean()), c_below_m0=float((c[m] < c0).mean()),
                            lowered=float(low.mean()), negative=float((c[m] < SLACK).mean()), lowered_margin=_dist(mm),
                            op_fixed=float((h["fix_op"][m] > 0).mean()), op_new=float((h["new_op"][m] > 0).mean()))
        res[X] = dict(b=b, b_ho=bh, c=cs_)
        gate &= b["pass"] and bh["pass"]
    res["gate"] = bool(gate)
    return res


def g0_md(res) -> str:
    """The G0 tables of the results page."""
    L = ["| Hinge | G0.b: logged path, new margins, zero hinge | line | G0.b (ii): hinge-only c = source log row's c | line |", "|:--|:--|:--|:--|:--|"]
    for X in ("agent", "road"):
        b, h = res[X]["b"], res[X]["b_ho"]
        L.append(f"| {X} | {100 * b['zero_share']:.2f} % of {b['n']} rows ({b['n_exceptions']} exceptions) | >= 99.5 % | "
                 f"{100 * h['equal_share']:.2f} % of {h['n']} (max difference among equal {h['max_abs_diff_equal']:.1e} m; {h['n_unequal']} unequal, "
                 f"{h['unequal_min_in_first_segment']} with a minimum in the first 0.5 s) | >= 99.5 % |")
    L += ["", "| Hinge | rows | n | S's margin violated by the logged path | c < m0 | margin lowered | margin < 0 | lowered margin q05 / q25 / q50 / q75 (m) | "
          "logged future as the plan from the row's own t0: non-zero with S's / the new margins |", "|:--|:--|--:|--:|--:|--:|--:|:--|:--|"]
    for X in ("agent", "road"):
        for lab, s in res[X]["c"].items():
            q = s["lowered_margin"]
            qq = " / ".join(f"{q[k]:+.2f}" for k in ("q05", "q25", "q50", "q75")) if q else "-"
            L.append(f"| {X} | {lab} | {s['n']} | {100 * s['viol_fixed']:.2f} % | {100 * s['c_below_m0']:.2f} % | {100 * s['lowered']:.2f} % | "
                     f"{100 * s['negative']:.2f} % | {qq} | {100 * s['op_fixed']:.2f} % / {100 * s['op_new']:.2f} % |")
    L += ["", "| Hinge | hinge-only rows | the other reading (path from the row's own t0) has a lower clearance | ... and would lower the margin | c - c_op q25 / q50 / q75 (m) |",
          "|:--|--:|--:|--:|:--|"]
    for X in ("agent", "road"):
        o = res[X]["b_ho"]["op_reading"]
        q = o["c_minus_c_op"]
        qq = " / ".join(f"{q[k]:.2f}" for k in ("q25", "q50", "q75")) if q else "-"
        L.append(f"| {X} | {o['n']} | {100 * o['c_op_lower_share']:.2f} % | {100 * o['c_op_lower_and_lowered']:.2f} % | {qq} |")
    return "\n".join(L) + "\n"
