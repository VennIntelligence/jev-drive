"""BODY1 progress diagnosis, which loss term shortens the plan (results/progress_diagnosis.md section 2). No weight is trained.

For a sample of TRAINING rows (shards s2 + s3, the Amendment 4 / 5 pilot's stores, built exactly as bd4_train.py builds them) and the plans of
a trained (P2H10B-F) and a base (P2H10-F) checkpoint, the gradient of every hinge term with respect to the 8 plan poses is taken per row and
split in the plan's own heading frame into along-path (timing / arc length) and cross-path (shape) components:
  imit/agent   A on imitation rows of the train logs          imit/hinge   P2H10's own drivable hinge (NAVSIM raster) on imitation rows
  <fam>/agent  A on hinge-only rows of ot1 / yr1 / bd4        <fam>/road   C on them (road-and-lane raster, note (x) fallback as in training)
  <fam>/navsim the same rows on the NAVSIM raster (what "A + B without C" would see; reported, not a trained term)
Per (checkpoint, term, speed bin):
  pos          share of rows with a non-zero hinge             v_pos        mean hinge on those rows
  along_share  sum(along^2) / sum(along^2 + cross^2) of the position gradient over the positive rows
  back         share of positive rows whose descent step moves the 4 s pose backwards (sum over poses of g . heading > 0)
  darc         first-order change of the 4 s arc length under a unit descent step on the positions, mean over ALL rows (m per unit step)
  pull         darc x the term's weight per batch (lambda x rows of the term per batch / imitation rows per batch, x w for hinge-only rows):
               the arc-length pull of the term in a training batch, comparable across terms
  cross_pull   the same weight x mean over all rows of the rms cross-path gradient (the shape pull)
  retime75/50  share of positive rows whose hinge is zero when the SAME path is driven at 0.75 / 0.5 of its arc length per step (path kept,
               timing changed): can slowing alone satisfy the term?
  shift        share of positive rows whose hinge is zero for the better of a +-1.0 m lateral ramp at 4 s (shape changed, arc kept)

  $DATA_DIR/envs/op-train/bin/python experiments/body1/scripts/prog_grad.py [--n-imit 8000 --n-ho 3000]     (GPU; through the pool)
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "lib"))
import argparse  # noqa: E402
import json  # noqa: E402
from dataclasses import replace  # noqa: E402

import numpy as np  # noqa: E402

import b1 as B  # noqa: E402

B.FAMS["bd4"] = "bd4_"
OUT = B.REPO / "experiments/body1/results/prog"
ROAD = "runs/body1/labels_road/navtrain_s01234567891011.npz"
FAMS, KK = ("ot1", "yr1", "bd4"), (4, 4, 5)
VB = [(-1, 1, "v < 1"), (1, 3, "1-3"), (3, 99, "> 3"), (-1, 99, "all")]


def retime(x, y, psi, c):
    """The same polyline (origin + 8 poses) sampled at c x its cumulative arc length per pose (torch, (n, 8) each)."""
    import torch
    z = torch.zeros_like(x[:, :1])
    X, Y, P = (torch.cat([z, q], 1) for q in (x, y, psi))
    s = torch.cat([z, torch.hypot(X[:, 1:] - X[:, :-1], Y[:, 1:] - Y[:, :-1]).cumsum(1)], 1)
    t = c * s[:, 1:]
    i = (torch.searchsorted(s.contiguous(), t.contiguous(), right=True) - 1).clamp(0, 7)
    s0, s1 = s.gather(1, i), s.gather(1, i + 1)
    w = ((t - s0) / (s1 - s0).clamp_min(1e-6)).clamp(0, 1)
    return tuple(q.gather(1, i) * (1 - w) + q.gather(1, i + 1) * w for q in (X, Y, P))


def shift(x, y, psi, a):
    """Lateral ramp of a m at 4 s (linear in time) in the plan's own heading frame."""
    import torch
    r = a * torch.arange(1, 9, device=x.device) / 8
    return x - torch.sin(psi) * r, y + torch.cos(psi) * r, psi


def main(a):
    import pandas as pd
    import torch
    import ot_rows as OR
    import pp_train as T
    from drivable_hinge import Hinge
    from loss43 import OffAgentHinge
    from experiments.op_adapt_r2.lib import op_adapt_r2 as R2
    from jevdrive.common import data_dir
    from jevdrive.data import splits
    from jevdrive.run import Run
    dev = torch.device("cuda")
    data = tuple(f"navtrain_full.s{k}of12" for k in a.shards)
    hod = tuple(f"{f}_{d}" for f in FAMS for d in data)
    cfg = T.Cfg(arm="P2", seed=0, data=data + hod, split="navsim/op-parity-full", frames="warp", host=True, hinge_lam=10.0, hinge_margin=0.3, agent_lam=10.0,
                agent_margin=0.3, agent_side_margin=0.0, agent_labels="runs/op_parity/agent_labels/navtrain_all-k32.npz")
    with Run("body1", "prog-grad", config=vars(a)) as run:
        S = T.Store(cfg.data, dev, need_side=False, frames="warp", host=True)
        names, logs = S.tab["names"], S.tab["log"]
        nd = [len(np.load(B.cache_root() / d / "tab.npz")["names"]) for d in cfg.data]
        fam = np.full(S.n, -1)
        o = sum(nd[:len(data)])
        for j in range(len(FAMS)):
            m = sum(nd[len(data) * (1 + j):len(data) * (2 + j)])
            fam[o:o + m] = j
            o += m
        is_ho = fam >= 0
        tr_rows, _, sp = T.split_rows(dict(names=np.where(is_ho, "", names), log=logs, is_b2d=S.is_b2d, is_wod=S.is_wod), cfg.split)
        trl, excl = splits.load(B.TRAIN), splits.load(B.VAL)
        for x in sp + (trl, excl):
            run.use_split(x)
        in_tr = trl.mask(logs)
        in_ho = in_tr & ~excl.mask(logs)
        hinge = Hinge([data_dir() / f for f in cfg.hinge_labels], names, dev, cfg.hinge_margin, list(cfg.hinge_footprint))
        off = OR.offsets(cfg.data)
        agent2 = OffAgentHinge(data_dir() / cfg.agent_labels, np.where(in_tr, names, ""), off, dev, cfg.agent_margin, cfg.agent_side_margin)
        road = OR.off_hinge(replace(cfg, hinge_labels=(ROAD,), hinge_margin=0.3), np.where(is_ho, names, ""), off, dev)
        nav = OR.off_hinge(cfg, np.where(is_ho, names, ""), off, dev)                 # the NAVSIM raster in the hinge-only rows' own frames
        hr = torch.nonzero(torch.as_tensor(is_ho, device=dev) & road.ok)[:, 0]        # note (x): rows that do not start on C's raster keep the NAVSIM one
        fb = []
        with torch.no_grad():
            for i in range(0, len(hr), 4096):
                r = hr[i:i + 4096]
                z = torch.zeros(len(r), 8, device=dev)
                fb.append(r[road.margins(z, z, z, r).amin(1) < 0])
        fb = torch.cat(fb)
        road.sdf[fb], road.ok[fb] = hinge.sdf[fb], hinge.ok[fb]
        rng = np.random.default_rng(0)
        has_fut = S.has_fut.cpu().numpy()
        im_pool = tr_rows[in_tr[tr_rows] & has_fut[tr_rows]]
        n_imit = (128 - sum(KK)) * (1 - cfg.d_frac) * float(has_fut[tr_rows].mean())      # imitation rows of a training batch (expected)
        groups = [("imit", rng.choice(im_pool, min(a.n_imit, len(im_pool)), replace=False), (("agent", agent2, 10.0), ("hinge", hinge, 10.0)))]
        for j, f in enumerate(FAMS):
            pool = np.flatnonzero((fam == j) & in_ho)
            w = a.ho_w * 10.0 * KK[j] / n_imit
            groups.append((f, rng.choice(pool, min(a.n_ho, len(pool)), replace=False), (("agent", agent2, w), ("road", road, w), ("navsim", nav, 0.0))))
        run.info(f"rows: imitation {len(groups[0][1])} of {len(im_pool)}; hinge-only " + ", ".join(f"{g} {len(r)}" for g, r, _ in groups[1:]) + f"; fallback rows {len(fb)}; "
                 f"expected imitation rows per batch {n_imit:.1f}")
        W = torch.as_tensor(R2.t_weights(T.T8), device=dev)

        def value(h, x, y, psi, r):                                                   # per-row hinge (n,), 0 on rows without a label
            ok = h.ok[r]
            v = torch.zeros(len(r), device=dev)
            if ok.any():
                v[ok] = (h.per_step(x[ok], y[ok], psi[ok], r[ok]) if h is agent2 else torch.relu(h.margin - h.margins(x[ok], y[ok], psi[ok], r[ok]))).mean(1)
            return v, ok
        rows = []
        for tag in a.tags:
            model = T.load_pmodel(tag, dev)
            for gname, rr, terms in groups:
                acc = {t: [] for t, *_ in terms}
                v0 = S.tab["speed"][rr]
                for i in range(0, len(rr), 256):
                    r = torch.as_tensor(rr[i:i + 256], device=dev)
                    with torch.no_grad():
                        p = model(S.front[r], S.ego[r], S.tc[r], None, None).float()[:, S.pi].view(-1, 33, 15)
                        x, y, psi = (q.detach() for q in T.rear(p, S.cam_x[r], W))
                    for t, h, _ in terms:
                        xg, yg, pg = (q.clone().requires_grad_(True) for q in (x, y, psi))
                        v, ok = value(h, xg, yg, pg, r)
                        gx, gy, gp = torch.autograd.grad(v.sum(), (xg, yg, pg), allow_unused=True)
                        gx, gy, gp = (torch.zeros_like(x) if q is None else q for q in (gx, gy, gp))
                        al, cr = gx * torch.cos(psi) + gy * torch.sin(psi), -gx * torch.sin(psi) + gy * torch.cos(psi)
                        z = torch.zeros_like(x[:, :1])
                        dx, dy = torch.diff(torch.cat([z, x], 1), dim=1), torch.diff(torch.cat([z, y], 1), dim=1)
                        n = torch.hypot(dx, dy).clamp_min(1e-4)
                        ux, uy = dx / n, dy / n                                        # unit segment directions; d arc / d p_k = u_k - u_{k+1}
                        ax, ay = ux - torch.cat([ux[:, 1:], z], 1), uy - torch.cat([uy[:, 1:], z], 1)
                        darc = -(gx * ax + gy * ay).sum(1)
                        with torch.no_grad():
                            res = [value(h, *q, r)[0] for q in (retime(x, y, psi, 0.75), retime(x, y, psi, 0.5), shift(x, y, psi, 1.0), shift(x, y, psi, -1.0))]
                        acc[t].append(torch.stack([v.detach(), ok.float(), (al ** 2).sum(1), (cr ** 2).sum(1), al.sum(1), darc, cr.pow(2).mean(1).sqrt(), al[:, -1],
                                                   res[0], res[1], torch.minimum(res[2], res[3]), gp.abs().sum(1)], 1).cpu().numpy())
                for t, h, w in terms:
                    A = np.concatenate(acc[t])
                    for lo, hi, vn in VB:
                        m = (v0 > lo) & (v0 <= hi) & (A[:, 1] > 0)
                        if m.sum() < 20:
                            continue
                        q, pos = A[m], A[m][:, 0] > 0
                        P_ = q[pos]
                        rows.append(dict(ckpt=tag, term=f"{gname}/{t}", v=vn, n=int(m.sum()), pos=float(pos.mean()), n_pos=int(pos.sum()), v_pos=float(P_[:, 0].mean()) if pos.any() else np.nan,
                                         along_share=float(P_[:, 2].sum() / max(P_[:, 2].sum() + P_[:, 3].sum(), 1e-12)) if pos.any() else np.nan,
                                         back=float((P_[:, 4] > 0).mean()) if pos.any() else np.nan, shorter=float((P_[:, 5] < 0).mean()) if pos.any() else np.nan,
                                         darc=float(q[:, 5].mean()), weight=w, pull=float(w * q[:, 5].mean()), cross_pull=float(w * q[:, 6].mean()),
                                         along4_pos=float(P_[:, 7].mean()) if pos.any() else np.nan,
                                         retime75=float((P_[:, 8] == 0).mean()) if pos.any() else np.nan, retime50=float((P_[:, 9] == 0).mean()) if pos.any() else np.nan,
                                         shift=float((P_[:, 10] == 0).mean()) if pos.any() else np.nan))
            del model
        D = pd.DataFrame(rows)
        OUT.mkdir(parents=True, exist_ok=True)
        D.to_csv(OUT / "grad_terms.csv", index=False, float_format="%.5f")
        (OUT / "grad_meta.json").write_text(json.dumps(dict(tags=a.tags, shards=a.shards, n_imit_per_batch=n_imit, ho_w=a.ho_w, fallback_rows=int(len(fb)),
                                                            rows={g: int(len(r)) for g, r, _ in groups}), indent=1) + "\n")
        run.info("\n" + D[D.v == "all"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        run.summary.update(n_rows=int(sum(len(r) for _, r, _ in groups)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["P2H10-F-s0", "P2H10B-F-s0", "P2H10-F-s1", "P2H10B-F-s1"])
    ap.add_argument("--shards", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--n-imit", type=int, default=8000)
    ap.add_argument("--n-ho", type=int, default=3000)
    ap.add_argument("--ho-w", type=float, default=3.0)
    main(ap.parse_args())
