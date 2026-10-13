#!/usr/bin/env python3
"""LON probe (plans/2026-10-13-lon-prereg.md, "Weights"): the three terms of onpol_nuplan lib/on_lonsafe.py on the plans each checkpoint makes
at the states of an on-policy pool (the rows the arm trains on), their firing shares, and the along-heading gradient scale of each term against
the imitation term on the same rows. One GPU job; $DATA_DIR/envs/op-train/bin/python.

  rf_lonprobe.py --pool on1_L18RB0 --tags RSGboth-L18-s0 RSG-L18-s0 RSGbC2-L18-s0 --out <dir> [--rclass on1_L18RB0t]
  -> <dir>/lonprobe-<pool>.json (+ per-row npz). Per plan source (the pool's served plans = the rollout policy RB0-F-s0's, then each tag; plus
     the target `fut` itself, i.e. how often the human's own target would be charged): per term the firing share (h > 0) over all rows and by
     the row class of --rclass's tab (turn / curve / straight / none), mean h when firing. For the first tag (the plans the arm starts to see
     once it drives like RSGboth): per firing row r the along-heading gradient norms |g_imit,r| (pp_train Losses.dist on the 8 rear-axle poses
     vs `fut`, per row) and |g_term,r| (unweighted h), and w_r = |g_imit,r| / (ho_w |g_term,r|): the weight at which the term's along-heading
     push equals the imitation term's on that row (both enter the total divided by the same n_imit). Median / quartiles reported.

Store as rf_lonread.py (rm_train.build with one offline shard; the pool rows are fed exactly as in training).
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R / "experiments/route_mem/scripts"), str(_R / "experiments/onpol_nuplan/lib"), str(_R / "experiments/route_follow/scripts")]
import argparse, json  # noqa: E401,E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

import rm_train as RT  # noqa: E402  (sets the repo paths)
import on_lonsafe as LS  # noqa: E402
from rf_lonread import REC  # noqa: E402

HO_W = 3.0


def main(a):
    import torch
    import torch.nn.functional as F
    import pp_train as T
    import ot_rows as OR
    dev = torch.device("cuda")
    ns_a = SimpleNamespace(**REC, road_labels=RT.OT.JT.ROAD, on=a.pool, init="", rm_from=a.tags[0], tag=a.tags[0], exp="route_follow", lr_scale=1.0,
                           prefetch=1, dev_card_gb=4.0)
    ns, _, _, _, _ = RT.build(ns_a, dev)
    S, model, on0 = ns.S, ns.model, ns.on0
    Wt = ns.losses(ns.mode).W
    tab = np.load(OR.CR / a.pool / "tab.npz")
    pn = tab["names"]
    pack = LS.load(OR.CR / a.pool, pn, dev)
    rc = np.full(len(pn), "all", object)
    if a.rclass and (OR.CR / a.rclass / "tab.npz").exists():
        t2 = np.load(OR.CR / a.rclass / "tab.npz")
        assert np.array_equal(t2["names"], pn), "rclass pool rows differ"
        rc = t2["rclass"].astype(str)
    pi = torch.as_tensor(S.pi, device=dev)
    fut = torch.as_tensor(tab["fut"], device=dev)
    sig = [torch.as_tensor(s, dtype=torch.float32, device=dev) for s in (T.SIG_X, T.SIG_Y, T.SIG_PSI)]

    def plans_of(tag):
        if tag != a.tags[0]:
            ck = torch.load(T.proot("runs", tag) / "ckpt-final.pt", map_location="cpu", weights_only=False)
            model.load_state(ck["model"])
        model.eval()
        res = []
        with torch.no_grad():
            for i in range(0, len(pn), 256):
                rt = torch.arange(on0 + i, on0 + min(i + 256, len(pn)), device=dev)
                p = model(S.front[rt.cpu().numpy()], S.ego[rt], S.tc[rt], None, None, nv=None if ns.nv is None else ns.nv[rt]).float()[:, pi].view(-1, 33, 15)
                res.append(torch.stack(T.rear(p, S.cam_x[rt], Wt), -1))
        return torch.cat(res)

    def run_terms(P, grad=False):
        out = {k: [] for k in LS.TERMS} | {f"g_{k}": [] for k in LS.TERMS} | {"g_imit": []}
        for i in range(0, len(P), 2048):
            r = torch.arange(i, min(i + 2048, len(P)), device=dev)
            p = P[r].clone().requires_grad_(grad)
            Tm = LS.terms(p, r, pack)
            for k in LS.TERMS:
                out[k].append(Tm[k].detach().cpu().numpy())
            if grad:
                u = torch.stack([torch.cos(p[..., 2]), torch.sin(p[..., 2])], -1).detach()
                along = lambda L: (torch.autograd.grad(L.sum(), p, retain_graph=True)[0][..., :2] * u).sum(-1).norm(dim=1)  # noqa: E731
                for k in LS.TERMS:
                    out[f"g_{k}"].append(along(Tm[k]).cpu().numpy())
                f = fut[r]
                h = lambda z: F.huber_loss(z, torch.zeros_like(z), reduction="none", delta=1.0)  # noqa: E731
                di = (h((p[..., 0] - f[..., 0]) / sig[0]) + h((p[..., 1] - f[..., 1]) / sig[1]) + h((p[..., 2] - f[..., 2]) / sig[2])).mean(1)
                out["g_imit"].append(along(di).cpu().numpy())
        return {k: np.concatenate(v) for k, v in out.items() if v}

    def summ(H):
        d = {}
        for k in LS.TERMS:
            f = H[k] > 0
            d[k] = dict(share=float(f.mean()), rows=int(f.sum()), mean_when_firing=float(H[k][f].mean()) if f.any() else None,
                        by_class={c: round(float(f[rc == c].mean()), 4) for c in sorted(set(rc.tolist()))})
        d["any"] = float(np.any([H[k] > 0 for k in LS.TERMS], 0).mean())
        return d
    meta = dict(pool=a.pool, rows=int(len(pn)), stopped_targets=float(pack["stop"].float().mean()),
                classes={c: int((rc == c).sum()) for c in sorted(set(rc.tolist()))}, sources={}, grad={})
    H = {"fut (target)": run_terms(fut), "served (RB0-F-s0)": run_terms(torch.as_tensor(tab["served"], device=dev))}
    for tag in a.tags:
        H[tag] = run_terms(plans_of(tag), grad=tag == a.tags[0])
    for k, v in H.items():
        meta["sources"][k] = summ(v)
        print(k, json.dumps(meta["sources"][k]), flush=True)
    G = H[a.tags[0]]
    for k in LS.TERMS:
        f = (G[k] > 0) & (G[f"g_{k}"] > 1e-9)
        if not f.any():
            meta["grad"][k] = None
            continue
        w = G["g_imit"][f] / (HO_W * G[f"g_{k}"][f])
        q = lambda x: [round(float(np.quantile(x, z)), 5) for z in (0.25, 0.5, 0.75)]  # noqa: E731
        meta["grad"][k] = dict(rows=int(f.sum()), g_imit_q=q(G["g_imit"][f]), g_term_q=q(G[f"g_{k}"][f]), w_eq_q=q(w))
    print("grad", json.dumps(meta["grad"]), flush=True)
    out = _pl.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"lonprobe-{a.pool}.json").write_text(json.dumps(meta, indent=1))
    np.savez(out / f"lonprobe-{a.pool}.npz", names=pn, rclass=rc.astype(str), **{f"{s}|{k}": v for s, d in H.items() for k, v in d.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rclass", default="")
    main(ap.parse_args())
