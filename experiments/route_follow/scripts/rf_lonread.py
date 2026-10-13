#!/usr/bin/env python3
"""STOPLINE offline read (plans/2026-10-13-stopline-prereg.md, "Offline read"): the longitudinal hinges of onpol_nuplan lib/on_lon.py (B stop
line, P progress guard) on the plans that each checkpoint makes at the states of an on-policy pool, against the pool's lon pack
(scripts/on_lon_pack.py). One GPU job; $DATA_DIR/envs/op-train/bin/python.

  rf_lonread.py --pool on1_L18RSGallheld --tags RSG-L18-s0 RSG-L18-s1 RSGB-L18-s0 RSGB-L18-s1 --out <dir>
  -> <dir>/lonread-<pool>.json (+ per-row npz): per tag, over the pool rows with labels (pack ok): B active share / rows / scenes / mean
     when active, the same on the rows with a red eligible line in the 4 s window (`b_mu` any), P active share / mean; the pool's stored
     served plans as the first column (the rollout policy's own plans at those states).

The store is rm_train.build's (on_train.setup + route branch + route features) with one offline shard (--data navtrain_full.s0of12,
--yr-mass 0): the offline rows only fill the store, nothing is trained; the pool rows are fed exactly as in training (route input on,
8 slots unless the pool says fewer). Checkpoints with the branch are loaded the way build() loads --rm-from.
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R / "experiments/route_mem/scripts"), str(_R / "experiments/onpol_nuplan/lib")]
import argparse, json  # noqa: E401,E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

import rm_train as RT  # noqa: E402  (sets the repo paths)
import on_lon as OL  # noqa: E402

REC = dict(data=["navtrain_full.s0of12"], split="navsim/op-parity-full", hinge_lam=10.0, hinge_margin=0.3, ho="", ho_w=3.0, ho_excl="", shape=True,
           agent_lam=10.0, agent_margin=0.3, agent_side_margin=0.0, agent_labels="runs/op_parity/agent_labels/navtrain_all-k32.npz", road_lam=10.0,
           road_margin=0.3, label_bank=True, yr_mass=0.0, agent_on_only=True, on_k=13, on_mode="self", on_split="", on_lon="", on_lon_lam=10.0, on_road="",
           ego_drop=0.0, ego_drop_speed=0.0, batch=128, steps=1, warmup=1, eval_every=1, seed=0, rm_lr=3e-4, rm_warmup=100, rm_drop=0.25)


def main(a):
    import torch
    import pp_train as T
    from jevdrive.common import data_dir
    dev = torch.device("cuda")
    import ot_rows as OR
    ns_a = SimpleNamespace(**REC, road_labels=RT.OT.JT.ROAD, on=a.pool, init="", rm_from=a.tags[0], tag=a.tags[0], exp="route_follow", lr_scale=1.0,
                           prefetch=1, dev_card_gb=4.0)
    ns, _, _, _, _ = RT.build(ns_a, dev)
    S, model, on0 = ns.S, ns.model, ns.on0
    W = ns.losses(ns.mode).W
    pn = np.load(OR.CR / a.pool / "tab.npz")["names"]
    pack = OL.load(OR.CR / a.pool, pn, dev)
    z = np.load(OL.pack_path(OR.CR / a.pool), allow_pickle=False)
    order = {OL._bare(n): i for i, n in enumerate(z["names"].tolist())}
    o = np.array([order[OL._bare(n)] for n in pn.tolist()])
    rows_name = a.pool.split("_", 1)[-1]
    meta = np.load(data_dir() / "runs/onpol_nuplan/rows" / rows_name / "meta.npz", allow_pickle=False)
    scene = meta["scene"].astype(str)
    assert np.array_equal(meta["names"], pn), "rows meta and pool tab differ"
    ok = z["ok"][o]
    red = z["b_mu"][o].any((1, 2)) & ok
    pi = torch.as_tensor(S.pi, device=dev)

    def hinges(plans):
        out = {"B": [], "P": []}
        for i in range(0, len(plans), 2048):
            r = torch.arange(i, min(i + 2048, len(plans)), device=dev)
            with torch.no_grad():
                t = OL.terms(plans[i:i + 2048].to(dev), r, pack)
            out["B"].append(t["lonB"].cpu().numpy())
            out["P"].append(t["lonP"].cpu().numpy())
        return {k: np.concatenate(v) for k, v in out.items()}

    def plans_of(tag):
        if tag != a.tags[0]:
            ck = torch.load(T.proot("runs", tag) / "ckpt-final.pt", map_location="cpu", weights_only=False)
            model.load_state(ck["model"])
        model.eval()
        res = []
        with torch.no_grad():
            for i in range(0, len(pn), 256):
                r = np.arange(on0 + i, on0 + min(i + 256, len(pn)))
                rt = torch.as_tensor(r, device=dev)
                p = model(S.front[r], S.ego[rt], S.tc[rt], None, None, nv=None if ns.nv is None else ns.nv[rt]).float()[:, pi].view(-1, 33, 15)
                x, y, psi = T.rear(p, S.cam_x[rt], W)
                res.append(torch.stack([x, y, psi], -1).cpu())
        return torch.cat(res)

    def summ(h):
        d = {}
        for it in ("B", "P"):
            act = (h[it] > 0) & ok
            d[it] = dict(active_share=float(act[ok].mean()), active_rows=int(act.sum()), active_scenes=len(set(scene[act].tolist())),
                         mean_when_active=float(h[it][act].mean()) if act.any() else None, mean_all=float(h[it][ok].mean()))
        actr = (h["B"] > 0) & red
        d["B_on_red_rows"] = dict(rows=int(red.sum()), active_share=float(actr[red].mean()) if red.any() else None,
                                  mean_when_active=float(h["B"][actr].mean()) if actr.any() else None, sum=float(h["B"][red].sum()))
        return d
    res = dict(pool=a.pool, rows=int(len(pn)), rows_ok=int(ok.sum()), rows_red_in_window=int(red.sum()), scenes=len(set(scene.tolist())),
               items_in_pack=[str(x) for x in z["items"]], tags={})
    served = torch.as_tensor(meta["poses"].astype(np.float32))
    H = {"served": hinges(served)}
    res["tags"]["served (rollout policy)"] = summ(H["served"])
    for tag in a.tags:
        H[tag] = hinges(plans_of(tag))
        res["tags"][tag] = summ(H[tag])
        print(tag, json.dumps(res["tags"][tag]), flush=True)
    out = _pl.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"lonread-{a.pool}.json").write_text(json.dumps(res, indent=1))
    np.savez(out / f"lonread-{a.pool}.npz", names=pn, ok=ok, red=red, **{f"{k}_{it}": v[it] for k, v in H.items() for it in ("B", "P")})
    print(json.dumps({k: v for k, v in res.items() if k != "tags"}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
