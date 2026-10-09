"""Off-track recovery probe on the adaptation ladder (lane OT2, piece A; decision 209): does the shipped openpilot plan already return to
the path after a lateral / yaw offset, and did open-loop adaptation remove that?

Rows: the held-out (<split>-dev) tokens of the off-track caches written by ot_rows.py prep (real navtrain frames re-projected to a statically
perturbed pose, decision 198), paired with the same tokens at the logged pose. Per model the plan is read at both and the lateral shift between
the two is regressed on the shift a full return to the logged path asks for, split into its lateral-offset and its yaw part (ot_rows.cmd_probe's
definition, unchanged): 1 = the plan returns to the logged path by that time, 0 = the offset is ignored, < 0 = the plan moves further away.
What this adds to ot_rows.py probe: all 12 shards, the whole ladder in one run, log-cluster bootstrap CIs and paired differences, horizons past
4 s (plan points to 10 s; only the shipped model was trained there), a split by |dy|, and AP2 under its own input standard.

Inputs per model (the frame each is normally read in):
  P0            shipped Cinque through the parity port: frozen vision tokens of the W frames -> its own policy -> plan; no ego input, no adapter
  parity tags   the same tokens + the adapter's ego features of the tab (NAVSIM standard: logged velocity / acceleration / command, history
                poses re-expressed in the perturbed frame)
  AP2 tags      the same tokens (m = 4 keyframes) + ego features under the AlpaSim standard (ap2_core.ego_table: vy = 0, ay = 0, command from
                the rebuilt AlpaSim route re-expressed in the perturbed frame); `<tag>@nav` reads it with the NAVSIM-standard features instead
All models see identical image tokens; the plan is exported to the rear axle the same way (pp_train.rear).

  $DATA_DIR/envs/op-train/bin/python experiments/op_parity/scripts/ot_ladder.py --name ladder --tags P0 P2-F-s0 SH30-F-s0 AP2-AB-s0 OT30-F-s0
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_R / "experiments/op_parity/scripts"),
                 str(_R / "experiments/alpasim/lib")]
import argparse, json  # noqa: E401,E402

import numpy as np  # noqa: E402

import ot_rows as OR  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

TL = np.array([1.0, 2.0, 4.0, 6.0, 8.0, 10.0])          # plan horizons read (s); the adapters are trained to 4 s
AROOT = data_dir() / "runs/alpasim/ap2"


def boot_slopes(X, y, logs, B=10000, seed=0):
    """Least squares y ~ X (n, 2) without intercept -> coefficient (2,) and its B cluster-bootstrap replicates (B, 2), clusters = logs."""
    u, inv = np.unique(logs, return_inverse=True)
    G, g = np.zeros((len(u), 2, 2)), np.zeros((len(u), 2))
    np.add.at(G, inv, X[:, :, None] * X[:, None, :])
    np.add.at(g, inv, X * y[:, None])
    idx = np.random.default_rng(seed).integers(0, len(u), (B, len(u)))
    return np.linalg.solve(G.sum(0), g.sum(0)), np.linalg.solve(G[idx].sum(1), g[idx].sum(1)[..., None])[..., 0]


def ci(c, bs):
    lo, hi = np.percentile(bs, [2.5, 97.5], 0)
    return [float(c), float(lo), float(hi)]


def wp_to_frame(wp, dy, dpsi):
    """Route waypoints (n, 20, 2) in the logged rig frame -> the perturbed rig frame (0, dy, dpsi); NaN padding stays NaN."""
    c, s = np.cos(dpsi)[:, None], np.sin(dpsi)[:, None]
    x, y = wp[..., 0], wp[..., 1] - dy[:, None]
    return np.stack([c * x + s * y, -s * x + c * y], -1)


def main(a):
    import torch
    import ap2_core as AC
    import pp_train as T
    from jevdrive.data import splits
    from jevdrive.run import Run
    OR.OT = a.ot
    dev = torch.device("cuda")
    with Run("op_parity", f"otladder-{a.name}", config=vars(a)) as run:
        ot = tuple(f"{a.ot}_{d}" for d in a.data)
        S = T.Store(ot, dev, need_side=False, frames="warp", host=True)
        Bs = T.Store(tuple(a.data), dev, need_side=False, frames="warp", host=True)
        nb = np.cumsum([0] + [len(np.load(OR.CR / d / "tab.npz")["names"]) for d in a.data])
        src = np.concatenate([np.load(OR.CR / d / "tab.npz")["src_row"] + o for d, o in zip(ot, nb)])
        assert (Bs.tab["names"][src] == S.tab["names"]).all()
        off = OR.offsets(ot).astype(np.float64)
        sp = splits.load(f"{a.split}-dev")
        run.use_split(sp)
        rz = [np.load(AROOT / "route" / f"{d}.npz") for d in a.data]
        assert np.concatenate([z["names"] for z in rz]).tolist() == Bs.tab["names"].tolist()
        wp4, ok4 = np.concatenate([z["wp"][:, 3] for z in rz]), np.concatenate([z["ok"][:, 3] for z in rz])
        rows = np.flatnonzero(sp.mask(S.tab["names"]) & ok4[src])
        rows = rows[: a.limit] if a.limit else rows
        sb = src[rows]
        log = S.tb["log"][rows]
        run.info(f"{len(rows)} held-out off-track rows of {len(a.data)} shards, {len(np.unique(log))} logs; |dy| mean {np.abs(off[rows, 0]).mean():.3f} m, "
                 f"|dpsi| mean {np.degrees(np.abs(off[rows, 1])).mean():.2f} deg")
        sub = lambda tab, r: {k: tab[k][r] for k in ("pose", "vel", "acc", "fut")}  # noqa: E731
        ego_ap = {"ot": AC.ego_table(sub(S.tb, rows), np.repeat(wp_to_frame(wp4[sb], off[rows, 0], off[rows, 1])[:, None], 4, 1))[:, 3],
                  "base": AC.ego_table(sub(Bs.tb, sb), np.repeat(wp4[sb][:, None], 4, 1))[:, 3]}
        cmd_same = float((ego_ap["ot"][:, :4].argmax(1) == ego_ap["base"][:, :4].argmax(1)).mean())
        W8 = torch.as_tensor(T.R2.t_weights(T.T8), device=dev)
        WL = torch.as_tensor(T.R2.t_weights(TL), device=dev)
        pi = torch.as_tensor(S.pi, device=dev)

        def plans(model, St, rr, ego):
            o8, oL = [], []
            with torch.no_grad():
                for i in range(0, len(rr), 256):
                    r = torch.as_tensor(rr[i:i + 256], device=dev)
                    e = St.ego[r] if ego is None else torch.from_numpy(ego[i:i + 256]).to(dev)
                    p = model(St.front[r], e, St.tc[r]).float()[:, pi].view(-1, 33, 15)
                    o8.append(torch.stack(T.rear(p, St.cam_x[r], W8), -1).cpu().numpy())
                    oL.append(torch.stack(T.rear(p, St.cam_x[r], WL), -1).cpu().numpy())
            return np.concatenate(o8).astype(np.float64), np.concatenate(oL).astype(np.float64)
        fut_o, fut_b = S.tb["fut"][rows].astype(np.float64), Bs.tb["fut"][sb].astype(np.float64)
        dy_, dp_ = off[rows, 0:1], off[rows, 1:2]
        Xd, Xp = -dy_ * np.cos(dp_) * np.ones_like(fut_b[..., 0]), -np.sin(dp_) * fut_b[..., 0] + (np.cos(dp_) - 1) * fut_b[..., 1]
        big = np.abs(off[rows, 0]) >= 0.5 * np.abs(off[rows, 0]).max()
        res, reps = {}, {}
        for spec in a.tags:
            tag, _, std = spec.partition("@")
            if tag == "P0":
                model, ap = T.load_pmodel(tag, dev), False
            else:
                model, route, _ = AC.load_model(tag, dev)
                ck = torch.load(T.proot("runs", tag) / "ckpt-final.pt", map_location="cpu", weights_only=False)
                ap = (ck.get("ap2") or {}).get("std", "navsim") == "alpasim" and std != "nav"
                assert not route
                tsplit = ck.get("cfg", {}).get("split", "?")
                if tsplit != a.split:
                    run.info(f"WARNING {tag}: trained with split {tsplit}, probe rows are {a.split}-dev")
            (po, poL), (pb, pbL) = plans(model, S, rows, ego_ap["ot"] if ap else None), plans(model, Bs, sb, ego_ap["base"] if ap else None)
            r = dict(n=len(rows), logs=int(len(np.unique(log))), inputs="alpasim" if ap else ("none" if tag == "P0" else "navsim"),
                     ade_ot=float(np.linalg.norm(po[..., :2] - fut_o[..., :2], axis=-1).mean()),
                     ade_base=float(np.linalg.norm(pb[..., :2] - fut_b[..., :2], axis=-1).mean()),
                     lat4_ot=float(np.abs(po[:, 7, 1] - fut_o[:, 7, 1]).mean()), lat4_base=float(np.abs(pb[:, 7, 1] - fut_b[:, 7, 1]).mean()))
            rep = {}
            for t, j in (("1s", 1), ("2s", 3), ("4s", 7)):                       # ot_rows.cmd_probe's definition (the logged future gives the yaw part)
                c, bs = boot_slopes(np.c_[Xd[:, j], Xp[:, j]], po[:, j, 1] - pb[:, j, 1], log)
                r[f"dy_{t}"], r[f"yaw_{t}"] = ci(c[0], bs[:, 0]), ci(c[1], bs[:, 1])
                rep[f"dy_{t}"], rep[f"yaw_{t}"] = bs[:, 0], bs[:, 1]
            for k, t in enumerate(TL):                                           # all horizons, the yaw part from the model's own plan at the logged pose
                X = np.c_[-dy_[:, 0] * np.cos(dp_[:, 0]), -np.sin(dp_[:, 0]) * pbL[:, k, 0] + (np.cos(dp_[:, 0]) - 1) * pbL[:, k, 1]]
                c, bs = boot_slopes(X, poL[:, k, 1] - pbL[:, k, 1], log)
                r[f"L_dy_{t:g}s"], r[f"L_yaw_{t:g}s"] = ci(c[0], bs[:, 0]), ci(c[1], bs[:, 1])
            for nm, msk in (("small", ~big), ("large", big)):                    # dead zone: does the response depend on the size of the offset
                c, bs = boot_slopes(np.c_[Xd[msk, 7], Xp[msk, 7]], (po[:, 7, 1] - pb[:, 7, 1])[msk], log[msk])
                r[f"dy_4s_{nm}"] = ci(c[0], bs[:, 0])
            res[spec], reps[spec] = r, rep
            run.info(f"{spec}: dy 1/2/4 s {r['dy_1s'][0]:.3f} / {r['dy_2s'][0]:.3f} / {r['dy_4s'][0]:.3f}, yaw {r['yaw_1s'][0]:.3f} / {r['yaw_2s'][0]:.3f} / "
                     f"{r['yaw_4s'][0]:.3f}; ADE off-track {r['ade_ot']:.3f} logged {r['ade_base']:.3f}")
            del model
        diffs = {}
        for pair in a.pairs:                                                     # paired differences A - B on the same bootstrap draws
            x, y = pair.split(":")
            diffs[pair] = {k: ci(res[x][k][0] - res[y][k][0], reps[x][k] - reps[y][k]) for k in reps[x]}
        f = lambda v: f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]"  # noqa: E731
        L = [f"{len(rows)} held-out off-track rows ({a.split}-dev, cache `{a.ot}`, {len(a.data)} shards, {len(np.unique(log))} logs); |dy| mean "
             f"{np.abs(off[rows, 0]).mean():.2f} m (max {np.abs(off[rows, 0]).max():.2f}), |dpsi| mean {np.degrees(np.abs(off[rows, 1])).mean():.2f} deg; "
             f"95% CIs: cluster bootstrap over logs, B 10000. AlpaSim route command equal at both poses: {cmd_same:.3f}.", "",
             "| model | inputs | response to the lateral offset 1 s | 2 s | 4 s | response to the yaw offset 1 s | 2 s | 4 s | ADE off-track / logged pose (m) |",
             "|:--|:--|:--|:--|:--|:--|:--|:--|:--|"]
        L += [f"| {t} | {r['inputs']} | {f(r['dy_1s'])} | {f(r['dy_2s'])} | {f(r['dy_4s'])} | {f(r['yaw_1s'])} | {f(r['yaw_2s'])} | {f(r['yaw_4s'])} | "
              f"{r['ade_ot']:.3f} / {r['ade_base']:.3f} |" for t, r in res.items()]
        L += ["", "Horizons past 4 s (yaw part defined on the model's own plan; adapters are trained to 4 s only) and the 4 s lateral response by |dy| half:", "",
              "| model | lateral 4 s | 6 s | 8 s | 10 s | yaw 4 s | 6 s | 8 s | 10 s | lateral 4 s, small abs(dy) | large abs(dy) |", "|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|"]
        L += [f"| {t} | " + " | ".join(f(r[f"L_{q}_{h}s"]) for q in ("dy", "yaw") for h in (4, 6, 8, 10)) + f" | {f(r['dy_4s_small'])} | {f(r['dy_4s_large'])} |"
              for t, r in res.items()]
        if diffs:
            L += ["", "| paired difference | lateral 1 s | 2 s | 4 s | yaw 1 s | 2 s | 4 s |", "|:--|:--|:--|:--|:--|:--|:--|"]
            L += [f"| {p.replace(':', ' - ')} | " + " | ".join(f(d[f"{q}_{t}"]) for q in ("dy", "yaw") for t in ("1s", "2s", "4s")) + " |" for p, d in diffs.items()]
        OR.OUT.mkdir(parents=True, exist_ok=True)
        (OR.OUT / f"ladder_{a.name}.json").write_text(json.dumps(dict(models=res, diffs=diffs, n=len(rows), cmd_same=cmd_same), indent=1))
        (OR.OUT / f"ladder_{a.name}.md").write_text("\n".join(L) + "\n")
        run.info("\n" + "\n".join(L))
        run.summary |= {"n": len(rows), "out": str(OR.OUT / f"ladder_{a.name}.md")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--tags", nargs="+", required=True, help="P0 | a run tag | <AP2 tag>@nav (AP2 read with NAVSIM-standard ego features)")
    ap.add_argument("--pairs", nargs="*", default=[], help="A:B paired differences of the slopes")
    ap.add_argument("--data", nargs="+", default=[f"navtrain_full.s{i}of12" for i in range(12)])
    ap.add_argument("--ot", default="ot1", help="off-track cache prefix (ot1 = +-0.5 m / 2 deg)")
    ap.add_argument("--split", default="navsim/op-parity-full")
    ap.add_argument("--limit", type=int, default=0)
    main(ap.parse_args())
