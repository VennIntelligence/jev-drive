#!/usr/bin/env python3
"""ONPOL-LONG Amendment 11: ego-status shortcut reliance (open loop) and its relation to closed-loop drift.

  rel   --tags T .. --out F.json      (op-train env, one card) every checkpoint on the dev rows of the Y recipe (normal dev rows + yaw-rate dev rows,
        on_train.setup with --yr-mass 0.1): ADE / FDE of the 8 plan poses to the log with the ego status intact and zeroed
        (parity_adapter.EGO_STATE: vx, vy, ax, ay, history poses; `present` and the command kept), and the plan shift between the two passes
        (mean over the 8 poses; longitudinal |dx| and lateral |dy| separately). Reliance = ADE(zeroed) - ADE(intact).
  drift --reports DIR .. --rel F.json --out DIR   (AlpaSim venv) per checkpoint the drift slope k >= 4 (as ol_report.py) from the reports'
        per_scene_k.parquet on their common scenes, log-clustered 95 % CI of each checkpoint's slope and of slope differences against
        YS-L18 (seed mean), and the Spearman correlation of reliance with drift slope across checkpoints -> ego_drift.md / .json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[3]


def cmd_rel(a):
    import torch
    sys.path[:0] = [str(R / "experiments/onpol_nuplan/scripts")]
    import on_train as OT
    import parity_adapter as PA
    import pp_train as T
    dev = torch.device("cuda")
    p = argparse.ArgumentParser()
    OT.recipe_args(p)
    p.set_defaults(steps=1, batch=128, warmup=1, eval_every=1, on_k=13, on_mode="", on_split="", on_lon="", on_lon_lam=10.0, lr_scale=1.0)
    ns = OT.setup(p.parse_args(["--yr-mass", "0.1", "--label-bank"]), dev)
    S, model = ns.S, ns.model.eval()
    Sd = T.dev_store(S, ns.dv_rows, 4.0)
    W = torch.as_tensor(T.R2.t_weights(T.T8), device=dev)
    pi = torch.as_tensor(S.pi, device=dev)

    def ev(St, rows, bs=128):
        acc = {k: [] for k in ("ade", "ade0", "fde", "fde0", "shift", "shift_lon", "shift_lat")}
        with torch.no_grad():
            for i in range(0, len(rows), bs):
                r = torch.as_tensor(rows[i:i + bs], device=dev)
                xy = []
                for zero in (False, True):
                    e = St.ego[r].clone()
                    if zero:
                        e[:, PA.EGO_STATE] = 0
                    p_ = model(St.front[r], e, St.tc[r], None, None, nv=None if St.nv is None else St.nv[r]).float()[:, pi].view(-1, 33, 15)
                    x, y, _ = T.rear(p_, St.cam_x[r], W)
                    xy.append((x, y))
                    ok = St.has_fut[r]
                    d = torch.hypot(x - St.fut[r][..., 0], y - St.fut[r][..., 1])[ok]
                    acc["ade0" if zero else "ade"].append(d.mean(1))
                    acc["fde0" if zero else "fde"].append(d[:, -1])
                (x1, y1), (x0, y0) = xy
                acc["shift"].append(torch.hypot(x1 - x0, y1 - y0).mean(1))
                acc["shift_lon"].append((x1 - x0).abs().mean(1))
                acc["shift_lat"].append((y1 - y0).abs().mean(1))
        m = {k: float(torch.cat(v).mean()) for k, v in acc.items()}
        return m | {"reliance": m["ade0"] - m["ade"], "reliance_fde": m["fde0"] - m["fde"], "rows": int(len(rows))}
    res = {}
    for t in a.tags:
        ck = torch.load(T.proot("runs", t) / "ckpt-final.pt", map_location="cpu", weights_only=False)["model"]
        assert ck["arm"] == model.arm, (t, ck["arm"])
        model.load_state(ck)                                         # also sets the checkpoint's TR1 keys (SEX: noego), {} otherwise
        res[t] = {"tr1": ck.get("tr1") or {}, "dev": ev(Sd, ns.dv_o), "dev_yr": ev(S, ns.dv_t)}
        print(t, json.dumps(res[t]), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")


def cmd_drift(a):
    import pandas as pd
    K = pd.concat([pd.read_parquet(Path(d) / "per_scene_k.parquet") for d in a.reports]).drop_duplicates(["label", "scene", "k"])
    S = pd.concat([pd.read_csv(Path(d) / "per_scene.csv") for d in a.reports]).drop_duplicates(["label", "scene"])
    K["tag"], S["tag"] = K.label.str.split("@").str[0], S.label.str.split("@").str[0]
    tags = [t for t in a.tags if t in set(K.tag)]
    ok = S[S.offroad_or_collision_at_fault.notna()]
    sc = sorted(set.intersection(*[set(ok[ok.tag == t].scene) for t in tags]))
    K = K[K.scene.isin(sc) & (K.k >= 4) & K.tag.isin(tags)]
    M = K.pivot_table(index=["scene", "k"], columns="tag", values="dist_gt")       # per scene and decision; nan once a rollout ended
    logs = pd.Series(sc, index=sc).str.rsplit("-", n=1).str[0]
    L = np.array(sorted(logs.unique()))
    li = np.searchsorted(L, logs.reindex(M.index.get_level_values(0)).to_numpy())
    kv = M.index.get_level_values(1).to_numpy(float)
    ks = np.unique(kv)
    draws = np.random.default_rng(0).integers(0, len(L), (a.boot, len(L)))

    def slopes(col, w):                                           # w (B, logs) cluster weights -> (B,) OLS slope of the per-k means over k
        v = col.to_numpy()
        f = np.isfinite(v)
        out = []
        for ww in w:
            wr = ww[li] * f
            mk = np.array([(wr[kv == k] * np.nan_to_num(v[kv == k])).sum() / max(wr[kv == k].sum(), 1e-9) for k in ks])
            x = ks - ks.mean()
            out.append(((mk - mk.mean()) * x).sum() / (x * x).sum())
        return np.array(out)
    one = np.ones((1, len(L)))
    Wb = np.stack([np.bincount(d, minlength=len(L)) for d in draws])
    sl, bs = {}, {}
    for t in tags:
        sl[t], bs[t] = float(slopes(M[t], one)[0]), slopes(M[t], Wb)
    grp = {g: [t for t in v.split(",") if t in sl] for g, v in (x.split("=") for x in a.groups)}
    for g, v in grp.items():
        if v:
            sl[g], bs[g] = float(np.mean([sl[t] for t in v])), np.mean([bs[t] for t in v], 0)
    rel = json.loads(Path(a.rel).read_text()) if a.rel else {}
    md = [f"# ONPOL-LONG Amendment 11: drift slope (k >= 4, m / decision) and ego-status reliance; {len(sc)} common scenes, {len(L)} logs", "",
          "| checkpoint | drift slope [log-clustered 95 % CI] | dev ADE intact / zeroed (m) | reliance (m) | plan shift total / lon / lat (m) | yaw-rate dev reliance (m) |",
          "|:--|:--|:--|--:|:--|--:|"]
    for t in list(tags) + [g for g in grp if g in sl]:
        r = rel.get(t, {}).get("dev")
        ry = rel.get(t, {}).get("dev_yr")
        md.append(f"| {t} | {sl[t]:.4f} [{np.percentile(bs[t], 2.5):.4f}, {np.percentile(bs[t], 97.5):.4f}] | "
                  + (f"{r['ade']:.3f} / {r['ade0']:.3f} | {r['reliance']:+.3f} | {r['shift']:.3f} / {r['shift_lon']:.3f} / {r['shift_lat']:.3f} | {ry['reliance']:+.3f} |"
                     if r else "- | - | - | - |"))
    res = {"scenes": len(sc), "logs": len(L), "slope": sl, "slope_ci": {t: [float(np.percentile(bs[t], 2.5)), float(np.percentile(bs[t], 97.5))] for t in sl}}
    md += ["", "| a - b | drift slope difference [log-clustered 95 % CI] |", "|:--|:--|"]
    res["pairs"] = {}
    for q in a.pairs:
        x, y = q.split(":")
        if x in sl and y in sl:
            d = bs[x] - bs[y]
            res["pairs"][q] = [sl[x] - sl[y], float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
            md.append(f"| {x} - {y} | {sl[x] - sl[y]:+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}] |")
    pts = [(rel[t]["dev"]["reliance"], sl[t]) for t in tags if t in rel]
    if len(pts) >= 3:
        from scipy.stats import spearmanr
        rho = spearmanr(*zip(*pts))
        res["spearman"] = [float(rho.statistic), float(rho.pvalue), len(pts)]
        md += ["", f"Spearman (dev reliance, drift slope) over {len(pts)} checkpoints: {rho.statistic:+.3f} (p {rho.pvalue:.3f})."]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ego_drift.md").write_text("\n".join(md) + "\n")
    (out / "ego_drift.json").write_text(json.dumps(res, indent=1) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("rel")
    p.add_argument("--tags", nargs="+", required=True), p.add_argument("--out", required=True)
    p = sub.add_parser("drift")
    p.add_argument("--reports", nargs="+", required=True), p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--groups", nargs="*", default=[], help="NAME=TAG,TAG seed means"), p.add_argument("--pairs", nargs="*", default=[], help="A:B")
    p.add_argument("--rel", default=""), p.add_argument("--out", required=True), p.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    {"rel": cmd_rel, "drift": cmd_drift}[a.cmd](a)
