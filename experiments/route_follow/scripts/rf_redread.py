#!/usr/bin/env python3
"""RED school reads (prereg experiments/route_follow/plans/2026-10-13-red-prereg.md). One GPU job each; $DATA_DIR/envs/op-train/bin/python.
The store is rm_train.build's (on_train.setup + route branch + route features) with one offline shard (navtrain_full.s0of12) and the pool;
pool rows are fed exactly as in training (route input on, 8 slots), checkpoints with the branch loaded as build() loads --rm-from.

  r1   [--pool on1_RED] [--tag RB0-F-s0] [--wrong RSG-L18-s0] --out DIR
       Gate R1 for log-state pools: the pool rows whose source token sits in shard s0 against those source rows in the same store.
       (i) inputs byte-equal (tokens, ego + route memory, teacher; log target on green rows); (a) plans of `tag` on pool rows == plans on the
       source rows within one fp16 ulp per value (same batch composition); controls, which must FAIL: (b) source rows shuffled, (c) the wrong
       checkpoint on the pool rows; share of rows with max |dpose| > 0.03 m. -> DIR/r1.json
  read [--pool on1_REDheld] --tags T1 T2 .. --out DIR [--arm A=T1,T2 --ctrl C=T3,T4 ..]
       Line (a) on the held rows (body1 hold logs, log targets): per tag, the plan's 4 s arc (rear axle, plan_arc), the front bumper's route
       arc (on_lon terms on the pool's lon pack), the B hinge; red rows' stop rate before the line on rows that would reach it at their t0
       speed (v0 x 4 s >= d), green rows' arc; the matched red - green arc gap on moving rows (v0 >= 2 m/s): per (d, v) cell mean(red) -
       mean(green), weighted by the red rows of the cells that have both classes. Group means = seed means. CIs: 2 000 log resamples (seed 0),
       paired. The log's own numbers (tab fut_log) as the first column. -> DIR/redread.json, .md, per-row npz
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
           road_margin=0.3, label_bank=True, yr_mass=0.0, agent_on_only=True, on_k=14, on_mode="both", on_split="", on_lon="", on_lon_lam=10.0, on_road="",
           ego_drop=0.0, ego_drop_speed=0.0, on_weight="", batch=128, steps=1, warmup=1, eval_every=1, seed=0, rm_lr=3e-4, rm_warmup=100, rm_drop=0.25)
N_BOOT, V_MOVE, BS = 2000, 2.0, 256


def store(pool, tag, dev):
    ns_a = SimpleNamespace(**REC, road_labels=RT.OT.JT.ROAD, on=pool, init="", rm_from=tag, tag=tag, exp="route_follow", lr_scale=1.0, prefetch=1, dev_card_gb=4.0)
    ns, _, _, _, _ = RT.build(ns_a, dev)
    return ns


def plans(ns, rows, dev):
    """Rear-axle plans (n, 8, 3) of the loaded model on store rows (batches of BS in the given order)."""
    import torch
    import pp_train as T
    W = ns.losses(ns.mode).W
    pi = torch.as_tensor(ns.S.pi, device=dev)
    ns.model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), BS):
            r = rows[i:i + BS]
            rt = torch.as_tensor(r, device=dev)
            p = ns.model(ns.S.front[r], ns.S.ego[rt], ns.S.tc[rt], None, None, nv=None if ns.nv is None else ns.nv[rt]).float()[:, pi].view(-1, 33, 15)
            x, y, psi = T.rear(p, ns.S.cam_x[rt], W)
            out.append(torch.stack([x, y, psi], -1).cpu())
    return torch.cat(out).numpy()


def load_ckpt(ns, tag):
    import torch
    import pp_train as T
    ck = torch.load(T.proot("runs", tag) / "ckpt-final.pt", map_location="cpu", weights_only=False)
    ns.model.load_state(ck["model"])


def cmd_r1(a):
    import torch
    import ot_rows as OR
    dev = torch.device("cuda")
    ns = store(a.pool, a.tag, dev)
    S, on0 = ns.S, ns.on0
    meta = np.load(OR.CR / a.pool / "red.npz")
    tokens = S.tab["names"][:on0]
    pos = {t: i for i, t in enumerate(tokens.tolist())}
    pi_ = np.array([pos.get(t, -1) for t in meta["token"].tolist()])
    m = np.flatnonzero(pi_ >= 0)
    rp, rs = on0 + m, pi_[m]
    tp, ts = torch.as_tensor(rp, device=dev), torch.as_tensor(rs, device=dev)
    eq = lambda u, v: bool(torch.equal(u, v))                       # noqa: E731
    grn = meta["cls"][m] == "green"
    res = dict(pool=a.pool, tag=a.tag, wrong=a.wrong, rows_checked=int(len(m)), rows_pool=int(len(meta["token"])),
               inputs=dict(front=eq(S.front[rp].float(), S.front[rs].float()), ego_route=eq(S.ego[tp], S.ego[ts]), teacher=eq(S.t_out[tp], S.t_out[ts]),
                           tc=eq(S.tc[tp], S.tc[ts]), cam=eq(S.cam_x[tp], S.cam_x[ts]),
                           fut_green=eq(S.fut[tp[torch.as_tensor(grn, device=dev)]], S.fut[ts[torch.as_tensor(grn, device=dev)]])))
    Pp, Ps = plans(ns, rp, dev), plans(ns, rs, dev)
    ulp = np.spacing(np.abs(Ps).astype(np.float16)).astype(np.float32)
    d = np.abs(Pp - Ps)
    res["a_within_one_fp16_ulp_rows"] = float((d <= ulp).all((1, 2)).mean())
    res["a_bit_equal_rows"] = float((d == 0).all((1, 2)).mean())
    res["a_max_abs_m"] = float(d.max())
    perm = np.random.default_rng(0).permutation(len(rs))
    db = np.abs(Pp - Ps[perm])[..., :2].max((1, 2))
    res["b_shuffled_rows_differ_gt_0p03"] = float((db > 0.03).mean())
    load_ckpt(ns, a.wrong)
    dc = np.abs(plans(ns, rp, dev) - Ps)[..., :2].max((1, 2))
    res["c_wrong_ckpt_rows_differ_gt_0p03"] = float((dc > 0.03).mean())
    res["pass"] = bool(all(res["inputs"].values()) and res["a_within_one_fp16_ulp_rows"] >= 0.999 and res["b_shuffled_rows_differ_gt_0p03"] >= 0.95
                       and res["c_wrong_ckpt_rows_differ_gt_0p03"] >= 0.5)
    out = _pl.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "r1.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    raise SystemExit(0 if res["pass"] else 2)


# ---------------------------------------------------------------- line (a)
def metrics(P, M):
    """P (n, 8, 3) plans, M the held meta -> per-row arrays: arc4, crossed (front past the target line within 4 s), B."""
    import torch
    t = torch.as_tensor(P, dtype=torch.float32)
    T = OL.terms(t, torch.arange(len(P)), M["pack"])
    sf = T["steps"]["sf"].numpy()
    arc4 = OL.plan_arc(t[..., 0], t[..., 1])[:, 7].numpy()
    crossed = sf.max(1) > M["b_d"]
    return dict(arc4=arc4, crossed=crossed, B=T["lonB"].numpy())


def summarise(X, M, idx=None):
    """X per-row metrics, M meta, idx (optional resampled row index) -> the line (a) numbers."""
    i = np.arange(len(M["cls"])) if idx is None else idx
    red, v, d, c = M["cls"][i] == "red", M["v0"][i], M["d"][i], M["cell"][i]
    arc, cr = X["arc4"][i], X["crossed"][i]
    mv = v >= V_MOVE
    num = den = 0.0
    for k in np.unique(c[mv]):
        r, g = mv & red & (c == k), mv & ~red & (c == k)
        if r.any() and g.any():
            num += r.sum() * (arc[r].mean() - arc[g].mean())
            den += r.sum()
    reach = red & (v * 4.0 >= d)
    return dict(gap_moving=num / max(den, 1), stop_rate_reach=float(1 - cr[reach].mean()) if reach.any() else np.nan,
                red_arc_moving=float(arc[mv & red].mean()), green_arc_moving=float(arc[mv & ~red].mean()), green_arc_all=float(arc[~red].mean()),
                red_B_mean=float(X["B"][i][red].mean()), red_cross_all=float(cr[red].mean()))


def cmd_read(a):
    import torch
    import ot_rows as OR
    dev = torch.device("cuda")
    ns = store(a.pool, a.tags[0], dev)
    S, on0 = ns.S, ns.on0
    pn = np.load(OR.CR / a.pool / "tab.npz")["names"]
    tb = np.load(OR.CR / a.pool / "tab.npz")
    z = np.load(OR.CR / a.pool / "lon.npz")
    assert np.array_equal(z["names"], pn) and np.array_equal(S.tab["names"][on0:], pn)
    q = tb["q"].astype(int)
    M = dict(cls=tb["cls"].astype(str), v0=tb["v0"], d=tb["d_line"], cell=tb["cellid"], log=tb["log"].astype(str), pack=OL.load_pack(OR.CR / a.pool / "lon.npz", "cpu"),
             b_d=z["b_d"][np.arange(len(pn)), q])
    rows = on0 + np.arange(len(pn))
    X = {"log": metrics(tb["fut_log"].astype(np.float32), M)}
    for k, tag in enumerate(a.tags):
        if k:
            load_ckpt(ns, tag)
        X[tag] = metrics(plans(ns, rows, dev), M)
        print(tag, json.dumps(summarise(X[tag], M)), flush=True)
    groups = dict(g.split("=", 1) for g in a.groups)
    groups = {k: v.split(",") for k, v in groups.items()}

    def gsum(name, idx=None):
        tags = groups.get(name, [name])
        s = [summarise(X[t], M, idx) for t in tags]
        return {k: float(np.mean([x[k] for x in s])) for k in s[0]}
    names = ["log"] + a.tags + list(groups)
    res = dict(pool=a.pool, rows=int(len(pn)), red=int((M["cls"] == "red").sum()), green=int((M["cls"] == "green").sum()), logs=int(len(set(M["log"].tolist()))),
               red_moving=int(((M["cls"] == "red") & (M["v0"] >= V_MOVE)).sum()), green_moving=int(((M["cls"] == "green") & (M["v0"] >= V_MOVE)).sum()),
               red_reach=int(((M["cls"] == "red") & (M["v0"] * 4 >= M["d"])).sum()), point={n: gsum(n) for n in names}, pairs={})
    ul = np.unique(M["log"])
    by = [np.flatnonzero(M["log"] == u) for u in ul]
    rng = np.random.default_rng(0)
    boots = [np.concatenate([by[j] for j in rng.integers(0, len(ul), len(ul))]) for _ in range(N_BOOT)]
    for pr in a.pairs:
        x, y = pr.split(":")
        B = {k: [] for k in ("gap_moving", "stop_rate_reach", "green_arc_moving", "red_B_mean")}
        for idx in boots:
            sx, sy = gsum(x, idx), gsum(y, idx)
            for k in B:
                B[k].append(sx[k] - sy[k])
        px, py = res["point"][x], res["point"][y]
        res["pairs"][pr] = {k: dict(diff=px[k] - py[k], lo=float(np.nanquantile(v, 0.025)), hi=float(np.nanquantile(v, 0.975))) for k, v in B.items()}
    out = _pl.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "redread.json").write_text(json.dumps(res, indent=1))
    np.savez(out / "redread.npz", names=pn, **{f"{t}_{k}": v for t, x in X.items() for k, v in x.items()})
    L = [f"# RED line (a): held nuPlan red / green rows ({a.pool}), {res['rows']} rows ({res['red']} red, {res['green']} green, {res['logs']} logs); "
         f"moving red / green {res['red_moving']} / {res['green_moving']}; red rows that would reach the line at v0 {res['red_reach']}", "",
         "| plans of | matched red - green 4 s arc, moving (m) | stop rate before the line (reach rows) | red arc moving | green arc moving | green arc all | red B mean | red rows crossing |",
         "|:--|--:|--:|--:|--:|--:|--:|--:|"]
    for n, s in res["point"].items():
        L.append(f"| {n} | {s['gap_moving']:+.2f} | {s['stop_rate_reach']:.3f} | {s['red_arc_moving']:.2f} | {s['green_arc_moving']:.2f} | {s['green_arc_all']:.2f} | "
                 f"{s['red_B_mean']:.4f} | {s['red_cross_all']:.3f} |")
    L += ["", "| a - b | gap (m) | stop rate | green arc moving (m) | red B |", "|:--|:--|:--|:--|:--|"]
    for pr, s in res["pairs"].items():
        L.append(f"| {pr} | " + " | ".join(f"{s[k]['diff']:+.3f} [{s[k]['lo']:+.3f}, {s[k]['hi']:+.3f}]"
                                           for k in ("gap_moving", "stop_rate_reach", "green_arc_moving", "red_B_mean")) + " |")
    (out / "redread.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("r1")
    p.add_argument("--pool", default="on1_RED"), p.add_argument("--tag", default="RB0-F-s0"), p.add_argument("--wrong", default="RSG-L18-s0")
    p.add_argument("--out", required=True)
    p = sub.add_parser("read")
    p.add_argument("--pool", default="on1_REDheld"), p.add_argument("--tags", nargs="+", required=True), p.add_argument("--out", required=True)
    p.add_argument("--groups", nargs="*", default=[], help="NAME=T1,T2 seed means"), p.add_argument("--pairs", nargs="*", default=[], help="A:B (groups or tags)")
    a = ap.parse_args()
    dict(r1=cmd_r1, read=cmd_read)[a.cmd](a)
