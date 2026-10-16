#!/usr/bin/env python3
"""DESIRE training side of the composed WAIT + desire arm (experiments/wait_cap/plans/2026-10-16-wait-train-prereg.md Amendment C; design
plans/2026-10-16-desire-route-design.md section 2). Box, repo root, op-train python. Helpers in lib/desire_train.py; trainer flags in
experiments/route_mem/scripts/rm_train.py (--desire-rows, --d1-k, --d2-lam); chain in experiments/wait_cap/scripts/wt.sh (wd-*).

  teacher --data navtrain_full.sXof12   (GPU) shipped Cinque (P0, port fp16) on every row of the shard (frames warp, the training Store), the
                                        8 lane-change arms of desire_train (LCL / LCR x AGES) and none, through PModel.forward(desire=);
                                        d = rear-axle x, y (8 poses, 0.5 .. 4 s) with the arm minus none -> runs/desire_cap/train/teacher/<data>.npz.
                                        G-proto (first 256 rows, every arm): the desire= kwarg equals dc.py's policy_feeds wrap bit for bit
  rows                                  (CPU) the D1 row table: navtrain `lc` rows (rf_cfpool geom) of every shard and the merged pool's rows
                                        (on1_L18RBW) where lib/route_desire.py's rule fires on the row's own route memory; navtrain rows also need
                                        the rule's side = the human's side -> runs/desire_cap/train/d1_rows.npz (names, side, src, rule) + d1_rows.json
  read --tag T                          (CPU) the registered desire reads of one checkpoint from dc.py plans (nav: C_fast; pai: P_lca) ->
                                        printed json (wt.py wdgate imports `desire_read`)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(REPO / "lib"), str(HERE)]
import desire_train as DT  # noqa: E402

DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
OUT = DATA / "runs/desire_cap/train"
POOL = "on1_L18RBW"
SHARDS = [f"navtrain_full.s{i}of12" for i in range(12)]          # joint_train.FULL (the trainer's --data)


# ---------------------------------------------------------------- teacher (GPU)
def cmd_teacher(a):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    sys.path[:0] = [str(REPO / p) for p in ("scripts", "experiments/op_adapt_r2/lib", "experiments/op_parity/scripts")]
    import pp_train as T
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    out = OUT / "teacher"
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{a.data}.npz"
    if f.exists():
        print(f, "exists")
        return
    S = T.Store([a.data], dev, need_side=False, frames="warp", host=True)
    model = T.load_pmodel("P0", dev)
    W = torch.as_tensor(T.R2.t_weights(T.T8), device=dev)
    sl = model.net.slices
    pi = torch.arange(sl["plan"].start, sl["plan"].start + 495, device=dev)
    arms = [(s, k) for s in (1, -1) for k in range(len(DT.AGES))]
    assert [int(DT.arm_of(s, k)) for s, k in arms] == list(range(DT.N_ARM))

    def xy(o, r):
        x, y, _ = T.rear(o.float()[:, pi].view(-1, 33, 15), S.cam_x[r], W)
        return torch.stack([x, y], -1)

    # G-proto: PModel.forward(desire=) against dc.py's protocol (policy_feeds wrapped, `_to_copy` replaced), first 256 rows, every arm
    _pf, cur = T.A.policy_feeds, {"x": None}

    def pf(net, ctx, at, traffic=None):
        g = _pf(net, ctx, at, traffic)
        if cur["x"] is not None:
            g["_to_copy"] = cur["x"].to(g["_to_copy"].dtype).expand_as(g["_to_copy"]).contiguous()
        return g
    r = torch.arange(min(256, S.n), device=dev)
    worst = 0.0
    with torch.no_grad():
        for s, k in arms:
            q = DT.queue(torch.full((len(r),), s, device=dev), torch.full((len(r),), k, device=dev), dev)
            o1 = model(S.front[r], S.ego[r], S.tc[r], None, None, nv=None, desire=q)
            x = torch.zeros(1, 1, 33, 8, device=dev)
            x[0, 0, DT.BLK[k], DT.LC[s]] = 1.0
            cur["x"] = x
            T.A.policy_feeds = pf
            o2 = model(S.front[r], S.ego[r], S.tc[r], None, None, nv=None)
            T.A.policy_feeds, cur["x"] = _pf, None
            worst = max(worst, float((o1.float() - o2.float()).abs().max()))
    assert worst == 0.0, f"G-proto: desire= differs from dc.py's protocol by {worst}"
    D = np.zeros((S.n, DT.N_ARM, 8, 2), np.float32)
    with torch.no_grad():
        for i in range(0, S.n, 256):
            r = torch.arange(i, min(i + 256, S.n), device=dev)
            f0 = S.front[r]
            p0 = xy(model(f0, S.ego[r], S.tc[r], None, None, nv=None), r)
            for s, k in arms:
                q = DT.queue(torch.full((len(r),), s, device=dev), torch.full((len(r),), k, device=dev), dev)
                D[i:i + len(r), DT.arm_of(s, k)] = (xy(model(f0, S.ego[r], S.tc[r], None, None, nv=None, desire=q), r) - p0).cpu().numpy()
    np.savez(f, names=S.tab["names"], d=D, g_proto=worst)
    lat4 = D[:, :, 7, 1]
    print(json.dumps(dict(data=a.data, rows=int(S.n), g_proto_max_abs=worst, lcl_y4_mean=float(lat4[:, :4].mean()), lcr_y4_mean=float(-lat4[:, 4:].mean()))))


# ---------------------------------------------------------------- D1 row table (CPU)
def rule_side(F, v):
    """lib/route_desire.py's rule on route-memory features F (n, 200 | 203 | (n, 40, 5)) and speeds v (n,) -> side (n,) int (+1 left, -1 right,
    0 none): intent()'s side, gated at its V_MIN (the D0 line's served value). Nothing is decided here."""
    import route_desire as RD
    side = RD.intent(np.asarray(F, np.float32).reshape(len(F), -1))["side"]
    return np.where(np.asarray(v) >= RD.V_MIN, side, 0).astype(np.int64)


def _rule_rev():
    p = subprocess.run(["git", "log", "-1", "--format=%h", "--", "lib/route_desire.py"], cwd=REPO, capture_output=True, text=True)
    return p.stdout.strip() or "untracked"


def cmd_rows(a):
    cr, ft = DATA / "runs/op_parity/cache", DATA / "runs/route_mem/feat"
    POOL, OUT_ = a.pool, Path(a.out)
    N, S_, SRC = [], [], []
    import route_desire as RD
    rec = dict(rule=_rule_rev(), v_min=RD.V_MIN, pool_name=POOL, nav={}, pool={})
    for d in SHARDS:
        z = np.load(DATA / "runs/route_follow/cf/rows" / f"{d}.npz")
        t = np.load(cr / d / "tab.npz")
        fz = np.load(ft / f"{d}.npz")
        assert np.array_equal(fz["names"], t["names"])
        m = z["cls"] == "lc"
        r = z["row"][m].astype(np.int64)
        assert (t["names"][r] == z["token"][m]).all()
        F = fz["f"][r].astype(np.float32)
        F[~fz["ok"][r]] = 0
        sr = rule_side(F, t["speed"][r])
        fy = t["fut"][r, 7, 1]                                        # the human's rear-axle y at 4 s (left +)
        poly = z["poly"][m]                                           # the stay-lane centreline (row frame, arc -10 .. +90 m every 2 m)
        arc4 = np.hypot(*np.diff(np.concatenate([np.zeros((len(r), 1, 2)), t["fut"][r, :, :2]], 1), axis=1).transpose(2, 0, 1)).sum(1)
        j = np.clip(np.round((arc4 + 10.0) / 2.0).astype(int), 0, poly.shape[1] - 1)
        py = poly[np.arange(len(r)), j, 1]
        sh = np.sign(np.where(np.isfinite(py), fy - py, fy)).astype(np.int64)
        keep = (sr != 0) & (sr == sh)
        rec["nav"][d] = dict(lc=int(m.sum()), lc_ok=int((m & z["ok"]).sum()), fired=int((sr != 0).sum()), agree=int(keep.sum()),
                             disagree=int(((sr != 0) & (sr != sh)).sum()))
        N.append(t["names"][r][keep]), S_.append(sr[keep]), SRC.append(np.full(int(keep.sum()), "nav"))
    t = np.load(cr / POOL / "tab.npz")
    fz = np.load(ft / f"{POOL}.npz")
    assert np.array_equal(fz["names"], t["names"])
    F = fz["f"].astype(np.float32)
    F[~fz["ok"]] = 0
    assert "speed" in t.files, f"{POOL}: no speed in tab"
    sp = rule_side(F, t["speed"])
    k = sp != 0
    pool = t["pool"].astype(str) if "pool" in t.files else np.full(len(k), POOL)
    rec["pool"] = dict(rows=int(len(k)), fired=int(k.sum()), by_prefix={p: int((k & (pool == p)).sum()) for p in sorted(set(pool.tolist()))},
                       left=int((sp > 0).sum()), right=int((sp < 0).sum()))
    N.append(t["names"][k]), S_.append(sp[k]), SRC.append(np.full(int(k.sum()), "pool"))
    OUT_.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_ / "d1_rows.npz", names=np.concatenate(N), side=np.concatenate(S_).astype(np.int8), src=np.concatenate(SRC), rule=rec["rule"])
    rec["nav_total"] = dict(lc=sum(v["lc"] for v in rec["nav"].values()), fired=sum(v["fired"] for v in rec["nav"].values()),
                            agree=sum(v["agree"] for v in rec["nav"].values()))
    (OUT_ / "d1_rows.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


# ---------------------------------------------------------------- desire read (CPU)
def desire_read(tag, root=DATA / "runs/desire_cap"):
    """Amendment C 4's desire reads of one checkpoint (dc.py plans files; dc.py's own response functions): C_fast pooled LC 4 s response,
    LCL / LCR 4 s, direction share (> 0.1 m), at pulse age 0; PAI P_lca route-side 4 s response (reported); P0's from its stored plan."""
    import dc
    Rn, Rp = np.load(root / "rows_nav.npz"), np.load(root / "rows_pai.npz")

    def Q(src, t):
        z = np.load(root / "plans" / src / f"{t}.npz")
        return dict(x=z["x"], y=z["y"], psi=z["psi"], arms=[str(x) for x in z["arms"]])
    out = {}
    for t in ("P0", tag):
        qn, qp = Q("nav", t), Q("pai", t)
        m = Rn["in_C_fast"]
        r4 = dc.lc_pooled(qn)[m, dc.I4]
        mp = Rp["in_P_lca"]
        out[t] = dict(mean4=float(r4.mean()), lcl4=float(dc.resp(qn, "LCL@0")[m, dc.I4].mean()), lcr4=float(dc.resp(qn, "LCR@0")[m, dc.I4].mean()),
                      dir=float((r4 > 0.1).mean()), rows=int(m.sum()), pai_lca4=float(dc.side_resp(qp, Rp["side"], "LC")[mp, dc.I4].mean()),
                      pai_lca_rows=int(mp.sum()))
    P0, J = out["P0"], out[tag]
    g = dict(mean4=J["mean4"] >= 0.9 * P0["mean4"], lcl4=J["lcl4"] >= 0.8, lcr4=J["lcr4"] >= 0.8, dir=J["dir"] >= 0.95)
    return dict(tag=tag, P0=P0, arm=J, gates=g, passed=bool(all(g.values())), pai_lca4_vs_1_5=J["pai_lca4"] >= 1.5)


def cmd_read(a):
    print(json.dumps(desire_read(a.tag), indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("teacher").add_argument("--data", required=True)
    p = sp.add_parser("rows")
    p.add_argument("--pool", default=POOL), p.add_argument("--out", default=str(OUT))
    sp.add_parser("read").add_argument("--tag", required=True)
    a = ap.parse_args()
    dict(teacher=cmd_teacher, rows=cmd_rows, read=cmd_read)[a.cmd](a)
