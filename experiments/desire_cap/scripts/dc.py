#!/usr/bin/env python3
"""DESIRE-CAP (plans/2026-10-16-desire-cap-prereg.md): does the shipped Cinque (P0) have a desire pathway (laneChangeLeft / Right moves the
plan a lane, turnLeft / Right turns it) on our training input protocol, and which stage of our lineage keeps, weakens or kills it? Open loop,
no training: the same frame tokens, ego row and route memory, with openpilot's desire pulse placed in the policy's desire queue.

  rows    (box, op-train python, CPU)  dc.py rows            the registered row sets: NAVSIM capledger_test cells (C_fast, C_slow, T_L / T_R,
                                       LC_L / LC_R) and PAI on-policy rows of on1_PAIonp (P_keep, P_lca, P_turn) -> runs/desire_cap/rows_<src>.npz
  plans   (box, op-train python, GPU)  dc.py plans --tags ..  per checkpoint and desire arm the rear-axle plan at 1 / 2 / 4 / 6 / 8 s and the
                                       desire_state head -> runs/desire_cap/plans/<src>/<tag>.npz (gates G-none, G-rep in the run log)
  report  (Mac or box, CPU)            dc.py report --root R --out RESULTS   responses, ratios vs P0, the registered lines

Desire queue (read from the ONNX): `desire` (8,) -> state_desire_q (132, 1, 8) at 20 Hz -> reshape (33, 4, 8), max over 4 -> the policy
input `_to_copy` (1, 33, 8), block 32 = the newest 0.2 s. modeld feeds only the rising edge, so a pulse of age D s sits in block
32 - round(D / 0.2). The arm sets that one entry; jevdrive.op_adapt.policy_feeds (zeros) is wrapped, nothing shared is edited.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO)]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
OUT = DATA / "runs/desire_cap"
SRC = {"nav": "idx_capledger_test", "pai": "on1_PAIonp"}
TS = (1.0, 2.0, 4.0, 6.0, 8.0)                    # plan read times (s)
I4 = TS.index(4.0)
DES = {"TL": 1, "TR": 2, "LCL": 3, "LCR": 4}       # log.Desire
SIDE = {"TL": 1, "TR": -1, "LCL": 1, "LCR": -1}
AGES = (0.0, 0.4, 1.0, 2.0)
ARMS = [("none", 0, None)] + [(f"{k}@{a:g}", d, 32 - int(round(a / 0.2))) for k, d in DES.items() for a in AGES]
RM_ARMS = [("none", 0, None)] + [(f"{k}@0", d, 32) for k, d in DES.items()]
V_LC = 8.9                                         # DesireHelper LANE_CHANGE_SPEED_MIN (20 mph)
OFFS = -6.0 + 2.0 * np.arange(40)                  # rmem slots
TAGS = ["P0"] + [f"{s}-s{k}" for s in ("P2-F", "P2H10-F", "YR10m10-F", "YS-L18", "RB0-F", "RSG-L18", "RBSG2-L18") for k in (0, 1)]
STAGES = [("P0", None), ("P2-F", "P0"), ("P2H10-F", "P2-F"), ("YR10m10-F", "P2H10-F"), ("YS-L18", "YR10m10-F"),
          ("RB0-F", "P0"), ("RSG-L18", "RB0-F"), ("RBSG2-L18", "RB0-F")]


# ---------------------------------------------------------------- rows (CPU)
def cmd_rows(a):
    cr = DATA / "runs/op_parity/cache"
    out = OUT
    out.mkdir(parents=True, exist_ok=True)
    # NAVSIM: capledger_test cells (experiments/cap_ledger), cache order
    names = np.load(cr / SRC["nav"] / "tab.npz")["names"]
    z = np.load(DATA / "runs/cap_ledger/select/rows.npz")
    pos = {t: i for i, t in enumerate(z["token"].tolist())}
    k = np.array([pos[n] for n in names.tolist()])
    g = lambda c: z[c][k]  # noqa: E731
    v0 = g("v0")
    cells = {"C_fast": g("in_C") & (v0 >= V_LC), "C_slow": g("in_C") & (v0 < V_LC), "T_L": g("in_T_L"), "T_R": g("in_T_R"),
             "LC_L": g("in_LC_L"), "LC_R": g("in_LC_R")}
    side = np.where(cells["T_L"] | cells["LC_L"], 1, np.where(cells["T_R"] | cells["LC_R"], -1, 0))
    keep = np.logical_or.reduce(list(cells.values()))
    sel = np.flatnonzero(keep)
    np.savez(out / "rows_nav.npz", idx=sel, names=names[sel], log=g("log")[sel], v0=v0[sel], side=side[sel], off4=g("off4")[sel],
             lane_c=g("lane_c")[sel], lane_ok=g("lane_ok")[sel], **{f"in_{c}": m[sel] for c, m in cells.items()})
    # PAI on-policy rows: cells from the fed route memory (prereg section 3)
    t = np.load(cr / SRC["pai"] / "tab.npz")
    f = np.load(DATA / "runs/route_mem/feat" / f"{SRC['pai']}.npz")
    assert np.array_equal(f["names"], t["names"]), "route features rows != cache rows"
    Fm = f["f"].reshape(-1, 40, 5)
    i = lambda s: int(np.flatnonzero(OFFS == s)[0])  # noqa: E731
    val, y = Fm[..., 0] > 0.5, Fm[..., 2] * 20.0
    hd = np.degrees(np.arctan2(Fm[..., 4], Fm[..., 3]))
    allv = val[:, i(0):i(60) + 1].all(1)
    l0, l40, l60 = y[:, i(0)], y[:, i(40)], y[:, i(60)]
    hmax, hfar = np.abs(hd[:, i(0):i(60) + 1]).max(1), np.abs(hd[:, i(30):i(60) + 1]).max(1)
    sp, st = t["speed"], t["turn"] == "straight"
    pk = allv & (np.abs(l40) < 0.5) & (np.abs(l60) < 0.7) & (hmax < 5) & (sp >= 5) & st
    pl = allv & (np.abs(l40) >= 2.0) & (np.abs(l40) <= 5.5) & (np.abs(l60 - l40) <= 1.0) & (hfar < 10) & (hmax < 25) & (np.abs(l0) < 1.0) & (sp >= 5) & st
    pt = (t["turn"] == "approach") & np.isin(t["side"], ["left", "right"])
    side = np.where(pl, np.sign(l40), 0) + np.where(pt, np.where(t["side"] == "left", 1, -1), 0)
    sel = np.flatnonzero(pk | pl | pt)
    np.savez(out / "rows_pai.npz", idx=sel, names=t["names"][sel], log=t["log"][sel], v0=sp[sel], side=side[sel].astype(int),
             in_P_keep=pk[sel], in_P_lca=pl[sel], in_P_turn=pt[sel])
    for s in ("nav", "pai"):
        r = np.load(out / f"rows_{s}.npz")
        print(s, len(r["idx"]), {c[3:]: (int(r[c].sum()), len(set(r["log"][r[c]].tolist()))) for c in r.files if c.startswith("in_")})


# ---------------------------------------------------------------- plans (GPU)
def cmd_plans(a):
    import torch
    sys.path[:0] = [str(REPO / p) for p in ("lib", "scripts", "experiments/op_adapt_r2/lib", "experiments/op_parity/scripts", "experiments/route_mem/lib",
                                             "experiments/route_mem/scripts")]
    import pp_train as T
    import rmem as RM
    from jevdrive.run import Run
    from jevdrive.data import splits
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    _pf = T.A.policy_feeds
    cur = {"x": None}

    def pf(net, ctx, at, traffic=None):                  # the desire queue of the arm; None = the shipped zeros (call unchanged)
        f = _pf(net, ctx, at, traffic)
        if cur["x"] is not None:
            f["_to_copy"] = cur["x"].to(f["_to_copy"].dtype).expand_as(f["_to_copy"]).contiguous()
        return f
    T.A.policy_feeds = pf

    def queue(d, blk):
        if d == 0:
            return None
        x = torch.zeros(1, 1, 33, 8, device=dev)
        x[0, 0, blk, d] = 1.0
        return x

    W = torch.as_tensor(T.R2.t_weights(np.array(TS)), device=dev)
    with Run("desire_cap", "plans", config=vars(a)) as run:
        for src in a.srcs:
            data = SRC[src]
            if src == "nav":
                run.use_split(splits.load("navsim/capledger_test"))
            R = np.load(OUT / f"rows_{src}.npz")
            S = T.Store([data], dev, need_side=False, frames="warp")
            assert np.array_equal(S.tab["names"][R["idx"]], R["names"]), f"{src}: rows file vs cache"
            z = np.load(DATA / "runs/route_mem/feat" / f"{data}.npz")
            assert np.array_equal(z["names"], S.tab["names"]), "route features rows != cache rows"
            Fr = z["f"].astype(np.float32)
            Fr[~z["ok"]] = 0
            Fr = torch.as_tensor(Fr, device=dev)
            idx = torch.as_tensor(R["idx"], device=dev)
            od = OUT / "plans" / src
            od.mkdir(parents=True, exist_ok=True)
            for tag in a.tags:
                f = od / f"{tag}.npz"
                if f.exists() and not a.force:
                    run.info(f"{src} {tag}: exists")
                    continue
                if tag == "P0":
                    model, br = T.load_pmodel("P0", dev), False
                else:
                    ck = torch.load(T.proot("runs", tag) / "ckpt-final.pt", map_location="cpu", weights_only=False)
                    model = T.PModel(ck["model"]["arm"]).to(dev).eval()
                    br = RM.has_branch(tag)
                    if br:
                        sd = json.loads((T.proot("runs", tag) / RM.SIDECAR).read_text())
                        assert not sd.get("reflex") and not sd["head"].get("order") and not sd["head"].get("latfeat"), f"{tag}: unsupported branch variant"
                        RM.attach(model, **{k: v for k, v in sd["head"].items() if k not in ("order", "latfeat")})
                        model.to(dev)
                    model.load_state(ck["model"])
                    model.eval()
                assert model.lead_cols is None, f"{tag}: lead adapter not supported"
                sl = model.net.slices
                pi = torch.arange(sl["plan"].start, sl["plan"].start + 495, device=dev)
                dsl = sl.get("desire_state")

                def run_arms(arms, rm_on=True):
                    X, Y, P, D = [], [], [], []
                    for _, d, blk in arms:
                        cur["x"] = queue(d, blk)
                        xs, ys, ps, ds = [], [], [], []
                        with torch.no_grad():
                            for i in range(0, len(idx), 256):
                                r = idx[i:i + 256]
                                e = torch.cat([S.ego[r].float(), Fr[r] if rm_on else torch.zeros_like(Fr[r])], 1) if br else S.ego[r]
                                o = model(S.front[r], e, S.tc[r], None, None, nv=None).float()
                                x, y, psi = T.rear(o[:, pi].view(-1, 33, 15), S.cam_x[r], W)
                                xs.append(x.cpu()), ys.append(y.cpu()), ps.append(psi.cpu())
                                if dsl is not None:
                                    ds.append(torch.softmax(o[:, dsl], -1).cpu())
                        cur["x"] = None
                        X.append(torch.cat(xs)), Y.append(torch.cat(ys)), P.append(torch.cat(ps))
                        D.append(torch.cat(ds) if ds else torch.zeros(len(idx), 8))
                    return [torch.stack(v).numpy().astype(np.float32) for v in (X, Y, P, D)]

                # G-none: the wrapped path with no desire = the original forward, bit for bit (first batch)
                with torch.no_grad():
                    r = idx[:256]
                    e = torch.cat([S.ego[r].float(), Fr[r]], 1) if br else S.ego[r]
                    T.A.policy_feeds = _pf
                    o0 = model(S.front[r], e, S.tc[r], None, None, nv=None)
                    T.A.policy_feeds = pf
                    o1 = model(S.front[r], e, S.tc[r], None, None, nv=None)
                gnone = bool(torch.equal(o0, o1))
                assert gnone, f"{tag}: G-none failed"
                x, y, p, d = run_arms(ARMS)
                extra = {}
                if br:
                    xr, yr, pr, dr = run_arms(RM_ARMS, rm_on=False)
                    extra = dict(rm_x=xr, rm_y=yr, rm_psi=pr, rm_ds=dr, rm_arms=np.array([k for k, _, _ in RM_ARMS]))
                np.savez(f, names=R["names"], arms=np.array([k for k, _, _ in ARMS]), x=x, y=y, psi=p, ds=d, branch=br, ds_ok=dsl is not None, **extra)
                run.info(f"{src} {tag}: rows {len(idx)}, branch {br}, G-none {gnone}, desire_state {dsl is not None}; written")
                run.summary[f"{src}/{tag}"] = dict(rows=int(len(idx)), branch=bool(br), g_none=gnone)
                model = None
                torch.cuda.empty_cache()
            S = None
            torch.cuda.empty_cache()
    T.A.policy_feeds = _pf


# ---------------------------------------------------------------- report (CPU)
def cboot(v, g, B=10000, seed=0, den=None):
    """Mean of v (or ratio mean(v) / mean(den)) with a cluster bootstrap over groups g -> (point, lo, hi)."""
    v = np.asarray(v, float)
    u, inv = np.unique(np.asarray(g).astype(str), return_inverse=True)
    sv, n = np.bincount(inv, v, len(u)), np.bincount(inv, None, len(u))
    sd = np.bincount(inv, np.asarray(den, float), len(u)) if den is not None else n
    w = np.random.default_rng(seed).multinomial(len(u), np.full(len(u), 1 / len(u)), size=B)
    num = (w @ sv) / (w @ n)
    dd = (w @ sd) / (w @ n) if den is not None else 1.0
    pt = sv.sum() / sd.sum() if den is not None else sv.sum() / n.sum()
    lo, hi = np.percentile(num / dd, [2.5, 97.5])
    return float(pt), float(lo), float(hi)


def fm(t, f=".2f"):
    return f"{t[0]:{f}} [{t[1]:{f}}, {t[2]:{f}}]"


def load_stage(root, src, stage):
    tags = ["P0"] if stage == "P0" else [f"{stage}-s0", f"{stage}-s1"]
    zs = [np.load(root / "plans" / src / f"{t}.npz") for t in tags if (root / "plans" / src / f"{t}.npz").exists()]
    if not zs:
        return None
    arms = [str(x) for x in zs[0]["arms"]]
    out = {k: np.mean([z[k] for z in zs], 0) for k in ("x", "y", "psi", "ds")}
    if "rm_y" in zs[0].files:
        out |= {k: np.mean([z[k] for z in zs], 0) for k in ("rm_x", "rm_y", "rm_psi")}
        out["rm_arms"] = [str(x) for x in zs[0]["rm_arms"]]
    out["arms"], out["n_seeds"], out["ds_ok"] = arms, len(zs), bool(zs[0]["ds_ok"])
    return out


def resp(Q, arm, key="y", pre=""):
    """Signed response (n, len(TS)) of an arm against none: y in m, psi in degrees."""
    arms = Q[pre + "arms"] if pre else Q["arms"]
    a, b = Q[pre + key][arms.index(arm)], Q[pre + key][arms.index("none")]
    s = SIDE[arm.split("@")[0]]
    return s * (a - b) * (np.degrees(1.0) if key == "psi" else 1.0)


def lc_pooled(Q, age="0", key="y", pre=""):
    return 0.5 * (resp(Q, f"LCL@{age}", key, pre) + resp(Q, f"LCR@{age}", key, pre))


def side_resp(Q, side, kind, age="0", key="y", opposite=False, pre=""):
    """Per-row response to the desire on the row's side (+1 left / -1 right); kind T or LC; opposite: the other side's desire."""
    s = -side if opposite else side
    L, Rr = (f"TL@{age}", f"TR@{age}") if kind == "T" else (f"LCL@{age}", f"LCR@{age}")
    return np.where((s > 0)[:, None], resp(Q, L, key, pre), resp(Q, Rr, key, pre))


def classify_base(m, dirc, kind):
    if kind == "lc":
        return "full" if m >= 2.5 else ("partial" if m >= 0.5 and dirc >= 0.90 else "absent")
    return "full" if m >= 10 else ("partial" if m >= 3 else "absent")


def keep_class(r):
    pt, lo, hi = r
    if pt < 0.30 and hi < 0.50:
        return "killed"
    if pt >= 0.70 and lo > 0.50:
        return "kept"
    return "weakened"


def cmd_report(a):
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    L, res = ["# DESIRE-CAP open loop", "", "Responses are paired against the same row's none arm, signed toward the desire side; stage = mean of its seeds row by row. "
              "CI = log / scene cluster bootstrap (B 10 000, seed 0).", ""], {}
    Rn, Rp = np.load(root / "rows_nav.npz"), np.load(root / "rows_pai.npz")
    Qn = {st: load_stage(root, "nav", st) for st, _ in STAGES}
    Qp = {st: load_stage(root, "pai", st) for st, _ in STAGES}
    st_ok = [st for st, _ in STAGES if Qn[st] is not None]
    g = Rn["log"]
    # ---- primary lane change
    m = Rn["in_C_fast"]
    P0lc = lc_pooled(Qn["P0"])[m, I4]
    L += ["## 1. Lane change (primary): C_fast (v0 >= 8.9 m/s, human keeps lane), LCL / LCR pooled, pulse age 0", "",
          f"rows {int(m.sum())}, logs {len(set(g[m].tolist()))}", "",
          "| stage | seeds | R_y 1 s | 2 s | 4 s mean [CI] | 4 s median | 6 s | 8 s | R_psi 4 s (deg) | dir ok | full / partial / flat | LCL 4 s | LCR 4 s | ratio vs P0 [CI] | class |",
          "|:--|--:|--:|--:|:--|--:|--:|--:|--:|--:|:--|--:|--:|:--|:--|"]
    for st in st_ok:
        Q = Qn[st]
        R = lc_pooled(Q)[m]
        r4 = R[:, I4]
        ps = lc_pooled(Q, key="psi")[m, I4]
        dirc = float((r4 > 0.1).mean())
        sh = [(r4 >= 2.5).mean(), ((r4 >= 0.5) & (r4 < 2.5)).mean(), (r4 < 0.5).mean()]
        rl, rr = resp(Q, "LCL@0")[m, I4].mean(), resp(Q, "LCR@0")[m, I4].mean()
        rat = cboot(r4, g[m], den=P0lc) if st != "P0" else (1.0, 1.0, 1.0)
        cls = classify_base(r4.mean(), dirc, "lc") if st == "P0" else keep_class(rat)
        res.setdefault("lc", {})[st] = dict(mean4=float(r4.mean()), med4=float(np.median(r4)), dir=dirc, ratio=rat, cls=cls, shares=[float(x) for x in sh],
                                            m8=float(R[:, 4].mean()), shape="flat" if np.median(r4) < 0.5 else ("full" if np.median(r4) >= 2.5 else "graded"))
        L.append(f"| {st} | {Q['n_seeds']} | {R[:, 0].mean():.2f} | {R[:, 1].mean():.2f} | {fm(cboot(r4, g[m]))} | {np.median(r4):.2f} | {R[:, 3].mean():.2f} | "
                 f"{R[:, 4].mean():.2f} | {ps.mean():.1f} | {dirc:.3f} | {sh[0]:.2f} / {sh[1]:.2f} / {sh[2]:.2f} | {rl:.2f} | {rr:.2f} | "
                 f"{'-' if st == 'P0' else fm(rat)} | {cls} |")
    # ---- primary turn
    mt = Rn["in_T_L"] | Rn["in_T_R"]
    sd = Rn["side"]
    P0t = side_resp(Qn["P0"], sd, "T", key="psi")[mt, I4]
    L += ["", "## 2. Turn (primary): T_L + T_R (human turning in a junction), turn desire on the human's side, pulse age 0", "",
          f"rows {int(mt.sum())}, logs {len(set(g[mt].tolist()))}", "",
          "| stage | R_psi 4 s mean [CI] (deg) | median | R_y 4 s (m) | R_psi 2 s | opposite-side desire R_psi 4 s | ratio vs P0 [CI] | class |",
          "|:--|:--|--:|--:|--:|--:|:--|:--|"]
    for st in st_ok:
        Q = Qn[st]
        rp = side_resp(Q, sd, "T", key="psi")[mt]
        ry = side_resp(Q, sd, "T")[mt, I4]
        op = side_resp(Q, sd, "T", key="psi", opposite=True)[mt, I4]
        rat = cboot(rp[:, I4], g[mt], den=P0t) if st != "P0" else (1.0, 1.0, 1.0)
        cls = classify_base(rp[:, I4].mean(), None, "t") if st == "P0" else keep_class(rat)
        res.setdefault("turn", {})[st] = dict(mean4=float(rp[:, I4].mean()), ratio=rat, cls=cls, opp=float(op.mean()))
        L.append(f"| {st} | {fm(cboot(rp[:, I4], g[mt]), '.1f')} | {np.median(rp[:, I4]):.1f} | {ry.mean():.2f} | {rp[:, 1].mean():.1f} | {op.mean():.1f} | "
                 f"{'-' if st == 'P0' else fm(rat)} | {cls} |")
    # ---- pulse age (dose / timing)
    L += ["", "## 3. Pulse age (open loop: frames do not move after the pulse)", "",
          "| stage | LC R_y 4 s @ 0 / 0.4 / 1.0 / 2.0 s | LC R_y 8 s @ same | turn R_psi 4 s @ same |", "|:--|:--|:--|:--|"]
    for st in st_ok:
        Q = Qn[st]
        lc4 = " / ".join(f"{lc_pooled(Q, f'{x:g}')[m, I4].mean():.2f}" for x in AGES)
        lc8 = " / ".join(f"{lc_pooled(Q, f'{x:g}')[m, 4].mean():.2f}" for x in AGES)
        t4 = " / ".join(f"{side_resp(Q, sd, 'T', f'{x:g}', key='psi')[mt, I4].mean():.1f}" for x in AGES)
        L.append(f"| {st} | {lc4} | {lc8} | {t4} |")
    # ---- companion readings
    import importlib.util
    spec = importlib.util.spec_from_file_location("led_sl", REPO / "experiments/cap_ledger/scripts/led.py")
    signed_lat = None
    try:
        led = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(led)
        signed_lat = led.signed_lat
    except Exception as ex:  # noqa: BLE001
        L.append(f"(led.signed_lat unavailable: {ex})")
    L += ["", "## 4. Companion readings (report only)", "",
          "| stage | C_slow LC R_y 4 s | C_slow turn R_psi 4 s (TL/TR pooled) | LC_L+LC_R success none -> same-side LC@0 | LC_L+LC_R R_y 4 s | "
          "P_keep LC R_y 4 s | P_lca route-side LC R_y 4 s | P_turn route-side turn R_psi 4 s | desire_state p(own) LC@0 C_fast none -> LC |",
          "|:--|--:|--:|:--|--:|--:|--:|--:|:--|"]
    ml = Rn["in_LC_L"] | Rn["in_LC_R"]
    for st in st_ok:
        Q, P = Qn[st], Qp[st]
        cs = Rn["in_C_slow"]
        tsl = 0.5 * (resp(Q, "TL@0", "psi") + resp(Q, "TR@0", "psi"))[cs, I4].mean()
        succ = "-"
        if signed_lat is not None:
            def lc_ok(arm_y, arm_x):
                o = np.array([signed_lat(Rn["lane_c"][k].astype(float), np.array([[arm_x[k], arm_y[k]]]))[0] if Rn["lane_ok"][k] else np.nan
                              for k in np.flatnonzero(ml)])
                h = Rn["off4"][ml]
                return np.mean((np.sign(o) == np.sign(h)) & (np.abs(o) >= 0.5 * np.abs(h)))
            ai = Q["arms"].index
            yn, xn = Q["y"][0][:, I4], Q["x"][0][:, I4]
            yd = np.where(Rn["side"] > 0, Q["y"][ai("LCL@0")][:, I4], Q["y"][ai("LCR@0")][:, I4])
            xd = np.where(Rn["side"] > 0, Q["x"][ai("LCL@0")][:, I4], Q["x"][ai("LCR@0")][:, I4])
            succ = f"{lc_ok(yn, xn):.3f} -> {lc_ok(yd, xd):.3f}"
        lcr = side_resp(Q, Rn["side"], "LC")[ml, I4].mean()
        pk = pl = ptn = float("nan")
        if P is not None:
            pk = lc_pooled(P)[Rp["in_P_keep"], I4].mean()
            pl = side_resp(P, Rp["side"], "LC")[Rp["in_P_lca"], I4].mean()
            ptn = side_resp(P, Rp["side"], "T", key="psi")[Rp["in_P_turn"], I4].mean()
        dsx = "-"
        if Q["ds_ok"]:
            ai = Q["arms"].index
            p_none = 0.5 * (Q["ds"][0][m, 3] + Q["ds"][0][m, 4])
            p_lc = 0.5 * (Q["ds"][ai("LCL@0")][m, 3] + Q["ds"][ai("LCR@0")][m, 4])
            dsx = f"{p_none.mean():.3f} -> {p_lc.mean():.3f}"
        res.setdefault("comp", {})[st] = dict(pk=float(pk), plca=float(pl), pturn=float(ptn))
        L.append(f"| {st} | {lc_pooled(Q)[cs, I4].mean():.2f} | {tsl:.1f} | {succ} | {lcr:.2f} | {pk:.2f} | {pl:.2f} | {ptn:.1f} | {dsx} |")
    # ---- route memory cleared (branch stages)
    L += ["", "## 5. Route-memory stages with the memory cleared (served empty memory), pulse age 0", "",
          "| stage | C_fast LC R_y 4 s (memory fed) | (memory cleared) | T turn R_psi 4 s (fed) | (cleared) |", "|:--|--:|--:|--:|--:|"]
    for st in st_ok:
        Q = Qn[st]
        if "rm_y" not in Q:
            continue
        L.append(f"| {st} | {lc_pooled(Q)[m, I4].mean():.2f} | {lc_pooled(Q, pre='rm_')[m, I4].mean():.2f} | "
                 f"{side_resp(Q, sd, 'T', key='psi')[mt, I4].mean():.1f} | {side_resp(Q, sd, 'T', key='psi', pre='rm_')[mt, I4].mean():.1f} |")
    # ---- gates / lines
    L += ["", "## 6. Registered lines", ""]
    b = res["lc"]["P0"]
    L.append(f"- Line 1 (base lane change): P0 {b['mean4']:.2f} m at 4 s, direction ok {b['dir']:.3f} -> **{b['cls']}**; 8 s {b['m8']:.2f} m"
             + (" (full but slow)" if b["mean4"] < 2.5 and b["m8"] >= 3.0 else ""))
    bt = res["turn"]["P0"]
    L.append(f"- Line 2 (base turn): P0 {bt['mean4']:.1f} deg at 4 s -> **{bt['cls']}**")
    for kind in ("lc", "turn"):
        if res[kind]["P0"]["cls"] == "absent":
            L.append(f"- Line 3 ({kind}): not judged (P0 absent)")
            continue
        cl = {st: res[kind][st]["cls"] for st in res[kind]}
        killer = [st for st, pre in STAGES if pre and st in cl and cl[st] == "killed" and cl.get(pre) != "killed"]
        L.append(f"- Line 3 ({kind}): " + ", ".join(f"{st} {res[kind][st]['ratio'][0]:.2f} {cl[st]}" for st in cl if st != "P0")
                 + f"; first killing stage per branch: {killer or 'none'}")
    L.append("- Line 4 (shape, lane change, median at 4 s): " + ", ".join(f"{st} {res['lc'][st]['shape']}" for st in res["lc"]))
    if (DATA / "runs/cap_ledger/plans/H/P0.npz").exists() and (root / "plans/nav/P0.npz").exists():
        c = np.load(DATA / "runs/cap_ledger/plans/H/P0.npz")
        pos = {n: i for i, n in enumerate(c["names"].tolist())}
        k = np.array([pos[n] for n in Rn["names"].tolist()])
        q = np.load(root / "plans/nav/P0.npz")
        d = np.hypot(q["x"][0][:, I4] - c["plan"][k, 7, 0], q["y"][0][:, I4] - c["plan"][k, 7, 1])
        L.append(f"- G-rep: P0 none vs cap_ledger plans/H/P0 at 4 s: median {np.median(d):.2e} m, max {d.max():.2e} m")
    (out / "desire_cap.md").write_text("\n".join(L) + "\n")
    (out / "lines.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("rows")
    p = sp.add_parser("plans")
    p.add_argument("--tags", nargs="+", default=TAGS)
    p.add_argument("--srcs", nargs="+", default=["nav", "pai"])
    p.add_argument("--force", action="store_true")
    p = sp.add_parser("report")
    p.add_argument("--root", default=str(OUT))
    p.add_argument("--out", default=str(REPO / "experiments/desire_cap/results"))
    a = ap.parse_args()
    {"rows": cmd_rows, "plans": cmd_plans, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
