#!/usr/bin/env python3
"""ROUND3 (plans/2026-10-21-round3-prereg.md): where does round 2's launch loss come from? Offline diagnosis, the J-W draw file and the read-outs.
Box, repo root (op-train env for the pool rows, .venv for the read stores; both have numpy / pandas).

  r3.py rows                      state of every PAI human row (plain + both stretches) by its own fed speed v0 and 4 s label: standing
        (v0 < 0.3 m/s and the label path < 0.5 m), launching (v0 < 1 m/s, path >= 0.5 m), stopping (v0 >= 1, the label's last 1 s < 0.5 m),
        moving; for standing rows the human's wait after the label window (plain-row sequence of the scene, 10 Hz); counted over R2JF's
        drawable rows and as expected rows per batch; the same states for the 32 S / trainval pool rows (JT00S) -> results/rows_state.{md,json}
  r3.py launch                    on the launch scenes (handoff < 1 m/s, log moves > 5 m) of the stored reads: the human's launch time t_L
        (log 0.5 m past its handoff pose), the read's model plan (poses_model) at the first decision at or after t_L, the first decision whose
        plan covers >= 2 m, the ego's own launch -> results/launch_plans.{md,json,csv}
  r3.py weights --out R3JW        J-W: R2JF with every standing PAI human row (the `rows` rule, each row's own label) at weight 0, each part
        renormalised to its R2JF share -> $DATA_DIR/runs/defect_fix/w/R3JW.npz + results/w_R3JW.json
  r3.py steps                     results/steps.json (both arms 9 500 steps = round 2's J and CTRL) -> $O/steps.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
CR, W = DATA / "runs/op_parity/cache", DATA / "runs/defect_fix/w"
O, RES = DATA / "runs/round3", REPO / "experiments/round3/results"
PARTS = {"ph": "on1_PAIH", "ph05": "on1_PAIH05", "ph15": "on1_PAIH15"}
V_STAND, L_STAND, V_LAUNCH = 0.3, 0.5, 1.0                     # the registered row-state rule (prereg)
K_PAI, K_S = 32, 32                                            # expected PAI human / S rows per batch (round 2, J)


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float))


def path_len(fut):
    """fut (n, 8, 3) poses at 0.5 .. 4 s -> (n,) path length from the origin, (n,) length of the last 1 s."""
    P = np.concatenate([np.zeros((len(fut), 1, 2), np.float32), fut[:, :, :2]], 1)
    seg = np.linalg.norm(np.diff(P, axis=1), axis=-1)
    return seg.sum(1), seg[:, -2:].sum(1)


def state(v0, fut):
    L, L1 = path_len(fut)
    st = np.full(len(v0), "moving", dtype="U9")
    st[(v0 >= V_LAUNCH) & (L1 < 0.5)] = "stopping"
    st[(v0 < V_LAUNCH) & (L >= L_STAND)] = "launching"
    st[(v0 < V_STAND) & (L < L_STAND)] = "standing"
    st[(v0 >= V_STAND) & (v0 < V_LAUNCH) & (L < L_STAND)] = "creeping"
    return st, L


def wait_after(tab):
    """Plain rows: per row the log time (s) from the row's t until the human has moved 0.5 m (inf if not within the clip's rows + the last
    row's label), minus 4 s (the label window)."""
    names, log, v = tab["names"].astype(str), tab["log"].astype(str), tab["speed"].astype(np.float64)
    k = np.array([int(x.rsplit(":", 1)[1]) for x in names])
    fut = tab["fut"]
    out = np.full(len(names), np.nan)
    by = defaultdict(list)
    for i, s in enumerate(log):
        by[s].append(i)
    for s, ix in by.items():
        ix = np.array(ix)[np.argsort(k[ix])]
        kk = k[ix]
        s_ = np.r_[0.0, np.cumsum(0.5 * (v[ix][1:] + v[ix][:-1]) * 0.1 * np.diff(kk))]  # station along the 10 Hz plain rows
        L, _ = path_len(fut[ix[-1:]])
        ext_t = kk[-1] * 0.1 + np.arange(1, 9) * 0.5                                 # past the last row: its label
        Pl = np.concatenate([[[0, 0]], fut[ix[-1], :, :2]])
        ext_s = s_[-1] + np.cumsum(np.linalg.norm(np.diff(Pl, axis=0), axis=1))
        T_ = np.r_[kk * 0.1, ext_t]
        S_ = np.r_[s_, ext_s]
        for j, i in enumerate(ix):
            m = np.nonzero(S_ > S_[j] + 0.5)[0]
            out[i] = (T_[m[0]] - kk[j] * 0.1 - 4.0) if len(m) else np.inf
    return out


def wait_bucket(w):
    return np.select([w <= 2, w <= 6, w <= 10, np.isfinite(w)], ["<=2 s", "2-6 s", "6-10 s", ">10 s"], "not in clip")


def load_part(p):
    t = np.load(CR / PARTS[p] / "tab.npz")
    return {k: t[k] for k in t.files}


def pai_states():
    """-> {part: dict(names, state, L, wait)}; stretch rows take the wait of their plain base row scaled to the stretch clock (/ f)."""
    T = {p: load_part(p) for p in PARTS}
    out = {}
    w_plain = wait_after(T["ph"])
    wp = dict(zip(T["ph"]["names"].astype(str), w_plain))
    for p, t in T.items():
        st, L = state(t["speed"].astype(np.float64), t["fut"])
        if p == "ph":
            w = w_plain
        else:  # base row's wait (log s after its own 4 s window); the stretch label covers 4 f s of log
            w = np.array([wp.get(b, np.nan) for b in t["base"].astype(str)]) + 4.0 - 4.0 * t["f"]
        out[p] = dict(names=t["names"].astype(str), log=t["log"].astype(str), state=st, L=L, wait=w)
    return out


STATES = ("standing", "creeping", "launching", "stopping", "moving")
BUCKETS = ("<=2 s", "2-6 s", "6-10 s", ">10 s", "not in clip")


def cmd_rows(a):
    S = pai_states()
    w = np.load(W / "R2JF.npz")
    wn = dict(zip(w["names"].astype(str), w["w"]))
    rec, L = dict(rule=dict(standing=f"v0 < {V_STAND} and label path < {L_STAND} m", creeping=f"{V_STAND} <= v0 < {V_LAUNCH}, path < {L_STAND} m",
                            launching=f"v0 < {V_LAUNCH} and path >= {L_STAND} m", stopping=f"v0 >= {V_LAUNCH} and the label's last 1 s < 0.5 m",
                            moving="else"), parts={}), []
    L += ["# ROUND3: PAI human rows by state (own fed speed and own 4 s label)", "",
          f"Rule: standing v0 < {V_STAND} m/s and label path < {L_STAND} m; creeping {V_STAND} <= v0 < {V_LAUNCH}, path < {L_STAND} m; launching v0 < "
          f"{V_LAUNCH}, path >= {L_STAND} m; stopping v0 >= {V_LAUNCH}, last 1 s of the label < 0.5 m; moving otherwise. `drawable` = R2JF weight > 0 "
          "(outside held and the exam links); `per batch` = expected rows of that state in J's batch (32 PAI human rows).", "",
          "| part | drawable | " + " | ".join(STATES) + " | standing per batch | launching per batch |", "|:--|--:|" + "--:|" * (len(STATES) + 2)]
    tot = defaultdict(float)
    for p, d in S.items():
        wt = np.array([wn.get(x, 0.0) for x in d["names"]])
        dr = wt > 0
        sh = wt.sum()
        c = {s: int(((d["state"] == s) & dr).sum()) for s in STATES}
        pb = {s: float(wt[d["state"] == s].sum() * K_PAI) for s in STATES}
        for s in STATES:
            tot[s] += pb[s]
        stand = (d["state"] == "standing") & dr
        wb = wait_bucket(d["wait"][stand])
        rec["parts"][p] = dict(drawable=int(dr.sum()), share=float(sh), counts=c, per_batch=pb,
                               standing_wait={b: int((wb == b).sum()) for b in BUCKETS},
                               standing_scenes=int(len(set(d["log"][stand]))),
                               standing_rows_per_scene_med=float(np.median(np.unique(d["log"][stand], return_counts=True)[1])) if stand.any() else 0.0)
        L.append(f"| {p} | {int(dr.sum())} | " + " | ".join(f"{c[s]} ({c[s] / max(dr.sum(), 1):.1%})" for s in STATES)
                 + f" | {pb['standing']:.2f} | {pb['launching']:.2f} |")
    L.append(f"| all (per batch) | | " + " | ".join(f"{tot[s]:.2f}" for s in STATES) + f" | {tot['standing']:.2f} | {tot['launching']:.2f} |")
    rec["per_batch_all"] = dict(tot)
    L += ["", "Standing rows: the human's wait after the label window (log s until it has moved 0.5 m; stretch rows on their own clock)", "",
          "| part | standing scenes | " + " | ".join(BUCKETS) + " |", "|:--|--:|" + "--:|" * len(BUCKETS)]
    for p, r in rec["parts"].items():
        n = sum(r["standing_wait"].values())
        L.append(f"| {p} | {r['standing_scenes']} | " + " | ".join(f"{r['standing_wait'][b]} ({r['standing_wait'][b] / max(n, 1):.0%})" for b in BUCKETS) + " |")
    # the pool's S / trainval rows (both arms draw 32 of them a batch from JT00S)
    t = np.load(CR / "on1_JT/tab.npz")
    ws = np.load(W / "JT00S.npz")
    assert np.array_equal(ws["names"].astype(str), t["names"].astype(str))
    st, _ = state(t["speed"].astype(np.float64), t["fut"])
    wv = ws["w"] / ws["w"].sum()
    sb = {s: float(wv[st == s].sum() * K_S) for s in STATES}
    rec["s_rows_per_batch"] = sb
    L += ["", "The 32 S / trainval pool rows of a batch (JT00S, both arms): " + ", ".join(f"{s} {sb[s]:.2f}" for s in STATES), ""]
    L += ["Per batch, J vs CTRL (pool rows only; the navtrain rows are 51 in J vs 83 in CTRL, their states not counted here): standing "
          f"{sb['standing'] + tot['standing']:.2f} vs {sb['standing']:.2f}, launching {sb['launching'] + tot['launching']:.2f} vs {sb['launching']:.2f}."]
    dump(RES / "rows_state.json", rec)
    (RES / "rows_state.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


# ---------------------------------------------------------------- launch-moment plans of stored reads
HAND = 1.6e6
READS = {  # name -> (read root, tax pkl, subset): the stored reads of round 2 and their references
    "R2J-281ag": ("runs/round2/pai281/R2J-L18-s0-ag", "runs/pai_fix/tax/R2J-L18-s0-281ag.pkl", "281"),
    "NTV0-281ag": ("runs/defect_fix/pai_accgap/NTV0-L18-s0", "runs/pai_fix/tax/NTV0-L18-s0-281ag.pkl", "281"),
    "R2J-vl": ("runs/pai_fix/reads/R2J-L18-s0-vl", "runs/pai_fix/tax/R2J-L18-s0-vl.pkl", "fix"),
    "R2C-vl": ("runs/pai_fix/reads/R2C-L18-s0-vl", "runs/pai_fix/tax/R2C-L18-s0-vl.pkl", "fix"),
    "NTV0-vl": ("runs/pai_fix/reads/NTV0-L18-s0-vl", "runs/pai_fix/tax/NTV0-L18-s0-vl.pkl", "fix"),
    "R2JH-vl": ("runs/pai_fix/reads/R2JH-L18-s0-vl", "runs/pai_fix/tax/R2JH-L18-s0-vl.pkl", "fix"),
}


def plan_arc(P):
    P = np.asarray(P, np.float64)[:, :2]
    return float(np.linalg.norm(np.diff(np.r_[[[0.0, 0.0]], P], axis=0), axis=1).sum())


def _launch_scene(job):
    o, recs = job
    C, Lg = o["ctrl"], o["log"]
    t_c, T, P = np.asarray(C["t"], float), np.asarray(Lg["T"], float), np.asarray(Lg["P"], float)
    th = t_c[0] + HAND
    ph = np.array([np.interp(th, T, P[:, 0]), np.interp(th, T, P[:, 1])])
    d = np.hypot(*(P - ph).T)
    m = np.nonzero((T >= th) & (d > 0.5))[0]
    if not len(m):
        return None
    tL = float(T[m[0]])
    sl = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    h4 = float(np.interp(tL + 4e6, T, sl) - np.interp(tL, T, sl))
    ex, ey = np.asarray(C["x"], float), np.asarray(C["y"], float)
    eh = np.array([np.interp(th, t_c, ex), np.interp(th, t_c, ey)])
    me = np.nonzero((t_c >= th) & (np.hypot(ex - eh[0], ey - eh[1]) > 0.5))[0]
    tE = float(t_c[me[0]]) if len(me) else np.nan
    # lead at t_L: a vehicle centre 0-25 m ahead, |lat| < 2 m in the human's frame at t_L
    Y = np.unwrap(np.asarray(Lg["Y"], float))
    a = np.array([np.interp(tL, T, P[:, 0]), np.interp(tL, T, P[:, 1]), np.interp(tL, T, Y)])
    j = int(np.argmin(np.abs(np.asarray(o["tg"], float) - tL)))
    tr = o["traffic"][j]
    lead, lead_v = False, np.nan
    if len(tr["id"]):
        c, s = np.cos(a[2]), np.sin(a[2])
        dx, dy = np.asarray(tr["x"]) - a[0], np.asarray(tr["y"]) - a[1]
        lx, ly = c * dx + s * dy, -s * dx + c * dy
        veh = np.array([("bicycle" not in str(k)) and ("pedestrian" not in str(k)) for k in tr["cls"]])
        q = (lx > 0) & (lx < 25) & (np.abs(ly) < 2.0) & veh
        if q.any():
            lead, lead_v = True, float(np.asarray(tr["v"])[q][np.argmin(lx[q])])
    R = sorted((r for r in recs if r["t0"] >= th), key=lambda r: r["t0"])
    at = next((r for r in R if r["t0"] >= tL), None)
    go = next((r for r in R if plan_arc(r.get("poses_model") or r["poses"]) >= 2.0), None)
    win = [r for r in R if tL <= r["t0"] <= tL + 3e6]
    short = [(r["ego"][4] * 10 < 0.5) and plan_arc(r.get("poses_model") or r["poses"]) < 1.0 for r in win]
    pre = [plan_arc(r.get("poses_model") or r["poses"]) for r in R if th <= r["t0"] < tL]
    out = dict(scene=o["scene"], score=float(o["score"] or 0), t_L=(tL - th) / 1e6, h4=h4, lead=lead, lead_v=lead_v,
               ego_move=(tE - tL) / 1e6 if np.isfinite(tE) else np.nan, n_dec=len(R),
               plan_go=(go["t0"] - tL) / 1e6 if go else np.nan,
               short3=float(np.mean(short)) if short else np.nan,
               pre_arc_max=float(max(pre)) if pre else np.nan)
    if at is not None:
        Pm = at.get("poses_model") or at["poses"]
        out |= dict(v0_at=float(at["ego"][4] * 10), arc4_at=plan_arc(Pm), arc2_at=plan_arc(np.asarray(Pm)[:4]), dt_at=(at["t0"] - tL) / 1e6)
    return out


def cmd_launch(a):
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    sys.path[:0] = [str(REPO / "experiments/pai_fix/scripts")]
    import pai_fix as PF
    FA = DATA / "runs/pai_fix/facts"
    facts = {"R2J-281ag": "R2J-L18-s0-281ag", "NTV0-281ag": "NTV0-L18-s0-281ag", "R2J-vl": "R2J-L18-s0-vl", "R2C-vl": "R2C-L18-s0-vl",
             "NTV0-vl": "NTV0-L18-s0-vl", "R2JH-vl": "R2JH-L18-s0-vl"}
    ex = pd.read_csv(REPO / "experiments/pai_fix/scenes/exam.csv")
    sub_col = next(c for c in ("subset", "sub", "group") if c in ex.columns)
    gl = set(ex.scene[ex[sub_col].astype(str).str.lower().str.contains("launch")]) if "scene" in ex.columns else set()
    rows = []
    for name, (root, pkl, subset) in READS.items():
        F = pd.read_csv(FA / f"{facts[name]}.csv").set_index("scene")
        F = F[~F.rpc]
        Lm = PF.launch_mask(F)
        sc = set(F.index[Lm])
        A = {o["scene"]: o for o in pickle.load(open(DATA / pkl, "rb"))["rolls"] if "ctrl" in o and o["scene"] in sc}
        recs = defaultdict(list)
        for f in glob.glob(str(DATA / root / "runs/*/driver/drive.jsonl")):
            for line in open(f):
                if '"infer": true' not in line:
                    continue
                i = line.find('"scene": "')
                s = line[i + 10:i + 53]
                if s not in A:
                    continue
                r = json.loads(line)
                if r["kind"] == "drive":
                    recs[r["scene"]].append(r)
        with ProcessPoolExecutor(16) as px:
            out = list(px.map(_launch_scene, [(A[s], recs[s]) for s in sorted(A)]))
        for r in out:
            if r is None:
                continue
            r |= dict(read=name, subset=subset, launched=bool(PF.launched(F.loc[[r["scene"]]]).iloc[0]), g_launch=r["scene"] in gl,
                      fclass=F.loc[r["scene"], "fclass"])
            rows.append(r)
    D = pd.DataFrame(rows)
    RES.mkdir(parents=True, exist_ok=True)
    D.to_csv(RES / "launch_plans.csv", index=False)
    md, rec = ["# ROUND3: model plans at the human's launch (stored reads, launch scenes)", "",
               "t_L = the log is 0.5 m past its handoff pose; `arc4@tL` = path length of the model plan (poses_model) at the first decision at or after t_L; "
               "`go` = time from t_L to the first decision whose plan covers >= 2 m (negative: before the human); `short3` = share of decisions in "
               "[t_L, t_L + 3 s] with the ego still (< 0.5 m/s) and a plan under 1 m; `ego` = time from t_L until the ego is 0.5 m past its handoff "
               "pose. Medians; n = launch scenes of the read.", ""], {}

    def block(title, sel, reads):
        md.extend([f"## {title}", "", "| read | n | launched | arc4@tL med | arc4@tL < 1 m | go med (s) | no go | short3 mean | ego move med (s) | lead at t_L |",
                   "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|"])
        rec[title] = {}
        for rd in reads:
            g = D[(D.read == rd) & sel(D)]
            if not len(g):
                continue
            x = dict(n=len(g), launched=int(g.launched.sum()), arc4=float(g.arc4_at.median()), arc4_lt1=int((g.arc4_at < 1).sum()),
                     go=float(g.plan_go.median()), nogo=int(g.plan_go.isna().sum()), short3=float(g.short3.mean()), ego=float(g.ego_move.median()),
                     lead=int(g.lead.sum()))
            rec[title][rd] = x
            md.append(f"| {rd} | {x['n']} | {x['launched']} | {x['arc4']:.2f} | {x['arc4_lt1']} | {x['go']:+.2f} | {x['nogo']} | {x['short3']:.2f} | "
                      f"{x['ego']:+.2f} | {x['lead']} |")
        md.append("")

    block("PAI 281, all launch scenes (ACC + GAP)", lambda d: d.subset == "281", ["NTV0-281ag", "R2J-281ag"])
    pj = D[D.read == "R2J-281ag"].set_index("scene")
    pn = D[D.read == "NTV0-281ag"].set_index("scene")
    lost = sorted(set(pn.index[pn.launched]) & set(pj.index[~pj.launched]))
    block(f"PAI 281, the {len(lost)} launches NTV0 makes and R2J loses", lambda d: d.scene.isin(lost), ["NTV0-281ag", "R2J-281ag"])
    block("PAI-FIX G-LAUNCH (VC + LEAD)", lambda d: (d.subset == "fix") & d.g_launch, ["NTV0-vl", "R2C-vl", "R2J-vl", "R2JH-vl"])
    block("PAI-FIX all launch scenes (G-LAUNCH + TARGET, VC + LEAD)", lambda d: d.subset == "fix", ["NTV0-vl", "R2C-vl", "R2J-vl", "R2JH-vl"])
    # per-scene table of the lost launches
    md += [f"## The lost launches, per scene (R2J vs NTV0, ACC + GAP)", "",
           "| scene | lead at t_L (v) | t_L (s) | human 4 s (m) | NTV0 arc4@tL / go / ego | R2J arc4@tL / go / ego | R2J score | R2J class |", "|:--|:--|--:|--:|:--|:--|--:|:--|"]
    f2 = lambda v: "nan" if not np.isfinite(v) else f"{v:.1f}"  # noqa: E731
    for s in lost:
        n_, j_ = pn.loc[s], pj.loc[s]
        md.append(f"| {s[7:15]} | {'yes' if j_.lead else 'no'} ({f2(j_.lead_v)}) | {j_.t_L:.1f} | {j_.h4:.1f} | {f2(n_.arc4_at)} / {f2(n_.plan_go)} / {f2(n_.ego_move)} | "
                  f"{f2(j_.arc4_at)} / {f2(j_.plan_go)} / {f2(j_.ego_move)} | {j_.score:.2f} | {j_.fclass} |")
    rec["lost"] = lost
    dump(RES / "launch_plans.json", rec)
    (RES / "launch_plans.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def cmd_weights(a):
    S = pai_states()
    B = np.load(W / f"{a.base}.npz")
    names, w, grp = B["names"].astype(str), B["w"].astype(np.float64).copy(), B["grp"]
    idx = {x: i for i, x in enumerate(names)}
    rec = dict(base=a.base, rule=f"PAI human rows (plain and stretch) with v0 < {V_STAND} m/s and their own 4 s label path < {L_STAND} m -> weight 0; "
               "each part renormalised to its base share", parts={})
    for p, d in S.items():
        ix = np.array([idx[x] for x in d["names"]])
        sh = w[ix].sum()
        drop = (d["state"] == "standing") & (w[ix] > 0)
        keep = (w[ix] > 0) & ~drop
        w[ix[drop]] = 0.0
        w[ix[keep]] *= sh / w[ix[keep]].sum()
        rec["parts"][p] = dict(share=float(sh), drawable_before=int((drop | keep).sum()), dropped=int(drop.sum()), drawable_after=int(keep.sum()),
                               dropped_share=float(drop.sum() / max((drop | keep).sum(), 1)))
    assert abs(w.sum() - 1) < 1e-9 and abs(B["w"].sum() - 1) < 1e-9, (w.sum(), B["w"].sum())
    nonpai = ~np.isin(np.arange(len(w)), np.concatenate([[idx[x] for x in d["names"]] for d in S.values()]))
    assert np.array_equal(w[nonpai], B["w"][nonpai]), "non-PAI weights changed"
    rec |= dict(pai_drawable_rows=int(sum(v["drawable_after"] for v in rec["parts"].values())), out=a.out)
    np.savez(W / f"{a.out}.npz", names=B["names"], w=w, grp=grp)
    dump(RES / f"w_{a.out}.json", rec)
    print(json.dumps(rec, indent=1))


def cmd_steps(a):
    out = dict(rule="both arms take round 2's J / CTRL steps (prereg)", JW=dict(steps=a.n), JNF=dict(steps=a.n), CTRL=dict(steps=a.n, tag="R2C-L18-s0"))
    dump(O / "steps.json", out)
    dump(RES / "steps.json", out)
    print(json.dumps(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("rows")
    sub.add_parser("launch")
    p = sub.add_parser("weights")
    p.add_argument("--base", default="R2JF"), p.add_argument("--out", default="R3JW")
    p = sub.add_parser("steps")
    p.add_argument("--n", type=int, default=9500)
    a = ap.parse_args()
    dict(rows=cmd_rows, launch=cmd_launch, weights=cmd_weights, steps=cmd_steps)[a.cmd](a)


if __name__ == "__main__":
    main()
