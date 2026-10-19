#!/usr/bin/env python3
"""STAND-R (NR1 prereg supplement 20, plans/2026-10-20-nr1-supp20-standr.md; lib/nr1_free.py rolling / roll_cap, rm_train --on-stand-roll V).

  $TR stand_r.py gap --pool on1_JTX     (box, CPU) the station gap g = s_h(t + 4 s) - s_e (stop_label's h4c) of the pool's S rows
                                        -> $CR/<pool>/stand_gap.npz (names = the tab's), results/stand_r/gap.json
  $TR stand_r.py gate --out DIR         (box, CPU) offline gate G1-G3 -> DIR/gate.json, results/stand_r/gate.json; exit 0 pass, 2 fail
  $TR stand_r.py smoke --run T --log L --out JSON   60-step smoke: finite losses, the rolling part (loss/on/lon_standr_pos) > 0 at some step
  tmp/pcs_local/.venv/bin/python stand_r.py stage1 [--anchors]   (Mac) the stage 1 rule on the pulled reads -> results/stand_r/stage1.{json,md}
  tmp/pcs_local/.venv/bin/python stand_r.py easy --rows R --tag T --dec DIR --out NAME   (Mac) easy-miss classes of one complete held read
                                        (easy_items.py's classify; used for the B2 s1 + stop3 read)

Gate (decision-level, CPU, on B2's stored full-length reads STANDFMXH-L18-s0 / s1, ext_full_H / ext_full_H_s1 dec.parquet + hum.pkl):
  per decision, from the human's logged track: fut_reg arc = his station travel over 0.5 .. 4 s, fut arc = the ego's fed speed blended into his
  over 2 s (on_rows.progress), g = s_h(t0 + 4 s) - s_e; rows: human 4 s arc < 0.5 m and ego < 0.5 m/s (standing) or 0.5 <= v < 3 (rolling);
  cap and fire = nr1_free.stand_cap / roll_cap / stand_terms on the stored plan poses.
  G1  on the pool's standing + rolling rows, the human's two 4 s paths used as the plan fire 0 times.
  G2  rolling decisions (valid, k >= 2) of B2's easy red-line zero windows (easy_items misses, both seeds): fire share >= 0.80.
  G3  windows B2 scores 1 on: rolling decisions <= 1 % of their valid decisions (fires there described by stratum).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(HERE), str(REPO / "experiments/defect_fix/lib"), str(REPO / "experiments/onpol_nuplan/scripts"),
                str(REPO / "experiments/alpasim_proxy/scripts")]
import nr1_free as NF  # noqa: E402

DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
O = DATA / "runs/defect_fix"
CR = DATA / "runs/op_parity/cache"
RES = REPO / "experiments/defect_fix/results/stand_r"
ROWS = {"rb0": "L18RB0", "rtb1": "L18RTB1", "ys": "L18YS", "y": "L18Y", **{f"pn{i}": f"PN{i}" for i in range(6)}}
V_ROLL = 3.0
RED = "zero: static, red line, no lead"
EASY = "a: easy (garage = 1)"
CTL, ARMS = "STANDFMXH1K-L18-s0", {"R1": "STANDRH1K-L18-s0", "R2": "STANDR2H1K-L18-s0"}


# ---------------------------------------------------------------- gap labels
def _gap_job(job):
    import on_rows as OR
    log, R = job
    L = OR._log(log)
    G = OR.Path(L["P"], L["T"])
    T, S = G.TF * 1e-6, G.SF
    out = []
    for i, t_us, lead in R:
        lab = NF.stop_label(t_us * 1e-6, float(G.s_log(t_us)) + lead, T, S)
        out.append((i, lab["h4c"] if lab["path_ok"] else np.nan))
    return out


def cmd_gap(a):
    tab = np.load(CR / a.pool / "tab.npz")
    names = tab["names"].astype(str)
    S = tab["mode"].astype(str) == "self"
    part = np.array([n.split(":")[1] for n in names])
    jobs = defaultdict(list)
    for p, d in ROWS.items():
        rows = np.flatnonzero(S & (part == p))
        if not len(rows):
            continue
        zz = np.load(DATA / "runs/onpol_nuplan/rows" / d / "meta.npz", allow_pickle=True)
        z = {k: zz[k] for k in ("names", "speed", "log", "t_log_us", "lead")}
        pos = {n: i for i, n in enumerate(z["names"].astype(str).tolist())}
        mi = np.array([pos["on:" + names[i].split(":", 2)[2]] for i in rows])
        assert np.allclose(z["speed"][mi], tab["speed"][rows], atol=1e-3), f"{d}: meta speed != the pool's"
        for i, m in zip(rows, mi):
            jobs[str(z["log"][m])].append((int(i), int(z["t_log_us"][m]), float(z["lead"][m])))
    assert sum(map(len, jobs.values())) == int(S.sum()), "S rows without a rows dir"
    gap = np.full(len(names), np.nan, np.float32)
    with ProcessPoolExecutor(a.workers) as ex:
        for out in ex.map(_gap_job, list(jobs.items()), chunksize=1):
            for i, g in out:
                gap[i] = g
    f = CR / a.pool / "stand_gap.npz"
    assert not f.exists(), f"{f} exists: never overwritten"
    np.savez(f, names=names, gap=gap)
    # code check: on1_JTX's first rows are on1_JT's; stoppt.py labels' h4c on them (path_ok rows only there too)
    lz = np.load(O / "stoppt/labels.npz")
    ln = lz["names"].astype(str)
    n = min(int((lz["src"].astype(str) == "nu").sum()), len(names))
    same = np.array_equal(ln[:n], names[:n])
    m = np.isfinite(gap[:n]) & np.isfinite(lz["h4c"][:n].astype(np.float64))
    dmax = float(np.abs(gap[:n][m] - lz["h4c"][:n][m]).max()) if m.any() else np.nan
    hum4 = NF.cum_arc(tab["fut_reg"])[:, -1]
    cap, fin = NF.stand_cap(tab["fut"], tab["fut_reg"])
    NF.configure_stand(1.0, V_ROLL)
    st = S & fin & NF.standing(tab["speed"], hum4)
    roll = S & fin & NF.rolling(tab["speed"], hum4, gap) & ~st
    rq = lambda x: np.nanquantile(x, [0.1, 0.25, 0.5, 0.75, 0.9]).round(2).tolist() if len(x) else []  # noqa: E731
    rec = dict(pool=a.pool, rows=len(names), s_rows=int(S.sum()), s_gap_finite=int(np.isfinite(gap[S]).sum()), standing=int(st.sum()),
               rolling=int(roll.sum()), rolling_gap_q=rq(gap[roll]), rolling_gap_neg=int((roll & (gap < 0)).sum()),
               rolling_behind_beyond_b8=int((roll & (gap + NF.MARGIN > cap[:, -1])).sum()),
               rolling_by_part=dict(Counter(part[roll].tolist())), code_check=dict(rows=n, names_equal=bool(same), compared=int(m.sum()), max_abs_diff_m=dmax))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "gap.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))
    assert same and m.sum() > 1000 and dmax < 1e-3, "gap != stoppt labels' h4c on on1_JT's rows"


# ---------------------------------------------------------------- per-decision caps on stored arena reads
def track_arcs(tr, t0, s_e, v0):
    """-> fut arc (8,), fut_reg arc (8,), g, ok for one decision (t0 us on the track's clock)."""
    T, S = tr["T"], tr["S"]
    if t0 + 4.05e6 > T[-1] or t0 < T[0]:
        return None
    sh = lambda t: np.interp(t, T, S)  # noqa: E731
    j = 0.5e6 * np.arange(1, 9)
    reg = sh(t0 + j) - sh(t0)
    tau = np.arange(41) * 0.1
    vl = (sh(t0 + (tau + 0.05) * 1e6) - sh(t0 + (tau - 0.05) * 1e6)) / 0.1
    v = np.maximum(v0 + (vl - v0) * np.minimum(tau / 2.0, 1.0), 0.0)
    fut = np.r_[0.0, np.cumsum((v[1:] + v[:-1]) * 0.05)][5::5]
    return fut, reg, float(sh(t0 + 4e6) - s_e)


def decision_caps(D, hum):
    """D: native_gap_report.load's decisions -> arrays: kind (0 none, 1 standing, 2 rolling), cap (n, 8), fire (n,) on the stored plans."""
    import torch
    import native_gap as NGX
    trs = {}
    n = len(D)
    kind, cap = np.zeros(n, np.int8), np.zeros((n, 8), np.float32)
    for r, (i, t0, se, v) in enumerate(zip(D.id.values, D.t0.values, D.s_e.values, D.v.values)):
        if i not in trs:
            h = hum.get(i)
            trs[i] = None if h is None else dict(T=np.asarray(h["T"], float), S=NGX.Track(h["T"], h["P"], h["Y"]).S)
        if trs[i] is None:
            continue
        x = track_arcs(trs[i], float(t0), float(se), float(v))
        if x is None:
            continue
        fut, reg, g = x
        if reg[-1] >= NF.H_STAND:
            continue
        b = np.maximum(fut, reg) + NF.MARGIN
        if v < NF.V_STAND:
            kind[r], cap[r] = 1, b
        elif v < V_ROLL and np.isfinite(g):
            kind[r], cap[r] = 2, NF.roll_cap(b[None], [g])[0]
    P = torch.as_tensor(np.stack(D.poses.map(lambda q: np.asarray(q, np.float32).reshape(8, 3)).values))
    _, pr = NF.stand_terms(P, torch.as_tensor(cap), torch.as_tensor(kind > 0), parts=True)
    return kind, cap, pr["fire"].numpy()


def cmd_gate(a):
    import pandas as pd
    import torch
    import native_gap_report as NR
    NF.configure_stand(1.0, V_ROLL)
    # G1: the human's own paths on the pool's rows
    tab = np.load(CR / "on1_JTX/tab.npz")
    pk = NF.stand_load(CR / "on1_JTX", tab["names"], "cpu")
    ok = pk["ok"].numpy()
    g1 = {}
    for k in ("fut", "fut_reg"):
        P = torch.as_tensor(np.nan_to_num(tab[k][ok]).astype(np.float32))
        _, pr = NF.stand_terms(P, pk["cap"][torch.as_tensor(ok)], torch.ones(int(ok.sum()), dtype=torch.bool), parts=True)
        g1[k] = int(pr["fire"].sum())
    G = dict(pool=pk["rec"], G1=dict(fires=g1, rows=int(ok.sum()), rolling=int(pk["roll"].sum()), pass_=sum(g1.values()) == 0))
    # G2 / G3 on B2's stored reads
    mis = {"STANDFMXH-L18-s0": "misses_STANDFMXH-s0", "STANDFMXH-L18-s1": "misses_STANDFMXH-s1"}
    src = {"STANDFMXH-L18-s0": O / "stand_b/ext_full_H", "STANDFMXH-L18-s1": O / "stand_b/ext_full_H_s1"}
    red_n = red_f = ok_n = ok_r = 0
    G["per_read"] = {}
    for tag, d in src.items():
        D, W, _, C = NR.load(d)
        D = D[(D.tag == tag) & D.valid & (D.k >= 2)].reset_index(drop=True)
        hum = pickle.load(open(d / "hum.pkl", "rb"))
        kind, cap, fire = decision_caps(D, hum)
        M = pd.read_csv(REPO / f"experiments/alpasim_proxy/results/easy_items/{mis[tag]}.csv.gz")
        red = set(M.id[(M.cause == RED) & (M.strat == EASY)])
        y = W[W.tag == tag].drop_duplicates("id").set_index("id").score
        one = set(y.index[y >= 1 - 1e-4])
        mr, m1 = D.id.isin(red).values, D.id.isin(one).values
        st = C.reindex(D.id[m1 & (kind == 2) & fire]).get("stratum")
        r = dict(red_windows=len(red), red_rolling=int((mr & (kind == 2)).sum()), red_rolling_fire=int((mr & (kind == 2) & fire).sum()),
                 red_standing=int((mr & (kind == 1)).sum()), red_standing_fire=int((mr & (kind == 1) & fire).sum()),
                 red_windows_with_rolling=int(D.id[mr & (kind == 2)].nunique()),
                 one_windows=len(one), one_decisions=int(m1.sum()), one_rolling=int((m1 & (kind == 2)).sum()),
                 one_rolling_fire=int((m1 & (kind == 2) & fire).sum()), one_rolling_fire_windows=int(D.id[m1 & (kind == 2) & fire].nunique()),
                 one_standing_fire=int((m1 & (kind == 1) & fire).sum()),
                 one_rolling_fire_by_stratum={} if st is None else st.value_counts().to_dict())
        G["per_read"][tag] = r
        red_n, red_f, ok_n, ok_r = red_n + r["red_rolling"], red_f + r["red_rolling_fire"], ok_n + r["one_decisions"], ok_r + r["one_rolling"]
    G["G2"] = dict(decisions=red_n, fired=red_f, share=red_f / max(red_n, 1), line=0.80, pass_=red_n > 0 and red_f / red_n >= 0.80)
    G["G3"] = dict(decisions=ok_n, rolling=ok_r, share=ok_r / max(ok_n, 1), line=0.01, pass_=ok_r / max(ok_n, 1) <= 0.01)
    G["pass_"] = bool(G["G1"]["pass_"] and G["G2"]["pass_"] and G["G3"]["pass_"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in (out / "gate.json", RES / "gate.json"):
        f.write_text(json.dumps(G, indent=1, default=float))
    print(json.dumps(G, indent=1, default=float))
    print("GATE", "PASS" if G["pass_"] else "FAIL")
    raise SystemExit(0 if G["pass_"] else 2)


def cmd_smoke(a):
    d = sorted(glob.glob(f"{DATA}/runs/route_mem/train-{a.run}/*/"))[-1]
    sc = {}
    for line in open(d + "events.jsonl"):
        e = json.loads(line)
        if e["kind"] == "scalar":
            sc.setdefault(e["tag"], {})[e["step"]] = e["value"]
    fin = all(np.isfinite(v) for t, x in sc.items() if t.startswith("loss/") and "grad/" not in t for v in x.values())
    k = "loss/on/lon_standr_pos"
    rp = sc.get(k, {})
    lab = [ln for ln in open(a.log) if "lon_stand labels" in ln]
    r = dict(run=a.run, finite=bool(fin), steps=len(rp), roll_pos_max=max(rp.values()) if rp else None, roll_pos_steps_gt0=sum(v > 0 for v in rp.values()),
             stand_pos_mean=float(np.mean(list(sc.get("loss/on/lon_stand_pos", {0: np.nan}).values()))), labels=lab[-1].strip() if lab else "")
    r["pass_"] = bool(fin and rp and r["roll_pos_max"] > 0)
    Path(a.out).write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))
    raise SystemExit(0 if r["pass_"] else 2)


# ---------------------------------------------------------------- Mac: easy-miss classes on a (sub)set of held windows
def _ctx():
    import pandas as pd
    import pcs_local as PL
    import native_gap_report as NG
    res = REPO / "experiments/alpasim_proxy/results"
    sc, ty, w = PL.load(res / "pcs_local")
    it = pd.read_csv(res / "pcs_local/items_boardmix.csv.gz", index_col=0)
    S = pd.read_csv(REPO / "tmp/native_gap/set.csv.gz").drop_duplicates("id").set_index("id")
    Cn = pd.read_parquet(REPO / "tmp/native_gap/set_census.parquet").drop_duplicates("id").set_index("id")
    items = list(it.index)
    cue = NG.cue(S.loc[items], Cn)
    o = pd.read_parquet(REPO / "tmp/native_gap/offroad_stop3.parquet")
    hoff = set(o.index[o.h_off_any]) & set(items)
    z = PL.Scored(dict(np.load(res / "pcs_local/boardmix.npz", allow_pickle=True)), sc)
    G = sc.loc["garage", items]
    return dict(items=items, S=S, cue=cue, hoff=hoff, z=z, easy=set(G.index[G >= 1 - 1e-4]))


def classes(cx, R, tag, dec, keep):
    """-> dict: subset PCS, mean, easy-miss counts by class over the items in keep, for one tag of the windows table R."""
    import pandas as pd
    import easy_items as EI
    import native_gap_report as NG
    D, _, St, _ = NG.load(Path(dec))
    Dv = D[D.valid & (D.k >= 2) & (D.tag == tag)].sort_values("k")
    rear = St[St.tag == tag].groupby("id").collision_rear.max() > 0 if "collision_rear" in St else None
    g = R[R.tag == tag].drop_duplicates("id").set_index("id")
    ids = [i for i in cx["items"] if i in keep]
    y, typ = g.score.reindex(ids), g.type.reindex(ids)
    C = EI.classify(y, typ, ids, cx["S"], cx["cue"], cx["hoff"], {i: q for i, q in Dv.groupby("id")}, rear)
    E = C[C.id.isin(cx["easy"]) & ~C.cause.str.startswith("unsolvable")] if len(C) else C
    yy = g.score.reindex(cx["z"].items).to_numpy(float)
    yy = np.where(np.isin(cx["z"].items, ids), yy, np.nan)
    cnt = E.cause.value_counts().to_dict() if len(E) else {}
    return dict(windows=len(ids), mean=float(y.mean()), zeros=int((y <= 1e-4).sum()), pcs=cx["z"].pcs_of(yy), easy_windows=len(set(ids) & cx["easy"]),
                easy_miss=int(len(E)), red=int(cnt.get(RED, 0)), other=int(len(E) - cnt.get(RED, 0)), by_class=cnt,
                red_ids=sorted(E.id[E.cause == RED]) if len(E) else [])


def cmd_easy(a):
    import pandas as pd
    cx = _ctx()
    R = pd.read_csv(a.rows)
    out = {}
    for t in a.tag:
        keep = set(R[(R.tag == t) & R.score.notna()].id)
        out[t] = classes(cx, R, t, a.dec, keep)
    (RES / f"{a.out}.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({t: {k: v for k, v in r.items() if k != "red_ids"} for t, r in out.items()}, indent=1, default=float))


def cmd_stage1(a):
    import pandas as pd
    cx = _ctx()
    R = pd.read_csv(a.rows)
    tags = [CTL] if a.anchors else [CTL, *ARMS.values()]
    keep = set.intersection(*(set(R[(R.tag == t) & R.score.notna()].id) for t in tags))
    E = {t: classes(cx, R, t, a.dec, keep) for t in tags}
    J = dict(subset_windows=len(keep & set(cx["items"])), easy=E)
    if a.anchors:
        print(json.dumps(J, indent=1, default=float))
        return
    c = E[CTL]
    J["arms"] = {}
    L = ["# STAND-R pilot stage 1 (NR1 prereg supplement 20; native reads; control = the stored B2 pilot STANDFMXH1K-L18-s0)", "",
         f"Arena subset (chunks 0 / 3 / 6 / 9 / 12 / 15) on the frozen board-mix items every read has: {J['subset_windows']} windows.", ""]
    for A, T in ARMS.items():
        t = json.load(open(RES / f"s17table_{A}.json"))
        H, Wt, Ar, Dr = t["held"], t["wait"], t["arena"], next(iter(t["drift"].values()))
        e = E[T]
        cls = sorted(set(c["by_class"]) | set(e["by_class"]) - {RED})
        rows = [("main 1: easy red-line zeros", c["red"], e["red"], e["red"] <= c["red"] - 6, "<= control - 6"),
                ("main 2: subset PCS (frozen board-mix)", round(c["pcs"]), round(e["pcs"]), e["pcs"] >= c["pcs"] + 15, ">= control + 15"),
                ("guard 1: held 460 zeros", H[CTL]["zeros"], H[T]["zeros"], H[T]["zeros"] <= H[CTL]["zeros"] + 2, "<= control + 2"),
                ("guard 2: held 460 front collisions", H[CTL]["coll"]["front"], H[T]["coll"]["front"], H[T]["coll"]["front"] <= H[CTL]["coll"]["front"] + 2, "<= control + 2"),
                ("guard 2: subset AF-front windows", Ar["af_front_b"], Ar["af_front_a"], Ar["af_front_a"] <= Ar["af_front_b"] + 2, "<= control + 2"),
                ("guard 3: easy misses outside the red line", c["other"], e["other"], e["other"] <= c["other"] + 2, "<= control + 2"),
                *[(f"guard 4: easy {k}", c["by_class"].get(k, 0), e["by_class"].get(k, 0), e["by_class"].get(k, 0) <= c["by_class"].get(k, 0) + 2, "<= control + 2")
                  for k in cls if k != RED],
                ("guard 5: subset slow partial windows", t["partials"][CTL]["slow"], t["partials"][T]["slow"], t["partials"][T]["slow"] <= t["partials"][CTL]["slow"] + 3, "<= control + 3"),
                ("guard 5: subset k = 9 speed ratio", round(Ar["k9"]["b"], 4), round(Ar["k9"]["a"], 4), Ar["k9"]["a"] >= Ar["k9"]["b"] - 0.03, ">= control - 0.03"),
                *[(f"guard 6: wait a {k}", round(Wt[CTL][k]["wait"], 3), round(Wt[T][k]["wait"], 3), Wt[T][k]["wait"] >= Wt[CTL][k]["wait"] - 0.03, ">= control - 0.03") for k in Wt[T]],
                *[(f"guard 6: wait b median launch delay {k} (s)", round(Wt[CTL][k]["delay_med"], 2), round(Wt[T][k]["delay_med"], 2),
                   Wt[T][k]["delay_med"] <= Wt[CTL][k]["delay_med"] + 0.15, "<= control + 0.15") for k in Wt[T]],
                *[(f"guard 6: wait b launched {k}", round(Wt[CTL][k]["launched"], 3), round(Wt[T][k]["launched"], 3),
                   Wt[T][k]["launched"] >= 0.95 * Wt[CTL][k]["launched"], ">= 0.95 x control") for k in Wt[T]],
                ("guard 7: held 460 right drift, non-turn k 11-17 (arm - control, m)", "", round(Dr["k11_17"]["mean"], 3), Dr["k11_17"]["mean"] > -0.05, "> -0.05")]
        ok = all(r[3] for r in rows)
        J["arms"][A] = dict(tag=T, rows=[dict(read=r[0], control=r[1], arm=r[2], ok=bool(r[3]), line=r[4]) for r in rows], pass_=ok,
                            subset_mean=[c["mean"], e["mean"]], held_mean=[H[CTL]["mean"], H[T]["mean"]],
                            red_fixed=sorted(set(c["red_ids"]) - set(e["red_ids"])), red_new=sorted(set(e["red_ids"]) - set(c["red_ids"])))
        L += [f"## {A} `{T}`: {'PASS' if ok else 'FAIL'}", "", f"| read | {CTL} | {T} | line |", "|:--|--:|--:|:--|",
              *[f"| {r[0]} | {r[1]} | {r[2]} | {r[4]}: {'ok' if r[3] else 'FAIL'} |" for r in rows], "",
              f"- subset mean {c['mean']:.4f} -> {e['mean']:.4f}, zeros {c['zeros']} -> {e['zeros']}; easy misses {c['easy_miss']} -> {e['easy_miss']} "
              f"(of {e['easy_windows']} easy windows); held 460 mean {H[CTL]['mean']:.4f} -> {H[T]['mean']:.4f}",
              f"- red-line easy zeros fixed {len(J['arms'][A]['red_fixed'])}, new {len(J['arms'][A]['red_new'])}", ""]
    (RES / "stage1.json").write_text(json.dumps(J, indent=1, default=float))
    (RES / "stage1.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("gap")
    p.add_argument("--pool", required=True), p.add_argument("--workers", type=int, default=32)
    p = sp.add_parser("gate")
    p.add_argument("--out", required=True)
    p = sp.add_parser("smoke")
    p.add_argument("--run", required=True), p.add_argument("--log", required=True), p.add_argument("--out", required=True)
    p = sp.add_parser("stage1")
    p.add_argument("--rows", default=str(RES / "arena_windows.csv.gz")), p.add_argument("--dec", default=str(REPO / "tmp/stand_r/ext"))
    p.add_argument("--anchors", action="store_true", help="the control's numbers only (from the stored B2 pilot read)")
    p = sp.add_parser("easy")
    p.add_argument("--rows", required=True), p.add_argument("--tag", nargs="+", required=True), p.add_argument("--dec", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    dict(gap=cmd_gap, gate=cmd_gate, smoke=cmd_smoke, stage1=cmd_stage1, easy=cmd_easy)[a.cmd](a)
