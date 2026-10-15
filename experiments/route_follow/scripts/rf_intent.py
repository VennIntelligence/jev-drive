#!/usr/bin/env python3
"""INTENT HOLD serve-side diagnostic (plans/2026-10-15-intent-hold-prereg.md; experiments/alpasim/lib/intent_hold.py). Stored run roots are
read, never written. Runner: rf_intent.sh.

  replay  (Tokyo, python3)  --ref <identity root> --scenes <file> [--a 0.3]   open loop: intent_hold.step on every stored decision of the
          identity (its anchor, plan, fed speed, route memory, command): how often it would hold, why it resets, correction size, clips
  ihcheck (Tokyo, python3)  --run <root> --ref <identity root>   the switch as logged: every decision's branch recomputed with intent_hold.step
          from the logged anchor / plan / ih.v / rmem / cmd and the intent carried along the logged branches (4-decimal logs: near-threshold
          branch flips are listed and do not desync the state), served = poses_ih on every hold (max |diff|), and every plan up to a session's first on-decision equal to the identity's at the same k
  extract (Tokyo, python3)  --set NAME=<follow pkl>:<label>:<root> ... --scenes <file> --out <pkl>   rf_receding.extract_pai per set, and
          per decision the served plan (poses_ih where the hold was on, else poses) and the logged `ih` record
  creep   (Mac, .venv)  --pkl <extract> --out DIR   rf_switching's per-turn measure (window entry -/+ 20 m before the first zero; j = the
          2 s point of each plan vs the previous plan's path, per metre of ego travel; summed j; executed - planned heading at 2 s) on the
          model plan (`poses`, as decision 308) and on the served plan -> DIR/creep.{md,csv,json}
  lines   (Mac, .venv)  --turns <rf_v2pai turns.json> --counts <table counts.json> --spins <spins.json> --creep <creep.json> --a <identity
          label> --b <label> --out <md>   the registered lines
  fig     (Mac, .venv)  --pkl <extract> --sets <ident>,<ih> --scenes s1,s2,s3 --png <png>   BEV of three turns: the intent carried from
          the previous decision, the served plan and the executed track, against the identity's track
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import pickle
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
R = HERE.parents[2]
sys.path[:0] = [str(HERE), str(R / "experiments/alpasim/lib")]
RES = HERE.parent / "results"
import intent_hold as IH  # noqa: E402


def sessions(root):
    """{session: (scene, [inference records sorted by k])} of every driver log under a pai_eval root."""
    out = {}
    for f in glob.glob(os.path.join(root, "runs", "*", "driver", "drive.jsonl")) + glob.glob(os.path.join(root, "driver", "drive.jsonl")):
        for line in open(f):
            if '"infer": true' not in line:
                continue
            r = json.loads(line)
            out.setdefault(r["session"], [r.get("scene", ""), []])[1].append(r)
    for v in out.values():
        v[1].sort(key=lambda r: r["k"])
    return out


def speed(r):
    if "ih" in r:
        return float(r["ih"]["v"])
    if r.get("fix"):
        return float(r["fix"]["v0"])
    return float(10 * np.hypot(r["ego"][4], r["ego"][5]))


def replay(a):
    want = {x.split()[0] for x in open(a.scenes) if x.strip()}
    tot = {}
    cm, e0 = [], []
    for se, (sc, recs) in sorted(sessions(a.ref).items(), key=lambda x: x[1][0]):
        if sc not in want:
            continue
        st, why, on = IH.new(), {}, 0
        for r in recs:
            _, q = IH.step(st, r["anchor"], r["poses"], speed(r), r.get("rmem"), r["cmd"], a=a.a)
            why[q["why"] or "hold"] = why.get(q["why"] or "hold", 0) + 1
            if q["on"]:
                on += 1
                cm.append(q["cmax"])
                e0.append(abs(q["e0"]))
        for k, v in why.items():
            tot[k] = tot.get(k, 0) + v
        print(f"{sc[7:15]}: {len(recs)} decisions, hold {on}; {why}")
    cm, e0 = np.array(cm), np.array(e0)
    print(f"replay a {a.a}: {tot}; hold decisions {len(cm)}: |served - plan| max p50 {np.median(cm):.3f} p90 {np.quantile(cm, 0.9):.3f} max {cm.max():.3f} m; "
          f"|e0| p50 {np.median(e0):.3f} p90 {np.quantile(e0, 0.9):.3f} m; at the clip {int((cm >= IH.CLIP_M - 1e-6).sum())}")


def follow_logged(st, r):
    """Advance the intent state along the branch the driver logged (hold or reset); -> the served poses."""
    if not r["ih"]["on"]:
        st["I"] = IH.dense(IH.world(np.asarray(r["anchor"], float), r["poses"]))
        return np.asarray(r["poses"], float)
    far, IH.FAR_M = IH.FAR_M, np.inf
    try:
        srv, _ = IH.step(st, r["anchor"], r["poses"], 99.0, None, IH.LEFT, a=r["ih"]["a"])
    finally:
        IH.FAR_M = far
    return srv


def ihcheck(a):
    ref = {sc: {r["k"]: r["poses"] for r in recs} for sc, recs in sessions(a.ref).values()}
    n = flips = 0
    dmax = pre_max = 0.0
    pre_n = 0
    for se, (sc, recs) in sorted(sessions(a.run).items(), key=lambda x: x[1][0]):
        recs = [r for r in recs if "ih" in r]
        if not recs:
            continue
        k_on = next((r["k"] for r in recs if r["ih"]["on"]), None)
        st = IH.new()
        for r in recs:
            n += 1
            _, q = IH.step({"I": st["I"]}, r["anchor"], r["poses"], r["ih"]["v"], r.get("rmem"), r["cmd"], a=r["ih"]["a"])   # the branch
            if q["on"] != r["ih"]["on"] or q["why"] != r["ih"]["why"]:
                flips += 1
                print(f"  flip {sc[7:15]} k {r['k']}: logged {r['ih']}, recomputed {q}")
            srv = follow_logged(st, r)                                 # the state always follows the logged branch (a flip does not desync it)
            if r["ih"]["on"]:
                dmax = max(dmax, float(np.abs(np.asarray(srv) - np.array(r["poses_ih"])).max()))
            if (k_on is None or r["k"] <= k_on) and r["k"] in ref.get(sc, {}):
                pre_n += 1
                pre_max = max(pre_max, float(np.abs(np.array(r["poses"]) - np.array(ref[sc][r["k"]])).max()))
        print(f"{sc[7:15]}: {len(recs)} decisions, hold at {sum(r['ih']['on'] for r in recs)}, first k {k_on}, "
              f"why {dict((w, sum(r['ih']['why'] == w for r in recs)) for w in ('slow', 'short', 'noturn', 'empty', 'far'))}")
    print(f"ihcheck: {n} decisions, branch recomputed differs at {flips}; served vs recomputed max |diff| {dmax:.1e} (4-decimal log); "
          f"plans up to the first hold vs identity: {pre_n} compared, max |diff| {pre_max:.1e}")


def extract(a):
    import rf_receding as RC
    out = []
    for spec in a.set:
        tmp = a.out + ".part"
        RC.extract_pai(argparse.Namespace(set=[spec], strict=a.scenes, out=tmp, zeros="", zmap=None))
        recs = pickle.load(open(tmp, "rb"))
        os.remove(tmp)
        root = spec.split("=", 1)[1].split(":")[2]
        rids = {r["rid"] for r in recs}
        srv = {}
        for f in glob.glob(os.path.join(root, "**", "driver", "drive.jsonl"), recursive=True):
            for line in open(f):
                if '"infer": true' not in line:
                    continue
                r = json.loads(line)
                if r.get("session") in rids:
                    srv.setdefault(r["session"], {})[(int(r["t0"]), int(r["k"]))] = (r.get("poses_ih") or r["poses"], r.get("ih"))
        for r in recs:
            P = r["plans"]
            if P is None:
                continue
            m = srv.get(r["rid"], {})
            keys = list(zip(P["t0"].tolist(), P["k"].tolist()))
            P["poses_served"] = np.array([m[k][0] if k in m else P["poses"][i] for i, k in enumerate(keys)], float)
            P["ih"] = [m[k][1] if k in m else None for k in keys]
        out += recs
    pickle.dump(out, open(a.out, "wb"), protocol=4)
    print(len(out), "->", a.out)


# ---------------------------------------------------------------- Mac: creep, lines, figure
def creep(a):
    import pandas as pd
    import rf_switching as SW
    recs = pickle.load(open(a.pkl, "rb"))
    rows = []
    for rec in recs:
        if rec.get("plans") is None:
            continue
        for kind in ("model", "served"):
            r2 = copy.copy(rec)
            r2["plans"] = dict(rec["plans"])
            if kind == "served":
                r2["plans"]["poses"] = rec["plans"]["poses_served"]
            tr, dec = SW.turn_switching(r2, None)
            if tr is None:
                continue
            ih = [x for x in rec["plans"].get("ih", []) if x]
            rows.append(dict(set=rec["set"], scene=rec["scene"], kind=kind, completed=bool(tr["completed"]), cls=tr["cls"], side=tr["side"],
                             jpm=tr["jpm"], jsum=tr["jsum"], n_jpm=tr["n_jpm"], x2=tr["x2"], abs_e=tr["abs_e"], n_win=tr["n_win"],
                             hold=sum(bool(x["on"]) for x in ih), n_ih=len(ih)))
    T = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    T.to_csv(os.path.join(a.out, "creep.csv"), index=False)
    L = ["# INTENT HOLD: plan creep (rf_switching's measure)", "",
         "Per turn: j per metre of ego travel (median over consecutive decision pairs in entry -/+ 20 m before the first zero; - = each new plan's "
         "2 s point lies outside the previous plan's path), summed j (m), executed - planned heading at 2 s (deg). `model` = the policy's plan "
         "(`poses`, decision 308's measure), `served` = the plan the controller got (the hold's output; = model where the hold was off).", "",
         "| set | plans | turns c / f | j per m: completed | failed | all | summed j c / f (m) | exec - plan 2 s c / f (deg) |",
         "|:--|:--|:--|--:|--:|--:|:--|:--|"]
    S = {}
    for (st, kind), g in T.groupby(["set", "kind"], sort=False):
        c, f = g[g.completed], g[~g.completed]
        md = lambda x: float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan  # noqa: E731
        S[f"{st}|{kind}"] = dict(n_c=len(c), n_f=len(f), jpm_c=md(c.jpm), jpm_f=md(f.jpm), jpm_all=md(g.jpm), jsum_c=md(c.jsum), jsum_f=md(f.jsum),
                                 x2_c=md(c.x2), x2_f=md(f.x2))
        s = S[f"{st}|{kind}"]
        L.append(f"| {st} | {kind} | {len(c)} / {len(f)} | {s['jpm_c']:+.3f} | {s['jpm_f']:+.3f} | {s['jpm_all']:+.3f} | {s['jsum_c']:+.2f} / {s['jsum_f']:+.2f} | "
                 f"{s['x2_c']:+.1f} / {s['x2_f']:+.1f} |")
    L += ["", "## Per turn", "", "| scene | " + " | ".join(f"{st} {k}: done, cls, j/m, summed" for st, k in T.groupby(["set", "kind"], sort=False).groups) + " |",
          "|:--|" + "--|" * len(T.groupby(["set", "kind"]).groups)]
    for sc, g in T.groupby("scene", sort=False):
        cells = []
        for (st, k) in T.groupby(["set", "kind"], sort=False).groups:
            x = g[(g.set == st) & (g.kind == k)]
            cells.append("" if x.empty else f"{'C' if x.completed.iloc[0] else 'f'} {x.cls.iloc[0]} {x.jpm.iloc[0]:+.3f} {x.jsum.iloc[0]:+.2f}")
        L.append(f"| {sc[7:15]} | " + " | ".join(cells) + " |")
    Path(a.out, "creep.md").write_text("\n".join(L) + "\n")
    Path(a.out, "creep.json").write_text(json.dumps(S, indent=1, default=float))
    print("\n".join(L[:12 + len(S)]))


def lines(a):
    T = json.loads(Path(a.turns).read_text())
    C = json.loads(Path(a.counts).read_text())
    SP = json.loads(Path(a.spins).read_text())
    CR = json.loads(Path(a.creep).read_text())
    by = {(r["label"], r["scene"]): r for r in T["rows"]}
    rd = lambda f: [x.split()[0] for x in open(f) if x.strip()]  # noqa: E731
    strict, runway = rd(RES / "pai281/strict36.txt"), rd(RES / "approach/pai_runway22.txt")
    done = lambda lab, L: [s for s in L if by.get((lab, s), {}).get("completed")]  # noqa: E731
    A, B = a.a, a.b
    sa, sb, ra, rb = done(A, strict), done(B, strict), done(A, runway), done(B, runway)
    lost = [s[7:15] for s in sa if s not in sb]
    spin_a, spin_b = SP[A], SP[B]
    k = f"{a.set}|served"
    km = f"{a.set}|model"
    cr = CR.get(k, {})
    crm = CR.get(km, {})
    res = dict(strict=(len(sb), len(sb) >= 10), runway=(len(rb), len(rb) >= 6), keep5=(lost, not lost),
               new_zero_nonturn=(C["new_zero_nonturn"], len(C["new_zero_nonturn"]) <= 2), spins=(spin_b, spin_b <= 2),
               stop=(C["new_offroad_pre"], len(C["new_offroad_pre"]) >= 3),
               creep_served=(cr.get("jpm_c"), bool(np.isfinite(cr.get("jpm_c", np.nan)) and abs(cr["jpm_c"]) <= 0.03)),
               creep_model=(crm.get("jpm_c"), bool(np.isfinite(crm.get("jpm_c", np.nan)) and abs(crm["jpm_c"]) <= 0.03)))
    L = [f"# INTENT HOLD lines: {B} vs {A} (identity)", "",
         f"- strict 36 completed: {A} {len(sa)}, {B} **{len(sb)}** (line >= 10): {'PASS' if res['strict'][1] else 'FAIL'}",
         f"- runway 22 completed: {A} {len(ra)}, {B} **{len(rb)}** (line >= 6): {'PASS' if res['runway'][1] else 'FAIL'}",
         f"- the identity's {len(sa)} completed strict turns kept: lost {len(lost)} ({', '.join(lost)}): {'PASS' if not lost else 'FAIL'}",
         f"- new zeros outside the turns: {len(C['new_zero_nonturn'])} ({', '.join(C['new_zero_nonturn'])}) (line <= 2): {'PASS' if res['new_zero_nonturn'][1] else 'FAIL'}",
         f"- spins (rf_pai281's rule, the run's scenes): {A} {spin_a}, {B} **{spin_b}** (line <= 2): {'PASS' if res['spins'][1] else 'FAIL'}",
         f"- decision 293 shape gate: {A} {C['a_shape']}, {B} {C['b_shape']}",
         f"- stop rule (>= 3 new offroad zeros before the entry): {len(C['new_offroad_pre'])} ({', '.join(C['new_offroad_pre'])}): {'FIRED' if res['stop'][1] else 'not fired'}",
         f"- MECHANISM, creep per metre in the completed turns (served plan; line <= 0.03 m / m): {cr.get('jpm_c', np.nan):+.3f} (n {cr.get('n_c')}): "
         f"{'PASS' if res['creep_served'][1] else 'FAIL'}; failed {cr.get('jpm_f', np.nan):+.3f} (n {cr.get('n_f')}), all {cr.get('jpm_all', np.nan):+.3f}",
         f"- creep per metre on the model plan (decision 308's object): completed {crm.get('jpm_c', np.nan):+.3f}, failed {crm.get('jpm_f', np.nan):+.3f}, "
         f"all {crm.get('jpm_all', np.nan):+.3f}",
         f"- zeros {C['a_zero']} -> {C['b_zero']}; mean {C['mean_a']:.4f} -> {C['mean_b']:.4f} (paired {C['paired']:+.4f})",
         f"- gained strict turns: {', '.join(C['gained'])}"]
    allp = res["strict"][1] and res["runway"][1] and res["keep5"][1] and res["new_zero_nonturn"][1] and res["spins"][1]
    L += ["", f"Verdict: {'all completion lines pass' if allp else 'not all lines pass'}; mechanism {'PASS' if res['creep_served'][1] else 'FAIL'}."]
    Path(a.out).write_text("\n".join(L) + "\n")
    Path(a.out).with_suffix(".json").write_text(json.dumps(res, indent=1, default=float))
    print("\n".join(L))


def fig(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rf_pai_follow as F
    recs = pickle.load(open(a.pkl, "rb"))
    s_id, s_ih = a.sets.split(",")
    by = {(r["set"], r["scene"][7:15]): r for r in recs}
    scenes = a.scenes.split(",")
    fig, ax = plt.subplots(1, len(scenes), figsize=(5.2 * len(scenes), 5.6))
    for axi, sc in zip(np.atleast_1d(ax), scenes):
        rid, rih = by.get((s_id, sc)), by.get((s_ih, sc))
        Rr = rih["R"]
        ent, ext = float(rih["row"]["entry"]), float(rih["row"]["exit"])
        S = F.stations(Rr)
        win = (S >= ent - 45) & (S <= ext + 15)
        axi.plot(Rr[win, 0], Rr[win, 1], color="0.75", lw=6, solid_capstyle="round", zorder=0, label="route")
        for rr, col, lab in ((rid, "#d95f02", f"executed, identity ({'done' if rid['row'].get('completed') else rid['kind'] or 'failed'})"),
                             (rih, "#1b9e77", f"executed, intent hold ({'done' if rih['row'].get('completed') else rih['kind'] or 'failed'})")):
            C = rr["ctrl"]
            t = C["t"]
            st_r, _ = F.track(Rr, S, np.c_[C["x"], C["y"]])
            m = (t <= min(rr["tz"], t[-1])) & (st_r >= ent - 45) & (st_r <= ext + 25)
            axi.plot(C["x"][m], C["y"][m], color=col, lw=2.0, zorder=3, label=lab)
        P = rih["plans"]
        st = IH.new()
        k_last = -10
        n_drawn = 0
        stc, _ = F.track(Rr, S, np.c_[rih["ctrl"]["x"], rih["ctrl"]["y"]])
        for i in range(len(P["t0"])):
            q = P["ih"][i] or {}
            I_prev = None if st["I"] is None else st["I"].copy()
            follow_logged(st, dict(ih=q if q else dict(on=False), anchor=P["anchor"][i], poses=P["poses"][i]))
            t0 = P["t0"][i]
            if t0 >= rih["tz"]:
                break
            s_e = float(np.interp(t0, rih["ctrl"]["t"], stc))
            if not (ent - 30 <= s_e <= ext) or i - k_last < 10 or not q.get("on") or I_prev is None:
                continue
            k_last = i
            A0 = np.asarray(P["anchor"][i])
            Wm, Ws = IH.world(A0, P["poses"][i]), IH.world(A0, P["poses_served"][i])
            kw = dict(zorder=2, alpha=0.9)
            axi.plot(I_prev[:, 0], I_prev[:, 1], ls="--", color="#7570b3", lw=1.2, label="intent carried from the previous decision" if not n_drawn else None, **kw)
            axi.plot(Wm[:, 0], Wm[:, 1], color="0.35", lw=0.9, label="model plan" if not n_drawn else None, **kw)
            axi.plot(Ws[:, 0], Ws[:, 1], color="#e7298a", lw=1.2, label="served plan" if not n_drawn else None, **kw)
            axi.plot(*A0[:2], "o", color="#e7298a", ms=3, zorder=4)
            n_drawn += 1
        lo, hi = Rr[win].min(0) - 12, Rr[win].max(0) + 12
        axi.set_xlim(lo[0], hi[0]), axi.set_ylim(lo[1], hi[1])
        axi.set_aspect("equal")
        axi.set_title(f"{sc} ({rih['row']['side']} {rih['row']['turn_deg']:+.0f} deg); every 1 s while holding", fontsize=9)
        axi.tick_params(labelsize=7)
        axi.legend(fontsize=6.5, loc="best")
    fig.tight_layout()
    fig.savefig(a.png, dpi=150)
    print("->", a.png)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("replay")
    s.add_argument("--ref", required=True), s.add_argument("--scenes", required=True), s.add_argument("--a", type=float, default=IH.A)
    s = sp.add_parser("ihcheck")
    s.add_argument("--run", required=True), s.add_argument("--ref", required=True)
    s = sp.add_parser("extract")
    s.add_argument("--set", nargs="+", required=True), s.add_argument("--scenes", required=True), s.add_argument("--out", required=True)
    s = sp.add_parser("creep")
    s.add_argument("--pkl", required=True), s.add_argument("--out", required=True)
    s = sp.add_parser("lines")
    for k in ("turns", "counts", "spins", "creep", "a", "b", "set", "out"):
        s.add_argument(f"--{k}", required=True)
    s = sp.add_parser("fig")
    s.add_argument("--pkl", required=True), s.add_argument("--sets", required=True), s.add_argument("--scenes", required=True), s.add_argument("--png", required=True)
    a = ap.parse_args()
    {"replay": replay, "ihcheck": ihcheck, "extract": extract, "creep": creep, "lines": lines, "fig": fig}[a.cmd](a)


if __name__ == "__main__":
    main()
