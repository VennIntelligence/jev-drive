"""S-DROP (plans/2026-10-10-sdrop-prereg.md): checks and the collected read of the full-scale drop-one ablation of P2H10S. CPU, no model.

  smoke    prereg 3.3: every arm's 60-step run finished and logged the arm's own terms                -> results/sdrop/smoke.json
  trained  prereg 3.2 + 3.4: config identity of the six full runs against the stored P2H10S-F-s<k>, training sanity -> config_identity.json
  report   prereg 4 + 5: two-seed paired differences (arm - base, arm - P2H10S; logs resampled) from the per-state dumps of prog_ol.py,
           the bench unit files and bd4_g3d.py's tables; the lines of question 1 and 2; the attribution table -> summary.{md,json}, *.csv
Inputs of `report`: $DATA_DIR/runs/body1/prog/ol_sdrop_{navtest,hold}.parquet (+ _plans.npy), results/sdrop/g3_sdrop_*.json (cross-check),
results/sdrop/d_<arm>_vs_{base,S}.csv, jevdrive.bench unit files of navtest and navhard (@gimm).

  $DATA_DIR/envs/op-train/bin/python experiments/body1/scripts/sdrop_report.py report
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "lib"))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import argparse  # noqa: E402
import json  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

import b1 as B  # noqa: E402

B.FAMS["bd4"] = "bd4_"
OUT = B.REPO / "experiments/body1/results/sdrop"
ARMS, SEEDS = ("noA", "noB", "noC"), (0, 1)
DROPPED = {"noA": "A (agent hinge)", "noB": "B (hinge-only rows; C goes with them)", "noC": "C (road hinge on hinge-only rows)"}
tag = lambda arm, s: {"base": f"P2H10-F-s{s}", "S": f"P2H10S-F-s{s}"}.get(arm, f"P2H10S-{arm}-F-s{s}")  # noqa: E731
ALLOWED = {"noA": {"agent_lam", "body1.agent_lam"}, "noB": {"data", "body1.ho", "body1.ho_w", "body1.ho_excl"}, "noC": {"body1.road_lam"}}
W2_TOL = 5


def run_dir(t):
    d = sorted((B.data_dir() / "runs/op_parity" / f"train-{t}").glob("*"))
    assert d, f"no run dir of {t}"
    return d[-1]


def scalars(d):
    return {json.loads(ln)["tag"] for ln in open(d / "events.jsonl") if '"scalar"' in ln}


def cmd_smoke(_):
    res, ok = {}, True
    for arm in ARMS:
        d = run_dir(f"SDROP-SMOKE-{arm}")
        s = scalars(d)
        has = lambda q: f"loss/{q}" in s  # noqa: E731
        want = {"noA": not any(q.startswith("loss/agent") for q in s) and has("road_ho"), "noB": has("agent") and not any(q.endswith("_ho") for q in s),
                "noC": has("agent") and has("agent_ho") and not has("road_ho")}[arm]
        res[arm] = dict(run=str(d), done=(d / "DONE").exists(), terms_as_registered=bool(want), loss_scalars=sorted(q for q in s if q.startswith("loss/") and q.count("/") == 1))
        ok &= res[arm]["done"] and want
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "smoke.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    raise SystemExit(0 if ok else 2)


def flat(c):
    return {k: v for k, v in c.items() if k != "body1"} | {f"body1.{k}": v for k, v in c["body1"].items()}


def cmd_trained(_):
    res, ok = {}, True
    for arm in ARMS:
        for s in SEEDS:
            d, r = run_dir(tag(arm, s)), run_dir(tag("S", s))
            a, b = (flat(json.loads((x / "meta.json").read_text())["config"]) for x in (d, r))
            diff = {k: [a.get(k, "<absent>"), b.get(k, "<absent>")] for k in sorted(set(a) | set(b)) if a.get(k, "<absent>") != b.get(k, "<absent>")}
            off = {k for k in diff if k not in b and (k, a[k]) in (("body1.route_band", 0.0), ("body1.route_lam", 10.0))}   # switches added after the stored run, off
            bad = sorted(set(diff) - ALLOWED[arm] - {"body1.tag"} - off)
            done = json.loads((d / "DONE").read_text())
            sane = done["steps"] == 10000 and done["dev_ade"] <= 0.56 and done["dev_drift_off"] <= 0.30 and done["ho_per_batch"] == (0 if arm == "noB" else 13)
            res[tag(arm, s)] = dict(run=str(d), ref=str(r), differing_keys={k: v for k, v in diff.items() if k != "data"} | ({"data": "hinge-only cache dirs absent"} if "data" in diff else {}),
                                    unexpected=bad, added_since_off=sorted(off), **{k: done[k] for k in ("steps", "dev_ade", "dev_drift_off", "ho_per_batch", "train_s", "gpu_peak_gb")},
                                    ref_dev_ade=json.loads((r / "DONE").read_text())["dev_ade"], sane=bool(sane))
            ok &= sane and not bad
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "config_identity.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    raise SystemExit(0 if ok else 2)


# ---------------------------------------------------------------- report
def P_(a, b, g, m=None):
    """Paired difference a - b of per-unit arrays, clusters g resampled -> dict(new, ref, diff, lo, hi, n)."""
    from jevdrive import stats
    m = np.ones(len(a), bool) if m is None else m
    r = stats.paired(np.asarray(a, float)[m], np.asarray(b, float)[m], groups=np.asarray(g)[m])
    return dict(n=int(m.sum()), new=r["mean_a"], ref=r["mean_b"], diff=r["mean"], lo=r["lo"], hi=r["hi"])


def contrasts(val, g, m=None):
    """val: {group: [seed arrays]} -> rows for S - base and every arm - base / arm - S (seed means per unit), with per-seed means."""
    sm = {k: sum(v) / len(v) for k, v in val.items()}
    m_ = np.ones(len(g), bool) if m is None else m
    out = []
    for x, y in [("S", "base")] + [(a, r) for a in ARMS for r in ("base", "S")]:
        out.append(dict(arm=x, vs=y, **P_(sm[x], sm[y], g, m), per_seed=" / ".join(f"{np.asarray(v, float)[m_].mean():.5g}" for v in val[x])))
    return out


def verdict(rows, scale=1.0):
    """The common rule of prereg 5 on one effect: rows of `contrasts` -> {arm: carries | not needed | unresolved}, E = S - base."""
    by = {(r["arm"], r["vs"]): r for r in rows}
    E = by[("S", "base")]
    sg, out = np.sign(E["diff"]), {}
    for a in ARMS:
        xs, xb = by[(a, "S")], by[(a, "base")]
        ex0 = lambda r: r["lo"] > 0 or r["hi"] < 0  # noqa: E731
        carries = sg * xs["diff"] <= -0.5 * abs(E["diff"]) and ex0(xs)
        keeps = sg * xb["diff"] >= 0.5 * abs(E["diff"]) and ex0(xb)
        out[a] = dict(verdict="carries" if carries and not keeps else "not needed" if keeps and not carries else "both lines met" if carries else "unresolved",
                      share_kept=xb["diff"] / E["diff"] if E["diff"] else np.nan)
    return dict(effect=dict(diff=E["diff"], lo=E["lo"], hi=E["hi"], excludes_0=bool(E["lo"] > 0 or E["hi"] < 0)), arms=out)


def cmd_report(_):
    import pandas as pd
    import prog_ol as PO
    import route as RT
    import route_ol as RO
    from jevdrive.bench import tables as BT
    groups = ("base", "S") + ARMS
    rows, V, J = [], {}, {}

    def add(board, subset, metric, val, g, m=None, scale=1.0, line=False):
        rs = contrasts({k: [np.asarray(x, float) * scale for x in v] for k, v in val.items()}, g, m)
        for r in rs:
            rows.append(dict(board=board, subset=subset, metric=metric, **r))
        if line:
            V[f"{board} | {subset} | {metric}"] = verdict(rs)
        return rs

    # ---- navhard (G frame) and navtest through the bench unit files
    U = {k: [BT.load("navhard", tag(k, s) + "@gimm")[0] for s in SEEDS] for k in groups}
    idx = sorted(set.intersection(*[set(u.index) for v in U.values() for u in v]))
    assert len(idx) == 225, len(idx)
    g = U["base"][0].loc[idx, "log"].astype(str).to_numpy()
    for col in ("combined", "stage1", "stage2"):
        rs = add("navhard", "225 groups", col, {k: [u.loc[idx, col].to_numpy(float) for u in v] for k, v in U.items()}, g, line=True)
        if col == "stage2":
            by = {(r["arm"], r["vs"]): r for r in rs}
            G2 = by[("S", "base")]["diff"]
            J["q1"] = dict(G2=G2, half=G2 / 2, arms={a: dict(
                minus_S=by[(a, "S")], minus_base=by[(a, "base")], share_kept=by[(a, "base")]["diff"] / G2,
                verdict="carries" if by[(a, "S")]["diff"] <= -G2 / 2 and by[(a, "S")]["hi"] < 0 else
                        "not needed" if by[(a, "base")]["diff"] >= G2 / 2 and by[(a, "base")]["lo"] > 0 else "unresolved") for a in ARMS})
    J["navhard_combined_per_seed"] = {k: [float(u.loc[idx, "combined"].mean()) for u in v] for k, v in U.items()}
    N = {k: [BT.load("navtest", tag(k, s))[0] for s in SEEDS] for k in groups}
    Mn = pd.read_parquet(B.root() / "prog" / "ol_sdrop_navtest.parquet")
    tok = Mn.name.to_numpy()
    assert all(not u.reindex(tok).score.isna().any() for v in N.values() for u in v)
    gl, dy = Mn.log.to_numpy(), Mn.dyaw.to_numpy()
    bins = [("all", np.ones(len(tok), bool)), ("< 5 deg", dy < 5), ("5-20 deg", (dy >= 5) & (dy <= 20)), ("20-45 deg", (dy > 20) & (dy <= 45)), ("> 45 deg", dy > 45)]
    for name, m in bins:
        add("navtest", name, "EPDMS", {k: [u.reindex(tok).score.to_numpy(float) for u in v] for k, v in N.items()}, gl, m, 100.0, line=name in ("all", "> 45 deg"))
    for sub in ("NC", "DAC", "EP", "TTC"):
        add("navtest", "all", sub, {k: [u.reindex(tok)[sub].to_numpy(float) for u in v] for k, v in N.items()}, gl, None, 100.0)

    # ---- turn-oracle replay tables of bd4_g3d.py (arm - ref, two-seed mean, by log), copied in
    for a in ("S",) + ARMS:
        for ref in ("base", "S"):
            f = OUT / f"d_{a}_vs_{ref}.csv"
            if a == ref or not f.exists():
                continue
            D = pd.read_csv(f)
            for _, r in D[D.stratum.isin(["all", "T45 (> 45 deg)"]) & D.metric.isin(["DAC fail %", "inside-cut %", "cannot-make-turn %"])].iterrows():
                rows.append(dict(board="navtest replay", subset=r.stratum, metric=r.metric, arm=a, vs=ref, n=r.n, new=r.new, ref=r.base, diff=r["diff"], lo=r.lo, hi=r.hi,
                                 per_seed=" / ".join(f"{r[f'diff_s{s}']:+.3g}" for s in SEEDS) + " (diff)"))
    R_ = pd.DataFrame(rows)
    for met, sub in (("DAC fail %", "all"), ("DAC fail %", "T45 (> 45 deg)"), ("inside-cut %", "T45 (> 45 deg)"), ("cannot-make-turn %", "T45 (> 45 deg)")):
        q = R_[(R_.board == "navtest replay") & (R_.metric == met) & (R_.subset == sub)]
        if len(q) == 7:
            V[f"navtest replay | {sub} | {met}"] = verdict(q.to_dict("records"))

    # ---- own-plan reads from the dumps: contact rates, arc ratio, widening
    chk, arc_rows, w_rows = [], [], []
    for st, M in (("navtest", Mn), ("hold", pd.read_parquet(B.root() / "prog" / "ol_sdrop_hold.parquet"))):
        gl = M.log.to_numpy()
        col = lambda q, k: [M[f"{q}|{tag(k, s)}"].to_numpy() for s in SEEDS]  # noqa: E731
        rate = {"agent": {k: [h.astype(float) for h in col("hit", k)] for k in groups},
                "boundary": {k: [((bm < -0.20) & ~bt).astype(float) for bm, bt in zip(col("bm", k), col("bt0", k))] for k in groups}}
        fam = M.fam.to_numpy()
        subs = [("pooled", np.ones(len(M), bool)), ("> 45 deg", (M.dyaw > 45).to_numpy())]
        subs += [("on-log", fam == "log"), ("off-track", fam != "log")] + [(f, fam == f) for f in ("ot1", "yr1", "bd4")] if st == "hold" else []
        for q, val in rate.items():
            for name, m in subs:
                add(f"{st} own plan", name, f"{q} rate", val, gl, m, line=name in ("pooled", "on-log", "off-track"))
            for a in ARMS:                                                    # cross-check against bd4_g3.py's own per-seed pooled rate
                for s in SEEDS:
                    f = OUT / f"g3_sdrop_{st}_{a}_s{s}.json"
                    if f.exists():
                        p = json.loads(f.read_text())["pooled"][tag("base", s)][{"agent": "agent", "boundary": "bnd"}[q]]
                        chk.append(dict(set=st, arm=a, seed=s, rate=q, report=float(val[a][s].mean()), bd4_g3=p["new"], states_apart=abs(float(val[a][s].mean()) - p["new"]) * len(M)))
        grp = PO.group(M, tag("base", 0))
        asub = [("pooled", np.ones(len(M), bool)), ("open", grp == "open"), ("lead", grp == "lead")] + ([(f, fam == f) for f in ("log", "ot1", "yr1", "bd4")] if st == "hold" else [])
        lim = {("navtest", "pooled"): 0.995, ("navtest", "open"): 0.995, ("navtest", "lead"): 0.990, ("hold", "pooled"): 0.990} | {("hold", f): 0.980 for f in ("log", "ot1", "yr1", "bd4")}
        for name, m in asub:
            for k in ("S",) + ARMS:
                ps = [P_(x, y, gl, m) for x, y in zip(col("arc4", k), col("arc4", "base"))]
                p2 = P_(sum(col("arc4", k)) / 2, sum(col("arc4", "base")) / 2, gl, m)
                arc_rows.append(dict(set=st, subset=name, arm=k, n=p2["n"], ratio=p2["new"] / p2["ref"], lo=1 + p2["lo"] / p2["ref"], hi=1 + p2["hi"] / p2["ref"],
                                     ratio_s0=ps[0]["new"] / ps[0]["ref"], ratio_s1=ps[1]["new"] / ps[1]["ref"], line=lim.get((st, name), np.nan),
                                     met=bool(min(p["new"] / p["ref"] for p in ps) >= lim[(st, name)]) if (st, name) in lim else None))
        # widening (lib/route.py, as route_ol.py): signed lateral of the own plan against the logged path, + = outside of the logged turn
        tags = [c.split("|")[1] for c in M.columns if c.startswith("arc4|")]
        Pl = np.load(B.root() / "prog" / f"ol_sdrop_{st}_plans.npy")
        ns = SimpleNamespace(set=st, fams=["log", "ot1", "yr1", "bd4"], shards=list(range(B.NSH)))
        F = RO.logged(M, ns, {"hold": B.HOLD, "navtest": "navsim/navtest"}[st])
        off = M[["dy", "dpsi"]].to_numpy(np.float64)
        ok = ~np.isnan(F[:, :, 0]).any(1)
        turn = RT.turn_deg(F, off)
        side = -np.sign(turn)
        Q = {}
        for t in {tag(k, s) for k in groups for s in SEEDS}:
            d, lat = RT.path_dist_np(Pl[tags.index(t)], F, off)
            so = side[:, None] * lat
            Q[t] = dict(out4=so[:, -1], L4=(d.max(1) > 4).astype(float), **{f"W{r}": (so.max(1) > r).astype(float) for r in (1, 2, 3)})
        wsub = [("> 45 deg", np.abs(turn) > 45), ("> 45 deg right", turn < -45), ("> 45 deg left", turn > 45)]
        fsub = [("navtest", np.ones(len(M), bool))] if st == "navtest" else [("hold on-log", fam == "log"), ("hold four families", np.ones(len(M), bool))]
        for fn, fm in fsub:
            for bn, bm_ in wsub:
                m = fm & bm_ & ok
                for k in groups:
                    w_rows.append(dict(set=fn, bucket=bn, arm=k, n=int(m.sum()), **{f"{q}_s{s}": (float(Q[tag(k, s)][q][m].mean()) if q == "out4" else int(Q[tag(k, s)][q][m].sum()))
                                                                                    for q in ("W1", "W2", "W3", "L4", "out4") for s in SEEDS}))
                if bn == "> 45 deg":
                    for q, nm in (("out4", "signed lateral at 4 s (m, + = outside)"), ("W2", "W2 share")):
                        add(f"{fn} own plan", bn, nm, {k: [Q[tag(k, s)][q] for s in SEEDS] for k in groups}, gl, m, line=True)
    W = pd.DataFrame(w_rows)
    w = W[(W.set == "navtest") & (W.bucket == "> 45 deg")].set_index("arm")
    wide = {}
    for a in ARMS:
        ex = [int(w.loc[a, f"W2_s{s}"] - w.loc["base", f"W2_s{s}"]) for s in SEEDS]
        wide[a] = dict(W2=[int(w.loc[a, f"W2_s{s}"]) for s in SEEDS], excess_over_base=ex,
                       verdict="not wide" if max(ex) <= W2_TOL else "wide" if min(ex) > W2_TOL else "mixed")
    vs_ = {a: wide[a]["verdict"] for a in ARMS}
    holds = vs_["noA"] == "wide" and vs_["noC"] == "not wide" and vs_["noB"] == "not wide"
    fails = vs_["noC"] == "wide" or vs_["noA"] == "not wide"
    J["q2"] = dict(tolerance=W2_TOL, base_W2=[int(w.loc["base", f"W2_s{s}"]) for s in SEEDS], S_W2=[int(w.loc["S", f"W2_s{s}"]) for s in SEEDS], arms=wide,
                   only_road_hinge_on_hinge_only_rows="holds" if holds else "fails" if fails else "unresolved")
    A = pd.DataFrame(arc_rows)
    J["arc_guard_missed"] = A[A.met == False][["set", "subset", "arm", "ratio_s0", "ratio_s1", "line"]].to_dict("records")  # noqa: E712
    J["attribution"] = V
    J["cross_check_bd4_g3"] = dict(max_states_apart=max((c["states_apart"] for c in chk), default=None), n=len(chk))
    R_ = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    R_.to_csv(OUT / "contrasts.csv", index=False, float_format="%.6g")
    A.to_csv(OUT / "arc.csv", index=False, float_format="%.5f")
    W.to_csv(OUT / "widening.csv", index=False, float_format="%.4f")
    pd.DataFrame(chk).to_csv(OUT / "cross_check.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(J, indent=1, default=float) + "\n")
    f3 = lambda r: f"{r['diff']:+.4g} [{r['lo']:+.4g}, {r['hi']:+.4g}]"  # noqa: E731
    md = ["# S-DROP summary (sdrop_report.py; lines of plans/2026-10-10-sdrop-prereg.md section 5)", "",
          f"Question 1: G2 = {J['q1']['G2']:+.2f} (stage 2, P2H10S - P2H10, seeds 0 + 1); half = {J['q1']['half']:.2f}", "",
          "| arm | dropped | stage 2: arm - P2H10S | stage 2: arm - base | share of G2 kept | verdict |", "|:--|:--|:--|:--|--:|:--|"]
    md += [f"| {a} | {DROPPED[a]} | {f3(q['minus_S'])} | {f3(q['minus_base'])} | {q['share_kept']:.2f} | {q['verdict']} |" for a, q in J["q1"]["arms"].items()]
    md += ["", f"Question 2: W2 on navtest tokens over 45 deg, seeds 0 / 1: base {J['q2']['base_W2']}, P2H10S {J['q2']['S_W2']}; tolerance +{W2_TOL}. "
           f"\"Only the road hinge on hinge-only rows\": **{J['q2']['only_road_hinge_on_hinge_only_rows']}**", "", "| arm | W2 | excess over base | verdict |", "|:--|:--|:--|:--|"]
    md += [f"| {a} | {q['W2']} | {q['excess_over_base']} | {q['verdict']} |" for a, q in wide.items()]
    md += ["", "Attribution by the common rule (E = P2H10S - base on the same two seeds):", "", "| effect | E [95 % CI] | noA | noB | noC |", "|:--|:--|:--|:--|:--|"]
    md += [f"| {k} | {f3(v['effect'])}{'' if v['effect']['excludes_0'] else ' (includes 0)'} | " + " | ".join(f"{v['arms'][a]['verdict']} ({v['arms'][a]['share_kept']:.2f})" for a in ARMS) + " |"
           for k, v in V.items()]
    md += ["", "Share kept = (arm - base) / E. Full tables: contrasts.csv, arc.csv, widening.csv.", "", f"Arc guard missed: {J['arc_guard_missed'] or 'none'}", "",
           f"Cross-check against bd4_g3.py per-seed pooled rates: at most {J['cross_check_bd4_g3']['max_states_apart']} states apart over {len(chk)} numbers."]
    (OUT / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    raise SystemExit(0 if not chk or J["cross_check_bd4_g3"]["max_states_apart"] < 0.5 else 3)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["smoke", "trained", "report"])
    a = ap.parse_args()
    {"smoke": cmd_smoke, "trained": cmd_trained, "report": cmd_report}[a.cmd](a)
