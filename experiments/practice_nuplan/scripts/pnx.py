"""PNX: the two full-length arms on on1_JTX without the new standing rows (plans/2026-10-19-pn-prereg.md supplement 3). Box, repo root.

  $TR pnx.py wns              (CPU) JTX00NS = JTX00 (JTN's draw on on1_JTX, trainval rows at 38 / 70) with the new start-state S rows
        (pool pn*) at tab speed < 0.5 m/s set to weight 0, the other S rows scaled back to the S mass (trainval and held rows untouched)
        = lat_diag.py wns's rule; asserts the same code on JTX00S reproduces the stored JTX00SNS bit for bit -> runs/defect_fix/w/JTX00NS.npz
  $TR pnx.py smoke            (CPU) the 50-step smokes against their parents' training logs: dev ADE at step 0 equal (same init), loss/imit
        at steps 25 / 50 within 5 %, every loss scalar finite -> $W/smoke.json (exit 1 on a failure: the trainings do not start)
  $TR pnx.py sum              (CPU) after every read: held 460 @18 (df.cl_read), native arena held pairs (jt.py arena's rows), PAI 281
        ACC + GAP (pai281.py's md), closed-loop lateral drift (rf_follow decisions, lat_diag's rule), gates -> results/<arm>/, adoption.md
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(HERE), str(HERE.parents[1] / "defect_fix/scripts")]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
W = DATA / "runs/practice_nuplan/pnx"
WD = DATA / "runs/defect_fix/w"
POOL = DATA / "runs/op_parity/cache/on1_JTX"
RES = REPO / "experiments/practice_nuplan/results"
# arm: parent, smoke tag; PAI reference for both arms = NTV0 (d366's serve)
ARMS = {"NTVXS-L18-s0": dict(parent="NTV0-L18-s0", smoke="NTVXSSM-L18-s0"), "JTXS-L18-s1": dict(parent="JTN-L18-s1", smoke="JTXSSM-L18-s1")}
PAI_REF = "NTV0-L18-s0"
LAB = {"NTVXS-L18-s0": "NTVXS", "JTXS-L18-s1": "JTXS", "NTV0-L18-s0": "NTV0", "JTN-L18-s1": "JTNs1"}
NUM = r"([+-]?\d+\.\d+)"


def drop_static(src):
    """lat_diag.py wns's rule on any draw of on1_JTX: new (pn*) S rows standing at t0 -> weight 0, the S mass kept."""
    z, tb = np.load(WD / f"{src}.npz"), np.load(POOL / "tab.npz")
    assert np.array_equal(z["names"], tb["names"])
    w, grp = z["w"].astype(np.float64).copy(), z["grp"].astype("U12").copy()
    drop = np.char.startswith(tb["pool"].astype(str), "pn") & (tb["speed"] < 0.5) & (grp == "S")
    s0 = w[grp == "S"].sum()
    w[drop], grp[drop] = 0.0, "S_static0"
    w[grp == "S"] *= s0 / w[grp == "S"].sum()
    w /= w.sum()
    return z, w, grp, drop


def cmd_wns(a):
    _, w1, g1, _ = drop_static("JTX00S")
    ref = np.load(WD / "JTX00SNS.npz")
    assert np.array_equal(ref["w"], w1) and np.array_equal(ref["grp"].astype(str), g1.astype(str)), "rule does not reproduce JTX00SNS"
    z, w, grp, drop = drop_static("JTX00")
    out = WD / "JTX00NS.npz"
    assert not out.exists(), f"{out} exists"
    w0, g0 = z["w"].astype(np.float64), z["grp"].astype(str)
    tv = ~np.isin(g0, ["S", "S_held"])
    assert np.allclose(w[tv], w0[tv] / w0.sum(), rtol=1e-9, atol=0)  # trainval weights unchanged
    assert np.array_equal(g0 == "S_held", grp == "S_held")
    np.savez(out, names=z["names"], w=w, grp=grp)
    rec = dict(src=str(WD / "JTX00.npz"), rule="lat_diag.py wns (reproduces JTX00SNS from JTX00S bit for bit)", dropped=int(drop.sum()),
               s_rows_left=int((grp == "S").sum()), held=int((grp == "S_held").sum()), s_mass=float(w[grp == "S"].sum()), s_mass_src=float(w0[g0 == "S"].sum() / w0.sum()),
               tv_mass=float(w[tv].sum()), share={g: float(w[grp == g].sum()) for g in np.unique(grp)})
    out.with_suffix(".json").write_text(json.dumps(rec, indent=1))
    (RES / "w_JTX00NS.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


def events(tag):
    fs = sorted((DATA / f"runs/route_mem/train-{tag}").glob("*/events.jsonl"))
    assert fs, f"no events for {tag}"
    S = {}
    for line in open(fs[-1]):
        r = json.loads(line)
        if r.get("kind") == "scalar" and r["step"] <= 50:
            S.setdefault(r["step"], {})[r["tag"]] = r["value"]
    return S


def cmd_smoke(a):
    out, ok = {}, True
    for arm, c in ARMS.items():
        s, p = events(c["smoke"]), events(c["parent"])
        r = dict(parent=c["parent"], smoke=c["smoke"])
        r["ade0"] = (s[0].get("dev/ade"), p[0].get("dev/ade"))
        r["imit"] = {k: (s.get(k, {}).get("loss/imit"), p.get(k, {}).get("loss/imit")) for k in (25, 50)}
        r["pool_terms_25"] = {t: (s[25].get(t), p[25].get(t)) for t in sorted(s.get(25, {})) if t.startswith(("loss/on/", "loss/act/pool"))}
        r["held0"] = {t: (s[0].get(t), p[0].get(t)) for t in ("pool_held/agent_mean", "pool_held/road_mean", "pool_held/lat_side_mean")}
        fin = all(np.isfinite(v) for st in s.values() for t, v in st.items() if t.startswith("loss/"))
        g = dict(finite=bool(fin), ade0=abs(r["ade0"][0] - r["ade0"][1]) <= 1e-3,
                 imit=all(x is not None and y is not None and abs(x - y) <= 0.05 * abs(y) for x, y in r["imit"].values()))
        r["gates"], r["pass"] = g, all(g.values())
        ok &= r["pass"]
        out[arm] = r
        print(arm, json.dumps(r, default=float))
    W.mkdir(parents=True, exist_ok=True)
    (W / "smoke.json").write_text(json.dumps(out, indent=1, default=float))
    (RES / "smoke.json").write_text(json.dumps(out, indent=1, default=float))
    sys.exit(0 if ok else 1)


def arena_pairs(rows, a_, b_):
    import pandas as pd
    from jevdrive import stats
    S = pd.read_csv(DATA / "runs/alpasim_proxy/arena/set/set.csv.gz").set_index("id")
    H = S[S.split == "held"]
    X = {t: rows[(rows.tag == f"{t}@off") & rows.id.isin(H.index)].drop_duplicates("id").set_index("id") for t in (a_, b_)}
    bad = set().union(*(set(x.index[x.type == "route-fail"]) for x in X.values()))
    c = X[a_].index.intersection(X[b_].index).difference(bad)
    x, y = X[a_].loc[c, "score"].values, X[b_].loc[c, "score"].values
    p = stats.paired(x, y, groups=H.loc[c, "log"].values)
    st = H.loc[c, "stratum"].values
    return dict(windows=len(c), mean_a=float(x.mean()), mean_b=float(y.mean()), diff=p["mean"], lo=p["lo"], hi=p["hi"],
                zeros_a=int((x == 0).sum()), zeros_b=int((y == 0).sum()),
                strata={s: dict(n=int((st == s).sum()), a=float(x[st == s].mean()), b=float(y[st == s].mean())) for s in sorted(set(st))})


def drift(D, S, A, B):
    """lat_diag report section 2: signed lateral offset to the route (left +), A - B on the same (scene, k), both before their first zero
    event, non-turn scenes, decisions 11-17 (and 5-10), log-clustered 95 % CI."""
    from lat_diag import _boot
    P = D.pivot_table(index=["scene", "k"], columns="label", values="lat").reset_index()
    KZ = S.pivot_table(index="scene", columns="label", values="kz")
    turn = set(S[S.turn == True].scene)  # noqa: E712
    m = P.merge(KZ[[A, B]].rename(columns={A: "za", B: "zb"}).reset_index(), on="scene")
    m = m[((m.za < 0) | (m.k < m.za)) & ((m.zb < 0) | (m.k < m.zb))].dropna(subset=[A, B])
    o = {}
    for nm, ks, tr in (("k5_10", range(5, 11), False), ("k11_17", range(11, 18), False), ("turn_k11_17", range(11, 18), True)):
        q = m[m.k.isin(ks) & (m.scene.isin(turn) == tr)]
        mu, lo, hi = _boot(q[A] - q[B], q.scene.str.rsplit("-", n=1).str[0])
        o[nm] = dict(n=len(q), mean=mu, lo=lo, hi=hi)
    return o


def cmd_sum(a):
    import pandas as pd
    import df
    ci = lambda m, lo, hi, f="+.4f": f"{m:{f}} [{lo:{f}}, {hi:{f}}]"  # noqa: E731
    yn = lambda b: "**yes**" if b else "no"  # noqa: E731
    tags = [*ARMS, "NTV0-L18-s0", "JTN-L18-s1"]
    # held 460 @18
    H, L = {}, ["# Held 460 @18 (PN prereg supplement 3)", "", "Paired = row tag minus the named tag, session-clustered 95 % CI (defect_fix df.cl_read).", "", *df.CL_HEAD]
    for t in tags:
        refs = [r for r in ("RSG-L18-s0", *(x for x in ("NTV0-L18-s0", "JTN-L18-s1") if x != t)) if r != t]
        H[t] = df.cl_read(t, refs)
        L += df.fmt_cl(t, H[t], refs)
    (W / "held.md").write_text("\n".join(L) + "\n")
    # native arena held 2 228 (jt.py arena's collected rows)
    rows = pd.read_csv(W / "arena/rows.csv.gz")
    AR = {(x, y): arena_pairs(rows, x, y) for x in ARMS for y in ("NTV0-L18-s0", "JTN-L18-s1")}
    AR[("JTN-L18-s1", "NTV0-L18-s0")] = arena_pairs(rows, "JTN-L18-s1", "NTV0-L18-s0")
    # PAI 281 ACC + GAP
    PA = {}
    for arm in ARMS:
        for ref in {PAI_REF, ARMS[arm]["parent"]}:
            md = (W / f"pai_{arm}_vs_{ref}.md").read_text()
            jl, rl = f"{arm}+ACC+GAP", f"{ref}+ACC+GAP"
            p = tuple(map(float, re.search(rf"Paired mean \({re.escape(jl)} - {re.escape(rl)}\): \*\*{NUM} \[{NUM}, {NUM}\]", md).groups()))
            zr = {m[1]: dict(scenes=int(m[2]), mean=float(m[3]), zeros=int(m[4])) for m in re.finditer(r"^\| ([^|]+?) \| (\d+) \| (\d\.\d+) \| (\d+) \|", md, re.M)}
            prs = tuple(map(float, re.search(rf"Paired mean \({re.escape(jl)} - RSG\+ACC\+GAP\): \*\*{NUM} \[{NUM}, {NUM}\]", md).groups()))
            PA[arm, ref] = dict(pair=p, vs_rsg=prs, table=zr, scenes=zr[jl]["scenes"], zeros=zr[jl]["zeros"], zeros_ref=zr[rl]["zeros"], mean=zr[jl]["mean"])
    # closed-loop drift
    D, S = pd.read_parquet(W / "follow/decisions.parquet"), pd.read_csv(W / "follow/scenes.csv")
    DR = {arm: drift(D, S, f"{LAB[arm]}@18", f"{LAB[ARMS[arm]['parent']]}@18") for arm in ARMS}
    DR["JTXS-L18-s1 vs NTV0"] = drift(D, S, "JTXS@18", "NTV0@18")
    DR["NTVXS-L18-s0 vs JTN-L18-s1"] = drift(D, S, "NTVXS@18", "JTNs1@18")
    # gates per arm, both boards
    A = ["# Adoption: NTVXS-L18-s0 and JTXS-L18-s1 (PN prereg supplement 3)", "",
         "Native = arena held split, no hold (decision 371). Drift = closed-loop signed lateral offset to the route, arm minus parent, non-turn held 460 "
         "scenes, decisions 11-17, both before their first zero (decision 377's measure; NTVX - NTV0 was -0.110 [-0.131, -0.085]).", "",
         "| arm | parent | line | measured | pass |", "|---|---|---|---|:--|"]
    V = {}
    for arm, c in ARMS.items():
        P0 = c["parent"]
        ar, hz, hz0 = AR[arm, P0], H[arm]["held"]["zeros"], H[P0]["held"]["zeros"]
        dr, pa = DR[arm]["k11_17"], PA[arm, PAI_REF]
        g = dict(arena=ar["diff"] >= 0.005 and ar["lo"] > 0, held=hz <= hz0 + 3, drift=dr["mean"] > -0.05,
                 pai=pa["pair"][0] >= 0.01 and pa["pair"][1] > 0, pai_zeros=pa["zeros"] <= pa["zeros_ref"] + 2)
        nu, pai = g["arena"] and g["held"] and g["drift"], g["pai"] and g["pai_zeros"]
        V[arm] = dict(parent=P0, gates=g, nuplan=nu, pai=pai, arena=ar, held=dict(zeros=hz, parent_zeros=hz0, mean=H[arm]["held"]["mean"],
                      pair=H[arm]["held"]["pair"]), drift=DR[arm], pai281={k[1]: v for k, v in PA.items() if k[0] == arm})
        A += [f"| {arm} | {P0} | nuPlan native arena held vs {P0} native: >= +0.005, CI low > 0 | {ci(ar['diff'], ar['lo'], ar['hi'])} "
              f"({ar['windows']} windows; {ar['mean_a']:.4f} vs {ar['mean_b']:.4f}; zeros {ar['zeros_a']} / {ar['zeros_b']}) | {yn(g['arena'])} |",
              f"| {arm} | {P0} | held 460 @18 zeros <= {P0} + 3 | {hz} vs {hz0} (mean {H[arm]['held']['mean']:.4f}) | {yn(g['held'])} |",
              f"| {arm} | {P0} | drift vs {P0} > -0.05 m | {ci(dr['mean'], dr['lo'], dr['hi'], '+.3f')} m ({dr['n']} decisions) | {yn(g['drift'])} |",
              f"| {arm} | {PAI_REF} | PAI 281 ACC + GAP vs {PAI_REF} + ACC + GAP: >= +0.01, CI low > 0 | {ci(*pa['pair'])} ({pa['scenes']} scenes; "
              f"{pa['mean']:.4f}) | {yn(g['pai'])} |",
              f"| {arm} | {PAI_REF} | PAI zeros <= {PAI_REF}'s + 2 (same scenes) | {pa['zeros']} vs {pa['zeros_ref']} | {yn(g['pai_zeros'])} |"]
    A += ["", "Reported, not gated:", "", "| comparison | value |", "|---|---|"]
    for (x, y), r in AR.items():
        A.append(f"| native arena {x} - {y} | {ci(r['diff'], r['lo'], r['hi'])} ({r['windows']} w; zeros {r['zeros_a']} / {r['zeros_b']}) |")
    for (x, y), r in PA.items():
        A.append(f"| PAI 281 {x} - {y} (ACC + GAP) | {ci(*r['pair'])} (zeros {r['zeros']} / {r['zeros_ref']}); vs RSG + ACC + GAP {ci(*r['vs_rsg'])} |")
    for k, r in DR.items():
        A.append(f"| drift {k}: k 5-10 / k 11-17 / turn k 11-17 | " + " / ".join(f"{ci(r[q]['mean'], r[q]['lo'], r[q]['hi'], '+.3f')}" for q in ("k5_10", "k11_17", "turn_k11_17")) + " |")
    for arm in ARMS:
        for c_, p in H[arm]["held"]["pair"].items():
            A.append(f"| held 460 {arm} - {c_} | {ci(p['mean'], p['lo'], p['hi'])} (zeros {p['zeros']} / {p['zeros_ref']}) |")
    A += ["", *[f"{arm}: nuPlan **{'adopt' if V[arm]['nuplan'] else 'not adopted'}**, PAI **{'adopt' if V[arm]['pai'] else 'not adopted'}**." for arm in ARMS],
          "Stop3 was not read (reference only, not cheap: 36 more arena reads)."]
    for arm in ARMS:
        d = RES / arm
        d.mkdir(parents=True, exist_ok=True)
        (d / "verdict.json").write_text(json.dumps(V[arm], indent=1, default=float))
        for f in ("held.md", f"pai_{arm}_vs_{PAI_REF}.md", f"pai_{arm}_vs_{ARMS[arm]['parent']}.md", f"cmd_diff_{arm}.txt"):
            if (W / f).exists():
                (d / f).write_text((W / f).read_text())
    (RES / "adoption.md").write_text("\n".join(A) + "\n")
    print("\n".join(A))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for c in ("wns", "smoke", "sum"):
        sp.add_parser(c)
    a = ap.parse_args()
    dict(wns=cmd_wns, smoke=cmd_smoke, sum=cmd_sum)[a.cmd](a)
