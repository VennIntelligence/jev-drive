#!/usr/bin/env python3
"""ROUND9 CMD-PILOT gates and report (plans/2026-10-23-command-design.md section 5; lane scripts/cpl.sh). Box, repo root, repo venv.
  cpl.py smoke --tag T --out J       60-step smoke with --cmd-rule heading-all: DONE, every scalar finite, last logged step 60; G5: the trainer's
                                     per-part change share vs cmdd.py's rows.parquet (rows with a valid slot; <= 0.5 pt); G0: the command fed per sampled row
                                     (runs/route_mem/cmd_rule/T.npz) == cmdd.py's own recomputation (rule_h(20, 40) on the stored feat file;
                                     no valid slot -> the stored command), and == rows.parquet's h20_40 on the navtrain rows
  cpl.py verdict --dir O --out P     the registered reading lines (design section 5), paired CIs (old and new scorer), lateral zeros, D_new,
                                     route-turn zeros, the four listed scenes, NTV0-full-score new zeros, dev ADE, served-command agreement -> P.{md,json}
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(HERE)]
from jevdrive import stats  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

DD = data_dir()
RUNS = DD / "runs/op_parity/runs"
FACTS = DD / "runs/pai_fix/facts"
ROWS = DD / "runs/round9/cmd_design/rows.parquet"
LAT = ("offroad", "zero-other", "side")
SCENES = {"1a69f757": "long gentle right curve (off_straight)", "1e8e939e": "S-junction (off_junc_inside)",
          "baabc57d": "command flip before the junction", "596af0c7": "gentle right curve"}
ARMS = dict(HF="RB0F", HP50="RB0P50")
TAGS = dict(HF="HF-s0", HP50="HP50-s0", RB0F="RB0-F-s0", RB0P50="RB0-P50-s0", NTV0="NTV0-L18-s0")


def dump(p, obj):
    Path(p).write_text(json.dumps(obj, indent=1, default=float) + "\n")


def run_dir(tag):
    return sorted((DD / "runs/route_mem").glob(f"train-{tag}/*/"))[-1]


def cmd_rec(tag):
    m = re.search(r"--cmd-rule \(round7 joint arm\): (\{.*\})", (run_dir(tag) / "log.txt").read_text())
    return json.loads(m.group(1)) if m else None


def lab(c):
    return np.where(c.max(1) > 0.5, c.argmax(1), 3).astype(np.int8)


def cmd_smoke(a):
    import pandas as pd
    import cmdd
    import rmem as RM
    d = run_dir(a.tag)
    log = (d / "log.txt").read_text()
    ev = [json.loads(x) for x in (d / "events.jsonl").read_text().splitlines() if x.strip()]
    vals = np.array([e["value"] for e in ev if e.get("kind") == "scalar"], float)
    steps = [int(m) for m in re.findall(r"step (\d+): ", log)]
    ck = dict(done=(d / "DONE").exists(), scalars=int(len(vals)), finite=bool(len(vals) and np.isfinite(vals).all()),
              last_logged_step=max(steps) if steps else None)
    rec = cmd_rec(a.tag)
    assert rec and rec["rule"] == "heading-all", "no heading-all record in the log"
    # G5: per-part change shares against cmdd.py's rows (ok rows only: the trainer keeps the stored command where the memory is empty)
    R = pd.read_parquet(ROWS, columns=["part", "stored", "ok", "nval", "h20_40"])
    R["mem"] = R.ok & (R.nval > 0)                # a valid slot in the fed memory; else the trainer keeps the stored command (design 3 (1))
    ref = (R.mem & (R.stored != R.h20_40)).groupby(R.part).mean()
    g5 = {}
    for g, p in rec["parts"].items():
        k = "navtrain" if g == "navtrain_full" else g
        if k in ref.index:
            g5[g] = dict(trainer=p["changed"], cmdd=round(float(ref[k]), 4), diff_pt=round(100 * (p["changed"] - float(ref[k])), 2))
    g5_ok = bool(g5) and all(abs(v["diff_pt"]) <= 0.5 for v in g5.values())
    # G0: per sampled row, the fed command vs cmdd.py's independent recomputation on the stored feat file
    Z = np.load(DD / "runs/route_mem/cmd_rule" / f"{a.tag}.npz", allow_pickle=False)
    feat = DD / "runs/route_mem/feat"
    exp, old_ref, nav = np.full(len(Z["idx"]), -1, np.int8), np.full(len(Z["idx"]), -1, np.int8), {}
    nav_sizes = {}
    for dname in np.unique(Z["dir"]):
        m = Z["dir"] == dname
        F = np.load(feat / f"{dname}.npz")
        idx = Z["idx"][m]
        assert np.array_equal(F["names"][idx].astype(str), Z["name"][m]), f"{dname}: sample names differ from the feat file's"
        f = F["f"][idx].astype(np.float16)
        f[~F["ok"][idx]] = 0
        has = (f.astype(np.float32).reshape(len(f), RM.N_PT, RM.NF)[..., 0] > 0.5).any(1)
        tab = DD / "runs/op_parity/cache" / dname / "tab.npz"
        st = lab(np.load(tab)["ego"][idx, 1:4].astype(np.float32)) if tab.exists() else Z["old"][m]
        old_ref[m] = st
        exp[m] = np.where(has, cmdd.rule_h(cmdd.geom(f), 20, 40), st)
    for s in range(12):
        nav_sizes[f"navtrain_full.s{s}of12"] = len(np.load(feat / f"navtrain_full.s{s}of12.npz")["names"])
    off = dict(zip(nav_sizes, np.r_[0, np.cumsum(list(nav_sizes.values()))[:-1]]))
    navr = R[R.part == "navtrain"].reset_index(drop=True)
    assert len(navr) == sum(nav_sizes.values()), "rows.parquet navtrain rows != the 12 shards"
    mn = np.isin(Z["dir"], list(nav_sizes))
    pos = np.array([off[x] for x in Z["dir"][mn]]) + Z["idx"][mn]
    okn = navr.mem.to_numpy()[pos]
    hn = np.where(okn, navr.h20_40.to_numpy()[pos], navr.stored.to_numpy()[pos])
    agree = dict(rows=int(len(exp)), fed_vs_cmdd_rule=float((exp == Z["new"]).mean()), stored_vs_tab=float((old_ref == Z["old"]).mean()),
                 navtrain_rows=int(mn.sum()), fed_vs_rows_parquet=float((hn == Z["new"][mn]).mean()) if mn.any() else None,
                 no_memory=int((~Z["has"]).sum()), changed=float((Z["old"] != Z["new"]).mean()),
                 by_group={g: dict(rows=int((Z["grp"] == g).sum()), agree=float((exp == Z["new"])[Z["grp"] == g].mean())) for g in np.unique(Z["grp"])})
    g0_ok = agree["fed_vs_cmdd_rule"] >= 0.999 and (agree["fed_vs_rows_parquet"] is None or agree["fed_vs_rows_parquet"] >= 0.999)
    ok = ck["done"] and ck["finite"] and ck["last_logged_step"] == 60 and g5_ok and g0_ok
    out = dict(tag=a.tag, run=str(d), checks=ck, record=rec, g5=g5, g5_pass=g5_ok, g0=agree, g0_pass=bool(g0_ok), pass_=bool(ok))
    dump(a.out, out)
    print(json.dumps(out, indent=1))
    sys.exit(0 if ok else 2)


def f3(d):
    return "%+.3f [%+.3f, %+.3f]" % (d["mean"], d["lo"], d["hi"]) if d else "-"


def dev_ade(tag):
    f = RUNS / tag / "route_mem.json"
    return json.loads(f.read_text()).get("dev_ade") if f.exists() else None


def cmd_verdict(a):
    import pandas as pd
    o = Path(a.dir)
    R = json.loads((o / "results/pai281.json").read_text())
    rd, P = R["reads"], R["pairs"]
    Fct = {k: pd.read_csv(FACTS / f"{t}-281ag.csv").set_index("scene") for k, t in TAGS.items()}
    rpc = {k: int(F.rpc.fillna(False).astype(bool).sum()) for k, F in Fct.items()}
    lat = {k: sum(r[f"z_{c}"] for c in LAT) for k, r in rd.items()}
    rt = {k: r["z_route-turn"] for k, r in rd.items()}
    lau = {k: int(r["launch"].split("/")[0]) for k, r in rd.items()}
    ade = {k: dev_ade(TAGS[k]) for k in ("HF", "HP50", "RB0F", "RB0P50")}
    # new scorer
    PS = pd.read_csv(DD / "runs/alpasim/scorer_v2/report/per_scene.csv", header=[0, 1], index_col=0)["new"]
    new = {"RB0F": PS["RB0-F-s0-ag"], "RB0P50": PS["RB0-P50-s0-ag"], "NTV0": PS["NTV0-L18-s0-ag"]}
    for m in ARMS:
        f = o / f"sv2_{m}/new/rollouts.csv"
        new[m] = pd.read_csv(f).drop_duplicates("scene", keep="last").set_index("scene").score if f.exists() else None
    sv = None
    if all(v is not None for v in new.values()):
        bad = set().union(*[set(F.index[F.rpc.fillna(False).astype(bool)]) for F in Fct.values()])
        sc = sorted(set.intersection(*[set(v.dropna().index) for v in new.values()]) - bad)
        N = {k: v.reindex(sc).to_numpy(float) for k, v in new.items()}
        sv = dict(scenes=len(sc), mean={k: float(v.mean()) for k, v in N.items()}, zeros={k: int((v == 0).sum()) for k, v in N.items()},
                  pairs={p: stats.paired(N[p.split(":")[0]], N[p.split(":")[1]]) for p in ("HF:RB0F", "HP50:RB0P50", "HP50:HF", "RB0P50:RB0F")})
    # guard (HF vs RB0-F), lines
    g_mean = sv["mean"]["HF"] - sv["mean"]["RB0F"] if sv else None
    guard = {"new-scorer mean >= RB0-F - 0.03": g_mean is not None and g_mean >= -0.03,
             "route-turn <= RB0-F + 3": rt["HF"] <= rt["RB0F"] + 3, "lateral zeros <= RB0-F + 5": lat["HF"] <= lat["RB0F"] + 5,
             "launch >= 30 / 37": lau["HF"] >= 30, "dev ADE <= 0.538": ade["HF"] is not None and ade["HF"] <= 0.538}
    d_new, d_old = lat["HP50"] - lat["HF"], lat["RB0P50"] - lat["RB0F"]
    if not all(guard.values()):
        line = "HF guard FAILED: changing navtrain's command itself hurts the base -> fall back to R7J-style (AlpaSim-route rows + serve only); not judged further"
    elif d_new <= 8 and rt["HP50"] <= rt["HF"] - 5:
        line = "D_new <= 8 with the route-turn gain kept: command inconsistency was the main cause -> NTV1 heading on all rows, PAI human rows in"
    elif d_new >= 17:
        line = "D_new >= 17: the command is not the cause -> NTV1 keeps the heading command on all rows, no PAI human rows in the first NTV1"
    else:
        line = "between: partial -> propose seed 1 for both arms (~2.6 GPU h), nothing run"
    # scenes, NTV0-full-score new zeros
    common = R["scenes"]
    cells = ("RB0F", "HF", "RB0P50", "HP50")
    srows = []
    for s, what in SCENES.items():
        row = [s, what]
        for k in cells:
            F = Fct[k]
            hit = F.index[F.index.str.contains(s)]
            if len(hit):
                x = F.loc[hit[0]]
                nv = new[k].get(hit[0]) if new.get(k) is not None else None
                ns_ = "-" if nv is None or pd.isna(nv) else "%.2f" % nv
                row.append(f"{x.score:.2f} / {ns_} {x.fclass}")
            else:
                row.append("-")
        srows.append(row)
    full = Fct["NTV0"].index[Fct["NTV0"].score >= 0.98]
    newbad = {k: int((Fct[k].reindex(full).score == 0).sum()) for k in cells}
    ca = {m: json.loads((o / f"results/cmdagree_{m}.json").read_text()) if (o / f"results/cmdagree_{m}.json").exists() else None for m in ARMS}
    trec = {m: cmd_rec(TAGS[m]) for m in ARMS}
    cls = ["zeros", "z_offroad", "z_zero-other", "z_side", "z_route-turn", "z_overshoot", "z_AF-front", "p_rear-hit", "p_slow", "launch"]
    L = ["# CMD-PILOT verdict (PAI 281, ACC + GAP, Harmonizer OFF; HF / HP50 served with JEV_CMD=heading)", "",
         f"Old scorer: {common} scenes common to the five reads without RPC error; RPC scenes per read {rpc} (> 5 = re-read before judging).", "",
         "| read | mean (old) | lateral zeros | " + " | ".join(cls) + " | dev ADE |", "|:--|--:|--:|" + "--:|" * len(cls) + "--:|"]
    for k in ("RB0F", "HF", "RB0P50", "HP50", "NTV0"):
        r = rd[k]
        L.append(f"| {k} | {r['mean']:.3f} | {lat[k]} | " + " | ".join(str(r[c]) for c in cls) + f" | {ade.get(k) or '-'} |")
    L += ["", "Lateral zeros = zeros with fclass in {offroad, zero-other, side} (old-scorer facts; the scorer fix does not touch these classes).", "",
          "| lateral-zero table | F (all-row base) | P50 (58 of 115 imitation rows PAI human) | P50 - F |", "|:--|--:|--:|--:|",
          f"| old command (RB0-F-s0 / RB0-P50-s0) | {lat['RB0F']} | {lat['RB0P50']} | D_old = {d_old:+d} |",
          f"| heading command, all rows (HF / HP50) | {lat['HF']} | {lat['HP50']} | **D_new = {d_new:+d}** |",
          f"| route-turn zeros old / heading | {rt['RB0F']} / {rt['HF']} | {rt['RB0P50']} / {rt['HP50']} | |", "",
          "## Paired differences", "", "| pair | old scorer | new scorer |", "|:--|--:|--:|"]
    for p in ("HF:RB0F", "HP50:RB0P50", "HP50:HF", "RB0P50:RB0F"):
        L.append(f"| {p.replace(':', ' - ')} | {f3(P[p]['mean'])} | {f3(sv['pairs'][p]) if sv else '-'} |")
    if sv:
        L += ["", f"New scorer (scorer_v2, decision 453): {sv['scenes']} scenes common to the five reads; means "
              + ", ".join(f"{k} {v:.3f}" for k, v in sv["mean"].items()) + "; zeros " + ", ".join(f"{k} {v}" for k, v in sv["zeros"].items()) + "."]
    else:
        L += ["", "New scorer: not available for an arm (see sv2_*.log)."]
    L += ["", "## HF guard against RB0-F-s0 (design section 5)", ""] + [f"- {k}: {'pass' if v else 'FAIL'}" for k, v in guard.items()]
    L += [f"  (new-scorer mean diff {g_mean:+.3f}; route-turn {rt['HF']} vs {rt['RB0F']}; lateral {lat['HF']} vs {lat['RB0F']}; launch {rd['HF']['launch']}; "
          f"dev ADE {ade['HF']})" if g_mean is not None else "", "", f"**Line: {line}**", "",
          "## Listed scenes (old score / new score, fclass)", "", "| scene | what | " + " | ".join(cells) + " |", "|:--|:--|" + "--:|" * len(cells)]
    L += [f"| {' | '.join(r)} |" for r in srows]
    L += ["", f"New zeros on NTV0-L18-s0's full-score scenes (>= 0.98, {len(full)} scenes): " + ", ".join(f"{k} {v}" for k, v in newbad.items()), "",
          "## Served command vs the rule (G6, >= 99.9 %)", ""]
    for m, c in ca.items():
        L.append(f"- {m}: " + ("; ".join(f"{k} {v['inferences']} inferences, agree {v['agree']:.5f}" for k, v in c["reads"].items())
                               + f" -> {'pass' if c['passed'] else 'FAIL'}" if c else "no record"))
    L += ["", "## Training command record (rm_train cmd_all, full runs)", ""]
    for m, r in trec.items():
        L.append(f"- {m}: rows {r['rows']}, changed {r['changed']}, kept (no memory) {r['kept_no_memory']}; "
                 + ", ".join(f"{g} {p['changed']:.1%} (L/R {p['old_lr']:.1%} -> {p['new_lr']:.1%})" for g, p in r["parts"].items()) if r else f"- {m}: no record")
    Path(f"{a.out}.md").write_text("\n".join(L) + "\n")
    dump(f"{a.out}.json", dict(line=line, guard=guard, d_new=d_new, d_old=d_old, lat=lat, route_turn=rt, launch={k: rd[k]["launch"] for k in rd},
                               old=dict(scenes=common, mean={k: rd[k]["mean"] for k in rd}, pairs={p: P[p]["mean"] for p in P}), new=sv, rpc=rpc,
                               scenes=srows, ntv0_full_new_zeros=newbad, dev_ade=ade, cmdagree=ca, train_cmd=trec))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = ap.add_subparsers(dest="sub", required=True)
    p = s.add_parser("smoke")
    p.add_argument("--tag", required=True), p.add_argument("--out", required=True)
    p = s.add_parser("verdict")
    p.add_argument("--dir", required=True), p.add_argument("--out", required=True)
    a = ap.parse_args()
    dict(smoke=cmd_smoke, verdict=cmd_verdict)[a.sub](a)


if __name__ == "__main__":
    main()
