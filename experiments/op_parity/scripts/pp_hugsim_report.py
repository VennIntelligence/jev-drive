"""Tables of results/hugsim_spin10.md: the op_parity arms on the 10 PR #57 Cinque spinner scenarios (experiments/hugsim/scripts/derot_spin10.txt),
same harness, scoring and classes as experiments/hugsim/results/wajepa_ref.md (spin = heading error >= 60 deg vs the recorded route,
experiments/hugsim/scripts/spin_analysis.py; launch stall = peak speed over the first 40 steps < 1.6 m/s; class = spin, else the runner's end).

  extract  (box, envs/hugsim)  pp_hugsim_report.py extract     per-run behaviour of the pp-* tags ($DATA_DIR/runs/op_parity/hugsim/results.csv)
                               and of the stored Cinque reruns (cinque-fixed-base, runs/hugsim-derot) -> results/hugsim_spin10/extract.csv,
                               runs.csv (the runner's rows of those tags)
  report   (Mac)               pp_hugsim_report.py report      -> results/hugsim_spin10/tables.md
Reference rows: Cinque PR #57 `cinque-fixed` (2026-09-25) and WA-JEPA from experiments/hugsim/results (scored_op / scored_wajepa,
wajepa_ref/wajepa_extract.csv).
A second argument `spec` (extract spec / report spec) reads the preset-`spec` arms (tags pp-spec-*, scripts/pp_hugsim_spec.txt: the 10 spinners
+ the 19 other max_steps scenarios of decision 118's opctrl run) with the stored decision-118 run `cinque-opctrl` (runs/opctrl/closed) as
reference -> results/hugsim_spec/.
A third argument `full` (extract full / report full) reads the full run: all 64 scenarios (derot_all64.txt) for P0 and the checkpoints
P1-F-s0 ... P3-F-s1 (env FULL_TAGS, comma separated, overrides the arm list; P0 first), both presets (tags pp-<TAG> / pp-spec-<TAG>)
-> results/hugsim_full/ (extract.csv, speeds.json.gz, tables.md and the csv behind every table).
"""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
HR = REPO / "experiments/hugsim/results"
OUT = REPO / "experiments/op_parity/results/hugsim_spin10"
MODE = sys.argv[2] if len(sys.argv) > 2 else "exam"
if MODE == "spec":
    OUT = REPO / "experiments/op_parity/results/hugsim_spec"
    SPIN10 = [Path(p).stem for p in open(REPO / "experiments/op_parity/scripts/pp_hugsim_spec.txt").read().split()]
    ARMS = ["pp-spec-P0", "pp-spec-P1-s0", "pp-spec-P2-s0"]
    REF_TAG, REF_CSV = "cinque-opctrl", "runs/opctrl/closed/results.csv"
else:
    SPIN10 = [Path(p).stem for p in open(REPO / "experiments/hugsim/scripts/derot_spin10.txt").read().split()]
    ARMS = ["pp-P0", "pp-P1-s0", "pp-P2-s0", "pp-P3-s0"]
    REF_TAG, REF_CSV = "cinque-fixed-base", "runs/hugsim-derot/results.csv"


def extract():
    sys.path.insert(0, str(REPO / "experiments/hugsim/scripts"))
    import spin_analysis as SA
    D = Path(os.environ["DATA_DIR"])
    OUT.mkdir(parents=True, exist_ok=True)
    routes_json = D / "runs/op_parity/hugsim/routes.json"
    if not routes_json.exists():
        subprocess.run([sys.executable, str(REPO / "experiments/hugsim/scripts/spin_export_routes.py"), str(HR / "hugsim-exam/scored_op.csv"),
                        str(routes_json)], check=True)
    routes = json.load(open(routes_json))
    src = [(D / "runs/op_parity/hugsim/results.csv", lambda t: t in ARMS),
           (D / REF_CSV, lambda t: t == REF_TAG)]
    runs, rows = [], []
    for path, want in src:
        last = {}
        for r in csv.DictReader(open(path)):
            if want(r["tag"]) and r["scenario"] in SPIN10 and r["end"] != "crash":
                last[(r["tag"], r["scenario"])] = r                     # the latest finished row of a (tag, scenario)
        for (tag, sc), r in sorted(last.items()):
            runs.append(r)
            d = Path(r["run_dir"])
            pos, th, v, steer, plans = SA.load_run(d, r["agent"])
            res, ser = SA.analyse(pos, th, v, steer, plans, routes[r["scene"]])
            row = dict(tag=tag, scenario=sc, max_abs_e=res["max_abs_e"], spin=bool(res["spin"]), k60=res.get("k60", -1), v_max40=float(v[:40].max()),
                       v_max=float(v.max()), v_end=float(v[-1]), standing=float((v < 0.3).mean()), n=len(v))
            zf = d / "zs_steps.jsonl"
            par = [json.loads(x).get("parity") for x in open(zf)][1:] if zf.exists() else []
            par = [p for p in par if p]
            if par:
                row.update(bias_rms_mean=float(np.mean([p["bias_rms"] for p in par])), bias_rtt_ms=float(np.median([p["rtt_ms"] for p in par])),
                           parity_steps=len(par))
            rows.append(row)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / "extract.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    with open(OUT / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(runs[0]))
        w.writeheader()
        w.writerows(runs)
    print(len(rows), "rows ->", OUT)


def report():
    import pandas as pd
    ex = pd.read_csv(OUT / "extract.csv")
    runs = pd.read_csv(OUT / "runs.csv")
    wex = pd.read_csv(HR / "wajepa_ref/wajepa_extract.csv")
    ref = {"cinque-fixed (stored)": (pd.read_csv(HR / "hugsim-exam/scored_op.csv").query("tag == 'cinque-fixed'"), wex[wex.tag == "cinque-fixed"]),
           "wajepa": (pd.read_csv(HR / "hugsim-exam/scored_wajepa.csv").query("tag == 'wajepa'"), wex[wex.tag == "wajepa"])}
    arms = {}
    for t in [REF_TAG] + ARMS:
        r, e = runs[runs.tag == t], ex[ex.tag == t]
        if len(r):
            arms[t] = (r, e)
    if MODE != "spec":
        arms.update(ref)
    tab = {}
    for name, (r, e) in arms.items():
        r = r[r.scenario.isin(SPIN10)].drop_duplicates("scenario", keep="last").set_index("scenario")
        e = e[e.scenario.isin(SPIN10)].drop_duplicates("scenario", keep="last").set_index("scenario")
        d = r[["hdscore", "end", "steps", "rc"]].join(e[["spin", "max_abs_e", "v_max40", "standing"]], how="left")
        cls = d["end"].astype(str).replace({"max_steps": "stuck", "bg_collision": "bg_coll", "fg_collision": "fg_coll"})
        cls[d["spin"].astype(bool)] = "spin"
        d["cls"] = cls
        tab[name] = d
    names = {"cinque-fixed-base": "Cinque rerun (cinque-fixed-base, 2026-10-03)", "cinque-fixed (stored)": "Cinque PR #57 (cinque-fixed, 2026-09-25)",
             "wajepa": "WA-JEPA", "pp-P0": "P0 shipped, through the parity path", "pp-P1-s0": "P1 fine-tuned, no inputs",
             "pp-P2-s0": "P2 + ego / pose / command", "pp-P3-s0": "P3 + side / rear cameras",
             "cinque-opctrl": "Cinque spec (cinque-opctrl, decision 118)", "pp-spec-P0": "P0 shipped, parity path",
             "pp-spec-P1-s0": "P1 fine-tuned, no inputs", "pp-spec-P2-s0": "P2 + ego / pose / command"}
    def block(title, sel):
        L = [title, "| arm | n | HD-Score | RC | spins >= 60 deg | launch stalls (v_max first 40 steps < 1.6 m/s) | stuck (max_steps end) | "
             "standing share (v < 0.3 m/s) | complete | fg coll | bg coll | off_route | max heading err median (deg) |",
             "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for k, d in tab.items():
            d = d[d.index.isin(sel)]
            c, stuck = d["cls"], int((d["end"] == "max_steps").sum())
            n = lambda s: int((c == s).sum())  # noqa: E731
            L.append(f"| {names.get(k, k)} | {len(d)} | {d.hdscore.mean():.3f} | {d.rc.mean():.3f} | {int(d.spin.astype(bool).sum())} | {int((d.v_max40 < 1.6).sum())} | "
                     f"{stuck} | {d.standing.mean():.2f} | {n('complete')} | {n('fg_coll')} | {n('bg_coll')} | {n('off_route')} | {d.max_abs_e.median():.0f} |")
        return L
    hdr = "spec preset (tree opctrl)" if MODE == "spec" else "PR #57 controller"
    L = block(f"## T1 per arm on all {len(SPIN10)} scenarios (one run each, {hdr}, 400-step cap; class = spin, else end)\n", SPIN10)
    if MODE == "spec":
        sp = [Path(p).stem for p in open(REPO / "experiments/hugsim/scripts/derot_spin10.txt").read().split()]
        L += [""] + block("## T1a the 10 PR #57 spinner scenarios\n", sp) + [""] + block("## T1b the 19 other decision-118 stuck scenarios\n",
                                                                                            [s for s in SPIN10 if s not in sp])
    L += ["", "## T2 per scenario: HD-Score / class / max heading error (deg) / v_max over the first 40 steps (m/s)\n"]
    cols = list(tab)
    L.append("| scenario | " + " | ".join(names.get(k, k) for k in cols) + " |")
    L.append("|---|" + "---|" * len(cols))
    for sc in SPIN10:
        cells = []
        for k in cols:
            d = tab[k]
            if sc in d.index:
                x = d.loc[sc]
                cells.append(f"{x.hdscore:.3f} {x.cls} {x.max_abs_e:.0f} {x.v_max40:.1f}")
            else:
                cells.append("-")
        L.append(f"| {sc} | " + " | ".join(cells) + " |")
    pe = ex[ex.tag.isin(ARMS)]
    if "bias_rms_mean" in pe:
        L += ["", "## T3 parity path per arm (means over runs)\n", "| arm | parity steps | bias rms | bias server round trip median (ms) |", "|---|--:|--:|--:|"]
        for t, g in pe.groupby("tag"):
            L.append(f"| {names.get(t, t)} | {int(g.parity_steps.sum())} | {g.bias_rms_mean.mean():.4f} | {g.bias_rtt_ms.median():.1f} |")
    (OUT / "tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


# ---------------------------------------------------------------------------------------------------------------- full run
FULL = Path(os.environ.get("FULL_OUT") or REPO / "experiments/op_parity/results/hugsim_full")
FULL_TAGS = os.environ.get("FULL_TAGS", "P0,P1-F-s0,P1-F-s1,P2-F-s0,P2-F-s1,P3-F-s0,P3-F-s1").split(",")
PRESETS = {"exam": "pp-", "spec": "pp-spec-"}
STOP_V = 0.5            # P0 "stopped at the collision step": P0 speed there < 0.5 m/s
STAND = 0.5             # ... or P0 standing share (v < 0.3 m/s) > 0.5, or P0 ended max_steps


def extract_full():
    import gzip
    sys.path.insert(0, str(REPO / "experiments/hugsim/scripts"))
    import spin_analysis as SA
    D = Path(os.environ["DATA_DIR"])
    FULL.mkdir(parents=True, exist_ok=True)
    scen = {Path(p).stem for p in open(REPO / "experiments/hugsim/scripts/derot_all64.txt").read().split()}
    routes_json = D / "runs/op_parity/hugsim/routes.json"
    if not routes_json.exists():
        subprocess.run([sys.executable, str(REPO / "experiments/hugsim/scripts/spin_export_routes.py"), str(HR / "hugsim-exam/scored_op.csv"),
                        str(routes_json)], check=True)
    routes = json.load(open(routes_json))
    want = {pfx + t: (pr, t) for t in FULL_TAGS for pr, pfx in PRESETS.items()}
    last = {}
    for r in csv.DictReader(open(D / "runs/op_parity/hugsim/results.csv")):
        if r["tag"] in want and r["scenario"] in scen and r["end"] != "crash":
            last[(r["tag"], r["scenario"])] = r                         # the latest finished row of a (tag, scenario)
    rows, speeds = [], {}
    for (tag, sc), r in sorted(last.items()):
        d = Path(r["run_dir"])
        pos, th, v, steer, plans = SA.load_run(d, r["agent"])
        res, _ = SA.analyse(pos, th, v, steer, plans, routes[r["scene"]])
        rows.append(dict(r, preset=want[tag][0], arm=want[tag][1], max_abs_e=res["max_abs_e"], spin=bool(res["spin"]), k60=res.get("k60", -1),
                         v_max40=float(v[:40].max()), v_max=float(v.max()), v_end=float(v[-1]), standing=float((v < 0.3).mean()), n=len(v)))
        speeds[f"{tag}|{sc}"] = [round(float(x), 2) for x in v]
    with open(FULL / "extract.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with gzip.open(FULL / "speeds.json.gz", "wt") as f:
        json.dump(speeds, f)
    print(len(rows), "rows ->", FULL)


def report_full():
    import gzip
    import pandas as pd
    sys.path.insert(0, str(REPO))
    from jevdrive import stats
    ex = pd.read_csv(FULL / "extract.csv")
    speeds = json.load(gzip.open(FULL / "speeds.json.gz", "rt"))
    W = pd.read_csv(HR / "hugsim-exam/scored_wajepa.csv").query("tag == 'wajepa'").drop_duplicates("scenario", keep="last").set_index("scenario")
    WX = pd.read_csv(HR / "wajepa_ref/wajepa_extract.csv").query("tag == 'wajepa'").drop_duplicates("scenario", keep="last").set_index("scenario")
    W = W.join(WX[["spin", "v_max40", "standing"]])
    SC = list(W.index)                                                  # the 64, WA-JEPA's order
    meta = W[["dataset", "difficulty", "scene"]]
    ex["cls"] = ex.end.astype(str).replace({"max_steps": "stuck", "bg_collision": "bg_coll", "fg_collision": "fg_coll"})
    ex.loc[ex.spin.astype(bool), "cls"] = "spin"
    ex["stall"] = ex.v_max40 < 1.6
    CL = ["complete", "fg_coll", "bg_coll", "off_route", "spin"]
    W["cls"] = W.end.astype(str).replace({"max_steps": "stuck", "bg_collision": "bg_coll", "fg_collision": "fg_coll"})
    W.loc[W.spin.astype(bool), "cls"] = "spin"
    W["stall"] = W.v_max40 < 1.6

    def run(preset, tag):
        return ex[(ex.preset == preset) & (ex.arm == tag)].set_index("scenario").reindex(SC)

    def grp(t):                                                         # "P2-F-s0" -> "P2"; P0 stays
        return t.split("-")[0]
    out_csv = {}
    L = []
    P = L.append
    def md(df, floatfmt=".3f"):
        df = df.copy()
        return ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)] + [
            "| " + " | ".join(f"{x:{floatfmt}}" if isinstance(x, float) else str(x) for x in row) + " |" for row in df.itertuples(index=False)]

    # per-scenario HD vectors: arms, seed-means, WA-JEPA
    hd, units = {}, []                                                  # preset -> name -> Series over SC
    for pr in PRESETS:
        hd[pr] = {"WA-JEPA": W.hdscore.reindex(SC)}
        for t in FULL_TAGS:
            hd[pr][t] = run(pr, t).hdscore
        for g in dict.fromkeys(grp(t) for t in FULL_TAGS if grp(t) != "P0"):
            ts = [t for t in FULL_TAGS if grp(t) == g]
            if len(ts) > 1:
                hd[pr][g + " seed-mean"] = pd.concat([hd[pr][t] for t in ts], axis=1).mean(1, skipna=False)
    # ---- T1 per arm
    def row(name, d):
        d = d[d.hdscore.notna()]
        b = stats.bootstrap(d.hdscore.values)
        c = d.cls
        r = dict(arm=name, n=int(d.hdscore.notna().sum()), HD=stats.fmt(b), NC=d.nc.mean(), DAC=d.dac.mean(), TTC=d.ttc.mean(), Comfort=d.c.mean(), RC=d.rc.mean(),
                 spins=int(d.spin.astype(bool).sum()), stuck=int((d.end == "max_steps").sum()), stalls=int(d.stall.sum()), standing=d.standing.mean())
        r.update({k: int((c == k).sum()) for k in CL})
        return r
    for pr in PRESETS:
        rows = [dict(row("WA-JEPA", W.reindex(SC)), standing=W.standing.mean())]
        for t in FULL_TAGS:
            rows.append(row(t, run(pr, t)))
        for g in dict.fromkeys(grp(t) for t in FULL_TAGS if grp(t) != "P0"):
            ts = [t for t in FULL_TAGS if grp(t) == g]
            if len(ts) < 2:
                continue
            rs = pd.DataFrame([row(t, run(pr, t)) for t in ts]).drop(columns=["arm", "HD"])
            m = rs.mean().to_dict()
            m.update(arm=g + " seed-mean", HD=stats.fmt(stats.bootstrap(hd[pr][g + " seed-mean"].values)), n=int(hd[pr][g + " seed-mean"].notna().sum()))
            rows.append(m)
        df = pd.DataFrame(rows)
        for c in ("spins", "stuck", "stalls", *CL):                    # seed-mean rows carry means of counts
            df[c] = [f"{x:.1f}" if isinstance(x, float) and x != int(x) else int(x) for x in df[c]]
        out_csv[f"arms_{pr}"] = df
        P(f"## T1 per arm, preset `{pr}` ({'tree fixed, PR #57 controller' if pr == 'exam' else 'tree opctrl'}; one run per scenario, 400-step cap)\n")
        P("HD = mean over scenarios with bootstrap 95% CI (jevdrive.stats, unit = scenario); NC / DAC / TTC / Comfort / RC are scenario means; spins = heading error >= 60 deg "
          "(spin_analysis.py); stuck = max_steps end; stalls = peak speed over the first 40 steps < 1.6 m/s; standing = mean share of steps with v < 0.3 m/s; "
          "end classes: spin takes precedence over the runner's end. Seed-mean rows: HD of the per-scenario seed mean, other columns mean over seeds (counts are means).\n")
        P("| arm | n | HD-Score [95% CI] | NC | DAC | TTC | Comfort | RC | spins | stuck | stalls | standing | " + " | ".join(CL) + " |")
        P("|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|" + "--:|" * len(CL))
        for r in df.itertuples(index=False):
            P(f"| {r.arm} | {r.n} | {r.HD} | {r.NC:.3f} | {r.DAC:.3f} | {r.TTC:.3f} | {r.Comfort:.3f} | {r.RC:.3f} | {r.spins} | {r.stuck} | {r.stalls} | {r.standing:.2f} | "
              + " | ".join(str(getattr(r, k)) for k in CL) + " |")
        miss = {t: [s for s in SC if pd.isna(run(pr, t).hdscore.get(s))] for t in FULL_TAGS}
        miss = {t: m for t, m in miss.items() if m}
        P("\nMissing scenarios (no finished row): " + ("; ".join(f"{t}: {len(m)}" + (f" ({', '.join(m)})" if len(m) <= 8 else "") for t, m in miss.items()) if miss else "none") + ".\n")
        out_csv[f"missing_{pr}"] = pd.DataFrame([dict(arm=t, scenario=x) for t, m in miss.items() for x in m], columns=["arm", "scenario"])
    # ---- T2 paired
    comps = []
    for pr in PRESETS:
        h = hd[pr]
        names = list(h)

        def resolve(g):                                                 # group -> (label, series): seed-mean, else the sole tag
            if g + " seed-mean" in h:
                return g + " seed-mean", h[g + " seed-mean"]
            ts = [t for t in FULL_TAGS if grp(t) == g]
            return (ts[0], h[ts[0]]) if len(ts) == 1 else (None, None)
        for ga, gb in (("P2", "P1"), ("P3", "P1"), ("P2", "P0"), ("P2H10", "P2"), ("P2H3", "P2"), ("P2H10", "P0")):
            (na, a), (nb, b) = resolve(ga), resolve(gb)
            if a is not None and b is not None:
                comps.append((pr, a, b, na, nb))
            for t in FULL_TAGS:                                         # matched seeds
                tb = "P0" if gb == "P0" else t.replace(ga, gb, 1)
                if grp(t) == ga and tb in h and len([x for x in FULL_TAGS if grp(x) == ga]) > 1:
                    comps.append((pr, h[t], h[tb], t, tb))
        for n in names:
            if n != "WA-JEPA":
                comps.append((pr, h[n], h["WA-JEPA"], n, "WA-JEPA"))
    rows = []
    for pr, a, b, na, nb in comps:
        ok = a.notna() & b.notna()
        if ok.sum() < 3:
            continue
        sc = [s for s in SC if ok[s]]
        x, y = a[sc].values, b[sc].values
        d = x - y
        r1 = stats.paired(x, y)
        r2 = stats.paired(x, y, groups=meta.loc[sc, "scene"].values)
        rows.append(dict(preset=pr, a=na, b=nb, n=len(sc), diff=stats.fmt(r1), diff_scene=stats.fmt(r2), scenes=r2["units"], better=int((d > .02).sum()), worse=int((d < -.02).sum()),
                         tie=int((abs(d) <= .02).sum()), mean_a=r1["mean_a"], mean_b=r1["mean_b"]))
    pdf = pd.DataFrame(rows)
    out_csv["paired"] = pdf
    P("## T2 paired HD-Score differences (a - b; unit = scenario, and scene clusters with the ratio-of-sums bootstrap; better / worse / tie with |d| < 0.02)\n")
    P("| preset | a | b | n | mean a | mean b | a - b, scenarios [95% CI] | a - b, scenes [95% CI] | scenes | better | worse | tie |")
    P("|---|---|---|--:|--:|--:|---|---|--:|--:|--:|--:|")
    for r in pdf.itertuples(index=False):
        P(f"| {r.preset} | {r.a} | {r.b} | {r.n} | {r.mean_a:.3f} | {r.mean_b:.3f} | {r.diff} | {r.diff_scene} | {r.scenes} | {r.better} | {r.worse} | {r.tie} |")
    P("")
    # ---- T3 foreground collisions vs P0 stopping
    def p0_state(pr, sc, step):
        r = run(pr, "P0").loc[sc]
        if pd.isna(r.hdscore):
            return None
        v = speeds.get(f"{PRESETS[pr]}P0|{sc}", [])
        v_at = v[step - 1] if 0 < step <= len(v) else np.nan
        stopped = (r.end == "max_steps") or (r.standing > STAND) or (v_at < STOP_V)
        if stopped:
            cat = "P0 stuck / stopped there"
        elif r.end in ("fg_collision", "bg_collision"):
            cat = "P0 also collides"
        else:
            cat = "P0 completes or other"
        return dict(p0_end=r.end, p0_standing=r.standing, p0_v_at_step=v_at, p0_steps=r.steps, cat=cat)
    CATS = ["P0 stuck / stopped there", "P0 also collides", "P0 completes or other"]
    det, summ = [], []
    for pr in PRESETS:
        for t in FULL_TAGS:
            d = run(pr, t)
            fg = [s for s in SC if d.end.get(s) == "fg_collision"]
            cnt = dict.fromkeys(CATS, 0)
            for s in fg:
                step = int(d.steps[s])
                v = speeds.get(f"{PRESETS[pr]}{t}|{s}", [])
                st = p0_state(pr, s, step) if t != "P0" else None
                if st:
                    cnt[st["cat"]] += 1
                det.append(dict(preset=pr, arm=t, scenario=s, difficulty=meta.difficulty[s], step=step, v_arm=v[min(step, len(v)) - 1] if v else np.nan,
                                **(st or dict(p0_end="", p0_standing=np.nan, p0_v_at_step=np.nan, p0_steps=np.nan, cat="(P0 itself)"))))
            summ.append(dict(preset=pr, arm=t, n=int(d.hdscore.notna().sum()), fg_coll=len(fg), **cnt))
    ddf, sdf = pd.DataFrame(det), pd.DataFrame(summ)
    out_csv["fg_detail"], out_csv["fg_summary"] = ddf, sdf
    P("## T3 foreground collisions: does P0 stop where an arm collides?\n")
    P(f"Definition. An (arm, scenario) run with end `fg_collision` at step k is assigned, using the P0 run of the same preset and scenario, to the first matching class: "
      f"**P0 stuck / stopped there** = P0 ended max_steps, or P0 standing share (v < 0.3 m/s) > {STAND}, or P0 speed at step k < {STOP_V} m/s (P0 reached step k); "
      "**P0 also collides** = P0 ended fg_collision or bg_collision (and is not stopped); **P0 completes or other** = the rest (complete, off_route, P0 missing). "
      "Collision step k = the run's `steps`; speed = ego speed (m/s) in that step.\n")
    P("| preset | arm | n | fg collisions | P0 stuck / stopped there | P0 also collides | P0 completes or other |\n|---|---|--:|--:|--:|--:|--:|")
    for _, r in sdf.iterrows():
        P(f"| {r.preset} | {r.arm} | {r.n} | {r.fg_coll} | " + (" | ".join(str(r[k]) for k in CATS) if r.arm != "P0" else "- | - | -") + " |")
    # P0's own end in the same scenarios, whatever the class
    e2 = ddf[ddf.arm != "P0"].assign(p0_ended=lambda x: x.p0_end.replace({"fg_collision": "P0 fg_coll", "bg_collision": "P0 bg_coll", "max_steps": "P0 max_steps",
                                                                         "complete": "P0 other", "off_route": "P0 other"}))
    t3b = e2.pivot_table(index=["preset", "arm"], columns="p0_ended", values="step", aggfunc="size", fill_value=0).reset_index()
    out_csv["fg_p0_end"] = t3b
    P("\nThe same collisions by how P0 itself ended in that scenario (a stopped P0 often ended in its own collision at near-zero speed):\n")
    P("\n".join(md(t3b)))
    P("")
    P("\nPer-collision rows (arm, scenario, step, speeds, P0 state) are in `fg_detail.csv` / `fg_detail.md`.\n")
    # inverse view
    inv = []
    for pr in PRESETS:
        p0 = run(pr, "P0")
        stuck_sc = [s for s in SC if pd.notna(p0.hdscore.get(s)) and (p0.end[s] == "max_steps" or p0.standing[s] > STAND)]
        for t in FULL_TAGS:
            d = run(pr, t)
            c = d.loc[stuck_sc, "end"].astype(str)
            inv.append(dict(preset=pr, arm=t, p0_stuck=len(stuck_sc), complete=int((c == "complete").sum()), fg_coll=int((c == "fg_collision").sum()), bg_coll=int((c == "bg_collision").sum()),
                            off_route=int((c == "off_route").sum()), still_stuck=int((c == "max_steps").sum()), missing=int(d.loc[stuck_sc, "hdscore"].isna().sum()),
                            hd_on_set=d.loc[stuck_sc, "hdscore"].mean(), hd_p0=p0.loc[stuck_sc, "hdscore"].mean()))
    idf = pd.DataFrame(inv)
    out_csv["p0_stuck_outcomes"] = idf
    P("## T4 the inverse view: what each arm does in the scenarios where P0 is stuck\n")
    P(f"P0-stuck set per preset = P0 ended max_steps or P0 standing share > {STAND} (whole run).\n")
    P("| preset | arm | P0-stuck scenarios | completes | fg collision | bg collision | off_route | still stuck (max_steps) | missing | HD on the set | P0 HD on the set |\n|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for r in idf.itertuples(index=False):
        P(f"| {r.preset} | {r.arm} | {r.p0_stuck} | {r.complete} | {r.fg_coll} | {r.bg_coll} | {r.off_route} | {r.still_stuck} | {r.missing} | {r.hd_on_set:.3f} | {r.hd_p0:.3f} |")
    P("")
    # ---- T5 vs WA-JEPA per scenario
    P("## T5 against WA-JEPA, scenario by scenario (HD diff = arm - WA-JEPA; seed-mean arms use the per-scenario mean over seeds)\n")
    P("Paired diffs, wins / losses / ties: T2 rows with b = WA-JEPA.\n")
    big = []
    for pr in PRESETS:
        for n, s in hd[pr].items():
            if n == "WA-JEPA":
                continue
            d = (s - hd[pr]["WA-JEPA"]).dropna()
            for sc in d.index:
                if abs(d[sc]) > 0.3:
                    big.append(dict(preset=pr, arm=n, scenario=sc, difficulty=meta.difficulty[sc], hd_arm=s[sc], hd_wajepa=hd[pr]["WA-JEPA"][sc], diff=d[sc]))
    bdf = pd.DataFrame(big)
    out_csv["vs_wajepa_big"] = bdf
    P("Scenarios with |diff| > 0.3 (arms that beat / lose; full list in `vs_wajepa_big.csv`):\n")
    P("| preset | arm | beats WA-JEPA by > 0.3 | loses to WA-JEPA by > 0.3 |\n|---|---|---|---|")
    short = lambda s: s.replace("scene-", "")  # noqa: E731
    for pr in PRESETS:
        for n in hd[pr]:
            if n == "WA-JEPA":
                continue
            g = bdf[(bdf.preset == pr) & (bdf.arm == n)] if len(bdf) else bdf
            if not len(g):
                P(f"| {pr} | {n} | - | - |")
                continue
            up = g[g["diff"] > 0].sort_values("diff", ascending=False)
            dn = g[g["diff"] < 0].sort_values("diff")
            fm = lambda q: f"{len(q)}: " + ", ".join(f"{short(r.scenario)} ({r.diff:+.2f})" for r in q.itertuples())  # noqa: E731
            P(f"| {pr} | {n} | {fm(up)} | {fm(dn)} |")
    P("")
    P("### T5b HD-Score by difficulty and by dataset (n = 16 per cell; diff vs WA-JEPA with bootstrap CI, unit = scenario)\n")
    strat = []
    for pr in PRESETS:
        for key, levels in (("difficulty", ["easy", "medium", "hard", "extreme"]), ("dataset", ["nuscenes", "kitti360", "waymo", "pandaset"])):
            for lv in levels:
                sel = [s for s in SC if meta[key][s] == lv]
                for n, s in hd[pr].items():
                    if n == "WA-JEPA":
                        strat.append(dict(preset=pr, by=key, level=lv, arm=n, n=len(sel), hd=s[sel].mean(), diff="-"))
                        continue
                    ok = [x for x in sel if pd.notna(s[x])]
                    r = stats.paired(s[ok].values, hd[pr]["WA-JEPA"][ok].values) if len(ok) > 2 else None
                    strat.append(dict(preset=pr, by=key, level=lv, arm=n, n=len(ok), hd=s[ok].mean() if ok else np.nan, diff=stats.fmt(r) if r else "-"))
    sdf2 = pd.DataFrame(strat)
    out_csv["by_stratum"] = sdf2
    for pr in PRESETS:
        for key in ("difficulty", "dataset"):
            g = sdf2[(sdf2.preset == pr) & (sdf2.by == key)]
            arms = list(dict.fromkeys(g.arm))
            P(f"preset `{pr}`, by {key}: cell = HD mean (diff vs WA-JEPA [CI])\n")
            P(f"| {key} | " + " | ".join(arms) + " |\n|---|" + "---|" * len(arms))
            for lv in dict.fromkeys(g.level):
                cells = []
                for a in arms:
                    x = g[(g.level == lv) & (g.arm == a)].iloc[0]
                    cells.append(f"{x.hd:.3f}" + ("" if x.n == 16 else f" [n={x.n}]") + ("" if x["diff"] == "-" else f" ({x['diff']})"))
                P(f"| {lv} | " + " | ".join(cells) + " |")
            P("")
    FULL.mkdir(parents=True, exist_ok=True)
    for k, df in out_csv.items():
        df.to_csv(FULL / f"{k}.csv", index=False)
    cols = ["preset", "arm", "scenario", "difficulty", "step", "v_arm", "cat", "p0_end", "p0_standing", "p0_v_at_step", "p0_steps"]
    (FULL / "fg_detail.md").write_text("# Per-collision rows\n\n" + "\n".join(md(ddf[cols].round(2), ".2f")) + "\n")
    (FULL / "tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    if MODE == "full":
        {"extract": extract_full, "report": report_full}[sys.argv[1]]()
    else:
        {"extract": extract, "report": report}[sys.argv[1]]()
