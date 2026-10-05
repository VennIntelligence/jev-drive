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


if __name__ == "__main__":
    {"extract": extract, "report": report}[sys.argv[1]]()
