#!/usr/bin/env python3
"""ROUTE-MEM closed-loop read-out (plans/2026-10-12-route-mem-prereg.md, "Registered lines"). Box, AlpaSim's venv; every reused script is
called unedited.

  gate     the gates before any rollout: G-init / G-dev / G-rely from the two trainings, G-serve from the served check -> $W/gates.json;
           exit 0 iff the reads may start (both seeds pass G-init and G-dev, at least one passes G-rely, G-serve (b) and the eager
           plan equality pass)
  report   (1) onpol_long's ol_report.py: YRM against YS2G (control) and against YS-L18 (seed means); (2) merged views of the chunked reads and
           zeros_mech's zm_read.py collect / report through ol_zm.py (Run root runs/onpol_long/zm), ol_dirraster.py's line counts for YRM / YS2G /
           YS-L18; (3) the registered 98-window split (lines 1-2): road-exit zeros on the 38 / 34 held scenes where local_map's start-behind
           route covers / does not cover 20 m in the window; (4) the served memory's coverage (rm_build.py cover) -> $W/report/
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(R), str(R / "experiments/road_probe/scripts"), str(R / "experiments/onpol_long/scripts")]
from jevdrive.common import data_dir  # noqa: E402

W = data_dir() / "runs/route_mem"
OL = data_dir() / "runs/onpol_long"
H18 = data_dir() / "runs/alpasim_long/held/h18"
PY = data_dir() / "third_party/alpasim/.venv/bin/python"
ARM = ("YRM-L18-s0", "YRM-L18-s1")
YS2G = ("YS2G-L18-s0", "YS2G-L18-s1")
YSL = ("YS-L18-s0", "YS-L18-s1")
REF_DEV = 0.5286                                     # YS-L18 seed-mean dev ADE (DONE of on-train-YS-L18-s0 / s1)
INIT_DEV = 0.5284                                    # dev ADE of YS-L18-s1 at step 0 (YS2G's training record)
EV_CSV = R / "experiments/eyes_vs_head/results/events_P0.csv"
LM_READ = data_dir() / "runs/local_map/read/20261012-183650"


def run_dir(tag):
    return Path(sorted(glob.glob(str(W / f"train-{tag}" / "*/")))[-1])


def cmd_gate(a):
    G = {}
    for t in ARM:
        d = json.loads((run_dir(t) / "DONE").read_text())
        r = json.loads((W / "rely" / f"{t}.json").read_text())
        G[t] = dict(dev0=d["dev0_ade"], dev=d["dev_ade"], dev_nomem=d.get("dev_ade_nomem"), g_init=abs(d["dev0_ade"] - INIT_DEV) <= 1e-4,
                    g_dev=d["dev_ade"] <= REF_DEV + 0.02, rely=r, g_rely=bool(r["read"]))
    s = json.loads((W / "served" / "RMI-L18.json").read_text())
    e = json.loads((W / "served" / "same_RMI-L18-eager.json").read_text())
    G["serve"] = dict(memory=s, eager_same=e, g_serve=bool(s["pass"] and e["share_le_003"] >= 0.99))
    G["pass"] = bool(all(G[t]["g_init"] and G[t]["g_dev"] for t in ARM) and any(G[t]["g_rely"] for t in ARM) and G["serve"]["g_serve"])
    (W / "gates.json").write_text(json.dumps(G, indent=1, default=float))
    print(json.dumps(G, indent=1, default=float))
    raise SystemExit(0 if G["pass"] else 2)


def chunks(tag):
    return sorted(glob.glob(str(W / "read" / tag / "c*/run")))


def merged(tag, dirs):
    """onpol_long report-armb3's merged view of chunked reads (one run dir for zm_read)."""
    V = W / "read" / tag / "merged"
    subprocess.run(["rm", "-rf", str(V)], check=True)
    for s in ("aggregate", "driver-logs", "rollouts"):
        (V / s).mkdir(parents=True, exist_ok=True)
    (V / "generated-user-config-0.yaml").write_bytes((Path(dirs[0]) / "generated-user-config-0.yaml").read_bytes())
    with open(V / "driver-logs/drive.jsonl", "w") as f:
        for d in dirs:
            f.write((Path(d) / "driver-logs/drive.jsonl").read_text())
    for d in dirs:
        for q in glob.glob(f"{d}/rollouts/*"):
            (V / "rollouts" / Path(q).name).symlink_to(q)
    json.dump({"rollouts": [r for d in dirs for r in json.load(open(f"{d}/aggregate/results-summary.json"))["rollouts"]]},
              open(V / "aggregate/results-summary.json", "w"))
    return V


def window_flags():
    """The 98 road-exit zero windows of decisions 272 / 275 -> per held scene: covered (local_map's start-behind newest route reaches 20 m on
    >= half of the window decisions)."""
    import pandas as pd
    import rp as RP
    ev = RP.zero_events()
    e = pd.read_parquet(LM_READ / "est_zero.parquet")
    rows = []
    for x in ev:
        w = e[(e.tag == x["group"]) & (e.scene == x["scene"]) & (e.k >= int(x["k_ev"]) - 4) & (e.k <= int(x["k_ev"]))]
        rows.append(dict(scene=x["scene"], n=len(w), c=float(np.isfinite(w.L2_b_c20).sum())))
    D = pd.DataFrame(rows).groupby("scene").sum()
    return (D.c / D.n >= 0.5).rename("covered")


def split_lines(S, flags):
    """Lines 1-2: seed-mean road-exit (offroad or corridor) zeros of YRM and YS2G on the covered / uncovered window scenes, log-clustered CI of
    the difference (10 000 draws of logs, rng 0)."""
    S = S.copy()
    S["rx"] = ((S.score == 0) & ((S.offroad > 0) | (S.left_corridor_laterally > 0))).astype(float)
    piv = S.pivot(index="scene", columns="tag", values="rx")
    out = {}
    for name, sel in (("covered", flags[flags].index), ("uncovered", flags[~flags].index)):
        P = piv.reindex(sel)
        a, b = P[list(ARM)].mean(1).to_numpy(), P[list(YS2G)].mean(1).to_numpy()
        ok = np.isfinite(a) & np.isfinite(b)
        logs = np.array([s.rsplit("-", 1)[0] for s in P.index])[ok]
        L = np.unique(logs)
        li = np.searchsorted(L, logs)
        d = (a - b)[ok]
        rng = np.random.default_rng(0)
        tot = np.bincount(li, d, len(L))
        draws = np.stack([np.bincount(rng.integers(0, len(L), len(L)), minlength=len(L)) for _ in range(10000)])
        bs = draws @ tot
        out[name] = dict(scenes=int(ok.sum()), arm=float(a[ok].sum()), ys2g=float(b[ok].sum()), diff=float(d.sum()),
                         lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                         arm_seeds={t: float(P[t][ok].sum()) for t in ARM}, ys2g_seeds={t: float(P[t][ok].sum()) for t in YS2G})
    out["line1"] = bool(out["covered"]["arm"] <= out["covered"]["ys2g"] * 2 / 3)
    out["line2"] = bool(abs(out["uncovered"]["diff"]) <= out["uncovered"]["ys2g"] / 3)
    return out


def cmd_report(a):
    import pandas as pd
    import ol_dirraster as DR
    O = W / "report"
    O.mkdir(parents=True, exist_ok=True)
    ys0 = ",".join(sorted(glob.glob(str(OL / "read/c15-YS-L18-s0/c*/run"))))
    runs = [f"{t}@18={','.join(chunks(t))}" for t in ARM] + [f"{t}@18={OL / 'read' / t / 'run'}" for t in YS2G] + \
        [f"YS-L18-s0@18={ys0}", f"YS-L18-s1@18={OL / 'read/YS-L18-s1/run'}"]
    subprocess.run([str(PY), str(R / "experiments/onpol_long/scripts/ol_report.py"), "--family", "Y", "--arm", "RM", "--ctl", "S2G",
                    "--base-tags", "YS-L18-s0,YS-L18-s1", "--ref-mean", "--runs", *runs, "--out", str(O / "Y-RM")], check=True)
    S = pd.read_csv(O / "Y-RM/per_scene.csv")
    S["tag"] = S.label.str.split("@").str[0]
    flags = window_flags()
    res = dict(ol_report=json.loads((O / "Y-RM/report_YRM.json").read_text()), split=split_lines(S, flags))
    # zeros_mech mechanism counts (zm_read collect / report unedited through ol_zm.py; Run root runs/onpol_long/zm)
    zr = [f"{t}={merged(t, chunks(t))}" for t in ARM]
    zr += [f"YS-L18-s0={OL / 'read/c15-YS-L18-s0/merged'}", f"YS-L18-s1={OL / 'read/YS-L18-s1/run'}"]
    zr += [f"{t}={OL / 'read' / t / 'run'}" for t in YS2G] + [f"YR10m10-F-s0={H18 / 'YR10m10-F-s0'}", f"YR10m10-F-s1={H18 / 'YR10m10-F-s1'}"]
    ZW = R / "experiments/onpol_long/scripts/ol_zm.py"
    subprocess.run([str(PY), str(ZW), "collect", "--runs", *zr], check=True, env={**__import__("os").environ, "OMP_NUM_THREADS": "1"})
    CD = sorted(glob.glob(str(OL / "zm/zeros_mech/collect/*/")))[-1]
    rows = sorted(glob.glob(str(data_dir() / "runs/zeros_mech/rows/*/")))[-1]
    subprocess.run([str(PY), str(ZW), "report", "--collect", CD, "--rows", rows, "--out", str(O / "zm"), "--figs", str(O / "zm/figs")], check=False)
    res["zm_collect"] = CD
    if (O / "zm/per_rollout_reads.csv.gz").exists():
        DR.GROUPS = {"YRM": ARM, "YS2G": YS2G, "YS": YSL, "base": ("YR10m10-F-s0",)}
        DR.cmd_lines(argparse.Namespace(reads=str(O / "zm"), collect=CD, out=str(O / "lines.json")))
        res["mech"] = json.loads((O / "lines.json").read_text())
    if (W / "cover").exists():
        C = pd.concat([pd.read_parquet(f) for f in sorted((W / "cover").glob("YRM-*.parquet"))])
        res["cover"] = dict(decisions=len(C), c20=float(C.c20.mean()), first_med=float(C["first"].median()),
                            by_k={int(k): float(v) for k, v in C.groupby("k").c20.mean().items()})
    arm, ctl = res["ol_report"]["arm"], res["ol_report"]["control"]
    m = res.get("mech", {})
    L = {"1 covered window scenes: road-exit zeros <= 2/3 x YS2G": res["split"]["line1"],
         "2 uncovered window scenes: |YRM - YS2G| <= 1/3 x YS2G": res["split"]["line2"],
         "3 zeros <= 41.5": arm["zero"] <= 41.5, "4 GT distance <= 1.276 m": arm["gt"] <= 1.276,
         "5 early-inside across-traffic zeros <= YS2G": (m.get("YRM", {}).get("early_inside_across", np.inf) <= m.get("YS2G", {}).get("early_inside_across", -np.inf)),
         "6 at-fault zeros <= YS2G": arm["z_atf"] <= ctl["z_atf"]}
    res["lines"] = {k: bool(v) for k, v in L.items()}
    res["adopted"] = bool(L["3 zeros <= 41.5"] and L["4 GT distance <= 1.276 m"] and L["6 at-fault zeros <= YS2G"])
    (O / "route_mem.json").write_text(json.dumps(res, indent=1, default=float))
    sp = res["split"]
    md = ["# ROUTE-MEM: held 452 at 18 decisions, YRM (YS2G's recipe + route-memory branch) against YS2G and YS-L18", "",
          (O / "Y-RM/report_YRM.md").read_text(), "",
          "## 98-window split (registered lines 1-2; road-exit = offroad or corridor zeros, seed means, log-clustered CI of the total difference)", "",
          "| scenes | n | YRM | YS2G | YRM - YS2G [CI] |", "|:--|--:|--:|--:|:--|"]
    for k in ("covered", "uncovered"):
        x = sp[k]
        md.append(f"| {k} | {x['scenes']} | {x['arm']:.1f} | {x['ys2g']:.1f} | {x['diff']:+.1f} [{x['lo']:+.1f}, {x['hi']:+.1f}] |")
    if m:
        md += ["", "## Mechanism counts (zeros_mech zm_read, ol_dirraster.py lines; per seed mean)", "", "| group | zeros | early-inside across | late-outside kerb | oncoming 2 s before | straight corridor past log end |",
               "|:--|--:|--:|--:|:--|--:|"]
        for g, x in m.items():
            md.append(f"| {g} | {x['zeros']:.1f} | {x['early_inside_across']:.1f} | {x['late_outside_kerb']:.1f} | {x['onc_pre_share']:.2f} ({x['onc_pre_n'][0]} / {x['onc_pre_n'][1]}) | {x['straight_corridor_past_end']:.1f} |")
    if "cover" in res:
        md += ["", f"Served memory coverage (YRM reads): +20 m covered on {res['cover']['c20']:.3f} of {res['cover']['decisions']} decisions, median nearest valid offset {res['cover']['first_med']:.0f} m."]
    md += ["", "## Registered lines", ""] + [f"- {k}: **{'met' if v else 'not met'}**" for k, v in res["lines"].items()] + ["", f"**Adopted: {res['adopted']}**"]
    (O / "report_route_mem.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gate"), sub.add_parser("report")
    a = ap.parse_args()
    dict(gate=cmd_gate, report=cmd_report)[a.cmd](a)
