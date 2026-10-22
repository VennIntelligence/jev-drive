#!/usr/bin/env python3
"""ROUND9 NU-PAIBASE helper (plans/2026-10-22-nu-paibase.md; lane script nupb.sh).

  $TR nupb.py smoke --ok RC --ctl TAG --arm TAG --init BASE --finite JSON --out JSON
        (box) the arm's 60-step run: finite losses (stand_b.py smoke --finite-only, rc RC), meta argv init = BASE, and dev ADE / weights differ
        from the control run with init RB0-F-s0 (wrong-init control: an ignored --init would give a bit-identical run)
  tmp/pcs_local/.venv/bin/python nupb.py mac [--ext DIR]
        (Mac, after rsync of results/nu_paibase) frozen board-mix PCS (pcs_local.py score), easy-item tables (easy_items.py subject, ref
        JTN-L18-s1), static-layer zeros, and the registered adoption verdict -> results/nu_paibase/verdict.{md,json}
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RES = REPO / "experiments/round9/results/nu_paibase"
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
B2 = {0: "STANDFMXH-L18-s0", 1: "STANDFMXH-L18-s1"}
ARMS = {"70": ["NPB70-L18-s0", "NPB70-L18-s1"], "30": ["NPB30-L18-s0", "NPB30-L18-s1"]}
JTN = "JTN-L18-s1"


def cmd_smoke(a):
    import torch
    run = lambda t: sorted(glob.glob(f"{DATA}/runs/route_mem/train-{t}/*/"))[-1]  # noqa: E731
    argv = json.load(open(run(a.arm) + "meta.json"))["argv"]
    init = argv[len(argv) - 1 - argv[::-1].index("--init") + 1]
    ade = [json.load(open(run(t) + "DONE")).get("dev_ade") for t in (a.ctl, a.arm)]
    m = [torch.load(f"{DATA}/runs/op_parity/runs/{t}/ckpt-final.pt", map_location="cpu", weights_only=False)["model"] for t in (a.ctl, a.arm)]
    dw = max(float((m[0][g][k].float() - m[1][g][k].float()).abs().max()) for g in ("net", "parity") for k in m[0][g] if k in m[1][g])
    fin = json.load(open(a.finite))
    r = dict(ctl=a.ctl, arm=a.arm, finite_rc=a.ok, finite=fin, init=init, dev_ade=ade, weight_max_abs_diff_vs_ctl=dw)
    r["pass_"] = bool(a.ok == 0 and fin.get("pass_") and init == a.init and ade[0] != ade[1] and dw > 0)
    Path(a.out).write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))
    raise SystemExit(0 if r["pass_"] else 2)


def corridor(z):
    return sum(v for k, v in z.items() if "corridor" in k)


def cmd_mac(a):
    import pandas as pd
    py = sys.executable
    rows = RES / "arena_windows.csv.gz"
    ar, hd = json.load(open(RES / "arena.json")), json.load(open(RES / "held460.json"))
    # 1. frozen board-mix PCS, every tag of the window table from the same script
    if not (RES / "pcs.txt").exists():
        (RES / "pcs.txt").write_text(subprocess.run([py, str(REPO / "experiments/alpasim_proxy/scripts/pcs_local.py"), "score", "--rows", str(rows)],
                                                    capture_output=True, text=True, check=True).stdout)
    out = (RES / "pcs.txt").read_text()
    pcs = {m.group(1): float(m.group(2)) for m in re.finditer(r"^(\S+): windows .* PCS (\d+)$", out, re.M)}
    # 2. easy-item tables (decision 387 / 390), ref JTN-L18-s1
    tags = [*ARMS["70"], *ARMS["30"], *B2.values()]
    easy = {}
    if a.ext:                                                        # one process per tag, in parallel (each takes ~30 min)
        ps = [subprocess.Popen([py, str(REPO / "experiments/alpasim_proxy/scripts/easy_items.py"), "subject", "--rows", str(rows), "--tag", t, "--ref", JTN,
                                "--dec", a.ext, "--out", f"../../round9/results/nu_paibase/easy_{t}", "--misses", f"../../../round9/results/nu_paibase/misses_{t}"],
                               stdout=subprocess.DEVNULL, stderr=open(f"/tmp/nupb_easy_{t}.log", "w")) for t in a.easy_tags or tags]
        assert all(p.wait() == 0 for p in ps), "easy_items subject failed"
    for t in tags:
        if (RES / f"easy_{t}.json").exists():
            J = json.load(open(RES / f"easy_{t}.json"))
            st = {s["stratum"][:2]: s for s in J["strata"]}
            ez = st["a:"]
            easy[t] = dict(pcs_bm=J["base"].get("boardmix"), mean=J["base"]["mean"], zeros=J["base"]["zeros"], partials=J["base"]["partials"],
                           easy_zeros=ez["B_zeros"], easy_partials=ez["B_partials"], easy_lost=ez["B_lost"])
    # 3. static layer: held zeros by the window's stratum (tmp/native_gap/set.csv.gz), corridor zeros static / moving
    W = pd.read_csv(rows)
    S = pd.read_csv(REPO / "tmp/native_gap/set.csv.gz").drop_duplicates("id").set_index("id")
    W = W[W.id.isin(S.index[S.split == "held"])].assign(stratum=lambda d: S.stratum.reindex(d.id).values)
    W = W[W.type != "route-fail"]
    z = W[W.score == 0]
    stat = {t: dict(zeros=len(g), corridor_static=int((g.type.str.contains("corridor") & (g.stratum == "static")).sum()),
                    corridor_moving=int((g.type.str.contains("corridor") & (g.stratum != "static")).sum()),
                    offroad=int(g.type.str.startswith("offroad").sum()), af_front=int(g.type.str.startswith("AF-front").sum()),
                    static_all=int((g.stratum == "static").sum())) for t, g in z.groupby("tag")}
    # 4. registered verdict: both seeds, each against B2 of the same seed
    V, L = {}, ["# NU-PAIBASE verdict (plans/2026-10-22-nu-paibase.md, registered lines)", "",
                "Per arm against B2 of the same seed: (a) arena held native paired mean >= 0; (b) corridor zeros (held, every class containing corridor) "
                "< B2; (c) held 460 zeros <= B2 + 2; (d) frozen board-mix PCS >= B2. A share passes when both seeds pass all four.", "",
                "| arm | ref | arena paired [CI] | corridor zeros arm / B2 | held 460 zeros arm / B2 | held 460 paired [CI] | PCS arm / B2 | a | b | c | d |",
                "|:--|:--|--:|--:|--:|--:|--:|:-:|:-:|:-:|:-:|"]
    yn = lambda x: "pass" if x else "FAIL"  # noqa: E731
    for sh, ts in ARMS.items():
        ok = []
        for s, t in enumerate(ts):
            r = B2[s]; p, h = ar[f"{t}|{r}"], hd[f"{t}|{r}"]
            ca, cb = corridor(ar["zeros"][t]), corridor(ar["zeros"][r])
            c = dict(a=p["mean"] >= 0, b=ca < cb, c=h["zeros_a"] <= h["zeros_b"] + 2, d=pcs.get(t, -1) >= pcs.get(r, 1e9))
            ok.append(all(c.values()))
            V[t] = dict(ref=r, arena=p, corridor=[ca, cb], held=h, pcs=[pcs.get(t), pcs.get(r)], checks=c, pass_=all(c.values()))
            L.append(f"| {t} | {r} | {p['mean']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}] | {ca} / {cb} | {h['zeros_a']} / {h['zeros_b']} | "
                     f"{h['mean']:+.4f} [{h['lo']:+.4f}, {h['hi']:+.4f}] | {pcs.get(t, float('nan')):.0f} / {pcs.get(r, float('nan')):.0f} | "
                     + " | ".join(yn(c[k]) for k in "abcd") + " |")
        V[f"share{sh}"] = all(ok)
    L += ["", "Shares: " + ", ".join(f"P{sh} {'PASS' if V[f'share{sh}'] else 'FAIL'}" for sh in ARMS), "",
          "## Against JTN-L18-s1 (reported, not judged)", "", "| arm | arena paired [CI] | zeros arm / JTN | held 460 paired [CI] | held 460 zeros arm / JTN |",
          "|:--|--:|--:|--:|--:|"]
    for t in [*ARMS["70"], *ARMS["30"]]:
        p, h = ar[f"{t}|{JTN}"], hd[f"{t}|{JTN}"]
        L.append(f"| {t} | {p['mean']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}] | {p['zeros_a']} / {p['zeros_b']} | {h['mean']:+.4f} [{h['lo']:+.4f}, {h['hi']:+.4f}] | "
                 f"{h['zeros_a']} / {h['zeros_b']} |")
    L += ["", "## Zero layers (arena held, route-fail dropped; stratum from the arena set)", "",
          "| tag | zeros | corridor, static window | corridor, moving window | offroad | AF-front | all zeros in static windows | PCS |", "|:--|--:|--:|--:|--:|--:|--:|--:|"]
    for t in [*ARMS["70"], *ARMS["30"], *B2.values(), JTN]:
        x = stat.get(t, {})
        L.append(f"| {t} | {x.get('zeros', 0)} | {x.get('corridor_static', 0)} | {x.get('corridor_moving', 0)} | {x.get('offroad', 0)} | {x.get('af_front', 0)} | "
                 f"{x.get('static_all', 0)} | {pcs.get(t, float('nan')):.0f} |")
    if easy:
        L += ["", "## Easy items (garage 0014 = 1; easy_items.py subject, ref JTN-L18-s1)", "",
              "| tag | PCS board-mix | zeros | partials | easy zeros | easy partials | easy points lost |", "|:--|--:|--:|--:|--:|--:|--:|"]
        for t, e in easy.items():
            L.append(f"| {t} | {e['pcs_bm']:.0f} | {e['zeros']} | {e['partials']} | {e['easy_zeros']} | {e['easy_partials']} | {e['easy_lost']:.1f} |")
    (RES / "verdict.md").write_text("\n".join(L) + "\n")
    (RES / "verdict.json").write_text(json.dumps(dict(verdict=V, pcs=pcs, static=stat, easy=easy), indent=1, default=float))
    print("\n".join(L))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    q = sp.add_parser("smoke")
    for k in ("ctl", "arm", "init", "finite", "out"):
        q.add_argument(f"--{k}", required=True)
    q.add_argument("--ok", type=int, required=True)
    q = sp.add_parser("mac")
    q.add_argument("--ext", default="", help="local copy of the box's native_gap extract ($O/ext): run the easy-item tables (else use existing)")
    q.add_argument("--easy-tags", nargs="*", default=[])
    a = p.parse_args()
    dict(smoke=cmd_smoke, mac=cmd_mac)[a.cmd](a)


if __name__ == "__main__":
    main()
