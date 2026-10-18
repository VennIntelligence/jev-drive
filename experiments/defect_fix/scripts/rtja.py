"""RTJA, NR1 prereg supplement 12 (plans/2026-10-17-nr1-prereg.md): V2 base + the JTN stage + in-stage P0 retention at three doses on the
anchored route memory. Read-outs and checks of the chain in jt.sh (`rtja-*` stages). Box, repo root.

  acheck --data D ..          (any python with numpy) every anchored feature file a training reads: feat/<d>.a.npz rows (names) and `ok`
      equal feat/<d>.npz row by row, f of width DIM_O and finite; on1_JT also against its cache tab -> $J/rtja/acheck.json; rc 1 on a mismatch
  smoke --tag T               (any python) the smoke gates (supplement 12 (b)): no non-finite scalar; kd / d2 / agent / road / onc / soft / side
      active (loss/kd, loss/d2 > 0; act/pool_{agent,road,lat_onc,lat_soft,lat_side} > 0); sidecar head order + anchor (unless NOANCHOR);
      s/step -> $J/rtja/smoke.json; rc 1 on a failed gate
  wcheck --a CSV --b CSV      (any python) a chunked wait-set table against the whole read of the same tag, window by window (score, zero,
      wait_ok, launch_s) -> stdout json; rc 1 on any difference
  select --tags T .. --lams L ..   (op-train) the dose rule of supplement 12 on wait set v2 + held 460 -> results/rtja/dose.md, $J/rtja/select.json
      (winner or none); rc 3 when no arm passes (the winner's reads chained after it never start)
  adopt --tag T               (op-train) the adoption lines of supplement 12 from results/rtja/<T>/{arena.md, held.txt, pai281.md}
      -> results/rtja/adoption.md
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
J = DATA / "runs/defect_fix/jt/rtja"
RES = REPO / "experiments/defect_fix/results/rtja"
WS = DATA / "runs/alpasim_proxy/wait_set/reads"
LEAD = ("other_lead", "red_queue")
WAIT_MIN = {"other_lead": 0.78, "red_queue": 0.67}           # V2 (RTB1ARO-L18-s0) a-variant wait rate - 0.05, as the supplement states them
DELAY_MAX, LAUNCH_MIN, HELD_ZMAX = 0.15, 0.95, 37


def cmd_acheck(a):
    sys.path[:0] = [str(REPO / "experiments/route_mem/lib")]
    import rmem as RM
    root, out, bad = DATA / "runs/route_mem/feat", {}, []
    for d in a.data:
        z, za = np.load(root / f"{d}.npz"), np.load(root / f"{d}.a.npz")
        r = dict(rows=len(z["names"]), names_equal=bool(np.array_equal(z["names"], za["names"])),
                 ok_equal=bool(np.array_equal(z["ok"], za["ok"])), width=int(za["f"].shape[1]),
                 finite=bool(np.isfinite(za["f"][za["ok"]].astype(np.float32)).all()), ok_share=float(za["ok"].mean()))
        if d == "on1_JT":
            tab = np.load(DATA / "runs/op_parity/cache/on1_JT/tab.npz")["names"]
            r["tab_equal"] = bool(np.array_equal(tab, za["names"]))
        r["pass"] = r["names_equal"] and r["ok_equal"] and r["width"] == RM.DIM_O and r["finite"] and r.get("tab_equal", True)
        out[d] = r
        if not r["pass"]:
            bad.append(d)
    res = dict(passed=not bad, failed=bad, files=out)
    J.mkdir(parents=True, exist_ok=True)
    (J / "acheck.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(dict(passed=not bad, failed=bad, on1_JT=out.get("on1_JT"))))
    sys.exit(0 if not bad else 1)


def scalars(tag):
    f = sorted(glob.glob(str(DATA / "runs/route_mem" / f"train-{tag}" / "*" / "events.jsonl")))[-1]
    S = {}
    for line in open(f):
        e = json.loads(line)
        if e.get("kind") == "scalar":
            S.setdefault(e["tag"], {})[e["step"]] = e["value"]
    return f, S


def cmd_smoke(a):
    f, S = scalars(a.tag)
    mean = lambda k: float(np.mean(list(S[k].values()))) if k in S else float("nan")  # noqa: E731
    need = ["loss/kd", "loss/d2", "loss/act/pool_agent", "loss/act/pool_road", "loss/act/pool_lat_onc", "loss/act/pool_lat_soft", "loss/act/pool_lat_side"]
    if a.kd_lam == 0:
        need.remove("loss/kd")
    act = {k: mean(k) for k in need}
    nan = sorted(k for k, v in S.items() if any(not math.isfinite(x) for x in v.values()))
    sps = S.get("throughput/steps_per_s", {})
    head = json.loads((DATA / "runs/op_parity/runs" / a.tag / "route_mem.json").read_text()).get("head", {})
    want_anchor = not os.environ.get("NOANCHOR")
    gates = dict(no_nan=not nan, terms_active=all(v > 0 for v in act.values()),
                 head=(bool(head.get("order")) and bool(head.get("anchor"))) if want_anchor else not head.get("anchor"))
    res = dict(tag=a.tag, events=f, last_step=max(max(v) for v in S.values()), gates=gates, passed=all(gates.values()), nan=nan, active=act,
               head=head, steps_per_s=sps[max(sps)] if sps else None, s_per_step=1 / sps[max(sps)] if sps else None,
               kd_mean=mean("loss/kd"), d2_mean=mean("loss/d2"))
    J.mkdir(parents=True, exist_ok=True)
    (J / "smoke.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))
    sys.exit(0 if res["passed"] else 1)


def table(path):
    return {r["id"]: r for r in csv.DictReader(open(path))}


def cmd_wcheck(a):
    A, B = table(a.a), table(a.b)
    keys = ("score", "zero", "wait_ok", "launch_s", "fail")
    diff = [(i, {k: (A[i][k], B[i][k]) for k in keys if A[i][k] != B[i][k]}) for i in A if i in B]
    diff = [(i, d) for i, d in diff if d]
    res = dict(chunk_windows=len(A), common=sum(i in B for i in A), differing=len(diff), first=diff[:5], identical=not diff and all(i in B for i in A))
    print(json.dumps(res))
    sys.exit(0 if res["identical"] else 1)


def wait_stats(tag):
    rows = list(csv.DictReader(open(WS / tag / "table" / f"{tag}.csv")))
    out = {}
    for c in LEAD:
        ra = [r for r in rows if r["variant"] == "a" and r["cls"] == c]
        rb = [r for r in rows if r["variant"] == "b" and r["cls"] == c]
        val = lambda v: v not in ("", None, "nan")  # noqa: E731   (older tables write a missing value as "nan")
        dl = [float(r["delay_s"]) for r in rb if val(r["delay_s"])]
        out[c] = dict(n_a=len(ra), wait=sum(r["wait_ok"] == "True" for r in ra) / max(1, len(ra)), n_b=len(rb),
                      launched=sum(val(r["launch_s"]) for r in rb) / max(1, len(rb)), delay_med=float(np.median(dl)) if dl else float("nan"),
                      b_score=float(np.mean([float(r["score"]) for r in rb])) if rb else float("nan"))
    out["a_zeros"] = sum(r["variant"] == "a" and r["zero"] not in ("", "False") for r in rows)
    out["b_score"] = float(np.mean([float(r["score"]) for r in rows if r["variant"] == "b"]))
    return out


def held(tag):
    sys.path[:0] = [str(HERE)]
    import df
    r = df.cl_read(tag, ["RSG-L18-s0"])["held"]
    p = r["pair"].get("RSG-L18-s0", {})
    return dict(zeros=r["zeros"], mean=r["mean"], offroad=r["offroad"], pair_rsg=(p.get("mean"), p.get("lo"), p.get("hi")))


def cmd_select(a):
    rows, ok = [], []
    for t, lam in zip(a.tags, a.lams):
        w, h = wait_stats(t), held(t)
        g = {f"wait_{c}": w[c]["wait"] >= WAIT_MIN[c] for c in LEAD} | {f"delay_{c}": abs(w[c]["delay_med"]) <= DELAY_MAX for c in LEAD} \
            | {f"launch_{c}": w[c]["launched"] >= LAUNCH_MIN for c in LEAD} | {"held_zeros": h["zeros"] <= HELD_ZMAX}
        rows.append(dict(tag=t, lam=float(lam), wait=w, held=h, gates=g, passed=all(g.values())))
        if all(g.values()):
            ok.append(rows[-1])
    win = None
    if ok:
        z0 = min(r["held"]["zeros"] for r in ok)
        win = min((r for r in ok if r["held"]["zeros"] <= z0 + 1), key=lambda r: r["lam"])["tag"]
    refs = {t: wait_stats(t) for t in ("RTB1ARO-L18-s0", "JTN-L18-s0", "RSG-L18-s0")}
    L = ["# RTJA dose selection (NR1 prereg supplement 12)", "",
         f"Rule: per lead class (other_lead / red_queue) wait-set v2 variant a wait rate >= {WAIT_MIN['other_lead']} / {WAIT_MIN['red_queue']} (V2 - 0.05), "
         f"variant b median launch delay (car minus human, launched windows) within +-{DELAY_MAX} s and launched >= {LAUNCH_MIN:.0%}; held 460 @18 zeros "
         f"<= {HELD_ZMAX}. Among the passing arms the fewest held zeros (within 1: the smaller lambda).", "",
         "| arm | kd-lam | a wait other_lead / red_queue | b launched other_lead / red_queue | b delay med s | a zeros | b mean | held zeros | held mean | held vs RSG [CI] | pass |",
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|"]
    for r in rows:
        w, h, p = r["wait"], r["held"], r["held"]["pair_rsg"]
        fails = ", ".join(k for k, v in r["gates"].items() if not v) or "-"
        L.append(f"| {r['tag']} | {r['lam']:g} | {w['other_lead']['wait']:.3f} / {w['red_queue']['wait']:.3f} | {w['other_lead']['launched']:.3f} / "
                 f"{w['red_queue']['launched']:.3f} | {w['other_lead']['delay_med']:+.2f} / {w['red_queue']['delay_med']:+.2f} | {w['a_zeros']} | "
                 f"{w['b_score']:.3f} | {h['zeros']} | {h['mean']:.4f} | {p[0]:+.4f} [{p[1]:+.4f}, {p[2]:+.4f}] | "
                 f"{'yes' if r['passed'] else 'no (' + fails + ')'} |")
    for t, w in refs.items():
        L.append(f"| {t} (stored) | - | {w['other_lead']['wait']:.3f} / {w['red_queue']['wait']:.3f} | {w['other_lead']['launched']:.3f} / "
                 f"{w['red_queue']['launched']:.3f} | {w['other_lead']['delay_med']:+.2f} / {w['red_queue']['delay_med']:+.2f} | {w['a_zeros']} | "
                 f"{w['b_score']:.3f} | - | - | - | - |")
    L += ["", f"**Winner: {win}**" if win else "**No arm passes: stop (supplement 12), the winner's reads do not run.**"]
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "dose.md").write_text("\n".join(L) + "\n")
    J.mkdir(parents=True, exist_ok=True)
    (J / "select.json").write_text(json.dumps(dict(winner=win, arms=rows), indent=1, default=str))
    (RES / "dose.json").write_text(json.dumps(dict(winner=win, arms=rows), indent=1, default=str))
    print("\n".join(L))
    if not win:
        (J / "STOP").write_text("no arm passes the dose rule\n")
        sys.exit(3)


NUM = r"([+-]?\d+\.\d+)"


def cmd_adopt(a):
    R = RES / a.tag
    ar, pai = (R / "arena.md").read_text(), (R / "pai281.md").read_text()
    m = re.search(rf"\| {re.escape(a.tag)}@stop3 \| (\d+) \| {NUM} \| {NUM} \| {NUM} \[{NUM}, {NUM}\]", ar)
    st = dict(windows=int(m[1]), mean=float(m[2]), rsg=float(m[3]), diff=float(m[4]), lo=float(m[5]), hi=float(m[6]))
    h = held(a.tag)
    jl = f"{a.tag}+ACC+GAP"
    pj = re.search(rf"Paired mean \({re.escape(jl)} - JTN\+ACC\+GAP\): \*\*{NUM} \[{NUM}, {NUM}\]", pai)
    pr = re.search(rf"Paired mean \({re.escape(jl)} - RSG\+ACC\+GAP\): \*\*{NUM} \[{NUM}, {NUM}\]", pai)
    zr = {mm[1]: (int(mm[2]), float(mm[3]), int(mm[4])) for mm in re.finditer(r"^\| ([^|]+?) \| (\d+) \| (\d\.\d+) \| (\d+) \|", pai, re.M)}
    zj = zr[jl][2]
    nu = st["diff"] >= 0.005 and st["lo"] > 0 and h["zeros"] <= HELD_ZMAX
    pa = float(pj[1]) >= -0.005 and zj <= 67
    L = [f"# RTJA adoption lines: {a.tag}", "",
         "| board | line | measured | pass |", "|---|---|---|:--|",
         f"| nuPlan | arena held stop3 vs stored RSG + stop3: paired >= +0.005 and CI low > 0 | {st['diff']:+.4f} [{st['lo']:+.4f}, {st['hi']:+.4f}] "
         f"({st['windows']} windows; {st['mean']:.4f} vs {st['rsg']:.4f}) | {'yes' if st['diff'] >= 0.005 and st['lo'] > 0 else 'no'} |",
         f"| nuPlan | held 460 @18 zeros <= {HELD_ZMAX} | {h['zeros']} (mean {h['mean']:.4f}) | {'yes' if h['zeros'] <= HELD_ZMAX else 'no'} |",
         f"| PAI 281 | vs stored JTN + ACC + GAP: paired >= -0.005 | {pj[1]} [{pj[2]}, {pj[3]}] | {'yes' if float(pj[1]) >= -0.005 else 'no'} |",
         f"| PAI 281 | zeros <= 67 | {zj} (JTN+ACC+GAP {zr['JTN+ACC+GAP'][2]}, RSG+ACC+GAP {zr['RSG+ACC+GAP'][2]}) | {'yes' if zj <= 67 else 'no'} |",
         f"| PAI 281 | (reported) vs stored RSG + ACC + GAP | {pr[1]} [{pr[2]}, {pr[3]}] | - |", "",
         f"nuPlan: **{'adopt' if nu else 'not adopted'}**; PAI: **{'adopt' if pa else 'not adopted'}** (each board judged on its own; a pass builds the image "
         "with the anchored rmem.py, the audit and a second seed)."]
    (RES / "adoption.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("acheck")
    p.add_argument("--data", nargs="+", required=True)
    p = sub.add_parser("smoke")
    p.add_argument("--tag", required=True), p.add_argument("--kd-lam", type=float, default=1.0)
    p = sub.add_parser("wcheck")
    p.add_argument("--a", required=True), p.add_argument("--b", required=True)
    p = sub.add_parser("select")
    p.add_argument("--tags", nargs="+", required=True), p.add_argument("--lams", nargs="+", required=True)
    p = sub.add_parser("adopt")
    p.add_argument("--tag", required=True)
    a = ap.parse_args()
    dict(acheck=cmd_acheck, smoke=cmd_smoke, wcheck=cmd_wcheck, select=cmd_select, adopt=cmd_adopt)[a.cmd](a)
