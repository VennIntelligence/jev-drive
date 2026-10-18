"""NTVX-L18-s0 summary and adoption lines (plans/2026-10-19-pn-prereg.md supplement 1). Box, op-train env; run by `ntvx.sh sum` after
results/<tag>/{arena.md, pai281.md} exist.

  ntvx.py --tag NTVX-L18-s0 --ref NTV0-L18-s0 --res DIR   -> DIR/{held.md, wait.md, adoption.md, verdict.json}

nuPlan: arena held stop3 vs stored RSG + stop3 paired >= +0.005 with CI low > 0, and held 460 @18 zeros <= 37.
PAI 281 (ACC + GAP): vs stored <ref> + ACC + GAP paired >= -0.005, and zeros <= <ref>'s zeros on the same scenes + 2; vs RSG + ACC + GAP reported.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DF = HERE.parents[1] / "defect_fix/scripts"
sys.path[:0] = [str(DF)]
import df  # noqa: E402
import rtja  # noqa: E402

NUM = r"([+-]?\d+\.\d+)"
HELD_ZMAX, ARENA_MIN, PAI_MIN, PAI_DZ = 37, 0.005, -0.005, 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True), ap.add_argument("--ref", required=True), ap.add_argument("--res", required=True)
    a = ap.parse_args()
    R, T, T0 = Path(a.res), a.tag, a.ref
    refs = ["RSG-L18-s0", T0]

    # held 460 @18 against RSG and the reference arm
    L = ["# Held 460 @18", "", "Paired = row tag minus the named tag, session-clustered 95 % CI (defect_fix df.cl_read).", "", *df.CL_HEAD]
    hr = {}
    for t in (T, T0):
        hr[t] = df.cl_read(t, [r for r in refs if r != t])
        L += df.fmt_cl(t, hr[t], [r for r in refs if r != t])
    (R / "held.md").write_text("\n".join(L) + "\n")
    h = hr[T]["held"]

    # wait set v2 (reported only)
    W = ["# Wait set v2 (readc)", "", "Lead classes other_lead / red_queue: variant a wait rate, variant b launched share and median launch delay (car minus human).", "",
         "| tag | a wait | b launched | b delay med s | a zeros | b mean |", "|---|--:|--:|--:|--:|--:|"]
    for t in (T, T0, "RSG-L18-s0", "JTN-L18-s0"):
        w = rtja.wait_stats(t)
        W.append(f"| {t}{'' if t == T else ' (stored)'} | {w['other_lead']['wait']:.3f} / {w['red_queue']['wait']:.3f} | {w['other_lead']['launched']:.3f} / "
                 f"{w['red_queue']['launched']:.3f} | {w['other_lead']['delay_med']:+.2f} / {w['red_queue']['delay_med']:+.2f} | {w['a_zeros']} | {w['b_score']:.3f} |")
    (R / "wait.md").write_text("\n".join(W) + "\n")

    # arena held stop3 / off against RSG
    ar = (R / "arena.md").read_text()
    st = {}
    for t in (T, T0):
        for v in ("stop3", "off"):
            m = re.search(rf"\| {re.escape(t)}@{v} \| (\d+) \| {NUM} \| {NUM} \| {NUM} \[{NUM}, {NUM}\]", ar)
            st[t, v] = dict(windows=int(m[1]), mean=float(m[2]), rsg=float(m[3]), diff=float(m[4]), lo=float(m[5]), hi=float(m[6]))

    # PAI 281
    pai = (R / "pai281.md").read_text()
    jl, rl = f"{T}+ACC+GAP", f"{T0}+ACC+GAP"
    pp = lambda x, y: tuple(map(float, re.search(rf"Paired mean \({re.escape(x)} - {re.escape(y)}\): \*\*{NUM} \[{NUM}, {NUM}\]", pai).groups()))  # noqa: E731
    zr = {m[1]: dict(scenes=int(m[2]), mean=float(m[3]), zeros=int(m[4])) for m in re.finditer(r"^\| ([^|]+?) \| (\d+) \| (\d\.\d+) \| (\d+) \|", pai, re.M)}
    p_ref, p_rsg = pp(jl, rl), pp(jl, "RSG+ACC+GAP")

    s = st[T, "stop3"]
    g = dict(arena=s["diff"] >= ARENA_MIN and s["lo"] > 0, held=h["zeros"] <= HELD_ZMAX,
             pai_pair=p_ref[0] >= PAI_MIN, pai_zeros=zr[jl]["zeros"] <= zr[rl]["zeros"] + PAI_DZ)
    nu, pa = g["arena"] and g["held"], g["pai_pair"] and g["pai_zeros"]
    yn = lambda b: "yes" if b else "no"  # noqa: E731
    ci = lambda x: f"{x[0]:+.4f} [{x[1]:+.4f}, {x[2]:+.4f}]"  # noqa: E731
    hp = {c: h["pair"][c] for c in h["pair"]}
    A = [f"# Adoption lines: {T} (PN prereg supplement 1)", "",
         "| board | line | measured | pass |", "|---|---|---|:--|",
         f"| nuPlan | arena held stop3 vs stored RSG + stop3: paired >= +{ARENA_MIN} and CI low > 0 | {ci((s['diff'], s['lo'], s['hi']))} ({s['windows']} windows; "
         f"{s['mean']:.4f} vs {s['rsg']:.4f}; {T0} {ci(tuple(st[T0, 'stop3'][k] for k in ('diff', 'lo', 'hi')))}) | {yn(g['arena'])} |",
         f"| nuPlan | held 460 @18 zeros <= {HELD_ZMAX} | {h['zeros']} (mean {h['mean']:.4f}; {T0} {hr[T0]['held']['zeros']}) | {yn(g['held'])} |",
         f"| PAI 281 | vs stored {T0} + ACC + GAP: paired >= {PAI_MIN} | {ci(p_ref)} | {yn(g['pai_pair'])} |",
         f"| PAI 281 | zeros <= {T0}'s + {PAI_DZ} (same {zr[jl]['scenes']} scenes) | {zr[jl]['zeros']} vs {zr[rl]['zeros']} (RSG+ACC+GAP {zr['RSG+ACC+GAP']['zeros']}) | {yn(g['pai_zeros'])} |",
         f"| PAI 281 | (reported) vs stored RSG + ACC + GAP | {ci(p_rsg)}; means {zr[jl]['mean']:.4f} / {zr[rl]['mean']:.4f} / {zr['RSG+ACC+GAP']['mean']:.4f} | - |",
         f"| nuPlan | (reported) arena held off vs stored RSG | {ci(tuple(st[T, 'off'][k] for k in ('diff', 'lo', 'hi')))} ({T0} "
         f"{ci(tuple(st[T0, 'off'][k] for k in ('diff', 'lo', 'hi')))}) | - |",
         *[f"| nuPlan | (reported) held 460 vs {c} | {ci((p['mean'], p['lo'], p['hi']))} (zeros {p['zeros']} / {p['zeros_ref']}) | - |" for c, p in hp.items()], "",
         f"nuPlan: **{'adopt' if nu else 'not adopted'}**; PAI: **{'adopt' if pa else 'not adopted'}** (each board on its own; a pass builds the image, "
         "the audit and a second seed)."]
    (R / "adoption.md").write_text("\n".join(A) + "\n")
    (R / "verdict.json").write_text(json.dumps(dict(tag=T, ref=T0, gates=g, nuplan=nu, pai=pa, arena=st[T, "stop3"], arena_off=st[T, "off"],
                                                    held=dict(zeros=h["zeros"], mean=h["mean"], pair=hp), pai_vs_ref=p_ref, pai_vs_rsg=p_rsg,
                                                    pai_table=zr), indent=1, default=str))
    print("\n".join(A))


if __name__ == "__main__":
    main()
