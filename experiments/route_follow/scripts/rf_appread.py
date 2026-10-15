#!/usr/bin/env python3
"""APPROACH COURSE PAI mechanism read (plans/2026-10-15-approach-course-prereg.md Amendment 4, EXPLORATORY). Tokyo, plain python3 + numpy.
Stored rf_pai_follow extract pkls read, never written.

  rf_appread.py --a rsg=<pkl>[:<pkl label>] --b app1x=<pkl>[:<pkl label>] --strict <strict36.txt> --runway <pai_runway22.txt> --out <md>
      -> <md> and <md>.json

Per strict scene x label: rf_pai_follow.one (the label's own AlpaSim route, strict turn rule, completion) and on the same route the ego's
controller track (10 Hz, 5-sample box speed) and the human log (rf_pai_follow.one's log speed), both cut at the label's zero event:
  v_entry      ego speed at the first crossing of the entry station (rf_pai_follow.speed_at); v_h the human's there
  excess26     mean over the ego samples with station in [entry - 26, entry] of max(0, v - v_h(station)) (human speed at its first passage of
               that station), m/s; NaN when no sample there before the zero
  inward20     signed lateral offset to the route at the first crossing of entry - 20 m, + toward the turn's inside (left turn: left +)
  completed    rf_pai_follow.one
Paired b - a over the runway turns where both labels define the turn: mean, scene-bootstrap 95 % CI (B 10 000, rng 0), median of the
differences; completion counts on the runway turns and on the other strict turns.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
WIN = 26.0


def first_passage(lt, v, q):
    m = np.maximum.accumulate(lt)
    u, i = np.unique(m, return_index=True)
    return np.interp(q, u, v[i], left=np.nan, right=np.nan)


def read(sc, lab, o):
    import rf_pai_follow as F
    import rf_shape as SH
    r, _, _ = F.one(sc, lab, o, zeros=False)
    out = dict(scene=sc["scene"], turn=bool(r.get("turn")), completed=bool(r.get("completed", False)), outcome=r.get("outcome"), kind=r.get("kind"))
    if not out["turn"]:
        return out
    C, LG = o["ctrl"], sc["log"]
    R = F.dedup(o["route"])
    S = F.stations(R)
    t = np.asarray(C["t"], float)
    st, lat = F.track(R, S, np.c_[C["x"], C["y"]])
    v = SH.box5(C["vx"])
    tz = o["tz"] if o["tz"] is not None and not o["score"] else np.inf
    lt, _ = F.track(R, S, LG["P"])
    vlog = np.r_[0.0, np.hypot(*np.diff(LG["P"], axis=0).T) / np.maximum(np.diff(LG["T"]) / 1e6, 1e-3)]
    vlog = np.convolve(np.r_[vlog[:2], vlog, vlog[-2:]], np.ones(5) / 5, "valid")
    ent, sg = float(r["entry"]), (1.0 if r["side"] == "L" else -1.0)
    w = (st >= ent - WIN) & (st <= ent) & (t < tz)
    vh = first_passage(lt, vlog, st[w])
    ok = np.isfinite(vh)
    i = np.flatnonzero((st >= ent - 20.0) & (t < tz))
    inward = np.nan
    if len(i) and i[0] > 0:
        j = int(i[0])
        a = (ent - 20.0 - st[j - 1]) / max(st[j] - st[j - 1], 1e-9)
        inward = sg * float((1 - a) * lat[j - 1] + a * lat[j])
    out.update(side=r["side"], entry=ent, start_to_entry=float(ent - st[0]), v_entry=F.speed_at(t, st, v, ent, tz),
               v_h=F.speed_at(np.asarray(LG["T"], float), lt, vlog, ent), excess26=float(np.maximum(v[w][ok] - vh[ok], 0).mean()) if ok.any() else np.nan,
               inward20=inward)
    return out


def boot(d, B=10000):
    d = np.asarray(d, float)
    if not len(d):
        return dict(n=0)
    idx = np.random.default_rng(0).integers(0, len(d), (B, len(d)))
    m = d[idx].mean(1)
    return dict(n=int(len(d)), mean=float(d.mean()), lo=float(np.percentile(m, 2.5)), hi=float(np.percentile(m, 97.5)), median=float(np.median(d)))


def main(a):
    strict = [x.split()[0] for x in open(a.strict) if x.strip()]
    runway = {x.split()[0] for x in open(a.runway) if x.strip()}
    L = {}
    for spec in (a.a, a.b):
        lab, rest = spec.split("=", 1)
        f, src = (rest.rsplit(":", 1) if ":" in rest else (rest, lab))
        D = {sc["scene"]: sc for sc in pickle.load(open(f, "rb"))["scenes"]}
        L[lab] = {s: read(D[s], src, D[s]["rolls"][src]) for s in strict if s in D and D[s]["rolls"].get(src, {}).get("ctrl") is not None}
    (la, A), (lb, Bm) = L.items()
    res = dict(a=la, b=lb, scenes={s: {la: A.get(s), lb: Bm.get(s)} for s in strict}, counts={}, paired={})
    for nm, S in (("runway22", [s for s in strict if s in runway]), ("other14", [s for s in strict if s not in runway])):
        res["counts"][nm] = {lab: int(sum(bool((X.get(s) or {}).get("completed")) for s in S)) for lab, X in L.items()} | dict(n=len(S))
    rw = [s for s in strict if s in runway and (A.get(s) or {}).get("turn") and (Bm.get(s) or {}).get("turn")]
    for k in ("v_entry", "excess26", "inward20"):
        d = [Bm[s][k] - A[s][k] for s in rw if np.isfinite(A[s].get(k, np.nan)) and np.isfinite(Bm[s].get(k, np.nan))]
        res["paired"][k] = boot(d) | dict(a_median=float(np.nanmedian([A[s][k] for s in rw])), b_median=float(np.nanmedian([Bm[s][k] for s in rw])))
    res["paired"]["v_h_median"] = float(np.nanmedian([A[s]["v_h"] for s in rw]))
    f = lambda x: "-" if x is None or not np.isfinite(x) else f"{x:+.2f}"  # noqa: E731
    md = [f"# PAI 281 approach mechanism: {lb} vs {la} (rf_appread.py, EXPLORATORY Amendment 4)", "",
          f"Runway turns where both labels define the strict turn: {len(rw)} of 22. Human entry speed there (median): {res['paired']['v_h_median']:.2f} m/s.", "",
          f"| read | n | {la} median | {lb} median | mean {lb} - {la} [95 % CI] | median diff |", "|:--|--:|--:|--:|:--|--:|"]
    for k in ("v_entry", "excess26", "inward20"):
        p = res["paired"][k]
        md.append(f"| {k} | {p['n']} | {p['a_median']:.2f} | {p['b_median']:.2f} | " + (f"{p['mean']:+.2f} [{p['lo']:+.2f}, {p['hi']:+.2f}] | {p['median']:+.2f} |" if p["n"] else "- | - |"))
    md += ["", "| subset | n | " + " | ".join(f"{lab} completed" for lab in L) + " |", "|:--|--:|" + "--:|" * len(L)]
    md += [f"| {nm} | {c['n']} | " + " | ".join(str(c[lab]) for lab in L) + " |" for nm, c in res["counts"].items()]
    md += ["", f"| scene | runway | {la} done / v_entry / excess26 / inward20 | {lb} done / v_entry / excess26 / inward20 |", "|:--|:--|:--|:--|"]
    for s in strict:
        cell = lambda x: "-" if not x or not x.get("turn") else f"{'Y' if x['completed'] else 'n'} / {f(x['v_entry'])} / {f(x['excess26'])} / {f(x['inward20'])}"  # noqa: E731
        md.append(f"| {s[7:15]} | {'Y' if s in runway else ''} | {cell(A.get(s))} | {cell(Bm.get(s))} |")
    Path(a.out).write_text("\n".join(md) + "\n")
    Path(a.out + ".json").write_text(json.dumps(res, indent=1, default=float))
    print("\n".join(md[:12]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True), ap.add_argument("--b", required=True), ap.add_argument("--strict", required=True)
    ap.add_argument("--runway", required=True), ap.add_argument("--out", required=True)
    main(ap.parse_args())
