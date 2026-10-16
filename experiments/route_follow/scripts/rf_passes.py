#!/usr/bin/env python3
"""Perturbed recovery passes (results/rewind_onpol_all/passes.md; runner rf_rewind.sh train64 ostage with RW_PERT / RW_BASE / RW_ROWS).

Pass 0 is decision 319's all-scene rewind (RSG-L18-s0 alone, 30 decisions, expert from h = k_ref - 2 Delta). A pass X drives the same scenes
with JEV_PERTURB on decisions k .. k + n - 1 (branch_poses(g, s)) in the identity and before h in the handover sessions, so that the policy
reaches other states and fails on other turns in other ways; the expert labels are made by the same rule.

  dups   --stage S --base <pass-0 stage dirs, in override order>   per label unit (scene, Delta) of S: same state as pass 0 = pass 0 has the
         unit with the same h and S's identity before h is within the instrument's identity tolerance of pass 0's identity (served plan
         <= 0.15 m at every k < h, anchor <= 0.05 m and speed <= 0.05 m/s at every k <= h): the same expert session twice, label dropped.
         -> S/rows_dedup.csv (rows.csv + column dup, label False on dup units), S/dups.json; displacement of S's identity against pass 0's
         (anchor distance and speed ratio at k 4 / 10 / 20 / 29) -> S/disp.json
  bitid  --a <run dir> --b <run dirs, comma> [--out json]   per scene of a: max |diff| of served plan / anchor / ego over the common k
         against the same scene in b (the switch bit-identity checks); exit 0 when every compared scene is identical
  fails  --base <pass-0 stage dirs> --passes A=<stage>,B=<stage> --out <json>   identity-only yield: turns, failed turns, rewindable units,
         overlap / new against pass 0 and the earlier passes, displacement of the identity against pass 0
  tally  --p0 <pass-0 dir: units.csv, rows.csv, pool report> --passes A=<stage>,B=<stage> [--reports A=<report json>,..] --out <json>
         per pass and in total: strict turns, failed (event / off) turns, handovers, label rows / units / turns / logs, the overlap of failed
         turns with pass 0 and the new failed / label turns each pass adds
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rf_recov_roll import drive_records, speed_of  # noqa: E402

TOL_PLAN, TOL_POS, TOL_V = 0.15, 0.05, 0.05
KS = (4, 10, 20, 29)


def id_records(stages) -> dict:
    """{scene: {k: record}} of the identity sessions of these stage dirs (later stages override earlier)."""
    out = {}
    for st in stages:
        for run in sorted(glob.glob(f"{st}/id/c*/run")):
            if Path(run, "driver-logs/drive.jsonl").exists():
                for sc, rr in drive_records(run).items():
                    out[sc] = {int(r["k"]): r for r in rr}
    return out


def units_h(stages) -> dict:
    """{(scene, Delta): h} from the stages' tfail.json (later stages override earlier)."""
    U = {}
    for st in stages:
        f = Path(st, "tfail.json")
        if f.exists():
            U.update(json.loads(f.read_text())["units"])
    return {(sc, int(d)): h for sc, u in U.items() for d, h in (u.get("h") or {}).items() if h is not None}


def diff(ra, rb):
    """(plan, anchor, speed) max abs differences of two records."""
    return (float(np.abs(np.subtract(ra["poses"], rb["poses"])).max()),
            float(np.hypot(*np.subtract(ra["anchor"][:2], rb["anchor"][:2]))), abs(speed_of(ra) - speed_of(rb)))


def dups(a):
    S, base = Path(a.stage), [x for x in a.base.split(",") if x]
    me, b0 = id_records([S]), id_records(base)
    hS, h0 = units_h([S]), units_h(base)
    rows = list(csv.DictReader(open(S / "rows.csv")))
    lab = sorted({(r["scene"], int(r["delta"])) for r in rows if r["label"] == "True"})
    dup, rec = set(), []
    for sc, d in lab:
        h, hb = hS.get((sc, d)), h0.get((sc, d))
        u = dict(scene=sc, delta=d, h=h, h0=hb)
        if h is not None and h == hb and sc in me and sc in b0:
            A, B = me[sc], b0[sc]
            ks = [k for k in range(h + 1) if k in A and k in B]
            p = max((diff(A[k], B[k])[0] for k in ks if k < h), default=0.0)
            x = max((diff(A[k], B[k])[1] for k in ks), default=np.inf)
            v = max((diff(A[k], B[k])[2] for k in ks), default=np.inf)
            u |= dict(plan=p, pos=x, v=v, same=bool(len(ks) == h + 1 and p <= TOL_PLAN and x <= TOL_POS and v <= TOL_V))
            if u["same"]:
                dup.add((sc, d))
        rec.append(u)
    with open(S / "rows_dedup.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) + ["dup"] if rows else ["dup"])
        w.writeheader()
        for r in rows:
            q = (r["scene"], int(r["delta"])) in dup
            w.writerow(r | dict(dup=str(q), label="False" if q else r["label"]))
    n_rows = lambda keep: sum(r["label"] == "True" and r["within_tol"] == "True" and keep((r["scene"], int(r["delta"]))) for r in rows)  # noqa: E731
    out = dict(label_units=len(lab), dup_units=len(dup), label_rows_tol=n_rows(lambda u: True), dup_rows_tol=n_rows(lambda u: u in dup),
               same_h=sum(u["h"] == u["h0"] for u in rec), units=rec)
    (S / "dups.json").write_text(json.dumps(out, indent=1, default=float))
    # displacement of the perturbed identity against pass 0's (every scene both read)
    D = {k: [] for k in KS}
    vr, ident = [], 0
    com = sorted(set(me) & set(b0))
    for sc in com:
        A, B = me[sc], b0[sc]
        ks = sorted(set(A) & set(B))
        ident += all(diff(A[k], B[k])[:2] == (0.0, 0.0) for k in ks)
        for k in KS:
            if k in A and k in B:
                D[k].append(diff(A[k], B[k])[1])
        if 4 in A and 4 in B and speed_of(B[4]) > 1.0:
            vr.append(speed_of(A[4]) / speed_of(B[4]))
    q = lambda x, p: float(np.percentile(x, p)) if len(x) else None  # noqa: E731
    disp = dict(scenes=len(com), identical_all_k=ident,
                pos={str(k): dict(n=len(x), median=q(x, 50), p90=q(x, 90), gt_0p5=int(np.sum(np.asarray(x) > 0.5)), gt_2=int(np.sum(np.asarray(x) > 2.0)),
                                  zero=int(np.sum(np.asarray(x) == 0.0))) for k, x in D.items()},
                v_ratio_k4=dict(n=len(vr), median=q(vr, 50), p10=q(vr, 10), p90=q(vr, 90)))
    (S / "disp.json").write_text(json.dumps(disp, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "units"}), json.dumps(disp))


def bitid(a):
    A = drive_records(a.a)
    B = {}
    for r in a.b.split(","):
        B.update(drive_records(r))
    res, ok = {}, True
    for sc, rr in sorted(A.items()):
        if sc not in B:
            continue
        Bk = {int(r["k"]): r for r in B[sc]}
        ks = [int(r["k"]) for r in rr if int(r["k"]) in Bk]
        Ak = {int(r["k"]): r for r in rr}
        p = max(float(np.abs(np.subtract(Ak[k]["poses"], Bk[k]["poses"])).max()) for k in ks)
        x = max(float(np.abs(np.subtract(Ak[k]["anchor"], Bk[k]["anchor"])).max()) for k in ks)
        e = max(float(np.abs(np.subtract(Ak[k]["ego"], Bk[k]["ego"])).max()) for k in ks)
        pt = sorted(k for k in ks if "perturb" in Ak[k])
        res[sc] = dict(k=len(ks), n_a=len(rr), n_b=len(Bk), plan=p, anchor=x, ego=e, identical=p == x == e == 0.0, perturbed_k=pt,
                       expert_k=sorted(k for k in ks if "expert" in Ak[k])[:1])
        ok &= res[sc]["identical"] and len(ks) == len(rr) == len(Bk)
    out = dict(a=a.a, b=a.b, scenes=len(res), identical=sum(r["identical"] for r in res.values()), all_identical=bool(ok and res), per_scene=res)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    raise SystemExit(0 if out["all_identical"] else 2)


def tally(a):
    def load(d, rows_name):
        d = Path(d)
        U = list(csv.DictReader(open(d / "units.csv")))
        R = list(csv.DictReader(open(d / rows_name))) if (d / rows_name).exists() else list(csv.DictReader(open(d / "rows.csv")))
        fail = {u["scene"] for u in U if u["outcome"] in ("event", "off")}
        ho = {(u["scene"], d_) for u in U for d_ in (2, 3) if u.get(f"h_D{d_}") not in ("", "None", None)}
        lr = [r for r in R if r["label"] == "True" and r["within_tol"] == "True"]
        return dict(turns={u["scene"] for u in U}, fail=fail, handovers=ho, rows=len(lr), units={(r["scene"], r["delta"]) for r in lr},
                    lturns={r["scene"] for r in lr}, logs={r["log"] for r in lr}, dup_units={(r["scene"], r["delta"]) for r in R if r.get("dup") == "True"})
    P = {"0": load(a.p0, "rows.csv")}
    for spec in a.passes.split(","):
        k, d = spec.split("=")
        P[k] = load(d, "rows_dedup.csv")
    rep = {}
    for spec in (a.reports or "").split(","):
        if spec:
            k, f = spec.split("=")
            rep[k] = json.loads(Path(f).read_text())
    out, seen_f, seen_l = {}, set(), set()
    for k, x in P.items():
        out[k] = dict(strict_turns=len(x["turns"]), failed_turns=len(x["fail"]), handovers=len(x["handovers"]), label_rows=x["rows"],
                      label_units=len(x["units"]), label_turns=len(x["lturns"]), label_logs=len(x["logs"]), dup_units=len(x["dup_units"]),
                      failed_also_in_p0=len(x["fail"] & P["0"]["fail"]), failed_new_vs_p0=len(x["fail"] - P["0"]["fail"]),
                      failed_new_vs_earlier=len(x["fail"] - seen_f), label_turns_new_vs_earlier=len(x["lturns"] - seen_l),
                      p0_failed_not_failed_here=len(P["0"]["fail"] - x["fail"]))
        if k in rep:
            r = rep[k]
            out[k]["pool"] = dict(rows=r["rows"], units=r.get("units"), dev_rows=r["tags"].get("dev", {}).get("1"), kind=r["tags"].get("kind"))
        seen_f |= x["fail"]
        seen_l |= x["lturns"]
    out["total"] = dict(label_rows=sum(x["rows"] for x in P.values()), label_units=sum(len(x["units"]) for x in P.values()),
                        failed_turns_union=len(seen_f), label_turns_union=len(seen_l),
                        label_logs_union=len(set().union(*[x["logs"] for x in P.values()])),
                        pool_rows=sum(out[k]["pool"]["rows"] for k in P if "pool" in out[k]) if rep else None)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def fails(a):
    """Identity-only yield (the handovers were not run): per pass the strict turns, the failed (event / off) turns, the rewindable units
    (h per Delta), their overlap with pass 0 and with the earlier passes, and the identity's displacement against pass 0 -> --out json."""
    base = [x for x in a.base.split(",") if x]
    def units(stages):
        U = {}
        for st in stages:
            f = Path(st, "tfail.json")
            if f.exists():
                U.update(json.loads(f.read_text())["units"])
        return U
    def summ(U):
        turn = {s for s, u in U.items() if u.get("turn")}
        fail = {s for s, u in U.items() if u.get("turn") and u.get("outcome") in ("event", "off")}
        rw = {(s, d) for s, u in U.items() for d, h in (u.get("h") or {}).items() if h is not None}
        return turn, fail, rw
    b0 = id_records(base)
    t0, f0, r0 = summ(units(base))
    out = {"0": dict(turns=len(t0), failed=len(f0), rewindable_units=len(r0), outcome={})}
    for s, u in units(base).items():
        if u.get("turn"):
            out["0"]["outcome"][u["outcome"]] = out["0"]["outcome"].get(u["outcome"], 0) + 1
    seen_f, seen_r = set(f0), set(r0)
    for spec in a.passes.split(","):
        k, st = spec.split("=")
        U = units([st])
        t, f, r = summ(U)
        oc = {}
        for u in U.values():
            if u.get("turn"):
                oc[u["outcome"]] = oc.get(u["outcome"], 0) + 1
        me = id_records([st])
        com = sorted(set(me) & set(b0))
        D = {q: [diff(me[s][q], b0[s][q])[1] for s in com if q in me[s] and q in b0[s]] for q in KS}
        vr = [speed_of(me[s][4]) / speed_of(b0[s][4]) for s in com if 4 in me[s] and 4 in b0[s] and speed_of(b0[s][4]) > 1.0]
        pc = lambda x, p: float(np.percentile(x, p)) if len(x) else None  # noqa: E731
        out[k] = dict(scenes_read=len(me), turns=len(t), failed=len(f), outcome=oc, rewindable_units=len(r),
                      failed_also_p0=len(f & f0), failed_new_vs_p0=len(f - f0), failed_new_vs_earlier=len(f - seen_f),
                      p0_failed_not_here=len(f0 - f), turns_not_in_p0=len(t - t0),
                      rewindable_new_vs_p0=len(r - r0), rewindable_new_vs_earlier=len(r - seen_r),
                      disp_pos={str(q): dict(n=len(x), median=pc(x, 50), p90=pc(x, 90), share_gt_0p5=float(np.mean(np.asarray(x) > 0.5)) if x else None)
                                for q, x in D.items()},
                      v_ratio_k4=dict(n=len(vr), median=pc(vr, 50), p10=pc(vr, 10), p90=pc(vr, 90)),
                      identical_to_p0=sum(all(diff(me[s][q], b0[s][q])[:2] == (0.0, 0.0) for q in set(me[s]) & set(b0[s])) for s in com))
        seen_f |= f
        seen_r |= r
    out["union"] = dict(failed=len(seen_f), rewindable_units=len(seen_r))
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("dups")
    p.add_argument("--stage", required=True), p.add_argument("--base", required=True)
    p = sp.add_parser("bitid")
    p.add_argument("--a", required=True), p.add_argument("--b", required=True), p.add_argument("--out", default="")
    p = sp.add_parser("tally")
    p.add_argument("--p0", required=True), p.add_argument("--passes", required=True), p.add_argument("--reports", default="")
    p.add_argument("--out", required=True)
    p = sp.add_parser("fails")
    p.add_argument("--base", required=True), p.add_argument("--passes", required=True), p.add_argument("--out", required=True)
    a = ap.parse_args()
    dict(dups=dups, bitid=bitid, tally=tally, fails=fails)[a.cmd](a)


if __name__ == "__main__":
    main()
