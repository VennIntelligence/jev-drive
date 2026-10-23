#!/usr/bin/env python3
"""ROUND9 DIAG-READS (plans/2026-10-24-diag-reads.md). report --dir O: PAI 281 ACC + GAP, new scorer primary, old (pai_fix facts) reported.
  A  PCL1L-F-s0 Harmonizer ON vs its stored OFF read, next to NTV0-L18-s0 / R8K-L18-s2 stored OFF + ON reads (common scenes): means, paired
     ON - OFF, zero classes; renderer logs checked for loaded harmonizer weights.
  B  INTP75-NR re-read vs its first read: identical-score scenes, mean |diff|, paired difference, negative flips on NTV0-s0's full-score scenes
     (R8K-s2 and INTP75 first read alongside) -> O/results/diag.{md,json}.
Box, repo root, .venv."""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
DD = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
OK = 0.98
N0, R8 = "NTV0-L18-s0", "R8K-L18-s2"


def f3(d):
    return "%+.3f [%+.3f, %+.3f]" % (d["mean"], d["lo"], d["hi"])


def gpu_hours(o):
    from jevdrive.cl import pool as P
    J = P.snapshot()[0]["jobs"]
    tot = rd = 0.0
    n = 0
    for ln in (o / "jobs.txt").read_text().splitlines():
        j = J.get(ln.split()[0]) if ln.split() else None
        if not j or not j.get("t0") or j["spec"]["vram_gb"] <= 1:
            continue
        h = ((j.get("t1") or time.time()) - j["t0"]) / 3600
        tot += h
        n += 1
        rd += h if j["spec"]["name"].startswith("pai-") else 0.0
    return dict(gpu_job_h=tot, read_h=rd, gpu_jobs=n)


def harm_check(o):
    out = {}
    for d in sorted(glob.glob(str(o / "pai281/PCL1L-F-s0-agH/runs/*/sim/native-logs/renderer-*.log"))):
        t = Path(d).read_text(errors="replace")
        out[str(Path(d).relative_to(o))] = ('"enable_harmonizer": true' in t) and ("harmonizer_nontemporal.pt loaded successfully" in t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sub", choices=["report"])
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    import numpy as np
    import pandas as pd
    from jevdrive import stats
    sys.path.insert(0, str(REPO / "experiments/pai_fix/scripts"))
    import pai_fix as PF
    o = Path(a.dir)
    F_ = DD / "runs/pai_fix/facts"
    PS = pd.read_csv(DD / "runs/alpasim/scorer_v2/report/per_scene.csv", header=[0, 1], index_col=0)["new"]

    def rolls(f, lab):
        R = pd.read_csv(f)
        y = R[R.read == lab]
        assert len(y), f"no new-scorer rollouts for {lab} in {f}"
        return y.drop_duplicates("scene", keep="last").set_index("scene").score

    reads = {   # label -> (old facts csv, new scores)
        "P1L OFF": ("PCL1L-F-s0-281ag", rolls(DD / "runs/round9/p1l/sv2/new/rollouts.csv", "PCL1L-F-s0-ag")),
        "P1L ON": ("PCL1L-F-s0-281agH", rolls(o / "sv2/new/rollouts.csv", "PCL1L-F-s0-agH")),
        "NTV0-s0 OFF": (f"{N0}-281ag", PS[f"{N0}-ag"]), "NTV0-s0 ON": (f"{N0}-281agH", PS[f"{N0}-agH"]),
        "R8K-s2 OFF": (f"{R8}-281ag", PS[f"{R8}-ag"]), "R8K-s2 ON": (f"{R8}-281agH", PS[f"{R8}-agH"]),
        "INTP75 first": ("INTP75-NR-281ag", rolls(DD / "runs/round9/interp/sv2/new/rollouts.csv", "INTP75-NR-ag")),
        "INTP75 re-read": ("INTP75-NR-281ag-rr", rolls(o / "sv2/new/rollouts.csv", "INTP75-NR-ag-rr")),
    }
    OLD = {k: pd.read_csv(F_ / f"{v[0]}.csv").set_index("scene") for k, v in reads.items()}
    NEW = {k: v[1] for k, v in reads.items()}
    res = dict(A={}, B={})

    def block(keys):
        rpc = set().union(*[set(OLD[k].index[OLD[k].rpc.fillna(False).astype(bool)]) for k in keys])
        com = sorted(set.intersection(*[set(OLD[k].index[OLD[k].score.notna()]) for k in keys])
                     & set.intersection(*[set(NEW[k].dropna().index) for k in keys]) - rpc)
        sn = {k: NEW[k].reindex(com).to_numpy(float) for k in keys}
        so = {k: OLD[k].score.reindex(com).to_numpy(float) for k in keys}
        rd = {}
        for k in keys:
            m = PF.metrics(OLD[k].reindex(com))
            rd[k] = dict(new_mean=float(sn[k].mean()), old_mean=float(so[k].mean()), new_zeros=int((sn[k] == 0).sum()), old_zeros=int((so[k] == 0).sum()),
                         zcls={c: int(m.get(f"z_{c}", 0)) for c in PF.ZCLS})
        return com, sn, so, rd, len(rpc)

    kA = ["P1L OFF", "P1L ON", "NTV0-s0 OFF", "NTV0-s0 ON", "R8K-s2 OFF", "R8K-s2 ON"]
    com, sn, so, rd, nrpc = block(kA)
    res["A"] = dict(scenes=len(com), rpc_excluded=nrpc, reads=rd, harmonizer_loaded=harm_check(o), pairs={})
    for m in ("P1L", "NTV0-s0", "R8K-s2"):
        res["A"]["pairs"][f"{m} ON - OFF"] = dict(new=stats.paired(sn[f"{m} ON"], sn[f"{m} OFF"]), old=stats.paired(so[f"{m} ON"], so[f"{m} OFF"]),
                                                   zeros_old=int((so[f"{m} ON"] == 0).sum() - (so[f"{m} OFF"] == 0).sum()))
    for x, y in (("P1L ON", "NTV0-s0 ON"), ("P1L ON", "R8K-s2 ON")):
        res["A"]["pairs"][f"{x} - {y}"] = dict(new=stats.paired(sn[x], sn[y]), old=stats.paired(so[x], so[y]), zeros_old=int((so[x] == 0).sum() - (so[y] == 0).sum()))
    kB = ["INTP75 first", "INTP75 re-read", "NTV0-s0 OFF", "R8K-s2 OFF"]
    com, sn, so, rd, nrpc = block(kB)
    B = res["B"] = dict(scenes=len(com), rpc_excluded=nrpc, reads=rd)
    f, r = "INTP75 first", "INTP75 re-read"
    B["agree"] = {}
    for sc, S in (("new", sn), ("old", so)):
        d = S[r] - S[f]
        B["agree"][sc] = dict(identical=int((np.abs(d) < 1e-9).sum()), mean_abs_diff=float(np.abs(d).mean()), paired=stats.paired(S[r], S[f]))
    B["flips"] = {}
    for sc, S in (("new", sn), ("old", so)):
        full = S["NTV0-s0 OFF"] >= OK
        B["flips"][sc] = {k: dict(ntv0_full=int(full.sum()), neg=int((full & (S[k] < OK)).sum()), neg_zero=int((full & (S[k] == 0)).sum()),
                                  neg_pts=float((S["NTV0-s0 OFF"] - S[k])[full & (S[k] < OK)].sum()))
                          for k in ("R8K-s2 OFF", f, r)}
        ff, rr = full & (S[f] < OK), full & (S[r] < OK)
        B["flips"][sc]["both_reads_neg"] = int((ff & rr).sum())
        B["flips"][sc]["only_first"] = int((ff & ~rr).sum())
        B["flips"][sc]["only_reread"] = int((~ff & rr).sum())
    res["cost"] = gpu_hours(o)
    A = res["A"]
    L = ["# DIAG-READS: P1L Harmonizer ON, INTP75-NR re-read (PAI 281 ACC + GAP)", "",
         "New scorer (scorer_v2 new) primary, old in brackets. Zero classes: old-scorer facts.", "",
         f"## A. P1L with the Harmonizer ON ({A['scenes']} common scenes, {A['rpc_excluded']} RPC scenes excluded)", "",
         "Renderer logs (harmonizer enabled and weights loaded): " + (f"{sum(A['harmonizer_loaded'].values())} / {len(A['harmonizer_loaded'])} chunks" if A["harmonizer_loaded"] else "NO LOGS FOUND"), "",
         "| read | new mean (old) | zeros new (old) | " + " | ".join(PF.ZCLS) + " |", "|:--|--:|--:|" + "--:|" * len(PF.ZCLS)]
    for k, v in A["reads"].items():
        L.append(f"| {k} | {v['new_mean']:.3f} ({v['old_mean']:.3f}) | {v['new_zeros']} ({v['old_zeros']}) | " + " | ".join(str(v["zcls"][c]) for c in PF.ZCLS) + " |")
    L += ["", "| pair (95 % bootstrap CI, scenes as units) | new | old | zeros (old) |", "|:--|--:|--:|--:|"]
    for k, p in A["pairs"].items():
        L.append(f"| {k} | {f3(p['new'])} | {f3(p['old'])} | {p['zeros_old']:+d} |")
    L += ["", f"## B. INTP75-NR re-read vs first read ({B['scenes']} common scenes, {B['rpc_excluded']} RPC scenes excluded)", "",
          "| scorer | identical score | mean abs diff | re-read - first |", "|:--|--:|--:|--:|"]
    for sc, g in B["agree"].items():
        L.append(f"| {sc} | {g['identical']} / {B['scenes']} | {g['mean_abs_diff']:.3f} | {f3(g['paired'])} |")
    for sc in ("new", "old"):
        fl = B["flips"][sc]
        L += ["", f"Negative flips on NTV0-s0's full-score scenes ({sc} scorer; full = score >= {OK}, {fl['R8K-s2 OFF']['ntv0_full']} scenes):", "",
              "| read | negative flips | of them to 0 | points lost there |", "|:--|--:|--:|--:|"]
        for k in ("R8K-s2 OFF", f, r):
            x = fl[k]
            L.append(f"| {k} | {x['neg']} | {x['neg_zero']} | {x['neg_pts']:.2f} |")
        L.append(f"\nIn both INTP75 reads: {fl['both_reads_neg']}; only the first: {fl['only_first']}; only the re-read: {fl['only_reread']}.")
    c = res["cost"]
    L += ["", f"Cost: {c['gpu_job_h']:.2f} GPU job-hours ({c['read_h']:.2f} in the read chunks, {c['gpu_jobs']} GPU jobs)."]
    (o / "results").mkdir(exist_ok=True)
    (o / "results/diag.md").write_text("\n".join(L) + "\n")
    (o / "results/diag.json").write_text(json.dumps(res, indent=1, default=float))
    print("\n".join(L))


if __name__ == "__main__":
    main()
