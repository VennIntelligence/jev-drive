#!/usr/bin/env python3
"""ARENA-HOLD: RSG-L18-s0 with the serve-side static hold (JEV_HOLD, experiments/alpasim/lib/static_hold.py) on the arena's held-split
windows, paired against the stored arena read of RSG-L18-s0 on the same windows (prereg plans/2026-10-17-arena-prereg.md, supplement 1).
Box, repo root, AlpaSim's venv ($P). Driven by arena_hold.sh.

  $P arena_hold.py lists --set F --out DIR [--chunk 150]   the held windows packed per arena shard into read chunks DIR/lists/c<n>.txt
                                                           (<= --chunk windows, one shard each) + DIR/meta.json; the smoke list
                                                           DIR/lists/smoke.txt (12 held windows of one shard: 6 static, 4 launch, 2 other)
  $P arena_hold.py gate --off D --full D --ref-reads DIR --out F   the smoke gate: (1) the sh30hold driver with JEV_HOLD unset (that is
        sh30_driver.main() with nothing patched; the bit-level identity is experiments/alpasim/scripts/hold_check.py's) against the stored
        RSG arena read on the same windows: at most one window's score differs (a closed-loop re-read is not bit-identical: the arena
        smoke's RSG read against the chain's differs in 9 of 120 served plans, up to 0.20 m; plan differences are reported, not gated); (2) JEV_HOLD=full latches (hold.jsonl) on every window whose handoff speed is
        below 0.3 m/s and those rollouts end within 0.5 m of where they started (dist_traveled_m); exit 1 on any failure
  $P arena_hold.py report --set F --hold DIR --ref F --out DIR   the readout (supplement 1, section 4): per variant on the held windows,
        mean score and zero rate, the paired difference to RSG (log-clustered CI) overall and per stratum, gain / loss split, the hold log
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE)]
import arena as A  # noqa: E402

VARIANTS = ("full", "lead", "leash3")


def cmd_lists(a):
    import pandas as pd
    S = pd.read_csv(a.set)
    H = S[S.split == "held"].sort_values(["shard", "log", "start_idx"])
    out = Path(a.out)
    (out / "lists").mkdir(parents=True, exist_ok=True)
    chunks, n = [], 0
    for k, g in H.groupby("shard", sort=True):
        m = -(-len(g) // a.chunk)
        for j, part in enumerate(np.array_split(g.id.values, m)):
            Path(out / f"lists/c{n}.txt").write_text("".join(x + "\n" for x in part))
            chunks.append(dict(chunk=n, shard=int(k), windows=len(part)))
            n += 1
    k0 = int(H.shard.value_counts().idxmax())
    g = H[H.shard == k0]
    sm = pd.concat([g[g.stratum == s].sample(min(c, int((g.stratum == s).sum())), random_state=0) for s, c in (("static", 6), ("launch", 4), ("other", 2))])
    Path(out / "lists/smoke.txt").write_text("".join(x + "\n" for x in sm.id))
    meta = dict(windows=len(H), chunks=chunks, smoke=dict(shard=k0, ids=sm.id.tolist(), strata=sm.stratum.tolist()))
    json.dump(meta, open(out / "meta.json", "w"), indent=1)
    print(json.dumps(dict(windows=len(H), chunks=len(chunks), smoke_shard=k0)))


def _drive(path):
    """drive.jsonl -> {(scene, k): served poses}."""
    d = {}
    for f in glob.glob(str(path)):
        for line in open(f):
            r = json.loads(line)
            if r.get("kind") == "drive":
                d[(r["scene"], r["k"])] = r["poses"]
    return d


def _summary(run):
    return {r["clipgt_id"]: r for r in json.load(open(Path(run) / "aggregate/results-summary.json"))["rollouts"]}


def cmd_gate(a):
    ok = True
    off, full = _summary(a.off), _summary(a.full)
    ref = {}
    for run in glob.glob(f"{a.ref_reads}/c*/run"):
        if (Path(run) / "aggregate/results-summary.json").exists():
            ref.update({k: (v, run) for k, v in _summary(run).items() if k in off})
    L = [f"# ARENA-HOLD smoke gate ({len(off)} windows)", ""]
    miss = sorted(set(off) - set(ref))
    ds = [i for i in off if i in ref and (off[i].get("score"), off[i].get("failure_reason")) != (ref[i][0].get("score"), ref[i][0].get("failure_reason"))]
    D_off = _drive(f"{a.off}/driver-logs/drive.jsonl")
    D_ref = {}
    for run in sorted({r for _, r in ref.values()}):
        D_ref.update(_drive(f"{run}/driver-logs/drive.jsonl"))
    keys = [k for k in D_off if k in D_ref]
    dp = [k for k in keys if D_off[k] != D_ref[k]]
    mx = max((float(np.abs(np.subtract(D_off[k], D_ref[k])).max()) for k in dp), default=0.0)
    L.append(f"1 off: {len(off) - len(miss)} windows against the stored RSG read ({len(miss)} missing), {len(ds)} scores differ {ds}; "
             f"{len(keys)} decisions matched of {len(D_off)}, {len(dp)} served plans differ (max {mx:.3f} m)")
    ok &= not miss and len(ds) <= 1 and len(keys) > 0
    H = [json.loads(x) for f in glob.glob(f"{a.full}/driver-logs/hold.jsonl") for x in open(f)]
    lat = {}
    for r in H:
        if r["state"] in ("held", "free") and r["scene"] not in lat:
            lat[r["scene"]] = (r["state"], r["v"])
    held = [s for s, (st, _) in lat.items() if st == "held"]
    bad_latch = [s for s, (st, v) in lat.items() if (st == "held") != (v < 0.3)]
    moved = [s for s in held if s in full and float((full[s].get("metrics") or {}).get("dist_traveled_m") or 0) > 0.5]
    L.append(f"2 full: {len(lat)} sessions reached the handoff ({len(full)} windows), {len(held)} latched held, {len(bad_latch)} latch errors, "
             f"{len(moved)} held windows moved > 0.5 m")
    ok &= len(lat) == len(full) and not bad_latch and not moved and len(held) > 0
    L += ["", "PASS" if ok else "FAIL"]
    Path(a.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))
    sys.exit(0 if ok else 1)


def cmd_report(a):
    import pandas as pd
    from jevdrive import stats
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    S = pd.read_csv(a.set).set_index("id")
    Hs = S[S.split == "held"]
    R0 = pd.read_csv(a.ref)
    R0 = R0[(R0.tag == "RSG-L18-s0") & R0.id.isin(Hs.index)].drop_duplicates("id").set_index("id")
    rows = {}
    for v in VARIANTS:
        f = Path(a.hold) / "rows.csv.gz"
        R = pd.read_csv(f)
        R = R[(R.tag == v) & R.id.isin(Hs.index)].drop_duplicates("id").set_index("id")
        if len(R):
            rows[v] = R
    bad = set(R0.index[R0.type == "route-fail"]) | {i for R in rows.values() for i in R.index[R.type == "route-fail"]}
    L = ["# ARENA-HOLD readout: RSG-L18-s0 with JEV_HOLD on the arena's held-split windows", "",
         f"Held windows {len(Hs)}; stored RSG read on {len(R0)}; {len(bad)} dropped where a read failed before driving (route-fail). "
         "Paired = variant minus the stored RSG read on the common windows; CIs 95 %, bootstrap clustered by log (jevdrive.stats).", "",
         "| variant | windows | mean score | RSG same windows | paired diff [CI] | zero rate | RSG zero rate | zero-rate diff [CI] | gain (sum) | loss (sum) |",
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    per = []
    for v, R in rows.items():
        c = R.index.intersection(R0.index).difference(bad)
        x, y, grp = R.loc[c, "score"].values, R0.loc[c, "score"].values, Hs.loc[c, "log"].values
        d = stats.paired(x, y, groups=grp)
        dz = stats.paired((x == 0).astype(float), (y == 0).astype(float), groups=grp)
        L.append(f"| {v} | {len(c)} | {x.mean():.4f} | {y.mean():.4f} | {stats.fmt(d, '+.4f')} | {(x == 0).mean():.3f} | {(y == 0).mean():.3f} | "
                 f"{stats.fmt(dz, '+.4f')} | {np.clip(x - y, 0, None).sum():+.1f} | {np.clip(x - y, None, 0).sum():+.1f} |")
        for s in A.STRATA:
            cs = c[Hs.loc[c, "stratum"].values == s]
            if len(cs) == 0:
                continue
            xs, ys = R.loc[cs, "score"].values, R0.loc[cs, "score"].values
            ds = stats.paired(xs, ys, groups=Hs.loc[cs, "log"].values) if len(cs) >= 20 else None
            per.append(f"| {v} | {s} | {len(cs)} | {ys.mean():.3f} / {(ys == 0).mean():.3f} | {xs.mean():.3f} / {(xs == 0).mean():.3f} | "
                       f"{'-' if ds is None else stats.fmt(ds, '+.3f')} | {(xs - ys).sum() / len(c):+.4f} | {np.clip(xs - ys, 0, None).sum():+.1f} | "
                       f"{np.clip(xs - ys, None, 0).sum():+.1f} | {int(((ys > 0) & (xs == 0)).sum())} / {int(((ys == 0) & (xs > 0)).sum())} |")
    L += ["", "By stratum (RSG and variant: mean score / zero rate; contribution = the stratum's summed paired difference / all held windows, "
          "so the strata add up to the overall paired difference; new zeros / fixed zeros):", "",
          "| variant | stratum | windows | RSG | variant | paired diff [CI] | contribution | gain | loss | new / fixed zeros |",
          "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|", *per, ""]
    # zero types and the hold log
    L += ["Zero types on the held windows (AF-front / AF-side / offroad / corridor / zero-other) and failure types of partial scores:", "",
          "| read | " + " | ".join(A.ZC) + " | " + " | ".join(A.PC) + " |", "|---|" + "--:|" * (len(A.ZC) + len(A.PC))]
    for v, R in [("RSG (stored)", R0)] + list(rows.items()):
        c = R.index.intersection(Hs.index).difference(bad)
        t = R.loc[c, "type"]
        zc = t[R.loc[c, "score"] == 0].map(A.zclass)
        L.append(f"| {v} | " + " | ".join(str(int((zc == z).sum())) for z in A.ZC) + " | " + " | ".join(str(int((t == p).sum())) for p in A.PC) + " |")
    L.append("")
    for v in rows:
        H = [json.loads(x) for f in glob.glob(f"{a.hold}/../reads/{v}/c*/run/driver-logs/hold.jsonl") for x in open(f)]
        if not H:
            continue
        Hd = pd.DataFrame(H)
        first = Hd[Hd.state.isin(["held", "free"])].drop_duplicates("scene")
        rel = Hd[Hd.get("release", pd.Series(index=Hd.index, dtype=object)).notna()] if "release" in Hd else Hd.iloc[:0]
        L.append(f"- {v}: {len(first)} sessions past the handoff, {int((first.state == 'held').sum())} latched held; "
                 f"{int(Hd.get('cut', pd.Series(dtype=bool)).fillna(False).sum())} decisions cut; releases {rel.release.value_counts().to_dict() if len(rel) else {}}")
    (out / "hold.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("lists")
    p.add_argument("--set", required=True), p.add_argument("--out", required=True), p.add_argument("--chunk", type=int, default=150)
    p = sp.add_parser("gate")
    p.add_argument("--off", required=True), p.add_argument("--full", required=True), p.add_argument("--ref-reads", required=True), p.add_argument("--out", required=True)
    p = sp.add_parser("report")
    p.add_argument("--set", required=True), p.add_argument("--hold", required=True), p.add_argument("--ref", required=True), p.add_argument("--out", required=True)
    a = ap.parse_args()
    dict(lists=cmd_lists, gate=cmd_gate, report=cmd_report)[a.cmd](a)
