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
  $P arena_hold.py report --set F --hold DIR --ref F --out DIR [--variants V ...] [--vs TAG]   the readout (supplement 1, section 4): per
        variant on the held windows, mean score and zero rate, the paired difference to RSG (log-clustered CI) overall and per stratum,
        gain / loss split, the hold log; --vs adds the same pairing of every other variant against TAG (supplement 2: stop3 vs leash3)
  $P arena_hold.py diag --set F --corridor DIR --hold DIR --out F [--smoke F]   supplement 2, before its read: the three costs leash3 leaves
        on the held windows, from the stored per-window rows and the per-decision hold logs of full / lead / leash3 (launch windows against
        still-end static windows: serve-time observables and release rules; the AF-front zeros; the unlatched corridor zeros), and the
        offline estimate of stop3 (the serving rule of lib/static_hold.py replayed on the logged speeds); --smoke writes a 12-window smoke list
  $P arena_hold.py gate2 --run D --out F   supplement 2's smoke gate on a stop3 read: every session past the handoff; each latch (static or
        stop) is the rule applied to the logged speeds; held sessions stay within their leash (logged d0 <= cap + 0.3 m); >= 1 stop latch
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
LIB = HERE.parents[1] / "alpasim/lib"


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
    for v in a.variants:
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
    if a.vs and a.vs in rows:
        B = rows[a.vs]
        L += [f"Paired against {a.vs} (the same pairing with {a.vs}'s read in place of RSG's):", "",
              f"| variant | stratum | windows | {a.vs} | variant | paired diff [CI] | contribution | gain | loss | new / fixed zeros |",
              "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for v, R in rows.items():
            if v == a.vs:
                continue
            c = R.index.intersection(B.index).intersection(R0.index).difference(bad)
            for s in ("all",) + A.STRATA:
                cs = c if s == "all" else c[Hs.loc[c, "stratum"].values == s]
                if len(cs) == 0:
                    continue
                xs, ys = R.loc[cs, "score"].values, B.loc[cs, "score"].values
                ds = stats.paired(xs, ys, groups=Hs.loc[cs, "log"].values) if len(cs) >= 20 else None
                L.append(f"| {v} | {s} | {len(cs)} | {ys.mean():.4f} / {(ys == 0).mean():.3f} | {xs.mean():.4f} / {(xs == 0).mean():.3f} | "
                         f"{'-' if ds is None else stats.fmt(ds, '+.4f')} | {(xs - ys).sum() / len(c):+.4f} | {np.clip(xs - ys, 0, None).sum():+.1f} | "
                         f"{np.clip(xs - ys, None, 0).sum():+.1f} | {int(((ys > 0) & (xs == 0)).sum())} / {int(((ys == 0) & (xs > 0)).sum())} |")
        L.append("")
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
        nst = int(first.stop_a.notna().sum()) if "stop_a" in first else 0
        L.append(f"- {v}: {len(first)} sessions past the handoff, {int((first.state == 'held').sum())} latched held ({nst} by the stop rule); "
                 f"{int(Hd.get('cut', pd.Series(dtype=bool)).fillna(False).sum())} decisions cut; releases {rel.release.value_counts().to_dict() if len(rel) else {}}")
    (out / "hold.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


# ---------------------------------------------------------------- supplement 2: diagnosis of leash3's costs, the stop3 estimate, its gate
def _holdlog(pattern):
    import pandas as pd
    return pd.DataFrame([json.loads(x) for f in glob.glob(pattern) for x in open(f)]).drop(columns=["poses"], errors="ignore")


def _auc(pos, neg):
    """Mann-Whitney AUC of a score (higher = positive)."""
    from scipy.stats import rankdata
    x = np.r_[pos, neg]
    r = rankdata(x)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def stop_rule(v0, v1, dt, a_stop=0.5, t_stop=3.0, cap=3.0):
    """lib/static_hold.py's stop<m> latch on arrays (the module is replayed on the logs as well; this is for the sensitivity grid)."""
    a = (v1 - v0) / dt
    hit = (v1 >= 0.3) & (a < -a_stop) & (v1 / np.maximum(-a, 1e-9) <= t_stop)
    return hit, np.where(hit, cap + v1 ** 2 / np.maximum(-2 * a, 1e-9), np.nan)


def stop_estimate(W, hit, C):
    """Offline score of stop3 on the held windows leash3 leaves free (there leash3 = RSG): a flagged window whose RSG ego went past its
    leash C (arc after the handoff, rsg_dist - the human's first 0.5 s) is cut at C. Score then: 0 if RSG had a non-corridor zero (kept,
    conservative); 0 if the human ends still and the ego's stop C lies >= 4 m past the human's rest point; else progress =
    min((pre + C) / (0.8 gt), 1), or 1 when the human travels < 5 m (scene_score). Not cut: RSG's score."""
    pre = W["gt"] - W.arc_after_hand
    reach = W.rsg_dist - pre
    cut = hit & (reach > C)
    prog = np.where(W["gt"] < 5, 1.0, np.minimum((pre + np.minimum(reach, C)) / (0.8 * W["gt"]), 1.0))
    other = W.rsg_type.str.contains("AF|offroad") & ~W.rsg_type.str.contains("corridor")
    cz = W.still & (np.minimum(reach, C) - W.arc_after_hand >= 4.0)
    return np.where(cut, np.where(other | cz, 0.0, prog), W.rsg_score), cut


def cmd_diag(a):
    import importlib
    import os
    import pandas as pd
    H0, CD = Path(a.hold), Path(a.corridor)
    S = pd.read_csv(a.set)
    S = S[S.split == "held"].set_index("id")
    W = S[["log", "shard", "stratum", "v0", "lead", "lead_gap"]].join(pd.read_csv(CD / "hand.csv.gz").set_index("id")[["v_hand", "gt", "arc_after_hand"]])
    W = W.join(pd.read_csv(CD / "gtend_arena.csv.gz").set_index("id")[["seg_last_mm", "end_still_s"]])
    R0 = pd.read_csv(H0 / "report/ref_rows.csv.gz")
    R = pd.concat([R0[R0.tag == "RSG-L18-s0"].assign(tag="rsg"), pd.read_csv(H0 / "report/rows.csv.gz")])
    for t, g in R.groupby("tag"):
        W = W.join(g.drop_duplicates("id").set_index("id")[["score", "type", "dist"]].add_prefix(t + "_"))
    W = W[~(W.filter(like="_type").eq("route-fail").any(axis=1)) & W.leash3_score.notna()].copy()
    N = len(W)
    D = {v: _holdlog(f"{H0}/reads/{v}/c*/run/driver-logs/hold.jsonl") for v in ("lead", "leash3")}
    D = {v: d[d.scene.isin(W.index)] for v, d in D.items()}
    L3 = D["leash3"]
    first = L3[L3.state.isin(["held", "free"])].drop_duplicates("scene").set_index("scene")
    W["latched"], W["still"] = W.index.map(first.state).values == "held", W.seg_last_mm < 1
    W["cls"] = np.select([~W.latched, W.stratum.eq("launch") & (W["gt"] >= 5), W.stratum.eq("launch"), W.still], ["free", "launch, human >= 5 m",
                         "launch, human < 5 m", "static, still end"], "static, moving end")
    W["d_l3"] = W.leash3_score - W.rsg_score
    o = ["# ARENA-HOLD supplement 2: what leash3 still costs on the held windows, and the stop3 estimate", "",
         f"Held windows {N} (route-fail dropped). leash3 against RSG: {W.d_l3.sum():+.1f} summed ({W.d_l3.mean():+.4f}). Serve-time observables "
         "only: the driver's measured speed per call, the model's plan (2 s arc), the checkpoint's lead outputs (lead variant's log, ego held at 0 m).", "",
         "## (a) launch windows", "", "| latched class | windows | RSG | leash3 | leash3 - RSG (sum) |", "|---|--:|--:|--:|--:|"]
    for c, g in W.groupby("cls"):
        o.append(f"| {c} | {len(g)} | {g.rsg_score.mean():.3f} | {g.leash3_score.mean():.3f} | {g.d_l3.sum():+.1f} |")
    H = W[W.latched].copy()
    feat = {}
    for v, col, agg, nm in (("leash3", "arc_model", "max", "plan 2 s arc, max (leash3)"), ("lead", "arc_model", "max", "plan 2 s arc, max (ego held)"),
                            ("lead", "lead_v", "max", "lead speed, max"), ("lead", "lead_p", "median", "lead probability, median (low = go)"),
                            ("lead", "lead_d", "median", "lead distance, median")):
        d = D[v][D[v].t_s >= 1]
        x = d.groupby("scene")[col].agg(agg).reindex(H.index)
        feat[nm] = -x if "low = go" in nm else x
    pos, neg = H.cls.eq("launch, human >= 5 m"), H.cls.eq("static, still end")
    o += ["", f"Separability, launch with the human >= 5 m ({int(pos.sum())}, all of the cost) against still-end static ({int(neg.sum())}), per window "
          "over the calls from 1 s: AUC " + ", ".join(f"{k} {_auc(x[pos].fillna(-99).values, x[neg].fillna(-99).values):.2f}" for k, x in feat.items()) + "."]
    d = D["lead"][D["lead"].t_s >= 1]
    grid = []
    for pmin in (0.0, 0.5, 0.8):
        for vthr in (0.5, 1.0, 1.5, 2.0, 3.0):
            for athr in (0.0, 0.6, 0.8, 1.0, 1.2):
                rel = d[(d.lead_p > pmin) & (d.lead_v > vthr) & (d.arc_model >= athr)].scene.unique()
                r = H[H.index.isin(rel)]
                grid.append((-r.d_l3.sum(), pmin, vthr, athr, len(r), int(r.cls.eq("launch, human >= 5 m").sum()), int(r.cls.eq("static, still end").sum())))
    grid.sort(reverse=True)
    o += ["", "Release rules (release a latched session at the first call from 1 s where lead probability > p, lead speed > v and the plan's 2 s arc >= "
          f"arc; {len(grid)} rules): the released window is credited with RSG's score, so the net is an in-sample upper bound (it ignores the "
          "late release and picks the best of the grid):", "", "| p | v | arc | released | launch >= 5 m | still-end static | net (sum) |", "|--:|--:|--:|--:|--:|--:|--:|"]
    o += [f"| {g[1]} | {g[2]} | {g[3]} | {g[4]} | {g[5]} | {g[6]} | {g[0]:+.1f} |" for g in grid[:5]]
    # (b)
    fz = W[W.leash3_type.str.startswith("AF-front") & ~W.full_type.str.startswith("AF-front")]
    ld = D["lead"][D["lead"].scene.isin(fz.index) & (D["lead"].t_s >= 1)]
    o += ["", "## (b) front collisions", "",
          f"AF-front zeros leash3 {int(W.leash3_type.str.startswith('AF-front').sum())}, full {int(W.full_type.str.startswith('AF-front').sum())}; "
          f"the {len(fz)} that leash3 adds: strata {fz.stratum.value_counts().to_dict()}, {fz.log.nunique()} logs ({fz.groupby('log').size().max()} windows "
          f"from one), every one latched with a stopped lead at a logged gap {fz.lead_gap.min():.2f}-{fz.lead_gap.max():.2f} m that RSG also hits "
          f"(RSG AF-front in {int(fz.rsg_type.str.startswith('AF-front').sum())}); ego travel at the hit {fz.leash3_dist.min():.2f}-{fz.leash3_dist.max():.2f} m. "
          f"The checkpoint's lead head puts that lead at a median {ld.lead_d.median():.1f} m (probability {ld.lead_p.median():.2f}): no serve-time "
          "signal of the gap. A shorter leash for every latch:", "", "| leash (m) | AF-front avoided | launch progress lost (sum) | net (sum) |", "|--:|--:|--:|--:|"]
    lg = H[H.cls.eq("launch, human >= 5 m") & (H.leash3_score > 0)]
    pre = lg["gt"] - lg.arc_after_hand
    for Lm in (2.0, 2.3, 2.5):
        av = int((fz.leash3_dist > Lm).sum())
        lost = (np.minimum((pre + Lm) / (0.8 * lg["gt"]), 1) - np.minimum((pre + 3.0) / (0.8 * lg["gt"]), 1)).sum()
        o.append(f"| {Lm} | {av} | {lost:+.1f} | {av + lost:+.1f} |")
    # (c)
    F = W[~W.latched].copy()
    v = L3.pivot_table(index="scene", columns="t_s", values="v")
    F["v_0"], F["v_h"] = F.index.map(v[0.0]), F.index.map(v[0.5])
    F["acc"] = (F.v_h - F.v_0) / 0.5
    cz = F[F.rsg_type.str.contains("corridor")]
    o += ["", "## (c) corridor zeros the latch leaves", "",
          f"{int(W.leash3_type.str.contains('corridor').sum())} corridor zeros under leash3, all in unlatched sessions and the same windows under full / lead "
          f"(RSG's own). Measured speed at the handoff >= 0.3 m/s in every one; the human brakes and stops inside the window in most:", "",
          "| stratum | windows | handoff speed (m/s), median | deceleration over the 0-0.5 s calls (m/s^2), median | human still for the last (s), median | "
          "human arc after the handoff (m), median | RSG travel (m), median |", "|---|--:|--:|--:|--:|--:|--:|"]
    for st_, g in cz.groupby("stratum"):
        o.append(f"| {st_} | {len(g)} | {g.v_h.median():.2f} | {-g.acc.median():.2f} | {g.end_still_s.median():.1f} | {g.arc_after_hand.median():.2f} | {g.rsg_dist.median():.2f} |")
    o += ["", "A higher static latch threshold (every newly latched session leashed 3 m from the handoff point; same estimate as below):", ""]
    for V in (0.5, 0.8, 1.0):
        g = F[F.v_h < V]
        hit = pd.Series(True, index=g.index)
        sc, cut = stop_estimate(g, hit, pd.Series(3.0, index=g.index))
        o.append(f"- below {V} m/s: {len(g)} more latched ({g.stratum.value_counts().to_dict()}), net {(sc - g.rsg_score).sum():+.1f}")
    os.environ["JEV_HOLD"] = "stop3"
    sys.path.insert(0, str(LIB))
    import static_hold as SH
    SH = importlib.reload(SH)
    hit, C = {}, {}
    for sc_, g in L3[L3.scene.isin(F.index)].sort_values("t0").groupby("scene"):
        st = SH.new()
        for r in g.itertuples():
            if r.t_s > 0.5 + 1e-6:
                break
            _, info = SH.step(st, np.zeros((8, 3)), r.v, (0.0, 0.0), int(r.t0))
        hit[sc_], C[sc_] = st["state"] == "held", st["cap"]
    F["hit"], F["C"] = F.index.map(hit).fillna(False).astype(bool), F.index.map(C)
    F["est"], F["cut"] = stop_estimate(F, F.hit, F.C)
    F["dd"] = F.est - F.rsg_score
    fx = F[F.cut & F.still]
    o += ["", "## stop3: leash3 + a stop leash (lib/static_hold.py, JEV_HOLD=stop3)", "",
          "A session free at the handoff whose measured speed fell by more than 0.5 m/s^2 between the 0 s and the 0.5 s calls, with the "
          "constant-deceleration stop within 3 s, latches with the leash v^2 / (-2a) + 3 m around the handoff point. Replayed on leash3's logged "
          f"speeds (the module itself): {int(F.hit.sum())} sessions latch ({F[F.hit].stratum.value_counts().to_dict()}), {int(F.cut.sum())} of them "
          f"cut (RSG went past the leash). Corridor zeros fixed {int((F.cut & F.rsg_type.str.contains('corridor') & (F.est > 0)).sum())} of {len(cz)}, "
          f"new zeros {int(((F.rsg_score > 0) & (F.est == 0)).sum())}; still-end cut windows: leash end past the human's rest point "
          f"{(fx.C - fx.arc_after_hand).min():.2f}-{(fx.C - fx.arc_after_hand).max():.2f} m (median {(fx.C - fx.arc_after_hand).median():.2f}; a zero needs 4).", "",
          f"**Estimate against leash3: {F.dd.sum():+.1f} summed, {F.dd.sum() / N:+.4f} per held window** (gain {F.dd.clip(lower=0).sum():+.1f}, "
          f"loss {F.dd.clip(upper=0).sum():+.1f}); by stratum " + ", ".join(f"{k} {x:+.1f}" for k, x in F.groupby("stratum").dd.sum().items()) + ".", "",
          "Sensitivity (the same estimate, rule thresholds varied):", "", "| a_stop (m/s^2) | t_stop (s) | margin (m) | latched | net (sum) |", "|--:|--:|--:|--:|--:|"]
    for ast in (0.3, 0.5, 1.0):
        for tst in (2.0, 3.0, 4.0):
            for m in (2.0, 3.0):
                h, c = stop_rule(F.v_0.values, F.v_h.values, 0.5, ast, tst, m)
                e, _ = stop_estimate(F, pd.Series(h, index=F.index), pd.Series(c, index=F.index))
                o.append(f"| {ast} | {tst} | {m} | {int(h.sum())} | {(e - F.rsg_score).sum():+.1f} |")
    Path(a.out).write_text("\n".join(o) + "\n")
    print("\n".join(o))
    if a.smoke:
        k = int(F[F.cut].shard.value_counts().idxmax())
        g, h = F[F.shard == k], W[(W.shard == k) & W.latched]
        ids = list(g[g.cut].sort_values("dd").index[-6:]) + list(h[h.stratum == "static"].index[:3]) + list(g[~g.hit].index[:3])
        Path(a.smoke).write_text("".join(x + "\n" for x in ids))
        json.dump(dict(shard=k, ids=ids), open(Path(a.smoke).with_suffix(".json"), "w"), indent=1)
        print(f"smoke list: {len(ids)} windows of shard {k} -> {a.smoke}")


def cmd_gate2(a):
    Hd = _holdlog(f"{a.run}/driver-logs/hold.jsonl")
    summ = _summary(a.run)
    bad, n_stop, n_static, over = [], 0, 0, []
    for sc, g in Hd.sort_values("t0").groupby("scene"):
        pre, h = g[g.state == "pre"], g[g.state != "pre"]
        if not len(h) or not len(pre):
            bad.append((sc[-16:], "no handoff"))
            continue
        v0, v1, dt, st = pre.v.iloc[-1], h.v.iloc[0], h.t_s.iloc[0] - pre.t_s.iloc[-1], h.state.iloc[0]
        hit, C = stop_rule(np.array([v0]), np.array([v1]), dt)
        want, cap = ("held", 3.0) if v1 < 0.3 else ("held", float(C[0])) if hit[0] else ("free", None)
        if st != want or (want == "held" and v1 >= 0.3 and abs(h["cap"].iloc[0] - cap) > 0.01):
            bad.append((sc[-16:], st, want))
        if want == "held":
            n_stop += v1 >= 0.3
            n_static += v1 < 0.3
            if h.d0.max() > cap + 0.5:
                over.append((sc[-16:], round(h.d0.max(), 2), round(cap, 2)))
    ok = not bad and not over and n_stop > 0 and len(Hd.scene.unique()) == len(summ)
    L = ["# ARENA-HOLD supplement 2 smoke gate (stop3)", "",
         f"{len(Hd.scene.unique())} sessions logged, {len(summ)} windows; latch errors {bad}; {n_static} static latches, {n_stop} stop latches; "
         f"held sessions past their leash + 0.5 m {over}", "", "PASS" if ok else "FAIL"]
    Path(a.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("lists")
    p.add_argument("--set", required=True), p.add_argument("--out", required=True), p.add_argument("--chunk", type=int, default=150)
    p = sp.add_parser("gate")
    p.add_argument("--off", required=True), p.add_argument("--full", required=True), p.add_argument("--ref-reads", required=True), p.add_argument("--out", required=True)
    p = sp.add_parser("report")
    p.add_argument("--set", required=True), p.add_argument("--hold", required=True), p.add_argument("--ref", required=True), p.add_argument("--out", required=True)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS)), p.add_argument("--vs")
    p = sp.add_parser("diag")
    p.add_argument("--set", required=True), p.add_argument("--corridor", required=True), p.add_argument("--hold", required=True)
    p.add_argument("--out", required=True), p.add_argument("--smoke")
    p = sp.add_parser("gate2")
    p.add_argument("--run", required=True), p.add_argument("--out", required=True)
    a = ap.parse_args()
    dict(lists=cmd_lists, gate=cmd_gate, report=cmd_report, diag=cmd_diag, gate2=cmd_gate2)[a.cmd](a)
