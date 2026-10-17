#!/usr/bin/env python3
"""PAI CORRIDOR: does the nuPlan still-end corridor lever (results/corridor_rule.md, decisions 349 / 350) exist on PAI 281?
CPU only; stored reads are read, never written. Box, repo root, AlpaSim's venv ($P). Note: results/pai_corridor.md.

  $P pai_corridor.py human --runs D [D ...] --out F.csv.gz [--workers N]
        per scene of the run dirs (<D>/sim): the human's recorded rig track (the scorer's GT is this whole track): its length, the last
        segment (mm, cosine to the final heading), how long the human stands still at the end, its speed at the handoff (force-GT ends
        1.7 s after the first metric step) and its largest displacement from the handoff point
  $P pai_corridor.py steps --runs D [D ...] --tag T --out F.csv.gz
        per rollout: the ego's distance from its own handoff position at every metric step up to the corridor trip, the largest of it,
        the progress score, and the progress (normalised projection on the GT) at the trip
  pai_corridor.py report --rows T=F ... --steps T=F ... --human F --out F.md       tables + the offline leash estimate (Mac or box)

End-state classes are corridor_rule.py's ("still": last GT segment < 1 mm; "moving": >= 1 mm and cos >= 0.9; else "off-heading").
Latch = the leash3 rule of lib/static_hold.py: ego speed at the handoff < 0.3 m/s (the force-GT ego speed is the human's).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

CORRIDOR, T_FGT, V_LATCH, LEASH = 4.0, 1.7, 0.3, 3.0
HERE = Path(__file__).resolve().parent


def _scenes(runs):
    out = {}
    for D in runs:
        sim = Path(D) / "sim" if (Path(D) / "sim").exists() else Path(D)
        f = sim / "aggregate/metrics_unprocessed.parquet"
        if not f.exists():
            continue
        import polars as pl
        m = pl.read_parquet(f, columns=["name", "timestamps_us", "clipgt_id"]).filter(pl.col("name") == "dist_to_gt_trajectory")
        for cid, g in m.group_by("clipgt_id"):
            out.setdefault(cid[0], (str(sim / "generated-user-config-0.yaml"), int(g["timestamps_us"].min())))
    return out


def _human(job):
    import os
    sys.path.insert(0, str(Path(os.environ.get("JEV_REPO", HERE.parents[2])) / "experiments/route_follow/scripts"))
    import rf_tax as TX
    cfg, items = job
    L, out = TX.loader(cfg), []
    for s, t0 in items:
        tr = L.get_data_source(s).rig.trajectory
        T, P, Y = np.asarray(tr.timestamps_us, float), np.asarray(tr.positions, float)[:, :2], np.unwrap(np.asarray(tr.yaws, float))
        seg = np.diff(P, axis=0)
        n = np.hypot(*seg.T)
        h = Y[-1]
        th = t0 + T_FGT * 1e6
        xh = np.array([np.interp(th, T, P[:, 0]), np.interp(th, T, P[:, 1])])
        vh = float(np.hypot(np.interp(th + 1e5, T, P[:, 0]) - np.interp(th - 1e5, T, P[:, 0]),
                            np.interp(th + 1e5, T, P[:, 1]) - np.interp(th - 1e5, T, P[:, 1])) / 0.2)
        after = T >= th
        disp = np.hypot(*(P[after] - xh).T)
        go = np.flatnonzero(disp > LEASH)
        rev = np.cumsum(n[::-1])
        out.append(dict(id=s, n_pts=len(P), dt_ms=float(np.median(np.diff(T)) / 1e3), gt_len_raw=float(n.sum()),
                        seg_last_mm=float(n[-1] * 1e3), seg_cos=float(seg[-1] @ [np.cos(h), np.sin(h)] / n[-1]) if n[-1] > 0 else np.nan,
                        end_still_s=float((T[-1] - T[-1 - int(np.argmax(rev > 0.05))]) / 1e6) if n.sum() > 0.05 else float((T[-1] - T[0]) / 1e6),
                        v_hand=vh, disp_max=float(disp.max()) if len(disp) else 0.0,
                        t_leash_s=float((T[after][go[0]] - th) / 1e6) if len(go) else np.nan, track_s=float((T[-1] - T[0]) / 1e6),
                        t0_off_s=float((t0 - T[0]) / 1e6)))
    return out


def cmd_human(a):
    import pandas as pd
    from multiprocessing import Pool
    sc = _scenes(a.runs)
    by = {}
    for s, (cfg, t0) in sc.items():
        by.setdefault(cfg, []).append((s, t0))
    with Pool(a.workers) as p:
        rows = [r for rr in p.map(_human, list(by.items())) for r in rr]
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(json.dumps(dict(scenes=len(rows))))


def cmd_steps(a):
    import pandas as pd
    import polars as pl
    cols = ["dist_to_gt_trajectory", "dist_traveled_m", "progress", "progress_rel_to_total", "dist_to_gt_location"]
    out = []
    for D in a.runs:
        sim = Path(D) / "sim" if (Path(D) / "sim").exists() else Path(D)
        f = sim / "aggregate/metrics_unprocessed.parquet"
        if not f.exists():
            continue
        m = pl.read_parquet(f, columns=["name", "timestamps_us", "values", "clipgt_id", "rollout_id"]).filter(pl.col("name").is_in(cols))
        m = m.with_columns(pl.col("timestamps_us").cast(pl.Int64)).to_pandas()
        for (cid, rid), g in m.groupby(["clipgt_id", "rollout_id"]):
            p = g.pivot_table(index="timestamps_us", columns="name", values="values").sort_index()
            t = (p.index.values - p.index.values[0]) / 1e6
            d, trav = p["dist_to_gt_trajectory"].values, p["dist_traveled_m"].values
            trip = np.flatnonzero(d >= CORRIDOR)
            k = int(trip[0]) if len(trip) else len(d) - 1
            ih = int(np.searchsorted(t, T_FGT - 1e-6))
            dh = trav - trav[ih]                         # arc travelled since the handoff (the leash bounds the straight-line distance)
            out.append(dict(tag=a.tag, id=cid, arc_after_hand=float(dh[-1]), arc_hand_to_trip=float(dh[k]),
                            t_arc3_s=float(t[np.argmax(dh > LEASH)] - T_FGT) if (dh > LEASH).any() else np.nan,
                            prog_trip=float(p["progress"].values[k]), prog_end=float(p["progress"].values[-1]),
                            prt_end=float(p["progress_rel_to_total"].values[-1]), t_trip_s=float(t[k]) if len(trip) else np.nan))
    pd.DataFrame(out).to_csv(a.out, index=False)
    print(json.dumps(dict(tag=a.tag, rollouts=len(out))))


def _boot(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    b = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return x.mean(), np.quantile(b, 0.025), np.quantile(b, 0.975)


def cmd_report(a):
    import pandas as pd
    H = pd.read_csv(a.human).set_index("id")
    H["end"] = np.where(H.seg_last_mm < 1, "still", np.where(H.seg_cos >= 0.9, "moving", "off-heading"))
    H["latch"] = H.v_hand < V_LATCH
    L = ["# PAI corridor: per-step scorer metrics of the stored PAI 281 reads", "",
         f"Human tracks: {len(H)} scenes; GT length median {H.gt_len_raw.median():.1f} m, < 5 m: {(H.gt_len_raw < 5).sum()}; "
         f"end state still / moving / off-heading: {(H.end == 'still').sum()} / {(H.end == 'moving').sum()} / {(H.end == 'off-heading').sum()}; "
         f"human speed at the 1.7 s handoff < {V_LATCH} m/s (leash latch): {int(H.latch.sum())}", "",
         "## Per read: zeros, corridor zeros, and corridor zeros on still-end scenes", "",
         "A corridor zero = score 0 with left_corridor_laterally set (the failure reason can read offroad / collision, which take precedence).", "",
         "| read | rollouts | mean | zeros | failure = corridor | corridor zeros | tripped (d >= 4 m) | still-end scenes | still-end corridor zeros "
         "| of which the trip is past the GT end (progress >= 0.99) | of which latched |",
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    St = {t: pd.read_csv(f).set_index("id") for t, f in (s.split("=", 1) for s in a.steps)}
    R = {}
    for t, f in (s.split("=", 1) for s in a.rows):
        r = pd.read_csv(f).set_index("id").join(H[["end", "latch", "gt_len_raw", "v_hand", "disp_max", "end_still_s", "t_leash_s"]])
        r = r.join(St[t][["arc_after_hand", "arc_hand_to_trip", "t_arc3_s", "prog_trip", "prog_end", "prt_end"]]) if t in St else r
        R[t] = r
        s = r[r.end == "still"]
        sc = s[(s.corridor == 1) & (s.score == 0)]
        L.append(f"| {t} | {len(r)} | {r.score.mean():.4f} | {(r.score == 0).sum()} | {(r.fail == 'left_corridor_laterally').sum()} | "
                 f"{int(((r.corridor == 1) & (r.score == 0)).sum())} | {int(r.tripped.sum())} | {len(s)} | {len(sc)} | "
                 f"{int((sc.prog_trip >= 0.99).sum()) if 'prog_trip' in sc else '-'} | {int(sc.latch.sum())} |")
    L += ["", "## The rule on PAI: corridor zeros given a trip, by the human's end state", "",
          "| read | end | scenes | tripped | corridor zeros | zero / tripped | tripped past the GT end | corridor zeros among them | median lateral / d there |",
          "|---|---|--:|--:|--:|--:|--:|--:|--:|"]
    for t, r in R.items():
        for e, g in r.groupby("end"):
            tr = g[g.tripped]
            z, past = tr[(tr.corridor == 1) & (tr.score == 0)], tr[tr.prog_trip >= 0.99]
            L.append(f"| {t} | {e} | {len(g)} | {len(tr)} | {len(z)} | {len(z) / max(len(tr), 1):.2f} | {len(past)} | "
                     f"{int(((past.corridor == 1) & (past.score == 0)).sum())} | {(past.lat_trip / past.d_trip).median():.2f} |")
    L += ["", "## Latched scenes (human < 0.3 m/s at the handoff): what each read did, and the leash3 estimate", "",
          "Leash3 offline: the ego is kept within 3 m of its handoff point for the whole session. Then d <= 3 m < 4 m (the latch point is on "
          "the GT), so no corridor zero; progress score = 1 if the GT is < 5 m, else min(1, s / (0.8 L)) with s <= 3 m the ego's progress "
          "along the GT (upper bound: s = 3). Collisions / offroad inside the leash are not modelled (nuPlan leash3 added 11 front "
          "collisions on 1 004 latched sessions).", ""]
    lat = H[H.latch]
    L += ["| scene | GT length (m) | end | human end-still (s) | human first > 3 m from the handoff point (s) | "
          + " | ".join(f"{t} score / fail / arc after handoff (m)" for t in R) + " | leash3 score (upper) |",
          "|---|--:|---|--:|--:|" + "---|" * len(R) + "--:|"]
    est = {}
    for s, h in lat.sort_values("gt_len_raw").iterrows():
        lv = 1.0 if h.gt_len_raw < 5 else min(1.0, LEASH / (0.8 * h.gt_len_raw))
        cells = []
        for t, r in R.items():
            if s in r.index:
                x = r.loc[s]
                cells.append(f"{x.score:.3f} / {x.fail if isinstance(x.fail, str) and x.fail else '-'} / "
                             f"{x.arc_after_hand:.1f}" if "arc_after_hand" in r else f"{x.score:.3f}")
                est.setdefault(t, {})[s] = lv - x.score
            else:
                cells.append("missing")
        L.append(f"| {s[7:15]} | {h.gt_len_raw:.1f} | {h.end} | {h.end_still_s:.1f} | "
                 f"{'never' if np.isnan(h.t_leash_s) else f'{h.t_leash_s:.1f}'} | " + " | ".join(cells) + f" | {lv:.3f} |")
    L += ["", "## Net offline estimate of leash3 per read (mean over the read's rollouts; latched scenes only change)", "",
          "| read | latched | still-to-the-end (GT < 5 m) gain | launch scenes cost | net per scene (281) |", "|---|--:|--:|--:|--:|"]
    for t, e in est.items():
        g = sum(v for s, v in e.items() if H.loc[s, "gt_len_raw"] < 5)
        c = sum(v for s, v in e.items() if H.loc[s, "gt_len_raw"] >= 5)
        L.append(f"| {t} | {len(e)} | {g:+.3f} | {c:+.3f} | {(g + c) / len(R[t]):+.4f} |")
    # the still-end corridor zeros that a handoff latch cannot reach: the human stops mid-scene and stays to the end
    L += ["", "## Still-end corridor zeros by how the human got there (all reads)", "",
          "| read | still-end corridor zeros | latched at handoff | human stopped after the handoff (moving at handoff) | trip past the GT end |"
          " median human end-still (s) |", "|---|--:|--:|--:|--:|--:|"]
    for t, r in R.items():
        sc = r[(r.end == "still") & (r.corridor == 1) & (r.score == 0)]
        L.append(f"| {t} | {len(sc)} | {int(sc.latch.sum())} | {int((~sc.latch).sum())} | "
                 f"{int((sc.prog_trip >= 0.99).sum()) if 'prog_trip' in sc else '-'} | {sc.end_still_s.median():.1f} |")
    L += ["", "Still-end scenes with a corridor zero in some read (C = corridor flag):", "", "| scene | GT (m) | v at handoff | human end-still (s) | "
          + " | ".join(f"{t} score / fail / prog at trip" for t in R) + " |", "|---|--:|--:|--:|" + "---|" * len(R)]
    ids = sorted(set().union(*[set(r[(r.end == "still") & (r.corridor == 1) & (r.score == 0)].index) for r in R.values()]))
    for s in ids:
        h = H.loc[s]
        cells = []
        for r in R.values():
            if s not in r.index:
                cells.append("missing")
                continue
            x = r.loc[s]
            cells.append(f"{x.score:.2f} / {x.fail if isinstance(x.fail, str) and x.fail else '-'} / "
                         f"{x.prog_trip:.3f}" + (" C" if x.corridor else ""))
        L.append(f"| {s[7:15]} | {h.gt_len_raw:.1f} | {h.v_hand:.2f} | {h.end_still_s:.1f} | " + " | ".join(cells) + " |")
    Path(a.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("human")
    p.add_argument("--runs", nargs="+", required=True), p.add_argument("--out", required=True), p.add_argument("--workers", type=int, default=8)
    p = sp.add_parser("steps")
    p.add_argument("--runs", nargs="+", required=True), p.add_argument("--tag", required=True), p.add_argument("--out", required=True)
    p = sp.add_parser("report")
    p.add_argument("--rows", nargs="+", required=True), p.add_argument("--steps", nargs="*", default=[]), p.add_argument("--human", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    dict(human=cmd_human, steps=cmd_steps, report=cmd_report)[a.cmd](a)
