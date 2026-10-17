#!/usr/bin/env python3
"""ARENA: the largest local closed-loop set we can read overnight, private-like: 5.5 s windows placed uniformly in log time over every
public reconstruction (the 1491 AlpaSim nuPlan public scenes' assets), so standing / waiting windows appear at their natural rate inside
the reconstructed spans (prereg plans/2026-10-17-arena-prereg.md). Box, repo root, AlpaSim's venv ($P); the build also needs
PYTHONPATH=$DATA_DIR/envs/nuplan_devkit_overlay (wait_set.py build).

  $P arena.py select [--workers N] [--grid S]   every candidate window on a log-time grid (default 1.0 s) inside each reconstruction,
                                                with the waiting set's coverage / undrawn-vehicle measures and the human-state features
                                                -> $DATA_DIR/runs/alpasim_proxy/arena/select/<ts>/cands.csv.gz
  $P arena.py pick --cands F --step S --out DIR [--shard N] [--chunk N]
                                                the kept windows on the coarser grid S (a multiple of --grid; nested), packed into
                                                log-grouped shards (one trajdata cache each) and read chunks -> DIR/{set.json.gz, shards/}
  $P arena.py check --set F --cache DIR            the cached ego track of every window of one shard (56 steps, asset, start index) -> DIR/check.json
  $P arena.py collect --set F --reads DIR --out F   one row per (tag, window) of every finished read chunk under DIR/<tag>/c*/
  $P arena.py report --set F --rows F --out DIR    tables + figures of the readout (prereg section 4)

Definitions (fixed before any model read; wait_set.py's where named):
  frames      raw lidar_pc rows of a log by timestamp (20 Hz), trajdata's order. A window = 111 frames [w0, w0 + 110] (5.5 s, 10
              decisions); id <log>-<token of frame w0 + 30>; its renderer asset = the source public scene's (central_tokens_config
              asset_folder), as wait_set.py builds it.
  grid        w0 a multiple of the grid in frames counted from the log's first frame (global per log, so two reconstructions of the
              same log that overlap in time give the same window ids: the union of their spans is sampled once).
  inside      (1) time: the window's first and last frame lie within the reconstruction's camera timestamps (+-0.1 s); (2) space:
              ego_cov <= 1.0 m (wait_set.py: the human's 20 Hz poses against the camera frame positions).
  drop        (2) of wait_set.py: an undrawn vehicle in the forward corridor or as the lead (inv_path > 0 or lead_drawn False);
              window outside the log; a duplicate id keeps one row, split = train if any of its source scenes is non-held.
  split       held = the source public scene is in alpasim_nuplan/onpol-held (the L18 on-policy pools were rolled on the others).
  features    v0, v_end (|vx, vy| at the first / last frame), path (10 Hz polyline), maxdisp (max distance from the start), dyaw (|heading
              change| first -> last, deg), as scene_mix.py; stratum, first match: static (maxdisp < 1 m), launch (v0 < 1 m/s), braking
              (v_end < v0 - 2 m/s), turn (dyaw >= 30), cruise (v0 >= 5, |v_end - v0| < 1, dyaw < 5), other (defect_fix df_zero.py's rule).
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(HERE)]
import wait_set as WS  # noqa: E402

DATA = WS.DATA
OUT = DATA / "runs/alpasim_proxy/arena"
HZ, WIN, BEFORE = WS.HZ, WS.WIN, WS.BEFORE
TOL_US = 100_000
STRATA = ("static", "launch", "braking", "turn", "cruise", "other")


def stratum(maxdisp, v0, v_end, dyaw):
    return np.select([maxdisp < 1, v0 < 1, v_end < v0 - 2, dyaw >= 30, (v0 >= 5) & (np.abs(v_end - v0) < 1) & (dyaw < 5)], list(STRATA[:5]), "other")


# ---------------------------------------------------------------- select
def one_log(job):
    log, scenes, held, grid = job
    L = WS.Log(log)
    n, rows = len(L.tok), []
    G = int(round(grid * HZ))
    for sc in scenes:
        asset = WS.yaml_cfg(sc)["road_block_name"]
        pos, vts, models = WS.asset_info(asset)
        i_pub = L.idx.get(sc.rsplit("-", 1)[1])
        ws_pub = None if i_pub is None else i_pub - BEFORE
        lo, hi = vts.min() - TOL_US, vts.max() + TOL_US
        cand = [w0 for w0 in range(0, n - WIN + 1, G) if L.ts[w0] >= lo and L.ts[w0 + WIN - 1] <= hi]
        span = max(0.0, (vts.max() - vts.min()) / 1e6)
        for w0 in cand:
            st = WS.window_stats(L, w0, None, models, pos, vts)
            fr = slice(w0, w0 + WIN)
            p = L.xy[w0:w0 + WIN:2]
            rows.append(dict(id=f"{log}-{L.tok[w0 + BEFORE].hex()}", src=sc, split="held" if sc in held else "train", log=log, asset=asset,
                             token=L.tok[w0 + BEFORE].hex(), start_idx=int(w0), t0=float(L.ts[w0] - L.ts[0]) / 1e6, span_s=span,
                             pub_off_s=None if ws_pub is None else (w0 - ws_pub) / HZ,
                             v0=float(L.v[w0]), v_end=float(L.v[w0 + WIN - 1]), path=float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()),
                             maxdisp=float(np.hypot(*(L.xy[fr] - L.xy[w0]).T).max()),
                             dyaw=float(np.degrees(abs(np.angle(np.exp(1j * (L.yaw[w0 + WIN - 1] - L.yaw[w0])))))), **st))
        if not cand:
            rows.append(dict(src=sc, split="held" if sc in held else "train", log=log, asset=asset, span_s=span, drop="no window inside the video"))
    return rows


def cmd_select(a):
    from jevdrive import par
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("alpasim_proxy", "arena/select", config=vars(a)) as run:
        allsp, held = splits.load("alpasim_nuplan/onpol-all"), splits.load("alpasim_nuplan/onpol-held")
        run.use_split(allsp)
        run.use_split(held)
        hm = set(held.members)
        by = {}
        for s in sorted(allsp.members):
            by.setdefault(s.rsplit("-", 1)[0], []).append(s)
        res = par.pmap(one_log, [(lg, ss, hm, a.grid) for lg, ss in sorted(by.items())], run=run, workers=a.workers, desc="logs")
        res.raise_if_failed()
        rows = [r for rr in res.values for r in rr]
        for r in rows:
            if r.get("drop"):
                continue
            if r["ego_cov"] > WS.COV_EGO:
                r["drop"] = "ego outside the reconstruction's camera frames"
            elif r["inv_path"] > 0 or r["lead_drawn"] is False:
                r["drop"] = "undrawn vehicle in the forward corridor or as the lead"
        # duplicates (overlapping reconstructions of one log): keep the first kept row; split = train if any kept source is non-held
        first, splits_of = {}, {}
        for r in rows:
            if r.get("drop"):
                continue
            splits_of.setdefault(r["id"], set()).add(r["split"])
            if r["id"] in first:
                r["drop"] = "duplicate id (overlapping reconstructions)"
            else:
                first[r["id"]] = r
        for i, r in first.items():
            r["split"] = "train" if "train" in splits_of[i] else "held"
            r["n_src"] = len(splits_of[i])
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with gzip.open(run.path("cands.csv.gz"), "wt", newline="") as fh:
            w = csv.DictWriter(fh, keys)
            w.writeheader()
            w.writerows(rows)
        from collections import Counter
        kept = [r for r in rows if not r.get("drop")]
        summ = dict(grid_s=a.grid, scenes=len(allsp.members), candidates=sum(1 for r in rows if "id" in r), kept=len(kept),
                    kept_held=sum(r["split"] == "held" for r in kept), drops=dict(Counter(r["drop"] for r in rows if r.get("drop"))),
                    strata=dict(Counter(stratum(np.array([r["maxdisp"]]), np.array([r["v0"]]), np.array([r["v_end"]]), np.array([r["dyaw"]]))[0]
                                        for r in kept)))
        run.summary.update(summ)
        print(json.dumps(summ, indent=1))


# ---------------------------------------------------------------- pick: grid, shards, chunks
def cmd_pick(a):
    import pandas as pd
    C = pd.read_csv(a.cands)
    K = C[C["drop"].isna() & C["id"].notna()].copy()
    G = int(round(a.step * HZ))
    K = K[K.start_idx % G == 0].sort_values(["log", "start_idx"]).reset_index(drop=True)
    K["stratum"] = stratum(K.maxdisp.values, K.v0.values, K.v_end.values, K.dyaw.values)
    # shards: whole logs packed in log order up to --shard windows; chunks of --chunk windows inside a shard (log / time order)
    shard, cur, n = [], 0, 0
    for lg, g in K.groupby("log", sort=True):
        if n and n + len(g) > a.shard:
            cur, n = cur + 1, 0
        shard += [cur] * len(g)
        n += len(g)
    K["shard"] = shard
    K["chunk"] = -1
    c = 0
    for s, g in K.groupby("shard", sort=True):
        k = len(g)
        m = -(-k // a.chunk)
        K.loc[g.index, "chunk"] = c + (np.arange(k) * m // k)
        c += m
    out = Path(a.out)
    (out / "lists").mkdir(parents=True, exist_ok=True)
    pub = pd.read_csv(REPO / "experiments/alpasim_proxy/results/scene_mix/public_windows.csv")
    pub = pub[pub.kind == "public"].dropna(subset=["v0"])
    ids = set(K.id)
    shards = []
    for s, g in K.groupby("shard", sort=True):
        # controls: 2 public windows of the shard's logs whose ids are not arena ids (build_cache.py compare against the shipped cache)
        ctl = [dict(id=x, log=x.rsplit("-", 1)[0], token=x.rsplit("-", 1)[1], asset=x) for x in sorted(pub[pub.log.isin(set(g.log))].scene) if x not in ids][:2]
        win = g[["id", "log", "token", "asset", "start_idx", "src", "split"]].to_dict("records")
        json.dump(dict(windows=win, controls=ctl), open(out / f"lists/shard{s}.json", "w"))
        Path(out / f"lists/shard{s}_controls.txt").write_text("".join(x["id"] + "\n" for x in ctl))
        shards.append(dict(shard=int(s), windows=len(g), chunks=sorted(int(x) for x in g.chunk.unique()), controls=len(ctl)))
    for ch, g in K.groupby("chunk", sort=True):
        Path(out / f"lists/c{ch}.txt").write_text("".join(x + "\n" for x in g.id))
    cols = ["id", "src", "split", "log", "asset", "token", "start_idx", "t0", "span_s", "pub_off_s", "v0", "v_end", "path", "maxdisp", "dyaw", "stratum",
            "ego_cov", "near", "inv20", "lead", "lead_gap", "shard", "chunk"]
    K[cols].to_csv(out / "set.csv.gz", index=False)
    meta = dict(step_s=a.step, windows=len(K), held=int((K.split == "held").sum()), logs=int(K.log.nunique()), shards=shards,
                chunks=int(K.chunk.max() + 1), strata={k: int(v) for k, v in K.stratum.value_counts().items()},
                strata_held={k: int(v) for k, v in K[K.split == "held"].stratum.value_counts().items()})
    json.dump(meta, open(out / "meta.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != "shards"}, indent=1))


# ---------------------------------------------------------------- check one shard's cache
def cmd_check(a):
    import dill
    import pandas as pd
    S = json.load(open(a.set))
    rows = []
    for w in S["windows"]:
        d = Path(a.cache) / "nuplan_test" / w["id"]
        m = dill.load(open(d / "scene_metadata_dt0.10.dill", "rb"))
        rows.append(dict(id=w["id"], steps=int(m.length_timesteps), asset_ok=m.data_access_info.get("asset_folder") == w["asset"],
                         start_ok=m.data_access_info.get("start_idx") == w["start_idx"]))
    R = pd.DataFrame(rows)
    ok = (R.steps == 56) & R.asset_ok & R.start_ok
    summ = dict(windows=len(R), ok=int(ok.sum()), bad=R[~ok].id.tolist()[:20])
    json.dump(dict(summary=summ, rows=rows), open(Path(a.cache) / "check.json", "w"), indent=1)
    print(json.dumps(summ))
    sys.exit(0 if ok.all() else 1)


# ---------------------------------------------------------------- collect every read
def failure_type(r):
    """wait_cap nx_regress.failure_type (same order and names; copied: that module imports the op-train stack)."""
    m, sc = r.get("metrics") or {}, float(r.get("score") or 0.0)
    pos = lambda k: (m.get(k) or 0) > 0  # noqa: E731
    if sc == 0:
        z = []
        if pos("collision_at_fault"):
            z.append("AF-front" if pos("collision_front") else "AF-side" if pos("collision_lateral") else "AF-rear")
        z += [n for k, n in (("offroad", "offroad"), ("left_corridor_laterally", "corridor")) if pos(k)]
        return "+".join(z) if z else "route-fail" if str(r.get("failure_reason") or "").startswith("Waypoint") else "zero-other"
    if sc >= 0.98:
        return "ok"
    if pos("collision_any"):
        return "rear-hit" if pos("collision_rear") else "hit"
    dist, gt = float(m.get("dist_traveled_m") or 0.0), float(m.get("gt_dist_traveled_m") or 0.0)
    if gt > 5 and dist < 2:
        return "no-launch"
    ratio = dist / gt if gt > 0 else float("nan")
    return "creep" if ratio < 0.3 else "slow"


def cmd_collect(a):
    rows = []
    for t in sorted(os.listdir(a.reads)):
        for d in sorted(glob.glob(f"{a.reads}/{t}/c*/run")):
            f = Path(d) / "aggregate/results-summary.json"
            if not f.exists():
                continue
            for r in json.loads(f.read_text())["rollouts"]:
                m = r.get("metrics") or {}
                pos = lambda q: int((m.get(q) or 0) > 0)  # noqa: E731
                rows.append(dict(tag=t, id=r["clipgt_id"], chunk=Path(d).parent.name, score=float(r.get("score") or 0.0), type=failure_type(r),
                                 fail=(r.get("failure_reason") or "")[:80], at_fault=pos("collision_at_fault"), front=pos("collision_front"),
                                 side=pos("collision_lateral"), rear=pos("collision_rear"), offroad=pos("offroad"), corridor=pos("left_corridor_laterally"),
                                 dist=float(m.get("dist_traveled_m") or 0.0), gt=float(m.get("gt_dist_traveled_m") or 0.0)))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(a.out, "wt", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print(json.dumps(dict(rows=len(rows), per_tag=dict(Counter(r["tag"] for r in rows)))))


# ---------------------------------------------------------------- report
LABEL = {"RSG-L18-s0": "RSG", "NR1F-L18-s0": "NR1F", "YR10m10-F-s0": "YR", "APY10m10-AB-s0": "APY", "DD-sample": "DD", "LTF-sample": "LTF",
         "starter": "go-straight", "G0014": "garage", "replay": "replay"}
ORDER = list(LABEL)
PRIVATE = {"G0014": 0.848, "YR10m10-F-s0": 0.800, "starter": 0.799, "DD-sample": 0.772, "LTF-sample": 0.738, "APY10m10-AB-s0": 0.640}
PUBLIC = {"G0014": 0.740, "YR10m10-F-s0": 0.925, "starter": 0.522, "DD-sample": 0.797, "LTF-sample": 0.824, "APY10m10-AB-s0": 0.935}
ZC = ("AF-front", "AF-side", "offroad", "corridor", "zero-other")
PC = ("rear-hit", "hit", "no-launch", "creep", "slow")


def zclass(t):
    return t.split("+")[0] if t.startswith("AF-") else "offroad" if "offroad" in t else "corridor" if t == "corridor" else "zero-other"


def cmd_report(a):
    import pandas as pd
    from jevdrive import stats
    out = Path(a.out)
    (out / "figs").mkdir(parents=True, exist_ok=True)
    S = pd.read_csv(a.set)
    R = pd.read_csv(a.rows)
    R = R[R.id.isin(set(S.id))].drop_duplicates(["tag", "id"])
    tags = [t for t in ORDER if t in set(R.tag)] + sorted(set(R.tag) - set(ORDER))
    lab = lambda t: LABEL.get(t, t)  # noqa: E731
    # valid windows (prereg 4.1, amendment 1): no read failed there before driving (route-fail: the map route generation, policy independent);
    # with a replay read also the windows where the human replay fails or scores 0
    bad = set(R[R.type == "route-fail"].id)
    rep = R[R.tag == "replay"].set_index("id")
    if len(rep):
        bad |= set(rep[(rep.score == 0) | (rep.fail.fillna("") != "")].index)
    V = S[~S.id.isin(bad)].set_index("id")
    R = R[R.id.isin(V.index)].copy()
    R["split"], R["stratum"], R["log"] = R.id.map(V.split), R.id.map(V.stratum), R.id.map(V.log)
    R["zero"] = R.score == 0
    R["zc"] = np.where(R.zero, R.type.map(zclass), "")
    R["dv"] = (R["dist"] - R["gt"]) / 5.5
    R["ratio"] = np.where(R["gt"] > 5, R["dist"] / R["gt"].clip(lower=1e-6), np.nan)
    n_read = R.groupby("tag").size()
    complete = [t for t in tags if n_read.get(t, 0) >= 0.98 * len(V)]
    L = ["# ARENA readout", "", f"Set: {len(S)} windows ({int((S.split == 'held').sum())} held-split) on {S.log.nunique()} logs, step "
         f"{a.step} s; {len(bad)} windows dropped where a read failed before driving (route-fail; prereg 4.1) -> {len(V)} valid. Reads complete (>= 98 % of valid windows): "
         + ", ".join(lab(t) for t in complete) + "; partial: " + (", ".join(f"{lab(t)} {n_read[t]}" for t in tags if t not in complete) or "none") + ".",
         "", "Strata of the valid windows (all / held): " + ", ".join(f"{s} {int((V.stratum == s).sum())} / {int(((V.stratum == s) & (V.split == 'held')).sum())}"
                                                                         for s in STRATA) + ". CIs: 95 %, bootstrap clustered by log.", ""]
    parts = (("all", V.index), ("held", V.index[V.split == "held"]), ("non-held", V.index[V.split == "train"]))

    def own(t, ids):
        g = R[(R.tag == t) & R.id.isin(ids)]
        if g.empty:
            return None
        b = stats.bootstrap(g.score.values, groups=g.log.values)
        return g, b

    for name, ids in parts:
        L += [f"## {name} ({len(ids)} windows)", "", "| model | windows read | mean score [CI] | zero rate | " + " | ".join(ZC) + " | " + " | ".join(PC)
              + " | score = 1 | (0, 0.5) | [0.5, 0.9) | [0.9, 1) |", "|---|--:|--:|--:|" + "--:|" * (len(ZC) + len(PC) + 4)]
        for t in tags:
            x = own(t, ids)
            if x is None:
                continue
            g, b = x
            s = g.score
            L.append(f"| {lab(t)} | {len(g)} | {stats.fmt(b)} | {g.zero.mean():.3f} | " + " | ".join(str(int((g.zc == c).sum())) for c in ZC) + " | "
                     + " | ".join(str(int((g.type == c).sum())) for c in PC)
                     + f" | {(s == 1).mean():.3f} | {((s > 0) & (s < 0.5)).mean():.3f} | {((s >= 0.5) & (s < 0.9)).mean():.3f} | {((s >= 0.9) & (s < 1)).mean():.3f} |")
        L.append("")
        if "RSG-L18-s0" in tags:
            L += [f"Paired against RSG ({name}; common windows; model minus RSG):", "", "| model | windows | score diff [CI] | zero-rate diff [CI] | dv diff on windows either scores < 1 (m/s) [CI] |",
                  "|---|--:|--:|--:|--:|"]
            r0 = R[(R.tag == "RSG-L18-s0") & R.id.isin(ids)].set_index("id")
            for t in tags:
                if t in ("RSG-L18-s0", "replay"):
                    continue
                m = R[(R.tag == t) & R.id.isin(ids)].set_index("id")
                c = m.index.intersection(r0.index)
                if len(c) < 30:
                    continue
                grp = r0.loc[c, "log"].values
                d1 = stats.paired(m.loc[c, "score"].values, r0.loc[c, "score"].values, groups=grp)
                d2 = stats.paired(m.loc[c, "zero"].values.astype(float), r0.loc[c, "zero"].values.astype(float), groups=grp)
                u = c[(m.loc[c, "score"] < 1).values | (r0.loc[c, "score"] < 1).values]
                d3 = stats.paired(m.loc[u, "dv"].values, r0.loc[u, "dv"].values, groups=r0.loc[u, "log"].values) if len(u) > 10 else None
                L.append(f"| {lab(t)} | {len(c)} | {stats.fmt(d1, '+.4f')} | {stats.fmt(d2, '+.4f')} | {'-' if d3 is None else stats.fmt(d3, '+.2f')} |")
            L.append("")
        # strata
        L += [f"By stratum ({name}): mean score / zero rate per model; paired score diff vs RSG [CI] in the second table.", "",
              "| stratum | windows | " + " | ".join(lab(t) for t in tags) + " |", "|---|--:|" + "--:|" * len(tags)]
        for s in STRATA:
            si = [i for i in ids if V.at[i, "stratum"] == s]
            cells = []
            for t in tags:
                g = R[(R.tag == t) & R.id.isin(si)]
                cells.append("-" if g.empty else f"{g.score.mean():.3f} / {g.zero.mean():.3f}")
            L.append(f"| {s} | {len(si)} | " + " | ".join(cells) + " |")
        L.append("")
        if "RSG-L18-s0" in tags:
            oth = [t for t in tags if t not in ("RSG-L18-s0", "replay")]
            L += ["| stratum | " + " | ".join(f"{lab(t)} - RSG" for t in oth) + " |", "|---|" + "--:|" * len(oth)]
            for s in STRATA:
                si = [i for i in ids if V.at[i, "stratum"] == s]
                r0 = R[(R.tag == "RSG-L18-s0") & R.id.isin(si)].set_index("id")
                cells = []
                for t in oth:
                    m = R[(R.tag == t) & R.id.isin(si)].set_index("id")
                    c = m.index.intersection(r0.index)
                    cells.append("-" if len(c) < 20 else stats.fmt(stats.paired(m.loc[c, "score"].values, r0.loc[c, "score"].values, groups=r0.loc[c, "log"].values), "+.3f"))
                L.append(f"| {s} | " + " | ".join(cells) + " |")
            L.append("")
        # speed relative to the human
        q = lambda x: " / ".join(f"{v:+.2f}" for v in np.nanquantile(x, [0.1, 0.25, 0.5, 0.75, 0.9])) if np.isfinite(x).sum() else "-"  # noqa: E731
        L += [f"Speed relative to the human ({name}; windows the model scores below 1.0; dv = (ego arc - human arc) / 5.5 s; quantiles q10 / q25 / median / q75 / q90):", "",
              "| model | split | windows | slower / faster | dv (m/s) | arc ratio (human arc > 5 m) | median dv by stratum |", "|---|---|--:|--:|---|---|---|"]
        for t in tags:
            if t == "replay":
                continue
            g = R[(R.tag == t) & R.id.isin(ids) & (R.score < 1)]
            for sp, h in (("zero", g[g.zero]), ("partial", g[~g.zero])):
                if h.empty:
                    continue
                bys = ", ".join(f"{s} {h[h.stratum == s].dv.median():+.2f} ({int((h.stratum == s).sum())})" for s in STRATA if (h.stratum == s).any())
                L.append(f"| {lab(t)} | {sp} | {len(h)} | {int((h.dv < 0).sum())} / {int((h.dv > 0).sum())} | {q(h.dv.values)} | "
                         f"{q(h.ratio.values).replace('+', '')} | {bys} |")
        L.append("")
    # anchors
    L += ["## Private anchors", "", "| anchor | private | arena all [CI] | held | non-held | public 1491 |", "|---|--:|--:|--:|--:|--:|"]
    pubs = PUBLIC
    for t, p in PRIVATE.items():
        cells = []
        for name, ids in parts:
            x = own(t, ids)
            cells.append("-" if x is None else (stats.fmt(x[1]) if name == "all" else f"{x[1]['mean']:.3f}"))
        L.append(f"| {lab(t)} | {p:.3f} | " + " | ".join(cells) + f" | {pubs[t]:.3f} |")
    y, ap = R[R.tag == "YR10m10-F-s0"].set_index("id"), R[R.tag == "APY10m10-AB-s0"].set_index("id")
    c = y.index.intersection(ap.index)
    if len(c) > 30:
        d = stats.paired(y.loc[c, "score"].values, ap.loc[c, "score"].values, groups=y.loc[c, "log"].values)
        verdict = "orders them like private (YR > APY, CI above 0)" if d["lo"] > 0 else "does not separate them" if d["hi"] > 0 else "orders them backwards"
        L += ["", f"YR - APY on {len(c)} common windows: {stats.fmt(d, '+.4f')} (private +0.160, public -0.010): the arena {verdict}."]
    from scipy.stats import kendalltau
    for name, ids in parts:
        have = [t for t in PRIVATE if n_read.get(t, 0) >= 0.98 * len(V)]
        if len(have) >= 3:
            ar = [R[(R.tag == t) & R.id.isin(ids)].score.mean() for t in have]
            tau = kendalltau(ar, [PRIVATE[t] for t in have]).statistic
            tp = kendalltau([PUBLIC[t] for t in have], [PRIVATE[t] for t in have]).statistic
            L.append(f"- {name}: Kendall tau-b of arena score vs private over {len(have)} anchors ({', '.join(lab(t) for t in have)}): {tau:+.2f} (public 1491: {tp:+.2f}); "
                     f"arena order: {' > '.join(lab(t) for t in sorted(have, key=lambda t: -R[(R.tag == t) & R.id.isin(ids)].score.mean()))}")
    L.append("")
    # figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"RSG-L18-s0": "#1f5fa8", "NR1F-L18-s0": "#d9822b", "YR10m10-F-s0": "#6a3d9a", "APY10m10-AB-s0": "#c0392b", "DD-sample": "#2a9d63",
           "LTF-sample": "#8a8a8a", "starter": "#333333", "G0014": "#b0457a"}
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.5))
    for i, (name, ids) in enumerate(parts[1:]):
        ax = axs[i, 0]
        for t, cc in col.items():
            g = R[(R.tag == t) & R.id.isin(ids)]
            if len(g):
                xs = np.sort(g.score.values)
                ax.step(xs, np.arange(1, len(xs) + 1) / len(xs), where="post", color=cc, lw=1.5, label=f"{lab(t)} (mean {g.score.mean():.3f}, zeros {g.zero.mean():.3f})")
        ax.set_title(f"{name}: score CDF", fontsize=10)
        ax.set_xlabel("window score")
        ax.legend(fontsize=7, frameon=False, loc="upper left")
        for j, sp in enumerate(("zero", "partial")):
            ax = axs[i, j + 1]
            bins = np.linspace(-6, 3, 46)
            for t, cc in col.items():
                g = R[(R.tag == t) & R.id.isin(ids) & (R.score < 1)]
                h = g[g.zero] if sp == "zero" else g[~g.zero]
                if len(h):
                    ax.hist(h.dv.clip(-6, 3), bins=bins, histtype="step", lw=1.5, color=cc, weights=np.full(len(h), 1 / len(h)),
                            label=f"{lab(t)} (n={len(h)}, med {h.dv.median():+.2f})")
            ax.axvline(0, color="k", lw=0.8)
            ax.set_title(f"{name}: {sp} windows, ego minus human mean speed", fontsize=10)
            ax.set_xlabel("m/s (clipped to [-6, 3])")
            ax.legend(fontsize=7, frameon=False, loc="upper left")
    for ax in axs.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "figs/arena_scores_speed.png", dpi=120)
    L += ["## Figure", "", "![scores and speed](figs/arena_scores_speed.png)", "",
          "`figs/arena_scores_speed.png`: rows held / non-held; left the CDF of window scores per model (the jump at 0 is the zero rate, the curve's "
          "height just below 1 is the share of imperfect windows); middle / right the ego-minus-human mean speed on zero and partial windows "
          "(each curve normalised to its own count). What to look at: whether the anchors' CDFs stack in their private order (garage, YR, go-straight, "
          "DD, LTF, APY), whether go-straight (holds at standstill) moves up against the launching models as on private, and whether NR1F shifts the "
          "partial-window speed mass toward 0 (closer to the human) against RSG or the zero mass left (stalling).", ""]
    (out / "arena.md").write_text("\n".join(L) + "\n")
    R.to_csv(out / "rows_valid.csv.gz", index=False)
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("select")
    p.add_argument("--workers", type=int, default=48), p.add_argument("--grid", type=float, default=1.0)
    p = sp.add_parser("pick")
    p.add_argument("--cands", required=True), p.add_argument("--step", type=float, required=True), p.add_argument("--out", required=True)
    p.add_argument("--shard", type=int, default=1200), p.add_argument("--chunk", type=int, default=150)
    p = sp.add_parser("check")
    p.add_argument("--set", required=True), p.add_argument("--cache", required=True)
    p = sp.add_parser("collect")
    p.add_argument("--reads", required=True), p.add_argument("--out", required=True)
    p = sp.add_parser("report")
    p.add_argument("--set", required=True), p.add_argument("--rows", required=True), p.add_argument("--out", required=True), p.add_argument("--step", default="?")
    a = ap.parse_args()
    dict(select=cmd_select, pick=cmd_pick, check=cmd_check, collect=cmd_collect, report=cmd_report)[a.cmd](a)
