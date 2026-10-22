#!/usr/bin/env python3
"""Per-read facts cache: read -> facts -> summary (docs/closed-loop-runbook.md "Read -> facts cache -> summary").

A closed-loop read's per-window facts are extracted once, per chunk, at the end of the read (the read job itself or a CPU job after
it); summaries concatenate and join the cached tables instead of re-extracting every window of every tag from the rollouts.

Benches (one chunk = one AlpaSim run dir; the chunk key is the run's parent dir name):
  arena    nuPlan arena held split    $DATA_DIR/runs/defect_fix/jt/arena/<tag>@<serve>/c<k>/run (jt.sh aread);
                                      RSG-L18-s0@off = the arena's stored read $DATA_DIR/runs/alpasim_proxy/arena/reads/RSG-L18-s0/c*/run
  held460  nuPlan held 460 @18        $DATA_DIR/runs/route_mem/read/<tag>/c<i>/run (rm_chain.sh read)
  pai      PAI 281 / LAUNCH-37 / PAI-FIX: pai_fix.sh tax already writes $DATA_DIR/runs/pai_fix/facts/<name>.csv after each read (one
           row per scene: score, zero class, speed ratio, stratum, ...); `table --bench pai` joins those, nothing is cached here.

Cache: $DATA_DIR/runs/facts/<bench>/<read key>/<chunk>/ with
  scores.parquet   every rollout of the run (first per scene): score, ftype (nx_regress.failure_type), failure_reason, metrics m_*
  dec / win / steps.parquet, hum.pkl   the native_gap extract (experiments/defect_fix/scripts/native_gap.py: one row per decision with the
                   fed speed v, the human's speed vh at the ego's station, turn20, lead gap, plan poses; per window score / type; the
                   scorer's per-step metrics; the human tracks), restricted to the arena held split for `arena`; tag = the read key
  lat.pkl          (arena) the d395 lateral-zero split rows of the chunk's held offroad / corridor zeros (lat_zero.window / classify, the
                   rule of nuplan_dag nd.py latsplit)
  set_census.parquet -> the bench's shared copy (so a chunk dir is itself a native_gap extract dir)
  DONE             json: the results-summary stamp it was built from, counts, timings, window errors
A chunk is rebuilt when its run's results-summary.json changed (a re-read).

  $AP -m jevdrive.facts chunk --bench B --run DIR                 one chunk -> cache (the last step of a read job; never fails the read)
  $AP -m jevdrive.facts fill --bench B --keys K.. [--workers N]   every missing / stale chunk of these reads, all windows in one pool
  $AP -m jevdrive.facts ext --read LABEL=KEY.. --out X           native_gap-format extract (dec / win / steps / hum / set_census) of the
                                                                  arena reads from the cache, filling missing chunks first
  $AP -m jevdrive.facts latsplit --read LABEL=KEY.. --out CSV    nd.py latsplit's table from the cached lat rows
  python -m jevdrive.facts table --bench B --tags T.. --refs R.. --out MD [--title S]
        score-only: per (tag, ref) pair the windows / scenes read by both, means, paired 95 % CI (jevdrive.stats.paired, log-clustered),
        zeros, lateral zeros, and zero classes (arena / held460: nd.py quick's rules; pai: r2.py r281's common scenes)
AP = $DATA_DIR/third_party/alpasim/.venv/bin/python (the scene loader). Tag arguments of `table` are read keys; arena keys without
"@" mean "@off".
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
ROOT = DATA / "runs/facts"
AR = DATA / "runs/alpasim_proxy/arena"
JA = DATA / "runs/defect_fix/jt/arena"
HELD = DATA / "runs/route_mem/read"
PAI = DATA / "runs/pai_fix/facts"
CENSUS = DATA / "runs/wait_cap/census/windows/20261017-142817/windows.parquet"
BENCHES = ("arena", "held460")
LATERAL = ("offroad", "corridor", "AF-side")
for _p in ("experiments/defect_fix/scripts", "experiments/alpasim_proxy/scripts", "experiments/wait_cap/scripts", ""):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))


# ---------------------------------------------------------------- locations
def akey(k):
    """arena read key: a bare tag means its native read (@off)."""
    return k if "@" in k else f"{k}@off"


def read_root(bench, key):
    if bench == "arena":
        return AR / "reads" / key.split("@")[0] if key == "RSG-L18-s0@off" else JA / key
    if bench == "held460":
        return HELD / key
    raise ValueError(bench)


def runs(bench, key):
    return sorted(glob.glob(f"{read_root(bench, key)}/c*/run"))


def key_of(bench, run):
    k = Path(run).parents[1].name
    return f"{k}@off" if bench == "arena" and Path(run).parents[2] == AR / "reads" else k


def cdir(bench, key, run):
    return ROOT / bench / key / Path(run).parent.name


def stamp(run):
    s = (Path(run) / "aggregate/results-summary.json").stat()
    return f"{s.st_mtime_ns}:{s.st_size}"


def fresh(bench, key, run):
    f = cdir(bench, key, run) / "DONE"
    try:
        return json.loads(f.read_text())["stamp"] == stamp(run)
    except (OSError, ValueError, KeyError):
        return False


def arena_set():
    import pandas as pd
    return pd.read_csv(AR / "set/set.csv.gz").set_index("id")


def census(out):
    """native_gap's set_census.parquet (census labels joined to the arena set; static)."""
    import pandas as pd
    S = arena_set()
    Wd = pd.read_parquet(CENSUS)
    Wd = Wd[Wd.kind == "uniform"].copy()
    Wd["kq"] = (Wd.t0 / 0.5).round().astype(int)
    Sx = S.reset_index()
    Sx["kq"] = (Sx.t0 / 0.5).round().astype(int)
    Sx.merge(Wd, on=["log", "kq"], how="left", suffixes=("", "_c")).to_parquet(out)


def shared_census():
    f = ROOT / "arena/set_census.parquet"
    if not f.exists() or f.stat().st_mtime < max((AR / "set/set.csv.gz").stat().st_mtime, CENSUS.stat().st_mtime):
        f.parent.mkdir(parents=True, exist_ok=True)
        census(f.with_suffix(".tmp"))
        os.replace(f.with_suffix(".tmp"), f)
    return f


# ---------------------------------------------------------------- extraction
def _scores(run):
    """every rollout of a run (first per scene, nx_regress.rollouts' rule) -> rows."""
    import arena as A
    rows, seen = [], set()
    for r in json.load(open(Path(run) / "aggregate/results-summary.json"))["rollouts"]:
        if r["clipgt_id"] in seen:
            continue
        seen.add(r["clipgt_id"])
        m = r.get("metrics") or {}
        rows.append(dict(id=r["clipgt_id"], rollout_id=r.get("rollout_id"), score=float(r.get("score") or 0.0), ftype=A.failure_type(r),
                         failure_reason=str(r.get("failure_reason") or ""),
                         **{f"m_{k}": float(v) for k, v in m.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}))
    return rows


def _lat_rows(tag, D, St, Wn, H, S):
    """nd.py latsplit's rows for one read (the d395 rule, lat_zero.classify; a split a2 turn >= 28 deg / a1 other)."""
    import lat_zero as LZ
    held = set(S.index[S.split == "held"])
    rows = []
    z = Wn[(Wn.tag == tag) & (Wn.score == 0) & Wn.id.isin(held)]
    z = z[z.type.fillna("").str.contains("offroad|corridor")]
    for _, m in z.iterrows():
        if m.id not in H:
            rows.append(dict(tag=tag, id=m.id, type=m.type, cls="c: no human track"))
            continue
        w = LZ.window(m.id, tag, D, St, LZ.Track(H[m.id]))
        c = LZ.classify(w, "offroad+corridor" if "offroad" in m.type and "corridor" in m.type else "offroad" if "offroad" in m.type
                        else "corridor" if "corridor" in m.type else m.type)
        sub = ("a2: turn" if w.get("turn_deg", 0) >= 28 else "a1: stop / slow-down") if c.startswith("a") else ""
        rows.append(dict(tag=tag, id=m.id, type=m.type, stratum=S.stratum.get(m.id), cls=c, asub=sub,
                         **{k: w.get(k) for k in ("r3", "slow", "kappa", "turn_deg", "outward", "exit_off", "v_pre", "vh_pre")}))
    return rows


def _lat_job(job):
    import pandas as pd
    d = Path(job)
    key = json.loads((d / "DONE.part").read_text())["key"]
    D, St, Wn = (pd.read_parquet(d / f"{n}.parquet") if (d / f"{n}.parquet").exists() else pd.DataFrame() for n in ("dec", "steps", "win"))
    H = pickle.load(open(d / "hum.pkl", "rb"))
    if not len(Wn):
        return []
    return _lat_rows(key, D, St, Wn, H, arena_set())


def extract(units, workers):
    """units [(bench, key, run)] -> each unit's cache dir. Run files are parsed in one pool; each window's human track and logged traffic
    are loaded once (native_gap._window) for every unit that read it; then per unit: tables, lat rows (arena), DONE."""
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import Pool
    import native_gap as NG
    if not units:
        return
    t = {"start": time.time()}
    S = arena_set()
    held = set(S.index[S.split == "held"])
    with Pool(max(1, min(workers, len(units)))) as p:
        res = p.map(NG._read, [(i, run) for i, (_, _, run) in enumerate(units)])
        sc = p.map(_scores, [run for _, _, run in units])
    t["read"] = time.time()
    D, cfg, W, ST = {}, {}, {}, {}
    for (i, run, dr, rs, st), (bench, key, _) in zip(res, units):
        keep = (lambda x: x in held) if bench == "arena" else (lambda x: True)  # noqa: E731
        for r in dr:
            if keep(r["id"]):
                D.setdefault(r["id"], {}).setdefault(i, []).append(r)
                cfg.setdefault(r["id"], f"{run}/generated-user-config-0.yaml")
        W[i] = [dict(tag=key, id=w, chunk=Path(run).parent.name, **x) for w, x in rs.items() if keep(w)]
        st = st[st.id.map(keep)].copy() if st is not None else pd.DataFrame()
        st.insert(0, "tag", key)
        ST[i] = st
    with ProcessPoolExecutor(max(1, min(workers, len(D)))) as p:      # the loader starts its own workers: not a daemonic Pool
        out = list(p.map(NG._window, [(w, cfg[w], D[w]) for w in sorted(D)], chunksize=2))
    t["windows"] = time.time()
    dec, hum, errs = {}, {}, {}
    for o, (w, h), e in out:
        for r in o:
            dec.setdefault(r["tag"], []).append(r)
        for i in D[w]:
            if h is not None:
                hum.setdefault(i, {})[w] = h
            if e:
                errs.setdefault(i, []).append(e)
    cen = shared_census() if any(b == "arena" for b, _, _ in units) else None
    dirs = []
    for i, (bench, key, run) in enumerate(units):
        d = cdir(bench, key, run)
        d.mkdir(parents=True, exist_ok=True)
        (d / "DONE").unlink(missing_ok=True)
        rows = [dict(r, tag=key) for r in dec.get(i, [])]
        X = pd.DataFrame(rows)
        if len(X):
            X["poses"] = X.poses.map(lambda x: np.asarray(x, np.float32).tolist())
        X.to_parquet(d / "dec.parquet")
        pd.DataFrame(W[i]).to_parquet(d / "win.parquet")
        ST[i].to_parquet(d / "steps.parquet")
        pd.DataFrame(sc[i]).to_parquet(d / "scores.parquet")
        pickle.dump(hum.get(i, {}), open(d / "hum.pkl", "wb"), protocol=4)
        if cen is not None and not (d / "set_census.parquet").exists():
            os.symlink(cen, d / "set_census.parquet")
        (d / "DONE.part").write_text(json.dumps(dict(key=key, run=str(run), stamp=stamp(run))))
        dirs.append((d, bench, key, run, len(rows), len(W[i]), errs.get(i, [])))
    t["write"] = time.time()
    lat = [d for d, b, *_ in dirs if b == "arena"]
    with Pool(max(1, min(workers, len(lat) or 1))) as p:
        L = dict(zip(lat, p.map(_lat_job, lat)))
    t["lat"] = time.time()
    tm = {k: round(t[k] - t[p_], 2) for p_, k in zip(["start", "read", "windows", "write"], ["read", "windows", "write", "lat"])}
    try:
        rev = os.popen(f"git -C {REPO} rev-parse --short HEAD 2>/dev/null").read().strip()
    except OSError:
        rev = ""
    for d, bench, key, run, nd_, nw, er in dirs:
        if d in L:
            pickle.dump(L[d], open(d / "lat.pkl", "wb"), protocol=4)
        (d / "DONE").write_text(json.dumps(dict(key=key, bench=bench, run=str(run), stamp=stamp(run), decisions=nd_, windows=nw, errors=er,
                                                code=rev, batch_units=len(units), batch_seconds=tm, t=time.strftime("%F %T")), indent=1))
        (d / "DONE.part").unlink(missing_ok=True)
    print(json.dumps(dict(units=len(units), windows=len(D), errors=sum(len(x[-1]) for x in dirs), seconds=tm)), flush=True)


def fill(bench, keys, workers, strict=True):
    """extract every missing / stale chunk of the reads in one batch. strict: a chunk without results-summary.json is an error."""
    units, missing = [], []
    for k in keys:
        rr = runs(bench, k)
        if not rr:
            missing.append(f"{k}: no chunk under {read_root(bench, k)}")
        for r in rr:
            if not (Path(r) / "aggregate/results-summary.json").exists():
                missing.append(f"{k}: {r} has no results-summary.json")
            elif not fresh(bench, k, r):
                units.append((bench, k, r))
    if missing and strict:
        raise SystemExit("unread chunks:\n  " + "\n  ".join(missing))
    print(f"fill {bench}: {len(units)} chunks to extract", flush=True)
    extract(units, workers)


def chunk_dirs(bench, key):
    return [cdir(bench, key, r) for r in runs(bench, key)]


# ---------------------------------------------------------------- joins
def load(bench, reads, names=("dec", "win", "steps"), fill_workers=None):
    """{label: key} -> concatenated tables (tag := label), chunk order = sorted run dirs (native_gap's order); hum by window (first label)."""
    import pandas as pd
    if fill_workers:
        fill(bench, list(reads.values()), fill_workers)
    out = {n: [] for n in names}
    hum = {}
    for lab, key in reads.items():
        ds = chunk_dirs(bench, key)
        for d in ds:
            if not (d / "DONE").exists():
                raise SystemExit(f"no facts for {d} (fill first)")
            for n in names:
                x = pd.read_parquet(d / f"{n}.parquet")
                if len(x):
                    if "tag" in x:
                        x["tag"] = lab
                    out[n].append(x)
            if "dec" in names:
                for w, h in pickle.load(open(d / "hum.pkl", "rb")).items():
                    hum.setdefault(w, h)
    return {n: pd.concat(v, ignore_index=True) if v else pd.DataFrame() for n, v in out.items()}, hum


def cmd_chunk(a):
    """end of a read job: never fails it (a missing cache is rebuilt by fill / ext later)."""
    try:
        from jevdrive.common import n_cpus
        key = key_of(a.bench, a.run)
        if fresh(a.bench, key, a.run) and not a.force:
            print(f"facts {a.bench} {key} {Path(a.run).parent.name}: cached")
            return
        extract([(a.bench, key, a.run)], a.workers or n_cpus())
    except Exception as e:  # noqa: BLE001
        print(f"facts chunk {a.run} failed (the read stands; fill rebuilds it): {e!r}", file=sys.stderr)


def cmd_fill(a):
    from jevdrive.common import n_cpus
    keys = [akey(k) if a.bench == "arena" else k for k in a.keys]
    fill(a.bench, keys, a.workers or n_cpus(), strict=not a.allow_missing)


def _reads(xs):
    """LABEL=KEY or LABEL=<read dir> (free_m.py ext's arguments) -> {label: arena key}."""
    def k(v):
        if "/" not in v:
            return akey(v)
        p = Path(v)
        return f"{p.name}@off" if p.parent == AR / "reads" else p.name
    return {x.split("=", 1)[0]: k(x.split("=", 1)[1].rstrip("/")) for x in xs}


def cmd_ext(a):
    """native_gap.cmd_extract's output from the cache: same tables, same row order (windows sorted, labels in --read order, k)."""
    import shutil
    from jevdrive.common import n_cpus
    t0 = time.time()
    R = _reads(a.read)
    T, hum = load("arena", R, fill_workers=a.workers or n_cpus())
    t1 = time.time()
    O = Path(a.out)
    O.mkdir(parents=True, exist_ok=True)
    dec = T["dec"]
    rank = {lab: i for i, lab in enumerate(R)}
    dec = dec.assign(_r=dec.tag.map(rank)).sort_values(["id", "_r", "k"], kind="mergesort").drop(columns="_r").reset_index(drop=True)
    dec.to_parquet(O / "dec.parquet")
    T["win"].to_parquet(O / "win.parquet")
    T["steps"].to_parquet(O / "steps.parquet")
    pickle.dump({w: hum[w] for w in sorted(hum)}, open(O / "hum.pkl", "wb"), protocol=4)
    shutil.copyfile(shared_census(), O / "set_census.parquet")
    print(json.dumps(dict(decisions=len(dec), windows=int(dec.id.nunique()) if len(dec) else 0, load_s=round(t1 - t0, 1),
                          write_s=round(time.time() - t1, 1))), flush=True)


def cmd_latsplit(a):
    import pandas as pd
    from jevdrive.common import n_cpus
    R = _reads(a.read)
    fill("arena", list(R.values()), a.workers or n_cpus())
    rows = []
    for lab, key in R.items():
        for d in chunk_dirs("arena", key):
            rows += [dict(r, tag=lab) for r in pickle.load(open(d / "lat.pkl", "rb"))]
    X = pd.DataFrame(rows)
    X.to_csv(a.out, index=False)
    print(X.groupby(["tag", "cls"]).size().unstack(fill_value=0).to_string())


def scores(bench, key):
    """window / scene id -> scores row (first over the chunks in sorted order)."""
    import pandas as pd
    if bench == "pai":
        return pd.read_csv(PAI / f"{key}.csv").set_index("scene")
    fill(bench, [key], int(os.environ.get("FACTS_WORKERS", "8")))
    X = pd.concat([pd.read_parquet(d / "scores.parquet") for d in chunk_dirs(bench, key)], ignore_index=True)
    return X.drop_duplicates("id").set_index("id")


def cmd_table(a):
    """score-only tables from the cache. arena: held windows read by both, route-fail windows dropped, log-clustered; held460: scenes read by
    both, clustered by the scene's log (id up to the last '-'); pai: scenes every listed read scored without a driver RPC error."""
    import pandas as pd
    from jevdrive import stats
    b = a.bench
    tags = list(dict.fromkeys([*a.tags, *a.refs]))
    key = (lambda t: akey(t)) if b == "arena" else (lambda t: t)  # noqa: E731
    X = {t: scores(b, key(t)) for t in tags}
    lat = lambda s: s.ftype.fillna("").map(lambda f: any(k in f for k in LATERAL)) & (s.score == 0)  # noqa: E731
    if b == "arena":
        S = arena_set().reset_index().drop_duplicates("id").set_index("id")
        held = set(S.index[S.split == "held"])
    elif b == "pai":
        common = sorted(set.intersection(*[set(F.index[F.score.notna() & ~F.rpc.fillna(False).astype(bool)]) for F in X.values()]))
        X = {t: F.reindex(common).assign(ftype=F.reindex(common).fclass) for t, F in X.items()}
    L = [f"# {a.title}: score-only ({b}, facts cache)", "",
         {"arena": "Arena held windows read by both, route-fail windows dropped (free_m arena's rule); log-clustered bootstrap.",
          "held460": "Held 460 @18: scenes read by both; bootstrap clustered by log.",
          "pai": f"PAI: the {len(X[tags[0]])} scenes every read scored without a driver RPC error; scenes as units."}[b]
         + " Paired = arm - ref, 95 % CI (jevdrive.stats.paired).", "",
         "| arm | ref | n | mean arm / ref | paired [CI] | zeros arm / ref | lateral zeros arm / ref |", "|:--|:--|--:|--:|--:|--:|--:|"]
    out = {}
    for t in a.tags:
        for r in [x for x in a.refs if x != t]:
            A_, B_ = X[t], X[r]
            ids = sorted(set(A_.index) & set(B_.index) & (held if b == "arena" else set(A_.index)))
            if b == "arena":
                ids = [i for i in ids if A_.ftype[i] != "route-fail" and B_.ftype[i] != "route-fail"]
            if not ids:
                continue
            x, y = A_.score.reindex(ids).to_numpy(float), B_.score.reindex(ids).to_numpy(float)
            g = (S.log.reindex(ids).values if b == "arena" else np.array([i.rsplit("-", 1)[0] for i in ids]) if b == "held460" else None)
            p = stats.paired(x, y, groups=g)
            la, lb = int(lat(A_.loc[ids]).sum()), int(lat(B_.loc[ids]).sum())
            out[f"{t}|{r}"] = dict(n=len(ids), a=float(x.mean()), b=float(y.mean()), **{k: float(p[k]) for k in ("mean", "lo", "hi")},
                                   zeros_a=int((x == 0).sum()), zeros_b=int((y == 0).sum()), lat_a=la, lat_b=lb)
            L.append(f"| {t} | {r} | {len(ids)} | {x.mean():.4f} / {y.mean():.4f} | {p['mean']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}] | "
                     f"{(x == 0).sum()} / {(y == 0).sum()} | {la} / {lb} |")
    L += ["", "Zero classes" + (" (held windows)" if b == "arena" else "") + ":", "", "| class | " + " | ".join(tags) + " |", "|:--|" + "--:|" * len(tags)]
    C = {t: Counter(F.ftype[(F.score == 0) & (F.index.isin(held) if b == "arena" else True)]) for t, F in X.items()}
    for k in sorted({k for c in C.values() for k in c}, key=lambda k: -sum(c.get(k, 0) for c in C.values())):
        L.append(f"| {k} | " + " | ".join(str(C[t].get(k, 0)) for t in tags) + " |")
    out["zeros"] = {t: dict(C[t]) for t in tags}
    text = "\n".join(L) + "\n"
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)
        Path(a.out).with_suffix(".json").write_text(json.dumps(out, indent=1, default=float))
    print(text, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    q = sp.add_parser("chunk")
    q.add_argument("--bench", choices=BENCHES, required=True), q.add_argument("--run", required=True)
    q.add_argument("--workers", type=int, default=0), q.add_argument("--force", action="store_true")
    q = sp.add_parser("fill")
    q.add_argument("--bench", choices=BENCHES, required=True), q.add_argument("--keys", nargs="+", required=True)
    q.add_argument("--workers", type=int, default=0), q.add_argument("--allow-missing", action="store_true")
    for n in ("ext", "latsplit"):
        q = sp.add_parser(n)
        q.add_argument("--read", nargs="+", required=True), q.add_argument("--out", required=True), q.add_argument("--workers", type=int, default=0)
    q = sp.add_parser("table")
    q.add_argument("--bench", choices=(*BENCHES, "pai"), required=True), q.add_argument("--tags", nargs="+", required=True)
    q.add_argument("--refs", nargs="+", required=True), q.add_argument("--out"), q.add_argument("--title", default="Read")
    a = p.parse_args()
    dict(chunk=cmd_chunk, fill=cmd_fill, ext=cmd_ext, latsplit=cmd_latsplit, table=cmd_table)[a.cmd](a)


if __name__ == "__main__":
    main()
