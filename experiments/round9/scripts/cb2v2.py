#!/usr/bin/env python3
"""ROUND9 CB2: nuPlan reads under the fixed corridor scorer (alpasim e2e_challenge 1b024f0, PR #215; decisions 453 / 480). AlpaSim venv.

The fix changes only eval/scorers/ground_truth.py. Offline, a nuPlan ASL re-evaluates every metric bit for bit except the three map metrics
(offroad, wrong_lane, min_distance_to_lane_boundary: the runtime's trajdata vec_map is not in the ASL; checked 2026-10-25 on a stored B2
rollout). So a rollout's new metrics = the stored metrics.parquet with every metric the offline evaluator emits replaced by its value under
the eval package on PYTHONPATH (the map metrics stay stored), then AlpaSim's own aggregation. A rollout can change under the fix only if it
is a corridor zero or its plain GT distance reaches 4 m (scorer_v2.py's truncation argument); `prune` keeps the ASL of exactly those
(plus every zero / failed rollout), so a CB2 read is rescored exactly.

  prune --run D                         after a read: drop rollout.asl of scored, unfailed rollouts whose plain GT distance stays < 4 m
  score --runs D.. --out R [--variant old|new] [--workers N]
        mirror under R (asl symlinked, other parquets symlinked), re-evaluate the kept ASLs, merge, aggregate -> R/rollouts.csv
        (clip, score_stored, score, failure reasons, evaluated); variant old must reproduce every stored score (exit 2 otherwise)
  NEW=$DATA_DIR/third_party/alpasim-e2e-1b024f0/src/eval/src; PYTHONPATH=$NEW $AP cb2v2.py score --variant new ...
  table --bench arena|held460 --v2 KEY=CSV.. --tags .. --refs .. --out MD   jevdrive.facts table with the scores of the listed keys
        replaced by their rescored values (zero class of a new zero: corridor-v2)
  rows --v2 KEY=CSV.. --rows IN --out OUT   an arena windows table (tag, id, score, type) with the listed tags' scores replaced (for pcs_local)
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

for _k in ("POLARS_MAX_THREADS", "RAYON_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
REPO = Path(__file__).resolve().parents[3]
CORRIDOR_M = 4.0
MAP_METRICS = ("offroad", "wrong_lane", "min_distance_to_lane_boundary")


def plain_max(parquet):
    import polars as pl
    w = pl.read_parquet(parquet).filter(pl.col("name") == "dist_to_gt_trajectory")
    return float(w["values"].max()) if len(w) else 0.0


def cmd_prune(a):
    D = Path(a.run)
    n = k = 0
    for r in json.loads((D / "aggregate/results-summary.json").read_text())["rollouts"]:
        for f in glob.glob(str(D / "rollouts" / r["clipgt_id"] / "*" / "rollout.asl")):
            pq = Path(f).parent / "metrics.parquet"
            keep = bool(r.get("failure_reason")) or not (r.get("score") or 0) > 0 or not pq.exists() or plain_max(pq) >= CORRIDOR_M
            if keep:
                k += 1
            else:
                os.remove(f)
                n += 1
    print(f"pruned {n} rollout.asl, kept {k} (zero / failed / plain GT distance >= {CORRIDOR_M} m)")


def eval_one(task):
    asl, cfg_path, stored = task
    try:
        import asyncio
        import polars as pl
        from eval.asl_loader import load_scenario_eval_input_from_asl
        from eval.metadata import get_metadata
        from eval.scenario_evaluator import ScenarioEvaluator
        from eval.schema import EvalConfig
        from omegaconf import OmegaConf
        cfg = OmegaConf.merge(EvalConfig, OmegaConf.load(cfg_path))
        inp = asyncio.run(load_scenario_eval_input_from_asl(asl, cfg, {}, get_metadata(Path(cfg_path).parent)))
        import logging
        logging.disable(logging.ERROR)                      # the map scorers log their (expected) missing-vec_map error
        df = ScenarioEvaluator(cfg).evaluate(inp).metrics_df
        st = pl.read_parquet(stored)
        new_names = set(df["name"].unique().to_list())
        keep = st.filter(~pl.col("name").is_in(list(new_names)))
        assert all(n.startswith(MAP_METRICS) for n in keep["name"].unique().to_list()), sorted(set(keep["name"].unique().to_list()))
        out = pl.concat([keep, df.select(st.columns)], how="vertical_relaxed").sort(["name", "timestamps_us"])
        out.write_parquet(Path(asl).parent / "metrics.parquet")
        key = ["name", "timestamps_us"]
        same = st.select(key + ["values", "valid"]).sort(key).equals(out.select(key + ["values", "valid"]).sort(key))
        return dict(asl=asl, ok=True, parquet_same=same)
    except Exception:
        return dict(asl=asl, ok=False, err=traceback.format_exc()[-1500:])


def agg_one(sim):
    try:
        from eval.aggregation.main import _run_aggregation_core
        from eval.schema import EvalConfig
        from omegaconf import OmegaConf
        cfg = OmegaConf.merge(EvalConfig, OmegaConf.load(Path(sim) / "eval-config.yaml"))
        _run_aggregation_core([Path(sim)], Path(sim) / "aggregate", cfg)
        return dict(sim=sim, ok=True)
    except Exception:
        return dict(sim=sim, ok=False, err=traceback.format_exc()[-1500:])


def cmd_shard(a):
    fn = dict(eshard=eval_one, ashard=agg_one)[a.cmd]
    with open(a.out, "a") as f:
        for x in json.loads(Path(a.inp).read_text()):
            f.write(json.dumps(fn(tuple(x) if isinstance(x, list) else x), default=float) + "\n")
            f.flush()


def shards(cmd, items, key, wdir, workers):
    """plain subprocess shards writing JSON lines (scorer_v2.py's pattern; a dead shard only loses its unfinished items, rerun twice)."""
    if wdir.exists():
        shutil.rmtree(wdir)
    wdir.mkdir(parents=True)
    res = {}
    for att in range(3):
        todo = [x for x in items if key(x) not in res]
        if not todo:
            break
        n = min(workers, len(todo))
        ps = []
        for k in range(n):
            fi, fo = wdir / f"a{att}_{k:03d}.json", wdir / f"a{att}_{k:03d}.jsonl"
            fi.write_text(json.dumps(todo[k * len(todo) // n:(k + 1) * len(todo) // n]))
            ps.append((subprocess.Popen([sys.executable, __file__, cmd, str(fi), str(fo)], stdout=subprocess.DEVNULL,
                                        stderr=open(wdir / f"a{att}_{k:03d}.err", "w")), fo))
        for p, fo in ps:
            p.wait()
            for ln in (fo.read_text().splitlines() if fo.exists() else []):
                r = json.loads(ln)
                if r["ok"] or att == 2:
                    res[r.get("asl") or r.get("sim")] = r
        print(f"{cmd}: attempt {att + 1}: {len(res)} / {len(items)}", flush=True)
    return res


def cmd_score(a):
    import inspect
    import eval.scorers.ground_truth as gt
    fixed = "1 m of displacement" in inspect.getsource(gt)
    assert fixed == (a.variant == "new"), f"variant {a.variant} but {gt.__file__} fixed={fixed}"
    R = Path(a.out)
    if R.exists():
        shutil.rmtree(R)                                    # only this lane's own mirror
    tasks, sims, meta = [], [], {}
    for k, src in enumerate(sorted(Path(x) for x in a.runs)):
        dst = R / f"{src.parent.name}" / "run"
        (dst / "rollouts").mkdir(parents=True)
        for f in ("eval-config.yaml", "run_metadata.yaml"):
            shutil.copy2(src / f, dst / f)
        summ = {r["rollout_id"]: r for r in json.loads((src / "aggregate/results-summary.json").read_text())["rollouts"]}
        for rd in sorted(src.glob("rollouts/*/*")):
            if not (rd / "_complete").exists() or not (rd / "metrics.parquet").exists():
                continue
            o = dst / "rollouts" / rd.parent.name / rd.name
            o.mkdir(parents=True)
            (o / "_complete").touch()
            s = summ.get(rd.name, {})
            asl = (rd / "rollout.asl").exists()
            meta[str(o)] = dict(chunk=src.parent.name, clip=rd.parent.name, rollout=rd.name, score_stored=s.get("score"),
                                fr_stored=s.get("failure_reason"), has_asl=asl, plain_max=plain_max(rd / "metrics.parquet"))
            if asl:
                (o / "rollout.asl").symlink_to(rd / "rollout.asl")
                tasks.append([str(o / "rollout.asl"), str(dst / "eval-config.yaml"), str(rd / "metrics.parquet")])
            else:
                (o / "metrics.parquet").symlink_to(rd / "metrics.parquet")
        sims.append(str(dst))
    w = a.workers or len(os.sched_getaffinity(0))
    print(f"{len(sims)} runs, {len(meta)} rollouts, {len(tasks)} ASLs, {w} workers", flush=True)
    ev = shards("eshard", tasks, lambda t: t[0], R / "_shards_e", w)
    bad = [r for r in ev.values() if not r["ok"]] + [dict(asl=t[0], err="never evaluated") for t in tasks if t[0] not in ev]
    ag = shards("ashard", sims, lambda x: x, R / "_shards_a", w)
    bad += [r for r in ag.values() if not r["ok"]] + [dict(sim=s, err="never aggregated") for s in sims if s not in ag]
    if bad:
        (R / "ERROR").write_text(json.dumps(bad, indent=1))
        sys.exit(f"{len(bad)} failures; see {R / 'ERROR'}")
    rows = []
    for sim in sims:
        summ = {r["rollout_id"]: r for r in json.loads((Path(sim) / "aggregate/results-summary.json").read_text())["rollouts"]}
        for o, m in meta.items():
            if o.startswith(sim + "/"):
                s = summ.get(m["rollout"], {})
                e = ev.get(str(Path(o) / "rollout.asl"), {})
                rows.append(m | dict(score=s.get("score"), failure_reason=s.get("failure_reason"), evaluated=bool(e), parquet_same=e.get("parquet_same")))
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(R / "rollouts.csv", index=False)
    risk = int(((df.score_stored > 0) & ~df.has_asl & (df.plain_max >= CORRIDOR_M)).sum())
    diff = df[(df.score - df.score_stored).abs() > 1e-9]
    rec = dict(variant=a.variant, rollouts=len(df), evaluated=int(df.evaluated.sum()), unresolved_risk=risk, changed=len(diff),
               up=int((diff.score > diff.score_stored).sum()), down=int((diff.score < diff.score_stored).sum()),
               mean_stored=float(df.score_stored.mean()), mean=float(df.score.mean()))
    (R / "summary.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec), flush=True)
    if a.variant == "old" and rec["changed"]:
        sys.exit(2)


def _v2map(specs):
    import pandas as pd
    out = {}
    for s in specs:
        k, f = s.split("=", 1)
        d = pd.read_csv(f).drop_duplicates("clip").set_index("clip")
        out[k] = d
    return out


def cmd_table(a):
    sys.path.insert(0, str(REPO))
    import jevdrive.facts as F
    V = _v2map(a.v2)
    orig = F.scores

    def scores(bench, key):
        X = orig(bench, key)
        k = key.split("@")[0] if bench == "arena" and key.endswith("@off") and key.split("@")[0] in V else key
        if k not in V:
            return X
        d = V[k]
        new = d.score.reindex(X.index)
        assert new.notna().all(), f"{key}: {int(new.isna().sum())} windows without a rescored value"
        ft = X.ftype.where(~((new == 0) & (X.score > 0)), "corridor-v2").where(~((new > 0) & (X.score == 0)), "")
        return X.assign(score=new.to_numpy(float), ftype=ft)
    F.scores = scores
    F.cmd_table(argparse.Namespace(bench=a.bench, tags=a.tags, refs=a.refs, out=a.out, title=a.title))


def cmd_rows(a):
    import pandas as pd
    V = _v2map(a.v2)
    R = pd.read_csv(a.rows)
    for k, d in V.items():
        m = R.tag == k
        R.loc[m, "score"] = d.score.reindex(R.id[m]).to_numpy(float)
        assert R.loc[m, "score"].notna().all(), k
    R.to_csv(a.out, index=False)


def main():
    if len(sys.argv) == 4 and sys.argv[1] in ("eshard", "ashard"):
        return cmd_shard(argparse.Namespace(cmd=sys.argv[1], inp=sys.argv[2], out=sys.argv[3]))
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    q = sp.add_parser("prune"); q.add_argument("--run", required=True)
    q = sp.add_parser("score"); q.add_argument("--runs", nargs="+", required=True); q.add_argument("--out", required=True)
    q.add_argument("--variant", default="new", choices=("old", "new")); q.add_argument("--workers", type=int, default=0)
    q = sp.add_parser("table"); q.add_argument("--bench", required=True); q.add_argument("--v2", nargs="+", required=True)
    q.add_argument("--tags", nargs="+", required=True); q.add_argument("--refs", nargs="+", required=True); q.add_argument("--out"); q.add_argument("--title", default="v2")
    q = sp.add_parser("rows"); q.add_argument("--v2", nargs="+", required=True); q.add_argument("--rows", required=True); q.add_argument("--out", required=True)
    a = p.parse_args()
    dict(prune=cmd_prune, score=cmd_score, table=cmd_table, rows=cmd_rows)[a.cmd](a)


if __name__ == "__main__":
    main()
