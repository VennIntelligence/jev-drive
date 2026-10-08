#!/usr/bin/env python
"""op_parity nt-labels: simulator sub-scores of SH30's navtrain plans on the turn tokens, identity vs speed x 0.8 / x 0.6.

One restartable command (`nt_labels.py run`, every step skips what is finished):
  meta    op_lb-style meta.json of the 12 navtrain_full shards (jevdrive.bench's plan / export stages read it)
  plans   per shard: SH30-F-s0 / s1 plans from the pp_prep token cache (bench.navsim.parity_plans, GPU pool job), then
          op_interp nav-export (adapter base, the bench navtest export) -> (N, 8, 3) rear-axle poses
  assemble  turn tokens (nt_cache plan) x 2 seeds x {id, v080, v060} -> labels/poses.npz (turn_ceiling.transform, identity = export bit for bit)
  score   `python -m jevdrive.bench score-poses --traffic non_reactive --mcache v2_navtrain` (needs the turn shards of nt_cache done)
  table   labels/labels_turn.csv.gz keyed by token + labels/base_rates.md
Outputs: $DATA_DIR/runs/op_parity/nt_cache/labels/. SH30 was TRAINED on these tokens (navtrain_full), so the plans are in-sample.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from jevdrive.common import data_dir

MODELS = ("SH30-F-s0", "SH30-F-s1")
K = 12
VARIANTS = {"id": 1.0, "v080": 0.8, "v060": 0.6}
SUBS = ["NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC"]
SUBCOL = dict(zip(SUBS, ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance", "traffic_light_compliance",
                         "ego_progress", "time_to_collision_within_bound", "lane_keeping", "history_comfort"]))


def O() -> Path:
    return data_dir() / "runs/op_parity/nt_cache"


def L() -> Path:
    return O() / "labels"


def shard_data(i: int) -> str:
    return f"navtrain_full.s{i}of{K}"


def ol(i: int, *p) -> Path:
    return data_dir() / "runs/bench/ol" / shard_data(i) / Path(*p)


def stem(model: str) -> str:
    from jevdrive.bench.models import resolve
    from jevdrive.bench.navsim import stem as st
    return st(resolve(model))


def atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


# ---------------------------------------------------------------- meta + plans (op-train env, GPU) + export (repo venv)
def cmd_plans(a):
    """Pool job for shard a.shard: meta (once), plans of both seeds."""
    import numpy as np
    from jevdrive.bench import navsim as N
    N._pp_path()
    import pp_prep
    i = a.shard
    mf = data_dir() / "runs/op_lb" / shard_data(i) / "meta.json"
    if not mf.exists():
        mt, _, _ = pp_prep.full_meta(shard_data(i))
        mt = {k: ([np.asarray(x).tolist() for x in v] if k in ("pose", "vel") else v) for k, v in mt.items()}
        mf.parent.mkdir(parents=True, exist_ok=True)
        atomic(mf, json.dumps(mt))
    N.ol_root(shard_data(i))
    for m in MODELS:
        out = ol(i, "plans", f"{stem(m)}.npz")
        if not out.exists():
            t0 = time.time()
            N.parity_plans(m, "navtest", str(out), data=shard_data(i))
            print(f"shard {i} {m}: plans in {time.time() - t0:.0f} s", flush=True)


def cmd_export(a):
    import subprocess
    i = a.shard
    for m in MODELS:
        out = N_pred(i, m)
        if not out.exists():
            subprocess.run([sys.executable, str(Path(__file__).resolve().parents[3] / "experiments/op_openloop/lib/op_interp.py"), "nav-export",
                            "--data", shard_data(i), "--adapters", "base", "--plans", stem(m)],
                           env={**os.environ, "OPI_ROOT": "bench/ol"}, check=True)
        assert out.exists(), out


def N_pred(i, m) -> Path:
    return ol(i, "preds", f"{stem(m).replace('@', '-')}__base.npz")


# ---------------------------------------------------------------- assemble
def turn_tokens() -> list:
    return (O() / "plan/tokens_turn.txt").read_text().split()


def cmd_assemble(a):
    import numpy as np
    from jevdrive.run import Run
    import turn_ceiling as TC
    out = L() / "poses.npz"
    if out.exists() and not a.force:
        print("poses.npz exists")
        return
    toks = turn_tokens()
    want = {t: j for j, t in enumerate(toks)}
    with Run("op_parity", "nt_labels/assemble", config=vars(a)) as run:
        P = {m: np.full((len(toks), 8, 3), np.nan, np.float32) for m in MODELS}
        fut = np.full((len(toks), 8, 3), np.nan, np.float32)
        for i in range(K):
            tab = np.load(data_dir() / f"runs/op_parity/cache/{shard_data(i)}/tab.npz")
            for m in MODELS:
                z = np.load(N_pred(i, m))
                assert z["tokens"].tolist() == tab["names"].tolist(), f"shard {i}: export tokens != token cache rows"
                for r, t in enumerate(z["tokens"].tolist()):
                    j = want.get(t)
                    if j is not None:
                        P[m][j] = z["poses"][r]
            idx = {t: r for r, t in enumerate(tab["names"].tolist())}
            for t in toks:
                if t in idx:
                    fut[want[t]] = tab["fut"][idx[t]]
        for m in MODELS:
            miss = int(np.isnan(P[m][:, 0, 0]).sum())
            assert miss == 0, f"{m}: {miss} turn tokens without a plan"
        assert not np.isnan(fut[:, 0, 0]).any()
        out_arr = {"tokens": np.array(toks)}
        ade = {}
        for si, m in enumerate(MODELS):
            for v, sc in VARIANTS.items():
                Q = P[m] if sc == 1.0 else TC.transform(P[m], 0.0, 1.0, sc)
                out_arr[f"s{si}_{v}"] = Q
            ade[m] = float(np.median(np.linalg.norm(P[m][:, :, :2] - fut[:, :, :2], axis=-1).mean(1)))
        L().mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f".poses.{os.getpid()}.npz")
        np.savez(tmp, **out_arr)
        os.replace(tmp, out)
        atomic(L() / "tokens.txt", "\n".join(toks) + "\n")
        run.summary.update(n=len(toks), median_ade_to_log_m=ade)
        run.info("assembled %d tokens x %d keys; median ADE to the logged future %s", len(toks), len(out_arr) - 1, ade)


# ---------------------------------------------------------------- score + table
def cmd_score(a):
    from jevdrive.bench import poses as BP, runner as R
    keys = [f"s{s}_{v}" for s in (0, 1) for v in VARIANTS]
    left = [x["name"] for x in json.loads((O() / "plan/shards_turn.json").read_text()) if not (O() / "done" / f"{x['name']}.json").exists()]
    if left and not a.limit:
        raise SystemExit(f"{len(left)} turn shards of the v2_navtrain cache are not done yet (nt_cache.py run --stage turn); re-run after")
    toks = BP.read_tokens(L() / "tokens.txt") if not a.limit else BP.read_tokens(L() / "tokens.txt")[: a.limit]
    out = L() / (f"score_{a.limit}.csv" if a.limit else "score.csv")
    if out.exists():
        print(out, "exists")
        return
    d = BP.submit(L() / "poses.npz", out, keys, toks, owner="op_parity", traffic="non_reactive", mcache="v2_navtrain", priority=-1.0)
    if not R.wait([d], poll_s=60, quiet=True):
        raise SystemExit(f"score-poses {d} failed; re-run the same command to resume")


def cmd_table(a):
    import numpy as np
    import pandas as pd
    from jevdrive.bench import tables as T
    sc = pd.read_csv(L() / "score.csv")
    toks = (L() / "tokens.txt").read_text().split()
    assert len(sc) == len(toks) * 6 and sc[["key", "token"]].duplicated().sum() == 0, "score.csv incomplete or duplicated"
    wide = {}
    for k, g in sc.groupby("key"):
        g = g.set_index("token").loc[toks]
        for s, c in SUBCOL.items():
            wide[f"{k}_{s}"] = g[c].to_numpy()
        wide[f"{k}_score"] = g["score"].to_numpy()
    df = pd.DataFrame(wide, index=pd.Index(toks, name="token"))
    z = {}
    for i in range(K):
        tab = np.load(data_dir() / f"runs/op_parity/cache/{shard_data(i)}/tab.npz")
        for n, f, v in zip(tab["names"], tab["fut"], tab["speed"]):
            z[n] = T.motion(np.asarray(f, float), float(v))[0]
    df.insert(0, "dyaw", [z[t] for t in toks])
    df.to_csv(L() / "labels_turn.csv.gz", float_format="%.6g")
    big = np.abs(df.dyaw.to_numpy()) >= 45
    rows = []
    for name, m in (("turn >= 20 deg", np.ones(len(df), bool)), ("20-45 deg", ~big), (">= 45 deg", big)):
        for sd in (0, 1):
            i0 = lambda c: df[f"s{sd}_id_{c}"].to_numpy()[m]
            r = dict(bucket=name, seed=sd, n=int(m.sum()), score_id=100 * i0("score").mean(), DAC_fail=(i0("DAC") == 0).mean(),
                     NC_fail=(i0("NC") == 0).mean(), NC_below1=(i0("NC") < 1).mean(),
                     either_fail=((i0("DAC") == 0) | (i0("NC") == 0)).mean())
            for v in ("v080", "v060"):
                for s in ("DAC", "NC"):
                    f = i0(s) == 0
                    ok = df[f"s{sd}_{v}_{s}"].to_numpy()[m] > 0
                    r[f"{v}_{s}_repairs"] = ok[f].mean() if f.any() else np.nan            # of the identity failures, share this variant passes
                    r[f"{v}_{s}_breaks"] = (~ok)[~f].mean()                                # of the identity passes, share it fails
                fe = (i0("DAC") == 0) | (i0("NC") == 0)
                oke = (df[f"s{sd}_{v}_DAC"].to_numpy()[m] > 0) & (df[f"s{sd}_{v}_NC"].to_numpy()[m] > 0)
                r[f"{v}_either_repairs"] = oke[fe].mean() if fe.any() else np.nan
                r[f"{v}_score_gain"] = 100 * (df[f"s{sd}_{v}_score"].to_numpy()[m] - i0("score")).mean()
            best = np.maximum(df[f"s{sd}_v080_score"].to_numpy()[m], df[f"s{sd}_v060_score"].to_numpy()[m])
            r["best_of_id_v080_v060_gain"] = 100 * (np.maximum(best, i0("score")) - i0("score")).mean()
            rows.append(r)
    R = pd.DataFrame(rows)
    R.to_csv(L() / "base_rates.csv", index=False, float_format="%.4f")
    atomic(L() / "base_rates.md", "```\n" + R.round(4).to_string(index=False) + "\n```\n")
    print(R.round(4).to_string(index=False))


def cmd_run(a):
    from jevdrive.bench import runner as R
    L().mkdir(parents=True, exist_ok=True)
    rd = L() / "run"
    me = str(Path(__file__).resolve())
    S = []
    for i in range(K):
        if not all(N_pred(i, m).exists() for m in MODELS):
            S.append(R.Stage(f"pl{i}", [R.py("op-train"), me, "plans", "--shard", str(i)], done=str(ol(i, "plans", f"{stem(MODELS[-1])}.npz")),
                             vram=16, cpu=6, ram=24, tries=2))
            S.append(R.Stage(f"ex{i}", [R.py("jev"), me, "export", "--shard", str(i)], done=str(N_pred(i, MODELS[-1])), vram=0.5, cpu=2, ram=8,
                             after=[f"pl{i}"], tries=2))
    if S:
        S.append(R.Stage("fin", ["touch", str(rd / "DONE")], done=str(rd / "DONE"), vram=0.5, cpu=1, ram=1, after=[s.name for s in S]))
        R.submit(rd, "ntl-plans", S, owner="op_parity", priority=-1.0)
        if not R.wait([rd], poll_s=60, quiet=True) and not all(N_pred(i, m).exists() for i in range(K) for m in MODELS):
            raise SystemExit("plan / export stages failed; re-run the same command to resume")
    cmd_assemble(argparse.Namespace(force=False))
    cmd_score(argparse.Namespace(limit=0))
    cmd_table(a)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    for n in ("plans", "export"):
        sp.add_parser(n).add_argument("--shard", type=int, required=True)
    sp.add_parser("assemble").add_argument("--force", action="store_true")
    sp.add_parser("score").add_argument("--limit", type=int, default=0)
    sp.add_parser("table")
    sp.add_parser("run")
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
