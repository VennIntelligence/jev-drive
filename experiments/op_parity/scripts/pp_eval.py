"""op_parity NAVSIM readout (plans/2026-10-06-parity-prereg.md): navtest plans of every arm from the pp_prep cache, the official v2 devkit
(navsim main @ 0a380a9, EPDMS, the harness that reproduced WA-JEPA's 91.71), paired bootstraps over navtest logs.

  plans   --models P0 P1-init P2-init P3-init P2-s0 P3-s0 P3-s0:noside ...
          per model: plan at t0 of every lb_navtest token -> $DATA_DIR/runs/op_lb/lb_navtest/plans/gimm@cinque_PP<model>.npz (op_lb format);
          `:noside` feeds P3 with every side camera masked (ablation). Also writes equivalence.json: max / p99 |plan - P0| per model, and
          P0 (port) vs the stored ONNX run `gimm@cinque.npz` (the shipped model on onnxruntime / TensorRT).
  score   --models ... [--ver v2]   nav-export (op_interp adapter `base`) + experiments/zeroshot_openloop/archive/navsim_zs_score.sh per stem
  report  --models ... [--ref P1-s0]  per-token CSVs -> results/navtest_<tag>.md / .csv: EPDMS + subscores per arm, paired diffs vs the ref arm
          and vs WA-JEPA's per-token scores (same tokens, same devkit and metric cache), bootstrap over logs (jevdrive.stats)
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_pl.Path(__file__).parent)]
import argparse, json, os, subprocess  # noqa: E401,E402

import numpy as np  # noqa: E402

from jevdrive.common import data_dir  # noqa: E402

DATA, FRAMES = "lb_navtest", "gimm"       # set from --data / --frames (Stage B: lb_hq_navtestX, protocols warp / keys / real)
WAJEPA_CSV = "runs/top10_t2/navsim/wajepa/20260926-122804/v2/2026.09.26.16.32.53.csv"
V2 = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance", "traffic_light_compliance", "ego_progress",
      "time_to_collision_within_bound", "lane_keeping", "history_comfort", "two_frame_extended_comfort"]


def stem(m):
    return f"{FRAMES}@cinque_PP{m.replace(':', '_')}"


def cmd_plans(a):
    from jevdrive.bench.navsim import parity_plans
    from jevdrive.run import Run
    from jevdrive.data import splits
    import op_lb as OL
    with Run("op_parity", "navtest-plans", config=vars(a)) as run:
        run.use_split(splits.load("navsim/navtest"))
        pdir = OL.root(DATA, "plans")
        eq, plans = {}, {}
        for m in ["P0"] + [x for x in a.models if x != "P0"]:
            out = pdir / f"{stem(m)}.npz"
            parity_plans(model_spec(m), "navtest", str(out), batch=a.batch, data=DATA)
            plans[m] = np.load(out)["plan_mu"]
            d = np.abs(plans[m] - plans["P0"])
            eq[m] = {"max_abs_vs_P0": float(d.max()), "p99_abs_vs_P0": float(np.percentile(d.max((1, 2)), 99)),
                     "mean_pos_l2_vs_P0": float(np.linalg.norm(plans[m][:, :, :2] - plans["P0"][:, :, :2], axis=-1).mean())}
            run.info(f"{m}: {eq[m]}")
        onnx = pdir / "gimm@cinque.npz"
        if onnx.exists() and FRAMES == "gimm" and DATA == "lb_navtest":
            z = np.load(onnx)
            assert z["names"].tolist() == OL.meta(DATA)["names"]
            if "plan_mu" in z:
                d = np.abs(plans["P0"] - z["plan_mu"])
                eq["P0_vs_onnx"] = {"max_abs": float(d.max()), "p99_row_max": float(np.percentile(d.max((1, 2)), 99)),
                                    "median_row_max": float(np.median(d.max((1, 2)))),
                                    "pos4s_l2_mean": float(np.linalg.norm(plans["P0"][:, 22, :2] - z["plan_mu"][:, 22, :2], axis=-1).mean())}
                run.info(f"P0 vs ONNX: {eq['P0_vs_onnx']}")
        out = data_dir() / "runs/op_parity/navtest"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"equivalence_{a.tag}_{DATA}_{FRAMES}.json").write_text(json.dumps(eq, indent=1))
        run.summary["equivalence"] = eq


def cmd_score(a):
    if DATA == "lb_navtest" and a.ver == "v2":
        from jevdrive import bench
        from jevdrive.bench import navsim, runner
        from jevdrive.bench.models import resolve
        specs = [model_spec(m) for m in a.models]
        if os.environ.get("CL_POOL_JOB"):
            # Historical CPU worker: share the scoring stages, without queueing work behind its own lease.
            for spec in specs:
                m = resolve(spec, check=True)
                d = bench.run_dir(spec, "navtest")
                stages = navsim.stages(m, "navtest", d, shards=1)
                if not navsim.plan_file(m, "navtest").exists():
                    raise RuntimeError(f"plans missing for {spec}; use jevdrive.bench run from the orchestrator")
                for stage in stages:
                    if stage.name in ("prep", "plans") or _pl.Path(stage.done).exists():
                        continue
                    subprocess.run(stage.cmd, check=True, cwd=_R, env=dict(os.environ, **stage.env))
        else:
            if not bench.wait([bench.run(spec, "navtest") for spec in specs]):
                raise SystemExit(1)
        return
    env = dict(os.environ, OPI_ROOT="op_lb")
    jev = str(data_dir() / "envs" / "jevdrive" / "bin" / "python")
    stems = [stem(m) for m in a.models]
    subprocess.check_call([jev, str(_R / "experiments/op_openloop/lib/op_interp.py"), "nav-export", "--data", DATA, "--adapters", "base",
                           "--plans", *stems], env=env)
    if DATA != "lb_navtest":                                   # subset run dir: score only its tokens (full v2_navtest metric cache)
        env["TOKENS_FILE"] = str(data_dir() / "runs" / "op_lb" / DATA / "tokens.txt")
    sc = str(_R / "experiments/zeroshot_openloop/archive/navsim_zs_score.sh")       # per stem (op_interp_score.sh would score every unscored file)
    for st in stems:
        name = f"opi_{DATA}_{st.replace('@', '-')}__base"
        if eval_csv(st[len(f"{FRAMES}@cinque_PP"):], a.ver) is not None:
            continue
        f = data_dir() / "runs" / "op_lb" / DATA / "preds" / f"{st.replace('@', '-')}__base.npz"
        with open(data_dir() / "runs" / "op_lb" / DATA / f"score_{name}.log", "w") as log:
            subprocess.check_call([sc, "score", a.ver, "navtest", name, str(f)], env=env, stdout=log, stderr=subprocess.STDOUT)


def read_csv(path):
    import pandas as pd
    df = pd.read_csv(path)
    avg = df[df.token.astype(str).str.startswith("average")]
    tok = df[~df.token.astype(str).str.startswith("average") & ~df.token.astype(str).str.startswith("extended_pdm_score")]
    return tok.set_index("token"), (avg.iloc[0] if len(avg) else None)


def eval_csv(m, ver="v2"):
    fs = sorted((data_dir() / "runs" / "navsim" / "eval").glob(f"{ver}_navtest_opi_{DATA}_{stem(m).replace('@', '-')}__base/*/*.csv"))
    legacy = fs[-1] if fs else None
    if DATA == "lb_navtest" and ver == "v2":
        from jevdrive.bench.compat import navtest_csv
        return navtest_csv(model_spec(m), legacy)
    return legacy


def model_spec(m):
    tag, sep, opt = m.partition(":")
    return f"{tag}@{FRAMES}" + (f":{opt}" if sep else "")


def pred_file(m):
    legacy = data_dir() / "runs" / "op_lb" / DATA / "preds" / f"{stem(m).replace('@', '-')}__base.npz"
    if DATA == "lb_navtest":
        from jevdrive.bench.compat import pred_file as bench_pred
        return bench_pred(model_spec(m), legacy=legacy)
    return legacy


def cmd_report(a):
    import pandas as pd
    from jevdrive import stats
    import op_lb as OL
    mt = OL.meta(DATA)
    tab = np.load(data_dir() / "runs" / "op_parity" / "cache" / DATA / "tab.npz")
    log = dict(zip(tab["names"].tolist(), tab["log"].tolist()))
    arms = {}
    for m in a.models:
        f = eval_csv(m, a.ver)
        if f is None:
            print(f"no score for {m}")
            continue
        arms[m] = read_csv(f)
    arms["WA-JEPA"] = read_csv(data_dir() / WAJEPA_CSV)
    toks = sorted(set.intersection(*[set(t.index) for t, _ in arms.values()]))
    from jevdrive import navsim_zs as Z
    fz = np.load(Z.root("index") / "navtest_future.npz")
    fut = dict(zip(fz["tokens"].tolist(), fz["poses"]))
    plen = lambda P: np.linalg.norm(np.diff(np.concatenate([np.zeros_like(P[:, :1, :2]), P[:, :, :2]], 1), axis=-1), axis=-1).sum(1)  # noqa: E731
    rows = []
    for m, (t, avg) in arms.items():
        geo = {}
        pf = pred_file(m)
        if pf.exists():
            z = np.load(pf)
            F = np.stack([fut[k] for k in z["tokens"]])
            Lg, Pp = plen(F), z["poses"]
            mv = Lg > 2.0
            geo = {"ade_vs_log": float(np.linalg.norm(Pp[:, :, :2] - F[:, :, :2], axis=-1).mean()),
                   "speed_ratio_med": float(np.median(plen(Pp)[mv] / Lg[mv]))}
        r = {"arm": m, "n": len(t), "EPDMS (devkit average row)": 100 * float(avg.score) if avg is not None else np.nan,
             "EPDMS (token mean)": 100 * float(t.score.mean())} | geo
        r |= {s: 100 * float(t[s].mean()) for s in V2 if s in t}
        rows.append(r)
    out = _R / "experiments" / "op_parity" / "results"
    out.mkdir(exist_ok=True)
    stats.write_table(rows, out / f"navtest_{a.tag}_arms", floatfmt=".2f", note=f"navtest, devkit navsim main @0a380a9 v2 EPDMS; {len(toks)} tokens in common")
    g = np.array([log[t] for t in toks])
    pr = []
    for x, y in [(m, a.ref) for m in arms if m not in (a.ref, "WA-JEPA")] + [(m, "WA-JEPA") for m in arms if m != "WA-JEPA"]:
        if x not in arms or y not in arms:
            continue
        sx, sy = (100 * arms[k][0].loc[toks, "score"].to_numpy(float) for k in (x, y))
        r = stats.paired(sx, sy, groups=g)
        pr.append({"pair": f"{x} - {y}", **{k: r[k] for k in ("n", "units", "mean", "lo", "hi", "mean_a", "mean_b")}})
    stats.write_table(pr, out / f"navtest_{a.tag}_paired", floatfmt=".2f", note="per-token EPDMS x 100, cluster bootstrap over navtest logs (B 10000)")
    print(pd.DataFrame(rows).to_string()), print(pd.DataFrame(pr).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    ap.add_argument("--data", default="lb_navtest")
    ap.add_argument("--frames", default="gimm", choices=["gimm", "warp", "keys", "real", "vh140"])
    p = sp.add_parser("plans")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--tag", default="pilot")
    p = sp.add_parser("score")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--ver", default="v2")
    p = sp.add_parser("report")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--ref", default="P1-s0")
    p.add_argument("--ver", default="v2")
    p.add_argument("--tag", default="pilot")
    a = ap.parse_args()
    DATA, FRAMES = a.data, a.frames
    {"plans": cmd_plans, "score": cmd_score, "report": cmd_report}[a.cmd](a)
