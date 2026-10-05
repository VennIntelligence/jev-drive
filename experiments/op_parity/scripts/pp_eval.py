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

DATA = "lb_navtest"
WAJEPA_CSV = "runs/top10_t2/navsim/wajepa/20260926-122804/v2/2026.09.26.16.32.53.csv"
V2 = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance", "traffic_light_compliance", "ego_progress",
      "time_to_collision_within_bound", "lane_keeping", "history_comfort", "two_frame_extended_comfort"]


def stem(m):
    return f"gimm@cinque_PP{m.replace(':', '_')}"


def cmd_plans(a):
    import torch
    import pp_train as T
    from jevdrive import op_adapt as A
    from jevdrive.run import Run
    from jevdrive.data import splits
    import op_lb as OL
    dev = torch.device("cuda")
    with Run("op_parity", "navtest-plans", config=vars(a)) as run:
        run.use_split(splits.load("navsim/navtest"))
        S = T.Store([DATA], dev, need_side=True)
        names = S.tab["names"]
        assert names.tolist() == OL.meta(DATA)["names"]
        pdir = OL.root(DATA, "plans")
        eq, plans = {}, {}
        for m in ["P0"] + [x for x in a.models if x != "P0"]:
            tag, _, opt = m.partition(":")
            model = T.load_pmodel(tag, dev)
            sl = model.net.slices
            pi = np.arange(sl["plan"].start, sl["plan"].start + 495)
            ps = np.arange(sl["plan"].start + 495, sl["plan"].start + 990)
            mu, sd = np.zeros((S.n, 33, 15), np.float32), np.zeros((S.n, 33, 15), np.float32)
            with torch.no_grad():
                for i in range(0, S.n, a.batch):
                    r = torch.arange(i, min(i + a.batch, S.n), device=dev)
                    mask = torch.zeros(len(r), 3, dtype=torch.bool, device=dev) if opt == "noside" else None
                    o = model(S.front[r], S.ego[r], S.tc[r], S.side[r], mask).float().cpu().numpy()
                    mu[i:i + len(r)] = o[:, pi].reshape(-1, 33, 15)
                    sd[i:i + len(r)] = np.exp(np.minimum(o[:, ps], 11)).reshape(-1, 33, 15)
            plans[m] = mu
            np.savez(pdir / f"{stem(m)}.npz", names=names, plan_pos=mu[:, :, 0:3], plan_vel=mu[:, :, 3:6], plan_yaw=mu[:, :, 11], plan_mu=mu,
                     plan_std=sd, steps=31, info=json.dumps({"model": f"op_parity {m}", "source": "experiments/op_parity/scripts/pp_eval.py"}))
            d = np.abs(mu - plans["P0"])
            eq[m] = {"max_abs_vs_P0": float(d.max()), "p99_abs_vs_P0": float(np.percentile(d.max((1, 2)), 99)),
                     "mean_pos_l2_vs_P0": float(np.linalg.norm(mu[:, :, :2] - plans["P0"][:, :, :2], axis=-1).mean())}
            run.info(f"{m}: {eq[m]}")
            del model
            torch.cuda.empty_cache()
        onnx = pdir / "gimm@cinque.npz"
        if onnx.exists():
            z = np.load(onnx)
            assert z["names"].tolist() == names.tolist()
            ref = z["plan_mu"] if "plan_mu" in z else None
            if ref is not None:
                d = np.abs(plans["P0"] - ref)
                eq["P0_vs_onnx"] = {"max_abs": float(d.max()), "p99_row_max": float(np.percentile(d.max((1, 2)), 99)),
                                    "median_row_max": float(np.median(d.max((1, 2)))),
                                    "pos4s_l2_mean": float(np.linalg.norm(plans["P0"][:, 22, :2] - ref[:, 22, :2], axis=-1).mean())}
                run.info(f"P0 vs ONNX: {eq['P0_vs_onnx']}")
        out = data_dir() / "runs" / "op_parity" / "navtest"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"equivalence_{a.tag}.json").write_text(json.dumps(eq, indent=1))
        run.summary["equivalence"] = eq


def cmd_score(a):
    env = dict(os.environ, OPI_ROOT="op_lb")
    jev = str(data_dir() / "envs" / "jevdrive" / "bin" / "python")
    stems = [stem(m) for m in a.models]
    subprocess.check_call([jev, str(_R / "experiments/op_openloop/lib/op_interp.py"), "nav-export", "--data", DATA, "--adapters", "base",
                           "--plans", *stems], env=env)
    sc = str(_R / "experiments/zeroshot_openloop/archive/navsim_zs_score.sh")       # per stem (op_interp_score.sh would score every unscored file)
    for st in stems:
        name = f"opi_{DATA}_{st.replace('@', '-')}__base"
        if eval_csv(st[len("gimm@cinque_PP"):], a.ver) is not None:
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
    return fs[-1] if fs else None


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
    rows = []
    for m, (t, avg) in arms.items():
        r = {"arm": m, "n": len(t), "EPDMS (devkit average row)": 100 * float(avg.score) if avg is not None else np.nan,
             "EPDMS (token mean)": 100 * float(t.score.mean())}
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
    p = sp.add_parser("plans")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--batch", type=int, default=512)
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
    {"plans": cmd_plans, "score": cmd_score, "report": cmd_report}[a.cmd](a)
