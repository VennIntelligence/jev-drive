"""op-adapt L readouts (op-train venv; prereg section 'Auslesung'). Every adapted model is read against the original on the same
rows, with cluster bootstraps over whole segments / scenes.

  eval     feature pass of one model ('O' or a run tag) on the held-out sets -> $L/readout/<model>/eval/<set>.npz
             wodval  WOD-E2E val, every even frame with a full 9-slot context and a human future (segment clusters)
             nusval  nuScenes val, 5 Hz lattice with a full context (scene clusters; not trained on, not conditioned on intent)
             rater   the 479 WOD-E2E val rater frames (N-rfs; r2's rater cache, read only)
  read     capture, false-trigger, drift, slow / fast, ADE per slice and set + RFS (raw and x1.06, per category), paired against O
           -> $L/readout/<model>/summary.json, metrics.csv
  navtest  NAVSIM navtest PDMS of the native plan (r2's decision-36 harness, port O in the same run)
  report   every run's summary -> research/results/op-adapt-L/*.csv

  CUDA_VISIBLE_DEVICES=0 taskset -c 8-19 python scripts/op_adapt_l_readout.py eval --model O
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from jevdrive import op_adapt as A  # noqa: E402
from jevdrive import op_adapt_l as L  # noqa: E402
from jevdrive import op_adapt_r2 as R  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

B = 2000


def rdir(model, *p):
    return L.lroot("readout", model, *p)


def model_path(model):
    return None if model == "O" else L.lroot("runs", model) / "ckpt-final.pt"


# ---------------------------------------------------------------- rater intent
def rater_names():
    from jevdrive import wod_zeroshot as Z
    return Z.load_sets()["rater"]["name"].astype(str)


def eval_rows(D: L.Data) -> dict:
    return {"wodval": D.rows("wodval", "val", need_future=True),
            "nusval": D.rows("nus", "val", need_future=True)}


def cmd_eval(a):
    import op_adapt_l_prep as P
    dev = torch.device("cuda")
    D = L.Data(("wod", "wodval", "nus"))
    m = L.load_model(model_path(a.model), dev)
    out = rdir(a.model, "eval")
    rows = eval_rows(D)
    for name, dn in (("wodval", "wodval"), ("nusval", "nus")):
        t0 = time.time()
        o = L.fwd_rows(m, D, dn, rows[name], dev, threads=a.threads, intent=not a.no_intent)
        np.savez(out / f"{name}.npz", rows=rows[name], **o)
        print(f"{name}: {len(rows[name])} rows in {time.time() - t0:.0f} s", flush=True)
    # rater frames (r2's cache, read only); intent from the WOD index
    rd = R.Domain("rater", L.r2t())
    D.dom["rater"] = rd
    names = rd.col("key").astype(str)
    D.tab["rater"] = {"intent": P.wod_kin(names)["intent"]}
    t0 = time.time()
    o = L.fwd_rows(m, D, "rater", np.arange(len(rd)), dev, intent=not a.no_intent)
    np.savez(out / "rater.npz", names=names, **o)
    print(f"rater: {len(rd)} rows in {time.time() - t0:.0f} s", flush=True)


def load_eval(model, name):
    p = rdir(model, "eval") / f"{name}.npz"
    return dict(np.load(p, allow_pickle=True)) if p.exists() else None


# ---------------------------------------------------------------- metrics
SLICE_CAP = {"start": "cap_start", "stop": "cap_stop", "turn_onset": "cap_turn_onset"}
FALSE = {"stay": ["false_start"], "control": ["false_stop", "false_turn"], "straight_int": ["false_turn"]}


def set_metrics(tag, tab, rows, groups, cam, pa, po, xs=1.0, extra_flags=()):
    """Rows of one held-out set: per slice / contrast the paired capture / false-trigger deltas and drift-type rates."""
    def rm(pl):
        pl = pl.copy()
        if xs != 1.0:
            pl[:, :, 0] *= xs
        return L.row_metrics(pl, tab, rows, cam)
    ma, mo = rm(pa), rm(po)
    res = []
    fl = {s: np.asarray(tab[f"s_{s}"])[rows] for s in ("start", "stay", "stop", "turn_onset", "control", "straight_int", "in_turn",
                                                     "nudge", "lane_change")}
    other = ~(fl["start"] | fl["stop"] | fl["turn_onset"])
    fl["other"] = other
    fl["all"] = np.ones(len(rows), bool)
    gi_cache = {}
    for s, keys in [(s, [SLICE_CAP[s]]) for s in SLICE_CAP] + list(FALSE.items()) + [
            (s, []) for s in ("in_turn", "nudge", "lane_change", "other", "all")]:
        m = fl[s]
        if not m.any():
            continue
        for k in keys + ["lon3", "lat3", "ade4", "slow", "fast"]:
            d = L.paired_delta(ma[k][m], mo[k][m], groups[m], B=B)
            res.append({"set": tag, "slice": s, "metric": k, "xscale": xs, **d})
    # drift on frames outside the imitated slices
    dr = A.plan_drift(pa, po)
    m = other
    gi, Lc, W = L.cluster_boot(groups[m], B)
    for nm, x in (("drift_median", None), ("drift_p95", None)):
        pass
    med, p95 = float(np.median(dr[m])), float(np.percentile(dr[m], 95))
    res.append({"set": tag, "slice": "other", "metric": "drift_median", "xscale": xs, "n": int(m.sum()), "adapt": med})
    res.append({"set": tag, "slice": "other", "metric": "drift_p95", "xscale": xs, "n": int(m.sum()), "adapt": p95})
    mean, ci = L.boot_mean(dr[m], gi, Lc, W)
    res.append({"set": tag, "slice": "other", "metric": "drift_mean", "xscale": xs, "n": int(m.sum()), "adapt": mean, "lo": ci[0], "hi": ci[1]})
    for s in ("control", "straight_int"):
        if fl[s].any():
            res.append({"set": tag, "slice": s, "metric": "drift_median", "xscale": xs, "n": int(fl[s].sum()), "adapt": float(np.median(dr[fl[s]]))})
    return res


def rfs_metrics(model):
    from jevdrive import waymo as W
    from jevdrive import wod_zeroshot as Z
    za, zo = load_eval(model, "rater"), load_eval("O", "rater")
    if za is None or zo is None:
        return []
    sets = Z.load_sets()["rater"]
    names = za["names"].astype(str)
    pos = pd.Series(np.arange(len(sets["name"])), index=sets["name"].astype(str))
    k = pos.reindex(names).to_numpy().astype(int)
    calib = json.loads((Z.root() / "op_calib.json").read_text())
    dev_xy = np.stack([np.array(calib[n.rsplit("-", 1)[0]]["1"]["extrinsic"]).reshape(4, 4)[:2, 3] for n in names])
    sp = W.init_speed(sets["past"][k])
    cl = sets["cluster"][k].astype(str)
    rows = []
    for xs in (1.0, 1.06):
        sc = {}
        for who, z in (("adapt", za), ("orig", zo)):
            p = z["plan"]
            wp = np.stack([Z.openpilot_to_wod(p[i, :, 0:3], p[i, :, 11], A.T_IDXS, dev_xy[i]) for i in range(len(p))])[:, :, :2].copy()
            wp[..., 0] *= xs
            sc[who] = np.asarray(W.rater_feedback_score(wp, sets["traj"][k], sets["scores"][k], sp), float)
        agg = {w: W.rfs_by_cluster(s, cl)[0] for w, s in sc.items()}
        rng = np.random.default_rng(0)
        codes, uniq = pd.factorize(pd.Series(names).map(lambda n: n.rsplit("-", 1)[0]))
        idx = [np.flatnonzero(codes == c) for c in range(len(uniq))]

        def stat(i, cat=None):
            return W.rfs_by_cluster(sc["adapt"][i], cl[i])[0] - W.rfs_by_cluster(sc["orig"][i], cl[i])[0]
        dd = [stat(np.concatenate([idx[c] for c in rng.integers(len(uniq), size=len(uniq))])) for _ in range(1000)]
        rows.append({"set": "rater", "slice": "all", "metric": "rfs", "xscale": xs, "n": int(len(names)), "adapt": agg["adapt"],
                     "orig": agg["orig"], "delta": agg["adapt"] - agg["orig"], "lo": float(np.percentile(dd, 2.5)),
                     "hi": float(np.percentile(dd, 97.5))})
        _, per_a = W.rfs_by_cluster(sc["adapt"], cl)
        _, per_o = W.rfs_by_cluster(sc["orig"], cl)
        cnt = pd.Series(cl).value_counts()
        for c in per_a.index:
            m = cl == c
            dfr = sc["adapt"][m] - sc["orig"][m]
            bs = [dfr[rng.integers(m.sum(), size=m.sum())].mean() for _ in range(500)]
            rows.append({"set": "rater", "slice": f"cat:{c}", "metric": "rfs", "xscale": xs, "n": int(cnt[c]), "adapt": float(per_a[c]),
                         "orig": float(per_o[c]), "delta": float(per_a[c] - per_o[c]), "lo": float(np.percentile(bs, 2.5)),
                         "hi": float(np.percentile(bs, 97.5))})
    return rows


def cmd_read(a):
    D = L.Data(("wod", "wodval", "nus"))
    rows = eval_rows(D)
    out = []
    for name, dn in (("wodval", "wodval"), ("nusval", "nus")):
        za, zo = load_eval(a.model, name), load_eval("O", name)
        if za is None or zo is None:
            continue
        assert np.array_equal(za["rows"], zo["rows"])
        r = za["rows"]
        tab = D.tab[dn]
        groups = np.asarray(tab["seq"])[r]
        for xs in (1.0, 1.06):
            out += set_metrics(name, tab, r, groups, D.cam[dn], za["plan"], zo["plan"], xs)
    out += rfs_metrics(a.model)
    df = pd.DataFrame(out)
    df.insert(0, "model", a.model)
    df.to_csv(rdir(a.model) / "metrics.csv", index=False)
    print(df[(df.xscale == 1.0)].query("metric in ['cap_start','cap_stop','cap_turn_onset','false_start','false_stop','false_turn','rfs']")
          [["set", "slice", "metric", "n", "adapt", "orig", "delta", "lo", "hi"]].round(3).to_string(index=False))


# ---------------------------------------------------------------- navtest
def cmd_navtest(a):
    import op_adapt_r2_readout as RO
    import op_lb as OL
    dev = torch.device("cuda")
    models = {"port": L.load_model(None, dev)} | {m: L.load_model(model_path(m), dev) for m in a.models}
    split = "lb_navtest"
    names, out = RO.nav_plans(models, split, dev, a.batch)
    pdir = OL.root(split, "plans")
    stems = []
    for k in models:
        mu = out[k]
        stem = f"gimm@cinque_L{k}"
        np.savez(pdir / f"{stem}.npz", names=np.array(names), plan_pos=mu[:, :, 0:3], plan_vel=mu[:, :, 3:6], plan_yaw=mu[:, :, 11],
                 plan_mu=mu, plan_std=out[f"{k}_std"], steps=31, info=json.dumps({"model": f"port {k}", "source": "op_adapt_l_readout"}))
        stems.append(stem)
    env = dict(os.environ, OPI_ROOT="op_lb")
    jev = str(data_dir() / "envs" / "jevdrive" / "bin" / "python")
    repo = Path(__file__).resolve().parents[1]
    subprocess.check_call([jev, str(repo / "scripts/op_interp.py"), "nav-export", "--data", split, "--adapters", "base", "--plans", *stems], env=env)
    subprocess.check_call([str(repo / "scripts/op_interp_score.sh"), a.cpus, split, "v1", "navtest"], env=env)
    res = RO.read_nav_scores(split, "v1", "navtest", [s.replace("@", "-") + "__base" for s in stems])
    o = res.get("gimm-cinque_Lport__base")
    for m in a.models:
        x = res.get(f"gimm-cinque_L{m}__base")
        rdir(m, "navtest.json").write_text(json.dumps({"port_O": o, "model": x, "delta": (x["score"] - o["score"]) if x and o else None,
                                                       "line": ">= O - 1"}, indent=1, default=str))
    print(json.dumps({k: v["score"] for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("eval")
    p.add_argument("--model", required=True)
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--no-intent", action="store_true", help="feed intent 0 (unknown) even to an intent-conditioned model")
    p = sp.add_parser("read")
    p.add_argument("--model", required=True)
    p = sp.add_parser("navtest")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--cpus", default="8-30")
    a = ap.parse_args()
    {"eval": cmd_eval, "read": cmd_read, "navtest": cmd_navtest}[a.cmd](a)
