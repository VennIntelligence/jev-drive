"""factor_wm G1 diagnostics (no training; results/g1-diag.md).

  h1      (op-train, one GPU) WOD g0b launch clips (40): S0 vs S3 at t0, history = the real 10 logged frames vs the t0 frame repeated 10 times
          (HUGSIM's static warm-up); readouts action[1] (acceleration), plan speed at 1 s, should_stop (v < 0.3 and a < 0.1)
  h2      (CPU) HUGSIM 64 guard runs (op_guard <cand>/full/hugsim): steps 1..40 (launch window) per scene: model action accel, plan speed
          model_v; S0 (shipped) vs S3 (fw-S3), paired over scenes; all scenes and S3's launch-stall scenes
  navhard (CPU) S3 - S0 navhard two-stage EPDMS = mean over 225 mapping groups; paired log-cluster bootstrap (group -> log of its stage-1 token)

  CUDA_VISIBLE_DEVICES=<card> $DATA_DIR/envs/op-train/bin/python experiments/factor_wm/scripts/fw_diag.py h1
  $DATA_DIR/envs/op-train/bin/python experiments/factor_wm/scripts/fw_diag.py h2 navhard
Output: $DATA_DIR/runs/factor_wm/report/diag-<cmd>.json
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fw_common as C  # noqa: E402

OUT = C.root("report")
GUARD = C.data_dir() / "runs" / "op_guard"


def h1():
    import torch
    import fw_g0 as G
    from jevdrive import op_adapt as A
    S = C.Clips("g0b")
    cs = [c for c in range(S.n) if S.t["cat"][c] == "launch"]
    tc = S.t["tc"][cs].astype(np.float32)
    v0 = S.t["v"][cs, C.T0]
    res = {}
    for arm, model in (("S0", "shipped"), ("S3", str(C.root("runs", "S3") / "ckpt-final.pt"))):
        R = G.Runner(model, "cuda", 2)
        for hist in ("real", "static"):
            f = np.asarray(S.imgs[cs][:, : C.NH])
            if hist == "static":
                f = np.repeat(f[:, C.T0: C.T0 + 1], C.NH, axis=1)
            prev, cur = f[:, :-1].reshape(-1, *f.shape[2:]), f[:, 1:].reshape(-1, *f.shape[2:])
            T = torch.cat([R.trunk(prev[i:i + 64], cur[i:i + 64]) for i in range(0, len(cur), 64)]).reshape(len(cs), 9, *R.trunk(prev[:1], cur[:1]).shape[1:])
            with torch.no_grad():
                o = R.m(T, torch.ones(len(cs), 9, dtype=torch.bool, device=R.dev), torch.from_numpy(tc).to(R.dev).to(R.dtype))["outputs"].float().cpu().numpy()
            acc = o[:, R.H.a0 + 1]
            plan = o[:, R.H.pi].reshape(-1, 33, 15)
            v1 = np.array([np.interp(1.0, A.T_IDXS, p[:, 3]) for p in plan])
            res[f"{arm}/{hist}"] = dict(acc=acc.tolist(), v1=v1.tolist(), stop=((v0 < 0.3) & (acc < 0.1)).tolist())
        R.pool.shutdown(wait=False)
    summ = {k: dict(acc_median=float(np.median(v["acc"])), v1_median=float(np.median(v["v1"])), should_stop_share=float(np.mean(v["stop"])))
            for k, v in res.items()}
    for hist in ("real", "static"):
        d = np.array(res[f"S3/{hist}"]["acc"]) - np.array(res[f"S0/{hist}"]["acc"])
        e = np.array(res[f"S3/{hist}"]["v1"]) - np.array(res[f"S0/{hist}"]["v1"])
        summ[f"S3-S0/{hist}"] = dict(acc=_boot(d), v1=_boot(e))
    json.dump(dict(n=len(cs), summary=summ, per_clip=res), open(OUT / "diag-h1.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


def _boot(d, groups=None, n=4000, seed=0):
    from jevdrive import stats
    r = stats.bootstrap(np.asarray(d, float), groups, n_boot=n, seed=seed)
    return dict(mean=float(np.mean(d)), ci=[float(r["lo"]), float(r["hi"])])


def steps(cand):
    out = {}
    d = json.load(open(GUARD / cand / "full" / "lines" / "hugsim.json"))
    for s, r in d["provenance"]["per_scene"].items():
        acc, v1, v = [], [], []
        for ln in open(Path(r["run_dir"]) / "zs_steps.jsonl"):
            j = json.loads(ln)
            if "step" in j and 1 <= j["step"] <= 40:
                acc.append(j.get("accel", np.nan))
                mv = j.get("model_v") or [np.nan] * 4
                v1.append(mv[2] if len(mv) > 2 else np.nan)
                v.append(j.get("v", np.nan))
        out[s] = dict(acc=float(np.nanmean(acc)), v1=float(np.nanmean(v1)), vmax=float(np.nanmax(v)) if v else np.nan, n=len(acc))
    return out


def h2():
    a, b = steps("shipped"), steps("fw-S3")
    sc = sorted(set(a) & set(b))
    res = {}
    for name, sel in (("all", sc), ("S3_launch_stall", [s for s in sc if b[s]["vmax"] < 1.6]),
                      ("S3_stall_not_S0", [s for s in sc if b[s]["vmax"] < 1.6 <= a[s]["vmax"]])):
        if not sel:
            continue
        r = dict(n=len(sel))
        for k in ("acc", "v1"):
            x, y = np.array([b[s][k] for s in sel]), np.array([a[s][k] for s in sel])
            r[k] = dict(S0=float(np.median(y)), S3=float(np.median(x)), diff=_boot(x - y))
        res[name] = r
    json.dump(dict(window="steps 1..40 (after the 5 s warm-up)", fields="accel = action[1] as served; v1 = model_v[2] (plan speed)", res=res,
                   per_scene=dict(S0=a, S3=b)), open(OUT / "diag-h2.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


def navhard():
    from jevdrive import navsim_zs as Z
    lg = {e["token"]: e["log_name"] for e in Z.load_index("navhard_two_stage", slim=True)}
    rd = lambda c: {r["group"]: r for r in csv.DictReader(open(GUARD / c / "nav" / "navhard" / "harness_groups.csv"))}  # noqa: E731
    a, b = rd("shipped"), rd("fw-S3")
    gs = sorted(set(a) & set(b), key=int)
    d = np.array([float(b[g]["combined"]) - float(a[g]["combined"]) for g in gs])
    logs = np.array([lg.get(a[g]["orig"], "?") for g in gs])
    r = dict(n_groups=len(gs), n_logs=len(set(logs)), unmapped=int(np.sum(logs == "?")), diff_log_cluster=_boot(d, logs), diff_group=_boot(d))
    json.dump(r, open(OUT / "diag-navhard.json", "w"), indent=1)
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    for c in sys.argv[1:]:
        {"h1": h1, "h2": h2, "navhard": navhard}[c]()
