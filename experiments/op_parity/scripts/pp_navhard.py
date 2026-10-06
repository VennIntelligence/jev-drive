"""op_parity navhard readout (results/navhard.md): NAVSIM navhard_two_stage v2 EPDMS (the devkit of the navtest readout) for P0 and the
full-run checkpoints P1 / P2 / P3 x 2 seeds (protocol W), and WA-JEPA's released checkpoint in the same harness.

Our arms: pp_prep.py --data lb_navhard (tab, side cameras) + --frames warp (W front tokens), pp_eval.py --data lb_navhard --frames warp plans,
op_interp nav-export (adapter `base`), then experiments/op_guard/scripts/nav_harness.py (envs/navsim2): per-token devkit pdm_score + the
devkit's two-stage aggregation; the mean over the 225 scene-mapping groups is the official number, so paired CIs resample groups
(clustered by the log of the group's stage-1 token).
WA-JEPA: its own runner (experiments/top10/lib/top10_t2/wajepa_run.py, the repo's NAVSIM feature builder, fp32 = its NAVSIM path) on
requests built here from the navhard index (4 cameras L0 / F0 / R0 / B0 x 4 frames at 2 Hz, 4 history poses, ego velocity / acceleration,
command), validated on navtest tokens against WA-JEPA's stored navtest export (the run that reproduced 91.71).

  req     (op-train)  pp_navhard.py req --split navhard_two_stage --out R.npz       [--split navtest --n 64: the validation request]
  wcheck  (op-train)  pp_navhard.py wcheck --name N                                  scored navtest check vs the stored export (EPDMS)
  preds   (any)       pp_navhard.py preds --wajepa P.npz --out preds.npz             WA-JEPA output -> the harness pose file (tokens, poses)
  report  (op-train)  pp_navhard.py report                                           -> results/navhard.md tables (navhard_arms / _paired)
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "scripts"), str(_pl.Path(__file__).parent)]
import argparse, json, os  # noqa: E401,E402

import numpy as np  # noqa: E402

DATA = _pl.Path(os.environ.get("DATA_DIR", _pl.Path.home() / "data"))
CAMS = ("CAM_L0", "CAM_F0", "CAM_R0", "CAM_B0")                 # wajepa_run's per-time camera order
WORK = DATA / "runs/op_parity/navhard"
ARMS = {"P0": ["P0"], "P1": ["P1-F-s0", "P1-F-s1"], "P2": ["P2-F-s0", "P2-F-s1"], "P3": ["P3-F-s0", "P3-F-s1"], "WA-JEPA": ["wajepa"]}
REFS = {"shipped GIMM (op_guard `shipped`, decision 37 / 143 S0)": DATA / "runs/op_guard/shipped/nav/navhard",
        "factor_wm S3 (decision 143)": DATA / "runs/op_guard/fw-S3/nav/navhard"}


def cmd_req(a):
    from jevdrive import navsim_zs as Z
    idx = Z.load_index(a.split)
    if a.n:
        idx = idx[:: max(1, len(idx) // a.n)][: a.n]
    img = [[e["cams"][f][c]["path"] for f in range(4) for c in CAMS] for e in idx]
    hist = np.stack([np.asarray(e["pose"], np.float32) for e in idx])          # (n, 4, 3) in the t0 rear-axle frame, oldest first
    ego = np.stack([np.r_[np.asarray(e["vel"], np.float32)[-1], np.asarray(e["acc"], np.float32)[-1]] for e in idx])
    cmd = np.array([int(np.argmax(np.asarray(e["cmd"])[-1])) for e in idx])
    assert np.allclose(hist[:, -1], 0), "history poses are not relative to t0"
    np.savez(a.out, keys=np.array([e["token"] for e in idx]), img=np.array(img), hist=hist, ego=ego, cmd=cmd)
    _pl.Path(a.out).with_suffix(".tokens").write_text("\n".join(e["token"] for e in idx) + "\n")   # TOKENS_FILE for subset scoring
    print(f"{len(idx)} requests ({a.split}) -> {a.out}")


def cmd_wcheck(a):
    """The runner on our navtest request vs WA-JEPA's stored navtest export (the run that reproduced 91.71), at the score level: v2 EPDMS
    of the same tokens, paired over logs. Pass: |mean difference| < 0.5. (Trajectories differ by float noise of the flow sampler across
    hosts: 64 tokens, ADE 0.07 m, max 0.52 m; inputs verified identical to the agent's SceneLoader path.)"""
    from jevdrive import stats
    from jevdrive import navsim_zs as Z
    import pp_eval as E
    fs = sorted((DATA / "runs/navsim/eval" / f"v2_navtest_{a.name}").glob("*/*.csv"))
    ours, _ = E.read_csv(fs[-1])
    ref, _ = E.read_csv(DATA / E.WAJEPA_CSV)
    toks = sorted(set(ours.index) & set(ref.index))
    lg = {e["token"]: e["log_name"] for e in Z.load_index("navtest", slim=True)}
    r = stats.paired(100 * ours.loc[toks, "score"].to_numpy(float), 100 * ref.loc[toks, "score"].to_numpy(float), groups=np.array([lg[t] for t in toks]))
    res = {"n": len(toks), **{k: r[k] for k in ("mean", "lo", "hi", "mean_a", "mean_b", "units")}, "pass": bool(abs(r["mean"]) < 0.5)}
    print(json.dumps(res))
    (WORK / "wajepa" / "wcheck.json").write_text(json.dumps(res, indent=1))
    if not res["pass"]:
        raise SystemExit("WA-JEPA request path does not reproduce the stored navtest scores")


def cmd_preds(a):
    z = np.load(a.wajepa)
    np.savez(a.out, tokens=z["keys"], poses=z["traj"][:, :, :3].astype(np.float32))
    print(f"{len(z['keys'])} poses -> {a.out}")


def harness_dir(m: str) -> _pl.Path:
    return WORK / "harness" / m


def _boot(d, g=None):
    from jevdrive import stats
    r = stats.bootstrap(d) if g is None else stats.paired(d, np.zeros_like(d), groups=g)
    return r


def cmd_report(a):
    import pandas as pd
    from jevdrive import stats
    from jevdrive import navsim_zs as Z
    lg = {e["token"]: e["log_name"] for e in Z.load_index("navhard_two_stage", slim=True)}
    runs = {}
    for arm, ms in ARMS.items():
        got = [m for m in ms if (harness_dir(m) / "harness_groups.csv").exists()]
        if got:
            runs[arm] = [(m, pd.read_csv(harness_dir(m) / "harness_groups.csv").set_index("group"),
                          json.loads((harness_dir(m) / "harness_summary.json").read_text())) for m in got]
    for name, d in REFS.items():
        if (d / "harness_groups.csv").exists():
            runs[name] = [(name, pd.read_csv(d / "harness_groups.csv").set_index("group"), json.loads((d / "harness_summary.json").read_text()))]
    g0 = next(iter(runs.values()))[0][1]
    logs = np.array([lg.get(o, "?") for o in g0.orig])
    rows = []
    for arm, v in runs.items():
        assert all((t.orig.values == g0.orig.values).all() for _, t, _ in v), "group order differs"
        r = {"arm": arm, "seeds": len(v)}
        for c in ("combined", "stage1", "stage2"):
            r[c] = float(np.mean([s[c] for _, _, s in v]))
        r["combined s0 / s1"] = " / ".join(f"{s['combined']:.2f}" for _, _, s in v)
        rows.append(r)
    out = _R / "experiments/op_parity/results"
    stats.write_table(rows, out / "navhard_arms", floatfmt=".2f",
                      note="navhard_two_stage, 5 912 tokens / 225 scene-mapping groups, devkit navsim main @0a380a9 v2 EPDMS (devkit aggregation); seed means")
    grp = lambda arm, c="combined": np.mean([t[c].to_numpy(float) for _, t, _ in runs[arm]], 0)  # noqa: E731
    pr = []
    pairs = [("P1", "P0"), ("P2", "P1"), ("P3", "P1"), ("P3", "P2"), ("P2", "P0")] + [(x, "WA-JEPA") for x in ("P0", "P1", "P2", "P3")]
    pairs += [("P0", k) for k in REFS] + [("P2", k) for k in REFS]
    for x, y in pairs:
        if x in runs and y in runs:
            for c in ("combined", "stage1", "stage2"):
                r = stats.paired(grp(x, c), grp(y, c), groups=logs)
                pr.append({"pair": f"{x} - {y}", "score": c, **{k: r[k] for k in ("mean", "lo", "hi", "mean_a", "mean_b", "n", "units")}})
    stats.write_table(pr, out / "navhard_paired", floatfmt=".2f",
                      note="per-group two-stage EPDMS (devkit calculate_individual_mapping_scores), seed means, bootstrap over groups clustered by the log of the stage-1 token, B 10000")
    print(pd.DataFrame(rows).to_string()), print(pd.DataFrame(pr).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("req")
    p.add_argument("--split", default="navhard_two_stage")
    p.add_argument("--n", type=int, default=0)
    p.add_argument("--out", required=True)
    p = sp.add_parser("wcheck")
    p.add_argument("--name", required=True, help="the navsim_zs_score.sh run name of the navtest check")
    p = sp.add_parser("preds")
    p.add_argument("--wajepa", required=True)
    p.add_argument("--out", required=True)
    sp.add_parser("report")
    a = ap.parse_args()
    {"req": cmd_req, "wcheck": cmd_wcheck, "preds": cmd_preds, "report": cmd_report}[a.cmd](a)
