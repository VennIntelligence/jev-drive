"""Anonymous WOD-E2E test submission package of WLG (plans/2026-10-08-wod-submit-rule.md). Builds the file only; nothing is uploaded.

  bias     (op-train env, CPU, box) stop-gated intent_bias of the 1 505 submission frames for both WLG seeds, served exactly as val
           (pp_wod.wod_ego -> adapter -> zero where the fed speed < the checkpoint's stop_gate); checks the index-derived ego rows against the
           stored val bias files; parameter count of the served model -> $DATA_DIR/runs/op_parity/wod/submit/{bias-test-<tag>.npz, params.json}
  serve    not a subcommand: `wod_zeroshot_openpilot.py --set test --onnx pp-<tag>.onnx --tag <tag> --bias bias-test-<tag>.npz` (wod_submit_chain.sh)
  predict  (jevdrive env, CPU, box) seed combination by the written rule, val check, test / val statistics, thumbnails -> .../submit/{traj.npz, check.json, stats.md, bev.png}
  package  (anywhere with the proto, e.g. the Mac) traj.npz -> tar.gz through jevdrive.waymo.write_submission, round trip, field-by-field comparison
           with a reference submission. All identity comes from arguments; the account name from --account, $WOD_ACCOUNT_NAME, the gitignored .env,
           or --account-from-reference (the reference's own account field, read at runtime and never printed).

  python experiments/op_parity/scripts/wod_submit.py package --traj tmp/wod_submit/traj.npz --out tmp/wod_submit/lpa-v1.tar.gz --reference ~/Downloads/<ref>.tar.gz
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_pl.Path(__file__).parent)]
import argparse, json, os, tarfile  # noqa: E401,E402

import numpy as np  # noqa: E402

TAGS = ("WLG-full-s0", "WLG-full-s1")
TOTAL_VAL_RFS = 8.187                     # decision 169: mean of the two seeds' per-frame scores
V_STOP = 0.5
DESCRIPTION = (
    "The publicly released openpilot Cinque v3 driving model (382M parameters), adapted to WOD-E2E. The vision encoder is frozen. The "
    "plan-prediction layers are fine-tuned together with a small adapter that maps the ego state (speed, acceleration, 1.5 s pose history) and "
    "the routing command to an additive bias on the policy input tokens. Training used WOD-E2E training sequences only (imitation of the logged "
    "future at 0.5-4 s, plus distillation to the original model on anchor rows). The adapter is switched off (zero bias) when the current speed "
    "is below 0.5 m/s, in training and in inference. FRONT, FRONT_LEFT and FRONT_RIGHT images are reprojected into the model's camera view; "
    "the model uses 10 s of causal image history. The mean plan is converted to 20 ego-frame waypoints at 4 Hz and averaged over two training "
    "seeds. No validation or test labels were used for training; the recipe and the 0.5 m/s threshold were chosen with the WOD-E2E validation "
    "set; no test labels or leaderboard scores were used.")


def sdir():
    from jevdrive.common import data_dir
    d = data_dir() / "runs/op_parity/wod/submit"
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_inputs():
    """(names in manifest order, past (n, 16, 6), intent (n,)) of the submission frames from the WOD index."""
    from jevdrive import waymo as W
    names = W.submission_frames()
    df = W.load_index()
    key = {n: i for i, n in enumerate(W.frame_names(df))}
    rows = np.array([key[n] for n in names])
    past, _ = W.load_ego()
    return np.array(names), past[rows], df.intent.to_numpy()[rows]


# ---------------------------------------------------------------- bias
def cmd_bias(a):
    import torch
    import pp_hugsim as H
    import pp_train as T
    from pp_wod import wod_ego
    from jevdrive.run import Run
    with Run("op_parity", "wod-submit-bias", config=vars(a)) as run:
        names, past, intent = test_inputs()
        assert np.isfinite(past).all(), "non-finite past states on test"
        ego, _ = wod_ego(past, intent)
        out = sdir()
        stats = {}
        for tag in TAGS:
            gate = float(torch.load(T.proot("runs", tag) / "ckpt-final.pt", map_location="cpu", weights_only=False)["cfg"]["stop_gate"])
            assert gate == V_STOP, (tag, gate)
            m = H.pmodel(tag, torch.device("cpu"))
            with torch.no_grad():
                b = np.concatenate([m.adapter(torch.from_numpy(ego[i:i + 256]), None, None).to(torch.float16).numpy() for i in range(0, len(ego), 256)])
            off = ego[:, 4] * 10.0 < gate
            b[off] = 0
            np.savez(out / f"bias-test-{tag}.npz", names=names, bias=b, ego=ego, gated=off)
            n_ad = sum(p.numel() for p in m.adapter.parameters())
            stats[tag] = dict(gate=gate, gated=int(off.sum()), n=len(off), adapter_params=n_ad, bias_rms=float(np.sqrt(np.mean(b.astype(np.float32) ** 2))))
            run.info(f"{tag}: {stats[tag]}")
            # the serving path is the val one: the same function on the same index rows reproduces the stored val ego rows and bias
            z = np.load(T.data_dir() / "runs/op_parity/wod" / f"bias-{tag}.npz")
            from jevdrive import waymo as W
            df = W.load_index()
            key = {n: i for i, n in enumerate(W.frame_names(df))}
            vn = z["names"].astype(str)
            vp, _ = W.load_ego()
            ve, _ = wod_ego(vp[[key[n] for n in vn]], df.intent.to_numpy()[[key[n] for n in vn]])
            stats[tag]["val_ego_max_abs_diff"] = float(np.abs(ve - z["ego"]).max())
            assert stats[tag]["val_ego_max_abs_diff"] < 1e-5, "index-derived ego differs from the stored val ego"
        # served model size: ONNX initialisers (base + fine-tuned plan layers) + adapter
        import onnx
        from onnx import numpy_helper
        mo = onnx.load(str(T.data_dir() / "runs/op_parity/hugsim/onnx" / f"pp-{TAGS[0]}.onnx"), load_external_data=False)
        n_onnx = int(sum(int(np.prod(i.dims)) for i in mo.graph.initializer))
        total = n_onnx + stats[TAGS[0]]["adapter_params"]
        stats["onnx_params"], stats["total_params"] = n_onnx, total
        (out / "params.json").write_text(json.dumps(stats, indent=1))
        run.info(f"onnx initialiser params {n_onnx:,}, adapter {stats[TAGS[0]]['adapter_params']:,}, total {total:,}")


# ---------------------------------------------------------------- predict
def traj_stats(P, v0):
    """P (n, 20, 2) -> dict of plan statistics; v0 (n,) the metric's initial speed."""
    d5 = np.linalg.norm(P[:, -1], axis=-1)
    step = np.linalg.norm(np.diff(np.concatenate([np.zeros((len(P), 1, 2)), P], 1), axis=1), axis=-1)
    q = lambda x: np.percentile(x, [5, 25, 50, 75, 95])  # noqa: E731
    st = v0 < V_STOP
    return {"n": len(P), "NaN/inf": int((~np.isfinite(P)).sum()), "shape": str(tuple(P.shape[1:])),
            "d5 p5/25/50/75/95 (m)": " / ".join(f"{x:.1f}" for x in q(d5)), "d5 mean (m)": float(d5.mean()),
            "path length 5 s p50 (m)": float(np.median(step.sum(1))), "|y5| p50 / p95 (m)": " / ".join(f"{x:.2f}" for x in np.percentile(np.abs(P[:, -1, 1]), [50, 95])),
            "x5 < 0 share": float((P[:, -1, 0] < 0).mean()), "d5 < 1 m share": float((d5 < 1).mean()),
            "v0 < 0.5 share (gated)": float(st.mean()), "d5 < 1 m share | v0 < 0.5": float((d5[st] < 1).mean()) if st.any() else float("nan"),
            "d5 < 1 m share | v0 >= 0.5": float((d5[~st] < 1).mean()), "v0 p50 (m/s)": float(np.median(v0)), "last-step speed p50 (m/s)": float(np.median(step[:, -1] * 4))}


def cmd_predict(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from jevdrive import waymo as W
    from jevdrive import wod_zeroshot as Z
    from jevdrive.run import Run
    from pp_wod import load_preds
    from wod_launch_report import Ctx
    out = sdir()
    with Run("op_parity", "wod-submit-predict", config=vars(a)) as run:
        # ---- val check of the seed rule
        C = Ctx()
        pv = [C.preds(t, all_frames=True) for t in TAGS]                       # (1 437, 20, 2) per seed, rater frames first
        mean_v = np.mean(pv, 0)
        s = [C.cm(C.rfs(p)) for p in pv]
        s_mean = C.cm(C.rfs(mean_v))
        spread = abs(s[0] - s[1])
        tol = max(0.05, spread)
        pooled = float(np.mean(s))
        ok = abs(s_mean - TOTAL_VAL_RFS) <= tol
        chk = dict(seed_rfs=s, seed_mean_of_scores=pooled, rfs_mean_trajectory=s_mean, ref=TOTAL_VAL_RFS, tol=tol, accepted=bool(ok),
                   rule="mean trajectory accepted if |RFS(mean) - 8.187| <= max(0.05, |RFS(s0) - RFS(s1)|), else seed 0 (plans/2026-10-08-wod-submit-rule.md)")
        run.info(json.dumps(chk))
        # ---- test
        names, past, intent = test_inputs()
        pt = [load_preds(t, names) for t in TAGS]
        for t, p in zip(TAGS, pt):
            assert p.shape == (len(names), 20, 2) and np.isfinite(p).all(), (t, p.shape)
        final_t = np.mean(pt, 0) if ok else pt[0]
        final_v = mean_v if ok else pv[0]
        chk["submitted"] = "mean of the two seeds" if ok else "seed 0"
        (out / "check.json").write_text(json.dumps(chk, indent=1))
        np.savez(out / "traj.npz", names=names, traj=final_t.astype(np.float32), seed0=pt[0].astype(np.float32), seed1=pt[1].astype(np.float32))
        # ---- statistics, test next to val
        v0_t = W.init_speed(past).astype(np.float64)
        sets = Z.load_sets()
        v0_v = np.concatenate([W.init_speed(sets[k]["past"]) for k in ("rater", "extra")]).astype(np.float64)
        rows = {"val rater + extra": (final_v, v0_v), "val rater (479)": (final_v[:C.n], v0_v[:C.n]), "test submission frames": (final_t, v0_t)}
        tab = {k: traj_stats(*v) for k, v in rows.items()}
        tab["test submission frames"]["seed gap |s0 - s1| at 5 s, mean (m)"] = float(np.linalg.norm(pt[0][:, -1] - pt[1][:, -1], axis=-1).mean())
        tab["val rater + extra"]["seed gap |s0 - s1| at 5 s, mean (m)"] = float(np.linalg.norm(pv[0][:, -1] - pv[1][:, -1], axis=-1).mean())
        fr = lambda s: np.array([int(n.rsplit("-", 1)[1]) for n in s])  # noqa: E731
        tab["val rater + extra"]["frame index p5 / p50 / p95"] = " / ".join(str(int(x)) for x in np.percentile(fr(C.names), [5, 50, 95]))
        tab["test submission frames"]["frame index p5 / p50 / p95"] = " / ".join(str(int(x)) for x in np.percentile(fr(names), [5, 50, 95]))
        keys = list(dict.fromkeys(k for t in tab.values() for k in t))
        fmt = lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)  # noqa: E731
        md = ["| statistic | " + " | ".join(tab) + " |", "|:--|" + "--:|" * len(tab)]
        md += [f"| {k} | " + " | ".join(fmt(t.get(k, "")) for t in tab.values()) + " |" for k in keys]
        (out / "stats.md").write_text("\n".join(md) + "\n\n```json\n" + json.dumps(chk, indent=1) + "\n```\n")
        run.info("\n".join(md))
        # ---- 6 BEV thumbnails of the test plans: fixed, label-free selection (seed 0 rng, first of each stratum)
        rng = np.random.default_rng(0)
        order = rng.permutation(len(names))
        strata = [("stopped", v0_t < V_STOP), ("0.5-3 m/s", (v0_t >= V_STOP) & (v0_t < 3)), ("fast straight", (v0_t >= 8) & (intent == 1)),
                  ("left", (intent == 2) & (v0_t >= V_STOP)), ("right", (intent == 3) & (v0_t >= V_STOP)), ("random", np.ones(len(names), bool))]
        fig, ax = plt.subplots(2, 3, figsize=(10, 9))
        for axx, (nm, m) in zip(ax.ravel(), strata):
            cand = [i for i in order if m[i]]
            i = cand[0] if cand else order[0]
            for p, c, lw in ((pt[0], "tab:blue", 0.8), (pt[1], "tab:orange", 0.8), (final_t, "k", 1.8)):
                axx.plot(-p[i, :, 1], p[i, :, 0], "-o", c=c, lw=lw, ms=2)
            axx.plot(-past[i, :, 1], past[i, :, 0], "-", c="tab:green", lw=1.2)
            axx.plot(0, 0, "r^")
            axx.set_aspect("equal")
            axx.set_title(f"{nm}: {names[i][:8]}-{names[i].rsplit('-', 1)[1]}\nv0 {v0_t[i]:.1f} m/s, intent {int(intent[i])}", fontsize=8)
            axx.set_xlabel("left (m)", fontsize=7)
            axx.set_ylabel("forward (m)", fontsize=7)
            axx.grid(alpha=0.3)
        fig.suptitle("Test plans (black: submitted, blue / orange: seeds, green: past track), ego frame")
        fig.tight_layout()
        fig.savefig(out / "bev.png", dpi=110)
        run.info(f"wrote {out}")


# ---------------------------------------------------------------- package
def account(a) -> str:
    if a.account:
        return a.account
    if os.environ.get("WOD_ACCOUNT_NAME"):
        return os.environ["WOD_ACCOUNT_NAME"]
    env = _R / ".env"
    if env.exists():
        for ln in env.read_text().splitlines():
            if ln.startswith("WOD_ACCOUNT_NAME="):
                return ln.split("=", 1)[1].strip().strip("\"'")
    if a.account_from_reference:
        from jevdrive import waymo as W
        return W.read_submission(a.reference)[2]["account_name"]
    raise SystemExit("no account: pass --account, set WOD_ACCOUNT_NAME (env or the gitignored .env), or use --account-from-reference")


def cmd_package(a):
    from jevdrive import waymo as W
    z = np.load(a.traj)
    names, traj = [str(n) for n in z["names"]], z["traj"]
    params = json.loads(a.params.read_text()) if a.params else {}
    n_par = params.get("total_params")
    npar = a.num_params or (f"{round(n_par / 1e6)}M" if n_par else "")
    meta = dict(account_name=account(a), unique_method_name=a.name, authors=["Anonymous"], affiliation="Anonymous", method_link="",
                description=DESCRIPTION, uses_public_model_pretraining=True, public_model_names=["openpilot Cinque v3"], num_model_parameters=npar)
    path = W.write_submission(names, traj, a.out, meta)
    gn, gt, gm = W.read_submission(path)
    assert gn == names and np.allclose(gt, traj, atol=1e-5), "round trip changed names or values"
    assert set(W.submission_frames()) == set(gn), "frame set differs from the manifest"
    rows = []
    shown = {k: ("<hidden>" if k == "account_name" else v) for k, v in gm.items()}
    with tarfile.open(path) as t:
        mem = [(m.name, m.isfile()) for m in t.getmembers()]
    out = {"package": str(path), "members": mem, "meta (account hidden)": shown, "n": len(gn), "shape": list(gt.shape)}
    if a.reference:
        rn, rt, rm = W.read_submission(a.reference)
        with tarfile.open(a.reference) as t:
            rmem = [(m.name, m.isfile()) for m in t.getmembers()]
        cmp = {"member names equal": [m for m, _ in mem] == [m for m, _ in rmem], "metadata field set equal": sorted(gm) == sorted(rm),
               "metadata fields only in one": sorted(set(gm) ^ set(rm)), "frame name set equal": set(gn) == set(rn), "frame order equal": gn == rn,
               "trajectory shape equal": gt.shape == rt.shape, "reference shape": list(rt.shape), "dtype equal": gt.dtype == rt.dtype,
               "metadata values that differ": sorted(k for k in gm if k in rm and gm[k] != rm[k]),
               "account equal to reference": gm["account_name"] == rm.get("account_name"),
               "d5 mean (m) package / reference": [float(np.linalg.norm(gt[:, -1], axis=-1).mean()), float(np.linalg.norm(rt[:, -1], axis=-1).mean())]}
        out["comparison"] = cmp
    print(json.dumps(out, indent=1, default=str))
    if a.report:
        a.report.write_text(json.dumps(out, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("bias").set_defaults(f=cmd_bias)
    sp.add_parser("predict").set_defaults(f=cmd_predict)
    p = sp.add_parser("package")
    p.add_argument("--traj", type=_pl.Path, required=True)
    p.add_argument("--out", type=_pl.Path, required=True)
    p.add_argument("--name", default="LPA-v1", help="unique_method_name: a neutral code name")
    p.add_argument("--account", default="")
    p.add_argument("--account-from-reference", action="store_true")
    p.add_argument("--reference", type=_pl.Path)
    p.add_argument("--params", type=_pl.Path, help="params.json of the bias stage (num_model_parameters = served model + adapter)")
    p.add_argument("--num-params", default="", help="override num_model_parameters, e.g. 386M")
    p.add_argument("--report", type=_pl.Path)
    p.set_defaults(f=cmd_package)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
