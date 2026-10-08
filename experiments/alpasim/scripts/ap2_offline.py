"""Offline read of AlpaSim-aligned inputs on navtest (plans/2026-10-08-alpasim-aligned-prereg.md): every model plans all 12 146 navtest
tokens as decision k = m - 1 of an AlpaSim rollout, m = 1..4 keyframes, with the inputs AlpaSim would deliver (ap2_core.ego_table: route
command from the rebuilt route, DynamicState definitions, truncated history) and its own cold-start rule.

  plans   --models label=tag[:rule] ...   rule = zero (AP2: only the real slots) | backwarp (SH30 as served today); default: the rule the
          checkpoint was trained with, backwarp for op_parity tags. Also `<label>_nav`: the NAVSIM-standard inputs (m = 4, the tab's ego).
          -> $DATA_DIR/runs/alpasim/ap2/offline/<name>/poses.npz (tokens + (N, 8, 3) per `<label>_m<m>`), subset.txt (2 000 seed-0 tokens +
          the scene tokens of --scenes, for `python -m jevdrive.bench score-poses --traffic non_reactive`)
  report  ADE / 4 s longitudinal error vs the log per label and m, paired differences vs --ref (bootstrap over logs), EPDMS of the scored
          subset when --scores CSV is given -> <out>.md / .json
  closed  --runs name=<AlpaSim run dir> ...   driver logs of closed-loop runs against the logs (plan at decision 3 vs the logged future, ...)

  $DATA_DIR/envs/op-train/bin/python experiments/alpasim/scripts/ap2_offline.py plans --name pilot --models SH30=SH30-F-s0 A=AP2P-A-s0
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_R / "experiments/op_parity/scripts"),
                 str(_R / "experiments/alpasim/lib")]
import argparse, json  # noqa: E401,E402

import numpy as np  # noqa: E402

from jevdrive.common import data_dir  # noqa: E402

AROOT = data_dir() / "runs/alpasim/ap2"
DATA = "lb_navtest"


def cmd_plans(a):
    import torch
    import ap2_core as AC
    import ap2_inputs as AI
    from jevdrive import navsim_zs as Z
    from jevdrive import op_interp as I
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("alpasim", f"ap2-offline-{a.name}", config=vars(a)) as run:
        run.use_split(splits.load("navsim/navtest"))
        dev = torch.device("cuda")
        cr = data_dir() / "runs/op_parity/cache"
        tab = dict(np.load(cr / DATA / "tab.npz"))
        N = len(tab["names"])
        rz = np.load(AROOT / "route" / f"{DATA}.npz")
        assert rz["names"].tolist() == tab["names"].tolist()
        front = np.load(cr / f"{DATA}@warp" / "front.npy", mmap_mode="r")
        cold = {r: np.load(AROOT / "cache" / DATA / f, mmap_mode="r") for r, f in (("zero", "cold.npy"), ("backwarp", "bw.npy"))}
        tc = np.where(tab["lht"][:, None], [[0.0, 1.0]], [[1.0, 0.0]]).astype(np.float32)
        out = {"tokens": tab["names"]}
        for spec in a.models:
            label, tag = spec.split("=", 1)
            tag, _, rule = tag.partition(":")
            model, route, trained = AC.load_model(tag, dev)
            rule = rule or trained
            ego = {m: AC.ego_table(tab, rz["wp"], route)[:, m - 1] for m in (1, 2, 3, 4)}
            if not route:
                ego["nav"] = tab["ego"]
            s = model.net.slices["plan"].start
            for key, eg in ego.items():
                m = 4 if key == "nav" else key
                poses = np.zeros((N, 8, 3), np.float32)
                with torch.no_grad():
                    for i in run.tqdm(range(0, N, 256), desc=f"{label} {key}"):
                        r = np.arange(i, min(i + 256, N))
                        if m == 4:
                            f, n = np.asarray(front[i:i + 256]), 8
                        elif rule == "backwarp":
                            f, n = np.asarray(cold[rule][i:i + 256, m - 1]), 8
                        else:
                            f, n = np.asarray(cold[rule][i:i + 256, AI.COLD_AT[m]]), AI.N_SLOT[m]
                        o = model(torch.from_numpy(f).to(dev), torch.from_numpy(eg[r]).to(dev), torch.from_numpy(tc[r]).to(dev)).float()
                        mu = o[:, s:s + 33 * 15].reshape(-1, 33, 15).cpu().numpy()
                        poses[r] = [I.to_rear(mu[j, :, 0:3], mu[j, :, 11], I.T_IDXS, tab["cam"][q, :2], Z.T_OUT, "lever") for j, q in enumerate(r)]
                k = f"{label}_{'nav' if key == 'nav' else 'm%d' % key}"
                out[k] = poses
                ade = np.linalg.norm(poses[..., :2] - tab["fut"][..., :2], axis=-1).mean(1)
                run.info(f"{k} ({tag}, rule {rule}, route {route}): ADE vs log {np.nanmean(ade):.4f} m, n slots {n}")
                run.summary[f"ade_{k}"] = float(np.nanmean(ade))
            del model
        d = AROOT / "offline" / a.name
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / "poses.npz", **out)
        first = [s.rsplit("-", 1)[1] for s in _pl.Path(a.scenes).read_text().split()] if a.scenes else []
        have = set(tab["names"].tolist())
        ok = ~np.isnan(tab["fut"]).any((1, 2)) & rz["ok"].all(1)
        rest = [t for t in tab["names"][np.random.default_rng(0).permutation(N)] if ok[np.flatnonzero(tab["names"] == t)[0]]]
        sub = list(dict.fromkeys([t for t in first if t in have] + rest[: a.n_sub]))
        (d / "subset.txt").write_text("\n".join(sub) + "\n")
        run.summary |= {"poses": str(d / "poses.npz"), "keys": [k for k in out if k != "tokens"], "subset": len(sub)}


def cmd_report(a):
    from jevdrive import stats
    d = AROOT / "offline" / a.name
    z = np.load(d / "poses.npz")
    tab = np.load(data_dir() / "runs/op_parity/cache" / DATA / "tab.npz")
    rz = np.load(AROOT / "route" / f"{DATA}.npz")
    fut, log = tab["fut"], tab["log"]
    labels = list(dict.fromkeys(k.rsplit("_", 1)[0] for k in z.files if k != "tokens"))
    use = ~np.isnan(fut).any((1, 2)) & rz["ok"].all(1)
    err = {k: np.where(use, np.linalg.norm(z[k][..., :2] - fut[..., :2], axis=-1).mean(1), np.nan) for k in z.files if k != "tokens"}
    dx4 = {k: np.where(use, z[k][:, -1, 0] - fut[:, -1, 0], np.nan) for k in err}
    sc = None
    if a.scores:
        import csv
        sc = {}
        for r in csv.DictReader(open(a.scores)):
            sc.setdefault(r["key"], {})[r["token"]] = 100 * float(r["score"])
    row = {t: i for i, t in enumerate(z["tokens"].tolist())}
    sub = [t for t in (d / "subset.txt").read_text().split()]
    sub_log = log[[row[t] for t in sub]]
    R, L = {}, [f"Tokens: {int(use.sum())} of {len(use)} navtest tokens with a logged future and a rebuilt route for every m; EPDMS (no EC, non-reactive traffic) on "
                f"{len(sub)} of them. Paired differences vs `{a.ref}` at the same m, bootstrap over logs.", ""]
    hdr = "| arm | inputs | ADE vs log (m) | vs ref [95% CI] | x error at 4 s (m) |" + (" EPDMS subset | vs ref [95% CI] |" if sc else "")
    L += [hdr, "|:--|:--|--:|:--|--:|" + ("--:|:--|" if sc else "")]
    for lab in labels:
        for key in ("nav", "m4", "m3", "m2", "m1"):
            k = f"{lab}_{key}"
            if k not in err:
                continue
            rk = f"{a.ref}_{key}" if f"{a.ref}_{key}" in err else None
            p = stats.paired(err[k], err[rk], groups=log) if rk and rk != k else None
            rec = {"ade": float(np.nanmean(err[k])), "dx4": float(np.nanmean(dx4[k])), "ade_vs_ref": p}
            line = (f"| {lab} | {'NAVSIM standard, m = 4' if key == 'nav' else 'AlpaSim standard, m = ' + key[1]} | {rec['ade']:.3f} | "
                    f"{stats.fmt(p) if p else ''} | {rec['dx4']:+.2f} |")
            if sc and k in sc:
                e = np.array([sc[k].get(t, np.nan) for t in sub])
                q = stats.paired(e, np.array([sc[rk].get(t, np.nan) for t in sub]), groups=sub_log) if rk and rk != k and rk in sc else None
                rec |= {"epdms": float(np.nanmean(e)), "epdms_vs_ref": q}
                line += f" {rec['epdms']:.2f} | {stats.fmt(q, '.2f') if q else ''} |"
            elif sc:
                line += " | |"
            R[k] = rec
            L.append(line)
    out = _pl.Path(a.out) if a.out else d / "report"
    out.with_suffix(".md").write_text("\n".join(L) + "\n")
    out.with_suffix(".json").write_text(json.dumps(R, indent=1))
    print("\n".join(L))


def cmd_closed(a):
    """Closed-loop runs against the logs: per run, the 8-pose plan of decision 3 (= the scene token's t0) vs the logged future, the fed
    command vs NAVSIM's `driving_command`, the fed ego state vs the recorded one, planned 4 s travel per decision, extra driver counters."""
    tab = np.load(data_dir() / "runs/op_parity/cache" / DATA / "tab.npz")
    row = {t: i for i, t in enumerate(tab["names"].tolist())}
    L = ["| run | scenes | plan at decision 3 vs logged future (m) | command = NAVSIM command at decision 3 | fed vx / ax minus recorded at decision 3 (mean abs) | "
         "planned x at 4 s, decisions 0 / 1 / 2 / 3 / 6 / 9 (m, mean) | logged x at 4 s from decision 3 (m) | extra counters |", "|:--|--:|--:|--:|:--|:--|--:|:--|"]
    for spec in a.runs:
        name, run = spec.split("=", 1)
        rows = [json.loads(x) for x in open(_pl.Path(run) / "driver-logs/drive.jsonl")]
        dr, cl = [r for r in rows if r["kind"] == "drive"], [r for r in rows if r["kind"] == "close"]
        d3 = [(r, row[r["scene"].rsplit("-", 1)[1]]) for r in dr if r["k"] == 3 and r["scene"].rsplit("-", 1)[1] in row]
        ade = [float(np.linalg.norm(np.array(r["poses"])[:, :2] - tab["fut"][i][:, :2], axis=1).mean()) for r, i in d3]
        agree = sum(r["cmd"] == int(np.argmax(tab["cmd"][i, -1])) for r, i in d3)
        dv = np.mean([abs(10 * r["ego"][4] - tab["vel"][i, -1, 0]) for r, i in d3])
        da = np.mean([abs(3 * r["ego"][6] - tab["acc"][i, -1, 0]) for r, i in d3])
        x4 = [np.mean([r["poses"][-1][0] for r in dr if r["k"] == k]) for k in (0, 1, 2, 3, 6, 9)]
        extra = {k: sum(c.get(k, 0) for c in cl) for k in ("state_rotated", "state_rotated_slow", "cmd_rule_diff", "cold")}
        L.append(f"| {name} | {len(cl)} | {np.nanmean(ade):.2f} | {agree} / {len(d3)} | {dv:.2f} m/s / {da:.2f} m/s^2 | " + " / ".join(f"{v:.1f}" for v in x4) +
                 f" | {np.nanmean([tab['fut'][i][-1, 0] for _, i in d3]):.1f} | {json.dumps(extra)} |")
    out = "\n".join(L) + "\n"
    if a.out:
        _pl.Path(a.out).write_text(out)
    print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plans")
    p.add_argument("--name", required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--scenes", default="")
    p.add_argument("--n-sub", type=int, default=2000)
    r = sub.add_parser("report")
    r.add_argument("--name", required=True)
    r.add_argument("--ref", required=True)
    r.add_argument("--scores", default="")
    r.add_argument("--out", default="")
    c = sub.add_parser("closed")
    c.add_argument("--runs", nargs="+", required=True, help="name=<run dir> ...")
    c.add_argument("--out", default="")
    a = ap.parse_args()
    {"plans": cmd_plans, "report": cmd_report, "closed": cmd_closed}[a.cmd](a)
