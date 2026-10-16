#!/usr/bin/env python3
"""TTS (experiments/route_follow/plans/2026-10-16-tts-prereg.md): serve-side test-time candidate selection, lib/tts.py. Box, repo root.

  rf_tts.py fit [--smoke]          (envs/op-train, one GPU) the weight fit on TRAIN-split own states of on1_L18RSG (prereg 3) -> $O/fit/
                                   {weights.json, fit.json, fit.md, arrays.npz}; rc 3 when the registered stop line holds (no reads)
                                   train / dev = C2's split (rf_recov.dev_of, own.npz `dev`), not jevdrive.data.splits: the weights must be fit
                                   on exactly the rows the consequence head's split calls train
  rf_tts.py ident --pair NAME=OLD:NEW .. --on NAME=DIR ..     (any numpy python) the bit-identity gate (prereg 4): every decision's poses of
                                   NEW (JEV_TTS unset) equal OLD (the code before the change); every inference of an ON run has a `tts`
                                   record -> $O/ident.json; rc 2 when a pair differs or an ON run lacks records
  rf_tts.py held --dir D --runs DIR.. --ref-runs DIR..       (AlpaSim venv) the held lines from D/held18 (ol_confirm) and D/follow (rf_follow) +
                                   switch statistics and latency from the drive logs -> D/held.json, held.md
  rf_tts.py pai --dir D --runs DIR.. --ref-runs DIR.. --held H  (AlpaSim venv) the PAI lines from D/pai_londiag.json, D/pai_turns.json and the
                                   registered adoption rule with the held lines (H/held.json) -> D/pai.json, pai.md
  rf_tts.py stats --runs DIR..     switch statistics of TTS runs (printed)
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np

R_ = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(R_), str(HERE), str(R_ / "lib")]
D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
O = D / "runs/route_follow/tts"
CONS = D / "runs/route_follow/stack/cons"
FEAT = D / "runs/route_mem/feat/on1_L18RSG.npz"
B = 10_000

# prereg 3: the grid and the selection rule
W_AGENT, W_ROUTE, W_DEV, MARGIN = (0.0, 0.1, 0.25, 0.5), (0.0, 0.025, 0.05, 0.1, 0.2), (0.02, 0.05, 0.1, 0.2, 0.4), (0.02, 0.05, 0.1, 0.2)
D_MAX, SW_MAX, R_MAX, TIE, STOP = 0.15, 0.25, 0.0, 0.002, 0.005


def ci(x, g, n=B):
    from jevdrive import stats
    r = stats.bootstrap(np.asarray(x, float), None if g is None else np.asarray(g), n_boot=n)
    return [round(float(r["mean"]), 5), round(float(r["lo"]), 5), round(float(r["hi"]), 5)]


# ---------------------------------------------------------------- fit
def cmd_fit(a):
    import torch
    import tts as TS
    import conshead as CH
    import rf_stack_c2 as C2
    from agent_hinge import AgentHinge
    from drivable_hinge import Hinge
    from jevdrive.run import Run
    out = O / ("fit_smoke" if a.smoke else "fit")
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    with Run("route_follow", f"tts-fit{'-smoke' if a.smoke else ''}", seed=0, config=vars(a)) as run:
        z = dict(np.load(CONS / "own.npz"))
        n = len(z["rows"])
        idx = np.arange(n) if not a.smoke else np.sort(np.random.default_rng(0).choice(n, 600, replace=False))
        front = np.load(CONS / "own_front.npy", mmap_mode="r")
        names, log, dv = z["names"][idx], z["log"][idx], z["dev"][idx].astype(bool)
        inner = np.array([zlib.crc32(("inner:" + str(x)).encode()) % 10 == 0 for x in log])
        P0 = z["plan"][idx].astype(np.float64)
        C = np.stack([TS.cands_np(p) for p in P0]).astype(np.float32)                         # (n, K, 8, 3)
        Fz = np.load(FEAT)
        fmap = {str(k): i for i, k in enumerate(Fz["names"])}
        fi = np.array([fmap.get(str(x), -1) for x in names])
        assert (fi >= 0).all(), f"{(fi < 0).sum()} rows without route features in {FEAT}"
        RF = Fz["f"][fi].astype(np.float32)
        run.info(f"rows {len(idx)} (train {(~dv).sum()}, inner {(inner & ~dv).sum()}, dev {dv.sum()}), K {TS.K}")
        # geometric labels of every candidate (C2's geometry)
        tab_names = np.load(C2.CR / C2.OWN / "tab.npz")["names"]
        ah, rh = AgentHinge(C2.CR / C2.OWN / "agent.npz", tab_names, dev, 0.0, 0.0), Hinge(C2.CR / C2.OWN / "road.npz", tab_names, dev, 0.0)
        lab, _, _ = C2.label(ah, rh, C, z["rows"][idx], dev)
        yA, yR = lab[:, :, 0].any(-1), lab[:, :, 1].any(-1)
        y = yA | yR
        # check: candidate 0 labels = C2's labels of the same plan
        same0 = float((lab[:, 0] == z["lab"][idx, 0]).mean())
        del ah, rh
        torch.cuda.empty_cache()
        # head probabilities (one batched pass per row block)
        head = CH.load(CONS / "head.pt", dev)
        pa, pr = np.zeros((len(idx), TS.K), np.float32), np.zeros((len(idx), TS.K), np.float32)
        ego = z["ego"][idx].astype(np.float32)
        with torch.no_grad():
            for i in range(0, len(idx), 512):
                j = slice(i, i + 512)
                Fr = torch.as_tensor(np.ascontiguousarray(front[idx[j]]), device=dev)
                p = torch.sigmoid(head(Fr, torch.as_tensor(ego[j], device=dev), torch.as_tensor(C[j], device=dev))).amax(-1).float().cpu().numpy()
                pa[j], pr[j] = p[..., 0], p[..., 1]
        dr = np.stack([TS.route_dist(C[i], TS.route_pts(RF[i])) for i in range(len(idx))])
        has_r = np.array([len(TS.route_pts(RF[i])) >= TS.MIN_PTS for i in range(len(idx))])
        dvd = TS.dev_dist(C)
        seg = np.array([TS.segment(RF[i]) for i in range(len(idx))])
        np.savez_compressed(out / "arrays.npz", names=names, log=log, dev=dv, inner=inner, yA=yA, yR=yR, pa=pa, pr=pr, dr=dr, dv=dvd, has_r=has_r, seg=seg)
        tr = ~dv
        rows_ = np.arange(len(idx))

        def evaluate(w, m):
            S = TS.score(pr[m], pa[m], dr[m], dvd[m], w)
            k = TS.choose(S, w["margin"])
            r = rows_[m]
            return k, dict(C=float(y[r, k].mean()), D=float(dvd[r, k].mean()), SW=float((k > 0).mean()),
                           R=float((dr[r, k] - dr[r, 0])[has_r[r]].mean()) if has_r[r].any() else 0.0)
        grid = []
        for wa in W_AGENT:
            for wr in W_ROUTE:
                for wd in W_DEV:
                    for mg in MARGIN:
                        w = dict(w_road=1.0, w_agent=wa, w_route=wr, w_dev=wd, margin=mg)
                        grid.append(dict(w=w, **evaluate(w, tr)[1]))
        feas = [g for g in grid if g["D"] <= D_MAX and g["SW"] <= SW_MAX and g["R"] <= R_MAX]
        c_own = float(y[tr, 0].mean())
        if not feas:
            raise SystemExit("no feasible weight set")
        cmin = min(g["C"] for g in feas)
        best = sorted([g for g in feas if g["C"] <= cmin + TIE], key=lambda g: (-g["w"]["w_route"], g["D"], -g["w"]["margin"]))[0]
        w = best["w"]
        stop = c_own - best["C"] < STOP
        # descriptive reads of the frozen weights per split
        oracle = np.where(y[:, 0], np.argmin(y, 1), 0)                                          # fewest true contacts, ties -> own
        rep = {}
        for nm, m in (("train", tr), ("inner", tr & inner), ("dev", dv)):
            if not m.any():
                continue
            k, ev = evaluate(w, m)
            r = rows_[m]
            g = log[r]
            rep[nm] = dict(rows=int(m.sum()), logs=int(len(set(g))), **{q: round(v, 5) for q, v in ev.items()},
                           contact_own=ci(y[r, 0], g), contact_sel=ci(y[r, k], g), contact_diff=ci(y[r, k].astype(float) - y[r, 0], g),
                           agent_diff=ci(yA[r, k].astype(float) - yA[r, 0], g), road_diff=ci(yR[r, k].astype(float) - yR[r, 0], g),
                           oracle_contact=round(float(y[r, oracle[r]].mean()), 5),
                           picked={TS.NAMES[j]: int(c) for j, c in zip(*np.unique(k, return_counts=True))},
                           switch_by_seg={s_: [int(((k > 0) & (seg[r] == s_)).sum()), int((seg[r] == s_).sum())] for s_ in ("approach", "junction", "straight", "none")},
                           contact_by_cand=y[r].mean(0).round(4).tolist())
        res = dict(K=TS.K, names=list(TS.NAMES), weights=w, stop=bool(stop), c_own_train=c_own, c_best_train=best["C"], best=best,
                   feasible=len(feas), grid_size=len(grid), rule=dict(D_MAX=D_MAX, SW_MAX=SW_MAX, R_MAX=R_MAX, TIE=TIE, STOP=STOP),
                   label_check_cand0_equals_c2=same0, reads=rep, head=str(CONS / "head.pt"), feat=str(FEAT),
                   grid_top=sorted(feas, key=lambda g: g["C"])[:15], unconstrained_min=min(grid, key=lambda g: g["C"]))
        (out / "fit.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
        (out / "weights.json").write_text(json.dumps(w, indent=1) + "\n")
        (out / "fit.md").write_text(fit_md(res))
        run.summary.update(weights=w, stop=bool(stop), c_own=c_own, c_best=best["C"])
        print(fit_md(res))
        if stop and not a.smoke:
            raise SystemExit(3)


def fit_md(r):
    f = lambda v, p=4: f"{v[0]:+.{p}f} [{v[1]:+.{p}f}, {v[2]:+.{p}f}]"  # noqa: E731
    w = r["weights"]
    md = ["# TTS weight fit (prereg plans/2026-10-16-tts-prereg.md section 3)", "",
          f"Rows: on1_L18RSG own states kept by C2 (t0 not in contact, both labels), K = {r['K']} candidates ({', '.join(r['names'])}). "
          f"Grid {r['grid_size']} sets, feasible {r['feasible']} (mean deviation <= {r['rule']['D_MAX']} m, switch share <= {r['rule']['SW_MAX']}, "
          f"route distance change <= {r['rule']['R_MAX']} m). Candidate-0 labels equal C2's on {r['label_check_cand0_equals_c2']:.4f} of the bins.", "",
          f"**Frozen weights**: w_road {w['w_road']}, w_agent {w['w_agent']}, w_route {w['w_route']} / m, w_dev {w['w_dev']} / m, margin {w['margin']}. "
          f"Train contact {r['c_own_train']:.4f} (own plan) -> {r['c_best_train']:.4f} (selected); stop line (< {r['rule']['STOP']} lower): "
          f"**{'STOP, no reads' if r['stop'] else 'not hit, reads go ahead'}**. Unconstrained grid minimum {r['unconstrained_min']['C']:.4f} "
          f"(D {r['unconstrained_min']['D']:.3f} m, switch {r['unconstrained_min']['SW']:.3f}).", "",
          "| split | rows / logs | contact own | contact selected | d contact [log CI] | d agent | d road | switch | mean dev (m) | d route (m) | oracle |",
          "|:--|--:|--:|--:|:--|:--|:--|--:|--:|--:|--:|"]
    for nm, x in r["reads"].items():
        md.append(f"| {nm} | {x['rows']} / {x['logs']} | {x['contact_own'][0]:.4f} | {x['contact_sel'][0]:.4f} | {f(x['contact_diff'])} | {f(x['agent_diff'])} | "
                  f"{f(x['road_diff'])} | {x['SW']:.3f} | {x['D']:.3f} | {x['R']:+.3f} | {x['oracle_contact']:.4f} |")
    md += ["", "Picked candidates and switches by segment (switched / decisions):", ""]
    for nm, x in r["reads"].items():
        md.append(f"- {nm}: picked {x['picked']}; by segment {x['switch_by_seg']}")
    md += ["", "Contact rate by candidate (train): " + ", ".join(f"{n} {v:.3f}" for n, v in zip(r["names"], r["reads"]["train"]["contact_by_cand"])), "",
           "train = the head's own training rows (its probabilities are optimistic there); inner = train logs held out of the head's gradient; dev = held-out "
           "logs (4 logs). Only train chose the weights.", ""]
    return "\n".join(md)


# ---------------------------------------------------------------- drive logs
def drive_logs(run):
    """Run dir(s) -> {scene: [inference records of the last session]} (nuPlan: run/driver-logs; PAI: sim/driver-logs)."""
    out = {}
    for f in sorted(glob.glob(str(Path(run) / "**/driver-logs/drive.jsonl"), recursive=True)) or sorted(glob.glob(str(Path(run) / "drive.jsonl"))):
        st, rec = {}, collections.defaultdict(list)
        for line in open(f):
            r = json.loads(line)
            if r.get("kind") == "start":
                st[r["session"]] = r
            elif r.get("kind") == "drive" and "poses" in r and r.get("infer", True):
                rec[r["session"]].append(r)
        if (Path(f).parent / "start.jsonl").exists():
            for line in open(Path(f).parent / "start.jsonl"):
                r = json.loads(line)
                st.setdefault(r["session"], r)
        for u, s in st.items():
            sc = s.get("scene") or (rec[u][0]["scene"] if rec[u] else None)
            if sc and rec[u]:
                out[sc] = sorted(rec[u], key=lambda r: r["k"])
    return out


def cmd_ident(a):
    res, ok = {}, True
    for p in a.pair:
        name, dirs = p.split("=", 1)
        A, Bn = (drive_logs(x) for x in dirs.split(":"))
        d, nd, miss = [], 0, 0
        for sc in sorted(set(A) | set(Bn)):
            ka = {r["k"]: np.asarray(r["poses"]) for r in A.get(sc, [])}
            kb = {r["k"]: np.asarray(r["poses"]) for r in Bn.get(sc, [])}
            miss += len(set(ka) ^ set(kb))
            for k in set(ka) & set(kb):
                d.append(float(np.abs(ka[k] - kb[k]).max()))
        d = np.array(d)
        e = dict(scenes=len(set(A) & set(Bn)), decisions=int(len(d)), unmatched=miss, equal_share=float((d == 0).mean()) if len(d) else None,
                 max_abs_m=float(d.max()) if len(d) else None, identical=bool(len(d) and (d == 0).all() and miss == 0 and len(set(A)) == len(set(Bn)) >= 2))
        ok &= e["identical"]
        res[name] = e
    for p in a.on or []:
        name, dirs = p.split("=", 1)
        X = drive_logs(dirs)
        recs = [r for v in X.values() for r in v]
        t = [r["tts"] for r in recs if "tts" in r]
        ms = np.array([x["ms"] for x in t]) if t else np.zeros(0)
        e = dict(scenes=len(X), inferences=len(recs), with_tts=len(t), switched=int(sum(x["k"] > 0 for x in t)),
                 picked=dict(collections.Counter(x["name"] for x in t if x["k"])), reasons=dict(collections.Counter(x["reason"] for x in t)),
                 ms_p50=float(np.percentile(ms, 50)) if len(ms) else None, ms_p95=float(np.percentile(ms, 95)) if len(ms) else None,
                 ok=bool(len(X) >= 2 and len(recs) and len(t) == len(recs)))
        ok &= e["ok"]
        res[name] = e
    res["pass"] = bool(ok)
    O.mkdir(parents=True, exist_ok=True)
    (Path(a.out) if a.out else O / "ident.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    raise SystemExit(0 if ok else 2)


def switch_stats(runs, ref_runs=None):
    import tts as TS
    X = {}
    for r in runs:
        X.update(drive_logs(r))
    t = [(sc, r) for sc, v in X.items() for r in v if "tts" in r]
    seg = collections.Counter(r["tts"]["seg"] for _, r in t)
    sw = collections.Counter(r["tts"]["seg"] for _, r in t if r["tts"]["k"])
    by_type = {s_: dict(collections.Counter(TS.TYPE[r["tts"]["k"]] for _, r in t if r["tts"]["k"] and r["tts"]["seg"] == s_)) for s_ in seg}
    by_name = dict(collections.Counter(r["tts"]["name"] for _, r in t if r["tts"]["k"]))
    ms = np.array([r["tts"]["ms"] for _, r in t]) if t else np.zeros(1)
    tot = lambda recs: np.array([r["ms"]["total"] if isinstance(r.get("ms"), dict) and "total" in r["ms"] else r.get("total_ms", np.nan) for r in recs], float)  # noqa: E731
    out = dict(scenes=len(X), decisions=len(t), switched=int(sum(sw.values())), switch_share=round(sum(sw.values()) / max(len(t), 1), 4),
               scenes_with_switch=len({sc for sc, r in t if r["tts"]["k"]}),
               by_segment={s_: dict(decisions=int(seg[s_]), switched=int(sw[s_]), share=round(sw[s_] / max(seg[s_], 1), 4), types=by_type.get(s_, {}))
                           for s_ in ("approach", "junction", "straight", "none")},
               picked=dict(sorted(by_name.items(), key=lambda x: -x[1])), reasons=dict(collections.Counter(r["tts"]["reason"] for _, r in t)),
               tts_ms=[round(float(np.percentile(ms, q)), 2) for q in (50, 95, 99)],
               total_ms=[round(float(np.nanpercentile(tot([r for _, r in t]), q)), 2) for q in (50, 95)] if t else None)
    if ref_runs:
        Y = {}
        for r in ref_runs:
            Y.update(drive_logs(r))
        tt = tot([r for v in Y.values() for r in v])
        out["ref_total_ms"] = [round(float(np.nanpercentile(tt, q)), 2) for q in (50, 95)] if len(tt) else None
    return out


def cmd_stats(a):
    print(json.dumps(switch_stats(a.runs, a.ref_runs), indent=1))


# ---------------------------------------------------------------- held / PAI lines
def cmd_held(a):
    import pandas as pd
    import rf_recov_post as RP
    d = Path(a.dir)
    arm, ref = a.arm, a.ref
    cj = json.loads((d / "held18/confirm.json").read_text())["held18"][arm]
    S = pd.read_csv(d / "held18/per_scene.csv")
    S["tag"] = S.label.str.split("@").str[0]
    z = S.score == 0
    S["z_col"], S["z_off"], S["z_cor"] = ((z & (S[c] > 0)).astype(int) for c in ("collision_at_fault", "offroad", "left_corridor_laterally"))
    S["zero"], S["z_atf"] = z.astype(int), (z & (S.offroad_or_collision_at_fault > 0)).astype(int)
    sc = set(S[(S.tag == arm) & S.offroad_or_collision_at_fault.notna()].scene) & set(S[(S.tag == ref) & S.offroad_or_collision_at_fault.notna()].scene)
    X = S[S.scene.isin(sc)]
    split = {}
    for c in ("zero", "z_atf", "z_col", "z_off", "z_cor"):
        P = X.pivot_table(index="scene", columns="tag", values=c, aggfunc="first")
        dd = (P[arm] - P[ref])
        logs = np.array([s_.rsplit("-", 1)[0] for s_ in dd.index])
        r = ci(dd.to_numpy(), logs)
        split[c] = dict(arm=int(P[arm].sum()), ref=int(P[ref].sum()), d=int(P[arm].sum() - P[ref].sum()), d_ci_total=[round(v * len(dd), 2) for v in r])
    h2 = RP.h2_held(d, [arm], [ref])
    st = switch_stats(a.runs, a.ref_runs)
    zeros_ok, atf_ok = cj["zero"][0] <= 3, cj["z_atf"][0] <= 3
    turns_ok = None if h2 is None else h2["seed_mean_diff"] >= 3
    res = dict(arm=arm, ref=ref, scenes=len(sc), confirm=cj, split=split, turns=h2, stats=st, zeros_ok=bool(zeros_ok), atf_ok=bool(atf_ok),
               turns_ok=turns_ok, stage2=bool(zeros_ok))
    (d / "held.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    f = lambda v, p=1: f"{v[0]:+.{p}f} [{v[1]:+.{p}f}, {v[2]:+.{p}f}]"  # noqa: E731
    md = [f"# TTS held 460 @18: {arm} vs the stored {ref} read", "",
          f"Paired {len(sc)} scenes, log-clustered 95 % CI ({B} draws). Serve = rm_chain.sh read (the stored read's path) + JEV_TTS=1 with the frozen weights "
          f"(results/tts/weights.json).", "",
          "| line | value [95 % CI] | registered | result |", "|:--|:--|:--|:--|",
          f"| held zeros d | {f(cj['zero'])} | <= +3 (point) | {'PASS' if zeros_ok else 'FAIL'} |",
          f"| held at-fault zeros d | {f(cj['z_atf'])} | <= +3 (point) | {'PASS' if atf_ok else 'FAIL'} |",
          f"| held turns completed d (of {h2['common_turns'] if h2 else '?'}) | {f([h2['seed_mean_diff'], *h2['paired'][1:]]) if h2 else 'pending'} | >= +3 (or the PAI line) | "
          f"{'-' if turns_ok is None else 'PASS' if turns_ok else 'not met'} |",
          f"| held mean score d | {f(cj['score'], 4)} | reported | - |", f"| held GT distance d (m) | {f(cj['gt'], 3)} | reported | - |", "",
          "Zeros by kind (a scene can count in several kinds): " + "; ".join(f"{k} {v['arm']} vs {v['ref']} (d {v['d']:+d}, CI {v['d_ci_total'][1]:+.1f} .. {v['d_ci_total'][2]:+.1f})"
                                                                      for k, v in split.items()) + ".",
          ("Turns: " + "; ".join(f"{t} {c['completed']} / {c['turns']}" for t, c in h2["counts"].items()) + ".") if h2 else "Turns: pending.", "",
          f"Stage 2 (PAI 281): {'goes ahead (held zeros d <= +3)' if zeros_ok else 'not run (held zeros d > +3)'}.", "", switch_md(st)]
    (d / "held.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def switch_md(st):
    md = [f"Switch statistics ({st['decisions']} decisions, {st['scenes']} scenes): switched {st['switched']} ({100 * st['switch_share']:.1f} %), in {st['scenes_with_switch']} scenes; "
          f"reasons {st['reasons']}.", "", "| segment | decisions | switched | share | candidate types picked |", "|:--|--:|--:|--:|:--|"]
    for s_, v in st["by_segment"].items():
        md.append(f"| {s_} | {v['decisions']} | {v['switched']} | {100 * v['share']:.1f} % | {v['types']} |")
    md += ["", f"Picked candidates: {st['picked']}.",
           f"Latency: tts step p50 / p95 / p99 {st['tts_ms']} ms; driver total per inference p50 / p95 {st['total_ms']} ms"
           + (f" vs reference {st.get('ref_total_ms')} ms." if st.get("ref_total_ms") else ".")]
    return "\n".join(md)


def cmd_pai(a):
    d = Path(a.dir)
    pl = json.loads((d / "pai_londiag.json").read_text())
    pt = json.loads((d / "pai_turns.json").read_text())
    tr = json.loads((d / "pai_track.json").read_text()) if (d / "pai_track.json").exists() else {}
    held = json.loads((Path(a.held) / "held.json").read_text())
    strict = [x.split()[0] for x in open(a.strict) if x.strip()]
    by = {(r["label"], r["scene"]): r for r in pt["rows"]}
    ps = {lab: sum(bool(by.get((lab, s_), {}).get("completed")) for s_ in strict) for lab in ("arm", "ref")}
    n, r = pl["new"], pl["ref_counts"]
    st = switch_stats(a.runs, a.ref_runs)
    pai_ok = ps["arm"] >= 8 and n["zeros"] <= r["zeros"] + 3
    adopt = bool(held["zeros_ok"] and held["atf_ok"] and (bool(held["turns_ok"]) or pai_ok))
    res = dict(strict36=ps, londiag=dict(new=n, ref=r, exam72_mean_diff=pl.get("exam72_mean_diff")), track=tr, stats=st, pai_line=bool(pai_ok), adopt=adopt,
               held=dict(zeros=held["confirm"]["zero"], at_fault=held["confirm"]["z_atf"], turns=(held["turns"] or {}).get("seed_mean_diff")))
    (d / "pai.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    md = [f"# TTS PAI 281 seed 0: RSG-L18-s0 + JEV_TTS vs the box reference RSG-L18-s0", "",
          f"Serve on both: gate (track {tr or '?'}), gains 20, JEV_VCONT 1.0, JEV_LEAD 1, the Tokyo reference's 8 chunk lists; arm adds JEV_TTS=1 with the frozen weights.", "",
          "| line | arm | reference | d |", "|:--|--:|--:|--:|",
          f"| zeros (all 281) | {n['zeros']} | {r['zeros']} | {n['zeros'] - r['zeros']:+d} |",
          f"| at-fault (collision zeros) | {n['coll_zeros']} | {r['coll_zeros']} | {n['coll_zeros'] - r['coll_zeros']:+d} |",
          f"| strict 36 completed | {ps['arm']} | {ps['ref']} | {ps['arm'] - ps['ref']:+d} |",
          f"| cor_route_end | {n['cor_route_end']} | {r['cor_route_end']} | {n['cor_route_end'] - r['cor_route_end']:+d} |",
          f"| rear-ended | {n['rear_ended']} | {r['rear_ended']} | {n['rear_ended'] - r['rear_ended']:+d} |",
          f"| mean score | {n['mean']:.4f} | {r['mean']:.4f} | {n['mean'] - r['mean']:+.4f} |", "",
          f"PAI line (strict 36 >= 8 and zeros <= reference + 3): **{'met' if pai_ok else 'not met'}**. Held: zeros d {held['confirm']['zero'][0]:+.1f}, at-fault d "
          f"{held['confirm']['z_atf'][0]:+.1f}, turns d {res['held']['turns']}.", "",
          f"**Registered rule (held zeros <= +3 and at-fault <= +3, and (held turns >= +3 or the PAI line)): {'ADOPT as a serve option' if adopt else 'NOT ADOPTED'}.**", "",
          switch_md(st)]
    (d / "pai.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("fit")
    p.add_argument("--smoke", action="store_true")
    p = sp.add_parser("ident")
    p.add_argument("--pair", nargs="+", required=True), p.add_argument("--on", nargs="*"), p.add_argument("--out", default="")
    for nm in ("held", "pai"):
        p = sp.add_parser(nm)
        p.add_argument("--dir", required=True), p.add_argument("--runs", nargs="+", required=True), p.add_argument("--ref-runs", nargs="*")
        p.add_argument("--arm", default="RSG-L18-s0-tts"), p.add_argument("--ref", default="RSG-L18-s0")
        if nm == "pai":
            p.add_argument("--held", required=True), p.add_argument("--strict", default=str(R_ / "experiments/route_follow/results/pai281/strict36.txt"))
    p = sp.add_parser("stats")
    p.add_argument("--runs", nargs="+", required=True), p.add_argument("--ref-runs", nargs="*")
    a = ap.parse_args()
    dict(fit=cmd_fit, ident=cmd_ident, held=cmd_held, pai=cmd_pai, stats=cmd_stats)[a.cmd](a)
