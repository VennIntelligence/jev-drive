"""JT, the joint L18 arm of NR1 prereg supplement 9 (plans/2026-10-17-nr1-prereg.md): one stage from RB0-F-s0 that trains navtrain + yaw-rate
imitation, the big on-policy pool (S hinges, lateral 0.5v2, the two-sided arc term) and trainval imitation rows (reasoned standing / launch
at a dose) in the same batches. Box, repo root; chain in jt.sh.

  weights --pool on1_JT --k-s 32 --k-tv 38 --hold-pct 15 --tv D10 [--no-wait] --out NAME   (op-train, CPU) per-row draw weights of the mixed
      pool for rm_train --on-wfile -> runs/defect_fix/w/NAME.npz (+ grp: S / S_held / the trainval groups of df.py weights) and
      results/jt/w_NAME.json. S rows (`mode` self): mass k_s / (k_s + k_tv), uniform over the rows of the logs not held out
      (crc32(log) % 100 >= hold-pct; rm_train --on-hold-pct applies the same rule and then finds them at weight 0); trainval rows (`mode`
      both): mass k_tv / (k_s + k_tv) split as df.py's file --tv (names on:<token> -> on:tv:<token>); --no-wait: the standing / launch
      groups at 0, the rest renormalised (= df.py weights --wait 0 --launch 0).
  arena --tags T@V .. --out DIR   (alpasim venv, CPU) the arena held-split readout of JT's reads (runs/defect_fix/jt/arena/<tag>@<off|leash3>)
      against the stored RSG-L18-s0 reads (no hold: the arena's reads; leash3: ARENA-HOLD's): per stratum, zero types, partial types, paired
      CIs clustered by log -> DIR/arena.md + rows.csv.gz
  held --tags T ..   (op-train) held 460 @18 against RSG-L18-s0 (df.py cl_read) -> stdout
  diag --tag T [--lo 500] [--hi 1000]   (any python) a training run's events.jsonl: per-term trunk gradient norms and shares, activity, loss
      terms, dev ADE, pool slices and the wait / launch probe over steps lo..hi, plus the arc weight rule of supplement 9 amendment A
  gate --tag JTSM-L18-s0 --w 0.0025 --wjson results/jt/w_JT10.json --out FILE [--ratio R] [--no-probe-gate]   the smoke gates of supplement 9 (no NaN; every pool term
      active; tv imitation logged; dev ADE at the end <= step 0 + 0.06 (amendment C; + 0.03 reported); probe stand_r stay1m above step 0) -> FILE (json: pass, arc W by the
      amendment-A rule, steps = floor(4.6 x S trainable rows / 32 / 500) x 500); exit 2 when a gate fails (the arms after it do not start)
  defer --after ID,ID [--dry]   (repo .venv) the main session's rule (10-18): the still-QUEUED third-party anchor reads
      ar-read-{DD-sample,LTF-sample,starter,G0014}-c* are cancelled and resubmitted with their exact recorded spec (cmd, env, resources,
      priority, tries, log dir) plus --after the JT trainings; ar-sum-all2 is resubmitted identically after the new ids and the anchor reads
      still running; ids and a reason appended to runs/alpasim_proxy/arena/{jobs.txt,stages.txt}. Running jobs are never touched.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zlib
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
CR = DATA / "runs/op_parity/cache"
OUT = DATA / "runs/defect_fix"
RES = REPO / "experiments/defect_fix/results/jt"
ARENA = DATA / "runs/alpasim_proxy/arena"


def held_logs(logs, pct):
    return {g for g in set(logs) if zlib.crc32(g.encode()) % 100 < pct}


def cmd_weights(a):
    tb = np.load(CR / a.pool / "tab.npz")
    names, mode, logs = tb["names"].astype(str), tb["mode"].astype(str), tb["log"].astype(str)
    S = mode == "self"
    hl = held_logs(logs[S], a.hold_pct)
    held = S & np.isin(logs, sorted(hl))
    w = np.zeros(len(names))
    grp = np.where(held, "S_held", np.where(S, "S", "")).astype("U12")
    fs = a.k_s / (a.k_s + a.k_tv)
    w[S & ~held] = fs / (S & ~held).sum()
    tv = np.flatnonzero(~S)
    z = np.load(OUT / "w" / f"{a.tv}.npz")
    pos = {t: i for i, t in enumerate(z["names"].astype(str).tolist())}
    j = np.array([pos["on:" + n.split(":", 2)[2]] for n in names[tv]])
    wt, gt = z["w"][j].astype(np.float64), z["grp"].astype(str)[j]
    assert len(set(j.tolist())) == len(j) == len(z["names"]), "trainval rows != the wfile's rows"
    if a.no_wait:
        wt = np.where(np.isin(gt, ["stand_r", "stand_u", "launch"]), 0.0, wt)
    w[tv] = (1.0 - fs) * wt / wt.sum()
    grp[tv] = gt
    w = w / w.sum()
    k = a.k_s + a.k_tv
    rec = dict(pool=a.pool, rows=len(names), s_rows=int(S.sum()), tv_rows=int((~S).sum()), k_s=a.k_s, k_tv=a.k_tv, on_k=k, hold_pct=a.hold_pct,
               held_logs=sorted(hl), s_logs=len(set(logs[S])), held_rows=int(held.sum()), tv=a.tv, no_wait=a.no_wait,
               share={g: float(w[grp == g].sum()) for g in np.unique(grp)}, per_batch={g: float(w[grp == g].sum() * k) for g in np.unique(grp)},
               rows_by_group={g: int((grp == g).sum()) for g in np.unique(grp)},
               draws_per_row_10k={g: float(w[grp == g].sum() * k * 1e4 / max((grp == g).sum(), 1)) for g in np.unique(grp)},
               s_parts={p: int((S & (tb["pool"].astype(str) == p)).sum()) for p in np.unique(tb["pool"].astype(str)[S])})
    (OUT / "w").mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "w" / f"{a.out}.npz", names=tb["names"], w=w, grp=grp)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"w_{a.out}.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


def cmd_arena(a):
    import pandas as pd
    sys.path[:0] = [str(REPO), str(REPO / "experiments/alpasim_proxy/scripts")]
    import arena as A
    from jevdrive import stats
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    os.system(f"{sys.executable} {REPO}/experiments/alpasim_proxy/scripts/arena.py collect --reads {a.reads} --out {out}/rows.csv.gz > /dev/null")
    Sx = pd.read_csv(ARENA / "set/set.csv.gz").set_index("id")
    Hs = Sx[Sx.split == "held"]
    R = pd.read_csv(out / "rows.csv.gz")
    ref = {"off": pd.read_csv(a.ref_off), "leash3": pd.read_csv(a.ref_leash3)}
    ref["off"] = ref["off"][ref["off"].tag == "RSG-L18-s0"]
    ref["leash3"] = ref["leash3"][ref["leash3"].tag == "leash3"]
    if a.ref_stop3:                                                    # JTB / JTN (supplement 10): ARENA-HOLD 2's stored RSG + stop3 read
        r3 = pd.read_csv(a.ref_stop3)
        ref["stop3"] = r3[r3.tag == "stop3"]
    ref = {k: v[v.id.isin(Hs.index)].drop_duplicates("id").set_index("id") for k, v in ref.items()}
    L = [f"# JT arena held-split readout ({len(Hs)} windows)", "",
         "Paired = tag minus RSG-L18-s0 under the same serve (off: the arena's stored RSG read; leash3 / stop3: ARENA-HOLD's stored RSG + leash3 / "
         "ARENA-HOLD 2's RSG + stop3 read) "
         "on the common windows without a route-fail in either; 95 % CIs, bootstrap clustered by log (jevdrive.stats). "
         "Contribution = the stratum's summed paired difference / all common windows.", "",
         "| tag @ serve | windows | mean | RSG same serve | paired diff [CI] | zero rate | RSG zero rate | zero-rate diff [CI] |",
         "|---|--:|--:|--:|--:|--:|--:|--:|"]
    per, typ, typb = [], [], []
    for tv in a.tags:
        tag, v = tv.split("@")
        X = R[(R.tag == tv) & R.id.isin(Hs.index)].drop_duplicates("id").set_index("id")
        B = ref[v]
        if not len(X):
            L.append(f"| {tv} | 0 | - | - | - | - | - | - |")
            continue
        bad = set(B.index[B.type == "route-fail"]) | set(X.index[X.type == "route-fail"])
        c = X.index.intersection(B.index).difference(bad)
        x, y, g = X.loc[c, "score"].values, B.loc[c, "score"].values, Hs.loc[c, "log"].values
        d, dz = stats.paired(x, y, groups=g), stats.paired((x == 0).astype(float), (y == 0).astype(float), groups=g)
        L.append(f"| {tv} | {len(c)} | {x.mean():.4f} | {y.mean():.4f} | {stats.fmt(d, '+.4f')} | {(x == 0).mean():.3f} | {(y == 0).mean():.3f} | "
                 f"{stats.fmt(dz, '+.4f')} |")
        for s in A.STRATA:
            cs = c[Hs.loc[c, "stratum"].values == s]
            if not len(cs):
                continue
            xs, ys = X.loc[cs, "score"].values, B.loc[cs, "score"].values
            ds = stats.paired(xs, ys, groups=Hs.loc[cs, "log"].values) if len(cs) >= 20 else None
            per.append(f"| {tv} | {s} | {len(cs)} | {ys.mean():.3f} / {(ys == 0).mean():.3f} | {xs.mean():.3f} / {(xs == 0).mean():.3f} | "
                       f"{'-' if ds is None else stats.fmt(ds, '+.3f')} | {(xs - ys).sum() / len(c):+.4f} | "
                       f"{int(((ys > 0) & (xs == 0)).sum())} / {int(((ys == 0) & (xs > 0)).sum())} |")
        for nm, Q in ((f"RSG @ {v}", B.loc[c]), (tv, X.loc[c])):
            zc = Q[Q.score == 0].type.map(A.zclass).value_counts()
            pc = Q[Q.score > 0].type.value_counts()
            typ.append(f"| {nm} | " + " | ".join(str(int(zc.get(k, 0))) for k in A.ZC) + " | " + " | ".join(str(int(pc.get(k, 0))) for k in A.PC) + " |")
        cb = c[Hs.loc[c, "stratum"].values == "braking"]                 # supplement 10: zero / partial types in the braking windows
        for nm, Q in ((f"RSG @ {v} [braking]", B.loc[cb]), (f"{tv} [braking]", X.loc[cb])):
            zc = Q[Q.score == 0].type.map(A.zclass).value_counts()
            pc = Q[Q.score > 0].type.value_counts()
            typb.append(f"| {nm} | " + " | ".join(str(int(zc.get(k, 0))) for k in A.ZC) + " | " + " | ".join(str(int(pc.get(k, 0))) for k in A.PC) + " |")
    L += ["", "By stratum (RSG and tag: mean / zero rate; new / fixed zeros against RSG under the same serve):", "",
          "| tag @ serve | stratum | windows | RSG | tag | paired diff [CI] | contribution | new / fixed zeros |", "|---|---|--:|--:|--:|--:|--:|--:|", *per,
          "", "Zero types and partial-score types on the common windows:", "",
          "| read | " + " | ".join(A.ZC) + " | " + " | ".join(A.PC) + " |", "|---|" + "--:|" * (len(A.ZC) + len(A.PC)), *typ, "",
          "The same in the braking-stratum windows only:", "",
          "| read | " + " | ".join(A.ZC) + " | " + " | ".join(A.PC) + " |", "|---|" + "--:|" * (len(A.ZC) + len(A.PC)), *typb, ""]
    (out / "arena.md").write_text("\n".join(L))
    print("\n".join(L))


def cmd_held(a):
    sys.path[:0] = [str(HERE)]
    import df
    for t in a.tags:
        r = df.cl_read(t, ["RSG-L18-s0"])
        print(json.dumps({k: v for k, v in r.items() if k != "held"} | {"held": {q: r["held"][q] for q in r["held"] if q != "pair"} if r["held"] else None,
                          "pair": r["held"]["pair"] if r["held"] else None}, default=str))
        print("\n".join(df.fmt_cl(t, r, ["RSG-L18-s0"])))


def cmd_diag(a):
    import glob
    f = sorted(glob.glob(str(DATA / "runs/route_mem" / f"train-{a.tag}" / "*" / "events.jsonl")))[-1]
    S = {}
    for line in open(f):
        e = json.loads(line)
        if e.get("kind") == "scalar":
            t_ = e["tag"][5:] if e["tag"].startswith(("loss/grad/", "loss/act/")) else e["tag"]      # rm_train logs --diag under loss/
            S.setdefault(t_, {})[e["step"]] = e["value"]
    win = lambda k: [v for s_, v in S.get(k, {}).items() if a.lo <= s_ <= a.hi]  # noqa: E731
    m = lambda k: float(np.mean(win(k))) if win(k) else float("nan")  # noqa: E731
    out = dict(events=f, last_step=max(max(v) for v in S.values()))
    out["grad"] = {k[5:]: m(k) for k in sorted(S) if k.startswith("grad/")}
    gs = {k: v for k, v in out["grad"].items() if k in ("imit", "nav_hinge", "pool_hinge", "lat", "lon")}
    tot = sum(v for v in gs.values() if np.isfinite(v))
    out["grad_share_of_sum"] = {k: v / tot for k, v in gs.items()} if tot else {}
    out["act"] = {k[4:]: m(k) for k in sorted(S) if k.startswith("act/")}
    out["loss"] = {k: m(k) for k in sorted(S) if k.startswith("loss/") and not k.endswith("_n")}
    out["dev"] = {k: S[k] for k in sorted(S) if k.startswith("dev/")}
    out["pool"] = {k: S[k] for k in sorted(S) if k.startswith(("pool_held/", "pool_fit/", "probe_"))}
    out["nan"] = [k for k, v in S.items() if any(not np.isfinite(x) for x in v.values())]
    gi, gl = out["grad"].get("imit"), out["grad"].get("lon")
    if gi and gl and np.isfinite(gi) and np.isfinite(gl) and gl > 0:
        out["arc_rule"] = dict(g_imit=gi, g_lon=gl, ratio_target=a.ratio, ratio_smoke=gl / gi,
                               w_final=float(f"{a.w * min(max(a.ratio * gi / gl, 0.5), 2.0):.2g}"), w_smoke=a.w)
    print(json.dumps(out, indent=1, default=str))
    return out


def cmd_gate(a):
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        d = cmd_diag(a)
    dv = d["dev"].get("dev/ade", {})
    s0, s1 = min(dv), max(dv)
    pr = d["pool"].get("probe_stand_r/stay1m", {})
    acts = {k: v for k, v in d["act"].items() if k.startswith("pool_")}
    g = dict(no_nan=not d["nan"], acts_positive=bool(acts) and all(v > 0 for v in acts.values()) and len(acts) >= 6,
             tv_imit_logged=np.isfinite(d["loss"].get("loss/on/imit", float("nan"))),
             dev_ade=dv[s1] <= dv[s0] + 0.06, probe_stay_rising=bool(pr) and pr[max(pr)] > pr[min(pr)])   # amendment C: + 0.06 (0.03 reported)
    if a.no_probe_gate:                                                # supplement 10 (JTB: no wait rows drawn): the probe is reported, not gated
        g.pop("probe_stay_rising")
    w = json.loads(Path(a.wjson).read_text())
    ntr = w["s_rows"] - w["held_rows"]
    res = dict(tag=a.tag, gates=g, passed=all(g.values()), dev_ade_strict=dv[s1] <= dv[s0] + 0.03, acts=acts, dev_ade={str(s0): dv[s0], str(s1): dv[s1]},
               probe_stay1m={str(k): v for k, v in pr.items()}, grad=d["grad"], grad_share=d["grad_share_of_sum"], nan=d["nan"],
               arc=d.get("arc_rule"), w_final=(d.get("arc_rule") or {}).get("w_final"), s_train_rows=ntr, steps=int(4.6 * ntr / 32 // 500 * 500))
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))
    if not res["passed"] or not res["w_final"]:
        raise SystemExit(2)


def cmd_defer(a):
    import re
    import subprocess
    import time
    cl = [str(REPO / ".venv/bin/python"), "-m", "jevdrive.cl"]
    q = subprocess.run(cl + ["queue"], capture_output=True, text=True, check=True).stdout.splitlines()
    pat = re.compile(r"^ar-read-(DD-sample|LTF-sample|starter|G0014)-c\d+$")
    rows = [x.split() for x in q if len(x.split()) > 3]
    nm = lambda r: r[3] if r[1] == "running" else r[2]  # noqa: E731  (a running row has its card column first)
    queued = [r[0] for r in rows if r[1] == "queued" and pat.match(nm(r))]
    running = [r[0] for r in rows if r[1] == "running" and pat.match(nm(r))]
    sm = [r[0] for r in rows if r[1] == "queued" and nm(r) == "ar-sum-all2"]
    print(json.dumps(dict(queued=len(queued), running=running, summary=sm)))
    if a.dry:
        return

    def spec(i):
        return json.loads(subprocess.run(cl + ["show", i], capture_output=True, text=True, check=True).stdout)["spec"]

    def resubmit(sp, after):
        f = ["submit", "--owner", sp["owner"], "--name", sp["name"], "--priority", f"{sp['priority']:g}", "--tries", str(sp["tries"]), "--no-check",
             "--log-dir", sp["log_dir"], "--vram", f"{sp['vram_gb']:g}", "--cpu", str(sp["cpu"]), "--ram", f"{sp['ram_gb']:g}",
             "--timeout-h", f"{sp['timeout_h']:g}", "--cwd", sp["cwd"], "--after", ",".join(after)]
        f += [x for k, v in sp["env"].items() for x in ("--env", f"{k}={v}")]
        out = subprocess.run(cl + f + ["--"] + sp["cmd"], capture_output=True, text=True, check=True, cwd=sp["cwd"]).stdout
        return out.split()[-1]
    specs = {i: spec(i) for i in queued + sm}
    subprocess.run(cl + ["cancel"] + queued + sm, check=True)
    time.sleep(20)                                                     # the dispatcher acts within one round
    new = [(resubmit(specs[i], a.after.split(",")), specs[i]["name"]) for i in queued]
    ns = [(resubmit(specs[i], [x for x, _ in new] + running), "ar-sum-all2") for i in sm]
    A = DATA / "runs/alpasim_proxy/arena"
    with open(A / "jobs.txt", "a") as fh:
        fh.writelines(f"{i} {n}\n" for i, n in new + ns)
    with open(A / "stages.txt", "a") as fh:
        fh.write(f"{time.strftime('%F %T')} deferred by defect_fix JT (main session's rule: JT jobs memory-blocked > 20 min while anchor reads were admitted): "
                 f"{len(new)} queued anchor reads cancelled and resubmitted with their exact specs --after the JT trainings {a.after}; "
                 f"ar-sum-all2 resubmitted as {ns[0][0] if ns else '-'} after them + {len(running)} running; ids in jobs.txt\n")
    print(json.dumps(dict(resubmitted=len(new), summary=ns)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("weights")
    p.add_argument("--pool", default="on1_JT"), p.add_argument("--k-s", type=int, default=32), p.add_argument("--k-tv", type=int, default=38)
    p.add_argument("--hold-pct", type=int, default=15), p.add_argument("--tv", default="D10"), p.add_argument("--no-wait", action="store_true")
    p.add_argument("--out", required=True)
    p = sub.add_parser("arena")
    p.add_argument("--tags", nargs="+", required=True), p.add_argument("--out", required=True)
    p.add_argument("--reads", default=str(OUT / "jt/arena"))
    p.add_argument("--ref-off", default=str(ARENA / "hold/report/ref_rows.csv.gz"), help="the arena's stored reads (RSG-L18-s0 rows)")
    p.add_argument("--ref-leash3", default=str(ARENA / "hold/report/rows.csv.gz"), help="ARENA-HOLD's reads (tag leash3 = RSG + leash3)")
    p.add_argument("--ref-stop3", default=str(ARENA / "hold/report2/rows.csv.gz"), help="ARENA-HOLD 2's reads (tag stop3 = RSG + stop3)")
    p = sub.add_parser("held")
    p.add_argument("--tags", nargs="+", required=True)
    p = sub.add_parser("diag")
    p.add_argument("--tag", required=True), p.add_argument("--lo", type=int, default=500), p.add_argument("--hi", type=int, default=1000)
    p.add_argument("--w", type=float, default=0.0025, help="the arc weight of the run")
    p.add_argument("--ratio", type=float, default=0.2, help="target arc / imit trunk-gradient ratio (JT 0.2; JTB 0.1)")
    p = sub.add_parser("gate")
    p.add_argument("--tag", required=True), p.add_argument("--lo", type=int, default=500), p.add_argument("--hi", type=int, default=1000)
    p.add_argument("--w", type=float, default=0.0025), p.add_argument("--wjson", required=True), p.add_argument("--out", required=True)
    p.add_argument("--ratio", type=float, default=0.2), p.add_argument("--no-probe-gate", action="store_true")
    p = sub.add_parser("defer")
    p.add_argument("--after", required=True), p.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    dict(weights=cmd_weights, arena=cmd_arena, held=cmd_held, diag=cmd_diag, gate=cmd_gate, defer=cmd_defer)[a.cmd](a)
