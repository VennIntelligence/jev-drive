#!/usr/bin/env python3
"""ROUND9 POOL-PILOT helpers (plans/2026-10-23-pool-design.md, section 6 = the registration; lane scripts/pcl.sh). Box, repo root, repo venv.
  pcl.py wfile                                       JT00S-rb0.npz: JT00S with every row but the live rb0 (RB0-F-s0's roll) rows at weight 0
  pcl.py argv ARM TAG [--steps N] [--on-start T] [--save-at S|none]   the arm's trainer argv, one token a line (ARM: P1 P1B P0 RB0F)
  pcl.py diff ARM                                    token diff of the arm's argv against its base's stored argv
  pcl.py ident --tag T --ref R --upto N --stop M [--allow-new] --out J -- CMD..   run CMD, stop once a scalar at step >= M exists; every
                                                     scalar of R at steps <= N (throughput/* aside) must be present and bit-identical
  pcl.py tcheck --tag T --on-start T0 --out J        post-T batch check of a stopped / finished run: pool rows per logged window as planned
  pcl.py smoke --tag T [--snap S] [--on-start T0] --out J    60-step smoke: DONE, finite, last step 60, ckpt(s), batch config, pool rows
  pcl.py det --a A --b B --out J                     two runs of one argv: every scalar and on_hash_60 equal
  pcl.py ratio --tag T --on-start T0 --train-id ID --out J   wait for the 500 steps after T, S / imitation gradient ratio vs the band;
                                                     out of band: cancel the training job, exit 2
  pcl.py post --tag T [--ref R --on-start T0] --out J  training DONE, final dev ADE <= 0.566; (informational) scalars <= T vs R
  pcl.py bs --out DIR                                behavior_stats run (the box copy, as it ran for NTV0) on the three reads
  pcl.py verdict --dir O --out P                     registered lines, paired CIs, class table, gap shares, highway read, cost -> P.{json,md}
"""
import argparse
import difflib
import glob
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from jevdrive.common import data_dir  # noqa: E402

DD = data_dir()
RUNS = DD / "runs/op_parity/runs"
O_ = DD / "runs/round9/pool_pilot"
WFILE = "runs/round9/pool_pilot/w/JT00S-rb0.npz"
TAGS = dict(P1="PCL1-F-s0", P1B="PCL1B-F-s0", P0="PCL0-L18-s0")
BASE = dict(P1="RB0L-F-s0", P1B="RB0-F-s0", P0="NTV0-L18-s0", RB0F="RB0-F-s0", P1L="RB0L-F-s0")
T_ON = dict(P1=10000, P1B=5000, P0=0, P1L=10000)
OFFREF = dict(P1="RB0L-F-s0", P1B="RB0-F-s0")             # each arm's offline counterpart (its argv without the S flags)
S_FLAGS = ["--on", "--on-k", "--on-mode", "--on-wfile", "--on-hold-pct", "--lat-onc", "--lat-soft", "--lat-side", "--lat-file", "--lat-depth",
           "--lat-soft-depth", "--lat-buf", "--on-arc", "--on-arc-rows", "--on-split"]
S_SWITCH = ["--on-lon-all", "--on-normal", "--lat-host", "--diag"]
# registered (plan section 6)
SH_HI, SH_LO = 0.74, 0.21                                 # gap-share lines
RATIO_NTV0 = (0.30, 0.80)                                 # NTV0-L18-s0's own S / imit ratio range over 500-step windows (its stored --diag log)
RATIO_BAND = (RATIO_NTV0[0] * 0.7, RATIO_NTV0[1] * 1.3)   # +-30 % of it: the stop band
RATIO_DESIGN = (0.32 * 0.7, 0.46 * 1.3)                   # the design's literal band (reported next to it)
ADE_MAX = 0.566
S_KEY = re.compile(r"^loss/(agent_ho|road_ho|on/lat_\w+_ho|on/lon_\w+_ho)$")   # the S rows' hinge / lateral / lon terms
HW_SLOW = dict(median=0.95, lt85=0.15, pts=2.0)


def dump(p, obj):
    Path(p).write_text(json.dumps(obj, indent=1, default=float) + "\n")


def run_dirs(tag):
    return sorted((DD / "runs/route_mem").glob(f"train-{tag}/*/"))


def events(d):
    return [json.loads(x) for x in (d / "events.jsonl").read_text().splitlines() if x.strip()]


def scalars(d, upto=None):
    return {(e["tag"], e["step"]): float(e["value"]) for e in events(d)
            if e.get("kind") == "scalar" and (upto is None or e["step"] <= upto) and not e["tag"].startswith("throughput/")}


def stored_argv(tag):
    return json.load(open(sorted(glob.glob(f"{DD}/runs/route_mem/train-{tag}/*/meta.json"))[-1]))["argv"]


def put(a, k, x, last=False):
    i = [j for j, t in enumerate(a) if t == k]
    assert i and (last or len(i) == 1), (k, a)
    a[i[-1] + 1] = x


def last_val(a, k):
    return a[len(a) - 1 - a[::-1].index(k) + 1]


def build_argv(arm, tag, steps=None, on_start=None, save_at=None, seed=None):
    if arm == "P1L":                                          # round9 P1L (plans/2026-10-24-p1l.md): P1's argv + NTV0-L18's schedule from T
        a, n = build_argv("P1", tag, steps, on_start, save_at), stored_argv("NTV0-L18-s0")
        return a + ["--lr2-at", last_val(a, "--on-start"), "--lr2-scale", last_val(n, "--lr-scale"), "--lr2-warmup", last_val(n, "--warmup"),
                    "--lr2-floor", last_val(n, "--lr-floor")]
    a = list(stored_argv(BASE[arm]))
    put(a, "--tag", tag)
    if arm in ("P1", "P1B"):                                  # the offline argv + NTV0's S flags (effective values), wfile -> rb0 only
        assert not any(t in a for t in S_FLAGS + S_SWITCH + ["--init"]), a
        n = stored_argv("NTV0-L18-s0")
        for k in S_FLAGS:
            a += [k, WFILE if k == "--on-wfile" else last_val(n, k)]
        for k in S_SWITCH:
            assert k in n, k
            a.append(k)
        a += ["--on-start", str(T_ON[arm] if on_start is None else on_start)]
    elif arm == "P0":                                         # NTV0's argv, only the wfile (and tag) changed
        put(a, "--on-wfile", WFILE)
    if seed is not None:                                      # seed-1 repeat of an arm (p1s1.sh): only --seed changes
        put(a, "--seed", str(seed))
    if steps is not None:
        put(a, "--steps", str(steps), last=True)
    if save_at is not None:
        if "--save-at" in a:
            i = a.index("--save-at")
            del a[i:i + 2]
        if save_at != "none":
            a += ["--save-at", save_at]
    return a


def cmd_wfile(a):
    w = np.load(DD / "runs/defect_fix/w/JT00S.npz")
    tb = np.load(DD / "runs/op_parity/cache/on1_JT/tab.npz")
    assert np.array_equal(w["names"].astype(str), tb["names"].astype(str))
    part = tb["pool"].astype(str)
    W = np.where(part == "rb0", w["w"], 0.0)
    nz = W > 0
    assert (part[nz] == "rb0").all() and (tb["mode"].astype(str)[nz] == "self").all() and int(nz.sum()) == 13793, int(nz.sum())
    assert np.array_equal(W[nz], w["w"][nz])                  # rb0 rows keep their JT00S weight (equal weights: uniform over them)
    out = DD / WFILE
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, names=w["names"], w=W, grp=w["grp"])
    rec = dict(src="runs/defect_fix/w/JT00S.npz", pool="on1_JT", part="rb0 (on1_L18RB0 = RB0-F-s0's default-serve roll, pool_design.md)",
               rows=len(W), nonzero=int(nz.sum()), nonzero_parts=sorted(set(part[nz])), weights=sorted(set(W[nz].tolist())))
    dump(out.with_suffix(".json"), rec)
    print(json.dumps(rec, indent=1))


def cmd_argv(a):
    print("\n".join(build_argv(a.arm, a.tag, a.steps, a.on_start, a.save_at, a.seed)))


def cmd_diff(a):
    b = stored_argv(BASE[a.arm])
    x = build_argv(a.arm, TAGS.get(a.arm, "PCL1L-F-s0"))
    print(" ".join(x))
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, b, x, autojunk=False).get_opcodes():
        if op != "equal":
            print(f"{op}: {BASE[a.arm]} {b[i1:i2]} -> arm {x[j1:j2]}")


def cmd_ident(a):
    assert not run_dirs(a.tag), f"{a.tag} has a run dir already"
    p = subprocess.Popen(a.cmd, start_new_session=True)
    d = None
    while p.poll() is None:                                    # stop at the first scalar at step >= stop: every scalar <= upto is flushed
        time.sleep(5)
        d = d or (run_dirs(a.tag) or [None])[-1]
        if d and (d / "events.jsonl").exists() and any(e.get("kind") == "scalar" and e["step"] >= a.stop for e in events(d)):
            break
    rc = p.poll()
    if rc is None:
        os.killpg(p.pid, signal.SIGTERM)
        try:
            p.wait(60)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL), p.wait()
    d = d or (run_dirs(a.tag) or [None])[-1]
    assert d, "no run dir"
    rec = compare(d, run_dirs(a.ref)[-1], a.upto, a.allow_new)
    rec |= dict(tag=a.tag, ref=a.ref, run=str(d), trainer_rc_before_stop=rc)
    dump(a.out, rec)
    print(json.dumps(rec, indent=1))
    sys.exit(0 if rec["bit_identical"] else 2)


def compare(d, r, upto, allow_new):
    x, y = scalars(d, upto), scalars(r, upto)
    keys = sorted(set(x) & set(y))
    diff = {f"{k[0]}@{k[1]}": abs(x[k] - y[k]) for k in keys}
    only_new = sorted({k[0] for k in set(x) - set(y)})
    only_ref = sorted(f"{k[0]}@{k[1]}" for k in set(y) - set(x))
    steps = sorted({k[1] for k in y})
    rec = dict(ref_run=str(r), upto=upto, steps=steps, scalars_compared=len(keys), identical=int(sum(v == 0 for v in diff.values())),
               max_abs_diff=max(diff.values()) if diff else None, worst=sorted(diff.items(), key=lambda kv: -kv[1])[:8],
               only_ref=only_ref, only_new_tags=only_new, allow_new=allow_new)
    rec["bit_identical"] = bool(keys and rec["identical"] == len(keys) and not only_ref and (allow_new or not only_new))
    return rec


def pool_rows(d):
    return {e["step"]: float(e["value"]) for e in events(d) if e.get("kind") == "scalar" and e["tag"] == "loss/on/batch_pool_rows"}


def want_rows(steps, T, k=32):
    """Mean pool rows per batch over each logged window (prev, s]: draw n (0-based) of step n + 1 carries k rows iff n >= T."""
    out, prev = {}, 0
    for s in steps:
        out[s] = k * sum(1 for n in range(prev, s) if n >= T) / (s - prev)
        prev = s
    return out


def s_terms(d, after):
    """S loss terms logged after step `after`: present, finite, and the hinge-only mass nonzero somewhere."""
    v = [e for e in events(d) if e.get("kind") == "scalar" and e["step"] > after and S_KEY.match(e["tag"])]
    tags = sorted({e["tag"] for e in v})
    return dict(tags=tags, finite=bool(v) and all(np.isfinite(e["value"]) for e in v), nonzero=any(e["value"] != 0 for e in v))


def cmd_tcheck(a):
    d = run_dirs(a.tag)[-1]
    got = pool_rows(d)
    want = want_rows(sorted(got), a.on_start)
    st = s_terms(d, a.on_start)
    pre = scalars(d, a.on_start)
    pre_s = sorted(k for k, v in pre.items() if S_KEY.match(k[0]) and v != 0)    # before T no S term may carry mass
    ok = bool(got) and all(abs(got[s] - want[s]) < 1e-9 for s in got) and st["finite"] and st["nonzero"] and not pre_s
    rec = dict(tag=a.tag, run=str(d), on_start=a.on_start, pool_rows=got, want=want, s_terms_after=st, s_terms_before=[f"{k[0]}@{k[1]}" for k in pre_s],
               pass_=ok)
    dump(a.out, rec)
    print(json.dumps(rec, indent=1))
    sys.exit(0 if ok else 2)


def cmd_smoke(a):
    d = run_dirs(a.tag)[-1]
    log = (d / "log.txt").read_text()
    b = re.search(r"per batch (\d+) normal, (\d+) yaw-rate, (\d+) on-policy \(mode (\w+)", log)
    vals = np.array([e["value"] for e in events(d) if e.get("kind") == "scalar"], float)
    steps = [int(m) for m in re.findall(r"step (\d+): ", log)]
    ck = dict(done=(d / "DONE").exists(), scalars=int(len(vals)), finite=bool(len(vals) and np.isfinite(vals).all()),
              last_logged_step=max(steps) if steps else None, final=(RUNS / a.tag / "ckpt-final.pt").exists(),
              batch=list(b.groups()) if b else None, batch_want=["83", "13", "32", "mixed"])
    ok = ck["done"] and ck["finite"] and ck["last_logged_step"] == 60 and ck["final"] and ck["batch"] == ck["batch_want"]
    if a.snap:
        ck["snapshot"] = (RUNS / a.snap / "ckpt-final.pt").exists()
        ok = ok and ck["snapshot"]
    if a.on_start is not None:
        got = pool_rows(d)
        ck["pool_rows"], ck["want"] = got, want_rows(sorted(got), a.on_start)
        ck["s_terms_after"] = s_terms(d, a.on_start)
        ok = ok and bool(got) and all(abs(got[s] - ck["want"][s]) < 1e-9 for s in got) and ck["s_terms_after"]["finite"]
    else:
        ck["s_terms_after"] = s_terms(d, 0)
        ok = ok and ck["s_terms_after"]["finite"] and ck["s_terms_after"]["nonzero"]
    rec = dict(tag=a.tag, run=str(d), checks=ck, pass_=bool(ok))
    dump(a.out, rec)
    print(json.dumps(rec, indent=1))
    sys.exit(0 if ok else 2)


def cmd_det(a):
    da, db = run_dirs(a.a)[-1], run_dirs(a.b)[-1]
    rec = compare(da, db, None, False)
    h = [json.loads((x / "DONE").read_text()).get("on_hash_60") for x in (da, db)]
    rec |= dict(a=a.a, b=a.b, on_hash_60=dict(a=h[0], b=h[1], equal=h[0] == h[1] and h[0] is not None))
    rec["bit_identical"] = rec["bit_identical"] and rec["on_hash_60"]["equal"]
    dump(a.out, rec)
    print(json.dumps(rec, indent=1))
    sys.exit(0 if rec["bit_identical"] else 2)


def grad_ratio(d, lo, hi):
    g = {}
    for e in events(d):
        if e.get("kind") == "scalar" and e["tag"] in ("loss/grad/pool", "loss/grad/imit") and lo < e["step"] <= hi:
            g.setdefault(e["step"], {})[e["tag"]] = float(e["value"])
    s = sorted(k for k, v in g.items() if len(v) == 2)
    P, I = np.array([g[k]["loss/grad/pool"] for k in s]), np.array([g[k]["loss/grad/imit"] for k in s])
    return dict(steps=s, pool=P.tolist(), imit=I.tolist(), per_step=(P / I).tolist(), ratio=float(P.mean() / I.mean()) if len(s) else None)


def cmd_ratio(a):
    from jevdrive.cl import pool as P
    hi = a.on_start + 500
    while True:                                               # in-queue gate: waits on the training's own log (no agent loop)
        ds = run_dirs(a.tag)
        if ds and (ds[-1] / "events.jsonl").exists():
            st = [e["step"] for e in events(ds[-1]) if e.get("kind") == "scalar" and e["tag"] == "loss/grad/imit"]
            if st and max(st) >= hi:
                break
        j = P.snapshot()[0]["jobs"].get(a.train_id, {})
        if j.get("state") in ("done", "failed", "cancelled"):
            if ds and (ds[-1] / "events.jsonl").exists():
                break
            sys.exit(f"training {a.train_id} ended ({j.get('state')}) without a log")
        time.sleep(60)
    r = grad_ratio(ds[-1], a.on_start, hi)
    ok = r["ratio"] is not None and len(r["steps"]) == 5 and RATIO_BAND[0] <= r["ratio"] <= RATIO_BAND[1]
    rec = dict(tag=a.tag, on_start=a.on_start, window=[a.on_start, hi], band=RATIO_BAND, band_from_ntv0_measured=RATIO_NTV0,
               design_band=RATIO_DESIGN, in_design_band=r["ratio"] is not None and RATIO_DESIGN[0] <= r["ratio"] <= RATIO_DESIGN[1], **r, pass_=bool(ok))
    dump(a.out, rec)
    print(json.dumps(rec, indent=1))
    if not ok:
        subprocess.run([sys.executable, "-m", "jevdrive.cl", "cancel", a.train_id])
        sys.exit(2)


def dev_ade(tag, step=None):
    if step is None:
        f = RUNS / tag / "route_mem.json"
        return json.loads(f.read_text()).get("dev_ade") if f.exists() else None
    v = [e["value"] for e in events(run_dirs(tag)[-1]) if e.get("kind") == "scalar" and e["tag"] == "dev/ade" and e["step"] == step]
    return float(v[-1]) if v else None


def cmd_post(a):
    d = run_dirs(a.tag)[-1]
    ade = dev_ade(a.tag)
    rec = dict(tag=a.tag, done=(d / "DONE").exists(), final=(RUNS / a.tag / "ckpt-final.pt").exists(), dev_ade=ade, ade_max=ADE_MAX)
    if a.ref:                                                 # informational: the whole pre-T segment vs the offline counterpart's log
        rec["pre_T_vs_ref"] = compare(d, run_dirs(a.ref)[-1], a.on_start, True) | dict(ref=a.ref)
    ok = rec["done"] and rec["final"] and ade is not None and ade <= ADE_MAX
    rec["pass_"] = bool(ok)
    dump(a.out, rec)
    print(json.dumps(rec, indent=1))
    sys.exit(0 if ok else 2)


def cmd_bs(a):
    """behavior_stats.py as it ran for the stored NTV0 / RB0-F reads (the box copy under runs/pai_av/behavior_stats/code), with the arms
    added to its READS at run time; `run` only (the highway read itself is computed in verdict from its steps / scene files)."""
    import importlib.util
    sp = importlib.util.spec_from_file_location("behavior_stats", DD / "runs/pai_av/behavior_stats/code/behavior_stats.py")
    m = importlib.util.module_from_spec(sp)
    sys.modules["behavior_stats"] = m
    sp.loader.exec_module(m)
    for t in TAGS.values():
        m.READS[t] = (m.TAX / f"{t}-281ag.pkl", m.FACTS / f"{t}-281ag.csv", f"{t}-ag")
    m.cmd_run(argparse.Namespace(out=a.out, workers=a.workers, limit=0, reads=",".join(TAGS.values())))


# ---------------------------------------------------------------- verdict
REFS = {"NTV0-L18-s0": "NTV0-L18-s0", "RB0-F-s0": "RB0-F-s0", "RB0L-F-s0": "RB0L-F-s0", "NTV0NS-L18-s0": "NTV0NS-L18-s0"}
SV2_LANE = {"RB0L-F-s0": "round9/rb0_long", "NTV0NS-L18-s0": "round9/ntv0_nos"}


def new_scores(o):
    import pandas as pd
    P = pd.read_csv(DD / "runs/alpasim/scorer_v2/report/per_scene.csv", header=[0, 1], index_col=0)["new"]
    out = {t: P[f"{t}-ag"] for t in ("NTV0-L18-s0", "RB0-F-s0")}
    for t, lane in SV2_LANE.items():
        x = pd.read_csv(DD / "runs" / lane / "sv2/new/rollouts.csv")
        out[t] = x[x.read == f"{t}-ag"].drop_duplicates("scene", keep="last").set_index("scene").score
    f = o / "sv2/new/rollouts.csv"
    if f.exists():
        x = pd.read_csv(f)
        for t in TAGS.values():
            y = x[x.read == f"{t}-ag"]
            if len(y):
                out[t] = y.drop_duplicates("scene", keep="last").set_index("scene").score
    return out


def f3(d):
    return "%+.3f [%+.3f, %+.3f]" % (d["mean"], d["lo"], d["hi"]) if d else "-"


def line_of(arm, g, ci_lo):
    if g >= SH_HI and ci_lo is not None and ci_lo > 0:
        return "carries the gain"
    if g <= SH_LO:
        return "does not carry it"
    return "undecided" + (" (share >= 74 % but the CI vs its offline counterpart reaches 0)" if g >= SH_HI else "")


def highway(o, new):
    import pandas as pd
    BS = DD / "runs/pai_av/behavior_stats/out"
    MP = pd.read_csv(BS / "scenes_map.csv").set_index("scene")
    hw_sc = MP.index[MP.scene_cls == "highway"]
    ramp_sc = list(MP.index[MP.scene_cls == "ramp"])
    rows, per = [], {}
    for lab in list(TAGS.values()) + ["NTV0-L18-s0", "RB0-F-s0"]:
        src = (o / "bs") if lab in TAGS.values() else BS
        f = src / f"steps_{lab}.parquet"
        if not f.exists():
            continue
        T = pd.read_parquet(f)
        T = T[(T.free_e) & (T.free_h == True) & (T.vh >= 3) & (T.t >= 2.0)]  # noqa: E712  (= behavior_stats Q1)
        x = T[T.cls == "highway"]
        q = x.v / x.vh
        FA = pd.read_csv(DD / "runs/pai_fix/facts" / f"{lab}-281ag.csv").set_index("scene")
        sc = pd.read_csv(src / f"scene_{lab}.csv").set_index("scene")
        nw = new.get(lab)
        z = FA[["fclass"]].join(sc[["cls_end"]], how="left")
        z["new"] = nw.reindex(z.index) if nw is not None else np.nan
        z["new"] = z.new.fillna(FA.score)
        pts = float((1 - z.new)[z.fclass.isin(["slow", "rear-hit"]) & (z.cls_end == "highway")].sum())
        r = dict(read=lab, steps=len(x), median=float(q.median()), p10=float(q.quantile(0.1)), lt85=float((q < 0.85).mean()),
                 gt110=float((q > 1.1).mean()), dv=float((x.v - x.vh).mean()), slow_rear_pts=pts)
        r["slow"] = bool(r["median"] < HW_SLOW["median"] or r["lt85"] >= HW_SLOW["lt85"] or pts >= HW_SLOW["pts"])
        rows.append(r)
        ps = x.assign(q=q).groupby("scene").agg(n=("q", "size"), med=("q", "median")).reindex(hw_sc)
        per[lab] = ps
        rp = T[T.scene.isin(ramp_sc)]
        r["ramp"] = dict(scenes=ramp_sc, steps=len(rp), median=float((rp.v / rp.vh).median()) if len(rp) else None)
    return rows, per


def cmd_verdict(a):
    import pandas as pd
    from jevdrive import stats
    sys.path.insert(0, str(REPO / "experiments/pai_fix/scripts"))
    import pai_fix as PF
    o = Path(a.dir)
    labs = list(TAGS.values()) + list(REFS)
    OLD = {t: pd.read_csv(DD / "runs/pai_fix/facts" / f"{t}-281ag.csv").set_index("scene") for t in labs}
    NEW = new_scores(o)
    rpc = set().union(*[set(F.index[F.rpc.fillna(False).astype(bool)]) for F in OLD.values()])
    have = [t for t in labs if t in NEW]
    common = sorted(set.intersection(*[set(OLD[t].index[OLD[t].score.notna()]) for t in labs])
                    & set.intersection(*[set(NEW[t].dropna().index) for t in have]) - rpc)
    n = len(common)
    so = {t: OLD[t].score.reindex(common).to_numpy(float) for t in labs}
    sn = {t: NEW[t].reindex(common).to_numpy(float) for t in have}
    res = dict(scenes=n, reads={}, pairs={}, lines={})
    for t in labs:
        m = PF.metrics(OLD[t].reindex(common))
        res["reads"][t] = dict(old=m, new_mean=float(sn[t].mean()) if t in sn else None, new_zeros=int((sn[t] == 0).sum()) if t in sn else None,
                               dev_ade=dev_ade(t))
    pairs = [(x, y) for x in TAGS.values() for y in REFS] + [(TAGS["P1"], TAGS["P1B"]), (TAGS["P1"], TAGS["P0"])]
    for x, y in pairs:
        k = f"{x}:{y}"
        res["pairs"][k] = dict(old=stats.paired(so[x], so[y]), new=stats.paired(sn[x], sn[y]) if x in sn and y in sn else None)
        cl = {}
        for c in PF.ZCLS + PF.PCLS:
            u, v = (OLD[x].reindex(common).fclass == c).to_numpy(float), (OLD[y].reindex(common).fclass == c).to_numpy(float)
            q = stats.paired(u, v)
            cl[c] = dict(a=int(u.sum()), b=int(v.sum()), diff=n * q["mean"], lo=n * q["lo"], hi=n * q["hi"])
        u, v = (so[x] == 0).astype(float), (so[y] == 0).astype(float)
        q = stats.paired(u, v)
        cl["zeros"] = dict(a=int(u.sum()), b=int(v.sum()), diff=n * q["mean"], lo=n * q["lo"], hi=n * q["hi"])
        res["pairs"][k]["classes"] = cl
    for sc_, S in (("new", sn), ("old", so)):
        if not all(t in S for t in ("RB0-F-s0", "NTV0-L18-s0")):
            continue
        b, nt = S["RB0-F-s0"].mean(), S["NTV0-L18-s0"].mean()
        res["lines"][sc_] = dict(rb0f=float(b), ntv0=float(nt), line_hi=float(b + SH_HI * (nt - b)), line_lo=float(b + SH_LO * (nt - b)), arms={})
        for arm, t in TAGS.items():
            if t not in S:
                continue
            g = float((S[t].mean() - b) / (nt - b))
            off = OFFREF.get(arm)
            ci = res["pairs"][f"{t}:{off}"][sc_] if off else None
            res["lines"][sc_]["arms"][arm] = dict(tag=t, mean=float(S[t].mean()), share=g, vs_offline=ci,
                                                  line=line_of(arm, g, ci["lo"] if ci else None) if off else
                                                  ("at NTV0 (>= 74 %)" if g >= SH_HI else "at RB0-F (<= 21 %)" if g <= SH_LO else "between"))
    hw, per = highway(o, NEW)
    res["highway"] = hw
    res["ratio"] = {arm: json.loads((o / f"results/ratio_{arm}.json").read_text()) for arm in TAGS if (o / f"results/ratio_{arm}.json").exists()}
    res["post"] = {arm: json.loads((o / f"results/post_{arm}.json").read_text()) for arm in TAGS if (o / f"results/post_{arm}.json").exists()}
    res["ade"] = {"PCL1-F-s0 @ 10000": dev_ade(TAGS["P1"], 10000), "PCL1-F-s0 @ 19500": dev_ade(TAGS["P1"], 19500),
                  "PCL1B-F-s0 @ 5000": dev_ade(TAGS["P1B"], 5000), "PCL1B-F-s0 @ 10000": dev_ade(TAGS["P1B"], 10000),
                  "PCL0-L18-s0 (RB0-F + 9 500)": dev_ade(TAGS["P0"]), "RB0L-F-s0 @ 10000": dev_ade("RB0L-F-s0", 10000),
                  "RB0L-F-s0 @ 19500": dev_ade("RB0L-F-s0", 19500), "RB0-F-s0": dev_ade("RB0-F-s0"), "NTV0-L18-s0": dev_ade("NTV0-L18-s0"),
                  "NTV0NS-L18-s0": dev_ade("NTV0NS-L18-s0")}
    res["cost"] = cost(o)
    L = ["# POOL-PILOT verdict (PAI 281, ACC + GAP, Harmonizer OFF)", "",
         f"{n} scenes scored by all seven reads with no driver RPC error in any of them (both scorers on the same set).", "",
         "## Reads", "", "| read | old mean | new mean | zeros old / new | offroad | drift + side | route-turn | overshoot | AF-front | rear-hit p | slow p | launched | dev ADE |",
         "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for t, r in res["reads"].items():
        m = r["old"]
        L.append(f"| {t} | {m['mean']:.3f} | {r['new_mean']:.3f} | {m['zeros']} / {r['new_zeros']} | {m['z_offroad']} | {m['z_zero-other']} + {m['z_side']} | "
                 f"{m['z_route-turn']} | {m['z_overshoot']} | {m['z_AF-front']} | {m['p_rear-hit']} | {m['p_slow']} | {m['launch']} | "
                 + (f"{r['dev_ade']:.4f} |" if r["dev_ade"] is not None else "- |"))
    L += ["", "## Paired differences (95 % bootstrap CI, scenes as units)", "", "| pair | old scorer | new scorer | zeros (old) |", "|:--|--:|--:|--:|"]
    for k, p in res["pairs"].items():
        z = p["classes"]["zeros"]
        L.append(f"| {k.replace(':', ' - ')} | {f3(p['old'])} | {f3(p['new'])} | {z['diff']:+.0f} [{z['lo']:+.0f}, {z['hi']:+.0f}] |")
    L += ["", "Zero / partial classes vs NTV0-L18-s0 and NTV0NS-L18-s0 (count difference, 95 % CI):", "",
          "| arm - ref | " + " | ".join(PF.ZCLS + PF.PCLS) + " |", "|:--|" + "--:|" * len(PF.ZCLS + PF.PCLS)]
    for x in TAGS.values():
        for y in ("NTV0-L18-s0", "NTV0NS-L18-s0", "RB0-F-s0"):
            c = res["pairs"][f"{x}:{y}"]["classes"]
            L.append(f"| {x} - {y} | " + " | ".join(f"{c[q]['diff']:+.0f} [{c[q]['lo']:+.0f}, {c[q]['hi']:+.0f}]" for q in PF.ZCLS + PF.PCLS) + " |")
    L += ["", "## Registered lines (gap share g = (arm - RB0-F) / (NTV0-s0 - RB0-F); >= 74 % with the CI vs the offline counterpart above 0 / <= 21 %)", ""]
    for sc_, q in res["lines"].items():
        L.append(f"- {sc_} scorer: RB0-F {q['rb0f']:.3f}, NTV0-s0 {q['ntv0']:.3f} -> lines {q['line_hi']:.3f} / {q['line_lo']:.3f}")
        for arm, r in q["arms"].items():
            L.append(f"  - {arm} {r['tag']}: {r['mean']:.3f}, g = **{r['share']:.0%}**"
                     + (f", vs {OFFREF[arm]} {f3(r['vs_offline'])}" if r["vs_offline"] else "") + f" -> **{r['line']}**")
    L += ["", "## Highway (behavior_stats Q1 rules; free-road v_ego / v_human on highway stations; slow + rear-hit points by end class, new scorer)", "",
          "| read | steps | median | p10 | < 0.85 | > 1.10 | mean dv m/s | slow + rear pts | registered line |", "|:--|--:|--:|--:|--:|--:|--:|--:|:--|"]
    for r in hw:
        L.append(f"| {r['read']} | {r['steps']} | {r['median']:.2f} | {r['p10']:.2f} | {r['lt85']:.0%} | {r['gt110']:.0%} | {r['dv']:+.2f} | {r['slow_rear_pts']:.2f} | "
                 f"{'highway slowness present' if r['slow'] else 'not slow on highway'} |")
    L += ["", "Highway-majority scenes with a free-road median ratio < 0.90 (scene: median / stations):", ""]
    for lab, ps in per.items():
        bad = ps[ps.med < 0.90]
        L.append(f"- {lab}: {len(bad)} of {int(ps.n.notna().sum())} scenes with free-road steps" + (": " + ", ".join(
            f"{s[7:15]} {r.med:.2f} / {int(r.n)}" for s, r in bad.iterrows()) if len(bad) else ""))
    L.append("- ramp (listed only): " + "; ".join(f"{r['read']} {r['ramp']['steps']} steps, median {r['ramp']['median']}" for r in hw))
    L += ["", "## Gates in the run", ""]
    for arm, r in res["ratio"].items():
        L.append(f"- S / imit gradient ratio, 500 steps after T ({arm}): {r['ratio']:.3f} (band {RATIO_BAND[0]:.2f}-{RATIO_BAND[1]:.2f}; design's literal "
                 f"{RATIO_DESIGN[0]:.3f}-{RATIO_DESIGN[1]:.3f}: {'in' if r['in_design_band'] else 'out'}) -> {'pass' if r['pass_'] else 'FAIL'}")
    for arm, r in res["post"].items():
        p = r.get("pre_T_vs_ref")
        L.append(f"- post ({arm}): final dev ADE {r['dev_ade']:.4f} (<= {ADE_MAX})" + (
            f"; whole pre-T segment vs {p['ref']}: {p['identical']} / {p['scalars_compared']} scalars bit-identical, steps <= {p['upto']}" if p else ""))
    L += ["", "| navtrain dev ADE | m |", "|:--|--:|"] + [f"| {k} | {v:.4f} |" if v is not None else f"| {k} | - |" for k, v in res["ade"].items()]
    L += ["", "## Cost", "", f"GPU jobs {res['cost']['gpu_job_h']:.1f} job-hours (training {res['cost']['train_h']:.2f}, reads {res['cost']['read_h']:.2f}, "
          f"gates / smokes {res['cost']['gate_h']:.2f}); wall {res['cost']['wall_h']:.1f} h from the first start to the last end."]
    Path(f"{a.out}.md").write_text("\n".join(L) + "\n")
    dump(f"{a.out}.json", res | dict(highway_scenes={k: v.dropna().reset_index().to_dict("records") for k, v in per.items()}))
    print("\n".join(L))


def cost(o):
    from jevdrive.cl import pool as P
    J = P.snapshot()[0]["jobs"]
    ids = [ln.split()[0] for ln in (o / "jobs.txt").read_text().splitlines() if ln.split()]
    tr = rd = gt = 0.0
    t0s, t1s = [], []
    for i in ids:
        j = J.get(i)
        if not j or not j.get("t0") or j["spec"]["vram_gb"] <= 1:
            continue
        t1 = j.get("t1") or time.time()
        h = (t1 - j["t0"]) / 3600
        t0s.append(j["t0"]), t1s.append(t1)
        nm = j["spec"]["name"]
        if nm.startswith("pai-"):
            rd += h
        elif "-train-" in nm:
            tr += h
        else:
            gt += h
    return dict(train_h=tr, read_h=rd, gate_h=gt, gpu_job_h=tr + rd + gt, wall_h=(max(t1s) - min(t0s)) / 3600 if t0s else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = ap.add_subparsers(dest="sub", required=True)
    s.add_parser("wfile")
    p = s.add_parser("argv")
    p.add_argument("arm", choices=("P1", "P1B", "P0", "RB0F", "P1L")), p.add_argument("tag")
    p.add_argument("--steps", type=int), p.add_argument("--on-start", type=int), p.add_argument("--save-at"), p.add_argument("--seed", type=int)
    p = s.add_parser("diff")
    p.add_argument("arm", choices=("P1", "P1B", "P0", "P1L"))
    p = s.add_parser("ident")
    p.add_argument("--tag", required=True), p.add_argument("--ref", required=True), p.add_argument("--upto", type=int, required=True)
    p.add_argument("--stop", type=int, required=True), p.add_argument("--allow-new", action="store_true"), p.add_argument("--out", required=True)
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    p = s.add_parser("tcheck")
    p.add_argument("--tag", required=True), p.add_argument("--on-start", type=int, required=True), p.add_argument("--out", required=True)
    p = s.add_parser("smoke")
    p.add_argument("--tag", required=True), p.add_argument("--snap", default=""), p.add_argument("--on-start", type=int), p.add_argument("--out", required=True)
    p = s.add_parser("det")
    p.add_argument("--a", required=True), p.add_argument("--b", required=True), p.add_argument("--out", required=True)
    p = s.add_parser("ratio")
    p.add_argument("--tag", required=True), p.add_argument("--on-start", type=int, required=True), p.add_argument("--train-id", required=True)
    p.add_argument("--out", required=True)
    p = s.add_parser("post")
    p.add_argument("--tag", required=True), p.add_argument("--ref", default=""), p.add_argument("--on-start", type=int, default=0)
    p.add_argument("--out", required=True)
    p = s.add_parser("bs")
    p.add_argument("--out", required=True), p.add_argument("--workers", type=int, default=16)
    p = s.add_parser("verdict")
    p.add_argument("--dir", required=True), p.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.sub == "ident":
        a.cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
        assert a.cmd, "ident needs the trainer command after --"
    dict(wfile=cmd_wfile, argv=cmd_argv, diff=cmd_diff, ident=cmd_ident, tcheck=cmd_tcheck, smoke=cmd_smoke, det=cmd_det, ratio=cmd_ratio,
         post=cmd_post, bs=cmd_bs, verdict=cmd_verdict)[a.sub](a)


if __name__ == "__main__":
    main()
