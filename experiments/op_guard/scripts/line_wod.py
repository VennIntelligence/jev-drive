"""Guard line `wod`: WOD-E2E val, unified interface (true roof height 1.81 m, real 10 Hz frames fed twice, 10 s real warm-up, no x1.06).

  RFS          the 479 val rater frames through the serving harness scripts/wod_zeroshot_openpilot.py (shipped Cinque, or --onnx for a
               candidate), WOD's own scorer (jevdrive.waymo.rater_feedback_score, per-category RFS), paired vs shipped with a segment
               bootstrap as experiments/op_adapt_h/scripts/h_wod.py. Rule "no drop" = the upper 95% bound of the paired delta >= 0.
  false start  the op_adapt_L / op_adapt_H standstill metric (op_adapt_l.row_metrics: on `stay` rows, plan displacement at 4 s >= 3 m)
               on its own row set: the 4 116 `stay` rows of WOD-E2E val (op_adapt_L wodval domain, 144 segments) through the training
               port (portcand.py: the candidate's ONNX weights in the port), paired cluster bootstrap by segment. Rule <= shipped + 2 pp.
               The 22 `stay` frames among the 479 rater frames (serving path) are reported next to it, without a pass (1 frame = 4.5 pp).
subset = full here (479 rater frames + 4 116 stay rows, ~8 min cold per candidate).

Caches (used unless --force; listed in provenance.cache): serving predictions of the same harness already on the box
(preds/op_cinque for shipped, preds/op_cinque_Oit_dw3-s0 for it_dw3-s0: valid when all 479 exist and are newer than the ONNX); the
shipped port readout of op_adapt_L ($L/readout/O/eval/wodval.npz, same port and rows). --force recomputes both into guard-owned dirs.

  DATA_DIR=... $DATA_DIR/envs/op-train/bin/python experiments/op_guard/scripts/line_wod.py --candidate it_dw3-s0 --gpu 3 --cpus 60-79
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import guardlib as G  # noqa: E402
import portcand  # noqa: E402,F401  (sys.path for the port modules)

LINE = "wod"
KNOWN_PREDS = {"shipped": "op_cinque", "it_dw3-s0": "op_cinque_Oit_dw3-s0"}     # earlier serving runs of this exact harness
SHIPPED_ONNX = Path.home() / "data/models/openpilot/cinque.ort.onnx"
B = 2000


def cpu_list(s: str) -> list[int]:
    out = []
    for part in filter(None, s.split(",")):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def rater():
    from jevdrive import wod_zeroshot as Z
    s = Z.load_sets()["rater"]
    return s, s["name"].astype(str)


# ---------------------------------------------------------------- serving predictions
def serving_preds(c: dict, a, prov: dict) -> Path:
    from jevdrive import wod_zeroshot as Z
    _, names = rater()
    onnx = c["onnx"]
    known = KNOWN_PREDS.get(c["name"])
    if known and not a.force:
        d = Z.root("preds", known)
        files = [d / f"{n}.npz" for n in names]
        if all(f.exists() for f in files) and (onnx is None or min(f.stat().st_mtime for f in files) >= Path(onnx).stat().st_mtime):
            prov["cache"].append(f"serving preds {d} (same harness, 479/479, newer than the ONNX)")
            return d
    tag = f"guard-{c['name']}"
    d = Z.root("preds", f"op_cinque_{tag}")
    if a.force and d.exists():
        shutil.rmtree(d)
    cpus = cpu_list(a.cpus) or list(range(os.cpu_count() or 8))
    py = G.data_dir() / "envs" / "openpilot" / "bin" / "python"
    cmd = [str(py), str(G.REPO / "scripts" / "wod_zeroshot_openpilot.py"), "--set", "rater", "--workers", str(max(2, len(cpus) - 2)),
           "--onnx", str(onnx or SHIPPED_ONNX), "--tag", tag]
    if a.cpus:
        cmd = ["taskset", "-c", a.cpus] + cmd
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu))
    print("+", " ".join(cmd), flush=True)
    t = time.time()
    subprocess.run(cmd, check=True, env=env, cwd=G.REPO)
    prov["serving_s"] = round(time.time() - t, 1)
    prov["serving_cmd"] = " ".join(cmd)
    return d


def rater_arrays(c: dict, a, out: Path, prov: dict) -> dict:
    """Per-frame RFS and 4 s rear-axle displacement on the rater frames -> out/wod_rater.npz (resumable)."""
    p = out / "wod_rater.npz"
    if p.exists() and not a.force:
        prov["cache"].append(f"{p} (this line's own earlier result)")
        return dict(np.load(p, allow_pickle=True))
    from jevdrive import waymo as W
    s, names = rater()
    d = serving_preds(c, a, prov)
    wod = np.stack([np.load(d / f"{n}.npz")["wod"] for n in names]).astype(np.float64)[..., :2]
    f = np.asarray(W.rater_feedback_score(wod, s["traj"].astype(np.float64), s["scores"].astype(np.float64), W.init_speed(s["past"])), float)
    z = dict(names=names, rfs=f, disp4=np.linalg.norm(wod[:, 15], axis=-1), pred_dir=str(d))
    np.savez(p, **z)
    return z


# ---------------------------------------------------------------- port: WOD val stay rows
def stay_arrays(c: dict, a, out: Path, prov: dict) -> dict:
    p = out / "wod_stay.npz"
    if p.exists() and not a.force:
        prov["cache"].append(f"{p} (this line's own earlier result)")
        return dict(np.load(p, allow_pickle=True))
    import torch
    from experiments.op_adapt_l.lib import op_adapt_l as L
    D = L.Data(("wodval",))
    tab = D.tab["wodval"]
    rows = D.rows("wodval", "val", need_future=True)
    srows = rows[np.asarray(tab["s_stay"])[rows]]
    plan = None
    ro = L.lroot("readout", "O", "eval") / "wodval.npz"
    if c["onnx"] is None and not a.force and ro.exists():
        z = np.load(ro)
        pos = {r: i for i, r in enumerate(z["rows"])}
        if all(r in pos for r in srows):
            plan = z["plan"][[pos[r] for r in srows]]
            prov["cache"].append(f"shipped port plans {ro} (op_adapt_L readout of the same port and rows)")
    if plan is None:
        dev = torch.device("cuda")
        m, info = portcand.load(c["onnx"], dev)
        prov["port"] = info
        t = time.time()
        with torch.no_grad():
            plan = L.fwd_rows(m, D, "wodval", srows, dev, intent=False)["plan"]
        prov["port_s"] = round(time.time() - t, 1)
    fs = L.row_metrics(plan, tab, srows, D.cam["wodval"])["false_start"]
    z = dict(rows=srows, false_start=fs, groups=np.asarray(tab["seq"])[srows], plan=plan.astype(np.float16))
    np.savez(p, **z)
    return z


def arrays(name: str, a, mode: str, prov: dict) -> tuple[dict, dict]:
    c = G.resolve(name)
    out = G.run_dir(c["name"], mode)
    return rater_arrays(c, a, out, prov), stay_arrays(c, a, out, prov)


def rfs_paired(fa, fb, cl, seq):
    """RFS(a) - RFS(b) with the segment bootstrap of h_wod.py (same draws for both)."""
    import pandas as pd
    from jevdrive import waymo as W
    codes, uniq = pd.factorize(pd.Series(seq))
    idx = [np.flatnonzero(codes == k) for k in range(len(uniq))]
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(B):
        i = np.concatenate([idx[k] for k in rng.integers(len(uniq), size=len(uniq))])
        bs.append(W.rfs_by_cluster(fa[i], cl[i])[0] - W.rfs_by_cluster(fb[i], cl[i])[0])
    return W.rfs_by_cluster(fa, cl)[0], W.rfs_by_cluster(fb, cl)[0], float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    a = G.line_args(LINE, __doc__.splitlines()[0]).parse_args()
    c = G.resolve(a.candidate)
    out = G.run_dir(c["name"], a.mode)
    if G.done(out, LINE) and not a.force:
        print(f"{out}/lines/{LINE}.json exists (use --force)")
        return
    t0 = time.time()
    rule = G.LINES[LINE][2]
    prov = {"cache": [], "interface": "WOD spec: true roof height 1.81 m, real 10 Hz frames fed twice, 10 s warm-up, open loop, no x1.06",
            "subset": "subset = full (479 rater frames + 4116 WOD-val stay rows)"}
    try:
        ra, sa = arrays(c["name"], a, a.mode, prov)
        pr = {"cache": []}
        rb, sb = (ra, sa) if c["name"] == G.SHIPPED else arrays(G.SHIPPED, argparse_like(a), a.mode, pr)
        prov["cache"] += [f"shipped: {x}" for x in pr["cache"]]
        s, names = rater()
        assert (ra["names"].astype(str) == names).all() and (rb["names"].astype(str) == names).all()
        cl = s["cluster"].astype(str)
        seq = np.array([n.rsplit("-", 1)[0] for n in names])
        va, vb, lo, hi = rfs_paired(ra["rfs"], rb["rfs"], cl, seq)
        rows = [G.row("wod.rfs", "RFS, 479 rater frames (serving ONNX)", va, vb, rule="RFS no drop (paired 95% upper bound >= 0)",
                      ok=bool(hi >= 0), ci=[lo, hi], note="segment bootstrap B=2000; shipped vs itself = 0")]
        from experiments.op_adapt_l.lib import op_adapt_l as L
        assert np.array_equal(sa["rows"], sb["rows"])
        d = L.paired_delta(sa["false_start"], sb["false_start"], sa["groups"].astype(str), B=B)
        rows.append(G.row("wod.false_start", "false-start rate, WOD val stay rows (port, plan >= 3 m at 4 s)", d["adapt"], d["orig"],
                          rule="false start <= shipped + 2 pp", ok=G.at_most(d["adapt"], d["orig"], 0.02), ci=[d["lo"], d["hi"]],
                          note=f"n {d['n']} rows / {d['clusters']} segments; op_adapt_L/H metric and rows"))
        st = rater_stay_mask(names)
        fa, fb = ra["disp4"][st] >= L.FALSE_START_M, rb["disp4"][st] >= L.FALSE_START_M
        rows.append(G.row("wod.false_start_rater", "false-start rate, the stay frames among the rater frames (serving)", float(fa.mean()),
                          float(fb.mean()), rule="info (n too small for a 2 pp line)", ok=None,
                          note=f"n {int(st.sum())} frames (1 frame = {100 / max(st.sum(), 1):.1f} pp)"))
        status = "ok"
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        rows, status = [G.row("wod", "error", None, rule=rule, note=f"{type(e).__name__}: {e}")], "error"
    p = G.write_line(out, LINE, c["name"], a.mode, rows, t0, status, prov)
    print(json.dumps(json.loads(p.read_text()), indent=1, default=str))


def argparse_like(a):
    """The reference's arrays use the cache even when the candidate is forced (shipped is recomputed by its own --force run)."""
    import copy
    b = copy.copy(a)
    b.force = False
    return b


def rater_stay_mask(names):
    import op_adapt_l_prep as P
    return np.asarray(P.flags(P.wod_kin(names))["stay"], bool)


if __name__ == "__main__":
    main()
