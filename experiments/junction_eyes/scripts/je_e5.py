#!/usr/bin/env python
"""junction_eyes E5 (research/next-round/junction-eyes-plan.md 4.5 G2 (b)): one Cinque ONNX (factory or a merged fine-tuned copy) on
held-out comma1M segments, stepped modeld-style at 20 Hz on the native rig: frames.segment_model_frames (road 910 / wide 455 from the
segment's own calibration), desire 0, traffic (1, 0), action_t (0.275, 0.525), OPModel on onnxruntime cuda-iob -- the stepping of
experiments/op_fov fov_replay.py and the comma read-out of experiments/op_wide_ft wide_real.py.

Read-outs (per --tag, under $DATA_DIR/runs/junction_eyes/e5/<tag>/):
  ADE4    whole segment, the model stepped from a zero state over every frame. Scored frames: >= 6 s in (fov_replay WARM = 120), v > 2 m/s,
          localizer pose known 4 s ahead. Per frame: mean L2 (x, y) of the plan mean at the model's own T_IDXS in (0, 4] s (20 points, last
          3.91 s; fov_report's ADE5 convention cut at 4 s) against the localizer positions in that frame's calib frame (fov_report.calib_future).
          Per segment: mean over its scored frames (seg.csv).
  A_act   op_fov's action curvature gain on the turn windows (fov_replay.scan_segment, unchanged rules), each window replayed from a zero
          state 6 s before scoring and scored by fov_report.window_metrics unchanged (also A_plan, H2, lead, peak_ratio, FDE3; straight
          windows: ADE5, v_ratio, lane_w10) -> windows.csv. Window npz files keep fov_replay's layout (arm name "M").
  --ref T paired against tag T's run (the factory) on the shared segments / windows: ADE4 ratio arm / ref per segment (mean of the
          per-segment ratios = op_wide_ft's straight-ratio convention, bootstrap over segments; also the ratio of pooled frame sums, same
          resamples); turn-window metrics arm - ref, cluster bootstrap over segments (jevdrive.stats, B 10 000, seed 0) -> e5.{csv,md},
          summary.json. Gate lines of G2 (b): ADE4 ratio <= 1.05, |dA_act| <= 0.05.

Segments (--segs): a file (one name per line, a JSON list, or a JSON object with a `heldout` / `held_out` / `test` / `segments` list; window
names `<seg>_<start>` are reduced to their segment) or a split ref (comma1m/<name>@vK). Only segments with both fcamera.hevc and
ecamera.hevc can be read (126 of the 602 on the box, all 1928 x 1208); others are dropped and listed in summary.json.
PLACEHOLDER: without --segs, the 60 segments with both cameras and >= 1 scan event that come first in sha256(segment name) order. This rule
is not the held-out set: it MUST be replaced by executor 2's keep-set manifest (path in experiments/junction_eyes/results/pixels.md) before
any G2 read.

  # box, one pool job (openpilot env: av + onnxruntime-gpu; the report step runs in op-train for pandas):
  python -m jevdrive.cl submit --name je-e5-<tag> --vram 16 --cpu 32 --ram 64 -- \\
      $DATA_DIR/envs/openpilot/bin/python experiments/junction_eyes/scripts/je_e5.py run --onnx <merged.onnx> --tag <tag> --ref factory [--segs F]
  factory reference: --onnx ~/data/models/openpilot/cinque.ort.onnx --tag factory
  report only:  $DATA_DIR/envs/op-train/bin/python experiments/junction_eyes/scripts/je_e5.py report --tag <tag> --ref factory
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "experiments/op_fov/scripts")]
N_SEGS = 60
WARM = 120                                  # fov_replay.WARM: 6 s (> the 5 s context) before scoring
HORIZON, V_MIN = 4.0, 2.0
ACTION_T = (0.275, 0.525)
ARM = "M"                                   # arm name inside the fov_replay-layout window files
TURN_M = ("A_act", "A_plan", "H2", "lead", "peak_ratio", "FDE3")
STRAIGHT_M = ("ADE5", "v_ratio", "lane_w10")
CACHE_VERSION = "je-e5-v1"
FACTORY = Path.home() / "data/models/openpilot/cinque.ort.onnx"


def comma():
    from jevdrive.common import data_dir
    return data_dir() / "datasets" / "comma1M"


def root(*p):
    from jevdrive.common import data_dir
    d = data_dir() / "runs" / "junction_eyes" / "e5" / Path(*p)
    d.mkdir(parents=True, exist_ok=True)
    return d


def readable(seg):
    return all((comma() / seg / f).exists() for f in ("fcamera.hevc", "ecamera.hevc", "frame_info.safetensors", "localizer.safetensors"))


def scan(seg):
    import fov_replay as FR
    FR.ROOT = comma()
    return FR.scan_segment(seg)


# ---------------------------------------------------------------- segments
def load_segs(spec):
    p = Path(spec)
    if p.exists():
        txt = p.read_text()
        try:
            obj = json.loads(txt)
        except ValueError:
            obj = [ln.split()[0] for ln in txt.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
        if isinstance(obj, dict):
            obj = next(obj[k] for k in ("heldout", "held_out", "test", "segments") if k in obj)
        names, split = [str(s) for s in obj], None
    else:
        from jevdrive.data import splits
        split = splits.load(spec)
        names = list(split.members)
    segs = [s if (comma() / s).is_dir() else s.rsplit("_", 1)[0] for s in names]
    return sorted(set(segs)), split


def fallback_segs(run):
    """PLACEHOLDER rule (see the docstring): first N_SEGS by sha256(name) among readable segments with >= 1 scan event."""
    from jevdrive import par
    f = root() / "scan.json"
    if f.exists():
        ev = json.loads(f.read_text())
    else:
        segs = sorted(p.name for p in comma().iterdir() if p.is_dir() and readable(p.name))
        res = par.pmap(scan, segs, workers=workers(), run=run, desc="scan")
        res.raise_if_failed()
        ev = dict(zip(segs, res.values))
        f.write_text(json.dumps(ev))
    ok = [s for s, e in ev.items() if e]
    return sorted(ok, key=lambda s: hashlib.sha256(s.encode()).hexdigest())[:N_SEGS]


def workers():
    from jevdrive.common import n_cpus
    return max(1, min(n_cpus(), len(os.sched_getaffinity(0))) // 4)


# ---------------------------------------------------------------- replay
_M = None


def replay(frames, lanes=False):
    """Zero state, one modeld step per frame -> act (n, 2 raw mu + 2 std), plan mu (n, 33, 15) [, lane lines mu (n, 4, 33, 2), prob (n, 4)]."""
    from jevdrive.openpilot.model import mdn_mu, sigmoid
    _M.reset()
    sl = _M.slices
    raw = np.stack([_M.step(x, desire=np.zeros(8), traffic=(1, 0), action_t=ACTION_T) for x in frames])
    out = dict(act=raw[:, sl["action"]].astype(np.float32), plan=np.stack([mdn_mu(r[sl["plan"]], (33, 15)) for r in raw]).astype(np.float32))
    if lanes:
        out["ll"] = np.stack([mdn_mu(r[sl["lane_lines"]], (4, 33, 2)) for r in raw]).astype(np.float32)
        out["lp"] = np.stack([sigmoid(r[sl["lane_lines_prob"]])[1::2] for r in raw]).astype(np.float32)
    return out


def future_calib(meta, idx, ts):
    """fov_report.calib_future for many frames at once: (len(idx), len(ts), 3) logged positions in frame i's calib frame, NaN past the end."""
    from scipy.spatial.transform import Rotation
    t, pos = meta["t_loc"], meta["pos"]
    tq = t[idx, None] + ts[None]
    p = np.stack([np.interp(tq, t, pos[:, k]) for k in range(3)], -1) - pos[idx, None]
    cfd = Rotation.from_euler("xyz", meta["rpy_calib"]).as_matrix().T
    out = np.einsum("ab,mkb->mka", cfd, np.einsum("mja,mkj->mka", meta["R"][idx], p))
    out[tq > t[-1]] = np.nan
    return out


def seg_job(job):
    """One segment: whole-segment replay (ADE4) + every scan window replayed from its own start (fov_replay layout) -> npz files."""
    global _M
    import fov_replay as FR
    from jevdrive import cache
    from jevdrive.openpilot.frames import load_segment_meta, segment_model_frames
    from jevdrive.openpilot.model import T_IDXS, OPModel
    seg, onnx, out, backend = job
    out = Path(out)
    f = out / "segs" / f"{seg}.npz"
    key = cache.key(params=dict(seg=seg, warm=WARM, horizon=HORIZON, v_min=V_MIN), inputs=[onnx], code=[seg_job, replay, future_calib],
                    version=CACHE_VERSION)
    if cache.done(f, key):
        return str(f)
    meta = load_segment_meta(comma() / seg)
    frames, meta = segment_model_frames(comma() / seg, meta, max_frames=1400)
    if _M is None:
        _M = OPModel(str(onnx), backend)
    full = replay(frames)
    v, yr = FR.motion(meta)
    n = min(len(frames), len(meta["t_loc"]))
    ts = T_IDXS[(T_IDXS > 0) & (T_IDXS <= HORIZON)]
    idx = np.arange(WARM, n)
    idx = idx[(v[idx] > V_MIN) & (meta["t_loc"][idx] + ts[-1] <= meta["t_loc"][-1])]
    ade = np.full(len(frames), np.nan, np.float32)
    k = np.flatnonzero(T_IDXS == ts[0])[0]
    ade[idx] = np.linalg.norm(full["plan"][idx, k:k + len(ts), :2] - future_calib(meta, idx, ts)[..., :2], axis=-1).mean(1)
    evs, gap = scan(seg), []
    for w in evs:
        lo, hi = w["start"], min(w["hi"], len(frames))
        r = replay(frames[lo:hi], lanes=True)
        gap.append(float(np.abs(r["act"][WARM:] - full["act"][lo + WARM:hi]).max()) if hi - lo > WARM else np.nan)
        wf = out / "win" / f"{FR.wname(w)}.npz"
        cache.cached(wf, key, lambda: {**{f"{ARM}_{q}": x for q, x in r.items()}, f"{ARM}_miss": np.zeros(2), "t": meta["t_loc"], "v": v,
                                       "yr": yr, "lo": lo, "hi": hi, "pos": meta["pos"], "R": meta["R"], "rpy_calib": meta["rpy_calib"],
                                       "ev": json.dumps(w)})
    cache.cached(f, key, lambda: dict(act=full["act"], plan=full["plan"], ade4=ade, v=v[:len(frames)].astype(np.float32),
                                      wins=np.array([FR.wname(w) for w in evs]), win_vs_full_act=np.array(gap, np.float32)))
    return str(f)


def cmd_run(a):
    from jevdrive import par
    from jevdrive.run import Run
    from fov_report import window_metrics
    onnx = Path(a.onnx).expanduser().resolve()
    assert onnx.exists(), onnx
    with Run("junction_eyes", f"e5-{a.tag}", config=vars(a)) as run:
        segs, split = load_segs(a.segs) if a.segs else (fallback_segs(run), None)
        if split is not None:
            run.use_split(split)
        dropped = [s for s in segs if not readable(s)]
        segs = [s for s in segs if readable(s)]
        if dropped:
            run.log.warning("%d segments lack a camera stream / localizer and are dropped: %s", len(dropped), dropped[:10])
        out = root(a.tag)
        (out / "segs").mkdir(exist_ok=True)
        (out / "win").mkdir(exist_ok=True)
        nw = a.workers or workers()
        res = par.pmap(seg_job, [(s, str(onnx), str(out), a.backend) for s in segs], workers=nw, run=run, desc="segments")
        res.raise_if_failed()
        srows, wrows, gaps = [], [], []
        for s, fp in zip(segs, res.values):
            z = np.load(fp)
            ade = z["ade4"]
            ok = np.isfinite(ade)
            srows.append(dict(seg=s, n_frames=len(ade), n_scored=int(ok.sum()), ADE4=float(ade[ok].mean()) if ok.any() else np.nan,
                              ADE4_sum=float(ade[ok].sum())))
            gaps += [g for g in z["win_vs_full_act"].tolist() if np.isfinite(g)]
            for w in z["wins"].tolist():
                wrows += window_metrics(out / "win" / f"{w}.npz")
        write_csv(out / "seg.csv", srows)
        write_csv(out / "windows.csv", wrows)
        (out / "run.json").write_text(json.dumps(dict(tag=a.tag, onnx=str(onnx), onnx_sig=sig(onnx), segs=segs, dropped=dropped,
                                                      split=split.id if split else (a.segs or "PLACEHOLDER sha256-first-60"),
                                                      run_dir=str(run.dir), workers=nw), indent=1))
        run.summary.update(segments=len(segs), dropped=len(dropped), windows=len(wrows), out=str(out),
                           win_vs_full_act_max=max(gaps) if gaps else None)
        run.info("ADE4 mean over segments %.3f m; turn windows %d, straight %d; max |act(window replay) - act(whole-segment replay)| "
                 "after the 6 s warm-up %.2g", np.nanmean([r["ADE4"] for r in srows]), sum(r["kind"] == "turn" for r in wrows),
                 sum(r["kind"] == "straight" for r in wrows), max(gaps) if gaps else np.nan)
    report(a)


def sig(p):
    st = Path(p).stat()
    return dict(size=st.st_size, mtime_ns=st.st_mtime_ns)


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- report
def report(a):
    try:
        import pandas  # noqa: F401
    except ImportError:                     # openpilot env: no pandas; the report runs in op-train
        from jevdrive.common import data_dir
        py = data_dir() / "envs" / "op-train" / "bin" / "python"
        subprocess.run([str(py), __file__, "report", "--tag", a.tag] + (["--ref", a.ref] if a.ref else []), check=True)
        return
    cmd_report(a)


def cmd_report(a):
    import pandas as pd
    from jevdrive import stats
    out = root(a.tag)
    seg, win = pd.read_csv(out / "seg.csv"), pd.read_csv(out / "windows.csv")
    rows = [dict(set="segment", contrast=a.tag, metric="ADE4", **stats.bootstrap(seg.ADE4.to_numpy(float)))]
    for kind, ms in (("turn", TURN_M), ("straight", STRAIGHT_M)):
        w = win[win.kind == kind]
        rows += [dict(set=kind, contrast=a.tag, metric=m, **stats.bootstrap(w[m].to_numpy(float), groups=w.seg.to_numpy()))
                 for m in ms if m in w and w[m].notna().any()]
    summ = dict(tag=a.tag, segments=len(seg), turn_windows=int((win.kind == "turn").sum()), straight_windows=int((win.kind == "straight").sum()),
                ADE4=float(seg.ADE4.mean()), ADE4_pooled=float(seg.ADE4_sum.sum() / seg.n_scored.sum()),
                A_act=float(win.loc[win.kind == "turn", "A_act"].mean()) if (win.kind == "turn").any() else None)
    if a.ref:
        ro = root(a.ref)
        rs, rw = pd.read_csv(ro / "seg.csv"), pd.read_csv(ro / "windows.csv")
        j = seg.merge(rs, on="seg", suffixes=("", "_ref"))
        assert (j.n_scored == j.n_scored_ref).all(), "scored frames differ between the runs (the mask depends on the data only)"
        r = stats.bootstrap((j.ADE4 / j.ADE4_ref).to_numpy(float))
        rows.append(dict(set="segment", contrast=f"{a.tag} / {a.ref}", metric="ADE4", **r))
        idx = np.random.default_rng(stats.SEED).integers(0, len(j), (stats.N_BOOT, len(j)))
        x, y = j.ADE4_sum.to_numpy(float), j.ADE4_sum_ref.to_numpy(float)
        b = x[idx].sum(1) / y[idx].sum(1)
        lo, hi = np.quantile(b, [stats.ALPHA / 2, 1 - stats.ALPHA / 2])
        rp = dict(n=len(j), units=len(j), mean=float(x.sum() / y.sum()), lo=float(lo), hi=float(hi), n_boot=stats.N_BOOT, seed=stats.SEED,
                  alpha=stats.ALPHA)
        rows.append(dict(set="segment", contrast=f"{a.tag} / {a.ref} (pooled)", metric="ADE4", **rp))
        pw = {}
        for kind, ms in (("turn", TURN_M), ("straight", STRAIGHT_M)):
            jw = win[win.kind == kind].merge(rw[rw.kind == kind], on=["win", "seg"], suffixes=("", "_ref"))
            for m in ms:
                if m in jw and jw[m].notna().any():
                    pw[m] = stats.paired(jw[m].to_numpy(float), jw[f"{m}_ref"].to_numpy(float), groups=jw.seg.to_numpy())
                    rows.append(dict(set=kind, contrast=f"{a.tag} - {a.ref}", metric=m, **pw[m]))
        da = pw.get("A_act")
        summ |= dict(ref=a.ref, shared_segments=len(j), missing_in_ref=sorted(set(seg.seg) - set(rs.seg)),
                     ADE4_ratio=dict(mean=r["mean"], lo=r["lo"], hi=r["hi"]), ADE4_ratio_pooled=dict(mean=rp["mean"], lo=rp["lo"], hi=rp["hi"]),
                     dA_act=dict(mean=da["mean"], lo=da["lo"], hi=da["hi"], n=da["n"], units=da["units"]) if da else None,
                     gate_G2b=dict(line="ADE4 ratio <= 1.05 and |dA_act| <= 0.05 (plan 4.5); evaluated on the point and on the CI",
                                   ade_point=r["mean"] <= 1.05, ade_ci=r["hi"] <= 1.05,
                                   act_point=bool(da and abs(da["mean"]) <= 0.05), act_ci=bool(da and da["lo"] >= -0.05 and da["hi"] <= 0.05)))
    stem = out / (f"e5_vs_{a.ref}" if a.ref else "e5")
    stats.write_table(rows, stem, note=f"comma1M E5, {len(seg)} segments; ADE4 one value per segment; windows: cluster bootstrap over segments")
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(Path(str(stem) + ".md").read_text())
    print(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("run", "report"))
    ap.add_argument("--onnx", default=str(FACTORY), help="Cinque ONNX (merged copy); never modified")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ref", default="", help="tag of the factory run to pair against (same segments)")
    ap.add_argument("--segs", default="", help="held-out segment list (file or split ref); default: the PLACEHOLDER rule")
    ap.add_argument("--workers", type=int, default=0, help="segment workers (default: usable cores / 4), one CUDA session each")
    ap.add_argument("--backend", default="cuda-iob")
    a = ap.parse_args()
    cmd_run(a) if a.cmd == "run" else cmd_report(a)
