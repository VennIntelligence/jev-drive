#!/usr/bin/env python
"""op_wide_ft secondary readouts on real data (plans/2026-10-05-wide-ft-prereg.md): the fine-tuned arms open loop, route = the logged future as a
hindsight navigation polyline (lib/route_adapter bear features, no noise; the served ONNX's `intent_bias` = NumpyAdapter of it), no desire.

Model arms (name: ONNX run tag under $DATA_DIR/runs/op_wide_ft/onnx, wide model-frame focal):
  S58  shipped, 455 (58.7 deg)   S116 shipped, 160 (116 deg; = op_fov W116)   A58  wf-w58-s0, 455
  B116 wf-w116-s0, 160           B58  wf-w116-s0, 455 (the W116-trained model fed the shipped wide frame)

  comma  --split comma1m/fov-full@v1 --out DIR      comma1M windows of experiments/op_fov (events.json), op_fov's warp (fov_replay.ArmWarper) at
                                                    road 910 / the arm's wide focal -> DIR/<win>.npz in fov_replay's layout (pool GPU job, .venv)
  comma-table DIR OUT_STEM                          fov_report.window_metrics per window x arm; paired contrasts, cluster bootstrap over segments
  pai-frames                                        (alpamayo venv, CPU) the d136 cases: front-wide f-theta -> road + wide 455 / 160 frames and the
                                                    hindsight route per 20 Hz step -> $DATA_DIR/runs/op_wide_ft/pai/<clip>_<t>.npz
  pai-run                                           (.venv, pool GPU) every arm on those steps -> pai/preds_<arm>.npz (alpamayo_turns oprun layout)
  pai-table OUT_STEM                                A_H / A_S / lag50 per case x arm (alpamayo_turns.sample_metrics), paired contrasts
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "lib"), str(REPO / "experiments/op_fov/scripts"), str(REPO / "experiments/alpamayo_turns/scripts")]
W = None
ARMS = {"S58": ("shipped", 455.0), "S116": ("shipped", 160.0), "A58": ("wf-w58-s0", 455.0), "B116": ("wf-w116-s0", 160.0),
        "B58": ("wf-w116-s0", 455.0)}
CONTRASTS = (("B116", "A58"), ("B116", "B58"), ("B58", "A58"), ("A58", "S58"), ("B116", "S58"), ("S116", "S58"))
CACHE_VERSION = "wide-real-v1"


def wroot(*p):
    from jevdrive.common import data_dir
    d = data_dir() / "runs" / "op_wide_ft" / Path(*p)
    d.mkdir(parents=True, exist_ok=True)
    return d


def onnx_of(tag):
    return None if tag == "shipped" else wroot("onnx") / f"{tag}.onnx"


def route_feat(enc, path_xy):
    """Dense future path in the ego frame (x fwd, y left, starts at the ego) -> clean route feature vector (lib/route_adapter)."""
    import route_adapter as RA
    p = np.asarray(path_xy, float)
    p = p[np.isfinite(p).all(1)]
    if len(p) < 3 or np.linalg.norm(p[-1]) < 5.0:
        return np.zeros(RA.ENC_DIM[enc], np.float32)                 # no usable route: "no command"
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    keep = np.r_[True, np.diff(d) > 0.05]
    poly, pm = RA.route_poly_from_path(p[keep][1:])
    return RA.features(enc, poly, pm)


class Models:
    """One OPModel session per distinct ONNX, plus the route adapter of each fine-tuned run."""

    def __init__(self, arms, backend):
        import route_adapter as RA
        from jevdrive.openpilot.model import OPModel
        self.m, self.ad = {}, {}
        for a in arms:
            tag = ARMS[a][0]
            if tag in self.m:
                continue
            o = onnx_of(tag)
            self.m[tag] = OPModel(str(o) if o else "cinque", backend)
            ap = o.with_suffix(".adapter.npz") if o else None
            self.ad[tag] = RA.NumpyAdapter(ap) if ap is not None and ap.exists() else None

    def run(self, arm, frames, feats):
        """frames (n, 2, 6, 128, 256), feats (n, F) or None -> raw outputs (n, D)."""
        tag = ARMS[arm][0]
        m, ad = self.m[tag], self.ad[tag]
        m.reset()
        m.extra = {}
        out = []
        for i, x in enumerate(frames):
            if ad is not None and feats is not None:
                m.extra["intent_bias"] = ad.bias(feats[i])
            out.append(m.step(x, desire=np.zeros(8), traffic=(1, 0), action_t=(0.275, 0.525)))
        return np.array(out)


# ---------------------------------------------------------------- comma1M
_MOD = None


def comma_window(job):
    """One comma1M window x every arm -> npz (fov_replay layout: <arm>_act / _plan / _ll / _lp / _miss + motion), cached."""
    global _MOD
    import fov_replay as FR
    from fov_report import calib_future
    from jevdrive import cache
    from jevdrive.openpilot.frames import decode_hevc, load_segment_meta
    from jevdrive.openpilot.model import mdn_mu, sigmoid
    w, arms, out, backend = job
    f = Path(out) / f"{FR.wname(w)}.npz"
    key = cache.key(params=dict(w=w, arms={a: ARMS[a] for a in arms}), code=[comma_window, route_feat], version=CACHE_VERSION,
                    inputs=[onnx_of(ARMS[a][0]) for a in arms if onnx_of(ARMS[a][0])])
    if cache.done(f, key):
        return str(f)
    meta = load_segment_meta(FR.ROOT / w["seg"])
    lo, hi = w["start"], w["hi"]
    focals = sorted({ARMS[a][1] for a in arms})
    warps = {fw: FR.ArmWarper(meta["rpy_calib"], 910.0, fw) for fw in focals}
    fr = {fw: np.zeros((hi - lo, 2, 6, 128, 256), np.uint8) for fw in focals}
    for n, (pr, pw) in enumerate(zip(decode_hevc(FR.ROOT / w["seg"] / "fcamera.hevc"), decode_hevc(FR.ROOT / w["seg"] / "ecamera.hevc"))):
        if n >= hi:
            break
        if n >= lo:
            for fw in focals:
                warps[fw](pr, pw, fr[fw][n - lo])
    v, yr = FR.motion(meta)
    z = dict(t=meta["t_loc"], pos=meta["pos"], R=meta["R"], rpy_calib=meta["rpy_calib"])
    ts = np.arange(0.0, 40.0, 0.25)
    feats = np.stack([route_feat("bear", (lambda q: np.stack([q[:, 0], -q[:, 1]], -1))(calib_future(z, i, ts))) for i in range(lo, hi)])
    if _MOD is None:
        _MOD = Models(arms, backend)
    sl = next(iter(_MOD.m.values())).slices
    res = {}
    for a in arms:
        raw = _MOD.run(a, fr[ARMS[a][1]], feats)
        res |= {f"{a}_act": raw[:, sl["action"]].astype(np.float32),
                f"{a}_plan": np.array([mdn_mu(r[sl["plan"]], (33, 15)) for r in raw], np.float32),
                f"{a}_ll": np.array([mdn_mu(r[sl["lane_lines"]], (4, 33, 2)) for r in raw], np.float32),
                f"{a}_lp": np.array([sigmoid(r[sl["lane_lines_prob"]])[1::2] for r in raw], np.float32),
                f"{a}_miss": np.array([warps[ARMS[a][1]].miss["road"], warps[ARMS[a][1]].miss["wide"]])}
    cache.cached(f, key, lambda: dict(res, t=meta["t_loc"], v=v, yr=yr, lo=lo, hi=hi, pos=meta["pos"], R=meta["R"], rpy_calib=meta["rpy_calib"],
                                      ev=json.dumps(w), feat=feats))
    return str(f)


def cmd_comma(a):
    import fov_replay as FR
    from jevdrive import par
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("op_wide_ft", "comma", config=vars(a)) as run:
        sp = splits.load(a.split)
        run.use_split(sp)
        ev = {FR.wname(e): e for e in json.load(open(REPO / "experiments/op_fov/results/events.json"))}
        wins = [ev[m] for m in sp.members]
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        arms = a.arms.split(",")
        res = par.pmap(comma_window, [(w, arms, str(out), a.backend) for w in wins], workers=a.workers, run=run, desc="windows")
        run.summary.update(windows=len(wins), ok=res.ok, failed=len(res.errors), out=str(out))
        res.raise_if_failed()


def cmd_comma_table(a):
    import pandas as pd
    from fov_report import window_metrics
    from jevdrive import stats
    rows = [r for f in sorted(Path(a.src).glob("*.npz")) for r in window_metrics(f)]
    d = pd.DataFrame(rows)
    d.to_csv(str(a.stem) + "_metrics.csv", index=False)
    out = []
    for kind, ms in (("turn", ("A_act", "lead", "A_plan", "H2", "peak_ratio", "FDE3")), ("straight", ("lane_w10", "v_ratio", "ADE5"))):
        dk = d[d.kind == kind]
        for arm in sorted(dk.arm.unique()):
            x = dk[dk.arm == arm]
            for m in ms:
                if m in x and x[m].notna().any():
                    out.append(dict(set=kind, contrast=arm, metric=m, **stats.bootstrap(x[m].to_numpy(float), groups=x.seg.to_numpy())))
        piv = {m: dk.pivot(index="win", columns="arm", values=m) for m in ms if m in dk}
        seg = dk.groupby("win").seg.first()
        for x_, y_ in CONTRASTS:
            for m, p in piv.items():
                if x_ in p and y_ in p and p[x_].notna().any():
                    if kind == "straight":                                  # guardrails: ratio to the reference arm per window
                        r = stats.bootstrap((p[x_] / p[y_]).to_numpy(float), groups=seg[p.index].to_numpy())
                        op = "ratio"
                    else:
                        r = stats.paired(p[x_].to_numpy(float), p[y_].to_numpy(float), groups=seg[p.index].to_numpy())
                        op = "diff"
                    out.append(dict(set=kind, contrast=f"{x_} {'/' if op == 'ratio' else '-'} {y_}", metric=m, **r))
    stats.write_table(out, Path(a.stem), note="comma1M %d turn / %d straight windows; cluster bootstrap over segments" % (
        (d[d.arm == d.arm.iloc[0]].kind == "turn").sum(), (d[d.arm == d.arm.iloc[0]].kind == "straight").sum()))
    print(Path(str(a.stem) + ".md").read_text())


# ---------------------------------------------------------------- PhysicalAI-AV (decision 136 cases)
def ftheta_index(cal, K):
    """pai_openpilot.ftheta_index with the model-frame intrinsics K: gather indices into the f-theta image, coverage (share inside the image)."""
    from jevdrive import camgeom as G
    r = G.pinhole_rays(np, K, G.OP_W, G.OP_H) @ cal["R"]
    th = np.arccos(np.clip(r[..., 2], -1, 1))
    rho = np.maximum(np.hypot(r[..., 0], r[..., 1]), 1e-12)
    rad = np.polynomial.polynomial.polyval(th, cal["fw"])
    u, v = cal["cx"] + rad * r[..., 0] / rho, cal["cy"] + rad * r[..., 1] / rho
    ok = (u >= -.5) & (u <= cal["w"] - .5) & (v >= -.5) & (v <= cal["h"] - .5) & (r[..., 2] > 0)
    x = np.clip(np.rint(u), 0, cal["w"] - 1).astype(np.int64)
    y = np.clip(np.rint(v), 0, cal["h"] - 1).astype(np.int64)
    return (y * cal["w"] + x).ravel(), float(ok.mean())


def cmd_pai_frames(a):
    from PIL import Image
    import turns as T
    from jevdrive import camgeom as G
    from jevdrive.alpamayo import data as D
    sys.path.insert(0, str(REPO / "experiments/zeroshot_openloop/archive"))
    import pai_openpilot as PO
    from jevdrive.run import Run
    Kw = {455.0: G.OP_K["wide"], 160.0: np.array([[160.0, 0, G.OP_K["wide"][0, 2]], [0, 160.0, G.OP_K["wide"][1, 2]], [0, 0, 1]])}
    with Run("op_wide_ft", "pai-frames", config=vars(a)) as run:
        avdi = D.interface()
        cases = json.loads((REPO / "experiments/alpamayo_turns/results/cases.json").read_text())
        out = wroot("pai")
        for c in run.tqdm(cases, desc="cases"):
            cal = T.calib(avdi, c["clip"], G)
            ir, cov_r = ftheta_index(cal, G.OP_K["road"])
            iw = {f: ftheta_index(cal, K) for f, K in Kw.items()}
            cam = avdi.get_clip_feature(c["clip"], avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV, maybe_stream=False)
            ego = avdi.get_clip_feature(c["clip"], avdi.features.LABELS.EGOMOTION, maybe_stream=False)
            for tk, off in T.T0_OFFSETS.items():
                t0 = c["t0"] + off
                ts = t0 - T.DT * np.arange(T.N_STEPS - 1, -1, -1, dtype=np.int64)
                imgs, fts = cam.decode_images_from_timestamps(ts)
                ycc = [np.asarray(Image.fromarray(im).convert("YCbCr")) for im in imgs]
                fr = {f"frames{int(f)}": np.stack([np.stack([PO.pack(y, ir), PO.pack(y, iw[f][0])]) for y in ycc]) for f in Kw}
                # hindsight route per step: egomotion (rear axle) from the step time to the end of the log, in that step's ego frame
                tf = np.arange(ts[0], ego.timestamps[-1], 100_000, dtype=np.int64)
                st = ego(tf)
                P, yaw = st.pose.translation[:, :2], np.unwrap(st.pose.rotation.as_euler("ZYX")[:, 0])
                feats = []
                for t in ts:
                    i = int(np.searchsorted(tf, t))
                    c_, s_ = np.cos(yaw[i]), np.sin(yaw[i])
                    d = P[i:] - P[i]
                    feats.append(route_feat("bear", np.stack([c_ * d[:, 0] + s_ * d[:, 1], -s_ * d[:, 0] + c_ * d[:, 1]], -1)[:1500]))
                np.savez(out / f"{c['clip']}_{tk}.npz", **fr, feat=np.stack(feats), t=ts, t0=t0, cam_xyz=cal["xyz"], cov_road=cov_r,
                         cov_wide455=iw[455.0][1], cov_wide160=iw[160.0][1])
                run.info(f"{c['case']} {tk}: coverage road {cov_r:.3f} wide455 {iw[455.0][1]:.3f} wide160 {iw[160.0][1]:.3f}; feat t0 {np.round(feats[-1], 2).tolist()}")


def cmd_pai_run(a):
    import turns as T
    from jevdrive.openpilot.model import T_IDXS, decode
    from jevdrive.run import Run
    arms = a.arms.split(",")
    with Run("op_wide_ft", "pai-run", config=vars(a)) as run:
        M = Models(arms, a.backend)
        files = sorted(wroot("pai").glob("*_[ps].npz"))
        for arm in arms:
            res = {}
            for f in run.tqdm(files, desc=arm):
                z = np.load(f)
                alp = np.load(T.cache("alp") / f.name)
                raw = M.run(arm, z[f"frames{int(ARMS[arm][1])}"], z["feat"])[-1]
                hist = alp["hist_xyz"]
                v = float(np.linalg.norm(hist[-1, :2] - hist[-2, :2]) / 0.1)
                o = decode(raw, next(iter(M.m.values())).slices, v, (0.275, 0.525))
                yaw = -np.asarray(o["plan_yaw"], np.float64)
                p = np.stack([o["plan_pos"][:, 0], -o["plan_pos"][:, 1]], -1)
                dv = z["cam_xyz"][:2]
                cy, sy = np.cos(yaw), np.sin(yaw)
                rear = dv + p - np.stack([cy * dv[0] - sy * dv[1], sy * dv[0] + cy * dv[1]], -1)
                res[f.stem] = dict(yaw=np.interp(T.T, T_IDXS, yaw), xy=np.stack([np.interp(T.T, T_IDXS, rear[:, k]) for k in range(2)], -1),
                                   curv=-o["curvature"])
            np.savez(wroot("pai") / f"preds_{arm}.npz", keys=np.array(list(res)), yaw=np.stack([r["yaw"] for r in res.values()]),
                     xy=np.stack([r["xy"] for r in res.values()]), curv=np.array([r["curv"] for r in res.values()]))
        run.summary["arms"] = arms


def cmd_pai_table(a):
    import pandas as pd
    import turns as T
    from jevdrive import stats
    cases = json.loads((REPO / "experiments/alpamayo_turns/results/cases.json").read_text())
    preds = {p.stem[6:]: np.load(p) for p in wroot("pai").glob("preds_*.npz")}
    rows = []
    for c in cases:
        for tk in T.T0_OFFSETS:
            k = f"{c['clip']}_{tk}"
            alp = np.load(T.cache("alp") / f"{k}.npz", allow_pickle=True)
            for arm, z in preds.items():
                i = list(z["keys"]).index(k)
                m = T.sample_metrics(z["yaw"][i][None], z["xy"][i][None], alp["gt_yaw"], alp["gt_xy"])
                rows.append(dict(case=c["case"], t0=tk, arm=arm, **{kk: float(np.nanmean(v)) for kk, v in m.items()}))
    d = pd.DataFrame(rows)
    d.to_csv(str(a.stem) + "_per_case.csv", index=False)
    out = []
    for tk in T.T0_OFFSETS:
        dd = d[d.t0 == tk]
        for m in ("A_H", "A_S", "lag50", "R_end", "ade64"):
            piv = dd.pivot(index="case", columns="arm", values=m)
            for arm in piv:
                out.append(dict(t0=tk, contrast=arm, metric=m, **stats.bootstrap(piv[arm].to_numpy(float))))
            for x_, y_ in CONTRASTS:
                if x_ in piv and y_ in piv:
                    out.append(dict(t0=tk, contrast=f"{x_} - {y_}", metric=m, **stats.paired(piv[x_].to_numpy(float), piv[y_].to_numpy(float))))
    stats.write_table(out, Path(a.stem), note="PhysicalAI-AV decision-136 cases (n = 10 per t0); bootstrap over cases")
    print(Path(str(a.stem) + ".md").read_text())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("comma", "comma-table", "pai-frames", "pai-run", "pai-table"))
    ap.add_argument("src", nargs="?")
    ap.add_argument("stem", nargs="?")
    ap.add_argument("--split", default="comma1m/fov-full@v1")
    ap.add_argument("--out", default="")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--backend", default="cuda-iob")
    a = ap.parse_args()
    if a.cmd == "pai-table":
        a.stem = a.stem or a.src
    globals()["cmd_" + a.cmd.replace("-", "_")](a)
