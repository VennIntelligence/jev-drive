"""OD2 WOD column (plans/2026-10-04-od2-prereg.md section 2): shipped Cinque and it_dw3-s0, each with and without the decision-94
uncertainty selector, on the 479 WOD-E2E val rater frames through the training port (op_adapt_l / op_adapt_H readout path: 10
frames on the 5 Hz lattice, f-18 .. f, zero state; port O reads 8.004 against the serving ONNX's 8.005).

  run    render the 10 frames of every rater frame (the exam renderer), build the rot0 variant (each history frame re-projected in
         place to the t0 heading; headings from WOD's past velocity vectors, see headings()), port forward of O and it_dw3-s0 on
         both -> $H/wod/plans.npz (plan MDN mean + std per model x variant) + headings
  check  rot0 sign check: horizontal phase-correlation shift of history frame 0 against the t0 frame, native vs rot0, on the
         turning samples (no model, no score)
  score  RFS / ADE per row, selector at --ratios, x1.06 trick rows, paired bootstraps over segments
         -> experiments/op_adapt_h/results/one_driver/wod.{md,json}

  CUDA_VISIBLE_DEVICES=0 taskset -c 0-39 $DATA_DIR/envs/op-train/bin/python experiments/op_adapt_h/scripts/h_wod.py run
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[3] / _d) for _d in ("", "scripts", "experiments/op_adapt_r2/lib")]
import argparse, json, time  # noqa: E401,E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from jevdrive.common import data_dir  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
H_ROOT = data_dir() / "runs" / "op_adapt_H"
OUT = H_ROOT / "wod"
RES = REPO / "experiments/op_adapt_h/results/one_driver"
MODELS = ("O", "it_dw3-s0")
VARIANTS = ("native", "rot0")
NIMG = 10
T_LAT = -0.2 * np.arange(NIMG - 1, -1, -1)       # image times on the 5 Hz lattice, -1.8 .. 0 s
V_MIN = 1.0                                       # m/s; below it the velocity direction is not a heading
T_SEL = np.array([10.0 * (i / 32) ** 2 for i in range(33)]) <= 4.0 + 1e-6


def sets():
    from jevdrive import wod_zeroshot as Z
    s = Z.load_sets()["rater"]
    calib = json.loads((Z.root() / "op_calib.json").read_text())
    names = s["name"].astype(str)
    seq = np.array([n.rsplit("-", 1)[0] for n in names])
    cam = np.stack([np.array(calib[q]["1"]["extrinsic"]).reshape(4, 4)[:3, 3] for q in seq])
    return s, names, seq, cam


def headings(past: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n, 10) heading (rad, left +) of the ego at the lattice times relative to t0, and (n,) bool 'any valid sample'.
    WOD-E2E stores no past yaw: the heading is the direction of the past velocity vector (rear axle, so it follows the body),
    minus its t0 value; samples below V_MIN hold the nearest valid value; a history that is slow throughout is not rotated."""
    tp = (np.arange(past.shape[1]) - (past.shape[1] - 1)) * 0.25
    out, ok = np.zeros((len(past), NIMG)), np.zeros(len(past), bool)
    for i, p in enumerate(past.astype(np.float64)):
        v = p[:, 2:4]
        m = np.linalg.norm(v, axis=1) >= V_MIN
        if not m.any():
            continue
        ok[i] = True
        th = np.unwrap(np.arctan2(v[m, 1], v[m, 0]))
        psi = np.interp(T_LAT, tp[m], th)             # np.interp holds the edge values
        out[i] = psi - np.interp(0.0, tp[m], th)
    return out, ok


# ---------------------------------------------------------------- CPU workers: render + rot0
def _init():
    import wod_zeroshot_openpilot as WZ
    from jevdrive import wod_zeroshot as Z
    spans, _ = Z.load_spans()
    WZ._init(spans, json.loads((Z.root() / "op_calib.json").read_text()), str(data_dir() / "datasets" / "waymo_e2e" / "front3"))


def hist(name):
    s, f = name.rsplit("-", 1)
    return [f"{s}-{int(f) - 2 * j:03d}" for j in range(NIMG - 1, -1, -1)]


def _job(job):
    import drive_backbones_openpilot as DB
    from experiments.op_adapt_h.lib import op_adapt_h as H
    import wod_zeroshot_openpilot as WZ
    name, cam, psi = job
    hn = hist(name)
    have = np.array([h in WZ._ctx["spans"] for h in hn])
    if not have.all():                  # a gap in the clip: keep the contiguous tail, as r2's rater cache (op_adapt_r2_readout rater)
        have[: np.flatnonzero(~have).max() + 1] = False
    imgs = np.zeros((NIMG, 2, 6, 128, 256), np.uint8)
    imgs[have] = DB.render([h for h, ok in zip(hn, have) if ok])
    rot = np.stack([H.warp(f, cam, 0.0, -p) if j < NIMG - 1 and ok else f for j, (f, p, ok) in enumerate(zip(imgs, psi, have))])
    return imgs, rot, have[:-1] & have[1:]


def cmd_run(a):
    import torch
    from experiments.op_adapt_h.lib import op_adapt_h as H
    from experiments.op_adapt_l.lib import op_adapt_l as L
    from drive_backbones_openpilot import bounded_map
    OUT.mkdir(parents=True, exist_ok=True)
    s, names, seq, cam = sets()
    psi, ok = headings(s["past"])
    from jevdrive import wod_zeroshot as Z
    spans, _ = Z.load_spans()
    miss = [n for n in names if not all(h in spans for h in hist(n))]
    print(f"{len(miss)} rater frames miss history images (contiguous tail kept, earlier slots invalid, as r2's rater cache): {miss}")
    n = len(names) if not a.limit else a.limit
    dev = torch.device("cuda")
    models = {m: L.load_model(None if m == "O" else H_ROOT / "runs" / m / "ckpt-final.pt", dev) for m in MODELS}
    net = models["O"].net
    ps = net.slices["plan"].start
    assert net.slices["plan"].stop - ps == 990, "plan head is not a single 33 x 15 Gaussian"
    mu = np.zeros((len(MODELS), 2, n, 33, 15), np.float32)
    sd = np.zeros_like(mu)
    tc = torch.tensor([[1.0, 0.0]] * a.bs, device=dev)
    jobs = [(names[i], cam[i], psi[i]) for i in range(n)]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, initializer=_init) as ex:
        it = bounded_map(ex, _job, jobs, 3 * a.bs)
        for i0 in range(0, n, a.bs):
            ch = [next(it) for _ in range(min(a.bs, n - i0))]
            b = len(ch)
            sv = torch.from_numpy(np.stack([c[2] for c in ch])).to(dev)
            for v in range(2):
                x = torch.from_numpy(np.stack([c[v] for c in ch])).to(dev)
                tr = H.trunks(net, x) * sv[:, :, None, None, None]
                with torch.no_grad():
                    for k, m in enumerate(models.values()):
                        o = m(tr, sv, tc[:b])["outputs"].float()
                        mu[k, v, i0:i0 + b] = o[:, ps:ps + 495].reshape(b, 33, 15).cpu().numpy()
                        sd[k, v, i0:i0 + b] = torch.exp(torch.clamp(o[:, ps + 495:ps + 990], max=11)).reshape(b, 33, 15).cpu().numpy()
            print(f"{i0 + b}/{n} in {time.time() - t0:.0f} s", flush=True)
    np.savez(OUT / ("plans.npz" if not a.limit else f"plans.first{n}.npz"), names=names[:n], mu=mu, std=sd, psi=psi[:n], psi_ok=ok[:n],
             models=np.array(MODELS), variants=np.array(VARIANTS),
             info=json.dumps({"path": "op_adapt_l port, 10 frames f-18..f at 5 Hz, zero state, tc right-hand, no intent",
                              "rot0": "history frame j warped by -psi_j (op_adapt_h.warp, rotation about the vehicle origin)",
                              "psi": f"WOD past velocity direction minus t0, |v| >= {V_MIN} m/s, nearest-valid hold"}))
    print(f"done: {n} frames x {len(MODELS)} models x 2 variants in {time.time() - t0:.0f} s")


def cmd_check(a):
    """Sign check on samples whose history heading at -1.8 s differs by more than --deg from t0."""
    import cv2
    from experiments.op_adapt_h.lib import op_adapt_h as H  # noqa: F401
    from jevdrive import op_interp as I
    s, names, seq, cam = sets()
    psi, _ = headings(s["past"])
    sel = np.flatnonzero(np.abs(np.degrees(psi[:, 0])) > a.deg)[: a.n]
    _init()
    res = []
    for i in sel:
        imgs, rot, _ = _job((names[i], cam[i], psi[i]))

        def shift(f0, f1):
            y0 = I.unpack(f0[0])[0].astype(np.float32)[:128]
            y1 = I.unpack(f1[0])[0].astype(np.float32)[:128]
            return cv2.phaseCorrelate(y0, y1)[0][0]
        res.append((names[i], float(np.degrees(psi[i, 0])), shift(imgs[0], imgs[-1]), shift(rot[0], rot[-1])))
    for r in res:
        print(f"{r[0]}  psi(-1.8 s) {r[1]:+6.1f} deg  dx native {r[2]:+7.1f} px  rot0 {r[3]:+7.1f} px")
    nat, r0 = np.abs([r[2] for r in res]), np.abs([r[3] for r in res])
    print(f"median |dx| native {np.median(nat):.1f} px, rot0 {np.median(r0):.1f} px; rot0 smaller on {np.mean(r0 < nat):.0%} of {len(res)}")


# ---------------------------------------------------------------- scoring
def cmd_score(a):
    import pandas as pd
    from jevdrive import stats
    from jevdrive import waymo as W
    from jevdrive import wod_zeroshot as Z
    from jevdrive.openpilot.model import T_IDXS
    z = np.load(OUT / "plans.npz", allow_pickle=True)
    s, names, seq, cam = sets()
    assert (z["names"].astype(str) == names).all()
    calib = json.loads((Z.root() / "op_calib.json").read_text())
    dev_xy = np.stack([np.array(calib[q]["1"]["extrinsic"]).reshape(4, 4)[:2, 3] for q in seq])
    cl, sp = s["cluster"].astype(str), W.init_speed(s["past"])
    traj, scores, fut = s["traj"].astype(np.float64), s["scores"].astype(np.float64), s["future"][:, :, :2].astype(np.float64)
    mu, sd = z["mu"], z["std"]

    def wod(p):          # (n, 33, 15) plan mean -> (n, 20, 2) WOD rear-axle waypoints
        return np.stack([Z.openpilot_to_wod(q[:, 0:3], q[:, 11], T_IDXS, d) for q, d in zip(p, dev_xy)]).astype(np.float64)

    def lat_std(k, v):
        return sd[k, v][:, T_SEL, 1].sum(1)
    rows = {}
    for k, m in enumerate(MODELS):
        lab = "shipped" if m == "O" else m
        wn, wr = wod(mu[k, 0]), wod(mu[k, 1])
        rows[lab] = (wn, None)
        rows[f"{lab} rot0 (every frame)"] = (wr, None)
        for r in a.ratios:
            pick = lat_std(k, 1) < r * lat_std(k, 0)
            rows[f"{lab} + selector r{r:g}"] = (np.where(pick[:, None, None], wr, wn), pick)
    groups = seq
    per, out = {}, []
    for lab, (w, pick) in rows.items():
        for xs in (1.0, 1.06):
            ww = w.copy()
            ww[..., 0] *= xs
            f = np.asarray(W.rater_feedback_score(ww, traj, scores, sp), float)
            ade = np.linalg.norm(ww - fut, axis=-1).mean(1)
            key = lab if xs == 1.0 else f"{lab} x1.06"
            per[key] = (f, ade)
            out.append({"row": key, "trick_x1.06": xs != 1.0, "n": len(f), "rfs": W.rfs_by_cluster(f, cl)[0], "rfs_frame": float(f.mean()),
                        "floored": float((f <= W.RFS_FLOOR + 1e-9).mean()), "ade": float(ade.mean()),
                        "pick_rate": None if pick is None else float(pick.mean()), "n_pick": None if pick is None else int(pick.sum())})
    rng_B = dict(n_boot=a.boot, seed=0)
    codes, uniq = pd.factorize(pd.Series(groups))
    idx = [np.flatnonzero(codes == c) for c in range(len(uniq))]
    rng = np.random.default_rng(0)
    draws = [np.concatenate([idx[c] for c in rng.integers(len(uniq), size=len(uniq))]) for _ in range(a.boot)]

    def rfs_delta(fa, fb):
        d0 = W.rfs_by_cluster(fa, cl)[0] - W.rfs_by_cluster(fb, cl)[0]
        bs = [W.rfs_by_cluster(fa[i], cl[i])[0] - W.rfs_by_cluster(fb[i], cl[i])[0] for i in draws]
        return d0, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    refs = {"vs shipped": "shipped", "vs it_dw3": "it_dw3-s0"}
    for r in out:
        for nm, ref in refs.items():
            if nm == "vs it_dw3" and not r["row"].startswith("it_dw3"):
                continue
            fa, aa = per[r["row"]]
            fb, ab = per[ref]
            d = rfs_delta(fa, fb)
            r[f"d_rfs {nm}"], r[f"d_rfs {nm} lo"], r[f"d_rfs {nm} hi"] = d
            q = stats.paired(aa, ab, groups=groups, **rng_B)
            r[f"d_ade {nm}"], r[f"d_ade {nm} lo"], r[f"d_ade {nm} hi"] = q["mean"], q["lo"], q["hi"]
    df = pd.DataFrame(out)
    RES.mkdir(parents=True, exist_ok=True)
    sel_ratio = {("shipped" if m == "O" else m): {"min": float(q.min()), "p05": float(np.percentile(q, 5)), "median": float(np.median(q)),
                                                    **{f"frac_lt_{t:g}": float((q < t).mean()) for t in (0.6, 0.8, 1.0)}}
                 for k, m in enumerate(MODELS) for q in [lat_std(k, 1) / lat_std(k, 0)]}
    meta = {"rot0_over_native_lat_std": sel_ratio, "n_frames": len(names), "n_segments": len(uniq), "n_boot": a.boot, "ratios": a.ratios, "psi_valid": float(z["psi_ok"].mean()),
            "psi_abs_deg_at_-1.8s_median": float(np.median(np.abs(np.degrees(z["psi"][:, 0])))), "info": json.loads(str(z["info"]))}
    (RES / "wod.json").write_text(json.dumps({"meta": meta, "rows": out}, indent=1, default=float) + "\n")
    print(df.round(3).to_string())
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("run")
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--workers", type=int, default=36)
    p.add_argument("--limit", type=int, default=0)
    p = sp.add_parser("check")
    p.add_argument("--deg", type=float, default=5.0)
    p.add_argument("--n", type=int, default=30)
    p = sp.add_parser("score")
    p.add_argument("--ratios", type=float, nargs="+", default=[0.6])
    p.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    {"run": cmd_run, "check": cmd_check, "score": cmd_score}[a.cmd](a)
