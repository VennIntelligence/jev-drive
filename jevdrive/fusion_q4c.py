"""Fusion diagnostics Q4c: openpilot's own lead output on the P5 observation frames, against a GT lead.
Pre-registration: todos/2026-09-25-fusion-diagnostics.md (Q4, and the [Q4c] lines of the deviation log).

  equiv   the re-run's `temporal` (op_streams_lead) against experiment 1's (op_streams), stream by stream, bit for bit
  gt      per observation frame: the nearest vehicle whose footprint meets the ego lane corridor (route centreline
          ahead +- 1.5 m), <= 80 m; its reference point (footprint point nearest the camera) in the rear-axle frame
  score   recall (lead_prob[0] > 0.5 when a GT lead exists), false alarms, distance error (x_pred - GT x from the
          camera) by distance bin x weather, per model

Frames as in fusion_q4: ego = rear axle on the ground, x forward, y left. openpilot's lead x is measured from the
device (camera); the virtual road camera sits at the physical front camera, (1.519, 0.026, 1.806) in that frame.
"""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import fusion_q4 as Q4
from .common import data_dir, get_logger

log = get_logger(__name__)
MODELS = ("cinque", "lebowski")
HALF_W, MAX_LEAD, PATH_LEN, DZ_MAX = 1.5, 80.0, 90.0, 8.0
CAM = np.array([1.519, 0.026])            # front camera = openpilot device origin, rear-axle frame
EGO_X = -Q4.REAR_AXLE_X                   # hero actor location in the rear-axle frame
PROB = 0.5
GRID = (7, 5)                              # footprint samples along x, y
WEATHER = {"all": lambda f: np.ones(len(f), bool), "day": lambda f: f.sun_altitude.to_numpy() >= 0,
           "night": lambda f: f.sun_altitude.to_numpy() < 0, "rain": lambda f: f.precipitation.to_numpy() > 30}
N_BOOT = 500


def p5(*parts) -> Path:
    return data_dir() / "processed" / "carla_p5" / Path(*parts)


# ================================================================ equivalence and model outputs

def load_streams(model: str, sub: str = "op_streams_lead", prefix: str = "p5_") -> dict:
    return {f.stem: f for f in sorted(p5(sub, model).glob(f"{prefix}*.npz"))}


def equiv(model: str) -> dict:
    """New `temporal` against experiment 1's, on every re-run stream."""
    new, old = load_streams(model), load_streams(model, "op_streams")
    n, same, worst, missing = 0, 0, 0.0, 0
    for k, f in new.items():
        if k not in old:
            missing += 1
            continue
        with np.load(f) as a, np.load(old[k]) as b:
            assert (a["name"] == b["name"]).all(), k
            d = np.abs(a["temporal"] - b["temporal"]).max(1)
        n, same, worst = n + len(d), same + int((d == 0).sum()), max(worst, float(d.max()))
    return {"model": model, "streams": len(new), "streams_missing_old": missing, "rows": n,
            "identical_rows": same, "max_abs_diff": worst}


def outputs(model: str) -> pd.DataFrame:
    """Decoded lead at t = 0 of selection 0 for every stored target: x, y (device frame) and P(lead)."""
    parts = []
    for f in load_streams(model).values():
        with np.load(f) as z:
            assert z["lead"].shape[1] == 144 and z["lead_prob"].shape[1] == 3, (f, z["lead"].shape)
            mu = z["lead"][:, :72].reshape(-1, 3, 6, 4)          # openpilot.model.mdn_mu, row by row
            p = 1 / (1 + np.exp(-np.clip(z["lead_prob"][:, 0], -11, None)))   # openpilot.model.sigmoid
            parts.append(pd.DataFrame({"frame_name": z["name"].astype(str), "x_pred": mu[:, 0, 0, 0],
                                       "y_pred": mu[:, 0, 0, 1], "prob": p}))
    return pd.concat(parts, ignore_index=True)


# ================================================================ GT lead

def _route_path(route: pd.DataFrame, hx, hy, hyaw) -> np.ndarray:
    """Route centreline ahead of the hero in the rear-axle frame, starting at the hero's own position
    (fusion_diag.gt_run's choice of the route point, extended to PATH_LEN)."""
    th = np.radians(hyaw)
    dx, dy = route.x.to_numpy() - hx, route.y.to_numpy() - hy
    lx, ly = np.cos(th) * dx + np.sin(th) * dy, -np.sin(th) * dx + np.cos(th) * dy
    pts = np.c_[lx - Q4.REAR_AXLE_X, -ly]
    dist = np.hypot(pts[:, 0] - EGO_X, pts[:, 1])
    dist[np.abs((route.yaw.to_numpy() - hyaw + 180) % 360 - 180) > 90] = np.inf
    if not np.isfinite(dist).any():
        return np.zeros((1, 2))
    pts = pts[int(dist.argmin()):]
    fwd = pts[:, 0] > EGO_X
    pts = pts[int(fwd.argmax()):] if fwd.any() else pts[-1:]
    path = np.r_[[[EGO_X, 0.0]], pts]
    cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    return path[: max(2, int(np.searchsorted(cum, PATH_LEN)) + 1)]


def _gt_attempt(args) -> pd.DataFrame:
    from .fusion_diag import project
    adir, frames = args
    adir = Path(adir)
    pose = {}
    with open(adir / "pose.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["frame"] in frames:
                pose[r["frame"]] = (r["x"], r["y"], r["z"], r["yaw"])
    a = np.load(adir / "actors.npz")
    kinds = json.loads((adir / "actor_kinds.json").read_text())
    hero = json.loads((adir / "meta.json").read_text()).get("hero_id")
    route = pd.read_json(adir / "route.json")
    gu, gv = np.meshgrid(np.linspace(-1, 1, GRID[0]), np.linspace(-1, 1, GRID[1]), indexing="ij")
    gu, gv = gu.ravel(), gv.ravel()
    rows = []
    for fr in sorted(frames):
        row = {"adir": str(adir), "frame": int(fr), "lead": False, "path_len": 0.0, "n_vehicles": 0}
        if fr not in pose:
            rows.append(row | {"no_pose": True})
            continue
        hx, hy, hz, hyaw = pose[fr]
        path = _route_path(route, hx, hy, hyaw)
        row["path_len"] = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) if len(path) > 1 else 0.0
        m = (a["frame"] == fr) & (a["id"] != hero) & (np.abs(a["xyz"][:, 2] - hz) < DZ_MAX)
        k = [kinds.get(str(int(i))) for i in a["id"][m]]
        veh = np.array([x is not None and x[0].startswith("vehicle.") for x in k], bool)
        if not veh.any() or len(path) < 2:
            rows.append(row)
            continue
        ids, P, Y = a["id"][m][veh], a["xyz"][m][veh], a["yaw"][m][veh]
        k = [x for x, v in zip(k, veh) if v]
        bb = np.array([x[2] for x in k], np.float64)
        ay, th = np.radians(Y), np.radians(hyaw)
        cx = P[:, 0] + np.cos(ay) * bb[:, 0] - np.sin(ay) * bb[:, 1]          # fusion_q4._p5_gt_attempt's transform
        cy = P[:, 1] + np.sin(ay) * bb[:, 0] + np.cos(ay) * bb[:, 1]
        dx, dy = cx - hx, cy - hy
        lx, ly = np.cos(th) * dx + np.sin(th) * dy, -np.sin(th) * dx + np.cos(th) * dy
        ex, ey, eyaw = lx - Q4.REAR_AXLE_X, -ly, -(ay - th)
        c, s = np.cos(eyaw), np.sin(eyaw)
        u, v = gu[None] * bb[:, 3:4], gv[None] * bb[:, 4:5]                    # (n, 35) footprint samples
        px, py = ex[:, None] + c[:, None] * u - s[:, None] * v, ey[:, None] + s[:, None] * u + c[:, None] * v
        sa, da, _, ins = project(path, np.c_[px.ravel(), py.ravel()])
        hit = (ins & (sa > 0) & (np.abs(da) <= HALF_W)).reshape(len(ex), -1).any(1)
        ref = Q4.nearest_on_box(np.c_[ex, ey], eyaw, bb[:, 3:5], CAM)
        dist = np.hypot(ref[:, 0], ref[:, 1])
        row["n_vehicles"] = int(len(ex))
        cand = np.flatnonzero(hit & (dist <= MAX_LEAD))
        if len(cand):
            j = cand[dist[cand].argmin()]
            row.update(lead=True, lead_id=int(ids[j]), lead_type=k[j][0], gt_dist=float(dist[j]),
                       gt_x_cam=float(ref[j, 0] - CAM[0]), gt_y_cam=float(ref[j, 1] - CAM[1]),
                       n_corridor=int(hit.sum()))
        rows.append(row)
    return pd.DataFrame(rows)


def gt(workers: int) -> pd.DataFrame:
    t = pd.read_parquet(p5("index.parquet"))
    t = t[t.role == "obs"].reset_index(drop=True)
    t["adir"] = t.files.map(lambda f: f[3].rsplit("/cams/", 1)[0])
    jobs = [(adir, set(g.frame.astype(int))) for adir, g in t.groupby("adir")]
    with ProcessPoolExecutor(workers) as ex:
        g = pd.concat(ex.map(_gt_attempt, jobs, chunksize=2), ignore_index=True)
    out = t[["frame_name", "adir", "frame", "base_id", "world", "seed", "sun_altitude", "precipitation"]].merge(
        g, on=["adir", "frame"], how="left", validate="one_to_one")
    assert out.lead.notna().all()
    return out


# ================================================================ score

def _bin(d: np.ndarray) -> np.ndarray:
    b = np.digitize(d, Q4.DIST_BINS[1:-1])
    lab = np.array([f"{lo}-{hi}" for lo, hi in zip(Q4.DIST_BINS[:-1], Q4.DIST_BINS[1:])])
    return np.where(np.isfinite(d), lab[np.clip(b, 0, len(lab) - 1)], "none")


def score(fr: pd.DataFrame) -> pd.DataFrame:
    from .p5_exam import boot_ratio
    rows = []
    for model, g in fr.groupby("model"):
        g = g.assign(bin=_bin(g.gt_dist.to_numpy(float)), det=g.prob.to_numpy() > PROB)
        for wn, wf in WEATHER.items():
            w = g[wf(g)]
            neg = w[~w.lead]
            fa = boot_ratio(neg.det.to_numpy(float), np.ones(len(neg)), neg.base_id.to_numpy(), N_BOOT) if len(neg) else (np.nan,) * 3
            for b in ["all"] + [f"{lo}-{hi}" for lo, hi in zip(Q4.DIST_BINS[:-1], Q4.DIST_BINS[1:])]:
                pos = w[w.lead & ((w.bin == b) if b != "all" else True)]
                rec = boot_ratio(pos.det.to_numpy(float), np.ones(len(pos)), pos.base_id.to_numpy(), N_BOOT) if len(pos) else (np.nan,) * 3
                hit = pos[pos.det]
                err = (hit.x_pred - hit.gt_x_cam).to_numpy()
                r = {"model": model, "weather": wn, "dist_bin": b, "frames": len(w), "gt_lead": len(pos),
                     "routes": pos.base_id.nunique(), "recall": rec[0], "recall_lo": rec[1], "recall_hi": rec[2],
                     "detected": len(hit), "err_abs_median": float(np.median(np.abs(err))) if len(err) else np.nan,
                     "err_abs_p90": float(np.quantile(np.abs(err), 0.9)) if len(err) else np.nan,
                     "err_signed_median": float(np.median(err)) if len(err) else np.nan,
                     "ratio_median": float(np.median(hit.x_pred / hit.gt_x_cam)) if len(err) else np.nan,
                     "lat_err_abs_median": float(np.median(np.abs(hit.y_pred - hit.gt_y_cam))) if len(err) else np.nan}
                if b == "all":
                    r.update(no_lead=len(neg), false_alarm=fa[0], fa_lo=fa[1], fa_hi=fa[2],
                             false_alarm_x_le_80=float((neg.det & (neg.x_pred <= MAX_LEAD)).mean()) if len(neg) else np.nan)
                rows.append(r)
    return pd.DataFrame(rows)


def run(rl, workers: int):
    from .fusion_diag import RESULTS
    eq = pd.DataFrame([equiv(m) for m in MODELS])
    eq.to_csv(rl.dir / "q4c_equivalence.csv", index=False)
    rl.log.info("temporal equivalence vs experiment 1\n%s", eq.to_markdown(index=False))
    rl.event("equivalence", rows=eq.to_dict("records"))
    if not ((eq.identical_rows == eq.rows) & (eq.streams_missing_old == 0)).all():
        rl.log.error("temporal differs from experiment 1: stopping before any lead is scored")
        raise SystemExit(1)
    G = gt(workers)
    G.to_parquet(rl.dir / "q4c_gt.parquet", index=False)
    rl.log.info("GT: %d obs frames, %d with a lead (%.1f%%), path < 80 m on %d; by world %s", len(G), int(G.lead.sum()),
                100 * G.lead.mean(), int((G.path_len < MAX_LEAD).sum()), G.groupby("world").lead.mean().round(3).to_dict())
    fr = []
    for m in MODELS:
        o = outputs(m)
        f = G.merge(o, on="frame_name", how="left")
        assert f.prob.notna().all(), f"{m}: {int(f.prob.isna().sum())} obs frames without output"
        fr.append(f.assign(model=m))
    fr = pd.concat(fr, ignore_index=True)
    fr.to_parquet(rl.dir / "q4c_frames.parquet", index=False)
    tab = score(fr)
    tab.to_csv(rl.dir / "q4c_lead.csv", index=False)
    out = RESULTS / "q4c"
    out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / "q4c_lead.csv", index=False, float_format="%.4g")
    eq.to_csv(out / "q4c_equivalence.csv", index=False)
    cols = ["model", "weather", "dist_bin", "gt_lead", "recall", "recall_lo", "recall_hi", "err_abs_median", "err_abs_p90",
            "err_signed_median", "ratio_median", "false_alarm", "no_lead"]
    rl.log.info("lead recall / distance error\n%s", tab[cols].to_markdown(index=False, floatfmt=".3f"))
    return tab


def main():
    import argparse, os
    from .runlog import RunLog
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    rl = RunLog("fusion_diag", "q4c")
    rl.event("start", args=vars(a))
    run(rl, a.workers or len(os.sched_getaffinity(0)))
    rl.event("end")
    rl.close()


if __name__ == "__main__":
    main()
