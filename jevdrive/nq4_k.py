"""Night queue 4, K-prep: the recipe ladder K0-K3 as two-fold cross-fitted readouts for the closed loop
(todos/2026-09-26-night-queue-4.md, section K, and the [K] entries under it, written before any K number).

  split    runs/nq4/k/route_split.json: every Bench2Drive-220 route and every recorded P5 v1 BA route gets R1 / R2,
           stratified by scenario class (alternating along the class's route ids, recorded routes first)
  labels   K1's route targets: TFv6's 10 checkpoints (first at 2.5 m, then every 1 m) on the recorder's dense route,
           rear-axle frame of each frame (runs/nq4/k/labels/route.npz)
  fit      K0 / K1 / the M-C Delta behind K3's constant, per fold R1, R2 and on all rows (the code-path control); the
           I3 rows and the P6 v0 exam rows ride along, never entering a fit or a standardisation. Writes
           runs/nq4/k/<level>/<fold>/head.npz (numpy apply) and runs/nq4/k/openloop/*.npz
  check-*  equivalence and optimisation checks (full-data K0 vs lane B's heads.npz, GPU vs CPU eigh)

The apply side (`KHead`, `path_from_route`, `g3`, `readout`, `Tfv6Rules`) is numpy only: the head server
(envs/openpilot), the Bench2Drive agent (envs/scout-tfv6) and the offline checks import it, so all paths share it.

    python -m jevdrive.nq4_k split | labels | fit | check-eigh
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

LEVELS = ("K0", "K1", "K2", "K3")
ARMS = {"k0": "K0", "k1": "K1", "k2": "K2", "k3": "K3"}          # the head server's arm names
FOLDS = ("R1", "R2")
SPEEDS = np.array([0.0, 4.0, 8.0, 10.0, 13.88888888, 16.0, 17.77777777, 20.0])   # LEAD target_speed_classes
ROUTE_FIRST, ROUTE_STEP, ROUTE_N = 2.5, 1.0, 10                  # LEAD smooth_path / num_route_points_prediction
TQ = 0.25 * np.arange(1, 21)                                     # the plan grid handed to P7
TTC_ON, TTC_OFF, CLOSING = 2.0, 6.0, 0.1                         # real_g1.g3, fixed in the [G1] 08:40 entry
SET_BA, SET_I3, SET_P6 = "carla_p5v1_ba", "hugsim_pairs", "carla_p6"
MODEL = "cinque"


def kdir(*p) -> Path:
    return Path(os.environ["DATA_DIR"]) / "runs" / "nq4" / "k" / Path(*p)


# ================================================================ split and readout choice (numpy / json only)

def load_split(path: Path | None = None) -> dict:
    return json.loads(Path(path or kdir("route_split.json")).read_text())


def readout(split: dict, route_id: str, view: str = "unseen") -> str | None:
    """The fold whose readout drives this route: unseen = the fold that never saw its recordings (never-recorded routes:
    R1, the todo's rule); seen = its own fold (None for a never-recorded route)."""
    if str(route_id).isdigit() and int(route_id) >= 100000:   # a night-queue-4 G variant id 100 b + 90 + code (nq4_g.vid)
        route_id = str(int(route_id) // 100)
    r = split["routes"].get(str(route_id))
    if r is None or not r["recorded"]:
        return "R1" if view == "unseen" else None
    f = r["fold"]
    assert view in ("unseen", "seen"), view
    return f if view == "seen" else ("R2" if f == "R1" else "R1")


def split(rl=None) -> dict:
    """[K] 17:40 (1): classes = the route's scenario type in bench2drive220.xml (P5 family for the 46 routes outside
    220); inside a class the recorded routes (id ascending) then the unrecorded 220 routes (id ascending) alternate
    R1 / R2; the first fold rotates over the classes with an odd count (class name order)."""
    import xml.etree.ElementTree as ET

    import pandas as pd
    from . import elicit_i3 as I, p5_exam as E
    out = kdir("route_split.json")
    if out.exists():
        raise SystemExit(f"{out} exists: the split is fixed once written")
    with I.p5_set(SET_BA):
        t, _, _, _, _, pairs = E.load()
    xml = Path(os.environ["DATA_DIR"]) / "third_party/Bench2Drive/leaderboard/data/bench2drive220.xml"
    r220 = {r.get("id"): [s.get("type") for s in r.iter("scenario")] for r in ET.parse(xml).getroot().iter("route")}
    assert all(len(v) == 1 for v in r220.values())
    fam = pairs.groupby("base_id").family.first()
    rec = t.groupby("base_id").agg(n=("frame_name", "size"), n_train=("role", lambda r: int((r == "train").sum())),
                                   source=("source", "first"))
    routes = {}
    for rid in sorted(set(r220) | set(rec.index), key=int):
        routes[rid] = {"class": r220[rid][0] if rid in r220 else str(fam[rid]), "recorded": rid in rec.index,
                       "in220": rid in r220,
                       "frames": int(rec.n.get(rid, 0)), "train_rows": int(rec.n_train.get(rid, 0))}
    classes = sorted({v["class"] for v in routes.values()})
    start = 0
    for c in classes:
        ids = [r for r in routes if routes[r]["class"] == c]
        order = sorted([r for r in ids if routes[r]["recorded"]], key=int) + sorted([r for r in ids if not routes[r]["recorded"]], key=int)
        for j, r in enumerate(order):
            routes[r]["fold"] = FOLDS[(j + start) % 2]
        if len(order) % 2:
            start ^= 1
    doc = {"rule": split.__doc__.strip().replace("\n", " "), "written": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
           "routes": routes}
    for f in FOLDS:
        doc[f] = sorted([r for r, v in routes.items() if v["fold"] == f and v["recorded"]], key=int)
    doc["unrecorded_220"] = sorted([r for r, v in routes.items() if v["in220"] and not v["recorded"]], key=int)
    summary = pd.DataFrame(routes).T.groupby(["class", "fold"]).recorded.sum().unstack(fill_value=0)
    doc["summary"] = {"recorded": {f: len(doc[f]) for f in FOLDS},
                      "train_rows": {f: int(sum(routes[r]["train_rows"] for r in doc[f])) for f in FOLDS},
                      "in220": {f: sum(v["in220"] and v["fold"] == f for v in routes.values()) for f in FOLDS},
                      "max_class_imbalance_recorded": int((summary["R1"] - summary["R2"]).abs().max())}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))
    print(json.dumps(doc["summary"]))
    return doc


# ================================================================ apply (numpy only)

def two_hot(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """LEAD encode_two_hot: (ids (n, 2), weights (n, 2)) with linear interpolation between neighbouring classes."""
    v = np.clip(np.asarray(v, np.float64), 0.0, SPEEDS[-1])
    j = np.clip(np.searchsorted(SPEEDS, v, side="right") - 1, 0, len(SPEEDS) - 2)
    w = (v - SPEEDS[j]) / (SPEEDS[j + 1] - SPEEDS[j])
    return np.stack([j, j + 1], 1), np.stack([1 - w, w], 1).astype(np.float32)


def path_from_route(route: np.ndarray, v: np.ndarray, ds: np.ndarray | None = None) -> np.ndarray:
    """(..., 20, 2) plan: the polyline [origin, 10 checkpoints] at arc length s(t) = v t (+ ds(t), K3), straight on
    along the last segment beyond the last checkpoint; batched over leading axes."""
    route = np.asarray(route, np.float64)
    P = np.concatenate([np.zeros(route.shape[:-2] + (1, 2)), route], -2)          # (..., 11, 2)
    seg = np.diff(P, axis=-2)
    L = np.linalg.norm(seg, axis=-1)                                                # (..., 10)
    S = np.concatenate([np.zeros(L.shape[:-1] + (1,)), np.cumsum(L, -1)], -1)       # (..., 11)
    s = np.asarray(v, np.float64)[..., None] * TQ
    if ds is not None:
        s = np.maximum.accumulate(np.maximum(s + ds, 0.0), axis=-1)
    last = seg[..., -1, :] / np.maximum(L[..., -1:], 1e-9)
    j = np.clip((s[..., :, None] >= S[..., None, 1:]).sum(-1), 0, L.shape[-1] - 1)  # segment index per sample
    s0 = np.take_along_axis(S, j, -1)
    lj = np.take_along_axis(L, j, -1)
    a = np.where(lj > 1e-9, (s - s0) / np.maximum(lj, 1e-9), 0.0)
    p0 = np.take_along_axis(P, j[..., None].repeat(2, -1), -2)
    d = np.take_along_axis(seg, j[..., None].repeat(2, -1), -2)
    inside = s <= S[..., -1:]
    out = np.where(inside[..., None], p0 + np.clip(a, 0, 1)[..., None] * d,
                   P[..., -1:, :] + (s - S[..., -1:])[..., None] * last[..., None, :])
    return out


def lead_decode(lead: np.ndarray, lead_prob: np.ndarray):
    """real_g1.lead_decode: (x, v) of selection 0 at t = 0 and P(lead)."""
    mu = np.asarray(lead, np.float64)[..., :72].reshape(lead.shape[:-1] + (3, 6, 4))
    p = 1 / (1 + np.exp(-np.clip(np.asarray(lead_prob, np.float64)[..., 0], -11, None)))
    return mu[..., 0, 0, 0], mu[..., 0, 0, 2], p


def g3(lead: np.ndarray, lead_prob: np.ndarray, v_ego: np.ndarray) -> np.ndarray:
    """real_g1.g3: P(lead) * clip((6 - TTC) / (6 - 2), 0, 1), TTC when closing faster than 0.1 m/s."""
    x, v, p = lead_decode(lead, lead_prob)
    close = np.asarray(v_ego, np.float64) - v
    with np.errstate(divide="ignore", invalid="ignore"):
        ttc = np.where(close > CLOSING, np.maximum(x, 0) / np.maximum(close, CLOSING), np.inf)
    return (p * np.clip((TTC_OFF - ttc) / (TTC_OFF - TTC_ON), 0, 1)).astype(np.float32)


def v_ego_of(ego: np.ndarray) -> np.ndarray:
    """|velocity| at t0 from the 100-d ego input (past (16, 6) flattened: row 15 = [x, y, vx, vy, ax, ay])."""
    e = np.asarray(ego, np.float64)
    return np.hypot(e[..., 92], e[..., 93])


def _softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


class KHead:
    """The four levels of one fold: K0 = `ridge ego` + `ridge_late` future; K1 = K2 = route (ridge ego + ridge_late on
    the 10 checkpoints) + target speed (ego logits + `temporal` logits, two-hot expectation) -> path_from_route;
    K3 = K1 + g3 x c along the path. All outputs (20, 2) rear-axle, x forward, y left, 0.25 ... 5 s."""

    def __init__(self, root: Path | None = None):
        self.root = Path(root or kdir())
        self.p = {}

    def params(self, level: str, fold: str) -> dict:
        key = (level, fold)
        if key not in self.p:
            with np.load(self.root / level / fold / "head.npz") as z:
                self.p[key] = {k: z[k] for k in z.files}
        return self.p[key]

    @staticmethod
    def _lin(x, W):
        return x @ W[:-1] + W[-1]

    def k0(self, fold, ego, op):
        p = self.params("K0", fold)
        xe, xo = (ego - p["ego_mu"]) / p["ego_sd"], (op - p["op_mu"]) / p["op_sd"]
        return self._lin(xe, p["We"]) + self._lin(xo, p["Wp"])

    def k1(self, fold, ego, op, level="K1"):
        p = self.params(level, fold)
        xe, xo = (ego - p["ego_mu"]) / p["ego_sd"], (op - p["op_mu"]) / p["op_sd"]
        route = (self._lin(xe, p["Re"]) + self._lin(xo, p["Rp"])).reshape(np.shape(ego)[:-1] + (ROUTE_N, 2))
        prob = _softmax(np.asarray(self._lin(xe, p["Ce"]) + self._lin(xo, p["Cp"]), np.float64))
        return route, prob @ SPEEDS

    def predict(self, arm: str, fold: str, ego, op, lead=None, lead_prob=None):
        """(path (20, 2) float64, info) for one request (or a batch along the leading axis)."""
        level = ARMS.get(arm, arm)
        if level == "K0":
            return np.asarray(self.k0(fold, ego, op), np.float64).reshape(np.shape(ego)[:-1] + (20, 2)), {}
        route, v = self.k1(fold, ego, op, level)
        info = {"v_target": v}
        ds = None
        if level == "K3":
            g = g3(lead, lead_prob, v_ego_of(ego))
            ds = np.asarray(g, np.float64)[..., None] * self.params("K3", fold)["c"][:, 0]
            info["g3"] = g
        return path_from_route(route, v, ds), info


# ================================================================ TFv6 rules (agent side, numpy only)

class Tfv6Rules:
    """LEAD 730bc1a lead/inference/sensor_agent.py, ForceMovePostProcessor then StopSignPostProcessor, with the README
    95-DS switches (creeping, stop sign) and ClosedLoopConfig's thresholds; line for line except the two inputs we do
    not have ([K] 17:40 (5)): the safety box is a callable (actor boxes instead of LiDAR points) evaluated only while
    creeping, and the stop-sign "detection" is a world-frame box centre (the oracle stand-in) kept in world
    coordinates, which is what the author's update_stop_box does to the ego-frame box every tick."""
    STUCK_THRESHOLD, STUCK_MOVE, STUCK_THROTTLE = 1100, 20, 0.4
    STOP_DIST, STOP_COOL_DOWN, STOP_SLOW_COUNT, STOP_SLOW_THROTTLE = 1.0, 120, 40, 0.1
    MAX_X_METER = 64.0

    def __init__(self):
        self.stuck_detector = self.force_move = 0
        self.stop_box = None                  # (id, world x, world y): the author's stop_sign_buffer (maxlen 1)
        self.clear_cool_down = self.slower_count = self.slower_cool_down = 0
        self.cleared = set()                  # ids the rule cleared (the author's labels drop cleared signs)

    def creep(self, speed, throttle, brake, safety_occupied):
        if speed < 0.1:
            self.stuck_detector += 1
        else:
            self.stuck_detector = 0
        if self.stuck_detector > self.STUCK_THRESHOLD:
            self.force_move = self.STUCK_MOVE
        occupied = None
        if self.force_move > 0:
            occupied = bool(safety_occupied())
            if not occupied:
                throttle, brake = max(self.STUCK_THROTTLE, throttle), 0.0
                self.force_move -= 1
            else:
                throttle, brake = 0.0, 1.0
                self.force_move = self.STUCK_MOVE
        return throttle, brake, occupied

    def stop_sign(self, speed, throttle, brake, det, ego_xy, ego_yaw):
        """det: (id, world x, world y) of this tick's detection or None; ego (vehicle centre, CARLA world, yaw rad)."""
        if self.clear_cool_down > 0:
            self.clear_cool_down -= 1
        if self.slower_cool_down > 0:
            self.slower_cool_down -= 1
        stop = False
        if det is not None and det[0] not in self.cleared:
            self.stop_box = tuple(det)
        dist = None
        if self.stop_box is not None:
            dist = float(math.hypot(self.stop_box[1] - ego_xy[0], self.stop_box[2] - ego_xy[1]))
            if dist < self.STOP_DIST and self.clear_cool_down <= 0:
                if speed > 0.01:
                    stop = True
                else:
                    self.cleared.add(self.stop_box[0])
                    self.stop_box = None
                    self.clear_cool_down = self.STOP_COOL_DOWN
                    self.slower_count = 0
            elif self.slower_cool_down <= 0 and dist < self.STOP_DIST:
                self.slower_count = self.STOP_SLOW_COUNT
                self.slower_cool_down = self.STOP_COOL_DOWN
        if self.stop_box is not None and dist is not None and dist > self.MAX_X_METER:
            self.stop_box = None
        if stop:
            throttle, brake = 0.0, 1.0
        if self.slower_count > 0:
            throttle = float(np.clip(throttle, 0.0, self.STOP_SLOW_THROTTLE))
            self.slower_count -= 1
        return throttle, brake, dist

    def step(self, speed, throttle, brake, safety_occupied, det, ego_xy, ego_yaw):
        """-> (throttle, brake, log): the author's order, creeping first, then the stop sign."""
        t0, b0 = throttle, brake
        throttle, brake, occ = self.creep(speed, throttle, brake, safety_occupied)
        throttle, brake, dist = self.stop_sign(speed, throttle, brake, det, ego_xy, ego_yaw)
        return throttle, brake, {"rule_in": [t0, b0], "rule_out": [throttle, brake], "safety": occ, "stop_det": det,
                                 "stop_dist": dist, "force_move": self.force_move, "stuck": self.stuck_detector}


def replay_rules(ticks: list) -> dict:
    """Offline rule-8 check: re-run Tfv6Rules on the logged per-tick inputs and compare the outputs bit for bit."""
    r, n, bad = Tfv6Rules(), 0, 0
    for rec in ticks:
        k = rec.get("rules")
        if k is None:
            continue
        occ = k["safety"]
        thr, brk, _ = r.step(k["v"], k["rule_in"][0], k["rule_in"][1], lambda: occ,
                             None if k["stop_det"] is None else tuple(k["stop_det"]), k["ego_c"][:2], k["ego_c"][2])
        n += 1
        bad += not (thr == k["rule_out"][0] and brk == k["rule_out"][1])
    return {"ticks": n, "different": bad}


# ================================================================ K2 step: the lead re-run against the stored `temporal`

def check_lead(set_: str) -> dict:
    """Every re-run stream's `temporal` must equal the stored op_streams_vis stream bit for bit (same plan, same targets)."""
    base = Path(os.environ["DATA_DIR"]) / "processed" / set_
    new, old = base / "op_streams_lead" / MODEL, base / "op_streams_vis" / MODEL
    fs = sorted(f for f in new.glob("*.npz") if ".tmp" not in f.name)
    bad, n, mx = [], 0, 0.0
    for f in fs:
        with np.load(f) as a, np.load(old / f.name) as b:
            ok = np.array_equal(a["name"], b["name"]) and a["temporal"].dtype == b["temporal"].dtype \
                and np.array_equal(a["temporal"], b["temporal"])
            n += len(a["name"])
            if not ok:
                bad.append(f.name)
                if a["temporal"].shape == b["temporal"].shape:
                    mx = max(mx, float(np.abs(a["temporal"] - b["temporal"]).max()))
    res = {"set": set_, "streams": len(fs), "stored_streams": len(list(old.glob("*.npz"))), "rows": n,
           "different": len(bad), "max_abs": mx, "examples": bad[:5]}
    out = kdir("checks")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"lead_vs_vis_{set_}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))
    if bad or len(fs) != res["stored_streams"]:
        raise SystemExit("lead re-run differs from the stored temporal: " + json.dumps(res))
    return res


# ================================================================ K1 labels (repo .venv, CPU)

REAR_AXLE_X = -1.388633220                                        # p4_carla: rear axle behind the vehicle centre


def _route_label_one(args):
    """Route checkpoints for the rows of one recorded attempt: the recorder's dense route (route.json, CARLA world) in
    each frame's rear-axle frame; s* = arc length of the origin's projection, first checkpoint at s* + sqrt(2.5^2 - d^2)
    (where the 2.5 m circle meets a straight route at lateral offset d), then every 1 m; straight on past the end."""
    import pandas as pd
    adir, frames, prog = args
    a = Path(adir)
    pose = pd.read_json(a / "pose.jsonl", lines=True).drop_duplicates("frame").set_index("frame")
    route = pd.read_json(a / "route.json")
    R = np.stack([route.x.to_numpy(float), -route.y.to_numpy(float)], -1)       # right-handed world
    out, dperp = np.zeros((len(frames), ROUTE_N, 2), np.float32), np.zeros(len(frames), np.float32)
    for i, (f, p) in enumerate(zip(frames, prog)):
        r = pose.loc[f]
        th = -math.radians(float(r.yaw))
        c, h = np.array([float(r.x), -float(r.y)]), np.array([math.cos(th), math.sin(th)])
        ra = c + REAR_AXLE_X * h
        lo, hi = max(int(p) - 3, 0), min(int(p) + 60, len(R))
        d = R[lo:hi] - ra
        e = np.stack([h[0] * d[:, 0] + h[1] * d[:, 1], -h[1] * d[:, 0] + h[0] * d[:, 1]], -1)
        if len(e) < 2:                                    # the route's last point: straight on along the heading
            e = np.array([[0.0, 0.0], [1.0, 0.0]]) if not len(e) else np.r_[e, e[-1:] + [1.0, 0.0]]
        seg = np.diff(e, axis=0)
        L = np.linalg.norm(seg, axis=1)
        S = np.r_[0.0, np.cumsum(L)]
        tt = np.clip(-(e[:-1] * seg).sum(1) / np.maximum(L ** 2, 1e-12), 0, 1)
        q = e[:-1] + tt[:, None] * seg
        j = int(np.argmin(np.linalg.norm(q, axis=1)))
        dp = float(np.linalg.norm(q[j]))
        s0 = S[j] + tt[j] * L[j] + math.sqrt(max(ROUTE_FIRST ** 2 - dp ** 2, 0.0))
        sq = s0 + ROUTE_STEP * np.arange(ROUTE_N)
        last = seg[-1] / max(L[-1], 1e-9)
        x = np.interp(sq, S, e[:, 0]) + np.maximum(sq - S[-1], 0) * last[0]
        y = np.interp(sq, S, e[:, 1]) + np.maximum(sq - S[-1], 0) * last[1]
        out[i], dperp[i] = np.stack([x, y], -1), dp
    return out, dperp


def speed_label(fut: np.ndarray) -> np.ndarray:
    """[K] 17:30 (4): the expert's mean speed over 0.75 ... 1.25 s (future rows 2 and 4 of the 0.25 s grid)."""
    return np.linalg.norm(fut[:, 4] - fut[:, 2], axis=-1) / 0.5


def labels(workers: int = 4) -> dict:
    from concurrent.futures import ProcessPoolExecutor

    import pandas as pd
    from . import elicit_i3 as I, p5_exam as E
    with I.p5_set(SET_BA):
        t, _, fut, _, _, _ = E.load()
    t = t.assign(adir=[str(Path(f[0]).parents[2]) for f in t.files], i=np.arange(len(t)))
    groups = [(a, g.frame.to_numpy(), g.route_progress.to_numpy(), g.i.to_numpy()) for a, g in t.groupby("adir")]
    route, dperp = np.zeros((len(t), ROUTE_N, 2), np.float32), np.zeros(len(t), np.float32)
    with ProcessPoolExecutor(workers) as ex:
        for (a, fr, pr, ii), (r, d) in zip(groups, ex.map(_route_label_one, [g[:3] for g in groups], chunksize=8)):
            route[ii], dperp[ii] = r, d
    v = speed_label(fut)
    out = kdir("labels")
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "route.npz", frame_name=t.frame_name.to_numpy().astype(str), route=route, dperp=dperp, speed=v.astype(np.float32))
    st = {"rows": len(t), "attempts": len(groups), "dperp_p50": float(np.median(dperp)), "dperp_p95": float(np.percentile(dperp, 95)),
          "off_route_gt_2_5m": float((dperp > ROUTE_FIRST).mean()), "first_ckpt_x_p50": float(np.median(route[:, 0, 0])),
          "speed_p50": float(np.median(v)), "speed_max": float(v.max()), "speed_gt_20": float((v > 20).mean())}
    (out / "route_stats.json").write_text(json.dumps(st, indent=1))
    print(json.dumps(st))
    return st


# ================================================================ fits (repo .venv, GPU)

def _gpu_eigh(dev: str, ridge: bool = False):
    """nq3_q6._gpu_eigh: float64 eigh of the pair-Delta grams (and, with ridge, of the ridge grams) on `dev`. [K] 17:58:
    the ridge grams stay on the CPU by default: `ridge ego`'s gram at lam 3e-5 is ill-conditioned enough that the GPU
    eigh moves its predictions by 1.3 mm, while d <= 512 costs nothing on the CPU."""
    import torch
    from . import planner, reactivity_mc as MC
    MC.EIGH_DEVICE = dev
    if ridge:
        def _gram_eigh(A):
            e, V = torch.linalg.eigh((A.T @ A).double().to(dev))
            return e.float().to(A.device), V.float().to(A.device)
        planner.gram_eigh = _gram_eigh


def data(with_lead: bool = True) -> dict:
    """P5 v1 BA rows first, then the I3 exam rows and the P6 v0 exam rows riding along (never fitted, never standardised on)."""
    import pandas as pd
    from . import elicit_i3 as I, nq3_q1 as Q1, p5_exam as E, p5_openpilot as PO, p5_pairs as P
    arrays = ("lead", "lead_prob") if with_lead else ()
    with I.p5_set(SET_BA):
        t, past, fut, obs, null, pairs = E.load()
        op = PO.load(t, (MODEL,), sub="op_streams_vis")[f"op-{MODEL} temporal"]
        Q = P.load_features(t, ("L18_last",))["L18_last"]
        ld = PO.load(t, (MODEL,), arrays, sub="op_streams_lead") if with_lead else {}
    with I.p5_set(SET_I3):
        ta, pa, _, _, _, _ = E.load()
        keep = ta.frame_name.isin(set(I.needed())).to_numpy()
        t3, past3 = ta[keep].reset_index(drop=True), pa[keep]
        op3 = PO.load(t3, (MODEL,), sub="op_streams")[f"op-{MODEL} temporal"]
        ld3 = PO.load(t3, (MODEL,), arrays, sub="op_streams_lead") if with_lead else {}
    t6, past6 = Q1._p6_rows()
    with I.p5_set(SET_P6):
        op6 = PO.load(t6, (MODEL,), sub="op_streams_vis")[f"op-{MODEL} temporal"]
        try:                                    # the P6 lead re-run comes after the fit (export-p6 fills K3 there)
            ld6 = PO.load(t6, (MODEL,), arrays, sub="op_streams_lead") if with_lead else {}
        except (FileNotFoundError, AssertionError):
            ld6 = {f"op-{MODEL} {k}": np.full((len(t6), ld[f"op-{MODEL} {k}"].shape[1]), np.nan, np.float32) for k in arrays}
    lab = np.load(kdir("labels", "route.npz"))
    assert (lab["frame_name"].astype(str) == t.frame_name.to_numpy()).all()
    sp = load_split()
    n, n3, n6 = len(t), len(t3), len(t6)
    fold = t.base_id.astype(str).map(lambda r: sp["routes"][r]["fold"]).to_numpy()
    d = {"t": t, "t3": t3, "t6": t6, "n": n, "n3": n3, "n6": n6, "past": past, "fut": fut, "obs": obs, "null": null,
         "fold": fold, "Q": Q, "split": sp,
         "ego": np.r_[E.ego_input(t, past), E.ego_input(t3, past3), E.ego_input(t6, past6)].astype(np.float32),
         "op": np.r_[op, op3, op6].astype(np.float32), "route": lab["route"], "speed": lab["speed"],
         "seq": np.r_[t.base_id.astype(str), "i3:" + t3.base_id.astype(str), "p6:" + t6.base_id.astype(str)]}
    if with_lead:
        d["lead"] = np.r_[ld[f"op-{MODEL} lead"], ld3[f"op-{MODEL} lead"], ld6[f"op-{MODEL} lead"]].astype(np.float32)
        d["lead_prob"] = np.r_[ld[f"op-{MODEL} lead_prob"], ld3[f"op-{MODEL} lead_prob"], ld6[f"op-{MODEL} lead_prob"]].astype(np.float32)
    pos = pd.Series(np.arange(n), index=t.frame_name)
    d["ip"] = np.r_[pos[obs.fn_plus].to_numpy(), pos[null.fn_plus].to_numpy()]
    d["im"] = np.r_[pos[obs.fn_minus].to_numpy(), pos[null.fn_null].to_numpy()]
    d["grp"] = np.r_[obs.base_id.to_numpy(), null.base_id.to_numpy()].astype(str)
    return d


def _stats(X, tr):
    """planner.standardize's statistics as numpy (float64 mean / population sd, sd <= 1e-6 -> 1, cast to float32)."""
    import torch
    mu, sd = X[tr].double().mean(0), X[tr].double().std(0, correction=0)
    return mu.float().cpu().numpy(), torch.where(sd > 1e-6, sd, 1).float().cpu().numpy()


def fit_fold(D: dict, tr: np.ndarray, rows_k: np.ndarray, pair_keep: np.ndarray, rl, tag: str, levels=("K0", "K1", "K3")) -> dict:
    """One fold's readouts on train rows `tr` (all rows ride along). rows_k: the BA rows of this fold (K3's constant);
    pair_keep: its pair rows. Returns {level: head params} and {level: predictions (N, 20, 2)}."""
    from types import SimpleNamespace

    import torch
    from sklearn.model_selection import GroupShuffleSplit
    from . import navsim_heads as H, planner, reactivity_mc as MC, waymo_stage_a as sa
    dev = "cuda"
    N, seq = len(D["ego"]), D["seq"]
    n = D["n"]
    ALL = np.arange(N)
    Ego, Xop = torch.as_tensor(D["ego"], device=dev), torch.as_tensor(D["op"], device=dev)
    fut = np.zeros((N, 20, 2), np.float32)
    fut[:n] = D["fut"]
    F = torch.as_tensor(fut.reshape(N, -1), device=dev)
    sp = SimpleNamespace(train=tr, val=tr, seq=seq)
    em, es = _stats(Ego, tr)
    om, osd = _stats(Xop, tr)
    Xe, Xi = planner.standardize(Ego, tr), planner.standardize(Xop, tr)
    heads, preds, info = {}, {}, {}

    def ridge_late(Y, T):                           # reactivity_mc.fit_fold's prior, for any target
        y3 = Y.reshape(N, T, 2).cpu().numpy()
        _, st_e, We = sa.ridge_cv(Xe, Y, sp, y3)
        base = planner.linear_apply(We, Xe, ALL)[0]
        R0 = Y - base
        _, st_p, Wp = sa.ridge_cv(Xi, R0, sp, R0.reshape(N, T, 2).cpu().numpy())
        return We[0].cpu().numpy(), Wp[0].cpu().numpy(), base + planner.linear_apply(Wp, Xi, ALL)[0], st_e["lam"], st_p["lam"]
    import time
    tk = time.time()
    We, Wp, prior, le, lp = ridge_late(F, 20)
    info["K0"] = {"lam_ego": le, "lam_op": lp, "n_train": int(len(tr)), "wall_s": time.time() - tk}
    heads["K0"] = {"ego_mu": em, "ego_sd": es, "op_mu": om, "op_sd": osd, "We": We, "Wp": Wp}
    preds["K0"] = prior.reshape(N, 20, 2).cpu().double().numpy()
    if "K1" in levels or "K3" in levels:
        tk = time.time()
        rt = np.zeros((N, ROUTE_N, 2), np.float32)
        rt[:n] = D["route"]
        Re, Rp, rpred, le, lp = ridge_late(torch.as_tensor(rt.reshape(N, -1), device=dev), ROUTE_N)
        t_route = time.time() - tk
        ids, w = np.zeros((N, 2), np.int64), np.zeros((N, 2), np.float32)
        ids[:n], w[:n] = two_hot(D["speed"])
        tgt = (ids, w)
        a, b = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(tr, groups=seq[tr]))
        fit_r, sel_r = tr[a], tr[b]
        ti, tw = torch.as_tensor(ids, device=dev), torch.as_tensor(w, device=dev)

        def ce(W, X, rows, off=None):               # mean two-hot cross-entropy of every lam's W on `rows`
            z = planner.linear_apply(W, X, rows)
            if off is not None:
                z = z + off[rows]
            lp_ = z.log_softmax(-1)
            return (-(lp_.gather(2, ti[rows].unsqueeze(0).expand(len(W), -1, -1)) * tw[rows]).sum(-1).mean(-1)).cpu().numpy()

        def pick(X, off=None, what="cls"):
            W, _ = planner.ce_solve(X, tgt, fit_r, planner.LAM_CLS, len(SPEEDS), offset=off)
            return float(planner.LAM_CLS[planner._pick(ce(W, X, sel_r, off), planner.LAM_CLS, what)])
        lam_e = pick(Xe, what="speed ego")
        Ce, _ = planner.ce_solve(Xe, tgt, tr, [lam_e], len(SPEEDS))
        off = planner.linear_apply(Ce, Xe, ALL)[0]
        inner = H._group_folds(seq[tr], 5, seed=0)
        for k in range(5):                          # night2_n3.p5cls: out-of-fold ego logits on the training rows
            Wk, _ = planner.ce_solve(Xe, tgt, tr[inner != k], [lam_e], len(SPEEDS))
            off[tr[inner == k]] = planner.linear_apply(Wk, Xe, tr[inner == k])[0]
        lam_l = pick(Xi, off, "speed late")
        Cp, st_c = planner.ce_solve(Xi, tgt, tr, [lam_l], len(SPEEDS), offset=off)
        info["K1"] = {"lam_route_ego": le, "lam_route_op": lp, "lam_speed_ego": lam_e, "lam_speed_op": lam_l, **st_c,
                      "wall_s_route": t_route, "wall_s_speed": time.time() - tk - t_route}
        k1 = {"ego_mu": em, "ego_sd": es, "op_mu": om, "op_sd": osd, "Re": Re, "Rp": Rp,
              "Ce": Ce[0].cpu().numpy(), "Cp": Cp[0].cpu().numpy(), "speeds": SPEEDS}
        heads["K1"] = heads["K2"] = k1
        hk = KHead.__new__(KHead)
        hk.p = {("K1", "x"): k1}
        route, v = hk.k1("x", D["ego"], D["op"])
        preds["K1"] = preds["K2"] = path_from_route(route, v)
        info["K1"]["torch_vs_numpy_route_max"] = float(np.abs(route.reshape(N, -1) - rpred.cpu().numpy()).max())
        preds["_v"], preds["_route"] = v, route
    if "K3" in levels:
        tk = time.time()
        Q = torch.as_tensor(np.r_[D["Q"], np.zeros((N - n, D["Q"].shape[1]), np.float32)], device=dev)
        ip, im, grp = D["ip"][pair_keep], D["im"][pair_keep], D["grp"][pair_keep]
        Rpair = (F[ip] - F[im]) - (prior[ip] - prior[im])
        Z = torch.cat([MC._std(Q, tr), MC._std(Xop, tr)], 1)
        zbar = Z[tr].mean(0)
        Zc, Dz, mu = Z[tr] - zbar, Z[ip] - Z[im], len(ip) / len(tr)
        score = np.zeros(len(MC.LAMS))
        for a_, b_ in MC._inner_splits(grp):
            Ws = MC._solve_pair(Dz[a_], Rpair[a_], Zc, mu, MC.LAMS)
            score += [float(((Dz[b_] @ W - Rpair[b_]) ** 2).sum()) for W in Ws]
        best = int(np.argmin(score))
        W = MC._solve_pair(Dz, Rpair, Zc, mu, [MC.LAMS[best]])[0]
        delta = ((Z[rows_k] - zbar) @ W).reshape(-1, 20, 2).cpu().double().numpy()
        g = g3(D["lead"][rows_k], D["lead_prob"][rows_k], v_ego_of(D["ego"][rows_k])).astype(np.float64)
        c = (g[:, None, None] * delta).sum(0) / g.sum()
        c = (c * np.array([1.0, 0.0])).astype(np.float64)
        heads["K3"] = {**heads["K1"], "c": c, "ttc": np.array([TTC_ON, TTC_OFF, CLOSING])}
        info["K3"] = {"lam_mc": float(MC.LAMS[best]), "lam_edge": best in (0, len(MC.LAMS) - 1), "n_pair": int(len(ip)),
                      "g3_sum": float(g.sum()), "g3_open": float((g > 0.5).mean()), "c_x_2s": float(c[7, 0]), "c_x_4s": float(c[15, 0]),
                      "wall_s": time.time() - tk}
        gall = g3(D["lead"], D["lead_prob"], v_ego_of(D["ego"])).astype(np.float64)
        preds["K3"] = path_from_route(preds["_route"], preds["_v"], gall[:, None] * c[:, 0])
        preds["_g3"] = gall
    for k, v in info.items():
        rl.event("k_fit", fold=tag, level=k, **v)
        rl.log.info("%s %s: %s", tag, k, v)
    return heads, preds, info


def _fold_masks(D, tag):
    role, n = D["t"].role.to_numpy(), D["n"]
    own = np.ones(n, bool) if tag == "full" else D["fold"] == tag
    return np.flatnonzero((role == "train") & own), np.flatnonzero(own), own[D["ip"]]


def _save_heads(heads: dict, tag: str):
    for level, h in heads.items():
        d = kdir(level, tag)
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / "head.npz", **{k: np.asarray(v) for k, v in h.items()})


def _numpy_check(D, tag) -> dict:
    """The saved heads through KHead (what the server runs) against the fit's own predictions."""
    kh = KHead()
    ego, op = D["ego"], D["op"]
    return {"K0": float(np.nanmax(np.abs(kh.predict("K0", tag, ego, op)[0] - PRED[tag]["K0"]))),
            "K1": float(np.nanmax(np.abs(kh.predict("K1", tag, ego, op)[0] - PRED[tag]["K1"]))),
            "K3": float(np.nanmax(np.abs(kh.predict("K3", tag, ego, op, D["lead"], D["lead_prob"])[0] - PRED[tag]["K3"])))}


PRED: dict = {}


def fit(eigh: str = "cuda"):
    """[K] 17:30 (3)-(7): R1, R2 and the full-data control; heads -> <level>/<fold>/head.npz; cross-fitted open-loop
    predictions -> openloop/; the full K0 against lane B's heads.npz."""
    import time

    import pandas as pd
    from . import nq3_cl as CL
    from .runlog import RunLog
    rl = RunLog("nq4_k", "fit")
    _gpu_eigh(eigh)
    t0 = time.time()
    D = data()
    rl.log.info("data: %d BA rows, %d I3, %d P6 (%.0f s)", D["n"], D["n3"], D["n6"], time.time() - t0)
    info, timing = {}, {}
    for tag in ("R1", "R2", "full"):
        t1 = time.time()
        tr, rows_k, pk = _fold_masks(D, tag)
        heads, PRED[tag], info[tag] = fit_fold(D, tr, rows_k, pk, rl, tag)
        _save_heads(heads, tag)
        timing[tag] = time.time() - t1
        info[tag]["numpy_vs_fit"] = _numpy_check(D, tag)
        rl.log.info("%s: %.0f s, numpy apply vs fit max |diff| %s", tag, timing[tag], info[tag]["numpy_vs_fit"])
        assert max(info[tag]["numpy_vs_fit"].values()) < 1e-3, info[tag]["numpy_vs_fit"]
    # full-data K0 = CL3's recipe: lane B's heads.npz must come out again
    H = CL.Heads()
    n = D["n"]
    ref = H.prior(D["ego"][:n], D["op"][:n]).reshape(n, 20, 2)
    lb = {"max_abs_pred": float(np.abs(ref - PRED["full"]["K0"][:n]).max()),
          "We_max_abs": float(np.abs(H.p["We"] - np.load(kdir("K0", "full", "head.npz"))["We"]).max()),
          "Wp_max_abs": float(np.abs(H.p["Wp"] - np.load(kdir("K0", "full", "head.npz"))["Wp"]).max()),
          "lam": [info["full"]["K0"]["lam_ego"], info["full"]["K0"]["lam_op"],
                  json.loads((CL.head_dir() / "meta.json").read_text())["lam_ego"], json.loads((CL.head_dir() / "meta.json").read_text())["lam_op"]]}
    rl.log.info("full K0 vs lane B heads.npz: %s", lb)
    assert lb["max_abs_pred"] <= 1e-3, lb
    export(D, rl)
    # [K] 17:30 (3): on the same frames the cross-fitted K0 differs from CL3's full-data head only through its training set
    other = np.where(D["fold"] == "R1", "R2", "R1")
    xf = np.where((other == "R1")[:, None, None], PRED["R1"]["K0"][:n], PRED["R2"]["K0"][:n])
    dd = np.abs(xf - ref).max((1, 2))
    lb["xfit_unseen_vs_laneB_per_row_max_abs_m"] = {"p50": float(np.median(dd)), "p95": float(np.percentile(dd, 95)),
                                                    "max": float(dd.max())}
    res = {"info": info, "timing_s": timing, "k0_full_vs_laneB": lb, "eigh": eigh}
    kdir("checks").mkdir(parents=True, exist_ok=True)
    (kdir("checks") / "fit.json").write_text(json.dumps(res, indent=1, default=float))
    rl.close()


def export(D, rl):
    """[K] 17:30 (7): per level, BA rows by the readout that never saw their route (and by their own fold: seen), I3 by
    R1 (R2 kept), P6 exam rows by the readout the closed loop would use on their route."""
    n, n3, n6 = D["n"], D["n3"], D["n6"]
    sp = D["split"]
    out = kdir("openloop")
    out.mkdir(parents=True, exist_ok=True)
    other = np.where(D["fold"] == "R1", "R2", "R1")
    r6 = np.array([readout(sp, str(b).split("-")[0].split("_")[0], "unseen") for b in D["t6"].base_id])
    for level in LEVELS:
        P = {f: PRED[f][level] for f in ("R1", "R2", "full")}
        unseen = np.where((other == "R1")[:, None, None], P["R1"][:n], P["R2"][:n])
        seen = np.where((D["fold"] == "R1")[:, None, None], P["R1"][:n], P["R2"][:n])
        aux = {}
        for key in ("_v", "_g3"):
            if key in PRED["R1"] and level in ("K1", "K2", "K3") and (key == "_v" or level == "K3"):
                aux[key[1:] + "_unseen"] = np.where(other == "R1", PRED["R1"][key][:n], PRED["R2"][key][:n]).astype(np.float32)
        np.savez_compressed(out / f"ba_{level}.npz", frame_name=D["t"].frame_name.to_numpy().astype(str), unseen=unseen.astype(np.float32),
                            seen=seen.astype(np.float32), full=P["full"][:n].astype(np.float32), readout_unseen=other,
                            **aux)
        np.savez_compressed(out / f"i3_{level}.npz", frame_name=D["t3"].frame_name.to_numpy().astype(str),
                            R1=P["R1"][n:n + n3].astype(np.float32), R2=P["R2"][n:n + n3].astype(np.float32))
        p6 = np.where((r6 == "R1")[:, None, None], P["R1"][n + n3:], P["R2"][n + n3:])
        np.savez_compressed(out / f"p6_{level}.npz", frame_name=D["t6"].frame_name.to_numpy().astype(str), unseen=p6.astype(np.float32),
                            readout=r6, R1=P["R1"][n + n3:].astype(np.float32), R2=P["R2"][n + n3:].astype(np.float32))
    rl.log.info("open-loop exports -> %s (P6 readouts %s)", out, dict(zip(*np.unique(r6, return_counts=True))))


def export_p6():
    """K3 on the P6 exam rows once their lead re-run exists (the fit wrote NaN there)."""
    from . import elicit_i3 as I, nq3_q1 as Q1, p5_exam as E, p5_openpilot as PO
    t6, past6 = Q1._p6_rows()
    with I.p5_set(SET_P6):
        op6 = PO.load(t6, (MODEL,), sub="op_streams_vis")[f"op-{MODEL} temporal"]
        ld = PO.load(t6, (MODEL,), ("lead", "lead_prob"), sub="op_streams_lead")
    ego6 = E.ego_input(t6, past6).astype(np.float32)
    kh = KHead()
    z = dict(np.load(kdir("openloop", "p6_K3.npz"), allow_pickle=True))
    assert (z["frame_name"].astype(str) == t6.frame_name.to_numpy()).all()
    for f in FOLDS:
        z[f] = kh.predict("K3", f, ego6, op6, ld[f"op-{MODEL} lead"], ld[f"op-{MODEL} lead_prob"])[0].astype(np.float32)
    z["unseen"] = np.where((z["readout"] == "R1")[:, None, None], z["R1"], z["R2"])
    np.savez_compressed(kdir("openloop", "p6_K3.npz"), **z)
    print({"p6_rows": len(t6), "nan_left": int(np.isnan(z["unseen"]).any((1, 2)).sum())})


def check_eigh():
    """[K] 17:30 (K3 step): fold R1 fitted with the float64 grams diagonalised on the CPU (the stock path) and on the
    GPU; predictions must agree within 1 mm; the wall times are the before / after of the optimisation."""
    import time

    from .runlog import RunLog
    rl = RunLog("nq4_k", "check-eigh")
    D = data()
    tr, rows_k, pk = _fold_masks(D, "R1")
    res = {}
    from . import planner
    stock = planner.gram_eigh
    for dev, ridge in (("cuda", False), ("cpu", False), ("cuda", True)):
        planner.gram_eigh = stock
        _gpu_eigh(dev, ridge)
        tag = dev + ("+ridge" if ridge else "")
        t0 = time.time()
        _, PRED[tag], inf = fit_fold(D, tr, rows_k, pk, rl, f"R1-{tag}")
        res[f"wall_s_{tag}"] = time.time() - t0
        res[f"sections_{tag}"] = {k: {kk: vv for kk, vv in v.items() if kk.startswith("wall")} for k, v in inf.items()}
    planner.gram_eigh = stock
    for a, b in (("cuda", "cpu"), ("cuda+ridge", "cpu")):
        for level in ("K0", "K1", "K3"):
            res[f"{a}_vs_{b}_{level}_max_abs_m"] = float(np.nanmax(np.abs(PRED[a][level] - PRED[b][level])))
        res[f"{a}_vs_{b}_c_max_abs_m"] = float(np.nanmax(np.abs(PRED[a]["K3"] - PRED[b]["K3"])))
    rl.log.info("GPU vs CPU eigh: %s", res)
    kdir("checks").mkdir(parents=True, exist_ok=True)
    (kdir("checks") / "eigh.json").write_text(json.dumps(res, indent=1))
    rl.close()
    assert max(res[f"cuda_vs_cpu_{lv}_max_abs_m"] for lv in ("K0", "K1", "K3")) <= 1e-3, res


def cl_verdict(d: Path) -> dict:
    """K5: the offline recomputation (scripts/nq3_cl_check.py op) identical on every request of every K arm, the fold
    the agent used = the split's unseen readout, and the TFv6 rules replayed from ticks.jsonl identical on every tick."""
    d = Path(d)
    sp = load_split()
    res = {"model": json.loads((d / "check_op.json").read_text()), "rules": [], "folds": []}
    ok = True
    for r in res["model"]:
        n = r["requests"]
        good = n > 0 and r["img2"] == n and r["op"] == n and r["path"] == n and r.get("lead", n) == n
        r["identical"] = good
        ok &= good
        a = Path(r["attempt"])
        arm, rid = a.parents[2].name, a.parents[0].name
        fs = sorted((a / "frames").glob("*.npz"))
        with np.load(fs[0]) as z:
            used = str(z["kfold"]) if "kfold" in z.files else None
        want = readout(sp, rid, "unseen")
        res["folds"].append({"arm": arm, "route": rid, "kfold": used, "expected": want})
        ok &= used == want
        if arm in ("k2", "k3"):
            ticks = [json.loads(line) for line in (a / "ticks.jsonl").read_text().splitlines() if line.strip()]
            rr = replay_rules(ticks)
            rr.update(arm=arm, route=rid, stop_ticks=sum(1 for t in ticks if (t.get("rules") or {}).get("stop_dist") is not None
                                                         and t["rules"]["stop_dist"] < Tfv6Rules.STOP_DIST),
                      rule_changed=sum(1 for t in ticks if t.get("rules") and t["rules"]["rule_in"] != t["rules"]["rule_out"]))
            res["rules"].append(rr)
            ok &= rr["ticks"] > 0 and rr["different"] == 0
    res["pass"] = bool(ok)
    (d / "verdict.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"pass": res["pass"], "rules": res["rules"], "folds": res["folds"]}))
    if not ok:
        raise SystemExit("rule-8 check failed")
    return res


def ready() -> str:
    """runs/nq4/k/READY: weights, agent modes, output convention and the checks behind them; refuses unless every
    check file exists and passed."""
    import pandas as pd
    ck = kdir("checks")
    fitj = json.loads((ck / "fit.json").read_text())
    eig = json.loads((ck / "eigh.json").read_text())
    ver = json.loads(kdir("steps", "cl", "verdict.json").read_text())
    leads = [json.loads((ck / f"lead_vs_vis_{s_}.json").read_text()) for s_ in (SET_BA, SET_P6)]
    assert ver["pass"] and all(x["different"] == 0 for x in leads)
    assert fitj["k0_full_vs_laneB"]["max_abs_pred"] <= 1e-3
    gpu = kdir("steps", "cl", "gpu").read_text().strip()
    lines = [f"# K-prep READY {pd.Timestamp.now(tz='Asia/Shanghai'):%Y-%m-%d %H:%M} CST (todos/2026-09-26-night-queue-4.md, K)", "",
             "## Weights (numpy apply: jevdrive.nq4_k.KHead; one set per level and fold, the fits are deterministic)"]
    for lv in LEVELS:
        lines.append(f"- {lv}: " + ", ".join(str(kdir(lv, f, "head.npz")) for f in (*FOLDS, "full"))
                     + ("   (K2 = K1's weights; the rules live in the agent)" if lv == "K2" else ""))
    lines += ["", "## Agent modes (scripts/b2d_zeroshot_agent.py, model head) and server",
              '- config: {"model": "head", "arm": "k0"|"k1"|"k2"|"k3", "k_view": "unseen"|"seen", "k_split": "' + str(kdir("route_split.json"))
              + '", "socket": <head server socket>, "warmup_s": 5.0, "desire": true, "head_cam_tick": 0.0, "controller": "fixed", '
              '"controller_preset": "pursuit", "controller_config": todos/2026-09-23-tfv6-controller/controller-eval/P7.json, "seed": <TM seed>, "dump_every": 0}',
              "- route python: envs/scout-tfv6 (as lane B's head arms); the fold is chosen per route from BENCHMARK_ROUTE_ID:"
              " unseen = the readout that never saw the route's recordings (never-recorded routes: R1), seen = its own (recorded routes only; the agent refuses otherwise)",
              "- k2 / k3: TFv6 rules (LEAD 730bc1a creeping + stop sign) on P7's throttle / brake, logged per tick in ticks.jsonl (key rules)",
              "- head server: CUDA_VISIBLE_DEVICES=<g> envs/openpilot/bin/python scripts/nq3_cl_server.py --pool <workers> --socket <S> "
              "(no Qwen / YOLO server needed; one server serves all four K arms and both folds)",
              "- plans.jsonl gets v_target (K1-K3) and g3 (K3) per plan from the server's info", "",
              "## Output convention",
              "- path (20, 2) float64, rear-axle frame, x forward, y left, t = 0.25 ... 5.0 s, handed to P7 as lane B's head arms",
              "- K0 = ridge ego + ridge_late (CL3's recipe); K1 = TFv6 route (10 checkpoints, 2.5 m + 1 m steps) + target speed "
              "(8 classes, two-hot expectation), path at s = v t; K2 = K1 + rules; K3 = K2 + g3 x c along the path", "",
              "## Checks",
              f"- lead re-run temporal vs stored op_streams_vis: {[{k: x[k] for k in ('set', 'streams', 'rows', 'different')} for x in leads]}",
              f"- full-data K0 vs lane B heads.npz: {fitj['k0_full_vs_laneB']}",
              f"- numpy apply vs fit, max |diff| m: { {t: fitj['info'][t]['numpy_vs_fit'] for t in fitj['info']} }",
              f"- GPU vs CPU eigh (fold R1): { {k: v for k, v in eig.items() if 'max_abs' in k or k.startswith('wall')} }",
              f"- closed-loop rule 8 (GPU {gpu}, routes {sorted({x['route'] for x in ver['folds']})}): pass = {ver['pass']}; "
              + "; ".join(f"{r['attempt'].split('/steps/cl/')[-1]}: {r['requests']} requests identical {r['identical']}" for r in ver["model"]),
              f"- rules replay: {ver['rules']}", f"- folds used: {ver['folds']}", "",
              "## Pilot sanity checklist per level (user rule: 1 route, then a 10-route pilot, then the batch; F's chain applies it)",
              "Pilot = the same 10 routes for every level, run unseen with TM seed 0, next to CL3 (lane B, seed 0) on those routes. Stop the level and write ERROR on any failed item:",
              "1. completion: all 10 routes end with a route result (CARLA crashes retried by b2d_run, max 2 attempts); agent_summary.json shows arm, k_view and n_plans > 0; no Python traceback in any route log",
              "2. blocked: routes ending in 'Agent got blocked' <= CL3's count on the same 10 routes + 2",
              "3. DS: the level's mean DS over the 10 routes >= CL3's mean on the same routes - 25 (a far lower mean means a wrong fold, head file or path convention, not a recipe effect)",
              "4. plans.jsonl: every path finite; K1-K3: v_target within [0, 20] m/s and its median over plans with speed > 2 m/s above 2 m/s",
              "5. K3: g3 within [0, 1]; the share of plans with g3 > 0.5 over the 10 routes between 0.5% and 20% (open-loop on P5 v1 BA: 3.7-5.0% per fold); 0% or > 30% = gate miswired",
              "6. K2 / K3: every tick of ticks.jsonl has a 'rules' record; ticks where the rules changed throttle / brake < 5% of ticks outside the stop-sign routes",
              "7. seen runs (K0, K3): the agent refuses a never-recorded route, so a seen batch lists only recorded routes (route_split.json, recorded = true)", "",
              "## Open-loop exports (capability readouts, not judged here)",
              f"- {kdir('openloop')}/ba_<level>.npz (unseen / seen / full, readout_unseen, v / g3), i3_<level>.npz (R1, R2; main = R1), "
              "p6_<level>.npz (unseen by the route's readout, R1, R2)", ""]
    txt = "\n".join(lines)
    kdir("READY").write_text(txt)
    print(txt)
    return txt


# ================================================================ entry point

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("split", "check-lead", "labels", "fit", "check-eigh", "export-p6", "cl-verdict", "ready"))
    ap.add_argument("--dir", default="")
    ap.add_argument("--set", default=SET_BA)
    a = ap.parse_args()
    if a.step == "split":
        split()
    elif a.step == "check-lead":
        check_lead(a.set)
    elif a.step == "labels":
        labels()
    elif a.step == "fit":
        fit()
    elif a.step == "check-eigh":
        check_eigh()
    elif a.step == "export-p6":
        export_p6()
    elif a.step == "cl-verdict":
        cl_verdict(Path(a.dir))
    elif a.step == "ready":
        ready()


if __name__ == "__main__":
    main()
