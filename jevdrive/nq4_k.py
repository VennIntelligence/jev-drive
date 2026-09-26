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
        thr, brk, _ = r.step(rec["v"], k["rule_in"][0], k["rule_in"][1], lambda: occ,
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


# ================================================================ entry point

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("split", "check-lead"))
    ap.add_argument("--set", default=SET_BA)
    a = ap.parse_args()
    if a.step == "split":
        split()
    elif a.step == "check-lead":
        check_lead(a.set)


if __name__ == "__main__":
    main()
