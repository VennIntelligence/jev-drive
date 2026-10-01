"""Unit checks of the S_jev scorer (experiments/op_adapt_r2/archive/op_adapt_score.py). `python -m pytest experiments/op_adapt_r2/archive/test_op_adapt_score.py` or
`python experiments/op_adapt_r2/archive/test_op_adapt_score.py` (no pytest needed)."""
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.op_adapt_r2.archive import op_adapt_score as S  # noqa: E402

EGO = S.EGO["carla"]


def _map(junction=True, oneway=False):
    """Straight two-lane road along +x (|y| <= 3.5 m: y < 0 the ego's lane, heading 0; y > 0 the oncoming lane, heading
    pi) from x = -50 to 200 m, plus (junction) a 20 m square crossing at x in [40, 60] with a road going +y."""
    from experiments.op_adapt_r2.archive.op_adapt_score_data import rasterize
    polys = [(np.array([[-50, -3.5], [200, -3.5], [200, 0], [-50, 0]]), 0.0, 1, S.F_DRIVE | S.F_LANE),
             (np.array([[-50, 0], [200, 0], [200, 3.5], [-50, 3.5]]), 0.0 if oneway else math.pi, 2, S.F_DRIVE | S.F_LANE)]
    if junction:
        polys += [(np.array([[40, -10], [60, -10], [60, 10], [40, 10]]), None, 0, S.F_DRIVE | S.F_JUNC),
                  (np.array([[46.5, 10], [50, 10], [50, 120], [46.5, 120]]), math.pi / 2, 3, S.F_DRIVE | S.F_LANE)]
    return rasterize(polys)


def _slot(actors=None, v0=10.0, n=S.NT, mapq=None, ref=None):
    ref = np.column_stack([np.linspace(0, 200, 401), np.zeros(401)]) if ref is None else ref
    return S.Slot(EGO, (0.0, -1.75, 0.0), v0, n, actors or S.Actors.empty(), ref, mapq or _map())


def _const(v, y=0.0, n=S.NT):
    return np.column_stack([v * S.TS[:n], np.full(n, y)])


def _actor(xy, kind=S.PED, hl=0.3, hw=0.3, speed=0.0, h=0.0, vel=(0.0, 0.0), still=0.0):
    c = np.asarray(xy, float)[None] + np.outer(S.TS, vel)
    return S.Actors(c[:, None], np.full((S.NT, 1), h), np.array([hl]), np.array([hw]), c[:, None].copy(),
                    np.ones((S.NT, 1), bool), np.full((S.NT, 1), speed), np.array([kind]), np.array([True]), np.array([still]))


def _cat(*a):
    return S.Actors(*[np.concatenate([getattr(x, f) for x in a], axis=1 if getattr(a[0], f).ndim >= 2 and f not in ("hl", "hw", "kind", "vis") else 0)
                      for f in ("c", "h", "hl", "hw", "ref", "valid", "speed", "kind", "vis")])


def test_box_overlap():
    o = S.obb_overlap
    assert o((0, 0), 0, 2, 1, (3.9, 0), 0, 2, 1) and not o((0, 0), 0, 2, 1, (4.1, 0), 0, 2, 1)
    assert not o((0, 0), 0, 2, 1, (0, 2.1), 0, 2, 1) and o((0, 0), 0, 2, 1, (0, 1.9), 0, 2, 1)
    # rotated 45 deg: corner reach sqrt(2) * 1 for a 1 x 1 half-size box
    assert o((0, 0), 0, 1, 1, (2.4, 0), math.pi / 4, 1, 1) and not o((0, 0), 0, 1, 1, (2.45, 0), math.pi / 4, 1, 1)
    # a diagonal miss that the circumscribed circles would call a hit
    assert not o((0, 0), 0, 1, 1, (2.1, 2.1), 0, 1, 1)


def test_dac_straight_and_junction():
    s = _slot()
    p = np.stack([_const(10.0), _const(10.0, 4.5), _const(10.0, -2.0)])     # in lane / over the far edge / off right
    m = S.raw_metrics(s, p)
    assert list(m["DAC"]) == [True, False, False], m["DAC"]
    # a left turn through the junction onto the +y road stays drivable; the same turn cut 8 m early leaves the road
    th = np.clip((S.TS - 1.0) / 2.0, 0, 1) * math.pi / 2
    def turn(x0):
        r = 5.0
        x = np.where(S.TS < 1.0, 10 * S.TS * (x0 / 10), x0 + r * np.sin(th))
        y = np.where(S.TS < 1.0, 0.0, r * (1 - np.cos(th)))
        y = np.where(S.TS > 3.0, y + 8 * (S.TS - 3.0), y)
        return np.column_stack([x, y])
    s2 = S.Slot(EGO, (0.0, -1.75, 0.0), 10.0, S.NT, S.Actors.empty(), s.ref, _map())
    ok, bad = turn(44.0), turn(30.0)
    m = S.raw_metrics(s2, np.stack([ok, bad]))
    assert m["DAC"][0] and not m["DAC"][1], m["DAC"]


def test_ddc_levels_and_junction():
    s = _slot(v0=8.0)
    def move_over(dy, t0=0.5, dur=0.5):                   # into the oncoming lane (y + dy) and stay
        ramp = np.clip((S.TS - t0) / dur, 0, 1)
        return np.column_stack([8.0 * S.TS, dy * ramp])
    brief = np.column_stack([8.0 * S.TS, 2.2 * np.clip(1 - np.abs(S.TS - 1.0) / 0.15, 0, 1)])  # centre 0.45 m over the line, < 0.1 s
    p = np.stack([_const(8.0), move_over(3.5), brief])
    m = S.raw_metrics(s, p)
    assert m["DDC"][0] == 1.0 and m["DDC"][1] == 0.0 and m["DDC"][2] == 1.0, (m["DDC"], m["D_onc"])
    # driving against the lane: 1.1 s windows of 1.65 / 2.75 / 8.8 m -> 1 / 0.5 / 0
    for v, want in ((1.5, 1.0), (2.5, 0.5), (8.0, 0.0)):
        so = S.Slot(EGO, (0.0, 1.75, 0.0), v, S.NT, S.Actors.empty(), s.ref, _map())
        m = S.raw_metrics(so, _const(v)[None])
        assert m["DDC"][0] == want, (v, m["D_onc"])
    # inside the junction the same wrong-way travel is exempt
    sj = S.Slot(EGO, (40.5, 1.75, 0.0), 2.5, 30, S.Actors.empty(), s.ref, _map())
    m = S.raw_metrics(sj, _const(2.5)[None])
    assert m["DDC"][0] == 1.0, m["D_onc"]


def _cands(v0, route=None):
    plan = np.zeros((33, 15))
    plan[:, 0] = v0 * S.T_IDXS
    plan[:, 3] = v0
    return S.candidates({"op": plan, "op_L": plan, "op_R": plan}, v0, 1.5, route=route)


def test_yield_exemption_and_static_obstacle():
    ref = np.column_stack([np.linspace(0, 200, 401), np.zeros(401)])
    P, _ = _cands(10.0, ref)
    ped = _actor((25.0, 0.0), S.PED)
    s = _slot(ped)
    r = S.score_slot(s, P, S.CANDS)
    i = {k: S.CANDS.index(k) for k in S.CANDS}
    assert not r["NC"][i["op"]] and not r["NC"][i["hold"]] and r["exempt"]
    assert r["NC"][i["op_stop"]] and r["P"][i["op_stop"]] == 1.0, r["P"]          # stopping for the person: full progress
    assert r["gate"] == 1 and (r["P"] == 1.0).all()                                 # v5: P = 1 for every candidate
    assert r["NC"][i["shift_L"]] and r["P"][i["shift_L"]] == 1.0                   # the detour is capped at 1 as well
    car = _actor((25.0, 0.0), S.STATIC, hl=2.3, hw=0.9, still=8.0)
    r2 = S.score_slot(_slot(car, mapq=_map(oneway=True)), P, S.CANDS)      # a free same-direction lane on the left
    assert not r2["exempt"] and r2["NC"][i["shift_L"]] and r2["P"][i["op_stop"]] < 0.5, r2["P"]   # waiting behind a car parked >= T_W is penalised
    for st in (0.0, 4.9):                                                    # dwell gate: a blocker that has waited < T_W is waited for
        r3 = S.score_slot(_slot(_actor((25.0, 0.0), S.STATIC, hl=2.3, hw=0.9, still=st), mapq=_map(oneway=True)), P, S.CANDS)
        assert r3["exempt"] and r3["gate"] == 2 and (r3["P"] == 1.0).all(), (st, r3["P"])
    r4 = S.score_slot(_slot(_cat(car, _actor((25.0, 0.0), S.PED)), mapq=_map(oneway=True)), P, S.CANDS)
    assert r4["gate"] == 1 and (r4["P"] == 1.0).all()                          # a moving blocker wins over a long-static one
    mv = _actor((25.0, 0.0), S.VEH, hl=2.3, hw=0.9, speed=2.0, vel=(2.0, 0.0))
    assert S.score_slot(_slot(mv), P, S.CANDS)["exempt"]                          # a moving car earns the exemption


def test_progress_floor():
    P, _ = _cands(1.0)
    r = S.score_slot(_slot(v0=1.0), P, S.CANDS)
    assert r["prog"].max() < S.PROG_MIN and (r["P"] == 1.0).all()
    P, _ = _cands(10.0)
    r = S.score_slot(_slot(v0=10.0), P, S.CANDS)
    assert r["P"][S.CANDS.index("op_stop")] < 1.0


def test_ttc_and_comfort():
    stop = _actor((20.0, 0.0), S.VEH, hl=2.3, hw=0.9)
    p = np.stack([_const(10.0, 0, S.NT)])
    s = _slot(stop)
    m = S.raw_metrics(s, p)
    assert not m["TTC"][0]
    follow = _actor((-6.5, 0.0), S.VEH, hl=2.3, hw=0.9, speed=10.0, vel=(10.0, 0.0))    # a follower at the same speed
    side = _actor((20.0, 3.5), S.VEH, hl=2.3, hw=0.9)                                  # a car parked in the next lane
    assert S.raw_metrics(_slot(_cat(follow, side)), p)["TTC"][0]
    P, _ = _cands(12.0)
    m = S.raw_metrics(_slot(v0=12.0), P)
    c = dict(zip(S.CANDS, m["C"]))
    assert c["op"] and not c["brake_hard"], c


def test_static_geometry_in_nc_only_by_overlap():
    """v4: scene objects enter NC as boxes (ref = NaN keeps them out of the gap term)."""
    pole = _actor((20.0, 0.0), S.STATIC, hl=0.2, hw=0.2)
    pole.ref[:] = np.nan
    beside = _actor((20.0, 1.4), S.STATIC, hl=0.2, hw=0.2)          # 1.4 m left of the path: gap-lane but no overlap
    beside.ref[:] = np.nan
    p = np.stack([_const(10.0)])
    assert not S.raw_metrics(_slot(pole), p)["NC"][0]
    assert S.raw_metrics(_slot(beside), p)["NC"][0]


def test_candidates_op_rows_and_rej():
    plan = np.zeros((33, 15))
    plan[:, 0] = 9.0 * S.T_IDXS
    plan[:, 1] = 0.05 * S.T_IDXS ** 2
    plan[:, 3], plan[:, 6] = 9.0, 0.3
    P, trj = S.candidates({"op": plan, "op_L": plan, "op_R": plan}, 9.0, 1.2)
    assert np.array_equal(trj[0], np.stack([plan[:21, 0], plan[:21, 1], plan[:21, 3], plan[:21, 6]], -1).astype(np.float32))
    assert P.shape == (13, S.NT, 2) and np.allclose(P[0, 0], (0.0, 0.0))
    centre = np.column_stack([np.linspace(-10, 100, 221), np.full(221, -1.8)])       # centre line 1.8 m to the right
    P, _ = S.candidates({"op": plan, "op_L": plan, "op_R": plan}, 9.0, 1.2, centre=centre, names=S.CANDS_OFF)
    rej = P[-1]
    assert np.allclose(rej[0], 0, atol=1e-6) and abs(rej[31, 1] + 1.8) < 1e-6 and abs(rej[-1, 1] + 1.8) < 1e-6
    ahead = centre[centre[:, 0] >= 8.0]                          # PDM centre line that begins 8 m ahead of the start
    P, _ = S.candidates({"op": plan, "op_L": plan, "op_R": plan}, 9.0, 1.2, centre=ahead, names=S.CANDS_OFF)
    assert np.allclose(P[-1, 0], 0, atol=1e-6) and abs(P[-1, 31, 1] + 1.8) < 1e-6 and np.all(np.diff(P[-1, :, 1]) <= 1e-9)
    stand = np.zeros((33, 15))
    P, _ = S.candidates({"op": stand, "op_L": stand, "op_R": stand}, 0.0, 1.2, centre=centre, names=S.CANDS_OFF)
    assert np.allclose(P[-1], 0)                                  # no merge without moving


def test_nc_gap_matches_wl_gap_front():
    """NC's gap term against experiments.world_model.lib.wl._gap_front on synthetic CARLA runs (world y south, yaw clockwise; actors
    outside the 2 m self-exclusion radius so a front-half overlap always implies gap < 2 m as well)."""
    from experiments.world_model.lib import wl
    rng = np.random.default_rng(0)
    agree, n = 0, 0
    for trial in range(60):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            yaw = rng.uniform(-180, 180)
            c, s_ = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
            v = 8.0
            ticks = np.arange(1, 62)
            x = 100 + v * 0.05 * (ticks - 1) * c
            y = 50 + v * 0.05 * (ticks - 1) * s_
            with open(d / "pose.jsonl", "w") as f:
                for t, xx, yy in zip(ticks, x, y):
                    f.write(json.dumps({"frame": int(t + 99), "x": xx, "y": yy, "z": 0.0, "yaw": yaw, "vx": v * c, "vy": v * s_}) + "\n")
            with open(d / "frames.jsonl", "w") as f:
                for t in ticks:
                    f.write(json.dumps({"frame": int(t + 99), "tick": int(t)}) + "\n")
            # one actor, standing, somewhere around the path (ego frame: fwd, right)
            fx, ry = rng.uniform(3, 40), rng.uniform(-3, 3)
            ax, ay = 100 + fx * c - ry * s_, 50 + fx * s_ + ry * c
            np.savez(d / "actors.npz", frame=ticks + 99, id=np.full(len(ticks), 7), xyz=np.tile([ax, ay, 0.0], (len(ticks), 1)).astype(np.float32),
                     yaw=np.zeros(len(ticks), np.float32), v=np.zeros((len(ticks), 2), np.float32))
            (d / "actor_kinds.json").write_text(json.dumps({"7": ["walker.pedestrian.0001", "background", [0, 0, 0, 0.2, 0.2, 0.9]]}))
            ref = wl._gap_front(d, 1, 61)["gap_min_m"] < 2.0
            # the same run through the scorer (mirrored world; rear axle = transform - 1.389 m along the heading)
            h = -math.radians(yaw)
            px, py = x - EGO.gap_o * c, -(y - EGO.gap_o * s_)
            g = {"p": np.column_stack([px, py])[None], "h": np.full((1, 61), h), "v": np.full((1, 61), v)}
            g["u"] = np.stack([np.cos(g["h"]), np.sin(g["h"])], -1)
            A = S.Actors(np.tile([[ax, -ay]], (61, 1))[:, None], np.zeros((61, 1)), np.array([0.2]), np.array([0.2]),
                         np.tile([[ax, -ay]], (61, 1))[:, None], np.ones((61, 1), bool), np.zeros((61, 1)), np.array([S.PED]), np.array([True]))
            ok, _, _ = S.nc(g, EGO, A, 61)
            agree += int((not ok[0]) == ref)
            n += 1
    assert agree == n, f"{agree}/{n}"


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)


def test_yield_when_hold_passes_ahead():
    """v5: the moving pedestrian crosses 6 m ahead of a 10 m/s ego; hold gets past it (NC = 1) while op (slower) does not.
    Stopping to yield must not lose against hold on progress."""
    P, _ = _cands(10.0)
    i = {k: S.CANDS.index(k) for k in S.CANDS}
    ped = _actor((14.0, 4.0), S.PED, vel=(0.0, -1.5))
    r = S.score_slot(_slot(ped), P, S.CANDS)
    assert r["NC"][i["hold"]] and r["NC"][i["op_stop"]] and r["gate"] in (0, 1)
    if r["gate"] == 0:                                   # the person was not on op / hold's path: nothing to test
        return
    assert (r["P"] == 1.0).all()
