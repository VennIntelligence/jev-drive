"""Unit checks of the WL-2 training and readout code (CPU only): folds, windows, leakage, ego sign, H, cg vs reg."""
import numpy as np
import pandas as pd

from experiments.world_model.archive import nq4_w as W
from experiments.world_model.archive import wl2_model as M2


def synth(n_routes=24, rows=40):
    """Rows of `n_routes` routes (3 datasets, 3 classes, 2 eval routes), 5 Hz segments, some intervention rows."""
    rng = np.random.RandomState(0)
    out = []
    for r in range(n_routes):
        ds = r % 3
        t = pd.DataFrame({"base_id": str(r), "route_id": f"r{r}", "ds": ds, "cls": ["ped", "cutin", "obstacle"][r % 3] if ds else np.nan,
                          "tick": np.arange(rows) * W.TICKS, "split": "eval" if r < 2 else "train"})
        t["src"] = (ds > 0) * (np.arange(rows) % 7 > 3)
        t["action"] = np.where(t.src == 1, rng.choice(["op", "shift_R", "op_slow", "shift_L_slow"], rows), None)
        out.append(t)
    m = pd.concat(out, ignore_index=True)
    key = m.route_id
    seg = (key != key.shift()) | (m.tick.diff() != W.TICKS)
    s = seg.cumsum()
    pos, n = m.groupby(s).cumcount(), m.groupby(s).route_id.transform("size")
    m["win_start"] = ((n - pos) >= W.WIN).to_numpy()
    return m


def test_folds_partition_and_stratify():
    m = synth()
    ev = {"0", "1"}
    f = M2.route_folds(m, ev)
    assert not set(f) & ev and len(f) == 22
    assert set(f.values()) <= set(range(5))
    assert all(np.bincount(list(f.values()), minlength=5) >= 4)          # 22 routes over 5 folds
    assert f == M2.route_folds(m, ev)                                    # deterministic


def test_windows_no_leakage_and_holdout():
    m = synth()
    ev = {"0", "1"}
    f = M2.route_folds(m, ev)
    for arm in ("A", "B", "W", "Bs", "Bh", "Bhc"):
        for seed in range(3):
            tr, va, trr, var = M2.rows_windows(m, arm, seed, ev, f, None)
            assert len(tr) and len(va)
            assert not set(m.base_id.iloc[tr]) & set(m.base_id.iloc[va])
            assert not set(m.base_id.iloc[np.r_[tr, va]]) & ev
            if arm == "W":
                assert (m.ds.iloc[tr] == 0).all()
            if arm in ("Bs",):
                assert (m.ds.iloc[tr] > 0).all()
            if arm == "Bh":
                fut = tr[:, None] + W.HIST - 1 + np.arange(W.FUT)
                assert not (m.action.isin(M2.HOLD["Bh"]).to_numpy()[fut] & (m.src.to_numpy()[fut] == 1)).any()
    a = M2.rows_windows(m, "B", 0, ev, f, None)[0]
    h = M2.rows_windows(m, "Bh", 0, ev, f, None)[0]
    assert len(h) < len(a) and set(h) <= set(a)


def test_crossfit_holds_out_the_fold_including_eval_routes():
    m = synth()
    ev = {"0", "1"}
    fx = M2.route_folds(m, ev, seed=1, include_eval=True)
    assert {"0", "1"} <= set(fx)
    tr, va, trr, var = M2.rows_windows(m, "Ax", 2, ev, None, fx)
    held = {b for b, k in fx.items() if k == 2}
    assert not set(m.base_id.iloc[np.r_[tr, va]]) & held and set(m.base_id.iloc[tr]) & ev


def test_ego_sign_left_positive():
    """A pose left of a straight route (CARLA: y right, yaw clockwise) has e_y > 0; a curve keeps the sign."""
    import json
    import tempfile
    from pathlib import Path
    from experiments.world_model.archive.wl_dryrun import _ego_group
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for name, R, poses in (("straight", [(x, 0.0) for x in range(0, 100, 1)], [(50.0, -1.0, 0.0), (50.0, 1.0, 0.0)]),
                               ("curve", [(30 * np.sin(a), 30 * (1 - np.cos(a))) for a in np.linspace(0, 1.5, 200)], None)):
            (d / name).mkdir()
            (d / name / "route.json").write_text(json.dumps([{"x": float(x), "y": float(y)} for x, y in R]))
            if poses is None:                                   # the route bends towards +y (CARLA right); a point 1 m further out is on its left
                a = 0.7
                cx, cy = 30 * np.sin(a), 30 * (1 - np.cos(a))
                nx, ny = -np.sin(a), np.cos(a)                  # towards the centre of the bend = right of the heading in CARLA
                poses = [(cx - nx * 1.0, cy - ny * 1.0, np.degrees(a))]
            pd.DataFrame([{"frame": i, "x": x, "y": y, "yaw": yaw} for i, (x, y, yaw) in enumerate(poses)]).to_json(d / name / "pose.jsonl", orient="records", lines=True)
        e = _ego_group((str(d / "straight"), np.array([0, 1])))
        assert e[0, 0] > 0.9 and e[1, 0] < -0.9 and abs(e[0, 1]) < 1e-3     # CARLA y points right: y = -1 is left of the +x heading
        e = _ego_group((str(d / "curve"), np.array([0])))
        assert e[0, 0] > 0.9 and abs(e[0, 1]) < 0.05


def test_cg_is_reg_minus_ttc():
    from experiments.world_model.archive import wl2_report as R
    t = pd.DataFrame({"collision": [True, False, False, False], "gap_min_m": [5.0, 1.0, 5.0, 5.0], "ttc_min_s": [9.0, 9.0, 0.5, 9.0],
                      "unsafe": [True, True, True, False]})
    cg = R.label_cg(t)
    assert cg.tolist() == [1.0, 1.0, 0.0, 0.0]
    assert ((t.unsafe.astype(float) - cg) > 0).tolist() == [False, False, True, False]     # differ only where the TTC term fires


def test_H():
    from experiments.world_model.archive import wl2_report as R
    x = pd.DataFrame({"U_op": [1.0, 1.0, 0.0, 1.0], "U_pick": [0.0, 1.0, 0.0, 1.0], "U_or": [0.0, 0.0, 0.0, 1.0]})
    assert abs(R.H(x) - (0.75 - 0.5) / (0.75 - 0.25)) < 1e-9
    assert np.isnan(R.H(x.assign(U_op=x.U_or)))
