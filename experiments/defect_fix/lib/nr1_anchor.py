"""LON-ANCHOR: a station-anchored two-sided longitudinal arc term on the pool's S rows (rm_train --on-anchor W; prereg
experiments/defect_fix/plans/2026-10-20-lon-anchor-prereg.md). Carried by nr1_lat (the pool's one --on-lon module): `lon_anchor` =
10 x W x term, the trainer multiplies by ho_w and sums over the pool rows / n_imit, as every NR1 term.

Target (labels built offline into <pool>/lon_anchor.npz by scripts/lon_anchor.py labels): the ego is at station s_e on the human's
logged path; t_h = the first log time the human was at s_e; the target arc at the plan's time points tau_j = 0.5 j s (j = 1..8) is
the human's own travel from there over the same tau: s_t,j = S_h(t_h + tau_j) - s_e (S_h = the human's station, non-decreasing).
So the human's speed profile is started from where the ego is now, not from the same log time (JT's arc term, nr1_lon: the log
time's profile blended from the ego's speed over 2 s). No feasibility shaping, no windows.

Rows: S rows (tab mode `self`) whose label is defined: the ego's state inside the log's time span and not behind the human's start,
the human reaching s_e (t_h finite) and the log covering t_h + 4 s. One exclusion: a lead on the human's path (an object centre
within LEAD_LAT of the path at LEAD_LO .. LEAD_HI m ahead of s_e) closer to the ego at t than the human's lead was at t_h (by more
than GAP_TOL; the human's lead unknown counts as farther) -> the row is skipped (never pull toward a lead).

Term: d_j = s_plan,j - s_t,j (s_plan = the cumulative polyline arc of the plan's 8 poses from the origin), smooth L1 (beta 0.5) of d
summed over the 8 poses (= nr1_lon.terms with this target): too fast and too slow cost the same. Longitudinal only: each pose moves
only along its own heading (nr1_lon.lon_only), no gradient across it, none on the yaw: path shape untouched.
"""
from __future__ import annotations

import numpy as np

H_FUT, DT = 4.0, 0.5
LEAD_LAT, LEAD_LO, LEAD_HI, GAP_TOL = 1.8, 2.0, 60.0, 0.5
WHY = ("ok", "not_s", "no_path", "no_reach", "log_end", "lead", "lead_unknown")


def first_at(s_e, T, S):
    """The first log time (s) at which the human's station S (non-decreasing, at log times T) reaches s_e; inf if never."""
    S = np.maximum.accumulate(np.asarray(S, np.float64))
    if s_e > S[-1]:
        return np.inf
    i = int(np.searchsorted(S, s_e, "left"))
    if i == 0:
        return float(T[0])
    return float(T[i - 1] + (T[i] - T[i - 1]) * (s_e - S[i - 1]) / max(S[i] - S[i - 1], 1e-9))


def target(t, s_e, T, S):
    """One state (log time t s, ego station s_e) -> (s_t (8,) f8 or None, t_h, why)."""
    T, S = np.asarray(T, np.float64), np.maximum.accumulate(np.asarray(S, np.float64))
    if len(T) < 2 or not (T[0] <= t <= T[-1]) or s_e < S[0] - 1.0:
        return None, np.nan, "no_path"
    th = first_at(s_e, T, S)
    if not np.isfinite(th):
        return None, np.inf, "no_reach"
    if th + H_FUT > T[-1] + 1e-6:
        return None, th, "log_end"
    return np.interp(th + DT * np.arange(1, 9), T, S) - s_e, th, "ok"


def lead_gap(P, a, B):
    """Path points P (m, 2) at arcs a (m,) ahead of s_e (1 m grid), object centres B (k, 2) -> the smallest arc, among the objects
    whose nearest path point lies in [LEAD_LO, LEAD_HI] and within LEAD_LAT of the centre, of that nearest point (inf: none)."""
    m = (a >= LEAD_LO - 1.0) & (a <= LEAD_HI + 1.0)
    if not len(B) or not m.any():
        return np.inf
    d = np.hypot(B[:, None, 0] - P[None, m, 0], B[:, None, 1] - P[None, m, 1])
    j = d.argmin(1)
    an, dn = a[m][j], d[np.arange(len(B)), j]
    hit = (dn < LEAD_LAT) & (an >= LEAD_LO) & (an <= LEAD_HI)
    return float(an[hit].min()) if hit.any() else np.inf


def lead_skip(gap_e, gap_h):
    """-> why ("ok" / "lead" / "lead_unknown"); gap_h NaN = the human's lead could not be measured."""
    if not np.isfinite(gap_e):
        return "ok"
    if np.isnan(gap_h):
        return "lead_unknown"
    return "lead" if gap_e < gap_h - GAP_TOL else "ok"


def load(pool_dir, names, dev):
    """-> pack: s_t (N, 8), ok (N,) on dev from <pool_dir>/lon_anchor.npz (names = the pool's tab names in order)."""
    import torch
    from pathlib import Path
    z = np.load(Path(pool_dir) / "lon_anchor.npz")
    assert np.array_equal(z["names"].astype(str), np.asarray(names).astype(str)), "nr1_anchor: lon_anchor.npz rows != the pool's"
    ok, s_t = z["ok"].astype(bool), np.nan_to_num(z["s_t"]).astype(np.float32)
    assert np.isfinite(z["s_t"][ok]).all(), "nr1_anchor: non-finite target on an ok row"
    why = z["why"].astype(str)
    rec = dict(rows=int(len(ok)), anchor_rows=int(ok.sum()), why={w: int((why == w).sum()) for w in WHY if (why == w).any()})
    return dict(s_t=torch.as_tensor(s_t, device=dev), ok=torch.as_tensor(ok, device=dev), rec=rec)


def term(poses, rows, pack):
    import nr1_lon
    return nr1_lon.terms(poses, pack["s_t"][rows], pack["ok"][rows])


if __name__ == "__main__":                     # known-answer checks (CPU)
    import torch
    import nr1_lon
    T = np.arange(0, 20.01, 0.5)
    S = np.minimum(10.0 * T, 100.0)            # 10 m/s, then standing at 100 m from t = 10 s
    st, th, w = target(3.0, 40.0, T, S)        # ego at 40 m at t = 3 s (the human was there at 4 s): his 4 .. 8 s profile
    assert w == "ok" and abs(th - 4.0) < 1e-9 and np.allclose(st, 5.0 * np.arange(1, 9)), (st, th)
    st, th, w = target(3.0, 80.0, T, S)        # ego at 80 m: the human comes to rest at 100 m 2 s later
    assert w == "ok" and np.allclose(st, np.minimum(5.0 * np.arange(1, 9), 20.0)), st
    assert target(17.0, 90.0, T, 5.0 * T)[2] == "log_end" and target(3.0, 150.0, T, S)[2] == "no_reach" and target(25.0, 10.0, T, S)[2] == "no_path"
    P, a = np.c_[np.arange(0.0, 70.0), np.zeros(70)], np.arange(0.0, 70.0)
    assert lead_gap(P, a, np.array([[30.0, 1.0]])) == 30.0 and lead_gap(P, a, np.array([[30.0, 3.0]])) == np.inf
    assert lead_skip(20.0, 30.0) == "lead" and lead_skip(30.0, 30.2) == "ok" and lead_skip(np.inf, 5.0) == "ok" and lead_skip(5.0, np.nan) == "lead_unknown"
    t = torch.linspace(0.5, 4.0, 8)
    s_t = torch.as_tensor(np.minimum(5.0 * np.arange(1, 9), 20.0), dtype=torch.float32)[None]
    hum = torch.stack([s_t[0], torch.zeros(8), torch.zeros(8)], -1)[None]        # the human's own profile as the plan: ~0
    pack = dict(s_t=s_t, ok=torch.tensor([True]))
    assert term(hum, torch.tensor([0]), pack).abs().max() < 1e-2
    fast = torch.stack([10.0 * t, torch.zeros(8), torch.zeros(8)], -1)[None].requires_grad_(True)
    v = term(fast, torch.tensor([0]), pack)
    v.sum().backward()
    assert v > 10 and (fast.grad[0, :, 0] > -0.05).all() and fast.grad[0, 4:, 0].min() > 0.9 and fast.grad[0, :, 1:].abs().max() == 0      # too fast: pulled back, along x only
    slow = torch.stack([1.0 * t, torch.zeros(8), torch.zeros(8)], -1)[None]
    assert term(slow, torch.tensor([0]), pack) > 1                                  # two-sided: too slow costs too
    assert term(fast, torch.tensor([0]), dict(s_t=s_t, ok=torch.tensor([False]))) == 0
    print("nr1_anchor known-answer ok", float(v))
