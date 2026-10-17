"""SELF-ITER hard-example mining (experiments/self_iter): per-row open-loop errors of a policy against the human log, the behaviour class of
a row, and the draw weights rm_train uses with --si-scan (experiments/route_mem/scripts/rm_train.py, block SELF-ITER).

Errors (plan and human future: (n, 8, >= 2) rear-axle poses at 0.5 .. 4 s in the row's own frame): the error vector of each pose split along /
across the human path's unit tangent (central differences; a standing human -> +x); at 2 s (pose 3) and 4 s (pose 7) along-track `al2` / `al4`
(+ = the plan ahead of the human) and cross-track `cr2` / `cr4` (+ = left); `arc4` = the plan's 4 s cumulative arc length minus the human's (the
signed longitudinal bias, rotation invariant); `ade` = mean pose L2 (the dev_eval metric). The hardness rank uses `ade` unless --si-key says otherwise.

Classes (the human's own motion, one per row, first match): stopped (v0 < 0.5 and 4 s arc < 1 m), starting (v0 < 2 and end speed >= v0 + 1),
turning (|heading change at 4 s| >= 20 deg), braking (end speed <= v0 - 1.5), else cruising. d331's cells (cap_ledger led.cells_of) are stored
beside them by si.py select where the log could be labelled.

Draw (rm_train --si-scan FILE): the scan's `ade` (or --si-key) ranks the rows WITHIN each set (navtrain train rows; the new trainval rows of
the merged pool), never across them (the navtrain rows are the policy's own training rows, the trainval rows are unseen: their errors are not
comparable). --si-mode top: weight --si-mult on the top --si-top share of the set, 1 elsewhere; --si-mode rank: weight (rank / n) ^ --si-gamma
(rank 1 = the easiest), mean 1. Normal (navtrain) rows: numpy's weighted draw of the nN normal rows on rm_train's own stream; pool rows: the pool
draw's p, the base rows (`pool` == s, the S rows) keep the mass --si-k-s / --on-k (uniform inside), the new rows (`pool` == t, `hold` != 1)
take the rest, held-out logs (`hold` == 1) weight 0. --si-mult 1 = the uniform draw over the same rows (the data-only arm).
"""
from __future__ import annotations

import numpy as np

CLASSES = ("stopped", "starting", "turning", "braking", "cruising")
TURN_DEG = 20.0


def arc(P):
    """(n, 8, >= 2) poses from the origin -> (n, 8) cumulative arc length."""
    Z = np.concatenate([np.zeros_like(P[:, :1, :2]), P[..., :2]], 1)
    return np.cumsum(np.hypot(*np.diff(Z, axis=1).transpose(2, 0, 1)), 1)


def tangent(G):
    Z = np.concatenate([np.zeros_like(G[:, :1, :2]), G[..., :2]], 1)
    d = np.concatenate([Z[:, 2:] - Z[:, :-2], Z[:, -1:] - Z[:, -2:-1]], 1)
    n = np.linalg.norm(d, axis=-1, keepdims=True)
    return np.where(n > 0.2, d / np.maximum(n, 1e-9), np.array([1.0, 0.0]))


def errors(P, G) -> dict:
    """Plan P, human future G (n, 8, >= 2) -> per-row error dict (float32)."""
    P, G = np.asarray(P, np.float64), np.asarray(G, np.float64)
    T = tangent(G)
    e = P[..., :2] - G[..., :2]
    al = (e * T).sum(-1)
    cr = e[..., 1] * T[..., 0] - e[..., 0] * T[..., 1]
    out = dict(ade=np.hypot(*e.transpose(2, 0, 1)).mean(1), fde=np.hypot(*e[:, 7].T), al2=al[:, 3], al4=al[:, 7], cr2=cr[:, 3], cr4=cr[:, 7],
               arc4=arc(P)[:, 7] - arc(G)[:, 7])
    return {k: v.astype(np.float32) for k, v in out.items()}


def classes(G, v0):
    """Human future G (n, 8, 3: x, y, yaw) and t0 speed -> (n,) class names (CLASSES, first match)."""
    G, v0 = np.asarray(G, np.float64), np.asarray(v0, np.float64)
    a4 = arc(G)[:, 7]
    vend = np.hypot(*(G[:, 7, :2] - G[:, 6, :2]).T) / 0.5
    dps = np.degrees(np.abs((G[:, 7, 2] + np.pi) % (2 * np.pi) - np.pi))
    return np.select([(v0 < 0.5) & (a4 < 1.0), (v0 < 2.0) & (vend >= v0 + 1.0), dps >= TURN_DEG, vend <= v0 - 1.5],
                     list(CLASSES[:4]), CLASSES[4]).astype("U8")


def set_weights(err, mode="top", top=0.1, mult=3.0, gamma=1.0):
    """Hardness weights inside one set: (n,) errors -> (n,) weights, mean 1."""
    err = np.asarray(err, np.float64)
    n = len(err)
    rk = np.empty(n)
    rk[np.argsort(err, kind="stable")] = np.arange(1, n + 1)
    if mode == "top":
        w = np.where(rk > n * (1.0 - top), mult, 1.0)
    elif mode == "rank":
        w = (rk / n) ** gamma
    else:
        raise ValueError(f"--si-mode {mode}")
    return w / w.mean()


def scan_err(path, key):
    z = np.load(path)
    return dict(zip(z["names"].astype(str).tolist(), z[key].astype(np.float64).tolist())), str(z["policy"]) if "policy" in z.files else ""


def draws(a, ns, S, tr_o):
    """rm_train's SELF-ITER block -> (normal-row p over tr_o, pool p over the pool rows or None, store rows exempt from the pool's no-drop
    rule (the new trainval rows: plain imitation rows), record)."""
    from jevdrive.common import data_dir
    assert not (a.on_weight or a.on_cap_scene), "--si-scan: not with --on-weight / --on-cap-scene (it sets the pool draw itself)"
    E, pol = scan_err(a.si_scan if a.si_scan.startswith("/") else data_dir() / a.si_scan, a.si_key)
    kw = dict(mode=a.si_mode, top=a.si_top, mult=a.si_mult, gamma=a.si_gamma)
    nm = S.tab["names"].astype(str)
    e_nav = np.array([E.get(x, np.nan) for x in nm[tr_o].tolist()])
    hit = np.isfinite(e_nav)
    assert hit.mean() >= 0.99, f"--si-scan: {hit.mean():.3f} of the normal train rows scanned"
    w_nav = np.ones(len(tr_o))
    w_nav[hit] = set_weights(e_nav[hit], **kw)
    rec = dict(scan=str(a.si_scan), policy=pol, key=a.si_key, **kw, nav_rows=int(len(tr_o)), nav_scanned=int(hit.sum()),
               nav_top_share=float(w_nav[hit][e_nav[hit] >= np.quantile(e_nav[hit], 0.9)].sum() / w_nav.sum()))
    pw, free = None, np.zeros(0, np.int64)
    if a.on:
        tb = np.load(data_dir() / "runs/op_parity/cache" / a.on / "tab.npz")
        pool = tb["pool"].astype(str) if "pool" in tb.files else np.full(len(tb["names"]), "s")
        hold = tb["hold"].astype(str) == "1" if "hold" in tb.files else np.zeros(len(pool), bool)
        new = (pool == "t") & ~hold
        if new.any():
            k_s, k_on = a.si_k_s, ns.k_on
            assert 0 < k_s < k_on, f"--si-k-s {k_s} / --on-k {k_on}"
            pn = nm[ns.on0:]
            e_t = np.array([E.get(x, np.nan) for x in pn[new].tolist()])
            ht = np.isfinite(e_t)
            assert ht.mean() >= 0.99, f"--si-scan: {ht.mean():.3f} of the new pool rows scanned"
            w_t = np.ones(int(new.sum()))
            w_t[ht] = set_weights(e_t[ht], **kw)
            base = pool == "s"
            pw = np.zeros(len(pool))
            pw[base] = (k_s / k_on) / base.sum()
            pw[new] = (1 - k_s / k_on) * w_t / w_t.sum()
            free = ns.on0 + np.flatnonzero(new)
            rec |= dict(pool=a.on, pool_base_rows=int(base.sum()), pool_new_rows=int(new.sum()), pool_hold_rows=int(hold.sum()),
                        pool_new_scanned=int(ht.sum()), k_on=k_on, k_s=k_s, expected_new_per_batch=float(k_on - k_s),
                        new_top_share=float(w_t[ht][e_t[ht] >= np.quantile(e_t[ht], 0.9)].sum() / w_t.sum()))
    p_nav = w_nav / w_nav.sum()
    return p_nav, pw, free, rec
