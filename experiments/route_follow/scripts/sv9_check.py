#!/usr/bin/env python3
"""SPIN v9 offline checks (plans/2026-10-15-spin-v9-prereg.md), numpy only.
  sv9_check.py same            3000 random plans / states: v9's guard switched off, outside its window (t_s >= H_T) or below H_V gives the
                               v8 `gate` reference bit for bit (arrays and the decision)
  sv9_check.py replay --run app1x=<root> rsg=<root> [--scenes list] --md out.md
                               the logged handoff decisions (k 17 .. 26) rebuilt under v8 and v9 (offline: the logged served plan, the controller
                               row's speed / acceleration, t_s = (now - first call) s, lead = the logged lead cut > 0): the reference's speed
                               drop over the MPC's tracked window (1 .. 2 s) and its 2 s arc deficit against 2 x v0, spin vs no-spin rollouts
  sv9_check.py plans --a <root> --b <root>   served poses of every (scene, k) both roots logged: identical count and max |difference|
                               (the flag-off smoke against the stored identity), and the guard records of b (track.hand)
  sv9_check.py screen --b <v9 root> --scenes <list>   per rollout: guard calls whose v9 reference differs from v8's (> 1 cm at 0.5 .. 4 s,
                               rebuilt from the logged plan, the driver's v0 and the controller row's acceleration; the v9 rebuild is checked
                               against the logged reference). A rollout with no such call drove v8's closed loop.
  sv9_check.py fixscreen --b <gate root> --scenes <list> --cfg jlon_v=8   Amendment 5: per rollout of a gate read, the decisions
                               whose reference changes under the candidate (> 2 cm at 0.5 .. 2 s), their speed range, the largest change, and the
                               1 s / 2 s arc of plan, v8 and candidate at the first changed decision"""
import argparse, os, sys
import numpy as np
here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [here, os.path.join(here, "../../alpasim/lib")]
import track_ref as TR

assert TR.HAND == 0, "run with JEV_TRACK unset"


def same():
    rng, bad, n = np.random.default_rng(0), 0, 3000
    for i in range(n):
        v0 = float(rng.uniform(0, 38))
        vp = np.clip(v0 + rng.normal(0, 4) + rng.normal(0, 1.5, 8).cumsum(), 0, None)
        h = np.cumsum(rng.normal(0, 0.05 if i % 3 else 0.3, 8))
        s = np.cumsum(vp * 0.5)
        P = np.stack([s * np.cos(h / 2), s * np.sin(h / 2), h], 1)
        a0 = float(rng.uniform(-6, 2))
        r8, i8 = TR.build(P, v0, 0.0, mode="gate", a0=a0)
        for kw in (dict(hand=0, t_s=1.0), dict(hand=1, t_s=float(rng.uniform(TR.H_T, 20))), dict(hand=1, t_s=1.0, h_v=v0 + 0.01)):
            r9, i9 = TR.build(P, v0, 0.0, mode="gate", a0=a0, lead=bool(i % 2), **kw)
            bad += int(not all(np.array_equal(a, b) for a, b in zip(r8, r9)) or i9 != i8)
    print(f"{n} plans x 3 off-cases: {bad} differ from v8")
    return bad


def replay(a):
    from sv9_diag import rollouts, K0
    keep = set(open(a.scenes).read().split()) if a.scenes else None
    L = ["# v8 vs v9 references at the logged handoff decisions (k 17 .. 26), sv9_check.py replay", "",
         "Per rollout the worst decision of the window: drop = v0 - reference speed at 1.5 s (the middle of the MPC's tracked 1 .. 2 s), "
         "deficit = 2 x v0 - reference arc at 2 s. Quantiles 10 / 50 / 90 %.", "",
         "| label | group | n | served plan drop m/s | v8 drop | v9 drop | served deficit m | v8 deficit | v9 deficit | v9 guard calls (lead) |",
         "|:--|:--|--:|:--|:--|:--|:--|:--|:--|:--|"]
    q = lambda v: "[" + ", ".join(f"{x:.1f}" for x in np.quantile(v, (0.1, 0.5, 0.9))) + "]"  # noqa: E731
    for lr in a.run:
        lab, root = lr.split("=", 1)
        D, rows = rollouts(root, keep), []
        for sc, d in D.items():
            g, R = d["rows"], d["recs"]
            if 0 not in R:
                continue
            t_first = R[0]["now"]
            w = []
            for k in range(K0, K0 + 10):
                if k not in R:
                    continue
                r = R[k]
                i = int(np.abs(g["timestamp_us"] - r["t0"]).argmin())
                v0, a0 = float(np.hypot(g["vx"][i], g["vy"][i])), float(g["acceleration"][i])
                P = np.asarray(r["poses"], float)
                lead = (((r.get("fix") or {}).get("lead") or {}).get("cut", 0.0) or 0.0) > 0
                ts = (r["now"] - t_first) * 1e-6
                out = []
                for kw in (dict(hand=0), dict(hand=1)):
                    (t, x, y, _), inf = TR.build(P, v0, 0.0, mode="gate", a0=a0, t_s=ts, lead=lead, **kw)
                    sp = np.hypot(np.diff(x), np.diff(y)) / 0.1
                    out += [v0 - float(sp[14]), 2 * v0 - float(np.hypot(np.diff(x[:21]), np.diff(y[:21])).sum()), inf.get("hand", 0)]
                seg = np.hypot(*np.diff(np.r_[[[0.0, 0.0]], P[:, :2]], axis=0).T)
                w.append([v0 - seg[2] / 0.5, 2 * v0 - seg[:4].sum(), *out, int(lead)])
            if w:
                w = np.array(w)
                rows.append((d["hs"], w[:, [0, 1, 2, 3, 5, 6]].max(0), int(w[:, 7].sum()), int((w[:, 7] * w[:, 8]).sum())))
        for gname, sel in (("handoff spin", lambda r: r[0]), ("no handoff spin", lambda r: not r[0])):
            S = [r for r in rows if sel(r)]
            if not S:
                continue
            M = np.array([r[1] for r in S])
            L.append(f"| {lab} | {gname} | {len(S)} | {q(M[:, 0])} | {q(M[:, 2])} | {q(M[:, 4])} | {q(M[:, 1])} | {q(M[:, 3])} | {q(M[:, 5])} | "
                     f"{sum(r[2] for r in S)} ({sum(r[3] for r in S)}) |")
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def plans(a):
    import glob, json
    def recs(root):
        out = {}
        for f in glob.glob(os.path.join(root, "runs", "*", "driver", "drive.jsonl")):
            for line in open(f):
                if '"infer": true' in line:
                    r = json.loads(line)
                    out[(r["scene"], r["k"])] = r
        return out
    A, B = recs(a.a), recs(a.b)
    ks = sorted(set(A) & set(B))
    d = [float(np.abs(np.array(A[k]["poses"]) - np.array(B[k]["poses"])).max()) for k in ks]
    sc = sorted({k[0] for k in ks})
    print(f"{len(sc)} scenes, {len(ks)} common decisions (a {len(A)}, b {len(B)}): identical {sum(x == 0 for x in d)}, max |diff| {max(d, default=0):.3g}")
    hb = [k for k in sorted(B) if (B[k].get("track") or {}).get("hand")]
    print(f"b: {sum('track' in r for r in B.values())} track records, {len(hb)} with the v9 guard, k range "
          f"{min((k[1] for k in hb), default=None)}..{max((k[1] for k in hb), default=None)}")
    return 0 if ks and max(d) == 0 else 1


def fixscreen(a):
    from sv9_diag import rollouts
    kv = {k: float(v) for k, v in (x.split("=") for x in a.cfg.split(","))}
    D = rollouts(a.b, set(open(a.scenes).read().split()))
    arc = lambda X, Y, j: float(np.hypot(np.diff(X[:j + 1]), np.diff(Y[:j + 1])).sum())  # noqa: E731
    tot = ch = nr = 0
    print("scene | decisions | changed | v0 range of changed | max change m | first changed k: v0, arc 1 s / 2 s plan, v8, candidate")
    for sc, d in sorted(D.items()):
        g, R = d["rows"], d["recs"]
        n = c = 0; vs, mx, first, g8, gc = [], 0.0, None, [], []
        for k in sorted(R):
            r = R[k]; tr = r.get("track")
            if not tr:
                continue
            n += 1
            i = int(np.abs(g["timestamp_us"] - r["t0"]).argmin())
            P, a0 = np.asarray(r["poses"], float), float(g["acceleration"][i])
            r8, _ = TR.build(P, tr["v0"], 0.0, mode="gate", a0=a0)
            rc, _ = TR.build(P, tr["v0"], 0.0, mode="gate", a0=a0, **kv)
            dd = float(max(np.hypot(r8[1][j] - rc[1][j], r8[2][j] - rc[2][j]) for j in (5, 10, 15, 20)))
            if dd > 0.02:
                c += 1; vs.append(tr["v0"]); mx = max(mx, dd)
                s1 = float(np.hypot(*np.diff(np.r_[[[0.0, 0.0]], P[:2, :2]], axis=0).T).sum())
                g8.append(arc(r8[1], r8[2], 10) - s1); gc.append(arc(rc[1], rc[2], 10) - s1)
                if first is None:
                    seg = np.hypot(*np.diff(np.r_[[[0.0, 0.0]], P[:, :2]], axis=0).T)
                    first = (k, round(tr["v0"], 1), f"{seg[:2].sum():.2f}/{seg[:4].sum():.2f}", f"{arc(r8[1], r8[2], 10):.2f}/{arc(r8[1], r8[2], 20):.2f}",
                             f"{arc(rc[1], rc[2], 10):.2f}/{arc(rc[1], rc[2], 20):.2f}")
        tot += n; ch += c; nr += int(c > 0)
        print(f"{sc[7:15]} | {n} | {c} | {min(vs, default=0):.1f}-{max(vs, default=0):.1f} | {mx:.2f} | {first} | "
              f"1 s arc minus the served plan's, mean over changed: v8 {np.mean(g8) if g8 else 0:+.3f}, candidate {np.mean(gc) if gc else 0:+.3f} m")
    print(f"total: {ch} / {tot} decisions changed, {nr} / {len(D)} rollouts")


def screen(a):
    from sv9_diag import rollouts
    keep = set(open(a.scenes).read().split())
    D = rollouts(a.b, keep)
    print("scene | guard calls | v9 != v8 | rebuild mismatch | first differing k (t_s, v0, plan drop m/s)")
    for sc, d in sorted(D.items()):
        g, R = d["rows"], d["recs"]
        n = nd = mis = 0
        first = None
        for k in sorted(R):
            r = R[k]
            tr = r.get("track") or {}
            if not tr.get("hand"):
                continue
            n += 1
            i = int(np.abs(g["timestamp_us"] - r["t0"]).argmin())
            P, a0 = np.asarray(r["poses"], float), float(g["acceleration"][i])
            r9, _ = TR.build(P, tr["v0"], 0.0, mode="gate", a0=a0, t_s=tr["t_s"], lead=bool(tr["lead"]), hand=1)
            r8, _ = TR.build(P, tr["v0"], 0.0, mode="gate", a0=a0)
            mis += int(np.abs(TR.poses8(r9) - np.asarray(r["poses_track"])).max() > 0.02)
            if np.abs(TR.poses8(r9) - TR.poses8(r8)).max() > 0.01:
                nd += 1
                if first is None:
                    first = (k, tr["t_s"], tr["v0"], round(tr["v0"] - float(np.hypot(*P[2, :2] - P[1, :2]) / 0.5), 1))
        print(f"{sc[7:15]} | {n} | {nd} | {mis} | {first}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("same", "replay", "plans", "screen", "fixscreen"))
    ap.add_argument("--cfg")
    ap.add_argument("--a")
    ap.add_argument("--b")
    ap.add_argument("--run", nargs="+")
    ap.add_argument("--scenes")
    ap.add_argument("--md")
    a = ap.parse_args()
    if a.cmd == "same":
        sys.exit(1 if same() else 0)
    sys.exit(plans(a)) if a.cmd == "plans" else screen(a) if a.cmd == "screen" else fixscreen(a) if a.cmd == "fixscreen" else replay(a)
