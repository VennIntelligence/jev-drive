#!/usr/bin/env python3
"""SPIN v9 read (plans/2026-10-15-spin-v9-prereg.md), numpy only, Tokyo host. One checkpoint: an arm root against its identity root on the
union and the checkpoint's spin set / control set. Per rollout: score, zero kind, sf_detect spin (handoff / fast / slow) and its onset,
rf_pai281's handoff spin, at-fault / front / rear collisions, the first offroad / any event time from the rollout's metrics.parquet (needs
pyarrow: run with /data/envs/op-train/bin/python on Tokyo; a zero does not end the rollout, the controller runs to 19.9 s).
Lines: L1 spins (sf_detect, any kind) on the spin set <= 2; L2 controls: zeros, paired bootstrap 97.5 % of the summed difference <= +2,
and the paired mean difference >= -0.01; L3 no new offroad zero whose first offroad step is <= 4.7 s (3 s after the 1.7 s handoff) over the union.
Bootstrap over scenes, 10 000 draws, seed 0. A rollout without metrics is skipped (sf_read.py's rule); an arm may list root1,root2.
  sv9_read.py --label APP1X --id <root> --arm <root>[,<root>] --spin <list> --ctrl <list> --union <list> [--ctrl-all <list>] --md out.md"""
import argparse, csv, glob, json, os, sys
import numpy as np
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "../../steer_long/scripts"))
from sf_detect import detect


def events(f, ts0):
    """First step (s, controller clock + 0.1) with offroad / collision_any / left_corridor_laterally > 0, None when it never fires."""
    import pyarrow.parquet as pq
    if not os.path.exists(f):
        return dict(offroad=None, collision_any=None, left_corridor_laterally=None)
    m = pq.read_table(f, columns=["name", "timestamps_us", "values"]).to_pydict()
    out = {}
    for nm in ("offroad", "collision_any", "left_corridor_laterally"):
        hit = [ts for n, ts, v in zip(m["name"], m["timestamps_us"], m["values"]) if n == nm and ts is not None and (v or 0) > 0]
        out[nm] = round((min(hit) - ts0) / 1e6 + 0.1, 2) if hit else None
    return out


def read(roots, keep):
    out = {}
    for root in roots.split(","):
        for sim in sorted(glob.glob(os.path.join(root, "runs", "*", "sim"))):
            f = os.path.join(sim, "aggregate", "results-summary.json")
            if not os.path.exists(f):
                continue
            for r in json.load(open(f))["rollouts"]:
                c = os.path.join(sim, "controller", f"alpasim_controller_{r['rollout_id']}.csv")
                if r["clipgt_id"] not in keep or not r.get("metrics") or not os.path.exists(c):
                    continue
                rows = list(csv.DictReader(open(c)))
                g = {k: np.array([float(x[k]) for x in rows]) for k in ("timestamp_us", "vx", "vy", "wz", "front_steering_angle")}
                t = (g["timestamp_us"] - g["timestamp_us"][0]) / 1e6 + 0.1
                d = detect(t, g["vx"], g["vy"], g["wz"], g["front_steering_angle"])
                w = (t > 1.7) & (t <= 3.2)
                m = r["metrics"]
                sc = float(r.get("score") or 0)
                kind = "" if sc else ("collision" if m.get("collision_at_fault") else "offroad" if m.get("offroad") else
                                      "corridor" if m.get("left_corridor_laterally") else "other")
                ev = events(os.path.join(sim, "rollouts", r["clipgt_id"], r["rollout_id"], "metrics.parquet"), g["timestamp_us"][0])
                out[r["clipgt_id"]] = dict(score=sc, kind=kind, spin=d["kind"], t_spin=d["t_first"], t_off=ev["offroad"], t_ev=min(
                                           (x for x in ev.values() if x is not None), default=None),
                                           hspin=bool(w.any() and np.abs(g["vx"] * g["wz"])[w].max() > 6),
                                           atf=float(m.get("collision_at_fault") or 0) > 0, front=float(m.get("collision_front") or 0) > 0,
                                           rear=float(m.get("collision_rear") or 0) > 0)
    return out


def boot(x, rng):
    b = rng.integers(0, len(x), (10000, len(x)))
    return np.quantile(x[b].sum(1), (0.025, 0.975)), np.quantile(x[b].mean(1), (0.025, 0.975))


def main():
    ap = argparse.ArgumentParser()
    for k in ("label", "id", "arm", "spin", "ctrl", "union", "md"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--ctrl-all")
    ap.add_argument("--a6", default="", help="Amendment 6 lines against a v8 reference as --id: spins <= id, control zeros <= id, "
                    "these comma-separated scene ids (8 hex) scored > 0, no new zero of any kind with its first event <= 4.7 s")
    a = ap.parse_args()
    L = lambda f: [x for x in open(f).read().split() if x]  # noqa: E731
    U = L(a.union)
    I, A = read(a.id, set(U)), read(a.arm, set(U))
    rng = np.random.default_rng(0)
    out = [f"# SPIN v9 read: {a.label}, arm `{a.arm}` vs identity `{a.id}`", ""]
    res = {"atf": {}}
    out += ["| set | scenes (both) | spins sf any (id) | handoff spins (id) | zeros (id) | new / recovered zeros | zeros diff 95 % CI | "
            "at-fault / front / rear (id) | at-fault diff 95 % CI | mean (id) | diff 95 % CI |", "|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|"]
    sets = [("spin set", a.spin), ("controls", a.ctrl)] + ([("controls, all 46", a.ctrl_all)] if a.ctrl_all else []) + [("union", a.union)]
    for name, f in sets:
        sc = [s for s in L(f) if s in I and s in A]
        z = lambda D: np.array([D[s]["score"] == 0 for s in sc], float)  # noqa: E731
        dz, ds = z(A) - z(I), np.array([A[s]["score"] - I[s]["score"] for s in sc])
        (zl, zh), _ = boot(dz, rng)
        _, (ml, mh) = boot(ds, rng)
        da = np.array([float(A[s]["atf"]) - float(I[s]["atf"]) for s in sc])
        (al, ah), _ = boot(da, rng)
        res.setdefault("atf", {})[name] = f"{da.sum():+.0f} [{al:+.0f}, {ah:+.0f}]"
        cnt = lambda D, k: sum(bool(D[s][k]) for s in sc)  # noqa: E731
        res[name] = dict(n=len(sc), spins=cnt(A, "spin"), zh=zh, md=ds.mean(), missing=len(L(f)) - len(sc), atf=res["atf"].pop(name))
        out.append(f"| {name} | {len(sc)} | {cnt(A, 'spin')} ({cnt(I, 'spin')}) | {cnt(A, 'hspin')} ({cnt(I, 'hspin')}) | {int(z(A).sum())} ({int(z(I).sum())}) | "
                   f"{int((dz > 0).sum())} / {int((dz < 0).sum())} | {dz.sum():+.0f} [{zl:+.0f}, {zh:+.0f}] | {cnt(A, 'atf')} / {cnt(A, 'front')} / {cnt(A, 'rear')} "
                   f"({cnt(I, 'atf')} / {cnt(I, 'front')} / {cnt(I, 'rear')}) | {res[name]['atf']} | {np.mean([A[s]['score'] for s in sc]):.4f} ({np.mean([I[s]['score'] for s in sc]):.4f}) | "
                   f"{ds.mean():+.4f} [{ml:+.4f}, {mh:+.4f}] |")
    eo = lambda r: r["kind"] == "offroad" and r["t_off"] is not None and r["t_off"] <= 4.7  # noqa: E731
    early = [s for s in U if s in A and s in I and eo(A[s]) and not eo(I[s])]
    l1, l2 = res["spin set"]["spins"] <= 2, res["controls"]["zh"] <= 2 and res["controls"]["md"] >= -0.01
    l3 = not early
    if a.a6:
        must = [x for x in a.a6.split(",") if x and x != "none"]
        ze = lambda r: r["score"] == 0 and r["t_ev"] is not None and r["t_ev"] <= 4.7  # noqa: E731
        early = [s for s in U if s in A and s in I and ze(A[s]) and not ze(I[s])]
        sid = {s[7:15]: s for s in U}
        cz = lambda D: sum(D[s]["score"] == 0 for s in L(a.ctrl) if s in A and s in I)  # noqa: E731
        bad = [m for m in must if m in sid and sid[m] in A and A[sid[m]]["score"] == 0]
        si = sum(bool(I[s]["spin"]) for s in L(a.spin) if s in A and s in I)
        l1, l2, l3 = res["spin set"]["spins"] <= si, cz(A) <= cz(I) and not bad, not early
        out += ["", f"- A6 L1 spin set spins {res['spin set']['spins']} <= reference {si}: {'pass' if l1 else 'FAIL'}",
                f"- A6 L2 control zeros {cz(A)} <= reference {cz(I)}, scored: {must} (zero: {bad}): {'pass' if l2 else 'FAIL'}",
                f"- A6 L3 new zeros (any kind) with the first event <= 4.7 s: {len(early)} {[s[7:15] for s in early]}: {'pass' if l3 else 'FAIL'}"]
    if not a.a6:
        out += ["", f"- L1 spin set spins {res['spin set']['spins']} <= 2: {'pass' if l1 else 'FAIL'}",
                f"- L2 controls zeros CI upper {res['controls']['zh']:+.0f} <= +2 and mean diff {res['controls']['md']:+.4f} >= -0.01: {'pass' if l2 else 'FAIL'}",
                f"- L3 new offroad zeros with the event <= 4.7 s over the union: {len(early)} {early and [s[7:15] for s in early] or ''}: {'pass' if l3 else 'FAIL'}"]
    out += [f"- missing rollouts (no metrics in either root): " + ", ".join(f"{k} {v['missing']}" for k, v in res.items() if k != "atf"), "",
            "## Per scene (union; changed score, zero or spin only)", "",
            "| scene | sets | id score | arm score | id zero / spin (onset s) | arm zero / spin (onset s) | id / arm first event s |", "|:--|:--|--:|--:|:--|:--|--:|"]
    member = {s: " + ".join(n for n, f in sets[:-1] if s in set(L(f))) for s in U}
    for s in U:
        if s not in A or s not in I:
            out.append(f"| {s[7:15]} | {member[s]} | missing | | | | |")
            continue
        i, x = I[s], A[s]
        if abs(i["score"] - x["score"]) < 1e-9 and i["spin"] == x["spin"] and i["kind"] == x["kind"]:
            continue
        f = lambda r: f"{r['kind'] or '-'} / {r['spin'] or '-'}" + (f" ({r['t_spin']:.1f})" if r["spin"] else "")  # noqa: E731
        out.append(f"| {s[7:15]} | {member[s]} | {i['score']:.3f} | {x['score']:.3f} | {f(i)} | {f(x)} | {i['t_ev'] or '-'} / {x['t_ev'] or '-'} |")
    out.append("")
    out.append(f"VERDICT {a.label}: L1 {'pass' if l1 else 'FAIL'}, L2 {'pass' if l2 else 'FAIL'}, L3 {'pass' if l3 else 'FAIL'}")
    open(a.md, "w").write("\n".join(out) + "\n")
    print("\n".join(out[:12] + out[-1:]))


if __name__ == "__main__":
    main()
