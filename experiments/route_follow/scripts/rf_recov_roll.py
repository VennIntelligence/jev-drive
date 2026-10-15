#!/usr/bin/env python3
"""RECOVERY-ENVELOPE row pools (prereg experiments/route_follow/plans/2026-10-16-recovery-envelope-prereg.md, sections R and
"推快的接近"; runner rf_recov_roll.sh). The rows are on_rows.py geo's (unedited) on sessions run with the frame dump; this script picks and
tags them and writes one pool in the on-policy format (experiments/onpol_nuplan/plans/on_pool_format.md) that on_rows.py tok, ol_labels.py,
rm_build.py pool and on_lon_pack.py then complete exactly as for on1_L18RSG.

R (on1_RCVd316; the all-scene rewind later): the D2 / D3 expert handovers of a rewind stage re-run with SH30_DUMP.
  rcheck   (any python)   --stage S --prior <read.json of the source>   the re-run against the source read: per unit the handover decision, the
           outcome, k_z, completion, the identity tolerance (n-read's within_tol, plan 0.15 m / track 5 cm / v 0.05 m/s against the source's
           identity sessions), and the served plans of the source's own D session (before h: the policy's; from h: the expert's) -> S/check.json
  rsel     (AlpaSim venv) --stage S --geo D2=<geo rows>,D3=<geo rows> --out S/sel.json   units = handover sessions with h as in from_D<d>.txt,
           no zero before h, the turn completed and no zero-score event at all (n-yield's label); rows = decisions k >= h whose rear-axle
           anchor is at most rf_follow's exit + 10 m on the session's own route (n-read pkl), before the first zero (none by the unit rule)
           --rows-csv <n-yield rows.csv> --check S/check.json (the all-scene rewind): rows = exactly the source's label rows that were
           within tolerance there, kept where the re-run is itself label-true and within tolerance of the source (identity: n-read's
           within_tol; the source's own session: plans before h within 0.15 m)
R rows train on the expert's served plan: tab `fut` = meta `poses` (the dump's served plan = the expert's from h on, asserted against the
record's `expert` flag and `poses_model`); the log-path target on_rows wrote is kept as `fut_path`, the model's own plan as `served_model`.

Fast approaches (on1_RCVfast): RSG-L18-s0 pushed with JEV_PERTURB (g 1, arc scale s, n decisions) 40-60 m before a qualifying stop line
(front bumper) or a strict-turn entry (rear axle), then driving on.
  fsel     (op-train)     --pool L18RSG --turns <train64 scenes> --out <json>   per scene the push decision k_p from the stored 18-decision
           rollouts (rows of on1_L18RSG + its lon.npz on the scene context ctx.npz, rf_loninv's geometry)
  fprep    (any python)   --sel <fsel json> --stage <stage dir> --count <n per class | all> --levels 1.1,1.15 --n 2   stage list + JEV_PERTURB files
  fread    (AlpaSim venv) --stage <stage dir> --sel <fsel json> --out <json>   pushed sessions against the unpushed identity of the same
           launch config: fed speed at k_p, at the end of the push window, at the line / entry by travel from k_p; zero events (rf_follow's
           rule: at-fault or offroad, or dist_to_gt >= 4 m) per decision
  frows    (any python)   --stages <dirs> --sel <fsel json> --geo L<lvl>=<geo rows>,.. --out <json>   rows k >= k_p, self rows
Common:
  assemble (op-train)     --name N --sel <sel json>   -> runs/op_parity/cache/on1_N/{tab,agent,road,drivable}.npz, runs/onpol_nuplan/rows/N/
           {meta,all}.npz, r2.json (merged), sel.json; names on:<prefix>:<scene>:<seed>:<k>; string tags (mode, ...), dev = crc32(log) % 100 < 15
  strip    (any python)   --runs <run dirs>   drop the slot frames (`cur`) of the dumps after the tokens exist
  report   (any python)   --name N --out <dir>   row counts per tag and the gate files of the pool -> <out>/report_N.{json,md}
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
import zlib
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

R_ = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
CR, ROWS = DATA / "runs/op_parity/cache", DATA / "runs/onpol_nuplan/rows"
PAST, DEV_PCT, V_QUAL, D_APP = 10.0, 15, 3.0, 40.0          # rf_follow's exit + 10 m; rf_apppool.is_dev; lon_inventory's >= 3 m/s set
NONROW = {"classes", "layers", "x0", "y0", "res", "di", "pi"}


def is_dev(log) -> bool:
    return zlib.crc32(str(log).encode()) % 100 < DEV_PCT


def drive_records(run) -> dict:
    """drive.jsonl -> {scene: [records sorted by k]} of the last session of each scene (ao_e1.drive_records' rule)."""
    S, best = {}, {}
    for line in open(Path(run) / "driver-logs/drive.jsonl"):
        r = json.loads(line)
        if r["kind"] == "start":
            S[r["session"]] = (r["scene"], r["n"], [])
        elif r["kind"] == "drive" and r["session"] in S:
            S[r["session"]][2].append(r)
    for sc, n, rr in S.values():
        if rr and (sc not in best or n > best[sc][0]):
            best[sc] = (n, sorted(rr, key=lambda r: r["k"]))
    return {sc: rr for sc, (n, rr) in best.items()}


def summary_runs(dirs) -> dict:
    """Run dirs -> {scene: run dir} from their results-summary."""
    out = {}
    for D in dirs:
        f = Path(D) / "aggregate/results-summary.json"
        if f.exists():
            for r in json.loads(f.read_text())["rollouts"]:
                out.setdefault(r["clipgt_id"], str(D))
    return out


def project(R, S, p, lo=-np.inf, hi=np.inf):
    """rf_pai_follow.project: point p -> (station, signed lateral) on polyline R within stations [lo, hi]."""
    a, b = R[:-1], R[1:]
    d = b - a
    L2 = np.maximum((d ** 2).sum(1), 1e-12)
    u = np.clip(((p - a) * d).sum(1) / L2, 0, 1)
    q = a + u[:, None] * d
    s = S[:-1] + u * np.sqrt(L2)
    dist = np.hypot(*(p - q).T)
    ok = (s >= lo) & (s <= hi)
    if not ok.any():
        ok[:] = True
    i = int(np.flatnonzero(ok)[np.argmin(dist[ok])])
    cr = d[i, 0] * (p[1] - a[i, 1]) - d[i, 1] * (p[0] - a[i, 0])
    return float(s[i]), float(np.sign(cr) * dist[i])


def stations_of(R, recs) -> dict:
    """rf_follow.one's decision stations: anchors projected in order with the -30 / +60 m hint window -> {k: station}."""
    S = np.r_[0.0, np.cumsum(np.hypot(*np.diff(R, axis=0).T))]
    out, hint = {}, None
    for r in recs:
        lo, hi = (-np.inf, np.inf) if hint is None else (hint - 30.0, hint + 60.0)
        hint = out[int(r["k"])] = project(R, S, np.asarray(r["anchor"], float)[:2], lo, hi)[0]
    return out


def speed_of(r) -> float:
    e = r["ego"]
    return float(np.hypot(e[4], e[5]) * 10.0)


def zeros_of(run, scene) -> int:
    """rf_follow's k_z: first decision with an at-fault event (offroad, front / lateral collision) or dist_to_gt >= 4 m; -1 if none."""
    if str(R_ / "experiments/alpasim_long/scripts") not in sys.path:
        sys.path[:0] = [str(R_ / "experiments/alpasim_long/scripts")]
    import long_read as LR
    ev = LR.one(("x", run, scene))
    return min([r["k"] for r in ev if r["atf"] or (r["dist_gt"] or 0) >= 4.0], default=-1)


# ---------------------------------------------------------------- R
def rcheck(a):
    S = Path(a.stage)
    mine = {s["scene"]: {r["tag"]: r for r in s["rows"]} for s in json.loads((S / "read.json").read_text())["scenes"]}
    prior = {s["scene"]: {r["tag"]: r for r in s["rows"]} for s in json.loads(Path(a.prior).read_text())["scenes"]} if a.prior else {}
    pr = {d: summary_runs([r for st in a.prior_runs.split(",") if st for r in sorted(glob.glob(f"{st}/D{d}/c*/run"))]) for d in (2, 3)}
    out, tot = [], {}
    for d in (2, 3):
        hs = {s: int(h) for s, h in (ln.split() for ln in open(S / f"from_D{d}.txt") if ln.strip())}
        for sc, h in sorted(hs.items()):
            x, p = mine.get(sc, {}).get(f"D{d}"), prior.get(sc, {}).get(f"D{d}")
            u = dict(scene=sc, D=d, h=h, have=x is not None, prior=p is not None)
            if x is not None:
                u |= dict(h_run=x.get("h"), completed=x.get("completed"), outcome=x.get("outcome"), kz=x.get("kz"), within_tol=x.get("within_tol"),
                          pre_plan_maxdiff=x.get("pre_plan_maxdiff"), pre_track_maxdiff=x.get("pre_track_maxdiff"), dv_h=x.get("dv_h"),
                          zero_before_h=x.get("zero_before_h"))
            if x is not None and p is not None:
                u |= dict(p_completed=p.get("completed"), p_outcome=p.get("outcome"), p_kz=p.get("kz"), p_h=p.get("h"),
                          same_outcome=bool(x.get("completed") == p.get("completed") and x.get("outcome") == p.get("outcome")
                                            and (x.get("kz", -1) >= 0) == (p.get("kz", -1) >= 0)))
            prun = (p or {}).get("run") or pr[d].get(sc)
            if x is not None and x.get("run") and prun:
                A, B = drive_records(x["run"]).get(sc, []), drive_records(prun).get(sc, [])
                Bk = {r["k"]: r for r in B}
                pre = [float(np.abs(np.subtract(r["poses"], Bk[r["k"]]["poses"])).max()) for r in A if r["k"] < h and r["k"] in Bk]
                post = [float(np.abs(np.subtract(r["poses"], Bk[r["k"]]["poses"])).max()) for r in A if r["k"] >= h and r["k"] in Bk]
                u |= dict(vs_prior_pre_max=max(pre, default=None), vs_prior_pre_identical=bool(pre and max(pre) == 0.0),
                          vs_prior_post_h_max=post[0] if post else None, vs_prior_post_max=max(post, default=None), prior_run=prun)
            out.append(u)
        U = [u for u in out if u["D"] == d]
        V = [u for u in U if u["have"] and u.get("prior")]
        tot[f"D{d}"] = dict(units=len(U), read=sum(u["have"] for u in U), valid=sum(bool(u.get("h_run") == u["h"] and not u.get("zero_before_h")) for u in U),
                            within_tol=sum(bool(u.get("within_tol")) for u in U), completed=sum(bool(u.get("completed")) for u in U),
                            label=sum(bool(u.get("h_run") == u["h"] and not u.get("zero_before_h") and u.get("completed") and (u.get("kz", -1) < 0)) for u in U),
                            prior_completed=sum(bool(u.get("p_completed")) for u in V), same_outcome=sum(bool(u.get("same_outcome")) for u in V),
                            pre_identical=sum(bool(u.get("vs_prior_pre_identical")) for u in V),
                            pre_within_015=sum(bool(u.get("vs_prior_pre_max") is not None and u["vs_prior_pre_max"] <= 0.15) for u in V),
                            post_h_identical=sum(bool(u.get("vs_prior_post_h_max") == 0.0) for u in V), compared=len(V))
    (S / "check.json").write_text(json.dumps(dict(totals=tot, units=out), indent=1, default=float))
    print(json.dumps(tot, indent=1))


def rsel(a):
    S = Path(a.stage)
    X = json.loads((S / "read.json").read_text())
    rd = {s["scene"]: {r["tag"]: r for r in s["rows"]} for s in X["scenes"]}
    srcs, units = [], []
    allow = None
    if a.rows_csv:                                       # the source's label rows (label AND within_tol): the row set to reproduce
        import csv
        allow = {(r["scene"], int(r["delta"]), int(r["k"])) for r in csv.DictReader(open(a.rows_csv)) if r["label"] == "True" and r["within_tol"] == "True"}
    chk = {(u["scene"], u["D"]): u for u in json.loads(Path(a.check).read_text())["units"]} if a.check else {}
    for spec in a.geo.split(","):
        tag, g = spec.split("=")
        d = int(tag[1:])
        hs = {s: int(h) for s, h in (ln.split() for ln in open(S / f"from_D{d}.txt") if ln.strip())}
        M = np.load(ROWS / g / "meta.npz")
        tb = np.load(CR / f"on1_{g}" / "tab.npz")
        names, scene, kk, run = M["names"].tolist(), M["scene"].tolist(), M["k"].astype(int), M["run"].tolist()
        rows, xp_bad, same = {}, 0, 0
        for sc, h in sorted(hs.items()):
            x = rd.get(sc, {}).get(tag)
            u = dict(scene=sc, D=d, h=h, read=x is not None)
            idx = [i for i, s in enumerate(scene) if s == sc]
            if x is not None:
                kz = x.get("kz", -1)
                valid = bool(x.get("h") == h and not x.get("zero_before_h"))
                keep = bool(valid and x.get("completed") and kz < 0)
                if a.check:                                  # within tolerance of the source's identity (n-read) and of the source's own session
                    c = chk.get((sc, d), {})
                    u["tol_source"] = bool(x.get("within_tol") and c.get("vs_prior_pre_max") is not None and c["vs_prior_pre_max"] <= 0.15)
                    keep = keep and u["tol_source"]
                if allow is not None:
                    u["source_rows"] = sum((sc, d, q) in allow for q in range(60))
                    keep = keep and u["source_rows"] > 0
                u |= dict(valid=valid, completed=x.get("completed"), outcome=x.get("outcome"), kz=kz, keep=keep, within_tol=x.get("within_tol"),
                          stratum=x.get("stratum"), unit_src=x.get("unit_src"), geo_rows=len(idx))
                if keep:
                    pk = pickle.load(open(S / "pkl" / f"{sc}.pkl", "rb"))["rolls"][tag]
                    recs = drive_records(x["run"])[sc]
                    assert all(run[i] == x["run"] for i in idx), f"{sc}: geo rows from another run than the read's"
                    st = stations_of(np.asarray(pk["R"], float), recs)
                    rk = {int(r["k"]): r for r in recs}
                    lim = x["exit"] + PAST
                    pick = ([i for i in idx if kk[i] >= h and (sc, d, int(kk[i])) in allow] if allow is not None else
                            [i for i in idx if kk[i] >= h and st.get(int(kk[i]), np.inf) <= lim])
                    for i in pick:
                        r = rk[int(kk[i])]
                        xp_bad += int("expert" not in r)                         # the served plan from h on must be the expert's
                        same += int(np.abs(np.subtract(r["poses"], r["poses_model"])).max() < 1e-4) if "poses_model" in r else 0
                        assert np.abs(np.asarray(r["poses"]) - tb["served"][i]).max() < 1e-3, f"{names[i]}: tab served is not the record's poses"
                        rows[names[i]] = dict(mode="both", unit=f"{sc}~D{d}", D=f"D{d}", kind="h" if kk[i] == h else "post",
                                              usrc=str(x.get("unit_src")), tol="1" if x.get("within_tol") else "0")
                    u |= dict(rows=len(pick), rows_h=sum(kk[i] == h for i in pick), h_row_kept=bool(any(kk[i] == h for i in idx)),
                              k_last=int(max(kk[i] for i in pick)) if pick else None, exit_plus10=lim)
            units.append(u)
        assert xp_bad == 0, f"{xp_bad} picked rows without the expert flag"
        srcs.append(dict(geo=g, prefix=tag, fut="served", rows=rows, plan_equal_model=same))
    n = {s["prefix"]: len(s["rows"]) for s in srcs}
    json.dump(dict(kind="R", stage=str(S), sources=srcs, units=units), open(a.out, "w"), indent=1, default=float)
    print("units kept", sum(bool(u.get("keep")) for u in units), "of", len(units), "rows", n, "->", a.out)


# ---------------------------------------------------------------- fast approaches
def fsel(a):
    """Push decision per scene from the stored rollouts of on1_<pool> (rf_loninv.policy's line / turn geometry, speed floor 3 m/s)."""
    sys.path[:0] = [str(R_), str(R_ / "experiments/route_follow/scripts")]
    import rf_loninv as LI
    rd, cache = ROWS / a.pool, CR / f"on1_{a.pool}"
    tab, m, Z = np.load(cache / "tab.npz"), np.load(rd / "meta.npz"), dict(np.load(rd / "lon.npz"))
    X = dict(np.load(DATA / "runs/onpol_nuplan/ctx/ctx.npz"))
    names = tab["names"]
    assert (m["names"] == names).all() and (Z["names"] == names).all()
    N = len(names)
    scene, k, speed = m["scene"].astype(str), m["k"].astype(int), tab["speed"].astype(np.float64)
    sidx = {s: i for i, s in enumerate(X["scene"].tolist())}
    sj = np.array([sidx[s] for s in scene])
    onr = Z["ok"].astype(bool) & (np.abs(Z["l0"]) <= LI.LAT)
    s0, sf0 = Z["s0"].astype(np.float64), Z["sf0"].astype(np.float64)
    ty, ld, st, elig = X["ln_type"][sj], X["ln_d"][sj].astype(np.float64), X["ln_st"][sj], X["ln_elig"][sj]
    dF = ld - sf0[:, None]
    kt, t = X["kf_ts"][sj], m["t_log_us"].astype(np.int64)
    dist = np.where(kt > 0, np.abs(kt - t[:, None]), np.int64(1) << 60)
    near = dist.argmin(1)
    known = dist[np.arange(N), near] <= 3e5
    st_now = np.where(known[:, None], st[np.arange(N), :, near], -1)
    q = LI.first_ahead(elig & (ty > 0), dF)
    d = LI.take(dF, q)
    lt = LI.take(ty, q)
    qual = onr & (q >= 0) & (d <= D_APP) & ((lt == 1) | ((lt == 2) & (LI.take(st_now, q) == 1))) & (speed >= V_QUAL)
    out, why = {}, {}

    def pick(rows, dd, wins):
        """Rows of one scene with distance dd ahead -> (row, how): a served decision k >= 2 at >= 3 m/s in the first window that has one
        (closest to 50 m)."""
        for lo, hi, how in wins:
            c = [(abs(x - 50.0), r) for r, x in zip(rows, dd) if k[r] >= 2 and onr[r] and speed[r] >= V_QUAL and lo <= x <= hi]
            if c:
                return min(c)[1], how
        return None, None
    W_STOP = ((40.0, 60.0, "in"), (30.0, 70.0, "near"), (20.0, 30.0, "short"))
    W_TURN = ((40.0, 60.0, "in"), (30.0, 70.0, "near"))
    for sc in sorted(set(scene[qual])):
        r_q = np.flatnonzero(qual & (scene == sc))
        r0 = r_q[np.argmin(k[r_q])]
        li = int(q[r0])
        rows = np.flatnonzero(scene == sc)
        r, how = pick(rows, ld[rows, li] - sf0[rows], W_STOP)
        if r is None:
            why[sc] = dict(cls="stop", reason="no decision 20-70 m before the line at >= 3 m/s",
                           d_k2=float((ld[rows, li] - sf0[rows])[k[rows] == 2][0]) if (k[rows] == 2).any() else None)
            continue
        out[sc] = dict(cls="stop", kp=int(k[r]), d=float(ld[r, li] - sf0[r]), how=how, v=float(speed[r]), line=li, line_type=int(ty[r, li]),
                       k_first_qual=int(k[r0]), name=str(names[r]))
    nstop = len(set(scene[qual]))
    # turns: the train64 scenes first, then the other pool scenes in crc32 order, until --turn-n scenes have a decision 30-70 m before the
    # first strict-turn entry ahead (rear axle; the train64 turns mostly start 2-28 m before their entry: no approach to push)
    t64 = [x.split()[0] for x in open(a.turns) if x.strip()]
    rest = sorted(set(scene.tolist()) - set(t64) - set(out), key=lambda x: zlib.crc32(x.encode()))
    nturn = 0
    for sc in t64 + rest:
        if nturn >= a.turn_n:
            break
        rows = np.flatnonzero(scene == sc)
        if not len(rows):
            why[sc] = dict(cls="turn", reason="not in the pool")
            continue
        g = sj[rows[0]]
        TT = LI.strict_turns(X["route"][g].astype(np.float64), float(X["hlen"][g]))
        r_first = rows[np.argmin(k[rows])]
        ent = [e for e, _ in TT if e >= s0[r_first]]
        if not ent:
            if sc in t64:
                why[sc] = dict(cls="turn", reason="no strict turn ahead on the context route")
            continue
        r, how = pick(rows, ent[0] - s0[rows], W_TURN)
        if r is None:
            if sc in t64:
                why[sc] = dict(cls="turn", reason="no decision 30-70 m before the entry at >= 3 m/s", d_k0=float(ent[0] - s0[r_first]))
            continue
        if sc in out:                                   # a scene in both sets keeps its stop line
            out[sc]["also_turn"] = True
            continue
        out[sc] = dict(cls="turn", kp=int(k[r]), d=float(ent[0] - s0[r]), how=how, v=float(speed[r]), entry=float(ent[0]), name=str(names[r]),
                       train64=sc in t64)
        nturn += 1
    rec = dict(pool=a.pool, stop_scenes=nstop, turn_scenes_train64=len(t64), turn_n=a.turn_n, picked=len(out),
               by={f"{c}/{h}": sum(v["cls"] == c and v["how"] == h for v in out.values()) for c in ("stop", "turn") for h in ("in", "near", "short")},
               turn_train64=sum(bool(v.get("train64")) for v in out.values()), skipped=why, scenes=out)
    json.dump(rec, open(a.out, "w"), indent=1)
    print(json.dumps({k_: v for k_, v in rec.items() if k_ not in ("scenes", "skipped")}), "skipped", len(why))


def fprep(a):
    """A fast stage's scene list and perturbation files: --count scenes per class not run by an earlier stage (fsel order), or all."""
    sel = json.loads(Path(a.sel).read_text())["scenes"]
    st = Path(a.stage)
    done = {x.split()[0] for f in glob.glob(str(st.parent / "*" / "list.txt")) if Path(f).parent != st for x in open(f) if x.strip()}
    todo = [s for s in sorted(sel) if s not in done]
    pick = todo if a.count == "all" else [s for c in ("stop", "turn") for s in [x for x in todo if sel[x]["cls"] == c][: int(a.count)]]
    (st / "list.txt").write_text("".join(f"{s}\n" for s in pick))
    for lv in a.levels.split(","):
        (st / f"pt_L{round(float(lv) * 100)}.txt").write_text("# scene k g s n\n" + "".join(f"{s} {sel[s]['kp']} 1.0 {float(lv)} {a.n}\n" for s in pick))
    print(len(pick), "scenes (", sum(sel[s]["cls"] == "stop" for s in pick), "stop ), levels", a.levels, "n", a.n, "->", st)


def fread(a):
    """One fast stage: every pushed launch L<lvl> against the identity launch `id` of the same scenes."""
    st = Path(a.stage)
    sel = json.loads(Path(a.sel).read_text())["scenes"]
    runs = {p.name: summary_runs(sorted(glob.glob(str(p / "c*/run")))) for p in sorted(st.iterdir()) if p.is_dir() and (p.name == "id" or p.name.startswith("L"))}
    idr = runs.pop("id", {})
    res = []
    for lvl, R in sorted(runs.items()):
        for sc, D in sorted(R.items()):
            u = sel[sc]
            A = drive_records(D).get(sc, [])
            B = drive_records(idr[sc]).get(sc, []) if sc in idr else []
            x = dict(scene=sc, level=lvl, cls=u["cls"], kp=u["kp"], d=u["d"], how=u["how"], run=D, n_rec=len(A))
            if not A:                                       # no driver session (the scene did not open on this cache)
                res.append(x)
                continue
            pt = [r["perturb"] for r in A if "perturb" in r]
            x |= dict(n_pushed=len(pt), push=pt[0] if pt else None)
            n = pt[0]["n"] if pt else 0
            for lab, recs in (("p", A), ("i", B)):
                rk = {int(r["k"]): r for r in recs}
                if u["kp"] not in rk:
                    continue
                ks = sorted(k_ for k_ in rk if k_ >= u["kp"])
                P = np.array([rk[k_]["anchor"][:2] for k_ in ks], float)
                v = np.array([speed_of(rk[k_]) for k_ in ks])
                tr = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
                x[f"v_kp_{lab}"] = float(v[0])
                x[f"v_win_{lab}"] = float(v[n]) if n < len(v) else np.nan
                x[f"v_max_win_{lab}"] = float(v[: n + 3].max())
                j = np.flatnonzero(tr >= u["d"])
                x[f"v_at_{lab}"] = float(np.interp(u["d"], tr[j[0] - 1: j[0] + 1], v[j[0] - 1: j[0] + 1])) if len(j) and j[0] > 0 else np.nan
                x[f"k_at_{lab}"] = int(ks[j[0]]) if len(j) else None
                x[f"travel_{lab}"] = float(tr[-1])
                x[f"kz_{lab}"] = zeros_of(D if lab == "p" else idr[sc], sc)
                x[f"n_dec_{lab}"] = len(recs)
            if B:
                x["plan_pre_maxdiff"] = max((float(np.abs(np.subtract(r["poses"], {q["k"]: q for q in B}[r["k"]]["poses"])).max())
                                             for r in A if r["k"] < u["kp"] and r["k"] in {q["k"] for q in B}), default=None)
            res.append(x)
    summ = {}
    for lvl in sorted({x["level"] for x in res}):
        for c in ("stop", "turn", "all"):
            X = [x for x in res if x["level"] == lvl and (c == "all" or x["cls"] == c)]
            if not X:
                continue
            g = lambda key: np.array([x.get(key, np.nan) for x in X], float)  # noqa: E731
            rw, ra = g("v_win_p") / g("v_win_i"), g("v_at_p") / g("v_at_i")
            summ[f"{lvl}/{c}"] = dict(sessions=len(X), v_kp_p=float(np.nanmean(g("v_kp_p"))), v_kp_i=float(np.nanmean(g("v_kp_i"))),
                                      v_win_p=float(np.nanmean(g("v_win_p"))), v_win_i=float(np.nanmean(g("v_win_i"))),
                                      ratio_win_median=float(np.nanmedian(rw)), ratio_win_q25_q75=[float(np.nanpercentile(rw, 25)), float(np.nanpercentile(rw, 75))],
                                      ratio_win_in_125_140=float(np.nanmean((rw >= 1.25) & (rw <= 1.40))),
                                      reached_p=int(np.isfinite(g("v_at_p")).sum()), reached_i=int(np.isfinite(g("v_at_i")).sum()),
                                      v_at_p=float(np.nanmean(g("v_at_p"))) if np.isfinite(g("v_at_p")).any() else None,
                                      v_at_i=float(np.nanmean(g("v_at_i"))) if np.isfinite(g("v_at_i")).any() else None,
                                      ratio_at_median=float(np.nanmedian(ra)) if np.isfinite(ra).any() else None,
                                      zero_after_kp_p=int(sum(x.get("kz_p", -1) >= x["kp"] for x in X)), zero_after_kp_i=int(sum(x.get("kz_i", -1) >= x["kp"] for x in X)),
                                      zero_any_p=int(sum(x.get("kz_p", -1) >= 0 for x in X)), zero_any_i=int(sum(x.get("kz_i", -1) >= 0 for x in X)),
                                      pre_identical=int(sum(x.get("plan_pre_maxdiff") == 0.0 for x in X)))
    json.dump(dict(stage=str(st), summary=summ, sessions=res), open(a.out, "w"), indent=1, default=float)
    print(json.dumps(summ, indent=1, default=float))


def frows(a):
    sel = json.loads(Path(a.sel).read_text())["scenes"]
    kz = {}
    for f in a.reads.split(","):
        for x in json.loads(Path(f).read_text())["sessions"]:
            kz[(x["level"], x["scene"])] = x.get("kz_p", -1)
    srcs = []
    for spec in a.geo.split(","):
        lvl, g = spec.split("=")
        M = np.load(ROWS / g / "meta.npz")
        rows = {}
        for nm, sc, k in zip(M["names"].tolist(), M["scene"].tolist(), M["k"].astype(int).tolist()):
            u = sel.get(sc)
            if u is None or k < u["kp"]:
                continue
            z = kz.get((lvl, sc), -1)
            rows[nm] = dict(mode="self", push=lvl, cls=u["cls"], how=u["how"], kp=str(u["kp"]), post_zero="1" if 0 <= z <= k else "0",
                            win="1" if k < u["kp"] + a.n else "0")
        srcs.append(dict(geo=g, prefix=lvl, fut="path", rows=rows))
    json.dump(dict(kind="fast", sources=srcs), open(a.out, "w"), indent=1)
    print({s["prefix"]: len(s["rows"]) for s in srcs}, "->", a.out)


# ---------------------------------------------------------------- pool
def _sub(z: dict, idx, n_src):
    return {k: (v[idx] if (k not in NONROW and v.ndim >= 1 and len(v) == n_src) else v) for k, v in z.items()}


def assemble(a):
    sel = json.loads(Path(a.sel).read_text())
    out_p, out_r = CR / f"on1_{a.name}", ROWS / a.name
    out_p.mkdir(parents=True, exist_ok=True), out_r.mkdir(parents=True, exist_ok=True)
    T, M, AG, RD, DV, AL, R2 = [], [], [], [], [], [], []
    tagk = sorted({k for s in sel["sources"] for r in s["rows"].values() for k in r})
    for s in sel["sources"]:
        g, pre = s["geo"], s["prefix"]
        tab, meta = dict(np.load(CR / f"on1_{g}" / "tab.npz")), dict(np.load(ROWS / g / "meta.npz"))
        nm = tab["names"]
        n_src = len(nm)
        assert (meta["names"] == nm).all()
        idx = np.array([i for i, x in enumerate(nm.tolist()) if x in s["rows"]], int)
        assert len(idx) == len(s["rows"]), f"{g}: {len(s['rows']) - len(idx)} selected rows not in the geo rows"
        new = np.array([f"on:{pre}:{x[3:]}" for x in nm[idx].tolist()])
        t = _sub(tab, idx, n_src)
        recs = {}
        for run in sorted(set(meta["run"][idx].tolist())):
            for sc, rr in drive_records(run).items():
                for r in rr:
                    recs[(run, sc, int(r["k"]))] = r
        sm = np.array([recs[(meta["run"][i], meta["scene"][i], int(meta["k"][i]))].get("poses_model", recs[(meta["run"][i], meta["scene"][i], int(meta["k"][i]))]["poses"])
                       for i in idx], np.float32).reshape(-1, 8, 3)
        t["served_model"] = sm
        t["fut_path"] = t["fut"].copy()
        if s["fut"] == "served":                       # R: the expert's served plan at the decision, in the row's own (anchor) frame
            t["fut"] = t["served"].astype(np.float32).copy()
        for k in tagk:
            t[k] = np.array([str(s["rows"][x].get(k, "")) for x in nm[idx].tolist()])
        t["dev"] = np.array(["1" if is_dev(x) else "0" for x in t["log"].tolist()])
        t["names"] = new
        T.append(t)
        m = _sub(meta, idx, n_src)
        m["names"] = new
        M.append(m)
        for f, L in (("agent", AG), ("road", RD), ("drivable", DV)):
            z = _sub(dict(np.load(CR / f"on1_{g}" / f"{f}.npz")), idx, n_src)
            assert (z["tokens"] == nm[idx]).all()
            z["tokens"] = new
            L.append(z)
        al = dict(np.load(ROWS / g / "all.npz"))
        rmap = np.full(n_src + 1, -1)
        rmap[idx] = np.arange(len(idx)) + sum(len(x["names"]) for x in T[:-1])
        al["excl"] = np.where((al["excl"] == "") & (rmap[al["row"]] < 0), "unselected", al["excl"])
        al["row"] = np.where(al["row"] >= 0, rmap[al["row"]], -1)
        al["src"] = np.full(len(al["row"]), pre)
        AL.append(al)
        R2.append(json.loads((ROWS / g / "r2.json").read_text()))

    def cat(L):
        keys = set.intersection(*[set(z) for z in L])
        return {k: (np.concatenate([z[k] for z in L]) if (k not in NONROW and np.ndim(L[0][k]) >= 1 and k != "classes") else L[0][k]) for k in keys}
    tab = cat(T)
    assert len(set(tab["names"].tolist())) == len(tab["names"]) and np.isfinite(tab["fut"]).all()
    np.savez(out_p / "tab.npz", **tab)
    meta = cat(M)
    np.savez(out_r / "meta.npz", **meta)
    for f, L in (("agent", AG), ("road", RD), ("drivable", DV)):
        np.savez(out_p / f"{f}.npz", **cat(L))
        (out_r / f"{f}.npz").unlink(missing_ok=True)
        (out_r / f"{f}.npz").symlink_to(out_p / f"{f}.npz")
    np.savez(out_r / "all.npz", **cat(AL))
    an = sum(r["act_n"] for r in R2)
    r2 = dict(rollouts=sum(r["rollouts"] for r in R2), with_sim=sum(r["with_sim"] for r in R2), ego_max_m=max(r["ego_max_m"] for r in R2),
              act_n=an, act_max_m=max(r["act_max_m"] for r in R2), act_share_within=sum(r["act_share_within"] * r["act_n"] for r in R2) / max(an, 1),
              frame_excluded=[x for r in R2 for x in r["frame_excluded"]], sources={s["geo"]: r for s, r in zip(sel["sources"], R2)})
    r2["pass"] = all(r["pass"] for r in R2)
    (out_r / "r2.json").write_text(json.dumps(r2, indent=1) + "\n")
    (out_r / "sel.json").write_text(json.dumps(sel, indent=1, default=float))
    cnt = {k: {v: int((tab[k] == v).sum()) for v in sorted(set(tab[k].tolist()))} for k in tagk + ["dev"]}
    print(json.dumps(dict(pool=str(out_p), rows=len(tab["names"]), scenes=len(set(meta["scene"].tolist())), tags=cnt)))


def strip(a):
    from concurrent.futures import ProcessPoolExecutor
    fs = sorted(f for r in a.runs for f in glob.glob(f"{r}/driver-logs/dump/*.npz"))
    with ProcessPoolExecutor(16) as ex:
        print("stripped", sum(ex.map(_strip1, fs, chunksize=64)), "of", len(fs))


def _strip1(f):
    z = dict(np.load(f))
    if "cur" not in z:
        return 0
    del z["cur"]
    np.savez_compressed(f + ".tmp.npz", **z)
    os.replace(f + ".tmp.npz", f)
    return 1


def report(a):
    tab, rd = np.load(CR / f"on1_{a.name}" / "tab.npz"), ROWS / a.name
    tags = [k for k in tab.files if tab[k].dtype.kind == "U" and k not in ("names", "log")]
    rec = dict(name=a.name, rows=int(len(tab["names"])), logs=len(set(tab["log"].tolist())),
               tags={k: {v: int((tab[k] == v).sum()) for v in sorted(set(tab[k].tolist()))} for k in tags})
    for k in ("kind", "push", "cls"):
        if k in tags:
            rec[f"{k}_x_dev"] = {f"{v}/{d}": int(((tab[k] == v) & (tab["dev"] == d)).sum()) for v in sorted(set(tab[k].tolist())) for d in ("0", "1")}
    if "push" in tags and "cls" in tags:
        rec["push_x_cls"] = {f"{p}/{c}": int(((tab["push"] == p) & (tab["cls"] == c)).sum()) for p in sorted(set(tab["push"].tolist())) for c in sorted(set(tab["cls"].tolist()))}
    if "unit" in tags:
        rec["units"] = len(set(tab["unit"].tolist()))
        rec["units_dev"] = len(set(tab["unit"][tab["dev"] == "1"].tolist()))
    files = dict(r1=rd / "r1.json", r2=rd / "r2.json", gates=rd / "gates.json", report=rd / "report.json",
                 lon=DATA / "runs/onpol_nuplan/lon/gates" / f"lon_pool-{a.name}.json", kc=CR / f"on1_{a.name}" / "kc_items.json")
    for k, f in files.items():
        if f.exists():
            z = json.loads(f.read_text())
            rec[k] = ({q: z.get(q) for q in ("pass", "a", "b", "c", "tags")} if k == "r1" else
                      {q: z.get(q) for q in ("pass", "ego_max_m", "act_share_within", "rollouts", "with_sim")} if k == "r2" else
                      {q: z.get(q) for q in ("pass", "fail", "items", "R1", "lane_ok")} if k == "gates" else
                      {q: z.get(q) for q in ("rows", "excluded", "imit_ade_mean", "agent_active", "road_active", "road_off_at_t0", "targets")} if k == "report" else
                      {q: z.get(q) for q in ("rows", "rows_ok", "not_ok", "items_in_pack", "items_dropped_by_the_0p2_percent_rule")} if k == "lon" else z.get("items"))
        else:
            rec[k] = None
    fe = DATA / "runs/route_mem/feat" / f"on1_{a.name}.npz"
    if fe.exists():
        f = np.load(fe)
        rec["feat"] = dict(rows=int(len(f["names"])), ok=float(f["ok"].mean()), route0_share_005=float((f["route0_d"][f["ok"]] <= 0.05).mean()) if f["ok"].any() and "route0_d" in f.files else None,
                           same_names=bool(np.array_equal(f["names"], tab["names"])))
    rz = CR / f"on1_{a.name}" / "route.npz"
    if rz.exists():
        rec["route_ok"] = float(np.load(rz)["ok"].mean())
    fr = CR / f"on1_{a.name}@warp" / "front.npy"
    rec["front_rows"] = int(np.load(fr, mmap_mode="r").shape[0]) if fr.exists() else None
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"report_{a.name}.json").write_text(json.dumps(rec, indent=1, default=float) + "\n")
    print(json.dumps(rec, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("rcheck")
    p.add_argument("--stage", required=True), p.add_argument("--prior", default="")
    p.add_argument("--prior-runs", default="", help="stage dirs whose D<d>/c*/run hold the source's own D sessions")
    p = sp.add_parser("rsel")
    p.add_argument("--stage", required=True), p.add_argument("--geo", required=True), p.add_argument("--out", required=True)
    p.add_argument("--rows-csv", default="", help="a rewind n-yield rows.csv: keep exactly its label & within_tol rows (scene, delta, k)")
    p.add_argument("--check", default="", help="S/check.json: units must be within 0.15 m of the source's own session before h")
    p = sp.add_parser("fsel")
    p.add_argument("--pool", default="L18RSG"), p.add_argument("--turns", required=True), p.add_argument("--out", required=True)
    p.add_argument("--turn-n", type=int, default=64, help="turn scenes (train64 first, then the pool in crc32 order)")
    p = sp.add_parser("fprep")
    p.add_argument("--sel", required=True), p.add_argument("--stage", required=True), p.add_argument("--count", required=True)
    p.add_argument("--levels", required=True), p.add_argument("--n", type=int, default=2)
    p = sp.add_parser("fread")
    p.add_argument("--stage", required=True), p.add_argument("--sel", required=True), p.add_argument("--out", required=True)
    p = sp.add_parser("frows")
    p.add_argument("--reads", required=True), p.add_argument("--sel", required=True), p.add_argument("--geo", required=True)
    p.add_argument("--out", required=True), p.add_argument("--n", type=int, default=2)
    p = sp.add_parser("assemble")
    p.add_argument("--name", required=True), p.add_argument("--sel", required=True)
    p = sp.add_parser("strip")
    p.add_argument("--runs", nargs="+", required=True)
    p = sp.add_parser("report")
    p.add_argument("--name", required=True), p.add_argument("--out", required=True)
    a = ap.parse_args()
    dict(rcheck=rcheck, rsel=rsel, fsel=fsel, fprep=fprep, fread=fread, frows=frows, assemble=assemble, strip=strip, report=report)[a.cmd](a)


if __name__ == "__main__":
    main()
