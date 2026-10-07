"""op_parity on WOD train (plans/2026-10-07-wod-parity-prereg.md, results/wod_parity.md): the P2 input-parity recipe (pp_train.py) trained on
WOD-E2E train frames with WOD's own inputs, read on WOD val through the existing harness (scripts/wod_zeroshot_openpilot.py --bias, pp_wod.py bias).

  prep    (op-train env, one GPU) the WOD train trunk cache processed/op_adapt/wodtrain (experiments/op_adapt_r1/lib/op_adapt_cache.py wodtrain:
          shipped Cinque's stage-3 output `permute_73` of the harness renderer's road + wide frames, image pair (f - 2, f), at every EVEN frame of
          2 103 contiguous streams) -> stage 4 -> `view_39` tokens -> $DATA_DIR/runs/op_parity/cache/wod_r2/
            ticks.npy      (T, 32, 512) fp16, one per cached slot, stream after stream
            front_idx.npy  (N, 9) the 9 policy context slots of a row (frames f - 16 .. f step 2 = 0.2 s, oldest first), ALL real: the harness feeds
                           10 s of frames, so its 9 slots are real frames (the navtrain recipe had 8 + a zero slot)
            tab.npz        names (frame), log (sequence), ego (N, 20) = pp_wod.wod_ego (the eval mapping, unchanged), pose, intent, fut (N, 8, 3),
                           cam (FRONT camera position, op_calib), lht False, speed
            teacher.npz    shipped Cinque (port fp16) on the same 9 slots: out, plan, di, pi
          Rows: slot j >= 9 of a stream (its 9 slots and the oldest slot's previous frame are real frames), the frame has the logged 5 s future,
          its sequence is in wod/r2-train or wod/r2-dev. Targets (fut): x, y at 0.5 .. 4 s = future_states steps 2, 4, .., 16 (exact on the 4 Hz
          lattice); yaw = chord heading of the positions 0.25 s either side, held from the previous key when that chord is < 0.1 m (0 at t0).
          Pilot: cache/wod_pilot (own tab / front_idx / teacher, ticks.npy symlinked): PILOT rows drawn uniformly (seed 0) from the r2-train and
          r2-dev rows, sized like the navtrain pilot (4 874 / 526).
  check   (op-train env, one GPU) the same token path on WOD val (processed/op_adapt/wodval: same renderer and protocol) vs the stored shipped
          harness run preds/op_cinque, on the rater + extra frames at slot >= 9 of a val stream: plan xy difference (prereg gate: mean < 0.05 m)
  report  (jevdrive env, CPU) arm groups vs each other on WOD val: RFS (479 rater frames), ADE@3s / @5s (1 437 frames), night / day (night_gap luma
          labels), the night-gap difference-in-differences; paired bootstraps over sequences -> results/wod_parity/<name>.{csv,json}

  python experiments/op_parity/scripts/wod_parity.py prep
  python experiments/op_parity/scripts/wod_parity.py report --name pilot --arms WP2=WP2-pilot-s0 WP1=WP1-pilot-s0 --ref shipped
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_pl.Path(__file__).parent)]
import argparse, json, time  # noqa: E401,E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

from jevdrive.common import data_dir  # noqa: E402

SRC = data_dir() / "processed" / "op_adapt"
CACHE = data_dir() / "runs" / "op_parity" / "cache"
NCTX = 9
AT = (0.275, 0.525)
PILOT = (4874, 526)
SPLIT = "wod/r2"
B = 4000


def fut_targets(future: np.ndarray) -> np.ndarray:
    """(n, 20, 3) WOD future_states -> (n, 8, 3) x, y, yaw at 0.5 .. 4 s, rear axle (the pp_prep / pp_train target layout)."""
    n = len(future)
    P = np.concatenate([np.zeros((n, 1, 2)), future[..., :2].astype(np.float64)], 1)    # index i <-> t = 0.25 i
    keys = np.arange(2, 17, 2)
    out = np.zeros((n, 8, 3))
    out[:, :, :2] = P[:, keys]
    prev = np.zeros(n)
    for j, i in enumerate(keys):
        ch = P[:, i + 1] - P[:, i - 1]
        prev = np.where(np.linalg.norm(ch, axis=-1) >= 0.1, np.arctan2(ch[:, 1], ch[:, 0]), prev)
        out[:, j, 2] = prev
    return out.astype(np.float32)


def calib():
    from jevdrive import drive_backbones as DB
    from jevdrive import wod_zeroshot as Z
    return json.loads((Z.root() / "op_calib.json").read_text()) | json.loads((DB.root() / "op_calib_trainval.json").read_text())


def stream_rows(streams, want, need_future=True):
    """streams (plan json) -> slot offsets per stream, rows [(stream index, slot j, frame name)] for slots j >= NCTX whose name is in `want`."""
    from jevdrive import waymo as W
    df = W.load_index()
    key = dict(zip(W.frame_names(df), range(len(df))))
    hf = df.has_future.to_numpy()
    off, T, rows = [], 0, []
    for si, s in enumerate(streams):
        off.append(T)
        fr = np.array([int(n.rsplit("-", 1)[1]) for n in s["names"]])
        assert (np.diff(fr) == 2).all(), f"stream {s['key']} not contiguous on the even frames"
        for j in range(NCTX, len(s["names"])):
            n = s["names"][j]
            if n in want and (not need_future or hf[key[n]]):
                rows.append((si, j, n, key[n]))
        T += len(s["names"])
    return np.array(off), T, rows


def bounded(ex, fn, items, depth):
    """ex.map(fn, items) in order with at most `depth` items in flight (a trunk file is ~29 MB)."""
    from collections import deque
    it, q = iter(items), deque()
    for x in it:
        q.append(ex.submit(fn, x))
        if len(q) >= depth:
            break
    while q:
        yield q.popleft().result()
        for x in it:
            q.append(ex.submit(fn, x))
            break


def encode(streams, src, rows, off, T, ticks_out=None, dev="cuda"):
    """Stage 4 of every slot of `streams` (trunk files under src) -> tokens (into ticks_out memmap if given); teacher (shipped port) on `rows`.
    Returns out (N, n_di), plan (N, 33, 15), di, pi."""
    import torch
    from jevdrive import op_adapt as A
    net = A.load("cinque", torch.float16).to(dev).eval()
    di, pi = A.distill_index(net.slices), A.plan_index(net.slices)
    by = {}
    for r, (si, j, _, _) in enumerate(rows):
        by.setdefault(si, []).append((r, j))
    todo = sorted(by) if ticks_out is None else range(len(streams))
    t_out = np.zeros((len(rows), len(di)), np.float32)
    t_plan = np.zeros((len(rows), 33, 15), np.float32)
    tc = torch.tensor([[1.0, 0.0]], device=dev)

    def load(si):
        z = np.load(src / f"{streams[si]['key']}.npz")
        assert list(z["names"]) == list(streams[si]["names"]) and int(z["stride"]) == 1, streams[si]["key"]
        return si, z["trunk"]
    t0, done = time.time(), 0
    with ThreadPoolExecutor(8) as ex, torch.no_grad():
        for si, trunk in bounded(ex, load, todo, 16):
            x = torch.from_numpy(trunk).to(dev)
            tok = torch.cat([net.run_batched({A.TRUNK_OUT: x[i:i + 256, None].to(net.dtype)}, ["view_39"])["view_39"].reshape(-1, *A.H_SHAPE)
                             for i in range(0, len(x), 256)])
            if ticks_out is not None:
                ticks_out[off[si]:off[si] + len(tok)] = tok.cpu().numpy()
            if si in by:
                r, j = map(np.array, zip(*by[si]))
                ctx = torch.as_tensor(j[:, None] - np.arange(NCTX - 1, -1, -1)[None], device=dev)
                H = tok[ctx]
                o = A._policy(net, H, AT, tc.expand(len(H), 2), None)["outputs"].float()
                t_out[r] = o[:, di].cpu().numpy()
                t_plan[r] = o[:, pi].cpu().numpy().reshape(-1, 33, 15)
            done += 1
            if done % 200 == 0:
                print(f"{done}/{len(todo)} streams, {time.time() - t0:.0f} s", flush=True)
    return t_out, t_plan, di, pi


# ---------------------------------------------------------------- prep (train cache)
def cmd_prep(a):
    import pp_wod as PW
    from jevdrive import waymo as W
    from jevdrive.data import splits
    from jevdrive.run import Run
    tr, dv, val = splits.load(f"{SPLIT}-train"), splits.load(f"{SPLIT}-dev"), splits.load("wod/val")
    splits.check_disjoint(tr, dv, val)
    streams = json.loads((SRC / "wod_train_plan.json").read_text())["streams"]
    if a.limit:
        streams = streams[: a.limit]
    out = CACHE / ("wod_r2" + (f"-first{a.limit}" if a.limit else ""))
    out.mkdir(parents=True, exist_ok=True)
    with Run("op_parity", "wod-prep" + (f"-first{a.limit}" if a.limit else ""), config=vars(a)) as run:
        run.use_split(tr), run.use_split(dv), run.use_split(val)
        df = W.load_index()
        seq = df.sequence.astype(str).to_numpy()
        names_all = W.frame_names(df)
        ok = (df.split.astype(str).to_numpy() == "train") & (tr.mask(seq) | dv.mask(seq))
        off, T, rows = stream_rows(streams, set(names_all[ok]))
        idx = np.array([r[3] for r in rows])
        past, fut = W.load_ego()
        past, fut, intent = past[idx], fut[idx], df.intent.to_numpy()[idx]
        ego, pose = PW.wod_ego(past, intent)
        cal = calib()
        sq = seq[idx]
        cam = np.array([np.array(cal[s]["1"]["extrinsic"]).reshape(4, 4)[:3, 3] for s in sq], np.float32)
        tab = dict(names=np.array([r[2] for r in rows]), log=sq, ego=ego, pose=pose.astype(np.float32), intent=intent, fut=fut_targets(fut),
                   cam=cam, lht=np.zeros(len(rows), bool), speed=W.past_kinematics(past)["v"].astype(np.float32))
        fi = np.array([off[si] + j - np.arange(NCTX - 1, -1, -1) for si, j, _, _ in rows], np.int64)
        assert np.isfinite(tab["fut"]).all() and np.isfinite(ego).all()
        np.savez(out / "tab.npz", **tab)
        np.save(out / "front_idx.npy", fi)
        run.info(f"{len(streams)} streams, {T} slots, {len(rows)} rows ({int(tr.mask(sq).sum())} r2-train / {int(dv.mask(sq).sum())} r2-dev), "
                 f"intent counts {np.bincount(intent, minlength=4).tolist()}")
        tk = np.lib.format.open_memmap(out / "ticks.npy", "w+", np.float16, (T, *(32, 512)))
        t_out, t_plan, di, pi = encode(streams, SRC / "wodtrain", rows, off, T, tk)
        tk.flush()
        np.savez(out / "teacher.npz", out=t_out, plan=t_plan, di=di, pi=pi)
        bad = [i for i in range(0, T, max(1, T // 2000)) if not np.abs(tk[i].astype(np.float32)).any()]
        assert not bad, f"all-zero token rows: {bad[:5]}"
        # pilot subset: uniform rows of r2-train / r2-dev (seed 0), ticks shared
        rng = np.random.default_rng(0)
        sel = np.sort(np.concatenate([rng.choice(np.flatnonzero(m), min(k, int(m.sum())), replace=False)
                                      for m, k in ((tr.mask(sq), PILOT[0]), (dv.mask(sq), PILOT[1]))]))
        pdir = CACHE / ("wod_pilot" + (f"-first{a.limit}" if a.limit else ""))
        pdir.mkdir(exist_ok=True)
        if not (pdir / "ticks.npy").exists():
            (pdir / "ticks.npy").symlink_to(out / "ticks.npy")
        np.savez(pdir / "tab.npz", **{k: v[sel] for k, v in tab.items()})
        np.save(pdir / "front_idx.npy", fi[sel])
        np.savez(pdir / "teacher.npz", out=t_out[sel], plan=t_plan[sel], di=di, pi=pi)
        st = dict(streams=len(streams), slots=int(T), rows=len(rows), rows_train=int(tr.mask(sq).sum()), rows_dev=int(dv.mask(sq).sum()),
                  sequences=int(len(set(sq))), pilot_rows=int(len(sel)), pilot_train=int(tr.mask(sq[sel]).sum()),
                  pilot_sequences=int(len(set(sq[sel]))), intent=np.bincount(intent, minlength=4).tolist(),
                  ego_mean=ego.mean(0).round(3).tolist(), ego_std=ego.std(0).round(3).tolist(),
                  fut4s_m_p50=float(np.median(np.linalg.norm(tab["fut"][:, -1, :2], axis=-1))), cam_mean=cam.mean(0).round(3).tolist(),
                  ticks_gb=T * 32 * 512 * 2 / 2 ** 30)
        (out / "meta.json").write_text(json.dumps(st, indent=1))
        run.summary |= st
        run.info(json.dumps(st))


# ---------------------------------------------------------------- check (val: cached token path vs the harness)
def cmd_check(a):
    from jevdrive import wod_zeroshot as Z
    from jevdrive.run import Run
    S = Z.load_sets()
    want = set(np.concatenate([S[k]["name"] for k in ("rater", "extra")]).astype(str).tolist())
    streams = json.loads((SRC / "wod_val_plan.json").read_text())["streams"]
    with Run("op_parity", "wod-check", config=vars(a)) as run:
        off, T, rows = stream_rows(streams, want, need_future=False)
        run.info(f"{len(rows)} of {len(want)} rater + extra frames sit at slot >= {NCTX} of a val stream")
        _, plan, _, _ = encode(streams, SRC / "wodval", rows, off, T)
        d = Z.root("preds", "op_cinque")
        ref = np.stack([np.load(d / f"{r[2]}.npz")["plan_pos"] for r in rows]).astype(np.float64)
        e = np.linalg.norm(plan[:, :, :2] - ref[:, :, :2], axis=-1)
        st = dict(n=len(rows), mean_m=float(e.mean()), p99_m=float(np.percentile(e, 99)), max_m=float(e.max()),
                  mean_at_4s_m=float(e[:, 20].mean()), gate_mean_lt_0p05=bool(e.mean() < 0.05))
        run.summary |= st
        run.info(json.dumps(st))
        (data_dir() / "runs" / "op_parity" / "wod" / "check.json").write_text(json.dumps(st, indent=1))


# ---------------------------------------------------------------- report
def cmd_report(a):
    import pandas as pd
    import pp_wod as PW
    from jevdrive import waymo as W
    S = PW.frames()
    r, x = S["rater"], S["extra"]
    seq = np.concatenate([r["sequence"], x["sequence"]]).astype(str)
    names = np.concatenate([r["name"], x["name"]]).astype(str)
    fut = np.concatenate([r["future"], x["future"]])[..., :2].astype(np.float64)
    cl = r["cluster"].astype(str)
    nr = len(r["name"])
    lum = pd.read_csv(_R / "experiments/leaderboard_audit/results/night_gap/seq_lum.csv").set_index("sequence").l.reindex(seq).to_numpy()
    lab = np.where(lum < 50, "night", np.where(lum < 120, "dusk", "day"))
    groups = dict(g.split("=", 1) for g in a.arms)
    groups = {"shipped": "shipped"} | {k: v for k, v in groups.items()}
    tags = sorted({t for v in groups.values() for t in v.split("+")})
    P = {t: PW.load_preds(t, names) for t in tags}
    rfs = {t: np.asarray(W.rater_feedback_score(P[t][:nr], r["traj"].astype(np.float64), r["scores"].astype(np.float64), W.init_speed(r["past"])),
                         float) for t in tags}
    err = {t: np.linalg.norm(P[t] - fut, axis=-1) for t in tags}
    per = {}                                                                          # group -> per-frame seed-mean metric arrays
    for g, v in groups.items():
        ts = v.split("+")
        per[g] = dict(rfs=np.mean([rfs[t] for t in ts], 0), ade3=np.mean([err[t][:, :12].mean(1) for t in ts], 0),
                      ade5=np.mean([err[t].mean(1) for t in ts], 0), seeds=ts)
    cr, ur = pd.factorize(pd.Series(seq[:nr]))
    ca, ua = pd.factorize(pd.Series(seq))
    ir = [np.flatnonzero(cr == k) for k in range(len(ur))]
    ia = [np.flatnonzero(ca == k) for k in range(len(ua))]
    rng = np.random.default_rng(0)
    dr = [np.concatenate([ir[k] for k in rng.integers(len(ur), size=len(ur))]) for _ in range(B)]
    da = [np.concatenate([ia[k] for k in rng.integers(len(ua), size=len(ua))]) for _ in range(B)]
    ccode, cu = pd.factorize(pd.Series(cl))

    def rfs_f(s, i):                                                                 # W.rfs_by_cluster on rows i (cluster mean, then mean over present)
        c = ccode[i]
        n = np.bincount(c, minlength=len(cu))
        m = np.bincount(c, s[i], minlength=len(cu))
        return float((m[n > 0] / n[n > 0]).mean())
    labr = lab[:nr]
    sub_r = {"all": np.ones(nr, bool), "night": labr == "night", "day": labr == "day"}
    sub_a = {"all": np.ones(len(names), bool), "night": lab == "night", "day": lab == "day"}

    def stat(A, Bg, m, sub):
        """point and draws of A - B for metric m on subset sub (RFS: rater rows / sequences; ADE: all frames / sequences)."""
        if m == "rfs":
            f = lambda i: rfs_f(per[A]["rfs"], i) - rfs_f(per[Bg]["rfs"], i)  # noqa: E731
            msk, draws = sub_r[sub], dr
            pt = f(np.flatnonzero(msk))
        else:
            f = lambda i: per[A][m][i].mean() - per[Bg][m][i].mean()  # noqa: E731
            msk, draws = sub_a[sub], da
            pt = f(np.flatnonzero(msk))
        return pt, np.array([f(i[msk[i]]) for i in draws])
    rows = []
    for g in groups:
        row = {"arm": g, "tags": groups[g]}
        for m, sm in (("rfs", sub_r), ("ade3", sub_a), ("ade5", sub_a)):
            for sub, msk in sm.items():
                i = np.flatnonzero(msk)
                row[f"{m}_{sub}"] = rfs_f(per[g]["rfs"], i) if m == "rfs" else float(per[g][m][i].mean())
        rows.append(row)
    pairs = []
    for p in a.pairs:
        A, Bg = p.split(":")
        row = {"pair": f"{A} - {Bg}"}
        for m in ("rfs", "ade3", "ade5"):
            dd = {}
            for sub in ("all", "night", "day"):
                pt, dw = stat(A, Bg, m, sub)
                dd[sub] = (pt, dw)
                row[f"d_{m}_{sub}"], (row[f"d_{m}_{sub}_lo"], row[f"d_{m}_{sub}_hi"]) = pt, np.percentile(dw, [2.5, 97.5])
            if m == "rfs":                                                            # night-gap change: (A - B)_night - (A - B)_day, same draws
                pt = dd["night"][0] - dd["day"][0]
                row["dd_rfs_night_minus_day"], (row["dd_lo"], row["dd_hi"]) = pt, np.percentile(dd["night"][1] - dd["day"][1], [2.5, 97.5])
        pairs.append(row)
    for g, v in groups.items():                                                       # seed vs seed (noise reference)
        ts = v.split("+")
        if len(ts) == 2:
            per[f"{g}:s0"], per[f"{g}:s1"] = ({"rfs": rfs[t], "ade3": err[t][:, :12].mean(1), "ade5": err[t].mean(1)} for t in ts)
            pt, dw = stat(f"{g}:s0", f"{g}:s1", "rfs", "all")
            pairs.append({"pair": f"{ts[0]} - {ts[1]}", "d_rfs_all": pt, "d_rfs_all_lo": np.percentile(dw, 2.5), "d_rfs_all_hi": np.percentile(dw, 97.5)})
    lg = np.linalg.norm(fut[:, -1], axis=-1)
    mv = lg > 2.0
    speed = {t: float(np.median(np.linalg.norm(P[t][:, -1], axis=-1)[mv] / lg[mv])) for t in tags}
    meta = dict(n_rater=int(nr), n_frames=len(names), n_night_rater=int(sub_r["night"].sum()), n_day_rater=int(sub_r["day"].sum()),
                n_night_all=int(sub_a["night"].sum()), n_day_all=int(sub_a["day"].sum()), B=B, groups=groups,
                plan_5s_displacement_over_log_median=speed)
    out = _R / "experiments/op_parity/results/wod_parity"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"{a.name}_arms.csv", index=False)
    pd.DataFrame(pairs).to_csv(out / f"{a.name}_paired.csv", index=False)
    (out / f"{a.name}_meta.json").write_text(json.dumps(meta, indent=1))
    pd.set_option("display.width", 250, "display.max_columns", 99)
    print(pd.DataFrame(rows).to_string(float_format=lambda v: f"{v:.3f}"))
    print(pd.DataFrame(pairs).T.to_string(float_format=lambda v: f"{v:.3f}"))
    print(json.dumps(meta, indent=1))
    if a.gate is not None:                                                        # pilot gate (prereg): d RFS (first pair) >= threshold
        d = pairs[0]["d_rfs_all"]
        ok = d >= a.gate
        (out / f"{a.name}_gate.json").write_text(json.dumps({"pair": pairs[0]["pair"], "d_rfs": d, "threshold": a.gate, "pass": bool(ok)}))
        print(f"GATE {'PASS' if ok else 'STOP'}: {pairs[0]['pair']} d RFS {d:+.3f} vs threshold {a.gate:+.3f}")
        raise SystemExit(0 if ok else 3)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prep")
    p.add_argument("--limit", type=int, default=0, help="first n streams (smoke)")
    sp.add_parser("check")
    p = sp.add_parser("report")
    p.add_argument("--name", required=True)
    p.add_argument("--arms", nargs="+", required=True, help="GROUP=tag[+tag] (preds/op_cinque_<tag>; seed mean over the tags); shipped is implicit")
    p.add_argument("--pairs", nargs="+", required=True, help="A:B group differences (the first one is gated with --gate)")
    p.add_argument("--gate", type=float, default=None, help="pilot gate: stop (exit 3) if the first pair's d RFS < this")
    a = ap.parse_args()
    {"prep": cmd_prep, "check": cmd_check, "report": cmd_report}[a.cmd](a)
