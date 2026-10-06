"""op_parity training cache of the collected B2D dataset b2dc-train@v2 (experiments/b2d_collect, decision 152): everything pp_train reads, built
from the stored 20 Hz model frames with Cinque's FROZEN vision encoder. Native 0.2 s frame pairs: no warp, no interpolation.

-> $DATA_DIR/runs/op_parity/cache/b2d_v2/
  ticks.npy        (T, 32, 512) fp16  hidden tokens `view_39` of the pair (frame t - 4, frame t) at EVERY EVEN tick t >= 4 of every clip (clip after clip)
  front_idx.npy    (N, 8) int64       row -> its 8 context slots (t0 - 28, ..., t0 step 4 ticks = 0.2 s, oldest first) as indices into ticks.npy.
                                      pp_prep's front.npy row = ticks[front_idx[row]]; nothing is copied (the 8 slots of neighbouring rows are the
                                      same tokens: 8x smaller than a front.npy). Slots before tick 4 repeat tick 4 (`clamped` in extra.npz: the car
                                      stands at the spawn, so the earliest frame is the context)
  tab.npz          the pp_prep tab: names "<route>_<t0 :05d>", log = route id, ego (N, 20), pose / vel / acc (4 history poses), cmd, fut (N, 8, 3) logged
                   rear-axle poses at 0.5 .. 4 s, cam (1.59, 0, 1.86), lht False, speed
  teacher.npz      shipped Cinque on the same rows: out (N, 1086), plan (N, 33, 15), di, pi (pp_prep.make_teacher)
  hinge_labels.npz tokens = names, ok, sdf (N, 128, 96) fp16: the drivable SDF at t0 (op_probe's grid; lib/drivable_hinge.Hinge(..., footprint="mkz"))
  extra.npz        route, type, town, tick, clamped, turn_next / turn_dist (RAW distance (m) to the next LEFT / RIGHT junction option along the route,
                   so the command lookahead can be recalibrated), act_kappa / act_accel (PDM-Lite action at t0 + 0.2 s), ctl, target_speed, progress
  plan.json        clip table (rows, tick ranges, collision cut), counters; MANIFEST-like meta.json

Rows: every EVEN tick t0 (the SDF labels exist at even ticks) with a complete 4 s logged future. COLLISION RULE: a clip with a leaderboard collision
(33 of 998) is cut at the first one: the collision tick is the first tick whose ego position is within 1 m of the closest approach to the event's
logged location, and rows keep only t0 <= t_col - 5 s (the 4 s label window plus 1 s of pre-impact braking never touches the collision).
Resumable: `tokens` skips clips with a marker file.

  b2d_prep.py plan                          (CPU, tmux)  clip table, tab / extra / hinge labels / index files, preallocated ticks.npy + teacher
  b2d_prep.py tokens --shard i --of n       (GPU pool job per card) frames -> tokens + teacher of the shard's clips
  b2d_prep.py finish                        teacher.npz, meta.json
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_R / "experiments/b2d_collect/lib"),
                 str(_R / "experiments/op_parity/scripts")]
import argparse, csv, json, re, time  # noqa: E401,E402
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

from jevdrive.common import data_dir, n_cpus  # noqa: E402

SRC = data_dir() / "runs" / "b2d_collect" / "data" / "all"
OUT = data_dir() / "runs" / "op_parity" / "cache" / "b2d_v2"
CUT_S = 5.0                                    # collision rule: rows end 5 s (4 s window + 1 s) before the collision
CAM = (1.59, 0.0, 1.86)                        # b2d_collect rig: road + wide camera, rear-axle frame
N_OUT, N_PLAN = 1086, 495                      # pp_prep teacher widths (shipped Cinque distill / plan columns)
VERSION = "b2d1"


# ---------------------------------------------------------------- collision cut
def collision_tick(clip: _pl.Path, loc: np.ndarray) -> tuple:
    """(first collision tick or -1, closest approach m) from the leaderboard record's collision locations (CARLA x, y of the ego at the event)."""
    res = clip.parent / "results.json"
    rec = json.loads(res.read_text())["_checkpoint"]["records"][0]["infractions"]
    pts = [tuple(map(float, m)) for k, v in rec.items() if k.startswith("collisions_") for s in v
           for m in re.findall(r"at \(x=([-\d.]+), y=([-\d.]+)", s)]
    if not pts:
        return -1, 0.0
    best, dm = None, 0.0
    for x, y in pts:
        d = np.hypot(loc[:, 0] - x, loc[:, 1] - y)
        t = int(np.flatnonzero(d <= d.min() + 1.0)[0])
        if best is None or t < best:
            best, dm = t, float(d.min())
    return best, dm


def _clip_job(r: dict) -> dict:
    """One clip -> its rows (labels + sdf) as arrays."""
    clip = _pl.Path(r["clip"])
    lab = np.load(clip / "labels.npz")
    ego = np.load(clip / "ego.npz")
    sd = np.load(clip / "sdf.npz")
    n = len(lab["speed"])
    t_col, d_col = collision_tick(clip, ego["loc"][:, :2])
    tmax = n - 1 if t_col < 0 else t_col - int(CUT_S * 20)
    ok = lab["fut_ok"].copy()
    ok[tmax + 1:] = False
    ok[:4] = False
    t0 = np.flatnonzero(ok & (np.arange(n) % 2 == 0))
    out = dict(route=r["route_id"], clip=r["clip"], type=r["type"], town=r["town"], turn=r["turn"], n=n, t_col=t_col, d_col=d_col, n_rows=len(t0))
    if len(t0) == 0:
        return out | dict(t0=t0)
    st, sd_all = sd["ticks"], sd["sdf"]                                 # NpzFile reads a member on every access: load once
    j = np.searchsorted(st, t0)
    has = (j < len(st)) & (st[np.minimum(j, len(st) - 1)] == t0)
    sdf = np.zeros((len(t0), 128, 96), np.float16)
    sdf[has] = sd_all[j[has]]
    g = lambda k: lab[k][t0]  # noqa: E731
    cmd = g("cmd")
    return out | dict(t0=t0, sdf=sdf, sdf_ok=has, ego=g("ego"), hist=g("hist"), vel=g("vel"), acc=g("acc"), cmd=cmd, fut=g("fut"), speed=g("speed"),
                      turn_next=g("turn_next"), turn_dist=g("turn_dist"), act_kappa=g("act_kappa"), act_accel=g("act_accel"), ctl=g("ctl"),
                      target_speed=g("target_speed"), progress=g("progress"))


# ---------------------------------------------------------------- plan
def plan(a):
    from jevdrive.run import Run
    with open(SRC / "index.csv") as fh:
        idx = list(csv.DictReader(fh))
    OUT.mkdir(parents=True, exist_ok=True)
    with Run("op_parity", "b2d-prep-plan", config=vars(a)) as run:
        from jevdrive.data import splits
        run.use_split(splits.load("b2d/b2dc-v2-train")), run.use_split(splits.load("b2d/b2dc-v2-val"))
        with ProcessPoolExecutor(min(48, max(1, n_cpus() - 8))) as ex:
            info = list(run.tqdm(ex.map(_clip_job, idx, chunksize=2), total=len(idx), desc="clips"))
        info = [c for c in info if c["n_rows"] > 0]
        N = sum(c["n_rows"] for c in info)
        run.info(f"{len(info)} clips with rows, {N} rows; collision cuts: {sum(c['t_col'] >= 0 for c in info)}")
        cat = lambda k: np.concatenate([c[k] for c in info])  # noqa: E731
        route = np.concatenate([[c["route"]] * c["n_rows"] for c in info])
        tick = cat("t0").astype(np.int32)
        # tick store layout: clip after clip, even ticks 4 .. last row tick
        off, T = 0, 0
        fi = np.zeros((N, 8), np.int64)
        r0 = 0
        for c in info:
            t0 = c["t0"]
            c["tick_off"], c["tick_last"], c["row_off"] = T, int(t0.max()), r0
            slot = np.maximum(t0[:, None] - 4 * np.arange(7, -1, -1)[None], 4)
            fi[r0:r0 + len(t0)] = T + (slot - 4) // 2
            T += (int(t0.max()) - 4) // 2 + 1
            r0 += len(t0)
        np.save(OUT / "front_idx.npy", fi)
        tab = dict(names=np.array([f"{r}_{t:05d}" for r, t in zip(route, tick)]), log=route, ego=cat("ego"), pose=cat("hist"), vel=cat("vel"),
                   acc=cat("acc"), cmd=np.repeat(cat("cmd")[:, None], 4, 1), fut=cat("fut"),
                   cam=np.tile(np.asarray(CAM, np.float32), (N, 1)), lht=np.zeros(N, bool), speed=cat("speed"))
        assert np.isfinite(tab["fut"]).all() and np.isfinite(tab["ego"]).all(), "non-finite labels in the rows"
        np.savez(OUT / "tab.npz", **tab)
        clamp = (tick - 28) < 4
        np.savez(OUT / "extra.npz", route=route, tick=tick, clamped=clamp, type=np.concatenate([[c["type"]] * c["n_rows"] for c in info]),
                 town=np.concatenate([[c["town"]] * c["n_rows"] for c in info]), turn_side=np.concatenate([[c["turn"]] * c["n_rows"] for c in info]),
                 turn_next=cat("turn_next"), turn_dist=cat("turn_dist"), act_kappa=cat("act_kappa"), act_accel=cat("act_accel"), ctl=cat("ctl"),
                 target_speed=cat("target_speed"), progress=cat("progress"))
        np.savez(OUT / "hinge_labels.npz", tokens=tab["names"], ok=cat("sdf_ok"), sdf=np.concatenate([c["sdf"] for c in info]))
        # sharded work list: contiguous clip ranges of about equal row counts
        for c in info:
            for k in ("sdf", "sdf_ok", "ego", "hist", "vel", "acc", "cmd", "fut", "speed", "turn_next", "turn_dist", "act_kappa", "act_accel", "ctl",
                      "target_speed", "progress", "t0"):
                c.pop(k, None)
        (OUT / "plan.json").write_text(json.dumps(dict(version=VERSION, rows=N, ticks=T, clips=info, cut_s=CUT_S, cam=CAM), indent=0))
        np.lib.format.open_memmap(OUT / "ticks.npy", "w+", np.float16, (T, 32, 512)).flush()
        np.lib.format.open_memmap(OUT / "teacher_out.npy", "w+", np.float32, (N, N_OUT)).flush()
        np.lib.format.open_memmap(OUT / "teacher_plan.npy", "w+", np.float32, (N, 33, 15)).flush()
        (OUT / "tokens_done").mkdir(exist_ok=True)
        gb = lambda x: x / 2 ** 30  # noqa: E731
        col = [c for c in info if c["t_col"] >= 0]
        run.summary |= dict(rows=N, ticks=T, clips=len(info), collision_clips=len(col), collision_unlocated=sum(c["d_col"] > 3.0 for c in col),
                            ticks_gb=gb(T * 32 * 512 * 2), hinge_gb=gb(N * 128 * 96 * 2), clamped_rows=int(clamp.sum()),
                            rows_dropped_by_collision_rule=int(sum(c["n"] // 2 - c["n_rows"] for c in col)))
        run.info(json.dumps(run.summary))


# ---------------------------------------------------------------- tokens + teacher (one card)
def tokens(a):
    import torch
    import pp_prep as PP
    import b2dc_frames as F
    from jevdrive import op_adapt as A
    from jevdrive.run import Run
    pl = json.loads((OUT / "plan.json").read_text())
    clips = pl["clips"]
    rows = np.cumsum([0] + [c["n_rows"] for c in clips])
    mine = [c for i, c in enumerate(clips) if rows[i] * a.of // rows[-1] == a.shard and not (OUT / "tokens_done" / c["route"]).exists()]
    if a.limit:
        mine = mine[: a.limit]
    dev = torch.device("cuda")
    with Run("op_parity", f"b2d-prep-tokens-{a.shard}of{a.of}", config=vars(a)) as run:
        net, enc = PP.encoder(dev)
        di, pi = A.distill_index(net.slices), A.plan_index(net.slices)
        assert len(di) == N_OUT and len(pi) == N_PLAN
        tk = np.load(OUT / "ticks.npy", mmap_mode="r+")
        to, tp = np.load(OUT / "teacher_out.npy", mmap_mode="r+"), np.load(OUT / "teacher_plan.npy", mmap_mode="r+")
        fi = np.load(OUT / "front_idx.npy", mmap_mode="r")
        run.info(f"{len(mine)} clips on card {a.shard} of {a.of}")

        def load(c):
            T = c["tick_last"]
            return c, F.read_pairs(_pl.Path(c["clip"]) / "frames.mp4", frames=np.arange(0, T + 1, 2))

        t0, n_pairs = time.time(), 0
        tc = torch.tensor([[1.0, 0.0]], device=dev)
        with ThreadPoolExecutor(a.threads) as ex:
            futs = []
            it = iter(mine)
            for c in it:                                                   # bounded read-ahead
                futs.append(ex.submit(load, c))
                if len(futs) >= a.threads:
                    break
            done = 0
            while futs:
                c, pairs = futs.pop(0).result()
                for c2 in it:
                    futs.append(ex.submit(load, c2))
                    break
                m = len(pairs) - 2                                         # ticks 4, 6, .., T: pair (j - 2, j) of the even-tick list
                tok = enc(pairs[:-2], pairs[2:])
                assert tok.shape[0] == m == (c["tick_last"] - 4) // 2 + 1
                tk[c["tick_off"]:c["tick_off"] + m] = tok
                # teacher on the clip's rows (pp_prep.make_teacher on the same 8 slots)
                r0, r1 = c["row_off"], c["row_off"] + c["n_rows"]
                local = fi[r0:r1] - c["tick_off"]
                with torch.no_grad():
                    for i in range(0, r1 - r0, 256):
                        H = torch.from_numpy(tok[local[i:i + 256]]).to(dev)
                        H = torch.cat([torch.zeros_like(H[:, :1]), H], 1)
                        valid = torch.ones(H.shape[:2], dtype=torch.bool, device=dev)
                        valid[:, 0] = False
                        o = A._policy(net, H, (0.275, 0.525), tc.expand(len(H), 2), valid)["outputs"].float()
                        to[r0 + i:r0 + i + len(H)] = o[:, di].cpu().numpy()
                        tp[r0 + i:r0 + i + len(H)] = o[:, pi].cpu().numpy().reshape(-1, 33, 15)
                (OUT / "tokens_done" / c["route"]).write_text("")
                n_pairs += m
                done += 1
                if done % 10 == 0 or not futs:
                    run.info(f"{done}/{len(mine)} clips, {n_pairs / (time.time() - t0):.0f} pairs/s")
        tk.flush(), to.flush(), tp.flush()
        run.summary |= dict(clips=len(mine), pairs=n_pairs, pairs_per_s=n_pairs / max(1e-9, time.time() - t0), loop_s=time.time() - t0)


def finish(a):
    from jevdrive.run import Run
    pl = json.loads((OUT / "plan.json").read_text())
    missing = [c["route"] for c in pl["clips"] if not (OUT / "tokens_done" / c["route"]).exists()]
    assert not missing, f"{len(missing)} clips without tokens: {missing[:5]}"
    with Run("op_parity", "b2d-prep-finish", config=vars(a)) as run:
        import torch
        from jevdrive import op_adapt as A
        net = A.load("cinque", torch.float16)
        np.savez(OUT / "teacher.npz", out=np.load(OUT / "teacher_out.npy"), plan=np.load(OUT / "teacher_plan.npy"), di=A.distill_index(net.slices),
                 pi=A.plan_index(net.slices))
        tk = np.load(OUT / "ticks.npy", mmap_mode="r")
        bad = [i for i in range(0, len(tk), max(1, len(tk) // 2000)) if not np.abs(tk[i].astype(np.float32)).any()]
        assert not bad, f"all-zero token rows (unwritten): {bad[:5]}"
        (OUT / "meta.json").write_text(json.dumps(dict(version=VERSION, rows=pl["rows"], ticks=pl["ticks"], clips=len(pl["clips"]), cut_s=CUT_S,
                                                        done=time.strftime("%F %T")), indent=1))
        run.summary |= dict(rows=pl["rows"], ticks=pl["ticks"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "tokens", "finish"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8, help="clip decode read-ahead")
    ap.add_argument("--limit", type=int, default=0)
    from jevdrive.run import cli_args
    cli_args(ap)
    a = ap.parse_args()
    {"plan": plan, "tokens": tokens, "finish": finish}[a.cmd](a)
