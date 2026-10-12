"""junction_eyes keep set (comma1M, native 1.22 m camera, no geometry labels): model frames of turn / straight event windows,
the held-out segments, and the factory-Cinque teacher cache. Layout and numbers: experiments/junction_eyes/results/keep_set.md.

  census                 camera type / ecamera / length of every segment dir -> $RUN/census.json
  select                 op_fov scan events + census -> held-out segments, kept windows, rows -> $RUN/selection.json
  register               (Mac) selection.json -> splits comma1m/junction-eyes-{keep,heldout}
  frames                 stream-decode + warp every kept window (native focals) -> $RUN/frames.npy (F, 2, 6, 128, 256) uint8
  pairs                  the rows' t0 image pairs in the shared layout -> $RUN/t0.npy (N, 2, 2, 6, 128, 256) uint8
  teacher --shard i --of n   factory Cinque stepped at 20 Hz over its share of the windows (op_adapt.Stepper) -> $RUN/teacher/
RUN = $DATA_DIR/runs/junction_eyes/keep.

Only 126 of the 600 segment dirs have both cameras at 1928x1208 (the wide camera is a model input): the rest cannot make a model frame
pair and are not used. Hold-out: 60 of the 126 (12 turn-bearing + 48 others, numpy default_rng(0)), never read by a training job.
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R)]
import argparse, json, time  # noqa: E401,E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from jevdrive.common import data_dir  # noqa: E402

ROOT = data_dir() / "datasets" / "comma1M"
RUN = data_dir() / "runs" / "junction_eyes" / "keep"
EVENTS = _R / "experiments/op_fov/results/events.json"       # fov_replay.py scan, frozen
SEL = _R / "experiments/junction_eyes/results/keep_selection.json"
SEED, N_HOLD, N_HOLD_TURN = 0, 60, 12
WARM, PRE, POST = 120, 80, 200                              # fov_replay's: 6 s Stepper warm-up, rows from onset - 4 s to onset + 10 s
BACK = 4                                                    # the t0 pair is (frame - 4, frame): 0.2 s apart
ACTION_T = (0.275, 0.525)                                   # as navsim_zs_openpilot / fov_replay
FRAME = (2, 6, 128, 256)


def segs():
    return sorted(p.name for p in ROOT.iterdir() if p.is_dir() and (p / "frame_info.safetensors").exists())


def _census1(s):
    from jevdrive.openpilot.frames import load_safetensors
    fi = load_safetensors(ROOT / s / "frame_info.safetensors")
    return dict(seg=s, wh=[int(fi["fcamera/width"][0]), int(fi["fcamera/height"][0])], frames=int(fi["fcamera/frame_count"][0]),
                ecam=bool("ecamera/width" in fi and int(fi["ecamera/frame_count"][0]) > 0 and (ROOT / s / "ecamera.hevc").exists()),
                device=int(fi["device_type"]))


def cmd_census(a):
    from collections import Counter
    from jevdrive import par
    from jevdrive.run import Run
    with Run("junction_eyes", "keep-census", config=vars(a)) as run:
        res = par.pmap(_census1, segs(), run=run, desc="segments")
        res.raise_if_failed()
        RUN.mkdir(parents=True, exist_ok=True)
        (RUN / "census.json").write_text(json.dumps(res.values))
        run.info("%s", Counter((tuple(v["wh"]), v["ecam"], v["device"]) for v in res.values))


# ---------------------------------------------------------------- select
def cmd_select(a):
    cen = {c["seg"]: c for c in json.loads((RUN / "census.json").read_text())}
    usable = sorted(s for s, c in cen.items() if c["wh"] == [1928, 1208] and c["ecam"])
    ev = [e for e in json.loads(EVENTS.read_text()) if e["seg"] in cen]
    turn_segs = sorted({e["seg"] for e in ev if e["kind"] == "turn"})
    rng = np.random.default_rng(SEED)
    h_turn = sorted(rng.choice(turn_segs, N_HOLD_TURN, replace=False).tolist())
    rest = [s for s in usable if s not in turn_segs]
    h_other = sorted(rng.choice(rest, N_HOLD - N_HOLD_TURN, replace=False).tolist())
    held = sorted(h_turn + h_other)
    keep = [s for s in usable if s not in held]
    kev = [e for e in ev if e["seg"] in keep]
    turns = sorted((e for e in kev if e["kind"] == "turn"), key=lambda e: (e["seg"], e["start"]))
    stra = sorted((e for e in kev if e["kind"] == "straight"), key=lambda e: (e["seg"], e["start"]))
    nrows = lambda e: e["hi"] - (e["start"] + WARM)  # noqa: E731
    n_turn = sum(map(nrows, turns))
    pick, tot = [], 0
    for i in rng.permutation(len(stra)):                    # straight windows until the rows match the turn rows (half and half)
        if tot >= n_turn:
            break
        pick.append(stra[i])
        tot += nrows(stra[i])
    wins = sorted(turns + pick, key=lambda e: (e["seg"], e["start"]))
    off, row0 = 0, 0
    for w in wins:
        w.update(off=off, n=w["hi"] - w["start"], row0=row0, nrows=nrows(w))
        off, row0 = off + w["n"], row0 + w["nrows"]
    out = dict(seed=SEED, n_usable=len(usable), usable=usable, held=held, held_turn_bearing=h_turn, keep=keep,
               windows=[{k: w[k] for k in ("kind", "seg", "start", "onset", "end", "hi", "off", "n", "row0", "nrows")} for w in wins],
               frames=off, rows=row0, rows_turn=n_turn, rows_straight=tot, turn_events_all=len([e for e in ev if e["kind"] == "turn"]),
               turn_events_kept=len(turns), straight_events_kept_all=len(stra), straight_windows_used=len(pick))
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "selection.json").write_text(json.dumps(out, indent=0))
    print({k: v for k, v in out.items() if not isinstance(v, list)})


def cmd_register(a):
    from jevdrive.data import splits
    s = json.loads(SEL.read_text())
    org = ("experiments/junction_eyes/scripts/je_keep.py select: comma1M segments with both 1928x1208 cameras (126 of 600 dirs); 60 held out "
           "(12 turn-bearing + 48 others, numpy default_rng(0)); events = op_fov scan (experiments/op_fov/results/events.json)")
    for name, mem, note in (("junction-eyes-keep", s["keep"], "keep-set source segments (66)"),
                            ("junction-eyes-heldout", s["held"], "E5 held-out segments: never read by any junction_eyes training job")):
        r = splits.define("comma1m", name, mem, "segment", org, used_by=["junction_eyes"], status="frozen", notes=note)
        print(r.id)
    splits.check_disjoint(splits.load("comma1m/junction-eyes-keep"), splits.load("comma1m/junction-eyes-heldout"))


# ---------------------------------------------------------------- frames
def _seg_job(job):
    """One segment: decode both cameras once, warp, write each kept window of it into the shared memmap at its offset."""
    from jevdrive.openpilot.frames import load_segment_meta, segment_model_frames
    seg, wins = job
    hi = max(w["hi"] for w in wins)
    fr, _ = segment_model_frames(ROOT / seg, load_segment_meta(ROOT / seg), max_frames=hi)
    assert len(fr) >= hi, f"{seg}: decoded {len(fr)} frames, window needs {hi}"
    out = np.load(RUN / "frames.npy", mmap_mode="r+")
    for w in wins:
        out[w["off"]:w["off"] + w["n"]] = fr[w["start"]:w["hi"]]
    out.flush()
    return seg


def cmd_frames(a):
    import os
    import shutil
    from jevdrive import par
    from jevdrive.run import Run
    s = json.loads((RUN / "selection.json").read_text())
    with Run("junction_eyes", "keep-frames", config=vars(a)) as run:
        need, free = s["frames"] * int(np.prod(FRAME)) / 2 ** 30, shutil.disk_usage(RUN).free / 2 ** 30
        run.info(f"{s['frames']} frames, {need:.1f} GB, {free:.0f} GB free")
        assert free - need > 300, "disk floor 300 GB"
        if not (RUN / "frames.npy").exists():
            np.lib.format.open_memmap(RUN / "frames.npy", "w+", np.uint8, (s["frames"],) + FRAME).flush()
        by = {}
        for w in s["windows"]:
            by.setdefault(w["seg"], []).append(w)
        res = par.pmap(_seg_job, list(by.items()), workers=a.workers or max(1, par.cpus() // 2), run=run, desc="segments")
        res.raise_if_failed()
        run.info("%s", res.summary())


def rows_of(s):
    """(win, g, vframe) per row: window index, row of frames.npy, frame index in the segment video."""
    win, g, vf = [], [], []
    for i, w in enumerate(s["windows"]):
        f = np.arange(w["start"] + WARM, w["hi"])
        win.append(np.full(len(f), i)), g.append(w["off"] + f - w["start"]), vf.append(f)
    return np.concatenate(win), np.concatenate(g), np.concatenate(vf)


def cmd_pairs(a):
    """rows.npz in the layout of experiments/junction_eyes/lib/je_data.KeepStore (frames.npy is its frames file: every 20 Hz frame of the
    windows, so any row's 5 Hz context f - 4k is stored): fidx (N, 10) frames [t0 - 1.8 s .. t0] 0.2 s apart, teacher (N, n_out) = the
    factory raw outputs at t0, seg, held (all False: the held-out segments are not in this store), action_t. Extras: win, vframe, turn,
    temporal, vision (the teacher's select_4 / mean taps)."""
    from jevdrive.run import Run
    s = json.loads((RUN / "selection.json").read_text())
    win, g, vf = rows_of(s)
    with Run("junction_eyes", "keep-rows", config=vars(a)) as run:
        T = {k: np.load(RUN / f"teacher_{k}.npy", mmap_mode="r") for k in ("raw", "temporal", "vision")}
        fidx = g[:, None] - BACK * np.arange(9, -1, -1)[None]
        assert (fidx >= 0).all()
        turn = np.array([s["windows"][i]["kind"] == "turn" for i in win])
        seg = np.array([s["windows"][i]["seg"] for i in win])
        np.savez(RUN / "rows.npz", fidx=fidx.astype(np.int64), teacher=np.asarray(T["raw"][g]), seg=seg, held=np.zeros(len(g), bool),
                 action_t=np.array(ACTION_T), win=win, vframe=vf, turn=turn, temporal=np.asarray(T["temporal"][g]), vision=np.asarray(T["vision"][g]))
        run.info(f"{len(g)} rows ({int(turn.sum())} turn)")


# ---------------------------------------------------------------- teacher
def cmd_teacher(a):
    import torch
    from jevdrive import op_adapt as A
    from jevdrive.run import Run
    s = json.loads((RUN / "selection.json").read_text())
    dev = torch.device("cuda")
    with Run("junction_eyes", f"keep-teacher-{a.shard}of{a.of}", config=vars(a)) as run:
        net = A.load("cinque").to(dev)
        st = A.Stepper(net, "cinque")
        F = np.load(RUN / "frames.npy", mmap_mode="r")
        td = RUN / "teacher"
        td.mkdir(exist_ok=True)
        mine = [i for i in range(len(s["windows"])) if i % a.of == a.shard]
        t0 = time.time()
        for k, i in enumerate(mine):
            w = s["windows"][i]
            f = td / f"win{i:04d}.npz"
            if f.exists():
                continue
            st.reset()
            raw, tmp, vis = [], [], []
            for g in range(w["off"], w["off"] + w["n"]):                           # every 20 Hz frame, the real history
                r, t, v = st.step(np.asarray(F[g]), ACTION_T)
                raw.append(r), tmp.append(t), vis.append(v)
            np.savez(f.with_suffix(".tmp.npz"), raw=np.stack(raw).astype(np.float32), temporal=np.stack(tmp).astype(np.float32),
                     vision=np.stack(vis).astype(np.float32))
            f.with_suffix(".tmp.npz").rename(f)
            run.info(f"window {i} ({k + 1}/{len(mine)}): {w['n']} steps, {(time.time() - t0) / (k + 1):.1f} s/window")
        if a.shard == 0:
            import pickle
            (td / "slices.pkl").write_bytes(pickle.dumps(net.slices))


def cmd_teacher_merge(a):
    """The per-window files -> raw.npy (F, D) f32 / temporal.npy (F, 512) / vision.npy (F, 512), aligned with frames.npy, + slices.json."""
    s = json.loads((RUN / "selection.json").read_text())
    td = RUN / "teacher"
    z0 = np.load(td / "win0000.npz")
    D = z0["raw"].shape[1]
    out = {k: np.lib.format.open_memmap(RUN / f"teacher_{k}.npy", "w+", np.float32, (s["frames"], z0[k].shape[1])) for k in ("raw", "temporal", "vision")}
    for i, w in enumerate(s["windows"]):
        z = np.load(td / f"win{i:04d}.npz")
        for k in out:
            out[k][w["off"]:w["off"] + w["n"]] = z[k]
    for o in out.values():
        o.flush()
    import pickle
    sl = pickle.loads((td / "slices.pkl").read_bytes())
    (RUN / "teacher_slices.json").write_text(json.dumps({k: [v.start, v.stop] for k, v in sl.items()}))
    print("merged", {k: o.shape for k, o in out.items()}, "D", D)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    from jevdrive.run import cli_args
    for c in ("census", "select", "register", "pairs", "merge"):
        cli_args(sp.add_parser(c))
    p = sp.add_parser("frames")
    p.add_argument("--workers", type=int, default=0)
    cli_args(p)
    p = sp.add_parser("teacher")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--of", type=int, default=1)
    cli_args(p)
    a = ap.parse_args()
    {"census": cmd_census, "select": cmd_select, "register": cmd_register, "frames": cmd_frames, "pairs": cmd_pairs,
     "teacher": cmd_teacher, "merge": cmd_teacher_merge}[a.cmd](a)
