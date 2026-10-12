"""junction_eyes slot pairs of AlpaSim decisions in the shared (N, 2, 2, 6, 128, 256) uint8 layout ([prev, cur] x [road, wide] x 6 x 128 x 256).

The t0 encoding pair of a decision is (slot 6, slot 7) of the driver's 8 policy slots (on_rows.make_front: prev of slot 7 = slot 6).

  held     held-452 dumps (road_probe/roll: `cur` stored) -> pairs straight from the dump
  recon    RECONSTRUCTED slot pairs: the onpol_long roll dumps had `cur` stripped (ol_chain.sh strip), only the t0 JPEG, ego, mu stay. The
           driver's own synthesis is rerun on the JPEGs of decisions k - 3 .. k (the 4 keyframes, 0.5 s apart) with the ego history taken
           from the dump's `ego` vector (poses exact, newest velocity exact, older body velocities by finite differences of the poses):
           sh30_core.pack_fast -> stack_keys -> lattice_gpu, the driver's default synth
  validate recon on held-452 dumps (cur and JPEG both stored): pixels against `cur`, tokens against the tokens of `cur`

Rows of recon / held: --rows FILE (npy of dump paths, in the order of the output).
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "experiments/alpasim/lib")]
import argparse, json, re  # noqa: E401,E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from jevdrive.common import data_dir  # noqa: E402

STEP_US = 500_000
_RUNS = {}


def run_index(rundir: Path):
    """drive.jsonl of one roll chunk -> {start: {n: cam}, drive: {(n, k): record}}."""
    rundir = Path(rundir)
    if rundir not in _RUNS:
        cam, sess, drive = {}, {}, {}
        for line in open(rundir / "driver-logs" / "drive.jsonl"):
            r = json.loads(line)
            if r["kind"] == "start":
                sess[r["session"]] = r["n"]
                cam[r["n"]] = {k: np.asarray(v, np.float32) for k, v in r["cam"].items()}
            elif r["kind"] == "drive":
                drive[(sess[r["session"]], r["k"])] = r
        _RUNS[rundir] = dict(cam=cam, drive=drive)
    return _RUNS[rundir]


def parse(dump: Path):
    m = re.match(r"s(\d+)_k(\d+)\.npz", Path(dump).name)
    return Path(dump).parents[2], int(m[1]), int(m[2])


def body_vel(P, vlast):
    """Body-frame velocities (4, 2) at the 4 keys: newest = the dump's, others finite differences of the poses (0.5 s)."""
    from jevdrive import op_interp as I
    t = 0.5
    W = np.zeros((4, 2))
    W[0] = (P[1, :2] - P[0, :2]) / t
    W[1:3] = (P[2:4, :2] - P[0:2, :2]) / (2 * t)
    V = np.zeros((4, 2))
    for j in range(3):
        V[j] = I.rot2(P[j, 2]).T @ W[j]
    V[3] = vlast
    return V


def recon_decision(dump: Path, packs: dict, cache: dict, dev, with_cur=False):
    """-> cur (8, 2, 6, 128, 256) uint8 np of the decision, valid (8,), n_keys, problems (list of str)."""
    import sh30_core as C
    from jevdrive import op_interp as I
    rundir, n, k = parse(dump)
    R = run_index(rundir)
    rec = R["drive"][(n, k)]
    d = np.load(dump)
    ego = d["ego"].astype(np.float64)
    nk = rec["n_keys"]
    bad = []
    keys = []
    for j in range(nk):
        kk = k - j
        r = R["drive"].get((n, kk))
        if r is None or abs(r["t0"] - (rec["t0"] - j * STEP_US)) > 2000:
            bad.append(f"key {j}: decision {kk} missing or off the 0.5 s grid")
            break
        key = (str(rundir), n, kk)
        if key not in cache:
            f = Path(rundir) / "driver-logs" / "dump" / f"s{n:02d}_k{kk}.npz"
            cache[key] = C.pack_fast(np.load(f)["jpeg"].tobytes(), R["cam"][n])
        keys.append(cache[key])
    if bad:
        return None, None, nk, bad
    keys = keys[::-1]
    P = ego[8:20].reshape(4, 3) * np.array([10.0, 10.0, 1.0])
    V = body_vel(P, ego[4:6] * 10.0)
    e = 4 - len(keys)
    V[:e] = V[e]
    K = C.stack_keys(keys)
    track = I.track_navsim(P, V)
    cur, valid = C.lattice_gpu(K, e, track, d["cam_t"].astype(np.float64), "backwarp", dev)
    return cur.cpu().numpy(), valid, nk, bad


def pair_of(cur):
    """(slot 6, slot 7) = the encoding pair of the t0 slot -> (2, 2, 6, 128, 256)."""
    return np.stack([cur[6], cur[7]])


def cmd_validate(a):
    import torch
    import sys
    sys.path.insert(0, str(_R / "experiments/op_parity/scripts"))
    import pp_prep as P
    from jevdrive import op_adapt as A
    from jevdrive.run import Run
    dev = torch.device("cuda")
    root = data_dir() / "runs/road_probe/roll"
    files = sorted(root.glob("c*/run/driver-logs/dump/*.npz"))
    rng = np.random.default_rng(0)
    sel = [files[i] for i in rng.choice(len(files), a.n, replace=False)]
    with Run("junction_eyes", "slots-validate", config=vars(a)) as run:
        net, enc = P.encoder(dev)
        cache, res, skipped = {}, [], 0
        for f in run.tqdm(sel, desc="decisions"):
            cur, valid, nk, bad = recon_decision(f, None, cache, dev)
            if cur is None:
                skipped += 1
                continue
            ref = np.load(f)["cur"]
            res.append((nk, ref, cur, valid))
        stats = {"n": len(res), "skipped_missing_keys": skipped}
        d = np.stack([np.abs(r[1][[5, 6, 7]].astype(np.int16) - r[2][[5, 6, 7]].astype(np.int16)) for r in res])
        for i, s in enumerate((5, 6, 7)):
            stats[f"slot{s}_pix_mean_abs"] = float(d[:, i].mean())
            stats[f"slot{s}_pix_max"] = int(d[:, i].max())
            stats[f"slot{s}_pix_frac_differ"] = float((d[:, i] > 0).mean())
            stats[f"slot{s}_pix_frac_gt2"] = float((d[:, i] > 2).mean())
        stats["frames_all_equal_frac"] = float(np.mean([(r[1][5:8] == r[2][5:8]).all() for r in res]))
        by = {}
        for nk, ref, cur, _ in res:
            by.setdefault(nk, []).append(np.abs(ref[7].astype(int) - cur[7].astype(int)).mean())
        stats["slot7_pix_mean_abs_by_n_keys"] = {int(k): [len(v), float(np.mean(v))] for k, v in sorted(by.items())}
        pr = np.stack([pair_of(r[1]) for r in res])
        pc = np.stack([pair_of(r[2]) for r in res])
        tr = enc(pr[:, 0], pr[:, 1]).astype(np.float32)
        tc = enc(pc[:, 0], pc[:, 1]).astype(np.float32)
        e = np.abs(tr - tc)
        stats["tok_mean_abs"], stats["tok_max_abs"], stats["tok_rms"] = float(e.mean()), float(e.max()), float(np.sqrt((tr ** 2).mean()))
        stats["tok_per_row_max_p50_p99"] = [float(np.percentile(e.max((1, 2)), q)) for q in (50, 99)]
        # fp16 noise floor: the same dump pair encoded twice in different batch sizes
        t1 = enc(pr[:64, 0], pr[:64, 1]).astype(np.float32)
        t2 = np.concatenate([enc(pr[i:min(i + 7, 64), 0], pr[i:min(i + 7, 64), 1]) for i in range(0, 64, 7)]).astype(np.float32)
        stats["tok_noise_floor_mean_abs"] = float(np.abs(t1 - t2).mean())
        stats["tok_noise_floor_max_abs"] = float(np.abs(t1 - t2).max())
        run.info(json.dumps(stats, indent=1))
        out = data_dir() / "runs/junction_eyes/slots"
        out.mkdir(parents=True, exist_ok=True)
        (out / "validate.json").write_text(json.dumps(stats, indent=1))


SLOTS = data_dir() / "runs/junction_eyes/slots"
ROWS = data_dir() / "runs/junction_eyes/step0/rows"
CACHE = data_dir() / "runs/op_parity/cache"
PAIR = (2, 2, 6, 128, 256)


def scene_index(rundir):
    """scene -> session n of one roll chunk (start records)."""
    key = ("scene", str(rundir))
    if key not in _RUNS:
        d = {}
        for line in open(Path(rundir) / "driver-logs" / "drive.jsonl"):
            r = json.loads(line)
            if r["kind"] == "start":
                d[r["scene"]] = r["n"]
        _RUNS[key] = d
    return _RUNS[key]


def cmd_held(a):
    """held-452: (slot 6, slot 7) straight from the dumps' `cur`, rows in held452.parquet order -> held452_pairs.npy; tokens against the
    cached tokens of road_probe/tok/held (their slot-7 token) on --check rows."""
    import pandas as pd
    import torch
    import sys
    sys.path.insert(0, str(_R / "experiments/op_parity/scripts"))
    import pp_prep as P
    from jevdrive.run import Run
    df = pd.read_parquet(ROWS / "held452.parquet").sort_values("rid").reset_index(drop=True)
    tab = np.load(data_dir() / "runs/road_probe/tok/held/tab.npz")
    with Run("junction_eyes", "slots-held", config=vars(a)) as run:
        SLOTS.mkdir(parents=True, exist_ok=True)
        out = np.lib.format.open_memmap(SLOTS / "held452_pairs.npy", "w+", np.uint8, (len(df),) + PAIR)
        for i, r in enumerate(run.tqdm(df.itertuples(), total=len(df))):
            rd = Path(tab["run"][r.tok_row])
            assert tab["scene"][r.tok_row] == r.scene and tab["k"][r.tok_row] == r.k
            f = rd / "driver-logs" / "dump" / f"s{scene_index(rd)[r.scene]:02d}_k{r.k}.npz"
            out[i] = pair_of(np.load(f)["cur"])
        out.flush()
        net, enc = P.encoder(torch.device("cuda"))
        idx = np.random.default_rng(0).choice(len(df), a.check, replace=False)
        ref = np.load(data_dir() / "runs/road_probe/tok/held/front.npy", mmap_mode="r")
        t = enc(out[np.sort(idx), 0], out[np.sort(idx), 1]).astype(np.float32)
        x = np.stack([ref[df.tok_row[i]][7] for i in np.sort(idx)]).astype(np.float32)
        st = dict(rows=len(df), check_rows=int(a.check), tok_mean_abs=float(np.abs(t - x).mean()), tok_max_abs=float(np.abs(t - x).max()), tok_rms=float(np.sqrt((x ** 2).mean())))
        run.info(json.dumps(st))
        (SLOTS / "held452_pairs.json").write_text(json.dumps(st, indent=1))


def cmd_jon(a):
    """J_on RECONSTRUCTED slot pairs for shard i of n of J_on.parquet (rid order) -> jon_pairs.s<i>of<n>.npy + ok mask."""
    import pandas as pd
    import torch
    from jevdrive.run import Run
    dev = torch.device("cuda")
    df = pd.read_parquet(ROWS / "J_on.parquet").sort_values("rid").reset_index(drop=True)
    sh = np.arange(len(df))[a.shard::a.of]
    with Run("junction_eyes", f"slots-jon-{a.shard}of{a.of}", config=vars(a)) as run:
        SLOTS.mkdir(parents=True, exist_ok=True)
        out = np.lib.format.open_memmap(SLOTS / f"jon_pairs.s{a.shard}of{a.of}.npy", "w+", np.uint8, (len(sh),) + PAIR)
        ok = np.zeros(len(sh), bool)
        key = [(parse(Path(df.dump[i]))[0].as_posix(), parse(Path(df.dump[i]))[1], parse(Path(df.dump[i]))[2]) for i in sh]
        order = sorted(range(len(sh)), key=lambda j: key[j])
        cache, cur_sess = {}, None
        for j in run.tqdm(order, total=len(order)):
            if key[j][:2] != cur_sess:
                cache.clear()
                cur_sess = key[j][:2]
            cur, valid, nk, bad = recon_decision(Path(df.dump[sh[j]]), None, cache, dev)
            if cur is None:
                continue
            out[j] = pair_of(cur)
            ok[j] = True
        out.flush()
        np.save(SLOTS / f"jon_pairs.s{a.shard}of{a.of}.ok.npy", ok)
        np.save(SLOTS / f"jon_pairs.s{a.shard}of{a.of}.idx.npy", sh)
        run.info(f"{int(ok.sum())} of {len(sh)} rows reconstructed")


def cmd_gdump(a):
    """G-dump: reconstructed (slot 6, slot 7) pairs of 500 J_on decisions through pp_prep.encoder against d275's
    on1_{L18Y,L18YS}@warp/front.npy slot-7 tokens -> step0/gdump.json."""
    import pandas as pd
    import torch
    import sys
    sys.path.insert(0, str(_R / "experiments/op_parity/scripts"))
    import pp_prep as P
    from jevdrive.run import Run
    df = pd.read_parquet(ROWS / "J_on.parquet").sort_values("rid").reset_index(drop=True)
    idx = np.sort(np.random.default_rng(0).choice(len(df), a.n, replace=False))
    with Run("junction_eyes", "slots-gdump", config=vars(a)) as run:
        net, enc = P.encoder(torch.device("cuda"))
        cache, pr, ref, rows = {}, [], [], []
        order = sorted(idx, key=lambda i: tuple(parse(Path(df.dump[i]))[0:1]) + parse(Path(df.dump[i]))[1:])
        cur_sess, fronts = None, {}
        for i in run.tqdm(order, total=len(order)):
            rd, n, k = parse(Path(df.dump[i]))
            if (rd, n) != cur_sess:
                cache.clear()
                cur_sess = (rd, n)
            cur, valid, nk, bad = recon_decision(Path(df.dump[i]), None, cache, torch.device("cuda"))
            if cur is None:
                continue
            f = df.tok_file[i]
            fronts.setdefault(f, np.load(CACHE / f / "front.npy", mmap_mode="r"))
            pr.append(pair_of(cur)), ref.append(np.asarray(fronts[f][df.tok_row[i]][7])), rows.append(int(df.rid[i]))
        pr, ref = np.stack(pr), np.stack(ref).astype(np.float32)
        t = enc(pr[:, 0], pr[:, 1]).astype(np.float32)
        e = np.abs(t - ref)
        v = json.loads((SLOTS / "validate.json").read_text())
        res = dict(wording="reconstructed slot pairs", rows=len(rows), requested=int(a.n), tok_mean_abs=float(e.mean()), tok_max_abs=float(e.max()),
                   tok_rms=float(np.sqrt((ref ** 2).mean())), per_row_max_p50=float(np.percentile(e.max((1, 2)), 50)),
                   per_row_max_p99=float(np.percentile(e.max((1, 2)), 99)), held452_validation=v,
                   rule="pass iff tok_mean_abs <= 2e-3 and per_row_max_p99 <= 0.05 (fp16 noise of the encoder on identical pixels is ~1e-4 mean)")
        res["pass"] = bool(res["tok_mean_abs"] <= 2e-3 and res["per_row_max_p99"] <= 0.05)
        run.info(json.dumps(res, indent=1))
        o = data_dir() / "runs/junction_eyes/step0"
        o.mkdir(parents=True, exist_ok=True)
        (o / "gdump.json").write_text(json.dumps(res, indent=1))


def cmd_zero(a):
    """zero-98: the stored driver messages replayed through the real driver class (as eyes_vs_head ev.py replay); the slot pair (6, 7) of the
    served decision is captured from Core.plan -> zero98_pairs.npy in zero98.parquet order (+ ok mask)."""
    import pickle
    import tempfile
    import pandas as pd
    import torch
    import sys
    sys.path[:0] = [str(_R / "experiments/eyes_vs_head/scripts")]
    import ev as EV
    import sh30_driver as D
    from jevdrive.run import Run
    df = pd.read_parquet(ROWS / "zero98.parquet").sort_values("rid").reset_index(drop=True)
    want = {(r.tag, r.scene, int(r.k)): i for i, r in enumerate(df.itertuples())}
    with Run("junction_eyes", "slots-zero", config=vars(a)) as run:
        SLOTS.mkdir(parents=True, exist_ok=True)
        out = np.lib.format.open_memmap(SLOTS / "zero98_pairs.npy", "w+", np.uint8, (len(df),) + PAIR)
        ok = np.zeros(len(df), bool)
        pb, ctx = D.egodriver_pb2, EV.Ctx()
        for tag in sorted({t for t, _, _ in want}):
            core = D.C.Core(tag, "cuda", "backwarp", 1.0, "gpu")
            drv = D.Driver(core, Path(tempfile.mkdtemp()), 0, False)
            last, orig = {}, core.plan

            def plan(*x, _o=orig, **k):
                last["o"] = _o(*x, **k)
                return last["o"]
            core.plan = plan
            for scene in sorted({sc for t, sc, _ in want if t == tag}):
                uuid = None
                for kind, raw in pickle.load(open(EV.OUT / "msgs" / tag / f"{scene}.pkl", "rb")):
                    if kind == "driver_session_request":
                        req = pb.DriveSessionRequest.FromString(raw)
                        uuid = req.session_uuid
                        drv.start_session(req, ctx)
                    elif kind == "driver_camera_image":
                        drv.submit_image_observation(pb.RolloutCameraImage.FromString(raw), ctx)
                    elif kind == "driver_ego_trajectory":
                        drv.submit_egomotion_observation(pb.RolloutEgoTrajectory.FromString(raw), ctx)
                    elif kind == "route_request":
                        drv.submit_route(pb.RouteRequest.FromString(raw), ctx)
                    elif kind == "driver_request":
                        try:
                            drv.drive(pb.DriveRequest.FromString(raw), ctx)
                        except RuntimeError:
                            continue
                        k = drv.sessions[uuid].count["drive"] - 1
                        i = want.get((tag, scene, k))
                        if i is not None:
                            out[i], ok[i] = pair_of(last["o"]["cur"]), True
                drv.sessions.pop(uuid, None)
            run.info(f"{tag}: {int(ok.sum())} of {len(df)} so far")
            del core, drv
            torch.cuda.empty_cache()
        out.flush()
        np.save(SLOTS / "zero98_pairs.ok.npy", ok)
        run.info(f"{int(ok.sum())} of {len(df)} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    from jevdrive.run import cli_args
    p = sp.add_parser("validate")
    p.add_argument("--n", type=int, default=600)
    cli_args(p)
    p = sp.add_parser("held")
    p.add_argument("--check", type=int, default=500)
    cli_args(p)
    p = sp.add_parser("jon")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--of", type=int, default=1)
    cli_args(p)
    cli_args(sp.add_parser("zero"))
    p = sp.add_parser("gdump")
    p.add_argument("--n", type=int, default=500)
    cli_args(p)
    a = ap.parse_args()
    {"validate": cmd_validate, "held": cmd_held, "jon": cmd_jon, "gdump": cmd_gdump, "zero": cmd_zero}[a.cmd](a)
