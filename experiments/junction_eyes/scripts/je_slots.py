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
        t2 = np.concatenate([enc(pr[i:i + 7, 0], pr[i:i + 7, 1]) for i in range(0, 64, 7)]).astype(np.float32)
        stats["tok_noise_floor_mean_abs"] = float(np.abs(t1 - t2).mean())
        stats["tok_noise_floor_max_abs"] = float(np.abs(t1 - t2).max())
        run.info(json.dumps(stats, indent=1))
        out = data_dir() / "runs/junction_eyes/slots"
        out.mkdir(parents=True, exist_ok=True)
        (out / "validate.json").write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    from jevdrive.run import cli_args
    p = sp.add_parser("validate")
    p.add_argument("--n", type=int, default=600)
    cli_args(p)
    a = ap.parse_args()
    {"validate": cmd_validate}[a.cmd](a)
