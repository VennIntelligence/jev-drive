"""P3 vehicle deletion pairs (fc65452:todos/2026-09-26-night-queue-4.md, P section, amendment 2; user approval 2026-09-28: every
segment with a valid vehicle event under the registered rules, no count target).

  select   experiments/night_queue_4/results/p3/filter/vehicle_events.csv (valid events) -> runs/nq4/p3veh/{scenes.json, targets/<j>.json}:
           segments in crc32 order, scene j = the j-th; per segment the crc32-minimal valid event is the target
           (f0 = its first conflict frame, delete_tracks = [the event vehicle], node_type RigidNodes)
  prep     (CPU / network) download the scene-flow tfrecord, drivestudio prep into processed/waymo_ds_veh/training/<j>,
           the same artifact check as the pedestrian pool, then drop the raw file; segments already prepped in the
           pedestrian pool are linked, not downloaded again. Several downloads and one prep process per core.
  scene    (one card) sky masks -> OmniRe train (the registered pedestrian config, except that every vehicle is a rigid
           node: RigidNodes.init.only_moving = false, so a stopped vehicle can be deleted) -> render real / x+ / x- at
           the target. Markers runs/nq4/p3veh/{sky,train,render}_<j>.{done,rc}; pid file scene_<j>.pid.

  python3 experiments/p3_ped_exam/archive/veh.py select
  scripts/tmux_run.sh p3-veh-prep python3 experiments/p3_ped_exam/archive/veh.py prep --cpus 178,179,180,181,182,183
  python3 experiments/p3_ped_exam/archive/veh.py scene --j 0 --gpu 6 --cpus 172,173          (normally started by experiments/p3_ped_exam/archive/expand.py)
"""
import argparse
import csv
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATA = Path(os.environ["DATA_DIR"])
REPO = Path(__file__).resolve().parents[3]
V = DATA / "runs/nq4/p3veh"
PROC = DATA / "processed/waymo_ds_veh/training"
PED_PROC = DATA / "processed/waymo_ds/training"
RAW = DATA / "datasets/waymo_perception/sceneflow"
GCLOUD = DATA / "tools/google-cloud-sdk/bin/gcloud"
PY_DS, PY_PREP = DATA / "envs/drivestudio/bin/python", DATA / "envs/p3-wodprep/bin/python"
THREADS1 = {k: "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                             "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")}
_lock = threading.Lock()


def log(msg, **ev):
    with _lock:
        line = f"{time.strftime('%F %T')} {msg}"
        print(line, flush=True)
        with open(V / "log.txt", "a") as f:
            f.write(line + "\n")
        with open(V / "events.jsonl", "a") as f:
            f.write(json.dumps({"t": time.time(), "msg": msg, **ev}) + "\n")


def crc(s: str) -> int:
    return zlib.crc32(s.encode())


def select(_):
    V.mkdir(parents=True, exist_ok=True)
    ev = [r for r in csv.DictReader(open(REPO / "experiments/night_queue_4/results/p3/filter/vehicle_events.csv")) if r["valid"] == "True"]
    sel = {r["name"]: r for r in csv.DictReader(open(REPO / "experiments/night_queue_4/results/p3/selection_wod.csv"))}
    segs = sorted({r["seg"] for r in ev}, key=crc)
    (V / "targets").mkdir(exist_ok=True)
    for j, s in enumerate(segs):
        e = min((r for r in ev if r["seg"] == s), key=lambda r: crc(f"{s}/{r['track']}/{r['f0']}"))
        t = {"scene": j, "key": f"p3_v{j:03d}", "segment": s, "split": sel[s]["split"], "f0": int(e["f0"]), "t0": float(e["t0"]),
             "v0": float(e["v0"]), "target_track": e["track"], "delete_tracks": [e["track"]], "node_type": "RigidNodes",
             "cat": e["cat"], "ttc": float(e["ttc"]), "n_frames": int(sel[s]["n_frames"])}
        (V / "targets" / f"{j:03d}.json").write_text(json.dumps(t, indent=1))
    (V / "scenes.json").write_text(json.dumps({"segments": segs, "gcs": [sel[s]["gcs"] for s in segs]}, indent=1))
    print(json.dumps({"segments": len(segs), "events": len(ev)}))


def complete(d: Path, n: int) -> bool:
    """The pedestrian pool's artifact check (experiments/p3_ped_exam/archive/pool.py complete), on one processed scene dir."""
    files = [d / "frame_info.json", d / "instances/instances_info.json", d / "instances/frame_instances.json"]
    for c in range(5):
        files += [d / "extrinsics" / f"{c}.txt", d / "intrinsics" / f"{c}.txt"]
    for f in range(n):
        files += [d / "ego_pose" / f"{f:03d}.txt", d / "lidar" / f"{f:03d}.bin"]
        for c in range(5):
            files.append(d / "images" / f"{f:03d}_{c}.jpg")
            files += [d / "dynamic_masks" / kind / f"{f:03d}_{c}.png" for kind in ("all", "human", "vehicle")]
    if not all(p.is_file() and p.stat().st_size > 0 for p in files):
        return False
    return len(json.loads(files[2].read_text())) == n


def prep(a):
    V.mkdir(parents=True, exist_ok=True)
    PROC.mkdir(parents=True, exist_ok=True)
    sc = json.loads((V / "scenes.json").read_text())
    ped = json.loads((DATA / "runs/nq4/p3/pool/scenes66.json").read_text())["segments"]
    tg = [json.loads((V / "targets" / f"{j:03d}.json").read_text()) for j in range(len(sc["segments"]))]
    env_gc = dict(os.environ, http_proxy="http://127.0.0.1:7890", https_proxy="http://127.0.0.1:7890",
                  CLOUDSDK_PYTHON=str(DATA / "tools/google-cloud-sdk/platform/bundledpythonunix/bin/python3"))
    env_prep = dict(os.environ, CUDA_VISIBLE_DEVICES="", **THREADS1)
    cpus = a.cpus.split(",")
    free = queue.Queue()
    for c in cpus:
        free.put(c)
    failed = []

    def one(j):
        name, n, d = sc["segments"][j], tg[j]["n_frames"], PROC / f"{j:03d}"
        if (V / f"prep_{j:03d}.done").exists():
            return
        if name in ped and not d.exists():                      # already prepped in the pedestrian pool: link it
            k = ped.index(name)
            if complete(PED_PROC / f"{k:03d}", n):
                d.symlink_to(PED_PROC / f"{k:03d}")
                (V / f"prep_{j:03d}.done").write_text(f"{time.strftime('%F %T')} linked pedestrian pool {k:03d}\n")
                log(f"prep {j:03d}: linked pedestrian pool scene {k:03d}", event="prep_link", j=j, k=k)
                return
        raw = RAW / f"segment-{name}_with_camera_labels.tfrecord"
        if not complete(d, n):
            if not raw.exists():
                t0 = time.time()
                part = raw.with_suffix(".part")
                r = subprocess.run([str(GCLOUD), "storage", "cp", sc["gcs"][j], str(part)], env=env_gc, capture_output=True,
                                   text=True, timeout=3600)
                if r.returncode:
                    part.unlink(missing_ok=True)
                    raise RuntimeError(f"download: {r.stderr[-300:]}")
                part.rename(raw)
                log(f"download {j:03d} {raw.stat().st_size / 1e9:.2f} GB in {time.time() - t0:.0f} s", event="download", j=j)
            cpu = free.get()                                    # one prep process per core
            try:
                t0 = time.time()
                with open(V / f"prep_{j:03d}.out", "a") as out:
                    rc = subprocess.call(["taskset", "-c", cpu, str(PY_PREP), "experiments/p3_ped_exam/archive/ds.py", "prep", "--raw", str(RAW), "--out",
                                          str(PROC.parent), "--scenes", str(V / "scenes.json"), "--workers", "1", "--ids", str(j)],
                                         cwd=REPO, env=env_prep, stdout=out, stderr=subprocess.STDOUT, timeout=7200)
            finally:
                free.put(cpu)
            if rc or not complete(d, n):
                raise RuntimeError(f"prep rc={rc}, complete={complete(d, n)}")
            log(f"prep {j:03d} done in {time.time() - t0:.0f} s on cpu {cpu}", event="prep", j=j)
        (V / f"prep_{j:03d}.done").write_text(time.strftime("%F %T") + "\n")
        raw.unlink(missing_ok=True)

    def guarded(j):
        try:
            one(j)
        except Exception as e:                                     # noqa: BLE001
            failed.append(j)
            log(f"FAILED {j:03d}: {e}", event="error", j=j, err=str(e))

    todo = [j for j in range(len(sc["segments"])) if not (V / f"prep_{j:03d}.done").exists()]
    log(f"prep: {len(todo)} segments, cpus {cpus}", event="start", todo=todo)
    # one worker per core plus the downloads in flight; the disk floor stops admission
    with ThreadPoolExecutor(len(cpus) + a.downloads) as ex:
        futs = []
        for j in todo:
            if shutil.disk_usage(DATA).free / 1e9 < a.min_free_gb:
                log(f"STOP at {j:03d}: disk floor", event="stop", j=j)
                break
            futs.append(ex.submit(guarded, j))
        for f in futs:
            f.result()
    ok = sorted(j for j in range(len(sc["segments"])) if (V / f"prep_{j:03d}.done").exists())
    (V / ("PREP_DONE" if not failed and len(ok) == len(sc["segments"]) else "PREP_ERROR")).write_text(
        json.dumps({"ready": len(ok), "failed": failed}) + "\n")
    log(f"prep end: {len(ok)} ready, failed {failed}", event="end")


def scene(a):
    """sky -> train -> render of vehicle scene j on one card (steps with a .done marker are skipped)."""
    j = a.j
    jj = f"{j:03d}"
    (V / f"scene_{jj}.pid").write_text(str(os.getpid()))
    d = PROC / jj
    tg = V / "targets" / f"{jj}.json"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", HF_HUB_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false", **THREADS1)
    out_root, run = DATA / "ckpt/nq4_p3_veh", DATA / "ckpt/nq4_p3_veh/p3" / jj
    steps = [("sky", [str(PY_DS), "experiments/p3_ped_exam/archive/ds.py", "sky", str(d)], 3600),
             ("train", [str(PY_DS), "experiments/p3_ped_exam/archive/ds.py", "train", "--scene", str(j), "--data-root", str(PROC), "--out-root", str(out_root),
                        "--rigid-all"], a.train_timeout),
             ("render", [str(PY_DS), "experiments/p3_ped_exam/archive/ds.py", "render", "--run", str(run), "--target", str(tg), "--out",
                         str(DATA / "processed/nq4_p3_veh/scenes" / f"p3_v{jj}")], 1800)]
    for name, argv, tmo in steps:
        if (V / f"{name}_{jj}.done").exists():
            continue
        with open(V / f"{name}_{jj}.out", "a") as out:
            rc = subprocess.call(["timeout", str(tmo), "taskset", "-c", a.cpus, *argv], cwd=REPO, env=env, stdout=out, stderr=subprocess.STDOUT)
        (V / f"{name}_{jj}.rc").write_text(str(rc))
        if rc:
            sys.exit(f"{name} {jj} rc={rc}")
        (V / f"{name}_{jj}.done").write_text(time.strftime("%F %T") + "\n")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("select")
    p = sp.add_parser("prep")
    p.add_argument("--cpus", required=True)
    p.add_argument("--downloads", type=int, default=3)
    p.add_argument("--min-free-gb", type=float, default=150)
    p = sp.add_parser("scene")
    p.add_argument("--j", type=int, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--cpus", required=True)
    p.add_argument("--train-timeout", type=int, default=6 * 3600)
    a = ap.parse_args()
    {"select": select, "prep": prep, "scene": scene}[a.cmd](a)


if __name__ == "__main__":
    main()
