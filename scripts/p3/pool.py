"""P3 pool prep for a later >= 60-scene expansion (the expansion's GPU reconstructions need a separate OK).

Scene k = the k-th qualifying daytime segment of research/results/nq4/p3/selection_wod.csv in the registered crc32
order, so 000-009 are the gate scenes (runs/nq4/p3/scenes.json, asserted equal) and 010-065 the pool. Per pool
segment, in order: download the scene-flow tfrecord (gcloud through Clash; gcloud checks the object hash), drivestudio
prep with the same converter (scripts/p3/ds.py prep, one process per segment, CPU only), the same artifact check as the
scene 0-9 prep resume, then drop the raw tfrecord (public, re-downloadable; the processed dir is what training reads).
Sky masks (ds.py sky, same SegFormer revision and code path as scenes 0-9) run on --sky-gpu, a card P3 already holds.
Admission stops before a segment would leave less than --min-free-gb free on the data disk (after --reserve-gb and
the segments still in flight). Also writes pool/scenes66.json and pool/targets/<k>.json (the render targets).

  scripts/tmux_run.sh p3-pool python3 scripts/p3/pool.py --cpus 198,199,202,203,206,207 --sky-gpu 6
Outputs: $DATA_DIR/runs/nq4/p3/pool/{log.txt, events.jsonl, STATUS.md, prep_<k>.done}, runs/nq4/p3/sky_<k>.done.
"""
import argparse
import csv
import json
import os
import queue
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATA = Path(os.environ["DATA_DIR"])
REPO = Path(__file__).resolve().parents[2]
D = DATA / "runs/nq4/p3"
P = D / "pool"
RAW = DATA / "datasets/waymo_perception/sceneflow"
PROC = DATA / "processed/waymo_ds/training"
GCLOUD = DATA / "tools/google-cloud-sdk/bin/gcloud"
PROC_GB, RAW_GB = 2.6, 1.2          # measured scene 000: processed 2.2 GB (+ sky), raw ~1.0 GB
THREADS1 = {k: "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                             "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")}
_lock = threading.Lock()


def log(msg, **ev):
    with _lock:
        line = f"{time.strftime('%F %T')} {msg}"
        print(line, flush=True)
        with open(P / "log.txt", "a") as f:
            f.write(line + "\n")
        if ev:
            with open(P / "events.jsonl", "a") as f:
                f.write(json.dumps({"t": time.time(), **ev}) + "\n")


def complete(k: int, n: int) -> bool:
    """The scene 0-9 prep-resume artifact check (runs/nq4/p3/prep.resume.py), unchanged."""
    s = PROC / f"{k:03d}"
    files = [s / "frame_info.json", s / "instances/instances_info.json", s / "instances/frame_instances.json"]
    for c in range(5):
        files += [s / "extrinsics" / f"{c}.txt", s / "intrinsics" / f"{c}.txt"]
    for f in range(n):
        files += [s / "ego_pose" / f"{f:03d}.txt", s / "lidar" / f"{f:03d}.bin"]
        for c in range(5):
            files.append(s / "images" / f"{f:03d}_{c}.jpg")
            files += [s / "dynamic_masks" / kind / f"{f:03d}_{c}.png" for kind in ("all", "human", "vehicle")]
    if not all(p.is_file() and p.stat().st_size > 0 for p in files):
        return False
    for p in files[:3]:
        json.loads(p.read_text())
    return len(json.loads(files[2].read_text())) == n


def sky_ok(k: int) -> bool:
    s = PROC / f"{k:03d}"
    imgs = [f for f in (s / "images").glob("*.jpg") if int(f.stem.split("_")[1]) in (0, 1, 2)]
    return bool(imgs) and all((s / "sky_masks" / f"{f.stem}.png").is_file() for f in imgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpus", required=True, help="comma list; one prep process per core")
    ap.add_argument("--sky-gpu", type=int, required=True)
    ap.add_argument("--min-free-gb", type=float, default=150)
    ap.add_argument("--reserve-gb", type=float, default=15, help="kept for the running P3 scenes' checkpoints")
    ap.add_argument("--first", type=int, default=10)
    ap.add_argument("--last", type=int, default=65)
    a = ap.parse_args()
    P.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = open(P / "owner.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (P / "pid").write_text(str(os.getpid()))
    rows = [r for r in csv.DictReader(open(REPO / "research/results/nq4/p3/selection_wod.csv")) if r["qualifies"] == "True"]
    assert len(rows) == 66 and all(r["day"] == "True" for r in rows)
    gate = json.loads((D / "scenes.json").read_text())["segments"]
    assert [r["name"] for r in rows[:10]] == gate, "pool order must extend the registered gate scenes"
    (P / "scenes66.json").write_text(json.dumps({"segments": [r["name"] for r in rows]}, indent=1))
    (P / "targets").mkdir(exist_ok=True)
    for k, r in enumerate(rows):   # same schema as jevdrive.nq4_p3.scenes()
        (P / "targets" / f"{k:03d}.json").write_text(json.dumps(
            {"scene": k, "key": f"p3_{k:03d}", "segment": r["name"], "split": r["split"], "f0": int(float(r["f0"])),
             "t0": float(r["t0"]), "v0": float(r["v0"]), "target_track": r["target_track"],
             "delete_tracks": json.loads(r["delete_tracks"])}, indent=1))
    cpus = a.cpus.split(",")
    env_gc = dict(os.environ, http_proxy="http://127.0.0.1:7890", https_proxy="http://127.0.0.1:7890",
                  CLOUDSDK_PYTHON=str(DATA / "tools/google-cloud-sdk/platform/bundledpythonunix/bin/python3"))
    env_prep = dict(os.environ, CUDA_VISIBLE_DEVICES="", **THREADS1)
    env_sky = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.sky_gpu), HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                   **THREADS1)
    todo = [k for k in range(a.first, a.last + 1)
            if not ((P / f"prep_{k:03d}.done").exists() and (D / f"sky_{k:03d}.done").exists())]
    log(f"pool prep: {len(todo)} segments to do ({todo[:3]}...), cpus {cpus}, sky GPU {a.sky_gpu}",
        event="start", todo=todo, cpus=cpus)
    inflight, sky_q, free_cpus = set(), queue.Queue(), queue.Queue()
    for c in cpus:
        free_cpus.put(c)
    stats = {"downloaded": 0, "prepped": 0, "sky": 0, "failed": []}

    def free_gb():
        return shutil.disk_usage(DATA).free / 1e9

    def prep_one(k):
        name, n = rows[k]["name"], int(rows[k]["n_frames"])
        raw = RAW / f"segment-{name}_with_camera_labels.tfrecord"
        cpu = free_cpus.get()
        try:
            if not complete(k, n):
                t0 = time.time()
                with open(P / f"prep_{k:03d}.out", "a") as out:
                    rc = subprocess.call(["taskset", "-c", cpu, str(DATA / "envs/p3-wodprep/bin/python"), "scripts/p3/ds.py",
                                          "prep", "--raw", str(RAW), "--out", str(PROC.parent), "--scenes",
                                          str(P / "scenes66.json"), "--workers", "1", "--ids", str(k)],
                                         cwd=REPO, env=env_prep, stdout=out, stderr=subprocess.STDOUT, timeout=7200)
                if rc or not complete(k, n):
                    raise RuntimeError(f"prep {k:03d} rc={rc}, artifacts complete={complete(k, n)}")
                log(f"prep {k:03d} done in {time.time() - t0:.0f} s on cpu {cpu}", event="prep", k=k, s=time.time() - t0)
            (P / f"prep_{k:03d}.done").write_text(time.strftime("%F %T") + "\n")
            raw.unlink(missing_ok=True)
            stats["prepped"] += 1
            sky_q.put(k)
        except Exception as e:
            stats["failed"].append(k)
            log(f"FAILED prep {k:03d}: {e}", event="error", k=k, stage="prep", msg=str(e))
        finally:
            free_cpus.put(cpu)
            with _lock:
                inflight.discard(k)

    def sky_worker():
        while (k := sky_q.get()) is not None:
            if not sky_ok(k):
                t0 = time.time()
                with open(P / f"sky_{k:03d}.out", "a") as out:
                    rc = subprocess.call(["taskset", "-c", cpus[0], str(DATA / "envs/drivestudio/bin/python"),
                                          "scripts/p3/ds.py", "sky", str(PROC / f"{k:03d}")],
                                         cwd=REPO, env=env_sky, stdout=out, stderr=subprocess.STDOUT, timeout=3600)
                if rc or not sky_ok(k):
                    stats["failed"].append(k)
                    log(f"FAILED sky {k:03d} rc={rc}", event="error", k=k, stage="sky")
                    continue
                log(f"sky {k:03d} done in {time.time() - t0:.0f} s", event="sky", k=k, s=time.time() - t0)
            (D / f"sky_{k:03d}.done").write_text(time.strftime("%F %T") + " pool\n")
            stats["sky"] += 1

    sky_t = threading.Thread(target=sky_worker)
    sky_t.start()
    stop_reason = None
    with ThreadPoolExecutor(len(cpus)) as ex:
        for k in todo:
            name = rows[k]["name"]
            if (P / f"prep_{k:03d}.done").exists():
                sky_q.put(k)
                continue
            while True:     # admission: disk floor after in-flight segments, and at most 2 downloads ahead of prep
                with _lock:
                    need = (len(inflight) + 1) * (PROC_GB + RAW_GB) + a.reserve_gb
                    busy = len(inflight)
                if busy < len(cpus) + 2:
                    break
                time.sleep(10)
            if free_gb() - need < a.min_free_gb:
                stop_reason = f"disk floor: {free_gb():.0f} GB free, next segment needs {need:.0f} GB above {a.min_free_gb} GB"
                log(f"STOP admission at scene {k:03d}: {stop_reason}", event="stop", k=k, reason=stop_reason)
                break
            raw = RAW / f"segment-{name}_with_camera_labels.tfrecord"
            if not complete(k, int(rows[k]["n_frames"])) and not raw.exists():
                t0 = time.time()
                part = raw.with_suffix(".part")
                r = subprocess.run([str(GCLOUD), "storage", "cp", rows[k]["gcs"], str(part)], env=env_gc,
                                   capture_output=True, text=True, timeout=3600)
                if r.returncode:
                    part.unlink(missing_ok=True)
                    stats["failed"].append(k)
                    log(f"FAILED download {k:03d}: {r.stderr[-500:]}", event="error", k=k, stage="download")
                    continue
                part.rename(raw)
                sz = raw.stat().st_size
                log(f"download {k:03d} {sz / 1e9:.2f} GB in {time.time() - t0:.0f} s ({sz / 1e6 / (time.time() - t0):.1f} MB/s)",
                    event="download", k=k, bytes=sz, s=time.time() - t0)
                stats["downloaded"] += 1
            with _lock:
                inflight.add(k)
            ex.submit(prep_one, k)
    sky_q.put(None)
    sky_t.join()
    done = sorted(k for k in range(a.first, a.last + 1) if (P / f"prep_{k:03d}.done").exists() and (D / f"sky_{k:03d}.done").exists())
    summary = {"ready": done, "n_ready": len(done), "failed": sorted(set(stats["failed"])), "stop": stop_reason,
               "free_gb": round(free_gb(), 1)}
    (P / "STATUS.md").write_text(f"# P3 pool prep ({time.strftime('%F %T')})\n\n```\n{json.dumps(summary, indent=1)}\n```\n")
    log(f"end: {summary}", event="end", **summary)
    (P / ("DONE" if not stats["failed"] else "ERROR")).write_text(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
