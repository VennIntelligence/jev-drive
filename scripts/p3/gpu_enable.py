"""Low-thread, exclusive P3 GPU continuation; CPU prep has a separate owner.

Short 3000-iteration technical smoke and default 30000-iteration scene zero
have separate checkpoints, processed data and markers. No visual PASS or GO
is inferred here. Every admission retains 1024 tasks plus 128 launch margin
(measured single-affinity sky CUDA and OmniRe imports each used five threads).
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sky", "scene0"])
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--cpus", required=True)
    a = ap.parse_args()
    data = Path(os.environ["DATA_DIR"])
    repo = Path(__file__).resolve().parents[2]
    os.chdir(repo)
    d = data / "runs/nq4/p3"
    d.mkdir(parents=True, exist_ok=True)
    lock = open(d / f"gpu_enable_{a.mode}.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (d / f"gpu_enable_{a.mode}.pid").write_text(str(os.getpid()))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu), TOKENIZERS_PARALLELISM="false", HF_HUB_DISABLE_XET="1",
               HF_ENDPOINT="https://hf-mirror.com", MAX_JOBS="1", CUDA_HOME="/usr/local/cuda-12.8")
    for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"]:
        env[k] = "1"
    ds = str(data / "envs/drivestudio/bin/python")
    jv = str(data / "envs/jevdrive/bin/python")
    op = str(data / "envs/openpilot/bin/python")
    proc = data / "processed/waymo_ds/training"

    def event(**kw):
        with open(d / "gpu_enable.events.jsonl", "a") as f:
            f.write(json.dumps(dict(t=time.time(), owner=os.getpid(), mode=a.mode, gpu=a.gpu, **kw)) + "\n")

    def wait_for(path):
        started = time.time()
        while not path.exists():
            if (d / "gpu_enable_sky.ERROR").exists() or (d / "prep.failed").exists():
                raise RuntimeError(f"Dependency failed while waiting for {path}")
            ownerfile = d / ("gpu_enable_sky.pid" if path.name == "sky_000.done" else "prep.pid")
            if ownerfile.exists() and not Path(f"/proc/{int(ownerfile.read_text())}").exists():
                raise RuntimeError(f"Dependency owner exited while waiting for {path}")
            if time.time() - started > 14400:
                raise TimeoutError(f"Dependency wait exceeded four hours: {path}")
            time.sleep(15)

    def step(name, argv, seconds, extra=None, verify=None):
        if (d / f"{name}.done").exists():
            if verify:
                verify()
            return
        admission = open(d / "gpu_admission.lock", "w")
        fcntl.flock(admission, fcntl.LOCK_EX)
        cg = Path("/sys/fs/cgroup")
        while True:
            used = int((cg / "pids.current").read_text())
            limit = (cg / "pids.max").read_text().strip()
            if limit != "max" and int(limit) - used < 1152:
                event(event="admission_wait", step=name, pids_current=used, pids_max=limit)
                time.sleep(15)
                continue
            break
        event(event="start", step=name, argv=argv, cpus=a.cpus, pids_current=used)
        (d / f"{name}.rc").unlink(missing_ok=True)
        with open(d / f"{name}.out", "a") as out:
            result = subprocess.Popen(["timeout", str(seconds), "taskset", "-c", a.cpus, *argv],
                                      env={**env, **(extra or {})}, stdout=out, stderr=subprocess.STDOUT)
            # Serialize admission through startup so both cards cannot consume
            # the same observed spare tasks. Recheck capacity for every child.
            time.sleep(5)
            event(event="startup_capacity", step=name, child=result.pid,
                  pids_current=int((cg / "pids.current").read_text()))
            fcntl.flock(admission, fcntl.LOCK_UN)
            admission.close()
            result.wait()
        (d / f"{name}.rc").write_text(str(result.returncode))
        event(event="exit", step=name, rc=result.returncode)
        if result.returncode:
            raise RuntimeError(f"{name} rc={result.returncode}; inspect {d / (name + '.out')}")
        if verify:
            verify()
        (d / f"{name}.done").write_text(time.strftime("%F %T") + "\n")

    def files_ok(*paths):
        for path in paths:
            if not path.is_file() or not path.stat().st_size:
                raise RuntimeError(f"Missing or empty output: {path}")

    def sky_ok(sd):
        expected = [f for f in (sd / "images").glob("*.jpg") if int(f.stem.split("_")[1]) in (0, 1, 2)]
        if not expected:
            raise RuntimeError(f"No sky input images: {sd}")
        files_ok(*(sd / "sky_masks" / (f.stem + ".png") for f in expected))

    def train_ok(runroot):
        files_ok(runroot / "p3/000/config.yaml")
        checkpoints = list((runroot / "p3/000").glob("checkpoint_*.pth"))
        if not checkpoints:
            raise RuntimeError(f"No trained checkpoint: {runroot}")
        files_ok(*checkpoints)

    if a.mode == "sky":
        for k in range(10):
            sd = proc / f"{k:03d}"
            wait_for(sd / "instances/instances_info.json")
            step(f"sky_{k:03d}", [ds, "scripts/p3/ds.py", "sky", str(sd)], 3600, verify=lambda: sky_ok(sd))
        return

    wait_for(d / "sky_000.done")
    for short in (True, False):
        tag = "short3000" if short else "formal"
        dataset = "nq4_p3_short3000" if short else "nq4_p3"
        runroot = data / "ckpt" / dataset
        scene = data / "processed" / dataset / "scenes/p3_000"
        extra = {"P3_SET": dataset, "P5_SET": dataset}
        train = [ds, "scripts/p3/ds.py", "train", "--scene", "0", "--data-root", str(proc), "--out-root", str(runroot)]
        if short:
            train += ["--iters", "3000"]
        step(f"{tag}_train_000", train, 3600 if short else 14400, verify=lambda: train_ok(runroot))
        step(f"{tag}_render_000", [ds, "scripts/p3/ds.py", "render", "--run", str(runroot / "p3/000"),
                                  "--target", str(d / "targets/000.json"), "--out", str(scene)], 1800,
             verify=lambda: files_ok(scene / "meta.json", scene / "render_stats.csv", *(scene / w / "frames.jsonl" for w in ("real", "plus", "minus"))))
        step(f"{tag}_index", [jv, "-m", "jevdrive.nq4_p3", "index", "--processed-root", str(proc)], 600, extra,
             verify=lambda: files_ok(*(data / "processed" / dataset / f for f in ("index.parquet", "past.npy", "future.npy", "op_plan.json"))))
        step(f"{tag}_op", [op, "scripts/p5_openpilot.py", "--models", "cinque", "lebowski", "--workers", "1"], 7200, extra)
        step(f"{tag}_opfin", [jv, "-m", "jevdrive.p5_openpilot", "finalize", "--models", "cinque,lebowski"], 900, extra)
        step(f"{tag}_exam", [jv, "-m", "jevdrive.nq4_p3", "exam"], 3600, extra)
        examroot = data / "runs/nq4" / (f"{dataset}-exam" if short else "p3-exam")
        exams = sorted(p for p in examroot.iterdir() if (p / "null_gate.csv").exists())
        if not exams:
            raise RuntimeError("Exam exited without null_gate.csv")
        step(f"{tag}_report", [jv, "-m", "jevdrive.nq4_p3", "report", "--exam-dir", str(exams[-1]),
                              "--out", str(d / tag / "report")], 600, extra,
             verify=lambda: files_ok(d / tag / "report/scenes.csv", d / tag / "report/null_gate_pooled.csv"))
    (d / "SCENE0_REVIEW_REQUIRED").write_text("Default 30000-iteration scene zero readout complete. Inspect automatic gates and four-panel ghosting figures; no GO or visual PASS granted.\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import sys
        mode = sys.argv[1] if len(sys.argv) > 1 else "unknown"
        d = Path(os.environ["DATA_DIR"]) / "runs/nq4/p3"
        (d / f"gpu_enable_{mode}.ERROR").write_text(f"{type(exc).__name__}: {exc}\n")
        raise
