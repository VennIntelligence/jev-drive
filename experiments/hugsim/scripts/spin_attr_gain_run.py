"""Run spin_attr_cpu_gain.py over a jobs file, N processes in parallel (box, CPU).
    python spin_attr_gain_run.py <jobs.json> <out_dir> [procs=12] [threads=4] [steps=10]
Writes <out_dir>/<key with | -> _>.json and .log per job, and DONE when all are finished."""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

jobs = json.load(open(sys.argv[1]))
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
procs = int(sys.argv[3]) if len(sys.argv) > 3 else 12
threads = sys.argv[4] if len(sys.argv) > 4 else "4"
steps = sys.argv[5] if len(sys.argv) > 5 else "10"
py = os.path.expanduser("~/data/envs/openpilot/bin/python")
script = str(Path(__file__).resolve().with_name("spin_attr_cpu_gain.py"))
env = dict(os.environ, OMP_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads)


def run(j):
    f = out / j["key"].replace("|", "_")
    if f.with_suffix(".json").exists():
        return
    with open(f.with_suffix(".log"), "w") as lg:
        subprocess.run([py, script, sys.argv[1], j["key"], str(f.with_suffix(".json")), "--threads", threads, "--steps", steps], stdout=lg, stderr=lg, env=env)


with ThreadPoolExecutor(procs) as ex:
    list(ex.map(run, jobs))
(out / "DONE").touch()
