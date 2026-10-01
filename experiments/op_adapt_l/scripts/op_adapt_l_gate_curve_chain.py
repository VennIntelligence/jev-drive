"""Staged tmux supervisor for gate and curve; every child inherits the granted CPU/GPU limits.

Waits for both pilot sentinels, then the gate grid/lock and two full curve runs. Only after that intermediate
check does it enqueue the six remaining runs. Samples every physical GPU every five seconds.
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl  # restructure: dirs of the script modules this file imports by bare name
_sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[3] / _d) for _d in ("experiments/op_adapt_l/scripts",)]

import concurrent.futures
import json
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from op_adapt_l_gate_curve import ROOT, grant_check  # noqa: E402
from op_adapt_l_train import Log  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
PYTHON = sys.executable
CPU_BLOCKS = {"0":"0-2", "1":"3-5", "2":"6-7,74"}
SLOTS = queue.Queue()
for _ in range(2):
    for gpu, cores in CPU_BLOCKS.items():
        SLOTS.put((gpu, cores))


def wait_for(name):
    directory=ROOT/name
    while not (directory/"DONE").exists():
        if (directory/"ERROR").exists():
            raise RuntimeError(f"Prerequisite failed: {directory}; inspect ERROR")
        time.sleep(5)


def child(script, command, name, gpu, cores, log, size=None, seed=None):
    # Re-read the grant at each process boundary; do not source and overwrite the login-shell WORKERS.
    grant_check(gpu=False)
    fields=dict(line.strip().split("=",1) for line in (ROOT/"GO").read_text().splitlines()
                if line.strip() and not line.startswith("#") and "=" in line)
    if gpu and gpu not in fields["GPUS"].strip('"').split():
        raise RuntimeError(f"GPU {gpu} no longer granted")
    if (ROOT/name/"DONE").exists(): return
    cmd=["taskset","-c",cores,PYTHON,str(REPO/"scripts"/script),command]
    if size is not None: cmd += ["--size",str(size)]
    if seed is not None: cmd += ["--seed",str(seed)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OPENBLAS_CORETYPE="Haswell",
             OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
    log.info(f"Launching {name} on GPU {gpu or 'CPU'}, cores {cores}")
    log.event("child_start",name=name,gpu=gpu,cores=cores,command=cmd)
    # Child Log mirrors useful output to its own log.txt; avoid duplicate progress-bar output in chain log.
    with (ROOT/"chain"/f"{name}.stdout.txt").open("a") as stdout:
        p=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=stdout,stderr=subprocess.STDOUT)
        log.event("child_pid",name=name,pid=p.pid)
        rc=p.wait()
    log.event("child_end",name=name,rc=rc)
    if rc or not (ROOT/name/"DONE").exists():
        raise RuntimeError(f"Child failed: {name} rc={rc}; inspect its ERROR and stdout")


def curve_job(size,seed,log):
    gpu,cores=SLOTS.get()
    try:
        child("op_adapt_l_gate_curve.py","curve-train",f"curve-{size}-s{seed}",gpu,cores,log,size,seed)
        d=ROOT/f"curve-{size}-s{seed}"
        ck=json.loads((d/"dev.json").read_text())
        if ck["nonfinite"] != 0: raise RuntimeError(f"Intermediate nonfinite loss in {d}")
        import torch
        state=torch.load(d/"ckpt-final.pt",map_location="cpu",weights_only=False)
        for x in state["model"]["net"].values():
            if not torch.isfinite(x).all(): raise RuntimeError(f"Nonfinite checkpoint in {d}")
        del state
        child("op_adapt_l_gate_curve.py","curve-eval",f"eval-{size}-s{seed}",gpu,cores,log,size,seed)
    finally:
        SLOTS.put((gpu,cores))


def gate_job(log):
    gpu,cores=SLOTS.get()
    try:
        for command in ("gate-fit","gate-lock"):
            child("op_adapt_l_gate_curve.py",command,command,gpu,cores,log)
        child("op_adapt_l_gate_curve_readout.py","gate-val","gate-val",gpu,cores,log)
    finally:
        SLOTS.put((gpu,cores))


def monitor(log,stop):
    with (ROOT/"chain/gpu_util.csv").open("a",buffering=1) as stream:
        if stream.tell()==0: stream.write("time,gpu,utilization_pct,memory_used_mib,memory_total_mib\n")
        while not stop.is_set():
            result=subprocess.run(["nvidia-smi","--query-gpu=index,utilization.gpu,memory.used,memory.total",
                                   "--format=csv,noheader,nounits"],capture_output=True,text=True)
            for line in result.stdout.splitlines():
                fields=[x.strip() for x in line.split(",")]
                stream.write(f"{time.time():.3f},{','.join(fields)}\n")
                if len(fields)==4:
                    log.scalar(f"gpu{fields[0]}/utilization_pct",float(fields[1]),int(time.time()))
            stop.wait(5)


def main():
    directory=ROOT/"chain";directory.mkdir(parents=True,exist_ok=True)
    if (directory/"DONE").exists() or (directory/"ERROR").exists(): raise SystemExit("Existing chain sentinel")
    log=Log(directory)
    assert log.tb is not None
    stop=threading.Event();thread=threading.Thread(target=monitor,args=(log,stop),daemon=True);thread.start()
    try:
        log.event("start",resources=grant_check(gpu=False))
        for name in tqdm(("gate-pilot","curve-unit"),desc="Pilot prerequisites"):
            wait_for(name)
        # Two complete runs are the preregistered intermediate stage; models and validation are never selected here.
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures=[executor.submit(gate_job,log),executor.submit(curve_job,"25",0,log),executor.submit(curve_job,"25",1,log)]
            for f in tqdm(concurrent.futures.as_completed(futures),total=3,desc="Intermediate stage"):
                f.result()
        log.event("intermediate_check",completed=2,nonfinite=0,status="passed")
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures=[executor.submit(curve_job,size,seed,log) for size in ("110","300","all") for seed in (0,1)]
            for f in tqdm(concurrent.futures.as_completed(futures),total=6,desc="Full learning curve"):
                f.result()
        # CPU-only summaries use the same granted cores, after GPU workers have exited.
        for size in tqdm(("25","110","300","all"),desc="Cluster readouts"):
            child("op_adapt_l_gate_curve_readout.py","curve-read",f"curve-read-{size}","","0-7,74",log,size)
        for command in ("frontier","figures"):
            child("op_adapt_l_gate_curve_readout.py",command,command,"","0-7,74",log)
        log.event("end",status="complete")
        (directory/"DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    except BaseException as e:
        (directory/"ERROR").write_text(traceback.format_exc());log.info(f"ERROR {e}");log.event("error",error=str(e));raise
    finally:
        stop.set();thread.join(timeout=10);log.tb.close()


if __name__=="__main__": main()
