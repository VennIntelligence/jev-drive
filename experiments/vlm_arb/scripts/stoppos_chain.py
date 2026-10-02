"""The stoppos chain: one resumable script, STATUS / DONE / ERROR in its run dir (jevdrive.run.Run).

  .venv/bin/python experiments/vlm_arb/scripts/stoppos_chain.py [--resume RUN_DIR] [--stages a,b] [--gpu 1]

Stages in order (a stage with `stages/<name>.done` in the run dir is skipped on a rerun):
  labels    frame tables, route tables and the stream list ($DATA_DIR/processed/vlm_arb_stoppos)
  extract1  3 evenly spaced streams per set (staged launch), then extract10 (10 per set), then extract (all)
  analyze   stoppos_analyze.py: fidelity, native outputs, probes, heads, replay, scaling, report (CPU)
GPU: only the card given by --gpu (default 1), only in the extract stages; the analysis is CPU.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from jevdrive.run import Run, cli_args  # noqa: E402

DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
PROC = DATA / "processed/vlm_arb_stoppos"
PY = str(REPO / ".venv/bin/python")
PYOP = str(DATA / "envs/openpilot/bin/python")
STAGES = ["labels", "extract1", "extract10", "extract", "analyze"]


def sh(run, cmd, env=None, log=None):
    run.info("$ %s", " ".join(cmd))
    with open(run.dir / f"{log or 'cmd'}.log", "a") as f:
        p = subprocess.run(cmd, cwd=str(REPO), env=dict(os.environ, **(env or {})), stdout=f, stderr=subprocess.STDOUT)
    if p.returncode:
        tail = (run.dir / f"{log or 'cmd'}.log").read_text().splitlines()[-25:]
        raise RuntimeError("%s failed (%d):\n%s" % (cmd[1], p.returncode, "\n".join(tail)))


def main(a):
    with Run("vlm_arb_stoppos", a.tag, seed=0, config=vars(a), resume=a.resume) as run:
        done = run.dir / "stages"
        done.mkdir(exist_ok=True)
        want = a.stages.split(",") if a.stages else STAGES
        for st in STAGES:
            if st not in want or (done / f"{st}.done").exists():
                continue
            run.info("stage %s", st)
            (run.dir / "STATUS").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + f" stoppos_chain: stage {st}\n")
            if st == "labels":
                sh(run, [PY, str(SCR / "stoppos_labels.py"), "build"], log="labels")
            elif st.startswith("extract"):
                per = {"extract1": ["--per-src", "3"], "extract10": ["--per-src", "10"], "extract": []}[st]
                feats = DATA / "runs/vlm_arb_stoppos/extract"
                (feats / "ERROR").unlink(missing_ok=True)
                sh(run, [PYOP, str(SCR / "stoppos_extract.py"), "--out", str(feats), "--streams", str(PROC / "streams.json"),
                         "--workers", str(a.workers), *per], env={"CUDA_VISIBLE_DEVICES": str(a.gpu), "OMP_NUM_THREADS": "2"}, log=st)
                (DATA / "runs/vlm_arb_stoppos/extract/DONE").unlink(missing_ok=True)
            elif st == "analyze":
                sh(run, [PY, str(SCR / "stoppos_analyze.py"), "--run", str(run.dir)], env={"OMP_NUM_THREADS": str(a.threads), "CUDA_VISIBLE_DEVICES": str(a.gpu)}, log="analyze")
            (done / f"{st}.done").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        run.summary["stages"] = sorted(p.stem for p in done.glob("*.done"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    cli_args(ap)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--stages", default="")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--threads", type=int, default=24)
    main(ap.parse_args())
