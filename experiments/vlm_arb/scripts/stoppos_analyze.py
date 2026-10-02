"""Analysis stage of the stoppos chain (plan 2026-10-03-stoppos-probe.md): native outputs and replay fidelity, cross-validated probes and
heads, tables and figures, data-size curve. Outputs in RUN_DIR/results (copied to experiments/vlm_arb/results/stoppos by hand, never
written into the box checkout). Each step is skipped when its output exists.

  .venv/bin/python experiments/vlm_arb/scripts/stoppos_analyze.py --run RUN_DIR
"""
import argparse
import subprocess
import sys
from pathlib import Path

SCR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCR))
import stoppos_data as D  # noqa: E402
import stoppos_native as N  # noqa: E402

PY = sys.executable


def main(a):
    run = Path(a.run)
    res = run / "results"
    res.mkdir(exist_ok=True)

    def status(m):
        import time
        (run / "STATUS").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + " stoppos_analyze: " + m + "\n")
        print(m, flush=True)
    frames = D.load_frames()
    if not (res / "native_discrimination.csv").exists():
        status("native outputs (logs)")
        N.native_logs(res)
    if not (res / "fidelity.csv").exists():
        status("replay fidelity")
        N.fidelity(res, frames)
    if not (run / "oof.npz").exists():
        status("probes (CV)")
        subprocess.run([PY, str(SCR / "stoppos_probe.py"), "--run", str(run)], check=True)
    if not (res / "primary_tap.txt").exists():
        status("report tables")
        subprocess.run([PY, str(SCR / "stoppos_report.py"), "--run", str(run), "--out", str(res)], check=True)
    tap = (res / "primary_tap.txt").read_text().strip()
    if not (res / f"scaling_{tap}.csv").exists():
        status("data size curve")
        subprocess.run([PY, str(SCR / "stoppos_scaling.py"), "--run", str(run), "--tap", tap, "--out", str(res)], check=True)
    status("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    main(ap.parse_args())
