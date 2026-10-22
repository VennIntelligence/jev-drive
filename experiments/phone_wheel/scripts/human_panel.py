#!/usr/bin/env python3
"""Control panel of the Bench2Drive human exam: lib/exam_web.py (web page, default) or lib/exam_panel.py (Tk, --tk)
with the backend lib/b2d_exam.py. The desktop icon (human_panel.sh) starts the server and opens it as a Chrome app window.

    /usr/bin/python3 experiments/phone_wheel/scripts/human_panel.py --port 8771    # web panel on 127.0.0.1 + the Tailscale IP
    /usr/bin/python3 experiments/phone_wheel/scripts/human_panel.py --tk           # the old Tk window (system Python has Tcl)
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2] / "lib"), str(HERE.parent / "lib")]
from b2d_exam import B2DExam  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tk", action="store_true", help="the Tk window instead of the web panel")
    ap.add_argument("--port", type=int, default=8771)
    ap.add_argument("--host", action="append", help="bind address (repeatable; default 127.0.0.1 + the Tailscale IPv4)")
    ap.add_argument("--no-side", action="store_true", help="never start the wheel daemon / viewer (tests)")
    a = ap.parse_args()
    if a.tk:
        from exam_panel import panel
        panel(B2DExam())
    else:
        from exam_web import serve
        serve(B2DExam(), a.port, a.host, side=not a.no_side)
