"""Chase-camera recording for the vlm_arb arms: the unchanged `RecordedOpDrive` recorder (op_drive_record_agent.py, camera, encoder and scene log
as in the privileged-ceiling batch) mixed into the arm's own agent. No driving logic is touched (Python 3.8).

  GIF_BASE=vlm  -> RecordedVlmArb(RecordedOpDrive, VlmArbAgent)   arms drive / vred / vred2 / vred3 (lib/vlm_arb_agent.py)
  otherwise     -> RecordedOpDrive over lib/op_arb_agent.py       arms pbyp / pbyp2 / pbyp2ng (privileged, PC_ENABLE=1)

Used as OP_ARB_AGENT by vlm_arb_gif_chain.py. Output per attempt: chase_raw.mp4 (1280x720, 10 fps, no overlay), video_frames.jsonl (frame ids),
video_summary.json, scene.jsonl.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for _d in (str(REPO / "scripts"), str(REPO / "lib")):
    if _d not in sys.path:
        sys.path.insert(0, _d)
import vlm_arb_agent  # noqa: E402,F401  (puts lib/ first: op_arb_agent below is lib/op_arb_agent.py, as the original runs used)
from op_arb_agent import OpArbAgent  # noqa: E402,F401
from vlm_arb_agent import VlmArbAgent  # noqa: E402

sys.path.append(str(REPO / "experiments/op_closed_loop/archive"))
from op_drive_record_agent import RecordedOpDrive  # noqa: E402


class RecordedVlmArb(RecordedOpDrive, VlmArbAgent):
    pass


def get_entry_point():
    return "RecordedVlmArb" if os.environ.get("GIF_BASE") == "vlm" else "RecordedOpDrive"
