"""The GIF lane: rerun chosen vlm_arb cases with the chase camera on (same arm, route, traffic seed, configuration as the original unit) so event clips can be cut.

  scripts/tmux_run.sh vlm-gif .venv/bin/python -m jevdrive.cl run experiments/vlm_arb/scripts/vlm_arb_gif_chain.py \
      --lane vlm-gif --workers-per-card 3 --arg set=<name>[,<name>]

Per-card Qwen servers (arms vred*): `$DATA_DIR/envs/jevdrive/bin/python experiments/vlm_arb/scripts/vlm_qwen_server.py supervise --cards 0,1,2 --run $DATA_DIR/runs/vlm_arb_gif/qwen`.
At most 3 workers per card (the load the VLM latency was calibrated at). Outputs: $DATA_DIR/runs/vlm_arb/arms/v2-<unit>/attempts/<route>/<n>/chase_raw.mp4. Nothing here changes driving
logic: the recorder is vlm_arb_record_agent.py (RecordedOpDrive of the privileged-ceiling batch).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlm_arb_chain as base  # noqa: E402
import vlm_v2_chain as v2  # noqa: E402
import vlm_vred_chain as vc  # noqa: E402
from vlm_arb_common import RUN  # noqa: E402

NAME, ROOT = "vlm-gif", "vlm_arb_gif"
base.SLOT_WORKERS, base.SLOT_CORES = 3, 9
AGENT = "experiments/vlm_arb/scripts/vlm_arb_record_agent.py"
L = v2.L_REGISTERED


def unit(arm, real, seed, shard, ids, env=None, rep=0):
    e = dict(env or {}, OP_ARB_AGENT=AGENT)
    if real in ("drive", "vred", "vred3"):
        e["GIF_BASE"] = "vlm"
    j = base.unit("gif%s-%s" % ("" if rep == 0 else "r%d" % rep, arm), seed, shard, ids, e, "", 0, base=real)
    j.ok = None
    return j


def sets(rep):
    u = lambda *a, **k: unit(*a, rep=rep, **k)  # noqa: E731
    qv = vc.qwen_env(L)
    return {
        "vred_a": lambda: [u("vred", "vred", 0, "a", ["24944", "27297", "16390"], qv)],
        "vred_b": lambda: [u("vred", "vred", 0, "b", ["16529"], qv)],
        "vred_s1": lambda: [u("vred", "vred", 1, "c", ["15612"], qv)],
        "vred2": lambda: [u("vred2", "vred", 1, "a", ["15483"], dict(qv, **v2.STOPLINE))],
        "vred3": lambda: [u("vred3", "vred3", 0, "a", ["27297"], v2.v3_env())],
        "drive_a": lambda: [u("drive", "drive", 0, "a", ["24944", "19324", "27043"])],
        "drive_b": lambda: [u("drive", "drive", 0, "b", ["9196", "37969"])],
        "pbyp2": lambda: [u("pbyp2", "pbyp2", 0, "a", ["24497", "19324"])],
        "pbyp2ng": lambda: [u("pbyp2ng", "pbyp2ng", 0, "a", ["19832", "2520"])],
        "pbyp": lambda: [u("pbyp", "pbyp", 0, "a", ["27297", "17280"])],
    }


def jobs(args):
    RUN.mkdir(parents=True, exist_ok=True)
    rep = int(args.get("rep", 0))
    table = sets(rep)
    out = []
    for name in args["set"].split(","):
        out += table[name]()
    return out
