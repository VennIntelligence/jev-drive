#!/usr/bin/env python
"""Run one route under a list of harness configurations and print the comparison table.

This is how every "before and after" number in docs/bench2drive-cost.md was produced: same route,
same tick budget, same server, one variable changed at a time. It starts its own CARLA server so
that a sweep is one command, and it always runs the `base` variant first, because a number without
its own baseline measured on the same box on the same day is not a number.

    $DATA_DIR/envs/carla/bin/python scripts/b2d_sweep.py --out $DATA_DIR/runs/b2d/sweep1
    ... --only base,front3_dec4          # a subset
    ... --ticks 400 --route-id 24240

Python 3.8: runs in envs/carla.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

DATA_DIR = Path(os.environ["DATA_DIR"])
CARLA_ROOT = Path(os.environ.get("CARLA_ROOT", DATA_DIR / "third_party/carla/CARLA_0.9.15"))
BENCH2DRIVE = Path(os.environ.get("BENCH2DRIVE_ROOT", DATA_DIR / "third_party/Bench2Drive"))
PYTHON = str(DATA_DIR / "envs/carla/bin/python")
HERE = Path(__file__).resolve().parent

# Each variant is the argument list that differs from `base`. Grouped by the question it answers.
VARIANTS = [
    # what does a sensor cost?
    ("norig", ["--rig", "none"]),
    ("front1", ["--rig", "front1"]),
    ("base", ["--rig", "front3"]),
    # Repeats of `base`, to size the run-to-run noise. Without them a 5% "improvement" cannot be
    # told from the same configuration measured twice.
    ("base2", ["--rig", "front3"]),
    ("base3", ["--rig", "front3"]),
    ("b2d6", ["--rig", "b2d6"]),
    # is the cost per pixel or per sensor?
    ("front3_800x450", ["--rig", "front3", "--width", "800", "--height", "450"]),
    ("front3_400x225", ["--rig", "front3", "--width", "400", "--height", "225"]),
    # the copy chain in the leaderboard's image callback
    ("fastcopy", ["--rig", "front3", "--fast-copy"]),
    ("zerocopy", ["--rig", "front3", "--zero-copy"]),
    ("nospectator", ["--rig", "front3", "--no-spectator"]),
    ("cachelights", ["--rig", "front3", "--cache-lights"]),
    ("small_dec4", ["--rig", "front3", "--width", "800", "--height", "450", "--decimate", "4",
                    "--zero-copy", "--no-spectator"]),
    ("small_dec4_lights", ["--rig", "front3", "--width", "800", "--height", "450",
                           "--decimate", "4", "--zero-copy", "--no-spectator", "--cache-lights"]),
    # the policy's real control rate instead of the simulator's tick rate
    ("dec2", ["--rig", "front3", "--decimate", "2"]),
    ("dec4", ["--rig", "front3", "--decimate", "4"]),
    ("dec10", ["--rig", "front3", "--decimate", "10"]),
    # with a policy that costs what ours costs
    ("policy129", ["--rig", "front3", "--policy", "sleep", "--infer-ms", "129"]),
    ("policy129_dec4", ["--rig", "front3", "--policy", "sleep", "--infer-ms", "129",
                        "--decimate", "4"]),
    ("policy129_dec4_overlap", ["--rig", "front3", "--policy", "sleep", "--infer-ms", "129",
                                "--decimate", "4", "--overlap"]),
    ("policy129_all", ["--rig", "front3", "--policy", "sleep", "--infer-ms", "129",
                       "--decimate", "4", "--overlap", "--zero-copy", "--no-spectator"]),
    # the ladder of real models (research/carla-efficiency.md). These need a policy server:
    #   b2d_policy_server.py --socket /tmp/b2d-policy.sock --backbone qwen
    # and then --extra "--policy gpu --policy-socket /tmp/b2d-policy.sock".
    # The stand-in above leaves the GPU idle; these compete with the renderer for the same card,
    # which is the difference between an optimistic floor and the real cost.
    ("gpu_dino", ["--rig", "front1"]),
    ("gpu_dino_small", ["--rig", "front1", "--width", "448", "--height", "252",
                        "--decimate", "4", "--overlap", "--zero-copy"]),
    ("gpu_qwen", ["--rig", "front3"]),
    ("gpu_qwen_dec4", ["--rig", "front3", "--decimate", "4"]),
    ("gpu_qwen_all", ["--rig", "front3", "--decimate", "4", "--overlap", "--zero-copy",
                      "--no-spectator"]),
    # Render straight at the model's input size. Free on the simulator side (per-camera cost does
    # not depend on resolution) and it deletes the resize, which is most of the policy's latency.
    ("gpu_qwen_small", ["--rig", "front3", "--width", "800", "--height", "450",
                        "--decimate", "4", "--overlap", "--zero-copy", "--no-spectator"]),
    ("gpu_qwen_final", ["--rig", "front3", "--width", "800", "--height", "450", "--decimate", "4",
                        "--overlap", "--zero-copy", "--no-spectator", "--cache-lights"]),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--routes", default=str(BENCH2DRIVE / "leaderboard/data/bench2drive220.xml"))
    p.add_argument("--route-id", default="24240", help="Town10HD, in the base package")
    p.add_argument("--ticks", type=int, default=300)
    p.add_argument("--server-index", type=int, default=20)
    p.add_argument("--quality", default="Epic")
    p.add_argument("--only", default="")
    p.add_argument("--extra", default="", help="extra args appended to every variant")
    p.add_argument("--reuse-server", action="store_true",
                   help="assume a server is already up on the index's port")
    return p.parse_args()


def start_server(index, quality, log):
    env = dict(os.environ, VK_ICD_FILENAMES="/etc/vulkan/icd.d/nvidia_icd.json")
    port = 2000 + 4 * index
    with open(log, "wb") as fh:
        proc = subprocess.Popen(
            [str(CARLA_ROOT / "CarlaUE4.sh"), "-RenderOffScreen", "-nosound",
             "-carla-rpc-port=%d" % port, "-quality-level=%s" % quality],
            stdout=fh, stderr=subprocess.STDOUT, env=env, preexec_fn=os.setsid)
    import socket
    for _ in range(90):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                time.sleep(3)
                return proc
        time.sleep(2)
    raise RuntimeError("server did not come up, see %s" % log)


def main():
    a = parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    wanted = set(x for x in a.only.split(",") if x) or None
    variants = [(n, v) for n, v in VARIANTS if wanted is None or n in wanted]

    proc = [None]

    def ensure_server():
        """Several sessions share this box and a stray cleanup elsewhere can take our server with
        it - one did, mid-sweep. A dead server must cost one variant, not the sweep."""
        if a.reuse_server:
            return
        if proc[0] is None or proc[0].poll() is not None:
            proc[0] = start_server(a.server_index, a.quality, str(out / "carla.log"))

    rows = []
    try:
        for name, extra in variants:
            ensure_server()
            vout = out / name
            cmd = [PYTHON, str(HERE / "b2d_route.py"), "--routes", a.routes,
                   "--route-id", a.route_id, "--port", str(2000 + 4 * a.server_index),
                   "--tm-port", str(8000 + a.server_index), "--out", str(vout),
                   "--max-ticks", str(a.ticks)] + extra + a.extra.split()
            print("\n=== %s ===\n%s" % (name, " ".join(cmd)), flush=True)
            with open(str(vout.parent / (name + ".log")), "wb") as fh:
                vout.mkdir(parents=True, exist_ok=True)
                subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT)
            rows.append(collect(name, vout))
            (out / "rows.json").write_text(json.dumps(rows, indent=2))
            print(json.dumps(rows[-1]), flush=True)
    finally:
        if proc[0] is not None:
            try:
                os.killpg(os.getpgid(proc[0].pid), signal.SIGKILL)
            except OSError:
                pass
    print("\n" + table(rows))
    (out / "table.md").write_text(table(rows))
    return 0


def collect(name, vout):
    row = {"variant": name}
    f = vout / "route_result.json"
    if not f.exists():
        row["status"] = "no result"
        return row
    d = json.loads(f.read_text())
    p, ag = d.get("profile", {}), d.get("agent", {}) or {}
    row["status"] = d.get("status")
    row["total_ms"] = p.get("total_ms_mean")
    row["fps"] = round(1e3 / p["total_ms_mean"], 2) if p.get("total_ms_mean") else None
    row["realtime"] = round(0.05 * 1e3 / p["total_ms_mean"], 3) if p.get("total_ms_mean") else None
    for k in ("world_tick", "provider", "agent", "tree", "spectator", "copy", "frombuffer"):
        row[k + "_ms"] = p.get(k + "_ms_mean")
    row["sensor_wait_ms"] = ag.get("sensor_wait_ms_mean")
    row["infer_ms"] = ag.get("infer_ms_mean")
    row["mib_in"] = p.get("mib_in")
    row["ticks"] = p.get("ticks")
    return row


def table(rows):
    cols = ["variant", "total_ms", "fps", "realtime", "world_tick_ms", "agent_ms",
            "sensor_wait_ms", "infer_ms", "tree_ms", "copy_ms", "mib_in", "status"]
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join([head, sep] + body)


if __name__ == "__main__":
    sys.exit(main())
