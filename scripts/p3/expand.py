"""P3 expansion (user 2026-09-28, option 4): reconstruct and render pool scenes 010-065 with the registered formal chain.

Each scene runs `scripts/p3/gpu_enable.py scene --scene k` exactly as scenes 000-009 did (OmniRe without SMPL, default
30000 iterations, then real / x+ / x- at the registered target; pool/targets/<k>.json, copied to targets/ after an
equality check). Staged: scene 010 -> check -> 011-019 -> check -> 020-065 -> check -> the pooled readout of all rendered
scenes (gpu_enable.py readout, tag expand66) on the test card. A stage that fails the checklist stops the lane.

Cards, re-evaluated every POLL_S (event-triggered, work-conserving, nothing is ever evicted). No nvidia-smi polling
(it takes the driver's device lock; on 2026-09-28 polling storms helped CARLA start-ups stall): CARLA servers per card are
counted from /proc (-graphicsadapter), and free VRAM is queried at most once per SMI_S, only when a slot is free and a
scene is waiting. While $DATA_DIR/runs/nq4/p3/expand/HOLD exists no new scene starts (running ones go on); a restarted
lane adopts the live scene processes from their gpu_enable pid files.
  GPU 6 (the test card) always; card k in 0-5 only when it has left the nq4-g grant (not in the row's gpus, or the row is
  done / revoked), runs no CARLA server of any owner, and is not named in a wm-loop demand file.
  At most SLOTS scenes per card, a new one only with >= MIN_FREE_GB free VRAM. Cores: 2 per slot, GPU 6 -> 168-171,
  card k -> the last 4 cores of that card's G slice (24 k + 20 .. 24 k + 23).
Checklist (the scene-0 list): train and render rc0; deleted_missing == 0; determinism max |d| == 0; full-image PSNR
>= 25 dB. PSNR below 25 dB is a flag (at most one per 10 scenes, as p3_004 in the gate set); below 22 dB, a technical
failure after one retry, or more flags than that stop the lane (ERROR).
Outputs: $DATA_DIR/runs/nq4/p3/expand/{log.txt, events.jsonl, STATUS.md, stage<i>.ok, DONE, ERROR}.

  scripts/tmux_run.sh p3-expand python3 scripts/p3/expand.py
"""
import csv
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

DATA = Path(os.environ["DATA_DIR"])
REPO = Path(__file__).resolve().parents[2]
D = DATA / "runs/nq4/p3"
E = D / "expand"
STAGES = [[10], list(range(11, 20)), list(range(20, 66))]
TEST_GPU, SLOTS, MIN_FREE_GB, POLL_S, SMI_S, TRAIN_TIMEOUT = 6, 2, 30, 120, 300, 6 * 3600
CORES = {6: ["168,169", "170,171"], **{k: [f"{24 * k + 20},{24 * k + 21}", f"{24 * k + 22},{24 * k + 23}"] for k in range(6)}}


def log(msg, **ev):
    line = f"{time.strftime('%F %T')} {msg}"
    print(line, flush=True)
    with open(E / "log.txt", "a") as f:
        f.write(line + "\n")
    with open(E / "events.jsonl", "a") as f:
        f.write(json.dumps({"t": time.time(), "msg": msg, **ev}) + "\n")


_SMI = [0.0, {}]


def free_gb() -> dict:
    """index -> free VRAM (GB), one nvidia-smi query at most every SMI_S."""
    if time.time() - _SMI[0] > SMI_S:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=60).stdout
        _SMI[:] = [time.time(), {int(i): (float(t) - float(u)) / 1024 for i, t, u in (l.split(",") for l in q.strip().splitlines())}]
    return dict(_SMI[1])


def carla_per_card() -> dict:
    """index -> CARLA servers, from the server command lines (-graphicsadapter=k), no driver query."""
    out = {}
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmd = (p / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if cmd and b"CarlaUE4" in cmd[0]:
            k = next((int(a.split(b"=")[1]) for a in cmd if a.startswith(b"-graphicsadapter=")), 0)
            out[k] = out.get(k, 0) + 1
    return out


class Adopted:
    """A scene process started by an earlier lane instance, followed through /proc."""

    def __init__(self, pid: int):
        self.pid = pid

    def poll(self):
        return None if Path(f"/proc/{self.pid}").exists() else -1


def g_cards() -> set:
    """Cards still granted to the G lane (runs/sched/table.tsv row nq4-g)."""
    for r in csv.DictReader(open(DATA / "runs/sched/table.tsv"), delimiter="\t"):
        if r["lane"] == "nq4-g":
            if r["status"].startswith(("done", "revoked")):
                return set()
            return {int(x) for x in r["gpus"].split(",") if x.strip().isdigit()}
    return set()


def demand_cards() -> set:
    out = set()
    for f in (DATA / "runs/sched/demand").glob("wm-loop*.json"):
        try:
            out |= {int(k) for k, v in json.loads(f.read_text()).items() if v}
        except (ValueError, OSError):
            pass
    return out


def done(k: int) -> bool:
    return (D / f"formal_render_{k:03d}.done").exists() and (DATA / f"processed/nq4_p3/scenes/p3_{k:03d}/meta.json").exists()


def check(k: int) -> tuple[bool, bool, dict]:
    """(technical ok, psnr flag, facts) for one rendered scene."""
    sd = DATA / f"processed/nq4_p3/scenes/p3_{k:03d}"
    m = json.loads((sd / "meta.json").read_text())
    rows = list(csv.DictReader(open(sd / "render_stats.csv")))
    psnr = sum(float(r["psnr"]) for r in rows) / len(rows)
    f = {"scene": k, "deleted_missing": len(m["deleted_missing"]), "determinism": m["determinism_max_abs"], "psnr": round(psnr, 2),
         "rc_train": (D / f"formal_train_{k:03d}.rc").read_text().strip(), "rc_render": (D / f"formal_render_{k:03d}.rc").read_text().strip()}
    tech = f["deleted_missing"] == 0 and f["determinism"] == 0 and f["rc_train"] == "0" and f["rc_render"] == "0" and psnr >= 22
    return tech, psnr < 25, f


def status(running, stage_i):
    lines = [f"# P3 expansion {time.strftime('%F %T')}", "", f"stage {stage_i + 1} of {len(STAGES)}; pid {os.getpid()}", "",
             "| scene | card | cores | started |", "|--:|--:|:--|:--|"]
    lines += [f"| {k:03d} | {g} | {c} | {time.strftime('%H:%M', time.localtime(t0))} |" for k, (p, g, c, t0) in sorted(running.items())]
    n_done = sum(done(k) for s in STAGES for k in s)
    lines += ["", f"done {n_done} / {sum(map(len, STAGES))} pool scenes; log {E / 'log.txt'}"]
    (E / "STATUS.md").write_text("\n".join(lines) + "\n")


def main():
    E.mkdir(parents=True, exist_ok=True)
    lock = open(E / "owner.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (E / "pid").write_text(str(os.getpid()))
    (E / "ERROR").unlink(missing_ok=True)
    pool = json.loads((D / "pool/scenes66.json").read_text())["segments"]
    for k in range(10, 66):                       # render targets: pool targets, equal to the registered selection
        src, dst = D / "pool/targets" / f"{k:03d}.json", D / "targets" / f"{k:03d}.json"
        t = json.loads(src.read_text())
        assert t["segment"] == pool[k] and t["scene"] == k, k
        if dst.exists():
            assert json.loads(dst.read_text()) == t, f"target {k} differs"
        else:
            dst.write_text(src.read_text())
        assert (D / f"sky_{k:03d}.done").exists(), f"sky masks of {k} missing"
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    running, tries = {}, {}
    for k in range(10, 66):                       # adopt live scene processes of an earlier instance
        pf = D / f"gpu_enable_scene_{k:03d}.pid"
        if pf.exists() and not done(k):
            pid = int(pf.read_text())
            try:
                cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
            except OSError:
                continue
            if b"gpu_enable.py" in cmd and b"scene" in cmd:
                args = cmd.split(b"\0")
                g = int(args[args.index(b"--gpu") + 1])
                c = args[args.index(b"--cpus") + 1].decode()
                running[k] = (Adopted(pid), g, c, time.time())
                log(f"adopted scene {k:03d} (pid {pid}, GPU {g})", event="adopt", scene=k, pid=pid)
    for si, stage in enumerate(STAGES):
        if (E / f"stage{si}.ok").exists():
            continue
        log(f"stage {si + 1}: scenes {stage[0]:03d}-{stage[-1]:03d}", event="stage", stage=si, scenes=stage)
        while True:
            for k, (p, g, c, t0) in list(running.items()):
                rc = p.poll()
                if rc is None:
                    continue
                del running[k]
                if done(k):
                    log(f"scene {k:03d} done on GPU {g} in {(time.time() - t0) / 60:.0f} min", event="scene_done", scene=k, gpu=g,
                        minutes=round((time.time() - t0) / 60, 1))
                else:
                    tries[k] = tries.get(k, 0) + 1
                    log(f"scene {k:03d} failed rc={rc} on GPU {g} (try {tries[k]})", event="scene_failed", scene=k, gpu=g, rc=rc)
                    (D / f"gpu_enable_scene_{k:03d}.ERROR").unlink(missing_ok=True)
                    if tries[k] >= 2:
                        (E / "ERROR").write_text(f"scene {k:03d} failed twice; see {D}/formal_*_{k:03d}.out\n")
                        log("ERROR: stopping; running scenes finish on their own", event="error", scene=k)
                        return
            todo = [k for k in stage if not done(k) and k not in running]
            if not todo and not running:
                break
            if (E / "HOLD").exists() or not todo:
                status(running, si)
                time.sleep(POLL_S)
                continue
            carla = carla_per_card()
            eligible = [TEST_GPU] + [g for g in range(6) if g not in g_cards() and g not in demand_cards() and not carla.get(g)]
            want = [g for g in eligible if sum(v[1] == g for v in running.values()) < SLOTS]
            info = free_gb() if want else {}
            for g in want:
                mine = [k for k, v in running.items() if v[1] == g]
                while todo and len(mine) < SLOTS and info.get(g, 0) >= MIN_FREE_GB:
                    k = todo.pop(0)
                    cores = next(c for c in CORES[g] if c not in [running[j][2] for j in mine])
                    with open(E / f"scene_{k:03d}.out", "a") as out:
                        p = subprocess.Popen(["python3", "scripts/p3/gpu_enable.py", "scene", "--scene", str(k), "--gpu", str(g),
                                              "--cpus", cores, "--train-timeout", str(TRAIN_TIMEOUT)],
                                             cwd=REPO, env=env, stdout=out, stderr=subprocess.STDOUT)
                    running[k] = (p, g, cores, time.time())
                    mine.append(k)
                    info[g] -= MIN_FREE_GB
                    log(f"scene {k:03d} started on GPU {g}, cores {cores}, pid {p.pid}", event="scene_start", scene=k, gpu=g, pid=p.pid)
                    time.sleep(20)                 # let the first allocations land before the next VRAM reading
            status(running, si)
            time.sleep(POLL_S)
        res = [check(k) for k in stage]
        flags = [f for ok, fl, f in res if fl]
        bad = [f for ok, fl, f in res if not ok]
        (E / f"stage{si}.checklist.json").write_text(json.dumps([f for _, _, f in res], indent=1))
        if bad or len(flags) > max(1, len(stage) // 10):
            (E / "ERROR").write_text(f"stage {si + 1} checklist failed: bad {bad}, psnr flags {flags}\n")
            log("ERROR: checklist failed", event="checklist_failed", bad=bad, flags=flags)
            return
        (E / f"stage{si}.ok").write_text(time.strftime("%F %T") + "\n")
        log(f"stage {si + 1} checklist ok ({len(flags)} PSNR flags)", event="stage_ok", stage=si, flags=flags)
    scenes = sorted(int(p.parent.name[3:]) for p in (DATA / "processed/nq4_p3/scenes").glob("p3_*/meta.json"))
    log(f"readout over {len(scenes)} scenes on GPU {TEST_GPU}", event="readout", scenes=scenes)
    rc = subprocess.call(["python3", "scripts/p3/gpu_enable.py", "readout", "--tag", "expand66", "--scenes", *map(str, scenes),
                          "--gpu", str(TEST_GPU), "--cpus", "168-171"], cwd=REPO, env=env,
                         stdout=open(E / "readout.out", "a"), stderr=subprocess.STDOUT)
    if rc:
        (E / "ERROR").write_text(f"readout rc={rc}; see {E / 'readout.out'}\n")
        log("ERROR: readout failed", event="error", rc=rc)
        return
    (E / "DONE").write_text(time.strftime("%F %T") + "\n")
    log("DONE", event="done")


if __name__ == "__main__":
    main()
