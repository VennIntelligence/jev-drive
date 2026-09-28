"""P3 expansion (user 2026-09-28, option 4): reconstruct and render pool scenes 010-065 with the registered formal chain,
and (user approval 2026-09-28) the vehicle deletion scenes of scripts/p3/veh.py, each kind with its own staged gate
(vehicles: scene 0 -> 1-9 -> the rest, a scene becomes ready when its prep is done); the two queues share the slots
round-robin, and a kind that fails its checklist stops alone (ERROR.<kind>).

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
  At most SLOTS scenes per card, a new one only with >= MIN_FREE_GB free VRAM (a scene takes ~20 GB, so >= 8 GB stay free
  for CARLA). Cards the coordinator shares while CARLA runs on them are listed in expand/shared_cards.json ({"3": 1}):
  they take that many scenes. Cores: 2 per slot, GPU 6 -> 168-171,
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
CORES = {6: ["168,169", "172,173"], **{k: [f"{24 * k + 22},{24 * k + 23}", f"{24 * k + 20},{24 * k + 21}"] for k in range(6)}}


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


class Kind:
    """One queue of scenes with its own staged gate: pedestrian pool scenes (gpu_enable.py scene) or vehicle deletion
    scenes (veh.py scene)."""

    def __init__(self, name, stages, run_dir, scene_dir, marker, cmd, ready, pidfile):
        self.name, self.stages, self.run, self.sd, self.marker, self.cmd, self.ready, self.pidfile = \
            name, stages, run_dir, scene_dir, marker, cmd, ready, pidfile
        self.si, self.active, self.tries = 0, True, {}
        while self.si < len(stages) and (E / f"{name}.stage{self.si}.ok").exists():
            self.si += 1

    def done(self, k):
        return (self.run / self.marker("render", k, "done")).exists() and (self.sd(k) / "meta.json").exists()

    def check(self, k):
        """(technical ok, psnr flag, facts): the scene-0 checklist."""
        m = json.loads((self.sd(k) / "meta.json").read_text())
        rows = list(csv.DictReader(open(self.sd(k) / "render_stats.csv")))
        psnr = sum(float(r["psnr"]) for r in rows) / len(rows)
        rc = [(self.run / self.marker(x, k, "rc")).read_text().strip() for x in ("train", "render")]
        f = {"kind": self.name, "scene": k, "deleted_missing": len(m["deleted_missing"]), "determinism": m["determinism_max_abs"],
             "psnr": round(psnr, 2), "rc_train": rc[0], "rc_render": rc[1]}
        return f["deleted_missing"] == 0 and f["determinism"] == 0 and rc == ["0", "0"] and psnr >= 22, psnr < 25, f

    def todo(self, running):
        if not self.active or self.si >= len(self.stages):
            return []
        return [k for k in self.stages[self.si] if not self.done(k) and (self.name, k) not in running and self.ready(k)
                and not (E / "skip" / f"{self.name}_{k:03d}").exists()]        # held by an operator (e.g. an orphan run)


def kinds():
    ped = Kind("ped", STAGES, D, lambda k: DATA / f"processed/nq4_p3/scenes/p3_{k:03d}",
               lambda step, k, ext: f"formal_{step}_{k:03d}.{ext}",
               lambda k, g, c: ["python3", "scripts/p3/gpu_enable.py", "scene", "--scene", str(k), "--gpu", str(g), "--cpus", c,
                                "--train-timeout", str(TRAIN_TIMEOUT)],
               lambda k: (D / f"sky_{k:03d}.done").exists(), lambda k: D / f"gpu_enable_scene_{k:03d}.pid")
    V = DATA / "runs/nq4/p3veh"
    n = len(json.loads((V / "scenes.json").read_text())["segments"]) if (V / "scenes.json").exists() else 0
    veh = Kind("veh", [[0], list(range(1, min(10, n))), list(range(10, n))] if n else [], V,
               lambda k: DATA / f"processed/nq4_p3_veh/scenes/p3_v{k:03d}", lambda step, k, ext: f"{step}_{k:03d}.{ext}",
               lambda k, g, c: ["python3", "scripts/p3/veh.py", "scene", "--j", str(k), "--gpu", str(g), "--cpus", c,
                                "--train-timeout", str(TRAIN_TIMEOUT)],
               lambda k: (V / f"prep_{k:03d}.done").exists(), lambda k: V / f"scene_{k:03d}.pid")
    # insertion items (amendment 2, option 3; user 2026-09-28: batch over every reconstructed pedestrian scene with a
    # valid donor): registered variants plus the grounded ones (--fix), one scene at a time after its reconstruction
    I = D / "insert_batch"
    ins = Kind("ins", [list(range(10)), list(range(10, 66))], I, lambda k: I / f"p3_{k:03d}", None,
               lambda k, g, c: ["taskset", "-c", c, str(DATA / "envs/drivestudio/bin/python"), "scripts/p3/insert.py", "--scene", str(k),
                                "--fix", "--out", str(I), "--gpu-tag", str(g)],
               lambda k: k < 10 or ped.done(k), lambda k: I / f"scene_{k:03d}.pid")
    ins.done = lambda k: (I / f"p3_{k:03d}" / "meta.json").exists()
    ins.check = lambda k: (True, False, {"kind": "ins", "scene": k,
                                         "chosen": json.loads((I / f"p3_{k:03d}" / "meta.json").read_text()).get("chosen")})
    return [ped, veh, ins]


def status(running, ks):
    lines = [f"# P3 expansion {time.strftime('%F %T')}", "", f"pid {os.getpid()}", ""]
    for kd in ks:
        n_done = sum(kd.done(k) for s in kd.stages for k in s)
        lines.append(f"- {kd.name}: stage {min(kd.si + 1, len(kd.stages))} of {len(kd.stages)}, done {n_done} / "
                     f"{sum(map(len, kd.stages))}{'' if kd.active else ' (STOPPED: see ERROR.' + kd.name + ')'}")
    lines += ["", "| kind | scene | card | cores | started |", "|:--|--:|--:|:--|:--|"]
    lines += [f"| {kd} | {k:03d} | {g} | {c} | {time.strftime('%H:%M', time.localtime(t0))} |" for (kd, k), (p, g, c, t0) in sorted(running.items())]
    lines += ["", f"log {E / 'log.txt'}"]
    (E / "STATUS.md").write_text("\n".join(lines) + "\n")


def main():
    E.mkdir(parents=True, exist_ok=True)
    lock = open(E / "owner.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (E / "pid").write_text(str(os.getpid()))
    for k in (0, 1, 2):                            # legacy single-kind stage markers
        if (E / f"stage{k}.ok").exists() and not (E / f"ped.stage{k}.ok").exists():
            (E / f"stage{k}.ok").rename(E / f"ped.stage{k}.ok")
    pool = json.loads((D / "pool/scenes66.json").read_text())["segments"]
    for k in range(10, 66):                       # render targets: pool targets, equal to the registered selection
        src, dst = D / "pool/targets" / f"{k:03d}.json", D / "targets" / f"{k:03d}.json"
        t = json.loads(src.read_text())
        assert t["segment"] == pool[k] and t["scene"] == k, k
        if dst.exists():
            assert json.loads(dst.read_text()) == t, f"target {k} differs"
        else:
            dst.write_text(src.read_text())
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    ks = kinds()
    running = {}
    for kd in ks:                                  # adopt live scene processes of an earlier instance
        for k in [k for s in kd.stages for k in s]:
            pf = kd.pidfile(k)
            if pf.exists() and not kd.done(k):
                pid = int(pf.read_text())
                try:
                    args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
                except OSError:
                    continue
                gk = b"--gpu" if b"--gpu" in args else b"--gpu-tag" if b"--gpu-tag" in args else None
                if gk and (b"scene" in args or b"--scene" in args):
                    g = int(args[args.index(gk) + 1])
                    c = args[args.index(b"--cpus") + 1].decode() if b"--cpus" in args else CORES[g][0]
                    running[(kd.name, k)] = (Adopted(pid), g, c, time.time())
                    log(f"adopted {kd.name} {k:03d} (pid {pid}, GPU {g})", event="adopt", kind=kd.name, scene=k, pid=pid)
    by = {kd.name: kd for kd in ks}
    while True:
        for (kn, k), (p, g, c, t0) in list(running.items()):
            if p.poll() is None:
                continue
            del running[(kn, k)]
            kd = by[kn]
            if kd.done(k):
                log(f"{kn} {k:03d} done on GPU {g} in {(time.time() - t0) / 60:.0f} min", event="scene_done", kind=kn, scene=k, gpu=g,
                    minutes=round((time.time() - t0) / 60, 1))
            else:
                kd.tries[k] = kd.tries.get(k, 0) + 1
                log(f"{kn} {k:03d} failed on GPU {g} (try {kd.tries[k]})", event="scene_failed", kind=kn, scene=k, gpu=g)
                (D / f"gpu_enable_scene_{k:03d}.ERROR").unlink(missing_ok=True)
                if kd.tries[k] >= 2:
                    kd.active = False
                    (E / f"ERROR.{kn}").write_text(f"{kn} scene {k:03d} failed twice\n")
                    log(f"ERROR: {kn} stopped (scene {k:03d} failed twice); the other kind goes on", event="error", kind=kn, scene=k)
        for kd in ks:                              # stage gates
            if not kd.active or kd.si >= len(kd.stages):
                continue
            st = kd.stages[kd.si]
            if all(kd.done(k) for k in st) and not any(kn == kd.name for kn, _ in running):
                res = [kd.check(k) for k in st]
                flags, bad = [f for _, fl, f in res if fl], [f for ok, _, f in res if not ok]
                (E / f"{kd.name}.stage{kd.si}.checklist.json").write_text(json.dumps([f for _, _, f in res], indent=1))
                if bad or len(flags) > max(1, len(st) // 10):
                    kd.active = False
                    (E / f"ERROR.{kd.name}").write_text(f"stage {kd.si + 1} checklist failed: bad {bad}, psnr flags {flags}\n")
                    log(f"ERROR: {kd.name} stage {kd.si + 1} checklist failed", event="checklist_failed", kind=kd.name, bad=bad, flags=flags)
                    continue
                (E / f"{kd.name}.stage{kd.si}.ok").write_text(time.strftime("%F %T") + "\n")
                log(f"{kd.name} stage {kd.si + 1} checklist ok ({len(flags)} PSNR flags)", event="stage_ok", kind=kd.name, stage=kd.si)
                kd.si += 1
        if all(not kd.active or kd.si >= len(kd.stages) for kd in ks) and not running:
            break
        queues = [kd.todo(running) for kd in ks]
        order = [(kd.name, k) for tup in __import__("itertools").zip_longest(*queues) for kd, k in zip(ks, tup) if k is not None]
        if order and not (E / "HOLD").exists():
            carla = carla_per_card()
            eligible = [TEST_GPU] + [g for g in range(6) if g not in g_cards() and g not in demand_cards() and not carla.get(g)]
            try:                                   # cards shared with CARLA by the coordinator's grant: {"3": 1, ...} slots
                shared = {int(g): int(n) for g, n in json.loads((E / "shared_cards.json").read_text()).items()}
            except (OSError, ValueError):
                shared = {}
            eligible += [g for g in shared if g not in eligible and g not in demand_cards()]
            cap = {g: shared.get(g, SLOTS) if g in shared and g != TEST_GPU else SLOTS for g in eligible}
            want = [g for g in eligible if sum(v[1] == g for v in running.values()) < cap[g]]
            info = free_gb() if want else {}
            for g in want:
                mine = [u for u, v in running.items() if v[1] == g]
                while order and len(mine) < cap[g] and info.get(g, 0) >= MIN_FREE_GB:
                    kn, k = order.pop(0)
                    cores = next(c for c in CORES[g] if c not in [running[u][2] for u in mine])
                    with open(E / f"{kn}_{k:03d}.out", "a") as out:
                        p = subprocess.Popen(by[kn].cmd(k, g, cores), cwd=REPO, env={**env, "CUDA_VISIBLE_DEVICES": str(g)} if kn == "ins" else env,
                                             stdout=out, stderr=subprocess.STDOUT,
                                             start_new_session=True)   # survives a lane restart; adopted by pid
                    running[(kn, k)] = (p, g, cores, time.time())
                    mine.append((kn, k))
                    info[g] -= MIN_FREE_GB
                    log(f"{kn} {k:03d} started on GPU {g}, cores {cores}, pid {p.pid}", event="scene_start", kind=kn, scene=k, gpu=g, pid=p.pid)
                    time.sleep(20)
        status(running, ks)
        time.sleep(POLL_S)
    if by["ped"].si >= len(STAGES):
        scenes = sorted(int(p.parent.name[3:]) for p in (DATA / "processed/nq4_p3/scenes").glob("p3_*/meta.json"))
        log(f"pedestrian readout over {len(scenes)} scenes on GPU {TEST_GPU}", event="readout", scenes=scenes)
        rc = subprocess.call(["python3", "scripts/p3/gpu_enable.py", "readout", "--tag", "expand66", "--scenes", *map(str, scenes),
                              "--gpu", str(TEST_GPU), "--cpus", "168-171"], cwd=REPO, env=env,
                             stdout=open(E / "readout.out", "a"), stderr=subprocess.STDOUT)
        log(f"pedestrian readout rc={rc}", event="readout_done", rc=rc)
    (E / "DONE").write_text(time.strftime("%F %T") + "\n")
    log("DONE", event="done")


if __name__ == "__main__":
    main()
