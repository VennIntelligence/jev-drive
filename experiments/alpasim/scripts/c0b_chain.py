#!/usr/bin/env python3
"""C0b chain: five drivers on every landed public AlpaSim nuPlan scene, as GPU-pool jobs with a stall watchdog.

  scripts/tmux_run.sh c0b python3 experiments/alpasim/scripts/c0b_chain.py          (box; plain python3, stdlib only)

Stages: (0) scene lists from the shards that have a .done marker; (1) pilots: WA-JEPA and OT30-F-s0 on 8 scenes, SH30 and AP2
on 24 overlap scenes of the 400-scene runs (determinism check: scores must reproduce exactly, otherwise the full list is
rerun for that driver); (3) everything else, each driver chunk its own pool job. Writes STATUS / DONE / ERROR in
$DATA_DIR/runs/alpasim/c0b/. A job whose run shows no new finished rollout for STALL_MIN minutes is cancelled through the
pool and resubmitted (the runtime has no resume), at most MAX_TRIES times. Finished runs are pruned: rollout.asl is kept
only for rollouts with a failure reason (the zero-score scenes).
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

DATA = Path(os.environ["DATA_DIR"])
REPO = Path(__file__).resolve().parents[3]
RA = DATA / "runs/alpasim"
O = RA / "c0b"
ROOT = DATA / "datasets/alpasim_nuplan"
STALL_MIN = float(os.environ.get("STALL_MIN", 20))
MAX_TRIES = 3
POLL_S = 60
OVERRIDES = ["+e2e_challenge_nuplan=full", "runtime.nr_workers=2", "runtime.endpoints.renderer.n_concurrent_rollouts=8",
             "runtime.endpoints.driver.n_concurrent_rollouts=8", "runtime.endpoints.controller.n_concurrent_rollouts=8",
             "defines.nre_cache_size=9"]
C0 = {"sh30": RA / "c0/sh30_400_rerun", "ap2": RA / "c0/ap2_400_r1"}   # last night's 400-scene runs (one timestamped dir inside)
PY = [sys.executable, "-m", "jevdrive.cl"]


def log(msg):
    line = f"{time.strftime('%F %T')} {msg}"
    print(line, flush=True)
    (O / "log.txt").open("a").write(line + "\n")


def status(msg):
    (O / "STATUS").write_text(f"{time.strftime('%F %T')} {msg}\n")
    log(msg)


def fail(msg):
    (O / "ERROR").write_text(f"{time.strftime('%F %T')} {msg}\n")
    log("ERROR " + msg)
    sys.exit(1)


def free_gb():
    s = os.statvfs(ROOT)
    return s.f_bavail * s.f_frsize / 1e9


def only_run_dir(p):
    return sorted(Path(p).iterdir())[-1]


def load_scores(run):
    d = json.loads((Path(run) / "aggregate/results-summary.json").read_text())
    return {r["clipgt_id"]: r for r in d["rollouts"]}


# ---------------------------------------------------------------- lists
def build_lists():
    L = O / "lists"
    L.mkdir(parents=True, exist_ok=True)
    marks = sorted(Path(p).name.split(".")[-3].split("_")[-1] for p in glob.glob(str(ROOT / ".done.MTGS_asset_navtest_assets_part*.tar.gz")))
    scenes = sorted(p.name for p in (ROOT / "navtest/assets").iterdir() if p.is_dir())
    if len(scenes) != 100 * len(marks):
        fail(f"{len(scenes)} scene dirs for {len(marks)} .done shards {marks}: an extraction is in progress or a shard is not 100 scenes")
    shard = {s: marks[i // 100] for i, s in enumerate(scenes)}   # shards are contiguous 100-scene blocks of the sorted names
    (L / "shards.tsv").write_text("".join(f"{s}\t{shard[s]}\n" for s in scenes))
    old = [s.strip() for s in (RA / "scenes_public_landed4.txt").read_text().split() if s.strip()]
    new = [s for s in scenes if s not in set(old)]
    ov = old[:: len(old) // 24][:24]
    w = lambda n, xs: (L / n).write_text("\n".join(xs) + "\n")
    w("all.txt", scenes), w("new.txt", new), w("overlap24.txt", ov)
    for i in range(3):
        w(f"chunk{i}.txt", scenes[i::3])
    pilot = [s.strip() for s in (RA / "scenes_public_pilot8.txt").read_text().split() if s.strip()]
    w("pilot8.txt", pilot)
    status(f"lists: {len(scenes)} scenes from shards {marks}; new {len(new)}, overlap check {len(ov)}, pilot {len(pilot)}")
    return {n: L / f"{n}.txt" for n in ["all", "new", "overlap24", "pilot8", "chunk0", "chunk1", "chunk2"]}


# ---------------------------------------------------------------- jobs
class Job:
    def __init__(self, name, drv, env, scenes, prio, vram=24):
        self.name, self.drv, self.env, self.scenes, self.prio, self.vram = name, drv, env, Path(scenes), prio, vram
        self.n = len(self.scenes.read_text().split())
        self.tries, self.id, self.D = 0, None, None
        self.t_seen = self.t_prog = self.last_n = None
        self.state = "new"        # new -> active -> done
        self.dirs = []             # finished run dirs (one per successful try, plus fill jobs)

    def submit(self):
        self.tries += 1
        if self.tries > MAX_TRIES:
            fail(f"{self.name}: {MAX_TRIES} tries exhausted")
        self.D = O / "runs" / self.name / time.strftime("%Y%m%d-%H%M%S")
        self.D.mkdir(parents=True)
        ram = int(10 + 0.12 * self.n)
        cmd = PY + ["submit", "--name", f"alpasim-c0b-{self.name}", "--vram", str(self.vram), "--cpu", "16", "--ram", str(ram),
                    "--timeout-h", "6", "--tries", "1", "--priority", str(self.prio), "--log-dir", str(self.D / "pool"), "--",
                    "env", *[f"{k}={v}" for k, v in self.env.items()], "bash", "experiments/alpasim/scripts/run.sh", str(self.D), self.drv,
                    "--scene-list", str(self.scenes), *OVERRIDES]
        out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        if out.returncode:
            fail(f"submit {self.name}: {out.stderr[-500:]}")
        self.id = out.stdout.split()[-1]
        self.state, self.t_seen, self.t_prog, self.last_n = "active", None, None, 0
        log(f"{self.name}: submitted {self.id} try {self.tries} -> {self.D}")

    def n_done(self):
        n = 0
        r = self.D / "rollouts"
        if r.is_dir():
            for s in os.scandir(r):
                if s.is_dir():
                    n += sum(1 for x in os.scandir(s.path) if x.is_dir() and (Path(x.path) / "_complete").exists())
        return n

    def cancel(self, why):
        log(f"{self.name}: {why}; cancelling {self.id}")
        subprocess.run(PY + ["cancel", self.id], cwd=REPO, capture_output=True)
        time.sleep(20)
        shutil.rmtree(self.D / "rollouts", ignore_errors=True)

    def poll(self):
        """Returns True when this job is finished and verified."""
        if self.state != "active":
            return self.state == "done"
        now = time.time()
        if (self.D / "pool/ERROR").exists():
            why = (self.D / "pool/ERROR").read_text()[:200].replace("\n", " ")
            log(f"{self.name}: pool ERROR {why}")
            shutil.rmtree(self.D / "rollouts", ignore_errors=True)
            self.submit()
            return False
        if (self.D / "pool/DONE").exists():
            return self.finish()
        if (self.D / "driver-tmp").exists():             # run.sh started
            n = self.n_done()
            if self.t_seen is None:
                self.t_seen = self.t_prog = now
            if n > self.last_n:
                self.last_n, self.t_prog = n, now
            elif now - self.t_prog > STALL_MIN * 60:
                self.cancel(f"stall: {n}/{self.n} rollouts, none new for {STALL_MIN:.0f} min")
                self.submit()
        return False

    def finish(self):
        try:
            R = load_scores(self.D)
        except Exception as e:
            log(f"{self.name}: no results-summary ({e!r})")
            self.submit()
            return False
        want = set(self.scenes.read_text().split())
        missing = sorted(want - set(R))
        for sid, r in R.items():       # keep the asl only where the rollout has a failure reason (the zero-score scenes)
            if not r.get("failure_reason"):
                for f in glob.glob(str(self.D / "rollouts" / sid / "*" / "rollout.asl")):
                    os.remove(f)
        self.dirs.append(self.D)
        if missing and self.tries < MAX_TRIES:
            log(f"{self.name}: {len(missing)} of {self.n} scenes missing, resubmitting those")
            fill = self.scenes.with_name(f"{self.name}.fill{self.tries}.txt")
            fill.write_text("\n".join(missing) + "\n")
            self.scenes, self.n = fill, len(missing)
            self.submit()
            return False
        if missing:
            fail(f"{self.name}: {len(missing)} scenes still missing after {self.tries} tries")
        self.state = "done"
        log(f"{self.name}: done, {len(R)} rollouts in {self.D}")
        return True


def driver_health(dirs):
    """Counts over driver-logs close records: (sessions, drive calls, inference errors, input errors)."""
    s = d = ie = ip = 0
    for D in dirs:
        for line in (D / "driver-logs/drive.jsonl").open():
            if '"close"' in line:
                r = json.loads(line)
                if r["kind"] == "close":
                    s += 1
                    d += r.get("drive", 0)
                    ie += r.get("inference_error", 0)
                    ip += r.get("input_error", 0)
    return s, d, ie, ip


def wait_all(jobs, tag, watch=()):
    """Poll until every job in `jobs` is done; the jobs in `watch` get the stall watchdog meanwhile."""
    while True:
        for j in watch:
            j.poll()
        done = [j.poll() for j in jobs]
        fg = free_gb()
        status(f"{tag}: " + ", ".join(f"{j.name} {j.n_done() if j.state == 'active' and j.D else (j.n if j.state == 'done' else 0)}/{j.n}"
                                       + ("" if j.state != "done" else " ok") + (f" t{j.tries}" if j.tries > 1 else "") for j in jobs)
               + f"; free {fg:.0f} GB")
        if fg < 150 and not (O / "DISK_LOW").exists():
            (O / "DISK_LOW").write_text(f"free {fg:.0f} GB\n")
            (O / "ERROR").write_text(f"{time.strftime('%F %T')} free disk {fg:.0f} GB < 150 GB; jobs left running, nothing new submitted\n")
        if all(done):
            return
        time.sleep(POLL_S)


def write_manifest(groups):
    (O / "manifest.json").write_text(json.dumps({k: [str(d) for d in v] for k, v in groups.items()}, indent=1))


def report():
    r = subprocess.run([sys.executable, str(REPO / "experiments/alpasim/scripts/c0b_report.py"), "--manifest", str(O / "manifest.json"),
                        "--shards", str(O / "lists/shards.tsv"), "--out", str(O / "report")], cwd=REPO, capture_output=True, text=True)
    log(f"report rc {r.returncode} {r.stderr[-300:]}")


def main():
    O.mkdir(parents=True, exist_ok=True)
    for f in ("DONE", "ERROR", "DISK_LOW"):
        (O / f).unlink(missing_ok=True)
    L = build_lists()
    old = {k: only_run_dir(v) for k, v in C0.items()}
    # ---- stage 1: pilots
    st1 = [Job("pilot-wajepa", "wajepa", {"WAJ_AMP": 0}, L["pilot8"], 20, 26), Job("pilot-ot0", "sh30", {"SH30_TAG": "OT30-F-s0"}, L["pilot8"], 20),
           Job("check-sh30", "sh30", {"SH30_TAG": "SH30-F-s0"}, L["overlap24"], 20), Job("check-ap2", "ap2", {"AP2_TAG": "AP2-AB-s0"}, L["overlap24"], 20)]
    for j in st1:
        j.submit()
    wait_all(st1, "stage1")
    for j in st1[:2]:
        s, d, ie, ip = driver_health(j.dirs)
        R = load_scores(j.dirs[-1])
        log(f"{j.name}: sessions {s}, drive calls {d}, inference errors {ie}, input errors {ip}, rollouts {len(R)}")
        if ie or ip or s != 8 or d < 8 * 3:
            fail(f"{j.name} pilot unhealthy: sessions {s} drive {d} errors {ie}/{ip}")
    reuse = {}
    for j, k in zip(st1[2:], ("sh30", "ap2")):
        A, B = load_scores(old[k]), load_scores(j.dirs[-1])
        bad = [s for s in B if abs(A[s]["score"] - B[s]["score"]) > 1e-9 or bool(A[s].get("failure_reason")) != bool(B[s].get("failure_reason"))]
        reuse[k] = not bad
        log(f"{k}: {len(B)} overlap scenes rerun, {len(bad)} differ from last night's run -> {'reuse the 400' if not bad else 'RERUN ALL'}")
        (O / f"overlap_check_{k}.json").write_text(json.dumps(dict(n=len(B), differ=bad, max_abs=max(abs(A[s]["score"] - B[s]["score"]) for s in B))))
    # ---- stage 3: everything
    full = []
    for k, tag in (("sh30", "SH30-F-s0"), ("ap2", "AP2-AB-s0")):
        full.append(Job(f"{k}-{'new' if reuse[k] else 'all'}", k, {"SH30_TAG" if k == "sh30" else "AP2_TAG": tag}, L["new" if reuse[k] else "all"], 15))
    for i in range(3):
        full.append(Job(f"wajepa-c{i}", "wajepa", {"WAJ_AMP": 0}, L[f"chunk{i}"], 10, 26))
    for s in (0, 1):
        for i in range(3):
            full.append(Job(f"ot{s}-c{i}", "sh30", {"SH30_TAG": f"OT30-F-s{s}"}, L[f"chunk{i}"], 5))
    for j in full:
        j.submit()
    G = lambda pre: [j for j in full if j.name.startswith(pre)]
    groups = {}
    def dirs(pre, extra=()):
        return [*extra, *[d for j in G(pre) for d in j.dirs]]
    # drivers 1-2 first so that the per-scene table can be published early
    wait_all(G("sh30") + G("ap2"), "stage3-ours", watch=full)
    groups["SH30-F-s0"] = dirs("sh30", [old["sh30"]] if reuse["sh30"] else []) + ([st1[2].dirs[-1]] if reuse["sh30"] else [])
    groups["AP2-AB-s0"] = dirs("ap2", [old["ap2"]] if reuse["ap2"] else []) + ([st1[3].dirs[-1]] if reuse["ap2"] else [])
    write_manifest(groups), report()
    status("SH30 and AP2 on the full list done; per-scene table in report/")
    wait_all(full, "stage3")
    groups["WA-JEPA (reference)"] = dirs("wajepa")
    groups["OT30-F-s0"] = dirs("ot0")
    groups["OT30-F-s1"] = dirs("ot1")
    write_manifest(groups), report()
    (O / "DONE").write_text(time.strftime("%F %T") + "\n")
    status("all done")


if __name__ == "__main__":
    main()
