#!/usr/bin/env python3
"""C0c: extend the C0b five-driver scores to the AlpaSim public shards that landed after C0b, as GPU-pool jobs.

  scripts/tmux_run.sh c0c python3 experiments/alpasim/scripts/c0c_chain.py <batch-name>        (box; stdlib only)

Scenes = every extracted scene not yet scored (not in the previous manifest); the simulator is deterministic and C0b checked
exact reproduction, so scored scenes are reused and only the new ones run. Drivers: SH30-F-s0, AP2-AB-s0, OT30-F-s0, OT30-F-s1
and WA-JEPA (reference row only, fp32). Each driver chunk is its own pool job, at most MAX_ACTIVE at a time (normal priority,
the training lane has the cards first). Same 20-minute stall watchdog and zero-score-only .asl pruning as c0b_chain.
Outputs in $DATA_DIR/runs/alpasim/c0c/<batch>/; the cumulative manifest is c0c/manifest.json (starts from c0b/manifest.json).
"""
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import c0b_chain as C

MAX_ACTIVE = 3
CHUNK = 100          # scenes per WA-JEPA / OT30 job


def main():
    batch = sys.argv[1]
    C.O = O = C.RA / "c0c" / batch
    O.mkdir(parents=True, exist_ok=True)
    for f in ("DONE", "ERROR", "DISK_LOW"):
        (O / f).unlink(missing_ok=True)
    cum = C.RA / "c0c/manifest.json"
    man = json.loads((cum if cum.exists() else C.RA / "c0b/manifest.json").read_text())
    prev = C.RA / "c0c" / "scored_scenes.txt"
    old_tsv = C.RA / "c0c/shards.tsv" if (C.RA / "c0c/shards.tsv").exists() else C.RA / "c0b/lists/shards.tsv"
    old_shard = dict(l.split("\t") for l in old_tsv.read_text().split("\n") if l)
    scored = set(prev.read_text().split()) if prev.exists() else set(old_shard)
    L = O / "lists"
    L.mkdir(exist_ok=True)
    marks = sorted(Path(p).name.split(".")[-3].split("_")[-1] for p in C.glob.glob(str(C.ROOT / ".done.MTGS_asset_navtest_assets_part*.tar.gz")))
    scenes = sorted(p.name for p in (C.ROOT / "navtest/assets").iterdir() if p.is_dir())
    if len(scenes) != 100 * len(marks):
        C.fail(f"{len(scenes)} scene dirs for {len(marks)} .done shards {marks}: an extraction is in progress")
    # old scenes keep their labels; new scenes are labelled by contiguous 100-scene blocks over the newly landed shards (inferred)
    new = [s for s in scenes if s not in scored]
    fresh = [m for m in marks if m not in set(old_shard.values())]
    shard = {**old_shard, **{s: fresh[i // 100] for i, s in enumerate(new) if i // 100 < len(fresh)}}
    (L / "shards.tsv").write_text("".join(f"{s}\t{shard[s]}\n" for s in scenes))
    shutil.copy(L / "shards.tsv", C.RA / "c0c/shards.tsv")
    C.status(f"batch {batch}: {len(scenes)} scenes from shards {marks}; {len(new)} new")
    if not new:
        C.fail("no new scenes")
    k = max(1, math.ceil(len(new) / CHUNK))
    def w(n, xs):
        (L / n).write_text("\n".join(xs) + "\n")
        return L / n
    jobs = [C.Job("sh30", "sh30", {"SH30_TAG": "SH30-F-s0"}, w("all.txt", new), 15),
            C.Job("ap2", "ap2", {"AP2_TAG": "AP2-AB-s0"}, L / "all.txt", 15)]
    for name, drv, env, prio, vram in (("wajepa", "wajepa", {"WAJ_AMP": 0}, 10, 26), ("ot0", "sh30", {"SH30_TAG": "OT30-F-s0"}, 5, 24),
                                       ("ot1", "sh30", {"SH30_TAG": "OT30-F-s1"}, 5, 24)):
        for i in range(k):
            jobs.append(C.Job(f"{name}-c{i}", drv, env, w(f"{name}-c{i}.txt", new[i::k]), prio, vram))
    # run with at most MAX_ACTIVE active jobs; poll() carries the stall watchdog
    queue = list(jobs)
    while True:
        act = [j for j in jobs if j.state == "active"]
        fg = C.free_gb()
        while queue and len(act) < MAX_ACTIVE and fg > 150:
            j = queue.pop(0)
            j.submit()
            act.append(j)
        for j in jobs:
            j.poll()
        fg = C.free_gb()
        C.status("c0c: " + ", ".join(f"{j.name} {j.n_done() if j.state == 'active' and j.D else (j.n if j.state == 'done' else 0)}/{j.n}"
                                     + (" ok" if j.state == "done" else "") + (f" t{j.tries}" if j.tries > 1 else "") for j in jobs) + f"; free {fg:.0f} GB")
        if fg < 150:
            (O / "DISK_LOW").write_text(f"free {fg:.0f} GB\n")
        if all(j.state == "done" for j in jobs):
            break
        time.sleep(C.POLL_S)
    G = lambda pre: [str(d) for j in jobs if j.name.startswith(pre) for d in j.dirs]
    for key, pre in (("SH30-F-s0", "sh30"), ("AP2-AB-s0", "ap2"), ("WA-JEPA (reference)", "wajepa"), ("OT30-F-s0", "ot0"), ("OT30-F-s1", "ot1")):
        man[key] = man[key] + G(pre)
    cum.write_text(json.dumps(man, indent=1))
    (C.RA / "c0c/scored_scenes.txt").write_text("\n".join(scenes) + "\n")
    r = subprocess.run([sys.executable, str(C.REPO / "experiments/alpasim/scripts/c0b_report.py"), "--manifest", str(cum), "--shards",
                        str(L / "shards.tsv"), "--out", str(O / "report")], cwd=C.REPO, capture_output=True, text=True)
    C.log(f"report rc {r.returncode} {r.stderr[-300:]}")
    (O / "DONE").write_text(time.strftime("%F %T") + "\n")
    C.status("all done")


if __name__ == "__main__":
    main()
