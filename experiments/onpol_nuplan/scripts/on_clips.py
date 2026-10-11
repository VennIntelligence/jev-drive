#!/usr/bin/env python3
"""ONPOL-NUPLAN clips (plans/2026-10-11-onpol-prereg.md, "Read", descriptive): COL1's rollout clip of every rollout of every fixed and broken
scene of an arm against the control (s0, s1). The clip code is experiments/alpasim/scripts/{col1_lead.py, col1_clip.py}, imported unedited
with their state directory pointed at $ON_CLIPS (default $DATA_DIR/runs/onpol_nuplan/clips); COL1's own directory is never touched.
on_clips.sh runs the steps in order.

  cases   [--stats experiments/onpol_nuplan/results/read/on_stats.json] [--arms ON-IMIT ON-SELF ON-BOTH]        (AlpaSim venv)
          the fixed / broken scenes (three zero kinds) of on_report.py -> cases.pkl (the rollout records from the runs' read_sim.pkl),
          cases.csv (col1_clip's case list: one row per rollout, id <scene token>_<tag>), rows.json (for the index).
          --test-run TAG=RUN_DIR [--limit N]: instead, every failed rollout of one run dir (the test on train-split scenes).
  msgs    (AlpaSim venv) col1_lead `msgs`: the logged driver-side messages of every case        maps: col1_lead `maps` (privileged, drawn only)
  replay  (envs/op-train, one GPU: a pool job) col1_lead `replay`: every decision again through the real driver class -> model frames, lead head
  clip    [--jobs N] [--webp] (AlpaSim venv) col1_clip.render per case in parallel -> gif/<id>.gif (<= 1.4 MiB); --webp adds a 480-px
          webp/<id>.webp review copy (the repo takes clips under ~500 KB only: the index lists every size, nothing is copied by this script)
  index   index.md: scene, arm, fixed / broken, the zero kind per seed of both sides, the clips         (copy to results/read/clips.md)
  prune   [--stage held] after ON_KEEP_ASL=all: delete rollout.asl of the rollouts without a failure reason under read/<stage> (ot2_loop's rule)

One clip shows one rollout (col1_clip.py's layout: bird's-eye view with the served plan, the two model frames, speed / gap / lead prob); the
tool has no side-by-side layout, so the two sides of a scene are separate clips named by tag. A clip needs the rollout's rollout.asl:
ot2_loop keeps it for failed rollouts, so by default a fixed scene has the control's two clips and a broken scene the arm's two; with
ON_KEEP_ASL=all in on_read.sh every rollout of the scene has one. The replay is open loop through the eager driver on the logged messages:
the green plan in the bird's-eye view is the served one, the lead panels are the replay's.
"""
import argparse
import csv
import glob
import json
import multiprocessing
import os
import pickle
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ALP = REPO / "experiments/alpasim/scripts"
sys.path[:0] = [str(ALP), str(ALP.parent / "lib"), str(REPO)]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
W = DATA / "runs/onpol_nuplan"
O = Path(os.environ.get("ON_CLIPS") or W / "clips")
RES = REPO / "experiments/onpol_nuplan/results/read"
C2 = "control (s0, s1)"
SHORT = {"collision_at_fault": "collision", "offroad": "offroad", "left_corridor_laterally": "corridor", "other": "other zero", "": "-"}
_CTX = None


class _Sh30(dict):
    def __missing__(self, k):
        return "sh30"


def col1():
    """col1_lead / col1_clip with their state under O."""
    import col1_lead as LE
    LE.O, LE.LD, LE.DRV = O, O / "lead", _Sh30()
    (O / "lead").mkdir(parents=True, exist_ok=True)
    return LE


def cmd_cases(a):
    sims, C, rows, cases = {}, {}, [], {}

    def rollout(tag, scene, dirs):
        """The run dir of `dirs` that still has the scene's rollout.asl -> its record into C; False when no log is left."""
        run = next((d for d in dirs if glob.glob(f"{d}/rollouts/{scene}/*/rollout.asl")), None)
        if run is None:
            return False
        if run not in sims:
            sims[run] = pickle.load(open(f"{run}/read_sim.pkl", "rb"))
        C[tag, scene] = sims[run][scene] | {"run": run}
        return True
    if a.test_run:
        tag, run = a.test_run.split("=", 1)
        S = json.loads(Path(run, "aggregate/results-summary.json").read_text())["rollouts"]
        for r in sorted((r for r in S if r.get("failure_reason")), key=lambda r: r["clipgt_id"])[:a.limit or None]:
            s = r["clipgt_id"]
            kind = next((v for k, v in SHORT.items() if k and (r.get("score_metrics") or {}).get(k)), "other zero")
            ok = rollout(tag, s, [run])
            rows.append(dict(arm="TEST", what="failed", scene=s, turn="", tags=[dict(tag=tag, side="arm", kind=kind, clip=f"{s.rsplit('-', 1)[1][:8]}_{tag}" if ok else None)]))
    else:
        st = json.loads(Path(a.stats).read_text())
        assert not st.get("standin"), "the stand-in stats describe stored reads, not this lane's run dirs"
        T = {r["scene"]: r for r in csv.DictReader(Path(a.stats).with_name("on_per_scene.csv").open())}
        for f in st["flips"]:
            if f["ref"] != C2 or f["kinds"] != "zeros" or (a.arms and f["arm"] not in a.arms):
                continue
            for what in ("fixed", "broken"):
                for s in f[what + "_scenes"]:
                    tg = []
                    for side, g in (("control", C2), ("arm", f["arm"])):
                        for t in st["tags"][g]:
                            ok = rollout(t, s, st["sources"][t])
                            tg.append(dict(tag=t, side=side, kind=SHORT.get(T[s][f"{t}|zero_class"], "-"), clip=f"{s.rsplit('-', 1)[1][:8]}_{t}" if ok else None))
                    rows.append(dict(arm=f["arm"], what=what, scene=s, turn=T[s]["turn4s_deg"], tags=tg))
    for r in rows:
        for t in r["tags"]:
            if t["clip"]:
                cases[t["clip"]] = dict(id=t["clip"], set=t["tag"], scene=r["scene"], obj="", kind=t["kind"], origin="ONPOL-NUPLAN read", look="")
    O.mkdir(parents=True, exist_ok=True)
    txt = "id,set,scene,obj,kind,origin,look\n" + "".join(",".join(c.values()) + "\n" for c in cases.values())
    if (O / "cases.csv").exists() and (O / "cases.csv").read_text() != txt:              # another case list: the replay of the old one is stale
        shutil.rmtree(O / "lead/replay", ignore_errors=True)
        for f in ("lead/spec.json", "lead/REPLAY_DONE"):
            (O / f).unlink(missing_ok=True)
    (O / "cases.csv").write_text(txt)
    pickle.dump(C, open(O / "cases.pkl", "wb"), protocol=4)
    (O / "clip_scenes.txt").write_text("".join(f"{c['set']} {c['scene']}\n" for c in cases.values()))
    (O / "rows.json").write_text(json.dumps(rows, indent=1))
    print(f"cases: {len(rows)} scene rows, {len(cases)} rollouts with a log, {sum(t['clip'] is None for r in rows for t in r['tags'])} without -> {O}")


def cmd_msgs(a):
    col1().cmd_msgs(argparse.Namespace(frames=None, jobs=a.jobs))


def cmd_maps(a):
    col1().cmd_maps(argparse.Namespace(frames=str(O / "clip_scenes.txt")))


def cmd_replay(a):
    col1().cmd_replay(argparse.Namespace(force=False))


def _render(c):
    C, cases, maps, rep, a = _CTX
    try:
        C.render(c, cases[c["set"], c["scene"]], rep[c["set"]][c["scene"]], C.L.road(maps[c["scene"]]) if c["scene"] in maps else None, str(O / "gif"), a.max_mb)
        if a.webp:
            from PIL import Image, ImageSequence
            g = Image.open(O / "gif" / f"{c['id']}.gif")
            fr = [(f.convert("RGB").resize((480, round(480 * g.height / g.width)), Image.LANCZOS), f.info.get("duration", 250)) for f in ImageSequence.Iterator(g)]
            fr[0][0].save(O / "webp" / f"{c['id']}.webp", save_all=True, append_images=[x for x, _ in fr[1:]], duration=[d for _, d in fr], loop=0, quality=30, method=4)
        return None
    except Exception as e:                                           # noqa: BLE001  one bad rollout must not stop the others
        return f"{c['id']}: {e!r}"


def cmd_clip(a):
    global _CTX
    col1()
    import col1_clip as C
    C.O, C.LD = O, O / "lead"
    rows = list(csv.DictReader((O / "cases.csv").open()))
    cases, maps = pickle.load(open(O / "cases.pkl", "rb")), (pickle.load(open(O / "lead/maps.pkl", "rb")) if (O / "lead/maps.pkl").exists() else {})
    rep = {g: pickle.load(open(O / "lead/replay" / f"{g}.pkl", "rb")) for g in sorted({r["set"] for r in rows})}
    for d in ("gif", "webp"):
        (O / d).mkdir(exist_ok=True)
    todo = [r for r in rows if a.force or not (O / "gif" / f"{r['id']}.gif").exists()]
    _CTX = (C, cases, maps, rep, a)
    with ProcessPoolExecutor(a.jobs, mp_context=multiprocessing.get_context("fork")) as ex:
        bad = [e for e in ex.map(_render, todo) if e]
    print(f"clip: {len(todo) - len(bad)} rendered, {len(rows) - len(todo)} already there, {len(bad)} failed", *bad, sep="\n")
    sys.exit(2 if bad else 0)


def cmd_index(a):
    rows = json.loads((O / "rows.json").read_text())

    def clip(t):
        if not t["clip"]:
            return "no log"
        fs = [(d, O / d / f"{t['clip']}.{d}") for d in ("gif", "webp")]
        return " ".join(f"`{d}/{f.name}` ({f.stat().st_size / 1e6:.2f} MB)" for d, f in fs if f.exists()) or "not rendered"
    L = ["# ONPOL-NUPLAN: rollout clips of the fixed and broken scenes (generated by scripts/on_clips.py index; do not edit)", "",
         f"Clips on the box under `{O}/`. One clip = one rollout (COL1's layout: bird's-eye view with the served plan, the two model frames of the newest slot, speed / gap / "
         "lead prob). Fixed = zero (collision, offroad or corridor) in both control seeds (s0, s1) and in neither seed of the arm; broken = the reverse. "
         "`no log` = the rollout passed and its rollout.asl was pruned (ot2_loop keeps failed rollouts; ON_KEEP_ASL=all keeps all). Map, actors and the logged path in "
         "the clips are privileged and drawn for the reader only.", "",
         "| arm | fixed / broken | scene | turn 4 s (deg) | control s0 / s1 | arm s0 / s1 | clips (tag: file, size) |", "|:--|:--|:--|--:|:--|:--|:--|"]
    for r in rows:
        side = lambda s: " / ".join(t["kind"] for t in r["tags"] if t["side"] == s) or "-"  # noqa: E731
        L.append(f"| {r['arm']} | {r['what']} | `{r['scene']}` | {r['turn']} | {side('control')} | {side('arm')} | " + "; ".join(f"{t['tag']}: {clip(t)}" for t in r["tags"]) + " |")
    n = {w: sum(r["what"] == w for r in rows) for w in ("fixed", "broken", "failed")}
    small = sorted(f.name for f in (O / "webp").glob("*.webp") if f.stat().st_size < 5e5) if (O / "webp").exists() else []
    L += ["", f"{len(rows)} rows ({', '.join(f'{v} {k}' for k, v in n.items() if v)}); {len(list((O / 'gif').glob('*.gif')))} GIFs, "
          f"{len(small)} WebP copies under 500 KB (the size the repo takes)."]
    (O / "index.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def cmd_prune(a):
    n = b = 0
    for dirs in json.loads((W / "read" / a.stage / "manifest.json").read_text()).values():
        for d in dirs:
            for r in json.loads(Path(d, "aggregate/results-summary.json").read_text())["rollouts"]:
                if not r.get("failure_reason"):
                    for f in glob.glob(f"{d}/rollouts/{r['clipgt_id']}/*/rollout.asl"):
                        n, b = n + 1, b + os.path.getsize(f)
                        os.remove(f)
    print(f"prune: {n} rollout.asl of passed rollouts removed, {b / 1e9:.1f} GB")


def main():
    from jevdrive.common import n_cpus
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n, f in (("cases", cmd_cases), ("msgs", cmd_msgs), ("maps", cmd_maps), ("replay", cmd_replay), ("clip", cmd_clip), ("index", cmd_index), ("prune", cmd_prune)):
        p = sub.add_parser(n)
        p.set_defaults(fn=f)
        p.add_argument("--jobs", type=int, default=max(4, min(32, n_cpus() // 4)))
        if n == "cases":
            p.add_argument("--stats", default=str(RES / "on_stats.json")), p.add_argument("--arms", nargs="*", default=["ON-IMIT", "ON-SELF", "ON-BOTH"])
            p.add_argument("--test-run"), p.add_argument("--limit", type=int, default=0)
        if n == "prune":
            p.add_argument("--stage", default="held", help="stage of on_read.sh whose run dirs are pruned")
        if n == "clip":
            p.add_argument("--webp", action="store_true"), p.add_argument("--force", action="store_true"), p.add_argument("--max-mb", type=float, default=1.4)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
