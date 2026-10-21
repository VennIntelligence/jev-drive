#!/usr/bin/env python3
"""LOSS-VIDEOS: one or two representative clips per failure class of the LOSS-MAP (loss_map.py; R6DAG on PAI 281, ACC + GAP), each next to a
contrast read (NTV0 or a native read) where one of ours solves the scene.

Sources of a clip side (CLIPS below):
  stored  the stored 281 read's own rollout: zero-score rollouts keep rollout.asl, rendered offline on CPU with AlpaSim's eval video
          (eval.main on a copy of the asl, the chunk's own sceneset usdz for the map, video.render_video=true); bit-identical rollout
  rerun   a re-read of the scene on the GPU pool (pai_box.sh, the stored read's serve, KEEP_VIDEO=1), used for partial-score rollouts and
          contrasts that kept no asl; checked against the stored score in collect
  r5      the NTV0 re-read video of research/round5/offroad_videos ($DATA_DIR/runs/round5/offroad_videos/videos), already checked there

  asl      (box, pool CPU job; AlpaSim venv)  stored sides -> $O/videos/<s8>_<tag>.mp4
  launch   (box)  pool jobs: the asl job, one pai_box.sh submit per rerun group (PRIO 5), one collect job --after all
  wait     (box)  blocks until $O/DONE or a failed / cancelled job of this lane (one run_in_background waiter)
  collect  (box, AlpaSim venv)  $O/facts.json: per side score (re-read and stored), event onset, 10 Hz ego speed on the video clock
  page     (Mac, .venv: cv2)  $O/{facts.json, videos/} -> research/round7/loss_diag/media/ clips + index.html + manifest.json
"""
from __future__ import annotations

import argparse
import csv
import glob
import html
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
REPO = Path(__file__).resolve().parents[3]
O = DATA / "runs/round7/loss_videos"
AP = DATA / "third_party/alpasim/.venv/bin/python"
FFMPEG = DATA / "third_party/alpasim/.venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
STORED = {"R6DAG-L18-s0": DATA / "runs/round6/dag/pai281/R6DAG-L18-s0-ag", "R6DAG-L18-s1": DATA / "runs/round6/dag/pai281/R6DAG-L18-s1-ag"}
R5 = DATA / "runs/round5/offroad_videos/videos"
AG = "JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7"
NATIVE = "JEV_VCONT=1.0 JEV_LEAD=1"
WIZ = "controller.gains.rel_front_steering_angle_weight=10"
RERUN = {"dag1-ag": ("R6DAG-L18-s1", AG), "ntv0-ag": ("NTV0-L18-s0", AG), "ntv0-native": ("NTV0-L18-s0", NATIVE)}
PRIO = 5
SHORT = {"R6DAG-L18-s1": "R6DAG-s1", "R6DAG-L18-s0": "R6DAG-s0", "NTV0-L18-s0": "NTV0"}

# (class key, scene8, sides); side = (tag, source, serve label). Order = page order (table rank), 1-2 clips per class.
CLIPS = [
    ("off-turn-in", "ea468790", [("R6DAG-L18-s1", "stored", "ag"), ("NTV0-L18-s0", "r5", "ag")]),
    ("off-turn-in", "b680a689", [("R6DAG-L18-s1", "stored", "ag"), ("NTV0-L18-s0", "r5", "ag")]),
    ("slow", "bbc53aa8", [("R6DAG-L18-s1", "rerun:dag1-ag", "ag"), ("NTV0-L18-s0", "r5", "ag")]),
    ("rear-move", "d90fe123", [("R6DAG-L18-s1", "rerun:dag1-ag", "ag"), ("NTV0-L18-s0", "rerun:ntv0-ag", "ag")]),
    ("drift", "4cb05d65", [("R6DAG-L18-s1", "stored", "ag"), ("NTV0-L18-s0", "rerun:ntv0-ag", "ag")]),
    ("overshoot", "127c4f85", [("R6DAG-L18-s1", "stored", "ag")]),
    ("overshoot", "47db2eb6", [("R6DAG-L18-s1", "stored", "ag"), ("NTV0-L18-s0", "rerun:ntv0-native", "native")]),
    ("rear-launch", "cbb22323", [("R6DAG-L18-s1", "rerun:dag1-ag", "ag"), ("NTV0-L18-s0", "rerun:ntv0-ag", "ag")]),
    ("off-straight", "4f7cb072", [("R6DAG-L18-s1", "stored", "ag")]),
    ("off-junc-straight", "a5599793", [("R6DAG-L18-s1", "stored", "ag")]),
    ("off-turn-out", "7381c9fa", [("R6DAG-L18-s1", "stored", "ag")]),
    ("coll-side", "268949d2", [("R6DAG-L18-s1", "stored", "ag"), ("NTV0-L18-s0", "rerun:ntv0-ag", "ag")]),
    ("coll-front", "3ce13eed", [("R6DAG-L18-s1", "stored", "ag")]),
    ("route-turn", "302c5c99", [("R6DAG-L18-s1", "stored", "ag")]),
]


def scene_ids():
    """scene8 -> full clipgt id, from the stored facts (box) or the loss_map.csv (Mac)."""
    f = DATA / "runs/pai_fix/facts/R6DAG-L18-s1-281ag.csv"
    src = f if f.exists() else REPO / "experiments/round7/results/loss_map.csv"
    return {r["scene"][7:15]: r["scene"] for r in csv.DictReader(open(src))}


def stored_rollout(tag, scene):
    d = glob.glob(f"{STORED[tag]}/runs/*/sim/rollouts/{scene}/*/")
    return Path(d[0]) if d else None


# ---------------------------------------------------------------- box
def cmd_asl(a):
    import yaml
    ids = scene_ids()
    (O / "videos").mkdir(parents=True, exist_ok=True)
    (O / "bin").mkdir(exist_ok=True)
    if not (O / "bin/ffmpeg").exists():
        os.symlink(FFMPEG, O / "bin/ffmpeg")
    work = {}
    for _, s8, sides in CLIPS:
        for tag, src, _ in sides:
            if src != "stored":
                continue
            r = stored_rollout(tag, ids[s8])
            if r is None or not (r / "rollout.asl").exists():
                sys.exit(f"no stored asl for {s8} {tag}")
            sim = r.parents[2]
            dst = O / "asl" / tag / sim.parent.name
            work.setdefault(dst, sim)
            rd = dst / "rollouts" / r.parent.name / r.name
            rd.mkdir(parents=True, exist_ok=True)
            for f in ("rollout.asl", "_complete"):
                if not (rd / f).exists():
                    shutil.copyfile(r / f, rd / f)
    for dst, sim in work.items():
        for y in sim.glob("*.yaml"):
            shutil.copyfile(y, dst / y.name)
        c = yaml.safe_load(open(dst / "eval-config.yaml"))
        c["video"]["render_video"], c["num_processes"] = True, 8
        yaml.safe_dump(c, open(dst / "eval-config.yaml", "w"))
        usdz = yaml.safe_load(open(sim / "generated-user-config-0.yaml"))["scene_provider"]["usdz"]["data_dir"]
        env = dict(os.environ, PATH=f"{O / 'bin'}:{os.environ['PATH']}")
        rc = subprocess.run([str(AP), "-m", "eval.main", "--asl_search_glob", f"{dst}/rollouts/**/rollout.asl", "--config_path", str(dst / "eval-config.yaml"),
                             "--usdz_glob", f"{usdz}/*.usdz", "--log-level", "WARNING"], cwd=DATA / "third_party/alpasim/src/eval", env=env).returncode
        if rc:
            sys.exit(f"eval.main failed ({rc}) on {dst}")
        for mp4 in dst.glob("rollouts/*/*/*camera_front_wide*.mp4"):
            shutil.copyfile(mp4, O / "videos" / f"{mp4.parents[1].name[7:15]}_{dst.parent.name}.mp4")
    print(len(list((O / "videos").glob("*.mp4"))), "videos")


def sub(name, *args):
    cmd = [str(REPO / ".venv/bin/python"), "-m", "jevdrive.cl", "submit", "--owner", "round7-lossvid", "--name", name, "--tries", "1",
           "--priority", str(PRIO), "--no-check", "--log-dir", str(O / "pool" / name), *args]
    out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    jid = out.stdout.split()[-1] if out.returncode == 0 and out.stdout.split() else None
    if not jid:
        sys.exit(f"submit {name} failed: {out.stdout} {out.stderr}")
    return jid


def cmd_launch(a):
    if (O / "jobs.txt").exists() and (O / "jobs.txt").read_text().strip():
        sys.exit(f"already submitted ({O / 'jobs.txt'})")
    ids = scene_ids()
    (O / "lists").mkdir(parents=True, exist_ok=True)
    jobs = [(sub("lossvid-asl", "--vram", "0.1", "--cpu", "8", "--ram", "24", "--timeout-h", "2", "--", str(AP), str(Path(__file__).resolve()), "asl"),
             "lossvid-asl")]
    for g, (tag, serve) in RERUN.items():
        sc = sorted({ids[s8] for _, s8, sides in CLIPS for t, src, _ in sides if src == f"rerun:{g}"})
        (O / "lists" / f"{g}.txt").write_text("\n".join(sc) + "\n")
        env = dict(os.environ, SERVE=serve, WIZ=WIZ, CONC="4", ALPASIM_FAST="1", KEEP_VIDEO="1", KEEP_ASL="1", PRIO=str(PRIO))
        out = subprocess.run(["bash", str(REPO / "experiments/alpasim/scripts/pai_box.sh"), "submit", str(O / "reads" / g), tag, str(O / "lists" / f"{g}.txt")],
                             cwd=REPO, env=env, capture_output=True, text=True)
        got = [ln.split()[:2] for ln in out.stdout.splitlines() if ln.split()[1:2] and ln.split()[1].startswith("pai-")]
        if out.returncode or not got:
            sys.exit(f"pai_box submit {g} failed: {out.stdout} {out.stderr}")
        jobs += [(j, n) for j, n in got]
    c = sub("lossvid-collect", "--after", ",".join(j for j, _ in jobs), "--vram", "0.1", "--cpu", "2", "--ram", "8", "--timeout-h", "1", "--",
            str(AP), str(Path(__file__).resolve()), "collect")
    jobs.append((c, "lossvid-collect"))
    (O / "jobs.txt").write_text("".join(f"{j} {n}\n" for j, n in jobs))
    code = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    with open(O / "stages.txt", "a") as fh:
        fh.write(f"{time.strftime('%F %T')} launch (code {code}): {' '.join(j for j, _ in jobs)}\n")
    print(*(f"{j} {n}" for j, n in jobs), sep="\n")


def cmd_wait(a):
    sys.path.insert(0, str(REPO))
    from jevdrive.cl import pool as P
    while not (O / "DONE").exists():
        last = dict(reversed(ln.split()[:2]) for ln in open(O / "jobs.txt") if ln.split())
        jobs = P.snapshot()[0]["jobs"]
        bad = [f"{n} {i} {jobs[i]['state']}" for n, i in last.items() if jobs.get(i, {}).get("state") in ("failed", "cancelled")]
        if bad:
            sys.exit("failed:\n  " + "\n  ".join(bad))
        time.sleep(300)
    print("DONE", (O / "DONE").read_text().strip())


def ctrl(sim, rid):
    import numpy as np
    rows = list(csv.reader(open(Path(sim) / f"controller/alpasim_controller_{rid}.csv")))
    h = {k: i for i, k in enumerate(rows[0])}
    A = np.array(rows[1:], float)
    return A[:, h["timestamp_us"]], np.hypot(A[:, h["vx"]], A[:, h["vy"]])


def side_facts(sim, scene, rid, score):
    import numpy as np
    import polars as pl
    T, v = ctrl(sim, rid)
    t0 = T[0] - 100_000
    m = pl.read_parquet(Path(sim) / "rollouts" / scene / rid / "metrics.parquet")
    on = {}
    for k in ("offroad", "collision_at_fault", "collision_rear", "collision_any", "wrong_lane"):
        x = m.filter((pl.col("name") == k) & (pl.col("values") > 0))["timestamps_us"].to_list()
        if x:
            on[k] = round((min(x) - t0) / 1e6, 2)
    return dict(rid=rid, score=score, on=on, t=((T - t0) / 1e6).round(2).tolist(), v=np.round(v, 2).tolist())


def cmd_collect(a):
    ids = scene_ids()
    F = {t: {r["scene"]: r for r in csv.DictReader(open(DATA / f"runs/pai_fix/facts/{t}-281ag.csv"))} for t in ("R6DAG-L18-s1", "R6DAG-L18-s0", "NTV0-L18-s0")}
    summ = {}
    for root in [*STORED.values(), *(O / "reads" / g for g in RERUN)]:
        for p in glob.glob(f"{root}/runs/*/sim/aggregate/results-summary.json"):
            sim = Path(p).parents[1]
            key = "stored:" + root.name[:-3] if root in STORED.values() else "rerun:" + root.name
            for r in json.load(open(p))["rollouts"]:
                summ[(key, r["clipgt_id"])] = (sim, r["rollout_id"], float(r.get("score") or 0.0))
    out, bad = {}, []
    for cls, s8, sides in CLIPS:
        s = ids[s8]
        for tag, src, serve in sides:
            k = f"{s8}|{tag}|{src}"
            if src == "stored":
                sim, rid, sc = summ[("stored:" + tag, s)]
                x = side_facts(sim, s, rid, sc)
                x["video"] = f"{s8}_{tag}.mp4" if (O / "videos" / f"{s8}_{tag}.mp4").exists() else None
            elif src == "r5":
                src5 = json.load(open(DATA / "runs/round5/offroad_videos/facts.json"))["rollouts"][f"{s}|{tag}"]
                x = dict(rid=src5["rid"], score=src5["score"], on={"offroad": src5["t_off"]} if src5["t_off"] is not None else {}, t=src5["t"], v=src5["v"])
                dst = O / "videos" / f"{s8}_{tag}_r5.mp4"
                shutil.copyfile(R5 / src5["video"], dst)
                x["video"] = dst.name
            else:
                g = src.split(":")[1]
                sim, rid, sc = summ[("rerun:" + g, s)]
                x = side_facts(sim, s, rid, sc)
                mp4 = sorted((sim / "rollouts" / s / rid).glob("*camera_front_wide*.mp4"))
                dst = O / "videos" / f"{s8}_{tag}_{g}.mp4"
                if mp4:
                    shutil.copyfile(mp4[0], dst)
                x["video"] = dst.name if mp4 else None
            st = F.get(tag, {}).get(s) if serve == "ag" else None
            x.update(stored=float(st["score"]) if st else None, fclass=st["fclass"] if st else None, primary=st["primary"] if st else None,
                     t_ev=float(st["t_ev_s"]) if st and st["t_ev_s"] else None, rear_t=float(st["rear_t_s"]) if st and st["rear_t_s"] else None,
                     tag=tag, src=src, serve=serve, scene=s, cls=cls)
            if x["stored"] is not None and abs(x["score"] - x["stored"]) > 1e-3:
                bad.append(k)
            out[k] = x
    (O / "facts.json").write_text(json.dumps(dict(sides=out, mismatch=bad), indent=0))
    miss = [k for k, x in out.items() if not x["video"]]
    print(len(out), "sides,", len(miss), "without video", miss, ";", len(bad), "score mismatches", bad)
    (O / "DONE").write_text(time.strftime("%F %T") + f" sides {len(out)} novideo {len(miss)} mismatch {len(bad)}\n")


# ---------------------------------------------------------------- Mac
W1, HD, BV, CH = 640, 64, 576, 360               # one side: width, header, bird's-eye panel (square, centred), camera height
BEV, CAM = (slice(30, 504), slice(62, 536)), (slice(517, 992), slice(29, 872))   # panels of the 900 x 1000 DEFAULT eval frame
WHY = {"offroad": "offroad", "collision_at_fault": "collision", "collision_rear": "rear-ended"}


def speed(f):
    """Fast-forward factor (jtn_diag.speed): ~30 fps, never above 3x, at least ~6 s on screen, never below 2x."""
    try:
        r, d = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate:format=duration",
                              "-of", "csv=p=0", str(f)], capture_output=True, text=True).stdout.split()
        n, m = r.split("/")
        fps, dur = float(n) / float(m), float(d)
    except (ValueError, OSError):
        return 2.0
    return round(max(2.0, min(3.0, 30.0 / fps, dur / 6.0)), 2)


def frames(f):
    import cv2
    cap, fr = cv2.VideoCapture(str(f)), []
    while True:
        ok, im = cap.read()
        if not ok:
            return fr, cap.get(cv2.CAP_PROP_FPS) or 10.0
        fr.append(im)


def t_event(x):
    """Event on the rollout clock: offroad / at-fault collision onset, else the facts' event (overshoot: corridor at route end), else rear hit."""
    for k in ("offroad", "collision_at_fault"):
        if k in x["on"]:
            return x["on"][k], WHY[k]
    if x.get("t_ev") is not None:
        return x["t_ev"], x.get("fclass") or "zero"
    if "collision_rear" in x["on"]:
        return x["on"]["collision_rear"], "rear-ended"
    return None, ""


def clip(sides, src, dst, t_ev, pre=8.0, post=3.0):
    """Sides next to each other, [t_ev - pre, t_ev + post] on the rollout clock; header: read, serve, score, t, ego speed, event countdown."""
    import cv2
    import numpy as np
    fr = [frames(src / x["video"]) for x in sides]
    if not all(f for f, _ in fr):
        sys.exit(f"no frames in {[x['video'] for x in sides]}")
    fps = fr[0][1]
    n = min(len(f) for f, _ in fr)
    i0, i1 = max(0, int(round((t_ev - pre) * fps))), min(n, int(round((t_ev + post) * fps)) + 1)
    W, H = len(sides) * W1 + 8 * (len(sides) - 1), HD + BV + CH
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{fps:g}", "-i", "-",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "27", "-preset", "slow", "-movflags", "+faststart", str(dst)],
                         stdin=subprocess.PIPE)
    ev = [t_event(x) for x in sides]
    for i in range(i0, i1):
        ti = i / fps
        im = np.full((H, W, 3), 255, np.uint8)
        for j, x in enumerate(sides):
            X0 = j * (W1 + 8)
            f = fr[j][0][i]
            im[HD:HD + BV, X0 + (W1 - BV) // 2:X0 + (W1 + BV) // 2] = cv2.resize(f[BEV], (BV, BV), interpolation=cv2.INTER_AREA)
            im[HD + BV:, X0:X0 + W1] = cv2.resize(f[CAM], (W1, CH), interpolation=cv2.INTER_AREA)
            te, why = ev[j]
            hit = te is not None and ti >= te and x["score"] < 0.98
            cv2.rectangle(im, (X0, 0), (X0 + W1, HD - 4), (0, 0, 200) if hit else (40, 40, 40), -1)
            lab = SHORT.get(x["tag"], x["tag"]) + ("" if x["serve"] == "ag" else " native")
            cv2.putText(im, f"{lab}  score {x['score']:.2f}" + (f" ({why})" if x["score"] < 0.98 and why else ""), (X0 + 8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
            v = float(np.interp(ti, x["t"], x["v"]))
            tail = f"  {why.upper()}" if hit else (f"  {why} in {te - ti:.1f} s" if te is not None and x["score"] < 0.98 else "")
            cv2.putText(im, f"t {ti:5.1f} s   v {v:4.1f} m/s{tail}", (X0 + 8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        p.stdin.write(im.tobytes())
    p.stdin.close(), p.wait()
    return dict(t0=round(i0 / fps, 2), t1=round((i1 - 1) / fps, 2), fps=fps)


def cmd_page(a):
    import numpy as np
    sys.path.insert(0, str(Path(__file__).parent))
    import loss_page
    D = Path(a.dir)
    M, src = D / "media", D / "media/src"
    FX = json.loads((src / "facts.json").read_text())
    S = FX["sides"]
    LM = json.loads((REPO / "experiments/round7/results/loss_map.json").read_text())
    clips = {}
    for cls, s8, sides in CLIPS:
        xs = [S[f"{s8}|{t}|{sr}"] for t, sr, _ in sides]
        if not all(x["video"] for x in xs):
            sys.exit(f"no video for {s8}")
        te = t_event(xs[0])[0] or next((t_event(x)[0] for x in xs if t_event(x)[0] is not None), 10.0)
        dst = M / f"{s8}.mp4"
        cj = json.loads((M / "clips.json").read_text()) if (M / "clips.json").exists() else {}
        w = clip(xs, src / "videos", dst, te) if a.force or not dst.exists() or s8 not in cj else cj[s8]
        cj[s8] = w
        (M / "clips.json").write_text(json.dumps(cj))
        clips[s8] = dict(cls=cls, file=f"media/{dst.name}", mb=round(dst.stat().st_size / 1e6, 2), rate=speed(dst), t_ev=round(te, 2), clip=w,
                         n=len(xs), sides=[dict(tag=x["tag"], serve=x["serve"], src=x["src"], rerun=round(x["score"], 3), stored=x["stored"],
                                                v_ev=round(float(np.interp(te, x["t"], x["v"])), 2), event=t_event(x)) for x in xs])
        print(s8, cls, [(x["tag"], x["serve"], round(x["score"], 2), x["stored"]) for x in xs], "t_ev", round(te, 1), "rate", clips[s8]["rate"], flush=True)
    (D / "manifest.json").write_text(json.dumps(dict(clips=[dict(scene8=k, **v) for k, v in clips.items()], rerun_mismatch=FX["mismatch"]),
                                                indent=1, ensure_ascii=False))
    (D / "index.html").write_text(loss_page.render(LM, clips, FX["mismatch"]))
    print(len(clips), "clips,", round(sum(c["mb"] for c in clips.values()), 1), "MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for c in ("asl", "launch", "wait", "collect"):
        sp.add_parser(c)
    p = sp.add_parser("page")
    p.add_argument("dir"), p.add_argument("--force", action="store_true")
    a = ap.parse_args()
    dict(asl=cmd_asl, launch=cmd_launch, wait=cmd_wait, collect=cmd_collect, page=cmd_page)[a.cmd](a)
