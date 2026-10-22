#!/usr/bin/env python3
"""Scene-to-scene gap of a resident human stack (README "Scene-to-scene gap"): from one scene's last drive step to the next scene's
first frame and hand-off hold, split by phase, from the logs the stack already writes (stdlib only, runs on the box).

    python3 experiments/alpasim_human/scripts/gap_probe.py <resident dir> [...] [--json out.json]

Sources: feed/res/*.json (t_sent, t_done per request), driver/drive.jsonl (drive records: wall t at return, started; close records:
wait_s, wall_per_sim), compose.log (runtime: "Requesting available cameras for scene_id=" = runtime-side load done, "Session
STARTING", "Simulation ended", "Session COMPLETED"; times are UTC clock times, matched to the nearest epoch).
Per pair (A, B) of consecutive requests (B sent after A was sent):
  metrics    A's last drive return -> A "Session COMPLETED" (sim end event, AlpaSim's in-rollout metrics / asl write)
  dispatch   A completed -> B request sent to the runtime (feed result, loop, feed); negative = B was sent before A completed
  rt_load    B sent -> runtime asks the renderer for cameras (artifact, map, traffic: runtime side)
  nre_load   -> B "Session STARTING" (renderer backend creation, physics / controller sessions)
  first      -> B's first drive call returned (the first frame the viewer shows)
  runup      -> B's hold point reached (force-GT replay to the hand-off; first started drive minus its wait_s)
  gap_frame  A's last drive -> B's first drive (what the user sees as loading);  gap_hold  A's last drive -> B held at the hand-off
With prefetch (HUMAN_PREFETCH=1) B is loaded and held before A ends: gap_hold is then A's last drive -> B shown active (driver
sessions.jsonl "active" record), and the per-phase split is reported for B's background load.
"""
import datetime as dt
import json
import re
import statistics as stt
import sys
from pathlib import Path

TS = re.compile(r"^runtime-\d+-\d+\s+\|\s+(\d\d):(\d\d):(\d\d)\.(\d+) \[(W\d+)\]")


def jl(p):
    try:
        return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
    except OSError:
        return []


def epoch(h, m, s, frac, ref):
    """UTC clock time -> the epoch nearest to ref."""
    d = dt.datetime.fromtimestamp(ref, dt.timezone.utc)
    best = None
    for k in (-1, 0, 1):
        day = (d + dt.timedelta(days=k)).replace(hour=int(h), minute=int(m), second=int(s), microsecond=int(frac.ljust(6, "0")[:6]))
        t = day.timestamp()
        best = t if best is None or abs(t - ref) < abs(best - ref) else best
    return best


def runtime_events(log, ref):
    """{uuid: {start, end, done}} and [(t, scene)] camera requests from the runtime's lines."""
    ev, cams, last_end = {}, [], {}
    for ln in open(log, errors="replace"):
        m = TS.match(ln)
        if not m:
            continue
        t = epoch(*m.groups()[:4], ref)
        w = m.group(5)
        if "Requesting available cameras for scene_id=" in ln:
            cams.append((t, ln.split("scene_id='")[1].split("'")[0]))
        elif "Session STARTING: uuid=" in ln:
            ev.setdefault(ln.split("uuid=")[1].split()[0], {})["start"] = t
        elif "Simulation ended" in ln:
            last_end[w] = t
        elif "Session COMPLETED: uuid=" in ln or "Session FAILED" in ln or ("uuid=" in ln and "CANCELLED" in ln):
            u = ln.split("uuid=")[1].split()[0].rstrip(",")
            ev.setdefault(u, {})["done"] = t
            ev[u]["end"] = last_end.pop(w, None)
    return ev, cams


def analyse(rd):
    rd = Path(rd)
    res = sorted((json.loads(p.read_text()) for p in (rd / "feed/res").glob("*.json")), key=lambda r: r.get("t_sent", 0))
    res = [r for r in res if r.get("rollouts")]
    if not res:
        return []
    drives, closes, active = {}, {}, {}
    for r in jl(rd / "driver/drive.jsonl"):
        if r.get("kind") == "drive":
            drives.setdefault(r["session"], []).append(r)
        elif r.get("kind") == "close":
            closes[r["session"]] = r
    for r in jl(rd / "driver/sessions.jsonl"):
        if r.get("kind") == "active":
            active.setdefault(r["session"], r["t"])
    ev, cams = runtime_events(rd / "compose.log", res[0]["t_sent"])
    out = []
    for a, b in zip(res, res[1:]):
        ua, ub = a["rollouts"][0]["uuid"], b["rollouts"][0]["uuid"]
        da, db = drives.get(ua), drives.get(ub)
        if not da or not db:
            continue
        ca, cb = closes.get(ua, {}), closes.get(ub, {})
        if ca.get("aborted") or cb.get("aborted"):
            continue
        last_a, first_b = da[-1]["t"], db[0]["t"]
        st_b = next((r["t"] for r in db if r.get("started")), None)
        hold_b = st_b - (cb.get("wait_s") or 0.0) if st_b else None
        cam_b = next((t for t, s in cams if s == b["scene"] and t >= b["t_sent"] - 0.5), None)
        e_a, e_b = ev.get(ua, {}), ev.get(ub, {})
        row = {"a": a["scene"][7:15], "b": b["scene"][7:15], "driver": db[0].get("driver", "human"),
               "metrics": e_a.get("done", a["t_done"]) - last_a, "dispatch": b["t_sent"] - e_a.get("done", a["t_done"]),
               "rt_load": (cam_b - b["t_sent"]) if cam_b else None, "nre_load": (e_b["start"] - cam_b) if cam_b and "start" in e_b else None,
               "first": (first_b - e_b["start"]) if "start" in e_b else None, "runup": (hold_b - first_b) if hold_b else None,
               "gap_frame": first_b - last_a, "gap_hold": (hold_b - last_a) if hold_b else None,
               "wps_a": ca.get("wall_per_sim"), "wps_b": cb.get("wall_per_sim"), "unpaced_b": cb.get("unpaced_per_sim"),
               "late_b": cb.get("late"), "wait_b": cb.get("wait_s")}
        if ub in active:                                         # prefetch: B became the shown session at this time
            row["gap_hold"] = max(active[ub], hold_b or 0) - last_a
            row["gap_frame"] = active[ub] - last_a
        out.append(row)
    return out


def main():
    args = sys.argv[1:]
    js = None
    if "--json" in args:
        i = args.index("--json")
        js, args = args[i + 1], args[:i] + args[i + 2:]
    allrows = {}
    cols = ["metrics", "dispatch", "rt_load", "nre_load", "first", "runup", "gap_frame", "gap_hold", "wps_b", "unpaced_b"]
    for rd in args:
        rows = analyse(rd)
        allrows[rd] = rows
        print("## %s: %d scene pairs" % (rd, len(rows)))
        print("| a -> b | driver | " + " | ".join(cols) + " |")
        for r in rows:
            print("| %s -> %s | %s | " % (r["a"], r["b"], r["driver"]) + " | ".join("-" if r[c] is None else "%.2f" % r[c] for c in cols) + " |")
        if rows:
            med = [stt.median([r[c] for r in rows if r[c] is not None] or [float("nan")]) for c in cols]
            print("| median | | " + " | ".join("%.2f" % v for v in med) + " |")
        print()
    if js:
        Path(js).write_text(json.dumps(allrows, indent=1))


if __name__ == "__main__":
    main()
