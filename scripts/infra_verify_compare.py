#!/usr/bin/env python3
"""Behaviour equivalence of the harness optimisations (scripts/infra_scale.sh verify): compare each variant's
attempt with the reference run image by image and tick by tick.

    python3 scripts/infra_verify_compare.py $DATA_DIR/runs/infra-acceptance/scale verify-base-a \
        verify-base-b verify-res64 verify-lights verify-threads8

Per variant: camera frames compared (md5 of the raw BGRA buffer, matched per camera by position in the stream),
the share that are bitwise identical, the first differing tick, and the largest difference of the hero's truth pose
and of the applied control over the common ticks. Prints a markdown table and writes verify.json next to the runs.
"""
import glob
import json
import sys
from pathlib import Path


def attempt(root, tag):
    d = sorted(glob.glob(str(Path(root) / tag / "rung-01/w00/attempts/*/1")))
    return Path(d[0]) if d else None


def frames(adir):
    by = {}
    for line in open(adir / "frame_hash.jsonl"):
        rec = json.loads(line)
        by.setdefault(rec[0], []).append((rec[1], rec[2], rec[3] if len(rec) > 3 else None))
    return {t: [(m, th) for _, m, th in sorted(v)] for t, v in by.items()}


def thumb_diff(a, b):
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def ticks(adir):
    rows = [json.loads(line) for line in open(adir / "ticks.jsonl")]
    return [(r.get("truth"), (r["throttle"], r["steer"], r["brake"])) for r in rows]


def compare(ref, var):
    fr, fv = frames(ref), frames(var)
    n = same = 0
    first = None
    early, alld = [], []   # mean |grey difference| (0-255) of matched thumbnails: first 10 frames per camera, all
    for tag in fr:
        a, b = fr[tag], fv.get(tag, [])
        for i, (x, y) in enumerate(zip(a, b)):
            n += 1
            same += x[0] == y[0]
            if x[0] != y[0] and (first is None or i < first):
                first = i
            if x[1] is not None and y[1] is not None:
                d = thumb_diff(x[1], y[1])
                alld.append(d)
                if i < 10:
                    early.append(d)
    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else None  # noqa: E731
    tr, tv = ticks(ref), ticks(var)
    k = min(len(tr), len(tv))
    dpose = max((max(abs(p - q) for p, q in zip(a[0], b[0])) for a, b in zip(tr[:k], tv[:k])
                 if a[0] and b[0]), default=None)
    dctl = max((max(abs(p - q) for p, q in zip(a[1], b[1])) for a, b in zip(tr[:k], tv[:k])), default=None)
    first_tick = next((i for i, (a, b) in enumerate(zip(tr[:k], tv[:k])) if a != b), None)
    return {"frames": n, "frames_identical": same, "first_frame_diff_index": first,
            "thumb_diff_first10_median": med(early), "thumb_diff_median": med(alld), "ticks_ref": len(tr),
            "ticks_var": len(tv), "first_tick_diff": first_tick, "max_pose_diff": dpose, "max_control_diff": dctl}


def main():
    root, ref, variants = sys.argv[1], sys.argv[2], sys.argv[3:]
    r = attempt(root, ref)
    out = {}
    print("| variant | frames identical | first differing frame | thumbnail diff, first 10 / all (median) | ticks | "
          "first differing tick | max pose diff (m / rad) | max control diff |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for v in variants:
        a = attempt(root, v)
        if a is None or r is None or not (a / "ticks.jsonl").exists():
            print("| %s | no run (missing or crashed) | | | | | | |" % v)
            continue
        c = out[v] = compare(r, a)
        fmt = lambda x: "-" if x is None else "%.2f" % x  # noqa: E731
        print("| %s | %d / %d | %s | %s / %s | %d | %s | %s | %s |" % (
            v, c["frames_identical"], c["frames"], c["first_frame_diff_index"],
            fmt(c["thumb_diff_first10_median"]), fmt(c["thumb_diff_median"]), c["ticks_var"],
            c["first_tick_diff"], "%.4f" % c["max_pose_diff"] if c["max_pose_diff"] is not None else "-",
            "%.4f" % c["max_control_diff"] if c["max_control_diff"] is not None else "-"))
    (Path(root) / "verify.json").write_text(json.dumps({"reference": ref, "variants": out}, indent=1))


if __name__ == "__main__":
    main()
