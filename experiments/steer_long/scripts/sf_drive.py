#!/usr/bin/env python3
"""SPIN-FIX: the driver's inference records (drive.jsonl, infer == true) of the given sessions -> one gzipped JSONL (Tokyo host).
  sf_drive.py <sessions.json> <out.jsonl.gz> <pai_eval root> ...
Kept fields: session, scene, now, t0, anchor, poses (served), poses_model, fix, ego, cmd."""
import glob, gzip, json, os, sys

S = set(json.load(open(sys.argv[1])))
K = ("session", "scene", "now", "t0", "anchor", "poses", "poses_model", "fix", "ego", "cmd")
n = 0
with gzip.open(sys.argv[2], "wt") as g:
    for root in sys.argv[3:]:
        for f in sorted(glob.glob(os.path.join(root, "runs", "*", "driver", "drive.jsonl"))):
            for line in open(f):
                if '"infer": true' not in line or line[line.find('"session": "') + 12:][:36] not in S:
                    continue
                r = json.loads(line)
                g.write(json.dumps({k: r.get(k) for k in K}) + "\n")
                n += 1
print(n, "records")
