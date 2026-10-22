"""Stored per-scene AlpaSim reads of the panel's Driver checkpoints -> one table (lib/alpasim_exam.stored_scores reads it).

  # Mac, repo root: each host prints its rows (read-only there), the Mac writes the table and copies it to Tokyo
  export TAGS=$(ssh ujs@100.108.238.8 'cd ~/mycode/jev-drive/experiments/alpasim_human/lib && python3 -c "import alpasim_exam as a; print(\",\".join(c[0] for c in a.driver_choices()[1:]))"')
  for h in autodl:box:/root/autodl-tmp/ujs:/root/autodl-tmp/ujs/jev-drive tokyo:tokyo:/data:/home/ujs/mycode/jev-drive; do IFS=: read s n d r <<< $h
    ssh ${s/tokyo/ujs@100.108.238.8} "TAGS=$TAGS DATA_DIR=$d python3 - rows --host $n --repo $r" < experiments/alpasim_human/scripts/stored_scores.py; done \
    | python3 experiments/alpasim_human/scripts/stored_scores.py write --out experiments/alpasim_human/results/stored_scores.tsv
  scp experiments/alpasim_human/results/stored_scores.tsv ujs@100.108.238.8:/data/runs/alpasim_human/refs/

Columns: tag, track (pai / nuplan), scene_id, score (mean over the read's rollouts of the scene), why (PAI facts: fclass (ok -> pass / partial), primary in
brackets; results-summary reads: alpasim_exam.why of the scene's worst rollout), read (short name), config (serve), src (the read root
under runs/ of the host, box: = the GPU box, tokyo: = Tokyo). TAGS = the Driver tags (alpasim_exam.driver_choices); rows of another seed of the
same family (tag minus -s<k>) are kept only for a Driver tag that has no row of its own on that track.
Configs: "ACC+GAP" = the PAI submission serve (JEV_VCONT 1.0, LEAD, ACC, GAP 0.7,2.7, steering weight 10), "ACC+GAP+harm" = the same
with the Harmonizer on, "ACC" = without GAP, "VC+LEAD" = PAI native serve, "native" = nuPlan, every serving switch off, "stop3" /
"leash3" = nuPlan hold serves, "unrecorded" = the run dir keeps no serve. The panel's model mode serves PAI with ACC+GAP and nuPlan
native (scripts/model_driver.sh), so those are the matching reads; the API prefers them. Plain python3 (stdlib).
"""
import csv
import glob
import json
import os
import re
import sys
from pathlib import Path

D = Path(os.environ.get("DATA_DIR", "/data")) / "runs"
COLS = ["tag", "track", "scene_id", "score", "why", "read", "config", "src"]
# PAI facts (box runs/pai_fix/facts, pai_fix.py facts: rf_tax extract of a stored read): file suffix -> (read, config)
# (the native -vl.csv reads are left out: every tag that has one also has an ACC+GAP read of the same scenes)
FACTS = [("-281ag.csv", "PAI 281", "ACC+GAP"), ("-281agH.csv", "PAI 281", "ACC+GAP+harm"), ("-ag.csv", "PAI-FIX exam (val160 + T281)", "ACC+GAP")]
# results-summary reads: (host, tag, track, read, config, run-dir glob under runs/); RSG's native g10 / ACC-only 281 reads (route_follow/
# pai_g10/{g10,acc}) are left out beside its ACC+GAP read
RS = [("box", "RSG-L18-s0", "pai", "PAI 281 (d330 acc_gap/read/s2)", "ACC+GAP", "route_follow/acc_gap/read/s2/runs/*"),
      ("box", "RBSG2-L18-s0", "pai", "PAI 281 (pai_g10/rbsg2g10)", "VC+LEAD", "route_follow/pai_g10/rbsg2g10/runs/*"),
      ("box", "JTN-L18-s0", "pai", "PAI 281 (d359 defect_fix/pai_accgap)", "ACC+GAP", "defect_fix/pai_accgap/JTN-L18-s0/runs/*"),
      ("box", "JTN-L18-s1", "pai", "PAI 281 (defect_fix/pai_accgap)", "ACC+GAP", "defect_fix/pai_accgap/JTN-L18-s1/runs/*"),
      ("box", "NTV0-L18-s1", "pai", "PAI 281 (defect_fix/pai_accgap)", "ACC+GAP", "defect_fix/pai_accgap/NTV0-L18-s1/runs/*")]
RS += [("box", t, "pai", "PAI 281 (onpol_long/pai281/%s)" % d, "unrecorded", "onpol_long/pai281/%s/*" % d)
       for t, d in [("YS-L18-s0", "ys0"), ("YS-L18-s1", "ys1"), ("YNONE-L18-s0", "ynone0"), ("YNONE-L18-s1", "ynone1"),
                    ("YS2G-L18-s0", "ys2g0"), ("YS2G-L18-s1", "ys2g1"), ("YR10m10-F-s0", "ybase")]]
RS += [("box", None, "nuplan", "nuPlan public 1491 @10 (%s)" % "/".join(g.split("/")[:2]), "native", g + "/<tag>/*")
       for g in ("route_follow/guard/read10", "route_follow/slow/read10", "route_follow/allrows/read10", "onpol_long/read10")]
RS += [("tokyo", t, "nuplan", "nuPlan 48 (native_ref/%s)" % d, c, "alpasim/native_ref/" + d)
       for t, d, c in [("JTN-L18-s1", "JTN-L18-s1-native", "native"), ("JTN-L18-s1", "JTN-L18-s1-holdstop3", "stop3"),
                       ("JTN-L18-s0", "JTN-L18-s0-holdstop3", "stop3"), ("RSG-L18-s0", "RSG-L18-s0-holdstop3", "stop3"),
                       ("RSG-L18-s0", "RSG-L18-s0-holdleash3", "leash3"), ("RTB1ARO-L18-s0", "RTB1ARO-L18-s0", "native"),
                       ("YR10m10-F-s0", "YR10m10-F-s0", "native"), ("P2H10-F-s0", "P2H10-F-s0", "native")]]
RS += [("tokyo", t, "pai", "PAI %s (alpasim/%s)" % (n, g.split("/")[0]), "unrecorded", "alpasim/" + g)
       for t, n, g in [("P2H10-F-s0", "281", "pai_base/runs/p2h10-f-s0*"), ("P2H10-F-s1", "281", "pai_base_s1/runs/p2h10-f-s1*"),
                       ("P2H10S-F-s1", "281", "pai_s1/runs/p2h10s-f-s1*"), ("P2H10-F-s0", "stage 1", "pai1/runs/a10")]]


def short(p):
    """A run dir as its read root relative to runs/ (the chunk dirs of a read share it)."""
    p = os.path.relpath(str(p), str(D))
    return re.sub(r"/(runs/[^/]+|b\d+|[^/]*_c\d+)(/.*)?$", "", p)


def family(tag):
    return re.sub(r"-s\d+$", "", tag)


def config_of(run_dir, default):
    """Serve of a pai_box.sh read (config.txt beside runs/), else `default`."""
    cfg = Path(run_dir).parent.parent / "config.txt"
    if default != "unrecorded" or not cfg.is_file():
        return default
    s = cfg.read_text()
    return "ACC+GAP" if "JEV_GAP" in s else "ACC" if "JEV_ACC=1" in s else "VC+LEAD" if "JEV_LEAD=1" in s else "unrecorded"


def rs_rows(tag, track, read, config, dirs, keep=None):
    """Rows of a results-summary read (chunk dirs merged by scene): mean score, why of the worst rollout."""
    from alpasim_exam import why                                         # noqa: E402 (repo of the host)
    per = {}
    for d in dirs:
        for c in (Path(d) / "sim/aggregate/results-summary.json", Path(d) / "aggregate/results-summary.json"):
            if c.is_file():
                for r in json.loads(c.read_text())["rollouts"]:
                    per.setdefault(r["clipgt_id"], []).append((r["score"] or 0.0, r, d))
                break
    for sid, v in per.items():
        if keep is None or sid in keep:
            lo = min(v, key=lambda x: x[0])
            yield [tag, track, sid, "%.4f" % (sum(x[0] for x in v) / len(v)), why(lo[1]), read, config_of(lo[2], config), short(lo[2])]


def rows(host, repo):
    sys.path.insert(0, str(Path(repo) / "experiments/alpasim_human/lib"))
    tags = [t for t in os.environ.get("TAGS", "").split(",") if t]
    fams = {family(t) for t in tags} or None
    out = []
    if host == "box":
        for f in sorted(glob.glob(str(D / "pai_fix/facts/*.csv"))):
            name = Path(f).name
            for suf, read, config in FACTS:
                if name.endswith(suf) and not name.startswith("v160-"):
                    tag = name[:-len(suf)]
                    break
            else:
                if not name.startswith("v160-"):
                    continue
                tag, read, config = name[5:-4], "PAI val160 (route_follow/rsgacc roll)", "ACC+GAP"
            if fams is None or family(tag) in fams:
                for r in csv.DictReader(open(f)):
                    fc, pr = r.get("fclass") or r.get("ptype") or "", r.get("primary") or ""
                    fc = ("pass" if float(r["score"]) >= 1 else "partial") if fc == "ok" else fc      # as alpasim_exam.why words it
                    out.append([tag, "pai", r["scene"], "%.4f" % float(r["score"]), fc + (" (%s)" % pr if pr and pr not in ("ok", fc) else ""),
                                read, config, short(f)])
    nu48 = set((D / "alpasim/scenes_navtest_full_part001_48.txt").read_text().split())
    for h, tag, track, read, config, g in RS:
        if h != host:
            continue
        if tag is None:                                                  # one dir per tag under g
            base = g.split("/<tag>")[0]
            for td in sorted(glob.glob(str(D / base / "*"))):
                t = Path(td).name
                if fams is None or family(t) in fams:
                    out += rs_rows(t, track, read, config, sorted(glob.glob(td + "/*")), nu48)
        else:
            out += rs_rows(tag, track, read, config, sorted(glob.glob(str(D / g))), nu48 if track == "nuplan" else None)
    for r in out:
        r[7] = "%s:%s" % (host, r[7])
        print("\t".join(str(x).replace("\t", " ") for x in r))


def write(out):
    seen, L = set(), []
    for ln in sys.stdin:
        r = ln.rstrip("\n").split("\t")
        if len(r) == len(COLS) and tuple(r[:3] + r[5:7]) not in seen:
            seen.add(tuple(r[:3] + r[5:7]))
            L.append(r)
    tags = [t for t in os.environ.get("TAGS", "").split(",") if t]
    have = {(r[0], r[1]) for r in L}
    need = {(family(t), tr) for t in tags for tr in ("pai", "nuplan") if (t, tr) not in have}   # family seeds stand in only for these
    L = [r for r in L if not tags or r[0] in tags or (family(r[0]), r[1]) in need]
    L.sort(key=lambda r: (r[0], r[1], r[6], r[5], r[2]))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\t".join(COLS) + "\n" + "".join("\t".join(r) + "\n" for r in L))
    print("%s: %d rows, %d tags" % (out, len(L), len({r[0] for r in L})))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["rows"]:
        rows(a[a.index("--host") + 1], a[a.index("--repo") + 1])
    elif a[:1] == ["write"]:
        write(a[a.index("--out") + 1])
    else:
        print(__doc__)
