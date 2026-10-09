"""The SIGKILLs of one night against the box sampler (INFRA2, 2026-10-10; results/sigkill-trigger/README.md).

For every pool try that ended with rc 137 on --day: the kill time (mtime of the job's rc.<try>), the victim's age,
its RSS and rank among the three largest processes in the last boxwatch sample before the kill, memory.current / anon /
file then, and over the 30 s before: the maximum of memory.current, memory.high events and anon growth. Then the
question the table was built for: which box state separates the kills from the rest of the night (samples by
memory.current x largest RSS, with the kills in each cell).

    python experiments/cl_infra/scripts/sigkill_trigger.py --day 20261010 [--out results.csv]
"""
import argparse, bisect, csv, json, os, sys  # noqa: E401
from pathlib import Path

G = 2 ** 30
sec = lambda s: int(s[:2]) * 3600 + int(s[3:5]) * 60 + float(s[6:])  # noqa: E731
hms = lambda t: "%02d:%02d:%02d" % (t // 3600, t % 3600 // 60, t % 60)  # noqa: E731


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", required=True, help="YYYYmmdd")
    ap.add_argument("--data", default=os.environ.get("DATA_DIR", ""))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    D = Path(a.data)
    day = "%s-%s-%s" % (a.day[:4], a.day[4:6], a.day[6:])
    rows = []
    for l in (D / "runs/boxwatch" / (a.day + ".tsv")).read_text().splitlines():
        p = l.split("\t")
        try:
            top = [(int(x.split(":")[0]), int(x.split(":")[1]) / 2 ** 20) for x in p[9].split()]
            rows.append((sec(p[0]), int(p[1]) / G, int(p[2]) / G, int(p[3]) / G, int(p[4]), top))
        except (ValueError, IndexError):
            continue
    T = [r[0] for r in rows]
    launch, names, kills = {}, {}, []
    for l in (D / "runs/pool/events.jsonl").open():
        if day not in l[:20]:
            continue
        e = json.loads(l)
        if e["kind"] == "launch":
            launch[(e["id"], e["try_"])] = (sec(e["t"][11:]), e["pid"])
            names[e["id"]] = e["name"]
    for (jid, tr), (t0, pid) in sorted(launch.items()):
        rc = D / "runs/pool/jobs" / jid / ("rc.%d" % tr)
        if rc.exists() and rc.read_text().strip() == "137":
            lt = __import__("time").localtime(rc.stat().st_mtime)
            kills.append((lt.tm_hour * 3600 + lt.tm_min * 60 + lt.tm_sec, jid, tr, t0, pid))
    kills.sort()
    out = []
    for t, jid, tr, t0, pid in kills:
        i = bisect.bisect_right(T, t - 2) - 1          # the rc file is written 1-3 s after the kill
        r, w = rows[i], [x for x in rows if t - 32 <= x[0] <= t]
        v = [(k + 1, x[1]) for k, x in enumerate(r[5]) if pid < x[0] <= pid + 3]    # taskset -> python: the next pids
        out.append(dict(kill=hms(t), job=names[jid], id=jid, try_=tr, age_s=round(t - t0), victim_rss_gb=round(v[0][1], 1) if v else "",
                        rss_rank=v[0][0] if v else "", top1_rss_gb=round(r[5][0][1], 1), cur_gb=round(r[1], 1), anon_gb=round(r[2], 1),
                        file_gb=round(r[3], 1), cur_max30_gb=round(max(x[1] for x in w), 1), high_events30=w[-1][4] - w[0][4],
                        anon_growth30_gb=round(w[-1][2] - w[0][2], 1)))
    wr = csv.DictWriter(open(a.out, "w") if a.out else sys.stdout, fieldnames=list(out[0]))
    wr.writeheader()
    wr.writerows(out)
    kt = [k[0] - 2 for k in kills]
    cb = lambda c: sum(c >= x for x in (500, 520, 530, 536, 542, 547))  # noqa: E731
    rb = lambda x: sum(x >= y for y in (5, 10, 15, 20, 25))  # noqa: E731
    tab = {}
    for i, r in enumerate(rows[:-1]):
        c = tab.setdefault((cb(r[1]), rb(r[5][0][1])), [0, 0])
        c[0] += 1
        c[1] += sum(r[0] <= t < rows[i + 1][0] for t in kt)
    print("\n%d kills in %d jobs, %s - %s; victim in the top 3 by RSS in %d (rank 1 in %d); memory.current max over the 30 s before: "
          "min %.1f GiB; with 0 memory.high events in those 30 s: %d; anon at the kill %.0f - %.0f GiB" % (
              len(out), len({o["id"] for o in out}), out[0]["kill"], out[-1]["kill"], sum(o["rss_rank"] != "" for o in out),
              sum(o["rss_rank"] == 1 for o in out), min(o["cur_max30_gb"] for o in out), sum(o["high_events30"] == 0 for o in out),
              min(o["anon_gb"] for o in out), max(o["anon_gb"] for o in out)), file=sys.stderr)
    print("\n5 s samples (kills) by memory.current (rows, GiB) and the largest RSS (columns, GB)\n| current | < 5 | 5-10 | 10-15 | 15-20 | 20-25 | >= 25 |\n|:--|--:|--:|--:|--:|--:|--:|",
          file=sys.stderr)
    for c, lab in enumerate(("< 500", "500-520", "520-530", "530-536", "536-542", "542-547", ">= 547")):
        print("| %s | %s |" % (lab, " | ".join("%d (%d)" % tuple(tab.get((c, q), (0, 0))) for q in range(6))), file=sys.stderr)


if __name__ == "__main__":
    main()
