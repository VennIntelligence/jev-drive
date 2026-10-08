"""Where the box's card-hours and core-hours went on one day, and what the pool's accounting change does to the same
submissions. Offline: reads a copy of the pool spool (state.json, status.json) and the box sampler's TSV.

  diagnose   card-hours with a GPU job vs idle, the idle time split by cause, and why ready GPU jobs waited
  replay     the day's submissions (recorded submit times, --after chains, run times, measured peaks) through the
             pool's own placement functions under the old accounting (trust_measured = false) and the new one

    scp autodl:data/runs/pool/{state,status}.json DIR/ ; scp autodl:data/runs/boxwatch/20261008.tsv DIR/
    python experiments/cl_infra/scripts/pool_usage.py DIR --day 2026-10-08 [--tz 8]

Limits. The pool recorded no per-card utilisation and no wait reasons before 2026-10-08, so "card busy" means a pool
job with >= 1 GB measured VRAM on the card, and the blocker of a waiting job is reconstructed from what ran at the
time. Peak RSS per job was not recorded either: the replay gives every job the same fraction of its declared RAM (the
day's mean non-reclaimable memory over the mean declared RAM of running jobs). Submit times are kept as recorded, so
lanes that would have submitted their next stage earlier gain nothing here: the replay is a lower bound.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import dataclasses
import datetime
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from jevdrive.cl import pool as P  # noqa: E402

STEP = 20.0                                    # the dispatcher's round
SEED = re.compile(r"-s\d+\b")


def load(d: Path, day: str, tz: float):
    st = json.loads((d / "state.json").read_text())["jobs"]
    status = json.loads((d / "status.json").read_text())
    z = datetime.timezone(datetime.timedelta(hours=tz))
    t0 = datetime.datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=z).timestamp()
    now = min(status["t"], t0 + 86400)
    anon = []
    f = d / (day.replace("-", "") + ".tsv")
    if f.exists():
        for ln in f.read_text().splitlines()[1:]:
            p = ln.split("\t")
            try:
                h, m, s = map(int, p[0].split(":"))
                anon.append((t0 + h * 3600 + m * 60 + s, int(p[2]) / 2 ** 30))
            except (ValueError, IndexError):
                pass
    return st, status, t0, now, anon


def ready_t(j, st):
    return max([j["t_submit"]] + [st[a]["t1"] for a in j["spec"].get("after") or [] if a in st and st[a].get("t1")])


def uses_gpu(j):
    return float(j.get("vram_peak") or 0) >= 1.0


def serial_runs(j) -> int:
    """How many independent runs one job does in a row, as far as its command shows (--jobs A B C, chained shells)."""
    cmd = j["spec"]["cmd"]
    toks = cmd.split() if isinstance(cmd, str) else [t for c in cmd for t in str(c).split()]
    if "--jobs" in toks:
        i = toks.index("--jobs")
        n = next((k for k, t in enumerate(toks[i + 1:]) if t.startswith("-")), len(toks) - i - 1)
        if n >= 2:
            return n
    m = re.match(r"(\d+) runs", P.serial_hint(cmd))
    return int(m.group(1)) if m else 0


def diagnose(st, status, t0, now, anon):
    box, cfg = status["box"], status["cfg"]
    cards = sorted(int(g) for g in status["cards"])
    cap = {int(g): c["total_gb"] - cfg["headroom_gb"] for g, c in status["cards"].items()}
    budget = box["cores"] * cfg["cpu_overcommit"]
    ram_cap = 0.85 * box["mem_max_gb"]
    n = int((now - t0) / STEP)
    H = (now - t0) / 3600
    ran = [j for j in st.values() if j.get("t0") and j.get("t1", now) > t0 and j["t0"] < now and j.get("gpu") in cards]
    run_at = collections.defaultdict(list)
    for j in ran:
        for i in range(max(int((j["t0"] - t0) / STEP), 0), min(int((j.get("t1", now) - t0) / STEP) + 1, n)):
            run_at[i].append(j)
    at = [a[0] for a in anon]

    def anon_gb(t):
        return anon[min(bisect.bisect_left(at, t), len(anon) - 1)][1] if anon else 0.0
    # Serial jobs: several runs inside one job, or a seed sibling submitted only after this one ended.
    serial = {}
    by_fam = collections.defaultdict(list)
    for j in ran:
        if uses_gpu(j):
            by_fam[(SEED.sub("-s#", j["spec"]["name"]), j["spec"].get("owner"))].append(j)
            if serial_runs(j) >= 2:
                serial[j["id"]] = "%d runs in one job" % serial_runs(j)
    for fam in by_fam.values():
        for a in fam:
            for b in fam:
                if a is not b and a["spec"]["name"] != b["spec"]["name"] and a.get("t1") and 0 <= b["t_submit"] - a["t1"] < 900:
                    serial.setdefault(a["id"], "seed sibling %s submitted after it ended" % b["spec"]["name"])
    waiting = collections.defaultdict(list)                      # step -> [(job, blocker)]
    wait_h = collections.Counter()
    for j in st.values():
        s = j["spec"]
        if not P.is_gpu(s) or not j.get("t0") or j["t0"] < t0:
            continue
        r = ready_t(j, st)
        for i in range(max(int((r - t0) / STEP) + 1, 0), min(int((j["t0"] - t0) / STEP), n)):
            t, R = t0 + i * STEP, run_at[i]
            ok = [g for g in cards if not s.get("gpus") or g in s["gpus"]]
            decl = {g: sum(max(x["spec"]["vram_gb"], x.get("vram_peak") or 0) for x in R if x["gpu"] == g) for g in cards}
            meas = {g: sum(P.VRAM_MARGIN * (x.get("vram_peak") or x["spec"]["vram_gb"]) + P.VRAM_PAD_GB for x in R if x["gpu"] == g and P.is_gpu(x["spec"]))
                    for g in cards}
            own = min(s["vram_gb"], P.VRAM_MARGIN * (j.get("vram_peak") or s["vram_gb"]) + P.VRAM_PAD_GB)
            young = sum(x["spec"].get("ram_gb") or 0 for x in R if t - x["t0"] < P.YOUNG_S)
            if not any(decl[g] + s["vram_gb"] <= cap[g] for g in ok):
                k = "VRAM over-declared (fits by measured peaks)" if any(meas[g] + own <= cap[g] for g in ok) else "VRAM really full"
            elif anon_gb(t) + young + (s.get("ram_gb") or 0) > ram_cap:
                k = "RAM gate: declared RAM of young jobs on top of measured memory"
            elif s.get("train") and all(sum(1 for x in R if x["gpu"] == g and x["spec"].get("train")) >= cfg["train_per_card"] for g in ok):
                k = "training jobs per card cap"
            elif sum(x["spec"].get("cpu") or 1 for x in R) + (s.get("cpu") or 1) > budget:
                k = "CPU budget (declared cores)"
            elif t - r <= 60:
                k = "dispatch latency (<= 1 min)"
            else:
                k = "not reconstructed (reservation, start limit, pins, PIDs)"
            wait_h[k] += STEP / 3600
            waiting[i].append(k)
    idle = collections.Counter()
    busy_h = {g: 0.0 for g in cards}
    for i in range(n):
        R = run_at[i]
        on = {g: [x for x in R if x["gpu"] == g and uses_gpu(x)] for g in cards}
        cores = sum(x.get("cores_max") or 0 for x in R)
        for g in cards:
            if on[g]:
                busy_h[g] += STEP / 3600
                continue
            others = [x for h in cards if h != g for x in on[h]]
            if waiting[i]:
                k = "ready GPU job queued: " + collections.Counter(waiting[i]).most_common(1)[0][0]
            elif any(x["id"] in serial for x in others):
                k = "serial in lane (a job on another card does several runs in a row)"
            elif cores >= 0.6 * box["cores"]:
                k = "CPU-bound stage running (>= 60 % of the quota), no GPU job ready"
            elif others:
                k = "nothing queued, other card(s) busy"
            else:
                k = "nothing queued, every card idle"
            idle[k] += STEP / 3600
    print("## Diagnosis, %.2f h window, cards %s\n" % (H, cards))
    print("card-hours: %.1f available, %.1f with a GPU-using pool job (%s), %.1f idle" % (
        H * len(cards), sum(busy_h.values()), ", ".join("card %d %.1f" % kv for kv in busy_h.items()), sum(idle.values())))
    print("\n| idle card-hours by cause | h | share of idle |\n|:--|--:|--:|")
    for k, v in idle.most_common():
        print("| %s | %.2f | %.0f %% |" % (k, v, 100 * v / max(sum(idle.values()), 1e-9)))
    gj = [j for j in ran if P.is_gpu(j["spec"]) and j["t0"] >= t0]
    print("\nready GPU jobs waited %.1f job-hours in total (%d jobs, %d waited > 2 min)" % (
        sum(wait_h.values()), len(gj), sum(1 for j in gj if j["t0"] - ready_t(j, st) > 120)))
    print("\n| blocker while a ready GPU job waited | job-hours | share |\n|:--|--:|--:|")
    for k, v in wait_h.most_common():
        print("| %s | %.2f | %.0f %% |" % (k, v, 100 * v / max(sum(wait_h.values()), 1e-9)))
    hrs = STEP / 3600
    used = sum(x.get("cores_max") or 0 for i in range(n) for x in run_at[i]) * hrs
    decl = sum(x["spec"].get("cpu") or 1 for i in range(n) for x in run_at[i]) * hrs
    vd = sum(x["spec"]["vram_gb"] for i in range(n) for x in run_at[i] if P.is_gpu(x["spec"])) * hrs
    vp = sum(x.get("vram_peak") or 0 for i in range(n) for x in run_at[i] if P.is_gpu(x["spec"])) * hrs
    rd = sum(x["spec"].get("ram_gb") or 0 for i in range(n) for x in run_at[i]) * hrs
    print("\ncore-hours: %.0f available (quota %.0f), <= %.0f used (sum of each job's peak cores, an upper bound), %.0f declared"
          % (box["cores"] * H, box["cores"], used, decl))
    print("VRAM GB-hours: %.0f available, %.0f declared, %.0f at measured peak; RAM GB-hours: %.0f declared, %.0f measured "
          "non-reclaimable (mean %.0f GB)" % (sum(cap.values()) * H, vd, vp, rd, sum(a[1] for a in anon) / max(len(anon), 1) * H,
                                             sum(a[1] for a in anon) / max(len(anon), 1)))
    print("\nserial jobs found: %s" % ("; ".join("%s (%s, %.0f min)" % (st[i]["spec"]["name"], w, (st[i].get("t1", now) - st[i]["t0"]) / 60)
                                                 for i, w in serial.items()) or "none"))
    ram_frac = (sum(a[1] for a in anon) / max(len(anon), 1) * H) / max(rd, 1e-9)
    return ran, serial, min(ram_frac, 1.0)


def replay(st, status, t0, now, ran, ram_frac, new: bool, fan: dict = None, label: str = "", keys: bool = False):
    """Run the day's jobs through the pool's placement with the old or the new accounting. fan: job id -> k splits it
    into k jobs of 1/k the run time (the fan-out rule applied to a job that did k runs in a row). keys: give bench
    stages the CL_HIST_KEY that jevdrive.bench.runner sets from 2026-10-08 on (history per stage kind)."""
    from jevdrive.bench.runner import hist_key
    box, cfg = status["box"], dict(status["cfg"], trust_measured=new)
    cfg = dict(P.DEFAULTS, **cfg)
    cards = sorted(int(g) for g in status["cards"])
    total = {int(g): c["total_gb"] for g, c in status["cards"].items()}
    budget, ram_cap = box["cores"] * cfg["cpu_overcommit"], 0.85 * box["mem_max_gb"]
    jobs = {}
    for j in ran:
        if j["t0"] < t0:
            continue
        k = (fan or {}).get(j["id"], 1)
        for i in range(k):
            jid = j["id"] if k == 1 else "%s#%d" % (j["id"], i)
            spec = dict(j["spec"]) if k == 1 else dict(j["spec"], name="%s-%d" % (j["spec"]["name"], i))
            hk = hist_key(spec["cmd"], spec["vram_gb"]) if keys else ""
            if hk:
                spec["env"] = dict(spec.get("env") or {}, CL_HIST_KEY=hk)
            jobs[jid] = dict(id=jid, spec=spec, t_submit=j["t_submit"], dur=(j.get("t1", now) - j["t0"]) / k, src=j["id"],
                             peak=float(j.get("vram_peak") or 0), cores=j.get("cores_max"), ok=j["state"] in ("done", "running"))
    children = collections.defaultdict(list)
    for jid, j in jobs.items():
        children[j["src"]].append(jid)
    hist, running, done_t, start = {}, {}, {}, {}
    queue = sorted(jobs.values(), key=lambda j: (-float(j["spec"].get("priority") or 0), j["t_submit"]))
    idle_since, blocked = {}, collections.Counter()
    t = min(j["t_submit"] for j in jobs.values())
    busy = 0.0
    while len(done_t) < len(jobs) and t < t0 + 3 * 86400:
        for jid in [i for i, r in running.items() if t >= r["t0"] + jobs[i]["dur"]]:
            r, j = running.pop(jid), jobs[jid]
            done_t[jid] = t
            if j["ok"] and j["dur"] >= P.HIST_MIN_WALL_S:
                e = hist.setdefault(P.hist_key(j["spec"]), dict(vram=[], cores=[], ram=[]))
                for key, v in (("vram", j["peak"]), ("cores", j["cores"] or 0), ("ram", ram_frac * (j["spec"].get("ram_gb") or 0))):
                    if v > 0:
                        e[key] = (e[key] + [v])[-P.HIST_KEEP:]
        acct = {g: P.CardAcct(g, 0, total[g], 0.0) for g in cards}
        cpu_used = mem_used = young_ram = 0.0
        for jid, r in running.items():
            j, age = jobs[jid], t - r["t0"]
            live = age >= 60                                          # a job reaches its measured peaks after a minute
            view = dict(spec=j["spec"], t0=r["t0"], vram_peak=j["peak"] if live else 0.0, rss_now=ram_frac * (j["spec"].get("ram_gb") or 0) if live else 0.0,
                        cores_peak=j["cores"] if live else None)
            known = P.known_caps(hist, j["spec"]) if new else {}
            a = acct[r["gpu"]]
            a.pool_gb += max(view["vram_peak"], P.vram_charge(view, t, known, new))
            a.jobs += P.is_gpu(j["spec"]) if new else 1
            a.train += bool(j["spec"].get("train"))
            a.carla_pool += j["spec"].get("carla") or 0
            a.exclusive = a.exclusive or bool(j["spec"].get("exclusive"))
            cpu_used += P.cpu_charge(view, t, known.get("cpu") if new else None)
            mem_used += view["rss_now"]
            if age < P.YOUNG_S:
                young_ram += P.ram_reserve(view, t, known.get("ram_gb")) if new else float(j["spec"].get("ram_gb") or 0)
        for g, a in acct.items():
            if a.jobs == 0 and not a.exclusive:
                idle_since.setdefault(g, t)
            else:
                idle_since.pop(g, None)
        busy += sum(1 for g in cards if any(r["gpu"] == g and jobs[i]["peak"] >= 1 for i, r in running.items())) * STEP / 3600
        starts = 0
        for j in queue:
            jid, s = j["id"], P.Spec(**j["spec"])
            if jid in start or t < j["t_submit"]:
                continue
            gpu_job = P.is_gpu(j["spec"])
            if starts >= cfg["max_starts"]:
                blocked["start limit"] += gpu_job and "ready" in j
                continue
            if any(c not in done_t for a in s.after for c in children.get(a, [])):
                continue
            j.setdefault("ready", t)
            known = P.known_caps(hist, j["spec"]) if new else {}
            n_cpu = max(1, min(max(s.cpu, 1), known.get("cpu", max(s.cpu, 1))))
            ram_need = min(s.ram_gb, known.get("ram_gb", s.ram_gb))
            v_need = P.vram_need(j["spec"], known, new)
            soft = cpu_used + n_cpu > budget
            if mem_used + young_ram + ram_need > ram_cap or (soft and s.vram_gb <= 1):
                blocked["RAM gate"] += gpu_job
                continue
            pool = list(acct.values())
            if soft:
                pool = [a for a in pool if a.jobs == 0 and not a.exclusive and t - idle_since.get(a.index, t) >= cfg["idle_s"]]
            a, _ = P.choose(dataclasses.replace(s, vram_gb=v_need), pool, cfg, jid)
            if a is None:
                why = "CPU budget" if soft else "training cap" if s.train and all(c.train >= cfg["train_per_card"] for c in pool) else "VRAM"
                blocked[why] += gpu_job
                continue
            start[jid] = t
            running[jid] = dict(t0=t, gpu=a.index)
            starts += 1
            a.jobs += P.is_gpu(j["spec"]) if new else 1
            a.pool_gb += v_need
            a.train += s.train
            a.carla_pool += s.carla
            a.exclusive = a.exclusive or s.exclusive
            cpu_used += n_cpu
            young_ram += ram_need
        t += STEP
    gpu = [j for j in jobs.values() if P.is_gpu(j["spec"]) and j["id"] in start]
    wait = sum(start[j["id"]] - j["ready"] for j in gpu) / 3600
    allw = sum(start[j["id"]] - j["ready"] for j in jobs.values() if j["id"] in start) / 3600
    span = (max(done_t.values()) - min(j["t_submit"] for j in jobs.values())) / 3600
    return dict(label=label, jobs=len(jobs), gpu_jobs=len(gpu), gpu_wait_h=wait, all_wait_h=allw, over_2min=sum(1 for j in gpu if start[j["id"]] - j["ready"] > 120),
                max_wait_min=max((start[j["id"]] - j["ready"]) / 60 for j in gpu), span_h=span, busy_h=busy, blocked={k: v * STEP / 3600 for k, v in blocked.items()}, start=start, done=done_t, jobs_map=jobs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path)
    ap.add_argument("--day", required=True)
    ap.add_argument("--tz", type=float, default=8.0, help="the box's UTC offset in hours")
    a = ap.parse_args()
    st, status, t0, now, anon = load(a.dir, a.day, a.tz)
    ran, serial, ram_frac = diagnose(st, status, t0, now, anon)
    fan = {i: serial_runs(st[i]) for i in serial if serial_runs(st[i]) >= 2}
    actual = sum(j["t0"] - ready_t(j, st) for j in ran if P.is_gpu(j["spec"]) and j["t0"] >= t0) / 3600
    print("\n## Replay of the day's submissions (RAM use per job = %.2f x declared)\n" % ram_frac)
    print("recorded: ready GPU jobs waited %.1f job-hours\n" % actual)
    rows = [replay(st, status, t0, now, ran, ram_frac, False, label="old accounting (declared)"),
            replay(st, status, t0, now, ran, ram_frac, True, label="new accounting (measured, history from empty)")]
    rows.append(replay(st, status, t0, now, ran, ram_frac, True, label="new accounting + bench history per stage kind", keys=True))
    if fan:
        rows.append(replay(st, status, t0, now, ran, ram_frac, True, fan, "the same + serial jobs fanned out", keys=True))
    print("| replay | jobs | GPU-job wait, job-hours | GPU jobs waiting > 2 min | longest wait, min | all-job wait, job-hours |\n|:--|--:|--:|--:|--:|--:|")
    for r in rows:
        print("| %(label)s | %(jobs)d | %(gpu_wait_h).1f | %(over_2min)d | %(max_wait_min).0f | %(all_wait_h).1f |" % r)
    print("\nGPU-job wait by blocker, job-hours: " + "; ".join(
        "%s: %s" % (r["label"].split(" (")[0], ", ".join("%s %.1f" % kv for kv in sorted(r["blocked"].items(), key=lambda kv: -kv[1]))) for r in rows))
    for jid, k in fan.items():
        j = st[jid]
        new = rows[-1]
        ends = [new["done"][c] for c in new["done"] if new["jobs_map"][c]["src"] == jid]
        begins = [new["start"][c] for c in new["start"] if new["jobs_map"][c]["src"] == jid]
        print("\n%s (%d runs in one job): recorded %.0f min on one card; fanned out %.0f min from first start to last end"
              % (j["spec"]["name"], k, (j.get("t1", now) - j["t0"]) / 60, (max(ends) - min(begins)) / 60))


if __name__ == "__main__":
    main()
