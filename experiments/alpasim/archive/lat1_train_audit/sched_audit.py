"""Training-side audit, scheduling: what the pool's own records say about idle cards / idle cores. Read-only over
$DATA_DIR/runs/pool/{events.jsonl,usage.jsonl,jobs/*/spec.json}. Prints JSON.

  python3 sched_audit.py [--since "2026-10-08 12:00:00"]
"""
import json, os, sys, time, glob, collections, argparse, statistics as st  # noqa: E401

P = os.path.join(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"), "runs/pool")
ap = argparse.ArgumentParser()
ap.add_argument("--since", default="2026-10-08 12:00:00")
a = ap.parse_args()
ts = lambda s: time.mktime(time.strptime(s, "%Y-%m-%d %H:%M:%S"))  # noqa: E731
T0 = ts(a.since)
ev = [json.loads(l) for l in open(os.path.join(P, "events.jsonl"))]
launch, end = {}, {}
for e in ev:
    if e["kind"] == "launch":
        launch[e["id"]] = e                                  # the last try
    elif e["kind"] == "end":
        end[e["id"]] = e
jobs = []
for i, e in end.items():
    if i not in launch or ts(launch[i]["t"]) < T0:
        continue
    try:
        sp = json.load(open(os.path.join(P, "jobs", i, "spec.json")))["spec"]
    except Exception:  # noqa: BLE001
        continue
    l = launch[i]
    jobs.append(dict(id=i, name=l["name"], t0=ts(l["t"]), wall=e.get("wall_s") or 0.0, cpu=sp.get("cpu") or 0, vram=sp.get("vram_gb") or 0.0,
                     booked=l.get("booked_gb") or 0.0, cores=e.get("cores_max") or 0.0, vpeak=e.get("vram_peak") or 0.0, waited=l.get("waited") or {},
                     state=e.get("state"), owner=sp.get("owner"), train=sp.get("train"), gpu=l.get("gpu")))
out = dict(since=a.since, jobs=len(jobs))
H = 3600.0
pin = [j for j in jobs if j["cpu"] > 0 and j["wall"] > 60]
out["cpu_declared_vs_measured"] = dict(
    jobs=len(pin), declared_core_h=round(sum(j["cpu"] * j["wall"] for j in pin) / H, 1), peak_core_h=round(sum(j["cores"] * j["wall"] for j in pin) / H, 1),
    median_peak_over_declared=round(st.median(j["cores"] / j["cpu"] for j in pin), 2),
    share_of_jobs_peak_below_half_declared=round(sum(j["cores"] < 0.5 * j["cpu"] for j in pin) / len(pin), 2),
    note="peak = the job's highest measured core use (cores_max), so peak core-hours is an upper bound of what was used")
gj = [j for j in jobs if j["vram"] >= 2 and j["wall"] > 60]
out["vram_declared_vs_measured"] = dict(
    jobs=len(gj), declared_gb_h=round(sum(j["vram"] * j["wall"] for j in gj) / H, 1), peak_gb_h=round(sum(j["vpeak"] * j["wall"] for j in gj) / H, 1),
    median_peak_over_declared=round(st.median(j["vpeak"] / j["vram"] for j in gj), 2),
    jobs_never_on_card=sum(j["vpeak"] < 0.3 for j in gj), gb_h_booked_by_jobs_never_on_card=round(sum(j["vram"] * j["wall"] for j in gj if j["vpeak"] < 0.3) / H, 1))
fam = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0, 0.0])
for j in jobs:
    k = j["name"].split("@")[0].rstrip("0123456789").split("-s")[0][:28]
    f = fam[k]
    f[0] += 1; f[1] += j["wall"] / H; f[2] += j["cpu"] * j["wall"] / H; f[3] += j["cores"] * j["wall"] / H; f[4] += j["vram"] * j["wall"] / H; f[5] += j["vpeak"] * j["wall"] / H  # noqa: E702
out["families_by_declared_core_h"] = [dict(name=k, jobs=v[0], wall_h=round(v[1], 1), declared_core_h=round(v[2], 1), peak_core_h=round(v[3], 1),
                                           declared_gb_h=round(v[4], 1), peak_gb_h=round(v[5], 1)) for k, v in sorted(fam.items(), key=lambda kv: -kv[1][2])[:18]]
w = collections.Counter()
for j in jobs:
    for k, v in j["waited"].items():
        w[k] += v
out["launch_wait_by_reason"] = {k: round(v, 1) for k, v in w.most_common()}
out["launch_wait_note"] = "sum over jobs of the launch event's `waited` values (units as the pool records them)"

us = [json.loads(l) for l in open(os.path.join(P, "usage.jsonl"))]
us = [u for u in us if u["t"] >= T0]
n = len(us)
meas, chg, bud, quota = (st.mean(u["cpu"][0] for u in us), st.mean(u["cpu"][1] for u in us), us[-1]["cpu"][2], us[-1]["quota"])
blocked = [u for u in us if "cpu" in u.get("q_gpu", {}) or "cpu" in u.get("q_cpu", {})]
out["usage_samples"] = dict(
    samples=n, minutes_per_sample=round((us[-1]["t"] - us[0]["t"]) / 60 / max(1, n - 1), 2), quota_cores=quota, budget_cores=bud,
    mean_measured_cores=round(meas, 1), mean_charged_cores=round(chg, 1),
    samples_with_work_blocked_on_cpu=len(blocked),
    of_those_mean_measured_cores=round(st.mean(u["cpu"][0] for u in blocked), 1) if blocked else None,
    of_those_mean_charged_cores=round(st.mean(u["cpu"][1] for u in blocked), 1) if blocked else None,
    of_those_measured_below_half_quota=sum(u["cpu"][0] < quota / 2 for u in blocked),
    queued_jobs_blocked_on_cpu_job_minutes=round(sum(u.get("q_gpu", {}).get("cpu", 0) + u.get("q_cpu", {}).get("cpu", 0) for u in blocked) *
                                                 (us[-1]["t"] - us[0]["t"]) / 60 / max(1, n - 1), 0))
cs = collections.defaultdict(lambda: collections.Counter())
for u in us:
    for c, (util, used, booked, nj, *_) in u["cards"].items():
        s = cs[c]
        s["n"] += 1
        if nj:
            s["with_job"] += 1
            s["util_sum"] += util
            s["idle_lt10"] += util < 10
            s["lt30"] += util < 30
            s["ge80"] += util >= 80
            s["used_gb"] += used; s["booked_gb"] += booked                                                                              # noqa: E702
out["cards_when_a_gpu_job_is_on_them"] = {c: dict(share_of_time_with_job=round(s["with_job"] / s["n"], 2), mean_util=round(s["util_sum"] / max(1, s["with_job"]), 1),
                                                   share_util_below_10=round(s["idle_lt10"] / max(1, s["with_job"]), 2), share_util_below_30=round(s["lt30"] / max(1, s["with_job"]), 2),
                                                   share_util_at_least_80=round(s["ge80"] / max(1, s["with_job"]), 2),
                                                   mean_vram_used_over_booked=round(s["used_gb"] / max(1e-9, s["booked_gb"]), 2)) for c, s in sorted(cs.items())}


def trace(name, t0, t1, card=None):
    w = [u for u in us if ts(t0) <= u["t"] <= ts(t1)]
    if not w:
        return
    out.setdefault("traces", {})[name] = dict(
        window=[t0, t1], card=card, util_per_min={c: [u["cards"][c][0] for u in w] for c in (w[0]["cards"] if card is None else [str(card)])},
        jobs_on_card={c: [u["cards"][c][3] for u in w] for c in (w[0]["cards"] if card is None else [str(card)])},
        measured_cores=[round(u["cpu"][0]) for u in w], charged_cores=[round(u["cpu"][1]) for u in w])


for j in jobs:                                              # the audited runs, by name
    if j["wall"] > 300 and any(k in j["name"] for k in ("ap2-prep", "ap2prep", "ap2-train", "AP2", "ot3-ap2", "otprep", "yr1")):
        t0, t1 = (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)) for x in (j["t0"], j["t0"] + j["wall"]))
        trace(f"{j['name']} {j['id']} (declared cpu {j['cpu']}, peak cores {j['cores']}, vram {j['vram']} / peak {j['vpeak']})", t0, t1, j["gpu"])
print(json.dumps(out, indent=1))
