"""Worker-profile experiment readout (fc65452:todos/2026-10-01-cl-lib.md): per-arm throughput, threads, start failures, DS, and
DS agreement between arms, from the lane root of experiments/cl_infra/archive/cl_worker_profile.py.

    .venv/bin/python -m experiments.cl_infra.archive.cl_profile_report [--root $DATA_DIR/runs/cl_lib/profile] [--old name=dir ...]

Writes <root>/results/{arms.csv, routes.csv, pairs.csv, summary.md} and, when Stage B exists, workers_per_card.png/pdf.
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
from pathlib import Path

import pandas as pd

from jevdrive.common import data_dir

OLD20 = set("2390,24211,1711,2373,3564,1833,1852,1956,2668,4183,11381,1825,2084,2086,2091,2115,2286,24330,17563,26458".split(","))
RENDER_TIMEOUT = b"GameThread timed out waiting for RenderThread"


def route_ds(arm: Path) -> dict:
    out = {}
    for f in sorted((arm / "done").glob("*.json")):
        d = json.loads(f.read_text())
        res = arm / "attempts" / f.stem / str(d["attempt"]) / "results.json"
        try:
            rec = json.loads(res.read_text())["_checkpoint"]["records"][0]
            out[f.stem] = dict(ds=round(rec["scores"]["score_composed"], 2), status=rec["status"],
                               ticks=d.get("ticks"), wall_s=d.get("wall_s"))
        except (OSError, ValueError, KeyError, IndexError):
            out[f.stem] = dict(ds=None, status="no_result", ticks=d.get("ticks"), wall_s=d.get("wall_s"))
    return out


def start_failures(arm: Path) -> dict:
    """Attempts that died before ticking (server died / hung at 0 ticks) and server logs ending in the RenderThread
    timeout, plus how many servers were started."""
    early = 0
    for a in arm.glob("attempts/*/*/attempt.json"):
        d = json.loads(a.read_text())
        if d.get("status") != "finished" and not d.get("ticks"):
            early += 1
    logs = list((arm / "servers").glob("carla-*.log"))
    rt = sum(RENDER_TIMEOUT in p.read_bytes()[-20000:] for p in logs)
    return dict(servers_started=len(logs), early_attempt_failures=early, render_timeouts=rt)


def util_window(util: pd.DataFrame, gpu: int, t0: float, t1: float) -> dict:
    u = util[(util.gpu == gpu) & (util.t >= t0) & (util.t <= t1)]
    if not len(u):
        return {}
    full = u[u.lane_servers == u.lane_servers.max()]
    return dict(cores_busy_mean=round(u.cores_busy.mean(), 2), cores_busy_p95=round(u.cores_busy.quantile(.95), 2),
                gpu_util_mean=round(u.util_pct.mean(), 1), vram_peak_gb=round(u.mem_mib.max() / 1024, 1),
                threads_per_server=round((full.server_threads / full.lane_servers.clip(lower=1)).median(), 1),
                threads_per_client=round((u.client_threads / u.lane_clients.clip(lower=1))[u.lane_clients > 0].median(), 1)
                if (u.lane_clients > 0).any() else None,
                lane_threads_peak=int((u.server_threads + u.client_threads + u.other_threads).max()),
                pids_peak=int(u.pids_current.max()))


def arm_row(root: Path, name: str, state: dict, util: pd.DataFrame) -> dict:
    arm = root / "arms" / name
    job = json.loads((root / "jobs" / name / ("job.%d.json" % state["tries"])).read_text())
    meta = job["meta"]
    ev = [json.loads(l) for l in (arm / "events.jsonl").read_text().splitlines()]
    ends = [e for e in ev if e["kind"] == "route_end" and e.get("status") == "finished"]
    starts = [e for e in ev if e["kind"] == "route_start"]
    rds = route_ds(arm)
    w = int(meta["w"])
    rates = [e["ticks"] / e["wall_s"] for e in ends if e.get("ticks") and e.get("wall_s")]
    span = max(e["t"] for e in ends) - min(e["t"] for e in starts) if ends and starts else float("nan")
    ds_old = [r["ds"] for k, r in rds.items() if k in OLD20 and r["ds"] is not None]
    ds_all = [r["ds"] for r in rds.values() if r["ds"] is not None]
    row = dict(arm=name, kind=meta["kind"], t=meta.get("t"), w=w, card=job["gpu"], cores=len(_cpus(job["cpus"])),
               routes=meta.get("routes"), finished=len(rds), wall_min=round(state.get("wall_s", 0) / 60, 1),
               tps_saturated=round(w * statistics.mean(rates), 2) if rates else None,
               tps_makespan=round(sum(e["ticks"] for e in ends) / span, 2) if ends else None,
               worker_tps=round(statistics.mean(rates), 3) if rates else None,
               route_wall_median_s=round(statistics.median(e["wall_s"] for e in ends), 1) if ends else None,
               ds_old20=round(statistics.mean(ds_old), 2) if ds_old else None,
               ds_all=round(statistics.mean(ds_all), 2) if ds_all else None,
               attempts=sum(1 for _ in arm.glob("attempts/*/*/attempt.json")))
    row.update(start_failures(arm))
    row.update(util_window(util, job["gpu"], job["t0"], state.get("t1", job["t0"] + 1e9)))
    return row


def _cpus(spec: str) -> list:
    from jevdrive.cl.box import parse_cpus
    return parse_cpus(spec) if spec else []


def pairs(ds: dict) -> pd.DataFrame:
    rows = []
    for a, b in itertools.combinations(sorted(ds), 2):
        common = sorted(set(ds[a]) & set(ds[b]), key=int)
        for r in common:
            x, y = ds[a][r]["ds"], ds[b][r]["ds"]
            if x is None or y is None:
                continue
            kinds = "-".join(sorted((a.split(":")[0], b.split(":")[0])))
            rows.append(dict(a=a, b=b, kinds=kinds, route=r, d_ds=round(abs(x - y), 2), same=x == y))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(data_dir() / "runs/cl_lib/profile"))
    ap.add_argument("--old", nargs="*", default=[
        "stock:old-base-c=" + str(data_dir() / "runs/infra/carla-threads/routes/base-c"),
        "reduced:old-red-a=" + str(data_dir() / "runs/infra/carla-threads/routes/red-a"),
        "reduced:old-red-b=" + str(data_dir() / "runs/infra/carla-threads/routes/red-b")])
    a = ap.parse_args()
    root = Path(a.root)
    st = json.loads((root / "state.json").read_text())["jobs"]
    util = pd.read_csv(root / "util.csv")
    util["cores_busy"] = pd.to_numeric(util.cores_busy, errors="coerce")
    arms = pd.DataFrame([arm_row(root, n, s, util) for n, s in sorted(st.items()) if s["state"] == "done"
                         and (root / "arms" / n / "events.jsonl").exists()])
    out = root / "results"
    out.mkdir(exist_ok=True)
    arms.to_csv(out / "arms.csv", index=False)
    # DS agreement: "<kind>:<arm>" keys so pair kinds read stock-stock / reduced-stock / reduced-reduced
    ds = {"%s:%s" % (r.kind, r.arm): route_ds(root / "arms" / r.arm) for r in arms.itertuples() if r.arm[0] in "AB"}
    for spec in a.old:
        key, path = spec.split("=", 1)
        if Path(path, "done").exists():
            ds[key] = route_ds(Path(path))
    p = pairs(ds)
    p.to_csv(out / "pairs.csv", index=False)
    routes = pd.DataFrame({k: {r: v["ds"] for r, v in d.items()} for k, d in ds.items()})
    routes.index.name = "route"
    routes.to_csv(out / "routes.csv")
    lines = ["# Worker-profile experiment readout", "", "## Arms", "", arms.to_markdown(index=False), "",
             "## DS agreement by pair kind (all routes common to both arms)", ""]
    if len(p):
        new = p[~p.a.str.contains("old-") & ~p.b.str.contains("old-")]
        for title, q in (("this experiment", new), ("incl. 2026-09-27 runs", p)):
            g = q.groupby("kinds").agg(pairs=("same", "size"), ds_identical=("same", "mean"), mean_abs_dds=("d_ds", "mean"))
            lines += ["### " + title, "", g.round(3).to_markdown(), ""]
        flip = new[~new.same].groupby("route").size().sort_values(ascending=False)
        lines += ["Routes whose DS differs in at least one pair of this experiment (pairs): " +
                  ", ".join("%s (%d)" % (r, n) for r, n in flip.items()), ""]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    b = arms[arms.arm.str.startswith(("B-", "A-"))]
    if b.w.nunique() > 1:
        figure(b, out)


def figure(arms: pd.DataFrame, out: Path):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from jevdrive import plots
    with mpl.rc_context(plots.STYLE):
        fig, ax = plt.subplots(figsize=(plots.COL, 2.0))
        for kind, c in (("reduced", plots.OKABE_ITO[5]), ("stock", plots.OKABE_ITO[6])):
            q = arms[arms.kind == kind].groupby("w").tps_saturated.agg(["mean", "min", "max"]).reset_index()
            if not len(q):
                continue
            ax.errorbar(q.w, q["mean"], yerr=[q["mean"] - q["min"], q["max"] - q["mean"]], fmt="-o", color=c,
                        capsize=2, label=kind)
        ax.set_xlabel("CARLA workers per card (25-core slice)")
        ax.set_ylabel("sim ticks / s per card")
        ax.grid(True, axis="y")
        plots.legend_below(fig, ax)
        plots.save(fig, out, "workers_per_card")


if __name__ == "__main__":
    main()
