"""Figures and the flat result table of the CARLA harness scaling ladders (scripts/infra_scale.sh).

Input: <runs>/<step>/rungs.jsonl as written by scripts/b2d_scale.py, pulled from the box.
Output: research/results/infra-acceptance/profiling/rungs.csv and research/figs/infra-scale-*.{pdf,png}.

    .venv/bin/python scripts/make_infra_scale_figs.py --runs <local copy of $DATA_DIR/runs/infra-acceptance/scale>
"""
import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from jevdrive.plots import PAGE, STYLE, save

SERIES = {  # step -> (label, colour, line style)
    "t12-alp": ("Town12, Alpamayo rig", "#0072B2", "-"),
    "t03-alp": ("Town03, Alpamayo rig", "#009E73", "-"),
    "t13-t8": ("Town13, Alpamayo rig", "#D55E00", "-"),
    "t13-lights-t8": ("Town13, Alpamayo rig, cached lights", "#E69F00", "--"),
    "t12-op-t8": ("Town12, openpilot+TCP rig (shared GPU)", "#CC79A7", "-"),
    "t12-alp-res64": ("Town12, Alpamayo rig, 64x64 viewport", "#56B4E9", "--"),
    "t12-base6": ("Town12, Alpamayo rig (GPU 0 baseline)", "#000000", "none"),
    "t12-t8": ("Town12, Alpamayo rig, 8 client threads", "#7F7F7F", "none"),
}


def load(runs: Path) -> pd.DataFrame:
    rows = []
    for step in SERIES:
        f = runs / step / "rungs.jsonl"
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            r = json.loads(line)
            c = r.get("cores", {})
            g = r.get(f"gpu{r['gpus'][0]}", {})
            if not r.get("agg_ticks_s"):
                continue   # no steady window (every worker must tick at once; see b2d_scale --recompute)
            # a rung whose servers crashed in setup is measured on its survivors: plot it at that many servers
            eff = r.get("workers_effective", r["workers"])
            rows.append(dict(
                step=step, gpus=len(r["gpus"]), servers_per_gpu=eff // len(r["gpus"]), workers=eff,
                servers_started=r["workers"],
                window_s=r["window_s"], agg_ticks_s=r.get("agg_ticks_s"), ms_per_tick_worker=r.get("ms_per_tick_worker"),
                worker_ticks_s_min=r.get("worker_ticks_s_min"), worker_ticks_s_max=r.get("worker_ticks_s_max"),
                server_cores_each=c.get("server", {}).get("cores_each"), route_cores_each=c.get("route", {}).get("cores_each"),
                server_rss_gb=c.get("server", {}).get("rss_gb_each"), route_rss_gb=c.get("route", {}).get("rss_gb_each"),
                server_threads=c.get("server", {}).get("threads_each"), route_threads=c.get("route", {}).get("threads_each"),
                cores_mine=r.get("cores_mine"), cores_container=r.get("cores_container"),
                cores_background=r.get("cores_background"), throttled_s_per_s=r.get("throttled_s_per_s"),
                load1=r.get("load1_mean"), gpu_util=g.get("util_mean"), vram_gb=g.get("mem_gb_mean"),
                vram_gb_max=g.get("mem_gb_max"), setup_s=r.get("setup_s_median"), routes_capped=r.get("routes_capped"),
                total_ms=r.get("total_ms_mean"), world_tick_ms=r.get("world_tick_ms_mean"), tree_ms=r.get("tree_ms_mean"),
                agent_ms=r.get("agent_ms_mean"), cpus=r.get("cpus")))
    return pd.DataFrame(rows)


def throughput(df: pd.DataFrame, out: Path):
    with mpl.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(PAGE, 2.1))
        for step, (label, color, ls) in SERIES.items():
            d = df[(df.step == step) & (df.gpus == 1)].sort_values("servers_per_gpu")
            if not len(d):
                continue
            axes[0].plot(d.servers_per_gpu, d.agg_ticks_s, linestyle=ls, marker="o", color=color, label=label)
            one = d[d.servers_per_gpu == 1]
            if len(one):
                n = d.servers_per_gpu
                axes[0].plot(n, one.agg_ticks_s.iloc[0] * n, ":", color=color, lw=0.6)
            axes[1].plot(d.servers_per_gpu, d.ms_per_tick_worker, linestyle=ls, marker="o", color=color, label=label)
        axes[0].set(xlabel="CARLA servers on one GPU", ylabel="aggregate ticks/s")
        axes[1].set(xlabel="CARLA servers on one GPU", ylabel="per-worker ms/tick")
        for ax in axes:
            ax.grid(True)
            ax.set_ylim(bottom=0)
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        h, l = axes[1].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=3)
        save(fig, out, "infra-scale-throughput")


def resources(df: pd.DataFrame, out: Path):
    with mpl.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(PAGE, 2.0))
        fig.subplots_adjust(wspace=0.38)
        for step, (label, color, ls) in SERIES.items():
            d = df[(df.step == step) & (df.gpus == 1)].sort_values("servers_per_gpu")
            if not len(d):
                continue
            kw = dict(linestyle=ls, color=color)
            # CPU per simulated tick: cores x workers / aggregate ticks/s
            axes[0].plot(d.servers_per_gpu, d.server_cores_each * d.workers / d.agg_ticks_s, marker="o", label=label, **kw)
            axes[0].plot(d.servers_per_gpu, d.route_cores_each * d.workers / d.agg_ticks_s, marker="^", alpha=0.6, **kw)
            axes[1].plot(d.servers_per_gpu, d.cores_mine, marker="o", **kw)
            axes[2].plot(d.servers_per_gpu, d.gpu_util, marker="o", **kw)
        axes[0].set(xlabel="CARLA servers on one GPU", ylabel="core-seconds per tick")
        axes[0].text(0.97, 0.43, "circles: CARLA server\ntriangles: route client", transform=axes[0].transAxes,
                     va="top", ha="right", fontsize=6)
        axes[1].set(xlabel="CARLA servers on one GPU", ylabel="cores used by the ladder")
        axes[2].set(xlabel="CARLA servers on one GPU", ylabel="GPU utilisation (%)", ylim=(0, 100))
        for ax in axes:
            ax.grid(True)
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        axes[0].set_ylim(bottom=0)
        axes[1].set_ylim(bottom=0)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=3)
        save(fig, out, "infra-scale-resources")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True)
    p.add_argument("--figs", default="research/figs")
    p.add_argument("--results", default="research/results/infra-acceptance/profiling")
    a = p.parse_args()
    df = load(Path(a.runs))
    res = Path(a.results)
    res.mkdir(parents=True, exist_ok=True)
    df.to_csv(res / "rungs.csv", index=False)
    throughput(df, Path(a.figs))
    resources(df, Path(a.figs))
    print(df[["step", "gpus", "servers_per_gpu", "agg_ticks_s", "ms_per_tick_worker", "server_cores_each",
              "route_cores_each", "cores_mine", "gpu_util", "vram_gb", "cores_background"]].to_string(index=False))


if __name__ == "__main__":
    main()
