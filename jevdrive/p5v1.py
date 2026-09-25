"""P5 v1: the CARLA counterfactual pair exam with a second expert and ~100 base routes
(todos/2026-09-25-reactivity-program/i1-p5v1.md). Everything below reuses P5 v0 (jevdrive/p5_pairs.py,
jevdrive/p5_exam.py) unchanged: the variant XML, the recorder, the determinism check, the observation frames, the
expert labels and the label-validity table. What is new is only which routes, which expert, and where things go.

  build     v1 route set = bench2drive220 + bench2drive_0.0.4_val, same 10 families, same crasher list, duplicates
            dropped -> runs/p5v1/pairs.xml, research/results/p5-v1/cases.csv
  ids       the variant ids one expert still has to drive (comma list, for scripts/p5v1_gen.sh)
  link-v0   BehaviorAgent on the v0 routes is v0's own generation: link those attempts into runs/p5v1/gen-ba
  index     per expert, the v0 pipeline (pairs, observation frames, labels, frame index) -> processed/carla_p5v1_<e>
  validity  per expert label-validity table (p5_exam.exam, no examinees; PDM-Lite also in v0's recording window) and
            the two experts on the same pairs

Experts: "ba" = BehaviorAgent(normal) in the official Bench2Drive 0.0.4 tree (exactly v0), "pdm" = PDM-Lite in
SimLingo's Bench2Drive copy.
"""
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from . import p5_pairs as P
from .common import data_dir, get_logger

log = get_logger(__name__)
REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "research" / "results" / "p5-v1"
SOURCES = ("third_party/Bench2Drive/leaderboard/data/bench2drive220.xml",
           "third_party/Bench2Drive/leaderboard/data/bench2drive_0.0.4_val.xml")
EXPERTS = ("ba", "pdm")
V0_GEN = "runs/p5_pairs/gen"


def root(*parts) -> Path:
    p = data_dir() / "runs" / "p5v1" / Path(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def gen(expert: str) -> Path:
    return root("gen-" + expert)


def cases() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "cases.csv", dtype={"base_id": str, "plus": str, "minus": str, "null": str})


def build():
    c = P.build([data_dir() / s for s in SOURCES], xml=root() / "pairs.xml", results=RESULTS)
    worlds = 2 * len(c) + int((c.null.fillna("") != "").sum())
    tab = c.groupby(["family", "source"]).base_id.nunique().unstack(fill_value=0)
    tab["pairs"] = c.groupby("family").size()
    log.info("%d base routes, %d pairs, %d worlds per expert\n%s", c.base_id.nunique(), len(c), worlds,
             tab.to_markdown())
    return c


def variants(c: pd.DataFrame) -> list[str]:
    """Every world of every case, grouped by base route (consecutive runs share a town)."""
    out = []
    for _, g in c.sort_values(["town", "base_id", "seed"]).groupby(["town", "base_id"], sort=False):
        for _, r in g.iterrows():
            out += [r.plus, r.minus] + ([r.null] if isinstance(r.null, str) and r.null else [])
    return out


def ids(expert: str):
    g = gen(expert)
    todo = [v for v in variants(cases()) if not (g / "done" / (v + ".json")).exists()]
    print(",".join(todo))


def link_v0():
    """BehaviorAgent on a v0 route = v0's own run: same XML variant (checked element by element), same agent code
    path, same tree. Link the attempts and copy done/ so b2d_run skips them."""
    v0 = data_dir() / V0_GEN
    old = {r.get("id"): ET.tostring(r).strip() for r in ET.parse(data_dir() / "runs/p5_pairs/pairs.xml").getroot()}
    new = {r.get("id"): ET.tostring(r).strip() for r in ET.parse(root() / "pairs.xml").getroot()}
    g = gen("ba")
    (g / "done").mkdir(exist_ok=True)
    (g / "attempts").mkdir(exist_ok=True)
    n = 0
    for v in variants(cases()):
        if v not in old or not (v0 / "done" / (v + ".json")).exists():
            continue
        assert old[v] == new[v], "variant %s differs from v0's XML" % v
        dst = g / "attempts" / v
        if not dst.exists():
            dst.symlink_to(v0 / "attempts" / v)
        rec = json.loads((v0 / "done" / (v + ".json")).read_text())
        rec["reused_from"] = str(v0)
        (g / "done" / (v + ".json")).write_text(json.dumps(rec, indent=2))
        n += 1
    log.info("linked %d v0 runs into %s", n, g)


def use(expert: str):
    """Point the v0 pipeline at one expert's v1 set."""
    os.environ["P5_SET"] = "carla_p5v1_" + expert
    P.RESULTS = RESULTS


def index(expert: str):
    use(expert)
    P.index(gen(expert))


# v0's recorder stop rules (scripts/p5_pair_agent.py DEFAULT). PDM-Lite runs record longer (deviation 3 in the todo);
# for the two-expert comparison their observation frames are cut back to what v0's rules would have recorded.
V0_STOP = dict(max_sim_s=50.0, after_trigger_s=20.0, stuck_s=30.0)
FUTURE_TICKS = 100                         # the 5 s future every observation frame needs


def _v0_end(adir: Path) -> int:
    """The last tick this world would have recorded under v0's stop rules (max_sim_s, trigger + after_trigger_s,
    standing still longer than stuck_s after t = 10 s), from its own pose and trigger time."""
    pose = pd.read_json(adir / "pose.jsonl", lines=True).drop_duplicates("frame")
    t_trig = json.loads((adir / "p5_summary.json").read_text()).get("t_trigger")
    end = V0_STOP["max_sim_s"] if t_trig is None else min(V0_STOP["max_sim_s"], t_trig + V0_STOP["after_trigger_s"])
    since = None
    for t, vx, vy in zip(pose.t, pose.vx, pose.vy):
        since = (since if since is not None else t) if np.hypot(vx, vy) < 0.2 else None
        if since is not None and t - since > V0_STOP["stuck_s"] and t > 10:
            end = min(end, t)
            break
    return int(round(end / P.TICK))


def v0_window(expert: str, obs: pd.DataFrame, null: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Observation and null frames whose 5 s future lies inside what both worlds would have recorded under v0's rules."""
    c = cases().set_index(["base_id", "seed"])
    g, ends = gen(expert), {}

    def end(rid):
        if rid not in ends:
            a = P.attempt(g, rid)
            ends[rid] = _v0_end(a) if a is not None else -1
        return ends[rid]

    def keep(df, other):
        if not len(df):
            return df
        lim = [min(end(c.loc[(b, s), "plus"]), end(c.loc[(b, s), other])) for b, s in zip(df.base_id, df.seed)]
        return df[df.k.to_numpy() + FUTURE_TICKS <= np.array(lim)]

    return keep(obs, "minus"), keep(null, "null")


def load(expert: str):
    use(expert)
    d = P.processed()
    return (pd.read_parquet(d / "obs.parquet"), pd.read_parquet(d / "null.parquet"),
            pd.read_csv(d / "pairs.csv", dtype={"base_id": str}))


def _pair_view(pairs: pd.DataFrame, obs: pd.DataFrame, tau: float) -> pd.DataFrame:
    """One row per case: determinism outcome and the expert's reaction on its own observation frames."""
    obs = obs.assign(reactive=np.abs(obs.d_expert) > tau)
    rows = []
    for (b, s), o in obs.groupby(["base_id", "seed"]):
        r = o[o.reactive]
        pr = pairs[(pairs.base_id == b) & (pairs.seed == s)].iloc[0]
        rows.append({"base_id": b, "seed": s, "n_obs": len(o), "n_reactive": len(r),
                     "lead_s": (r.k.min() - pr.t_vis) * P.TICK if len(r) else np.nan,
                     "d_min": float(o.d_expert.min()), "d_reactive_median": float(r.d_expert.median()) if len(r) else np.nan})
    view = pairs[["base_id", "seed", "family", "reason"]].merge(pd.DataFrame(rows, columns=[
        "base_id", "seed", "n_obs", "n_reactive", "lead_s", "d_min", "d_reactive_median"]), on=["base_id", "seed"], how="left")
    return view.fillna({"n_obs": 0, "n_reactive": 0})


def _validity_one(tag: str, obs, null, pairs) -> tuple[dict, float]:
    from . import p5_exam as E
    res = E.exam(obs, null, pairs, [])
    v = res["validity"]
    pooled = v[v.in_pooled]
    v.to_csv(RESULTS / f"label_validity_{tag}.csv", index=False)
    log.info("%s: tau %.3f\n%s", tag, res["tau_exp"], v.to_markdown(index=False))
    return {"set": tag, "tau_exp": res["tau_exp"],
            "null_p95": float(np.quantile(np.abs(null.d_expert), 0.95)) if len(null) else np.nan,
            "pairs": int(v.pairs.sum()), "deterministic": int(v.deterministic_to_visibility.sum()),
            "obs_frames": int(v.obs_frames.sum()), "reactive_frames": int(v.reactive_frames.sum()),
            "pooled_families": ",".join(res["pooled_families"]),
            "pooled_reactive_frames": int(pooled.reactive_frames.sum()),
            "pooled_reactive_share": float(pooled.reactive_frames.sum() / max(pooled.obs_frames.sum(), 1)),
            "all_reactive_share": float(v.reactive_frames.sum() / max(v.obs_frames.sum(), 1))}, res["tau_exp"]


def compare(a: pd.DataFrame, b: pd.DataFrame, tag: str) -> pd.DataFrame:
    """The two experts on the same pairs (pairs deterministic under both), per family."""
    m = a.merge(b, on=["base_id", "seed", "family"], suffixes=("_ba", "_pdm"))
    m.to_csv(RESULTS / f"experts_pairs_{tag}.csv", index=False)
    rows = []
    for fam, g in [("all", m)] + list(m.groupby("family")):
        both = g[(g.reason_ba == "ok") & (g.reason_pdm == "ok")]
        ra, rb = both.n_reactive_ba > 0, both.n_reactive_pdm > 0
        rows.append({"family": fam, "pairs": len(g), "ok_ba": int((g.reason_ba == "ok").sum()),
                     "ok_pdm": int((g.reason_pdm == "ok").sum()), "ok_both": len(both),
                     "reactive_ba": int(ra.sum()), "reactive_pdm": int(rb.sum()), "reactive_both": int((ra & rb).sum()),
                     "reactive_only_ba": int((ra & ~rb).sum()), "reactive_only_pdm": int((~ra & rb).sum()),
                     "frames_reactive_ba": int(both.n_reactive_ba.sum()), "frames_reactive_pdm": int(both.n_reactive_pdm.sum()),
                     "lead_s_ba": float(both.lead_s_ba.median()), "lead_s_pdm": float(both.lead_s_pdm.median()),
                     "d_reactive_ba": float(both.d_reactive_median_ba.median()),
                     "d_reactive_pdm": float(both.d_reactive_median_pdm.median()),
                     "d_min_ba": float(both.d_min_ba.median()), "d_min_pdm": float(both.d_min_pdm.median())})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / f"experts_compare_{tag}.csv", index=False)
    log.info("experts on the same pairs (%s)\n%s", tag, out.to_markdown(index=False))
    return out


def validity():
    """Label validity per expert (PDM-Lite also cut back to v0's recording window), and the two experts on the same
    pairs: PDM-Lite in v0's window (the like-for-like comparison) and in its full window."""
    rows, views = [], {}
    for e in EXPERTS:
        use(e)
        if not (P.processed() / "obs.parquet").exists():
            log.info("no index for %s yet", e)
            continue
        obs, null, pairs = load(e)
        sets = [(e, obs, null)]
        if e == "pdm":
            sets.append(("pdm_v0window",) + v0_window(e, obs, null))
        for tag, o, n in sets:
            row, tau = _validity_one(tag, o, n, pairs)
            rows.append(row)
            views[tag] = _pair_view(pairs, o, tau)
    pd.DataFrame(rows).to_csv(RESULTS / "summary.csv", index=False)
    if "ba" in views and "pdm" in views:
        compare(views["ba"], views["pdm_v0window"], "v0window")
        compare(views["ba"], views["pdm"], "full")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["build", "ids", "link-v0", "index", "validity"])
    p.add_argument("--expert", choices=EXPERTS)
    a = p.parse_args()
    if a.cmd == "build":
        build()
    elif a.cmd == "ids":
        ids(a.expert)
    elif a.cmd == "link-v0":
        link_v0()
    elif a.cmd == "index":
        for e in [a.expert] if a.expert else EXPERTS:
            index(e)
    else:
        validity()


if __name__ == "__main__":
    main()
