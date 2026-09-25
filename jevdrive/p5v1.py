"""P5 v1: the CARLA counterfactual pair exam with a second expert and ~100 base routes
(todos/2026-09-25-reactivity-program/i1-p5v1.md). Everything below reuses P5 v0 (jevdrive/p5_pairs.py,
jevdrive/p5_exam.py) unchanged: the variant XML, the recorder, the determinism check, the observation frames, the
expert labels and the label-validity table. What is new is only which routes, which expert, and where things go.

  build     v1 route set = bench2drive220 + bench2drive_0.0.4_val, same 10 families, same crasher list, duplicates
            dropped -> runs/p5v1/pairs.xml, research/results/p5-v1/cases.csv
  ids       the variant ids one expert still has to drive (comma list, for scripts/p5v1_gen.sh)
  link-v0   BehaviorAgent on the v0 routes is v0's own generation: link those attempts into runs/p5v1/gen-ba
  index     per expert, the v0 pipeline (pairs, observation frames, labels, frame index) -> processed/carla_p5v1_<e>
  validity  per expert label-validity table (p5_exam.exam, no examinees) and the two experts on the same pairs

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


def _pair_view(expert: str, tau: float) -> pd.DataFrame:
    """One row per case: determinism outcome and the expert's reaction on its own observation frames."""
    use(expert)
    d = P.processed()
    pairs = pd.read_csv(d / "pairs.csv", dtype={"base_id": str})
    obs = pd.read_parquet(d / "obs.parquet")
    obs = obs.assign(reactive=np.abs(obs.d_expert) > tau)
    rows = []
    for (b, s), o in obs.groupby(["base_id", "seed"]):
        r = o[o.reactive]
        pr = pairs[(pairs.base_id == b) & (pairs.seed == s)].iloc[0]
        rows.append({"base_id": b, "seed": s, "n_obs": len(o), "n_reactive": len(r),
                     "lead_s": (r.k.min() - pr.t_vis) * P.TICK if len(r) else np.nan,
                     "d_min": float(o.d_expert.min()), "d_reactive_median": float(r.d_expert.median()) if len(r) else np.nan})
    view = pairs[["base_id", "seed", "family", "reason"]].merge(pd.DataFrame(rows), on=["base_id", "seed"], how="left")
    return view.fillna({"n_obs": 0, "n_reactive": 0})


def validity():
    from . import p5_exam as E
    out, taus = {}, {}
    for e in EXPERTS:
        use(e)
        d = P.processed()
        if not (d / "obs.parquet").exists():
            log.info("no index for %s yet", e)
            continue
        obs, null = pd.read_parquet(d / "obs.parquet"), pd.read_parquet(d / "null.parquet")
        pairs = pd.read_csv(d / "pairs.csv", dtype={"base_id": str})
        res = E.exam(obs, null, pairs, [])
        v = res["validity"]
        pooled = v[v.in_pooled]
        summary = {"expert": e, "tau_exp": res["tau_exp"],
                   "null_p95": float(np.quantile(np.abs(null.d_expert), 0.95)) if len(null) else np.nan,
                   "pairs": int(v.pairs.sum()), "deterministic": int(v.deterministic_to_visibility.sum()),
                   "obs_frames": int(v.obs_frames.sum()), "reactive_frames": int(v.reactive_frames.sum()),
                   "pooled_families": ",".join(res["pooled_families"]),
                   "pooled_reactive_frames": int(pooled.reactive_frames.sum()),
                   "pooled_reactive_share": float(pooled.reactive_frames.sum() / max(pooled.obs_frames.sum(), 1)),
                   "all_reactive_share": float(v.reactive_frames.sum() / max(v.obs_frames.sum(), 1))}
        v.to_csv(RESULTS / f"label_validity_{e}.csv", index=False)
        log.info("%s: tau %.3f\n%s", e, res["tau_exp"], v.to_markdown(index=False))
        out[e], taus[e] = summary, res["tau_exp"]
    pd.DataFrame(out.values()).to_csv(RESULTS / "summary.csv", index=False)
    if len(out) < 2:
        return
    a, b = (_pair_view(e, taus[e]) for e in EXPERTS)
    m = a.merge(b, on=["base_id", "seed", "family"], suffixes=("_ba", "_pdm"))
    m.to_csv(RESULTS / "experts_pairs.csv", index=False)
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
    cmp_ = pd.DataFrame(rows)
    cmp_.to_csv(RESULTS / "experts_compare.csv", index=False)
    log.info("experts on the same pairs\n%s", cmp_.to_markdown(index=False))


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
