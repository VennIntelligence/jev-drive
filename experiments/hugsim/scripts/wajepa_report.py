"""Tables of results/wajepa_ref.md: WA-JEPA vs Cinque (PR #57) vs LTF vs cv on the exam's 64 scenarios, paired per scenario.
Reads results/hugsim-exam/scored_{op,base,wajepa}.csv, scenarios.csv and results/wajepa_ref/wajepa_extract.csv (scripts/wajepa_extract.py, box);
writes results/wajepa_ref/{tables.md,paired.csv}.
    .venv/bin/python experiments/hugsim/scripts/wajepa_report.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1] / "results"
sys.path.insert(0, str(R.parents[2]))
from jevdrive import stats  # noqa: E402

X = R / "hugsim-exam"
op, base, wj = (pd.read_csv(X / f"scored_{n}.csv") for n in ("op", "base", "wajepa"))
sc = pd.read_csv(X / "scenarios.csv").set_index("scenario")
ex = pd.read_csv(R / "wajepa_ref/wajepa_extract.csv")
spinners = [Path(p).stem for p in open(R.parents[0] / "scripts/derot_spin10.txt").read().split()]
ARMS = {"wajepa": wj[wj.tag == "wajepa"], "cinque": op[op.tag == "cinque-fixed"], "ltf": base[base.tag == "ltf-fixed"], "cv": base[base.tag == "cv-fixed"]}
ARMS = {k: v.set_index("scenario").sort_index() for k, v in ARMS.items()}
idx = ARMS["cinque"].index
for k, v in ARMS.items():
    assert list(v.index) == list(idx), (k, len(v))
exi = {t: ex[ex.tag == t].set_index("scenario").reindex(idx) for t in ("wajepa", "cinque-fixed")}
out = []
P = out.append


def cls(arm, tag):
    d, e = ARMS[arm], exi[tag]
    c = d["end"].astype(str).copy()
    c[e["spin"].astype(bool)] = "spin"
    return c.replace({"max_steps": "stuck", "bg_collision": "bg_coll", "fg_collision": "fg_coll"})


W, C = ARMS["wajepa"], ARMS["cinque"]
W["cls"], C["cls"] = cls("wajepa", "wajepa"), cls("cinque", "cinque-fixed")
grp = sc.loc[idx, "scene"].values
# 1. metrics
P("## T1 metrics on the 64 scenarios (mean over scenarios; PR #57 controller; one run each)\n")
P("| agent | HD-Score | NC | DAC | TTC | Comfort | RC | PDMS | complete | spin | stuck (max_steps) | bg coll | fg coll | off_route |\n|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
for k, d in ARMS.items():
    if k == "wajepa" or k == "cinque":
        c = d["cls"]
    else:
        c = d["end"].replace({"max_steps": "stuck", "bg_collision": "bg_coll", "fg_collision": "fg_coll"})
    n = lambda s: int((c == s).sum())
    P(f"| {k} | {d.hdscore.mean():.3f} | {d.nc.mean():.3f} | {d.dac.mean():.3f} | {d.ttc.mean():.3f} | {d.c.mean():.3f} | {d.rc.mean():.3f} | {d.pdms.mean():.3f} | {n('complete')} | {n('spin') if k in ('wajepa','cinque') else 'n/a'} | {n('stuck')} | {n('bg_coll')} | {n('fg_coll')} | {n('off_route')} |")
P("\nLTF / cv rows: spin column not recomputed here (controller_spin.md T1: 0 / 64 for both under PR #57); their end classes are the runner's.\n")
# 2. paired CI
P("## T2 paired difference in HD-Score (WA-JEPA minus other), bootstrap 95% CI (jevdrive.stats.paired, B = 10000)\n")
P("| versus | mean diff scenarios (unit = scenario) | mean diff scene clusters (unit = scene, ratio of sums) | WA better / worse / tie (|d| < 0.02) |\n|---|---|---|---|")
for k in ("cinque", "ltf", "cv"):
    d = W.hdscore.values - ARMS[k].hdscore.values
    a = stats.paired(W.hdscore.values, ARMS[k].hdscore.values)
    b = stats.paired(W.hdscore.values, ARMS[k].hdscore.values, groups=grp)
    P(f"| {k} | {stats.fmt(a)} | {stats.fmt(b)} ({b['units']} scenes) | {(d > .02).sum()} / {(d < -.02).sum()} / {(abs(d) <= .02).sum()} |")
# 3. strata
P("\n## T3 HD-Score by difficulty and by dataset (n per cell = 16)\n")
for key, lab in (("difficulty", "difficulty"), ("dataset", "dataset")):
    levels = ["easy", "medium", "hard", "extreme"] if key == "difficulty" else ["nuscenes", "kitti360", "waymo", "pandaset"]
    P(f"| {lab} | " + " | ".join(f"{k}" for k in ARMS) + " | WA - Cinque [95% CI] |\n|---|" + "--:|" * len(ARMS) + "---|")
    for lv in levels:
        m = (C[key] == lv).values
        a = stats.paired(W.hdscore.values[m], C.hdscore.values[m])
        P(f"| {lv} | " + " | ".join(f"{ARMS[k].hdscore.values[m].mean():.3f}" for k in ARMS) + f" | {stats.fmt(a)} |")
    P("")
# 4. interaction strata
s = sc.loc[idx]
P("## T4 by scenario content (interaction = the scenario has actors; ahead = an actor in front of the ego route)\n")
P("| stratum | n | WA-JEPA | Cinque | LTF | cv | WA - Cinque [95% CI] |\n|---|--:|--:|--:|--:|--:|---|")
strata = {"no actors (n_actors = 0)": s.n_actors.values == 0, "actors, none ahead": (s.n_actors.values > 0) & (s.n_ahead.values == 0),
          "actor ahead (n_ahead >= 1)": s.n_ahead.values >= 1, "medium + actors": (C.difficulty == "medium").values & (s.n_actors.values > 0),
          "medium + actor ahead": (C.difficulty == "medium").values & (s.n_ahead.values >= 1), "turning route (turn != straight)": s.turn.values != "straight"}
for name, m in strata.items():
    if m.sum() == 0:
        continue
    a = stats.paired(W.hdscore.values[m], C.hdscore.values[m])
    P(f"| {name} | {m.sum()} | " + " | ".join(f"{ARMS[k].hdscore.values[m].mean():.3f}" for k in ARMS) + f" | {stats.fmt(a)} |")
# share of the paired gap by difficulty
gap = W.hdscore.values - C.hdscore.values
P("\nShare of the total paired gap (sum of WA - Cinque over the 64) by difficulty: " + ", ".join(f"{lv} {gap[(C.difficulty == lv).values].sum() / gap.sum():.2f}" for lv in ("easy", "medium", "hard", "extreme")) + f" (sum {gap.sum():.2f}).\n")
# 5. per-scenario paired table
P("## T5 per scenario (sorted by WA - Cinque)\n")
P("| scenario | diff | tier | actors / ahead | WA HD | WA class | WA steps | Cinque HD | Cinque class | Cinque steps | LTF HD | cv HD |\n|---|---|---|---|--:|---|--:|--:|---|--:|--:|--:|")
tab = pd.DataFrame(dict(scenario=idx, d=gap, tier=C.difficulty.values, ds=C.dataset.values, act=[f"{a} / {b}" for a, b in zip(s.n_actors, s.n_ahead)], wh=W.hdscore.values, wc=W.cls.values, ws=W.steps.values,
                        ch=C.hdscore.values, cc=C.cls.values, cs=C.steps.values, lh=ARMS["ltf"].hdscore.values, vh=ARMS["cv"].hdscore.values))
tab.to_csv(R / "wajepa_ref/paired.csv", index=False)
for _, r in tab.sort_values("d").iterrows():
    P(f"| {r.scenario} | {r.d:+.3f} | {r.tier} | {r.act} | {r.wh:.3f} | {r.wc} | {r.ws} | {r.ch:.3f} | {r.cc} | {r.cs} | {r.lh:.3f} | {r.vh:.3f} |")
# 6. failure classes where Cinque fails and WA succeeds
P("\n## T6 where Cinque fails and WA-JEPA succeeds (Cinque HD < 0.5, WA-JEPA HD >= 0.5 and complete)\n")
m = (tab.ch < .5) & (tab.wh >= .5) & (tab.wc == "complete")
P(f"{int(m.sum())} scenarios. Cinque failure classes among them: " + ", ".join(f"{k} {v}" for k, v in tab[m].cc.value_counts().items()) + ".")
m2 = (tab.wh < .5) & (tab.ch >= .5) & (tab.cc == "complete")
P(f"The reverse (WA-JEPA HD < 0.5, Cinque HD >= 0.5 and complete): {int(m2.sum())} scenarios; WA-JEPA classes: " + ", ".join(f"{k} {v}" for k, v in tab[m2].wc.value_counts().items()) + ".\n")
P("Confusion of failure class, Cinque (rows) x WA-JEPA (columns), all 64:\n")
ct = pd.crosstab(tab.cc, tab.wc)
P(ct.to_markdown() + "\n")
# 7. spinner subset
P("## T7 the 10 PR #57 Cinque spinner scenarios\n")
sp = tab[tab.scenario.isin(spinners)].set_index("scenario")
E1, E2 = exi["wajepa"].loc[sp.index], exi["cinque-fixed"].loc[sp.index]
P("| scenario | Cinque HD | Cinque class | Cinque max heading err (deg) | WA HD | WA class | WA max heading err (deg) | WA v_max in first 10 s (m/s) | WA end speed | WA steps |\n|---|--:|---|--:|--:|---|--:|--:|--:|--:|")
for sname, r in sp.iterrows():
    P(f"| {sname} | {r.ch:.3f} | {r.cc} | {E2.loc[sname].max_abs_e:.0f} | {r.wh:.3f} | {r.wc} | {E1.loc[sname].max_abs_e:.0f} | {E1.loc[sname].v_max40:.1f} | {E1.loc[sname].v_end:.1f} | {r.ws} |")
P(f"\nSpinners: Cinque {int(E2.spin.astype(bool).sum())} / 10 spin (definition check), WA-JEPA {int(E1.spin.astype(bool).sum())} / 10. Mean HD on the 10: WA-JEPA {sp.wh.mean():.3f}, Cinque {sp.ch.mean():.3f}. "
  f"Launch stall (v_max over the first 40 steps < 1.6 m/s): WA-JEPA {int((E1.v_max40 < 1.6).sum())} / 10, Cinque {int((E2.v_max40 < 1.6).sum())} / 10; end = max_steps: WA-JEPA {int((sp.wc == 'stuck').sum())}, Cinque {int((sp.cc == 'stuck').sum())}.\n")
P("## T8 spin and launch behaviour on all 64 (heading error >= 60 deg, definition of controller_spin.md; launch stall = v_max over the first 40 steps < 1.6 m/s)\n")
P("| agent | spins >= 60 deg | >= 45 deg not recomputed | launch stalls | standing > 50% of steps | max_steps |\n|---|--:|--|--:|--:|--:|")
for nm, e, d in (("WA-JEPA", exi["wajepa"], W), ("Cinque (PR #57)", exi["cinque-fixed"], C)):
    P(f"| {nm} | {int(e.spin.astype(bool).sum())} | - | {int((e.v_max40 < 1.6).sum())} | {int((e.standing > .5).sum())} | {int((d.end == 'max_steps').sum())} |")
e = exi["wajepa"]
P(f"\nWA-JEPA fallbacks (inference exceptions, braking plan): {int(e.n_failures.sum())} steps in {int((e.n_failures > 0).sum())} scenarios. Padded-history steps per scenario: {e.n_hist_pad.min():.0f}-{e.n_hist_pad.max():.0f}.")
P("\nBACK-camera tile brightness (mean pixel at step 10, 0 = black), by dataset: " + ", ".join(f"{k} {v:.1f}" for k, v in e.assign(ds=C.dataset.values).groupby("ds").back_mean.mean().items()) + "; front tile: " + ", ".join(f"{k} {v:.1f}" for k, v in e.assign(ds=C.dataset.values).groupby("ds").front_mean.mean().items()) + ".")
(R / "wajepa_ref/tables.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
