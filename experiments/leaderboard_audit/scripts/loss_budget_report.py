"""Loss budget, final step: render results/loss_budget.md from the four board JSONs (navtest, navhard, hugsim, b2d).

  .venv/bin/python experiments/leaderboard_audit/scripts/loss_budget_report.py
"""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"
J = {b: json.load(open(R / "loss_budget" / f"{b}.json")) for b in ("navtest", "navhard", "hugsim", "b2d")}


def f(x, nd=1):
    return "n/a" if x is None else f"{x:.{nd}f}"


def pct(x):
    return "n/a" if x is None else f"{100 * x:.0f}%"


def diff(a, b, nd=1):
    return "n/a" if a is None or b is None else f"{a - b:+.{nd}f}"


NAV_LABEL = {"wrong_direction": "wrong direction (decision 88)", "road_edge_wide": "off-road, model road edge too wide (decision 88; shipped only)",
             "early_clip": "early clip (decision 88)", "collisions": "collisions (NC < 1 or TTC < 1)", "ep_progress": "EP / progress (EP set to 1)",
             "comfort": "comfort (HC, EC set to 1)", "stopped_slow": "stopped or too slow", "no_command": "turn missed, no command (desire off)",
             "offroad_all": "context: every DAC failure (superset of the three rows above it)", "other_gate": "context: DDC / TLC failures",
             "ep_at_ref": "context: EP raised only to the reference token's EP"}
ORDER = ["wrong_direction", "road_edge_wide", "early_clip", "offroad_all", "collisions", "ep_progress", "ep_at_ref", "comfort", "stopped_slow", "no_command", "other_gate"]


def nav_table(b):
    r = J[b]
    s, c = r["native"], r["best"]
    out = [f"| class | share of failing tokens (shipped) | ceiling, shipped ({f(s['score'], 2)}) | ceiling, best ({f(c['score'], 2)}) | recovered so far by best (shipped minus best ceiling) |",
           "|:--|--:|--:|--:|--:|"]
    for k in ORDER:
        a, bb = s["rows"][k], c["rows"][k]
        out.append(f"| {NAV_LABEL[k]} | {pct(a['share'])}{'' if a['n'] is None else ' (' + str(a['n']) + ')'} | {f(a['ceiling'], 2)} | {f(bb['ceiling'], 2)} | {diff(a['ceiling'], bb['ceiling'], 2)} |")
    out.append(f"| **all trajectory classes together** (wrong direction, road edge, early clip, collisions, stopped, no command; replaced by the reference) | - | {f(s['all_traj_only'], 2)} | {f(c['all_traj_only'], 2)} | {diff(s['all_traj_only'], c['all_traj_only'], 2)} |")
    out.append(f"| **everything together** (the above + EP set to 1 + comfort set to 1) | - | {f(s['all'], 2)} | {f(c['all'], 2)} | {diff(s['all'], c['all'], 2)} |")
    return "\n".join(out)


def hugsim_table():
    h = J["hugsim"]
    s, c = h["cinque:fixed"], h["it_dw3+sel3"]
    names = ["stopped / max_steps", "collision: foreground (fg)", "spin (heading > 45 deg off route)", "collision: background (bg)", "route-end crash (collision at RC >= 0.9)",
             "off-road / off-route", "complete (HD < 1)"]
    out = [f"| class | scenarios, shipped (share of failures) | ceiling to best observed (our arms / + LTF, cv, Lebowski), shipped HD {f(s['hd_mean'])} | ceiling to HD = 1, shipped | scenarios, best | ceiling to best observed (our / all), best HD {f(c['hd_mean'])} | ceiling to HD = 1, best | recovered so far (HD = 1 ceiling, shipped minus best) |",
           "|:--|--:|--:|--:|--:|--:|--:|--:|"]
    z = dict(n=0, share_of_failures=0, ceiling_best_ours=None, ceiling_best_any=None, ceiling_one=None)
    for n in names:
        a, b = s["rows"].get(n, z), c["rows"].get(n, z)
        if a["n"] == 0 and b["n"] == 0:
            continue
        a1, b1 = (a["ceiling_one"] or 0.0), (b["ceiling_one"] or 0.0)
        out.append(f"| {n} | {a['n']} ({pct(a['share_of_failures'])}) | {f(a['ceiling_best_ours'] or 0)} / {f(a['ceiling_best_any'] or 0)} | {f(a1)} | {b['n']} | "
                   f"{f(b['ceiling_best_ours'] or 0)} / {f(b['ceiling_best_any'] or 0)} | {f(b1)} | {a1 - b1:+.1f} |")
    out.append(f"| **all failing scenarios together** | {s['n_failing']} | {f(s['all_best_ours'])} / {f(s['all_best_any'])} | {f(s['all_one'])} | {c['n_failing']} | {f(c['all_best_ours'])} / {f(c['all_best_any'])} | {f(c['all_one'])} | {s['all_one'] - c['all_one']:+.1f} |")
    return "\n".join(out)


B_LABEL = {"red_light": "red light", "stop_sign": "stop sign", "collision_vehicle": "collision with vehicle", "collision_layout": "collision with layout (static)",
           "collision_pedestrian": "collision with pedestrian", "route_deviation": "route deviation (outside route lanes)",
           "blocked_timeout": "blocked / timeout (Agent got blocked, TickRuntime cap): run completes", "collision_all": "all collision classes together"}


def b2d_table():
    d, v = J["b2d"]["drive"], J["b2d"]["vmerge2"]
    out = [f"| class | share of failing runs, drive (events) | ceiling, drive (DS {f(d['ds'], 2)}) | ceiling, drive, exposure-aware | share, vmerge2 (events) | ceiling, vmerge2 (DS {f(v['ds'], 2)}) | ceiling, vmerge2, exposure-aware | recovered so far (drive minus vmerge2 ceiling) |",
           "|:--|--:|--:|--:|--:|--:|--:|--:|"]
    for k in B_LABEL:
        a, b = d["rows"][k], v["rows"][k]
        out.append(f"| {B_LABEL[k]} | {pct(a['share_of_failing_runs'])} ({a['events']}) | {f(a['ceiling'], 2)} | {f(a.get('ceiling_exposure_aware'), 2)} | {pct(b['share_of_failing_runs'])} ({b['events']}) | "
                   f"{f(b['ceiling'], 2)} | {f(b.get('ceiling_exposure_aware'), 2)} | {diff(a['ceiling'], b['ceiling'], 2)} |")
    out.append(f"| **everything removed** (DS = 100) | {d['n_failing_runs']} failing runs of 76 | {f(d['all_removed'], 2)} | - | {v['n_failing_runs']} failing runs of 76 | {f(v['all_removed'], 2)} | - | {diff(d['all_removed'], v['all_removed'], 2)} |")
    return "\n".join(out)


def summary():
    ns, nh, h, b = J["navtest"]["native"]["rows"], J["navhard"]["native"]["rows"], J["hugsim"]["cinque:fixed"]["rows"], J["b2d"]["drive"]["rows"]
    nb, hb, bb = J["navtest"]["best"]["rows"], J["navhard"]["best"]["rows"], J["b2d"]["vmerge2"]["rows"]
    hc = J["hugsim"]["it_dw3+sel3"]["rows"]
    g = lambda d, k, key="ceiling_best_ours": (d.get(k) or {}).get(key, 0.0) or 0.0  # noqa: E731
    lev = [
        ("stopped / too slow / blocked", ns["stopped_slow"]["ceiling"], nh["stopped_slow"]["ceiling"], g(h, "stopped / max_steps"), b["blocked_timeout"]["ceiling"],
         nb["stopped_slow"]["ceiling"], hb["stopped_slow"]["ceiling"], g(hc, "stopped / max_steps"), bb["blocked_timeout"]["ceiling"]),
        ("off-road (every DAC failure; B2D: route deviation)", ns["offroad_all"]["ceiling"], nh["offroad_all"]["ceiling"], 0.0, b["route_deviation"]["ceiling"],
         nb["offroad_all"]["ceiling"], hb["offroad_all"]["ceiling"], g(hc, "off-road / off-route"), bb["route_deviation"]["ceiling"]),
        ("collisions (HUGSIM fg + bg + route-end; B2D all classes)", ns["collisions"]["ceiling"], nh["collisions"]["ceiling"],
         g(h, "collision: foreground (fg)") + g(h, "collision: background (bg)") + g(h, "route-end crash (collision at RC >= 0.9)"), b["collision_all"]["ceiling"],
         nb["collisions"]["ceiling"], hb["collisions"]["ceiling"], g(hc, "collision: foreground (fg)") + g(hc, "collision: background (bg)") + g(hc, "route-end crash (collision at RC >= 0.9)"), bb["collision_all"]["ceiling"]),
        ("EP / progress (EP raised to the reference's)", ns["ep_at_ref"]["ceiling"], nh["ep_at_ref"]["ceiling"], None, None, nb["ep_at_ref"]["ceiling"], hb["ep_at_ref"]["ceiling"], None, None),
        ("red light / stop sign (B2D); DDC + TLC (navhard)", ns["other_gate"]["ceiling"], nh["other_gate"]["ceiling"], None, b["red_light"]["ceiling"] + b["stop_sign"]["ceiling"],
         nb["other_gate"]["ceiling"], hb["other_gate"]["ceiling"], None, bb["red_light"]["ceiling"] + bb["stop_sign"]["ceiling"]),
        ("spin / wrong direction", ns["wrong_direction"]["ceiling"], nh["wrong_direction"]["ceiling"], g(h, "spin (heading > 45 deg off route)"), None,
         nb["wrong_direction"]["ceiling"], hb["wrong_direction"]["ceiling"], g(hc, "spin (heading > 45 deg off route)"), None),
        ("comfort", ns["comfort"]["ceiling"], nh["comfort"]["ceiling"], None, None, nb["comfort"]["ceiling"], hb["comfort"]["ceiling"], None, None),
        ("turn missed, no command (desire off)", ns["no_command"]["ceiling"], nh["no_command"]["ceiling"], None, None, nb["no_command"]["ceiling"], hb["no_command"]["ceiling"], None, None),
    ]
    tot = lambda row, o: sum(x for x in row[o:o + 4] if x is not None)  # noqa: E731
    lev.sort(key=lambda r: -tot(r, 1))
    out = ["| rank | lever | navtest | navhard | HUGSIM 64 | B2D 19 routes | sum over boards (reference driver) | sum over boards (current best) |", "|--:|:--|--:|--:|--:|--:|--:|--:|"]
    for i, r in enumerate(lev, 1):
        cells = []
        for j in range(4):
            a, c = r[1 + j], r[5 + j]
            cells.append("n/a" if a is None else f"{a:.1f} ({c:.1f})" if c is not None else f"{a:.1f}")
        out.append(f"| {i} | {r[0]} | " + " | ".join(cells) + f" | {tot(r, 1):.1f} | {tot(r, 5):.1f} |")
    return "\n".join(out), lev


def main():
    sm, lev = summary()
    nt, nh, hg, bd = J["navtest"], J["navhard"], J["hugsim"], J["b2d"]
    ns, nn = nt["native"], nh["native"]
    md = f"""# Loss budget: how many points come back if one failure class were fixed perfectly

2026-10-04. **Every number below is an oracle ceiling, not an achievable gain.** A class is fixed by substituting the reference, a better arm, or a
term value of 1 on its own failing cases, then re-scoring with the board's scorer; nothing is predicted and no new driving was run. Points are the board's own scale
(navtest PDMS, navhard two-stage EPDMS, HUGSIM mean HD x 100, B2D mean DS); they are not comparable across boards. Reference driver = shipped Cinque; "current best" = it_dw3-s0 +
selector (decision 101) on navtest / navhard / HUGSIM, and `vmerge2` on B2D (the adapted model is not in B2D yet). Code: `scripts/loss_budget_*.py`.
"Recovered so far" = reference-driver ceiling minus best-driver ceiling per class (how much of that class's headroom the best driver already removed); it is a
difference of two ceilings and the classes overlap, so it does not add up to the measured total gain.

## Method per board

- **navtest** (official v1 PDMS = NC x DAC x (5 EP + 5 TTC + 2 C) / 12, n 12 146; DDC is reported only and not in the score). Per-token results of the shipped and best arms are the official CSVs; the in-process
  harness (skill_pack/scripts/offroad_lib) re-simulates the plans for the DAC features. A class's tokens take the human trajectory's official token scores
  (`v1_navtest_human`, PDMS {f(nt['reference_score'], 2)}) except that a token is never made worse than its own score (clip; the unclipped variant is in the json, difference < 0.02).
- **navhard** (two-stage EPDMS, n 5 912, the devkit's scorer and aggregation in-process; the harness reproduces 33.33 and 35.76). The official human run does not exist (human trajectory is None on the synthetic stage-2 frames), so the substitute is the
  PDM-Closed reference trajectory scored in-process (its own combined EPDMS is {f(nh['reference_score'], 2)}); clip as above; HC and EC stay each arm's own in the substitution and are
  handled by the comfort row.
- **Failing token** = arm token score below 0.8 x the reference token's score (navtest {ns['n_failing']} / 12 146 shipped, {nt['best']['n_failing']} best; navhard {nn['n_failing']} / 5 912 shipped, {nh['best']['n_failing']} best; 95% of the navhard failing tokens are stage 2).
  Classes (overlapping; one token can be in several): wrong direction = plan end on the other side of the reference end (decision 88 rule: |y| > 1 m both, opposite sign, gap > 2 m);
  road edge wide = stage-2 DAC failure where the model's own road edge at the departing corner is more than 1 m beyond the map boundary (roadedge_table.pkl of decision 88; computed for the shipped model only, the
  best driver's edge output was not rerun); early clip = DAC failure whose first departure is within 1.5 s and the start is more than 0.5 m or 0.1 rad off the route centreline (decision 88 rule, applied to both stages);
  collisions = NC < 1 or TTC < 1; EP / progress = failing and EP < 0.8, oracle sets EP to 1 on every token with EP < 0.8 (the "term if perfect" convention of q4), or raises it only to the reference token's EP (the `EP at reference` row, the more realistic one);
  stopped or too slow = failing, EP < 0.5 and the plan's 4 s end is under half of the reference's (or v0 < 1 m/s and the end under 2 m); comfort = HC / EC set to 1 on all tokens (share = failing tokens with HC or EC < 1);
  no command = failing, the reference turns (|yaw at 4 s| >= 0.3 rad), the navigation command is left or right, and the plan turns less than half as much or the other way (wrong direction is a subset).
- **HUGSIM 64** (PR #57 controller; the runs of `hugsim/results/derot`, `op_adapt_h/results/hugsim64` and `one_driver`, copied under `results/loss_budget/hugsim_inputs/`). Each scenario of the shipped (`cinque-fixed`) and the best (`it_dw3-s0` + sel3) run gets one class: spin
  (the logged spin flag, priority), else the run's end (bg / fg collision, max_steps, off_route), collisions with route completion >= 0.9 are "route-end crash". Ceilings set the class's scenarios to their best HD over our 14 Cinque-family arms (all runs listed in the json; arms
  on only 10 scenarios count where they exist) or additionally over the LTF, cv and Lebowski arms of the exam, or to 1.0. Best-observed is an oracle per scenario (the best arm is chosen after the fact).
- **B2D** (19 diagnostic routes x seeds 0-3, 76 runs per arm, `vlm_arb/results/vmerge2_runs.csv`; low-scoring dev routes, not a random sample). DS = RC x 0.5^ped x 0.6^veh x 0.65^layout x 0.7^red x 0.8^stop x lane-departure factor (recovered per run; reproduces the logged DS exactly).
  Removing a class sets its factor to 1; "blocked / timeout" lets the run complete (RC = 100) either mechanically (other factors unchanged) or exposure-aware (DS of the same route's completed runs, which carries the infractions the longer route exposes).

## navtest (PDMS; shipped {f(ns['score'], 2)}, best {f(nt['best']['score'], 2)}, human {f(nt['reference_score'], 2)})

{nav_table('navtest')}

**Reading.** The headroom on navtest is progress and collisions, not the three navhard failure classes: EP is the largest single term (raised to the human's own EP it is {f(ns['rows']['ep_at_ref']['ceiling'], 1)} points; the {f(ns['rows']['ep_progress']['ceiling'], 1)} for EP = 1 exceeds what the human trajectory itself reaches, so the "everything" row is above the human PDMS gap of {f(nt['reference_score'] - ns['score'], 1)}),
collisions {f(ns['rows']['collisions']['ceiling'], 1)} and DAC failures {f(ns['rows']['offroad_all']['ceiling'], 1)}. Wrong direction ({f(ns['rows']['wrong_direction']['ceiling'], 2)}), early clip ({f(ns['rows']['early_clip']['ceiling'], 2)}) and comfort (0) are nearly empty on navtest: these are navhard (stage-2) phenomena, not a property of the model on real frames.
Missed turns without a command cost {f(ns['rows']['no_command']['ceiling'], 2)}, and standing still or crawling {f(ns['rows']['stopped_slow']['ceiling'], 2)}. The best driver's ceilings are within ~0.3 of the shipped ones on every row, matching its +0.52 total: adaptation plus selector have not moved any class's headroom on navtest. Classes are overlapping membership sets (collision tokens are 46% of the failing tokens, DAC 43%), so rows do not add up; the joint rows do.

## navhard (two-stage EPDMS; shipped {f(nn['score'], 2)}, best {f(nh['best']['score'], 2)}, PDM reference {f(nh['reference_score'], 2)})

{nav_table('navhard')}

**Reading.** On navhard the lost points are in the drivable-area family: replacing every DAC-failing token by the reference is worth {f(nn['rows']['offroad_all']['ceiling'], 1)} points for the shipped driver ({f(nh['best']['rows']['offroad_all']['ceiling'], 1)} for the best), of which the three decision 88 classes are wrong direction {f(nn['rows']['wrong_direction']['ceiling'], 1)}, early clip {f(nn['rows']['early_clip']['ceiling'], 1)} and road edge too wide {f(nn['rows']['road_edge_wide']['ceiling'], 1)}
(the road-edge class is the largest of the three and overlaps early clip). Progress is the next block: EP to the reference's is {f(nn['rows']['ep_at_ref']['ceiling'], 1)} and the stopped / too-slow tokens {f(nn['rows']['stopped_slow']['ceiling'], 1)} (overlapping), then comfort {f(nn['rows']['comfort']['ceiling'], 1)}
(HC or EC is below 1 on {pct(nn['rows']['comfort']['share'])} of the failing tokens; the q4 table of decision 88 puts nearly all of it on EC), collisions {f(nn['rows']['collisions']['ceiling'], 1)} and the missed-turn class {f(nn['rows']['no_command']['ceiling'], 1)} (wrong direction is a subset of it, so giving the model the command is the broader lever, while the wrong-direction class alone is {f(nn['rows']['wrong_direction']['ceiling'], 1)}). All trajectory classes together are {f(nn['all_traj_only'], 1)} for the shipped driver; the sum of the single rows is larger because the classes overlap.
What the best driver already took: wrong direction {diff(nn['rows']['wrong_direction']['ceiling'], nh['best']['rows']['wrong_direction']['ceiling'], 1)} (the selector's job: 172 to 82 wrong-direction tokens) and stopped / too slow {diff(nn['rows']['stopped_slow']['ceiling'], nh['best']['rows']['stopped_slow']['ceiling'], 1)}, early clip {diff(nn['rows']['early_clip']['ceiling'], nh['best']['rows']['early_clip']['ceiling'], 1)};
collisions did not move ({f(nn['rows']['collisions']['ceiling'], 1)} to {f(nh['best']['rows']['collisions']['ceiling'], 1)}) and comfort did not move. Caveats: the substitute is the PDM reference, whose own EPDMS is {f(nh['reference_score'], 1)} (95% of the failing tokens are stage 2, where the reference starts from the displaced pose);
the road-edge class could not be evaluated for the best driver without rerunning the model's edge output.

## HUGSIM 64 (HD x 100; shipped {f(hg['cinque:fixed']['hd_mean'])}, best {f(hg['it_dw3+sel3']['hd_mean'])}; best observed over our arms {f(hg['best_obs_ours_mean'])}, over all arms {f(hg['best_obs_any_mean'])})

{hugsim_table()}

**Reading.** HUGSIM is a collision-and-standing-still board. For the shipped driver 26 of 61 failing scenarios end in a foreground collision (43%), 14 stand until max_steps (23%) and 10 spin (16%). Measured against what some arm already achieved on the same scenario, the largest single ceiling is stopped / max_steps ({f(hg['cinque:fixed']['rows']['stopped / max_steps']['ceiling_best_ours'])} points), then spin ({f(hg['cinque:fixed']['rows']['spin (heading > 45 deg off route)']['ceiling_best_ours'])}) and foreground collisions ({f(hg['cinque:fixed']['rows']['collision: foreground (fg)']['ceiling_best_ours'])});
against the pure HD = 1 bound the foreground collisions dominate ({f(hg['cinque:fixed']['rows']['collision: foreground (fg)']['ceiling_one'])}), because the arms run so far have never survived those scenarios (best-observed gain is only {f(hg['cinque:fixed']['rows']['collision: foreground (fg)']['ceiling_best_ours'])}), i.e. the foreground collisions are the largest unexplored headroom and the stopped / spin ones are the largest already-demonstrated headroom.
The best driver has already taken most of the spin and stopped headroom (spin {f(hg['cinque:fixed']['rows']['spin (heading > 45 deg off route)']['ceiling_one'])} to {f(hg['it_dw3+sel3']['rows']['spin (heading > 45 deg off route)']['ceiling_one'])}, stopped {f(hg['cinque:fixed']['rows']['stopped / max_steps']['ceiling_one'])} to {f(hg['it_dw3+sel3']['rows']['stopped / max_steps']['ceiling_one'])} on the HD = 1 scale) and its failures are now {f(100 * hg['it_dw3+sel3']['rows']['collision: foreground (fg)']['share_of_failures'], 0)}% foreground collisions.
Caveats: single runs per cell (same-day reruns reproduce 8 of 10 spin scenarios, so small class counts are within run noise); the spin class takes priority over the end type (8 of the 10 shipped spins end in a background collision, so "bg collision" excludes them); "foreground / background" are HUGSIM's own end codes; `complete (HD < 1)` is a run that reached the end with penalties.

## B2D (DS; drive {f(bd['drive']['ds'], 2)}, vmerge2 {f(bd['vmerge2']['ds'], 2)}; 19 routes x 4 seeds)

{b2d_table()}

**Reading.** For `drive` the largest ceilings are blocked / timeout ({f(bd['drive']['rows']['blocked_timeout']['ceiling'], 1)}; {f(bd['drive']['rows']['blocked_timeout']['ceiling_exposure_aware'], 1)} exposure-aware) and red lights ({f(bd['drive']['rows']['red_light']['ceiling'], 1)}), then vehicle collisions ({f(bd['drive']['rows']['collision_vehicle']['ceiling'], 1)}); stop sign, layout collisions and route deviation are each at most {f(max(bd['drive']['rows'][k]['ceiling'] for k in ('stop_sign', 'collision_layout', 'route_deviation')), 1)}.
`vmerge2` has taken most of the red-light ({f(bd['drive']['rows']['red_light']['ceiling'] - bd['vmerge2']['rows']['red_light']['ceiling'], 1)} recovered), stop-sign and blocked headroom ({f(bd['drive']['rows']['blocked_timeout']['ceiling'] - bd['vmerge2']['rows']['blocked_timeout']['ceiling'], 1)}; the paired table of vmerge2.md attributes the gain to the obstacle routes) but vehicle collisions went up from {f(bd['drive']['rows']['collision_vehicle']['ceiling'], 1)} to {f(bd['vmerge2']['rows']['collision_vehicle']['ceiling'], 1)} (collisions are the remaining lever: {f(bd['vmerge2']['rows']['collision_all']['ceiling'], 1)} points for all collision classes), and blocked / timeout is still {f(bd['vmerge2']['rows']['blocked_timeout']['ceiling'], 1)}.
**Caveat, exposure.** Removing one class can expose another: a run that now completes the route meets the lights, junctions and traffic the cut-off run never reached. The exposure-aware column replaces a blocked run by the mean of its route's completed runs and raises the blocked ceiling only slightly for `drive` ({f(bd['drive']['rows']['blocked_timeout']['ceiling'], 1)} to {f(bd['drive']['rows']['blocked_timeout']['ceiling_exposure_aware'], 1)}) but by a third for `vmerge2` ({f(bd['vmerge2']['rows']['blocked_timeout']['ceiling'], 1)} to {f(bd['vmerge2']['rows']['blocked_timeout']['ceiling_exposure_aware'], 1)}). The mechanical rows of the five multiplier classes are first-order: those runs' other infractions are held fixed, but fewer red lights mean fewer stops and later exposures to collisions are not modelled. The 19 routes were picked from earlier low scorers, so shares are specific to this set.

## Summary: levers ranked across boards (ceiling points for the reference driver; the current best driver's remaining ceiling in brackets)

{sm}

Rows are sorted by the sum over the four boards of the shipped-driver ceilings; the unit differs per board (PDMS, EPDMS, HD x 100, DS), so the sum is a rough index and the per-board cells are what to read. Row definitions: navhard / navtest "off-road" is the whole DAC-failure family (it contains wrong direction, early clip and road edge, which are therefore not listed again);
HUGSIM numbers use the best-observed-on-our-arms ceiling (the HD = 1 bound is in the HUGSIM table and is up to 9x larger for foreground collisions); B2D red light row adds stop sign; the navhard / navtest cells of "red light / stop sign / DDC + TLC" are the DDC and TLC failures, which are tiny.
EP rows use the reference-EP variant (EP = 1 gives {f(nn['rows']['ep_progress']['ceiling'], 1)} on navhard and {f(ns['rows']['ep_progress']['ceiling'], 1)} on navtest).

**Ranking in words.** (1) Standing still, crawling and being blocked has the largest sum and is the largest single lever on HUGSIM ({f(hg['cinque:fixed']['rows']['stopped / max_steps']['ceiling_best_ours'], 1)}) and B2D ({f(bd['drive']['rows']['blocked_timeout']['ceiling'], 1)}); on navhard ({f(nn['rows']['stopped_slow']['ceiling'], 1)}) it ranks behind the DAC family and EP, and it overlaps EP (stopped tokens are low-progress tokens), so the two are not additive. The current best has taken much of it on navhard and HUGSIM and about half on B2D. (2) The drivable-area family is by far the largest navhard lever ({f(nn['rows']['offroad_all']['ceiling'], 1)}), and it is a navhard-specific problem (navtest {f(ns['rows']['offroad_all']['ceiling'], 1)}, HUGSIM and B2D about 0).
(3) Collisions are about 3-7 points on every board and the one lever the best drivers did not reduce (B2D vmerge2 went up). (4) Progress (EP) is 3-10 points on the two NAVSIM boards but a different thing from HUGSIM / B2D completion. (5) Red lights are worth {f(bd['drive']['rows']['red_light']['ceiling'], 1)} on B2D drive and are already mostly taken by vmerge2. (6) Spin / wrong direction is worth 2-5 and the selector already took most of it. (7) Comfort ({f(nn['rows']['comfort']['ceiling'], 1)}) is a navhard-only lever (EC on stage 2) and nothing on navtest. (8) The missing command (desire off) is bounded by about {f(nn['rows']['no_command']['ceiling'], 1)} points on navhard and {f(ns['rows']['no_command']['ceiling'], 1)} on navtest, with the wrong-direction part already counted above.

## Files

`results/loss_budget/{{navtest,navhard,hugsim,b2d}}.json` (all numbers, per-class counts), `hugsim_classes_*.csv` (class of every scenario), `hugsim_inputs/` (copied run results), code `scripts/loss_budget_{{prep,nav,hugsim,b2d,report}}.py`
(prep and nav run on the box in the navsim2 env, ~7 min on 40 cores, CPU only; hugsim, b2d and report run on any machine with pandas). Intermediate pickles stay on the box under `runs/leaderboard_audit/loss_budget/`.
"""
    (R / "loss_budget.md").write_text(md)
    print(md[:200])


if __name__ == "__main__":
    main()
