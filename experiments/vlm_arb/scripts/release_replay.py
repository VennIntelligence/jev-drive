"""Offline replay of red-light release policies on the logged Qwen3-VL-4B answers of the vred / vred2 / vred3 runs (numpy, Python 3.9).

Input : tmp/vlm_arb_offline/tmp_release_ex/*.json.gz (release_extract.py on the box; every arm, used for the light timelines).
Output: tmp/vlm_arb_offline/release_replay.pkl (read by release_replay_report.py).
Definitions: plans/2026-10-02-offline-analyses-definitions.md, section A. A replay on logged answers cannot show what the car
would have done after a different release decision (later frames would differ).
"""
import os
import pickle
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import route_common as rc  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
EX = os.path.join(ROOT, "tmp/vlm_arb_offline/tmp_release_ex")
REAR = rc.REAR
MERGE_GAP = 6.0                 # s: R2 segments closer than this belong to one episode
TAIL = 3.0                      # s of answers kept after a logged green release
REFRACT = 3.0                   # s between counted false-release events of one policy
ORIGINAL = ("vred", "vred2", "vred3")
POS = ["A before stop line", "B stop line..entrance", "C past entrance"]


def pos_class(d_stop, d_ent):
    if d_ent is not None and d_ent <= 0:
        return 2
    if d_stop is not None and d_stop <= 0:
        return 1
    return 0


def segments(s):
    """Runs of consecutive arbitration steps in which R2 was active: [(t_start, t_end, idx_end)]."""
    seg, cur = [], None
    for i, r in enumerate(s):
        on = "R2" in r[4].split("+")
        if on and cur is None:
            cur = [r[0], r[0], i]
        elif on:
            cur[1], cur[2] = r[0], i
        elif cur is not None:
            seg.append(tuple(cur))
            cur = None
    if cur is not None:
        seg.append(tuple(cur))
    return seg


def episodes_of(d):
    s, ans = d["s"], [a for a in d["a"] if a[2]]
    eps, cur = [], None
    for sg in segments(s):
        if cur is not None and sg[0] - cur["segs"][-1][1] <= MERGE_GAP:
            cur["segs"].append(sg)
        else:
            if cur is not None:
                eps.append(cur)
            cur = {"segs": [sg]}
    if cur is not None:
        eps.append(cur)
    out = []
    for e in eps:
        t0, t1 = e["segs"][0][0], e["segs"][-1][1]
        causes = []                                  # (cause, time the segment ended, t_eff of the second green)
        for sg in e["segs"]:
            j = sg[2] + 1
            nxt = s[j] if j < len(s) else None
            if nxt is None:
                causes.append(("run_end", sg[1], None))
            elif nxt[6] or "R5" in nxt[4].split("+"):
                causes.append(("r5", nxt[0], None))
            else:
                got = [a for a in ans if a[1] <= nxt[0] + 1e-6][-2:]
                if len(got) == 2 and all(a[3] == "green_for_ego" for a in got):
                    causes.append(("green", nxt[0], got[-1][1]))
                else:
                    causes.append(("other", nxt[0], None))
        last = causes[-1]
        t_end = last[1] + (TAIL if last[0] == "green" else 1.0)
        jids = Counter(r[8] for r in s if t0 - 1e-6 <= r[0] <= t1 + 1e-6 and r[8] is not None)
        out.append(dict(t0=t0, t1=t1, t_end=t_end, causes=causes, jid=jids.most_common(1)[0][0] if jids else None))
    return out


def answer_dicts(d, info):
    """Every ok answer with the car's position (s-step at the query time) and the truth of the governing light of the next junction."""
    ts, ss = rc.ego_s_at(d)
    sig = sorted((info["entries"][j], rc.governing_light(info, j), j) for j in info["entries"] if rc.governing_light(info, j) is not None)
    out = []
    for a in d["a"]:
        if not a[2]:
            continue
        i = int(np.abs(ts - a[0]).argmin())
        es = float(ss[i]) if abs(ts[i] - a[0]) < 0.06 else None
        rec = dict(tq=a[0], te=a[1], lab=a[3], pg=a[4] or 0.0, pr=a[5] or 0.0, po=a[6] or 0.0, pn=a[7] or 0.0, lat=a[8], es=es,
                   truth=None, pos=None, d_stop=None, d_ent=None, jid=None)
        if es is not None:
            for E, L, j in sig:                                   # nearest signalised junction not yet cleared by 6 m
                if E - es - REAR >= -6.0:
                    rec.update(d_ent=E - es - REAR, d_stop=L["arc"] - es - REAR, jid=j, truth=rc.light_state(L, d['attempt'], a[0]))
                    rec["pos"] = pos_class(rec["d_stop"], rec["d_ent"])
                    break
        out.append(rec)
    return out


# ------------------------------------------------------------------ policies
def llr(pg, pr, eps=1e-3, cap=7.0):
    return float(np.clip(np.log((pg + eps) / (pr + eps)), -cap, cap))


class Pol:
    def __init__(self, name, kind, **kw):
        self.name, self.kind, self.kw = name, kind, kw
        self.reset()

    def reset(self):
        self.n, self.S, self.prev_red, self.run = 0, 0.0, False, 0

    def step(self, a):
        k, kw = self.kind, self.kw
        pg, pr = a["pg"], a["pr"]
        if k == "K":
            self.n = self.n + 1 if a["lab"] == "green_for_ego" else 0
            return self.n >= kw["K"]
        if k == "thr":
            return pg >= kw["p"]
        if k == "Kthr":
            self.n = self.n + 1 if (a["lab"] == "green_for_ego" and pg >= kw["p"]) else 0
            return self.n >= kw["K"]
        if k == "cusum":
            self.S = 0.0 if pr >= 0.9 else max(0.0, self.S + llr(pg, pr))
            return self.S >= kw["theta"]
        if k == "trans":                         # a confident red answer directly before a run of K confident greens
            if pg >= kw["p"]:
                self.run = self.run + 1 if (self.run > 0 or self.prev_red) else 0
            else:
                self.run = 0
            self.prev_red = pr >= kw.get("red", 0.9)
            return self.run >= kw["K"]
        if k == "gate":                          # K=2 before the stop line, K=4 with p>=0.99 once the front bumper is past it
            past = a.get("pos") in (1, 2)
            self.n = self.n + 1 if (a["lab"] == "green_for_ego" and (pg >= 0.99 or not past)) else 0
            return self.n >= (4 if past else 2)
        raise ValueError(k)


def policies():
    P = [Pol("K=1", "K", K=1), Pol("K=2 (current)", "K", K=2), Pol("K=3", "K", K=3), Pol("K=4", "K", K=4)]
    P += [Pol("single p>=%g" % p, "thr", p=p) for p in (0.5, 0.8, 0.9, 0.95, 0.99)]
    P += [Pol("K=2 and p>=%g" % p, "Kthr", K=2, p=p) for p in (0.9, 0.99)]
    P += [Pol("cusum >= %g" % th, "cusum", theta=th) for th in (3, 5, 7, 10)]
    P += [Pol("transition: red p>=0.9, then 1 green p>=0.9", "trans", K=1, p=0.9, red=0.9),
          Pol("transition: red p>=0.9, then 2 greens p>=0.9", "trans", K=2, p=0.9, red=0.9),
          Pol("transition: red p>=0.7, then 2 greens p>=0.9", "trans", K=2, p=0.9, red=0.7),
          Pol("transition: red p>=0.5, then 1 green p>=0.9", "trans", K=1, p=0.9, red=0.5),
          Pol("position-gated: K=2 before the line, K=4 and p>=0.99 past it", "gate")]
    return P


def replay(pol, answers, t0):
    """Release events [(index, t_eff)] with re-arm after each release and a refractory period; none before the hold starts."""
    pol.reset()
    ev, last = [], -1e9
    for i, a in enumerate(answers):
        if pol.step(a):
            pol.reset()
            if a["te"] >= t0 - 1e-6 and a["te"] - last >= REFRACT:
                ev.append((i, a["te"]))
                last = a["te"]
    return ev


def delay_after_green(pol, answers, t_g, t_stop):
    """Fresh state at the green onset: seconds from t_g to the release; None if not released before t_stop."""
    pol.reset()
    for a in answers:
        if a["te"] < t_g:
            continue
        if a["te"] > t_stop:
            break
        if pol.step(a):
            return a["te"] - t_g
    return None


def build():
    docs = rc.load_docs(EX)
    by_route = defaultdict(list)
    for d in docs:
        by_route[d["route"]].append(d)
    infos = {r: rc.route_info(v) for r, v in by_route.items()}
    eps_out, pol_rows, hold_ans, all_ans, ev_rows, held_nolight = [], [], [], [], [], []
    pols = policies()
    for d in docs:
        base = d["arm"].replace("gif_", "").split("_gif")[0]
        is_vred = d["arm"] in ORIGINAL or d["arm"].startswith("gif_vred")
        if not is_vred:
            continue
        rerun = d["arm"].startswith("gif_")
        info = infos[d["route"]]
        ans = answer_dicts(d, info)
        if not rerun:
            for a in ans:
                if a["truth"] is not None and -50 < (a["d_stop"] or 1e9) < 50:
                    all_ans.append(dict(arm=d["arm"], route=d["route"], **{k: a[k] for k in ("pos", "truth", "pg", "pr", "lab")}))
        for e in episodes_of(d):
            L = rc.governing_light(info, e["jid"]) if e["jid"] is not None else None
            row = dict(arm=d["arm"], base=base, rerun=int(rerun), seed=d["seed"], route=d["route"], unit=d["unit"], jid=e["jid"], t0=round(e["t0"], 2),
                       t1=round(e["t1"], 2), t_end=round(e["t_end"], 2), end_causes="|".join(c[0] for c in e["causes"]), last_cause=e["causes"][-1][0])
            if L is None:
                row.update(no_light=1)
                eps_out.append(row)
                continue
            row["no_light"] = 0
            window = [a for a in ans if e["t0"] - 3.0 <= a["te"] <= e["t_end"]]
            row["n_ans"] = len([a for a in window if a["te"] >= e["t0"]])
            row["truth_start"] = rc.light_state(L, d['attempt'], e["t0"])
            onsets = rc.green_onsets(L, d['attempt'], e["t0"] - 0.2, e["t_end"])
            row["t_g"] = round(onsets[0], 2) if onsets else None
            row["green_len"] = round(rc.leave_green(L, d['attempt'], onsets[0], onsets[0] + 999.0) - onsets[0], 2) if onsets else None
            logged = [rc.light_state(L, d['attempt'], c[2]) for c in e["causes"] if c[0] == "green"]
            row.update(logged_releases=len(logged), logged_false=sum(1 for x in logged if x is not None and x != 0),
                       logged_release_times=[round(c[2], 2) for c in e["causes"] if c[0] == "green"])
            eps_out.append(row)
            if rerun:
                continue
            for a in window:
                if a["te"] >= e["t0"] - 1e-6 and a["truth"] is not None:
                    hold_ans.append(dict(arm=d["arm"], route=d["route"], **{k: a[k] for k in ("pos", "truth", "pg", "pr", "lab")}))
            tg = onsets[0] if onsets else None
            if tg is not None:
                t_stop = min(rc.leave_green(L, d['attempt'], tg, e["t_end"]), e["t_end"])
                near = [a for a in window if a["pos"] is not None and abs(a["te"] - tg) < 1.0]
                pos_g = near[0]["pos"] if near else None
            for p in pols:
                evs = replay(p, window, e["t0"])
                nfalse, first = 0, None
                for (i, te) in evs:
                    st = rc.light_state(L, d['attempt'], te)
                    pc = window[i]["pos"]
                    if first is None:
                        first = (te, st, pc)
                    if st is not None and st != 0:
                        nfalse += 1
                        ev_rows.append((p.name, d["arm"], d["route"], d["seed"], pc, te, st))
                rec = dict(policy=p.name, arm=d["arm"], route=d["route"], seed=d["seed"], t0=e["t0"], n_events=len(evs), n_false_events=nfalse,
                           first_false=int(first is not None and first[1] is not None and first[1] != 0), first_pos=None if first is None else first[2],
                           first_te=None if first is None else first[0], first_truth=None if first is None else first[1])
                if tg is not None:
                    rec.update(has_green=1, delay=delay_after_green(p, window, tg, t_stop), pos_g=pos_g, green_len=t_stop - tg, tg=tg)
                else:
                    rec.update(has_green=0, delay=None, pos_g=None, green_len=None, tg=None)
                pol_rows.append(rec)
    return docs, infos, eps_out, pol_rows, hold_ans, all_ans, ev_rows


if __name__ == "__main__":
    docs, infos, eps, pol_rows, hold_ans, all_ans, ev_rows = build()
    pickle.dump(dict(eps=eps, pol_rows=pol_rows, hold_ans=hold_ans, all_ans=all_ans, ev_rows=ev_rows,
                     entries={r: i["entries"] for r, i in infos.items()},
                     lights={r: [dict(arc=L["arc"], runs=len(L["tl"])) for L in i["lights"]] for r, i in infos.items()}),
                open(os.path.join(ROOT, "tmp/vlm_arb_offline/release_replay.pkl"), "wb"))
    print(len(eps), "episodes", len(pol_rows), "policy-episode rows", len(hold_ans), "hold answers", len(all_ans), "all answers")
