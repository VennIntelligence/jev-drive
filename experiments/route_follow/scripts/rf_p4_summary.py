"""P4 box read: the secondary numbers of results/p4_self/README.md from read.json + summary.json + plan.json (stdlib only).
   python3 rf_p4_summary.py results/p4_self"""
import collections, json, sys

R = sys.argv[1]
rows = json.load(open(f"{R}/read.json"))["rows"]
S = json.load(open(f"{R}/summary.json"))
P = json.load(open(f"{R}/plan.json"))["scenes"]
sc_ids = sorted({r["scene"] for r in rows}, key=lambda s: (rows[[r["scene"] for r in rows].index(s)]["side"], s))
ident = {r["scene"]: r for r in rows if r["tag"] == "id"}
br = [r for r in rows if r["tag"] != "id"]
sid = lambda s: s[7:15]
kind = lambda r: r["zero"] or ("end" if not r["done"] else "done")

print("## (a) identity (the policy alone)\n")
d = [sid(s) for s in sc_ids if ident[s]["done"]]
print(f"completes {len(d)} / {len(sc_ids)}: {', '.join(d)}\n")
print("## (b) identity rollout, first failure per scene\n")
print("`end rel entry` = route station of the ego at the end / zero event minus the strict entry (m): < 0 = before the turn entry.\n")
print("| scene | side | topo | done | first failure | t (s) | end rel entry (m) | before entry | v_entry | min v inside |\n|:--|:--|:--|:--|:--|--:|--:|:--|--:|--:|")
nb = 0
cnt = collections.Counter()
for s in sc_ids:
    r = ident[s]
    pre = (not r["done"]) and r["end_rel_entry"] is not None and r["end_rel_entry"] < 0
    nb += pre
    if not r["done"]:
        cnt[(kind(r), "before entry" if pre else "in / after turn")] += 1
    print(f"| {sid(s)} | {r['side']} | {P[s]['topo']} | {'yes' if r['done'] else 'no'} | {kind(r) if not r['done'] else '-'} | "
          f"{r['t_zero'] if r['t_zero'] is not None and r['t_zero'] < 1e6 else '-'} | {r['end_rel_entry']} | {'**yes**' if pre else '-'} | {r['v_entry']} | {r['v_min_inside']} |")
print(f"\nIdentity fails before the turn entry: {nb} / {len(sc_ids)} (not a turn failure). Failing identity by kind: "
      + "; ".join(f"{k[0]} {k[1]}: {n}" for k, n in sorted(cnt.items())) + "\n")

print("## (c) completions per cell, per profile, per D (distinct rollouts; a duplicated distance is run once and counted under the largest D)\n")
cells = sorted({r["tag"].split("-", 2)[2] for r in br})
D = sorted({r["D"] for r in br}, reverse=True)
src = lambda r: r["src"]
def tab(keyf, keys, rs):
    n = {k: sum(keyf(r) == k for r in rs) for k in keys}; c = {k: sum(keyf(r) == k and r["done"] for r in rs) for k in keys}
    g = {k: sum(keyf(r) == k and r["done"] and r["gate"] for r in rs) for k in keys}
    return n, c, g
print("| cell | rollouts | done | done + shape gate |\n|:--|--:|--:|--:|")
n, c, g = tab(lambda r: r["tag"].split("-", 2)[2], cells, br)
for k in cells: print(f"| {k} | {n[k]} | {c[k]} | {g[k]} |")
print("\n| D | profile raw (done / n) | ve | ve08 | all |\n|--:|--:|--:|--:|--:|")
prof = lambda r: r["tag"].rsplit("-", 1)[1]
for dd in D:
    rs = [r for r in br if r["D"] == dd]
    print(f"| {dd} | " + " | ".join(f"{sum(r['done'] for r in rs if prof(r) == p)} / {sum(prof(r) == p for r in rs)}" for p in ("raw", "ve", "ve08")) + f" | {sum(r['done'] for r in rs)} / {len(rs)} |")
print("\n| profile | rollouts | done | done + gate |\n|:--|--:|--:|--:|")
n, c, g = tab(prof, ["raw", "ve", "ve08"], br)
for k in n: print(f"| {k} | {n[k]} | {c[k]} | {g[k]} |")
print("\n| gain | rollouts | done | done + gate |\n|:--|--:|--:|--:|")
n, c, g = tab(lambda r: r["tag"].split("-")[2], ["g1", "g1.25", "g1.5"], br)
for k in n: print(f"| {k} | {n[k]} | {c[k]} | {g[k]} |")

print("\n## (d) fallback, rear-ended, crawl, shape gate\n")
log = [r for r in br if r["src"] == "log"]
print(f"- src=log rollouts: {len(log)} ({len({(r['scene'], r['D']) for r in log})} distance(s): " + ", ".join(sorted({f'{sid(r['scene'])} D{r['D']}' for r in log})) + f"), done {sum(r['done'] for r in log)}")
print(f"- rear-ended (collision_rear): {sum(r['rear'] for r in rows)} rollouts, of which counted as done {sum(r['rear'] and r['done'] for r in rows)}")
app = [r for r in br if r["D"] != 0 or r["src"] == "log"]
print(f"- crawl (min speed h..entry < 0.5 v_e): approach {sum(bool(r['crawl']) for r in br)} / {len(br)} branch rollouts (D0 rollouts start at the entry: 0); among done {sum(bool(r['crawl']) and r['done'] for r in br)} / {sum(r['done'] for r in br)}")
print(f"- crawl inside [entry, exit]: {sum(bool(r['crawl_inside']) for r in br)} / {len(br)}; among done {sum(bool(r['crawl_inside']) and r['done'] for r in br)} / {sum(r['done'] for r in br)}")
dn = [r for r in br if r["done"]]
print(f"- completing branch cells: {len(dn)}; pass the full shape gate: {sum(r['gate'] for r in dn)}; gate parts failing among the {len(dn)}: "
      + ", ".join(f"{p} {sum(not r['gate_parts'][p] for r in dn)}" for p in ("kmax", "dk", "alat", "v_entry")))
print(f"- completing cells with realized entry speed <= class v_e + 1: {sum(r['gate_parts']['v_entry'] for r in dn)}")
print(f"- determinism before h (pre_h_diff > 0): {sum((r['pre_h_diff'] or 0) > 0 for r in br if r['src'] == 'self')} of {sum(r['src']=='self' for r in br)} self rollouts")

print("\n## (e) distinct handovers per scene\n")
nd = collections.Counter(len(S["cov"][s]) for s in sc_ids)
print("distinct handover decisions per scene: " + ", ".join(f"{k}: {v} scenes" for k, v in sorted(nd.items())) + f"; total {sum(len(S['cov'][s]) for s in sc_ids)}")
print(f"scenes with one or two distinct handovers: {sum(len(S['cov'][s]) <= 2 for s in sc_ids)} (a scene with one cannot reach the >= 2 line)")
print("| scene | distinct handovers | h per D |\n|:--|--:|:--|")
for s in sc_ids:
    print(f"| {sid(s)} | {len(S['cov'][s])} | " + ", ".join(f"D{k}:{v['h']}" + ("(log)" if v["src"] == "log" else "") for k, v in S["cov"][s].items()) + " |")
print("\n### scenes counted by the line (>= 2 distinct handovers with a completing, entry-speed-ok cell)\n")
ok = [sid(s) for s in sc_ids if sum(bool(v["done_slow"]) for v in S["cov"][s].values()) >= 2]
print(f"{len(ok)}: {', '.join(ok)}")
