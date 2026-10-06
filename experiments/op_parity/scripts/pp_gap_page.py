"""op_parity gap page: gap_tables.json + cases.json + clips/ (+ the HUGSIM fragment) -> results/gap/index.html (self-contained, relative paths, opens from disk).

  python experiments/op_parity/scripts/pp_gap_page.py --gap DIR --out experiments/op_parity/results/gap [--hugsim FRAGMENT.html]
  (DIR holds gap_tables.json, cases.json and clips/*.gif; the GIFs are copied next to the page. stdlib only.)
"""
import argparse
import html
import json
import shutil
from pathlib import Path

TERMS = ["NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC", "EC"]
LONG = {"NC": "no at-fault collisions", "DAC": "drivable area compliance", "DDC": "driving direction compliance", "TLC": "traffic light compliance",
        "EP": "ego progress", "TTC": "time to collision", "LK": "lane keeping", "HC": "history comfort", "EC": "extended comfort (two-frame)"}
CSS = """
:root{--bg:#fff;--fg:#1b1f23;--mut:#5b6570;--line:#d9dee3;--soft:#f4f6f8;--red:#c62828;--blue:#1f5fbf;--bar:#e8a0a0;--bar2:#a9c4ee}
@media (prefers-color-scheme:dark){:root{--bg:#14171a;--fg:#e4e7ea;--mut:#9aa4ae;--line:#2c3238;--soft:#1d2226;--red:#ef7373;--blue:#7aa7ec;--bar:#7a3b3b;--bar2:#3a5580}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:0 auto;padding:16px 16px 60px;max-width:1080px}
h1{font-size:24px;margin:.4em 0 .2em}h2{font-size:19px;margin:1.8em 0 .4em;border-bottom:1px solid var(--line);padding-bottom:4px}h3{font-size:16px;margin:1.3em 0 .3em}
p,li{max-width:80ch}.mut{color:var(--mut)}.small{font-size:13px}
table{border-collapse:collapse;margin:.6em 0 1em;font-size:13.5px;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--line);padding:4px 9px;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
th{background:var(--soft);font-weight:600}.wrap{overflow-x:auto}tr.tot td{font-weight:600;border-top:2px solid var(--line)}
.neg{color:var(--red)}.bar{display:inline-block;height:9px;background:var(--bar);vertical-align:middle;margin-right:5px;border-radius:2px}.bar.b0{background:var(--bar2)}
figure{margin:.8em 0 1.4em}figure img{max-width:100%;height:auto;border:1px solid var(--line)}figcaption{font-size:13px;color:var(--mut);max-width:80ch}
.cases{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:10px 18px}
code{background:var(--soft);padding:1px 4px;border-radius:3px;font-size:13px}details{margin:.6em 0}summary{cursor:pointer;color:var(--blue)}
.box{background:var(--soft);padding:8px 14px;border-radius:6px}
"""


def f(x, d=2):
    return f"{x:+.{d}f}".replace("-", "−")


def ci(m, lo, hi, d=2):
    return f"{f(m, d)} [{f(lo, d)}, {f(hi, d)}]"


def gap_table(p0, p2, caption):
    """p0, p2: {term: stats, _score: ...} for the same token set; sorted by P2's loss."""
    order = sorted(TERMS, key=lambda t: -p2[t]["loss"])
    mx = max(max(abs(p2[t]["loss"]), abs(p0[t]["loss"])) for t in TERMS) or 1
    rows = []
    for t in order:
        a, b = p2[t], p0[t]
        sig = "" if a["loss_lo"] <= 0 <= a["loss_hi"] else " *"
        rows.append(
            f"<tr><td><b>{t}</b> <span class=mut>{LONG[t]}</span></td><td>{b['arm']:.1f}</td><td>{a['arm']:.1f}</td><td>{a['wa']:.1f}</td>"
            f"<td class={'neg' if a['diff_hi'] < 0 else ''}>{ci(a['diff'], a['diff_lo'], a['diff_hi'])}</td>"
            f"<td><span class=bar style='width:{max(0, a['loss']) / mx * 70:.0f}px'></span>{f(a['loss'])}{sig}</td>"
            f"<td class=mut>{f(a['loss_lo'])}, {f(a['loss_hi'])}</td>"
            f"<td><span class='bar b0' style='width:{max(0, b['loss']) / mx * 70:.0f}px'></span>{f(b['loss'])}</td>"
            f"<td>{a['arm_fail_rate']:.1f} / {a['wa_fail_rate']:.1f}</td><td>{a['arm_fail_wa_pass']} / {a['wa_fail_arm_pass']}</td></tr>")
    s2, s0 = p2["_score"], p0["_score"]
    rows.append(f"<tr class=tot><td>EPDMS (sum of the 9)</td><td>{s0['arm']:.2f}</td><td>{s2['arm']:.2f}</td><td>{s2['wa']:.2f}</td>"
                f"<td>{ci(-s2['gap'], -s2['gap_hi'], -s2['gap_lo'])}</td><td>{f(s2['shap_sum'])}</td><td class=mut>gap {f(s2['gap'])}</td><td>{f(s0['shap_sum'])}</td><td></td><td></td></tr>")
    return (f"<div class=wrap><table><caption class='mut small' style='text-align:left;caption-side:top'>{caption}</caption><thead><tr><th>sub-metric</th>"
            "<th>P0</th><th>P2</th><th>WA-JEPA</th><th>P2 − WA [95% CI]</th><th>P2 loss (EPDMS pts)</th><th>CI</th><th>P0 loss</th>"
            "<th>fail % P2 / WA</th><th>P2 (any seed) fails, WA passes / reverse</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hugsim")
    a = ap.parse_args()
    gap, out = Path(a.gap), Path(a.out)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    T = json.loads((gap / "gap_tables.json").read_text())
    C = json.loads((gap / "cases.json").read_text())
    nt = T["navtest"]
    p2, p0 = nt["arms"]["P2"], nt["arms"]["P0"]
    order = sorted(TERMS, key=lambda t: -p2[t]["loss"])
    top = ", ".join(f"<b>{t}</b> {f(p2[t]['loss'])}" for t in order[:4])
    h = [f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>op_parity gap</title><style>{CSS}</style></head><body>"]
    h.append("<h1>Where op_parity P2 still loses to WA-JEPA</h1>")
    h.append("<p class=mut>P2 = Cinque + ego / pose history / command inputs, fine-tuned on navtrain (seed mean of 2), protocol W; P0 = shipped Cinque, same frames; "
             "WA-JEPA = released checkpoint, same devkit (navsim main @ 0a380a9, v2 EPDMS). Source: <code>experiments/op_parity/results/{full,navhard,hugsim_full,hugsim_spec}.md</code>; "
             "generated by <code>experiments/op_parity/scripts/pp_gap_*.py</code> from stored per-token scores, no model reruns.</p>")
    h.append(f"<div class=box><b>Summary.</b> navtest: P2 {p2['_score']['arm']:.2f} vs WA-JEPA {p2['_score']['wa']:.2f} (gap {p2['_score']['gap']:.2f}); "
             f"P2's gap by sub-metric, EPDMS points: {top}. P0's gap is {p0['_score']['gap']:.2f}. Details, navhard and HUGSIM below.</div>")
    h.append("<h2>Attribution rule</h2><p class=small>The v2 EPDMS of a token is <code>NC&middot;DAC&middot;DDC&middot;TLC &middot; (5 EP + 5 TTC + 2 LK + 2 HC + 2 EC) / (14 + 2&middot;[EC present])</code> "
             "(reproduces the devkit's per-token score to 1e-15). For the gap WA-JEPA &minus; arm, each sub-score is a player and v(S) is the token score when the terms in S take "
             "WA-JEPA's values and the rest the arm's own. The exact Shapley value of a term (average over all 512 coalitions) splits each token's gap additively, so the per-term losses "
             "add up to the mean-score gap exactly (last row). A positive loss means the term costs the arm EPDMS points against WA-JEPA; a term can be negative when the arm is better there. "
             "Because DAC / NC / DDC / TLC multiply the whole score, their loss includes the progress and comfort terms those failures zero out. CI: cluster bootstrap over logs (navtest) or "
             "scene-mapping groups (navhard), B 10 000, paired; * = loss CI excludes 0. Sub-metric means are x100; the seed mean of P2 is the per-token mean of two seeds; Shapley is computed per seed, then averaged.</p>")
    h.append(f"<h2>1. navtest EPDMS v2 ({nt['n']} tokens, {nt['units']} logs)</h2>")
    h.append(gap_table(p0, p2, "Sub-score means (x100), P2 − WA-JEPA per sub-metric and each sub-metric's Shapley loss, sorted by P2's loss."))
    h.append("<p class=small><b>What to look at.</b> DAC is about 60% of P2's remaining gap (+2.19 of 3.50 points, 4.7% vs 1.8% of tokens failing): off-road plans, and through the multiplier "
             "the progress and comfort those tokens lose. NC (+0.61) is next; EP, TTC, DDC, TLC and LK are each 0.1-0.2 points. EC and HC cost nothing (P2 is at or above WA-JEPA). "
             "P0's gap (11.2) is spread over DAC 4.2, NC 1.9, EC 1.7, EP 1.6: the ego inputs closed EC and EP almost fully but only half of DAC / NC. "
             "The last column counts tokens where P2 (either seed) fails a sub-metric that WA-JEPA passes against the reverse; for DAC it is 460 vs 110.</p>")
    for key, title in (("navhard_G", "navhard two-stage, protocol G (GIMM frames, the fair harness for P0 and P2)"), ("navhard_W", "navhard two-stage, protocol W (warped frames)")):
        N = T[key]
        inner = ""
        for st in ("stage1", "stage2"):
            s2_, s0_ = N["arms"]["P2"][st], N["arms"]["P0"][st]
            n = N[f"n_{st}"]
            cap = (f"Stage {st[-1]}: {n} tokens" + (" (unweighted token means; the official stage 2 weights tokens by the arm's own stage-1 outcome: "
                   f"P0 {s0_['_official']:.2f}, P2 {s2_['_official']:.2f}, WA-JEPA {s2_['_official_wa']:.2f})" if st == "stage2" else
                   f" (official stage 1: P0 {s0_['_official']:.2f}, P2 {s2_['_official']:.2f}, WA-JEPA {s2_['_official_wa']:.2f})"))
            inner += gap_table(s0_, s2_, cap)
        if key == "navhard_G":
            h.append(f"<h2>2. {title}</h2>{inner}")
            h.append("<p class=small><b>What to look at.</b> Protocol G. Stage 1 (real scenes, gap 8.8): DAC +4.0 (wide CI, 450 tokens), EC +2.3 (P2 fails the two-frame comfort on 44% of tokens vs 20%), "
                     "DDC +1.3, EP +1.1. Stage 2 (rendered follow-ups, unweighted, gap 3.3): DAC +1.9, EP +1.9 (P2 is slower: its progress is 74.5 vs 82.6), NC +0.6; EC and DDC favour P2. "
                     "So DAC leads on both stages; what differs is that stage 1 adds comfort (EC) and stage 2 adds progress (EP). The official combined score does not decompose additively "
                     "(stage 2 is weighted by the arm's stage-1 outcome), so the per-stage tables are the attribution; the combined gaps are in <code>results/navhard.md</code>.</p>")
        else:
            h.append(f"<details><summary>{title}</summary>{inner}</details>")
    h.append("<h2>3. HUGSIM 64</h2>")
    if a.hugsim and Path(a.hugsim).exists():
        h.append(Path(a.hugsim).read_text())
        src = Path(a.hugsim).parent / "clips"
        if src.exists():
            for g in src.glob("*.gif"):
                shutil.copy(g, out / "clips" / g.name)
    else:
        h.append("<p class=mut>HUGSIM section not generated.</p>")
    h.append("<h2>4. navtest failure cases</h2>")
    h.append(f"<p class=small><b>Selection rule.</b> {html.escape(C['selection'])} Every GIF: left, bird's-eye view in the t0 ego frame (map areas grey, route lanes blue, logged agents "
             "from the metric cache, each plan with its ego box moving in real time, P2 = seed 0); right, the road and wide frames P2 actually received (protocol W: 4 keyframes + 6 warped "
             "lattice frames, the 8 context slots t = &minus;1.4 &hellip; 0 s) played first, then the last input frame held while the future runs. Plans are open-loop (not tracked by the simulator), "
             "so a box leaving the grey area is the plan, not the PDM-simulated ego.</p>")
    byM = {}
    for c in C["cases"]:
        byM.setdefault(c["metric"], []).append(c)
    for m in C["order"]:
        cs = byM.get(m, [])
        if not cs:
            continue
        a_ = p2[m]
        h.append(f"<h3>{m}: {LONG[m]} (P2 loss {f(a_['loss'])}, {cs[0]['n_candidates']} candidate tokens)</h3><div class=cases>")
        for c in cs:
            shutil.copy(gap / "clips" / f"{c['id']}.gif", out / "clips" / f"{c['id']}.gif")
            j = TERMS.index(m)
            s = c["sub"]
            h.append(f"<figure><img src='clips/{c['id']}.gif' alt='{c['id']}' loading=lazy><figcaption><code>{c['token']}</code> ({c['cmd']}, {c['speed']:.1f} m/s). {m}: "
                     f"P2 {s['P2s0'][j]:.2f} / {s['P2s1'][j]:.2f} (seeds), P0 {s['P0'][j]:.2f}, WA-JEPA {s['WA'][j]:.2f}. Shapley share of the token gap {c['shapley_share']:.2f}.</figcaption></figure>")
        h.append("</div>")
    h.append("<h2>Caveats</h2><ul class=small><li>One fine-tuning recipe and two seeds; WA-JEPA is a single run through its own sampler (noise across hosts: request path differs from its stored navtest export by &minus;0.45 on 1 021 tokens).</li>"
             "<li>Shapley shares depend on the formula's structure (multiplicative gates); a different attribution would shift mass between DAC / NC and the weighted terms but the ordering of what fails is read from the fail-rate and transition columns.</li>"
             "<li>navhard stage-2 tables use unweighted token means; EC is absent for some tokens (dropped from the denominator as the devkit does).</li></ul>")
    h.append("</body></html>")
    (out / "index.html").write_text("\n".join(h))
    print("wrote", out / "index.html")


if __name__ == "__main__":
    main()
