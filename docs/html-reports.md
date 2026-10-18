# Human-readable reports

Every page meant for people (research synthesis, diagnosis, round report) is one HTML page written by Claude: the
agent that owns the results writes the page for its lane; the main session writes cross-lane pages and small updates.
agy (Google Antigravity) is no longer used (2026-10-09). Agent-facing records stay Markdown: `research/decisions.md`,
`research/decisions/`, topic READMEs, `research/lit/`.

## Layout
- `research/<topic>/index.html` + `research/<topic>/figs/`; a link line in `research/README.md`.
- Pure Chinese text (headings, tables, captions); code, figure scripts, commits English.
- Self-contained: inline CSS, relative image paths, readable in the repo web reader ([web-reader.md](web-reader.md)).
  House style: `research/four-directions/index.html`, `research/turn-gain/index.html`.
- Figures: matplotlib via `research/plot_style.py`; every figure and GIF has a caption saying what to look at.
- Keep it short: key conclusions, the numbers behind them with their source file, pointers to raw data on the box.
  Drop detail nobody will use; drop figures that do not carry a conclusion.

## Procedure
1. The writer works from the source files (results tables, decisions, per-scene CSVs), never from memory of a chat.
   Numbers only from source files; missing or contradicting sources are stated on the page.
2. A lane's agent writes its own page as the last step of the lane, after the agent-facing record is committed.
   A page that spans lanes is written by the main session, or by one dispatched agent given the source file list.
3. Whoever did not write the page checks every number and claim on it against the sources (main for a lane's page,
   a dispatched reader for a main-written page) before it is called done.
4. Commit only the page, its figures and the figure script, plus the README link; push. tmp/ never keeps reports.

## Video pages
Any page with rollout or scene videos (galleries, diagnosis indexes):
- **Speed bar on top.** A sticky bar at the very top: buttons 自动 / 1× / 2× / 3× / 5× / 10× / 16× plus a 0.5–16× slider.
  自动 is each clip's own rate; any other choice overrides every clip at once, updates each clip's rate tag, and is
  kept in `localStorage.vidRate` so it carries across pages. 16× is the browser ceiling. The web reader serves pages
  in a CSP sandbox (opaque origin) where touching `localStorage` throws: wrap every access in try / catch, or the
  whole script dies and no clip loads. Test a page through `/read/_preview/...` before calling it done.
- **Auto rate.** `max(2, min(3, 30/fps, dur/6))`, shown as a tag on each clip; 2 fps clips are choppy below 2×.
- **Lazy loading.** No `autoplay`, no `src` up front: `data-src` + `preload="none"`, one IntersectionObserver fetches
  a clip within about one screen, a second plays it only while >= 50 % visible and pauses it when it leaves.
- Videos stay gitignored next to the page; commit the page and a manifest.
- Reference implementation: `experiments/defect_fix/scripts/jtn_diag.py` (`speed()`, the bar and the observer script).
