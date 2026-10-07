# Human-readable reports with agy

Every page meant for people (research synthesis, diagnosis, round report) is one HTML page written by agy
(Google Antigravity CLI), not by a Claude subagent. Agent-facing records stay Markdown: `research/decisions.md`,
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
1. Main session writes a brief to `tmp/<topic>-brief.md`: goal, audience, source files, page structure,
   constraints (above), done criteria (README link; commit only the page, its figures, the figure script; push;
   print a 5-line summary and stop). Numbers only from source files; missing or contradicting sources are stated.
2. Each task gets a fresh agy in its own tmux window, so it starts with a clean context (never append a new task to an
   old agy session): `tmux new-window -d -t jev -n agy-<topic> -c <repo>`, then send
   `agy --model gemini-3.8-flash-high --dangerously-skip-permissions -i "Read tmp/<topic>-brief.md and do it."`.
   Independent pages run in parallel, one window each, as a pipeline: check each page as it lands. Parallel windows
   commit only their own paths (`git commit -- <paths>`, `git pull --rebase` before push) and leave shared files
   (indexes, shared figures) to one final pass. Corrections for the same page go to that window. Keep the current model and config ("Out of credits" in the
   footer is not a blocker). Close the window when the page is accepted, or leave it.
3. Main checks every number and claim on the page against the sources and sends corrections to the same pane.
4. Delete the brief once the page is committed. tmp/ never keeps reports.
